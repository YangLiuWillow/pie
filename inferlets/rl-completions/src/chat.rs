//! `POST /v1/chat/completions` — turn-0 / passthrough (§3.2).
//!
//! In the gateway's cumulative-token mode this endpoint is hit exactly once
//! per session, for turn 0: `messages` is `system?` + `user` (+ optional
//! `tools`), never assistant/tool history — that only appears on the
//! cumulative `/v1/completions` path (§3.1). So this renders the turn-0
//! prompt through the model's own chat template (the tokens the gateway's
//! renderer bridge will extend on turn 1 — Pie is the template authority
//! here, Risk R1), generates, parses tool calls server-side (the job vLLM's
//! `--tool-call-parser` does), and returns the chat shape.
//!
//! The response MUST still carry root `prompt_token_ids` and
//! `choices[0].token_ids` (§3.2) — the gateway seeds its cumulative
//! accumulator from turn 0's exact ids, so the append-only invariant that
//! makes turns 1+ mergeable depends on this echo.
//!
//! History replay for assistant/tool turns is deliberately NOT here: the
//! spec gates it behind the renderer-parity harness (Risk R1), and the
//! cumulative path never needs it.

use crate::completions::{error_response, strip_trailing_stop};
use inferlet::model::Model;
use inferlet::sample::Sampler;
use inferlet::{chat, runtime, tools, Context, GrammarConstraint};
use serde::Deserialize;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};

const DEFAULT_MAX_TOKENS: usize = 4096;
const PREFILL_CHUNK: usize = 256;
const WEIGHT_VERSION: u64 = 0;

#[derive(Deserialize)]
struct ChatBody {
    messages: Vec<Msg>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    model: Option<String>,
    // stream accepted but turn-0 is short; non-streaming only for now.
    #[serde(default)]
    #[allow(dead_code)]
    stream: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    logprobs: Option<serde_json::Value>,
}


#[derive(Deserialize)]
struct Msg {
    role: String,
    #[serde(default)]
    content: Option<String>,
}

fn uniq() -> String {
    runtime::instance_id()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(12)
        .collect()
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The gateway sends OpenAI tool schemas as `{type:"function", function:{...}}`.
/// The SDK's tool equip expects the inner function object as a JSON string.
fn tool_schema_strings(tools: &Option<Vec<serde_json::Value>>) -> Vec<String> {
    tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"))
        .filter_map(|t| t.get("function").map(|f| f.to_string()))
        .collect()
}

pub async fn handle(body_bytes: Vec<u8>, responder: Responder) -> Finished {
    let body: ChatBody = match serde_json::from_slice(&body_bytes) {
        Ok(b) => b,
        Err(e) => return error_response(responder, 400, &format!("Invalid JSON: {e}")).await,
    };

    let models = runtime::models();
    let served = match models.first() {
        Some(n) => n.clone(),
        None => return error_response(responder, 500, "no models available").await,
    };
    let model = match Model::load(&served) {
        Ok(m) => m,
        Err(e) => return error_response(responder, 500, &e.to_string()).await,
    };
    let model_name = body.model.clone().unwrap_or(served);

    // ── Render the turn-0 prompt through the model's chat template ──
    let tool_schemas = tool_schema_strings(&body.tools);
    let mut prompt: Vec<u32> = Vec::new();
    let system_text: Option<String> = body
        .messages
        .iter()
        .find(|m| m.role == "system")
        .and_then(|m| m.content.clone());

    if !tool_schemas.is_empty() {
        // The template folds the system message + tool schemas into one
        // system turn.
        if let Some(s) = &system_text {
            prompt.extend(chat::system(&model, s));
        }
        match tools::equip_prefix(&model, &tool_schemas) {
            Ok(t) => prompt.extend(t),
            Err(e) => return error_response(responder, 500, &format!("equip: {e}")).await,
        }
    } else if let Some(s) = &system_text {
        prompt.extend(chat::system(&model, s));
    }
    for m in &body.messages {
        match m.role.as_str() {
            "user" => prompt.extend(chat::user(&model, m.content.as_deref().unwrap_or(""))),
            // system already handled; assistant/tool never appear on turn 0
            // in cumulative mode (see module docs) — ignore defensively.
            _ => {}
        }
    }
    prompt.extend(chat::cue(&model)); // "assistant, your turn" header
    if prompt.is_empty() {
        return error_response(responder, 400, "empty prompt after rendering").await;
    }

    let temperature = body.temperature.unwrap_or(1.0);
    let sampler = if temperature <= 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP { temperature, p: body.top_p.unwrap_or(1.0) }
    };
    let max_tokens = body.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

    // ── Prefill (final chunk stays buffered for the generator) ──
    let mut ctx = match Context::new(&model) {
        Ok(c) => c,
        Err(e) => return error_response(responder, 500, &e.to_string()).await,
    };
    {
        let chunks: Vec<&[u32]> = prompt.chunks(PREFILL_CHUNK).collect();
        let (last, head) = chunks.split_last().unwrap();
        for chunk in head {
            ctx.append(chunk);
            if let Err(e) = ctx.flush().await {
                return error_response(responder, 500, &e.to_string()).await;
            }
        }
        ctx.append(last);
    }

    // ── Generate, decoding text and tool calls in parallel ──
    let stop_ids = chat::stop_tokens(&model);
    let mut g = ctx.generate(sampler).max_tokens(max_tokens).stop(&stop_ids);
    // Only constrain when tools are present: native_matcher traps the
    // instance when handed an empty schema set.
    if !tool_schemas.is_empty() {
        if let Some(matcher) = tools::native_matcher(&model, &tool_schemas) {
            g = g.constrain(GrammarConstraint::new(matcher));
        }
    }
    let mut tool_decoder = (!tool_schemas.is_empty()).then(|| tools::Decoder::new(&model));
    let mut token_ids: Vec<u32> = Vec::new();
    let mut calls: Vec<(String, String)> = Vec::new();
    let mut finish = "length";

    'outer: loop {
        let step = match g.next() {
            Ok(Some(s)) => s,
            Ok(None) => break,
            Err(e) => return error_response(responder, 500, &e.to_string()).await,
        };
        let out = match step.execute().await {
            Ok(o) => o,
            Err(e) => return error_response(responder, 500, &e.to_string()).await,
        };
        for &t in &out.tokens {
            token_ids.push(t);
            if let Some(dec) = tool_decoder.as_mut() {
                if let Ok(tools::Event::Call(name, args)) = dec.feed(&[t]) {
                    calls.push((name, args));
                }
            }
            if stop_ids.contains(&t) {
                finish = "stop";
                break 'outer;
            }
            if token_ids.len() >= max_tokens {
                break 'outer;
            }
        }
    }

    // Stop token stays in token_ids (§3.1 accounting), never in text.
    let text_ids = strip_trailing_stop(&token_ids, &stop_ids);
    let text = model.tokenizer().decode(text_ids).unwrap_or_default();

    let uid = uniq();
    let tool_calls: Vec<serde_json::Value> = calls
        .iter()
        .enumerate()
        .map(|(i, (name, args))| {
            serde_json::json!({
                "index": i,
                "id": format!("call_{uid}_{i}"),
                "type": "function",
                "function": { "name": name, "arguments": args },
            })
        })
        .collect();

    let mut message = serde_json::json!({ "role": "assistant", "content": text });
    if !tool_calls.is_empty() {
        message["tool_calls"] = serde_json::Value::Array(tool_calls);
        finish = "tool_calls";
    }

    let response = serde_json::json!({
        "id": format!("chatcmpl-{uid}"),
        "object": "chat.completion",
        "created": now(),
        "model": model_name,
        "prompt_token_ids": prompt,       // §3.2: gateway seeds its accumulator
        "weight_version": WEIGHT_VERSION,
        "choices": [{
            "index": 0,
            "message": message,
            "token_ids": token_ids,       // §3.2: REQUIRED
            "finish_reason": finish,
            "logprobs": null,
        }],
        "usage": {
            "prompt_tokens": prompt.len(),
            "completion_tokens": token_ids.len(),
            "total_tokens": prompt.len() + token_ids.len(),
        },
    });

    let http = Response::builder()
        .header("Content-Type", "application/json")
        .body(response.to_string().into_body())
        .unwrap();
    responder.respond(http).await
}
