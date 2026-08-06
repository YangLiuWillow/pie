//! `POST /v1/completions` — the cumulative-token-mode contract (§3.1).
//!
//! Request (gateway → worker):
//! ```jsonc
//! {
//!   "prompt": [151644, ...],        // list[int] — pre-tokenized; or string (debug)
//!   "add_special_tokens": false,    // accepted, irrelevant for id prompts
//!   "logprobs": true,               // bool OR int — lenient; v1 returns none
//!   "return_token_ids": true,
//!   "max_tokens": N, "temperature": T, "top_p": P,
//!   "stream": true                  // SSE variant
//! }
//! ```
//!
//! Response requirements the gateway parses (data_process.py):
//! root `prompt_token_ids` (echo of the input ids), `choices[0].token_ids`
//! (REQUIRED — missing ⇒ EnrichMismatchError ⇒ rollout retry burn),
//! `choices[0].text`, `finish_reason` ("stop" | "length"), optional root
//! `weight_version`, `usage`.
//!
//! Invariants:
//! - The sampled stop token IS included in `token_ids` (the next turn's
//!   cumulative prompt contains the full sealed assistant turn, so the
//!   trainer's prefix-merge needs it accounted for) but NOT in `text`
//!   (vLLM semantics: stop strings never appear in output text).
//! - Raw output only: no visibility filtering, no tool-call parsing —
//!   `<think>`/`<tool_call>` blocks stream out verbatim as text/ids; the
//!   gateway's renderer owns parsing (rllm proxy.py `_parsed_chat_message`).

use inferlet::model::Model;
use inferlet::sample::Sampler;
use inferlet::{runtime, Context};
use serde::Deserialize;
use wstd::http::body::BodyForthcoming;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};
use wstd::io::AsyncWrite;

const DEFAULT_MAX_TOKENS: usize = 4096;

/// Max tokens per prefill forward pass — a single multi-thousand-token
/// forward outlives the engine's per-forward timeout on CPU drivers
/// (codex-responses lesson).
const PREFILL_CHUNK: usize = 1024;

/// Stamped on every response. Weight updates don't exist yet (gap G1);
/// when they do, this becomes the served checkpoint's version.
const WEIGHT_VERSION: u64 = 0;

#[derive(Deserialize)]
struct CompletionsBody {
    prompt: serde_json::Value,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
    // Accepted for wire compatibility; unused in v1.
    #[serde(default)]
    #[allow(dead_code)]
    logprobs: Option<serde_json::Value>, // bool from the gateway, int from vLLM clients
    #[serde(default)]
    #[allow(dead_code)]
    add_special_tokens: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    return_token_ids: Option<bool>,
    #[serde(default)]
    model: Option<String>,
}

struct Turn {
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    sampler: Sampler,
    stream: bool,
    model_name: String,
}

fn uniq_fragment() -> String {
    runtime::instance_id()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(12)
        .collect()
}

fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn parse_turn(body_bytes: &[u8], model: &Model) -> Result<Turn, String> {
    let body: CompletionsBody =
        serde_json::from_slice(body_bytes).map_err(|e| format!("Invalid JSON: {e}"))?;

    let prompt_ids: Vec<u32> = match &body.prompt {
        serde_json::Value::Array(items) => items
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as u32)
                    .ok_or_else(|| "prompt array must contain token ids (u32)".to_string())
            })
            .collect::<Result<_, _>>()?,
        serde_json::Value::String(text) => model.tokenizer().encode(text),
        _ => return Err("prompt must be a list of token ids or a string".to_string()),
    };
    if prompt_ids.is_empty() {
        return Err("prompt must not be empty".to_string());
    }

    let temperature = body.temperature.unwrap_or(1.0);
    let top_p = body.top_p.unwrap_or(1.0);
    let sampler = if temperature <= 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP { temperature, p: top_p }
    };

    Ok(Turn {
        prompt_ids,
        max_tokens: body.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        sampler,
        stream: body.stream.unwrap_or(false),
        model_name: body.model.unwrap_or_default(),
    })
}

pub async fn handle(body_bytes: Vec<u8>, responder: Responder) -> Finished {
    let models = runtime::models();
    let served = match models.first() {
        Some(n) => n.clone(),
        None => return error_response(responder, 500, "no models available").await,
    };
    let model = match Model::load(&served) {
        Ok(m) => m,
        Err(e) => return error_response(responder, 500, &e.to_string()).await,
    };

    let mut turn = match parse_turn(&body_bytes, &model) {
        Ok(t) => t,
        Err(msg) => return error_response(responder, 400, &msg).await,
    };
    if turn.model_name.is_empty() {
        turn.model_name = served;
    }

    if turn.stream {
        handle_streaming(model, turn, responder).await
    } else {
        handle_non_streaming(model, turn, responder).await
    }
}

/// Prefill the raw prompt ids. NO `cue()` — a cumulative prompt already ends
/// with the generation header the renderer put there; adding another would
/// corrupt the token accounting.
///
/// The FINAL chunk stays in the buffer unflushed: the generator's first step
/// drains it, so its forward pass carries tokens. Flushing everything first
/// would make that step a zero-token forward, which the portable driver
/// rejects ("plan: request 0 has zero tokens").
async fn prefill(ctx: &mut Context, prompt_ids: &[u32]) -> Result<(), String> {
    let chunks: Vec<&[u32]> = prompt_ids.chunks(PREFILL_CHUNK).collect();
    let (last, head) = chunks.split_last().expect("prompt verified non-empty");
    for chunk in head {
        ctx.append(chunk);
        ctx.flush().await.map_err(|e| e.to_string())?;
    }
    ctx.append(last);
    Ok(())
}

/// Incremental UTF-8-safe text of `ids[..]`, as a delta against what was
/// already emitted. Decode failures (mid-codepoint boundaries) hold the
/// delta back until a later token completes the sequence.
struct IncrementalText {
    emitted: String,
}

impl IncrementalText {
    fn new() -> Self {
        Self { emitted: String::new() }
    }
    fn delta(&mut self, model: &Model, ids: &[u32]) -> String {
        match model.tokenizer().decode(ids) {
            Ok(full) if full.len() >= self.emitted.len() && full.starts_with(&self.emitted) => {
                let d = full[self.emitted.len()..].to_string();
                self.emitted = full;
                d
            }
            _ => String::new(),
        }
    }
}

struct GenOutcome {
    token_ids: Vec<u32>,
    finish_reason: &'static str,
}

// ─── Non-streaming ─────────────────────────────────────────────────────────

async fn handle_non_streaming(model: Model, turn: Turn, responder: Responder) -> Finished {
    let mut ctx = match Context::new(&model) {
        Ok(c) => c,
        Err(e) => return error_response(responder, 500, &e.to_string()).await,
    };
    if let Err(e) = prefill(&mut ctx, &turn.prompt_ids).await {
        return error_response(responder, 500, &e).await;
    }

    let stop_ids = inferlet::chat::stop_tokens(&model);
    let outcome = match generate(&mut ctx, &turn, &stop_ids).await {
        Ok(o) => o,
        Err(e) => return error_response(responder, 500, &e).await,
    };

    // Stop token stays in token_ids, never in text.
    let text_ids = strip_trailing_stop(&outcome.token_ids, &stop_ids);
    let text = model.tokenizer().decode(text_ids).unwrap_or_default();

    let response = serde_json::json!({
        "id": format!("cmpl-{}", uniq_fragment()),
        "object": "text_completion",
        "created": now_unix_secs(),
        "model": turn.model_name,
        "prompt_token_ids": turn.prompt_ids,
        "weight_version": WEIGHT_VERSION,
        "choices": [{
            "index": 0,
            "text": text,
            "token_ids": outcome.token_ids,
            "finish_reason": outcome.finish_reason,
            "logprobs": null,
        }],
        "usage": {
            "prompt_tokens": turn.prompt_ids.len(),
            "completion_tokens": outcome.token_ids.len(),
            "total_tokens": turn.prompt_ids.len() + outcome.token_ids.len(),
        },
    });

    let http_response = Response::builder()
        .header("Content-Type", "application/json")
        .body(response.to_string().into_body())
        .unwrap();
    responder.respond(http_response).await
}

async fn generate(
    ctx: &mut Context,
    turn: &Turn,
    stop_ids: &[u32],
) -> Result<GenOutcome, String> {
    let mut g = ctx.generate(turn.sampler.clone()).max_tokens(turn.max_tokens).stop(stop_ids);
    let mut token_ids: Vec<u32> = Vec::new();
    let mut finish_reason = "length";

    'outer: loop {
        let step = match g.next().map_err(|e| e.to_string())? {
            Some(s) => s,
            None => break,
        };
        let out = step.execute().await.map_err(|e| e.to_string())?;
        for &t in &out.tokens {
            token_ids.push(t);
            if stop_ids.contains(&t) {
                finish_reason = "stop";
                break 'outer;
            }
            if token_ids.len() >= turn.max_tokens {
                break 'outer;
            }
        }
    }
    Ok(GenOutcome { token_ids, finish_reason })
}

fn strip_trailing_stop<'a>(ids: &'a [u32], stop_ids: &[u32]) -> &'a [u32] {
    match ids.last() {
        Some(last) if stop_ids.contains(last) => &ids[..ids.len() - 1],
        _ => ids,
    }
}

// ─── Streaming (SSE) ───────────────────────────────────────────────────────
//
// Chunk contract (gateway's build_trace_record_from_chunks): the FIRST chunk
// carries root `prompt_token_ids`; every chunk carries `choices[0].token_ids`
// deltas; the last data chunk carries `finish_reason` + `usage`; terminator
// `data: [DONE]`.

async fn handle_streaming(model: Model, turn: Turn, responder: Responder) -> Finished {
    let sse_response = Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(BodyForthcoming)
        .unwrap();
    let mut body = responder.start_response(sse_response);

    let id = format!("cmpl-{}", uniq_fragment());
    let created = now_unix_secs();

    macro_rules! emit {
        ($payload:expr) => {{
            let line = format!("data: {}\n\n", $payload);
            if body.write_all(line.as_bytes()).await.is_err() {
                return Finished::finish(body, Ok(()), None);
            }
            if body.flush().await.is_err() {
                return Finished::finish(body, Ok(()), None);
            }
        }};
    }
    macro_rules! fail {
        ($msg:expr) => {{
            let err = serde_json::json!({"error": {"type": "server_error", "message": $msg}});
            emit!(err.to_string());
            emit!("[DONE]");
            return Finished::finish(body, Ok(()), None);
        }};
    }

    let mut ctx = match Context::new(&model) {
        Ok(c) => c,
        Err(e) => fail!(e.to_string()),
    };

    // First chunk before the (potentially slow, CPU) prefill: carries the
    // prompt-id echo and doubles as the client's liveness signal.
    let head = serde_json::json!({
        "id": id, "object": "text_completion", "created": created,
        "model": turn.model_name,
        "prompt_token_ids": turn.prompt_ids,
        "weight_version": WEIGHT_VERSION,
        "choices": [{"index": 0, "text": "", "token_ids": [], "finish_reason": null}],
    });
    emit!(head.to_string());

    // Chunked prefill with an SSE comment per chunk as a keepalive. The final
    // chunk stays buffered for the generator's first step (see `prefill`).
    {
        let chunks: Vec<&[u32]> = turn.prompt_ids.chunks(PREFILL_CHUNK).collect();
        let (last, head) = chunks.split_last().expect("prompt verified non-empty");
        for chunk in head {
            ctx.append(chunk);
            if let Err(e) = ctx.flush().await {
                fail!(e.to_string());
            }
            if body.write_all(b": prefill\n\n").await.is_err() {
                return Finished::finish(body, Ok(()), None);
            }
            let _ = body.flush().await;
        }
        ctx.append(last);
    }

    let stop_ids = inferlet::chat::stop_tokens(&model);
    let mut g = ctx.generate(turn.sampler.clone()).max_tokens(turn.max_tokens).stop(&stop_ids);
    let mut all_ids: Vec<u32> = Vec::new();
    let mut text = IncrementalText::new();
    let mut finish_reason = "length";

    'outer: loop {
        let step = match g.next() {
            Ok(Some(s)) => s,
            Ok(None) => break,
            Err(e) => fail!(e.to_string()),
        };
        let out = match step.execute().await {
            Ok(o) => o,
            Err(e) => fail!(e.to_string()),
        };
        if out.tokens.is_empty() {
            continue;
        }

        let mut step_ids: Vec<u32> = Vec::new();
        let mut stopped = false;
        for &t in &out.tokens {
            step_ids.push(t);
            all_ids.push(t);
            if stop_ids.contains(&t) {
                finish_reason = "stop";
                stopped = true;
                break;
            }
            if all_ids.len() >= turn.max_tokens {
                stopped = true;
                break;
            }
        }

        let delta = text.delta(&model, strip_trailing_stop(&all_ids, &stop_ids));
        let chunk = serde_json::json!({
            "id": id, "object": "text_completion", "created": created,
            "model": turn.model_name,
            "choices": [{"index": 0, "text": delta, "token_ids": step_ids, "finish_reason": null}],
        });
        emit!(chunk.to_string());

        if stopped {
            break 'outer;
        }
    }

    let tail = serde_json::json!({
        "id": id, "object": "text_completion", "created": created,
        "model": turn.model_name,
        "choices": [{"index": 0, "text": "", "token_ids": [], "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": turn.prompt_ids.len(),
            "completion_tokens": all_ids.len(),
            "total_tokens": turn.prompt_ids.len() + all_ids.len(),
        },
    });
    emit!(tail.to_string());
    emit!("[DONE]");
    Finished::finish(body, Ok(()), None)
}

// ─── Errors ────────────────────────────────────────────────────────────────

pub async fn error_response(responder: Responder, status: u16, message: &str) -> Finished {
    let error = serde_json::json!({
        "error": { "type": "invalid_request", "message": message }
    });
    let response = Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();
    responder.respond(response).await
}
