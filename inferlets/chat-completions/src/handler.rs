//! Request handler for POST /v1/chat/completions (qwen-code-facing).
//!
//! Contract (audit §1 hard-requirements table, `docs/qwen-code-rl-audit.md`):
//! - Malformed request → 400 with an OpenAI error body; 500 is reserved for
//!   genuine server faults (500s trigger 7×-app × 3×-SDK retry storms).
//! - Never a context-length 400 (fires H1 full-history compaction
//!   client-side): overflow/generation faults degrade to
//!   `finish_reason:"length"` with whatever streamed.
//! - Text turns must end with non-empty content; tool-call ids must be
//!   unique across the whole session (fresh WASM instance per request, so
//!   ids derive from `runtime::instance_id()`).
//!
//! Generation runs UNCONSTRAINED (grammar-constraining the whole turn
//! suppressed reasoning text and collapsed t=0 trajectories into action
//! loops — openhands-completion finding), with a fenced-JSON fallback
//! parser for models that write calls as ``` blocks. There is deliberately
//! no grammar-forced phase-2 call: constrained decoding traps the guest on
//! the portable driver (see the note at the former phase-2 site).

use crate::filter::VisibleFilter;
use crate::render;
use crate::session::{self, CanonItem};
use crate::streaming::{ChunkMeta, usage_object};
use crate::types::{ChatCompletionRequest, ChatMessage, tool_schema_envelopes};
use wstd::http::body::BodyForthcoming;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};
use wstd::io::AsyncWrite;

use inferlet::model::Model;
use inferlet::sample::Sampler;
use inferlet::{Context, chat, runtime, tools};

/// Defaults when the client sends none — Qwen3 no-think guidance (qwen-code
/// sends `max_tokens` always but `temperature`/`top_p` only if configured).
const DEFAULT_MAX_TOKENS: usize = 4096;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_P: f32 = 0.8;

/// Max tokens per prefill forward pass. A single multi-thousand-token
/// forward outlives the engine's per-forward timeout on slow (CPU/Metal)
/// drivers; bounded chunks keep each forward under it, and each chunk
/// boundary doubles as a keepalive point for the client's idle watchdog.
const PREFILL_CHUNK: usize = 1024;

fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Per-request unique id fragment. The daemon instantiates a fresh WASM
/// instance per request, so a static counter would restart at zero every
/// time and hand the client colliding tool-call ids — the exact `call_0`
/// dedup bug both prior integrations hit. `instance-id` is unique per
/// instantiation.
fn uniq_fragment() -> String {
    runtime::instance_id()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(12)
        .collect()
}

/// Everything the turn needs, resolved from the parsed request.
struct TurnSetup {
    messages: Vec<ChatMessage>,
    tool_schemas: Vec<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    no_think: bool,
    stop_strings: Vec<String>,
    include_usage: bool,
}

impl TurnSetup {
    fn sampler(&self) -> Sampler {
        if self.temperature <= 0.0 {
            Sampler::Argmax
        } else {
            Sampler::TopP { temperature: self.temperature, p: self.top_p }
        }
    }
}

fn parse_setup(body_bytes: &[u8]) -> Result<(TurnSetup, bool), String> {
    let request: ChatCompletionRequest =
        serde_json::from_slice(body_bytes).map_err(|e| format!("Invalid JSON: {e}"))?;
    if request.messages.is_empty() {
        return Err("`messages` must be a non-empty array".to_string());
    }
    let stream = request.stream;
    Ok((
        TurnSetup {
            tool_schemas: tool_schema_envelopes(&request.tools),
            max_tokens: request.effective_max_tokens(DEFAULT_MAX_TOKENS),
            temperature: request.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            top_p: request.top_p.unwrap_or(DEFAULT_TOP_P),
            no_think: request.no_think(),
            stop_strings: request.stop_strings(),
            include_usage: request.include_usage(),
            messages: request.messages,
        },
        stream,
    ))
}

/// A context plus the rendered-but-not-yet-prefilled prompt tokens.
struct BuiltContext {
    ctx: Context,
    /// Full history on a miss, just the new suffix on a resume hit.
    prefill: Vec<u32>,
    cached_tokens: u32,
    /// Snapshot this turn resumed from (deleted only after the new boundary
    /// saves — never before, so a client-level retry can still hit it).
    resumed_from: Option<String>,
    debug: String,
}

fn build_context(model: &Model, setup: &TurnSetup) -> Result<BuiltContext, String> {
    // Resume attempt: hash the prefix up to the last assistant message; the
    // previous request saved its post-generation context under exactly that
    // name. `open` forks the snapshot without consuming it (a qwen-code
    // turn-level retry re-POSTs the same request and must be able to
    // re-hit).
    let mut miss_debug = String::from("first turn (no resume point)");
    if let Some(split) = session::split_resume_point(&setup.messages) {
        let canons = session::canon_messages(&setup.messages[..split]);
        let name = session::snapshot_name(&setup.tool_schemas, setup.no_think, canons.iter());
        match Context::open(model, &name) {
            Ok(ctx) => {
                let cached = ctx.seq_len();
                let mut suffix = Vec::new();
                render::render_messages(model, &setup.messages[split..], setup.no_think, &mut suffix)
                    .map_err(|e| e.to_string())?;
                return Ok(BuiltContext {
                    ctx,
                    prefill: suffix,
                    cached_tokens: cached,
                    debug: format!("resume hit {name} (cached {cached})"),
                    resumed_from: Some(name),
                });
            }
            Err(e) => miss_debug = format!("resume miss {name}: {e}"),
        }
    }

    // Miss or first turn: full rebuild.
    let tokens = render::render_full(model, &setup.messages, &setup.tool_schemas, setup.no_think)
        .map_err(|e| e.to_string())?;
    let ctx = Context::new(model).map_err(|e| e.to_string())?;
    Ok(BuiltContext { ctx, prefill: tokens, cached_tokens: 0, resumed_from: None, debug: miss_debug })
}

/// Close the turn and save the KV snapshot under the address the *next*
/// request will hash to: incoming messages + the assistant output produced
/// this turn. On success, deletes the boundary this turn resumed from.
/// Failures are non-fatal (the next request pays a full rebuild).
async fn save_snapshot(
    ctx: &mut Context,
    model: &Model,
    setup: &TurnSetup,
    resumed_from: Option<&str>,
    visible_text: &str,
    calls: &[ToolCallOut],
) -> String {
    let mut canons = session::canon_messages(&setup.messages);
    if !visible_text.is_empty() {
        canons.push(CanonItem::Msg { role: "assistant".to_string(), text: visible_text.to_string() });
    }
    for c in calls {
        canons.push(CanonItem::Call {
            id: c.id.clone(),
            name: c.name.clone(),
            args: c.arguments.clone(),
        });
    }
    let name = session::snapshot_name(&setup.tool_schemas, setup.no_think, canons.iter());
    // The sampled stop token never enters the context (generation truncates
    // at it), so seal the assistant turn explicitly, then flush the buffer
    // through a forward pass so the snapshot holds committed KV.
    ctx.seal();
    if let Err(e) = ctx.flush().await {
        return format!("save flush failed: {e}");
    }
    match ctx.save(&name) {
        Ok(()) => {
            if let Some(old) = resumed_from {
                let _ = Context::delete(model, old);
            }
            format!("saved {name} (seq {})", ctx.seq_len())
        }
        Err(e) => format!("save {name} failed: {e}"),
    }
}

struct ToolCallOut {
    id: String,
    name: String,
    arguments: String,
}

/// Result of running phase-1 (+ phase-2/fence fallbacks) generation.
struct TurnOutcome {
    visible_text: String,
    /// Raw decoded generation (think markup included) — fallback source
    /// when the visible channel came out empty.
    raw_text: String,
    calls: Vec<ToolCallOut>,
    prompt_tokens: u32,
    n_generated: usize,
    hit_max: bool,
    /// Set when generation died mid-turn (degraded to "length", audit §1
    /// req. 8 — never surface a context-length error to qwen-code).
    gen_error: Option<String>,
}

impl TurnOutcome {
    fn finish_reason(&self) -> &'static str {
        if !self.calls.is_empty() {
            "tool_calls"
        } else if self.hit_max || self.gen_error.is_some() {
            "length"
        } else {
            "stop"
        }
    }

    /// The one canonical content string for this turn — used for BOTH the
    /// response and the snapshot address. qwen-code echoes response content
    /// back verbatim next turn, so any response/save divergence here breaks
    /// every subsequent KV resume (observed live: an empty-visible turn
    /// answered " " but saved no assistant message — permanent miss).
    ///
    /// Empty text on a no-tool-call turn trips qwen-code's
    /// NO_RESPONSE_TEXT retry loop, so fall back to the raw generation with
    /// think markup stripped (the model's actual words when the budget died
    /// inside a think block), and to a non-whitespace placeholder as the
    /// last resort. Phase 2 makes this near-unreachable with tools equipped.
    fn final_content(&self) -> String {
        if !self.visible_text.is_empty() || !self.calls.is_empty() {
            return self.visible_text.clone();
        }
        let cleaned = self
            .raw_text
            .replace("<think>", "")
            .replace("</think>", "")
            .trim()
            .to_string();
        if cleaned.is_empty() { "…".to_string() } else { cleaned }
    }
}

pub async fn handle_chat_completions(body_bytes: Vec<u8>, responder: Responder) -> Finished {
    let (setup, stream) = match parse_setup(&body_bytes) {
        Ok(v) => v,
        Err(msg) => return error_response(responder, 400, "invalid_request_error", &msg).await,
    };

    if stream {
        handle_streaming(setup, responder).await
    } else {
        handle_non_streaming(setup, responder).await
    }
}

// ─── Streaming ─────────────────────────────────────────────────────────────

async fn handle_streaming(mut setup: TurnSetup, responder: Responder) -> Finished {
    let uniq = uniq_fragment();
    let models = runtime::models();
    let meta = ChunkMeta {
        id: format!("chatcmpl-{uniq}"),
        model: models.first().cloned().unwrap_or_else(|| "unknown".to_string()),
        created: now_unix_secs(),
    };

    let sse_response = Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(BodyForthcoming)
        .unwrap();
    let mut body = responder.start_response(sse_response);

    macro_rules! emit {
        ($event:expr) => {{
            if body.write_all($event.as_bytes()).await.is_err() {
                return Finished::finish(body, Ok(()), None);
            }
            if body.flush().await.is_err() {
                return Finished::finish(body, Ok(()), None);
            }
        }};
    }

    // Role chunk first — announces the assistant turn and doubles as the
    // first keepalive before a potentially long prefill.
    emit!(meta.role_chunk());

    // Headers are already on the wire: from here every failure must be
    // shaped as a well-formed turn (finish_reason "length"), never an SSE
    // abort — a mid-stream cut triggers qwen-code's synthetic-continuation
    // path (H3).
    macro_rules! degrade {
        ($msg:expr) => {{
            eprintln!("[chat-completions] degraded turn: {}", $msg);
            emit!(meta.content_delta(" "));
            emit!(meta.finish_chunk("length"));
            if setup.include_usage {
                emit!(meta.usage_chunk(0, 0, 0));
            }
            emit!(ChunkMeta::done());
            return Finished::finish(body, Ok(()), None);
        }};
    }

    let model = match models
        .first()
        .ok_or_else(|| "no models available".to_string())
        .and_then(|n| Model::load(n).map_err(|e| e.to_string()))
    {
        Ok(m) => m,
        Err(e) => degrade!(format!("model load failed: {e}")),
    };

    render::sanitize_messages(&mut setup.messages, &model);

    let BuiltContext { mut ctx, prefill, cached_tokens, resumed_from, debug: build_debug } =
        match build_context(&model, &setup) {
            Ok(b) => b,
            Err(e) => degrade!(format!("context build failed: {e}")),
        };

    // Chunked prefill, one keepalive per chunk (empty delta — resets the
    // client's 240 s idle watchdog through the OpenAI SDK).
    for chunk in prefill.chunks(PREFILL_CHUNK) {
        ctx.append(chunk);
        if let Err(e) = ctx.flush().await {
            degrade!(format!("prefill failed: {e}"));
        }
        emit!(meta.keepalive());
    }

    let mut emitted_visible = false;
    let outcome = {
        // Generation with per-event emission inlined (the emit! macro owns
        // `body`, so the loop lives here rather than in a helper).
        ctx.cue();
        let prompt_tokens = ctx.seq_len() + ctx.buffer().len() as u32;

        let mut stop_token_ids = chat::stop_tokens(&model);
        // Also stop at the turn-START marker: at t=0 a looping model starts
        // simulating the next turn instead of stopping, and leaked
        // "<|im_start|>" round-trips through message content into fake turn
        // boundaries on the next request (openhands traj job 18825434).
        let turn_start = model.tokenizer().encode("<|im_start|>");
        if turn_start.len() == 1 && !stop_token_ids.contains(&turn_start[0]) {
            stop_token_ids.push(turn_start[0]);
        }

        let has_tools = !setup.tool_schemas.is_empty();
        let mut tool_decoder = has_tools.then(|| tools::Decoder::new(&model));
        let mut text_decoder = chat::Decoder::new(&model);
        let mut filter = VisibleFilter::new();

        let mut generated: Vec<u32> = Vec::new();
        let mut visible_text = String::new();
        let mut calls: Vec<ToolCallOut> = Vec::new();
        let mut gen_error: Option<String> = None;

        let mut g = ctx
            .generate(setup.sampler())
            .max_tokens(setup.max_tokens)
            .stop(&stop_token_ids);

        'outer: loop {
            let step = match g.next() {
                Ok(Some(s)) => s,
                Ok(None) => break,
                Err(e) => {
                    gen_error = Some(e.to_string());
                    break;
                }
            };
            let out = match step.execute().await {
                Ok(o) => o,
                Err(e) => {
                    gen_error = Some(e.to_string());
                    break;
                }
            };

            for &t in &out.tokens {
                generated.push(t);

                if let Some(dec) = tool_decoder.as_mut() {
                    if let Ok(tools::Event::Call(name, args)) = dec.feed(&[t]) {
                        // Looping models emit the same call several times in
                        // one turn; executing the copies just burns agent
                        // iterations.
                        let dup = calls.iter().any(|c| c.name == name && c.arguments == args);
                        if !dup {
                            let call_id = format!("call_{uniq}_{}", calls.len());
                            emit!(meta.tool_call_delta(calls.len(), &call_id, &name, &args));
                            calls.push(ToolCallOut { id: call_id, name, arguments: args });
                        }
                    }
                }

                if stop_token_ids.contains(&t) {
                    break 'outer;
                }

                match text_decoder.feed(&[t]) {
                    Ok(chat::Event::Delta(s)) => {
                        let v = filter.feed(&s);
                        if !v.is_empty() {
                            emit!(meta.content_delta(&v));
                            emitted_visible = true;
                            visible_text.push_str(&v);
                        }
                    }
                    Ok(chat::Event::Done(_)) => break 'outer,
                    _ => {}
                }
            }

            // Client-supplied stop strings (not sent by qwen-code; checked
            // on the visible tail for API completeness — deltas already on
            // the wire are not retracted).
            if !setup.stop_strings.is_empty()
                && setup.stop_strings.iter().any(|s| visible_text.ends_with(s))
            {
                break;
            }

            if generated.len() >= setup.max_tokens {
                break;
            }
        }

        // Flush any held-back filter tail.
        let tail = filter.finish();
        if !tail.is_empty() {
            emit!(meta.content_delta(&tail));
            emitted_visible = true;
            visible_text.push_str(&tail);
        }
        // No trimming/truncation past this point: `visible_text` is exactly
        // the bytes already streamed, qwen-code echoes those back verbatim
        // as the assistant content next turn, and the snapshot address must
        // hash that same string or every subsequent KV resume misses.

        // Fallback for models that write a tool call as a fenced JSON block
        // instead of <tool_call> tags (observed on Qwen coder models at t=0
        // with the grammar constraint off). The fence bytes stay in the
        // content (already on the wire); the calls are surfaced on top.
        if calls.is_empty() && has_tools {
            for (_, name, args) in parse_fenced_tool_calls(&visible_text) {
                let dup = calls.iter().any(|c| c.name == name && c.arguments == args);
                if !dup {
                    let call_id = format!("call_{uniq}_{}", calls.len());
                    emit!(meta.tool_call_delta(calls.len(), &call_id, &name, &args));
                    calls.push(ToolCallOut { id: call_id, name, arguments: args });
                }
            }
        }
        // Coder-XML salvage: `<function=…>` blocks that arrived without the
        // `<tool_call>` wrapper (native decoder never armed).
        if calls.is_empty() && has_tools && visible_text.contains("<function=") {
            for (name, args) in parse_coder_xml_calls(&visible_text, &setup.tool_schemas) {
                let call_id = format!("call_{uniq}_{}", calls.len());
                emit!(meta.tool_call_delta(calls.len(), &call_id, &name, &args));
                calls.push(ToolCallOut { id: call_id, name, arguments: args });
            }
        }

        // NOTE — no phase-2 forced tool call here, deliberately. The
        // openhands-completion pattern (replay the prose on a fork and force
        // one call under the tool-call grammar) TRAPS the guest on the
        // portable driver: grammar-constrained decoding isn't implemented
        // there, and a trap kills the stream before the finish chunk —
        // qwen-code then burns its 4 NO_FINISH_REASON retries on identical
        // failures (observed live, 2026-08-10). A no-tool-call turn is
        // handled gracefully by qwen-code; a dead stream is not. Re-add
        // behind a capability probe when the driver grows grammar support.

        let n_generated = generated.len();
        TurnOutcome {
            hit_max: n_generated >= setup.max_tokens,
            raw_text: model.tokenizer().decode(&generated).unwrap_or_default(),
            visible_text,
            calls,
            prompt_tokens,
            n_generated,
            gen_error,
        }
    };
    let final_content = outcome.final_content();

    if let Some(e) = &outcome.gen_error {
        eprintln!("[chat-completions] generation degraded to length-finish: {e}");
    }

    // A text turn that streamed nothing visible must still deliver
    // non-empty content (NO_RESPONSE_TEXT retry loop otherwise). The
    // fallback IS the turn's canonical content, so it also feeds the
    // snapshot address below.
    if !emitted_visible && outcome.calls.is_empty() {
        emit!(meta.content_delta(&final_content));
    }

    // Save the KV snapshot *before* signalling completion: qwen-code fires
    // the follow-up request the moment the stream closes, and the snapshot
    // must already exist for the resume to hit.
    let save_debug = save_snapshot(
        &mut ctx,
        &model,
        &setup,
        resumed_from.as_deref(),
        &final_content,
        &outcome.calls,
    )
    .await;
    eprintln!("[chat-completions] {build_debug}; {save_debug}");

    emit!(meta.finish_chunk(outcome.finish_reason()));
    if setup.include_usage {
        emit!(meta.usage_chunk(
            outcome.prompt_tokens,
            outcome.n_generated as u32,
            cached_tokens
        ));
    }
    emit!(ChunkMeta::done());

    Finished::finish(body, Ok(()), None)
}

// ─── Non-streaming ─────────────────────────────────────────────────────────
//
// qwen-code always streams; this path exists for curl debugging and the
// acceptance tests. Same pipeline, no mid-turn emission, no phase-2 fork
// (debug surface only — keep it simple).

async fn handle_non_streaming(mut setup: TurnSetup, responder: Responder) -> Finished {
    let uniq = uniq_fragment();
    let models = runtime::models();
    let model_name = models.first().cloned().unwrap_or_else(|| "unknown".to_string());
    let model = match models
        .first()
        .ok_or_else(|| "no models available".to_string())
        .and_then(|n| Model::load(n).map_err(|e| e.to_string()))
    {
        Ok(m) => m,
        Err(e) => return error_response(responder, 500, "server_error", &e).await,
    };

    render::sanitize_messages(&mut setup.messages, &model);

    let BuiltContext { mut ctx, prefill, cached_tokens, resumed_from, debug: build_debug } =
        match build_context(&model, &setup) {
            Ok(b) => b,
            Err(e) => return error_response(responder, 500, "server_error", &e).await,
        };

    for chunk in prefill.chunks(PREFILL_CHUNK) {
        ctx.append(chunk);
        if let Err(e) = ctx.flush().await {
            return error_response(responder, 500, "server_error", &e.to_string()).await;
        }
    }
    ctx.cue();
    let prompt_tokens = ctx.seq_len() + ctx.buffer().len() as u32;

    let mut stop_token_ids = chat::stop_tokens(&model);
    let turn_start = model.tokenizer().encode("<|im_start|>");
    if turn_start.len() == 1 && !stop_token_ids.contains(&turn_start[0]) {
        stop_token_ids.push(turn_start[0]);
    }

    let has_tools = !setup.tool_schemas.is_empty();
    let mut tool_decoder = has_tools.then(|| tools::Decoder::new(&model));
    let mut text_decoder = chat::Decoder::new(&model);
    let mut filter = VisibleFilter::new();

    let mut generated: Vec<u32> = Vec::new();
    let mut visible_text = String::new();
    let mut calls: Vec<ToolCallOut> = Vec::new();
    let mut gen_error: Option<String> = None;

    let mut g = ctx
        .generate(setup.sampler())
        .max_tokens(setup.max_tokens)
        .stop(&stop_token_ids);

    'outer: loop {
        let step = match g.next() {
            Ok(Some(s)) => s,
            Ok(None) => break,
            Err(e) => {
                gen_error = Some(e.to_string());
                break;
            }
        };
        let out = match step.execute().await {
            Ok(o) => o,
            Err(e) => {
                gen_error = Some(e.to_string());
                break;
            }
        };
        for &t in &out.tokens {
            generated.push(t);
            if let Some(dec) = tool_decoder.as_mut() {
                if let Ok(tools::Event::Call(name, args)) = dec.feed(&[t]) {
                    let dup = calls.iter().any(|c| c.name == name && c.arguments == args);
                    if !dup {
                        calls.push(ToolCallOut {
                            id: format!("call_{uniq}_{}", calls.len()),
                            name,
                            arguments: args,
                        });
                    }
                }
            }
            if stop_token_ids.contains(&t) {
                break 'outer;
            }
            if let Ok(chat::Event::Delta(s)) = text_decoder.feed(&[t]) {
                visible_text.push_str(&filter.feed(&s));
            }
        }
        if generated.len() >= setup.max_tokens {
            break;
        }
    }
    drop(g);
    visible_text.push_str(&filter.finish());
    let visible_text = visible_text.trim_end().to_string();

    if calls.is_empty() && has_tools && visible_text.contains("<function=") {
        for (name, args) in parse_coder_xml_calls(&visible_text, &setup.tool_schemas) {
            calls.push(ToolCallOut {
                id: format!("call_{uniq}_{}", calls.len()),
                name,
                arguments: args,
            });
        }
    }

    let outcome = TurnOutcome {
        hit_max: generated.len() >= setup.max_tokens,
        n_generated: generated.len(),
        raw_text: model.tokenizer().decode(&generated).unwrap_or_default(),
        visible_text,
        calls,
        prompt_tokens,
        gen_error,
    };
    let final_content = outcome.final_content();
    if let Some(e) = &outcome.gen_error {
        eprintln!("[chat-completions] generation degraded to length-finish: {e}");
    }

    let save_debug = save_snapshot(
        &mut ctx,
        &model,
        &setup,
        resumed_from.as_deref(),
        &final_content,
        &outcome.calls,
    )
    .await;
    eprintln!("[chat-completions] {build_debug}; {save_debug}");

    let tool_calls_json: Vec<serde_json::Value> = outcome
        .calls
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            })
        })
        .collect();

    let response = serde_json::json!({
        "id": format!("chatcmpl-{uniq}"),
        "object": "chat.completion",
        "created": now_unix_secs(),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": if final_content.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(final_content.clone())
                },
                "tool_calls": if tool_calls_json.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::Array(tool_calls_json)
                },
            },
            "logprobs": null,
            "finish_reason": outcome.finish_reason(),
        }],
        "usage": usage_object(outcome.prompt_tokens, outcome.n_generated as u32, cached_tokens),
    });

    let http_response = Response::builder()
        .header("Content-Type", "application/json")
        .body(response.to_string().into_body())
        .unwrap();
    responder.respond(http_response).await
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Salvage parser for Qwen3-Coder's XML tool dialect when the model omits
/// the `<tool_call>` wrapper (observed on 30B with `general`-style prompt
/// examples: `<function=name>` blocks arrive bare, the native decoder never
/// arms, and the call leaks into content as text). Parses
/// `<function=NAME><parameter=key>value</parameter>…</function>` into
/// `(name, arguments_json)` pairs, typing values via the tool schemas the
/// same way vLLM's Qwen3CoderToolParser does (parity reference:
/// integrations/openhands/pie_openhands/qwen3coder_parser.py).
fn parse_coder_xml_calls(text: &str, tool_schemas: &[String]) -> Vec<(String, String)> {
    // name -> {param -> type} from the schema envelopes.
    let mut param_types: std::collections::HashMap<String, std::collections::HashMap<String, String>> =
        std::collections::HashMap::new();
    for schema in tool_schemas {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(schema) else { continue };
        let Some(name) = v.get("name").and_then(|n| n.as_str()) else { continue };
        let mut types = std::collections::HashMap::new();
        if let Some(props) = v.pointer("/parameters/properties").and_then(|p| p.as_object()) {
            for (k, spec) in props {
                if let Some(t) = spec.get("type").and_then(|t| t.as_str()) {
                    types.insert(k.clone(), t.to_string());
                }
            }
        }
        param_types.insert(name.to_string(), types);
    }

    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(rel) = text[pos..].find("<function=") {
        let fn_at = pos + rel;
        let after = &text[fn_at + "<function=".len()..];
        let Some(name_end) = after.find('>') else { break };
        let name = after[..name_end].trim().to_string();
        let body_start = fn_at + "<function=".len() + name_end + 1;
        let Some(body_len) = text[body_start..].find("</function>") else { break };
        let body = &text[body_start..body_start + body_len];
        pos = body_start + body_len + "</function>".len();

        let types = param_types.get(&name);
        let mut args = serde_json::Map::new();
        let mut bpos = 0;
        while let Some(prel) = body[bpos..].find("<parameter=") {
            let p_at = bpos + prel;
            let pafter = &body[p_at + "<parameter=".len()..];
            let Some(key_end) = pafter.find('>') else { break };
            let key = pafter[..key_end].trim().to_string();
            let val_start = p_at + "<parameter=".len() + key_end + 1;
            let Some(val_len) = body[val_start..].find("</parameter>") else { break };
            // The template frames values with newlines; strip exactly one
            // leading and one trailing newline (vLLM parser behavior).
            let raw = &body[val_start..val_start + val_len];
            let val = raw.strip_prefix('\n').unwrap_or(raw);
            let val = val.strip_suffix('\n').unwrap_or(val);
            bpos = val_start + val_len + "</parameter>".len();

            let typed: serde_json::Value = match types.and_then(|t| t.get(&key)).map(String::as_str) {
                Some("integer") => val.trim().parse::<i64>().map(Into::into)
                    .unwrap_or_else(|_| serde_json::Value::String(val.to_string())),
                Some("number") => val.trim().parse::<f64>().ok()
                    .and_then(|f| serde_json::Number::from_f64(f).map(serde_json::Value::Number))
                    .unwrap_or_else(|| serde_json::Value::String(val.to_string())),
                Some("boolean") => match val.trim() {
                    "true" => serde_json::Value::Bool(true),
                    "false" => serde_json::Value::Bool(false),
                    _ => serde_json::Value::String(val.to_string()),
                },
                Some("object") | Some("array") => serde_json::from_str(val.trim())
                    .unwrap_or_else(|_| serde_json::Value::String(val.to_string())),
                _ => serde_json::Value::String(val.to_string()),
            };
            args.insert(key, typed);
        }
        if !name.is_empty() {
            out.push((name, serde_json::Value::Object(args).to_string()));
        }
    }
    out
}

/// Extract tool calls written as fenced JSON blocks: a ``` fence (with or
/// without a language tag) whose body is an object with a string `name` and
/// an object `arguments`. Returns `(fence_byte_offset, name,
/// arguments_json)` per match, in order. (Port from openhands-completion.)
fn parse_fenced_tool_calls(text: &str) -> Vec<(usize, String, String)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(rel) = text[pos..].find("```") {
        let fence_at = pos + rel;
        let after = &text[fence_at + 3..];
        let Some(nl) = after.find('\n') else { break };
        let body_and_more = &after[nl + 1..];
        let Some(end) = body_and_more.find("```") else { break };
        let body = body_and_more[..end].trim();
        pos = fence_at + 3 + nl + 1 + end + 3;
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
            let name = v.get("name").and_then(|n| n.as_str());
            let args = v.get("arguments").filter(|a| a.is_object());
            if let (Some(name), Some(args)) = (name, args) {
                out.push((fence_at, name.to_string(), args.to_string()));
            }
        }
    }
    out
}

pub async fn error_response(
    responder: Responder,
    status_code: u16,
    error_type: &str,
    message: &str,
) -> Finished {
    let error = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "param": null,
            "code": null,
        }
    });
    let response = Response::builder()
        .status(status_code)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();
    responder.respond(response).await
}

#[cfg(test)]
mod tests {
    use super::parse_fenced_tool_calls;

    #[test]
    fn fenced_tool_call_is_extracted() {
        let text = "Prose first.\n\n```json\n{\"name\": \"run_shell_command\", \"arguments\": {\"command\": \"ls\"}}\n```";
        let calls = parse_fenced_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "run_shell_command");
        assert_eq!(&text[..calls[0].0], "Prose first.\n\n");
    }

    #[test]
    fn non_tool_fences_are_ignored() {
        let text = "```python\nprint('hi')\n```\n```json\n{\"foo\": 1}\n```";
        assert!(parse_fenced_tool_calls(text).is_empty());
    }
}

#[cfg(test)]
mod coder_xml_tests {
    use super::parse_coder_xml_calls;

    #[test]
    fn bare_function_block_is_salvaged_and_typed() {
        let schemas = vec![serde_json::json!({
            "name": "run_shell_command",
            "description": "d",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"}}}
        }).to_string()];
        let text = "I'll create it.\n\n<function=run_shell_command>\n<parameter=command>\necho 'x' > /tmp/a.txt\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_coder_xml_calls(text, &schemas);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "run_shell_command");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["command"], "echo 'x' > /tmp/a.txt");
        assert_eq!(args["timeout"], 30);
    }

    #[test]
    fn no_function_block_yields_nothing() {
        assert!(parse_coder_xml_calls("plain text </tool_call>", &[]).is_empty());
    }
}
