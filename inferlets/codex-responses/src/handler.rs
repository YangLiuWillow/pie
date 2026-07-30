//! Request handler for POST /responses (Codex-facing).
//!
//! Contract notes (from the Codex client source, codex-rs):
//! - Codex dispatches SSE on the JSON `type` field, not the `event:` line.
//! - Required for a turn: one `response.output_item.done` per produced item,
//!   then `response.completed` whose `response.id` must be present. Usage is
//!   optional but read when present (incl. `input_tokens_details.cached_tokens`).
//! - `function_call` items must carry `call_id`, `name`, and `arguments`
//!   (JSON-encoded string). Item `id`s must contain a `_` to be preserved,
//!   and must be unique across requests (Codex-side dedup by id).
//! - Codex sends no temperature / top_p / max_output_tokens; defaults here
//!   follow Qwen3 no-think guidance (0.7 / 0.8).
//! - Errors mid-stream: `response.failed` with `response.error.{code,message}`;
//!   unknown codes are treated as retryable by the client.

use crate::filter::VisibleFilter;
use crate::replay;
use crate::session::{self, CanonItem};
use crate::streaming::StreamEmitter;
use crate::types::*;
use wstd::http::body::BodyForthcoming;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};
use wstd::io::AsyncWrite;

use inferlet::model::Model;
use inferlet::sample::Sampler;
use inferlet::{chat, runtime, tools, Context, GrammarConstraint};

const DEFAULT_MAX_TOKENS: usize = 4096;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_P: f32 = 0.8;

fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Per-request unique id fragment. The daemon instantiates a fresh WASM
/// instance per request, so a static counter would restart at zero every
/// time and hand Codex colliding item ids (the exact `call_0` dedup bug the
/// OpenHands integration hit). `instance-id` is unique per instantiation.
fn uniq_fragment() -> String {
    runtime::instance_id()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(12)
        .collect()
}

/// Everything the turn needs, resolved from the parsed request.
struct TurnSetup {
    instructions: Option<String>,
    items: Vec<InputItem>,
    tool_schemas: Vec<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    /// Snapshot namespace — Codex's `prompt_cache_key` (session UUID), or
    /// "global" for clients that don't send one.
    cache_scope: String,
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

/// A context plus the rendered-but-not-yet-prefilled prompt tokens. The
/// caller runs the prefill so it can interleave SSE keepalives (Codex kills
/// a stream that is silent for 5 minutes, and a long CPU prefill is).
struct BuiltContext {
    ctx: Context,
    /// Rendered tokens still to prefill (full history on a miss, just the
    /// new suffix on a resume hit).
    prefill: Vec<u32>,
    cached_tokens: u32,
    /// Name of the snapshot we resumed from (deleted after the new boundary
    /// saves — never before, so a Codex turn-level retry can still hit it).
    resumed_from: Option<String>,
    /// Diagnostic trail for the response's `debug` field.
    debug: String,
}

/// Max tokens per prefill forward pass. A single multi-thousand-token
/// forward outlives the engine's per-forward timeout on slow (CPU) drivers
/// — the response is abandoned, the driver keeps grinding, and every queued
/// request behind it starves. Bounded chunks keep each forward well under
/// the timeout; the cue stays in the buffer so the generator's first step
/// runs the normal sample-from-pending path.
const PREFILL_CHUNK: usize = 1024;

fn build_context(model: &Model, setup: &TurnSetup) -> Result<BuiltContext, String> {
    // Resume attempt: hash the prefix up to the last assistant-produced
    // item; the previous request saved its post-generation context under
    // exactly that name. `open` forks the snapshot without consuming it
    // (Codex retries re-POST the same request and must be able to re-hit).
    let mut miss_debug = String::from("first turn (no resume point)");
    if let Some(split) = session::split_resume_point(&setup.items) {
        let canons: Vec<CanonItem> =
            setup.items[..split].iter().filter_map(session::canon).collect();
        let name = session::snapshot_name(
            &setup.cache_scope,
            setup.instructions.as_deref(),
            &setup.tool_schemas,
            canons.iter(),
        );
        match Context::open(model, &name) {
            Ok(ctx) => {
                let cached = ctx.seq_len();
                let mut suffix = Vec::new();
                replay::render_items(model, &setup.items[split..], &mut suffix)
                    .map_err(|e| e.to_string())?;
                return Ok(BuiltContext {
                    ctx,
                    prefill: suffix,
                    cached_tokens: cached,
                    debug: format!("resume hit {name} (cached {cached})"),
                    resumed_from: Some(name),
                });
            }
            Err(e) => {
                miss_debug = format!("resume miss {name}: {e}");
            }
        }
    }

    // Miss or first turn: full rebuild.
    let tokens = replay::render_full(
        model,
        setup.instructions.as_deref(),
        &setup.items,
        &setup.tool_schemas,
    )
    .map_err(|e| e.to_string())?;
    let ctx = Context::new(model).map_err(|e| e.to_string())?;
    Ok(BuiltContext {
        ctx,
        prefill: tokens,
        cached_tokens: 0,
        resumed_from: None,
        debug: miss_debug,
    })
}

/// Close the turn and save the KV snapshot under the address the *next*
/// request will hash to: incoming items + the items produced this turn.
/// On success, deletes the boundary this turn resumed from (a retry of the
/// *next* request no longer needs it; a retry of *this* one just rebuilds).
/// Returns a debug summary either way.
async fn save_snapshot(
    ctx: &mut Context,
    model: &Model,
    setup: &TurnSetup,
    built_resumed_from: Option<&str>,
    visible_text: &str,
    calls: &[(String, String, String)], // (call_id, name, arguments)
) -> String {
    let mut canons: Vec<CanonItem> = setup.items.iter().filter_map(session::canon).collect();
    if !visible_text.is_empty() {
        canons.push(CanonItem::Msg { role: "assistant", text: visible_text.to_string() });
    }
    for (call_id, name, args) in calls {
        canons.push(CanonItem::Call { call_id, name, args });
    }
    let name = session::snapshot_name(
        &setup.cache_scope,
        setup.instructions.as_deref(),
        &setup.tool_schemas,
        canons.iter(),
    );
    // The sampled stop token never enters the context (generation truncates
    // at it), so seal the assistant turn explicitly, then flush the buffer
    // through a forward pass so the snapshot holds committed KV.
    ctx.seal();
    if let Err(e) = ctx.flush().await {
        return format!("save flush failed: {e}");
    }
    match ctx.save(&name) {
        Ok(()) => {
            if let Some(old) = built_resumed_from {
                let _ = Context::delete(model, old);
            }
            format!("saved {name} (seq {})", ctx.seq_len())
        }
        Err(e) => format!("save {name} failed: {e}"),
    }
}

fn parse_setup(body_bytes: &[u8]) -> Result<(TurnSetup, bool), String> {
    let request: CreateResponseBody =
        serde_json::from_slice(body_bytes).map_err(|e| format!("Invalid JSON: {e}"))?;

    let tool_schemas = function_tool_envelopes(&request.tools);
    let items = request.input.into_items();

    // Snapshot names may not contain `/` beyond the namespace separators;
    // sanitize the client-supplied key.
    let cache_scope = request
        .prompt_cache_key
        .as_deref()
        .map(|k| {
            k.chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
                .collect::<String>()
        })
        .filter(|k| !k.is_empty())
        .unwrap_or_else(|| "global".to_string());

    Ok((
        TurnSetup {
            instructions: request.instructions.filter(|s| !s.is_empty()),
            items,
            tool_schemas,
            max_tokens: request.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            temperature: request.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            top_p: request.top_p.unwrap_or(DEFAULT_TOP_P),
            cache_scope,
        },
        request.stream,
    ))
}

pub async fn handle_responses(body_bytes: Vec<u8>, responder: Responder) -> Finished {
    let (setup, stream) = match parse_setup(&body_bytes) {
        Ok(v) => v,
        Err(msg) => return error_response(responder, 400, "invalid_request", &msg).await,
    };

    if stream {
        handle_streaming(setup, responder).await
    } else {
        handle_non_streaming(setup, responder).await
    }
}

// ─── Streaming ─────────────────────────────────────────────────────────────

async fn handle_streaming(setup: TurnSetup, responder: Responder) -> Finished {
    let uniq = uniq_fragment();
    let response_id = format!("resp_{uniq}");
    let message_id = format!("msg_{uniq}");

    let sse_response = Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(BodyForthcoming)
        .unwrap();
    let mut body = responder.start_response(sse_response);
    let mut emitter = StreamEmitter::new();

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

    let models = runtime::models();
    let model_name = models.first().cloned().unwrap_or_else(|| "unknown".to_string());
    let mut response = ResponseResource::new(response_id.clone(), model_name.clone());

    emit!(emitter.response_created(&response));
    response.status = ResponseStatus::InProgress;
    emit!(emitter.response_in_progress(&response));

    macro_rules! fail {
        ($code:expr, $msg:expr) => {{
            response.status = ResponseStatus::Failed;
            response.error = Some(ErrorPayload {
                error_type: "server_error".to_string(),
                code: Some($code.to_string()),
                message: $msg,
                param: None,
            });
            emit!(emitter.response_failed(&response));
            emit!(StreamEmitter::done());
            return Finished::finish(body, Ok(()), None);
        }};
    }

    let model = match models
        .first()
        .ok_or_else(|| "no models available".to_string())
        .and_then(|n| Model::load(n).map_err(|e| e.to_string()))
    {
        Ok(m) => m,
        Err(e) => fail!("model_load_failed", e),
    };

    let BuiltContext { mut ctx, prefill, cached_tokens, resumed_from, debug: build_debug } =
        match build_context(&model, &setup) {
            Ok(b) => b,
            Err(e) => fail!("context_build_failed", e),
        };

    // Chunked prefill, one keepalive event per chunk: `response.in_progress`
    // is in Codex's ignored-event set but still resets its 5-minute SSE
    // idle timer — a multi-minute CPU prefill would otherwise kill the turn.
    for chunk in prefill.chunks(PREFILL_CHUNK) {
        ctx.append(chunk);
        if let Err(e) = ctx.flush().await {
            fail!("prefill_failed", e.to_string());
        }
        emit!(emitter.response_in_progress(&response));
    }
    ctx.cue();
    let prompt_tokens = ctx.seq_len() + ctx.buffer().len() as u32;

    // ── Generation loop ──
    let stop_ids = chat::stop_tokens(&model);
    let mut g = ctx
        .generate(setup.sampler())
        .max_tokens(setup.max_tokens)
        .stop(&stop_ids);
    if let Some(matcher) = tools::native_matcher(&model, &setup.tool_schemas) {
        g = g.constrain(GrammarConstraint::new(matcher));
    }

    let mut tool_decoder =
        (!setup.tool_schemas.is_empty()).then(|| tools::Decoder::new(&model));
    let mut text_decoder = chat::Decoder::new(&model);
    let mut filter = VisibleFilter::new();

    let mut visible_text = String::new();
    let mut calls: Vec<(String, String, String)> = Vec::new();
    let mut output_items: Vec<OutputItem> = Vec::new();
    let mut msg_index: Option<u32> = None;
    let mut next_index: u32 = 0;
    let mut n_generated: usize = 0;
    let mut gen_error: Option<String> = None;

    // Lazily open the streaming message item on first visible text.
    macro_rules! open_message {
        () => {{
            if msg_index.is_none() {
                let idx = next_index;
                next_index += 1;
                msg_index = Some(idx);
                let item = OutputItem::Message(OutputMessage {
                    id: message_id.clone(),
                    role: Role::Assistant,
                    status: ItemStatus::InProgress,
                    content: vec![],
                });
                emit!(emitter.output_item_added(idx, &item));
                let part =
                    OutputContentPart::OutputText { text: String::new(), annotations: vec![] };
                emit!(emitter.content_part_added(&message_id, idx, 0, &part));
            }
            msg_index.unwrap()
        }};
    }

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
            n_generated += 1;

            if let Some(dec) = tool_decoder.as_mut() {
                if let Ok(tools::Event::Call(name, args)) = dec.feed(&[t]) {
                    let call_id = format!("call_{uniq}_{}", calls.len());
                    let item = OutputItem::FunctionCall(OutputFunctionCall {
                        id: format!("fc_{uniq}_{}", calls.len()),
                        call_id: call_id.clone(),
                        name: name.clone(),
                        arguments: args.clone(),
                        status: ItemStatus::Completed,
                    });
                    emit!(emitter.output_item_added(next_index, &item));
                    emit!(emitter.output_item_done(next_index, &item));
                    output_items.push(item);
                    next_index += 1;
                    calls.push((call_id, name, args));
                }
            }

            if stop_ids.contains(&t) {
                break 'outer;
            }

            match text_decoder.feed(&[t]) {
                Ok(chat::Event::Delta(s)) => {
                    let v = filter.feed(&s);
                    if !v.is_empty() {
                        let idx = open_message!();
                        emit!(emitter.output_text_delta(&message_id, idx, 0, &v));
                        visible_text.push_str(&v);
                    }
                }
                Ok(chat::Event::Done(_)) => break 'outer,
                _ => {}
            }
        }
    }

    if let Some(e) = gen_error {
        fail!("generation_failed", e);
    }

    // Flush any held-back filter tail.
    let tail = filter.finish();
    if !tail.is_empty() {
        let idx = open_message!();
        emit!(emitter.output_text_delta(&message_id, idx, 0, &tail));
        visible_text.push_str(&tail);
    }
    let visible_text = visible_text.trim_end().to_string();

    let hit_max = n_generated >= setup.max_tokens;
    let item_status = if hit_max { ItemStatus::Incomplete } else { ItemStatus::Completed };

    // Close the message item — opening an empty one first if the model
    // produced nothing at all (Codex needs at least one output item).
    if msg_index.is_none() && calls.is_empty() {
        let _ = open_message!();
    }
    if let Some(idx) = msg_index {
        let part = OutputContentPart::OutputText {
            text: visible_text.clone(),
            annotations: vec![],
        };
        emit!(emitter.output_text_done(&message_id, idx, 0, &visible_text));
        emit!(emitter.content_part_done(&message_id, idx, 0, &part));
        let item = OutputItem::Message(OutputMessage {
            id: message_id.clone(),
            role: Role::Assistant,
            status: item_status.clone(),
            content: vec![part],
        });
        emit!(emitter.output_item_done(idx, &item));
        output_items.push(item);
    }

    // Save the KV snapshot *before* signalling completion: Codex fires the
    // follow-up request the moment `response.completed` lands, and the
    // snapshot must already exist for the resume to hit. Failures are
    // non-fatal (next request pays a full rebuild) and land in `debug`.
    let save_debug = save_snapshot(
        &mut ctx,
        &model,
        &setup,
        resumed_from.as_deref(),
        &visible_text,
        &calls,
    )
    .await;
    response.debug = Some(format!("{build_debug}; {save_debug}"));

    response.status =
        if hit_max { ResponseStatus::Incomplete } else { ResponseStatus::Completed };
    if hit_max {
        response.incomplete_details =
            Some(IncompleteDetails { reason: "max_output_tokens".to_string() });
    }
    response.output = output_items;
    response.completed_at = Some(now_unix_secs());
    response.usage = Some(Usage {
        input_tokens: prompt_tokens,
        input_tokens_details: InputTokensDetails { cached_tokens },
        output_tokens: n_generated as u32,
        output_tokens_details: OutputTokensDetails { reasoning_tokens: 0 },
        total_tokens: prompt_tokens + n_generated as u32,
    });
    emit!(emitter.response_completed(&response));
    emit!(StreamEmitter::done());

    Finished::finish(body, Ok(()), None)
}

// ─── Non-streaming ─────────────────────────────────────────────────────────

async fn handle_non_streaming(setup: TurnSetup, responder: Responder) -> Finished {
    let uniq = uniq_fragment();
    let response_id = format!("resp_{uniq}");
    let message_id = format!("msg_{uniq}");

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

    let stop_ids = chat::stop_tokens(&model);
    let mut g = ctx
        .generate(setup.sampler())
        .max_tokens(setup.max_tokens)
        .stop(&stop_ids);
    if let Some(matcher) = tools::native_matcher(&model, &setup.tool_schemas) {
        g = g.constrain(GrammarConstraint::new(matcher));
    }

    let mut tool_decoder =
        (!setup.tool_schemas.is_empty()).then(|| tools::Decoder::new(&model));
    let mut text_decoder = chat::Decoder::new(&model);
    let mut filter = VisibleFilter::new();
    let mut visible_text = String::new();
    let mut calls: Vec<(String, String, String)> = Vec::new();
    let mut n_generated: usize = 0;

    'outer: loop {
        let step = match g.next() {
            Ok(Some(s)) => s,
            Ok(None) => break,
            Err(e) => {
                return error_response(responder, 500, "server_error", &e.to_string()).await
            }
        };
        let out = match step.execute().await {
            Ok(o) => o,
            Err(e) => {
                return error_response(responder, 500, "server_error", &e.to_string()).await
            }
        };
        for &t in &out.tokens {
            n_generated += 1;
            if let Some(dec) = tool_decoder.as_mut() {
                if let Ok(tools::Event::Call(name, args)) = dec.feed(&[t]) {
                    let call_id = format!("call_{uniq}_{}", calls.len());
                    calls.push((call_id, name, args));
                }
            }
            if stop_ids.contains(&t) {
                break 'outer;
            }
            match text_decoder.feed(&[t]) {
                Ok(chat::Event::Delta(s)) => visible_text.push_str(&filter.feed(&s)),
                Ok(chat::Event::Done(_)) => break 'outer,
                _ => {}
            }
        }
    }
    visible_text.push_str(&filter.finish());
    let visible_text = visible_text.trim_end().to_string();

    let hit_max = n_generated >= setup.max_tokens;
    let item_status = if hit_max { ItemStatus::Incomplete } else { ItemStatus::Completed };

    let mut output: Vec<OutputItem> = Vec::new();
    for (i, (call_id, name, args)) in calls.iter().enumerate() {
        output.push(OutputItem::FunctionCall(OutputFunctionCall {
            id: format!("fc_{uniq}_{i}"),
            call_id: call_id.clone(),
            name: name.clone(),
            arguments: args.clone(),
            status: ItemStatus::Completed,
        }));
    }
    if !visible_text.is_empty() || output.is_empty() {
        output.push(OutputItem::Message(OutputMessage {
            id: message_id,
            role: Role::Assistant,
            status: item_status,
            content: vec![OutputContentPart::OutputText {
                text: visible_text.clone(),
                annotations: vec![],
            }],
        }));
    }

    let save_debug = save_snapshot(
        &mut ctx,
        &model,
        &setup,
        resumed_from.as_deref(),
        &visible_text,
        &calls,
    )
    .await;

    let now = now_unix_secs();
    let response = ResponseResource {
        id: response_id,
        object: "response".to_string(),
        created_at: now,
        completed_at: Some(now),
        status: if hit_max { ResponseStatus::Incomplete } else { ResponseStatus::Completed },
        incomplete_details: hit_max
            .then(|| IncompleteDetails { reason: "max_output_tokens".to_string() }),
        model: model_name,
        output,
        error: None,
        usage: Some(Usage {
            input_tokens: prompt_tokens,
            input_tokens_details: InputTokensDetails { cached_tokens },
            output_tokens: n_generated as u32,
            output_tokens_details: OutputTokensDetails { reasoning_tokens: 0 },
            total_tokens: prompt_tokens + n_generated as u32,
        }),
        debug: Some(format!("{build_debug}; {save_debug}")),
    };

    let json = serde_json::to_string(&response).unwrap_or_default();
    let http_response = Response::builder()
        .header("Content-Type", "application/json")
        .body(json.into_body())
        .unwrap();
    responder.respond(http_response).await
}

// ─── Errors ────────────────────────────────────────────────────────────────

async fn error_response(
    responder: Responder,
    status_code: u16,
    error_type: &str,
    message: &str,
) -> Finished {
    let error = serde_json::json!({
        "error": {
            "type": error_type,
            "message": message,
            "code": null,
            "param": null,
        }
    });
    let response = Response::builder()
        .status(status_code)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();
    responder.respond(response).await
}
