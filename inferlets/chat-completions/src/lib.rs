//! OpenAI-compatible `/v1/chat/completions` serving inferlet (opencode-facing).
//!
//! One request per process launch: the gateway (`gateway/src/ingress/openai.rs`)
//! hands the raw OpenAI request JSON in as the launch input and re-frames what
//! this inferlet emits. Every `session::send` payload is one JSON document on
//! the gateway⇄inferlet envelope (authoritative contract in the gateway
//! module docs):
//!
//! - first message: `{"status": <u16>}` — the HTTP status to respond with;
//! - streaming (`"stream": true`): each subsequent message is one ready-made
//!   `chat.completion.chunk` document (the gateway adds `data:` framing and
//!   `[DONE]`; keepalive `: ping` comments are the gateway's job too);
//! - non-streaming: exactly one subsequent message — the full
//!   `chat.completion` (or OpenAI error) body.
//!
//! Wire discipline (opencode AUDIT + qwen-code invariants):
//! - malformed input → `{"status":400}` + OpenAI error body, NEVER a process
//!   error (opencode retries 5xx without bound — 500 is reserved for genuine
//!   server faults);
//! - never a context-length error: overflow/generation faults degrade to
//!   `finish_reason:"length"` with whatever streamed;
//! - first delta carries `role:"assistant"`; tool-call deltas are atomic
//!   (index+id+name+arguments); the turn ends with a `finish_reason` chunk
//!   (`stop`/`length`/`tool_calls`, never `error_finish`), then a usage chunk
//!   when `stream_options.include_usage`.
//!
//! Module map: [`engine`] — PTIR prefill/decode core (ported from the
//! `tests/inferlets/chat-completion` reference); [`turn`] — per-token
//! orchestration (tool decode, fencing, stop policy, chunk emission). All
//! wire JSON comes from `pie-openai-serving`'s builders.
//!
//! ## Open seams (deliberately out of scope this milestone)
//!
//! - **KV snapshot sessions**: content-addressed resume via
//!   `pie_openai_serving::session` (canon/snapshot_address/split_resume_point
//!   — already ported and tested) + `working-set update-index/from-index`.
//!   Attach in [`build_prompt`] (resume = render only the suffix past the
//!   split point) and after generation (save under the next turn's address).
//!   `usage.prompt_tokens_details.cached_tokens` stays 0 until then.
//! - **Grammar-constrained tool calls**: no grammar-forced phase-2 call, on
//!   the old handler's evidence — constraining suppressed reasoning text and
//!   collapsed t=0 trajectories, and constrained decoding traps the guest on
//!   drivers without grammar support. Re-add behind a capability probe via
//!   `tools::format`/`tools::create_matcher` when that lands.
//! - **Qwen3-Coder XML dialect** (`ToolFormat::Coder`, `<function=…>`
//!   parsing): salvage seam marked in `turn::TurnState::salvage`; the
//!   decoder/template halves live model-side.

mod engine;
mod turn;

use inferlet::{chat, model, runtime, session, tools};
use pie_openai_serving::error::{INVALID_REQUEST_ERROR, SERVER_ERROR, error_body, parse_request};
use pie_openai_serving::streaming::ChunkMeta;
use pie_openai_serving::{RenderOp, plan_render, sanitize_messages};
use serde_json::{Value, json};
use turn::TurnState;

/// Defaults when the client sends none. opencode always sends `max_tokens`
/// but omits `temperature`/`top_p` for unknown model ids (AUDIT §1a);
/// sampling defaults follow the reference inferlet
/// (`tests/inferlets/chat-completion`).
const DEFAULT_MAX_TOKENS: usize = 4096;
const DEFAULT_TEMPERATURE: f32 = 0.6;
const DEFAULT_TOP_P: f32 = 0.95;

fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Per-request unique id fragment for tool-call/completion ids. One fresh
/// WASM instance per request, so uniqueness must come from the runtime:
/// `system::instance-id()` is unique per instantiation.
fn uniq_fragment() -> String {
    let frag: String = runtime::instance_id()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(12)
        .collect();
    if frag.is_empty() { "local".to_string() } else { frag }
}

fn send_json(v: &Value) {
    session::send(&v.to_string());
}

/// Envelope rejection: `{"status": s}` then one OpenAI error body. Used
/// before any streaming has begun (the gateway answers plain JSON, not SSE).
fn reject(status: u16, error_type: &str, message: &str) -> inferlet::Result<String> {
    send_json(&json!({ "status": status }));
    send_json(&error_body(error_type, message));
    Ok(String::new())
}

/// Map the engine-free render plan 1:1 onto the WIT template surface.
/// `Cue` renders through `chat::cue_no_think` — decision D1: this milestone
/// always serves the no-think channel (matches the token-exact parity
/// verdict against HF `enable_thinking=False`); a thinking channel would
/// branch here on `req.no_think()` and route reasoning to
/// `reasoning_content`, which nothing client-side needs yet.
fn render_ops(ops: &[RenderOp]) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for op in ops {
        match op {
            RenderOp::EquipAfterSystem { system, tools: schemas } => {
                out.extend(tools::equip_after_system(system.as_deref(), schemas)?);
            }
            RenderOp::User(t) => out.extend(chat::user(t)),
            RenderOp::Assistant(t) => out.extend(chat::assistant(t)),
            RenderOp::AssistantWithToolCalls { content, calls } => {
                let wit_calls: Vec<tools::ToolCall> = calls
                    .iter()
                    .map(|(name, args)| tools::ToolCall {
                        name: name.clone(),
                        arguments_json: args.clone(),
                    })
                    .collect();
                out.extend(tools::assistant_with_tool_calls(content.as_deref(), &wit_calls));
            }
            RenderOp::AnswerBatch(batch) => out.extend(tools::answer_batch(batch)),
            RenderOp::Cue => out.extend(chat::cue_no_think()),
        }
    }
    Ok(out)
}

/// Sanitize → plan → render the full conversation to prompt tokens.
///
/// SEAM (KV snapshot sessions): the resume path replaces this whole-history
/// render — `split_resume_point` on the sanitized messages, hash the prefix
/// with `snapshot_address`, `working-set from-index` on hit, then render
/// only the suffix via `plan_render_messages`. Response/save unification:
/// sanitization runs before BOTH rendering and canonicalization.
fn build_prompt(req: &mut pie_openai_serving::ChatCompletionRequest) -> Result<Vec<u32>, BuildError> {
    let specials: Vec<String> = model::special_tokens()
        .into_iter()
        .filter_map(|t| String::from_utf8(t.bytes).ok())
        .filter(|s| !s.is_empty())
        .collect();
    sanitize_messages(&mut req.messages, &specials);

    let ops = plan_render(req).map_err(|e| BuildError::Invalid(e.to_string()))?;
    render_ops(&ops).map_err(BuildError::Fault)
}

enum BuildError {
    /// Client's fault → 400 (misplaced system, unsupported role).
    Invalid(String),
    /// Host/template fault → 500.
    Fault(String),
}

#[inferlet::main]
async fn main(input: String) -> inferlet::Result<String> {
    // ── Parse + validate. The only 400s: bad JSON, empty messages (here),
    // misplaced system / unsupported role (from the render plan below).
    // Unknown fields are ignored at the serde layer, never rejected.
    let mut req = match parse_request(input.as_bytes()) {
        Ok(r) => r,
        Err(msg) => return reject(400, INVALID_REQUEST_ERROR, &msg),
    };

    let streaming = req.stream;
    let include_usage = req.include_usage();
    let temperature = req.temperature.filter(|t| t.is_finite()).unwrap_or(DEFAULT_TEMPERATURE);
    let top_p = req
        .top_p
        .filter(|p| p.is_finite() && *p > 0.0 && *p <= 1.0)
        .unwrap_or(DEFAULT_TOP_P);
    // Bounded only by the request (and the driver's real KV capacity, which
    // surfaces mid-decode and degrades to "length" — never a 4xx).
    let max_tokens = req.effective_max_tokens(DEFAULT_MAX_TOKENS);
    let has_tools = !req.tools.is_empty();
    let stop_strings = req.stop_strings();

    // ── Render (fast, host-template calls) before committing a status, so a
    // genuine template fault can still answer a clean 500.
    let prompt = match build_prompt(&mut req) {
        Ok(p) => p,
        Err(BuildError::Invalid(msg)) => return reject(400, INVALID_REQUEST_ERROR, &msg),
        Err(BuildError::Fault(msg)) => {
            eprintln!("[chat-completions] render fault: {msg}");
            return reject(500, SERVER_ERROR, "internal error while rendering the prompt");
        }
    };
    let prompt_tokens = prompt.len() as u32;

    // Stop set: the model's chat stop tokens, plus the turn-START marker —
    // at t=0 a looping model starts simulating the next turn instead of
    // stopping, and leaked "<|im_start|>" round-trips through message
    // content into fake turn boundaries next request (old handler, verified
    // on openhands traj job 18825434).
    let mut stop_ids = chat::stop_tokens();
    let turn_start = model::encode("<|im_start|>");
    if turn_start.len() == 1 && !stop_ids.contains(&turn_start[0]) {
        stop_ids.push(turn_start[0]);
    }

    let uniq = uniq_fragment();
    let meta = ChunkMeta {
        id: format!("chatcmpl-{uniq}"),
        model: model::name(),
        created: now_unix_secs(),
    };
    let mut state = TurnState::new(meta, streaming, uniq, stop_ids, stop_strings, has_tools);

    // ── Commit the envelope. From here every failure must be shaped as a
    // well-formed turn (finish_reason "length"), never a fault — a dead
    // stream burns client retries; a degraded turn is handled gracefully.
    if streaming {
        send_json(&json!({ "status": 200 }));
        // Role chunk first: announces the assistant turn and doubles as the
        // first stream bytes before a potentially long prefill.
        let role = state.meta.role_chunk();
        state.emit(&role);
    }

    // ── Generate. Per-token policy (tool decode → stop set → filtered
    // content deltas → stop strings) lives in `TurnState::on_token`.
    let gen_error = engine::generate(
        &prompt,
        &engine::GenConfig { temperature, top_p, max_tokens },
        |t| state.on_token(t),
    )
    .await;
    if let Some(e) = &gen_error {
        eprintln!("[chat-completions] generation degraded to length-finish: {e}");
    }

    // Flush the filter's held-back tail, then salvage tool calls the native
    // decoder missed (fenced JSON in visible text; unclosed hermes blocks in
    // the raw generation, which the filter swallowed).
    state.flush_tail();
    let raw_text = model::decode(&state.generated).unwrap_or_default();
    state.salvage(has_tools, &raw_text);

    let hit_max = state.generated.len() >= max_tokens;
    let finish_reason = state.finish_reason(hit_max, gen_error.is_some());

    if streaming {
        // A text turn that streamed nothing visible still delivers content
        // (whatever the model actually decoded, even whitespace-adjacent —
        // see `final_content`).
        if !state.emitted_visible && state.calls.is_empty() {
            let fallback = state.final_content(&raw_text);
            let chunk = state.meta.content_delta(&fallback);
            state.emit(&chunk);
        }

        // SEAM (KV snapshot sessions): save the post-generation working set
        // under the next turn's address HERE — before the finish chunk, so
        // the snapshot exists when the client fires its follow-up request.

        let finish = state.meta.finish_chunk(finish_reason);
        state.emit(&finish);
        if include_usage {
            // cached_tokens: 0 until the session seam lands (then: resumed
            // prefix depth; must stay ≤ prompt_tokens).
            let usage =
                state.meta.usage_chunk(prompt_tokens, state.generated.len() as u32, 0);
            state.emit(&usage);
        }
        // The gateway appends `data: [DONE]` on clean Eos.
    } else {
        // Non-streaming (curl/debug/acceptance path): exactly one body
        // message. The old handler trimmed trailing whitespace here (and
        // only here — streamed bytes are already on the wire).
        state.visible_text = state.visible_text.trim_end().to_string();
        let content = state.final_content(&raw_text);
        let calls: Vec<(String, String, String)> = state
            .calls
            .iter()
            .map(|c| (c.id.clone(), c.name.clone(), c.arguments.clone()))
            .collect();
        send_json(&json!({ "status": 200 }));
        send_json(&state.meta.completion_response(
            &content,
            &calls,
            finish_reason,
            prompt_tokens,
            state.generated.len() as u32,
            0,
        ));
    }

    // The return value is instrumentation, not wire data (the envelope
    // carries the response); a summary aids `pie run` debugging.
    Ok(format!(
        "finish={finish_reason} prompt={prompt_tokens} generated={} calls={}{}",
        state.generated.len(),
        state.calls.len(),
        gen_error.map(|e| format!(" degraded: {e}")).unwrap_or_default(),
    ))
}
