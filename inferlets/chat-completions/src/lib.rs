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
//! ## Prefix cache
//!
//! One request per process does NOT mean one prefill per turn. [`apc`] parks
//! the rendered history's KV in the engine's own index, which outlives the
//! instance, so the next request resumes at the deepest cut it can find and
//! prefills only the delta. `usage.prompt_tokens_details.cached_tokens` reports
//! the depth actually reached, not the depth planned.
//!
//! ## Open seams (deliberately out of scope this milestone)
//!
//! - **Grammar-constrained tool calls**: no grammar-forced phase-2 call, on
//!   the old handler's evidence — constraining suppressed reasoning text and
//!   collapsed t=0 trajectories, and constrained decoding traps the guest on
//!   drivers without grammar support. Re-add behind a capability probe via
//!   `tools::format`/`tools::create_matcher` when that lands.
//! - **Qwen3-Coder XML dialect** (`ToolFormat::Coder`, `<function=…>`
//!   parsing): salvage seam marked in `turn::TurnState::salvage`; the
//!   decoder/template halves live model-side.

mod apc;
mod engine;
mod turn;

use inferlet::{chat, model, runtime, session, tools};
use pie_openai_serving::error::{INVALID_REQUEST_ERROR, SERVER_ERROR, error_body, parse_request};
use pie_openai_serving::streaming::ChunkMeta;
use pie_openai_serving::{RenderOp, cut_leading_reasoning, plan_render, sanitize_messages};
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

/// Map one engine-free render op onto the WIT template surface.
///
/// `Cue` renders through `chat::cue_no_think` — decision D1: this milestone
/// always serves the no-think channel (matches the token-exact parity
/// verdict against HF `enable_thinking=False`); a thinking channel would
/// branch here on `req.no_think()` and route reasoning to
/// `reasoning_content`, which nothing client-side needs yet.
fn render_one(op: &RenderOp, out: &mut Vec<u32>) -> Result<(), String> {
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
    Ok(())
}

fn render_ops(ops: &[RenderOp]) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for op in ops {
        render_one(op, &mut out)?;
    }
    Ok(out)
}

/// Render the history ops, recording the token length after each one plus
/// stride points inside the long ones.
///
/// The boundaries are the safe resume points: a unit edge lands between two
/// render ops and a stride point lands inside one, and BOTH are literal token
/// prefixes of this render by construction — the address is computed over the
/// tokens, so a cut cannot name a prefix the render does not have. Ascending
/// and duplicate-free, which `prefix_addresses` requires.
///
/// Identical to `opencode-session`'s `render_history`, deliberately: the two
/// arms must cut the same conversation the same way, or an A/B measures the cut
/// policy instead of where the KV lives.
fn render_history(ops: &[RenderOp]) -> Result<(Vec<u32>, Vec<u32>), String> {
    let mut out = Vec::new();
    let mut bounds: Vec<u32> = Vec::with_capacity(ops.len() * 4);
    for op in ops {
        let before = out.len() as u32;
        render_one(op, &mut out)?;
        let after = out.len() as u32;
        let mut at = before.next_multiple_of(apc::BOUNDARY_STRIDE);
        while at < after {
            if at > before {
                bounds.push(at);
            }
            at += apc::BOUNDARY_STRIDE;
        }
        if bounds.last() != Some(&after) {
            bounds.push(after);
        }
    }
    Ok((out, bounds))
}

/// The rendered prompt, split where the prefix cache can address it.
struct Prompt {
    /// `history ‖ cue` — what actually gets prefilled.
    tokens: Vec<u32>,
    /// Where the history ends. Nothing past this is addressable: replaying this
    /// turn as history renders no cue at all, so KV covering the cue sits under
    /// an address no future render produces.
    history_len: u32,
    /// Ascending cut candidates within the history.
    boundaries: Vec<u32>,
}

/// Sanitize → plan → render the full conversation to prompt tokens, split at
/// the trailing cue and marked with cut candidates for [`apc`].
fn build_prompt(
    req: &mut pie_openai_serving::ChatCompletionRequest,
) -> Result<Prompt, BuildError> {
    let specials: Vec<String> = model::special_tokens()
        .into_iter()
        .filter_map(|t| String::from_utf8(t.bytes).ok())
        .filter(|s| !s.is_empty())
        .collect();
    sanitize_messages(&mut req.messages, &specials);

    let ops = plan_render(req).map_err(|e| BuildError::Invalid(e.to_string()))?;
    // Split the plan at its trailing `Cue`: the history is what gets addressed,
    // the cue is generation scaffolding and must not be.
    let (history_ops, cue_ops) = match ops.split_last() {
        Some((RenderOp::Cue, head)) => (head, &ops[ops.len() - 1..]),
        _ => return Err(BuildError::Fault("render plan did not end with a cue".into())),
    };
    let (mut tokens, boundaries) = render_history(history_ops).map_err(BuildError::Fault)?;
    let history_len = tokens.len() as u32;
    tokens.extend(render_ops(cue_ops).map_err(BuildError::Fault)?);
    Ok(Prompt {
        tokens,
        history_len,
        boundaries,
    })
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
    let prompt_tokens = prompt.tokens.len() as u32;

    // ── Prefix cache. Both halves are planned here, before the envelope is
    // committed, because both are pure token arithmetic plus index lookups —
    // no fire, nothing that can fail the turn. `resume` is `None` on the first
    // request of a conversation and on any miss, and the engine then runs
    // exactly the cold path.
    let plan = apc::Plan::build(
        &model::name(),
        &prompt.tokens[..prompt.history_len as usize],
        &prompt.boundaries,
        inferlet::ptir::attention::prelude::kv_page_size(),
    );
    let resume = plan.resume();
    let publish = plan.publish_set();

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
    // Kept for the [apc] log line below, which `uniq` is moved into TurnState before.
    let tag = uniq.clone();
    let meta = ChunkMeta {
        id: format!("chatcmpl-{uniq}"),
        model: model::name(),
        created: now_unix_secs(),
    };
    let mut state = TurnState::new(meta, streaming, uniq, stop_ids, stop_strings, has_tools,
        &pie_openai_serving::types::tool_schema_envelopes(&req.tools));

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
    let run = engine::generate(
        &prompt.tokens,
        &engine::GenConfig { temperature, top_p, max_tokens },
        engine::Apc { resume, publish },
        |t| state.on_token(t),
    )
    .await;
    let gen_error = run.error;
    if let Some(e) = &gen_error {
        eprintln!("[chat-completions] generation degraded to length-finish: {e}");
    }
    // Reuse FRACTION, not a hit flag: a resume that lands on a shallow cut and
    // re-prefills most of the history still reports a hit, and that is exactly
    // how a prefix-cache regression hides behind a green check.
    //
    // Formatted into ONE string and emitted with ONE write. `eprintln!` reaches
    // the host one format ARGUMENT at a time, each logged as its own record and
    // interleaved with other threads' — and, under opencode, with the other
    // concurrent request's. Reading "reuse " and taking the next log line gives
    // whichever thread logged next, so the first monitor built on this line
    // reported a reuse percentage with no number in it. The `uniq` fragment
    // tags the line because two requests really are in flight at once.
    let line = format!(
        "[apc] {tag} reuse {:.1}% ({}/{} prompt tokens) from {} ladder cut(s); parked {}\n",
        100.0 * apc::reuse_fraction(run.cached_tokens, prompt_tokens),
        run.cached_tokens,
        prompt_tokens,
        plan.len(),
        run.published,
    );
    eprint!("{line}");

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

        // The prefix for the NEXT request was parked back in `engine`, right
        // after prefill — it is pure history, so it does not wait on this turn's
        // outcome, and the pipeline it needs to order on is closed by the time
        // decode ends.

        let finish = state.meta.finish_chunk(finish_reason);
        state.emit(&finish);
        if include_usage {
            // The depth the engine ACTUALLY resumed at, which is 0 when it
            // refused a resume and rebuilt. Always ≤ prompt_tokens: cuts are
            // bounded by the history, and the history is a prefix of the render.
            let usage = state.meta.usage_chunk(
                prompt_tokens,
                state.generated.len() as u32,
                run.cached_tokens,
            );
            state.emit(&usage);
        }
        // The gateway appends `data: [DONE]` on clean Eos.
    } else {
        // Non-streaming (curl/debug/acceptance path): exactly one body
        // message. The old handler trimmed trailing whitespace here (and
        // only here — streamed bytes are already on the wire). The leading
        // reasoning cut rides on the same "nothing is sent yet" licence: a
        // model that closes a think block it never opened (Qwen3.6) leaves
        // its reasoning in front of the answer, and this is the last point
        // at which that is still removable.
        state.visible_text =
            cut_leading_reasoning(state.visible_text.trim_end()).to_string();
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
            run.cached_tokens,
        ));
    }

    // The return value is instrumentation, not wire data (the envelope
    // carries the response); a summary aids `pie run` debugging.
    Ok(format!(
        "finish={finish_reason} prompt={prompt_tokens} cached={} generated={} calls={}{}",
        run.cached_tokens,
        state.generated.len(),
        state.calls.len(),
        gen_error.map(|e| format!(" degraded: {e}")).unwrap_or_default(),
    ))
}
