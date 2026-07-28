//! openhands-coder-session — stateful native-tool-calling inferlet for the
//! PieLLM integration (Phase 1 of
//! `integrations/openhands/docs/OPENHANDS_CODER_SESSION_DESIGN.md`).
//!
//! Same request/response contract as `openhands-completion` (which stays as
//! the stateless fallback and Phase-0 baseline), plus a per-conversation
//! session protocol:
//!
//!   Input adds:
//!     session_id:        Option<String>  (enables session mode)
//!     session_action:    Option<String>  ("delete" → drop the saved context and exit)
//!     kv_verify:         bool            (Phase 2 fidelity assertions)
//!     use_grammar:       bool            (default true; disable for parity with an
//!                                          unconstrained vLLM baseline)
//!
//!   Output adds:
//!     session: { id, mode, len, hash, prefill_tokens }
//!       mode is "fresh" | "extended" | "rebuilt" | "stateless" | "deleted".
//!       `len`/`hash` describe this call's prompt render; the host echoes them
//!       back on the next call. `prefill_tokens` counts only the prompt tokens
//!       actually computed this call (vs `prompt_tokens`, which stays the full
//!       prompt length for OpenHands usage-accounting parity with the baseline).
//!
//! How the session works (content-addressed prefix cache — see
//! [`prefix_cache`]):
//!
//! 1. Render the **full** message list to tokens exactly as the stateless
//!    inferlet would — semantics unchanged; every token the model sees is
//!    byte-identical to the stateless render. The render excludes the trailing
//!    generation cue so the snapshot ends on a message boundary, and it also
//!    reports the token length at each safe render-unit boundary.
//! 2. Every call saves its full render under `hash(tokens)`. History is
//!    append-only, so the previous turn's full render is a literal token-prefix
//!    of this render and re-appears as one of the reported boundaries. Scan the
//!    boundaries longest-first, `Context::open`-ing `hash(full[..L])`; the most
//!    recent saved boundary hits and we append only the suffix. Any miss
//!    (first call, snapshot lost, condenser rewrote history) falls through to a
//!    clean rebuild — always semantically safe, just slower.
//! 3. The lookup key is self-derived from token content, not host-echoed
//!    hints, and names only *full host renders* (never a predicted assistant
//!    reply), so the save-side and lookup-side names match byte-for-byte —
//!    argument re-serialization on the host can never cause a miss.
//!
//! The snapshot is a prompt-only checkpoint: each new call re-prefills the
//! previous assistant turn plus the new tool results (O(delta)), not the full
//! history (O(conversation)). Because names are content-addressed, distinct
//! boundaries coexist, so retry / branch / truncate re-hit their still-valid
//! earlier boundary and a delegated sub-agent that shares a task prefix hits
//! the parent's boundary with no explicit fork protocol.

use inferlet::{
    Context, FutureStringExt, Result, chat, model::Model, runtime, sample::Sampler, session, tools,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

mod prefix_cache;

/// Milliseconds elapsed since `t`, as f64.
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Cache `compat` namespace component — the host bumps this (via a future
/// input field) to invalidate all snapshots on app-schema drift. Empty for
/// now; `snapshot_name` maps empty → "0".
const CACHE_COMPAT: &str = "";
use serde_json::Value;

// ─── Input ─────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    messages: Vec<Message>,

    #[serde(default)]
    tools: Vec<ToolSpec>,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_temperature")]
    temperature: f32,

    #[serde(default = "default_top_p")]
    top_p: f32,

    #[serde(default)]
    stop: Vec<String>,

    /// Informational only — runtime picks the first available model.
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,

    // ── Session protocol ──────────────────────────────────────────────
    //
    // `session_id` alone is the cache namespace: the prefix cache self-keys
    // from token content. The legacy host-coordination hints that used to sit
    // here (`session_prev_len` / `session_prev_hash` and the `session_fork_*`
    // parent pointers) are gone — the harness stopped sending them when
    // self-keying landed, and serde ignores unknown fields, so an older caller
    // that still emits them keeps deserializing fine.
    #[serde(default)]
    session_id: Option<String>,

    #[serde(default)]
    session_action: Option<String>,

    #[serde(default)]
    kv_verify: bool,

    #[serde(default = "default_true")]
    use_grammar: bool,
}

fn default_max_tokens() -> usize { 2048 }

/// Token budget for the phase-2 forced tool call — one call plus slack
/// (multi-line editor arguments can run long).
const FORCED_CALL_MAX_TOKENS: usize = 1024;
fn default_temperature() -> f32 { 0.0 }
fn default_top_p() -> f32 { 0.95 }
fn default_true() -> bool { true }

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallIn>>,
    /// Present on "tool" role turns; unused by Qwen's template (the reply
    /// wrapper carries no name) but accepted for OpenAI-shape compatibility.
    #[serde(default)]
    #[allow(dead_code)]
    tool_call_id: Option<String>,
}

#[derive(Deserialize)]
struct ToolCallIn {
    function: ToolCallFunction,
}

#[derive(Deserialize)]
struct ToolCallFunction {
    name: String,
    /// JSON-encoded arguments object, OpenAI style.
    arguments: String,
}

#[derive(Deserialize)]
struct ToolSpec {
    function: ToolSpecFunction,
}

#[derive(Deserialize)]
struct ToolSpecFunction {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    parameters: Value,
}

// ─── Output ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct Output {
    text: String,
    tool_calls: Vec<ToolCallOut>,
    stop_reason: String,
    prompt_tokens: usize,
    tokens_generated: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    session: Option<SessionOut>,
    // ── Temporary phase-1/phase-2 diagnostics (job 18932089 follow-up) ──
    // `debug_full_text` is phase-1's raw decoded generation (pre tool-call
    // stripping); `debug_phase1_marker` is whether it contained a literal
    // `<tool_call>` block the decoder may have failed to parse;
    // `debug_phase2_fired` records whether the forced-call fallback ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    debug_full_text: Option<String>,
    debug_phase1_marker: bool,
    debug_phase2_fired: bool,

    /// Per-phase wallclock breakdown of this call. See [`Timings`].
    #[serde(skip_serializing_if = "Option::is_none")]
    timings: Option<Timings>,
}

#[derive(Serialize)]
struct ToolCallOut {
    id: String,
    name: String,
    arguments: String,
}

/// Per-phase wallclock inside one call, in milliseconds.
///
/// Diagnostic for the ~9.5 s/call of non-decode time measured on A/B job
/// 19212030: decode ran at parity with vLLM (63 vs 65 tok/s), generated token
/// counts matched, and APC cut prefill to ~5%, yet Pie took 13.5 s/call
/// against the baseline's 3.9 s. These fields split the call so the residual
/// can be attributed instead of guessed at. Always emitted — the cost is a
/// handful of clock reads.
#[derive(Serialize, Default)]
struct Timings {
    /// `runtime::models` + `Model::load` + `sanitize_messages`.
    setup_ms: f64,
    /// `render_prompt`: one tokenizer round-trip per message, over the whole
    /// history, every call (the inferlet is a fresh wasm instance per call and
    /// cannot carry the previous render forward).
    render_ms: f64,
    /// `fnv1a64` over the full render, plus one `content_hash` per reuse
    /// candidate — each of which re-hashes from token 0.
    hash_ms: f64,
    /// `Context::open` attempts against candidate boundary names.
    open_ms: f64,
    /// How many `Context::open` calls were made (cap is MAX_OPEN_ATTEMPTS).
    open_attempts: usize,
    /// `append` + `flush` — the actual prompt prefill.
    prefill_ms: f64,
    /// `ctx.save` of this call's full render.
    save_ms: f64,
    /// `ctx.fork()` for the phase-2 fallback. Taken on every call whenever
    /// grammar + tools are on, even though phase 2 fired 0/169 times in
    /// 19212030 — forking copies the cued prompt's working pages, so at ~30k
    /// tokens this is a prime suspect for the residual.
    fork_ms: f64,
    /// The phase-1 generation loop (decode + tool-call parsing).
    decode_ms: f64,
    /// Whole call, entry to exit.
    total_ms: f64,
}

#[derive(Serialize)]
struct SessionOut {
    id: String,
    /// "fresh" | "extended" | "forked" | "rebuilt" | "stateless" | "deleted"
    mode: String,
    /// Token count of this call's prompt render (excluding the cue).
    len: usize,
    /// FNV-1a-64 hash of this call's prompt render, lowercase hex.
    hash: String,
    /// Prompt tokens actually prefilled this call (excluding the cue).
    prefill_tokens: usize,
}

// ─── Entry point ───────────────────────────────────────────────────────────

/// Serve one request.
///
/// Called once per process in one-shot mode, or once per received message in
/// daemon mode (see `main`). The body is unchanged from when this WAS `main` —
/// including `Model::load`, which stays per-request deliberately: it binds to a
/// model the runtime already has resident, so it is a handle lookup rather than
/// a weight load, and `timings.setup_ms` measures it either way. Hoisting it
/// would have forced `&model` -> `&&Model` churn through 20 call sites for a
/// cost the telemetry can now simply show us.
async fn handle_request(mut input: Input) -> Result<Output> {
    let t_call = Instant::now();
    let mut timings = Timings::default();

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    sanitize_messages(&mut input.messages, &model);
    timings.setup_ms = ms(t_call);

    // ── Session teardown (conversation ended on the host) ──────────────
    if input.session_action.as_deref() == Some("delete") {
        let sid = input
            .session_id
            .as_deref()
            .ok_or("session_action=delete requires session_id")?;
        // Ignore "not found" — deletion must be idempotent (the harness
        // calls it from a finally block, including after failed runs).
        //
        // Two names to clear. The legacy single-slot snapshot from the
        // pre-APC scheme, and — the one that matters — the whole
        // `apc/{sid}/` namespace this session filled with one
        // content-addressed snapshot per call. Those pin their KV pages
        // until deleted, and their names are derived from token content the
        // host does not keep, so a namespace-prefix delete is the only way
        // to release them. Skipping it leaks a full conversation's KV per
        // conversation: on a hard-allocated cache (cuda_native,
        // swap_pool_size=0) a few back-to-back conversations exhaust the
        // budget and later ones block forever waiting for pages.
        let _ = Context::delete(&model, &session_name(sid));
        let _ = Context::delete(&model, &prefix_cache::namespace(sid, CACHE_COMPAT));
        return Ok(Output {
            text: String::new(),
            tool_calls: Vec::new(),
            stop_reason: "session_deleted".to_string(),
            prompt_tokens: 0,
            tokens_generated: 0,
            session: Some(SessionOut {
                id: sid.to_string(),
                mode: "deleted".to_string(),
                len: 0,
                hash: String::new(),
                prefill_tokens: 0,
            }),
            debug_full_text: None,
            debug_phase1_marker: false,
            debug_phase2_fired: false,
            // Teardown does no rendering or inference — nothing to attribute.
            timings: None,
        });
    }

    let tool_schemas: Vec<String> = input
        .tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            })
            .to_string()
        })
        .collect();

    // Canonical prompt render — identical to the stateless inferlet's
    // token stream, minus the trailing cue (appended after session
    // bookkeeping so the snapshot ends on a message boundary). `boundaries`
    // marks the token length at each safe render-unit split, so the reuse
    // search below can name candidate prefixes without re-rendering.
    let t0 = Instant::now();
    let (full_tokens, boundaries) = render_prompt(&model, &input.messages, &tool_schemas)?;
    timings.render_ms = ms(t0);

    let t0 = Instant::now();
    let full_hash = fnv1a64(&full_tokens);
    timings.hash_ms = ms(t0);

    let (mut ctx, session) = build_context(
        &model,
        &input,
        model_name,
        &full_tokens,
        full_hash,
        &boundaries,
        &mut timings,
    )
    .await?;

    ctx.cue();

    // `seq_len()` only counts committed + working tokens already flushed to
    // the host — the cue just added is still in the local buffer.
    let prompt_token_count = ctx.seq_len() as usize + ctx.buffer().len();

    // Greedy when temperature == 0; otherwise top-p nucleus.
    let sampler = if input.temperature <= 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        }
    };

    // Also stop at the turn-START marker: at t=0 a looping model starts
    // simulating the next turn instead of stopping, and the leaked
    // "<|im_start|>" text round-trips through OpenHands message content
    // into fake turn boundaries on the next request, degenerating into
    // token salad (traj job 18825434).
    let mut stop_token_ids = chat::stop_tokens(&model);
    let turn_start = model.tokenizer().encode("<|im_start|>");
    if turn_start.len() == 1 && !stop_token_ids.contains(&turn_start[0]) {
        stop_token_ids.push(turn_start[0]);
    }
    let has_tools = !tool_schemas.is_empty();
    let mut tool_decoder = has_tools.then(|| tools::Decoder::new(&model));

    // Phase 1 below runs UNCONSTRAINED — constraining the whole turn with
    // the tool-call grammar suppressed all reasoning text and collapsed
    // t=0 agent trajectories into action loops. The fork snapshots the
    // cued prompt so that, when the model produces no tool call at all, a
    // short phase-2 pass can replay the prose and force one well-formed
    // call under the grammar (tool_choice=required at a natural boundary).
    // `use_grammar: false` disables phase 2 for parity with a fully
    // unconstrained baseline. The fork is transient and destroyed before
    // returning, so session snapshot bookkeeping is unaffected.
    //
    // STATUS: phase 2 belongs to grammar mode only, and grammar mode is not
    // the path currently in use. `--python-tool-parser` forces
    // `use_grammar: false` (llm.py), so the gate below is false and no fork is
    // taken — job 19249847 measured fork_ms at 0.0. Across debug logs, phase 2
    // fired 12 of 13 calls on v7 (job 18937308, Qwen2.5-Coder-32B, grammar
    // mode) and 0 of 753 calls over the twelve runs since, all of which used
    // the Python parser. Kept deliberately: it is the fallback that makes
    // grammar mode usable on a model that narrates instead of calling, it
    // costs nothing when disabled, and deleting it would throw away working
    // behaviour for no measured gain. Revisit only if grammar mode is dropped
    // outright — at which point `use_grammar`, the matcher plumbing, and this
    // whole path go together.
    let t0 = Instant::now();
    let mut phase2_fork = if input.use_grammar && has_tools {
        Some(ctx.fork()?)
    } else {
        None
    };
    timings.fork_ms = ms(t0);

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut tool_calls: Vec<ToolCallOut> = Vec::new();
    let mut stop_reason = "length";

    let t_decode = Instant::now();
    let mut g = ctx
        .generate(sampler.clone())
        .max_tokens(input.max_tokens)
        .stop(&stop_token_ids);

    'outer: while let Some(step) = g.next()? {
        let out = step.execute().await?;

        for &t in &out.tokens {
            generated.push(t);

            if let Some(dec) = tool_decoder.as_mut() {
                if let tools::Event::Call(name, arguments) = dec.feed(&[t])? {
                    // Looping models emit the same call several times in one
                    // turn; executing the copies just burns agent iterations.
                    let dup = tool_calls
                        .iter()
                        .any(|c| c.name == name && c.arguments == arguments);
                    if !dup {
                        tool_calls.push(ToolCallOut {
                            id: format!("call_{}", tool_calls.len()),
                            name,
                            arguments,
                        });
                    }
                }
            }

            // EOS / chat-template stop token hit — Generator's .stop() should
            // already have flagged this, but record it explicitly.
            if stop_token_ids.contains(&t) {
                stop_reason = "eos";
                break 'outer;
            }
        }

        // Stop-string check: decode just the tail to keep cost O(stop_len).
        if !input.stop.is_empty() {
            let max_stop_len = input.stop.iter().map(|s| s.len()).max().unwrap_or(0);
            // ~4 chars/token upper-bounds the tail tokens we need to decode.
            let tail_tokens = (max_stop_len / 2).max(8).min(generated.len());
            let tail_start = generated.len() - tail_tokens;
            if let Ok(tail) = model.tokenizer().decode(&generated[tail_start..]) {
                if input.stop.iter().any(|s| tail.ends_with(s)) {
                    stop_reason = "stop";
                    break;
                }
            }
        }

        if generated.len() >= input.max_tokens {
            break;
        }
    }

    timings.decode_ms = ms(t_decode);

    // The Generator consumes its stop token internally, so a natural stop
    // can fall through the explicit eos/stop-string checks above with the
    // "length" default still in place. Only report "length" when the token
    // budget was actually exhausted — OpenHands treats "length" as a
    // truncated response.
    if stop_reason == "length" && generated.len() < input.max_tokens {
        stop_reason = "eos";
    }

    let full_text = model
        .tokenizer()
        .decode(&generated)
        .unwrap_or_else(|_| String::from("[decode error]"));

    // Fallback for models that write a tool call as a fenced JSON block
    // instead of <tool_call> tags (observed on Qwen2.5-Coder-32B at t=0
    // with the grammar constraint off — traj job 18819521).
    let mut fence_split_at: Option<usize> = None;
    if tool_calls.is_empty() {
        for (offset, name, args) in parse_fenced_tool_calls(&full_text) {
            fence_split_at.get_or_insert(offset);
            let dup = tool_calls
                .iter()
                .any(|c| c.name == name && c.arguments == args);
            if !dup {
                tool_calls.push(ToolCallOut {
                    id: format!("call_{}", tool_calls.len()),
                    name,
                    arguments: args,
                });
            }
        }
    }

    // Phase 2 — forced tool call. The model narrated without acting
    // (no <tool_call>, no fence); at t=0 that repeats verbatim through
    // every nudge until the run stucks out. Replay the prose on the
    // pre-generation fork (raw token ids — no re-encode) and force one
    // well-formed call under the tool-call grammar.
    // Runs even when phase 1 produced nothing at all (immediate stop
    // token, traj job 18904720 completion 5) — an empty response forces
    // OpenHands into a no-op nudge round-trip.
    let mut phase2_fired = false;
    if tool_calls.is_empty() {
        if let Some(mut fk) = phase2_fork.take() {
            if let Some(matcher) = tools::native_matcher(&model, &tool_schemas) {
                phase2_fired = true;
                let mut prose = generated.clone();
                if prose.last().is_some_and(|t| stop_token_ids.contains(t)) {
                    prose.pop();
                }
                // Template renders content, then '\n', then the first
                // <tool_call> block.
                fk.append(&prose);
                fk.append(&model.tokenizer().encode("\n"));

                let mut dec2 = tools::Decoder::new(&model);
                let mut g2 = fk
                    .generate(sampler)
                    .max_tokens(FORCED_CALL_MAX_TOKENS)
                    .stop(&stop_token_ids)
                    .constrain(inferlet::GrammarConstraint::new(matcher));
                let mut forced = 0usize;
                'forced: while let Some(step) = g2.next()? {
                    let out = step.execute().await?;
                    for &t in &out.tokens {
                        forced += 1;
                        if let tools::Event::Call(name, arguments) = dec2.feed(&[t])? {
                            tool_calls.push(ToolCallOut {
                                id: format!("call_{}", tool_calls.len()),
                                name,
                                arguments,
                            });
                            // One forced call is the point of phase 2 —
                            // letting the grammar run on pads the turn
                            // with junk-argument extra calls (traj job
                            // 18904720 completion 6).
                            break 'forced;
                        }
                        if stop_token_ids.contains(&t) {
                            break 'forced;
                        }
                    }
                    if forced >= FORCED_CALL_MAX_TOKENS {
                        break;
                    }
                }
                drop(g2);
            }
            // Plain drop, not destroy(): eager destroy + the handle's own
            // resource drop double-deletes host-side. Instance exit
            // collects the anonymous fork.
            drop(fk);
        }
    }
    drop(phase2_fork.take());

    let text = if tool_calls.is_empty() {
        trim_trailing_stop(&full_text, &input.stop).to_string()
    } else if let Some(at) = fence_split_at {
        // Fence-parsed calls: content is the prose before the first fence.
        full_text[..at].trim().to_string()
    } else {
        // Only the free text that preceded the first tool call is meaningful
        // content — the <tool_call> blocks themselves are already captured
        // in `tool_calls` above.
        full_text
            .split("<tool_call>")
            .next()
            .unwrap_or("")
            .trim()
            .to_string()
    };

    if !tool_calls.is_empty() {
        stop_reason = "tool_calls";
    }

    // No next-boundary prediction: the reusable boundary is always a full
    // host-message render (saved in `build_context`), never a guessed
    // assistant reply. History is append-only, so this call's saved full
    // render is a literal token-prefix of the next call's render and is
    // re-discovered there by the boundary search — byte-identical name, no
    // re-serialization drift.
    let debug_phase1_marker = full_text.contains("<tool_call>");
    Ok(Output {
        text,
        tool_calls,
        stop_reason: stop_reason.to_string(),
        prompt_tokens: prompt_token_count,
        tokens_generated: generated.len(),
        session,
        debug_full_text: Some(full_text.clone()),
        debug_phase1_marker,
        debug_phase2_fired: phase2_fired,
        timings: Some(Timings {
            total_ms: ms(t_call),
            ..timings
        }),
    })
}

/// Extract tool calls written as fenced JSON blocks: a ``` fence (with or
/// without a language tag) whose body is an object with a string `name`
/// and an object `arguments`. Returns `(fence_byte_offset, name,
/// arguments_json)` per match, in order.
// ─── Process entry point: one-shot or daemon ───────────────────────────────

/// Raw-JSON entry point.
///
/// The `#[inferlet::main]` macro passes `String` in and out untouched (it only
/// serializes *typed* parameters — inferlet-macros/src/lib.rs:113-142), so the
/// one-shot path below is byte-identical to what callers saw when `main` took
/// `Input` and returned `Output`.
///
/// TWO MODES, selected by a `daemon` flag in the launch payload:
///
/// - **one-shot** (default, unchanged): the launch payload IS the request.
///   Serve it, return the `Output`, exit. Every existing caller keeps working.
///
/// - **daemon**: launch once, then serve requests off `session::receive()`
///   until the host says stop. This is what makes a fair comparison against a
///   persistent vLLM server possible.
///
/// WHY THE DAEMON MODE EXISTS. In one-shot mode the host pays, on EVERY LLM
/// call, a websocket connect, an authenticate, a process launch, admission
/// queueing, and a teardown — none of which vLLM's already-running HTTP server
/// charges per request. Measuring Pie that way benchmarks
/// `PieLLM._call_pie`'s call pattern, not Pie's serving. The per-request design
/// was a deliberate, documented choice (docs/OPENHANDS_CODER_SESSION_DESIGN.md
/// line 56, which also names `launch_daemon` as the alternative "if a long-lived
/// server inferlet is ever preferred"). It is now preferred.
///
/// WHAT DAEMON MODE DOES *NOT* CHANGE: the KV path. Each request still renders
/// the full history, still resolves reuse through content-addressed snapshots,
/// still saves. That keeps `--kv-verify` and the prefill-reuse percentage
/// directly comparable against the one-shot arms, so the only variable moving
/// is process lifetime. Holding a `Context` live in this loop across requests
/// would be a further (larger) win and a separate experiment — it would also
/// take Pie past what vLLM does, which is a different claim than the one this
/// change is meant to support.
///
/// WIRE PROTOCOL (all frames are single-line JSON):
///   inferlet -> host   {"ready":true}                      once, after launch
///   host -> inferlet   {<Input fields>}                    a request
///   host -> inferlet   {"daemon_action":"shutdown"}        stop serving
///   inferlet -> host   {"ok":true,"result":{<Output>}}     a served request
///   inferlet -> host   {"ok":false,"error":"..."}          a failed request
///
/// Errors are reported IN-BAND rather than by returning `Err`, because
/// returning would kill the process and take the host's next N calls with it.
/// A malformed request must cost one call, not the conversation.
#[inferlet::main]
async fn main(raw: String) -> Result<String> {
    let launch: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("invalid launch payload: {e}"))?;

    let daemon = launch
        .get("daemon")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    if !daemon {
        let input: Input = serde_json::from_value(launch)
            .map_err(|e| format!("invalid request payload: {e}"))?;
        let out = handle_request(input).await?;
        return serde_json::to_string(&out)
            .map_err(|e| format!("failed to serialize output: {e}"));
    }

    session::send(r#"{"ready":true}"#);

    let mut served: u64 = 0;
    loop {
        // `None` means the host closed the connection without saying goodbye
        // (harness crash, SSH drop). Exit rather than spin.
        let Some(msg) = session::receive().wait_async().await else {
            break;
        };

        let parsed: serde_json::Value = match serde_json::from_str(&msg) {
            Ok(v) => v,
            Err(e) => {
                session::send(&error_frame(&format!("invalid request JSON: {e}")));
                continue;
            }
        };

        if parsed.get("daemon_action").and_then(serde_json::Value::as_str)
            == Some("shutdown")
        {
            break;
        }

        let frame = match serde_json::from_value::<Input>(parsed) {
            Ok(input) => match handle_request(input).await {
                Ok(out) => serde_json::to_string(&serde_json::json!({
                    "ok": true,
                    "result": out,
                }))
                .unwrap_or_else(|e| error_frame(&format!("serialize failed: {e}"))),
                Err(e) => error_frame(&e),
            },
            Err(e) => error_frame(&format!("invalid request fields: {e}")),
        };
        session::send(&frame);
        served += 1;
    }

    Ok(format!("{{\"served\":{served}}}"))
}

/// An in-band error frame. Built by hand so that constructing it can never
/// itself fail and leave the host waiting forever for a reply.
fn error_frame(msg: &str) -> String {
    let escaped = msg
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ")
        .replace('\r', " ");
    format!(r#"{{"ok":false,"error":"{escaped}"}}"#)
}

fn parse_fenced_tool_calls(text: &str) -> Vec<(usize, String, String)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(rel) = text[pos..].find("```") {
        let fence_at = pos + rel;
        let after = &text[fence_at + 3..];
        // Skip the language tag line (e.g. "json\n"); a fence with no
        // newline at all has no body.
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

// ─── Session context construction ──────────────────────────────────────────

/// Build the generation context for this call, either by extending the
/// saved session snapshot or by rebuilding from scratch, and refresh the
/// snapshot to equal this call's prompt render.
///
/// Returns the context (prompt fully flushed, no cue yet) and the session
/// telemetry block (None in stateless mode).
async fn build_context(
    model: &Model,
    input: &Input,
    model_id: &str,
    full_tokens: &[u32],
    full_hash: u64,
    boundaries: &[usize],
    timings: &mut Timings,
) -> Result<(Context, Option<SessionOut>)> {
    let Some(sid) = input.session_id.as_deref() else {
        // Stateless: behave exactly like openhands-completion. Leave the
        // tokens in the buffer — the generator's first step prefills them
        // together with the cue, matching the baseline's single-pass shape.
        let mut ctx = Context::new(model)?;
        ctx.append(full_tokens);
        return Ok((ctx, None));
    };

    // Content-addressed reuse, self-keyed from token content alone. Every call
    // saves its *full* host-message render under `hash(full_tokens)`. Because
    // OpenHands history is append-only, a previous turn's full render is a
    // literal token-prefix of this render and re-appears here as one of
    // `boundaries` (the safe render-unit split points). So we scan those
    // boundaries longest-first, opening `hash(full[..L])`; the most recent
    // saved boundary wins. Both save and lookup name a full host render — no
    // predicted assistant reply — so the names match byte-for-byte and JSON
    // re-serialization drift can never cause a miss.
    //
    // A candidate slice is by construction a literal prefix of `full_tokens`,
    // so no strict-prefix recheck is needed: the `seq_len()` guard below only
    // rejects a snapshot whose stored length disagrees (never a wrong suffix).
    //
    // This also subsumes the old fork protocol: delegation reuse happens for
    // free when a sub-agent's prefix tokens equal a boundary the parent already
    // saved (same tokens ⇒ same name ⇒ hit), with no `session_fork_*` hints.
    //
    // The scan is capped: the previous turn's boundary sits within the handful
    // of messages appended since (assistant reply + tool/user results), so a
    // small window always covers it; an older-than-window boundary just costs a
    // clean rebuild.
    const MAX_OPEN_ATTEMPTS: usize = 8;
    let full_len = full_tokens.len();
    let mut had_candidates = false;
    let mut opened: Option<(Context, usize)> = None;
    let mut attempts = 0;
    for &len in boundaries.iter().rev() {
        if len == 0 || len >= full_len {
            continue;
        }
        had_candidates = true;
        if attempts >= MAX_OPEN_ATTEMPTS {
            break;
        }
        attempts += 1;
        let t_h = Instant::now();
        let name = prefix_cache::snapshot_name(sid, CACHE_COMPAT, model_id, &full_tokens[..len]);
        timings.hash_ms += ms(t_h);
        let t_o = Instant::now();
        let open_result = Context::open(model, &name);
        timings.open_ms += ms(t_o);
        if let Ok(c) = open_result {
            // Opening shares the snapshot's committed KV pages by refcount;
            // appending the suffix allocates fresh pages, so the immutable
            // snapshot is never mutated by the live generation past here.
            if c.seq_len() as usize == len {
                opened = Some((c, len));
                break;
            }
        }
        // Open failure / length mismatch: snapshot lost or unsafe — try the
        // next-shorter boundary, else fall through to a clean rebuild.
    }

    timings.open_attempts = attempts;

    let t_prefill = Instant::now();
    let (mut ctx, mode, prefill) = match opened {
        Some((mut c, len)) => {
            let suffix = &full_tokens[len..];
            c.append(suffix);
            (c, "extended", suffix.len())
        }
        None => {
            let mut c = Context::new(model)?;
            c.append(full_tokens);
            // "fresh" when there was no interior boundary to reuse (first turn /
            // single render unit); "rebuilt" when candidates existed but all
            // missed (snapshot lost, or the condenser rewrote history).
            let mode = if had_candidates { "rebuilt" } else { "fresh" };
            (c, mode, full_tokens.len())
        }
    };

    // Materialize the prompt KV so the snapshot below captures it. (An
    // empty append — identical retried prompt — makes this a no-op.)
    ctx.flush().await?;
    timings.prefill_ms = ms(t_prefill);

    if input.kv_verify && ctx.seq_len() as usize != full_tokens.len() {
        return Err(format!(
            "kv-verify: context holds {} tokens after prefill, render has {}",
            ctx.seq_len(),
            full_tokens.len()
        ));
    }

    // Save this call's full render under its own content-addressed name so a
    // later turn whose reusable prefix equals this render hits it. An identical
    // name means identical KV is already saved (benign) — ignore the save
    // error rather than delete+resave, so distinct boundaries coexist (that is
    // what makes retry / branch / truncate re-hit their earlier boundary).
    let t_h = Instant::now();
    let full_name = prefix_cache::snapshot_name(sid, CACHE_COMPAT, model_id, full_tokens);
    timings.hash_ms += ms(t_h);
    let t_save = Instant::now();
    let _ = ctx.save(&full_name);
    timings.save_ms = ms(t_save);

    let session = SessionOut {
        id: sid.to_string(),
        mode: mode.to_string(),
        len: full_tokens.len(),
        hash: format!("{full_hash:x}"),
        prefill_tokens: prefill,
    };
    Ok((ctx, Some(session)))
}

fn session_name(session_id: &str) -> String {
    format!("oh-session-{session_id}")
}

/// FNV-1a 64-bit over the little-endian bytes of the token IDs.
fn fnv1a64(tokens: &[u32]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
    }
    h
}

// ─── History rendering ─────────────────────────────────────────────────────

/// Render `messages` turn by turn into a token vector, matching the model's
/// chat template byte-for-byte (including merged tool-call/tool-response
/// turns). Identical token stream to `openhands-completion`'s
/// `replay_history`, but collected into a Vec instead of appended to a
/// context, so the session logic can hash/slice it. Excludes the cue.
/// Strip the tokenizer's special-token strings out of round-tripped message
/// content. `Tokenizer::encode` maps a literal special-token string (e.g.
/// "<|im_start|>" leaked into an assistant reply) back to the real token id,
/// so replaying it would plant fake turn boundaries mid-message and corrupt
/// the chat structure (traj job 18825434 degenerated into token salad this
/// way).
fn sanitize_messages(messages: &mut [Message], model: &Model) {
    let (_ids, byte_seqs) = model.tokenizer().special_tokens();
    let specials: Vec<String> = byte_seqs
        .into_iter()
        .filter_map(|b| String::from_utf8(b).ok())
        .filter(|s| !s.is_empty())
        .collect();
    if specials.is_empty() {
        return;
    }
    let clean = |s: &mut String| {
        for sp in &specials {
            if s.contains(sp.as_str()) {
                *s = s.replace(sp.as_str(), "");
            }
        }
    };
    for m in messages.iter_mut() {
        if let Some(c) = m.content.as_mut() {
            clean(c);
        }
        if let Some(calls) = m.tool_calls.as_mut() {
            for c in calls.iter_mut() {
                clean(&mut c.function.name);
                clean(&mut c.function.arguments);
            }
        }
    }
}

/// Renders the full token stream and, alongside it, the token length at each
/// safe render-unit boundary (after the system+tools block, after each
/// user/assistant/system message, and after each merged tool batch). These
/// boundaries are exactly the points at which a prior turn's full render could
/// end, so `build_context` names KV-snapshot lookup candidates by slicing
/// `full[..len]` at them — no per-candidate re-render, and every slice is a
/// literal prefix of the full render. The final boundary equals the full
/// length; the reuse search skips it.
fn render_prompt(
    model: &Model,
    messages: &[Message],
    tool_schemas: &[String],
) -> Result<(Vec<u32>, Vec<usize>)> {
    let mut out: Vec<u32> = Vec::new();
    let mut boundaries: Vec<usize> = Vec::new();
    let mut equipped = false;
    let mut i = 0;

    // The model's chat template folds a leading system message's content
    // into the *same* system turn as the tool schemas (see
    // `equip_after_system_prefix`'s doc comment) rather than two separate
    // consecutive system turns — handle that one turn specially before the
    // general per-message loop below.
    if !tool_schemas.is_empty() && messages.first().map(|m| m.role.as_str()) == Some("system") {
        let content = messages[0].content.as_deref();
        out.extend(tools::equip_after_system_prefix(model, content, tool_schemas)?);
        equipped = true;
        i = 1;
        boundaries.push(out.len());
    }

    while i < messages.len() {
        let msg = &messages[i];

        // `equip_prefix` glues the tool schemas onto the *following* message's
        // turn — no boundary is recorded between them, so a split never falls
        // mid-turn.
        if !equipped && !tool_schemas.is_empty() && msg.role != "system" {
            out.extend(tools::equip_prefix(model, tool_schemas)?);
            equipped = true;
        }

        match msg.role.as_str() {
            "system" => {
                out.extend(chat::system(model, msg.content.as_deref().unwrap_or("")));
                i += 1;
            }
            "user" => {
                out.extend(chat::user(model, msg.content.as_deref().unwrap_or("")));
                i += 1;
            }
            "assistant" => {
                match &msg.tool_calls {
                    Some(calls) if !calls.is_empty() => {
                        let pairs: Vec<(String, String)> = calls
                            .iter()
                            .map(|c| (c.function.name.clone(), c.function.arguments.clone()))
                            .collect();
                        out.extend(tools::assistant_with_tool_calls_prefix(
                            model,
                            msg.content.as_deref(),
                            &pairs,
                        ));
                    }
                    _ => {
                        out.extend(chat::assistant(model, msg.content.as_deref().unwrap_or("")));
                    }
                }
                i += 1;
            }
            "tool" => {
                // Merge this and any immediately-consecutive tool results into
                // one replayed turn (the model was fine-tuned on the merged
                // form — see answer_batch's doc comment).
                let mut batch: Vec<(String, String)> = Vec::new();
                while i < messages.len() && messages[i].role == "tool" {
                    batch.push((String::new(), messages[i].content.clone().unwrap_or_default()));
                    i += 1;
                }
                out.extend(tools::answer_batch_prefix(model, &batch));
            }
            other => return Err(format!("unsupported message role: {other}")),
        }
        boundaries.push(out.len());
    }

    if !equipped && !tool_schemas.is_empty() {
        out.extend(tools::equip_prefix(model, tool_schemas)?);
    }

    Ok((out, boundaries))
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn trim_trailing_stop<'a>(text: &'a str, stops: &[String]) -> &'a str {
    for s in stops {
        if let Some(stripped) = text.strip_suffix(s.as_str()) {
            return stripped;
        }
    }
    text
}
