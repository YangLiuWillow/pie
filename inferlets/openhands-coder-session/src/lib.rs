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
//!     session_prev_len:  usize           (echo of last response's session.len; 0 first call)
//!     session_prev_hash: Option<String>  (echo of last response's session.hash, hex)
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
//! How the session works (design doc §1):
//!
//! 1. Render the **full** message list to tokens exactly as the stateless
//!    inferlet would — semantics unchanged; every token the model sees is
//!    byte-identical to the stateless render. The render excludes the trailing
//!    generation cue so the snapshot ends on a message boundary.
//! 2. If the previous prompt (identified by the host-echoed length + FNV-1a
//!    hash) is a token-level prefix of the new render, `Context::open` the
//!    saved snapshot and append only the suffix. Otherwise (condenser rewrote
//!    history, first call, snapshot lost) rebuild from scratch — always
//!    semantically safe, just slower.
//! 3. Refresh the snapshot (delete + save under the same name) *before*
//!    generation, so the saved KV always equals the canonical prompt render —
//!    never the model's own sampled tokens, whose bytes can differ from the
//!    replayed form after argument re-serialization on the host.
//!
//! The snapshot is a prompt-only checkpoint: each new call re-prefills the
//! previous assistant turn plus the new tool results (O(delta)), not the full
//! history (O(conversation)).

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler, tools};
use serde::{Deserialize, Serialize};
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
    #[serde(default)]
    session_id: Option<String>,

    #[serde(default)]
    session_prev_len: usize,

    #[serde(default)]
    session_prev_hash: Option<String>,

    #[serde(default)]
    session_action: Option<String>,

    #[serde(default)]
    kv_verify: bool,

    #[serde(default = "default_true")]
    use_grammar: bool,
}

fn default_max_tokens() -> usize { 2048 }
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
}

#[derive(Serialize)]
struct ToolCallOut {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct SessionOut {
    id: String,
    /// "fresh" | "extended" | "rebuilt" | "stateless" | "deleted"
    mode: String,
    /// Token count of this call's prompt render (excluding the cue).
    len: usize,
    /// FNV-1a-64 hash of this call's prompt render, lowercase hex.
    hash: String,
    /// Prompt tokens actually prefilled this call (excluding the cue).
    prefill_tokens: usize,
}

// ─── Entry point ───────────────────────────────────────────────────────────

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    // ── Session teardown (conversation ended on the host) ──────────────
    if input.session_action.as_deref() == Some("delete") {
        let sid = input
            .session_id
            .as_deref()
            .ok_or("session_action=delete requires session_id")?;
        // Ignore "not found" — deletion must be idempotent (the harness
        // calls it from a finally block, including after failed runs).
        let _ = Context::delete(&model, &session_name(sid));
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
    // bookkeeping so the snapshot ends on a message boundary).
    let full_tokens = render_prompt(&model, &input.messages, &tool_schemas)?;
    let full_hash = fnv1a64(&full_tokens);

    let (mut ctx, session) = build_context(&model, &input, &full_tokens, full_hash).await?;

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

    let stop_token_ids = chat::stop_tokens(&model);
    let has_tools = !tool_schemas.is_empty();
    let mut tool_decoder = has_tools.then(|| tools::Decoder::new(&model));

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut tool_calls: Vec<ToolCallOut> = Vec::new();
    let mut stop_reason = "length";

    let mut g = ctx
        .generate(sampler)
        .max_tokens(input.max_tokens)
        .stop(&stop_token_ids);

    // `native_matcher` traps host-side on an empty schema list, so gate on
    // has_tools as well as the flag.
    if input.use_grammar && has_tools {
        if let Some(matcher) = tools::native_matcher(&model, &tool_schemas) {
            g = g.constrain(inferlet::GrammarConstraint::new(matcher));
        }
    }

    'outer: while let Some(step) = g.next()? {
        let out = step.execute().await?;

        for &t in &out.tokens {
            generated.push(t);

            if let Some(dec) = tool_decoder.as_mut() {
                if let tools::Event::Call(name, arguments) = dec.feed(&[t])? {
                    tool_calls.push(ToolCallOut {
                        id: format!("call_{}", tool_calls.len()),
                        name,
                        arguments,
                    });
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

    let full_text = model
        .tokenizer()
        .decode(&generated)
        .unwrap_or_else(|_| String::from("[decode error]"));

    let text = if tool_calls.is_empty() {
        trim_trailing_stop(&full_text, &input.stop).to_string()
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

    Ok(Output {
        text,
        tool_calls,
        stop_reason: stop_reason.to_string(),
        prompt_tokens: prompt_token_count,
        tokens_generated: generated.len(),
        session,
    })
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
    full_tokens: &[u32],
    full_hash: u64,
) -> Result<(Context, Option<SessionOut>)> {
    let Some(sid) = input.session_id.as_deref() else {
        // Stateless: behave exactly like openhands-completion. Leave the
        // tokens in the buffer — the generator's first step prefills them
        // together with the cue, matching the baseline's single-pass shape.
        let mut ctx = Context::new(model)?;
        ctx.append(full_tokens);
        return Ok((ctx, None));
    };

    let name = session_name(sid);
    let prev_len = input.session_prev_len;
    let prev_hash = input
        .session_prev_hash
        .as_deref()
        .and_then(|h| u64::from_str_radix(h, 16).ok());

    // Extension test at the token level (design doc §1): the previous
    // prompt render must be a literal prefix of the new one.
    let is_extension = prev_len > 0
        && prev_len <= full_tokens.len()
        && prev_hash.is_some()
        && fnv1a64(&full_tokens[..prev_len]) == prev_hash.unwrap();

    let mut opened: Option<Context> = None;
    if is_extension {
        if let Ok(c) = Context::open(model, &name) {
            // The snapshot must hold exactly the tokens the host thinks it
            // does. A mismatch is not a fidelity violation, just a stale
            // snapshot (e.g. a retried call whose previous attempt updated
            // the snapshot but never delivered its response) — rebuild.
            if c.seq_len() as usize == prev_len {
                opened = Some(c);
            }
        }
        // Open failure: snapshot lost (server restart) — rebuild.
    }

    let (mut ctx, mode, prefill) = match opened {
        Some(mut c) => {
            let suffix = &full_tokens[prev_len..];
            c.append(suffix);
            (c, "extended", suffix.len())
        }
        None => {
            let mut c = Context::new(model)?;
            c.append(full_tokens);
            let mode = if prev_len == 0 { "fresh" } else { "rebuilt" };
            (c, mode, full_tokens.len())
        }
    };

    // Materialize the prompt KV so the snapshot below captures it. (An
    // empty append — identical retried prompt — makes this a no-op.)
    ctx.flush().await?;

    if input.kv_verify && ctx.seq_len() as usize != full_tokens.len() {
        return Err(format!(
            "kv-verify: context holds {} tokens after prefill, render has {}",
            ctx.seq_len(),
            full_tokens.len()
        ));
    }

    // Refresh the snapshot: the saved context must always equal this
    // call's canonical prompt render. Committed pages are content-hashed
    // and shared by refcount, so the live context generating past this
    // point cannot mutate the snapshot.
    let _ = Context::delete(model, &name);
    ctx.save(&name)?;

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
fn render_prompt(
    model: &Model,
    messages: &[Message],
    tool_schemas: &[String],
) -> Result<Vec<u32>> {
    let mut out: Vec<u32> = Vec::new();
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
    }

    while i < messages.len() {
        let msg = &messages[i];

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
    }

    if !equipped && !tool_schemas.is_empty() {
        out.extend(tools::equip_prefix(model, tool_schemas)?);
    }

    Ok(out)
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
