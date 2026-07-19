//! openhands-completion — native-tool-calling inferlet for the PieLLM integration.
//!
//! Contract (must stay in sync with `pie_openhands.llm.PieLLM._call_pie`):
//!
//!   Input:
//!     messages:     Vec<Message>  (OpenAI-style chat turns — see below)
//!     tools:        Vec<ToolSpec> (default empty; OpenAI `tools[]` shape)
//!     max_tokens:   usize   (default 2048)
//!     temperature:  f32     (default 0.0 — greedy)
//!     top_p:        f32     (default 0.95)
//!     stop:         Vec<String>  (default empty)
//!     model:        Option<String>  (informational; runtime picks the first model)
//!
//!   Message: { role, content?, tool_calls?: [{id?, function: {name, arguments}}],
//!              tool_call_id?, name? }
//!   ToolSpec: { function: { name, description?, parameters? } }
//!
//!   Output:
//!     text:             String        (any free text preceding tool_calls, or the
//!                                       full reply when there are none)
//!     tool_calls:       Vec<ToolCallOut>  ({id, name, arguments} — arguments is a
//!                                       JSON-encoded string, empty when none)
//!     stop_reason:      "stop" | "length" | "eos" | "tool_calls"
//!     prompt_tokens:    usize
//!     tokens_generated: usize
//!
//! Each request rebuilds the *entire* conversation from scratch — this inferlet
//! does not pin KV state across requests (that's `openhands-coder-session`, a
//! later phase). Past assistant tool-calls and past tool-result turns are
//! replayed via `Instruct::assistant_with_tool_calls`/`answer_batch` (exposed
//! through `tools::assistant_with_tool_calls_prefix`/`answer_batch_prefix`) so
//! they come back out byte-identical to what the model's own chat template
//! would have produced — see
//! `integrations/openhands/docs/TOOL_CALL_HISTORY_REPLAY_DESIGN.md`.

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler, tools};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── Input ─────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Input {
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
}

fn default_max_tokens() -> usize { 2048 }
fn default_temperature() -> f32 { 0.0 }
fn default_top_p() -> f32 { 0.95 }

/// Token budget for the phase-2 forced tool call — one call plus slack
/// (multi-line editor arguments can run long).
const FORCED_CALL_MAX_TOKENS: usize = 1024;

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
}

#[derive(Serialize)]
struct ToolCallOut {
    id: String,
    name: String,
    arguments: String,
}

// ─── Entry point ───────────────────────────────────────────────────────────

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

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

    let mut ctx = Context::new(&model)?;
    replay_history(&mut ctx, &model, &input.messages, &tool_schemas)?;
    ctx.cue();

    // `seq_len()` only counts committed + working tokens already flushed to
    // the host — the turns just built above are still sitting in the local
    // buffer until the first forward pass, so they must be added by hand.
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

    // Phase 1 below runs UNCONSTRAINED — constraining the whole turn with
    // the tool-call grammar suppressed all reasoning text and collapsed
    // t=0 agent trajectories into action loops. The fork snapshots the
    // cued prompt so that, when the model produces no tool call at all, a
    // short phase-2 pass can replay the prose and force one well-formed
    // call under the grammar (tool_choice=required at a natural boundary).
    let mut phase2_fork = if has_tools { Some(ctx.fork()?) } else { None };

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut tool_calls: Vec<ToolCallOut> = Vec::new();
    let mut stop_reason = "length";

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
            tool_calls.push(ToolCallOut {
                id: format!("call_{}", tool_calls.len()),
                name,
                arguments: args,
            });
        }
    }

    // Phase 2 — forced tool call. The model narrated without acting
    // (no <tool_call>, no fence); at t=0 that repeats verbatim through
    // every nudge until the run stucks out. Replay the prose on the
    // pre-generation fork (raw token ids — no re-encode) and force one
    // well-formed call under the tool-call grammar.
    if tool_calls.is_empty() && !generated.is_empty() {
        if let Some(mut fk) = phase2_fork.take() {
            if let Some(matcher) = tools::native_matcher(&model, &tool_schemas) {
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

    Ok(Output {
        text,
        tool_calls,
        stop_reason: stop_reason.to_string(),
        prompt_tokens: prompt_token_count,
        tokens_generated: generated.len(),
    })
}

// ─── History replay ────────────────────────────────────────────────────────

/// Replay `messages` turn by turn, matching the model's chat template
/// byte-for-byte (including merged tool-call/tool-response turns) rather
/// than concatenating hand-formatted strings.
fn replay_history(
    ctx: &mut Context,
    model: &Model,
    messages: &[Message],
    tool_schemas: &[String],
) -> Result<()> {
    let mut equipped = false;
    let mut i = 0;

    // The model's chat template folds a leading system message's content
    // into the *same* system turn as the tool schemas (see
    // `equip_after_system_prefix`'s doc comment) rather than two separate
    // consecutive system turns — handle that one turn specially before the
    // general per-message loop below.
    if !tool_schemas.is_empty() && messages.first().map(|m| m.role.as_str()) == Some("system") {
        let content = messages[0].content.as_deref();
        ctx.append(&tools::equip_after_system_prefix(model, content, tool_schemas)?);
        equipped = true;
        i = 1;
    }

    while i < messages.len() {
        let msg = &messages[i];

        if !equipped && !tool_schemas.is_empty() && msg.role != "system" {
            ctx.append(&tools::equip_prefix(model, tool_schemas)?);
            equipped = true;
        }

        match msg.role.as_str() {
            "system" => {
                ctx.system(msg.content.as_deref().unwrap_or(""));
                i += 1;
            }
            "user" => {
                ctx.user(msg.content.as_deref().unwrap_or(""));
                i += 1;
            }
            "assistant" => {
                match &msg.tool_calls {
                    Some(calls) if !calls.is_empty() => {
                        let pairs: Vec<(String, String)> = calls
                            .iter()
                            .map(|c| (c.function.name.clone(), c.function.arguments.clone()))
                            .collect();
                        let tokens = tools::assistant_with_tool_calls_prefix(
                            model,
                            msg.content.as_deref(),
                            &pairs,
                        );
                        ctx.append(&tokens);
                    }
                    _ => {
                        ctx.assistant(msg.content.as_deref().unwrap_or(""));
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
                ctx.append(&tools::answer_batch_prefix(model, &batch));
            }
            other => return Err(format!("unsupported message role: {other}").into()),
        }
    }

    if !equipped && !tool_schemas.is_empty() {
        ctx.append(&tools::equip_prefix(model, tool_schemas)?);
    }

    Ok(())
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Extract tool calls written as fenced JSON blocks: a ``` fence (with or
/// without a language tag) whose body is an object with a string `name`
/// and an object `arguments`. Returns `(fence_byte_offset, name,
/// arguments_json)` per match, in order.
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

fn trim_trailing_stop<'a>(text: &'a str, stops: &[String]) -> &'a str {
    for s in stops {
        if let Some(stripped) = text.strip_suffix(s.as_str()) {
            return stripped;
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::parse_fenced_tool_calls;

    #[test]
    fn fenced_tool_call_is_extracted() {
        let text = "Let's search first.\n\n```json\n{\"name\": \"terminal\", \"arguments\": {\"command\": \"grep -r x .\"}}\n```";
        let calls = parse_fenced_tool_calls(text);
        assert_eq!(calls.len(), 1);
        let (at, name, args) = &calls[0];
        assert_eq!(name, "terminal");
        assert!(args.contains("grep -r x ."));
        assert_eq!(&text[..*at], "Let's search first.\n\n");
    }

    #[test]
    fn non_tool_fences_are_ignored() {
        let text = "Example:\n```python\nprint('hi')\n```\nand JSON that is not a call:\n```json\n{\"foo\": 1}\n```";
        assert!(parse_fenced_tool_calls(text).is_empty());
    }

    #[test]
    fn multiple_fences_mixed() {
        let text = "```json\n{\"name\": \"a\", \"arguments\": {}}\n```\ntext\n```json\n{\"name\": \"b\", \"arguments\": {\"k\": 2}}\n```";
        let calls = parse_fenced_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].1, "a");
        assert_eq!(calls[1].1, "b");
    }

    #[test]
    fn unterminated_fence_is_ignored() {
        assert!(parse_fenced_tool_calls("```json\n{\"name\": \"a\", \"arguments\": {}}").is_empty());
    }

    #[test]
    fn arguments_must_be_object() {
        assert!(parse_fenced_tool_calls("```json\n{\"name\": \"a\", \"arguments\": \"str\"}\n```").is_empty());
    }
}
