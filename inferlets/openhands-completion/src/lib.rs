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

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut tool_calls: Vec<ToolCallOut> = Vec::new();
    let mut stop_reason = "length";

    let mut g = ctx
        .generate(sampler)
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

fn trim_trailing_stop<'a>(text: &'a str, stops: &[String]) -> &'a str {
    for s in stops {
        if let Some(stripped) = text.strip_suffix(s.as_str()) {
            return stripped;
        }
    }
    text
}
