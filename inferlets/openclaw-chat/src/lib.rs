//! openclaw-chat — streaming chat completion inferlet for the OpenClaw integration.
//!
//! Accepts OpenAI-shaped messages and tools (passed as JSON strings from the
//! manifest), runs the forward pass with native tool-call support, and emits
//! incremental output as JSON-line events on stdout so the JS client can map
//! them to OpenClaw's `AssistantMessageEvent` stream.
//!
//! **Session persistence (KV cache pinning):**
//!   Pass `session_id` to save the KV cache after generation. On follow-up
//!   turns, pass the same `session_id` plus `resume_from` (the number of
//!   messages already in the saved context). The inferlet opens the saved
//!   snapshot and only appends the *new* messages, avoiding a full replay.
//!
//! Stdout event protocol (one JSON object per line):
//!   {"type":"text_delta","delta":"..."}
//!   {"type":"tool_call","id":"call_0","name":"...","arguments":"..."}
//!   {"type":"done","stop_reason":"stop"|"length"|"tool_calls","prompt_tokens":N,"generated_tokens":N,"session_id":"..."}

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
}

fn default_max_tokens() -> usize { 4096 }
fn default_temperature() -> f32 { 0.0 }
fn default_top_p() -> f32 { 0.95 }

#[derive(Deserialize)]
struct RawInput {
    messages: String,
    #[serde(default)]
    tools: Option<String>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    stop: Option<String>,
    /// When set, the KV cache is saved under this name after generation.
    /// On follow-up turns, pass the same value to resume from the saved state.
    #[serde(default)]
    session_id: Option<String>,
    /// Number of messages already baked into the saved context. When
    /// resuming a session, only messages[resume_from..] are appended.
    #[serde(default)]
    resume_from: Option<usize>,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallIn>>,
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
struct FinalOutput {
    text: String,
    tool_calls: Vec<ToolCallOut>,
    stop_reason: String,
    prompt_tokens: usize,
    generated_tokens: usize,
}

#[derive(Serialize, Clone)]
struct ToolCallOut {
    id: String,
    name: String,
    arguments: String,
}

// ─── Entry point ───────────────────────────────────────────────────────────

#[inferlet::main]
async fn main(raw: RawInput) -> Result<String> {
    let messages: Vec<Message> = serde_json::from_str(&raw.messages)
        .map_err(|e| format!("failed to parse messages: {e}"))?;
    let tools_vec: Vec<ToolSpec> = match &raw.tools {
        Some(s) if !s.is_empty() => serde_json::from_str(s)
            .map_err(|e| format!("failed to parse tools: {e}"))?,
        _ => Vec::new(),
    };
    let stop: Vec<String> = match &raw.stop {
        Some(s) if !s.is_empty() => serde_json::from_str(s)
            .map_err(|e| format!("failed to parse stop: {e}"))?,
        _ => Vec::new(),
    };
    let input = Input {
        messages,
        tools: tools_vec,
        max_tokens: raw.max_tokens,
        temperature: raw.temperature,
        top_p: raw.top_p,
        stop,
    };
    let session_id = raw.session_id;
    let resume_from = raw.resume_from.unwrap_or(0);

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

    // Session persistence: if we have a session_id and resume_from > 0,
    // try to open the saved context and only append the new messages.
    let mut ctx = if resume_from > 0 {
        if let Some(ref sid) = session_id {
            match Context::open(&model, sid) {
                Ok(saved) => {
                    let mut ctx = saved;
                    let new_messages = &input.messages[resume_from.min(input.messages.len())..];
                    replay_incremental(&mut ctx, &model, new_messages, &tool_schemas)?;
                    ctx.cue();
                    ctx
                }
                Err(_) => {
                    // Snapshot missing (evicted/expired) — fall back to full replay.
                    let mut ctx = Context::new(&model)?;
                    replay_history(&mut ctx, &model, &input.messages, &tool_schemas)?;
                    ctx.cue();
                    ctx
                }
            }
        } else {
            let mut ctx = Context::new(&model)?;
            replay_history(&mut ctx, &model, &input.messages, &tool_schemas)?;
            ctx.cue();
            ctx
        }
    } else {
        let mut ctx = Context::new(&model)?;
        replay_history(&mut ctx, &model, &input.messages, &tool_schemas)?;
        ctx.cue();
        ctx
    };

    let prompt_token_count = ctx.seq_len() as usize + ctx.buffer().len();

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
    let mut chat_decoder = chat::Decoder::new(&model);

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut tool_calls: Vec<ToolCallOut> = Vec::new();
    let mut full_text = String::new();
    let mut stop_reason = "length";

    let mut g = ctx
        .generate(sampler)
        .max_tokens(input.max_tokens)
        .stop(&stop_token_ids);

    if let Some(matcher) = tools::native_matcher(&model, &tool_schemas) {
        g = g.constrain(inferlet::GrammarConstraint::new(matcher));
    }

    'outer: while let Some(step) = g.next()? {
        let out = step.execute().await?;

        for &t in &out.tokens {
            generated.push(t);

            // Stream text deltas via the chat decoder.
            match chat_decoder.feed(&[t])? {
                chat::Event::Delta(s) => {
                    full_text.push_str(&s);
                    let event = serde_json::json!({"type": "text_delta", "delta": s});
                    println!("{}", event);
                }
                chat::Event::Done(s) => {
                    full_text = s;
                    stop_reason = "stop";
                }
                _ => {}
            }

            if let Some(dec) = tool_decoder.as_mut() {
                if let tools::Event::Call(name, arguments) = dec.feed(&[t])? {
                    let tc = ToolCallOut {
                        id: format!("call_{}", tool_calls.len()),
                        name,
                        arguments,
                    };
                    let event = serde_json::json!({
                        "type": "tool_call",
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    });
                    println!("{}", event);
                    tool_calls.push(tc);
                }
            }

            if stop_token_ids.contains(&t) {
                stop_reason = "stop";
                break 'outer;
            }
        }

        if !input.stop.is_empty() {
            let max_stop_len = input.stop.iter().map(|s| s.len()).max().unwrap_or(0);
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

    let text = if tool_calls.is_empty() {
        trim_trailing_stop(&full_text, &input.stop).to_string()
    } else {
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

    // Save the context for session persistence if a session_id was provided.
    // The save includes all committed KV pages, so the next turn can resume
    // without replaying the full history.
    if let Some(ref sid) = session_id {
        let _ = ctx.save(sid);
    }

    let done_event = serde_json::json!({
        "type": "done",
        "stop_reason": stop_reason,
        "prompt_tokens": prompt_token_count,
        "generated_tokens": generated.len(),
        "session_id": session_id,
        "turn_message_count": input.messages.len(),
    });
    println!("{}", done_event);

    let output = FinalOutput {
        text,
        tool_calls,
        stop_reason: stop_reason.to_string(),
        prompt_tokens: prompt_token_count,
        generated_tokens: generated.len(),
    };
    Ok(serde_json::to_string(&output).unwrap())
}

// ─── History replay ────────────────────────────────────────────────────────

fn replay_history(
    ctx: &mut Context,
    model: &Model,
    messages: &[Message],
    tool_schemas: &[String],
) -> Result<()> {
    let mut equipped = false;
    let mut i = 0;

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

// ─── Incremental replay (session resume) ──────────────────────────────────

/// Append only the new messages to a resumed context. The saved snapshot
/// already contains the KV state for all prior turns, so we skip the
/// system prompt and tool equipping (already baked in) and just append
/// the tail of the conversation.
fn replay_incremental(
    ctx: &mut Context,
    model: &Model,
    new_messages: &[Message],
    _tool_schemas: &[String],
) -> Result<()> {
    for msg in new_messages {
        match msg.role.as_str() {
            "user" => {
                ctx.user(msg.content.as_deref().unwrap_or(""));
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
            }
            "tool" => {
                // Collect consecutive tool results into a batch.
                let batch = vec![(String::new(), msg.content.clone().unwrap_or_default())];
                ctx.append(&tools::answer_batch_prefix(model, &batch));
            }
            _ => {}
        }
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
