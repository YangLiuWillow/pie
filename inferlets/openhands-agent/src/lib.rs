//! SWE-Bench agent — full agent loop inside a Pie inferlet.
//!
//! Every generation step is constrained to a flat JSON schema via
//! `constrain_with(JsonSchema)`, so the model *must* produce a valid
//! `{thought, action, command, path, old_str, new_str, message}` object.
//! Tool execution is proxied to an external Python tool server over HTTP.
//!
//! Pie features exercised:
//!   - `constrain_with(JsonSchema)` — guaranteed structured output every step.
//!   - Per-step schema switching — FINISH_SCHEMA on last step.
//!   - `ctx.idle()` — yields GPU pages during HTTP tool calls.
//!   - KV-cache continuity — full conversation persists across all steps.
//!   - `wstd::http::Client` — outbound HTTP POST for tool execution.
//!   - Context condensation — saves a checkpoint after the system prompt
//!     and rebuilds from it when the context nears the model's limit.

use inferlet::{
    sample::Sampler, model::Model, runtime, Context, Result,
    wstd::http::{Client, Request, body::IntoBody},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    task: String,
    tool_server_url: String,
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default = "default_max_tokens")]
    max_tokens_per_step: usize,
    #[serde(default = "default_context_limit")]
    context_token_limit: u32,
    #[serde(default = "default_obs_limit")]
    max_observation_chars: usize,
}

fn default_max_steps() -> u32 { 50 }
fn default_max_tokens() -> usize { 16384 }
fn default_context_limit() -> u32 { 28000 }
fn default_obs_limit() -> usize { 8000 }

const MAX_CONSECUTIVE_FAILURES: u32 = 3;
const DEGENERATE_THRESHOLD: f64 = 0.4;
const CONDENSE_HEADROOM: u32 = 8000;

const SYSTEM_PROMPT: &str = "\
You are an expert software engineer solving a GitHub issue. Each turn you \
output one JSON object with these fields:

  {\"thought\": \"...\", \"action\": \"...\", \"command\": \"...\", \"path\": \"...\", \"old_str\": \"...\", \"new_str\": \"...\", \"message\": \"...\"}

Actions:
  - \"bash\": Run a shell command. Fill `command`; leave others empty.
  - \"edit\": Edit or create a file.
      * To replace text: fill `path`, `old_str` (exact unique match), `new_str`.
      * To create a file: fill `path` and `new_str`; leave `old_str` empty.
  - \"finish\": You are done. Fill `message` with a summary; leave others empty.

ALL fields must be present in every response (use \"\" for unused fields).

Examples:

User: Fix the bug in /workspace/django/utils/text.py
Assistant: {\"thought\": \"Let me look at the file first.\", \"action\": \"bash\", \"command\": \"cat -n /workspace/django/utils/text.py | head -50\", \"path\": \"\", \"old_str\": \"\", \"new_str\": \"\", \"message\": \"\"}

User: Observation:
     1  from functools import wraps
     2  def slugify(value):
     3      return value.lower().replace(' ', '-')
Assistant: {\"thought\": \"The slugify function doesn't handle unicode. I need to add unicode normalization.\", \"action\": \"edit\", \"command\": \"\", \"path\": \"/workspace/django/utils/text.py\", \"old_str\": \"def slugify(value):\\n    return value.lower().replace(' ', '-')\", \"new_str\": \"import unicodedata\\n\\ndef slugify(value):\\n    value = unicodedata.normalize('NFKD', value)\\n    return value.lower().replace(' ', '-')\", \"message\": \"\"}

User: Observation:
File edited: /workspace/django/utils/text.py
Assistant: {\"thought\": \"Let me verify the fix works.\", \"action\": \"bash\", \"command\": \"cd /workspace && python -c \\\"from django.utils.text import slugify; print(slugify('café'))\\\"\", \"path\": \"\", \"old_str\": \"\", \"new_str\": \"\", \"message\": \"\"}

User: Observation:
cafe
Assistant: {\"thought\": \"The fix works correctly. Unicode characters are now normalized.\", \"action\": \"finish\", \"command\": \"\", \"path\": \"\", \"old_str\": \"\", \"new_str\": \"\", \"message\": \"Fixed slugify to handle unicode by adding NFKD normalization.\"}

Guidelines:
  - Make minimal changes to fix the issue. Do not refactor unrelated code.
  - Do NOT modify test files — tests are already handled.
  - The development environment is already set up (dependencies installed).
  - Do NOT use interactive editors (nano, vim, vi, emacs). They will not \
work in this environment. Use the \"edit\" action or \
`python -c \"...\"` for file modifications.
  - Be thorough: read the problem, explore the code, create a reproduction, \
fix it, then verify.
  - IMPORTANT for edits: When adding code inside a function or method, your \
old_str MUST include the `def` line so the new code is placed inside the \
function body. Never insert code between a blank line and a `def` line — \
that places it outside the function.
  - Keep your edits concise. Avoid dumping entire files into old_str/new_str.

Follow these phases:
  1. READING: Read and understand the problem. Identify error messages, \
method names, file names, stack traces.
  2. EXPLORATION: Use grep/find to locate relevant files and code.
  3. TEST CREATION: Create a minimal reproduction script before fixing.
  4. FIX IMPLEMENTATION: Make the minimal edit to fix the issue.
  5. VERIFICATION: Run your reproduction script to confirm the fix. Run \
existing tests related to the modified code.
  6. FINAL REVIEW: Re-read the problem and ensure all requirements are met.";

const ACTION_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought":  { "type": "string", "minLength": 1 },
        "action":   { "type": "string", "enum": ["bash", "edit", "finish"] },
        "command":  { "type": "string" },
        "path":     { "type": "string" },
        "old_str":  { "type": "string" },
        "new_str":  { "type": "string" },
        "message":  { "type": "string" }
    },
    "required": ["thought", "action", "command", "path", "old_str", "new_str", "message"],
    "additionalProperties": false
}"#;

const FINISH_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought":  { "type": "string", "minLength": 1 },
        "action":   { "type": "string", "const": "finish" },
        "command":  { "type": "string" },
        "path":     { "type": "string" },
        "old_str":  { "type": "string" },
        "new_str":  { "type": "string" },
        "message":  { "type": "string", "minLength": 1 }
    },
    "required": ["thought", "action", "command", "path", "old_str", "new_str", "message"],
    "additionalProperties": false
}"#;

#[derive(Serialize)]
struct ToolRequest<'a> {
    action: &'a str,
    command: &'a str,
    path: &'a str,
    old_str: &'a str,
    new_str: &'a str,
}

/// A recorded turn for context condensation replay.
struct Turn {
    assistant_json: String,
    observation: String,
}

/// Returns true if the text has a high ratio of non-ASCII/control characters,
/// indicating the model has degenerated.
fn is_degenerate(text: &str) -> bool {
    if text.len() < 10 {
        return false;
    }
    let non_ascii = text.chars().filter(|c| !c.is_ascii() || c.is_control()).count();
    let total = text.chars().count().max(1);
    (non_ascii as f64 / total as f64) > DEGENERATE_THRESHOLD
}

/// Truncate an observation string to a character limit, keeping head and tail.
fn truncate_observation(obs: &str, max_chars: usize) -> String {
    if obs.len() <= max_chars {
        return obs.to_string();
    }
    let half = max_chars / 2;
    let head: String = obs.chars().take(half).collect();
    let tail: String = {
        let chars: Vec<char> = obs.chars().collect();
        let start = chars.len().saturating_sub(half);
        chars[start..].iter().collect()
    };
    let omitted = obs.len() - max_chars;
    format!("{head}\n\n... ({omitted} chars truncated) ...\n\n{tail}")
}

async fn call_tool_server(
    tool_server_url: &str,
    action: &str,
    command: &str,
    path: &str,
    old_str: &str,
    new_str: &str,
) -> std::result::Result<String, String> {
    let payload = ToolRequest { action, command, path, old_str, new_str };
    let body = serde_json::to_vec(&payload).map_err(|e| format!("serialize: {e}"))?;
    let uri = format!("{}/execute", tool_server_url);

    let request = Request::post(&uri)
        .header("Content-Type", "application/json")
        .body(body.into_body())
        .map_err(|e| format!("build request: {e}"))?;

    let response = Client::new()
        .send(request)
        .await
        .map_err(|e| format!("send request: {e}"))?;

    let mut resp_body = response.into_body();
    let mut buf = Vec::new();
    use inferlet::wstd::io::AsyncRead;
    resp_body
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("read response: {e}"))?;

    let resp: Value = serde_json::from_slice(&buf)
        .map_err(|e| format!("parse response ({}B): {e}", buf.len()))?;

    Ok(resp
        .get("observation")
        .and_then(Value::as_str)
        .unwrap_or("(no observation)")
        .to_string())
}

/// Rebuild the context from scratch using the saved system-prompt checkpoint,
/// replaying only the most recent turns to stay within the token budget.
fn condense_context(
    model: &Model,
    task: &str,
    history: &[Turn],
    context_token_limit: u32,
) -> Result<Context> {
    println!("[condense] rebuilding context ({} turns in history)", history.len());

    // Start with system + task to measure the fixed prefix cost.
    let mut ctx = Context::new(model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(task);

    // Estimate tokens consumed by the fixed prefix (rough: 4 chars/token).
    let prefix_tokens = ctx.buffer().len() as u32;
    let budget = context_token_limit.saturating_sub(prefix_tokens + CONDENSE_HEADROOM);

    // Walk backward through history, accumulating turns until we'd exceed budget.
    let mut replay_start = history.len();
    let mut est_tokens: u32 = 0;
    for (i, turn) in history.iter().enumerate().rev() {
        let turn_chars = turn.assistant_json.len() + turn.observation.len() + 30;
        let turn_tokens = (turn_chars / 4) as u32;
        if est_tokens + turn_tokens > budget {
            break;
        }
        est_tokens += turn_tokens;
        replay_start = i;
    }

    let dropped = replay_start;
    let kept = history.len() - dropped;
    if dropped > 0 {
        let summary = format!(
            "[Earlier conversation with {dropped} turns has been condensed. \
             The {kept} most recent turns follow.]"
        );
        ctx.user(&summary);
    }

    // Replay the kept turns.
    for turn in &history[replay_start..] {
        ctx.cue();
        ctx.assistant(&turn.assistant_json);
        ctx.seal();
        ctx.user(&format!("Observation:\n{}", turn.observation));
    }
    ctx.cue();

    println!("[condense] dropped {dropped}, replaying {kept} turns (est {est_tokens} tokens)");
    Ok(ctx)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&input.task);
    ctx.cue();

    let mut final_message: Option<String> = None;
    let mut consecutive_failures: u32 = 0;
    let mut history: Vec<Turn> = Vec::new();

    for step in 1..=input.max_steps {
        // Check if we need to condense before generating.
        let est_seq_len = ctx.seq_len() + ctx.buffer().len() as u32;
        if est_seq_len + CONDENSE_HEADROOM > input.context_token_limit && !history.is_empty() {
            ctx = condense_context(&model, &input.task, &history, input.context_token_limit)?;
        }

        let schema = if step == input.max_steps {
            FINISH_SCHEMA
        } else {
            ACTION_SCHEMA
        };

        let raw = ctx
            .generate(Sampler::Multinomial { temperature: 0.7, draws: 0 })
            .max_tokens(input.max_tokens_per_step)
            .constrain_with(inferlet::JsonSchema(schema))?
            .collect_text()
            .await?;

        let v = match serde_json::from_str::<Value>(&raw) {
            Ok(v) => v,
            Err(e) => {
                consecutive_failures += 1;
                println!("[step {step}] truncated at max_tokens ({e}), skipping (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})");
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    println!("[step {step}] too many consecutive failures, forcing finish");
                    break;
                }
                ctx.user("Observation: Your response was truncated because it was too long. Be more concise — use smaller edits and shorter commands.");
                ctx.cue();
                continue;
            }
        };

        let thought = v.get("thought").and_then(Value::as_str).unwrap_or("");
        let action = v.get("action").and_then(Value::as_str).unwrap_or("");
        let command = v.get("command").and_then(Value::as_str).unwrap_or("");
        let path = v.get("path").and_then(Value::as_str).unwrap_or("");
        let old_str = v.get("old_str").and_then(Value::as_str).unwrap_or("");
        let new_str = v.get("new_str").and_then(Value::as_str).unwrap_or("");
        let message = v.get("message").and_then(Value::as_str).unwrap_or("");

        // Detect degenerate output (garbled CJK/symbol soup).
        if is_degenerate(thought) {
            consecutive_failures += 1;
            println!("[step {step}] degenerate output detected (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})");
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                println!("[step {step}] too many degenerate outputs, forcing finish");
                break;
            }
            // Condense and retry — degeneration usually means context overflow.
            if !history.is_empty() {
                ctx = condense_context(&model, &input.task, &history, input.context_token_limit)?;
            }
            continue;
        }

        consecutive_failures = 0;
        println!("[step {step}] thought: {thought}");
        println!("[step {step}] action: {action}");

        if action == "finish" {
            println!("[step {step}] message: {message}");
            final_message = Some(message.to_string());
            break;
        }

        let _idle = ctx.idle();
        let observation = match call_tool_server(
            &input.tool_server_url,
            action,
            command,
            path,
            old_str,
            new_str,
        )
        .await
        {
            Ok(obs) => obs,
            Err(e) => format!("Tool server error: {e}"),
        };
        drop(_idle);

        let observation = truncate_observation(&observation, input.max_observation_chars);
        println!("[step {step}] observation ({} chars)", observation.len());

        history.push(Turn {
            assistant_json: raw.clone(),
            observation: observation.clone(),
        });

        ctx.user(&format!("Observation:\n{observation}"));
        ctx.cue();
    }

    match &final_message {
        Some(m) => println!("\nAgent finished: {m}"),
        None => println!("\nAgent did not finish within {max} steps", max = input.max_steps),
    }

    let result = serde_json::json!({
        "finished": final_message.is_some(),
        "message": final_message.unwrap_or_default(),
        "steps": history.len(),
    });
    Ok(result.to_string())
}
