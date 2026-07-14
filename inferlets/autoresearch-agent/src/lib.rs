//! Autoresearch agent — autonomous ML research loop inside a Pie inferlet.
//!
//! Implements Karpathy's autoresearch pattern: edit train.py -> run experiment
//! -> check val_bpb -> keep or discard -> repeat.

use inferlet::{
    sample::Sampler, model::Model, runtime, Context, Result,
    wstd::http::{Client, Request, body::IntoBody},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    program: String,
    tool_server_url: String,
    #[serde(default = "default_max_experiments")]
    max_experiments: u32,
    #[serde(default = "default_max_tokens")]
    max_tokens_per_step: usize,
    #[serde(default = "default_context_limit")]
    context_token_limit: u32,
    #[serde(default = "default_obs_limit")]
    max_observation_chars: usize,
}

fn default_max_experiments() -> u32 { 100 }
fn default_max_tokens() -> usize { 4096 }
fn default_context_limit() -> u32 { 28000 }
fn default_obs_limit() -> usize { 8000 }

const MAX_CONSECUTIVE_FAILURES: u32 = 3;
const DEGENERATE_THRESHOLD: f64 = 0.4;
const CONDENSE_HEADROOM: u32 = 8000;

const SYSTEM_PROMPT: &str = "You are an autonomous ML research agent. Your goal is to iteratively improve a training script (`train.py`) by running experiments and tracking results.\n\nEach turn you output one JSON object with these fields:\n\n  {\"thought\": \"...\", \"action\": \"...\", \"command\": \"...\", \"path\": \"...\", \"old_str\": \"...\", \"new_str\": \"...\", \"message\": \"...\"}\n\nActions:\n  - \"bash\": Run a shell command. Fill `command`; leave others empty.\n  - \"edit\": Edit or create a file.\n      * To replace text: fill `path`, `old_str` (exact unique match), `new_str`.\n      * To create a file: fill `path` and `new_str`; leave `old_str` empty.\n  - \"read_file\": Read a file. Fill `path`; leave others empty.\n  - \"finish\": You are done. Fill `message` with a summary of all experiments and the best val_bpb achieved; leave others empty.\n\nALL fields must be present in every response (use \"\" for unused fields).\n\nWorkflow for each experiment:\n  1. Study current train.py and results so far.\n  2. Form a hypothesis and make a targeted edit to train.py.\n  3. Commit before running: `git add -A && git commit -m \"experiment: <description>\"`.\n  4. Run the experiment: `uv run train.py > run.log 2>&1`.\n  5. Extract the result: `grep \"^val_bpb:\" run.log`.\n  6. If val_bpb improved, advance the branch: `echo \"<exp#>\\t<description>\\t<val_bpb>\" >> results.tsv`.\n  7. If val_bpb did NOT improve, reset: `git reset --hard HEAD~1`.\n  8. Repeat with a new hypothesis.\n\nGuidelines:\n  - Only modify `train.py` unless the program instructions say otherwise.\n  - Each experiment should test ONE hypothesis. Do not make multiple changes at once.\n  - Always commit before running so you can cleanly revert failed experiments.\n  - Be systematic: try the most impactful changes first (architecture, lr, batch size) before fine-tuning.\n  - Read run.log carefully for errors or warnings before concluding an experiment failed.\n  - Keep a running mental model of what you have tried and what worked.";

const ACTION_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought":  { "type": "string", "minLength": 1 },
        "action":   { "type": "string", "enum": ["bash", "edit", "read_file", "finish"] },
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

struct Turn {
    assistant_json: String,
    observation: String,
}

fn is_degenerate(text: &str) -> bool {
    if text.len() < 10 {
        return false;
    }
    let non_ascii = text.chars().filter(|c| !c.is_ascii() || c.is_control()).count();
    let total = text.chars().count().max(1);
    (non_ascii as f64 / total as f64) > DEGENERATE_THRESHOLD
}

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

fn condense_context(
    model: &Model,
    task: &str,
    history: &[Turn],
    context_token_limit: u32,
) -> Result<Context> {
    println!("[condense] rebuilding context ({} turns in history)", history.len());

    let mut ctx = Context::new(model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(task);

    let prefix_tokens = ctx.buffer().len() as u32;
    let budget = context_token_limit.saturating_sub(prefix_tokens + CONDENSE_HEADROOM);

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

    let task_prompt = format!(
        "Here are your research instructions (program.md):\n\n{}\n\nBegin your research.",
        input.program
    );

    let mut ctx = Context::new(&model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&task_prompt);
    ctx.cue();

    let mut final_message: Option<String> = None;
    let mut consecutive_failures: u32 = 0;
    let mut history: Vec<Turn> = Vec::new();

    for step in 1..=input.max_experiments {
        let est_seq_len = ctx.seq_len() + ctx.buffer().len() as u32;
        if est_seq_len + CONDENSE_HEADROOM > input.context_token_limit && !history.is_empty() {
            ctx = condense_context(&model, &task_prompt, &history, input.context_token_limit)?;
        }

        let schema = if step == input.max_experiments {
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
                ctx.user("Observation: Your response was truncated because it was too long. Be more concise.");
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

        if is_degenerate(thought) {
            consecutive_failures += 1;
            println!("[step {step}] degenerate output detected (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})");
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                println!("[step {step}] too many degenerate outputs, forcing finish");
                break;
            }
            if !history.is_empty() {
                ctx = condense_context(&model, &task_prompt, &history, input.context_token_limit)?;
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

        // For read_file, convert to a bash cat command.
        let (effective_action, effective_command) = if action == "read_file" {
            ("bash", format!("cat -n {}", path))
        } else {
            (action, command.to_string())
        };

        let _idle = ctx.idle();
        let observation = match call_tool_server(
            &input.tool_server_url,
            effective_action,
            &effective_command,
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
        None => println!("\nAgent did not finish within {max} steps", max = input.max_experiments),
    }

    let result = serde_json::json!({
        "finished": final_message.is_some(),
        "message": final_message.unwrap_or_default(),
        "steps": history.len(),
    });
    Ok(result.to_string())
}
