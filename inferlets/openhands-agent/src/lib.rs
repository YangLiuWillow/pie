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

use std::time::Instant;

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
    #[serde(default = "default_max_empty_finishes")]
    max_empty_finishes: u32,
    /// Test-time scaling: number of candidate branches to fork (default 1 =
    /// single trajectory, fully backward-compatible).
    #[serde(default = "default_num_branches")]
    num_branches: usize,
    /// Fork before generating step `branch_at_step + 1`. All branches share
    /// the identical prefix (steps 1..=branch_at_step) and diverge from there.
    /// 0 = fork at the very start (only system+task shared). Only used when
    /// num_branches > 1.
    #[serde(default)]
    branch_at_step: u32,
    /// Sampling temperature. 0 = greedy (Argmax). Branches only diverge with
    /// temperature > 0 (forked greedy contexts generate identically).
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// Context condensation strategy: "rebuild" (default — summarize dropped
    /// turns + re-prefill a fresh context) or "mask" (drop stale middle turns
    /// by masking their KV out of attention — 0 re-prefill, the B2-mask win).
    #[serde(default = "default_condense_mode")]
    condense_mode: String,
    /// mask mode: number of most-recent turns to keep attended (older middle
    /// turns are masked).
    #[serde(default = "default_condense_keep_recent")]
    condense_keep_recent: u32,
    /// Capture mode: when true (single-trajectory only), emit the full recorded
    /// trajectory (per-turn assistant JSON + observation strings + turn_starts)
    /// in the output so it can be REPLAYED offline through both condensers over
    /// the identical token stream — removing the layer-B nondeterminism confound
    /// that made the live A/B (job 19059956) run two different trajectories.
    #[serde(default)]
    dump_trajectory: bool,
}

fn default_max_steps() -> u32 { 50 }
fn default_max_tokens() -> usize { 16384 }
fn default_context_limit() -> u32 { 28000 }
fn default_obs_limit() -> usize { 8000 }
fn default_max_empty_finishes() -> u32 { 3 }
fn default_num_branches() -> usize { 1 }
fn default_top_p() -> f32 { 1.0 }
fn default_condense_mode() -> String { "rebuild".to_string() }
fn default_condense_keep_recent() -> u32 { 12 }

const MAX_CONSECUTIVE_FAILURES: u32 = 3;
const DEGENERATE_THRESHOLD: f64 = 0.4;
const CONDENSE_HEADROOM: u32 = 8000;
const STUCK_WINDOW: usize = 5;
// After condensing, skip the next N condensation checks to avoid pathological
// condense-every-step loops when turns fill right up to the budget.
const CONDENSE_COOLDOWN: u32 = 5;
// When condensing, target this fraction of the budget — leaving room for
// several more steps before the next condensation trigger.
const CONDENSE_TARGET_FRAC: f64 = 0.70;

const SYSTEM_PROMPT: &str = "\
You are an expert software engineer solving a GitHub issue. Each turn you \
output one JSON object with these fields:

  {\"thought\": \"...\", \"action\": \"...\", \"command\": \"...\", \"path\": \"...\", \"old_str\": \"...\", \"new_str\": \"...\", \"insert_line\": 0, \"start_line\": 0, \"end_line\": 0, \"message\": \"...\"}

Actions:
  - \"bash\": Run a shell command. Fill `command`; leave others empty/0.
  - \"edit\": Edit or create a file.
      * To replace text: fill `path`, `old_str` (exact unique match), `new_str`.
      * To create a file: fill `path` and `new_str`; leave `old_str` empty.
  - \"read_file\": Read a file with line numbers. Fill `path`. \
Optionally set `start_line` and `end_line` to view only a range of lines \
(1-indexed). Leave them as 0 to view the entire file.
  - \"insert\": Insert text at a specific line number. Fill `path`, \
`insert_line` (line number to insert AFTER), and `new_str`.
  - \"undo_edit\": Undo the last edit to a file. Fill `path`; leave others empty/0.
  - \"finish\": You are done. Fill `message` with a summary; leave others empty/0.

ALL fields must be present in every response (use \"\" for unused string fields, 0 for unused number fields).

Examples:

User: Fix the bug in /workspace/django/utils/text.py
Assistant: {\"thought\": \"Let me look at the file first.\", \"action\": \"read_file\", \"command\": \"\", \"path\": \"/workspace/django/utils/text.py\", \"old_str\": \"\", \"new_str\": \"\", \"insert_line\": 0, \"start_line\": 0, \"end_line\": 0, \"message\": \"\"}

User: Observation:
     1\tfrom functools import wraps
     2\tdef slugify(value):
     3\t    return value.lower().replace(' ', '-')
Assistant: {\"thought\": \"The slugify function doesn't handle unicode. I need to add unicode normalization. I'll use the exact text from the file for old_str.\", \"action\": \"edit\", \"command\": \"\", \"path\": \"/workspace/django/utils/text.py\", \"old_str\": \"def slugify(value):\\n    return value.lower().replace(' ', '-')\", \"new_str\": \"import unicodedata\\n\\ndef slugify(value):\\n    value = unicodedata.normalize('NFKD', value)\\n    return value.lower().replace(' ', '-')\", \"insert_line\": 0, \"start_line\": 0, \"end_line\": 0, \"message\": \"\"}

User: Observation:
File edited: /workspace/django/utils/text.py
Context:
     1\timport unicodedata
     2\t
     3\tdef slugify(value):
     4\t    value = unicodedata.normalize('NFKD', value)
     5\t    return value.lower().replace(' ', '-')
Assistant: {\"thought\": \"Let me verify the fix works.\", \"action\": \"bash\", \"command\": \"cd /workspace && python -c \\\"from django.utils.text import slugify; print(slugify('café'))\\\"\", \"path\": \"\", \"old_str\": \"\", \"new_str\": \"\", \"insert_line\": 0, \"start_line\": 0, \"end_line\": 0, \"message\": \"\"}

User: Observation:
cafe
Assistant: {\"thought\": \"The fix works correctly. Unicode characters are now normalized.\", \"action\": \"finish\", \"command\": \"\", \"path\": \"\", \"old_str\": \"\", \"new_str\": \"\", \"insert_line\": 0, \"start_line\": 0, \"end_line\": 0, \"message\": \"Fixed slugify to handle unicode by adding NFKD normalization.\"}

For large files, use read_file with start_line/end_line to view specific sections:
Assistant: {\"thought\": \"The file is large, let me view lines 200-250.\", \"action\": \"read_file\", \"command\": \"\", \"path\": \"/workspace/sympy/core/operations.py\", \"old_str\": \"\", \"new_str\": \"\", \"insert_line\": 0, \"start_line\": 200, \"end_line\": 250, \"message\": \"\"}

To insert new code at a specific line:
Assistant: {\"thought\": \"I need to add an import at line 5.\", \"action\": \"insert\", \"command\": \"\", \"path\": \"/workspace/foo.py\", \"old_str\": \"\", \"new_str\": \"import os\", \"insert_line\": 5, \"start_line\": 0, \"end_line\": 0, \"message\": \"\"}

Guidelines:
  - Make minimal, surgical changes to fix the issue. Do not refactor unrelated code.
  - Do NOT modify test files — tests are already handled.
  - The development environment is already set up (dependencies installed).
  - Do NOT use interactive editors (nano, vim, vi, emacs). They will not \
work in this environment.

CRITICAL edit rules:
  - ALWAYS use read_file to see the exact file content BEFORE attempting an edit. \
Copy the exact text from the file for old_str — do not type it from memory.
  - If an edit fails with \"old_str not found\", use read_file with start_line/end_line \
to see the actual content around your target, then copy the exact text for old_str.
  - old_str must be at most 50 lines. For larger changes, break into multiple edits.
  - new_str must be at most 100 lines. For larger changes, break into multiple edits.
  - NEVER replace an entire function/class/module. Only replace the specific lines that need to change.
  - When adding code inside a function, your old_str MUST include the `def` line \
so the new code is placed inside the function body.
  - If str_replace keeps failing, try: (a) use read_file with start_line/end_line to \
see the exact target lines, (b) use insert to add code at a line number, or \
(c) use undo_edit to revert and try again.

CRITICAL fix-quality rules:
  - Fix the ROOT CAUSE, not the symptom. If a function returns a wrong value, \
trace backward through the code to find WHERE the wrong value is produced. \
The bug is usually upstream of where the error manifests — in the logic that \
computes the value, not in the code that uses it.
  - NEVER add an if-guard or special case at the error site without first \
understanding why the bad state occurs. A guard that papers over the symptom \
(e.g., \"if x >= n: return fallback\") will fail the project's tests because \
it does not fix the underlying logic.
  - Before writing your fix, state the root cause in your thought: which \
variable has the wrong value, why, and which line of code is responsible.
  - If the fix is a one-line guard at the crash/error site, it is almost \
certainly wrong. Look upstream.

Follow these phases:
  1. READING: Read and understand the problem. Identify error messages, \
method names, file names, stack traces.
  2. EXPLORATION: Use grep/find to locate relevant files and code. Use \
read_file to examine them. Trace the data flow from the bug's origin to \
where the symptom appears — the fix belongs at the origin.
  3. TEST CREATION: Create a minimal reproduction script that demonstrates \
the bug (e.g., prints wrong output or raises the error). Run it to confirm \
it fails.
  4. ROOT CAUSE ANALYSIS: Before coding the fix, identify the exact line(s) \
that produce the wrong behavior. Add debug prints or read surrounding code \
to confirm your diagnosis.
  5. FIX IMPLEMENTATION: Use read_file to see exact content, then make \
the minimal edit to fix the root cause.
  6. VERIFICATION: Run your reproduction script to confirm the fix produces \
the correct output. Then find and run the existing test suite for the module \
you changed (e.g., `python -m pytest path/to/tests/ -x -q`). If tests fail, \
your fix is wrong — go back to step 4.
  7. FINAL REVIEW: Re-read the problem and ensure all requirements are met. \
Confirm `git diff` shows your changes.";

const ACTION_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought":     { "type": "string", "minLength": 1 },
        "action":      { "type": "string", "enum": ["bash", "edit", "read_file", "insert", "undo_edit", "finish"] },
        "command":     { "type": "string" },
        "path":        { "type": "string" },
        "old_str":     { "type": "string" },
        "new_str":     { "type": "string" },
        "insert_line": { "type": "integer" },
        "start_line":  { "type": "integer" },
        "end_line":    { "type": "integer" },
        "message":     { "type": "string" }
    },
    "required": ["thought", "action", "command", "path", "old_str", "new_str", "insert_line", "start_line", "end_line", "message"],
    "additionalProperties": false
}"#;

const FINISH_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought":     { "type": "string", "minLength": 1 },
        "action":      { "type": "string", "const": "finish" },
        "command":     { "type": "string" },
        "path":        { "type": "string" },
        "old_str":     { "type": "string" },
        "new_str":     { "type": "string" },
        "insert_line": { "type": "integer" },
        "start_line":  { "type": "integer" },
        "end_line":    { "type": "integer" },
        "message":     { "type": "string", "minLength": 1 }
    },
    "required": ["thought", "action", "command", "path", "old_str", "new_str", "insert_line", "start_line", "end_line", "message"],
    "additionalProperties": false
}"#;

#[derive(Serialize)]
struct ToolRequest<'a> {
    action: &'a str,
    command: &'a str,
    path: &'a str,
    old_str: &'a str,
    new_str: &'a str,
    insert_line: i64,
    start_line: i64,
    end_line: i64,
    workspace_id: &'a str,
}

#[derive(Serialize, Clone)]
struct StepMetrics {
    step: u32,
    action: String,
    generate_s: f64,
    tool_s: f64,
    prompt_tokens: u32,
    completion_tokens: u32,
    seq_len_after: u32,
}

/// A recorded turn for context condensation replay.
#[derive(Clone)]
struct Turn {
    assistant_json: String,
    observation: String,
}

#[derive(Clone)]
struct RecentAction {
    action: String,
    command: String,
    path: String,
    failed: bool,
}

/// Detect stuck patterns in the last N actions.
/// Returns a hint string if the agent is cycling, None otherwise.
fn detect_stuck(recent: &[RecentAction]) -> Option<&'static str> {
    if recent.len() < 3 {
        return None;
    }
    let last = recent.len();

    // Pattern 1: same (action, path) with failures 3+ times in a row.
    // e.g., edit /foo → fail, edit /foo → fail, edit /foo → fail
    let tail3 = &recent[last.saturating_sub(3)..];
    if tail3.len() == 3
        && tail3.iter().all(|a| a.failed)
        && tail3.iter().all(|a| a.action == tail3[0].action && a.path == tail3[0].path)
    {
        if tail3[0].action == "edit" {
            return Some(
                "STUCK: You have tried the same edit 3 times and it keeps failing. \
                 Try a different approach: use `bash` with `sed` to make the change, \
                 or use read_file with a different line range to see the exact content."
            );
        }
        return Some(
            "STUCK: You have repeated the same failing action 3 times. \
             Stop and try a completely different approach."
        );
    }

    // Pattern 2: read_file/edit cycle on the same path.
    // e.g., read /foo, edit /foo (fail), read /foo, edit /foo (fail)
    let tail4 = &recent[last.saturating_sub(4)..];
    if tail4.len() == 4 {
        let is_cycle = tail4[0].action == "read_file"
            && tail4[1].action == "edit"
            && tail4[1].failed
            && tail4[2].action == "read_file"
            && tail4[3].action == "edit"
            && tail4[3].failed
            && tail4.iter().all(|a| a.path == tail4[0].path);
        if is_cycle {
            return Some(
                "STUCK: You are cycling between read_file and edit on the same file, \
                 but the edit keeps failing. Try using `bash` with `sed -i` to make \
                 the change directly, e.g.: sed -i 's/old text/new text/' /path/to/file"
            );
        }
    }

    // Pattern 3: same bash command repeated 3+ times.
    if tail3.len() == 3
        && tail3.iter().all(|a| a.action == "bash" && !a.command.is_empty())
        && tail3.iter().all(|a| a.command == tail3[0].command)
    {
        return Some(
            "STUCK: You have run the exact same bash command 3 times. \
             The output is not changing. Try a different command or approach."
        );
    }

    // Pattern 4: alternating between two actions (A, B, A, B) with failures.
    if tail4.len() == 4 {
        let alternating = tail4[0].action == tail4[2].action
            && tail4[1].action == tail4[3].action
            && tail4[0].action != tail4[1].action
            && tail4.iter().filter(|a| a.failed).count() >= 2;
        if alternating {
            return Some(
                "STUCK: You are alternating between two actions without making progress. \
                 Step back, re-read the problem statement, and try a completely different \
                 approach to locate and fix the issue."
            );
        }
    }

    // Pattern 5: 5 consecutive actions on the same path with at least 3 failures.
    let tail5 = &recent[last.saturating_sub(STUCK_WINDOW)..];
    if tail5.len() == STUCK_WINDOW
        && tail5.iter().all(|a| a.path == tail5[0].path && !a.path.is_empty())
        && tail5.iter().filter(|a| a.failed).count() >= 3
    {
        return Some(
            "STUCK: You have been working on the same file for many steps without \
             success. Step back and reconsider your approach. Can you use `bash` to \
             make the change with sed, or is there a different file you should edit?"
        );
    }

    None
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
    workspace_id: &str,
    action: &str,
    command: &str,
    path: &str,
    old_str: &str,
    new_str: &str,
    insert_line: i64,
    start_line: i64,
    end_line: i64,
) -> std::result::Result<String, String> {
    let payload = ToolRequest { action, command, path, old_str, new_str, insert_line, start_line, end_line, workspace_id };
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

async fn check_has_diff(tool_server_url: &str, workspace_id: &str) -> bool {
    let payload = ToolRequest {
        action: "has_diff",
        command: "",
        path: "",
        old_str: "",
        new_str: "",
        insert_line: 0,
        start_line: 0,
        end_line: 0,
        workspace_id,
    };
    let body = match serde_json::to_vec(&payload) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let uri = format!("{}/execute", tool_server_url);
    let request = match Request::post(&uri)
        .header("Content-Type", "application/json")
        .body(body.into_body())
    {
        Ok(r) => r,
        Err(_) => return false,
    };
    let response = match Client::new().send(request).await {
        Ok(r) => r,
        Err(_) => return false,
    };
    let mut resp_body = response.into_body();
    let mut buf = Vec::new();
    use inferlet::wstd::io::AsyncRead;
    if resp_body.read_to_end(&mut buf).await.is_err() {
        return false;
    }
    let resp: Value = match serde_json::from_slice(&buf) {
        Ok(v) => v,
        Err(_) => return false,
    };
    resp.get("has_diff").and_then(Value::as_bool).unwrap_or(false)
}

const SUMMARIZE_PROMPT: &str = "\
Summarize this coding-assistant conversation concisely. Include:
1. Files examined and their relevant sections (paths, line numbers)
2. Root cause or key findings so far
3. Changes made (edits, file creations) — include exact paths
4. What worked and what failed
5. Current state and likely next steps

Be specific about file paths, function/class names, and line numbers. \
Under 400 words.";

const MAX_SUMMARY_INPUT_CHARS: usize = 40_000;
const MAX_TURN_CHARS_FOR_SUMMARY: usize = 2000;

fn safe_truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

async fn summarize_dropped_turns(
    model: &Model,
    turns: &[Turn],
) -> Result<String> {
    let mut ctx = Context::new(model)?;
    ctx.system(SUMMARIZE_PROMPT);

    let mut conversation = String::new();
    let mut total_chars = 0;
    for (i, turn) in turns.iter().enumerate() {
        let turn_text = format!(
            "--- Step {} ---\nAssistant: {}\nObservation:\n{}\n\n",
            i + 1,
            safe_truncate(&turn.assistant_json, MAX_TURN_CHARS_FOR_SUMMARY),
            safe_truncate(&turn.observation, MAX_TURN_CHARS_FOR_SUMMARY),
        );
        total_chars += turn_text.len();
        if total_chars > MAX_SUMMARY_INPUT_CHARS {
            conversation.push_str(&format!(
                "(... {} more steps omitted ...)\n",
                turns.len() - i
            ));
            break;
        }
        conversation.push_str(&turn_text);
    }
    ctx.user(&conversation);
    ctx.cue();

    let summary = ctx
        .generate(Sampler::Argmax)
        .max_tokens(1024)
        .collect_text()
        .await?;

    Ok(summary)
}

/// Rebuild the context from scratch, using an LLM-generated summary of
/// dropped turns to preserve context from early exploration.
async fn condense_context(
    model: &Model,
    task: &str,
    history: &[Turn],
    context_token_limit: u32,
) -> Result<Option<Context>> {
    println!("[condense] rebuilding context ({} turns in history)", history.len());

    // Start with system + task to measure the fixed prefix cost.
    let mut ctx = Context::new(model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(task);

    // Estimate tokens consumed by the fixed prefix.
    // buffer().len() returns chars; divide by 3 for a conservative token estimate
    // (most code/JSON tokenizes at ~3-3.5 chars/token, not 4).
    let prefix_tokens = (ctx.buffer().len() / 3) as u32;
    // Reserve 1200 tokens for the LLM summary.
    let summary_budget: u32 = 1200;
    let budget = context_token_limit
        .saturating_sub(prefix_tokens + CONDENSE_HEADROOM + summary_budget);

    // Target a fraction of the budget so there's headroom for several more
    // steps before the next condensation trigger.
    let target_budget = (budget as f64 * CONDENSE_TARGET_FRAC) as u32;

    // Walk backward through history, accumulating turns until we'd exceed budget.
    let mut replay_start = history.len();
    let mut est_tokens: u32 = 0;
    for (i, turn) in history.iter().enumerate().rev() {
        let turn_chars = turn.assistant_json.len() + turn.observation.len() + 30;
        let turn_tokens = (turn_chars / 3) as u32;
        if est_tokens + turn_tokens > target_budget {
            break;
        }
        est_tokens += turn_tokens;
        replay_start = i;
    }

    let dropped = replay_start;
    let kept = history.len() - dropped;

    // Nothing to drop → skip the expensive rebuild entirely.
    if dropped == 0 {
        println!("[condense] nothing to drop ({kept} turns fit), skipping rebuild");
        return Ok(None);
    }

    {
        let dropped_turns = &history[..replay_start];
        let summary = match summarize_dropped_turns(model, dropped_turns).await {
            Ok(s) => {
                println!("[condense] LLM summary generated ({} chars)", s.len());
                s
            }
            Err(e) => {
                println!("[condense] summarization failed ({e}), using simple note");
                format!(
                    "Earlier conversation explored the codebase for {dropped} steps. \
                     Details were condensed due to context limits."
                )
            }
        };

        // Rebuild context with the LLM summary.
        ctx = Context::new(model)?;
        ctx.system(SYSTEM_PROMPT);
        ctx.user(task);
        ctx.user(&format!(
            "[Summary of earlier exploration ({dropped} steps)]\n{summary}"
        ));
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
    Ok(Some(ctx))
}

/// Immutable per-run configuration shared by all branches.
struct BranchCfg<'a> {
    model: &'a Model,
    task: &'a str,
    tool_server_url: &'a str,
    max_steps: u32,
    max_tokens_per_step: usize,
    context_token_limit: u32,
    max_observation_chars: usize,
    max_empty_finishes: u32,
    temperature: f32,
    top_p: f32,
    condense_mode: CondenseMode,
    condense_keep_recent: u32,
    dump_trajectory: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum CondenseMode { Rebuild, Mask }

/// Mutable loop state carried across a suspend/resume (fork) boundary and
/// cloned once per branch at the fork point.
#[derive(Default, Clone)]
struct LoopState {
    consecutive_failures: u32,
    empty_finishes: u32,
    history: Vec<Turn>,
    recent_actions: Vec<RecentAction>,
    ran_tests_after_edit: bool,
    test_nudged: bool,
    condense_cooldown: u32,
    step_metrics: Vec<StepMetrics>,
    total_prompt_tokens: u32,
    total_completion_tokens: u32,
    /// mask-mode: KV start position of each recorded history turn (parallel to
    /// `history`). Used to compute the drop-middle mask range.
    turn_starts: Vec<u32>,
}

/// Effective (attended) context length = resident seq_len minus masked tokens.
fn effective_len(ctx: &Context) -> u32 {
    let masked: u32 = ctx.masked_ranges().iter().map(|(s, e)| e - s).sum();
    (ctx.seq_len() + ctx.buffer().len() as u32).saturating_sub(masked)
}

/// mask-mode condensation: keep the fixed prefix (system+task) + the last
/// `keep_recent` turns attended; mask the middle turns' KV out of attention.
/// No rebuild, no re-prefill. Returns true if a new range was masked.
fn mask_condense(ctx: &mut Context, turn_starts: &[u32], keep_recent: u32) -> bool {
    let n = turn_starts.len();
    let keep = keep_recent as usize;
    if n <= keep || turn_starts.is_empty() {
        return false;
    }
    let prefix_end = turn_starts[0]; // start of the first turn = end of system+task
    let kept_start = turn_starts[n - keep]; // start of the first kept recent turn
    if kept_start <= prefix_end {
        return false;
    }
    ctx.mask_range(prefix_end, kept_start);
    println!(
        "[condense-mask] masked KV [{prefix_end}, {kept_start}) ({} turns dropped, {keep} kept)",
        n - keep,
    );
    true
}

#[derive(Serialize)]
struct BranchResult {
    workspace_id: String,
    finished: bool,
    message: String,
    steps: usize,
    total_wall_s: f64,
    total_generate_s: f64,
    total_tool_s: f64,
    total_prompt_tokens: u32,
    total_completion_tokens: u32,
    #[serde(rename = "per_step")]
    step_metrics: Vec<StepMetrics>,
    /// Captured trajectory for offline replay (only populated when
    /// `dump_trajectory`); empty otherwise so the normal output shape is
    /// unchanged.
    #[serde(skip_serializing, default)]
    history: Vec<TurnDump>,
    #[serde(skip_serializing, default)]
    turn_starts: Vec<u32>,
}

#[derive(Serialize, Default)]
struct TurnDump {
    assistant: String,
    observation: String,
}

enum BranchOutcome {
    Finished(BranchResult),
    /// Trunk hit the branch point: hand back the live context + state so the
    /// caller can fork K ways and resume each from `next_step`.
    Suspended { ctx: Context, state: LoopState, next_step: u32 },
}

fn make_sampler(temperature: f32, top_p: f32) -> Sampler {
    if temperature > 0.0 {
        Sampler::TopP { temperature, p: top_p }
    } else {
        Sampler::Argmax
    }
}

/// Snapshot the `from_id` workspace into a fresh copy per `to_id`.
async fn fork_workspace(
    tool_server_url: &str,
    from_id: &str,
    to_ids: &[String],
) -> std::result::Result<(), String> {
    let payload = serde_json::json!({
        "action": "fork_workspace",
        "from_id": from_id,
        "to_ids": to_ids,
    });
    let body = serde_json::to_vec(&payload).map_err(|e| format!("serialize: {e}"))?;
    let uri = format!("{}/execute", tool_server_url);
    let request = Request::post(&uri)
        .header("Content-Type", "application/json")
        .body(body.into_body())
        .map_err(|e| format!("build request: {e}"))?;
    let response = Client::new().send(request).await.map_err(|e| format!("send: {e}"))?;
    let mut resp_body = response.into_body();
    let mut buf = Vec::new();
    use inferlet::wstd::io::AsyncRead;
    resp_body.read_to_end(&mut buf).await.map_err(|e| format!("read: {e}"))?;
    let resp: Value = serde_json::from_slice(&buf).map_err(|e| format!("parse: {e}"))?;
    if resp.get("exit_code").and_then(Value::as_i64).unwrap_or(0) != 0 {
        return Err(resp.get("observation").and_then(Value::as_str).unwrap_or("fork failed").to_string());
    }
    Ok(())
}

/// Run one agent trajectory from `start_step` on the given `workspace_id`.
/// If `branch_at_step > 0`, suspend (return the live ctx + state) right before
/// generating step `branch_at_step + 1`, so the caller can fork.
async fn run_branch(
    cfg: &BranchCfg<'_>,
    mut ctx: Context,
    workspace_id: &str,
    mut state: LoopState,
    start_step: u32,
    branch_at_step: u32,
) -> Result<BranchOutcome> {
    let branch_start = Instant::now();
    let mut final_message: Option<String> = None;

    for step in start_step..=cfg.max_steps {
        // Fork point: suspend before generating step (branch_at_step + 1).
        if branch_at_step > 0 && step > branch_at_step {
            return Ok(BranchOutcome::Suspended { ctx, state, next_step: step });
        }

        // Check if we need to condense before generating. Effective length
        // subtracts already-masked KV (mask mode), so a masked context doesn't
        // re-trigger every step.
        if state.condense_cooldown > 0 {
            state.condense_cooldown -= 1;
        }
        let est_seq_len = effective_len(&ctx);
        if state.condense_cooldown == 0
            && est_seq_len + CONDENSE_HEADROOM > cfg.context_token_limit
            && !state.history.is_empty()
        {
            match cfg.condense_mode {
                CondenseMode::Mask => {
                    mask_condense(&mut ctx, &state.turn_starts, cfg.condense_keep_recent);
                }
                CondenseMode::Rebuild => {
                    if let Some(new_ctx) = condense_context(cfg.model, cfg.task, &state.history, cfg.context_token_limit).await? {
                        ctx = new_ctx;
                    }
                }
            }
            state.condense_cooldown = CONDENSE_COOLDOWN;
        }

        let schema = if step == cfg.max_steps {
            FINISH_SCHEMA
        } else {
            ACTION_SCHEMA
        };

        let tokens_before = ctx.seq_len() + ctx.buffer().len() as u32;
        let gen_start = Instant::now();
        let raw = ctx
            .generate(make_sampler(cfg.temperature, cfg.top_p))
            .max_tokens(cfg.max_tokens_per_step)
            .constrain_with(inferlet::JsonSchema(schema))?
            .collect_text()
            .await?;
        let gen_elapsed = gen_start.elapsed();
        let tokens_after = ctx.seq_len();
        let step_prompt = tokens_before;
        let step_completion = tokens_after.saturating_sub(tokens_before);
        state.total_prompt_tokens += step_prompt;
        state.total_completion_tokens += step_completion;

        let v = match serde_json::from_str::<Value>(&raw) {
            Ok(v) => v,
            Err(e) => {
                state.consecutive_failures += 1;
                println!("[{workspace_id}][step {step}] truncated at max_tokens ({e}), skipping (failure {}/{MAX_CONSECUTIVE_FAILURES})", state.consecutive_failures);
                if state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    println!("[{workspace_id}][step {step}] too many consecutive failures, forcing finish");
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
        let insert_line = v.get("insert_line").and_then(Value::as_i64).unwrap_or(0);
        let start_line = v.get("start_line").and_then(Value::as_i64).unwrap_or(0);
        let end_line = v.get("end_line").and_then(Value::as_i64).unwrap_or(0);
        let message = v.get("message").and_then(Value::as_str).unwrap_or("");

        // Detect degenerate output (garbled CJK/symbol soup).
        if is_degenerate(thought) {
            state.consecutive_failures += 1;
            println!("[{workspace_id}][step {step}] degenerate output detected (failure {}/{MAX_CONSECUTIVE_FAILURES})", state.consecutive_failures);
            if state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                println!("[{workspace_id}][step {step}] too many degenerate outputs, forcing finish");
                break;
            }
            // Condense and retry — degeneration usually means context overflow.
            if !state.history.is_empty() {
                match cfg.condense_mode {
                    CondenseMode::Mask => {
                        mask_condense(&mut ctx, &state.turn_starts, cfg.condense_keep_recent);
                    }
                    CondenseMode::Rebuild => {
                        if let Some(new_ctx) = condense_context(cfg.model, cfg.task, &state.history, cfg.context_token_limit).await? {
                            ctx = new_ctx;
                        }
                    }
                }
                state.condense_cooldown = CONDENSE_COOLDOWN;
            }
            continue;
        }

        state.consecutive_failures = 0;
        println!("[{workspace_id}][step {step}] thought: {thought}");
        println!("[{workspace_id}][step {step}] action: {action}");

        if action == "finish" {
            println!("[{workspace_id}][step {step}] message: {message}");
            let _idle = ctx.idle();
            let has_diff = check_has_diff(cfg.tool_server_url, workspace_id).await;
            drop(_idle);
            if !has_diff && state.empty_finishes < cfg.max_empty_finishes {
                state.empty_finishes += 1;
                println!("[{workspace_id}][step {step}] finish with no diff (attempt {}/{}), nudging", state.empty_finishes, cfg.max_empty_finishes);
                state.history.push(Turn {
                    assistant_json: raw.clone(),
                    observation: String::new(),
                });
                state.turn_starts.push(tokens_before);
                state.step_metrics.push(StepMetrics {
                    step, action: action.to_string(), generate_s: gen_elapsed.as_secs_f64(),
                    tool_s: 0.0, prompt_tokens: step_prompt, completion_tokens: step_completion,
                    seq_len_after: tokens_after,
                });
                ctx.user(
                    "Observation: Your changes produced no diff — the repository is \
                     identical to its starting state. The issue is NOT resolved yet. \
                     Re-read the problem statement carefully, explore the codebase to \
                     find the right file and function, and make the necessary code change. \
                     Do NOT finish until you have verified that your edit shows up in \
                     `git diff`."
                );
                ctx.cue();
                continue;
            }
            if has_diff && !state.ran_tests_after_edit && !state.test_nudged {
                state.test_nudged = true;
                println!("[{workspace_id}][step {step}] finish without running tests, nudging");
                state.history.push(Turn {
                    assistant_json: raw.clone(),
                    observation: String::new(),
                });
                state.turn_starts.push(tokens_before);
                state.step_metrics.push(StepMetrics {
                    step, action: action.to_string(), generate_s: gen_elapsed.as_secs_f64(),
                    tool_s: 0.0, prompt_tokens: step_prompt, completion_tokens: step_completion,
                    seq_len_after: tokens_after,
                });
                ctx.user(
                    "Observation: You have not verified your fix by running the \
                     relevant test suite. Before finishing, you MUST: \
                     1) Run your reproduction script to confirm it now produces correct output. \
                     2) Find and run the existing tests for the module you changed \
                     (e.g., `python -m pytest path/to/tests/test_module.py -x -q`). \
                     If any test fails, your fix is WRONG — go back and fix the root cause. \
                     Do NOT finish until tests pass."
                );
                ctx.cue();
                continue;
            }
            state.step_metrics.push(StepMetrics {
                step, action: action.to_string(), generate_s: gen_elapsed.as_secs_f64(),
                tool_s: 0.0, prompt_tokens: step_prompt, completion_tokens: step_completion,
                seq_len_after: tokens_after,
            });
            final_message = Some(message.to_string());
            break;
        }

        let _idle = ctx.idle();
        let tool_start = Instant::now();
        let observation = match call_tool_server(
            cfg.tool_server_url,
            workspace_id,
            action,
            command,
            path,
            old_str,
            new_str,
            insert_line,
            start_line,
            end_line,
        )
        .await
        {
            Ok(obs) => obs,
            Err(e) => format!("Tool server error: {e}"),
        };
        let tool_elapsed = tool_start.elapsed();
        drop(_idle);

        let failed = observation.contains("Error:") || observation.contains("not found");
        state.recent_actions.push(RecentAction {
            action: action.to_string(),
            command: command.to_string(),
            path: path.to_string(),
            failed,
        });
        if state.recent_actions.len() > STUCK_WINDOW + 2 {
            state.recent_actions.remove(0);
        }

        // Track whether the agent verified its fix with tests.
        if (action == "edit" || action == "insert") && !failed {
            state.ran_tests_after_edit = false;
        }
        if action == "bash" && (command.contains("pytest") || command.contains("unittest")) {
            state.ran_tests_after_edit = true;
        }

        let observation = truncate_observation(&observation, cfg.max_observation_chars);
        println!("[{workspace_id}][step {step}] observation ({} chars)", observation.len());

        let mut obs_with_hint = if observation.contains("old_str not found") {
            format!(
                "{observation}\n\nHINT: Your old_str did not match the file content exactly. \
                 Use read_file with start_line/end_line to view the exact target lines, \
                 then copy the exact text for old_str. Or try insert/undo_edit."
            )
        } else if observation.contains("lines (max ") {
            format!(
                "{observation}\n\nHINT: Break your edit into smaller pieces. \
                 Edit only the specific lines that need to change, not the entire function or class."
            )
        } else {
            observation.clone()
        };

        if let Some(stuck_hint) = detect_stuck(&state.recent_actions) {
            println!("[{workspace_id}][step {step}] stuck detected");
            obs_with_hint = format!("{obs_with_hint}\n\n{stuck_hint}");
        }

        state.step_metrics.push(StepMetrics {
            step,
            action: action.to_string(),
            generate_s: gen_elapsed.as_secs_f64(),
            tool_s: tool_elapsed.as_secs_f64(),
            prompt_tokens: step_prompt,
            completion_tokens: step_completion,
            seq_len_after: tokens_after,
        });

        state.history.push(Turn {
            assistant_json: raw.clone(),
            observation: observation.clone(),
        });
        state.turn_starts.push(tokens_before);

        ctx.user(&format!("Observation:\n{obs_with_hint}"));
        ctx.cue();
    }

    match &final_message {
        Some(m) => println!("[{workspace_id}] finished: {m}"),
        None => println!("[{workspace_id}] did not finish within {} steps", cfg.max_steps),
    }

    let total_generate_s: f64 = state.step_metrics.iter().map(|m| m.generate_s).sum();
    let total_tool_s: f64 = state.step_metrics.iter().map(|m| m.tool_s).sum();

    let (history, turn_starts) = if cfg.dump_trajectory {
        let dump = state
            .history
            .iter()
            .map(|t| TurnDump {
                assistant: t.assistant_json.clone(),
                observation: t.observation.clone(),
            })
            .collect();
        (dump, state.turn_starts.clone())
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(BranchOutcome::Finished(BranchResult {
        workspace_id: workspace_id.to_string(),
        finished: final_message.is_some(),
        message: final_message.unwrap_or_default(),
        steps: state.history.len(),
        total_wall_s: branch_start.elapsed().as_secs_f64(),
        total_generate_s,
        total_tool_s,
        total_prompt_tokens: state.total_prompt_tokens,
        total_completion_tokens: state.total_completion_tokens,
        step_metrics: state.step_metrics,
        history,
        turn_starts,
    }))
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let cfg = BranchCfg {
        model: &model,
        task: &input.task,
        tool_server_url: &input.tool_server_url,
        max_steps: input.max_steps,
        max_tokens_per_step: input.max_tokens_per_step,
        context_token_limit: input.context_token_limit,
        max_observation_chars: input.max_observation_chars,
        max_empty_finishes: input.max_empty_finishes,
        temperature: input.temperature,
        top_p: input.top_p,
        condense_mode: if input.condense_mode == "mask" {
            CondenseMode::Mask
        } else {
            CondenseMode::Rebuild
        },
        condense_keep_recent: input.condense_keep_recent,
        dump_trajectory: input.dump_trajectory,
    };

    let mut ctx = Context::new(&model)?;
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&input.task);
    ctx.cue();

    let run_start = Instant::now();
    let num_branches = input.num_branches.max(1);

    // ── Single-trajectory (backward-compatible) path ──────────────────────
    if num_branches == 1 {
        let r = match run_branch(&cfg, ctx, "0", LoopState::default(), 1, 0).await? {
            BranchOutcome::Finished(r) => r,
            BranchOutcome::Suspended { .. } => unreachable!("branch_at_step=0 never suspends"),
        };
        let mut result = serde_json::json!({
            "finished": r.finished,
            "message": r.message,
            "steps": r.steps,
            "metrics": {
                "total_wall_s": run_start.elapsed().as_secs_f64(),
                "total_generate_s": r.total_generate_s,
                "total_tool_s": r.total_tool_s,
                "total_prompt_tokens": r.total_prompt_tokens,
                "total_completion_tokens": r.total_completion_tokens,
                "num_generate_calls": r.step_metrics.len(),
                "per_step": r.step_metrics,
            },
        });
        if input.dump_trajectory {
            result["trajectory"] = serde_json::json!({
                "system": SYSTEM_PROMPT,
                "task": input.task,
                "turns": r.history,
                "turn_starts": r.turn_starts,
            });
        }
        return Ok(result.to_string());
    }

    // ── Multi-branch (test-time scaling) path ─────────────────────────────
    // Run the trunk on workspace "0" up to the branch point.
    let (base_ctx, base_state, next_step) =
        match run_branch(&cfg, ctx, "0", LoopState::default(), 1, input.branch_at_step).await? {
            BranchOutcome::Suspended { ctx, state, next_step } => (ctx, state, next_step),
            BranchOutcome::Finished(r) => {
                // Trunk finished before reaching the branch point — no fan-out.
                let result = serde_json::json!({
                    "branches": [serde_json::to_value(&r).unwrap_or(Value::Null)],
                    "branch_step": input.branch_at_step,
                    "num_branches": 1,
                    "note": "trunk finished before branch point",
                    "total_wall_s": run_start.elapsed().as_secs_f64(),
                });
                return Ok(result.to_string());
            }
        };

    // Snapshot the trunk workspace into per-branch copies "1".."K-1".
    let to_ids: Vec<String> = (1..num_branches).map(|i| i.to_string()).collect();
    fork_workspace(cfg.tool_server_url, "0", &to_ids)
        .await
        .map_err(|e| format!("fork_workspace: {e}"))?;

    // Fork the KV context K ways: index 0 = trunk (workspace "0"), 1.. = copies.
    let mut forks: Vec<Context> = Vec::with_capacity(num_branches - 1);
    for _ in 1..num_branches {
        forks.push(base_ctx.fork()?);
    }
    let mut branch_ctxs: Vec<Context> = Vec::with_capacity(num_branches);
    branch_ctxs.push(base_ctx);
    branch_ctxs.extend(forks);

    // Resume all branches concurrently from the branch point.
    let futs = branch_ctxs.into_iter().enumerate().map(|(i, bctx)| {
        let wid = i.to_string();
        let st = base_state.clone();
        let cfg = &cfg;
        async move { run_branch(cfg, bctx, &wid, st, next_step, 0).await }
    });
    let outcomes = futures::future::join_all(futs).await;

    let mut branch_vals: Vec<Value> = Vec::new();
    let mut any_finished = false;
    for oc in outcomes {
        if let BranchOutcome::Finished(r) = oc? {
            any_finished |= r.finished;
            branch_vals.push(serde_json::to_value(&r).unwrap_or(Value::Null));
        }
    }

    let result = serde_json::json!({
        "branches": branch_vals,
        "branch_step": input.branch_at_step,
        "num_branches": num_branches,
        "any_finished": any_finished,
        "total_wall_s": run_start.elapsed().as_secs_f64(),
    });
    Ok(result.to_string())
}
