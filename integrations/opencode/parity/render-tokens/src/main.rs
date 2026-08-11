//! Renderer-parity harness: render an OpenAI chat-completions request to
//! token ids using the REAL serving crates — `pie_openai_serving::plan_render`
//! for the op plan, `QwenInstruct` (with the exact `ChatMLConfig` that
//! `model/src/instruct.rs` binds for `arch_name = "qwen3"`) for the template,
//! `pie_tokenizer` for encoding — so that `check_render.py` can diff the
//! result against HuggingFace `tokenizer.apply_chat_template`.
//!
//! Usage: render-tokens <tokenizer.json> <request.json>
//!
//! `<request.json>` is either a raw OpenAI chat-completions body or a wire
//! capture from `tests/inferlets/fixtures/opencode/wire/` (an object with a
//! `body` field holding the request as an object or a JSON string).
//!
//! stdout: the concatenated token ids as a JSON array.
//! stderr: the decoded prompt string (special tokens included), for debugging.

use anyhow::{Context, Result, bail};
use pie_model_common::instruct::Instruct;
use pie_model_qwen_3::chat::{ChatMLConfig, QwenInstruct};
use pie_openai_serving::render::{RenderOp, plan_render};
use pie_openai_serving::types::ChatCompletionRequest;
use pie_tokenizer::Tokenizer;
use std::io::Write;
use std::sync::Arc;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        bail!("usage: render-tokens <tokenizer.json> <request.json>");
    }

    let tokenizer = Arc::new(
        Tokenizer::from_file(std::path::Path::new(&args[1]))
            .with_context(|| format!("loading tokenizer from {}", args[1]))?,
    );

    // Accept either a raw request body or a wire-capture fixture wrapper.
    let raw = std::fs::read_to_string(&args[2])
        .with_context(|| format!("reading request from {}", args[2]))?;
    let mut value: serde_json::Value = serde_json::from_str(&raw).context("parsing request JSON")?;
    if let Some(body) = value.get("body") {
        value = match body {
            serde_json::Value::String(s) => {
                serde_json::from_str(s).context("parsing fixture .body string")?
            }
            other => other.clone(),
        };
    }
    let request: ChatCompletionRequest =
        serde_json::from_value(value).context("deserializing ChatCompletionRequest")?;

    // The exact config `model/src/instruct.rs::create` binds for "qwen3".
    let instruct = QwenInstruct::new(
        tokenizer.clone(),
        ChatMLConfig {
            has_thinking: true,
            has_tools: true,
            generation_suffix: "",
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        },
    );

    let ops = plan_render(&request).map_err(|e| anyhow::anyhow!("plan_render: {e}"))?;

    let mut ids: Vec<u32> = Vec::new();
    for op in &ops {
        let toks = match op {
            RenderOp::EquipAfterSystem { system, tools } => {
                instruct.equip_after_system(system.as_deref(), tools)
            }
            RenderOp::User(msg) => instruct.user(msg),
            RenderOp::Assistant(msg) => instruct.assistant(msg),
            RenderOp::AssistantWithToolCalls { content, calls } => {
                instruct.assistant_with_tool_calls(content.as_deref(), calls)
            }
            RenderOp::AnswerBatch(results) => instruct.answer_batch(results),
            // The harness's HF side renders with enable_thinking=False, so
            // the pie side takes the matching no-think cue (D1).
            RenderOp::Cue => instruct.cue_no_think(),
        };
        ids.extend(toks);
    }

    println!("{}", serde_json::to_string(&ids)?);
    // Decoded prompt (special tokens kept) to stderr for eyeballing diffs.
    let decoded = tokenizer.decode(&ids, false);
    std::io::stderr().write_all(decoded.as_bytes())?;
    Ok(())
}
