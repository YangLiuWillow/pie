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
use pie_model_qwen_3::chat::{ChatMLConfig, QwenInstruct, ToolDialect};
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

    // The exact config `model/src/instruct.rs::create` binds for "qwen3" —
    // including the tool dialect, which is decided by that module's OWN
    // predicate rather than a copy of it. Rendering Coder's XML dialect as
    // Hermes JSON (or the reverse) produces a prompt the server never sends,
    // and this harness would then certify parity against a fiction. The model
    // name is taken from the tokenizer path, which names the checkpoint;
    // `PARITY_MODEL_NAME` overrides it when the path does not.
    let model_name = std::env::var("PARITY_MODEL_NAME").unwrap_or_else(|_| args[1].clone());
    let coder = pie_model::instruct::is_coder_lineage("qwen3", &model_name);
    // TWO predicates now, mirroring the server exactly. They were one, on the
    // rule that "a checkpoint with no thinking channel is the Coder release,
    // and the Coder release speaks XML tool calls" — until Qwen3.6, a thinking
    // model that speaks XML. `has_thinking` is still `!coder` (it was once
    // hardcoded `true` here, so a Coder render carried an empty
    // `<think></think>` cue the server never emits — 4 tokens of pure harness
    // artifact, reported as a pie-vs-vLLM divergence); the dialect now comes
    // from its own predicate. Both are `instruct`'s, never copies: a harness
    // that guesses either one certifies parity against a prompt the server
    // does not send.
    // The arch stem is hardcoded "qwen3" here, so a 3.5/3.6 checkpoint is
    // recognised by its DEPLOYMENT name -- which is what the fixtures carry.
    let tool_dialect = pie_model::instruct::tool_dialect(&model_name, &model_name);
    eprintln!("[render-tokens] coder={coder} tool_dialect={tool_dialect:?} (from {model_name})");
    let instruct = QwenInstruct::new(
        tokenizer.clone(),
        ChatMLConfig {
            has_thinking: !coder,
            has_tools: true,
            tool_dialect,
            // Mirrors the registry: only Qwen3.5/3.6 leads with the tools
            // block. Derived here rather than hardcoded, so the harness cannot
            // certify an ordering the server does not use.
            system_before_tools: !matches!(tool_dialect, ToolDialect::Qwen35Xml),
            empty_reasoning_header: matches!(tool_dialect, ToolDialect::Qwen35Xml),
            // Mirrors the registry: Qwen3.5/3.6 opens the turn inside a
            // reasoning block, Qwen3 opens it bare.
            generation_suffix: if matches!(tool_dialect, ToolDialect::Qwen35Xml) {
                "<think>\n"
            } else {
                ""
            },
            tool_response_trailing_newline: matches!(tool_dialect, ToolDialect::Coder),
            thinking_off_suffix: "<think>\n\n</think>\n\n",
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
            RenderOp::Assistant(msg, p) => instruct.assistant_at(msg, p.after_query, p.is_last),
            RenderOp::AssistantWithToolCalls { content, calls, pos } => instruct
                .assistant_with_tool_calls_at(
                    content.as_deref(),
                    calls,
                    pos.after_query,
                    pos.is_last,
                ),
            RenderOp::AnswerBatch(results) => instruct.answer_batch(results),
            // The mode rides on the op, set by the request's
            // `chat_template_kwargs.enable_thinking` — the same path the
            // server takes. It was an env var while `cue`/`cue-no-think` were
            // two WIT functions and the guest picked one.
            RenderOp::Cue(thinking) => {
                if *thinking { instruct.cue() } else { instruct.cue_no_think() }
            }
        };
        ids.extend(toks);
    }

    println!("{}", serde_json::to_string(&ids)?);
    // Decoded prompt (special tokens kept) to stderr for eyeballing diffs.
    let decoded = tokenizer.decode(&ids, false);
    std::io::stderr().write_all(decoded.as_bytes())?;
    Ok(())
}
