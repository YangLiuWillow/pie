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
use pie_model_qwen_3::chat::{ChatMLConfig, CoderSchema, QwenInstruct, ToolDialect};
use pie_openai_serving::render::{RenderOp, plan_render};
use pie_openai_serving::types::ChatCompletionRequest;
use pie_tokenizer::Tokenizer;
use std::io::Write;
use std::sync::Arc;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // `--compile <tokenizer.json> <out.pietok>`: pie's own compiled tokenizer,
    // for the committed render-parity fixtures. Parsing HF JSON costs ~100 ms
    // and 11-20 MB; the compiled form loads in under a millisecond from a
    // third of the bytes, so the fixtures carry that instead.
    if args.len() == 4 && args[1] == "--compile" {
        let tok = Tokenizer::from_file(std::path::Path::new(&args[2]))
            .with_context(|| format!("loading tokenizer from {}", args[2]))?;
        let canonical = tok.to_canonical().context("compiling to pie.tokenizer/1")?;
        // A flat container, not JSON: `pie.tokenizer/1`'s objects are raw
        // bytes, and serialising them as JSON arrays-of-integers turned 6.8 MB
        // into 20.6 MB. Per object: u32 name length, name, u32 data length,
        // data -- all little-endian.
        let mut blob: Vec<u8> = Vec::new();
        for (name, bytes) in canonical.objects() {
            blob.extend_from_slice(&(name.len() as u32).to_le_bytes());
            blob.extend_from_slice(name.as_bytes());
            blob.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            blob.extend_from_slice(bytes);
        }
        std::fs::write(&args[3], &blob)
            .with_context(|| format!("writing {}", args[3]))?;
        eprintln!("[render-tokens] compiled {} -> {} ({} bytes)",
                  args[2], args[3], blob.len());
        return Ok(());
    }

    if args.len() != 3 {
        bail!("usage: render-tokens <tokenizer.json> <request.json>\n\
                      render-tokens --compile <tokenizer.json> <out.pietok>");
    }

    let tokenizer = Arc::new(
        Tokenizer::from_file(std::path::Path::new(&args[1]))
            .with_context(|| format!("loading tokenizer from {}", args[1]))?,
    );

    // Accept either a raw request body or a wire-capture fixture wrapper.
    let raw = std::fs::read_to_string(&args[2])
        .with_context(|| format!("reading request from {}", args[2]))?;
    // A JSON ARRAY is a batch: every element is rendered with the SAME
    // tokenizer and config, and the output is an array of id arrays in the same
    // order. Parsing an 11-20 MB `tokenizer.json` costs ~0.7 s and the parity
    // suite has 140 cells; one invocation per arm instead of one per cell is
    // the difference between a 100 s run and a 4 s one.
    let parsed: serde_json::Value = serde_json::from_str(&raw).context("parsing request JSON")?;
    let bodies: Vec<serde_json::Value> = match parsed {
        serde_json::Value::Array(v) => v,
        other => vec![other],
    };
    let batch = bodies.len() > 1 || matches!(
        serde_json::from_str::<serde_json::Value>(&raw), Ok(serde_json::Value::Array(_)));
    let requests: Vec<ChatCompletionRequest> = bodies
        .into_iter()
        .map(|mut value| {
            if let Some(body) = value.get("body") {
                value = match body {
                    serde_json::Value::String(s) => serde_json::from_str(s)
                        .context("parsing fixture .body string")?,
                    other => other.clone(),
                };
            }
            serde_json::from_value(value).context("deserializing ChatCompletionRequest")
        })
        .collect::<Result<_>>()?;

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
    // Mutation switch: flip ONE renderer fact so the parity suite can be asked
    // which shapes actually catch which mistake. A shape that catches nothing
    // no other shape catches is redundant; a mutation no shape catches is a
    // hole. Test-only, and never set in a real run.
    let mutate = std::env::var("PARITY_MUTATE").unwrap_or_default();
    let instruct = QwenInstruct::new(
        tokenizer.clone(),
        ChatMLConfig {
            has_thinking: !coder,
            has_tools: true,
            tool_dialect,
            // Mirrors the registry: only Qwen3.5/3.6 leads with the tools
            // block. Derived here rather than hardcoded, so the harness cannot
            // certify an ordering the server does not use.
            system_before_tools: !matches!(tool_dialect, ToolDialect::Qwen35Xml)
                ^ (mutate == "system_before_tools"),
            empty_reasoning_header: matches!(tool_dialect, ToolDialect::Qwen35Xml)
                ^ (mutate == "empty_reasoning_header"),
            // Mirrors the registry: Qwen3.5/3.6 opens the turn inside a
            // reasoning block, Qwen3 opens it bare.
            generation_suffix: if matches!(tool_dialect, ToolDialect::Qwen35Xml)
                ^ (mutate == "generation_suffix")
            {
                "<think>\n"
            } else {
                ""
            },
            tool_response_trailing_newline: matches!(tool_dialect, ToolDialect::Coder)
                ^ (mutate == "tool_response_trailing_newline"),
            // The harness picks per arm, which is exactly the operator choice
            // the field models: PARITY_CODER_SCHEMA=qwen renders Qwen's
            // published variant, anything else the mlx/GGUF one.
            coder_schema: if (std::env::var("PARITY_CODER_SCHEMA").as_deref() == Ok("qwen"))
                ^ (mutate == "coder_schema")
            {
                CoderSchema::QwenMain
            } else {
                CoderSchema::MlxGguf
            },
            thinking_off_suffix: if mutate == "thinking_off_suffix" {
                ""
            } else {
                "<think>\n\n</think>\n\n"
            },
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        },
    );

    let mut all: Vec<Vec<u32>> = Vec::with_capacity(requests.len());
    for request in &requests {
    let ops = plan_render(request).map_err(|e| anyhow::anyhow!("plan_render: {e}"))?;

    let mut ids: Vec<u32> = Vec::new();
    for op in &ops {
        let toks = match op {
            RenderOp::EquipAfterSystem { system, tools } => {
                // `drop_empty_system` restores the `text_opt()` behaviour that
                // normalised a `content: ""` system turn out of existence.
                let system = match system.as_deref() {
                    Some("") if mutate == "drop_empty_system" => None,
                    other => other,
                };
                instruct.equip_after_system(system, tools)
            }
            RenderOp::User(msg) => instruct.user(msg),
            // Positional mutations act on the OP STREAM, so the serving crate
            // needs no test hooks: `ignore_is_last` forgets which turn is
            // final, `header_on_pre_query` claims every replayed turn follows
            // the query.
            RenderOp::Assistant(msg, p) => instruct.assistant_at(
                msg,
                p.after_query || mutate == "header_on_pre_query",
                p.is_last && mutate != "ignore_is_last",
            ),
            RenderOp::AssistantWithToolCalls { content, calls, pos } => instruct
                .assistant_with_tool_calls_at(
                    content.as_deref(),
                    calls,
                    pos.after_query || mutate == "header_on_pre_query",
                    pos.is_last && mutate != "ignore_is_last",
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
        all.push(ids);
    }

    if batch {
        println!("{}", serde_json::to_string(&all)?);
    } else {
        println!("{}", serde_json::to_string(&all[0])?);
        // Decoded prompt (special tokens kept) to stderr for eyeballing diffs.
        // Only for a single request -- interleaving N of them helps nobody.
        let decoded = tokenizer.decode(&all[0], false);
        std::io::stderr().write_all(decoded.as_bytes())?;
    }
    Ok(())
}
