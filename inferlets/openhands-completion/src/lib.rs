//! openhands-completion — Phase 1 inferlet for the PieLLM integration.
//!
//! Contract (must stay in sync with `pie_openhands.llm.PieLLM._call_pie`):
//!
//!   Input:
//!     prompt:       String  (already rendered with the model's chat template)
//!     max_tokens:   usize   (default 2048)
//!     temperature:  f32     (default 0.0 — greedy)
//!     top_p:        f32     (default 0.95)
//!     stop:         Vec<String>  (default empty)
//!     model:        Option<String>  (informational; runtime picks the first model)
//!
//!   Output:
//!     text:             String   (decoded generated tokens)
//!     stop_reason:      "stop" | "length" | "eos"
//!     prompt_tokens:    usize
//!     tokens_generated: usize
//!
//! Phase 1 does NOT pin KV state across requests — that's `openhands-coder-session`
//! in Phase 2. We exit after every completion and the runtime releases the pages.

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};

// ─── Input / Output ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Input {
    prompt: String,

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

#[derive(Serialize)]
struct Output {
    text: String,
    stop_reason: String,
    prompt_tokens: usize,
    tokens_generated: usize,
}

// ─── Entry point ───────────────────────────────────────────────────────────

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let tokenizer = model.tokenizer();
    let prompt_tokens = tokenizer.encode(&input.prompt);
    let prompt_token_count = prompt_tokens.len();

    let mut ctx = Context::new(&model)?;
    ctx.append(&prompt_tokens);

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

    let mut generated: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut stop_reason = "length";

    let mut g = ctx
        .generate(sampler)
        .max_tokens(input.max_tokens)
        .stop(&stop_token_ids);

    'outer: while let Some(step) = g.next()? {
        let out = step.execute().await?;

        for &t in &out.tokens {
            generated.push(t);

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
            if let Ok(tail) = tokenizer.decode(&generated[tail_start..]) {
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

    let text = tokenizer
        .decode(&generated)
        .unwrap_or_else(|_| String::from("[decode error]"));

    // Trim a trailing stop string from the visible output (the agent loop
    // is sensitive to extra terminators leaking through).
    let visible = trim_trailing_stop(&text, &input.stop).to_string();

    Ok(Output {
        text: visible,
        stop_reason: stop_reason.to_string(),
        prompt_tokens: prompt_token_count,
        tokens_generated: generated.len(),
    })
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
