//! Decode-throughput diagnostic inferlet.
//!
//! Prefills an L-token prompt, then decodes EXACTLY `decode_tokens` tokens with
//! NO stop condition (so the count is deterministic regardless of EOS). Returns
//! prefill/decode token counts as JSON. The client measures wallclock at two
//! `decode_tokens` values (e.g. 64 and 512) and takes the slope
//!   decode_s_per_tok = (t_hi - t_lo) / (N_hi - N_lo)
//! which cancels the fixed prefill(L) cost, isolating pure decode throughput.
//! Sweeping L across calls shows whether decode tok/s collapses with context
//! length (the OpenHands-slowdown hypothesis) for this engine.

use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// Prompt to prefill (caller pads it to the target length).
    prompt: String,
    /// Exact number of tokens to decode, ignoring EOS.
    #[serde(default = "default_decode_tokens")]
    decode_tokens: usize,
}

fn default_decode_tokens() -> usize {
    128
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx = Context::new(&model)?;
    // Raw prompt as a user turn + cue; the padding lives inside `prompt`.
    ctx.user(&input.prompt).cue();
    let prefill_tokens = ctx.seq_len() as usize;

    // Decode exactly `decode_tokens` — NO `.stop(...)`, so EOS is ignored and
    // the loop always runs the full count.
    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: 0.9,
            p: 0.95,
        })
        .max_tokens(input.decode_tokens);

    let mut generated = 0usize;
    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        generated += out.tokens.len();
    }

    let result = serde_json::json!({
        "prefill_tokens": prefill_tokens,
        "decode_tokens_requested": input.decode_tokens,
        "decode_tokens_generated": generated,
    });
    Ok(result.to_string())
}
