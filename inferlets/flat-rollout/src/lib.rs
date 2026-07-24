//! `flat-rollout` — Step 1 rollout inferlet: single completion, no branching.
//!
//! Generates `n` independent completions from a raw prompt and emits, per
//! completion, the data a GRPO trainer needs: the generated token ids, a
//! per-token logprob, and an `env_mask` (all 1 here, since every completion
//! token is model-generated).
//!
//! Optional `adapter_path` (1c-ii on-policy sync): when set, the LoRA is applied
//! on EVERY forward pass — prompt prefill, generation, AND logprob scoring — so
//! Pie's logprobs match TRL's recompute (which applies the adapter at every
//! position). Without that consistency the GRPO importance ratio drifts even
//! on-policy.
//!
//! Two single-slot phases (the portable driver aborts on two output slots in one
//! pass): generate with the auto-sampler, then teacher-force score with a probe.
//!
//! Input : `{ "prompt", "max_tokens", "temperature", "top_p", "n",
//!            "adapter_path"? }`
//! Output: `{ "model", "prompt_ids", "completions": [
//!            { "completion_ids", "logprobs", "env_mask", "text" }, ... ] }`

use inferlet::adapter::Adapter;
use inferlet::model::Model;
use inferlet::sample::{Distribution, Sampler};
use inferlet::{chat, runtime, Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "d_max_tokens")]
    max_tokens: usize,
    #[serde(default = "d_temperature")]
    temperature: f32,
    #[serde(default = "d_top_p")]
    top_p: f32,
    #[serde(default = "d_n")]
    n: usize,
    #[serde(default)]
    adapter_path: Option<String>,
}

fn d_max_tokens() -> usize { 32 }
fn d_temperature() -> f32 { 0.8 }
fn d_top_p() -> f32 { 0.95 }
fn d_n() -> usize { 4 }

const BEGIN: &str = "<<<ROLLOUT_JSON>>>";
const END: &str = "<<<ROLLOUT_END>>>";
const LOGPROB_TOPK: u32 = 512;
const LOGPROB_FLOOR: f32 = -30.0;

/// Teacher-force `target` after the prompt (with `adapter` applied) and return
/// `ln p(target[i] | prefix)` at T=1. `base` has the prompt head committed;
/// `tail` is the last prompt token, the anchor that predicts target[0].
async fn score_logprobs(
    base: &Context,
    tail: u32,
    target: &[u32],
    adapter: Option<&Adapter>,
) -> Result<Vec<f32>> {
    let mut ctx = base.fork()?;
    let mut pending = vec![tail];
    let mut lps = Vec::with_capacity(target.len());

    for &t in target {
        let mut pass = ctx.forward();
        pass.input(&pending);
        if let Some(a) = adapter {
            pass.adapter(a);
        }
        let last_idx = (pending.len() - 1) as u32;
        let h = pass.probe(last_idx, Distribution { temperature: 1.0, k: LOGPROB_TOPK });
        let out = pass.execute().await?;
        let lp = match out.distribution(h) {
            Some((ids, probs)) => ids
                .iter()
                .position(|&id| id == t)
                .map(|i| probs[i])
                .filter(|&p| p > 0.0)
                .map(|p| p.ln())
                .unwrap_or(LOGPROB_FLOOR),
            None => LOGPROB_FLOOR,
        };
        lps.push(lp);
        pending = vec![t];
    }
    Ok(lps)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?.clone();
    let model = Model::load(&model_name)?;
    let tokenizer = model.tokenizer();

    let prompt_ids = tokenizer.encode(&input.prompt);
    if prompt_ids.is_empty() {
        return Err("empty prompt".into());
    }

    // Optional LoRA (on-policy sync). Open-or-create a fixed name, then load the
    // file — on a persistent server this refreshes the weights each call.
    let adapter = match &input.adapter_path {
        Some(path) => {
            let a = match Adapter::open(&model, "rl-lora") {
                Some(a) => a,
                None => Adapter::create(&model, "rl-lora")?,
            };
            a.load(path)?;
            Some(a)
        }
        None => None,
    };

    let stops = chat::stop_tokens(&model);

    // Prefill the prompt head WITH the adapter (so prompt KV matches the policy),
    // keeping the last prompt token as the anchor both phases start from.
    let split = prompt_ids.len() - 1;
    let tail = prompt_ids[split];
    let mut base = Context::new(&model)?;
    if split > 0 {
        let mut pass = base.forward();
        pass.input(&prompt_ids[..split]);
        if let Some(a) = &adapter {
            pass.adapter(a);
        }
        pass.execute().await?;
    }

    let mut completions = Vec::with_capacity(input.n);
    for _ in 0..input.n {
        // Phase 1 — generate (auto-sampler only, one slot), adapter applied.
        let completion_ids = {
            let mut ctx = base.fork()?;
            ctx.append(&[tail]); // last prompt token pending; generate drains it first
            let mut g = ctx
                .generate(Sampler::TopP { temperature: input.temperature, p: input.top_p })
                .max_tokens(input.max_tokens)
                .stop(&stops);
            if let Some(a) = &adapter {
                g = g.adapter(a);
            }
            g.collect_tokens().await?
        };

        // Phase 2 — score per-token logprobs (probe only, one slot), same adapter.
        let logprobs = score_logprobs(&base, tail, &completion_ids, adapter.as_ref()).await?;

        let text = tokenizer.decode(&completion_ids).unwrap_or_default();
        let env_mask = vec![1u32; completion_ids.len()];
        completions.push(serde_json::json!({
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "env_mask": env_mask,
            "text": text,
        }));
    }

    let out = serde_json::json!({
        "model": model_name,
        "prompt_ids": prompt_ids,
        "completions": completions,
    });
    let json = serde_json::to_string(&out).map_err(|e| e.to_string())?;

    println!("{BEGIN}{json}{END}");
    Ok(json)
}
