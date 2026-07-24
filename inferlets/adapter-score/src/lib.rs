//! `adapter-score` — 1c-ii LoRA numerical-parity probe.
//!
//! Teacher-forces a fixed `target` after `prompt`, applying an optional LoRA
//! adapter on EVERY forward pass (prompt prefill included, so the prompt KV
//! matches peft, which applies the LoRA at every position), and returns the
//! per-token `ln p(target[i] | prefix)` at T=1. Deterministic → diffable against
//! HF + peft applying the same LoRA, to trust on-policy weight sync before
//! wiring it into training.
//!
//! Input : `{ "prompt": "...", "target": [id,...], "adapter_path": "..."? }`
//! Output: `{ "model", "prompt_ids", "logprobs", "adapter_applied" }`

use inferlet::adapter::Adapter;
use inferlet::model::Model;
use inferlet::sample::Distribution;
use inferlet::{runtime, Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    prompt: String,
    target: Vec<u32>,
    #[serde(default)]
    adapter_path: Option<String>,
}

const BEGIN: &str = "<<<ADAPTER_SCORE_JSON>>>";
const END: &str = "<<<ADAPTER_SCORE_END>>>";
const TOPK: u32 = 512;
const FLOOR: f32 = -30.0;

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
    if input.target.is_empty() {
        return Err("empty target".into());
    }

    // Load the adapter (if given). Open-or-create by a fixed name, then load the
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

    let mut ctx = Context::new(&model)?;

    // Prefill the prompt head WITH the adapter (last prompt token kept as the
    // anchor that predicts target[0]).
    let split = prompt_ids.len() - 1;
    let tail = prompt_ids[split];
    if split > 0 {
        let mut pass = ctx.forward();
        pass.input(&prompt_ids[..split]);
        if let Some(a) = &adapter {
            pass.adapter(a);
        }
        pass.execute().await?;
    }

    // Teacher-force the target, one probe slot per pass (portable-safe).
    let mut pending = vec![tail];
    let mut logprobs = Vec::with_capacity(input.target.len());
    for &t in &input.target {
        let mut pass = ctx.forward();
        pass.input(&pending);
        if let Some(a) = &adapter {
            pass.adapter(a);
        }
        let last = (pending.len() - 1) as u32;
        let h = pass.probe(last, Distribution { temperature: 1.0, k: TOPK });
        let out = pass.execute().await?;
        let lp = match out.distribution(h) {
            Some((ids, probs)) => ids
                .iter()
                .position(|&id| id == t)
                .map(|i| probs[i])
                .filter(|&p| p > 0.0)
                .map(|p| p.ln())
                .unwrap_or(FLOOR),
            None => FLOOR,
        };
        logprobs.push(lp);
        pending = vec![t];
    }

    let out = serde_json::json!({
        "model": model_name,
        "prompt_ids": prompt_ids,
        "logprobs": logprobs,
        "adapter_applied": adapter.is_some(),
    });
    let json = serde_json::to_string(&out).map_err(|e| e.to_string())?;

    println!("{BEGIN}{json}{END}");
    Ok(json)
}
