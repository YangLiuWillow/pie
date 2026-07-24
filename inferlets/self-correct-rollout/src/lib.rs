//! `self-correct-rollout` — Step 2a: two-turn self-correction, CHAT-TEMPLATED.
//!
//! A proper multi-turn chat, so an instruct model answers focused and stops on
//! `<|im_end|>` instead of rambling past the token cap:
//!   system + user(question) + cue          → the PROMPT
//!   turn 1  (assistant answer)               → trainable, env_mask 1
//!   inject: seal + user(reflect) + cue       → masked,   env_mask 0
//!   turn 2  (assistant final answer)         → trainable, env_mask 1
//!
//! Emits one flat training row per rollout:
//!   prompt_ids     = system ++ user(question) ++ cue
//!   completion_ids = t1 ++ inject ++ t2
//!   env_mask       = 1*|t1| ++ 0*|inject| ++ 1*|t2|
//!   logprobs       = zeros  (TRL's GRPO loss on our path — num_iterations=1,
//!                    use_vllm=False — ignores the sampling logprobs: ratio is 1
//!                    and the KL is to the ref model. So we skip teacher-force scoring.)
//!   final_answer   = decoded t2   (the reward reads THIS, not turn-1 numbers)
//!
//! The adapter (on-policy LoRA) is applied on the prompt prefill and both turns.
//! Generation only — no scoring pass.

use inferlet::adapter::Adapter;
use inferlet::model::Model;
use inferlet::sample::Sampler;
use inferlet::{chat, runtime, Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// The question (raw text; the inferlet chat-templates it).
    prompt: String,
    #[serde(default = "d_n")]
    n: usize,
    #[serde(default = "d_max_tokens")]
    max_tokens: usize,
    #[serde(default = "d_temperature")]
    temperature: f32,
    #[serde(default = "d_top_p")]
    top_p: f32,
    #[serde(default)]
    adapter_path: Option<String>,
    #[serde(default = "d_system")]
    system: String,
    #[serde(default = "d_reflect")]
    reflect_prompt: String,
}

fn d_n() -> usize { 4 }
fn d_max_tokens() -> usize { 256 }
fn d_temperature() -> f32 { 0.8 }
fn d_top_p() -> f32 { 0.95 }
fn d_system() -> String {
    "You are a careful math assistant. Solve the problem step by step, then end \
     with the final answer as a single number.".to_string()
}
fn d_reflect() -> String {
    "Review your solution above for any mistake. Give the corrected final answer \
     as a single number.".to_string()
}

const BEGIN: &str = "<<<ROLLOUT_JSON>>>";
const END: &str = "<<<ROLLOUT_END>>>";

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?.clone();
    let model = Model::load(&model_name)?;
    let tokenizer = model.tokenizer();

    // Prompt = system + user(question) + cue (chat-templated).
    let mut prompt_ids = Vec::new();
    prompt_ids.extend(chat::system(&model, &input.system));
    prompt_ids.extend(chat::user(&model, &input.prompt));
    prompt_ids.extend(chat::cue(&model));
    if prompt_ids.is_empty() {
        return Err("empty prompt".into());
    }

    // Injected turn = close assistant + user(reflect) + cue (all masked).
    let mut inject = Vec::new();
    inject.extend(chat::seal(&model));
    inject.extend(chat::user(&model, &input.reflect_prompt));
    inject.extend(chat::cue(&model));

    let stops = chat::stop_tokens(&model);

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

    // Prefill the prompt head WITH the adapter; keep the last token as the anchor.
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

    let sampler = Sampler::TopP { temperature: input.temperature, p: input.top_p };

    let mut completions = Vec::with_capacity(input.n);
    for _ in 0..input.n {
        let mut ctx = base.fork()?;
        ctx.append(&[tail]); // last prompt token pending → first turn drains it

        // Turn 1: initial answer (stops on <|im_end|>).
        let t1 = {
            let mut g = ctx.generate(sampler.clone()).max_tokens(input.max_tokens).stop(&stops);
            if let Some(a) = &adapter {
                g = g.adapter(a);
            }
            g.collect_tokens().await?
        };

        // Inject the reflect turn (masked), then Turn 2: final answer.
        ctx.append(&inject);
        let t2 = {
            let mut g = ctx.generate(sampler.clone()).max_tokens(input.max_tokens).stop(&stops);
            if let Some(a) = &adapter {
                g = g.adapter(a);
            }
            g.collect_tokens().await?
        };

        // Linearize: completion = t1 ++ inject ++ t2 ; mask the injected turn out.
        let mut completion_ids = Vec::with_capacity(t1.len() + inject.len() + t2.len());
        completion_ids.extend_from_slice(&t1);
        completion_ids.extend_from_slice(&inject);
        completion_ids.extend_from_slice(&t2);

        let mut env_mask = Vec::with_capacity(completion_ids.len());
        env_mask.extend(std::iter::repeat(1u32).take(t1.len()));
        env_mask.extend(std::iter::repeat(0u32).take(inject.len()));
        env_mask.extend(std::iter::repeat(1u32).take(t2.len()));

        let logprobs = vec![0.0f32; completion_ids.len()]; // unused by TRL on our path
        let final_answer = tokenizer.decode(&t2).unwrap_or_default();
        let turn1_text = tokenizer.decode(&t1).unwrap_or_default();
        let text = tokenizer.decode(&completion_ids).unwrap_or_default();

        completions.push(serde_json::json!({
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "env_mask": env_mask,
            "final_answer": final_answer,
            "turn1_text": turn1_text,
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
