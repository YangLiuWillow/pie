//! `self-correct-rollout` — Step 2a: two-turn self-correction, CHAT-TEMPLATED.
//!
//! A proper multi-turn chat, so an instruct model answers focused and stops on
//! `<|im_end|>` instead of rambling past the token cap:
//!   system + user(question) + cue            → the PROMPT
//!   turn 1  (assistant answer)               → trainable, env_mask 1
//!   inject: <|im_end|>\n + user(reflect) + cue → masked, env_mask 0
//!   turn 2  (assistant final answer)         → trainable, env_mask 1
//!   turn 2's <|im_end|>, if it stopped on its own → trainable, env_mask 1
//!
//! Emits one flat training row per rollout:
//!   prompt_ids     = system ++ user(question) ++ cue
//!   completion_ids = t1 ++ inject ++ t2 [++ <|im_end|>]
//!   env_mask       = 1*|t1| ++ 0*|inject| ++ 1*|t2| [++ 1]
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
// `/no_think` is Qwen3's soft switch to skip the <think> block: without it a
// reasoning model spends the whole token budget inside <think> and the turn gets
// truncated before it answers. It also makes self-correction a real test — with
// thinking ON the model already corrects itself inside <think>, so the injected
// reflect turn would be redundant; OFF, the reflect turn is where correction has
// to happen, which is what we want RL to teach.
fn d_system() -> String {
    "You are a careful math assistant. Solve the problem and give the final answer \
     as a single number. /no_think".to_string()
}
fn d_reflect() -> String {
    "Review your solution above for any mistake. Give the corrected final answer \
     as a single number. /no_think".to_string()
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

    let stops = chat::stop_tokens(&model);

    // Close the assistant turn. NOT `chat::seal()`: that returns every stop id
    // (`runtime/src/model/instruct/qwen3.rs:410` → `stop_ids`), which for Qwen3 is
    // BOTH `<|im_end|>` and `<|endoftext|>` (`runtime/src/model/instruct.rs:178`).
    // `<|endoftext|>` is a document separator — injecting it between turn 1 and the
    // reflect prompt tells the model an unrelated document follows, right before we
    // ask it to review its own answer. The template's own message builders close a
    // turn with `turn_suffix` = `<|im_end|>` ++ newline (`qwen3.rs:197-198`); use
    // that. First stop id is the turn terminator for every ChatML config in the
    // runtime (the `<|endoftext|>` entries come second).
    let turn_end = *stops.first().ok_or("model exposes no stop tokens")?;
    let mut turn_close = vec![turn_end];
    turn_close.extend(tokenizer.encode("\n"));

    // Injected turn = close assistant + user(reflect) + cue (all masked).
    let mut inject = Vec::new();
    inject.extend_from_slice(&turn_close);
    inject.extend(chat::user(&model, &input.reflect_prompt));
    inject.extend(chat::cue(&model));

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

        // Did turn 2 end because the model emitted a stop token, or because it hit
        // the cap? Nothing else ends generation early, so length decides it.
        let t2_terminated = t2.len() < input.max_tokens;

        // Linearize: completion = t1 ++ inject ++ t2 [++ turn_end] ; mask the
        // injected turn out.
        //
        // The generator truncates AT the stop token (`generation.rs:645-647`), so a
        // naturally-ended turn 2 arrives without its `<|im_end|>`. Re-append it,
        // TRAINABLE, for two reasons:
        //   * termination becomes a decision the model gets gradient on. Without it
        //     nothing in the objective says "stop here" — length control rests
        //     entirely on the cap plus the accident that the reward's last-number
        //     rule punishes rambling.
        //   * TRL calls a completion truncated when `ids[-1]` is not eos/pad
        //     (`grpo_trainer.py:2419`). Ending on a normal token made EVERY rollout
        //     read as clipped: `completions/clipped_ratio` pinned at 1.0 and the
        //     terminated-length metrics permanently empty (`:2238-2246`) — and with
        //     `mask_truncated_completions=True` the whole batch's loss mask would be
        //     zeroed (`:2419-2424`), training on nothing, silently.
        //
        // Only when it really terminated. A rollout that hit the cap IS truncated
        // and must stay unterminated, or the accounting starts lying the other way.
        //
        // Turn 1's terminator is deliberately left masked: `inject` is appended
        // unconditionally, so marking it trainable would teach the model to emit a
        // stop it may never have chosen (a capped turn 1 gets closed regardless).
        // Training it would need the same length test applied to t1 — worth doing,
        // but it is a separate change with its own effect on what gets learned.
        let mut completion_ids =
            Vec::with_capacity(t1.len() + inject.len() + t2.len() + 1);
        completion_ids.extend_from_slice(&t1);
        completion_ids.extend_from_slice(&inject);
        completion_ids.extend_from_slice(&t2);

        let mut env_mask = Vec::with_capacity(completion_ids.len());
        env_mask.extend(std::iter::repeat(1u32).take(t1.len()));
        env_mask.extend(std::iter::repeat(0u32).take(inject.len()));
        env_mask.extend(std::iter::repeat(1u32).take(t2.len()));
        if t2_terminated {
            completion_ids.push(turn_end);
            env_mask.push(1);
        }

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
