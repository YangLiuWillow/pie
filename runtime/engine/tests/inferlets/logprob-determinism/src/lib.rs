//! **Is pie's decode deterministic at the VALUE layer, not just the token layer?**
//!
//! ## Why this exists
//!
//! A sibling investigation on CUDA (H200, Qwen3-1.7B, greedy) found prefill
//! exactly reproducible and decode not: across 16 token-identical runs, **0 of
//! 32 tokens had the same logprob twice**, mean per-token spread 0.024 nats.
//!
//! Metal was checked against that and came back clean — nine 200-token greedy
//! generations, three server boots, two binaries, byte-identical text. But text
//! identity is a COARSER claim than logprob identity: an argmax is stable under
//! any perturbation smaller than the gap to the runner-up, so drifting values
//! can hide entirely behind identical tokens. That check could not have seen the
//! CUDA finding even if Metal had it.
//!
//! Nothing on this branch could: `inferlets/openai-serving` hardcodes
//! `"logprobs": null` in six places, so the value layer is not merely untested
//! through the server — it is unobservable. `entropycheck` reads logits but
//! fires exactly one prefill, which is the arm that is deterministic on CUDA
//! too, so repeating it would be a clean number answering nothing.
//!
//! ## What this measures
//!
//! A greedy decode loop that publishes, per generated token:
//!
//!   * the token id — the argmax, i.e. what a text diff would see;
//!   * **the chosen token's logprob** — `reduce_max(log_softmax(logits))`, which
//!     is the same quantity `characterize_noise.py` compares (its reference
//!     number is 0/32 identical on H200);
//!   * **the whole-vocab entropy** — `entropy(softmax(logits))`, a sum over all
//!     151936 lanes and therefore sensitive to drift anywhere in the
//!     distribution, not only near the top.
//!
//! Two quantities rather than one because they fail differently: a logprob can
//! be bit-stable while the tail moves, and the entropy catches that; the entropy
//! is one scalar over a huge sum and could mask compensating errors, which the
//! logprob would not.
//!
//! The values are printed as RAW BITS (`f32::to_bits`) and not as decimals.
//! Determinism here is a bitwise question, and `{:.6}` would round two different
//! floats onto the same text and report a clean run that never happened.
//!
//! ## Reading the output
//!
//! One `[lpdet]` line per step. Run it N times and compare the lines: identical
//! bits across runs means the value layer is deterministic; identical token ids
//! with differing bits is exactly the CUDA finding, reproduced on Metal.

use inferlet::Result;
use inferlet::ptir::attention::prelude::*;
use inferlet::{model as wit_model};

/// Tokens to generate. Long enough that a drift which needs a few steps to
/// accumulate has room to show, short enough to stay a quick check.
const STEPS: usize = 32;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    // The same prompt shape the CUDA harness uses: a short instruction whose
    // continuation has several near-ties in it, since a value-layer difference
    // only becomes a TOKEN difference where the top-2 gap is narrow, and this
    // probe wants to observe both cases.
    let mut prompt = wit_model::encode(
        "You are a coding assistant. The file greet.py contains a typo: the word \
         'retrun' should be 'return'. Explain step by step how you would fix it \
         using the shell, and how you would verify it worked.\n\n",
    );
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + STEPS as u32 + 1).div_ceil(page_size);
    // Grow the working set to cover `tokens`, never past `max_pages`. Same
    // helper `inferlets/generate` carries — a decode fire whose write slot has
    // no page behind it fails at submit, and the loop below crosses a page
    // boundary every `page_size` steps.
    let reserve_to_tokens = |tokens: u32| -> std::result::Result<(), String> {
        let target = tokens.div_ceil(page_size).saturating_add(1).min(max_pages);
        let current = ws.page_len();
        if current < target {
            ws.reserve(target - current)?;
        }
        Ok(())
    };
    reserve_to_tokens(n.max(1)).context("ws.reserve prompt")?;
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    // ── prefill ──────────────────────────────────────────────────────────────
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(0..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, max_pages]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((0..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((0..n).map(|p| p % page_size)).named("w_off_p");
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");
    let g0_lp = Channel::new([1], dtype::f32).named("g0_lp");
    let g0_h = Channel::new([1], dtype::f32).named("g0_h");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &indptr_p)?;
    fwd_p.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len_p,
            pages: &pages_p,
            page_indptr: &page_indptr_p,
            w_slot: &w_slot_p,
            w_off: &w_off_p,
            positions: &positions_p,
            mask: None,
        },
    )?;
    fwd_p.epilogue(move || {
        let logits = intrinsics::logits();
        // argmax of the logits and argmax of their log-softmax are the same
        // index (log-softmax is monotonic), so `reduce_max` of the log-softmax
        // IS the chosen token's logprob — no gather needed.
        let lp = log_softmax(&logits);
        g0_ch.put(&reduce_argmax(&logits));
        g0_lp.put(&reduce_max(&lp));
        g0_h.put(&entropy(softmax(&logits)));
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;
    let t0 = g0_ch.take_host::<Vec<i32>>().await?;
    let l0 = g0_lp.take_host::<Vec<f32>>().await?;
    let h0 = g0_h.take_host::<Vec<f32>>().await?;
    let (t0, l0, h0) = (
        *t0.first().ok_or("prefill token empty")?,
        *l0.first().ok_or("prefill logprob empty")?,
        *h0.first().ok_or("prefill entropy empty")?,
    );
    // Step 0 is the PREFILL's own value. On CUDA this arm is deterministic and
    // decode is not, so keeping the two distinguishable in the output is the
    // point of labelling it separately.
    println!(
        "[lpdet] step=0 phase=prefill tok={t0} lp_bits={:08x} h_bits={:08x}",
        l0.to_bits(),
        h0.to_bits()
    );

    // ── decode ───────────────────────────────────────────────────────────────
    let tok_in = Channel::from([t0]).named("tok_in");
    let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
    let positions = Channel::from([n]).named("positions");
    let pages = Channel::from_iter(0..max_pages).named("pages");
    let page_indptr =
        Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
    let w_slot = Channel::from([n / page_size]).named("w_slot");
    let w_off = Channel::from([n % page_size]).named("w_off");
    let kv_len = Channel::from([n + 1]).named("kv_len");
    let out_tok = Channel::new([1], dtype::i32).named("out_tok");
    let out_lp = Channel::new([1], dtype::f32).named("out_lp");
    let out_h = Channel::new([1], dtype::f32).named("out_h");

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    fwd.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (n / page_size)..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    fwd.epilogue(move || {
        let logits = intrinsics::logits();
        let lp = log_softmax(&logits);
        let tok = reduce_argmax(&logits);

        // Geometry advance, same shape as `inferlets/generate`: every
        // loop-carried value is re-put on the DEVICE, so the host never feeds
        // the embed token back and cannot accidentally serialise the loop.
        let length = kv_len.take();
        let next_length = &length + 1u32;
        let page_count = next_length.div_ceil(page_size);
        tok_in.put(&tok);
        kv_len.put(&next_length);
        positions.put(&length);
        w_slot.put(&length / page_size);
        w_off.put(&length % page_size);
        page_indptr.put(&indptr(1, &page_count));

        out_tok.put(&tok);
        out_lp.put(&reduce_max(&lp));
        out_h.put(&entropy(softmax(&logits)));
    });

    let mut tokens: Vec<i32> = vec![t0];
    for step in 1..STEPS {
        reserve_to_tokens(n + step as u32)
            .with_context(|| format!("reserve decode @{step}"))?;
        fwd.submit(&pipe)
            .with_context(|| format!("decode submit @{step}"))?;
        let t = out_tok.take_host::<Vec<i32>>().await?;
        let l = out_lp.take_host::<Vec<f32>>().await?;
        let h = out_h.take_host::<Vec<f32>>().await?;
        let (t, l, h) = (
            *t.first().ok_or("decode token empty")?,
            *l.first().ok_or("decode logprob empty")?,
            *h.first().ok_or("decode entropy empty")?,
        );
        tokens.push(t);
        println!(
            "[lpdet] step={step} phase=decode tok={t} lp_bits={:08x} h_bits={:08x}",
            l.to_bits(),
            h.to_bits()
        );
    }
    pipe.close();

    let summary = format!("{tokens:?}");
    println!("[lpdet] tokens {summary}");
    Ok(summary)
}
