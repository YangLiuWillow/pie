//! PTIR generation core: chunked prefill + device-carried decode loop.
//!
//! Ported from `tests/inferlets/chat-completion/src/lib.rs` (the
//! generation-core reference on this engine): PTIR prefill with in-graph
//! top-p/Gumbel sampling in the epilogue, then a 1-wide decode loop whose
//! geometry is loop-carried on device and drained through `run_ahead`. Three
//! deltas from the reference:
//!
//! - prefill is CHUNKED via [`prefill_chunks`] (the naive-baseline shape):
//!   serving prompts run to many thousands of tokens and a one-shot fire
//!   cannot exceed the driver's `max_embed_length()`;
//! - the caller supplies a per-token callback instead of the decode loop
//!   owning the chat decoder — all stop/decode/emission policy lives in
//!   `turn.rs`, this module only moves tokens;
//! - the body runs over BOTH forward-pass kinds a served model can report
//!   (see below), because a coding agent's model is as likely to be a GDN
//!   hybrid as a pure-attention one.
//!
//! ## Two pass kinds, one body
//!
//! `pie:inferlet` exposes three forward interfaces, and `ForwardPass` is
//! three unrelated types — an attention-only pass cannot even name a folded
//! recurrent state. A hybrid model (Qwen3.5/3.6 GDN, Nemotron-H Mamba2)
//! interleaves attention layers with recurrent ones, and the driver REJECTS a
//! forward whose rs-working-set count does not match its request rows:
//!
//! ```text
//! resolved forward has 1 request row(s), but recurrent-state model bound
//! 0 rs-working-set(s); expected 1
//! ```
//!
//! which arrives as a submit failure, degrades the turn to
//! `finish_reason:"length"`, and serves an empty completion in ~50 ms — the
//! exact symptom the first Qwen3.6-35B-A3B bring-up produced. So the binding
//! is abstracted behind [`BindState`] (one impl per interface) and the
//! generation body is expanded once per kind by `define_generate!`, written
//! once so the two cannot drift. [`generate`] dispatches on
//! `model::pass_kind()`.
//!
//! Serving never buffers: every fire folds straight into the recurrence
//! (`RsGeometry { fold_len: None, buffer: 0..0 }`). Buffering is for
//! speculation — a tail that may be rejected — and this loop accepts every
//! token it samples. The SAME rs working set is bound by the prefill chunks
//! and the decode fires, so decode continues the prefill's folded state
//! rather than starting cold.
//!
//! ## SEAM (KV snapshot sessions): this loop's fold OVERSHOOTS, and must not
//!
//! `run_ahead` submits speculatively: "up to one window of fires may still be
//! in flight" when `on_token` breaks, and their cells are never taken. Never
//! taken, but EXECUTED — and therefore folded. For KV that is harmless by
//! construction, because a later fire's `kv_len` and page CSR only ever cover
//! valid tokens, so the overshoot is masked. A fold has no `kv_len`: it
//! advances on every fire that executes and cannot be rewound.
//!
//! Harmless today and only today — one request per process, the working set
//! is discarded when the turn ends, so nothing ever reads the over-advanced
//! fold. The moment sessions publish that state it is wrong, and wrong
//! silently: a resumed fold a few tokens ahead of its KV still generates
//! fluent text. Measured driver-side by the qwen-code session as
//! `recurrent slot 0 is at position 165, this fire starts at 160`.
//!
//! So before this loop's state can be sealed on a hybrid pass, the fold
//! position and the KV length have to agree BY CONSTRUCTION — either no
//! speculation on a hybrid pass (costing part of the 90 tok/s decode measured
//! with `run_ahead`), or sealing at the fold's position rather than the
//! accepted length, which is only sound once the stop token is written rather
//! than truncated-at. Do not build the session seam on top of this loop
//! without resolving it.
//!
//! Failures never escape as `Err` to the wire: [`generate`] reports the
//! first error and the caller degrades the turn to `finish_reason:"length"`
//! (the audit's overflow discipline — a KV/context overflow mid-decode is
//! "generate what fits", never a 4xx/5xx).

use inferlet::ptir::attention::prelude::*;
use std::ops::RangeBounds;

pub struct GenConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

/// In-graph top-p + temperature sampler over the read-out row logits
/// `[1,vocab]`. `r` is the taken `[2]` u32 rng state (`[key, ctr]`) driving
/// the Gumbel noise. Returns the sampled token `[1]` i32. Zero temperature
/// is exact greedy decoding; positive temperatures use nucleus sampling.
fn sample_token(r: &Tensor, temperature: f32, top_p: f32) -> Tensor {
    let logits = intrinsics::logits(); // [1, vocab] f32 (read-out row)
    if temperature == 0.0 {
        return reduce_argmax(&logits);
    }
    let scaled = &logits / temperature.max(1e-4);
    nucleus_sample(&scaled, top_p, r)
}

/// What this generator needs from a forward pass, over the interfaces it can
/// run on. The two `attention` signatures differ precisely so that an
/// attention-only algorithm cannot name a folded recurrent state; saying what
/// they have in common FOR THIS algorithm is the guest's job (the shape
/// `tests/inferlets/text-completion-bench` uses).
trait BindState {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>;
}

impl BindState for inferlet::ptir::attention::ForwardPass {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        // A pure-attention model has no recurrent state, and the type system
        // proves this set can only be empty.
        debug_assert!(rs.is_empty());
        self.attention(ws, geom)
    }
}

impl BindState for inferlet::ptir::hybrid::ForwardPass {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        // Serving folds every fire; nothing is ever buffered (see module docs).
        self.attention(
            Some(KvBinding {
                working_set: ws,
                geometry: geom,
            }),
            rs,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )
    }
}

/// Prefill `prompt` and decode up to `cfg.max_tokens` tokens, handing each
/// sampled token to `on_token` (which owns stop policy — `Break` ends the
/// turn). Returns the first engine error, if any; tokens already delivered
/// stand regardless.
pub async fn generate(
    prompt: &[u32],
    cfg: &GenConfig,
    mut on_token: impl FnMut(u32) -> std::ops::ControlFlow<()>,
) -> Option<String> {
    if cfg.max_tokens == 0 {
        return None;
    }
    match model::pass_kind() {
        model::ForwardKind::Attention => generate_attention(prompt, cfg, &mut on_token).await.err(),
        model::ForwardKind::Hybrid => generate_hybrid(prompt, cfg, &mut on_token).await.err(),
        // No registered model reports recurrent-only, and this loop's paged
        // prompt geometry has nothing to bind on a pass with no KV at all —
        // so it degrades the turn rather than pretending.
        model::ForwardKind::Recurrent => Some(
            "this model is recurrent-only; the serving inferlet has no KV-free path".to_string(),
        ),
    }
}

macro_rules! define_generate {
    ($name:ident, $kind:ident) => {
        async fn $name(
            prompt: &[u32],
            cfg: &GenConfig,
            on_token: &mut impl FnMut(u32) -> std::ops::ControlFlow<()>,
        ) -> Result<()> {
            use inferlet::ptir::$kind::{ForwardPass, run_ahead, submit_frame};

            let temperature = cfg.temperature;
            let top_p = cfg.top_p;
            let max_tokens = cfg.max_tokens;

            let mut prompt_tokens = prompt.to_vec();
            if prompt_tokens.is_empty() {
                prompt_tokens.push(0);
            }
            let n = prompt_tokens.len() as u32;

            // Tokens per pool page. Taken from the driver, never assumed: these
            // passes build their own write descriptors and page CSRs, and the driver
            // reads both back at ITS page size (see the reference inferlet's note).
            let page_t = kv_page_size();

            // Shared logical page pool: prompt + decode headroom, page-rounded.
            // `reserve` is purely logical (no memory held until a forward writes),
            // so the real capacity bound surfaces as a fire failure mid-turn — which
            // the caller degrades to finish_reason "length", never an error status.
            let pool_pages = (n + max_tokens as u32 + 2).div_ceil(page_t);
            let ws = WorkingSet::new();
            let slots = ws.reserve(pool_pages).context("ws.reserve")?;
            let pool_ids = slots.ids().to_vec();

            // The folded recurrent state, on the kinds that have one: one set
            // per request row (this loop is one row), shared by the prefill
            // chunks and the decode fires so decode continues the prefill's
            // state. `pass_kind() != Attention` is exactly the class predicate
            // the driver's rs arity check uses.
            let rs_ws: Vec<RsWorkingSet> =
                if model::pass_kind() != model::ForwardKind::Attention {
                    vec![RsWorkingSet::new()]
                } else {
                    Vec::new()
                };

            // ── ONE PIPELINE: prefill chunks and decode are one sequential stream.
            let pipe = Pipeline::new();

            // ───────────────── 1. PREFILL (chunked, C = max_embed_length) ─────────
            // Every chunk's epilogue samples (an epilogue put has to be drained or
            // the channel fills); only the LAST chunk's token continues the prompt.
            let prompt_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
            let spans = prefill_chunks(n, None);
            let mut g0 = 0i32;
            for &(base, end) in &spans {
                let len = end - base;
                let toks_p = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
                let embed_indptr_p = Channel::from([0u32, len]).named("embed_indptr_p");
                let positions_p = Channel::from_iter(base..end).named("positions_p");

                // Explicit write descriptor: cell c → pool_ids[c/page_t] @ c%page_t.
                let w_slot_pv: Vec<u32> =
                    (base..end).map(|c| pool_ids[(c / page_t) as usize]).collect();
                let w_off_pv: Vec<u32> = (base..end).map(|c| c % page_t).collect();
                let w_slot_p = Channel::from(w_slot_pv).named("w_slot_p");
                let w_off_p = Channel::from(w_off_pv).named("w_off_p");
                let klen_p = Channel::from([end]).named("klen_p");
                let pages_p = Channel::from(pool_ids.clone()).named("pages_p");
                // The page CSR tracks kv_len (= end), never the pool size — the
                // driver derives kv_len from it (reference inferlet, "SOURCE OF
                // TRUTH" note). No AttnMask port: causal is what the CSR already
                // says (same reference, R4-4 rationale).
                let page_indptr_p = Channel::from([0u32, end.div_ceil(page_t)]).named("pidx_p");
                let rng_p = Channel::from([0x51ed_u32, 0]).named("rng_p");
                let g0_ch = Channel::new([1], dtype::i32).named("g0");

                let fwd_p = ForwardPass::new();
                fwd_p.embed(&toks_p, &embed_indptr_p)?;
                fwd_p.bind_state(
                    &ws,
                    KvGeometry {
                        readable_pages: ..,
                        writable_pages: ..,
                        kv_len: &klen_p,
                        pages: &pages_p,
                        page_indptr: &page_indptr_p,
                        w_slot: &w_slot_p,
                        w_off: &w_off_p,
                        positions: &positions_p,
                        mask: None,
                    },
                    &rs_ws,
                )?;
                fwd_p.epilogue(move || {
                    let r = rng_p.take();
                    let tok = sample_token(&r, temperature, top_p);
                    let r_next = &r + iota(2);
                    g0_ch.put(&tok);
                    rng_p.put(&r_next);
                });

                fwd_p
                    .submit(&pipe)
                    .with_context(|| format!("prefill submit @{base}"))?;
                g0 = g0_ch
                    .take_host::<i32>()
                    .await
                    .with_context(|| format!("prefill take @{base}"))?;
            }

            // First sampled token — the caller decides whether it ends the turn.
            let done = matches!(on_token(g0 as u32), std::ops::ControlFlow::Break(()));

            // ───────────────── 2. DECODE LOOP (1-wide, run-ahead) ─────────────────
            let budget = if done {
                0
            } else {
                max_tokens.saturating_sub(1) // g0 already delivered
            };
            if budget > 0 {
                let pool_pages_total = pool_ids.len() as u32;
                let slot_n = pool_ids[(n / page_t) as usize];
                let tok_in = Channel::from([g0]).named("tok_in");
                let pos = Channel::from([n]).named("pos");
                let fill = Channel::from([n + 1]).named("fill");
                let klen = Channel::from([n + 1]).named("klen");
                let w_slot = Channel::from([slot_n]).named("w_slot");
                let w_off = Channel::from([n % page_t]).named("w_off");
                let pages = Channel::from(pool_ids.clone()).named("pages");
                let page_indptr =
                    Channel::from([0u32, (n + 1).div_ceil(page_t)]).named("page_indptr");
                let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
                let out = Channel::new([1], dtype::i32)
                    .capacity(channel_capacity() as u32)
                    .named("out");
                let rng = Channel::from([0x9e37_u32, 0]).named("rng");
                let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");

                let fwd = ForwardPass::new();
                fwd.embed(&tok_in, &lane1)?;
                fwd.bind_state(
                    &ws,
                    KvGeometry {
                        readable_pages: ..,
                        writable_pages: (n / page_t)..,
                        kv_len: &klen,
                        pages: &pages,
                        page_indptr: &page_indptr,
                        w_slot: &w_slot,
                        w_off: &w_off,
                        positions: &pos,
                        mask: None,
                    },
                    &rs_ws,
                )?;
                fwd.epilogue(move || {
                    // TAKES + compute first, PUTS last (value-id discipline).
                    let base = fill.take(); // [1] u32 — position this fire writes
                    let pids = pool_ids_ch.take();
                    let r = rng.take();

                    let tok = sample_token(&r, temperature, top_p); // [1] i32
                    let r_next = &r + iota(2);

                    let logical_slot = &base / page_t;
                    let w_slot_v = gather(&pids, &logical_slot);
                    let w_off_v = &base % page_t;
                    let klen_v = &base + 1u32;
                    let next_free = &base + 1u32;
                    let pages_v = reshape(&pids, [pool_pages_total]);
                    // Page count tracks the new kv length, never the pool size.
                    let page_count = klen_v.div_ceil(page_t);
                    let pidx_v = indptr(1, &page_count);

                    // Device-resolved geometry is loop-carried: the host never
                    // drains these rings, so every fire's values are re-put here.
                    tok_in.put(&tok);
                    out.put(&tok);
                    w_slot.put(&w_slot_v);
                    w_off.put(&w_off_v);
                    klen.put(&klen_v);
                    pos.put(&base);
                    fill.put(&next_free);
                    pages.put(&pages_v);
                    page_indptr.put(&pidx_v);
                    rng.put(&r_next);
                    pool_ids_ch.put(&pids);
                });

                // MEASUREMENT SWITCH (not a shipped feature — see the
                // module's overshoot seam). The default `run_ahead` path
                // speculates: up to one window of fires stays in flight when
                // the callback breaks, executed and therefore FOLDED. The
                // sequential path submits one fire and takes its result
                // before submitting the next, so nothing is ever folded that
                // the turn did not accept — the property a hybrid seal needs.
                //
                // It is hand-written rather than `run_ahead(.., 1, ..)` in a
                // loop, because `run_ahead` calls `on.close()` as soon as its
                // budget is spent (ptir.rs: "call close right after the last
                // submit"), so a second call submits into a closed pipeline.
                // That does not error — it HANGS, which cost a 300 s timeout
                // to discover.
                const SEQUENTIAL_DECODE: bool = false;
                if !SEQUENTIAL_DECODE {
                    run_ahead(&pipe, &fwd, budget, async || {
                        let t = out.take_host::<Vec<i32>>().await?;
                        let token = *t.first().unwrap_or(&0) as u32;
                        Ok(on_token(token))
                    })
                    .await?;
                } else {
                    for _ in 0..budget {
                        submit_frame(&pipe, &[Some(&fwd)])?;
                        let t = out.take_host::<Vec<i32>>().await?;
                        let token = *t.first().unwrap_or(&0) as u32;
                        if on_token(token).is_break() {
                            break;
                        }
                    }
                    pipe.close();
                }
            }

            // Any fire still in flight after an early stop is left untaken; close
            // releases the scheduler wait-set, reclaims them, and rejects further
            // submissions.
            pipe.close();
            Ok(())
        }
    };
}

define_generate!(generate_attention, attention);
define_generate!(generate_hybrid, hybrid);
