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

/// What the prefix cache asks of one turn: where to start, and what to park.
///
/// `resume` is `None` on a miss (and on the first turn of a conversation), in
/// which case this is exactly the cold path that shipped before — same page
/// pool, same `writable_pages: ..`, same spans.
pub struct Apc {
    /// KV already parked for a page-aligned prefix of `prompt`.
    pub resume: Option<crate::apc::Resume>,
    /// `(page-aligned cut, address)` pairs to park once prefill has written
    /// them. Every cut must be within the history — never the cue.
    pub publish: Vec<(u32, String)>,
}

/// What actually happened, as opposed to what was planned.
///
/// `cached_tokens` is read back from the generator rather than assumed from the
/// plan, because the generator REFUSES a resume whose bookkeeping disagrees
/// with its working set (see `define_generate!`) and rebuilds instead. Reporting
/// the planned depth would then overstate `usage.cached_tokens` on exactly the
/// turns that paid full price — the shape of prefix-cache defect that hides
/// behind a green hit flag.
pub struct Outcome {
    /// First engine error, if any. Tokens already delivered stand regardless.
    pub error: Option<String>,
    /// Prompt tokens served from parked KV. Zero on a cold or refused resume.
    pub cached_tokens: u32,
    /// Cuts successfully parked for later turns.
    pub published: usize,
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
    apc: Apc,
    mut on_token: impl FnMut(u32) -> std::ops::ControlFlow<()>,
) -> Outcome {
    let mut out = Outcome {
        error: None,
        cached_tokens: 0,
        published: 0,
    };
    if cfg.max_tokens == 0 {
        return out;
    }
    let result = match model::pass_kind() {
        model::ForwardKind::Attention => {
            generate_attention(prompt, cfg, apc, &mut out, &mut on_token).await
        }
        model::ForwardKind::Hybrid => {
            generate_hybrid(prompt, cfg, apc, &mut out, &mut on_token).await
        }
        // No registered model reports recurrent-only, and this loop's paged
        // prompt geometry has nothing to bind on a pass with no KV at all —
        // so it degrades the turn rather than pretending.
        model::ForwardKind::Recurrent => Err(
            "this model is recurrent-only; the serving inferlet has no KV-free path".to_string(),
        ),
    };
    out.error = result.err();
    out
}

macro_rules! define_generate {
    ($name:ident, $kind:ident) => {
        async fn $name(
            prompt: &[u32],
            cfg: &GenConfig,
            apc: Apc,
            outcome: &mut Outcome,
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

            // ── RESUME, or the cold path ─────────────────────────────────────
            // A resumed working set holds a page-aligned prefix of THIS prompt,
            // parked by an earlier request's instance. Three things have to hold
            // before its geometry is sound, and none of them is guaranteed by
            // the address alone:
            //
            //   cached < n            — else this fire has nothing to write, and
            //                           the engine refuses an empty writable
            //                           declaration outright;
            //   cached % page_t == 0  — else `slice` parked fewer tokens than the
            //                           address names, and the suffix would graft
            //                           onto a prefix ending in the wrong place;
            //   page_len == cached/page_t
            //                         — the parked set is exactly its prefix, so
            //                           anything else means the entry was built by
            //                           a different publisher than this module.
            //
            // Disagreement DROPS the resume and rebuilds, the session arm's
            // `kv_verify` discipline: a rebuild is slow, a wrong prefix is silent.
            // Not an error — a correct turn at full price is still a correct turn.
            let mut resumed_address: Option<String> = None;
            let (ws, cached) = match apc.resume {
                Some(r) => {
                    let have = r.ws.page_len();
                    if r.cached_tokens < n
                        && r.cached_tokens % page_t == 0
                        && have == r.cached_tokens / page_t
                    {
                        resumed_address = Some(r.address);
                        (r.ws, r.cached_tokens)
                    } else {
                        eprintln!(
                            "[apc] refusing a resume at {} tokens (prompt {n}, page {page_t}, \
                             parked pages {have}); rebuilding",
                            r.cached_tokens
                        );
                        // An entry that fails its own geometry is permanently
                        // unusable — the address no longer describes the pages —
                        // and it re-fails every later lookup while pinning its
                        // chain. Removing it is the kv_verify discipline the
                        // session arm applies to a bad branch: drop, don't hoard.
                        match WorkingSet::remove_index(r.address.as_bytes()) {
                            Ok(_) => {}
                            Err(e) => eprintln!("[apc] remove of bad entry failed: {e}"),
                        }
                        (WorkingSet::new(), 0)
                    }
                }
                None => (WorkingSet::new(), 0),
            };
            outcome.cached_tokens = cached;

            // Shared logical page pool: prompt + decode headroom, page-rounded.
            // `reserve` is purely logical (no memory held until a forward writes),
            // so the real capacity bound surfaces as a fire failure mid-turn — which
            // the caller degrades to finish_reason "length", never an error status.
            //
            // On a resume the pool already contains the parked pages, so only the
            // shortfall is reserved. `pool_ids` is built as `[already there] ++
            // [granted]` rather than `0..pool_pages`, so the page table stays
            // correct even if a grant ever comes back non-adjacent — the one
            // assumption in this geometry that costs nothing to avoid making.
            let have = ws.page_len();
            // QUANTISED, and this is not a micro-optimisation — it is what keeps
            // the driver's program cache from filling.
            //
            // `pool_pages` reaches the traced graph as a SHAPE CONSTANT (the
            // decode epilogue's `reshape(&pids, [pool_pages_total])`), so every
            // distinct value compiles a distinct program. Computed exactly, it
            // changes with every prompt length — one new program per turn. The
            // Metal driver caches 64 (`m1_runtime.cpp` `kMaxProgramCacheEntries`),
            // and past that `register_program` answers "executable cache is full"
            // and every later turn degrades to `finish_reason:"length"`.
            //
            // Measured: with the context ring at 16,384 nothing reached it,
            // because trajectories died on the ring first. Raising the ring to
            // 65,536 let two instances run 6-10x longer, they compiled ~36
            // programs between them, and the three instances after them failed in
            // 2 s, 2 s and 21 s against a server that could no longer register
            // anything. One wall traded for another.
            //
            // Rounding to `POOL_GRANULARITY` pages collapses that to a handful of
            // shapes. It costs nothing real: `reserve` is purely logical — no
            // memory is held until a forward writes — so an over-reservation is a
            // longer page-id list and nothing else.
            //
            // This is ONE source, not provably the only one: a prefill chunk's
            // token count also reaches the trace, and a resumed turn's delta is a
            // different length every turn. Whether that alone can still fill the
            // cache is a measurement, not an argument, so the next run counts
            // `executable cache is full` rather than assuming this fixed it.
            //
            // The durable fix is not here. A cache that REJECTS when full instead
            // of evicting turns a capacity limit into a dead server, and no guest
            // can be written that never needs a 65th shape.
            // POWER-OF-TWO, not a fixed 256-page step, and the difference is
            // the server's concurrency ceiling.
            //
            // A fixed 256-page granularity reserves 8192 tokens for EVERY
            // request no matter how short. `reserve` is logical, but the
            // cluster admission gate is not: with `total_pages = 2048` that is
            // 2048/256 = exactly 8 concurrent requests, and the ninth is
            // refused with
            //
            //   admission rejected: cluster saturated: no healthy worker has
            //   KV/seq headroom
            //
            // Measured 2026-08-14: 8 of 32 concurrent requests completed, 24
            // were rejected at 0.0 s. The comment this replaces claimed
            // rounding "costs nothing real ... an over-reservation is a longer
            // page-id list and nothing else". That was wrong: it costs the
            // concurrency ceiling, and it cost it silently because the
            // rejection surfaces as a 503 rather than anywhere near this code.
            //
            // Powers of two keep the property the rounding existed for -- the
            // pool size is a CHANNEL SHAPE, and shape churn compiles a new
            // program (~600 ms) -- while making the waste proportional. A
            // 12-shape ladder (8, 16, ... 2048 pages) sits far inside the
            // 64-entry program cache, and over-reservation is now at most 2x
            // the request's own need instead of a flat 8192 tokens. A short
            // request reserves 512 tokens rather than 8192, so the same pool
            // holds 64+ of them; a long conversation still lands on 256 or 512
            // and is unaffected.
            const POOL_FLOOR_PAGES: u32 = 8;
            // ...EXCEPT near the top of the pool, where "at most 2x the need"
            // stops being proportional waste and becomes the whole pool. On a
            // hybrid the pool is exactly one max-length conversation
            // (`context.cpp` clamps it to `ceil(max_model_len/page)`), so the
            // first turn whose power of two lands on the full pool makes the
            // pool report 100% occupied, the pressure bucket saturates, and
            // the NEXT request is refused. Measured: a ramped conversation
            // died 503 after 30,855 tokens — need crossed 1,024 pages, rounded
            // to 2,048 of 2,048 — which is `9dc2bd785`'s phantom-reservation
            // kill (B, at 55,099) reincarnated through coarser rounding. The
            // session arm's fix ports directly: keep the coarse ladder while
            // it cannot hurt, and narrow to a fine step once the power of two
            // would eat the admission headroom (~94% pressure gate, mirrored
            // as 6% kept free — same constant, same pointer as opencode-
            // session's `enforce_retention`). The fine band adds at most a
            // couple of 128-page shapes to the program-cache ladder.
            const FINE_STEP_PAGES: u32 = 128;
            let need = (n + max_tokens as u32 + 2)
                .div_ceil(page_t)
                .max(POOL_FLOOR_PAGES);
            let (_, pool_total) = kv_pool_status();
            let p2 = need.next_power_of_two();
            let band = pool_total.saturating_sub(pool_total / 100 * 6);
            let pool_pages = if pool_total > 0 && p2 > band {
                // Fine rounding, capped at the band — and never below the
                // genuine need: a turn that truly requires more than the band
                // reserves what it needs, and the refusal that may follow is
                // the true pool limit speaking, not the rounding.
                need.next_multiple_of(FINE_STEP_PAGES).min(band).max(need)
            } else {
                p2
            }
            .max(have);
            let mut pool_ids: Vec<u32> = (0..have).collect();
            if pool_pages > have {
                let slots = ws.reserve(pool_pages - have).context("ws.reserve")?;
                pool_ids.extend_from_slice(slots.ids());
            }

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
            //
            // On a resume only the SUFFIX is submitted: `cached..n` instead of
            // `0..n`. That is the whole saving — the parked pages are read by
            // attention (they are inside `readable_pages`) but never re-embedded,
            // never re-projected, and never rewritten.
            let prompt_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
            let spans = prefill_chunks(n - cached, None);
            let mut g0 = 0i32;
            for &(span_base, span_end) in &spans {
                // `prefill_chunks` splits a LENGTH evenly; these are absolute
                // positions in the prompt, which is what every channel below
                // (tokens, positions, write descriptors) is indexed by.
                let (base, end) = (span_base + cached, span_end + cached);
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
                        // Everything parked plus everything this turn writes is
                        // readable; only the pages at or past the resume point are
                        // writable. The parked pages are structurally SHARED with
                        // whatever else resumed from the same address, so a write
                        // into them would corrupt another request's prefix. The
                        // bound is enforced device-side as `kv_write_lower_bounds`
                        // (`runtime/engine/src/pipeline/fire.rs:1520`), not merely
                        // asserted here.
                        //
                        // `cached == 0` makes this `0..`, which is `..` — the cold
                        // path is unchanged, not a special case.
                        readable_pages: ..,
                        writable_pages: (cached / page_t)..,
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
            // Held rather than propagated, so a turn that dies mid-DECODE still
            // parks its history below. What gets parked is prefill's output, and
            // by here prefill has succeeded — dropping it because generation
            // failed would make the NEXT turn pay full price too, on a
            // conversation that is evidently already near a limit.
            //
            // A prefill failure is the opposite case and must NOT reach the
            // parking block: it still returns early through `?` above, on
            // purpose. `publish` can only check `ws.page_len()`, which is the
            // RESERVED extent, not the WRITTEN one — so after a prefill that
            // died on chunk 3 of 5 the guard would happily park pages 0..cut
            // that no forward ever wrote, under an address claiming they hold
            // that prefix. Every later turn would then resume onto uninitialised
            // KV and generate fluently from nothing.
            //
            // Observed live rather than reasoned about: on the 5-instance soak
            // two turns hit `20061` prompt tokens against a 16,384-token context
            // ring (`max_model_len / kv_page_size`, the clamp in
            // `context.cpp:245 effective_total_pages()`), degraded to
            // `finish_reason:"length"`, and correctly logged `parked 0`.
            let mut decode_error: Option<String> = None;
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
                // Ring sized a full frame of margin ABOVE the advertised
                // capacity, not at it. `channel_capacity()` already bakes in a
                // staging margin, but the engine's ticket check is more
                // conservative still, and a continuation landing inside that
                // margin is SILENTLY SKIPPED at reader-cell validation rather
                // than refused. `text-completion-bench` measured the
                // continuation lost on 12% of frames when sized at exactly
                // `cap` — run-ahead collapsing with no error anywhere — and
                // sizes at `cap + 7 * live_slots`. We inherited the exact-`cap`
                // form from the `tests/inferlets/chat-completion` reference.
                let out = Channel::new([1], dtype::i32)
                    .capacity((channel_capacity() + 7 * live_slots()) as u32)
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
                    decode_error = run_ahead(&pipe, &fwd, budget, async || {
                        let t = out.take_host::<Vec<i32>>().await?;
                        let token = *t.first().unwrap_or(&0) as u32;
                        Ok(on_token(token))
                    })
                    .await
                    .err();
                } else {
                    for _ in 0..budget {
                        if let Err(e) = submit_frame(&pipe, &[Some(&fwd)]) {
                            decode_error = Some(e);
                            break;
                        }
                        let t = match out.take_host::<Vec<i32>>().await {
                            Ok(t) => t,
                            Err(e) => {
                                decode_error = Some(e.to_string());
                                break;
                            }
                        };
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

            // ───────────────── 3. PARK THE PREFIX ─────────────────────────────
            // AFTER every write, and this ordering is forced, not stylistic.
            //
            // `slice` makes the parent's pages structurally shared, so the next
            // fire that writes near them needs a copy-on-write KV copy — and on
            // the Metal driver `copy_kv` answers PIE_STATUS_UNSUPPORTED for any
            // checkpoint that is not a GDN hybrid (`driver/metal/src/context.cpp`,
            // `copy_kv_impl`: "this increment only supports the qwen3.6
            // (GDN-hybrid) checkpoint geometry"). Qwen3-Coder-30B is pure
            // attention, so parking before decode poisoned the decode channel
            // with `pre-launch KV copy rejected: pie_metal_copy_kv failed with
            // status -3` and the turn returned its FIRST TOKEN ONLY — a fluent,
            // plausible, truncated answer that no wire-level check would flag.
            //
            // Parking last leaves no fire after the sharing is introduced, so no
            // copy is ever planned. A fresh pipeline is needed because `slice` is
            // ordered on one and `run_ahead` closed the turn's: the history pages
            // settled during prefill, whose takes were awaited, so there is
            // nothing left for the ordering to protect.
            if !apc.publish.is_empty() {
                let park = Pipeline::new();
                for (cut, address) in &apc.publish {
                    if crate::apc::publish(&park, &ws, *cut, address, page_t) {
                        outcome.published += 1;
                    }
                }
                park.close();
            }

            // ─────────── 4. RETIRE THE ENTRY THIS TURN RESUMED FROM ───────────
            // The parked entry pins the previous turn's page chain, and the
            // extension above privatized the stratum it shared — so until the
            // old entry goes, the pool holds TWO generations of this
            // conversation. Measured before this existed: a single ramped
            // conversation died 503 at 30,855 tokens, half of the session arm's
            // 61,711 on the identical pool, with live chain + previous entry +
            // reservation summing to the pool exactly. Under the agent
            // benchmark that halved ceiling surfaced as 172 refusals in five
            // instances (35% of calls) once real conversations passed ~30k.
            //
            // Removal is gated on this turn having PARKED a replacement
            // (`published > 0`): a turn that failed to park must leave the old
            // entry resumable, or the next turn pays a full cold rebuild for
            // this turn's failure. And an address this turn re-published is
            // skipped — `update_index` on the same key already replaced the
            // entry, and removing it would delete the fresh park, turning
            // every no-growth turn (a retry, an idempotent tool loop) into a
            // permanent cache miss.
            if let Some(addr) = resumed_address {
                let republished = apc.publish.iter().any(|(_, a)| a == &addr);
                if outcome.published > 0 && !republished {
                    match WorkingSet::remove_index(addr.as_bytes()) {
                        Ok(_) => {}
                        Err(e) => eprintln!("[apc] retire of superseded entry failed: {e}"),
                    }
                }
            }

            match decode_error {
                Some(e) => Err(e),
                None => Ok(()),
            }
        }
    };
}

define_generate!(generate_attention, attention);
define_generate!(generate_hybrid, hybrid);
