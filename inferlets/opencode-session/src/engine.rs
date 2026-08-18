//! Resumable PTIR generation: fork-or-fresh session state, chunked prefill of
//! the turn's rendered delta, then cue + sequential decode on scratch state that
//! is thrown away. What survives the turn is rendered history and nothing else.
//!
//! This is Strategy A's `chat-completions/src/engine.rs` with the session seam
//! closed. Everything that module marked as SEAM is implemented here; the parts
//! it got right (both pass kinds behind one binding trait, the decode ring
//! sized ABOVE `channel_capacity()`, degrade-never-error) are carried over
//! unchanged. The session machinery is ported from the qwen-code port
//! (`~/Documents/Liszt_ai/pie-qwen`, `liu/qwen-code-dev`), which has been
//! running this shape live; the deltas from it are called out below and in
//! `docs/opencode-integration-progress.md`.
//!
//! ## Why the state is `{ ws, rs }` and never just `ws`
//!
//! On a hybrid (GDN) model the folded recurrent state is part of the
//! conversation prefix exactly as much as the KV pages are. Retaining KV at
//! token N while the fold still stands at 0 does not error — it silently
//! serves a *different context* and generates fluent, wrong output. So the two
//! are forked, extended, sealed and retained as one unit, or not at all.
//!
//! ## Why decode is sequential, and why that is not a tuning choice
//!
//! `run_ahead` keeps a window of fires in flight and lets the tail of it run
//! past the stop token — the SDK's own words: "up to one window of fires may
//! still be in flight … their cells are simply never taken". Never *taken*,
//! but EXECUTED, and therefore FOLDED. For KV that is harmless by
//! construction: every later fire's `kv_len` and page CSR only ever cover
//! valid tokens, so the overshoot is masked. A fold has no `kv_len`. It
//! advances on every fire that executes and cannot be rewound.
//!
//! Under Strategy A that never mattered — the working set died with the
//! request, so nothing read the over-advanced fold. Here the state is reused
//! every turn, so the error compounds across a session. Measured driver-side
//! by the qwen-code session as `recurrent slot 0 is at position 165, this fire
//! starts at 160`.
//!
//! Sequential decode — submit one fire, take its result, submit the next —
//! makes the fold position and `kv_len` agree by construction. Measured cost
//! on Qwen3.6-35B-A3B: 0.4% (91.0 → 90.6 tok/s single-stream), against the ~3×
//! that KV reuse is worth on an agent loop.
//!
//! It is hand-written rather than `run_ahead(.., 1, ..)` in a loop, because
//! `run_ahead` calls `on.close()` as soon as its budget is spent, so the second
//! call submits into a closed pipeline. That does not error — it HANGS.
//!
//! ## Nothing generated is ever retained — and that removes the seal problem
//!
//! A turn is three spans: the rendered delta, the generation cue, and the
//! decoded tokens. Only the FIRST is retained. `render_len` is the boundary,
//! and the state handed back to the caller is valid exactly up to it.
//!
//! The obvious alternative — retain through the generated turn, appending
//! `<|im_end|>\n` to "seal" it — is what the qwen-code port does, and it is
//! wrong in a way that took a differential test to see. On Qwen the generation
//! cue is `<|im_start|>assistant\n<think>\n\n</think>\n\n`, while replaying that
//! same turn as history renders `<|im_start|>assistant\n` with thinking
//! stripped. A sealed state therefore holds ~4 tokens that no re-render of the
//! conversation will ever produce, every turn, cumulatively. Measured here: a
//! resumed prompt of 64 tokens against a cold rebuild of 60, answering
//! differently at temperature 0. The same asymmetry exists in HF's template; it
//! is invisible only because a stateless server re-renders every turn.
//!
//! Retaining at `render_len` makes the retained state `render(messages[..split])`
//! by construction, which concatenates with `render(messages[split..]) + cue` to
//! exactly the full render. The assistant turn is re-prefilled next turn from
//! the client's echo — tens to a few hundred tokens against a history of tens of
//! thousands.
//!
//! It also deletes a whole hazard class. There is no seal fire, so the
//! qwen-code seal saga — a free-standing `seal()` cannot bind a fold on a fresh
//! pipeline (poison epoch), and forking it there mints a new sequence id
//! ("recurrent slot 1 holds sequence 2^63, this fire is sequence 2^63+1") —
//! simply does not arise. And the address no longer hashes anything the server
//! produced, so the response/save unification invariant (a trim or a fallback in
//! the response path silently breaking every later resume) has nothing left to
//! break.

use inferlet::model;
use inferlet::pie::inferlet::forward::ForwardPass as WitAttention;
use inferlet::pie::inferlet::forward_hybrid::ForwardPass as WitHybrid;
use inferlet::ptir::attention::prelude::*;
use inferlet::ptir::{KvBinding, Pass, PassWit, RsGeometry, RsWorkingSet};
use std::ops::{ControlFlow, RangeBounds};

/// The retained per-session device state: KV pages plus, on a recurrent-state
/// model, the folded state belonging to the same prefix. See the module docs
/// for why these two cannot be retained separately.
pub struct SessionState {
    pub ws: WorkingSet,
    /// Empty on attention-only models. Exactly one entry on a recurrent-state
    /// model: the driver wants one rs working set per request row, and this
    /// algorithm fires a single row. Held as the slice the bind takes, since
    /// `RsWorkingSet` is deliberately not `Clone` — a folded state has one
    /// owner and is shared only through `fork`.
    pub rs: Vec<RsWorkingSet>,
}

impl SessionState {
    /// Fresh state for whichever forward kind this model reports.
    pub fn new() -> Self {
        SessionState {
            ws: WorkingSet::new(),
            rs: match model::pass_kind() {
                model::ForwardKind::Attention => Vec::new(),
                _ => vec![RsWorkingSet::new()],
            },
        }
    }
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

/// How this turn gets its starting state.
///
/// There is still no fork variant HERE, but that is now a statement about
/// where the fork happens rather than about whether forking works. A turn
/// resumes by extending the state it was given, in place; the fork it needs
/// is taken later, at the scratch boundary in `generate`, so that the thing
/// retained is the fold as it stood at `render_len`.
///
/// Both halves of a copy-on-write branch work on Metal now:
///
/// - **KV.** The scheduler built every pre-launch CoW plan with
///   `PIE_MEMORY_DOMAIN_CUDA_DEVICE` hardcoded, and Metal refuses a copy whose
///   domain is not `METAL_SHARED`. Fixed at the backend boundary
///   (`runtime/engine/src/driver/backend.rs::copy_kv`).
/// - **The fold.** `RsWorkingSet::fork` shares the folded slot by refcount and
///   the first write copies it. That copy used to strand the child: a
///   recurrent fire's sequence id is DERIVED from its slot, `(1<<63) |
///   rs_slot_id`, and `MetalExecutor::copy_state` carried the PARENT's id onto
///   the copy, so the child announced an id its own record contradicted —
///   `recurrent slot 1 holds sequence 2^63, this fire is sequence 2^63+1`.
///   The driver now rebases the id onto the destination slot
///   (`rebase_linear_sequence`), which is what makes a CoW child able to
///   continue itself.
///
/// The cost that remains is concurrency, not correctness: two turns still do
/// not extend one parent at the same time. Turns are sequential and retention
/// drops the parent it extended, so nothing does that today — and B-3 (forking
/// the working set for subagents) is no longer blocked on the recurrent side.
pub enum Resume {
    /// First turn: build fresh state.
    Cold,
    /// Extend owned state in place. Returned in `Generation::state`.
    InPlace(SessionState, u32),
}

pub struct GenConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

pub struct Generation {
    /// The extended state, valid to `total_len` and ready to retain — or to
    /// drop, if `gen_error` is set.
    pub state: SessionState,
    /// Valid tokens in the retained state: the RENDER boundary, not the end of
    /// generation. On a hybrid model the fold stands here too, because every
    /// fire past this point went to a fork. See the module docs.
    pub total_len: u32,
    /// Accepted generated tokens (what the caller surfaced), for `hit_max`
    /// and usage accounting. The tokens themselves went to `on_token`.
    pub accepted: usize,
    pub hit_max: bool,
    /// Set when generation died mid-turn. The caller degrades the turn to
    /// `finish_reason:"length"` (never a context-length 4xx — opencode
    /// retries 5xx without bound) and MUST NOT retain the state: a partially
    /// folded turn is worse than a clean miss.
    pub gen_error: Option<String>,
}

/// In-graph top-p + temperature sampler over the read-out row logits
/// `[1,vocab]`. `r` is the taken `[2]` u32 rng state (`[key, ctr]`) driving the
/// Gumbel noise. Zero temperature is exact greedy decoding.
fn sample_token(r: &Tensor, temperature: f32, top_p: f32) -> Tensor {
    let logits = intrinsics::logits();
    if temperature <= 0.0 {
        return reduce_argmax(&logits);
    }
    let scaled = &logits / temperature.max(1e-4);
    nucleus_sample(&scaled, top_p, r)
}

/// What this generator needs from a forward pass, over the interfaces it can
/// run on.
///
/// `pie:inferlet` exposes `forward-attention`, `forward-hybrid` and
/// `forward-recurrent` as three *unrelated* wit-bindgen types whose `attention`
/// signatures deliberately diverge — an attention-only algorithm must not be
/// able to name a folded recurrent state. Saying what they have in common FOR
/// THIS algorithm is the guest's job. The driver REJECTS a forward whose
/// rs-working-set count does not match its request rows:
///
/// ```text
/// resolved forward has 1 request row(s), but recurrent-state model bound
/// 0 rs-working-set(s); expected 1
/// ```
///
/// which arrives as a submit failure and serves an empty completion in ~50 ms —
/// the exact symptom of the first Qwen3.6-35B-A3B bring-up.
trait BindState {
    /// `rs_fold` is the recurrent fold mode for this fire:
    ///
    /// - `None` — fold everything. Rendered history: it belongs in the state.
    /// - `Some(ch)` — fold `ch` tokens (we always pass 0) and BUFFER the rest.
    ///   Generation scaffolding: the cue and the decoded tokens go into the
    ///   recurrent buffer instead of the fold, so they are visible to attention
    ///   for this turn and cost nothing to abandon afterwards.
    ///
    /// The channel is owned by the caller because a bound port must outlive the
    /// `bind_state` call and stay alive until `submit`.
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
        rs_fold: Option<&Channel>,
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
        _rs_fold: Option<&Channel>,
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        // A pure-attention model has no recurrent state, and the type system
        // proves this set can only be empty. Nothing to fold, so the mode is
        // meaningless here.
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
        rs_fold: Option<&Channel>,
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        // The SAME rs working set is bound by every fire of the turn, so each
        // continues the previous one rather than restarting it. What differs is
        // whether the fire's tokens land in the FOLD or in the BUFFER — see
        // `bind_state`'s docs and the module docs on why generation must not
        // fold.
        let kv = Some(KvBinding {
            working_set: ws,
            geometry: geom,
        });
        match rs_fold {
            None => self.attention(
                kv,
                rs,
                RsGeometry {
                    fold_len: None,
                    buffer: 0..0,
                },
            ),
            Some(ch) => self.attention(
                kv,
                rs,
                RsGeometry {
                    fold_len: Some(ch),
                    buffer: ..,
                },
            ),
        }
    }
}

/// Run one turn on top of `resume` (or from nothing), then seal it.
///
/// `prefill` is the rendered turn suffix on a resume hit and the full rendered
/// history on a miss — this module does not care which, only that the caller's
/// `resume` token count and `prefill` tokens are contiguous.
///
/// `on_prefill_chunk` fires after each committed prefill chunk (the keepalive
/// point on a multi-thousand-token prompt). `on_token` receives each accepted
/// token and owns all stop policy; `Break` ends the turn.
pub async fn generate(
    resume: Resume,
    delta: &[u32],
    cue: &[u32],
    cfg: &GenConfig,
    stop_ids: &[u32],
    on_prefill_chunk: impl FnMut(),
    on_token: impl FnMut(u32) -> ControlFlow<()>,
) -> Result<Generation> {
    match model::pass_kind() {
        model::ForwardKind::Attention => {
            generate_for::<WitAttention>(
                resume,
                delta,
                cue,
                cfg,
                stop_ids,
                on_prefill_chunk,
                on_token,
            )
            .await
        }
        model::ForwardKind::Hybrid => {
            generate_for::<WitHybrid>(
                resume,
                delta,
                cue,
                cfg,
                stop_ids,
                on_prefill_chunk,
                on_token,
            )
            .await
        }
        // No registered model reports recurrent-only, and this loop's paged
        // prompt geometry has nothing to bind on a pass with no KV at all — so
        // it degrades the turn rather than pretending.
        model::ForwardKind::Recurrent => Err(
            "this model is recurrent-only; the session inferlet has no KV-free path".to_string(),
        ),
    }
}

async fn generate_for<W>(
    resume: Resume,
    delta: &[u32],
    cue: &[u32],
    cfg: &GenConfig,
    stop_ids: &[u32],
    mut on_prefill_chunk: impl FnMut(),
    mut on_token: impl FnMut(u32) -> ControlFlow<()>,
) -> Result<Generation>
where
    W: PassWit,
    Pass<W>: BindState,
{
    let t0 = std::time::Instant::now();
    let page_t = kv_page_size();
    let pipe = Pipeline::new();

    // Fresh, forked, or extended in place — see `Resume` for why the last one
    // exists and when it is the only option that runs.
    let (state, n0) = match resume {
        Resume::Cold => (SessionState::new(), 0),
        Resume::InPlace(owned, cached) => (owned, cached),
    };

    let n_delta = delta.len() as u32;
    if n_delta == 0 {
        return Err("empty delta".to_string());
    }
    // The RETENTION BOUNDARY. Everything below `render_len` is rendered
    // history and nothing else — no cue, no generated token. That is what makes
    // the retained state a true prefix of any later full render.
    let render_len = n0 + n_delta;
    let n = render_len + cue.len() as u32;

    // Logical page pool: history + cue + decode budget, page-rounded. A fork
    // already carries the parent's pages, so reserve only the shortfall.
    // `reserve` is purely logical, so the real capacity bound surfaces as a
    // fire failure mid-turn — which the caller degrades to finish_reason
    // "length", never an error status.
    // QUANTIZED, and that is a performance fix rather than bookkeeping.
    //
    // `pool_pages` becomes the length of the `pages_p` channel in every fire of
    // this turn, and a channel's SHAPE is part of the program container
    // (`ChannelDecl.shape`; only the seed value is per-instance). The exact page
    // count grows by ~6 every turn as the history does, so an unquantized pool
    // hands the driver a container it has never seen on EVERY turn, and the
    // driver compiles one. Measured on the 7.4k-token replay: a 189-token
    // prefill took 1159 ms, against ~590 ms at the same turn's own per-token
    // rate -- ~570 ms of compile, on every turn, matching the ~600 ms first-fire
    // cost `decode-rows-probe` measures for any unseen shape.
    //
    // Rounding collapses that to one compile per 8192 tokens of growth. It costs
    // nothing real: `reserve` is purely logical -- no memory is held until a
    // forward writes -- so an over-reservation is a longer page-id list and
    // nothing else. Same constant and same reasoning as
    // `inferlets/chat-completions`, which got this fix first.
    const POOL_GRANULARITY: u32 = 256;
    let pool_pages_want = (n + cfg.max_tokens as u32 + 2)
        .div_ceil(page_t)
        .next_multiple_of(POOL_GRANULARITY);
    let have = state.ws.page_len();
    if pool_pages_want > have {
        state.ws.reserve(pool_pages_want - have).context("ws.reserve")?;
    }
    let pool_pages = pool_pages_want.max(have);

    let temperature = cfg.temperature;
    let top_p = cfg.top_p;

    // ── 1. Prefill the RENDERED DELTA onto the state we will retain ──────
    // Each chunk is one fire: positions and write descriptors cover the chunk,
    // `kv_len` and the page CSR cover everything valid so far. Chunking is not
    // optional — serving prompts run to many thousands of tokens and a one-shot
    // fire cannot exceed the driver's `max_embed_length()`.
    //
    // Nothing here samples: the read-out rows are mid-prompt. The turn's first
    // token comes from the cue, in phase 2.
    prefill_span::<W>(
        &pipe,
        &state,
        n0,
        delta,
        pool_pages,
        temperature,
        top_p,
        &mut on_prefill_chunk,
    )
    .await?;
    let t_prefill = t0.elapsed();

    // ── 2. Everything from here is SCRATCH ───────────────────────────────
    // The cue and the decoded tokens must not end up in the retained state. In
    // KV that is free — they are written past `render_len`, into cells the next
    // turn overwrites, and no later fire's `kv_len` ever covers them.
    //
    // A fold has no `kv_len` and cannot be rewound, so the recurrent half needs
    // the pre-generation state kept rather than reconstructed. FORK IT HERE:
    // `RsWorkingSet::fork` is an O(1) refcount share, and the first writing
    // fire of the generation below copies the folded slot on write. The child
    // this returns therefore keeps the fold exactly as it stands at
    // `render_len`, while generation runs away from it on a private copy. That
    // child is what gets retained.
    //
    // This is the "generate on a fork" shape the previous design tried and
    // abandoned, and the reason it did not work was never the shape. A
    // recurrent fire's sequence id is DERIVED from its slot — `(1<<63) |
    // rs_slot_id` — and `MetalExecutor::copy_state` carried the parent's id
    // onto the copy, so the child announced an id its own record contradicted:
    //
    // ```text
    // [pie-driver-metal] instance 2 launch failed: paged continuation:
    //   recurrent slot 1 holds sequence 9223372036854775808,
    //   this fire is sequence 9223372036854775809
    // ```
    //
    // The driver now rebases that id onto the destination slot
    // (`rebase_linear_sequence`, pinned in `executor_geometry_test`), so a
    // copy-on-write child can continue itself and forking is usable.
    //
    // Generation therefore FOLDS normally. The buffer-and-discard scheme this
    // replaces was CUDA-only in a way nothing checked: Metal validates
    // `rs_fold_lens` in `batch/compose.cpp` and then never reads it again, and
    // refuses the buffer READ path outright, so every fire folded regardless
    // and `discard_buffered` silently left the guest believing a state stood at
    // `render_len` while the device had folded the whole scratch span. That is
    // the bug this replaces; see `integrations/opencode/finding-hybrid-session-
    // resume.md`.
    let ws = &state.ws;
    let rs = &state.rs[..];
    let pool_ids: Vec<u32> = (0..pool_pages).collect();

    // The retention snapshot, taken before a single scratch token is written.
    // Empty on an attention-only model, where KV alone rewinds for free.
    let retain_rs: Vec<RsWorkingSet> = state
        .rs
        .iter()
        .map(|r| r.fork(&pipe))
        .collect::<Result<_>>()
        .context("rs.fork (retention snapshot)")?;
    let recurrent = !state.rs.is_empty();

    // ── 3. Prefill the cue and sample the turn's first token ─────────────
    let g0 = prefill_span::<W>(
        &pipe,
        &state,
        render_len,
        cue,
        pool_pages,
        temperature,
        top_p,
        &mut on_prefill_chunk,
    )
    .await?;
    // The turn's first token exists here, so this is the guest's own TTFT. The
    // client's time-to-first-content minus this is everything outside the
    // guest: shim, gateway, WebSocket, stderr capture.
    let t_first = t0.elapsed();

    let mut accepted = 0usize;
    let mut gen_error: Option<String> = None;
    let g0u = g0 as u32;
    let mut done = stop_ids.contains(&g0u);
    if !done {
        accepted += 1;
        if on_token(g0u).is_break() || cfg.max_tokens <= 1 {
            done = true;
        }
    }

    // ── 4. Sequential decode (1-wide, host-driven) ───────────────────────
    // One fire per token, every port carrying a host-known value.
    //
    // The Strategy A loop instead carries the geometry on DEVICE — the epilogue
    // re-puts `tok_in`/`pos`/`klen`/… for the next fire — so `run_ahead` can
    // keep a window in flight without the host in the path. Two things make
    // that the wrong shape here:
    //
    // - sequential decode takes every token to the host anyway (it must: a fold
    //   advances on every fire that executes, so nothing may be speculated), so
    //   the round-trip the device-carried form avoids is already being paid;
    // - a buffered recurrent fire is planned host-side, and the strict geometry
    //   gate rejects a device-fed token channel outright:
    //   `EmbedTokens is not host-derivable: channel 0 has no host-known value`
    //   (`pipeline/fire/geometry.rs`). The loop-carried form CANNOT satisfy it.
    //
    // Building a fire per token costs microseconds of host work against ~11 ms
    // of GPU per token on the 35B, and it keeps one decode implementation
    // rather than one per pass kind.
    let mut next_tok = g0;
    // Token history for the drafter: the turn's own rendered tokens plus what
    // it has generated. Prompt-lookup drafts by finding where the recent output
    // occurred earlier IN THIS TURN, so the haystack has to include both.
    let mut seen: Vec<u32> = Vec::with_capacity(delta.len() + cue.len() + 64);
    seen.extend_from_slice(delta);
    seen.extend_from_slice(cue);
    seen.push(g0u);
    // Speculation is gated on there being NO recurrent fold, and that is the
    // same condition that makes it safe. A verify fire embeds rows it may
    // reject; for KV those are masked by the next fire's `kv_len`, but a fold
    // advances on every row that executes and cannot be rewound — the exact
    // hazard this module's header describes for `run_ahead`. Pure attention
    // (Qwen3-Coder-30B) has no fold, so a rejected row costs nothing but the
    // work already done.
    //
    // Greedy only, and that is a correctness bound rather than a shortcut.
    // Accepting a draft because it matches the model's ARGMAX is equivalent to
    // the per-token loop only when the per-token loop would itself have taken
    // the argmax. Under nucleus sampling it silently changes the output
    // distribution — a bug that would present as a speedup. Exact speculative
    // sampling is possible here (the rng is deterministic: step `i` uses
    // counter `i`, so each row could be sampled with the counter the sequential
    // loop would have used), but it needs the sampler to work per-row over a
    // `[rows, vocab]` matrix, which is unverified. Not assumed away — gated.
    // `SPEC_OFF=1 cargo build` disables drafting, which is the CONTROL arm:
    // built from this same source so an A/B cannot compare two builds that
    // differ in more than speculation.
    let speculate = !recurrent
        && temperature == 0.0
        && crate::draft::DRAFT_K > 0
        && option_env!("SPEC_OFF").is_none();
    let mut spec = crate::draft::Accounting::default();
    if !done {
        let budget = cfg.max_tokens.saturating_sub(1); // g0 already delivered
        let mut i = 0usize;
        while i < budget {
            // Absolute position this fire writes: g0 lands at `n`, and each
            // later fire one further on.
            let p = n + i as u32;
            // Rows: the confirmed token, then the draft. `d` is empty whenever
            // nothing repeats, and the fire degrades to exactly the 1-row form
            // it replaces.
            let d: Vec<u32> = if speculate {
                let room = budget - i - 1;
                crate::draft::draft(&seen, &seen, crate::draft::DRAFT_K.min(room))
            } else {
                Vec::new()
            };
            let rows = 1 + d.len() as u32;
            let mut row_toks: Vec<i32> = Vec::with_capacity(rows as usize);
            row_toks.push(next_tok);
            row_toks.extend(d.iter().map(|&t| t as i32));
            let toks = Channel::from(row_toks).named("toks_d");
            let embed_indptr = Channel::from([0u32, rows]).named("embed_indptr_d");
            let positions = Channel::from_iter(p..p + rows).named("positions_d");
            let w_slot =
                Channel::from_iter((p..p + rows).map(|q| q / page_t)).named("w_slot_d");
            let w_off =
                Channel::from_iter((p..p + rows).map(|q| q % page_t)).named("w_off_d");
            let klen = Channel::from([p + rows]).named("klen_d");
            let pages = Channel::from(pool_ids.clone()).named("pages_d");
            let page_indptr =
                Channel::from([0u32, (p + rows).div_ceil(page_t)]).named("pidx_d");
            // ROW indices within this fire, not absolute positions. Verified
            // against `pipeline/instance.rs`, where a single-token fire asserts
            // `sampling_indices == vec![0]`. They coincide in `specverify`
            // only because its fire starts at position 0.
            let readout = Channel::from_iter(0..rows).named("readout_d");
            let out = Channel::new([rows], dtype::i32).named("out_d");
            // Vary the counter per step, or every token would be sampled from
            // the same Gumbel draw.
            let rng = Channel::from([0x9e37_u32, i as u32]).named("rng_d");

            let fwd: Pass<W> = Pass::new();
            if let Err(e) = fwd.embed(&toks, &embed_indptr) {
                gen_error = Some(e);
                break;
            }
            if let Err(e) = fwd.readout(&readout) {
                gen_error = Some(e);
                break;
            }
            if let Err(e) = fwd.bind_state(
                ws,
                KvGeometry {
                    readable_pages: ..,
                    writable_pages: (p / page_t)..,
                    kv_len: &klen,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
                rs,
                // Fold. The retention snapshot was forked before this loop, so
                // the state these fires advance is already a private copy.
                None,
            ) {
                gen_error = Some(e);
                break;
            }
            let sink = out.clone();
            fwd.epilogue(move || {
                let r = rng.take();
                // One value per read-out row. `sample_token` reduces the
                // `[rows, vocab]` logits to `[rows]`; at temperature 0 that is
                // a per-row argmax, which is what the accept test compares
                // against. The verdict is computed HOST-side rather than with
                // `specverify`'s in-graph cumprod: this loop already takes
                // every fire's result to the host, so the round trip the
                // in-graph form avoids is paid either way, and Rust is easier
                // to test than a tensor DAG.
                let tok = sample_token(&r, temperature, top_p);
                let r_next = &r + iota(2);
                sink.put(&tok);
                rng.put(&r_next);
            });
            if let Err(e) = submit_frame(&pipe, &[Some(&fwd)]) {
                gen_error = Some(e);
                break;
            }
            let t = match out.take_host::<Vec<i32>>().await {
                Ok(t) => t,
                Err(e) => {
                    gen_error = Some(e);
                    break;
                }
            };
            if t.len() != rows as usize {
                gen_error = Some(format!(
                    "verify fire returned {} row(s), expected {rows}",
                    t.len()
                ));
                break;
            }
            // ── ACCEPT ────────────────────────────────────────────────────
            // Row j's logits predict position p+j+1. `d[j]` is this turn's
            // guess for that token, so row j confirms draft token j. The first
            // mismatch ends the run and everything after it is discarded — a
            // later coincidental agreement must NOT leak through, which is the
            // property `specverify`'s cross-row cumprod enforces in-graph and
            // this `take_while` enforces here.
            let take = d
                .iter()
                .zip(t.iter())
                .take_while(|(want, got)| **want as i32 == **got)
                .count();
            // Row `take` was conditioned only on accepted tokens, so its own
            // sample is the model's genuine next token whatever happened after.
            let fresh = t[take] as u32;
            if speculate {
                spec.record(d.len(), take);
            }
            // Every row executed, so every row is in KV. Only `take + 1` of
            // them are KEPT; the rest are masked by the next fire's `kv_len`,
            // which is why this is sound without a fold to rewind.
            let mut stop = false;
            let mut emitted = 0usize;
            for tok in d[..take].iter().copied().chain(std::iter::once(fresh)) {
                // A stop token is sampled but never embedded: truncated at,
                // never written.
                if stop_ids.contains(&tok) {
                    stop = true;
                    break;
                }
                seen.push(tok);
                emitted += 1;
                accepted += 1;
                if on_token(tok).is_break() || accepted >= cfg.max_tokens {
                    stop = true;
                    break;
                }
            }
            next_tok = fresh as i32;
            // Advance by what was KEPT, not by what was fired.
            i += emitted.max(1);
            if stop {
                break;
            }
        }
    }
    {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let line = format!(
            "[opencode-session] gen_ms prefill={:.1} first_token={:.1} decode_end={:.1} \
             (cumulative) delta={} cue={} gen={}\n",
            ms(t_prefill),
            ms(t_first),
            ms(t0.elapsed()),
            n_delta,
            cue.len(),
            accepted
        );
        eprint!("{line}");
    }
    if spec.proposed > 0 {
        // ONE formatted string, ONE write. `eprintln!` hands the host one
        // format ARGUMENT at a time and each becomes its own log record, which
        // is why the first reading of this line came back as
        // "speculation:45% accepted (36/80 drafted over48 fires)" with the
        // spaces gone and the fields interleavable with another turn's.
        let line = format!(
            "[opencode-session] speculation {:.0}% accepted ({}/{} drafted, {} verify fires, {} fires saved)\n",
            100.0 * spec.acceptance(),
            spec.accepted,
            spec.proposed,
            spec.verify_fires,
            spec.fires_saved()
        );
        eprint!("{line}");
    }

    // Close releases the scheduler wait-set and rejects further submissions.
    pipe.close();

    // Retain the SNAPSHOT, not the state generation just advanced. KV needs
    // nothing done to it — the scratch cells past `render_len` are simply never
    // covered by a later `kv_len`. The recurrent half cannot be rewound at all,
    // which is why the fork above exists: `retain_rs` still holds the fold as
    // it stood at `render_len`, and the state the decode loop folded its way to
    // is dropped here with the turn.
    let state = SessionState {
        ws: state.ws,
        rs: retain_rs,
    };

    let hit_max = accepted >= cfg.max_tokens;
    Ok(Generation {
        state,
        total_len: render_len,
        accepted,
        hit_max,
        gen_error,
    })
}

/// Prefill `tokens` at absolute position `at` on `st`, chunked, returning the
/// token sampled from the LAST chunk's read-out row.
///
/// The caller ignores that token for rendered history (a mid-prompt read-out
/// means nothing) and uses it as the turn's first generated token when the span
/// is the generation cue. Every chunk samples regardless: an epilogue `put` has
/// to be drained or the channel fills.
/// Prefill spans chosen so no fire lands in the driver's slow row-count class.
///
/// MEASURED, in `runtime/engine/tests/inferlets/decode-rows-probe`: a fire of
/// `r` rows pays a flat ~560 ms penalty when `r % 8` is 1..=6, and does not when
/// it is 0 or 7. At a 7420-token context:
///
/// ```text
///   rows   186    187    188    189    190    191    192
///   ms    1142   1145   1149   1147   1147    583    583
/// ```
///
/// 191 is prime and fast while 188 is a multiple of 4 and slow, so it is not
/// alignment in any ordinary sense; and the same widths are fast at ctx 7420 as
/// at 7424, so it is the ROW COUNT, not `kv_len`. Below ~16 rows the penalty
/// disappears entirely (12 rows: 99 ms; 20 rows: 554 ms).
///
/// That is why this exists instead of `prefill_chunks`. A resumed opencode turn
/// prefills its 189-token delta as one 189-row fire and pays the penalty on
/// EVERY turn — 1147 ms against the 583 ms the same work costs at 192 rows,
/// which was most of the second per turn that neither prefill nor decode could
/// account for. Splitting 189 into 184 + 5 puts both fires in the fast class.
///
/// The root cause is in the driver and is not fixed here; this is the guest
/// steering around it. Two consequences worth knowing: it costs one extra fire
/// per span (the remainder), which is cheap only because a sub-16-row fire is
/// exempt; and it moves chunk boundaries, so token-level output can differ from
/// the old chunking exactly the way any re-chunking can (`apc-graft-probe`
/// measures that effect as benign argmax flips near ties).
fn aligned_prefill_chunks(n: u32, embed_cap: u32) -> Vec<(u32, u32)> {
    /// Fires at or above this width obey the `% 8` rule; below it they do not.
    const EXEMPT_BELOW: u32 = 16;
    const ALIGN: u32 = 8;
    // Keep the driver's structural per-launch ceiling, rounded down so a full
    // chunk is itself in the fast class.
    // `embed_cap` is passed rather than read from `max_embed_length()` so this
    // stays a pure function: the host binding is unavailable under `cargo test`,
    // and a rule this easy to get subtly wrong needs unit tests more than it
    // needs one fewer argument.
    let cap = (embed_cap.max(ALIGN) / ALIGN) * ALIGN;
    let mut out = Vec::new();
    let mut at = 0;
    while at < n {
        let left = n - at;
        let take = if left > cap {
            cap
        } else if left < EXEMPT_BELOW || left % ALIGN == 0 {
            left
        } else {
            left - (left % ALIGN)
        };
        out.push((at, at + take));
        at += take;
    }
    out
}

async fn prefill_span<W>(
    pipe: &Pipeline,
    st: &SessionState,
    at: u32,
    tokens: &[u32],
    pool_pages: u32,
    temperature: f32,
    top_p: f32,
    on_chunk: &mut impl FnMut(),
) -> Result<i32>
where
    W: PassWit,
    Pass<W>: BindState,
{
    if tokens.is_empty() {
        return Ok(0);
    }
    let page_t = kv_page_size();
    let pool_ids: Vec<u32> = (0..pool_pages).collect();
    let chunks = aligned_prefill_chunks(tokens.len() as u32, max_embed_length() as u32);
    let last_ci = chunks.len() - 1;
    let mut last = 0i32;

    for (ci, &(s, e)) in chunks.iter().enumerate() {
        let (abs_s, abs_e) = (at + s, at + e);
        let toks_v: Vec<i32> = tokens[s as usize..e as usize]
            .iter()
            .map(|&t| t as i32)
            .collect();
        let toks = Channel::from(toks_v).named("toks_p");
        let embed_indptr = Channel::from([0u32, e - s]).named("embed_indptr_p");
        let positions = Channel::from_iter(abs_s..abs_e).named("positions_p");
        let w_slot_v: Vec<u32> = (abs_s..abs_e).map(|p| p / page_t).collect();
        let w_off_v: Vec<u32> = (abs_s..abs_e).map(|p| p % page_t).collect();
        let w_slot = Channel::from(w_slot_v).named("w_slot_p");
        let w_off = Channel::from(w_off_v).named("w_off_p");
        let klen = Channel::from([abs_e]).named("klen_p");
        let pages = Channel::from(pool_ids.clone()).named("pages_p");
        // The page CSR is the driver's source of truth for kv_len on the wire:
        // count tracks the valid length exactly, never the pool size. No
        // AttnMask port — causal is what the CSR already says.
        let page_indptr = Channel::from([0u32, abs_e.div_ceil(page_t)]).named("pidx_p");
        let outc = Channel::new([1], dtype::i32).named("g0");
        // Owned here so the port outlives the bind and reaches `submit`.

        let fwd: Pass<W> = Pass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.bind_state(
            &st.ws,
            KvGeometry {
                readable_pages: ..,
                // Start writable at this span's own boundary. Declaring the
                // whole range would CoW-copy the shared prefix for nothing —
                // precisely the cost this strategy exists to avoid. (Strategy A
                // declares `..`; it has no shared prefix to spoil.)
                writable_pages: (abs_s / page_t)..,
                kv_len: &klen,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
            &st.rs[..],
            None,
        )?;
        let rng_p = Channel::from([0x51ed_u32, 0]).named("rng_p");
        fwd.epilogue(move || {
            let r = rng_p.take();
            let tok = sample_token(&r, temperature, top_p);
            let r_next = &r + iota(2);
            outc.put(&tok);
            rng_p.put(&r_next);
        });
        fwd.submit(pipe)
            .with_context(|| format!("prefill submit @{abs_s}"))?;
        let v = outc
            .take_host::<i32>()
            .await
            .with_context(|| format!("prefill take @{abs_s}"))?;
        if ci == last_ci {
            last = v;
        }
        on_chunk();
    }
    Ok(last)
}

#[cfg(test)]
mod aligned_chunk_tests {
    use super::aligned_prefill_chunks;

    /// The property the driver cares about: no fire in the penalised class.
    fn fast(spans: &[(u32, u32)], cap: u32) -> bool {
        spans.iter().all(|&(s, e)| {
            let r = e - s;
            r <= cap && (r < 16 || r % 8 == 0)
        })
    }

    #[test]
    fn covers_the_span_exactly_and_in_order() {
        for n in [1u32, 5, 16, 189, 1000, 2048, 7211, 9999] {
            let spans = aligned_prefill_chunks(n, 2048);
            assert_eq!(spans.first().map(|s| s.0), Some(0), "n={n}");
            assert_eq!(spans.last().map(|s| s.1), Some(n), "n={n}");
            for w in spans.windows(2) {
                assert_eq!(w[0].1, w[1].0, "gap or overlap at n={n}");
            }
            assert!(spans.iter().all(|&(s, e)| e > s), "empty span at n={n}");
        }
    }

    #[test]
    fn no_span_lands_in_the_slow_row_class() {
        // The whole point. 189 is the width the opencode replay actually fires
        // and the one that measured 2x; a regression here is silent and costs
        // half a second per turn, so it is pinned by value.
        assert_eq!(aligned_prefill_chunks(189, 2048), vec![(0, 184), (184, 189)]);
        for n in 1..600u32 {
            assert!(fast(&aligned_prefill_chunks(n, 2048), 2048), "n={n}");
        }
    }

    #[test]
    fn a_short_span_stays_one_fire() {
        // Below the threshold the penalty does not apply, so splitting would
        // add a fire and buy nothing.
        assert_eq!(aligned_prefill_chunks(5, 2048), vec![(0, 5)]);
        assert_eq!(aligned_prefill_chunks(15, 2048), vec![(0, 15)]);
    }
}
