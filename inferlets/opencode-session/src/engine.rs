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
/// There is no fork variant, and that is a finding rather than a
/// simplification. Copy-on-write branching of a session's state does not work
/// on this stack today, in two independent ways:
///
/// - **KV.** The scheduler builds every pre-launch CoW plan with
///   `PIE_MEMORY_DOMAIN_CUDA_DEVICE` hardcoded, and Metal refuses a copy whose
///   domain is not `METAL_SHARED`. Fixed in this branch at the backend boundary
///   (`runtime/engine/src/driver/backend.rs::copy_kv`), so `WorkingSet::fork`
///   now works on Metal — but on its own that was not enough.
/// - **The fold.** `RsWorkingSet::fork` mints a new sequence id and the driver
///   rejects the child as a continuation of its parent:
///   `recurrent slot 1 holds sequence 2^63, this fire is sequence 2^63+1`.
///   Nothing in the guest can work around that.
///
/// So a turn extends the state it resumed, in place, on both pass kinds. That
/// is sound because nothing generated is retained: KV past `render_len` is
/// scratch the next turn overwrites, and the recurrent scratch is buffered
/// rather than folded and then discarded (see the module docs).
///
/// The cost is branching: two turns cannot extend one parent concurrently.
/// Turns are sequential and retention drops the parent it extended, so nothing
/// does that today — but B-3 (forking the working set for subagents) needs real
/// CoW on both halves and is blocked on the recurrent side.
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
    let pool_pages_want = (n + cfg.max_tokens as u32 + 2).div_ceil(page_t);
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
        false,
        &mut on_prefill_chunk,
    )
    .await?;

    // ── 2. Everything from here is SCRATCH ───────────────────────────────
    // The cue and the decoded tokens must not end up in the retained state. In
    // KV that is free — they are written past `render_len`, into cells the next
    // turn overwrites, and no later fire's `kv_len` ever covers them.
    //
    // A fold has no `kv_len` and cannot be rewound, so on a recurrent-state
    // model the same fires would corrupt the state they are supposed to leave
    // alone. The obvious fix is to generate on a fork; it does not work. The
    // KV half is refused on Metal (see `Resume`), and the recurrent half is
    // refused everywhere:
    //
    // ```text
    // [pie-driver-metal] instance 2 launch failed: paged continuation:
    //   recurrent slot 1 holds sequence 9223372036854775808,
    //   this fire is sequence 9223372036854775809
    // ```
    //
    // `RsWorkingSet::fork` mints a NEW sequence id, and the driver rejects it
    // as a continuation of the folded state it was forked from. (The qwen-code
    // session hit this too, from the other direction — sealing on a fresh
    // pipeline.)
    //
    // So generation does not fold at all: it BUFFERS. `fold_len: Some(0)` puts
    // each fire's tokens in the recurrent buffer instead of the fold, where
    // attention still reads them for this turn — `runtime/engine/tests/
    // inferlets/gdn-foldcommit` pins buffer-then-fold against fold-directly —
    // and `discard_buffered` then abandons them for free. That is exactly the
    // REJECT half of fold-commit, which is what a speculative tail is, and a
    // generation we do not intend to retain is one.
    //
    // The happy consequence: both pass kinds now extend in place and NOTHING
    // forks. `Resume::Fork` is gone.
    let ws = &state.ws;
    let rs = &state.rs[..];
    let pool_ids: Vec<u32> = (0..pool_pages).collect();

    // Buffer capacity for the whole scratch span. Purely logical until a fire
    // leaves tokens in it.
    let scratch_tokens = cue.len() as u32 + cfg.max_tokens as u32 + 1;
    if let Some(r) = state.rs.first() {
        let per_page = r.buffer_page_size().max(1);
        r.alloc_buffer(scratch_tokens.div_ceil(per_page))
            .context("rs.alloc_buffer")?;
    }
    let buffering = !state.rs.is_empty();

    // ── 3. Prefill the cue and sample the turn's first token ─────────────
    let g0 = prefill_span::<W>(
        &pipe,
        &state,
        render_len,
        cue,
        pool_pages,
        temperature,
        top_p,
        buffering,
        &mut on_prefill_chunk,
    )
    .await?;

    let mut accepted = 0usize;
    let mut gen_error: Option<String> = None;
    // Fires whose tokens landed in the recurrent buffer during decode. The cue
    // span is counted separately; together they are exactly what must be
    // abandoned before this state is retained.
    let mut decode_fires = 0u32;
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
    if !done {
        let budget = cfg.max_tokens.saturating_sub(1); // g0 already delivered
        for i in 0..budget {
            // Absolute position this fire writes: g0 lands at `n`, and each
            // later fire one further on.
            let p = n + i as u32;
            let toks = Channel::from([next_tok]).named("toks_d");
            let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr_d");
            let positions = Channel::from([p]).named("positions_d");
            let w_slot = Channel::from([p / page_t]).named("w_slot_d");
            let w_off = Channel::from([p % page_t]).named("w_off_d");
            let klen = Channel::from([p + 1]).named("klen_d");
            let pages = Channel::from(pool_ids.clone()).named("pages_d");
            let page_indptr =
                Channel::from([0u32, (p + 1).div_ceil(page_t)]).named("pidx_d");
            let out = Channel::new([1], dtype::i32).named("out_d");
            // Vary the counter per step, or every token would be sampled from
            // the same Gumbel draw.
            let rng = Channel::from([0x9e37_u32, i as u32]).named("rng_d");
            let fold0 = Channel::from([0u32]).named("fold0_d");

            let fwd: Pass<W> = Pass::new();
            if let Err(e) = fwd.embed(&toks, &embed_indptr) {
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
                buffering.then_some(&fold0),
            ) {
                gen_error = Some(e);
                break;
            }
            let sink = out.clone();
            fwd.epilogue(move || {
                let r = rng.take();
                let tok = sample_token(&r, temperature, top_p);
                let r_next = &r + iota(2);
                sink.put(&tok);
                rng.put(&r_next);
            });
            if let Err(e) = submit_frame(&pipe, &[Some(&fwd)]) {
                gen_error = Some(e);
                break;
            }
            let t = match out.take_host::<i32>().await {
                Ok(t) => t,
                Err(e) => {
                    gen_error = Some(e);
                    break;
                }
            };
            // The fire executed, so its embedded token is in KV and in the
            // recurrent buffer whether or not the token it SAMPLED is
            // surfaced. Count before the stop test, or `discard_buffered`
            // leaves one token behind and the next turn folds it.
            decode_fires += 1;
            let token = t as u32;
            next_tok = t;
            // A stop token is sampled but never embedded: it is truncated at,
            // never written. That is why `decode_fires` is not `accepted`.
            if stop_ids.contains(&token) {
                break;
            }
            accepted += 1;
            if on_token(token).is_break() || accepted >= cfg.max_tokens {
                break;
            }
        }
    }

    // Close releases the scheduler wait-set and rejects further submissions.
    pipe.close();

    // Abandon the scratch span. KV needs nothing — those cells are simply never
    // covered by a later `kv_len`. The recurrent buffer does: it still holds the
    // cue and every decoded token, and the next turn's delta prefill folds over
    // `[buffer | its own tokens]`, so anything left here would be folded into
    // the retained state as if the client had sent it.
    if let Some(r) = state.rs.first() {
        let buffered = cue.len() as u32 + decode_fires;
        if buffered > 0 {
            if let Err(e) = r.discard_buffered(buffered) {
                // The state is no longer trustworthy: its buffer holds tokens
                // the conversation does not contain. Fail the turn so the
                // caller drops it rather than retaining a poisoned prefix.
                gen_error = Some(format!("discard_buffered({buffered}): {e}"));
            }
        }
    }

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
async fn prefill_span<W>(
    pipe: &Pipeline,
    st: &SessionState,
    at: u32,
    tokens: &[u32],
    pool_pages: u32,
    temperature: f32,
    top_p: f32,
    buffered: bool,
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
    let chunks = prefill_chunks(tokens.len() as u32, None);
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
        let fold0 = Channel::from([0u32]).named("fold0_p");

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
            buffered.then_some(&fold0),
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
