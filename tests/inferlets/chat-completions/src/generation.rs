//! PTIR prefill + decode for one chat turn, modeled on
//! `tests/inferlets/chat-completion/src/lib.rs` (the upstream sample) and
//! `runtime/engine/tests/inferlets/prefix-cache-e2e/src/lib.rs` (the
//! suffix-only-resume discipline).
//!
//! Resume semantics: the caller passes the retained parent `WorkingSet` and
//! its valid token count. The turn generates on an O(1) copy-on-write fork
//! (the parent stays intact for qwen-code turn-level retries), prefills only
//! the suffix rows — `writable_pages` starts at the cached boundary so the
//! shared prefix is never CoW-copied — and declares `kv_len` over the full
//! context. Speculative run-ahead fires past the stop token leave garbage KV
//! beyond the accepted length; it is never referenced because every later
//! fire's `kv_len`/page-CSR only ever cover valid tokens, and `seal()`
//! overwrites the head of it with the turn suffix.

use core::ops::RangeBounds;
use inferlet::model;
use inferlet::ptir::attention::prelude::*;
use inferlet::ptir::{KvBinding, Pass, PassWit, RsGeometry, RsWorkingSet};

/// The WIT pass resources behind `ptir::{attention,hybrid}::ForwardPass`.
/// Named directly so the generation body can be generic over the pass kind
/// instead of textually duplicated per kind.
use inferlet::pie::inferlet::forward::ForwardPass as WitAttention;
use inferlet::pie::inferlet::forward_hybrid::ForwardPass as WitHybrid;

/// The one thing that differs between the three forward interfaces, for THIS
/// algorithm. `pie:inferlet` exposes `forward-attention`, `forward-hybrid` and
/// `forward-recurrent` as three *unrelated* wit-bindgen types whose
/// `attention` signatures deliberately diverge — an attention-only algorithm
/// must not be able to name a folded recurrent state. Saying what they have in
/// common here is the guest's job.
///
/// Shape taken from `inferlets/chat-completions/src/engine.rs` on
/// `liu/opencode-integration` (commit 04db752f4), which solved this first and
/// has it verified live on Qwen3.6-35B-A3B.
trait BindState {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> core::result::Result<(), String>
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
    ) -> core::result::Result<(), String>
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
    ) -> core::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        // Serving folds every fire and never buffers. The SAME rs working set
        // is bound by the prefill chunks, the decode fires and the seal, so
        // each continues the previous fold rather than restarting it.
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

/// The retained per-session device state. On a hybrid model the folded
/// recurrent state is part of the conversation prefix just as much as the KV
/// pages are: resuming KV at token N while the fold still stands at 0 would
/// silently serve a different context. So the two are retained and forked
/// together, or not at all.
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

    /// Copy-on-write child of both halves, ordered on the same pipeline.
    /// `RsWorkingSet::fork` shares the current folded state and buffered
    /// suffix, which is what makes append-only turn growth resumable.
    pub fn fork(&self, on: &Pipeline) -> Result<SessionState> {
        let mut rs = Vec::with_capacity(self.rs.len());
        for r in &self.rs {
            rs.push(r.fork(on).context("rs.fork")?);
        }
        Ok(SessionState {
            ws: self.ws.fork(on).context("ws.fork")?,
            rs,
        })
    }
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GenParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

pub struct Generation {
    /// KV pages plus, on a recurrent-state model, the folded state that
    /// belongs to the same prefix. Retained as a unit.
    pub state: SessionState,
    /// Valid tokens in KV: prompt + accepted (stop token excluded — it is
    /// truncated at, never written).
    pub total_len: u32,
    /// Accepted generated tokens, in order.
    pub generated: Vec<u32>,
    pub hit_max: bool,
    /// Set when generation died mid-turn — the caller degrades the turn to
    /// `finish_reason:"length"` (audit §1 req. 8: never surface a
    /// context-length error to qwen-code) and skips session retention.
    pub gen_error: Option<String>,
}

/// In-graph sampler over the read-out row logits. Zero temperature is exact
/// greedy decoding; positive temperatures use nucleus sampling. `r` is the
/// taken `[2]` u32 rng state (`[key, ctr]`).
fn sample_token(r: &Tensor, temperature: f32, top_p: f32) -> Tensor {
    let logits = intrinsics::logits(); // [1, vocab] f32
    if temperature <= 0.0 {
        return reduce_argmax(&logits);
    }
    let scaled = &logits / temperature.max(1e-4);
    nucleus_sample(&scaled, top_p, r)
}

/// Decode headroom for the seal fire (`<|im_end|>\n` is 2 tokens; margin is
/// cheap because reservation is purely logical).
const SEAL_MARGIN: u32 = 8;

/// Run one turn: fork-or-fresh working set, chunked prefill of `prefill`
/// (the rendered suffix on resume, the full history otherwise), then the
/// device-carried decode loop.
///
/// `on_prefill_chunk` fires after each committed prefill chunk (keepalive
/// point). `on_token` receives each accepted token; returning `false` stops
/// generation (client stop-strings).
pub async fn generate(
    resume: Option<(&SessionState, u32)>,
    prefill: &[u32],
    params: &GenParams,
    stop_ids: &[u32],
    on_prefill_chunk: impl FnMut(),
    on_token: impl FnMut(u32) -> bool,
) -> Result<Generation> {
    // One body, selected by the model's forward kind. `run_ahead` and every
    // `Pass` method except `attention` are already generic over the wit type,
    // so the algorithm is written once and cannot drift between kinds.
    match model::pass_kind() {
        model::ForwardKind::Attention => {
            generate_for::<WitAttention>(
                resume, prefill, params, stop_ids, on_prefill_chunk, on_token,
            )
            .await
        }
        model::ForwardKind::Hybrid => {
            generate_for::<WitHybrid>(resume, prefill, params, stop_ids, on_prefill_chunk, on_token)
                .await
        }
        // No registered model reports recurrent-only, and this loop's paged
        // prompt geometry has nothing to bind on a pass with no KV at all —
        // so it degrades the turn rather than pretending.
        model::ForwardKind::Recurrent => Err(
            "this model is recurrent-only; the serving inferlet has no KV-free path".to_string(),
        ),
    }
}

async fn generate_for<W>(
    resume: Option<(&SessionState, u32)>,
    prefill: &[u32],
    params: &GenParams,
    stop_ids: &[u32],
    mut on_prefill_chunk: impl FnMut(),
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<Generation>
where
    W: PassWit,
    Pass<W>: BindState,
{
    let page_t = kv_page_size();
    let pipe = Pipeline::new();

    let (state, n0) = match resume {
        Some((parent, cached)) => (parent.fork(&pipe)?, cached),
        None => (SessionState::new(), 0),
    };
    let ws = &state.ws;
    let rs = &state.rs[..];

    let n_suffix = prefill.len() as u32;
    if n_suffix == 0 {
        return Err("empty prefill".to_string());
    }
    let n = n0 + n_suffix;

    // Logical page pool: full context + decode budget + seal headroom,
    // page-rounded. Reserve only the shortfall past what the fork carries.
    let pool_pages = (n + params.max_tokens as u32 + SEAL_MARGIN + 2).div_ceil(page_t);
    let have = ws.page_len();
    if pool_pages > have {
        ws.reserve(pool_pages - have).context("ws.reserve")?;
    }
    let pool_pages = pool_pages.max(have);
    // WorkingSet-relative page indexes ARE the addressing scheme — the guest
    // never holds physical ids.
    let pool_ids: Vec<u32> = (0..pool_pages).collect();

    let temperature = params.temperature;
    let top_p = params.top_p;

    // ── 1. Chunked prefill of the suffix rows ────────────────────────────
    // Each chunk is one fire: positions/write descriptors cover the chunk,
    // kv_len and the page CSR cover everything valid so far. The last chunk
    // samples g0 from its read-out row.
    let chunks = prefill_chunks(n_suffix, None);
    let last_ci = chunks.len() - 1;
    let mut g0: i32 = 0;
    for (ci, &(s, e)) in chunks.iter().enumerate() {
        let (abs_s, abs_e) = (n0 + s, n0 + e);
        let toks_v: Vec<i32> = prefill[s as usize..e as usize]
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
        // The page CSR is the source of truth for kv_len on the wire: count
        // tracks the valid length exactly, never the pool size.
        let page_indptr = Channel::from([0u32, abs_e.div_ceil(page_t)]).named("pidx_p");
        let outc = Channel::new([1], dtype::i32).named("g0");

        let fwd: Pass<W> = Pass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.bind_state(
            ws,
            KvGeometry {
                readable_pages: ..,
                // Start at the cached/committed boundary: declaring the
                // whole range would CoW-copy the shared prefix for nothing.
                writable_pages: (abs_s / page_t)..,
                kv_len: &klen,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
            rs,
        )?;
        if ci == last_ci {
            let rng_p = Channel::from([0x51ed_u32, 0]).named("rng_p");
            fwd.epilogue(move || {
                let r = rng_p.take();
                let tok = sample_token(&r, temperature, top_p);
                let r_next = &r + iota(2);
                outc.put(&tok);
                rng_p.put(&r_next);
            });
        } else {
            // Non-final chunks: the read-out row is mid-prompt; sink it.
            fwd.epilogue(move || {
                let tok = reduce_argmax(intrinsics::logits());
                outc.put(&tok);
            });
        }
        fwd.submit(&pipe).context("prefill submit")?;
        let v = outc.take_host::<i32>().await.context("prefill take")?;
        if ci == last_ci {
            g0 = v;
        }
        on_prefill_chunk();
    }

    // ── 2. Host-side accounting for g0 ───────────────────────────────────
    let mut generated: Vec<u32> = Vec::new();
    let mut gen_error: Option<String> = None;
    let g0u = g0 as u32;
    let mut done = stop_ids.contains(&g0u);
    if !done {
        generated.push(g0u);
        if !on_token(g0u) || params.max_tokens <= 1 {
            done = true;
        }
    }

    // ── 3. Device-carried decode loop (1-wide) ───────────────────────────
    if !done {
        let slot_n = pool_ids[(n / page_t) as usize];
        let tok_in = Channel::from([g0]).named("tok_in");
        let pos = Channel::from([n]).named("pos");
        let fill = Channel::from([n + 1]).named("fill");
        let klen = Channel::from([n + 1]).named("klen");
        let w_slot = Channel::from([slot_n]).named("w_slot");
        let w_off = Channel::from([n % page_t]).named("w_off");
        let pages = Channel::from(pool_ids.clone()).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_t)]).named("page_indptr");
        let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
        let out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("out");
        let rng = Channel::from([0x9e37_u32, 0]).named("rng");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");

        let fwd: Pass<W> = Pass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.bind_state(
            ws,
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
            rs,
        )?;
        let pool_pages_u = pool_pages;
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
            let pages_v = reshape(&pids, [pool_pages_u]);
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

        let budget = params.max_tokens.saturating_sub(1); // g0 already emitted
        let max_tokens = params.max_tokens;
        let stop: Vec<u32> = stop_ids.to_vec();
        let result = run_ahead(&pipe, &fwd, budget, async || {
            let t = out.take_host::<Vec<i32>>().await?;
            let token = *t.first().unwrap_or(&0) as u32;
            if stop.contains(&token) {
                return Ok(ControlFlow::Break(()));
            }
            generated.push(token);
            if !on_token(token) || generated.len() >= max_tokens {
                return Ok(ControlFlow::Break(()));
            }
            Ok(ControlFlow::Continue(()))
        })
        .await;
        if let Err(e) = result {
            gen_error = Some(e);
        }
    }
    // Any fire still in flight after an early stop is left untaken; close
    // releases the scheduler wait-set, reclaims them, and rejects further
    // submissions.
    pipe.close();

    let hit_max = generated.len() >= params.max_tokens;
    Ok(Generation {
        total_len: n + generated.len() as u32,
        state,
        generated,
        hit_max,
        gen_error,
    })
}

/// Append `seal_tokens` (the turn suffix `<|im_end|>\n`) to KV at position
/// `at`, committing the assistant turn so the retained working set replays
/// as sealed history. One fire on a fresh pipeline; the host take at the end
/// guarantees the write landed before the working set is stored/forked.
pub async fn seal(state: &SessionState, at: u32, seal_tokens: &[u32]) -> Result<u32> {
    if seal_tokens.is_empty() {
        return Ok(at);
    }
    match model::pass_kind() {
        model::ForwardKind::Attention => seal_for::<WitAttention>(state, at, seal_tokens).await,
        model::ForwardKind::Hybrid => seal_for::<WitHybrid>(state, at, seal_tokens).await,
        model::ForwardKind::Recurrent => Err(
            "this model is recurrent-only; the serving inferlet has no KV-free path".to_string(),
        ),
    }
}

/// The seal binds the SAME recurrent state the turn generated on, so the fold
/// advances over the turn suffix too. If it did not, the retained state would
/// stand at the pre-seal boundary while KV stood at the post-seal one, and the
/// next turn would resume a fold that is short by exactly the suffix.
async fn seal_for<W>(state: &SessionState, at: u32, seal_tokens: &[u32]) -> Result<u32>
where
    W: PassWit,
    Pass<W>: BindState,
{
    let ws = &state.ws;
    let rs = &state.rs[..];
    let m = seal_tokens.len() as u32;
    let page_t = kv_page_size();
    let end = at + m;
    let need = end.div_ceil(page_t);
    let have = ws.page_len();
    if need > have {
        ws.reserve(need - have).context("seal reserve")?;
    }
    let pool = need.max(have);
    let pool_ids: Vec<u32> = (0..pool).collect();

    let toks_v: Vec<i32> = seal_tokens.iter().map(|&t| t as i32).collect();
    let toks = Channel::from(toks_v).named("toks_s");
    let embed_indptr = Channel::from([0u32, m]).named("embed_indptr_s");
    let positions = Channel::from_iter(at..end).named("positions_s");
    let w_slot_v: Vec<u32> = (at..end).map(|p| p / page_t).collect();
    let w_off_v: Vec<u32> = (at..end).map(|p| p % page_t).collect();
    let w_slot = Channel::from(w_slot_v).named("w_slot_s");
    let w_off = Channel::from(w_off_v).named("w_off_s");
    let klen = Channel::from([end]).named("klen_s");
    let pages = Channel::from(pool_ids).named("pages_s");
    let page_indptr = Channel::from([0u32, end.div_ceil(page_t)]).named("pidx_s");
    let sink = Channel::new([1], dtype::i32).named("sink_s");

    let pipe = Pipeline::new();
    let fwd: Pass<W> = Pass::new();
    fwd.embed(&toks, &embed_indptr)?;
    fwd.bind_state(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (at / page_t)..,
            kv_len: &klen,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
        rs,
    )?;
    fwd.epilogue(move || {
        let tok = reduce_argmax(intrinsics::logits());
        sink.put(&tok);
    });
    fwd.submit(&pipe).context("seal submit")?;
    let _ = sink.take_host::<i32>().await.context("seal take")?;
    pipe.close();
    Ok(end)
}
