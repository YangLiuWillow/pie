//! **Rows scaling on a HYBRID model** — what a k-row buffered verify fire
//! costs relative to a 1-row fire, at a real context.
//!
//! `decode-rows-probe` answers this for attention models and REFUSES on a
//! hybrid: it builds its pass through `pie:inferlet/forward`, and the engine
//! rejects that interface on a model whose pass kind is `hybrid`. This probe
//! is the same measurement built through `forward-hybrid`, with the recurrent
//! state bound the only way a repeated timing loop can bind it: every timed
//! fire BUFFERS its rows (`fold_len = 0`) and discards them afterwards, so
//! the fold never moves and every sample fires from the identical state.
//! That shape is not a compromise — it IS the speculative verify fire the
//! measurement exists to price (`finding-metal-buffered-rs.md` established
//! it executes with pinned state parity on Metal).
//!
//! Output feeds `beta(ctx) = (fire(rows) − fire(1)) / ((rows−1) · fire(1))`,
//! the slope `draft::Policy` needs before speculation can be gated on for
//! hybrids. Until that number exists the honest policy is `!recurrent`.
//!
//! Input: `"ctx=7424 rows=1,2,3,5,8"` (both optional; those are the
//! defaults). Rows above 8 are pointless to ask for — the driver reads out at
//! most 8 logits rows, which is the hard ceiling on verify width anyway.
//!
//! Method notes, each learned the hard way elsewhere in this suite:
//!  * WARMUP fires per row count are discarded — each row count is a new
//!    program shape and its first fire pays the ~600 ms compile.
//!  * The report is the MEDIAN; a background compile or page fault in one
//!    sample must not move the number.
//!  * Samples land in the SAME KV cells every time (positions ctx..ctx+r,
//!    overwritten per fire) and the same empty buffer (discarded per fire),
//!    so nothing drifts across samples.

use inferlet::ptir::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};
use std::time::Instant;

/// Discarded fires per row count, for the shape compile and cache warmth.
const WARMUP: usize = 2;
/// Timed fires per row count.
const SAMPLES: usize = 10;
/// The driver's logits-readout ceiling; also the widest verify that exists.
const MAX_ROWS: u32 = 8;

const DEFAULT_CTX: u32 = 7424;
const DEFAULT_ROWS: &[u32] = &[1, 2, 3, 5, 8];

const PROMPT: &str = "The history of computing is a history of abstractions, each one \
    hiding a machine that the previous generation had to program directly. ";

fn parse(input: &str) -> (u32, Vec<u32>) {
    let mut ctx = DEFAULT_CTX;
    let mut rows: Vec<u32> = DEFAULT_ROWS.to_vec();
    for part in input.split_whitespace() {
        if let Some(v) = part.strip_prefix("ctx=") {
            if let Ok(v) = v.parse() {
                ctx = v;
            }
        } else if let Some(v) = part.strip_prefix("rows=") {
            let parsed: Vec<u32> = v.split(',').filter_map(|s| s.parse().ok()).collect();
            if !parsed.is_empty() {
                rows = parsed;
            }
        }
    }
    rows.retain(|&r| r >= 1 && r <= MAX_ROWS);
    (ctx, rows)
}

fn median(mut v: Vec<u64>) -> u64 {
    v.sort_unstable();
    v[v.len() / 2]
}

/// One buffered fire of `rows` rows at `ctx`: embed, read out every row,
/// argmax in the epilogue, take. Returns wall micros from build to take.
async fn timed_fire(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    pipe: &Pipeline,
    toks: &[u32],
    ctx: u32,
    page: u32,
    max_pages: u32,
) -> std::result::Result<u64, String> {
    let rows = toks.len() as u32;
    let start = Instant::now();
    let end = ctx + rows;
    let t = Channel::from(toks.iter().map(|&x| x as i32).collect::<Vec<i32>>()).named("h_toks");
    let embed_indptr = Channel::from([0u32, rows]).named("h_indptr");
    let positions = Channel::from_iter(ctx..end).named("h_pos");
    let w_slot = Channel::from((ctx..end).map(|c| c / page).collect::<Vec<u32>>()).named("h_wslot");
    let w_off = Channel::from((ctx..end).map(|c| c % page).collect::<Vec<u32>>()).named("h_woff");
    let kv_len = Channel::from([end]).named("h_klen");
    let pages = Channel::from((0..max_pages).collect::<Vec<u32>>()).named("h_pages");
    let page_indptr = Channel::from([0u32, end.div_ceil(page)]).named("h_pidx");
    let readout = Channel::from_iter(0..rows).named("h_readout");
    let out = Channel::new([rows], dtype::i32).named("h_out");
    // Fold NOTHING: the rows land in the recurrent buffer and are discarded
    // by the caller, so every sample starts from the same fold.
    let fold_len = Channel::from([0u32]).named("h_fold");

    let fwd = ForwardPass::new();
    fwd.embed(&t, &embed_indptr)?;
    fwd.readout(&readout)?;
    fwd.attention(
        Some(KvBinding {
            working_set: ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: (ctx / page)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        }),
        std::slice::from_ref(rs),
        RsGeometry {
            fold_len: Some(&fold_len),
            buffer: ..,
        },
    )
    .with_context(|| format!("bind rows={rows} ctx={ctx}"))?;
    let sink = out.clone();
    fwd.epilogue(move || sink.put(&reduce_argmax(intrinsics::logits())));
    fwd.submit(pipe)
        .with_context(|| format!("submit rows={rows} ctx={ctx}"))?;
    let got = out
        .take_host::<Vec<i32>>()
        .await
        .with_context(|| format!("take rows={rows} ctx={ctx}"))?;
    if got.len() != rows as usize {
        return Err(format!(
            "rows={rows}: read out {} value(s), expected {rows} — the fire is \
             not carrying one row per read-out index and its time means nothing",
            got.len()
        ));
    }
    Ok(start.elapsed().as_micros() as u64)
}

/// The speculation cadence's COMMIT: re-embed `toks` at the fold boundary
/// with `fold_len = toks.len()`, no readout, empty epilogue (a fold fire
/// returns before the output projection; declaring sample rows is refused,
/// and a pass with no stages at all has no program).
async fn commit_fire(
    ws: &WorkingSet,
    rs: &RsWorkingSet,
    pipe: &Pipeline,
    toks: &[u32],
    at: u32,
    page: u32,
    max_pages: u32,
) -> std::result::Result<(), String> {
    let n = toks.len() as u32;
    let end = at + n;
    let t = Channel::from(toks.iter().map(|&x| x as i32).collect::<Vec<i32>>()).named("c_toks");
    let embed_indptr = Channel::from([0u32, n]).named("c_indptr");
    let positions = Channel::from_iter(at..end).named("c_pos");
    let w_slot = Channel::from((at..end).map(|c| c / page).collect::<Vec<u32>>()).named("c_wslot");
    let w_off = Channel::from((at..end).map(|c| c % page).collect::<Vec<u32>>()).named("c_woff");
    let kv_len = Channel::from([end]).named("c_klen");
    let pages = Channel::from((0..max_pages).collect::<Vec<u32>>()).named("c_pages");
    let page_indptr = Channel::from([0u32, end.div_ceil(page)]).named("c_pidx");
    let fold_len = Channel::from([n]).named("c_fold");

    let fwd = ForwardPass::new();
    fwd.embed(&t, &embed_indptr)?;
    fwd.attention(
        Some(KvBinding {
            working_set: ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: (at / page)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        }),
        std::slice::from_ref(rs),
        RsGeometry {
            fold_len: Some(&fold_len),
            buffer: ..,
        },
    )
    .with_context(|| format!("commit bind @{at}"))?;
    fwd.epilogue(|| {});
    fwd.submit(pipe)
        .with_context(|| format!("commit submit @{at}"))?;
    Ok(())
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    if wit_model::pass_kind() == wit_model::ForwardKind::Attention {
        return Ok("skipped: hybrid-rows-probe needs a recurrent-state model".to_string());
    }
    let (ctx, rows_list) = parse(&input);

    let pipe = Pipeline::new();
    let ws = WorkingSet::new();
    let rs = RsWorkingSet::new();
    let page = kv_page_size();
    // Room for every sample: positions advance by `rows` per timed fire (see
    // the timing loop for why), so the tail grows past ctx by the whole run.
    let max_pages = (ctx + 256).div_ceil(page);
    ws.reserve(max_pages).context("ws.reserve")?;
    // The buffered rows need CAPACITY before any fire lands in the buffer;
    // `buffer: ..` binds what exists, it does not allocate. One grant covers
    // the widest fire, and per-sample `discard_buffered` empties tokens
    // without touching capacity.
    let rs_page = inferlet::model::rs_buffer_page_size().max(1);
    rs.alloc_buffer(MAX_ROWS.div_ceil(rs_page).max(1))
        .context("rs.alloc_buffer")?;

    // Synthetic prompt of exactly `ctx` tokens.
    let base = wit_model::encode(PROMPT);
    if base.is_empty() {
        return Err("prompt encoded to nothing".into());
    }
    let prompt: Vec<u32> = base.iter().cycle().take(ctx as usize).copied().collect();

    // Chunked FOLDING prefill to ctx, the same shape the session engine uses.
    let chunk = (max_embed_length() as u32).max(1);
    let mut at = 0u32;
    while at < ctx {
        let end = (at + chunk).min(ctx);
        let toks = &prompt[at as usize..end as usize];
        let n = end - at;
        let t = Channel::from(toks.iter().map(|&x| x as i32).collect::<Vec<i32>>());
        let embed_indptr = Channel::from([0u32, n]);
        let positions = Channel::from_iter(at..end);
        let w_slot = Channel::from((at..end).map(|c| c / page).collect::<Vec<u32>>());
        let w_off = Channel::from((at..end).map(|c| c % page).collect::<Vec<u32>>());
        let kv_len = Channel::from([end]);
        let pages = Channel::from((0..max_pages).collect::<Vec<u32>>());
        let page_indptr = Channel::from([0u32, end.div_ceil(page)]);
        let out = Channel::new([1], dtype::i32).named("p_out");

        let fwd = ForwardPass::new();
        fwd.embed(&t, &embed_indptr)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: (at / page)..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            }),
            std::slice::from_ref(&rs),
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )
        .with_context(|| format!("prefill bind @{at}"))?;
        let sink = out.clone();
        fwd.epilogue(move || sink.put(&reduce_argmax(intrinsics::logits())));
        fwd.submit(&pipe)
            .with_context(|| format!("prefill submit @{at}"))?;
        out.take_host::<i32>()
            .await
            .with_context(|| format!("prefill take @{at}"))?;
        at = end;
    }

    // Row tokens: any real tokens will do; timing does not depend on content.
    let filler: Vec<u32> = base.iter().cycle().take(MAX_ROWS as usize).copied().collect();

    // ── mode `cadence`: does buffer → commit → next window work at REAL
    // context? gdn-foldcommit proved it at "hello world" length, which runs
    // the ring path; the paged path keeps a per-slot position record that a
    // buffered fire ADVANCES (measured: slot at 7425 after a 1-row fold-0
    // fire at 7424) and `discard_buffered` does not rewind. The speculation
    // cadence's commit fire re-embeds the accepted prefix AT the fold
    // boundary, which that record now disagrees with. This mode measures
    // whether the disagreement is real, one step at a time.
    if input.contains("cadence") {
        let mut report = format!("cadence ctx={ctx}\n");
        let step = |tag: &str, out: std::result::Result<u64, String>| match out {
            Ok(_) => format!("{tag}: OK\n"),
            Err(e) => format!("{tag}: REFUSED — {e}\n"),
        };
        // Window 1: buffered verify of 2 rows at ctx.
        let o = timed_fire(&ws, &rs, &pipe, &filler[..2], ctx, page, max_pages).await;
        let failed = o.is_err();
        report.push_str(&step("window1 buffered(2) @ctx", o));
        if !failed {
            // Commit 1: re-embed one token at ctx with fold_len=1, no logits.
            // A commit fire has NOTHING TO AWAIT (no logits, empty epilogue),
            // so `Ok` here means SUBMITTED, not launched: its refusal, if
            // any, surfaces as the pipeline failure the NEXT step inherits,
            // with the driver's reason in the engine log. Measured at ctx
            // 7424: window2 refuses and the log says `slot 0 is at position
            // 7426, this fire starts at 7424` — 7424 is THIS commit's start,
            // so the commit is what the driver refused.
            let o = commit_fire(&ws, &rs, &pipe, &filler[..1], ctx, page, max_pages).await;
            let failed = o.is_err();
            report.push_str(&step("window1 commit(1) @ctx submitted (launch verdict on next step)", o.map(|_| 0)));
            if !failed {
                let _ = rs.discard_buffered(1);
                // Window 2: buffered verify at the new boundary.
                let o =
                    timed_fire(&ws, &rs, &pipe, &filler[..2], ctx + 1, page, max_pages).await;
                report.push_str(&step("window2 buffered(2) @ctx+1", o));
            }
        }
        pipe.close();
        return Ok(report);
    }

    // ── timing: each sample fires at a FRESH position, `pos += rows`,
    // because the paged position record advances with every buffered fire
    // and cannot be rewound (see `cadence` above). The buffer is discarded
    // per sample so the store never accumulates tokens (an append onto a
    // non-empty buffer is the refused read path); the driver's record then
    // agrees with the next sample's start position and every fire runs the
    // identical shape one step later. The recurrence after a discard reads
    // the fold as if the discarded rows never ran — WRONG as a conversation,
    // identical as a workload, which is all a timing probe is entitled to.
    let mut report = format!("hybrid-rows ctx={ctx} warmup={WARMUP} samples={SAMPLES}\n");
    let mut pos = ctx;
    for &r in &rows_list {
        let toks = &filler[..r as usize];
        let mut times = Vec::with_capacity(SAMPLES);
        for i in 0..(WARMUP + SAMPLES) {
            let us = timed_fire(&ws, &rs, &pipe, toks, pos, page, max_pages)
                .await
                .map_err(|e| format!("rows={r} sample {i} @pos {pos}: {e}"))?;
            rs.discard_buffered(r)
                .map_err(|e| format!("rows={r} sample {i} discard: {e}"))?;
            pos += r;
            if i >= WARMUP {
                times.push(us);
            }
        }
        let med = median(times.clone());
        report.push_str(&format!(
            "rows={r} median_us={med} samples_us={:?}\n",
            times
        ));
    }
    pipe.close();
    Ok(report)
}
