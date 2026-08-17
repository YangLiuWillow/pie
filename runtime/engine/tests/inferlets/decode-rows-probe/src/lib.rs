//! **Does a k-row decode fire cost one KV read, or k?**
//!
//! Speculative decoding in `inferlets/opencode-session` removed 1.85x of the
//! decode fires on a real agent turn and bought 1.04x of wall clock. Those two
//! numbers can only both be true if a fire that verifies k tokens costs roughly
//! k times a fire that decodes one — i.e. if the rows do NOT share the read of
//! the KV cache. That is the whole premise of speculation, so it is worth
//! measuring directly rather than inferring from an end-to-end run where a
//! dozen other things also changed.
//!
//! ## The measurement
//!
//! A decode step's cost splits in two (regression over 44 serving calls,
//! R^2 0.984):
//!
//! ```text
//!   t = FIXED + SLOPE * context
//!       FIXED ~ 7.4 ms   the MoE weight read, once per fire
//!       SLOPE ~ 3.21 ms per 1k tokens of KV
//! ```
//!
//! Both halves are per-FIRE in that model. The question is what happens to each
//! when a fire carries k rows instead of 1, so this probe measures at two
//! contexts and solves the same two-term model per row count:
//!
//! ```text
//!   SLOPE(k) flat in k    -> the k rows share one KV read. Speculation pays,
//!                            and the ceiling is (1 + accepted) tokens per fire.
//!   SLOPE(k) ~ k * SLOPE(1) -> every row re-reads the whole cache. A verified
//!                            draft then costs what decoding those tokens one
//!                            at a time would have cost, and no acceptance rate
//!                            can rescue it.
//! ```
//!
//! Reporting only the ratio at one context would confuse the two: at a short
//! context the fixed term dominates and even a k-times-worse KV read looks
//! nearly free; at a long context the reverse. The slope is the quantity that
//! answers the question, and it needs two points.
//!
//! ## What is held identical
//!
//! Every fire runs against the same working set, the same pool of pages, and a
//! `kv_len` that differs only by the k rows being written. The rows are written
//! at `ctx..ctx+k`, overwriting whatever was prefilled there — deliberately.
//! This probe compares TIMES, and a KV entry costs the same to read whatever
//! value it holds, so corrupting a few tokens of context buys geometry that is
//! exactly a decode step's and costs nothing that is measured.
//!
//! The first fires of each shape are discarded: a new `(rows, context)` pair is
//! a new program, and the driver compiles it on first use. That compile is a
//! real cost of speculation on this engine, so it is reported separately rather
//! than hidden inside the average.

use inferlet::Result;
use inferlet::ptir::attention::prelude::*;
use std::time::Instant;

/// The two contexts. Far enough apart that the slope between them is not noise:
/// at ~3.2 ms per 1k, 14k of separation is ~45 ms of signal against a ~1 ms
/// spread between repeats.
///
/// DEFAULTS ONLY — override per run with the inferlet argument, e.g.
/// `-- short=7424,long=28160,rows=1:5:8`. They are constants no longer because
/// two questions this probe is asked cannot be answered at a fixed 7424:
///
///  * `sdpa_paged_decode_hshare`'s unroll is gated INSIDE the kernel on
///    `(q_pos + 1) >= 8192`, so at 7424 both arms of an unroll A/B run the same
///    code. With SHORT and LONG both 7424 — which is what they were — the probe
///    could not reach that path at any row count, and an A/B on it returned
///    1.003x and read as "the unroll is worth nothing".
///  * the server measures that same switch at 1.13x at 16k and 0.85x at 28k, so
///    the crossover is somewhere between, and finding it needs a context sweep
///    this probe had no way to express.
const SHORT: u32 = 7424;
const LONG: u32 = 7424;

/// The first positional argument, out of the JSON envelope the host delivers.
///
/// `pie run ... -- short=7424,long=28160` arrives as
/// `{"_positional":["short=7424,long=28160"]}`, NOT as the bare string. This
/// matters beyond the parsing: `inferlets/generate` does
/// `input.trim().parse().unwrap_or(DEFAULT_MAX_TOKENS)` on the same envelope,
/// so its argument never parses and it silently generates the default count —
/// which is why asking it for 24 tokens yields 5. Read the envelope.
///
/// Hand-scanned rather than taking a serde_json dependency for one field. Only
/// valid because every value this probe accepts is `[0-9a-z=,:]` — no escapes,
/// no embedded quotes. A value that could contain `"` needs a real parser.
fn positional(input: &str) -> &str {
    const KEY: &str = "\"_positional\":[\"";
    match input.find(KEY) {
        Some(at) => {
            let rest = &input[at + KEY.len()..];
            rest.find('"').map_or(rest, |end| &rest[..end])
        }
        // An envelope with no positional entry means no arguments, which is the
        // defaults. Anything that is not an envelope is taken literally, so the
        // probe stays callable with a bare string.
        None if input.trim_start().starts_with('{') => "",
        None => input.trim(),
    }
}

/// `short=`, `long=` and `rows=` out of the inferlet argument, defaults above.
///
/// Unparseable input FAILS rather than falling back: a typo'd `rows=1:5:8`
/// silently measuring the default nine row counts at the default context is the
/// same class of bug as an A/B whose two arms are the same binary — it produces
/// a full, plausible table that answers a different question than the one asked.
fn config(input: &str) -> Result<(u32, u32, Vec<u32>)> {
    let (mut short, mut long, mut rows) = (SHORT, LONG, ROWS.to_vec());
    for field in input.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (key, value) = field
            .split_once('=')
            .ok_or_else(|| format!("bad probe argument {field:?}: want key=value"))?;
        match key {
            "short" => short = value.parse().map_err(|_| format!("bad short {value:?}"))?,
            "long" => long = value.parse().map_err(|_| format!("bad long {value:?}"))?,
            "rows" => {
                rows = value
                    .split(':')
                    .map(|r| r.parse::<u32>().map_err(|_| format!("bad row {r:?}")))
                    .collect::<std::result::Result<Vec<u32>, String>>()?;
                if rows.is_empty() {
                    return Err("rows= is empty".to_string());
                }
            }
            _ => return Err(format!("unknown probe argument {key:?}")),
        }
    }
    Ok((short, long, rows))
}

/// Row counts. 1 is the baseline decode step; 4 is `draft::DRAFT_K` plus the
/// always-real row, which is the shape speculation actually fires; 8 is there to
/// show whether whatever happens between 1 and 4 keeps happening.
/// Extended past the speculation question to cover PREFILL widths, because a
/// resumed serving turn's prefill is a fire of a couple of hundred rows and it
/// measured ~1150 ms at a 7.4k context — far more than either half of the cost
/// model predicts. 32 is `sdpa_should_tile`'s crossover, so 8 and 64 bracket the
/// kernel switch; 189 is the exact width the opencode replay fires.
// Swept to find the INTERCEPT, which is the whole question here: a cached
// agentic turn prefills ~192 fresh tokens and takes 0.403 s, i.e. 476 tok/s
// against 2157 tok/s on the cold turn with the same kernels. That gap is a
// fixed per-fire term, and a fire-cost-versus-rows line is what prices it.
//
// 1 is a decode step. 184/192 are the fast widths and 189 the width the
// opencode replay actually fires -- kept together because the driver's mod-8
// row-count cliff (`r mod 8` in 1..6 costs a flat ~560 ms) was measured on the
// OLD kernels and has not been re-checked since attention and both GEMMs moved.
const ROWS: [u32; 9] = [1, 8, 32, 64, 128, 184, 189, 192, 512];

/// Fires per configuration, including warmup.
/// The driver refuses a fire that reads more logits rows than this.
const MAX_READOUT_ROWS: u32 = 8;

/// Last fire's guest-side and submit-side cost. A probe is single-threaded and
/// runs one fire at a time, so a static is enough and avoids threading a
/// struct through every call site.
static mut BUILD_US: u64 = 0;
static mut SUBMIT_US: u64 = 0;

const FIRES: usize = 10;
const WARMUP: usize = 2;

fn notes(seed: u32) -> String {
    let mut text = String::new();
    for i in seed..seed + 900 {
        text.push_str(&format!(
            "Repository note {i}: module pkg/mod_{i}.py defines helper_{i}(x), \
             which returns x * {i} and is covered by tests/test_mod_{i}.py.\n"
        ));
    }
    text
}

/// One prefill fire, `base..end`, writing at or past `base`. Used only to fill
/// the cache so there is a realistic amount of KV to read.
async fn prefill(
    ws: &WorkingSet,
    pipe: &Pipeline,
    toks: &[i32],
    base: u32,
    end: u32,
    page: u32,
    pool_ids: &[u32],
) -> std::result::Result<(), String> {
    let len = end - base;
    let t = Channel::from(&toks[base as usize..end as usize]).named("p_toks");
    let embed_indptr = Channel::from([0u32, len]).named("p_indptr");
    let positions = Channel::from_iter(base..end).named("p_pos");
    let w_slot = Channel::from(
        (base..end)
            .map(|c| pool_ids[(c / page) as usize])
            .collect::<Vec<u32>>(),
    )
    .named("p_wslot");
    let w_off = Channel::from((base..end).map(|c| c % page).collect::<Vec<u32>>()).named("p_woff");
    let kv_len = Channel::from([end]).named("p_klen");
    let pages = Channel::from(pool_ids.to_vec()).named("p_pages");
    let page_indptr = Channel::from([0u32, end.div_ceil(page)]).named("p_pidx");
    let out = Channel::new([1], dtype::i32).named("p_out");

    let fwd: ForwardPass = ForwardPass::new();
    fwd.embed(&t, &embed_indptr)?;
    fwd.attention(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (base / page)..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    fwd.epilogue(move || out.put(&reduce_argmax(intrinsics::logits())));
    fwd.submit(pipe)
        .with_context(|| format!("prefill submit {base}..{end}"))?;
    out.take_host::<i32>()
        .await
        .with_context(|| format!("prefill take {base}..{end}"))?;
    Ok(())
}

/// One `rows`-row decode fire at context `ctx`, timed end to end.
///
/// This is exactly the shape `opencode-session` fires when it verifies a draft:
/// `rows` tokens embedded, written at `ctx..ctx+rows`, read out per row, argmax
/// per row. For `rows == 1` it is exactly an ordinary decode step. Nothing else
/// differs between the two, which is what makes the comparison mean something.
///
/// The clock covers building the trace as well as running it, because the decode
/// loop pays both on every token and a cheap fire behind an expensive trace is
/// not a cheap step.
async fn timed_fire(
    ws: &WorkingSet,
    pipe: &Pipeline,
    tok: i32,
    ctx: u32,
    rows: u32,
    page: u32,
    pool_ids: &[u32],
) -> std::result::Result<u64, String> {
    let start = Instant::now();
    let end = ctx + rows;
    let t = Channel::from(vec![tok; rows as usize]).named("d_toks");
    let embed_indptr = Channel::from([0u32, rows]).named("d_indptr");
    let positions = Channel::from_iter(ctx..end).named("d_pos");
    let w_slot = Channel::from(
        (ctx..end)
            .map(|c| pool_ids[(c / page) as usize])
            .collect::<Vec<u32>>(),
    )
    .named("d_wslot");
    let w_off = Channel::from((ctx..end).map(|c| c % page).collect::<Vec<u32>>()).named("d_woff");
    let kv_len = Channel::from([end]).named("d_klen");
    let pages = Channel::from(pool_ids.to_vec()).named("d_pages");
    let page_indptr = Channel::from([0u32, end.div_ceil(page)]).named("d_pidx");
    // ROW indices into this fire's own input, not absolute positions: the fire
    // embeds exactly `rows` tokens, so its rows are 0..rows.
    //
    // Capped at 8. The driver refuses more -- "this fire would read more than
    // the driver's 8 logits rows" -- which is a hard ceiling on how wide a
    // speculative verify can be, and the reason the prefill-width rows here read
    // out ONE row instead of all of them. That is what a real prefill does
    // anyway: `prefill_span` samples a single mid-prompt row and discards it.
    let out_rows = rows.min(MAX_READOUT_ROWS);
    let readout = Channel::from_iter(0..out_rows).named("d_readout");
    let out = Channel::new([out_rows], dtype::i32).named("d_out");

    let fwd: ForwardPass = ForwardPass::new();
    fwd.embed(&t, &embed_indptr)?;
    fwd.readout(&readout)?;
    fwd.attention(
        ws,
        KvGeometry {
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
    )?;
    fwd.epilogue(move || out.put(&reduce_argmax(intrinsics::logits())));
    // Three clocks, because "the fire costs 563 ms" does not say WHERE. `build`
    // is guest-side only (channel construction, the trace); `submit` is the WIT
    // call that hands the trace to the engine; `await` is everything after --
    // engine plan, driver encode, GPU, and the result's trip back.
    let t_build = start.elapsed();
    fwd.submit(pipe)
        .with_context(|| format!("decode submit rows={rows} ctx={ctx}"))?;
    let t_submit = start.elapsed();
    let got = out
        .take_host::<Vec<i32>>()
        .await
        .with_context(|| format!("decode take rows={rows} ctx={ctx}"))?;
    unsafe {
        BUILD_US = t_build.as_micros() as u64;
        SUBMIT_US = (t_submit - t_build).as_micros() as u64;
    }
    if got.len() != out_rows as usize {
        return Err(format!(
            "rows={rows} ctx={ctx}: read out {} value(s), expected {rows} — the \
             fire is not carrying one row per read-out index and its time means \
             nothing",
            got.len()
        ));
    }
    Ok(start.elapsed().as_micros() as u64)
}

/// Median, which is what to report when a background compile or a page fault can
/// double one sample and nothing can halve one.
fn median(v: &[u64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_unstable();
    let n = s.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        s[n / 2] as f64
    } else {
        (s[n / 2 - 1] + s[n / 2]) as f64 / 2.0
    }
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let page = kv_page_size();
    let (short_arg, long_arg, row_set) = config(positional(&_input))?;
    let ctx_long = (long_arg / page) * page;
    let ctx_short = (short_arg / page) * page;
    if ctx_short > ctx_long {
        return Err(format!(
            "short {ctx_short} exceeds long {ctx_long}: the slope solve below \
             divides by (long - short)"
        ));
    }

    // Enough tokens to fill the long context, plus room for the widest fire.
    let text = notes(0);
    let mut toks: Vec<i32> = model::encode(&text).into_iter().map(|t| t as i32).collect();
    let need = (ctx_long + 2048 + 64) as usize;
    while toks.len() < need {
        let more = toks.clone();
        toks.extend(more);
    }
    toks.truncate(need);

    let ws = WorkingSet::new();
    let want = (ctx_long + 2048 + 64).div_ceil(page);
    let have = ws.page_len();
    let mut pool_ids: Vec<u32> = (0..have).collect();
    if want > have {
        pool_ids.extend_from_slice(ws.reserve(want - have)?.ids());
    }
    let pipe = Pipeline::new();

    println!(
        "[rows] page={page} short={ctx_short} long={ctx_long} pool_pages={} chunk_cap={}",
        pool_ids.len(),
        max_embed_length()
    );

    for (b, e) in prefill_chunks(ctx_long, None) {
        prefill(&ws, &pipe, &toks, b, e, page, &pool_ids).await?;
    }
    println!("[rows] prefilled {ctx_long} tokens");

    // (rows, context) -> median us, and the first-fire cost of that shape.
    let mut table: Vec<(u32, u32, f64, u64)> = Vec::new();
    for &ctx in &[ctx_short, ctx_long] {
        for &rows in &row_set {
            let mut samples = Vec::new();
            for i in 0..FIRES {
                let us = timed_fire(&ws, &pipe, toks[0], ctx, rows, page, &pool_ids).await?;
                samples.push(us);
                if i == 0 {
                    // Reported, not averaged in: this is the driver compiling a
                    // program it has not seen, which every new row count pays
                    // once and which a serving loop pays again whenever the
                    // 64-entry program cache evicts it.
                    println!("[rows] ctx={ctx} rows={rows} first_fire_us={us}");
                }
            }
            let first = samples[0];
            let steady = median(&samples[WARMUP..]);
            println!(
                "[rows] ctx={ctx} rows={rows} build_ms={:.2} submit_ms={:.2}",
                unsafe { BUILD_US } as f64 / 1000.0,
                unsafe { SUBMIT_US } as f64 / 1000.0
            );
            println!(
                "[rows] ctx={ctx} rows={rows} median_ms={:.2} samples_ms={:?}",
                steady / 1000.0,
                samples[WARMUP..]
                    .iter()
                    .map(|u| (*u as f64 / 1000.0 * 100.0).round() / 100.0)
                    .collect::<Vec<f64>>()
            );
            table.push((rows, ctx, steady, first));
        }
    }

    // Solve t = fixed + slope * context per row count, from the two points.
    // `slope` is the KV read; `fixed` is everything that happens once per fire
    // regardless of how much cache there is (weights, launch, trace build).
    let at = |rows: u32, ctx: u32| -> f64 {
        table
            .iter()
            .find(|(r, c, _, _)| *r == rows && *c == ctx)
            .map(|(_, _, t, _)| *t / 1000.0)
            .unwrap_or(0.0)
    };
    let span_k = (ctx_long - ctx_short) as f64 / 1000.0;
    let base_slope = (at(1, ctx_long) - at(1, ctx_short)) / span_k;
    let mut lines = Vec::new();
    for &rows in &row_set {
        let s = (at(rows, ctx_long) - at(rows, ctx_short)) / span_k;
        let fixed = at(rows, ctx_short) - s * (ctx_short as f64 / 1000.0);
        lines.push(format!(
            "rows={rows} fixed_ms={fixed:.2} slope_ms_per_1k={s:.3} \
             slope_vs_1row={:.2}x long_ms={:.1} long_vs_1row={:.2}x",
            if base_slope > 0.0 { s / base_slope } else { 0.0 },
            at(rows, ctx_long),
            if at(1, ctx_long) > 0.0 {
                at(rows, ctx_long) / at(1, ctx_long)
            } else {
                0.0
            }
        ));
    }
    for l in &lines {
        println!("[rows] {l}");
    }

    // The verdict, stated as the thing speculation needs to be true. At the long
    // context a 5-row fire either costs about one fire (rows share the KV read,
    // and 4 accepted tokens arrive nearly free) or about five (they do not).
    let k = ROWS.iter().copied().find(|&r| r == 189).unwrap_or(189);
    let ratio = if at(1, ctx_long) > 0.0 {
        at(k, ctx_long) / at(1, ctx_long)
    } else {
        0.0
    };
    let verdict = if ratio < 1.5 {
        "ROWS_SHARE_KV"
    } else if ratio > (k as f64) * 0.7 {
        "ROWS_PAY_FULL_KV"
    } else {
        "ROWS_PARTIALLY_SHARE_KV"
    };
    let result = format!(
        "DECODE_ROWS_PROBE {verdict} k={k} ratio_at_{ctx_long}={ratio:.2}x \
         base_slope_ms_per_1k={base_slope:.3} | {}",
        lines.join(" | ")
    );
    println!("{result}");
    pipe.close();
    Ok(result)
}
