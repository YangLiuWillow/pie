//! B2-mask microbench — masked vs rebuild condensation over a LONG run.
//!
//! Stage-1 measured ONE condensation. This measures the **compounding**: a
//! long-horizon agent runs many turns, its context repeatedly overflows a
//! `context_limit`, and each time it condenses down to `sink_size` head tokens
//! plus `keep_recent` recent tokens (dropping the stale middle). Every
//! condensation is a re-prefill that vLLM+prefix-caching must pay (dropping the
//! middle is not a prefix op → positions shift → rebuild the kept suffix), and
//! that Pie avoids by masking the middle in place (0 re-prefill, no RoPE work).
//!
//! Two paths run the SAME schedule:
//!   - **MASK** (Pie-only): one growing Context; at each condensation just widen
//!     the attention mask's gap → 0 re-prefill; the runtime trims masked pages.
//!   - **REBUILD** (vLLM+APC / the stock condenser): at each condensation build a
//!     fresh Context of [sink + keep_recent] and re-prefill it.
//!
//! Headline = SUM of rebuild re-prefill over all condensations (grows with run
//! length) vs mask's 0. Coherence = the two paths keep the SAME attended token
//! set between condensations, so their decoded tokens should match — masking is
//! free AND behaviorally equivalent to rebuild, not a lossy shortcut. Synthetic
//! token ids; only timing + the mask-vs-rebuild token agreement matter.

use inferlet::{Context, Result, model::Model, runtime, sample::{Sampler, Logprob}};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Input {
    /// QUALITY mode: real tokenized text. When non-empty, the inferlet predicts
    /// the next token at the last `num_queries` positions three ways (full /
    /// mask / rebuild) and returns them as JSON — isolating the positional
    /// effect of mask- vs rebuild-condensation. When empty, runs the timing
    /// compounding bench below.
    #[serde(default)]
    token_ids: Vec<u32>,
    #[serde(default = "default_num_queries")]
    num_queries: u32,
    /// FUTURE-GEN quality mode: history length P before condensation. When > 0
    /// (and token_ids non-empty), predicts the CONTINUATION tokens [P, P+nq)
    /// (true future generation, not in-window re-prediction) after condensing
    /// to sink + last `keep_recent` of the history. Sweeping P past the model's
    /// trained range is the beyond-trained-length stress test.
    #[serde(default)]
    history_len: u32,
    #[serde(default = "default_turns")]
    turns: u32,
    #[serde(default = "default_turn_tokens")]
    turn_tokens: u32,
    #[serde(default = "default_decode_per_turn")]
    decode_per_turn: u32,
    #[serde(default = "default_context_limit")]
    context_limit: u32,
    #[serde(default = "default_sink_size")]
    sink_size: u32,
    #[serde(default = "default_keep_recent")]
    keep_recent: u32,
    /// TRAJ-REPLAY mode: token offset where each recorded turn (assistant +
    /// observation) begins, parallel to a captured OpenHands trajectory. When
    /// non-empty (with token_ids), replays the FIXED token stream through
    /// full/mask/rebuild condensers at every turn where the context would
    /// overflow `context_limit`, probing the model's prediction of the REAL
    /// continuation. This removes the layer-B nondeterminism confound: both
    /// condensers see byte-identical tokens, so any perplexity gap is purely the
    /// condensation mechanism (position-preservation vs re-positioning).
    #[serde(default)]
    turn_starts: Vec<u32>,
    /// TRAJ-REPLAY: number of most-recent turns kept attended on condense (the
    /// system+task prefix [0, turn_starts[0]) is always kept; the middle is
    /// dropped). Mirrors the agent's `condense_keep_recent`.
    #[serde(default = "default_keep_recent_turns")]
    keep_recent_turns: u32,
}

fn default_keep_recent_turns() -> u32 { 12 }

fn default_num_queries() -> u32 { 64 }
fn default_turns() -> u32 { 120 }
fn default_turn_tokens() -> u32 { 300 }
fn default_decode_per_turn() -> u32 { 8 }
fn default_context_limit() -> u32 { 8000 }
fn default_sink_size() -> u32 { 64 }
fn default_keep_recent() -> u32 { 2000 }

fn tok(i: u32) -> u32 { 1000 + (i % 30000) }

/// BRLE for [sink True, gap False, window True] over `seq_len`. All-true if the
/// sequence still fits inside sink+window (no middle to drop yet). Used for a
/// single-token decode pass (n == 1 → exactly one mask).
fn build_mask(seq_len: u32, sink: u32, window: u32) -> Vec<u32> {
    let kept = sink + window;
    if seq_len <= kept {
        vec![0, seq_len]
    } else {
        vec![0, sink, seq_len - kept, window]
    }
}

/// Per-query masks for a multi-token prefill. `attention_mask` needs ONE BRLE
/// per query position (n masks for n new tokens). Query j (absolute position
/// `logical + j`) attends causally to [0, sink) ∪ [logical - window, logical+j]
/// — sink head + the recent window + the new tokens up to itself — with the
/// stale middle [sink, logical - window) masked out. The gap is constant across
/// j (= logical - sink - window); only the recent run grows with j.
fn build_prefill_masks(logical: u32, tpt: u32, sink: u32, window: u32) -> Vec<Vec<u32>> {
    (0..tpt)
        .map(|j| {
            let p1 = logical + j + 1; // causal length for this query
            let recent = window + j + 1; // sink-excluded attended recent span
            if p1 <= sink + recent {
                vec![0, p1] // no gap yet — fully causal
            } else {
                vec![0, sink, p1 - sink - recent, recent]
            }
        })
        .collect()
}

/// QUALITY mode. Predict the next token at the last `num_queries` positions of a
/// real token stream three ways, differing only in how the kept context is
/// treated, and return the predictions as JSON for the host to score:
///   - FULL: attend to the entire prefix (gold reference).
///   - MASK: attend to sink + recent window, tokens at their ORIGINAL positions
///     (Pie's free condensation — no RoPE re-encode).
///   - REBUILD: attend to sink + recent window, tokens RE-POSITIONED compactly
///     (what the stock condenser / vLLM+APC does).
/// The MASK-vs-REBUILD gap isolates the positional-encoding effect; both-vs-FULL
/// isolates the information loss from dropping the middle.
async fn quality_mode(
    model: &Model, ids: &[u32], sink: u32, keep: u32, num_queries: u32,
) -> Result<String> {
    let l = ids.len() as u32;
    if l < sink + keep + num_queries + 2 {
        return Err("token_ids too short for sink+keep+num_queries".into());
    }
    // Query positions p predict ids[p+1], for the last `num_queries` positions.
    let p_first = l - 1 - num_queries; // inclusive
    let p_last = l - 2; // inclusive
    let full_idx: Vec<u32> = (p_first..=p_last).collect();

    let gold: Vec<u32> = (p_first..=p_last).map(|p| ids[(p + 1) as usize]).collect();

    // FULL — probe log p(gold[p] | full prefix) at each query position.
    let mut fctx = Context::new(model)?;
    let mut fp = fctx.forward();
    fp.input(ids);
    let fhs: Vec<_> = full_idx.iter().zip(&gold)
        .map(|(&p, &g)| fp.probe(p, Logprob(g))).collect();
    let fout = fp.execute().await?;
    let full_lp: Vec<f32> = fhs.iter()
        .map(|&h| fout.logprobs(h).and_then(|s| s.first().copied()).unwrap_or(f32::NAN))
        .collect();

    // MASK — same prefill, per-query sink+window masks (ORIGINAL positions).
    let mut mctx = Context::new(model)?;
    let mut mp = mctx.forward();
    mp.input(ids);
    let masks: Vec<Vec<u32>> = (0..l).map(|p| {
        let p1 = p + 1;
        if p1 <= sink + keep {
            vec![0, p1]
        } else {
            vec![0, sink, p1 - sink - keep, keep]
        }
    }).collect();
    mp.attention_mask(&masks);
    let mhs: Vec<_> = full_idx.iter().zip(&gold)
        .map(|(&p, &g)| mp.probe(p, Logprob(g))).collect();
    let mout = mp.execute().await?;
    let mask_lp: Vec<f32> = mhs.iter()
        .map(|&h| mout.logprobs(h).and_then(|s| s.first().copied()).unwrap_or(f32::NAN))
        .collect();

    // REBUILD — compact context [sink head] ++ [last keep tokens], re-positioned.
    let mut rctx = Context::new(model)?;
    let mut compact: Vec<u32> = ids[..sink as usize].to_vec();
    compact.extend_from_slice(&ids[(l - keep) as usize..]);
    let mut rp = rctx.forward();
    rp.input(&compact);
    // Doc position p = (l - keep) + (i - sink) for compact index i >= sink.
    let rebuild_idx: Vec<u32> = (p_first..=p_last).map(|p| sink + p - (l - keep)).collect();
    let rhs: Vec<_> = rebuild_idx.iter().zip(&gold)
        .map(|(&i, &g)| rp.probe(i, Logprob(g))).collect();
    let rout = rp.execute().await?;
    let rebuild_lp: Vec<f32> = rhs.iter()
        .map(|&h| rout.logprobs(h).and_then(|s| s.first().copied()).unwrap_or(f32::NAN))
        .collect();

    Ok(format!(
        "{{\"full_lp\":{:?},\"mask_lp\":{:?},\"rebuild_lp\":{:?}}}",
        full_lp, mask_lp, rebuild_lp,
    ))
}

/// Commit `tokens` into `ctx` in <=4096-token chunks (a single fire_batch over a
/// long prompt OOMs on big models). KV accumulates; sampled tokens discarded.
async fn prefill_chunked(ctx: &mut Context, tokens: &[u32]) -> Result<()> {
    for chunk in tokens.chunks(4096) {
        let mut p = ctx.forward();
        p.input(chunk);
        let _ = p.sample(&[chunk.len() as u32 - 1], Sampler::Argmax);
        p.execute().await?;
    }
    Ok(())
}

/// FUTURE-GEN quality mode — predict the continuation [P, P+nq) three ways after
/// condensing the history [0, P) to sink + last `keep`. This is true forward
/// generation (not in-window re-prediction), and sweeping P is the
/// beyond-trained-length stress test:
///   - FULL: continuation attends to the entire history at original positions.
///   - MASK: attends sink + recent window at ORIGINAL positions (Pie, free) —
///     positions grow with P, so this is what breaks past the trained range.
///   - REBUILD: attends sink + recent window RE-POSITIONED compactly (stock
///     condenser) — positions stay small regardless of P.
async fn quality_futuregen(
    model: &Model, ids: &[u32], sink: u32, keep: u32, nq: u32, p: u32,
) -> Result<String> {
    let l = ids.len() as u32;
    if p < sink + keep + 2 || p + nq + 1 > l {
        return Err("token_ids too short for history_len + num_queries".into());
    }
    let cont = &ids[p as usize..(p + nq) as usize]; // teacher-forced continuation
    let gold: Vec<u32> = (0..nq).map(|j| ids[(p + j + 1) as usize]).collect();

    // FULL — full history, continuation attends causally to everything.
    let mut fctx = Context::new(model)?;
    prefill_chunked(&mut fctx, &ids[..p as usize]).await?;
    let mut fp = fctx.forward();
    fp.input(cont);
    let fhs: Vec<_> = (0..nq).zip(&gold).map(|(j, &g)| fp.probe(j, Logprob(g))).collect();
    let fout = fp.execute().await?;
    let full_lp: Vec<f32> = fhs.iter()
        .map(|&h| fout.logprobs(h).and_then(|s| s.first().copied()).unwrap_or(f32::NAN)).collect();

    // MASK — same history KV; continuation queries attend sink + recent (original
    // positions), middle masked. Per-query BRLE over [0, P+j].
    let mut mctx = Context::new(model)?;
    prefill_chunked(&mut mctx, &ids[..p as usize]).await?;
    let mut mp = mctx.forward();
    mp.input(cont);
    let masks: Vec<Vec<u32>> = (0..nq).map(|j| {
        let end = p + j + 1; // causal length for this continuation query
        let recent = keep + j + 1; // sink-excluded attended span
        if end <= sink + recent { vec![0, end] }
        else { vec![0, sink, end - sink - recent, recent] }
    }).collect();
    mp.attention_mask(&masks);
    let mhs: Vec<_> = (0..nq).zip(&gold).map(|(j, &g)| mp.probe(j, Logprob(g))).collect();
    let mout = mp.execute().await?;
    let mask_lp: Vec<f32> = mhs.iter()
        .map(|&h| mout.logprobs(h).and_then(|s| s.first().copied()).unwrap_or(f32::NAN)).collect();

    // REBUILD — compact context [sink] ++ [last keep of history], re-positioned;
    // continuation attends causally to that (small positions regardless of P).
    let mut rctx = Context::new(model)?;
    let mut compact: Vec<u32> = ids[..sink as usize].to_vec();
    compact.extend_from_slice(&ids[(p - keep) as usize..p as usize]);
    prefill_chunked(&mut rctx, &compact).await?;
    let mut rp = rctx.forward();
    rp.input(cont);
    let rhs: Vec<_> = (0..nq).zip(&gold).map(|(j, &g)| rp.probe(j, Logprob(g))).collect();
    let rout = rp.execute().await?;
    let rebuild_lp: Vec<f32> = rhs.iter()
        .map(|&h| rout.logprobs(h).and_then(|s| s.first().copied()).unwrap_or(f32::NAN)).collect();

    Ok(format!(
        "{{\"history_len\":{},\"full_lp\":{:?},\"mask_lp\":{:?},\"rebuild_lp\":{:?}}}",
        p, full_lp, mask_lp, rebuild_lp,
    ))
}

/// TRAJ-REPLAY quality mode — replay a REAL captured OpenHands trajectory (fixed
/// token stream `ids` + per-turn offsets `turn_starts`) through full/mask/rebuild
/// condensers. At every turn `t` where the running context [0, turn_starts[t])
/// exceeds `context_limit` (and there are >`keep` turns so a middle exists to
/// drop), condense to prefix + last `keep` turns and probe the model's next-token
/// logprob over the REAL continuation (the first `nq` tokens the agent actually
/// produced at that turn) three ways:
///   - FULL:    attends the entire history at original positions (reference).
///   - MASK:    attends prefix + last-`keep`-turns at ORIGINAL positions, middle
///              masked (Pie — free, 0 re-prefill).
///   - REBUILD: attends prefix ++ last-`keep`-turns RE-POSITIONED compactly (the
///              stock condenser / what vLLM+APC must rebuild).
/// Because the probed tokens are identical across arms, the mask-vs-rebuild
/// logprob gap isolates the positional effect on a real agent trajectory.
async fn traj_replay(
    model: &Model, ids: &[u32], turn_starts: &[u32], keep: u32, nq: u32, limit: u32,
) -> Result<String> {
    let l = ids.len() as u32;
    let nt = turn_starts.len() as u32;
    if nt < keep + 2 {
        return Err("trajectory has too few turns for keep_recent_turns".into());
    }
    let prefix_end = turn_starts[0]; // end of system+task = start of turn 0

    // FULL and MASK both keep the entire [0, p) KV resident, so p must stay
    // within the model's trained/max context — which is ALSO mask's applicable
    // regime (beyond trained length, mask's ever-growing positions break and
    // rebuild's re-positioning is required; see hardening job 19069519). Cap
    // probe history at MAX_HISTORY so the run stays valid.
    const MAX_HISTORY: u32 = 60000;

    // Probe at every turn whose start overflows the limit and leaves a droppable
    // middle. Cap the count (stride) so a very long trajectory stays affordable.
    let mut cand: Vec<u32> = Vec::new();
    // Require t > keep (strictly) so at least one middle turn is actually
    // dropped (t == keep leaves an empty middle → mask ≡ full, nothing to test).
    for t in 0..nt {
        let p = turn_starts[t as usize];
        if p > limit && p <= MAX_HISTORY && t > keep && p + nq + 1 <= l {
            cand.push(t);
        }
    }
    if cand.is_empty() {
        return Err("no turn overflows context_limit (trajectory fits — lower context_limit or capture a longer run)".into());
    }
    let max_probes = 16usize;
    let stride = (cand.len() + max_probes - 1) / max_probes;
    let probes: Vec<u32> = cand.iter().step_by(stride.max(1)).copied().collect();

    let mut out_turn: Vec<u32> = Vec::new();
    let mut out_p: Vec<u32> = Vec::new();
    let mut out_kept: Vec<u32> = Vec::new();     // dropped-middle token count
    let mut out_full: Vec<f32> = Vec::new();     // mean logprob over nq
    let mut out_mask: Vec<f32> = Vec::new();
    let mut out_rebuild: Vec<f32> = Vec::new();
    let mut out_reprefill_ms: Vec<f64> = Vec::new(); // rebuild re-prefill cost

    let mean = |xs: &[f32]| -> f32 {
        let v: Vec<f32> = xs.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() { f32::NAN } else { v.iter().sum::<f32>() / v.len() as f32 }
    };

    for &t in &probes {
        let p = turn_starts[t as usize];
        let kept_start = turn_starts[(t - keep) as usize]; // first kept recent turn
        let cont = &ids[p as usize..(p + nq) as usize];
        let gold: Vec<u32> = (0..nq).map(|j| ids[(p + j + 1) as usize]).collect();

        // FULL — entire history [0, p).
        let mut fctx = Context::new(model)?;
        prefill_chunked(&mut fctx, &ids[..p as usize]).await?;
        let mut fp = fctx.forward();
        fp.input(cont);
        let fhs: Vec<_> = (0..nq).zip(&gold).map(|(j, &g)| fp.probe(j, Logprob(g))).collect();
        let fout = fp.execute().await?;
        let full_lp: Vec<f32> = fhs.iter()
            .map(|&h| fout.logprobs(h).and_then(|s| s.first().copied()).unwrap_or(f32::NAN)).collect();

        // MASK — same full KV, continuation queries drop [prefix_end, kept_start)
        // at ORIGINAL positions. Per-query causal-length BRLE.
        let mut mctx = Context::new(model)?;
        prefill_chunked(&mut mctx, &ids[..p as usize]).await?;
        let mut mp = mctx.forward();
        mp.input(cont);
        let masks: Vec<Vec<u32>> = (0..nq).map(|j| {
            let end = p + j + 1; // causal length for this continuation query
            // keep [0, prefix_end) True, [prefix_end, kept_start) False (dropped),
            // [kept_start, end) True. BRLE starts-with-False (0-len false run).
            if kept_start <= prefix_end {
                vec![0, end] // nothing to drop → full causal
            } else {
                vec![0, prefix_end, kept_start - prefix_end, end - kept_start]
            }
        }).collect();
        mp.attention_mask(&masks);
        let mhs: Vec<_> = (0..nq).zip(&gold).map(|(j, &g)| mp.probe(j, Logprob(g))).collect();
        let mout = mp.execute().await?;
        let mask_lp: Vec<f32> = mhs.iter()
            .map(|&h| mout.logprobs(h).and_then(|s| s.first().copied()).unwrap_or(f32::NAN)).collect();

        // REBUILD — compact [prefix] ++ [last keep turns], re-positioned; time the
        // re-prefill (the cost mask avoids and APC must pay to drop the middle).
        let mut rctx = Context::new(model)?;
        let mut compact: Vec<u32> = ids[..prefix_end as usize].to_vec();
        compact.extend_from_slice(&ids[kept_start as usize..p as usize]);
        let rt = Instant::now();
        prefill_chunked(&mut rctx, &compact).await?;
        let reprefill_ms = rt.elapsed().as_secs_f64() * 1000.0;
        let mut rp = rctx.forward();
        rp.input(cont);
        let rhs: Vec<_> = (0..nq).zip(&gold).map(|(j, &g)| rp.probe(j, Logprob(g))).collect();
        let rout = rp.execute().await?;
        let rebuild_lp: Vec<f32> = rhs.iter()
            .map(|&h| rout.logprobs(h).and_then(|s| s.first().copied()).unwrap_or(f32::NAN)).collect();

        out_turn.push(t);
        out_p.push(p);
        out_kept.push(kept_start - prefix_end);
        out_full.push(mean(&full_lp));
        out_mask.push(mean(&mask_lp));
        out_rebuild.push(mean(&rebuild_lp));
        out_reprefill_ms.push(reprefill_ms);
    }

    Ok(format!(
        "{{\"mode\":\"traj_replay\",\"num_turns\":{},\"num_probes\":{},\"nq\":{},\
\"context_limit\":{},\"keep_recent_turns\":{},\
\"probe_turns\":{:?},\"history_len\":{:?},\"dropped_mid_tokens\":{:?},\
\"full_lp\":{:?},\"mask_lp\":{:?},\"rebuild_lp\":{:?},\"rebuild_reprefill_ms\":{:?}}}",
        nt, probes.len(), nq, limit, keep,
        out_turn, out_p, out_kept, out_full, out_mask, out_rebuild, out_reprefill_ms,
    ))
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("No models available")?)?;

    // TRAJ-REPLAY mode: fixed real-trajectory stream + per-turn offsets.
    if !input.token_ids.is_empty() && !input.turn_starts.is_empty() {
        return traj_replay(
            &model, &input.token_ids, &input.turn_starts,
            input.keep_recent_turns, input.num_queries, input.context_limit,
        ).await;
    }

    // FUTURE-GEN quality mode (history_len > 0) or in-window quality mode.
    if !input.token_ids.is_empty() {
        if input.history_len > 0 {
            return quality_futuregen(
                &model, &input.token_ids, input.sink_size,
                input.keep_recent, input.num_queries, input.history_len,
            ).await;
        }
        return quality_mode(
            &model, &input.token_ids, input.sink_size,
            input.keep_recent, input.num_queries,
        ).await;
    }

    let turns = input.turns;
    let tpt = input.turn_tokens;
    let dpt = input.decode_per_turn;
    let limit = input.context_limit;
    let sink = input.sink_size;
    let keep = input.keep_recent;

    // ─── MASK path ───────────────────────────────────────────────────────
    // One growing context. `logical` = raw appended tokens. `window` = the
    // recent span kept attended (grows as we append, drops to `keep` on
    // condense). `gap` masked middle = logical - sink - window.
    let mut ctx = Context::new(&model)?;
    let mut logical: u32 = 0;
    let mut window: u32 = 0; // attended recent span (excludes sink)
    let mut mask_condensations = 0u32;
    let mut mask_reprefill_ms = 0.0_f64;   // definitionally 0 across condensations
    let mut mask_prefill_ms = 0.0_f64;     // incremental new-token prefill (both pay)
    let mut mask_decode_ms = 0.0_f64;
    let mut mask_tokens: Vec<u32> = Vec::new();

    for _turn in 0..turns {
        // Append this turn's new observation tokens.
        let new: Vec<u32> = (logical..logical + tpt).map(tok).collect();
        let t = Instant::now();
        let mut pass = ctx.forward();
        pass.input(&new);
        let masks = build_prefill_masks(logical, tpt, sink, window);
        pass.attention_mask(&masks);
        let hh = pass.sample(&[tpt - 1], Sampler::Argmax);
        let out = pass.execute().await?;
        let mut next = out.token(hh).ok_or("empty mask prefill")?;
        mask_prefill_ms += t.elapsed().as_secs_f64() * 1000.0;
        logical += tpt;
        window += tpt;

        // Decode this turn's action tokens under the current mask.
        let td = Instant::now();
        for _ in 0..dpt {
            let mut pass = ctx.forward();
            pass.input(&[next]);
            let seq = pass.start_position() + 1;
            let m = build_mask(seq, sink, window);
            pass.attention_mask(&[m]);
            let hh = pass.sample(&[0], Sampler::Argmax);
            let out = pass.execute().await?;
            next = out.token(hh).ok_or("empty mask decode")?;
            logical += 1;
            window += 1;
        }
        mask_decode_ms += td.elapsed().as_secs_f64() * 1000.0;
        mask_tokens.push(next);

        // Condense: drop the middle by shrinking the attended window to `keep`.
        // Pure mask update — 0 re-prefill.
        if sink + window > limit {
            mask_condensations += 1;
            window = keep;
            // mask_reprefill_ms += 0.0 (definitional)
        }
    }

    // ─── REBUILD path ────────────────────────────────────────────────────
    // Same schedule; at each condensation build a fresh [sink + keep] context
    // and re-prefill it (the cost vLLM+APC must pay to drop the middle).
    let mut rctx = Context::new(&model)?;
    let mut effective: u32 = 0;  // sink + attended recent (drives the trigger)
    let mut rebuild_condensations = 0u32;
    let mut rebuild_reprefill_ms = 0.0_f64;
    let mut rebuild_prefill_ms = 0.0_f64;
    let mut rebuild_decode_ms = 0.0_f64;
    let mut rebuild_tokens: Vec<u32> = Vec::new();
    let mut global_pos: u32 = 0; // mirrors the mask path's logical clock for tok()

    for _turn in 0..turns {
        let new: Vec<u32> = (global_pos..global_pos + tpt).map(tok).collect();
        let t = Instant::now();
        let mut pass = rctx.forward();
        pass.input(&new);
        let hh = pass.sample(&[tpt - 1], Sampler::Argmax);
        let out = pass.execute().await?;
        let mut next = out.token(hh).ok_or("empty rebuild prefill")?;
        rebuild_prefill_ms += t.elapsed().as_secs_f64() * 1000.0;
        effective += tpt;
        global_pos += tpt;

        let td = Instant::now();
        for _ in 0..dpt {
            let mut pass = rctx.forward();
            pass.input(&[next]);
            let hh = pass.sample(&[0], Sampler::Argmax);
            let out = pass.execute().await?;
            next = out.token(hh).ok_or("empty rebuild decode")?;
            effective += 1;
            global_pos += 1;
        }
        rebuild_decode_ms += td.elapsed().as_secs_f64() * 1000.0;
        rebuild_tokens.push(next);

        if sink + effective > limit {
            rebuild_condensations += 1;
            // Rebuild: fresh context of [sink head] + [last `keep` tokens].
            rctx = Context::new(&model)?;
            let mut kept: Vec<u32> = (0..sink).map(tok).collect();
            kept.extend((global_pos - keep..global_pos).map(tok));
            let klen = kept.len() as u32;
            let tr = Instant::now();
            let mut rp = rctx.forward();
            rp.input(&kept);
            let _ = rp.sample(&[klen - 1], Sampler::Argmax);
            let _ = rp.execute().await?;
            rebuild_reprefill_ms += tr.elapsed().as_secs_f64() * 1000.0;
            effective = keep;
        }
    }

    // ─── Coherence: mask vs rebuild decoded-token agreement ──────────────
    let compared = mask_tokens.len().min(rebuild_tokens.len());
    let matches = (0..compared)
        .filter(|&i| mask_tokens[i] == rebuild_tokens[i])
        .count();

    println!("=== mask-condense-bench (compounding) ===");
    println!("turns={turns} turn_tokens={tpt} decode_per_turn={dpt} context_limit={limit} sink={sink} keep_recent={keep}");
    println!("mask_condensations={mask_condensations}");
    println!("rebuild_condensations={rebuild_condensations}");
    println!("mask_reprefill_total_ms={mask_reprefill_ms:.3}       # Pie: 0 across the whole run");
    println!("rebuild_reprefill_total_ms={rebuild_reprefill_ms:.3}   # vLLM+APC: SUM over condensations (compounds)");
    println!("reprefill_saved_total_ms={rebuild_reprefill_ms:.3}    # cumulative pie-mask advantage");
    println!("mask_prefill_ms={mask_prefill_ms:.3} rebuild_prefill_ms={rebuild_prefill_ms:.3}   # incremental new-token prefill (both pay)");
    println!("mask_decode_ms={mask_decode_ms:.3} rebuild_decode_ms={rebuild_decode_ms:.3}");
    println!("coherence_token_match={matches}/{compared}   # mask ≡ rebuild behaviorally (mod engine nondeterminism)");
    Ok(String::new())
}
