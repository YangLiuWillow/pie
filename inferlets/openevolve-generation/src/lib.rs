//! openevolve-generation — the "analyze-then-diverge" candidate-generation
//! inferlet for the OpenEvolve ↔ Pie integration (Path B, B-batched).
//!
//! One call = one sampled parent's whole generation batch:
//!
//!   Stage 0  L1p  — build or OPEN the parent-prompt block (system + task +
//!                   parent code + top-K). A named snapshot keyed by
//!                   (run_id, parent_id, topk_sig); content-deterministic, so
//!                   any worker of the run reuses its committed KV.
//!   Stage 1  L1g  — fork L1p and GENERATE a shared analysis A (parent
//!                   weaknesses + M improvement directions) *once*. This is the
//!                   load-bearing node: its KV covers generated tokens, and all
//!                   children fork off it. vLLM+APC can only match this via a
//!                   client round-trip exposed to eviction — see
//!                   memory `project_openevolve_integration`.
//!   Stage 2  leaves — fork L1g into N children in-memory; each appends a tiny
//!                   per-leaf steer s_k and decodes one SEARCH/REPLACE diff.
//!                   All N share the P+A KV by refcount; only the steer suffix
//!                   is prefilled per child.
//!
//! Lifetime (settled): L1p is a persistent named snapshot (bounded by the
//! MAP-Elites archive size; backend prunes stale topk_sig versions via
//! action="delete"). L1g is deleted at end-of-batch by default (A is sampled →
//! keeping it forever leaks memory and risks staleness); `keep_analysis` /
//! `reuse_analysis` opt into cross-generation reuse as a measured arm.
//!
//! TODO(hardening): (1) pre-render sections to token ids on the backend for
//! byte-exact cross-process reuse instead of relying on chat-template
//! determinism; (2) trim the L1g snapshot to a clean message boundary (exclude
//! any trailing cue) like openhands-coder-session; (3) kv_verify assertions.

use futures::future;
use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};

mod prefix_cache;

// ─── Input ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Input {
    run_id: String,
    parent_id: String,
    /// Legacy host-computed cache key. The inferlet now content-addresses L1p/L1g
    /// from the rendered prompt tokens (see [`prefix_cache`]), so this is no
    /// longer load-bearing; kept so existing backend payloads keep deserializing.
    #[allow(dead_code)]
    topk_sig: String,

    sections: Sections,

    #[serde(default = "default_num_children")]
    num_children: usize,

    /// Per-leaf divergence steer. Length 0 (temperature-only diversity) or N.
    #[serde(default)]
    steers: Vec<String>,

    #[serde(default)]
    reuse_analysis: bool,
    #[serde(default)]
    keep_analysis: bool,
    /// Skip Stage 1 entirely: no shared analysis is generated; the N leaves
    /// fork the L1p prompt block directly. num_children=1 + skip_analysis is a
    /// plain (L1p-cached) completion — the LLMInterface single-call path.
    #[serde(default)]
    skip_analysis: bool,
    #[serde(default)]
    action: Option<String>,

    #[serde(default = "default_analysis_max_tokens")]
    analysis_max_tokens: usize,
    #[serde(default = "default_child_max_tokens")]
    child_max_tokens: usize,

    #[serde(default = "default_analysis_temperature")]
    analysis_temperature: f32,
    /// Length 0 = default, 1 = broadcast to all N, N = per-leaf.
    #[serde(default)]
    child_temperature: Vec<f32>,
    #[serde(default)]
    child_top_p: Vec<f32>,
}

#[derive(Deserialize)]
struct Sections {
    system: String,
    task: String,
    parent_block: String,
    analysis_instruction: String,
}

fn default_num_children() -> usize { 4 }
fn default_analysis_max_tokens() -> usize { 512 }
fn default_child_max_tokens() -> usize { 1024 }
fn default_analysis_temperature() -> f32 { 0.7 }
fn default_child_top_p() -> f32 { 0.95 }

// ─── Output ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct Output {
    parent_id: String,
    analysis: AnalysisOut,
    children: Vec<ChildOut>,
    telemetry: Telemetry,
}

#[derive(Serialize)]
struct AnalysisOut {
    text: String,
    /// FNV-1a-64 hex of A's token ids (content address of this analysis).
    ver: String,
    /// "generated" | "reused"
    mode: String,
}

#[derive(Serialize)]
struct ChildOut {
    steer_index: usize,
    diff: String,
    tokens_generated: usize,
}

#[derive(Serialize)]
struct Telemetry {
    /// "opened" (cross-worker KV hit) | "built"
    l1p_mode: String,
    l1p_prefill_tokens: usize,
    /// "generated" | "reused"
    l1g_mode: String,
    /// Tokens the shared analysis A decoded to (0 when reused).
    l1g_decode_tokens: usize,
    /// Tokens of P+A shared by every child (the forked prefix length).
    shared_prefix_tokens: usize,
    /// Sum of per-child steer-suffix prefill (the only per-child prefill).
    leaves_prefill_tokens: usize,
    /// (N-1)·shared_prefix_tokens — the re-prefill Pie avoided vs. N naive
    /// independent completions. The headline benchmark number.
    shared_prefill_saved: usize,
}

// ─── Snapshot naming (content-addressed; username-scoped globally) ─────────
//
// L1p/L1g are keyed by a hash of the *rendered L1p prompt tokens* — automatic
// prefix caching. `run_id` namespaces the snapshots (isolates concurrent runs
// sharing one Pie username); `key` is the content hash of the exact prompt the
// KV holds. Two workers that render the same parent-prompt produce the same
// name and share KV, with no dependence on the backend's `topk_sig` (kept in
// the input only for backward compat / telemetry). L1g is keyed by the *prompt*
// content (not the variable analysis tokens) so a later worker can reopen "the
// analysis for this prompt" by identity — the same reuse semantics as the old
// (parent, topk) key, but self-derived from the actual tokens.

fn l1p_name(run_id: &str, key: &str) -> String {
    format!("oe/p/{run_id}/{key}")
}
fn l1g_name(run_id: &str, key: &str) -> String {
    format!("oe/g/{run_id}/{key}")
}

/// Render the L1p parent-prompt block (system + user) to tokens — the exact
/// stream Stage 0 commits, so its content hash addresses that KV.
fn render_l1p(model: &Model, s: &Sections) -> Vec<u32> {
    let mut t = chat::system(model, &s.system);
    t.extend(chat::user(model, &format!(
        "{}\n\n{}\n\n{}",
        s.task, s.parent_block, s.analysis_instruction
    )));
    t
}

fn fnv1a64(tokens: &[u32]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
    }
    h
}

/// temp <= 0 → greedy; else nucleus sampling.
fn sampler(temperature: f32, top_p: f32) -> Sampler {
    if temperature <= 0.0 {
        Sampler::Argmax
    } else {
        Sampler::TopP { temperature, p: top_p }
    }
}

/// Broadcast rule for the per-leaf sampler vectors.
fn pick(vals: &[f32], k: usize, default: f32) -> f32 {
    match vals.len() {
        0 => default,
        1 => vals[0],
        _ => vals.get(k).copied().unwrap_or(default),
    }
}

// ─── Entry point ───────────────────────────────────────────────────────────

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    // Content-addressed key = hash of the rendered L1p prompt tokens. Self-keyed
    // from the actual tokens, so cross-worker reuse doesn't depend on the
    // backend's topk_sig (now non-load-bearing).
    let l1p_tokens = render_l1p(&model, &input.sections);
    let key = prefix_cache::content_hash(&model_name, prefix_cache::TEMPLATE_MARKER, &l1p_tokens);
    let l1p = l1p_name(&input.run_id, &key);
    let l1g = l1g_name(&input.run_id, &key);

    // action="delete": backend-driven lifecycle pruning of a dropped elite.
    if input.action.as_deref() == Some("delete") {
        let _ = Context::delete(&model, &l1p);
        let _ = Context::delete(&model, &l1g);
        return Ok(Output {
            parent_id: input.parent_id,
            analysis: AnalysisOut { text: String::new(), ver: String::new(), mode: "deleted".into() },
            children: Vec::new(),
            telemetry: Telemetry {
                l1p_mode: "deleted".into(), l1p_prefill_tokens: 0,
                l1g_mode: "deleted".into(), l1g_decode_tokens: 0,
                shared_prefix_tokens: 0, leaves_prefill_tokens: 0, shared_prefill_saved: 0,
            },
        });
    }

    // ── Stage 0: build or open L1p (parent-prompt block) ──────────────────
    let (base, l1p_mode, l1p_prefill) = match Context::open(&model, &l1p) {
        Ok(c) => {
            // Cross-worker hit: committed KV shared by refcount, nothing to prefill.
            (c, "opened", 0usize)
        }
        Err(_) => {
            let mut c = Context::new(&model)?;
            c.append(&l1p_tokens);
            c.flush().await?;
            let prefill = c.seq_len() as usize;
            // Content-addressed: an existing snapshot of this name already holds
            // identical KV, so a colliding save (get-or-create race between
            // same-parent workers) is benign — ignore it, don't delete+resave.
            let _ = c.save(&l1p);
            (c, "built", prefill)
        }
    };

    // ── Stage 1: reuse or generate the shared analysis A (L1g) ────────────
    // skip_analysis short-circuits Stage 1: leaves fork the prompt block
    // directly (plain L1p-cached completion; the single-call path).
    let reuse_open = if input.skip_analysis {
        None
    } else if input.reuse_analysis {
        Context::open(&model, &l1g).ok()
    } else {
        None
    };

    let (with_a, analysis_text, a_ver, l1g_mode, l1g_decode) = if input.skip_analysis {
        (base.fork()?, String::new(), String::new(), "skipped", 0usize)
    } else { match reuse_open {
        Some(c) => {
            // Reused: A's tokens live in the snapshot; we don't re-decode it.
            // (ver left empty on reuse — the caller keyed by parent+topk, which
            //  is what addresses the reused analysis.)
            (c, String::new(), String::new(), "reused", 0usize)
        }
        None => {
            let mut c = base.fork()?; // in-memory fork of the prompt block
            c.cue();
            let a_tokens = c
                .generate(sampler(input.analysis_temperature, default_child_top_p()))
                .max_tokens(input.analysis_max_tokens)
                .collect_tokens()
                .await?;
            let ver = format!("{:x}", fnv1a64(&a_tokens));
            let text = model
                .tokenizer()
                .decode(&a_tokens)
                .unwrap_or_else(|_| String::from("[decode error]"));
            // Publish L1g for sibling / cross-generation reuse (best-effort).
            let _ = Context::delete(&model, &l1g);
            let _ = c.save(&l1g);
            (c, text, ver, "generated", a_tokens.len())
        }
    } };

    // ── Stage 2: fork N leaves off P+A, decode diffs concurrently ─────────
    let n = input.num_children.max(1);
    let shared_prefix_tokens = with_a.seq_len() as usize;

    let leaves = (0..n)
        .map(|k| {
            let mut leaf = with_a.fork()?; // in-memory KV share of P+A
            let steer = input.steers.get(k).cloned().unwrap_or_default();
            let temp = pick(&input.child_temperature, k, default_analysis_temperature());
            let top_p = pick(&input.child_top_p, k, default_child_top_p());
            let max_tokens = input.child_max_tokens;
            Ok(async move {
                if !steer.is_empty() {
                    leaf.user(&steer);
                }
                leaf.cue();
                let prefill = leaf.seq_len() as usize - shared_prefix_tokens;
                let tokens = leaf
                    .generate(sampler(temp, top_p))
                    .max_tokens(max_tokens)
                    .collect_tokens()
                    .await?;
                let diff = leaf
                    .model()
                    .tokenizer()
                    .decode(&tokens)
                    .unwrap_or_else(|_| String::from("[decode error]"));
                Ok::<(usize, String, usize, usize), String>((k, diff, tokens.len(), prefill))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let results = future::join_all(leaves).await;

    let mut children = Vec::with_capacity(n);
    let mut leaves_prefill_tokens = 0usize;
    for r in results {
        let (k, diff, n_gen, prefill) = r?;
        leaves_prefill_tokens += prefill;
        children.push(ChildOut { steer_index: k, diff, tokens_generated: n_gen });
    }
    children.sort_by_key(|c| c.steer_index);

    // Drop the shared analysis unless we were told to keep it (reuse implies
    // keep). Nothing to drop when Stage 1 was skipped.
    if !input.skip_analysis && !input.keep_analysis && !input.reuse_analysis {
        let _ = Context::delete(&model, &l1g);
    }

    let shared_prefill_saved = shared_prefix_tokens * n.saturating_sub(1);

    Ok(Output {
        parent_id: input.parent_id,
        analysis: AnalysisOut { text: analysis_text, ver: a_ver, mode: l1g_mode.into() },
        children,
        telemetry: Telemetry {
            l1p_mode: l1p_mode.into(),
            l1p_prefill_tokens: l1p_prefill,
            l1g_mode: l1g_mode.into(),
            l1g_decode_tokens: l1g_decode,
            shared_prefix_tokens,
            leaves_prefill_tokens,
            shared_prefill_saved,
        },
    })
}
