//! Native Parallel Reasoner (NPR, arXiv 2512.07461) — Phase 2.
//!
//! Reimplements the NPR Engine's fork/merge scheduler (patched SGLang,
//! `schedule_batch.py` / `scheduler.py` in the NPR release) as guest-side
//! inferlet logic:
//!
//! - Decode until the model closes a `</guideline>` block, parse its
//!   `<plan>i:` entries, and fork one context per plan off the shared
//!   prefix (`Context::fork` — committed pages are shared copy-on-write).
//! - Decode all `<step>` branches concurrently; the engine batches the
//!   concurrent forward passes. Branch tokens are charged `x degree`
//!   against the global budget, mirroring NPR's branch-aware token ledger.
//! - Join the branches and continue decoding the `<takeaway>` Reduce stage;
//!   rounds repeat until the final answer.
//!
//! Two join modes (`join_mode` input):
//!
//! - `"refill"` (default, faithful — NPR Algorithm 2 semantics): sibling
//!   step tokens are re-filled into branch 1's context at **overlapped
//!   position ids** (every sibling restarts at the position right after
//!   `</guideline>`) with per-token BRLE attention masks exposing only the
//!   shared prefix + the sibling's own tokens — reproducing the KV the
//!   NPR Engine gets by stitching branch pages. The `<takeaway>` continues
//!   at `p_fork + max(branch extents)`; all later decoding runs with
//!   positions decoupled from KV slots (`Generator::position_offset` +
//!   slot-causal masks). Nested parallel blocks fall back to sequential
//!   decoding in this mode (exact nested refill needs per-token position
//!   and visibility records; NPR trajectories are overwhelmingly flat
//!   sequences of depth-1 rounds).
//! - `"textual"` (phase-1 baseline): sibling tokens appended causally with
//!   sequential positions — off-distribution for the NPR-trained model but
//!   useful as an A/B baseline. Supports nesting.
//!
//! `selftest=true` runs a numeric oracle instead of the normal flow: a
//! sibling refilled with hole masks at overlapped positions must produce
//! the same last-token distribution as the same text decoded alone off the
//! shared prefix (its KV is then bit-equivalent up to kernel batching);
//! a causal refill without hole masks is the negative control.
//!
//! Tag detection is byte-level over the vocab table, so it works whether
//! the tags are single special tokens (the NPR checkpoint) or ordinary
//! BPE splits (any stock model, e.g. for smoke tests).

use futures::future;
use inferlet::model::{Model, Tokenizer};
use inferlet::sample::{Distribution, Sampler};
use inferlet::{Context, Result, chat, runtime};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::time::Instant;

/// The NPR format instruction (verbatim `evals/prompts/npr.txt` from the
/// NPR release); appended after the question in the user turn, matching
/// `evals/evaluate.py` prompt construction.
const INSTRUCTION: &str = include_str!("../prompt.txt");

const TAG_GUIDELINE_END: &[u8] = b"</guideline>";
const TAG_STEP_END: &[u8] = b"</step>";

/// Max tokens per refill forward pass (well under the driver's
/// max_forward_tokens; chunked refills carry per-chunk mask rows).
const REFILL_CHUNK: usize = 512;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_question")]
    question: String,
    /// Global token budget. NPR charges each branch token multiplied by its
    /// parallel degree so K branches together spend like one sequential run.
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    /// Max `<plan>` entries per `<guideline>` (NPR: 5; more = malformed,
    /// falls back to sequential decoding).
    #[serde(default = "default_max_plans")]
    max_plans: usize,
    /// Max nesting depth of parallel blocks (NPR: 5). Effective only in
    /// `textual` join mode; `refill` restricts forking to depth 1.
    #[serde(default = "default_max_depth")]
    max_depth: usize,
    /// Minimum remaining budget to allow a fork (NPR: 1024).
    #[serde(default = "default_min_fork_budget")]
    min_fork_budget: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// "refill" (faithful NPR join) or "textual" (phase-1 causal append).
    #[serde(default = "default_join_mode")]
    join_mode: String,
    /// Test hook: text injected as if already generated, before decoding
    /// starts. A primer ending in `</guideline>` triggers the fork path
    /// immediately — lets the fork/join machinery be exercised with a
    /// stock model that does not emit the NPR format on its own.
    #[serde(default)]
    primer: Option<String>,
    /// Run the refill-join numeric oracle instead of the normal flow.
    #[serde(default)]
    selftest: bool,
    /// Test hook: cap each branch at this many generated tokens (treated
    /// as a terminal stop). Lets smoke tests with stock models — which
    /// never emit `</step>` — leave budget for the post-join stages.
    #[serde(default)]
    max_step_tokens: Option<usize>,
}

fn default_question() -> String {
    "What is the domain of the function f(x) = (2 - x) / log(2 - log(x - 2)), \
     where log is the base 10 logarithm function? Express your answer in \
     interval notation."
        .to_string()
}
fn default_max_new_tokens() -> usize {
    30000
}
fn default_max_plans() -> usize {
    5
}
fn default_max_depth() -> usize {
    5
}
fn default_min_fork_budget() -> usize {
    1024
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    0.7
}
fn default_join_mode() -> String {
    "refill".to_string()
}

// =============================================================================
// Shared state
// =============================================================================

struct Ledger {
    budget: usize,
    charged: usize,
}

impl Ledger {
    fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.charged)
    }
}

#[derive(Default)]
struct Stats {
    parallel_blocks: usize,
    branches_total: usize,
    max_depth_seen: usize,
    sequential_fallbacks: usize,
    tokens_generated: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum JoinMode {
    Refill,
    Textual,
}

struct Shared {
    tokenizer: Tokenizer,
    /// token id -> raw bytes, covering ordinary vocab + special tokens.
    token_bytes: Vec<Vec<u8>>,
    /// special-token byte string -> id, for composing tag fills.
    special_ids: HashMap<Vec<u8>, u32>,
    chat_stops: Vec<u32>,
    temperature: f32,
    top_p: f32,
    max_plans: usize,
    max_depth: usize,
    min_fork_budget: usize,
    max_step_tokens: Option<usize>,
    join_mode: JoinMode,
    ledger: RefCell<Ledger>,
    stats: RefCell<Stats>,
}

impl Shared {
    fn new(model: &Model, input: &Input) -> Result<Self> {
        let tokenizer = model.tokenizer();
        let (ids, seqs) = tokenizer.vocabs();
        let (sids, sseqs) = tokenizer.special_tokens();
        let max_id = ids
            .iter()
            .chain(sids.iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let mut token_bytes = vec![Vec::new(); max_id + 1];
        for (id, bytes) in ids.iter().zip(seqs) {
            token_bytes[*id as usize] = bytes;
        }
        let mut special_ids = HashMap::new();
        for (id, bytes) in sids.iter().zip(sseqs) {
            token_bytes[*id as usize] = bytes.clone();
            special_ids.insert(bytes, *id);
        }
        let join_mode = match input.join_mode.as_str() {
            "refill" => JoinMode::Refill,
            "textual" => JoinMode::Textual,
            other => return Err(format!("unknown join_mode: {other}")),
        };
        Ok(Shared {
            tokenizer,
            token_bytes,
            special_ids,
            chat_stops: chat::stop_tokens(model),
            temperature: input.temperature,
            top_p: input.top_p,
            max_plans: input.max_plans,
            max_depth: input.max_depth,
            min_fork_budget: input.min_fork_budget,
            max_step_tokens: input.max_step_tokens,
            join_mode,
            ledger: RefCell::new(Ledger {
                budget: input.max_new_tokens,
                charged: 0,
            }),
            stats: RefCell::new(Stats::default()),
        })
    }

    fn sampler(&self) -> Sampler {
        Sampler::TopP {
            temperature: self.temperature,
            p: self.top_p,
        }
    }

    fn bytes_of(&self, id: u32) -> &[u8] {
        self.token_bytes
            .get(id as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Encode text that may contain NPR structural tags. If a tag is a
    /// registered special token (the NPR checkpoint), its single id is
    /// spliced in directly — plain `encode` cannot be trusted to match
    /// added special tokens. Otherwise the tag text is BPE-encoded like
    /// any other text (stock-model smoke tests).
    fn encode_with_tags(&self, text: &str) -> Vec<u32> {
        const TAGS: [&str; 8] = [
            "<guideline>",
            "</guideline>",
            "<plan>",
            "</plan>",
            "<step>",
            "</step>",
            "<takeaway>",
            "</takeaway>",
        ];
        let mut out = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            let mut earliest: Option<(usize, &str)> = None;
            for tag in TAGS {
                if let Some(pos) = rest.find(tag) {
                    if earliest.map_or(true, |(p, _)| pos < p) {
                        earliest = Some((pos, tag));
                    }
                }
            }
            let Some((pos, tag)) = earliest else {
                out.extend(self.tokenizer.encode(rest));
                break 'outer;
            };
            if pos > 0 {
                out.extend(self.tokenizer.encode(&rest[..pos]));
            }
            match self.special_ids.get(tag.as_bytes()) {
                Some(&id) => out.push(id),
                None => out.extend(self.tokenizer.encode(tag)),
            }
            rest = &rest[pos + tag.len()..];
        }
        out
    }
}

// =============================================================================
// PCtx — a context plus its slot→position mapping
// =============================================================================

/// A context whose position ids may be decoupled from KV-slot indices.
///
/// Invariant: every token that enters KV at slot `s` (from now until the
/// next join changes it) carries position id `s + delta`. `delta` is 0
/// until the first refill join compresses positions.
struct PCtx {
    ctx: Context,
    delta: i64,
}

impl PCtx {
    fn new(ctx: Context) -> Self {
        PCtx { ctx, delta: 0 }
    }

    /// KV slot the next buffered/generated token will occupy.
    fn next_slot(&self) -> u32 {
        self.ctx.seq_len() + self.ctx.buffer().len() as u32
    }

    /// Position id the next buffered/generated token will carry.
    fn next_pos(&self) -> u32 {
        (self.next_slot() as i64 + self.delta) as u32
    }

    /// Re-anchor so the next token carries position `p`.
    fn set_next_pos(&mut self, p: u32) {
        self.delta = p as i64 - self.next_slot() as i64;
    }

    fn fork(&self) -> Result<PCtx> {
        Ok(PCtx {
            ctx: self.ctx.fork()?,
            delta: self.delta,
        })
    }
}

/// Slot-causal mask rows for `n` tokens appended after `kv_before` slots:
/// row i sees every slot already in KV plus itself — `[0, kv_before+i+1]`.
fn causal_rows(kv_before: u32, n: usize) -> Vec<Vec<u32>> {
    (0..n).map(|i| vec![0, kv_before + i as u32 + 1]).collect()
}

/// Refill `tokens` into `p.ctx` at explicit `positions` with the given
/// per-token mask rows (chunked). Tokens occupy the next free KV slots and
/// commit; the model recomputes their KV under the supplied layout.
async fn refill(p: &mut PCtx, tokens: &[u32], positions: &[u32], rows: &[Vec<u32>]) -> Result<()> {
    debug_assert_eq!(tokens.len(), positions.len());
    debug_assert_eq!(tokens.len(), rows.len());
    let mut off = 0;
    while off < tokens.len() {
        let end = (off + REFILL_CHUNK).min(tokens.len());
        println!(
            "[npr] refill chunk: {} tokens at slots {}.., positions {}..{}",
            end - off,
            p.ctx.seq_len(),
            positions[off],
            positions[end - 1]
        );
        let mut pass = p.ctx.forward();
        pass.input(&tokens[off..end]);
        pass.positions(&positions[off..end]);
        pass.attention_mask(&rows[off..end]);
        pass.pass_speculation(false);
        pass.execute().await?;
        off = end;
    }
    Ok(())
}

// =============================================================================
// Segment decoding
// =============================================================================

enum Stop {
    /// The model just closed a `</guideline>` — fork point.
    GuidelineEnd,
    /// The model just closed a `</step>` — branch complete.
    StepEnd,
    /// Chat-template end-of-turn.
    Eos,
    /// Global token budget exhausted.
    Budget,
}

struct Segment {
    tokens: Vec<u32>,
    bytes: Vec<u8>,
    stop: Stop,
}

/// Decode one segment: until a watched tag closes, the turn ends, or the
/// budget runs out. Tag tokens stay in the stream (NPR `no_stop_trim`
/// semantics): the Generator has no tag stops, so the tag token is
/// returned to us *and* staged in the context buffer for the next pass.
async fn decode_segment(
    sh: &Shared,
    p: &mut PCtx,
    degree: usize,
    watch_step: bool,
    token_cap: Option<usize>,
) -> Result<Segment> {
    let mut tokens = Vec::new();
    let mut bytes = Vec::new();
    let delta = p.delta;
    // Pass-level speculation (run-ahead staging) is disabled throughout:
    // staged passes assume plain causal continuation, and stale staged
    // entries for destroyed branch contexts can race the join's refills.
    let mut generator = p
        .ctx
        .generate(sh.sampler())
        .stop(&sh.chat_stops)
        .position_offset(delta)
        .disable_pass_speculation();
    loop {
        if sh.ledger.borrow().remaining() < degree
            || token_cap.is_some_and(|cap| tokens.len() >= cap)
        {
            return Ok(Segment {
                tokens,
                bytes,
                stop: Stop::Budget,
            });
        }
        let Some(token) = generator.next_token().await? else {
            return Ok(Segment {
                tokens,
                bytes,
                stop: Stop::Eos,
            });
        };
        tokens.push(token);
        bytes.extend_from_slice(sh.bytes_of(token));
        sh.ledger.borrow_mut().charged += degree;
        sh.stats.borrow_mut().tokens_generated += 1;
        if bytes.ends_with(TAG_GUIDELINE_END) {
            return Ok(Segment {
                tokens,
                bytes,
                stop: Stop::GuidelineEnd,
            });
        }
        if watch_step && bytes.ends_with(TAG_STEP_END) {
            return Ok(Segment {
                tokens,
                bytes,
                stop: Stop::StepEnd,
            });
        }
    }
}

// =============================================================================
// Fork / branch / join
// =============================================================================

struct BranchResult {
    p: PCtx,
    /// Everything this branch added beyond the fork prefix: the
    /// `\n<step>\n{i}:` header, generated tokens, and any nested joins
    /// (textual mode only — refill mode restricts branches to flat
    /// contiguous content).
    tokens: Vec<u32>,
    bytes: Vec<u8>,
    /// The branch ended on end-of-turn or budget instead of `</step>`.
    terminal: bool,
}

enum ForkOutcome {
    /// Branches ran and were joined into the caller's context.
    Forked {
        tokens: Vec<u32>,
        bytes: Vec<u8>,
        terminal: bool,
    },
    /// Malformed/over-budget/too-deep — caller keeps decoding sequentially
    /// (NPR resets `finished_reason` and lets the request continue).
    Sequential,
}

/// Handle a just-closed `</guideline>`: parse plans and, if the block is
/// well-formed and affordable, fork/decode/join. `segment_text` is the
/// decoded text of the segment that ended with this `</guideline>`.
fn try_fork<'a>(
    sh: &'a Rc<Shared>,
    p: &'a mut PCtx,
    segment_text: &'a str,
    depth: usize,
) -> Pin<Box<dyn Future<Output = Result<ForkOutcome>> + 'a>> {
    Box::pin(async move {
        let plans = parse_plans(segment_text);
        let affordable = sh.ledger.borrow().remaining() >= sh.min_fork_budget;
        // Refill mode forks only at depth 1: exact nested refill needs
        // per-token position/visibility records (see module docs).
        let depth_cap = match sh.join_mode {
            JoinMode::Refill => 1,
            JoinMode::Textual => sh.max_depth,
        };
        if plans.is_empty() || plans.len() > sh.max_plans || depth > depth_cap || !affordable {
            sh.stats.borrow_mut().sequential_fallbacks += 1;
            return Ok(ForkOutcome::Sequential);
        }
        let degree = plans.len();
        {
            let mut stats = sh.stats.borrow_mut();
            stats.parallel_blocks += 1;
            stats.branches_total += degree;
            stats.max_depth_seen = stats.max_depth_seen.max(depth);
        }

        // Captured before forking: the fork point in slot and position
        // space. The parent's pending buffer (which ends with the
        // `</guideline>` token) is inherited by every child, so the shared
        // prefix every sibling may see spans slots [0, fork_slots).
        let fork_slots = p.next_slot();
        let p_fork = p.next_pos();
        println!(
            "[npr] depth {depth}: forking {degree} branches at pos {p_fork}: {:?}",
            plans
        );

        let mut branch_futures = Vec::with_capacity(degree);
        for label in plans {
            let child = p.fork()?;
            branch_futures.push(run_branch(sh.clone(), child, label, depth, degree));
        }
        let results = future::join_all(branch_futures).await;
        let mut branches = Vec::with_capacity(degree);
        for r in results {
            branches.push(r?);
        }
        println!(
            "[npr] depth {depth}: all {degree} branches done ({} tokens total), joining ({})",
            branches.iter().map(|b| b.tokens.len()).sum::<usize>(),
            match sh.join_mode {
                JoinMode::Refill => "refill",
                JoinMode::Textual => "textual",
            }
        );

        let mut iter = branches.into_iter();
        let first = iter.next().expect("at least one branch");
        let mut base = first.p;
        let mut tokens = first.tokens;
        let mut bytes = first.bytes;
        let mut terminal = first.terminal;

        match sh.join_mode {
            JoinMode::Textual => {
                // Phase-1 join: append sibling tokens causally in plan order
                // + the takeaway cue. Sequential positions.
                for sibling in iter {
                    base.ctx.append(&sibling.tokens);
                    tokens.extend_from_slice(&sibling.tokens);
                    bytes.extend_from_slice(&sibling.bytes);
                    terminal |= sibling.terminal;
                    sibling.p.ctx.destroy();
                }
                let takeaway = sh.encode_with_tags("<takeaway>\n");
                base.ctx.append(&takeaway);
                tokens.extend_from_slice(&takeaway);
                bytes.extend_from_slice(b"<takeaway>\n");
            }
            JoinMode::Refill => {
                // Faithful join (NPR Algorithm 2 realized as recomputation):
                //
                // 1. Drain base's pending tail (its `</step>` tag) into KV at
                //    its own positions so the whole base branch is resident.
                let mut max_extent = tokens.len() as u32; // base extent
                {
                    let pend = base.ctx.take_buffer();
                    if !pend.is_empty() {
                        let start = (base.ctx.seq_len() as i64 + base.delta) as u32;
                        let positions: Vec<u32> =
                            (start..start + pend.len() as u32).collect();
                        let rows = causal_rows(base.ctx.seq_len(), pend.len());
                        refill(&mut base, &pend, &positions, &rows).await?;
                    }
                }
                // 2. Refill each sibling's tokens at positions restarting at
                //    p_fork, masked to see only [0, fork_slots) plus its own
                //    tokens — reproducing the KV each branch computed in its
                //    own context (bit-equivalent up to kernel batching).
                for sibling in iter {
                    let n = sibling.tokens.len();
                    max_extent = max_extent.max(n as u32);
                    let sib_start_slot = base.ctx.seq_len();
                    let positions: Vec<u32> = (p_fork..p_fork + n as u32).collect();
                    let rows: Vec<Vec<u32>> = (0..n)
                        .map(|i| {
                            vec![
                                0,
                                fork_slots,
                                sib_start_slot - fork_slots,
                                i as u32 + 1,
                            ]
                        })
                        .collect();
                    refill(&mut base, &sibling.tokens, &positions, &rows).await?;
                    tokens.extend_from_slice(&sibling.tokens);
                    bytes.extend_from_slice(&sibling.bytes);
                    terminal |= sibling.terminal;
                    sibling.p.ctx.destroy();
                }
                // 3. The Reduce stage: `<takeaway>\n` starts at
                //    p_fork + max(extent) — the position right after the
                //    longest branch — and sees everything. The final cue
                //    token stays buffered so the next Generator step has
                //    input; positions re-anchor accordingly.
                let takeaway = sh.encode_with_tags("<takeaway>\n");
                let max_end = p_fork + max_extent;
                let head = &takeaway[..takeaway.len() - 1];
                if !head.is_empty() {
                    let positions: Vec<u32> =
                        (max_end..max_end + head.len() as u32).collect();
                    let rows = causal_rows(base.ctx.seq_len(), head.len());
                    refill(&mut base, head, &positions, &rows).await?;
                }
                base.ctx.append(&takeaway[takeaway.len() - 1..]);
                base.set_next_pos(max_end + takeaway.len() as u32);
                tokens.extend_from_slice(&takeaway);
                bytes.extend_from_slice(b"<takeaway>\n");
            }
        }

        // The pre-fork parent context is superseded by the merged branch.
        let merged = std::mem::replace(p, base);
        // (merged is the old parent PCtx; drop its context explicitly.)
        merged.ctx.destroy();

        Ok(ForkOutcome::Forked {
            tokens,
            bytes,
            terminal,
        })
    })
}

/// Decode one `<step>` branch to its `</step>`. In textual mode nested
/// parallel blocks recurse; in refill mode they fall back to sequential
/// decoding inside the branch. Charges `degree` budget units per token.
fn run_branch(
    sh: Rc<Shared>,
    mut p: PCtx,
    label: String,
    depth: usize,
    degree: usize,
) -> Pin<Box<dyn Future<Output = Result<BranchResult>>>> {
    Box::pin(async move {
        // NPR forks each child with `prefix + "\n<step>\n{i}:"`.
        let header = format!("\n<step>\n{label}:");
        let header_tokens = sh.encode_with_tags(&header);
        p.ctx.append(&header_tokens);
        let mut tokens = header_tokens;
        let mut bytes = header.into_bytes();
        loop {
            let segment = decode_segment(&sh, &mut p, degree, true, sh.max_step_tokens).await?;
            tokens.extend_from_slice(&segment.tokens);
            bytes.extend_from_slice(&segment.bytes);
            match segment.stop {
                Stop::StepEnd => {
                    println!("[npr] branch {label}: </step> after {} tokens", tokens.len());
                    return Ok(BranchResult {
                        p,
                        tokens,
                        bytes,
                        terminal: false,
                    });
                }
                Stop::GuidelineEnd => {
                    // Nested parallel block inside this step.
                    let text = String::from_utf8_lossy(&segment.bytes).into_owned();
                    match try_fork(&sh, &mut p, &text, depth + 1).await? {
                        ForkOutcome::Forked {
                            tokens: join_tokens,
                            bytes: join_bytes,
                            terminal,
                        } => {
                            tokens.extend_from_slice(&join_tokens);
                            bytes.extend_from_slice(&join_bytes);
                            if terminal {
                                return Ok(BranchResult {
                                    p,
                                    tokens,
                                    bytes,
                                    terminal: true,
                                });
                            }
                        }
                        ForkOutcome::Sequential => {}
                    }
                }
                Stop::Eos | Stop::Budget => {
                    println!(
                        "[npr] branch {label}: terminal (eos/budget) after {} tokens",
                        tokens.len()
                    );
                    return Ok(BranchResult {
                        p,
                        tokens,
                        bytes,
                        terminal: true,
                    });
                }
            }
        }
    })
}

// =============================================================================
// Parsing helpers
// =============================================================================

/// Extract the plan labels of the *last* `<guideline>` block in the
/// segment, mirroring NPR's `<plan>\s*([0-9]+(?:\.[0-9]+)*)\s*:` regex
/// over `prefix_input[prefix_input.rfind("<guideline>"):]`.
fn parse_plans(segment: &str) -> Vec<String> {
    let Some(start) = segment.rfind("<guideline>") else {
        return Vec::new();
    };
    let mut rest = &segment[start..];
    let mut plans = Vec::new();
    while let Some(pos) = rest.find("<plan>") {
        rest = &rest[pos + "<plan>".len()..];
        let trimmed = rest.trim_start();
        let label: String = trimmed
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if label.is_empty() || !label.starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        if trimmed[label.len()..].trim_start().starts_with(':') {
            plans.push(label.trim_end_matches('.').to_string());
        }
    }
    plans
}

/// Last `\boxed{...}` in the trajectory, brace-matched.
fn extract_boxed(text: &str) -> Option<String> {
    let idx = text.rfind("boxed{")?;
    let body = &text[idx + "boxed{".len()..];
    let mut depth = 1usize;
    let mut out = String::new();
    for c in body.chars() {
        match c {
            '{' => {
                depth += 1;
                out.push(c);
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(out);
                }
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    None
}

// =============================================================================
// Selftest — refill-join numeric oracle
// =============================================================================

/// One selftest experiment: fork the base, optionally fill `a` causally,
/// then fill `b` with the given explicit positions/mask rows and probe the
/// distribution at b's last token.
async fn fill_and_probe(
    base: &Context,
    a: Option<&[u32]>,
    b: &[u32],
    positions: Option<Vec<u32>>,
    rows: Option<Vec<Vec<u32>>>,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut ctx = base.fork()?;
    if let Some(a) = a {
        let mut pass = ctx.forward();
        pass.input(a);
        pass.execute().await?;
    }
    let mut pass = ctx.forward();
    pass.input(b);
    if let Some(pos) = &positions {
        pass.positions(pos);
    }
    if let Some(r) = &rows {
        pass.attention_mask(r);
    }
    let h = pass.probe(
        b.len() as u32 - 1,
        Distribution {
            temperature: 1.0,
            k: 10,
        },
    );
    let out = pass.execute().await?;
    let (ids, probs) = out.distribution(h).ok_or("selftest: probe missing")?;
    Ok((ids.to_vec(), probs.to_vec()))
}

/// Total variation distance over the union of two top-k supports.
fn tv(a: &(Vec<u32>, Vec<f32>), b: &(Vec<u32>, Vec<f32>)) -> f32 {
    let mut m: HashMap<u32, (f32, f32)> = HashMap::new();
    for (i, p) in a.0.iter().zip(&a.1) {
        m.entry(*i).or_default().0 = *p;
    }
    for (i, p) in b.0.iter().zip(&b.1) {
        m.entry(*i).or_default().1 = *p;
    }
    m.values().map(|(x, y)| (x - y).abs()).sum::<f32>() / 2.0
}

/// Isolation matrix for the refill join:
///   e_pos    — explicit positions equal to natural       (positions plumbing)
///   e_mask   — explicit causal rows, natural positions   (mask plumbing)
///   e_short  — refill after a sub-page sibling A          (holes, no page trim)
///   e_refill — refill after a page-spanning sibling A     (holes + page trim)
///   e_ctl    — B causally after A, no masks               (negative control)
/// All of e_pos/e_mask/e_short/e_refill must match the reference; e_ctl
/// must not.
async fn selftest(model: &Model, sh: &Shared) -> Result<String> {
    let mut base = Context::new(model)?;
    base.user("Solve: what is the domain of f(x) = 1/log(2 - log(x - 2))? Reason briefly.");
    base.flush().await?;
    base.cue();
    let pend = base.take_buffer();
    let mut pass = base.forward();
    pass.input(&pend);
    pass.execute().await?;

    let p_fork = base.seq_len();
    let fork_slots = base.seq_len();

    let text_a = "\nStep 1: the inner logarithm needs x - 2 > 0, so x > 2. \
                  Also log(x-2) < 2 means x < 102.";
    let text_b = "\nStep 2: the outer logarithm needs 2 - log(x - 2) > 0 and \
                  the denominator must be non-zero, so 2 - log(x-2) != 1.";
    let toks_a = sh.tokenizer.encode(text_a);
    let toks_a_short = &toks_a[..8.min(toks_a.len())];
    let toks_b = sh.tokenizer.encode(text_b);
    let nb = toks_b.len();

    let natural_pos: Vec<u32> = (p_fork..p_fork + nb as u32).collect();
    let causal = |kv_before: u32| causal_rows(kv_before, nb);
    let holes = |n_a: u32| -> Vec<Vec<u32>> {
        (0..nb)
            .map(|i| vec![0, fork_slots, n_a, i as u32 + 1])
            .collect()
    };

    println!("[npr selftest] p_fork={p_fork} n_a={} n_a_short={} n_b={nb}", toks_a.len(), toks_a_short.len());
    println!("[npr selftest] e_ref...");
    let e_ref = fill_and_probe(&base, None, &toks_b, None, None).await?;
    println!("[npr selftest] e_pos...");
    let e_pos = fill_and_probe(&base, None, &toks_b, Some(natural_pos.clone()), None).await?;
    println!("[npr selftest] e_mask...");
    let e_mask = fill_and_probe(&base, None, &toks_b, None, Some(causal(fork_slots))).await?;
    println!("[npr selftest] e_4run...");
    // 4-run rows with a zero-length hole — semantically identical to causal;
    // discriminates multi-run row handling from hole semantics.
    let rows_4run: Vec<Vec<u32>> = (0..nb)
        .map(|i| vec![0, fork_slots, 0, i as u32 + 1])
        .collect();
    let e_4run = fill_and_probe(&base, None, &toks_b, None, Some(rows_4run)).await?;
    println!("[npr selftest] r_shift...");
    // Hole isolation at natural slot positions (no position overlap):
    // B after A(8) masked to prefix+self, vs B alone at the same shifted
    // positions. Both use explicit slot-causal/hole rows.
    let shifted_pos: Vec<u32> = (0..nb)
        .map(|i| p_fork + toks_a_short.len() as u32 + i as u32)
        .collect();
    let r_shift = fill_and_probe(
        &base,
        None,
        &toks_b,
        Some(shifted_pos.clone()),
        Some(causal(fork_slots)),
    )
    .await?;
    println!("[npr selftest] m_hole_nat...");
    let m_hole_nat = fill_and_probe(
        &base,
        Some(toks_a_short),
        &toks_b,
        Some(shifted_pos),
        Some(
            (0..nb)
                .map(|i| vec![0, fork_slots, toks_a_short.len() as u32, i as u32 + 1])
                .collect(),
        ),
    )
    .await?;
    println!("[npr selftest] e_short...");
    let e_short = fill_and_probe(
        &base,
        Some(toks_a_short),
        &toks_b,
        Some(natural_pos.clone()),
        Some(holes(toks_a_short.len() as u32)),
    )
    .await?;
    println!("[npr selftest] e_refill...");
    let e_refill = fill_and_probe(
        &base,
        Some(&toks_a),
        &toks_b,
        Some(natural_pos.clone()),
        Some(holes(toks_a.len() as u32)),
    )
    .await?;
    println!("[npr selftest] e_ctl...");
    let e_ctl = fill_and_probe(&base, Some(&toks_a), &toks_b, None, None).await?;

    let tvs = [
        ("positions", tv(&e_ref, &e_pos)),
        ("mask", tv(&e_ref, &e_mask)),
        ("mask_4run", tv(&e_ref, &e_4run)),
        ("hole_natural", tv(&r_shift, &m_hole_nat)),
        ("refill_short", tv(&e_ref, &e_short)),
        ("refill", tv(&e_ref, &e_refill)),
        ("control", tv(&e_ref, &e_ctl)),
    ];
    println!("[npr selftest] reference top-3: {:?}", &e_ref.0[..3.min(e_ref.0.len())]);
    for (name, v) in &tvs {
        println!("[npr selftest] TV(reference, {name}) = {v:.4}");
    }
    println!(
        "[npr selftest] tops: pos={:?} mask={:?} short={:?} refill={:?} ctl={:?}",
        &e_pos.0[..3.min(e_pos.0.len())],
        &e_mask.0[..3.min(e_mask.0.len())],
        &e_short.0[..3.min(e_short.0.len())],
        &e_refill.0[..3.min(e_refill.0.len())],
        &e_ctl.0[..3.min(e_ctl.0.len())],
    );

    let pass = tvs[..6].iter().all(|(_, v)| *v < 0.05) && tvs[6].1 > 0.05;
    Ok(inferlet::serde_json::json!({
        "selftest_pass": pass,
        "tv": tvs.iter().map(|(n, v)| (n.to_string(), *v)).collect::<HashMap<_, _>>(),
    })
    .to_string())
}

// =============================================================================
// Entry point
// =============================================================================

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();

    let models = runtime::models();
    let model_name = models.first().ok_or("no models available")?;
    let model = Model::load(model_name)?;

    let sh = Rc::new(Shared::new(&model, &input)?);

    if input.selftest {
        return selftest(&model, &sh).await;
    }

    // Prompt per NPR `evals/evaluate.py`: user turn = question + "\n\n" +
    // format instruction, then the generation cue. The user turn is flushed
    // (committing the shared prefix); the cue tokens stay buffered so the
    // first Generator step has input to sample from.
    let mut p = PCtx::new(Context::new(&model)?);
    p.ctx
        .user(&format!("{}\n\n{}", input.question, INSTRUCTION));
    p.ctx.flush().await?;
    p.ctx.cue();

    let mut trajectory_bytes: Vec<u8> = Vec::new();

    if let Some(primer) = input.primer.as_deref() {
        let tokens = sh.encode_with_tags(primer);
        p.ctx.append(&tokens);
        trajectory_bytes.extend_from_slice(primer.as_bytes());
        if primer.trim_end().ends_with("</guideline>") {
            match try_fork(&sh, &mut p, primer, 1).await? {
                ForkOutcome::Forked { bytes, .. } => {
                    trajectory_bytes.extend_from_slice(&bytes);
                }
                ForkOutcome::Sequential => {}
            }
        }
    }

    loop {
        let segment = decode_segment(&sh, &mut p, 1, false, None).await?;
        trajectory_bytes.extend_from_slice(&segment.bytes);
        match segment.stop {
            Stop::GuidelineEnd => {
                let text = String::from_utf8_lossy(&segment.bytes).into_owned();
                match try_fork(&sh, &mut p, &text, 1).await? {
                    ForkOutcome::Forked {
                        bytes, terminal, ..
                    } => {
                        trajectory_bytes.extend_from_slice(&bytes);
                        if terminal {
                            break;
                        }
                    }
                    ForkOutcome::Sequential => {}
                }
            }
            // `</step>` is not watched at top level; Eos/Budget end the run.
            Stop::StepEnd | Stop::Eos | Stop::Budget => break,
        }
    }

    let trajectory = String::from_utf8_lossy(&trajectory_bytes).into_owned();
    let answer = extract_boxed(&trajectory);
    let stats = sh.stats.borrow();
    let ledger = sh.ledger.borrow();

    println!(
        "[npr] done: {} parallel blocks, {} branches, depth {}, {} fallbacks, \
         {} tokens generated ({} charged of {} budget), {:?} elapsed",
        stats.parallel_blocks,
        stats.branches_total,
        stats.max_depth_seen,
        stats.sequential_fallbacks,
        stats.tokens_generated,
        ledger.charged,
        ledger.budget,
        start.elapsed()
    );

    Ok(inferlet::serde_json::json!({
        "answer": answer,
        "join_mode": input.join_mode,
        "parallel_blocks": stats.parallel_blocks,
        "branches_total": stats.branches_total,
        "max_depth": stats.max_depth_seen,
        "sequential_fallbacks": stats.sequential_fallbacks,
        "tokens_generated": stats.tokens_generated,
        "tokens_charged": ledger.charged,
        "token_budget": ledger.budget,
        "elapsed_ms": start.elapsed().as_millis() as u64,
        "trajectory": trajectory,
    })
    .to_string())
}
