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
//! Three join modes (`join_mode` input):
//!
//! - `"adopt"` (phase 3, faithful + no recomputation): identical layout and
//!   metadata to `"refill"`, but the sibling KV is grafted by a device-side
//!   row copy (`Context::adopt_kv`) instead of being recomputed — the host
//!   derives tokens/positions from the sibling's lineage and synthesizes
//!   the same hole masks the refill would carry. Only the sibling's staged
//!   buffer tail (typically its `</step>` token) is refilled. Falls back to
//!   refill per sibling on any host refusal.
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
    /// Selftest hook: pad the prompt with N filler sentences so the prompt
    /// fill exceeds a reduced `max_forward_tokens` and exercises the
    /// runtime's chunked prefill. Cross-boot comparison of the reported
    /// reference distribution (`ref_ids`/`ref_probs`) between a chunked and
    /// an unchunked config is the chunk-equivalence test; within-boot TVs
    /// cannot see prompt-KV differences (the prefix is common-mode).
    #[serde(default)]
    selftest_prompt_pad: usize,
    /// Test hook: cap each branch at this many generated tokens (treated
    /// as a terminal stop). Lets smoke tests with stock models — which
    /// never emit `</step>` — leave budget for the post-join stages.
    #[serde(default)]
    max_step_tokens: Option<usize>,
    /// Reuse the prompt prefill across runs of the same question via a
    /// content-addressed context snapshot (RatioThink-style save/open).
    /// A hit forks the saved post-prompt context instead of re-prefilling;
    /// a miss builds it and saves. Snapshots are cut at the render
    /// boundary — the generation cue is never captured. Off by default.
    #[serde(default)]
    prompt_cache: bool,
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
    /// Logical position right after the prompt — the engine's
    /// `init_input_len` in position space. Set once after the prompt fill.
    prompt_end_pos: u32,
}

impl Ledger {
    fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.charged)
    }

    /// The reference engine's ×degree charge is per-request and TRANSIENT:
    /// a branch's own tokens count ×degree only while its block is open;
    /// the merge re-bases the block's content into the origin at ×1 and the
    /// degree resets (`schedule_batch.py:693-697` — `origin_input_ids`
    /// counts ×1, only `output_ids` of the live request multiply). Our
    /// cumulative ledger must therefore refund the multiplier at each join,
    /// or a degree-K trajectory is charged ~K× the reference for identical
    /// content — which starved 78% of runs in the first sweep.
    fn refund_join(&mut self, block_tokens: usize, degree: usize) {
        self.charged = self
            .charged
            .saturating_sub(block_tokens.saturating_mul(degree.saturating_sub(1)));
    }

    /// The engine's primary budget check is POSITIONAL
    /// (`right_most_pos - init_input_len >= max_new_tokens - 128`,
    /// `schedule_batch.py:687`): siblings overlap positions, so this meters
    /// the longest path through the parallel structure, not the token sum.
    fn position_exhausted(&self, next_pos: u32) -> bool {
        next_pos.saturating_sub(self.prompt_end_pos) as usize + 128 >= self.budget
    }

    /// The engine's no-fork guard is also positional
    /// (`schedule_batch.py:1586`: within 1024 of the positional cap).
    fn position_remaining(&self, next_pos: u32) -> usize {
        self.budget
            .saturating_sub(next_pos.saturating_sub(self.prompt_end_pos) as usize)
    }
}

#[derive(Default)]
struct Stats {
    parallel_blocks: usize,
    branches_total: usize,
    max_depth_seen: usize,
    sequential_fallbacks: usize,
    tokens_generated: usize,
    /// Sibling tokens grafted via adopt_kv instead of refill recomputation.
    tokens_adopted: usize,
    /// Siblings that fell back to refill after an adopt_kv refusal.
    adopt_fallbacks: usize,
    /// Wall-clock spent in join phases (sibling refill/adopt + takeaway
    /// cue fill), across all parallel blocks. The number the join
    /// mechanism (refill vs adopt) actually moves.
    join_ms: u64,
}

#[derive(Clone, Copy, PartialEq)]
enum JoinMode {
    /// Phase 3: graft sibling KV via `Context::adopt_kv` (device row copy,
    /// no recomputation); falls back to refill per sibling on any host
    /// refusal. Join layout and metadata are identical to refill.
    Adopt,
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
            "adopt" => JoinMode::Adopt,
            "refill" => JoinMode::Refill,
            "textual" => JoinMode::Textual,
            other => return Err(format!("unknown join_mode: {other}")),
        };
        // Diagnostic: report how each structural tag resolved. A tag whose
        // special-token bytes are missing from the tokenizer's table will
        // misrender in trajectories (and would break literal-tag parsing).
        for tag in ["<guideline>", "</guideline>", "<plan>", "</plan>", "<step>", "</step>", "<takeaway>", "</takeaway>"] {
            match special_ids.get(tag.as_bytes()) {
                Some(id) => {
                    let render = String::from_utf8_lossy(
                        token_bytes.get(*id as usize).map(|v| v.as_slice()).unwrap_or(&[]),
                    );
                    if render != tag {
                        println!("[npr] tag {tag} = special id {id}, but renders as {render:?}");
                    }
                }
                None => println!("[npr] tag {tag}: no special token (plain BPE)"),
            }
        }
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
                prompt_end_pos: 0,
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
    // Engine-faithful ×degree check is PER-REQUEST (`schedule_batch.py:
    // 693-697`): a branch tests `inherited(×1) + own_output × degree`, not
    // the sum of all siblings' multiplied outputs against a shared ledger —
    // the shared-sum version starves any long parallel block mid-flight at
    // roughly budget/degree total tokens (the silent ~10k truncation).
    inherited: usize,
    own_before: usize,
) -> Result<Segment> {
    let mut tokens = Vec::new();
    let mut bytes = Vec::new();
    let delta = p.delta;
    // Logical position of the next token; generation is causal, so it
    // advances by exactly one per generated token. Tracked locally because
    // the generator holds the &mut borrow of the context.
    let mut cur_pos = p.next_pos();
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
        // Engine-faithful finish checks (`schedule_batch.py::check_finished`):
        // primary is positional (longest path through the parallel structure);
        // the ×degree charge is the secondary, transient check.
        let degree_charge = inherited + (own_before + tokens.len()) * degree;
        if sh.ledger.borrow().position_exhausted(cur_pos)
            || degree_charge + 128 >= sh.ledger.borrow().budget
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
        cur_pos += 1;
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
        // Engine no-fork guard is positional (`schedule_batch.py:1586`):
        // within min_fork_budget (their 1024) of the positional cap →
        // degrade to sequential.
        let affordable = sh.ledger.borrow().position_remaining(p.next_pos()) >= sh.min_fork_budget;
        // Refill mode forks only at depth 1: exact nested refill needs
        // per-token position/visibility records (see module docs).
        let depth_cap = match sh.join_mode {
            JoinMode::Refill | JoinMode::Adopt => 1,
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

        let charged_before_block = sh.ledger.borrow().charged;
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
        // Merge re-bases the block's content to ×1 in the reference engine;
        // refund the transient (degree-1)× multiplier now that the block is
        // closed. Every charge inside a flat block is a multiple of `degree`
        // (nested textual blocks make this an approximation after their own
        // inner refunds; refill/adopt blocks are depth-1 and exact).
        {
            let mut ledger = sh.ledger.borrow_mut();
            let block_charged = ledger.charged.saturating_sub(charged_before_block);
            ledger.refund_join(block_charged / degree.max(1), degree);
        }
        println!(
            "[npr] depth {depth}: all {degree} branches done ({} tokens total), joining ({})",
            branches.iter().map(|b| b.tokens.len()).sum::<usize>(),
            match sh.join_mode {
                JoinMode::Adopt => "adopt",
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

        let join_start = Instant::now();
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
            JoinMode::Refill | JoinMode::Adopt => {
                // Faithful join (NPR Algorithm 2). Refill realizes it as
                // recomputation; adopt grafts the sibling's KV rows via
                // `Context::adopt_kv` (identical layout and metadata, no
                // recomputation) and falls back to refill per sibling.
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
                // 2. Bring each sibling's tokens into base at positions
                //    restarting at p_fork, visible only to [0, fork_slots)
                //    plus its own tokens — the KV each branch computed in
                //    its own context (bit-identical when adopted,
                //    kernel-noise-equal when refilled).
                for sibling in iter {
                    let n = sibling.tokens.len();
                    max_extent = max_extent.max(n as u32);
                    let mut adopted = false;
                    if sh.join_mode == JoinMode::Adopt {
                        match adopt_sibling(&mut base, &sibling, fork_slots, p_fork).await? {
                            Some(n_adopted) => {
                                sh.stats.borrow_mut().tokens_adopted += n_adopted;
                                adopted = true;
                            }
                            None => {
                                sh.stats.borrow_mut().adopt_fallbacks += 1;
                            }
                        }
                    }
                    if !adopted {
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
                    }
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

        sh.stats.borrow_mut().join_ms += join_start.elapsed().as_millis() as u64;

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

/// Graft one sibling's KV into `base` via `Context::adopt_kv`, then refill
/// only its pending buffer tail (typically the `</step>` token that is
/// staged as the next generator input but not yet in KV).
///
/// Returns `Ok(Some(n))` on success, `Ok(None)` when the adopt was refused
/// **before anything landed in base** — the caller falls back to a full
/// refill — and `Err` only for failures after the graft, where a fallback
/// refill would duplicate the adopted tokens.
async fn adopt_sibling(
    base: &mut PCtx,
    sibling: &BranchResult,
    fork_slots: u32,
    p_fork: u32,
) -> Result<Option<usize>> {
    let sib = &sibling.p;
    let pend = sib.ctx.buffer().to_vec();
    let n_adopt = sib.ctx.seq_len().saturating_sub(fork_slots);
    // The KV-resident tokens plus the staged tail must reproduce the branch
    // transcript exactly — anything else means our slot accounting is off
    // and the copy would graft the wrong range.
    if n_adopt as usize + pend.len() != sibling.tokens.len() {
        println!(
            "[npr] adopt: kv accounting mismatch ({} resident + {} pending != {} transcript)",
            n_adopt,
            pend.len(),
            sibling.tokens.len()
        );
        return Ok(None);
    }
    let dst_start = match base.ctx.adopt_kv(&sib.ctx, fork_slots, n_adopt) {
        Ok(d) => d,
        Err(e) => {
            println!("[npr] adopt_kv refused: {e}");
            return Ok(None);
        }
    };
    if !pend.is_empty() {
        // Tail continues the sibling's stream: positions after its adopted
        // tokens, visible to [0, fork_slots) plus the whole sibling range.
        let start = p_fork + n_adopt;
        let positions: Vec<u32> = (start..start + pend.len() as u32).collect();
        let rows: Vec<Vec<u32>> = (0..pend.len())
            .map(|i| {
                vec![
                    0,
                    fork_slots,
                    dst_start - fork_slots,
                    n_adopt + i as u32 + 1,
                ]
            })
            .collect();
        refill(base, &pend, &positions, &rows).await?;
    }
    Ok(Some(n_adopt as usize))
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
        // The engine's per-request charge counts inherited generated tokens
        // at ×1 (they live in `origin_input_ids` post-rebase). Snapshot the
        // ledger at branch start: post-refund `charged` is exactly that
        // ×1 basis.
        let inherited = sh.ledger.borrow().charged;
        let mut own = 0usize;
        let mut tokens = header_tokens;
        let mut bytes = header.into_bytes();
        loop {
            let segment =
                decode_segment(&sh, &mut p, degree, true, sh.max_step_tokens, inherited, own)
                    .await?;
            own += segment.tokens.len();
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
///
/// A segment spans exactly one guideline block (it starts right after the
/// previous `<takeaway>`/turn start and ends at `</guideline>`), so when
/// the literal `<guideline>` open tag is absent from the rendered bytes —
/// the tokenizer's byte table can misrender added special tokens, and a
/// model may occasionally skip the open tag — scan the whole segment.
fn parse_plans(segment: &str) -> Vec<String> {
    let start = segment.rfind("<guideline>").unwrap_or(0);
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

/// Adopt-join oracle: fill `sib_content` causally in a fork of `base` (the
/// "branch"), adopt its KV into a fresh fork of `base` via `adopt_kv`,
/// refill only `tail` with hole rows, and probe the next-token distribution
/// at the tail. With `sib_content = B[..n-1]`, `tail = B[n-1..]` this must
/// reproduce the straight-line distribution after B. With wrong
/// `sib_content` of the same length (the negative control) it must NOT:
/// the discriminating content lives solely in the copied region, so a
/// silently failed or stale copy cannot produce a passing result.
async fn adopt_and_probe(
    base: &Context,
    fork_slots: u32,
    sib_content: &[u32],
    tail: &[u32],
    p_fork: u32,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut sib = base.fork()?;
    let mut pass = sib.forward();
    pass.input(sib_content);
    pass.execute().await?;

    let mut dst = base.fork()?;
    let n_adopt = sib_content.len() as u32;
    let dst_start = dst.adopt_kv(&sib, fork_slots, n_adopt)?;

    let positions: Vec<u32> = (0..tail.len() as u32).map(|i| p_fork + n_adopt + i).collect();
    let rows: Vec<Vec<u32>> = (0..tail.len())
        .map(|i| vec![0, fork_slots, dst_start - fork_slots, n_adopt + i as u32 + 1])
        .collect();
    let mut pass = dst.forward();
    pass.input(tail);
    pass.positions(&positions);
    pass.attention_mask(&rows);
    let h = pass.probe(
        tail.len() as u32 - 1,
        Distribution {
            temperature: 1.0,
            k: 10,
        },
    );
    let out = pass.execute().await?;
    let (ids, probs) = out.distribution(h).ok_or("selftest: adopt probe missing")?;
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
async fn selftest(model: &Model, sh: &Shared, pad: usize) -> Result<String> {
    let mut base = Context::new(model)?;
    let mut prompt =
        String::from("Solve: what is the domain of f(x) = 1/log(2 - log(x - 2))? Reason briefly.");
    for i in 0..pad {
        prompt.push_str(&format!(
            " Contextual note {i}: the composition of logarithms constrains the domain \
             through each layer, and every constraint must hold simultaneously."
        ));
    }
    base.user(&prompt);
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

    // Per-SHAPE warmup (correction from a parallel session: the first-sight
    // transient is keyed by GEMM dimensions, NOT per-boot — a throwaway of
    // one length warms only that length, and unseen shapes deviate by up to
    // ~1.9 nats on first sight). Burn every distinct arm-fill length once
    // on a scratch context so all measured arms below run shape-warm. The
    // prompt fill's own first-sight lands in the shared prefix KV, which is
    // common-mode across arms and cancels in every comparison.
    {
        let mut lens: Vec<usize> =
            vec![nb, toks_a.len(), toks_a_short.len(), nb.saturating_sub(1), 1];
        lens.sort_unstable();
        lens.dedup();
        for m in lens {
            if m == 0 {
                continue;
            }
            let mut warm = base.fork()?;
            let fill: Vec<u32> = toks_b.iter().cycle().take(m).copied().collect();
            let mut pass = warm.forward();
            pass.input(&fill);
            pass.execute().await?;
            warm.destroy();
        }
        println!("[npr selftest] shape warmup done");
    }

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
    println!("[npr selftest] e_ref2 (noise floor)...");
    // The identical computation a second time: TV(ref, ref2) is the pure
    // run-to-run kernel-noise floor. Every other arm's TV should be read
    // against it — an arm near the floor differs by noise; an arm well
    // above it differs systematically (layout, masks, or a bug).
    let e_ref2 = fill_and_probe(&base, None, &toks_b, None, None).await?;
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
    println!("[npr selftest] e_adopt...");
    // Adopt all but B's last token from a causal branch fill, refill only
    // the last token — the real join's shape with a 1-token tail.
    let e_adopt =
        adopt_and_probe(&base, fork_slots, &toks_b[..nb - 1], &toks_b[nb - 1..], p_fork).await?;
    println!("[npr selftest] e_adopt_ctl...");
    // Same geometry, wrong content: the reversed B-prefix. Only the copied
    // region differs, so this must diverge from the reference.
    let wrong: Vec<u32> = toks_b[..nb - 1].iter().rev().copied().collect();
    let e_adopt_ctl =
        adopt_and_probe(&base, fork_slots, &wrong, &toks_b[nb - 1..], p_fork).await?;
    println!("[npr selftest] e_ctl...");
    let e_ctl = fill_and_probe(&base, Some(&toks_a), &toks_b, None, None).await?;

    let tvs = [
        ("noise_floor", tv(&e_ref, &e_ref2)),
        ("positions", tv(&e_ref, &e_pos)),
        ("mask", tv(&e_ref, &e_mask)),
        ("mask_4run", tv(&e_ref, &e_4run)),
        ("hole_natural", tv(&r_shift, &m_hole_nat)),
        ("refill_short", tv(&e_ref, &e_short)),
        ("refill", tv(&e_ref, &e_refill)),
        ("adopt", tv(&e_ref, &e_adopt)),
        ("adopt_control", tv(&e_ref, &e_adopt_ctl)),
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

    let pass = tvs[..8].iter().all(|(_, v)| *v < 0.05)
        && tvs[8].1 > 0.05
        && tvs[9].1 > 0.05;
    Ok(inferlet::serde_json::json!({
        "selftest_pass": pass,
        "prompt_pad": pad,
        "ref_ids": e_ref.0,
        "ref_probs": e_ref.1,
        "tv": tvs.iter().map(|(n, v)| (n.to_string(), *v)).collect::<HashMap<_, _>>(),
    })
    .to_string())
}

// =============================================================================
// Prompt prefix cache
// =============================================================================

/// FNV-1a over a byte stream; enough for a content-addressed snapshot name
/// (a collision requires two *distinct* questions the same user actually
/// runs against the same model and inferlet version).
fn fnv64(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Content-addressed snapshot name for the post-prompt context. Keyed on
/// the model (the chat template lives there), the exact question text (the
/// instruction is compiled in), and this crate's version (template or
/// prompt-construction drift must miss, never false-hit). 0xFF separators
/// cannot appear inside UTF-8 strings, preventing field-shift collisions.
fn prompt_cache_name(model_name: &str, question: &str) -> String {
    let mut h = fnv64(0xcbf2_9ce4_8422_2325, model_name.as_bytes());
    h = fnv64(h, &[0xFF]);
    h = fnv64(h, question.as_bytes());
    h = fnv64(h, &[0xFF]);
    h = fnv64(h, env!("CARGO_PKG_VERSION").as_bytes());
    format!("npr-prompt/{h:016x}")
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
        return selftest(&model, &sh, input.selftest_prompt_pad).await;
    }

    // Prompt per NPR `evals/evaluate.py`: user turn = question + "\n\n" +
    // format instruction, then the generation cue. The user turn is flushed
    // (committing the shared prefix); the cue tokens stay buffered so the
    // first Generator step has input to sample from.
    //
    // With `prompt_cache` the flushed post-prompt context is snapshotted
    // under a content-addressed name and reused by later runs of the same
    // question (avg@k repeats): a hit forks the snapshot and skips the
    // prompt prefill entirely. The snapshot is cut BEFORE the cue — the cue
    // is appended fresh on both paths, so the reused KV is exactly the
    // canonical prompt rendering. Saves race benignly across concurrent
    // repeats: the name is content-addressed, so AlreadyExists means an
    // identical snapshot is already in place.
    let mut prompt_cache_state = "off";
    let mut p = if input.prompt_cache {
        let name = prompt_cache_name(model_name, &input.question);
        match Context::open(&model, &name) {
            Ok(ctx) if ctx.seq_len() > 0 => {
                prompt_cache_state = "hit";
                PCtx::new(ctx)
            }
            _ => {
                let mut ctx = Context::new(&model)?;
                ctx.user(&format!("{}\n\n{}", input.question, INSTRUCTION));
                ctx.flush().await?;
                match ctx.save(&name) {
                    Ok(()) => prompt_cache_state = "miss_saved",
                    Err(e) if e.starts_with("Snapshot name already exists:") => {
                        prompt_cache_state = "miss_exists"
                    }
                    Err(e) => {
                        println!("[npr] prompt cache save failed (continuing): {e}");
                        prompt_cache_state = "miss_save_failed";
                    }
                }
                PCtx::new(ctx)
            }
        }
    } else {
        let mut ctx = Context::new(&model)?;
        ctx.user(&format!("{}\n\n{}", input.question, INSTRUCTION));
        ctx.flush().await?;
        PCtx::new(ctx)
    };
    p.ctx.cue();
    // Anchor the positional budget: everything up to and including the cue
    // is prompt (the engine's `init_input_len`); generation is metered from
    // here in position space (longest path, since siblings overlap).
    sh.ledger.borrow_mut().prompt_end_pos = p.next_pos();

    let mut trajectory_bytes: Vec<u8> = Vec::new();
    // Why the top-level loop ended — recorded in the output JSON so a
    // truncated trajectory names its own cause instead of being diagnosed
    // from token counts (a silent ~10k stop cost an evening of guessing).
    let mut stop_reason = "unknown";

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
        let segment = {
            let inherited = sh.ledger.borrow().charged;
            decode_segment(&sh, &mut p, 1, false, None, inherited, 0).await?
        };
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
                            // A branch hit EOS/budget inside the block; the
                            // join completed but the run ends here.
                            stop_reason = "branch_terminal";
                            break;
                        }
                    }
                    ForkOutcome::Sequential => {}
                }
            }
            // `</step>` is not watched at top level; Eos/Budget end the run.
            Stop::StepEnd => {
                stop_reason = "step_end";
                break;
            }
            Stop::Eos => {
                stop_reason = "eos";
                break;
            }
            Stop::Budget => {
                stop_reason = "budget";
                break;
            }
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
        "tokens_adopted": stats.tokens_adopted,
        "adopt_fallbacks": stats.adopt_fallbacks,
        "join_ms": stats.join_ms,
        "prompt_cache": prompt_cache_state,
        "stop_reason": stop_reason,
        "tokens_charged": ledger.charged,
        "token_budget": ledger.budget,
        "elapsed_ms": start.elapsed().as_millis() as u64,
        "trajectory": trajectory,
    })
    .to_string())
}
