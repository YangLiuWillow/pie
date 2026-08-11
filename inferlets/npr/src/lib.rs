//! Native Parallel Reasoner (NPR, arXiv 2512.07461) — Phase 1.
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
//! - Join: adopt branch 1's context, append the sibling branches' tokens
//!   and a `<takeaway>\n` header, and continue decoding (Map-Process-Reduce
//!   rounds repeat until the final answer; nested blocks recurse).
//!
//! Phase-1 simplification (documented in docs/npr-inferlet-design.md): the
//! join is *textual* — sibling step tokens are re-filled causally with
//! sequential positions and full causal attention, instead of NPR's
//! overlapped positions + KV stitching. Phase 2 replaces this with the
//! faithful refill join (explicit position IDs + per-token BRLE masks).
//! The per-`<step>` repetition penalty (1.02) is also deferred.
//!
//! Tag detection is byte-level over the vocab table, so it works whether
//! the tags are single special tokens (the NPR checkpoint) or ordinary
//! BPE splits (any stock model, e.g. for smoke tests).

use futures::future;
use inferlet::model::{Model, Tokenizer};
use inferlet::sample::Sampler;
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
    /// Max nesting depth of parallel blocks (NPR: 5).
    #[serde(default = "default_max_depth")]
    max_depth: usize,
    /// Minimum remaining budget to allow a fork (NPR: 1024).
    #[serde(default = "default_min_fork_budget")]
    min_fork_budget: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// Test hook: text injected as if already generated, before decoding
    /// starts. A primer ending in `</guideline>` triggers the fork path
    /// immediately — lets the fork/join machinery be exercised with a
    /// stock model that does not emit the NPR format on its own.
    #[serde(default)]
    primer: Option<String>,
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
    ledger: RefCell<Ledger>,
    stats: RefCell<Stats>,
}

impl Shared {
    fn new(model: &Model, input: &Input) -> Self {
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
        Shared {
            tokenizer,
            token_bytes,
            special_ids,
            chat_stops: chat::stop_tokens(model),
            temperature: input.temperature,
            top_p: input.top_p,
            max_plans: input.max_plans,
            max_depth: input.max_depth,
            min_fork_budget: input.min_fork_budget,
            ledger: RefCell::new(Ledger {
                budget: input.max_new_tokens,
                charged: 0,
            }),
            stats: RefCell::new(Stats::default()),
        }
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
            // Find the earliest tag occurrence in `rest`.
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
    ctx: &mut Context,
    degree: usize,
    watch_step: bool,
) -> Result<Segment> {
    let mut tokens = Vec::new();
    let mut bytes = Vec::new();
    let mut generator = ctx.generate(sh.sampler()).stop(&sh.chat_stops);
    loop {
        if sh.ledger.borrow().remaining() < degree {
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
    ctx: Context,
    /// Everything this branch added beyond the fork prefix: the
    /// `\n<step>\n{i}:` header, generated tokens, and any nested joins.
    tokens: Vec<u32>,
    bytes: Vec<u8>,
    /// The branch ended on end-of-turn or budget instead of `</step>`.
    terminal: bool,
}

enum ForkOutcome {
    /// Branches ran and were joined into `ctx`.
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
    ctx: &'a mut Context,
    segment_text: &'a str,
    depth: usize,
) -> Pin<Box<dyn Future<Output = Result<ForkOutcome>> + 'a>> {
    Box::pin(async move {
        let plans = parse_plans(segment_text);
        let affordable = sh.ledger.borrow().remaining() >= sh.min_fork_budget;
        if plans.is_empty() || plans.len() > sh.max_plans || depth > sh.max_depth || !affordable {
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
        println!(
            "[npr] depth {depth}: forking {degree} branches: {:?}",
            plans
        );

        // Fork one child per plan. The children inherit the parent's
        // pending buffer (which holds the `</guideline>` token), so every
        // branch sequence includes the closed guideline block.
        let mut branch_futures = Vec::with_capacity(degree);
        for label in plans {
            let child = ctx.fork()?;
            branch_futures.push(run_branch(sh.clone(), child, label, depth, degree));
        }
        let results = future::join_all(branch_futures).await;
        let mut branches = Vec::with_capacity(degree);
        for r in results {
            branches.push(r?);
        }
        println!(
            "[npr] depth {depth}: all {degree} branches done ({} tokens total), joining",
            branches.iter().map(|b| b.tokens.len()).sum::<usize>()
        );

        // Textual join (phase 1): adopt branch 1's context — it already
        // holds prefix + its own step — then append the sibling branches'
        // tokens in plan order and cue the Reduce stage with `<takeaway>\n`
        // (NPR `merge_zombie_batch_to_run`). Phase 2 replaces this append
        // with the position/mask-faithful refill join.
        let mut iter = branches.into_iter();
        let first = iter.next().expect("at least one branch");
        let mut merged = first.ctx;
        let mut tokens = first.tokens;
        let mut bytes = first.bytes;
        let mut terminal = first.terminal;
        for sibling in iter {
            merged.append(&sibling.tokens);
            tokens.extend_from_slice(&sibling.tokens);
            bytes.extend_from_slice(&sibling.bytes);
            terminal |= sibling.terminal;
            sibling.ctx.destroy();
        }
        let takeaway = sh.encode_with_tags("<takeaway>\n");
        merged.append(&takeaway);
        tokens.extend_from_slice(&takeaway);
        bytes.extend_from_slice(b"<takeaway>\n");

        // The pre-fork parent context is superseded by the merged branch.
        let parent = std::mem::replace(ctx, merged);
        parent.destroy();

        Ok(ForkOutcome::Forked {
            tokens,
            bytes,
            terminal,
        })
    })
}

/// Decode one `<step>` branch to its `</step>`, recursing into nested
/// parallel blocks. Charges `degree` budget units per generated token.
fn run_branch(
    sh: Rc<Shared>,
    mut ctx: Context,
    label: String,
    depth: usize,
    degree: usize,
) -> Pin<Box<dyn Future<Output = Result<BranchResult>>>> {
    Box::pin(async move {
        // NPR forks each child with `prefix + "\n<step>\n{i}:"`.
        let header = format!("\n<step>\n{label}:");
        let header_tokens = sh.encode_with_tags(&header);
        ctx.append(&header_tokens);
        let mut tokens = header_tokens;
        let mut bytes = header.into_bytes();
        loop {
            let segment = decode_segment(&sh, &mut ctx, degree, true).await?;
            tokens.extend_from_slice(&segment.tokens);
            bytes.extend_from_slice(&segment.bytes);
            match segment.stop {
                Stop::StepEnd => {
                    println!(
                        "[npr] branch {label}: </step> after {} tokens",
                        tokens.len()
                    );
                    return Ok(BranchResult {
                        ctx,
                        tokens,
                        bytes,
                        terminal: false,
                    });
                }
                Stop::GuidelineEnd => {
                    // Nested parallel block inside this step.
                    let text = String::from_utf8_lossy(&segment.bytes).into_owned();
                    match try_fork(&sh, &mut ctx, &text, depth + 1).await? {
                        ForkOutcome::Forked {
                            tokens: join_tokens,
                            bytes: join_bytes,
                            terminal,
                        } => {
                            tokens.extend_from_slice(&join_tokens);
                            bytes.extend_from_slice(&join_bytes);
                            if terminal {
                                return Ok(BranchResult {
                                    ctx,
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
                        ctx,
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
// Entry point
// =============================================================================

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let start = Instant::now();

    let models = runtime::models();
    let model_name = models.first().ok_or("no models available")?;
    let model = Model::load(model_name)?;

    let sh = Rc::new(Shared::new(&model, &input));

    // Prompt per NPR `evals/evaluate.py`: user turn = question + "\n\n" +
    // format instruction, then the generation cue. The user turn is flushed
    // (committing the shared prefix); the cue tokens stay buffered so the
    // first Generator step has input to sample from.
    let mut ctx = Context::new(&model)?;
    ctx.user(&format!("{}\n\n{}", input.question, INSTRUCTION));
    ctx.flush().await?;
    ctx.cue();

    let mut trajectory_bytes: Vec<u8> = Vec::new();

    if let Some(primer) = input.primer.as_deref() {
        let tokens = sh.encode_with_tags(primer);
        ctx.append(&tokens);
        trajectory_bytes.extend_from_slice(primer.as_bytes());
        if primer.trim_end().ends_with("</guideline>") {
            match try_fork(&sh, &mut ctx, primer, 1).await? {
                ForkOutcome::Forked { bytes, .. } => {
                    trajectory_bytes.extend_from_slice(&bytes);
                }
                ForkOutcome::Sequential => {}
            }
        }
    }

    loop {
        let segment = decode_segment(&sh, &mut ctx, 1, false).await?;
        trajectory_bytes.extend_from_slice(&segment.bytes);
        match segment.stop {
            Stop::GuidelineEnd => {
                let text = String::from_utf8_lossy(&segment.bytes).into_owned();
                match try_fork(&sh, &mut ctx, &text, 1).await? {
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
