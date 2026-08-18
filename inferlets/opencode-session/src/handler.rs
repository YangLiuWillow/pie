//! Turn orchestration for the long-lived session daemon: resume, generate,
//! emit, retain.
//!
//! ## What a "session" is here, and what it is not
//!
//! The process is per opencode session; the retained STATE is per conversation
//! *branch*, keyed by content address. That distinction is the whole design:
//!
//! - keying on a client-supplied session id would make a mid-conversation edit
//!   (opencode rewrites history: compaction, truncation, message editing) resume
//!   a prefix that no longer matches what the client thinks it sent, and serve
//!   fluent output from the wrong context with nothing to catch it;
//! - keying on the content address makes every such divergence a MISS, which
//!   costs one full rebuild and is always correct.
//!
//! So the failure mode of this design is "slower", never "wrong". That is why
//! it is the first thing built, ahead of the delta wire in
//! `docs/opencode-integration.md` §2 — a shadow-diff mismatch fails the other
//! way.
//!
//! ## The retention protocol
//!
//! A turn retains its RENDERED HISTORY and nothing else, addressed by the canon
//! of this request's own messages. The next request arrives with our assistant
//! turn echoed back plus whatever it appended; [`split_retain_point`] lands
//! immediately before that assistant message, so the prefix it hashes is exactly
//! the message list we retained under, and the suffix it prefills is the
//! assistant turn plus the new material.
//!
//! Retaining *through* the generated turn would reuse more, and it is what the
//! qwen-code port does — but the generated span sits behind a generation cue
//! that history replay does not reproduce, so the retained prefix drifts from
//! any re-render of the same conversation. See the `engine` module docs for the
//! measurement.
//!
//! One consequence worth naming: because the address hashes only what the client
//! sent, the response/save unification hazard is gone. Under the other scheme a
//! trailing-whitespace trim in the response path silently broke every later
//! resume, and nothing pointed at it.

use crate::engine::{self, GenConfig, SessionState};
use crate::turn::TurnState;
use crate::wire::{Envelope, Sink, recover_req_id, send_error};

use inferlet::ptir::attention::prelude::{kv_page_size, kv_pool_status};
use inferlet::{chat, model, runtime, tools};
use pie_openai_serving::error::{INVALID_REQUEST_ERROR, SERVER_ERROR, parse_request};
use pie_openai_serving::streaming::ChunkMeta;
use pie_openai_serving::{
    ChatCompletionRequest, RenderOp, TEMPLATE_MARKER, cut_leading_reasoning, plan_render,
    prefix_addresses, sanitize_messages,
};

/// Defaults when the client sends none. opencode always sends `max_tokens` but
/// omits `temperature`/`top_p` for unknown model ids (AUDIT §1a).
const DEFAULT_MAX_TOKENS: usize = 4096;
const DEFAULT_TEMPERATURE: f32 = 0.6;
const DEFAULT_TOP_P: f32 = 0.95;

/// Default retained-KV budget, in tokens, when the launcher supplies none.
///
/// ## Why a token budget and not a branch count
///
/// A branch count is not a resource bound. Eight branches of a 200-token chat
/// is nothing; eight branches of an 8k-token coding session is 64k tokens
/// against a pool of `total_pages * kv_page_size` — 16,384 in the profile this
/// integration ships. Over-committing does not degrade: the engine kills the
/// process, which takes the gateway WebSocket down with it, and every retained
/// branch dies at once. Measured here on the sixth turn of a 6-turn bench, and
/// it is the same failure class OpenHands hit ("the cache did not evict, it
/// blocked forever" — their first A/B wedged at 4 of 13 instances).
///
/// ## Why the guest has to guess
///
/// Nothing on the `pie:inferlet` surface reports the KV pool size. The guest is
/// asked to manage residency without being told the budget: `kv_page_size()`
/// and `max_embed_length()` exist, a pool capacity does not. So the launcher
/// passes one in (`{"retain_tokens": N}`), because it is the only party that
/// has read the driver config, and this default is what a 16k pool can hold
/// while still leaving room for the live turn's scratch.
pub const DEFAULT_RETAIN_TOKENS: u32 = 8192;

fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// One retained conversation branch.
impl Retained {
    /// Tokens of KV this branch pins.
    fn tip(&self) -> u32 {
        self.boundaries.last().map(|(l, _)| *l).unwrap_or(0)
    }

    /// Structural self-check, run before this branch is trusted for a resume.
    ///
    /// ## What this catches that the address cannot
    ///
    /// The content address already makes a false hit near-impossible: a hit
    /// means `hash(model ‖ template ‖ full[..L])` matched AND the recorded
    /// length equals `L`, so the tokens agree by construction. What it cannot
    /// see is the *state* drifting away from what we recorded about it — a
    /// working set that came back smaller than its bookkeeping claims, a
    /// boundary list that stopped being ordered, a tip that disagrees with the
    /// last entry. Those are engine/driver-side failures, and the address is
    /// blind to all of them because it only describes tokens.
    ///
    /// This is the cheap half of OpenHands' `kv_verify`, which asserts
    /// `ctx.seq_len() == full_tokens.len()` and **errors rather than silently
    /// rebuilding**. Their `seq_len()` has no equivalent here — a `WorkingSet`
    /// reports pages, not tokens — so the check is page-granular: the pages
    /// must be able to *hold* the tokens claimed. Weaker, but it still fails
    /// loudly on a truncated or mis-sized set instead of serving from it.
    ///
    /// Returns `Err(reason)` when the branch must not be resumed.
    fn verify(&self, page_t: u32) -> Result<(), String> {
        if self.boundaries.is_empty() {
            return Err("no boundaries".to_string());
        }
        let mut prev = 0u32;
        for (len, _) in &self.boundaries {
            if *len <= prev {
                return Err(format!("boundaries not strictly ascending at {len} (prev {prev})"));
            }
            prev = *len;
        }
        let tip = self.tip();
        let pages = self.state.ws.page_len();
        let capacity = pages.saturating_mul(page_t);
        if capacity < tip {
            return Err(format!(
                "working set holds {pages} pages ({capacity} tokens) but the branch \
                 claims {tip} — state is smaller than its bookkeeping"
            ));
        }
        Ok(())
    }
}

/// The cache budget in tokens: the operator's `retain_tokens`, clamped so one
/// conversation's history cannot claim a shared pool, less what the turn about
/// to run needs.
///
/// Pure, because this is the arithmetic the whole retention rule reduces to and
/// it is the part that can be wrong without crashing — a budget too generous
/// wedges the pool, one too tight collapses reuse, and both look like an
/// ordinary slow run.
fn cache_budget_tokens(retain_tokens: u32, pool_tokens: u32, reserve_tokens: u32) -> u32 {
    const MAX_POOL_SHARE_PERCENT: u32 = 60;
    let mut budget = retain_tokens;
    if pool_tokens > 0 {
        budget = budget.min(pool_tokens / 100 * MAX_POOL_SHARE_PERCENT);
    }
    budget.saturating_sub(reserve_tokens)
}

/// Does a newly retained tip make `older` redundant?
///
/// True when the tip's render passes THROUGH `older`'s own tip -- same token
/// length, same address -- because then every boundary `older` could be resumed
/// at is also a boundary of the new state, which covers strictly more.
///
/// An `older` with no boundaries can never be matched by any resume, so it is
/// redundant by a different route: it is pages nothing can reach.
///
/// Address equality is the whole test. Two branches at the same length with
/// different addresses are different conversations that must both survive --
/// that is the case this must never get wrong, since dropping the wrong one
/// costs a full re-prefill of someone else's history.
fn branch_is_superseded(tip: &[(u32, String)], older: &[(u32, String)]) -> bool {
    let Some(older_tip) = older.last() else {
        return true;
    };
    tip.iter().any(|b| b == older_tip)
}

pub struct Retained {
    state: SessionState,
    /// Ascending `(token length, address of the render's first `len` tokens)`
    /// for every boundary this state's KV is valid at; the last is its current
    /// length. Keeping the whole list — rather than only the tip — is what lets
    /// a retry, a truncation or a branch re-hit a still-valid EARLIER boundary
    /// instead of rebuilding. One `SessionState` serves them all, because the
    /// tokens below any boundary are untouched by later extension.
    boundaries: Vec<(u32, String)>,
}

pub struct Daemon {
    /// Retained branches, least-recently-used first.
    sessions: Vec<Retained>,
    /// Total tokens of retained KV this process will hold. See
    /// [`DEFAULT_RETAIN_TOKENS`].
    retain_tokens: u32,
    /// Per-instance id fragment, shared by every turn this process serves.
    uniq: String,
    /// Turn counter. Tool-call ids must be unique across the whole SESSION, not
    /// just the turn — see `TurnState::uniq`.
    counter: u64,
    /// Model special tokens, for sanitization. Fetched once: it is a host call
    /// over the whole vocabulary's special set, and it cannot change under a
    /// running process.
    specials: Vec<String>,
    /// Chat stop set, plus the turn-START marker.
    stop_ids: Vec<u32>,
    /// The submission pipeline, PROCESS-lived rather than turn-lived. Parked
    /// between turns (`Pipeline::park` — leaves the frame wait-set without
    /// running down `submit_deadline`, and a parked lane is never killed by
    /// the silence timeout), so the next turn's first submit rejoins instead
    /// of paying a cold lane join per turn — a cost that lands in TTFT, the
    /// column the A-vs-B comparison is quoted on.
    ///
    /// `None` after a failed turn: a pipeline failure is sticky (every later
    /// submit inherits the reason), so the failed pipeline is dropped — drop
    /// closes it, and already-submitted fires still drain — and the next turn
    /// starts a fresh one.
    pipe: Option<inferlet::ptir::Pipeline>,
}

impl Daemon {
    pub fn new(retain_tokens: u32) -> Self {
        let uniq: String = runtime::instance_id()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .take(8)
            .collect();

        // Stop set: the model's chat stop tokens, plus the turn-START marker.
        // At t=0 a looping model starts simulating the next turn instead of
        // stopping, and a leaked `<|im_start|>` round-trips through message
        // content into fake turn boundaries on the next request.
        let mut stop_ids = chat::stop_tokens();
        let turn_start = model::encode("<|im_start|>");
        if turn_start.len() == 1 && !stop_ids.contains(&turn_start[0]) {
            stop_ids.push(turn_start[0]);
        }

        let specials: Vec<String> = model::special_tokens()
            .into_iter()
            .filter_map(|t| String::from_utf8(t.bytes).ok())
            .filter(|s| !s.is_empty())
            .collect();

        Self {
            sessions: Vec::new(),
            retain_tokens,
            uniq: if uniq.is_empty() { "local".to_string() } else { uniq },
            counter: 0,
            specials,
            stop_ids,
            pipe: None,
        }
    }

    pub async fn handle(&mut self, raw: &str) {
        let env: Envelope = match serde_json::from_str(raw) {
            Ok(e) => e,
            Err(e) => {
                send_error(
                    &recover_req_id(raw),
                    400,
                    INVALID_REQUEST_ERROR,
                    &format!("invalid envelope: {e}"),
                );
                return;
            }
        };
        let req_id = env.req_id.clone();
        self.run_turn(&req_id, env).await;
    }

    async fn run_turn(&mut self, req_id: &str, env: Envelope) {
        // Phase clock. A turn's wall time is the only number the client can see,
        // and on the 7.4k-token replay it is ~1.5 s of which the model accounts
        // for ~0.25 s. Naming where the rest goes needs the guest to say so:
        // nothing outside it can distinguish rendering from addressing from
        // waiting on a fire. Cheap enough to leave in — five `Instant::now()`
        // against a turn that runs for a second.
        let t_entry = std::time::Instant::now();
        let sink = Sink::new(req_id);


        let body = match serde_json::to_vec(&env.body) {
            Ok(b) => b,
            Err(e) => {
                send_error(req_id, 400, INVALID_REQUEST_ERROR, &format!("invalid body: {e}"));
                return;
            }
        };
        let mut req: ChatCompletionRequest = match parse_request(&body) {
            Ok(r) => r,
            Err(msg) => {
                send_error(req_id, 400, INVALID_REQUEST_ERROR, &msg);
                return;
            }
        };

        let streaming = req.stream;
        let include_usage = req.include_usage();
        let temperature = req
            .temperature
            .filter(|t| t.is_finite())
            .unwrap_or(DEFAULT_TEMPERATURE);
        let top_p = req
            .top_p
            .filter(|p| p.is_finite() && *p > 0.0 && *p <= 1.0)
            .unwrap_or(DEFAULT_TOP_P);
        let max_tokens = req.effective_max_tokens(DEFAULT_MAX_TOKENS);
        let has_tools = !req.tools.is_empty();
        let stop_strings = req.stop_strings();
        // Sanitize before rendering, so the address hashes what actually
        // replays. Note the tool schemas need no separate hashing any more:
        // `plan_render` folds them into the system turn, so they are part of
        // the token stream and therefore part of the address for free. Same for
        // the thinking channel — it is whatever the cue rendered.
        sanitize_messages(&mut req.messages, &self.specials);

        // ── Render the FULL history, once ────────────────────────────────
        // Not just the resume suffix. Rendering everything costs host template
        // calls (~7 ms on a 7k-token history upstream) and buys three things a
        // suffix-only render cannot:
        //
        //   * the address can hash TOKEN IDS instead of messages, so a chat
        //     template or tokenizer change misses cleanly instead of handing
        //     the model KV rendered by the old template;
        //   * every render-unit boundary becomes a candidate resume point, so a
        //     retry or a truncation re-hits an earlier one;
        //   * the hit can be GATED on the boundary length, which is what makes
        //     "a false hit is impossible" true rather than hoped for.
        //
        // The saving that matters was never the render; it was the prefill.
        let ops = match plan_render(&req) {
            Ok(o) => o,
            Err(e) => {
                send_error(req_id, 400, INVALID_REQUEST_ERROR, &e.to_string());
                return;
            }
        };
        // Split the plan at its trailing `Cue`. The history is what gets
        // retained; the cue is generation scaffolding and must NOT be —
        // replaying this turn as history renders no cue at all, so a retained
        // cue is tokens no re-render will ever produce (engine docs).
        let (delta_ops, cue_ops) = match ops.split_last() {
            Some((RenderOp::Cue(_), head)) => (head, &ops[ops.len() - 1..]),
            _ => {
                eprintln!("[opencode-session] render plan did not end with a cue");
                send_error(req_id, 500, SERVER_ERROR, "internal error while rendering the prompt");
                return;
            }
        };
        let rendered = render_history(delta_ops).and_then(|(f, b)| render_ops(cue_ops).map(|c| (f, b, c)));
        let (full, bounds, cue) = match rendered {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[opencode-session] render fault: {e}");
                send_error(req_id, 500, SERVER_ERROR, "internal error while rendering the prompt");
                return;
            }
        };
        if full.is_empty() {
            send_error(
                req_id,
                400,
                INVALID_REQUEST_ERROR,
                "`messages` rendered to an empty prompt",
            );
            return;
        }

        // Address every boundary in one streaming pass. The LAST entry is this
        // turn's own render — the address we will retain under. The rest are
        // resume candidates.
        let t_render = t_entry.elapsed();
        let model_id = model::name();
        let addressed = prefix_addresses(&model_id, TEMPLATE_MARKER, &full, &bounds);
        let t_address = t_entry.elapsed();
        let _save_address = match addressed.last() {
            Some((_, a)) => a.clone(),
            None => {
                eprintln!("[opencode-session] render produced no boundaries");
                send_error(req_id, 500, SERVER_ERROR, "internal error while rendering the prompt");
                return;
            }
        };

        // ── Resume: scan candidates longest-first ────────────────────────
        // Every boundary is tried, longest-first. OpenHands caps this at 8
        // because each attempt is a `Context::open` — an engine round trip. Ours
        // is a string compare against an in-memory list, so ~30 boundaries × ≤8
        // branches is a few hundred comparisons and the cap would only cost
        // hits: a cross-conversation match lands ~2.1k tokens into a ~7.3k-token
        // render, which is nowhere near the longest boundary and a cap of 8
        // would never reach it.
        //
        // The final boundary (this turn's whole render) is deliberately NOT a
        // candidate: resuming there would leave an empty delta, and re-rendering
        // one trailing unit is cheaper than a special case for it.
        const MAX_ATTEMPTS: usize = usize::MAX;
        let page_t = kv_page_size();
        let mut resume_at: Option<(usize, u32)> = None; // (branch index, boundary)
        for (b, addr) in addressed.iter().rev().skip(1).take(MAX_ATTEMPTS) {
            if let Some(i) = self
                .sessions
                .iter()
                .position(|r| r.boundaries.iter().any(|(l, a)| l == b && a == addr))
            {
                // kv_verify: a matching address is necessary but not
                // sufficient. Refuse the resume and drop the branch rather than
                // serving from a state that disagrees with its own bookkeeping;
                // a rebuild is slow, a wrong prefix is silent.
                if let Err(why) = self.sessions[i].verify(page_t) {
                    eprint!(
                        "{}",
                        format!(
                            "[opencode-session] kv_verify REFUSED a resume at boundary {b}: \
                             {why}; dropping the branch and rebuilding\n"
                        )
                    );
                    self.sessions.remove(i);
                    continue;
                }
                resume_at = Some((i, *b));
                break;
            }
        }

        let t_resume = t_entry.elapsed();

        // ── Settle the resume decision BEFORE anything is derived from it ──
        //
        // Everything this turn slices divides `full` at the resume boundary:
        // the delta it prefills, the tokens it reports, the demand it hands
        // retention, and — through `render_len` — the length the retained
        // state claims. So the one decision that can send the turn cold has
        // to come first.
        //
        // The first version of this guard ran AFTER `delta` was sliced and
        // only zeroed `cached_tokens`. Its "cold rebuild" then prefilled the
        // stale suffix `full[b..]` at positions 0..n, and `retain_turn` filed
        // that state under the full render's prefix addresses filtered to
        // `<= n` — a branch whose addresses hash true history prefixes while
        // its state holds the wrong tokens, and whose recorded tip is the
        // last stride cut below `n` while its fold stands at `n`. The next
        // resume matched that tip, passed this guard (b == tip), and fired
        // below the fold:
        //
        //   instance 1003 launch failed: paged continuation:
        //   recurrent slot 1 is at position 733, this fire starts at 512
        //
        // 733 = 8297 - 7564: the length of the suffix the "cold" rebuild
        // actually prefilled. All three post-guard poison epochs were this
        // (806/768, 789/768, 733/512 — each fold at the mislabeled length,
        // each fire at the last 256-stride cut under it). Had the driver not
        // checked, the resume would have served output computed from tokens
        // the addresses disclaim — on a pure-attention model, silently.
        //
        // The branch is TAKEN out of the list: the turn extends that very
        // working set (nothing forks — see `engine::Resume`), so leaving a
        // second handle behind would advertise a prefix whose tail is about
        // to be overwritten. `retain_turn` puts the extended state back; a
        // failed turn drops it, which is a clean miss.
        //
        // Resuming at boundary `b` invalidates every boundary ABOVE it —
        // those tokens are about to be rewritten by this turn's delta — so
        // the surviving list is truncated to `<= b` before the state is
        // reused.
        let mut owned_parent: Option<Retained> = None;
        let mut resume_boundary: u32 = 0;
        if let Some((idx, b)) = resume_at {
            let r = self.sessions.remove(idx);
            let tip = r.boundaries.last().map(|(l, _)| *l).unwrap_or(0);
            // REWINDING IS A KV-ONLY POWER. Truncating the boundary list
            // moves this branch's KV back to `b`, because paged KV is
            // reversibly discardable. The FOLD is not: it stands at `tip`
            // and nothing in the guest can move it left. The engine's own
            // interface says so as a correctness gate -- "evicting KV does
            // not undo the fold" (`forward-hybrid.wit`).
            //
            // So on a recurrent model an earlier-boundary resume is REFUSED
            // and the turn rebuilds cold — genuinely cold, from the top of
            // `full`. A cold rebuild costs one prefill; the alternative is a
            // poisoned instance or, without the driver's check, a fluent
            // answer computed from a state that never existed.
            if b < tip && !r.state.rs.is_empty() {
                eprint!(
                    "{}",
                    format!(
                        "[opencode-session] refusing an earlier-boundary resume \
                         ({b} < {tip}) on a recurrent model: the fold cannot rewind; \
                         rebuilding cold\n"
                    )
                );
                // Dropping `r` releases the branch, KV and fold together.
            } else {
                let mut r = r;
                r.boundaries.retain(|(l, _)| *l <= b);
                owned_parent = Some(r);
                resume_boundary = b;
            }
        }

        let cached_tokens = resume_boundary;
        let delta = &full[cached_tokens as usize..];
        let prompt_tokens = cached_tokens + (delta.len() + cue.len()) as u32;

        self.counter += 1;
        let uniq = format!("{}{}", self.uniq, self.counter);
        let meta = ChunkMeta {
            id: format!("chatcmpl-{uniq}"),
            model: model::name(),
            created: now_unix_secs(),
        };
        let mut state = TurnState::new(
            meta,
            sink.clone(),
            streaming,
            uniq,
            self.stop_ids.clone(),
            stop_strings,
            has_tools,
            &pie_openai_serving::types::tool_schema_envelopes(&req.tools),
        );

        // ── Commit the stream. From here every failure must be shaped as a
        // well-formed turn (finish_reason "length"), never an error — a dead
        // stream burns opencode's unbounded 5xx retries.
        if streaming {
            let role = state.meta.role_chunk();
            state.emit(&role);
        }

        let stop_ids = self.stop_ids.clone();

        // The keepalive chunk is byte-identical every time, so build it once
        // and hand the prefill callback its own `Sink` clone. Both engine
        // callbacks would otherwise need the turn state — one to read it, one
        // to mutate it — which is two live borrows of the same value.
        let keepalive = state.meta.keepalive();
        let ka_sink = sink.clone();


        // Make room for what THIS TURN will actually write. Every branch still
        // in `self.sessions` is now a non-keeper: the one being resumed was
        // just removed into `owned_parent`, so nothing here can evict the
        // prefix this turn is about to build on.
        //
        // Demand, not occupancy. A percentage watermark cannot answer the
        // question that matters — both failures this replaces were pools at a
        // perfectly ordinary 87%. One turn needed 512 pages and 256 were free,
        // so it starved while the watermark said "fine"; the other spared
        // nothing at 85% and evicted the branch the next turn needed, so reuse
        // went to zero and every turn re-prefilled ~48k. What decides both is
        // whether the pages this turn needs are available, which is knowable
        // here and nowhere earlier: it is the delta, not the whole prompt,
        // because the resumed prefix is already resident.
        self.enforce_retention(delta.len() as u32 + cue.len() as u32 + max_tokens as u32);

        // The process-lived pipeline: taken for the turn, parked and put back
        // on success, dropped on any failure (see the field doc).
        let pipe = self.pipe.take().unwrap_or_default();
        let run = {
            let resume = match owned_parent.take() {
                Some(r) => engine::Resume::InPlace(r.state, cached_tokens),
                None => engine::Resume::Cold,
            };
            let st = &mut state;
            engine::generate(
                &pipe,
                resume,
                &delta,
                &cue,
                &GenConfig { temperature, top_p, max_tokens },
                &stop_ids,
                || {
                    if streaming {
                        ka_sink.chunk(&keepalive);
                    }
                },
                |t| st.on_token(t),
            )
            .await
        };
        let t_gen = t_entry.elapsed();

        // A setup failure (no prefill, unsupported pass kind, a fork the driver
        // refused) still has to answer as a turn, not as a fault.
        let run = match run {
            Ok(g) => g,
            Err(e) => {
                // `pipe` drops here: a setup failure may have poisoned it, and
                // a pipeline failure is sticky. The next turn starts fresh.
                eprintln!("[opencode-session] generation setup failed: {e}");
                self.degrade(&sink, &mut state, streaming, include_usage);
                return;
            }
        };
        if let Some(e) = &run.gen_error {
            eprintln!("[opencode-session] generation degraded to length-finish: {e}");
        } else {
            // A clean turn keeps its pipeline. Park FIRST: a lane that goes
            // silent without parking is eventually terminated by the silence
            // timeout, and the gap to the next user turn is unbounded.
            pipe.park();
            self.pipe = Some(pipe);
        }

        state.flush_tail();
        let raw_text = model::decode(&state.generated).unwrap_or_default();
        state.salvage(has_tools, &raw_text);

        let finish_reason = state.finish_reason(run.hit_max, run.gen_error.is_some());

        if !streaming {
            // Non-streaming is the acceptance/debug path: nothing is on the
            // wire yet, so trailing whitespace and a leading un-opened think
            // block are still removable. This is the LAST point at which that
            // is true, and it must happen before `final_content` so the
            // response and the retention address hash the same string.
            state.visible_text = cut_leading_reasoning(state.visible_text.trim_end()).to_string();
        }
        let content = state.final_content(&raw_text);

        if streaming && !state.emitted_visible && state.calls.is_empty() {
            let chunk = state.meta.content_delta(&content);
            state.emit(&chunk);
        }

        // ── Retain BEFORE signalling completion ──────────────────────────
        // opencode fires its follow-up the moment the stream closes, and the
        // retained state must already exist for that resume to hit. Retention
        // failures are non-fatal: the next turn pays a full rebuild.
        // `cached` vs `prefill` is the whole measurement in one line: on a hit
        // the second number is the delta and the first is what we did not pay
        // for. `gen` is the engine's own accepted count — if it ever disagrees
        // with the decoded token count, the turn state machine and the decode
        // loop have diverged.
        let accepted = run.accepted;
        let retained = self.retain_turn(run, addressed);
        // Built as ONE string and printed with a single placeholder. A
        // multi-fragment `eprintln!` is split by the runtime's stderr capture
        // into one client message PER FRAGMENT, so the interpolated form
        // arrives at the shim as a dozen separate lines and the measurement is
        // unreadable exactly where it matters.
        let t_retain = t_entry.elapsed();
        // Pool occupancy AFTER retaining, because that is the number the
        // gateway's admission gate reads on the NEXT turn: it rejects when the
        // worker's `kv_pressure_bucket` reaches 240/255 (94.1%), and a
        // rejected turn used to take the WebSocket -- and every retained
        // branch -- down with it. Printed per turn so the occupancy at the
        // rejection is measured rather than inferred.
        let (pool_avail, pool_total) = kv_pool_status();
        let pool_pct = if pool_total == 0 {
            0
        } else {
            (pool_total - pool_avail) as u64 * 100 / pool_total as u64
        };
        let line = format!(
            "[opencode-session] turn {} cached={cached_tokens} delta={} cue={} gen={accepted} \
             pool={}/{} ({}%) {retained}\n",
            self.counter,
            delta.len(),
            cue.len(),
            pool_total - pool_avail,
            pool_total,
            pool_pct
        );
        eprint!("{}", line);
        // Cumulative from turn entry, so each field is "everything up to here"
        // and the differences are the phases. Printed cumulatively rather than
        // as durations because a missing phase then shows up as a flat segment
        // instead of silently vanishing into its neighbour.
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let phases = format!(
            "[opencode-session] phases_ms turn={} render={:.1} address={:.1} \
             resume={:.1} generate={:.1} retain={:.1} (cumulative from turn entry)\n",
            self.counter,
            ms(t_render),
            ms(t_address),
            ms(t_resume),
            ms(t_gen),
            ms(t_retain)
        );
        eprint!("{}", phases);

        let n_generated = state.generated.len() as u32;
        if streaming {
            let finish = state.meta.finish_chunk(finish_reason);
            state.emit(&finish);
            if include_usage {
                let usage = state.meta.usage_chunk(prompt_tokens, n_generated, cached_tokens);
                state.emit(&usage);
            }
            sink.done();
        } else {
            let calls: Vec<(String, String, String)> = state
                .calls
                .iter()
                .map(|c| (c.id.clone(), c.name.clone(), c.arguments.clone()))
                .collect();
            sink.response(&state.meta.completion_response(
                &content,
                &calls,
                finish_reason,
                prompt_tokens,
                n_generated,
                cached_tokens,
            ));
            sink.done();
        }
    }

    /// THE retention rule. One invariant, one place, two evaluation points.
    ///
    /// This replaces three mechanisms that grew one per failure — a demand
    /// check before the turn, an admissibility check after it, and a
    /// supersession sweep — none of which shared a budget, and all of which
    /// substituted for `retain_tokens`: a field declared, documented, plumbed
    /// from the shim, printed at startup, and enforced NOWHERE. The budget was
    /// always the design; nothing implemented it, so each symptom got a patch.
    ///
    /// The model is ownership, and it has exactly two categories:
    ///
    /// * The TIP is the working set. The next turn resumes it, so it is kept
    ///   even when it alone exceeds the budget — evicting it is the reuse
    ///   collapse measured at 85%: `cached=0 delta=52888` for the rest of a
    ///   run, a permanent full-price prefill traded for a transient one.
    /// * Every OLDER branch is cache. Bounded by `retain_tokens`, in the
    ///   tokens that field is denominated in, and dropped oldest-first.
    ///
    /// The one exception is the wedge, and it is the only case that may take
    /// the tip: if what this guest holds leaves the pool with no room for
    /// another turn, the engine refuses the next one and this guest — the only
    /// thing that could free those pages — never runs again to free them.
    /// Measured: an idle server refused a four-token request permanently.
    /// Losing the tip costs one cold prefill; keeping it costs the server.
    ///
    /// Deliberately NOT expressed against the gateway's admission constant.
    /// The guest used to carry `240/255` in two places, which made a gateway
    /// retune a silent guest bug. "Leave room for another turn" is a statement
    /// this guest can make on its own terms.
    fn enforce_retention(&mut self, reserve_tokens: u32) -> usize {
        // The engine's admission headroom, MIRRORED — and named as a mirror
        // rather than dressed up as a guest-local rule.
        //
        // The gateway refuses a turn once the worker's pressure bucket reaches
        // 240 of 255, i.e. above ~94% used, so staying under that needs ~6%
        // free and no more. I first wrote this as "leave room for another
        // turn" at 10% to avoid the cross-layer constant, and that is strictly
        // worse: it fires at 90% used, a full turn before the engine would
        // have refused anything, and on a pool sized to exactly one
        // conversation it takes the tip every turn. Measured immediately —
        // `turn 25 pool=0`, then `cached=0 delta=57296`, the same reuse
        // collapse an 85% threshold caused earlier.
        //
        // So the guest does need this number. Pretending otherwise cost a turn
        // of reuse per turn. The honest fix is for the host to publish its
        // admission headroom (a WIT addition); until then this mirrors it,
        // with the pointer, so a gateway retune has one place to update.
        const KEEP_FREE_PERCENT: u32 = 6;

        let page = kv_page_size().max(1);
        let (_, pool_total) = kv_pool_status();
        let pool_tokens = pool_total.saturating_mul(page);

        // The cache budget: the operator's figure, clamped so no single
        // conversation's history can claim a shared pool.
        let budget =
            cache_budget_tokens(self.retain_tokens, pool_tokens, reserve_tokens);

        let mut dropped = 0usize;
        // Older branches are cache: drop oldest-first until they fit. The tip
        // is excluded from both the cost and the eviction — `sessions.len()
        // > 1` — because it is the working set, not cache.
        loop {
            let cache_tokens: u32 = self
                .sessions
                .iter()
                .rev()
                .skip(1)
                .map(|r| r.tip())
                .sum();
            if self.sessions.len() <= 1 || cache_tokens <= budget {
                break;
            }
            let gone = self.sessions.remove(0);
            dropped += 1;
            eprint!(
                "{}",
                format!(
                    "[opencode-session] dropped a cached branch ({} tokens); \
                     cache {cache_tokens} > budget {budget}, {} left\n",
                    gone.tip(),
                    self.sessions.len()
                )
            );
        }

        // The wedge exception. Only now, and only if the pool itself has no
        // room left for another turn.
        let keep_free = pool_total / 100 * KEEP_FREE_PERCENT;
        while !self.sessions.is_empty() {
            let (available, total) = kv_pool_status();
            if total == 0 || available >= keep_free.max(1) {
                break;
            }
            let gone = self.sessions.remove(0);
            dropped += 1;
            eprint!(
                "{}",
                format!(
                    "[opencode-session] dropped a branch ({} tokens) to leave the \
                     pool room for another turn ({available}/{total} free); the \
                     next turn re-prefills\n",
                    gone.tip()
                )
            );
        }
        dropped
    }

    fn retain_turn(
        &mut self,
        run: engine::Generation,
        addressed: Vec<(u32, String)>,
    ) -> String {
        if let Some(e) = run.gen_error {
            // Dropping `run` releases the state.
            return format!("not retained (generation error: {e})");
        }
        let total = run.total_len;
        // The state must hold the ENTIRE render it is about to be addressed
        // as. `addressed` hashes prefixes of this request's full render; a
        // state whose length differs holds some OTHER token sequence, and
        // filing it under these addresses is cache poison — a later resume
        // would reuse KV whose address disclaims its content. That is not
        // hypothetical: the stale-delta refusal bug built a state from
        // `full[b..]` at positions 0..n and retained it here, and the only
        // reason it surfaced as a driver refusal rather than silent wrong
        // output is that the model had a fold to disagree with the resume
        // position. Refusing to retain converts the whole class into one
        // loud line and a re-prefill.
        let render_len = addressed.last().map(|(l, _)| *l).unwrap_or(0);
        if render_len != total {
            return format!(
                "not retained (state holds {total} tokens of a {render_len}-token \
                 render — addressing it would poison the cache)"
            );
        }
        // Store EVERY boundary of this render, not just the tip. The retained
        // KV covers [0, total), so every boundary at or below it is a valid
        // resume point — and the interior ones are the only thing another
        // conversation can ever match, since two sessions agree on a shared
        // head and diverge before the end. Storing the tip alone silently
        // reduces the cache to "resume your own last turn", which is what it
        // did until this was measured: two conversations with byte-identical
        // 7.2k-token heads shared nothing.
        let boundaries: Vec<(u32, String)> =
            addressed.into_iter().filter(|(l, _)| *l <= total).collect();
        let address = boundaries
            .last()
            .map(|(_, a)| a.clone())
            .unwrap_or_default();
        let candidate = Retained { state: run.state, boundaries };
        // kv_verify on the way IN as well as on the way out: a state that fails
        // its own invariants must never enter the map, or the next turn pays
        // the lookup only to refuse it.
        if let Err(why) = candidate.verify(kv_page_size()) {
            return format!("not retained (kv_verify: {why})");
        }
        // Reclaim what this tip supersedes, AFTER it is retained, so a refusal
        // between the two leaves the chain resumable rather than empty.
        //
        // Ported from test-time-bench's decoder, which keeps at most two
        // snapshots per conversation line -- first boundary + tip -- and calls
        // eviction "closing the arena-page leak that kept it opt-in". Its leak
        // is ours: a branch nothing will ever resume still holds its pages, and
        // pages held past their usefulness are what saturates the pool and gets
        // the turn refused.
        //
        // Adapted, not copied, because the two designs hold state differently.
        // TTB launches per turn and saves a separate context per boundary, so
        // it must pay for the first one to keep a shared head resumable. Here a
        // boundary is a LABEL into one live state, so the tip already offers
        // every earlier cut of its own render at no extra pages -- the "first"
        // half of first-and-tip is free, and buying it as a second branch would
        // spend a whole working set on something already covered.
        //
        // Superseded means: this render passed THROUGH that branch's tip, so
        // everything it could serve, the tip serves. The resume path already
        // removes the branch it resumed; this catches the ones a MISS leaves
        // behind, which nothing else drops until the pool is under pressure.
        let tip_boundaries = candidate.boundaries.clone();
        let before = self.sessions.len();
        // The tip is held OUT of the scan rather than skipped by index: it
        // supersedes itself, and a self-drop would retain nothing at all.
        self.sessions
            .retain(|branch| !branch_is_superseded(&tip_boundaries, &branch.boundaries));
        let dropped = before - self.sessions.len();
        self.sessions.push(candidate);

        let trimmed = self.enforce_retention(0);

        let mut line = format!("retained {} (len {total}", &address[..16]);
        if dropped > 0 {
            line.push_str(&format!(", superseded {dropped}"));
        }
        if trimmed > 0 {
            line.push_str(&format!(", trimmed {trimmed}"));
        }
        line.push(')');
        line
    }


    /// A turn that failed before producing anything still has to look like a
    /// turn. Non-whitespace content, `finish_reason:"length"`, clean close.
    fn degrade(&self, sink: &Sink, state: &mut TurnState, streaming: bool, include_usage: bool) {
        const DEGRADED: &str = "The server could not complete this turn.";
        if streaming {
            let c = state.meta.content_delta(DEGRADED);
            state.emit(&c);
            let f = state.meta.finish_chunk("length");
            state.emit(&f);
            if include_usage {
                let u = state.meta.usage_chunk(0, 0, 0);
                state.emit(&u);
            }
            sink.done();
        } else {
            sink.response(&state.meta.completion_response(DEGRADED, &[], "length", 0, 0, 0));
            sink.done();
        }
    }
}

/// Map the engine-free render plan 1:1 onto the WIT template surface.
///
/// `Cue` renders through `chat::cue(thinking)` — the mode travels on the op,
/// no-think channel, matching the token-exact parity verdict against HF
/// `enable_thinking=False`, and matching Strategy A so the A/B compares servers
/// rather than renderers.
/// Render the history ops, recording the token length after each one.
///
/// The boundaries are the safe resume points: every one lands on a render-unit
/// edge (the system+tools block, a user/assistant turn, a merged tool batch),
/// so a candidate prefix is a literal token prefix of this render **by
/// construction** — a bad split cannot plant a wrong suffix. The final boundary
/// is the whole render.
/// Extra resume candidates emitted *inside* a long render op, every this many
/// tokens.
///
/// Render-unit boundaries alone are too coarse to share anything across
/// conversations, and the reason is a position accident. opencode's head is ONE
/// op — `EquipAfterSystem` folds the system turn and the tool schemas together —
/// so its only boundary is the whole ~7.3k-token head, which never matches
/// between two sessions because the system prompt carries a per-session
/// environment block:
///
/// ```text
/// Working directory: /private/tmp/.../oc-project
/// Is directory a git repo: no
/// Today's date: Tue Aug 11 2026
/// ```
///
/// That block sits 8,695 chars into a 9,648-char system message, *ahead* of
/// 21,188 chars of tool schemas that ARE byte-identical across sessions. So a
/// prefix cache can reach only the 28% before it — and only if a boundary
/// exists there, which per-op boundaries do not provide.
///
/// Striding fixes it without knowing anything about opencode: wherever two
/// token streams happen to agree, some stride boundary lands inside the
/// agreement and the scan finds it. It is the same idea as vLLM's block-level
/// APC, at coarser granularity because each boundary costs a digest snapshot
/// and a scan entry rather than a page-table entry.
const BOUNDARY_STRIDE: u32 = 256;

fn render_history(ops: &[RenderOp]) -> Result<(Vec<u32>, Vec<u32>), String> {
    let mut out = Vec::new();
    let mut bounds: Vec<u32> = Vec::with_capacity(ops.len() * 4);
    for op in ops {
        let before = out.len() as u32;
        render_one(op, &mut out)?;
        let after = out.len() as u32;
        // Interior stride points, then the op's own edge. Ascending and
        // duplicate-free, which `prefix_addresses` and the boundary list both
        // require.
        let mut at = before.next_multiple_of(BOUNDARY_STRIDE);
        while at < after {
            if at > before {
                bounds.push(at);
            }
            at += BOUNDARY_STRIDE;
        }
        if bounds.last() != Some(&after) {
            bounds.push(after);
        }
    }
    Ok((out, bounds))
}

fn render_ops(ops: &[RenderOp]) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for op in ops {
        render_one(op, &mut out)?;
    }
    Ok(out)
}

/// Map one engine-free render op onto the WIT template surface.
///
/// `Cue` renders through `chat::cue(thinking)` — the mode travels on the op,
/// no-think channel, matching the token-exact parity verdict against HF
/// `enable_thinking=False`, and matching Strategy A so the A/B compares servers
/// rather than renderers.
fn render_one(op: &RenderOp, out: &mut Vec<u32>) -> Result<(), String> {
    match op {
        RenderOp::EquipAfterSystem { system, tools: schemas } => {
            out.extend(tools::equip_after_system(system.as_deref(), schemas)?);
        }
        RenderOp::User(t) => out.extend(chat::user(t)),
        RenderOp::Assistant(t, p) => {
            out.extend(chat::assistant(t, p.after_query, p.is_last))
        }
        RenderOp::AssistantWithToolCalls { content, calls, pos } => {
            let wit_calls: Vec<tools::ToolCall> = calls
                .iter()
                .map(|(name, args)| tools::ToolCall {
                    name: name.clone(),
                    arguments_json: args.clone(),
                })
                .collect();
            out.extend(chat::assistant_call(
                content.as_deref(),
                &wit_calls,
                pos.after_query,
                pos.is_last,
            ));
        }
        RenderOp::AnswerBatch(batch) => out.extend(tools::answer_batch(batch)),
        RenderOp::Cue(thinking) => out.extend(chat::cue(*thinking)),
    }
    Ok(())
}

#[cfg(test)]
mod retention_tests {
    use super::branch_is_superseded;

    fn b(pairs: &[(u32, &str)]) -> Vec<(u32, String)> {
        pairs.iter().map(|(l, a)| (*l, (*a).to_string())).collect()
    }

    /// The ordinary case: this turn resumed the branch and extended it, so the
    /// new state passes through the old tip and covers strictly more.
    #[test]
    fn a_tip_supersedes_the_branch_it_grew_from() {
        let tip = b(&[(100, "aa"), (200, "bb"), (300, "cc")]);
        assert!(branch_is_superseded(&tip, &b(&[(100, "aa"), (200, "bb")])));
    }

    /// THE case that must never be got wrong. Two conversations reach the same
    /// token length with different content; dropping the other one costs a full
    /// re-prefill of a history this branch cannot serve.
    #[test]
    fn same_length_different_address_is_a_different_conversation() {
        let tip = b(&[(100, "aa"), (200, "bb")]);
        assert!(!branch_is_superseded(&tip, &b(&[(100, "aa"), (200, "ZZ")])));
    }

    /// A branch that shares a head but diverged is still live: its own tip is
    /// not on this render, so this state cannot serve it.
    #[test]
    fn a_branch_that_diverged_survives_its_shared_head() {
        let tip = b(&[(100, "aa"), (200, "bb"), (300, "cc")]);
        assert!(!branch_is_superseded(&tip, &b(&[(100, "aa"), (250, "dd")])));
    }

    /// A longer branch is not covered by a shorter tip -- supersession is not
    /// symmetric, and treating it as such would drop the better branch.
    #[test]
    fn a_shorter_tip_does_not_supersede_a_longer_branch() {
        let tip = b(&[(100, "aa"), (200, "bb")]);
        assert!(!branch_is_superseded(&tip, &b(&[(100, "aa"), (400, "ee")])));
    }

    /// Boundaries are what a resume matches on, so a branch with none can never
    /// be reached again. It is pages nothing can claim.
    #[test]
    fn an_unreachable_branch_is_reclaimed() {
        assert!(branch_is_superseded(&b(&[(100, "aa")]), &[]));
    }

    /// An interior match is enough: the resume path truncates a branch's
    /// boundaries to the cut it resumed at, so an older branch's tip commonly
    /// sits in the middle of the new render rather than at its end.
    #[test]
    fn an_interior_boundary_match_counts() {
        let tip = b(&[(100, "aa"), (200, "bb"), (300, "cc")]);
        assert!(branch_is_superseded(&tip, &b(&[(100, "aa")])));
    }
}

#[cfg(test)]
mod budget_tests {
    use super::cache_budget_tokens;

    const POOL: u32 = 65_536; // 2048 pages x 32

    /// The operator's figure is honoured when it is the smaller bound. This is
    /// the field that was declared, documented, plumbed and enforced NOWHERE
    /// until the retention rule was consolidated.
    #[test]
    fn the_operators_budget_is_what_binds_when_it_is_smaller() {
        assert_eq!(cache_budget_tokens(32_768, POOL, 0), 32_768);
    }

    /// ...and cannot claim a shared pool no matter what it is set to. A budget
    /// larger than the pool is how the guest wedges the engine: it holds
    /// everything, the gate refuses the next turn, and only a turn could free
    /// the pages.
    #[test]
    fn no_setting_lets_one_conversation_claim_the_pool() {
        assert!(cache_budget_tokens(u32::MAX, POOL, 0) < POOL);
        assert_eq!(cache_budget_tokens(1_000_000, POOL, 0), POOL / 100 * 60);
    }

    /// The turn about to run is charged against the same budget, in the same
    /// unit, rather than through a second mechanism with its own headroom.
    #[test]
    fn the_running_turn_is_charged_to_the_same_budget() {
        let idle = cache_budget_tokens(32_768, POOL, 0);
        assert_eq!(cache_budget_tokens(32_768, POOL, 8_192), idle - 8_192);
    }

    /// A turn bigger than the budget leaves nothing for cache and must say so
    /// as zero, not wrap. Saturating here is the difference between "evict
    /// everything" and "evict nothing, then wedge".
    #[test]
    fn a_turn_larger_than_the_budget_saturates_to_zero() {
        assert_eq!(cache_budget_tokens(32_768, POOL, 100_000), 0);
    }

    /// An unknown pool (status unavailable) must not silently clamp the budget
    /// to zero and evict the whole cache.
    #[test]
    fn an_unknown_pool_falls_back_to_the_operators_figure() {
        assert_eq!(cache_budget_tokens(32_768, 0, 0), 32_768);
    }
}
