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

use inferlet::ptir::attention::prelude::kv_page_size;
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

/// Hard cap on retained branches, as a backstop. The REAL bound is
/// [`Daemon::retain_tokens`] — see below.
const MAX_RETAINED: usize = 8;

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
            Some((RenderOp::Cue, head)) => (head, &ops[ops.len() - 1..]),
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
        let model_id = model::name();
        let addressed = prefix_addresses(&model_id, TEMPLATE_MARKER, &full, &bounds);
        let save_address = match addressed.last() {
            Some((_, a)) => a.clone(),
            None => {
                eprintln!("[opencode-session] render produced no boundaries");
                send_error(req_id, 500, SERVER_ERROR, "internal error while rendering the prompt");
                return;
            }
        };

        // ── Resume: scan candidates longest-first ────────────────────────
        // Capped, because each attempt is a scan and the useful hit is nearly
        // always the longest. Upstream measured 1.97 attempts per call.
        //
        // The final boundary (this turn's whole render) is deliberately NOT a
        // candidate: resuming there would leave an empty delta, and re-rendering
        // one trailing unit is cheaper than a special case for it.
        const MAX_ATTEMPTS: usize = 8;
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

        let cached_tokens = resume_at.map(|(_, b)| b).unwrap_or(0);
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
        // Boundaries carried over from the branch we resumed (empty on a cold
        // turn); this turn's own boundary is appended on retain.
        let mut parent_boundaries: Vec<(u32, String)> = Vec::new();

        // The branch is TAKEN out of the list: the turn extends that very
        // working set (nothing forks — see `engine::Resume`), so leaving a
        // second handle behind would advertise a prefix whose tail is about to
        // be overwritten. `retain_turn` puts the extended state back; a failed
        // turn drops it, which is a clean miss.
        //
        // Resuming at boundary `b` invalidates every boundary ABOVE it — those
        // tokens are about to be rewritten by this turn's delta — so the
        // surviving list is truncated to `<= b` before the state is reused.
        let mut owned_parent: Option<Retained> = None;
        if let Some((idx, b)) = resume_at {
            let mut r = self.sessions.remove(idx);
            let tip = r.boundaries.last().map(|(l, _)| *l).unwrap_or(0);
            if b < tip {
                eprintln!(
                    "[opencode-session] resuming at an earlier boundary ({b} < {tip}): \
                     a retry or a truncated history"
                );
            }
            r.boundaries.retain(|(l, _)| *l <= b);
            owned_parent = Some(r);
        }

        let run = {
            let resume = match owned_parent.take() {
                Some(r) => {
                    parent_boundaries = r.boundaries;
                    engine::Resume::InPlace(r.state, cached_tokens)
                }
                None => engine::Resume::Cold,
            };
            let st = &mut state;
            engine::generate(
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

        // A setup failure (no prefill, unsupported pass kind, a fork the driver
        // refused) still has to answer as a turn, not as a fault.
        let run = match run {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[opencode-session] generation setup failed: {e}");
                self.degrade(&sink, &mut state, streaming, include_usage);
                return;
            }
        };
        if let Some(e) = &run.gen_error {
            eprintln!("[opencode-session] generation degraded to length-finish: {e}");
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
        let retained = self.retain_turn(run, parent_boundaries, save_address);
        // Built as ONE string and printed with a single placeholder. A
        // multi-fragment `eprintln!` is split by the runtime's stderr capture
        // into one client message PER FRAGMENT, so the interpolated form
        // arrives at the shim as a dozen separate lines and the measurement is
        // unreadable exactly where it matters.
        let line = format!(
            "[opencode-session] turn {} cached={cached_tokens} delta={} cue={} gen={accepted} {retained}\n",
            self.counter,
            delta.len(),
            cue.len()
        );
        eprint!("{}", line);

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

    /// Retain the rendered-history state, addressed by the TOKENS it holds.
    ///
    /// The address is `hash(model ‖ template ‖ full_render)` — this turn's own
    /// render, and nothing the server produced. Next turn that same render
    /// reappears as an interior boundary of the next one (history is
    /// append-only), so it is found by the boundary scan without either side
    /// predicting anything.
    ///
    /// That "never name a prediction" rule is load-bearing and was measured
    /// upstream: an OpenHands version that predicted the next boundary from the
    /// inferlet's own text and tool calls saw its hit rate collapse to ~3%,
    /// because the host re-serializes JSON arguments with different bytes.
    /// Naming only full host renders restored 96.6%.
    ///
    /// Returns a log fragment. A turn that errored mid-generation is not
    /// retained: the delta prefill may have stopped part-way, so the length
    /// recorded here would over-claim.
    fn retain_turn(
        &mut self,
        run: engine::Generation,
        mut boundaries: Vec<(u32, String)>,
        address: String,
    ) -> String {
        if let Some(e) = run.gen_error {
            // Dropping `run` releases the state.
            return format!("not retained (generation error: {e})");
        }
        let total = run.total_len;
        boundaries.push((total, address.clone()));
        let candidate = Retained { state: run.state, boundaries };
        // kv_verify on the way IN as well as on the way out: a state that fails
        // its own invariants must never enter the map, or the next turn pays
        // the lookup only to refuse it.
        if let Err(why) = candidate.verify(kv_page_size()) {
            return format!("not retained (kv_verify: {why})");
        }
        self.sessions.push(candidate);

        // Evict oldest-first until BOTH bounds hold. The token budget is the
        // one that matters; the count is a backstop for pathological cases
        // (very many tiny branches). Dropping a `Retained` releases its KV
        // pages and, on a hybrid model, its folded recurrent state.
        //
        // The newest branch is never evicted even if it alone exceeds the
        // budget: it is the one the next turn will resume, and dropping it
        // would guarantee a rebuild every turn rather than risk one.
        loop {
            let held: u32 = self.sessions.iter().map(Retained::tip).sum();
            let over = held > self.retain_tokens || self.sessions.len() > MAX_RETAINED;
            if !over || self.sessions.len() <= 1 {
                if over {
                    eprintln!(
                        "[opencode-session] retained {held} tokens in 1 branch, over the \
                         {} budget — raise retain_tokens or lower the context",
                        self.retain_tokens
                    );
                }
                break;
            }
            let dropped = self.sessions.remove(0);
            eprintln!(
                "[opencode-session] evicted a branch ({} tokens); {held} held over a {} budget",
                dropped.tip(),
                self.retain_tokens
            );
        }
        format!("retained {} (len {total})", &address[..16])
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
/// `Cue` renders through `chat::cue_no_think` — this milestone always serves the
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
fn render_history(ops: &[RenderOp]) -> Result<(Vec<u32>, Vec<u32>), String> {
    let mut out = Vec::new();
    let mut bounds = Vec::with_capacity(ops.len());
    for op in ops {
        render_one(op, &mut out)?;
        bounds.push(out.len() as u32);
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
/// `Cue` renders through `chat::cue_no_think` — this milestone always serves the
/// no-think channel, matching the token-exact parity verdict against HF
/// `enable_thinking=False`, and matching Strategy A so the A/B compares servers
/// rather than renderers.
fn render_one(op: &RenderOp, out: &mut Vec<u32>) -> Result<(), String> {
    match op {
        RenderOp::EquipAfterSystem { system, tools: schemas } => {
            out.extend(tools::equip_after_system(system.as_deref(), schemas)?);
        }
        RenderOp::User(t) => out.extend(chat::user(t)),
        RenderOp::Assistant(t) => out.extend(chat::assistant(t)),
        RenderOp::AssistantWithToolCalls { content, calls } => {
            let wit_calls: Vec<tools::ToolCall> = calls
                .iter()
                .map(|(name, args)| tools::ToolCall {
                    name: name.clone(),
                    arguments_json: args.clone(),
                })
                .collect();
            out.extend(tools::assistant_with_tool_calls(content.as_deref(), &wit_calls));
        }
        RenderOp::AnswerBatch(batch) => out.extend(tools::answer_batch(batch)),
        RenderOp::Cue => out.extend(chat::cue_no_think()),
    }
    Ok(())
}
