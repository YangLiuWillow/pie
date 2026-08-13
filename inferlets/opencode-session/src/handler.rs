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

use std::collections::HashMap;

use crate::engine::{self, GenConfig, SessionState};
use crate::turn::TurnState;
use crate::wire::{Envelope, Sink, recover_req_id, send_error};

use inferlet::{chat, model, runtime, tools};
use pie_openai_serving::error::{INVALID_REQUEST_ERROR, SERVER_ERROR, parse_request};
use pie_openai_serving::streaming::ChunkMeta;
use pie_openai_serving::{
    ChatCompletionRequest, RenderOp, canon_messages, cut_leading_reasoning, plan_render,
    plan_render_suffix, sanitize_messages, snapshot_address, split_retain_point,
    tool_schema_envelopes,
};

/// Defaults when the client sends none. opencode always sends `max_tokens` but
/// omits `temperature`/`top_p` for unknown model ids (AUDIT §1a).
const DEFAULT_MAX_TOKENS: usize = 4096;
const DEFAULT_TEMPERATURE: f32 = 0.6;
const DEFAULT_TOP_P: f32 = 0.95;

/// Retained conversation branches. Each pins its KV pages (and, on a hybrid
/// model, its folded state) for as long as it lives, so this is a memory bound
/// as much as a hit-rate one. Eight covers a linear conversation plus the
/// retries and title side-calls opencode interleaves; past that the LRU tail is
/// almost certainly dead branches.
const MAX_RETAINED: usize = 8;

fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub struct Daemon {
    /// Content address → (retained state, valid token count).
    sessions: HashMap<String, (SessionState, u32)>,
    /// Insertion order for the LRU cap.
    lru: Vec<String>,
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
    pub fn new() -> Self {
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
            sessions: HashMap::new(),
            lru: Vec::new(),
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
        let no_think = req.no_think();
        let schemas = tool_schema_envelopes(&req.tools);

        // Sanitize before BOTH rendering and canonicalization, so the address
        // hashes what actually replays.
        sanitize_messages(&mut req.messages, &self.specials);

        // ── Resume ───────────────────────────────────────────────────────
        // Hash the prefix up to and including the last assistant message: the
        // previous turn retained its post-generation state under exactly that
        // address. Generation forks the hit, so a client-side retry of the same
        // turn can re-hit the same parent.
        let mut resume_key: Option<String> = None;
        let mut cached_tokens = 0u32;
        let retain_split = split_retain_point(&req.messages);
        if let Some(split) = retain_split {
            let canons = canon_messages(&req.messages[..split]);
            let key = snapshot_address(&schemas, no_think, canons.iter());
            if let Some((_, total)) = self.sessions.get(&key) {
                cached_tokens = *total;
                resume_key = Some(key);
            }
        }

        // ── Render: the suffix on a hit, the whole history on a miss ──────
        let ops = match &resume_key {
            Some(_) => {
                plan_render_suffix(&req.messages, retain_split.unwrap())
            }
            None => plan_render(&req),
        };
        let ops = match ops {
            Ok(o) => o,
            Err(e) => {
                send_error(req_id, 400, INVALID_REQUEST_ERROR, &e.to_string());
                return;
            }
        };
        // Split the plan at its trailing `Cue`. The delta is rendered history
        // and is what gets retained; the cue is generation scaffolding and must
        // NOT be — replaying this turn as history renders no cue at all, so a
        // retained cue is tokens no re-render will ever produce (engine docs).
        let (delta_ops, cue_ops) = match ops.split_last() {
            Some((RenderOp::Cue, head)) => (head, &ops[ops.len() - 1..]),
            _ => {
                eprintln!("[opencode-session] render plan did not end with a cue");
                send_error(req_id, 500, SERVER_ERROR, "internal error while rendering the prompt");
                return;
            }
        };
        let rendered = render_ops(delta_ops).and_then(|d| render_ops(cue_ops).map(|c| (d, c)));
        let (delta, cue) = match rendered {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[opencode-session] render fault: {e}");
                send_error(req_id, 500, SERVER_ERROR, "internal error while rendering the prompt");
                return;
            }
        };
        if delta.is_empty() {
            send_error(
                req_id,
                400,
                INVALID_REQUEST_ERROR,
                "`messages` rendered to an empty prompt",
            );
            return;
        }
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

        // The retained entry is TAKEN out of the map: the turn extends that very
        // working set (nothing forks — see `engine::Resume`), so leaving a
        // second handle in the map would advertise a prefix whose tail is about
        // to be overwritten. `retain_turn` puts the extended state back under
        // the new address; a failed turn drops it, which is a clean miss.
        let owned_parent = match &resume_key {
            Some(k) => {
                self.lru.retain(|x| x != k);
                self.sessions.remove(k)
            }
            None => None,
        };

        let run = {
            let resume = match owned_parent {
                Some((st, t)) => engine::Resume::InPlace(st, t),
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
        let retained = self.retain_turn(run, &req, &schemas, no_think, resume_key);
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

    /// Retain the rendered-history state under the address the NEXT request
    /// will hash.
    ///
    /// The address is the canon of **this request's own messages** — nothing the
    /// server produced enters it, because nothing the server produced is in the
    /// retained KV either (`engine` module docs). Next turn the client echoes our
    /// assistant turn back and appends; `split_retain_point` lands immediately
    /// before that assistant message, so the prefix it hashes is exactly this
    /// message list.
    ///
    /// Returns a log fragment. A turn that errored mid-generation is not
    /// retained. On the fork path its damage was confined to a scratch fork
    /// anyway, but on the in-place path the delta prefill may have stopped
    /// part-way, so the length recorded here would over-claim.
    fn retain_turn(
        &mut self,
        run: engine::Generation,
        req: &ChatCompletionRequest,
        schemas: &[String],
        no_think: bool,
        parent: Option<String>,
    ) -> String {
        if let Some(e) = run.gen_error {
            // Dropping `run` releases the state (and the scratch fork with it).
            return format!("not retained (generation error: {e})");
        }

        let canons = canon_messages(&req.messages);
        let key = snapshot_address(schemas, no_think, canons.iter());
        let total = run.total_len;

        // Drop the parent this turn extended: one live state per conversation
        // branch. The fork shares the parent's pages copy-on-write, so keeping
        // both pins two copies of a prefix that only one of them will ever be
        // resumed from.
        if let Some(p) = parent {
            if self.sessions.remove(&p).is_some() {
                self.lru.retain(|k| k != &p);
            }
        }
        if self.sessions.insert(key.clone(), (run.state, total)).is_none() {
            self.lru.push(key.clone());
        }
        while self.sessions.len() > MAX_RETAINED && !self.lru.is_empty() {
            let evict = self.lru.remove(0);
            self.sessions.remove(&evict);
            eprintln!("[opencode-session] evicted retained branch {evict}");
        }
        format!("retained {key} (len {total})")
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
fn render_ops(ops: &[RenderOp]) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for op in ops {
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
    }
    Ok(out)
}
