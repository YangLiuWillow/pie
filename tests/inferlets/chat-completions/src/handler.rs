//! Request orchestration for the chat-completions daemon.
//!
//! Contract (audit §1 hard-requirements table, `docs/qwen-code-rl-audit.md`):
//! - Malformed request → error event with a 400 status hint and an OpenAI
//!   error body; 500 is reserved for genuine server faults (500s trigger
//!   7×-app × 3×-SDK retry storms client-side).
//! - Never a context-length 400 (fires H1 full-history compaction
//!   client-side): overflow/generation faults degrade to
//!   `finish_reason:"length"` with whatever streamed.
//! - Text turns must end with non-empty content; tool-call ids must be
//!   unique across the whole session (long-lived instance, so
//!   `instance_id` fragment + a process-lifetime counter).
//!
//! Generation runs UNCONSTRAINED (grammar-constraining the whole turn
//! suppressed reasoning text and collapsed t=0 trajectories into action
//! loops — openhands-completion finding), with the salvage parsers on top.
//! There is deliberately no grammar-forced phase-2 call: constrained
//! decoding trapped the guest on the old portable driver, and a dead stream
//! is strictly worse than a no-call turn; re-add behind a capability probe.
//!
//! Session retention (the port's storage change): retained `SessionState`s
//! live in a process-local map keyed by the content address from
//! `session.rs`. On a hit the turn generates on a CoW fork; after a clean
//! turn the extended fork is stored under the echo-back address and the
//! parent entry is dropped (≤1 live retained set per conversation branch).
//! A daemon restart or an eviction is a clean full-rebuild miss.

use std::collections::HashMap;

use crate::chunk::{self, ChunkMeta};
use crate::filter::VisibleFilter;
use crate::generation::{self, GenParams, Generation};
use crate::render::Renderer;

use crate::salvage;
use crate::session::{self, CanonItem};
use crate::types::{ChatCompletionRequest, ChatMessage, tool_schema_envelopes};

use inferlet::model;
use inferlet::pie::inferlet::tools as tools_wit;
use crate::generation::SessionState;
use serde::Deserialize;

/// Defaults when the client sends none — Qwen3 no-think guidance (qwen-code
/// sends `max_tokens` always but `temperature`/`top_p` only if configured).
const DEFAULT_MAX_TOKENS: usize = 4096;
/// Content for a turn that failed after the stream opened. Must be
/// non-whitespace (see the degrade macro) and should read as a message,
/// since the client echoes it back as assistant history.
const DEGRADED_TURN_TEXT: &str = "The server could not complete this turn.";

const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_P: f32 = 0.8;

/// Retained conversation branches (each pins its KV pages while live).
const MAX_RETAINED_SESSIONS: usize = 8;

fn send(s: String) {
    inferlet::session::send(&s);
}

/// Inbound shim envelope. `now` is wall-clock unix seconds supplied by the
/// shim — the inferlet world imports no wall clock of its own.
#[derive(Deserialize)]
struct Envelope {
    #[serde(default)]
    req_id: String,
    body: serde_json::Value,
    #[serde(default)]
    now: i64,
}

struct ToolCallOut {
    id: String,
    name: String,
    arguments: String,
}

struct TurnSetup {
    messages: Vec<ChatMessage>,
    tool_schemas: Vec<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    no_think: bool,
    stop_strings: Vec<String>,
    include_usage: bool,
}

pub struct Daemon {
    renderer: Renderer,
    sessions: HashMap<String, (generation::SessionState, u32)>,
    lru: Vec<String>,
    uniq: String,
    counter: u64,
}

impl Daemon {
    pub fn new() -> Self {
        let uniq: String = inferlet::runtime::instance_id()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .take(8)
            .collect();
        Self {
            renderer: Renderer::new(),
            sessions: HashMap::new(),
            lru: Vec::new(),
            uniq,
            counter: 0,
        }
    }

    pub async fn handle(&mut self, raw: &str) {
        let env: Envelope = match serde_json::from_str(raw) {
            Ok(e) => e,
            Err(e) => {
                // Best-effort req_id recovery so the shim can route the error.
                let req_id = serde_json::from_str::<serde_json::Value>(raw)
                    .ok()
                    .and_then(|v| v.get("req_id").and_then(|r| r.as_str()).map(String::from))
                    .unwrap_or_default();
                send(chunk::ev_error(
                    &req_id,
                    400,
                    "invalid_request_error",
                    &format!("Invalid envelope: {e}"),
                ));
                return;
            }
        };
        self.run_turn(&env.req_id, env.now, env.body).await;
    }

    async fn run_turn(&mut self, req_id: &str, now: i64, body: serde_json::Value) {
        let request: ChatCompletionRequest = match serde_json::from_value(body) {
            Ok(r) => r,
            Err(e) => {
                send(chunk::ev_error(
                    req_id,
                    400,
                    "invalid_request_error",
                    &format!("Invalid JSON: {e}"),
                ));
                return;
            }
        };
        if request.messages.is_empty() {
            send(chunk::ev_error(
                req_id,
                400,
                "invalid_request_error",
                "`messages` must be a non-empty array",
            ));
            return;
        }
        // Attention and hybrid (GDN) models both serve here: `generation`
        // dispatches on `model::pass_kind()` and binds a recurrent-state
        // working set alongside the KV one when the model has a fold.
        // Recurrent-only is still refused, inside `generation`.

        let stream = request.stream;
        self.counter += 1;
        let uniq = format!("{}{}", self.uniq, self.counter);
        let mut setup = TurnSetup {
            tool_schemas: tool_schema_envelopes(&request.tools),
            max_tokens: request.effective_max_tokens(DEFAULT_MAX_TOKENS),
            temperature: request.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            top_p: request.top_p.unwrap_or(DEFAULT_TOP_P),
            no_think: request.no_think(),
            stop_strings: request.stop_strings(),
            include_usage: request.include_usage(),
            messages: request.messages,
        };
        let meta = ChunkMeta {
            id: format!("chatcmpl-{uniq}"),
            model: model::name(),
            created: now,
        };

        if request.echo_tokens {
            // C3 parity debug: full clean render (no resume, no session
            // interaction, no generation), answered as a non-stream response
            // regardless of `stream`.
            self.renderer.sanitize_messages(&mut setup.messages);
            match self
                .renderer
                .render_full(&setup.messages, &setup.tool_schemas, setup.no_think)
            {
                Ok(mut t) => {
                    t.extend_from_slice(self.renderer.cue());
                    let text = model::decode(&t).unwrap_or_default();
                    let resp = serde_json::json!({
                        "object": "pie.debug.render",
                        "model": meta.model,
                        "prompt_tokens": t.len(),
                        "rendered_ids": t,
                        "rendered_text": text,
                    });
                    send(chunk::ev_response(req_id, &resp));
                }
                Err(e) => send(chunk::ev_error(
                    req_id,
                    500,
                    "server_error",
                    &format!("render failed: {e}"),
                )),
            }
            return;
        }

        // Role chunk first — announces the assistant turn and doubles as the
        // first keepalive before a potentially long prefill. From here every
        // streaming failure must be shaped as a well-formed turn
        // (finish_reason "length"), never a dead stream — a mid-stream cut
        // triggers qwen-code's synthetic-continuation path (H3).
        if stream {
            send(chunk::ev_chunk(req_id, &meta.role_chunk()));
        }

        macro_rules! degrade {
            ($msg:expr) => {{
                eprintln!("[chat-completions] degraded turn: {}", $msg);
                if stream {
                    // NOT whitespace: qwen-code trims content before its
                    // non-empty check, so a " " delta still reads as an empty
                    // turn and burns the 4x NO_FINISH_REASON retry budget on
                    // an error that will not fix itself (dummy-driver harness
                    // run: 3 identical retries per task, then session abort).
                    send(chunk::ev_chunk(
                        req_id,
                        &meta.content_delta(DEGRADED_TURN_TEXT),
                    ));
                    send(chunk::ev_chunk(req_id, &meta.finish_chunk("length")));
                    if setup.include_usage {
                        send(chunk::ev_chunk(req_id, &meta.usage_chunk(0, 0, 0)));
                    }
                    send(chunk::ev_done(req_id));
                } else {
                    send(chunk::ev_error(req_id, 500, "server_error", &format!("{}", $msg)));
                }
                return;
            }};
        }

        self.renderer.sanitize_messages(&mut setup.messages);

        // ── Resume attempt ───────────────────────────────────────────────
        // Hash the prefix up to the last assistant message; the previous
        // request retained its post-generation working set under exactly
        // that address. Generation forks the hit (CoW, O(1)) so a qwen-code
        // turn-level retry can still re-hit the parent.
        let mut build_debug = String::from("first turn (no resume point)");
        let mut resume_hit: Option<(String, u32)> = None;
        if let Some(split) = session::split_resume_point(&setup.messages) {
            let canons = session::canon_messages(&setup.messages[..split]);
            let key = session::snapshot_name(&setup.tool_schemas, setup.no_think, canons.iter());
            if let Some((_, total)) = self.sessions.get(&key) {
                build_debug = format!("resume hit {key} (cached {total})");
                resume_hit = Some((key, *total));
            } else {
                build_debug = format!("resume miss {key}");
            }
        }

        let prefill: Vec<u32> = {
            let rendered = match &resume_hit {
                Some((_, _)) => {
                    let split = session::split_resume_point(&setup.messages).unwrap();
                    let mut suffix = Vec::new();
                    match self.renderer.render_messages(
                        &setup.messages[split..],
                        setup.no_think,
                        &mut suffix,
                    ) {
                        Ok(()) => Ok(suffix),
                        Err(e) => Err(e),
                    }
                }
                None => self.renderer.render_full(
                    &setup.messages,
                    &setup.tool_schemas,
                    setup.no_think,
                ),
            };
            match rendered {
                Ok(mut t) => {
                    t.extend_from_slice(self.renderer.cue());
                    t
                }
                Err(e) => degrade!(format!("render failed: {e}")),
            }
        };
        let cached_tokens = resume_hit.as_ref().map(|(_, t)| *t).unwrap_or(0);
        let prompt_tokens = cached_tokens + prefill.len() as u32;

        // ── Generation with per-event emission ───────────────────────────
        let params = GenParams {
            temperature: setup.temperature,
            top_p: setup.top_p,
            max_tokens: setup.max_tokens.max(1),
        };
        let stop_ids = self.renderer.stop_ids.clone();
        let has_tools = !setup.tool_schemas.is_empty();
        let tool_decoder = has_tools.then(tools_wit::Decoder::new);
        let chat_dec = inferlet::chat::Decoder::new();
        let mut filter = VisibleFilter::new();
        let mut visible_text = String::new();
        let mut emitted_visible = false;
        let mut calls: Vec<ToolCallOut> = Vec::new();

        let gen_result = {
            let resume_ref: Option<(&SessionState, u32)> = resume_hit
                .as_ref()
                .and_then(|(k, t)| self.sessions.get(k).map(|(st, _)| (st, *t)));
            let calls_ref = &mut calls;
            let visible_ref = &mut visible_text;
            let emitted_ref = &mut emitted_visible;
            let stop_strings = &setup.stop_strings;
            let uniq_ref = &uniq;
            generation::generate(
                resume_ref,
                &prefill,
                &params,
                &stop_ids,
                || {
                    // One keepalive per committed prefill chunk (empty delta
                    // — resets the client's 240 s idle watchdog through the
                    // OpenAI SDK).
                    if stream {
                        send(chunk::ev_chunk(req_id, &meta.keepalive()));
                    }
                },
                |t| {
                    if let Some(dec) = tool_decoder.as_ref() {
                        if let Ok(tools_wit::Event::Call(c)) = dec.feed(&[t]) {
                            // Looping models emit the same call several
                            // times in one turn; executing the copies just
                            // burns agent iterations.
                            let dup = calls_ref
                                .iter()
                                .any(|x| x.name == c.name && x.arguments == c.arguments_json);
                            if !dup {
                                let call_id = format!("call_{uniq_ref}_{}", calls_ref.len());
                                if stream {
                                    send(chunk::ev_chunk(
                                        req_id,
                                        &meta.tool_call_delta(
                                            calls_ref.len(),
                                            &call_id,
                                            &c.name,
                                            &c.arguments_json,
                                        ),
                                    ));
                                }
                                calls_ref.push(ToolCallOut {
                                    id: call_id,
                                    name: c.name,
                                    arguments: c.arguments_json,
                                });
                            }
                        }
                    }
                    match chat_dec.feed(&[t]) {
                        Ok(inferlet::chat::Event::Delta(s)) => {
                            let v = filter.feed(&s);
                            if !v.is_empty() {
                                if stream {
                                    send(chunk::ev_chunk(req_id, &meta.content_delta(&v)));
                                }
                                *emitted_ref = true;
                                visible_ref.push_str(&v);
                            }
                        }
                        Ok(inferlet::chat::Event::Done(_)) => return false,
                        _ => {}
                    }
                    // Client-supplied stop strings (not sent by qwen-code;
                    // checked on the visible tail for API completeness —
                    // deltas already on the wire are not retracted).
                    if !stop_strings.is_empty()
                        && stop_strings.iter().any(|s| visible_ref.ends_with(s))
                    {
                        return false;
                    }
                    true
                },
            )
            .await
        };

        let Generation { state, total_len, generated, hit_max, gen_error } = match gen_result {
            Ok(g) => g,
            Err(e) => degrade!(format!("generation setup failed: {e}")),
        };

        // Flush any held-back filter tail.
        let tail = filter.finish();
        if !tail.is_empty() {
            if stream {
                send(chunk::ev_chunk(req_id, &meta.content_delta(&tail)));
            }
            emitted_visible = true;
            visible_text.push_str(&tail);
        }
        // No trimming/truncation past this point in streaming mode:
        // `visible_text` is exactly the bytes already streamed, qwen-code
        // echoes those back verbatim as the assistant content next turn, and
        // the retained-session address must hash that same string or every
        // subsequent KV resume misses.
        if !stream {
            visible_text = visible_text.trim_end().to_string();
        }

        // Salvage passes for tool calls the native decoder missed.
        let raw_text = model::decode(&generated).unwrap_or_default();
        let push_salvaged = |name: String, args: String, calls: &mut Vec<ToolCallOut>| {
            let dup = calls.iter().any(|c| c.name == name && c.arguments == args);
            if !dup {
                let call_id = format!("call_{uniq}_{}", calls.len());
                if stream {
                    send(chunk::ev_chunk(
                        req_id,
                        &meta.tool_call_delta(calls.len(), &call_id, &name, &args),
                    ));
                }
                calls.push(ToolCallOut { id: call_id, name, arguments: args });
            }
        };
        // Fenced JSON blocks (fence bytes stay in the already-streamed
        // content; the calls are surfaced on top).
        if calls.is_empty() && has_tools {
            for (_, name, args) in salvage::parse_fenced_tool_calls(&visible_text) {
                push_salvaged(name, args, &mut calls);
            }
        }
        // Coder-XML: `<function=…>` blocks without the `<tool_call>` wrapper.
        if calls.is_empty() && has_tools && visible_text.contains("<function=") {
            for (name, args) in salvage::parse_coder_xml_calls(&visible_text, &setup.tool_schemas)
            {
                push_salvaged(name, args, &mut calls);
            }
        }
        // Hermes with the closing tag missing — the filter swallowed the
        // visible text, so scan the RAW generation.
        if calls.is_empty() && has_tools && raw_text.contains("<tool_call>") {
            for (name, args) in salvage::parse_hermes_tool_calls(&raw_text) {
                push_salvaged(name, args, &mut calls);
            }
        }

        let finish_reason = if !calls.is_empty() {
            "tool_calls"
        } else if hit_max || gen_error.is_some() {
            "length"
        } else {
            "stop"
        };

        // The one canonical content string for this turn — used for BOTH
        // the response and the retained-session address. Empty text on a
        // no-tool-call turn trips qwen-code's NO_RESPONSE_TEXT retry loop,
        // so fall back to the raw generation with think markup stripped, and
        // to a non-whitespace placeholder as the last resort.
        let final_content = if !visible_text.is_empty() || !calls.is_empty() {
            visible_text.clone()
        } else {
            let cleaned = raw_text
                .replace("<think>", "")
                .replace("</think>", "")
                .trim()
                .to_string();
            if cleaned.is_empty() { "…".to_string() } else { cleaned }
        };

        if let Some(e) = &gen_error {
            eprintln!("[chat-completions] generation degraded to length-finish: {e}");
        }
        if !emitted_visible && calls.is_empty() && stream {
            send(chunk::ev_chunk(req_id, &meta.content_delta(&final_content)));
        }

        // ── Retain the session *before* signalling completion ────────────
        // qwen-code fires the follow-up request the moment the stream
        // closes, and the retained set must already exist for the resume to
        // hit. Failures are non-fatal (the next request pays a full rebuild).
        let save_debug = if gen_error.is_none() {
            match generation::seal(&state, total_len, self.renderer.seal_tokens()).await {
                Ok(total_final) => {
                    let mut canons = session::canon_messages(&setup.messages);
                    if !final_content.is_empty() {
                        canons.push(CanonItem::Msg {
                            role: "assistant".to_string(),
                            text: final_content.clone(),
                        });
                    }
                    for c in &calls {
                        canons.push(CanonItem::Call {
                            id: c.id.clone(),
                            name: c.name.clone(),
                            args: c.arguments.clone(),
                        });
                    }
                    let new_key = session::snapshot_name(
                        &setup.tool_schemas,
                        setup.no_think,
                        canons.iter(),
                    );
                    let parent = resume_hit.as_ref().map(|(k, _)| k.clone());
                    self.retain(new_key.clone(), state, total_final, parent);
                    format!("retained {new_key} (seq {total_final})")
                }
                Err(e) => format!("seal failed, session dropped: {e}"),
            }
        } else {
            "session dropped (generation error)".to_string()
        };
        eprintln!("[chat-completions] {build_debug}; {save_debug}");

        let n_generated = generated.len() as u32;
        if stream {
            send(chunk::ev_chunk(req_id, &meta.finish_chunk(finish_reason)));
            if setup.include_usage {
                send(chunk::ev_chunk(
                    req_id,
                    &meta.usage_chunk(prompt_tokens, n_generated, cached_tokens),
                ));
            }
            send(chunk::ev_done(req_id));
        } else {
            let tool_calls_json: Vec<serde_json::Value> = calls
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    })
                })
                .collect();
            let response = serde_json::json!({
                "id": meta.id,
                "object": "chat.completion",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": if final_content.is_empty() {
                            serde_json::Value::Null
                        } else {
                            serde_json::Value::String(final_content.clone())
                        },
                        "tool_calls": if tool_calls_json.is_empty() {
                            serde_json::Value::Null
                        } else {
                            serde_json::Value::Array(tool_calls_json)
                        },
                    },
                    "logprobs": null,
                    "finish_reason": finish_reason,
                }],
                "usage": chunk::usage_object(prompt_tokens, n_generated, cached_tokens),
            });
            send(chunk::ev_response(req_id, &response));
        }

    }

    /// Insert a retained session; drop the parent entry it extended (≤1 live
    /// retained set per conversation branch — the old take-on-hit bound) and
    /// LRU-evict past the cap. Dropping a `SessionState` releases its KV
    /// pages and, on a hybrid model, its folded recurrent state.
    fn retain(&mut self, key: String, state: SessionState, total: u32, parent: Option<String>) {
        if let Some(p) = parent {
            if self.sessions.remove(&p).is_some() {
                self.lru.retain(|k| k != &p);
            }
        }
        if self.sessions.insert(key.clone(), (state, total)).is_none() {
            self.lru.push(key);
        }
        while self.sessions.len() > MAX_RETAINED_SESSIONS && !self.lru.is_empty() {
            let evict = self.lru.remove(0);
            self.sessions.remove(&evict);
            eprintln!("[chat-completions] evicted retained session {evict}");
        }
    }
}
