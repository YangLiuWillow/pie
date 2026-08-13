//! Per-token orchestration: token → (tool decode, stop policy, visible-text
//! filtering, chunk emission).
//!
//! The ordering discipline is Strategy A's `chat-completions/src/turn.rs`,
//! which is itself the validated qwen-code handler's streaming loop: per token
//! — (1) feed the native tool decoder and emit an atomic tool-call delta on a
//! completed call, (2) check the stop set, (3) feed the chat decoder and pass
//! its delta through [`VisibleFilter`] so think/tool-call markup (even a
//! partial `<tool_call>` straddling a chunk boundary) never leaks into content
//! deltas, (4) check client stop strings on the visible tail (deltas already on
//! the wire are not retracted). A turn with ≥1 call keeps decoding until the
//! model itself stops.
//!
//! ## Deliberately a copy, not a shared module
//!
//! This duplicates ~150 lines of Strategy A's `turn.rs`. Two reasons, and they
//! outweigh the duplication:
//!
//! - the emission target differs (envelope-per-request here, raw `session::send`
//!   there) and so does tool-call id scope (process-lifetime here, one-request
//!   there) — the shared part is the *state machine*, which cannot move into
//!   `pie-openai-serving` because it drives the WIT decoders;
//! - `inferlets/chat-completions` is FROZEN and shared with the openclaw
//!   session (handover §10). Strategy A is the control for the A/B measurement;
//!   editing it to factor out a helper would put the baseline at risk to save
//!   copying a file.
//!
//! If a third consumer appears, factor it then — with the frozen arm's suite
//! green on both sides of the change.

use crate::wire::Sink;
use inferlet::{chat, tools};
use pie_openai_serving::streaming::ChunkMeta;
use pie_openai_serving::{
    VisibleFilter, answer_after_reasoning, parse_fenced_tool_calls, parse_hermes_tool_calls,
};
use serde_json::Value;
use std::ops::ControlFlow;

pub struct ToolCallOut {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

pub struct TurnState {
    pub meta: ChunkMeta,
    sink: Sink,
    /// Streaming mode emits chunks as they happen; non-streaming collects the
    /// same state silently and the caller assembles one body.
    streaming: bool,
    /// Unique fragment for tool-call ids (`call_{uniq}_{n}`). Under Strategy A
    /// this was the per-instance id, because there was one WASM instance per
    /// request. Here ONE instance serves every turn of a session, so the
    /// fragment must also carry a per-turn counter — otherwise turn 2 hands the
    /// client `call_{uniq}_0` again, colliding with turn 1's id, and opencode
    /// matches tool results to the wrong call.
    uniq: String,
    stop_ids: Vec<u32>,
    stop_strings: Vec<String>,
    tool_decoder: Option<tools::Decoder>,
    chat_decoder: chat::Decoder,
    filter: VisibleFilter,
    pub generated: Vec<u32>,
    pub visible_text: String,
    pub calls: Vec<ToolCallOut>,
    pub emitted_visible: bool,
}

impl TurnState {
    pub fn new(
        meta: ChunkMeta,
        sink: Sink,
        streaming: bool,
        uniq: String,
        stop_ids: Vec<u32>,
        stop_strings: Vec<String>,
        has_tools: bool,
    ) -> Self {
        Self {
            meta,
            sink,
            streaming,
            uniq,
            stop_ids,
            stop_strings,
            tool_decoder: has_tools.then(tools::Decoder::new),
            chat_decoder: chat::Decoder::new(),
            filter: VisibleFilter::new(),
            generated: Vec::new(),
            visible_text: String::new(),
            calls: Vec::new(),
            emitted_visible: false,
        }
    }

    /// Emit one `chat.completion.chunk` on this turn's envelope, when
    /// streaming. The shim adds `data:` framing and `[DONE]`.
    pub fn emit(&self, chunk: &Value) {
        if self.streaming {
            self.sink.chunk(chunk);
        }
    }

    /// Handle one sampled token. `Break` ends the turn.
    pub fn on_token(&mut self, t: u32) -> ControlFlow<()> {
        self.generated.push(t);

        // (1) Native tool-call detection. Decoder errors are ignored — a
        // confused decoder must not kill the stream.
        let call = match &self.tool_decoder {
            Some(dec) => match dec.feed(&[t]) {
                Ok(tools::Event::Call(c)) => Some(c),
                _ => None,
            },
            None => None,
        };
        if let Some(c) = call {
            self.push_call(c.name, c.arguments_json);
        }

        // (2) Chat stop tokens end the turn.
        if self.stop_ids.contains(&t) {
            return ControlFlow::Break(());
        }

        // (3) Visible content. The chat decoder's raw delta includes
        // think/tool-call markup text; the filter suppresses it (fencing).
        match self.chat_decoder.feed(&[t]) {
            Ok(chat::Event::Delta(s)) => {
                let v = self.filter.feed(&s);
                if !v.is_empty() {
                    self.emit(&self.meta.content_delta(&v));
                    self.emitted_visible = true;
                    self.visible_text.push_str(&v);
                }
            }
            Ok(chat::Event::Done(_)) => return ControlFlow::Break(()),
            _ => {}
        }

        // (4) Client-supplied stop strings (not sent by opencode; checked on
        // the visible tail for API completeness).
        if !self.stop_strings.is_empty()
            && self.stop_strings.iter().any(|s| self.visible_text.ends_with(s))
        {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }

    /// Register a completed tool call: dedup (looping models emit the same call
    /// several times in one turn; executing the copies just burns agent
    /// iterations), assign the session-unique id, and emit the atomic tool-call
    /// delta (index + id + function.name + arguments in ONE delta — opencode's
    /// SDK throws unless the first delta per index carries id and name).
    pub fn push_call(&mut self, name: String, arguments: String) {
        let dup = self.calls.iter().any(|c| c.name == name && c.arguments == arguments);
        if dup {
            return;
        }
        let id = format!("call_{}_{}", self.uniq, self.calls.len());
        self.emit(&self.meta.tool_call_delta(self.calls.len(), &id, &name, &arguments));
        self.calls.push(ToolCallOut { id, name, arguments });
    }

    /// Flush the filter's held-back tail at end of generation.
    ///
    /// No trimming or truncation past this point in streaming mode:
    /// `visible_text` is exactly the bytes already streamed, opencode echoes
    /// those back verbatim as assistant content next turn, and the retention
    /// address hashes that same string — any divergence and every subsequent
    /// resume misses, silently.
    pub fn flush_tail(&mut self) {
        let tail = self.filter.finish();
        if !tail.is_empty() {
            self.emit(&self.meta.content_delta(&tail));
            self.emitted_visible = true;
            self.visible_text.push_str(&tail);
        }
    }

    /// Salvage passes for tool calls the native decoder missed — fenced JSON
    /// blocks in the visible text, then unclosed hermes blocks in the raw
    /// generation (which the filter swallowed). Runs only when the native
    /// decoder produced nothing.
    pub fn salvage(&mut self, has_tools: bool, raw_text: &str) {
        if !has_tools {
            return;
        }
        if self.calls.is_empty() {
            for (_, name, args) in parse_fenced_tool_calls(&self.visible_text) {
                self.push_call(name, args);
            }
        }
        if self.calls.is_empty() && raw_text.contains("<tool_call>") {
            for (name, args) in parse_hermes_tool_calls(raw_text) {
                self.push_call(name, args);
            }
        }
    }

    pub fn finish_reason(&self, hit_max: bool, gen_error: bool) -> &'static str {
        if !self.calls.is_empty() {
            "tool_calls"
        } else if hit_max || gen_error {
            "length"
        } else {
            "stop"
        }
    }

    /// The one canonical content string for this turn — used for BOTH the
    /// response body and the retention address. Any divergence between the two
    /// breaks every subsequent KV resume, because opencode echoes response
    /// content back verbatim.
    ///
    /// A text turn that produced nothing visible still delivers non-empty
    /// content: fall back to whatever the raw generation says AFTER its
    /// reasoning, then to a non-whitespace placeholder. Never to the reasoning
    /// itself — `answer_after_reasoning` treats an unterminated think block as
    /// no answer at all, because stripping the tags and keeping the body serves
    /// the model's private working-out as its answer: fluent, on-topic, wrong
    /// in kind, and invisible to every test.
    pub fn final_content(&self, raw_text: &str) -> String {
        if !self.visible_text.is_empty() || !self.calls.is_empty() {
            return self.visible_text.clone();
        }
        answer_after_reasoning(raw_text)
            .map(str::to_string)
            .unwrap_or_else(|| "…".to_string())
    }
}
