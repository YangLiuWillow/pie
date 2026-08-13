//! Per-turn orchestration state machine: token → (tool decode, stop policy,
//! visible-text filtering, chunk emission).
//!
//! The ordering discipline is ported from the validated qwen-code handler
//! (`openhands-integration-updated:inferlets/chat-completions/src/handler.rs`,
//! streaming loop): per token — (1) feed the native tool decoder and emit an
//! atomic tool-call delta on a completed call, (2) check the stop set,
//! (3) feed the chat decoder and pass its delta through [`VisibleFilter`] so
//! think/tool-call markup (even a partial `<tool_call>` straddling a chunk
//! boundary) never leaks into content deltas, (4) check client stop strings
//! on the visible tail (deltas already on the wire are not retracted).
//! A turn with ≥1 call keeps decoding until the model itself stops (the old
//! handler never cut generation after a call — the model closes the call
//! block and emits its stop token within a few tokens).
//!
//! All wire JSON goes through `pie-openai-serving`'s builders; nothing here
//! hand-rolls a chunk.

use inferlet::{chat, session, tools};
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
    /// Streaming mode emits chunks as they happen; non-streaming collects
    /// the same state silently and the caller assembles one body.
    streaming: bool,
    /// Per-process unique fragment for tool-call ids (`call_{uniq}_{n}`):
    /// one WASM instance per request, so a static counter would restart at
    /// zero every launch and hand the client colliding ids — the exact
    /// `call_0` dedup bug both prior integrations hit. Derived from
    /// `system::instance-id()`.
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
        streaming: bool,
        uniq: String,
        stop_ids: Vec<u32>,
        stop_strings: Vec<String>,
        has_tools: bool,
        tool_schemas: &[String],
    ) -> Self {
        Self {
            meta,
            streaming,
            uniq,
            stop_ids,
            stop_strings,
            tool_decoder: has_tools.then(|| tools::Decoder::with_tools(tool_schemas)),
            chat_decoder: chat::Decoder::new(),
            filter: VisibleFilter::new(),
            generated: Vec::new(),
            visible_text: String::new(),
            calls: Vec::new(),
            emitted_visible: false,
        }
    }

    /// Emit one envelope message (a ready-made chunk JSON document) when
    /// streaming; the gateway wraps it in a `data:` line.
    pub fn emit(&self, chunk: &Value) {
        if self.streaming {
            session::send(&chunk.to_string());
        }
    }

    /// Handle one sampled token. `Break` ends the turn.
    pub fn on_token(&mut self, t: u32) -> ControlFlow<()> {
        self.generated.push(t);

        // (1) Native tool-call detection. Decoder errors are ignored, as in
        // the old handler — a confused decoder must not kill the stream.
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

    /// Register a completed tool call: dedup (looping models emit the same
    /// call several times in one turn; executing the copies just burns agent
    /// iterations), assign the per-process-unique id, and emit the atomic
    /// tool-call delta (index + id + function.name + arguments in ONE delta
    /// — opencode's SDK throws unless the first delta per index carries id
    /// and name).
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
    /// No trimming/truncation past this point in streaming mode:
    /// `visible_text` is exactly the bytes already streamed, the client
    /// echoes those back verbatim as assistant content next turn, and the
    /// (future) snapshot address must hash that same string or every
    /// subsequent KV resume misses (SEAM: sessions).
    pub fn flush_tail(&mut self) {
        let tail = self.filter.finish();
        if !tail.is_empty() {
            self.emit(&self.meta.content_delta(&tail));
            self.emitted_visible = true;
            self.visible_text.push_str(&tail);
        }
    }

    /// Salvage passes for tool calls the native decoder missed — fenced
    /// JSON blocks in the visible text, then unclosed hermes blocks in the
    /// raw generation. Runs only when the native decoder produced nothing
    /// (matching the old handler's `calls.is_empty()` gating).
    ///
    /// SEAM (Qwen3-Coder XML dialect): the `<function=…>` bare-block
    /// salvage (`parse_coder_xml_calls`, schema-typed values) slots in
    /// between the two passes below when `ToolFormat::Coder` support lands
    /// — port it from the old handler.rs alongside a coder-aware
    /// `tools::Decoder`.
    pub fn salvage(&mut self, has_tools: bool, raw_text: &str) {
        if !has_tools {
            return;
        }
        if self.calls.is_empty() {
            let fenced = parse_fenced_tool_calls(&self.visible_text);
            for (_, name, args) in fenced {
                self.push_call(name, args);
            }
        }
        if self.calls.is_empty() && raw_text.contains("<tool_call>") {
            let hermes = parse_hermes_tool_calls(raw_text);
            for (name, args) in hermes {
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

    /// The one canonical content string for this turn — used for the
    /// response body (and, once sessions land, the snapshot address: any
    /// response/save divergence here breaks every subsequent KV resume —
    /// SEAM: sessions).
    ///
    /// A text turn that produced nothing visible still delivers non-empty
    /// content (qwen-code's NO_RESPONSE_TEXT retry loop; harmless for
    /// opencode): fall back to whatever the raw generation says AFTER its
    /// reasoning, then to a non-whitespace placeholder.
    ///
    /// This used to strip `<think>`/`</think>` and keep everything else,
    /// which served the model's private working-out as its answer in exactly
    /// the case the filter was right to suppress it — fluent, on-topic, and
    /// wrong in kind. `answer_after_reasoning` takes the text after the last
    /// closer and treats an unterminated block as no answer at all; see its
    /// docs for why that matters more the moment a cue opens the block.
    pub fn final_content(&self, raw_text: &str) -> String {
        if !self.visible_text.is_empty() || !self.calls.is_empty() {
            return self.visible_text.clone();
        }
        answer_after_reasoning(raw_text)
            .map(str::to_string)
            .unwrap_or_else(|| "…".to_string())
    }
}
