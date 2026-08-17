//! One table-driven decoder for the reasoning and tool channels, declared per
//! family as data rather than written per family as a state machine.
//!
//! ## Why this exists
//!
//! `ThinkingDecoder` and `QwenToolDecoder` are two hand-written state machines
//! whose *shape* is identical across every ChatML family — find an opening
//! delimiter, accumulate until a closing one, hand the body to a parser — and
//! whose *content* is four strings and a choice of argument syntax. Adding
//! Qwen3.6 needed neither new state nor new transitions; it needed a row
//! saying "thinking, and XML tool calls", a pairing the registry could not
//! express because one predicate drove both.
//!
//! So the family's format becomes a `const` [`ParserEngineConfig`], and the
//! two `dyn` decoders are built from it. This is vLLM's `parser/engine/`
//! split, which collapsed their per-model reasoning parsers to a re-export
//! apiece once the format was declarative.
//!
//! ## What is deliberately NOT copied from vLLM
//!
//! vLLM's `ParserState` carries `TOOL_NAME`, `TOOL_ARGS`, `TOOL_BETWEEN`
//! because its engine streams argument deltas to the client as they arrive.
//! [`ToolDecoder`] emits a whole [`ToolEvent::Call`] at once, so those states
//! would be machinery for a feature this surface does not have. States here
//! are the three the decoders actually occupy. When argument streaming lands,
//! that is when the states earn their place.
//!
//! ## The two channels match differently, and that is not an oversight
//!
//! Reasoning delimiters match on **token IDs**; tool delimiters match on
//! **decoded text**. `<think>` and `</think>` are single tokens in every
//! vocabulary here, so comparing ids is exact and costs no decode — the
//! robustness vLLM had to build `token_id_scanner.py` to recover. But
//! `<function=read_file>` is not a token, it is a name spliced between two
//! fragments, and no id sequence describes it. Matching text there is the only
//! option, and matching ids for reasoning is free, so each channel takes the
//! stronger tool available to it.

use crate::instruct::{ReasoningDecoder, ReasoningEvent, ToolDecoder, ToolEvent};
use pie_tokenizer::{Tokenizer, TokenizerDecoder};
use std::sync::Arc;

/// Turn a completed tool-call body into `(name, arguments-json)`.
///
/// A function pointer rather than an enum of known syntaxes, so a family can
/// supply a parser the shared crate has never heard of. `schemas` carries the
/// request's tool schemas, which the XML syntax needs and JSON does not: XML
/// carries no types, so `"1"` is a string until a schema says it is a number.
///
/// vLLM's `arg_converter: Callable[[str, bool], str]` is the same hook. The
/// `bool` there is `partial`, for streaming; there is no partial here for the
/// reason in the module docs.
pub type ParseCall = fn(body: &str, schemas: &[String]) -> Option<(String, String)>;

/// The reasoning channel's delimiters.
#[derive(Clone, Copy, Debug)]
pub struct ReasoningChannel {
    pub open: &'static str,
    pub close: &'static str,
    /// The generation prompt already opened the channel, so the stream starts
    /// inside it and only `close` is ever matched.
    ///
    /// This is the `generation_suffix: "<think>\n"` families (OLMo, nemotron_h)
    /// and NOT the Qwen ones, whose cue closes an empty block instead.
    pub starts_inside: bool,
}

/// The tool channel's delimiters and argument syntax.
#[derive(Clone, Copy)]
pub struct ToolChannel {
    pub open: &'static str,
    pub close: &'static str,
    /// A call the model emitted with NO wrapper still counts, opened by this
    /// prefix and closed by [`Self::unwrapped_close`].
    ///
    /// Not a nicety. Asked to read a file with ten tools offered,
    /// Qwen3-Coder-30B emits a well-formed `<function=read>…</function>` with
    /// no `<tool_call>` around it, and without this the entire call is dropped
    /// for want of an opening tag the model never wrote. The reference
    /// `qwen3coder_tool_parser` keys its quick check on `<function=` for the
    /// same reason.
    pub unwrapped_open: Option<&'static str>,
    pub unwrapped_close: &'static str,
    pub parse_call: ParseCall,
}

/// A family's output format, as data.
#[derive(Clone, Copy)]
pub struct ParserEngineConfig {
    pub name: &'static str,
    /// `None` for a family with no thinking channel — Qwen3-Coder, which
    /// shares Qwen3's architecture and answers nothing when cued with an empty
    /// think block.
    pub reasoning: Option<ReasoningChannel>,
    pub tools: Option<ToolChannel>,
}

impl ParserEngineConfig {
    /// Both decoders from one declaration.
    ///
    /// The analogue of vLLM's `make_adapters(Qwen3Parser)`, which returns the
    /// reasoning adapter and the tool adapter as a pair so the two cannot be
    /// registered from different declarations.
    pub fn decoders(
        &self,
        tokenizer: Arc<Tokenizer>,
        has_tools: bool,
        schemas: Vec<String>,
    ) -> (Box<dyn ReasoningDecoder>, Box<dyn ToolDecoder>) {
        (
            Box::new(ReasoningAdapter::new(*self, tokenizer.clone())),
            Box::new(ToolAdapter::new(*self, tokenizer, has_tools, schemas)),
        )
    }
}

// ─── reasoning ───────────────────────────────────────────────

/// The reasoning channel, matched on token ids.
///
/// Behaviour is `decoders::ThinkingDecoder`'s, including its partial-match
/// back-off: a run of tokens that starts to spell `close` and then diverges
/// must be replayed as content, or a `</think` that turns out to be prose is
/// swallowed.
pub struct ReasoningAdapter {
    decoder: TokenizerDecoder,
    open_ids: Vec<u32>,
    close_ids: Vec<u32>,
    inside: bool,
    starts_inside: bool,
    enabled: bool,
    text: String,
    match_pos: usize,
}

impl ReasoningAdapter {
    fn new(cfg: ParserEngineConfig, tokenizer: Arc<Tokenizer>) -> Self {
        let ch = cfg.reasoning;
        let enabled = ch.is_some();
        let (open_ids, close_ids, starts_inside) = match ch {
            Some(c) => (
                if c.starts_inside { Vec::new() } else { tokenizer.encode(c.open) },
                tokenizer.encode(c.close),
                c.starts_inside,
            ),
            None => (Vec::new(), Vec::new(), false),
        };
        Self {
            decoder: tokenizer.decoder(false),
            open_ids,
            close_ids,
            inside: enabled && starts_inside,
            starts_inside: enabled && starts_inside,
            enabled,
            text: String::new(),
            match_pos: 0,
        }
    }
}

impl ReasoningDecoder for ReasoningAdapter {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent {
        if !self.enabled {
            return ReasoningEvent::Delta(String::new());
        }
        if !self.inside {
            for &t in tokens {
                if self.match_pos < self.open_ids.len() && t == self.open_ids[self.match_pos] {
                    self.match_pos += 1;
                    if self.match_pos == self.open_ids.len() {
                        self.inside = true;
                        self.match_pos = 0;
                        self.decoder.reset();
                        self.text.clear();
                        return ReasoningEvent::Start;
                    }
                } else {
                    self.match_pos = 0;
                }
            }
            return ReasoningEvent::Delta(String::new());
        }

        let mut content = Vec::with_capacity(tokens.len());
        for &t in tokens {
            let mut matched = false;
            if self.match_pos < self.close_ids.len() && t == self.close_ids[self.match_pos] {
                self.match_pos += 1;
                matched = true;
            } else if self.match_pos > 0 {
                // The partial match was not the delimiter after all; the tokens
                // it consumed are content and have to be put back.
                content.extend_from_slice(&self.close_ids[..self.match_pos]);
                self.match_pos = 0;
                if !self.close_ids.is_empty() && t == self.close_ids[0] {
                    self.match_pos = 1;
                    matched = true;
                }
            }
            if matched {
                if self.match_pos == self.close_ids.len() {
                    let delta = self.decoder.feed(&content);
                    self.text.push_str(&delta);
                    self.text.push_str(&self.decoder.finish());
                    self.inside = false;
                    self.match_pos = 0;
                    self.decoder.reset();
                    return ReasoningEvent::Complete(std::mem::take(&mut self.text));
                }
            } else {
                content.push(t);
            }
        }
        let delta = self.decoder.feed(&content);
        self.text.push_str(&delta);
        ReasoningEvent::Delta(delta)
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.text.clear();
        self.match_pos = 0;
        self.inside = self.starts_inside;
    }
}

// ─── tools ───────────────────────────────────────────────────

/// The tool channel, matched on decoded text.
pub struct ToolAdapter {
    cfg: ParserEngineConfig,
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
    /// The open call arrived without its wrapper, so `unwrapped_close` is what
    /// ends it.
    unwrapped: bool,
    has_tools: bool,
    schemas: Vec<String>,
}

impl ToolAdapter {
    fn new(
        cfg: ParserEngineConfig,
        tokenizer: Arc<Tokenizer>,
        has_tools: bool,
        schemas: Vec<String>,
    ) -> Self {
        Self {
            cfg,
            decoder: tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
            unwrapped: false,
            has_tools,
            schemas,
        }
    }
}

impl ToolDecoder for ToolAdapter {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let Some(ch) = self.cfg.tools else {
            return ToolEvent::Start;
        };
        if !self.has_tools {
            return ToolEvent::Start;
        }
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        if !self.inside {
            if let Some(pos) = self.accumulated.find(ch.open) {
                self.inside = true;
                self.unwrapped = false;
                self.accumulated = self.accumulated[pos + ch.open.len()..].to_string();
                return ToolEvent::Start;
            }
            if let Some(open) = ch.unwrapped_open {
                // The prefix is KEPT rather than consumed: it carries the
                // function name, and `unwrapped_close` then closes what the
                // wrapper's close would have.
                if let Some(pos) = self.accumulated.find(open) {
                    self.inside = true;
                    self.unwrapped = true;
                    self.accumulated = self.accumulated[pos..].to_string();
                    return ToolEvent::Start;
                }
            }
            return ToolEvent::Start;
        }

        let close = if self.unwrapped { ch.unwrapped_close } else { ch.close };
        let Some(pos) = self.accumulated.find(close) else {
            return ToolEvent::Start;
        };
        // An unwrapped call needs its own closer INSIDE the body, because that
        // is what bounds the parameter list; a wrapped one is cut before it.
        let body_end = if self.unwrapped { pos + close.len() } else { pos };
        let body = self.accumulated[..body_end].trim().to_string();
        self.accumulated = self.accumulated[pos + close.len()..].to_string();
        self.inside = false;
        self.unwrapped = false;

        match (ch.parse_call)(&body, &self.schemas) {
            Some((name, args)) => ToolEvent::Call(name, args),
            None => ToolEvent::Start,
        }
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
        self.unwrapped = false;
    }
}

// ─── the one syntax that needs no family knowledge ───────────

/// `{"name": …, "arguments": {…}}` — the Hermes body.
///
/// Lives here rather than in a family crate because it reads a JSON object and
/// nothing about any model. The XML syntax does not: it types its arguments
/// from the request schemas, so it belongs with the family that speaks it.
/// Deliberately bug-for-bug with the decoder it replaces: a body that parses as
/// JSON but carries no `name` yields `("", …)` rather than `None`, because
/// `QwenToolDecoder` emitted `Call("", …)` there and a port that quietly tightens
/// behaviour is a port whose diff cannot be trusted. Worth fixing — as its own
/// change, with its own test.
pub fn parse_hermes_json(body: &str, _schemas: &[String]) -> Option<(String, String)> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let name = v["name"].as_str().unwrap_or("").to_string();
    let args = v["arguments"].to_string();
    Some((name, args))
}
