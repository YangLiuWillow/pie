//! The Qwen families' output formats, as three `const` rows.
//!
//! This is the declaration half of [`pie_model_common::parser_engine`]: the
//! engine holds the state machine, these hold what makes each family
//! different. Read against the checkpoints' own `chat_template.jinja`, which
//! is the only authority on what a checkpoint emits:
//!
//! ```text
//!   Qwen3-8B-4bit                       no `<function=`  -> Hermes JSON, thinks
//!   Qwen3-Coder-30B-A3B-Instruct-4bit      `<function=`  -> XML, does NOT think
//!   Qwen3.6-35B-A3B-4bit                   `<function=`  -> XML, thinks
//! ```
//!
//! That third row is the one the old design could not hold. `tool_dialect` and
//! `has_thinking` were both `is_coder_lineage`, on the rule that a checkpoint
//! with no thinking channel is the Coder release and the Coder release speaks
//! XML. Qwen3.6 is a thinking model that speaks XML, so the rule had to break
//! for it — and here it is simply a row with both columns set, needing no
//! predicate at all.
//!
//! Adding the next family is adding a row. If it is neither of these formats,
//! it is a row plus a `ParseCall`.

use pie_model_common::parser_engine::{
    ParserEngineConfig, ReasoningChannel, ToolChannel, parse_hermes_json,
};

use crate::chat::parse_coder_function_call;

const THINK: ReasoningChannel = ReasoningChannel {
    open: "<think>",
    close: "</think>",
    // NOT `starts_inside`. The Qwen cue closes an EMPTY think block
    // (`<think>\n\n</think>\n\n`) rather than leaving one open, so the stream
    // begins outside the channel and `<think>` is matched like any delimiter.
    starts_inside: false,
};

/// `<tool_call>{"name": …, "arguments": {…}}</tool_call>`.
///
/// No `unwrapped_open`: the Hermes body is a bare JSON object, so there is no
/// prefix that could identify a call that lost its wrapper. The Coder back-off
/// has `<function=` to key on; this has nothing.
const HERMES_TOOLS: ToolChannel = ToolChannel {
    open: "<tool_call>",
    close: "</tool_call>",
    unwrapped_open: None,
    unwrapped_close: "</tool_call>",
    parse_call: parse_hermes_json,
};

/// `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`.
const XML_TOOLS: ToolChannel = ToolChannel {
    open: "<tool_call>",
    close: "</tool_call>",
    unwrapped_open: Some("<function="),
    unwrapped_close: "</function>",
    parse_call: parse_coder_call,
};

/// Qwen3, Qwen3-30B-A3B, Qwen3-VL: thinking, Hermes tool calls.
pub const QWEN3: ParserEngineConfig = ParserEngineConfig {
    name: "qwen3",
    reasoning: Some(THINK),
    tools: Some(HERMES_TOOLS),
};

/// Qwen3-Coder: XML tool calls, and NO thinking channel — cueing it with an
/// empty think block produces an immediate EOS.
pub const QWEN3_CODER: ParserEngineConfig = ParserEngineConfig {
    name: "qwen3_coder",
    reasoning: None,
    tools: Some(XML_TOOLS),
};

/// Qwen3.5 / Qwen3.6: thinking AND XML tool calls.
pub const QWEN3_5: ParserEngineConfig = ParserEngineConfig {
    name: "qwen3_5",
    reasoning: Some(THINK),
    tools: Some(XML_TOOLS),
};

/// Strip the `<function=…></function>` shell, then read the parameters.
///
/// The engine hands over the whole matched body; for a wrapped call that is
/// what sat inside `<tool_call>`, and for an unwrapped one it is the
/// `<function=…></function>` itself. Both reduce to the same inner text, so
/// the unwrapping happens once, here, rather than as a special case in the
/// engine.
fn parse_coder_call(body: &str, schemas: &[String]) -> Option<(String, String)> {
    const OPEN: &str = "<function=";
    let fs = body.find(OPEN)?;
    let after = &body[fs + OPEN.len()..];
    let inner = match after.find("</function>") {
        Some(fe) => &after[..fe],
        None => after,
    };
    parse_coder_function_call(inner, schemas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_model_common::instruct::{ReasoningEvent, ToolEvent};
    use pie_tokenizer::Tokenizer;
    use std::sync::Arc;

    /// Every text the tests feed. The vocabulary is built to cover exactly
    /// this, because a piece the vocabulary cannot spell is silently dropped by
    /// `from_vocab`'s raw-char pipeline — which reads as "the decoder missed
    /// the call" and sends you hunting in the decoder.
    const CORPUS: &[&str] = &[
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"}}</tool_call>",
        "<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>",
        "<function=read_file><parameter=path>README.md</parameter></function>",
        "<think>thinking</think>hello",
    ];

    const SPECIALS: &[&str] = &[
        "<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "</function>",
        "<parameter=", "</parameter>",
    ];

    /// The delimiters as single tokens — which is what the real vocabularies do,
    /// and what the reasoning channel's id matching depends on — over a
    /// character-level tail that can spell everything else.
    fn tok() -> Arc<Tokenizer> {
        let mut pieces: Vec<String> = SPECIALS.iter().map(|s| s.to_string()).collect();
        let mut chars: Vec<char> = CORPUS.iter().flat_map(|s| s.chars()).collect();
        chars.sort_unstable();
        chars.dedup();
        pieces.extend(chars.into_iter().map(String::from));
        Arc::new(Tokenizer::from_vocab(&pieces))
    }

    /// Encode the way a REAL vocabulary does: each special as its own id, the
    /// rest character-wise.
    ///
    /// `from_vocab` builds a raw-char pipeline, so `encode` on a whole response
    /// spells `<think>` out as seven characters even though the vocabulary
    /// holds it as one piece. The reasoning channel matches on ids, so that
    /// fixture would never fire the delimiter and the test would fail against a
    /// decoder that is correct. Splicing the specials in reproduces the stream
    /// the engine actually receives.
    fn encode_like_a_real_vocab(t: &Tokenizer, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for (i, sp) in SPECIALS.iter().enumerate() {
                if rest.starts_with(sp) {
                    ids.push(i as u32);
                    rest = &rest[sp.len()..];
                    continue 'outer;
                }
            }
            let ch = rest.chars().next().expect("non-empty");
            ids.extend(t.encode(&ch.to_string()));
            rest = &rest[ch.len_utf8()..];
        }
        ids
    }

    fn feed_text(cfg: ParserEngineConfig, text: &str) -> Option<(String, String)> {
        let t = tok();
        assert!(CORPUS.contains(&text), "add {text:?} to CORPUS or it cannot be spelled");
        let (_, mut tools) = cfg.decoders(t.clone(), true, vec![]);
        // One token at a time: a decoder that only works on whole-response
        // slabs is not a streaming decoder, and the delimiters straddle chunk
        // boundaries in exactly the way that breaks.
        for id in encode_like_a_real_vocab(&t, text) {
            if let ToolEvent::Call(n, a) = tools.feed(&[id]) {
                return Some((n, a));
            }
        }
        None
    }

    #[test]
    fn hermes_body_is_read_as_json() {
        let got = feed_text(
            QWEN3,
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"}}</tool_call>",
        );
        let (name, args) = got.expect("no call decoded");
        assert_eq!(name, "read_file");
        assert!(args.contains("README.md"), "arguments lost: {args}");
    }

    #[test]
    fn xml_body_is_read_as_parameters() {
        let got = feed_text(
            QWEN3_5,
            "<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>",
        );
        let (name, args) = got.expect("no call decoded");
        assert_eq!(name, "read_file");
        assert!(args.contains("README.md"), "arguments lost: {args}");
    }

    /// The back-off that keeps a real Qwen3-Coder call from being dropped.
    #[test]
    fn an_xml_call_without_its_wrapper_still_decodes() {
        for cfg in [QWEN3_CODER, QWEN3_5] {
            let got = feed_text(
                cfg,
                "<function=read_file><parameter=path>README.md</parameter></function>",
            );
            let (name, _) = got.unwrap_or_else(|| panic!("{}: unwrapped call dropped", cfg.name));
            assert_eq!(name, "read_file", "{}", cfg.name);
        }
    }

    /// Hermes has no such back-off, and must not grow one by accident: there is
    /// no prefix in a bare JSON object to key on, so a "call" without its
    /// wrapper is indistinguishable from prose.
    #[test]
    fn hermes_has_no_unwrapped_back_off() {
        assert!(QWEN3.tools.unwrap().unwrapped_open.is_none());
    }

    /// The row the single predicate could not express.
    #[test]
    fn qwen3_5_thinks_and_speaks_xml() {
        assert!(QWEN3_5.reasoning.is_some(), "Qwen3.6 is a thinking model");
        assert_eq!(QWEN3_5.tools.unwrap().unwrapped_open, Some("<function="));
        // And its neighbours keep the pairings that made the old rule seem true.
        assert!(QWEN3_CODER.reasoning.is_none());
        assert_eq!(QWEN3_CODER.tools.unwrap().unwrapped_open, Some("<function="));
        assert!(QWEN3.reasoning.is_some());
        assert!(QWEN3.tools.unwrap().unwrapped_open.is_none());
    }

    #[test]
    fn a_family_with_no_thinking_channel_never_opens_one() {
        let t = tok();
        let (mut reasoning, _) = QWEN3_CODER.decoders(t.clone(), true, vec![]);
        for id in encode_like_a_real_vocab(&t, "<think>thinking</think>hello") {
            match reasoning.feed(&[id]) {
                ReasoningEvent::Delta(d) => assert!(d.is_empty(), "Coder emitted reasoning: {d:?}"),
                other => panic!("Coder opened a thinking channel: {other:?}"),
            }
        }
    }

    #[test]
    fn thinking_block_is_captured_then_closed() {
        let t = tok();
        let (mut reasoning, _) = QWEN3_5.decoders(t.clone(), true, vec![]);
        let mut complete = None;
        for id in encode_like_a_real_vocab(&t, "<think>thinking</think>hello") {
            if let ReasoningEvent::Complete(text) = reasoning.feed(&[id]) {
                complete = Some(text);
            }
        }
        assert_eq!(complete.as_deref(), Some("thinking"));
    }

    // ── the port is a port, not a rewrite ────────────────────────────

    /// The same token stream through the OLD hand-written decoders and the NEW
    /// table-driven ones, event for event.
    ///
    /// This is the only test here that would notice a behaviour change, and it
    /// is the reason the Hermes parser keeps its `("", …)` quirk: an
    /// improvement smuggled in during a port makes the port unreviewable.
    fn old_decoders(
        thinking: bool,
        dialect: crate::chat::ToolDialect,
        t: Arc<Tokenizer>,
    ) -> crate::chat::QwenInstruct {
        crate::chat::QwenInstruct::new(
            t,
            crate::chat::ChatMLConfig {
                has_thinking: thinking,
                has_tools: true,
                tool_dialect: dialect,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )
    }

    #[test]
    fn new_engine_matches_the_decoders_it_replaces() {
        use crate::chat::ToolDialect;
        use pie_model_common::instruct::Instruct;

        let rows = [
            (QWEN3, true, ToolDialect::Hermes),
            (QWEN3_CODER, false, ToolDialect::Coder),
            (QWEN3_5, true, ToolDialect::Coder),
        ];
        for (cfg, thinking, dialect) in rows {
            for text in CORPUS {
                let t = tok();
                let ids = encode_like_a_real_vocab(&t, text);

                let inst = old_decoders(thinking, dialect, t.clone());
                let mut old_tools = inst.tool_decoder_with_tools(&[]);
                let mut old_reason = inst.reasoning_decoder();
                let (mut new_reason, mut new_tools) = cfg.decoders(t.clone(), true, vec![]);

                for id in &ids {
                    let (o, n) = (old_tools.feed(&[*id]), new_tools.feed(&[*id]));
                    assert_eq!(
                        format!("{o:?}"),
                        format!("{n:?}"),
                        "{} tool event diverged on {text:?}",
                        cfg.name
                    );
                    let (o, n) = (old_reason.feed(&[*id]), new_reason.feed(&[*id]));
                    assert_eq!(
                        format!("{o:?}"),
                        format!("{n:?}"),
                        "{} reasoning event diverged on {text:?}",
                        cfg.name
                    );
                }
            }
        }
    }
}
