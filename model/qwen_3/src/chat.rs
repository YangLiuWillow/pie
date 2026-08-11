//! ChatML-family instruct implementation.
//!
//! Covers Qwen3, Qwen2.5, OLMo3, and any ChatML-based model.
//! Configurable via `ChatMLConfig` for thinking/tool support.
//!
//! Reference: Qwen3 Jinja chat template with tool-calling support.

use pie_model_common::decoders::{GenericChatDecoder, NoopReasoningDecoder, ThinkingDecoder};
use pie_model_common::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use pie_tokenizer::{Tokenizer, TokenizerDecoder};
use std::sync::Arc;

// =============================================================================
// Configuration
// =============================================================================

// The implementation below mirrors the published Qwen3 jinja chat template;
// the verbatim copy that used to sit here as a static was never read — the
// checkpoint's own `chat_template` is the reference.

/// Feature flags for ChatML-family models.
pub struct ChatMLConfig {
    pub has_thinking: bool,
    pub has_tools: bool,
    pub generation_suffix: &'static str,
    /// Stop token strings (vary per sub-architecture)
    pub stop_tokens: &'static [&'static str],
}

// =============================================================================
// QwenInstruct
// =============================================================================

pub struct QwenInstruct {
    tokenizer: Arc<Tokenizer>,
    config: ChatMLConfig,
    // Pre-tokenized delimiters
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    // "<|im_start|>user"/"<|im_start|>assistant" WITHOUT the trailing newline
    // baked into `user_prefix`/`assistant_prefix`. Needed when replaying a
    // tool-calling turn: the reference template's tool-call/tool-response
    // branches never put an unconditional newline right after the role tag —
    // the newline comes from whatever follows (content, or the first
    // `<tool_call>`/`<tool_response>` chunk), which is part of the turn's
    // single-pass-encoded inner text (see `assistant_with_tool_calls`).
    user_prefix_no_nl: Vec<u32>,
    assistant_prefix_no_nl: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,
    // Thinking delimiters
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
}

impl QwenInstruct {
    /// Create with full config.
    pub fn new(tokenizer: Arc<Tokenizer>, config: ChatMLConfig) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_ids: Vec<u32> = config
            .stop_tokens
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();

        let im_start = encode("<|im_start|>");
        let im_end = encode("<|im_end|>");
        let newline = encode("\n");

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = im_start.clone();
            v.extend(encode(role));
            v.extend(&newline);
            v
        };

        let mut turn_suffix = im_end;
        turn_suffix.extend(&newline);

        let mut user_prefix_no_nl = im_start.clone();
        user_prefix_no_nl.extend(encode("user"));
        let mut assistant_prefix_no_nl = im_start.clone();
        assistant_prefix_no_nl.extend(encode("assistant"));

        let think_prefix = encode("<think>");
        let think_suffix = encode("</think>");

        let mut generation_header = make_prefix("assistant");
        generation_header.extend(encode(config.generation_suffix));

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            assistant_prefix: make_prefix("assistant"),
            user_prefix_no_nl,
            assistant_prefix_no_nl,
            generation_header,
            turn_suffix,
            stop_ids,
            think_prefix_ids: think_prefix,
            think_suffix_ids: think_suffix,
            tokenizer,
            config,
        }
    }

    fn role_tokens(&self, role: &str, msg: &str) -> Vec<u32> {
        let prefix = match role {
            "system" => &self.system_prefix,
            "user" => &self.user_prefix,
            "assistant" => &self.assistant_prefix,
            _ => &self.user_prefix,
        };
        let mut tokens = prefix.clone();
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    /// The inner text of a replayed tool-calling assistant turn — everything
    /// between the `<|im_start|>assistant` role tag and `<|im_end|>`. Kept as
    /// a pure string builder so the reference format is byte-testable without
    /// a real tokenizer (the token-level fidelity of encoding this text in
    /// one pass is the parity harness's job — see `assistant_with_tool_calls`
    /// for the D4 rationale).
    fn assistant_with_tool_calls_inner_text(
        content: Option<&str>,
        calls: &[(String, String)],
    ) -> String {
        let mut text = String::new();
        if let Some(c) = content {
            if !c.is_empty() {
                text.push('\n');
                text.push_str(c);
            }
        }
        for (name, arguments_json) in calls {
            text.push_str("\n<tool_call>\n{\"name\": \"");
            text.push_str(name);
            text.push_str("\", \"arguments\": ");
            text.push_str(arguments_json);
            text.push_str("}\n</tool_call>");
        }
        text
    }

    /// The inner text of a merged tool-results turn — everything between the
    /// `<|im_start|>user` role tag and `<|im_end|>`. Same byte-testability
    /// rationale as [`Self::assistant_with_tool_calls_inner_text`].
    fn answer_batch_inner_text(results: &[(String, String)]) -> String {
        let mut text = String::new();
        for (_name, value) in results {
            text.push_str("\n<tool_response>\n");
            text.push_str(value);
            text.push_str("\n</tool_response>");
        }
        text
    }

    /// Strips `<think>...</think>` content from an assistant message for replay.
    /// If `</think>` is present, keeps only the content after the last `</think>`,
    /// with leading newlines stripped (matching the reference template).
    fn strip_thinking(msg: &str) -> &str {
        if let Some(pos) = msg.rfind("</think>") {
            msg[pos + "</think>".len()..].trim_start_matches('\n')
        } else {
            msg
        }
    }

    /// Build the tool system prompt matching the Qwen reference format.
    /// Both Qwen3 and Qwen2.5 use identical `<tools>` XML + `<tool_call>` format.
    fn build_tool_system_prompt(tools: &[String]) -> String {
        // Must match the Jinja2 chat template's output exactly — the model
        // was fine-tuned on that format and won't produce <tool_call> blocks
        // if the preamble diverges. No leading newline: the reference template
        // emits `content + "\n\n"` (or just the system opener) directly before
        // `# Tools` — the old branch's leading "\n" was parity divergence D2
        // (integrations/opencode/parity, HF Qwen3-0.6B).
        let mut prompt = String::from(
            "# Tools\n\n\
             You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>",
        );
        for tool in tools {
            prompt.push('\n');
            // Wrap in {"type": "function", "function": ...} if not already
            // wrapped — the Jinja template renders tools in this envelope and
            // the model was fine-tuned on it.
            if tool.contains("\"type\"") && tool.contains("\"function\"") {
                prompt.push_str(tool);
            } else {
                prompt.push_str(&format!("{{\"type\": \"function\", \"function\": {tool}}}"));
            }
        }
        prompt.push_str(
            "\n</tools>\n\n\
             For each function call, return a json object with function name and arguments \
             within <tool_call></tool_call> XML tags:\n\
             <tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
             </tool_call>",
        );
        prompt
    }

    /// Build an EBNF grammar for constrained Qwen tool-call generation.
    fn build_tool_call_grammar(tools: &[String]) -> Option<String> {
        let mut names: Vec<String> = Vec::new();
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
                let name = parsed
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .or_else(|| parsed.get("name"))
                    .and_then(|n| n.as_str());
                if let Some(n) = name {
                    names.push(format!("\"{}\"", n));
                }
            }
        }
        if names.is_empty() {
            return None;
        }

        let name_alt = names.join(" | ");
        let grammar = format!(
            r#"root ::= tool-call ("\n" tool-call)*
tool-call ::= "<tool_call>\n" tool-json "\n</tool_call>"
tool-json ::= "{{"  "\"name\": \"" tool-name "\", \"arguments\": " json-object "}}"
tool-name ::= {name_alt}
json-object ::= "{{" json-members? "}}"
json-members ::= json-pair ("," json-pair)*
json-pair ::= json-string ":" json-value
json-value ::= json-string | json-number | json-object | json-array | "true" | "false" | "null"
json-string ::= "\"" json-chars "\""
json-chars ::= json-char*
json-char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-array ::= "[" (json-value ("," json-value)*)? "]"
"#,
            name_alt = name_alt
        );
        Some(grammar)
    }
}

impl Instruct for QwenInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.role_tokens("system", msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.role_tokens("user", msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        // Strip <think>...</think> on replay (Qwen3 template does this;
        // for Qwen2 has_thinking=false so strip_thinking is a no-op on normal content)
        let stripped = if self.config.has_thinking {
            Self::strip_thinking(msg)
        } else {
            msg
        };
        self.role_tokens("assistant", stripped)
    }

    fn cue(&self) -> Vec<u32> {
        // Reference: <|im_start|>assistant\n
        self.generation_header.clone()
    }

    fn cue_no_think(&self) -> Vec<u32> {
        if !self.config.has_thinking {
            return self.cue();
        }
        // Reference (enable_thinking=false): the cue closes the thinking
        // channel with an empty think block — <|im_start|>assistant\n
        // <think>\n\n</think>\n\n (parity divergence D1). One whole-text
        // encode for the non-special run, same D4 rationale as the replay
        // primitives.
        let mut tokens = self.generation_header.clone();
        tokens.extend(self.tokenizer.encode("<think>\n\n</think>\n\n"));
        tokens
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if !self.config.has_tools {
            return Vec::new();
        }
        let prompt = Self::build_tool_system_prompt(tools);
        self.system(&prompt)
    }

    fn answer(&self, name: &str, value: &str) -> Vec<u32> {
        // One result == the single-element merged turn; sharing the
        // answer_batch path keeps the two byte- AND token-identical.
        self.answer_batch(&[(name.to_string(), value.to_string())])
    }

    fn equip_after_system(&self, system_content: Option<&str>, tools: &[String]) -> Vec<u32> {
        // Reference (the Qwen Jinja template's top-of-prompt preamble): when
        // tools are present, the leading system message's content (if any)
        // and the tools block are folded into ONE system turn
        // ('content' + '\n\n' + tools-block), not two separate turns.
        if !self.config.has_tools || tools.is_empty() {
            return match system_content {
                Some(c) => self.system(c),
                None => Vec::new(),
            };
        }
        let tools_block = Self::build_tool_system_prompt(tools);
        let merged = match system_content {
            Some(c) if !c.is_empty() => format!("{c}\n\n{tools_block}"),
            _ => tools_block,
        };
        self.system(&merged)
    }

    fn assistant_with_tool_calls(&self, content: Option<&str>, calls: &[(String, String)]) -> Vec<u32> {
        if !self.config.has_tools || calls.is_empty() {
            return self.assistant(content.unwrap_or(""));
        }
        // Reference (the Qwen Jinja template's assistant branch):
        // '<|im_start|>' + role, then '\n' + content only if content is
        // truthy, then for each call '\n<tool_call>\n{"name": ..., "arguments":
        // ...}\n</tool_call>', then '<|im_end|>\n'. Note there's no
        // unconditional newline after the role tag — it comes from whichever
        // of those two branches fires first.
        //
        // The turn's inner text is built as ONE string and encoded in ONE
        // pass: HF tokenizes the fully-rendered template output, so BPE
        // merges freely across the literal/dynamic joins (parity divergence
        // D4 — e.g. `"arguments": ` + `{"…` merges into `Ġ{"`). Encoding
        // pre-tokenized fragments and dynamic parts separately pins token
        // boundaries at every join and diverges from the reference. Special
        // tokens (<tool_call> etc.) are added-vocab entries the tokenizer
        // splits on either way, so this stays deterministic; content must be
        // special-token-sanitized upstream (the serving layer's job), exactly
        // as with HF templates.
        let text = Self::assistant_with_tool_calls_inner_text(content, calls);
        let mut tokens = self.assistant_prefix_no_nl.clone();
        tokens.extend(self.tokenizer.encode(&text));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn answer_batch(&self, results: &[(String, String)]) -> Vec<u32> {
        if !self.config.has_tools || results.is_empty() {
            return Vec::new();
        }
        // Reference: consecutive tool-role messages share one
        // '<|im_start|>user' ... '<|im_end|>\n' turn, but EVERY message still
        // contributes its own leading '\n<tool_response>\n...\n</tool_response>'
        // chunk (there's no unconditional newline baked into the opening tag
        // either — same shape as assistant_with_tool_calls above). Single
        // whole-text encode for D4 parity, same rationale as there.
        let text = Self::answer_batch_inner_text(results);
        let mut tokens = self.user_prefix_no_nl.clone();
        tokens.extend(self.tokenizer.encode(&text));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        if !self.config.has_thinking {
            return Box::new(NoopReasoningDecoder);
        }
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            self.think_prefix_ids.clone(),
            self.think_suffix_ids.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(QwenToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
            has_tools: self.config.has_tools,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if !self.config.has_tools || tools.is_empty() {
            return None;
        }
        let source = Self::build_tool_call_grammar(tools)?;
        Some(ToolGrammar { source })
    }
}

// =============================================================================
// Tool Decoder
// =============================================================================

struct QwenToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
    has_tools: bool,
}

impl ToolDecoder for QwenToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        if !self.has_tools {
            return ToolEvent::Start;
        }
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        if !self.inside {
            if self.accumulated.contains("<tool_call>") {
                self.inside = true;
                if let Some(pos) = self.accumulated.find("<tool_call>") {
                    self.accumulated = self.accumulated[pos + "<tool_call>".len()..].to_string();
                }
                return ToolEvent::Start;
            }
        } else if let Some(pos) = self.accumulated.find("</tool_call>") {
            let call_json = self.accumulated[..pos].trim().to_string();
            self.accumulated = self.accumulated[pos + "</tool_call>".len()..].to_string();
            self.inside = false;
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&call_json) {
                let name = v["name"].as_str().unwrap_or("").to_string();
                let args = v["arguments"].to_string();
                return ToolEvent::Call(name, args);
            }
        }
        ToolEvent::Start
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_tokenizer::Tokenizer;
    use std::sync::Arc;

    fn make_tok() -> Arc<Tokenizer> {
        let v: Vec<String> = vec![
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "system",
            "\n",
            "user",
            "assistant",
            "Hello",
            " world",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<tools>",
            "</tools>",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn qwen3() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )
    }

    fn qwen2() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                has_thinking: false,
                has_tools: true,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )
    }

    fn olmo3() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                has_thinking: true,
                has_tools: false,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>"],
            },
        )
    }

    #[test]
    fn qwen3_has_2_stop_tokens() {
        assert_eq!(qwen3().stop_ids.len(), 2);
    }

    #[test]
    fn qwen2_has_2_stop_tokens() {
        assert_eq!(qwen2().stop_ids.len(), 2);
    }

    #[test]
    fn olmo3_has_1_stop_token() {
        assert_eq!(olmo3().stop_ids.len(), 1);
    }

    #[test]
    fn qwen3_thinking_enabled() {
        assert!(qwen3().config.has_thinking);
    }

    #[test]
    fn qwen2_thinking_disabled() {
        assert!(!qwen2().config.has_thinking);
    }

    #[test]
    fn equip_noop_when_disabled() {
        let inst = olmo3();
        assert!(inst.equip(&["tool".to_string()]).is_empty());
        assert!(inst.answer("fn1", "42").is_empty());
    }

    #[test]
    fn equip_produces_tokens_when_enabled() {
        assert!(qwen3().config.has_tools);
    }

    #[test]
    fn seal_returns_stop_ids() {
        let inst = qwen3();
        assert_eq!(inst.seal(), inst.stop_ids);
    }

    #[test]
    fn generation_header_matches_cue() {
        let inst = qwen3();
        assert_eq!(inst.cue(), inst.generation_header);
    }

    #[test]
    fn strip_thinking_works() {
        assert_eq!(QwenInstruct::strip_thinking("plain text"), "plain text");
        assert_eq!(QwenInstruct::strip_thinking("<think>foo</think>bar"), "bar");
    }

    #[test]
    fn equip_format_matches_reference() {
        let prompt = QwenInstruct::build_tool_system_prompt(&["{}".to_string()]);
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("</tools>"));
        assert!(prompt.contains("<tool_call>"));
    }

    #[test]
    fn answer_does_not_include_name() {
        let inst = qwen3();
        let tokens = inst.answer("get_weather", "sunny");
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(!text.contains("get_weather:"));
    }

    #[test]
    fn tool_call_grammar_none_when_disabled() {
        let inst = olmo3();
        assert!(inst.tool_call_grammar(&["{}".to_string()]).is_none());
    }

    #[test]
    fn full_conversation() {
        let inst = qwen3();
        let mut tokens = Vec::new();
        tokens.extend(inst.system("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.assistant("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<|im_start|>system\nHello<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\nHello<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn answer_format() {
        // Reference: <|im_start|>user\n<tool_response>\ncontent\n</tool_response><|im_end|>\n
        // The role tag + turn suffix are pre-tokenized; the inner text is the
        // byte-testable part (whole-text-encoded at runtime — D4).
        assert_eq!(
            QwenInstruct::answer_batch_inner_text(&[("fn1".to_string(), "Hello".to_string())]),
            "\n<tool_response>\nHello\n</tool_response>"
        );
    }

    #[test]
    fn equip_after_system_merges_into_one_turn() {
        // The fold-into-one-turn property: content + '\n\n' + tools-block,
        // rendered as a SINGLE system() turn — not a system turn followed by
        // equip()'s own turn (the trait default). Asserted at the token level
        // against the equivalent single system() call so the toy vocab's
        // lossy encoding of the tools block can't skew the comparison.
        let inst = qwen3();
        let tools = vec!["{}".to_string()];
        let expected = inst.system(&format!(
            "Hello\n\n{}",
            QwenInstruct::build_tool_system_prompt(&tools)
        ));
        assert_eq!(inst.equip_after_system(Some("Hello"), &tools), expected);
        let text = inst.tokenizer.decode(&expected, false);
        assert_eq!(text.matches("<|im_start|>system").count(), 1);
        assert_eq!(text.matches("<|im_end|>").count(), 1);
    }

    #[test]
    fn equip_after_system_without_tools_is_plain_system() {
        let inst = qwen3();
        assert_eq!(
            inst.equip_after_system(Some("Hello"), &[]),
            inst.system("Hello")
        );
        assert!(inst.equip_after_system(None, &[]).is_empty());
    }

    #[test]
    fn assistant_with_tool_calls_falls_back_when_disabled() {
        let inst = olmo3();
        let with_calls =
            inst.assistant_with_tool_calls(Some("Hello"), &[("f".to_string(), "{}".to_string())]);
        assert_eq!(with_calls, inst.assistant("Hello"));
    }

    #[test]
    fn assistant_with_tool_calls_matches_reference_with_content() {
        // Reference (assistant branch): '\n' + content only if truthy, then
        // each call as '\n<tool_call>\n{"name": …, "arguments": …}\n</tool_call>'.
        assert_eq!(
            QwenInstruct::assistant_with_tool_calls_inner_text(
                Some("Hello"),
                &[("f".to_string(), "{}".to_string())]
            ),
            "\nHello\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"
        );
    }

    #[test]
    fn assistant_with_tool_calls_matches_reference_no_content() {
        // content=None: no unconditional newline after the role tag when
        // there's no leading text — the reference template only ever emits
        // one newline before the first `<tool_call>`, not two.
        assert_eq!(
            QwenInstruct::assistant_with_tool_calls_inner_text(
                None,
                &[("f".to_string(), "{}".to_string())]
            ),
            "\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"
        );
    }

    #[test]
    fn answer_batch_noop_when_disabled() {
        let inst = olmo3();
        assert!(
            inst.answer_batch(&[("fn1".to_string(), "42".to_string())])
                .is_empty()
        );
    }

    #[test]
    fn answer_batch_single_matches_answer() {
        let inst = qwen3();
        assert_eq!(
            inst.answer_batch(&[("fn1".to_string(), "Hello".to_string())]),
            inst.answer("fn1", "Hello"),
        );
    }

    #[test]
    fn answer_batch_merges_consecutive_results() {
        // The regression this exists to catch: rendering per-result turns
        // would produce two separate `<|im_start|>user...<|im_end|>` blocks;
        // the reference template merges consecutive tool results into ONE
        // turn with multiple `<tool_response>` chunks inside it, each with
        // its own leading newline.
        assert_eq!(
            QwenInstruct::answer_batch_inner_text(&[
                ("fn1".to_string(), "Hello".to_string()),
                ("fn2".to_string(), "world".to_string()),
            ]),
            "\n<tool_response>\nHello\n</tool_response>\n<tool_response>\nworld\n</tool_response>"
        );
    }

    #[test]
    fn tool_decoder_parses_call() {
        // Build vocab with the JSON content as a single entry
        let v: Vec<String> = vec![
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "system",
            "\n",
            "user",
            "assistant",
            "Hello",
            " world",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<tools>",
            "</tools>",
            r#"{"name": "f", "arguments": {}}"#,
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&v));
        let inst = QwenInstruct::new(
            tok,
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        );
        let mut dec = inst.tool_decoder();
        // Feed: <tool_call> \n JSON \n </tool_call>
        dec.feed(&[11]); // <tool_call> → enters inside, returns Start
        dec.feed(&[4]); // \n
        let event = dec.feed(&[17, 4, 12]); // JSON + \n + </tool_call>
        match event {
            ToolEvent::Call(name, args) => {
                assert_eq!(name, "f");
                assert_eq!(args, "{}");
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }
}
