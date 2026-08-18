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

/// How a checkpoint spells tool schemas and tool calls.
///
/// Both dialects wrap a call in `<tool_call>`, and that shared tag is what
/// makes getting this wrong so quiet: the model emits *something* either way.
/// What differs is everything inside it, and a model prompted in one dialect
/// answers in that dialect no matter which one the parser expects.
///
/// This is not a preference — it is a property of what the checkpoint was
/// fine-tuned on. Qwen3-Coder handed the Hermes preamble replies with a bare
/// `{"name": …, "arguments": {…}}` and no wrapper at all, so a decoder looking
/// for `<tool_call>` finds nothing, `tool_calls` comes back `null`, and a
/// tool-driven agent sees a wall of text where it expected a call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolDialect {
    /// Qwen3: JSON schemas in `<tools>`, and a call is one JSON object inside
    /// `<tool_call>`.
    Hermes,
    /// Qwen3-Coder: XML schemas in `<tools>`, and a call is nested XML —
    /// `<tool_call><function=name><parameter=k>v</parameter></function>`.
    /// Arguments arrive as *strings* and are typed from the schema on the way
    /// out, because XML carries no types.
    Coder,
    /// Qwen3.5 and later, including the `qwen3_5`-architected Qwen3.6: a hybrid.
    ///
    /// The SCHEMAS are JSON, exactly as Hermes writes them — the template does
    /// `{{- tool | tojson }}` inside `<tools>`. The CALLS are Coder's nested
    /// XML. Serving it as `Coder` renders `<function><name>` schema blocks the
    /// checkpoint's template never writes; serving it as `Hermes` instructs a
    /// call format the model was not trained to emit. It is neither, so it is
    /// its own row.
    Qwen35Xml,
}

impl ToolDialect {
    /// Does a *call* go out (and come back) as nested XML?
    ///
    /// Asked as a question rather than compared with `==`, because two
    /// dialects now answer yes and a `== ToolDialect::Coder` left behind at any
    /// one of the five dispatch sites renders a call in one format and parses
    /// it in another — silently, on Qwen3.6 only.
    pub fn emits_xml_calls(self) -> bool {
        matches!(self, ToolDialect::Coder | ToolDialect::Qwen35Xml)
    }
}

/// What the Qwen3.5+ `chat_template` says after the `<tools>` block, verbatim.
///
/// A raw literal, not a `\`-continued one: every space and blank line here is
/// the checkpoint's, and a continuation would let rustfmt's indentation into a
/// prompt whose bytes are the contract.
///
/// Transcribed from `mlx-community/Qwen3.6-35B-A3B-4bit`'s own
/// `chat_template.jinja` (line 53) and diffed against it — 815 characters,
/// identical to the template's text once the two leading newlines the caller
/// supplies are removed.
const QWEN35_XML_CALL_INSTRUCTION: &str = r#"If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"#;

/// Feature flags for ChatML-family models.
pub struct ChatMLConfig {
    pub has_thinking: bool,
    pub has_tools: bool,
    pub tool_dialect: ToolDialect,
    /// Does the caller's system message lead the tools turn, or follow it?
    ///
    /// Every Qwen template folds both into ONE system turn; which half leads
    /// is the template's to say, and they disagree. Read from the checkpoints:
    ///
    /// ```text
    ///   Qwen3-8B          system_message, then "# Tools"          -> true
    ///   Qwen3-Coder-30B   system_message, then "You have access"  -> true
    ///   Qwen3.6-35B       "# Tools" block, then system_message    -> false
    /// ```
    ///
    /// A pure reordering: same tokens, same count, different order. Nothing
    /// that measures length can see it, which is why serving Qwen3.6 with
    /// Qwen3's order survived until a positional diff went looking.
    pub system_before_tools: bool,
    /// Does a post-query assistant turn carry a reasoning block even when its
    /// reasoning is EMPTY?
    ///
    /// Qwen3.5/3.6 writes one either way; Qwen3 writes one only when the turn
    /// carries reasoning. Four tokens per replayed turn, so an agent loop of
    /// twenty turns diverges by eighty — invisible to any single-turn check,
    /// which is where it hid.
    pub empty_reasoning_header: bool,
    pub generation_suffix: &'static str,
    /// The generation suffix when the caller turns thinking OFF.
    ///
    /// A sibling of `generation_suffix` rather than a tweak to it, because the
    /// two are independent strings in the template:
    ///
    /// ```jinja
    ///   {%- if enable_thinking is defined and enable_thinking is false %}
    ///       {{- '<think>\n\n</think>\n\n' }}
    ///   {%- else %}
    ///       {{- '<think>\n' }}
    /// ```
    ///
    /// `cue_no_think` used to append a hardcoded `<think>\n\n</think>\n\n`
    /// to the generation header. That was right only while `generation_suffix`
    /// was empty; the moment a family sets one, the two concatenate and the
    /// cue carries BOTH — `<think>\n<think>\n\n</think>\n\n`. Ported from
    /// upstream dev-sslee, which pre-encodes the two headers separately.
    pub thinking_off_suffix: &'static str,
    /// Where the newline sits around a replayed `<tool_response>` block.
    ///
    /// The two families disagree, and each is transcribed from its own
    /// template:
    ///
    /// ```text
    ///   Qwen3 / Qwen3.5 / 3.6   '\n<tool_response>\n' … '\n</tool_response>'
    ///   Qwen3-Coder             '<tool_response>\n'   … '\n</tool_response>\n'
    /// ```
    ///
    /// Same characters, moved from the front to the back — so the turn is one
    /// token short rather than malformed, and it stayed invisible until a
    /// positional diff ran the Coder arm. Upstream cannot have seen this: its
    /// registry has no Qwen3-Coder row at all.
    pub tool_response_trailing_newline: bool,
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
    thinking_off_header: Vec<u32>,
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
        let mut thinking_off_header = make_prefix("assistant");
        thinking_off_header.extend(encode(config.thinking_off_suffix));

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            assistant_prefix: make_prefix("assistant"),
            user_prefix_no_nl,
            assistant_prefix_no_nl,
            generation_header,
            thinking_off_header,
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
        dialect: ToolDialect,
    ) -> String {
        let mut text = String::new();
        if let Some(c) = content {
            if !c.is_empty() {
                text.push('\n');
                text.push_str(c);
                // The Coder/Qwen3.5 templates trim the content and close it
                // with a newline of their own before the first call; Hermes
                // does not.
                if dialect.emits_xml_calls() {
                    text = format!("\n{}\n", c.trim());
                }
            }
        }
        for (name, arguments_json) in calls {
            if dialect.emits_xml_calls() {
                // Arguments go back out as the XML the model produced them in.
                // A value that was typed on the way IN (an int, a bool, an
                // object) is rendered as its plain text here, because that is
                // what the template does and what the model saw: XML has no
                // types, and re-quoting a number would replay a turn the model
                // never wrote.
                text.push_str(&format!("\n<tool_call>\n<function={name}>\n"));
                if let Ok(serde_json::Value::Object(args)) =
                    serde_json::from_str::<serde_json::Value>(arguments_json)
                {
                    for (k, v) in &args {
                        let body = match v.as_str() {
                            Some(s) => s.to_string(),
                            None => v.to_string(),
                        };
                        text.push_str(&format!("<parameter={k}>\n{body}\n</parameter>\n"));
                    }
                }
                text.push_str("</function>\n</tool_call>");
                continue;
            }
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
    fn answer_batch_inner_text(results: &[(String, String)], trailing_newline: bool) -> String {
        // Both shapes put exactly one newline between the role tag and the
        // first block; they differ in where the SEPARATOR lives, which is what
        // makes a multi-result turn diverge rather than just the first one.
        //
        //   Coder     '<|im_start|>user\n' then, per result,
        //             '<tool_response>\n' … '\n</tool_response>\n'
        //   Qwen3.x   '<|im_start|>user'   then, per result,
        //             '\n<tool_response>\n' … '\n</tool_response>'
        let mut text = String::new();
        if trailing_newline {
            text.push('\n');
        }
        for (_name, value) in results {
            if !trailing_newline {
                text.push('\n');
            }
            text.push_str("<tool_response>\n");
            text.push_str(value);
            text.push_str("\n</tool_response>");
            if trailing_newline {
                text.push('\n');
            }
        }
        text
    }

    /// Strips `<think>...</think>` content from an assistant message for replay.
    /// If `</think>` is present, keeps only the content after the last `</think>`,
    /// with leading newlines stripped (matching the reference template).
    fn strip_thinking(msg: &str) -> &str {
        Self::split_thinking(msg).1
    }

    /// `(reasoning, content)` — the same cut `strip_thinking` makes, keeping
    /// the half it throws away.
    ///
    /// The template does exactly this split and then chooses whether to render
    /// the reasoning half; discarding it here made that choice unavailable.
    fn split_thinking(msg: &str) -> (&str, &str) {
        let Some(close) = msg.rfind("</think>") else { return ("", msg) };
        let content = msg[close + "</think>".len()..].trim_start_matches('\n');
        let head = &msg[..close];
        let reasoning = match head.rfind("<think>") {
            Some(open) => &head[open + "<think>".len()..],
            None => head,
        };
        (reasoning.trim(), content)
    }

    /// A replayed assistant turn's body: reasoning block, then content.
    fn reasoning_header_for<'a>(&self, msg: &'a str, reasoning_header: bool) -> (String, &'a str) {
        let (reasoning, content) = if self.config.has_thinking {
            Self::split_thinking(msg)
        } else {
            ("", msg)
        };
        let renders = self.config.empty_reasoning_header || !reasoning.is_empty();
        if reasoning_header && self.config.has_thinking && renders {
            (format!("<think>\n{reasoning}\n</think>\n\n"), content)
        } else {
            (String::new(), content)
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

    /// The Qwen3.5+ tools turn: Hermes's JSON schemas under a different
    /// heading, followed by the XML call instruction.
    ///
    /// Transcribed from the checkpoint's `chat_template.jinja`:
    ///
    /// ```jinja
    /// {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    /// {%- for tool in tools %}{{- "\n" }}{{- tool | tojson }}{%- endfor %}
    /// {{- "\n</tools>" }}
    /// {{- '\n\nIf you choose to call a function ONLY reply ...' }}
    /// ```
    ///
    /// `tojson` serialises the tool as the caller passed it, which for an
    /// OpenAI request is the `{"type": "function", "function": …}` envelope —
    /// so the same envelope-if-missing rule as the Hermes builder applies, and
    /// for the same reason.
    fn build_tool_system_prompt_qwen35(tools: &[String]) -> String {
        let mut prompt =
            String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in tools {
            prompt.push('\n');
            if tool.contains("\"type\"") && tool.contains("\"function\"") {
                prompt.push_str(tool);
            } else {
                prompt.push_str(&format!("{{\"type\": \"function\", \"function\": {tool}}}"));
            }
        }
        prompt.push_str("\n</tools>\n\n");
        prompt.push_str(QWEN35_XML_CALL_INSTRUCTION);
        prompt
    }

    /// `render_item_list` from the Coder template: `[`a`, `b`]` for strings,
    /// bare for anything else, wrapped in a tag, and emitted only when the list
    /// is present and non-empty.
    fn coder_item_list(out: &mut String, list: Option<&serde_json::Value>, tag: &str) {
        let Some(items) = list.and_then(|v| v.as_array()) else { return };
        if items.is_empty() {
            return;
        }
        out.push_str(&format!("\n<{tag}>["));
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            match item.as_str() {
                Some(s) => out.push_str(&format!("`{s}`")),
                None => out.push_str(&item.to_string()),
            }
        }
        out.push_str(&format!("]</{tag}>"));
    }

    /// The Coder checkpoint's tool preamble, transcribed from its own
    /// `chat_template.jinja`.
    ///
    /// Transcribed rather than approximated: the model was fine-tuned on these
    /// exact bytes, and the whole failure this fixes is a preamble that looked
    /// reasonable and was not what training saw. The trailing `<IMPORTANT>`
    /// block is part of it — it is what tells the model to nest `<function=…>`
    /// inside `<tool_call>`, which is precisely the structure pie's decoder
    /// then looks for.
    fn build_tool_system_prompt_coder(tools: &[String]) -> String {
        let mut out = String::from("You have access to the following functions:\n\n<tools>");
        for tool in tools {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(tool) else { continue };
            // Accept either the bare function object or the OpenAI envelope.
            let f = v.get("function").unwrap_or(&v);
            let name = f.get("name").and_then(|x| x.as_str()).unwrap_or("");
            out.push_str(&format!("\n<function>\n<name>{name}</name>"));
            if let Some(d) = f.get("description").and_then(|x| x.as_str()) {
                out.push_str(&format!("\n<description>{}</description>", d.trim()));
            }
            out.push_str("\n<parameters>");
            let params = f.get("parameters");
            if let Some(props) = params.and_then(|p| p.get("properties")).and_then(|p| p.as_object())
            {
                for (pname, pf) in props {
                    out.push_str(&format!("\n<parameter>\n<name>{pname}</name>"));
                    if let Some(t) = pf.get("type") {
                        let t = t.as_str().map(str::to_string).unwrap_or_else(|| t.to_string());
                        out.push_str(&format!("\n<type>{t}</type>"));
                    }
                    if let Some(d) = pf.get("description").and_then(|x| x.as_str()) {
                        out.push_str(&format!("\n<description>{}</description>", d.trim()));
                    }
                    Self::coder_item_list(&mut out, pf.get("enum"), "enum");
                    // Any remaining schema key, tag-named after the template's
                    // normalisation. Mappings go out as JSON, scalars as text.
                    if let Some(obj) = pf.as_object() {
                        for (k, val) in obj {
                            if matches!(k.as_str(), "type" | "description" | "enum" | "required") {
                                continue;
                            }
                            let tag = k.replace(['-', ' '], "_").replace('$', "");
                            let body = if val.is_object() || val.is_array() {
                                val.to_string()
                            } else {
                                val.as_str().map(str::to_string).unwrap_or_else(|| val.to_string())
                            };
                            out.push_str(&format!("\n<{tag}>{body}</{tag}>"));
                        }
                    }
                    Self::coder_item_list(&mut out, pf.get("required"), "required");
                    out.push_str("\n</parameter>");
                }
            }
            Self::coder_item_list(&mut out, params.and_then(|p| p.get("required")), "required");
            out.push_str("\n</parameters>");
            if let Some(r) = f.get("return") {
                let body = if r.is_object() || r.is_array() {
                    r.to_string()
                } else {
                    r.as_str().map(str::to_string).unwrap_or_else(|| r.to_string())
                };
                out.push_str(&format!("\n<return>{body}</return>"));
            }
            out.push_str("\n</function>");
        }
        out.push_str("\n</tools>");
        out.push_str(
            "\n\nIf you choose to call a function ONLY reply in the following format with NO \
             suffix:\n\n\
             <tool_call>\n\
             <function=example_function_name>\n\
             <parameter=example_parameter_1>\n\
             value_1\n\
             </parameter>\n\
             <parameter=example_parameter_2>\n\
             This is the value for the second parameter\n\
             that can span\n\
             multiple lines\n\
             </parameter>\n\
             </function>\n\
             </tool_call>\n\n\
             <IMPORTANT>\n\
             Reminder:\n\
             - Function calls MUST follow the specified format: an inner \
             <function=...></function> block must be nested within <tool_call></tool_call> XML \
             tags\n\
             - Required parameters MUST be specified\n\
             - You may provide optional reasoning for your function call in natural language \
             BEFORE the function call, but NOT after\n\
             - If there is no function call available, answer the question like normal with your \
             current knowledge and do not tell the user about function calls\n\
             </IMPORTANT>",
        );
        out
    }

    /// The stand-in system turn the Coder template opens with when a request
    /// carries tools but no system message of its own.
    const CODER_DEFAULT_SYSTEM: &'static str =
        "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.";

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
        // Reference (enable_thinking=false): the template's own
        // `thinking_off_suffix`, pre-encoded beside the thinking-on header
        // rather than appended to it. Appending is what would double the cue
        // once a family sets `generation_suffix` — see the field's docs.
        self.thinking_off_header.clone()
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
        // The SCHEMA rendering is where the three dialects differ, and it does
        // not follow the call format: Qwen3.5 writes JSON schemas like Hermes
        // and XML calls like Coder.
        let coder = self.config.tool_dialect == ToolDialect::Coder;
        let tools_block = match self.config.tool_dialect {
            ToolDialect::Coder => Self::build_tool_system_prompt_coder(tools),
            ToolDialect::Qwen35Xml => Self::build_tool_system_prompt_qwen35(tools),
            ToolDialect::Hermes => Self::build_tool_system_prompt(tools),
        };
        let merged = match system_content {
            Some(c) if !c.is_empty() => {
                if self.config.system_before_tools {
                    format!("{c}\n\n{tools_block}")
                } else {
                    format!("{tools_block}\n\n{c}")
                }
            }
            // The Coder template does not open a bare tools turn: with tools
            // and no system message it emits its own stand-in system line
            // first, and the model saw that line in training.
            _ if coder => format!("{}\n\n{tools_block}", Self::CODER_DEFAULT_SYSTEM),
            _ => tools_block,
        };
        self.system(&merged)
    }

    fn assistant_at(&self, msg: &str, reasoning_header: bool) -> Vec<u32> {
        let (header, content) = self.reasoning_header_for(msg, reasoning_header);
        if header.is_empty() {
            return self.role_tokens("assistant", content);
        }
        // One string, one encode: BPE merges across the header/content join
        // the same way HF's does over its fully-rendered template output.
        self.role_tokens("assistant", &format!("{header}{content}"))
    }

    fn assistant_with_tool_calls_at(
        &self,
        content: Option<&str>,
        calls: &[(String, String)],
        reasoning_header: bool,
    ) -> Vec<u32> {
        if !self.config.has_tools || calls.is_empty() {
            return self.assistant_at(content.unwrap_or(""), reasoning_header);
        }
        let (header, _) = self.reasoning_header_for(content.unwrap_or(""), reasoning_header);
        if header.is_empty() {
            return self.assistant_with_tool_calls(content, calls);
        }
        // The header sits between the role tag and the turn's inner text, so
        // it joins the SAME single-pass encode the inner text already uses --
        // see the D4 note below for why that matters.
        let (_, stripped) = self.reasoning_header_for(content.unwrap_or(""), reasoning_header);
        let body = Self::assistant_with_tool_calls_inner_text(
            Some(stripped),
            calls,
            self.config.tool_dialect,
        );
        // `inner_text` opens with its own '\n' after the role tag; the header
        // replaces that, because the template writes '\n<think>' there.
        let body = body.strip_prefix('\n').unwrap_or(&body);
        let mut tokens = self.assistant_prefix_no_nl.clone();
        tokens.extend(self.tokenizer.encode(&format!("\n{header}{body}")));
        tokens.extend(&self.turn_suffix);
        tokens
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
        let text = Self::assistant_with_tool_calls_inner_text(content, calls, self.config.tool_dialect);
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
        let text = Self::answer_batch_inner_text(results, self.config.tool_response_trailing_newline);
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
        self.tool_decoder_with_tools(&[])
    }

    fn tool_decoder_with_tools(&self, tools: &[String]) -> Box<dyn ToolDecoder> {
        Box::new(QwenToolDecoder {
            decoder: self.tokenizer.decoder(false),
            accumulated: String::new(),
            inside: false,
            unwrapped: false,
            has_tools: self.config.has_tools,
            dialect: self.config.tool_dialect,
            schemas: tools.to_vec(),
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

/// Turn one Coder `<function=…>…</function>` body into `(name, arguments_json)`.
///
/// Transcribed from `qwen3coder_tool_parser.py`, which the CHECKPOINT ships and
/// vLLM loads as `--tool-call-parser qwen3_coder`. Two rules in it are not
/// guessable from the wire format and are the reason to follow the reference
/// rather than write a plausible XML reader:
///
///   * **XML carries no types.** Every parameter arrives as text, and the
///     schema is what decides whether `3` is the number 3 or the string "3".
///     Guessing by shape instead would send `{"timeout": 30}` to a tool whose
///     schema says `string`, and the mismatch surfaces as a tool error the
///     model then tries to reason about.
///   * **One leading and one trailing newline are part of the delimiter, not
///     the value.** They are stripped exactly once — a value that genuinely
///     ends in a blank line keeps the rest.
///
/// Unparseable values degrade to the raw string rather than dropping the call,
/// which is also what the reference does: a tool call with one odd argument is
/// worth more to an agent than no call at all.
/// Whether a captured name -- of a function or of a parameter -- could
/// plausibly be one.
///
/// A blocklist rather than an allowlist, deliberately: tool schemas are written
/// by whoever ships the tool, and a stricter rule than the defect requires
/// would silently refuse valid calls. These are the characters that only appear
/// when the name scan has run past its parameter and into the document --
/// whitespace, quotes, and the angle brackets and `=` of the markup itself.
fn is_plausible_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        // A closing tag read as an opener yields "/function" or "/parameter".
        // Upstream screens the same way; it matters once the opener recovery
        // below is willing to read a name out of a bare `<...>`.
        && !s.starts_with('/')
        && !s.chars().any(|c| {
            c.is_whitespace() || matches!(c, '<' | '>' | '=' | '"' | '\'')
        })
}

/// Locate the tag naming the function and return `(name, offset of its `>`)`.
///
/// Ported from upstream `dev-sslee` c1e0a84ab. The template teaches
/// `<function=NAME>` and that is tried first, but a model that closes with
/// `</function>` while opening some other way has still plainly named a
/// function, and refusing it throws away an action the model unambiguously
/// took. Two malformations were observed live from this checkpoint family:
///
/// ```text
///   <function>NAME          the tag is right, the name follows as text (3/29 calls)
///   <NAME>                  the bare form; vLLM's qwen3_xml accepts it
/// ```
///
/// Losing those costs the call and reports malformed syntax, which sends an
/// agent into re-issuing the action instead of acting on its result.
///
/// The recovered forms are accepted only where a false positive is
/// implausible: the tag must be the FIRST in the call, it must not be one of
/// the surface's own structural tags, and the CALLER must have found a
/// `</function>` close. Prose cannot reach here — this only ever sees the
/// inside of a `<tool_call>` block.
fn parse_xml_function_opener(call: &str) -> Option<(String, usize)> {
    const TAUGHT: &str = "<function=";
    if let Some(at) = call.find(TAUGHT) {
        let start = at + TAUGHT.len();
        let end = call[start..].find('>')? + start;
        return Some((call[start..end].trim().to_string(), end));
    }

    // The FIRST tag and nothing later, so a `<parameter=…>` deeper in the body
    // can never be mistaken for the function name.
    let open = call.find('<')?;
    let end = call[open + 1..].find('>')? + open + 1;
    let inner = call[open + 1..end].trim();

    // `<function>NAME`: read the name from after the tag, bounded by the first
    // whitespace or `<` so it cannot swallow the body.
    if inner == "function" {
        let name = call[end + 1..]
            .trim_start()
            .split(|c: char| c.is_whitespace() || c == '<')
            .next()
            .unwrap_or("");
        return is_plausible_name(name).then(|| (name.to_string(), end));
    }

    // Bare `<NAME>`, excluding the surface's own tags.
    if !is_plausible_name(inner) || matches!(inner, "tool_call" | "parameter") {
        return None;
    }
    Some((inner.to_string(), end))
}

/// A whole `<tool_call>` body -> `(name, arguments-json)`.
///
/// Takes the body WITH its opener, unlike `parse_coder_function_call`, because
/// recovering a malformed opener means deciding what the opener was — which the
/// two call sites cannot each do for themselves without drifting apart.
pub(crate) fn parse_xml_tool_call(call: &str, schemas: &[String]) -> Option<(String, String)> {
    let call = call.trim();
    let (name, name_end) = parse_xml_function_opener(call)?;
    let body_start = name_end + 1;
    match call[body_start..].find("</function>") {
        // The taught opener keeps the lenient tail: an unterminated call is a
        // truncated generation, and the parameters read so far are still real.
        None if call.contains("<function=") => {
            parse_coder_function_call(&call[call.find("<function=").unwrap() + 10..], schemas)
        }
        // A RECOVERED opener is accepted only with its close, which is the
        // guard that keeps a bare `<NAME>` from matching ordinary markup.
        None => None,
        Some(rel) => {
            let inner = &call[body_start..body_start + rel];
            parse_coder_params(&name, inner, schemas)
        }
    }
}

fn parse_coder_function_call(body: &str, schemas: &[String]) -> Option<(String, String)> {
    let gt = body.find('>')?;
    let name = body[..gt].trim().to_string();
    // Same unbounded scan, same failure, one level up. A model that writes
    // `<function=bash` and forgets the `>` sends this hunting to the `>` of the
    // NEXT tag, and the tool name becomes `bash\n<parameter=command`; one that
    // writes `<function<bash>` yields `<bash`. Both were emitted confidently
    // by this parser, with no error, before the screen. Upstream `dev-sslee`
    // fixed the same thing in b9ca2d050.
    if !is_plausible_name(&name) {
        return None;
    }
    parse_coder_params(&name, &body[gt + 1..], schemas)
}

/// The `<parameter=K>V</parameter>` list of a call whose NAME is already known.
///
/// Split out so the recovered-opener path (`parse_xml_tool_call`) and the
/// taught-opener path read parameters with one implementation. They diverged
/// on the typing rules the first time this was two copies.
fn parse_coder_params(name: &str, rest: &str, schemas: &[String]) -> Option<(String, String)> {
    let name = name.to_string();

    // The declared type of each parameter of THIS function, if we were told.
    let mut types: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for s in schemas {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(s) else { continue };
        let f = v.get("function").unwrap_or(&v);
        if f.get("name").and_then(|x| x.as_str()) != Some(name.as_str()) {
            continue;
        }
        if let Some(props) =
            f.get("parameters").and_then(|p| p.get("properties")).and_then(|p| p.as_object())
        {
            for (k, pf) in props {
                if let Some(t) = pf.get("type").and_then(|x| x.as_str()) {
                    types.insert(k.clone(), t.to_ascii_lowercase());
                }
            }
        }
        break;
    }

    let mut args = serde_json::Map::new();
    let mut tail = rest;
    while let Some(open) = tail.find("<parameter=") {
        let after = &tail[open + "<parameter=".len()..];
        let Some(gt) = after.find('>') else { break };
        let pname = after[..gt].trim().to_string();
        // The scan above is unbounded: it takes everything up to the next `>`
        // ANYWHERE in the document. When the model writes `<parameter=command=`
        // and forgets the `>`, the next `>` is the redirect in `2>/dev/null`,
        // and the name becomes ninety characters of shell. Captured live by
        // upstream `dev-sslee` on django__django-10914 turn 1; reproduced
        // against this parser, which emitted
        //
        //     {"command=\nfind /testbed ... 2": "/dev/null | head -20"}
        //
        // with no error at all, so the agent executed nothing useful and had no
        // way to know why.
        //
        // The screen is on the NAME only. A redirect inside a parameter VALUE
        // is ordinary shell and must still parse -- there is a test for exactly
        // that, so this cannot be mistaken for a fix that rejects shell syntax.
        if !is_plausible_name(&pname) {
            return None;
        }
        let vstart = &after[gt + 1..];
        // An unterminated parameter is a truncated generation, not a parse
        // failure: take the rest and let the caller decide.
        let (raw, consumed) = match vstart.find("</parameter>") {
            Some(end) => (&vstart[..end], end + "</parameter>".len()),
            None => (vstart, vstart.len()),
        };
        let mut val = raw;
        val = val.strip_prefix('\n').unwrap_or(val);
        val = val.strip_suffix('\n').unwrap_or(val);

        let ty = types.get(&pname).map(String::as_str).unwrap_or("string");
        let parsed = if val.eq_ignore_ascii_case("null") {
            serde_json::Value::Null
        } else if matches!(ty, "string" | "str" | "text" | "varchar" | "char" | "enum") {
            serde_json::Value::String(val.to_string())
        } else if ty.starts_with("int") || ty.starts_with("uint") || ty.starts_with("long")
            || ty.starts_with("short") || ty.starts_with("unsigned")
        {
            val.parse::<i64>()
                .map(Into::into)
                .unwrap_or_else(|_| serde_json::Value::String(val.to_string()))
        } else if ty.starts_with("num") || ty.starts_with("float") || ty.starts_with("double") {
            val.parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| serde_json::Value::String(val.to_string()))
        } else if matches!(ty, "boolean" | "bool" | "binary") {
            serde_json::Value::Bool(val.eq_ignore_ascii_case("true"))
        } else {
            serde_json::from_str::<serde_json::Value>(val)
                .unwrap_or_else(|_| serde_json::Value::String(val.to_string()))
        };
        args.insert(pname, parsed);
        tail = &vstart[consumed..];
    }
    Some((name, serde_json::Value::Object(args).to_string()))
}

struct QwenToolDecoder {
    decoder: TokenizerDecoder,
    accumulated: String,
    inside: bool,
    /// Whether the call being read arrived WITHOUT a `<tool_call>` wrapper, so
    /// `</function>` is what closes it. See the back-off in `feed`.
    unwrapped: bool,
    has_tools: bool,
    dialect: ToolDialect,
    /// The request's schemas, needed only by [`ToolDialect::Coder`] — it is the
    /// only source of argument types, because the wire format has none.
    schemas: Vec<String>,
}

impl ToolDecoder for QwenToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        if !self.has_tools {
            return ToolEvent::Start;
        }
        let text = self.decoder.feed(tokens);
        self.accumulated.push_str(&text);

        if !self.inside {
            if let Some(pos) = self.accumulated.find("<tool_call>") {
                self.inside = true;
                self.unwrapped = false;
                self.accumulated = self.accumulated[pos + "<tool_call>".len()..].to_string();
                return ToolEvent::Start;
            }
            // The reference parser's back-off, and it is not a nicety. Asked to
            // read a file with ten tools offered, Qwen3-Coder-30B emits a
            // perfectly well-formed `<function=read>…</function>` with NO
            // `<tool_call>` wrapper — and the entire call is dropped for want of
            // an opening tag the model never wrote. `qwen3coder_tool_parser`
            // keys its quick check on `<function=` for exactly this reason and
            // falls back to the whole output when the wrapper is absent.
            //
            // The prefix is KEPT rather than consumed: it carries the function
            // name, and `</function>` then closes what `</tool_call>` would have.
            if self.dialect.emits_xml_calls() {
                if let Some(pos) = self.accumulated.find("<function=") {
                    self.inside = true;
                    self.unwrapped = true;
                    self.accumulated = self.accumulated[pos..].to_string();
                    return ToolEvent::Start;
                }
            }
            return ToolEvent::Start;
        }

        let close = if self.unwrapped { "</function>" } else { "</tool_call>" };
        let Some(pos) = self.accumulated.find(close) else {
            return ToolEvent::Start;
        };
        // An unwrapped call needs its own closer INSIDE the body, because that
        // is what bounds the parameter list; a wrapped one is cut before it.
        let body_end = if self.unwrapped { pos + close.len() } else { pos };
        let call_body = self.accumulated[..body_end].trim().to_string();
        self.accumulated = self.accumulated[pos + close.len()..].to_string();
        self.inside = false;
        self.unwrapped = false;

        if self.dialect.emits_xml_calls() {
            // Whole body, opener included: deciding what a malformed opener was
            // is `parse_xml_tool_call`'s job, not each call site's.
            if let Some((name, args)) = parse_xml_tool_call(&call_body, &self.schemas) {
                return ToolEvent::Call(name, args);
            }
        } else if let Ok(v) = serde_json::from_str::<serde_json::Value>(&call_body) {
            let name = v["name"].as_str().unwrap_or("").to_string();
            let args = v["arguments"].to_string();
            return ToolEvent::Call(name, args);
        }
        ToolEvent::Start
    }

    fn reset(&mut self) {
        self.decoder.reset();
        self.accumulated.clear();
        self.inside = false;
        self.unwrapped = false;
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
                tool_dialect: ToolDialect::Hermes,
                system_before_tools: true,
                empty_reasoning_header: false,
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                tool_response_trailing_newline: false,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )
    }

    fn qwen2() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                tool_dialect: ToolDialect::Hermes,
                system_before_tools: true,
                empty_reasoning_header: false,
                has_thinking: false,
                has_tools: true,
                generation_suffix: "",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                tool_response_trailing_newline: false,
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )
    }

    fn olmo3() -> QwenInstruct {
        QwenInstruct::new(
            make_tok(),
            ChatMLConfig {
                tool_dialect: ToolDialect::Hermes,
                system_before_tools: true,
                empty_reasoning_header: false,
                has_thinking: true,
                has_tools: false,
                generation_suffix: "",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                tool_response_trailing_newline: false,
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

    /// Qwen3.5/3.6 writes JSON schemas and XML calls, and neither neighbour
    /// renders that.
    ///
    /// This is the test the shipped-then-corrected version of this change did
    /// not have. Routing Qwen3.6 to `ToolDialect::Coder` got the CALL format
    /// right and the SCHEMA format wrong -- `<function><name>` blocks the
    /// checkpoint's template never writes -- and every existing test passed,
    /// because they all asked about calls.
    /// A call the model plainly made, opened in a way the template never taught.
    ///
    /// Ported from upstream c1e0a84ab, whose live capture from this checkpoint
    /// family was `<list_files>` opened bare and closed with `</function>`.
    /// Refusing it costs the call and reports malformed syntax, which sends an
    /// agent into re-issuing the action instead of acting on its result.
    #[test]
    fn a_malformed_opener_is_recovered_when_the_call_still_closes() {
        // The taught form, unchanged.
        let (name, args) = parse_xml_tool_call(
            "<function=read_file>\n<parameter=path>README.md</parameter>\n</function>",
            &[],
        )
        .expect("taught opener");
        assert_eq!(name, "read_file");
        assert!(args.contains("README.md"));

        // `<function>NAME` -- the tag is right, the name follows as text.
        let (name, _) = parse_xml_tool_call(
            "<function>read_file\n<parameter=path>README.md</parameter>\n</function>",
            &[],
        )
        .expect("<function>NAME");
        assert_eq!(name, "read_file");

        // Bare `<NAME>`, the live capture's shape.
        let (name, args) = parse_xml_tool_call(
            "<list_files>\n<parameter=recursive>false</parameter>\n</function>",
            &[],
        )
        .expect("bare <NAME>");
        assert_eq!(name, "list_files");
        assert!(args.contains("recursive"));
    }

    /// The guards that keep recovery from inventing calls out of markup.
    #[test]
    fn opener_recovery_refuses_what_is_not_a_call() {
        // No `</function>` close: a recovered opener is not trusted without it.
        assert_eq!(
            parse_xml_tool_call("<list_files>\n<parameter=recursive>false</parameter>", &[]),
            None,
            "a bare opener with no close must not become a call"
        );
        // The surface's own structural tags are not function names.
        assert_eq!(
            parse_xml_tool_call("<parameter=path>README.md</parameter>\n</function>", &[]),
            None,
            "<parameter=...> was read as the function name"
        );
        // A closing tag read as an opener yields "/function".
        assert_eq!(parse_xml_tool_call("</function>", &[]), None);
        // And the live shell-redirect malformation still refuses.
        let live = concat!(
            "<function=bash>\n<parameter=command=\n",
            "find /testbed -name \"*.py\" | xargs grep -l FOO 2>/dev/null | head -20\n",
            "</parameter>\n</function>"
        );
        assert_eq!(parse_xml_tool_call(live, &[]), None);
        // ...while the repaired form keeps its redirect intact.
        let repaired = concat!(
            "<function=bash>\n<parameter=command>\n",
            "find /testbed -name \"*.py\" | xargs grep -l FOO 2>/dev/null | head -20\n",
            "</parameter>\n</function>"
        );
        let (name, args) = parse_xml_tool_call(repaired, &[]).expect("repaired parses");
        assert_eq!(name, "bash");
        assert!(args.contains("2>/dev/null"), "the redirect was mangled: {args}");
    }

    #[test]
    fn qwen35_renders_json_schemas_under_the_templates_own_heading() {
        let schema = r#"{"type": "function", "function": {"name": "read_file"}}"#.to_string();
        let block = QwenInstruct::build_tool_system_prompt_qwen35(&[schema.clone()]);

        // The heading is the Qwen3.5 template's, not Qwen3's.
        assert!(
            block.starts_with("# Tools\n\nYou have access to the following functions:\n\n<tools>"),
            "wrong tools heading:\n{block}"
        );
        // Schemas are JSON, verbatim -- NOT Coder's XML schema blocks.
        assert!(block.contains(&schema), "the schema was not written as JSON:\n{block}");
        assert!(
            !block.contains("<name>"),
            "Coder's XML schema blocks leaked into the Qwen3.5 preamble:\n{block}"
        );
        // And the call instruction is the checkpoint's, verbatim.
        assert!(block.ends_with(QWEN35_XML_CALL_INSTRUCTION), "instruction missing or altered");
        assert_eq!(
            QWEN35_XML_CALL_INSTRUCTION.len(),
            815,
            "the instruction is transcribed from chat_template.jinja; a length change \
             means the bytes the model was trained on no longer match"
        );

        // The three dialects render three different tools turns.
        let hermes = QwenInstruct::build_tool_system_prompt(&[schema.clone()]);
        let coder = QwenInstruct::build_tool_system_prompt_coder(&[schema]);
        assert_ne!(block, hermes, "Qwen3.5 collapsed into the Hermes preamble");
        assert_ne!(block, coder, "Qwen3.5 collapsed into the Coder preamble");
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
            QwenInstruct::answer_batch_inner_text(&[("fn1".to_string(), "Hello".to_string())], false),
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
                &[("f".to_string(), "{}".to_string())],
                ToolDialect::Hermes,
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
                &[("f".to_string(), "{}".to_string())],
                ToolDialect::Hermes,
            ),
            "\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"
        );
    }

    #[test]
    fn coder_call_survives_a_missing_tool_call_wrapper() {
        // Observed on Qwen3-Coder-30B replaying a real opencode turn with ten
        // tools offered: a well-formed `<function=read>…</function>` and no
        // `<tool_call>` around it. Requiring the wrapper drops the whole call.
        let schema = r#"{"name":"read","parameters":{"properties":{"filePath":{"type":"string"}}}}"#;
        let raw = "<function=read>\n<parameter=filePath>\n/tmp/hello\n</parameter>\n</function>";
        let fs = raw.find("<function=").unwrap();
        let after = &raw[fs + "<function=".len()..];
        let body = &after[..after.find("</function>").unwrap()];
        let (name, args) = parse_coder_function_call(body, &[schema.to_string()]).unwrap();
        assert_eq!(name, "read");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&args).unwrap()["filePath"],
            serde_json::json!("/tmp/hello")
        );
    }

    #[test]
    fn coder_call_types_arguments_from_the_schema() {
        // XML carries no types, so the schema is the only thing that can say
        // whether `30` is the number 30 or the string "30". Both appear here.
        let schema = r#"{"name":"read","parameters":{"properties":{
            "path":{"type":"string"},"offset":{"type":"integer"},
            "raw":{"type":"boolean"},"ratio":{"type":"number"}}}}"#
            .to_string();
        let body = "read>\n<parameter=path>\n30\n</parameter>\n\
                    <parameter=offset>\n30\n</parameter>\n\
                    <parameter=raw>\nTRUE\n</parameter>\n\
                    <parameter=ratio>\n1.5\n</parameter>\n";
        let (name, args) = parse_coder_function_call(body, &[schema]).unwrap();
        assert_eq!(name, "read");
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        // Same text, different types — decided by the schema, not by shape.
        assert_eq!(v["path"], serde_json::json!("30"));
        assert_eq!(v["offset"], serde_json::json!(30));
        assert_eq!(v["raw"], serde_json::json!(true));
        assert_eq!(v["ratio"], serde_json::json!(1.5));
    }

    #[test]
    fn coder_call_strips_exactly_one_delimiter_newline() {
        // The newline after `>` and the one before `</parameter>` belong to the
        // delimiter. A value that genuinely ends in a blank line keeps the rest.
        let schema = r#"{"name":"write","parameters":{"properties":{"body":{"type":"string"}}}}"#
            .to_string();
        let body = "write>\n<parameter=body>\nline1\n\n</parameter>\n";
        let (_, args) = parse_coder_function_call(body, &[schema]).unwrap();
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["body"], serde_json::json!("line1\n"));
    }

    #[test]
    fn coder_call_without_a_schema_degrades_to_strings_not_to_nothing() {
        // The decoder is constructible without schemas. A call must still be a
        // call: an agent can recover from a stringly-typed argument, not from
        // a tool call that was never reported.
        let body = "ls>\n<parameter=n>\n7\n</parameter>\n";
        let (name, args) = parse_coder_function_call(body, &[]).unwrap();
        assert_eq!(name, "ls");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&args).unwrap()["n"],
                   serde_json::json!("7"));
    }

    #[test]
    fn coder_preamble_carries_the_nesting_rule_the_decoder_relies_on() {
        let schema = r#"{"type":"function","function":{"name":"get_time",
            "description":"Get the time","parameters":{"type":"object",
            "properties":{"tz":{"type":"string","description":"zone"}},
            "required":["tz"]}}}"#
            .to_string();
        let p = QwenInstruct::build_tool_system_prompt_coder(&[schema]);
        assert!(p.starts_with("You have access to the following functions:\n\n<tools>"));
        assert!(p.contains("<function>\n<name>get_time</name>"));
        assert!(p.contains("<parameter>\n<name>tz</name>\n<type>string</type>"));
        assert!(p.contains("<required>[`tz`]</required>"));
        // The instruction the model needs in order to emit what we parse.
        assert!(p.contains("<tool_call>\n<function=example_function_name>"));
        assert!(p.contains("must be nested within <tool_call></tool_call> XML tags"));
        // And NOT the Hermes preamble it used to get.
        assert!(!p.contains("# Tools"));
    }

    #[test]
    fn coder_replay_round_trips_through_the_parser() {
        // A replayed turn must be re-readable by the decoder, or a multi-turn
        // agent drifts from what it was told it said.
        let calls = vec![("read".to_string(), r#"{"path":"/a.rs","offset":3}"#.to_string())];
        let text = QwenInstruct::assistant_with_tool_calls_inner_text(
            None,
            &calls,
            ToolDialect::Coder,
        );
        assert!(text.contains("<tool_call>\n<function=read>\n"));
        assert!(text.contains("<parameter=path>\n/a.rs\n</parameter>"));
        // The int went out unquoted, as the template renders it.
        assert!(text.contains("<parameter=offset>\n3\n</parameter>"));
        let schema = r#"{"name":"read","parameters":{"properties":{
            "path":{"type":"string"},"offset":{"type":"integer"}}}}"#
            .to_string();
        let inner = text.split("<function=").nth(1).unwrap().split("</function>").next().unwrap();
        let (name, args) = parse_coder_function_call(inner, &[schema]).unwrap();
        assert_eq!(name, "read");
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["path"], serde_json::json!("/a.rs"));
        assert_eq!(v["offset"], serde_json::json!(3));
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
            ], false),
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
                tool_dialect: ToolDialect::Hermes,
                system_before_tools: true,
                empty_reasoning_header: false,
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                thinking_off_suffix: "<think>\n\n</think>\n\n",
                tool_response_trailing_newline: false,
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


#[cfg(test)]
mod parameter_name_screen {
    use super::*;

    /// Captured live by upstream `dev-sslee` on django__django-10914 turn 1 and
    /// reproduced against this parser before the fix, which emitted a call whose
    /// argument KEY was ninety characters of shell text and whose VALUE was
    /// `/dev/null | head -20` -- silently.
    #[test]
    fn a_parameter_name_that_swallowed_a_shell_redirect_is_refused() {
        let live = concat!(
            "read>\n",
            "<parameter=command=\n",
            "find /testbed -type f -name \"*.py\" | xargs grep -l \"FILE_UPLOAD_PERMISSION\" 2>/dev/null | head -20\n",
            "</parameter>\n</function>"
        );
        assert_eq!(parse_coder_function_call(live, &[]), None);
    }

    /// THE GUARD ON THE GUARD. The same redirect inside a parameter VALUE is
    /// ordinary shell and must still parse. Without this test the fix above
    /// could be "reject anything containing a redirect" and look correct.
    #[test]
    fn a_redirect_inside_a_parameter_value_still_parses() {
        let ok = concat!(
            "bash>\n",
            "<parameter=command>\n",
            "grep -l FOO /testbed 2>/dev/null | head -20\n",
            "</parameter>\n</function>"
        );
        let (name, args) = parse_coder_function_call(ok, &[]).expect("must parse");
        assert_eq!(name, "bash");
        assert!(args.contains("2>/dev/null"), "redirect lost from the value: {args}");
        assert!(args.contains("\"command\""), "wrong key: {args}");
    }

    /// An empty name was the only thing upstream's older check caught, and this
    /// parser did not even have that.
    #[test]
    fn an_empty_parameter_name_is_refused() {
        assert_eq!(parse_coder_function_call("read>\n<parameter=>\nx\n</parameter>\n</function>", &[]), None);
    }
}


#[cfg(test)]
mod function_name_screen {
    use super::*;

    /// A missing `>` after the function name sends the scan to the `>` of the
    /// next tag. Before the screen this parser returned the tool name
    /// `bash\n<parameter=command` -- confidently, with no error.
    #[test]
    fn a_missing_closing_angle_does_not_invent_a_tool_name() {
        let body = "bash\n<parameter=command>\nls\n</parameter>\n</function>";
        assert_eq!(parse_coder_function_call(body, &[]), None);
    }

    /// `<function<bash>` rather than `<function=bash>`. Returned `<bash`.
    #[test]
    fn an_angle_where_the_equals_belongs_does_not_invent_a_tool_name() {
        let body = "<bash>\n<parameter=command>\nls\n</parameter>\n</function>";
        assert_eq!(parse_coder_function_call(body, &[]), None);
    }

    /// THE GUARD ON THE GUARD, again: an ordinary well-formed call must still
    /// parse, or the two tests above are satisfied by a parser that refuses
    /// everything.
    #[test]
    fn a_well_formed_call_still_parses() {
        let body = "bash>\n<parameter=command>\nls -la\n</parameter>\n</function>";
        let (name, args) = parse_coder_function_call(body, &[]).expect("must parse");
        assert_eq!(name, "bash");
        assert!(args.contains("ls -la"), "{args}");
    }
}
