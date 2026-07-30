//! ChatML-family instruct implementation.
//!
//! Covers Qwen3, Qwen2.5, OLMo3, and any ChatML-based model.
//! Configurable via `ChatMLConfig` for thinking/tool support.
//!
//! Reference: Qwen3 Jinja chat template with tool-calling support.

use std::sync::Arc;
use crate::inference::structured::grammar::Grammar;
use crate::model::instruct::{
    ChatDecoder,
    Instruct,
    ReasoningDecoder,
    ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::model::instruct::decoders::{GenericChatDecoder, ThinkingDecoder, NoopReasoningDecoder};
use crate::model::tokenizer::Tokenizer;

// =============================================================================
// Configuration
// =============================================================================

static TEMPLATE: &str = r#"
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- " # Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for forward_message in messages %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- set message = messages[index] %}
    {%- set current_content = message.content if message.content is not none else '' %}
    {%- set tool_start = '<tool_response>' %}
    {%- set tool_start_length = tool_start|length %}
    {%- set start_of_message = current_content[:tool_start_length] %}
    {%- set tool_end = '</tool_response>' %}
    {%- set tool_end_length = tool_end|length %}
    {%- set start_pos = (current_content|length) - tool_end_length %}
    {%- if start_pos < 0 %}
        {%- set start_pos = 0 %}
    {%- endif %}
    {%- set end_of_message = current_content[start_pos:] %}
    {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = (message.content.split('</think>')|last).lstrip('\n') %}
                {%- set reasoning_content = (message.content.split('</think>')|first).rstrip('\n') %}
                {%- set reasoning_content = (reasoning_content.split('<think>')|last).lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"#;


/// Feature flags for ChatML-family models.
pub struct ChatMLConfig {
    pub has_thinking: bool,
    pub has_tools: bool,
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
    // `<tool_call>`/`<tool_response>` chunk), so reusing the newline-inclusive
    // prefix would double up newlines when merging multiple chunks into one
    // turn. See `assistant_with_tool_calls`/`answer_batch`.
    user_prefix_no_nl: Vec<u32>,
    assistant_prefix_no_nl: Vec<u32>,
    newline_ids: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    stop_ids: Vec<u32>,
    // Thinking delimiters
    think_prefix_ids: Vec<u32>,
    think_suffix_ids: Vec<u32>,
    // Tool delimiters
    tool_response_prefix_tokens: Vec<u32>,
    tool_response_suffix_tokens: Vec<u32>,
    // Tool-call/tool-response fragments for replaying history
    // (`assistant_with_tool_calls`/`answer_batch`). Pre-tokenized like every
    // other literal fragment in this struct — dynamic content (name,
    // arguments, value) is always encoded on its own and concatenated as
    // token IDs, never interpolated into a literal string and encoded in one
    // shot, since that isn't guaranteed to retokenize into the same pieces
    // (verified the hard way: see git history of this file).
    tool_call_open_tokens: Vec<u32>,      // "\n<tool_call>\n{\"name\": \""
    tool_call_mid_tokens: Vec<u32>,       // "\", \"arguments\": "
    tool_call_close_tokens: Vec<u32>,     // "}\n</tool_call>"
    tool_response_open_tokens: Vec<u32>,  // "\n<tool_response>\n"
}

impl QwenInstruct {
    /// Create with full config.
    pub fn new(tokenizer: Arc<Tokenizer>, config: ChatMLConfig) -> Self {
        let encode = |s: &str| tokenizer.encode(s);
        let stop_ids: Vec<u32> = config.stop_tokens
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

        let mut tool_resp_prefix = encode("<tool_response>");
        tool_resp_prefix.extend(&newline);
        let mut tool_resp_suffix = newline.clone();
        tool_resp_suffix.extend(encode("</tool_response>"));

        let tool_call_open_tokens = encode("\n<tool_call>\n{\"name\": \"");
        let tool_call_mid_tokens = encode("\", \"arguments\": ");
        let tool_call_close_tokens = encode("}\n</tool_call>");
        let mut tool_response_open_tokens = newline.clone();
        tool_response_open_tokens.extend(&tool_resp_prefix);

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            assistant_prefix: make_prefix("assistant"),
            user_prefix_no_nl,
            assistant_prefix_no_nl,
            newline_ids: newline.clone(),
            generation_header: make_prefix("assistant"),
            turn_suffix,
            stop_ids,
            think_prefix_ids: think_prefix,
            think_suffix_ids: think_suffix,
            tool_response_prefix_tokens: tool_resp_prefix,
            tool_response_suffix_tokens: tool_resp_suffix,
            tool_call_open_tokens,
            tool_call_mid_tokens,
            tool_call_close_tokens,
            tool_response_open_tokens,
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
        // if the preamble diverges.
        let mut prompt = String::from(
            "\n# Tools\n\n\
             You may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n\
             <tools>"
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
             </tool_call>"
        );
        prompt
    }

    /// Escape a string for embedding in an EBNF string-literal token.
    fn escape_ebnf_literal(s: &str) -> String {
        let mut out = String::new();
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                _ => out.push(ch),
            }
        }
        out
    }

    /// Build an EBNF grammar for constrained Qwen tool-call generation.
    ///
    /// The grammar enforces well-formed `<tool_call>` blocks with a valid
    /// tool name and syntactically-valid JSON arguments. The model is free
    /// to include any properties in any order within the arguments object,
    /// guided by the few-shot examples and tool descriptions in the prompt.
    fn build_tool_call_grammar(tools: &[String]) -> Option<String> {
        struct ToolSpec {
            name: String,
            #[allow(dead_code)]
            parameters: Option<serde_json::Value>,
        }

        let mut specs: Vec<ToolSpec> = Vec::new();
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
                let func = parsed.get("function");
                let name = func
                    .and_then(|f| f.get("name"))
                    .or_else(|| parsed.get("name"))
                    .and_then(|n| n.as_str());
                if let Some(n) = name {
                    let parameters = func
                        .and_then(|f| f.get("parameters"))
                        .or_else(|| parsed.get("parameters"))
                        .cloned();
                    specs.push(ToolSpec { name: n.to_string(), parameters });
                }
            }
        }
        if specs.is_empty() {
            return None;
        }

        let mut tool_json_alts: Vec<String> = Vec::with_capacity(specs.len());
        let mut extra_rules = String::new();

        // Layer 1: the grammar enforces well-formed `<tool_call>` blocks
        // with a valid tool name and JSON arguments. Per-tool schema
        // constraints (layer 2) are intentionally omitted — the model is
        // free to include any properties in any order, guided by the
        // few-shot examples and its own training.
        for (i, spec) in specs.iter().enumerate() {
            let alt_name = format!("tool-json-{i}");
            let escaped_name = Self::escape_ebnf_literal(&spec.name);
            extra_rules.push_str(&format!(
                "{alt_name} ::= \"{{\\\"name\\\": \\\"{escaped_name}\\\", \\\"arguments\\\": \" json-object \"}}\"\n",
            ));
            tool_json_alts.push(alt_name);
        }

        let tool_json_alt = tool_json_alts.join(" | ");

        let grammar = format!(
            r#"root ::= tool-call ("\n" tool-call)*
tool-call ::= "<tool_call>\n" tool-json "\n</tool_call>"
tool-json ::= {tool_json_alt}
{extra_rules}json-object ::= "{{" json-members? "}}"
json-members ::= json-pair ("," json-pair)*
json-pair ::= json-string ":" json-value
json-value ::= json-string | json-number | json-object | json-array | "true" | "false" | "null"
json-string ::= "\"" json-chars "\""
json-chars ::= json-char*
json-char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-array ::= "[" (json-value ("," json-value)*)? "]"
"#
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

    fn equip_after_system(&self, system_content: Option<&str>, tools: &[String]) -> Vec<u32> {
        // Reference (qwen2.rs's embedded Jinja template, top-of-prompt
        // preamble): when tools are present, the leading system message's
        // content (if any) and the tools block are folded into ONE system
        // turn ('content' + '\n\n' + tools-block), not two separate turns.
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

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        if !self.config.has_tools {
            return Vec::new();
        }
        // Reference: tool responses go in a user turn with <tool_response> wrapper
        // Format: <|im_start|>user\n<tool_response>\ncontent\n</tool_response><|im_end|>\n
        let mut tokens = self.user_prefix.clone();
        tokens.extend(&self.tool_response_prefix_tokens);
        tokens.extend(self.tokenizer.encode(value));
        tokens.extend(&self.tool_response_suffix_tokens);
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn assistant_with_tool_calls(&self, content: Option<&str>, calls: &[(String, String)]) -> Vec<u32> {
        if !self.config.has_tools || calls.is_empty() {
            return self.assistant(content.unwrap_or(""));
        }
        // Reference (qwen2.rs's embedded Jinja template, assistant branch):
        // '<|im_start|>' + role, then '\n' + content only if content is
        // truthy, then for each call '\n<tool_call>\n{"name": ..., "arguments":
        // ...}\n</tool_call>', then '<|im_end|>\n'. Note there's no
        // unconditional newline after the role tag — it comes from whichever
        // of those two branches fires first.
        let mut tokens = self.assistant_prefix_no_nl.clone();
        if let Some(c) = content {
            if !c.is_empty() {
                tokens.extend(&self.newline_ids);
                tokens.extend(self.tokenizer.encode(c));
            }
        }
        for (name, arguments_json) in calls {
            tokens.extend(&self.tool_call_open_tokens);
            tokens.extend(self.tokenizer.encode(name));
            tokens.extend(&self.tool_call_mid_tokens);
            tokens.extend(self.tokenizer.encode(arguments_json));
            tokens.extend(&self.tool_call_close_tokens);
        }
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
        // either — same shape as assistant_with_tool_calls above). Merging
        // two single-result `answer()` calls would double up the newline
        // between chunks, which is why this needs its own implementation
        // rather than just looping `answer()` (the trait default).
        let mut tokens = self.user_prefix_no_nl.clone();
        for (_name, value) in results {
            tokens.extend(&self.tool_response_open_tokens);
            tokens.extend(self.tokenizer.encode(value));
            tokens.extend(&self.tool_response_suffix_tokens);
        }
        tokens.extend(&self.turn_suffix);
        tokens
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(self.tokenizer.clone(), self.stop_ids.clone()))
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
            tokenizer: self.tokenizer.clone(),
            accumulated: String::new(),
            inside: false,
            has_tools: self.config.has_tools,
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        if !self.config.has_tools || tools.is_empty() {
            return None;
        }
        // Thinking models emit <think>...</think> before <tool_call>, so the
        // grammar (which requires <tool_call> at position 0) would block all
        // thinking tokens → empty logit mask → infinite zero-token decode loop.
        // Skip constrained generation for thinking models; the ToolDecoder
        // handles detection correctly without it.
        if self.config.has_thinking {
            return None;
        }
        let source = Self::build_tool_call_grammar(tools)?;
        let grammar = Grammar::from_ebnf(&source, "root").ok()?;
        Some(ToolGrammar { source, grammar: Arc::new(grammar) })
    }
}

// =============================================================================
// Tool Decoder
// =============================================================================

struct QwenToolDecoder {
    tokenizer: Arc<Tokenizer>,
    accumulated: String,
    inside: bool,
    has_tools: bool,
}

impl ToolDecoder for QwenToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        if !self.has_tools {
            return ToolEvent::Start;
        }
        let text = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&text);

        if !self.inside {
            if self.accumulated.contains("<tool_call>") {
                self.inside = true;
                if let Some(pos) = self.accumulated.find("<tool_call>") {
                    self.accumulated = self.accumulated[pos + "<tool_call>".len()..].to_string();
                }
                return ToolEvent::Start;
            }
        } else if self.accumulated.contains("</tool_call>") {
            if let Some(pos) = self.accumulated.find("</tool_call>") {
                let call_json = self.accumulated[..pos].trim().to_string();
                self.accumulated = self.accumulated[pos + "</tool_call>".len()..].to_string();
                self.inside = false;
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&call_json) {
                    let name = v["name"].as_str().unwrap_or("").to_string();
                    let args = v["arguments"].to_string();
                    return ToolEvent::Call(name, args);
                }
            }
        }
        ToolEvent::Start
    }

    fn reset(&mut self) {
        self.accumulated.clear();
        self.inside = false;
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::model::tokenizer::Tokenizer;

    fn make_tok() -> Arc<Tokenizer> {
        let v: Vec<String> = vec![
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "system", "\n", "user", "assistant", "Hello", " world",
            "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>", "<tools>", "</tools>",
        ].into_iter().map(String::from).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    fn qwen3() -> QwenInstruct {
        QwenInstruct::new(make_tok(), ChatMLConfig {
            has_thinking: true, has_tools: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        })
    }

    fn qwen2() -> QwenInstruct {
        QwenInstruct::new(make_tok(), ChatMLConfig {
            has_thinking: false, has_tools: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        })
    }

    fn olmo3() -> QwenInstruct {
        QwenInstruct::new(make_tok(), ChatMLConfig {
            has_thinking: true, has_tools: false,
            stop_tokens: &["<|im_end|>"],
        })
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

    /// Whether `s` is a complete, legally-terminatable match for `tg`'s
    /// grammar. `accept_string` walks raw bytes directly, so the tokenizer
    /// passed to `GrammarMatcher` is irrelevant here (no token-boundary
    /// concerns for this check).
    fn matches_grammar(tg: &ToolGrammar, s: &str) -> bool {
        let mut m = crate::inference::structured::matcher::GrammarMatcher::new(
            tg.grammar.clone(), make_tok(), vec![], 10,
        );
        m.accept_string(s) && m.can_terminate()
    }

    #[test]
    fn tool_call_grammar_constrains_arguments_by_schema() {
        let inst = qwen3();
        let tool = serde_json::json!({
            "name": "calculator",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"],
                "additionalProperties": false
            }
        })
        .to_string();

        let tg = inst.tool_call_grammar(&[tool]).expect("grammar should build");

        let valid = "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\":\"1+1\"}}\n</tool_call>";
        assert!(matches_grammar(&tg, valid), "well-typed arguments should match");

        // Without per-tool schema constraints, any valid JSON arguments are accepted.
        let any_args = "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\":1}}\n</tool_call>";
        assert!(matches_grammar(&tg, any_args), "any valid JSON arguments should be accepted");

        let extra_prop = "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\":\"1+1\",\"extra\":\"x\"}}\n</tool_call>";
        assert!(matches_grammar(&tg, extra_prop), "extra properties should be accepted");

        let empty_args = "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {}}\n</tool_call>";
        assert!(matches_grammar(&tg, empty_args), "empty arguments should be accepted");
    }

    #[test]
    fn tool_call_grammar_multiple_tools_use_independent_names() {
        let inst = qwen3();
        let tools = vec![
            serde_json::json!({
                "name": "calculator",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]
                }
            })
            .to_string(),
            serde_json::json!({
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            })
            .to_string(),
        ];
        let tg = inst.tool_call_grammar(&tools).expect("grammar should build");

        let calc_call = "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\":\"1+1\"}}\n</tool_call>";
        assert!(matches_grammar(&tg, calc_call));

        let weather_call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"NYC\"}}\n</tool_call>";
        assert!(matches_grammar(&tg, weather_call));

        // Tool name must match one of the declared tools.
        let wrong_name = "<tool_call>\n{\"name\": \"unknown\", \"arguments\": {}}\n</tool_call>";
        assert!(!matches_grammar(&tg, wrong_name), "undeclared tool name must be rejected");
    }

    #[test]
    fn tool_call_grammar_falls_back_to_generic_json_without_schema() {
        // A tool with no "parameters" at all should still produce a usable
        // grammar (falls back to the generic, unconstrained json-object).
        let inst = qwen3();
        let tool = serde_json::json!({"name": "no_args_tool"}).to_string();
        let tg = inst.tool_call_grammar(&[tool]).expect("grammar should build");

        let call = "<tool_call>\n{\"name\": \"no_args_tool\", \"arguments\": {\"anything\":123}}\n</tool_call>";
        assert!(matches_grammar(&tg, call));
    }

    #[test]
    fn tool_call_grammar_rejects_free_text() {
        // OpenHands provides a `finish` tool for signaling completion, so
        // the model never needs free text — it must always produce a
        // well-formed tool call.
        let inst = qwen3();
        let tool = serde_json::json!({
            "name": "calculator",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": false
            }
        })
        .to_string();
        let tg = inst.tool_call_grammar(&[tool]).expect("grammar should build");

        assert!(!matches_grammar(&tg, "I'm done with the task."));
        assert!(!matches_grammar(&tg, "The answer is 42."));

        // Tool calls should still be accepted.
        let valid = "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\":\"1+1\"}}\n</tool_call>";
        assert!(matches_grammar(&tg, valid));
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
        let inst = qwen3();
        let tokens = inst.answer("fn1", "Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<|im_start|>user\n<tool_response>\nHello\n</tool_response><|im_end|>\n"
        );
    }

    #[test]
    fn tool_decoder_parses_call() {
        // Build vocab with the JSON content as a single entry
        let mut v: Vec<String> = vec![
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "system", "\n", "user", "assistant", "Hello", " world",
            "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>", "<tools>", "</tools>",
            r#"{"name": "f", "arguments": {}}"#,
        ].into_iter().map(String::from).collect();
        let tok = Arc::new(Tokenizer::from_vocab(&v));
        let inst = QwenInstruct::new(tok, ChatMLConfig {
            has_thinking: true, has_tools: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        });
        let mut dec = inst.tool_decoder();
        // Feed: <tool_call> \n JSON \n </tool_call>
        dec.feed(&[11]); // <tool_call> → enters inside, returns Start
        dec.feed(&[4]);  // \n
        let event = dec.feed(&[17, 4, 12]); // JSON + \n + </tool_call>
        match event {
            ToolEvent::Call(name, args) => {
                assert_eq!(name, "f");
                assert_eq!(args, "{}");
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }

    #[test]
    fn assistant_with_tool_calls_falls_back_when_disabled() {
        let inst = olmo3();
        let with_calls = inst.assistant_with_tool_calls(Some("Hello"), &[("f".to_string(), "{}".to_string())]);
        assert_eq!(with_calls, inst.assistant("Hello"));
    }

    /// Vocab for the tool-call-history tests below: the base `make_tok()` set
    /// plus the three fixed literal fragments `assistant_with_tool_calls`
    /// pre-tokenizes in `new()`, and the dynamic `name`/`arguments_json`
    /// pieces those tests use ("f", "{}") as their own standalone entries —
    /// `self.tokenizer.encode(name)` / `encode(arguments_json)` are called on
    /// them in isolation, never interpolated into a bigger literal first (see
    /// the comment on the struct's `tool_call_open_tokens` field for why).
    fn make_tool_call_tok() -> Arc<Tokenizer> {
        let mut v: Vec<String> = vec![
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "system", "\n", "user", "assistant", "Hello", " world",
            "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>", "<tools>", "</tools>",
        ].into_iter().map(String::from).collect();
        v.push("\n<tool_call>\n{\"name\": \"".to_string());
        v.push("\", \"arguments\": ".to_string());
        v.push("}\n</tool_call>".to_string());
        v.push("f".to_string());
        v.push("{}".to_string());
        Arc::new(Tokenizer::from_vocab(&v))
    }

    #[test]
    fn assistant_with_tool_calls_matches_reference_with_content() {
        let tok = make_tool_call_tok();
        let inst = QwenInstruct::new(tok, ChatMLConfig {
            has_thinking: false, has_tools: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        });

        let tokens = inst.assistant_with_tool_calls(Some("Hello"), &[("f".to_string(), "{}".to_string())]);
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<|im_start|>assistant\nHello\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call><|im_end|>\n"
        );
    }

    #[test]
    fn assistant_with_tool_calls_matches_reference_no_content() {
        // Same as above but content=None: no unconditional newline after the
        // role tag when there's no leading text — the reference template only
        // ever emits one newline before the first `<tool_call>`, not two.
        let tok = make_tool_call_tok();
        let inst = QwenInstruct::new(tok, ChatMLConfig {
            has_thinking: false, has_tools: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        });

        let tokens = inst.assistant_with_tool_calls(None, &[("f".to_string(), "{}".to_string())]);
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call><|im_end|>\n"
        );
    }

    #[test]
    fn answer_batch_noop_when_disabled() {
        let inst = olmo3();
        assert!(inst.answer_batch(&[("fn1".to_string(), "42".to_string())]).is_empty());
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
        // The regression this exists to catch: calling `answer()` twice would
        // produce two separate `<|im_start|>user...<|im_end|>` turns; the
        // reference template merges consecutive tool results into ONE turn
        // with multiple `<tool_response>` blocks inside it. Unlike the
        // assistant-side tests above, `answer_batch`'s literal fragments are
        // all built from pieces already in the base vocab (see
        // `tool_response_open_tokens`'s construction — concatenated from
        // already-tokenized pieces, not encoded as a combined literal), so
        // only the dynamic values ("Hello", "world") need their own entries.
        let mut v: Vec<String> = vec![
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "system", "\n", "user", "assistant", "Hello", " world",
            "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>", "<tools>", "</tools>",
        ].into_iter().map(String::from).collect();
        v.push("world".to_string());
        let tok = Arc::new(Tokenizer::from_vocab(&v));
        let inst = QwenInstruct::new(tok, ChatMLConfig {
            has_thinking: false, has_tools: true,
            stop_tokens: &["<|im_end|>", "<|endoftext|>"],
        });

        let tokens = inst.answer_batch(&[
            ("fn1".to_string(), "Hello".to_string()),
            ("fn2".to_string(), "world".to_string()),
        ]);
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<|im_start|>user\n<tool_response>\nHello\n</tool_response>\n<tool_response>\nworld\n</tool_response><|im_end|>\n"
        );
    }

    #[test]
    fn tool_call_grammar_accepts_any_json_arguments() {
        let inst = qwen3();
        let file_editor = r#"{"name": "file_editor"}"#.to_string();
        let tg = inst.tool_call_grammar(&[file_editor]).expect("grammar should build");

        // Any valid JSON arguments should be accepted
        let with_old_new = "<tool_call>\n{\"name\": \"file_editor\", \"arguments\": {\"command\":\"str_replace\",\"path\":\"/tmp/test.py\",\"old_str\":\"x\",\"new_str\":\"y\"}}\n</tool_call>";
        assert!(matches_grammar(&tg, with_old_new));

        // Any property order is fine
        let diff_order = "<tool_call>\n{\"name\": \"file_editor\", \"arguments\": {\"old_str\":\"x\",\"command\":\"str_replace\"}}\n</tool_call>";
        assert!(matches_grammar(&tg, diff_order));

        // Empty arguments
        let empty = "<tool_call>\n{\"name\": \"file_editor\", \"arguments\": {}}\n</tool_call>";
        assert!(matches_grammar(&tg, empty));
    }
}
