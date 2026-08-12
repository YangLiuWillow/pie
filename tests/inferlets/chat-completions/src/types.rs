//! Wire types for the OpenAI chat-completions request surface — scoped to
//! what qwen-code actually sends (see `docs/qwen-code-rl-audit.md` §1) plus
//! the common fields other OpenAI-SDK clients add. Unknown fields must be
//! ignored, never 400'd: qwen-code injects provider-convention fields like
//! `chat_template_kwargs` freely.

use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    /// Informational — the runtime picks the first available model.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,

    pub messages: Vec<ChatMessage>,

    #[serde(default)]
    pub tools: Vec<ToolSpec>,

    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Newer OpenAI alias for `max_tokens`.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,

    #[serde(default)]
    pub stream: bool,

    #[serde(default)]
    pub stream_options: StreamOptions,

    /// Debug flag (C3 renderer parity, `docs/qwen-code-dev-port.md`):
    /// render-only turn — return the full rendered token ids + decoded text
    /// instead of generating. Always answered non-stream; never touches KV
    /// sessions. Not part of the OpenAI surface.
    #[serde(default)]
    pub echo_tokens: bool,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub stop: Option<StopField>,

    /// vLLM/SGLang convention; qwen-code injects
    /// `{"enable_thinking": false}` for `^qwen` models with reasoning
    /// disabled on non-DashScope endpoints.
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
}

impl ChatCompletionRequest {
    pub fn effective_max_tokens(&self, default: usize) -> usize {
        self.max_tokens
            .or(self.max_completion_tokens)
            .unwrap_or(default)
    }

    pub fn include_usage(&self) -> bool {
        self.stream_options.include_usage
    }

    /// `chat_template_kwargs.enable_thinking == false` → render the
    /// no-think channel (`/no_think` per user turn, H17 discipline).
    pub fn no_think(&self) -> bool {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|v| v.get("enable_thinking"))
            .and_then(Value::as_bool)
            == Some(false)
    }

    pub fn stop_strings(&self) -> Vec<String> {
        match &self.stop {
            None => Vec::new(),
            Some(StopField::One(s)) => vec![s.clone()],
            Some(StopField::Many(v)) => v.clone(),
        }
    }
}

#[derive(Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,

    /// String or parts-array; qwen-code sends user and tool content as
    /// `[{type:"text", text}, …]` and system content as a plain string.
    #[serde(default)]
    pub content: Option<MessageContent>,

    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallIn>>,

    /// Present on `role:"tool"` turns.
    #[serde(default)]
    pub tool_call_id: Option<String>,

    /// Echoed back by qwen-code on assistant turns; excluded from both the
    /// rendered token stream and the snapshot address (H17: we serve the
    /// no-think channel, so reasoning must not perturb replay).
    #[serde(default)]
    #[allow(dead_code)]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    pub fn text(&self) -> String {
        self.text_sep("\n")
    }

    /// Flatten with an explicit part separator — `""` for the Qwen3.5/3.6
    /// lineage, `"\n"` everywhere else.
    pub fn text_sep(&self, sep: &str) -> String {
        self.content
            .as_ref()
            .map(|c| c.as_text_sep(sep))
            .unwrap_or_default()
    }

    pub fn calls(&self) -> &[ToolCallIn] {
        self.tool_calls.as_deref().unwrap_or(&[])
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Flatten to text, joining multiple text parts with `sep`.
    ///
    /// The separator is dialect-dependent and NOT cosmetic. Qwen3's hermes
    /// captures were taken against vLLM's chat-content normalization, which
    /// joins with `\n`; Qwen3.5/3.6's template loops the parts and emits
    /// `item.text` with NOTHING between them. Decoding vLLM's own
    /// `/v1/chat/completions/render` output caught this — 3 bytes of
    /// divergence over 43 KB, on 146 of the 23 fixtures' messages.
    pub fn as_text_sep(&self, sep: &str) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter(|p| p.part_type == "text")
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join(sep),
        }
    }

    /// Legacy `\n` join, correct for the hermes and Coder lineages.
    pub fn as_text(&self) -> String {
        self.as_text_sep("\n")
    }
}

#[derive(Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type", default)]
    pub part_type: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Deserialize)]
pub struct ToolCallIn {
    #[serde(default)]
    pub id: String,
    pub function: ToolCallFunction,
}

#[derive(Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments object, OpenAI style.
    #[serde(default)]
    pub arguments: String,
}

#[derive(Deserialize)]
pub struct ToolSpec {
    pub function: ToolSpecFunction,
}

#[derive(Deserialize)]
pub struct ToolSpecFunction {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Value,
}

/// Schema envelopes rendered into the tool system prompt — the
/// `{name, description, parameters}` shape `openhands-completion` verified
/// against Qwen's template.
pub fn tool_schema_envelopes(tools: &[ToolSpec]) -> Vec<String> {
    // jinja-tojson serialization (spaced separators, insertion order): the
    // HF template renders each tool with `| tojson`, and byte parity there
    // decides whether the model recognizes its fine-tuning format (C3).
    tools
        .iter()
        .map(|t| {
            crate::render_text::tojson(&serde_json::json!({
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen_code_shaped_request() {
        let body = serde_json::json!({
            "model": "Qwen3.6-27B",
            "messages": [
                {"role": "system", "content": "You are Qwen Code."},
                {"role": "user", "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"}
                ]},
                {"role": "assistant", "content": null,
                 "reasoning_content": "hidden",
                 "tool_calls": [{"id": "call_abc", "type": "function",
                                 "function": {"name": "read_file",
                                              "arguments": "{\"file_path\":\"/x\"}"}}]},
                {"role": "tool", "tool_call_id": "call_abc",
                 "content": [{"type": "text", "text": "file body"}]}
            ],
            "max_tokens": 64000,
            "stream": true,
            "stream_options": {"include_usage": true},
            "tools": [{"type": "function", "function":
                       {"name": "read_file", "description": "d",
                        "parameters": {"type": "object"}}}],
            "chat_template_kwargs": {"enable_thinking": false},
            "some_future_field": {"ignored": true}
        });
        let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        assert_eq!(req.messages.len(), 4);
        assert_eq!(req.messages[1].text(), "part one\npart two");
        assert_eq!(req.messages[2].calls()[0].function.name, "read_file");
        assert_eq!(req.messages[3].tool_call_id.as_deref(), Some("call_abc"));
        assert_eq!(req.effective_max_tokens(4096), 64000);
        assert!(req.stream);
        assert!(req.include_usage());
        assert!(req.no_think());
        assert_eq!(tool_schema_envelopes(&req.tools).len(), 1);
    }

    #[test]
    fn minimal_request_gets_defaults() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert!(!req.stream);
        assert!(!req.include_usage());
        assert!(!req.no_think());
        assert_eq!(req.effective_max_tokens(4096), 4096);
        assert!(req.stop_strings().is_empty());
    }

    #[test]
    fn real_wire_fixtures_parse() {
        // Byte-exact qwen-code `--openai-logging` captures checked in for
        // the rl-completions work double as parsing fixtures here.
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../fixtures/rl_completions/wire"
        );
        let mut n = 0;
        for episode in std::fs::read_dir(dir).unwrap().flatten() {
            if !episode.path().is_dir() {
                continue;
            }
            for f in std::fs::read_dir(episode.path()).unwrap().flatten() {
                let name = f.file_name().to_string_lossy().into_owned();
                if !name.starts_with("openai-") || !name.ends_with(".json") {
                    continue;
                }
                let capture: Value =
                    serde_json::from_str(&std::fs::read_to_string(f.path()).unwrap()).unwrap();
                let req: ChatCompletionRequest =
                    serde_json::from_value(capture["request"].clone())
                        .unwrap_or_else(|e| panic!("{name}: {e}"));
                assert!(!req.messages.is_empty(), "{name}: empty messages");
                n += 1;
            }
        }
        assert!(n > 0, "no wire fixtures found under {dir}");
    }
}
