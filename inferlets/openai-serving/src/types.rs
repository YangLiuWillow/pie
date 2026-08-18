//! Wire types for the OpenAI chat-completions request surface — scoped to
//! what opencode and qwen-code actually send (see
//! `tests/inferlets/fixtures/opencode/AUDIT.md` §1 and the qwen-code audit)
//! plus the common fields other OpenAI-SDK clients add. Unknown fields must
//! be IGNORED, never 400'd: opencode sends `tool_choice:"auto"`, qwen-code
//! injects provider-convention fields like `chat_template_kwargs` freely,
//! and both evolve. Serde's default behavior (no `deny_unknown_fields`
//! anywhere in this module) is load-bearing.

use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, Debug)]
pub struct ChatCompletionRequest {
    /// B-2 opt-in: server-side context editing for this request. Absent means
    /// the server attends to everything the client sent, which is the only
    /// safe default — see `context_policy`.
    #[serde(default)]
    pub pie_context_policy: Option<crate::context_policy::ContextPolicy>,

    /// Informational — the runtime picks the first available model.
    #[serde(default)]
    pub model: Option<String>,

    pub messages: Vec<ChatMessage>,

    /// Absent (not `[]`) on opencode's title side-call — `default` covers it.
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

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub stop: Option<StopField>,

    /// vLLM/SGLang convention; qwen-code injects
    /// `{"enable_thinking": false}` for `^qwen` models with reasoning
    /// disabled on non-DashScope endpoints. opencode never sends it.
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

    /// Did the caller explicitly ask for the thinking channel?
    ///
    /// Absent is treated as NO, which is not what the templates do — their
    /// `enable_thinking` defaults to true — and is deliberate for now:
    ///
    ///   * every guest hardcoded the no-think cue until this change, so
    ///     defaulting to thinking would silently flip the rendering of every
    ///     existing client (opencode never sends the field at all); and
    ///   * `generation_suffix` is still `""` for Qwen3.5/3.6, so the
    ///     thinking-on cue currently renders a bare assistant header where the
    ///     template renders `<think>\n`. Defaulting to thinking would default
    ///     to the wrong cue. All 13 thinking cells of the parity matrix fail
    ///     on exactly that.
    ///
    /// Flipping the default to match the templates is a separate, deliberate
    /// change, and it is gated on that field being right.
    pub fn thinking_requested(&self) -> bool {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|v| v.get("enable_thinking"))
            .and_then(Value::as_bool)
            == Some(true)
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

#[derive(Deserialize, Default, Debug)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,

    /// String or parts-array. opencode sends system/user/tool content as
    /// plain strings (parts only with attachments); qwen-code sends user
    /// and tool content as `[{type:"text", text}, …]`. Both normalize to a
    /// string via [`MessageContent::as_text`].
    #[serde(default)]
    pub content: Option<MessageContent>,

    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallIn>>,

    /// Present on `role:"tool"` turns.
    #[serde(default)]
    pub tool_call_id: Option<String>,

    /// Echoed back by qwen-code on assistant turns (opencode round-trips it
    /// only if the server streamed it — we never do); excluded from both
    /// the rendered token stream and the snapshot address (H17: we serve
    /// the no-think channel, so reasoning must not perturb replay).
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    pub fn text(&self) -> String {
        self.content
            .as_ref()
            .map(MessageContent::as_text)
            .unwrap_or_default()
    }

    /// Content normalized to `None` when empty. opencode replays assistant
    /// tool-call turns with `content: ""` (empty string, not null/absent) —
    /// an empty string must render like no content at all, or the replayed
    /// token stream diverges from what generation produced.
    pub fn text_opt(&self) -> Option<String> {
        Some(self.text()).filter(|s| !s.is_empty())
    }

    pub fn calls(&self) -> &[ToolCallIn] {
        self.tool_calls.as_deref().unwrap_or(&[])
    }
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Flatten to text. Multiple text parts join with `\n` — matching
    /// vLLM's chat-content normalization, which the wire fixtures were
    /// captured against.
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter(|p| p.part_type == "text")
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct ContentPart {
    #[serde(rename = "type", default)]
    pub part_type: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Deserialize, Debug)]
pub struct ToolCallIn {
    #[serde(default)]
    pub id: String,
    pub function: ToolCallFunction,
}

#[derive(Deserialize, Debug)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments object, OpenAI style.
    #[serde(default)]
    pub arguments: String,
}

#[derive(Deserialize, Debug)]
pub struct ToolSpec {
    pub function: ToolSpecFunction,
}

#[derive(Deserialize, Debug)]
pub struct ToolSpecFunction {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Opaque JSON Schema. opencode's includes `$schema` draft-2020-12
    /// keys and `maximum: 9007199254740991` (2^53−1) — pass through, never
    /// validate or 400. (`serde_json::Value` round-trips these losslessly:
    /// the bound fits an i64 exactly.)
    #[serde(default)]
    pub parameters: Value,
}

/// Canonical schema envelopes: `{name, description, parameters}` per tool
/// (the shape the host tool-template capability consumes, verified against
/// Qwen's template by `openhands-completion`), **name-sorted** by function
/// name. The same strings feed BOTH the rendered prompt and the snapshot
/// address, so this is the one place the ordering is decided.
///
/// opencode already name-sorts tools on the wire (`session/llm/request.ts:184`),
/// so the sort is normally a no-op — it exists so a client that reorders its
/// tool list between turns still hashes to the same address. Serialization
/// goes through `serde_json::Value`, whose object keys re-serialize in
/// deterministic (sorted) order regardless of wire key order — deliberate:
/// the address must not depend on client key ordering either.
pub fn tool_schema_envelopes(tools: &[ToolSpec]) -> Vec<String> {
    let mut sorted: Vec<&ToolSpec> = tools.iter().collect();
    sorted.sort_by(|a, b| a.function.name.cmp(&b.function.name));
    sorted
        .iter()
        .map(|t| {
            python_json(&serde_json::json!({
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            }))
        })
        .collect()
}

/// Serialize like Python's `json.dumps` defaults — `", "` and `": "`
/// separators — because that is what HF's Jinja `tojson` filter emits inside
/// `<tools>` blocks, and the rendered schema text must match the reference
/// template byte-for-byte (parity harness divergence D3). serde_json's
/// compact form (`,`/`:`) is NOT wire-compatible with the fine-tuned prompt.
/// (Caveat: `json.dumps` also escapes non-ASCII by default; fixtures are
/// ASCII-clean so this is unhandled until parity says otherwise.)
///
/// This string feeds both the rendered prompt AND the snapshot address
/// (`session::snapshot_address`), which must always change together.
pub fn python_json(value: &serde_json::Value) -> String {
    struct PyFmt;
    impl serde_json::ser::Formatter for PyFmt {
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if !first {
                w.write_all(b", ")?;
            }
            Ok(())
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if !first {
                w.write_all(b", ")?;
            }
            Ok(())
        }
    }
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFmt);
    serde::Serialize::serialize(value, &mut ser).expect("Value serialization is infallible");
    String::from_utf8(out).expect("serde_json emits UTF-8")
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
    fn opencode_empty_string_assistant_content_is_none() {
        // AUDIT §1b: assistant replay carries content:"" + tool_calls.
        let msg: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "read", "arguments": "{}"}}]
        }))
        .unwrap();
        assert_eq!(msg.text(), "");
        assert!(msg.text_opt().is_none());
        assert_eq!(msg.calls().len(), 1);
    }

    #[test]
    fn tolerates_schema_noise_and_unknown_fields() {
        // AUDIT §1c: $schema + maximum: 2^53−1 inside parameters, and
        // tool_choice at top level, must parse without error.
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "auto",
            "tools": [{"type": "function", "function": {
                "name": "bash", "description": "d",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {"timeout": {"type": "number",
                                               "maximum": 9007199254740991u64}}
                }}}]
        }))
        .unwrap();
        let envs = tool_schema_envelopes(&req.tools);
        assert!(envs[0].contains("json-schema.org/draft/2020-12/schema"));
        assert!(envs[0].contains("9007199254740991"));
    }

    #[test]
    fn tool_schema_envelopes_are_name_sorted() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"function": {"name": "write", "description": "", "parameters": {}}},
                {"function": {"name": "bash", "description": "", "parameters": {}}},
                {"function": {"name": "read", "description": "", "parameters": {}}}
            ]
        }))
        .unwrap();
        let names: Vec<String> = tool_schema_envelopes(&req.tools)
            .iter()
            .map(|s| serde_json::from_str::<Value>(s).unwrap()["name"]
                .as_str()
                .unwrap()
                .to_string())
            .collect();
        assert_eq!(names, ["bash", "read", "write"]);
    }
}
