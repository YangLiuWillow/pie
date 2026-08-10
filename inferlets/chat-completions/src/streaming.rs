//! SSE chunk framing for the chat-completions stream.
//!
//! Contract notes (audit §1 hard-requirements table):
//! - `Content-Type: text/event-stream` is set by the handler; every event
//!   here is a `data: {json}\n\n` frame, terminated by `data: [DONE]`.
//! - The final content-bearing chunk of a turn must carry `finish_reason`
//!   (`"stop"`/`"length"`; `"tool_calls"` for tool turns) — qwen-code
//!   raises `NO_FINISH_REASON` and retries otherwise. Never emit
//!   `"error_finish"`.
//! - Keepalives are empty-delta chunks, not SSE comments: qwen-code's
//!   240 s idle watchdog counts *chunks* delivered by the OpenAI SDK, and
//!   comments never surface through it. Empty deltas are valid per the
//!   OpenAI streaming spec and ignored by accumulation.
//! - When `stream_options.include_usage` is set, a final `choices: []`
//!   usage chunk precedes `[DONE]`; KV-session reuse is reported via
//!   `usage.prompt_tokens_details.cached_tokens`.

use serde_json::{Value, json};

pub struct ChunkMeta {
    pub id: String,
    pub model: String,
    pub created: i64,
}

impl ChunkMeta {
    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> String {
        sse(&json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "logprobs": null,
                "finish_reason": finish_reason,
            }],
        }))
    }

    /// First chunk of every stream: announces the assistant role.
    pub fn role_chunk(&self) -> String {
        self.chunk(json!({"role": "assistant", "content": ""}), None)
    }

    /// Empty-delta keepalive (resets the client's idle watchdog during
    /// long prefills without affecting accumulated content).
    pub fn keepalive(&self) -> String {
        self.chunk(json!({}), None)
    }

    pub fn content_delta(&self, text: &str) -> String {
        self.chunk(json!({"content": text}), None)
    }

    /// One complete tool call in a single delta. The OpenAI SDK accumulates
    /// `tool_calls` deltas by `index`; a whole call in one chunk is valid
    /// and keeps id/name/arguments atomic (audit: a call without a unique
    /// `id` is silently dropped by `cleanOrphanedToolCalls`).
    pub fn tool_call_delta(&self, index: usize, call_id: &str, name: &str, args: &str) -> String {
        self.chunk(
            json!({"tool_calls": [{
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }]}),
            None,
        )
    }

    /// Terminal chunk of the turn's content: empty delta + finish_reason.
    pub fn finish_chunk(&self, finish_reason: &str) -> String {
        self.chunk(json!({}), Some(finish_reason))
    }

    /// `include_usage` final chunk: `choices` empty per spec.
    pub fn usage_chunk(&self, prompt: u32, completion: u32, cached: u32) -> String {
        sse(&json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": usage_object(prompt, completion, cached),
        }))
    }

    pub fn done() -> String {
        "data: [DONE]\n\n".to_string()
    }
}

pub fn usage_object(prompt: u32, completion: u32, cached: u32) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    })
}

fn sse(v: &Value) -> String {
    format!("data: {}\n\n", v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ChunkMeta {
        ChunkMeta { id: "chatcmpl-x".into(), model: "m".into(), created: 1 }
    }

    fn parse(frame: &str) -> Value {
        let payload = frame.strip_prefix("data: ").unwrap().trim_end();
        serde_json::from_str(payload).unwrap()
    }

    #[test]
    fn finish_chunk_carries_finish_reason() {
        let v = parse(&meta().finish_chunk("tool_calls"));
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(v["object"], "chat.completion.chunk");
    }

    #[test]
    fn usage_chunk_has_empty_choices_and_cached_tokens() {
        let v = parse(&meta().usage_chunk(100, 20, 80));
        assert_eq!(v["choices"].as_array().unwrap().len(), 0);
        assert_eq!(v["usage"]["total_tokens"], 120);
        assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 80);
    }

    #[test]
    fn tool_call_delta_is_atomic() {
        let v = parse(&meta().tool_call_delta(0, "call_i_0", "read_file", "{}"));
        let tc = &v["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_i_0");
        assert_eq!(tc["function"]["name"], "read_file");
        assert_eq!(v["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn done_terminator() {
        assert_eq!(ChunkMeta::done(), "data: [DONE]\n\n");
    }
}
