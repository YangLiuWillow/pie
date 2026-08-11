//! chat.completion.chunk assembly + the shim envelope.
//!
//! The pre-rewrite inferlet wrote SSE frames straight onto its own HTTP
//! response; under the rewrite the transport is the gateway session channel,
//! so every outbound event is one `session::send` of a JSON envelope
//! `{"req_id", "event", "data"}` and the shim owns the SSE dressing
//! (`data: ` prefix, keepalive comments, `data: [DONE]`).
//!
//! Contract notes carried over from the audit §1 hard-requirements table:
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

/// Envelope events the shim understands.
pub fn ev_chunk(req_id: &str, chunk: &Value) -> String {
    json!({"req_id": req_id, "event": "chunk", "data": chunk}).to_string()
}

/// End of stream for this request ([DONE] is the shim's job).
pub fn ev_done(req_id: &str) -> String {
    json!({"req_id": req_id, "event": "done"}).to_string()
}

/// Non-streaming request: the complete chat.completion object.
pub fn ev_response(req_id: &str, response: &Value) -> String {
    json!({"req_id": req_id, "event": "response", "data": response}).to_string()
}

/// OpenAI error shape + an HTTP status hint for the shim (400 for client
/// faults, 500 for genuine server faults — 500s trigger retry storms, so
/// the handler only uses them for real breakage).
pub fn ev_error(req_id: &str, status: u16, error_type: &str, message: &str) -> String {
    json!({
        "req_id": req_id,
        "event": "error",
        "data": {
            "status": status,
            "error": {
                "message": message,
                "type": error_type,
                "param": null,
                "code": null,
            }
        }
    })
    .to_string()
}

pub struct ChunkMeta {
    pub id: String,
    pub model: String,
    pub created: i64,
}

impl ChunkMeta {
    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> Value {
        json!({
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
        })
    }

    /// First chunk of every stream: announces the assistant role.
    pub fn role_chunk(&self) -> Value {
        self.chunk(json!({"role": "assistant", "content": ""}), None)
    }

    /// Empty-delta keepalive (resets the client's idle watchdog during
    /// long prefills without affecting accumulated content).
    pub fn keepalive(&self) -> Value {
        self.chunk(json!({}), None)
    }

    pub fn content_delta(&self, text: &str) -> Value {
        self.chunk(json!({"content": text}), None)
    }

    /// One complete tool call in a single delta. The OpenAI SDK accumulates
    /// `tool_calls` deltas by `index`; a whole call in one chunk is valid
    /// and keeps id/name/arguments atomic (audit: a call without a unique
    /// `id` is silently dropped by `cleanOrphanedToolCalls`).
    pub fn tool_call_delta(&self, index: usize, call_id: &str, name: &str, args: &str) -> Value {
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
    pub fn finish_chunk(&self, finish_reason: &str) -> Value {
        self.chunk(json!({}), Some(finish_reason))
    }

    /// `include_usage` final chunk: `choices` empty per spec.
    pub fn usage_chunk(&self, prompt: u32, completion: u32, cached: u32) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": usage_object(prompt, completion, cached),
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ChunkMeta {
        ChunkMeta { id: "chatcmpl-x".into(), model: "m".into(), created: 1 }
    }

    #[test]
    fn finish_chunk_carries_finish_reason() {
        let v = meta().finish_chunk("tool_calls");
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(v["object"], "chat.completion.chunk");
    }

    #[test]
    fn usage_chunk_has_empty_choices_and_cached_tokens() {
        let v = meta().usage_chunk(100, 20, 80);
        assert_eq!(v["choices"].as_array().unwrap().len(), 0);
        assert_eq!(v["usage"]["total_tokens"], 120);
        assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 80);
    }

    #[test]
    fn tool_call_delta_is_atomic() {
        let v = meta().tool_call_delta(0, "call_i_0", "read_file", "{}");
        let tc = &v["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_i_0");
        assert_eq!(tc["function"]["name"], "read_file");
        assert_eq!(v["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn envelope_events_round_trip() {
        let c: Value = serde_json::from_str(&ev_chunk("r1", &meta().role_chunk())).unwrap();
        assert_eq!(c["req_id"], "r1");
        assert_eq!(c["event"], "chunk");
        assert_eq!(c["data"]["choices"][0]["delta"]["role"], "assistant");

        let d: Value = serde_json::from_str(&ev_done("r1")).unwrap();
        assert_eq!(d["event"], "done");

        let e: Value = serde_json::from_str(&ev_error("r1", 400, "invalid_request_error", "bad")).unwrap();
        assert_eq!(e["data"]["status"], 400);
        assert_eq!(e["data"]["error"]["type"], "invalid_request_error");
        assert_eq!(e["data"]["error"]["param"], Value::Null);
    }
}
