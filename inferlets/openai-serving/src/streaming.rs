//! Chunk construction for the chat-completions stream — pure JSON builders,
//! no HTTP and no flushing (framing/keepalive cadence is the caller's job:
//! per the integration plan, the inferlet emits ready-made chunk JSON and
//! the gateway wraps it in SSE `data:` lines).
//!
//! Contract notes (qwen-code audit §1 + opencode AUDIT §3–§5):
//! - Every content-bearing turn ends with a `finish_reason` chunk
//!   (`"stop"`/`"length"`; `"tool_calls"` for tool turns) — qwen-code
//!   raises `NO_FINISH_REASON` and retries otherwise (opencode maps a
//!   missing one to `"unknown"` without error, but emit it anyway). Never
//!   emit `"error_finish"`.
//! - The FIRST delta for a tool_call index must carry both `id` and
//!   `function.name`, or opencode's AI SDK throws
//!   `InvalidResponseDataError` and kills the stream. `tool_call_delta`
//!   emits the whole call atomically (id + name + arguments in one delta),
//!   which satisfies both clients — and a call without a unique `id` is
//!   silently dropped by qwen-code's `cleanOrphanedToolCalls`.
//! - Keepalives: opencode's `chunkTimeout` watchdog resets on raw bytes, so
//!   SSE comments ([`sse_ping`]) are the preferred keepalive; qwen-code's
//!   watchdog counts SDK-delivered chunks, so it needs empty-delta chunks
//!   ([`ChunkMeta::keepalive`]) instead. Both are ignored by accumulation.
//! - When `stream_options.include_usage` is set, a final `choices: []`
//!   usage chunk precedes `[DONE]`; KV-session reuse is reported via
//!   `usage.prompt_tokens_details.cached_tokens` (must be ≤ prompt_tokens
//!   or opencode's computed `noCache` goes negative).

use serde_json::{Value, json};

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

    /// First chunk of every stream: announces the assistant role. The
    /// `content: ""` alongside the role matches what the validated
    /// qwen-code implementation shipped; both SDKs accept it.
    pub fn role_chunk(&self) -> Value {
        self.chunk(json!({"role": "assistant", "content": ""}), None)
    }

    /// Empty-delta keepalive (resets a chunk-counting client watchdog
    /// during long prefills without affecting accumulated content).
    pub fn keepalive(&self) -> Value {
        self.chunk(json!({}), None)
    }

    pub fn content_delta(&self, text: &str) -> Value {
        self.chunk(json!({"content": text}), None)
    }

    /// One complete tool call in a single delta. The OpenAI SDK accumulates
    /// `tool_calls` deltas by `index`; a whole call in one chunk is valid
    /// and keeps id/name/arguments atomic.
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

    /// `include_usage` final chunk: `choices` empty per spec
    /// (capture-verified accepted by opencode).
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

    /// Non-streaming `chat.completion` response body (curl/debug/acceptance
    /// path). `content`/`tool_calls` are `null` — not absent, matching the
    /// validated implementation — when empty.
    pub fn completion_response(
        &self,
        content: &str,
        tool_calls: &[(String, String, String)], // (id, name, arguments)
        finish_reason: &str,
        prompt: u32,
        completion: u32,
        cached: u32,
    ) -> Value {
        let tool_calls_json: Vec<Value> = tool_calls
            .iter()
            .map(|(id, name, args)| {
                json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })
            })
            .collect();
        json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": if content.is_empty() {
                        Value::Null
                    } else {
                        Value::String(content.to_string())
                    },
                    "tool_calls": if tool_calls_json.is_empty() {
                        Value::Null
                    } else {
                        Value::Array(tool_calls_json)
                    },
                },
                "logprobs": null,
                "finish_reason": finish_reason,
            }],
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

/// Wrap a chunk in an SSE `data:` frame.
pub fn sse_frame(v: &Value) -> String {
    format!("data: {v}\n\n")
}

/// Stream terminator. opencode's parser consumes it; a close without it
/// also terminates cleanly, but send it anyway (AUDIT §4).
pub fn sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}

/// SSE comment keepalive — capture-verified invisible to opencode's JSON
/// layer while resetting its raw-byte `chunkTimeout` watchdog (AUDIT §2).
pub fn sse_ping() -> String {
    ": ping\n\n".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ChunkMeta {
        ChunkMeta { id: "chatcmpl-x".into(), model: "m".into(), created: 1 }
    }

    // Exact-shape goldens: `Value` equality distinguishes an explicit null
    // from an absent key, so these pin field presence/absence exactly.

    #[test]
    fn role_chunk_golden() {
        assert_eq!(
            meta().role_chunk(),
            json!({
                "id": "chatcmpl-x", "object": "chat.completion.chunk",
                "created": 1, "model": "m",
                "choices": [{"index": 0,
                             "delta": {"role": "assistant", "content": ""},
                             "logprobs": null, "finish_reason": null}]
            })
        );
    }

    #[test]
    fn content_delta_golden_and_frame() {
        let v = meta().content_delta("Hello");
        assert_eq!(
            v,
            json!({
                "id": "chatcmpl-x", "object": "chat.completion.chunk",
                "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": {"content": "Hello"},
                             "logprobs": null, "finish_reason": null}]
            })
        );
        // No usage key on content chunks; delta carries no role/tool_calls.
        assert!(v.get("usage").is_none());
        let frame = sse_frame(&v);
        assert!(frame.starts_with("data: {"));
        assert!(frame.ends_with("}\n\n"));
    }

    #[test]
    fn tool_call_delta_golden() {
        // First (and only) delta for the index carries id AND function.name
        // — opencode's SDK throws InvalidResponseDataError otherwise.
        assert_eq!(
            meta().tool_call_delta(0, "call_i_0", "read", "{\"filePath\":\"/x\"}"),
            json!({
                "id": "chatcmpl-x", "object": "chat.completion.chunk",
                "created": 1, "model": "m",
                "choices": [{"index": 0,
                             "delta": {"tool_calls": [{
                                 "index": 0, "id": "call_i_0", "type": "function",
                                 "function": {"name": "read",
                                              "arguments": "{\"filePath\":\"/x\"}"}}]},
                             "logprobs": null, "finish_reason": null}]
            })
        );
    }

    #[test]
    fn finish_chunk_golden() {
        assert_eq!(
            meta().finish_chunk("tool_calls"),
            json!({
                "id": "chatcmpl-x", "object": "chat.completion.chunk",
                "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": {},
                             "logprobs": null, "finish_reason": "tool_calls"}]
            })
        );
    }

    #[test]
    fn usage_chunk_golden() {
        // choices MUST be [] (capture-verified accepted), cached_tokens
        // under prompt_tokens_details (that is where the SDK reads it).
        assert_eq!(
            meta().usage_chunk(100, 20, 80),
            json!({
                "id": "chatcmpl-x", "object": "chat.completion.chunk",
                "created": 1, "model": "m",
                "choices": [],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                          "total_tokens": 120,
                          "prompt_tokens_details": {"cached_tokens": 80}}
            })
        );
    }

    #[test]
    fn keepalive_is_empty_delta() {
        let v = meta().keepalive();
        assert_eq!(v["choices"][0]["delta"], json!({}));
        assert_eq!(v["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn completion_response_nulls_when_empty() {
        let v = meta().completion_response("", &[], "stop", 10, 0, 0);
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(v["choices"][0]["message"]["tool_calls"], Value::Null);

        let calls = [("call_1".to_string(), "read".to_string(), "{}".to_string())];
        let v = v_with_calls(&calls);
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["id"], "call_1");
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    }

    fn v_with_calls(calls: &[(String, String, String)]) -> Value {
        meta().completion_response("ok", calls, "tool_calls", 10, 5, 0)
    }

    #[test]
    fn sse_terminators() {
        assert_eq!(sse_done(), "data: [DONE]\n\n");
        assert_eq!(sse_ping(), ": ping\n\n");
    }
}
