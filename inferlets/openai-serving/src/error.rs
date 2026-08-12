//! OpenAI-shaped error bodies + the request validation that decides 400 vs
//! 500 (extracted from the validated handler; the HTTP plumbing stays in
//! the inferlet).
//!
//! Status discipline (both audits agree, opencode is stricter):
//! - Malformed request → **400** with [`error_body`]
//!   (`type: "invalid_request_error"`). opencode fails 400s fast.
//! - **500 is reserved for genuine server faults** and must NEVER be
//!   returned for bad input: opencode's session-level retry loop replays
//!   5xx forever (unbounded attempts, 2 s·2^n backoff capped at 30 s);
//!   qwen-code mounts a 7×-app × 3×-SDK retry storm.
//! - Context length, split by WHEN it surfaces (openclaw AUDIT §3):
//!   - **mid-decode** KV exhaustion degrades to `finish_reason: "length"`
//!     with whatever streamed — never an error after the stream committed;
//!   - **pre-generation** "this prompt can never fit" should answer
//!     [`context_overflow_body`] as a 400: OpenClaw classifies the wording
//!     as `context_overflow` and triggers auto-compaction instead of
//!     failing; opencode fails the 400 fast and compacts client-side.
//!     Wiring is a seam until the guest can query context capacity (no WIT
//!     getter yet); until then the practical guard is the provider catalog's
//!     `contextWindow` matching the engine's `max_model_len`, which drives
//!     OpenClaw's own preflight.

use serde_json::{Value, json};

pub const INVALID_REQUEST_ERROR: &str = "invalid_request_error";
pub const SERVER_ERROR: &str = "server_error";

/// OpenAI error body: `{"error": {message, type, param: null, code: null}}`.
/// `param`/`code` are explicit nulls (matching OpenAI and the validated
/// implementation), not absent.
pub fn error_body(error_type: &str, message: &str) -> Value {
    json!({
        "error": {
            "message": message,
            "type": error_type,
            "param": null,
            "code": null,
        }
    })
}

/// Context-overflow 400 body. The wording is load-bearing (openclaw AUDIT
/// §3): "maximum context length" hits OpenClaw's `failover-explicit` table
/// and `context_length_exceeded` its `assistant-error` table, so the 400
/// classifies as `context_overflow` (message classification survives the
/// 400→format status rule) and triggers auto-compaction instead of a hard
/// fail. NEVER add rate-limit wording (`rate limit`, `too many requests`,
/// `tpm`, `tokens per minute`, `quota`) — each vetoes the overflow match.
/// Unlike [`error_body`], `code`/`param` are set: OpenAI's real overflow
/// error carries them and clients read `error.code`.
pub fn context_overflow_body(max_context_tokens: usize, requested_tokens: usize) -> Value {
    json!({
        "error": {
            "message": format!(
                "This model's maximum context length is {max_context_tokens} tokens. \
                 However, your messages resulted in {requested_tokens} tokens. \
                 Please reduce the length of the messages."
            ),
            "type": INVALID_REQUEST_ERROR,
            "param": "messages",
            "code": "context_length_exceeded",
        }
    })
}

/// Parse + validate a request body. `Err` means "answer 400 with this
/// message" — anything tolerable must be tolerated (unknown fields are
/// already ignored at the serde layer).
pub fn parse_request(body: &[u8]) -> Result<crate::types::ChatCompletionRequest, String> {
    let req: crate::types::ChatCompletionRequest =
        serde_json::from_slice(body).map_err(|e| format!("Invalid JSON: {e}"))?;
    if req.messages.is_empty() {
        return Err("`messages` must be a non-empty array".to_string());
    }
    Ok(req)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_body_golden() {
        assert_eq!(
            error_body(INVALID_REQUEST_ERROR, "Invalid JSON: oops"),
            json!({"error": {"message": "Invalid JSON: oops",
                             "type": "invalid_request_error",
                             "param": null, "code": null}})
        );
    }

    #[test]
    fn context_overflow_body_matches_openclaw_tables() {
        let v = context_overflow_body(32768, 41200);
        let msg = v["error"]["message"].as_str().unwrap();
        // The two phrases OpenClaw's overflow tables key on.
        assert!(msg.contains("maximum context length"));
        assert!(msg.contains("reduce the length of the messages"));
        assert_eq!(v["error"]["code"], "context_length_exceeded");
        assert_eq!(v["error"]["param"], "messages");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        // Veto words must never appear (each disables the overflow match).
        for veto in ["rate limit", "too many requests", "tpm", "tokens per minute", "quota"] {
            assert!(!msg.to_lowercase().contains(veto), "veto word {veto:?} present");
        }
    }

    #[test]
    fn parse_request_rejects_garbage_and_empty_messages() {
        assert!(parse_request(b"not json").unwrap_err().starts_with("Invalid JSON"));
        assert!(parse_request(br#"{"messages":[]}"#).unwrap_err().contains("non-empty"));
        assert!(parse_request(br#"{"messages":[{"role":"user","content":"hi"}]}"#).is_ok());
    }
}
