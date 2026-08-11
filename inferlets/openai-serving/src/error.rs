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
//! - Never a context-length 400 (fires client-side full-history
//!   compaction): overflow/generation faults degrade to
//!   `finish_reason: "length"` with whatever streamed.

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
    fn parse_request_rejects_garbage_and_empty_messages() {
        assert!(parse_request(b"not json").unwrap_err().starts_with("Invalid JSON"));
        assert!(parse_request(br#"{"messages":[]}"#).unwrap_err().contains("non-empty"));
        assert!(parse_request(br#"{"messages":[{"role":"user","content":"hi"}]}"#).is_ok());
    }
}
