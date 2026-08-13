//! The session envelope: how the client shim and this inferlet frame turns
//! over one sticky WebSocket.
//!
//! Strategy A had no envelope of its own — the gateway's OpenAI ingress owned
//! the framing, one process per request, and `session::send` payloads WERE the
//! response chunks. That works only because "which request is this?" is
//! answered by "the process it arrived in". A long-lived session process serves
//! many turns down one channel, so the turn id has to be on the wire.
//!
//! ## Protocol
//!
//! Client → inferlet, one JSON document per `signal_process`:
//!
//! ```jsonc
//! { "req_id": "…", "body": { /* verbatim OpenAI chat request */ } }
//! ```
//!
//! Inferlet → client, one JSON document per `session::send`:
//!
//! ```jsonc
//! { "req_id": "…", "event": "chunk",    "data": { /* chat.completion.chunk */ } }
//! { "req_id": "…", "event": "response", "data": { /* chat.completion       */ } }
//! { "req_id": "…", "event": "error",    "data": { "status": 400, "error": {…} } }
//! { "req_id": "…", "event": "done" }
//! ```
//!
//! `body` is passed through verbatim rather than pre-parsed by the shim: the
//! shim stays a transport and every wire decision keeps living in one place
//! (`pie-openai-serving`), which is also what keeps the A/B arms honest.
//!
//! ## Why `error` carries a status instead of the envelope leading with one
//!
//! Strategy A's contract is `{"status": <u16>}` FIRST, then the body — the
//! gateway needs a status before it can open an HTTP response. Here the shim
//! owns the HTTP response and can wait, so the status rides on the error event
//! and success needs none. This matters for the invariant that outlives both:
//! **never 500 on bad input** — opencode retries 5xx without bound, so
//! malformed requests get 400 and 500 is reserved for genuine faults.

use serde::Deserialize;
use serde_json::{Value, json};

/// One inbound turn.
#[derive(Deserialize)]
pub struct Envelope {
    #[serde(default)]
    pub req_id: String,
    pub body: Value,
}

/// Best-effort `req_id` recovery from a payload that failed to parse as an
/// [`Envelope`], so the shim can still route the error to the caller that is
/// blocked on it rather than leaving it to time out.
pub fn recover_req_id(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|v| v.get("req_id").and_then(|r| r.as_str()).map(String::from))
        .unwrap_or_default()
}

fn send(v: &Value) {
    inferlet::session::send(&v.to_string());
}

/// Emit an error event for a turn that never opened a stream. `status` is the
/// HTTP status the shim should answer with.
pub fn send_error(req_id: &str, status: u16, error_type: &str, message: &str) {
    let body = pie_openai_serving::error::error_body(error_type, message);
    send(&json!({
        "req_id": req_id,
        "event": "error",
        "data": { "status": status, "error": body.get("error").cloned().unwrap_or(Value::Null) },
    }));
}

/// The outbound half of one turn's envelope. Cheap to clone — it is just the
/// turn id — so the turn state machine can hold one without borrowing the
/// daemon.
#[derive(Clone)]
pub struct Sink {
    req_id: String,
}

impl Sink {
    pub fn new(req_id: &str) -> Self {
        Self { req_id: req_id.to_string() }
    }

    /// One ready-made `chat.completion.chunk`.
    pub fn chunk(&self, data: &Value) {
        send(&json!({ "req_id": self.req_id, "event": "chunk", "data": data }));
    }

    /// The single body of a non-streaming turn.
    pub fn response(&self, data: &Value) {
        send(&json!({ "req_id": self.req_id, "event": "response", "data": data }));
    }

    /// End of turn. The shim appends `data: [DONE]` on a streaming request.
    pub fn done(&self) {
        send(&json!({ "req_id": self.req_id, "event": "done" }));
    }
}
