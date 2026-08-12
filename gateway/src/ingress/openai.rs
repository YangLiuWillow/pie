//! OpenAI-compatible ingress: `POST /v1/chat/completions` (+ `/v1/models`,
//! `/health`) for stock OpenAI-speaking clients (opencode, curl, SDKs).
//!
//! The gateway stays thin: the request body is handed **verbatim** to the
//! serving inferlet (`LaunchProcess.input`), which owns all OpenAI wire logic
//! (parsing, rendering, generation, chunk construction — see the
//! `pie-openai-serving` crate). This adapter only:
//!   1. maps `Authorization: Bearer` → the trust-edge [`Identity`] (§5),
//!   2. launches the configured inferlet over the same `Sessions::create`
//!      path as `http.rs`,
//!   3. re-frames the inferlet's message events as SSE (streaming) or a JSON
//!      body (non-streaming), injecting empty-delta keepalive chunks during
//!      inferlet silence (see [`chunk_event_stream`]).
//!
//! ## Gateway ⇄ inferlet envelope
//!
//! Every `session::send` payload from the serving inferlet is one JSON
//! message on this contract:
//!   - first message: `{"status": <u16>}` — the HTTP status to respond with.
//!   - streaming request (`"stream": true`): each subsequent message is one
//!     ready-made `chat.completion.chunk` JSON document, forwarded on its own
//!     `data:` line verbatim; the gateway appends `data: [DONE]` after the
//!     turn's clean `Eos`.
//!   - non-streaming: exactly one subsequent message, the full response (or
//!     OpenAI error) body.
//! Process `stdout`/`stderr` events are instrumentation, never wire data —
//! they are dropped here (the runtime logs them). A process `error` event maps
//! to 500 (genuine server fault — the 400-vs-500 discipline lives in the
//! inferlet, which must classify malformed input itself; opencode retries 5xx
//! without bound, so 500 is reserved for real faults).
//!
//! ## Affinity
//!
//! Requests carrying a client-session signal route [`Affinity::Keyed`] —
//! stable HRW on the derived key — so consecutive turns of one agent session
//! land on the worker holding that session's KV snapshots. Key sources, in
//! priority order (see [`extract_affinity_key`]): opencode's
//! `x-session-affinity` / `x-session-id` headers, OpenClaw's (path-B)
//! `session_id` header, then OpenClaw's `prompt_cache_key` body field
//! (= `sessionId:boundaryCount`, sent when `compat.supportsPromptCacheKey`).
//! Requests with no signal stay [`Affinity::Ephemeral`] (p2c load spread) —
//! deliberately NOT a hash of identity+prompt, which would herd all traffic
//! from one config onto one worker; revisit with multi-worker evidence.
//! Single-worker deployments are unaffected either way.

use std::convert::Infallible;

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use futures::Stream;
use serde_json::{Value, json};

use crate::GatewayState;
use crate::ingress::identity;
use crate::session::{Affinity, Identity, SessionHandle, TokenRx, TurnInput};
use pie_client_api::{ClientMessage, ServerMessage};
use pie_worker_rpc::{Priority, Tokens};

/// OpenAI-style error body. Mirrors `pie-openai-serving`'s shape; duplicated
/// here (two `json!` lines) rather than importing the crate — the gateway must
/// not grow a dependency on serving-inferlet internals.
fn error_body(status: StatusCode, message: &str, err_type: &str) -> Response {
    (
        status,
        Json(json!({
            "error": { "message": message, "type": err_type, "param": null, "code": null }
        })),
    )
        .into_response()
}

/// Identity for the OpenAI surface: the trust-edge header when present
/// (same contract as every other ingress), otherwise derived from the Bearer
/// token — tenant `default`, user keyed by a hash prefix of the token so
/// per-key attribution survives without storing the secret. The token is
/// **not** verified here: like `x-pie-identity`, key checking is an edge
/// concern (private bind / mTLS / edge proxy), enforced at deploy.
fn extract_identity(headers: &HeaderMap) -> Result<Identity, String> {
    if headers.contains_key("x-pie-identity") {
        return identity::extract(headers).map_err(|e| e.to_string());
    }
    let bearer = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .filter(|t| !t.is_empty())
        .ok_or_else(|| "missing Authorization: Bearer or x-pie-identity".to_string())?;

    let digest = blake3::hash(bearer.as_bytes()).to_hex();
    let mut synth = HeaderMap::new();
    let ident_val = format!("default/key-{}", &digest.as_str()[..8]);
    synth.insert(
        "x-pie-identity",
        ident_val.parse().map_err(|_| "identity encode".to_string())?,
    );
    // Carry the tracing headers through so §5 extraction sees them.
    for h in ["x-forwarded-for", "x-request-id"] {
        if let Some(v) = headers.get(h) {
            synth.insert(h, v.clone());
        }
    }
    identity::extract(&synth).map_err(|e| e.to_string())
}

/// Derive the client-session affinity key, if the request carries one.
/// Priority: `x-session-affinity` (opencode's own sticky-routing header) →
/// `x-session-id` (opencode) → `session_id` (OpenClaw path B with
/// `compat.sendSessionAffinityHeaders`) → body `prompt_cache_key` (OpenClaw
/// with `compat.supportsPromptCacheKey`; value `sessionId:boundaryCount`).
/// The key is hashed to the router's `u64` HRW keyspace.
fn extract_affinity_key(headers: &HeaderMap, body: &Value) -> Option<u64> {
    let from_headers = ["x-session-affinity", "x-session-id", "session_id"]
        .iter()
        .find_map(|h| headers.get(*h).and_then(|v| v.to_str().ok()))
        .filter(|s| !s.is_empty());
    let key = match from_headers {
        Some(s) => s,
        None => body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())?,
    };
    let digest = blake3::hash(key.as_bytes());
    Some(u64::from_le_bytes(
        digest.as_bytes()[..8].try_into().expect("blake3 ≥ 8 bytes"),
    ))
}

/// `GET /health` — liveness for OpenAI-client launch scripts and probes.
pub async fn health() -> &'static str {
    "ok"
}

/// `GET /v1/models` — minimal models listing. The engine serves exactly one
/// model; ingress doesn't know its name (that's worker-side), so this lists
/// the configured serving inferlet's alias. Clients (opencode included) treat
/// the id as an opaque slug the server echoes back.
pub async fn models() -> Response {
    Json(json!({
        "object": "list",
        "data": [ { "id": "pie", "object": "model", "created": 0, "owned_by": "pie" } ]
    }))
    .into_response()
}

/// `POST /v1/chat/completions`.
pub async fn chat_completions(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let ident = match extract_identity(&headers) {
        Ok(id) => id,
        Err(e) => return error_body(StatusCode::UNAUTHORIZED, &e, "authentication_error"),
    };

    // Minimal request inspection: the inferlet owns full parsing (and must
    // tolerate unknown fields), but the gateway needs the stream mode to pick
    // the response framing, and non-JSON can be rejected without a launch.
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        Ok(_) => {
            return error_body(
                StatusCode::BAD_REQUEST,
                "request body must be a JSON object",
                "invalid_request_error",
            );
        }
        Err(e) => {
            return error_body(
                StatusCode::BAD_REQUEST,
                &format!("invalid JSON: {e}"),
                "invalid_request_error",
            );
        }
    };
    let stream_mode = parsed.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let turn = TurnInput {
        message: ClientMessage::LaunchProcess {
            corr_id: 0,
            inferlet: CHAT_INFERLET.to_string(),
            input: String::from_utf8_lossy(&body).into_owned(),
            capture_outputs: true,
        },
        blobs: Vec::new(),
        priority: Priority::Normal,
    };

    let affinity = match extract_affinity_key(&headers, &parsed) {
        Some(key) => Affinity::Keyed(key),
        None => Affinity::Ephemeral,
    };
    let (handle, rx) = match state.sessions.create(ident, turn, affinity).await {
        Ok(pair) => pair,
        Err(e) => {
            return error_body(
                StatusCode::SERVICE_UNAVAILABLE,
                &format!("admission: {e}"),
                "server_error",
            );
        }
    };

    if stream_mode {
        stream_response(handle, rx).await
    } else {
        unary_response(handle, rx).await
    }
}

/// The serving inferlet this surface launches. Fixed for now (gateway `Config`
/// is `deny_unknown_fields`; making this configurable is a config-schema
/// change to take deliberately, not smuggle in). MUST be the full
/// `name@major.minor.patch` form — the engine's `ProgramName::parse` rejects
/// bare names, which turns every chat request into a 500 (found by the PA.3
/// acceptance-suite dry review; the ingress tests now pin the format).
const CHAT_INFERLET: &str = "chat-completions@0.1.0";

/// One inferlet message event, decoded from the turn's token stream.
enum Msg {
    /// `session::send` payload.
    Payload(String),
    /// Launch/ack failure or process `error` event (genuine fault).
    Fault(String),
    /// Clean end of turn.
    Eos,
    /// Channel closed without Eos (worker drop / cancel).
    Aborted,
}

/// Pull the next meaningful event, skipping acks and stdout/stderr
/// instrumentation.
async fn next_msg(rx: &mut TokenRx) -> Msg {
    loop {
        match rx.recv().await {
            Some(Tokens::Chunk(ServerMessage::Response { ok, result, .. })) => {
                if !ok {
                    return Msg::Fault(result);
                }
                // launch ack — skip
            }
            Some(Tokens::Chunk(ServerMessage::ProcessEvent { event, value, .. })) => {
                match event.as_str() {
                    "message" => return Msg::Payload(value),
                    "error" => return Msg::Fault(value),
                    // stdout/stderr = instrumentation; return value is not
                    // wire data on this contract (the envelope carries it).
                    _ => {}
                }
            }
            Some(Tokens::Chunk(ServerMessage::File { .. })) => {}
            Some(Tokens::Eos) => return Msg::Eos,
            None => return Msg::Aborted,
        }
    }
}

/// Parse the envelope's first message: `{"status": <u16>}`.
fn parse_status(payload: &str) -> Option<StatusCode> {
    let v: Value = serde_json::from_str(payload).ok()?;
    let code = v.get("status")?.as_u64()?;
    StatusCode::from_u16(u16::try_from(code).ok()?).ok()
}

/// Non-streaming: status header + exactly one body message.
async fn unary_response(handle: SessionHandle, mut rx: TokenRx) -> Response {
    let status = match next_msg(&mut rx).await {
        Msg::Payload(p) => match parse_status(&p) {
            Some(s) => s,
            None => {
                return error_body(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inferlet broke the envelope contract (bad status header)",
                    "server_error",
                );
            }
        },
        Msg::Fault(e) => {
            return error_body(StatusCode::INTERNAL_SERVER_ERROR, &e, "server_error");
        }
        Msg::Eos | Msg::Aborted => {
            return error_body(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inferlet ended without a response",
                "server_error",
            );
        }
    };
    let body = loop {
        match next_msg(&mut rx).await {
            Msg::Payload(p) => break p,
            Msg::Fault(e) => {
                return error_body(StatusCode::INTERNAL_SERVER_ERROR, &e, "server_error");
            }
            Msg::Eos | Msg::Aborted => {
                return error_body(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inferlet ended without a response body",
                    "server_error",
                );
            }
        }
    };
    drop(handle); // one-shot: close the session once the body is in hand
    (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

/// Streaming: SSE. The status header still arrives first; a non-200 means the
/// inferlet rejected the request before generating (e.g. 400) — respond as a
/// plain JSON error rather than a 200 SSE stream, which is what OpenAI-compat
/// clients (and opencode's SDK) expect.
async fn stream_response(handle: SessionHandle, mut rx: TokenRx) -> Response {
    match next_msg(&mut rx).await {
        Msg::Payload(p) => match parse_status(&p) {
            Some(s) if s.is_success() => { /* fall through to the stream */ }
            Some(s) => {
                // Pre-stream rejection: next message is the error body.
                return match next_msg(&mut rx).await {
                    Msg::Payload(body) => (
                        s,
                        [(axum::http::header::CONTENT_TYPE, "application/json")],
                        body,
                    )
                        .into_response(),
                    _ => error_body(s, "request rejected", "invalid_request_error"),
                };
            }
            None => {
                return error_body(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inferlet broke the envelope contract (bad status header)",
                    "server_error",
                );
            }
        },
        Msg::Fault(e) => {
            return error_body(StatusCode::INTERNAL_SERVER_ERROR, &e, "server_error");
        }
        Msg::Eos | Msg::Aborted => {
            return error_body(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inferlet ended before responding",
                "server_error",
            );
        }
    }

    // No axum comment keepalive: OpenClaw's SSE sanitizer DROPS frames whose
    // only content is a comment, and its idle watchdogs reset only on parsed
    // chunks (openclaw AUDIT §2a / D-1). Empty-delta chunks are the one
    // keepalive both audited clients accept (opencode's watchdog resets on
    // raw bytes, so a real chunk trivially satisfies it too) — injected by
    // `chunk_event_stream` during inferlet silence.
    Sse::new(chunk_event_stream(handle, rx)).into_response()
}

/// Keepalive cadence during inferlet silence (long prefills). Must stay
/// under the tightest client budget: OpenClaw caps the idle watchdog at 60 s
/// for cron-triggered turns (`run/llm-idle-timeout.ts:28`).
const KEEPALIVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(15);

/// Empty-delta `chat.completion.chunk` keepalive. Invisible to chunk
/// accumulation on both audited clients; resets OpenClaw's parsed-chunk idle
/// watchdog, which SSE comments cannot (openclaw AUDIT §2a). Mirrors the
/// id/model/created of the chunks the inferlet emits (opencode's acceptance
/// suite pins chunk-id consistency across a stream) — the role chunk arrives
/// before the prefill gap, so the mirror is populated before the first
/// keepalive can fire; the placeholder covers only the sub-millisecond
/// pre-role window.
fn keepalive_chunk(meta: &Option<(String, String, i64)>) -> String {
    let (id, model, created) = meta
        .clone()
        .unwrap_or_else(|| ("chatcmpl-keepalive".to_string(), "pie".to_string(), 0));
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "logprobs": null, "finish_reason": null}],
    })
    .to_string()
}

/// Remember the stream's chunk identity from the first payload that carries
/// one (the role chunk), for keepalive mirroring.
fn note_chunk_meta(meta: &mut Option<(String, String, i64)>, payload: &str) {
    if meta.is_some() {
        return;
    }
    if let Ok(v) = serde_json::from_str::<Value>(payload) {
        if let (Some(id), Some(model)) = (v["id"].as_str(), v["model"].as_str()) {
            *meta = Some((
                id.to_string(),
                model.to_string(),
                v["created"].as_i64().unwrap_or(0),
            ));
        }
    }
}

/// Forward each envelope payload as one `data:` line; `[DONE]` on clean Eos.
/// Mirrors `http.rs::token_event_stream`, minus the ServerMessage re-encode:
/// payloads are already the wire-ready chunk JSON. When the inferlet stays
/// silent past [`KEEPALIVE_INTERVAL`] (prefill), an empty-delta keepalive
/// chunk goes out instead — the one envelope exception where the gateway
/// authors chunk JSON itself (documented at [`keepalive_chunk`]).
fn chunk_event_stream(
    handle: SessionHandle,
    rx: TokenRx,
) -> impl Stream<Item = Result<Event, Infallible>> {
    enum St {
        Streaming {
            rx: TokenRx,
            handle: SessionHandle,
            meta: Option<(String, String, i64)>,
        },
        End,
    }

    futures::stream::unfold(
        St::Streaming { rx, handle, meta: None },
        |st| async move {
            match st {
                St::Streaming { mut rx, handle, mut meta } => {
                    match tokio::time::timeout(KEEPALIVE_INTERVAL, next_msg(&mut rx)).await {
                        Err(_elapsed) => {
                            let ka = keepalive_chunk(&meta);
                            Some((
                                Ok(Event::default().data(ka)),
                                St::Streaming { rx, handle, meta },
                            ))
                        }
                        Ok(Msg::Payload(p)) => {
                            note_chunk_meta(&mut meta, &p);
                            Some((
                                Ok(Event::default().data(p)),
                                St::Streaming { rx, handle, meta },
                            ))
                        }
                        Ok(Msg::Eos) => {
                            let _ = &handle; // dropped after End ⇒ session closes
                            Some((Ok(Event::default().data("[DONE]")), St::End))
                        }
                        // Mid-stream fault/abort: the 200 is already on the
                        // wire, so surface an SSE error event (clients treat a
                        // broken stream as retryable; an explicit event beats a
                        // silent hang).
                        Ok(Msg::Fault(e)) => {
                            Some((Ok(Event::default().event("error").data(e)), St::End))
                        }
                        Ok(Msg::Aborted) => Some((
                            Ok(Event::default().event("error").data("stream aborted")),
                            St::End,
                        )),
                    }
                }
                St::End => None,
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keepalive_mirrors_stream_chunk_identity() {
        let mut meta = None;
        // Role chunk (what the inferlet actually emits first).
        note_chunk_meta(
            &mut meta,
            r#"{"id":"chatcmpl-abc","object":"chat.completion.chunk","created":42,"model":"qwen3","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}"#,
        );
        let ka: Value = serde_json::from_str(&keepalive_chunk(&meta)).unwrap();
        assert_eq!(ka["id"], "chatcmpl-abc");
        assert_eq!(ka["model"], "qwen3");
        assert_eq!(ka["created"], 42);
        // Empty delta, null finish — invisible to accumulation, resets
        // OpenClaw's parsed-chunk watchdog (openclaw AUDIT §2a).
        assert_eq!(ka["choices"][0]["delta"], json!({}));
        assert_eq!(ka["choices"][0]["finish_reason"], Value::Null);

        // First identity wins; later payloads don't rebind it.
        note_chunk_meta(&mut meta, r#"{"id":"other","model":"m2","created":9}"#);
        assert_eq!(meta.as_ref().unwrap().0, "chatcmpl-abc");
    }

    #[test]
    fn affinity_key_priority_and_stability() {
        let body_with_pck = json!({"prompt_cache_key": "sess-1:0", "stream": true});
        let empty_body = json!({"stream": true});

        // No signal anywhere → None (stays Ephemeral/p2c).
        assert_eq!(extract_affinity_key(&HeaderMap::new(), &empty_body), None);

        // Body prompt_cache_key alone keys affinity (OpenClaw compat path).
        let k_body = extract_affinity_key(&HeaderMap::new(), &body_with_pck);
        assert!(k_body.is_some());
        // Same key → same hash (HRW stability across turns).
        assert_eq!(k_body, extract_affinity_key(&HeaderMap::new(), &body_with_pck));
        // Different key → different hash (overwhelmingly).
        let other = json!({"prompt_cache_key": "sess-2:0"});
        assert_ne!(k_body, extract_affinity_key(&HeaderMap::new(), &other));

        // Headers beat the body field; x-session-affinity beats x-session-id.
        let mut headers = HeaderMap::new();
        headers.insert("x-session-id", "ses_b".parse().unwrap());
        let k_sid = extract_affinity_key(&headers, &body_with_pck);
        assert_ne!(k_sid, k_body);
        headers.insert("x-session-affinity", "ses_a".parse().unwrap());
        let k_aff = extract_affinity_key(&headers, &body_with_pck);
        assert_ne!(k_aff, k_sid);

        // Empty header values are ignored, not hashed.
        let mut empty_h = HeaderMap::new();
        empty_h.insert("x-session-id", "".parse().unwrap());
        assert_eq!(extract_affinity_key(&empty_h, &empty_body), None);
    }

    #[test]
    fn keepalive_placeholder_before_first_chunk() {
        let ka: Value = serde_json::from_str(&keepalive_chunk(&None)).unwrap();
        assert_eq!(ka["id"], "chatcmpl-keepalive");
        assert_eq!(ka["object"], "chat.completion.chunk");
        assert_eq!(ka["choices"][0]["delta"], json!({}));
    }
}
