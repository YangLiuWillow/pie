//! Codex-facing Responses-API server inferlet for Pie.
//!
//! Speaks the OpenAI Responses API wire protocol that the Codex CLI uses
//! for custom providers (`wire_api = "responses"`), with two Pie-native
//! upgrades over a plain completion server:
//!
//! 1. **Content-addressed KV-session reuse** — Codex re-sends the whole
//!    conversation every turn; this server saves the post-generation
//!    context as an engine-side snapshot named by a hash of the item
//!    prefix, and the next request resumes it, paying prefill only for the
//!    new tool outputs. See `session.rs`.
//! 2. **Native tool calling** — tool schemas render through the model's own
//!    chat template, history replays byte-identically via the `Instruct`
//!    trait, and `<tool_call>` blocks stream out as structured
//!    `function_call` items. See `replay.rs` / `handler.rs`.
//!
//! ## Endpoints
//!
//! - `POST /responses` (also `/v1/responses`) — create a response
//!
//! ## Usage
//!
//! ```bash
//! # Build
//! cargo build --target wasm32-wasip2 --release
//!
//! # Serve via a Pie daemon (fresh WASM instance per request), then point
//! # Codex at it:
//! #   [model_providers.pie]
//! #   base_url = "http://127.0.0.1:8080/v1"
//! #   wire_api = "responses"
//! ```

mod filter;
mod handler;
mod replay;
mod session;
mod streaming;
mod types;

use wstd::http::body::IncomingBody;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Method, Request, Response, StatusCode};
use wstd::io::AsyncRead;

#[wstd::http_server]
async fn main(mut req: Request<IncomingBody>, res: Responder) -> Finished {
    let path = req.uri().path();
    let method = req.method().clone();

    match (method, path) {
        (Method::POST, "/responses") | (Method::POST, "/v1/responses") => {
            let mut body_bytes = Vec::new();
            if read_body(req.body_mut(), &mut body_bytes).await.is_err() {
                return error_response(res, 400, "Failed to read request body").await;
            }
            handler::handle_responses(body_bytes, res).await
        }

        (Method::GET, "/") => {
            let info = r#"{
  "name": "Pie Codex Responses Server",
  "version": "0.1.0",
  "endpoints": {
    "POST /responses": "Create a response (Codex wire_api = \"responses\")"
  }
}
"#;
            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(info.into_body())
                .unwrap();
            res.respond(response).await
        }

        (Method::OPTIONS, _) => {
            let response = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                .header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                .body("".into_body())
                .unwrap();
            res.respond(response).await
        }

        _ => not_found(res).await,
    }
}

/// Read the entire request body into a Vec<u8>
async fn read_body(body: &mut IncomingBody, buf: &mut Vec<u8>) -> Result<(), ()> {
    let mut chunk = [0u8; 4096];
    loop {
        match body.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => return Err(()),
        }
    }
    Ok(())
}

async fn error_response(res: Responder, status: u16, message: &str) -> Finished {
    let error = serde_json::json!({
        "error": {
            "type": "invalid_request",
            "message": message,
            "code": null,
            "param": null,
        }
    });

    let response = Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();

    res.respond(response).await
}

async fn not_found(res: Responder) -> Finished {
    let error = serde_json::json!({
        "error": {
            "type": "not_found",
            "message": "Endpoint not found",
            "code": null,
            "param": null,
        }
    });

    let response = Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();

    res.respond(response).await
}
