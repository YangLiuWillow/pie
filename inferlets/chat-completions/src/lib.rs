//! OpenAI-compatible chat-completions HTTP server inferlet for Pie.
//!
//! Serves stock qwen-code (and any OpenAI-SDK client) with zero client-side
//! changes: point `OPENAI_BASE_URL` at this daemon. Cross-turn KV reuse via
//! content-addressed engine-side snapshots (see `session.rs`); wire
//! contract per `docs/qwen-code-rl-audit.md` §1.
//!
//! ## Endpoints
//!
//! - `POST /chat/completions`, `POST /v1/chat/completions` — create a
//!   completion (SSE when `"stream": true`)
//! - `GET /`, `GET /health` — server info / liveness
//!
//! ## Usage
//!
//! ```bash
//! # Build
//! cargo build -p chat-completions --target wasm32-wasip2 --release
//!
//! # Launch as an HTTP daemon (see integrations/qwen-code/launch_daemon.py)
//! # then:
//! curl -N http://127.0.0.1:8123/v1/chat/completions \
//!   -H 'Content-Type: application/json' \
//!   -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true,
//!        "stream_options":{"include_usage":true}}'
//! ```

mod filter;
mod handler;
mod render;
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
        // Accept both the bare and `/v1/`-prefixed path so callers can
        // point the OpenAI SDK at either `…/v1` or the bare host.
        (Method::POST, "/chat/completions") | (Method::POST, "/v1/chat/completions") => {
            let mut body_bytes = Vec::new();
            if read_body(req.body_mut(), &mut body_bytes).await.is_err() {
                return handler::error_response(
                    res,
                    400,
                    "invalid_request_error",
                    "Failed to read request body",
                )
                .await;
            }
            handler::handle_chat_completions(body_bytes, res).await
        }

        (Method::GET, "/") | (Method::GET, "/health") => {
            let info = serde_json::json!({
                "name": "Pie chat-completions server",
                "version": env!("CARGO_PKG_VERSION"),
                "endpoints": {"POST /v1/chat/completions": "create a chat completion"},
                "status": "ok",
            });
            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(info.to_string().into_body())
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

        _ => {
            let response = Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header("Content-Type", "application/json")
                .body(
                    serde_json::json!({"error": {
                        "message": "Endpoint not found",
                        "type": "invalid_request_error",
                        "param": null,
                        "code": null,
                    }})
                    .to_string()
                    .into_body(),
                )
                .unwrap();
            res.respond(response).await
        }
    }
}

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
