//! RL rollout completions server inferlet for Pie.
//!
//! Serves the wire contract rllm's model gateway expects from a worker in
//! **cumulative token mode** (pie-rl-verl-integration.md §3): the gateway
//! rewrites every turn ≥ 1 of a chat session into `POST /v1/completions`
//! with `prompt` as a *list of token ids* (the exact ids of the previous
//! turn's prompt + completion, extended by the renderer with the new
//! messages). The worker must echo those ids back untouched and report the
//! ids it generates — the trainer's multi-turn row merging depends on
//! byte-exact token accounting, not on text.
//!
//! ## Endpoints
//!
//! - `POST /v1/completions` (also `/completions`) — §3.1; `prompt` may be a
//!   token-id list (the real path) or a string (debug convenience).
//! - `GET /health` — 200, plain "ok" (gateway router polls every 10 s).
//! - `GET /` — server info.
//!
//! `POST /v1/chat/completions` (turn 0) is NOT implemented yet — planned
//! next; requires renderer-parity-verified template rendering (§6 R1).
//!
//! Logprobs: accepted leniently (`true`/`1`) but v1 returns none — rllm
//! trains without them (transform pads 0.0; PPO recomputes trainer-side).
//! Real per-token logprobs arrive with the Track B `sampled-logprob`
//! primitive (gap G2).
//!
//! ```bash
//! cargo build --target wasm32-wasip2 --release
//! ```

mod completions;

use wstd::http::body::IncomingBody;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Method, Request, Response, StatusCode};
use wstd::io::AsyncRead;

#[wstd::http_server]
async fn main(mut req: Request<IncomingBody>, res: Responder) -> Finished {
    let path = req.uri().path();
    let method = req.method().clone();

    match (method, path) {
        (Method::POST, "/v1/completions") | (Method::POST, "/completions") => {
            let mut body_bytes = Vec::new();
            if read_body(req.body_mut(), &mut body_bytes).await.is_err() {
                return completions::error_response(res, 400, "Failed to read request body").await;
            }
            completions::handle(body_bytes, res).await
        }

        (Method::GET, "/health") => {
            let response = Response::builder()
                .header("Content-Type", "text/plain")
                .body("ok".into_body())
                .unwrap();
            res.respond(response).await
        }

        (Method::GET, "/") => {
            let info = r#"{
  "name": "Pie RL Completions Server",
  "version": "0.1.0",
  "endpoints": {
    "POST /v1/completions": "Cumulative-token-mode completions (prompt: list[int] | string)",
    "GET /health": "Liveness for the rllm gateway router"
  }
}
"#;
            let response = Response::builder()
                .header("Content-Type", "application/json")
                .body(info.into_body())
                .unwrap();
            res.respond(response).await
        }

        _ => {
            let error = serde_json::json!({
                "error": { "type": "not_found", "message": "Endpoint not found" }
            });
            let response = Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header("Content-Type", "application/json")
                .body(error.to_string().into_body())
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
