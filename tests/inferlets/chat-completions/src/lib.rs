//! chat-completions — OpenAI chat-completions daemon for qwen-code, ported
//! to the rewritten engine (`docs/qwen-code-dev-port.md`).
//!
//! Transport: the shim (`integrations/qwen-code/shim.py`) owns the HTTP/SSE
//! surface and forwards each request over the gateway session channel as a
//! JSON envelope `{"req_id", "body", "now"}`; this inferlet's `run` is a
//! long-lived receive/send loop, one request at a time, streaming
//! chat.completion.chunk events back as `{"req_id", "event", "data"}`
//! envelopes. See `chunk.rs` for the event vocabulary.
//!
//! The pure modules (`types`, `session`, `chunk`, `salvage`, `filter`,
//! `render_text`) compile and unit-test natively; everything touching the
//! WIT ABI is wasm-only.

pub mod chunk;
pub mod filter;
pub mod render_text;
pub mod salvage;
pub mod session;
pub mod types;

#[cfg(target_arch = "wasm32")]
pub mod generation;
#[cfg(target_arch = "wasm32")]
pub mod handler;
#[cfg(target_arch = "wasm32")]
pub mod render;

#[cfg(target_arch = "wasm32")]
use inferlet::Result;

#[cfg(target_arch = "wasm32")]
#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let mut daemon = handler::Daemon::new();
    let mut served = 0u64;
    while let Some(msg) = inferlet::session::receive().await {
        daemon.handle(&msg).await;
        served += 1;
    }
    Ok(format!("served {served} requests"))
}
