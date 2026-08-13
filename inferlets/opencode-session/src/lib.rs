//! `opencode-session` — the Strategy B serving inferlet.
//!
//! One long-lived process per opencode session, launched once and then fed turn
//! envelopes over the sticky WebSocket (`GET /v1/ws` → `launch_process` →
//! `signal_process` per turn). It holds the conversation's KV working set — and,
//! on a hybrid model, the folded recurrent state that belongs to the same prefix
//! — in process memory for its whole life, so a turn costs prefill for what
//! CHANGED rather than for the whole history.
//!
//! ## Why this exists
//!
//! Strategy A (`inferlets/chat-completions`, frozen) serves stock opencode from
//! an OpenAI-compatible endpoint: one inferlet per request, KV discarded when
//! the request ends. It works, it is live, and the measurement that came out of
//! it is the reason for this crate: on a two-tool-call agentic task, ~3 turns
//! re-prefilling ~24k tokens of history took ~57 s of an 81.2 s task, while
//! generating the ~200 tokens of actual output took ~2 s. **~70% of an agentic
//! task is re-prefill of a history that changed by a few hundred tokens.**
//!
//! Three Strategy-A blockers dissolve here, and it is worth being precise about
//! *why*, because two of them were nearly re-solved the hard way:
//!
//! - Metal refuses two device-geometry programs in one batch, so N concurrent
//!   HTTP requests degrade every turn. A session inferlet is ONE program.
//! - A fold cannot be published across processes (`rs-working-set` has `fork`
//!   and no `update-index`/`from-index`). The fold never leaves this process.
//!   **This removes the persistence half and NOT the pipeline-binding half** —
//!   see `engine`'s docs on why the seal must be the last fire on the
//!   generation pipeline. A long-lived process does not fix that by itself.
//! - The control layer's FCFS policy terminates the most recently created
//!   inferlets, which under Strategy A means the turn the user is waiting on.
//!   N turns are N rows in one inferlet here, not N tenants.
//!
//! ## What is deliberately NOT here yet
//!
//! The wire is still stateless OpenAI full-history, with continuity recovered
//! server-side by content address (`handler`). The delta wire of
//! `docs/opencode-integration.md` §2 — an AI SDK provider shadowing server
//! state and sending only the suffix — buys B-2 (in-place context editing) and
//! B-3 (forking for subagents), and nothing else that hashing cannot already
//! express. It is second because its failure mode is worse: a hash mismatch is
//! a clean miss and a full rebuild, while a shadow-diff mismatch is silently
//! the wrong context.
//!
//! Module map: [`wire`] — the session envelope; [`handler`] — resume, retention
//! and turn orchestration; [`engine`] — resumable PTIR prefill/decode/seal;
//! [`turn`] — the per-token state machine. All wire JSON and all engine-free
//! logic come from `pie-openai-serving`, shared with Strategy A so the A/B
//! compares servers rather than renderers.

mod engine;
mod handler;
mod turn;
mod wire;

/// The session loop.
///
/// `receive()` yields one turn envelope at a time and the daemon serves them
/// sequentially. That is not a simplification to be undone later: pie's own
/// model is a single-threaded, event-driven runtime per inferlet, and these
/// turns share ONE working set — two turns extending the same KV prefix
/// concurrently would interleave their writes. opencode's own tool loop is
/// sequential per session anyway; parallelism across sessions is parallelism
/// across processes.
#[inferlet::main]
async fn main(_input: String) -> inferlet::Result<String> {
    let mut daemon = handler::Daemon::new();
    let mut served = 0u64;
    while let Some(msg) = inferlet::session::receive().await {
        daemon.handle(&msg).await;
        served += 1;
    }
    // Instrumentation only — every response already went out over the envelope.
    Ok(format!("served {served} turns"))
}
