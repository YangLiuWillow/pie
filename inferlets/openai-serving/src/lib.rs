//! `pie-openai-serving` — pure-logic library for serving the OpenAI
//! chat-completions dialect from pie inferlets. Everything about the wire
//! format and the session canon that does not need the engine lives here,
//! so it builds and tests natively (this crate is a member of the host
//! workspace and must never grow a wasm/WIT dependency); the wasm
//! inferlets (`chat-completions`, `opencode-session`) consume it as a path
//! dependency and supply the engine half (WIT template calls, prefill,
//! generation, working-set index/from-index).
//!
//! Module map:
//! - [`types`] — request wire types (unknown-field tolerant) + name-sorted
//!   tool-schema envelopes;
//! - [`error`] — 400-vs-500 discipline: OpenAI error bodies + request
//!   parsing/validation;
//! - [`streaming`] — `chat.completion.chunk` construction and SSE framing
//!   helpers, as pure JSON builders;
//! - [`session`] — content-addressed canon: canonical items, FNV-1a-64
//!   two-seed address, resume-point splitting;
//! - [`render`] — engine-free render plan ([`render::RenderOp`]) mapping
//!   1:1 onto the WIT template surface;
//! - [`filter`] — streaming visible-text filter (think/tool-call markup
//!   suppression with chunk-straddling holdback) + special-token message
//!   sanitization;
//! - [`salvage`] — fallback parsers for tool calls the native token-level
//!   decoder missed (fenced-JSON, unclosed hermes blocks).
//!
//! Provenance: ported from the validated qwen-code implementation at
//! `openhands-integration-updated:inferlets/chat-completions/` (33/33
//! acceptance, 99.7% local KV reuse), adjusted for the opencode wire
//! hazards catalogued in `tests/inferlets/fixtures/opencode/AUDIT.md`.

pub mod digest;
pub mod error;
pub mod filter;
pub mod render;
pub mod salvage;
pub mod session;
pub mod streaming;
pub mod types;

pub use digest::{TEMPLATE_MARKER, TokenDigest, prefix_addresses};
pub use filter::{
    VisibleFilter, answer_after_reasoning, cut_leading_reasoning, sanitize_messages,
};
pub use render::{
    RenderError, RenderOp, plan_render, plan_render_messages, plan_render_suffix,
};
pub use salvage::{parse_fenced_tool_calls, parse_hermes_tool_calls};
pub use session::{
    CanonItem, canon_messages, snapshot_address, split_resume_point, split_retain_point,
};
pub use streaming::{ChunkMeta, sse_done, sse_frame, sse_ping, usage_object};
pub use types::{ChatCompletionRequest, ChatMessage, MessageContent, tool_schema_envelopes};

// ─── Fixture-driven tests (real captured wire traffic) ─────────────────────

#[cfg(test)]
mod fixture_tests {
    use super::*;
    use serde_json::Value;
    use std::path::{Path, PathBuf};

    fn opencode_wire_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/inferlets/fixtures/opencode/wire")
    }

    /// Captured request files are `{seq, method, path, headers, body}`;
    /// the OpenAI request is the `body` field.
    fn load_body(name: &str) -> Value {
        let path = opencode_wire_dir().join(name);
        let capture: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        capture["body"].clone()
    }

    fn parse(name: &str) -> ChatCompletionRequest {
        serde_json::from_value(load_body(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    #[test]
    fn every_opencode_fixture_parses() {
        let mut n = 0;
        for f in std::fs::read_dir(opencode_wire_dir()).unwrap().flatten() {
            let name = f.file_name().to_string_lossy().into_owned();
            if !name.starts_with("req-") || !name.ends_with(".json") {
                continue;
            }
            let req = parse(&name);
            assert!(!req.messages.is_empty(), "{name}: empty messages");
            // Every opencode request streams with usage on.
            assert!(req.stream, "{name}: stream flag");
            assert!(req.include_usage(), "{name}: include_usage");
            n += 1;
        }
        assert_eq!(n, 5, "expected the 5 banked opencode captures");

        // qwen-code rl_completions captures double as fixtures when the
        // worktree has them (same wire dialect); absent here is fine.
        let rl = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/inferlets/fixtures/rl_completions/wire");
        if rl.is_dir() {
            for episode in std::fs::read_dir(&rl).unwrap().flatten() {
                if !episode.path().is_dir() {
                    continue;
                }
                for f in std::fs::read_dir(episode.path()).unwrap().flatten() {
                    let name = f.file_name().to_string_lossy().into_owned();
                    if !name.starts_with("openai-") || !name.ends_with(".json") {
                        continue;
                    }
                    let capture: Value =
                        serde_json::from_str(&std::fs::read_to_string(f.path()).unwrap())
                            .unwrap();
                    let req: ChatCompletionRequest =
                        serde_json::from_value(capture["request"].clone())
                            .unwrap_or_else(|e| panic!("{name}: {e}"));
                    assert!(!req.messages.is_empty(), "{name}: empty messages");
                }
            }
        }
    }

    #[test]
    fn req_005_round_trip_essentials() {
        // The history-replay capture: system, user, assistant+tool_calls
        // (content: ""), tool result — plus the 10 default build tools.
        let req = parse("req-005.json");
        assert_eq!(req.messages.len(), 4);
        assert_eq!(req.tools.len(), 10);
        assert_eq!(req.effective_max_tokens(4096), 32000);

        let assistant = &req.messages[2];
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.calls().len(), 1);
        assert_eq!(assistant.calls()[0].id, "call_record_001");
        assert_eq!(assistant.calls()[0].function.name, "read");
        // content:"" (empty string on the wire) normalizes to None.
        assert!(assistant.content.is_some());
        assert!(assistant.text_opt().is_none());

        let tool = &req.messages[3];
        assert_eq!(tool.role, "tool");
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_record_001"));
        assert!(!tool.text().is_empty());

        // Envelopes come out name-sorted (opencode sorts on the wire too).
        let names: Vec<String> = tool_schema_envelopes(&req.tools)
            .iter()
            .map(|s| serde_json::from_str::<Value>(s).unwrap()["name"]
                .as_str()
                .unwrap()
                .to_string())
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        assert!(names.contains(&"read".to_string()));
    }

    #[test]
    fn req_005_canon_address_is_stable_and_resumable() {
        let req = parse("req-005.json");
        let schemas = tool_schema_envelopes(&req.tools);

        // Same request parsed twice → same address.
        let again = parse("req-005.json");
        let addr = |r: &ChatCompletionRequest, upto: usize| {
            snapshot_address(&schemas, false, canon_messages(&r.messages[..upto]).iter())
        };
        assert_eq!(addr(&req, 4), addr(&again, 4));

        // Resume split lands right after the assistant turn: the trailing
        // tool result is the suffix the previous request could not have
        // snapshotted.
        let split = split_resume_point(&req.messages).unwrap();
        assert_eq!(split, 3);

        // What req-004's turn would have saved (its incoming 2 messages +
        // the read call it generated) equals what req-005's prefix hashes
        // to — the save/echo-back round trip.
        let mut save_canons = canon_messages(&req.messages[..2]);
        save_canons.push(CanonItem::Call {
            id: req.messages[2].calls()[0].id.clone(),
            name: req.messages[2].calls()[0].function.name.clone(),
            args: req.messages[2].calls()[0].function.arguments.clone(),
        });
        assert_eq!(
            snapshot_address(&schemas, false, save_canons.iter()),
            addr(&req, split)
        );

        // reasoning_content must not perturb the address.
        let mut body = load_body("req-005.json");
        body["messages"][2]["reasoning_content"] = Value::String("hidden thoughts".into());
        let with_reasoning: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        assert_eq!(addr(&with_reasoning, 4), addr(&req, 4));

        // But real content changes do.
        let mut body = load_body("req-005.json");
        body["messages"][1]["content"] = Value::String("something else".into());
        let changed: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        assert_ne!(addr(&changed, 4), addr(&req, 4));
    }

    #[test]
    fn req_005_render_plan() {
        let req = parse("req-005.json");
        let ops = plan_render(&req).unwrap();
        assert_eq!(ops.len(), 5, "ops: {ops:?}");

        match &ops[0] {
            RenderOp::EquipAfterSystem { system, tools } => {
                assert!(system.as_deref().unwrap_or("").len() > 1000, "build system prompt");
                assert_eq!(tools.len(), 10);
            }
            other => panic!("op0: {other:?}"),
        }
        assert!(matches!(&ops[1], RenderOp::User(t) if !t.is_empty()));
        match &ops[2] {
            RenderOp::AssistantWithToolCalls { content, calls } => {
                assert_eq!(*content, None, "content \"\" must render as None");
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].0, "read");
            }
            other => panic!("op2: {other:?}"),
        }
        match &ops[3] {
            RenderOp::AnswerBatch(batch) => {
                assert_eq!(batch.len(), 1);
                // tool_call_id → name resolved from the assistant turn.
                assert_eq!(batch[0].0, "read");
                assert_eq!(batch[0].1, req.messages[3].text());
            }
            other => panic!("op3: {other:?}"),
        }
        assert_eq!(ops[4], RenderOp::Cue);
    }

    #[test]
    fn req_001_title_side_call_plans_without_tools() {
        // No `tools` key at all; system + two user turns.
        let req = parse("req-001.json");
        assert!(req.tools.is_empty());
        let ops = plan_render(&req).unwrap();
        assert!(matches!(&ops[0],
            RenderOp::EquipAfterSystem { system: Some(_), tools } if tools.is_empty()));
        assert_eq!(ops.last(), Some(&RenderOp::Cue));
        // First turn: nothing to resume from.
        assert!(split_resume_point(&req.messages).is_none());
    }
}
