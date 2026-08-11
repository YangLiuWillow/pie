//! History rendering plan: turn an OpenAI chat message list into an
//! engine-free sequence of template operations.
//!
//! Rework of the validated `chat-completions/src/render.rs`: where the old
//! code called `inferlet::{chat,tools}` directly to produce token ids, this
//! produces a [`RenderOp`] sequence that the inferlet maps 1:1 onto WIT
//! calls (`tools.equip_after_system` / `chat.user` /
//! `tools.assistant_with_tool_calls` / `tools.answer_batch` / `chat.cue`) —
//! or onto anything else (the parity harness renders it via HF
//! `apply_chat_template`). The ordering/batching rules are ported verbatim;
//! what is deliberately NOT here:
//!
//! - `/no_think` user-turn decoration (model-specific; the inferlet applies
//!   it to every `User` op when `ChatCompletionRequest::no_think()` — every
//!   turn, not just the last, so a turn replays identically once it becomes
//!   history and saved KV snapshots don't diverge from the rebuilt stream);
//! - special-token sanitization (`sanitize_messages` needs the tokenizer's
//!   special-token table; the inferlet must run it before BOTH rendering
//!   and canonicalization so the address hashes what actually replays);
//! - prefill chunking (an engine concern).

use crate::types::{ChatCompletionRequest, ChatMessage, tool_schema_envelopes};

/// One template operation. `String` payloads are exactly what goes to the
/// corresponding WIT call; tool schemas are the name-sorted envelopes from
/// [`tool_schema_envelopes`] — the SAME strings the snapshot address hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderOp {
    /// The chat template folds a leading system message's content into the
    /// *same* system turn as the tool schemas (rather than two consecutive
    /// system turns). Emitted first whenever there is a leading system
    /// message OR a non-empty tool list; `tools` is empty for the no-tools
    /// case (opencode's title side-call), which renders as a plain system
    /// turn.
    EquipAfterSystem { system: Option<String>, tools: Vec<String> },
    User(String),
    Assistant(String),
    /// `content` is `None` when the wire carried `""`/null/absent —
    /// opencode replays tool-call turns with `content: ""`.
    AssistantWithToolCalls { content: Option<String>, calls: Vec<(String, String)> },
    /// Consecutive `role:"tool"` results merged into one replayed turn —
    /// the model was fine-tuned on the merged form. Pairs are
    /// `(tool_name, result_text)`; the name is recovered from the
    /// preceding assistant turn's `tool_call_id → name` mapping (empty
    /// string for an orphan id — matches the old branch, which never
    /// rendered names into results at all).
    AnswerBatch(Vec<(String, String)>),
    /// Generation cue (assistant header) — always last.
    Cue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderError {
    UnsupportedRole(String),
    /// A system/developer message anywhere but position 0: the fold into
    /// `EquipAfterSystem` only exists at the head, and no coding client
    /// sends mid-conversation system turns.
    MisplacedSystem,
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::UnsupportedRole(r) => write!(f, "unsupported message role: {r}"),
            RenderError::MisplacedSystem => {
                write!(f, "system message only supported as the first message")
            }
        }
    }
}

impl std::error::Error for RenderError {}

/// Plan the full conversation render for a request.
pub fn plan_render(req: &ChatCompletionRequest) -> Result<Vec<RenderOp>, RenderError> {
    plan_render_messages(&req.messages, tool_schema_envelopes(&req.tools))
}

/// Plan from an explicit message slice + pre-built schema envelopes (the
/// resume path renders only the suffix after the split point — pass the
/// suffix here with empty `tool_schemas`; a suffix never contains the
/// leading system message, so no equip op is produced for it).
pub fn plan_render_messages(
    messages: &[ChatMessage],
    tool_schemas: Vec<String>,
) -> Result<Vec<RenderOp>, RenderError> {
    let mut ops = Vec::with_capacity(messages.len() + 2);
    let mut rest = messages;

    let leading_system = matches!(
        messages.first().map(|m| m.role.as_str()),
        Some("system") | Some("developer")
    );
    if leading_system || !tool_schemas.is_empty() {
        let system = if leading_system {
            let s = messages[0].text_opt();
            rest = &messages[1..];
            s
        } else {
            None
        };
        ops.push(RenderOp::EquipAfterSystem { system, tools: tool_schemas });
    }

    // tool_call_id → tool name, accumulated from assistant turns as we walk
    // so each tool result resolves against the calls that precede it.
    let mut call_names: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    let mut i = 0;
    while i < rest.len() {
        let msg = &rest[i];
        match msg.role.as_str() {
            "system" | "developer" => return Err(RenderError::MisplacedSystem),
            "user" => {
                ops.push(RenderOp::User(msg.text()));
                i += 1;
            }
            "assistant" => {
                let calls = msg.calls();
                if calls.is_empty() {
                    ops.push(RenderOp::Assistant(msg.text()));
                } else {
                    for c in calls {
                        call_names.insert(c.id.clone(), c.function.name.clone());
                    }
                    let pairs: Vec<(String, String)> = calls
                        .iter()
                        .map(|c| (c.function.name.clone(), c.function.arguments.clone()))
                        .collect();
                    ops.push(RenderOp::AssistantWithToolCalls {
                        content: msg.text_opt(),
                        calls: pairs,
                    });
                }
                i += 1;
            }
            "tool" => {
                let mut batch: Vec<(String, String)> = Vec::new();
                while i < rest.len() && rest[i].role == "tool" {
                    let name = rest[i]
                        .tool_call_id
                        .as_ref()
                        .and_then(|id| call_names.get(id))
                        .cloned()
                        .unwrap_or_default();
                    // Tool result content may be a string (opencode) or a
                    // parts array (qwen-code) — `text()` normalizes both.
                    batch.push((name, rest[i].text()));
                    i += 1;
                }
                ops.push(RenderOp::AnswerBatch(batch));
            }
            other => return Err(RenderError::UnsupportedRole(other.to_string())),
        }
    }

    ops.push(RenderOp::Cue);
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(json: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn tool_batch_recovers_names_and_merges_consecutive_results() {
        let r = req(serde_json::json!({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "",
                 "tool_calls": [
                    {"id": "c1", "function": {"name": "read", "arguments": "{\"a\":1}"}},
                    {"id": "c2", "function": {"name": "grep", "arguments": "{\"b\":2}"}}]},
                {"role": "tool", "tool_call_id": "c2", "content": "out2"},
                {"role": "tool", "tool_call_id": "c1",
                 "content": [{"type": "text", "text": "out1"}]}
            ],
            "tools": [{"function": {"name": "read", "description": "", "parameters": {}}},
                      {"function": {"name": "grep", "description": "", "parameters": {}}}]
        }));
        let ops = plan_render(&r).unwrap();
        assert_eq!(ops.len(), 5);
        match &ops[0] {
            RenderOp::EquipAfterSystem { system, tools } => {
                assert_eq!(system.as_deref(), Some("sys"));
                assert_eq!(tools.len(), 2);
                // Name-sorted: grep before read.
                assert!(tools[0].contains("\"grep\""));
            }
            other => panic!("expected EquipAfterSystem, got {other:?}"),
        }
        assert_eq!(ops[1], RenderOp::User("go".into()));
        assert_eq!(
            ops[2],
            RenderOp::AssistantWithToolCalls {
                content: None, // empty-string content normalized away
                calls: vec![("read".into(), "{\"a\":1}".into()),
                            ("grep".into(), "{\"b\":2}".into())],
            }
        );
        // Two consecutive tool results merged into ONE batch, id→name
        // resolved, parts-array content flattened.
        assert_eq!(
            ops[3],
            RenderOp::AnswerBatch(vec![("grep".into(), "out2".into()),
                                       ("read".into(), "out1".into())])
        );
        assert_eq!(ops[4], RenderOp::Cue);
    }

    #[test]
    fn no_tools_no_system_yields_no_equip() {
        let r = req(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(
            plan_render(&r).unwrap(),
            vec![RenderOp::User("hi".into()), RenderOp::Cue]
        );
    }

    #[test]
    fn title_call_shape_system_without_tools() {
        // opencode's title side-call: system + 2 users, no tools key.
        let r = req(serde_json::json!({
            "messages": [
                {"role": "system", "content": "title prompt"},
                {"role": "user", "content": "Generate a title for this conversation:\n"},
                {"role": "user", "content": "\"say hello\""}
            ]
        }));
        let ops = plan_render(&r).unwrap();
        assert_eq!(
            ops[0],
            RenderOp::EquipAfterSystem { system: Some("title prompt".into()), tools: vec![] }
        );
        assert_eq!(ops.len(), 4);
    }

    #[test]
    fn reasoning_content_is_dropped() {
        let r = req(serde_json::json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer",
                 "reasoning_content": "secret chain of thought"},
                {"role": "user", "content": "next"}
            ]
        }));
        let ops = plan_render(&r).unwrap();
        assert_eq!(ops[1], RenderOp::Assistant("answer".into()));
    }

    #[test]
    fn misplaced_system_and_unknown_role_error() {
        let r = req(serde_json::json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "late"}
            ]
        }));
        assert_eq!(plan_render(&r), Err(RenderError::MisplacedSystem));

        let r = req(serde_json::json!({
            "messages": [{"role": "robot", "content": "?"}]
        }));
        assert_eq!(
            plan_render(&r),
            Err(RenderError::UnsupportedRole("robot".into()))
        );
    }
}
