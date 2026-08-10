//! Content-addressed KV-session reuse for the chat-completions surface.
//!
//! Port of `codex-responses/src/session.rs` to the chat message model.
//! qwen-code talks the stateless chat API: every turn re-sends the whole
//! conversation. After generating a turn we save the context as a named
//! engine-side KV snapshot, where the name is a hash of the *message list a
//! follow-up request would arrive with* (everything up to and including the
//! assistant turn we just produced). The next request strips its trailing
//! tool-result / new-user messages, hashes the remainder, and — on a hit —
//! resumes from the snapshot, paying prefill only for the stripped suffix
//! and the new cue.
//!
//! qwen-code sends no session key, so the namespace is purely
//! content-addressed under `qwenchat/`. No server-side session table
//! exists: the daemon gets a fresh WASM instance per request, and the only
//! cross-request state is the snapshot itself. A hash miss is always safe —
//! it just falls back to a full-history rebuild.

use crate::types::ChatMessage;

/// Neutral canonical form of one conversation unit — producible both from
/// incoming messages and from the output we generate, so the name we save
/// under equals the name the echo-back history will hash to.
///
/// `reasoning_content` is deliberately absent: it is excluded from the
/// rendered token stream (no-think channel, H17), so it must not perturb
/// the address either.
pub enum CanonItem {
    Msg { role: String, text: String },
    Call { id: String, name: String, args: String },
    CallOutput { id: String, text: String },
}

/// Canonicalize a message list. One assistant message with tool calls
/// expands to `Msg` (content, if non-empty) + one `Call` per tool call —
/// mirroring how the save path canonicalizes its own output (streamed
/// visible text + decoded calls).
pub fn canon_messages(messages: &[ChatMessage]) -> Vec<CanonItem> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        match m.role.as_str() {
            "tool" => out.push(CanonItem::CallOutput {
                id: m.tool_call_id.clone().unwrap_or_default(),
                text: m.text(),
            }),
            "assistant" => {
                let text = m.text();
                if !text.is_empty() {
                    out.push(CanonItem::Msg { role: "assistant".to_string(), text });
                }
                for c in m.calls() {
                    out.push(CanonItem::Call {
                        id: c.id.clone(),
                        name: c.function.name.clone(),
                        args: c.function.arguments.clone(),
                    });
                }
            }
            role => out.push(CanonItem::Msg { role: role.to_string(), text: m.text() }),
        }
    }
    out
}

/// FNV-1a 64-bit — no crypto needed, just a stable content address that two
/// different conversations are vanishingly unlikely to share (two seeds,
/// 128 address bits). A collision is benign anyway: the worst case of a
/// stale snapshot is degraded output for one conversation, and a miss is a
/// full rebuild.
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;

fn push_canon(item: &CanonItem, out: &mut Vec<u8>) {
    match item {
        CanonItem::Msg { role, text } => {
            out.extend_from_slice(b"\x01msg\x00");
            out.extend_from_slice(role.as_bytes());
            out.push(0);
            out.extend_from_slice(text.as_bytes());
        }
        CanonItem::Call { id, name, args } => {
            out.extend_from_slice(b"\x02fc\x00");
            out.extend_from_slice(id.as_bytes());
            out.push(0);
            out.extend_from_slice(name.as_bytes());
            out.push(0);
            out.extend_from_slice(args.as_bytes());
        }
        CanonItem::CallOutput { id, text } => {
            out.extend_from_slice(b"\x03fco\x00");
            out.extend_from_slice(id.as_bytes());
            out.push(0);
            out.extend_from_slice(text.as_bytes());
        }
    }
    out.push(0xff);
}

/// Hash of (tool schemas, no-think flag, canonical items) → snapshot name.
///
/// Tool schemas and the thinking channel are part of the address because
/// they are rendered into the token stream: a change to either invalidates
/// every saved prefix (H14 tool-list growth thus lands as a clean miss).
/// The system message travels inside `items` (it is `messages[0]` on this
/// API, unlike Responses' out-of-band `instructions`).
pub fn snapshot_name<'a>(
    tool_schemas: &[String],
    no_think: bool,
    items: impl IntoIterator<Item = &'a CanonItem>,
) -> String {
    let mut buf = Vec::with_capacity(1024);
    buf.push(no_think as u8);
    buf.push(0xfe);
    for schema in tool_schemas {
        buf.extend_from_slice(schema.as_bytes());
        buf.push(0xfe);
    }
    for item in items {
        push_canon(item, &mut buf);
    }
    let h = fnv1a(&buf, FNV_OFFSET);
    let h2 = fnv1a(&buf, h ^ 0x9e3779b97f4a7c15);
    format!("qwenchat/{:016x}{:016x}", h, h2)
}

/// Split an incoming message list at the resume point: everything up to and
/// including the last assistant message can have been snapshotted by the
/// previous request; the suffix after it is what qwen-code added since
/// (tool results, new user turn). Returns `None` when there is no assistant
/// message at all (first turn — nothing could have been snapshotted).
pub fn split_resume_point(messages: &[ChatMessage]) -> Option<usize> {
    let last = messages.iter().rposition(|m| m.role == "assistant")?;
    Some(last + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(json: serde_json::Value) -> Vec<ChatMessage> {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn address_is_stable_and_reasoning_free() {
        let a = msgs(serde_json::json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "reasoning_content": "thinking...",
             "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}]}
        ]));
        let b = msgs(serde_json::json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant",
             "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}]}
        ]));
        let schemas = vec!["{\"name\":\"f\"}".to_string()];
        let name_a = snapshot_name(&schemas, true, canon_messages(&a).iter());
        let name_b = snapshot_name(&schemas, true, canon_messages(&b).iter());
        assert_eq!(name_a, name_b);
        assert!(name_a.starts_with("qwenchat/"));

        // Any address-relevant change moves the name.
        assert_ne!(name_a, snapshot_name(&schemas, false, canon_messages(&a).iter()));
        assert_ne!(name_a, snapshot_name(&[], true, canon_messages(&a).iter()));
    }

    #[test]
    fn save_address_matches_echo_back() {
        // What the generating request computes: incoming messages + its own
        // output, canonicalized.
        let incoming = msgs(serde_json::json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"}
        ]));
        let mut canons = canon_messages(&incoming);
        canons.push(CanonItem::Msg { role: "assistant".into(), text: "ok, checking".into() });
        canons.push(CanonItem::Call {
            id: "call_x_0".into(), name: "read_file".into(), args: "{\"p\":1}".into(),
        });
        let saved = snapshot_name(&[], true, canons.iter());

        // What the follow-up request hashes: the echo-back history, trailing
        // tool result stripped at the resume point.
        let next = msgs(serde_json::json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok, checking", "reasoning_content": "…",
             "tool_calls": [{"id": "call_x_0",
                             "function": {"name": "read_file", "arguments": "{\"p\":1}"}}]},
            {"role": "tool", "tool_call_id": "call_x_0", "content": "body"}
        ]));
        let split = split_resume_point(&next).unwrap();
        assert_eq!(split, 3);
        let resumed = snapshot_name(&[], true, canon_messages(&next[..split]).iter());
        assert_eq!(saved, resumed);
    }

    #[test]
    fn first_turn_has_no_resume_point() {
        let m = msgs(serde_json::json!([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"}
        ]));
        assert!(split_resume_point(&m).is_none());
    }
}
