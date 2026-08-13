//! Content-addressed KV-session canon for the chat-completions surface —
//! pure canonicalization/hashing/splitting; the engine calls (dev:
//! `working-set update-index` / `from-index`) live in the inferlet.
//!
//! Port of the validated `chat-completions/src/session.rs` (qwen-code
//! branch). Stateless-chat clients (opencode, qwen-code) re-send the whole
//! conversation every turn. After generating a turn the inferlet indexes
//! its working set under a name that is a hash of the *message list a
//! follow-up request would arrive with* (everything up to and including the
//! assistant turn just produced). The next request strips its trailing
//! tool-result / new-user messages, hashes the remainder, and — on a hit —
//! resumes from the indexed set, paying prefill only for the stripped
//! suffix and the new cue. A miss is always safe: full-history rebuild.
//!
//! RESPONSE/SAVE UNIFICATION (the invariant this crate exists to hold in
//! one place): the content string sent back to the client and the string
//! canonicalized into the save address MUST be the same string. The client
//! echoes response content back verbatim next turn, so any divergence
//! breaks every subsequent KV resume (observed live on qwen-code: an
//! empty-visible turn answered `" "` but saved no assistant message —
//! permanent miss).

use crate::types::ChatMessage;

/// Neutral canonical form of one conversation unit — producible both from
/// incoming messages and from the output we generate, so the name we save
/// under equals the name the echo-back history will hash to.
///
/// `reasoning_content` is deliberately absent: it is excluded from the
/// rendered token stream (no-think channel, H17), so it must not perturb
/// the address either.
///
/// The system message travels as an ordinary `Msg { role: "system" }` item
/// (it is `messages[0]` on this API, unlike Responses' out-of-band
/// `instructions`); the name-sorted tool schemas travel in the address
/// *header* (see [`snapshot_address`]) because they are rendered into the
/// token stream ahead of everything else.
pub enum CanonItem {
    Msg { role: String, text: String },
    Call { id: String, name: String, args: String },
    CallOutput { id: String, text: String },
}

/// Canonicalize a message list. One assistant message with tool calls
/// expands to `Msg` (content, if non-empty — opencode's `content: ""`
/// therefore contributes nothing, matching `text_opt`) + one `Call` per
/// tool call — mirroring how the save path canonicalizes its own output
/// (streamed visible text + decoded calls).
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

/// Hash of (tool schemas, no-think flag, canonical items) → 32-hex-char
/// address. The caller prefixes its namespace (e.g. `ocsession/`) before
/// handing the name to `working-set update-index` / `from-index`.
///
/// `tool_schemas` must be the **name-sorted** envelopes from
/// [`crate::types::tool_schema_envelopes`] — schemas and the thinking
/// channel are part of the address because they are rendered into the token
/// stream: a change to either invalidates every saved prefix (H14 tool-list
/// growth thus lands as a clean miss).
pub fn snapshot_address<'a>(
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
    format!("{h:016x}{h2:016x}")
}

/// Split an incoming message list at the resume point: everything up to and
/// including the last assistant message can have been snapshotted by the
/// previous request; the suffix after it is what the client added since
/// (tool results, new user turn). Returns `None` when there is no assistant
/// message at all (first turn — nothing could have been snapshotted).
pub fn split_resume_point(messages: &[ChatMessage]) -> Option<usize> {
    let last = messages.iter().rposition(|m| m.role == "assistant")?;
    Some(last + 1)
}

/// Split *before* the trailing assistant turn: everything strictly earlier can
/// have been retained, and the suffix begins with the assistant message the
/// server itself produced last turn.
///
/// ## Why this exists next to [`split_resume_point`], rather than replacing it
///
/// The two encode different answers to "what is safe to keep in KV".
///
/// `split_resume_point` (+1) keeps the assistant turn in the retained prefix.
/// That is the maximum-reuse split, and it is what a snapshot-based design
/// wants — but it means the retained KV contains the tokens the model
/// *generated*, sitting behind a generation cue. On Qwen those two things
/// differ from the replayed form of the same turn: `cue_no_think()` emits
/// `<|im_start|>assistant\n<think>\n\n</think>\n\n`, while replaying that turn
/// as history emits `<|im_start|>assistant\n` and strips thinking. The retained
/// prefix is then ~4 tokens longer than any re-render of the same conversation
/// will ever produce, every turn, cumulatively — measured as a resumed prompt
/// of 64 tokens against a cold rebuild of 60, with different answers at
/// temperature 0.
///
/// This split keeps only rendered history. Nothing generated is ever retained,
/// so the retained prefix is `render(messages[..split])` by construction and
/// concatenates with `render(messages[split..]) + cue` to exactly the full
/// render. The assistant turn is re-prefilled each turn — tens to a few hundred
/// tokens against a history of tens of thousands.
///
/// It also dissolves the response/save unification hazard: the address no
/// longer hashes anything the server produced, so a trim or a fallback in the
/// response path can no longer silently break every subsequent resume.
///
/// Returns `None` when there is no assistant message (first turn).
pub fn split_retain_point(messages: &[ChatMessage]) -> Option<usize> {
    messages.iter().rposition(|m| m.role == "assistant")
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
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}]}
        ]));
        let schemas = vec!["{\"name\":\"f\"}".to_string()];
        let name_a = snapshot_address(&schemas, true, canon_messages(&a).iter());
        let name_b = snapshot_address(&schemas, true, canon_messages(&b).iter());
        assert_eq!(name_a, name_b);
        assert_eq!(name_a.len(), 32);

        // Any address-relevant change moves the name.
        assert_ne!(name_a, snapshot_address(&schemas, false, canon_messages(&a).iter()));
        assert_ne!(name_a, snapshot_address(&[], true, canon_messages(&a).iter()));
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
        let saved = snapshot_address(&[], true, canons.iter());

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
        let resumed = snapshot_address(&[], true, canon_messages(&next[..split]).iter());
        assert_eq!(saved, resumed);
    }

    #[test]
    fn retain_point_excludes_the_assistant_turn_and_round_trips() {
        // Turn 1's request. Retained under the canon of exactly these messages
        // — nothing the server produced enters the address.
        let sent = msgs(serde_json::json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"}
        ]));
        let saved = snapshot_address(&[], true, canon_messages(&sent).iter());

        // Turn 2: the client echoes our assistant turn back and appends.
        let next = msgs(serde_json::json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok, checking",
             "tool_calls": [{"id": "call_x_0",
                             "function": {"name": "read_file", "arguments": "{\"p\":1}"}}]},
            {"role": "tool", "tool_call_id": "call_x_0", "content": "body"}
        ]));
        let split = split_retain_point(&next).unwrap();
        // Before the assistant turn, where `split_resume_point` lands after it.
        assert_eq!(split, 2);
        assert_eq!(split_resume_point(&next).unwrap(), 3);

        let resumed = snapshot_address(&[], true, canon_messages(&next[..split]).iter());
        assert_eq!(saved, resumed);

        // And the suffix carries the assistant turn, so it gets re-rendered
        // rather than resumed out of KV.
        assert_eq!(next[split].role, "assistant");
    }

    #[test]
    fn first_turn_has_no_resume_point() {
        let m = msgs(serde_json::json!([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"}
        ]));
        assert!(split_resume_point(&m).is_none());
        assert!(split_retain_point(&m).is_none());
    }
}
