//! Content-addressed KV-session reuse.
//!
//! Codex talks the stateless Responses API: every turn re-sends the whole
//! conversation. A naive server pays full prefill on that growing history
//! every request. Pie lets us do better: after generating a turn we save the
//! context as a named engine-side KV snapshot, where the name is a hash of
//! the *item list a follow-up request would arrive with* (everything up to
//! and including the items we just produced). The next request strips its
//! trailing `function_call_output` / new-user items, hashes the remainder,
//! and — on a hit — resumes from the snapshot, paying prefill only for the
//! stripped suffix and the new cue.
//!
//! No server-side session table is needed: the daemon gets a fresh WASM
//! instance per request, and the only cross-request state is the snapshot
//! itself, addressed purely by content. `Context::take` (open + delete)
//! keeps at most one live snapshot per conversation branch. A hash miss is
//! always safe: it just falls back to a full-history rebuild.

use crate::types::InputItem;

/// Neutral canonical form of one conversation item — producible both from
/// incoming `InputItem`s and from the output items we generate, so the name
/// we save under equals the name the echo-back history will hash to.
pub enum CanonItem<'a> {
    Msg { role: &'a str, text: String },
    Call { call_id: &'a str, name: &'a str, args: &'a str },
    CallOutput { call_id: &'a str, text: String },
}

/// Canonicalize an input item. Returns `None` for items that don't affect
/// the replayed token stream (reasoning, references, unknown types) — they
/// must not perturb the address either.
pub fn canon(item: &InputItem) -> Option<CanonItem<'_>> {
    match item {
        InputItem::Message(m) => Some(CanonItem::Msg {
            role: m.role_str(),
            text: m.content.as_text(),
        }),
        InputItem::FunctionCall(fc) => Some(CanonItem::Call {
            call_id: &fc.call_id,
            name: &fc.name,
            args: &fc.arguments,
        }),
        InputItem::FunctionCallOutput(fco) => Some(CanonItem::CallOutput {
            call_id: &fco.call_id,
            text: fco.output_text(),
        }),
        InputItem::Reasoning { .. } | InputItem::ItemReference { .. } | InputItem::Other => None,
    }
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

fn push_canon(item: &CanonItem<'_>, out: &mut Vec<u8>) {
    match item {
        CanonItem::Msg { role, text } => {
            out.extend_from_slice(b"\x01msg\x00");
            out.extend_from_slice(role.as_bytes());
            out.push(0);
            out.extend_from_slice(text.as_bytes());
        }
        CanonItem::Call { call_id, name, args } => {
            out.extend_from_slice(b"\x02fc\x00");
            out.extend_from_slice(call_id.as_bytes());
            out.push(0);
            out.extend_from_slice(name.as_bytes());
            out.push(0);
            out.extend_from_slice(args.as_bytes());
        }
        CanonItem::CallOutput { call_id, text } => {
            out.extend_from_slice(b"\x03fco\x00");
            out.extend_from_slice(call_id.as_bytes());
            out.push(0);
            out.extend_from_slice(text.as_bytes());
        }
    }
    out.push(0xff);
}

/// Hash of (instructions, tool schemas, canonical items) → snapshot name.
///
/// Instructions and tools are part of the address because they are rendered
/// into the system/tool prefix of the token stream: a change to either
/// invalidates every saved prefix.
pub fn snapshot_name<'a>(
    scope: &str,
    instructions: Option<&str>,
    tool_schemas: &[String],
    items: impl IntoIterator<Item = &'a CanonItem<'a>>,
) -> String {
    let mut buf = Vec::with_capacity(1024);
    if let Some(ins) = instructions {
        buf.extend_from_slice(ins.as_bytes());
    }
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
    // Scoped under the Codex session key so a whole conversation's
    // snapshots share a `codex/{scope}/` prefix (cleanup-friendly).
    format!("codex/{scope}/{:016x}{:016x}", h, h2)
}

/// Split an incoming item list at the resume point: everything up to and
/// including the last assistant-produced item (assistant `message` or
/// `function_call`) can have been snapshotted by the previous request; the
/// suffix after it is what Codex added since (tool outputs, new user turn).
/// Returns `None` when there is no assistant-produced item at all (first
/// turn — nothing could have been snapshotted).
pub fn split_resume_point(items: &[InputItem]) -> Option<usize> {
    let last_assistant = items.iter().rposition(|it| match it {
        InputItem::FunctionCall(_) => true,
        InputItem::Message(m) => m.is_assistant(),
        _ => false,
    })?;
    Some(last_assistant + 1)
}
