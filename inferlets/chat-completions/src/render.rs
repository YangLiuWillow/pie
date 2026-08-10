//! History rendering: turn an OpenAI chat message list into the model's
//! canonical chat-template token stream, byte-identical to what the
//! template would produce.
//!
//! Ported from `openhands-completion::replay_history` (same message model),
//! but rendering into a `Vec<u32>` rather than a `Context` — like
//! `codex-responses/src/replay.rs` — so the caller can (a) prefill in
//! bounded chunks (a single multi-thousand-token forward pass outlives the
//! engine's per-forward timeout on slow drivers) and (b) render just the
//! suffix after resuming from a KV snapshot.

use crate::types::ChatMessage;
use inferlet::{Result, chat, model::Model, tools};

/// Render the full conversation: system/tool preamble first, then turns.
pub fn render_full(
    model: &Model,
    messages: &[ChatMessage],
    tool_schemas: &[String],
    no_think: bool,
) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    let mut rest = messages;

    // The chat template folds a leading system message's content into the
    // *same* system turn as the tool schemas (see `equip_after_system_prefix`)
    // rather than two consecutive system turns.
    if !tool_schemas.is_empty() {
        if messages.first().map(|m| m.role.as_str()) == Some("system") {
            let content = messages[0].text();
            out.extend(tools::equip_after_system_prefix(
                model,
                Some(content.as_str()).filter(|s| !s.is_empty()),
                tool_schemas,
            )?);
            rest = &messages[1..];
        } else {
            out.extend(tools::equip_prefix(model, tool_schemas)?);
        }
    }

    render_messages(model, rest, no_think, &mut out)?;
    Ok(out)
}

/// Render a run of messages (also used for the suffix after resuming from a
/// KV snapshot — the suffix never contains the leading system message, so
/// no equip handling is needed here).
pub fn render_messages(
    model: &Model,
    messages: &[ChatMessage],
    no_think: bool,
    out: &mut Vec<u32>,
) -> Result<()> {
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        match msg.role.as_str() {
            "system" | "developer" => {
                out.extend(chat::system(model, &msg.text()));
                i += 1;
            }
            "user" => {
                // `/no_think` on *every* user turn, not just the last: the
                // decoration must be position-independent so a turn replays
                // identically once it becomes history — otherwise saved KV
                // snapshots would diverge from the rebuilt token stream.
                // (Qwen3 soft switch; harmless trailing text elsewhere.)
                let text = msg.text();
                if no_think {
                    out.extend(chat::user(model, &format!("{} /no_think", text.trim_end())));
                } else {
                    out.extend(chat::user(model, &text));
                }
                i += 1;
            }
            "assistant" => {
                let calls = msg.calls();
                if calls.is_empty() {
                    out.extend(chat::assistant(model, &msg.text()));
                } else {
                    let pairs: Vec<(String, String)> = calls
                        .iter()
                        .map(|c| (c.function.name.clone(), c.function.arguments.clone()))
                        .collect();
                    let content = msg.text();
                    out.extend(tools::assistant_with_tool_calls_prefix(
                        model,
                        Some(content.as_str()).filter(|s| !s.is_empty()),
                        &pairs,
                    ));
                }
                i += 1;
            }
            "tool" => {
                // Merge consecutive tool results into one replayed turn —
                // the model was fine-tuned on the merged form (see
                // `answer_batch_prefix`'s doc comment).
                let mut batch: Vec<(String, String)> = Vec::new();
                while i < messages.len() && messages[i].role == "tool" {
                    batch.push((String::new(), messages[i].text()));
                    i += 1;
                }
                out.extend(tools::answer_batch_prefix(model, &batch));
            }
            other => return Err(format!("unsupported message role: {other}").into()),
        }
    }
    Ok(())
}

/// Strip the tokenizer's special-token strings out of round-tripped message
/// content. `Tokenizer::encode` maps a literal special-token string (e.g.
/// "<|im_start|>" leaked into an assistant reply) back to the real token
/// id, so replaying it would plant fake turn boundaries mid-message and
/// corrupt the chat structure (openhands traj job 18825434 degenerated into
/// token salad this way). Must run before both rendering and session
/// canonicalization so the address hashes what actually gets replayed.
pub fn sanitize_messages(messages: &mut [ChatMessage], model: &Model) {
    let (_ids, byte_seqs) = model.tokenizer().special_tokens();
    let specials: Vec<String> = byte_seqs
        .into_iter()
        .filter_map(|b| String::from_utf8(b).ok())
        .filter(|s| !s.is_empty())
        .collect();
    if specials.is_empty() {
        return;
    }
    let clean = |s: &mut String| {
        for sp in &specials {
            if s.contains(sp.as_str()) {
                *s = s.replace(sp.as_str(), "");
            }
        }
    };
    for m in messages.iter_mut() {
        if let Some(c) = m.content.as_mut() {
            match c {
                crate::types::MessageContent::Text(s) => clean(s),
                crate::types::MessageContent::Parts(parts) => {
                    for p in parts.iter_mut() {
                        clean(&mut p.text);
                    }
                }
            }
        }
        if let Some(calls) = m.tool_calls.as_mut() {
            for c in calls.iter_mut() {
                clean(&mut c.function.name);
                clean(&mut c.function.arguments);
            }
        }
    }
}
