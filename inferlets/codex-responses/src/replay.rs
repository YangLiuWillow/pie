//! History rendering: turn Responses input items into the model's canonical
//! chat-template token stream, byte-identical to what the template would
//! produce.
//!
//! Ported from `openhands-completion`'s `replay_history`, adapted to the
//! Responses item model: assistant tool calls arrive as separate
//! `function_call` items (optionally preceded by an assistant `message`
//! item), and tool results as `function_call_output` items. Qwen's template
//! merges assistant content + tool calls into one turn and consecutive tool
//! results into one reply wrapper, so items are re-grouped before replay —
//! see `integrations/openhands/docs/TOOL_CALL_HISTORY_REPLAY_DESIGN.md`.
//!
//! Renders into a `Vec<u32>` rather than a `Context` so the caller can
//! prefill in bounded chunks — a single multi-thousand-token forward pass
//! outlives the engine's per-forward timeout on slow (CPU) drivers and
//! wedges the whole driver queue.

use crate::types::InputItem;
use inferlet::{chat, model::Model, tools, Result};

/// Render the full conversation: tool/system preamble first, then the
/// grouped item turns.
pub fn render_full(
    model: &Model,
    instructions: Option<&str>,
    items: &[InputItem],
    tool_schemas: &[String],
) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    if !tool_schemas.is_empty() {
        // The chat template folds the system message and the tool schemas
        // into a single system turn.
        out.extend(tools::equip_after_system_prefix(
            model,
            instructions,
            tool_schemas,
        )?);
    } else if let Some(ins) = instructions {
        out.extend(chat::system(model, ins));
    }
    render_items(model, items, &mut out)?;
    Ok(out)
}

/// Render a run of items (also used for the suffix after resuming from a
/// KV snapshot).
pub fn render_items(model: &Model, items: &[InputItem], out: &mut Vec<u32>) -> Result<()> {
    let mut i = 0;
    while i < items.len() {
        match &items[i] {
            InputItem::Message(m) if m.is_assistant() => {
                let content = m.content.as_text();
                // Group with any immediately-following function_call items
                // (skipping reasoning items in between) — the template
                // renders them as one assistant turn.
                let (calls, next) = collect_calls(items, i + 1);
                if calls.is_empty() {
                    out.extend(chat::assistant(model, &content));
                } else {
                    out.extend(tools::assistant_with_tool_calls_prefix(
                        model,
                        Some(content.as_str()).filter(|s| !s.is_empty()),
                        &calls,
                    ));
                }
                i = next.max(i + 1);
            }
            InputItem::Message(m) => {
                match m.role_str() {
                    "system" | "developer" => out.extend(chat::system(model, &m.content.as_text())),
                    // `/no_think` on *every* user turn, not just the last:
                    // the decoration must be position-independent so a turn
                    // replays identically once it becomes history —
                    // otherwise saved KV snapshots would diverge from the
                    // rebuilt token stream. (Qwen3 soft switch; harmless
                    // trailing text on non-thinking models.)
                    _ => out.extend(chat::user(
                        model,
                        &format!("{} /no_think", m.content.as_text().trim_end()),
                    )),
                };
                i += 1;
            }
            InputItem::FunctionCall(_) => {
                // Tool calls with no preceding assistant message.
                let (calls, next) = collect_calls(items, i);
                out.extend(tools::assistant_with_tool_calls_prefix(model, None, &calls));
                i = next;
            }
            InputItem::FunctionCallOutput(_) => {
                // Merge consecutive tool results into one replayed turn
                // (the model was fine-tuned on the merged form).
                let mut batch: Vec<(String, String)> = Vec::new();
                while i < items.len() {
                    match &items[i] {
                        InputItem::FunctionCallOutput(fco) => {
                            batch.push((String::new(), fco.output_text()));
                            i += 1;
                        }
                        InputItem::Reasoning { .. } | InputItem::Other => i += 1,
                        _ => break,
                    }
                }
                out.extend(tools::answer_batch_prefix(model, &batch));
            }
            InputItem::Reasoning { .. } | InputItem::ItemReference { .. } | InputItem::Other => {
                i += 1;
            }
        }
    }
    Ok(())
}

/// Collect consecutive `function_call` items starting at `from`, skipping
/// interleaved reasoning items. Returns the (name, arguments) pairs and the
/// index just past the run.
fn collect_calls(items: &[InputItem], from: usize) -> (Vec<(String, String)>, usize) {
    let mut calls = Vec::new();
    let mut i = from;
    while i < items.len() {
        match &items[i] {
            InputItem::FunctionCall(fc) => {
                calls.push((fc.name.clone(), fc.arguments.clone()));
                i += 1;
            }
            InputItem::Reasoning { .. } => i += 1,
            _ => break,
        }
    }
    (calls, i)
}
