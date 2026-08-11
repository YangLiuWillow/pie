//! Salvage parsers: recover tool calls the native (token-level) tool
//! decoder missed. Ported from the validated qwen-code handler
//! (`openhands-integration-updated:inferlets/chat-completions/src/handler.rs`).
//!
//! Generation runs UNCONSTRAINED (grammar-constraining the whole turn
//! suppressed reasoning text and collapsed t=0 trajectories into action
//! loops — openhands-completion finding), so a model occasionally writes a
//! call in a shape the streaming decoder does not arm on. Two recoverable
//! shapes are handled here:
//!
//! - [`parse_fenced_tool_calls`] — a ``` fence whose body is
//!   `{"name": …, "arguments": {…}}` (observed on Qwen coder models at t=0
//!   with the grammar constraint off). The fence bytes stay in the content
//!   (already on the wire in streaming mode); the calls are surfaced on top.
//! - [`parse_hermes_tool_calls`] — `<tool_call>\n{json}` with the closing
//!   tag missing (observed on Qwen3.6-27B: the model stops at EOS right
//!   after the JSON, the native decoder never completes, and the block
//!   would otherwise be lost). Scan the RAW generation — the visible filter
//!   swallowed this text.
//!
//! SEAM (deliberately not here): the Qwen3-Coder XML dialect salvage
//! (`<function=…><parameter=…>` blocks, schema-typed values — the old
//! branch's `parse_coder_xml_calls`). Port it from the same handler.rs when
//! `ToolFormat::Coder` support lands; it needs the tool schemas for value
//! typing, so its signature is `(text, tool_schemas) -> calls`.

/// Extract tool calls written as fenced JSON blocks: a ``` fence (with or
/// without a language tag) whose body is an object with a string `name` and
/// an object `arguments`. Returns `(fence_byte_offset, name,
/// arguments_json)` per match, in order. (Port from openhands-completion.)
pub fn parse_fenced_tool_calls(text: &str) -> Vec<(usize, String, String)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(rel) = text[pos..].find("```") {
        let fence_at = pos + rel;
        let after = &text[fence_at + 3..];
        let Some(nl) = after.find('\n') else { break };
        let body_and_more = &after[nl + 1..];
        let Some(end) = body_and_more.find("```") else { break };
        let body = body_and_more[..end].trim();
        pos = fence_at + 3 + nl + 1 + end + 3;
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
            let name = v.get("name").and_then(|n| n.as_str());
            let args = v.get("arguments").filter(|a| a.is_object());
            if let (Some(name), Some(args)) = (name, args) {
                out.push((fence_at, name.to_string(), args.to_string()));
            }
        }
    }
    out
}

/// Salvage parser for hermes-style `<tool_call>\n{json}` blocks whose
/// closing `</tool_call>` never arrived. Closing tag optional — the same
/// leniency qwen-code's own client-side recovery parser applies. Scans raw
/// generated text, brace-balances the JSON object (string-aware), and
/// accepts `{"name": …, "arguments": {…}}`.
pub fn parse_hermes_tool_calls(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(rel) = text[pos..].find("<tool_call>") {
        let start = pos + rel + "<tool_call>".len();
        let rest = &text[start..];
        let Some(obj_rel) = rest.find('{') else { break };
        let obj = &rest[obj_rel..];
        // Balance braces outside of strings.
        let (mut depth, mut in_str, mut esc, mut end) = (0i32, false, false, None);
        for (i, c) in obj.char_indices() {
            if esc {
                esc = false;
                continue;
            }
            match c {
                '\\' if in_str => esc = true,
                '"' => in_str = !in_str,
                '{' if !in_str => depth += 1,
                '}' if !in_str => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else { break };
        pos = start + obj_rel + end;
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&obj[..end]) {
            if let Some(name) = v.get("name").and_then(|n| n.as_str()) {
                let args = v.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                out.push((name.to_string(), args.to_string()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── fenced (ported from the old handler's `tests` module) ────────────

    #[test]
    fn fenced_tool_call_is_extracted() {
        let text = "Prose first.\n\n```json\n{\"name\": \"run_shell_command\", \"arguments\": {\"command\": \"ls\"}}\n```";
        let calls = parse_fenced_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "run_shell_command");
        assert_eq!(&text[..calls[0].0], "Prose first.\n\n");
    }

    #[test]
    fn non_tool_fences_are_ignored() {
        let text = "```python\nprint('hi')\n```\n```json\n{\"foo\": 1}\n```";
        assert!(parse_fenced_tool_calls(text).is_empty());
    }

    // ─── hermes (ported from the old handler's `hermes_tests`) ────────────

    #[test]
    fn unclosed_tool_call_block_is_salvaged() {
        let text = "<tool_call>\n{\"name\": \"run_shell_command\", \"arguments\": {\"command\": \"echo \\\"a}b\\\" > x.txt\"}}";
        let calls = parse_hermes_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "run_shell_command");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["command"], "echo \"a}b\" > x.txt");
    }

    #[test]
    fn closed_block_and_multiple_calls() {
        let text = "<tool_call>\n{\"name\":\"a\",\"arguments\":{}}\n</tool_call>\n<tool_call>\n{\"name\":\"b\",\"arguments\":{\"k\":1}}";
        let calls = parse_hermes_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].0, "b");
    }

    #[test]
    fn garbage_json_is_skipped() {
        assert!(parse_hermes_tool_calls("<tool_call>\n{not json}").is_empty());
    }
}
