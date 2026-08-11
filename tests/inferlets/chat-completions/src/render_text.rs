//! Pure string builders for the ChatML render — the exact byte formats of
//! the pre-rewrite engine's `QwenInstruct` (old
//! `runtime/src/model/instruct/qwen3.rs`, the normative reference). The
//! rewrite's engine-side template diverges from these in three places
//! (separate tool system turn, per-call tool-response turns, no
//! assistant-with-tool-calls form), so the render is assembled inferlet-side
//! from these strings + `model::encode`; see `render.rs` for the token
//! assembly. Keeping the strings pure keeps them natively testable.

/// Serialize a JSON value the way transformers' chat templating does
/// (`tojson`: `json.dumps(x, ensure_ascii=False)` — `", "`/`": "`
/// separators, insertion-order keys, no sorting). Replayed tool schemas
/// and tool-call arguments must round-trip through this form because that
/// is byte-for-byte what a vLLM-served model saw (C3 parity finding,
/// `docs/qwen-code-dev-port.md` §8). Requires serde_json/preserve_order.
pub fn tojson(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Object(m) => {
            let inner: Vec<String> = m
                .iter()
                .map(|(k, val)| format!("{}: {}", serde_json::to_string(k).unwrap(), tojson(val)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Array(a) => {
            let inner: Vec<String> = a.iter().map(tojson).collect();
            format!("[{}]", inner.join(", "))
        }
        leaf => serde_json::to_string(leaf).unwrap(),
    }
}

/// Reference: the Qwen Jinja template's tool preamble. Must match exactly —
/// the model was fine-tuned on this format and won't produce `<tool_call>`
/// blocks if the preamble diverges. Starts directly at `# Tools` — the
/// `\n\n` seam after a system message belongs to `merged_system_content`
/// (the old engine's leading `\n` here produced a three-newline seam, a
/// real divergence from the HF template caught by the C3 parity check).
pub fn build_tool_system_prompt(tools: &[String]) -> String {
    let mut prompt = String::from(
        "# Tools\n\n\
         You may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n\
         <tools>",
    );
    for tool in tools {
        prompt.push('\n');
        // Wrap in {"type": "function", "function": ...} if not already
        // wrapped — the Jinja template renders tools in this envelope and
        // the model was fine-tuned on it.
        if tool.contains("\"type\"") && tool.contains("\"function\"") {
            prompt.push_str(tool);
        } else {
            prompt.push_str(&format!("{{\"type\": \"function\", \"function\": {tool}}}"));
        }
    }
    prompt.push_str(
        "\n</tools>\n\n\
         For each function call, return a json object with function name and arguments \
         within <tool_call></tool_call> XML tags:\n\
         <tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
         </tool_call>",
    );
    prompt
}

/// Reference (old `equip_after_system`): when tools are present, a leading
/// system message's content and the tools block fold into ONE system turn
/// (`content` + `"\n\n"` + tools-block), never two consecutive system turns.
pub fn merged_system_content(system_content: Option<&str>, tools: &[String]) -> String {
    let tools_block = build_tool_system_prompt(tools);
    match system_content {
        Some(c) if !c.is_empty() => format!("{c}\n\n{tools_block}"),
        _ => tools_block,
    }
}

/// `/no_think` soft switch on a user turn — position-independent (applied to
/// *every* user turn) so a turn replays identically once it becomes history.
pub fn no_think_decorate(text: &str) -> String {
    format!("{} /no_think", text.trim_end())
}

/// Reference (old `strip_thinking`): drop `<think>…</think>` from an
/// assistant message on replay. If `</think>` is present, keep only the
/// content after the last one, with leading newlines stripped.
pub fn strip_thinking(msg: &str) -> &str {
    if let Some(pos) = msg.rfind("</think>") {
        msg[pos + "</think>".len()..].trim_start_matches('\n')
    } else {
        msg
    }
}

// The seam literals of the piecewise assembly (old constructor's encode()
// arguments). `render.rs` encodes each once at startup.
pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const NL: &str = "\n";
pub const TOOL_CALL_OPEN: &str = "\n<tool_call>\n{\"name\": \"";
pub const TOOL_CALL_MID: &str = "\", \"arguments\": ";
pub const TOOL_CALL_CLOSE: &str = "}\n</tool_call>";
pub const TOOL_RESPONSE_PREFIX: &str = "<tool_response>\n";
pub const TOOL_RESPONSE_SUFFIX: &str = "\n</tool_response>";
pub const TOOL_RESPONSE_OPEN: &str = "\n<tool_response>\n";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_prompt_matches_reference_shape() {
        let schema = r#"{"name": "f", "description": "d", "parameters": {"type": "object"}}"#;
        let p = build_tool_system_prompt(&[schema.to_string()]);
        assert!(p.starts_with("# Tools\n\n"));
        assert!(p.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"f\""));
        assert!(p.ends_with("</tool_call>"));

        // Pre-wrapped schemas pass through unwrapped-again.
        let wrapped = r#"{"type": "function", "function": {"name": "g"}}"#;
        let p2 = build_tool_system_prompt(&[wrapped.to_string()]);
        assert!(p2.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"g\"}}"));
    }

    #[test]
    fn merged_system_folds_into_one_turn() {
        let tools = vec!["{\"name\": \"f\"}".to_string()];
        let merged = merged_system_content(Some("You are Qwen Code."), &tools);
        assert!(merged.starts_with("You are Qwen Code.\n\n# Tools"));
        let bare = merged_system_content(None, &tools);
        assert!(bare.starts_with("# Tools"));
        assert_eq!(merged_system_content(Some(""), &tools), bare);
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"command":"echo hi","nested":{"b":1,"a":[true,null,1.5]},"s":"café"}"#,
        )
        .unwrap();
        // Spaced separators, insertion order (not sorted), raw non-ASCII.
        assert_eq!(
            tojson(&v),
            r#"{"command": "echo hi", "nested": {"b": 1, "a": [true, null, 1.5]}, "s": "café"}"#
        );
    }

    #[test]
    fn strip_thinking_keeps_post_think_tail() {
        assert_eq!(strip_thinking("<think>\nhm\n</think>\n\nanswer"), "answer");
        assert_eq!(strip_thinking("plain"), "plain");
    }

    #[test]
    fn no_think_decoration_is_trim_stable() {
        assert_eq!(no_think_decorate("hi\n"), "hi /no_think");
        assert_eq!(no_think_decorate(no_think_decorate("hi").trim_end()), "hi /no_think /no_think");
    }
}
