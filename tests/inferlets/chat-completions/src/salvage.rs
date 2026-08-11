//! Salvage parsers for tool calls the native decoder missed — pure text
//! processing, shared by the streaming and non-streaming paths. All three
//! were forced by live runs (old plan §5a/§5b) and port unchanged:
//!
//! - fenced JSON blocks (Qwen coder models at t=0 with the grammar off),
//! - bare Qwen3-Coder `<function=…>` XML (30B omits the `<tool_call>`
//!   wrapper under `general`-style prompt examples),
//! - hermes `<tool_call>{json}` with the closing tag missing (27B stops at
//!   EOS right after the JSON).

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

/// Salvage parser for Qwen3-Coder's XML tool dialect when the model omits
/// the `<tool_call>` wrapper (observed on 30B with `general`-style prompt
/// examples: `<function=name>` blocks arrive bare, the native decoder never
/// arms, and the call leaks into content as text). Parses
/// `<function=NAME><parameter=key>value</parameter>…</function>` into
/// `(name, arguments_json)` pairs, typing values via the tool schemas the
/// same way vLLM's Qwen3CoderToolParser does (parity reference:
/// integrations/openhands/pie_openhands/qwen3coder_parser.py).
pub fn parse_coder_xml_calls(text: &str, tool_schemas: &[String]) -> Vec<(String, String)> {
    // name -> {param -> type} from the schema envelopes.
    let mut param_types: std::collections::HashMap<
        String,
        std::collections::HashMap<String, String>,
    > = std::collections::HashMap::new();
    for schema in tool_schemas {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(schema) else {
            continue;
        };
        let Some(name) = v.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let mut types = std::collections::HashMap::new();
        if let Some(props) = v.pointer("/parameters/properties").and_then(|p| p.as_object()) {
            for (k, spec) in props {
                if let Some(t) = spec.get("type").and_then(|t| t.as_str()) {
                    types.insert(k.clone(), t.to_string());
                }
            }
        }
        param_types.insert(name.to_string(), types);
    }

    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(rel) = text[pos..].find("<function=") {
        let fn_at = pos + rel;
        let after = &text[fn_at + "<function=".len()..];
        let Some(name_end) = after.find('>') else { break };
        let name = after[..name_end].trim().to_string();
        let body_start = fn_at + "<function=".len() + name_end + 1;
        let Some(body_len) = text[body_start..].find("</function>") else {
            break;
        };
        let body = &text[body_start..body_start + body_len];
        pos = body_start + body_len + "</function>".len();

        let types = param_types.get(&name);
        let mut args = serde_json::Map::new();
        let mut bpos = 0;
        while let Some(prel) = body[bpos..].find("<parameter=") {
            let p_at = bpos + prel;
            let pafter = &body[p_at + "<parameter=".len()..];
            let Some(key_end) = pafter.find('>') else { break };
            let key = pafter[..key_end].trim().to_string();
            let val_start = p_at + "<parameter=".len() + key_end + 1;
            let Some(val_len) = body[val_start..].find("</parameter>") else {
                break;
            };
            // The template frames values with newlines; strip exactly one
            // leading and one trailing newline (vLLM parser behavior).
            let raw = &body[val_start..val_start + val_len];
            let val = raw.strip_prefix('\n').unwrap_or(raw);
            let val = val.strip_suffix('\n').unwrap_or(val);
            bpos = val_start + val_len + "</parameter>".len();

            let typed: serde_json::Value =
                match types.and_then(|t| t.get(&key)).map(String::as_str) {
                    Some("integer") => val
                        .trim()
                        .parse::<i64>()
                        .map(Into::into)
                        .unwrap_or_else(|_| serde_json::Value::String(val.to_string())),
                    Some("number") => val
                        .trim()
                        .parse::<f64>()
                        .ok()
                        .and_then(|f| serde_json::Number::from_f64(f).map(serde_json::Value::Number))
                        .unwrap_or_else(|| serde_json::Value::String(val.to_string())),
                    Some("boolean") => match val.trim() {
                        "true" => serde_json::Value::Bool(true),
                        "false" => serde_json::Value::Bool(false),
                        _ => serde_json::Value::String(val.to_string()),
                    },
                    Some("object") | Some("array") => serde_json::from_str(val.trim())
                        .unwrap_or_else(|_| serde_json::Value::String(val.to_string())),
                    _ => serde_json::Value::String(val.to_string()),
                };
            args.insert(key, typed);
        }
        if !name.is_empty() {
            out.push((name, serde_json::Value::Object(args).to_string()));
        }
    }
    out
}

/// Salvage parser for hermes-style `<tool_call>\n{json}` blocks whose
/// closing `</tool_call>` never arrived (observed on Qwen3.6-27B: the model
/// stops at EOS right after the JSON, the native decoder never completes,
/// and the block leaks into content). Closing tag optional — the same
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

    #[test]
    fn bare_function_block_is_salvaged_and_typed() {
        let schemas = vec![
            serde_json::json!({
                "name": "run_shell_command",
                "description": "d",
                "parameters": {"type": "object", "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"}}}
            })
            .to_string(),
        ];
        let text = "I'll create it.\n\n<function=run_shell_command>\n<parameter=command>\necho 'x' > /tmp/a.txt\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_coder_xml_calls(text, &schemas);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "run_shell_command");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["command"], "echo 'x' > /tmp/a.txt");
        assert_eq!(args["timeout"], 30);
    }

    #[test]
    fn no_function_block_yields_nothing() {
        assert!(parse_coder_xml_calls("plain text </tool_call>", &[]).is_empty());
    }

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
