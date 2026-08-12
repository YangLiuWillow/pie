//! Pure string builders for the ChatML render — the exact byte formats of
//! the pre-rewrite engine's `QwenInstruct` (old
//! `runtime/src/model/instruct/qwen3.rs`, the normative reference). The
//! rewrite's engine-side template diverges from these in three places
//! (separate tool system turn, per-call tool-response turns, no
//! assistant-with-tool-calls form), so the render is assembled inferlet-side
//! from these strings + `model::encode`; see `render.rs` for the token
//! assembly. Keeping the strings pure keeps them natively testable.

/// Which tool-calling dialect the loaded model was fine-tuned on.
///
/// This is not cosmetic. Serving a Coder-tuned model the hermes/JSON tool
/// prompt makes it answer in prose and stop without ever calling a tool —
/// measured on the 2026-08-12 H200 A/B, where 2 of 5 agent tasks produced
/// zero tool calls and the rest took divergent paths (docs §11).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dialect {
    /// `<tool_call>{"name": …, "arguments": …}</tool_call>` (Qwen3, Qwen3.5).
    Hermes,
    /// `<tool_call><function=…><parameter=…>` XML (Qwen3-Coder).
    Coder,
}

impl Dialect {
    /// Pick a dialect from the model identity. pie exposes only the config's
    /// `[model] name` and the architecture, and `qwen3_moe` covers both the
    /// Coder and non-Coder MoE models — so the name is the only signal, and
    /// a Coder deployment must carry "coder" in it (the bench config does).
    ///
    /// **Detection must not depend on the spelling of `architecture`.** The
    /// engine passes the driver's arch *stem* — `architectures[0]` lowercased
    /// with the task suffix stripped — so `Qwen3_5MoeForConditionalGeneration`
    /// arrives as `qwen3_5moe`, without the underscore an HF `model_type`
    /// would have. Keying on that spelling is how the engine-side
    /// `instruct::create` silently dropped every tool schema for Qwen MoE/VL
    /// models (found on `liu/opencode-integration`, fixed there in
    /// `model/src/instruct.rs`). The substring tests below are chosen to hold
    /// under either spelling.
    pub fn detect(model_name: &str, architecture: &str) -> Dialect {
        let hay = format!("{model_name} {architecture}").to_lowercase();
        if hay.contains("coder") {
            Dialect::Coder
        } else {
            Dialect::Hermes
        }
    }
}

/// Whether the loaded model is a Qwen3.5/3.6-lineage *thinking* model, whose
/// chat template ends the generation prompt inside an OPEN `<think>` block
/// rather than after the bare role header.
///
/// This is a separate axis from [`Dialect`] on purpose: it decides how the
/// turn *starts*, not how tools are spelled, and the two need not move
/// together while the Qwen3.6 tool dialect is still unimplemented (§14).
///
/// Substring-based for the reason given on [`Dialect::detect`] — `qwen3_5`,
/// `qwen3_5moe`, `qwen3_5_moe`, `qwen3.6` and `qwen3_6` must all match.
pub fn lineage_opens_think(model_name: &str, architecture: &str) -> bool {
    let hay = format!("{model_name} {architecture}").to_lowercase();
    let hay: String = hay.chars().filter(|c| *c != '_' && *c != '.' && *c != '-').collect();
    hay.contains("qwen35") || hay.contains("qwen36") || hay.contains("qwen3next")
}

/// The generation-prompt tail for a thinking-lineage model: the block is
/// opened and deliberately left open, so the model continues *inside* it.
///
/// Why open rather than closed-and-empty: a filter that starts in think-mode
/// suppresses reasoning deterministically from token zero — no buffering, no
/// markers to miss, and a turn truncated by `max_tokens` mid-reasoning
/// yields empty content instead of leaked reasoning. The alternatives both
/// fail somewhere: emitting nothing leaves the model free to reason untagged
/// (observed on `liu/opencode-integration`), and emitting a *closed* empty
/// block leaves its own closer unmatched (also observed there). Neither
/// failure is reliably reproducible — it depends on prompt context and
/// possibly sampling temperature, still unresolved between the two branches
/// — which is exactly why starting inside the block is preferred: its
/// correctness does not rest on that question.
pub const THINK_OPEN: &str = "<think>\n";

/// The Coder template's stand-in system message when a request carries tools
/// but no system turn of its own.
pub const CODER_DEFAULT_SYSTEM: &str =
    "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.";

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

/// Qwen3-Coder's tool preamble, byte-for-byte with that model's chat
/// template (`chat_template.jinja`, the `# Tools` branch). It carries its
/// own leading `"\n\n"` because the template appends it directly to the
/// system message with no separator of its own.
pub fn build_coder_tool_system_prompt(tools: &[String]) -> String {
    use std::fmt::Write as _;

    // The template's render_extra_keys macro: mappings/sequences via
    // `tojson`, everything else via Jinja's `| string`. That filter is
    // Python's `str()`, so booleans and null render capitalized —
    // `<additionalProperties>True</additionalProperties>`, not `true`.
    fn render_val(v: &serde_json::Value) -> String {
        match v {
            serde_json::Value::Object(_) | serde_json::Value::Array(_) => tojson(v),
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Bool(true) => "True".to_string(),
            serde_json::Value::Bool(false) => "False".to_string(),
            serde_json::Value::Null => "None".to_string(),
            other => other.to_string(),
        }
    }
    fn extra_keys(out: &mut String, v: Option<&serde_json::Value>, handled: &[&str]) {
        if let Some(serde_json::Value::Object(map)) = v {
            for (k, val) in map {
                if handled.contains(&k.as_str()) {
                    continue;
                }
                let _ = write!(out, "\n<{k}>{}</{k}>", render_val(val));
            }
        }
    }

    let mut p = String::from(
        "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>",
    );
    for tool in tools {
        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) else {
            continue;
        };
        // Accept both {"type":"function","function":{…}} and a bare function.
        let func = parsed.get("function").unwrap_or(&parsed);
        let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let _ = write!(p, "\n<function>\n<name>{name}</name>");
        if let Some(d) = func.get("description").and_then(|d| d.as_str()) {
            let _ = write!(p, "\n<description>{}</description>", d.trim());
        }
        p.push_str("\n<parameters>");
        let params = func.get("parameters");
        if let Some(props) = params
            .and_then(|x| x.get("properties"))
            .and_then(|x| x.as_object())
        {
            for (pname, pfields) in props {
                let _ = write!(p, "\n<parameter>\n<name>{pname}</name>");
                if let Some(t) = pfields.get("type") {
                    let _ = write!(p, "\n<type>{}</type>", render_val(t));
                }
                if let Some(d) = pfields.get("description").and_then(|d| d.as_str()) {
                    let _ = write!(p, "\n<description>{}</description>", d.trim());
                }
                extra_keys(&mut p, Some(pfields), &["name", "type", "description"]);
                p.push_str("\n</parameter>");
            }
        }
        extra_keys(&mut p, params, &["type", "properties"]);
        p.push_str("\n</parameters>");
        extra_keys(&mut p, Some(func), &["type", "name", "description", "parameters"]);
        p.push_str("\n</function>");
    }
    p.push_str(
        "\n</tools>\n\n\
         If you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
         <tool_call>\n\
         <function=example_function_name>\n\
         <parameter=example_parameter_1>\n\
         value_1\n\
         </parameter>\n\
         <parameter=example_parameter_2>\n\
         This is the value for the second parameter\n\
         that can span\n\
         multiple lines\n\
         </parameter>\n\
         </function>\n\
         </tool_call>\n\n\
         <IMPORTANT>\n\
         Reminder:\n\
         - Function calls MUST follow the specified format: an inner <function=...></function> \
         block must be nested within <tool_call></tool_call> XML tags\n\
         - Required parameters MUST be specified\n\
         - You may provide optional reasoning for your function call in natural language BEFORE \
         the function call, but NOT after\n\
         - If there is no function call available, answer the question like normal with your \
         current knowledge and do not tell the user about function calls\n\
         </IMPORTANT>",
    );
    p
}

/// Inner text of the system turn, or `None` when the conversation has no
/// system turn at all (no system message and no tools).
pub fn system_turn_content(
    dialect: Dialect,
    system_content: Option<&str>,
    tools: &[String],
) -> Option<String> {
    let system = system_content.filter(|s| !s.is_empty());
    match dialect {
        Dialect::Hermes => match (system, tools.is_empty()) {
            (s, true) => s.map(str::to_string), // no tools: no preamble at all
            (s, false) => Some(merged_system_content(s, tools)),
        },
        Dialect::Coder => {
            if tools.is_empty() {
                return system.map(str::to_string);
            }
            let head = system.unwrap_or(CODER_DEFAULT_SYSTEM);
            Some(format!("{head}{}", build_coder_tool_system_prompt(tools)))
        }
    }
}

/// Coder assistant turn carrying tool calls: everything between
/// `<|im_start|>assistant` and `<|im_end|>`. Content is trimmed and wrapped
/// in newlines only when non-empty, matching the template's branch.
pub fn coder_assistant_calls_text(content: Option<&str>, calls: &[(String, String)]) -> String {
    let mut out = String::new();
    if let Some(c) = content {
        let c = c.trim();
        if !c.is_empty() {
            out.push('\n');
            out.push_str(c);
            out.push('\n');
        }
    }
    for (name, arguments_json) in calls {
        out.push_str("\n<tool_call>\n<function=");
        out.push_str(name);
        out.push_str(">\n");
        if let Ok(serde_json::Value::Object(map)) =
            serde_json::from_str::<serde_json::Value>(arguments_json)
        {
            for (key, value) in &map {
                out.push_str("<parameter=");
                out.push_str(key);
                out.push_str(">\n");
                out.push_str(&match value {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => tojson(value),
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                });
                out.push_str("\n</parameter>\n");
            }
        }
        out.push_str("</function>\n</tool_call>");
    }
    out
}

/// Coder tool-result run: inner text of the single `user` turn that a run of
/// consecutive `tool` messages collapses into. Note this differs from the
/// hermes batching — each block is newline-terminated here.
pub fn coder_tool_response_text(values: &[String]) -> String {
    let mut out = String::new();
    for v in values {
        out.push_str("<tool_response>\n");
        out.push_str(v);
        out.push_str("\n</tool_response>\n");
    }
    out
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

    /// The Coder builders must reproduce `apply_chat_template` byte-for-byte.
    /// The golden is generated from the real model's template (see
    /// `tests/coder_template_golden.json`, regenerate with transformers).
    #[test]
    fn coder_render_matches_the_real_template() {
        let golden: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/coder_template_golden.json"
        ))
        .expect("golden parses");
        let tools: Vec<String> = golden["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(tojson)
            .collect();
        let with_system = golden["cases"]["with_system_and_tools"].as_str().unwrap();
        let no_system = golden["cases"]["no_system_with_tools"].as_str().unwrap();

        // 1. System turn, with and without a system message of its own.
        let sys = system_turn_content(Dialect::Coder, Some("You are Qwen Code."), &tools).unwrap();
        let expect_sys = with_system
            .strip_prefix("<|im_start|>system\n")
            .unwrap()
            .split("<|im_end|>")
            .next()
            .unwrap();
        assert_eq!(sys, expect_sys, "coder system turn diverges");

        let bare = system_turn_content(Dialect::Coder, None, &tools).unwrap();
        let expect_bare = no_system
            .strip_prefix("<|im_start|>system\n")
            .unwrap()
            .split("<|im_end|>")
            .next()
            .unwrap();
        assert_eq!(bare, expect_bare, "coder default-system turn diverges");
        assert!(bare.starts_with(CODER_DEFAULT_SYSTEM));

        // 2. Assistant turn with a tool call (string, integer and object args).
        let calls = vec![(
            "run_shell_command".to_string(),
            r#"{"command":"ls -la","timeout":30,"opts":{"cwd":"/tmp"}}"#.to_string(),
        )];
        let got = coder_assistant_calls_text(Some("I'll look."), &calls);
        let seg = with_system.split("<|im_start|>assistant").nth(1).unwrap();
        let expect = seg.split("<|im_end|>").next().unwrap();
        assert_eq!(got, expect, "coder tool-call replay diverges");

        // 3. A run of tool results collapses into one user turn.
        let got = coder_tool_response_text(&[
            "a.txt\nb.txt".to_string(),
            "second result".to_string(),
        ]);
        let after = with_system.split("</tool_call><|im_end|>\n").nth(1).unwrap();
        let expect = after
            .strip_prefix("<|im_start|>user\n")
            .unwrap()
            .split("<|im_end|>")
            .next()
            .unwrap();
        assert_eq!(got, expect, "coder tool-response batching diverges");
    }

    /// The stem trap: the engine hands us `architectures[0]` lowercased with
    /// the task suffix stripped, NOT an HF `model_type`. Both spellings must
    /// reach the same answer, or the thinking lineage silently renders the
    /// non-thinking cue — the same class of bug that dropped tool schemas
    /// engine-side for every Qwen MoE/VL model.
    #[test]
    fn thinking_lineage_survives_both_arch_spellings() {
        // HF model_type spellings.
        for arch in ["qwen3_5_moe", "qwen3_5_moe_text", "qwen3_5", "qwen3_next"] {
            assert!(lineage_opens_think("default", arch), "model_type {arch}");
        }
        // Driver arch-stem spellings (underscores collapsed by the stem rule).
        for arch in ["qwen3_5moe", "qwen35moe", "qwen3next"] {
            assert!(lineage_opens_think("default", arch), "arch stem {arch}");
        }
        // And from the config name alone, which is what we actually rely on.
        for name in ["qwen3.6-35b-a3b", "Qwen3.5-35B-A3B", "qwen3_6"] {
            assert!(lineage_opens_think(name, "unknown"), "config name {name}");
        }
        // Non-thinking lineages must NOT open a block.
        for (name, arch) in [
            ("qwen3-coder-30b-a3b", "qwen3_moe"),
            ("default", "qwen3moe"),
            ("Qwen3-0.6B", "qwen3"),
            ("default", "llama"),
        ] {
            assert!(!lineage_opens_think(name, arch), "{name}/{arch}");
        }
    }

    /// The open block is deliberately left unclosed — closing it here is the
    /// failure mode the opencode branch hit, where the model's own closer
    /// then has no opener.
    #[test]
    fn think_open_is_an_opener_only() {
        assert_eq!(THINK_OPEN, "<think>\n");
        assert!(!THINK_OPEN.contains("</think>"));
    }

    #[test]
    fn dialect_detection_keys_on_the_model_identity() {
        assert_eq!(
            Dialect::detect("qwen3-coder-30b-a3b", "qwen3_moe"),
            Dialect::Coder
        );
        assert_eq!(Dialect::detect("default", "qwen3_moe"), Dialect::Hermes);
        assert_eq!(Dialect::detect("Qwen3-0.6B", "qwen3"), Dialect::Hermes);
    }

    #[test]
    fn hermes_system_turn_is_absent_only_without_system_and_tools() {
        assert!(system_turn_content(Dialect::Hermes, None, &[]).is_none());
        assert!(system_turn_content(Dialect::Coder, None, &[]).is_none());
        assert_eq!(
            system_turn_content(Dialect::Hermes, Some("S"), &[]).unwrap(),
            "S"
        );
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
