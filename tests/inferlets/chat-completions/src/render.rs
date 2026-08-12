//! History rendering: turn an OpenAI chat message list into the model's
//! canonical chat-template token stream, byte-identical to the pre-rewrite
//! engine's `QwenInstruct` output (`render_text.rs` holds the literals).
//!
//! The rewrite's engine-side template helpers are deliberately NOT used for
//! history replay — `tools::equip()` emits its own separate system turn,
//! `tools::answer()` emits one user turn per result, and there is no
//! assistant-with-tool-calls form at all. All three diverge from the HF
//! template the model was tuned on, and worse, from what this daemon's own
//! save path canonicalizes — so the whole turn structure is assembled here,
//! piecewise, exactly like the old engine did: pre-encoded seam tokens +
//! `model::encode` for content. Piecewise assembly also sidesteps any
//! question of whether embedded special-token strings match during a
//! one-shot encode.
//!
//! Rendering into a `Vec<u32>` (rather than any context object) lets the
//! caller prefill in bounded chunks and render just the suffix after a
//! session resume.

use crate::render_text as rt;
use crate::types::{ChatMessage, MessageContent};
use inferlet::{chat, model};

pub struct Renderer {
    system_prefix: Vec<u32>,
    user_prefix: Vec<u32>,
    assistant_prefix: Vec<u32>,
    user_prefix_no_nl: Vec<u32>,
    assistant_prefix_no_nl: Vec<u32>,
    newline: Vec<u32>,
    turn_suffix: Vec<u32>,
    generation_header: Vec<u32>,
    tool_call_open: Vec<u32>,
    tool_call_mid: Vec<u32>,
    tool_call_close: Vec<u32>,
    tool_response_open: Vec<u32>,
    tool_response_suffix: Vec<u32>,
    /// Engine stop set (`<|im_end|>`, `<|endoftext|>`), plus `<|im_start|>`
    /// when it encodes to a single id: at t=0 a looping model starts
    /// simulating the next turn instead of stopping, and a leaked
    /// `<|im_start|>` round-trips into fake turn boundaries next request.
    pub stop_ids: Vec<u32>,
    /// Special-token strings for message sanitization.
    specials: Vec<String>,
    /// Tool-calling dialect this model was tuned on. Coder-tuned models
    /// served the hermes prompt answer in prose and never call a tool.
    dialect: rt::Dialect,
}

impl Renderer {
    pub fn new() -> Self {
        let encode = |s: &str| model::encode(s);
        let im_start = encode(rt::IM_START);
        let newline = encode(rt::NL);

        let make_prefix = |role: &str| -> Vec<u32> {
            let mut v = im_start.clone();
            v.extend(encode(role));
            v.extend(&newline);
            v
        };
        let make_prefix_no_nl = |role: &str| -> Vec<u32> {
            let mut v = im_start.clone();
            v.extend(encode(role));
            v
        };

        let mut turn_suffix = encode(rt::IM_END);
        turn_suffix.extend(&newline);

        // Qwen3's generation_suffix is empty, so the cue is exactly the
        // assistant prefix. Rendered here (not via `chat::cue()`) so
        // generation and replay share one code path; the engine's cue for
        // ChatML models is the same bytes.
        let generation_header = make_prefix("assistant");

        let mut stop_ids = chat::stop_tokens();
        if im_start.len() == 1 && !stop_ids.contains(&im_start[0]) {
            stop_ids.push(im_start[0]);
        }

        let specials = model::special_tokens()
            .into_iter()
            .filter_map(|t| String::from_utf8(t.bytes).ok())
            .filter(|s| !s.is_empty())
            .collect();

        Self {
            system_prefix: make_prefix("system"),
            user_prefix: make_prefix("user"),
            assistant_prefix: make_prefix("assistant"),
            user_prefix_no_nl: make_prefix_no_nl("user"),
            assistant_prefix_no_nl: make_prefix_no_nl("assistant"),
            newline,
            turn_suffix,
            generation_header,
            tool_call_open: encode(rt::TOOL_CALL_OPEN),
            tool_call_mid: encode(rt::TOOL_CALL_MID),
            tool_call_close: encode(rt::TOOL_CALL_CLOSE),
            tool_response_open: encode(rt::TOOL_RESPONSE_OPEN),
            tool_response_suffix: encode(rt::TOOL_RESPONSE_SUFFIX),
            stop_ids,
            specials,
            dialect: rt::Dialect::detect(&model::name(), &model::architecture()),
        }
    }

    pub fn dialect(&self) -> rt::Dialect {
        self.dialect
    }

    fn role_tokens(&self, prefix: &[u32], msg: &str) -> Vec<u32> {
        let mut tokens = prefix.to_vec();
        tokens.extend(model::encode(msg));
        tokens.extend(&self.turn_suffix);
        tokens
    }

    pub fn system(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.system_prefix, msg)
    }

    pub fn user(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.user_prefix, msg)
    }

    /// Assistant replay strips `<think>…</think>` (the template does this;
    /// we serve the no-think channel, H17).
    pub fn assistant(&self, msg: &str) -> Vec<u32> {
        self.role_tokens(&self.assistant_prefix, rt::strip_thinking(msg))
    }

    /// The generation cue: `<|im_start|>assistant\n`.
    pub fn cue(&self) -> &[u32] {
        &self.generation_header
    }

    /// `<|im_end|>\n` — appended to KV after generation to seal the
    /// assistant turn (the sampled stop token never enters the context).
    pub fn seal_tokens(&self) -> &[u32] {
        &self.turn_suffix
    }

    /// Reference: `<|im_start|>assistant` + (`\n` + content, if any) + per
    /// call `\n<tool_call>\n{"name": "N", "arguments": A}\n</tool_call>` +
    /// `<|im_end|>\n`. No unconditional newline after the role tag — it
    /// comes from whichever branch fires first.
    pub fn assistant_with_tool_calls(
        &self,
        content: Option<&str>,
        calls: &[(String, String)],
    ) -> Vec<u32> {
        if calls.is_empty() {
            return self.assistant(content.unwrap_or(""));
        }
        if self.dialect == rt::Dialect::Coder {
            let mut tokens = self.assistant_prefix_no_nl.clone();
            tokens.extend(model::encode(&rt::coder_assistant_calls_text(content, calls)));
            tokens.extend(&self.turn_suffix);
            return tokens;
        }
        let mut tokens = self.assistant_prefix_no_nl.clone();
        if let Some(c) = content {
            if !c.is_empty() {
                tokens.extend(&self.newline);
                tokens.extend(model::encode(c));
            }
        }
        for (name, arguments_json) in calls {
            tokens.extend(&self.tool_call_open);
            tokens.extend(model::encode(name));
            tokens.extend(&self.tool_call_mid);
            // The HF template runs arguments through `tojson`, so a vLLM-served
            // model saw the re-serialized spaced form, not the compact string
            // qwen-code echoes. Normalize likewise; pass unparseable args raw.
            let args = match serde_json::from_str::<serde_json::Value>(arguments_json) {
                Ok(v) => crate::render_text::tojson(&v),
                Err(_) => arguments_json.clone(),
            };
            tokens.extend(model::encode(&args));
            tokens.extend(&self.tool_call_close);
        }
        tokens.extend(&self.turn_suffix);
        tokens
    }

    /// Reference: consecutive tool-role messages share one
    /// `<|im_start|>user … <|im_end|>\n` turn, but EVERY message still
    /// contributes its own leading `\n<tool_response>\n…\n</tool_response>`
    /// chunk — merging single `answer()` turns would double the newline
    /// between chunks.
    pub fn answer_batch(&self, results: &[(String, String)]) -> Vec<u32> {
        if results.is_empty() {
            return Vec::new();
        }
        if self.dialect == rt::Dialect::Coder {
            // Coder wraps each result in its own newline-terminated block
            // inside one `<|im_start|>user\n … <|im_end|>\n` turn.
            let values: Vec<String> = results.iter().map(|(_, v)| v.clone()).collect();
            let mut tokens = self.user_prefix.clone();
            tokens.extend(model::encode(&rt::coder_tool_response_text(&values)));
            tokens.extend(&self.turn_suffix);
            return tokens;
        }
        let mut tokens = self.user_prefix_no_nl.clone();
        for (_name, value) in results {
            tokens.extend(&self.tool_response_open);
            tokens.extend(model::encode(value));
            tokens.extend(&self.tool_response_suffix);
        }
        tokens.extend(&self.turn_suffix);
        tokens
    }

    /// Render the full conversation: merged system+tools turn first, then
    /// the remaining turns.
    pub fn render_full(
        &self,
        messages: &[ChatMessage],
        tool_schemas: &[String],
        no_think: bool,
    ) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        let mut rest = messages;

        // The chat template folds a leading system message's content into
        // the *same* system turn as the tool schemas rather than two
        // consecutive system turns. Under the Coder dialect a tools-bearing
        // request with no system message of its own still gets one, carrying
        // the template's stand-in text.
        let leading_system = if messages.first().map(|m| m.role.as_str()) == Some("system") {
            rest = &messages[1..];
            Some(messages[0].text())
        } else {
            None
        };
        if let Some(content) = rt::system_turn_content(
            self.dialect,
            leading_system.as_deref(),
            tool_schemas,
        ) {
            out.extend(self.system(&content));
        }

        self.render_messages(rest, no_think, &mut out)?;
        Ok(out)
    }

    /// Render a run of messages (also used for the suffix after a session
    /// resume — the suffix never contains the leading system message, so no
    /// tool-equip handling is needed here).
    pub fn render_messages(
        &self,
        messages: &[ChatMessage],
        no_think: bool,
        out: &mut Vec<u32>,
    ) -> Result<(), String> {
        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            match msg.role.as_str() {
                "system" | "developer" => {
                    out.extend(self.system(&msg.text()));
                    i += 1;
                }
                "user" => {
                    // `/no_think` on *every* user turn, not just the last:
                    // the decoration must be position-independent so a turn
                    // replays identically once it becomes history —
                    // otherwise retained KV prefixes would diverge from the
                    // rebuilt token stream.
                    let text = msg.text();
                    // Coder models have no thinking channel and no soft
                    // switch, so the decoration would be bare noise appended
                    // to every user turn — and a divergence from what a
                    // template-driven server sends.
                    if no_think && self.dialect != rt::Dialect::Coder {
                        out.extend(self.user(&rt::no_think_decorate(&text)));
                    } else {
                        out.extend(self.user(&text));
                    }
                    i += 1;
                }
                "assistant" => {
                    let calls = msg.calls();
                    if calls.is_empty() {
                        out.extend(self.assistant(&msg.text()));
                    } else {
                        let pairs: Vec<(String, String)> = calls
                            .iter()
                            .map(|c| (c.function.name.clone(), c.function.arguments.clone()))
                            .collect();
                        let content = msg.text();
                        out.extend(self.assistant_with_tool_calls(
                            Some(content.as_str()).filter(|s| !s.is_empty()),
                            &pairs,
                        ));
                    }
                    i += 1;
                }
                "tool" => {
                    // Merge consecutive tool results into one replayed turn
                    // — the model was fine-tuned on the merged form.
                    let mut batch: Vec<(String, String)> = Vec::new();
                    while i < messages.len() && messages[i].role == "tool" {
                        batch.push((String::new(), messages[i].text()));
                        i += 1;
                    }
                    out.extend(self.answer_batch(&batch));
                }
                other => return Err(format!("unsupported message role: {other}")),
            }
        }
        Ok(())
    }

    /// Strip the tokenizer's special-token strings out of round-tripped
    /// message content. Piecewise `encode` of content that contains a
    /// literal special-token string (e.g. `<|im_start|>` leaked into an
    /// assistant reply) maps it back to the real token id, planting fake
    /// turn boundaries mid-message (openhands traj job 18825434 degenerated
    /// into token salad this way). Must run before both rendering and
    /// session canonicalization so the address hashes what actually gets
    /// replayed.
    pub fn sanitize_messages(&self, messages: &mut [ChatMessage]) {
        if self.specials.is_empty() {
            return;
        }
        let clean = |s: &mut String| {
            for sp in &self.specials {
                if s.contains(sp.as_str()) {
                    *s = s.replace(sp.as_str(), "");
                }
            }
        };
        for m in messages.iter_mut() {
            if let Some(c) = m.content.as_mut() {
                match c {
                    MessageContent::Text(s) => clean(s),
                    MessageContent::Parts(parts) => {
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
}
