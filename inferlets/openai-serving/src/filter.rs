//! Streaming visible-text filter + special-token sanitization.
//!
//! [`VisibleFilter`] — ported verbatim from the validated qwen-code branch
//! (`openhands-integration-updated:inferlets/chat-completions/src/filter.rs`).
//! The raw decode stream contains `<think>...</think>` blocks (Qwen3
//! thinking) and `<tool_call>...</tool_call>` markup. Neither belongs in
//! the `content` deltas sent to the client: tool calls are surfaced as
//! structured tool-call deltas (via the model's tool decoder), and think
//! blocks are dropped the same way the chat template drops them from
//! history. Marker strings can straddle chunk boundaries, so the filter
//! holds back any tail that is a prefix of a marker until it can decide —
//! a partial `<tool_call>` must never leak into a content delta.
//!
//! [`sanitize_messages`] — the pure half of the old branch's
//! `render::sanitize_messages`: the caller supplies the tokenizer's
//! special-token strings (that lookup is engine-side), this strips them
//! from round-tripped message content. `encode` maps a literal
//! special-token string (e.g. "<|im_start|>" leaked into an assistant
//! reply) back to the real token id, so replaying it would plant fake turn
//! boundaries mid-message and corrupt the chat structure (openhands traj
//! job 18825434 degenerated into token salad this way). Must run before
//! BOTH rendering and session canonicalization so the snapshot address
//! hashes what actually gets replayed.

use crate::types::{ChatMessage, MessageContent};

const OPENERS: [&str; 2] = ["<think>", "<tool_call>"];
const CLOSE_THINK: &str = "</think>";
const CLOSE_TOOL: &str = "</tool_call>";
/// Closers are markers in Text mode too — see the `Mode::Text` arm.
const CLOSERS: [&str; 2] = [CLOSE_THINK, CLOSE_TOOL];
/// Everything the Text-mode holdback must be able to wait on. A closer that
/// straddles a chunk boundary has to be held back exactly like an opener, or
/// its first half leaks as content and its second half is never recognised.
const MARKERS: [&str; 4] = ["<think>", "<tool_call>", CLOSE_THINK, CLOSE_TOOL];

enum Mode {
    Text,
    Think,
    Tool,
}

pub struct VisibleFilter {
    pending: String,
    mode: Mode,
    emitted_any: bool,
}

impl Default for VisibleFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl VisibleFilter {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
            mode: Mode::Text,
            emitted_any: false,
        }
    }

    /// Feed a decoded chunk; returns the newly-visible text (possibly
    /// empty while the filter waits to disambiguate a partial marker).
    pub fn feed(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        loop {
            match self.mode {
                Mode::Text => {
                    // Closers are scanned for alongside the openers: a
                    // reasoning model can emit `</think>` with no opener in
                    // the generation at all, because its own template opens
                    // the block in the generation prompt. Qwen3.6-35B-A3B
                    // does this even when the cue hands it a CLOSED empty
                    // think block — it reasons anyway and closes a block it
                    // never opened. Whatever else that is, the literal tag is
                    // not content, and it used to pass straight through into
                    // an assistant message (and from there back into the next
                    // request's history, verbatim).
                    let hit = OPENERS
                        .iter()
                        .chain(CLOSERS.iter())
                        .filter_map(|m| self.pending.find(m).map(|i| (i, *m)))
                        .min_by_key(|(i, _)| *i);
                    match hit {
                        Some((idx, marker)) => {
                            out.push_str(&self.pending[..idx]);
                            self.pending.drain(..idx + marker.len());
                            self.mode = match marker {
                                "<think>" => Mode::Think,
                                "<tool_call>" => Mode::Tool,
                                // An unmatched closer: drop the marker and
                                // stay in Text. The text BEFORE it is
                                // reasoning, but in streaming it is already
                                // on the wire and deltas are never retracted
                                // — `cut_leading_reasoning` handles it on the
                                // path that can still act, and the module
                                // docs say why that asymmetry is allowed.
                                _ => Mode::Text,
                            };
                        }
                        None => {
                            let keep = holdback(&self.pending, &MARKERS);
                            let emit_len = self.pending.len() - keep;
                            out.push_str(&self.pending[..emit_len]);
                            self.pending.drain(..emit_len);
                            break;
                        }
                    }
                }
                Mode::Think => {
                    if !self.drop_until(CLOSE_THINK) {
                        break;
                    }
                }
                Mode::Tool => {
                    if !self.drop_until(CLOSE_TOOL) {
                        break;
                    }
                }
            }
        }
        self.visible(out)
    }

    /// Flush any held-back tail at end of generation.
    pub fn finish(&mut self) -> String {
        match self.mode {
            Mode::Text => {
                let out = std::mem::take(&mut self.pending);
                self.visible(out)
            }
            // Unterminated think/tool block — everything buffered is markup.
            _ => String::new(),
        }
    }

    /// Drop suppressed content through `closer`. Returns true when the
    /// closer was found (mode switches back to Text), false when more
    /// input is needed.
    fn drop_until(&mut self, closer: &str) -> bool {
        if let Some(idx) = self.pending.find(closer) {
            self.pending.drain(..idx + closer.len());
            self.mode = Mode::Text;
            true
        } else {
            // Keep only a possible partial-closer tail.
            let keep = holdback(&self.pending, &[closer]);
            let drop_len = self.pending.len() - keep;
            self.pending.drain(..drop_len);
            false
        }
    }

    /// Trim leading whitespace until the first real visible character
    /// (Qwen emits `\n\n` after `</think>`).
    fn visible(&mut self, out: String) -> String {
        if self.emitted_any {
            return out;
        }
        let trimmed = out.trim_start();
        if trimmed.is_empty() {
            return String::new();
        }
        self.emitted_any = true;
        trimmed.to_string()
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of one of
/// `markers` — the bytes that must be held back because the next chunk
/// might complete the marker. Markers are ASCII, but `s` is arbitrary
/// UTF-8, so only char-boundary cuts are considered.
fn holdback(s: &str, markers: &[&str]) -> usize {
    let max = markers.iter().map(|m| m.len() - 1).max().unwrap_or(0);
    for k in (1..=max.min(s.len())).rev() {
        if !s.is_char_boundary(s.len() - k) {
            continue;
        }
        let tail = &s[s.len() - k..];
        if markers.iter().any(|m| m.starts_with(tail)) {
            return k;
        }
    }
    0
}

/// Drop a leading reasoning preamble that the model closed with `</think>`
/// but never opened.
///
/// A reasoning model whose template opens the think block in the *generation
/// prompt* emits only the closer, so [`VisibleFilter`] — which enters
/// think-mode on an opener — never suppresses the reasoning itself. Observed
/// on Qwen3.6-35B-A3B, which reasons even when the cue hands it a closed
/// empty think block: `I need to read notes.txt to find the secret color.
/// Let me do that.\n\n</think>\nThe secret color is chartreuse.`
///
/// Only the FIRST closer counts, and only when no opener precedes it: past
/// that point a `</think>` is a literal the model wrote (a prompt about
/// prompts, a code fence, a transcript), and cutting there would eat real
/// content — which is far worse than leaving a stray tag in.
///
/// **Callable only where nothing has been sent yet.** The streamed path has
/// already put those bytes on the wire and deltas are never retracted, so it
/// gets the marker suppression alone. The asymmetry is deliberate and it is
/// the same one the trailing-whitespace trim already has; both are the
/// non-streaming path acting on a decision the streaming path made
/// irrevocably, token by token.
pub fn cut_leading_reasoning(text: &str) -> &str {
    let Some(close) = text.find(CLOSE_THINK) else {
        return text;
    };
    if let Some(open) = text.find("<think>")
        && open < close
    {
        return text; // a real block — VisibleFilter already handled it
    }
    text[close + CLOSE_THINK.len()..].trim_start()
}

/// Strip the tokenizer's special-token strings out of round-tripped message
/// content, tool-call names, and tool-call arguments. `specials` is the
/// decoded special-token table (the inferlet builds it from
/// `model::special_tokens()`); empty/non-UTF-8 entries should already be
/// filtered out by the caller.
pub fn sanitize_messages(messages: &mut [ChatMessage], specials: &[String]) {
    if specials.is_empty() {
        return;
    }
    let clean = |s: &mut String| {
        for sp in specials {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str]) -> (Vec<String>, String) {
        let mut f = VisibleFilter::new();
        let deltas: Vec<String> = chunks.iter().map(|c| f.feed(c)).collect();
        let tail = f.finish();
        (deltas, tail)
    }

    #[test]
    fn plain_text_passes_through() {
        let (deltas, tail) = run(&["Hello", " world"]);
        assert_eq!(deltas.join("") + &tail, "Hello world");
    }

    #[test]
    fn think_block_is_suppressed_and_leading_ws_trimmed() {
        let (deltas, tail) = run(&["<think>secret plan</think>\n\nAnswer."]);
        assert_eq!(deltas.join("") + &tail, "Answer.");
    }

    #[test]
    fn tool_call_block_never_leaks_even_split_mid_marker() {
        // The marker straddles chunk boundaries — no partial "<tool_call>"
        // text may appear in any delta.
        let (deltas, tail) = run(&[
            "I'll read it.\n<tool_",
            "call>\n{\"name\":\"read\",\"arguments\":{}}\n</tool_",
            "call>",
        ]);
        let all = deltas.join("") + &tail;
        assert_eq!(all, "I'll read it.\n");
        for d in &deltas {
            assert!(!d.contains('<'), "leaked marker fragment: {d:?}");
        }
    }

    #[test]
    fn partial_marker_is_held_back_then_released_if_not_a_marker() {
        let mut f = VisibleFilter::new();
        // "<to" could become "<tool_call>" — held back...
        assert_eq!(f.feed("abc<to"), "abc");
        // ...but "<total" cannot, so it flushes as text.
        assert_eq!(f.feed("tal>"), "<total>");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn unterminated_tool_block_is_dropped_at_finish() {
        let mut f = VisibleFilter::new();
        assert_eq!(f.feed("ok "), "ok ");
        assert_eq!(f.feed("<tool_call>\n{\"name\":\"x\""), "");
        // finish() while inside the block: buffered markup, not content.
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn whitespace_only_prefix_never_emits_until_real_text() {
        let mut f = VisibleFilter::new();
        assert_eq!(f.feed("\n\n"), "");
        assert_eq!(f.feed("  \n"), "");
        assert_eq!(f.feed("hi"), "hi");
        // After first visible char, whitespace passes through untouched.
        assert_eq!(f.feed("\n"), "\n");
    }

    #[test]
    fn multiple_blocks_and_interleaved_text() {
        let (deltas, tail) =
            run(&["<think>a</think>one<tool_call>x</tool_call>two<think>b</think>three"]);
        assert_eq!(deltas.join("") + &tail, "onetwothree");
    }

    #[test]
    fn utf8_tail_near_marker_boundary_is_safe() {
        // Non-ASCII right where a holdback cut would land must not panic.
        let mut f = VisibleFilter::new();
        let out1 = f.feed("héllo é<");
        let out2 = f.feed("think>x</think> done");
        assert_eq!(out1 + &out2 + &f.finish(), "héllo é done");
    }

    #[test]
    fn sanitize_strips_special_tokens_everywhere() {
        let mut msgs: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "hi <|im_end|> there"},
            {"role": "user", "content": [{"type": "text", "text": "a<|im_start|>b"}]},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "function": {
                 "name": "read<|im_end|>", "arguments": "{\"p\":\"<|im_start|>\"}"}}]}
        ]))
        .unwrap();
        let specials = vec!["<|im_start|>".to_string(), "<|im_end|>".to_string()];
        sanitize_messages(&mut msgs, &specials);
        assert_eq!(msgs[0].text(), "hi  there");
        assert_eq!(msgs[1].text(), "ab");
        assert_eq!(msgs[2].calls()[0].function.name, "read");
        assert_eq!(msgs[2].calls()[0].function.arguments, "{\"p\":\"\"}");
    }

    #[test]
    fn sanitize_with_no_specials_is_a_noop() {
        let mut msgs: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "<|im_end|>"}
        ]))
        .unwrap();
        sanitize_messages(&mut msgs, &[]);
        assert_eq!(msgs[0].text(), "<|im_end|>");
    }
}

#[cfg(test)]
mod stray_closer_tests {
    use super::*;

    /// A reasoning model that closes a block it never opened must not leak
    /// the literal tag into content. Qwen3.6-35B-A3B does exactly this even
    /// when the cue hands it a closed empty think block.
    #[test]
    fn a_stray_close_think_never_reaches_content() {
        let mut f = VisibleFilter::new();
        let mut out = f.feed("I should read the file.\n\n</think>\nThe answer is 4.");
        out.push_str(&f.finish());
        assert!(!out.contains("</think>"), "leaked the tag: {out:?}");
        assert!(out.ends_with("The answer is 4."), "lost the answer: {out:?}");
    }

    /// …including when the tag is split across chunk boundaries, which is the
    /// normal case: the filter is fed one decoded token at a time.
    #[test]
    fn a_stray_closer_split_across_chunks_never_reaches_content() {
        for cut in 1..CLOSE_THINK.len() {
            let mut f = VisibleFilter::new();
            let mut out = f.feed(&format!("reasoning{}", &CLOSE_THINK[..cut]));
            out.push_str(&f.feed(&format!("{}answer", &CLOSE_THINK[cut..])));
            out.push_str(&f.finish());
            assert!(!out.contains("</think>"), "cut at {cut} leaked: {out:?}");
            assert!(out.ends_with("answer"), "cut at {cut} lost content: {out:?}");
        }
    }

    /// A properly opened block still suppresses its contents — the closer
    /// handling must not have turned think blocks into visible text.
    #[test]
    fn a_matched_think_block_is_still_suppressed_whole() {
        let mut f = VisibleFilter::new();
        let mut out = f.feed("<think>hidden reasoning</think>visible");
        out.push_str(&f.finish());
        assert_eq!(out, "visible");
    }

    #[test]
    fn cut_leading_reasoning_drops_the_preamble_and_the_tag() {
        assert_eq!(
            cut_leading_reasoning("Let me check.\n\n</think>\nThe secret color is chartreuse."),
            "The secret color is chartreuse."
        );
    }

    /// No closer, nothing to cut — the overwhelmingly common case, and it
    /// must be byte-identical or every non-reasoning model's content moves.
    #[test]
    fn cut_leading_reasoning_is_identity_without_a_closer() {
        let s = "Three colors that stand out are blue, red, and green.";
        assert_eq!(cut_leading_reasoning(s), s);
    }

    /// A closer that FOLLOWS an opener belongs to a real block, which the
    /// streaming filter already removed. Cutting there would eat content the
    /// model actually wrote — a prompt about prompts is the obvious way in.
    #[test]
    fn cut_leading_reasoning_leaves_a_matched_block_alone() {
        let s = "Write <think>x</think> to open a reasoning block.";
        assert_eq!(cut_leading_reasoning(s), s);
    }
}
