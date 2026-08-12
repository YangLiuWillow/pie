//! Streaming visible-text filter.
//!
//! The raw decode stream contains `<think>...</think>` blocks (Qwen3
//! thinking) and `<tool_call>...</tool_call>` markup. Neither belongs in
//! the `output_text` deltas sent to the client: tool calls are surfaced as
//! structured `function_call` items (via `tools::Decoder`), and think
//! blocks are dropped the same way the chat template drops them from
//! history. Marker strings can straddle chunk boundaries, so the filter
//! holds back any tail that is a prefix of a marker until it can decide.

const OPENERS: [&str; 2] = ["<think>", "<tool_call>"];
const CLOSE_THINK: &str = "</think>";
const CLOSE_TOOL: &str = "</tool_call>";

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

impl VisibleFilter {
    pub fn new() -> Self {
        Self::starting(false)
    }

    /// Construct a filter for a turn that begins INSIDE a `<think>` block,
    /// because the cue opened one and left it open (Qwen3.5/3.6 lineage —
    /// see `render_text::THINK_OPEN`).
    ///
    /// This is the only construction that is correct regardless of whether
    /// the model tags its reasoning: there is no opener to miss, and a turn
    /// truncated mid-reasoning yields empty content rather than leaked
    /// reasoning, because the filter never left think-mode.
    pub fn starting(in_think: bool) -> Self {
        Self {
            pending: String::new(),
            mode: if in_think { Mode::Think } else { Mode::Text },
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
                    let hit = OPENERS
                        .iter()
                        .filter_map(|m| self.pending.find(m).map(|i| (i, *m)))
                        .min_by_key(|(i, _)| *i);
                    match hit {
                        Some((idx, marker)) => {
                            out.push_str(&self.pending[..idx]);
                            self.pending.drain(..idx + marker.len());
                            self.mode = if marker == "<think>" { Mode::Think } else { Mode::Tool };
                        }
                        None => {
                            let keep = holdback(&self.pending, &OPENERS);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Starting inside the block is correct whether or not the model tags
    /// its reasoning — the case that defeats every marker-based approach.
    #[test]
    fn think_mode_start_suppresses_untagged_reasoning() {
        let mut f = VisibleFilter::starting(true);
        // Untagged reasoning prose, exactly what was observed with a cue that
        // opened no block at all. Nothing here is a marker.
        assert_eq!(f.feed("Thinking Process:\n1. Analyze the request.\n"), "");
        assert_eq!(f.feed("2. Decide on a greeting.\n"), "");
        // Only after the model closes the block does content become visible.
        let out = f.feed("</think>\n\nHello there friend");
        assert_eq!(out.trim(), "Hello there friend");
    }

    /// A turn truncated by `max_tokens` before any closer yields EMPTY
    /// content rather than leaked reasoning. This is the property the
    /// non-streaming `cut_leading_reasoning` workaround cannot provide,
    /// because it needs a closer to fire at all.
    #[test]
    fn truncation_mid_reasoning_leaks_nothing() {
        let mut f = VisibleFilter::starting(true);
        assert_eq!(f.feed("Here's a thinking process:\n\n1. Analyze"), "");
        assert_eq!(f.feed(" the user input, which is"), "");
        // Stream ends here — nothing was ever emitted.
        assert!(!f.emitted_any);
    }

    /// The non-thinking lineage is unchanged: a filter that starts in text
    /// mode still emits text and still strips a tagged block.
    #[test]
    fn text_mode_start_is_unchanged() {
        let mut f = VisibleFilter::starting(false);
        assert_eq!(f.feed("plain "), "plain ");
        assert_eq!(f.feed("<think>hidden</think>tail"), "tail");
    }
}
