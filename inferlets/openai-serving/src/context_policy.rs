//! B-2: server-side context editing, expressed as a mask over retained KV.
//!
//! ## What this is for
//!
//! A coding agent's history is dominated by tool output, and most of it is dead
//! weight within a couple of turns — a file the model already read, a test run
//! it already reacted to. Every harness deals with this by **compacting**:
//! rewriting the history client-side and re-sending it. That is the one shape
//! where an append-only prefix cache cannot help anyone. vLLM's APC keyed the
//! old prefix; the rewritten history is a different prefix, so the next turn
//! re-prefills the whole thing — and compaction fires exactly when the session
//! is largest.
//!
//! pie can do it without re-prefilling, because the KV is still resident: leave
//! the tokens in place and stop attending to them.
//!
//! ## Why masking and not deletion
//!
//! Deletion looks cheaper and is not expressible. `WorkingSet::discard` removes
//! whole pages, and RoPE is baked into K **at write time**, so the surviving
//! tail still encodes its original absolute positions. Renumbering densely
//! breaks every relative distance against that tail; leaving a gap instead is
//! rejected by the driver, which requires `position_id < seqlen` ("position is
//! outside its request KV extent", `batch/forward.cpp`). Masking changes
//! neither the KV extent nor any position, so both constraints hold trivially.
//!
//! The cost is a dense `[query_tokens, kv_len]` bool mask per fire. That is
//! affordable *because of* the session design: on a resumed turn the fire only
//! spans the delta (~190 tokens on our bench), not the whole history, so the
//! mask is ~1.5 MB rather than the ~128 MB a full-prefill-width mask would be.
//! A cold turn has nothing stale to mask, so it never pays it.
//!
//! ## Opt-in, and why
//!
//! Nothing here fires unless the request asks for it. A server that silently
//! stops attending to tokens the client sent is serving a different context
//! than the one it was given — the exact silent-divergence class this
//! integration has spent its whole life avoiding. The client opts in per
//! request; the wire stays full-history.

use serde::Deserialize;

/// What each rendered span *is*, so a policy can talk about tool output without
/// the engine having to re-parse tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    /// The system turn plus tool schemas — never droppable: it is the contract
    /// the model is operating under, and dropping it changes behaviour rather
    /// than trimming context.
    System,
    User,
    Assistant,
    /// A merged batch of tool results. This is the droppable one.
    ToolOutput,
    /// The generation cue. Never retained, so never maskable.
    Cue,
}

/// One rendered span: `[start, end)` in the render's token space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub kind: SpanKind,
    pub start: u32,
    pub end: u32,
}

/// Per-request opt-in. Absent (the default) means nothing is ever masked.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct ContextPolicy {
    /// Keep the newest N tool-output spans attended; mask everything older.
    /// `None` disables the rule.
    #[serde(default)]
    pub keep_recent_tool_outputs: Option<u32>,
    /// Never mask a span shorter than this. Masking a 12-token "ok" saves
    /// nothing and costs a mask row, so the floor keeps the policy from
    /// producing churn with no benefit.
    #[serde(default)]
    pub min_span_tokens: Option<u32>,
}

impl ContextPolicy {
    pub fn is_noop(&self) -> bool {
        self.keep_recent_tool_outputs.is_none()
    }
}

/// Token ranges to stop attending to, given the rendered spans and a policy.
///
/// Returns ascending, non-overlapping `[start, end)` ranges. Only
/// [`SpanKind::ToolOutput`] is ever selected: masking a user turn would drop
/// the task, and masking the system turn would change the model's contract.
///
/// `retained_len` bounds the result — a span that extends past what the KV
/// actually holds cannot be masked, because there is nothing there to mask.
pub fn spans_to_mask(
    spans: &[Span],
    policy: &ContextPolicy,
    retained_len: u32,
) -> Vec<(u32, u32)> {
    let keep = match policy.keep_recent_tool_outputs {
        Some(k) => k,
        None => return Vec::new(),
    };
    let floor = policy.min_span_tokens.unwrap_or(0);

    let tool: Vec<&Span> = spans
        .iter()
        .filter(|s| s.kind == SpanKind::ToolOutput && s.end <= retained_len)
        .collect();
    let drop_count = tool.len().saturating_sub(keep as usize);
    tool.into_iter()
        .take(drop_count)
        .filter(|s| s.end.saturating_sub(s.start) >= floor)
        .map(|s| (s.start, s.end))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(kind: SpanKind, start: u32, end: u32) -> Span {
        Span { kind, start, end }
    }

    fn convo() -> Vec<Span> {
        vec![
            sp(SpanKind::System, 0, 100),
            sp(SpanKind::User, 100, 120),
            sp(SpanKind::Assistant, 120, 140),
            sp(SpanKind::ToolOutput, 140, 340), // oldest
            sp(SpanKind::Assistant, 340, 360),
            sp(SpanKind::ToolOutput, 360, 560),
            sp(SpanKind::Assistant, 560, 580),
            sp(SpanKind::ToolOutput, 580, 780), // newest
        ]
    }

    #[test]
    fn absent_policy_masks_nothing() {
        let p = ContextPolicy::default();
        assert!(p.is_noop());
        assert!(spans_to_mask(&convo(), &p, 780).is_empty());
    }

    #[test]
    fn keeps_the_newest_and_masks_the_rest() {
        let p = ContextPolicy { keep_recent_tool_outputs: Some(1), min_span_tokens: None };
        // Three tool outputs, keep 1 → the two oldest are masked, in order.
        assert_eq!(spans_to_mask(&convo(), &p, 780), vec![(140, 340), (360, 560)]);
    }

    #[test]
    fn never_masks_system_user_or_assistant() {
        let p = ContextPolicy { keep_recent_tool_outputs: Some(0), min_span_tokens: None };
        let masked = spans_to_mask(&convo(), &p, 780);
        // Every returned range must fall inside a ToolOutput span.
        for (s, e) in masked {
            assert!(
                convo().iter().any(|x| x.kind == SpanKind::ToolOutput
                    && x.start == s
                    && x.end == e),
                "masked ({s},{e}) is not a tool-output span"
            );
        }
    }

    #[test]
    fn short_spans_are_left_alone() {
        let spans = vec![
            sp(SpanKind::ToolOutput, 0, 12),   // tiny: not worth a mask row
            sp(SpanKind::ToolOutput, 12, 500),
            sp(SpanKind::ToolOutput, 500, 700),
        ];
        let p = ContextPolicy {
            keep_recent_tool_outputs: Some(1),
            min_span_tokens: Some(50),
        };
        assert_eq!(spans_to_mask(&spans, &p, 700), vec![(12, 500)]);
    }

    #[test]
    fn spans_past_the_retained_length_are_not_maskable() {
        // The last tool output is in this turn's DELTA, not in retained KV —
        // there is nothing resident to mask.
        let p = ContextPolicy { keep_recent_tool_outputs: Some(0), min_span_tokens: None };
        let masked = spans_to_mask(&convo(), &p, 560);
        assert_eq!(masked, vec![(140, 340), (360, 560)]);
    }

    #[test]
    fn keeping_more_than_exist_masks_nothing() {
        let p = ContextPolicy { keep_recent_tool_outputs: Some(9), min_span_tokens: None };
        assert!(spans_to_mask(&convo(), &p, 780).is_empty());
    }
}
