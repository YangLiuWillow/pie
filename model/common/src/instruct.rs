//! Instruct trait — model-specific conversational AI formatting and decoding.
//!
//! Each model architecture provides its own implementation. The API layer
//! delegates to the model's `Instruct` impl for all instruct operations.
//!
//! The *vocabulary* only. `create()` — the registry that picks an
//! implementation for an `arch_name` — is in `pie-model`, because it names
//! every generation and a generation crate cannot depend on the thing that
//! dispatches to it.

/// A model-provided tool-call grammar in EBNF form.
pub struct ToolGrammar {
    pub source: String,
}
// The shared decoders, re-exported so `instruct::decoders` stays a valid
// path: it is what every generation's template imports, and the templates
// became crates without their imports needing to know.
pub use crate::decoders;

/// Events emitted by the chat decoder.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Generated text chunk
    Delta(String),
    /// Special token encountered (token ID)
    Interrupt(u32),
    /// Generation complete (full accumulated text)
    Done(String),
}

/// Events emitted by the reasoning decoder.
#[derive(Debug, Clone)]
pub enum ReasoningEvent {
    /// Reasoning block started
    Start,
    /// Reasoning text chunk
    Delta(String),
    /// Reasoning complete (full reasoning text)
    Complete(String),
}

/// Events emitted by the tool decoder.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    /// Tool call detected
    Start,
    /// Complete tool call: (name, arguments-json)
    Call(String, String),
}

/// Classifies generated tokens into text deltas, interrupts, and done.
pub trait ChatDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent;
    fn reset(&mut self);
}

/// Detects reasoning/thinking blocks in the token stream.
pub trait ReasoningDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent;
    fn reset(&mut self);
}

/// Detects tool call blocks in the token stream.
pub trait ToolDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent;
    fn reset(&mut self);
}

/// Model-specific instruct implementation.
///
/// Each architecture provides its own impl with hardcoded tokens & logic.
/// The tokenizer is owned by the implementation to avoid redundant lookups.
pub trait Instruct: Send + Sync {
    fn system(&self, msg: &str) -> Vec<u32>;
    fn first_user(&self, msg: &str) -> Vec<u32> {
        self.user(msg)
    }
    fn user(&self, msg: &str) -> Vec<u32>;
    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.system(system);
        tokens.extend(self.user(user));
        tokens
    }
    fn assistant(&self, msg: &str) -> Vec<u32>;
    fn cue(&self) -> Vec<u32>;

    /// The generation cue with the thinking channel explicitly closed — the
    /// `enable_thinking=False` form of [`Instruct::cue`] (Qwen renders
    /// `<|im_start|>assistant\n<think>\n\n</think>\n\n`; HF/vLLM emit the
    /// same under that template kwarg). Default: identical to `cue()`,
    /// correct for architectures without a thinking channel.
    fn cue_no_think(&self) -> Vec<u32> {
        self.cue()
    }

    fn seal(&self) -> Vec<u32>;
    fn equip(&self, tools: &[String]) -> Vec<u32>;
    fn answer(&self, name: &str, value: &str) -> Vec<u32>;

    /// Register `tools`, merging `system_content` (a caller-supplied leading
    /// system message, if any) into the *same* system turn as the tool
    /// schemas — some chat templates (Qwen's, notably) fold both into one
    /// turn rather than emitting two separate consecutive system turns,
    /// and feeding the model the latter (out-of-distribution) shape hurts
    /// its tool-call-format reliability.
    ///
    /// Default concatenates a plain `system()` turn (if `system_content` is
    /// present) with `equip()`'s own turn — the old, pre-merge behavior,
    /// correct for architectures without a specific merge rule. Override
    /// when the architecture's template merges them; see `QwenInstruct` for
    /// a worked example.
    fn equip_after_system(&self, system_content: Option<&str>, tools: &[String]) -> Vec<u32> {
        let mut out = Vec::new();
        if let Some(c) = system_content {
            out.extend(self.system(c));
        }
        out.extend(self.equip(tools));
        out
    }

    /// Build tokens for a *replayed* assistant turn that made one or more
    /// tool calls: `content` is any free text that preceded the call(s), and
    /// `calls` are `(name, arguments_json)` pairs — `arguments_json` must
    /// already be a valid JSON-encoded string (the shape
    /// [`ToolDecoder::feed`]'s [`ToolEvent::Call`] produces). This exists
    /// because a plain [`Instruct::assistant`] only knows how to wrap a
    /// string, but reconstructing a past tool-calling turn byte-for-byte
    /// (to match the architecture's own template) requires knowing where
    /// the content ends and each call begins, not just concatenated text.
    ///
    /// Default falls back to a plain `assistant()` replay of `content` only,
    /// silently dropping `calls` — correct for architectures that don't
    /// support tools (mirrors `equip`/`answer`'s no-tool-support behavior
    /// elsewhere in this trait). Override when the architecture supports
    /// tool calling; see `QwenInstruct` for a worked example.
    fn assistant_with_tool_calls(&self, content: Option<&str>, calls: &[(String, String)]) -> Vec<u32> {
        let _ = calls;
        self.assistant(content.unwrap_or(""))
    }

    /// Build tokens for one or more tool results that should be replayed as
    /// a single merged reply turn — most chat templates group consecutive
    /// tool-role messages into one turn rather than one per result, and
    /// calling [`Instruct::answer`] once per result produces a *different*
    /// (and wrong) token sequence than the template's merged form.
    ///
    /// Default folds to repeated single-result `answer()` calls (one turn
    /// per result — the old, pre-merge behavior). Override when the
    /// architecture's template merges consecutive results into one turn;
    /// see `QwenInstruct` for a worked example.
    fn answer_batch(&self, results: &[(String, String)]) -> Vec<u32> {
        results
            .iter()
            .flat_map(|(name, value)| self.answer(name, value))
            .collect()
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder>;
    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder>;
    fn tool_decoder(&self) -> Box<dyn ToolDecoder>;
    /// Returns the parsed tool-call grammar that constrains generation to
    /// the architecture's tool-call format, given a list of tool schemas.
    /// Returns `None` if the architecture doesn't support constrained tool calling.
    fn tool_call_grammar(&self, _tools: &[String]) -> Option<ToolGrammar> {
        None
    }
}
