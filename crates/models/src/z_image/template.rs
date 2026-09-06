//! Z-Image's prompt rendering: the Qwen3 chat template with one user turn
//! and the generation cue, no system prompt —
//!
//! ```text
//! <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
//! ```
//!
//! which is what the reference pipeline's
//! `apply_chat_template(add_generation_prompt=True, enable_thinking=True)`
//! renders (verified against the snapshot's `tokenizer/`: 11 ids for a
//! three-word prompt, `<|im_start|>` 151644 first, `assistant\n` last; no
//! `<think>` block is emitted under `enable_thinking=True`). The `user(prompt)`
//! turn plus the `cue()` of [`crate::qwen_3::template::chatml`] is exactly
//! that, so the constructor is borrowed; nothing here is a chat.
//!
//! A guest encodes the prompt through the bound tokenizer with this
//! template (the SDK's `encode_text`), never padded: the `text` reading
//! takes the native length (`max_sequence_length` 512 is a truncation
//! bound, not a pad target — study §B.1), and padding to a multiple of 32
//! happens on the `refine` lane with learned rows.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    crate::qwen_3::template::chatml(tokenizer)
}
