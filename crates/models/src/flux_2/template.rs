//! FLUX.2 klein's prompt rendering: the Qwen3 chat template with one user
//! turn and the generation cue under `enable_thinking=False` —
//!
//! ```text
//! <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
//! ```
//!
//! which is what `Flux2KleinPipeline._get_qwen3_prompt_embeds` renders
//! (`apply_chat_template(add_generation_prompt=True, enable_thinking=False)`,
//! no system message — verified against the snapshot's `tokenizer/`: 20 ids
//! for the golden's eight-word prompt, `<|im_start|>` 151644 first, the
//! empty think block `151667 271 151668 271` last). The `user(prompt)` turn
//! plus the cue of a ChatML template whose generation suffix is that empty
//! think block is exactly this, so the constructor is `qwen_3`'s with one
//! field changed; nothing here is a chat.
//!
//! A guest encodes the prompt through the bound tokenizer with this
//! template (the SDK's `encode_text`), truncated at `TE_MAX_TOKENS` and not
//! padded (`forward.rs` on why).

use std::sync::Arc;

use chat_template::chatml::{ChatML, ChatMLInstruct};
use tokenizer::Tokenizer;

use crate::template::Instruct;

/// The empty reasoning block `enable_thinking=False` appends to the cue.
pub const NO_THINKING: &str = "<think>\n\n</think>\n\n";

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(ChatMLInstruct::new(
        tokenizer,
        ChatML {
            thinking: true,
            preserve_thinking: false,
            tools: false,
            generation_suffix: NO_THINKING,
            stop_tokens: crate::qwen_3::tokenizer::STOP_TOKENS,
        },
    ))
}
