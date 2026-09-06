//! LTX-2.5's prompt rendering: there is none. `LTX2Pipeline` hands
//! `prompt.strip()` to the Gemma tokenizer with `add_special_tokens=True`
//! and no chat template — the snapshot's `tokenizer/chat_template.jinja` is
//! the optional prompt ENHANCER's, a separate Gemma-4-E2B-it the sglang
//! pipeline does not run (study §C.2).
//!
//! The catalog wants a template column per row, and this family has no
//! `text` reading to render for yet (`model.rs`), so this row borrows
//! `qwen_3`'s ChatML constructor the way `wan_2` and `mini_dit` do — load-
//! bearing for nothing here. When the trunk lands, this becomes a raw-text
//! template: the stripped prompt, left-padded to 1024, nothing else.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    crate::qwen_3::template::chatml(tokenizer)
}
