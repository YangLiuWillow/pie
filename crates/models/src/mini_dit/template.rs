//! `mini-dit` has no chat surface. The catalog's template column is not
//! optional, so this row reuses `qwen_3`'s ChatML constructor over whatever
//! tokenizer it is handed — the same borrowing its [`super::tokenizer`]
//! contract does, and for the same reason.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    crate::qwen_3::template::chatml(tokenizer)
}
