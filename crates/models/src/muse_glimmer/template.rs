use std::sync::Arc;

use chat_template::atem::Atem;
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn muse_glimmer(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(Atem::new(tokenizer))
}
