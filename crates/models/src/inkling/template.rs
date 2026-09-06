use std::sync::Arc;

use chat_template::inkling::Inkling;
use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn inkling(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(Inkling::new(tokenizer))
}
