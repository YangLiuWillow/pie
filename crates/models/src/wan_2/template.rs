//! Wan 2.2's prompt rendering: there is none. `WanPipeline` hands the raw
//! prompt to `T5TokenizerFast`, whose post-processor appends `</s>` (id 1)
//! and nothing else — no system turn, no role markers, no bos —
//! `padding="max_length"` to 512 with `<pad>` (id 0), then truncates the
//! encoder's output to the real length and zero-pads the EMBEDS to 512
//! (study §C.6). The family's `text` reading runs the real-length ids
//! (`forward.rs`); the 512-row zero pad is the context lane's.
//!
//! **The tokenizer cannot be bound today** (`tokenizer.rs`), so this row
//! borrows `qwen_3`'s ChatML constructor the way `mini_dit` does — a
//! column the catalog requires, load-bearing for nothing here. When the
//! umT5 vocabulary loads, this becomes a raw-text template: `{prompt}` +
//! `</s>`, no cue.

use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    crate::qwen_3::template::chatml(tokenizer)
}
