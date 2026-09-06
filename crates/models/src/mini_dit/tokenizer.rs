//! `mini-dit` has no text side: its caption rows arrive as floats through a
//! `Context` port, never as token ids, and nothing in this family embeds or
//! decodes a vocabulary.
//!
//! The catalog's `Sku` still carries a tokenizer contract column, so this row
//! borrows the smallest one this build ships — `qwen_3`'s plain ChatML
//! contract — rather than inventing a vocabulary that does not exist. Nothing
//! here is load-bearing for a denoise pass; a guest that asks this row to
//! tokenize is asking the wrong model.

pub use crate::qwen_3::tokenizer::CONTRACT;
