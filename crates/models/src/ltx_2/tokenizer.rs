//! LTX-2.5's tokenizer is the text encoder's: the Gemma tokenizer shipped
//! under `tokenizer/` in the snapshot (`GemmaTokenizer`, a 262 144-piece
//! SentencePiece vocabulary the pipeline drives LEFT-padded to
//! `max_length = 1024` with `add_special_tokens=True`).
//!
//! This family declares no `text` reading yet (`model.rs` states why), so
//! nothing here reads ids: the connectors take the trunk's packed hidden
//! states on a float port, and the parity harness carries the reference's
//! numbers. Until a `text` reading lands, this row borrows the smallest
//! contract the build ships, as `wan_2` and `mini_dit` do.
//!
//! When it lands, the contract is `gemma_4`'s: the same vocabulary family,
//! with the pipeline's own padding (LEFT, to 1024) owned by the family's
//! encode arm and never by a guest (design D13).

pub use crate::qwen_3::tokenizer::CONTRACT;
