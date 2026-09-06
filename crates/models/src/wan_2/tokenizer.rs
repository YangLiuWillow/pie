//! Wan 2.2's tokenizer is umT5-xxl's: a SentencePiece **Unigram** model
//! (`tokenizer/tokenizer.json`: `model.type = "Unigram"`, 256 300 pieces
//! under 256 384 embedding rows, `unk_id` 3; a `Metaspace` pre-tokenizer
//! and decoder with `replacement = "▁"`, `prepend_scheme = "always"`; a
//! `TemplateProcessing` post-processor appending `</s>` = 1; `<pad>` = 0,
//! `<s>` = 2, 300 `<extra_id_*>` specials at 256 000+), with `spiece.model`
//! beside it.
//!
//! **`crates/tokenizer` cannot load it.** Its Hugging Face loader accepts
//! `model.type == "BPE"` alone (`loader/huggingface.rs`, `compile_profile`:
//! the byte-level, byte-fallback, sentencepiece-BPE and Metaspace-BPE
//! profiles all sit on a BPE merge table), and there is no `.model`
//! protobuf reader. What is missing, precisely: a Unigram model (the
//! piece→score table and a Viterbi best-segmentation over it, with
//! `unk` fallback and byte-fallback off), the `Metaspace` pre-tokenizer
//! at `prepend_scheme = "always"`, and the `TemplateProcessing`
//! post-processor's trailing `</s>`. Until then this row borrows the
//! smallest contract the build ships — `qwen_3`'s — as `mini_dit` does;
//! the `text` reading's ids must come from elsewhere (the goldens carry
//! the reference's prompt embeds, so `denoise` parity needs no tokenizer).

pub use crate::qwen_3::tokenizer::CONTRACT;
