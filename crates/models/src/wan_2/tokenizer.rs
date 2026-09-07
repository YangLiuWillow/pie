//! Wan 2.2's tokenizer is umT5-xxl's: a SentencePiece **Unigram** model
//! (`tokenizer/tokenizer.json`: `model.type = "Unigram"`, 256 300 pieces
//! under 256 384 embedding rows, `unk_id` 3; a `Metaspace` pre-tokenizer
//! and decoder with `replacement = "▁"`, `prepend_scheme = "always"`; a
//! `TemplateProcessing` post-processor appending `</s>` = 1; `<pad>` = 0,
//! `<s>` = 2, 300 `<extra_id_*>` specials at 256 000+), with `spiece.model`
//! beside it.
//!
//! **`crates/tokenizer` READS IT NOW** (2026-09-07). It used to accept
//! `model.type == "BPE"` alone, and the three things it was missing — a
//! Unigram model (the piece→score table and a Viterbi best-segmentation over
//! it, with `unk` fallback and byte-fallback off), the `Metaspace`
//! pre-tokenizer at `prepend_scheme = "always"`, and the
//! `TemplateProcessing` post-processor's trailing `</s>` — are all in
//! (`tokenizer::unigram`, and the `Unigram` arm of `Pipeline`). Checked
//! against `tokenizers` itself: umT5's real `tokenizer.json` answers
//! `"a red bicycle leaning on a blue wall"` with
//! `[289, 4062, 188625, 346, 291, 1350, 369, 289, 15258, 21006, 1]`, which
//! is the reference's own vector and the one `gates.py` feeds this row as
//! `--prompt-ids`.
//!
//! **And it BAKES now too.** `pie.tokenizer/1` grew a sixth object,
//! `tokenizer/unigram_scores` — one `f32` a token id — which is OPTIONAL:
//! a BPE tokenizer writes none, and that absence is what says "not a
//! Unigram", so every artifact written before this keeps loading unchanged.
//! Round-tripped on the real 256 300-piece vocabulary: bake, read back, and
//! the ids are identical.
//!
//! This row still borrows `qwen_3`'s contract, because a CONTRACT is a list
//! of markers a serving row needs pinned and umT5's has not been written.
//! Once it is, `pie model import` carries umT5's own vocabulary and the
//! guests can pass `--prompt` instead of `--prompt-ids`.

pub use crate::qwen_3::tokenizer::CONTRACT;
