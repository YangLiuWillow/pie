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
//! **What is still not done: BAKING it into a `.zt`.** The canonical form
//! (`pie.tokenizer/1`) states five objects and none of them holds a per-piece
//! score — `MERGE_TABLE` is empty for a Unigram and reusing it would be
//! exactly the reinterpretation that format exists to rule out. So
//! `tokenizer::canonical` refuses a Unigram BY NAME, `pie model import`
//! still cannot carry umT5's vocabulary into an artifact, and this row still
//! borrows the smallest contract the build ships (`qwen_3`'s) as `mini_dit`
//! does. Closing it is a sixth object plus the three readers of
//! `canonical::OBJECTS` (`worker::weights`, `runtime::model`, and the
//! format's own test).
//!
//! Until then the `text` reading's ids come from outside (the goldens carry
//! the reference's prompt embeds, so `denoise` parity needs no tokenizer).

pub use crate::qwen_3::tokenizer::CONTRACT;
