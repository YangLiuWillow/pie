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
//! **And the contract is umT5's own now**, not the borrowed `qwen_3` one:
//! `</s>` at 1, `<pad>` at 0 and `<unk>` at 3, pinned at their ids rather
//! than merely by name. An artifact carrying somebody else's vocabulary
//! answers a coherent picture of the WRONG prompt, and that is the failure
//! a contract turns into a named refusal at boot.
//!
//! What is left is re-importing the shipped artifact from the real
//! `tokenizer/`, which the staged snapshot replaced while the loader could
//! not read one. After that a guest passes `--prompt` rather than
//! `--prompt-ids`.

use ::tokenizer::contract::Contract;

/// The end-of-sequence umT5's `TemplateProcessing` appends to every encode.
/// Pinned at its id and not merely by name: `</s>` at any other id means a
/// different vocabulary, and the DiT would be conditioned on rows that spell
/// something else.
pub const EOS: &str = "</s>";
/// The pad the encoder's fixed 512-row context rectangle is filled with.
pub const PAD: &str = "<pad>";
/// What a character outside the 256 300 pieces becomes. `unk_id` in the
/// tokenizer's own JSON, and the reason a Unigram can spell every string.
pub const UNK: &str = "<unk>";

/// Every marker this row cites, in one place so the contract and the reader
/// spell them once.
pub const MARKERS: &[&str] = &[EOS, PAD, UNK];

/// The three ids that decide whether an artifact carries umT5's vocabulary
/// or somebody else's. A row served against the wrong one produces a
/// coherent picture of the wrong prompt, which is the failure a contract
/// exists to turn into a named refusal at boot.
pub const PINNED: &[(&str, u32)] = &[(PAD, 0), (EOS, 1), (UNK, 3)];

/// **umT5's own contract**, replacing the borrowed `qwen_3` one this row
/// used while `crates/tokenizer` could not read a Unigram.
pub const CONTRACT: Contract = Contract {
    markers: &[MARKERS],
    pinned: PINNED,
};
