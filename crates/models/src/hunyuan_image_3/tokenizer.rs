//! HunyuanImage 3's tokenizer contract: the Hunyuan LLM vocabulary
//! (133 120 rows under a `PreTrainedTokenizerFast`) plus the multimodal
//! specials the sequence assembler writes.
//!
//! What this row actually READS as one token — the reason each marker is
//! here — is the T2I sequence the trunk denoises inside
//! (`tokenization_hunyuan_image_3.py:899-938`):
//!
//! ```text
//! <|startoftext|> [prompt] <boi> <img_size_1024> <img_ratio_k> <timestep> <img> x h*w <eoi>
//! ```
//!
//! plus `<cfg>` (the unconditional branch replaces every prompt token with
//! it, so both CFG branches have identical length, identical masks and
//! identical rotary positions), and `<guidance>` on the CFG-distilled SKU.
//! **`<timestep_r>` is NOT demanded**: MeanFlow's second timestep token is
//! spelled only by the Instruct-Distil tokenizer, and the base repo's
//! vocabulary (verified: 127 957 BPE pieces plus 2 126 added tokens) does
//! not carry it. A row that reads it would refuse every base checkpoint.
//!
//! The stage markers `<think>` / `<recaption>` / `<answer>` are read by the
//! AR phases' forced-transition logits processor, which is a guest epilogue
//! mask over `logits()` (design D10) and needs each of them to be one id.
//!
//! **Three ids are pinned** because the row's identity hangs on them: they
//! are `config.json`'s own `im_start_id` / `im_end_id` / `image_token_id`,
//! which the modeling code reads at runtime to find the image span. A
//! tokenizer that spells `<boi>` elsewhere is not this checkpoint's, and a
//! misplaced `<img>` would silently move the canvas.
//!
//! The size and ratio tokens are a family (`<img_size_{256,512,1024,...}>`,
//! `<img_ratio_0..36>`); only the two the base T2I path writes by hand are
//! demanded here — `<img_size_1024>` and `<img_ratio_0>`, the first of the
//! run the resolution head samples from.

use ::tokenizer::contract::Contract;

/// `<|endoftext|>` (eos, id 127957) is `sep2` of the `hunyuan-image-3`
/// conversation template and the AR phases' only natural stop.
pub const END_OF_TEXT: &str = "<|endoftext|>";
pub const START_OF_TEXT: &str = "<|startoftext|>";

/// What a chat turn ends on. `</answer>` closes the Instruct template's
/// assistant image block, so a guest reading text back stops on either.
pub const STOP_TOKENS: &[&str] = &[END_OF_TEXT, "</answer>"];

/// The image-span markers: everything between `<boi>` and `<eoi>`.
pub const IMAGE_TOKENS: &[&str] = &[
    "<boi>",
    "<eoi>",
    "<img>",
    "<cfg>",
    "<timestep>",
    "<guidance>",
    "<joint_img_sep>",
    "<img_size_1024>",
    "<img_ratio_0>",
];

/// The AR stage markers the CoT / recaption / resolution phases force.
pub const STAGE_TOKENS: &[&str] = &[
    "<think>",
    "</think>",
    "<recaption>",
    "</recaption>",
    "<answer>",
];

pub const CONTRACT: Contract = Contract {
    markers: &[STOP_TOKENS, IMAGE_TOKENS, STAGE_TOKENS, &[START_OF_TEXT]],
    pinned: &[("<boi>", 128_000), ("<eoi>", 128_001), ("<img>", 128_006)],
};
