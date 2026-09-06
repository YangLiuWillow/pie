//! MiniMax H3's tokenizer is its encoder's: the `Qwen2TokenizerFast` of
//! Qwen3-VL-32B (`FL2VA/tokenizer/`, shared with `FL2VA/processor/`;
//! 151 936 embedding rows over a 151 669-token vocabulary, pad
//! `<|endoftext|>` 151643, eos `<|im_end|>` 151645, the vision specials
//! `<|vision_start|>` 151652 / `<|vision_end|>` 151653 /
//! `<|image_pad|>` 151655 / `<|video_pad|>` 151656 present but unread by
//! this text-only row). Its contract is the ChatML one `qwen_3` states —
//! the two stop markers — with nothing pinned: no special id is read by
//! this family's text.

pub use crate::qwen_3::tokenizer::CONTRACT;
