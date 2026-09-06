//! FLUX.2 klein's tokenizer is the encoder's: the `Qwen2TokenizerFast` of
//! Qwen3-4B (vocab 151 936, `<|im_start|>` 151644, `<|im_end|>` 151645,
//! `<|endoftext|>` 151643 as the pad, no bos). Its contract is the ChatML
//! one `qwen_3` states; nothing about the vocabulary is this family's.

pub use crate::qwen_3::tokenizer::CONTRACT;
