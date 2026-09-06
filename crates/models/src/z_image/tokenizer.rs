//! Z-Image's tokenizer is the encoder's: the `Qwen2Tokenizer` of Qwen3-4B
//! (`tokenizer/` in the snapshot; pad `<|endoftext|>` 151643, eos
//! `<|im_end|>` 151645, no bos). Its contract is the ChatML one `qwen_3`
//! states — the two stop markers — with nothing pinned: the vocabulary is
//! 151 669 tokens under 151 936 embedding rows, and no special id is read
//! by this family's text.

pub use crate::qwen_3::tokenizer::CONTRACT;
