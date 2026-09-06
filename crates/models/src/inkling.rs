//! Inkling (`thinkingmachines/Inkling`, HF `model_type` `inkling_mm_model`):
//! a 975B-parameter, 41B-active mixture of experts with a learned
//! relative-position bias in place of rotary embeddings. This family reads
//! the text; the vision hMLP, the audio embedding and the MTP heads are not
//! declared.

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// Identification order: the first row whose import fits the checkpoint
/// wins. The miniature comes last, so identification never picks it; a
/// gate names it by SKU.
pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "inkling",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::inkling,
            &tokenizer::CONTRACT,
            |tp: u32| Model::full(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        // `benches/shrink_checkpoint.py --layers 0-6 --experts 8`: both dense
        // layers, four local sparse layers and the first global one.
        (
            "inkling-mini-l7-e8",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::inkling,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(7, 8, Dtype::Bf16, Dtype::Bf16, tp),
        ),
    ]
}
