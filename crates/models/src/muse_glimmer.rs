//! Muse Glimmer (`meta-models/Muse-Glimmer-30B`, HF `model_type`
//! `muse_glimmer`): a dense 30B text decoder with a perception encoder
//! beside it. This family reads the text; the tower is not declared.

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// Identification order: the first row whose import fits the checkpoint
/// wins. The bf16 row is the checkpoint's own form and goes first; the
/// U4g64 row reads the same bf16 planes and quantizes on the way in, so it
/// is what `--sku` picks for a box the 60 GB bf16 stack does not fit (14.6
/// GiB, 35 tok/s decode on an L40S). No U8g64 row: CUDA has no 8-bit dense
/// reader, so such a row's weight tier is the bf16 upcast (85 GB) and the
/// card refuses it.
pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "muse-glimmer-30b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::muse_glimmer,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b30(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "muse-glimmer-30b",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::muse_glimmer,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b30(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "muse-glimmer-30b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::muse_glimmer,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b30(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        // The parity miniature, after every real row so identification never
        // picks it; a gate names it by SKU. Eight layers: two whole periods
        // (`benches/shrink_checkpoint.py --layers 0-3,48-51`).
        (
            "muse-glimmer-30b-mini-l8",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::muse_glimmer,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b30_mini(8, Dtype::Bf16, Dtype::Bf16, tp),
        ),
    ]
}
