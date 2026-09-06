//! `wan_2` — Alibaba's Wan 2.2, the third real generative family in the
//! catalog (design §0, milestone M3), starting with `Wan2.2-TI2V-5B`.
//!
//! A single-stream video DiT of thirty identical blocks — modulated
//! self-attention over a three-axis rope, cross-attention into a 512-row
//! umT5 context, a GELU FFN — under one shared `time_proj` and per-block
//! learned tables, behind the umT5-xxl encoder (relative-position bias,
//! no cache) and the Wan 2.2 causal-conv VAE (4×16×16, 48 channels, a
//! per-conv frame cache): one plan of four arms selected per lane by the
//! fact word ([`forward`] states the table a guest programs against, and
//! which arms are not yet readings a guest can name).
//!
//! Three rows: the flagship `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (fp32
//! transformer and VAE cast to bf16, bf16 encoder, flow shift 5.0, one
//! backbone) and the two miniatures `scripts/imagegen/wan22_golden.py
//! --mini` writes — `d128` (the real 128-wide head, the parity fixture)
//! and `nano` (a 24-wide head, for the rope split alone). The A14B rows
//! (two backbones at a sigma boundary, D9) are one more constructor and
//! one more reading bit away over the same text.

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// The label the media front-ends dispatch on. Nothing matches it today:
/// this family's pixel side is a VAE, not a vision tower.
pub const ARCH: &str = "wan_2";

/// The flagship first, the miniatures last (identification is catalog
/// order, and a miniature reads a checkpoint no operator ships).
pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "wan22-ti2v-5b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::ti2v_5b(Dtype::Bf16, tp),
        ),
        (
            "wan22-mini-d128",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini_d128(Dtype::Bf16, tp),
        ),
        (
            "wan22-mini-nano",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini_nano(Dtype::Bf16, tp),
        ),
    ];
    // The generative facts a guest sizes a job from (design D12), stated
    // beside the row rather than by the macro. Built from the same
    // constructor the row traces with, so the two cannot disagree.
    for row in &mut rows {
        let model = match row.recipe.text {
            "wan22-ti2v-5b" => Model::ti2v_5b(Dtype::Bf16, 1),
            "wan22-mini-d128" => Model::mini_d128(Dtype::Bf16, 1),
            "wan22-mini-nano" => Model::mini_nano(Dtype::Bf16, 1),
            other => unreachable!("no wan_2 row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
