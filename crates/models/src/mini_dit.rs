//! `mini-dit` — the synthetic generative family M0 is verified against.
//!
//! It is the first row in this catalog whose pass predicts a **velocity**
//! rather than logits: three lanes of one request (caption, image, context),
//! float ports in, `seam::VELOCITY` out, no kv space anywhere. Its reference
//! is `scripts/imagegen/mini_dit_ref.py` and its goldens live outside the
//! repo at `$PIE_IMAGEGEN_GOLDEN/mini-dit`; `scripts/imagegen/README.md` §3
//! states the architecture, the tensor names and the dump keys.
//!
//! Nothing real is served by this row. It exists so that the substrate — D2's
//! streams and groups, D3's float ports and readouts, D6's modulation, D7's
//! axis rope — has one catalog row that names every one of them, before a
//! 4-billion-parameter model does.

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// The label the media front-ends dispatch on. Nothing matches it today: this
/// family has no pixel side, and its patch rows arrive as floats.
pub const ARCH: &str = "mini_dit";

/// One row, one rank. The identification sweep asks this row last (it is the
/// final entry of `SKUS`), and its import reads a checkpoint of eight dozen
/// tensors no other family spells.
pub fn skus() -> Vec<crate::Sku> {
    let tap = forward::Tap::from_env();
    let mut rows = crate::skus![
        (
            "mini-dit",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(Dtype::Bf16, tp).tapped(forward::Tap::from_env()),
        ),
        // The same text two ranks wide (`mini-dit-bf16-kv-bf16-tp2`): two
        // heads per rank. The rung between the one-rank row and the
        // four-rank one, which is what makes the reduction noise a slope
        // rather than a single number.
        (
            "mini-dit",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(Dtype::Bf16, tp).tapped(forward::Tap::from_env()),
        ),
        // And four ranks wide (`mini-dit-bf16-kv-bf16-tp4`): one head per
        // rank, the D14 bring-up row (`Model::mini` states the tp
        // convention). An artifact imported at one rank serves either, each
        // rank reading its band.
        (
            "mini-dit",
            4,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(Dtype::Bf16, tp).tapped(forward::Tap::from_env()),
        ),
    ];
    // The generative facts a guest sizes a job from (design D12). Stated
    // beside the row rather than by the macro, which fills `None` for every
    // family — the `gemma_4_diffusion` precedent for `diffusion`.
    for row in &mut rows {
        row.generative = Some(forward::generative(tap.as_deref()));
    }
    rows
}
