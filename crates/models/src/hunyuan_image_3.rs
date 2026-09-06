//! `hunyuan_image_3` — Tencent's HunyuanImage 3.0, the AR-plus-diffusion
//! hybrid (design §0, milestone M6): one Hunyuan-A13B MoE trunk that
//! writes its own chain of thought, predicts its own output resolution as
//! a token, and then denoises an image *inside the same token sequence*.
//!
//! It is the catalog's only `forward-diffusion` GENERATIVE row (design
//! D10): a denoise step is paged prefix KV that never moves plus a canvas
//! of `h·w` image rows whose K/V is rewritten in place every step, and the
//! guest binds `input("latents")` beside `embed`. Four readings under one
//! plan ([`forward`] states the table a guest programs against): `encode`
//! (the causal text prefill and the AR phases), `denoise` (the canvas
//! against the frozen prefix), and the conv image head's two voxel arms
//! `image.in` / `image.out`.
//!
//! Four rows: the flagship `tencent/HunyuanImage-3.0` at one rank (what
//! `pie model import` identifies and converts) and at `tp = 4` (what
//! serves — 80 B parameters do not fit one card in bf16; design D10 puts
//! the 77 B of routed experts in `U8g64` and everything else in bf16), a
//! `U4g64` fallback row for a deployment that needs the memory for long
//! reference contexts, and the miniature
//! `scripts/imagegen/hy3_golden.py --mini` writes (two layers, eight
//! experts, a 64-wide head, random weights, the parity fixture).
//!
//! Not yet in this text: the SigLIP2 reference-image tower and the 3-D
//! `AutoencoderKLConv3D` (a guest decodes latents outside pie until they
//! land), the Instruct/Distil `<guidance>` and `<timestep_r>` token rows,
//! and CFG as two prefixes (two lanes, one per branch — the guest's
//! arithmetic, not this text's).

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// The `ROWS.arch` this family's rows carry — the checkpoint's own
/// `model_type`.
pub const ARCH: &str = "hunyuan_image_3_moe";

/// The flagship first, the miniature last (identification is catalog
/// order, and the miniature reads a checkpoint no operator ships).
pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "hunyuanimage3-80b-a13b",
            1,
            [Dtype::Bf16, Dtype::U8g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flagship(Dtype::Bf16, Dtype::U8g64, Dtype::Bf16, tp),
        ),
        (
            "hunyuanimage3-80b-a13b",
            4,
            [Dtype::Bf16, Dtype::U8g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flagship(Dtype::Bf16, Dtype::U8g64, Dtype::Bf16, tp),
        ),
        // D10's fallback: 4-bit experts halve the routed banks again
        // (≈10 GB a rank) at a golden diff this catalog does not yet own.
        (
            "hunyuanimage3-80b-a13b",
            4,
            [Dtype::Bf16, Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flagship(Dtype::Bf16, Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "hunyuanimage3-mini",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(Dtype::Bf16, Dtype::Bf16, tp),
        ),
    ];
    // Both fact columns, stated beside the row: this family is a diffusion
    // row (its passes are `forward-diffusion`, D10) AND a generative one
    // (readings, latent space, schedule — D12). Built from the same
    // constructor the row traces with, so the three cannot disagree.
    for row in &mut rows {
        let model = match row.recipe.text {
            "hunyuanimage3-80b-a13b" => {
                Model::flagship(Dtype::Bf16, row.recipe.weights[1], Dtype::Bf16, 1)
            }
            "hunyuanimage3-mini" => Model::mini(Dtype::Bf16, Dtype::Bf16, 1),
            other => unreachable!("no hunyuan_image_3 row is called `{other}`"),
        };
        row.diffusion = Some(crate::Diffusion {
            canvas: canvas_rows(&model),
            hidden: model.dims.hidden,
            // The reference has no self-conditioning: its cross-step state
            // is the KV cache, not a soft embedding.
            self_cond_taps: 0,
        });
        row.generative = Some(model.generative());
    }
    rows
}

/// The image rows one denoising loop refines — `model.canvas()`'s answer.
/// 1024² at stride 16 and patch 1 is 64 × 64; the miniature's reference
/// image is 128² = 8 × 8 (`hy3_golden.py --mini`).
fn canvas_rows(model: &Model) -> u32 {
    let side = if model.dims.layers > 4 { 64 } else { 8 };
    side * side
}
