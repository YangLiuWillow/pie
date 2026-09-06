//! `z-image` — Tongyi-MAI's Z-Image, the first real generative family in the
//! catalog (design §0, milestone M1).
//!
//! A single-stream DiT (`[image ‖ caption]`, image first) behind a Qwen3-4B
//! encoder, under one plan of three readings — `text`, `refine`, `denoise`
//! — selected per lane by the fact word ([`forward`] states the table a
//! guest programs against). The FLUX VAE is not read yet ([`forward`]'s
//! `vae_decode` stub; `import.rs` lists its tensors).
//!
//! Two rows: the flagship `Tongyi-MAI/Z-Image-Turbo` (fp32 transformer cast
//! to bf16, bf16 encoder, static shift 3.0, eight distilled steps, no CFG)
//! and the miniature `scripts/imagegen/zimage_golden.py --mini` writes (the
//! transformer alone at dim 256, random weights, the parity fixture). The
//! base `Tongyi-MAI/Z-Image` is the same text at shift 6.0 with CFG; it is
//! one more constructor away.
//!
//! Follow-up recorded here: the encoder is declared in this family
//! (`model::TextEncoder`) rather than as a `qwen_3::model::Model`, because
//! that family is Qwen3.5/3.6 (gated attention, GDN mixers, partial rope)
//! and Qwen3-4B is none of those things; when `qwen_3` grows a gate-less
//! all-attention row, this family's `te` should become it and
//! `import::text_encoder` should delegate under the `te.` prefix.

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// The label the media front-ends dispatch on. Nothing matches it today:
/// this family's pixel side is a VAE, not a vision tower.
pub const ARCH: &str = "z_image";

/// The flagship first, the miniature last (identification is catalog order,
/// and the miniature reads a checkpoint no operator ships).
pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "z-image-turbo",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::turbo(Dtype::Bf16, tp),
        ),
        (
            "z-image-mini",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(Dtype::Bf16, tp),
        ),
    ];
    // The generative facts a guest sizes a job from (design D12), stated
    // beside the row rather than by the macro (the `gemma_4_diffusion`
    // precedent for `diffusion`). Built from the same constructor the row
    // traces with, so the two cannot disagree.
    for row in &mut rows {
        let model = match row.recipe.text {
            "z-image-turbo" => Model::turbo(Dtype::Bf16, 1),
            "z-image-mini" => Model::mini(Dtype::Bf16, 1),
            other => unreachable!("no z-image row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
