//! `flux_2` — Black Forest Labs' FLUX.2, the second real generative family
//! in the catalog (design §0, milestone M2), starting with `FLUX.2-klein-4B`.
//!
//! An MM-DiT of five double-stream blocks over `[txt ‖ img]` then twenty
//! parallel single-stream blocks over the joined sequence, three shared
//! modulation linears for the whole trunk, a four-axis rope with a
//! reference-index axis, behind a Qwen3-4B encoder read at three depths,
//! and the `AutoencoderKLFlux2` autoencoder ([`vae`], both ways) — one plan
//! of four arms selected per lane by the fact word ([`forward`] states the
//! table a guest programs against).
//!
//! Two rows: the flagship `black-forest-labs/FLUX.2-klein-4B` (bf16 on
//! disk, Apache-2.0, four distilled steps, no guidance embedder) and the
//! miniature `scripts/imagegen/flux2_golden.py --mini` writes (the
//! transformer alone at dim 256 with a guidance embedder, random weights,
//! the parity fixture). `FLUX.2-klein-9B`, the `-base` rows (true CFG) and
//! `FLUX.2-dev` (Mistral-Small encoder, 15360-wide context) are further
//! constructors over the same text; `klein-9b-kv`'s masked layout is one
//! `RaggedMask` away (study §I.2).

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;
pub mod vae;

use model::Model;
use model_dsl::Dtype;

/// The label the media front-ends dispatch on. Nothing matches it today:
/// this family's pixel side is a VAE, not a vision tower.
pub const ARCH: &str = "flux_2";

/// The flagship first, the miniature last (identification is catalog
/// order, and the miniature reads a checkpoint no operator ships).
pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "flux2-klein-4b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::klein_4b(Dtype::Bf16, tp),
        ),
        (
            "flux2-mini",
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
            "flux2-klein-4b" => Model::klein_4b(Dtype::Bf16, 1),
            "flux2-mini" => Model::mini(Dtype::Bf16, 1),
            other => unreachable!("no flux_2 row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
