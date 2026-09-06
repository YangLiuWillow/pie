//! `minimax_h3` — MiniMaxAI's MiniMax-H3, the joint audio+video member of
//! the catalog (design §0, milestone M5).
//!
//! A 33.1 B single-stream DiT: fifty identical blocks at width 5376 (56
//! MHA heads of 128, per-head QK-RMSNorm, a 14336-wide SwiGLU) over ONE
//! packed row sequence `[text | keyframes/references | audio | video]`,
//! with two text-only refiner blocks ahead of it and modality-specific
//! adaLN-Zero modulation whose thirteen billion parameters are a
//! `Linear(2688 → 18·5376)` per block that every row gathers by
//! `3·timestep_index + modality_tag` ([`model`]'s header says how this
//! text states that gather without a new op). Video rows are (1, 2, 2)
//! patches of a 24-channel f16 t4 latent, audio rows 32-dimensional
//! latents of a 40 Hz stereo codec, and the two share ONE rotary time
//! axis measured in 1/40 s audio ticks — which is what keeps them in
//! sync. Behind it: Qwen3-VL-32B truncated after layer 50 with no final
//! norm. One plan of three readings — `text`, `refine`, `denoise` —
//! selected per lane by the fact word ([`forward`] states the table a
//! guest programs against).
//!
//! Four rows: the flagship `MiniMaxAI/MiniMax-H3` at its `FL2VA/`
//! partition (bf16, CFG-distilled, 50 sigma points / 49 evaluations, two
//! shifts — video 12, audio 3), its `-tp2` and `-tp4` worlds (the
//! flagship is 61.7 GiB of transformer beside a 48 GiB encoder; study
//! §H.4), and the miniature `scripts/imagegen/h3_golden.py --mini`
//! writes (two blocks at width 128, random weights, the parity fixture).
//! The `Ref2VA/` partition is the same text over a second transformer and
//! one more constructor away.
//!
//! Recorded follow-ups, in the order they matter:
//!
//! 1. **the vision tower.** The shipped conditioner is a VLM: keyframes
//!    and reference clips enter the DiT twice, once as Qwen3-VL vision
//!    tokens INSIDE the text stream (where they take the *visual* adaLN
//!    tag) and once as clean latent rows. This row runs the text-only
//!    path, where M-RoPE's three sections carry one position and the
//!    rotary is plain neox; wiring `qwen_3`'s tower under `te.` and
//!    splitting the text lane so its vision rows take modality 0 is next.
//! 2. **the two autoencoders.** The video VAE fits the `Spatial`
//!    vocabulary as it stands (`forward`'s header says how) and is
//!    blocked on the checkpoint alone: its weights sit at
//!    `video_vae/source/model.safetensors`, a nested folder discovery
//!    does not descend into. The audio codec is not: it wants transposed
//!    1-D convolutions, weight-norm reparameterisation and
//!    `Snake`/`SnakeBeta`, none of which `Spatial` states.
//! 3. **two velocity widths.** The video head is 96 wide and the audio
//!    head 32, and a plan carries one `velocity` export, so the audio
//!    prediction rides the `hidden` seam ([`forward::denoise`]).

pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// The label the media front-ends dispatch on. Nothing matches it today:
/// this family's pixel side is a VAE, not a vision tower — and its
/// encoder's tower is the recorded follow-up above.
pub const ARCH: &str = "minimax_h3";

/// The flagship first, its sharded worlds next, the miniature last
/// (identification is catalog order, and the miniature reads a checkpoint
/// no operator ships).
pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "minimax-h3-fl2va",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::fl2va(Dtype::Bf16, tp),
        ),
        (
            "minimax-h3-fl2va",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::fl2va(Dtype::Bf16, tp),
        ),
        (
            "minimax-h3-fl2va",
            4,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::fl2va(Dtype::Bf16, tp),
        ),
        (
            "minimax-h3-mini",
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
    // beside the row rather than by the macro. Built from the same
    // constructor the row traces with, so the two cannot disagree.
    for row in &mut rows {
        let model = match row.recipe.text {
            "minimax-h3-fl2va" => Model::fl2va(Dtype::Bf16, row.recipe.tp),
            "minimax-h3-mini" => Model::mini(Dtype::Bf16, row.recipe.tp),
            other => unreachable!("no minimax_h3 row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
