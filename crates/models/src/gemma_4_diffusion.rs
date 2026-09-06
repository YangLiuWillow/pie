//! Catalog rows for DiffusionGemma: Gemma 4's trunk run in two attention
//! modes (a causal encoder that writes the KV, a bidirectional denoiser that
//! only reads it) with a self-conditioning block on the denoiser's input.
//!
//! Step one of the bring-up: the trunk alone, imported from the diffusion
//! checkpoint's spelling and served as a plain causal model. The denoise
//! class and the self-conditioning block land on top of this row.

pub mod forward;
pub mod import;
pub mod model;

use model::Model;
use model_dsl::Dtype;

use crate::gemma_4::{template, tokenizer};

/// The `ROWS.arch` this family's rows carry — the checkpoint's own
/// `model_type`.
pub const ARCH: &str = "diffusion_gemma";

/// Identification order: the first row whose import fits the checkpoint wins.
/// Every row is a diffusion row: the canvas is a fact about the family.
pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        // The same trunk with 8-bit experts. The denoiser's stopping rule
        // keys on the entropy of its most confident rows, and 4-bit
        // experts raise that floor a hundredfold (1e-6 → 1e-4 nats), which
        // roughly triples the steps a block takes to converge; 8-bit
        // experts read twice the bytes a step and converge like the
        // reference. Chosen by `--sku`; the 4-bit row stays the default.
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U8g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b(Dtype::U8g64, Dtype::Bf16, tp),
        ),
        // Precision where it is spent: the two mixed rows say which half
        // the denoiser's convergence rides on. Dense U8 over 4-bit experts
        // costs the 4-bit row's bytes plus a tenth; 4-bit dense over 8-bit
        // experts costs the 8-bit row's.
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U8g64, Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b_experts(Dtype::U8g64, Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U4g64, Dtype::U8g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b_experts(Dtype::U4g64, Dtype::U8g64, Dtype::Bf16, tp),
        ),
        // The plan `mlx-community/diffusiongemma-26B-A4B-it-4bit` ships:
        // attention, dense MLP, router and embedding at 8 bits, experts
        // AND the self-conditioning block at the 4-bit default (the
        // converter's override list names the dense stack only). The
        // dense-U8 row above would read the block at 8 bits and refuse.
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U8g64, Dtype::U4g64, Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b_experts_self_cond(
                Dtype::U8g64,
                Dtype::U4g64,
                Dtype::U4g64,
                Dtype::Bf16,
                tp,
            ),
        ),
        // The dense weights as they came (bf16, ≈3 GiB) over 4-bit experts:
        // no dense dequant on the step at all, the same convergence as
        // 8-bit dense, a gigabyte and a half more than the mixed row.
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::Bf16, Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b_experts(Dtype::Bf16, Dtype::U4g64, Dtype::Bf16, tp),
        ),
    ];
    for row in &mut rows {
        row.diffusion = Some(crate::Diffusion {
            canvas: model::CANVAS,
            hidden: model::HIDDEN,
            self_cond_taps: crate::gemma_4::model::SELF_COND_TAPS,
        });
    }
    rows
}
