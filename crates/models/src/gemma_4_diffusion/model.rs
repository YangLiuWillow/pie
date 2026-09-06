use model_dsl::Dtype;

use crate::gemma_4;

/// `canvas_length`: the block one denoising loop refines, and the row count
/// of every denoise fire.
pub const CANVAS: u32 = 256;

/// The trunk's hidden width — the 26B-A4B's, the one published size.
pub const HIDDEN: u32 = 2816;

/// DiffusionGemma: one Gemma 4 trunk. The encoder and the decoder share
/// every weight (the checkpoint stores the trunk once, under the decoder's
/// name); what differs between the two is how a fire attends and what is
/// added to its input — the self-conditioning block, the one part the
/// decoder has that the encoder does not (`gemma_4::model::SelfCond`).
pub struct Model {
    pub trunk: gemma_4::model::Model,
}

impl Model {
    /// `google/diffusiongemma-26B-A4B-it`: the 26B-A4B mixture's trunk with
    /// its self-conditioning block.
    pub fn a4b(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model {
            trunk: gemma_4::model::Model::a4b_diffusion(w, kv, tp),
        }
    }

    /// The dense weights in `w`, the routed experts in `xw`.
    pub fn a4b_experts(w: Dtype, xw: Dtype, kv: Dtype, tp: u32) -> Model {
        Model {
            trunk: gemma_4::model::Model::a4b_diffusion_experts(w, xw, kv, tp),
        }
    }

    /// The dense weights in `w`, the routed experts in `xw`, the
    /// self-conditioning block in `sw`.
    pub fn a4b_experts_self_cond(w: Dtype, xw: Dtype, sw: Dtype, kv: Dtype, tp: u32) -> Model {
        Model {
            trunk: gemma_4::model::Model::a4b_diffusion_experts_self_cond(w, xw, sw, kv, tp),
        }
    }
}
