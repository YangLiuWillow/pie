//! `kquant`: the ggml K-quant super-block decode-in-dot, the Metal twin of
//! `kernels_cuda::linear::kquant`. A K-quant weight reaches here as one `U8`
//! byte rectangle (`engine_metal::Run::maybe_stored` re-badges the stored
//! `Dense` handle); the scales live inside each super-block and are decoded in
//! the dot rather than off a companion plane, which is what lets pie serve a
//! GGUF at the size it shipped instead of inflating it to the activation dtype
//! at import.
//!
//! PR1 lands the plumbing only — the entry points exist and dispatch reaches
//! them, but the shader is PR2. Until then a K-quant projection refuses at
//! dispatch (a clear `Unsupported`) rather than serving wrong math. The decode
//! math is diffed bit-exact against the `checkpoint` host reference decoder
//! (`decode_gguf_q{2,3,4,5,6}_k_block_into`) and the CUDA kernel it mirrors.

use crate::encode::Ctx;
use crate::error::Error;
use crate::tensor::Tensor;

/// `y = act x w^T` where `w` is a stored K-quant super-block rectangle.
pub fn matmul(ctx: &Ctx<'_>, act: Tensor, block: Tensor, y: Tensor) -> Result<(), Error> {
    decode_in_dot(ctx, "linear.kquant.matmul", act, block, y)
}

/// The head's stored-block arm; a Q4_K_M mix stores `output.weight` at q6_k.
pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, block: Tensor, y: Tensor) -> Result<(), Error> {
    decode_in_dot(ctx, "linear.kquant.lm_head", act, block, y)
}

/// The shared decode-in-dot entry. PR2 stamps the per-scheme K-quant point off
/// `block.dtype`; PR1 refuses so a plumbed-but-unimplemented path never returns
/// silent wrong values.
fn decode_in_dot(
    _ctx: &Ctx<'_>,
    op: &'static str,
    _act: Tensor,
    _block: Tensor,
    _y: Tensor,
) -> Result<(), Error> {
    Err(Error::Unsupported { op })
}
