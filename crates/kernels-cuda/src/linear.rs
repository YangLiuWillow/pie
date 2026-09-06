//! `Linear`: everything that multiplies by a weight — dense projections,
//! gated activations over one, the routed fan-out, the low-rank correction
//! that rides on a materialised output, and the quantizers that stage the
//! banks they read. One submodule per member of the family; the entries
//! inside keep one entry per IR variant.

#[cfg(feature = "cuda")]
mod dense;

pub mod gemm;

/// The lane-axis projection over an f32 activation (a lane vector's chain).
pub mod lane_gemm;

#[cfg(feature = "cuda")]
mod gemv;
/// The skinny bf16 projection with its epilogue folded in.
pub mod skinny;

/// The correction class: `y += B[a]·(A[a]·x)` over a routed adapter bank.
pub mod lora;

pub mod mlp;

pub mod moe;

/// The router that reads no logits: `linear.moe_hash_route`, a per-token
/// LOOKUP where its neighbours score a gate. Its own module because its
/// device text is its own unit, for the reason that file states.
pub mod moe_route;

/// The learned relative-position profile (`Linear::RelBias`).
pub mod rel_bias;

pub mod fp8;

pub mod kquant;

pub mod nvfp4;

pub mod quant;

/// The tiled tensor-core reading of `quant`'s post-affine form (§J4).
pub mod tiled;
