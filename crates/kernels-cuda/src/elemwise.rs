//! `Elementwise`: the per-element passes — norms and the residual folds,
//! rotary position, the sigmoid gates, and the hyper-connection mixers.
//! One submodule per member of the family; the entries inside keep one
//! entry per IR variant.

/// The bare, ungated activations, from one plane into another.
pub mod activation;

/// `x + y` and `x · y` over two rectangles of one shape.
pub mod binary;

/// The clipped linears' clamp — its own member of the family because the
/// sites it serves are projections and not a fused activation.
pub mod clip;

pub mod gate;

pub mod hc;

/// The centred norm: `norm`'s reductions plus the mean subtraction, and no
/// weight at all.
pub mod layernorm;

/// adaLN's three modulations, the gated residual write, and the two forms
/// that fold a norm into the first of them.
pub mod modulate;

pub mod norm;

pub mod rope;

/// The multi-axis rotary — `rope_mrope`'s statute generalised: up to four
/// axes, each with its own theta and its own ladder, over f32 positions that
/// may be fractional.
pub mod rope_axes;

/// The multimodal rotary — `rope`'s partial arm over an `(t, h, w)` triple.
/// Its own member of the family because it reads its own position stream
/// under its own statute, not because the rotation differs.
pub mod rope_mrope;

/// The dense relative-position bias table a bidirectional encoder's
/// attention adds to its logits, from the layer's bucket embedding.
pub mod relative_bucket_bias;

/// The sinusoidal timestep embedding, the one denoise input that is
/// arithmetic rather than an activation.
pub mod sinusoid;

// The entries a caller spells as the family's own verb
// (`elemwise::rope_axes(..)`, `elemwise::sinusoid(..)`,
// `elemwise::relative_bucket_bias(..)`), since each file carries exactly one.
pub use relative_bucket_bias::relative_bucket_bias;
pub use rope_axes::{RopeForm, rope_axes};
pub use sinusoid::sinusoid;
