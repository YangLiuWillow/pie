//! `Activation`: the bare, ungated activations — `silu`, `gelu_tanh`,
//! `tanh` — from one plane into another.
//!
//! The gated forms already exist inside the `Mlp*` entries, and
//! `norm::silu_scaled` is `silu` in place on one plane with a scale in front.
//! What a DiT graph needs and could not spell is the two-plane shape: the
//! adaLN MLP's `silu` between two linears, and the `tanh` a gate embedding
//! passes through.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/pointwise.cuh";

const BLOCK: u32 = 256;

/// `o = x · sigmoid(x)`. `o` may alias `x`.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16, f16 and f32 (the last
/// for a lane vector's chain, which stays f32); a refusal for operands that
/// do not share one shape, an empty rectangle, or an extent past a 32-bit
/// launch.
pub fn silu(ctx: &Ctx, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire(ctx, "elementwise.silu", "0", x, o)
}

/// `o = tanh(x)`. `o` may alias `x`.
///
/// # Errors
///
/// As [`silu`].
pub fn tanh(ctx: &Ctx, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire(ctx, "elementwise.tanh", "1", x, o)
}

/// `o = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. `o` may alias `x`.
///
/// **This entry is the one next door**: `linear::mlp::gelu_tanh` is already
/// the ungated two-plane gelu, seated and stamped, so this names it rather
/// than transcribing the polynomial a third time. The op the seat sees is
/// `linear.mlp_gelu_tanh`.
///
/// # Errors
///
/// As `linear::mlp::gelu_tanh`.
pub fn gelu_tanh(ctx: &Ctx, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    crate::linear::mlp::gelu_tanh(ctx, x, 1, o)
}

fn fire(ctx: &Ctx, op: &'static str, stamp: &str, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    let t = dtype_dispatch!(op, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    if x.dtype != o.dtype || x.rows < o.rows || x.width != o.width {
        return Err(refuse(
            op,
            format!(
                "the operand is {} x {} {:?} and the answer is {} x {} {:?}; an activation \
                 does not reshape",
                x.rows, x.width, x.dtype, o.rows, o.width, o.dtype
            ),
        ));
    }
    let n = o.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(op, "the element count", lanes)?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::activation<{t}, {stamp}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            o.arg(),
            stated(op, lanes)?.arg(),
            // Element-form seat's width: the launch is flat over rows*width,
            // so the kernel reads the staged row count and start as elements.
            stated(op, o.width)?.arg(),
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}
