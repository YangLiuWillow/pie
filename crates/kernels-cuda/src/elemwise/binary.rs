//! `Binary`: `x + y` and `x · y` over two rectangles of one shape.
//!
//! The trunks never needed either — every add there is a residual fold in
//! place on the stream (`norm::residual_add`) and every product is half of a
//! gated MLP. A DiT graph sums two conditioning streams and multiplies a gate
//! it did not compute in the same launch, so the plain three-operand shapes
//! live here.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/pointwise.cuh";

const BLOCK: u32 = 256;

/// `o = x + y`, per element. `o` may alias either input.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16, f16 and f32 (the last
/// for a lane vector's chain, which stays f32); a refusal for operands that
/// do not share one shape, an empty rectangle, or an extent past a 32-bit
/// launch.
pub fn add(ctx: &Ctx, x: Tensor, y: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire(ctx, "elementwise.add", "0", x, y, o)
}

/// `o = x · y`, per element. `o` may alias either input.
///
/// # Errors
///
/// As [`add`].
pub fn mul(ctx: &Ctx, x: Tensor, y: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire(ctx, "elementwise.mul", "1", x, y, o)
}

fn fire(
    ctx: &Ctx,
    op: &'static str,
    stamp: &str,
    x: Tensor,
    y: Tensor,
    o: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    for (what, plane) in [("left", x), ("right", y)] {
        if plane.dtype != o.dtype || plane.rows < o.rows || plane.width != o.width {
            return Err(refuse(
                op,
                format!(
                    "the {what} operand is {} x {} {:?} and the answer is {} x {} {:?}; \
                     this op does not broadcast",
                    plane.rows, plane.width, plane.dtype, o.rows, o.width, o.dtype
                ),
            ));
        }
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
            symbol(&format!("::pie::elemwise::binary<{t}, {stamp}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            y.arg(),
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
