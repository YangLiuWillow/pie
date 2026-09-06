//! `linear.matmul` over a LANE-shaped, f32 activation: the projection a lane
//! vector's chain runs — the timestep embedding into its adaLN modulation
//! (design D6) — for which the tensor-core gemm's bf16 activation contract
//! does not hold and the row count is the request count, not the token
//! count.
//!
//! **Numerics.** `y = act · w^T` with the activation read in its own element
//! (f32 or bf16), the weight in bf16, every product accumulated in f32 by
//! `fmaf` in `k` order within a lane and reduced across the warp, and one
//! rounding at the store into `y`'s element. A host reference that
//! accumulates in f32 agrees to accumulation order.
//!
//! **Not a tensor-core path.** A warp per output element is right for a few
//! hundred rows; a DiT with thousands of modulated rows per fire is served by
//! `gemm::act_x_wt` over a bf16 activation. The engine routes on the
//! activation's element.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/lane_gemm.cuh";

/// Eight warps: eight columns per block.
const BLOCK: u32 = 256;

fn element(op: &'static str, what: &str, dtype: Dtype) -> Result<&'static str, Error> {
    match dtype {
        Dtype::F32 => Ok("float"),
        Dtype::Bf16 => Ok("::pie::bf16"),
        Dtype::F16 => Ok("::pie::f16"),
        other => Err(refuse(
            op,
            format!("the {what} is {other:?}, and this projection reads f32, bf16 or f16"),
        )),
    }
}

/// `y = act x w^T` for a lane-shaped activation: `act` `[rows, k]` f32 (or
/// bf16), `w` `[n, k]` bf16 row-major, `y` `[rows, n]` in the activation's
/// element. A zero-row launch is a no-op, as on the dense arm.
///
/// # Errors
///
/// A refusal for an element outside f32/bf16/f16, a weight that does not
/// contract over the activation's width, or a result that is not `[rows, n]`.
pub fn act_x_wt(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    let ta = element(op, "activation", act.dtype)?;
    let tw = element(op, "weight", w.dtype)?;
    let ty = element(op, "result", y.dtype)?;
    if y.rows == 0 {
        return Ok(());
    }
    let n = nonzero(op, "the columns this projection lands", y.width)?;
    let k = nonzero(op, "the contraction this projection walks", act.width)?;
    if w.width != k || w.rows != n {
        return Err(refuse(
            op,
            format!(
                "the weight is {} x {} and this projection contracts {k} into {n}",
                w.rows, w.width
            ),
        ));
    }
    if act.rows < y.rows {
        return Err(refuse(
            op,
            format!(
                "the activation has {} rows and the result asks for {}",
                act.rows, y.rows
            ),
        ));
    }
    let warps = BLOCK / 32;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::lane_gemm<{ta}, {tw}, {ty}>")),
        )
        .apply(Launch::grid([n.div_ceil(warps), y.rows, 1], [BLOCK, 1, 1])),
        &[
            act.arg(),
            w.arg(),
            y.arg(),
            stated(op, y.rows)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )
}
