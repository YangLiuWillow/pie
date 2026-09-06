//! `Attention`: the conv VAE's mid-block attention — one head as wide as
//! the row, per lane over every voxel of the lane: `y = softmax(q·kᵀ ·
//! sm_scale) · v` with `q`, `k`, `v`, `y` all `[rows, C]` bf16 on the voxel
//! axis and the segments read off the lane table. A plain online-softmax
//! walk over the lane's keys (`spatial/attn.cuh`), not the ragged FA2
//! template, which is stamped at head widths 64/128/256 while a VAE's head
//! is its whole channel row.
//!
//! Numerics: fp32 scores (the logits pre-scaled by `sm_scale · log2 e` and
//! exponentiated with `exp2`), fp32 running max and sum, fp32 accumulation
//! over the values, one rounding at the store — the reference's
//! `upcast_softmax` attention, off by the summation order alone.

use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, aligned16, count, dtype_dispatch, refuse, stated};
use crate::spatial::lanes_of;
use crate::tensor::Tensor;

const FILE: &str = "spatial/attn.cuh";

const OP: &str = "spatial.attention";

/// Queries per warp; four warps a block.
const QPW: u32 = 4;

const BLOCK: u32 = 128;

/// The row widths the kernel is stamped at: every lane of a warp holds
/// `C / 32` channels as whole 16-byte words.
const WIDTHS: [u32; 3] = [256, 512, 1024];

/// `y = softmax(q·kᵀ · sm_scale) · v` per lane of `grid`.
///
/// `q`, `k`, `v`: `[rows, C]` bf16 at one shape, `C` one of 256/512/1024,
/// 16-byte aligned; `grid`: `[lanes, 4]` i32; `o`: `[rows, C]` bf16. Rows
/// no lane claims land zeros.
pub fn attention(
    ctx: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grid: Tensor,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    for (what, t) in [("key", k), ("value", v), ("output", *o)] {
        if t.rows != q.rows || t.width != q.width || t.dtype != q.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} is {}x{} {:?}; the query is {}x{} {:?}, and all four rectangles share one shape",
                    t.rows, t.width, t.dtype, q.rows, q.width, q.dtype
                ),
            ));
        }
    }
    if !WIDTHS.contains(&q.width) {
        return Err(refuse(
            OP,
            format!(
                "no attention unit is stamped at row width {}; the arm holds {WIDTHS:?}",
                q.width
            ),
        ));
    }
    if !(aligned16(q.ptr) && aligned16(k.ptr) && aligned16(v.ptr) && aligned16(o.ptr)) {
        return Err(refuse(OP, "the rectangles must be 16-byte aligned"));
    }
    let lanes = lanes_of(OP, "input", grid)?;
    let rows = count(OP, "rows", q.rows)?;
    let per_block = stated(OP, BLOCK / 32 * QPW)?;
    let blocks = q.rows.div_ceil(per_block.unsigned_abs());
    let scale_log2 = sm_scale * core::f32::consts::LOG2_E;
    let entry = crate::jit::symbol(&format!("::pie::spatial::attention<{}, {QPW}>", q.width));
    ctx.fire(
        OP,
        Fire::at(FILE, entry).apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            grid.arg(),
            o.arg(),
            lanes.arg(),
            rows.arg(),
            scale_log2.arg(),
        ],
    )
}
