//! `Attention`: the conv VAE's mid-block attention — one head as wide as
//! the row, over the BLOCK a [`Segment`] names: `y = softmax(q·kᵀ ·
//! sm_scale) · v` with `q`, `k`, `v`, `y` all `[rows, C]` bf16 on the voxel
//! axis and the blocks read off the lane table. A plain online-softmax
//! walk over the block's keys (`spatial/attn.cuh`), not the ragged FA2
//! template, which is stamped at head widths 64/128/256 while a VAE's head
//! is its whole channel row.
//!
//! A block is a whole lane (the image VAEs' mid block) or a run of frames
//! inside one ([`Segment::Frames`]; Wan 2.2's mid block attends one frame
//! at a time). Nothing here knows which family asked.
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

const BLOCK: u32 = 128;

/// The row widths the kernel is stamped at, each with the QUERIES PER WARP
/// it is stamped with: a warp lane holds `C / 32` channels of every one of
/// its queries twice over (the pre-scaled query and the accumulator), so
/// `QPW · C / 32` is the register state either side and the product is held
/// at 32 — four queries a warp at 256 channels, one at 1024. A wider group
/// at 1024 spills to local memory and runs several times slower.
const WIDTHS: [(u32, u32); 3] = [(256, 4), (512, 2), (1024, 1)];

/// Which rows of a lane one query attends — the kernel's half of
/// `model_ir::VoxelSegment`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    /// Every voxel of the lane: one attention block per clip.
    #[default]
    Lane,
    /// The run of `n` consecutive frames the query's own frame falls in;
    /// a lane whose frame count is not a multiple leaves a short run at the
    /// end. `Frames(1)` is one block per frame.
    Frames(u32),
}

impl Segment {
    /// What the kernel takes: `0` for the whole lane, `n` for a run of `n`
    /// frames.
    fn frames(self) -> u32 {
        match self {
            Segment::Lane => 0,
            Segment::Frames(n) => n,
        }
    }
}

/// `y = softmax(q·kᵀ · sm_scale) · v` per block of `grid`, the block being
/// a whole lane or a run of its frames ([`Segment`]).
///
/// `q`, `k`, `v`: `[rows, C]` bf16 at one shape, `C` one of 256/512/1024,
/// 16-byte aligned; `grid`: `[lanes, 4]` i32; `o`: `[rows, C]` bf16. Rows
/// no lane claims land zeros.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grid: Tensor,
    segment: Segment,
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
    let Some(&(_, qpw)) = WIDTHS.iter().find(|(width, _)| *width == q.width) else {
        return Err(refuse(
            OP,
            format!(
                "no attention unit is stamped at row width {}; the arm holds {:?}",
                q.width,
                WIDTHS.map(|(width, _)| width)
            ),
        ));
    };
    if segment == Segment::Frames(0) {
        return Err(refuse(
            OP,
            "a block of zero frames holds no keys; state `Segment::Lane` for the whole clip",
        ));
    }
    if !(aligned16(q.ptr) && aligned16(k.ptr) && aligned16(v.ptr) && aligned16(o.ptr)) {
        return Err(refuse(OP, "the rectangles must be 16-byte aligned"));
    }
    let lanes = lanes_of(OP, "input", grid)?;
    let rows = count(OP, "rows", q.rows)?;
    let seg = stated(OP, segment.frames())?;
    let per_block = stated(OP, BLOCK / 32 * qpw)?;
    let blocks = q.rows.div_ceil(per_block.unsigned_abs());
    let scale_log2 = sm_scale * core::f32::consts::LOG2_E;
    let entry = crate::jit::symbol(&format!("::pie::spatial::attention<{}, {qpw}>", q.width));
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
            seg.arg(),
            scale_log2.arg(),
        ],
    )
}
