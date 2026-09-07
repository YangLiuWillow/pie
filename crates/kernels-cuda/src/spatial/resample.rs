//! The voxel-axis reshapes: nearest upsample, pixel (un)shuffle, the
//! patchify pair the DiT boundary names them by, and the averaging
//! down-shuffle Wan 2.2's encoder shortcut is. Index arithmetic, one thread
//! per output element, any 16-bit element; only [`avg_down`] does any
//! arithmetic, and that is a mean in fp32.
//!
//! **CHANNEL ORDER.** The shuffles follow einops
//! `'b (c r1 r2 r3) t h w -> b c (t r1) (h r2) (w r3)'` — `torch.pixel_shuffle`
//! in two dimensions, and the transformer patchify
//! `'b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)'` inverted: within a
//! block the offsets `(i1, i2, i3)` are the fast index under the channel,
//! `c_in = c * r1*r2*r3 + (i1 * r2 + i2) * r3 + i3`. [`avg_down`] widens
//! the row the same way and then folds runs of it, so the two agree on
//! which elements a group holds.

use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated};
use crate::spatial::{flat_elements, lane_pair};
use crate::tensor::Tensor;

const FILE: &str = "spatial/resample.cuh";

const BLOCK: u32 = 256;

fn element(op: &'static str, dtype: dtype::Dtype) -> Result<&'static str, Error> {
    Ok(dtype_dispatch!(op, dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" }))
}

/// Nearest-neighbour upsample by `factor = [ft, fh, fw]`.
///
/// `x`: `[rows, C]`; `grid`/`o_grid`: the lane tables, the output box being
/// `(t_out, h*fh, w*fw)` with `t_out = keep_first_frame ? 1 + (t-1)*ft :
/// t*ft` — the causal video VAEs emit frame 0 once and every later frame
/// `ft` times. `o`: `[rows_out, C]` at `x`'s width and dtype.
#[allow(clippy::too_many_arguments)]
pub fn upsample_nearest(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    factor: [u32; 3],
    keep_first_frame: bool,
    o: &mut Tensor,
    o_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.upsample_nearest";
    let t = element(OP, x.dtype)?;
    debug_assert!(
        o.width == x.width && o.dtype == x.dtype,
        "`{OP}` keeps the channel width"
    );
    let lanes = lane_pair(OP, grid, o_grid)?;
    let c = count(OP, "the channel count", x.width)?;
    let [ft, fh, fw] = [
        count(OP, "the time factor", factor[0])?,
        count(OP, "the height factor", factor[1])?,
        count(OP, "the width factor", factor[2])?,
    ];
    let (blocks, total) = flat_elements(OP, *o, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            crate::jit::symbol(&format!("::pie::spatial::upsample_nearest<{t}>")),
        )
        .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            grid.arg(),
            o.arg(),
            o_grid.arg(),
            c.arg(),
            lanes.arg(),
            ft.arg(),
            fh.arg(),
            fw.arg(),
            i32::from(keep_first_frame).arg(),
            total.arg(),
        ],
    )
}

fn block_of(op: &'static str, r: [u32; 3]) -> Result<([i32; 3], u32), Error> {
    let r = [
        count(op, "r1", r[0])?,
        count(op, "r2", r[1])?,
        count(op, "r3", r[2])?,
    ];
    let volume = r[0]
        .unsigned_abs()
        .checked_mul(r[1].unsigned_abs())
        .and_then(|v| v.checked_mul(r[2].unsigned_abs()))
        .ok_or_else(|| refuse(op, "the block volume overflows"))?;
    Ok((r, volume))
}

/// Depth to space: `[rows, C*r1*r2*r3]` over `(t, h, w)` into
/// `[rows*r1*r2*r3, C]` over `(t*r1 - trim_t, h*r2, w*r3)`; `o_grid` states
/// the output boxes.
///
/// `trim_t` is a causal temporal upsampler's ANCHOR DROP: the leading
/// `trim_t` frames of the shuffled result are not emitted, so output frame
/// `i` reads shuffled frame `i + trim_t` (LTX-2.5's `LTXVideoUpsampler3d`
/// drops `r1 - 1`). `0` is the plain shuffle. `o` may over-allocate: only
/// the rows `o_grid` claims are written, the rest land zeros.
#[allow(clippy::too_many_arguments)]
pub fn pixel_shuffle(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    trim_t: u32,
    o: &mut Tensor,
    o_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.pixel_shuffle";
    let t = element(OP, x.dtype)?;
    debug_assert_eq!(o.dtype, x.dtype, "`{OP}` keeps the element");
    let lanes = lane_pair(OP, grid, o_grid)?;
    let (r, volume) = block_of(OP, r)?;
    if !x.width.is_multiple_of(volume) || o.width != x.width / volume {
        return Err(refuse(
            OP,
            format!(
                "{} channels do not unpack as {} x {volume} output channels",
                x.width, o.width
            ),
        ));
    }
    let c = count(OP, "the output channel count", o.width)?;
    let trim = stated(OP, trim_t)?;
    let (blocks, total) = flat_elements(OP, *o, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            crate::jit::symbol(&format!("::pie::spatial::pixel_shuffle<{t}>")),
        )
        .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            grid.arg(),
            o.arg(),
            o_grid.arg(),
            c.arg(),
            lanes.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
            trim.arg(),
            total.arg(),
        ],
    )
}

/// Space to depth, the inverse of [`pixel_shuffle`]: `[rows, C]` over
/// `(t, h, w)` into `[rows/(r1*r2*r3), C*r1*r2*r3]` over `(t/r1, h/r2,
/// w/r3)`. Every lane's box must divide by `r`.
pub fn pixel_unshuffle(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    o: &mut Tensor,
    o_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.pixel_unshuffle";
    let t = element(OP, x.dtype)?;
    debug_assert_eq!(o.dtype, x.dtype, "`{OP}` keeps the element");
    let lanes = lane_pair(OP, grid, o_grid)?;
    let (r, volume) = block_of(OP, r)?;
    if o.width != x.width.saturating_mul(volume) {
        return Err(refuse(
            OP,
            format!(
                "{} channels do not pack as {} = C x {volume} output channels",
                x.width, o.width
            ),
        ));
    }
    let c = count(OP, "the input channel count", x.width)?;
    let (blocks, total) = flat_elements(OP, *o, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            crate::jit::symbol(&format!("::pie::spatial::pixel_unshuffle<{t}>")),
        )
        .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            grid.arg(),
            o.arg(),
            o_grid.arg(),
            c.arg(),
            lanes.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
            total.arg(),
        ],
    )
}

/// `AvgDown3D`: the time axis zero-padded IN FRONT to a multiple of `r[0]`,
/// a channel-major space-to-depth by `r` ([`pixel_unshuffle`]'s ordering),
/// then the MEAN of each `group` consecutive widened channels.
///
/// `x`: `[rows, C]`; `o`: `[rows_out, C*r1*r2*r3/group]` at `x`'s dtype,
/// `o_grid` stating the `(ceil(t/r1), h/r2, w/r3)` boxes. `group == r1*r2*r3`
/// is the plain average pool over the block; `group == r2*r3` is a spatial
/// pool that keeps the time block as extra channels (Wan 2.2's every
/// shortcut). fp32 accumulation, one rounding at the store.
#[allow(clippy::too_many_arguments)]
pub fn avg_down(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    group: u32,
    o: &mut Tensor,
    o_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.avg_down";
    let t = element(OP, x.dtype)?;
    debug_assert_eq!(o.dtype, x.dtype, "`{OP}` keeps the element");
    let lanes = lane_pair(OP, grid, o_grid)?;
    let (r, volume) = block_of(OP, r)?;
    let widened = x.width.saturating_mul(volume);
    if group == 0 || !widened.is_multiple_of(group) || o.width != widened / group {
        return Err(refuse(
            OP,
            format!(
                "{} channels widen to {widened} and do not fold into {} groups of {group}",
                x.width, o.width
            ),
        ));
    }
    let c = count(OP, "the input channel count", x.width)?;
    let g = count(OP, "the group size", group)?;
    let (blocks, total) = flat_elements(OP, *o, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            crate::jit::symbol(&format!("::pie::spatial::avg_down<{t}>")),
        )
        .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            grid.arg(),
            o.arg(),
            o_grid.arg(),
            c.arg(),
            lanes.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
            g.arg(),
            total.arg(),
        ],
    )
}

/// `[rows, C]` voxels into `[tokens, C*pt*ph*pw]` patch tokens — the DiT
/// boundary's name for [`pixel_unshuffle`] by the patch `p`; `o_grid` is
/// the token box `(t/pt, h/ph, w/pw)` per lane.
pub fn patchify(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    p: [u32; 3],
    o: &mut Tensor,
    o_grid: Tensor,
) -> Result<(), Error> {
    pixel_unshuffle(ctx, x, grid, p, o, o_grid)
}

/// `[tokens, C*pt*ph*pw]` back to `[rows, C]` voxels — [`pixel_shuffle`]
/// by the patch `p`.
pub fn unpatchify(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    p: [u32; 3],
    o: &mut Tensor,
    o_grid: Tensor,
) -> Result<(), Error> {
    // A patch grid never drops a frame: the DiT boundary is a reshape.
    pixel_shuffle(ctx, x, grid, p, 0, o, o_grid)
}
