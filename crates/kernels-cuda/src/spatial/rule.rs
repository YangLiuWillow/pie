//! `spatial.grid`: the output grid of one op, computed on the device from
//! its input grid and a rule — one block, one thread, a serial walk over
//! the clips with the row offsets prefix-summed. What every wrapper that
//! changes the box launches ahead of itself, so a chunked decode's grids
//! are graph-captured data rather than host state.

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, stated};
use crate::spatial::lane_pair;
use crate::tensor::Tensor;

const FILE: &str = "spatial/rule.cuh";

const OP: &str = "spatial.grid";

/// How one box maps to the next — the host's spelling of `RuleGeom`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridRule {
    /// `(n + front + back - k) / stride + 1` per axis, time padded in front
    /// only under `causal_t`.
    Conv {
        k: [u32; 3],
        stride: [u32; 3],
        pad: [u32; 3],
        causal_t: bool,
    },
    /// `(t·ft, h·fh, w·fw)`, or `1 + (t-1)·ft` frames under `keep_first_frame`.
    Upsample {
        factor: [u32; 3],
        keep_first_frame: bool,
    },
    /// `(t·r1, h·r2, w·r3)`.
    Shuffle { r: [u32; 3] },
    /// `(t/r1, h/r2, w/r3)`; a box that does not divide lands no rows.
    Unshuffle { r: [u32; 3] },
}

/// `RuleGeom` in `spatial/rule.cuh`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Geom {
    kind: i32,
    a: [i32; 3],
    b: [i32; 3],
    c: [i32; 3],
    flag: i32,
    lanes: i32,
}

fn triple(op: &'static str, v: [u32; 3]) -> Result<[i32; 3], Error> {
    Ok([stated(op, v[0])?, stated(op, v[1])?, stated(op, v[2])?])
}

/// `o_grid = rule(grid)`: both `[lanes, 4]` i32.
///
/// # Errors
///
/// A refusal for a table that is not `[lanes, 4]` i32 or two tables of
/// different lane counts.
pub fn derive_grid(
    ctx: &Ctx,
    grid: Tensor,
    rule: GridRule,
    o_grid: &mut Tensor,
) -> Result<(), Error> {
    let lanes = lane_pair(OP, grid, *o_grid)?;
    let geom = match rule {
        GridRule::Conv {
            k,
            stride,
            pad,
            causal_t,
        } => Geom {
            kind: 0,
            a: triple(OP, k)?,
            b: triple(OP, stride)?,
            c: triple(OP, pad)?,
            flag: i32::from(causal_t),
            lanes,
        },
        GridRule::Upsample {
            factor,
            keep_first_frame,
        } => Geom {
            kind: 1,
            a: triple(OP, factor)?,
            b: [0; 3],
            c: [0; 3],
            flag: i32::from(keep_first_frame),
            lanes,
        },
        GridRule::Shuffle { r } => Geom {
            kind: 2,
            a: triple(OP, r)?,
            b: [0; 3],
            c: [0; 3],
            flag: 0,
            lanes,
        },
        GridRule::Unshuffle { r } => Geom {
            kind: 3,
            a: triple(OP, r)?,
            b: [0; 3],
            c: [0; 3],
            flag: 0,
            lanes,
        },
    };
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::spatial::grid_rule").apply(Launch::grid([1, 1, 1], [32, 1, 1])),
        &[
            grid.arg(),
            o_grid.arg(),
            ArgValue::Bytes {
                ptr: std::ptr::from_ref(&geom).cast(),
                len: size_of::<Geom>(),
            },
        ],
    )
}
