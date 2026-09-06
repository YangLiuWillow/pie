//! `RopeAxes`: the multi-axis rotary — up to four axes over one head, each
//! owning a contiguous block of rotary channels with its own theta and its
//! own full frequency ladder, from f32 positions that may be fractional.
//!
//! A file of its own beside `rope_mrope.rs` for the same reason that one sits
//! beside `rope.rs`: a different position stream (`[rows, axes]` f32, not
//! `[rows, 3]` i32) under a different statute (per-axis theta, per-axis
//! ladder, three pairings), not a differently-shaped rotation.

use crate::error::Error;
use dtype::Dtype;

use crate::elemwise::rope::ROTATE_BLOCK;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/rope_axes.cuh";

/// The most axes one rotation may carry (FLUX.2's four).
pub const MAX_AXES: usize = 4;

/// Which two channels an angle rotates. The form is the PAIRING and nothing
/// else — all three hand the same angle to the same axis.
///
/// **Not to be confused with `MropeForm::Interleaved`**, which names a
/// SECTION layout under rotate-half pairing. [`Interleaved`](RopeForm::Interleaved)
/// here is `rope_full`'s `interleaved` flag and sglang's `is_neox=False`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeForm {
    /// GPT-J: angle `i` of an axis' block turns `(x[b + 2i], x[b + 2i + 1])`.
    /// Z-Image's `view_as_complex` pairs.
    Interleaved,
    /// Rotate-half across the whole rotary span: angle `p` turns
    /// `(x[p], x[p + rotary_dim/2])`. Every `cat([freqs, freqs], -1)`
    /// reference; MiniMax H3.
    Neox,
    /// Rotate-half WITHIN the axis' own block of `2s` channels: angle `i`
    /// turns `(x[b + i], x[b + s + i])`. `MropeForm::Split` — Gemma's tower,
    /// LTX's `[x1 | x2]` halves.
    Split,
}

impl RopeForm {
    const fn stamp(self) -> &'static str {
        match self {
            RopeForm::Interleaved => "0",
            RopeForm::Neox => "1",
            RopeForm::Split => "2",
        }
    }
}

/// `o = rope(x)` over every head of the row: axis `a` owns `dims[a]`
/// contiguous rotary channels (in axis order, `Σ dims = rotary_dim`) and
/// turns its `i`-th angle at `thetas[a]^(−2i/dims[a])`; channels
/// `[rotary_dim, head_dim)` pass through. `positions` is `[rows, axes]` f32,
/// one coordinate per axis per row, and may be fractional. `o` may alias `x`.
///
/// The row is `width / head_dim` heads wide and every head is turned by the
/// same angles, so one call serves a whole q or k rectangle. Frequencies and
/// the rotation are f32 (`powf`, `__sincosf`, `rope.cuh`'s own), with one
/// rounding at the store.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for a
/// row that is not a whole number of heads, a rotated prefix wider than the
/// head, an axis whose block is odd or empty, blocks that do not tile the
/// rotated prefix, or a position stream that is not `[rows, axes]` f32.
#[allow(clippy::too_many_arguments)]
pub fn rope_axes(
    ctx: &Ctx,
    x: Tensor,
    positions: Tensor,
    dims: [u32; MAX_AXES],
    thetas: [f32; MAX_AXES],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_axes";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        x.rows == o.rows && x.width == o.width && x.dtype == o.dtype,
        "`{OP}` writes the rectangle it reads"
    );
    let rows = nonzero(OP, "rows", o.rows)?;
    nonzero(OP, "the head width this rotation states", head_dim)?;
    if !head_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the rotated prefix is {rotary_dim} wide, and the head it sits at the front \
                 of is {head_dim}"
            ),
        ));
    }
    if o.width == 0 || !o.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide row is not a whole number of {head_dim}-wide heads",
                o.width
            ),
        ));
    }
    let heads = o.width / head_dim;

    // The axis count is the position stream's own width: a rotation over
    // three axes reads three coordinates, and the fourth slot of `dims` is
    // then not an axis but an unused word.
    if positions.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {:?}, and this rotation reads f32 coordinates \
                 (they may be fractional)",
                positions.dtype
            ),
        ));
    }
    let axes = positions.width as usize;
    if axes == 0 || axes > MAX_AXES || positions.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {} x {}, and this rotation reads one coordinate per \
                 axis (at most {MAX_AXES}) for each of {rows} rows",
                positions.rows, positions.width
            ),
        ));
    }
    let mut spanned = 0u32;
    for (a, &d) in dims.iter().enumerate() {
        if a >= axes {
            if d != 0 {
                return Err(refuse(
                    OP,
                    format!(
                        "axis {a} owns {d} channels and the position stream carries {axes} \
                         axes; an axis with no coordinate is a text to fix"
                    ),
                ));
            }
            continue;
        }
        if d == 0 || !d.is_multiple_of(2) {
            return Err(refuse(
                OP,
                format!("axis {a} owns {d} channels, and an axis turns whole pairs"),
            ));
        }
        spanned += d;
    }
    if spanned != rotary_dim {
        return Err(refuse(
            OP,
            format!(
                "the axes own {spanned} channels between them and the rotated prefix is \
                 {rotary_dim} wide; the blocks tile it in axis order"
            ),
        ));
    }

    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::rope_axes<{t}, {}>",
                form.stamp()
            )),
        )
        .apply(Launch::per_row(rows, ROTATE_BLOCK)),
        &[
            x.arg(),
            positions.arg(),
            o.arg(),
            stated(OP, positions.width)?.arg(),
            stated(OP, dims[0])?.arg(),
            stated(OP, dims[1])?.arg(),
            stated(OP, dims[2])?.arg(),
            stated(OP, dims[3])?.arg(),
            thetas[0].arg(),
            thetas[1].arg(),
            thetas[2].arg(),
            thetas[3].arg(),
            stated(OP, rotary_dim)?.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, heads)?.arg(),
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}
