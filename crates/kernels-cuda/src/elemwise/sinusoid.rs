//! `Sinusoid`: the timestep embedding, the one input of a denoise pass that
//! is arithmetic rather than an activation. Its own file beside the norms
//! because it reads no rectangle at all — one f32 per row in, a `[rows, dim]`
//! f32 table out.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/sinusoid.cuh";

const BLOCK: u32 = 256;

/// `diffusers.get_timestep_embedding`, transcribed: `half = dim/2`
/// frequencies `exp(−ln(max_period)·i/half)`, angle `scale·(t·freq)`, row
/// `[sin | cos]` or `[cos | sin]` when `flip_sin_cos`, and a zero last column
/// when `dim` is odd.
///
/// The denominator is `half`, which is that function's
/// `downscale_freq_shift = 0` — the value the DiTs pass and the openai
/// original's. f32 in, f32 out, accurate `expf`/`sincosf`: the row feeds the
/// adaLN MLP whose product multiplies every activation in the block, and it
/// costs one row per lane per step.
///
/// # Errors
///
/// A refusal for a `t` stream that is not one f32 per row, an output that is
/// not `[rows, dim]` f32, a zero-row or zero-width rectangle, or a
/// `max_period` a logarithm has no answer for.
pub fn sinusoid(
    ctx: &Ctx,
    t: Tensor,
    dim: u32,
    max_period: f32,
    flip_sin_cos: bool,
    scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.sinusoid";
    let rows = nonzero(OP, "rows", o.rows)?;
    nonzero(OP, "the embedding width", dim)?;
    for (what, plane) in [("timestep", t), ("embedding", *o)] {
        if plane.dtype != Dtype::F32 {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is {:?}, and this embedding is computed and \
                     published in f32",
                    plane.dtype
                ),
            ));
        }
    }
    if t.width != 1 || t.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "the timestep stream is {} x {}, and this embedding reads one scalar for \
                 each of {rows} rows",
                t.rows, t.width
            ),
        ));
    }
    if o.width != dim {
        return Err(refuse(
            OP,
            format!(
                "the destination is {} wide and the embedding is {dim}",
                o.width
            ),
        ));
    }
    if max_period <= 0.0 || !max_period.is_finite() {
        return Err(refuse(
            OP,
            format!("the period is {max_period}, and its logarithm sets the ladder"),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::elemwise::sinusoid").apply(Launch::per_row(rows, BLOCK)),
        &[
            t.arg(),
            o.arg(),
            stated(OP, dim)?.arg(),
            max_period.arg(),
            i32::from(flip_sin_cos).arg(),
            scale.arg(),
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}
