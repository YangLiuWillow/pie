//! `RelativeBucketBias`: the dense relative-position bias table a
//! bidirectional encoder layer's attention adds to its logits, computed on
//! the device from the layer's bucket embedding. Its own file beside
//! `sinusoid` because, like it, the entry reads no activation rectangle —
//! one small weight in, one `[heads, 2·max_len − 1]` f32 table out — and its
//! geometry is a constant of the plan, not of the fire.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/relative_bucket_bias.cuh";

const BLOCK: u32 = 256;

/// The relative bias table `y[h][d + max_len − 1] = embedding[bucket(d)][h]`
/// for every signed distance `d` in `−(max_len − 1) ..= max_len − 1`, where
/// `bucket` is the T5 relative-position bucket function — Hugging Face's
/// `T5Attention._relative_position_bucket(d, bidirectional, num_buckets,
/// max_distance)` with `d = memory_position − context_position`
/// (`kj − qi`), transcribed in `kernels/elemwise/relative_bucket_bias.cuh`:
///
/// ```text
/// bucket = 0
/// if bidirectional: num_buckets /= 2; bucket += (d > 0) · num_buckets; n = |d|
/// else:             n = −min(d, 0)
/// max_exact = num_buckets / 2
/// if n < max_exact: bucket + n
/// else: bucket + min(num_buckets − 1,
///          max_exact + trunc(ln(n/max_exact) / ln(max_distance/max_exact) · (num_buckets − max_exact)))
/// ```
///
/// `embedding` is the checkpoint's `[num_buckets, heads]` plane as stored
/// (`relative_attention_bias.weight`), bf16 or f32; `y` is `[heads,
/// 2·max_len − 1]` f32, the layout `attention.ragged`'s `RelativeBias` arm
/// reads. The logarithm's ratio is computed in f32 as torch computes it, so
/// the table agrees with the reference model's bucket by bucket.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for an embedding that is not bf16 or f32; a
/// refusal for a table that is not `[heads, 2·max_len − 1]` f32, an
/// embedding with fewer than `num_buckets` rows or fewer than `heads`
/// columns, a bucket count too small to hold an exact band (`num_buckets`
/// below 4, or below 2 one-directional), or a `max_distance` at or below
/// `max_exact`, which leaves the logarithm's ratio with no answer.
pub fn relative_bucket_bias(
    ctx: &Ctx,
    embedding: Tensor,
    max_len: u32,
    num_buckets: u32,
    max_distance: f32,
    bidirectional: bool,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.relative_bucket_bias";
    let t = dtype_dispatch!(OP, embedding.dtype, { Bf16 => "::pie::bf16", F32 => "float" });
    let heads = nonzero(OP, "heads", y.rows)?;
    nonzero(OP, "max_len", max_len)?;
    let span = 2 * max_len - 1;
    if y.dtype != Dtype::F32 || y.width != span {
        return Err(refuse(
            OP,
            format!(
                "the table is {} x {} {:?}; this entry writes one f32 row of {span} \
                 (2 · {max_len} − 1) per head",
                y.rows, y.width, y.dtype
            ),
        ));
    }
    if embedding.rows < num_buckets || embedding.width < heads {
        return Err(refuse(
            OP,
            format!(
                "the bucket embedding is {} x {}, and the table reads {num_buckets} buckets of \
                 {heads} heads",
                embedding.rows, embedding.width
            ),
        ));
    }
    let directional_buckets = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };
    let max_exact = directional_buckets / 2;
    if max_exact == 0 {
        return Err(refuse(
            OP,
            format!("{num_buckets} buckets leave no exact band; the function needs at least 4"),
        ));
    }
    // torch: `math.log(max_distance / max_exact)` (a Python double) divides
    // an f32 tensor, so it is cast to f32 at the divide — computed here on
    // the host in f64 and cast once, exactly as the reference does.
    let ratio = f64::from(max_distance) / f64::from(max_exact);
    if ratio.is_nan() || ratio <= 1.0 || !ratio.is_finite() {
        return Err(refuse(
            OP,
            format!(
                "max_distance {max_distance} is at or below max_exact {max_exact}; the large \
                 band's logarithm has no answer"
            ),
        ));
    }
    let log_ratio = ratio.ln() as f32;
    let cells = u64::from(heads) * u64::from(span);
    let lanes = u32::try_from(cells).map_err(|_| {
        refuse(
            OP,
            format!("{cells} table cells do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::relative_bucket_bias<{t}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            embedding.arg(),
            y.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, span)?.arg(),
            stated(OP, max_len)?.arg(),
            stated(OP, embedding.width)?.arg(),
            stated(OP, num_buckets)?.arg(),
            i32::from(bidirectional).arg(),
            log_ratio.arg(),
        ],
    )
}
