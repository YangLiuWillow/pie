//! The `Collective` family: the cross-rank reductions and gathers.

use super::*;

pub fn all_reduce(buf: &Value) -> Value {
    let r = buf.rec();
    let buf_out = r.fresh(buf.ty().clone());
    r.push(
        Collective::AllReduce {
            buf: buf.id(),
            buf_out: buf_out.id(),
        },
        &[buf],
    );
    buf_out
}

/// Concatenates each rank's `width`-shard into the full tensor:
/// `[rows, width]` on every rank becomes `[rows, width * world]`.
///
/// **ONE ROW ONLY.** The entry behind this is `ncclAllGather`, which joins
/// whole buffers rank-major, and that is the same layout as a width concat
/// only when `rows == 1`. A wider value is refused at the fire
/// (`kernels_cuda::collective::all_gather`) rather than silently transposed.
/// Gathering wider wants a permute after the collective, which nothing builds
/// yet — see that entry before reaching for this on a batched fire.
pub fn all_gather(x: &Value, world: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(x.rows(), x.width() * u64::from(world), x.dtype()));
    r.push(
        Collective::AllGather {
            x: x.id(),
            y: y.id(),
        },
        &[x],
    );
    y
}

/// Sums across ranks, leaving each rank its `width`-shard of the result:
/// `[rows, width]` becomes `[rows, width / world]`.
///
/// **ONE ROW ONLY**, mirroring [`all_gather`]: `ncclReduceScatter` hands each
/// rank a contiguous block of the flat buffer, which is a width shard only at
/// one row. Refused at the fire for a wider value.
pub fn reduce_scatter(x: &Value, world: u32) -> Value {
    let world = u64::from(world);
    assert!(
        x.width().is_multiple_of(world),
        "a width of {} does not scatter {world} ways",
        x.width(),
    );
    let r = x.rec();
    let y = r.fresh(tensor(x.rows(), x.width() / world, x.dtype()));
    r.push(
        Collective::ReduceScatter {
            x: x.id(),
            y: y.id(),
        },
        &[x],
    );
    y
}
