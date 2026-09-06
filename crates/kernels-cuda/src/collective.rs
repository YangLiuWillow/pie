//! `Collective`: the tensor-parallel collectives, NCCL on the stream. One
//! entry per IR variant; every rank traces the same plan, so a collective
//! here is a sync point of an SPMD fire, enqueued like any launch and never
//! synchronised on.
//!
//! Communicator setup (`ncclGetUniqueId`, rank exchange,
//! `ncclCommInitRank`) is runtime/engine boot business, outside the plan;
//! the [`Ctx`] arrives with the communicator already open, or with none,
//! and a collective on a comm-less context is a typed refusal. A live
//! communicator can only exist because the engine already loaded `libnccl`
//! to build it, so no entry probes for the library.

use crate::error::Error;

use crate::jit::{Ctx, refuse};
use crate::tensor::Tensor;

/// `buf = Σ_ranks buf`, in place (the IR aliases `buf_out` onto `buf`).
pub fn all_reduce(ctx: &Ctx, buf: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_reduce";
    let comm = ctx.comm(OP)?;

    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        let dtype = wire_dtype(OP, buf.dtype)?;
        let Some((send, count)) = message(*buf) else {
            return Ok(());
        };
        let code = unsafe {
            nccl::ncclAllReduce(
                send,
                send.cast_mut(),
                count,
                dtype,
                nccl::ncclRedOp_t::ncclSum,
                comm.cast(),
                ctx.stream().cast(),
            )
        };
        answered(OP, "ncclAllReduce", code)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (comm, buf);
        Err(crate::jit::runtimeless(OP))
    }
}

/// Concatenates each rank's `x` into `y` on every rank, RANK-MAJOR: rank `k`
/// lands whole at `k * x.elements()`.
///
/// **THAT IS NOT THE WIDTH CONCAT THE IR DECLARES.** `model_dsl`'s
/// `collective::all_gather` shapes `y` as `[rows, width * world]` — each
/// rank's columns joined into every row — and the two layouts coincide only
/// at ONE ROW. At more rows this would land rank 0's whole rectangle, then
/// rank 1's, and the plan would read it as columns: a silent transpose. So a
/// wider fire is refused here rather than served wrong.
pub fn all_gather(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_gather";
    let comm = ctx.comm(OP)?;
    if x.rows > 1 {
        return Err(refuse(
            OP,
            format!(
                "this gather joins each rank's shard along the WIDTH (the IR shapes it \
                 `[rows, width * world]`) and ncclAllGather joins whole buffers rank-major; \
                 the two are the same layout only at one row, and this fire brought {}. \
                 Gathering wider wants a permute after the collective — `[world, rows, \
                 width]` to `[rows, world * width]` — which is not built.",
                x.rows,
            ),
        ));
    }
    debug_assert_eq!(x.dtype, y.dtype, "a gather does not change the dtype");
    debug_assert!(
        x.elements() > 0 && y.elements() % x.elements() == 0,
        "the gathered rectangle is a whole number of shards"
    );

    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        let dtype = wire_dtype(OP, x.dtype)?;
        let Some((send, count)) = message(x) else {
            return Ok(());
        };
        let code = unsafe {
            nccl::ncclAllGather(
                send,
                y.ptr as usize as *mut core::ffi::c_void,
                count,
                dtype,
                comm.cast(),
                ctx.stream().cast(),
            )
        };
        answered(OP, "ncclAllGather", code)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = comm;
        Err(crate::jit::runtimeless(OP))
    }
}

/// Sums `x` across ranks, leaving each rank its own shard in `y` — the shard
/// being a CONTIGUOUS BLOCK of the flat buffer, rank `k` taking
/// `[k * y.elements(), (k + 1) * y.elements())`.
///
/// **THAT IS NOT THE WIDTH SHARD THE IR DECLARES**, for
/// [`all_gather`]'s reason mirrored: `model_dsl`'s
/// `collective::reduce_scatter` shapes `y` as `[rows, width / world]` — each
/// rank keeping its columns of EVERY row — while NCCL hands it a block of
/// whole rows. The two coincide only at one row, so a wider fire is refused.
pub fn reduce_scatter(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.reduce_scatter";
    let comm = ctx.comm(OP)?;
    if x.rows > 1 {
        return Err(refuse(
            OP,
            format!(
                "this scatter keeps each rank its columns of every row (the IR shapes it \
                 `[rows, width / world]`) and ncclReduceScatter hands each rank a \
                 contiguous block of the flat buffer; the two are the same layout only at \
                 one row, and this fire brought {}. Scattering wider wants a permute before \
                 the collective, which is not built.",
                x.rows,
            ),
        ));
    }
    debug_assert_eq!(x.dtype, y.dtype, "a reduction does not change the dtype");
    debug_assert!(
        y.elements() > 0 && x.elements() % y.elements() == 0,
        "the reduced rectangle is a whole number of shards"
    );

    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        let dtype = wire_dtype(OP, x.dtype)?;
        let Some((recv, count)) = message(*y) else {
            return Ok(());
        };
        let code = unsafe {
            nccl::ncclReduceScatter(
                x.ptr as usize as *const core::ffi::c_void,
                recv.cast_mut(),
                count,
                dtype,
                nccl::ncclRedOp_t::ncclSum,
                comm.cast(),
                ctx.stream().cast(),
            )
        };
        answered(OP, "ncclReduceScatter", code)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = comm;
        Err(crate::jit::runtimeless(OP))
    }
}

/// The handle's dtype as NCCL spells it on the wire.
#[cfg(feature = "cuda")]
fn wire_dtype(
    op: &'static str,
    dtype: dtype::Dtype,
) -> Result<cudarc::nccl::sys::ncclDataType_t, Error> {
    use cudarc::nccl::sys::ncclDataType_t as t;

    Ok(crate::jit::dtype_dispatch!(op, dtype, {
        Bf16 => t::ncclBfloat16,
        F16 => t::ncclFloat16,
        F32 => t::ncclFloat32,
    }))
}

/// The rank-local message: its address and element count. `None` for an
/// empty rectangle — a conditioned fire may legitimately move nothing, and
/// a refusal here would kill the whole fire under graph capture.
#[cfg(feature = "cuda")]
fn message(t: Tensor) -> Option<(*const core::ffi::c_void, usize)> {
    let count = usize::try_from(t.elements()).ok()?;
    (count > 0).then_some((t.ptr as usize as *const core::ffi::c_void, count))
}

#[cfg(feature = "cuda")]
fn answered(
    op: &'static str,
    call: &'static str,
    code: cudarc::nccl::sys::ncclResult_t,
) -> Result<(), Error> {
    if code == cudarc::nccl::sys::ncclResult_t::ncclSuccess {
        return Ok(());
    }
    Err(crate::jit::Fault::Device {
        call,
        code: code as i32,
    }
    .at(op))
}
