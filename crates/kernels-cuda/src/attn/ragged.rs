//! `attention.ragged`: non-causal attention over arena q/k/v rectangles in
//! attention groups — the joint / cross / dual-stream attention of the
//! diffusion transformers (design D2). No kv pool, no page table, no host
//! plan and no mask slab: group `g`'s queries are the rows
//! `q_indptr[g]..q_indptr[g+1]` and attend every key `kv_indptr[g]..kv_indptr[g+1]`.
//!
//! Module path is `kernels_cuda::attn_ragged` via `#[path]` in `lib.rs`,
//! beside `attn_dense`, standing in for `attn::ragged`.
//!
//! **The kernel** is the vendored FlashInfer FA2 template
//! `BatchPrefillWithRaggedKVCacheKernel` (`MaskMode::kNone`, or `kCustom`
//! under the per-group reference mask, or `kNone` with the additive
//! relative-bias variant on its logits hook; NHD layout, bf16 in, fp32
//! accumulation, online softmax, bf16 out) — the same tensor-core kernel
//! `attention.prefill` runs, reading k/v straight out of two row-major
//! rectangles. The paged arms take a host-built plan; this one builds its
//! schedule on the device (`attn/ragged.cuh`'s `ragged_schedule`) from the
//! group tables it is handed, so the entry is two launches on one stream and
//! nothing reads the device from the host.
//!
//! **Numerics.** Scores are `q · k^T * sm_scale` in fp32 from bf16 operands
//! (`mma.sync` m16n8k16), softmax is the online fp32 form with `exp2` on
//! log2-scaled logits, and `P · V` accumulates in fp32 with `P` rounded to
//! bf16 for the tensor core — FA2's contract, the same as `attention.prefill`.
//! The output is rounded to bf16 once. Against an fp32 reference expect
//! ~1e-2 absolute at `|o| ≈ 1`.
//!
//! **Capture.** Both launches are plain kernels; the schedule tables live in a
//! named scratch slab (`ctx.scratch`) sized by the ceiling this call states
//! (`q.rows` and the table's group count), monotone in both, so an eager warm
//! fire at the bucket ceiling sizes the slab a capture then bakes. The grid of
//! the attention launch is `[padded, 1, kv_heads]` with
//! `padded = ceil((rows·group + groups·(tile − 1)) / tile)`, a tiling law in
//! rows at a fixed group ceiling. The seat is `Reads::RowsAndLanes`: tables
//! handed over whole (indexed by fire-global group, padded to the lane
//! ceiling with empty segments), values inside them plane-absolute, and no
//! seat word read at all — so a body of it replays at any row or lane offset.

use crate::attn::fa2::{self, RaggedArm, RaggedPoint};
use crate::attn::fa2_abi::{
    PrefillRaggedBiasParams, PrefillRaggedParams, PrefillRaggedRefParams, PrefillRaggedTagParams,
    UintFastdiv, sm_scale_or_default,
};
use crate::attn::kv;
use crate::attn::plan::Device;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::Tensor;
use dtype::Dtype;

const OP: &str = "attention.ragged";

/// The device planner's unit.
const SCHEDULE_FILE: &str = "attn/ragged.cuh";

/// The scratch slab the schedule tables live in — one per region.
const SCRATCH: &str = "attention.ragged.schedule";

/// The planner's block: one block, this wide, scanning groups a chunk at a
/// time.
const SCHEDULE_BLOCK: u32 = 1024;

/// The head widths this arm is stamped for. 512 has an fa2 unit, but the
/// ragged kernel's shared storage differs from the paged formula there, so it
/// stays refused until a shape asks for it.
const HEAD_DIMS: [u32; 3] = [64, 128, 256];

/// The `kv_chunk_size` the kernel divides by and never acts on (no kv
/// split): wide enough that every kv length is one chunk.
const KV_CHUNK_SENTINEL: i32 = i32::MAX;

/// Which keys a group's rows may see.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaggedMask {
    /// Every query row of a group attends every key row of the group.
    None,
    /// A group's tail is reference rows that see only each other. `ref_start`
    /// is `i32`, `[groups]` (indexed like the group tables — handed whole):
    /// the row of group `g`, counted from the group's own first row on both
    /// the query and the key side, where its reference rows begin. Rows at or
    /// past it attend only keys at or past it; rows before it attend every
    /// key, references included. A `ref_start` at or past the group's length,
    /// or a negative one (read as zero), leaves every row seeing every key.
    /// Counted the same on both sides, it is meant for the self-attention
    /// reading, where the query and key tables agree — and for at most ONE
    /// reference lane per group; [`ReferenceTags`](RaggedMask::ReferenceTags)
    /// is the general form.
    ReferenceSelfOnly { ref_start: Tensor },
    /// The contract's tag form (`IMAGEGEN_CONTRACT.md` §1): `q_tags` and
    /// `kv_tags` are `i32`, `[rows]` of the query and the key rectangle
    /// (indexed by the same absolute packed rows the CSRs name), `-1` for a
    /// row of a non-reference lane, else that lane's fire index. A query
    /// tagged `t >= 0` sees only keys tagged `t`; a query tagged `-1` sees
    /// every key of its segment, the references' included. Any number of
    /// reference lanes per group, each attending itself alone.
    ReferenceTags { q_tags: Tensor, kv_tags: Tensor },
    /// Every row of a group attends every key of the group, and an additive
    /// per-head bias that depends only on the signed distance `kj − qi`
    /// (both group-relative) is added to each scaled logit:
    /// `s = q·k · sm_scale + table[h][clamp(kj − qi + max_len − 1, 0,
    /// 2·max_len − 2)]`. `table` is `f32`, `[q_heads, 2·max_len − 1]`
    /// row-major, handed whole — one row per QUERY head, so a grouped kv
    /// head's queries each read their own row. Distances past `±(max_len −
    /// 1)` read the table's end columns, so a table built at the longest
    /// segment the fire may hold is exact, and one built shorter saturates
    /// (which is the T5 bucket function's own behaviour past
    /// `max_distance`). The umT5 relative position bias (a dense table per
    /// layer, `elemwise::relative_bucket_bias`) and an ALiBi slope table
    /// both fit.
    RelativeBias { table: Tensor, max_len: u32 },
}

/// FlashInfer's own CTA tile for the query axis at this head width, for
/// long queries: 128 packed (row, head) pairs below head width 256, where
/// `NUM_MMA_Q = 2` still fits the register budget; 64 at 256, where the
/// 128-row traits are `IsInvalid()` for every kv tile. A function of the
/// head width alone, so the instantiation never moves with the fire.
#[must_use]
const fn cta_tile_q(head_dim: u32) -> u32 {
    if head_dim >= 256 { 64 } else { 128 }
}

/// The head count a row's width spells at a stated head width.
fn row_heads(what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!("the {width}-wide {what} row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

/// The work items the grid is sized for at a stated row and group ceiling:
/// `Σ_g ceil(q_g·group / tile) ≤ ceil((rows·group + groups·(tile − 1)) / tile)`,
/// the bound every split of `rows` into `groups` groups stays under.
#[must_use]
pub fn padded_work_items(rows: u32, groups: u32, group_size: u32, head_dim: u32) -> u64 {
    let tile = u64::from(cta_tile_q(head_dim));
    let packed = u64::from(rows) * u64::from(group_size.max(1));
    (packed + u64::from(groups) * (tile - 1))
        .div_ceil(tile)
        .max(1)
}

/// Where each schedule table sits in the scratch slab, and the slab's size.
#[derive(Clone, Copy, Debug)]
struct Tables {
    request_indices: u64,
    qo_tile_indices: u64,
    kv_tile_indices: u64,
    block_valid_mask: u64,
    kv_chunk_size: u64,
    bytes: u64,
}

impl Tables {
    fn lay(padded: u64) -> Self {
        const fn align16(n: u64) -> u64 {
            n.div_ceil(16) * 16
        }
        let ints = align16(padded * 4);
        let request_indices = 0;
        let qo_tile_indices = request_indices + ints;
        let kv_tile_indices = qo_tile_indices + ints;
        let block_valid_mask = kv_tile_indices + ints;
        let kv_chunk_size = block_valid_mask + align16(padded);
        Self {
            request_indices,
            qo_tile_indices,
            kv_tile_indices,
            block_valid_mask,
            kv_chunk_size,
            bytes: kv_chunk_size + 16,
        }
    }
}

/// Non-causal ragged attention: every query row of a group attends every key
/// row of the same group.
///
/// `q` is `[q_rows, q_heads · head_dim]`, `k` and `v` are
/// `[kv_rows, kv_heads · head_dim]`, all bf16 and row-major; `o` lands one
/// bf16 row per query row at q's own shape. `q_indptr` and `kv_indptr` are
/// `i32`, `[groups + 1]`, the same `groups`, each spelling plane rows: group
/// `g`'s queries are rows `q_indptr[g]..q_indptr[g+1]` of `q` and its keys
/// rows `kv_indptr[g]..kv_indptr[g+1]` of `k`/`v`. The two tables may be one
/// (self-attention) or not (cross-attention); a group with no keys lands
/// zeros, a query row outside every group is left untouched. Grouped heads:
/// `q_heads % kv_heads == 0`. `sm_scale` is the caller's; a non-positive one
/// reads as `1/sqrt(head_dim)`, as on the paged arms.
///
/// Errs [`Error::DtypeUnsupported`] for anything but bf16, or a refusal for a
/// head width outside 64/128/256, a row width that does not divide by it,
/// ungrouped heads, a mismatched or empty group table, or a shape the fa2
/// traits cannot be stamped at on this device.
#[allow(clippy::too_many_arguments)]
pub fn ragged(
    ctx: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    sm_scale: f32,
    mask: RaggedMask,
    o: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    debug_assert_eq!(v.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    if !HEAD_DIMS.contains(&head_dim) {
        return Err(refuse(
            OP,
            format!(
                "no ragged fa2 unit is stamped at head width {head_dim}; the arm holds 64/128/256"
            ),
        ));
    }
    let num_q_heads = row_heads("query", q.width, head_dim)?;
    let num_kv_heads = row_heads("key", k.width, head_dim)?;
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {num_kv_heads} kv heads"),
        ));
    }
    if v.width != k.width {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide value row is not the {}-wide key row",
                v.width, k.width
            ),
        ));
    }
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "`{OP}` lands one output row per query row"
    );
    debug_assert_eq!(k.rows, v.rows, "`{OP}` reads one value row per key row");
    let groups = kv::lanes_of(OP, q_indptr)?;
    if kv_indptr.rows != q_indptr.rows || kv_indptr.dtype != q_indptr.dtype {
        return Err(refuse(
            OP,
            format!(
                "the query table spells {} groups and the key table {}; a group is one entry \
                 of each",
                q_indptr.rows.saturating_sub(1),
                kv_indptr.rows.saturating_sub(1)
            ),
        ));
    }
    let rows = count(OP, "the query rows this attention answers", q.rows)?;
    let group_size = num_q_heads / num_kv_heads;

    let padded = padded_work_items(q.rows, groups.unsigned_abs(), group_size, head_dim);
    let padded_u32 = u32::try_from(padded)
        .ok()
        .filter(|&n| n <= i32::MAX as u32)
        .ok_or_else(|| {
            refuse(
                OP,
                format!("{padded} work items do not fit the device's i32"),
            )
        })?;
    let tables = Tables::lay(padded);
    let bytes = usize::try_from(tables.bytes).map_err(|_| {
        refuse(
            OP,
            "the schedule tables do not fit this host's address space",
        )
    })?;
    let slab = ctx.scratch(OP, SCRATCH, bytes)? as u64;

    // Launch one: the schedule, from the group table. A `win` reader, so a
    // replay above lane zero names its own groups.
    ctx.fire(
        OP,
        Fire::at(SCHEDULE_FILE, symbol("::pie::attn::ragged_schedule"))
            .apply(Launch::grid([1, 1, 1], [SCHEDULE_BLOCK, 1, 1]).smem(SCHEDULE_BLOCK * 4)),
        &[
            q_indptr.arg(),
            groups.arg(),
            stated(OP, padded_u32)?.arg(),
            stated(OP, group_size)?.arg(),
            stated(OP, cta_tile_q(head_dim))?.arg(),
            KV_CHUNK_SENTINEL.arg(),
            ArgValue::Ptr(slab + tables.request_indices),
            ArgValue::Ptr(slab + tables.qo_tile_indices),
            ArgValue::Ptr(slab + tables.kv_tile_indices),
            ArgValue::Ptr(slab + tables.block_valid_mask),
            ArgValue::Ptr(slab + tables.kv_chunk_size),
            // No seat: the engine hands the group tables over WHOLE, indexed
            // by fire-global group and padded to the lane ceiling with empty
            // segments, so every entry the table names is a live-or-empty
            // group and a body replay above lane zero reads the same table
            // the eager walk does. The window's lane words (`win[2..4]`)
            // count the window's LANES, which a joint attention's groups are
            // not, so they are not read.
            ArgValue::ABSENT,
        ],
    )?;

    // Launch two: the attention. `o_indptr` is the query table itself — the
    // output lands at q's own rows — and nothing is split or merged.
    let params = PrefillRaggedParams {
        q: q.ptr,
        k: k.ptr,
        v: v.ptr,
        q_indptr: q_indptr.ptr,
        kv_indptr: kv_indptr.ptr,
        o: o.ptr,
        group_size: UintFastdiv::new(group_size),
        num_qo_heads: num_q_heads,
        num_kv_heads,
        q_stride_n: q.width,
        q_stride_h: head_dim,
        k_stride_n: k.width,
        k_stride_h: head_dim,
        v_stride_n: v.width,
        v_stride_h: head_dim,
        window_left: -1,
        logits_soft_cap: 0.0,
        sm_scale: sm_scale_or_default(sm_scale, head_dim),
        rope_rcp_scale: 1.0,
        rope_rcp_theta: 1.0,
        request_indices: slab + tables.request_indices,
        qo_tile_indices: slab + tables.qo_tile_indices,
        kv_tile_indices: slab + tables.kv_tile_indices,
        o_indptr: q_indptr.ptr,
        kv_chunk_size_ptr: slab + tables.kv_chunk_size,
        block_valid_mask: slab + tables.block_valid_mask,
        max_total_num_rows: rows.unsigned_abs(),
        padded_batch_size: padded_u32,
        partition_kv: false,
        ..PrefillRaggedParams::default()
    };
    let point = |arm: RaggedArm| RaggedPoint {
        head_dim,
        cta_tile_q: cta_tile_q(head_dim),
        arm,
        padded_batch_size: padded_u32,
        num_kv_heads,
        device: Device::probe(ctx).unwrap_or(Device::L40S),
    };
    match mask {
        RaggedMask::None => fa2::prefill_ragged(ctx, OP, point(RaggedArm::Full), &params),
        RaggedMask::ReferenceTags { q_tags, kv_tags } => {
            for (what, table, rows) in [("query", q_tags, q.rows), ("key", kv_tags, k.rows)] {
                if table.dtype != Dtype::I32 || table.rows < rows {
                    return Err(refuse(
                        OP,
                        format!(
                            "the {what} tag table is {:?} with {} entries; the mask reads one \
                             i32 per row of the {rows}-row {what} rectangle",
                            table.dtype, table.rows
                        ),
                    ));
                }
            }
            fa2::prefill_ragged(
                ctx,
                OP,
                point(RaggedArm::ReferenceTags),
                &PrefillRaggedTagParams {
                    base: params,
                    q_tags: q_tags.ptr,
                    kv_tags: kv_tags.ptr,
                },
            )
        }
        RaggedMask::ReferenceSelfOnly { ref_start } => {
            // Read at the absolute group id, like the group tables: it must
            // reach every group the table names.
            if ref_start.dtype != Dtype::I32 || ref_start.rows < groups.unsigned_abs() {
                return Err(refuse(
                    OP,
                    format!(
                        "the reference table is {:?} with {} entries; the mask reads one i32 \
                         per group of the {groups} the tables name",
                        ref_start.dtype, ref_start.rows
                    ),
                ));
            }
            fa2::prefill_ragged(
                ctx,
                OP,
                point(RaggedArm::ReferenceSelfOnly),
                &PrefillRaggedRefParams {
                    base: params,
                    ref_start: ref_start.ptr,
                },
            )
        }
        RaggedMask::RelativeBias { table, max_len } => {
            // The variant multiplies the raw logit by the block's `sm_scale`
            // (already defaulted above) before the add, and scales the online
            // softmax by `log2e` alone.
            let span = max_len
                .checked_mul(2)
                .and_then(|n| n.checked_sub(1))
                .filter(|_| max_len > 0)
                .ok_or_else(|| {
                    refuse(
                        OP,
                        format!("a relative bias over {max_len} positions has no table width"),
                    )
                })?;
            if table.dtype != Dtype::F32 || table.rows < num_q_heads || table.width != span {
                return Err(refuse(
                    OP,
                    format!(
                        "the relative bias table is {} x {} {:?}; the arm reads one f32 row of \
                         {span} (2 · {max_len} − 1) per query head, {num_q_heads} of them",
                        table.rows, table.width, table.dtype
                    ),
                ));
            }
            fa2::prefill_ragged(
                ctx,
                OP,
                point(RaggedArm::RelativeBias),
                &PrefillRaggedBiasParams {
                    base: params,
                    bias: table.ptr,
                    max_len,
                },
            )
        }
    }
}
