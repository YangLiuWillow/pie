//! `emit_streamed_region_msl` — the streamed form (`ptir_m4`): one kernel per
//! fused region, run as a **table of dispatches**, each over a grid of
//! `(element blocks × lanes)`.
//!
//! The grouped form gives a lane one threadgroup and walks the region's ops
//! inside it, barrier after barrier. A lane's epilogue is then bound to one
//! GPU core: a vocabulary-wide value streams through it at a small fraction
//! of the device's bandwidth, and thirty of them take milliseconds. Here the
//! barrier between ops is a dispatch boundary instead. Every element-
//! independent op runs across the whole grid; a reduction runs as one
//! dispatch per level of its fixed tree; an op whose walk carries state
//! (a scan, a sort, a pivot selection, a scatter) keeps the grouped form's
//! single-threadgroup path under `m4_groups.x == 1`. Lanes are the grid's
//! second axis, so a batch of instances is one dispatch per step.
//!
//! The kernel takes the grouped form's eleven bindings plus an `M4Step` at
//! buffer 11 — which op, which reduction level, how many groups the partial
//! pass had — and switches on it. The engine reads the step table this file
//! answers beside the source (`EmittedKernel::steps`) and issues the
//! dispatches in order; a `Reduce` step is several dispatches, an `Argmax`
//! two, and the engine derives their counts from the same descriptors the
//! kernel reads. Nothing about the arithmetic changes: the ops are the
//! runtime's `ptir_m1_execute_part` strided by the grid, and the reductions
//! reproduce the 32-wide tree level by level.

use crate::codegen::error::{EmitError, RegionForm};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;
use eta_ir::op::{intrinsic_tags, tags};
use eta_ir::wire::predicate_tags;

use crate::plan::{CompiledStage, Region};

use super::fused::{
    METAL_M3_REGION_THREADS, emit_logits_argmax, emit_logits_gather, emit_mtp_drafts,
    emit_score_gather,
};
use super::preamble::{RUNTIME_TEMPLATE, grouped_preamble};
use super::validate::{grouped_intrinsics_bindable, library_region_valid, used_channel_slots};
use crate::codegen::fault::M3_THREADS_EXCEEDED;
use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::slots::Slots;

/// How one step of a streamed region is dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StepKind {
    /// Element-independent: the whole grid, strided.
    Wide = 0,
    /// A walk with state, or a fused threadgroup pattern: one threadgroup.
    Single = 1,
    /// `reduce_sum/max/min`: one dispatch per level of the 32-wide tree.
    Reduce = 2,
    /// `reduce_argmax` over f32: a partial pass over the grid, then a final
    /// pass over the partials in one threadgroup.
    Argmax = 3,
}

/// Pack a step for `EmittedKernel::steps`.
#[must_use]
pub const fn streamed_step(node: u32, kind: StepKind) -> u32 {
    (node << 8) | kind as u32
}

/// The op a packed step runs.
#[must_use]
pub const fn step_node(step: u32) -> u32 {
    step >> 8
}

/// The kind a packed step is; `None` for a byte this version does not emit.
#[must_use]
pub const fn step_kind(step: u32) -> Option<StepKind> {
    match step & 0xFF {
        0 => Some(StepKind::Wide),
        1 => Some(StepKind::Single),
        2 => Some(StepKind::Reduce),
        3 => Some(StepKind::Argmax),
        _ => None,
    }
}

/// Levels of the 32-wide reduction tree over a row of `last` elements, until
/// one value is left; at least one (a row of one element still writes its
/// result). The runtime's `m4_reduce_two_levels` walks two of them per
/// dispatch when its threadgroup is a multiple of 32 wide, so the engine
/// dispatches once per entry of [`reduce_dispatch_levels`].
#[must_use]
pub fn reduce_levels(last: u32) -> u32 {
    let mut count = last;
    let mut levels = 0u32;
    while count > 1 {
        count = count.div_ceil(32);
        levels += 1;
    }
    levels.max(1)
}

/// The first level each `Reduce` dispatch starts at: `0, 2, 4, …` up to the
/// tree's depth. A dispatch folds its level and the next one when the next
/// exists and its threadgroup width allows; the runtime settles that per
/// dispatch, so the engine only has to issue this many.
#[must_use]
pub fn reduce_dispatch_levels(last: u32) -> Vec<u32> {
    (0..reduce_levels(last)).step_by(2).collect()
}

/// How an op runs in the streamed form, from its tag alone (plus the
/// pivot's predicate and the argmax operand's class). Everything
/// `ptir_m1_execute_part` strides without a barrier is `Wide`; the walks the
/// grouped form keeps on one threadgroup are `Single`; the fixed-tree
/// reductions and the f32 argmax have their own multi-dispatch shapes.
#[must_use]
pub fn op_step_kind(tag: u8, pred_tag: u8, argmax_over_f32: bool) -> StepKind {
    match tag {
        tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN => StepKind::Reduce,
        tags::REDUCE_ARGMAX => {
            if argmax_over_f32 {
                StepKind::Argmax
            } else {
                StepKind::Single
            }
        }
        tags::CUMSUM
        | tags::CUMPROD
        | tags::SORT_DESC
        | tags::TOP_K
        | tags::MATMUL
        | tags::SCATTER_ADD
        | tags::SCATTER_SET => StepKind::Single,
        tags::PIVOT_THRESHOLD => {
            if pred_tag == predicate_tags::PROB_GE {
                StepKind::Wide
            } else {
                StepKind::Single
            }
        }
        _ => StepKind::Wide,
    }
}

/// The wire byte of a value's dtype, or `None` outside ETA's four.
fn wire_dtype(value_types: &[crate::plan::SymbolicType], value: u32) -> Option<u8> {
    value_types
        .get(value as usize)
        .and_then(|ty| eta_ir::types::to_wire(ty.dtype))
}

/// A `Wide` step's body emitted directly for the element-independent ops
/// whose tag and dtypes are known here — the runtime's own arithmetic,
/// spelled with literal dtypes so every `m1_load_*` / `m1_store_*` folds to
/// one typed access and no tag is switched on per element. Everything else
/// answers `None` and takes the generic `ptir_m1_execute_part`.
#[allow(clippy::too_many_lines)]
fn direct_wide(
    op: &OpView,
    node: usize,
    base: u32,
    slots: &Slots,
    value_types: &[crate::plan::SymbolicType],
    alias: &crate::codegen::alias::AliasTable,
) -> Option<(String, String)> {
    let arg = |k: usize| op.args.get(k).map(|&a| alias.resolve(a));
    let dt = |value: Option<u32>| value.and_then(|v| wire_dtype(value_types, v));
    let d0 = dt(arg(0));
    let d1 = dt(arg(1));
    let d2 = dt(arg(2));
    let dout = wire_dtype(value_types, base);
    let a0 = &slots.a0;
    let a1 = &slots.a1;
    let a2 = &slots.a2;
    let o0 = &slots.o0;
    // Names carry the node so several ops share one loop body.
    let n = format!("m4_n_{node}");
    let sx = format!("sx_{node}");
    let sy = format!("sy_{node}");
    let sc = format!("sc_{node}");
    let x = format!("x_{node}");
    let y = format!("y_{node}");
    let c = format!("c_{node}");
    let mut pre = String::new();
    let mut inner = String::new();
    let _ = writeln!(pre, "    const uint {n} = descriptors[{base}].len;");
    // Operand `k` is read at `i` or, when it is a scalar, at 0 — `m1_pick`.
    let pick = |k: usize, name: &str, pre: &mut String| {
        if let Some(value) = arg(k) {
            let _ = writeln!(
                pre,
                "    const uint {name} = descriptors[{value}].len == 1u ? 0u : 1u;"
            );
        }
    };
    let guard = format!("      if (i < {n}) {{\n");
    match op.tag {
        tags::EXP | tags::LOG | tags::RECIP => {
            let (d0, _) = (d0?, dout?);
            pick(0, &sx, &mut pre);
            inner.push_str(&guard);
            let _ = writeln!(inner, "        const float {x} = m1_load_f({a0}, i * {sx}, {d0}u);");
            let expr = match op.tag {
                tags::EXP => format!("precise::exp({x})"),
                tags::LOG => format!("precise::log({x})"),
                _ => format!("1.0f / {x}"),
            };
            let _ = writeln!(inner, "        m1_store_f({o0}, i, {expr});");
            inner.push_str("      }\n");
        }
        tags::NEG | tags::ABS | tags::SIGN => {
            let d0 = d0?;
            if d0 == 3 {
                return None;
            }
            pick(0, &sx, &mut pre);
            inner.push_str(&guard);
            let line = match (d0, op.tag) {
                (0, tags::NEG) => format!("        m1_store_f({o0}, i, -m1_load_f({a0}, i * {sx}, 0u));\n"),
                (0, tags::ABS) => format!("        m1_store_f({o0}, i, abs(m1_load_f({a0}, i * {sx}, 0u)));\n"),
                (0, _) => format!("        {{ const float {x} = m1_load_f({a0}, i * {sx}, 0u); m1_store_f({o0}, i, {x} > 0 ? 1.0f : ({x} < 0 ? -1.0f : 0.0f)); }}\n"),
                (1, tags::NEG) => format!("        m1_store_i({o0}, i, int(0u - uint(m1_load_i({a0}, i * {sx}, 1u))));\n"),
                (1, tags::ABS) => format!("        {{ const int {x} = m1_load_i({a0}, i * {sx}, 1u); m1_store_i({o0}, i, {x} == INT_MIN ? {x} : abs({x})); }}\n"),
                (1, _) => format!("        {{ const int {x} = m1_load_i({a0}, i * {sx}, 1u); m1_store_i({o0}, i, {x} > 0 ? 1 : ({x} < 0 ? -1 : 0)); }}\n"),
                (_, tags::NEG) => format!("        m1_store_u({o0}, i, 0u - m1_load_u({a0}, i * {sx}, 2u));\n"),
                (_, tags::SIGN) => format!("        m1_store_u({o0}, i, m1_load_u({a0}, i * {sx}, 2u) != 0 ? 1u : 0u);\n"),
                (_, _) => format!("        m1_store_u({o0}, i, m1_load_u({a0}, i * {sx}, 2u));\n"),
            };
            inner.push_str(&line);
            inner.push_str("      }\n");
        }
        tags::CAST => {
            let (d0, dout) = (d0?, dout?);
            pick(0, &sx, &mut pre);
            inner.push_str(&guard);
            let (store, load) = match dout {
                0 => ("m1_store_f", "m1_load_f"),
                1 => ("m1_store_i", "m1_load_i"),
                2 => ("m1_store_u", "m1_load_u"),
                _ => ("m1_store_b", "m1_load_b"),
            };
            let _ = writeln!(inner, "        {store}({o0}, i, {load}({a0}, i * {sx}, {d0}u));");
            inner.push_str("      }\n");
        }
        tags::ADD
        | tags::SUB
        | tags::MUL
        | tags::DIV
        | tags::MAX_ELEM
        | tags::MIN_ELEM
        | tags::REM => {
            let (d0, d1) = (d0?, d1?);
            pick(0, &sx, &mut pre);
            pick(1, &sy, &mut pre);
            inner.push_str(&guard);
            match d0 {
                0 => {
                    let _ = writeln!(inner, "        const float {x} = m1_load_f({a0}, i * {sx}, {d0}u);");
                    let _ = writeln!(inner, "        const float {y} = m1_load_f({a1}, i * {sy}, {d1}u);");
                    let expr = match op.tag {
                        tags::ADD => format!("{x} + {y}"),
                        tags::SUB => format!("{x} - {y}"),
                        tags::MUL => format!("{x} * {y}"),
                        tags::DIV => format!("{x} / {y}"),
                        tags::MAX_ELEM => format!("m1_element_max({x}, {y})"),
                        tags::MIN_ELEM => format!("m1_element_min({x}, {y})"),
                        _ => format!("fmod({x}, {y})"),
                    };
                    let _ = writeln!(inner, "        m1_store_f({o0}, i, {expr});");
                }
                1 => {
                    let _ = writeln!(inner, "        const int {x} = m1_load_i({a0}, i * {sx}, {d0}u);");
                    let _ = writeln!(inner, "        const int {y} = m1_load_i({a1}, i * {sy}, {d1}u);");
                    let expr = match op.tag {
                        tags::ADD => format!("int(uint({x}) + uint({y}))"),
                        tags::SUB => format!("int(uint({x}) - uint({y}))"),
                        tags::MUL => format!("int(uint({x}) * uint({y}))"),
                        tags::DIV => format!("({y} == 0 ? 0 : {x} / {y})"),
                        tags::MAX_ELEM => format!("max({x}, {y})"),
                        tags::MIN_ELEM => format!("min({x}, {y})"),
                        _ => format!("({y} == 0 ? 0 : {x} % {y})"),
                    };
                    let _ = writeln!(inner, "        m1_store_i({o0}, i, {expr});");
                }
                _ => {
                    let _ = writeln!(inner, "        const uint {x} = m1_load_u({a0}, i * {sx}, {d0}u);");
                    let _ = writeln!(inner, "        const uint {y} = m1_load_u({a1}, i * {sy}, {d1}u);");
                    let expr = match op.tag {
                        tags::ADD => format!("{x} + {y}"),
                        tags::SUB => format!("{x} - {y}"),
                        tags::MUL => format!("{x} * {y}"),
                        tags::DIV => format!("({y} == 0 ? 0 : {x} / {y})"),
                        tags::MAX_ELEM => format!("max({x}, {y})"),
                        tags::MIN_ELEM => format!("min({x}, {y})"),
                        _ => format!("({y} == 0 ? 0 : {x} % {y})"),
                    };
                    let _ = writeln!(inner, "        m1_store_u({o0}, i, {expr});");
                }
            }
            inner.push_str("      }\n");
        }
        tags::GT | tags::GE | tags::EQ | tags::NE | tags::LT | tags::LE => {
            let (d0, d1) = (d0?, d1?);
            pick(0, &sx, &mut pre);
            pick(1, &sy, &mut pre);
            inner.push_str(&guard);
            let (ty, load) = match d0 {
                0 => ("float", "m1_load_f"),
                1 => ("int", "m1_load_i"),
                _ => ("uint", "m1_load_u"),
            };
            let _ = writeln!(inner, "        const {ty} {x} = {load}({a0}, i * {sx}, {d0}u);");
            let _ = writeln!(inner, "        const {ty} {y} = {load}({a1}, i * {sy}, {d1}u);");
            let cmp = match op.tag {
                tags::GT => format!("{x} > {y}"),
                tags::GE => format!("{x} >= {y}"),
                tags::EQ => format!("{x} == {y}"),
                tags::NE => format!("{x} != {y}"),
                tags::LT => format!("{x} < {y}"),
                _ => format!("{x} <= {y}"),
            };
            let _ = writeln!(inner, "        m1_store_b({o0}, i, {cmp});");
            inner.push_str("      }\n");
        }
        tags::AND | tags::OR => {
            let (d0, d1) = (d0?, d1?);
            pick(0, &sx, &mut pre);
            pick(1, &sy, &mut pre);
            inner.push_str(&guard);
            let _ = writeln!(inner, "        const bool {x} = m1_load_b({a0}, i * {sx}, {d0}u);");
            let _ = writeln!(inner, "        const bool {y} = m1_load_b({a1}, i * {sy}, {d1}u);");
            let _ = writeln!(
                inner,
                "        m1_store_b({o0}, i, {});",
                if op.tag == tags::AND { format!("{x} && {y}") } else { format!("{x} || {y}") }
            );
            inner.push_str("      }\n");
        }
        tags::NOT => {
            let d0 = d0?;
            pick(0, &sx, &mut pre);
            inner.push_str(&guard);
            let _ = writeln!(inner, "        m1_store_b({o0}, i, !m1_load_b({a0}, i * {sx}, {d0}u));");
            inner.push_str("      }\n");
        }
        tags::SELECT => {
            let (d0, d1, d2, dout) = (d0?, d1?, d2?, dout?);
            pick(0, &sc, &mut pre);
            pick(1, &sx, &mut pre);
            pick(2, &sy, &mut pre);
            inner.push_str(&guard);
            let (store, load) = match dout {
                0 => ("m1_store_f", "m1_load_f"),
                1 => ("m1_store_i", "m1_load_i"),
                2 => ("m1_store_u", "m1_load_u"),
                _ => ("m1_store_b", "m1_load_b"),
            };
            let _ = writeln!(inner, "        const bool {c} = m1_load_b({a0}, i * {sc}, {d0}u);");
            let _ = writeln!(
                inner,
                "        {store}({o0}, i, {c} ? {load}({a1}, i * {sx}, {d1}u) : {load}({a2}, i * {sy}, {d2}u));"
            );
            inner.push_str("      }\n");
        }
        tags::CONST => {
            inner.push_str(&guard);
            let bits = op.lit_bits;
            let store = match op.lit_dtype {
                0 => format!("m1_store_f({o0}, i, as_type<float>({bits}u));"),
                1 => format!("m1_store_i({o0}, i, int({bits}u));"),
                2 => format!("m1_store_u({o0}, i, {bits}u);"),
                _ => format!("m1_store_b({o0}, i, {});", if bits != 0 { "true" } else { "false" }),
            };
            let _ = writeln!(inner, "        {store}");
            inner.push_str("      }\n");
        }
        tags::IOTA => {
            inner.push_str(&guard);
            let _ = writeln!(inner, "        m1_store_u({o0}, i, i);");
            inner.push_str("      }\n");
        }
        tags::BROADCAST => {
            // Left-aligned broadcast. A scalar source reads element 0 for
            // every output; anything else walks the runtime's index
            // arithmetic per element — equal lengths do NOT mean identity
            // here (`[V]` into `[1, V]` aligns `V` against `1` and reads 0
            // throughout). Decided once per dispatch, uniformly.
            let (d0, dout) = (d0?, dout?);
            let src = arg(0)?;
            let mode = format!("bc_{node}");
            let _ = writeln!(
                pre,
                "    const uint {mode} = descriptors[{src}].len == 1u ? 0u : 2u;"
            );
            let (store, load) = match dout {
                0 => ("m1_store_f", "m1_load_f"),
                1 => ("m1_store_i", "m1_load_i"),
                2 => ("m1_store_u", "m1_load_u"),
                _ => ("m1_store_b", "m1_load_b"),
            };
            inner.push_str(&guard);
            let _ = writeln!(inner, "        uint {x} = {mode} == 0u ? 0u : i;");
            let _ = writeln!(inner, "        if ({mode} == 2u) {{");
            let _ = writeln!(inner, "          const M1ValueDesc bd0 = descriptors[{src}];");
            let _ = writeln!(inner, "          const M1ValueDesc bo0 = descriptors[{base}];");
            inner.push_str("          uint rem = i, source_index = 0;\n");
            inner.push_str("          uint source_stride[4] = {1, 1, 1, 1};\n");
            inner.push_str("          for (int dim = int(bo0.rank) - 2; dim >= 0; --dim)\n");
            inner.push_str("            source_stride[dim] = source_stride[dim + 1] * (uint(dim + 1) < bd0.rank ? bd0.dims[dim + 1] : 1u);\n");
            inner.push_str("          for (uint dim = 0; dim < bo0.rank; ++dim) {\n");
            inner.push_str("            uint stride = 1;\n");
            inner.push_str("            for (uint next = dim + 1; next < bo0.rank; ++next) stride *= bo0.dims[next];\n");
            inner.push_str("            const uint coordinate = rem / max(stride, 1u);\n");
            inner.push_str("            rem %= max(stride, 1u);\n");
            inner.push_str("            const uint source_dim = dim < bd0.rank ? bd0.dims[dim] : 1u;\n");
            inner.push_str("            if (source_dim != 1) source_index += coordinate * source_stride[dim];\n");
            inner.push_str("          }\n");
            let _ = writeln!(inner, "          {x} = source_index;");
            inner.push_str("        }\n");
            let _ = writeln!(inner, "        {store}({o0}, i, {load}({a0}, {x}, {d0}u));");
            inner.push_str("      }\n");
        }
        tags::RESHAPE | tags::CHAN_TAKE | tags::CHAN_READ => {
            // A materialised copy, element `i` of `n`.
            let dout = dout?;
            if dout == 3 {
                // A packed-bool channel root unpacks bits; keep the runtime's walk.
                if op.tag != tags::RESHAPE {
                    return None;
                }
            }
            inner.push_str(&guard);
            let (store, load) = match dout {
                0 => ("m1_store_f", "m1_load_f"),
                1 => ("m1_store_i", "m1_load_i"),
                2 => ("m1_store_u", "m1_load_u"),
                _ => ("m1_store_b", "m1_load_b"),
            };
            let _ = writeln!(inner, "        {store}({o0}, i, {load}({a0}, i, {dout}u));");
            inner.push_str("      }\n");
        }
        _ => return None,
    }
    Some((pre, inner))
}

/// The step record the engine hands the kernel, one per dispatch. Spelled
/// here and in `engine-metal`'s `program::launch`; `tests` below pin the
/// text.
pub const STEP_STRUCT: &str = "struct M4Step {\n  uint index;\n  uint level;\n  uint groups;\n  uint reserved;\n};\n";

/// One fused region as a streamed kernel, and its dispatch table.
///
/// # Errors
///
/// The grouped form's refusals: an intrinsic the lane record cannot bind, a
/// library region whose ABI does not hold, a node outside the stage.
pub fn emit_streamed_region(
    function_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<(String, Vec<u32>), EmitError> {
    if !library_region_valid(stage, region) {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedFused));
    }
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    grouped_intrinsics_bindable(&ops, region)?;
    let bases = result_bases(&ops);
    let channel_count = used_channel_slots(&ops);
    let value_types = &stage.normalized.value_types;

    let mut source = String::new();
    source.push_str(RUNTIME_TEMPLATE);
    source.push('\n');
    source.push_str(grouped_preamble());
    source.push_str(STEP_STRUCT);
    let _ = writeln!(source, "kernel void {function_name}(");
    source.push_str("    const device uchar* lane_bytes [[buffer(0)]],\n");
    source.push_str("    const device M1ValueDesc* all_descriptors [[buffer(1)]],\n");
    source.push_str("    const device M1OpParams* params [[buffer(2)]],\n");
    source.push_str("    const device uint* offsets [[buffer(3)]],\n");
    source.push_str("    device uchar* all_scratch [[buffer(4)]],\n");
    source.push_str("    const device M3GroupLayout* layout [[buffer(5)]],\n");
    source.push_str("    const device uint* channel_bindings [[buffer(6)]],\n");
    source.push_str("    device uchar* pending_flags [[buffer(7)]],\n");
    source.push_str("    const device uint* lane_indices [[buffer(8)]],\n");
    source.push_str("    const device M3RowMeta* all_row_meta [[buffer(9)]],\n");
    source.push_str("    const device uint* row_indices [[buffer(10)]],\n");
    source.push_str("    constant M4Step& step [[buffer(11)]],\n");
    source.push_str("    uint2 m4_group [[threadgroup_position_in_grid]],\n");
    source.push_str("    uint2 m4_groups [[threadgroups_per_grid]],\n");
    // Every grid attribute of a kernel must share one dimensionality, so the
    // thread ones are `uint2` too and read through `.x`.
    source.push_str("    uint2 m4_tid [[thread_position_in_threadgroup]],\n");
    source.push_str("    uint2 m4_threads [[threads_per_threadgroup]]) {\n");
    source.push_str("  const uint m3_tid = m4_tid.x;\n");
    source.push_str("  const uint m3_threads = m4_threads.x;\n");
    let _ = writeln!(
        source,
        "  threadgroup M1ArgmaxCandidate m3_tgbuf[{METAL_M3_REGION_THREADS}];"
    );
    // The lane is the grid's second axis; the first is element blocks.
    source.push_str("  const uint dispatch_lane = m4_group.y;\n");
    source.push_str("  if (dispatch_lane >= layout->lane_count) return;\n");
    source.push_str("  const uint lane_index = lane_indices[dispatch_lane];\n");
    source.push_str(
        "  const device M3LaneHeader* header = \
         reinterpret_cast<const device M3LaneHeader*>(lane_bytes);\n",
    );
    source.push_str(
        "  const device M3LaneRecord* lanes = \
         reinterpret_cast<const device M3LaneRecord*>(lane_bytes + sizeof(M3LaneHeader));\n",
    );
    source.push_str(
        "  const device M3LaneChannelSlot* slots = \
         reinterpret_cast<const device M3LaneChannelSlot*>(lane_bytes + \
         sizeof(M3LaneHeader) + header->lane_count * sizeof(M3LaneRecord));\n",
    );
    source.push_str("  const M3LaneRecord lane = lanes[lane_index];\n");
    source.push_str("  const M3RowMeta row_meta = all_row_meta[lane_index];\n");
    source.push_str(
        "  device M1Status* status = \
         reinterpret_cast<device M1Status*>(lane.commit_slot);\n",
    );
    // A faulted lane stops at the next dispatch; every thread of every
    // group sees the same word, so the return is uniform.
    source.push_str("  if (status->state != 1) return;\n");
    let _ = writeln!(
        source,
        "  if (m3_threads > {METAL_M3_REGION_THREADS}u) {{ \
         if (m3_tid == 0 && m4_group.x == 0) m1_fault(status, {M3_THREADS_EXCEEDED:#X}u); return; }}"
    );
    source.push_str(
        "  const device M1ValueDesc* descriptors = all_descriptors + \
         dispatch_lane * layout->value_count;\n",
    );
    source.push_str(
        "  const device M1OpParams* lane_params = params + \
         dispatch_lane * layout->reserved2;\n",
    );
    source.push_str(
        "  device uchar* scratch = all_scratch + dispatch_lane * layout->scratch_stride;\n",
    );
    source.push_str("  device uchar* temporary = scratch + layout->temporary_offset;\n");
    source.push_str(
        "  const device bfloat* logits = \
         reinterpret_cast<const device bfloat*>(lane.logits_base);\n",
    );
    for channel in 0..channel_count {
        let _ = writeln!(
            source,
            "  const uint dense_{channel} = channel_bindings[dispatch_lane * layout->reserved0 + {channel}];"
        );
        let _ = writeln!(
            source,
            "  const M3LaneChannelSlot channel_{channel} = slots[lane.channel_slot_offset + dense_{channel}];"
        );
        let _ = writeln!(
            source,
            "  const uint pending_index_{channel} = lane.channel_slot_offset + dense_{channel};"
        );
        // Re-derived every dispatch: a put in an earlier step set the flag,
        // and this step reads the cell it points at.
        let _ = writeln!(
            source,
            "  const device uchar* current_{channel} = reinterpret_cast<const device uchar*>(\
             pending_flags[pending_index_{channel}] != 0 ? channel_{channel}.pending_cell : \
             channel_{channel}.committed_cell);"
        );
        let _ = writeln!(
            source,
            "  device uchar* pending_{channel} = reinterpret_cast<device uchar*>(\
             channel_{channel}.pending_cell);"
        );
    }
    source.push_str("  const uint m4_gtid = m4_group.x * m3_threads + m3_tid;\n");
    source.push_str("  const uint m4_gthreads = m4_groups.x * m3_threads;\n");

    // The same view/alias decisions as the grouped emitter, so the two forms
    // read the same values at the same offsets.
    let escapes = crate::codegen::alias::escaping_values(region);
    let covers =
        |source: u32, result: u32| crate::codegen::alias::covers(value_types, source, result);
    let is_view_reshape = |node: usize| -> bool {
        let Some(op) = ops.get(node) else {
            return false;
        };
        op.tag == tags::RESHAPE
            && op.results == 1
            && op.args.len() == 1
            && !escapes.contains(&bases[node])
            && covers(op.args[0], bases[node])
    };
    let mut alias = crate::codegen::alias::AliasTable::new();
    for &node in &region.nodes {
        let node = node.index();
        if is_view_reshape(node) {
            let arg = ops[node].args[0];
            alias.elide(bases[node], arg);
        }
    }
    let mut consumers: BTreeMap<u32, usize> = BTreeMap::new();
    for &node in &region.nodes {
        if is_view_reshape(node.index()) {
            continue;
        }
        if let Some(op) = ops.get(node.index()) {
            for &arg in &op.args {
                *consumers.entry(alias.resolve(arg)).or_insert(0) += 1;
            }
        }
    }
    let mut fused_argmax: BTreeMap<usize, usize> = BTreeMap::new();
    let mut elided_gather: BTreeSet<usize> = BTreeSet::new();
    for &node in &region.nodes {
        let node = node.index();
        let Some(op) = ops.get(node) else { continue };
        if op.tag != tags::REDUCE_ARGMAX || op.args.len() != 1 {
            continue;
        }
        let source_value = alias.resolve(op.args[0]);
        if consumers.get(&source_value).copied().unwrap_or(0) != 1
            || escapes.contains(&source_value)
        {
            continue;
        }
        let producer = region.nodes.iter().map(|n| n.index()).find(|&n| {
            bases[n] == source_value
                && ops.get(n).is_some_and(|p| {
                    p.tag == tags::INTRINSIC_VAL
                        && (p.intr == intrinsic_tags::LOGITS
                            || p.intr == intrinsic_tags::MTP_LOGITS)
                })
        });
        if let Some(producer) = producer {
            fused_argmax.insert(node, producer);
            elided_gather.insert(producer);
        }
    }
    let value_ptr = |value: u32| format!("scratch + offsets[{value}]");
    let argmax_over_f32 = |op: &OpView| -> bool {
        op.args
            .first()
            .and_then(|&arg| value_types.get(alias.resolve(arg) as usize))
            .is_some_and(|ty| ty.dtype == eta_ir::Dtype::F32)
    };

    // What each node becomes, before any grouping.
    enum Body {
        /// A hand-emitted threadgroup pattern (the fused logits argmax).
        Single(String),
        /// A gather: strided over the grid, its own block.
        Gather(String),
        /// A direct element-independent op: strides, then per-element code.
        /// `len` is the element count the loop runs to; two shapes of one
        /// length share a loop. `reads` are the values it consumes and
        /// `writes` the one it produces; `remaps` says it reads an operand at
        /// an index other than its own (a broadcast), which is a cross-thread
        /// dependency if that operand was written in the same loop.
        Direct {
            pre: String,
            inner: String,
            /// The result's symbolic dims with the unit ones dropped: two
            /// values with the same key have the same element count at run
            /// time, whatever their ranks (`[1, V]` and `[V]` share a loop).
            /// The wire shape is not this — it is filled only for a few ops.
            len: Vec<crate::plan::Dimension>,
            reads: Vec<u32>,
            writes: u32,
            remaps: bool,
        },
        /// A generic op through the runtime, strided by the grid.
        Generic(String),
        /// A put through the runtime, plus its flag.
        Put(String),
        /// A stateful walk on one threadgroup.
        Walk(String),
        Reduce(String),
        Argmax(String),
    }
    let mut planned: Vec<(u32, Body)> = Vec::new();
    for &node in &region.nodes {
        let node = node.index();
        let Some(op) = ops.get(node) else {
            return Err(EmitError::RegionNodeOutOfRange(RegionForm::GroupedFused));
        };
        let base = bases[node];
        if elided_gather.contains(&node) || is_view_reshape(node) {
            continue;
        }
        let node_u32 = u32::try_from(node).map_err(|_| {
            EmitError::RegionNodeOutOfRange(RegionForm::GroupedFused)
        })?;
        if let Some(&producer) = fused_argmax.get(&node) {
            let slots = Slots::of(op, base, |value| value_ptr(alias.resolve(value)));
            let mut body = String::new();
            emit_logits_argmax(
                &mut body,
                bases[producer],
                ops[producer].intr == intrinsic_tags::MTP_LOGITS,
                &slots.o0,
            );
            planned.push((node_u32, Body::Single(body)));
            continue;
        }
        let mut slots = Slots::of(op, base, |value| value_ptr(alias.resolve(value)));
        if op.tag == tags::INTRINSIC_VAL && op.intr == intrinsic_tags::MTP_DRAFTS {
            let mut body = String::new();
            emit_mtp_drafts(&mut body, base, &slots.o0, "m4_gtid", "m4_gthreads");
            planned.push((node_u32, Body::Gather(body)));
            continue;
        }
        if op.tag == tags::INTRINSIC_VAL && op.intr == intrinsic_tags::ATTN_SCORE {
            let mut body = String::new();
            emit_score_gather(&mut body, base, &slots.o0, "m4_gtid", "m4_gthreads");
            planned.push((node_u32, Body::Gather(body)));
            continue;
        }
        if op.tag == tags::INTRINSIC_VAL
            && (op.intr == intrinsic_tags::LOGITS || op.intr == intrinsic_tags::MTP_LOGITS)
        {
            let mut body = String::new();
            emit_logits_gather(
                &mut body,
                base,
                op.intr == intrinsic_tags::MTP_LOGITS,
                &slots.o0,
                "m4_gtid",
                "m4_gthreads",
            );
            planned.push((node_u32, Body::Gather(body)));
            continue;
        }
        if op.tag == tags::CHAN_TAKE || op.tag == tags::CHAN_READ {
            slots.a0 = format!("current_{}", op.chan);
        } else if op.tag == tags::CHAN_PUT {
            slots.o0 = format!("pending_{}", op.chan);
        } else if op.tag == tags::INTRINSIC_VAL {
            // See the grouped emitter: `logits` is typed `bfloat*` for the
            // gathers above and the runtime takes `uchar*`.
            slots.a0 = "reinterpret_cast<const device uchar*>(logits)".to_string();
        }
        let kind = op_step_kind(op.tag, op.pred_tag, argmax_over_f32(op));
        let generic_wide = format!(
            "    ptir_m1_execute_part({}u, status, descriptors, lane_params + {node}, {}, {}, {}, {}, {}, temporary, m4_gtid, m4_gthreads);\n",
            op.tag, slots.a0, slots.a1, slots.a2, slots.o0, slots.o1
        );
        let body = match kind {
            StepKind::Wide if op.tag == tags::CHAN_PUT => Body::Put(format!(
                "{generic_wide}    if (m4_gtid == 0) pending_flags[pending_index_{}] = 1;\n",
                op.chan
            )),
            StepKind::Wide => match direct_wide(op, node, base, &slots, value_types, &alias) {
                Some((pre, inner)) => Body::Direct {
                    pre,
                    inner,
                    len: value_types
                        .get(base as usize)
                        .map(|ty| {
                            ty.dims
                                .iter()
                                .filter(|dim| **dim != crate::plan::Dimension::Static(1))
                                .cloned()
                                .collect()
                        })
                        .unwrap_or_default(),
                    reads: op.args.iter().map(|&a| alias.resolve(a)).collect(),
                    writes: base,
                    remaps: op.tag == tags::BROADCAST,
                },
                None => Body::Generic(generic_wide),
            },
            StepKind::Single => {
                let mut body = String::from("    if (m4_group.x != 0) return;\n");
                let _ = writeln!(
                    body,
                    "    ptir_m1_execute_mt({}u, status, descriptors, lane_params + {node}, {}, {}, {}, {}, {}, temporary, m3_tid, m3_threads, m3_tgbuf);",
                    op.tag, slots.a0, slots.a1, slots.a2, slots.o0, slots.o1
                );
                if op.tag == tags::CHAN_PUT {
                    let _ = writeln!(
                        body,
                        "    if (m3_tid == 0) pending_flags[pending_index_{}] = 1;",
                        op.chan
                    );
                }
                Body::Walk(body)
            }
            StepKind::Reduce => Body::Reduce(format!(
                "    m4_reduce_two_levels({}u, {}, {}, temporary, descriptors[lane_params[{node}].a0], step.level, m4_group.x, m4_groups.x, m3_tid, m3_threads, m3_tgbuf);\n",
                op.tag, slots.a0, slots.o0
            )),
            StepKind::Argmax => Body::Argmax(format!(
                "    if (step.level == 0u) m4_argmax_partial({}, temporary, descriptors[lane_params[{node}].a0], m4_group.x, m4_groups.x, m3_tid, m3_threads, m3_tgbuf);\n    else {{ if (m4_group.x != 0) return; m4_argmax_final(temporary, {}, descriptors[lane_params[{node}].a0], step.groups, m3_tid, m3_threads, m3_tgbuf); }}\n",
                slots.a0, slots.o0
            )),
        };
        planned.push((node_u32, body));
    }

    // Grouping: a run of direct ops of one shape is one loop — each thread
    // computes element `i` of every op in order, so an op may read what an
    // earlier op of the run wrote at `i` (same shape, same thread, program
    // order) and nothing else in the run. A run of puts is one dispatch too:
    // puts write disjoint cells. Everything else is its own dispatch.
    let mut steps = Vec::new();
    source.push_str("  switch (step.index) {\n");
    let mut at = 0usize;
    while at < planned.len() {
        let (first, body) = &planned[at];
        match body {
            Body::Direct { len, writes, .. } => {
                let len = len.clone();
                let mut produced: Vec<u32> = vec![*writes];
                let mut end = at + 1;
                while end < planned.len() {
                    let Body::Direct {
                        len: other,
                        reads,
                        writes,
                        remaps,
                        ..
                    } = &planned[end].1
                    else {
                        break;
                    };
                    if *other != len {
                        break;
                    }
                    // A remapping read of a value this loop writes would read
                    // another thread's element before it is there.
                    if *remaps && reads.iter().any(|value| produced.contains(value)) {
                        break;
                    }
                    produced.push(*writes);
                    end += 1;
                }
                let _ = writeln!(source, "  case {first}u: {{");
                let mut widest = String::new();
                for (_, member) in &planned[at..end] {
                    if let Body::Direct { pre, .. } = member {
                        source.push_str(pre);
                        if widest.is_empty() {
                            widest = pre
                                .lines()
                                .next()
                                .and_then(|line| line.split_whitespace().nth(2))
                                .unwrap_or("m4_n_0")
                                .to_string();
                        }
                    }
                }
                // The loop runs to the run's shape; every member has it.
                let _ = writeln!(
                    source,
                    "    for (uint i = m4_gtid; i < {widest}; i += m4_gthreads) {{"
                );
                for (_, member) in &planned[at..end] {
                    if let Body::Direct { inner, .. } = member {
                        source.push_str(inner);
                    }
                }
                source.push_str("    }\n    return;\n  }\n");
                steps.push(streamed_step(*first, StepKind::Wide));
                at = end;
            }
            Body::Put(_) => {
                let mut end = at + 1;
                while end < planned.len() && matches!(&planned[end].1, Body::Put(_)) {
                    end += 1;
                }
                let _ = writeln!(source, "  case {first}u: {{");
                for (_, member) in &planned[at..end] {
                    if let Body::Put(text) = member {
                        source.push_str(text);
                    }
                }
                source.push_str("    return;\n  }\n");
                steps.push(streamed_step(*first, StepKind::Wide));
                at = end;
            }
            Body::Single(text) | Body::Walk(text) => {
                let _ = writeln!(source, "  case {first}u: {{");
                source.push_str(text);
                source.push_str("    return;\n  }\n");
                steps.push(streamed_step(*first, StepKind::Single));
                at += 1;
            }
            Body::Gather(text) | Body::Generic(text) => {
                let _ = writeln!(source, "  case {first}u: {{");
                source.push_str(text);
                source.push_str("    return;\n  }\n");
                steps.push(streamed_step(*first, StepKind::Wide));
                at += 1;
            }
            Body::Reduce(text) => {
                let _ = writeln!(source, "  case {first}u: {{");
                source.push_str(text);
                source.push_str("    return;\n  }\n");
                steps.push(streamed_step(*first, StepKind::Reduce));
                at += 1;
            }
            Body::Argmax(text) => {
                let _ = writeln!(source, "  case {first}u: {{");
                source.push_str(text);
                source.push_str("    return;\n  }\n");
                steps.push(streamed_step(*first, StepKind::Argmax));
                at += 1;
            }
        }
    }
    source.push_str("  default: return;\n  }\n}\n");
    Ok((source, steps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tree_has_the_levels_the_runtime_walks() {
        assert_eq!(reduce_levels(0), 1);
        assert_eq!(reduce_levels(1), 1);
        assert_eq!(reduce_levels(32), 1);
        assert_eq!(reduce_levels(33), 2);
        assert_eq!(reduce_levels(1024), 2);
        assert_eq!(reduce_levels(1025), 3);
        assert_eq!(reduce_levels(248_320), 4);
    }

    #[test]
    fn a_step_round_trips() {
        let step = streamed_step(41, StepKind::Reduce);
        assert_eq!(step_node(step), 41);
        assert_eq!(step_kind(step), Some(StepKind::Reduce));
        assert_eq!(step_kind(0xFF), None);
    }

    #[test]
    fn the_walks_with_state_stay_on_one_threadgroup() {
        assert_eq!(op_step_kind(tags::EXP, 0, false), StepKind::Wide);
        assert_eq!(op_step_kind(tags::CUMSUM, 0, false), StepKind::Single);
        assert_eq!(
            op_step_kind(tags::PIVOT_THRESHOLD, predicate_tags::CUMMASS_LE, false),
            StepKind::Single
        );
        assert_eq!(
            op_step_kind(tags::PIVOT_THRESHOLD, predicate_tags::PROB_GE, false),
            StepKind::Wide
        );
        assert_eq!(op_step_kind(tags::REDUCE_SUM, 0, false), StepKind::Reduce);
        assert_eq!(op_step_kind(tags::REDUCE_ARGMAX, 0, true), StepKind::Argmax);
        assert_eq!(op_step_kind(tags::REDUCE_ARGMAX, 0, false), StepKind::Single);
    }
}
