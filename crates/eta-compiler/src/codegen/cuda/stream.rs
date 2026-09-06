//! **STREAMS** — a row block's elementwise ops fused into one loop.
//!
//! A row-parallel region runs one block per row, and until now every op in
//! it was its own loop over the row: `exp` read the row and wrote it, the
//! `div` after it read that and wrote again, the `reduce_sum` read it once
//! more. Over a `[256, 262144]` epilogue that is a 268 MB round trip per op,
//! and a sampler has a dozen of them — the epilogue's whole cost was that
//! traffic, not its arithmetic.
//!
//! A stream is a maximal run of consecutive region ops that can be evaluated
//! in ONE pass over the row's elements: the elementwise ops (`exp`, `div`,
//! `gt`, `select`, a scalar's `broadcast`, `rng`, ...) and the reductions
//! that read them (`reduce_sum/max/min/argmax`). Inside the pass each
//! intermediate lives in a register; it is stored to its scratch slot only
//! if something outside the stream reads it (a later op, another region, a
//! channel). A reduction's result is complete only after the pass, so an op
//! that reads one starts the next stream.
//!
//! The arithmetic is the runtime helpers' own, mirrored expression for
//! expression (`ptir_parallel_elementwise`, `ptir_parallel_reduce_f32`,
//! `ptir_fast_argmax` in `fused_block0.cuh`): the same loaders and
//! conversions, the same NaN-canonical max and min, the same argmax
//! candidate combine. What differs is the order a sum's terms meet — each
//! thread folds its strided elements first — so a fused `reduce_sum` may
//! round a last bit differently from the two-launch path.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write as _;

use eta_ir::Dtype;
use eta_ir::op::{intrinsic_tags, tags};

use crate::codegen::op_view::OpView;
use crate::plan::{CompiledStage, Dimension, Region};

/// A value's shape as a row block sees it.
#[derive(Clone, PartialEq, Eq)]
enum RowClass {
    /// The block's row of a value of the region's geometry: `width` elements.
    Full(Dimension),
    /// One element: a per-row vector's element, or a whole scalar.
    Scalar,
    /// Anything a stream does not index by element.
    Other,
}

/// What a stream does with one op.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// Elementwise: a register per element.
    Map,
    /// A scalar spread over the row: the register IS the scalar.
    Broadcast,
    /// A draw per element from a seed fixed before the pass.
    Rng,
    /// `reduce_sum/max/min`: an accumulator folded after the pass.
    Reduce,
    /// `reduce_argmax`: a candidate folded after the pass.
    Argmax,
    /// An intrinsic (the logits) read straight off its plane, element by
    /// element — no copy into scratch unless something outside reads it.
    Intrinsic,
}

/// One op of a stream.
struct Member {
    node: usize,
    role: Role,
}

/// The stage facts a stream is planned and emitted against.
pub(super) struct Streams<'a> {
    pub stage: &'a CompiledStage,
    pub region: &'a Region,
    pub ops: &'a [OpView],
    pub bases: &'a [u32],
    /// Per value: 1 = of the geometry, 2 = a per-row vector, 0 = whole.
    pub kinds: &'a [u8],
    /// `direct_intrinsic[node] != u16::MAX` marks an argmax the emitter
    /// answers straight off the intrinsic; a stream leaves it alone.
    pub direct_intrinsic: &'a [u16],
    /// Nodes the emitter skips outright.
    pub skipped: &'a [u8],
    /// Per value, every stage node that reads it.
    pub readers: Vec<Vec<usize>>,
}

impl<'a> Streams<'a> {
    pub fn new(
        stage: &'a CompiledStage,
        region: &'a Region,
        ops: &'a [OpView],
        bases: &'a [u32],
        kinds: &'a [u8],
        direct_intrinsic: &'a [u16],
        skipped: &'a [u8],
    ) -> Self {
        let mut readers = vec![Vec::new(); stage.normalized.value_types.len()];
        for (node, op) in ops.iter().enumerate() {
            for &arg in &op.args {
                if let Some(list) = readers.get_mut(arg as usize) {
                    list.push(node);
                }
            }
        }
        Self {
            stage,
            region,
            ops,
            bases,
            kinds,
            direct_intrinsic,
            skipped,
            readers,
        }
    }

    fn dtype(&self, value: u32) -> Dtype {
        self.stage.normalized.value_types[value as usize].dtype
    }

    fn class(&self, value: u32) -> RowClass {
        let ty = &self.stage.normalized.value_types[value as usize];
        match self.kinds.get(value as usize).copied().unwrap_or(0) {
            1 => ty
                .dims
                .last()
                .map(|d| RowClass::Full(d.clone()))
                .unwrap_or(RowClass::Other),
            2 => RowClass::Scalar,
            _ => {
                if ty.dims.iter().all(|d| *d == Dimension::Static(1)) {
                    RowClass::Scalar
                } else {
                    RowClass::Other
                }
            }
        }
    }

    /// The role an op takes in a stream of `width`, or `None` when it ends
    /// one. `width` is fixed by the stream's first op; `None` there means
    /// the op sets it.
    fn role(&self, node: usize, width: &mut Option<Dimension>) -> Option<Role> {
        if self.skipped[node] != 0 {
            return None;
        }
        let op = &self.ops[node];
        let out = self.bases[node];
        let out_dtype = self.dtype(out);
        if !matches!(out_dtype, Dtype::F32 | Dtype::I32 | Dtype::U32 | Dtype::Bool) {
            return None;
        }
        let full = |value: u32, width: &mut Option<Dimension>| -> bool {
            match self.class(value) {
                RowClass::Full(dim) => match width {
                    Some(w) => *w == dim,
                    None => {
                        *width = Some(dim);
                        true
                    }
                },
                _ => false,
            }
        };
        let indexed = |value: u32, width: &mut Option<Dimension>| -> bool {
            matches!(self.class(value), RowClass::Scalar) || full(value, width)
        };
        let scalar_dtypes = |values: &[u32]| {
            values
                .iter()
                .all(|&v| matches!(self.dtype(v), Dtype::F32 | Dtype::I32 | Dtype::U32 | Dtype::Bool))
        };
        let tag = op.tag;
        match tag {
            tags::EXP | tags::LOG | tags::RECIP | tags::CAST | tags::NOT => {
                if op.args.len() != 1 || !scalar_dtypes(&op.args) {
                    return None;
                }
                if tag == tags::NOT && self.dtype(op.args[0]) != Dtype::Bool {
                    return None;
                }
                let mut w = width.clone();
                if !full(out, &mut w) || !indexed(op.args[0], &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Map)
            }
            tags::NEG | tags::ABS | tags::SIGN => {
                if op.args.len() != 1 || self.dtype(op.args[0]) == Dtype::Bool || !scalar_dtypes(&op.args) {
                    return None;
                }
                let mut w = width.clone();
                if !full(out, &mut w) || !indexed(op.args[0], &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Map)
            }
            tags::ADD
            | tags::SUB
            | tags::MUL
            | tags::DIV
            | tags::MAX_ELEM
            | tags::MIN_ELEM
            | tags::REM
            | tags::GT
            | tags::GE
            | tags::EQ
            | tags::NE
            | tags::LT
            | tags::LE
            | tags::AND
            | tags::OR => {
                if op.args.len() != 2 || !scalar_dtypes(&op.args) {
                    return None;
                }
                let logic = matches!(tag, tags::AND | tags::OR);
                let left = self.dtype(op.args[0]);
                if logic && (left != Dtype::Bool || self.dtype(op.args[1]) != Dtype::Bool) {
                    return None;
                }
                if !logic && left == Dtype::Bool {
                    return None;
                }
                let mut w = width.clone();
                if !full(out, &mut w) || !indexed(op.args[0], &mut w) || !indexed(op.args[1], &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Map)
            }
            tags::SELECT => {
                if op.args.len() != 3 || !scalar_dtypes(&op.args) || self.dtype(op.args[0]) != Dtype::Bool {
                    return None;
                }
                let mut w = width.clone();
                if !full(out, &mut w) || !op.args.iter().all(|&a| indexed(a, &mut w)) {
                    return None;
                }
                *width = w;
                Some(Role::Map)
            }
            tags::BROADCAST => {
                if op.args.len() != 1 || !scalar_dtypes(&op.args) || self.class(op.args[0]) != RowClass::Scalar {
                    return None;
                }
                let mut w = width.clone();
                if !full(out, &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Broadcast)
            }
            tags::RNG | tags::RNG_KEYED => {
                if out_dtype != Dtype::F32 {
                    return None;
                }
                if tag == tags::RNG_KEYED && (op.args.len() != 1 || self.dtype(op.args[0]) != Dtype::U32) {
                    return None;
                }
                let mut w = width.clone();
                if !full(out, &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Rng)
            }
            tags::INTRINSIC_VAL => {
                // The fallback intrinsics (`layer`, `mtp_drafts`) stay with
                // the helper; a float plane of the row geometry streams.
                if op.intr == intrinsic_tags::LAYER || op.intr == intrinsic_tags::MTP_DRAFTS {
                    return None;
                }
                if out_dtype != Dtype::F32 {
                    return None;
                }
                let mut w = width.clone();
                if !full(out, &mut w) {
                    return None;
                }
                *width = w;
                Some(Role::Intrinsic)
            }
            tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN | tags::REDUCE_ARGMAX => {
                if op.args.len() != 1 || self.dtype(op.args[0]) != Dtype::F32 {
                    return None;
                }
                if tag == tags::REDUCE_ARGMAX && self.direct_intrinsic[node] != u16::MAX {
                    return None;
                }
                if self.class(out) != RowClass::Scalar {
                    return None;
                }
                let mut w = width.clone();
                if !full(op.args[0], &mut w) {
                    return None;
                }
                *width = w;
                Some(if tag == tags::REDUCE_ARGMAX { Role::Argmax } else { Role::Reduce })
            }
            _ => None,
        }
    }

    /// The stream starting at position `at` of the region's node list:
    /// its members in order, or `None` when that op is not a stream's.
    fn plan(&self, at: usize) -> Option<Vec<Member>> {
        let mut width: Option<Dimension> = None;
        let mut members: Vec<Member> = Vec::new();
        let mut reduced: Vec<u32> = Vec::new();
        for &index in &self.region.nodes[at..] {
            let node = index.index();
            let op = &self.ops[node];
            // An op that reads a reduction of this stream needs the pass
            // finished first.
            if op.args.iter().any(|a| reduced.contains(a)) {
                break;
            }
            let Some(role) = self.role(node, &mut width) else {
                break;
            };
            if matches!(role, Role::Reduce | Role::Argmax) {
                reduced.push(self.bases[node]);
            }
            members.push(Member { node, role });
        }
        if members.is_empty() { None } else { Some(members) }
    }

    /// Whether a value a stream's op defines must also land in scratch.
    fn escapes(&self, value: u32, members: &[Member]) -> bool {
        self.region.outputs.contains(&value)
            || self
                .readers
                .get(value as usize)
                .is_some_and(|list| list.iter().any(|node| !members.iter().any(|m| m.node == *node)))
    }
}

/// C spelling of a register of `dtype`.
fn c_type(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "float",
        Dtype::I32 => "int",
        Dtype::U32 => "m1_u32",
        _ => "bool",
    }
}

/// The loader letter for `dtype`: `m1_load_f/i/u/b`.
fn letter(dtype: Dtype) -> char {
    match dtype {
        Dtype::F32 => 'f',
        Dtype::I32 => 'i',
        Dtype::U32 => 'u',
        _ => 'b',
    }
}

/// The wire code of `dtype`, what a descriptor's `dtype` field holds.
fn code(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::F32 => 0,
        Dtype::I32 => 1,
        Dtype::U32 => 2,
        _ => 3,
    }
}

/// `expr` (a register of `from`) read as `to` — exactly what the typed
/// loader `m1_load_<to>` makes of a slot of `from`.
fn convert(expr: &str, from: Dtype, to: Dtype) -> String {
    if from == to {
        return expr.into();
    }
    match (to, from) {
        (Dtype::F32, Dtype::I32) | (Dtype::F32, Dtype::U32) => format!("float({expr})"),
        (Dtype::F32, _) => format!("({expr} ? 1.0f : 0.0f)"),
        (Dtype::I32, Dtype::F32) => format!("m1_float_to_i32({expr})"),
        (Dtype::I32, Dtype::U32) => format!("m1_bits_i32({expr})"),
        (Dtype::I32, _) => format!("({expr} ? 1 : 0)"),
        (Dtype::U32, Dtype::F32) => format!("m1_float_to_u32({expr})"),
        (Dtype::U32, Dtype::I32) => format!("(m1_u32)({expr})"),
        (Dtype::U32, _) => format!("({expr} ? 1u : 0u)"),
        (_, Dtype::F32) => format!("({expr} != 0.0f)"),
        (_, Dtype::I32) => format!("({expr} != 0)"),
        (_, _) => format!("({expr} != 0u)"),
    }
}

/// A read of `value` as `want` inside the pass: a register's conversion, a
/// hoisted scalar (`s<value>_<t>`, loaded once before the pass), or the
/// element's load (`l<value>_<t>`, once per pass).
#[allow(clippy::too_many_arguments)]
fn read(
    streams: &Streams<'_>,
    value: u32,
    want: Dtype,
    registers: &[(u32, Dtype)],
    hoisted: &mut Vec<(u32, Dtype)>,
    loaded: &mut Vec<(u32, Dtype)>,
    prologue: &mut String,
    body: &mut String,
    pointer: &mut dyn FnMut(u32) -> String,
) -> String {
    if let Some((_, from)) = registers.iter().find(|(v, _)| *v == value) {
        return convert(&format!("r{value}"), *from, want);
    }
    let from = streams.dtype(value);
    let l = letter(want);
    match streams.class(value) {
        RowClass::Scalar => {
            let name = format!("s{value}_{l}");
            if !hoisted.contains(&(value, want)) {
                hoisted.push((value, want));
                let _ = writeln!(
                    prologue,
                    "    const {} {name} = m1_load_{l}({}, 0u, {}u);",
                    c_type(want),
                    pointer(value),
                    code(from)
                );
            }
            name
        }
        _ => {
            let name = format!("l{value}_{l}");
            if !loaded.contains(&(value, want)) {
                loaded.push((value, want));
                let _ = writeln!(
                    body,
                    "      const {} {name} = m1_load_{l}({}, i, {}u);",
                    c_type(want),
                    pointer(value),
                    code(from)
                );
            }
            name
        }
    }
}

/// Emit the stream at `at` — when there is one — into `source`, returning
/// how many of the region's nodes it covered. `pointer(value)` spells a
/// value's scratch pointer (aliases resolved); `tail` is the per-op
/// barrier-and-status block the caller appends after every op.
pub(super) fn emit_stream(
    source: &mut String,
    streams: &Streams<'_>,
    at: usize,
    pointer: &mut dyn FnMut(u32) -> String,
    tail: &str,
) -> Option<usize> {
    let members = streams.plan(at)?;
    let ops = streams.ops;
    let bases = streams.bases;

    // The row's width: any member's full value (all agree by planning).
    let witness = members
        .iter()
        .find_map(|m| {
            let op = &ops[m.node];
            let out = bases[m.node];
            if matches!(streams.class(out), RowClass::Full(_)) {
                Some(out)
            } else {
                op.args.iter().copied().find(|&a| matches!(streams.class(a), RowClass::Full(_)))
            }
        })
        .expect("a stream has a full value");

    let mut s = String::new();
    let _ = writeln!(s, "  {{ // stream of {} op(s)", members.len());
    s.push_str("    const m1_u32 stream_lane = threadIdx.x & 31u;\n");
    s.push_str("    const m1_u32 stream_warp = threadIdx.x >> 5u;\n");
    s.push_str("    const m1_u32 stream_warps = blockDim.x >> 5u;\n");
    s.push_str("    float* stream_work = reinterpret_cast<float*>(temporary);\n");
    let _ = writeln!(s, "    const m1_u32 stream_width = descriptors[{witness}u].len;");

    // Registers: value -> (name, dtype) for values the pass defines.
    let mut registers: Vec<(u32, Dtype)> = Vec::new();
    // Hoisted scalar loads and the per-element loads already spelled: (value, as-dtype).
    let mut hoisted: Vec<(u32, Dtype)> = Vec::new();
    let mut loaded: Vec<(u32, Dtype)> = Vec::new();
    let mut prologue = String::new();
    let mut body = String::new();
    let mut epilogue = String::new();
    let mut float_reductions = 0u32;
    let mut argmaxes = 0u32;

    for member in &members {
        let node = member.node;
        let op = &ops[node];
        let out = bases[node];
        let out_dtype = streams.dtype(out);
        let tag = op.tag;
        match member.role {
            Role::Map | Role::Broadcast => {
                let expr: String = if tag == tags::BROADCAST {
                    read(streams, op.args[0], out_dtype, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer)
                } else if matches!(tag, tags::EXP | tags::LOG | tags::RECIP) {
                    let x = read(streams, op.args[0], Dtype::F32, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    match tag {
                        tags::EXP => format!("expf({x})"),
                        tags::LOG => format!("logf({x})"),
                        _ => format!("(1.0f / {x})"),
                    }
                } else if tag == tags::CAST {
                    read(streams, op.args[0], out_dtype, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer)
                } else if tag == tags::NOT {
                    let x = read(streams, op.args[0], Dtype::Bool, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    format!("(!{x})")
                } else if matches!(tag, tags::NEG | tags::ABS | tags::SIGN) {
                    let path = streams.dtype(op.args[0]);
                    let x = read(streams, op.args[0], path, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    match (tag, path) {
                        (tags::NEG, Dtype::F32) => format!("(-{x})"),
                        (tags::ABS, Dtype::F32) => format!("fabsf({x})"),
                        (tags::SIGN, Dtype::F32) => {
                            format!("({x} > 0.0f ? 1.0f : ({x} < 0.0f ? -1.0f : 0.0f))")
                        }
                        (tags::NEG, Dtype::I32) => format!("(int)(0u - (m1_u32){x})"),
                        (tags::ABS, Dtype::I32) => {
                            format!("((m1_u32){x} == 0x80000000u ? {x} : ({x} < 0 ? -{x} : {x}))")
                        }
                        (tags::SIGN, Dtype::I32) => format!("({x} > 0 ? 1 : ({x} < 0 ? -1 : 0))"),
                        (tags::NEG, _) => format!("(0u - {x})"),
                        (tags::ABS, _) => x,
                        (_, _) => format!("({x} != 0u ? 1u : 0u)"),
                    }
                } else if tag == tags::SELECT {
                    let c = read(streams, op.args[0], Dtype::Bool, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    let a = read(streams, op.args[1], out_dtype, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    let b = read(streams, op.args[2], out_dtype, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    format!("({c} ? {a} : {b})")
                } else if matches!(tag, tags::AND | tags::OR) {
                    let a = read(streams, op.args[0], Dtype::Bool, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    let b = read(streams, op.args[1], Dtype::Bool, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    if tag == tags::AND { format!("({a} && {b})") } else { format!("({a} || {b})") }
                } else {
                    // Binary arithmetic and compares: the left operand's
                    // dtype picks the path, as the helper's `d0.dtype` does.
                    let path = streams.dtype(op.args[0]);
                    let a = read(streams, op.args[0], path, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    let b = read(streams, op.args[1], path, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                    match (tag, path) {
                        (tags::GT, _) => format!("({a} > {b})"),
                        (tags::GE, _) => format!("({a} >= {b})"),
                        (tags::EQ, _) => format!("({a} == {b})"),
                        (tags::NE, _) => format!("({a} != {b})"),
                        (tags::LT, _) => format!("({a} < {b})"),
                        (tags::LE, _) => format!("({a} <= {b})"),
                        (tags::ADD, Dtype::F32) => format!("({a} + {b})"),
                        (tags::SUB, Dtype::F32) => format!("({a} - {b})"),
                        (tags::MUL, Dtype::F32) => format!("({a} * {b})"),
                        (tags::DIV, Dtype::F32) => format!("({a} / {b})"),
                        (tags::MAX_ELEM, Dtype::F32) => format!("m1_element_max({a}, {b})"),
                        (tags::MIN_ELEM, Dtype::F32) => format!("m1_element_min({a}, {b})"),
                        (_, Dtype::F32) => format!("fmodf({a}, {b})"),
                        (tags::ADD, Dtype::I32) => format!("(int)((m1_u32){a} + (m1_u32){b})"),
                        (tags::SUB, Dtype::I32) => format!("(int)((m1_u32){a} - (m1_u32){b})"),
                        (tags::MUL, Dtype::I32) => format!("(int)((m1_u32){a} * (m1_u32){b})"),
                        (tags::DIV, Dtype::I32) => format!("m1_i32_div({a}, {b})"),
                        (tags::MAX_ELEM, Dtype::I32) => format!("({a} > {b} ? {a} : {b})"),
                        (tags::MIN_ELEM, Dtype::I32) => format!("({a} < {b} ? {a} : {b})"),
                        (_, Dtype::I32) => format!("m1_i32_rem({a}, {b})"),
                        (tags::ADD, _) => format!("({a} + {b})"),
                        (tags::SUB, _) => format!("({a} - {b})"),
                        (tags::MUL, _) => format!("({a} * {b})"),
                        (tags::DIV, _) => format!("({b} == 0u ? 0u : {a} / {b})"),
                        (tags::MAX_ELEM, _) => format!("({a} > {b} ? {a} : {b})"),
                        (tags::MIN_ELEM, _) => format!("({a} < {b} ? {a} : {b})"),
                        (_, _) => format!("({b} == 0u ? 0u : {a} % {b})"),
                    }
                };
                let _ = writeln!(body, "      const {} r{out} = {expr};", c_type(out_dtype));
                registers.push((out, out_dtype));
                if streams.escapes(out, &members) {
                    let _ = writeln!(body, "      m1_store_{}({}, i, r{out});", letter(out_dtype), pointer(out));
                }
            }
            Role::Intrinsic => {
                // The emitter's own preamble for an intrinsic op (`fused.rs`),
                // then `ptir_parallel_intrinsic`'s row arithmetic per element.
                let slots = super::fused::PTIR_INTRINSIC_SLOTS;
                let _ = writeln!(prologue, "    M1OpParams p{node} = params[{node}u];");
                let _ = writeln!(prologue, "    p{node}.rng_seed = 0u;");
                let _ = writeln!(
                    prologue,
                    "    const m1_u32 intrinsic_index{node} = dispatch_lane * {slots}u + p{node}.intr;"
                );
                let _ = writeln!(prologue, "    p{node}.intrinsic_dtype = intrinsic_modes[intrinsic_index{node}];");
                let _ = writeln!(prologue, "    p{node}.imm = intrinsic_widths[intrinsic_index{node}];");
                let _ = writeln!(prologue, "    p{node}.intrinsic_row_stride = intrinsic_strides[intrinsic_index{node}];");
                let _ = writeln!(
                    prologue,
                    "    p{node}.intrinsic_row_offset = intrinsic_offsets[intrinsic_index{node}] + lane_row;"
                );
                let _ = writeln!(
                    prologue,
                    "    const m1_u8* ibase{node} = reinterpret_cast<const m1_u8*>(intrinsic_bases[intrinsic_index{node}]);"
                );
                let _ = writeln!(prologue, "    const M1ValueDesc idesc{node} = descriptors[{out}u];");
                let _ = writeln!(
                    prologue,
                    "    const m1_u32 iwidth{node} = idesc{node}.last == 0u ? p{node}.imm : idesc{node}.last;"
                );
                let _ = writeln!(
                    prologue,
                    "    const m1_u32 istride{node} = p{node}.intrinsic_row_stride == 0u ? iwidth{node} : p{node}.intrinsic_row_stride;"
                );
                let _ = writeln!(
                    prologue,
                    "    const m1_u64 ifirst{node} = (m1_u64)p{node}.intrinsic_row_offset + (m1_u64)p{node}.imm2;"
                );
                let _ = writeln!(
                    body,
                    "      const float r{out} = m1_intrinsic_row_load(ibase{node}, ifirst{node} + i / iwidth{node}, i % iwidth{node}, istride{node}, p{node}.intrinsic_dtype);"
                );
                registers.push((out, Dtype::F32));
                if streams.escapes(out, &members) {
                    let _ = writeln!(body, "      m1_store_f({}, i, r{out});", pointer(out));
                }
            }
            Role::Rng => {
                let _ = writeln!(prologue, "    M1OpParams p{node} = params[{node}u];");
                let _ = writeln!(prologue, "    p{node}.rng_seed = 0u;");
                let _ = writeln!(prologue, "    p{node}.imm3 = lane_row * descriptors[p{node}.o0].len;");
                if tag == tags::RNG {
                    let _ = writeln!(
                        prologue,
                        "    const m1_u64 seed{node} = ptir_rng_seed_eff_stream((m1_u32)p{node}.rng_seed, p{node}.imm);"
                    );
                } else {
                    let state = op.args[0];
                    let sp = pointer(state);
                    let _ = writeln!(
                        prologue,
                        "    const m1_u64 seed{node} = ptir_rng_keyed_seed(m1_load_u({sp}, 0u, 2u), descriptors[{state}u].len > 1u ? m1_load_u({sp}, 1u, 2u) : 0u);"
                    );
                }
                let _ = writeln!(
                    body,
                    "      const float u{node} = ptir_rng_hash_uniform(seed{node}, i + p{node}.imm3);"
                );
                let _ = writeln!(
                    body,
                    "      const float r{out} = p{node}.kind == 0u ? u{node} : -logf(-logf(u{node}));"
                );
                registers.push((out, Dtype::F32));
                if streams.escapes(out, &members) {
                    let _ = writeln!(body, "      m1_store_f({}, i, r{out});", pointer(out));
                }
            }
            Role::Reduce => {
                let x = read(streams, op.args[0], Dtype::F32, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                let (identity, combine): (&str, fn(&str, &str) -> String) = match tag {
                    tags::REDUCE_SUM => ("0.0f", |a, b| format!("({a} + {b})")),
                    tags::REDUCE_MAX => ("m1_neg_inf()", |a, b| format!("m1_canonical_max({a}, {b})")),
                    _ => ("m1_pos_inf()", |a, b| format!("m1_canonical_min({a}, {b})")),
                };
                let slot = float_reductions;
                float_reductions += 1;
                let _ = writeln!(prologue, "    float acc{node} = {identity};");
                let _ = writeln!(body, "      acc{node} = {};", combine(&format!("acc{node}"), &x));
                let _ = writeln!(epilogue, "    {{");
                let _ = writeln!(epilogue, "      float v = acc{node};");
                let _ = writeln!(epilogue, "      for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {{");
                let _ = writeln!(epilogue, "        const float other = __shfl_down_sync(0xffffffffu, v, offset);");
                let _ = writeln!(epilogue, "        v = {};", combine("v", "other"));
                let _ = writeln!(epilogue, "      }}");
                let _ = writeln!(
                    epilogue,
                    "      if (stream_lane == 0u) stream_work[{slot}u * 32u + stream_warp] = v;"
                );
                let _ = writeln!(epilogue, "    }}");
                let _ = writeln!(epilogue, "    __syncthreads();");
                let _ = writeln!(epilogue, "    if (threadIdx.x == 0u) {{");
                let _ = writeln!(epilogue, "      float total = {identity};");
                let _ = writeln!(
                    epilogue,
                    "      for (m1_u32 w = 0u; w < stream_warps; ++w) total = {};",
                    combine("total", &format!("stream_work[{slot}u * 32u + w]"))
                );
                let _ = writeln!(epilogue, "      m1_store_f({}, 0u, total);", pointer(out));
                let _ = writeln!(epilogue, "    }}");
            }
            Role::Argmax => {
                let x = read(streams, op.args[0], Dtype::F32, &registers, &mut hoisted, &mut loaded, &mut prologue, &mut body, pointer);
                argmaxes += 1;
                let _ = writeln!(prologue, "    M1ArgmaxCandidate cand{node}{{m1_neg_inf(), 0u, 0u, 0u}};");
                let _ = writeln!(
                    body,
                    "      cand{node} = m1_argmax_combine(cand{node}, M1ArgmaxCandidate{{{x}, i, m1_isnan({x}) ? 0u : 1u, 0u}});"
                );
                let _ = writeln!(epilogue, "    cand{node} = m1_argmax_warp_reduce(cand{node}, stream_lane);");
                let _ = writeln!(epilogue, "    if (stream_lane == 0u) stream_candidates[stream_warp] = cand{node};");
                let _ = writeln!(epilogue, "    __syncthreads();");
                let _ = writeln!(epilogue, "    if (stream_warp == 0u) {{");
                let _ = writeln!(
                    epilogue,
                    "      cand{node} = stream_lane < stream_warps ? stream_candidates[stream_lane] : M1ArgmaxCandidate{{m1_neg_inf(), 0u, 0u, 0u}};"
                );
                let _ = writeln!(epilogue, "      cand{node} = m1_argmax_warp_reduce(cand{node}, stream_lane);");
                let _ = writeln!(
                    epilogue,
                    "      if (stream_lane == 0u) m1_store_i({}, 0u, (int)cand{node}.index);",
                    pointer(out)
                );
                let _ = writeln!(epilogue, "    }}");
            }
        }
        // The next member's fold reuses the work slots and the candidate
        // array: a barrier between folds, the stream's tail after the last.
        if matches!(member.role, Role::Reduce | Role::Argmax) {
            epilogue.push_str("    __syncthreads();\n");
        }
    }

    if argmaxes > 0 {
        s.push_str("    __shared__ M1ArgmaxCandidate stream_candidates[32];\n");
    }
    s.push_str(&prologue);
    s.push_str("    for (m1_u32 i = threadIdx.x; i < stream_width; i += blockDim.x) {\n");
    s.push_str(&body);
    s.push_str("    }\n");
    s.push_str(&epilogue);
    s.push_str(tail);
    s.push_str("  }\n");
    source.push_str(&s);
    Some(members.len())
}

/// The values the CUDA emitter's streams keep in registers for `region`
/// and never store: what `emit_fused_region` plans, without emitting. Empty
/// for a region that is not row-parallel, not generated, or that the
/// emitter declines.
#[must_use]
pub fn spent_values(stage: &CompiledStage, region: &Region) -> Vec<u32> {
    if region.kind != crate::plan::RegionKind::Generated {
        return Vec::new();
    }
    if super::validate::validate_generated_region(stage, region).is_err() {
        return Vec::new();
    }
    let Some(geometry) = super::fused::row_geometry(stage, region) else {
        return Vec::new();
    };
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = crate::codegen::op_view::result_bases(&ops);
    let kinds = super::fused::row_kinds(stage, region, geometry);
    let direct = super::fused::analyze_direct_argmax(stage, region, &bases);
    let streams = Streams::new(stage, region, &ops, &bases, &kinds, &direct.intrinsic, &direct.skipped);
    let mut spent = Vec::new();
    let mut at = 0usize;
    while at < region.nodes.len() {
        let node = region.nodes[at].index();
        at += 1;
        if streams.skipped[node] != 0 {
            continue;
        }
        let Some(members) = streams.plan(at - 1) else {
            continue;
        };
        for member in &members {
            if matches!(member.role, Role::Map | Role::Broadcast | Role::Rng | Role::Intrinsic) {
                let value = bases[member.node];
                if !streams.escapes(value, &members) {
                    spent.push(value);
                }
            }
        }
        at += members.len() - 1;
    }
    spent
}
