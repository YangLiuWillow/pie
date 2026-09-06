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
    /// Per value, the stage node that defines it (`u32::MAX` for none).
    pub producers: Vec<u32>,
    /// Per node, a `top_k` that reads the intrinsic plane straight
    /// (`fused::analyze_direct_topk`): its operand need not be stored for it.
    pub direct_topk: &'a [Option<super::fused::TopKDirect>],
    /// The region's nodes in the order they are emitted (`emission_order`).
    pub order: Vec<usize>,
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
        direct_topk: &'a [Option<super::fused::TopKDirect>],
    ) -> Self {
        let mut readers = vec![Vec::new(); stage.normalized.value_types.len()];
        let mut producers = vec![u32::MAX; stage.normalized.value_types.len()];
        for (node, op) in ops.iter().enumerate() {
            for &arg in &op.args {
                if let Some(list) = readers.get_mut(arg as usize) {
                    list.push(node);
                }
            }
            for result in 0..op.results {
                if let Some(slot) = producers.get_mut((bases[node] + result) as usize) {
                    *slot = node as u32;
                }
            }
        }
        let order = emission_order(ops, bases, region);
        Self {
            stage,
            region,
            ops,
            bases,
            kinds,
            direct_intrinsic,
            skipped,
            readers,
            producers,
            direct_topk,
            order,
        }
    }

    /// Whether a full-row `value` can be recomputed inside a later pass
    /// instead of stored and re-read: it is the intrinsic itself, or a
    /// stream `Map`/`Broadcast` of this region over values that can be
    /// (scalars aside), no more than a few ops deep. Recomputing a divide
    /// costs nothing against a 268 MB round trip through scratch.
    fn rematerializable(&self, value: u32) -> bool {
        self.remat_depth(value, 0)
    }

    fn remat_depth(&self, value: u32, depth: u32) -> bool {
        if depth > 4 {
            return false;
        }
        match self.class(value) {
            RowClass::Scalar => return true,
            RowClass::Full(_) => {}
            RowClass::Other => return false,
        }
        let Some(&node) = self.producers.get(value as usize) else {
            return false;
        };
        if node == u32::MAX || self.skipped[node as usize] != 0 {
            return false;
        }
        let node = node as usize;
        if !self.region.nodes.iter().any(|n| n.index() == node) {
            return false;
        }
        let op = &self.ops[node];
        if op.results != 1 {
            return false;
        }
        if op.tag == eta_ir::op::tags::INTRINSIC_VAL {
            return matches!(self.role(node, &mut None), Some(Role::Intrinsic));
        }
        match self.role(node, &mut None) {
            Some(Role::Map | Role::Broadcast) => op.args.iter().all(|&a| self.remat_depth(a, depth + 1)),
            _ => false,
        }
    }

    /// Whether `node` is a stream member somewhere in this region — a reader
    /// that will recompute a rematerializable value rather than load it.
    fn stream_reader(&self, node: usize) -> bool {
        self.region.nodes.iter().any(|n| n.index() == node)
            && self.skipped[node] == 0
            && self.role(node, &mut None).is_some()
    }

    /// Whether `node` is a `top_k` that reads the intrinsic plane for `value`.
    fn topk_reads_direct(&self, node: usize, value: u32) -> bool {
        self.direct_topk.get(node).copied().flatten().is_some()
            && self.ops[node].args.first() == Some(&value)
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
            tags::EXP
            | tags::LOG
            | tags::RECIP
            | tags::SIN
            | tags::COS
            | tags::SQRT
            | tags::RSQRT
            | tags::CAST
            | tags::NOT => {
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
        for &node in &self.order[at..] {
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

    /// Whether a value a stream's op defines must also land in scratch: some
    /// reader outside this pass will load it — one that is not a later
    /// pass able to recompute it, nor a `top_k` reading the intrinsic.
    fn escapes(&self, value: u32, members: &[Member]) -> bool {
        let outside: Vec<usize> = self
            .readers
            .get(value as usize)
            .map(|list| {
                list.iter()
                    .copied()
                    .filter(|node| !members.iter().any(|m| m.node == *node))
                    .collect()
            })
            .unwrap_or_default();
        let remat = self.rematerializable(value);
        let served = |node: &usize| (remat && self.stream_reader(*node)) || self.topk_reads_direct(*node, value);
        if outside.iter().any(|node| !served(node)) {
            return true;
        }
        // A region output with no reader this stage names: something else
        // reads it; keep it.
        self.region.outputs.contains(&value) && outside.is_empty()
    }
}

/// The order a row-parallel region's nodes are emitted in.
///
/// A stream ends where an op reads a reduction of the same stream, so the
/// passes a region takes over its rows are set by how many reductions each
/// op waits on, not by where the program wrote it: a Gumbel-max sample and
/// an argmax of the scaled logits wait on none and belong in the first pass
/// beside the row max, however late the guest spelled them. Nodes are
/// scheduled by `(reductions waited on, kind, program order)` over the
/// region's own dataflow — channel takes and reads first, then constants
/// and reshapes (a reshape between two passes costs nothing; inside one it
/// would end it), then the elementwise run, then a gather (which reads a
/// row whole and would split a pass), then channel puts. Channel ops keep
/// their program order among themselves; every op follows its operands.
fn emission_order(ops: &[OpView], bases: &[u32], region: &Region) -> Vec<usize> {
    use eta_ir::op::tags;
    let nodes: Vec<usize> = region.nodes.iter().map(|n| n.index()).collect();
    let count = nodes.len();
    let producer = |value: u32| -> Option<usize> {
        nodes.iter().position(|&n| {
            let base = bases[n];
            value >= base && value < base + ops[n].results
        })
    };
    let reduction = |tag: u8| {
        matches!(
            tag,
            tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN | tags::REDUCE_ARGMAX
        )
    };
    let channel = |tag: u8| matches!(tag, tags::CHAN_TAKE | tags::CHAN_READ | tags::CHAN_PUT);
    let kind = |tag: u8| -> u32 {
        match tag {
            tags::CHAN_TAKE | tags::CHAN_READ => 0,
            tags::CONST | tags::RESHAPE => 1,
            tags::GATHER_ROW => 3,
            tags::CHAN_PUT => 4,
            _ => 2,
        }
    };
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut depth = vec![0u32; count];
    let mut last_channel: Option<usize> = None;
    for i in 0..count {
        let op = &ops[nodes[i]];
        for &arg in &op.args {
            if let Some(p) = producer(arg) {
                deps[i].push(p);
                depth[i] = depth[i].max(depth[p] + u32::from(reduction(ops[nodes[p]].tag)));
            }
        }
        if channel(op.tag) {
            if let Some(previous) = last_channel {
                deps[i].push(previous);
            }
            last_channel = Some(i);
        }
    }
    let mut placed = vec![false; count];
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let next = (0..count)
            .filter(|&i| !placed[i] && deps[i].iter().all(|&d| placed[d]))
            .min_by_key(|&i| (depth[i], kind(ops[nodes[i]].tag), i));
        let Some(i) = next else {
            // Unreachable in SSA; program order is always a valid answer.
            return nodes;
        };
        placed[i] = true;
        out.push(nodes[i]);
    }
    out
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

/// The pass body in its two spellings. The scalar loop (`s`) is the one
/// every stream had; the vector loop (`v`) takes four consecutive elements
/// a thread per iteration, loading and storing each full-row value as one
/// 16-byte access (`pre` before the four, `post` after) and picking the
/// element out of the register quad inside. The arithmetic between is the
/// same text in both. The kernel chooses the vector loop per block when the
/// row's width is a multiple of four and every pointer it touches is
/// 16-byte aligned (`ptrs`, `conds`), and falls back to the scalar loop
/// otherwise.
#[derive(Default)]
struct Bodies {
    s: String,
    v: String,
    pre: String,
    post: String,
    /// Pointer expressions the vector loop reads or writes 16 bytes at.
    ptrs: Vec<String>,
    /// Further block-uniform conditions of the vector loop.
    conds: Vec<String>,
}

impl Bodies {
    /// A line of arithmetic, the same in both loops.
    fn both(&mut self, line: &str) {
        self.s.push_str(line);
        self.s.push('\n');
        self.v.push_str(line);
        self.v.push('\n');
    }

    fn touch(&mut self, ptr: &str) {
        if !self.ptrs.iter().any(|p| p == ptr) {
            self.ptrs.push(ptr.into());
        }
    }

    /// A full-row element load into `name`: scalar in one loop, a quad
    /// picked in the other.
    fn load(&mut self, name: &str, want: Dtype, ptr: &str, from: u32) {
        let l = letter(want);
        let _ = writeln!(self.s, "      const {} {name} = m1_load_{l}({ptr}, i, {from}u);", c_type(want));
        let _ = writeln!(self.pre, "        const {} q_{name} = m1_load4_{l}({ptr}, i4 >> 2u, {from}u);", vec_type(want));
        let _ = writeln!(self.v, "      const {} {name} = m1_pick4_{l}(q_{name}, j);", c_type(want));
        self.touch(ptr);
    }

    /// A full-row element store of register `reg`: scalar in one loop,
    /// gathered into a quad and stored after the four in the other.
    fn store(&mut self, dtype: Dtype, ptr: &str, reg: &str) {
        let l = letter(dtype);
        let _ = writeln!(self.s, "      m1_store_{l}({ptr}, i, {reg});");
        let _ = writeln!(self.pre, "        {} w_{reg};", vec_type(dtype));
        let _ = writeln!(self.v, "      m1_set4_{l}(w_{reg}, j, {reg});");
        let _ = writeln!(self.post, "        m1_store4_{l}({ptr}, i4 >> 2u, w_{reg});");
        self.touch(ptr);
    }
}

/// C spelling of a quad of `dtype`.
fn vec_type(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "float4",
        Dtype::I32 => "int4",
        Dtype::U32 => "uint4",
        _ => "uchar4",
    }
}

/// The pass's emission state: the registers the pass defines, the scalar
/// loads hoisted before it, the element loads already spelled, the
/// intrinsics whose preamble is written, and the text.
struct Pass<'p> {
    registers: Vec<(u32, Dtype)>,
    hoisted: Vec<(u32, Dtype)>,
    loaded: Vec<(u32, Dtype)>,
    intrinsics: Vec<usize>,
    prologue: String,
    body: Bodies,
    pointer: &'p mut dyn FnMut(u32) -> String,
}

/// A read of `value` as `want` inside the pass: a register's conversion, a
/// hoisted scalar (`s<value>_<t>`, loaded once before the pass), a
/// recomputation of a rematerializable value, or the element's load
/// (`l<value>_<t>`, once per pass).
fn read(streams: &Streams<'_>, value: u32, want: Dtype, pass: &mut Pass<'_>) -> String {
    if let Some((_, from)) = pass.registers.iter().find(|(v, _)| *v == value) {
        return convert(&format!("r{value}"), *from, want);
    }
    let from = streams.dtype(value);
    let l = letter(want);
    match streams.class(value) {
        RowClass::Scalar => {
            let name = format!("s{value}_{l}");
            if !pass.hoisted.contains(&(value, want)) {
                pass.hoisted.push((value, want));
                let ptr = (pass.pointer)(value);
                let _ = writeln!(
                    pass.prologue,
                    "    const {} {name} = m1_load_{l}({ptr}, 0u, {}u);",
                    c_type(want),
                    code(from)
                );
            }
            name
        }
        _ => {
            if streams.rematerializable(value) {
                let reg = rematerialize(streams, value, pass);
                return convert(&reg, from, want);
            }
            let name = format!("l{value}_{l}");
            if !pass.loaded.contains(&(value, want)) {
                pass.loaded.push((value, want));
                let ptr = (pass.pointer)(value);
                pass.body.load(&name, want, &ptr, code(from));
            }
            name
        }
    }
}

/// Recompute a rematerializable `value` in this pass: its intrinsic read,
/// or its `Map` over recomputed operands. Answers the register's name.
fn rematerialize(streams: &Streams<'_>, value: u32, pass: &mut Pass<'_>) -> String {
    let node = streams.producers[value as usize] as usize;
    let op = &streams.ops[node];
    let out = streams.bases[node];
    if op.tag == tags::INTRINSIC_VAL {
        intrinsic_read(streams, node, out, pass);
    } else {
        let out_dtype = streams.dtype(out);
        let expr = map_expr(streams, node, out_dtype, pass);
        pass.body.both(&format!("      const {} r{out} = {expr};", c_type(out_dtype)));
        pass.registers.push((out, out_dtype));
    }
    format!("r{out}")
}

/// The intrinsic op's preamble (once per pass) and its per-element read
/// into `r<out>`: the scalar loop's row arithmetic, the vector loop's quad.
fn intrinsic_read(streams: &Streams<'_>, node: usize, out: u32, pass: &mut Pass<'_>) {
    let _ = streams;
    if !pass.intrinsics.contains(&node) {
        pass.intrinsics.push(node);
        let slots = super::fused::PTIR_INTRINSIC_SLOTS;
        let p = &mut pass.prologue;
        let _ = writeln!(p, "    M1OpParams p{node} = params[{node}u];");
        let _ = writeln!(p, "    p{node}.rng_seed = 0u;");
        let _ = writeln!(p, "    const m1_u32 intrinsic_index{node} = dispatch_lane * {slots}u + p{node}.intr;");
        let _ = writeln!(p, "    p{node}.intrinsic_dtype = intrinsic_modes[intrinsic_index{node}];");
        let _ = writeln!(p, "    p{node}.imm = intrinsic_widths[intrinsic_index{node}];");
        let _ = writeln!(p, "    p{node}.intrinsic_row_stride = intrinsic_strides[intrinsic_index{node}];");
        let _ = writeln!(p, "    p{node}.intrinsic_row_offset = intrinsic_offsets[intrinsic_index{node}] + lane_row;");
        let _ = writeln!(
            p,
            "    const m1_u8* ibase{node} = reinterpret_cast<const m1_u8*>(intrinsic_bases[intrinsic_index{node}]);"
        );
        let _ = writeln!(p, "    const M1ValueDesc idesc{node} = descriptors[{out}u];");
        let _ = writeln!(p, "    const m1_u32 iwidth{node} = idesc{node}.last == 0u ? p{node}.imm : idesc{node}.last;");
        let _ = writeln!(
            p,
            "    const m1_u32 istride{node} = p{node}.intrinsic_row_stride == 0u ? iwidth{node} : p{node}.intrinsic_row_stride;"
        );
        let _ = writeln!(p, "    const m1_u64 ifirst{node} = (m1_u64)p{node}.intrinsic_row_offset + (m1_u64)p{node}.imm2;");
        // The quad stays inside one intrinsic row: the row's width a
        // multiple of four and the pass never wider than a row.
        pass.body.conds.push(format!(
            "(iwidth{node} & 3u) == 0u && stream_width <= iwidth{node} && m1_intrinsic_row_vectorable(ibase{node}, ifirst{node}, istride{node}, p{node}.intrinsic_dtype)"
        ));
    }
    let _ = writeln!(
        pass.body.s,
        "      const float r{out} = m1_intrinsic_row_load(ibase{node}, ifirst{node} + i / iwidth{node}, i % iwidth{node}, istride{node}, p{node}.intrinsic_dtype);"
    );
    let _ = writeln!(
        pass.body.pre,
        "        const float4 q_r{out} = m1_intrinsic_row_load4(ibase{node}, ifirst{node} + i4 / iwidth{node}, i4 % iwidth{node}, istride{node}, p{node}.intrinsic_dtype);"
    );
    let _ = writeln!(pass.body.v, "      const float r{out} = m1_pick4_f(q_r{out}, j);");
    pass.registers.push((out, Dtype::F32));
}

/// The C expression of a `Map`/`Broadcast` op's element, its operands read
/// through [`read`] — the runtime helpers' arithmetic, expression for
/// expression.
fn map_expr(streams: &Streams<'_>, node: usize, out_dtype: Dtype, pass: &mut Pass<'_>) -> String {
    let op = &streams.ops[node];
    let tag = op.tag;
    if tag == tags::BROADCAST {
        read(streams, op.args[0], out_dtype, pass)
    } else if matches!(
        tag,
        tags::EXP | tags::LOG | tags::RECIP | tags::SIN | tags::COS | tags::SQRT | tags::RSQRT
    ) {
        let x = read(streams, op.args[0], Dtype::F32, pass);
        match tag {
            tags::EXP => format!("expf({x})"),
            tags::LOG => format!("logf({x})"),
            tags::SIN => format!("sinf({x})"),
            tags::COS => format!("cosf({x})"),
            tags::SQRT => format!("sqrtf({x})"),
            tags::RSQRT => format!("(1.0f / sqrtf({x}))"),
            _ => format!("(1.0f / {x})"),
        }
    } else if tag == tags::CAST {
        read(streams, op.args[0], out_dtype, pass)
    } else if tag == tags::NOT {
        let x = read(streams, op.args[0], Dtype::Bool, pass);
        format!("(!{x})")
    } else if matches!(tag, tags::NEG | tags::ABS | tags::SIGN) {
        let path = streams.dtype(op.args[0]);
        let x = read(streams, op.args[0], path, pass);
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
        let c = read(streams, op.args[0], Dtype::Bool, pass);
        let a = read(streams, op.args[1], out_dtype, pass);
        let b = read(streams, op.args[2], out_dtype, pass);
        format!("({c} ? {a} : {b})")
    } else if matches!(tag, tags::AND | tags::OR) {
        let a = read(streams, op.args[0], Dtype::Bool, pass);
        let b = read(streams, op.args[1], Dtype::Bool, pass);
        if tag == tags::AND { format!("({a} && {b})") } else { format!("({a} || {b})") }
    } else {
        // Binary arithmetic and compares: the left operand's dtype picks
        // the path, as the helper's `d0.dtype` does.
        let path = streams.dtype(op.args[0]);
        let a = read(streams, op.args[0], path, pass);
        let b = read(streams, op.args[1], path, pass);
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

    let mut pass = Pass {
        registers: Vec::new(),
        hoisted: Vec::new(),
        loaded: Vec::new(),
        intrinsics: Vec::new(),
        prologue: String::new(),
        body: Bodies::default(),
        pointer,
    };
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
                let expr = map_expr(streams, node, out_dtype, &mut pass);
                pass.body.both(&format!("      const {} r{out} = {expr};", c_type(out_dtype)));
                pass.registers.push((out, out_dtype));
                if streams.escapes(out, &members) {
                    let ptr = (pass.pointer)(out);
                    pass.body.store(out_dtype, &ptr, &format!("r{out}"));
                }
            }
            Role::Intrinsic => {
                intrinsic_read(streams, node, out, &mut pass);
                if streams.escapes(out, &members) {
                    let ptr = (pass.pointer)(out);
                    pass.body.store(Dtype::F32, &ptr, &format!("r{out}"));
                }
            }
            Role::Rng => {
                let _ = writeln!(pass.prologue, "    M1OpParams p{node} = params[{node}u];");
                let _ = writeln!(pass.prologue, "    p{node}.rng_seed = 0u;");
                let _ = writeln!(pass.prologue, "    p{node}.imm3 = lane_row * descriptors[p{node}.o0].len;");
                if tag == tags::RNG {
                    let _ = writeln!(
                        pass.prologue,
                        "    const m1_u64 seed{node} = ptir_rng_seed_eff_stream((m1_u32)p{node}.rng_seed, p{node}.imm);"
                    );
                } else {
                    let state = op.args[0];
                    let sp = (pass.pointer)(state);
                    let _ = writeln!(
                        pass.prologue,
                        "    const m1_u64 seed{node} = ptir_rng_keyed_seed(m1_load_u({sp}, 0u, 2u), descriptors[{state}u].len > 1u ? m1_load_u({sp}, 1u, 2u) : 0u);"
                    );
                }
                if op.kind == eta_ir::types::RngKind::Normal as u8 {
                    // Box-Muller reads two uniform lanes an element, so the
                    // draw is the runtime's own function rather than a
                    // transform of one `u{node}`.
                    pass.body.both(&format!("      const float r{out} = ptir_rng_hash_normal(seed{node}, i + p{node}.imm3);"));
                } else {
                    pass.body.both(&format!("      const float u{node} = ptir_rng_hash_uniform(seed{node}, i + p{node}.imm3);"));
                    pass.body.both(&format!("      const float r{out} = p{node}.kind == 0u ? u{node} : -logf(-logf(u{node}));"));
                }
                pass.registers.push((out, Dtype::F32));
                if streams.escapes(out, &members) {
                    let ptr = (pass.pointer)(out);
                    pass.body.store(Dtype::F32, &ptr, &format!("r{out}"));
                }
            }
            Role::Reduce => {
                let x = read(streams, op.args[0], Dtype::F32, &mut pass);
                let (identity, combine): (&str, fn(&str, &str) -> String) = match tag {
                    tags::REDUCE_SUM => ("0.0f", |a, b| format!("({a} + {b})")),
                    tags::REDUCE_MAX => ("m1_neg_inf()", |a, b| format!("m1_canonical_max({a}, {b})")),
                    _ => ("m1_pos_inf()", |a, b| format!("m1_canonical_min({a}, {b})")),
                };
                let slot = float_reductions;
                float_reductions += 1;
                let _ = writeln!(pass.prologue, "    float acc{node} = {identity};");
                pass.body.both(&format!("      acc{node} = {};", combine(&format!("acc{node}"), &x)));
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
                let ptr = (pass.pointer)(out);
                let _ = writeln!(epilogue, "      m1_store_f({ptr}, 0u, total);");
                let _ = writeln!(epilogue, "    }}");
            }
            Role::Argmax => {
                let x = read(streams, op.args[0], Dtype::F32, &mut pass);
                argmaxes += 1;
                let _ = writeln!(pass.prologue, "    M1ArgmaxCandidate cand{node}{{m1_neg_inf(), 0u, 0u, 0u}};");
                pass.body.both(&format!(
                    "      cand{node} = m1_argmax_combine(cand{node}, M1ArgmaxCandidate{{{x}, i, m1_isnan({x}) ? 0u : 1u, 0u}});"
                ));
                let _ = writeln!(epilogue, "    cand{node} = m1_argmax_warp_reduce(cand{node}, stream_lane);");
                let _ = writeln!(epilogue, "    if (stream_lane == 0u) stream_candidates[stream_warp] = cand{node};");
                let _ = writeln!(epilogue, "    __syncthreads();");
                let _ = writeln!(epilogue, "    if (stream_warp == 0u) {{");
                let _ = writeln!(
                    epilogue,
                    "      cand{node} = stream_lane < stream_warps ? stream_candidates[stream_lane] : M1ArgmaxCandidate{{m1_neg_inf(), 0u, 0u, 0u}};"
                );
                let _ = writeln!(epilogue, "      cand{node} = m1_argmax_warp_reduce(cand{node}, stream_lane);");
                let ptr = (pass.pointer)(out);
                let _ = writeln!(epilogue, "      if (stream_lane == 0u) m1_store_i({ptr}, 0u, (int)cand{node}.index);");
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
    let Pass { prologue, body, .. } = pass;
    s.push_str(&prologue);
    // The vector loop when the block can take it, else the scalar one.
    let mut conds: Vec<String> = vec!["(stream_width & 3u) == 0u".into()];
    conds.extend(body.ptrs.iter().map(|p| format!("m1_aligned16({p})")));
    conds.extend(body.conds.iter().cloned());
    let _ = writeln!(s, "    const bool stream_vec = {};", conds.join(" && "));
    s.push_str("    if (stream_vec) {\n");
    s.push_str("      for (m1_u32 i4 = threadIdx.x * 4u; i4 < stream_width; i4 += blockDim.x * 4u) {\n");
    s.push_str(&body.pre);
    s.push_str("#pragma unroll\n");
    s.push_str("        for (m1_u32 j = 0u; j < 4u; ++j) {\n");
    s.push_str("      const m1_u32 i = i4 + j;\n");
    s.push_str(&body.v);
    s.push_str("        }\n");
    s.push_str(&body.post);
    s.push_str("      }\n");
    s.push_str("    } else {\n");
    s.push_str("    for (m1_u32 i = threadIdx.x; i < stream_width; i += blockDim.x) {\n");
    s.push_str(&body.s);
    s.push_str("    }\n");
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
    let direct_topk = super::fused::analyze_direct_topk(stage);
    let streams = Streams::new(stage, region, &ops, &bases, &kinds, &direct.intrinsic, &direct.skipped, &direct_topk);
    let mut spent = Vec::new();
    let mut at = 0usize;
    while at < streams.order.len() {
        let node = streams.order[at];
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
