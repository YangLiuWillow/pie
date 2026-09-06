//! `Modulate`: adaLN's three shapes, the gated residual, and the two fused
//! forms that fold a norm into the first of them.
//!
//! A DiT block is a norm, a modulation, an attention or an MLP, and a gated
//! write back into the residual. The norms and the projections were already
//! here; this file is the rest — `x·(1+s)+b`, `x·(1+s)`, `tanh(g)·x`,
//! `r += g·y` — plus [`norm_modulate`] and [`gated_residual_norm_modulate`],
//! which land what a norm entry followed by one of these lands, in one pass
//! over the row.
//!
//! **Numerics.** Every arm reads bf16, computes in f32, and rounds once at
//! the store; the scale and the shift are one `fmaf`, so a host reference
//! must use a fused multiply-add too. The fused pair writes the residual with
//! exactly the rounding [`gated_residual_add`] writes and reduces the norm's
//! moments over THAT rounded row, so the residual is bit-equal to the unfused
//! chain's; the normed row stays in f32 through the modulation, so it lands
//! about one bf16 ulp from the chain's (nearer the f32 ideal, not further).

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/modulate.cuh";

const BLOCK: u32 = 256;

/// Which of the three modulation shapes an entry fires. The `m` rectangle is
/// `[.., 2·width]` for [`ScaleShift`](Form::ScaleShift) — scale first, shift
/// second — and `[.., width]` for the other two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// `o = x·(1+s) + b`.
    ScaleShift,
    /// `o = x·(1+s)`.
    Scale,
    /// `o = tanh(g)·x` — Z-Image's gate, whose `tanh(0) = 0` keeps the
    /// adaLN-Zero init while bounding the gate.
    TanhGate,
}

impl Form {
    const fn stamp(self) -> &'static str {
        match self {
            Form::ScaleShift => "0",
            Form::Scale => "1",
            Form::TanhGate => "2",
        }
    }

    /// How many `width`-wide vectors this form reads out of `m`.
    const fn vectors(self) -> u32 {
        match self {
            Form::ScaleShift => 2,
            Form::Scale | Form::TanhGate => 1,
        }
    }
}

/// Which norm a fused entry runs in front of the modulation. The modulation
/// is the affine, so two of the three carry no weight of their own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormKind {
    /// `(x − mean(x))·rsqrt(var(x) + eps)`, two reductions.
    LayerNormNoAffine { eps: f32 },
    /// `x·rsqrt(mean(x²) + eps)`.
    RmsNormNoScale { eps: f32 },
    /// `x·rsqrt(mean(x²) + eps)·w`, for the sites that keep a bank in front
    /// of the modulation.
    RmsNorm { weight: Tensor, eps: f32 },
}

impl NormKind {
    const fn stamp(self) -> &'static str {
        match self {
            NormKind::LayerNormNoAffine { .. } => "0",
            NormKind::RmsNormNoScale { .. } => "1",
            NormKind::RmsNorm { .. } => "2",
        }
    }

    const fn eps(self) -> f32 {
        match self {
            NormKind::LayerNormNoAffine { eps }
            | NormKind::RmsNormNoScale { eps }
            | NormKind::RmsNorm { eps, .. } => eps,
        }
    }

    fn weight(self) -> ArgValue {
        match self {
            NormKind::RmsNorm { weight, .. } => weight.arg(),
            _ => ArgValue::ABSENT,
        }
    }
}

/// The lane map as an argument: a `[rows]` i32 plane, or the null seat that
/// means "`m` is indexed by the row".
fn lane_map(op: &'static str, lane_of_row: Option<Tensor>, rows: u32) -> Result<ArgValue, Error> {
    let Some(map) = lane_of_row else {
        return Ok(ArgValue::ABSENT);
    };
    if map.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the lane map is {:?}, and this modulation reads an i32 lane per row",
                map.dtype
            ),
        ));
    }
    if map.elements() != u64::from(rows) {
        return Err(refuse(
            op,
            format!(
                "the lane map is {} x {}, and this modulation reads one lane for each of \
                 {rows} rows",
                map.rows, map.width
            ),
        ));
    }
    Ok(map.arg())
}

/// The `m` rectangle's element and width check, shared by every arm,
/// answering the element the kernel is stamped with for it: the rows'
/// own, or `float` for a vector that arrives from a lane chain kept in f32
/// (`IMAGEGEN_CONTRACT.md` §3: `m`/`g` are f32 or `x`'s dtype). Its ROW
/// count is checked by the caller instead: with a lane map bound the rows
/// are lanes and only the map's values name them, so nothing here can count
/// them.
fn modulation(
    op: &'static str,
    m: Tensor,
    width: u32,
    vectors: u32,
    dtype: Dtype,
) -> Result<&'static str, Error> {
    let tm = if m.dtype == dtype {
        dtype_dispatch!(op, dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" })
    } else if m.dtype == Dtype::F32 {
        "float"
    } else {
        return Err(refuse(
            op,
            format!(
                "the modulation plane is {:?} and the rows it modulates are {dtype:?}; \
                 a vector rides the activation's element or stays f32",
                m.dtype
            ),
        ));
    };
    if m.width != vectors * width {
        return Err(refuse(
            op,
            format!(
                "the modulation plane is {} wide, and this form reads {vectors} x {width}",
                m.width
            ),
        ));
    }
    Ok(tm)
}

/// The three unfused arms' one body.
fn fire(
    ctx: &Ctx,
    op: &'static str,
    form: Form,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        x.rows == o.rows && x.width == o.width,
        "`{op}` writes the rectangle it reads"
    );
    let width = nonzero(op, "the modulated width", o.width)?;
    let rows = nonzero(op, "rows", o.rows)?;
    let tm = modulation(op, m, width, form.vectors(), x.dtype)?;
    if lane_of_row.is_none() && m.rows < rows {
        return Err(refuse(
            op,
            format!("{} per-token modulation vectors for {rows} rows", m.rows),
        ));
    }
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::modulate<{t}, {tm}, {}>",
                form.stamp()
            )),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            x.arg(),
            m.arg(),
            lane_map(op, lane_of_row, rows)?,
            o.arg(),
            stated(op, width)?.arg(),
            stated(op, form.vectors() * width)?.arg(),
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}

/// `o = x·(1+s) + b`, adaLN's shape. `m` is `[lanes, 2·width]` (scale first,
/// shift second) read through `lane_of_row`, or `[rows, 2·width]` when no map
/// is bound. `o` may alias `x`.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for a
/// degenerate rectangle, a modulation plane of the wrong width or dtype, or a
/// lane map that is not one i32 per row.
pub fn scale_shift(
    ctx: &Ctx,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: &mut Tensor,
) -> Result<(), Error> {
    fire(
        ctx,
        "elementwise.modulate",
        Form::ScaleShift,
        x,
        m,
        lane_of_row,
        o,
    )
}

/// `o = x·(1+s)`, the shift-less half of [`scale_shift`]; `m` is `[.., width]`.
///
/// # Errors
///
/// As [`scale_shift`].
pub fn scale(
    ctx: &Ctx,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: &mut Tensor,
) -> Result<(), Error> {
    fire(
        ctx,
        "elementwise.modulate",
        Form::Scale,
        x,
        m,
        lane_of_row,
        o,
    )
}

/// `o = tanh(g)·x`; `m` is `[.., width]`.
///
/// # Errors
///
/// As [`scale_shift`].
pub fn tanh_gate(
    ctx: &Ctx,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: &mut Tensor,
) -> Result<(), Error> {
    fire(
        ctx,
        "elementwise.modulate",
        Form::TanhGate,
        x,
        m,
        lane_of_row,
        o,
    )
}

/// `r_out = r + g·y`, the gated residual write a DiT sub-block ends with.
/// `g` is `[lanes, width]` read through `lane_of_row`, or `[rows, width]`
/// without it. `r_out` may alias `r` — that is the in-place form.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for a
/// degenerate rectangle, a gate plane of the wrong width or dtype, a lane map
/// that is not one i32 per row, or operands that do not share one shape.
pub fn gated_residual_add(
    ctx: &Ctx,
    r: Tensor,
    g: Tensor,
    y: Tensor,
    lane_of_row: Option<Tensor>,
    r_out: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gated_residual_add";
    let t = dtype_dispatch!(OP, r.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        r.rows == y.rows && r.width == y.width && r.rows == r_out.rows && r.width == r_out.width,
        "`{OP}` folds three rectangles of one shape"
    );
    let width = nonzero(OP, "the folded width", r_out.width)?;
    let rows = nonzero(OP, "rows", r_out.rows)?;
    let tm = modulation(OP, g, width, 1, r.dtype)?;
    if lane_of_row.is_none() && g.rows < rows {
        return Err(refuse(
            OP,
            format!("{} per-token gate vectors for {rows} rows", g.rows),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::gated_residual_add<{t}, {tm}>")),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            r.arg(),
            g.arg(),
            y.arg(),
            lane_map(OP, lane_of_row, rows)?,
            r_out.arg(),
            stated(OP, width)?.arg(),
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}

/// `o = norm(x)·(1+s) + b`: what a norm entry followed by [`scale_shift`]
/// lands, one row reduction and one pass. `m` is `[.., 2·width]`, read the
/// way [`scale_shift`] reads it.
///
/// The normed row stays in f32 into the modulation rather than round-tripping
/// through bf16 the way the two launches do, so it lands about one bf16 ulp
/// from them — on the near side.
///
/// # Errors
///
/// As [`scale_shift`], plus a refusal for an [`NormKind::RmsNorm`] weight
/// that is not one scalar per column.
pub fn norm_modulate(
    ctx: &Ctx,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    norm: NormKind,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.norm_modulate";
    let (t, tm, width, rows) = fused_shapes(OP, x, m, lane_of_row, norm, o)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::norm_modulate<{t}, {tm}, {BLOCK}, {}>",
                norm.stamp()
            )),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            x.arg(),
            norm.weight(),
            m.arg(),
            lane_map(OP, lane_of_row, rows)?,
            o.arg(),
            stated(OP, width)?.arg(),
            stated(OP, 2 * width)?.arg(),
            norm.eps().arg(),
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}

/// The deferred-residual form: `r_out = r + g·y` and `o = norm(r_out)·(1+s)
/// + b`, two outputs from one pass. What [`gated_residual_add`] then
/// [`norm_modulate`] land — bit-equal on `r_out`, within about one bf16 ulp
/// on `o`. `r_out` may alias `r`.
///
/// # Errors
///
/// As [`norm_modulate`] and [`gated_residual_add`].
#[allow(clippy::too_many_arguments)]
pub fn gated_residual_norm_modulate(
    ctx: &Ctx,
    r: Tensor,
    g: Tensor,
    y: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    norm: NormKind,
    r_out: &mut Tensor,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gated_residual_norm_modulate";
    let (t, tm, width, rows) = fused_shapes(OP, r, m, lane_of_row, norm, o)?;
    debug_assert!(
        r.rows == y.rows && r.width == y.width && r.rows == r_out.rows && r.width == r_out.width,
        "`{OP}` folds three rectangles of one shape"
    );
    let tg = modulation(OP, g, width, 1, r.dtype)?;
    if tg != tm {
        return Err(refuse(
            OP,
            format!(
                "the gate plane is {:?} and the modulation plane {:?}; one fused pass reads \
                 both in one element",
                g.dtype, m.dtype
            ),
        ));
    }
    if lane_of_row.is_none() && g.rows < rows {
        return Err(refuse(
            OP,
            format!("{} per-token gate vectors for {rows} rows", g.rows),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::gated_residual_norm_modulate<{t}, {tm}, {BLOCK}, {}>",
                norm.stamp()
            )),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            r.arg(),
            g.arg(),
            y.arg(),
            r_out.arg(),
            norm.weight(),
            m.arg(),
            lane_map(OP, lane_of_row, rows)?,
            o.arg(),
            stated(OP, width)?.arg(),
            stated(OP, 2 * width)?.arg(),
            norm.eps().arg(),
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}

/// The refusals the two fused arms share, and the three numbers their launch
/// is spelled with.
fn fused_shapes(
    op: &'static str,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    norm: NormKind,
    o: &mut Tensor,
) -> Result<(&'static str, &'static str, u32, u32), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        x.rows == o.rows && x.width == o.width,
        "`{op}` writes the rectangle it reads"
    );
    let width = nonzero(op, "the normed width", o.width)?;
    let rows = nonzero(op, "rows", o.rows)?;
    let tm = modulation(op, m, width, 2, x.dtype)?;
    if lane_of_row.is_none() && m.rows < rows {
        return Err(refuse(
            op,
            format!("{} per-token modulation vectors for {rows} rows", m.rows),
        ));
    }
    if let NormKind::RmsNorm { weight, .. } = norm
        && weight.elements() != u64::from(width)
    {
        return Err(refuse(
            op,
            format!(
                "the norm's weight is a {} x {} plane, and this row reads one scalar per \
                 column of a {width}-wide rectangle",
                weight.rows, weight.width
            ),
        ));
    }
    Ok((t, tm, width, rows))
}
