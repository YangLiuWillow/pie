//! `Conv3d`: implicit-GEMM convolution over `[rows, C_in]` voxel rows into
//! `[rows_out, C_out]`, `conv2d` being the `kt = 1` case of the same entry.
//! Two device kernels compute it — fp32 FMA tiles (every shape) and bf16
//! `mma.sync` tiles (`C_in % 8 == 0`, 16-byte planes, sm_80+) — and
//! [`conv3d`] picks; [`conv3d_on`] lets a golden pin one.
//!
//! **THE WEIGHT LAYOUT.** `w` is `[C_out, kt*kh*kw*C_in]` bf16, K flattened
//! tap-major and channel-fastest: `K = ((it*kh + ih)*kw + iw)*C_in + c_in`,
//! i.e. PyTorch's `weight.permute(0, 2, 3, 4, 1).reshape(C_out, -1)` (cuDNN's
//! KRSC). A checkpoint that stores the natural `[C_out, C_in*kt*kh*kw]`
//! rectangle is relabelled once at load by [`conv_weight_taps_major`]. The
//! reason is the gather: one K step reads `BK` consecutive channels of one
//! tap from both operands as 16-byte words, which the natural order
//! (channels `kt*kh*kw` apart) would turn into two-byte strided loads on
//! every weight tile.
//!
//! **TIME.** `causal_t = false`: `pad[0]` zero frames on both sides.
//! `causal_t = true`: `pad[0]` frames in front and none behind, the frames
//! before the clip read from `cache` when one is given — the previous
//! tile's last `pad[0]` frames, `[sum over lanes of pad[0]*h*w, C_in]` in
//! lane order — else frame 0 ([`TimePad::Replicate`]) or zero
//! ([`TimePad::Zero`]). `h`/`w` padding is symmetric zero either way.
//!
//! Numerics: bf16 in, fp32 accumulation over the whole K, bias in fp32,
//! one rounding at the store. The two kernels sum K in different orders and
//! agree to fp32 rounding.

use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, count, dtype_dispatch, refuse, stated,
};
use crate::spatial::lane_pair;
use crate::tensor::Tensor;
use dtype::Dtype;

const FILE: &str = "spatial/conv.cuh";

const OP: &str = "spatial.conv3d";

/// The direct kernel's tile: output voxels by output channels.
const DIRECT_TILE: [u32; 2] = [64, 64];

const DIRECT_BLOCK: u32 = 256;

/// The tensor-core kernel's tile, its pipeline depth, and the dynamic
/// shared memory that buys (`conv3d_mma_smem<3>()` in the device text:
/// three stages of `(128 + 128) x 40` bf16, which also holds the epilogue
/// tile).
const MMA_TILE: [u32; 2] = [128, 128];

const MMA_STAGES: u32 = 3;

const MMA_BLOCK: u32 = 256;

const MMA_SMEM: u32 = MMA_STAGES * (MMA_TILE[0] + MMA_TILE[1]) * (32 + 8) * 2;

/// What a frame before the clip reads under `causal_t` when no cache is
/// given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimePad {
    /// Zeros — the first tile of a zero-padded causal convolution.
    Zero,
    /// The clip's own first frame, repeated.
    Replicate,
}

/// The static shape of one convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv3d {
    /// `[kt, kh, kw]`.
    pub k: [u32; 3],
    /// `[st, sh, sw]`.
    pub stride: [u32; 3],
    /// `[pt, ph, pw]`: symmetric zero padding, except that under
    /// `causal_t` `pt` is the front-only time padding.
    pub pad: [u32; 3],
    /// Time is padded in front only, from the cache when one is given.
    pub causal_t: bool,
    /// What the front frames read under `causal_t` without a cache.
    pub time_pad: TimePad,
}

impl Conv3d {
    /// A 2-D convolution: `kt = 1`, no time stride or padding.
    #[must_use]
    pub const fn conv2d(k: [u32; 2], stride: [u32; 2], pad: [u32; 2]) -> Self {
        Self {
            k: [1, k[0], k[1]],
            stride: [1, stride[0], stride[1]],
            pad: [0, pad[0], pad[1]],
            causal_t: false,
            time_pad: TimePad::Zero,
        }
    }

    /// Taps per output channel: `kt * kh * kw`.
    #[must_use]
    pub const fn taps(&self) -> u32 {
        self.k[0] * self.k[1] * self.k[2]
    }

    /// The output box of an input box — what a caller writes into `o_grid`:
    /// `(n + 2*pad - k) / stride + 1` per axis, with the time axis padded
    /// only in front under `causal_t`. `None` when the box is smaller than
    /// the kernel.
    #[must_use]
    pub fn out_extent(&self, [t, h, w]: [u32; 3]) -> Option<[u32; 3]> {
        let axis = |n: u32, k: u32, s: u32, front: u32, back: u32| {
            (n + front + back)
                .checked_sub(k)
                .map(|span| span / s.max(1) + 1)
        };
        let back_t = if self.causal_t { 0 } else { self.pad[0] };
        Some([
            axis(t, self.k[0], self.stride[0], self.pad[0], back_t)?,
            axis(h, self.k[1], self.stride[1], self.pad[1], self.pad[1])?,
            axis(w, self.k[2], self.stride[2], self.pad[2], self.pad[2])?,
        ])
    }
}

/// Which device kernel lands the convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvPath {
    /// Tensor cores when the shape and device admit them, else the direct
    /// kernel.
    Auto,
    /// The fp32 FMA kernel: every channel count and alignment.
    Direct,
    /// The bf16 `mma.sync` kernel; refused when the shape does not admit
    /// it rather than silently falling back.
    TensorCore,
}

/// `ConvGeom` in `spatial/conv.cuh`, field for field. `#[repr(C)]` because
/// the bytes cross the launch ABI as one by-value parameter.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Geom {
    c_in: i32,
    c_out: i32,
    kt: i32,
    kh: i32,
    kw: i32,
    st: i32,
    sh: i32,
    sw: i32,
    pt: i32,
    ph: i32,
    pw: i32,
    causal: i32,
    replicate: i32,
    lanes: i32,
    rows_out: i32,
}

/// Implicit-GEMM 3-D convolution: `o[m][n] = bias[n] + sum x[tap(m)][c] *
/// w[n][tap*C_in + c]`. See the module doc for the weight layout and the
/// time rules.
///
/// `x`: `[rows, C_in]` bf16; `grid`/`o_grid`: `[lanes, 4]` i32 lane tables
/// for the input and output boxes (the output box per lane is
/// [`Conv3d::out_extent`] of the input's); `w`: `[C_out, taps*C_in]` bf16;
/// `bias`: `C_out` f32 or none; `cache`: the causal front frames or none;
/// `o`: `[rows_out, C_out]` bf16. Output rows no lane claims land zeros.
///
/// Errs [`Error::DtypeUnsupported`] for anything but bf16 activations, or a
/// refusal for a weight that is not `[C_out, taps*C_in]`, a cache without
/// `causal_t`, a zero kernel or stride, or a lane table of the wrong shape.
#[allow(clippy::too_many_arguments)]
pub fn conv3d(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    conv: Conv3d,
    cache: Option<Tensor>,
    o: &mut Tensor,
    o_grid: Tensor,
) -> Result<(), Error> {
    conv3d_on(
        ctx,
        ConvPath::Auto,
        x,
        grid,
        w,
        bias,
        conv,
        cache,
        o,
        o_grid,
    )
}

/// [`conv3d`] on a named kernel.
#[allow(clippy::too_many_arguments)]
pub fn conv3d_on(
    ctx: &Ctx,
    path: ConvPath,
    x: Tensor,
    grid: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    conv: Conv3d,
    cache: Option<Tensor>,
    o: &mut Tensor,
    o_grid: Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    debug_assert_eq!(w.dtype, Dtype::Bf16, "`{OP}` reads a bf16 weight");
    debug_assert_eq!(o.dtype, Dtype::Bf16, "`{OP}` lands bf16");
    let lanes = lane_pair(OP, grid, o_grid)?;
    let c_in = count(OP, "the input channels", x.width)?;
    let c_out = count(OP, "the output channels", o.width)?;
    let rows_out = count(OP, "the output rows", o.rows)?;
    for (axis, v) in ["kt", "kh", "kw"].into_iter().zip(conv.k) {
        count(OP, axis, v)?;
    }
    for (axis, v) in ["st", "sh", "sw"].into_iter().zip(conv.stride) {
        count(OP, axis, v)?;
    }
    let taps = conv.taps();
    if w.rows != o.width || w.width != taps.saturating_mul(x.width) {
        return Err(refuse(
            OP,
            format!(
                "the weight is {}x{}; this convolution reads [{}, {}] = [C_out, kt*kh*kw*C_in]",
                w.rows,
                w.width,
                o.width,
                taps.saturating_mul(x.width)
            ),
        ));
    }
    if let Some(b) = bias
        && (b.dtype != Dtype::F32 || b.elements() != u64::from(o.width))
    {
        return Err(refuse(
            OP,
            format!(
                "the bias is {}x{} {:?}; expected {} f32",
                b.rows, b.width, b.dtype, o.width
            ),
        ));
    }
    if let Some(c) = cache {
        if !conv.causal_t || conv.pad[0] == 0 {
            return Err(refuse(
                OP,
                "a frame cache is read only under `causal_t` with a front pad",
            ));
        }
        if c.dtype != Dtype::Bf16 || c.width != x.width {
            return Err(refuse(
                OP,
                format!(
                    "the cache is {}x{} {:?}; it holds `[frames, C_in]` bf16",
                    c.rows, c.width, c.dtype
                ),
            ));
        }
    }

    let geom = Geom {
        c_in,
        c_out,
        kt: stated(OP, conv.k[0])?,
        kh: stated(OP, conv.k[1])?,
        kw: stated(OP, conv.k[2])?,
        st: stated(OP, conv.stride[0])?,
        sh: stated(OP, conv.stride[1])?,
        sw: stated(OP, conv.stride[2])?,
        pt: stated(OP, conv.pad[0])?,
        ph: stated(OP, conv.pad[1])?,
        pw: stated(OP, conv.pad[2])?,
        causal: i32::from(conv.causal_t),
        replicate: i32::from(conv.time_pad == TimePad::Replicate),
        lanes,
        rows_out,
    };

    let vectorised = x.width.is_multiple_of(8)
        && aligned16(x.ptr)
        && aligned16(w.ptr)
        && cache.is_none_or(|c| aligned16(c.ptr))
        && (!o.width.is_multiple_of(8) || aligned16(o.ptr));
    let tensor_core = vectorised
        && ctx
            .compute_capability_major()
            .is_some_and(|major| major >= 8);
    let mma = match path {
        ConvPath::Auto => tensor_core,
        ConvPath::Direct => false,
        ConvPath::TensorCore => {
            if !tensor_core {
                return Err(refuse(
                    OP,
                    format!(
                        "the tensor-core kernel needs C_in % 8 == 0 ({}), 16-byte planes and sm_80+",
                        x.width
                    ),
                ));
            }
            true
        }
    };

    let (entry, tile, block, smem) = if mma {
        (
            crate::jit::symbol(&format!("::pie::spatial::conv3d_mma<{MMA_STAGES}>")),
            MMA_TILE,
            MMA_BLOCK,
            MMA_SMEM,
        )
    } else {
        (
            if vectorised {
                "::pie::spatial::conv3d_direct<true>"
            } else {
                "::pie::spatial::conv3d_direct<false>"
            },
            DIRECT_TILE,
            DIRECT_BLOCK,
            0,
        )
    };
    let launch = Launch::grid(
        [o.rows.div_ceil(tile[0]), o.width.div_ceil(tile[1]), 1],
        [block, 1, 1],
    )
    .smem(smem);
    ctx.fire(
        OP,
        Fire::at(FILE, entry).apply(launch),
        &[
            x.arg(),
            grid.arg(),
            w.arg(),
            bias.map_or(ArgValue::ABSENT, |b| b.arg()),
            cache.map_or(ArgValue::ABSENT, |c| c.arg()),
            o.arg(),
            o_grid.arg(),
            ArgValue::Bytes {
                ptr: std::ptr::from_ref(&geom).cast(),
                len: size_of::<Geom>(),
            },
        ],
    )
}

/// The load-time relabelling: `src` is the checkpoint's natural
/// `[C_out, C_in*kt*kh*kw]` rectangle (`(c_in, kt, kh, kw)`, `kw` fastest —
/// `weight.reshape(C_out, -1)`), `dst` the same values as
/// `[C_out, kt*kh*kw*C_in]` in the tap-major channel-fastest order
/// [`conv3d`] reads. `taps = kt*kh*kw`.
pub fn conv_weight_taps_major(
    ctx: &Ctx,
    src: Tensor,
    c_in: u32,
    taps: u32,
    dst: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.conv_weight_taps_major";
    const BLOCK: u32 = 256;
    dtype_dispatch!(OP, src.dtype, { Bf16 => () });
    debug_assert_eq!(dst.dtype, Dtype::Bf16, "`{OP}` lands bf16");
    let c_in = count(OP, "the input channels", c_in)?;
    let taps = count(OP, "the tap count", taps)?;
    let c_out = count(OP, "the output channels", src.rows)?;
    let k = u64::from(src.width);
    if k != u64::from(c_in.unsigned_abs()) * u64::from(taps.unsigned_abs())
        || dst.rows != src.rows
        || dst.width != src.width
    {
        return Err(refuse(
            OP,
            format!(
                "{}x{} into {}x{} is not a relabelling of [C_out, {c_in}*{taps}]",
                src.rows, src.width, dst.rows, dst.width
            ),
        ));
    }
    let (blocks, _) = crate::spatial::flat_elements(OP, src, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::spatial::conv_weight_taps_major")
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[src.arg(), dst.arg(), c_out.arg(), c_in.arg(), taps.arg()],
    )
}
