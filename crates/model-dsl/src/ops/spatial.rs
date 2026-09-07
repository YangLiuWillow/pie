//! The `Spatial` family (design D8): what a VAE author writes. Activations
//! are `[Voxels, channels]` rectangles beside a `[Clips, 4]` grid; every
//! wrapper that changes the box hands back `(y, y_grid)`, the grid computed
//! on the device by [`grid`] from the input's and the op's [`GridRule`].
//!
//! One ResBlock of a conv VAE decoder, in this vocabulary:
//!
//! ```ignore
//! let h = spatial::group_norm(&x, &g, 32, &self.norm1_w, &self.norm1_b, 1e-6, true);
//! let (h, g1) = spatial::conv3d(&h, &g, &self.conv1, Some(&self.conv1_b), Conv::same3(), None);
//! let h = spatial::group_norm(&h, &g1, 32, &self.norm2_w, &self.norm2_b, 1e-6, true);
//! let (h, _) = spatial::conv3d(&h, &g1, &self.conv2, Some(&self.conv2_b), Conv::same3(), None);
//! let y = elemwise::add(&x, &h);          // the residual (same box, same grid)
//! ```
//!
//! (A `same3` conv keeps the box, so `g1 == g` numerically; the wrapper
//! still hands a grid back, and a text may keep using `g`.)

use super::*;
use model_ir::{GridRule, ParamLayout, RowAxis, Spatial, TimePad, VoxelSegment};

/// The grid table's type: `[Clips, 4]` i32.
fn grid_ty() -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Clips, Dim::Const(4)],
        dtype: Dtype::I32,
    }
}

fn expect_grid(what: &str, grid: &Value) {
    assert!(
        grid.rows() == Dim::Clips && grid.width() == 4 && grid.dtype() == Dtype::I32,
        "{what} must be a `[Clips, 4]` i32 grid table, not {:?}",
        grid.ty()
    );
}

fn expect_voxels(what: &str, x: &Value) {
    assert!(
        x.rows().axis() == Some(RowAxis::Voxels),
        "{what} must live on the voxel axis, not {:?}",
        x.rows()
    );
}

/// The row dim a growth by `k` lands: `Voxels` becomes `VoxelsTimes(k)`,
/// `VoxelsTimes(j)` becomes `VoxelsTimes(j·k)`; `k == 1` is the identity.
fn grown(rows: Dim, k: u32) -> Dim {
    match (rows, k) {
        (rows, 1) => rows,
        (Dim::Voxels, k) => Dim::VoxelsTimes(k),
        (Dim::VoxelsTimes(j), k) => Dim::VoxelsTimes(j * k),
        (other, _) => panic!("{other:?} is not a voxel row count"),
    }
}

/// The row dim a shrink by `k` lands: the factor is divided out where the
/// dim carries it, else the dim is kept and the rectangle over-allocates
/// (the grid says which rows are live).
fn shrunk(rows: Dim, k: u32) -> Dim {
    match rows {
        Dim::VoxelsTimes(j) if k > 1 && j % k == 0 && j / k > 1 => Dim::VoxelsTimes(j / k),
        Dim::VoxelsTimes(j) if k > 1 && j % k == 0 => Dim::Voxels,
        other => other,
    }
}

fn volume(r: [u32; 3]) -> u32 {
    r[0] * r[1] * r[2]
}

/// The static shape of one convolution — the fields `spatial.conv3d`
/// carries, named once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv {
    /// `[kt, kh, kw]`.
    pub k: [u32; 3],
    /// `[st, sh, sw]`.
    pub stride: [u32; 3],
    /// `[pt, ph, pw]`, the zero padding in FRONT of each axis; under
    /// `causal_t` `pt` is the front-only time padding.
    pub pad: [u32; 3],
    /// The zero padding BEHIND each axis — `pad` again for the symmetric
    /// convolutions every constructor here states; [`Conv::pad_back`]
    /// makes it asymmetric (diffusers' `Downsample2D`: `F.pad(x, (0, 1, 0,
    /// 1))` then a stride-2 3x3 with no padding is `conv2d([3, 3], [2, 2],
    /// [0, 0]).pad_back([0, 1, 1])`).
    pub pad_back: [u32; 3],
    /// Time padded in front only (from the cache when one is given).
    pub causal_t: bool,
    /// What a padded frame reads: the front frames under `causal_t` with
    /// no cache, both ends of a symmetric convolution
    /// ([`Conv::replicate_time`]).
    pub time_pad: TimePad,
}

impl Conv {
    /// A 2-D convolution (`kt = 1`, no time stride or padding).
    #[must_use]
    pub const fn conv2d(k: [u32; 2], stride: [u32; 2], pad: [u32; 2]) -> Conv {
        Conv {
            k: [1, k[0], k[1]],
            stride: [1, stride[0], stride[1]],
            pad: [0, pad[0], pad[1]],
            pad_back: [0, pad[0], pad[1]],
            causal_t: false,
            time_pad: TimePad::Zero,
        }
    }

    /// A 3-D convolution, symmetric in time.
    #[must_use]
    pub const fn conv3d(k: [u32; 3], stride: [u32; 3], pad: [u32; 3]) -> Conv {
        Conv {
            k,
            stride,
            pad,
            pad_back: pad,
            causal_t: false,
            time_pad: TimePad::Zero,
        }
    }

    /// The same convolution with the padding BEHIND each axis restated:
    /// `[t, h, w]` zero voxels after the box (the front stays `pad`).
    #[must_use]
    pub const fn pad_back(mut self, back: [u32; 3]) -> Conv {
        self.pad_back = back;
        self
    }

    /// The box-keeping 3×3×3: stride 1, pad 1 everywhere.
    #[must_use]
    pub const fn same3() -> Conv {
        Conv::conv3d([3, 3, 3], [1, 1, 1], [1, 1, 1])
    }

    /// The same shape with the time axis padded in front only by `kt - 1`
    /// (a causal video VAE's convolution).
    #[must_use]
    pub const fn causal(mut self, time_pad: TimePad) -> Conv {
        self.causal_t = true;
        self.pad[0] = self.k[0] - 1;
        self.pad_back[0] = 0;
        self.time_pad = time_pad;
        self
    }

    /// The same symmetric shape with its time padding read from the clip's
    /// own end frames instead of zeros — the first frame in front, the last
    /// behind. LTX-2.5's NON-causal decoder does this before every
    /// convolution (`torch.cat([x[:, :, :1], x, x[:, :, -1:]], dim=2)`),
    /// so a whole clip is one cacheless fire. `h`/`w` stay zero-padded.
    #[must_use]
    pub const fn replicate_time(mut self) -> Conv {
        self.time_pad = TimePad::Replicate;
        self
    }

    /// `kt·kh·kw`.
    #[must_use]
    pub const fn taps(&self) -> u32 {
        self.k[0] * self.k[1] * self.k[2]
    }

    /// The rule this convolution's output grid follows.
    #[must_use]
    pub const fn rule(&self) -> GridRule {
        GridRule::Conv {
            k: self.k,
            stride: self.stride,
            pad: self.pad,
            pad_back: self.pad_back,
            causal_t: self.causal_t,
        }
    }
}

/// The output grid of one rule applied to `grid`, computed on the device.
/// The wrappers below call it for you; a text calls it directly only to
/// derive a grid it feeds a cross-axis op with.
#[must_use]
pub fn grid(grid: &Value, rule: GridRule) -> Value {
    expect_grid("`spatial::grid`'s input", grid);
    let r = grid.rec();
    let y = r.fresh(grid_ty());
    r.push(
        Spatial::Grid {
            grid: grid.id(),
            rule,
            y: y.id(),
        },
        &[grid],
    );
    y
}

/// `y = conv(x)` over the clips of `grid`: `x` `[rows, C_in]` bf16, `w`
/// a `[C_out, C_in·taps]` weight declared with
/// [`Weight::conv_taps_major`], `bias` `[C_out]` f32, `cache` the causal
/// frame state (`Input::state(name)`) or `None`. Returns `(y, y_grid)`,
/// `y` `[rows, C_out]` at `x`'s dim: a convolution never grows rows.
#[must_use]
pub fn conv3d(
    x: &Value,
    grid: &Value,
    w: &Weight,
    bias: Option<&Weight>,
    conv: Conv,
    cache: Option<ValueId>,
) -> (Value, Value) {
    expect_voxels("`spatial::conv3d`'s input", x);
    expect_grid("`spatial::conv3d`'s grid", grid);
    assert_eq!(x.dtype(), Dtype::Bf16, "`spatial::conv3d` reads bf16 rows");
    let taps = conv.taps();
    assert!(
        w.layout
            == ParamLayout::ConvTapsMajor {
                c_in: x.width() as u32,
                taps
            },
        "`{}` must be declared `.conv_taps_major({}, {taps})` for this convolution; it is {:?}",
        w.name,
        x.width(),
        w.layout
    );
    assert!(
        w.shape.len() == 2 && w.shape[1] == x.width() * u64::from(taps),
        "`{}` is {:?}; `[C_out, {}·{taps}]` is what {} input channels convolve through",
        w.name,
        w.shape,
        x.width(),
        x.width()
    );
    if let Some(b) = bias {
        assert!(
            b.shape == vec![w.dim(0)] && b.dtype == Dtype::F32,
            "`{}` is {:?} {:?}; a conv bias is `[C_out]` f32",
            b.name,
            b.shape,
            b.dtype
        );
    }
    assert!(
        cache.is_none() || (conv.causal_t && conv.pad[0] > 0),
        "a frame cache is read only under `causal_t` with a front pad"
    );
    let r = x.rec();
    let y_grid = self::grid(grid, conv.rule());
    let y = r.fresh(tensor(x.rows(), w.dim(0), x.dtype()));
    r.push(
        Spatial::Conv3d {
            x: x.id(),
            grid: grid.id(),
            w: r.weight(w),
            bias: bias.map(|b| r.weight(b)),
            k: conv.k,
            stride: conv.stride,
            pad: conv.pad,
            pad_back: conv.pad_back,
            causal_t: conv.causal_t,
            time_pad: conv.time_pad,
            cache,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

/// `torch.nn.GroupNorm(groups, C, eps)` per clip, with an optional fused
/// SiLU. `weight`/`bias` are `[C]` f32. Fresh `y` at `x`'s type and grid.
#[must_use]
pub fn group_norm(
    x: &Value,
    grid: &Value,
    groups: u32,
    weight: &Weight,
    bias: &Weight,
    eps: f32,
    silu: bool,
) -> Value {
    expect_voxels("`spatial::group_norm`'s input", x);
    expect_grid("`spatial::group_norm`'s grid", grid);
    assert!(
        groups > 0 && x.width() % u64::from(groups) == 0,
        "{} channels do not split into {groups} groups",
        x.width()
    );
    for (what, w) in [("weight", weight), ("bias", bias)] {
        assert!(
            w.shape == vec![x.width()] && w.dtype == Dtype::F32,
            "the group norm {what} `{}` is {:?} {:?}; expected [{}] f32",
            w.name,
            w.shape,
            w.dtype,
            x.width()
        );
    }
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Spatial::GroupNorm {
            x: x.id(),
            grid: grid.id(),
            groups,
            weight: r.weight(weight),
            bias: r.weight(bias),
            eps,
            silu,
            y: y.id(),
        },
        &[x, grid],
    );
    y
}

/// The conv VAE's mid-block attention (`Spatial::Attention`): one head as
/// wide as the row, per clip over every voxel of the clip — `y =
/// softmax(q·kᵀ · sm_scale) · v`. `q`, `k`, `v` are `[rows, C]` bf16 on
/// the voxel axis at one type and grid; fresh `y` at `q`'s type. Not
/// `attn::ragged`, whose kernel is stamped at head widths 64/128/256 over
/// token-axis CSRs; a VAE's head is its whole channel row.
///
/// [`attention_over`] is the same op segmented by something narrower than
/// the clip.
#[must_use]
pub fn attention(q: &Value, k: &Value, v: &Value, grid: &Value, sm_scale: f32) -> Value {
    attention_over(q, k, v, grid, VoxelSegment::Clip, sm_scale)
}

/// [`attention`] over the block [`VoxelSegment`] names rather than the
/// whole clip: `VoxelSegment::Frames(1)` attends each frame on its own
/// (Wan 2.2's mid block), `Frames(n)` a run of `n` frames. The
/// segmentation is read off the same `[Clips, 4]` grid the rectangle
/// travels with — the voxel axis's answer to the token axis's
/// `GroupIndptr`/`LaneIndptr`.
#[must_use]
pub fn attention_over(
    q: &Value,
    k: &Value,
    v: &Value,
    grid: &Value,
    segment: VoxelSegment,
    sm_scale: f32,
) -> Value {
    expect_voxels("`spatial::attention`'s query", q);
    expect_grid("`spatial::attention`'s grid", grid);
    assert!(
        q.ty() == k.ty() && q.ty() == v.ty(),
        "`spatial::attention` reads q, k and v at one type; got {:?}, {:?}, {:?}",
        q.ty(),
        k.ty(),
        v.ty()
    );
    assert_eq!(
        q.dtype(),
        Dtype::Bf16,
        "`spatial::attention` reads bf16 rows"
    );
    assert!(
        segment != VoxelSegment::Frames(0),
        "`spatial::attention` over `Frames(0)` is a block with no rows in it"
    );
    let r = q.rec();
    let y = r.fresh(q.ty().clone());
    r.push(
        Spatial::Attention {
            q: q.id(),
            k: k.id(),
            v: v.id(),
            grid: grid.id(),
            segment,
            sm_scale,
            y: y.id(),
        },
        &[q, k, v, grid],
    );
    y
}

/// Nearest-neighbour upsample by `factor = [ft, fh, fw]`; under
/// `keep_first_frame` frame 0 is emitted once. Returns `(y, y_grid)`, `y`
/// grown by `ft·fh·fw` rows at `x`'s width.
#[must_use]
pub fn upsample_nearest(
    x: &Value,
    grid: &Value,
    factor: [u32; 3],
    keep_first_frame: bool,
) -> (Value, Value) {
    expect_voxels("`spatial::upsample_nearest`'s input", x);
    expect_grid("`spatial::upsample_nearest`'s grid", grid);
    assert!(
        volume(factor) > 0,
        "an upsample factor of {factor:?} is empty"
    );
    let rule = GridRule::Upsample {
        factor,
        keep_first_frame,
    };
    let r = x.rec();
    let y_grid = self::grid(grid, rule);
    let y = r.fresh(tensor(grown(x.rows(), rule.growth()), x.width(), x.dtype()));
    r.push(
        Spatial::UpsampleNearest {
            x: x.id(),
            grid: grid.id(),
            factor,
            keep_first_frame,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

/// Depth to space by `r`: `[rows, C·r1·r2·r3]` into `[rows·r1·r2·r3, C]`.
/// [`pixel_shuffle_trimming`] drops leading frames from the result.
#[must_use]
pub fn pixel_shuffle(x: &Value, grid: &Value, r: [u32; 3]) -> (Value, Value) {
    pixel_shuffle_trimming(x, grid, r, 0)
}

/// [`pixel_shuffle`] with a causal temporal upsampler's ANCHOR DROP: the
/// first `trim_t` frames of the shuffled result are thrown away, so a clip
/// of `t` frames lands `t·r1 - trim_t` of them (LTX-2.5's
/// `LTXVideoUpsampler3d` drops `r1 - 1`). The same flavour of statement as
/// [`upsample_nearest`]'s `keep_first_frame`: a time rule the box carries
/// and the rows follow. `y` keeps the untrimmed `VoxelsTimes(r1·r2·r3)`
/// row dim and over-allocates — the grid says which rows are live, and a
/// clip the trim would empty lands none.
#[must_use]
pub fn pixel_shuffle_trimming(x: &Value, grid: &Value, r: [u32; 3], trim_t: u32) -> (Value, Value) {
    expect_voxels("`spatial::pixel_shuffle`'s input", x);
    expect_grid("`spatial::pixel_shuffle`'s grid", grid);
    let vol = volume(r);
    assert!(
        vol > 0 && x.width() % u64::from(vol) == 0,
        "{} channels do not unpack by a {r:?} block",
        x.width()
    );
    let rule = GridRule::Shuffle { r, trim_t };
    let rec = x.rec();
    let y_grid = self::grid(grid, rule);
    let y = rec.fresh(tensor(
        grown(x.rows(), vol),
        x.width() / u64::from(vol),
        x.dtype(),
    ));
    rec.push(
        Spatial::PixelShuffle {
            x: x.id(),
            grid: grid.id(),
            r,
            trim_t,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

/// Space to depth by `r`: `[rows, C]` into `[rows / r1·r2·r3, C·r1·r2·r3]`;
/// every clip's box must divide by `r` (the device rule refuses one that
/// does not by landing no rows for it).
#[must_use]
pub fn pixel_unshuffle(x: &Value, grid: &Value, r: [u32; 3]) -> (Value, Value) {
    expect_voxels("`spatial::pixel_unshuffle`'s input", x);
    expect_grid("`spatial::pixel_unshuffle`'s grid", grid);
    let vol = volume(r);
    assert!(vol > 0, "an unshuffle block of {r:?} is empty");
    let rule = GridRule::Unshuffle { r };
    let rec = x.rec();
    let y_grid = self::grid(grid, rule);
    let y = rec.fresh(tensor(
        shrunk(x.rows(), vol),
        x.width() * u64::from(vol),
        x.dtype(),
    ));
    rec.push(
        Spatial::PixelUnshuffle {
            x: x.id(),
            grid: grid.id(),
            r,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

/// `AvgDown3D` by `factor = [ft, fh, fw]`, the widened channel averaged in
/// contiguous runs of `group`: the time axis zero-padded IN FRONT to a
/// multiple of `ft`, a CHANNEL-MAJOR space-to-depth
/// ([`pixel_unshuffle`]'s own `(c, it, ih, iw)` order), then the mean of
/// each `group` consecutive widened channels. `[rows, C]` in, `[rows',
/// C·ft·fh·fw / group]` out; returns `(y, y_grid)`.
///
/// `group = fh·fw` is a spatial average pool that keeps the time block as
/// extra channels — Wan 2.2's every shortcut. `group = ft·fh·fw` is the
/// plain average pool over the whole block. `y` keeps `x`'s row dim and
/// over-allocates; the grid says which rows are live.
#[must_use]
pub fn avg_down(x: &Value, grid: &Value, factor: [u32; 3], group: u32) -> (Value, Value) {
    expect_voxels("`spatial::avg_down`'s input", x);
    expect_grid("`spatial::avg_down`'s grid", grid);
    let vol = volume(factor);
    assert!(vol > 0, "an avg-down block of {factor:?} is empty");
    let widened = x.width() * u64::from(vol);
    assert!(
        group > 0 && widened % u64::from(group) == 0,
        "{widened} widened channels do not fold into groups of {group}"
    );
    let rule = GridRule::AvgDown { factor };
    let rec = x.rec();
    let y_grid = self::grid(grid, rule);
    let y = rec.fresh(tensor(x.rows(), widened / u64::from(group), x.dtype()));
    rec.push(
        Spatial::AvgDown {
            x: x.id(),
            grid: grid.id(),
            factor,
            group,
            y_grid: y_grid.id(),
            y: y.id(),
        },
        &[x, grid, &y_grid],
    );
    (y, y_grid)
}

/// Write this tile's last `frames` frames of `x` into each lane's slot of
/// the causal frame cache `cache` (an `Input::state` slab) and hand `x`
/// back — the store half of [`conv3d`]'s cache with no convolution around
/// it.
///
/// Wan 2.2's encoder head is the caller: its `downsample3d` resampler does
/// not convolve on the first chunk at all, it only remembers the frames
/// the NEXT chunk's convolution will pad with. The answer aliases `x`, so
/// a text writes `let x = spatial::store_frames(&x, &g, cache, 1);` and the
/// node sits on the dataflow where the reference's assignment sits.
#[must_use]
pub fn store_frames(x: &Value, grid: &Value, cache: ValueId, frames: u32) -> Value {
    expect_voxels("`spatial::store_frames`'s input", x);
    expect_grid("`spatial::store_frames`'s grid", grid);
    assert!(frames > 0, "a cache of no frames is no cache");
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Spatial::CacheStore {
            x: x.id(),
            grid: grid.id(),
            frames,
            cache,
            x_out: x_out.id(),
        },
        &[x, grid],
    );
    x_out
}

/// Voxels to patch tokens: `[rows, C]` over `grid` into `[Tokens,
/// C·pt·ph·pw]`, the token side laid out by `tgrid`
/// (`Input::token_grid(p)`). The one op that leaves the voxel axis.
#[must_use]
pub fn patchify(x: &Value, grid: &Value, p: [u32; 3], tgrid: &Value) -> Value {
    expect_voxels("`spatial::patchify`'s input", x);
    expect_grid("`spatial::patchify`'s grid", grid);
    expect_grid("`spatial::patchify`'s token grid", tgrid);
    let vol = volume(p);
    assert!(vol > 0, "a patch of {p:?} is empty");
    let r = x.rec();
    let y = r.fresh(tensor(Dim::Tokens, x.width() * u64::from(vol), x.dtype()));
    r.push(
        Spatial::Patchify {
            x: x.id(),
            grid: grid.id(),
            p,
            tgrid: tgrid.id(),
            y: y.id(),
        },
        &[x, grid, tgrid],
    );
    y
}

/// Patch tokens to voxels: `[Tokens, C·pt·ph·pw]` read through `tgrid`
/// into `[Voxels, C]` laid out by `grid` — the port grid, which the text
/// asserts is the token grid times `p`. The one op that enters the voxel
/// axis from the token one.
#[must_use]
pub fn unpatchify(x: &Value, tgrid: &Value, p: [u32; 3], grid: &Value) -> Value {
    assert!(
        x.rows().axis() == Some(RowAxis::Tokens),
        "`spatial::unpatchify` reads token rows, not {:?}",
        x.rows()
    );
    expect_grid("`spatial::unpatchify`'s token grid", tgrid);
    expect_grid("`spatial::unpatchify`'s grid", grid);
    let vol = volume(p);
    assert!(
        vol > 0 && x.width() % u64::from(vol) == 0,
        "{} channels do not unpack by a {p:?} patch",
        x.width()
    );
    let r = x.rec();
    let y = r.fresh(tensor(Dim::Voxels, x.width() / u64::from(vol), x.dtype()));
    r.push(
        Spatial::Unpatchify {
            x: x.id(),
            tgrid: tgrid.id(),
            p,
            grid: grid.id(),
            y: y.id(),
        },
        &[x, tgrid, grid],
    );
    y
}
