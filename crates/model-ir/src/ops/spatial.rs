//! The VAE family (design D8): ops over the voxel axis. An activation is
//! `[Dim::Voxels, channels]` — one row per voxel in `(t, h, w)` order, `w`
//! fastest, one clip's voxels contiguous — and the per-clip box lives in a
//! `[Dim::Clips, 4]` `i32` grid table `{t, h, w, row_offset}` beside it.
//! Nothing here folds space into the width: channels are the width, space
//! is the rows, and the grid says which row is which voxel.
//!
//! **GRIDS ARE VALUES.** The port's grid is [`RuntimeInput::Grid`]
//! (host-built); every later resolution's grid is computed ON THE DEVICE by
//! [`Spatial::Grid`] from its input's grid and a [`GridRule`] — one
//! single-block launch that applies the rule per clip and prefix-sums the
//! row offsets. A host-side derivation was the alternative (a prepare-phase
//! node, as attention plans are); it lost because every derived grid would
//! then need a host copy the shell keeps per fire, while the device rule is
//! capturable, stateless, and reads exactly what the kernels read.
//!
//! **ROW COUNTS.** `Dim::Voxels` is the port's voxel count; an op that grows
//! rows by a fixed factor (upsample, shuffle, unpatchify) lands
//! `Dim::VoxelsTimes(k)`, and one that shrinks them (a strided conv, an
//! unshuffle) keeps its input's dim and leaves the tail rows past the grid's
//! claim zero. The grid — not the dim — says which rows are live.
//!
//! **THE CROSS-AXIS PAIR.** [`Patchify`](Spatial::Patchify) and
//! [`Unpatchify`](Spatial::Unpatchify) are the ONLY nodes reading one row
//! axis and writing the other: the voxel rectangle against the token one,
//! the way `layout.scatter_rows` reads patch rows into token rows. Their
//! token side is described by [`RuntimeInput::TokenGrid`] — the same clip
//! table at token resolution, with row offsets into the TOKEN rectangle
//! (a clip's tokens are its lane's rows, clips of one lane consecutive).
//! The compiler places each on the unit of the axis it WRITES
//! (`unit::node_axis` reads outputs), so a patchify closes the voxel unit
//! and opens the token one, and an unpatchify the reverse.
//!
//! **CAUSAL TIME AND THE FRAME CACHE.** A `Conv3d` with `causal_t` pads
//! `pad[0]` frames in front only. With `cache: Some(state)` — a
//! `CacheRow::State` slab the text declares per causal conv, `[(kt-1) ·
//! max_plane, C_in]` per slot — the front frames are the previous tile's
//! last `kt-1` input frames of the same lane, gathered from the lane's slot
//! before the launch, and this tile's last `kt-1` input frames are stored
//! back after it, so a chunked decode carries state across fires. A slot
//! opened fresh is zero (`RsReset` zeroes every state row of the slot), which
//! is the zero-padded first tile; [`TimePad::Replicate`] is for the
//! cacheless single-tile case. WITHOUT `causal_t` the same flag pads BOTH
//! ends of the clip with its own end frames — LTX-2.5's non-causal decoder
//! (`torch.cat([x[:, :, :1], x, x[:, :, -1:]], dim=2)` before every conv),
//! which decodes a whole clip in one fire and carries no cache.

use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// What a padded frame reads: the frames before a clip under `causal_t`
/// with no cache, and the frames on both sides of it otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimePad {
    /// Zeros — the zero-padded causal convolution, or a plain zero-padded
    /// symmetric one.
    Zero,
    /// The clip's own end frames, repeated: frame 0 in front and — for a
    /// symmetric convolution — the last frame behind.
    Replicate,
}

/// How the voxel rows of one clip split into ATTENTION BLOCKS — the voxel
/// axis's answer to the token axis's `GroupIndptr` / `LaneIndptr`
/// (`IMAGEGEN_CONTRACT.md` §1), which say the same thing about token rows.
///
/// A CSR would need a rectangle whose length is the segment count, which on
/// this axis is `Σ t` over the clips and so is not a dim any budget states.
/// It does not need one: the segments a VAE attends over are REGULAR — a
/// whole clip, or a fixed run of frames inside it — so the `[Clips, 4]`
/// grid that already travels beside every voxel rectangle carries the
/// table, and this enum says how to read it. `segment.bounds(box, row)` is
/// the host twin of what the kernel computes per query row.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoxelSegment {
    /// One block per clip: every voxel of the clip attends every other
    /// (the image VAEs' mid block — FLUX's, Z-Image's).
    #[default]
    Clip,
    /// One block per run of `frames` consecutive frames of the clip: a
    /// query sees the `frames · h · w` voxels of its own run and nothing
    /// else. `Frames(1)` is Wan 2.2's mid block, which attends each frame
    /// on its own; a clip whose `t` is not a multiple leaves a short run at
    /// the end. `Frames(0)` is refused by the shells as a segment that
    /// holds no rows.
    Frames(u32),
}

impl VoxelSegment {
    /// The frame range `[begin, end)` of the block a voxel in frame
    /// `frame` of a `t`-frame clip attends.
    #[must_use]
    pub fn frames(self, t: u32, frame: u32) -> (u32, u32) {
        match self {
            VoxelSegment::Clip => (0, t),
            VoxelSegment::Frames(0) => (frame, frame),
            VoxelSegment::Frames(per) => {
                let begin = frame / per * per;
                (begin, (begin + per).min(t))
            }
        }
    }

    /// The row range `[begin, end)` — relative to the clip's `row_offset` —
    /// that the voxel at clip-local row `row` of a `[t, h, w]` clip
    /// attends. `None` for a row outside the clip.
    #[must_use]
    pub fn bounds(self, [t, h, w]: [u32; 3], row: u32) -> Option<(u32, u32)> {
        let plane = h * w;
        if plane == 0 || row >= t * plane {
            return None;
        }
        let (begin, end) = self.frames(t, row / plane);
        Some((begin * plane, end * plane))
    }
}

/// How one op's output box follows from its input box, per clip — what
/// [`Spatial::Grid`] applies on the device and what a DSL wrapper reads to
/// type the output rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GridRule {
    /// A convolution: `(n + front + back - k) / stride + 1` per axis, the
    /// front pad `pad`, the back pad `pad_back` (a symmetric convolution
    /// states them equal), the time axis padded only in front under
    /// `causal_t`.
    Conv {
        k: [u32; 3],
        stride: [u32; 3],
        pad: [u32; 3],
        pad_back: [u32; 3],
        causal_t: bool,
    },
    /// Nearest upsample: `(t·ft, h·fh, w·fw)`, or `1 + (t-1)·ft` frames
    /// under `keep_first_frame`.
    Upsample {
        factor: [u32; 3],
        keep_first_frame: bool,
    },
    /// Depth to space: `(t·r1, h·r2, w·r3)`, less the first `trim_t`
    /// frames of the result. The trim is the causal video VAEs' ANCHOR
    /// DROP: a temporal upsampler expands one latent frame into `r1`
    /// sample frames and then throws the leading `r1 - 1` of them away, so
    /// a clip of `t` latent frames lands `t·r1 - trim_t` sample frames
    /// (LTX-2.5's `LTXVideoUpsampler3d`, study §I.9). `trim_t = 0` is the
    /// plain shuffle, and the same flavour of statement as
    /// [`Upsample::keep_first_frame`](GridRule::Upsample): a time rule that
    /// the box carries and the rows follow.
    Shuffle { r: [u32; 3], trim_t: u32 },
    /// Space to depth: `(t/r1, h/r2, w/r3)`; every box must divide.
    Unshuffle { r: [u32; 3] },
}

impl GridRule {
    /// The output box of one input box, or `None` for a box the rule
    /// cannot map (smaller than the kernel, not divisible by the block).
    /// The device kernel computes the same thing; this is the host's copy,
    /// for readback and for tests.
    #[must_use]
    pub fn out_extent(self, [t, h, w]: [u32; 3]) -> Option<[u32; 3]> {
        match self {
            GridRule::Conv {
                k,
                stride,
                pad,
                pad_back,
                causal_t,
            } => {
                let axis = |n: u32, k: u32, s: u32, front: u32, back: u32| {
                    (n + front + back)
                        .checked_sub(k)
                        .map(|span| span / s.max(1) + 1)
                };
                let back_t = if causal_t { 0 } else { pad_back[0] };
                Some([
                    axis(t, k[0], stride[0], pad[0], back_t)?,
                    axis(h, k[1], stride[1], pad[1], pad_back[1])?,
                    axis(w, k[2], stride[2], pad[2], pad_back[2])?,
                ])
            }
            GridRule::Upsample {
                factor,
                keep_first_frame,
            } => {
                let t_out = if keep_first_frame && t > 0 {
                    1 + (t - 1) * factor[0]
                } else {
                    t * factor[0]
                };
                Some([t_out, h * factor[1], w * factor[2]])
            }
            GridRule::Shuffle { r, trim_t } => {
                // The anchor drop cannot eat the whole clip: a box it
                // empties is a box this rule does not map.
                let t_out = (t * r[0]).checked_sub(trim_t).filter(|n| *n > 0)?;
                Some([t_out, h * r[1], w * r[2]])
            }
            GridRule::Unshuffle { r } => {
                if r.iter().any(|&x| x == 0) || t % r[0] != 0 || h % r[1] != 0 || w % r[2] != 0 {
                    return None;
                }
                Some([t / r[0], h / r[1], w / r[2]])
            }
        }
    }

    /// By how much this rule can grow a rectangle's rows — the factor an
    /// output dim is `VoxelsTimes` by. `1` for a rule that never grows.
    #[must_use]
    pub fn growth(self) -> u32 {
        match self {
            GridRule::Conv { .. } | GridRule::Unshuffle { .. } => 1,
            GridRule::Upsample { factor, .. } => factor[0] * factor[1] * factor[2],
            // The trim only removes rows, so the block volume still bounds
            // the growth: a trimmed shuffle lands inside the rectangle a
            // plain one would.
            GridRule::Shuffle { r, .. } => r[0] * r[1] * r[2],
        }
    }

    /// The output table of an input table, row offsets prefix-summed in
    /// clip order — the host twin of the `spatial.grid` kernel. `None` when
    /// a clip's box does not map.
    #[must_use]
    pub fn apply(self, grid: &[i32]) -> Option<Vec<i32>> {
        let mut out = Vec::with_capacity(grid.len());
        let mut off: i64 = 0;
        for clip in grid.chunks_exact(4) {
            let boxed = [clip[0], clip[1], clip[2]].map(|n| u32::try_from(n).ok());
            let [t, h, w] = self.out_extent([boxed[0]?, boxed[1]?, boxed[2]?])?;
            out.extend_from_slice(&[t as i32, h as i32, w as i32, i32::try_from(off).ok()?]);
            off += i64::from(t) * i64::from(h) * i64::from(w);
        }
        Some(out)
    }
}

/// Ops over the voxel axis. Every `grid` is a `[Clips, 4]` `i32` table;
/// activations are `[rows, channels]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Spatial {
    /// `y = rule(grid)`: the output grid of one op, computed on the device
    /// per clip with prefix-summed row offsets. `[Clips, 4]` `i32` in, the
    /// same out.
    Grid {
        grid: ValueId,
        rule: GridRule,
        y: ValueId,
    },
    /// Implicit-GEMM convolution over `x: [rows, C_in]` into
    /// `y: [rows, C_out]`; `conv2d` is `k[0] == 1`. `w` is
    /// `[C_out, kt·kh·kw·C_in]` bf16 in TAP-MAJOR CHANNEL-FASTEST order —
    /// a checkpoint's natural `[C_out, C_in·kt·kh·kw]` rectangle is
    /// relabelled once at load (`ParamLayout::ConvTapsMajor`); `bias` is
    /// `[C_out]` f32. `y_grid` is `Grid { rule: Conv {..} }` of `grid`.
    /// `pad` is the zero padding IN FRONT of each axis and `pad_back` the
    /// padding BEHIND it (`F.pad(x, (0, 1, 0, 1))` before a stride-2 conv,
    /// the diffusers `Downsample2D`, is `pad [0, 0, 0]`, `pad_back [0, 1,
    /// 1]`); a symmetric convolution states them equal. Only the front pad
    /// shifts the tap window; the back pad reaches the kernel through the
    /// output box alone. `cache`: a `Def::Cache` state slab for the front
    /// frames under `causal_t` (module doc), read before and written after
    /// the launch. fp32 accumulation, one rounding at the store.
    Conv3d {
        x: ValueId,
        grid: ValueId,
        w: ValueId,
        bias: Option<ValueId>,
        k: [u32; 3],
        stride: [u32; 3],
        pad: [u32; 3],
        pad_back: [u32; 3],
        causal_t: bool,
        time_pad: TimePad,
        cache: Option<ValueId>,
        y_grid: ValueId,
        y: ValueId,
    },
    /// `torch.nn.GroupNorm` per clip: `y = silu?((x - mean) · rsqrt(var +
    /// eps) · weight[c] + bias[c])`, moments over every voxel of the clip
    /// times every channel of the group. `weight`/`bias` are `[C]` f32.
    /// Fresh `y` at `x`'s type; fp32 Welford moments.
    GroupNorm {
        x: ValueId,
        grid: ValueId,
        groups: u32,
        weight: ValueId,
        bias: ValueId,
        eps: f32,
        silu: bool,
        y: ValueId,
    },
    /// The conv VAE's mid-block attention: ONE head as wide as the row,
    /// over the block `segment` names — `y = softmax(q·kᵀ · sm_scale) · v`
    /// with `q`, `k`, `v`, `y` all `[rows, C]` bf16 on the voxel axis, the
    /// blocks read off `grid` through
    /// [`VoxelSegment`]: the whole clip (the image VAEs) or a run of
    /// frames inside it (Wan 2.2's mid block attends one frame at a time). Not `attention.ragged`: that
    /// kernel is stamped at head widths 64/128/256 and a VAE's head is its
    /// whole channel row (512 on the FLUX VAE), and its CSR is a token-axis
    /// table. fp32 scores, fp32 online softmax (the reference's
    /// `upcast_softmax`), fp32 accumulation, one rounding at the store. A
    /// plain online-softmax walk over the clip's keys, not a flash tiling:
    /// a VAE attends at its lowest resolution (`h·w` of a few thousand).
    Attention {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        grid: ValueId,
        segment: VoxelSegment,
        sm_scale: f32,
        y: ValueId,
    },
    /// Nearest-neighbour upsample by `factor = [ft, fh, fw]`; frame 0 is
    /// emitted once and every later frame `ft` times under
    /// `keep_first_frame` (the causal video VAEs). `y` is
    /// `[VoxelsTimes(k·ft·fh·fw), C]`.
    UpsampleNearest {
        x: ValueId,
        grid: ValueId,
        factor: [u32; 3],
        keep_first_frame: bool,
        y_grid: ValueId,
        y: ValueId,
    },
    /// Depth to space: `[rows, C·r1·r2·r3]` over `(t, h, w)` into
    /// `[rows·r1·r2·r3, C]` over `(t·r1, h·r2, w·r3)`, einops
    /// `'b (c r1 r2 r3) t h w -> b c (t r1) (h r2) (w r3)'`, LESS the first
    /// `trim_t` frames of the result — a causal temporal upsampler's anchor
    /// drop (`GridRule::Shuffle`). `y_grid` is `Grid { rule: Shuffle { r,
    /// trim_t } }` of `grid`, so `y`'s live rows shrink with the box while
    /// its dim keeps the untrimmed `VoxelsTimes(r1·r2·r3)` allocation.
    PixelShuffle {
        x: ValueId,
        grid: ValueId,
        r: [u32; 3],
        trim_t: u32,
        y_grid: ValueId,
        y: ValueId,
    },
    /// Space to depth, [`PixelShuffle`](Spatial::PixelShuffle) inverted;
    /// every clip's box must divide by `r`.
    PixelUnshuffle {
        x: ValueId,
        grid: ValueId,
        r: [u32; 3],
        y_grid: ValueId,
        y: ValueId,
    },
    /// Voxels to patch tokens: `[rows, C]` over `(t, h, w)` into
    /// `[Tokens, C·pt·ph·pw]` — the DiT boundary's name for an unshuffle by
    /// the patch `p`, landing on the TOKEN axis. `tgrid` is
    /// [`RuntimeInput::TokenGrid`](crate::RuntimeInput::TokenGrid): the
    /// clip table at token resolution with token-rectangle row offsets.
    Patchify {
        x: ValueId,
        grid: ValueId,
        p: [u32; 3],
        tgrid: ValueId,
        y: ValueId,
    },
    /// Patch tokens to voxels: `[Tokens, C·pt·ph·pw]` read through `tgrid`
    /// into `[Voxels, C]` laid out by `grid` (the port grid — the text
    /// asserts the voxel grid is the token grid times `p`).
    Unpatchify {
        x: ValueId,
        tgrid: ValueId,
        p: [u32; 3],
        grid: ValueId,
        y: ValueId,
    },
}

impl Operands for Spatial {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Grid { grid, .. } => sink.push(*grid),
            Self::Conv3d {
                x,
                grid,
                w,
                bias,
                cache,
                y_grid,
                ..
            } => {
                sink.extend([*x, *grid, *w]);
                sink.extend(bias.iter().copied());
                sink.extend(cache.iter().copied());
                sink.push(*y_grid);
            }
            Self::GroupNorm {
                x,
                grid,
                weight,
                bias,
                ..
            } => sink.extend([*x, *grid, *weight, *bias]),
            Self::Attention { q, k, v, grid, .. } => sink.extend([*q, *k, *v, *grid]),
            Self::UpsampleNearest {
                x, grid, y_grid, ..
            }
            | Self::PixelShuffle {
                x, grid, y_grid, ..
            }
            | Self::PixelUnshuffle {
                x, grid, y_grid, ..
            } => sink.extend([*x, *grid, *y_grid]),
            Self::Patchify { x, grid, tgrid, .. } => sink.extend([*x, *grid, *tgrid]),
            Self::Unpatchify { x, tgrid, grid, .. } => sink.extend([*x, *tgrid, *grid]),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Grid { y, .. }
            | Self::Conv3d { y, .. }
            | Self::GroupNorm { y, .. }
            | Self::Attention { y, .. }
            | Self::UpsampleNearest { y, .. }
            | Self::PixelShuffle { y, .. }
            | Self::PixelUnshuffle { y, .. }
            | Self::Patchify { y, .. }
            | Self::Unpatchify { y, .. } => sink.push(*y),
        }
    }
    fn aliases(&self, _sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            // Every member lands a fresh rectangle: a convolution reads
            // every neighbour of a row, so in place would be a race.
            Self::Grid { .. }
            | Self::Conv3d { .. }
            | Self::GroupNorm { .. }
            | Self::Attention { .. }
            | Self::UpsampleNearest { .. }
            | Self::PixelShuffle { .. }
            | Self::PixelUnshuffle { .. }
            | Self::Patchify { .. }
            | Self::Unpatchify { .. } => {}
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Grid { .. } => "spatial.grid",
            Self::Conv3d { .. } => "spatial.conv3d",
            Self::GroupNorm { .. } => "spatial.group_norm",
            Self::Attention { .. } => "spatial.attention",
            Self::UpsampleNearest { .. } => "spatial.upsample_nearest",
            Self::PixelShuffle { .. } => "spatial.pixel_shuffle",
            Self::PixelUnshuffle { .. } => "spatial.pixel_unshuffle",
            Self::Patchify { .. } => "spatial.patchify",
            Self::Unpatchify { .. } => "spatial.unpatchify",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GridRule, VoxelSegment};

    /// The host rule agrees with the shapes `torch` lands: a k=3 s=2 p=1
    /// conv halves (rounding up), a causal one pads its time front only,
    /// an upsample keeps the first frame once, and offsets prefix-sum.
    #[test]
    fn a_grid_rule_maps_boxes_the_way_torch_does() {
        let conv = GridRule::Conv {
            k: [3, 3, 3],
            stride: [1, 2, 2],
            pad: [1, 1, 1],
            pad_back: [1, 1, 1],
            causal_t: false,
        };
        assert_eq!(conv.out_extent([5, 7, 8]), Some([5, 4, 4]));
        let causal = GridRule::Conv {
            k: [3, 3, 3],
            stride: [1, 1, 1],
            pad: [2, 1, 1],
            pad_back: [2, 1, 1],
            causal_t: true,
        };
        assert_eq!(causal.out_extent([5, 7, 8]), Some([5, 7, 8]));
        // diffusers' `Downsample2D`: `F.pad(x, (0, 1, 0, 1))` then a 3x3
        // stride-2 convolution with no padding of its own halves an even box.
        let down = GridRule::Conv {
            k: [1, 3, 3],
            stride: [1, 2, 2],
            pad: [0, 0, 0],
            pad_back: [0, 1, 1],
            causal_t: false,
        };
        assert_eq!(down.out_extent([1, 64, 64]), Some([1, 32, 32]));
        assert_eq!(down.out_extent([1, 7, 9]), Some([1, 3, 4]));
        let up = GridRule::Upsample {
            factor: [2, 2, 2],
            keep_first_frame: true,
        };
        assert_eq!(up.out_extent([3, 4, 4]), Some([5, 8, 8]));
        assert_eq!(up.growth(), 8);
        assert_eq!(
            GridRule::Unshuffle { r: [1, 2, 2] }.out_extent([1, 3, 4]),
            None
        );
        let table = up.apply(&[1, 2, 2, 0, 3, 4, 4, 4]).expect("both boxes map");
        assert_eq!(table, vec![1, 4, 4, 0, 5, 8, 8, 16]);
    }

    /// A trimmed shuffle is the plain one minus its leading frames: the box
    /// loses `trim_t` frames, the offsets prefix-sum over the SHORTER
    /// boxes, and a trim that would empty a clip maps nothing.
    #[test]
    fn a_trimmed_shuffle_drops_its_anchor_frames_from_the_box() {
        let plain = GridRule::Shuffle {
            r: [2, 2, 2],
            trim_t: 0,
        };
        let trimmed = GridRule::Shuffle {
            r: [2, 2, 2],
            trim_t: 1,
        };
        assert_eq!(plain.out_extent([3, 4, 5]), Some([6, 8, 10]));
        assert_eq!(trimmed.out_extent([3, 4, 5]), Some([5, 8, 10]));
        // The rectangle a trimmed shuffle lands still fits the untrimmed
        // growth, which is what the row dim allocates.
        assert_eq!(trimmed.growth(), 8);
        assert_eq!(
            trimmed.apply(&[1, 1, 1, 0, 3, 4, 5, 1]),
            Some(vec![1, 2, 2, 0, 5, 8, 10, 4])
        );
        // `t·r1 == trim_t` leaves no frame at all, which is no box.
        assert_eq!(
            GridRule::Shuffle {
                r: [1, 2, 2],
                trim_t: 1
            }
            .out_extent([1, 2, 2]),
            None
        );
    }

    /// The two regular blocks a spatial attention segments by, read off a
    /// clip's box alone: the whole clip, or the run of frames a query sits
    /// in (short at the end when `t` does not divide).
    #[test]
    fn a_voxel_segment_reads_its_block_off_the_clips_box() {
        let clip = [3, 2, 2];
        assert_eq!(VoxelSegment::Clip.bounds(clip, 0), Some((0, 12)));
        assert_eq!(VoxelSegment::Clip.bounds(clip, 11), Some((0, 12)));
        for (row, want) in [(0, (0, 4)), (3, (0, 4)), (4, (4, 8)), (11, (8, 12))] {
            assert_eq!(VoxelSegment::Frames(1).bounds(clip, row), Some(want));
        }
        // Two frames a block over three frames: the tail block is short.
        assert_eq!(VoxelSegment::Frames(2).bounds(clip, 0), Some((0, 8)));
        assert_eq!(VoxelSegment::Frames(2).bounds(clip, 9), Some((8, 12)));
        // A row past the clip is no row.
        assert_eq!(VoxelSegment::Frames(1).bounds(clip, 12), None);
    }
}
