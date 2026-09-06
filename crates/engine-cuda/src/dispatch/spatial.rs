//! `Spatial` (design D8): the voxel axis's family, one arm per member over
//! `kernels_cuda::spatial`. Every launch takes the fire's whole clip table
//! and finds a row's lane itself (`crate::voxels`), so an arm resolves each
//! operand whole and hands it over; the grid a member reads and the grid
//! it writes are both plan values (`spatial.grid` derives the second on
//! the device ahead of the member that reads it).
//!
//! The causal convolution's frame cache is three launches: the lanes'
//! slots gathered into a `[Σ frames·h·w, C_in]` scratch rectangle, the
//! convolution reading it as its front frames, and this tile's last
//! `frames` input frames stored back into the slots — so a chunked decode
//! carries state across fires in the slot the sequence owns, zeroed with
//! every other state row when the slot is opened fresh.

use kernels_cuda::spatial;
use kernels_cuda::tensor::Tensor;
use model_exec::{DispatchSpatial, KernelError};
use model_ir::{GridRule, Spatial, TimePad};

use crate::run::Run;

impl DispatchSpatial for Run<'_> {
    fn dispatch(&mut self, op: &Spatial) -> Result<(), KernelError> {
        self.spatial(op).map_err(crate::error::kernel)
    }
}

fn rule(rule: GridRule) -> spatial::GridRule {
    match rule {
        GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        } => spatial::GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        },
        GridRule::Upsample {
            factor,
            keep_first_frame,
        } => spatial::GridRule::Upsample {
            factor,
            keep_first_frame,
        },
        GridRule::Shuffle { r } => spatial::GridRule::Shuffle { r },
        GridRule::Unshuffle { r } => spatial::GridRule::Unshuffle { r },
    }
}

impl Run<'_> {
    /// Arms in `kernels-cuda`'s error vocabulary, lifted by
    /// [`kernel`](crate::error::kernel) above the match.
    fn spatial(&mut self, op: &Spatial) -> Result<(), kernels_cuda::Error> {
        match op {
            Spatial::Grid { grid, rule: how, y } => spatial::derive_grid(
                self.ctx(),
                self.tensor(*grid),
                rule(*how),
                &mut self.tensor(*y),
            ),
            Spatial::Conv3d {
                x,
                grid,
                w,
                bias,
                k,
                stride,
                pad,
                pad_back,
                causal_t,
                time_pad,
                cache,
                y_grid,
                y,
            } => {
                let conv = spatial::Conv3d {
                    k: *k,
                    stride: *stride,
                    pad: *pad,
                    pad_back: *pad_back,
                    causal_t: *causal_t,
                    time_pad: match time_pad {
                        TimePad::Zero => spatial::TimePad::Zero,
                        TimePad::Replicate => spatial::TimePad::Replicate,
                    },
                };
                let x = self.tensor(*x);
                let grid = self.tensor(*grid);
                let bias = bias.map(|b| self.tensor(b));
                let Some(state) = cache else {
                    return spatial::conv3d(
                        self.ctx(),
                        x,
                        grid,
                        self.tensor(*w),
                        bias,
                        conv,
                        None,
                        &mut self.tensor(*y),
                        self.tensor(*y_grid),
                    );
                };
                // The frame cache: gather, convolve, store.
                let frames = conv.pad[0];
                let pool = self.recurrent(*state);
                let slot_ids = self
                    .clip_slots()
                    .ok_or_else(|| kernels_cuda::Error::Backend {
                        op: "spatial.conv3d",
                        detail:
                            "a causal convolution with a frame cache needs the fire's clip slot \
                             table, which no lane of it staged"
                                .to_string(),
                    })?;
                // Bounded above by `frames` copies of the input rectangle;
                // the kernels read exactly `Σ frames·h·w` rows of it.
                let rows = x.rows.saturating_mul(frames);
                let bytes = u64::from(rows) * u64::from(x.width) * 2;
                let scratch = self.ctx().scratch(
                    "spatial.conv3d",
                    "spatial.conv3d.frame_cache",
                    usize::try_from(bytes).unwrap_or(usize::MAX),
                )? as u64;
                let mut frame_cache = Tensor::new(scratch, rows, x.width, x.dtype);
                spatial::cache_gather(
                    self.ctx(),
                    pool.slab,
                    slot_ids,
                    grid,
                    frames,
                    &mut frame_cache,
                )?;
                spatial::conv3d(
                    self.ctx(),
                    x,
                    grid,
                    self.tensor(*w),
                    bias,
                    conv,
                    Some(frame_cache),
                    &mut self.tensor(*y),
                    self.tensor(*y_grid),
                )?;
                let mut slab = pool.slab;
                spatial::cache_store(
                    self.ctx(),
                    x,
                    frame_cache,
                    slot_ids,
                    grid,
                    frames,
                    &mut slab,
                )
            }
            Spatial::GroupNorm {
                x,
                grid,
                groups,
                weight,
                bias,
                eps,
                silu,
                y,
            } => spatial::group_norm(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *groups,
                self.tensor(*weight),
                self.tensor(*bias),
                *eps,
                *silu,
                &mut self.tensor(*y),
            ),
            Spatial::Attention {
                q,
                k,
                v,
                grid,
                sm_scale,
                y,
            } => spatial::attention(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*grid),
                *sm_scale,
                &mut self.tensor(*y),
            ),
            Spatial::UpsampleNearest {
                x,
                grid,
                factor,
                keep_first_frame,
                y_grid,
                y,
            } => spatial::upsample_nearest(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *factor,
                *keep_first_frame,
                &mut self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Spatial::PixelShuffle {
                x,
                grid,
                r,
                y_grid,
                y,
            } => spatial::pixel_shuffle(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *r,
                &mut self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Spatial::PixelUnshuffle {
                x,
                grid,
                r,
                y_grid,
                y,
            } => spatial::pixel_unshuffle(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *r,
                &mut self.tensor(*y),
                self.tensor(*y_grid),
            ),
            Spatial::Patchify {
                x,
                grid,
                p,
                tgrid,
                y,
            } => spatial::patchify(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*grid),
                *p,
                &mut self.tensor(*y),
                self.tensor(*tgrid),
            ),
            Spatial::Unpatchify {
                x,
                tgrid,
                p,
                grid,
                y,
            } => spatial::unpatchify(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*tgrid),
                *p,
                &mut self.tensor(*y),
                self.tensor(*grid),
            ),
        }
    }
}
