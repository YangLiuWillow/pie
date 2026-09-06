//! The third row axis's seat (design D8): what a load reserves for a plan
//! that states voxel rows, and how one fire's clips become the device
//! tables the `spatial.*` kernels read.
//!
//! A fire's clips arrive per lane as boxes `[t, h, w]` beside the port's
//! payload rows; the composition places each lane's clips at
//! `clip_offset`/`voxel_offset` on the voxel axis, and this module lays
//! the port grid (`RuntimeInput::Grid`, `{t, h, w, row_offset}` per clip
//! in fire order), the token-side grid (`RuntimeInput::TokenGrid`, when the
//! plan reads one) and the payload rectangle out in that order, then stages
//! them below the fire's other inputs: device bytes of their own, no
//! pinned mirror, pageable-safe (the driver copies before the call
//! returns), the way the patch axis's payload is staged.
//!
//! **ONE VOXEL CLASS PER FIRE (M0).** Every spatial kernel takes the whole
//! grid table and finds a row's lane itself (`seat::Reads::Nothing`: it
//! reads no window seat), so a voxel-axis launch runs over the fire's
//! whole voxel rectangle. A fire whose clip-carrying lanes fall in two
//! classes would have each class's region compute over the other's rows
//! too; the shell refuses that at prepare rather than compute it twice.

use model_compiler::VoxelLadder;
use model_exec::fire::LaneRow;
use model_ir::Dtype;

use crate::device::Buffer;
use crate::error::{Fault, Result};
use kernels_cuda::Tensor;

/// The voxel axis's reservation, or `None` for a load whose plan states no
/// voxel row.
#[derive(Debug, Clone, Copy)]
pub struct Seat {
    /// Most port voxel rows one fire may carry (`VoxelLadder::max_voxels`).
    pub rows: u64,
    /// Most clips one fire may carry (`VoxelLadder::max_clips`).
    pub clips: u64,
    /// The port's channel count, or `0` for a plan that reads no voxel
    /// port (a decoder fed by `spatial.unpatchify` from tokens).
    pub channels: u32,
    /// The port's element.
    pub dtype: Dtype,
    /// The patch a `RuntimeInput::TokenGrid` reader states, or `None` for
    /// a plan that reads none.
    pub token_patch: Option<[u32; 3]>,
}

impl Seat {
    /// One port row's bytes.
    fn row_bytes(&self) -> u64 {
        u64::from(self.channels) * model_compiler::arena::elem_bytes(self.dtype).unwrap_or(0)
    }

    /// The seat a plan states, against a ladder: the ceilings are the
    /// ladder's, the port's width and element the plan's own.
    #[must_use]
    pub fn of(trace: &model_ir::Trace, ladder: &VoxelLadder) -> Seat {
        let mut channels = 0u32;
        let mut dtype = Dtype::Bf16;
        let mut token_patch = None;
        for decl in &trace.values {
            match (&decl.def, &decl.ty) {
                (
                    model_ir::Def::Input(model_ir::RuntimeInput::Voxels { channels: c, .. }),
                    model_ir::Ty::Tensor { dtype: d, .. },
                ) => {
                    channels = channels.max(*c);
                    dtype = *d;
                }
                (model_ir::Def::Input(model_ir::RuntimeInput::TokenGrid { p }), _) => {
                    token_patch = Some(*p);
                }
                _ => {}
            }
        }
        Seat {
            rows: u64::from(ladder.max_voxels),
            clips: u64::from(ladder.max_clips),
            channels,
            dtype,
            token_patch,
        }
    }
}

/// Where the voxel tables sit on the device: one buffer, three regions.
pub struct Store {
    seat: Seat,
    buffer: Buffer,
    grid: u64,
    token_grid: u64,
    slots: u64,
    payload: u64,
}

/// The device handles one fire's clips resolve to.
#[derive(Debug, Clone, Copy)]
pub struct Handles {
    /// `[clips, 4]` i32: the port grid.
    pub grid: Tensor,
    /// `[clips, 4]` i32: the token-side grid, or `None` for a plan that
    /// reads none.
    pub token_grid: Option<Tensor>,
    /// `[voxel rows, channels]`: the port payload, or `None` for a plan
    /// that reads no voxel port.
    pub voxels: Option<Tensor>,
    /// `[clips]` i32: the recurrent slot of each clip's lane.
    pub slots: Tensor,
}

/// One lane's clips as the caller submits them, already in the port's
/// element: `payload` is `Σ t·h·w` rows of `channels` elements, clips
/// concatenated in the order `clips` lists them.
#[derive(Debug, Clone, Copy)]
pub struct Clips<'a> {
    /// Which lane of the submission these clips belong to.
    pub lane: u32,
    /// Each clip's box `[t, h, w]` at the port's resolution.
    pub clips: &'a [[u32; 3]],
    /// The port rows, `Σ t·h·w` of them, in the port's element; empty for
    /// a plan that reads no voxel port.
    pub payload: &'a [u8],
}

impl Clips<'_> {
    /// How many port voxel rows these clips total.
    #[must_use]
    pub fn voxels(&self) -> u64 {
        self.clips
            .iter()
            .map(|[t, h, w]| u64::from(*t) * u64::from(*h) * u64::from(*w))
            .sum()
    }
}

/// The host side of one fire's voxel tables, laid out in fire order.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Tables {
    /// `{t, h, w, row_offset}` per clip, in fire order.
    pub grid: Vec<i32>,
    /// `{t/pt, h/ph, w/pw, token_row_offset}` per clip, in fire order;
    /// empty for a plan that reads none.
    pub token_grid: Vec<i32>,
    /// The port payload, clips in fire order.
    pub payload: Vec<u8>,
    /// The recurrent slot of each clip's lane, in fire order.
    pub slots: Vec<i32>,
}

impl Tables {
    /// Lay one fire's clips out in fire order. `lanes` is the composition's
    /// lane table (fire order, carrying each lane's `source`, `clip_offset`
    /// and `voxel_offset`); `of_lane[source]` is that lane's submission or
    /// `None`; `slot_of[source]` its recurrent slot.
    ///
    /// # Errors
    ///
    /// [`Fault::VoxelPayload`] for a payload whose bytes are not `Σ t·h·w`
    /// port rows, a clip box that does not divide by the token patch, or a
    /// lane whose token rows are not its clips' token count.
    pub fn of(
        seat: &Seat,
        lanes: &[LaneRow],
        of_lane: &[Option<Clips<'_>>],
        slot_of: &[u32],
    ) -> Result<Tables> {
        let clips_total: usize = lanes.iter().map(|lane| lane.clips as usize).sum();
        let voxels_total: usize = lanes.iter().map(|lane| lane.voxels as usize).sum();
        let row_bytes = seat.row_bytes() as usize;
        let mut grid = vec![0i32; clips_total * 4];
        let mut token_grid = if seat.token_patch.is_some() {
            vec![0i32; clips_total * 4]
        } else {
            Vec::new()
        };
        let mut payload = vec![0u8; voxels_total * row_bytes];
        let mut slots = vec![0i32; clips_total];
        for lane in lanes {
            let Some(shot) = of_lane.get(lane.source as usize).copied().flatten() else {
                continue;
            };
            if shot.clips.len() != lane.clips as usize || shot.voxels() != u64::from(lane.voxels) {
                return Err(Fault::VoxelPayload {
                    lane: lane.source,
                    what: "the clips composed are not the clips submitted",
                });
            }
            let need = shot.voxels() as usize * row_bytes;
            if seat.channels > 0 && shot.payload.len() != need {
                return Err(Fault::VoxelPayload {
                    lane: lane.source,
                    what: "the payload is not one port row per voxel of the clips",
                });
            }
            if seat.channels > 0 {
                let at = lane.voxel_offset as usize * row_bytes;
                payload[at..at + need].copy_from_slice(shot.payload);
            }
            let mut row: i64 = i64::from(lane.voxel_offset);
            let mut token_row: i64 = i64::from(lane.row_offset);
            let slot = slot_of.get(lane.source as usize).copied().unwrap_or(0) as i32;
            for (c, [t, h, w]) in shot.clips.iter().enumerate() {
                slots[lane.clip_offset as usize + c] = slot;
                let at = (lane.clip_offset as usize + c) * 4;
                grid[at..at + 4].copy_from_slice(&[*t as i32, *h as i32, *w as i32, row as i32]);
                row += i64::from(*t) * i64::from(*h) * i64::from(*w);
                if let Some(p) = seat.token_patch {
                    if p.contains(&0) || t % p[0] != 0 || h % p[1] != 0 || w % p[2] != 0 {
                        return Err(Fault::VoxelPayload {
                            lane: lane.source,
                            what: "a clip's box does not divide by the plan's token patch",
                        });
                    }
                    let [tt, th, tw] = [t / p[0], h / p[1], w / p[2]];
                    token_grid[at..at + 4].copy_from_slice(&[
                        tt as i32,
                        th as i32,
                        tw as i32,
                        token_row as i32,
                    ]);
                    token_row += i64::from(tt) * i64::from(th) * i64::from(tw);
                }
            }
            if seat.token_patch.is_some()
                && token_row != i64::from(lane.row_offset) + i64::from(lane.rows)
            {
                return Err(Fault::VoxelPayload {
                    lane: lane.source,
                    what: "the lane's token rows are not its clips' token count",
                });
            }
        }
        Ok(Tables {
            grid,
            token_grid,
            payload,
            slots,
        })
    }
}

impl Store {
    /// Reserve the tables at the seat's ceilings.
    ///
    /// # Errors
    ///
    /// [`Fault::OutOfMemory`] or [`Fault::Device`] for the allocation.
    pub fn reserve(seat: Seat) -> Result<Store> {
        let align = |bytes: u64| bytes.next_multiple_of(256);
        let grid = 0u64;
        let token_grid = align(seat.clips * 16);
        let slots = token_grid
            + if seat.token_patch.is_some() {
                align(seat.clips * 16)
            } else {
                0
            };
        let payload = slots + align(seat.clips * 4);
        let total = payload + align(seat.rows * seat.row_bytes());
        Ok(Store {
            seat,
            buffer: Buffer::zeroed(total as usize)?,
            grid,
            token_grid,
            slots,
            payload,
        })
    }

    /// The seat this store was reserved at.
    #[must_use]
    pub fn seat(&self) -> Seat {
        self.seat
    }

    /// Bytes the reservation holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.buffer.bytes() as u64
    }

    /// One fire's tables onto the device, on `stream`, ahead of the
    /// launches that read them.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a fire past the reserved tables,
    /// [`Fault::Device`] for the copies.
    pub fn stage(&mut self, stream: *mut core::ffi::c_void, tables: &Tables) -> Result<Handles> {
        let clips = (tables.grid.len() / 4) as u64;
        if clips > self.seat.clips {
            return Err(Fault::Ceiling {
                what: "clips",
                need: clips,
                have: self.seat.clips,
            });
        }
        let rows = (tables.payload.len() as u64)
            .checked_div(self.seat.row_bytes())
            .unwrap_or(0);
        if rows > self.seat.rows {
            return Err(Fault::Ceiling {
                what: "voxel rows",
                need: rows,
                have: self.seat.rows,
            });
        }
        let base = self.buffer.ptr();
        self.buffer
            .stage(stream, self.grid, i32_bytes(&tables.grid))?;
        if !tables.token_grid.is_empty() {
            self.buffer
                .stage(stream, self.token_grid, i32_bytes(&tables.token_grid))?;
        }
        self.buffer
            .stage(stream, self.slots, i32_bytes(&tables.slots))?;
        if !tables.payload.is_empty() {
            self.buffer.stage(stream, self.payload, &tables.payload)?;
        }
        Ok(Handles {
            grid: Tensor::new(base + self.grid, clips as u32, 4, Dtype::I32),
            token_grid: (!tables.token_grid.is_empty())
                .then(|| Tensor::new(base + self.token_grid, clips as u32, 4, Dtype::I32)),
            voxels: (self.seat.channels > 0).then(|| {
                Tensor::new(
                    base + self.payload,
                    rows as u32,
                    self.seat.channels,
                    self.seat.dtype,
                )
            }),
            slots: Tensor::new(base + self.slots, clips as u32, 1, Dtype::I32),
        })
    }
}

fn i32_bytes(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` has no padding and any bit pattern is a valid byte.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }
}

/// A payload's `f32` numbers in the port's element, little-endian —
/// round-to-nearest-even into bf16, as the patch path does.
///
/// # Errors
///
/// The `&'static str` an `Error::Unsupported` carries, for a port element
/// this marshal cannot write.
pub fn port_bytes(values: &[f32], element: Dtype) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        Dtype::Bf16 => Ok(values
            .iter()
            .flat_map(|&v| crate::adapter::bf16_bits(v).to_le_bytes())
            .collect()),
        Dtype::F32 => Ok(values.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => {
            Err("a voxel submission against a plan whose port element is neither `bf16` nor `f32`")
        }
    }
}

/// The load-time relabelling of every convolution weight
/// (`ParamLayout::ConvTapsMajor`): the checkpoint landed the natural
/// `[C_out, C_in·taps]` rectangle; the kernel reads `[C_out, taps·C_in]`.
/// Each such plane is copied aside, relabelled back into its own bytes by
/// `conv_weight_taps_major`, and the copy freed — once, before the first
/// fire, so no dispatch arm ever sees the natural order.
///
/// # Errors
///
/// [`Fault::Device`] for a copy, [`Fault::Unbound`] for the launch or for
/// a conv weight that landed as something other than one dense plane.
pub(crate) fn relabel_conv_weights(
    device: &crate::device::Context,
    trace: &model_ir::Trace,
    weights: &crate::run::WeightTable,
) -> Result<()> {
    for (at, param) in trace.params.iter().enumerate() {
        let model_ir::ParamLayout::ConvTapsMajor { c_in, taps } = param.layout else {
            continue;
        };
        let Some(crate::run::WeightRow::Dense(handle)) = weights.0.get(at).copied().flatten()
        else {
            return Err(Fault::Unbound {
                what: format!(
                    "`{}`, a convolution weight that did not land as one dense plane",
                    param.name
                ),
            });
        };
        let bytes = usize::try_from(u64::from(handle.rows) * u64::from(handle.width) * 2)
            .unwrap_or(usize::MAX);
        let aside = Buffer::zeroed(bytes)?;
        crate::device::alloc::copy_d2d(device.stream(), aside.ptr(), handle.ptr, bytes)?;
        let natural = Tensor::new(aside.ptr(), handle.rows, handle.width, handle.dtype);
        let mut relabelled = handle;
        kernels_cuda::spatial::conv_weight_taps_major(
            device.ctx(),
            natural,
            c_in,
            taps,
            &mut relabelled,
        )
        .map_err(|fault| Fault::Unbound {
            what: format!("`{}`: {fault}", param.name),
        })?;
        // The copy is freed on return; the launch that read it must be done.
        device.synchronize()?;
    }
    Ok(())
}
