//! The far end of a step: settle it asynchronously, and read its numbers back on request.

use engine::fire::{LayerScores, Readout};

use crate::error::{Fault, Result};

use super::{Enqueued, Shell};

/// Where the numbers a caller might want are: host arithmetic over the arena
/// carve, spent only if somebody asks.
#[derive(Debug, Clone)]
pub(crate) struct Readback {
    /// Per submitted lane: the seam its rows read back from and that seam's
    /// rectangle — the trunk logits (bf16), or the float seam the lane's arm
    /// plants (`velocity` / `hidden`, bf16 or f32); `None` for a lane whose
    /// answer is its pixels alone.
    pub(crate) lanes: Vec<Option<(engine::fire::ReadoutSeam, kernels_cuda::Tensor)>>,
    /// One (layer, rectangle) per exported attention column.
    pub(crate) columns: Vec<(u32, kernels_cuda::Tensor)>,
    /// Per submitted lane: its last row.
    pub(crate) last_row: Vec<u32>,
    /// Per submitted lane: its first row.
    pub(crate) first_row: Vec<u32>,
    /// Per submitted lane: how many rows it owns.
    pub(crate) lane_rows: Vec<u32>,
    /// Per submitted lane: whether it asked to capture.
    pub(crate) captures: Vec<bool>,
    /// The pixel plane and its grid (D8), for a fire that decoded clips.
    pub(crate) pixels: Option<PixelsReadback>,
}

/// Where a VAE decode's pixels are: the `pixels` seam's plane, its
/// `[clips, 4]` output grid, and each lane's clip run in it.
#[derive(Debug, Clone)]
pub(crate) struct PixelsReadback {
    /// `[voxel rows·k, C]` on the voxel axis.
    pub(crate) plane: kernels_cuda::Tensor,
    /// `[clips, 4]` i32 at the OUTPUT resolution.
    pub(crate) grid: kernels_cuda::Tensor,
    /// Per submitted lane: `(clip_offset, clips)` in the fire's clip order.
    pub(crate) lane_clips: Vec<(u32, u32)>,
}

/// What a settled step answers; the readouts are empty until [`Shell::read_out`].
#[derive(Debug, Default)]
pub struct Settled {
    /// Each submitted lane's logits, the asked-for rows concatenated row-major.
    pub logits: Vec<Vec<f32>>,
    /// How many rows of [`Settled::logits`] each lane's entry holds.
    pub rows: Vec<u32>,
    /// Each submitted lane's captured attention mass, empty for a lane that asked for none.
    pub scores: Vec<Vec<LayerScores>>,
    /// Per submitted lane, which export seam its [`Settled::logits`] came off.
    pub seams: Vec<engine::fire::ReadoutSeam>,
    /// Each submitted lane's decoded pixels (D8): one `f32` row per output
    /// voxel of its clips in submission order, beside each clip's output
    /// box. Empty for a lane that submitted no clip.
    pub pixels: Vec<super::Pixels>,
    /// Where to read them from, or `None` for the arming pass.
    pub(super) readback: Option<Readback>,
}

/// Where an asynchronous step publishes that it is done.
pub struct Done {
    /// Which step of which frame this is.
    pub at: engine::StepDone,
    /// Where to say so.
    pub sink: engine::CompletionSink,
}

impl Shell {
    /// `settle`, plus somewhere to publish the completion: an event on the
    /// compute stream, a wait on the notify stream, a host callback behind it.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for an event or a launch the runtime refused; the
    /// slot is then released synchronously after a stream synchronize.
    pub fn settle_step<'a>(&mut self, enqueued: Enqueued<'a>, done: Option<Done>) -> Result<Settled>
    where
        Self: 'a,
    {
        let Enqueued {
            mut prepared,
            launches: _,
            readback,
        } = enqueued;

        // From here the slot belongs to the callback.
        let slot = prepared.slot.take();
        drop(prepared);

        let at = self.settlement.claim()?;
        let airborne = self.airborne.clone();
        airborne.enter();

        let ordered = self
            .settlement
            .event(at)
            .record(self.device.stream())
            .and_then(|()| self.settlement.event(at).wait(self.device.notify_stream()));
        if let Err(fault) = ordered {
            // Nothing was ordered, so nothing will call back: undo by hand.
            let _ = self.device.synchronize();
            airborne.abandon();
            self.settlement.recycler().give(at);
            drop(slot);
            return Err(fault);
        }

        // The usage counts, carried out on the notify stream; a refusal is dropped.
        if let Some(tier) = self.weights.experts() {
            let _ = tier.drain(self.device.notify_stream());
        }

        // Everything the callback touches is already a value.
        let recycler = self.settlement.recycler();
        let posted = self.device.host_fn(Box::new(move || {
            drop(slot);
            recycler.give(at);
            airborne.leave();
            if let Some(done) = done {
                (done.sink)(done.at, engine::StepOutcome::Committed);
            }
        }));
        if let Err(fault) = posted {
            let _ = self.device.synchronize();
            self.airborne.abandon();
            self.settlement.recycler().give(at);
            return Err(fault);
        }

        Ok(Settled {
            logits: Vec::new(),
            rows: Vec::new(),
            scores: Vec::new(),
            seams: readback.as_ref().map_or_else(Vec::new, |readback| {
                readback
                    .lanes
                    .iter()
                    .map(|lane| lane.map_or(engine::fire::ReadoutSeam::Pixels, |(seam, _)| seam))
                    .collect()
            }),
            pixels: Vec::new(),
            readback,
        })
    }

    /// The numbers door: wait for the compute stream, read each lane's last row.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for whatever the fire's work said, [`Fault::Unbound`]
    /// for a rectangle the carve did not place.
    pub fn read_out(&mut self, settled: &mut Settled) -> Result<()> {
        self.read_out_rows(settled, &[])
    }

    /// [`Shell::read_out`], told which rows of each lane's run to mirror, in
    /// the order asked; a lane past `want`'s end reads [`Readout::Last`].
    ///
    /// # Errors
    ///
    /// As [`Shell::read_out`], plus [`Fault::Ceiling`] for a stated row past
    /// the rows its lane owns.
    pub fn read_out_rows(&mut self, settled: &mut Settled, want: &[Readout]) -> Result<()> {
        // The wait is unconditional and the read is not.
        self.device.synchronize()?;
        let Some(readback) = settled.readback.as_ref() else {
            return Ok(());
        };

        let lanes = readback.last_row.len();
        let mut taken = vec![Vec::new(); lanes];
        let mut counts = vec![0u32; lanes];
        let mut raw = Vec::new();
        for lane in 0..lanes {
            // This lane's own seam and rectangle; the element it is stored
            // in is bf16 for logits, bf16 or f32 for a float seam.
            let logits = readback.lanes[lane].map_or_else(
                || kernels_cuda::Tensor::new(0, 0, 0, model_ir::Dtype::Bf16),
                |(_, plane)| plane,
            );
            let width = logits.width as usize;
            let element = if logits.dtype == model_ir::Dtype::F32 {
                4
            } else {
                2
            };
            raw.clear();
            raw.resize(width * element, 0);
            let owned = readback.lane_rows[lane];
            // A plan with no `out` seam (a VAE decode) has no logits to read.
            if owned == 0 || logits.width == 0 {
                continue;
            }
            let chosen: Vec<u32> = match want.get(lane) {
                None | Some(Readout::Last) => vec![readback.last_row[lane]],
                Some(Readout::None) => Vec::new(),
                // The head ran over the gathered readout rectangle, and a
                // lane's run of it holds the rows it stated, in the order it
                // stated them — so a caller asking for its own list again
                // reads position by position, not by fire row.
                Some(Readout::Rows(rows)) => {
                    if rows.len() as u32 > owned {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: rows.len() as u64,
                            have: u64::from(owned),
                        });
                    }
                    (0..rows.len() as u32)
                        .map(|i| readback.first_row[lane] + i)
                        .collect()
                }
            };
            let mut values = Vec::with_capacity(chosen.len() * width);
            for row in &chosen {
                self.arena.read(
                    logits.ptr + u64::from(*row) * width as u64 * element as u64,
                    &mut raw,
                )?;
                if element == 4 {
                    values.extend(
                        raw.chunks_exact(4)
                            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]])),
                    );
                } else {
                    values.extend(
                        raw.chunks_exact(2)
                            .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]]))),
                    );
                }
            }
            counts[lane] = u32::try_from(chosen.len()).unwrap_or(u32::MAX);
            taken[lane] = values;
        }

        // The capture columns: one `[fire rows, heads]` F32 rectangle per exported layer.
        let mut scores: Vec<Vec<LayerScores>> = vec![Vec::new(); lanes];
        if !readback.columns.is_empty() {
            let mut mass: Vec<u8> = Vec::new();
            for lane in 0..lanes {
                if !readback.captures[lane] {
                    continue;
                }
                let rows = readback.lane_rows[lane];
                let first = readback.first_row[lane];
                let mut layers = Vec::with_capacity(readback.columns.len());
                for (layer, column) in &readback.columns {
                    let heads = column.width;
                    let bytes = rows as usize * heads as usize * 4;
                    mass.clear();
                    mass.resize(bytes, 0);
                    self.arena.read(
                        column.ptr + u64::from(first) * u64::from(heads) * 4,
                        &mut mass,
                    )?;
                    layers.push(LayerScores {
                        layer: *layer,
                        rows,
                        heads,
                        lse: mass
                            .chunks_exact(4)
                            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                            .collect(),
                    });
                }
                scores[lane] = layers;
            }
        }

        // The pixels (D8): the output grid comes back first, then each
        // lane's clips' rows out of the plane in clip order.
        let mut pixels: Vec<super::Pixels> = vec![(Vec::new(), Vec::new()); lanes];
        if let Some(seat) = readback.pixels.as_ref() {
            let clips = seat.grid.rows as usize;
            let mut grid = vec![0u8; clips * 16];
            self.arena.read(seat.grid.ptr, &mut grid)?;
            let grid: Vec<i32> = grid
                .chunks_exact(4)
                .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            let channels = seat.plane.width as usize;
            let element = model_compiler::arena::elem_bytes(seat.plane.dtype).unwrap_or(0) as usize;
            let mut raw: Vec<u8> = Vec::new();
            for (&(first, count), answer) in seat.lane_clips.iter().zip(pixels.iter_mut()) {
                let mut values = Vec::new();
                let mut boxes = Vec::with_capacity(count as usize);
                for clip in first..first + count {
                    let at = clip as usize * 4;
                    let [t, h, w, off] = [grid[at], grid[at + 1], grid[at + 2], grid[at + 3]];
                    let voxels = t as usize * h as usize * w as usize;
                    boxes.push([t as u32, h as u32, w as u32]);
                    let bytes = voxels * channels * element;
                    raw.clear();
                    raw.resize(bytes, 0);
                    self.arena.read(
                        seat.plane.ptr + off as u64 * channels as u64 * element as u64,
                        &mut raw,
                    )?;
                    match seat.plane.dtype {
                        model_ir::Dtype::F32 => values.extend(
                            raw.chunks_exact(4)
                                .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]])),
                        ),
                        _ => values.extend(
                            raw.chunks_exact(2)
                                .map(|pair| bf16(u16::from_le_bytes([pair[0], pair[1]]))),
                        ),
                    }
                }
                *answer = (values, boxes);
            }
        }

        settled.logits = taken;
        settled.rows = counts;
        settled.scores = scores;
        settled.pixels = pixels;
        Ok(())
    }
}

/// One bf16, widened: the top sixteen bits of an f32.
fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
