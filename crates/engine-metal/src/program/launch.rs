//! One stage, encoded: the device rings, the buffers one fire binds, and
//! the dispatch that runs a region — carved into fused and grouped forms,
//! chosen per region since Metal's argument-slot ABI has a channel ceiling.

use std::mem::size_of;
use std::sync::Arc;

use eta_compiler::codegen::launch::{LaunchChannel, LaunchStagePlan};
use eta_exec::{
    Extents, LANE_ABI_VERSION, LaneChannelSlot, LaneHeader, LaneRecord, LaneShape, NO_TICKET,
    OpParams, OpRuntime, SCRATCH_ALIGN, Status, ValueDesc, describe, layout,
};
use eta_ir::Dtype;
use eta_ir::op::{IntrinsicId, tags};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

use super::compile::{Form, Region, StreamedStep};
use eta_compiler::codegen::metal::{StepKind, reduce_dispatch_levels};
use super::shared::SharedRing;

/// The buffer index the first channel's committed cell binds at, below
/// status/descriptors/params/offsets/scratch/temporary/logits.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const FIRST_CHANNEL_BUFFER: usize = 7;

/// Width, in bytes, of one element of `lane.logits_base`'s rectangle.
/// Always 2 (bf16); the score plane (F32) is separate. See [`Prepared::regroup`].
const INTRINSIC_ELEMENT_BYTES: u64 = 2;

/// Threads a grouped LIBRARY sampler's threadgroup must have — the
/// nucleus/top-k kernels require exactly 256 and refuse any other width.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(super) const LIBRARY_SAMPLER_THREADS: usize = 256;

/// Threads a grouped fused region's threadgroup gets, at most — the
/// argmax reduction buffer is sized to this and faults `0xB3` above it.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(super) const REGION_THREADS: u32 =
    eta_compiler::codegen::metal::fused::METAL_M3_REGION_THREADS;

/// One reservation's address at `offset`. Errors ([`Fault::Program`])
/// off Apple and for an offset outside the reservation.
fn address_of(buffer: &Buffer, offset: u64) -> Result<u64> {
    buffer.address_at(offset).ok_or_else(|| {
        Fault::program(
            "program::launch",
            format!(
                "offset {offset} of a {}-byte reservation has no GPU address, and the \
                 grouped form binds addresses rather than buffers",
                buffer.bytes()
            ),
        )
    })
}

/// What a ring cell's offset is rounded up to: the widest alignment any
/// scalar it holds asks for. See [`ChannelShape::cell_stride`].
const CELL_ALIGN: usize = 16;

/// One channel's ring geometry, as the launch package declares it. Unlike
/// the CUDA twin, Metal packs bools on the device, so there is one width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelShape {
    /// How many cells the ring holds, not counting the spare.
    pub capacity: u32,
    /// Lanes in one cell.
    pub numel: usize,
    /// The cell's element type.
    pub dtype: Dtype,
}

impl ChannelShape {
    /// The shape one declared channel states.
    #[must_use]
    pub fn of(declared: &LaunchChannel) -> ChannelShape {
        ChannelShape {
            capacity: declared.capacity.max(1),
            numel: declared
                .shape
                .iter()
                .map(|&dim| dim as usize)
                .product::<usize>()
                .max(1),
            dtype: eta_exec::concrete_dtype(declared.dtype),
        }
    }

    /// Bytes in one cell — the only cell this plane has.
    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        eta_exec::wire_cell_bytes(self.dtype, self.numel)
    }

    /// Bytes from one cell to the next. Rounded up to [`CELL_ALIGN`] (a
    /// bound offset must be aligned) — may exceed [`ChannelShape::cell_bytes`].
    #[must_use]
    pub fn cell_stride(&self) -> usize {
        self.cell_bytes().next_multiple_of(CELL_ALIGN).max(CELL_ALIGN)
    }
}

/// The device rings, one buffer per channel — not one slab like the CUDA
/// twin, so an overrun is a bounds check rather than a cross-channel read.
#[derive(Debug)]
pub struct Rings {
    slabs: Vec<Buffer>,
    shapes: Vec<ChannelShape>,
    /// The channels whose ring is not this instance's: `Some` for a
    /// device-only channel two passes share (the slab is a clone).
    shared: Vec<Option<Arc<SharedRing>>>,
}

impl Rings {
    /// Reserve one ring per shape: `capacity + 1` cells (the spare makes
    /// full/empty distinguishable). `adopted` rings are kept as-is; errors: bad shape or geometry.
    pub fn allocate(
        device: &Context,
        shapes: &[ChannelShape],
        adopted: &[Option<Arc<SharedRing>>],
    ) -> Result<Rings> {
        let mut slabs = Vec::with_capacity(shapes.len());
        let mut shared = Vec::with_capacity(shapes.len());
        for (channel, shape) in shapes.iter().enumerate() {
            if let Some(ring) = adopted.get(channel).and_then(Option::as_ref) {
                // The two declarations must agree, or one ring is addressed at two strides.
                if ring.shape() != *shape {
                    return Err(Fault::program(
                        "program::launch",
                        format!(
                            "channel {channel}'s shared ring was cut for {} cell(s) of \
                             {:?} at capacity {} and this instance declares {} of {:?} \
                             at capacity {}: one ring addressed at two strides is a \
                             wrong cell and never a fault",
                            ring.shape().numel,
                            ring.shape().dtype,
                            ring.shape().capacity,
                            shape.numel,
                            shape.dtype,
                            shape.capacity
                        ),
                    ));
                }
                slabs.push(ring.slab());
                shared.push(Some(Arc::clone(ring)));
                continue;
            }
            let cells = u64::from(shape.capacity) + 1;
            let bytes = cells
                .checked_mul(shape.cell_stride() as u64)
                .ok_or_else(|| Fault::program("program::launch", "a ring past what a u64 counts"))?;
            slabs.push(Buffer::zeroed(device, bytes.max(1))?);
            shared.push(None);
        }
        Ok(Rings {
            slabs,
            shapes: shapes.to_vec(),
            shared,
        })
    }

    /// The ring `channel` shares with other instances, or `None` if owned alone.
    #[must_use]
    pub fn shared(&self, channel: usize) -> Option<&Arc<SharedRing>> {
        self.shared.get(channel).and_then(Option::as_ref)
    }

    /// How many channels this instance carries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slabs.len()
    }

    /// Whether it carries none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slabs.is_empty()
    }

    /// One channel's shape.
    #[must_use]
    pub fn shape(&self, channel: usize) -> Option<ChannelShape> {
        self.shapes.get(channel).copied()
    }

    /// Where one cell begins inside its ring, in bytes: `sequence %
    /// (capacity + 1)` is the slot. Errors for an unknown channel.
    pub fn cell_offset(&self, channel: usize, sequence: u64) -> Result<u64> {
        let shape = self.shape(channel).ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )
        })?;
        let cells = u64::from(shape.capacity) + 1;
        Ok((sequence % cells) * shape.cell_stride() as u64)
    }

    /// The buffer one channel's ring lives in.
    pub(crate) fn slab(&self, channel: usize) -> Result<&Buffer> {
        self.slabs.get(channel).ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )
        })
    }

    /// Write one cell's wire bytes; errors for an unknown channel or a mis-sized payload.
    pub fn write_cell(&mut self, channel: usize, sequence: u64, bytes: &[u8]) -> Result<()> {
        let at = self.cell_offset(channel, sequence)?;
        let width = self
            .shape(channel)
            .map_or(0, |shape| shape.cell_bytes());
        if bytes.len() != width {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "channel {channel}'s cell is {width} bytes and this write carries {}",
                    bytes.len()
                ),
            ));
        }
        self.slabs[channel].write(at, bytes)
    }

    /// Read one cell's wire bytes; errors for a channel this instance doesn't carry.
    pub fn read_cell(&self, channel: usize, sequence: u64) -> Result<Vec<u8>> {
        let at = self.cell_offset(channel, sequence)?;
        let width = self
            .shape(channel)
            .map_or(0, |shape| shape.cell_bytes());
        let mut cell = vec![0u8; width];
        self.slabs[channel].read(at, &mut cell)?;
        Ok(cell)
    }
}

/// Where one channel stands this fire: committed front, pending back (sequence numbers).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    /// The cell a take reads.
    pub head: u64,
    /// The cell a put writes.
    pub tail: u64,
}

/// Rows the per-intrinsic tables carry: `IntrinsicId::SLOTS`, one past the
/// largest id — an overflowed id reads the next slot rather than faulting.
const INTRINSIC_SLOTS: usize = IntrinsicId::SLOTS as usize;

/// One intrinsic's rectangle, as this plane binds one: base/stride/offset
/// ARE the Metal binding; `width` is kept for the grouped form's stride.
#[derive(Debug, Clone)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Slot {
    /// The allocation the rectangle lives in, retained for the binding's life.
    base: Buffer,
    /// Row start in bytes — the CUDA side's `row_offset`, pre-multiplied.
    offset: u64,
    /// Row width in elements: the fused bounds check, the grouped stride (`GroupLayout::vocab`).
    width: u32,
}

/// What a stage's own ops say about an intrinsic's rectangle — the
/// reader's claim, argued with at bind rather than assumed.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Declared {
    /// Row width the reader's output resolved to — a CEILING, not an equality.
    width: u32,
    /// Rows of that width the reader gathers. Grouped walks by pitch;
    /// single-lane walks consecutively and is only right for one row.
    rows: u32,
    /// Most elements any one reader gathers (element count, not bytes).
    elements: u64,
}

/// One channel's two bound cells, resolved for this fire.
#[derive(Debug, Clone)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Bound {
    /// The ring the two cells live in, retained for the binding's life.
    slab: Buffer,
    committed: u64,
    pending: u64,
}

// The grouped seat

/// `M3GroupLayout` — the grouped kernel's scalar arguments, in one record.
/// A second spelling of emitted text, tied to it by
/// [`tests::the_group_layout_matches_the_emitted_struct`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct GroupLayout {
    /// Lanes this launch covers; a threadgroup past it returns.
    lane_count: u32,
    /// The per-lane stride of `all_descriptors`.
    value_count: u32,
    /// The per-lane stride of `all_scratch`.
    scratch_stride: u32,
    /// Where the temporary begins inside one lane's scratch.
    temporary_offset: u32,
    /// Row width of `lane.logits_base`'s rectangle — the grouped gather's stride.
    vocab: u32,
    /// The per-lane stride of `channel_bindings`.
    reserved0: u32,
    /// Rows per lane the library samplers grid by: `dispatch_lane =
    /// threadgroup / reserved1`. Zero makes those kernels return without running.
    reserved1: u32,
    /// The per-lane stride of `params`.
    reserved2: u32,
}

/// `M3RowMeta` — where one lane's rows live in `row_indices`, pinned like [`GroupLayout`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct RowMeta {
    /// This lane's first entry in `row_indices`.
    offset: u32,
    /// How many entries are the lane's.
    count: u32,
    /// Where DRAFT rows begin, from `offset`: trunk is
    /// `[offset, offset + mtp_offset)`, draft is the rest.
    mtp_offset: u32,
    /// Padding. Zero.
    reserved: u32,
}

/// Everything the grouped form binds that single-lane does not. Built at
/// `lane_count = 1`: channels move out of argument slots into a threadgroup.
#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct Grouped {
    /// The lane table: `LaneHeader`, per-lane `LaneRecord`s, then the flat
    /// `LaneChannelSlot` array. Bound whole at buffer 0.
    table: Buffer,
    /// The table's geometry, so an offset is asked for rather than computed.
    shape: LaneShape,
    /// One [`GroupLayout`] per region, identical but for `reserved1` — a
    /// shared record would race a dispatch.
    layouts: Vec<Buffer>,
    /// Stage-local channel slot → dense channel index, indexing the lane's
    /// slot window (a constant on the CUDA twin; a table here).
    bindings: Buffer,
    /// One byte per (lane, dense channel): already put this fire? How
    /// grouped keeps `current_k` across a threadgroup. Zeroed every fire.
    pending_flags: Buffer,
    /// Dispatch lane → lane-record index; identity here (the emitted kernel's indirection).
    lane_indices: Buffer,
    /// One [`RowMeta`] per lane.
    row_meta: Buffer,
    /// The rows of `lane.logits_base` this fire reads, trunk block first.
    row_indices: Buffer,
    /// Rows in the trunk block, which is also where the draft block begins.
    trunk_rows: u32,
    /// Rows in the draft block. Zero for a stage that reads no draft column.
    draft_rows: u32,
    /// The [`GroupLayout`] each `layouts` entry holds, kept so a word can
    /// change without reading a reservation back.
    layout_words: Vec<GroupLayout>,
    /// The one lane's record, kept the same way: changed here, then the
    /// whole record is written back rather than patched at an offset.
    record: LaneRecord,
    /// Where the draft rows begin, as `set_rows` was last told; a [`Batch`]
    /// re-lays the lane's rows from it.
    draft_base: u32,
}

/// Lanes one grouped launch of a single instance covers. One: an instance's
/// own tables seat itself. A frame that carries several instances of one
/// program launches them together through a [`Batch`] instead.
const GROUPED_LANES: u32 = 1;

/// The lane a single-instance grouped launch is.
const THE_LANE: u32 = 0;

impl Grouped {
    /// Carve and fill everything the grouped form binds that doesn't change
    /// per fire. `None` if the planner says grouped can't cover this stage.
    fn build(
        device: &Context,
        plan: &LaunchStagePlan,
        descriptors: &[ValueDesc],
        shape: LaneShape,
        status: &Buffer,
        extents: Extents,
        trunk_rows: u32,
        draft_rows: u32,
    ) -> Result<Option<Grouped>> {
        if !plan.needs.grouped_valid {
            return Ok(None);
        }
        let Some(bytes) = shape.bytes() else {
            return Err(Fault::program(
                "program::launch",
                "a lane table whose size does not fit a u64: the channel count and the \
                 lane count multiply past what an allocation can be",
            ));
        };
        let mut table = Buffer::zeroed(device, bytes)?;
        table.write(
            0,
            &record_bytes(&LaneHeader {
                abi_version: LANE_ABI_VERSION,
                lane_count: shape.lanes,
                channel_slots_per_lane: shape.channel_slots_per_lane,
                flags: 0,
            }),
        )?;
        // The commit slot is the status word's address: dereferenced first, so zero would null-deref.
        let commit_slot = status.address_at(0).ok_or_else(|| {
            Fault::program(
                "program::launch",
                "the status word has no GPU address, so a grouped kernel would have \
                 nothing to write its verdict through",
            )
        })?;
        let record = LaneRecord {
            commit_slot,
            channel_slot_offset: shape.slot_index(THE_LANE).unwrap_or(0),
            kv_len: extents.kv_len,
            page_count: extents.page_count,
            row_count: extents.row_count,
            token_count: extents.token_count,
            sampled_rows: extents.sampled_rows,
            query_len: extents.query_len,
            key_len: extents.key_len,
            ..LaneRecord::default()
        };
        let at = shape.record_offset(THE_LANE).ok_or_else(|| {
            Fault::program("program::launch", "the one lane is outside the lane table")
        })?;
        table.write(at, &record_bytes(&record))?;

        // The per-lane stride of `channel_bindings`.
        let stride = u32::try_from(plan.channel_bindings.len()).map_err(|_| {
            Fault::program("program::launch", "more channels than a u32 can count")
        })?;
        let binding_bytes: Vec<u8> = (0..GROUPED_LANES)
            .flat_map(|_| plan.channel_bindings.iter().copied())
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut bindings = Buffer::zeroed(device, binding_bytes.len().max(4) as u64)?;
        bindings.write(0, &binding_bytes)?;

        let flags = u64::from(GROUPED_LANES) * u64::from(shape.channel_slots_per_lane);
        let pending_flags = Buffer::zeroed(device, flags.max(1))?;

        let index_bytes: Vec<u8> = (0..GROUPED_LANES).flat_map(u32::to_le_bytes).collect();
        let mut lane_indices = Buffer::zeroed(device, index_bytes.len() as u64)?;
        lane_indices.write(0, &index_bytes)?;

        let row_meta = Buffer::zeroed(
            device,
            (GROUPED_LANES as u64) * size_of::<RowMeta>() as u64,
        )?;
        let rows = u64::from(trunk_rows) + u64::from(draft_rows);
        let row_indices = Buffer::zeroed(device, rows.max(1) * size_of::<u32>() as u64)?;

        // One layout per region; `reserved1` is an upper bound on rows, not an exact count.
        let scratch_layout = layout(descriptors).map_err(|why| {
            Fault::program(
                "program::launch",
                format!("this fire's scratch does not fit: {why:?}"),
            )
        })?;
        let base = GroupLayout {
            lane_count: GROUPED_LANES,
            value_count: u32::try_from(plan.value_types.len()).unwrap_or(u32::MAX),
            scratch_stride: u32::try_from(scratch_layout.total).unwrap_or(u32::MAX),
            temporary_offset: u32::try_from(scratch_layout.temporary).unwrap_or(u32::MAX),
            vocab: 0,
            reserved0: stride,
            reserved1: 1,
            reserved2: u32::try_from(plan.ops.len()).unwrap_or(u32::MAX),
        };
        let mut layout_words = Vec::with_capacity(plan.fused.len());
        let mut layouts = Vec::with_capacity(plan.fused.len());
        for region in &plan.fused {
            let rows = region
                .inputs
                .iter()
                .filter_map(|&value| descriptors.get(value as usize))
                .map(|desc| desc.rows)
                .max()
                .unwrap_or(1)
                .max(1);
            let words = GroupLayout {
                reserved1: rows,
                ..base
            };
            let mut buffer = Buffer::zeroed(device, size_of::<GroupLayout>() as u64)?;
            buffer.write(0, &record_bytes(&words))?;
            layout_words.push(words);
            layouts.push(buffer);
        }

        Ok(Some(Grouped {
            table,
            shape,
            layouts,
            bindings,
            pending_flags,
            lane_indices,
            row_meta,
            row_indices,
            trunk_rows,
            draft_rows,
            layout_words,
            record,
            draft_base: 0,
        }))
    }

    /// Point every region's layout at a rectangle `vocab` elements wide.
    /// The grouped gather indexes `row_indices[…] * vocab`.
    fn set_vocab(&mut self, vocab: u32) -> Result<()> {
        for (words, buffer) in self.layout_words.iter_mut().zip(self.layouts.iter_mut()) {
            words.vocab = vocab;
            buffer.write(0, &record_bytes(words))?;
        }
        Ok(())
    }

    /// Write the one lane's `RowMeta` and the rows it names: trunk rows
    /// first, draft rows after `draft_base`; `mtp_offset` marks the split.
    fn set_rows(&mut self, draft_base: u32) -> Result<()> {
        self.draft_base = draft_base;
        let bytes: Vec<u8> = (0..self.trunk_rows)
            .chain((0..self.draft_rows).map(|row| draft_base.saturating_add(row)))
            .flat_map(u32::to_le_bytes)
            .collect();
        if !bytes.is_empty() {
            self.row_indices.write(0, &bytes)?;
        }
        let meta = RowMeta {
            offset: 0,
            count: self.trunk_rows.saturating_add(self.draft_rows),
            mtp_offset: self.trunk_rows,
            reserved: 0,
        };
        self.row_meta.write(0, &record_bytes(&meta))
    }

    /// Point the lane at the rectangle its readout lives in.
    fn set_logits(&mut self, base: u64, row_offset: u32, row_count: u32) -> Result<()> {
        self.record.logits_base = base;
        self.record.logits_row_offset = row_offset;
        self.record.logits_row_count = row_count;
        self.write_record()
    }

    /// Point the lane at its block of the observability slab — a separate
    /// reservation from the readout. Zero for none bound; the gather faults, not derefs.
    fn set_scores(&mut self, base: u64, row_stride: u32) -> Result<()> {
        self.record.attn_score_base = base;
        self.record.attn_score_row_stride = row_stride;
        self.write_record()
    }

    /// Point the lane at its run of the draft head's token plane — its own
    /// reservation, pitched by the head's depth. Zero for none bound; the
    /// emitted gather faults rather than dereferencing it.
    fn set_drafts(&mut self, base: u64, depth: u32) -> Result<()> {
        self.record.mtp_drafts_base = base;
        self.record.mtp_drafts_depth = depth;
        self.write_record()
    }

    /// Write the one lane's record back whole, rather than patched at a
    /// field offset. Errors if the lane is outside its own table.
    fn write_record(&mut self) -> Result<()> {
        let at = self.shape.record_offset(THE_LANE).ok_or_else(|| {
            Fault::program("program::launch", "the one lane is outside the lane table")
        })?;
        self.table.write(at, &record_bytes(&self.record))
    }

    /// Resolve this fire's channel cells into the lane's slot window, and
    /// clear last fire's put flags. Errors when a cell has no device address.
    fn refresh(&mut self, bindings: &[u32], bound: &[Bound]) -> Result<()> {
        let bytes = self.pending_flags.bytes();
        self.pending_flags.zero_span(0, bytes)?;
        for (&dense, cell) in bindings.iter().zip(bound) {
            let slot = LaneChannelSlot {
                committed_cell: address_of(&cell.slab, cell.committed)?,
                pending_cell: address_of(&cell.slab, cell.pending)?,
                expected_head: NO_TICKET,
                expected_tail: NO_TICKET,
            };
            let at = self.shape.slot_offset(THE_LANE, dense).ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!(
                        "channel {dense} is outside the lane table's slot window, which \
                         was carved for the channels this instance carries"
                    ),
                )
            })?;
            self.table.write(at, &record_bytes(&slot))?;
        }
        Ok(())
    }
}

/// Everything one stage binds for one fire.
#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Prepared {
    /// `M1Status`, 16 bytes at buffer 0; starts at `state = 1` every fire.
    status: Buffer,
    descriptors: Buffer,
    params: Buffer,
    offsets: Buffer,
    /// The values' scratch AND the temporary after it: one allocation,
    /// bound twice (`scratch` at 4, `temporary` at 5).
    scratch: Buffer,
    /// One rectangle per intrinsic id. `None` at the trunk's slot (6) binds
    /// nil; elsewhere `None` binds nothing (a nil there could clobber a channel).
    intrinsics: [Option<Slot>; INTRINSIC_SLOTS],
    /// What this stage's ops declared about each intrinsic they read,
    /// argued with at bind. `None` if unread, or unread as a gather output.
    declared: [Option<Declared>; INTRINSIC_SLOTS],
    channel_count: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    /// This stage's local channel slot → the instance's dense channel index.
    bindings: Vec<u32>,
    /// Resolved cells, in stage-local slot order; filled by [`Prepared::refresh`].
    bound: Vec<Bound>,
    /// The grouped form's tables, built whenever the stage could take that
    /// path; which regions do is [`super::compile`]'s call.
    grouped: Option<Grouped>,
    /// Which intrinsics each FUSED region reads, as a bitmask indexed by
    /// region — checked at encode, once [`Region::form`] is known.
    region_intrinsics: Vec<u64>,
    /// Intrinsics bound WIDER than readers' declared row, across >1 row —
    /// the shape needing a row stride the fused gather doesn't have.
    strided: u64,
    /// The bytes behind `descriptors`, `params` and `offsets`, kept so a
    /// [`Batch`] can lay this instance's copy at its dispatch lane.
    descriptor_bytes: Vec<u8>,
    param_bytes: Vec<u8>,
    offset_bytes: Vec<u8>,
    /// The resolved value descriptors, for sizing a streamed region's grids.
    descriptor_table: Vec<ValueDesc>,
}

/// `M4Step` — the streamed kernel's per-dispatch word, bound at 11 with
/// `setBytes`. Spelled here and in `eta_compiler::codegen::metal::streamed`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct StepWord {
    /// The op (`case`) this dispatch runs.
    index: u32,
    /// A reduction's level, or 0 / 1 for an argmax's partial / final pass.
    level: u32,
    /// How many groups the argmax's partial pass had, for its final pass.
    groups: u32,
    reserved: u32,
}

/// `scratch-no-zero`: skip the host memset of a fire's scratch, to measure
/// what the host's touch of those pages costs the device.
fn scratch_zeroing_skipped() -> bool {
    crate::diag::on().scratch_no_zero
}

/// The streamed form's measurement knobs, as this module reads them.
/// `streamed-groups=n` caps the blocks a wide dispatch spreads over;
/// `streamed-repeat=n` issues each wide dispatch n times (or each dispatch of
/// the kind `streamed-repeat-kind=wide|partial|single|reduce|argmax`) — all
/// are idempotent, so this prices a dispatch; `streamed-limit=k` runs only the
/// first k steps, which breaks the program and times what ran.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
struct StreamedKnobs {
    max_groups: u32,
    repeat: usize,
    repeat_kind: Option<StepKind>,
    limit: usize,
}

/// The boot's word list, in this module's own vocabulary. `StepKind` is
/// private here, so the mapping from the typed word lives here too.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn streamed_knobs() -> StreamedKnobs {
    let diag = crate::diag::on();
    StreamedKnobs {
        max_groups: diag.streamed_groups.unwrap_or(STREAMED_MAX_GROUPS),
        repeat: diag.streamed_repeat.max(1),
        repeat_kind: diag.streamed_repeat_kind.map(|kind| match kind {
            crate::diag::StreamedKind::Wide => StepKind::Wide,
            crate::diag::StreamedKind::Partial => StepKind::Partial,
            crate::diag::StreamedKind::Single => StepKind::Single,
            crate::diag::StreamedKind::Reduce => StepKind::Reduce,
            crate::diag::StreamedKind::Argmax => StepKind::Argmax,
        }),
        limit: diag.streamed_limit.unwrap_or(usize::MAX),
    }
}

/// Most element blocks one streamed dispatch spreads over. Every thread of
/// every block walks the kernel's preamble — a chain of dependent loads from
/// the lane table — before it touches an element, so a wide grid pays that
/// chain once per wave of resident threads: measured on the emitted kernel,
/// a 248k-element pass cost 55 µs at 1024 blocks and 22 µs at 64, with the
/// loop itself a third of the latter. Sixty-four blocks of a few hundred
/// threads fill the device once; each thread then strides a handful of
/// elements.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const STREAMED_MAX_GROUPS: u32 = 64;

/// Level-`l + 1` chunks one threadgroup of a reduce dispatch folds; the
/// runtime's `M4_REDUCE_CHUNKS_PER_GROUP`. A dispatch at level `l` covers
/// `32 × this` level-`l` chunks per group.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const REDUCE_CHUNKS_PER_GROUP: u32 = 4;

/// The threadgroup a streamed region is dispatched with: the pipeline's
/// widest, capped at the region ceiling, rounded down to a power of two of
/// at least 32 — the reductions fold across SIMD groups and divide the
/// group into rounds, so its width must divide evenly.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub(super) fn streamed_threads(max_total: usize) -> usize {
    let capped = max_total.clamp(1, REGION_THREADS as usize);
    let pow2 = 1usize << (usize::BITS - 1 - capped.leading_zeros());
    pow2.max(32)
}

/// The dispatches a streamed region's step table unfolds into for these
/// descriptors: `(word, groups along x)`. A `Wide` step is one dispatch over
/// its result's length; `Single` one group; `Reduce` one per tree level, each
/// over that level's chunks; `Argmax` a partial pass over the row and a final
/// pass over the partials.
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
fn streamed_dispatches(
    steps: &[StreamedStep],
    descriptors: &[ValueDesc],
    threads: u32,
    temporary_bytes: u64,
) -> Vec<(StepWord, u32)> {
    let threads = threads.max(1);
    let knobs = streamed_knobs();
    let groups_for = |work: u32| -> u32 { work.div_ceil(threads).clamp(1, knobs.max_groups) };
    let mut out = Vec::with_capacity(steps.len());
    let repeats = |kind: StepKind| -> usize {
        match knobs.repeat_kind {
            Some(k) if k == kind => knobs.repeat,
            Some(_) => 1,
            None if kind == StepKind::Wide => knobs.repeat,
            None => 1,
        }
    };
    for (position, step) in steps.iter().enumerate().take(knobs.limit) {
        // The kernel's `case` is the step's position in the table.
        let word = StepWord {
            index: u32::try_from(position).unwrap_or(u32::MAX),
            ..StepWord::default()
        };
        match step.kind {
            StepKind::Wide | StepKind::Partial => {
                let sized_by = if step.kind == StepKind::Partial {
                    step.input
                } else {
                    step.result
                };
                let len = descriptors
                    .get(sized_by as usize)
                    .map_or(1, |desc| desc.len);
                for _ in 0..repeats(StepKind::Wide) {
                    out.push((word, groups_for(len)));
                }
            }
            StepKind::Single => {
                for _ in 0..repeats(StepKind::Single) {
                    out.push((word, 1));
                }
            }
            StepKind::Reduce => {
                let desc = descriptors.get(step.input as usize).copied().unwrap_or_default();
                // A dispatch folds level `l` (and `l + 1` inside the
                // threadgroup); its groups cover level `l`'s chunks, one chunk
                // per thread, per row.
                let mut count = desc.last;
                let mut at = 0u32;
                // The runtime folds two levels per dispatch, a group owning
                // `REDUCE_CHUNKS_PER_GROUP` chunks of the upper one; rows are
                // walked inside the group.
                for level in reduce_dispatch_levels(desc.last) {
                    while at < level {
                        count = count.div_ceil(32);
                        at += 1;
                    }
                    let chunks = count.div_ceil(32).max(1);
                    let groups = chunks.div_ceil(32 * REDUCE_CHUNKS_PER_GROUP).max(1);
                    for _ in 0..repeats(StepKind::Reduce) {
                        out.push((
                            StepWord {
                                level,
                                ..word
                            },
                            groups,
                        ));
                    }
                }
            }
            StepKind::Argmax => {
                let desc = descriptors.get(step.input as usize).copied().unwrap_or_default();
                // One candidate per (row, group) lands in the temporary.
                let candidate_bytes = 16u64;
                let fit = (temporary_bytes / candidate_bytes / u64::from(desc.rows.max(1)))
                    .clamp(1, u64::from(STREAMED_MAX_GROUPS)) as u32;
                let groups = groups_for(desc.last).min(fit);
                for _ in 0..repeats(StepKind::Argmax) {
                    out.push((
                        StepWord {
                            level: 0,
                            groups,
                            ..word
                        },
                        groups,
                    ));
                    out.push((
                        StepWord {
                            level: 1,
                            groups,
                            ..word
                        },
                        1,
                    ));
                }
            }
        }
    }
    // `streamed-nop=n`: n dispatches of one group that hit the kernel's
    // `default: return` — the production floor of a dispatch of this kernel
    // with these bindings, with no op behind it.
    let nops = crate::diag::on().streamed_nop;
    for _ in 0..nops {
        out.push((
            StepWord {
                index: u32::MAX,
                ..StepWord::default()
            },
            1,
        ));
    }
    if crate::diag::on().streamed_trace {
        let grid: Vec<String> = out
            .iter()
            .map(|(word, groups)| format!("{}:{}x{groups}", word.index, word.level))
            .collect();
        let sizes: Vec<String> = steps
            .iter()
            .map(|step| {
                format!(
                    "{:?}[r{}={} i{}={}]",
                    step.kind,
                    step.result,
                    descriptors.get(step.result as usize).map_or(0, |d| d.len),
                    step.input,
                    descriptors.get(step.input as usize).map_or(0, |d| d.len)
                )
            })
            .collect();
        eprintln!(
            "streamed: {} step(s) -> {} dispatch(es), {threads} threads, repeat {} of {:?}; grid {}; sizes {}",
            steps.len(),
            out.len(),
            knobs.repeat,
            knobs.repeat_kind,
            grid.join(" "),
            sizes.join(" ")
        );
    }
    out
}

/// What two instances must agree on to share one [`Batch`]: the grouped
/// kernel strides its per-dispatch-lane tables by these, so a member whose
/// numbers differ would read another member's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BatchKey {
    /// A hash of the descriptor, param, offset and binding bytes: members
    /// that share it share the tables, so the batch lays them down once.
    content: u64,
    channel_count: u32,
    channel_slots_per_lane: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    trunk_rows: u32,
    draft_rows: u32,
    regions: usize,
}

impl Prepared {
    /// Carve every buffer one stage needs, for a single-lane fire. Errors
    /// on a bad value shape, scratch overflow, or an unbound put channel.
    pub fn build(
        device: &Context,
        plan: &LaunchStagePlan,
        shapes: &[ChannelShape],
        extents: Extents,
    ) -> Result<Prepared> {
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
        let value_count = u32::try_from(plan.value_types.len())
            .map_err(|_| Fault::program("program::launch", "more values than a u32 can count"))?;

        // The value descriptors, and the scratch they size.
        let descriptors: Vec<ValueDesc> = plan
            .value_types
            .iter()
            .map(|value| {
                describe(value, &extents).map_err(|why| {
                    Fault::program(
                        "program::launch",
                        format!("a value's shape does not resolve against this fire: {why:?}"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let scratch_layout = layout(&descriptors).map_err(|why| {
            Fault::program(
                "program::launch",
                format!("this fire's scratch does not fit: {why:?}"),
            )
        })?;
        let scratch_stride = u32::try_from(scratch_layout.total)
            .map_err(|_| Fault::program("program::launch", "a scratch stride past a u32"))?;
        let temporary_offset = u32::try_from(scratch_layout.temporary)
            .map_err(|_| Fault::program("program::launch", "a temporary offset past a u32"))?;

        // Op params: the shared record, uploaded as-is (`M1OpParams` field for field).
        let mut records = Vec::with_capacity(plan.ops.len());
        let mut declared: [Option<Declared>; INTRINSIC_SLOTS] = [None; INTRINSIC_SLOTS];
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record = OpParams::of(op, result_base, OpRuntime::default());
            if let (true, Some(channel)) = (op.tag == tags::CHAN_PUT, op.channel) {
                // `sink_bytes` IS the cell: the put faults if the value is wider.
                let dense = plan
                    .channel_bindings
                    .get(channel as usize)
                    .copied()
                    .ok_or_else(|| {
                        Fault::program(
                            "program::launch",
                            format!(
                                "a put names stage-local channel {channel}, which the plan \
                                 does not bind"
                            ),
                        )
                    })?;
                let shape = shapes.get(dense as usize).copied().ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "a put targets channel {dense}, which this instance does not carry"
                        ),
                    )
                })?;
                record.sink_bytes = u32::try_from(shape.cell_bytes()).map_err(|_| {
                    Fault::program("program::launch", "a channel cell past what a u32 counts")
                })?;
            }
            // What this stage's readers claim about each rectangle's geometry;
            // rank-1 outputs (e.g. `mtp_drafts`) claim nothing.
            if let Some(intrinsic) = op.intrinsic
                && let Some(seat) = declared.get_mut(intrinsic as usize)
                && let Some(out) = descriptors.get(record.o0 as usize)
                && out.rank >= 2
            {
                let claim = Declared {
                    width: out.last,
                    rows: out.rows.max(1),
                    elements: u64::from(out.len),
                };
                *seat = Some(match *seat {
                    Some(prior) => Declared {
                        width: prior.width.max(claim.width),
                        rows: prior.rows.max(claim.rows),
                        elements: prior.elements.max(claim.elements),
                    },
                    None => claim,
                });
            }
            records.push(record);
            result_base += u32::from(op.result_count);
        }

        let param_bytes = records_bytes(&records);
        let mut params = Buffer::zeroed(
            device,
            (records.len() * size_of::<OpParams>()).max(size_of::<OpParams>()) as u64,
        )?;
        params.write(0, &param_bytes)?;

        let descriptor_bytes: Vec<u8> = descriptors.iter().flat_map(record_bytes).collect();
        let mut descriptor_buffer =
            Buffer::zeroed(device, descriptor_bytes.len().max(1) as u64)?;
        descriptor_buffer.write(0, &descriptor_bytes)?;

        let offset_bytes: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut offsets =
            Buffer::zeroed(device, offset_bytes.len().max(size_of::<u32>()) as u64)?;
        offsets.write(0, &offset_bytes)?;

        let scratch_bytes = u64::from(scratch_stride).max(SCRATCH_ALIGN);
        let scratch = Buffer::zeroed(device, scratch_bytes)?;
        let status = Buffer::zeroed(device, eta_exec::STATUS_BYTES as u64)?;

        // The grouped tables, carved beside the argument-slot seats — a stage may mix both forms.
        let lanes = LaneShape::of(
            GROUPED_LANES,
            u32::try_from(shapes.len()).map_err(|_| {
                Fault::program("program::launch", "more channels than a u32 can count")
            })?,
        );
        // The two row blocks `row_indices` carries: trunk is what readers
        // declared, draft is the larger of that and `mtp_rows`. A reader
        // with no declare gets one row, not zero.
        let reads = |wanted: IntrinsicId| {
            plan.ops
                .iter()
                .any(|op| op.intrinsic.is_some_and(|id| id as usize == wanted as usize))
        };
        // A stage that reads the draft plane without reading `logits` (a
        // block drafter's guest asks the head what it proposes and nothing
        // else) still spans its readout rows: `drafts_len` ids at depth one
        // are that many rows, which is what the plane's guard
        // (`emit_mtp_drafts`) multiplies the depth by. A chained head's
        // guest also reads `logits`, so this never widens what it declared.
        let trunk_rows = declared[IntrinsicId::Logits as usize]
            .map_or(0, |it| it.rows)
            .max(u32::from(reads(IntrinsicId::Logits)))
            .max(plan.drafts_len);
        let draft_rows = declared[IntrinsicId::MtpLogits as usize]
            .map_or(0, |it| it.rows)
            .max(plan.mtp_rows)
            .max(u32::from(
                reads(IntrinsicId::MtpLogits) || reads(IntrinsicId::MtpDrafts),
            ));
        let grouped = Grouped::build(
            device,
            plan,
            &descriptors,
            lanes,
            &status,
            extents,
            trunk_rows,
            draft_rows,
        )?;

        // Which intrinsics each fused region reads; `compile::grouped_region` agrees by construction.
        let region_intrinsics = plan
            .fused
            .iter()
            .map(|region| {
                region.nodes.iter().fold(0u64, |mask, &node| {
                    match plan.ops.get(node as usize).and_then(|op| op.intrinsic) {
                        Some(id) if (id as u32) < u64::BITS => mask | (1u64 << (id as u32)),
                        _ => mask,
                    }
                })
            })
            .collect();

        Ok(Prepared {
            status,
            descriptors: descriptor_buffer,
            params,
            offsets,
            scratch,
            intrinsics: [const { None }; INTRINSIC_SLOTS],
            declared,
            channel_count,
            value_count,
            scratch_stride,
            temporary_offset,
            bindings: plan.channel_bindings.clone(),
            bound: Vec::new(),
            grouped,
            region_intrinsics,
            strided: 0,
            descriptor_bytes,
            param_bytes,
            offset_bytes,
            descriptor_table: descriptors,
        })
    }

    /// The numbers a [`Batch`] member must share; `None` when this stage has
    /// no grouped seat, so it cannot be batched at all.
    #[must_use]
    pub fn batch_key(&self) -> Option<BatchKey> {
        use std::hash::{Hash, Hasher};
        let grouped = self.grouped.as_ref()?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.descriptor_bytes.hash(&mut hasher);
        self.param_bytes.hash(&mut hasher);
        self.offset_bytes.hash(&mut hasher);
        self.bindings.hash(&mut hasher);
        Some(BatchKey {
            content: hasher.finish(),
            channel_count: self.channel_count,
            channel_slots_per_lane: grouped.shape.channel_slots_per_lane,
            scratch_stride: self.scratch_stride,
            temporary_offset: self.temporary_offset,
            trunk_rows: grouped.trunk_rows,
            draft_rows: grouped.draft_rows,
            regions: grouped.layouts.len(),
        })
    }

    /// Resolve this fire's cells and reset everything a fire starts from.
    /// Errors when a stage-local slot names an uncarried channel.
    pub fn refresh(&mut self, rings: &Rings, cursors: &[Cursor]) -> Result<()> {
        self.refresh_cells(rings, cursors)?;
        if scratch_zeroing_skipped() {
            return Ok(());
        }
        // Zeroed every fire: an unwritten slot would read back the last fire's leftovers.
        let bytes = self.scratch.bytes();
        self.scratch.zero_span(0, bytes)
    }

    /// The scratch half of [`Prepared::refresh`], for a stage that was
    /// staged through [`Prepared::refresh_cells`] but has no grouped seat and
    /// so runs on its own scratch after all.
    pub fn zero_scratch(&mut self) -> Result<()> {
        let bytes = self.scratch.bytes();
        self.scratch.zero_span(0, bytes)
    }

    /// [`Prepared::zero_scratch`] on the device: a blit in `frame`'s order,
    /// then a fresh compute pass. What a batch does for its pool
    /// (`Batch::encode`), for the instance that launches alone.
    #[cfg(target_vendor = "apple")]
    pub fn zero_scratch_on(&mut self, frame: &mut Frame) -> Result<()> {
        if scratch_zeroing_skipped() {
            return Ok(());
        }
        let bytes = self.scratch.bytes();
        frame.fill(self.scratch.slab(), 0, bytes)?;
        frame.next_pass()?;
        Ok(())
    }

    /// [`Prepared::refresh`] without the scratch: what a [`Batch`] member
    /// needs, since its values live in the batch's pool and not here.
    pub fn refresh_cells(&mut self, rings: &Rings, cursors: &[Cursor]) -> Result<()> {
        self.bound.clear();
        self.bound.reserve(self.bindings.len());
        for (local, &dense) in self.bindings.iter().enumerate() {
            let channel = dense as usize;
            let cursor = cursors.get(channel).copied().ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!(
                        "stage-local channel {local} binds channel {dense}, which this \
                         instance does not carry"
                    ),
                )
            })?;
            self.bound.push(Bound {
                slab: rings.slab(channel)?.clone(),
                committed: rings.cell_offset(channel, cursor.head)?,
                pending: rings.cell_offset(channel, cursor.tail)?,
            });
        }
        let ready = Status {
            state: 1,
            fault: 0,
            reserved0: 0,
            reserved1: 0,
        };
        self.status.write(0, &record_bytes(&ready))?;
        // The same cells, as addresses in the lane table — a mixed stage would else hand grouped a stale table.
        if let Some(grouped) = self.grouped.as_mut() {
            grouped.refresh(&self.bindings, &self.bound)?;
        }
        Ok(())
    }

    /// Point one intrinsic's slot at the rectangle a model fire produced.
    /// `width` is argued with: the declared width is a ceiling, not an
    /// equality. Errors for an unbindable, mismatched or unstrideable rectangle.
    pub fn bind_intrinsic(
        &mut self,
        intrinsic: IntrinsicId,
        base: &Buffer,
        offset: u64,
        width: u32,
        dtype: Dtype,
    ) -> Result<()> {
        if eta_compiler::codegen::metal::m2_intrinsic_buffer(intrinsic as u16).is_none() {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "{intrinsic:?} has no argument index in the M2 slot table, so a \
                     rectangle bound for it would be read as another intrinsic's"
                ),
            ));
        }
        let declared = *self
            .declared
            .get(intrinsic as usize)
            .ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!(
                        "{intrinsic:?} is past the pitch the slot table is indexed with"
                    ),
                )
            })?;
        // The element type is picked by intrinsic id, so this side can't disagree with the kernel.
        let element = eta_compiler::codegen::metal::m2_intrinsic_element_bytes(intrinsic as u16)
            .map(u64::from)
            .ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!("{intrinsic:?} has no element width in the M2 slot table"),
                )
            })?;
        // The token plane is the one integer rectangle: four bytes an id.
        let wanted = match intrinsic {
            IntrinsicId::MtpDrafts => Dtype::I32,
            _ if element == 4 => Dtype::F32,
            _ => Dtype::Bf16,
        };
        if dtype != wanted {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "a rectangle for {intrinsic:?} landed as {dtype:?}; the emitted \
                     `0xA0` handler reads it as {wanted:?} and has no other element type \
                     for this intrinsic"
                ),
            ));
        }
        if let Some(declared) = declared {
            // The declared width is a ceiling: only asking for MORE than it is refused.
            if declared.width > width {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "this stage reads {intrinsic:?} as rows of {} elements and the \
                         rectangle offered is only {width} wide; a declared extent is a \
                         ceiling on the row it is pointed at, so serving this one would \
                         read past the end of every row",
                        declared.width
                    ),
                ));
            }
            // The multi-row read is where the forms part: fused walks with no
            // stride (only one row lands right); grouped carries a row pitch.
            if declared.width < width && declared.rows > 1 && !self.strideable(intrinsic) {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "this stage reads {intrinsic:?} as {} rows of {} elements out of a \
                         rectangle {width} wide, and every form that can run it walks its \
                         rows consecutively: it has no row stride to be told, so every row \
                         after the first would land {} elements short. A narrower read \
                         of ONE row is served; this one needs the grouped form, which \
                         carries a row pitch the way the CUDA handler's \
                         `intrinsic_row_stride` does — and this stage has no grouped seat \
                         for it, either because the plan said that path cannot cover the \
                         stage or because the emitter binds no address for {intrinsic:?}",
                        declared.rows,
                        declared.width,
                        width - declared.width
                    ),
                ));
            }
            // The reader walks `out0.len` elements off the offset — the bound the kernel can't check itself.
            let reach = offset.saturating_add(declared.elements.saturating_mul(element));
            if reach > base.bytes() {
                return Err(Fault::Ceiling {
                    what: "bytes in the rectangle an intrinsic is pointed at",
                    need: reach,
                    have: base.bytes(),
                });
            }
        }
        // `strided` is recomputed, not or-ed, so a rebind clears it as readily as it sets it.
        let bit = 1u64 << (intrinsic as u32);
        if declared.is_some_and(|it| it.width < width && it.rows > 1) {
            self.strided |= bit;
        } else {
            self.strided &= !bit;
        }
        self.intrinsics[intrinsic as usize] = Some(Slot {
            base: base.clone(),
            offset,
            width,
        });
        self.regroup()
    }

    /// Whether a narrow multi-row read of `intrinsic` has a form that serves
    /// it: a grouped seat, and an id the grouped emitter will bind.
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    fn strideable(&self, intrinsic: IntrinsicId) -> bool {
        self.grouped.is_some()
            && eta_compiler::codegen::metal::m3_intrinsic_bindable(intrinsic as u16)
    }

    /// Refuse a SINGLE-LANE region that reads an intrinsic bound at a
    /// shape only a strided form can serve. An unplaceable region index is refused too.
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    fn no_stride_owed(&self, region: &Region) -> Result<()> {
        let reads = self
            .region_intrinsics
            .get(region.region_index as usize)
            .copied()
            .unwrap_or(u64::MAX);
        if reads & self.strided == 0 {
            return Ok(());
        }
        Err(Fault::program(
            "program::launch",
            format!(
                "region {} of this stage reads a rectangle wider than the rows its \
                 own readers declared, across more than one row, and it compiled to \
                 the SINGLE-LANE form, whose gather walks `out0.len` consecutive \
                 elements off the binding and has no row pitch to be told. The \
                 grouped form carries one; this region did not get it",
                region.region_index
            ),
        ))
    }

    /// Re-derive the grouped form's rectangle words from the slot table.
    /// `logits`/`mtp_logits` share `lane.logits_base`; `attn_score` has its own base.
    fn regroup(&mut self) -> Result<()> {
        if self.grouped.is_none() {
            return Ok(());
        }
        // The score rectangle is a separate reservation, derived first so a scores-only read still gets it.
        let (score_base, score_stride) =
            match self.intrinsics[IntrinsicId::AttnScore as usize].as_ref() {
                Some(slot) => (address_of(&slot.base, slot.offset)?, slot.width),
                None => (0, 0),
            };
        self.grouped
            .as_mut()
            .expect("the seat was there one statement ago")
            .set_scores(score_base, score_stride)?;
        // The token plane likewise: its own base and pitch (the depth).
        let (drafts_base, drafts_depth) =
            match self.intrinsics[IntrinsicId::MtpDrafts as usize].as_ref() {
                Some(slot) => (address_of(&slot.base, slot.offset)?, slot.width),
                None => (0, 0),
            };
        self.grouped
            .as_mut()
            .expect("the seat was there one statement ago")
            .set_drafts(drafts_base, drafts_depth)?;
        let Some(trunk) = self.intrinsics[IntrinsicId::Logits as usize].as_ref() else {
            return Ok(());
        };
        // Stride is the rectangle's own row width — may express a narrow read of more than one row.
        let width = trunk.width;
        let logits_base = address_of(&trunk.base, trunk.offset)?;
        // Where the draft block starts; zero unless it's the same reservation a whole row-count apart.
        let stride = u64::from(trunk.width) * INTRINSIC_ELEMENT_BYTES;
        let draft_base = self.intrinsics[IntrinsicId::MtpLogits as usize]
            .as_ref()
            .filter(|drafts| drafts.width == trunk.width && stride != 0)
            .and_then(|drafts| {
                let there = address_of(&drafts.base, drafts.offset).ok()?;
                let apart = there.checked_sub(logits_base)?;
                (apart % stride == 0)
                    .then(|| u32::try_from(apart / stride).ok())
                    .flatten()
            })
            .unwrap_or(0);
        let grouped = self
            .grouped
            .as_mut()
            .expect("the seat was there one statement ago");
        let rows = grouped.trunk_rows.saturating_add(grouped.draft_rows);
        grouped.set_vocab(width)?;
        grouped.set_logits(logits_base, 0, rows)?;
        grouped.set_rows(draft_base)
    }

    /// Encode one generated region into a pass someone else opened, and do
    /// not commit — safe to call inside `serve::enqueue`, unlike [`Prepared::launch_region`].
    pub fn encode_into(&self, frame: &Frame, region: &Region) -> Result<()> {
        match region.form {
            Form::Fused => self.encode_fused(frame, region, &self.scratch, 0),
            Form::Streamed => self.encode_streamed(frame, region),
            Form::Grouped | Form::GroupedLibrary => self.encode_grouped(frame, region),
        }
    }

    /// Encode one streamed region for this instance alone: the grouped
    /// bindings plus the step word, one dispatch per entry of the region's
    /// table, a grid of `groups × 1`.
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_streamed(&self, frame: &Frame, region: &Region) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::ProtocolObject;
            use objc2_metal::{
                MTLComputeCommandEncoder, MTLComputePipelineState, MTLResource, MTLResourceUsage,
            };
            let grouped = self.grouped.as_ref().ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    "a region was compiled for the streamed form and this stage carries no \
                     lane table; the plan said the grouped path could not cover it",
                )
            })?;
            let layout = grouped
                .layouts
                .get(region.region_index as usize)
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!("region {} has no group layout", region.region_index),
                    )
                })?;
            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            // SAFETY: every reservation is retained by `self`; every offset is zero.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(grouped.table.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(self.scratch.raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(layout.raw()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(grouped.bindings.raw()), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(grouped.pending_flags.raw()), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(grouped.lane_indices.raw()), 0, 8);
                encoder.setBuffer_offset_atIndex(Some(grouped.row_meta.raw()), 0, 9);
                encoder.setBuffer_offset_atIndex(Some(grouped.row_indices.raw()), 0, 10);
            }
            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            resident(
                &self.status,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            for cell in &self.bound {
                resident(
                    &cell.slab,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
            for held in self.intrinsics.iter().flatten() {
                resident(&held.base, MTLResourceUsage::Read);
            }
            let threads = streamed_threads(region.pipeline().maxTotalThreadsPerThreadgroup());
            let temporary_bytes = u64::from(self.scratch_stride.saturating_sub(self.temporary_offset));
            dispatch_streamed(
                encoder,
                &streamed_dispatches(
                    &region.steps,
                    &self.descriptor_table,
                    u32::try_from(threads).unwrap_or(1),
                    temporary_bytes,
                ),
                threads,
                GROUPED_LANES as usize,
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    /// Encode one single-lane region with its values at `scratch_at` inside
    /// `scratch` — this instance's own scratch at zero, or its dispatch
    /// lane's slot of a [`Batch`] pool when the stage's grouped regions ran
    /// there, so both forms read and write the same values.
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_fused(
        &self,
        frame: &Frame,
        region: &Region,
        scratch: &Buffer,
        scratch_at: u64,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

            self.no_stride_owed(region)?;
            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            let scratch_at = usize::try_from(scratch_at).unwrap_or(usize::MAX);
            // SAFETY: every buffer is retained by `self`/`region`/the batch; every offset was bounds-checked.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.status.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(scratch.raw()), scratch_at, 4);
                encoder.setBuffer_offset_atIndex(
                    Some(scratch.raw()),
                    scratch_at + self.temporary_offset as usize,
                    5,
                );
                // The slot table, one `setBuffer` per bound rectangle. The trunk's
                // index is always written (nil OK); every other unbound slot is left unwritten.
                for (slot, held) in self.intrinsics.iter().enumerate() {
                    let Some(at) = u16::try_from(slot)
                        .ok()
                        .and_then(eta_compiler::codegen::metal::m2_intrinsic_buffer)
                    else {
                        continue;
                    };
                    match held {
                        Some(bound) => encoder.setBuffer_offset_atIndex(
                            Some(bound.base.raw()),
                            usize::try_from(bound.offset).unwrap_or(0),
                            at,
                        ),
                        None if at == eta_compiler::codegen::metal::M2_LOGITS_BUFFER => {
                            encoder.setBuffer_offset_atIndex(None, 0, at);
                        }
                        None => {}
                    }
                }
                for (local, bound) in self.bound.iter().enumerate() {
                    let at = FIRST_CHANNEL_BUFFER + local * 2;
                    encoder.setBuffer_offset_atIndex(
                        Some(bound.slab.raw()),
                        usize::try_from(bound.committed).unwrap_or(0),
                        at,
                    );
                    encoder.setBuffer_offset_atIndex(
                        Some(bound.slab.raw()),
                        usize::try_from(bound.pending).unwrap_or(0),
                        at + 1,
                    );
                }
            }
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreads_threadsPerThreadgroup(one, one);
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    /// Encode one grouped region: eleven fixed bindings, a residency
    /// declaration per reservation an address reaches, and a threadgroup per lane.
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_grouped(&self, frame: &Frame, region: &Region) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::ProtocolObject;
            use objc2_metal::{
                MTLComputeCommandEncoder, MTLComputePipelineState, MTLResource, MTLResourceUsage,
                MTLSize,
            };

            let grouped = self.grouped.as_ref().ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    "a region was compiled for the grouped form and this stage carries no \
                     lane table; the plan said the grouped path could not cover it",
                )
            })?;
            let layout = grouped
                .layouts
                .get(region.region_index as usize)
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "region {} has no group layout, so its library sampler would \
                             decompose its grid by a row count nobody stated",
                            region.region_index
                        ),
                    )
                })?;

            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            // SAFETY: every reservation is retained by `self`; every offset is zero (the kernel strides off `layout`).
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(grouped.table.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(self.scratch.raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(layout.raw()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(grouped.bindings.raw()), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(grouped.pending_flags.raw()), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(grouped.lane_indices.raw()), 0, 8);
                encoder.setBuffer_offset_atIndex(Some(grouped.row_meta.raw()), 0, 9);
                encoder.setBuffer_offset_atIndex(Some(grouped.row_indices.raw()), 0, 10);
            }

            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            resident(
                &self.status,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            for cell in &self.bound {
                resident(
                    &cell.slab,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
            for held in self.intrinsics.iter().flatten() {
                resident(&held.base, MTLResourceUsage::Read);
            }

            // A library sampler declines any width but 256; fused takes the narrower of its buffer width and the pipeline's.
            let rows = grouped
                .layout_words
                .get(region.region_index as usize)
                .map_or(1, |words| words.reserved1 as usize);
            let (groups, threads) = match region.form {
                Form::Fused | Form::Streamed => {
                    unreachable!("`encode_into` routes the single-lane and streamed forms")
                }
                Form::GroupedLibrary => ((GROUPED_LANES as usize) * rows, LIBRARY_SAMPLER_THREADS),
                Form::Grouped => (
                    GROUPED_LANES as usize,
                    region
                        .pipeline()
                        .maxTotalThreadsPerThreadgroup()
                        .clamp(1, REGION_THREADS as usize),
                ),
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups.max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    /// Encode and run one generated region, and wait for it — the host
    /// reads status/cells/scratch right after. Errors on GPU refusal or off Apple.
    pub fn launch_region(
        &self,
        device: &Context,
        pipelines: &crate::device::Pipelines,
        rings: &Rings,
        region: &Region,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            let _ = (pipelines, rings);
            let frame = device.frame()?;
            self.encode_into(&frame, region)?;
            frame.commit()
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, pipelines, rings, region);
            Err(Fault::Deviceless)
        }
    }

    /// The kernel's verdict, whole — `eta_exec::Status`, so a refusal can be
    /// named (`eta_exec::describe_fault`) rather than merely counted.
    pub fn status(&self) -> Result<Status> {
        let mut bytes = [0u8; eta_exec::STATUS_BYTES];
        self.status.read(0, &mut bytes)?;
        Status::read(&bytes).ok_or_else(|| {
            Fault::program(
                "program::launch",
                "the status word read back short, which is a reservation this plane carved",
            )
        })
    }

    /// How many channel slots this stage binds.
    #[must_use]
    pub const fn channel_count(&self) -> u32 {
        self.channel_count
    }

    /// How many values this stage's scratch carries.
    #[must_use]
    pub const fn value_count(&self) -> u32 {
        self.value_count
    }

    /// How wide one lane's scratch is.
    #[must_use]
    pub const fn scratch_stride(&self) -> u32 {
        self.scratch_stride
    }
}

/// One `#[repr(C)]` record's bytes. `T` must have no padding holes with
/// meaning — this file's records are flat `u32` structs.
/// Several instances of one program, launched together: one lane table, one
/// scratch pool, and one dispatch per grouped region with a threadgroup per
/// member. The grouped kernel was written for this shape — it strides its
/// per-lane tables by `dispatch_lane` and reads its record through
/// `lane_indices[dispatch_lane]` — and without it a frame carrying sixteen
/// samplers of one program dispatched each alone: sixteen threadgroups one
/// after another on a device that runs them side by side, so the epilogue
/// grew with the lane count while the forward did not.
///
/// A batch is keyed by [`BatchKey`]: what the kernel strides by. Members
/// that share the key share the descriptor, param, offset and binding
/// tables, which are laid down once at build; per fire the batch writes
/// each member's lane record, channel slots and row table, and zeroes the
/// scratch and flags it will use.
#[derive(Debug)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Batch {
    key: BatchKey,
    /// Dispatch lanes this batch seats; grown by rebuilding.
    lanes: u32,
    shape: LaneShape,
    table: Buffer,
    /// `lanes` copies of the members' shared descriptor bytes, one per
    /// dispatch lane (the kernel strides by `layout->value_count`).
    descriptors: Buffer,
    params: Buffer,
    offsets: Buffer,
    bindings: Buffer,
    pending_flags: Buffer,
    lane_indices: Buffer,
    row_meta: Buffer,
    row_indices: Buffer,
    /// `lanes × scratch_stride`: every member's values for one fire.
    scratch: Buffer,
    layouts: Vec<Buffer>,
    layout_words: Vec<GroupLayout>,
    rows_per_lane: u32,
}

impl Batch {
    /// A batch seating `lanes` members shaped like `template`. Errors when
    /// the template has no grouped seat or a table does not fit.
    pub fn build(device: &Context, template: &Prepared, lanes: u32) -> Result<Batch> {
        let key = template.batch_key().ok_or_else(|| {
            Fault::program(
                "program::launch",
                "a batch was asked for a stage with no grouped seat; nothing in it can be \
                 launched together",
            )
        })?;
        let grouped = template
            .grouped
            .as_ref()
            .expect("the key was answered off this seat one statement ago");
        let lanes = lanes.max(1);
        let shape = LaneShape::of(lanes, grouped.shape.channel_slots_per_lane);
        let bytes = shape.bytes().ok_or_else(|| {
            Fault::program(
                "program::launch",
                "a batch lane table whose size does not fit a u64",
            )
        })?;
        let mut table = Buffer::zeroed(device, bytes)?;
        table.write(
            0,
            &record_bytes(&LaneHeader {
                abi_version: LANE_ABI_VERSION,
                lane_count: lanes,
                channel_slots_per_lane: shape.channel_slots_per_lane,
                flags: 0,
            }),
        )?;
        let replicate = |device: &Context, bytes: &[u8]| -> Result<Buffer> {
            let mut buffer =
                Buffer::zeroed(device, (bytes.len() as u64 * u64::from(lanes)).max(4))?;
            for lane in 0..lanes {
                buffer.write(u64::from(lane) * bytes.len() as u64, bytes)?;
            }
            Ok(buffer)
        };
        let descriptors = replicate(device, &template.descriptor_bytes)?;
        let params = replicate(device, &template.param_bytes)?;
        let mut offsets = Buffer::zeroed(device, template.offset_bytes.len().max(4) as u64)?;
        offsets.write(0, &template.offset_bytes)?;
        let binding_bytes: Vec<u8> = template
            .bindings
            .iter()
            .copied()
            .flat_map(u32::to_le_bytes)
            .collect();
        let bindings = replicate(device, &binding_bytes)?;
        let pending_flags = Buffer::zeroed(
            device,
            (u64::from(lanes) * u64::from(shape.channel_slots_per_lane)).max(1),
        )?;
        let index_bytes: Vec<u8> = (0..lanes).flat_map(u32::to_le_bytes).collect();
        let mut lane_indices = Buffer::zeroed(device, index_bytes.len() as u64)?;
        lane_indices.write(0, &index_bytes)?;
        let row_meta = Buffer::zeroed(device, u64::from(lanes) * size_of::<RowMeta>() as u64)?;
        let rows_per_lane = grouped.trunk_rows.saturating_add(grouped.draft_rows);
        let row_indices = Buffer::zeroed(
            device,
            (u64::from(lanes) * u64::from(rows_per_lane) * size_of::<u32>() as u64).max(4),
        )?;
        let scratch = Buffer::zeroed(
            device,
            (u64::from(lanes) * u64::from(template.scratch_stride)).max(SCRATCH_ALIGN),
        )?;
        let mut layouts = Vec::with_capacity(grouped.layout_words.len());
        for words in &grouped.layout_words {
            let mut buffer = Buffer::zeroed(device, size_of::<GroupLayout>() as u64)?;
            buffer.write(0, &record_bytes(words))?;
            layouts.push(buffer);
        }
        Ok(Batch {
            key,
            lanes,
            shape,
            table,
            descriptors,
            params,
            offsets,
            bindings,
            pending_flags,
            lane_indices,
            row_meta,
            row_indices,
            scratch,
            layouts,
            layout_words: grouped.layout_words.clone(),
            rows_per_lane,
        })
    }

    /// What this batch was built for.
    #[must_use]
    pub fn key(&self) -> BatchKey {
        self.key
    }

    /// Dispatch lanes this batch seats.
    #[must_use]
    pub fn lanes(&self) -> u32 {
        self.lanes
    }

    /// Lay every member's lane down and encode each region once, in stage
    /// order: a grouped region as one dispatch of `members.len()`
    /// threadgroups, a single-lane region once per member with its values at
    /// that member's slot of the pool. Errors when a member does not match
    /// the key, when there are more members than lanes, or on a device fault.
    pub fn encode(
        &mut self,
        frame: &mut Frame,
        regions: &[Region],
        members: &mut [&mut Prepared],
    ) -> Result<()> {
        let count = u32::try_from(members.len()).unwrap_or(u32::MAX);
        if count > self.lanes {
            return Err(Fault::Ceiling {
                what: "dispatch lanes in a batch",
                need: u64::from(count),
                have: u64::from(self.lanes),
            });
        }
        if count == 0 {
            return Ok(());
        }
        let stride = u64::from(self.key.scratch_stride);
        // What a fire starts from: no puts yet, no values yet.
        self.pending_flags.zero_span(
            0,
            u64::from(count) * u64::from(self.shape.channel_slots_per_lane),
        )?;
        if !scratch_zeroing_skipped() {
            // On the device, not the host: a `memset` of the pool through the
            // shared mapping was a third of a millisecond of host time per
            // fire for a vocabulary-wide program, on the critical path between
            // one fire's readout and the next's commit. The blit is ordered by
            // the command buffer and costs the device a fraction of that.
            #[cfg(target_vendor = "apple")]
            {
                frame.fill(self.scratch.slab(), 0, u64::from(count) * stride)?;
                frame.next_pass()?;
            }
            #[cfg(not(target_vendor = "apple"))]
            self.scratch.zero_span(0, u64::from(count) * stride)?;
        }
        let frame: &Frame = frame;
        let vocab = members[0]
            .grouped
            .as_ref()
            .and_then(|g| g.layout_words.first())
            .map_or(0, |words| words.vocab);
        for (index, member) in members.iter().enumerate() {
            let lane = u32::try_from(index).unwrap_or(u32::MAX);
            if member.batch_key() != Some(self.key) {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "member {index} of a batch does not share its key; it would read \
                         another member's tables"
                    ),
                ));
            }
            let grouped = member
                .grouped
                .as_ref()
                .expect("the key was answered off this seat");
            if grouped.layout_words.first().map_or(0, |w| w.vocab) != vocab {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "member {index} of a batch reads a rectangle of another width than \
                         member 0; one launch has one row pitch"
                    ),
                ));
            }
            let mut record = grouped.record;
            record.channel_slot_offset = self.shape.slot_index(lane).unwrap_or(0);
            let at = self.shape.record_offset(lane).ok_or_else(|| {
                Fault::program("program::launch", "a batch member outside its lane table")
            })?;
            self.table.write(at, &record_bytes(&record))?;
            for (&dense, cell) in member.bindings.iter().zip(&member.bound) {
                let slot = LaneChannelSlot {
                    committed_cell: address_of(&cell.slab, cell.committed)?,
                    pending_cell: address_of(&cell.slab, cell.pending)?,
                    expected_head: NO_TICKET,
                    expected_tail: NO_TICKET,
                };
                let at = self.shape.slot_offset(lane, dense).ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "channel {dense} is outside the batch's slot window, which was \
                             carved for the channels its template carries"
                        ),
                    )
                })?;
                self.table.write(at, &record_bytes(&slot))?;
            }
            let row_base = lane * self.rows_per_lane;
            let meta = RowMeta {
                offset: row_base,
                count: self.rows_per_lane,
                mtp_offset: grouped.trunk_rows,
                reserved: 0,
            };
            self.row_meta
                .write(u64::from(lane) * size_of::<RowMeta>() as u64, &record_bytes(&meta))?;
            let rows: Vec<u8> = (0..grouped.trunk_rows)
                .chain((0..grouped.draft_rows).map(|row| grouped.draft_base.saturating_add(row)))
                .flat_map(u32::to_le_bytes)
                .collect();
            if !rows.is_empty() {
                self.row_indices
                    .write(u64::from(row_base) * size_of::<u32>() as u64, &rows)?;
            }
        }
        for (words, buffer) in self.layout_words.iter_mut().zip(self.layouts.iter_mut()) {
            words.lane_count = count;
            words.vocab = vocab;
            buffer.write(0, &record_bytes(words))?;
        }
        for region in regions {
            match region.form {
                Form::Fused => {
                    for (index, member) in members.iter().enumerate() {
                        member.encode_fused(frame, region, &self.scratch, index as u64 * stride)?;
                    }
                }
                Form::Grouped | Form::GroupedLibrary => {
                    self.encode_grouped(frame, region, members, count)?;
                }
                Form::Streamed => {
                    self.encode_streamed(frame, region, members, count)?;
                }
            }
        }
        Ok(())
    }

    /// One streamed region for every member: the batch's bindings, then one
    /// dispatch per step over a grid of `groups × count`.
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_streamed(
        &self,
        frame: &Frame,
        region: &Region,
        members: &[&mut Prepared],
        count: u32,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::ProtocolObject;
            use objc2_metal::{
                MTLComputeCommandEncoder, MTLComputePipelineState, MTLResource, MTLResourceUsage,
            };
            let layout = self
                .layouts
                .get(region.region_index as usize)
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!("region {} has no group layout in this batch", region.region_index),
                    )
                })?;
            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            // SAFETY: every buffer is retained by `self`; every offset is zero.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.table.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(self.scratch.raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(layout.raw()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(self.bindings.raw()), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(self.pending_flags.raw()), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(self.lane_indices.raw()), 0, 8);
                encoder.setBuffer_offset_atIndex(Some(self.row_meta.raw()), 0, 9);
                encoder.setBuffer_offset_atIndex(Some(self.row_indices.raw()), 0, 10);
            }
            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            for member in members {
                resident(
                    &member.status,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                for cell in &member.bound {
                    resident(
                        &cell.slab,
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                }
                for held in member.intrinsics.iter().flatten() {
                    resident(&held.base, MTLResourceUsage::Read);
                }
            }
            let template = members
                .first()
                .ok_or_else(|| Fault::program("program::launch", "a batch with no members"))?;
            let threads = streamed_threads(region.pipeline().maxTotalThreadsPerThreadgroup());
            let temporary_bytes =
                u64::from(self.key.scratch_stride.saturating_sub(self.key.temporary_offset));
            self.dump_streamed_tables(region, template)?;
            dispatch_streamed(
                encoder,
                &streamed_dispatches(
                    &region.steps,
                    &template.descriptor_table,
                    u32::try_from(threads).unwrap_or(1),
                    temporary_bytes,
                ),
                threads,
                count as usize,
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    /// `kernel-dump=<dir>`: beside a streamed region's source, its lane-0
    /// tables as `<entry>.tables` — `[value_count, params_per_lane,
    /// scratch_stride, temporary_offset]` as `u32`s, the 32-byte layout
    /// word, then the descriptors, the op params and the offsets — so a
    /// standalone harness can replay one lane of this dispatch table against
    /// the very shapes the program ran with. Written once per entry.
    #[cfg(target_vendor = "apple")]
    fn dump_streamed_tables(&self, region: &Region, template: &Prepared) -> Result<()> {
        let Some(dir) = crate::diag::on().kernel_dump.as_deref() else {
            return Ok(());
        };
        let path = dir.join(format!("{}.tables", region.module.entry()));
        if path.exists() {
            return Ok(());
        }
        let Some(words) = self.layout_words.get(region.region_index as usize) else {
            return Ok(());
        };
        let value_count = template.descriptor_table.len();
        let params_per_lane = words.reserved2 as usize;
        let mut descriptors = vec![0u8; value_count * size_of::<ValueDesc>()];
        self.descriptors.read(0, &mut descriptors)?;
        let mut params = vec![0u8; params_per_lane * size_of::<OpParams>()];
        self.params.read(0, &mut params)?;
        let mut offsets = vec![0u8; value_count * 4];
        self.offsets.read(0, &mut offsets)?;
        let mut bytes = Vec::new();
        for word in [
            value_count as u32,
            params_per_lane as u32,
            self.key.scratch_stride,
            self.key.temporary_offset,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&record_bytes(words));
        bytes.extend_from_slice(&descriptors);
        bytes.extend_from_slice(&params);
        bytes.extend_from_slice(&offsets);
        let _ = std::fs::write(path, bytes);
        Ok(())
    }

    /// One grouped region for every member: the batch's eleven bindings, a
    /// residency declaration per member reservation, `count` threadgroups
    /// (times the rows for a library sampler).
    #[cfg_attr(not(target_vendor = "apple"), allow(unused_variables))]
    fn encode_grouped(
        &self,
        frame: &Frame,
        region: &Region,
        members: &[&mut Prepared],
        count: u32,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::runtime::ProtocolObject;
            use objc2_metal::{
                MTLComputeCommandEncoder, MTLComputePipelineState, MTLResource, MTLResourceUsage,
                MTLSize,
            };
            let layout = self
                .layouts
                .get(region.region_index as usize)
                .ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!("region {} has no group layout in this batch", region.region_index),
                    )
                })?;
            let encoder = frame.encoder();
            encoder.setComputePipelineState(region.pipeline());
            if let Some(template) = members.first() {
                self.dump_streamed_tables(region, template)?;
            }
            // SAFETY: every buffer is retained by `self`; every offset is zero (the kernel strides off `layout`).
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.table.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(self.descriptors.raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(self.params.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(self.offsets.raw()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(self.scratch.raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(layout.raw()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(self.bindings.raw()), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(self.pending_flags.raw()), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(self.lane_indices.raw()), 0, 8);
                encoder.setBuffer_offset_atIndex(Some(self.row_meta.raw()), 0, 9);
                encoder.setBuffer_offset_atIndex(Some(self.row_indices.raw()), 0, 10);
            }
            let resident = |buffer: &Buffer, usage: MTLResourceUsage| {
                let resource: &ProtocolObject<dyn MTLResource> =
                    ProtocolObject::from_ref(&**buffer.slab());
                encoder.useResource_usage(resource, usage);
            };
            for member in members {
                resident(
                    &member.status,
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
                for cell in &member.bound {
                    resident(
                        &cell.slab,
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                }
                for held in member.intrinsics.iter().flatten() {
                    resident(&held.base, MTLResourceUsage::Read);
                }
            }
            let rows = self
                .layout_words
                .get(region.region_index as usize)
                .map_or(1, |words| words.reserved1 as usize);
            let (groups, threads) = match region.form {
                Form::Fused | Form::Streamed => {
                    unreachable!("`Batch::encode` routes the single-lane and streamed forms")
                }
                Form::GroupedLibrary => ((count as usize) * rows, LIBRARY_SAMPLER_THREADS),
                Form::Grouped => (
                    count as usize,
                    region
                        .pipeline()
                        .maxTotalThreadsPerThreadgroup()
                        .clamp(1, REGION_THREADS as usize),
                ),
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups.max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }
}

/// Issue a streamed region's dispatches: the step word at 11, then a grid of
/// `groups × lanes` threadgroups of `threads`.
#[cfg(target_vendor = "apple")]
fn dispatch_streamed(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    dispatches: &[(StepWord, u32)],
    threads: usize,
    lanes: usize,
) {
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
    for (word, groups) in dispatches {
        // SAFETY: `word` lives for the call; Metal copies the bytes.
        unsafe {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(word).cast(),
                size_of::<StepWord>(),
                11,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (*groups).max(1) as usize,
                height: lanes.max(1),
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
    }
}

fn record_bytes<T: Copy>(record: &T) -> Vec<u8> {
    // SAFETY: as stated above; the slice's life is this expression's.
    let bytes =
        unsafe { std::slice::from_raw_parts((record as *const T).cast::<u8>(), size_of::<T>()) };
    bytes.to_vec()
}

/// A slice of records, end to end.
fn records_bytes<T: Copy>(records: &[T]) -> Vec<u8> {
    records.iter().flat_map(record_bytes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_channel_this_instance_does_not_carry_is_refused_by_number() {
        let rings = Rings {
            slabs: Vec::new(),
            shapes: Vec::new(),
            shared: Vec::new(),
        };
        let said = rings.cell_offset(2, 0).expect_err("no such channel").to_string();
        assert!(said.contains('2'), "the refusal names the channel: {said}");
    }

    #[test]
    fn the_shared_op_record_is_the_emitted_one() {
        // `M1OpParams` is sixteen `uint`s; a drift here computes garbage silently.
        assert_eq!(size_of::<OpParams>(), 64);
        assert_eq!(size_of::<Status>(), 16);
    }

    #[test]
    fn the_first_channel_binds_where_the_emitter_writes_it() {
        // `emit_fused_region` writes `7 + channel * 2` / `8 + channel * 2`.
        assert_eq!(FIRST_CHANNEL_BUFFER, 7);
    }

    /// `M3GroupLayout`/`M3RowMeta` have no host struct to pin via `offset_of!`,
    /// so they're compared against the emitted text — a drift shifts every field after, silently.
    #[test]
    fn the_group_layout_matches_the_emitted_struct() {
        let preamble = eta_compiler::codegen::metal::preamble::grouped_preamble();
        let fields = |name: &str| -> Vec<String> {
            let open = format!("struct {name} {{");
            let at = preamble
                .find(&open)
                .unwrap_or_else(|| panic!("the grouped preamble declares no `{name}`"))
                + open.len();
            let body = &preamble[at..at + preamble[at..].find("};").expect("a closed struct")];
            body.split(';')
                .filter_map(|field| {
                    let field = field.trim();
                    (!field.is_empty()).then(|| field.to_string())
                })
                .collect()
        };
        assert_eq!(
            fields("M3GroupLayout"),
            [
                "uint lane_count",
                "uint value_count",
                "uint scratch_stride",
                "uint temporary_offset",
                "uint vocab",
                "uint reserved0",
                "uint reserved1",
                "uint reserved2",
            ],
            "`GroupLayout` in this file is the host half of the emitted \
             `M3GroupLayout`, and the two have parted"
        );
        assert_eq!(size_of::<GroupLayout>(), 8 * size_of::<u32>());
        assert_eq!(
            fields("M3RowMeta"),
            ["uint offset", "uint count", "uint mtp_offset", "uint reserved"],
            "`RowMeta` in this file is the host half of the emitted `M3RowMeta`, \
             and the two have parted"
        );
        assert_eq!(size_of::<RowMeta>(), 4 * size_of::<u32>());
    }

    /// The eleven bindings every grouped kernel takes, read off the
    /// emitter's text — a mismatch wouldn't fault, just compute garbage.
    #[test]
    fn the_grouped_samplers_take_the_bindings_this_file_writes() {
        // In binding order, so a reader sees the ABI rather than a count.
        const BOUND: [&str; 11] = [
            "lane_bytes",
            "all_descriptors",
            "params",
            "offsets",
            "all_scratch",
            "layout",
            "channel_bindings",
            "pending_flags",
            "lane_indices",
            "all_row_meta",
            "row_indices",
        ];
        let demanded = format!("threads != {LIBRARY_SAMPLER_THREADS}u");
        for signature in [
            eta_compiler::codegen::metal::nucleus::SIGNATURE,
            eta_compiler::codegen::metal::topk::SIGNATURE,
        ] {
            for (index, name) in BOUND.iter().enumerate() {
                assert!(
                    signature.contains(&format!("{name} [[buffer({index})]]")),
                    "a grouped library sampler does not take `{name}` at buffer \
                     {index}, and `encode_grouped` binds it there"
                );
            }
            assert!(
                signature.contains(&demanded),
                "a grouped library sampler no longer refuses every width but \
                 {LIBRARY_SAMPLER_THREADS}, so this dispatch's width is a guess"
            );
        }
    }
}
