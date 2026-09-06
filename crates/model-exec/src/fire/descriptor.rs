//! [`FireDescriptor`]: the fire's window table, as a struct and as bytes —
//! the one mutable channel into a recorded graph. The eager walk reads the
//! struct in place; a CUDA shell packs it into pinned memory and uploads it
//! once, in front of the launch.

use model_ir::ClassSet;

use crate::fire::Fault;
use crate::fire::compose::{ClassWindow, Composition, LaneRow, MaskSpan, WindowTable};
use crate::{Error, Result};

/// `"FIRE"`, big-endian in the spelling and little-endian on the wire — the
/// first four bytes of any descriptor.
pub const MAGIC: u32 = 0x4649_5245;

/// The layout's version. Bumped by any change to the wire layout; checked on
/// unpack, never negotiated. A descriptor carrying an older version is
/// refused by name ([`Fault::DescriptorAbi`]) and never regenerated, since
/// nothing persists a descriptor across builds.
pub const ABI_VERSION: u32 = 3;

/// Bytes before the class table.
pub const HEADER_BYTES: u64 = 40;

/// Bytes per class record: `row_offset, rows, lane_offset, lanes`.
pub const CLASS_BYTES: u64 = 16;

/// Bytes per lane record: `word, source, class, row_offset, rows`.
pub const LANE_BYTES: u64 = 24;

/// Bytes per PATCH lane record: `patch_offset, patches, image_offset, images`.
/// Not a second copy of the token record — word/source/class are lane
/// properties, already on the wire once.
pub const PATCH_LANE_BYTES: u64 = 16;

/// Bytes per VOXEL lane record: `voxel_offset, voxels, clip_offset, clips`.
pub const VOXEL_LANE_BYTES: u64 = 16;

/// One fire's window table, as the walk and the shells read it: a
/// [`Composition`] minus the provenance — counts, offsets, words, with no
/// borrow into the batch that produced it (a shell keeps this resident and
/// overwrites it in place every fire).
///
/// Packed layout ([`pack`](FireDescriptor::pack)/[`unpack`](FireDescriptor::unpack)),
/// little-endian, fixed-width records, no padding:
///
/// ```text
/// offset  bytes  field
///      0      4  magic "FIRE"
///      4      4  abi version
///      8      4  rows          total token rows this fire carries
///     12      4  lanes         how many lane records follow the class table
///     16      4  bucket        the shape bucket, i.e. which graph
///     20      4  classes       how many class records follow the header
///     24      4  patch_rows    total PATCH rows this fire carries
///     28      4  patch_bucket  the patch rung, i.e. which tower graph
///     32      4  voxel_rows    total VOXEL rows this fire carries
///     36      4  voxel_bucket  the voxel rung, i.e. which VAE graph
///     40     16  class[0]      row_offset, rows, lane_offset, lanes
///    ...     16  class[n-1]
///    ...     24  lane[0]       word (8), source, class, row_offset, rows
///    ...     24  lane[m-1]
///  --- the patch trailer, present iff patch_rows > 0 ---
///    ...     16  patch_class[0]  patch_offset, patches, image_offset, images
///    ...     16  patch_class[n-1]
///    ...     16  patch_lane[0]   patch_offset, patches, image_offset, images
///    ...     16  patch_lane[m-1]
///  --- the voxel trailer, present iff voxel_rows > 0 ---
///    ...     16  voxel_class[0]  voxel_offset, voxels, clip_offset, clips
///    ...     16  voxel_class[n-1]
///    ...     16  voxel_lane[0]   voxel_offset, voxels, clip_offset, clips
///    ...     16  voxel_lane[m-1]
/// ```
///
/// A fire with `patch_rows == 0` packs no patch trailer and one with
/// `voxel_rows == 0` no voxel trailer; ABI 3 grew the header by the two
/// voxel words (ABI 2's 32-byte header had none), so ABI 2 bytes are
/// refused by name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FireDescriptor {
    /// Total token rows.
    pub rows: u32,
    /// The shape bucket these rows round up to — which recorded graph runs.
    pub bucket: u32,
    /// One window per class of the artifact, indexed by class.
    pub classes: WindowTable,
    /// The lanes in fire order, carrying the permutation back to submission
    /// order — and, since ABI 2, each lane's place in BOTH seriations.
    pub lanes: Vec<LaneRow>,
    /// Total PATCH rows. Zero for every fire of a text-only artifact and for
    /// every text-only fire of a tower one, which is the same number and the
    /// true one in both cases.
    pub patch_rows: u32,
    /// How many IMAGES this fire carries — the patch axis's lane count.
    pub images: u32,
    /// Per region, the most rows one launch of it may cover, `0` for no
    /// cap. A streamed load caps the region that reads a router's seats
    /// (`slots / top_k` rows name at most `slots` experts), so a fire whose
    /// rows would route past the slab is walked as several runs, each seated
    /// at its own cut — sub-batching the segment. Empty means no region is
    /// capped; a shell that streams nothing leaves it so.
    pub run_caps: Vec<u32>,
    /// Per region, the most expert-major passes a capped run is walked in
    /// (`compose::pass_spans`): `ceil(experts / slots)` for a streamed
    /// router's segment, `0`/`1` to cut rows instead. Empty means none.
    pub run_passes: Vec<u32>,
    /// The patch rung these patch rows round up to — which tower graph runs.
    pub patch_bucket: u32,
    /// One PATCH window per class of the artifact, indexed by class — a
    /// second table over the same classes, since patch rows and token rows
    /// don't break at the same places. All-zero, and packed as nothing at
    /// all, for a fire with no patch rows.
    pub patch_classes: WindowTable,
    /// Total VOXEL rows (port voxels). Zero for every fire with no clip.
    pub voxel_rows: u32,
    /// How many CLIPS this fire carries — the voxel axis's lane count.
    pub clips: u32,
    /// The voxel rung these voxel rows round up to — which VAE graph runs.
    pub voxel_bucket: u32,
    /// One VOXEL window per class of the artifact — the third table over
    /// the same classes. All-zero, and packed as nothing at all, for a fire
    /// with no voxel rows.
    pub voxel_classes: WindowTable,
}

impl FireDescriptor {
    /// The descriptor for a composition.
    #[must_use]
    pub fn of(composition: &Composition) -> FireDescriptor {
        FireDescriptor {
            rows: composition.rows(),
            bucket: composition.bucket(),
            classes: composition.classes().clone(),
            lanes: composition.lanes().to_vec(),
            patch_rows: composition.patch_rows(),
            images: composition.images(),
            patch_bucket: composition.patch_bucket(),
            patch_classes: composition.patch_classes().clone(),
            voxel_rows: composition.voxel_rows(),
            clips: composition.clips(),
            voxel_bucket: composition.voxel_bucket(),
            voxel_classes: composition.voxel_classes().clone(),
            run_caps: Vec::new(),
            run_passes: Vec::new(),
        }
    }

    /// Does this fire carry the third row axis at all? The predicate the
    /// voxel trailer is keyed on, and the one a shell asks before launching
    /// a VAE exec.
    #[must_use]
    pub fn has_voxels(&self) -> bool {
        self.voxel_rows > 0
    }

    /// Does this fire carry the second row axis at all? The predicate the
    /// trailer is keyed on, and the one a shell asks before launching a
    /// tower exec.
    #[must_use]
    pub fn has_patches(&self) -> bool {
        self.patch_rows > 0
    }

    /// This fire's window table on one row axis — the token seriation for a
    /// trunk region, the patch seriation for a tower one. A region belongs
    /// to exactly one axis, so cutting its window is a lookup, never a
    /// merge.
    #[must_use]
    pub fn table(&self, axis: model_ir::RowAxis) -> &WindowTable {
        match axis {
            model_ir::RowAxis::Tokens => &self.classes,
            model_ir::RowAxis::Patches => &self.patch_classes,
            model_ir::RowAxis::Voxels => &self.voxel_classes,
        }
    }

    /// How many PATCH rows a node with this class mask runs over — the
    /// zero-row question on the second axis, and a forward onto
    /// [`table`](FireDescriptor::table) for the callers that name the axis in
    /// the method rather than in an argument.
    #[must_use]
    pub fn patch_rows_of(&self, mask: &ClassSet) -> u32 {
        self.table(model_ir::RowAxis::Patches).rows_of(mask)
    }

    /// How many lanes this fire carries.
    #[must_use]
    pub fn lane_count(&self) -> u32 {
        self.lanes.len() as u32
    }

    /// How many token rows a node with this class mask runs over — the
    /// zero-row question, asked against the table that crossed to the device.
    #[must_use]
    pub fn rows_of(&self, mask: &ClassSet) -> u32 {
        self.table(model_ir::RowAxis::PRIMARY).rows_of(mask)
    }

    /// The one row-and-lane interval a node with this class mask runs over —
    /// the window question, asked against the table that crossed to the
    /// device. [`WindowTable::span`] states what the two answers mean.
    ///
    /// # Errors
    ///
    /// The number of runs the mask covers, when that is more than one.
    pub fn span(&self, mask: &ClassSet) -> core::result::Result<Option<MaskSpan>, usize> {
        self.table(model_ir::RowAxis::PRIMARY).span(mask)
    }

    /// Every interval a node with this class mask runs over, ascending — the
    /// slow path's launches, asked against the table that crossed to the
    /// device. [`WindowTable::spans`] states what the list means.
    #[must_use]
    pub fn spans(&self, mask: &ClassSet) -> Vec<MaskSpan> {
        self.table(model_ir::RowAxis::PRIMARY).spans(mask)
    }

    /// The same, into a buffer the caller keeps — what the walk asks once per
    /// region ([`WindowTable::spans_into`] says why it is not a `Vec`).
    pub fn spans_into(&self, mask: &ClassSet, out: &mut Vec<MaskSpan>) {
        self.table(model_ir::RowAxis::PRIMARY).spans_into(mask, out);
    }

    /// How many bytes [`pack`](FireDescriptor::pack) will write.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        HEADER_BYTES
            + CLASS_BYTES * self.classes.len() as u64
            + LANE_BYTES * self.lanes.len() as u64
            + if self.has_patches() {
                CLASS_BYTES * self.patch_classes.len() as u64
                    + PATCH_LANE_BYTES * self.lanes.len() as u64
            } else {
                0
            }
            + if self.has_voxels() {
                CLASS_BYTES * self.voxel_classes.len() as u64
                    + VOXEL_LANE_BYTES * self.lanes.len() as u64
            } else {
                0
            }
    }

    /// The descriptor, flat.
    #[must_use]
    pub fn pack(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bytes() as usize);
        put32(&mut out, MAGIC);
        put32(&mut out, ABI_VERSION);
        put32(&mut out, self.rows);
        put32(&mut out, self.lane_count());
        put32(&mut out, self.bucket);
        put32(&mut out, self.classes.len() as u32);
        put32(&mut out, self.patch_rows);
        put32(&mut out, self.patch_bucket);
        put32(&mut out, self.voxel_rows);
        put32(&mut out, self.voxel_bucket);

        for window in self.classes.as_slice() {
            put32(&mut out, window.row_offset);
            put32(&mut out, window.rows);
            put32(&mut out, window.lane_offset);
            put32(&mut out, window.lanes);
        }
        for lane in &self.lanes {
            out.extend_from_slice(&lane.word.to_le_bytes());
            put32(&mut out, lane.source);
            put32(&mut out, lane.class);
            put32(&mut out, lane.row_offset);
            put32(&mut out, lane.rows);
        }
        // The trailer, nothing at all when there are no patch rows.
        if self.has_patches() {
            for window in self.patch_classes.as_slice() {
                put32(&mut out, window.row_offset);
                put32(&mut out, window.rows);
                put32(&mut out, window.lane_offset);
                put32(&mut out, window.lanes);
            }
            for lane in &self.lanes {
                put32(&mut out, lane.patch_offset);
                put32(&mut out, lane.patches);
                put32(&mut out, lane.image_offset);
                put32(&mut out, lane.images);
            }
        }
        // The voxel trailer, the same way.
        if self.has_voxels() {
            for window in self.voxel_classes.as_slice() {
                put32(&mut out, window.row_offset);
                put32(&mut out, window.rows);
                put32(&mut out, window.lane_offset);
                put32(&mut out, window.lanes);
            }
            for lane in &self.lanes {
                put32(&mut out, lane.voxel_offset);
                put32(&mut out, lane.voxels);
                put32(&mut out, lane.clip_offset);
                put32(&mut out, lane.clips);
            }
        }
        out
    }

    /// The descriptor these bytes carry.
    ///
    /// Checks what a device cannot: a malformed descriptor doesn't fault on
    /// the far side, it computes over whatever rows the wrong numbers name.
    /// So the length must be exact, and class rows must add up to the
    /// header's total.
    ///
    /// # Errors
    ///
    /// One of the five `Descriptor*` faults, naming which of those it was.
    pub fn unpack(bytes: &[u8]) -> Result<FireDescriptor> {
        if (bytes.len() as u64) < HEADER_BYTES {
            return Err(Error::Fire(Fault::DescriptorShort { bytes: bytes.len() }));
        }
        let magic = take32(bytes, 0);
        if magic != MAGIC {
            return Err(Error::Fire(Fault::DescriptorMagic { saw: magic }));
        }
        // Refused by name, both numbers included so the refusal is legible
        // without a hexdump.
        let saw = take32(bytes, 4);
        if saw != ABI_VERSION {
            return Err(Error::Fire(Fault::DescriptorAbi {
                saw,
                speaks: ABI_VERSION,
            }));
        }
        let rows = take32(bytes, 8);
        let lanes = u64::from(take32(bytes, 12));
        let bucket = take32(bytes, 16);
        let classes = u64::from(take32(bytes, 20));
        let patch_rows = take32(bytes, 24);
        let patch_bucket = take32(bytes, 28);
        let voxel_rows = take32(bytes, 32);
        let voxel_bucket = take32(bytes, 36);

        let trailer = if patch_rows > 0 {
            CLASS_BYTES * classes + PATCH_LANE_BYTES * lanes
        } else {
            0
        };
        let voxel_trailer = if voxel_rows > 0 {
            CLASS_BYTES * classes + VOXEL_LANE_BYTES * lanes
        } else {
            0
        };
        let want =
            HEADER_BYTES + CLASS_BYTES * classes + LANE_BYTES * lanes + trailer + voxel_trailer;
        if bytes.len() as u64 != want {
            return Err(Error::Fire(Fault::DescriptorLength {
                bytes: bytes.len(),
                want,
            }));
        }

        let mut table = Vec::with_capacity(classes as usize);
        let mut counted: u64 = 0;
        for c in 0..classes {
            let at = (HEADER_BYTES + CLASS_BYTES * c) as usize;
            let window = ClassWindow {
                row_offset: take32(bytes, at),
                rows: take32(bytes, at + 4),
                lane_offset: take32(bytes, at + 8),
                lanes: take32(bytes, at + 12),
            };
            counted += u64::from(window.rows);
            table.push(window);
        }
        if counted != u64::from(rows) {
            return Err(Error::Fire(Fault::DescriptorRows {
                counted,
                header: rows,
            }));
        }

        let base = HEADER_BYTES + CLASS_BYTES * classes;
        let mut placed = Vec::with_capacity(lanes as usize);
        for l in 0..lanes {
            let at = (base + LANE_BYTES * l) as usize;
            placed.push(LaneRow {
                word: take64(bytes, at),
                source: take32(bytes, at + 8),
                class: take32(bytes, at + 12),
                row_offset: take32(bytes, at + 16),
                rows: take32(bytes, at + 20),
                ..LaneRow::default()
            });
        }

        // The trailer's two halves are checked the way the token halves are:
        // patch windows must add up to the patch row count the header
        // claims.
        let mut patch_table = vec![ClassWindow::default(); classes as usize];
        if patch_rows > 0 {
            let at_classes = base + LANE_BYTES * lanes;
            let mut counted: u64 = 0;
            for c in 0..classes {
                let at = (at_classes + CLASS_BYTES * c) as usize;
                let window = ClassWindow {
                    row_offset: take32(bytes, at),
                    rows: take32(bytes, at + 4),
                    lane_offset: take32(bytes, at + 8),
                    lanes: take32(bytes, at + 12),
                };
                counted += u64::from(window.rows);
                patch_table[c as usize] = window;
            }
            if counted != u64::from(patch_rows) {
                return Err(Error::Fire(Fault::DescriptorPatchRows {
                    counted,
                    header: patch_rows,
                }));
            }
            let at_lanes = at_classes + CLASS_BYTES * classes;
            for (l, lane) in placed.iter_mut().enumerate() {
                let at = (at_lanes + PATCH_LANE_BYTES * l as u64) as usize;
                lane.patch_offset = take32(bytes, at);
                lane.patches = take32(bytes, at + 4);
                lane.image_offset = take32(bytes, at + 8);
                lane.images = take32(bytes, at + 12);
            }
        }
        let images = placed.iter().map(|lane| lane.images).sum();

        // And the voxel trailer, checked the same way.
        let mut voxel_table = vec![ClassWindow::default(); classes as usize];
        if voxel_rows > 0 {
            let at_classes = base + LANE_BYTES * lanes + trailer;
            let mut counted: u64 = 0;
            for c in 0..classes {
                let at = (at_classes + CLASS_BYTES * c) as usize;
                let window = ClassWindow {
                    row_offset: take32(bytes, at),
                    rows: take32(bytes, at + 4),
                    lane_offset: take32(bytes, at + 8),
                    lanes: take32(bytes, at + 12),
                };
                counted += u64::from(window.rows);
                voxel_table[c as usize] = window;
            }
            if counted != u64::from(voxel_rows) {
                return Err(Error::Fire(Fault::DescriptorVoxelRows {
                    counted,
                    header: voxel_rows,
                }));
            }
            let at_lanes = at_classes + CLASS_BYTES * classes;
            for (l, lane) in placed.iter_mut().enumerate() {
                let at = (at_lanes + VOXEL_LANE_BYTES * l as u64) as usize;
                lane.voxel_offset = take32(bytes, at);
                lane.voxels = take32(bytes, at + 4);
                lane.clip_offset = take32(bytes, at + 8);
                lane.clips = take32(bytes, at + 12);
            }
        }
        let clips = placed.iter().map(|lane| lane.clips).sum();

        Ok(FireDescriptor {
            rows,
            bucket,
            classes: WindowTable::new(table),
            lanes: placed,
            patch_rows,
            images,
            patch_bucket,
            patch_classes: WindowTable::new(patch_table),
            voxel_rows,
            clips,
            voxel_bucket,
            voxel_classes: WindowTable::new(voxel_table),
            run_caps: Vec::new(),
            run_passes: Vec::new(),
        })
    }
}

fn put32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// The `u32` at `at`. Every caller has already checked the length against the
/// header, so a short read here is impossible; the `unwrap_or` is what that
/// impossibility costs, and it reads zero rather than panicking in a fire
/// path.
fn take32(bytes: &[u8], at: usize) -> u32 {
    bytes
        .get(at..at + 4)
        .and_then(|b| b.try_into().ok())
        .map_or(0, u32::from_le_bytes)
}

fn take64(bytes: &[u8], at: usize) -> u64 {
    bytes
        .get(at..at + 8)
        .and_then(|b| b.try_into().ok())
        .map_or(0, u64::from_le_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::compose::{Lane, compose};
    use crate::fire::fixture::{Build, fact};
    use crate::{Error, fire::Fault};
    use model_compiler::{Budget, DeviceProfile, compile};
    use model_ir::Guard;

    fn budget() -> Budget {
        Budget::new(8, 64)
    }

    // A decode/prefill split, in four nodes.
    fn plan() -> Build {
        let mut b = Build::new();
        let x = b.input(4);
        let q = b.op(x, 4, Guard::Always);
        let d = b.op(q, 4, fact(0));
        let p = b.op(q, 4, Guard::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 4);
        let y = b.op(o, 4, Guard::Always);
        b.out(y);
        b
    }

    // That fire, as a descriptor.
    fn descriptor() -> FireDescriptor {
        let b = plan();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let lanes = [
            Lane::new(0, 7),
            Lane::new(0, 3),
            Lane::new(1, 1),
            Lane::new(1, 1),
            Lane::new(1, 1),
        ];
        FireDescriptor::of(&compose(&compiled, &budget(), &lanes).expect("composes"))
    }

    #[test]
    fn a_descriptor_survives_the_round_trip_whole() {
        let before = descriptor();
        let bytes = before.pack();

        assert_eq!(bytes.len() as u64, before.bytes());
        assert_eq!(
            bytes.len() as u64,
            HEADER_BYTES + CLASS_BYTES * 2 + LANE_BYTES * 5,
        );
        assert_eq!(FireDescriptor::unpack(&bytes), Ok(before));
    }

    #[test]
    fn the_header_says_fire_and_which_layout_it_is() {
        let bytes = descriptor().pack();
        assert_eq!(&bytes[0..4], &MAGIC.to_le_bytes());
        assert_eq!(&bytes[4..8], &ABI_VERSION.to_le_bytes());
        // rows, lanes, bucket, classes.
        assert_eq!(&bytes[8..12], &13u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &5u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &13u32.to_le_bytes());
        assert_eq!(&bytes[20..24], &2u32.to_le_bytes());
        // Words 6/7 were one reserved `u64` under ABI 1; ABI 2 spends them as
        // `patch_rows`/`patch_bucket`, and ABI 3 adds `voxel_rows`/
        // `voxel_bucket` as words 8/9. No image and no clip here, so all four
        // read zero.
        assert_eq!(&bytes[24..28], &0u32.to_le_bytes());
        assert_eq!(&bytes[28..32], &0u32.to_le_bytes());
        assert_eq!(&bytes[32..36], &0u32.to_le_bytes());
        assert_eq!(&bytes[36..40], &0u32.to_le_bytes());
    }

    #[test]
    fn bytes_that_are_not_a_descriptor_are_refused_and_named() {
        let good = descriptor().pack();

        assert!(matches!(
            FireDescriptor::unpack(&good[..16]),
            Err(Error::Fire(Fault::DescriptorShort { .. })),
        ));
        assert!(matches!(
            FireDescriptor::unpack(&[0u8; 8]),
            Err(Error::Fire(Fault::DescriptorShort { .. })),
        ));

        let mut foreign = good.clone();
        foreign[0] ^= 0xff;
        assert!(matches!(
            FireDescriptor::unpack(&foreign),
            Err(Error::Fire(Fault::DescriptorMagic { .. })),
        ));

        let mut newer = good.clone();
        newer[4..8].copy_from_slice(&(ABI_VERSION + 1).to_le_bytes());
        assert!(matches!(
            FireDescriptor::unpack(&newer),
            Err(Error::Fire(Fault::DescriptorAbi { .. })),
        ));

        // A header that claims one more class than the bytes carry: the record
        // after it would be read half out of the lane table.
        let mut miscounted = good.clone();
        miscounted[20..24].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(
            FireDescriptor::unpack(&miscounted),
            Err(Error::Fire(Fault::DescriptorLength { .. })),
        ));

        // Windows that do not add up to the total the header claims — the
        // corruption that does not fault on the far side, it computes.
        let mut wrong = good;
        wrong[8..12].copy_from_slice(&12u32.to_le_bytes());
        assert!(matches!(
            FireDescriptor::unpack(&wrong),
            Err(Error::Fire(Fault::DescriptorRows { .. })),
        ));
    }

    // Older bytes are refused by name, never regenerated into version 3.
    #[test]
    fn an_older_descriptor_is_refused_by_name_and_not_regenerated() {
        let before = descriptor();
        for older in [1u32, 2] {
            let mut bytes = before.pack();
            bytes[4..8].copy_from_slice(&older.to_le_bytes());

            let refusal = FireDescriptor::unpack(&bytes).expect_err("an old ABI is not v3");
            assert_eq!(
                refusal,
                Error::Fire(Fault::DescriptorAbi {
                    saw: older,
                    speaks: ABI_VERSION,
                }),
            );
            let said = refusal.to_string();
            assert!(said.contains(&older.to_string()) && said.contains('3'), "{said}");
            assert!(said.contains("never negotiated"), "{said}");
        }
    }
}
