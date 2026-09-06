//! Values: `Def` (provenance) × `Ty` (type), orthogonal; every value is a
//! `ValueId` — no `WeightId`, no separate cache handle type.

use serde::{Deserialize, Serialize};

use crate::guard::Guard;

/// One id space for every value in a plan: op outputs, weights, cache
/// bindings, runtime inputs, merges. Indexes `Trace::values`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Element type as data — re-export of [`dtype::Dtype`], the IR's one name
/// for an element type.
pub use dtype::Dtype;
/// The tiled affine layout's geometry, beside the [`Dtype`] that names it.
/// `model_dsl::Weight::planes` sizes a repacked weight's three planes with
/// these; `checkpoint` checks a declared repack target against the same two
/// numbers.
pub use dtype::{BIASES, SCALES, TILED_BAND, TILED_STEP};

/// The shape algebra's symbolic dims, sized by runtime budgets (`Tokens` →
/// max_tokens, `Lanes` → max_lanes, `Patches` → max_patches, `Voxels` →
/// max_voxels) when the arena
/// is cut. Which axis a value lives on is read off its type
/// ([`Dim::axis`]) rather than declared beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    Const(u64),
    /// This fire's token count.
    Tokens,
    /// MoE routed rows: tokens × top_k.
    TokensTimes(u32),
    /// Request count (geometry vectors).
    Lanes,
    /// Indptr-shaped: lanes + 1.
    LanesPlus(u32),
    /// This fire's patch count — the rows of the vision tower's window,
    /// concatenated over every image every lane submitted. Not a subset of
    /// the token rectangle: the patch axis gets its own seriation, bucket
    /// ladder and capture unit.
    Patches,
    /// The patch axis's own lane space: how many images this fire carries.
    /// `Images` is to [`Patches`](Dim::Patches) what [`Lanes`](Dim::Lanes)
    /// is to [`Tokens`](Dim::Tokens) — separate because a lane with no
    /// image contributes a lane and no image.
    Images,
    /// Indptr-shaped on the patch axis: `images + 1`.
    ImagesPlus(u32),
    /// This fire's voxel count at the voxel port — the rows of the third
    /// row axis, one row per voxel `(t, h, w)` of every clip every lane
    /// submitted, `w` fastest, clips contiguous in fire order. A VAE's
    /// activations live here (`[Voxels, channels]`); the per-clip box is
    /// the `[Clips, 4]` grid table ([`RuntimeInput::Grid`]).
    Voxels,
    /// `voxels * k`: a rectangle an upsample, a pixel shuffle or an
    /// unpatchify grew by a fixed factor from the port's count. An op that
    /// shrinks rows (a strided conv, an unshuffle) keeps its input's dim and
    /// over-allocates; the grid table says which rows are live.
    VoxelsTimes(u32),
    /// The voxel axis's own lane space: how many clips (images or videos)
    /// this fire carries. `Clips` is to [`Voxels`](Dim::Voxels) what
    /// `Images` is to `Patches`.
    Clips,
    /// Indptr-shaped on the voxel axis: `clips + k`.
    ClipsPlus(u32),
}

/// Which row space a symbolic dim sizes — the discriminator every per-axis
/// table is keyed by. Derived, never declared: [`Dim::axis`] reads it off
/// the type a model text already wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RowAxis {
    /// The token rectangle: `Tokens`, `TokensTimes(k)`, `Lanes`, `LanesPlus(k)`.
    Tokens,
    /// The patch rectangle: `Patches`, `Images`, `ImagesPlus(k)`.
    Patches,
    /// The voxel rectangle: `Voxels`, `VoxelsTimes(k)`, `Clips`,
    /// `ClipsPlus(k)` — the VAE's row space (design D8).
    Voxels,
}

impl RowAxis {
    /// The axis every plan has, and the one a plan that names no other is
    /// entirely made of.
    pub const PRIMARY: RowAxis = RowAxis::Tokens;

    /// Every axis, in discriminant order — what a pass that owes an answer
    /// per row space iterates, and what [`PerAxis`] is laid out along.
    /// `ALL[axis as usize] == axis`, so a `PerAxis` entry is reachable by
    /// the same integer the variant is.
    pub const ALL: [RowAxis; 3] = [RowAxis::Tokens, RowAxis::Patches, RowAxis::Voxels];

    /// How many row spaces there are — [`ALL`](RowAxis::ALL)'s length, and
    /// the width of every [`PerAxis`].
    pub const COUNT: usize = RowAxis::ALL.len();

    /// The name a refusal or a ledger line spells this axis with.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            RowAxis::Tokens => "tokens",
            RowAxis::Patches => "patches",
            RowAxis::Voxels => "voxels",
        }
    }
}

/// One value per row axis, addressed by the axis, so "which axis" is an
/// index rather than a pair of hand-kept fields. Total over the axes: every
/// axis has a value always, so a text-only fire's patch entry is the zero
/// window rather than a missing one. `T` is opaque to this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PerAxis<T>([T; RowAxis::COUNT]);

impl<T> PerAxis<T> {
    /// One value per axis, in [`RowAxis::ALL`]'s order.
    ///
    /// An array rather than one argument per axis: adding a variant then
    /// fails to compile at the fill site, where the caller has the value
    /// in hand to state it.
    pub const fn new(values: [T; RowAxis::COUNT]) -> PerAxis<T> {
        PerAxis(values)
    }

    /// One value per axis, computed from the axis.
    pub fn from_fn(mut f: impl FnMut(RowAxis) -> T) -> PerAxis<T> {
        PerAxis(RowAxis::ALL.map(&mut f))
    }

    /// The same axes, carrying something else — `f` applied entry by entry,
    /// in [`RowAxis::ALL`]'s order.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> PerAxis<U> {
        PerAxis(self.0.map(&mut f))
    }

    /// The entries in axis order, for a caller that wants to walk them
    /// without naming the axes.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// `(axis, value)` per axis, ascending — the loop a pass that owes one
    /// answer per row space writes instead of two blocks.
    pub fn iter(&self) -> impl Iterator<Item = (RowAxis, &T)> {
        RowAxis::ALL.into_iter().zip(self.0.iter())
    }
}

impl<T: Clone> PerAxis<T> {
    /// The same value on every axis.
    #[must_use]
    pub fn splat(value: T) -> PerAxis<T> {
        PerAxis::from_fn(|_| value.clone())
    }
}

impl<T> core::ops::Index<RowAxis> for PerAxis<T> {
    type Output = T;

    /// `table[axis]`, in bounds by construction: the array is
    /// [`RowAxis::COUNT`] wide.
    fn index(&self, axis: RowAxis) -> &T {
        &self.0[axis as usize]
    }
}

impl<T> core::ops::IndexMut<RowAxis> for PerAxis<T> {
    fn index_mut(&mut self, axis: RowAxis) -> &mut T {
        &mut self.0[axis as usize]
    }
}

impl Dim {
    /// The row space this dim sizes, or `None` for a [`Const`](Dim::Const)
    /// — a fixed block is not fire-aligned and so belongs to no axis.
    ///
    /// `Lanes`/`LanesPlus` answer `RowAxis::Tokens`;
    /// [`Images`](Dim::Images)/[`ImagesPlus`](Dim::ImagesPlus) answer
    /// `RowAxis::Patches` for the same reason.
    #[must_use]
    pub fn axis(self) -> Option<RowAxis> {
        match self {
            Dim::Const(_) => None,
            Dim::Tokens | Dim::TokensTimes(_) | Dim::Lanes | Dim::LanesPlus(_) => {
                Some(RowAxis::Tokens)
            }
            Dim::Patches | Dim::Images | Dim::ImagesPlus(_) => Some(RowAxis::Patches),
            Dim::Voxels | Dim::VoxelsTimes(_) | Dim::Clips | Dim::ClipsPlus(_) => {
                Some(RowAxis::Voxels)
            }
        }
    }
}

/// The kinds of host-owned plan objects an op may define. The payload is
/// backend-opaque; only the kind is IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructKind {
    AttnDecodePlan,
    AttnPrefillPlan,
    AttnPrefillPlanSm90,
    MlaPlan,
}

/// Which geometry vector of a cache space a runtime input binds. Each kind
/// says which op family reads it, so a fire owes exactly what its plan names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeomKind {
    /// Per-lane page-list bounds; read by the plan ops (`attention.plan_*`, `mla.plan`).
    Indptr,
    /// The flat page-id list the indptr bounds; read by the plan ops.
    Indices,
    /// Per-lane sequence lengths; read by the plan ops.
    SeqLens,
    /// Per-lane fill of the last page; read by the plan ops.
    LastPageLen,
    /// Per-lane total kv length; read by the plan builders (`attention.plan_*`, `mla.plan`).
    KvLen,
    /// Graph-padding row mask; read by the pool boundary ops (`pool.boundary_*`).
    RowValid,
    /// Token→lane map; read by `pool.attention_lse` (and the metal fire tables).
    RequestOfToken,
    /// Per-token destination page of a kv write; read by the `kv_append` ops.
    WritePage,
    /// Per-token in-page offset of a kv write; read by the `kv_append` ops.
    WriteOffset,
    /// Which ATTENTION GROUP each lane belongs to: `[Dim::Lanes]` `i32`, one
    /// group id per lane in fire order, ascending and dense from 0. A lane is
    /// `(request, stream)`; the lanes one request submitted share its group
    /// and attend together through `attention.ragged`. Space 0 only — the
    /// token axis's own table, whether or not a cache joined it.
    GroupOfLane,
    /// Per-GROUP bounds over the PACKED rows of one row selection:
    /// `[Dim::LanesPlus(1)]` `i32` (a fire has at most as many groups as
    /// lanes), where group `g`'s packed rows are `[indptr[g], indptr[g+1])`.
    /// Packed order is what [`RuntimeInput::RowPermutation`] of the same
    /// selection states: selected lanes sorted by `(group, stream, lane)`,
    /// each lane's rows contiguous. Groups no selected lane belongs to are
    /// skipped, so the CSR is over the groups present, ascending. Read by
    /// `attention.ragged`.
    GroupIndptr { select: Selection },
    /// Per-LANE bounds over the packed rows of one row selection:
    /// `[Dim::LanesPlus(1)]` `i32`, the finer CSR beneath
    /// [`GroupIndptr`](GeomKind::GroupIndptr) — selected lane `j` (in
    /// packed order) owns packed rows `[indptr[j], indptr[j+1])`. What a
    /// lane-block-diagonal ragged attention passes as its indptr.
    LaneIndptr { select: Selection },
    /// Per packed row of one selection, the fire lane index of the row when
    /// its lane's stream is [`Reference`](crate::request::Stream::Reference)
    /// and `-1` otherwise: `[Dim::Tokens]` `i32`. Read by `attention.ragged`
    /// under [`RaggedMask::ReferenceSelfOnly`](crate::ops::attn::RaggedMask).
    ReferenceTag { select: Selection },
}

/// A set of lanes named by their fact words: the lanes whose word satisfies
/// `word & mask == value`. What a row-packing input is keyed by, and the
/// spelling every guard a split arm carries has (a conjunction of fact
/// literals) — [`Selection::of`] reads one off such a guard. `mask == 0` is
/// every lane.
///
/// A pair of words rather than a `Guard` so the input stays `Copy`, `Hash`
/// and self-describing to the host: a fire evaluates it per lane with one
/// `and` and one compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Selection {
    /// The fact bits the selection reads.
    pub mask: u32,
    /// What those bits must be.
    pub value: u32,
}

impl Selection {
    /// Every lane of the fire.
    pub const ALL: Selection = Selection { mask: 0, value: 0 };

    /// The selection a guard states, when the guard is a conjunction of
    /// fact literals (`Always`, `Fact`, `Not(Fact)`, `And` of those) — the
    /// shape every split arm's guard has. `None` for a guard with an `Or`
    /// or a negated conjunction, which no single mask/value pair spells.
    #[must_use]
    pub fn of(guard: &Guard) -> Option<Selection> {
        let mut select = Selection::ALL;
        select.gather(guard).then_some(select)
    }

    fn gather(&mut self, guard: &Guard) -> bool {
        match guard {
            Guard::Always => true,
            Guard::Fact(bit) => self.literal(*bit, true),
            Guard::Not(inner) => match inner.as_ref() {
                Guard::Fact(bit) => self.literal(*bit, false),
                _ => false,
            },
            Guard::And(a, b) => self.gather(a) && self.gather(b),
            Guard::Or(..) => false,
        }
    }

    fn literal(&mut self, bit: u8, set: bool) -> bool {
        if bit >= 32 {
            return false;
        }
        let one = 1u32 << bit;
        let want = if set { one } else { 0 };
        // The same bit stated twice must agree, else the guard is empty.
        if self.mask & one != 0 && self.value & one != want {
            return false;
        }
        self.mask |= one;
        self.value |= want;
        true
    }

    /// Whether a lane with fact word `word` is in the selection.
    #[must_use]
    pub fn holds(self, word: u64) -> bool {
        (word as u32) & self.mask == self.value
    }
}

/// What the engine binds each fire. Geometry is a declared input, not implicit
/// engine state: cache ops become pure functions of visible inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeInput {
    Tokens,
    Positions,
    /// Trunk's triple-wide position stream: `[Dim::Tokens, 3]` i32, one
    /// `(t, h, w)` per token row, read by `elementwise.rope_mrope`. A
    /// second stream rather than a widened `Positions`, so a text that
    /// doesn't rotate this way pays nothing extra.
    MropePositions,
    /// Custom attention mask bits for a kv space; read by `attention.masked`.
    Mask { space: u32 },
    /// One geometry vector of a cache space; `space` matches the group the
    /// caches declare (`CacheRow::Kv::space`).
    Geometry { space: u32, kind: GeomKind },
    /// Which adapter bank each token row routes to; read by
    /// `linear.lora_correct`. `i32`, one entry per token row, `-1` for the
    /// base model. Bare (not keyed): an adapter is a property of the
    /// request, not of a page-id space.
    AdapterRoutes,
    /// The fire's patch rows, pre-unfolded: `[Dim::Patches, C·T·P²]`, one
    /// row per patch, every image of every lane concatenated in the fire's
    /// patch order. Decode/resize happen host-side.
    Patches,
    /// The patch axis's own indptr: `[Dim::ImagesPlus(1)]` `i32`, where
    /// image `i`'s patch rows are `[segments[i], segments[i + 1])` of the
    /// fire's patch rectangle. Read by `attention.dense`.
    PatchSegments,
    /// Where each row of the tower's output lands in the token rectangle:
    /// `i32`, one destination token row per tower row, read by
    /// `layout.scatter_rows`. The only vector crossing both axes, and the
    /// only one the fire path validates before launch (an out-of-range
    /// entry is an out-of-bounds device write nothing else would catch).
    PatchRoutes,
    /// The tower's own position stream: `[Dim::Patches, 3]` `i32`, one
    /// `(t, h, w)` per patch row, each patch's position in its own image's
    /// grid. The time column is zero for image input, reserved for video.
    PatchPositions,
    /// Which row of the learned position table each patch reads:
    /// `[Dim::Patches]` `i32` on the native grid, `[Dim::Patches, taps]`
    /// when resampled. Read as a gather through `layout.embed`; `taps` is
    /// the interpolation width (1 native, 4 bilinear, 16 bicubic).
    PatchEmbedRows,
    /// How much of each tap — `[Dim::Patches, taps]` `f32`, read by
    /// `layout.embed_weighted`. `f32` because this is geometry, not the
    /// activation element. A text on the native grid never declares this.
    PatchEmbedWeights,
    /// A block-diffusion denoiser's self-conditioning signal, as the taps of
    /// a weighted gather over the embedding table: `[Dim::Tokens, taps]`
    /// `i32` token ids (the previous step's top-`taps` predictions per
    /// canvas row) beside [`SelfCondWeights`](RuntimeInput::SelfCondWeights).
    /// Read by `layout.embed_weighted`; zero weights are "no signal".
    SelfCondRows,
    /// How much of each tap: `[Dim::Tokens, taps]` `f32`, the previous
    /// step's probabilities of those ids. `f32` because it is a weight, not
    /// the activation element.
    SelfCondWeights,
    /// The row order that packs one selection's rows by attention group:
    /// `[Dim::Tokens]` `i32`, where packed row `i` (for `i` below the
    /// selection's row count) is fire row `perm[i]`, and every later entry
    /// is `-1`. Selected lanes are ordered `(group, stream, lane)` with each
    /// lane's rows contiguous, so a group's rows are one contiguous run —
    /// what the ragged attention kernel's CSR wants. Read by
    /// `layout.pack_rows` (gather) and `layout.unpack_rows` (its inverse
    /// scatter). The host builds it from the lanes' words
    /// ([`Selection::holds`]) and [`GeomKind::GroupOfLane`].
    RowPermutation { select: Selection },
    /// A float port on the token axis: `[Dim::Tokens, width]`, `f32` or
    /// `bf16` as the model text states at the reader. The latent rows of a
    /// denoise reading, fed device-to-device from a guest channel cell at
    /// every submit. `port` is the family's index for the port (its first
    /// or only latent port is 0), so a text with two latent streams of one
    /// width (video and audio) declares two.
    Latents { port: u8, width: u32 },
    /// A float port on the lane axis: `[Dim::Lanes, width]` `f32`, one row
    /// per lane — a timestep, a guidance scale, a per-modality sigma. `port`
    /// as for [`Latents`](RuntimeInput::Latents).
    LaneVector { port: u8, width: u32 },
    /// A context lane's rows: `[Dim::Tokens, width]` `bf16` — an encoder's
    /// output or reference tokens, channel-fed and constant across steps.
    /// Only the rows of lanes whose stream is `Context` carry data; the
    /// rest of the rectangle is unwritten. `port` as for
    /// [`Latents`](RuntimeInput::Latents).
    Context { port: u8, width: u32 },
    /// Per-axis rotary positions as the guest states them: `[Dim::Tokens,
    /// axes]` `f32`, `axes <= 4` — `(t, h, w[, l])` per token row, fractional
    /// where a text wants it. Read by `elementwise.rope_axes`. A third
    /// position stream beside `Positions` (`[Tokens]` i32) and
    /// `MropePositions` (`[Tokens, 3]` i32), which keep their names.
    AxisPositions { port: u8, axes: u8 },
    /// The voxel axis's lane table: `[Dim::Clips, 4]` `i32`, one row per
    /// clip in fire order, `{t, h, w, row_offset}` — the clip's box at the
    /// voxel PORT's resolution and the first row of its voxels in the
    /// `[Dim::Voxels, ·]` rectangle (clips contiguous, offsets a prefix sum
    /// of `t·h·w`). What every `spatial.*` kernel reads to find a row's
    /// lane and `(t, h, w)`; derived grids of later resolutions are
    /// computed on the device by `spatial.grid`. The host builds it from
    /// the clips each lane submitted.
    Grid,
    /// A float port on the voxel axis: `[Dim::Voxels, channels]`, `f32` or
    /// `bf16` as the reader states — a VAE's input tile (latents to decode,
    /// pixels to encode), one row per voxel of [`Grid`](RuntimeInput::Grid).
    /// `port` as for [`Latents`](RuntimeInput::Latents).
    Voxels { port: u8, channels: u32 },
    /// The clip table at TOKEN resolution: `[Dim::Clips, 4]` `i32`, one
    /// row per clip in fire order, `{t/pt, h/ph, w/pw, token_row_offset}`
    /// for the patch `p` the reader states — the token side of
    /// `spatial.patchify` / `spatial.unpatchify`. A clip's tokens are its
    /// lane's token rows, clips of one lane consecutive in clip order, so
    /// the offset is the lane's first token row plus the earlier clips'
    /// token counts; the host checks that a lane's token count is the sum
    /// of its clips' and refuses a clip whose box does not divide by `p`.
    TokenGrid { p: [u32; 3] },
}

/// Raggedness is not a `Ty` — a leading symbolic `Dim` means the value is
/// fire-aligned and viewable through the fire's shared indptr.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ty {
    Tensor { shape: Vec<Dim>, dtype: Dtype },
    /// Opaque, host-owned, outside the arena; sized at plan-build time.
    Struct(StructKind),
}

/// Where a value comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Def {
    /// Bound by the engine each fire.
    Input(RuntimeInput),
    /// Index into `Trace::params`. Weights are plain values — no `WeightId`;
    /// the compiler skips non-`Op` defs during allocation.
    Weight(u32),
    /// Index into `Trace::caches` — storage only; geometry arrives as `Input`.
    /// Distinct from `Weight` because caches are written during a fire.
    Cache(u32),
    /// Output of `Trace::nodes[i]`; the index is cross-checked by the validator.
    Op(u32),
    /// φ-node: data, never dispatched — the compiler resolves it to slot
    /// aliasing.
    Merge(Vec<(ValueId, Guard)>),
}

/// One row of `Trace::values`: provenance and type, orthogonal by construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueDecl {
    pub def: Def,
    pub ty: Ty,
}

#[cfg(test)]
mod tests {
    use super::{Dim, Guard, PerAxis, RowAxis, Selection};

    /// A split arm's guard — a conjunction of fact literals — reads as one
    /// mask/value pair that admits exactly the lanes the guard admits; an
    /// `Or` has no such pair and says so.
    #[test]
    fn a_selection_is_a_split_arms_guard_as_two_words() {
        let arm = Guard::and(Guard::Fact(3), Guard::not(Guard::Fact(5)));
        let select = Selection::of(&arm).expect("a conjunction of literals");
        assert_eq!(select, Selection { mask: 0b101000, value: 0b001000 });
        for word in 0..64u64 {
            assert_eq!(select.holds(word), arm.holds(word), "word {word:#b}");
        }
        assert_eq!(Selection::of(&Guard::Always), Some(Selection::ALL));
        assert!(Selection::ALL.holds(u64::MAX));
        assert_eq!(Selection::of(&Guard::or(Guard::Fact(0), Guard::Fact(1))), None);
        assert_eq!(
            Selection::of(&Guard::and(Guard::Fact(0), Guard::not(Guard::Fact(0)))),
            None,
            "a contradiction admits no lane and is not a selection"
        );
    }

    /// The index is the variant's own integer, both ways: every axis reads
    /// back what was filled at it, and `Dim::axis` lands each symbolic dim
    /// on the entry its row space owns.
    #[test]
    fn a_per_axis_reads_back_what_each_axis_was_filled_with() {
        let mut table = PerAxis::new(["tokens", "patches", "voxels"]);
        assert_eq!(table[RowAxis::Tokens], "tokens");
        assert_eq!(table[RowAxis::Patches], "patches");
        assert_eq!(table.as_slice().len(), RowAxis::COUNT);

        // The array's order is the enum's, so the index is direct.
        for (at, axis) in RowAxis::ALL.into_iter().enumerate() {
            assert_eq!(table.as_slice()[at], axis.name());
        }

        // Writing through the index is the same address.
        table[RowAxis::Patches] = "second";
        assert_eq!(table[RowAxis::Patches], "second");
        assert_eq!(table[RowAxis::Tokens], "tokens", "the other axis stood still");

        // `from_fn` is the same fill said once.
        let named = PerAxis::from_fn(RowAxis::name);
        assert_eq!(named[RowAxis::Tokens], "tokens");
        assert_eq!(named[RowAxis::Patches], "patches");
        assert_eq!(named[RowAxis::Voxels], "voxels");

        // What a cut indexes with: every symbolic dim's own axis.
        let cut = PerAxis::new([10u32, 20, 30]);
        for (dim, want) in [
            (Dim::Tokens, 10),
            (Dim::TokensTimes(2), 10),
            (Dim::Lanes, 10),
            (Dim::LanesPlus(1), 10),
            (Dim::Patches, 20),
            (Dim::Images, 20),
            (Dim::ImagesPlus(1), 20),
            (Dim::Voxels, 30),
            (Dim::VoxelsTimes(8), 30),
            (Dim::Clips, 30),
            (Dim::ClipsPlus(1), 30),
        ] {
            assert_eq!(cut[dim.axis().expect("a symbolic dim names a row space")], want);
        }
        assert_eq!(Dim::Const(8).axis(), None, "a fixed block belongs to no axis");
    }
}
