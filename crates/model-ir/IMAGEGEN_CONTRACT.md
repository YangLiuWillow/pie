# The image/video-generation IR contract (M0-IR)

What `model-ir` / `model-dsl` state for the generative families (design D2, D3, D6, D7), and
what each shell is expected to provide per op. Shapes are `[rows, width]`; `Tokens`/`Lanes`
are `Dim::Tokens`/`Dim::Lanes`. Every op below has a `Dispatch` arm in `engine-cuda`,
`engine-metal`, `engine-vulkan`, `engine-wgpu` that refuses by name (`// M0: wired by …`);
CUDA agents replace the CUDA arms, the other shells stay refused this phase.

## 1. Lanes, streams, groups (D2)

- `model_ir::Stream { Text (default), Image, Video, Audio, Context, Reference }`,
  `Stream::code() -> u8` (0..6 in `ALL` order), `Stream::word(base) -> u64` (one-hot bit
  `base + code`). `Request::on_stream(Stream)` / `Request::stream()`,
  `Request::in_reading(u8)` / `Request::reading()` (0 = the family's default arm). A family
  packs the stream into its fact word itself; DSL guard: `Predicate::stream(base, stream)`.
- `Selection { mask: u32, value: u32 }` — a set of lanes: those whose word satisfies
  `word & mask == value` (`Selection::ALL` = every lane; `Selection::of(&Guard)` reads it off
  a split arm's guard, which is always a conjunction of fact literals). Every packing table
  below is keyed by one; the host evaluates it per lane with `Selection::holds(word)`.
- Packed order of a selection: selected lanes sorted by `(group, stream code, lane index)`,
  each lane's rows contiguous. Row indices in every table are **fire-absolute**; packed
  rectangles are indexed from row 0 (the `Layout::ScatterRows` precedent: `Run::fire_wide`).
- `GeomKind` (all in `Geometry { space: 0, .. }`, the token axis's own space, readable with
  no kv space declared): `GroupOfLane` `[Lanes] i32` (group id per lane, dense from 0, a
  request's lanes share one); `GroupIndptr { select }` `[LanesPlus(1)] i32` (per-group CSR
  over the selection's packed rows, groups present ascending); `LaneIndptr { select }`
  `[LanesPlus(1)] i32` (per-lane CSR over the same packed rows); `ReferenceTag { select }`
  `[Tokens] i32` (per packed row: the fire lane index if its lane is `Stream::Reference`,
  else -1).
- `RuntimeInput::RowPermutation { select }` `[Tokens] i32`: `perm[i]` = fire row of packed
  row `i` for `i < selected rows`, `-1` after.
- DSL readers on an `Input` arm (selection = the arm's guard): `group_of_lane()`,
  `group_indptr()`, `lane_indptr()`, `reference_tags()`, `row_permutation()`;
  `request_of_token()` no longer needs a kv space.
- `Layout::PackRows { x, perm } -> y` (`y[i] = x[perm[i]]`, `perm[i] >= 0`) and
  `Layout::UnpackRows { x, perm } -> y` (`y[perm[i]] = x[i]`); `y` is `x`'s exact type, fresh
  (no alias); rows not named are unwritten. DSL: `layout::pack_rows(x, perm)`,
  `layout::unpack_rows(x, perm)`. Kernel: `pack_rows(ctx, x, perm, &mut y)` /
  `unpack_rows(ctx, x, perm, &mut y)` over fire-wide rectangles; `seat::ENTRIES` row `Rows`.
- `Attention::Ragged { q, k, v, q_indptr, kv_indptr, head_dim, kv_heads, sm_scale,
  mask: RaggedMask } -> o`. Non-causal, no cache/plan/window, fp32 softmax and accumulate.
  Query segment `i` attends key segment `i` (`[indptr[i], indptr[i+1])` of each side's
  packed rectangle); segment counts agree. `q` `[Tokens, heads·head_dim]`, `k`/`v`
  `[Tokens, kv_heads·head_dim]` (`kv_heads | heads`, widths of the two rectangles unrelated),
  `o` = `q`'s type. `RaggedMask::None` (per-lane CSRs), `GroupBlockDiagonal` (per-group
  CSRs; same kernel, records intent), `ReferenceSelfOnly { q_tags, kv_tags }` (group CSRs plus
  the two `ReferenceTag` tables: a query with tag `t >= 0` sees only keys with tag `t`, a
  query with tag -1 sees its whole segment). A launch whose query selection is empty is a
  no-op. **`q` and `(k, v)` may come from different arms**: the recorder joins the operand
  guards with `Or` for this op alone (`record.rs::joins_arms`), the class sweep roots the
  node in every class it is live in (`classes.rs::spans_classes`) so each side's chain is
  demanded in its own class, and the DSL hands `o` back under `q`'s guard. DSL:
  `attn::ragged(q, k, v, q_indptr, kv_indptr, head_dim, sm_scale, mask)` (`kv_heads` read
  off `k`). Kernel: `attn_ragged::forward(ctx, q, k, v, q_indptr, kv_indptr, head_dim,
  kv_heads, sm_scale, mask, tags: Option<(Tensor, Tensor)>, &mut o)`, bf16 in/out, head_dim
  64/128/256 (the vendored FlashInfer ragged FA2 template); `seat::ENTRIES` `RowsAndLanes`
  and a rebind law like `attention.prefill`.
- `RaggedMask::RelativeBias { table, max_len }` (M3): the segment pairing plus an additive
  per-head bias on every logit, `s = q·k · sm_scale + table[h][clamp(kj − qi + max_len − 1)]`,
  `table` a `[heads, 2·max_len − 1]` f32 value (`Dim::Const` rows — a plan constant, handed
  whole; one row per QUERY head). The umT5 / T5 relative position bias and an ALiBi slope
  table both fit. The table is `Elementwise::RelativeBucketBias { embedding: [num_buckets,
  heads] weight (bf16/f32), max_len, num_buckets, max_distance, bidirectional } -> y`, HF's
  `_relative_position_bucket` transcribed in torch's f32 steps (statement on the op). DSL:
  `elemwise::relative_bucket_bias(inputs.recorder(), &weight, max_len, num_buckets,
  max_distance, bidirectional)`, `attn::relative_bias(&table, max_len) -> RaggedMask`.
  Kernels: `attn_ragged::RaggedMask::RelativeBias { table: Tensor, max_len }` (the FA2
  `RelativeBias` variant, `MaskMode::kNone`, bias added on the logits hook, `sm_scale_log2 =
  log2e`; a zero table is the plain arm bit for bit at a power-of-two `sm_scale`) and
  `elemwise::relative_bucket_bias(ctx, embedding, max_len, num_buckets, max_distance,
  bidirectional, &mut y)`, `seat::ENTRIES` `Reads::Nothing` (a constant launch).
- `Guard::narrow(outer, inner)`: a value read under a narrower guard that implies its
  producer's is spelled as `inner` (so a joint attention's answer splits back onto its arms).

## 2. Float ports and readouts (D3)

`port: u8` is the family's index for a port of that kind (0 = the first/only one).

- `RuntimeInput::Latents { port, width }` `[Tokens, width]`, dtype stated at the reader
  (`F32` or `Bf16`); `LaneVector { port, width }` `[Lanes, width] f32`;
  `Context { port, width }` `[Tokens, width] bf16` (only `Stream::Context` lanes' rows carry
  data); `AxisPositions { port, axes }` `[Tokens, axes] f32`, `1 <= axes <= 4` (named so
  because `Positions` — `[Tokens] i32` — already exists and keeps its name).
  DSL: `Input::latents(port, width, dtype)`, `lane_vector(port, width)`, `context(port,
  width)`, `axis_positions(port, axes)`. Shells: staged by the fire path (engine agent);
  today every shell's `Run::whole` panics by name.
- Seams `seam::VELOCITY = "velocity"` (`[rows, C·p^k]`) and `seam::HIDDEN = "hidden"`
  (`[rows, W]`, one per layer via `Seam::layer`); `seam::FLOAT_READOUTS`. `trace_hybrid`
  plants `out` on the returned value unless it is already under a float readout. Compiler:
  `EXPORT_SEAMS = [out, mtp, attn.scores, mtp.drafts, velocity, hidden]`,
  `FLOAT_READOUT_SEAMS`; both float seams live to plan end for their own classes.
  `engine-cuda::Exports::out` is `Option<ValueId>`; boot refuses only a plan with no export
  at all; readback of a float seam is the runtime agent's (today `Fault::Unbound`).

## 3. Modulation and activations (D6)

`m`/`g` are `f32` or `x`'s dtype (a lane-vector chain lands f32); a kernel dispatches on
both. `lane_of_row` is `request_of_token()` (`[Tokens] i32`) for a `[Lanes, ·]` vector, or
`None` for a `[Tokens, ·]` per-token vector. fp32 arithmetic, one rounding at the store.

- `Elementwise::Modulate { x, m, lane_of_row, form } -> y` (fresh, `x`'s type); `m` is
  `[rows, k·width]` laid out `[s | b]`. `ModulateForm::ScaleShift` `y = x·(1+s)+b`, k=2;
  `Scale` `y = x·(1+s)`, k=1; `TanhGate` `y = tanh(g)·x`, k=1 (`ModulateForm::slices()`).
  Kernel: `modulate(ctx, x, m, lane_of_row: Option<Tensor>, form, &mut y)`.
- `Elementwise::GatedResidualAdd { r, g, y, lane_of_row } -> r_out` (in place on `r`:
  `r += g·y`, `g` `[rows, width]`). Kernel: `gated_residual_add(ctx, g, y, lanes, &mut r)`.
- Fuse-only (`fuse::modulation`, not yet wired into the CUDA load chain):
  `NormModulate { x, norm: NormKind, normed, m, lane_of_row, form } -> (normed, y)` and
  `GatedResidualNormModulate { r, g, y, lane_of_row, r_out, norm, normed, m, form, out } ->
  (r_out, normed, out)`; `NormKind::Layernorm { eps }` (centred, no affine) or
  `Rmsnorm { head_dim, eps }`; every intermediate written as its own launch would write it.
- `Elementwise::Sinusoid { t, dim, max_period, flip_sin_cos, scale } -> y`: `t` `[rows, 1]
  f32`, `y` `[rows, dim] f32`, `half = dim/2`, `freq_i = exp(-ln(max_period)·i/half)`,
  `arg = scale·t·freq`, `y = [sin | cos]` (`[cos | sin]` under `flip_sin_cos`) — diffusers'
  `get_timestep_embedding` at `downscale_freq_shift = 0`. Kernel: `sinusoid(ctx, t, dim,
  max_period, flip, scale, &mut y)`.
- `Elementwise::Silu { x }`, `Gelu { x, tanh: bool }`, `Tanh { x }`: in place (`x_out`
  aliases `x`). `Mul { x, y } -> z`, `Add { x, y } -> z`: fresh, same type on all three.
  DSL: `elemwise::{modulate, gated_residual_add, sinusoid, silu, gelu, tanh, mul, add}`.
- **AN IN-PLACE FOLD IS THE LAST READ OF ITS OPERAND.** Every `aliases()` pair `(out, in)`
  is folded onto `in`'s rectangle unconditionally (`model_compiler::arena::fold_in_place`) —
  no copy is minted for an operand something else still reads. So a value read twice may be
  folded over at most once, and the second reader gets a copy: `elemwise::copy(v)` (`2v·½`,
  exact, one fresh rectangle). This is the shape adaLN keeps walking into — one timestep
  projection a fire, a `scale_shift_table` per block, `add_bias` folding the projection
  itself so block `n+1` reads the sum of every table before it (exact at one block, drifting
  with depth, invisible to a two-block miniature). `check` refuses it as `FoldThenRead`,
  guard-aware: only a reader whose lanes all lie inside the fold's is a fault, so two arms of
  one split folding one rectangle on disjoint rows still pass.

## 4. RoPE over guest positions (D7)

- `Elementwise::RopeAxes { x, positions, dims: [u32; 4], thetas: [f32; 4], form: RopeForm,
  rotary_dim, head_dim } -> x_out` (in place; call once for `q`, once for `k`). `x`
  `[rows, heads·head_dim]`, `positions` `[rows, axes] f32` where `axes` = leading non-zero
  `dims`; `dims[a]` = axis `a`'s CHANNEL count, even, `Σ dims == rotary_dim <= head_dim`
  (the tail of each head passes through). Pair `i` of axis `a` (block `b_a..b_a+dims[a]`)
  turns by `positions[a] · thetas[a]^(-2i/dims[a])`, angles fp32. `RopeForm::Interleaved`:
  pair `(b+2i, b+2i+1)` (FLUX, Z-Image); `Neox`: rotate_half over the rotated prefix, pair
  `(p, p + rotary_dim/2)`, axis by the block `p` falls in (MiniMax); `Split`: rotate_half
  within the block, `(b+i, b+dims[a]/2+i)` (Wan, LTX; `MropeForm::Split`'s pairing). DSL:
  `elemwise::rope_axes(x, positions, dims, thetas, form, rotary_dim, head_dim)`. Kernel:
  `rope_axes(ctx, positions, dims, thetas, form, rotary_dim, head_dim, &mut x)`, `Rows`.
  `RopeMrope` is untouched.

## 5. Where things live

`model-ir/src/{request,value,guard}.rs`, `ops/{attn,layout,elemwise}.rs`, `check.rs`
(port rows), `check/classes.rs` (`spans_classes`), `fuse.rs` (`modulation`);
`model-dsl/src/{forward,record,facts,lib}.rs`, `ops/{attn,layout,elemwise}.rs`;
`model-compiler/src/arena.rs` (`EXPORTS`). Tests: `model-dsl/tests/a_ragged_attention_joins_two_arms`,
`a_joint_attention_over_merged_streams_splits_back_onto_its_arms`,
`a_modulate_over_lanes_broadcasts_by_request_of_token`,
`a_forward_may_return_its_velocity_instead_of_logits`, `a_row_permutation_keeps_the_row_space`,
`the_axis_rope_states_its_axes_once`; `model-compiler/tests/a_ragged_attention_over_two_arms_bakes_into_one_region`;
`model-ir` unit tests in `fuse.rs`, `value.rs`, `request.rs`.

## 6. The voxel axis and the `Spatial` family (D8)

A VAE runs inside the model layer on a THIRD row axis. `RowAxis::Voxels`
(`RowAxis::ALL = [Tokens, Patches, Voxels]`, `COUNT = 3`, `PerAxis` three
wide); dims `Dim::Voxels` (the fire's PORT voxel count — `Σ t·h·w` over the
clips the lanes submitted), `Dim::VoxelsTimes(k)` (a rectangle an upsample /
shuffle / unpatchify grew by a fixed factor `k`; an op that shrinks rows keeps
its input's dim and over-allocates — the grid says which rows are live),
`Dim::Clips` (the axis's lane space: images or videos), `Dim::ClipsPlus(k)`.
`VoxelsPlus(k)` was not added: no table on this axis is indptr-shaped; the
clip table below carries the offsets.

- **Layout.** An activation is `[rows, channels]`, one row per voxel in
  `(t, h, w)` order (`w` fastest), one clip's voxels contiguous. The per-clip
  box is `RuntimeInput::Grid` `[Clips, 4] i32 = {t, h, w, row_offset}`,
  host-built in fire order with prefix-summed offsets. Every later
  resolution's grid is a VALUE computed on the device by `Spatial::Grid { grid,
  rule: GridRule, y }` (one single-block launch, `kernels/spatial/rule.cuh`) —
  chosen over a prepare-phase host node because a device rule is capturable
  and stateless, while a host derivation would need a per-fire host copy of
  every intermediate grid. `GridRule::{Conv{k,stride,pad,causal_t},
  Upsample{factor,keep_first_frame}, Shuffle{r,trim_t}, Unshuffle{r}}`;
  `GridRule::out_extent`/`apply`/`growth` are the host twins. `trim_t` is a
  causal temporal upsampler's ANCHOR DROP: a shuffle lands `t·r1 - trim_t`
  frames, LTX-2.5's `LTXVideoUpsampler3d` dropping `r1 - 1` after it expands
  one latent frame into `r1` sample frames (study §I.9). It is the same
  flavour of statement as `Upsample::keep_first_frame` — a time rule the box
  carries and the rows follow — and `growth()` is unchanged by it, so the
  output dim keeps the untrimmed `VoxelsTimes(r1·r2·r3)` and over-allocates
  the way every shrinking rule on this axis does. A box the trim empties
  maps to no rows.
- **Ports and readouts.** `RuntimeInput::Voxels { port, channels }`
  `[Voxels, channels]` f32/bf16 (DSL `Input::voxels(port, channels, dtype)`);
  a plan may read voxel ports of SEVERAL widths on several arms (Z-Image's
  `vae.decode` reads 16 channels, its `vae.encode` 3): the CUDA shell reserves
  the payload at the widest (`voxels::Seat.channels`) and reads a fire's
  width off its payload (`Seat.widths`, `Tables.channels`, M0: one width a
  fire), a `RuntimeInput::Voxels` bound at another width panics by name.
  `models::PortKind::Voxels` / WIT `port-kind.voxels` name the port to a
  guest: its channel is `[h, w, C]` (a still) or `[t, h, w, C]` (a clip), the
  shape being the clip's box (`runtime::validate_port_channel`); the readout
  is `models::ReadoutKind::Pixels` / WIT `readout-kind.pixels`. Every
  planting of `seam::PIXELS` is its own export (`Exports.pixels`, one per arm
  with its writer classes) and a fire reads back the planting of the class
  its clips ran in (`Exports::pixels_for`);
  `Input::grid()`; `RuntimeInput::TokenGrid { p }` `[Clips, 4] i32`
  `{t/pt, h/ph, w/pw, token_row_offset}` (`Input::token_grid(p)`), the token
  side of the patchify pair — a clip's tokens are its lane's token rows, clips
  of one lane consecutive; the shell checks a lane's token count is the sum
  of its clips'. `seam::PIXELS = "pixels"` is a float readout (`FLOAT_READOUTS`
  now `[velocity, hidden, pixels]`, compiler `EXPORT_SEAMS[6]`), planted on
  TWO values — `seam::at(seam::PIXELS, &[&y, &y_grid])` — so the reader
  slices the plane per clip through the output grid.
- **Ops** (`model-ir/src/ops/spatial.rs`, DSL `ops::spatial`): `Conv3d { x,
  grid, w, bias?, k, stride, pad, pad_back, causal_t, time_pad: TimePad,
  cache?, y_grid, y }` (bf16 in, fp32 accumulate, one rounding; `w` `[C_out,
  taps·C_in]` tap-major channel-fastest; `pad` is the zero padding IN FRONT
  of each axis and `pad_back` BEHIND it — equal for a symmetric convolution,
  `GridRule::Conv` carries both, and only the front pad shifts the kernel's
  tap window, the back pad reaching it through the output box alone; DSL
  `Conv::conv2d([3, 3], [2, 2], [0, 0]).pad_back([0, 1, 1])` is diffusers'
  `Downsample2D`, `F.pad(x, (0, 1, 0, 1))` then a stride-2 3×3);
  `GroupNorm { x, grid, groups, weight, bias, eps, silu, y }` (fp32 Welford
  per clip per group); `Attention { q, k, v, grid, segment, sm_scale, y }` —
  the conv VAE's mid-block attention, ONE head as wide as the row, over the
  block `segment` names, `q`/`k`/`v`/`y` all `[rows, C]` bf16 at one type
  (not `attention.ragged`: that kernel is stamped at head widths 64/128/256
  over token-axis CSRs, and a VAE's head is its whole channel row); fp32
  scores, online softmax and accumulation, one rounding at the store; kernel
  `spatial::attention(ctx, q, k, v, grid, segment, sm_scale, &mut y)`
  (`kernels/spatial/attn.cuh`, `C ∈ {256, 512, 640, 1024}` with the queries
  per warp stamped against it — 4 / 2 / 1 / 1, holding `QPW·C/32` fp32 of
  query and of accumulator at 32 registers or fewer, which is what keeps the
  1024-wide head off the local-memory spill path — an online-softmax walk
  over the block's keys, no flash tiling: a VAE attends at its lowest
  resolution. A lane moves its `C/32` channels as 16-byte words where that
  divides by 8 and as bf16 SCALARS where it does not, which today is 640
  alone — Wan 2.2's ENCODER mid block, whose 20-channel slice is neither
  whole words nor aligned); DSL
  `spatial::attention(q, k, v, grid, sm_scale)` and
  `spatial::attention_over(q, k, v, grid, segment, sm_scale)`.
  **THE VOXEL AXIS'S SEGMENT TABLE** is `VoxelSegment::{Clip, Frames(n)}`,
  what `GroupIndptr`/`LaneIndptr` (§1) are to the token axis: `Clip` is one
  block per clip (FLUX's, Z-Image's and FLUX.2's mid blocks), `Frames(n)` one
  block per run of `n` consecutive frames, short at the end when `t` does not
  divide — `Frames(1)` is Wan 2.2's mid block, which attends each frame on
  its own at `C = 1024`. It is an enum and not a CSR because the segment
  count on this axis is `Σ t` over the clips, which no budget states and no
  dim spells, while the blocks a VAE attends over are REGULAR and so are
  already described by the `[Clips, 4]` grid every voxel rectangle travels
  with; the enum says how to read that table. `VoxelSegment::bounds(box,
  row)` is the host twin, `spatial/grid.cuh::segment_of` the device one, and
  `Frames(0)` is refused by name at the DSL and at the kernel;
  `UpsampleNearest { x, grid, factor, keep_first_frame, y_grid, y }`;
  `PixelShuffle { x, grid, r, trim_t, y_grid, y }` /
  `PixelUnshuffle { x, grid, r, y_grid, y }` (einops
  `'(c r1 r2 r3) t h w -> c (t r1) (h r2) (w r3)'`);
  `AvgDown { x, grid, factor, group, y_grid, y }` — `AvgDown3D`, Wan 2.2's
  encoder residual shortcut: the TIME axis zero-padded IN FRONT to a multiple
  of `factor[0]` (`pad_t = (ft - t % ft) % ft`, which is why an `Unshuffle`,
  demanding a box that divides, cannot stand in), then the same channel-major
  space-to-depth, then the MEAN of each `group` consecutive widened channels
  — a reduction over the WIDTH, which on this axis is the channels, and never
  over the grid. `[rows, C] → [rows', C·ft·fh·fw/group]` under
  `GridRule::AvgDown { factor }` (`ceil(t/ft), h/fh, w/fw`); `group = fh·fw`
  is a spatial average pool that keeps the time block as extra channels
  (Wan's every case), `group = ft·fh·fw` the plain pool over the block; fp32
  accumulation, one rounding;
  `CacheStore { x, grid, frames, cache, x_out }` — the store half of a causal
  conv's frame cache with no convolution around it, `x_out` ALIASING `x`
  (the one aliasing member here). Wan 2.2's encoder head is the caller: its
  `downsample3d` resampler does not run its time convolution on the first
  chunk at all, it only remembers the frames the next chunk pads with, and
  that convolution over a one-frame box has no output box to hang a `Conv3d`
  on. `check::classes::writes_cache` roots it;
  `Patchify { x, grid, p,
  tgrid, y }` → `[Tokens, C·p³]`; `Unpatchify { x, tgrid, p, grid, y }` →
  `[Voxels, C]`. Every other member lands a fresh rectangle (a conv reads its
  neighbours). DSL: `spatial::conv3d(x, grid, w, bias, Conv, cache) -> (y,
  y_grid)`, `group_norm(..) -> y`, `upsample_nearest(..) -> (y, y_grid)`,
  `pixel_shuffle/pixel_shuffle_trimming/pixel_unshuffle(..) -> (y, y_grid)`,
  `avg_down(x, grid, factor, group) -> (y, y_grid)`,
  `store_frames(x, grid, cache, frames) -> x`,
  `patchify(x, grid, p,
  tgrid) -> y`, `unpatchify(x, tgrid, p, grid) -> y`, `Conv::{conv2d, conv3d,
  same3, causal(TimePad)}`. Shape rules: conv/norm keep rows; upsample and
  shuffle grow `Voxels → VoxelsTimes(vol)`; unshuffle divides a carried
  factor out or keeps the dim; avg-down keeps the dim and over-allocates.
- **Conv weights.** Declared as the checkpoint stores them (`[C_out,
  C_in·kt·kh·kw]`, `weight.reshape(C_out, -1)`) with
  `Weight::conv_taps_major(c_in, taps)`, interned as
  `Param { layout: ParamLayout::ConvTapsMajor { c_in, taps } }`; the CUDA
  shell relabels each such plane once at load (`voxels::relabel_conv_weights`
  → `kernels_cuda::spatial::conv_weight_taps_major`). `spatial::conv3d`
  refuses a weight declared natural.
- **Causal time and the frame cache.** `Conv::same3().causal(TimePad::Zero)`
  pads `kt-1` frames in front only. With `cache: Some(Input::state(name))` —
  a `CacheRow::State` slab the text declares per causal conv, `[front·
  max_plane, C_in]` per slot, `front` being the conv's causal FRONT PAD
  (`kt-1` for a `same`-padded one, and 1 for Wan's encoder `downsample3d`
  time conv, which pads nothing of its own and is handed one cached frame) — the CUDA arm gathers each clip's slot into a
  `[Σ frames·h·w, C_in]` scratch rectangle (`spatial::cache_gather`,
  `kernels/spatial/cache.cuh`), convolves, and stores this tile's last
  `front` input frames back (`spatial::cache_store`; the arm reads `front`
  off `pad[0]`), keyed by the fire's `[Clips]` slot table. `Shell::open(slot)` (the `RsReset` path) zeroes every state row
  of the slot, which is the zero-padded first tile; `TimePad::Replicate` is
  for the cacheless single-tile case. WITHOUT `causal_t`, `TimePad::Replicate`
  (DSL `Conv::same3().replicate_time()`) pads BOTH ends of the clip with its
  own end frames — LTX-2.5's non-causal decoder, which decodes a whole clip
  in one fire and carries no cache; `TimePad::Zero` there is the plain
  zero-padded symmetric convolution. `check::classes::writes_cache` roots a
  conv with a cache.
- **Compiler.** `Budgets.voxels: Option<VoxelLadder { max_voxels, buckets,
  max_clips }>` (`Budgets::with_voxels`, `ladder(RowAxis::Voxels)`,
  `max_voxels()`, `max_clips()`); `Error::Unsized { axis: Voxels }` for a
  voxel plan against no ladder; `RowExpr::{Voxels, VoxelsTimes(k), Clips,
  ClipsPlus(k)}`, `FireRows { voxels, clips }`; `CompiledModel.voxels:
  Option<AxisPlan>`; `FamilyCosts.spatial` (40 µs, GEMM-class). Units: a
  voxel-axis region is its own capture unit, ordered by first appearance like
  every unit (`unit::partition`); the patchify pair is placed on the unit of
  the axis it WRITES (outputs decide), so an encoder is `[Voxels, Tokens]` and
  a decoder from tokens `[Tokens, Voxels]`; a voxel-only plan is `[Voxels]`.
- **Exec.** `Lane::with_clips(word, rows, clips, voxels)`, `LaneRow
  { voxel_offset, voxels, clip_offset, clips }`, `Composition::{voxel_rows,
  clips, voxel_classes, voxel_bucket}`; faults `TooManyVoxels`, `TooManyClips`,
  `NoVoxelBucket`, `Vaeless`, `NoVoxelLadder`, `ClipGeometry`,
  `DescriptorVoxelRows`. **Descriptor ABI 3**: a 40-byte header (`voxel_rows`
  at word 8, `voxel_bucket` at word 9) and a voxel trailer (one `[row_offset,
  rows, lane_offset, lanes]` window per class, one `[voxel_offset, voxels,
  clip_offset, clips]` record per lane) present iff `voxel_rows > 0`; ABI 1
  and 2 bytes are refused by name. `model_exec::DispatchSpatial` is the
  seventh dispatch trait; Metal/Vulkan/wgpu refuse every member by name.
- **CUDA shell.** `Boot.voxels: Option<VoxelLadder>` (`api::voxel_ladder`
  derives one from `LoadBudgets.max_voxels/max_clips`, default 65 536 port
  voxels and `max_lanes` clips); `engine_cuda::voxels::{Seat, Store, Tables,
  Clips, Handles}` stage the grid, token grid, clip slots and payload below
  the fire's inputs; `FireBindings.{grid, token_grid, voxels, clip_slots}`;
  `dispatch/spatial.rs`. **M0 rules:** every spatial kernel takes the whole
  clip table and finds a row's lane itself (`seat::Reads::Nothing`), so a
  voxel launch runs over the fire's whole voxel rectangle and the shell
  refuses a fire whose clips fall in two classes (`Fault::VoxelPayload`).
  **Arming is PER AXIS** (M1): `Windows::admit_axes` marks every region on
  `RowAxis::Voxels` an `Admit::Island` — the arming pass fires no clip, so a
  voxel window it sees has zero rows and would read as capturable, and a
  spatial launch reads no seat that could retire a replay's padding — while
  the TOKEN regions of the same plan are judged as ever, so a flagship
  serves its DiT bodied with a VAE standing beside it in the artifact.
  `Shell::fire_voxels(lanes, clips) -> Vec<Pixels>` is the door;
  `Engine::submit` takes `Step.voxels: Vec<StepVoxels { lane, clips,
  payload }>` and answers `LaneReadout { seam: ReadoutSeam::Pixels, clips,
  values }`. **A voxel port has TWO feeds and a lane takes one**: the
  `payload` beside its clips (host-fed, the shell's own door), or
  `PortKind::Voxels` in `Lane::ports` (channel-fed) — the clips then carry
  the box alone, and `enqueue` copies the committed cell into the voxel
  payload at the lane's `voxel_offset`, device to device, casting an f32
  cell into a bf16 port the way `Latents` does. A lane doing neither is
  refused by name.
  Tests: `model-dsl/tests/a_conv_decoder_traces_on_the_voxel_axis`,
  `model-compiler/tests/the_third_row_axis_carves_its_own_arena`,
  `model-exec/tests/the_voxel_axis_seriates_its_own_clips`,
  `engine-cuda/tests/a_conv_decoder_fires_over_a_voxel_port` (GPU),
  `engine-cuda/tests/a_channel_fed_voxel_port_lands_the_committed_cell` (GPU),
  `engine-cuda/tests/a_two_axis_plan_arms_its_token_bodies` (GPU),
  `kernels-cuda/tests/the_spatial_attention_answers_the_cpu_reference` (GPU,
  256/512/1024 channels),
  `kernels-cuda/tests/the_spatial_attention_segments_by_frame` (GPU),
  `kernels-cuda/tests/a_trimmed_pixel_shuffle_drops_its_anchor_frames` (GPU).
- **The first real VAE (M1).** `models::z_image::vae` states the FLUX
  16-channel `AutoencoderKL` as the flagship's `vae.decode` (latent `[h·w, 16]`
  → pixels `[8h·8w, 3]` in `[-1, 1]`, the `z/scaling + shift` denormalise
  inside the plan) and `vae.encode` (pixels → the posterior MEAN `[h·w, 16]`,
  raw; the guest applies `(mean − shift)·scaling`) readings, both on the
  `pixels` seam; `models/tests/the_z_image_vae_bakes`, and the GPU parity gate
  `engine-cuda/tests/the_z_image_vae_answers_the_reference` against
  `scripts/imagegen/zimage_golden.py --vae` (decode cos 0.99998, mean |err|
  0.0023; encode at the bf16 reference's own distance, cos 0.9997).
  **FROM A GUEST** the same reading is `tests/inferlets/zimage-vae-parity`
  driven by `scripts/imagegen/zimage_vae_parity.py`: the latent bound as the
  port's channel (whose declared shape IS the clip's box, `[h, w, C]` or
  `[t, h, w, C]`, which is what `runtime::validate_port_channel` accepts),
  the answer read off `intrinsics::pixels(rows, 3)`, and the picture out
  through `frames.from-channel` + `session.send-frames` — measured at cos
  0.999981, mean |err| 0.00225, the host-fed gate's own distance.
  `layout.split_rows` launches its rows on `grid.x` now (`grid.y` is capped
  at 65 535 by every compute capability, which bit a 64k-row fire and a wide
  voxel rectangle alike).

## 7. How the CUDA engine serves §1–§2 (M0 round 2)

What `engine-cuda` (with `model-exec`) does with the tables and ports above — the refinements
a runtime or another shell must agree with. Tests: `engine-cuda/tests/a_double_block_fires_two_streams_through_the_engine`,
`a_second_fire_reads_the_port_cells_it_was_handed`, `a_missing_or_misshapen_port_feed_is_refused_by_name`,
`a_dit_plan_loads_at_a_sixty_four_k_row_ceiling`; `model-exec/src/fire/packing.rs` unit tests.

- **Lanes.** `Lane.stream` / `Lane.group` / `Lane.ports` reach the shell as stated
  (`serve::Seated { stream, group, ports }`); the word is the runtime's (`Model::word(.., stream,
  reading)`), never re-derived. `Lane.group == None` is a group of its own. A token-less lane
  (`tokens = vec![0; rows]`, `kv: KvDelta::default()`) is accepted: a plan with no kv space seats
  no page, makes no pool demand, and its lanes' rows are bounded by `Budget::max_tokens` alone.
- **Groups are fire-global** (`model_exec::fire::packing`). `GroupOfLane` numbers groups densely
  from 0 in order of first appearance in fire-lane order. Every `GroupIndptr { select }` is indexed
  by that id: a group none of the selection's lanes belong to is an EMPTY segment (the bound
  repeats), so the two sides of a cross-attention pair segment `g` with segment `g` even when one
  side has no lane in some group. Tables are staged `[lane ceiling + 1]` (padding repeats the last
  bound) and handed to `attention.ragged` whole; the kernel reads no seat word.
- **The packed rectangle stands at the selection's window.** Packed row `j` of selection `S` is
  fire row `origin_S + j`, `origin_S` = the first fire row of `S`'s lanes (0 for `Selection::ALL`,
  the joint attention). `RowPermutation`/`ReferenceTag` are fire-wide `[Tokens]` tables, `-1`
  outside `[origin_S, origin_S + rows_S)`; CSR bounds are absolute rows of that rectangle. A
  selection whose lanes' rows are not one contiguous run of the fire (classes seriated apart) is
  refused (`model_exec::fire::Fault::ScatteredSelection`) rather than packed over another class's
  rows.
- **`ReferenceSelfOnly`** is served in the contract's tag form: the two `ReferenceTag` tables,
  fire-wide and indexed by the packed rows the CSRs name, reach the kernel whole
  (`kernels_cuda::attn_ragged::RaggedMask::ReferenceTags { q_tags, kv_tags }`, a
  `REGISTER_LOGITS_MASK` variant), so a group may hold ANY number of reference lanes, each
  attending itself alone while every other row of the group sees them all (FLUX.2's KV layout,
  HunyuanImage 3). The kernel's one-tail form (`ReferenceSelfOnly { ref_start }`) stays as a fast
  case no engine arm uses.
- **Ports.** Every `(kind, port)` a lane's CLASS reads must be fed (`Lane::ports`) from a channel the
  instance ATTACHED to that lane carries (the `SelfCondInput::channels` precedent): the feed reads
  the channel's committed cell at the consumer head — what the instance's own `take` would read
  this fire — resolved at enqueue after the prologue. A missing feed, a feed for an undeclared
  port, or a lane with no attachment is refused by name at submit; a synthetic (arming) fire feeds
  nothing. Cell shape: `Latents`/`Context` `rows × width` in the port's dtype, `AxisPositions`
  `rows × axes` f32, `LaneVector` `1 × width` f32 (one cell per lane). An f32 cell into a bf16
  port is cast on the way (`linear.quant_cast_fp32_to`); any other mismatch is refused naming the
  port, the lane, the cell bytes and the wanted bytes. The rectangles live in the inputs store at
  `[max_tokens, width]` (`[max_lanes, width]` for lane vectors); rows a lane does not feed keep
  the last fire's bytes.
- **Readback.** The readout seam is `out` (logits, bf16) when the plan has one, else `velocity`,
  else the last `hidden` — `LaneReadout { seam, width, values }` through `settle_frame`, bf16 or
  f32 planes widened to f32, rows per `Readout::Rows`. `ModelProfile { has_velocity, velocity_width }`
  is read off the `velocity` seam's width; `vocab` is 0 for a plan with no `out`. An epilogue
  attachment gets `IntrinsicId::Velocity` (the velocity plane) and `IntrinsicId::Hidden` (the last
  hidden plane) bound at the lane's first row, `width = plane.width`, storage raw-bf16 or f32 as
  the arena holds it; `Logits` is bound only when the readout seam is logits.
  `IntrinsicId::Pixels` (D8, `[rows, C]` f32, gated by `ModelProfile { has_pixels, pixels_width }`
  — `pixels_width` is `0` when a plan's plantings disagree, a VAE's decode RGB beside its
  encode's 16-channel mean, and bind then checks rank and rows alone) is bound at the lane's
  first OUTPUT VOXEL, not its token row: a VAE lane's rows are on the third axis. Every grid past
  the port's is a device value, so the offset comes from `voxels::host_grid`, which replays the
  plan's `Spatial::Grid` chain through `GridRule::apply`, the rules' own host twins. A program
  reading an unbound `pixels` is refused at its mint by name.
- **Lane-shaped values** (`[Lanes, ·]`: a lane vector's chain) are carved and computed at the
  fire's lane carve (the key's lane ceiling for a body) and launched without the staged seat
  (`Run::unseated`); an f32 lane activation's `linear.matmul` takes `linear::lane_gemm`.
  `GeomKind::RequestOfToken` is staged by every fire (`[carve rows]`, lane 0 past the live rows).
- **Fusion.** `fuse::modulation` runs in the CUDA load chain; the fused arm launches the fused
  kernel only for `ScaleShift` over a whole-row norm (`Layernorm`, or `Rmsnorm` with `head_dim ==
  width`) and lands the traced pair otherwise; `normed` is written on its own only when some node
  reads it.
- **Bodies.** A new arming kind, `joint`, arms every present set a multi-class region spans (the
  MM-DiT fire: text and image classes together), one body per stated bucket, golden-checked
  against its eager walk like the rest. State `buckets` for a large ceiling: the default lattice
  is every power of two up to `max_tokens`, and each rung fires synthetics of that many rows at
  load. Measured on the miniature: `max_tokens = 65536, buckets = [8192, 32768]` loads in 1.2 s
  (arena 56 MiB, inputs 10 MiB — no mask slab is carved for a plan with no `attention.masked`
  arm; at a real context that slab and its nine pinned mirrors would be gigabytes).
- **Known limits.** Split-form and erf-gelu: `gelu(tanh = false)` is refused by name. A lane-shaped
  launch in a captured region whose window does not begin at fire row 0 is not exercised. The
  `layout.pack_rows` window must equal the selection's rows (a text reads `row_permutation()` on
  the arm it packs, or on the root for a merge).
