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

## 6. How the CUDA engine serves §1–§2 (M0 round 2)

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
- **`ReferenceSelfOnly`** is served with the kernel's one-tail-per-group mask (reference lanes pack
  last in their group; `packing::Packed::reference_start`). A group with two reference lanes is
  refused by name at submit.
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
