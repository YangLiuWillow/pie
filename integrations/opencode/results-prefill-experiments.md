# Prefill experiments — a running log

**Started 2026-08-16.** One experiment per section, appended as it finishes,
including the ones that fail. A negative result recorded is worth more than a
negative result repeated.

**Rule for every entry:** correctness against a float64 CPU reference BEFORE any
timing is quoted, the free parameter swept rather than assumed, and the A/B run
against the kernel actually being replaced.

## The state this starts from

Machine: Apple M5 Pro, 48 GB, idle (streaming roof 296.6 GB/s).
Model: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`.

Cold prefill of a 23,655-token prompt, `PIE_METAL_DISPATCH_TRACE` with
stride 8, six fires of ~4096 rows. Same prompt before and after the NAX landing,
so the two columns are comparable:

| | pre-NAX | now |
|---|---:|---:|
| traced wall | 66.77 s | **37.72 s** |
| attention | 69% | **47%** |
| routed MoE GEMM (`affine_qmm_t_routed`) | 19% | **35%** |
| dense projections (`affine_qmm_t`) | 9% | **16%** |
| everything else | 3% | 2% |

Attention per fire, pre-NAX (`sdpa_paged_mma`) against now (`sdpa_paged_nax`),
identical prompt — so this is the in-situ speedup, not the isolated one:

| fire | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| pre-NAX (ms) | 1379.7 | 4046.5 | 6849.5 | ~9600 | 12533.2 | 11730.2 |
| now (ms) | 439.8 | 1350.1 | 2398.2 | 3596.8 | 4922.3 | 4734.9 |
| ratio | 3.14× | 3.00× | 2.86× | ~2.7× | 2.55× | 2.48× |

**In situ it is 2.5–3.1×, against 3.35× measured in isolation, and the ratio
DECAYS with context.** That decay is the clue the first experiment chases: an
isolated probe is not memory-pressured and a real fire at 20k context is.

**Correction to a claim I made before measuring this:** I said the MoE GEMM was
now the largest term in a prefill fire. It is not — attention still is, at 47%.
MoE leads only in the early short-context fires (57.8% of fire 1 against
attention's 11.3%); attention overtakes it as the cache grows and is 63.8% of
fire 6. Both matter; the ordering depends on where in the prompt you look.

## Reference numbers to beat

Deterministic, fixed prompts, 200 generated tokens, one server per arm:

| prompt | pie TTFT now | mlx-lm TTFT | pie decode | mlx decode |
|---:|---:|---:|---:|---:|
| 5,840 | 6.36 s | 2.88 s | 53.7 tok/s | 66.3 |
| 16,090 | 15.75 s | 8.20 s | 40.3 | 47.6 |
| 28,390 | 30.94 s | 15.40 s | 27.9 | 35.9 |

Isolated kernel, 184 rows @ 7424 ctx: `sdpa_paged_nax` 2.08 ms/layer,
`sdpa_paged_mma` 6.97, MLX 1.555, simdgroup floor 4.1.

The memory roofline for the current attention tiling, worth having written down
because the first experiment is about exactly this:

* one threadgroup reads its KV head's whole K and V: 7424 × 128 × 2 B × 2 = 3.80 MB
* threadgroups = 32 q heads × ceil(184/64) tiles = 96
* so 365 MB per layer → **1.23 ms at 296 GB/s**
* compute is 22.4 GFLOP at NAX's 32.5 TFLOP/s → 0.69 ms

~~**Attention prefill is memory-bound, not compute-bound**, and the measured
2.08 ms sits 1.7x above its own memory roof. Every byte of that traffic is
redundant by a factor of 32.~~
**WITHDRAWN by experiment 1 below.** The arithmetic multiplies every
threadgroup's read by the threadgroup count, and that is wrong on this machine —
the query heads of a GQA group are adjacent in `tid.x`, run concurrently, and
hit in cache. Staging the block to remove the redundancy made the kernel 2.4x
SLOWER, which is not what removing real DRAM traffic does. The 32x is nominal.
Do not reuse the 1.23 ms figure; it is kept here only so the correction has
something to point at.

---
## Experiment 1 — stage K/V once per threadgroup, share across QH query heads

**Hypothesis.** Prefill attention is memory-bound. A K/V block is read 32× over
(4 simdgroups × 8 query heads per GQA group), the roofline for the current
tiling is 1.23 ms against 2.08 ms measured, and the in-situ speedup decays with
context. Staging the block in threadgroup memory once and sharing it should
recover most of that.

**Kernel.** `tools/rawmetal/kernels/sdpa_nax_stg.metal`. A threadgroup is QH
query heads × 4 row-tiles of simdgroups; K and V for a 32-key block staged in
16 KB of the 32 KB budget; registers per lane unchanged. QH swept, because the
constraint that capped the decode version at 2 — occupancy — does not bind here
(a 4096-row fire is 2048 threadgroups at QH=1).

**Result — REJECTED. Slower at every width.**

| | ms/layer | ×48 | vs unstaged 2.082 |
|---|---:|---:|---:|
| unstaged (shipped) | **2.082** | 99.9 | 1.00 |
| staged QH=1 | 5.007 | 240.3 | 0.42× |
| staged QH=2 | 3.284 | 157.6 | 0.63× |
| staged QH=4 | 2.810 | 134.9 | 0.74× |
| staged QH=8 | 2.290 | 109.9 | 0.91× |

All four correct (0 of 286720 wrong at 70 rows / 133 ctx).

**What it means, and it is worth more than the experiment cost.** Staging alone
costs 2.4× (QH=1 is 5.007 against 2.082 for the same work), and widening the
sharing recovers that cost monotonically — 5.007 → 3.284 → 2.810 → 2.290 —
without ever catching up. So the sharing *is* working, and it is buying back
something that was never being spent.

**The redundant device reads are already served by cache, not DRAM.** The
roofline in the section above multiplies every threadgroup's read by the number
of threadgroups, and that arithmetic is wrong for this machine: the eight query
heads of a GQA group are adjacent in `tid.x` and run concurrently on the same
cores, so they hit in cache rather than going to memory. The 32× amplification
is nominal, not actual.

**Therefore prefill attention is NOT memory-bound**, and the 1.23 ms figure in
the header of this file should not be used again. It is 10.75 TFLOP/s against
NAX's 32.5 — 3× off compute peak — with memory apparently not the limit, which
points the next experiment somewhere else entirely.

**Where that is.** Per 32-key block a lane does 16 MMAs and then a scalar
epilogue: 16 masked comparisons, 16 `exp2`, two row-reductions of four
`simd_shuffle_xor` each, and 64 accumulator rescales. On a unit that does the
MMAs in a handful of cycles, that epilogue is a plausible bottleneck — and it
runs once per 32 keys. **Widening the key block amortizes it**, which was
already experiment 2 for a different (and now discredited) reason.

## Experiment 2 — a wider key block, to amortize the softmax epilogue

**Hypothesis.** Experiment 1 ruled out memory. The remaining suspect was the
per-block scalar epilogue, run once per 32 keys only because `BK` was pinned to
the page size. Widening the block should make it run half or a quarter as often.

**Kernel.** `tools/rawmetal/kernels/sdpa_nax_bk.metal`, `NAXB_NPG` pages per
block. NPG=1 reproduces the shipped kernel, so it is the control and every ratio
is the epilogue and nothing else.

**Result — REJECTED, and badly. Monotonically worse, 8x worse at the widest.**

| NPG | keys/block | ms/layer | x48 | vs 2.082 |
|---:|---:|---:|---:|---:|
| 1 (control) | 32 | 2.146 | 103.0 | 0.97x |
| 2 | 64 | 2.967 | 142.4 | 0.70x |
| 4 | 128 | 8.073 | 387.5 | 0.26x |
| 8 | 256 | 16.241 | 779.6 | 0.13x |

All correct at all four widths, on three shapes each including one whose context
ends exactly on a page boundary so the last block reaches past the page list --
the guard that experiment was written around.

**The control is 2.146 against the shipped kernel's 2.082**, so restructuring
the loop to carry an array of page pointers costs 3% by itself. Worth knowing
before reusing that structure.

**What it means.** The cost is not the epilogue -- halving how often it runs
made things worse, not better. It scales with `S`, and `S` is `2*NPG` fragments
of 8 floats per lane:

| | NPG=1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| S (floats/lane) | 16 | 32 | 64 | 128 |
| P (bfloats/lane) | 16 | 32 | 64 | 128 |
| O (floats/lane) | 64 | 64 | 64 | 64 |
| approx total | 96 | 128 | 192 | **320** |

**This kernel is register-limited.** That is the one explanation consistent with
both experiments: not memory-bound (staging hurt), not epilogue-bound
(amortizing hurt), and degrading exactly as live registers grow. O alone is 64
floats per lane, which caps how many simdgroups stay resident per core, which
caps latency hiding -- and 10.75 TFLOP/s against a 32.5 peak is what poor
latency hiding looks like on a unit this fast.

**Where that points, and it is not another tile-size sweep.** Two things in this
kernel do work on every block that MLX does only where it is needed:

1. **`frag_load_rows` on every K and V load.** It carries a per-element
   `r < lim` test -- eight conditionals per fragment. MLX calls its equivalent
   only on the last key block (`if (!align_K && is_last_k)`) and uses the
   branchless `load` for all the others. At 7424 context that is ~232 blocks
   paying a tail check that only the last one needs.
2. **The causal mask on every block.** MLX applies it only from
   `kb_min_causal` onwards -- the blocks that straddle the diagonal. A prefill of
   184 rows at 7424 context has ~232 blocks of which ~6 straddle; the other ~226
   are entirely below the diagonal and need no mask at all. We run 16
   comparisons per lane per block on all of them.

Both remove work from the inner loop AND shorten live ranges. That is
experiment 3.

## Experiment 3 — pay for the mask and the tail only where they are owed

**Hypothesis.** From 1 and 2 by elimination: not memory, not the epilogue's
frequency, but the inner loop's instruction count and live ranges. Two things
ran on every key block that almost none of them owe — the per-element tail test
inside `frag_load_rows`, and the causal mask. MLX makes both conditional. Split
the key loop into a region below the diagonal (branchless loads, no comparisons)
and a region containing the straddle and the ragged tail.

**Kernel.** `tools/rawmetal/kernels/sdpa_nax_fast.metal`. `nax_block<MASKED>` is
a template, not a runtime flag — the point is that the fast region contains no
predicates, and a branch the compiler keeps is a branch not removed. `kb_safe`
comes from the smallest LIVE position in the simdgroup; dead rows of a partial
final tile take INT_MAX in that minimum, because taking their −1 would drag
every block into the masked region and the experiment would measure nothing.

**Result — ACCEPTED. 1.45×, and it passes MLX.**

| | ms/layer | ×48 | TFLOP/s |
|---|---:|---:|---:|
| `sdpa_paged_mma` (what shipped before NAX) | 6.97 | 334.6 | 3.2 |
| `sdpa_paged_nax` (current) | 2.082 | 99.9 | 10.75 |
| **experiment 3** | **1.435** | **68.9** | **15.5** |
| MLX, same shapes | 1.555 | 74.6 | 14.4 |

**pie's PAGED attention is now faster than MLX's CONTIGUOUS attention** on the
same shapes — 1.435 against 1.555 — and 4.86× the kernel that shipped two days
ago. Correct at five shapes including a partial final tile, a context ending on
a page boundary, and ctx=1.

Tile width re-swept, because removing the inner-loop work changed what the
kernel is limited by and the old optimum is not evidence about the new one:

| BQ | ms/layer | |
|---:|---:|---|
| 32 | 1.390 | |
| **64** | **1.435** | taken: within the probe's ~3% drift of BQ=32, and 128 threads — the same threadgroup as `sdpa_paged_mma`, which is what the driver already wires |
| 128 | 1.770 | loses |

**The lesson, which is the one this repo keeps relearning.** Two carefully
motivated experiments both failed, and both failed *informatively* — each ruled
out a class of explanation and the third followed from what was left. The
winning change removes work rather than rearranging it, and it was visible in
MLX's source the whole time (`align_K && is_last_k`, `kb >= kb_min_causal`). I
had read that file for the fragment layout and the mma contract and skipped the
predicates as bookkeeping. They were the optimization.

### Experiment 3, landed and measured end to end

`sdpa_paged_nax` updated in place; the ABI, the gating predicate and
`PIE_METAL_SDPA_NAX=0` are unchanged, so nothing outside the kernel moved.
Tests: llama_pso 32, llama_decode_step 232, kv_append_paged 13, numerics
51/18 pre-existing — all unchanged. Accuracy against the `sdpa_paged_mma` it
replaces is also unchanged (mean 1.23e-3 against 1.21e-3 at ctx 1024), which is
the expected result: the split loop computes the same thing, it just stops
asking questions it knows the answer to.

Deterministic TTFT, fixed prompts, one server per arm:

| prompt | before any NAX | after NAX | after exp 3 | mlx-lm |
|---:|---:|---:|---:|---:|
| 5,840 | 7.98 s | 6.36 | **6.03** | 2.88 |
| 16,090 | 27.70 s | 15.75 | **13.03** | 8.20 |
| 28,390 | 57.21 s | 30.94 | **21.80** | 15.40 |

Experiment 3 alone is worth 1.05× / 1.21× / 1.42×, growing with context — which
is the expected shape, since the fraction of blocks that are below the diagonal
and fully in-cache grows with the cache.

Cumulative against the kernel that shipped before any of this:
**1.32× / 2.13× / 2.62×.** The gap to mlx-lm on prefill is now **2.09× / 1.59× /
1.42×**, from 2.86× / 3.39× / 3.73× at the start.

Decode is untouched, as it should be: 53.2 / 40.0 / 27.8 tok/s against
54.9 / 33.5 / 28.9 with the switch off — same to within noise.

---
## Where prefill stands after experiment 3 — re-measured, not extrapolated

Same 23,655-token prompt, same trace settings, third time:

| traced wall | pre-NAX | after NAX | **after exp 3** |
|---|---:|---:|---:|
| | 66.77 s | 37.72 s | **29.98 s** |

Composition now (summed over all fires, the six prefill fires dominating):

| kernel | ms | share | was (after NAX) |
|---|---:|---:|---:|
| `affine_qmm_t_routed` (MoE GEMM) | 12898 | **43.6%** | 35% |
| `sdpa_paged_nax` (attention) | 9537 | **32.2%** | 47% |
| `affine_qmm_t` (dense projections) | 5774 | **19.5%** | 16% |
| everything else | ~1200 | 4.1% | 2% |

Attention fell from 17442 ms to 9537 ms in situ — **1.83×**, against 1.45× in
isolation. It does better in the real fire than on the bench, which is the
opposite of the usual direction and worth noting: the isolated probe runs one
context, and the win grows with context because the below-diagonal region does.

**The two quantized GEMMs are now 63% of prefill between them.** Neither has
been touched. Both are 4-bit affine group-64 matmuls on the simdgroup matrix
unit — the same unit, and the same ~5.5 TFLOP/s ceiling, that attention was
moved off. That is experiment 4.

---
## Experiment 4 — the routed MoE GEMM on the neural accelerators

**Hypothesis.** After experiment 3, `affine_qmm_t_routed` is the largest term in
a cold prefill (43.6%). It runs 6.6 TFLOP/s on the simdgroup matrix unit against
`matmul2d`'s 32.5, and it is not weight-bound at prefill widths -- 128 experts'
4-bit weights are ~302 MB per layer, 1.02 ms at 296 GB/s, against 44.8 ms
measured. So the unit is available to be changed, exactly as it was for
attention.

**Why the port is small.** The kernel already dequantizes 4-bit weights into
THREADGROUP MEMORY as bfloat and runs a bfloat matmul over the staged tiles. So
the loaders, the tiling, the dequantization, the two-fence K loop and the expert
slice are all untouched; only `mlx::steel::BlockMMA` becomes `NaxBlockMMA`
(8×8 simdgroup fragments → 16×16 NAX fragments, `tile_matmad` → `frag_mma`).
This is the same shape MLX's own `quantized_nax.h` takes.

`NaxBlockMMA` needs TN even, because `frag_mma` issues N=32 = two 16-wide
fragments. At WM=WN=2 that means BN ≥ 64 — satisfied by the tile a prefill
selects and not by the decode widths, which is the right place for the line
anyway: a matvec-shaped GEMM has nothing for a matrix unit of either kind.

**Result — ACCEPTED.** `tools/rawmetal/qmm_nax_probe.cpp`, both arms verified
against a float64 reference with distinct data per expert, per output column and
per input column, so a wrong expert slice or a transposed operand cannot pass.
Serving shape: a 4096-row fire routes 8 of 128 experts per token, so 32,768
sorted rows, N=768, K=2048.

| tile | shipped | NAX | |
|---|---:|---:|---:|
| bm=32, bn=64 | 15.56 ms (6.63 TFLOP/s) | **8.10 ms** (12.73) | **1.92×** |
| bm=64, bn=64 | 14.44 ms (7.14) | **6.52 ms** (15.81) | **2.21×** |

**Landing was one line, and safer than the attention one.** The NAX variant takes
the same tile, the same threadgroup and the same grid, so only the entrypoint
NAME differs — there is no launch site that could disagree with the choice.
`PIE_METAL_QMM_NAX=0` reverts it. Tests unchanged: llama_pso 32,
llama_decode_step 232, kv_append_paged 13, numerics 51/18 pre-existing.

**End to end, deterministic fixed prompts, TTFT:**

| prompt | `QMM_NAX=0` | `QMM_NAX=1` | |
|---:|---:|---:|---:|
| 5,840 | 6.03 s | **3.54 s** | 1.70× |
| 16,090 | 13.04 s | **9.05 s** | 1.44× |
| 28,390 | 22.11 s | **16.46 s** | 1.34× |

Decode unchanged (55.8 / 37.2 / 27.7 against 53.2 / 40.6 / 27.8 — noise), which
is the expected control: this kernel does not run in a decode step.

---

## Where prefill stands now

| prompt | at the start | **now** | mlx-lm | gap |
|---:|---:|---:|---:|---:|
| 5,840 | 7.98 s | **3.54 s** (2.25×) | 2.88 s | 1.23× |
| 16,090 | 27.70 s | **9.05 s** (3.06×) | 8.20 s | **1.10×** |
| 28,390 | 57.21 s | **16.46 s** (3.48×) | 15.40 s | **1.07×** |

**Prefill is now within 7–10% of mlx-lm at the context lengths an agentic turn
actually runs at**, from 2.9–3.7× behind two days ago. Generation is coherent
and the driver test suite is unchanged.

Remaining, in order:
1. `affine_qmm_t` — the DENSE projections, 19.5% and untouched. Same
   `NaxBlockMMA`, but the tile the driver picks is bm=64/bn=32, which gives
   TN=1 and does not satisfy `frag_mma`'s N=32. Needs either an N=16 descriptor
   or WN=1.
2. Attention is 32.2% and now beats MLX, so it is no longer the lever it was.
3. Decode is still ~1.2× behind mlx and untouched since head sharing.

## Experiment 5 — the dense projections on NAX

**The obstacle.** `NaxBlockMMA` applies unchanged except for the tile: the dense
path is bm=64/bn=32 at WM=WN=2, so TN=1, and `frag_mma` issues N=32 which needs
two 16-wide column fragments.

**The first fix was refused by the hardware API**, explicitly and usefully:

```
matmul2d_descriptor(16, 16, 16, ...)
  -> "At least one of M, N, or K must be 32 if both inputs are
      cooperative tensors"
```

**So the 32 went on K instead of N.** `frag_mma_k32` consumes two K-halves of A
and of B and produces one 16×16 accumulator. That suits the dense tile exactly —
BK is 32, so one instruction covers a whole K-block where the N=32 form takes
two. The other candidate, re-shaping the warps, was rejected on structure rather
than speed: the dense GEMM's threadgroup is dispatched as `{32, WM, WN}`, so
changing WM/WN moves a LAUNCH shape, and that is the class of change this repo
keeps getting wrong. The K=32 route keeps the launch identical, so landing stays
a matter of the entrypoint name.

**Result — ACCEPTED. 2.72×, the best rate yet.**

| shape | shipped | NAX | |
|---|---:|---:|---:|
| q_proj, M=4096 K=2048 N=4096, bm=64 bn=32 | 9.43 ms (7.29 TFLOP/s) | **3.46 ms** (19.84) | **2.72×** |

19.84 TFLOP/s is 61% of `matmul2d`'s 32.5 peak, against 22% for the kernel it
replaces. Correct on both arms against a float64 reference.

---

# Where this leaves prefill

**TTFT, deterministic fixed prompts, one server per arm:**

| prompt | at the start | **now** | mlx-lm | |
|---:|---:|---:|---:|---|
| 5,840 | 7.98 s | **2.60 s** | 2.88 s | **pie 1.11× faster** |
| 16,090 | 27.70 s | **7.38 s** | 8.20 s | **pie 1.11× faster** |
| 28,390 | 57.21 s | **14.31 s** | 15.40 s | **pie 1.08× faster** |

**Cumulative: 3.07× / 3.75× / 4.00×, and pie's prefill is now FASTER than
mlx-lm's at every context measured**, from 2.9–3.7× behind two days ago.

**The 6-turn canned agentic replay** (`bench_ab.py --turns 6`), both engines
measured in the same session because the standing rule forbids quoting a
remembered number for the other engine:

| | pie | mlx-lm |
|---|---:|---:|
| total | **7.11 s** | 7.64 s |
| turn 1, cold 7.2k prefill (TTFC) | **3.279 s** | 3.973 s |
| steady turn TTFC | 0.383–0.452 s | 0.430–0.442 s |
| decode | 46–48 tok/s | **56–62 tok/s** |

**pie is 1.07× faster overall and 1.21× faster on cold prefill; mlx is still
1.25× faster on decode.** This replay is prefill-dominated (10 generated tokens
a turn), so the total flatters pie — a turn that writes a long patch would
weight decode more heavily. Both halves are stated so neither is hidden.

Generation verified coherent after every landing: Fibonacci correct,
17×23 = 391, the first eight primes correct, a correct one-line description of a
KV cache. Driver suite unchanged throughout — llama_pso 32,
llama_decode_step 232, kv_append_paged 13, gptoss all, numerics 51/18
pre-existing.

## What is left

1. **Decode**, now the larger deficit: 1.25× behind mlx and untouched since GQA
   head sharing. Task #8 (splitting the key range so head sharing can go past
   QH=2) is the known lever and is unattempted.
2. `PIE_METAL_QMM_NAX=0` and `PIE_METAL_SDPA_NAX=0` revert the two GEMM and
   attention paths independently.
3. The NAX GEMM covers only the 4-bit group-64 tiles a prefill selects. Decode
   widths keep the simdgroup kernel, which is right — a matvec-shaped GEMM has
   nothing for a matrix unit of either kind — but it means `qmm_nax` does
   nothing for decode, by design and not by omission.

## Final check on the real agentic path

`tools/pie_ab.sh django__django-14373 3` — three reps per arm, because one run
cannot price anything here:

| rep | wall | turns | patch | server | TTFT | decode |
|---|---:|---:|---|---:|---:|---:|
| all-on 1 | 29 s | 5 | 412 B ✓ | 21.1 s | 8.1 | 13.0 |
| all-on 2 | 27 s | 5 | 412 B ✓ | 20.8 s | 8.2 | 12.7 |
| all-on 3 | 27 s | 5 | 412 B ✓ | 20.3 s | 8.2 | 12.1 |
| HSHARE=0 1 | 15 s | 3 | **0 B** | 7.7 s | 4.5 | 3.2 |
| HSHARE=0 2 | 31 s | 7 | 412 B ✓ | 23.9 s | 9.3 | 14.7 |
| HSHARE=0 3 | 64 s | 7 | 412 B ✓ | 56.3 s | 14.3 | 42.0 |

Five of six produce the correct 412-byte patch, and **the empty one is in the
CONTROL arm** — the same run-to-run nondeterminism documented in
`results-head-sharing-decode.md`, not a regression.

Against this instance's original baseline (`results-e2e-one-instance.md`), with
its caveat that totals carry large variance and per-call rates do not:

| | baseline pie | **now** | mlx-lm (2026-08-15) |
|---|---:|---:|---:|
| time in server | 42.6 s | **20.3–21.1 s** | 21.7 s |
| — TTFT | 24.9 s | **8.1–8.2 s** | 9.2 s |
| — decode | 17.7 s | **12.1–13.0 s** | 12.5 s |

**TTFT on the real agentic workload is 3.05× better and now below mlx-lm's**,
and total in-server time has gone from 1.96× mlx to roughly level. The mlx
column is from the earlier session and is NOT a same-session measurement, so it
is indicative here; the same-session comparisons are the fixed-prompt table and
the 6-turn replay above, and both agree.

## Composition re-traced after the GEMMs, and a four-way comparison

Both in `results-four-way.md`. The short version:

* traced wall on the same 23,655-token prompt: 66.77 → 37.72 → 29.98 →
  **17.57 s**, i.e. **3.80×** cumulative;
* in situ the GEMMs moved 3.17× (routed) and 2.90× (dense), better than the
  2.21× / 2.72× measured in isolation;
* **attention is the largest term again at 56.7%** — third rotation of the
  ordering, and the reason the rule is to re-trace rather than to plan off a
  share measured before the last change;
* against three other engines measured in the same session, pie now leads
  prefill (1.10–1.24× over mlx-lm, 1.7–2.7× over vLLM-metal) and trails decode
  (mlx-lm by 1.17–1.28×).
