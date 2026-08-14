# Plan: is "de-page, then compute" faster than paged attention?

**Status:** STEPS 1 AND 2 ARE ANSWERED and the investigation is CLOSED. De-paging
was not worth it; the gap is the matrix instruction, not the kernel. STEP 3 is a
neural-accelerator attention kernel. One change landed along the way.
**Branch:** `liu/opencode-integration`. **Machine:** Apple M5 Pro, 48 GB.
**Model:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` (48 layers, 32 q
heads, 4 kv heads, head_dim 128, page 32).

---

## The verdict, in one table

All at 184 rows @ 7424 ctx, the serving prefill fire. Every row measured by
`driver/metal/tools/rawmetal/sdpa_paged_probe.cpp`, reproduced across 4 runs.

| configuration | ms/layer | vs MLX | status |
|---|---:|---:|---|
| shipped paged MMA (before) | 7.53 | 4.84x | was the baseline |
| **+ `_p32` shifted addressing** | **6.92** | 4.45x | **LANDED**, bit-identical output |
| + full de-page into scratch | 6.17 | 3.97x | **NOT DONE** — see below |
| MLX `fast.scaled_dot_product_attention` | 1.555 | 1.00x | the target |

**De-paging works and is not worth doing.** Removing the page walk entirely is
worth 1.37 ms/layer; the gather that buys it costs 0.107; net +1.26 ms/layer, a
real 17% win. But it closes only **21% of the 4.8x gap**, and it costs a scratch
buffer, a gather pass, and the plumbing to keep them in step. Roughly half of it
(0.62 ms) turned out to need no de-paging at all, and that half is now landed.
The remaining half is 0.75 ms/layer for a structural change — the worst
ratio of effort to gap-closure on the table.

**The 4.6 ms that actually matters is the kernel's shape**, and it is
untouched by any of this. That is STEP 2b, and it is now the only step.

---

## What was landed, and how it is verified

`sdpa_paged_mma` gained a `PAGE_SIZE` template parameter and a `_p32`
instantiation, selected on **exact equality** with 32 in
`model/llama/kernels.cpp` and `model/gptoss/kernels.cpp`. Inside, `kp /
page_size` and `kp % page_size` become `kp >> 5` and `kp & 31`.

Why this shape and not a "page size is a power of two" test: **nothing in this
repo constrains `kv_page_size`.** It is an operator-set TOML field
(`worker/src/config.rs:1114`, default 32 at `:1191`; C++ default 32 at
`driver/metal/src/context.cpp:108`), and every check on it anywhere is `> 0`.
No test uses a non-power-of-two value, no planner rejects one. A shift inferred
from a pow2 assumption would read the wrong slot the first time someone writes
24, silently. `sdpa_paged.metal` reached the same conclusion first and gates its
own `_p32` on `== 32`; this mirrors that form deliberately.

Verification chain, all four links, because three of them could have been silent:

1. `llama_pso_test` — 32/32. The `_p32` name resolves; a missing instantiation
   fails by name at load.
2. `PIE_METAL_SDPA_TRACE=1 llama_numerics_test` — prints
   `rows=48 requests=1 ... hd=128 -> MMA`. **The suite really does dispatch the
   kernel that changed.** Without this the next link proves nothing: the suite's
   headline cases are "40 rows over 2 requests", and the MMA path needs 32 rows
   *per request*, so it was entirely plausible that the kernel never ran.
3. `llama_numerics_test` — output **byte-identical** to the pre-change baseline,
   `rel_l2` figures included, verified by `diff` against a `git stash` build.
   (The suite is 51 pass / 18 fail on BOTH sides: those 18 are pre-existing MoE
   routing-tie failures, first divergence at `kind 11` = a projection in layer 0,
   which runs before attention. Not caused by this and not fixed by it.)
4. `gptoss_decode_step_test`, `llama_decode_step_test` (221), and
   `kv_append_paged_pso_test` (13) all pass.

**Not yet confirmed end to end.** 0.62 ms/layer x 48 is ~30 ms off a 572 ms
prefill fire, about 5% — which is inside the run-to-run spread of the agentic
replay, so a single A/B there would not be able to see it cleanly. The isolated
number is solid and the output is bit-identical; the end-to-end claim is not
made.

---

## STEP 2b — the kernel's shape, and the measurement that proves it

**The multiply is the wall, and staging optimizations cannot reach it.** A
two-sided ablation splits the `_p32` kernel (6.87 ms/layer):

| half | ms/layer | share |
|---|---:|---:|
| move keys into threadgroup memory (staging + barriers, no MMA) | 2.74 | 40% |
| multiply them (MMAs + softmax, no staging) | 3.38 | 49% |
| halves sum | 6.12 | 89% of unablated — consistent |

Read the second row against MLX. pie's arithmetic **alone, with staging deleted
entirely, is 3.38 ms — 2.2x MLX's 1.555 ms for the whole kernel.** So even a
free, instantaneous staging leaves pie 2.2x behind. The entire staging budget is
2.74 ms and spending all of it is not enough. That is what rules out the two
obvious next moves before either was built (see do-not-retry).

pie's multiply-only rate is 6.6 TFLOP/s against MLX's 14.4 for move *and*
multiply together. The likely mechanism, untested: at `KT=16`, `KF` is 2, so the
QK and PV loops carry only two independent accumulator chains — too few to hide
matrix-unit latency. MLX's larger K blocks carry more. `KT` 16 -> 32 was tried
and lost 60%, but that was measured WITH staging, where the wider tile costs
occupancy; the ablation says the multiply and the move want opposite things,
which is exactly the trade a double-buffered/async-copy decomposition exists to
break.

Nothing about paging, page size, addressing or memory layout explains the gap —
all four have now been measured and none of them is it.

### What MLX actually does differently (read from its source, 2026-08-15)

MLX 0.31.3 ships its Metal kernel sources in the installed package
(`site-packages/mlx/include/mlx/backend/metal/kernels/steel/attn/`) and its
compiled instantiations are readable out of `lib/mlx.metallib`. No upstream
branch needed. **Every bf16 d=128 configuration it has:**

    steel_attention_bfloat16_bq32_bk16_bd128_wm4_wn1
    steel_attention_bfloat16_bq64_bk32_bd128_wm4_wn1
    steel_attention_bfloat16_bq64_bk64_bd128_wm4_wn1

`wm4_wn1` is 4 simdgroups, 128 threads. **pie already uses 4 simdgroups and 128
threads, and pie's QT=32 / KT=16 is exactly MLX's smallest d=128 tile.** So
"port MLX's tiling" was the wrong instruction: the thread geometry is identical
and the tile is one MLX ships. (MLX may dispatch a larger tile for this shape —
the heuristic lives in a .cpp that is not shipped — but no d=128 config of its
changes the thread count.)

The differences that ARE real, all from `steel_attention.h` and `attn/mma.h`:

1. **MLX accumulates in fp32; pie accumulates in fp16.** MLX's `Stile`, `Otile`,
   `Qtile`, `Ktile`, `Vtile` are all `MMATile<AccumType=float, ...>` over
   `simdgroup_matrix<float,8,8>`; the bf16 threadgroup data is converted
   elementwise on load into `vec<float,2>` fragments. pie multiplies and
   accumulates in `simdgroup_matrix<half,8,8>` throughout.
2. **Consequently MLX does not chunk.** Its QK loop is one unbroken chain of
   `TD = BD/8 = 16` matmads straight into `Stile`. pie's `DCH=4` exists ONLY to
   bound the half-rounding chain, and it costs 4x the accumulator inits (8 per
   pass against 2) and 4x the extractions to float (16 float adds against 4).
3. **MLX holds fragments as plain `vec<float,2>` thread values** and materializes
   `simdgroup_matrix` objects transiently inside `mma()` via `reinterpret_cast`.
   pie holds live `simdgroup_matrix<half,8,8> S[KF]` / `PV[DF]` objects across
   the loop — a different register-allocation story.
4. **`fast::exp2` with `log2(e)` folded into the scale** (`params->scale *
   M_LOG2E_F`), so the base change is free rather than a multiply per element.
5. **MLX pads its threadgroup tiles by 16 BYTES** (`padK = 16/sizeof(T)` = 8
   halves), preserving 16-byte alignment. The falsified pad-of-2 experiment
   broke that alignment, which is the likeliest reason it lost — so pad-of-8 is
   NOT closed by that result. It is still only a move-half fix.

`get_coord` in MLX's `BaseMMAFrag` is byte-identical to pie's `qid/fm/fn`. pie's
kernel comment already says it borrowed that layout. The fragment math is the
same; the numeric type and the chunking are not.

### The fp32 experiment: FALSIFIED, and what it uncovered

Switching S/P/O to fp32 accumulators, deleting `DCH`, and holding fragments as
`vec<float,2>` (all three together, MLX's exact form) measured **7.26-7.34 vs
6.90 ms/layer — 5 to 6% SLOWER**. pie's `DCH` data stands; the register-pressure
reconciliation was wrong. Eighth row in the do-not-retry table.

**But the failure is informative, because it means MLX's advantage is not in its
`simdgroup_matrix` kernel at all.** pie's tile IS MLX's simdgroup tile, the
fragment coordinates are identical, and a faithful port of its arithmetic makes
pie slower. A 2.9x margin does not live in micro-optimization of the same
instruction on the same shape.

### THE ACTUAL FINDING: MLX has a second matrix path, and pie can reach it

This machine is an **Apple M5 Pro**, and MLX 0.31.3's metallib contains a whole
second family built on `mpp::tensor_ops::matmul2d` — the **neural accelerators**,
not `simdgroup_matrix`: `gemm_splitk_nax`, `segmented_mm_nax`,
`gather_mm_rhs_nax`, `BaseNAXFrag`. Its fragment is **16x16** (`kU = 16`), not
8x8.

That is what the `bq64` attention configs are. Under an 8x8 fragment,
`bq64_wm4_wn1` gives `TQ = 64/(4*8) = 2` and fails
`static_assert(TQ == 1)` in `steel_attention.h`; under `kU=16` it gives 1 and
compiles. So MLX's d=128 attention line-up is one simdgroup kernel
(`bq32_bk16`, pie's shape) and two NAX kernels.

**`mpp::tensor_ops::matmul2d` compiles through pie's own runtime shader
compiler on this machine** — verified by `tools/rawmetal/kernels/nax_probe.metal`,
which the probe compiles and reports on. pie uses `simdgroup_matrix`
exclusively and has never touched the other unit.

Re-measured MLX on this machine, same shape, so the target is current:

| `MLX_SDPA_BLOCKS` | ms/layer |
|---|---:|
| unset (its own heuristic) | 1.305 |
| 32 | 1.201 |
| 64 | 1.169 |

The 1.555 figure this plan was written against was pessimistic; the gap is
larger than recorded, not smaller.

### STEP 2c — DONE. The neural accelerators are the answer.

`tools/rawmetal/matrix_rate_probe.cpp` prices the two matrix instructions
against each other with no attention, no memory traffic and no MLX in the
picture. Each runs in its NATIVE configuration, because the question is what
each unit can do, not what they do under a forced common constraint. Both carry
two independent accumulator chains, or the number would be instruction latency
rather than throughput.

| unit | TFLOP/s |
|---|---:|
| `simdgroup_matrix` 8x8, half operands + half accumulate — pie today | 5.38 |
| `mpp::tensor_ops::matmul2d` 16x32x16, bf16 + fp32 — the neural accelerators | **32.2** |

**Conservative ratio 4.67x, against the 2.82x pie's multiply half needs.**
Sufficient on its own.

Conservative because the simdgroup arm FAILS ITS OWN PLAUSIBILITY CHECK and the
probe says so in its output: pie's shipped attention reaches ~6.9 TFLOP/s on its
multiply half, which is ABOVE this 5.38, and a real kernel cannot beat its unit's
peak. So 5.38 is a floor and the ratio is quoted against pie's achieved 6.9
instead. See "the simdgroup arm" below — that discrepancy is not yet explained
and it cuts both ways.

## STEP 3 — write a neural-accelerator attention kernel

This is the rewrite, and it is a DIFFERENT project from the one this document
was opened to scope. "Port MLX's tiling" is retired: pie's tiling already IS
MLX's simdgroup tiling, byte for byte. The work is targeting a different
execution unit — narrower in surface area, deeper in unfamiliarity.

1. Prototype in `sdpa_paged_probe` first, against the same 184-row shape, before
   anything touches `driver/metal/src/model/`.
2. Gate on EXACT hardware support, the way `_p32` gates on `kv_page_size == 32`
   and for the same reason: never on an inferred capability.
3. `MetalPerformancePrimitives.h` is reachable through pie's runtime shader
   compiler at `MTLLanguageVersion4_0` — verified by
   `tools/rawmetal/kernels/nax_probe.metal`. No Xcode needed; the `metal` CLI is
   absent on this machine and does not matter.
4. Read `steel/attn/nax.h` for the operand protocol: `get_left_input_cooperative_
   tensor` / `get_right_...` / `get_destination_...`, fill by element, then
   `gemm_op.run(a, b, c)`. `BaseNAXFrag::get_coord` gives the lane's 8 elements
   (`kElemRows=2`, `kElemCols=4`), the 16x16 analogue of the 8x8 `qid/fm/fn`
   that pie already uses.

### RESOLVED: the arm was right, the ablation was wrong

Both candidate explanations were tested and the second one was correct.

**The microbenchmark is sound.** Sixteen accumulator chains give exactly the
same 5.15 TFLOP/s as eight (double the work, double the time), so it is not
latency-bound. fp32 accumulate gives 5.26, so half is not the slow path. A
duration sweep runs the WRONG WAY for throttling -- 3.96 TFLOP/s at 128
iterations rising to 5.18 at 2048 and flat at 8192 -- which is clock ramp-up,
not thermal decay. **~5.2 TFLOP/s is the genuine `simdgroup_matrix` ceiling on
this device.**

**The ablation was over-optimistic.** `sdpa_nostage_mma.metal` removed the
staging writes, which left `ktile`/`vtile` written NOWHERE in the program --
provably loop-invariant, so the compiler hoisted the `simdgroup_load`s clean out
of the key loop. Barriers order accesses; they do not manufacture a dependency
that does not exist. Fixed by writing one element per thread per pass (1/16 of
the real staging cost), which is enough to defeat the hoist.

**And the corrected number closes the argument.** pie's multiply half now prices
at ~5.4 TFLOP/s against a measured ceiling of 5.15-5.26: **pie's attention
arithmetic is already running the simdgroup matrix unit at essentially 100% of
what it can do.** There is nothing left to win on this instruction -- not a
better tile, not a better accumulator, not a better fragment form. That is a far
stronger case for STEP 3 than "2.2x behind" ever was, and it is the reason the
eight falsified hypotheses were always going to fail.

**CONFIRMED on an idle machine, by ratio.** Bracketed run (see method below):
move 6.28 ms/layer (65%), multiply 4.25 (44%), reference 9.11/9.17. The multiply
arm works out to 23.7 GFLOP / 4.25 ms = **5.58 TFLOP/s against a same-session
simdgroup ceiling of 5.29** -- 5% over, inside run-to-run, and no longer the
34%-over impossibility the hoisted version produced. Both of those numbers come
from kernels that touch only registers and threadgroup memory, so they are
directly comparable and neither is affected by the DRAM caveat below.

### Two methods this cost, worth keeping

**Bracket every A/B with a reference on BOTH sides.** Arms run sequentially
inside one probe invocation, so a later arm is measured in a different machine
state than an earlier one. `sdpa_paged_probe` now measures the `_p32` reference
immediately before AND after the ablation arms and prints the drift, refusing to
vouch for the split if it exceeds 5%. Without this there is no way to tell a
real result from a machine that moved underneath it -- and the machine did move,
twice.

**Absolute ms DO NOT reconcile across sessions; ratios within one invocation
do.** On 2026-08-15 every memory-touching arm read 1.3-1.5x its earlier value on
a fully idle machine with 54% memory free. The cause is not compute and not
thermal: `matrix_rate_probe`, which runs entirely in registers, reproduced to
within 4% (5.05-5.29 and 30.65 TFLOP/s), while the `kv_depage` gather -- pure
DRAM bandwidth -- fell from **283 GB/s to 131**. So the SoC's arithmetic was
fine and its memory bandwidth was roughly halved. Most likely another GPU client
(the desktop app compositing), but that is UNPROVEN and no attribution should be
claimed. Consequence for anyone reading the ms figures in this document: compare
them only against figures from the same invocation.

### A gate that passed when it should not have

The corrected ablation first ran while an 8 GB python job held 17% CPU.
`require_quiet_gpu` checks free memory and swap, both of which were fine, so it
said OK -- and every timing came back ~1.4x high. Worse, the inflation was NOT
uniform: the memory-bound arm degraded more than the compute-bound one, turning
a 39/49 move/multiply split into 65/45. A contended benchmark is worse than a
dead one, because it still produces plausible numbers. The gate now warns on CPU
contention as well.

### The original open question (kept for the record)

Two explanations, and they point in opposite directions. Both are cheap to test
and neither is done:

- **The microbenchmark understates.** Eight accumulator chains may not cover the
  instruction's latency, or half-accumulate may be slower than fp32-accumulate on
  this hardware. Manually unrolling the accumulator array (on the theory that an
  indexed `simdgroup_matrix[]` spills) moved it by 0.0.
- **The ablation overstates.** `sdpa_nostage_mma.metal` removes the staging
  writes, so `ktile`/`vtile` are loop-invariant and the compiler is free to hoist
  the `simdgroup_load`s out of the key loop. If it did, the 3.38 ms "multiply"
  half is optimistic and the real multiply is slower — which would make the
  move/multiply split wrong, though the 89% sum check bounds how wrong.

The STEP 3 conclusion survives either way: 32.2 against 5.38 is 6.0x and against
6.9 is 4.7x, and both clear 2.82x. But the split in the ablation table should not
be quoted as settled until this is resolved.

---

## Numbers to beat (all measured 2026-08-14/15, this machine, this model)

| quantity | value |
|---|---:|
| pie `sdpa_paged_mma` `_p32`, 184 rows @ 7424 ctx | **6.87 ms/layer** (3.26 TFLOP/s) |
| ...of which: staging + barriers, no MMA | 2.74 ms/layer (40%) |
| ...of which: MMAs + softmax, no staging | **3.38 ms/layer** (49%) — **still 2.2x MLX's whole kernel** |
| pie `sdpa_paged_tiled` (fallback) | 21.0 ms/layer (1.1 TFLOP/s) |
| MLX `fast.scaled_dot_product_attention`, same shapes | **1.305 ms/layer** re-measured (1.17 forced) — was recorded 1.555 |
| gather 7424 keys pages -> contiguous | 0.107 ms/layer (284 GB/s, near roof) |
| pie MoE `affine_qmm_t_routed` (~215 ms/fire) vs MLX `gather_qmm` sorted (242 ms) | **already level — do not touch** |
| whole 184-row prefill fire @ 7424 ctx | 572 ms (pre-`_p32`) |
| 6-turn agentic replay, pie strategy B | 17.0 s (vs mlx-lm 9.2 s, vLLM 10.2 s) |
| SWE-bench known-5, graded | pie **4/5**, mlx-lm 2/5, vLLM-metal 1/5 |
| 8-concurrent throughput, 32 requests | pie 88.2 tok/s, vLLM 206.9, mlx-lm 203.8 |

Arithmetic per layer at these shapes: 22.4 GFLOP (QK^T + AV).

---

## DO NOT RETRY — falsified by measurement

Nine hypotheses have died. Each looked obvious; each is recorded so the next
session does not spend the afternoon again.

| hypothesis | result |
|---|---|
| `KT` 16 -> 32 (halve staging barriers) | **60% slower** (11.92 vs 7.44 ms) |
| threadgroup memory 16 -> 8 KB by aliasing the dead `qtile` (2 -> 4 resident groups) | **no change** (7.46 vs 7.44) |
| `DCH` 4 -> 8 / 16 | **2x slower** (15.46 / 13.17) |
| page size 32 -> 64 / 128 / 256 | **flat** (572/571/586/572 ms end to end) — the page WALK is not the cost |
| per-device tuning constants (`moe_tile_mid_per`, `qmm_bn_crossover_tg`) | M1 defaults are **best**; family-10 fallthrough is deliberate and tested |
| **pad Kᵀ's threadgroup stride 16 -> 18** to kill the 8-way bank conflict on the transposed staging write | **1.6% SLOWER** (7.69 vs 7.57). The conflict is real, but an odd leading dimension evidently costs `simdgroup_load` more than the staging write saves. Do not retry without separating the two — and note that separating them is probably not worth the afternoon |
| **de-page KV into contiguous scratch, then attend** | works, +1.26 ms/layer net, and **closes only 21% of the gap**. Judged not worth the redesign. Reopen only if the MLX port lands and the last 21% is what stands between pie and parity |
| **untransposed K staging + `simdgroup_load(transpose=true)`** | NOT TRIED, and closed by arithmetic rather than by a run: it can only attack the 2.74 ms move half, and the 3.38 ms multiply half alone is already 2.2x MLX. Even a perfect fix leaves the gap open |
| **`QT` 32 -> 64 (8 simdgroups), halving context walks per output row** | same. A genuinely different trade from the dead `KT` 16 -> 32 — it halves total staged bytes rather than merely trading barriers for occupancy — but it is still a move-half fix, and the move half is not what stands between pie and MLX |
| **fp32 accumulators + no `DCH` chunking + MLX's `vec<float,2>` fragment form** | **5-6% SLOWER** (7.26-7.34 vs 6.90). All three changed together, faithful to MLX's `steel_attention.h`. pie's original `DCH` finding stands and the register-pressure reconciliation was wrong. The useful residue: MLX's advantage is therefore NOT in its simdgroup kernel, since pie already has that kernel's exact tile |

The shipped kernel constants are at a local optimum and their comments explain
why. **Parameter tuning is exhausted, and so is memory layout. Only the
decomposition is left.**

---

## Tree state

Uncommitted. Nothing pushed. The load-bearing changes:

**Fixes, verified:**
- `driver/metal/src/kernels/sdpa_paged_mma.metal` + `model/{llama,gptoss}/kernels.cpp`
  — the `_p32` addressing above. 7.53 -> 6.92 ms/layer, output bit-identical.
- `runtime/engine/src/scheduler/worker.rs` — `LaunchGrouping::accepts()` now
  refuses a second device-geometry program per batch. Fixes SILENT OUTPUT
  TRUNCATION under concurrency (1 token of a 64-token budget, reported as
  `finish_reason:"length"`). Verified 64/64/64 at concurrency 1/2/4.
- `inferlets/chat-completions/src/engine.rs` — pool reservation rounds to the
  next POWER OF TWO instead of 256 pages. Fixes `cluster saturated` admission
  rejections that capped the server at 8 concurrent requests. Verified 32/32.
- `inferlets/opencode-session/src/engine.rs` — `aligned_prefill_chunks()` keeps
  every prefill fire out of the driver's slow row-count class (`rows % 8` in
  1..=6 costs a flat ~570 ms). Worth 1.69x on the agentic replay.

**Instrumentation added:**
- `driver/metal/tools/rawmetal/sdpa_paged_probe.cpp` — THE tool. Now prices four
  configurations plus the gather, and prints the gap-closure percentage rather
  than a bare win/lose, because "faster" was the wrong question.
- `driver/metal/tools/rawmetal/kernels/sdpa_contig_mma.metal` — one `#define`
  and one `#include`, giving a twin of the shipped kernel with the page walk
  removed. Kept: it is how the 21% was measured and how it would be re-measured.
- `driver/metal/tools/rawmetal/kernels/kv_depage.metal` — the gather.
- `driver/metal/src/model/llama/encode.cpp` — a `kernel_ablated` hook. **Its
  attribution is UNRELIABLE**: it converts llama's `Kind` to `Kernel` via
  `pso_kind()`, which is a lossy PSO-selection map (all projections collapse to
  `QmvGate`, attention matches nothing). Use `PIE_METAL_DISPATCH_TRACE` instead.
- `driver/metal/src/abi.cpp` — `PIE_METAL_FRAME_DIAG=1` describes a refused frame.
- `integrations/opencode/tools/require_quiet_gpu.sh` — preflight + post-run
  validity gate for benchmark arms.

**Results docs:** `integrations/opencode/results-{turn-latency,three-engine,speculation}.md`.
`results-turn-latency.md` carries a prominent CORRECTION at the top — its
Round 3 attribution was wrong for the reason above.

---

## Traps that cost real time

1. **An instrument's silence is not a measurement.** A dead vLLM server reported
   "0 patches"; a `grep -c` fallback (`$(grep -c ... || echo 0)` yields `"0\n0"`)
   declared a healthy arm void; an ablation matching nothing reported "this
   kernel is free". All three read as data. The `_p32` verification nearly made
   it four: a byte-identical numerics diff would have been equally byte-identical
   if the suite never dispatched the kernel, and only `PIE_METAL_SDPA_TRACE`
   distinguished "unchanged" from "never ran".
2. **Check a new instrument against something already known.** The attention
   probe was only trustworthy once it reproduced the MMA-vs-tiled ordering. Three
   earlier configurations of it were wrong (single-dispatch timing included the
   sync floor; 1-row MMA is a shape the driver never dispatches; tiled with the
   MMA launch shape inverted the result).
3. **A/B one thing.** The de-page result is trustworthy because the twin differs
   from the shipped kernel by three operations and shares everything else,
   including the bytes touched — the probe's identity page table means both read
   the same memory in the same order. The bank-conflict result is NOT trustworthy
   in the same way: padding the stride changed both the staging write and the
   fragment load, so its 1.6% loss does not say which one moved.
4. **"Is it faster?" is the wrong question when a 4.8x gap is open.** The
   original decision rule here was `< 7.44 ms -> proceed`, and de-paging passes
   it. Judged against the gap instead of against zero, it is a 21% answer to a
   379% problem. Write decision rules against the goal, not against the baseline.
5. **A falsification is only as good as the health of the rest of the run.** The
   pool-granularity hypothesis was "disproved" by a sweep that was really
   measuring the truncation bug.
6. **Two 30B servers do not fit on 48 GB.** Run one engine at a time; use the
   validity gate. A Metal OOM latches — first failure is pressure, everything
   after is a dead engine, and a run of identical fast failures is never a model
   result.

## If STEP 2b dies

If the MLX port does not land, the remaining honest options are: (a) accept the
gap and compete on the agentic axis where pie already wins (4/5 vs 2/5 on
SWE-bench), (b) profile with a Metal GPU capture rather than more constant
sweeps, or (c) close the ~1 s per-turn gap elsewhere — prefill is 71% of a
steady turn and attention is only part of it.
