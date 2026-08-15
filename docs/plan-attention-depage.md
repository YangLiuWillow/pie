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

## RESOLVED: the DRAM discrepancy was a cycling background job

`~/boN/.venv/bin/python`, ~14 GB resident, restarting every few minutes. It sat
at 17% CPU -- under the contention bar -- and left free memory looking fine, so
both existing gates passed. The harm was MEMORY PRESSURE: the compressor showed
612k pages stored and 65M decompressions, and `roofline_probe`'s read-only
streaming roof fell to **69.8 GB/s** against the ~200 GB/s this machine reaches
when quiet. Its whole-step figure also swung 15.4 -> 34.2 ms between
back-to-back runs.

The signature is diagnostic and worth recognizing again: **compute untouched,
DRAM halved.** `matrix_rate_probe` (registers only) reproduced to within 4%
while every memory-touching arm inflated 1.3-2.2x. That asymmetry does not
break an A/B, it TILTS one -- toward whichever arm is more memory-bound, which
is exactly how a 39/49 move/multiply split became 65/45.

`require_quiet_gpu` now refuses outright on any non-GUI process over 8 GB
resident. `roofline_probe` is the ground truth if the proxy is ever in doubt.

## STEP 3 — write a neural-accelerator attention kernel

### The instruction swap alone is NOT enough, and the arithmetic says so

In clean units the kernel is move 2.71 ms + multiply 3.38 ms = 6.87. NAX is 6.0x
the simdgroup unit, so the multiply becomes ~0.56 ms. Leaving the staging alone:

    2.71 + 0.56 = 3.27 ms  against MLX's 1.31  -->  still 2.5x behind

So a kernel that only changes the instruction is not worth writing. The staging
has to shrink with it.

### The staging is bandwidth-bound, and only BQ moves it

Every threadgroup stages the WHOLE context. At QT=32 that is
`32 heads x 6 row-tiles = 192` threadgroups, each streaming
`7424 x 128 x 2 tensors x 2 bytes = 3.8 MB`: **730 MB per layer**, which at
2.71 ms is ~269 GB/s. The real KV is only 15 MB; the rest is re-reads
(each kv head is read by 8 q heads x 6 tiles = 48 times) that mostly hit cache.

Two separate quantities, and confusing them is how `KT 16 -> 32` died:

    staged BYTES   proportional to 1/BQ           -- independent of BK entirely
    staged PASSES  proportional to 1/(BQ x BK)    -- barriers and loop overhead

`KT 16 -> 32` moved only the second and paid occupancy for it. That is why it
lost 60%, and why it says nothing about BQ.

### The instruction change FORCES the tile change. They are one change.

NAX's fragment is **16x16**, not 8x8. With 4 simdgroups, `TQ = BQ / (4 x 16)`,
and the kernel requires `TQ == 1` -- so **BQ = 64 is forced**. That is exactly
the change that halves staged bytes. This is not a coincidence to exploit
later; it is why every one of MLX's NAX attention configs is `bq64`, and why
its 8x8 config is `bq32` (pie's shape).

Target: **BQ = 64, BK = 32 or 64**, giving roughly

    staging ~1.35 ms + multiply ~0.56 ms = ~1.9 ms   against MLX's 1.31

with the residual most likely in pass overhead. Close enough to be worth
building; not close enough to promise parity.

### The enabling trick, taken from MLX

`steel_attention.h` aliases K and V onto the SAME threadgroup buffer --
`Ks = KV_smem; Vs = KV_smem` -- because V is not needed until after S is
computed and masked. That is what lets BK reach 64 at D=128 at all.

### ANSWERED: the cap is 32 KB, so BK = 32

`tools/rawmetal/device_caps.mm` (framework-only, no driver dependency) asks the
device instead of assuming. Apple M5 Pro, families Apple7/8/9 + Metal3:

    maxThreadgroupMemoryLength = 32768 bytes   -- the comment was right

With K and V aliased onto one buffer and MLX's 16-byte pad, at D=128 bf16:

    BQ=32 BK=16   8.5 + 6.0  = 14.5 KB   FITS   (pie today, 8x8 fragment)
    BQ=64 BK=16  17.0 + 6.0  = 23.0 KB   FITS
    BQ=64 BK=32  17.0 + 10.0 = 27.0 KB   FITS   <-- the target
    BQ=64 BK=64  17.0 + 18.0 = 35.0 KB   OVER

**So BK=32, and that is a shipped MLX configuration (`bq64_bk32_bd128`)** --
useful corroboration that the tile is real and not a guess. MLX's `bq64_bk64`
cannot fit this budget with Q staged, so if it dispatches here it must be
loading Q some other way.

**The occupancy risk, named up front:** 27 KB of 32 leaves room for ONE
resident threadgroup where pie's 14.5 KB leaves two. That is precisely the
trap `KT 16 -> 32` fell into. The mitigating difference is that BQ=64 halves
the number of threadgroups AND halves staged bytes, where KT only traded
barriers for occupancy -- but it is a risk, not a certainty, and the first
measurement must check it.

The lever if it bites: **do not stage Q at all.** Loading Q fragments straight
from device memory drops the tile to 10 KB and restores two or three resident
groups. pie's 8x8 kernel tried that and LOST (818.7 -> 799.7 tok/s), but at
BQ=64 the tile is twice as expensive to stage and the trade may invert.

### The fragment layout changes, and pie's central invariant SURVIVES

This was the real risk to the port and it needed checking before any code.
pie's kernel header rests on one property:

> A lane's two elements are always in the SAME ROW ... so the online softmax --
> row max, row sum, the rescale factor -- is per-lane state, never a
> threadgroup round trip. This one does not store S at all.

The NAX fragment is 16x16 with `kElemRows = 2`, `kElemCols = 4`,
`kElemRowsJump = 8`, against the 8x8's `kElemRows = 1, kElemCols = 2`. A lane
now holds EIGHT elements spanning TWO rows (`fm` and `fm+8`), four columns
each. That looks like it breaks the invariant. It does not:

    fm = (qid & 4) | ((lane >> 1) & 3)     depends on lane bits {4, 2, 1}
    fn = ((qid & 2) | (lane & 1)) * 4      depends on lane bits {3, 0}

`fm` and `fn` depend on DISJOINT bits, so the lanes sharing a row are those
varying bits 3 and 0 -- `{l, l^1, l^8, l^9}`, exactly as at 8x8. Verified
exhaustively over all 32 lanes for all 8 values of `fm`.

**So the two `simd_shuffle_xor` row reduction ports unchanged, and S still
never touches threadgroup memory.** What changes is bookkeeping, not design:

  * per-lane softmax state DOUBLES -- `max_score[2]`, `sum_exp[2]`, one per row
    half (this is MLX's `kRowsPerThread = 2`);
  * the per-fragment column loop is 4 wide, not 2;
  * masking indexes `fn..fn+3` and both `fm` and `fm+8`.

### BUILD ORDER — start here

Five stages. Each ends in a measurement or a test, and **none of them touches
`driver/metal/src/model/` until stage 4.** Do not write the whole kernel and
then debug it; the probe loop is seconds and the serving loop is minutes.

### STAGE 1 RESULT: passes, but the plan's 6x was wrong — it is 3.07x

NAX Q.K^T at the real serving shape, BQ=64 BK=32 d=128, operand fill included:

    0.695 ms   16.81 TFLOP/s   (11.7 GFLOP)
      vs 5.48 simdgroup ceiling      -> 3.07x   (rule was >= 3x: PASSES)
      vs 32.46 NAX ceiling, fill-free -> 52% of it

**Half the instruction's throughput goes to filling its operands.** The
microbenchmark's 6x was measured filling once and looping; an attention kernel
refills every pass, and that costs half. The rule passes, barely, so stage 2
proceeds -- but every projection in this document that used 6x must be redone
at 3.07x:

    multiply  3.38 / 3.07 = 1.10 ms   (not 0.56)
    staging   halved by BQ=64 ~1.35 ms
    total     ~2.45 ms    against MLX's 1.31   ->  still ~1.9x behind

So NAX takes pie's attention from 6.87 to roughly 2.5 ms -- a **2.8x win worth
having, and not parity.** Say that plainly rather than letting the earlier 6x
arithmetic stand.

The remaining 48% is the target for a stage-1b if anyone wants it: the fill is
24 scalar threadgroup loads per matmul against a `simdgroup_load` that fetches
a whole fragment as one instruction. MLX pays the same tax, which is some
evidence it is inherent to the cooperative-tensor path rather than a mistake
here.

**DO NOT RETRY: hoisting Q out of the pass loop.** Q is genuinely
loop-invariant and caching its eight A-fragments as `bfloat qfrag[TD][8]`
measured **7.88 TFLOP/s against 16.81** -- less than half. Sixty-four bfloats a
lane, indexed by a loop variable, spill to the stack, and a spilled operand
costs more than the threadgroup re-read it replaces. Same shape as the
accumulator-array trap in `matrix_rate.metal`.

**Stage 1 — the multiply, alone.** A NAX twin computing only `S = Q K^T` over
the real shapes, timed in `sdpa_paged_probe` against the existing MMA kernel's
multiply half. No softmax, no PV, no correctness. This answers the only
question that can still kill the plan: does NAX's 6x hold at attention shapes
and tile sizes, or does the operand-fill cost (8 elements per lane in and 16
out, through cooperative tensors) eat it?
**Decision rule, written first:** if the multiply half does not fall by at
least 3x, stop and re-plan -- the arithmetic in this document assumed 6x.

### CORRECTION FIRST: the stage-2 kernel does NOT compute attention

Written before the timings below, because they must be read through it.

A CPU reference over one threadgroup and one key block, with values chosen exact
in bf16 so any mismatch is a mapping error and never rounding:

    128 of 2048 scores WRONG   (worst relative error 15.55)

**So "the full attention pass computes in 1.652 ms" is not a claim I can make.**
A timing cannot distinguish correct attention from a wrong operand orientation:
both issue the same matmuls over the same bytes at the same rate. The figures
below are a sound measure of the WORK -- matmul count, staging traffic, register
behaviour -- and they are what the projections rest on. They are not evidence of
a correct kernel, and the correctness gate had to be built to discover that.

The error pattern is structured and points at the cause:

    wrong by element index e : 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8   (uniform)
    wrong by column          : cols 0-7 and 16-23 only  -> fn in {0,4}
    wrong by row%16          : rows 0-3 and 8-11 only   -> fm in {0,1,2,3}

`fm` and `fn` are the two halves of the fragment coordinate, and both predicates
are on lane bits -- so this is a LANE LAYOUT mismatch, not an index slip.

**Leading hypothesis, untested:** the kernel assumes all three cooperative
tensors share one lane layout. They need not. The left operand is 16x16 and the
right and destination are 16x32; 8 elements a lane in the left could be arranged
2 rows x 4 cols (what the code assumes), or 4x2, or 8x1, and MLX's
`BaseNAXFrag::get_coord` describes only ONE of the layouts in play.

**The one-hot probe has been RUN.** K set to an identity so `S[row][col]` must
equal `Q[row][col]` exactly, then twice: `Q[r][d] = r` and `Q[r][d] = d`, so
every output element reports the row and the dim it was built from.

    lane 8 (fm=0 fn=8): 0 0 0 0 8 8 8 8   /  8 9 10 11 8 9 10 11   BOTH CORRECT
    lane 0 (fm=0 fn=0): 64 64 80 80 ...   /  152 216 152 216        WRONG

**Lanes with `fn >= 8` are exactly right; lanes with `fn < 8` are wrong.** That
splits on lane bit 3 and on nothing else.

The sharpest clue is in the second mode: 152 is `sum(d = 2..17)` and 216 is
`sum(d = 6..21)` -- sixteen consecutive dims each, but NOT aligned to a
16-block. So a destination element that should hold a single `Q[row][col]`
instead holds a full 16-wide contraction taken at a SHIFTED offset in the head
dimension. The contraction length is right; where it starts is not.

**REFUTED, so nobody re-runs it:** that the 16 destination elements are 2 rows x
8 CONSECUTIVE columns (`col = fn*2 + j`) rather than two 4-column fragments.
Measured 1163 of 2048 wrong against the current mapping's 128 -- much worse, so
the two-fragment read-back is closer to right and the fault is elsewhere.

### THE APPROACH IS WRONG, not just the indices

Read from the real header,
`/System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers/MPPTensorOpsMatMul2d.h`
(the `metal` CLI is absent but the framework headers are on disk):

- the descriptor's 6th argument is `relaxed_precision`, so that label was right;
- **the documented primary API is `op.run(tA, tB, tC)` over TENSOR SLICES**, e.g.
  `A.static_slice<dynamic_extent, 64>(0, tgid.y*64)`. The library performs the
  memory-to-register mapping itself.

Hand-filling `get_left_input_cooperative_tensor()` element by element -- which
this kernel copies from MLX -- is a secondary path, and **the lane-to-element
mapping of a cooperative tensor is documented NOWHERE**. MLX can use it because
it fuses softmax between the two matmuls and has evidently derived the mapping
empirically; we adopted the hard path without needing to.

Both operand fills were checked against `BaseNAXFrag::load` line by line and
they match it exactly (`dst[i*kElemCols + j] = src[(fm + i*8)*str_x + fn + j]`,
fragment `nf` at `ct_b[nf*8 + ...]`). So the fills are faithful to MLX and the
kernel is still wrong -- which is the evidence that the cooperative tensor's
layout is not `BaseNAXFrag`'s, at least not in the way assumed.

**So the next step is not another index permutation.** Rewrite the matmul on the
tensor-slice API, where there is no mapping to get wrong, and get a CORRECT
kernel first. Only then consider hand-filled cooperative tensors, and only if
the fused softmax actually needs them -- with the correct kernel available as
the reference that would have caught this in an afternoon.

**Still open if the slice API is somehow unusable:** the left operand is 16x16 while the
right and destination are 16x32, and only the left's `k` axis is implicated by
the shifted-contraction evidence. Probe the LEFT operand's layout on its own --
one-hot in `ct_a` rather than in Q -- instead of inferring it through a full
matmul. Original note follows.

**How to settle it without guessing:** drive the kernel with a one-hot Q --
a single 1 at a known (row, dim) and zeros elsewhere -- and read which output
elements light up. That names the left operand's layout directly instead of
permuting indices until the count drops. The correctness harness in
`matrix_rate_probe` already prints the three histograms; it needs one more mode.

### STAGE 2 TIMINGS (of work done, not of correct attention)

BQ=64 BK=32 d=128, serving shape, reproduced across three runs to three
decimals. Each row adds exactly one thing to the row above it:

    Q.K^T only            0.701 ms   16.65 TFLOP/s   3.05x simdgroup
    + P.V                 1.490 ms   15.68 TFLOP/s   2.87x
    + online softmax      1.652 ms   14.14 TFLOP/s   2.59x
                                     (shipped 8x8 whole pass: 6.87; MLX: 1.31)

**The output accumulator does not spill** -- the single most likely way for
this design to die. At d=128 a NAX simdgroup owns 16 rows x 128 dims = 64
floats a lane, double the 8x8 kernel's 32, and the Q-hoist failure had just
shown 64 bfloats spilling. P.V costs 0.794 ms against Q.K^T's 0.698: near
perfect scaling, so it stayed in registers.

**The softmax is nearly free** -- 0.158 ms, 10% of the pass. pie's per-lane row
reduction ports to the 16x16 fragment intact, exactly as the lane algebra
predicted. One max and one sum per ROW HALF instead of one each; everything
else unchanged; S still never touches threadgroup memory.

K and V now alias one threadgroup buffer. Without it the tile is 36.3 KB
against the measured 32 KB cap.

### PENDING: the staging half, and why its first number is not quoted

Stage 2c (real staging from device) and stage 5 (staging alone) give the NAX
kernel the same two-sided ablation the shipped one got. Both are DRAM-bound,
and both were first measured while a 14.2 GB `pie-boN` process was resident --
the exact condition under which a memory-bound arm inflates and a
register-bound one does not. The compute rows above were unaffected (they
reproduce to three decimals); the staging row was not, and is deliberately not
recorded here.

A clean re-run is queued behind `require_quiet_gpu`, with `roofline_probe`'s
streaming roof captured alongside as the bandwidth ground truth. Do not accept
a staging number taken while the roof is below ~200 GB/s.

**Prepared, ready to run with it:** `sdpa_nax_straightk.metal`. The current
staging writes K transposed, putting consecutive lanes `BK+PAD = 40` halves
apart -- 20 words, `gcd(20,32) = 4`, so 32 lanes reach 8 banks. The variant
stages K straight (like V, contiguous write) and sets the descriptor's
`transpose_b` so the instruction does the transposing. It changes ONLY the
write; the padding experiment that failed on the 8x8 kernel moved the fragment
load as well, which is why that result does not settle this one.

**Stage 2 — the full kernel, probe-only.** Add online softmax and `O += P V`.
Per-lane state is `max_score[2]` / `sum_exp[2]`, one per row half; the row
reduction is still `simd_shuffle_xor` by 1 then 8. Mask indexes `fn..fn+3` at
both `fm` and `fm+8`. Still timed, still not wired.

**Stage 3 — numerics.** Compare against the shipped kernel on the SAME inputs.
The probe's buffers are zeroed and cannot validate anything, so this needs real
data: `llama_numerics_test` is the gate, and it must dispatch the new kernel --
verify with `PIE_METAL_SDPA_TRACE=1` that it prints `-> MMA` at `hd=128`,
because the suite's headline cases are "40 rows over 2 requests" and the matrix
path needs 32 rows PER REQUEST. A byte-identical diff means nothing if the
kernel never ran.

**Stage 4 — wire it.** Behind `PIE_METAL_NAX_ATTN=1` in
`model/llama/encode.cpp` where `llama_sdpa_mma_this_fire` selects. Gate on
EXACT hardware support, never inferred -- follow the `_p32` precedent
(`kv_page_size == 32`), not a family check that happens to be true today.
Remember `launch_shape` is asked separately from `pso_for` and is not handed a
`LlamaPsos`: if the two disagree the grid describes a different kernel than the
one that runs, which is wrong numbers rather than a crash.

**Stage 5 — end to end.** `integrations/opencode/bench_ab.py` 6-turn replay and
the SWE-bench known-5. Only here does the 17.0 s number move.

### Things a fresh session will not guess

- **MLX's Metal sources ship in the installed wheel.** No upstream branch, no
  Xcode.
  `~/.venv-vllm-metal/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/steel/attn/`
  -- `nax.h` is the operand protocol, `kernels/steel_attention_nax.h` the loop.
  Compiled entry points are readable with `strings .../lib/mlx.metallib`.
- **The operand protocol** is `get_left_input_cooperative_tensor<A,B,C>()`,
  `get_right_...`, `get_destination_...`, fill by element, then
  `gemm_op.run(ct_a, ct_b, ct_c)`. Descriptor:
  `matmul2d_descriptor(16, 32, 16, transpose_a, transpose_b, true,
  mode::multiply_accumulate)`.
- **Probe twins are one `#define` and one `#include`** of the shipped kernel --
  never a copy, or the A/B measures the drift between them.
- **Build:** `cmake -S driver/metal -B /tmp/metaltools -DPIE_METAL_BUILD_TOOLS=ON
  -DCMAKE_BUILD_TYPE=Release` then `cmake --build /tmp/metaltools --target
  sdpa_paged_probe -j 8`. The `metal` CLI is NOT installed and is not needed;
  compilation goes through `newLibraryWithSource:` at MTLLanguageVersion4_0.
- **`source integrations/opencode/tools/require_quiet_gpu.sh && require_quiet_gpu 2`
  before every timing run.** A ~14 GB python job in `~/boN` cycles on this
  machine and halves DRAM bandwidth without touching compute. The gate now
  refuses on it. Absolute ms from different invocations are NOT comparable.

### Remaining after that

pie's kernel assumes "the 32 KB a threadgroup gets". **That number is a comment;
nothing in this driver ever queries the device.** At BQ=64, BK=64, D=128 with
MLX's 16-byte padding the estimate is Q 17.4 KB + aliased KV 18.4 KB = 35.8 KB,
which does not fit 32 KB -- yet MLX ships that configuration. So either this
family allows more than 32 KB, or the layout is tighter than the estimate.
Query `maxThreadgroupMemoryLength` first and let the answer pick BK; do not
assume 32 and quietly settle for BK=32.



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

---

## Ported from upstream `dev-sslee`, and one claim corrected

Two Qwen3 tool-call parsing defects, both present in our diverged copy of
`model/qwen_3/src/chat.rs`, both reproduced here before fixing:

- a parameter name scan running past its parameter into a shell redirect,
  yielding a ninety-character argument KEY and no error;
- a function name scan doing the same one level up, yielding tool names like
  `bash\n<parameter=command` and `<bash>`, dispatched confidently.

**CORRECTION to the first commit message.** It says these are "on the measured
path, not a hypothetical". That overstates what is known. Upstream captured the
parameter bug live on `django__django-10914`, which is NOT in our known-5
(`django-12276`, `13028`, `13089`, `14373`, `15569`). Our five are all django
instances run with the same model and the same tool dialect, so the same
failure mode is entirely plausible on them -- but no agent output was retained
from those runs, so there is NO evidence it actually fired. The fixes are
justified by reproduction against our parser; they are not justified by our
benchmark results, and nothing in the 4/5 should be attributed to them.

Worth fixing for the next run: the SWE-bench harness keeps predictions and
grades but discards the agent transcript, which is why this cannot be checked
retrospectively. Upstream found their bug "by reading the captured bytes rather
than the divergence counts" -- we cannot do that yet.

The CUDA partial-RoPE fix in the same branch needs no port: it cites our Metal
`rope.metal` as one of three references proving the CUDA form wrong, and that
is accurate -- ours uses `half = rope_dims/2` with `[rope_dims, head_dim)`
pass-through.
