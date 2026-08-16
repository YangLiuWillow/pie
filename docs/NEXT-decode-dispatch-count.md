# Resume here — decode: attention is 30% of a step and 2.1x off its roofline

> **This file previously said decode was dispatch-count-bound, with 20.1% of a
> step in `moe_route_sort` and `silu_mul`. That was wrong twice over and is
> retracted below.** Both figures were instrument error. The real target is
> attention.

---

## 1. Where things stand

Four engines, one session, idle machine (roof 292-296 GB/s), `tools/four_way.sh`.
"pie original" is the same binary with the new kernels switched off.

| prompt | **pie now** | pie original | mlx-lm | vLLM-metal |
|---:|---:|---:|---:|---:|
| **TTFT** 5,840 | **2.80 s** | 8.25 | 2.98 | 4.43 |
| **TTFT** 16,090 | **7.37 s** | 27.84 | 8.85 | 17.10 |
| **TTFT** 28,390 | **13.93 s** | 57.40 | 17.33 | 37.15 |
| **decode** 5,840 | 54.4 tok/s | 46.9 | **66.1** | 51.3 |
| **decode** 16,090 | 41.0 | 26.1 | **47.5** | 30.3 |
| **decode** 28,390 | 27.8 | 19.8 | **35.6** | 16.8 |
| 6-turn replay | **7.14 s** | 16.30 | 8.02 | 10.18 |

**Prefill is pie's** (1.06-1.24x over mlx-lm, 1.6-2.7x over vLLM-metal).
**Decode is mlx-lm's** (1.16-1.28x). Closing decode is the remaining job.

### Landed, with the switch that reverts each

| kernel | win | revert |
|---|---|---|
| `sdpa_paged_decode_hshare` - one KV read per GQA pair | 1.18x @5.8k -> 1.46x @28k | `PIE_METAL_SDPA_HSHARE=0` |
| `sdpa_paged_nax` - fused flash attention on the neural accelerators | 4.86x over `sdpa_paged_mma` | `PIE_METAL_SDPA_NAX=0` |
| `affine_qmm_t_routed_nax` - routed MoE GEMM on NAX | 1.92x / 2.21x by tile | `PIE_METAL_QMM_NAX=0` |
| `affine_qmm_t_nax` - dense projections on NAX | 2.72x | `PIE_METAL_QMM_NAX=0` |
| NAX row gate lowered 64 -> 32 | 2.28x on a 32-row fire | `PIE_METAL_SDPA_NAX_MIN_ROWS=64` |
| `sdpa_paged_decode_split` + `..._split_combine` - the key range split across threadgroups at QH=4 | kernel 1.16x @8k-16k; **+1.65% on a decode step** | `PIE_METAL_SDPA_SPLIT=0` |

`driver/metal/src/kernels/nax_frag.h` is the shared substrate: 16x16 register
fragments, `frag_mma` (N=32), `frag_mma_k32`, and the lane layout.

### Test baseline

`llama_pso_test` 32 - `llama_decode_step_test` **243** - `llama_bind_test` 38 -
`kv_append_paged_pso_test` 13 - `gptoss_decode_step_test` all -
**`llama_numerics_test` 50/19**.

**The numerics baseline moved 51/18 -> 50/19 deliberately.** 18 are pre-existing
MoE routing ties; the 19th came from lowering the NAX row gate and is the same
class. Evidence in `sdpa_nax_min_rows`'s comment. **Do not "fix" it by raising
the gate without reading that comment.**

---

## 2. THE TASK: decode attention (split-K now landed; the chain is what is left)

### The sound composition of a decode step — RE-MEASURED, and re-labelled

`tools/decode_retrace.sh`, ablation on `decode-rows-probe` at rows=1, ctx 7424,
baseline **17.35 ms** on the current build (split-K landed).

| what is actually removed | cost | share | its own roofline | off by |
|---|---:|---:|---:|---:|
| attention (`sdpa`) | 5.05 ms | **29.1%** | 2.47 ms | **2.04×** |
| **all three** routed expert projections | 4.03 ms | 23.2% | 3.06 ms | 1.32× |
| **all nine** dense matvecs, incl. the LM head | 2.96 ms | 17.1% | 1.53 ms | **1.93×** |
| everything else | ~5.3 ms | ~30.6% | — | — |

**The labels matter and the previous version of this table got them wrong.**
`PIE_METAL_ABLATE` matches the kind `pso_kind` MAPS TO, and that map is
many-to-one:

```
case QmvQ, QmvK, QmvV, QmvO, QmvGate, QmvUp, QmvDown, Router, LmHead
                        -> Kernel::QmvGate       // NINE kinds, one token
case ExpertGate, ExpertUp, ExpertDown
                        -> Kernel::LlExpertGate  // THREE kinds, one token
```

So `qmv_gate` does not ablate "the dense projections" — it ablates every dense
matvec in the model **including the LM head**, and this checkpoint is a mixture,
so `QmvGate` itself is never even dispatched. `ll_expert_gate` ablates the whole
routed FFN, not the gate projection. The old rows were right in MAGNITUDE (5.26 /
3.79 / 2.86 against 5.05 / 4.03 / 2.96 here, and the rooflines were already
computed for the whole groups) and wrong in NAME, which invites planning work
against a kernel that is not what was measured.

**A token that maps to no dispatched kind ablates NOTHING and reports a clean
zero.** `qmv_q`, `qmv_o`, `qmv_up`, `qmv_down` and `ll_router` are all in that
position here, and they measured −0.05, −0.04, −0.09, −0.02 and −0.03 ms. Five
known-zeros is a free calibration: **this instrument's noise floor is ±0.09 ms,
±0.5% of a step**, so the three real numbers above are resolved by 30–50×.

Those zeros also settle the drift. The baseline repeated last read 17.77 ms,
+2.42%, and the script says so — but late drift cannot propagate backwards
through five zeros measured against the first baseline, so 17.35 is the right
reference for the sweep and the tail run is what moved.

**The rooflines must be recomputed for the GROUP, and the first version of this
table was not** — it compared the measured 2.96 ms against 1.53 ms, a floor
derived from "dense weights 452 MB", which is q+k+v+o at 4 bits with **no
group scales and no LM head**. The ablation removes the LM head. Counting what
is actually removed, at 4 bits plus a bf16 scale and bias per group of 64:

| | bytes | |
|---|---:|---|
| lm_head, K=2048 N=151936, ×1 | 175.0 MB | one dispatch per fire |
| q ×48 | 226.5 | |
| k ×48, v ×48 | 28.3 + 28.3 | |
| o ×48 | 226.5 | |
| router ×48 | 7.1 | |
| **total** | **691.7 MB** | → **2.34 ms** at 296 GB/s |

So the dense matvecs are **1.27× off their roofline with ~0.62 ms available**,
not 1.93× and 1.4 ms. That is 3.6% of a step, and it is the LEAST attractive of
the three, not the most.

**Attention remains the target by a wide margin**: 5.05 ms against a 2.47 ms KV
roofline is 2.05× off, with **2.58 ms — 14.9% of a decode step — available.**
That is more than the other two blocks' headroom combined.

*This correction is the same error the row labels above caused, one level up:
a measured group compared against a roofline computed for a different set. It
survived into a commit. Recompute the floor for exactly what the ablation
removes, every time.*

### Why, and what to build

`sdpa_paged_decode_hshare` shares one KV read across a PAIR of query heads
(QH=2). With 32 query heads over 4 KV heads that still reads each KV head's data
**four times**. QH=4 and QH=8 were measured and are SLOWER, but not because the
sharing fails -- because the grid is `n_q_heads / QH` threadgroups (32, 16, 8, 4)
and this device stops being filled below 16.

**Flash-decoding was the fix, and it is now LANDED** — split the key range across
threadgroups so the grid stays tall at QH=4, then merge the partial (max, sum, o)
in a second pass. See "LANDED: split-K decode attention" below for the numbers,
the wiring and the four instrument faults found on the way.

It delivered 1.16x on attention, not the ~1.8x estimated here. The estimate
assumed the kernel could be brought to ~1.2x of its roofline; it is
latency-bound rather than bandwidth-bound, so the roofline was never the thing
holding it back and reaching it was never on offer. **The estimate below is
retained as written, wrong, because the reasoning error in it is instructive:
a roofline bounds what is possible, and says nothing about what is achievable
for a kernel whose constraint is somewhere else.**

*Estimated at the time:* attention 5.26 -> ~2.9 ms if it reaches ~1.2x of
roofline. That is 2.4 ms of a 17.55 ms step, ~13.5%, which at 16k would be
41.0 -> ~47.4 tok/s against mlx-lm's 47.5. Parity, roughly, from this one
change.

---

## 2a. RETRACTED: the "20.1% in two small kernels" target

This file previously said `moe_route_sort` was 10.2% and `silu_mul` 9.9%. Both
were instrument error, and the two instruments failed in different ways. Recorded
in full because the same traps are waiting for the next person.

**The dispatch trace overstates SMALL kernels.** It brackets every dispatch with
timestamps, a fixed cost, so a kernel near the launch floor is inflated by tens
of times. It said `silu_mul` cost 38 us a layer; it costs about 0.4. It is
roughly right for the big kernels (30.0/21.6/16.3 measured against 33.8/25.8/10.0
traced) and useless below ~10%.

**Plain ablation is UNSOUND for a kernel that emits indices.** `moe_route_sort`
produces `perm`, `inv` and `tile_expert` -- addresses. Removing it sends every
downstream kernel chasing garbage, and over a probe run `moe_combine_sorted`
went 345.9 -> 4241.0 ms while `affine_qmv_fast_..._b_8` went 797.2 -> 134.9. The
net looked like a clean 2.04 ms saving and was nothing of the kind.

**The sound measurement** is `PIE_METAL_MOE_SORT_SKIP_AFTER=0`, which runs the
sort on layer 0 only and leaves an earlier layer's indices in place: still in
range, still structurally valid, so downstream access patterns are real. 47 of 48
sorts cost **0.24 ms**, i.e. ~5.1 us each -- and `tools/rawmetal/moe_sort_probe.cpp`,
dispatching the kernel completely alone, independently says **~5.0 us**.

So the sort is **1.4% of a decode step**, not 11.6%, and `silu_mul` is nothing at
all. Neither is worth fusing. There is no dispatch-count problem.

### Decode attention is LATENCY-bound, not bandwidth-bound

This is the finding that should steer the next attempt, and it rules out the
obvious plan. Achieved bandwidth on UNIQUE bytes, from the head-sharing sweep:

| ctx | QH=1 | QH=2 | if the redundant GQA reads were NOT cached, QH=1 would need |
|---:|---:|---:|---:|
| 2,047 | 6.9% of roof | 7.4% | 55% of roof |
| 8,191 | 15.1% | 18.5% | **121%** |
| 12,287 | 16.6% | 23.0% | **133%** |
| 16,383 | 18.2% | 26.4% | **146%** |

**Above 8k the uncached traffic would exceed the memory roof, which is
impossible — so the redundant reads are already largely cache-served.** And the
kernel sits at 18–26% of the roof on the bytes it genuinely must move. A kernel
nowhere near the bandwidth roof, which cannot be moving the traffic that would
put it there, is *waiting*.

**So flash-decoding is the wrong plan** — *and this conclusion was wrong, while
every measurement it rests on was right.* The traffic argument holds: the
redundant GQA reads ARE cache-served, and splitting the key range to save them
would indeed have bought nothing. What the argument missed is that splitting
does something else entirely. It puts more independent dependency chains in
flight, which is exactly what a latency-bound kernel is short of, and it lets
QH=4 run without the grid collapsing. Landed, it is worth 1.16x.

**The lesson is about the shape of the inference, not the data.** "This
mechanism cannot help for reason X" is only sound if X is the mechanism's only
effect. Ruling out a technique by refuting one of its rationales is not ruling
it out.

### REJECTED: unrolling the key loop

The natural attack on latency: issue UNROLL keys' loads before any of their
arithmetic, so the loads overlap. Correct at every width (0 of 4096 wrong,
all 32 heads).

| | 12k | 16k |
|---|---:|---:|
| QH=2 U=1 | 0.361 ms | 0.432 |
| QH=2 U=2 | 0.373 | 0.455 |
| **QH=2 U=4** | **0.342** | **0.402** (1.07×) |
| QH=4 U=1 | 0.521 | 0.655 |
| QH=4 U=2 | 0.469 | 0.574 |

7% in isolation at 12k and 16k — and **a 19% REGRESSION end to end at short
context**: landed, decode at 5,840 tokens went 53.9 → 44.3 tok/s, while 16k and
28k barely moved. Reverting restored 53.9. Discarded.

The probe's timing sweep only covered 12k and 16k, so it never saw the shape
where the extra registers cost more than the overlap buys. **Sweep the context
as well as the parameter** — an isolated win at two long contexts is not a win.

The prototype keeps the `UNROLL` knob
(`tools/rawmetal/kernels/sdpa_hshare_decode.metal`) so the negative result is
reproducible; the shipped kernel does not have it.

### LANDED: split-K decode attention, 1.15-1.16x above 8k

Two kernels in `src/kernels/sdpa_paged.metal`. The first gives each threadgroup
a SLICE of the key range and writes a partial softmax — its own running max, its
own sum, an unnormalized accumulator; `sdpa_paged_split_combine` rescales each
partial to the global max and merges them, which is the online softmax's own
rule applied once across splits instead of once per key.

**The motivation is concurrency, not traffic.** Flash-decoding is usually sold
as a way to cover a whole GQA group without the grid collapsing — but the
traffic argument does not hold here (see above: the redundant reads are already
cache-served). What it actually buys on this machine is more independent chains
in flight for a latency-bound kernel: at QH=2 over 32 query heads the grid is 16
threadgroups; splitting S ways makes it 16·S with each chain 1/S as long.

It also makes QH=4 viable, which **confirms the earlier occupancy diagnosis** —
QH=4 lost as a single kernel purely because the grid fell to 8 threadgroups.

Every configuration re-measured on the fixed harness (48 reps amortized in one
command buffer), against the kernel in use compiled and timed **in the same run,
interleaved**, on a GPU warmed until its clocks stopped moving. ms/layer, 32
query heads over 4 KV:

| config | 2k | 8k | 12k | 16k | | 2k | 8k | 12k | 16k |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| QH=2 S=1 *(was in use)* | 0.040 | 0.120 | 0.173 | 0.233 | | — | — | — | — |
| QH=2 S=2 | 0.038 | 0.126 | 0.183 | 0.239 | | 1.04× | 0.96× | 0.95× | 0.98× |
| QH=2 S=4 | 0.042 | 0.118 | 0.177 | 0.250 | | 0.94× | 1.02× | 0.98× | 0.93× |
| QH=2 S=8 | 0.049 | 0.113 | 0.161 | 0.236 | | 0.81× | 1.07× | 1.08× | 0.99× |
| QH=4 S=2 | 0.037 | 0.114 | 0.165 | 0.210 | | 1.08× | 1.06× | 1.05× | 1.11× |
| **QH=4 S=4** | **0.041** | **0.104** | **0.151** | **0.200** | | **0.96×** | **1.16×** | **1.15×** | **1.16×** |
| QH=4 S=8 | 0.053 | 0.112 | 0.156 | 0.202 | | 0.75× | 1.07× | 1.11× | 1.15× |

**QH=4 S=4 is the pick**, and two independent clean runs agree to within 0.03×.
Correct at every context — 0 of 4096 wrong, all 32 heads, against a float64
reference — including 127 and 511, where the last splits cover no keys at all
and the −inf/0 sentinel is what keeps the merge from folding in an accumulator
that was never written.

**The 2k cell is 0.96×, a 4% LOSS, and it ships that way.** The earlier sweep
put it at 1.02×; the fixed harness says slightly negative. It ships because the
host cannot gate on it: `pso_for` and `launch_shape` are handed the geometry,
the row count and the request count, and **not the context length**, so "split
only above 8k" is not a question either site can ask. The trade is ~1.2% of a
decode step given up at short context against ~4.5% gained at long, and short
context is where a decode step is cheapest in absolute terms anyway.

The short-context column is in the sweep on purpose. The rejected key-loop
unroll won 7% at 12k/16k, was landed on that evidence, and cost 19% end to end
at 5,840 tokens — because its sweep never looked below 12k.

Expected end to end: attention is 30% of a decode step, so ~15% of it is ~4.5%
of a step — 41.0 → roughly 43 tok/s at 16k. Real, and still short of mlx-lm's
47.5.

### End to end: +1.5% on a decode step, and why that is a third of the prediction

**The rate probe cannot answer this.** Decode rate through the HTTP server, one
boot, seven prompt sizes, split-K on:

| tokens | 5,840 | 9,940 | 13,015 | 16,090 | 19,165 | 22,240 | 28,390 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **on** | 55.2 | 42.6 | **47.0** | 39.8 | 37.4 | 34.2 | 30.9 |
| **off** | 54.3 | 49.7 | 43.7 | 41.1 | 35.0 | 32.9 | — |
| ratio | 1.02 | 0.86 | 1.08 | 0.97 | 1.07 | 1.04 | — |

13,015 is faster than 9,940 **on the same arm**, and a decode cannot get cheaper
on more context — so that is ~10% of scatter against a ~4% effect. An earlier
three-point run gave 1.02× / 0.94× / 1.09× with a 0.4% repeat-arm drift control,
and its 0.94× at 16k reproduced across two independent boots. **Reproducibility
is not accuracy**: the same prompt sequence leaves the prefix cache and page pool
in the same state every time, so a state-dependent distortion repeats perfectly.

**`decode-rows-probe` can.** Fixed context, fixed row count, ten fires with the
first two discarded, straight through the driver — no HTTP, no shim, no prefix
cache, no tokenizer. `tools/split_fire_ab.sh` alternates the two settings:

| | rep 1 | rep 2 | rep 3 | rep 4 | rep 5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| split **on** | 17.33 | 17.35 | 17.39 | 17.33 | 17.31 | **17.342 ms** |
| split **off** | 17.64 | 17.60 | 17.70 | 17.62 | 17.58 | **17.628 ms** |
| ratio | 1.018× | 1.014× | 1.018× | 1.017× | 1.016× | **1.0165×** |

**+1.65% on a decode step**, every pair between 1.014× and 1.018×, with a
within-arm spread of 0.35%. The arms separate by five times their own noise,
which is what makes this readable where the rate probe is not.

#### The prediction was +4.3%, and the gap is the probe's own cache

Attention is 30% of this step and the kernel is 1.16× faster, so the arithmetic
says 4.3%. Working backwards from the measured 1.65%, the **in-situ** attention
speedup is ~1.06×, not 1.16× — the isolated ratio overstates the transferable
gain by about 2.8×.

The difference is in how `sdpa_paged_probe` times a kernel: it repeats the
dispatch 48 times inside one command buffer **against the same K and V
buffers**, so every rep after the first reads a cache that is already warm. A
real fire's 48 layers each read *different* KV — 730 MB of it at this context,
far past any cache. Split-K's advantage is latency hiding, and there is less
latency to hide when the data is already close.

So the probe's ratios are an upper bound on what transfers, and by roughly 3×
here. That caveat applies to every kernel ranked on that harness, including the
head-sharing numbers above. **It does not invalidate the ranking** — the
configurations were compared with each other under identical conditions — but a
kernel ratio from that probe should not be turned into an end-to-end prediction
without this discount.

### LANDED — the wiring, and the four sites that must agree

`sdpa_paged_decode_split` and `sdpa_paged_split_combine` now live in
`src/kernels/sdpa_paged.metal` with the full `bind::SdpaPaged` signature, the
partials added to that ABI at `PartialO = 18` and `PartialMS = 19`.

**No new DAG `Kind`.** The precedent was already in `encode_llama_step`: a split
projection is two dispatches, and the reduce rides the same argument table as
its GEMM rather than taking a DAG entry. Split-K attention is the same shape, so
the combine is emitted inline after the split, from the same predicate.

`sdpa_split_this_fire` is that predicate, and **four** sites read it:

| site | what it decides |
|---|---|
| `build_llama_psos` | whether both pipelines are compiled — fatally, together |
| `pso_for` | the split kernel instead of head sharing |
| `launch_shape` | a grid `kSdpaSplitHeads` shorter in x, `kSdpaSplit` deep in z |
| `encode_llama_step` | **whether the combine runs at all** |

The fourth is the new one and the worst to get wrong: a split with no combine
leaves the attention output holding whatever the activation pool last put there.
No crash, no slowdown, wrong logits. `PIE_METAL_SDPA_SPLIT=0` is the complete
way back, read from the same function so it cannot half-apply.

Gate: `rows == 1`, `requests <= 1`, paged, page 32, d128, `n_q_heads % 4 == 0`
and `gqa % 4 == 0`. The row bound is **correctness, not preference** — the
partials buffer has no row axis, so a second row would write through the first
row's. `llama_sdpa_partial_elems` sizes it (~66 KB, one buffer for the whole
model) from that same predicate, and `bind.cpp` binds both slots at every
layer's attention unconditionally, because whether a given FIRE splits is a
row-count decision made long after the table is written.

New test `check_the_split_grid_matches_its_kernel` pins each clause and both
grid extents; `llama_decode_step_test` is 233 → 243. The head-sharing grid test
now asks at **two** rows, because at one row the split takes that fire and the
question would be about a kernel that does not run.

### Six instrument faults found while landing it — read these first

Landing the kernel was the easy half. Verifying it turned up six separate ways
this repo's instruments lie. Three had already contaminated numbers quoted
above; the fourth nearly caused a correct result to be thrown away; the fifth
produced the most convincing null result imaginable out of nothing at all.

**−1. THE A/B RAN TWO ARMS OF THE SAME BINARY.** The first complete end-to-end
run gave 54.2 / 40.8 / 27.8 tok/s with split-K off against 54.3 / 40.8 / 27.9
with it on — a perfect 1.00× at every prompt size, with **byte-identical output
text**, and a repeat-first-last drift control reading 0.0%. Every quality signal
said this was a careful measurement of nothing.

`target/release/pie` was two hours older than the kernel. `cargo build --release
-p pie` does not build it — the binary crate is `pie-bin`, and cargo fails the
package match rather than building the wrong thing, so a build step that looks
like it worked can leave the old binary in place. Both arms were the pre-split
code.

The tell was available and I nearly talked myself past it: split-K reorders the
softmax into fp32 partials, so 200 greedy tokens coming out **byte-identical**
is not what "the kernel ran and was neutral" looks like. `tools/split_ab.sh`
now builds `pie-bin` itself and refuses to run if any driver source is newer
than the binary afterwards.

And the thing that actually settled it is the one this file already prescribes:
`PIE_METAL_SDPA_TRACE=1` now reports this gate too, and says
`hd=128 page=32 nq=32 nkv=4 rows=1 requests=1 paged=1 on=1 -> SPLIT`. **A
throughput number cannot tell "the fast path is no faster" from "the fast path
never ran."** That sentence was already in this repo, about this exact gate's
two neighbours. It cost a day anyway.

**−2. HALF AN A/B LOOKS EXACTLY LIKE AN A/B.** The first run of
`tools/split_fire_ab.sh` printed two decode-step timings, 17.40 ms and 17.38 ms,
under headings that said one was split-on and one was split-off. Both were
split-on. **Both `off` runs had died on model load** — 24.8 GiB was wired by
servers earlier measurements had left behind, so the weights no longer fit — and
the script printed an empty string for them and carried on. Two numbers that
agree to 0.1% is a *persuasive* null result.

A run that produced no timing is now fatal, with the tail of its log; and the
script refuses to start if a `pie serve` is holding memory. Note the wired
memory was not leaked — killing the servers took it from 24.8 GiB back to 2.25.

**0. THE GPU RAMPS, AND UNTIL IT HAS YOU ARE MEASURING A CLOCK STATE.** This is
the largest effect in this file by a wide margin and it is not subtle. The QH=2
kernel at 16k measured **1.475, 1.154, 0.837 and 0.233 ms/layer on four runs of
the same binary** — a 6× spread, ordered by how long the machine had been under
load. On the half-ramped runs split-K read as a 0.72–0.95× LOSS at every
context, consistently, across two independent runs; on the fully ramped one it
is a 1.15–1.16× win. **Two reproducible, mutually-consistent measurements said
reject, and they were both wrong.**

What caught it was not judgement but the monotonicity guard: a decode's
attention cannot get cheaper on more keys, and every bad run said 16k was
faster than 12k — in *both* kernels, which is the signature of the machine
changing under the measurement rather than of one kernel being better. The arm
now warms until two successive measurements of the same work agree within 5%,
reports how many passes that took, and refuses to be quoted if it never
settles.

The general rule, which this file did not have and now does: **a probe that has
not shown its clocks are stable has not made a measurement.** Absolute numbers
from a cold GPU are meaningless, and so are ratios taken across a ramp.

**1. `llama_numerics_test` cannot express a gqa > 2 geometry.** The split needs
gqa 4 at one row, and both routes to it from `base_geometry` diverge *before
attention*, with every attention switch off:

| change | first divergence |
|---|---|
| `n_kv_heads` 2 → 1 (gqa 4, all other widths unchanged) | `QmvO` |
| `hidden` 512 → 1024, 8 heads over 2 | `QmvGate` |

The second is not about gqa at all — `hidden = 1024` at **gqa 2** fails in the
same place and worse (rel_l2 30.2 against 4.4). And a **ring-KV control** at gqa
4, on a path this landing never touches, fails identically to the paged one. So
the test's reference is sound at exactly one set of widths, and
`base_geometry`'s comment understates its own constraint: 1024 satisfies both
rules it names (`K % 512 == 0`, `N % 8 == 0`) and still does not work. The
numerics baseline is therefore **unchanged at 50/19** and the split has no
coverage there. Fixing this is worth doing on its own — it is a hole in the one
test that checks arithmetic end to end.

**2. This probe's arms are not independent.** `sdpa_paged_probe` allocates per
arm and frees nothing, and calls `make_resident()` on the growing heap each
time — so an arm's cost depends on how much ran before it. The shipped split arm
sits at the end of the decode section and read the QH=2 baseline at
0.148 / 0.329 / **0.975 / 0.984**, against a real cost near 0.192 / 0.307 /
0.369 / 0.430, with **12k slower than 16k** — impossible for one kernel on more
work. `PIE_SDPA_PROBE_SPLIT_ONLY=1` now runs that arm alone on a fresh heap, and
the arm carries a monotonicity guard that says "this run is NOISE" out loud
rather than printing a table that looks like a result.

**3. Single-dispatch timing measures the sync floor, not the kernel.** `bench`
in this same file already records this — it priced a 1-row fire's attention at
3.536 ms/layer — but `split_run`, `hshare_run` and `krow_run` all timed ONE
dispatch per command buffer anyway. Run alone on an idle GPU, `split_run`
reported the QH=2 kernel at **0.99 ms/layer at 2k, 8k, 12k and 16k alike**. A
cost that does not vary with the amount of work is the clearest possible sign
that the work is not what is being measured. `split_run` now repeats the pair 48
times inside one command buffer — a model's worth of layers, which is what a
fused fire actually pays — and divides.

**Every earlier number in this file's split-K table was measured with fault 3
present and against a baseline quoted from a different run.** The arm now
compiles the QH=2 kernel it replaces and times it **interleaved**, context by
context, in the same run.

### What is left to try, in order

Headroom, measured minus each block's own roofline, on the corrected table:

| block | measured | roofline | available | of a step |
|---|---:|---:|---:|---:|
| **attention** | 5.05 ms | 2.47 | **2.58 ms** | **14.9%** |
| routed expert projections | 4.03 | 3.06 | 0.97 | 5.6% |
| dense matvecs incl. LM head | 2.96 | 2.34 | 0.62 | 3.6% |

**1. Attention.** More headroom than the other two combined, and the only block
more than 2× off its floor. The per-key chain is load → dot → **`simd_sum`** →
online-softmax update → rescale → accumulate. Unrolling the loads did not pay
and split-K bought 1.06× in situ, which together point at the cross-lane
reduction and the dependent softmax update rather than at memory.
`sdpa_paged.metal`'s own comments name the alternative: **KEY_PER_LANE** — one
key per lane walking all of D, removing the `simd_sum` per key at the cost of D
registers per lane. A restructure, not a constant, and the last untried shape.

**2. The routed FFN**, 0.97 ms available across three projections.

**3. The dense matvecs**, 0.62 ms. Nine dispatches behind one pipeline, so a fix
would reach all of them at once — but there is little to reach. If it is
attempted, split the 2.96 ms across the nine first: the LM head is K=2048 ×
N=151936 and runs ONCE, the others are narrow and run 48×, and no ablation token
can separate them.

**Measure any candidate the way split-K had to be measured**: the isolated
kernel ratio overstated the transferable gain by 2.8×, so price it with
`tools/split_fire_ab.sh` before believing an end-to-end number.

---

## 3. Other open items, in priority order

1. **The mod-8 driver cliff, now relatively worse than ever.** A fire whose row
   count is `r mod 8` in 1..6 pays a flat penalty: 189 rows costs **868.9 ms**
   against 184 rows at **249.9** — +619 ms, ~3.5× the base cost where it used to
   be 2×. Making the kernels faster made this defect proportionally far more
   damaging. The guest steers around it (`aligned_prefill_chunks`); the driver
   cause has never been found.
2. **~127 ms of a cached agentic turn is outside the driver entirely.** A cached
   turn's 403 ms TTFC breaks down as ~255 ms prefill fire + ~18 ms first-token
   decode + ~3 ms submit, leaving ~127 ms in the guest render, the shim, the
   gateway and HTTP. Nothing has attributed it.
3. **pie strategy B fails under concurrent load** (10 of 32 completed, 22 HTTP
   500). Unchanged and undiagnosed.
4. **The NAX kernels ignore a user attention mask.** A mask is per-fire and
   neither selection site is handed it, so it cannot be gated on. This inherits
   the assumption `sdpa_paged_decode..._p32` (FAST_FULL) already shipped with;
   pre-existing, not introduced, but if masks are ever enabled on llama several
   kernels are wrong together.

---

## 4. Method rules that must survive compaction

1. **Correctness before rate, always.** A NAX kernel was once measured at 1.652
   ms/layer and reported three times before a CPU reference showed it computed
   the wrong thing. A timing cannot tell correct attention from a transposed
   operand.
2. **Compare a new kernel to the one it REPLACES, not to exact arithmetic.** The
   NAX attention changes serving output (332 tokens where there were 382). That
   is only defensible because its error against a float64 reference is
   indistinguishable from the old kernel's.
3. **Re-trace before choosing a target.** The prefill ordering has rotated three
   times: attention → routed GEMM → attention. Every time a term is fixed the
   ordering changes.
4. **A single agentic run cannot price a change.** Three reps per arm on
   `django__django-14373` gave 3/4/5/19 turns and a 0-byte patch in one rep of
   BOTH arms. Use the deterministic instruments; use `tools/pie_ab.sh` when an
   agentic number is unavoidable.
5. **A four-arm benchmark carries ~10% of THERMAL drift at the long end.** The
   first arm repeated last is 11–12% slower at 28k, on an idle machine.
   Interleave if that margin matters.
6. **Verify a geometry is covered before trusting a correctness claim.** Every
   NAX correctness shape was gqa 8 until `llama_numerics_test`'s gqa 2 surfaced
   a failure that turned out to be a routing tie — but only checking settled it.
7. **Early-return bisection is invalid where the removed code has no other
   consumer.** Dead-code elimination deletes everything upstream whose only sink
   was past the return, so every cut point measures the same thing and the cost
   appears to arrive all at once at the last one. Found here on
   `moe_route_sort`. Vary one thing at a time with every result still
   observable, in an isolation probe.
8. **A dispatch trace inflates SMALL kernels.** It brackets every dispatch with
   timestamps, a fixed cost, so a kernel whose real cost is near the launch floor
   can be overstated by tens of times — `silu_mul` by 38×. Price small kernels by
   ablation, and cross-check any trace share below ~10% before acting on it.
9. **`PIE_METAL_ABLATE` takes the kind `pso_kind` MAPS TO, and that map is
   many-to-one.** Nine dense matvecs share `qmv_gate` (q, k, v, o, gate, up,
   down, router AND the lm head); three routed projections share
   `ll_expert_gate`. So one token ablates a whole group, and a token for a kind
   that is never a `pso_kind` ablates nothing while printing an armed-looking
   banner. Label a composition row by what was REMOVED, not by the token.
   Deliberately ablating known-zero kinds is also free calibration: five of them
   here bounded the instrument at +/-0.09 ms.
10. **Prove the binary contains the change, and that BOTH arms ran, before
   believing an A/B.** Two arms of the same binary produce a perfect 1.00x,
   identical output and a clean drift control — the most convincing null result
   available, and it means nothing. Separately, an arm that fails to boot prints
   nothing and leaves the other arm's numbers sitting under both headings; a run
   that produced no measurement must be fatal, not blank. `cargo build -p pie` does NOT build the server; the crate is
   `pie-bin` and the wrong name fails the package match rather than building.
   Build inside the harness, assert no driver source is newer than the binary,
   and make the gate say out loud that it fired.
11. **A probe that has not shown its clocks are stable has not measured
   anything.** This GPU ramps under sustained load, and the same kernel on the
   same binary read 1.475 / 1.154 / 0.837 / 0.233 ms/layer across four runs
   ordered by how warm the machine was. Ratios taken across a ramp are as bad
   as absolutes: split-K read as a consistent LOSS on two half-ramped runs and a
   1.16x win once settled. Warm until successive measurements agree, and put a
   physical sanity check on the output — attention cannot get cheaper on more
   keys, and that guard caught every bad run here.
12. **An instrument can reproduce the bug it was built to find.** A tile-width
   sweep hardcoded the launch shape while varying the kernel constant and
   reported a correct kernel as 24576 elements wrong. A roofline gate parsed
   `$NF` and compared the string `"GB/s"` against 250, passing for every machine
   state.

---

## 5. Commands

```sh
# build the probes and tests
cmake -S driver/metal -B /tmp/metaltools -DPIE_METAL_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/metaltools -j 8

# the decode composition (this file's measurement)
integrations/opencode/tools/boot_pie.sh dec PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b \
  PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096 \
  PIE_METAL_DISPATCH_TRACE=1 PIE_METAL_TRACE_STRIDE=8
# ...then one long-prompt request; read /tmp/pie_dec.log for [trace] tables

# fire cost vs row count, through the real engine and driver
cd runtime/engine/tests/inferlets && cargo build --target wasm32-wasip2 --release -p decode-rows-probe
./target/release/pie -c /tmp/rows-probe/config.toml run \
  --path runtime/engine/tests/inferlets/target/wasm32-wasip2/release/decode_rows_probe.wasm \
  --manifest runtime/engine/tests/inferlets/decode-rows-probe/Pie.toml

# the four-way, and the deterministic rate probe on any engine
bash integrations/opencode/tools/four_way.sh
python3 integrations/opencode/tools/rate_probe.py --base-url URL --model M --label L
```

## 6. The results files, and what each is for

| file | what it holds |
|---|---|
| `docs/HANDOVER.md` | the wider state; read first |
| `integrations/opencode/results-four-way.md` | the four-engine comparison, prefix-cache analysis, drift control |
| `integrations/opencode/results-prefill-experiments.md` | every prefill experiment **including the two that failed**, the per-fire cost attribution, and the decode trace |
| `integrations/opencode/results-nax-prefill.md` | the NAX attention landing |
| `integrations/opencode/results-head-sharing-decode.md` | the GQA decode kernel, and the agentic-variance finding |
| `integrations/opencode/results-e2e-one-instance.md` | the original three-engine baseline, **with its correction** |
