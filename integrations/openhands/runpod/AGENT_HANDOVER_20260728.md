# Agent handover — 2026-07-28

> Supersedes `AGENT_HANDOVER_H200.md` and `RUN_STATE.md` for **findings**. Their
> machine-state and storage rules still apply. `RUN_STATE.md` §0 and
> `AGENT_HANDOVER_H200.md` §1 are now **partly wrong** — see §3 below before you
> act on anything either says about xqa or the A100 arm.

## 1. The thirty-second version

The experiment set out to explain why `litellm+vLLM` beat `pie-openhands` by
~10-16×. **It was one inverted conditional in the CUDA driver**, which disabled
flash-decoding for exactly the batch sizes that need it most. Fixed in
`d9aaf9f1`.

| | before | after |
|---|---|---|
| Pie s/iter (SWE-bench, 1 instance) | 7.72 | **1.07** |
| Pie decode | 15.2 tok/s | **154.7 tok/s** |
| gap to vLLM `fair` | ~12× | **1.34×** (1 instance; §4b puts it at 1.73× over 13) |

Everything is committed and pushed to `origin/openhands-integration-updated`
(`YangLiuWillow/pie`). Nothing went near upstream.

## 2. The fix

`driver/cuda/src/ops/attention_flashinfer.cu` overrode FlashInfer's own
`split_kv` decision whenever `batch_size <= 512` on sm_80+. The memory planner
caps `max_forward_requests` at 512, so **the override fired on every decode this
driver has ever run** and FlashInfer's decision was never used at any batch size.

The condition is backwards relative to its own rationale. Without a KV split,
decode parallelism is roughly `batch_size * num_kv_heads` CTAs:

| batch | CTAs on a 132-SM H200, 4 KV heads | non-split appropriate? |
|---|---|---|
| 1 | 4 (~3% of device) | **no — catastrophic** |
| 66 | 264 (~2 waves) | yes |
| 512 | 2048 | yes |

Low batch is precisely when the split is needed, and "the TP1 latency shapes we
care about" *are* low batch.

Decode rate vs context (H200, Qwen3-Coder-30B-A3B, batch 1, page_size 32):

| ctx tokens | before | after |
|---|---|---|
| 298 | 157 tok/s | 166 |
| 4,558 | 48 | 160 |
| 18,218 | 14 | 137 |
| 27,318 | **10** | **125** |

Old behaviour is still reachable with `PIE_CUDA_SMALL_BATCH_NONSPLIT_DECODE=1`,
because this changes decode scheduling for every sm_80+ shape and the evidence
is one model on one GPU. If you find a shape where non-split wins, record the
measurement next to `legacy_small_batch_nonsplit_enabled()`.

**This was not a new diagnosis.** `benches/MULTIMODAL_BENCH.md:92` recorded it in
June as *"a real perf bug for ALL long-context decode, not just multimodal"* and
worked around it by baking `PIE_FLASHINFER_FORCE_SPLIT_KV_SMALL=1` into
`benches/pie_mm_bench.py:168`. Nothing else in the tree set it. Their controlled
test topped out at **~900 KV tokens**, where the fix is worth ~8% — which likely
explains why it stayed a per-benchmark opt-in for two months. At 27k it is 12.8×.

**Still to do:** file this upstream (`pie-project/pie` appears to have no issue
for it; `gh` is not installed on the pod so this was not confirmed), and drop
the now-redundant env var from `benches/pie_mm_bench.py:168`.

## 3. What is now VOID — read before trusting the older handovers

**xqa is worth nothing for this model, and the H200 migration's premise was
wrong.**

- A context sweep with `xqa_decode=on` vs `off` is **identical at every length**
  (27,318 tokens: 101.6 vs 103.4 ms/token).
- Reason: `model_type = "qwen3_moe"` routes through the **qwen3_5** code
  (`entry.cpp:488-490`), and `qwen3_5_forward.cpp` / `qwen3_5_moe_forward.cpp` /
  `qwen3_5_moe_model.cpp` contain **zero** references to xqa (`llama_like.cpp`
  has 18). The `xqa_decode=on` banner prints the *llama_like* forward config,
  which this model never uses.
- Therefore `PIE_CUDA_KV_PAGE_SIZE=32` buys nothing here, and **the A100 Pie arm
  was voided over a banner describing a code path Qwen3-Coder-30B-A3B does not
  take on either GPU.** The real limiter (split-KV) was present on both.

That does not retroactively validate the A100 numbers — it means the stated
reason for discarding them was wrong. The A100 also lacks the split fix's
benefit only in the sense that it was never applied there; re-measuring on A100
is now a legitimate question rather than a settled one.

**Also void:** `AGENT_HANDOVER_H200.md` §5's "REQUIRED first edit" is done, and
its three-tier scope note is superseded — see §6.

## 4. Current measured state

> **The single-instance table below is superseded — read §4b.** The 13-instance
> run puts the median-latency ratio at 1.73×, not 1.15×/1.34×.

Head-to-head, `django__django-14373`, temperature 0, same pod, same harness,
both arms fresh (2026-07-28 20:30):

| | Pie (cuda_native) | vLLM (fair) | ratio |
|---|---|---|---|
| s/iter | **1.07** | **0.80** | 1.34× |
| median call latency | 1.38 s | 1.20 s | 1.15× |
| effective tok/s | 135.6 | 187.3 | 1.38× |
| iterations | 40 | 38 | — |
| wall | 43 s | 30 s | — |
| max prompt tokens | 20,465 | 22,628 | cap 131,072 |
| errors | none | none | — |

Pie's own attribution (`_metadata.pie_timings`, now in every row):

| phase | share | median |
|---|---|---|
| decode | **87.7%** | 1016 ms |
| prefill | **11.7%** | 162 ms |
| render + hash + open + save + setup | 0.4% | — |
| transport | 0.2% | 1.4 ms |

KV reuse **92.79%** (20,462 prefilled of 283,618 rendered), vs vLLM APC's ~95.1%
on the A100 arm. Comparable — Pie's prefix cache is working.

**One instance with divergent trajectories (40 vs 38 iterations) cannot support
a 1.34× claim.** Treat it as indicative. Accuracy is still unmeasured and cannot
be measured on a GPU pod (no docker/apptainer/swebench).

### 4b. Superseded by the 13-instance run — the ratio is 1.73×, not 1.34×

`40_concurrency_sweep.sh` ran four arms back-to-back (21:09–22:07). All 13
instances in all four arms: **rc=0, zero errors, zero empty patches, zero
0-iteration failures, zero stuck-retries.** Predictions in
`predictions/ab_h200_{pie_auto_p32,litellm_fair}_c{1,8}_20260728_*.jsonl`,
logs in `logs/sweep_20260728_210923/`.

| arm | s/iter | med lat | p90 | p99 | max | inst/GPU-hr | agg tok/s |
|---|---|---|---|---|---|---|---|
| pie c1 | 1.265 | 1.21 | 4.59 | 9.70 | 14.1 | 42.9 | 99.5 |
| litellm c1 | 1.566 | **0.70** | 2.94 | 7.47 | **303.6** | 33.7 | 75.8 |
| pie c8 | 4.098 | 3.00 | 10.74 | 26.9 | 53.1 | 64.1 | 141.6 |
| litellm c8 | 1.765 | **1.76** | 7.26 | 15.4 | 19.2 | **161.4** | **530.4** |

**On serving latency vLLM wins at every percentile through p99** — 1.73× on the
median at c1, not the 1.34× §4 reported from one instance. §4's number came from
`django__django-14373`, which happens to be one of Pie's better instances.

**Do not read Pie's better `s/iter` and `inst/GPU-hr` at c1 as a win.** Both come
from wall clock, and litellm-c1 has two instances (`django__django-13089`,
`django__django-16485`) whose walls are ~340-360 s against a ~50 s cohort. That
is not tool time — `sum(response_latencies)` is 331 s and 348 s, i.e. essentially
all of it, and **one single call took 303.55 s**. One anomalous vLLM stall is the
entire margin. Excluding those two instances litellm-c1 is faster than pie-c1 on
every instance. The stall is unexplained and worth one look before anyone cites
either arm's wall clock; `s/iter` is only trajectory-robust against *iteration
count*, not against a five-minute outlier call.

## 5. Where the remaining gap lives (single-stream)

> Written against the 1.34× single-instance figure; §4b revises the ratio to
> 1.73× and §6a shows it becomes 2.52× at concurrency 8. The two causes below
> still hold and the second one now matters more, not less — the *shares* are
> what is stale, not the mechanisms.

Two causes, both measured:

1. **Decode is ~17% slower per token (~83% of the gap).** Pie's *decode-only*
   rate (154.7 tok/s) is below vLLM's *end-to-end* rate (187.3). Closing decode
   alone would take 1.34× → ~1.05×. `MULTIMODAL_BENCH.md` independently measured
   this on **different hardware and a different model** (L40, Qwen3-VL) and
   concluded decode is *"weight-bandwidth-bound, not slow… ~15% behind vLLM at
   most; near the HBM floor."* Two unrelated benchmarks landing on ~15-17% is
   strong evidence this is a real property, not a bug. Do not expect a one-liner.

2. **Small prefills under-occupy (~17% of the gap).** ~1,077 genuinely-new
   tokens per turn run at ~5,250 tok/s, against ~20,400 tok/s when the prefill
   is large (27,318 tokens in 1.338 s). Same occupancy story as the decode bug,
   one phase over.

Both trace to one root property: **Pie issues one fire at a time on one stream**,
so any phase that does not individually fill the GPU runs at a fraction of peak.

**Amended by §6a:** "one fire at a time" is right about the *stream*, wrong if
read as "one request per forward" — with concurrent traffic Pie does merge
requests (mean R=2.63 at c8, R up to 7). The problem is not that batching is
absent; it is that it forms small batches and amortizes them poorly.

## 6. What to do next

### 6a. The concurrency question — ANSWERED, and the answer is bad

**Concurrency widens the gap. It does not close it.** Scaling c1 → c8:

| | c1 | c8 | scaling |
|---|---|---|---|
| pie inst/GPU-hr | 42.9 | 64.1 | **1.49×** |
| litellm inst/GPU-hr | 33.7 | 161.4 | **4.79×** |
| pie agg tok/s | 99.5 | 141.6 | 1.42× |
| litellm agg tok/s | 75.8 | 530.4 | 7.00× |

At c8 vLLM delivers **2.52× Pie's throughput** on the same 13 instances. The
single-stream 1.73× becomes 2.52× under load.

**The mechanism, measured — Pie batches, but the batches are small and the
amortization is weak.** The driver logs `req_id=… R=…` on a 1-in-100 sample of
forwards (`executor.cpp:3183`, `handled % 100`), which is enough for a
distribution:

| | c1 | c8 |
|---|---|---|
| logged forwards | 1080 | 389 |
| R distribution | **100% R=1** | R=1 168, R=2 44, R=3 88, R=4 22, R=5 13, R=6 27, R=7 27 |
| mean R | 1.00 | **2.63** |

Cross-check, and it is a tight one: 389 × 100 = 38,900 forwards for 100,708
generated tokens = **2.59 tokens/forward**, against a mean R of 2.63 from the
sample. The sampling is sound and the batching is real.

**Corrected derivation.** The 1.85×-per-forward figure below came from makespan,
which is contaminated by tool time and by litellm's 303 s stall. The clean route
uses only `pie_timings` decode totals: each decoded token waits exactly one
forward, so `t_forward = decode_total_s / tokens`. That gives **7.36 ms at R=1**
(783.8 s / 106,449) and **18.48 ms at R=2.63** (1860.9 s / 100,708) — 2.51× the
time for 2.63× the work, i.e. **4.8% amortization**. Near zero, which is exactly
what a `cublasGemmBatchedEx` M=1 fallback predicts: each route is an independent
GEMV, so cost is linear in routes by construction (§6a-bis).

vLLM's own log reports mean **5.77** concurrent requests at c8 (22 time-samples,
mostly 4–8) against Pie's 2.63. Normalizing to marginal cost per added request:

| | mean batch | per-stream slowdown | marginal request costs |
|---|---|---|---|
| pie | 2.63 | 2.25× | **0.78** of a solo request |
| litellm | 5.77 | 2.63× | **0.34** of a solo request |

So vLLM amortizes ~2.3× better per added request. Note the trap: raw per-stream
*retention* is 0.44 for Pie against 0.38 for vLLM, which reads as Pie scaling
better and is meaningless — the two run at different batch sizes, and retention
does not normalize for that.

So at concurrency 8 Pie issues **2.8× fewer forwards** but each takes longer.
Net: 1.42×. Two distinct losses, do not conflate them:

1. **Mean batch is 2.6, not 8.** Agents spend real time in tool execution, so
   the eight conversations are rarely all decoding in the same window.
2. **The marginal request costs 0.78 of a solo one** (vLLM: 0.34). This is now
   explained rather than hypothesized: at `routes ≈ 21` the MoE decode is still
   below the aligned path's gate, so it runs batched GEMVs whose cost is linear
   in routes. Raising the gate does not fix it — `aligned8` measured 9.5%
   *slower* at batch 1 (§6a-bis). The aligned path wins only where its gate
   already puts it, so closing this needs a kernel that amortizes at low route
   counts, not a threshold change.

**The `7.47× at N=8` figure in the pre-2026-07-28 notes does not transfer, and
should not be quoted again.** `logs/probe_conc.log` shows what it measured: eight
streams held in lockstep steady-state decode, producing genuine `R=8 N=8
sampled=8` forwards. Real agent traffic never holds that shape. This is the §8
lesson in a new costume — a microbenchmark reporting a scaling factor is not that
scaling factor being available to the workload.

Next lever, if this is pursued: find out why mean R is 2.6 and whether the
scheduler can hold requests briefly to build larger batches, and separately why
R=2.6 buys only 1.4×. The second is the bigger prize — fixing batch *formation*
alone caps out at ~8/2.63 = 3× more work per forward, which at the current
amortization curve is worth far less than 3×.

<details><summary>Original §6a text, now superseded — kept for the reasoning</summary>

The overlap lever is a **throughput** lever, not a single-stream one. In the
current A/B there is exactly one conversation, and turn N's prefill must precede
turn N's decode — that 11.7% is serial by construction and no scheduling change
removes it from a single-agent latency measurement.

Where it matters is concurrency, and the headroom is already measured: a
concurrency sweep at 27k context showed **7.47× aggregate scaling at N=8 with
per-stream latency flat** (109 → 113 ms/token).

**Run the SWE-bench A/B at concurrency 4-8, both arms, and report
instances/GPU-hour** (`TEST_PLAN.md` already names that as the throughput
metric). Single-stream we are at 1.34×; concurrent is where Pie's scheduling
either closes or widens the gap, and nobody knows which.

Note a refinement to `MULTIMODAL_BENCH.md`'s proposal: it scopes a **second CUDA
stream** because vision encoding cannot merge into an LLM forward pass. Prefill
and decode *can* merge — that is exactly what vLLM does (`v1/core/sched/
scheduler.py:398-407`: *"There's no 'decoding phase' nor 'prefill phase' in the
scheduler"* — one forward pass carries both). So for this workload the lever is
likely **unified/chunked-prefill scheduling**, not a second stream.

</details>

### 6a-bis. Decode: the MoE kernel-selection lever is CLOSED (measured)

`qwen3_5_moe_forward.cpp` picks a decode path off `routes = batch * top_k`
(top_k=8 here). At batch 1 `routes=8`; at the concurrency these arms reach
(mean R=2.63) `routes~21`. Both are far below the aligned path's `min_routes=64`
gate, and the WMMA path is off by default — so **every decode in this study ran
the `cublasGemmBatchedEx` M=1 fallback**, cuBLAS's worst shape.

That looked like free performance. It is not. `41_moe_decode_sweep.sh`,
one instance, temperature 0, all rc=0 with no errors:

| variant | decode tok/s | max prompt | vs base |
|---|---|---|---|
| base (cuBLAS M=1) | **147.5** | 25,520 | — |
| p16 (page_size 16) | 147.3 | 28,152 | ~0 (10% longer ctx, same rate) |
| wmma | **87.2** | 33,416 | **-41%** |
| aligned8 (`min_routes=8`) | **133.5** | 28,073 | **-9.5%** |

**Both gated-off kernels are worse at batch 1, so the defaults are right and the
gates are correctly tuned.** Trajectories diverged (contexts differ up to 31%),
but the context curve in §2 costs only ~9% per 50% more context, so neither the
41% nor the 9.5% deficit is a context artifact. Page size 16 vs 32 is a wash,
independently confirming §3's "page size buys nothing here".

**The decode gap is not reachable by configuration.** Everything env-tunable has
now been tried. What remains is kernel work.

### 6a-ter. Where decode time actually goes — and why the profiler misleads

`PIE_QWEN35_MOE_PROFILE=1` puts `full_attn` at 47% of decode kernel time and MoE
GEMMs at 32% (390 forwards, median KV 21,001). **Do not plan off those shares.**
The profiler syncs every stage, which suppresses overlap and blocks graph
capture; its leaves sum to 8.93 ms against ~6.8 ms unprofiled, and it inflates
attention hardest. It also depresses throughput enough to trip a websocket
timeout — that run died after 11 calls, `rc=0`, `iters=0`, with a decode rate
that looked perfectly legitimate. Never read a rate off a profiled run.

The profiler-free decomposition is the **slope of decode cost against context**,
which needs no instrumentation at all. From §2's post-fix sweep:

| ctx | tok/s | ms/token |
|---|---|---|
| 298 | 166 | 6.02 |
| 4,558 | 160 | 6.25 |
| 18,218 | 137 | 7.30 |
| 27,318 | 125 | 8.00 |

Slope 4,558 → 27,318: 1.75 ms for 22,760 KV tokens = 2.24 GB, i.e. **~1.28 TB/s
effective KV read** — about 27% of an H200's ~4.8 TB/s. Intercept: **~6.0 ms of
context-independent cost**, which at ~6.6 GB of active weights is **~1.1 TB/s**,
also ~23% of peak.

So at 25k context the split is roughly **72% fixed / 28% attention** — the
reverse of what the profiler reported, and it means the dominant decode cost is
the weight/MoE/router/launch path, not attention. Both components run at about a
quarter of the machine's bandwidth.

### 6a-quater. The decomposition, measured on both arms

`42_context_sweep.sh` + `context_sweep_client.py`, 6 contexts × 3 reps, both
arms, same prompts, same pod (`logs/ctxsweep_20260728_234433/`).

**Method.** Same prompt generated short (8 tokens) and long (136), subtracted:
`(lat_long - lat_short) / (tok_long - tok_short)`. Prefill, render, transport and
process launch are generation-independent, so they cancel — which is what makes
two entirely different client stacks comparable. A warmup call per context leaves
both measured calls equally prefix-cached, otherwise the subtraction would
quietly return decode *minus* prefill. **Validated:** Pie's differenced number
tracks its own `timings.decode_ms` within 1.5% at every one of six contexts.

| prompt tokens | pie ms/tok | vllm ms/tok |
|---|---|---|
| 1,036 | 5.45 | 4.18 |
| 3,988 | 5.88 | 4.28 |
| 8,020 | 5.93 | 4.40 |
| 16,012 | 6.24 | 4.61 |
| 24,004 | 7.06 | 4.81 |
| 31,996 | 7.20 | 5.01 |

| | pie | vllm | ratio |
|---|---|---|---|
| intercept (fixed per-token) | 5.494 ms | 4.173 ms | **1.32×** |
| slope (per 1k KV tokens) | 0.0563 ms | 0.0265 ms | **2.12×** |
| implied KV read | **1,747 GB/s** (36% of peak) | **3,708 GB/s** (77% of peak) | 2.12× |
| R² | 0.952 | 0.998 | — |

Gap decomposition, modelled from the fits:

| ctx | pie | vllm | ratio | gap = fixed + attention |
|---|---|---|---|---|
| 16k | 6.39 ms (156 tok/s) | 4.60 ms (218) | 1.39× | 1.80 = **1.32 (74%)** + 0.48 (26%) |
| 25k | 6.90 ms (145 tok/s) | 4.84 ms (207) | 1.43× | 2.07 = **1.32 (64%)** + 0.74 (36%) |
| 32k | 7.29 ms (137 tok/s) | 5.02 ms (199) | 1.45× | 2.27 = **1.32 (58%)** + 0.95 (42%) |

The modelled 1.43× at 25k sits under the A/B's measured 1.59× (§6a), as it
should — the sweep isolates decode, the A/B carries per-call overhead too.

**What this says to do.**

1. **Attention is the recoverable half.** Pie reads KV at 1,747 GB/s; vLLM hits
   3,708 GB/s *on the same GPU, same model, same contexts*, so ~77% of peak is
   demonstrably achievable here and Pie is leaving 2.12× on the table. This is
   the only part of the decode gap with a proven target.
2. **The fixed 1.32× is the larger term but the harder one.** It is weights +
   MoE + router + launch, and §6a-bis already showed kernel *selection* cannot
   move it — all three MoE decode paths were tried and the default won.
3. Attention's share grows with context (26% → 42% from 16k to 32k), so it gets
   more valuable as agent histories lengthen, which is the direction this
   workload goes.

**And a third correction to the profiler.** It reported `full_attn` at 47% of
decode. The slope says attention is 1.41 ms of 6.90 ms at 25k — **20%**. The
profiler overstated it by ~2.4×. Three hypotheses in this study have now been
wrong (MoE weights at 82%, attention at 47%, batching better than vLLM's), and
each was corrected by a measurement that cost minutes. Measure first.

Pie's fit is visibly noisier than vLLM's (R² 0.952 vs 0.998; residuals to
±0.22 ms, with 24k sitting high). Worth one look before micro-optimizing against
the slope — it may be a page-boundary or split-KV threshold effect rather than
noise.

### 6b. `Context::suspend()` during tool execution — the Pie-only angle

SWE-bench agents spend seconds to minutes per iteration running pytest with KV
resident and the GPU idle. `RUN_STATE.md` §4 flagged `suspend()` as a throughput
win; it has never been tested. vLLM has no equivalent — it must keep blocks or
evict-and-recompute. This is a **capability** claim rather than a speed claim.

### 6c. Deferred, with reasons

- ~~**Full 13-instance A/B**~~ — **done**, see §4b. It moved the headline from
  1.34× to 1.73×, which is exactly why this gate existed. Keep the gate for the
  next ratio anyone wants to publish.
- **Stage 2 (delta rendering / live `Context`).** Its targets — render, hash,
  open, save — measure **0.4% combined**. Worth building as a programmability
  demonstration (the human's stated reason), **not** as a performance fix. Do
  not let the 11.7% prefill number pull you into justifying it: that is real
  compute on new tokens, and delta rendering does not remove it.
- **Profiling the 17% decode gap.** Needs `ncu`, which is **blocked on this pod**
  (`ERR_NVGPUCTRPERM`; `RmProfilingAdminOnly: 1`, and the kernel module cannot be
  reloaded from a container). Needs a pod created with `--cap-add=SYS_ADMIN`.

## 7. Machine state and how to resume

```bash
source /workspace/pie-bench-env.sh
cd /workspace/pie/integrations/openhands/runpod
```

- `/workspace` (MooseFS network volume) persists: repo, model (~57 GB),
  predictions. `/root` is **wiped when the pod is replaced** — venvs, cargo tree,
  git credentials.
- The storage rule from `AGENT_HANDOVER_H200.md` §3 still holds absolutely:
  small-file work on `/workspace` **stalls indefinitely with no error**.
- `00_setup_h200.sh` is idempotent and now self-heals two things the 2026-07-28
  image broke: it provisions Python 3.12 via `uv` (the image is conda-based with
  3.11 and no 3.12) and builds venvs with `uv venv --seed` (`$PY -m venv` fails
  on a uv-provisioned interpreter — ensurepip exits non-zero).
- §8 now runs `pie driver cuda-native doctor` first — answers "is cuda_native in
  this binary and can it see the GPU" in under a second, versus the ~3 minutes
  the banner probe spends loading 58 GB before failing the same way.
- **git identity is not set on a fresh pod.** Set it locally to match history:
  `git config user.name yangliuwillow; git config user.email liu.yang.ly337@yale.edu`.
- **No `gh`, no credentials.** Pushing needs a PAT. Do not paste one into the
  session transcript — export it in your own shell, or use a deploy key.

### Running an arm

```bash
# one instance, pie (split-KV is now the default — no env var needed)
AB_INSTANCES=django__django-14373 PIE_CFG_VARIANT=auto_p32 ARM=pie bash 30_ab_run.sh
# vLLM fair
AB_INSTANCES=django__django-14373 VLLM_TIER=fair ARM=litellm bash 30_ab_run.sh
```

`AB_INSTANCES` narrows the set without editing the 13-instance array.
`PIE_CFG_VARIANT` ∈ `auto_p32 | latency_p32 | auto_p16` selects the toml **and**
the `PIE_CUDA_KV_PAGE_SIZE` that variant needs — the toml key is inert on the
planner path, so the env var is the only real control.

The Pie arm runs on a **persistent daemon** by default now (one inferlet process
and websocket per conversation, not per call). `--pie-oneshot` restores the old
launch-per-call transport so its cost stays measurable.

## 8. Cautions

- **`nvidia-smi utilization.gpu` is not SM occupancy.** It reports the fraction
  of time at least one kernel was resident. A single-CTA kernel grinding through
  2.7 GB reads 97%. This misled the investigation for a while.
- **A banner reporting a feature is not that feature being on.** The
  `xqa_decode=on` line describes a config struct this model never uses (§3). The
  repo's existing lesson — *"trust what the driver prints over what the config
  says"* — needs a corollary: *check the driver is printing about the code path
  you are actually running.*
- **`pgrep -f` / `pkill -f` match your own shell.** Cost time again this session.
  Kill by PID; verify with `ss -ltnp` and `nvidia-smi`, not `pgrep`.
- **Four separate env toggles produced null results** before the real cause was
  found (`PIE_CUDA_PREFILL_DECODE_MIN_KV_PAGES`, `..._NOGRAPHS`, page size, an
  added decode-graph flag). All four configured the **llama_like** path, which
  `qwen3_moe` does not take. If a toggle changes nothing, suspect it is not
  reaching the executed code before concluding the mechanism is innocent.
- **Read-then-assert is the failure mode of this session.** Three code-reading
  claims were wrong and each was caught by the human, not by me: that
  `qwen3_moe` used `prepare_llama_like_decode_plan`; that `full_attn_min_R`
  routed decode to a recompute path; and that Pie's prefix hits are "restored
  from a named snapshot" (they are handle operations on GPU-resident pages —
  `open`/`save` medians are 0.3 ms, which cannot move 2 GB). **The measurements
  held; the narration around them did not.** Prefer a number over a reading.

## 9. Commits

Pushed to `origin/openhands-integration-updated` (`YangLiuWillow/pie`):

| commit | what |
|---|---|
| `d9aaf9f1` | **the fix** — default to flash-decoding instead of force-disabling split-KV |
| `9d77878e` | persistent daemon transport, always-on `pie_timings`, `Prediction`/`Result` field fix |
| `da0cc551` | H200 bring-up repairs for the new image, `doctor` preflight, config variants, runner params |

`9d77878e` also fixed a latent bug worth knowing about: `_extract_metrics` has
returned `max_prompt_tokens` / `prompt_tokens_per_call` since 2026-07-27, but
neither `Prediction` (`swe_bench.py`) nor `Result` (`humanevalfix.py`) had the
fields in this tree, so `Prediction(**metrics)` raised `TypeError` and **every
instance was recorded as a 0-iteration failure with an empty patch.** The
matching version lived on the previous pod's local disk and went with it.
