# pie vs vLLM-metal vs mlx-lm: same weights, same agent, two SWE-bench instances

**Date:** 2026-08-14. **Machine:** Apple M5 Pro, 48 GB (mildly contended — a peer
session held ~12.5 GB of GPU work throughout).
**Weights:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` — the *same
artifact* on all three engines. **Agent:** stock opencode, temperature 0.

- **pie**: strategy B (`opencode-session`), with the day's fixes — aligned
  prefill chunks, quantized page pool, `max_forward_tokens=4096`.
- **vLLM-metal**: 0.27.0 + `vllm_metal` plugin, prefix caching on.
- **mlx-lm 0.31.3**: `mlx_lm.server`, its own prompt cache.

llama.cpp was deliberately excluded: it needs a GGUF, which is a different
quantization, so any result would confound engine with quantization.

## 1. Speed — 6-turn canned opencode replay

| engine | total | cold TTFC (7.2k prompt) | steady turn | steady TTFC |
|---|---:|---:|---:|---:|
| mlx-lm 0.31.3 | **9.22 s** | **4.72 s** | 0.73 s | 0.49 s |
| vLLM-metal | 10.17 s | 5.99 s | **0.65 s** | **0.45 s** |
| pie strategy B | 17.02 s | 11.0 s | 0.97 s | 0.73 s |

pie is last, and the gap is prefill: 2.3x slower than mlx-lm to first token on a
cold 7.2k prompt. Consistent with everything else measured this week — pie's
decode is competitive, its prefill is not.

## 2. Correctness — SWE-bench Verified, first 2 of the known-solvable 5

Graded by `swebench.harness.run_evaluation` in Docker (colima), dataset
`SWE-bench/SWE-bench_Verified`.

| engine | resolved | empty patches | wall | 12276 | 13028 |
|---|---:|---:|---:|---|---|
| **pie** | **1/2** | 1 | 526 s | 356 s, 1719 B, **resolved** | 161 s, empty |
| **mlx-lm** | **1/2** | 1 | 1856 s | **47 s**, 411 B, **resolved** | timeout @1800 s, empty |
| vLLM-metal | **INVALID** | — | 154 s | server died | server died |

> **The vLLM-metal arm of THIS run is void and must not be quoted.** Its server
> died mid-run with `[METAL] Command buffer execution failed: Insufficient
> Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` -> `EngineDeadError` ->
> HTTP 500, and opencode recorded `Cannot connect to API ... exit 1` on both
> instances after 80 s and 65 s. It never ran the model. The cause was almost
> certainly GPU contention: a peer session held ~12.5 GB throughout, and vLLM's
> footprint on top of that exceeded the device.
>
> This is exactly the failure mode the boot helpers exist to catch, and it got
> through because liveness was proved at BOOT (`/v1/models` answered in 22 s)
> and never re-checked after. A server that is up at t=0 and dead at t=60 looks
> identical in the summary. Every future arm now greps its own log for engine
> death before its numbers are reported.
>
> The earlier 5-instance vLLM result (1/5 resolved) came from a separate run and
> is unaffected.

**`django__django-13028` is non-discriminating.** All three produced an empty
patch, and pie produced an empty patch on it in the earlier 5-instance run too.
So the entire signal is `django__django-12276`.

On that one instance: pie and mlx-lm both resolve it. vLLM is unmeasured here. pie's earlier 5-instance run also resolved 12276 (545 s,
411 B), so pie's result reproduces across runs with a different patch (1719 B).

## 3. What the wall-clock column does NOT say

mlx-lm has the **longest** total (1856 s) while being the **fastest** per turn.
It burned the full 1800 s cap on 13028, where pie gave up after 161 s. So
"total benchmark wall" measures how long an agent persists on an unsolvable
case at least as much as it measures engine speed. Read the per-instance
column, not the total.

The same caution kills the tempting read of vLLM's 154 s: it is not fast, it is
absent — 0 patches on 2 instances, matching the earlier 5-instance result
(1/5 resolved, mean 2.8 model calls against pie's 8.2).

## 4. Honest limits

- **n = 2, and only 1 discriminating.** A resolution rate over one instance is
  a directional signal, nothing more. It cannot separate pie from mlx-lm.
- **Contended box** throughout.
- The 13028 timeout for mlx-lm is a harness cap, not a model verdict.

## Reproducing

```sh
bash integrations/opencode/tools/swe3.sh     # drive: pie, vLLM, mlx-lm on 2 instances
bash <scratchpad>/grade.sh                   # score all three via swebench in Docker
```


## 5. The full known-5, graded (added after the 2-instance run)

mlx-lm on all five known-solvable instances, validity-gated (`server_errors=0,
client_unreachable=0, alive_at_end=1`), the first two cases mildly contended and
the rest on a quiet box.

| engine | resolved | wall | model calls / case | resolved instances |
|---|---:|---:|---:|---|
| **pie** strategy B | **4/5** | 3294 s | 8.2 | 12276, 13089, 14373, 15569 |
| mlx-lm 0.31.3 | 2/5 | **369 s** | 8.8 | 12276, 14373 |
| vLLM-metal | 1/5 | ~290 s | 2.8 | (earlier run) |

per-case, mlx-lm: 41 s / 219 s / 15 s / 27 s / 49 s.

**pie resolves twice what mlx-lm does and takes nine times as long.** And
mlx-lm's resolved set is a STRICT SUBSET of pie's — there is no instance where
mlx-lm succeeded and pie failed.

### The early-exit explanation fits vLLM and NOT mlx-lm

The tempting story after the vLLM result was "the fast engines are fast because
they give up". That is right for vLLM (2.8 model calls per case against pie's
8.2) and **wrong for mlx-lm**: it served 44 chat completions across the five
cases, ~8.8 per case, comparable to pie. It did the same amount of agent work,
about ten times faster per call, and still resolved half as many.

So there are (at least) two distinct mechanisms behind "same harness, different
engine, different result":

1. **Truncated effort** — the agent stops early. vLLM's signature. Visible as a
   low model-call count, and cheap to detect.
2. **Equal effort, worse trajectory** — the agent works just as hard and ends
   up somewhere worse. mlx-lm's signature here. NOT visible in call counts, and
   the only handle on it is the resolution rate itself.

Mechanism 2 is the one that should worry anyone benchmarking: it is invisible
to every cheap proxy (latency, tokens, call counts) and shows up only in the
expensive graded outcome.

### What this does NOT establish

It does not establish that pie's *serving* makes the agent smarter. Five
instances with one of them (13028) unsolvable by every engine is four
discriminating cases, and the arms differ in prefix-cache behaviour, chunking,
numerics and tool-call rendering all at once. 4/5 vs 2/5 on n=4 is a
one-instance-either-way result. It is a real signal and it is not a measurement
of a mechanism.

The honest summary: on this workload pie is the slowest engine tested and the
most accurate one, mlx-lm is by far the fastest and middling, and vLLM-metal is
neither — and nobody should quote any of it as more than five instances of
evidence.


## 6. A CORRECTNESS bug in pie under concurrency (found while measuring throughput)

**This is the most serious finding in this document and it outranks everything
above it.** pie silently truncates generation as soon as a second request is in
flight. It returns HTTP 200 and `finish_reason: "length"` — the code for "I hit
max_tokens" — after emitting ONE token against a 64-token cap.

Only concurrency varies; 8 requests, `max_tokens=64`, same server, same prompts:

| concurrency | completion_tokens per request |
|---:|---|
| 1 | 64, 64, 64, 64, 64, 64, 64, 64 |
| 2 | 1, 1, 1, 1, 5, 5, 5, 5 |
| 3 | 1, 1, 1, 1, 3, 3, 29, 64 |
| 4 | 0, 0, 1, 1, 1, 3, 5, 5 |
| 8 | 1, 1, 1, 1, 1, 1, 3, 5 |

The content is a clean prefix ("Here", "Here are the numbers from"), so the model
is not stopping and no error path is taken. `finish_reason=length` after 1 of 64
tokens is internally inconsistent: the budget accounting is wrong when more than
one inferlet instance is live.

### The control: it is pie's bug, not the harness's

Same probe (`tools/trunc_probe.py`), same prompt, same cap, same machine:

| engine | conc 1 | conc 2 | conc 4 |
|---|---|---|---|
| mlx-lm 0.31.3 | 64 | 64, 64 | 64, 64, 64, 64 |
| vLLM-metal | 64 | 64, 64 | 64, 64, 64, 64 |
| **pie strategy A** | **64** | **1, 5** | **0, 1, 1, 3** |

Both baselines are clean at every level. The harness, the prompt, the weights and
the box are shared across all three arms, so the defect is pie's.

### Why nothing earlier caught it

**Every measurement in this document ran one request at a time.** `bench_ab.py`
replays turns sequentially, `run_swebench.py` drives one instance at a time, and
the probes fire one at a time. pie was never asked to serve concurrently until
this section. The 4/5 SWE-bench result is real *in that regime* and is not a
statement about pie under load.

### Two rejected hypotheses, and a warning about proxy metrics

Rejected by measurement: `POOL_GRANULARITY = 256` exhausting the 2048-page pool
(predicts a ceiling of exactly 8, but 12 requests at concurrency 8 all returned
200), and this morning's `max_forward_tokens = 4096` widening (reverting to 2048
fails identically). The mechanism is unresolved; the next step is instrumenting
the guest's token-budget path with two instances live.

**The methodological lesson is the reusable part.** The first pass through this
data recorded "12/12 completed" and called it a success. Completion COUNT looked
perfect; the responses carried one token each. A harness that checks status codes
and latency — which is most harnesses, including this one until now —
cannot see this class of bug at all. `tput_bench.py` now records
`completion_tokens` per request for exactly that reason.

## 6b. Concurrent throughput — pie does not complete the workload

8 concurrent requests, 32 total, `max_tokens=128`, unique prompts (to defeat all
three prompt caches symmetrically), aggregate tokens/s computed from the
servers' own `completion_tokens`. One engine at a time. Harness:
`tput_bench.py`; driver `tools/tput.sh`.

| engine | completed | output tok/s | wall | p50 latency |
|---|---:|---:|---:|---:|
| vLLM-metal | **32/32** | **206.9** | 19.8 s | 4.95 s |
| mlx-lm 0.31.3 | **32/32** | **203.8** | 20.1 s | 5.05 s |
| pie strategy A | 8/32 | — | — | — |
| pie strategy B | 6/32 | — | — | — |

**pie's cells are blank on purpose.** It did not complete the workload, so it has
no throughput number here. Reporting 6.0 or 60.2 tok/s from a handful of
survivors would be worse than reporting nothing — it would look like a slow
engine rather than a failed one.

- **Strategy A**: 24 of 32 requests returned **HTTP 503**, with 67 driver-side
  `rejected: pie_metal_launch failed with status -1`. Exactly 8 requests
  succeeded, matching `max_forward_requests = 8`. Worse, the survivors emitted
  almost nothing: 24 output tokens across 18 completions in one run, 11 across
  8 in another.
- **Strategy B**: 26 of 32 returned HTTP 500, with the shim logging
  `ConnectionClosedOK` on its WebSocket. This one is arguably out of scope —
  strategy B is a session-per-conversation design and 32 unrelated prompts is
  not its workload — but it is what the mode does when asked.

vLLM and mlx-lm are within 1.5% of each other and both saturate cleanly.

### Two hypotheses tested and rejected

Neither guess survived, so the cause is **unresolved** and stated as such:

1. **KV pool exhaustion from `POOL_GRANULARITY = 256`.** The arithmetic is
   seductive: 256 pages x 32 tokens = 8192 tokens reserved per request, and
   `total_pages = 2048` / 256 = exactly 8 concurrent. It predicts the observed
   ceiling of 8 precisely. But a 12-request sweep at concurrency 1/2/4/6/8
   completed **12/12 at every level**, which the hypothesis forbids.
2. **The `max_forward_tokens = 4096` widening from this morning.** Rerunning at
   the stock 2048 (`rows per fire: 2048, activation pool 168 MB`) failed
   identically — 8/32 completed. Not the cause, and today's latency fix is
   exonerated.

What is left unexplained is why 12 requests at concurrency 8 succeed and 32 at
the same concurrency do not, and why the successful responses are near-empty.
That is the next thing to chase, and it wants the engine's admission path
instrumented rather than another guess from outside.

### What this does and does not mean

It does **not** mean pie cannot serve concurrent traffic — the upstream project
publishes 8-concurrent Apple Silicon throughput numbers, so the capability
exists in some configuration. It means **this** configuration, the one that
produced every latency and SWE-bench number in this document, collapses under
8-way concurrent load. That is worth knowing before anyone reads the 4/5
SWE-bench result as a general statement about the engine: it was measured with
one request in flight at a time, which is the regime pie was working in.


## 7. The bug, fixed

**Root cause.** `runtime/engine/src/scheduler/worker.rs`,
`LaunchGrouping::accepts()`. Every clause that kept a device-geometry program
out of a shared batch paired it with a USER MASK:

```rust
|| (self.has_user_mask && self.has_device_geometry)
|| (request.request.has_user_mask && self.has_device_geometry)
|| (request.request.device_resolved_geometry && self.has_user_mask)
```

The plain case — two unmasked device-geometry fires from two concurrent
instances — matched none of them and was composed into one batch. The Metal
driver refuses exactly that:

```
[pie-driver-metal] launch: 2 device-geometry programs in one batch
                           (at most one is supported)
```

and its own comment says the rule is "the same structural constraint the
runtime's scheduler already upholds ... a defensive re-check here so a
scheduling bug fails the launch loudly". The driver's defensive re-check was
the ONLY thing enforcing it.

**Why it was silent.** The refusal poisons the fire's channel, `take_host`
errors, and `chat-completions` deliberately degrades any decode error to
`finish_reason:"length"` (so a fault never becomes a 5xx that burns opencode's
retries). The client therefore saw HTTP 200, `finish_reason:"length"`, and one
token of a 64-token budget — a well-formed, plausible, wrong answer.

**The fix** is one clause: a request carrying device geometry is not accepted
into a group that already has one. The grouping loop then defers it to the
wave's next step, which is what it already does for solo submissions and dense
device masks.

```rust
|| (request.request.device_resolved_geometry && self.has_device_geometry)
```

**Verification.** `tools/trunc_probe.py`, cap 64:

| concurrency | before | after |
|---:|---|---|
| 1 | 64 | 64 |
| 2 | 1, 5 | **64, 64** |
| 4 | 0, 1, 1, 3 | **64, 64, 64, 64** |

Driver refusals: 7 -> **0**. Degraded turns: 6 -> **0**. pie now matches
vLLM-metal and mlx-lm at every level.

**No regression on the sequential path**: the 6-turn replay is 17.32 s against
17.02 s before (run-to-run noise), steady turn 0.97-1.00 s, prefix reuse intact
at 7211 of 7403 tokens cached.

### A second bug, previously masked by the first — also fixed (see §8)

At 8 concurrent / 32 requests, 8 requests now complete with FULL 128-token
outputs (1024 tokens total, up from 11-24 tokens of garbage) — and the other 24
are rejected with HTTP 503 at latency **0.0 s**, instantly, after the first
batch finishes. That is a distinct admission/process-slot problem, it is LOUD
rather than silent, and it was invisible until the truncation was fixed. Not
diagnosed here.

So pie still has no honest 8-concurrent throughput number. What it has is a
correct one at concurrency <= 8-in-flight, and a known second defect above it.


## 8. The concurrency bug, fixed — and pie finally has a throughput number

**Root cause: the pool granularity I had wrongly exonerated.** The admission
gate's own words, once the response BODY was read instead of just its status:

```
admission rejected: cluster saturated: no healthy worker has KV/seq headroom
```

`inferlets/chat-completions` rounded every request's page reservation up to
`POOL_GRANULARITY = 256` pages = 8192 tokens, however short the request. With
`total_pages = 2048` that is 2048/256 = **exactly 8** concurrent requests, and
the ninth is refused. The observed ceiling was 8. The arithmetic was exact.

**I had rejected this hypothesis earlier on contaminated evidence.** A 12-request
sweep at concurrency 8 "passed 12/12", which the hypothesis forbids — but that
sweep ran BEFORE the §7 truncation fix, when every request failed in
milliseconds and never held its pages long enough to exhaust anything. The
control was measuring the other bug. Two bugs in one path made each other's
evidence unreliable, which is the general hazard: *a falsification is only as
good as the health of everything else in the run.*

**The fix**: reserve the next POWER OF TWO of pages rather than the next
multiple of 256.

```rust
const POOL_FLOOR_PAGES: u32 = 8;
let pool_pages = (n + max_tokens as u32 + 2)
    .div_ceil(page_t).max(POOL_FLOOR_PAGES).next_power_of_two().max(have);
```

This keeps the property the rounding existed for — the pool size is a channel
SHAPE, and shape churn compiles a new ~600 ms program — while making the waste
proportional instead of flat. The ladder is 9 shapes (8, 16, ... 2048), far
inside the 64-entry program cache; a short request now reserves 512 tokens
instead of 8192, and a long conversation still lands on 256 or 512 and is
unchanged.

The comment being replaced claimed the rounding "costs nothing real ... an
over-reservation is a longer page-id list and nothing else". It cost the
server's concurrency ceiling, and it cost it invisibly, because the consequence
surfaced as a 503 from the gateway with no connection to the line that caused it.

### Verified

| | before | after |
|---|---|---|
| 8 concurrent / 32 requests | 8 completed, 24 x HTTP 503 | **32/32 completed, 0 failed** |
| output tokens | 11-1024 (truncated/partial) | **4096** (32 x 128, full) |
| driver refusals | 7 | **0** |
| admission rejections | 24 | **0** |
| degraded turns | 6 | **0** |
| tokens at concurrency 1/2/4 | 64 / 1,5 / 0,1,1,3 | **64 / 64,64 / 64,64,64,64** |

### Throughput, at last

| engine | completed | output tok/s | p50 latency |
|---|---:|---:|---:|
| vLLM-metal | 32/32 | **206.9** | 4.95 s |
| mlx-lm 0.31.3 | 32/32 | **203.8** | 5.05 s |
| **pie strategy A (both fixes)** | **32/32** | **88.2** | 11.58 s |

pie now completes the workload correctly and is **2.3x behind** the other two on
aggregate throughput — which is a real, quotable number for the first time, and
consistent with the single-stream picture (competitive decode, slow prefill).

Strategy B still carries the fixed 256-page granularity; it is a
session-per-conversation mode holding few long-lived working sets, where the
flat reservation is defensible. Nothing in §1-5 is affected — those were all
measured with one request in flight.


## 9. Pricing pie's paged attention against MLX, directly

`driver/metal/tools/rawmetal/sdpa_paged_probe.cpp` (new) fires
`sdpa_paged_mma_bfloat16_d_128` alone — one PSO, one argument table, dispatches
fused into one command buffer to strip the per-CB sync floor — on the shapes a
serving prefill actually runs.

| kernel | ms/layer | x48 layers | TFLOP/s |
|---|---:|---:|---:|
| **MLX `fast.scaled_dot_product_attention`** | **1.555** | 74.6 | **14.4** |
| pie `sdpa_paged_mma` | 7.554 | 362.6 | 3.0 |
| pie `sdpa_paged_tiled` (the fallback) | 21.070 | 1011.4 | 1.1 |

184 queries x 32 heads x 128 dim against 7424 keys x 4 KV heads, bf16 —
22.4 GFLOP per layer of QK^T + AV.

**pie's attention is 4.9x MLX's on identical arithmetic**, and reaches about a
fifth of MLX's matrix throughput.

### Why this number is trustworthy where the earlier ones were not

It reproduces a fact established independently: MMA measures **2.79x** faster
than tiled here, against **2.17x** end to end (572 ms vs 1240 ms with
`PIE_METAL_SDPA_MMA=0`). A microbenchmark that recovers a known end-to-end
ordering is worth more than one that merely produces a number.

Three configurations were discarded on the way, each for a stated reason:

- **Single-dispatch timing** (`bench_kernel`) includes launch+sync per dispatch,
  which dominates at these shapes. Fused-dispatch amortization replaced it.
- **1-row MMA** is not a configuration the driver ever dispatches:
  `llama_sdpa_mma_this_fire` requires >= 32 rows per request, so a decode step
  takes `sdpa_paged_decode`. Timing MMA at one row priced attention at 161 ms
  inside a 21 ms fire.
- **Tiled with the MMA launch shape** (128 threads vs the 1024 it needs) ranked
  tiled *faster* than MMA, inverting the known result.

Each was caught by a plausibility check against an already-measured fact rather
than by inspection. That is the only reason the surviving number is credible.

### What it means

**The gap is kernel quality, not the paged design.** Page granularity does not
move it (32/64/128/256 tokens per page: 572/571/586/572 ms end to end), so the
page walk is not what costs. pie is running a matrix kernel at ~3.0 TFLOP/s
where MLX gets ~14.4 on the same arithmetic.

That is good news for the architecture and bad news for the kernel: paging --
the thing pie cannot give up, because KV reuse is the product -- is *not* what
makes it slow. A 4.9x is a large but ordinary optimization target, and pie's own
`sdpa_paged_mma` already took 2.79x out of `sdpa_paged_tiled`, so the line has
been moving.

For the MoE half there is nothing to do: pie's `affine_qmm_t_routed` measures
~215 ms per fire against MLX `gather_qmm`'s 242 ms sorted. Already level.

---

## 9. Re-measured 2026-08-15 on a genuinely clean machine

The earlier run was taken while a stuck `pie run` job held 12.6 GB. This one is
after killing it, with `roofline_probe`'s streaming roof at **296.4 GB/s**
against the 69.8 it read while contended, and with the pie binary rebuilt so it
contains this session's scheduler fixes rather than Aug 14 code.

| engine | completed | output tok/s | wall | p50 latency |
|---|---:|---:|---:|---:|
| vLLM-metal | 32/32 | **206.8** | 19.8 s | 4.95 s |
| mlx-lm 0.31.3 | 32/32 | **205.1** | 20.0 s | 4.99 s |
| **pie strategy A** | **32/32** | **83.6** | 49.0 s | 12.17 s |
| pie strategy B | 10/32 | — | — | — |

**pie strategy A is 2.47x behind**, and the figure is stable: 83.6 here against
88.2 before, on a different machine state and a rebuilt binary.

**It confirms the serialization diagnosis.** 83.6 tok/s over 8 streams is 95.7
ms per token per stream. `results-throughput-cause.md` predicted ~106 ms if pie
serializes concurrent decodes and ~33 ms for eight tokens if it batches them.
95.7 is 10% off the serialized prediction and nowhere near the batched one. The
ceiling asserted in `LaunchGrouping` is the one being measured.

### Strategy B failed, and the gate said VALID

10 of 32 completed; 22 returned HTTP 500. Its 67.8 tok/s is not reported, for
the reason pie had no number at all in §6b: a throughput figure computed from
survivors measures the survivors.

Strategy B is a session-per-conversation mode and 8-way independent traffic is
not what it is for, so failing is defensible. **Failing with 500 is not** --
`chat-completions`'s own wire discipline reserves 5xx for genuine faults
precisely because opencode retries them without bound.

**The gate missed it.** `arm_is_valid` asked three questions -- OOM in the
server log, client unreachable, alive at the end -- and all three passed,
because the server WAS alive and healthy and simply refused most of the
traffic. None asked whether the requests worked. Now fixed. Third calibration
correction that function has needed, and the pattern is identical each time: a
check written against the last failure does not anticipate the next one.
