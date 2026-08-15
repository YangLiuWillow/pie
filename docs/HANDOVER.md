# Handover — pie on Apple Silicon, opencode integration

**Written 2026-08-15.** Branch `liu/opencode-integration`, 34 commits ahead of
`fork/liu/opencode-integration`, **nothing pushed**, working tree clean.

Every number below was re-measured on **2026-08-15 on an idle machine**
(`roofline_probe` streaming roof 298.3 GB/s) unless the line says otherwise.
Where a number could not be re-verified, it says so.

- **Machine:** Apple M5 Pro, 48 GB. `maxThreadgroupMemoryLength` = 32768 bytes
  (queried, not assumed — `tools/rawmetal/device_caps.mm`).
- **Model:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`. 48 layers,
  32 q heads, 4 kv heads, head_dim 128, 128 experts top-8, 96 KiB KV/token.
  All three engines serve the same weights.

---

## 1. The goal, and where it stands

Run **opencode** (agentic coding CLI) against **pie**, locally, and be
competitive with vLLM-metal and mlx-lm on the same hardware and weights.

Two claims, measured separately, because a single number hides both:

| | pie | mlx-lm | vLLM-metal |
|---|---:|---:|---:|
| SWE-bench known-5, graded | **4/5** | 2/5 | 1/5 |
| 6-turn agentic replay | 17.0 s | 9.2 s | 10.2 s |
| Throughput, 8 concurrent × 32 | 83.6 tok/s | 205.1 | 206.8 |

**pie wins correctness and loses speed by ~2.5x.** That has been the shape all
along and none of this session's work has changed it, because none of the
kernel work is wired into the driver yet.

The SWE-bench and replay rows are from **2026-08-14** and were NOT re-run today.
The throughput row is from today, on a clean machine, with a freshly built
binary.

---

## 2. STANDING RULE for benchmarking

**Always run all three engines — pie, vLLM-metal, mlx-lm — on the same instance,
one at a time, in the same session.** Never quote a pie number against a
remembered number for another engine.

Two 30B servers do not fit in 48 GB; that is measured, not assumed (a vLLM arm
OOM'd and dead-latched on 2026-08-14 while a peer held 12.5 GB). `tools/tput.sh`
already does boot → measure → kill → next for all three.

Before every arm:

```sh
source integrations/opencode/tools/require_quiet_gpu.sh
require_quiet_gpu 20          # exits non-zero if the machine cannot hold it
/tmp/metaltools/bin/roofline_probe | grep "streaming roof"   # expect ~290 GB/s
```

**`roofline_probe` is the authority, not the process list.** On 2026-08-15 a
background job dropped the roof to 69.8 GB/s while leaving free memory ample and
CPU near idle; every memory-bound measurement inflated 1.3–2.2x and every
register-bound one did not. That asymmetry does not break an A/B, it TILTS one.

---

## 3. What was fixed (all verified, all in the tree)

### Correctness

1. **Silent output truncation under concurrency.**
   `runtime/engine/src/scheduler/worker.rs`, `LaunchGrouping::accepts()`.
   Two device-geometry programs were composed into one batch, which the Metal
   driver refuses; the refusal degraded to `finish_reason:"length"` with **one
   token of a 64-token budget** and HTTP 200. Verified 64/64/64 at concurrency
   1/2/4.

2. **Concurrency ceiling of 8.** `inferlets/chat-completions/src/engine.rs`.
   Pool reservation rounded to 256 pages, so the 9th request hit
   `cluster saturated`. Now rounds to the next power of two. Verified 32/32.

3. **Two Qwen3 tool-call parser bugs**, ported from upstream `dev-sslee` and
   reproduced against our own diverged parser before fixing
   (`model/qwen_3/src/chat.rs`):
   - a parameter-name scan running into a shell redirect, yielding a
     90-character argument KEY and no error;
   - a function-name scan doing the same, yielding tool names like
     `bash\n<parameter=command`.
   **Not attributable to our benchmark results** — upstream caught theirs on
   `django-10914`, which is not in our known-5, and we retain no agent
   transcripts. The fixes stand on reproduction, not on the 4/5.

4. **Flaky scheduler tests.** All 32 async tests in `scheduler::worker` lease
   from the process-wide `pie_waker::WakerTable` and cargo runs them in
   parallel. Measured 3 of 5 full-module runs failing before; 15 of 15 clean
   after a serializing guard.

### Performance

5. **`_p32` shifted page addressing** — `driver/metal/src/kernels/sdpa_paged_mma.metal`
   plus the selection sites in `model/{llama,gptoss}/kernels.cpp`.
   Re-measured today: **7.090 → 6.318 ms/layer (10.9%)**, output byte-identical
   to the pre-change baseline. Gated on `kv_page_size == 32` exactly, never on
   a power-of-two inference, because **nothing in this repo validates
   kv_page_size** — it is an operator-set TOML field and every check on it is
   `> 0`.

6. **Prefill row-count cliff worked around** in `inferlets/opencode-session`.
   A fire with `rows % 8` in 1..=6 costs a flat ~570 ms.
   **30.6 s → 17.0 s** on the agentic replay (measured 2026-08-14).

---

## 4. Kernel work — two prototypes, NEITHER wired into the driver

### 4a. k-row decode with a shared KV read — FINISHED, measured, unlanded

`driver/metal/tools/rawmetal/kernels/sdpa_krow_decode.metal`. Keeps the decode
kernel's key-parallel decomposition and gives each simdgroup all k query rows,
so a key loaded once serves all of them. **Correct at every k** against a CPU
reference. 16k context, KROWS=1 being the same kernel doing what
`sdpa_paged_decode` does:

| k | this kernel | shipped per-row | |
|---:|---:|---:|---|
| 2 | **1.26x** | 1.54x | 18% faster |
| 4 | **1.77x** | 2.39x | 26% faster |
| 5 | **2.04x** | 2.82x | **28% faster** |
| 6 | 4.43x | — | **cliff** |
| 8 | 6.45x | 4.08x | 58% slower |

Linear at 0.26x per extra row to k=5, then a 2.4x step on ONE row: register
spill (`q[k][4] + o[k][4]` is 40 floats at k=5, 48 at k=6).

**This is the highest-value unlanded work.** k=5 is exactly the speculative
verify size, speculation is already implemented and waiting on it, and the same
kernel is what concurrent decode batching would need.

### 4b. NAX (M5 neural accelerator) prefill attention — PARTIAL

`mpp::tensor_ops::matmul2d` on the M5 neural accelerators, reachable through
pie's runtime shader compiler (verified, no Xcode needed — the `metal` CLI is
absent and does not matter).

**Correct and measured** (`kernels/nax_slice_qk.metal`, `nax_slice_pv.metal`):
Q·Kᵀ 0.618 ms/layer, P·V 0.758, sum **1.376 ms** against MLX's whole pass at
1.31 and pie's current 6.32.

**The fused version does not work.** Isolated by elimination:
- matmul destination in threadgroup memory: **fine** (0/2048 wrong);
- **a cooperative tensor does not carry across `run()` calls** — P·V twice into
  one cooperative O gave 7316/8192 wrong, worst relative exactly 1.0000 (zero
  where the reference is not). A lifetime problem, not a layout one.

That explains MLX: `steel_attention_nax.h` keeps O in its own register array and
materializes cooperative tensors transiently inside each matmul. **The correct
design is MLX's**, and the layout it needs is queryable —
`get_multidimensional_index(i)` returns each element's (row, col);
`get_capacity()` the count. An evening was spent deriving that from a matmul's
output before finding the accessor.

---

## 5. The throughput gap — diagnosed, asserted, confirmed

**pie serializes concurrent decodes.** Not a scheduling accident:

- `runtime/engine/src/pipeline/fire.rs:1458` sets
  `device_resolved_geometry = decode_envelope.is_some()`, so **every** decode
  fire carries it;
- `driver/metal/src/context.cpp:1006` — the driver supports **at most one**
  device-geometry program per batch.

Two concurrent decodes can therefore never share a fire.

Confirmed three ways: a source trace, a unit test in `LaunchGrouping` (with the
control that two requests *without* device geometry DO co-batch), and the
benchmark — 83.6 tok/s over 8 streams is **95.7 ms per token per stream**
against a predicted ~106 ms serialized and ~33 ms batched.

Behind it sits a second ceiling: a k-row decode does not share its KV read
(slope `0.464 + 0.703·rows`), so even with co-batching allowed the win would be
~2x at agentic context lengths. §4a is the fix for that half.

---

## 6. OPEN AND UNRESOLVED — do not paper over these

1. **pie's attention kernel appears to exceed its own matrix unit's ceiling.**
   The ablation prices the multiply half at 3.401 ms for 23.7 GFLOP = **6.97
   TFLOP/s**, and `matrix_rate_probe` measures the simdgroup ceiling at **5.46–
   5.48** (three configurations agree: 8 chains, 16 chains, fp32 accumulate).
   A kernel cannot beat its unit. One of the two is wrong.
   - Ruled out for the microbenchmark: latency-bound (16 chains identical to 8),
     accumulator type, and duration (a sweep runs 3.96 → 5.18 TFLOP/s with
     length, i.e. clock ramp-up, not throttling).
   - Ruled out for the ablation: loop-invariant hoisting was found and fixed
     once (one write per thread per pass); it may not be fully defeated.
   **Consequence: the move/multiply split (43% / 54%) is NOT reliable.** Do not
   build a plan on it.

2. **pie strategy B fails under concurrent load.** 10 of 32 completed, 22
   returned HTTP 500. Failing at 8-way independent traffic is defensible for a
   session-per-conversation mode; **failing with 5xx is not**, because
   opencode retries 5xx without bound.

3. **A `pie run` job hung for 20 hours** holding 12.6 GB (killed 2026-08-15).
   25 seconds of CPU total; main thread parked in tokio `block_on` on a future
   that never resolved, driver thread healthy and idle in `serve_forever`. Not
   reproduced, not diagnosed. Same subsystem as the wait-slot machinery.

4. **`llama_numerics_test` is 51 pass / 18 fail, pre-existing.** MoE routing
   ties; first divergence at a layer-0 projection, before attention. Verified
   identical before and after this session's kernel changes.

---

## 7. Tests — what exists and how to run it

| suite | command | state 2026-08-15 |
|---|---|---|
| engine scheduler | `cargo test -p pie-engine --lib scheduler::worker` | **50 pass** |
| Qwen3 tool-call parser | `cd model/qwen_3 && cargo test --features chat` | **36 pass** |
| opencode-session inferlet | `cd inferlets/opencode-session && cargo test` | **9 pass** |
| chat-completions inferlet | `cd inferlets/chat-completions && cargo test` | **7 pass** |
| llama PSOs | `/tmp/metaltools/bin/llama_pso_test` | **32 pass** |
| llama decode step | `/tmp/metaltools/bin/llama_decode_step_test` | **221 pass** |
| kv_append paged PSOs | `/tmp/metaltools/bin/kv_append_paged_pso_test` | **13 pass** |
| gpt-oss decode step | `/tmp/metaltools/bin/gptoss_decode_step_test` | **all pass** |
| llama numerics | `/tmp/metaltools/bin/llama_numerics_test` | 51 pass / **18 pre-existing fail** |

Building the Metal tests and probes:

```sh
cmake -S driver/metal -B /tmp/metaltools -DPIE_METAL_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/metaltools -j 8
```

**A numerics diff proves nothing unless the kernel ran.** The suite's headline
cases are "40 rows over 2 requests" and the matrix path needs 32 rows *per
request*. Confirm with `PIE_METAL_SDPA_TRACE=1`, which prints
`rows=48 requests=1 ... hd=128 -> MMA`.

---

## 8. Probes and benchmarks

### Probes (no serving stack, seconds to run)

| probe | answers |
|---|---|
| `sdpa_paged_probe` | attention cost per configuration; the k-row decode A/B; the move/multiply ablation (see §6.1) |
| `matrix_rate_probe` | simdgroup vs neural-accelerator throughput; NAX correctness against CPU references; tile sweep |
| `roofline_probe` | **the machine's streaming roof — the bandwidth authority** |
| `device_caps` | threadgroup memory cap, GPU families, tile budgets |

### Benchmarks (need a serving stack and a quiet machine)

| harness | what it measures |
|---|---|
| `tools/tput.sh` → `tput_bench.py` | 8-concurrent aggregate throughput, all three engines |
| `bench_ab.py` | 6-turn agentic replay, per-turn latency |
| `tools/swe3.sh`, `tools/grade.sh` | SWE-bench instances end to end, then graded |

`tput_bench.py` uses **unique prompt prefixes** to defeat all three prompt
caches symmetrically, and computes tok/s from each server's own
`completion_tokens` — not an estimate.

**Rebuild before benchmarking.** On 2026-08-15 the binary was a day older than
its sources; measuring it would have measured last night's code:

```sh
cargo build --release -p pie-bin --features driver-metal
cd inferlets/chat-completions && cargo build --target wasm32-wasip2 --release
```

---

## 9. Traps that cost real time

1. **An instrument's silence is not a measurement.** A dead vLLM server
   reported "0 patches"; `$(grep -c … || echo 0)` yields `"0\n0"` and voided a
   healthy arm; an ablation matching nothing reported a kernel as free; a
   byte-identical numerics diff came from a suite that never ran the kernel.
2. **Correctness before throughput.** A NAX kernel was measured at 1.652
   ms/layer and reported three times before a CPU reference showed it computed
   the wrong thing. A timing cannot distinguish correct attention from a
   transposed operand — both do the same FLOPs at the same rate.
3. **Match the occupancy.** A rate measurement dispatched one threadgroup
   against another arm's 96 and reported a 64x API penalty that was an idle GPU.
4. **Sweep the free parameter before concluding.** "2.30x, the rule fails" was
   honest and premature; ten minutes of tile sweeping moved it to 3.67x.
5. **A constraint-forced choice must be revisited when the constraint goes.**
   BK=32 was forced by the 32 KB threadgroup cap and carried silently into an
   API that never touches threadgroup memory. The tile was where the
   throughput was.
6. **Look for the accessor before the microscope.** The cooperative tensor's
   lane layout was derived from matmul output over an evening;
   `get_multidimensional_index()` returns it.
7. **A gate written against the last failure does not anticipate the next.**
   `require_quiet_gpu` needed three corrections in one day: it over-fitted to
   swap, then to a free-memory ratio, then stamped an arm VALID that returned
   22 HTTP 500s because it only ever asked whether the *server* died.

---

## 10. What I would do next, in order

1. **Land the k-row decode kernel** (§4a). Finished, correct, measured, and it
   unblocks speculation that is already written. Needs a third launch shape and
   a selection rule — and `llama_sdpa_mma_this_fire`'s warning applies: if
   `pso_for` and `launch_shape` disagree, the grid describes a different kernel
   than the one that runs. Wrong numbers, not a crash.
2. **Resolve §6.1** before trusting any prefill plan built on the split.
3. **Fix strategy B's 5xx** (§6.2) — a correctness issue on the agentic path.
4. **Finish the NAX fused kernel** (§4b) using MLX's structure. Largest
   potential win, longest path, and it only moves turn latency — not the
   throughput gap, which is §5.
