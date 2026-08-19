# M5 Pro Metal bring-up and tensor-path plan

**Machine:** Apple M5 Pro, 48 GB unified memory — the configuration BaseRT
evaluated on (arXiv:2607.19438 §4.1).
**Status:** proposed, not started.

---

## Read this first: what the driver already has

An earlier draft of this plan assumed a greenfield Metal driver and scheduled
work that is already shipped. Correcting that is the most useful thing this
document does, so it goes at the top.

| Assumed missing | Actually present |
|---|---|
| q4 GEMV kernel | `src/kernels/quantized_qmv.metal` — MLX `qmv_fast_impl` ported |
| q4 GEMM kernel | `src/kernels/quantized_qmm_t.metal` — MLX `qmm_t_impl` + steel primitives, with occupancy sweeps **closed in both directions** around a 2×2 warp shape |
| Affine format / packing | `src/kernels/affine_format.hpp`, `mxfp4_codec.h` |
| Weight conversion + quantization | `loader/` — a `Cast` into a quantized encoding is how "quantize this" is spelled; `pie-loader convert` runs it on CPU |
| Checkpoint container | `ztensor` 2.1.1 / `ztensor-compat`, spec draft at repo root |
| GEMV↔GEMM dispatch threshold | `src/device_tuning.{hpp,cpp}`, per Apple family, with `benches/tune_device.py` to measure it |
| Correctness harness vs. MLX | `tests/mlx/` (reference model graphs), `tests/parity/` (per-layer taps + `cosine_bisect.py`), `*_numerics_test.cpp` |
| Roofline / occupancy tooling | `tools/rawmetal/roofline_probe.cpp`, `dvfs_probe.mm`, `bench_kernels.cpp` |
| Three-way benchmarking | `benches/three_way.py`, `pie_bench.py`, `mlx_bench.py`, `llamacpp_bench.py` |
| **GatedDeltaNet kernels** | `gdn_core.metal`, `gdn_prep.metal`, `gdn_params.h`, plus the `model/qwen_3_5` crate and `qwen3_5_geometry_test.cpp` |
| Qwen3.6-27B running at all | already benchmarked — `device_tuning.hpp` records 23.0 → 32.1 tok/s from a crossover change |

**So the question is not "how do we add quantized Metal kernels to Pie." It is
"what does an M5 Pro change about a driver tuned on an M1 Max."**

### The four real gaps

1. **No M5 entry in `device_tuning.cpp`.** Apple families through M4 have
   overrides; an M5 falls back to M1 Max constants. The file's own header names
   this as the costly failure: "the GEMM crossover sits three rows too high, so
   the batches where the GEMM already wins are still served by the GEMV."
2. **No cooperative-tensor compute path.** `mtl4_context.mm` uses the Metal 4
   *command* API (MTL4CommandQueue, MTL4Compiler, MTL4ArgumentTable,
   MTLResidencySet, timestamps) but every matmul goes through
   `simdgroup_matrix`. `mpp::matmul2d` appears once, in a comment. On M5 the
   Neural Accelerators are reachable only through the tensor API, and BaseRT
   attributes up to 4× prefill to exactly this.
3. **No narrow-type measurement on M5.** Whether fp8/int8 have a native path on
   the Neural Accelerators is unpublished anywhere. It decides whether W8A8 is
   ever worth building for the batched-decode regime.
4. **Quality gating is cosine/tap-based, not divergence-based.** No
   per-category KL divergence against a bf16 reference.

---

## Conventions

**Benchmark protocol** (BaseRT §4.1, so numbers stay comparable): prompt lengths
128/256/512/1024/2048; 128 generated tokens; 5 repetitions; mean ± stddev; AC
power; arms alternated, not batched; medians.

**Pinned baselines:** llama.cpp **b9960**, mlx-lm **0.31.3**, MLX **0.32.0**.
Do not upgrade — see `CLAUDE.md`.

**Reference shorthand**

| Tag | Document |
|---|---|
| **[M5]** | arXiv:2607.19438 — BaseRT with M5 Neural Accelerators. §2.1–2.3 hardware/API, §3.1–3.2 kernels & dispatch, §4.1 setup, Tables 1–3, §5.1 limitations |
| **[BRT]** | arXiv:2607.00501 — BaseRT base paper. §3.1 architecture descriptors, §3.2 zero-allocation decode loop, §3.3 kernel fusion, §3.4 prefill |
| **[RIG]** | arXiv:2606.12765 — Rigel, Metal 4.1 tensor path reverse-engineered on M4 Max |
| **[TQ]** | arXiv:2604.16957 — Open-TQ-Metal, compressed-domain int4 attention |
| **[DT]** | `driver/metal/src/device_tuning.hpp` — the local convention for tuning constants |

---

# Phase 0 — Ground truth on this machine

## T0.1 — Build and run the existing suite

```sh
cargo build --workspace --all-targets --exclude pie-server-py
cargo test --workspace
# on-device suite (mostly #[ignore]d)
cargo test -p pie-gpu-tests -- --ignored
```

**Success metric:** the Metal driver builds and the on-device tests pass on M5
Pro. Any failure here is an M5 compatibility bug and is the first thing to fix —
every later number depends on the driver being correct on this silicon.

**Knowledge needed:** `driver/metal/CMakeLists.txt`; `CLAUDE.md` build section.

---

## T0.2 — Record what the driver thinks this machine is

```sh
# apple_family and gpu_core_count as the driver resolves them
cargo test -p pie-gpu-tests descriptor_facts -- --ignored --nocapture
```

**Success metric:** `apple_family` and `gpu_core_count` recorded in
`driver/metal/docs/m5-pro-measurements.md`, plus confirmation of **which**
`DeviceTuning` block an M5 currently selects (expected: the default, i.e. M1 Max).

---

## T0.3 — Three-way baseline

```sh
python benches/three_way.py   # pie vs mlx vs llama.cpp, pinned versions
```

Models: start with the small dense ones so the variable is the driver, not the
architecture.

**Success metric:** the "Model throughput" table in
`driver/metal/docs/m5-pro-measurements.md` filled for both models, all engines.

Note `*.csv` and `/docs/` are gitignored repo-wide, so raw harness output stays a
local artifact — the committed record is the markdown table, with the method
alongside each row. That matches the `device_tuning.hpp` convention.

---

## T0.4 — BaseRT as a fourth data point

```sh
curl -LsSf https://basecompute.co/install.sh | sh
export PATH="$HOME/.basert:$PATH"
basert pull Qwen/Qwen3-0.6B
basert pull meta-llama/Llama-3.2-1B-Instruct
basert bench Qwen/Qwen3-0.6B
basert inspect Qwen/Qwen3-0.6B    # .base header + tensor inventory
```

Compare against **[M5] Tables 1–3**.

**Success metric:** BaseRT reproduces its own published figures on this machine
within **±10%**. If it does not, the discrepancy is thermal/OS/config and must be
identified before any Pie-vs-BaseRT claim is made.

Also worth running once, for information no document contains:

```sh
basert pull Qwen/Qwen3.6-27B && basert bench Qwen/Qwen3.6-27B
```

Qwen3.6-27B is config #10 of the fifteen in **[M5] §4.1**, yet neither paper
mentions linear attention, DeltaNet, or chunkwise scan anywhere. This tells you
empirically whether BaseRT handles GatedDeltaNet.

> **Licensing.** BaseRT's engine is proprietary and binary-only. Do not vendor or
> disassemble it. Read the EULA before publishing comparative numbers.

---

# Phase 1 — Characterize M5 Pro and land the tuning entry

*Highest value, lowest effort work available. The driver is currently running
M1 Max constants on this machine.*

## T1.1 — Sweep the crossovers

```sh
python benches/tune_device.py            # prints a tuning_for() block
python benches/tune_device.py --control  # same arms at a batch where they agree
```

Read the script header before running. Its documented methodology error is the
one to avoid: **a threshold only means something at batches that straddle it**.
The script picks the batch by predicate rather than taking a row count on faith;
`--control` is the noise check, and if control arms do not land on top of each
other, the machine is too noisy for anything else here to mean much.

**Success metric:** a `tuning_for()` block for the M5 Apple family, with each
overridden field carrying its own measurement table in the **[DT]** house style.
Control arms agree within noise.

## T1.2 — Land the entry

Add the M5 case to `device_tuning.cpp`.

**Success metric — the invariant from [DT], non-negotiable:** a
default-constructed `DeviceTuning` still reproduces the M1 Max numbers exactly,
and an unrecognised device is unchanged. Verify by test, not by inspection.

**Expected payoff:** the equivalent change on another family moved Qwen3.6-27B
from 23.0 to 32.1 tok/s (+40%) and gemma-4-31b from 17.9 to 30.0 (+68%). Measure
the actual delta on M5 and record it.

## T1.3 — Roofline constants

```sh
./roofline_probe <kernels> 128 32     # see tools/rawmetal/roofline_probe.cpp
./dvfs_probe                          # clock behaviour under sustained load
```

Capture: peak sustained bandwidth (GB/s), peak achieved TFLOP/s, and the
resulting ridge point `peak_FLOPS / peak_bandwidth`.

**Success metric:** all three in `driver/metal/docs/m5-pro-measurements.md`, stable to
±3% over 5 runs. **These supersede every published M5 figure** — the public ones
conflict (153 GB/s vs ~120 GB/s) and neither is credible for a Pro tier.

**Sanity check against T1.1:** solve for the batch at which a q4 linear layer
crosses the ridge,

```
intensity(M) = 2·M·K·N / (0.5·K·N + 2·M·K + 2·M·N)
                          ^weights    ^acts in ^acts out
```

and confirm it lands within a factor of 2 of the measured `qmm_min_batch`. A
larger gap means either the roofline inputs or a kernel's efficiency is off, and
it is worth knowing which before Phase 2.

---

# Phase 2 — Is the tensor path real on this chip?

*A measurement phase that gates Phase 3. Do not write cooperative-tensor kernels
before it returns.*

## T2.1 — `matmul2d` vs `simdgroup_matrix`

Extend `tools/rawmetal/` with a standalone probe comparing, at identical shapes
(4096³, 2048³, 1024³, and the real prefill shape M=128 K=N=5120):

1. the existing `simdgroup_matrix` path
2. `mpp::matmul2d` with `cooperative_tensor` accumulators

**Success metric:** the ratio `matmul2d ÷ simdgroup_matrix`, plus absolute peak
TFLOP/s.

**Decision rule:**

| Ratio | Action |
|---|---|
| ≲ 1.2× | Neural Accelerators are not delivering through this API on M5 Pro. **Skip Phase 3.** Redirect to bandwidth-side work. |
| ≫ 1.2× | Phase 3 is justified; the ratio is its expected ceiling. |

**[RIG]** measured **1.05–1.21× on M4 Max** and concluded no dedicated matrix
unit exists there, with a ~14.8 TFLOP/s ceiling inside the ALU limit. An M5
result materially above that band is the first published evidence for this tier —
**[M5] §5.1** explicitly leaves M5 base and Max untested.

**Knowledge needed:** **[M5] §2.1–2.3**; **[RIG]** methodology (three independent
signals for where an op executes; the reconstructed 8×8 `cooperative_tensor`
fragment layout, which Apple does not document); note the existing
`roofline_probe` already does whole-step TFLOP/s the way this needs to.

## T2.2 — Narrow types

Repeat T2.1 with fp16, fp8 (E4M3), and int8 operands.

**Success metric:** throughput ratios relative to fp16.

| Result | Meaning |
|---|---|
| ≈ 0.9–1.1× | emulated, as on M4 (**[RIG]**: fp8 = 0.94× fp16). Weight-only quantization stays the only format worth supporting; W8A8 is dead on this hardware. |
| ≳ 1.8× | native narrow path. W8A8/FP8 becomes worth building for batched decode and prefill. Open a separate design note. |

Also re-verify accumulator width (**[RIG]** proved ≥ fp32 on M4) — it sets the
numerics budget for anything built on this path.

**This question has no published answer for M5.** Whatever it returns is
reportable and it decides Pie's format roadmap.

---

# Phase 3 — Cooperative-tensor prefill GEMM

**Gate: only if T2.1 cleared 1.2×.**

## T3.1 — Port the quantized GEMM to `matmul2d`

Add a cooperative-tensor variant alongside `affine_qmm_t_aligned` in
`quantized_qmm_t.metal` (or a sibling file — it must stay self-contained;
`newLibraryWithSource` does no include resolution).

Constraints, from **[M5] §3.1**:

- read operands **directly from device memory**; BaseRT explicitly avoids
  threadgroup-memory staging on this path
- keep the **same packed weight layout** as the SIMD path so no reconversion is
  needed at the phase boundary
- dequantization stays in the inner loop, never materialized

**Success metric:** at the prefill shape, ≥ 0.6 × the T2.1-measured tensor peak,
**and strictly faster than the shipping `simdgroup_matrix` GEMM**. If it is not
faster, the fragment layout or the operand feeding is wrong — do not ship it.

## T3.2 — Dispatch

Add the selection to `device_tuning` as a per-family constant, **not** as a
branch in the kernel. The registry answers *which kernel*; the tuning table
answers *how to launch it* (**[BRT] §3.1**, **[M5] §3.2**).

**Success metric:** default-constructed `DeviceTuning` unchanged (the **[DT]**
invariant); measured crossover recorded; token agreement with the reference
interpreter preserved.

## T3.3 — Attention

Only after T3.1 lands: route `sdpa_paged_mma.metal`'s QK^T and PV through the
tensor path with exact online softmax (**[M5] §3.1**).

**Success metric:** prefill throughput improvement at 2048-token prompts;
numerics tests unchanged.

---

# Phase 4 — Divergence-based quality gates

The parity harness (`tests/parity/`, `cosine_bisect.py`, per-layer taps) is good
at *localising* a numerics bug. It does not answer "is the quantized model still
the same model."

## T4.1 — Per-category KL

Add a gate computing token-by-token KL divergence over top-40 logprobs against a
bf16 reference, **by category**: {code, long-context, tool-calling, non-Latin,
general}. Record top-1 agreement alongside.

**Why not perplexity:** it is an aggregate and hides structured damage.
Published KL benchmarking shows Q8 at 0.069 overall but **0.177 on tool calling**;
on another model, 0.466 on long documents and 0.222 on non-Latin scripts against
0.069 on science. A 1%-perplexity gate passes quants that are visibly worse at
the thing you built them for.

**Success metric:** the gate runs in CI on a converted checkpoint; a deliberately
over-aggressive quantization (q2, group 128) is **rejected** by it while passing a
perplexity check. Demonstrating the gate catches something is the metric — not
that it runs.

---

# Phase 5 — Qwen3.6 / 3.8-27B on M5 Pro

Both are the same architecture: 64 layers as 16 × (3 × GatedDeltaNet→FFN → 1 ×
GatedAttention→FFN), hidden 5120, intermediate 17408, full attention 24 Q / 4 KV
at **head_dim 256**, GDN 48 V / 16 QK at head_dim 128, vocab 248,320 padded,
262K context, plus a vision encoder. Qwen3.8 adds an MTP draft head.

Qwen3.6-27B already runs (**[DT]** records it). So this phase is not a bring-up:

| Task | Metric |
|---|---|
| T5.1 — re-tune 27B crossovers on M5 with the Phase 1 entry | tok/s delta vs. the M1-Max-constant baseline |
| T5.2 — KV budget | 16 layers × 2 × 4 heads × 256 × 2 B = **64 KB/token**; ~16 GiB at 262K in bf16 vs. 48 GB total. Quantized KV is required, not optional. Target int4 → ~4.3 GB. See **[TQ]** compressed-domain attention (48× at 128K, identical top-1 greedy tokens). |
| T5.3 — head_dim 256 tiling under the tensor path | if Phase 3 landed: 8 elements/lane breaks the one-simdgroup-per-head mapping and doubles accumulator registers. Re-sweep; do not assume the 128 shape scales. |
| T5.4 — Qwen3.8 generation crate + MTP | a `model/qwen_3_8` crate if the descriptor diverges; MTP head wired to a speculative-decoding inferlet |

**T5.4 is the strategic one.** BaseRT names speculative decoding as future work
(**[M5] §5.2**) precisely because it converts memory-bound decode into
compute-bound batched verification. Qwen3.8 ships a draft head. Pie already runs
speculative decoding as guest code with no engine change. Draft head + inferlet +
tensor-path verification is a result BaseRT structurally cannot produce.

---

# Risk register

| Risk | Signal | Mitigation |
|---|---|---|
| M5 falls back to M1 constants indefinitely | T0.2 confirms default block selected | Phase 1 is first for this reason |
| Tensor path is a dud on M5 Pro | T2.1 ratio ≲ 1.2× | Phase 3 is gated. Cheaper to learn in Phase 2 than Phase 3. |
| Breaking the `device_tuning` invariant | default constants change | it is a test, not a convention — assert it |
| Deleting "dead" kernel instantiations | undispatched templates in `quantized_qmm_t.metal` | they keep closed sweeps re-runnable; documented in `CLAUDE.md` |
| BaseRT baseline doesn't reproduce | >10% vs **[M5]** tables | investigate thermals/OS before any comparison claim |
| Publishing comparative numbers | paper draft citing BaseRT throughput | read the engine EULA first |

---

# Definition of done

1. `driver/metal/docs/m5-pro-measurements.md` throughput table — pie, mlx-lm, llama.cpp, BaseRT.
2. `driver/metal/docs/m5-pro-measurements.md` — bandwidth, TFLOP/s, ridge, crossovers,
   `matmul2d` ratio, narrow-type ratios.
3. An M5 block in `device_tuning.cpp`, each field carrying its measurement, with
   the default-construction invariant asserted by test.
4. A decided, recorded answer on the tensor path — either a shipped
   cooperative-tensor GEMM with a measured win, or a written record of why not.
5. A per-category KL gate that demonstrably rejects a bad quant.
6. Qwen3.6-27B tok/s on M5 Pro, before and after the tuning entry.

---

# Verify before executing

Read via a shallow clone of `dev` and a fetch of external docs, not by running
anything. Confirm first:

1. **Test target names** — `cargo test -p pie-gpu-tests descriptor_facts` and the
   `--ignored` invocations are inferred from the workspace comment on
   `tests/gpu`, not from a run. Check `tests/gpu/Cargo.toml`.
2. **`roofline_probe` invocation** — `./roofline_probe <kernels> 128 32` is copied
   from a comment in `quantized_qmm_t.metal`. Confirm against
   `tools/rawmetal/CMakeLists.txt`.
3. **`three_way.py` / `tune_device.py` arguments** — read `--help` before use.
4. **Apple family for M5** — `device_tuning.cpp` resolves newest-first; confirm
   which `MTLGPUFamilyApple<N>` an M5 answers before adding a `case`.
5. **Whether a `model/qwen_3_6` descriptor is needed** — `model/qwen_3_5` exists;
   3.6 may already be covered by it.
