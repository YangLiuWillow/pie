# Measurements — Apple M5 Pro, 48 GB

Constants measured on this machine. **These supersede published figures.**
Public M5 bandwidth numbers conflict (153 GB/s vs ~120 GB/s) and neither is
credible for a Pro tier.

Every row must carry how it was taken. A number without a method is a guess with
a decimal point.

| Field | Value | Method | Date |
|---|---|---|---|
| macOS / Darwin | macOS 26.5.1 (25F80), Darwin 25.5.0 | `sw_vers`, `uname -r`. Note: [M5] §4.1 ran Darwin 25.4 | 2026-08-19 |
| Toolchain | Apple clang 21.0.0 (CLT; no offline `metal` compiler — kernels compile at run time via `newLibraryWithSource`), cmake 4.4.2, rustc 1.97.1 | `c++ --version`; `xcrun -sdk macosx metal --version` fails, CLT-only install | 2026-08-19 |
| `apple_family` | **9** as the driver sees it; the device itself answers **yes to Apple10** (and no to 11/12), which the driver never asks | `tools/rawmetal/device_identity_probe`, clean env (`env -i`) | 2026-08-19 |
| `gpu_core_count` | **20** | same probe — IOKit `gpu-core-count` | 2026-08-19 |
| `DeviceTuning` block selected | **case 9 (M3/M4)**: `qmm_bn_crossover_tg` = 96, everything else at M1 Max defaults. NOT the M1 Max default block the plan expected — the family probe stops at Apple9, so an M5 is indistinguishable from an M3/M4 until the probe list is extended to Apple10 (Phase 1) | same probe: selected vs default-constructed fields | 2026-08-19 |

## Suite state (T0.1)

Build and on-device suite on this machine, 2026-08-19, branch
`docs/m5-pro-bringup-plan` tree:

| Step | Result | Method |
|---|---|---|
| `cargo build --workspace --all-targets --exclude pie-server-py` | OK (2m42s cold) | exit 0, warnings only |
| `cargo rustc -p pie-loader-capi --lib --crate-type staticlib` | OK | produces `target/debug/libpie_loader_capi.a`, found by the CMake configure |
| `cmake -S driver/metal -B /tmp/metaltests -DCMAKE_BUILD_TYPE=Release -DPIE_METAL_BUILD_TOOLS=ON` + build | OK | 42 ctest targets + rawmetal tools |
| `cargo test --workspace --exclude pie-server-py` | 772 passed / 1 failed after two machine-independent fixes (below); the failure is pre-existing engine/DSL drift, analysed below | 50 suites. The plain `--workspace` form CANNOT link on macOS — `pie-server-py` hard-enables `driver-cuda`, and feature unification poisons `pie-engine`'s lib test with undefined CUDA symbols |
| `ctest --test-dir /tmp/metaltests` | **40/42**; both failures analysed below | 75s wall |

Fixes needed to get there (in the T0.1 PR, none touch the driver):

1. **`chat::cue()` → `chat::cue(true)`** in 16 guest inferlets under
   `tests/inferlets/` and `runtime/engine/tests/inferlets/`. Commit
   `1503c7c8b` changed the WIT to `cue(thinking: bool)` and migrated only the
   production `inferlets/`; the old `cue()` was the thinking-on variant
   (host mapped `thinking ? cue() : cue_no_think()`), so `cue(true)` is
   behavior-preserving. This unblocked `pie-bin`'s `boot_smoke` and the
   `text-completion-bench` inferlet T0.3 needs.
2. **Two `rng_contract` scan exemptions**: `llama_numerics_test.cpp`'s
   murmur3-style test-weight hash trips the `>>40` needle, and
   `openai-serving/src/session.rs`'s FNV fingerprint trips the golden-ratio
   word. Both are the scan's own "unrelated user" idiom, not contract owners.

ctest failures, analysed:

- **`llama_numerics_test`: 50 pass / 19 fail — exactly the recorded baseline**
  (`docs/HANDOVER.md` §5.4: 18 pre-existing MoE routing ties + 1 deliberate
  row-gate tie). Matching the recorded baseline is a pass for T0.1; no new
  failure appeared on M5.
- **`host_buffer_bind_test`: fails only at the 12 GB sparse-mapping step**,
  `kIOGPUCommandBufferCallbackErrorOutOfMemory` at commit — **under
  contention**: a concurrent `pie serve` from another session (pie-opencode
  checkout) held 16 GB with a model GPU-resident, leaving ~10.6 GB free. The
  same binary **passes at 6 GB** (1024/1024 elements at +5.37 GB offset), so
  the wrap/bind/read mechanism is intact on M5. **Follow-up**: re-run at
  12 GB on a quiet host to confirm the pass; only if it still fails does the
  residency-accounting question (whole sparse mapping backed at commit vs
  demand-faulted) open up.

Pre-existing breakage, recorded not fixed (engine/DSL, not driver, not M5):
`pie-engine`'s `north_star_e2e::north_star_mtp_grammar_composition` fails
deterministically — "seed channel 9: 16 bytes, expected 128". The 16-byte
seed is `mtpverify`'s `[k=4]` i32 draft; a 128-byte channel at n=32 is
`positions` — the guest's seed order and the registered program's channel
numbering have drifted, most plausibly in the recent SDK channel refactors
(`ead310dd4`, `65cf1a848`). Machine-independent; needs its own task.

Known flake, recorded not fixed: `pie-engine`'s
`terminal_cells_recycle_only_after_native_attempt_retirement` failed once
under concurrent build load and passes 5/5 + 3/3 isolated. The terminal-cell
pool is a process-global `SegQueue` shared by parallel tests, and
`pool_contains` races with any concurrent `WorkItemCompletion::new` pop; a fix
wants a design decision (serialise pool-touching tests), not a Phase 0 patch.

## Roofline

| Field | Value | Method | Date |
|---|---|---|---|
| Peak sustained bandwidth (GB/s) | _TBD_ | `roofline_probe`, ≥8 GB stream, 5 runs, ±3% | |
| Peak achieved TFLOP/s (fp16, simdgroup_matrix) | _TBD_ | `roofline_probe` whole-step | |
| Ridge point (FLOP/byte) | _TBD_ | peak_FLOPS ÷ peak_bandwidth | |
| Sustained-clock behaviour | _TBD_ | `dvfs_probe` | |

## Tensor path (Phase 2)

| Field | Value | Method | Date |
|---|---|---|---|
| `matmul2d` ÷ `simdgroup_matrix`, fp16 | _TBD_ | T2.1, shapes 4096³/2048³/1024³/(128×5120²) | |
| Peak TFLOP/s via `matmul2d` | _TBD_ | T2.1 | |
| fp8 (E4M3) ÷ fp16 | _TBD_ | T2.2 | |
| int8 ÷ fp16 | _TBD_ | T2.2 | |
| Accumulator width | _TBD_ | T2.2, Rigel-style probe | |
| **Phase 3 go/no-go** | _TBD_ | ratio > 1.2× ⇒ go | |

Reference points for comparison — **[RIG]** on M4 Max: `matmul2d` 1.05–1.21×
over `simdgroup_matrix`, ceiling ~14.8 TFLOP/s inside the ALU limit; fp8 at
0.94× fp16 (emulated); accumulator ≥ fp32.

## Crossovers (Phase 1)

Fill from `benches/tune_device.py`. Each field needs its own arm-by-arm table in
the `device_tuning.hpp` house style before it goes into the source.

| Field | M1 Max default | M5 Pro measured | Control agrees? | Date |
|---|---|---|---|---|
| `qmm_min_batch` | 8 | _TBD_ | | |
| `qmm_min_batch_emulated` | _see [DT]_ | _TBD_ | | |
| routed / MoE crossovers | _see [DT]_ | _TBD_ | | |
| `sdpa_tile_min_rows_per_request` | _see [DT]_ | _TBD_ | | |

## Model throughput — three-way baseline (T0.3, 2026-08-19)

**Method.** `benches/three_way.py`, one invocation per (rep, prompt length,
model): every invocation runs one rep of each engine back to back (arms
alternated), 5 invocations per cell, reps outermost. Each invocation runs two
latency-mode shapes of 4 sequential requests (warmup 2): `max_tokens=1` and
`max_tokens=128`. Per rep, from mean request latencies:

    prefill tok/s = prompt_tokens / lat(mt=1)              # launch-inclusive
    decode  tok/s = Δoutput_tokens / (lat(mt=128) − lat(mt=1))

Cells are mean ± stddev over the 5 reps. Checkpoints: `mlx-community/…-4bit`
served by pie and mlx-lm 0.31.3 (MLX 0.32.0); `unsloth/…-Q4_0.gguf` by
llama.cpp b9960 (`Q4_0` is the closest bpw match — see `three_way.py`'s
header). Machine gate before the sweep: `roofline_probe` streaming roof
283 GB/s cold one-shot (recorded warm roof 294–298); GPU tenancy held
exclusively (peer sessions coordinated off the device). pie ring sized by
`--max-model-len 16384` — see the harness-fix note below. Raw JSONL + sweep
log: `docs/bench-archives/t03_*` (local, gitignored).

**Read the numbers with three caveats.**

1. Prefill is **TTFT-derived and launch-inclusive** — engine-server overhead
   is in the denominator. It is comparable across these engines at the same
   length; it is NOT comparable to [M5] Tables 2–3, whose pp is bench-style
   raw batch prefill (T0.4 reproduces BaseRT with its own instrument instead).
2. All runs are `--no-ignore-eos` (mlx-lm cannot pin output length). On
   Llama-3.2-1B the engines EOS at different points (actual-work table
   below), so its decode cells average over fewer tokens and carry more
   noise; Qwen ran the full 128 everywhere.
3. pie applies its own chat template (~2 extra prompt tokens here; per-engine
   prompt counts recorded below).

### Prefill tok/s (launch-inclusive)

| Model | Engine | pp128 | pp256 | pp512 | pp1024 | pp2048 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B q4 | pie | 5274 ± 32 | 6235 ± 47 | 6703 ± 34 | 7101 ± 48 | 7108 ± 11 |
| Qwen3-0.6B q4 | mlx-lm | 1794 ± 18 | 3337 ± 14 | 5314 ± 203 | 7964 ± 204 | 9467 ± 99 |
| Qwen3-0.6B q4 | llama.cpp | 8053 ± 533 | 10389 ± 489 | 10759 ± 433 | 11141 ± 226 | 9554 ± 112 |
| Llama-3.2-1B q4 | pie | 3108 ± 18 | 3278 ± 18 | 3351 ± 14 | 3419 ± 12 | 3347 ± 26 |
| Llama-3.2-1B q4 | mlx-lm | 2486 ± 4 | 3366 ± 732 | 5575 ± 31 | 7589 ± 45 | 8454 ± 291 |
| Llama-3.2-1B q4 | llama.cpp | 6574 ± 272 | 7459 ± 332 | 8054 ± 180 | 8435 ± 50 | 7776 ± 284 |

### Decode tok/s

| Model | Engine | pp128 | pp256 | pp512 | pp1024 | pp2048 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B q4 | pie | **379.5 ± 1.5** | **368.1 ± 3.4** | **351.9 ± 6.4** | **327.2 ± 2.6** | **285.6 ± 2.9** |
| Qwen3-0.6B q4 | mlx-lm | 345.6 ± 2.4 | 296.7 ± 88.9 | 325.1 ± 6.0 | 291.7 ± 2.6 | 252.5 ± 2.8 |
| Qwen3-0.6B q4 | llama.cpp | 349.9 ± 1.5 | 330.0 ± 26.3 | 330.6 ± 3.1 | 310.3 ± 1.6 | 283.2 ± 2.2 |
| Llama-3.2-1B q4 | pie | **291.6 ± 2.9** | **281.3 ± 6.2** | 269.3 ± 3.7 | 246.8 ± 0.8 | 196.7 ± 29.5 |
| Llama-3.2-1B q4 | mlx-lm | 281.6 ± 3.2 | 373.8 ± 214.1 | 269.5 ± 4.5 | 262.5 ± 1.9 | 238.1 ± 26.9 |
| Llama-3.2-1B q4 | llama.cpp | 267.1 ± 5.4 | 279.7 ± 14.9 | 267.3 ± 7.9 | 260.3 ± 2.8 | 235.4 ± 38.9 |

### Actual work per cell (mean prompt tokens / mean output tokens at mt=128)

| Model | Engine | pp128 | pp256 | pp512 | pp1024 | pp2048 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B q4 | pie / mlx / ll.cpp | 152/128 all | 278/128 all | 530/128 all | 1043/128 all | 2069/128 all |
| Llama-3.2-1B q4 | pie | 177/128 | 303/128 | 555/128 | 1068/128 | 2094/128 |
| Llama-3.2-1B q4 | mlx-lm | 175/84 | 301/22 | 553/128 | 1066/48 | 2092/99 |
| Llama-3.2-1B q4 | llama.cpp | 175/49 | 301/8 | 553/19 | 1066/46 | 2092/88 |

**What the table says.** pie holds the best decode at every Qwen cell
(+8–13% over mlx-lm, +1–9% over llama.cpp) and the short-context Llama cells;
its long-context Llama decode trails (196.7 at pp2048, noisy — the unequal
EOS behaviour makes that cell soft for every engine). pie's launch-inclusive
prefill is flat with length (~7.1k on 0.6B, ~3.4k on 1B): fastest startup at
short prompts (24 ms TTFT at pp128 vs mlx-lm's 71 ms) but a per-layer
dispatch-bound plateau that mlx-lm passes by pp1024. This is the small-dense
regime — the same tree measured the opposite prefill ordering on the 30B MoE
(`docs/HANDOVER.md`) — and it is the M1-constants baseline that Phase 1's
tuning entry and T5.1 re-measure.

**Harness fix this task required** (in `benches/three_way.py`, same PR):
pie_bench's own `--max-model-len` default (2048) sizes the Metal KV ring at
exactly 64 pages, so the 2048-token arm was refused ("allocation of 65 units
can never fit") while the other engines ran it. `three_way.py` now forwards
its `--max-model-len` to pie exactly as it already did to llama.cpp, and
grew a `--pie-extra` passthrough for pie-only knobs. Ring-size control: one
qwen/pp512 pie rep under the new ring — prefill 6716 vs recorded 6702 ± 34,
decode 354.4 vs 351.9 ± 6.4 — so pre-fix cells stand.

### Qwen3.6-27B (deferred to Phase 1/5)

| Model | Quant | Engine | Prefill tok/s | Decode tok/s | Date |
|---|---|---|---|---|---|
| Qwen3.6-27B | q4 | pie (case-9 constants) | _TBD_ | _TBD_ | |
| Qwen3.6-27B | q4 | pie (M5 entry) | _TBD_ | _TBD_ | |

BaseRT's own published figures are in **[M5] Tables 1–3**; ±10% reproduction
with BaseRT's own instrument is the T0.4 gate (next section when measured).
