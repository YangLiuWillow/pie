# Measurements — Apple M5 Pro, 48 GB

Constants measured on this machine. **These supersede published figures.**
Public M5 bandwidth numbers conflict (153 GB/s vs ~120 GB/s) and neither is
credible for a Pro tier.

Every row must carry how it was taken. A number without a method is a guess with
a decimal point.

| Field | Value | Method | Date |
|---|---|---|---|
| macOS / Darwin | _TBD_ | `sw_vers` | |
| Metal toolchain | _TBD_ | `xcrun -sdk macosx metal --version` | |
| `apple_family` | _TBD_ | `descriptor_facts` test, newest-first resolve | |
| `gpu_core_count` | _TBD_ | IOKit `gpu-core-count` | |
| `DeviceTuning` block selected | _TBD_ | T0.2 — expected: default (M1 Max) | |

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

## Model throughput

Protocol: prompts 128/256/512/1024/2048; 128 generated tokens; 5 reps; mean ±
stddev; AC power; arms alternated.

| Model | Quant | Engine | Prefill tok/s | Decode tok/s | Date |
|---|---|---|---|---|---|
| Qwen3-0.6B | q4 | pie | _TBD_ | _TBD_ | |
| Qwen3-0.6B | q4 | mlx-lm 0.31.3 | _TBD_ | _TBD_ | |
| Qwen3-0.6B | q4 | llama.cpp b9960 | _TBD_ | _TBD_ | |
| Qwen3-0.6B | q4 | BaseRT | _TBD_ | _TBD_ | |
| Llama-3.2-1B | q4 | … | | | |
| Qwen3.6-27B | q4 | pie (M1 constants) | _TBD_ | _TBD_ | |
| Qwen3.6-27B | q4 | pie (M5 entry) | _TBD_ | _TBD_ | |

BaseRT's own published figures for its configs are in **[M5] Tables 1–3**;
±10% reproduction is the T0.4 gate.
