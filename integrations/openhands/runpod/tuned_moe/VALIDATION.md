# Validation of the borrowed A800 MoE config on this A100

Measured **2026-07-27 17:47-17:52 UTC** on the actual benchmark pod
(NVIDIA A100-SXM4-80GB, sm_80, vLLM 0.25.1, Triton 3.6.0, tp=1, bf16).

Method: `benchmark_moe.py` **without `--tune`** benchmarks whatever config the
loader resolves. Run twice, identical otherwise:

```bash
# A: default heuristic config
env -u VLLM_TUNED_CONFIG_FOLDER python /workspace/benchmark_moe.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --dtype auto --tp-size 1

# B: borrowed A800 config
VLLM_TUNED_CONFIG_FOLDER=/workspace/tuned_moe python /workspace/benchmark_moe.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --dtype auto --tp-size 1
```

Run B confirms the file was actually used, rather than silently falling back:

```
INFO fused_moe.py:1093] Using configuration from
  /workspace/tuned_moe/E=128,N=768,device_name=NVIDIA_A100-SXM4-80GB.json for MoE layer.
```

Logs: `../pie/integrations/openhands/logs/moe_bench_{default,tuned}_20260727_174740.log`

## Result — faster at 17 of 18 batch sizes

| batch | default (us) | A800 cfg (us) | delta |
|---:|---:|---:|---:|
| 1 | 75.51 | 68.76 | **-8.9%** |
| 2 | 118.57 | 107.20 | **-9.6%** |
| 4 | 196.94 | 197.72 | +0.4% |
| 8 | 305.36 | 304.86 | -0.2% |
| 16 | 466.98 | 462.99 | -0.9% |
| 24 | 563.43 | 551.95 | -2.0% |
| 32 | 621.50 | 612.42 | -1.5% |
| 48 | 685.58 | 677.65 | -1.2% |
| 64 | 708.42 | 690.02 | -2.6% |
| 96 | 737.86 | 705.25 | -4.4% |
| 128 | 757.59 | 712.83 | -5.9% |
| 256 | 799.55 | 755.59 | -5.5% |
| 512 | 841.14 | 818.75 | -2.7% |
| 1024 | 1068.78 | 958.26 | **-10.3%** |
| 1536 | 1136.42 | 1070.30 | -5.8% |
| 2048 | 1501.36 | 1219.86 | **-18.7%** |
| 3072 | 2015.26 | 1649.02 | **-18.2%** |
| 4096 | 2505.40 | 2285.47 | -8.8% |

**Overall -8.3%** summed across the grid. By regime:

- **decode (bs 1-8): -2.6%**, but -8.9% at bs=1, which is the single-stream
  agent decode case this benchmark actually runs in.
- **prefill (bs >= 1024): -12.7%**, peaking at -18.7%.

Both regimes the A/B exercises get faster. The largest gains land in the
chunked-prefill range, which is where the baseline spends its non-reused
prompt work.

## Reading this honestly

- **Single-shot per point.** The +0.4% at bs=4 and -0.2% at bs=8 are inside
  run-to-run noise and should be read as "no change". The double-digit wins at
  1024/2048/3072 are far outside it.
- This validates that the borrowed config is **better than the default
  heuristic on this silicon**. It does **not** prove it is optimal — a native
  autotune could still beat it. If that matters for the writeup, run
  `benchmark_moe.py --tune` later and report the residual gap; it changes
  nothing already collected.
- The config selected at each batch size differs structurally from the default
  (e.g. bs=2048: `BLOCK_SIZE_N` 128 -> 256), so this is a real kernel change,
  not measurement drift.

See `PROVENANCE.md` for where the file came from and why an A800 config is
valid on an A100.
