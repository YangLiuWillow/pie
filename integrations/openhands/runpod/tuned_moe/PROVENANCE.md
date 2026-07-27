# Provenance of `E=128,N=768,device_name=NVIDIA_A100-SXM4-80GB.json`

**This file was not produced by our own autotuner.** It is SGLang's published
tuned config for the **NVIDIA A800-SXM4-80GB**, renamed to the A100 device name.
State this in any writeup that uses the `fair` tier.

## Source

```
repo:   github.com/sgl-project/sglang  (branch main, fetched 2026-07-27)
path:   python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/
        triton_3_2_0/E=128,N=768,device_name=NVIDIA_A800-SXM4-80GB.json
```

## Why an A800 config is an A100 config here

A800 is the China-export variant of the **same GA100 silicon**: identical
compute, identical 80 GB HBM2e, ~2 TB/s memory bandwidth. The *only* difference
is NVLink, cut from 600 GB/s to 400 GB/s.

We serve **tp=1 on a single GPU**, so NVLink is never touched by the fused MoE
kernel. For Triton autotuning — which depends on SM count, shared memory per SM,
and memory bandwidth — the two parts are equivalent.

Corroborating evidence in the file itself: it caps `num_stages` at 2-4, whereas
the shipped H200 config uses `num_stages=5` at bs=1. That is what sm_80 tuning
looks like — 164 KB of shared memory per SM will not hold as deep a pipeline as
Hopper's 228 KB.

## Why the format is compatible

vLLM's `get_moe_configs` carries the comment
`Adapted from: https://github.com/sgl-project/sglang/pull/2628`. The JSON schema
is identical (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`), and
vLLM pops `triton_version` and ignores it (`fused_moe.py`, `get_moe_configs`).

Coverage matches what our own autotuner would have produced: all 18 batch-size
keys, 1 -> 4096.

## Caveats to disclose

1. Tuned by SGLang, not by us.
2. Tuned under **Triton 3.2.0**; we run **Triton 3.6.0**. vLLM ignores the
   version field, and configs are not version-gated in vLLM's loader.
3. The `device_name` in the filename was changed from A800 to A100. This is a
   deliberate substitution, justified above, and **verified by measurement** --
   see `VALIDATION.md` alongside this file.

## Why this instead of running the autotuner

`benchmark_moe.py --tune` on this pod measured ~57 min for the *first* of 18
batch-size sweeps, with later sweeps slower: 15-24 h+ total, and nothing is
written until all 18 finish. More importantly, a self-run tune is unverifiable by
a reader, whereas a borrowed config plus a published before/after kernel-time
measurement is.
