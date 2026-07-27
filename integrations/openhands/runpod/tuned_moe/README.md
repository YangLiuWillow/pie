# `tuned_moe/` — the MoE kernel config behind the `fair` tier

This directory is the **version-controlled record** of the tuned fused-MoE
config used by the vLLM `fair` arm. It is not read at runtime.

| | |
|---|---|
| **Runtime location** | `/workspace/tuned_moe/` on the benchmark pod |
| **Selected by** | `VLLM_TUNED_CONFIG_FOLDER=/workspace/tuned_moe` |
| **This directory** | the same three files, committed so the arm is reproducible |

The two copies are identical. The runtime one lives outside the repo because
`/workspace` is the MooseFS volume the pod actually serves from.

## Files

- **`E=128,N=768,device_name=NVIDIA_A100-SXM4-80GB.json`** — the config. The
  filename is exactly what `get_config_file_name(E=128, N=768, dtype=None)`
  computes for Qwen3-Coder-30B-A3B at tp=1 in bf16 on this device; vLLM will not
  find it under any other name.
- **`PROVENANCE.md`** — where it came from (SGLang's A800 config), why an A800
  config is valid on an A100, and the caveats to disclose.
- **`VALIDATION.md`** — the measured before/after on the actual A100:
  **-8.3% overall**, faster at 17 of 18 batch sizes.

## Using it

```bash
export VLLM_TUNED_CONFIG_FOLDER=/workspace/tuned_moe
```

`get_moe_configs` searches that folder **before** vLLM's packaged `configs/`
dir, so vLLM's own files are never modified and the tier is selected purely by
whether the variable is exported. Confirm it took effect — vLLM names the file
it loaded:

```
INFO [fused_moe.py:1093] Using configuration from
  /workspace/tuned_moe/E=128,N=768,device_name=NVIDIA_A100-SXM4-80GB.json for MoE layer.
```

If you instead see `Using default MoE config. Performance might be
sub-optimal!`, the variable did not take and the run is **not** a `fair` arm.
`assert_vllm_fair.sh` hard-fails on that.

## Do not export this for the default-MoE tiers

`crippled` and `graphs-only` are both defined as *default* MoE. Leave
`VLLM_TUNED_CONFIG_FOLDER` unset for them — `assert_vllm_fair.sh` hard-fails
either tier if a tuned config is live, so this is verified rather than assumed.

## Caveat carried from the parent experiment

This config makes the vLLM baseline faster and is measured to do so. It has no
bearing on the separate, larger finding in `RUN_STATE.md` §0: Pie's fast
attention paths are gated to compute capability ≥ 9, so the A100 Pie arm does
not measure Pie properly regardless of how well the baseline is tuned.
