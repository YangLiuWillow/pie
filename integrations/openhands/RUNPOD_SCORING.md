# Scoring the neutral-50 A/B on RunPod CPUs

The HPC Apptainer scorer kept queueing, so score these predictions with the
**official SWE-bench Docker harness** on a RunPod CPU pod instead. Both files
are in the standard SWE-bench prediction format
(`instance_id`, `model_name_or_path`, `model_patch`), so nothing from this repo's
`vendor/benchmarks` / `score_swebench_apptainer.py` path is needed.

## Files (committed under `integrations/openhands/predictions/`)

- `neutral_50_pie.jsonl` — Pie arm (cuda_native + self-keyed APC, Qwen3-Coder-30B-A3B, t0), 50 predictions.
- `baseline_t0_full_50.jsonl` — litellm/vLLM baseline arm, same 50 instances.

The 50 instance ids are implicit in each file (the harness scores exactly the
`instance_id`s present), so no separate subset list is required. This is the
neutral first-50 deterministic subset of SWE-bench Verified.

## Run (per pod)

Docker must be available on the pod. Install the harness and score each arm:

```bash
pip install swebench            # official Princeton harness
cd integrations/openhands

# Pie arm
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path predictions/neutral_50_pie.jsonl \
  --run_id neutral50_pie \
  --max_workers 8 \
  --cache_level env

# litellm baseline arm (same harness => apples-to-apples)
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path predictions/baseline_t0_full_50.jsonl \
  --run_id neutral50_litellm \
  --max_workers 8 \
  --cache_level env
```

Each run writes `<model_name_or_path>.<run_id>.json` with `resolved_ids`.
Tune `--max_workers` to the pod's CPU/RAM.

## Expected head-to-head

Partial HPC Apptainer scoring got through 29 of 50 before hitting an inode-quota
error (infra, not a Pie issue). On those 29: **Pie resolved 13, litellm 7** — Pie
got every instance litellm did plus 6 more, with zero only-litellm wins. The
prior full baseline was **litellm 13/50**. The remaining 21 are the harder tail
(matplotlib/sympy/sphinx/pylint), so treat the final gap as open until this full
50/50 Docker scoring completes.
