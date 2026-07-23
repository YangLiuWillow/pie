#!/usr/bin/env bash
#SBATCH --job-name=abpie-cuda-native-13
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/ab_cuda_native_13_%j.out

# A/B PIE ARM: pie-glued OpenHands on the NATIVE CUDA driver + self-keyed
# content-addressed APC (commit c9fdf557), across the same 13 instances the
# litellm+OH baseline resolved with the same model. Head-to-head partner of
# bench_litellm_baseline_30bcoder_13.sbatch (litellm+vLLM arm). Measures
# accuracy (patch → apptainer score) AND speed (per-turn KV reuse + wallclock)
# of the APC path vs the stock litellm baseline.
#
# Swaps vs bench_parity_30bcoder_13: cuda_native driver (repo pie binary built
# with driver-cuda) instead of the vllm-driver grammarfix binary; the proven
# --python-tool-parser (host qwen3_coder XML parser) path.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_BIN=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
export PIE_PORT=18097
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_ab_cuda_native_13
export PIE_DEBUG_LOG=logs/pie_debug_ab_cuda_native_13_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-cuda-native-apc+qwen3-coder-30b-t0 \
OUTPUT=predictions/ab_cuda_native_13_${SLURM_JOB_ID}.jsonl \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-12276 \
    --instance-id django__django-13028 \
    --instance-id django__django-13089 \
    --instance-id django__django-14373 \
    --instance-id django__django-15569 \
    --instance-id django__django-16485 \
    --instance-id matplotlib__matplotlib-22719 \
    --instance-id pydata__xarray-4075 \
    --instance-id pydata__xarray-4966 \
    --instance-id scikit-learn__scikit-learn-10908 \
    --instance-id scikit-learn__scikit-learn-12973 \
    --instance-id scikit-learn__scikit-learn-13496 \
    --instance-id sympy__sympy-19346 \
    --python-tool-parser \
    --temperature 0 \
    --max-iterations 100

PRED=predictions/ab_cuda_native_13_${SLURM_JOB_ID}.jsonl
echo ""
echo "=== session KV-reuse (hit-rate) across the run ==="
grep -oE "modes=\{[^}]*\}" logs/ab_cuda_native_13_${SLURM_JOB_ID}.out | sort | uniq -c || true
echo "=== kv-verify errors (want none) ==="
grep -c "kv-verify:" logs/ab_cuda_native_13_${SLURM_JOB_ID}.out || echo 0
echo ""
echo "=== non-empty patch summary ==="
.venv/bin/python - "$PRED" <<'PY'
import json,sys
n=ne=0
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    d=json.loads(line); p=(d.get("model_patch") or "").strip()
    n+=1; ne+= 1 if p else 0
    print(f"{d.get('instance_id'):40s} patch_bytes={len(p)}")
print(f"\nTOTAL {n}  non-empty {ne}")
PY
