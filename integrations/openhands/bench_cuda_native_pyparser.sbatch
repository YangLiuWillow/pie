#!/usr/bin/env bash
#SBATCH --job-name=cuda-native-pyparser
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/cuda_native_pyparser_%j.out

# Does the PROVEN patch-producing path work on the native CUDA driver?
# openhands-coder-session (Pattern B) + Qwen3-Coder-30B + --python-tool-parser
# (host-side vLLM qwen3_coder XML parser), the config that turned django-13028
# from a 0-byte to a 1039-byte RESOLVED patch on the vLLM driver. Only change
# here: CFG -> cuda_native, and PIE_BIN -> the cuda-enabled repo binary. This
# also exercises Pie session snapshots on the native driver for the first time.
#
# KV_VERIFY intentionally OFF for this first pass — we want a clean patch
# signal; a KV-correctness run is the follow-up.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

# The cuda_native binary dynamically links libcudart/cublas/nccl.
module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_BIN=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
export PIE_PORT=18099
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_cuda_native_pyparser
export PIE_DEBUG_LOG=logs/pie_debug_cuda_native_pyparser_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=cuda-native+qwen3-coder-30b-pyparser \
OUTPUT=predictions/cuda_native_pyparser_${SLURM_JOB_ID}.jsonl \
OUTPUT_PREFIX=cuda_native_pyparser \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-13028 \
    --python-tool-parser \
    --temperature 0 \
    --max-iterations 100

echo ""
echo "=== patch summary ==="
.venv/bin/python - "predictions/cuda_native_pyparser_${SLURM_JOB_ID}.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    d=json.loads(line); p=(d.get("model_patch") or "").strip()
    print(f"{d.get('instance_id'):40s} patch_bytes={len(p)}")
    if p: print(p[:800])
PY
