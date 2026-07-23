#!/usr/bin/env bash
#SBATCH --job-name=cuda-native-apc-kv
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/cuda_native_apc_kv_%j.out

# Validate the NEW content-addressed prefix cache (self-keyed APC) in the
# rebuilt openhands-coder-session inferlet, on the native CUDA driver.
# KV_VERIFY=1 asserts each turn's reused-prefix + suffix totals the full
# render (catches a bad prefix split). What to check in the log:
#   - session modes={fresh:1, extended:N}  -> the content-addressed cache HITS
#     across turns (my canonical assistant re-render matches the host resend)
#   - no "kv-verify:" error                -> length invariant holds
#   - a non-empty patch on django-13028    -> correct end to end
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_BIN=/nfs/roberts/project/pi_ql324/ly337/pie/target/release/pie
export PIE_PORT=18100
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_cuda_native_apc_kv
export PIE_DEBUG_LOG=logs/pie_debug_cuda_native_apc_kv_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=cuda-native+apc+kvverify \
OUTPUT=predictions/cuda_native_apc_kv_${SLURM_JOB_ID}.jsonl \
OUTPUT_PREFIX=cuda_native_apc_kv \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-13028 \
    --python-tool-parser \
    --temperature 0 \
    --max-iterations 100

echo ""
echo "=== session modes seen (hit-rate) ==="
grep -oE "modes=\{[^}]*\}" logs/cuda_native_apc_kv_${SLURM_JOB_ID}.out | sort | uniq -c || true
echo "=== kv-verify errors (want none) ==="
grep -c "kv-verify:" logs/cuda_native_apc_kv_${SLURM_JOB_ID}.out || echo 0
echo "=== patch summary ==="
.venv/bin/python - "predictions/cuda_native_apc_kv_${SLURM_JOB_ID}.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    d=json.loads(line); p=(d.get("model_patch") or "").strip()
    print(f"{d.get('instance_id'):40s} patch_bytes={len(p)}")
    if p: print(p[:600])
PY
