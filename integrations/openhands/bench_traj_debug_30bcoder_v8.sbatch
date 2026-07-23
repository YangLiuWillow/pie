#!/usr/bin/env bash
#SBATCH --job-name=traj-debug-30bcoder-v8
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/traj_debug_30bcoder_v8_%j.out

# v8: FIRST validation of openhands-coder-session on Qwen3-Coder-30B-A3B,
# the model OpenHands+litellm actually solves with (13/50 resolved).
# The pie runtime now implements Qwen3-Coder's native tool format
# (<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function>
# </tool_call>) — see runtime qwen3.rs ToolFormat::Coder, auto-selected
# for qwen3_moe models whose repo/path contains "Coder". django-14373 is
# in the litellm-resolved set. PIE_DEBUG_LOG captures phase-1 raw text:
# SUCCESS = model emits <function=…> calls natively (debug_phase1_marker
# should now catch <tool_call>), decoder parses them, non-empty patch.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
# Qwen3-Coder-30B weights are cached here.
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_trajdebug_ws_30bcoder_v8

COMPLETIONS_DIR=logs/completions_debug_30bcoder_v8_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR"
export PIE_DEBUG_LOG=logs/pie_debug_30bcoder_v8_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-session+qwen3-coder-30b-t0-trajdebug-v8 \
OUTPUT=predictions/traj_debug_30bcoder_v8_${SLURM_JOB_ID}.jsonl \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-14373 \
    --temperature 0 \
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"

echo ""
echo "=== phase debug log ==="
wc -l "$PIE_DEBUG_LOG" 2>/dev/null || echo "(no debug log)"
echo "=== prediction patch size ==="
.venv/bin/python - "predictions/traj_debug_30bcoder_v8_${SLURM_JOB_ID}.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    d=json.loads(line); p=d.get("model_patch") or ""
    print(f"{d.get('instance_id')}: patch_bytes={len(p)}")
PY
