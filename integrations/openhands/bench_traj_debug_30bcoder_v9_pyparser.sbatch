#!/usr/bin/env bash
#SBATCH --job-name=traj-debug-30bcoder-v9-pyparser
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/traj_debug_30bcoder_v9_pyparser_%j.out

# v9: Option B — host-side tool parsing via a verbatim port of vLLM's
# qwen3_coder parser (pie_openhands/qwen3coder_parser.py), enabled with
# --python-tool-parser. This implies unconstrained generation (no grammar,
# use_grammar=off) and skips the JSON few-shot examples, so the model emits
# its native Qwen3-Coder XML <function=…> format and we parse it EXACTLY as
# the litellm+vLLM baseline does — including the lenient back-off that
# recovers a call when the <tool_call> wrapper is malformed/missing.
#
# Target: django-13028, the ONE instance where the coder-session parity run
# (job 18951267) produced a 0-byte patch (10-turn narrate-without-act loop)
# while the litellm baseline (job 18962143) resolved it. Hypothesis: the
# inferlet's Rust decoder missed a <function=> call that vLLM's lenient
# parser extracts. PIE_DEBUG_LOG captures phase-1 raw text so we can see the
# exact generation; completions are logged too.
#
# NOTE: pure-Python change — PIE_BIN and the coder-session WASM are UNCHANGED
# from v8 (no runtime/inferlet rebuild). Watch two things in the output:
#   1) patch_bytes > 0 and resolves (score separately), and
#   2) session modes/prefill counts stay comparable to the parity run
#      (fresh:1/extended:N) — confirms cache reuse survived Option B.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

export PIE_BIN=/nfs/roberts/scratch/pi_ql324/ly337/target-grammarfix/release/pie
export PIE_PORT=18098
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_trajdebug_ws_30bcoder_v9

COMPLETIONS_DIR=logs/completions_debug_30bcoder_v9_pyparser_${SLURM_JOB_ID}
mkdir -p "$COMPLETIONS_DIR"
export PIE_DEBUG_LOG=logs/pie_debug_30bcoder_v9_pyparser_${SLURM_JOB_ID}.jsonl

BACKEND=pie-session \
KV_VERIFY=1 \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-session+qwen3-coder-30b-t0-v9-pyparser \
OUTPUT=predictions/traj_debug_30bcoder_v9_pyparser_${SLURM_JOB_ID}.jsonl \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-13028 \
    --python-tool-parser \
    --temperature 0 \
    --max-iterations 100 \
    --log-completions "$COMPLETIONS_DIR"

echo ""
echo "=== phase debug log ==="
wc -l "$PIE_DEBUG_LOG" 2>/dev/null || echo "(no debug log)"
echo "=== prediction patch size ==="
.venv/bin/python - "predictions/traj_debug_30bcoder_v9_pyparser_${SLURM_JOB_ID}.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    d=json.loads(line); p=d.get("model_patch") or ""
    md=d.get("_metadata") or {}
    print(f"{d.get('instance_id')}: patch_bytes={len(p)}")
    print("  pie_session:", md.get("pie_session"))
PY
