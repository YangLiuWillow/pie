#!/usr/bin/env bash
#SBATCH --job-name=pie-agent-30b-moe-full
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/swebench_agent_30b_moe_full_%j.out

# Pattern A full SWE-Bench evaluation: 50-problem deterministic subset.
# Qwen3-Coder-30B-A3B-Instruct (MoE: 30B total, 3B active) on 1x RTX Pro 6000 (96GB).
# Uses openhands-agent inferlet with JSON-schema-constrained output.
# --resume skips instances already present in OUTPUT from prior (preempted) runs.

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

OUTPUT=predictions/pie_agent_qwen3_coder_30b_moe_50.jsonl \
BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=pie-agent+qwen3-coder-30b-a3b \
OUTPUT_PREFIX=pie_agent_qwen3_coder_30b_moe_full \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --subset-size 50 \
    --max-iterations 50 \
    --resume
