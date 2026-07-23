#!/usr/bin/env bash
#SBATCH --job-name=tbench-pie-easy
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/terminalbench/logs/tbench_pie_easy_%j.out

# TerminalBench evaluation: PIE agent (Pattern A) on easy tasks.
# Uses Apptainer containers (no Docker needed).

set -euo pipefail

TBENCH_DIR=/tmp/terminal-bench \
CFG=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/tests/fixtures/pie_cuda_vllm_config_30b_moe.toml \
TASK_LIST=easy \
MAX_STEPS=50 \
OUTPUT=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/terminalbench/results/pie_agent_easy.jsonl \
  bash /nfs/roberts/project/pi_ql324/ly337/pie/integrations/terminalbench/run_terminalbench.sh
