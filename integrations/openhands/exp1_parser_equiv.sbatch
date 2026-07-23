#!/usr/bin/env bash
#SBATCH --job-name=exp1-parser
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/exp1_parser_%j.out

# Exp 1 — parser equivalence (offline; GPU node only for a consistent env).
# Parses captured raw model outputs with our python port AND the real vLLM
# qwen3_coder parser, diffs them. No pie serve needed.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
mkdir -p logs
PYTHONPATH="" .venv/bin/python exp1_parser_equiv.py
