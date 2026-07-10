#!/usr/bin/env bash
#SBATCH --job-name=base-heval-32b
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/baseline_humanevalfix_32b_%j.out

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
HARNESS=benchmarks.run_humanevalfix \
  bash run_litellm_baseline.sh --subset-size 20
