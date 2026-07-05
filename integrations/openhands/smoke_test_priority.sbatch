#!/usr/bin/env bash
#SBATCH --job-name=pie-oh-smoke-pgpu
#SBATCH --account=prio_ql324
#SBATCH --partition=priority_gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/smoke_%j.out

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands
bash run_pie_backend.sh --subset-size 2
