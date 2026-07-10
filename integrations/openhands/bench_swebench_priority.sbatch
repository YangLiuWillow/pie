#!/usr/bin/env bash
#SBATCH --job-name=pie-swebench-p
#SBATCH --account=prio_ql324
#SBATCH --partition=priority_gpu
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/swebench_%j.out

set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

bash run_pie_backend.sh --subset-size 50
