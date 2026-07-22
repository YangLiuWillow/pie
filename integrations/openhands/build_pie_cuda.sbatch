#!/usr/bin/env bash
#SBATCH --job-name=pie_cuda_build
#SBATCH --partition=day
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/pie_cuda_build_%j.log
# Build `pie` with the native CUDA driver (driver-cuda) on a compute node.
# GPU-free: the CUDA arch is set explicitly (sm_120), so nvcc cross-compiles
# without a device. The CPM cache under $REPO/.cpm-cache is pre-populated, so
# no network is needed.
set -euo pipefail

REPO=/nfs/roberts/project/pi_ql324/ly337/pie
mkdir -p "$REPO/integrations/openhands/logs"
cd "$REPO"

export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-32}"
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-32}"

bash integrations/openhands/build_pie_cuda.sh
