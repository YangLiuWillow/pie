#!/usr/bin/env bash
#SBATCH --job-name=cuda-native-smoke
#SBATCH --account=pi_ql324
#SBATCH --partition=gpu_rtx6000
#SBATCH --qos=normal
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands/logs/cuda_native_smoke_%j.out

# Smoke test: OpenHands↔Pie on the native CUDA driver (type = "cuda_native")
# instead of the Python vLLM subprocess driver. Single SWE-bench instance,
# Qwen3-Coder-30B-A3B. Confirms the cuda_native-enabled `pie` boots, loads the
# model, and drives a full agent trajectory end to end on a Blackwell GPU.
#
# Requires target/release/pie built with --features driver-portable,driver-cuda
# (integrations/openhands/build_pie_cuda.sbatch). PIE_BIN is left at the run
# script default, which points at that repo binary.
set -euo pipefail
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands

# Native CUDA driver needs the toolkit + NCCL runtime .so at load time.
module load CUDA/12.8.0 NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

export PIE_PORT=18082
export HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/.cache/huggingface
export SWEB_WS_ROOT=/tmp/sweb_cuda_native_smoke

BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
LABEL=cuda-native+qwen3-coder-30b-smoke \
OUTPUT=predictions/cuda_native_smoke_${SLURM_JOB_ID}.jsonl \
OUTPUT_PREFIX=cuda_native_smoke \
REQUEST_TIMEOUT_S=3600 \
  bash run_pie_backend.sh \
    --instance-id django__django-14373 \
    --temperature 0 \
    --max-iterations 100
