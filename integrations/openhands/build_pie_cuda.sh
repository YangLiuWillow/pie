#!/usr/bin/env bash
# Build `pie` with the native CUDA driver (driver-cuda) compiled in, so the
# OpenHands↔Pie backend can run on driver type = "cuda_native" instead of the
# Python vLLM subprocess driver.
#
# Target GPU: RTX Pro 6000 Blackwell (sm_120) — the gpu_rtx6000 partition that
# the OpenHands benchmarks run on. Override CMAKE_CUDA_ARCHITECTURES for other
# GPUs (h200=90, b200=100, rtx_5000_ada/l40s=89).
#
# Usage: bash integrations/openhands/build_pie_cuda.sh
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO"

# --- Toolchain modules ------------------------------------------------------
module load CMake/3.31.8-GCCcore-13.3.0 \
            CUDA/12.8.0 \
            NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.8.0

# --- Build environment ------------------------------------------------------
export CUDACXX="$EBROOTCUDA/bin/nvcc"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-120}"
# Persist CPM's GitHub downloads (flashinfer, cutlass, tomlplusplus, CLI11,
# nlohmann/json, zstd) so re-configures and offline compute-node builds reuse
# them instead of re-cloning.
export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-$REPO/.cpm-cache}"
# NCCL lives under an EasyBuild module, not /usr — point the driver's CMake +
# the linker at it (ships libnccl.so + nccl.h).
export PIE_NCCL_HOME="$EBROOTNCCL"

echo "=== pie CUDA-driver build ==="
echo "  nvcc:  $(nvcc --version | grep -oE 'release [0-9.]+')"
echo "  arch:  sm_$CMAKE_CUDA_ARCHITECTURES"
echo "  cpm:   $CPM_SOURCE_CACHE"
echo "  nccl:  $PIE_NCCL_HOME"
echo

cargo build -p pie-server --release --features driver-portable,driver-cuda

echo
echo "=== driver list ==="
"$REPO/target/release/pie" driver list
