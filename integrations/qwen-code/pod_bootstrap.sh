#!/usr/bin/env bash
# Idempotent GPU-pod bring-up for the qwen-code ↔ Pie stack (rewritten engine).
# Converges any Ubuntu CUDA pod (e.g. RunPod pytorch images) to a ready state:
#
#   toolkit ≥ 12.9  →  rust toolchain  →  pie-bin (driver-cuda)  →
#   chat_completions.wasm  →  python client deps  →  model imported
#
# Why the toolkit step exists: driver/cuda/src/ops/gemm.cpp uses
# cublasGemmGroupedBatchedEx (CUDA ≥ 12.5) and the cublasLt block-scale modes
# CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128/VEC128_32F (CUDA ≥ 12.9), so the
# stock cuda-12.4 pod images fail the build. Verified path: NVIDIA apt repo +
# cuda-toolkit-12-9 (2026-08-11, RTX 3090 pod, build green in 9 min).
#
# Usage (from the repo root on the pod):
#   bash integrations/qwen-code/pod_bootstrap.sh [hf-model-id]
# Env: CUDA_ARCH=86 overrides the sm autodetect; MODEL overrides the model arg.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:-${1:-Qwen/Qwen3-0.6B}}"
CUDA_HOME_WANT=/usr/local/cuda-12.9

# ── 1. CUDA toolkit ≥ 12.9 ──────────────────────────────────────────────────
nvcc_minor() { "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p'; }
NVCC=""
for c in "$CUDA_HOME_WANT/bin/nvcc" nvcc /usr/local/cuda/bin/nvcc; do
    v=$(nvcc_minor "$c" || true)
    [ -n "$v" ] && [ "$v" -ge 129 ] && { NVCC="$(command -v "$c" || echo "$c")"; break; }
done
if [ -z "$NVCC" ]; then
    echo "── installing cuda-toolkit-12-9 (stock image toolkit is too old for gemm.cpp)"
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb >/dev/null
    apt-get update -qq >/dev/null
    apt-get install -y -qq cuda-toolkit-12-9 >/dev/null
    NVCC="$CUDA_HOME_WANT/bin/nvcc"
fi
export CUDACXX="$NVCC"
CUDA_LIB="$(dirname "$(dirname "$NVCC")")/lib64"
export LD_LIBRARY_PATH="$CUDA_LIB:${LD_LIBRARY_PATH:-}"
echo "── nvcc: $NVCC ($($NVCC --version | tail -1))"

# ── 2. GPU arch ─────────────────────────────────────────────────────────────
if [ -z "${CUDA_ARCH:-}" ]; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
fi
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
echo "── CUDA arch: sm_$CUDA_ARCH"

# ── 3. Host tooling ─────────────────────────────────────────────────────────
command -v cargo >/dev/null 2>&1 || {
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none >/dev/null
}
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
python3 -c 'import cmake' 2>/dev/null || pip install -q cmake ninja
pip install -q websockets msgpack blake3 cryptography huggingface_hub
apt-get install -y -qq jq netcat-openbsd rsync >/dev/null 2>&1 || true

# ── 4. Build server + inferlet ──────────────────────────────────────────────
echo "── building pie-bin (driver-cuda, sm_$CUDA_ARCH)"
(cd "$REPO" && cargo build --release -p pie-bin --features driver-cuda)
echo "── building chat_completions.wasm"
(cd "$REPO/tests/inferlets" && cargo build --target wasm32-wasip2 --release -p chat-completions)

# ── 5. Model ────────────────────────────────────────────────────────────────
CFG="$REPO/integrations/qwen-code/pie_config_cuda.toml"
# Check for the converted .zt artifact specifically — `model list` also
# prints raw HF-cache snapshots, which a prefetch satisfies without there
# being anything servable (this skipped the 30B import on the H200 pod).
ARTIFACT="$HOME/.pie/models/$(echo "$MODEL" | sed 's|/|--|').zt"
[ -f "$ARTIFACT" ] || {
    echo "── importing $MODEL"
    "$REPO/target/release/pie" --config "$CFG" model import "$MODEL"
}
"$REPO/target/release/pie" --config "$CFG" doctor || true

cat <<EOF

── bootstrap done. Start the stack:
   PIE_CONFIG=$CFG LD_LIBRARY_PATH=$CUDA_LIB $REPO/target/release/pie serve &
   python3 $REPO/integrations/qwen-code/shim.py --pie ws://127.0.0.1:18080 --port 8123 \\
       --wasm $REPO/tests/inferlets/target/wasm32-wasip2/release/chat_completions.wasm \\
       --manifest $REPO/tests/inferlets/chat-completions/Pie.toml &
   python3 $REPO/integrations/qwen-code/test_acceptance.py --base http://127.0.0.1:8123
EOF
