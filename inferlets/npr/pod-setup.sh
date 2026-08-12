#!/usr/bin/env bash
# NPR-on-pie bootstrap for a RunPod CUDA pod (Ubuntu-based PyTorch/CUDA image).
#
#   bash pod-setup.sh            # full setup: deps, engine, inferlet, model, config
#
# Afterwards:
#   cd /workspace/pie
#   PIE_CONFIG=/workspace/npr-cuda.toml nohup ./target/release/pie serve > /workspace/serve.log 2>&1 &
#   /workspace/venv/bin/python inferlets/npr/client.py --input '{"selftest": true}'
#   /workspace/venv/bin/python inferlets/npr/client.py --input '{"max_new_tokens": 30000}'
set -euo pipefail

REPO=${REPO:-https://github.com/YangLiuWillow/pie.git}
BRANCH=${BRANCH:-npr-inferlet}
WORK=${WORK:-/workspace}
MODEL_REPO=${MODEL_REPO:-bigai-NPR/NPR-4B}

echo "== apt deps =="
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake build-essential pkg-config libssl-dev python3-venv git curl

# driver/cuda needs CMake >= 3.23; Ubuntu 22.04 ships 3.22. Kitware's binary
# tarball is the least invasive fix (no apt repo, no pip into the system env).
CMAKE_MIN=3.23
have_cmake=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
if [ -z "$have_cmake" ] || [ "$(printf '%s\n%s\n' "$CMAKE_MIN" "$have_cmake" | sort -V | head -1)" != "$CMAKE_MIN" ]; then
    echo "cmake ${have_cmake:-none} < $CMAKE_MIN — installing a newer one"
    CMAKE_VER=${CMAKE_VER:-3.31.6}
    curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-x86_64.tar.gz" \
        | tar xz -C /opt
    export PATH="/opt/cmake-${CMAKE_VER}-linux-x86_64/bin:$PATH"
    cmake --version | head -1
fi

# nvcc is not on PATH in the plain nvidia/cuda images.
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

echo "== rust =="
if ! command -v cargo >/dev/null 2>&1; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustup target add wasm32-wasip2

echo "== clone =="
mkdir -p "$WORK"
cd "$WORK"
if [ ! -d pie ]; then
    git clone --branch "$BRANCH" "$REPO" pie
else
    (cd pie && git fetch origin "$BRANCH" && git checkout "$BRANCH" && git pull)
fi
cd pie

echo "== build engine (cuda) =="
cargo build --release -p pie-bin --no-default-features --features driver-cuda

echo "== build npr inferlet =="
(cd inferlets/npr && cargo build --target wasm32-wasip2 --release)

echo "== python client =="
python3 -m venv "$WORK/venv"
# websockets + blake3 for client.py / evals; torch + safetensors for the
# fp32 -> bf16 cast (pie's loader does not cast).
"$WORK/venv/bin/pip" install -q -e client/python "huggingface_hub[cli]" \
    websockets blake3 torch safetensors numpy

echo "== model download ($MODEL_REPO) =="
"$WORK/venv/bin/hf" download "$MODEL_REPO" >/dev/null
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--${MODEL_REPO//\//--}/snapshots/*/ | head -1)
echo "snapshot: $SNAP"

# The published NPR-4B is fp32; serving it directly fails with
# "gemm_act_x_w: unsupported dtype combo (act=bf16, w=fp32, y=bf16)".
BF16="$WORK/$(basename "$MODEL_REPO")-bf16"
if [ ! -f "$BF16/config.json" ]; then
    echo "== cast to bf16 -> $BF16 =="
    "$WORK/venv/bin/python" inferlets/npr/convert_bf16.py "$SNAP" "$BF16"
fi
SNAP="$BF16"

echo "== config =="
cat > "$WORK/npr-cuda.toml" <<EOF
[gateway]
listen = "127.0.0.1:8092"

[worker.server]
host = "127.0.0.1"
port = 8093

[worker.auth]
enabled = false

[[worker.model]]
name = "default"
hf_repo = "$SNAP"

[worker.model.driver]
type = "cuda_native"
device = ["cuda:0"]

[worker.model.driver.options]
gpu_mem_utilization = 0.90
EOF

echo "== done =="
echo "start:  cd $WORK/pie && PIE_CONFIG=$WORK/npr-cuda.toml nohup ./target/release/pie serve > $WORK/serve.log 2>&1 &"
echo "test:   $WORK/venv/bin/python $WORK/pie/inferlets/npr/client.py --input '{\"selftest\": true}'"
echo "eval:   $WORK/venv/bin/python $WORK/pie/inferlets/npr/evals/run_eval.py --k 8 --concurrency 32"
