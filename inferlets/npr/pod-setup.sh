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
"$WORK/venv/bin/pip" install -q -e client/python "huggingface_hub[cli]"

echo "== model download ($MODEL_REPO) =="
"$WORK/venv/bin/hf" download "$MODEL_REPO" >/dev/null
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--${MODEL_REPO//\//--}/snapshots/*/ | head -1)
echo "snapshot: $SNAP"

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
