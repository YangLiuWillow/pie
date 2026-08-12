#!/usr/bin/env bash
# Boot the pie arm for the A/B: pie serve (CUDA) + OpenAI shim on :8123.
# Usage: bash start_pie_arm.sh [config.toml]   (default: the H200 30B config)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CFG="${1:-$REPO/integrations/qwen-code/bench/pie_config_h200_30b.toml}"
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda-12.9/lib64}"
export LD_LIBRARY_PATH="$CUDA_LIB:${LD_LIBRARY_PATH:-}"

PIE_CONFIG="$CFG" "$REPO/target/release/pie" serve > /workspace/pie_serve.log 2>&1 &
echo $! > /workspace/pie_serve.pid
for _ in $(seq 1 180); do nc -z 127.0.0.1 18080 && break; sleep 2; done
nc -z 127.0.0.1 18080 || { echo "pie serve failed:"; tail -20 /workspace/pie_serve.log; exit 1; }

python3 "$REPO/integrations/qwen-code/shim.py" --pie ws://127.0.0.1:18080 --port 8123 \
    --wasm "$REPO/tests/inferlets/target/wasm32-wasip2/release/chat_completions.wasm" \
    --manifest "$REPO/tests/inferlets/chat-completions/Pie.toml" > /workspace/shim.log 2>&1 &
echo $! > /workspace/shim.pid
for _ in $(seq 1 60); do nc -z 127.0.0.1 8123 && break; sleep 2; done
nc -z 127.0.0.1 8123 || { echo "shim failed:"; tail -20 /workspace/shim.log; exit 1; }
echo "pie arm up: http://127.0.0.1:8123/v1"
