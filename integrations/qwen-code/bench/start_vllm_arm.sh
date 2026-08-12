#!/usr/bin/env bash
# Boot the vLLM arm for the A/B on :18000 — the §5b "fair tier": prefix
# caching + CUDA graphs (both default-on), native qwen3_coder tool parser.
# vLLM pinned to the old run's version for comparability.
# Usage: bash start_vllm_arm.sh [model-id]
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
VENV=/workspace/vllm-env
VLLM_PIN="${VLLM_PIN:-0.25.1}"

[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" show vllm >/dev/null 2>&1 || "$VENV/bin/pip" install -q "vllm==$VLLM_PIN"

"$VENV/bin/vllm" serve "$MODEL" \
    --port 18000 --dtype bfloat16 --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    > /workspace/vllm.log 2>&1 &
echo $! > /workspace/vllm.pid
for _ in $(seq 1 300); do
    curl -sf http://127.0.0.1:18000/v1/models >/dev/null && break; sleep 5
done
curl -sf http://127.0.0.1:18000/v1/models >/dev/null || { echo "vllm failed:"; tail -20 /workspace/vllm.log; exit 1; }
echo "vllm arm up: http://127.0.0.1:18000/v1"
