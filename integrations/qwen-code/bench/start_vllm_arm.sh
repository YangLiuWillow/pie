#!/usr/bin/env bash
# Boot the vLLM arm for the A/B on :18000 — the §5b "fair tier": prefix
# caching + CUDA graphs (both default-on), native qwen3_coder tool parser.
# vLLM pinned to the old run's version for comparability.
# Usage: bash start_vllm_arm.sh [model-id]
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
VENV=/workspace/vllm-env
VLLM_PIN="${VLLM_PIN:-0.25.1}"

# Driver gate. vllm 0.25.1 ships CUDA-13-linked wheels (torch cu130,
# extensions against libcudart.so.13) and needs driver >= 580; the pod that
# ran the old §5b A/B had 580.159.04. On an older driver torch refuses to
# init ("NVIDIA driver ... too old"), and force-swapping torch to cu128
# does NOT help — vllm's own extensions still want libcudart.so.13. Pick a
# pod with a new enough driver, or set VLLM_PIN to a cu12-era release and
# record the version delta as a caveat in the results.
DRV_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if [ "${DRV_MAJOR:-0}" -lt 580 ] && [ "$VLLM_PIN" = "0.25.1" ]; then
    echo "ERROR: driver $DRV_MAJOR.x is too old for vllm $VLLM_PIN (needs >= 580)." >&2
    echo "       Re-provision on a newer-driver pod, or set VLLM_PIN=<cu12-era release>." >&2
    exit 2
fi

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
