#!/usr/bin/env bash
# =============================================================================
# Generate a tuned fused-MoE Triton config for THIS GPU using vLLM's OWN
# autotuner (benchmark_moe.py). This is a documented, supported vLLM step, so it
# counts as "fair optimization effort" — it is vLLM's maintainers' tooling, not
# us hand-writing a kernel. Run it ONCE if 10_vllm_serve_fair.sh / assert_vllm_fair.sh
# reports "Using default MoE config" for the A100.
#
# The autotuner writes a JSON like
#   E=128,N=768,device_name=NVIDIA_A100-SXM4-80GB.json
# into vLLM's fused_moe configs directory, where vLLM then auto-loads it on the
# next serve. (E = #experts, N = intermediate size — both read from the model.)
#
# Usage: MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct bash 11_autotune_moe.sh
# Takes a while (it sweeps kernel configs). Do it before the timed runs.
# =============================================================================
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
PIE_VENV=${PIE_VENV:-$HOME/.venvs/pie-vllm}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
DTYPE=${DTYPE:-auto}

PY="$PIE_VENV/bin/python"

# Locate vLLM's benchmark_moe.py inside the installed package (path moved across
# versions; search both the package tree and any sibling benchmarks/ dir).
VLLM_DIR=$("$PY" -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
BENCH=$(find "$VLLM_DIR" "$VLLM_DIR/.." -maxdepth 4 -name "benchmark_moe.py" 2>/dev/null | head -1 || true)

if [ -z "${BENCH:-}" ]; then
    echo "Could not find benchmark_moe.py under $VLLM_DIR."
    echo "Fetch it from the vLLM source tree matching your installed version:"
    "$PY" -c "import vllm; print('  installed vLLM:', vllm.__version__)"
    echo "  https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py"
    echo "Then: $PY benchmark_moe.py --model $MODEL --tune --dtype $DTYPE"
    exit 1
fi

echo "=== Autotuning fused-MoE for $MODEL on this GPU ==="
"$PY" -c "import torch; print('  GPU:', torch.cuda.get_device_name(0))"
echo "  benchmark_moe.py: $BENCH"
echo "  (writes a tuned E=..,N=..,device_name=..A100..json into vLLM's config dir)"
echo ""

PYTHONPATH="" HF_HOME="$HF_HOME" \
  "$PY" "$BENCH" --model "$MODEL" --tune --dtype "$DTYPE"

echo ""
echo "Done. Relaunch 10_vllm_serve_fair.sh — the banner should no longer warn"
echo "about a default MoE config, and assert_vllm_fair.sh should pass clean."
