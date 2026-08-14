set -uo pipefail
REPO=/Users/liuyang/Documents/Liszt_ai/pie-opencode
OUT=/private/tmp/claude-501/-Users-liuyang-Documents-Liszt-ai-pie-opencode/e6929145-9422-4835-9670-317e9f8eebd9/scratchpad/tput
PIEPY=/Users/liuyang/.venvs/pie/bin/python
C=8; N=32; MT=128
mkdir -p "$OUT"; cd "$REPO/integrations/opencode"
source tools/require_quiet_gpu.sh

stop_all() {
  pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null; pkill -f session_shim.py 2>/dev/null
  pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null
  pkill -f mlx_lm.server 2>/dev/null; sleep 8
}
bench() { # tag, url, model
  python3 tput_bench.py --base-url "$2" --model "$3" --arm "$1" \
     --concurrency $C --requests $N --max-tokens $MT --out "$OUT/$1.json" 2>&1 | tee "$OUT/$1.txt"
}

# ── pie, strategy A: the stateless per-request mode, which is the one built
#    for independent concurrent traffic. Strategy B is a session-per-
#    conversation shim and is measured separately below.
stop_all; require_quiet_gpu 18 || exit 1
PIE_PYTHON=$PIEPY tools/boot_pie.sh tputA PIE_STRATEGY=a PIE_MODEL=qwen3-coder-30b \
   PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY 2>&1 | tail -1
bench pie-a http://127.0.0.1:8080 qwen3-coder-30b
arm_is_valid /tmp/pie_tputA.log "$OUT/pie-a.txt" http://127.0.0.1:8080/health

# ── pie, strategy B: same session inferlet the SWE-bench arm used.
stop_all; require_quiet_gpu 18 || exit 1
PIE_PYTHON=$PIEPY tools/boot_pie.sh tputB PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b \
   PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY 2>&1 | tail -1
bench pie-b http://127.0.0.1:8080 qwen3-coder-30b
arm_is_valid /tmp/pie_tputB.log "$OUT/pie-b.txt" http://127.0.0.1:8080/v1/models

# ── vLLM-metal ──
stop_all; require_quiet_gpu 18 || exit 1
VLLM_MAX_MODEL_LEN=65536 tools/boot_vllm.sh tputv 2>&1 | tail -1
bench vllm http://127.0.0.1:8000 qwen3-coder-30b
arm_is_valid /tmp/vllm_tputv.log "$OUT/vllm.txt" http://127.0.0.1:8000/v1/models

# ── mlx-lm ──
stop_all; require_quiet_gpu 18 || exit 1
nohup /tmp/venv-mlxlm/bin/mlx_lm.server --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
   --port 8001 --host 127.0.0.1 > "$OUT/mlx-server.log" 2>&1 &
until curl -s -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 3; done
bench mlx http://127.0.0.1:8001 mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
arm_is_valid "$OUT/mlx-server.log" "$OUT/mlx.txt" http://127.0.0.1:8001/v1/models

stop_all
echo "=== TPUT DONE ==="
