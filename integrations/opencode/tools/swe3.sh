set -uo pipefail
REPO=/Users/liuyang/Documents/Liszt_ai/pie-opencode
OUT=/private/tmp/claude-501/-Users-liuyang-Documents-Liszt-ai-pie-opencode/e6929145-9422-4835-9670-317e9f8eebd9/scratchpad/swe3
PIEPY=/Users/liuyang/.venvs/pie/bin/python
INST="django__django-12276 django__django-13028"
mkdir -p "$OUT"; cd "$REPO/integrations/opencode"

stop_all() {
  pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null; pkill -f session_shim.py 2>/dev/null
  pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null
  pkill -f mlx_lm.server 2>/dev/null; sleep 6
}

run_arm() {  # $1 tag, $2 opencode model string
  echo "===== $1 ====="
  local t0=$(date +%s)
  /Users/liuyang/.venv-vllm-metal/bin/python run_swebench.py --instances $INST --model "$2" \
      --label "$1" --out "$OUT/preds-$1.jsonl" --timeout 1800 \
      2>&1 | tee "$OUT/$1.log" | grep -aE "instance|patch|empty|resolved|FAIL|error" | tail -20
  echo "[$1] wall $(( $(date +%s) - t0 ))s"
}

# ── pie, strategy B, with today's fixes ──
stop_all
PIE_PYTHON=$PIEPY "$REPO/integrations/opencode/tools/boot_pie.sh" sweB \
  PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
  PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY 2>&1 | tail -1
run_arm pie "pie/qwen3-coder-30b"

# ── vLLM-metal ──
stop_all
VLLM_MAX_MODEL_LEN=65536 "$REPO/integrations/opencode/tools/boot_vllm.sh" swev 2>&1 | tail -1
run_arm vllm "vllm/qwen3-coder-30b"

# ── mlx-lm 0.31.3 ──
stop_all
nohup /tmp/venv-mlxlm/bin/mlx_lm.server --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
   --port 8001 --host 127.0.0.1 > /tmp/mlxlm_swe.log 2>&1 &
until curl -s -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 3; done; echo "mlx-lm up"
run_arm mlx "mlx/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

stop_all
echo "=== DONE ==="; ls -la "$OUT"
