set -uo pipefail
REPO=/Users/liuyang/Documents/Liszt_ai/pie-opencode
OUT=/private/tmp/claude-501/-Users-liuyang-Documents-Liszt-ai-pie-opencode/e6929145-9422-4835-9670-317e9f8eebd9/scratchpad/mlx5
INST="django__django-12276 django__django-13028 django__django-13089 django__django-14373 django__django-15569"
mkdir -p "$OUT"; cd "$REPO/integrations/opencode"
pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null; pkill -f session_shim.py 2>/dev/null
pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null; pkill -f mlx_lm.server 2>/dev/null; sleep 6

nohup /tmp/venv-mlxlm/bin/mlx_lm.server --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
   --port 8001 --host 127.0.0.1 > "$OUT/server.log" 2>&1 &
until curl -s -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do sleep 3; done; echo "mlx-lm up"

t0=$(date +%s)
/Users/liuyang/.venv-vllm-metal/bin/python run_swebench.py --instances $INST \
    --model "mlx/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit" \
    --label mlx5 --out "$OUT/preds-mlx5.jsonl" --timeout 1800 \
    2>&1 | tee "$OUT/drive.log" | grep -aE "^\[|ok in|timeout|error|non-empty" | tail -20
echo "[mlx5] wall $(( $(date +%s) - t0 ))s"

# ── VALIDITY GATE: an arm whose server died is not a result ──
echo "=== validity check ==="
DEAD=$(grep -acE "Insufficient Memory|EngineDead|kIOGPUCommandBuffer|Traceback" "$OUT/server.log" 2>/dev/null)
UNREACH=$(grep -acE "Cannot connect to API|Unable to connect" "$OUT/drive.log" 2>/dev/null)
ALIVE=$(curl -s -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 && echo 1)
echo "server_errors=$DEAD  opencode_unreachable=$UNREACH  server_alive_at_end=$ALIVE"
if [ "$DEAD" != "0" ] || [ "$UNREACH" != "0" ] || [ "$ALIVE" != "1" ]; then
  echo "!!! ARM INVALID — the server died or was unreachable. Do not quote these numbers."
else
  echo "arm VALID"
fi
pkill -f mlx_lm.server
echo "=== DONE ==="
