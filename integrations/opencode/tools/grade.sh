set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
OUT=/private/tmp/claude-501/-Users-liuyang-Documents-Liszt-ai-pie-opencode/e6929145-9422-4835-9670-317e9f8eebd9/scratchpad/swe3
PY=/tmp/venv-swebench/bin/python
cd "$OUT"
for arm in pie vllm mlx; do
  P="$OUT/preds-$arm.jsonl"
  [ -s "$P" ] || { echo "[$arm] no predictions file"; continue; }
  n=$(wc -l < "$P")
  echo "===== grading $arm ($n predictions) ====="
  $PY -m swebench.harness.run_evaluation \
      --dataset_name SWE-bench/SWE-bench_Verified \
      --predictions_path "$P" --max_workers 2 --run_id "swe2_$arm" \
      2>&1 | tail -25
done
echo "=== reports ==="; ls -la "$OUT"/*.json 2>/dev/null; ls -la *.json 2>/dev/null | head
