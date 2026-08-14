cd /Users/liuyang/Documents/Liszt_ai/pie-opencode
OUT=/tmp/ablate_results2.txt
: > $OUT
for K in "" ll_shared_gate_proj ll_shared_up ll_shared_down ll_shared_combine \
         qmv_gate qmv_up qmv_down qmv_in qmv_out silu_mul layer_out attn_gate \
         embed_gather q_split kv_append; do
  pkill -f "target/release/pie .*run" 2>/dev/null
  for _ in $(seq 1 20); do lsof -ti :18080 >/dev/null 2>&1 || break; sleep 1; done
  sleep 3
  LOG=$(PIE_METAL_ABLATE="$K" ./target/release/pie -c /tmp/rows-probe/config.toml run \
    --path runtime/engine/tests/inferlets/target/wasm32-wasip2/release/decode_rows_probe.wasm \
    --manifest runtime/engine/tests/inferlets/decode-rows-probe/Pie.toml 2>&1)
  MS=$(printf '%s' "$LOG" | grep -a "ctx=7424 rows=184 median_ms" | grep -oE "median_ms=[0-9.]+" | head -1 | cut -d= -f2)
  ARMED=$(printf '%s' "$LOG" | grep -ac "NOT DISPATCHED")
  printf "%-20s %8s  armed=%s\n" "${K:-BASELINE}" "${MS:-FAILED}" "$ARMED" >> $OUT
done
echo DONE >> $OUT
