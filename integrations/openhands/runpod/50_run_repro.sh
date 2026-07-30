#!/usr/bin/env bash
# Wrapper for 50_overcommit_repro.py: boots pie serve on the overcommit toml,
# runs the repro, snapshots the tail of the server log on wedge.
set -uo pipefail
source /workspace/pie-bench-env.sh
RUNPOD=/workspace/pie/integrations/openhands/runpod
LOGS=/workspace/pie/integrations/openhands/logs
TS=$(date +%Y%m%d_%H%M%S)
SRV=$LOGS/repro_pie_serve_${TS}.log
export PIE_CUDA_KV_PAGE_SIZE=32
/workspace/pie/target/release/pie serve \
    --config "$RUNPOD/pie_cuda_native_config_30b_moe_h100_overcommit.toml" \
    --port 18097 --no-auth > "$SRV" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 120); do
    grep -q "pie-server serving on" "$SRV" 2>/dev/null && break
    kill -0 $SRV_PID 2>/dev/null || { echo "SERVER-DIED"; tail -5 "$SRV"; exit 1; }
    sleep 5
done
grep -m1 "memory planner:" "$SRV"
grep -m1 "swap_pool=" "$SRV"
/root/venvs/harness/bin/python "$RUNPOD/50_overcommit_repro.py" "$@"
RC=$?
echo "REPRO-EXIT $RC (server log: $SRV)"
if [ $RC -eq 2 ]; then
    echo "=== server log tail at wedge ==="; tail -30 "$SRV"
fi
kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null
exit $RC
