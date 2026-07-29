#!/usr/bin/env bash
# =============================================================================
# Decode cost vs context, both arms — AGENT_HANDOVER_20260728.md §6a-ter.
#
# The decode gap needs a decomposition before any kernel work: how much of
# ms/token is context-independent (weights, MoE, router, launch) and how much
# scales with KV. The slope/intercept split says which half to attack, and the
# same split on vLLM says how much of each is actually recoverable. Every
# hypothesis this study has advanced without that decomposition has been wrong.
#
# The measurement is differencing (see context_sweep_client.py): the same prompt
# generated short and long, subtracted. Prefill, render and transport cancel, so
# the two arms are comparable despite entirely different client stacks.
#
# Arms run strictly sequentially — ~60 GB of weights each, neither may see the
# other's memory pressure.
#
# Usage:  bash 42_context_sweep.sh
#         SWEEP_CONTEXTS=1000,8000,32000 bash 42_context_sweep.sh
# =============================================================================
set -uo pipefail   # not -e: one arm failing must not lose the other

RUNPOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$RUNPOD_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RUNPOD_DIR/../../.." && pwd)"
source /workspace/pie-bench-env.sh

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-$HARNESS_DIR/logs}
OUT_DIR="$LOG_DIR/ctxsweep_${TS}"
mkdir -p "$OUT_DIR"

CONTEXTS=${SWEEP_CONTEXTS:-1000,4000,8000,16000,24000,32000}
REPS=${SWEEP_REPS:-3}
# mode=batch sweeps CONCURRENCY at fixed context instead of context at batch 1.
# It exists to find the R>17 CUDA-graph cliff on the decode-as-prefill path
# (qwen3_5_forward.cpp:1149 gates enable_graph on total_tokens, which for decode
# is the request count). pie arm only.
MODE=${SWEEP_MODE:-context}
CONCURRENCIES=${SWEEP_CONCURRENCIES:-1,4,8,16,20,24,32,48}
BATCH_CONTEXT=${SWEEP_BATCH_CONTEXT:-16000}
ARMS=${SWEEP_ARMS:-"pie vllm"}
PIE_PORT=${PIE_PORT:-18097}
VLLM_PORT=${VLLM_PORT:-18000}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
# pie_client lives in the harness venv; the pie-vllm venv does not have it.
CLIENT_PY=${CLIENT_PY:-/root/venvs/harness/bin/python}
PIE_BIN=${PIE_BIN:-$REPO_ROOT/target/release/pie}
CFG=${CFG:-$RUNPOD_DIR/pie_cuda_native_config_30b_moe_h200.toml}

echo "=== context sweep $TS"
echo "    arms=$ARMS contexts=$CONTEXTS reps=$REPS"
echo "    out=$OUT_DIR"

# Kill by PID only. `pkill -f 'pie serve'` matches this script's own command
# line and has cost this investigation time twice (§8).
wait_for_free_gpu() {
    local used
    for _ in $(seq 1 120); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${used:-99999}" -lt 2000 ] && { echo "    GPU free (${used} MiB)"; return 0; }
        sleep 5
    done
    echo "    WARNING: GPU still holds ${used} MiB — continuing anyway"
}

wait_for_line() {   # <log> <pattern> <label> <pid>
    for _ in $(seq 1 240); do
        grep -q "$2" "$1" 2>/dev/null && { echo "    $3 ready"; return 0; }
        kill -0 "$4" 2>/dev/null || { echo "    ERROR: $3 died"; tail -25 "$1"; return 1; }
        sleep 5
    done
    echo "    ERROR: $3 not ready within timeout"; tail -25 "$1"; return 1
}

for arm in $ARMS; do
    echo ""
    echo "=== [$(date +%H:%M:%S)] arm=$arm"
    wait_for_free_gpu
    srv_log="$OUT_DIR/${arm}_server.log"
    out_json="$OUT_DIR/${arm}.json"

    if [ "$arm" = "pie" ]; then
        export PIE_CUDA_KV_PAGE_SIZE=32     # auto_p32, matching the A/B arms
        "$PIE_BIN" serve --config "$CFG" --port "$PIE_PORT" --no-auth \
            > "$srv_log" 2>&1 &
        SRV_PID=$!
        wait_for_line "$srv_log" "pie-server serving on" "pie serve" "$SRV_PID" || {
            kill "$SRV_PID" 2>/dev/null; continue; }
        "$CLIENT_PY" "$RUNPOD_DIR/context_sweep_client.py" \
            --arm pie --uri "ws://127.0.0.1:$PIE_PORT" --repo "$REPO_ROOT" \
            --model "$MODEL" --contexts "$CONTEXTS" --reps "$REPS" \
            --mode "$MODE" --concurrencies "$CONCURRENCIES" \
            --batch-context "$BATCH_CONTEXT" \
            --out "$out_json" 2>&1 | tee "$OUT_DIR/${arm}_client.log"
    else
        PYTHONPATH="" HF_HOME=$HF_HOME \
          "$PIE_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --port "$VLLM_PORT" \
            --enable-prefix-caching \
            --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
            --max-model-len "${MAX_MODEL_LEN:-131072}" \
            --generation-config vllm \
            > "$srv_log" 2>&1 &
        SRV_PID=$!
        wait_for_line "$srv_log" "Application startup complete" "vllm" "$SRV_PID" || {
            kill "$SRV_PID" 2>/dev/null; continue; }
        "$CLIENT_PY" "$RUNPOD_DIR/context_sweep_client.py" \
            --arm vllm --base-url "http://127.0.0.1:$VLLM_PORT/v1" \
            --model "$MODEL" --contexts "$CONTEXTS" --reps "$REPS" \
            --out "$out_json" 2>&1 | tee "$OUT_DIR/${arm}_client.log"
    fi

    echo "    stopping $arm server (PID $SRV_PID)"
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
done

echo ""
echo "=== sweep done — $OUT_DIR"
"$CLIENT_PY" - "$OUT_DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
fits = {}
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    o = json.load(open(p))
    fits[o["arm"]] = o
    print(f"\n=== {o['arm']}")
    if o["rows"] and "concurrency" in o["rows"][0]:
        print(f"  {'R':>4} {'t_forward_ms':>13} {'aggregate_tok_s':>16}")
        for r in o["rows"]:
            t, a = r["t_forward_ms"], r["aggregate_tok_s"]
            print(f"  {r['concurrency']:>4} {'' if t is None else round(t,3):>13} "
                  f"{'' if a is None else round(a,1):>16}")
        continue
    print(f"  {'prompt_tok':>10} {'ms/token':>9} {'self':>8}")
    for r in o["rows"]:
        s = r.get("self_reported_decode_ms_per_token")
        print(f"  {str(r['prompt_tokens']):>10} "
              f"{'' if r['decode_ms_per_token'] is None else round(r['decode_ms_per_token'],3):>9} "
              f"{'' if s is None else round(s,3):>8}")
    print(f"  fit: {json.dumps(o['fit'])}")
if len(fits) == 2 and all(f.get("fit") for f in fits.values()):
    p, v = fits.get("pie", {}).get("fit"), fits.get("vllm", {}).get("fit")
    if p and v:
        print("\n=== pie / vllm")
        print(f"  intercept (fixed per-token):  {p['intercept_ms']:.3f} vs "
              f"{v['intercept_ms']:.3f} ms   ratio {p['intercept_ms']/v['intercept_ms']:.2f}x")
        print(f"  slope (per 1k KV tokens):     {p['slope_ms_per_ktoken']:.4f} vs "
              f"{v['slope_ms_per_ktoken']:.4f} ms  ratio "
              f"{p['slope_ms_per_ktoken']/v['slope_ms_per_ktoken']:.2f}x")
        if p.get("kv_read_GB_per_s") and v.get("kv_read_GB_per_s"):
            print(f"  implied KV read bandwidth:    {p['kv_read_GB_per_s']:.0f} vs "
                  f"{v['kv_read_GB_per_s']:.0f} GB/s")
PY
