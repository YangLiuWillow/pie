#!/usr/bin/env bash
# One opencode + qwen3-8b benchmark run, reported the way test-time-bench does.
#
#   ./run.sh pie  [n]     # pie serving the artifact
#   ./run.sh vllm [n]     # vLLM-metal serving the same artifact
#
# The sequence is deliberate, and each step exists because its absence has cost
# this integration a measurement:
#
#   1. VERIFY the grader image pins before spending an hour driving. A run
#      graded against images that drifted is not comparable to the last one,
#      and nothing downstream would say so.
#   2. BOOT through the arm's helper, which proves the live server is the one
#      this script started (not merely that something answers /health).
#   3. CAPTURE provenance into TTB_* — the engine config and its sha256, the KV
#      pool READ from the trace rather than derived, the artifact name.
#   4. DRIVE from the pinned dataset snapshot, so the prompt is a dataset
#      version rather than an editable constant.
#   5. EMIT run-summary.v2, which is the only artifact anyone should quote.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
ARM="${1:?usage: run.sh <pie|vllm> [n_cases]}"
N="${2:-20}"
# 4-bit, not bf16: pie's Metal llama path binds every projection through an
# affine-U4 kernel and REFUSES a checkpoint with no `.scales` tensors. The bf16
# repack was imported and rejected at load. See models/qwen3-8b.toml.
ARTIFACT="${TTB_ARTIFACT:-mlx-community--Qwen3-8B-4bit}"
# The opencode.json model id for this artifact, and which instance set to drive.
#   TTB_INSTANCES=snapshot        -> TTB's swe-bench-lite-first-20 (default)
#   TTB_INSTANCES=known-solvable  -> this repo's KNOWN_SOLVABLE_BOTH, graded
#                                    against Verified. That set is a Coder-30B
#                                    instrument; using it with a weaker model
#                                    makes a zero unattributable, which is the
#                                    opposite of what it is for.
TTB_MODEL_ID="${TTB_MODEL_ID:-qwen3-8b}"
TTB_INSTANCES="${TTB_INSTANCES:-snapshot}"
# Explicit instance ids, e.g. TTB_INSTANCE_IDS="django__django-13089". Overrides
# the first-N selection so both arms can be pointed at the same single case.
TTB_INSTANCE_IDS="${TTB_INSTANCE_IDS:-}"
SNAPSHOT="$HERE/dataset.swe-bench-lite-first-20.json"
CONFIG="${TTB_CONFIG:-$HERE/config.opencode-qwen3-8b.json}"
STAMP="$(date +%H%M%S)"
OUT="${TTB_OUT_DIR:-/tmp/ttb-$ARM-$STAMP}"
mkdir -p "$OUT"
cd "$REPO"

# The driver needs `datasets` in known-solvable mode (it resolves instances from
# HuggingFace); snapshot mode does not, because the snapshot is a local file.
# The system python3 on macOS is 3.9 and has neither `datasets` nor the `str |
# Path` syntax some of these scripts use, so resolve an interpreter that works
# instead of discovering it 40 minutes into a run.
PY_BIN="${TTB_PYTHON:-}"
if [ -z "$PY_BIN" ]; then
    for cand in python3 \
        /private/tmp/claude-501/*/*/scratchpad/venv-shim/bin/python3 \
        "$HOME/.venv-vllm-metal/bin/python3"; do
        for c in $cand; do
            if [ -x "$c" ] && "$c" -c "import datasets" >/dev/null 2>&1; then PY_BIN="$c"; break 2; fi
        done
    done
fi
if [ -z "$PY_BIN" ]; then
    echo "FATAL: no python with \`datasets\` found; set TTB_PYTHON" >&2; exit 1
fi
echo "── driver python: $PY_BIN"

if [ "${TTB_INSTANCES:-snapshot}" = snapshot ] && [ ! -f "$SNAPSHOT" ]; then
    echo "FATAL: no dataset snapshot — run ./fetch_dataset.sh" >&2; exit 1
fi

# 1. Grading must be pinned before driving, not after.
export PATH="$HOME/.local/lima/bin:$HOME/.local/bin:$PATH"
# The pin check needs a live daemon, and step 1b below stops the VM to free
# memory for the model — so a PREVIOUS run leaves it stopped and this one
# cannot verify. Start it if needed; 1b stops it again either way.
if [ "${TTB_SKIP_PIN_CHECK:-0}" != "1" ] && command -v colima >/dev/null 2>&1; then
    if ! docker info >/dev/null 2>&1; then
        echo "── starting the grader VM just to verify pins (stopped again before the drive)"
        colima start --cpu 6 --memory 14 --disk 80 --vm-type vz --vz-rosetta >/dev/null 2>&1 || true
    fi
fi
if [ "${TTB_SKIP_PIN_CHECK:-0}" != "1" ]; then
    "$HERE/pin_images.sh" verify || {
        echo "Re-pin deliberately (pin_images.sh write) or pull the missing images." >&2
        echo "TTB_SKIP_PIN_CHECK=1 drives anyway — then the run is NOT gradeable-comparable." >&2
        exit 1
    }
fi

# 1b. The grader VM and the model server contend for the SAME memory. colima
#     was resident at 8.3 GB (of a 14 GB allocation) when a Coder-30B reboot
#     failed with "needs 19.48 GiB resident (16.00 available)" — the agent then
#     talked to a dead server and the case was recorded as an agent error with
#     an empty patch. Docker is needed to GRADE, not to DRIVE, so it is stopped
#     here and restarted for scoring.
if [ "${TTB_KEEP_DOCKER:-0}" != "1" ] && command -v colima >/dev/null 2>&1; then
    if colima status >/dev/null 2>&1; then
        echo "── stopping the grader VM for the duration of the drive (it holds GBs the model needs)"
        colima stop >/dev/null 2>&1 || true
    fi
fi

# 2. Boot, per arm, through the helper that proves identity.
case "$ARM" in
pie)
    # TTB_PIE_EXTRA passes further env to the boot helper, e.g.
    # "PIE_STRATEGY=b PIE_PYTHON=/path/to/venv/bin/python3" for the session
    # arm. Without it every run silently takes run_pie_opencode.sh's default
    # strategy `a` — which is how a whole afternoon of pie-vs-vLLM numbers came
    # to compare a no-KV-reuse arm against vLLM's prefix cache.
    BOOT="integrations/opencode/tools/boot_pie.sh ttb PIE_MODEL=$ARTIFACT PIE_MAX_MODEL_LEN=32768 PIE_KV_TRACE=1 ${TTB_PIE_EXTRA:-}"
    MODEL="pie/$TTB_MODEL_ID"
    ENDPOINT="http://127.0.0.1:8080"
    ;;
vllm)
    # Same artifact, served from the HF cache id rather than the .zt store.
    BOOT="VLLM_MODEL=$(echo "$ARTIFACT" | sed 's/--/\//') VLLM_SERVED_NAME=$TTB_MODEL_ID VLLM_MAX_MODEL_LEN=32768 integrations/opencode/tools/boot_vllm.sh ttb"
    MODEL="vllm/$TTB_MODEL_ID"
    ENDPOINT="http://127.0.0.1:8000"
    ;;
*) echo "FATAL: arm must be pie or vllm" >&2; exit 2 ;;
esac

echo "── booting $ARM"
eval "$BOOT"

# 3. Provenance. Read, never derived — the KV pool in particular, because three
#    layers own a `total_pages` and the one you would edit is not the one that
#    decides (see results-swebench.md).
export TTB_ENGINE_ENDPOINT="$ENDPOINT"
export TTB_MODEL_ARTIFACT="$ARTIFACT"
export TTB_BACKEND="Apple Metal (M-series, 48 GB unified)"
export OPENCODE_VERSION="$("$HOME/.opencode/bin/opencode" --version 2>/dev/null | head -1)"
if [ "$ARM" = pie ]; then
    # Resolve the mktemp config to a REAL path — a glob left in the string
    # hashes to nothing and the summary silently carries a null provenance
    # field, which is the failure this whole block exists to prevent.
    cfg=$(grep -o "T/pie_opencode\.[A-Za-z0-9]*" /tmp/serve_ttb.out 2>/dev/null | head -1 || true)
    if [ -n "$cfg" ]; then
        real=$(ls -d /var/folders/*/T/"${cfg#T/}" 2>/dev/null | head -1 || true)
        [ -n "$real" ] && export TTB_ENGINE_CONFIG="$real"
    fi
    # `avail= N/M` appears only once a working set installs, so a freshly
    # booted server has NO such line and grep exits 1. Under `set -o pipefail`
    # that killed this script before it ever drove a case. Tolerated here; the
    # meaningful read is the one after the run.
    pool=$(grep -o 'avail= *[0-9]*/[0-9]*' /tmp/pie_ttb.log 2>/dev/null | head -1 | sed 's|.*/||' || true)
    [ -n "$pool" ] && export TTB_KV_POOL_PAGES="$pool"
fi

# 4. Drive from the snapshot. Restart per instance on BOTH arms — asymmetric
#    restarts are how a flattering number got made here once already.
if [ -n "$TTB_INSTANCE_IDS" ]; then
    # shellcheck disable=SC2206
    SELECT=(--instances $TTB_INSTANCE_IDS)
    SUITE="swe-bench-verified-known-solvable"
    GRADE_DATASET="SWE-bench/SWE-bench_Verified"
    echo "── driving explicit instance(s): $TTB_INSTANCE_IDS"
elif [ "$TTB_INSTANCES" = known-solvable ]; then
    SELECT=(--known-solvable --n "$N")
    SUITE="swe-bench-verified-known-solvable"
    GRADE_DATASET="SWE-bench/SWE-bench_Verified"
    echo "── driving $N known-solvable cases (graded against Verified)"
else
    SELECT=(--dataset-snapshot "$SNAPSHOT" --n "$N")
    SUITE="swe-bench-lite-first-20-opencode"
    GRADE_DATASET="SWE-bench/SWE-bench_Lite"
    echo "── driving $N cases from the snapshot"
fi
"$PY_BIN" integrations/opencode/run_swebench.py \
    "${SELECT[@]}" \
    --model "$MODEL" --timeout "$(python3 -c "
import json;print(json.load(open('$CONFIG'))['budgets']['max_wall_time_s']['value'])")" \
    --label "opencode-$TTB_MODEL_ID-$ARM" \
    --out "$OUT/preds.jsonl" \
    --workdir "$OUT/workspaces" \
    --restart-cmd "$BOOT" 2>&1 | tee "$OUT/drive.log"

# Pool again AFTER the run: the trace is per-boot, and a per-instance restart
# means the last line is the last server's, which is the one that mattered last.
if [ "$ARM" = pie ]; then
    pool=$(grep -o 'avail= *[0-9]*/[0-9]*' /tmp/pie_ttb.log 2>/dev/null | tail -1 | sed 's|.*/||' || true)
    [ -n "$pool" ] && export TTB_KV_POOL_PAGES="$pool"
fi

# 5. Persist provenance BEFORE emitting, so a re-emit after grading (a later
#    process, without this environment) still carries it.
"$PY_BIN" - "$OUT/provenance.json" <<'PROV'
import json, os, sys
keys = ("TTB_ENGINE_ENDPOINT", "TTB_ENGINE_CONFIG", "TTB_MODEL_ARTIFACT",
        "TTB_BACKEND", "TTB_KV_POOL_PAGES", "OPENCODE_VERSION")
prov = {k: os.environ.get(k) for k in keys}
# Hash the engine config HERE, while the server that used it is still alive.
# It is an mktemp file: once the last server exits it is gone, so a summary
# re-emitted after grading can never hash it. Recording the path alone loses
# the field entirely — observed on a resolved run.
cfg = prov.get("TTB_ENGINE_CONFIG")
if cfg and os.path.exists(cfg):
    import hashlib
    prov["TTB_ENGINE_CONFIG_SHA256"] = hashlib.sha256(open(cfg, "rb").read()).hexdigest()
json.dump(prov, open(sys.argv[1], "w"), indent=2)
print(f"wrote {sys.argv[1]}")
PROV

# 6. The artifact.
"$PY_BIN" integrations/opencode/ttb_summary.py \
    --cases "$OUT/preds.cases.jsonl" \
    --label "opencode-$TTB_MODEL_ID-$ARM" \
    --suite "$SUITE" \
    --config "$CONFIG" \
    --out "$OUT/run-summary.json"

echo
echo "── artifacts in $OUT"
echo "   preds.jsonl        predictions for the official grader"
echo "   preds.cases.jsonl  per-case record"
echo "   run-summary.json   the artifact to quote (success_rate is null until graded)"
echo
echo "Grade (restart the VM first — it was stopped to free memory for the model):"
echo "  colima start --cpu 6 --memory 14 --disk 80 --vm-type vz --vz-rosetta"
echo "  python -m swebench.harness.run_evaluation \\"
echo "      --dataset_name $GRADE_DATASET \\"
echo "      --predictions_path $OUT/preds.jsonl --max_workers 2 --run_id ttb-$ARM-$STAMP"
