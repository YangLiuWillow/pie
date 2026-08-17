#!/usr/bin/env bash
# Three-engine SWE-bench generation, SYMMETRIC, unattended.
#
# ## What this fixes about the 2026-08-13 run
#
# That run scored pie 4/5 against vLLM-metal 1/5 and said plainly in its own
# write-up that the arms were not symmetric: pie got `--restart-cmd` and a fresh
# server per instance, vLLM ran one server throughout. Giving one arm a
# mitigation the other lacks is how a flattering number gets made. **Every arm
# here restarts per instance**, including mlx-lm, which that comparison never
# graded at all.
#
# ## What it can and cannot produce tonight
#
# It produces PREDICTIONS for three arms. It cannot grade them: the official
# `swebench.harness.run_evaluation` needs Docker, and on this machine the
# colima VM will not start (`lima not found`), with no brew to install it. That
# is fine as a split of work -- generation is GPU-bound and must run serially
# here, grading is Docker/CPU and can run any time later against the saved
# `preds-*.jsonl`. Nothing about tonight's output has to be redone.
#
# A NON-EMPTY PATCH IS NOT A RESOLVED INSTANCE. This script reports patch bytes
# because that is what it can honestly report, and the summary says so in those
# words. The earlier write-up makes the same distinction and it is the one that
# gets forgotten first.
#
# ## Budget
#
# Instances vary enormously -- pie took 309-1265 s each in the graded run, vLLM
# 10-146 s (an agent giving up, not an agent hurrying). So this is TIME-BOUNDED,
# not count-bounded: it starts a new instance only if the deadline is far enough
# away, and writes partial results as it goes. A run that is cut off mid-way
# still leaves every completed instance usable.
set -uo pipefail

REPO="${PIE_REPO:-/Users/liuyang/Documents/Liszt_ai/pie-opencode}"
OUT="${SWE_OUT:-/tmp/swe-overnight}"
PIEPY=/Users/liuyang/.venvs/pie/bin/python
HARNESS_PY=/Users/liuyang/.venv-vllm-metal/bin/python
MLXMODEL=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
# Wall-clock budget in seconds. Default ~8h leaves margin before morning.
BUDGET="${SWE_BUDGET:-28800}"
PER_INSTANCE_TIMEOUT="${SWE_TIMEOUT:-1800}"

if [ ! -x "$REPO/target/release/pie" ]; then
    echo "FATAL: no $REPO/target/release/pie" >&2; exit 1
fi
mkdir -p "$OUT"
cd "$REPO/integrations/opencode" || { echo "FATAL: no integrations/opencode" >&2; exit 1; }

START=$(date +%s)
left() { echo $(( BUDGET - ( $(date +%s) - START ) )); }
log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

stop_all() {
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    pkill -f "vllm serve" 2>/dev/null
    pkill -f "VLLM::EngineCore" 2>/dev/null
    pkill -f mlx_lm.server 2>/dev/null
    sleep 8
}
trap 'stop_all; log "stopped"' EXIT

# The instance set. Deliberately the SAME five the graded run used, plus five
# more, and the reason is stated rather than hidden: the original five are "the
# baseline's wins", so they are biased toward solvable and NOT toward any
# engine. Keeping them makes tonight comparable to the graded result; adding
# five unseen ones is what starts to break that bias. Both subsets are reported
# separately in the summary for exactly this reason.
KNOWN="django__django-12276 django__django-13028 django__django-13089 django__django-14373 django__django-15569"
# Verified present in SWE-bench_Verified before committing hours to them: an
# invented id (django__django-11039, which is NOT in the dataset) was in this
# list until the check caught it.
UNSEEN="django__django-10914 django__django-11099 django__django-11133 django__django-12308 django__django-13158"
INSTANCES="$KNOWN $UNSEEN"

# Each arm's restart command, run by the runner BEFORE every instance. This is
# the symmetry the earlier comparison lacked.
PIE_RESTART="pkill -f '$REPO/target/release/pie .*serve'; pkill -f session_shim.py; sleep 8; \
PIE_PYTHON=$PIEPY $REPO/integrations/opencode/tools/boot_pie.sh sweN \
  PIE_STRATEGY=b PIE_MODEL=qwen3-coder-30b PIE_MAX_MODEL_LEN=65536 \
  PIE_MAX_FORWARD_TOKENS=4096 PIE_PYTHON=$PIEPY >/dev/null 2>&1"
VLLM_RESTART="pkill -f 'vllm serve'; pkill -f 'VLLM::EngineCore'; sleep 8; \
VLLM_MAX_MODEL_LEN=65536 $REPO/integrations/opencode/tools/boot_vllm.sh sweN >/dev/null 2>&1"
MLX_RESTART="pkill -f mlx_lm.server; sleep 8; \
nohup /tmp/venv-mlxlm/bin/mlx_lm.server --model $MLXMODEL --port 8001 --host 127.0.0.1 \
  >/tmp/mlx_swe.log 2>&1 & \
for i in \$(seq 1 60); do curl -s -m 3 -o /dev/null http://127.0.0.1:8001/v1/models && break; sleep 3; done"

run_arm() {  # $1 tag, $2 opencode model string, $3 restart cmd
    local tag=$1 model=$2 restart=$3
    local remaining; remaining=$(left)
    # Do not START an arm that cannot finish even one instance.
    if [ "$remaining" -lt $(( PER_INSTANCE_TIMEOUT + 600 )) ]; then
        log "SKIP $tag: only ${remaining}s left, less than one instance needs"
        return 0
    fi
    log "===== $tag starting (${remaining}s budget left) ====="
    stop_all
    local t0; t0=$(date +%s)
    $HARNESS_PY run_swebench.py \
        --instances $INSTANCES --model "$model" --label "$tag" \
        --out "$OUT/preds-$tag.jsonl" --timeout "$PER_INSTANCE_TIMEOUT" \
        --restart-cmd "$restart" \
        > "$OUT/$tag.log" 2>&1
    local rc=$? n=0
    [ -f "$OUT/preds-$tag.jsonl" ] && n=$(wc -l < "$OUT/preds-$tag.jsonl" | tr -d ' ')
    log "$tag finished rc=$rc wall=$(( $(date +%s) - t0 ))s predictions=$n"
    # An arm that produced nothing is reported as such, not silently omitted --
    # a missing arm and a failing arm look identical in a results table.
    [ "$n" -gt 0 ] || log "WARNING: $tag produced NO predictions; see $OUT/$tag.log"
}

log "=== overnight SWE-bench generation ==="
log "instances: $(echo $INSTANCES | wc -w | tr -d ' ')  budget: ${BUDGET}s  per-instance timeout: ${PER_INSTANCE_TIMEOUT}s"
log "grading is NOT possible on this machine tonight (colima: lima not found)"

run_arm pie  "pie/qwen3-coder-30b"  "$PIE_RESTART"
run_arm mlx  "mlx/$MLXMODEL"        "$MLX_RESTART"
run_arm vllm "vllm/qwen3-coder-30b" "$VLLM_RESTART"

log "=== summary (PATCH BYTES, NOT RESOLUTION) ==="
$HARNESS_PY - "$OUT" <<'PY' 2>&1 | tee -a "$OUT/run.log"
import json, sys, glob, os
out = sys.argv[1]
known = set("django__django-12276 django__django-13028 django__django-13089 "
            "django__django-14373 django__django-15569".split())
arms = {}
for f in sorted(glob.glob(f"{out}/preds-*.jsonl")):
    # `preds-<arm>.cases.jsonl` is the runner's per-case METADATA (case_id,
    # seconds, workspace) and is matched by this glob too. Its rows have no
    # `instance_id`, so `d.get("instance_id")` returned None and sorting
    # None against str crashed the summary after all three arms had run.
    if ".cases." in f:
        continue
    tag = os.path.basename(f)[len("preds-"):-len(".jsonl")]
    rows = {}
    for line in open(f):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        rows[d.get("instance_id")] = len(d.get("model_patch") or "")
    arms[tag] = rows
if not arms:
    print("no predictions at all"); raise SystemExit
ids = sorted(set().union(*[set(v) for v in arms.values()]))
names = sorted(arms)
print(f"\n{'instance':<28} " + " ".join(f"{n:>10}" for n in names) + "   set")
for i in ids:
    cells = " ".join(f"{arms[n].get(i, -1):>10}" for n in names)
    print(f"{i:<28} {cells}   {'known' if i in known else 'unseen'}")
print(f"\n{'non-empty patches':<28} " + " ".join(
    f"{sum(1 for v in arms[n].values() if v > 0):>10}" for n in names))
print(f"{'  of which known-solvable':<28} " + " ".join(
    f"{sum(1 for k,v in arms[n].items() if v > 0 and k in known):>10}" for n in names))
print(f"{'  of which unseen':<28} " + " ".join(
    f"{sum(1 for k,v in arms[n].items() if v > 0 and k not in known):>10}" for n in names))
print(f"{'instances attempted':<28} " + " ".join(f"{len(arms[n]):>10}" for n in names))
print("""
-1 means the arm has no row for that instance (not attempted, or crashed).

A NON-EMPTY PATCH IS NOT A RESOLVED INSTANCE. These counts are a weaker claim
than accuracy: a patch that applies and fails its tests counts here and does not
count in a graded score. The 2026-08-13 run had 2/5 non-empty in both arms while
grading 4/5 against 1/5, so this column does NOT rank engines. Grade with
`tools/grade.sh` once Docker works.""")
PY
log "=== DONE ==="
ls -la "$OUT"/preds-*.jsonl 2>/dev/null | tee -a "$OUT/run.log"
