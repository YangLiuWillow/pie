#!/usr/bin/env bash
# Grade tonight's three arms, automatically, once generation finishes.
#
# ## The PATH fact that made this look impossible
#
# `colima status` reports `lima not found, run 'brew install lima'`, there is no
# brew, no Docker Desktop, and `/var/run/docker.sock` does not exist. All of that
# is true and all of it is misleading: **colima was already running the whole
# time.** It bundles its own lima at `~/.local/lima/bin/limactl`, which is not on
# PATH, so only the CLI could not reach a live VM. One export fixes it:
#
#     export PATH=$HOME/.local/lima/bin:$PATH
#
# 10 GB VM, docker 29.5.2, 31 swebench images already cached. Worth writing down
# because the error message names the wrong remedy and sends you to a package
# manager that is not installed.
#
# ## Why this waits rather than runs
#
# Generation is GPU-bound and serial; grading is Docker/CPU and would contend
# with it. So this blocks until the generation harness prints `=== DONE ===`,
# then grades every arm that produced predictions.
#
# ## What it reports
#
# `resolved`, from `swebench.harness.run_evaluation`. That is the accuracy
# number the patch-bytes column in the generation summary explicitly cannot
# stand in for: the 2026-08-13 run had 2/5 non-empty patches in BOTH arms while
# grading 4/5 against 1/5.
set -uo pipefail

export PATH=$HOME/.local/lima/bin:$PATH
OUT="${SWE_OUT:-/tmp/swe-overnight}"
PY=/tmp/venv-swebench/bin/python
RUN_TAG="${SWE_RUN_TAG:-overnight}"
WORKERS="${SWE_WORKERS:-4}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/grade.log"; }

# Wait for generation. Bounded, so a generation hang does not leave this pending
# forever with nothing said.
DEADLINE=$(( $(date +%s) + ${SWE_GRADE_WAIT:-36000} ))
while ! grep -q "=== DONE ===" "$OUT/run.log" 2>/dev/null; do
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        log "generation did not finish within the wait window; grading what exists"
        break
    fi
    sleep 120
done
log "=== grading starts ==="

docker version >/dev/null 2>&1 || { log "FATAL: docker unreachable even with lima on PATH"; exit 1; }

for arm in pie mlx vllm; do
    P="$OUT/preds-$arm.jsonl"
    if [ ! -s "$P" ]; then
        # Reported, not skipped silently: a missing arm and a failed arm are
        # indistinguishable in a results table unless one of them says so.
        log "[$arm] NO PREDICTIONS -- not graded"
        continue
    fi
    n=$(wc -l < "$P" | tr -d ' ')
    log "[$arm] grading $n predictions"
    ( cd "$OUT" && $PY -m swebench.harness.run_evaluation \
        --dataset_name SWE-bench/SWE-bench_Verified \
        --predictions_path "$P" --max_workers "$WORKERS" \
        --run_id "${RUN_TAG}_${arm}" ) > "$OUT/grade-$arm.log" 2>&1
    log "[$arm] grader rc=$? (see grade-$arm.log)"
done

log "=== RESOLVED COUNTS ==="
$PY - "$OUT" "$RUN_TAG" <<'PY' 2>&1 | tee -a "$OUT/grade.log"
import json, glob, os, sys
out, run_tag = sys.argv[1], sys.argv[2]
known = set("django__django-12276 django__django-13028 django__django-13089 "
            "django__django-14373 django__django-15569".split())
# The summary report is `<model_name_or_path>.<run_id>.json` in the CWD --
# `swebench.harness.reporting.make_run_report` builds it as
#     Path(predictions[0][KEY_MODEL].replace("/","__") + f".{run_id}" + ".json")
# so for label `pie` and run_id `overnight_pie` it is `pie.overnight_pie.json`.
#
# An earlier version of this filtered on `"report" in filename`, which matches
# NONE of those and DOES match the per-instance
# `logs/run_evaluation/<run>/<model>/<inst>/report.json` files -- a different
# schema with no `resolved_ids`. It would have printed "resolved 0/?" for every
# arm: wrong, and shaped exactly like an answer.
reports = {}
for arm in ("pie", "mlx", "vllm"):
    exact = os.path.join(out, f"{arm}.{run_tag}_{arm}.json")
    cands = [exact] if os.path.exists(exact) else sorted(
        glob.glob(os.path.join(out, f"{arm}.*.json")))
    for f in cands:
        try: d = json.load(open(f))
        except Exception: continue
        # Identify by SCHEMA, not by name: the summary has resolved_ids.
        if "resolved_ids" in d:
            reports[arm] = (f, d)
            break
    if arm not in reports:
        print(f"{arm}: no summary report found (looked for {os.path.basename(exact)})")
if not reports:
    print("no grading reports found; read grade-*.log")
    raise SystemExit
for arm, (f, d) in sorted(reports.items()):
    ids = d.get("resolved_ids") or []
    tot = d.get("total_instances") or d.get("submitted_instances") or "?"
    kn = sum(1 for i in ids if i in known)
    un = sum(1 for i in ids if i not in known)
    print(f"\n{arm}:  resolved {len(ids)}/{tot}   (known-solvable {kn}/5, unseen {un}/5)")
    print(f"   report: {os.path.basename(f)}")
    for i in sorted(ids):
        print(f"     + {i}  {'known' if i in known else 'unseen'}")
print("""
Read the two subsets separately. The known-solvable five are the earlier run's
wins and are biased toward solvable; only the unseen five are an unbiased read,
and five instances move 20 points per instance either way.""")
PY
log "=== GRADING DONE ==="
