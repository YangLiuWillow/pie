#!/bin/bash
# Runs entirely ON THE POD, detached. Local session death cannot affect it.
# Args baked in by launch.sh via env: ARMS, OUT (results filename), EXTRA (run_eval flags).
ARMS=${ARMS:-adopt,adopt_nopen}; OUT=${OUT:-aime25-pen-ab.jsonl}
EXTRA=${EXTRA:---k 2 --limit 25 --concurrency 8 --prompt-cache --abort-after 10}
BRANCH=${BRANCH:-npr-inferlet}; REPO_RAW=${REPO_RAW:-https://raw.githubusercontent.com/YangLiuWillow/pie}

# Two roots, because they want opposite things.
#
# STATE is what must survive: status, logs, and the results JSONL, streamed off
# by the local poller (HANDOVER §10 lost a whole sweep to a container-disk
# wipe). It lives on the mounted volume.
#
# BUILD is where the toolchain works. RunPod's /workspace volume can be a
# *network* filesystem — observed 2026-08-20 as `mfs#us-nc-1.runpod.net:9421`
# (MooseFS) — and linking there dies with `ld terminated with signal 7
# [Bus error]` because ld mmaps its output. So the source tree and target dir
# go on the container disk, which is local, large and fast; nothing there needs
# to outlive the pod.
export STATE=${STATE:-/workspace}
BUILD=${BUILD:-/build}
mkdir -p "$STATE" "$BUILD" "$STATE/results"
status() { echo "$1" > "$STATE/CHAIN_STATUS"; }

cd "$BUILD"
status SETUP
curl -fsSL "$REPO_RAW/$BRANCH/inferlets/npr/pod-setup.sh" -o pod-setup.sh
BRANCH=$BRANCH WORK=$BUILD bash pod-setup.sh > "$STATE/setup.log" 2>&1
if ! grep -q "^== done ==" "$STATE/setup.log"; then status SETUP_FAIL; exit 1; fi
# Record exactly which code produced the numbers (DESIGN.md house style).
(cd "$BUILD/pie" && git rev-parse HEAD) > "$STATE/BUILD_SHA" 2>/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> "$STATE/BUILD_SHA" 2>/dev/null

status SERVE
cd "$BUILD/pie" && PIE_CONFIG="$BUILD/npr-cuda.toml" nohup ./target/release/pie serve > "$STATE/serve.log" 2>&1 &
sleep 45
# CUDA sentinel: selftest prints "[npr] selftest toplogits: ids_match=true max_dev=... max_logit=...".
# HARD GATE — a penalty path that is not CUDA-correct makes every A/B number meaningless,
# so we refuse to spend the sweep on it (HANDOVER.md §12 check 1).
#
# Retry rather than trusting a fixed sleep: model load time is not a constant
# (checkpoint size, page-cache state, host disk), and now that the gate is fatal
# a server that was merely slow to come up would throw the pod away. Only a
# selftest that actually ran and disagreed should stop the sweep.
status SELFTEST
for attempt in 1 2 3 4 5 6; do
  "$BUILD/venv/bin/python" "$BUILD/pie/inferlets/npr/client.py" --input '{"selftest": true}' > "$STATE/selftest.log" 2>&1
  grep -q "selftest toplogits:" "$STATE/selftest.log" && break
  echo "[chain] selftest attempt $attempt: no sentinel line yet; server may still be loading" >> "$STATE/selftest-attempts.log"
  sleep 60
done
if ! grep -q "selftest toplogits: ids_match=true" "$STATE/selftest.log"; then
  status SELFTEST_FAIL; exit 1
fi
# max_logit ~= 0-1 means the sentinel path returned probabilities, not raw logits.
if "$BUILD/venv/bin/python" - <<'PY'
import re,sys,os
m=re.search(r"max_logit=([-\d.]+)", open(os.environ["STATE"]+"/selftest.log").read())
sys.exit(0 if (m and abs(float(m.group(1))) < 2.0) else 1)
PY
then status SELFTEST_FAIL; exit 1; fi

status SWEEP
# --out on the volume: the results file is the one artifact that must survive
# the pod, and the poller reads it from there.
cd "$BUILD/pie/inferlets/npr/evals"
"$BUILD/venv/bin/python" run_eval.py --arms "$ARMS" $EXTRA --out "$STATE/results/$OUT" > "$STATE/sweep.log" 2>&1
status "DONE exit=$?"
