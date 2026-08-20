#!/bin/bash
# Runs entirely ON THE POD, detached. Local session death cannot affect it.
# Args baked in by launch.sh via env: ARMS, OUT (results filename), EXTRA (run_eval flags).
ARMS=${ARMS:-adopt,adopt_nopen}; OUT=${OUT:-aime25-pen-ab.jsonl}
EXTRA=${EXTRA:---k 2 --limit 25 --concurrency 8 --prompt-cache --abort-after 10}
BRANCH=${BRANCH:-npr-inferlet}; REPO_RAW=${REPO_RAW:-https://raw.githubusercontent.com/YangLiuWillow/pie}
cd /workspace
echo SETUP > /workspace/CHAIN_STATUS
curl -fsSL "$REPO_RAW/$BRANCH/inferlets/npr/pod-setup.sh" -o pod-setup.sh
BRANCH=$BRANCH bash pod-setup.sh > setup.log 2>&1
if ! grep -q "^== done ==" setup.log; then echo SETUP_FAIL > /workspace/CHAIN_STATUS; exit 1; fi
# Record exactly which code produced the numbers (DESIGN.md house style).
(cd /workspace/pie && git rev-parse HEAD) > /workspace/BUILD_SHA 2>/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> /workspace/BUILD_SHA 2>/dev/null
echo SERVE > /workspace/CHAIN_STATUS
cd /workspace/pie && PIE_CONFIG=/workspace/npr-cuda.toml nohup ./target/release/pie serve > /workspace/serve.log 2>&1 &
sleep 45
# CUDA sentinel: selftest prints "[npr] selftest toplogits: ids_match=true max_dev=... max_logit=...".
# HARD GATE — a penalty path that is not CUDA-correct makes every A/B number meaningless,
# so we refuse to spend the sweep on it (HANDOVER.md §12 check 1).
echo SELFTEST > /workspace/CHAIN_STATUS
/workspace/venv/bin/python /workspace/pie/inferlets/npr/client.py --input '{"selftest": true}' > /workspace/selftest.log 2>&1
if ! grep -q "selftest toplogits: ids_match=true" /workspace/selftest.log; then
  echo SELFTEST_FAIL > /workspace/CHAIN_STATUS; exit 1
fi
# max_logit ~= 0-1 means the sentinel path returned probabilities, not raw logits.
if /workspace/venv/bin/python - <<'PY'
import re,sys
m=re.search(r"max_logit=([-\d.]+)", open("/workspace/selftest.log").read())
sys.exit(0 if (m and abs(float(m.group(1))) < 2.0) else 1)
PY
then echo SELFTEST_FAIL > /workspace/CHAIN_STATUS; exit 1; fi
echo SWEEP > /workspace/CHAIN_STATUS
cd /workspace/pie/inferlets/npr/evals && mkdir -p results
/workspace/venv/bin/python run_eval.py --arms "$ARMS" $EXTRA --out "results/$OUT" > /workspace/sweep.log 2>&1
echo "DONE exit=$?" > /workspace/CHAIN_STATUS
