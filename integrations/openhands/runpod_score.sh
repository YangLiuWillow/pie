#!/usr/bin/env bash
# runpod_score.sh — CPU-pod companion to runpod_submit.sh. Same create → ssh →
# run → TERMINATE-on-any-exit lifecycle, but deploys a cheap CPU pod so SWE-bench
# scoring/eval (CPU-bound, container-heavy) never burns GPU-hours. Mirrors the
# Slurm split where agents run on gpu_rtx6000 and scoring runs on the `day` CPU
# partition.
#
# Cost model is identical to runpod_submit.sh: billed per second the pod exists,
# so the trap terminates on success, failure, or Ctrl-C. Results are written to
# the persistent network volume, so termination never loses them.
#
# ── Required env ────────────────────────────────────────────────────────────
#   RUNPOD_API_KEY     RunPod API key.
#   RUNPOD_VOLUME_ID   Network-volume id (same one the agent run wrote its
#                      predictions to; mounted at $MOUNT).
#   RUNPOD_IMAGE       Image to run. NOTE: the scorer needs its container runtime
#                      (apptainer OR docker) available — Dockerfile.cuda-native
#                      ships NEITHER. Use an image that has your eval harness's
#                      runtime, or run the swebench docker harness on a
#                      docker-enabled pod. (This launcher is runtime-agnostic; it
#                      just runs whatever command you pass after `--`.)
#   SSH_KEY            Private key whose .pub is injected (default ~/.ssh/id_ed25519).
#
# ── The job ─────────────────────────────────────────────────────────────────
#   Everything after `--` is the remote command, run from $WORKDIR. e.g.:
#     runpod_score.sh -- .venv/bin/python integrations/openhands/score_swebench_apptainer.py \
#         integrations/openhands/predictions/ab_cuda_native_13_<jobid>.jsonl \
#         --report-file integrations/openhands/predictions/ab_cuda_native_13_<jobid>.report.json \
#         --sandbox-root /workspace/swebench_sandboxes --timeout 3600
#
# ── Knobs (env) ─────────────────────────────────────────────────────────────
#   POD_KIND=cpu|gpu   cpu (default) = CPU pod; gpu = cheap-GPU fallback if CPU
#                      deploy is unavailable (uses the proven GPU mutation).
#   CPU_INSTANCE_ID    CPU tier for POD_KIND=cpu (default cpu3c-4-8 = 4 vCPU/8GB).
#                      ⚠️ CONFIRM available ids: `runpodctl` / RunPod console —
#                      the CPU deploy field is the one bit to sanity-check here.
#   GPU_TYPE           GPU for POD_KIND=gpu fallback (default a small/cheap card).
#   CLOUD=SECURE       Cloud for the deploy.  CONTAINER_DISK_GB=80 (eval sandboxes).
#   MOUNT=/workspace   WORKDIR=$MOUNT/pie   NAME=pie-score   POLL_S=15   DETACH=1
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
: "${RUNPOD_VOLUME_ID:?set RUNPOD_VOLUME_ID}"
: "${RUNPOD_IMAGE:?set RUNPOD_IMAGE}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
POD_KIND="${POD_KIND:-cpu}"
CPU_INSTANCE_ID="${CPU_INSTANCE_ID:-cpu3c-4-8}"
GPU_TYPE="${GPU_TYPE:-NVIDIA RTX A4000}"     # cheap fallback card
CLOUD="${CLOUD:-SECURE}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-80}" # eval builds per-instance sandboxes
MOUNT="${MOUNT:-/workspace}"
WORKDIR="${WORKDIR:-$MOUNT/pie}"
NAME="${NAME:-pie-score}"
POLL_S="${POLL_S:-15}"
DETACH="${DETACH:-1}"                         # scoring is long → detach by default
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# Remote command = everything after `--`.
JOB_ARGS=()
while [ $# -gt 0 ]; do case "$1" in --) shift; JOB_ARGS=("$@"); break;; *) shift;; esac; done
[ ${#JOB_ARGS[@]} -gt 0 ] || { echo "ERROR: no job. Usage: $0 -- <remote scoring command>"; exit 2; }
JOB_CMD="${JOB_ARGS[*]}"

command -v jq   >/dev/null || { echo "ERROR: need jq";   exit 2; }
command -v curl >/dev/null || { echo "ERROR: need curl"; exit 2; }
[ -f "$SSH_KEY" ]     || { echo "ERROR: no private key at $SSH_KEY"; exit 2; }
[ -f "$SSH_KEY.pub" ] || { echo "ERROR: no public key at $SSH_KEY.pub"; exit 2; }
PUBKEY="$(cat "$SSH_KEY.pub")"

POD_ID=""
cleanup() {
  local rc=$?
  if [ -n "$POD_ID" ]; then
    echo ">> terminating pod $POD_ID"
    gql "mutation { podTerminate(input: {podId: \"$POD_ID\"}) }" >/dev/null 2>&1 || \
      echo "!! terminate call failed — CHECK https://runpod.io/console/pods for a leaked pod ($POD_ID)"
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

gql() {
  local q; q=$(jq -Rn --arg q "$1" '{query:$q}')
  local resp; resp=$(curl -fsS -H 'Content-Type: application/json' -d "$q" "$API")
  if echo "$resp" | jq -e '.errors' >/dev/null 2>&1; then
    echo "GraphQL error: $(echo "$resp" | jq -c '.errors')" >&2; return 1
  fi
  echo "$resp" | jq '.data'
}

# ── 1. Create the pod (CPU by default; cheap-GPU fallback) ──────────────────
START_CMD="mkdir -p /root/.ssh && echo \\\"\$PUBLIC_KEY\\\" >> /root/.ssh/authorized_keys && (service ssh start || /usr/sbin/sshd) && sleep infinity"
COMMON_INPUT="
    cloudType: $CLOUD
    name: \"$NAME\"
    imageName: \"$RUNPOD_IMAGE\"
    containerDiskInGb: $CONTAINER_DISK_GB
    networkVolumeId: \"$RUNPOD_VOLUME_ID\"
    volumeMountPath: \"$MOUNT\"
    ports: \"22/tcp\"
    dockerArgs: \"bash -c '$START_CMD'\"
    env: [{ key: \"PUBLIC_KEY\", value: \"$PUBKEY\" }]
"
if [ "$POD_KIND" = "cpu" ]; then
  echo ">> deploying CPU pod ($CPU_INSTANCE_ID, $CLOUD)"
  # ⚠️ CPU deploy: instanceId selects a CPU tier; gpuCount 0. If RunPod rejects
  #    this shape, run `runpodctl` to list current CPU instance ids, or fall back
  #    to POD_KIND=gpu (proven mutation below).
  MUT="mutation { podFindAndDeployOnDemand(input: { gpuCount: 0 instanceId: \"$CPU_INSTANCE_ID\" $COMMON_INPUT }) { id } }"
  POD_ID=$(gql "$MUT" | jq -r '.podFindAndDeployOnDemand.id')
else
  echo ">> deploying GPU pod fallback ($GPU_TYPE, $CLOUD)"
  MUT="mutation { podFindAndDeployOnDemand(input: { gpuCount: 1 gpuTypeId: \"$GPU_TYPE\" $COMMON_INPUT }) { id } }"
  POD_ID=$(gql "$MUT" | jq -r '.podFindAndDeployOnDemand.id')
fi
[ -n "$POD_ID" ] && [ "$POD_ID" != "null" ] || { echo "ERROR: pod create returned no id (CPU tier unavailable? try POD_KIND=gpu or another CPU_INSTANCE_ID)"; POD_ID=""; exit 1; }
echo ">> pod $POD_ID"

# ── 2. Wait for RUNNING + a public SSH port ─────────────────────────────────
echo ">> waiting for RUNNING + ssh ..."
SSH_IP=""; SSH_PORT=""
for _ in $(seq 1 120); do
  DATA=$(gql "query { pod(input: {podId: \"$POD_ID\"}) { desiredStatus runtime { ports { ip publicPort privatePort isIpPublic } } } }" || true)
  read -r SSH_IP SSH_PORT < <(echo "$DATA" | jq -r '
    (.pod.runtime.ports // []) | map(select(.privatePort==22 and .isIpPublic)) | .[0] // {} | "\(.ip // "") \(.publicPort // "")"')
  [ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ] && break
  sleep "$POLL_S"
done
[ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ] || { echo "ERROR: pod never exposed ssh"; exit 1; }
SSH=(ssh -p "$SSH_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectTimeout=15 "root@$SSH_IP")
for _ in $(seq 1 20); do "${SSH[@]}" true 2>/dev/null && break; sleep 5; done
echo ">> ssh up: root@$SSH_IP:$SSH_PORT"

# ── 3. Run the scoring job & wait ───────────────────────────────────────────
echo ">> job: $JOB_CMD"
if [ "$DETACH" = "1" ]; then
  LOG="$MOUNT/.runpod_score.log"; RC="$MOUNT/.runpod_score.rc"
  "${SSH[@]}" "cd '$WORKDIR' && rm -f '$RC' && nohup bash -lc '$JOB_CMD; echo \$? > $RC' > '$LOG' 2>&1 &"
  echo ">> detached; polling $RC (tailing $LOG)"
  LINES=0
  while true; do
    NEW=$("${SSH[@]}" "tail -n +$((LINES+1)) '$LOG' 2>/dev/null" || true)
    [ -n "$NEW" ] && { echo "$NEW"; LINES=$(( LINES + $(printf '%s\n' "$NEW" | wc -l) )); }
    DONE=$("${SSH[@]}" "cat '$RC' 2>/dev/null" || true)
    [ -n "$DONE" ] && { JOB_RC="$DONE"; break; }
    sleep "$POLL_S"
  done
else
  set +e
  "${SSH[@]}" "cd '$WORKDIR' && $JOB_CMD"
  JOB_RC=$?
  set -e
fi
echo ">> job exit code: $JOB_RC"

# Reports/scores persist on the network volume; cleanup() terminates the pod now.
exit "$JOB_RC"
