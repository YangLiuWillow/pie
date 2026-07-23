#!/usr/bin/env bash
# runpod_submit.sh — submit a one-shot GPU job to RunPod, wait for it, tear the
# pod down. The Slurm `sbatch` + blocking `squeue` analog for RunPod.
#
# Cost model: RunPod bills per second the pod exists, so the whole point is to
# TERMINATE the moment the job ends. This script terminates on ANY exit (success,
# failure, Ctrl-C) via a trap — a leaked pod is a running meter. Results are
# written to the persistent network volume, so termination never loses them.
#
# Cheapest sane config (defaults below): COMMUNITY cloud + spot (interruptible)
# + A100 80GB + a pre-staged network volume. Your benchmark harness already has
# --resume + auto-restart, so spot preemption is safe.
#
# ── Required env ────────────────────────────────────────────────────────────
#   RUNPOD_API_KEY     RunPod API key.
#   RUNPOD_VOLUME_ID   Network-volume id (holds the repo + HF model cache +
#                      predictions/logs; mounted at $MOUNT). Datacenter-locked.
#   RUNPOD_IMAGE       Docker image with the driver-cuda pie build + sshd.
#                      Build + push from repo-root Dockerfile.cuda-native:
#                        docker build -f Dockerfile.cuda-native \
#                          --build-arg CUDA_ARCH=80 -t <you>/pie-cuda-native:a100 .
#                        docker push <you>/pie-cuda-native:a100
#                      (CUDA_ARCH must match the GPU_TYPE below: 80=A100.)
#   SSH_KEY            Path to the private key whose PUBLIC half is injected into
#                      the pod (default ~/.ssh/id_ed25519).
#
# ── The job ─────────────────────────────────────────────────────────────────
#   Everything after `--` is the remote command, run from $WORKDIR on the pod.
#   e.g.  runpod_submit.sh -- bash integrations/openhands/run_pie_backend.sh \
#             --instance-id django__django-13028 --python-tool-parser
#
# ── Useful knobs (env) ──────────────────────────────────────────────────────
#   GPU_TYPE="NVIDIA A100 80GB PCIe"   CLOUD=COMMUNITY   SPOT=1   BID=0.8
#   CONTAINER_DISK_GB=40   MOUNT=/workspace   WORKDIR=/workspace/pie
#   DETACH=1   # run the job detached on the pod + poll a sentinel (survives SSH
#              # drops — use for multi-hour sweeps). Default 0 = stream in foreground.
#   POLL_S=15  NAME=pie-job
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
: "${RUNPOD_VOLUME_ID:?set RUNPOD_VOLUME_ID}"
: "${RUNPOD_IMAGE:?set RUNPOD_IMAGE}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
GPU_TYPE="${GPU_TYPE:-NVIDIA A100 80GB PCIe}"
CLOUD="${CLOUD:-COMMUNITY}"
SPOT="${SPOT:-1}"
BID="${BID:-0.8}"                 # $/GPU-hr ceiling for spot
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-40}"
MOUNT="${MOUNT:-/workspace}"
WORKDIR="${WORKDIR:-$MOUNT/pie}"
NAME="${NAME:-pie-job}"
POLL_S="${POLL_S:-15}"
DETACH="${DETACH:-0}"
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# Remote command = everything after `--`.
JOB_ARGS=()
while [ $# -gt 0 ]; do case "$1" in --) shift; JOB_ARGS=("$@"); break;; *) shift;; esac; done
[ ${#JOB_ARGS[@]} -gt 0 ] || { echo "ERROR: no job. Usage: $0 -- <remote command>"; exit 2; }
JOB_CMD="${JOB_ARGS[*]}"

command -v jq   >/dev/null || { echo "ERROR: need jq";   exit 2; }
command -v curl >/dev/null || { echo "ERROR: need curl"; exit 2; }
[ -f "$SSH_KEY" ]     || { echo "ERROR: no private key at $SSH_KEY"; exit 2; }
[ -f "$SSH_KEY.pub" ] || { echo "ERROR: no public key at $SSH_KEY.pub"; exit 2; }
PUBKEY="$(cat "$SSH_KEY.pub")"

POD_ID=""
# ── Always tear the pod down, whatever happens ──────────────────────────────
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

# ── GraphQL helper: gql '<query>' → data JSON (aborts on errors) ────────────
gql() {
  local q; q=$(jq -Rn --arg q "$1" '{query:$q}')
  local resp; resp=$(curl -fsS -H 'Content-Type: application/json' -d "$q" "$API")
  if echo "$resp" | jq -e '.errors' >/dev/null 2>&1; then
    echo "GraphQL error: $(echo "$resp" | jq -c '.errors')" >&2; return 1
  fi
  echo "$resp" | jq '.data'
}

# ── 1. Create the pod ───────────────────────────────────────────────────────
# A custom image must run sshd and honor PUBLIC_KEY. dockerArgs below starts one
# even on a bare image (needs openssh-server installed); drop it if the image's
# own start script already does this (RunPod base templates do).
START_CMD="mkdir -p /root/.ssh && echo \\\"\$PUBLIC_KEY\\\" >> /root/.ssh/authorized_keys && (service ssh start || /usr/sbin/sshd) && sleep infinity"
COMMON_INPUT="
    cloudType: $CLOUD
    gpuCount: 1
    gpuTypeId: \"$GPU_TYPE\"
    name: \"$NAME\"
    imageName: \"$RUNPOD_IMAGE\"
    containerDiskInGb: $CONTAINER_DISK_GB
    networkVolumeId: \"$RUNPOD_VOLUME_ID\"
    volumeMountPath: \"$MOUNT\"
    ports: \"22/tcp\"
    dockerArgs: \"bash -c '$START_CMD'\"
    env: [{ key: \"PUBLIC_KEY\", value: \"$PUBKEY\" }]
"
if [ "$SPOT" = "1" ]; then
  echo ">> renting spot pod ($GPU_TYPE, bid \$$BID/GPU-hr, $CLOUD)"
  MUT="mutation { podRentInterruptable(input: { bidPerGpu: $BID $COMMON_INPUT }) { id } }"
  POD_ID=$(gql "$MUT" | jq -r '.podRentInterruptable.id')
else
  echo ">> deploying on-demand pod ($GPU_TYPE, $CLOUD)"
  MUT="mutation { podFindAndDeployOnDemand(input: { $COMMON_INPUT }) { id } }"
  POD_ID=$(gql "$MUT" | jq -r '.podFindAndDeployOnDemand.id')
fi
[ -n "$POD_ID" ] && [ "$POD_ID" != "null" ] || { echo "ERROR: pod create returned no id (no capacity for $GPU_TYPE? raise BID / change GPU_TYPE / CLOUD)"; POD_ID=""; exit 1; }
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
# sshd inside the container may need a few seconds past RUNNING.
for _ in $(seq 1 20); do "${SSH[@]}" true 2>/dev/null && break; sleep 5; done
echo ">> ssh up: root@$SSH_IP:$SSH_PORT"

# ── 3. Run the job & wait ───────────────────────────────────────────────────
echo ">> job: $JOB_CMD"
if [ "$DETACH" = "1" ]; then
  # Detached: job keeps running if our SSH drops; we poll a sentinel on the
  # persistent volume and tail the log. Re-run with the same POD to re-attach.
  LOG="$MOUNT/.runpod_job.log"; RC="$MOUNT/.runpod_job.rc"
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
  # Foreground: stream stdout live; SSH exit code == job exit code.
  set +e
  "${SSH[@]}" "cd '$WORKDIR' && $JOB_CMD"
  JOB_RC=$?
  set -e
fi
echo ">> job exit code: $JOB_RC"

# ── 4. (optional) pull results back before the pod dies ─────────────────────
# Predictions/logs already persist on the network volume, so this is only for a
# local copy. Set FETCH_LOCAL to a dir to enable.
if [ -n "${FETCH_LOCAL:-}" ]; then
  echo ">> rsync results → $FETCH_LOCAL"
  mkdir -p "$FETCH_LOCAL"
  rsync -az -e "ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    "root@$SSH_IP:$WORKDIR/integrations/openhands/predictions/" "$FETCH_LOCAL/" || echo "!! rsync failed (non-fatal)"
fi

# cleanup() (trap) terminates the pod now, on this normal exit.
exit "$JOB_RC"
