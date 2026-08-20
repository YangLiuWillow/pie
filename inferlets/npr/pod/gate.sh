#!/usr/bin/env bash
# gate.sh POD_ID [WAIT_MINUTES]
#
# Wait for one already-deployed pod to become reachable, then run the same
# three gates hunt.sh applies (control bandwidth, authenticated HF bandwidth,
# cuInit). On accept, write $NPR_STATE/pod-{id,ssh}.txt so launch.sh/poll.sh
# can pick it up. On reject or timeout, terminate the pod.
#
# Split out of hunt.sh because deploy and gate fail independently: RunPod
# reported a pod RUNNING with a publicIp that later reverted to empty and whose
# sshd never accepted a connection, and hunt.sh's fused loop responded by
# deploying *another* pod while the first kept billing. Gating a pod you
# already have is also what you want after a hunt is interrupted.
set -u
cd "$(dirname "$0")"; source ./rp.sh
ID=${1:?usage: gate.sh POD_ID [WAIT_MINUTES]}
WAIT_MIN=${2:-12}
SSHO="-n -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"

reject() {  # $1 = reason
  echo "REJECTED $ID: $1 — terminating"
  curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$ID -H "Authorization: Bearer $RUNPOD_API_KEY" -o /dev/null
  exit 1
}

# --- Fast container-health probe via RunPod's SSH proxy -----------------------
# The proxy (ssh.runpod.io) is reachable as soon as the pod is rented, long
# before publicIp/portMappings appear -- and it answers even when they never do.
# It needs a PTY (`-tt`), and it is only a diagnostic channel: no scp, so the
# real pipeline still runs over direct IP. What it buys is telling
# "still booting" apart from "this container will never start", which
# otherwise costs a full WAIT_MIN of billing per broken host. Observed
# 2026-08-20 on two consecutive pods: `OCI runtime exec failed ... write
# init-p: broken pipe` -- the container process could not start at all.
PROXY=$(curl -s --max-time 30 https://api.runpod.io/graphql \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d "{\"query\":\"query { pod(input: {podId: \\\"$ID\\\"}) { machine { podHostId } } }\"}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); p=((d.get('data') or {}).get('pod') or {}).get('machine') or {}; print(p.get('podHostId') or '')" 2>/dev/null)
if [ -n "$PROXY" ]; then
  for t in $(seq 1 8); do
    OUT=$(ssh -tt -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 "$PROXY@ssh.runpod.io" \
          'python3 -c "import ctypes; print(\"cuInit\", ctypes.CDLL(\"libcuda.so.1\").cuInit(0))"' </dev/null 2>&1)
    # Match every way a dead container has announced itself so far. A pattern
    # we do not match is not harmless: it falls through to the full WAIT_MIN
    # of billing, which is exactly what "container is not running" cost once.
    case "$OUT" in
      *"cuInit 0"*)                    echo "proxy: container healthy, cuInit=0"; break;;
      *"OCI runtime exec"*)            reject "container will not start (OCI runtime exec failed)";;
      *"is not running"*)              reject "container is not running";;
      *"Error response from daemon"*)  reject "docker daemon refused exec: $(echo "$OUT" | tail -2 | tr '\n' ' ')";;
      *"cuInit "*)                     reject "cuInit != 0 via proxy (dead CUDA device)";;
    esac
    sleep 20
  done
fi

DEADLINE=$(( $(date +%s) + WAIT_MIN * 60 ))
IP=""; PORT=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  INFO=$(curl -s --max-time 20 https://rest.runpod.io/v1/pods/$ID -H "Authorization: Bearer $RUNPOD_API_KEY")
  I=$(echo "$INFO" | python3 -c "import json,sys; p=json.load(sys.stdin); print(p.get('publicIp') or '')" 2>/dev/null)
  P=$(echo "$INFO" | python3 -c "import json,sys; p=json.load(sys.stdin); print((p.get('portMappings') or {}).get('22') or '')" 2>/dev/null)
  # The REST view flaps: a pod can report an ip/port and then report empty
  # again while sshd is still coming up. Remember the last non-empty pair and
  # keep probing it rather than resetting to "no address known".
  [ -n "$I" ] && IP="$I"; [ -n "$P" ] && PORT="$P"
  if [ -n "$IP" ] && [ -n "$PORT" ] && ssh $SSHO -p "$PORT" root@"$IP" true 2>/dev/null; then
    echo "reachable at $IP:$PORT"
    CTRL=$(ssh $SSHO -p $PORT root@$IP 'curl -s -o /dev/null -w "%{speed_download}" --max-time 15 "https://speed.cloudflare.com/__down?bytes=30000000"' 2>/dev/null | cut -d. -f1)
    HFS=$(ssh $SSHO -p $PORT root@$IP "curl -s -o /dev/null -w '%{speed_download}' -L --max-time 20 -H 'Authorization: Bearer $HF_TOKEN' https://huggingface.co/bigai-NPR/NPR-4B/resolve/main/model-00001-of-00004.safetensors" 2>/dev/null | cut -d. -f1)
    CUI=$(ssh $SSHO -p $PORT root@$IP 'python3 -c "import ctypes; print(ctypes.CDLL(\"libcuda.so.1\").cuInit(0))"' 2>/dev/null | tail -1)
    # driver/cuda is built from source on the pod, so a missing nvcc is a
    # ~25-minute setup that ends in SETUP_FAIL. Cheaper to find out now.
    NVCC=$(ssh $SSHO -p $PORT root@$IP 'command -v nvcc || ls /usr/local/cuda/bin/nvcc 2>/dev/null' 2>/dev/null | tail -1)
    echo "control: ${CTRL:-0} B/s | hf(auth): ${HFS:-0} B/s | cuInit: ${CUI:-?} | nvcc: ${NVCC:-MISSING}"
    # cuInit is the one gate that is not about speed: nvidia-smi looks healthy
    # on hosts where the driver cannot actually create a context, and such a
    # pod burns the whole setup before failing.
    [ "${CUI:-1}" = "0" ]            || reject "cuInit=${CUI:-?} (dead CUDA device)"
    [ -n "${NVCC:-}" ]               || reject "no nvcc — driver/cuda cannot be built on this image"
    [ "${CTRL:-0}" -ge 5000000 ]     || reject "control bandwidth ${CTRL:-0} B/s"
    [ "${HFS:-0}" -ge 2000000 ]      || reject "HF bandwidth ${HFS:-0} B/s"
    echo "$IP $PORT" > "$NPR_STATE/pod-ssh.txt"; echo "$ID" > "$NPR_STATE/pod-id.txt"
    echo "ACCEPTED $ID $IP:$PORT"; exit 0
  fi
  sleep 15
done
reject "never became reachable within ${WAIT_MIN}m"
