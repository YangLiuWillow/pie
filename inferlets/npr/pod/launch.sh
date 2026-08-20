#!/usr/bin/env bash
# launch.sh ARMS OUT [EXTRA...]  — copies chain.sh to the accepted pod and starts it detached.
set -eu
cd "$(dirname "$0")"; source ./rp.sh
ARMS=${1:-adopt,adopt_nopen}; OUT=${2:-aime25-pen-ab.jsonl}; shift 2 || true; EXTRA=${*:-}
read IP PORT < "$NPR_STATE/pod-ssh.txt"
echo "$OUT" > "$NPR_STATE/out.txt"
SSHO="-i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
scp -q $SSHO -P $PORT chain.sh root@$IP:/workspace/chain.sh
ssh -n $SSHO -p $PORT root@$IP "cd /workspace && ARMS='$ARMS' OUT='$OUT' ${EXTRA:+EXTRA='$EXTRA'} setsid nohup bash chain.sh > chain.log 2>&1 < /dev/null & sleep 2; cat /workspace/CHAIN_STATUS"
