#!/bin/bash
# Stateless local poller. Streams results + logs every 3 min; terminates the pod ONLY on a healthy DONE.
# Safe to kill and relaunch at any time. No traps.
cd "$(dirname "$0")"; source ./rp.sh
LR=$(cd ../evals && pwd)/results; mkdir -p "$LR"
read IP PORT < "$NPR_STATE/pod-ssh.txt"; POD=$(cat "$NPR_STATE/pod-id.txt"); OUT=$(cat "$NPR_STATE/out.txt")
EXPECT=${EXPECT_ROWS:-0}   # optional: row count that also counts as success
# Must match chain.sh's STATE root: status, logs and results live on the volume,
# while the build tree lives on the pod's local container disk.
STATE_ROOT=${STATE_ROOT:-/workspace}
SSHO="-n -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
SCPO="-q -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
LASTST=""; LASTN=0; FAILS=0; LASTSELF=""
for i in $(seq 1 250); do
  ST=$(ssh $SSHO -p $PORT root@$IP "cat $STATE_ROOT/CHAIN_STATUS 2>/dev/null" 2>/dev/null | head -1)
  if [ -z "$ST" ]; then FAILS=$((FAILS+1)); [ $FAILS -ge 5 ] && { echo "UNREACHABLE: ssh failed 5x"; FAILS=0; }; sleep 180; continue; fi
  FAILS=0
  scp $SCPO -P $PORT root@$IP:$STATE_ROOT/results/$OUT "$LR/" 2>/dev/null
  scp $SCPO -P $PORT "root@$IP:$STATE_ROOT/sweep.log" "root@$IP:$STATE_ROOT/selftest.log" "root@$IP:$STATE_ROOT/BUILD_SHA" "$NPR_STATE/" 2>/dev/null
  # Surface the CUDA sentinel line the moment it lands — it gates the whole sweep.
  ST_LINE=$(grep -h "selftest toplogits" "$NPR_STATE/selftest.log" 2>/dev/null | head -1)
  if [ -n "$ST_LINE" ] && [ "$ST_LINE" != "$LASTSELF" ]; then echo "[$(date '+%H:%M')] $ST_LINE"; LASTSELF="$ST_LINE"; fi
  # The results file does not exist until the sweep's first row lands; treat a
  # missing file as zero rather than letting the redirect error into the log.
  if [ -f "$LR/$OUT" ]; then N=$(wc -l < "$LR/$OUT" 2>/dev/null | tr -d ' '); else N=0; fi
  N=${N:-0}
  case "$ST" in
    DONE*)
      if [ "$ST" != "DONE exit=0" ] && { [ "$EXPECT" -eq 0 ] || [ "$N" -lt "$EXPECT" ]; }; then
        echo "CHAIN ENDED BADLY ($ST, rows=$N) — pod $POD left up for diagnosis"; exit 1; fi
      scp $SCPO -P $PORT "root@$IP:$STATE_ROOT/serve.log" "$NPR_STATE/" 2>/dev/null
      curl -s --max-time 60 -X DELETE https://rest.runpod.io/v1/pods/$POD -H "Authorization: Bearer $RUNPOD_API_KEY" -o /dev/null
      echo "SWEEP COMPLETE: $ST, rows=$N, pod $POD terminated"; exit 0;;
    SETUP_FAIL) echo "SETUP FAILED — pod $POD left up for inspection"; exit 1;;
    SELFTEST_FAIL)
      scp $SCPO -P $PORT "root@$IP:$STATE_ROOT/serve.log" "$NPR_STATE/" 2>/dev/null
      echo "CUDA SELFTEST FAILED — no sweep was run; pod $POD left up for diagnosis"
      grep -h "selftest" "$NPR_STATE/selftest.log" 2>/dev/null | tail -5
      exit 1;;
  esac
  if [ "$ST" != "$LASTST" ] || [ $((N - LASTN)) -ge 20 ]; then echo "[$(date '+%H:%M')] status=$ST rows=$N"; LASTST="$ST"; LASTN=$N; fi
  sleep 180
done
echo "poller cycle cap reached; pod $POD may still be up"
