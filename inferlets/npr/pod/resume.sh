#!/usr/bin/env bash
# resume.sh ARMS OUT  — restart an interrupted sweep on a FRESH pod without
# losing the rows already collected.
#
# The premise: results stream to the local worktree every poll interval, so a
# dead pod costs at most one interval of rows — but only if those rows get back
# onto the replacement pod before run_eval.py starts, because `--resume` decides
# what to skip by reading the `--out` file that lives *on the pod*. Without this
# step a restart silently redoes everything, which on a multi-hour sweep is the
# difference between losing 3 minutes and losing the run.
#
#   bash hunt.sh                       # get a fresh pod first
#   bash resume.sh adopt,adopt_nopen aime25-pen-ab.jsonl
#
# Errored rows are deliberately not treated as done by run_eval.py, so they are
# retried rather than silently dropped.
set -eu
cd "$(dirname "$0")"; source ./rp.sh
ARMS=${1:-adopt,adopt_nopen}; OUT=${2:-aime25-pen-ab.jsonl}
STATE_ROOT=${STATE_ROOT:-/workspace}
LOCAL=$(cd ../evals && pwd)/results/$OUT
read IP PORT < "$NPR_STATE/pod-ssh.txt"
SSHO="-i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

if [ ! -s "$LOCAL" ]; then
  echo "no local rows at $LOCAL — nothing to resume; use launch.sh for a fresh sweep"; exit 1
fi
HAVE=$(wc -l < "$LOCAL" | tr -d ' ')
echo "seeding $HAVE local rows onto the new pod"
ssh -n $SSHO -p "$PORT" root@"$IP" "mkdir -p $STATE_ROOT/results"
scp -q $SSHO -P "$PORT" "$LOCAL" root@"$IP":"$STATE_ROOT/results/$OUT"
# Verify the seed landed intact before spending pod time on it: a truncated
# upload would look like a partial sweep and silently re-run the missing rows.
REMOTE=$(ssh -n $SSHO -p "$PORT" root@"$IP" "wc -l < $STATE_ROOT/results/$OUT" 2>/dev/null | tr -d ' ')
[ "$REMOTE" = "$HAVE" ] || { echo "seed mismatch: local $HAVE vs remote $REMOTE — aborting"; exit 1; }
echo "seed verified: $REMOTE rows on pod"
exec bash ./launch.sh "$ARMS" "$OUT"
