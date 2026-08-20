#!/usr/bin/env bash
# Deploy on-demand pods until one passes gate.sh (container health, network,
# cuInit). Writes $NPR_STATE/pod-id.txt and pod-ssh.txt ("IP PORT").
# gate.sh terminates whatever it rejects, so a pod we deploy and do not accept
# is never left billing.
set -u
cd "$(dirname "$0")"; source ./rp.sh
PUB=$(cat ~/.ssh/id_ed25519_runpod.pub)
GPUS=("NVIDIA L40S" "NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA GeForce RTX 4090")
WAIT_MIN=${WAIT_MIN:-10}
deploy_json() {  # $1 = gpuTypeId
python3 - "$1" "$PUB" <<'PY'
import json, sys
q = "mutation Deploy($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id machineId costPerHr desiredStatus } }"
inp = {"cloudType": "SECURE", "gpuTypeId": sys.argv[1], "gpuCount": 1,
       "containerDiskInGb": 80, "volumeInGb": 40, "volumeMountPath": "/workspace",
       "minVcpuCount": 8, "minMemoryInGb": 30,
       "name": "npr-sweep-" + sys.argv[1].split()[-1].lower(),
       "imageName": "nvidia/cuda:12.8.1-devel-ubuntu22.04", "ports": "22/tcp",
       "env": [{"key": "PUBLIC_KEY", "value": sys.argv[2]}],
       "supportPublicIp": True, "startSsh": True}
print(json.dumps({"query": q, "variables": {"input": inp}}))
PY
}
for round in $(seq 1 20); do
  for GPU in "${GPUS[@]}"; do
    R=$(deploy_json "$GPU" | curl -s --max-time 90 https://api.runpod.io/graphql -H 'Content-Type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" -d @-)
    ID=$(echo "$R" | python3 -c "import json,sys; d=json.load(sys.stdin); p=(d.get('data') or {}).get('podFindAndDeployOnDemand'); print(p['id'] if p else '')" 2>/dev/null)
    [ -z "$ID" ] && continue
    echo "landed $GPU pod $ID"
    # gate.sh owns the whole accept/reject decision, including terminating what
    # it rejects. Fusing deploy and gate is what once left a pod billing while
    # the hunt moved on to the next GPU type.
    if bash ./gate.sh "$ID" "$WAIT_MIN"; then
      echo "ACCEPTED $GPU $ID"; exit 0
    fi
  done
  sleep 45
done
echo "HUNT FAILED after 20 rounds"; exit 1
