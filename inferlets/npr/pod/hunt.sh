#!/usr/bin/env bash
# Deploy on-demand pods until one passes the network + cuInit gates.
# Writes $NPR_STATE/pod-id.txt and pod-ssh.txt ("IP PORT"). Rejected pods are terminated.
set -u
cd "$(dirname "$0")"; source ./rp.sh
PUB=$(cat ~/.ssh/id_ed25519_runpod.pub)
GPUS=("NVIDIA L40S" "NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA GeForce RTX 4090")
SSHO="-n -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
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
    for w in $(seq 1 40); do
      INFO=$(curl -s --max-time 20 https://rest.runpod.io/v1/pods/$ID -H "Authorization: Bearer $RUNPOD_API_KEY")
      IP=$(echo "$INFO" | python3 -c "import json,sys; p=json.load(sys.stdin); print(p.get('publicIp') or '')")
      PORT=$(echo "$INFO" | python3 -c "import json,sys; p=json.load(sys.stdin); print((p.get('portMappings') or {}).get('22') or '')")
      if [ -n "$IP" ] && [ -n "$PORT" ] && ssh $SSHO -p $PORT root@$IP true 2>/dev/null; then
        CTRL=$(ssh $SSHO -p $PORT root@$IP 'curl -s -o /dev/null -w "%{speed_download}" --max-time 15 "https://speed.cloudflare.com/__down?bytes=30000000"' 2>/dev/null | cut -d. -f1)
        HFS=$(ssh $SSHO -p $PORT root@$IP "curl -s -o /dev/null -w '%{speed_download}' -L --max-time 20 -H 'Authorization: Bearer $HF_TOKEN' https://huggingface.co/bigai-NPR/NPR-4B/resolve/main/model-00001-of-00004.safetensors" 2>/dev/null | cut -d. -f1)
        CUI=$(ssh $SSHO -p $PORT root@$IP 'python3 -c "import ctypes; print(ctypes.CDLL(\"libcuda.so.1\").cuInit(0))"' 2>/dev/null | tail -1)
        echo "control: ${CTRL:-0} B/s | hf(auth): ${HFS:-0} B/s | cuInit: ${CUI:-?}"
        if [ "${CTRL:-0}" -ge 5000000 ] && [ "${HFS:-0}" -ge 2000000 ] && [ "${CUI:-1}" = "0" ]; then
          echo "$IP $PORT" > "$NPR_STATE/pod-ssh.txt"; echo "$ID" > "$NPR_STATE/pod-id.txt"
          echo "ACCEPTED $GPU $ID $IP:$PORT"; exit 0
        fi
        echo "rejected; terminating $ID"
        curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$ID -H "Authorization: Bearer $RUNPOD_API_KEY" -o /dev/null
        break
      fi
      sleep 15
    done
    # The wait loop can exhaust without ever getting an IP/port (RunPod hands
    # back a RUNNING pod whose networking never materializes). Falling through
    # to the next GPU would leave that pod billing unattended, so reap any pod
    # we deployed and did not accept.
    if [ ! -s "$NPR_STATE/pod-id.txt" ] || [ "$(cat "$NPR_STATE/pod-id.txt")" != "$ID" ]; then
      echo "unaccepted; terminating $ID"
      curl -s --max-time 30 -X DELETE https://rest.runpod.io/v1/pods/$ID -H "Authorization: Bearer $RUNPOD_API_KEY" -o /dev/null
    fi
  done
  sleep 45
done
echo "HUNT FAILED after 20 rounds"; exit 1
