# Kill-proof RunPod sweep pipeline

Reconstructed from the session that produced HANDOVER.md §11 (the originals
lived in a harness scratchpad that a reboot wiped — hence durable copies here).
Everything that matters runs *on the pod*, detached; the local side is a
stateless poller that only pulls files and terminates the pod on a healthy
finish. Local session death, harness task-killers, or laptop reboots cannot
lose results.

Prereqs (local): `~/Documents/Liszt_ai/.env` with `RUNPOD_API_KEY=` and
`HF_TOKEN=`; ssh key `~/.ssh/id_ed25519_runpod`; python3; macOS (no `setsid`,
hence the Python detach idiom below).

```bash
cd inferlets/npr/pod
export NPR_STATE=$HOME/.npr-pod            # durable state dir (pod id, ssh addr, logs)
bash hunt.sh                               # foreground: finds a healthy pod, writes $NPR_STATE/pod-{id,ssh}.txt
bash gate.sh POD_ID [WAIT_MIN]             # or: gate one pod you already have (see below)
bash launch.sh "adopt,adopt_nopen" aime25-pen-ab.jsonl   # scp chain.sh + start it detached on the pod
python3 -c 'import subprocess,os; subprocess.Popen(["bash","poll.sh"],stdout=open(os.environ["NPR_STATE"]+"/poll.log","a"),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)'
tail -F $NPR_STATE/poll.log
```

Rules baked in (each one was learned the hard way, see HANDOVER.md §10/§11):

* `volumeInGb: 40` + `volumeMountPath: /workspace` — container disk is wiped on stop.
* Host gate: `cuInit(0)` must return `0` (nvidia-smi looks fine on broken hosts).
* Network gate: ≥5 MB/s control, ≥2 MB/s authenticated HF download.
* **No `trap ... EXIT` that terminates the pod** — the harness kills background
  tasks; a trap once deleted a pod mid-setup.
* `poll.sh` terminates the pod only on `DONE exit=0` (or the expected row
  count); any other end state leaves the pod up for diagnosis — remember to
  terminate it yourself (`bash rp.sh; curl -X DELETE https://rest.runpod.io/v1/pods/$ID ...`).
* Never touch pods you did not create (other sessions' pods, e.g. `ss-pie-patissier`).

## `gate.sh` — when deploy and gate need to come apart

`hunt.sh` fuses "deploy a pod" with "wait for it and gate it", and those two
fail independently. Observed 2026-08-20: RunPod returned a `RUNNING` L40S whose
`publicIp` appeared, then reverted to `""` on the next REST poll, and whose sshd
never accepted a connection. `hunt.sh` waited out its loop, moved on to the next
GPU type — and left the first pod billing at $0.99/hr with nobody watching it.
Two fixes:

* `hunt.sh` now terminates any pod it deployed and did not accept.
* `gate.sh POD_ID [WAIT_MIN]` runs the gates against a pod you already have, and
  remembers the last non-empty ip/port instead of resetting when the REST view
  flaps. Use it to adopt a pod after an interrupted hunt, or to re-gate one
  by hand. It terminates the pod on any rejection, so a rejected pod never
  becomes a leak.

Whatever the path in, always confirm the account is actually empty when you are
done — `curl -s https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY"`
— rather than assuming a DELETE call worked.
