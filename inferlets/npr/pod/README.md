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
