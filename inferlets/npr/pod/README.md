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
* `gate.sh POD_ID [WAIT_MIN]` owns the whole accept/reject decision, and
  `hunt.sh` now just deploys and hands each pod to it. It also remembers the
  last non-empty ip/port instead of resetting when the REST view flaps. Use it
  directly to adopt a pod after an interrupted hunt, or to re-gate one by hand.
  It terminates the pod on any rejection, so a rejected pod never becomes a leak.

### Fail fast on a broken container

`gate.sh` probes RunPod's SSH proxy (`<podHostId>@ssh.runpod.io`, from GraphQL
`machine { podHostId }`) before waiting on a public IP. The proxy answers as
soon as the pod is rented — often long before `publicIp`/`portMappings` appear,
and it still answers when they never do. It needs a PTY (`-tt`; without one it
replies `Error: Your SSH client doesn't support PTY`) and supports no scp, so
it is a diagnostic channel only: the pipeline still moves files over direct IP.

What it buys is separating *still booting* from *never going to start*. Two
consecutive pods on 2026-08-20 sat at `desiredStatus: RUNNING` with
`runtime.ports: null` and answered the proxy with

```
OCI runtime exec failed: exec failed: unable to start container process:
error writing config to pipe: write init-p: broken pipe: unknown
```

— the container process could not start at all. Without the probe each of those
costs a full `WAIT_MIN` of billing before the gate gives up; with it they are
rejected in about a minute. Bad hosts are the norm, not the exception here
(HANDOVER §11: 3 of 4 community L40S hosts had broken CUDA), so the cost of
finding a good pod is dominated by how fast you can discard a bad one.

A pattern the probe does *not* match is not harmless — it falls through to the
full `WAIT_MIN`. The matched set is `OCI runtime exec`, `is not running`,
`Error response from daemon`, and a non-zero `cuInit`; extend it rather than
letting a new failure string time out silently.

### The image is part of the gate

Three consecutive pods on `nvidia/cuda:12.8.1-devel-ubuntu22.04` came up with a
container that never started. That image has `nvcc` but no sshd, and RunPod's
`startSsh`/`PUBLIC_KEY` injection has to build the ssh environment itself. The
`runpod/pytorch` images ship sshd *and* `/usr/local/cuda`, which is what
HANDOVER §5 means by "a runpod/pytorch `*-devel` image" — and the account's own
saved template for this project is `Pie-test-image`
(`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`). That is now `hunt.sh`'s
default, overridable with `IMAGE=`.

`gate.sh` also checks for `nvcc` before accepting, because `driver/cuda` is
built from source on the pod: a missing compiler is otherwise a ~25-minute
bootstrap that ends in `SETUP_FAIL`.

### Resuming after a pod death (`resume.sh`)

`run_eval.py --resume` (the default) decides what to skip by reading the
`--out` file **on the pod**. The poller streams that file to the local worktree
every 3 minutes, so a dead pod costs at most one interval of rows — but only if
those rows get back onto the replacement pod before the sweep restarts.
Otherwise the restart silently redoes everything, which on a multi-hour sweep is
the difference between losing 3 minutes and losing the run.

```bash
bash hunt.sh                                    # fresh pod
bash resume.sh adopt,adopt_nopen aime25-pen-ab.jsonl
```

It seeds the local rows onto the new pod, verifies the line count matches before
spending any pod time (a truncated upload otherwise looks like a partial sweep),
and hands off to `launch.sh`. Verified behaviour, tested against a real partial
file with an errored row appended: 15 lines in, 14 counted as done, 86 of 100
runs queued — **errored rows are deliberately not counted as done**, so they are
retried rather than silently dropped.

What a restart actually costs, on top of the ≤1 poll interval of lost rows: the
fresh pod redoes bootstrap — engine build, 15 GB checkout download, bf16 cast —
which is ~25-35 min, plus the hunt. It does *not* redo the sshd/MooseFS triage,
because the image default and the BUILD/STATE split above are now in the
scripts.

Whatever the path in, always confirm the account is actually empty when you are
done — `curl -s https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY"`
— rather than assuming a DELETE call worked.
