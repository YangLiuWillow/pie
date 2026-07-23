# RunPod network-volume staging

One-time prep of the RunPod **network volume** so `runpod_submit.sh` jobs find
everything they need already on disk. Do this once per volume; job pods then
mount it read/write and start instantly.

The `pie` binary (with the `cuda_native` driver) lives in the **image**
(`Dockerfile.cuda-native`), *not* the volume. Everything runtime-variable lives
on the **volume**:

```
$MOUNT/                                  # volume root (default /workspace)
  pie/                                   # $WORKDIR — the repo working tree (rsync'd)
    integrations/openhands/.venv/        #   harness venv (BUILT on the pod)
    integrations/openhands/tests/fixtures/pie_cuda_native_config_30b_moe.toml
    inferlets/openhands-coder-session/target/wasm32-wasip2/release/openhands_coder_session.wasm  # prebuilt, rsync'd
    client/python/                       #   pie client (editable-installed into the venv)
  hf/                                    # $HF_HOME — model cache (DOWNLOADED on the pod)
    hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/...
```

**Volume size:** model ~57 GB + venv ~0.8 GB + repo (no `target/`) ~a few GB →
provision **≥ 80 GB**. (KV cache is GPU memory, not the volume.)

## Why these transports (not a git clone)

- **Repo → rsync from the HPC login node.** `run_pie_backend.sh` currently has
  uncommitted working-tree edits, and the inferlet `.wasm` is never committed
  (`target/` is gitignored). A `git clone` would silently ship a stale harness
  and **no wasm**. rsync ships the exact validated tree. (The runtime image has
  no Rust, so the wasm cannot be rebuilt on the pod — it must arrive prebuilt.)
- **Model → download on the pod.** `Qwen/Qwen3-Coder-30B-A3B-Instruct` is
  public; pulling 57 GB over the pod's cloud uplink beats pushing it from the
  HPC uplink.
- **venv → rebuild on the pod.** The HPC venv's `bin/python` symlinks an
  EasyBuild path (`/apps/software/.../Python/3.12.3/...`) that does not exist on
  the pod, so it cannot be relocated.

## Steps

### 0. Prereqs (on the HPC login node)
- The prebuilt inferlet wasm exists (rebuild if you touched the inferlet):
  ```bash
  cd inferlets/openhands-coder-session && cargo build --release --target wasm32-wasip2
  ```
- `RUNPOD_API_KEY`, the volume's `RUNPOD_VOLUME_ID`, and the pushed image
  (`RUNPOD_IMAGE`, from `Dockerfile.cuda-native`) are ready. Your SSH pubkey is
  the one `runpod_submit.sh` uses (`SSH_KEY`, default `~/.ssh/id_ed25519`).

### 1. Launch a **persistent** staging pod with the volume mounted
Staging needs the pod to stay up while you rsync into it, so do **not** use
`runpod_submit.sh` here (it auto-terminates on exit). Create a persistent pod
via the RunPod console or `runpodctl`, mounting `RUNPOD_VOLUME_ID` at `$MOUNT`
(`/workspace`) with `RUNPOD_IMAGE`, exposing TCP 22, injecting your `PUBLIC_KEY`.
A cheap GPU or CPU pod is fine — staging is I/O-bound, not compute. Note its
public SSH `IP:PORT`.

> The pod create/ssh-wait GraphQL is identical to `runpod_submit.sh` §1–§2 if
> you'd rather script it; the only difference is you skip the terminate trap.

### 2. rsync the repo working tree → volume (from the HPC login node)
```bash
POD_IP=...; POD_PORT=...            # from step 1
SSH="ssh -p $POD_PORT -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# Repo minus heavy/derived dirs. --include rules keep ONLY the coder-session
# wasm out of the otherwise-excluded target/ trees.
rsync -az --info=progress2 -e "$SSH" \
  --include='inferlets/openhands-coder-session/target/***' \
  --exclude='**/target/' \
  --exclude='.git/' \
  --exclude='integrations/openhands/.venv/' \
  --exclude='**/__pycache__/' \
  --exclude='**/node_modules/' \
  --exclude='integrations/openhands/logs/' \
  --exclude='integrations/openhands/predictions/' \
  --exclude='.cpm-cache/' \
  /nfs/roberts/project/pi_ql324/ly337/pie/ \
  root@$POD_IP:/workspace/pie/
```
(If rsync isn't in the image, `apt-get install -y rsync` on the pod first, or
use `tar | ssh`.)

### 3. Finish on the pod: model + venv + verify
```bash
$SSH root@$POD_IP 'bash /workspace/pie/integrations/openhands/stage_on_pod.sh'
```
`stage_on_pod.sh` is idempotent: it downloads the model into `$HF_HOME`
(resumable, skips existing), rebuilds `.venv`, and verifies the full surface
(`pie driver list` shows `cuda_native`; `openhands`/`pie_client`/`litellm`
import; model snapshot present; wasm present). Re-run freely if it's interrupted.

### 4. Tear the staging pod down
Terminate it from the console/`runpodctl` (the volume persists). Leaving it up
is a running meter.

## After staging — run a job
From the HPC login node, `runpod_submit.sh` spins a job pod, runs, and
auto-terminates:
```bash
PIE_BIN=/usr/local/bin/pie \
HF_HOME=/workspace/hf \
CFG=tests/fixtures/pie_cuda_native_config_30b_moe.toml \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
BACKEND=pie-session KV_VERIFY=1 \
  ./runpod_submit.sh -- bash integrations/openhands/run_pie_backend.sh \
    --instance-id django__django-13028 --python-tool-parser \
    --temperature 0 --max-iterations 100
```
First job = a clean-environment shakedown of the exact django-13028 run already
green on Slurm (job 19210565), so any failure is unambiguously RunPod-side.
