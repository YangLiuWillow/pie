# Agent handover — running the fair-parity A/B on the provisioned A100 pod

You are taking over an experiment that is **already provisioned and part-way
through**. Read this file, `TEST_PLAN.md` (the experimental design), and
`README.md` (file index) before touching anything.

Your job: run the remaining arms, collect the numbers, and report them. The
human wants to stop relaying terminal output — so act, verify, and summarise;
don't ask for permission on anything listed as routine below.

---

## 1. What this experiment is (one paragraph)

An earlier writeup (`../docs/pie-vs-litellm-writeup.md`) claimed a Pie-backed
OpenHands agent beat the OpenHands→vLLM baseline by ~26% wall time, but that
baseline was **crippled** — `--enforce-eager` (CUDA graphs off) plus an untuned
MoE kernel. This rerun gives vLLM its proper config and measures whether the gap
survives. vLLM runs at three effort tiers (`crippled`, `graphs-only`, `fair`);
Pie is a fourth line. **We are trying to break our own result** — a defensible
finding beats a flattering one. See `TEST_PLAN.md` §1–§3 for hypotheses and §7
for the decision rules you should apply when reporting.

---

## 2. Machine state — what is already true

Verified on 2026-07-27. Do not redo any of it.

| Thing | Where | Notes |
|---|---|---|
| GPU | — | NVIDIA A100-SXM4-80GB, compute cap 8.0 (sm_80) |
| repo | `/workspace/pie` | branch `openhands-integration-updated`; may be a few commits behind — `git fetch --depth 1 origin openhands-integration-updated && git checkout -B openhands-integration-updated FETCH_HEAD` is safe **between** arms, never during one |
| pie binary | `/workspace/pie/target/release/pie` | 140 MB, `cuda_native` compiled in, built for sm_80 |
| build tree | `/root/pie-target` | `target/` is a **symlink** to it |
| wasm inferlet | `inferlets/openhands-coder-session/target/wasm32-wasip2/release/openhands_coder_session.wasm` | built |
| harness venv | `/root/venvs/harness` | python 3.12; `.venv` in `integrations/openhands/` is a **symlink** to it |
| vLLM venv | `/root/venvs/pie-vllm` | python 3.12, vLLM **0.25.1**; a verified-complete copy of the `/workspace` original |
| model | `/workspace/hf-cache` | `Qwen/Qwen3-Coder-30B-A3B-Instruct`, ~60 GB, downloaded |
| env file | `/workspace/pie-bench-env.sh` | **source it in every shell** |
| caches | `/root/.uv-cache`, `/root/.pip-cache`, `/root/.cpm-cache` | must stay on local disk — see §5 |

Every shell starts:

```bash
source /workspace/pie-bench-env.sh
cd /workspace/pie/integrations/openhands/runpod
```

---

## 3. The storage rule — violate this and things hang, not fail

`/workspace` is a **MooseFS network volume** (`df -h /workspace` shows
`mfs#...runpod.net:9421`). Every file operation is a network round-trip.
Small-file-heavy work does not run slow there — it **stalls indefinitely**,
with no error, which is very easy to misdiagnose as a network or disk problem.
It is neither: `curl` to PyPI's CDN from this pod runs at 125 MB/s and the
volume has 148 TB free.

- **Local disk (`/root`)**: caches, venvs, build trees. Anything with many
  small files.
- **`/workspace`**: the repo, the model weights, predictions, logs. Large
  sequential files.

If you ever run `uv pip install`, `export UV_CACHE_DIR=/root/.uv-cache` first.
uv streams downloads into its cache *before* linking them into the venv, so a
cache on MooseFS stalls the install regardless of where the venv lives. This
cost the human several hours; don't repeat it.

---

## 4. Run order

The Pie arm was started at ~10:07 on 2026-07-27 and may still be running or
already done. Check `../predictions/` before starting anything.

```bash
# 1. Pie arm — native CUDA driver, KV_VERIFY=1, no vLLM involved
ARM=pie bash 30_ab_run.sh

# 2+3. vLLM baseline at the two tiers that need no autotune
VLLM_TIER=crippled    ARM=litellm bash 30_ab_run.sh
VLLM_TIER=graphs-only ARM=litellm bash 30_ab_run.sh

# 4. the long one — kernel-config sweep, hours
bash 11_autotune_moe.sh

# 5. the tier the autotune unlocks
VLLM_TIER=fair        ARM=litellm bash 30_ab_run.sh

# 6. the effort-axis table
python summarize_ab.py \
  pie=../predictions/ab_a100_pie_*.jsonl \
  litellm-crippled=../predictions/ab_a100_litellm_crippled_*.jsonl \
  litellm-graphs-only=../predictions/ab_a100_litellm_graphs-only_*.jsonl \
  litellm-fair=../predictions/ab_a100_litellm_fair_*.jsonl
```

This order deliberately differs from `TEST_PLAN.md` §6, which puts the autotune
first. The autotune runs for hours and gates only the `fair` tier, so three of
the four lines land before you spend that time. In particular
`crippled → graphs-only` — the gap quantifying what `--enforce-eager` cost the
original writeup — is available early.

**Run one arm at a time and verify it before starting the next.** A queued
block of arms is how a single missing dependency burned four runs at once.

`11_autotune_moe.sh` will probably fail to find `benchmark_moe.py`: it ships in
vLLM's source `benchmarks/kernels/`, which is not in the wheel. Fetch it
version-matched:

```bash
curl -sL https://raw.githubusercontent.com/vllm-project/vllm/v0.25.1/benchmarks/kernels/benchmark_moe.py \
  -o /workspace/benchmark_moe.py
$PIE_VENV/bin/python /workspace/benchmark_moe.py --model "$MODEL" --tune --dtype auto
```

The `v0.25.1` tag must match the installed engine — the autotuner writes a
config keyed to kernel internals.

---

## 5. How to tell a healthy arm from a dead one

**Within 2 minutes of starting**, confirm the first instance is really
iterating. The failure mode that matters produces a *complete-looking* run of
13 rows, each `0 iters, 0-byte patch, error=...`, in about 30 seconds.

```bash
tail -f /workspace/pie/integrations/openhands/logs/*.log \
  | grep -v 'Cost calculation failed\|warnings.warn'
```

- `Cost calculation failed: This model isn't mapped yet` — **benign**, once per
  LLM call. litellm has no price entry for a self-hosted model. It touches
  nothing the benchmark measures. Its presence is actually evidence the agent
  is calling the model.
- `[pie-driver-cuda] req_id=... R=1 N=1 sampled=1 max_kv=NNNNN` with `max_kv`
  climbing — healthy decode on the Pie arm.
- Weight load is ~61 GB from MooseFS; several minutes of silence at boot is
  normal. `nvidia-smi` memory climbing is the progress bar.
- `ModuleNotFoundError` — a dependency is undeclared. Install it into
  `/root/venvs/harness` with `--exclude-newer 2026-05-15` (see §6), delete the
  ruined predictions file, restart the arm.

Between arms, make sure nothing still holds the GPU — 60 GB of weights per
engine will not co-reside on 80 GB:

```bash
pgrep -af 'pie serve|vllm' ; nvidia-smi --query-gpu=memory.used --format=csv
```

---

## 6. Version pins — do not "upgrade" these

Each exists because the unpinned version broke:

| Pin | Why |
|---|---|
| **Python 3.12** (both venvs) | On 3.13 the vLLM set *resolves and installs*, then torch dies at import in the TorchScript overload parser (`torch/_sources.py parse_def` → `IndentationError`). A successful resolve is not evidence the stack runs. |
| **vLLM 0.25.1** | Unpinned takes whatever PyPI serves that day. 0.26.0 exists (released 2026-07-25) — do **not** switch mid-experiment; a tier comparison needs one engine version. |
| **harness deps at `--exclude-newer 2026-05-15`** | `openhands-sdk 1.21.1` (2026-05-08) declares 14 open-ended floors. At today's newest, litellm 1.93.0 fails to import its own `MessagesInterceptor` and fastmcp 3.x moved `Client`. Pin time, not packages. |
| **`openhands-tools==1.21.1`** | `benchmarks/swe_bench.py:370` imports `openhands.tools.preset.default`. Ships in lockstep with the sdk — keep the versions identical. |

Any install into the harness venv:

```bash
export UV_CACHE_DIR=/root/.uv-cache
VIRTUAL_ENV=/root/venvs/harness uv pip install --exclude-newer 2026-05-15 <pkg>
```

---

## 7. What to record — this is the deliverable

Per arm (`TEST_PLAN.md` §6):

1. **The `assert_vllm_fair.sh` banner block** for each vLLM tier. This is the
   auditable proof the tier is what it claims, symmetric to how the current
   writeup quotes the crippled banner. Paste it verbatim.
2. **The `summarize_ab.py` row** — wall, iterations, **s/iter**, median
   per-call latency, tokens.
3. **Pie arm only**: prefill-reuse % and the kv-verify error count (**must be
   0**).
4. **The version set** from §6 above, once, alongside the banners.

Report **s/iter and median per-call latency, never raw wall time** — the two
agents walk different trajectories, so raw wall conflates path length with
serving speed (`TEST_PLAN.md` §5, §8).

### Two things to state accurately, not favourably

- **`KV_VERIFY=1` is a token-count check, not numerical KV equivalence.** It
  reaches exactly one line — `inferlets/openhands-coder-session/src/lib.rs:739`,
  `ctx.seq_len() as usize != full_tokens.len()`. That catches the failure mode
  broken prefix reuse produces and is worth reporting, but do not describe it as
  proving reused KV blocks are numerically identical to a fresh render. (Cost:
  one integer comparison per call, so it does not perturb the timings.)
- **The 13-instance set in `30_ab_run.sh` is the baseline's own prior wins**, so
  its accuracy ceiling is parity. Clean for timing; **not** an accuracy signal.
  `../predictions/neutral_50_pie.jsonl` exists if an accuracy run is wanted.

The expected outcome, stated up front in `TEST_PLAN.md` §3: on A100 the decode
gap likely **largely closes**. That is the point of the experiment and does not
threaten the config-independent claims (prefill reuse, fork/branch semantics).
If Pie still wins clearly at `fair`, apply §7's rule — audit for a confound
before celebrating.

---

## 8. Decide these yourself; escalate only these

**Just do it:** restarting a failed arm, installing a missing dependency under
the §6 rules, deleting a ruined predictions file, fetching `benchmark_moe.py`,
killing stray servers, re-running `summarize_ab.py`.

**Ask the human first:** changing any version pin, changing the instance set,
switching to vLLM 0.26.0, modifying `30_ab_run.sh`'s harness flags, or anything
that would invalidate an already-collected arm.

**Never:** re-run `bootstrap_runpod.sh` with default paths (it would try to
rebuild the vLLM venv from scratch — pass `PIE_VENV=/root/venvs/pie-vllm`), or
`git checkout` during a running arm.
