# Test plan — fair-parity rerun of Pie vs. litellm+vLLM on A100 SXM

Companion to `README.md` (which is the file index + run order). This document is
the **experimental design**: what question we are answering, why the experiment
is built the way it is, exactly what to run, and how to read the result. Written
so a reader who was not in the room can reproduce the run and defend the numbers.

---

## 1. The question

The existing writeup (`docs/pie-vs-litellm-writeup.md`) reports a Pie-backed
OpenHands agent beating the vanilla OpenHands→vLLM baseline by ~26% wall time on
SWE-bench. The writeup itself flags that the vLLM baseline was **crippled**: it
ran `--enforce-eager` (CUDA graphs off) with an untuned fused-MoE kernel on a
Blackwell card that had no tuned MoE config. So the honest open question is:

> **When vLLM is given its proper, documented configuration, does the wall-clock
> gap survive — and if not, which of Pie's advantages are config-independent?**

We are explicitly *trying to break our own result*. A defensible finding is more
valuable than a flattering one.

---

## 2. What "fair" means here — the optimization-effort principle

Tuning is a slider, not a switch: you can always spend more engineer-hours on
either engine. So "just optimize vLLM more" is not a well-defined target. The
rule we adopt:

> **Give each engine the optimization level a competent practitioner reaches
> using that project's own shipped, documented, supported tooling — and no custom
> per-benchmark kernel work that the other side did not also get.**

Applied:

- **vLLM's fair effort is flag-level + its own autotuner.** Not passing
  `--enforce-eager` (so CUDA graphs capture) is zero effort. Generating a tuned
  MoE config with vLLM's own `benchmark_moe.py` is *one documented step* and
  counts as fair — it is vLLM's maintainers' tool, not us hand-writing a kernel.
- **Pie's fair effort is its native driver as shipped.** The hand-fused Qwen3-MoE
  experts are legitimate: they are part of the engine, exactly as vLLM's Triton
  MoE autotuning is part of vLLM. We do **not** write new A100-specific kernels
  for this benchmark.

Both engines carry their maintainers' baked-in optimization effort. The
benchmark's only job is to not cripple either **at config time**.

### Why we measure the axis instead of picking one "fair" point

Rather than argue where the fair point sits, we run vLLM at three tiers and let
the data show what each increment of effort buys. This turns "optimization
effort" from a debate into a measured variable.

| Tier (`VLLM_TIER`) | CUDA graphs | MoE kernel | Effort it represents |
|---|---|---|---|
| `crippled` | off (`--enforce-eager`) | default | the original writeup's baseline (reproduce it) |
| `graphs-only` | **on** | default (no autotuner) | *zero* extra effort — just don't pass a bad flag |
| `fair` | **on** | **autotuned** | one documented autotune step |

Interpreting the gaps:
- **`crippled → graphs-only`** = what `--enforce-eager` alone cost the baseline.
- **`graphs-only → fair`** = what the MoE autotuner is worth on this GPU.
- **`fair` vs. `pie`** = the actual engine-vs-engine question, both at their
  shipped best.

Pie is a fourth line on the same plot.

---

## 3. Hypotheses and what would confirm / refute each

- **H1 (parity).** With vLLM at `fair`, the per-iteration wall time gap shrinks
  toward parity (Pie within noise, or a modest edge). *Refuted if* `fair` vLLM is
  clearly faster per iteration than Pie, or if Pie still wins by a wide margin
  (which would suggest something other than decode config is driving it).
- **H2 (config-independent wins).** Pie's prefill-reuse (≈95.8% via persistent
  sessions), accuracy, and fork/branch semantics do **not** depend on how vLLM's
  decode is configured, so they hold across all three tiers. *Refuted if* the
  reuse or accuracy numbers move with the vLLM tier (they should not — they are
  Pie-side properties).
- **H3 (structural advantage is overcommit-only).** In the fits-in-cache serial
  regime, a well-tuned vLLM and Pie are close; the regime where they diverge in
  kind is memory overcommit (swap vs. evict-and-recompute). This is the stretch
  experiment (§8), not required for the parity claim.

**Expected outcome, stated up front for honesty:** on A100 the decode gap likely
**largely closes**. That is the point of the experiment, and it does not threaten
H2. A likely-true summary sentence is: *"With vLLM properly tuned, per-iteration
serving speed is roughly at parity; Pie's durable advantages are explicit
persistent KV reuse, fork/branch semantics, and (workload-dependent) graceful
overcommit — not raw batch-1 decode."*

---

## 4. Design: what is held constant, what varies

**Held identical across all arms** (everything above the LLM boundary):
- Model `Qwen/Qwen3-Coder-30B-A3B-Instruct`, temperature 0.0.
- The stock OpenHands agent: tools, condenser (`max_size=240, keep_first=2`),
  8-phase system prompt, `--max-iterations 100`, serial (`tool_concurrency=1`).
- The same instance set, one GPU, serial execution.

**The only independent variable is the serving layer:**
- `pie` — OpenHands→PieLLM→coder-session inferlet→native CUDA driver (KV resident).
- `litellm` — OpenHands→litellm HTTP→`vllm serve`, at `VLLM_TIER ∈ {crippled,
  graphs-only, fair}`.

**Controls / guards:**
- `assert_vllm_fair.sh` gates every vLLM launch on the engine banner matching its
  declared tier — a silently-mis-tiered baseline cannot enter the numbers.
- `KV_VERIFY=1` on the Pie arm asserts every reused KV block is numerically
  identical to a from-scratch render (0 kv-verify errors is a pass condition).
- Fresh server per arm (60 GB of weights per engine won't co-reside on 80 GB
  anyway) — no cross-engine contention.

---

## 5. Metrics — and why each, measured honestly

| Metric | Source | Why it's the honest one |
|---|---|---|
| **wall time / agent iteration** | `_metadata.wall_clock_s / agent_iterations` | Trajectories diverge (two stochastic agents take different paths); raw wall time conflates path length with serving speed. Per-iteration divides that out. Same harness, same counter, both arms. |
| **median per-call latency** | `_metadata.response_latencies[]` | End-to-end per-LLM-call latency, recorded identically for every arm by the same harness. The cleanest single serving-speed number. |
| **prefill reuse %** | Pie session `prefill_tokens` vs `len` | Config-independent Pie property (H2). Reported for FLOPs/energy/cost, not as the wall-clock driver. |
| **accuracy (resolved / N)** | SWE-bench harness | Directional on ≤50 instances; use a neutral set so it can move both ways. |
| **decode tok/s** (microbench) | `20_decode_microbench.py` | Controlled batch-1 decode, **both engines through one client codepath**. |

### The measurement-fairness fix (do not repeat the old mistake)

The current writeup compares **85.5 tok/s (Pie, GPU-only)** against **64.2 tok/s
(vLLM, end-to-end incl. HTTP + tokenization)** — not apples-to-apples even before
config. This rerun measures decode the **same way** on both engines:
`20_decode_microbench.py` streams a fixed prompt with `ignore_eos` (so both emit
exactly N tokens) over an OpenAI-compatible endpoint and times it client-side.
Report `decode_tok/s` for both engines from this one path — never GPU-only vs.
end-to-end again.

---

## 6. Procedure

Prereqs (see `00_setup_a100.sh` LOGISTICS block): A100 **80 GB SXM (sm_80)**
confirmed; big `/workspace` volume; Pie **rebuilt on the box for sm_80** (do not
copy the sm_120 Blackwell binary); vLLM + harness venvs; model downloaded.

```bash
# 0. one-time setup on a bare pod (provisions the box, clones this ref, then
#    runs 00_setup_a100.sh itself). Run under tmux — the CUDA build is 30-60 min.
bash bootstrap_runpod.sh
source /workspace/pie-bench-env.sh     # every later shell needs this

# 1. generate the tuned MoE config once (needed only for the `fair` tier)
bash 11_autotune_moe.sh          # relaunch check: assert_vllm_fair.sh ... fair must pass clean

# 2. Pie arm (native CUDA driver, A100 config)
ARM=pie bash 30_ab_run.sh

# 3. vLLM baseline at all three effort tiers (each writes a distinct predictions file)
VLLM_TIER=crippled    ARM=litellm bash 30_ab_run.sh
VLLM_TIER=graphs-only ARM=litellm bash 30_ab_run.sh
VLLM_TIER=fair        ARM=litellm bash 30_ab_run.sh

# 4. effort-axis table (s/iter + median latency, shared-instance head-to-head)
python summarize_ab.py \
  pie=../predictions/ab_a100_pie_*.jsonl \
  litellm-crippled=../predictions/ab_a100_litellm_crippled_*.jsonl \
  litellm-graphs-only=../predictions/ab_a100_litellm_graphs-only_*.jsonl \
  litellm-fair=../predictions/ab_a100_litellm_fair_*.jsonl
# NB ../predictions — 30_ab_run.sh cd's to integrations/openhands before
# writing, so the jsonl lands one level above this runpod/ directory.

# 5. (optional) controlled decode microbench, both engines measured identically
python 20_decode_microbench.py --base-url http://localhost:18000/v1 \
       --model Qwen/Qwen3-Coder-30B-A3B-Instruct --label vllm-fair
```

**Pinned versions — record these once, alongside the banners.** A tier
comparison is only meaningful against a known software set:

| | pinned to | why |
|---|---|---|
| vLLM | `VLLM_VERSION=0.25.1` | unpinned takes whatever PyPI serves that day |
| Python (both venvs) | 3.12 | on 3.13 the vLLM set *resolves*, then torch dies at import (TorchScript overload parser). A successful resolve is not evidence the stack runs. |
| harness deps | `uv --exclude-newer 2026-05-15` | `openhands-sdk 1.21.1` (2026-05-08) declares 14 open-ended floors; at today's newest, litellm 1.93.0 fails to import its own `MessagesInterceptor` and fastmcp 3.x moved `Client`. Pin time, not packages. |

**What to record for the writeup, per arm:**
1. The `assert_vllm_fair.sh` banner block (proves the tier — the auditable,
   symmetric counterpart to the crippled banner the current writeup quotes).
2. The `summarize_ab.py` row (wall, iters, s/iter, median latency, tokens).
3. Pie arm only: KV-reuse % and kv-verify error count (want 0).
4. Accuracy from scoring the predictions (see `RUNPOD_SCORING.md`).

**Instance set:** the 13-instance set in `30_ab_run.sh` is the baseline's own
prior wins, so its accuracy ceiling is parity — fine for the timing signal but
not for accuracy. For an accuracy signal that can move both ways, swap in the
neutral-50 ids before running (accuracy is scored separately; timing is unaffected).

---

## 7. Interpretation / decision rules

- If **`fair` vLLM per-iteration ≈ Pie** → report **parity confirmed (H1)**; the
  headline becomes "no serving-speed penalty" + Pie's config-independent wins.
- If **`fair` vLLM clearly faster** → report it plainly; Pie's case rests on H2
  (reuse, fork semantics, accuracy) and H3 (overcommit), not decode.
- If **Pie still clearly faster at `fair`** → do **not** celebrate yet; audit for
  a confound (trajectory-length divergence not divided out, an unfair vLLM flag,
  Pie skipping work). A surviving per-iteration win after those checks is the
  strong result.
- **`graphs-only` vs `fair`** quantifies the autotuner's value; if the gap is
  tiny, the "we didn't run the autotuner" objection is moot and `graphs-only` is
  a sufficient baseline.

---

## 8. Threats to validity (and how each is handled)

- **Trajectory divergence** — two agents walk different paths; raw wall time is
  not comparable. *Handled:* normalize by iteration + report median per-call
  latency.
- **Silent mis-configuration** — the whole point is a fair vLLM. *Handled:*
  `assert_vllm_fair.sh` gates every launch and the banner is published.
- **Pie crippled by the platform move** — Pie's fused kernels are compiled for
  sm_80 but not per-GPU autotuned. *Acknowledged:* weak Pie decode on A100 is an
  honest result, not vLLM cheating; the writeup must say so.
- **Small-N accuracy** — ≤50 instances is directional, not a leaderboard.
  *Handled:* state it; use a neutral set; inspect losses for engine vs. model-
  reasoning causes.
- **Context-length truncation** — `--max-model-len 32768` matches the prior
  baseline; the condenser bounds context, but note it if any trajectory hits the
  cap.

---

## 9. Stretch — the only regime that structurally differs

Not required for the parity claim; run it if the parity result lands near-even
and we want to locate Pie's structural edge.

Push the working set past KV capacity (many concurrent sessions, or contexts
larger than the cache) and the designs diverge **in kind**:
- **vLLM + APC** degrades by evict-and-recompute (pay in GPU FLOPs; robust, never
  OOMs from this).
- **Pie** is designed to degrade by swap (cold KV pages spill to a host pool and
  page back; pay in PCIe bandwidth, no recompute) — **enable it** via
  `pie_cuda_native_config_30b_moe_a100.toml` (`swap_pool_size=8192`).

Hypothesis: when the reused prefix is large and recompute is expensive, swap
beats recompute; when the prefix is small or host bandwidth is the bottleneck,
recompute beats swap. Sweep concurrency / context size past capacity and compare
degradation curves. In the current writeup this regime is hypothesis-only (Pie's
swap was off there), so it is the experiment that would actually settle where
Pie's structural advantage pays off.

---

## 10. Storage sizing (runpod)

runpod separates **container disk** (ephemeral — wiped when the pod stops) from
the **persistent volume** mounted at `/workspace` (survives stop/restart).

**Do not put everything on `/workspace`.** An earlier version of this section
said to, and it is wrong. `/workspace` is a **MooseFS network volume**
(`df -h /workspace` shows `mfs#...runpod.net:9421`), so every file operation is
a network round-trip. Small-file-heavy work does not merely run slow there — it
**stalls**. Measured on the A100 pod: `uv pip install` of 67 small wheels parked
43 parallel downloads at ~15 KiB each and never progressed, while `curl` to the
same CDN ran at 125 MB/s. Note that `uv` streams downloads into `UV_CACHE_DIR`
*before* linking them into the venv, so a cache on `/workspace` stalls the
install no matter where the venv itself lives.

Split by **access pattern**, not by what you wish would persist:

| Goes on | What | Why |
|---|---|---|
| **container disk** (`$FAST`, default `/root`) | uv/pip/CPM caches, both venvs, cargo `target/` | small files, hot, cheap to rebuild |
| **`/workspace`** | the repo, the ~60 GB model weights | large sequential files — what MooseFS is actually good at — and expensive to refetch |

`bootstrap_runpod.sh` does this split, and symlinks `target/` and
`integrations/openhands/.venv` back into the repo because
`run_pie_backend.sh` hardcodes the latter and `CARGO_TARGET_DIR` is global
(setting it would also redirect the wasm inferlet build).

The cost is that venvs and the build tree are lost on pod **stop**; re-running
`bootstrap_runpod.sh` rebuilds them. The weights — the only genuinely expensive
artifact — persist.

**Container disk: 100 GB.** The runpod default of 60 GB is workable but tight:
venvs (~18) + cargo `target/` (~30–40) + CPM cache (~5) ≈ 55–60 GB with nothing
to spare. **Persistent volume (`/workspace`): 100 GB** (150–200 GB if also
scoring SWE-bench here) — it now holds only the repo and the weights.

| Item | Size | Lives on |
|---|---|---|
| Model weights + HF cache (`Qwen3-Coder-30B-A3B`, bf16) | ~60–65 GB | `/workspace` |
| pie checkout | <1 GB | `/workspace` |
| Pie build — `target/` (release, CUDA) + CPM source cache (cutlass/flashinfer) | ~40–50 GB | `$FAST` |
| vLLM venv (torch + vllm + kernels) | ~12–15 GB | `$FAST` |
| Harness venv (openhands sdk + deps) | ~2–3 GB | `$FAST` |
| uv/pip caches | ~8–10 GB | `$FAST` |
| coder-session wasm, predictions, logs | <1 GB | `/workspace` |

Note the base PyTorch/CUDA image (~20 GB) also sits on the container disk, which
is why 60 GB total leaves so little room once the build tree lands there.
Reclaim ~8–10 GB after setup with `uv cache clean` if it gets tight.

**SWE-bench scoring is the wildcard.** Building/pulling per-instance test images
is large and file-count heavy (a prior run hit an inode quota at ~15M files, each
sandbox being a full rootfs). Prefer scoring as a separate step (see
`RUNPOD_SCORING.md`), prune sandboxes between instances, and if scoring on this
same box add **+50–100 GB** to the volume. The A/B *timing* run itself does not
need this.
