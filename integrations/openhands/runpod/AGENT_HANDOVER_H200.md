# Agent handover — the A/B on a fresh H200 pod

> **Read this file before touching anything.** Then `TEST_PLAN.md` (experimental
> design) and `SESSION_NOTES_20260727.txt` (what the A100 run actually found).
> `AGENT_HANDOVER.md` and `RUN_STATE.md` describe the **A100** pod and are
> superseded for machine state — but `RUN_STATE.md` §0 is the reason this pod
> exists and is still required reading.

Your job: bring up the pod with `00_setup_h200.sh`, collect the arms, report the
numbers. Act and verify; don't ask permission for anything in §8's "just do it"
list.

---

## 1. Why this pod is an H200 (the finding that reset the experiment)

The A100 rerun was going to test whether a **properly configured vLLM** closes
the ~26% gap the original writeup claimed for Pie. It measured something else
entirely.

**Pie's two fast attention paths are hard-gated to compute capability ≥ 9:**

```
driver/cuda/src/entry.cpp:988-989
    fwd_cfg.use_prefill_decode_plan =
        (serving_prop.major >= 9 || force_prefill_decode_plan) && ...

driver/cuda/src/ops/attention_xqa.cu:274
    return current_device_major() >= 9;      // xqa_decode_bf16_supported
```

The A100 is sm_80, **major 8**. Both were **off for the entire run**, silently.
The driver said so in its own banner and it was read past:

```
[pie-driver-cuda] ... prefill_decode_plan=off xqa_decode=off decode_plan_graph=on
```

`PIE_CUDA_XQA_DECODE` **defaults to on** — xqa was not disabled by choice, the
*support check* fails on Ampere.

`build_pie_cuda.sh:6` confirms the original writeup targeted **RTX Pro 6000
Blackwell, sm_120, major 12**, where both gates pass. So the original "+26%"
was measured with Pie's fast paths **on**, and the A100 rerun measured Pie with
them **off**. The config-effort axis the rerun was built around
(`--enforce-eager`, tuned MoE) is not what moved the number.

**H200 is sm_90, major 9 — both gates pass.** That is the entire reason for this
pod. `00_setup_h200.sh` §8 verifies it at runtime and refuses to continue if the
banner reads `off`.

### What H200 does NOT restore

Two further gates need **major ≥ 12** and stay off here:

| gate | source | needs | A100 (8) | **H200 (9)** | RTX 6000 (12) |
|---|---|---|---|---|---|
| `use_prefill_decode_plan` | `entry.cpp:989` | ≥ 9 | ✗ | **✓** | ✓ |
| `xqa_decode_bf16_supported` | `attention_xqa.cu:274` | ≥ 9 | ✗ | **✓** | ✓ |
| `wide_prefill_device` | `cuda_memory_planner.cpp:219` | ≥ 12 | ✗ | **✗** | ✓ |
| `prefill_candidate_cap` 16384 | `cuda_memory_planner.cpp:250` | ≥ 12 | ✗ | **✗** | ✓ |

So this is **Pie-with-fast-attention, not the exact configuration the original
writeup ran**. A null result here does not strictly refute the original claim —
only sm_120 would. **State this in the writeup.** H200 was chosen because vLLM
ships a tuned MoE config for it (§4), so neither side needs tuning from us.

---

## 2. Carry-over results from the A100 pod

Both arms completed there. Treat them as follows.

| | value | status |
|---|---|---|
| Pie: 13/13, 17.38 s/iter, median 17.93 s, 95.34% prefill reuse | `predictions/ab_a100_pie_20260727_104638.jsonl` | **INVALID as a Pie measurement** (§1). Keep for the record. |
| vLLM fair: 13/13, 1.682 s/iter, median 1.206 s | `predictions/ab_a100_litellm_fair_20260727_175145.jsonl` | Valid as a *vLLM-on-A100* datapoint. |

**Do not publish "vLLM is ~10× faster than Pie" from the A100 pair.** It measures
Pie's fallback path and is as much a hardware artifact as the claim it was meant
to correct. `TEST_PLAN.md` §7 requires auditing a Pie *win* for confounds; the
rule is symmetric.

**One A100 result is real and survives:** vLLM's automatic prefix caching hit
**95.1%** against Pie's **95.34%** explicit session reuse. On this workload stock
vLLM matches Pie's prefill reuse. That does not depend on the arch gates and is
the most defensible thing collected so far.

---

## 3. The storage rule — violate it and things hang, not fail

`/workspace` is a **MooseFS network volume** (`df -h /workspace` shows
`mfs#...runpod.net:9421`). Small-file-heavy work there does not run slow — it
**stalls indefinitely, with no error**, which is very easy to misdiagnose as a
network problem. It is not: `curl` to PyPI from these pods runs at ~125 MB/s.

- **Local disk (`/root`)** — venvs, caches, the cargo build tree. Anything with
  many small files. **This is wiped when the pod is replaced.**
- **`/workspace`** — repo, model weights, predictions, logs. Large sequential
  files. Survives only if you re-attached the same network volume.

Before any `uv pip install`: `export UV_CACHE_DIR=/root/.uv-cache`. uv streams
downloads into its cache *before* linking them into the venv, so a cache on
MooseFS stalls the install regardless of where the venv lives. This cost hours
once.

Every shell starts:

```bash
source /workspace/pie-bench-env.sh
cd /workspace/pie/integrations/openhands/runpod
```

---

## 4. H200 inverts the MoE-config logic — this is the biggest procedural change

**vLLM ships a complete tuned bf16 config for this exact shape on H200:**

```
E=128,N=768,device_name=NVIDIA_H200.json     18 batch-size keys, 1 → 4096
```

Consequences:

- **`fair` needs no autotune and no `VLLM_TUNED_CONFIG_FOLDER`.** It is simply
  the default. Do **not** export that variable on this pod.
- vLLM normalizes the whole H200 family
  (`if "H200" in device_name.split("_")` in `get_config_file_name`), so **H200
  NVL and H200 SXM both resolve to the same file**. No renaming needed.
- On A100 the *tuned* config was the special case. **Here the *untuned* one is.**
  `VLLM_TUNED_CONFIG_FOLDER` only *prepends* a search path — it cannot suppress
  the packaged file. So `crippled` and `graphs-only`, both defined as *default*
  MoE, now require physically moving the shipped JSON aside:

```bash
CFGDIR=$($PIE_VENV/bin/python -c "import os,vllm.model_executor.layers.fused_moe.fused_moe as m;print(os.path.join(os.path.dirname(m.__file__),'configs'))")
mv "$CFGDIR/E=128,N=768,device_name=NVIDIA_H200.json" /root/moe_shipped.json.bak   # before crippled
mv /root/moe_shipped.json.bak "$CFGDIR/E=128,N=768,device_name=NVIDIA_H200.json"   # restore after
```

`assert_vllm_fair.sh` hard-fails `crippled`/`graphs-only` if a tuned config is
live, so a forgotten restore is caught rather than silently contaminating an arm.
Its remediation hints say "unset `VLLM_TUNED_CONFIG_FOLDER`", which is the A100
advice — on this pod the fix is the `mv` above.

### Recommendation: drop the `graphs-only` tier on H200

`graphs-only` exists to answer "what do you get without running the autotuner?"
On H200 nobody runs the autotuner — the config ships. The tier would mean "what
you get if you delete a file vLLM ships", which is not a configuration any real
user is in. The honest axis here is two points:

- **`crippled`** — the original writeup's `--enforce-eager` + default MoE
- **`fair`** — stock modern vLLM

That is also a cleaner story for a blog post. Escalate before removing it if you
disagree.

---

## 4a. Context cap — `MAX_MODEL_LEN=131072` (decided, 2026-07-27)

`00_setup_h200.sh` writes this into the env file. Both serve scripts already read
it (`run_litellm_baseline_fair.sh:62`, `10_vllm_serve_fair.sh:87`), so no edit.

**Why it changed from the A100's 32768.** That was a memory necessity, not a
choice: the A100's 12.59 GiB of KV held 137,472 tokens, so 131072 would have left
room for ~1.05 sequences. H200 has ~72 GB of KV — about 762,000 tokens at
**96 KiB/token** (48 layers × 4 KV heads × 128 dim × 2 × 2 bytes bf16) — so
131072 costs ~12.6 GB and still leaves ~5.8× concurrency. §0 of the setup script
computes this and hard-fails if the value does not fit.

**Why it matters.** The serving cap is the **only** context limit in this stack:

- litellm has **no entry for a self-hosted model** — that is the benign
  `Cost calculation failed: This model isn't mapped yet` line, once per call — so
  nothing truncates client-side.
- the condenser bounds history by **message count (240), not tokens**.
- **Pie has no fixed cap at all.** Its ceiling is memory-planned (~762k tokens
  here). So a cap that is too low fails vLLM on inputs Pie serves fine, which
  surfaces as a *failed instance in one arm only* and reads as an accuracy
  difference when it is a configuration artifact.

131072 is the OpenHands SWE-bench norm (128k). The model is native **262144**
with `rope_scaling: None`; do not exceed that without YaRN, which would change
model behaviour rather than just the ceiling.

**Nothing came near the old cap** — the A100 `fair` arm logged zero length
rejections at 32768. This is insurance against trajectory divergence (§9), not a
fix for an observed failure.

---

## 5. Run order

```bash
source /workspace/pie-bench-env.sh
cd /workspace/pie/integrations/openhands/runpod

bash 00_setup_h200.sh          # idempotent; §8 of it MUST print "PASS"

# 1. Pie arm — the one that was invalid on A100
ARM=pie bash 30_ab_run.sh

# 2. vLLM fair — shipped MoE config; do NOT set VLLM_TUNED_CONFIG_FOLDER
VLLM_TIER=fair ARM=litellm bash 30_ab_run.sh

# 3. vLLM crippled — move the shipped JSON aside FIRST (§4), restore after
VLLM_TIER=crippled ARM=litellm bash 30_ab_run.sh

# 4. the table
python summarize_ab.py \
  pie=../predictions/*_pie_*.jsonl \
  litellm-fair=../predictions/*_litellm_fair_*.jsonl \
  litellm-crippled=../predictions/*_litellm_crippled_*.jsonl
```

### REQUIRED first edit — `30_ab_run.sh` is hardcoded to the A100

Two lines will silently do the wrong thing on this pod:

```bash
# line ~55: forces the A100 toml even if you export CFG
export CFG=$RUNPOD_DIR/pie_cuda_native_config_30b_moe_a100.toml
# lines ~44,57: names outputs ab_a100_*, which summarize_ab.py globs
export OUTPUT=predictions/ab_a100_pie_${TS}.jsonl
```

Leaving these means (a) Pie runs with the 80 GB A100 memory config on a 141 GB
card, and (b) **H200 predictions land in the `ab_a100_*` namespace and get merged
with the A100 arms by `summarize_ab.py`'s glob** — which would silently average
an invalid Pie arm into a valid one. Make them overridable:

```bash
export CFG=${CFG:-$RUNPOD_DIR/pie_cuda_native_config_30b_moe_h200.toml}
export OUTPUT=${OUTPUT:-predictions/ab_h200_pie_${TS}.jsonl}
export OUTPUT=${OUTPUT:-predictions/ab_h200_litellm_${VLLM_TIER}_${TS}.jsonl}
```

This is on the §8 escalate list normally. It is **pre-authorized** here because
the alternative is corrupt data. Do it before the first arm and say so in your
report.

### Verify each arm before starting the next

1. **GPU released** — `nvidia-smi` ~0 MiB, no `pie serve` / `vllm` process. ~58 GB
   of weights per engine will not co-reside comfortably.
2. **Pie arm only: the gate banner reads `prefill_decode_plan=on xqa_decode=on`.**
   If it reads `off`, stop — the arm is invalid and there is no point collecting it.
3. **Tier banner matches** (`assert_vllm_fair.sh`; `STRICT_FAIR=1` aborts before
   the harness starts). Paste it verbatim into the writeup.
4. **First instance genuinely iterating within 2 min.** The failure mode is 13
   rows of `0 iters / 0-byte patch / error=...` in ~30 s.
5. **Rows carry latencies** — `_metadata.response_latencies` non-empty and
   `num_llm_calls > 0`.

Launch detached so SSH loss cannot kill an arm:

```bash
TS=$(date +%Y%m%d_%H%M%S)
LOG=/workspace/pie/integrations/openhands/logs/ab_<arm>_harness_${TS}.log
setsid nohup bash -c 'source /workspace/pie-bench-env.sh; \
  cd /workspace/pie/integrations/openhands/runpod; \
  VLLM_TIER=<tier> ARM=<arm> exec bash 30_ab_run.sh' > "$LOG" 2>&1 < /dev/null &
```

Per-instance progress (the predictions file is ground truth):

```bash
python3 -c "
import json,glob
for f in sorted(glob.glob('/workspace/pie/integrations/openhands/predictions/ab_h200_*.jsonl')):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    print(f.split('/')[-1], len(rows), 'of 13')
    for r in rows:
        m=r.get('_metadata',{}); it=m.get('agent_iterations',0)
        lat=m.get('response_latencies') or []
        med=sorted(lat)[len(lat)//2] if lat else 0
        print(f\"  {r['instance_id']:45} {it:>4} it {len(r.get('model_patch') or ''):>6}B \"
              f\"{m.get('wall_clock_s',0)/it if it else 0:>5.1f} s/it med={med:.1f}s\")
"
```

---

## 6. Version pins — do not "upgrade" these

| pin | why |
|---|---|
| **Python 3.12** (both venvs) | On 3.13 the vLLM set *resolves and installs*, then torch dies at import in the TorchScript overload parser (`torch/_sources.py parse_def` → `IndentationError`). A successful resolve is not evidence the stack runs. |
| **vLLM 0.25.1** | A tier comparison needs one engine version, and the A100 `fair` arm used it. 0.26.0 exists — do not switch mid-experiment. |
| **harness deps `--exclude-newer 2026-05-15`** | `openhands-sdk 1.21.1` declares 14 open-ended floors. At today's newest, litellm 1.93.0 fails to import its own `MessagesInterceptor` and fastmcp 3.x moved `Client`. Pin time, not packages. |
| **`openhands-tools` == `openhands-sdk`** | `benchmarks/swe_bench.py:370` imports `openhands.tools.preset.default`. They ship in lockstep. `00_setup_h200.sh` §4 asserts equality. |

Any install into the harness venv:

```bash
export UV_CACHE_DIR=/root/.uv-cache
VIRTUAL_ENV=/root/venvs/harness uv pip install --exclude-newer 2026-05-15 <pkg>
```

---

## 7. What to record — this is the deliverable

Per arm:

1. **The `assert_vllm_fair.sh` banner block** for each vLLM tier, verbatim.
2. **The Pie driver's feature banner**, verbatim — the `prefill_decode_plan=` /
   `xqa_decode=` line. This is new and it is now the most important single line
   in the writeup: it is the proof the A100 mistake was not repeated.
3. **The `summarize_ab.py` row** — wall, iterations, **s/iter**, median per-call
   latency, tokens.
4. **Pie arm only**: prefill-reuse % and the kv-verify error count (**must be 0**).
5. **The version set** from §6, once, alongside the banners.
6. **The compute capability** and which gates it enables (§1 table).
7. **`max_prompt_tokens` across all rows, against `MAX_MODEL_LEN`.** Rows now
   carry `_metadata.max_prompt_tokens` and `_metadata.prompt_tokens_per_call`,
   so the writeup can state *"no request exceeded N tokens, against a cap of
   131072"* rather than assuming it. Check it per arm:

   ```bash
   python3 -c "
   import json,glob
   for f in sorted(glob.glob('/workspace/pie/integrations/openhands/predictions/ab_h200_*.jsonl')):
       rows=[json.loads(l) for l in open(f) if l.strip()]
       mx=max((r['_metadata'].get('max_prompt_tokens',0) for r in rows), default=0)
       print(f'{f.split(\"/\")[-1]:50} max_prompt_tokens={mx:,}')
   "
   ```

   If any arm's max approaches the cap, say so explicitly — it means the cap was
   nearly binding and the arms were not on equal footing.

Report **s/iter and median per-call latency, never raw wall time** — the two
agents walk different trajectories, so raw wall conflates path length with
serving speed (`TEST_PLAN.md` §5, §8).

### State these accurately, not favourably

- **`KV_VERIFY=1` is a token-count check, not numerical KV equivalence.** It
  reaches exactly one line — `inferlets/openhands-coder-session/src/lib.rs:739`,
  `ctx.seq_len() as usize != full_tokens.len()`. It catches what broken prefix
  reuse produces. Do not describe it as proving reused KV blocks are numerically
  identical to a fresh render.
- **The 13-instance set is the baseline's own prior wins**, so its accuracy
  ceiling is parity. Clean for timing; **not** an accuracy signal.
- **Accuracy has not been measured and cannot be measured on a GPU pod** (no
  docker/apptainer/swebench). Deferred by the human. See `RUN_STATE.md` §3a.
- **H200 is not sm_120**, so this does not strictly refute the original claim (§1).

---

## 8. Decide these yourself; escalate only these

**Just do it:** restarting a failed arm, installing a missing dependency under
§6's rules, deleting a ruined predictions file, killing stray servers, re-running
`summarize_ab.py`, the §5 `30_ab_run.sh` parameterization, moving the shipped MoE
JSON aside for `crippled` and restoring it.

**Ask the human first:** changing any version pin, changing the instance set,
switching vLLM versions, changing harness flags, removing the `graphs-only`
tier, or anything that would invalidate a collected arm.

`MAX_MODEL_LEN` is **already decided** (§4a) — 131072. Do not change it without
asking, but do not treat setting it as a change; it is the configured default.

**Never:** collect a Pie arm when the driver banner reads
`prefill_decode_plan=off`, or `git checkout` during a running arm.

---

## 9. Gotchas that have already cost time

- **`pkill -f '<pattern>'` and `pgrep -f` match your own shell** when the pattern
  appears in its command line. This produced exit 144 **three times** in one
  session. Kill by PID; test liveness with `kill -0 <pid>`.
- **"It compiles" is not "it imports."** `py_compile` does not execute imports,
  so a preflight passed while `ray` was missing.
- **Weight load is ~58 GB**; several minutes of silence at boot is normal.
  `nvidia-smi` memory climbing is the progress bar. Once the page cache is warm
  it can be ~90 s — a fast boot is not evidence something was skipped.
- **`Cost calculation failed: This model isn't mapped yet`** is **benign**, once
  per LLM call — litellm has no price entry for a self-hosted model. Its presence
  is evidence the agent is calling the model.
- **Trajectories diverge run-to-run even at temperature 0.** The same instance
  went 66 iters / 1008 s and 56 iters / 790 s. This is exactly why `TEST_PLAN.md`
  §5/§8 normalizes by iteration — **never compare raw wall time**.
- **The context cap is the only limit in the stack, and the arms are asymmetric.**
  Resolved for H200 by setting `MAX_MODEL_LEN=131072` (§4a). Do not lower it back
  toward the A100's 32768 without reading that section — instances averaged ~27k
  prompt tokens per call there, so the margin was thinner than it looked.
- **Paths in the older docs are stale.** `AGENT_HANDOVER.md` says the model is at
  `/workspace/hf-cache`; it is actually at `/workspace/.cache/huggingface/`.
  Any `/Users/yangliu/...` path in a doc refers to the human's Mac, not the pod.

---

## 10. Files

| file | what |
|---|---|
| `00_setup_h200.sh` | this pod's bring-up. Idempotent. §8 of it is the gate check. |
| `pie_cuda_native_config_30b_moe_h200.toml` | Pie config for 141 GB / sm_90 |
| `AGENT_HANDOVER_H200.md` | this file |
| `RUN_STATE.md` §0 | the arch-gate finding, in full |
| `SESSION_NOTES_20260727.txt` | plain-text narrative of the whole A100 session |
| `TEST_PLAN.md` | experimental design, hypotheses, decision rules |
| `tuned_moe/` | the borrowed A800 config used for the A100 `fair` arm. **Not needed on H200** — kept as the record of that arm. |
| `assert_vllm_fair.sh` | tier gate. Its remediation hints assume the A100 flow (§4). |
