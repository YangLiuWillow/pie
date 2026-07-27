# Live run state — fair-parity A/B on the A100 pod

**Companion to `AGENT_HANDOVER.md`.** That file is the standing handover (machine
state, storage rules, version pins, escalation policy) and is still accurate.
This file is the *session* state: what has been changed, what is running right
now, and exactly where to pick up. Read `AGENT_HANDOVER.md` first, then this.

Last updated: **2026-07-27 17:08 UTC**.

---

## 1. Status right now

Last updated **2026-07-27 17:08 UTC**.

| | |
|---|---|
| Pie arm | **COMPLETE — 13 / 13**, `../predictions/ab_a100_pie_20260727_104638.jsonl` |
| Current job | **MoE autotune**, PID `88099`, started 16:43 UTC |
| Autotune log | `../logs/autotune_moe_20260727_164323.log` |
| Tuned output | `/workspace/tuned_moe/` (NOT vLLM's package dir — see §3) |
| Autotune ETA | **~6–12 h, not "a couple"** — see below |

### Autotune duration — plan for hours, not one hour

`benchmark_moe.py` sweeps **1920 configs per batch size × 18 batch sizes**
(`benchmark_moe.py:912-931`), sequentially on the one A100. Measured rate at
`bs=1`: ~1.3–1.6 it/s → **~20 min per sweep**, and the larger batch sizes are
slower. Sampled `nvidia-smi` utilization reads 0% most of the time — that is
Triton compile dominating between short kernel bursts, **not** a stall.

`/workspace/tuned_moe/` stays **empty until the very end**: `save_configs` fires
only after all 18 sweeps return (`benchmark_moe.py:1005-1007`). There is no
partial-progress artifact, so a kill forfeits everything spent so far.

The `fair`-first reorder still holds under this ETA — both orderings finish at
about the same time, but `fair` first puts the headline Pie-vs-fair number ~8 h
earlier.

### Pie arm result (clean)

13/13 unique instances, **no empty patches, no missing telemetry, nothing hit the
100-iteration cap** (max 90). Totals: **13 594 s wall (3.78 h), 782 iterations,
378 LLM calls, 17.38 s/iter overall**, median LLM latency **17.9 s** across all
378 calls. Mean 733 735 prompt tokens / 7 192 completion tokens per instance.

Per-instance wall times ranged 537 s (`django-14373`) to 2175 s
(`scikit-learn-10908`, 90 iters). Per-iteration cost varied 10.7–24.2 s/iter,
which is why TEST_PLAN §5/§8 normalizes by iteration rather than wall clock.

Re-derive state after a reconnect:

```bash
source /workspace/pie-bench-env.sh
cd /workspace/pie/integrations/openhands
wc -l predictions/ab_a100_pie_*.jsonl
pgrep -af 'release/pie serve|run_swe_bench --backend'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

The arm is detached, so **losing SSH does not kill it**. Reattach by reading the
logs, not by restarting anything.

---

## 2. Uncommitted code changes — do not lose these

`git status` is dirty on purpose. `/workspace` persists across pod stop, so the
edits survive, but they are **not committed**:

```
 M integrations/openhands/benchmarks/swe_bench.py
 M integrations/openhands/benchmarks/humanevalfix.py
 M integrations/openhands/runpod/assert_vllm_fair.sh
 M integrations/openhands/runpod/run_litellm_baseline_fair.sh
```

Base commit: `883747e`. **Ask the human before committing** (they have not
requested it). All four changes are already active in the running arm's code
path except where noted.

### 2.1 Pie arm recorded no latencies or tokens (`swe_bench.py`)

**Root cause.** `AgentBase.get_all_llms()` (`openhands/sdk/agent/base.py:638,658`)
yields only objects whose type is *exactly* `LLM` — "no subclasses", by design.
`LocalConversation` feeds that generator into `llm_registry.add`, and the
registry is what subscribes `ConversationStats.register_llm`. A `PieLLM` is a
subclass, so it was never registered, `stats.usage_to_metrics` stayed empty, and
`get_combined_metrics()` returned a zeroed `Metrics`. Every pie row had
`num_llm_calls=0`, `response_latencies=[]`, `total_tokens=0`.

The telemetry was never broken — `LLM.completion` calls `telemetry.on_response`
around the `_transport_call` override and records latency unconditionally into
`llm.metrics`. Only the hand-off to the conversation was lost.

**Fix.** Added `_iter_llms()` (walks the agent for LLMs *including* subclasses)
and `_extract_metrics(conv, *roots)`, which merges metrics the registry skipped,
deduped by `id(metrics)` because registered LLMs store the same object in
`usage_to_metrics`. Verified both arms report identical totals through different
plumbing. Same one-line call-site fix applied in `humanevalfix.py:274`.

### 2.2 Patch capture dropped agent-created files (`swe_bench.py`)

Was `git diff HEAD`, which cannot see untracked files. Upstream does
`git add -A` → `git commit --no-verify` → `git diff <base> HEAD`. Now stages and
diffs `--cached` against HEAD — identical output, and no committer identity
needed in the throwaway clone (this is upstream's own `get_staged_git_patch`).

**Consequence, which is upstream's too:** the agent's Phase-4 reproduction script
is a new file and now lands in the patch. Instance 1's patch contains both
`django/forms/widgets.py` and `test_changes.py`. **Patch byte-sizes are therefore
not comparable to `predictions/neutral_50_pie.jsonl` or
`predictions/baseline_t0_full_50.jsonl`**, which were collected under the old rule.

### 2.3 Condenser shared the agent's LLM (`swe_bench.py`)

Upstream builds a separate condenser LLM (`build_eval_llm(..., usage_id="condenser")`).
We passed the same object, so on the Pie arm summarization interleaved into the
agent's own KV session — corrupting the very number prefill-reuse is meant to
measure. Now `llm.model_copy(update={"usage_id": "condenser"})`; `PieLLM.model_copy`
clears the session id, giving the condenser its own namespace.

Follow-on: `model_copy` shallow-copies private attrs, so the copy shared the
agent's `Metrics`. `LLMRegistry.add` calls `_ensure_independent_metrics` for
exactly this reason but only for LLMs it registers — never a `PieLLM`. Mirrored
with `reset_metrics()`. Confirmed live: both `usage_id: 'default'` and
`usage_id: 'condenser'` appear in the running agent's config dump.

### 2.4 Tier gate was strict in one direction only (`assert_vllm_fair.sh`)

Under-delivery failed; **over-delivery only warned**. Two holes:

- `graphs-only` with a tuned MoE config → warn. `11_autotune_moe.sh` writes its
  JSON into vLLM's package config dir **permanently**, so any `graphs-only` run
  after the autotune would silently have been a second `fair` run — collapsing
  the `graphs-only → fair` gap to ~0 and making "the autotuner isn't worth
  running" look proven when it was never tested.
- `crippled` with CUDA graphs ON → warn, which would understate what
  `--enforce-eager` cost.

Both are now hard failures. Tested across all 9 tier × banner combinations:
each tier passes only its own configuration.

### 2.5 Mislabeling trap (`run_litellm_baseline_fair.sh`)

`LABEL` and `OUTPUT` defaulted to `litellm-fair...` / `litellm_fair_<ts>.jsonl`
**regardless of tier**. Invoked directly rather than through `30_ab_run.sh`, a
crippled run would have been filed under the fair arm — `summarize_ab.py`
selects arms by filename glob. Both now derive from `$VLLM_TIER`.

### 2.6 `crippled` had no tuned-MoE check (`assert_vllm_fair.sh`)

The gate hard-failed `graphs-only` on a tuned config but had **no equivalent
branch for `crippled`**, whose definition is equally "`--enforce-eager` +
*default* MoE". Harmless while the autotune always ran last; load-bearing under
the §3 reorder, where `crippled` now runs *after* the autotune — an un-moved
JSON would have handed it the good kernel and understated what `--enforce-eager`
cost, which is precisely the `crippled → graphs-only` gap the axis exists to
measure. Added the symmetric hard fail.

Retested all 12 tier × banner combinations (3 tiers × {graphs on/off} ×
{MoE default/tuned}): each tier passes **only** its own configuration. The change
is strictly stricter, so it can reject a contaminated arm but never a valid one.

---

## 3. Remaining run order — one at a time, verify before the next

**Reordered 2026-07-27 ~14:05 UTC at the human's request: `fair` first**, to get
the headline Pie-vs-fair comparison ~6 h sooner. `fair` hard-fails on an untuned
MoE config (`assert_vllm_fair.sh:57`), so the autotune necessarily comes first.

**The ordering hazard is now gone — do not reintroduce the `mv` dance.** vLLM
0.25.1 honours `VLLM_TUNED_CONFIG_FOLDER`, and `fused_moe.py:1078-1089` searches
that folder **before** the package `configs/` dir. So the autotune writes to a
private folder via `--save-dir`, and the tier is selected purely by whether that
env var is exported. **vLLM's package config dir is never mutated**, so there is
no move-aside, no restore, and no way to contaminate an arm by forgetting one.

Confirmed absent from the package dir: vLLM 0.25.1 ships 326 MoE configs and no
`E=128,N=768` A100 entry (only `N=192`), so with the env var unset the
default-MoE tiers genuinely get the untuned kernel, and the autotune is
genuinely load-bearing.

```bash
source /workspace/pie-bench-env.sh
cd /workspace/pie/integrations/openhands/runpod
TUNED_DIR=/workspace/tuned_moe

# 1. Pie — COMPLETE (see §1)

# 2. the autotune — RUNNING. 11_autotune_moe.sh will NOT find benchmark_moe.py
#    (ships in vLLM source, not the wheel). Already fetched at the matching tag
#    to /workspace/benchmark_moe.py — run it directly.
#    --tp-size 1 IS MANDATORY: the script defaults to tp_size=2, which tunes
#    E=64,N=384. We serve tp=1, which needs E=128,N=768. See §6.
$PIE_VENV/bin/python /workspace/benchmark_moe.py \
    --model "$MODEL" --tune --dtype auto --tp-size 1 --save-dir "$TUNED_DIR"

# 3. the tier the autotune unlocks — THE HEADLINE ARM.
#    The env var is what makes it 'fair'.
VLLM_TUNED_CONFIG_FOLDER="$TUNED_DIR" VLLM_TIER=fair ARM=litellm bash 30_ab_run.sh

# 4/5. the default-MoE tiers — simply do NOT export VLLM_TUNED_CONFIG_FOLDER.
#      The gate hard-fails both if a tuned config is live, so this is verified,
#      not assumed.
VLLM_TIER=graphs-only ARM=litellm bash 30_ab_run.sh
VLLM_TIER=crippled    ARM=litellm bash 30_ab_run.sh

# 6. the effort-axis table
python summarize_ab.py \
  pie=../predictions/ab_a100_pie_*.jsonl \
  litellm-crippled=../predictions/ab_a100_litellm_crippled_*.jsonl \
  litellm-graphs-only=../predictions/ab_a100_litellm_graphs-only_*.jsonl \
  litellm-fair=../predictions/ab_a100_litellm_fair_*.jsonl
```

Launch detached so SSH loss cannot kill an arm:

```bash
TS=$(date +%Y%m%d_%H%M%S)
LOG=/workspace/pie/integrations/openhands/logs/ab_<arm>_harness_${TS}.log
setsid nohup bash -c 'source /workspace/pie-bench-env.sh; \
  cd /workspace/pie/integrations/openhands/runpod; \
  VLLM_TIER=<tier> ARM=litellm exec bash 30_ab_run.sh' > "$LOG" 2>&1 < /dev/null &
```

### Verify each arm before starting the next

1. **GPU fully released** — `nvidia-smi` must read ~0 MiB and no `pie serve` /
   `vllm` process. 60 GB per engine will not co-reside on 80 GB.
2. **Banner matches the tier** — `assert_vllm_fair.sh` output; `STRICT_FAIR=1`
   aborts before the harness starts. Paste it into the writeup.
3. **First instance genuinely iterating within 2 min.** The failure mode is 13
   rows of `0 iters / 0-byte patch / error=...` in ~30 s. A real success is a
   nonzero iteration count and a non-empty patch.
4. **Rows carry latencies** — `_metadata.response_latencies` non-empty and
   `num_llm_calls > 0`. If a pie row shows these empty, fix §2.1 is not in the
   running code path.

Per-instance progress (the ground truth is the predictions file):

```bash
python3 -c "
import json,glob
for f in sorted(glob.glob('/workspace/pie/integrations/openhands/predictions/ab_a100_*.jsonl')):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    print(f.split('/')[-1], len(rows), 'of 13')
    for r in rows:
        m=r.get('_metadata',{}); p=r.get('model_patch') or ''; it=m.get('agent_iterations',0)
        lat=m.get('response_latencies') or []
        med=sorted(lat)[len(lat)//2] if lat else 0
        print(f\"  {r['instance_id']:45} {it:>4} it {len(p):>6}B {m.get('wall_clock_s',0):>6.0f}s \"
              f\"{m.get('wall_clock_s',0)/it if it else 0:>5.1f} s/it med={med:.1f}s\")
"
```

---

## 3b. METHOD CHANGE — autotune abandoned, config borrowed (human, 17:45 UTC)

**The `fair` tier no longer comes from our own autotuner.** `benchmark_moe.py
--tune` was killed at ~17:45 after 1 of 18 batch-size sweeps in 57 min
(15-24 h+ projected, nothing written until all 18 finish).

Replaced by **SGLang's published A800-SXM4-80GB config**, renamed to the A100
device name and installed at
`/workspace/tuned_moe/E=128,N=768,device_name=NVIDIA_A100-SXM4-80GB.json`.
A800 is the same GA100 silicon as A100 — identical compute, 80 GB HBM2e,
~2 TB/s — differing only in NVLink bandwidth, which a tp=1 single-GPU fused MoE
kernel never touches.

Full justification in `/workspace/tuned_moe/PROVENANCE.md`; the measured
before/after on this exact GPU is in `/workspace/tuned_moe/VALIDATION.md`
(**-8.3% overall**, -8.9% at bs=1, -18.7% at bs=2048; faster at 17 of 18 batch
sizes). **Both files must be cited in the writeup** — the tier is no longer
"we tuned it", it is "we used a third-party tuned config and measured that it
beats the default".

No published A100 config exists for this shape in vLLM (any version), SGLang, or
the community repos — checked 2026-07-27. H100 has no bf16 `E=128,N=768` either;
vLLM ships that bf16 shape only for H200 / B200 / H20 / MI308X, and SGLang adds
H100 and A800.

Optional follow-up, costs nothing already collected: run the full autotune later
and report how close SGLang's config came.

---

## 3a. Accuracy — deliberately deferred (human, 2026-07-27 ~17:00 UTC)

**This writeup is a timing result. Do not put an accuracy number in it.**
Revisit only after the autotune and the `fair` arm have both landed.

Why, and what the constraints are when it is revisited:

- **Nothing has been scored.** The prediction rows carry only `instance_id`,
  `model_name_or_path`, `model_patch`, `_metadata` — no `resolved` field, no eval
  report anywhere in the tree.
- **This pod cannot score.** No `docker`, no `apptainer`/`singularity`, no
  `swebench` package in either venv. Scoring is CPU work and belongs on a
  separate pod — see `../RUNPOD_SCORING.md` for the exact harness invocation.
- **The 13-instance set cannot yield a meaningful accuracy number** regardless of
  tooling: it is the baseline's own prior wins (`30_ab_run.sh:28-36`), so the
  ceiling is parity. Timing instrument only (`AGENT_HANDOVER.md` §7).
- **The only accuracy data that exists is partial and stale.** Apptainer scoring
  of the neutral-50 pair reached 29/50 before an inode-quota error: Pie 13,
  litellm 7, zero only-litellm wins. Those files predate every §2 fix and the
  new patch-capture rule, so scoring them measures the *old* harness.
- **9 of the 13 current patches contain agent-created scratch tests**
  (`test_fix.py`, `test_comprehensive.py`, `validate_fix.py`, …); xarray-4075 and
  matplotlib-22719 carry three each. Only `django-15569`, `scikit-learn-13496`,
  `sympy-19346` are clean. This matches upstream's `get_staged_git_patch`
  exactly — not a deviation — but scoring applies those scripts into the testbed.
- Plus RUN_STATE §5 items 1–2 (no dependency install / no conda; 100 vs 500
  iterations), which put any resolve rate out of line with published SWE-bench.

---

## 4. Preflight already done (2026-07-27, do not redo)

| | |
|---|---|
| `benchmark_moe.py` | fetched at tag `v0.25.1`, 1074 lines, compiles against the venv → `/workspace/benchmark_moe.py` |
| Harness venv | openhands-sdk **1.21.1** = openhands-tools **1.21.1** (§6 lockstep ✓), litellm **1.84.0** (below the broken 1.93.0) |
| vLLM venv | 0.25.1, torch 2.11.0+cu130, CUDA 13.0 |
| litellm arm CLI | all 12 flags accepted by `run_swe_bench` — no arg-parse failure waiting hours in |
| Disk | `/root` 43 G free, `/workspace` 148 T |
| Tuned MoE for this model | **none ships.** Model is `E=128, N=768, bf16`; vLLM 0.25.1 ships 35 A100 MoE configs but the only `E=128` one is `N=192`. So `graphs-only` genuinely gets the untuned kernel and the autotune is genuinely load-bearing. |

---

## 5. Protocol vs upstream (OpenHands/benchmarks)

Compared against the pinned submodule commit `4e5469e`, which is *exactly*
current upstream HEAD, cloned to `/root/oh-benchmarks` for diffing.

**Matches:** dataset `princeton-nlp/SWE-bench_Verified` / `test`; condenser
`max_size=240, keep_first=2`; `get_default_tools(enable_browser=False)`;
`cli_mode=True`; fake-user nudge text is **byte-identical**, `max_fake_responses=10`;
8-phase prompt near-identical.

**Still differs (accepted, must be stated in the writeup):**

1. **Execution environment — the largest gap.** Upstream runs each instance in
   the official `docker.io/swebench/` image, repo copied from `/testbed`, with
   the conda `testbed` env preinstalled. We use a bare `git clone` into a temp
   dir, **no dependency install, no conda**. Our prompt still tells the model
   the environment is ready when it is not, so Phases 2/4/7 can burn iterations.
   Hits both arms identically → **timing comparison unaffected**.
2. **`--max-iterations 100`** vs upstream's CLI default **500**. Deliberate.
3. **Prompt placement** — upstream sends the whole thing as one *user* message
   from `prompts/default.j2`; we split it, phases going into the *system* prompt
   via `extra_instructions`.
4. Upstream calls `create_agent_context()` (loads public skills); we do not. We
   add a StuckDetector nudge upstream lacks; upstream escalates after ≥2 fake
   responses and we do not.

**Bottom line:** the parity/timing result is unaffected by all of it — both arms
share one harness, so the only independent variable is still the serving layer.
**Accuracy is not comparable to published SWE-bench numbers** (items 1–2, plus
the pre-fix patch bug). The 13-instance set is the baseline's own prior wins, so
its accuracy ceiling is parity anyway — clean for timing, not an accuracy signal.

---

## 6. Gotchas hit this session

- **`benchmark_moe.py` needs `ray`, which was not installed.** The earlier
  preflight recorded it as "compiles against the venv OK" — that was
  `py_compile`, which does **not** execute imports, so a missing third-party
  module passes it. `import ray` then failed, and ray's own `msgpack` after it.
  Both installed into `/root/venvs/pie-vllm` with `uv pip install --no-deps`
  (ray 2.56.1, msgpack 1.2.1); `pip freeze` diffed before/after confirms
  **vllm 0.25.1 / torch 2.11.0 / numpy 2.3.5 / transformers 5.14.1 / triton 3.6.0
  all unmoved**. Lesson: "it compiles" is not "it imports".
- **`benchmark_moe.py --tp-size` defaults to 2 and silently tunes the wrong
  shape.** With `tp_size=2` it computes `E = 128//2 = 64` and
  `shard_intermediate_size = 2*768/2 = 768` → writes **`E=64,N=384`**. We serve
  single-A100 `tp=1` (the serve script passes no tensor-parallel flag), which
  needs **`E=128,N=768`**. Caught in the launch banner ~20 s in; the first
  attempt was killed before it wrote anything. **Always pass `--tp-size 1`.**
  Note the tier gate *would* have caught this eventually — `fair` hard-fails on
  a default-MoE banner — but only after burning the full autotune.
- **`pkill -f '<pattern>'` kills your own shell** when the pattern appears in its
  command line. Cost one confusing exit-144 in the earlier session and **a second
  one this session** while cleaning up ray processes. Kill by PID, always.
- **`pgrep -f 'run_swe_bench'` matches your own tooling** too — verify with
  `ps -o pid,ppid,etime,cmd -p <pids>` before concluding an arm is alive.
- **The harness's `N iters` tally never reached a log file.**
  `run_pie_backend.sh:104` redirects only the *pie server*; the harness's stdout
  went to the tmux pane. `grep 'iters' logs/*.log` returned nothing and this was
  *not* a symptom of anything. Current arms are launched with stdout redirected
  to `logs/ab_*_harness_*.log`, so the tally is greppable now.
- **Weight reload can be ~90 s, not minutes**, once the page cache is warm — a
  fast boot is not evidence something was skipped.
- **Trajectories diverge run-to-run even at temperature 0.** The same instance
  went 66 iters / 1008 s pre-fix and 56 iters / 790 s post-fix. Giving the
  condenser its own KV session changes the reuse pattern, and reuse-vs-recompute
  is not bit-identical, so one differing token cascades. This is exactly why
  TEST_PLAN §5/§8 normalizes by iteration — **never compare raw wall time**.
- The pre-fix arm's single row is preserved at
  `../logs/ab_a100_pie_20260727_100716.PRE-FIX.jsonl.bak`. It is kept **out of
  `predictions/`** on purpose: `summarize_ab.py` globs `ab_a100_pie_*.jsonl` and
  would otherwise merge an un-instrumented arm into the current one.

---

## 7. Escalate, do not improvise

Per `AGENT_HANDOVER.md` §8, stop and ask the human before: changing a version
pin, changing the instance set, switching vLLM versions, modifying
`30_ab_run.sh`'s harness flags, or anything that would invalidate a collected
arm. Restarting a failed arm, installing a missing dependency under §6 rules,
deleting a ruined predictions file, and killing stray servers are all fine to
just do.

**Standing authorization (human, 2026-07-27 11:10 UTC): run the §3 sequence
straight through without asking between arms.** Launching the next arm, running
the autotune, and producing the summary need no confirmation — the per-arm
verification in §3 is still mandatory, it just gates the run rather than a
question. The escalation list above is unchanged and still overrides this.
