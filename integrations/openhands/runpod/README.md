# Fair-parity rerun on A100 SXM (runpod)

Rerun of `docs/pie-vs-litellm-writeup.md` with the vLLM baseline **not crippled**,
so the wall-clock claim rests on a defensible comparison. The original baseline
ran `--enforce-eager` (CUDA graphs off) with an untuned MoE kernel on a Blackwell
card that had no tuned MoE config — the writeup itself flags this. This bundle
gives vLLM its proper config on A100 (its most mature, most-tuned target) and
turns "trust me, it's fair" into a mechanical banner check.

## The fairness principle we're benchmarking

Maximal tuning is as unfair as zero tuning — you can always sink more hours into
either side. The rule here:

> **Give each engine the optimization level a competent practitioner reaches
> using that project's own shipped, documented, supported tooling — and no
> custom per-benchmark kernel work the other side didn't also get.**

- **vLLM's fair config is all flag-level or shipped-autotuner effort**: drop
  `--enforce-eager` (CUDA graphs on); if no tuned MoE config ships for the card,
  run vLLM's *own* `benchmark_moe.py` (that counts — it's vLLM's tool, not us
  hand-writing a kernel); prefix caching on.
- **Pie's fair config is its native driver as shipped** (the hand-fused Qwen3-MoE
  experts are legitimate — part of the engine, same as vLLM's Triton MoE
  autotuning is part of vLLM), swap pool configured. No new A100-specific kernels
  written just for this benchmark.

Both engines carry their maintainers' baked-in effort. The benchmark's only job
is to not cripple either at **config** time.

## We measure the effort axis, not one "fair" point

Instead of arguing where "fair" sits, run vLLM at three tiers and let the numbers
show what each increment of effort buys:

| `VLLM_TIER` | CUDA graphs | MoE kernel | What it isolates |
|---|---|---|---|
| `crippled` | off (`--enforce-eager`) | default | reproduces the original writeup baseline |
| `graphs-only` | **on** | default (no autotuner) | value of just *not* passing a bad flag (zero extra effort) |
| `fair` | **on** | **autotuned** | value of the one documented autotune step |

`graphs-only` is the "vLLM without the auto-tuner" point. The `crippled → graphs-only`
gap is what `--enforce-eager` cost; the `graphs-only → fair` gap is what the MoE
autotuner is worth. Pie is the fourth line on the same plot.

**Expected outcome (be honest up front):** on A100 the decode gap may largely
close — that's the point of the experiment. Pie's clean, config-independent wins
(95.8% prefill reuse via explicit persistent sessions, fork/branch semantics,
accuracy) don't depend on decode speed.

## Feasibility notes settled before scripting

- **Pie native driver runs on A100 (sm_80).** The CUDA build auto-detects arch
  (`driver/cuda/cmake/DetectCudaArchitecture.cmake`); `moe_dispatch.cu` has
  `__CUDA_ARCH__ >= 800` paths and the weight loader uses fast sm_80 cuBLASLt
  kernels. **Rebuild on the box — do not copy the sm_120 Blackwell binary.**
- **Caveat that could flip fairness the other way:** Pie's fused kernels are
  hand-written, not per-GPU autotuned. On A100 they're "compiled for sm_80," not
  necessarily profiled for it. If Pie's decode looks weak vs a now-tuned vLLM,
  that's an honest result, not vLLM cheating.

## Files

| File | Role |
|---|---|
| `AGENT_HANDOVER.md` | **for an agent taking over a provisioned pod**: machine state, the storage rule, run order, version pins and why, what to record, what to decide vs escalate |
| `bootstrap_runpod.sh` | **start here on a bare pod**: preflight, toolchain, python 3.12, clone, NCCL, one shared env file, venvs — then hands off to `00_setup_a100.sh`. Idempotent/resumable. |
| `00_setup_a100.sh` | one-time: rebuild Pie for sm_80, build coder-session wasm, vLLM venv, harness venv, model download. **Read its LOGISTICS block first.** |
| `10_vllm_serve_fair.sh` | launch vLLM at `VLLM_TIER`; asserts the banner matches the tier before serving |
| `11_autotune_moe.sh` | run vLLM's `benchmark_moe.py` once to generate the tuned MoE config (the `fair` tier) |
| `assert_vllm_fair.sh` | tier-aware banner check; paste its output into the writeup |
| `20_decode_microbench.py` | identical-measurement decode tok/s (fixes the 85.5-GPU-only vs 64.2-e2e bug) — same client codepath for both engines |
| `pie_cuda_native_config_30b_moe_a100.toml` | Pie native config for A100 80GB, swap pool on |
| `run_litellm_baseline_fair.sh` | fair baseline arm (boot vLLM at tier + gate + run harness) |
| `30_ab_run.sh` | orchestrate one arm (`ARM=pie` / `ARM=litellm` + `VLLM_TIER=...`) |
| `summarize_ab.py` | totals + **s/iter** + median per-call latency across arms (the trajectory-robust numbers) |

## Run order

```bash
# 0. one-time, on a bare pod: provisions the box, clones this ref, then runs
#    00_setup_a100.sh itself. Run it under tmux — the CUDA build is 30-60 min.
#    (If the box is already provisioned, skip straight to 00_setup_a100.sh and
#     read its LOGISTICS block — git ref w/ §4+bugC fixes, disk, GPU.)
bash bootstrap_runpod.sh
source /workspace/pie-bench-env.sh   # every later shell needs this

# 1. generate the tuned MoE config once (only needed for the `fair` tier)
bash 11_autotune_moe.sh

# 2. Pie arm (native CUDA driver, A100 config)
ARM=pie bash 30_ab_run.sh

# 3. vLLM baseline at all three effort tiers
VLLM_TIER=crippled    ARM=litellm bash 30_ab_run.sh
VLLM_TIER=graphs-only ARM=litellm bash 30_ab_run.sh
VLLM_TIER=fair        ARM=litellm bash 30_ab_run.sh

# 4. the effort-axis table (s/iter + median latency, shared-instance head-to-head)
python summarize_ab.py \
  pie=../predictions/ab_a100_pie_*.jsonl \
  litellm-crippled=../predictions/ab_a100_litellm_crippled_*.jsonl \
  litellm-graphs-only=../predictions/ab_a100_litellm_graphs-only_*.jsonl \
  litellm-fair=../predictions/ab_a100_litellm_fair_*.jsonl
# NB ../predictions — 30_ab_run.sh cd's to integrations/openhands before
# writing, so the jsonl lands one level above this runpod/ directory.

# (optional) clean decode microbench, both engines measured identically
python 20_decode_microbench.py --base-url http://localhost:18000/v1 \
       --model Qwen/Qwen3-Coder-30B-A3B-Instruct --label vllm-fair
```

## Reporting rules (carry into the writeup)

1. **Report s/iter and median per-call latency, not raw wall time** — trajectories
   diverge; raw wall conflates path length with serving speed. Both are recorded
   identically by the harness (`_metadata.wall_clock_s / agent_iterations /
   response_latencies`).
2. **Paste each tier's banner** (`assert_vllm_fair.sh` output) so the config is
   auditable — symmetric to how the current writeup quotes the crippled banner.
3. **Measure decode the same way on both engines** — the microbench uses one
   client codepath; never compare Pie-GPU-only against vLLM-end-to-end again.
4. **Keep the clean claims separate from the decode claim.** Prefill reuse,
   fork/branch semantics, and accuracy stand regardless of how the decode tiers
   land.

## Stretch: the only regime that structurally differs

Overcommit sweep — Pie **swap on** (`pie_cuda_native_config_30b_moe_a100.toml`,
`swap_pool_size=8192`) vs vLLM evict-and-recompute — pushing the working set past
KV capacity (many concurrent sessions or contexts larger than cache). This is the
one place the architectures diverge in kind, not degree, and the current writeup
lists it as hypothesis-only (Pie's swap was off there). Not required for the
parity claim; highest-value if the parity run lands near-even.
