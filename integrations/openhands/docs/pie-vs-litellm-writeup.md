# Gluing Pie into OpenHands: what actually gets faster, and why

*A measured comparison between a Pie-backed OpenHands agent and the vanilla
OpenHands → vLLM setup that the OpenHands docs ship, on SWE-bench, with the same
open-source model — including an honest accounting of what's a real engine
property and what's a benchmarking artifact.*

---

## TL;DR

We ran the exact same OpenHands coding agent two ways on SWE-bench Verified
issues, changing **only the thing that serves the tokens**:

- **Baseline** — vanilla OpenHands talking to a `vllm serve` endpoint over
  `litellm` (`base_url=http://localhost:18000/v1`, `--enable-prefix-caching`).
  This is the SDK's own first-class path for a self-hosted open-source model.
- **Pie** — the same agent, but its LLM calls are routed through a thin adapter
  into a persistent **Pie session** that keeps the KV cache resident on the
  server between agent steps.

Same model (`Qwen3-Coder-30B-A3B-Instruct`), same temperature (0.0), same
prompts, same tools, same condenser, one GPU, serial execution.

| Metric | Baseline (vLLM + litellm) | Pie-glued | Delta |
|---|---|---|---|
| **Wall time, 13 shared instances** | 1,939 s | 1,430 s | **−26%** |
| **Wall time per agent iteration** | 2.15 s | 1.67 s | **−23%** |
| **Prompt tokens actually prefilled** | (not exposed) | 4.2% of 12.1 M | **95.8% served from KV reuse** |
| **Accuracy, neutral 50-instance set** | 13 / 50 | **18 / 50** | **+5** |

**Read the "Fairness" section before quoting the timing numbers.** The vLLM
baseline was run with CUDA graphs disabled and an untuned MoE kernel, which
handicaps its decode. The token-reuse and accuracy numbers are clean; the
wall-clock gap is real but partly a configuration artifact, not a pure
engine-vs-engine result.

---

## The setup

The two arms are identical above the LLM boundary. Both load the same agent:

```python
model = "Qwen3-Coder-30B-A3B-Instruct"
temperature = 0.0
tools = ["terminal", "file_editor", "task_tracker"]  # + FinishTool, ThinkTool
condenser = LLMSummarizingCondenser(max_size=240, keep_first=2)
tool_concurrency_limit = 1
```

All of that — the agent, the tools, the condenser, the 8-phase system prompt —
is **stock upstream OpenHands**. The tools live in `openhands/tools/*/definition.py`
and `openhands/sdk/tool/builtins/`; the Pie integration adds no tools of its own.
The only thing Pie swaps is the transport underneath `openhands.sdk.LLM`:

```
Baseline:   OpenHands ──litellm.completion()──HTTP/JSON──▶ vllm serve (APC on)  ──▶ GPU
Pie:        OpenHands ──PieLLM._transport_call()──▶ Pie session (KV resident)   ──▶ GPU
```

`litellm` is the SDK's *only* built-in transport — the base `LLM` class docstring
says it interacts with models "through the litellm library," and `base_url` +
an `openai/<model>` id pointed at a local vLLM server is exactly the path the SDK
documents for self-hosted models. Pie subclasses `LLM` and overrides the one
method (`_transport_call`) where the parent would otherwise call
`litellm.completion(...)`. Everything else — message formatting, tool-call
parsing, retries, telemetry — is untouched.

An agentic SWE-bench trajectory is a punishing workload for a serving stack: the
agent takes tens of steps per issue, and **every step re-sends the entire growing
conversation** and asks the model to append one more turn. Where those tokens go,
and how many get recomputed, is the whole game.

Both engines do prefix caching, so the interesting question is *not* "caching vs
no caching."

---

## Result 1: the 13-instance A/B (timing + resolution)

Same 13 issues, both arms. `✓`/`✗` = whether that arm's patch actually resolved
the issue under the SWE-bench test harness:

| Instance | Base s | Base it | Base? | Pie s | Pie it | Pie? |
|---|---:|---:|:--:|---:|---:|:--:|
| django-12276 | 121.1 | 78 | ✓ | 99.9 | 58 | ✓ |
| django-13028 | 246.7 | 88 | ✓ | 140.6 | 68 | ✓ |
| django-13089 | 120.1 | 66 | ✓ | 69.9 | 46 | ✓ |
| django-14373 | 94.4 | 40 | ✓ | 74.2 | 36 | ✓ |
| django-15569 | 72.4 | 52 | ✓ | 69.0 | 48 | ✓ |
| django-16485 | 121.3 | 48 | ✓ | 136.6 | 66 | ✓ |
| matplotlib-22719 | 181.5 | 88 | ✓ | 112.1 | 66 | ✓ |
| xarray-4075 | 143.3 | 78 | ✓ | 122.4 | 76 | ✓ |
| xarray-4966 | 114.4 | 68 | ✓ | 140.8 | 88 | ✗ |
| sklearn-10908 | 197.9 | 112 | ✓ | 187.8 | 128 | ✗ |
| sklearn-12973 | 283.7 | 148 | ✓ | 74.7 | 56 | ✓ |
| sklearn-13496 | 152.1 | 104 | ✓ | 101.2 | 64 | ✓ |
| sympy-19346 | 90.0 | 40 | ✓ | 101.2 | 58 | ✓ |
| **Total** | **1,938.9** | **900** | **13/13** | **1,430.4** | **858** | **11/13** |

Two things to say honestly about this table:

**On accuracy:** this 13-set *is* the baseline's own previously-resolved issues,
so its ceiling is parity — it can only show whether Pie *reproduces* the
baseline's wins. 11/13 means "no meaningful regression," nothing more. (For a
signal that can actually move, see the neutral-50 result below.)

**On timing — mind the confound:** the two arms did not walk the same trajectory.
sklearn-12973 took the baseline 148 agent steps and Pie 56. That's not the engine
being 3.8× faster; it's two stochastic agents finding different paths. To divide
out path length, normalize to **wall time per agent iteration** (same harness,
same iteration counter, both arms):

```
Baseline:  1,938.9 s / 900 iters = 2.15 s/iter
Pie:       1,430.4 s / 858 iters = 1.67 s/iter   →  ~23% less per iteration
```

The aggregate iteration counts are close (900 vs 858, ~5% apart), so at the
suite level the divergence roughly washes out and the ~26% wall gap is *mostly*
the serving layer, not Pie getting lucky with shorter paths. Per-*instance* it's
noisy; in aggregate it's a real serving-layer difference — whose sources we now
decompose, and whose fairness we then question.

---

## Result 2: where the per-iteration difference comes from

Both engines cache prefixes, so the win is *not* "Pie caches and vLLM doesn't."
It has two real contributions plus one large asterisk.

### (a) Prefill avoided — huge token savings, modest wall-clock effect

Pie keeps the KV cache resident in a **persistent session** per instance. Every
step after the first runs in `extended` mode: it appends the new tokens (latest
tool output + the model's previous turn) onto the KV already on the GPU and
prefills **only that delta**. Exactly one step per instance runs in `rebuilt`
mode (the cold initial prefill). Measured across the 13 instances:

| | Prompt tokens seen (rendered) | Actually prefilled | Reused from KV |
|---|---:|---:|---:|
| **Total** | 12,102,716 | 511,626 | **95.8%** |

That's a **23.6× reduction in prefill compute**, with **zero KV-verify errors**
(the reused cache was always numerically identical to a from-scratch render — the
inferlet asserts this every call when `kv_verify` is on).

**But be honest about the wall-clock impact.** vLLM's Automatic Prefix Caching is
*also* on in the baseline, and in this serial single-session regime its hit rate
is also very high. So both engines skip most prefill; the dramatic *token*
savings are real (they're the right metric for prefill FLOPs / energy / cost),
but they are **not** the main source of the wall-clock gap, because the baseline
wasn't paying full prefill either. Where Pie's session still helps the clock:
even on an APC hit, the vanilla path re-serializes and re-ships the **entire**
conversation each step, and vLLM must re-tokenize and re-hash the full (growing)
prompt just to *locate* the cached prefix — CPU work proportional to full context
on every call. Pie ships only the delta.

### (b) Decode throughput — where the wall-clock gap mostly lives

An agent step spends most of its wall time **decoding** the reply (the prompt is
mostly cached), so decode speed dominates the per-iteration number. In a
decode-only microbenchmark on this build, Pie's native CUDA driver — which
hand-fuses the Qwen3-MoE `gate_up_proj`/`down_proj` experts into 3-D tensors —
sustained **85.5 tok/s** of pure GPU decode (277 generated tokens in 3.24 s at
batch 1).

This is the headline that needs the asterisk. Read the next section before
believing "Pie's engine decodes faster."

### Fairness: the vLLM baseline was configured badly for decode

When we read the actual `vllm serve` logs, the baseline was **not** given its
best decode configuration. Two specific problems, both visible in the engine
banner:

1. **`enforce_eager: True` — CUDA graphs OFF.** The launch explicitly passed
   `--enforce-eager`, and vLLM logged *"Enforce eager set, overriding
   optimization level to -O0"* and *"Cudagraph is disabled under eager mode."*
   CUDA graphs are one of the single largest decode optimizations in vLLM: at
   batch-1 decode, per-layer kernel-launch overhead dominates, and graphs are
   exactly what eliminate it. Running eager is close to a worst case for decode
   latency.

2. **Untuned MoE kernel.** vLLM warned: *"Using default MoE config. Performance
   might be sub-optimal! Config file not found …
   NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json."* There is no tuned
   fused-MoE Triton config for this Blackwell GPU, so it fell back to an untuned
   default — another decode-throughput penalty specific to the MoE layers.

So the decode comparison is **not** apples-to-apples in two ways at once:
- The **configurations differ**: Pie ran its optimized native kernels; vLLM ran
  eager with untuned MoE. A properly-configured vLLM (drop `--enforce-eager` so
  CUDA graphs capture, and generate a tuned MoE config for the Blackwell card)
  would very likely close most or all of the decode gap. We have not yet run
  that head-to-head.
- The **numbers aren't measured the same way**: the 85.5 tok/s is Pie's *GPU-only*
  decode; the 64.2 tok/s we sometimes quote for the baseline is an *end-to-end*
  per-call rate that includes HTTP, scheduling, and tokenization. vLLM's own
  logged generation throughput in-run was lower still (~35 tok/s at points),
  consistent with eager mode.

**Honest conclusion for Result 2:** the wall-clock win is real and reproducible,
but the largest single contributor — decode speed — is substantially a
**vLLM-configuration artifact**, not a demonstrated fundamental engine advantage.
The cleanly-defensible Pie property here is the **95.8% prefill reuse via explicit
persistent sessions**; the decode-speed claim should be treated as "un-tuned vLLM
vs tuned Pie" until we rerun vLLM with CUDA graphs and a tuned MoE kernel.

---

## Result 3: accuracy on a neutral set

On a neutral 50-instance set — chosen *without* reference to either arm's results,
so it can move in both directions — Pie resolved **18/50** vs the baseline's
**13/50**:

```
both resolved:  10     only Pie:  8     only baseline:  3
```

The +5 is a genuine signal (this set's ceiling is not parity). We inspected the 3
Pie losses (xarray-4966, sklearn-10908, sympy-19346): all three are ordinary
model-reasoning misses on Pie's trajectory — right file, right function,
incomplete fix (a too-narrow type guard, a bypassed-but-not-populated attribute, a
wrong `repr` format) — **not** engine, caching, or patch-application failures.
Symmetric with the 8 Pie-only wins: same model, temperature 0, trajectories
diverge, wins and losses trade off.

---

## When Pie does *not* win — and the overcommit question

The most important section, because "26% faster" is not the whole truth.

**The linear / unpressured regime.** When the KV cache comfortably fits and there
is a single hot session, vLLM's APC is near-perfect and decode dominates. There
the prefill-savings axis collapses to almost nothing and the gap reduces to
whatever raw decode difference exists — which, per the Fairness section, is mostly
about how each engine was configured. Matched-KV, single-GPU, fits-in-cache: a
well-tuned vLLM and Pie should be close.

**Memory overcommit — the regime where the architectures genuinely differ.** Push
the working set past KV capacity (many concurrent sessions, or contexts larger
than the cache) and the two designs diverge in kind, not degree:

- **vLLM + APC** degrades by **evict-and-recompute**: on a prefix-cache miss it
  drops KV blocks and *recomputes* them by re-prefilling. Cost is paid in GPU
  FLOPs (redo the prefill), and it's robust — it never OOMs from this, it just
  gets slower.
- **Pie** is designed to degrade by **swap**: cold KV pages spill to a host-side
  swap pool and are paged back on demand, so a suspended session's KV survives
  without recomputation. Cost is paid in PCIe bandwidth (move KV host↔device)
  rather than recompute.

Which is better is workload-dependent and **we have not measured it head-to-head
yet** — importantly, *in this very A/B Pie's swap pool was disabled*
(`swap_pool=0 pages` in the driver log), so Pie had no offload advantage here and
would itself OOM under real overcommit. The hypothesis worth testing: when the
reused prefix is large and recomputation is expensive (long shared context,
expensive prefill), **swap should beat recompute** — Pie moves bytes while vLLM
redoes math. When the prefix is small or host bandwidth is the bottleneck,
**recompute should beat swap**. Turning on Pie's native swap pool and running the
overcommit sweep against vLLM's evict-and-recompute is the experiment that would
actually settle where Pie's structural advantage pays off. Until then, Pie's
in-regime edge is: explicit persistent sessions (deterministic KV retention, no
eviction risk), 95.8% prefill reuse, and fork/branch semantics litellm cannot
express — not a proven overcommit win.

---

## Reproduction notes

- Model `Qwen/Qwen3-Coder-30B-A3B-Instruct`, temperature 0.0, single GPU
  (RTX PRO 6000 Blackwell), serial.
- Baseline: `vllm serve` v0.16.0 on `:18000`, V1 engine, FLASH_ATTN, Triton MoE,
  `--enable-prefix-caching`, **`--enforce-eager` (CUDA graphs off)**, untuned MoE
  config; driven by litellm with `model=openai/…` + `base_url`.
- Pie: native CUDA driver (`pie-driver-cuda`, qwen3_moe, fused experts), one
  persistent session per instance, `swap_pool=0`.
- Timing A/B: baseline job `19212031`, Pie job `19251912`, same 13 instances.
- Neutral accuracy: `neutral_50_pie` (18/50) vs `baseline_t0_full_50` (13/50),
  Apptainer SWE-bench harness, job `19500867`.
- Prefill/reuse: `session["prefill_tokens"]` vs `session["len"]` reported by the
  Pie inferlet per call, recorded in `PieLLM._record_session_response` and
  aggregated in `pie_session_summary()`; logged per instance by `swe_bench.py`.

*Caveats worth repeating: (1) the arms take different trajectories — normalize by
iteration for the serving-layer claim; (2) the decode-speed gap is confounded by
an eager-mode, untuned-MoE vLLM baseline and is not a clean engine comparison;
(3) accuracy deltas are on ≤50 instances — directional, not a leaderboard; (4) the
overcommit advantage is a hypothesis, not yet measured, and Pie's swap was off
here.*
