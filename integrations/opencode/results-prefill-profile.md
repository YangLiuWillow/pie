# Where pie's time goes on Metal — prefill profile, 2026-08-12

*Machine: `Lius-MacBook-Pro`, 48 GB, Metal. Both stacks serve the same
`mlx-community` checkpoints, one server at a time. Tool:
`integrations/opencode/profile_prefill.py`.*

**All numbers below are at MATCHED config** (see the parity table). An earlier
revision of this document measured pie in a KV-starved configuration and reached
two conclusions that did not survive matching — both are corrected at the end
rather than deleted, because the way they failed is the transferable part.

## Config parity — do this before quoting any ratio

Read what each stack *actually booted with*, not what you asked for.

| knob | vLLM-metal | pie (now) | pie (before) |
|---|---|---|---|
| checkpoint | `mlx-community` 4-bit | same artifact | same |
| compute dtype | `torch.bfloat16` | `bfloat16` | ✓ |
| `max_model_len` | 16384 | 16384 | ✓ |
| graph capture | `enforce_eager=False` | n/a on Metal | ✓ not crippled |
| prefill chunk | `max_num_batched_tokens=2048` | `max_forward_tokens=2048` | ✗ 1024 |
| KV block | `block_size=16` | `kv_page_size=32` | (immaterial — tested) |
| **KV pool** | **193,536 tokens** | **65,536 tokens** | ✗ **16,384** |

`total_pages = 512` was this repo's inherited default and it is **KV-starved**:
raising it to 2048 is worth **1.73× on prefill for free**. It was never
memory-forced — pie's activation pool sat at 24 MB of a 1024 MB budget.
`run_pie_opencode.sh` now generates the matched values by default
(`PIE_TOTAL_PAGES`, `PIE_MAX_FORWARD_TOKENS`).

## Method: fit a line, don't divide an aggregate

"pie does 186 tok/s, vLLM does 1183" cannot localize a bottleneck: it folds
fixed per-call cost together with per-token compute, and those are fixed by
unrelated work. So the profiler sweeps prompt length and reads the **slope**:

```
ttfc(n) = fixed_cost + n / marginal_rate
```

Read the **per-segment marginals**, not the fitted intercept — on a convex curve
least-squares drives the intercept negative and it means nothing.

## Result

Marginal prefill rate per context segment, matched config:

| | **Coder-30B (MoE)** | | | **Qwen3-0.6B (dense)** | | |
|---|---:|---:|---:|---:|---:|---:|
| segment | pie | vLLM | gap | pie | vLLM | gap |
| ~470→1310 | 692 | 2491 | 3.6× | 3221 | 11552 | 3.6× |
| ~1310→2560 | 449 | 1662 | 3.7× | 1868 | 7974 | 4.3× |
| ~2560→3810 | 335 | 1511 | 4.5× | 1339 | 6145 | 4.6× |
| ~3810→5060 | 256 | 1135 | 4.4× | 983 | 4740 | 4.8× |
| **mean gap** | | | **4.1×** | | | **4.3×** |

Fixed per-call cost: **pie ~0 ms, vLLM ~117 ms.**

## The bottleneck

**A uniform ~4× per-token deficit in the Metal forward pass.** The striking
thing is how *flat* it is across everything that could have explained it:

- **Same gap dense and MoE** (4.3× vs 4.1×). A 0.6B dense model has no expert
  routing, no grouped GEMM, and a trivial working set — and the gap is
  identical. **The MoE path is not the problem.**
- **Same gap across 50× of model size** (0.6B vs 30B).
- **Mildly worse with context** on both stacks (3.6× → 4.4×/4.8×), so pie's
  attention scaling is slightly behind but is not the story.

Ruled out by measurement:

- **Not per-call overhead.** pie ~0 ms against vLLM's ~117 ms — **pie wins that
  axis outright.** Nothing about wasm instantiation, rendering, address hashing
  or dispatch is where the time goes; optimising there recovers nothing.
- **Not prefill chunk size.** 1024 → 2048 → 4096 moves mid-length prefill ~18%
  and converges. Worth taking; not the gap. The OpenHands CUDA lever (12.0k →
  24.1k tok/s via "chunk 2048 + 64-row tiles") **does not transfer at that
  magnitude to Metal.**
- **Not KV page size.** 16 vs 32 is immaterial.
- **Not MoE.** See above — this is the conclusion that reversed on matching.

So it is the general per-token GEMM/attention path in the Metal driver, and it
is a driver task rather than an integration one.

## End-to-end, matched (Coder-30B, 6-turn agentic replay)

| turn | pie A | pie B | vLLM |
|---:|---:|---:|---:|
| 1 (cold) | 37.13 s | 26.54 s | 6.76 s |
| 2–6 (steady) | 27–43 s | ~4.0 s | ~0.67 s |
| **total** | **203.44 s** | **46.41 s** | **10.12 s** |

Comparable metrics (the totals are *not* comparable — pie generated 96 tokens
per turn, vLLM 11):

| | pie B | vLLM | ratio |
|---|---:|---:|---|
| cold prefill | 296 tok/s | 1216 tok/s | 4.1× |
| steady-state ttfc | 1.90 s | 0.46 s | 4.1× |
| steady-state decode | 46.2 tok/s | 51.7 tok/s | **1.12× (near parity)** |
| prefix reuse | ~97.5% cached | 78.9% hit | both work |

Strategy B over Strategy A, matched: **4.4×**. Matching the config was itself
worth 1.50× on arm A and 1.31× on arm B.

**Decode is at near-parity; prefill is 4.1× behind.** That is the whole story in
one line.

## Cross-check against CUDA

`openhands-integration-updated:integrations/openhands/runpod`, same model, H100,
fairly configured vLLM:

| | pie | vLLM | gap |
|---|---:|---:|---|
| prefill marginal (tuned) | 24.1k tok/s | 44.9k tok/s | **1.86×** |
| c1 in-call tok/s | 147.4 | 160.4 | 1.09× |
| c8 s/iter, matched instances | 4.26–4.68 | 3.90–4.68 | 1.00–1.08× |

**The prefill gap is not a Metal artifact — it is 1.86× on tuned CUDA.** Metal
widens it to ~4×. Consistent direction across two backends and two harnesses.

## Two conclusions that reversed, and why

Both came from measuring pie starved while vLLM was not. Recorded because the
failure mode is more useful than the numbers.

1. **"The gap is a flat ~6× multiplier."** It was 6.4× starved and is 4.1×
   matched — and matched it *grows* with context rather than being flat. The
   flatness was an artifact of a bottleneck that dominated everything else.
2. **"MoE adds 1.44× on top of a 4.4× dense baseline."** Matched, MoE adds
   **0.94× — nothing.** The starved pool penalised the 30B's large KV footprint
   far more than the 0.6B's small one, manufacturing an architecture-shaped
   difference out of a memory-sizing one. This one had a plausible mechanism
   (OpenHands found MoE GEMM tactics were *their* main CUDA prefill lever), which
   is exactly why it was worth re-deriving rather than believing.

Also corrected: this repo cited OpenHands measuring pie **~26% faster** than
litellm+vLLM. That baseline was deliberately crippled (`--enforce-eager` + an
untuned MoE kernel); their fair rerun puts pie at **parity to ~9% behind**.

## Three cache traps, none of which announced itself

Every one was caught by a number being *implausible*, not by anything failing:

1. **Bench warm-up primed the prefix cache** by sending turn 1's own messages —
   converting turn 1 from a cold prefill into a hit, and only on arms that *have*
   a cache. Read as 0.338 s ttfc, ~22k tok/s.
2. **The profiler's own sweep was nested prefixes.** Building longer prompts by
   repeating one filler makes every prompt a literal prefix of the longer ones.
   Read as **80,543 tok/s marginal on a 30B laptop**. Fixed with a *leading*
   nonce — a trailing one leaves the shared part in front.
3. **`cached_tokens` is unpopulated by vllm-metal** — it reports 0 while its own
   logger reports 78.9% hits. Trusting the usage field would have concluded
   "vLLM has no prefix caching" and flattered pie by ~4×.

## What to do with this

1. **Take the config fix everywhere.** `total_pages = 2048` is free and worth
   1.5× end-to-end. Caveat from the CUDA work: the chunk lever *inverts* under
   concurrency (chunk 2048 at c8 shrank the pool and collapsed), so this is a c1
   recommendation.
2. **Stop optimising this integration for prefill wins.** Strategy B's 4.4×
   against pie's own baseline does not close a 4.1× kernel gap. The session
   work's remaining value is what APC structurally cannot express — B-2 in-place
   context editing, B-3 subagent forking.
3. **If the gap is worth closing it is a driver task**, and the dense/MoE
   equivalence says start at the general per-token path, not the expert GEMMs.

## Reproduce

```sh
python3 integrations/opencode/profile_prefill.py \
  --base-url http://127.0.0.1:8080 --label pie --model coder30b \
  --repeat 2 --words 400,1200,2400,3600,4800
```
