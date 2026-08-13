# Where pie's time goes on Metal — prefill profile, 2026-08-12

*Machine: `Lius-MacBook-Pro`, 48 GB, Metal. Both stacks serve the same
`mlx-community` checkpoints, one server at a time. Tool:
`integrations/opencode/profile_prefill.py`.*

## Method: fit a line, don't divide an aggregate

"pie does 186 tok/s, vLLM does 1183" is a ratio of aggregates, and it cannot
point at a bottleneck: it folds per-call overhead (fixed, doesn't scale with
prompt length) together with forward-pass compute (scales). Those are fixed by
unrelated work. So the profiler sweeps prompt length and reads the **slope**:

```
ttfc(n) = fixed_cost + n / marginal_rate
```

This is the OpenHands evaluation's first method lesson — "measure the layer,
don't divide the aggregate" — which is how three wrong conclusions got published
there before being retracted.

**Two cache traps had to be closed before any of these numbers meant anything**,
and neither announced itself:

1. The bench warm-up sent turn 1's own messages, priming a prefix cache and
   turning turn 1 from a cold prefill into a hit — but only on arms that *have*
   a cache.
2. The profiler's own sweep built longer prompts by repeating one filler, so
   every prompt was a literal **prefix** of the longer ones. A caching backend
   answered each point almost entirely from cache. That read as
   **80,543 tok/s marginal on a 30B MoE laptop** — impossible by two orders of
   magnitude. Fixed by leading each prompt with a unique nonce; a trailing
   nonce would not work, because the shared part would still be the prefix.

Both were caught by numbers being *implausible*, not by anything failing.

> ## Correction, same day: the first version of this profile was not config-matched
>
> It compared pie at `total_pages = 512` (16,384 KV tokens) against vLLM at
> **193,536** KV tokens — a 12× asymmetry inherited from this repo's
> memory-constrained profile and never checked against the baseline. Raising
> pie to `total_pages = 2048` is worth **1.73× on prefill, for free**, and moves
> the gap from 6.4× to **3.9×**.
>
> The headline below ("flat ~6×") is therefore **superseded**: it measured a
> starved pie. Matched, the gap is 3.9× mean and it **grows with context**
> (3.6× → 4.4×) rather than being flat. Kept rather than deleted, because
> "measure the layer" does not help if the two layers are configured
> differently — a fair-parity check belongs beside every ratio.
>
> `kv_page_size` 16 vs 32 was tested and makes no material difference; the KV
> pool size is the knob.
>
> | segment | pie 512p | pie 2048p | speedup | vLLM | gap |
> |---|---:|---:|---:|---:|---:|
> | 444→1284 | 326 | 692 | 2.12× | 2491 | 3.6× |
> | 1284→2532 | 255 | 449 | 1.76× | 1662 | 3.7× |
> | 2532→3780 | 243 | 335 | 1.38× | 1511 | 4.5× |
> | 3780→5028 | 179 | 256 | 1.43× | 1135 | 4.4× |
> | **mean** | **251** | **433** | **1.73×** | **1700** | **3.9×** |
>
> **Still standing after the correction:** the bottleneck is the forward pass,
> not per-call overhead (pie ~0 ms vs vLLM ~117 ms); chunk size is a trim, not
> the gap; and the gap is real on tuned CUDA too (1.86×).
>
> **Needs re-deriving at matched config:** the dense-vs-MoE decomposition below
> (4.4× dense / 6.4× MoE → "MoE adds 1.44×"). Both pie arms there ran at
> `total_pages = 512`, and a 0.6B dense model has a far smaller KV footprint, so
> the two were not starved equally. Do not quote the 1.44× until it is re-run.

## Config parity — check this before any ratio

| knob | vLLM-metal | pie (original profile) | matched? |
|---|---|---|---|
| checkpoint | `mlx-community` 4-bit | same artifact | ✓ |
| compute dtype | `torch.bfloat16` | `bfloat16` | ✓ |
| `max_model_len` | 16384 | 16384 | ✓ |
| graph capture | `enforce_eager=False` | n/a on Metal | ✓ (not crippled) |
| prefill chunk | `max_num_batched_tokens=2048` | `max_forward_tokens=1024` | ✗ half |
| KV block | `block_size=16` | `kv_page_size=32` | ✗ (immaterial) |
| **KV pool** | **193,536 tokens** | **16,384 tokens** | ✗ **12× smaller** |

pie's activation pool was 24 MB of a 1024 MB budget at chunk 1024, so nothing
about the small chunk or the small pool was memory-forced — both were inherited
defaults.

## Result (original, un-matched — superseded by the correction above)

Marginal prefill rate, per context segment:

| | **Coder-30B (MoE)** | | | **Qwen3-0.6B (dense)** | | |
|---|---:|---:|---:|---:|---:|---:|
| segment | pie | vLLM | gap | pie | vLLM | gap |
| 444→1284 | 326 | 2491 | 7.6× | 3010 | 11552 | 3.8× |
| 1284→2532 | 310 | 1662 | 5.4× | 1893 | 7974 | 4.2× |
| 2532→3780 | 245 | 1511 | 6.2× | 1314 | 6145 | 4.7× |
| 3780→5028 | 179 | 1135 | 6.3× | 960 | 4740 | 4.9× |
| **mean gap** | | | **6.4×** | | | **4.4×** |

Fixed per-call cost: **pie ~0 ms, vLLM ~117 ms.**

## What the bottleneck is — and what it is not

**It is raw forward-pass throughput in the Metal driver.** The gap is roughly
flat across context lengths (7.6/5.4/6.2/6.3), so it is a multiplier on compute,
not a scaling problem that appears at length.

Ruled out, each by measurement:

- **Not per-call overhead.** pie's fixed cost is ~0 ms against vLLM's ~117 ms —
  **pie wins this axis outright.** Nothing about wasm instantiation, rendering,
  address hashing or dispatch is the problem. Optimising there would recover
  nothing.
- **Not attention scaling.** Both stacks decay with context at comparable
  relative rates (pie 326→179, 1.8×; vLLM 2491→1135, 2.2×). pie's decay is if
  anything *shallower* on the MoE model.
- **Not prefill chunk size.** Sweeping `max_forward_tokens` 1024 → 2048 → 4096
  moved mid-length prefill by ~18% and converged at long context. Worth taking
  (4096 beat 1024 at every length ≥2532), but it is a trim, not the gap. The
  OpenHands CUDA lever — 12.0k → 24.1k tok/s via "chunk 2048 + 64-row tiles" —
  **does not transfer at that magnitude to Metal.**
- **Not MoE alone.** The MoE path adds **1.44×** on top of a **4.4× baseline gap
  that is already present on a small dense model**. So the expert GEMM is a real
  secondary factor, but ~70% of the gap exists before MoE enters the picture.

**The single most useful number here is the dense 4.4×.** A 0.6B dense model has
no MoE routing, no expert GEMM, and a trivial working set — and pie is still
4.4× behind on the same hardware and checkpoint. That localizes the bulk of the
gap to general Metal forward-pass efficiency (GEMM kernels, dispatch batching,
activation handling), not to anything specific to MoE, to sessions, or to the
inferlet machinery.

## Cross-check against CUDA

The `openhands-integration-updated:integrations/openhands/runpod` handovers
measured the same model on an H100 with a *fairly configured* vLLM:

| | pie | vLLM | gap |
|---|---:|---:|---|
| prefill marginal (H100, tuned) | 24.1k tok/s | 44.9k tok/s | **1.86×** |
| c1 in-call tok/s | 147.4 | 160.4 | 1.09× |
| c8 s/iter, matched instances | 4.26–4.68 | 3.90–4.68 | 1.00–1.08× |

So **the prefill gap is not a Metal artifact — it exists on tuned CUDA at
1.86×.** Metal widens it to 4.4–6.4×; it does not create it. The direction of
the finding is consistent across two backends and two independent harnesses.

### Correction to the record

Earlier notes in this repo (including `results-pie-vs-vllm-metal.md` as first
written) cited OpenHands as measuring **pie ~26% faster than litellm+vLLM**. That
number came from a baseline the runpod handover describes as **crippled**:
`--enforce-eager` (CUDA graphs off) plus an untuned MoE kernel. Their own rerun
existed specifically to break it — *"we are trying to break our own result — a
defensible finding beats a flattering one."* Fairly configured, pie is at
**parity to ~9% behind**, not 26% ahead. Any argument resting on the 26% figure
should be re-derived.

## What to do with this

1. **Stop optimising the integration for prefill wins.** Strategy B's ~13×
   steady-state improvement is real against pie's own baseline and does not close
   a 6.4× kernel gap. The session work's remaining value is the things vLLM's APC
   *cannot* express — in-place context editing (B-2), subagent forking (B-3) —
   not raw prefill.
2. **Take the chunk-size trim anyway**: `max_forward_tokens = 4096` beat 1024 at
   every length ≥2532 on this workload, at no cost at short lengths. Cheap.
   (Caveat from the CUDA work: the same lever *inverts* under concurrency —
   chunk 2048 at c8 shrank the pool and collapsed. This is a c1 recommendation.)
3. **If the gap is worth closing, it is a driver task, not an integration task**,
   and the dense 4.4× says start with the general GEMM path rather than the MoE
   experts.

## Reproduce

```sh
python3 integrations/opencode/profile_prefill.py \
  --base-url http://127.0.0.1:8080 --label pie --model coder30b \
  --repeat 2 --words 400,1200,2400,3600,4800
```

Read the per-segment marginals, not the fitted intercept: on a convex curve the
least-squares intercept goes negative and means nothing. Derive the fixed cost
from the shortest point against the first segment's marginal instead.
