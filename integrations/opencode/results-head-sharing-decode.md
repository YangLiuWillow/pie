# Decode attention: one KV read per GQA group — 2026-08-15

**Machine:** Apple M5 Pro, 48 GB, idle (streaming roof 294.5–297.9 GB/s).
**Model:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` — 32 query heads
over 4 KV heads, head_dim 128, page size 32.

## The defect

`sdpa_paged_decode` launches **one threadgroup per query head**, and each one
walks the whole KV cache for its own KV head. With `gqa_factor = 8`, eight
threadgroups read the same K and V — an eightfold amplification on the one
thing a decode step is bandwidth-bound on.

Measured on the serving workload rather than a synthetic one.
`PIE_METAL_DISPATCH_TRACE` on a real 23.6k-token prefill-then-decode:

| | share of a decode fire |
|---|---:|
| `sdpa_paged_decode_bfloat16_d_128_p32` | **75–77%** |

The unique KV at 12k context is 1.21 GB, which is 4.1 ms at this machine's
298 GB/s roof. The fire spends about 20 ms there.

## Why the head axis and not the row axis

`docs/HANDOVER.md` §10 ranks the **k-row** decode kernel first — it shares a KV
read across k query *rows*. That is the right fix for a speculative verify and
for co-batched concurrent decode, and it does nothing at all for the case an
agentic turn is made of: **one stream, one token, so k = 1**. The GQA group is
8 wide at k = 1. It is the only sharing a single-stream decode has.

The two compose, and the new kernel deliberately keeps the k-row prototype's
shape so they can later be one kernel.

## Sweeping the width — and QH=2 is the answer, for a reason worth knowing

`sdpa_paged_probe`'s new head-sharing arm, correctness first (a CPU reference
with **distinct q per query head and distinct k/v per KV head**, so a kernel
that computed head 3 and stored it as head 5 could not pass; `krow_run`'s
uniform data could not have caught that). All widths correct on all 32 heads.

| ms/layer | 2k | 8k | 12k | 16k |
|---|---:|---:|---:|---:|
| QH=1 (shipped shape) | 0.205 | 0.376 | 0.512 | 0.622 |
| **QH=2** | **0.192** | **0.307** | **0.369** | **0.430** |
| QH=4 | 0.227 | 0.394 | 0.508 | 0.662 |
| QH=8 | 0.403 | 1.029 | 1.459 | 1.892 |

QH=4 and QH=8 read a quarter and an eighth of the bytes and are **slower**.
The cause is occupancy, not registers: the grid is `n_q_heads / QH`
threadgroups wide — 32, 16, 8, 4 — and this device stops being filled somewhere
below 16. A kernel reading a quarter of the bytes on a quarter-idle GPU loses to
one reading all of them on a busy one.

Going wider than 2 therefore needs the *key* range split across threadgroups as
well (flash-decoding), so the grid stays tall. That is a different kernel, not a
different constant, and is not attempted here. **It is where the remaining ~4×
to the memory roofline lives.**

## Landed, and priced end to end

`driver/metal/src/kernels/sdpa_paged.metal` (`sdpa_paged_decode_hshare`),
selected by `sdpa_head_share_this_fire` — one predicate, asked by the compile
site and by **both** `pso_for` and `launch_shape`. That sharing is the whole
correctness condition: the grid is half as wide as the per-head kernel's, so
the two sites disagreeing computes half the heads and reports nothing.
`llama_decode_step_test` pins it (232 pass, and it passes with the switch in
either position).

**Deterministic decode rate, same prompts, one server per arm:**

| prompt tokens | head sharing on | off | |
|---:|---:|---:|---:|
| 5,840 | 55.4 tok/s | 46.9 | **1.18×** |
| 16,090 | 36.7 | 27.4 | **1.34×** |
| 28,390 | 28.9 | 19.8 | **1.46×** |

TTFT in the same runs: 8.21/8.23, 27.70/27.80, 57.23/57.43 s — **prefill is
untouched**, which is the control this change needs.

At the kernel, on one identical 23.6k prompt through the dispatch trace:
attention **83.2 → 55.1 ms (1.51×)** and **61.0 → 41.7 ms (1.46×)**, with all
six prefill fires unchanged within 0.5%.

**The output does not change.** `bench_ab.py`'s canned 4-turn replay generated
70 / 215 / 320 / 95 tokens per turn on both arms, and two single-request A/Bs —
250 tokens at short context, 382 tokens at 28.4k — are **byte-identical**.

`PIE_METAL_SDPA_HSHARE=0` is the complete way back: it removes the pipeline and
both selection sites, so the switch cannot half-apply.

## A methodological correction, which cost the afternoon and is worth more than the kernel

The first end-to-end re-run after landing this showed pie finishing in 17 s with
a **0-byte patch**, against 45 s and a 412-byte patch before. That reads exactly
like a broken decode kernel.

It is not. Running the same instance **three times per arm**:

| rep | wall | turns | patch |
|---|---:|---:|---|
| head sharing 1 / 2 / 3 | 54 / 87 / 133 s | 5 / 4 / 19 | 412 B / **0 B** / 412 B |
| baseline 1 / 2 / 3 | 22 / 48 / 53 s | 3 / 5 / 5 | **0 B** / 412 B / 412 B |

**The baseline fails the same way, at the same rate.** The agent loop is not a
deterministic function of the engine: opencode's own prompt varies run to run
(7406 / 7408 / 7425 tokens for the same turn), and once the prompt differs,
greedy decoding diverges and the agent takes a different path — 3 turns or 19.

Two consequences, both binding on anything measured here from now on:

1. **A single agentic run cannot price a change**, in either direction. The
   `results-e2e-one-instance.md` baseline is one sample and its wall clock
   carries a variance this large. Its *per-call rates* — TTFT per fresh token,
   decode tok/s — are per-call properties and survive; its totals do not.
2. **A fast arm with an empty patch is the failure mode this repo already
   knows** (`results-swebench.md`: vLLM's 10-second "wins"). It reappeared here
   pointing at my own change, and the only thing that separated the two claims
   was running the control.

The numbers quoted above are all from deterministic instruments — the probe,
the dispatch trace, the canned replay, and fixed-prompt single requests —
for exactly this reason.

## Reproducing

```sh
/tmp/metaltools/bin/sdpa_paged_probe            # the width sweep, with references
/tmp/metaltools/bin/llama_decode_step_test      # the grid/predicate agreement
PIE_METAL_SDPA_HSHARE=0 /tmp/metaltools/bin/llama_decode_step_test   # and the way back
bash integrations/opencode/tools/pie_ab.sh django__django-14373 3    # the control
```
