# Prefill attention on the neural accelerators — 2026-08-15

**Machine:** Apple M5 Pro, 48 GB, idle. **Model:**
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, 32 q heads over 4 kv heads,
head_dim 128, page size 32.

## Why the unit had to change

pie's prefill attention was compute-bound on the simdgroup matrix unit, and the
ceiling sat just above where it ran:

| | ms/layer | TFLOP/s |
|---|---:|---:|
| `sdpa_paged_mma` (shipped) | 6.97 | 3.2 |
| **simdgroup ceiling** (`matrix_rate_probe`, three configurations agree) | **4.1** | **5.48** |
| MLX, same shapes | 1.555 | 14.4 |
| `matmul2d` on the neural accelerators | — | 32.5 |

MLX runs at 14.4 TFLOP/s, which is **above the simdgroup ceiling**. That is
arithmetic, not inference: no arrangement of a simdgroup kernel reaches it. The
unit was the difference.

## The kernel

`driver/metal/src/kernels/sdpa_paged_nax.metal`, with the fragment helpers in
`nax_frag.h`. **2.08 ms/layer at 184 rows / 7424 ctx — 3.35× the shipped
kernel, at 10.75 TFLOP/s, below the simdgroup floor.**

Three things carry it, and the first two are corrections of the previous
session's failed fused attempt:

1. **O and S live in plain `thread` register arrays.** A cooperative tensor does
   not survive a `run()` call — measured at 7316 of 8192 elements wrong, worst
   relative exactly 1.0000. Every cooperative tensor is now created, filled, run
   and read back inside one `frag_mma`. This is MLX's structure
   (`steel/attn/nax.h`), reimplemented against the same public API.
2. **No threadgroup staging at all.** Staging was measured at 2.921 ms/layer on
   its own — more than the whole multiply. Fragments load straight from device
   memory.
3. **The paged bridge is a coincidence worth stating.** `kv_page_size` is 32 and
   a NAX fragment is 16 keys, so a 32-key block is *exactly one page*, never
   straddling. One page-table read per block, then a constant row stride inside
   it. No scratch buffer, no de-paging pass. If page size is ever not 32 this
   kernel must not be selected, and the host tests exact equality rather than
   inferring a power of two.

## Correctness, before any timing was quoted

`results-*.md` records a NAX kernel measured at 1.652 ms/layer and reported
three times before a CPU reference showed it computed the wrong thing. So:
correct against a float64 reference at six shapes, with **distinct q per query
head and distinct k/v per KV head** so right arithmetic against the wrong head
cannot pass. The shapes are deliberately unfriendly — the friendly ones hide
exactly the masking bugs this kernel can have:

| rows | ctx | | |
|---:|---:|---|---|
| 64 | 96 | both aligned | 0 of 262144 wrong |
| 64 | 100 | ctx % 32 = 4 | 0 of 262144 |
| 40 | 96 | rows below one tile | 0 of 163840 |
| 70 | 133 | both ragged | 0 of 286720 |
| 17 | 1 | tiny, ctx 1 | 0 of 69632 |
| 184 | 224 | serving width | 0 of 753664 |

**And accuracy against the kernel it REPLACES**, which is the comparison that
actually matters — a new kernel does not have to match exact arithmetic, it has
to not be worse than the old one. Same inputs, same float64 reference:

| ctx | kernel | mean | p99 | worst |
|---:|---|---:|---:|---:|
| 224 | `sdpa_paged_mma` | 1.35e-3 | 4.07e-3 | 4.66e-3 |
| 224 | `sdpa_paged_nax` | 1.39e-3 | 4.19e-3 | 4.95e-3 |
| 1024 | `sdpa_paged_mma` | 1.21e-3 | 3.80e-3 | 4.46e-3 |
| 1024 | `sdpa_paged_nax` | **1.23e-3** | **3.76e-3** | **4.21e-3** |

Indistinguishable, and slightly better at the longer context. **This matters
because the kernel does change serving output** — the same 28.4k prompt that
gave byte-identical text with head sharing on and off gives 332 tokens instead
of 382 with NAX on. That is the ordinary consequence of swapping kernels
(`apc-graft-probe` measured the same shape of divergence from re-chunking a
prefill, which changes no arithmetic at all), and the table above is what says
so rather than a hope.

## The tile width was swept, not assumed

| BQ | ms/layer | |
|---:|---:|---|
| 32 | 2.053 | |
| **64** | **2.082** | taken — within the probe's own ~1.3% drift of 32, and 128 threads, the same threadgroup as `sdpa_paged_mma` |
| 128 | 2.832 | loses |

The first version of that sweep hardcoded a 64-row tile and 128 threads **in the
harness** while varying the kernel constant, and duly reported the BQ=128 kernel
as 24576 elements wrong. The kernel was fine; the grid was a different kernel's.
That is the `pso_for` / `launch_shape` disagreement this repo documents,
reproduced inside the instrument built to measure it.

## End to end, on fixed prompts

Deterministic: same prompts, 200 generated tokens, one server per arm. Both
landed kernels on, against both off, against mlx-lm.

**TTFT (prefill):**

| prompt | pie, both off | pie, both on | mlx-lm |
|---:|---:|---:|---:|
| 5,840 | 8.23 s | **6.36 s** (1.29×) | 2.88 s |
| 16,090 | 27.80 s | **15.75 s** (1.77×) | 8.20 s |
| 28,390 | 57.43 s | **30.94 s** (1.86×) | 15.40 s |

**Decode:**

| prompt | pie, both off | pie, both on | mlx-lm |
|---:|---:|---:|---:|
| 5,840 | 46.9 tok/s | **53.7** (1.15×) | 66.3 |
| 16,090 | 27.4 | **40.3** (1.47×) | 47.6 |
| 28,390 | 19.8 | **27.9** (1.41×) | 35.9 |

**The gap to mlx roughly halved on both axes**: prefill from 2.9–3.7× behind to
1.9–2.2×, decode from 1.4–1.8× to 1.2–1.3×. **It is not parity**, and the
remaining prefill gap is no longer attention — with attention down 3.35×, the
routed MoE GEMM is now the largest term in a prefill fire.

## Gating, and the way back

`sdpa_nax_this_fire` is asked by the compile site and by **both** `pso_for` and
`launch_shape`. The NAX tile is 64 rows where the matrix kernel's is 32, so the
two sites agreeing is the correctness condition; disagreeing runs half the fire
and reports nothing. `llama_decode_step_test` pins it (232 pass) and passes with
`PIE_METAL_SDPA_NAX` in either position.

`requests == 1` is load-bearing rather than caution: a 64-row tile spanning two
requests spans two page lists and the kernel resolves one base pointer per tile.
A prefill fire is one request contributing thousands of rows; anything
co-batched falls back to `sdpa_paged_mma`.

**Stated rather than left implicit:** the kernel reads no user attention mask.
A mask is a per-fire property and neither selection site is handed it, so it
cannot be gated on. This inherits exactly the assumption
`sdpa_paged_decode_..._p32` (FAST_FULL) already ships with for this family at
page size 32 — a pre-existing property, not one introduced here, but if masks
are ever enabled on llama both kernels are wrong together.

## Tests

| suite | |
|---|---|
| `llama_pso_test` | 32 pass |
| `llama_decode_step_test` | 232 pass (and 232 with `PIE_METAL_SDPA_NAX=0`) |
| `kv_append_paged_pso_test` | 13 pass |
| `gptoss_decode_step_test` | all pass |
| `llama_numerics_test` | 51 pass / 18 **pre-existing** fail, unchanged |

One pre-existing assertion had to be updated rather than merely re-run: the
tiled-grid coverage test asserted a 32-row tile unconditionally, and a 95-row
single-request fire now takes the 64-row NAX tile. It now asks the same
predicate the launch did, so it holds under either path — which is a stronger
test than it was, not a weakened one.

## Next

The routed MoE GEMM. It was 38% of a prefill fire when attention was 43%;
attention is now a third of what it was, so the mixture is the largest term and
the next 2× on prefill is there, not in attention.
