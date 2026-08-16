# Handover — pie on Apple Silicon, opencode integration

**Rewritten 2026-08-16 from re-measured numbers.** The previous version of this
file is superseded in three load-bearing ways and they are called out in §2,
because each was a belief that steered work in the wrong direction.

- **Machine:** Apple M5 Pro, 48 GB. Streaming roof 294–298 GB/s
  (`roofline_probe`, the bandwidth authority — not the process list).
- **Model:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`. 48 layers,
  32 q heads, 4 kv heads, head_dim 128, 128 experts top-8, page size 32.
  All engines serve the same weights.

---

## 1. Where it stands

**pie's prefill is now faster than mlx-lm's; pie's decode is still ~1.25×
slower.** Both measured in the same session, on the same machine, deterministic
fixed prompts:

| prompt | pie TTFT | mlx-lm | | pie decode | mlx-lm |
|---:|---:|---:|---|---:|---:|
| 5,840 | **2.60 s** | 2.88 s | pie 1.11× | 54.6 tok/s | 66.3 |
| 16,090 | **7.38 s** | 8.20 s | pie 1.11× | 40.1 | 47.6 |
| 28,390 | **14.31 s** | 15.40 s | pie 1.08× | 27.9 | 35.9 |

TTFT is **3.07× / 3.75× / 4.00×** better than two days ago.

6-turn canned agentic replay (`bench_ab.py --turns 6`), both engines in one
session: **pie 7.11 s against mlx-lm 7.64 s**; cold 7.2k prefill 3.279 s against
3.973 s; decode 46–48 tok/s against 56–62.

Correctness is where it was — pie 4/5 on the SWE-bench known-5 against vLLM's
1/5 (`results-swebench.md`); nothing since has touched the agent path's logic,
and generation is verified coherent after every kernel landing.

---

## 2. Three corrections to the previous handover

1. **"pie wins correctness and loses speed by ~2.5×" is out of date.** That was
   an 8-way concurrency throughput number. An agentic turn is ONE stream, so the
   concurrency ceiling never binds, and the single-stream picture was different
   and is now reversed on prefill.

2. **The k-row decode kernel was ranked first and is the wrong axis.** It shares
   a KV read across query ROWS, and a single-stream agentic decode has k=1. The
   axis that is 8 wide at k=1 is the GQA head group, and sharing there gave
   1.18–1.46× (§3). The k-row work remains correct and unlanded; it pays for
   speculative verify and co-batched decode, neither of which this workload runs.

3. **The prefill attention gap was not tunable and is now closed.** pie ran
   3.2 TFLOP/s against a *measured* 5.48 simdgroup ceiling; MLX ran 14.4, above
   that ceiling, so it was on the neural accelerators and no arrangement of a
   simdgroup kernel could reach it. pie's paged attention is now 1.435 ms/layer
   against MLX's contiguous 1.555.

---

## 3. What landed, and the knob that reverts each

| | kernel | win | revert |
|---|---|---|---|
| decode attention | `sdpa_paged_decode_hshare` — one KV read per GQA pair | 1.18× @5.8k, 1.46× @28k | `PIE_METAL_SDPA_HSHARE=0` |
| prefill attention | `sdpa_paged_nax` — fused flash attention on the neural accelerators | 4.86× over `sdpa_paged_mma` | `PIE_METAL_SDPA_NAX=0` |
| routed MoE GEMM | `affine_qmm_t_routed_nax` | 1.92× / 2.21× by tile | `PIE_METAL_QMM_NAX=0` |
| dense projections | `affine_qmm_t_nax` | 2.72× | `PIE_METAL_QMM_NAX=0` |

`driver/metal/src/kernels/nax_frag.h` is the shared substrate: 16×16 register
fragments, `frag_mma` (N=32) and `frag_mma_k32`, and the lane layout everything
depends on. The design is MLX's (`steel/attn/nax.h`), reimplemented against the
same public API.

**The hazard all four share, and the one that must not be got wrong.** Where a
kernel's grid differs from the one it replaces, `pso_for` and `launch_shape`
decide independently and disagreeing runs a fraction of the fire while reporting
nothing. Every predicate is therefore asked from ONE function by the compile
site and both selection sites, and `llama_decode_step_test` pins it with each
switch in either position. The two GEMM kernels deliberately keep the same tile,
threadgroup and grid, so for them only the entrypoint NAME differs and no launch
site can disagree at all — that is why their landing was one line.

---

## 4. Full prefill composition, re-measured after every change

Same 23,655-token prompt, `PIE_METAL_DISPATCH_TRACE=1 PIE_METAL_TRACE_STRIDE=8`:

| traced wall | pre-NAX | after NAX attn | after exp 3 |
|---|---:|---:|---:|
| | 66.77 s | 37.72 s | 29.98 s |

Shares move a lot as terms are fixed, which is why they are re-measured rather
than carried: attention went 69% → 47% → 32%, the routed GEMM 19% → 35% → 44%.
**Do not plan off a share measured before the last change.** Both GEMMs have
since been moved to NAX and the composition is stale again — re-trace before
choosing the next target.

---

## 5. OPEN — do not paper over these

1. **Decode is the larger deficit now**, ~1.25× behind mlx-lm and untouched
   since head sharing. The known lever is splitting the key range
   (flash-decoding) so head sharing can go past QH=2: QH=4 and QH=8 read a
   quarter and an eighth of the bytes and are *slower*, purely because the grid
   is `n_q_heads/QH` threadgroups and this device stops being filled below 16.
   Unattempted.

2. **The NAX kernels ignore a user attention mask.** A mask is a per-fire
   property and neither selection site is handed it, so it cannot be gated on.
   This inherits exactly the assumption `sdpa_paged_decode_..._p32` (FAST_FULL)
   already shipped with for this family — pre-existing, not introduced — but if
   masks are ever enabled on llama, several kernels are wrong together.

3. **A single agentic run cannot price a change.** Three reps per arm on
   `django__django-14373` gave 3/4/5/19 turns and a 0-byte patch in one rep of
   BOTH arms. opencode's own prompt varies run to run (7406/7408/7425 tokens for
   the same turn), so greedy decoding diverges and the agent takes a different
   path. Use the deterministic instruments — the probes, the dispatch trace,
   `bench_ab.py`'s canned replay, fixed-prompt single requests — and use
   `tools/pie_ab.sh` when an agentic number is unavoidable.

4. **`llama_numerics_test` is 51 pass / 18 fail, pre-existing** and unchanged
   through all of this. MoE routing ties, first divergence at a layer-0
   projection.

5. **pie strategy B fails under concurrent load** (10 of 32 completed, 22 HTTP
   500). Unchanged and undiagnosed.

---

## 6. Tests and instruments

```sh
cmake -S driver/metal -B /tmp/metaltools -DPIE_METAL_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/metaltools -j 8
```

| suite | state |
|---|---|
| `llama_pso_test` | 32 pass |
| `llama_decode_step_test` | 232 pass (and with each NAX/HSHARE switch off) |
| `kv_append_paged_pso_test` | 13 pass |
| `gptoss_decode_step_test` | all pass |
| `llama_numerics_test` | 51 / 18 pre-existing fail |

| probe | answers |
|---|---|
| `sdpa_paged_probe` | attention per configuration; the head-sharing, staging, block-width and mask-split experiments; accuracy of each kernel against the one it replaces |
| `qmm_nax_probe` | the routed and dense quantized GEMMs, simdgroup against neural accelerator |
| `matrix_rate_probe` | the two matrix units' ceilings — 5.48 against 32.5 TFLOP/s |
| `roofline_probe` | the streaming roof; **the bandwidth authority** |

| harness | measures |
|---|---|
| `tools/turnlog.py` | per-call TTFT / decode / tokens, identically on any OpenAI-compatible server |
| `tools/e2e1.sh` | one SWE-bench instance, three engines, decomposed |
| `tools/pie_ab.sh` | N reps per arm, to separate a change from run-to-run variance |
| `bench_ab.py` | the canned 6-turn replay — deterministic, unlike a live agent |

---

## 7. Traps that cost real time

The older ones still hold (`results-turn-latency.md` §"Traps"). Added since:

1. **An instrument can reproduce the bug it was built to find.** The tile-width
   sweep hardcoded a 64-row tile and 128 threads in the HARNESS while varying the
   kernel constant, and reported a correct kernel as 24576 elements wrong. The
   roofline gate in `e2e1.sh` parsed `$NF` and compared the string `"GB/s"`
   against 250 — passing for every machine state, including a contended one.
2. **A failed experiment that rules out a class is worth more than a small win.**
   Staging K/V (2.4× slower) ruled out memory; widening the key block (up to 8×
   slower) ruled out the epilogue; what was left — remove per-block work that
   almost no block owes — was 1.45×. Both failures are recorded in
   `results-prefill-experiments.md` and neither should be retried.
3. **Read the reference for its predicates, not only its math.** The winning
   attention change (`align_K && is_last_k`, `kb >= kb_min_causal`) was visible
   in MLX's source the whole time. It was read for the fragment layout and the
   mma contract, and the predicates were skipped as bookkeeping.
4. **Compare a new kernel to the one it REPLACES, not to exact arithmetic.** The
   NAX attention changes serving output (332 tokens where there were 382). That
   is only defensible because its error against a float64 reference is
   indistinguishable from the old kernel's — 1.23e-3 against 1.21e-3 mean.

---

## 8. What I would do next, in order

1. **Decode**, now the larger gap. Flash-decoding to lift head sharing past
   QH=2 (§5.1).
2. **Re-trace the prefill composition** before touching prefill again — both
   GEMMs moved, and every share in §4 is stale.
3. The k-row decode kernel, if and when speculation or co-batched decode
   matters; it is correct, measured and unlanded.
