# pie (now) vs pie (original) vs mlx-lm vs vLLM-metal — 2026-08-16

**One session, four arms, boot → measure → kill → next.** No number here is
quoted from memory: every engine ran today, on this machine, next to the others.
`tools/four_way.sh` is the harness.

**Run twice**: once while a macOS Spotlight indexer held a full CPU core, and
once after it was killed, on a machine whose only other load was the driving
agent (streaming roof 292–296 GB/s, nothing else over 5% CPU). **The clean run
is the one reported below**; the contended run is kept because comparing them
answered a question neither could answer alone (§"What the contention cost").

**"Original pie" is the same binary** with the three kernels this work added
switched off (`PIE_METAL_SDPA_HSHARE=0 PIE_METAL_SDPA_NAX=0
PIE_METAL_QMM_NAX=0`) — a stronger control than an old commit, since build,
config and server are identical and only the kernels differ, and the honest test
of whether any of those switches half-applies.

---

## TTFT (prefill)

| prompt | **pie now** | pie original | mlx-lm | vLLM-metal |
|---:|---:|---:|---:|---:|
| 5,840 | **2.80 s** | 8.25 | 2.98 | 4.43 |
| 16,090 | **7.37 s** | 27.84 | 8.85 | 17.10 |
| 28,390 | **13.93 s** | 57.40 | 17.33 | 37.15 |

**pie is the fastest of the three engines on prefill at every size.**

| pie now is faster than | 5,840 | 16,090 | 28,390 |
|---|---:|---:|---:|
| its own original | 2.95× | 3.78× | 4.12× |
| mlx-lm | 1.06× | 1.20× | 1.24× |
| vLLM-metal | 1.58× | 2.32× | 2.67× |

## Decode

| prompt | pie now | pie original | **mlx-lm** | vLLM-metal |
|---:|---:|---:|---:|---:|
| 5,840 | 54.4 tok/s | 46.9 | **66.1** | 51.3 |
| 16,090 | 41.0 | 26.1 | **47.5** | 30.3 |
| 28,390 | 27.8 | 19.8 | **35.6** | 16.8 |

**mlx-lm still wins decode**, by 1.21× / 1.16× / 1.28×. pie beats vLLM-metal at
every size (1.06× / 1.35× / 1.65×) and is 1.16× / 1.57× / 1.40× over its own
original.

## The 6-turn canned agentic replay

| | total | vs pie now |
|---|---:|---:|
| **pie now** | **7.14 s** | — |
| mlx-lm | 8.02 s | pie 1.12× |
| vLLM-metal | 10.18 s | pie 1.43× |
| pie original | 16.30 s | pie 2.28× |

**pie is fastest.** Read with the caveat that this replay generates ~10 tokens a
turn and so is prefill-dominated — it weights pie's strength and under-weights
mlx-lm's. A turn that writes a long patch would move toward mlx.

---

## What the contention cost — and a correction

The contended run and the clean run agree to **within ~3% on every cell**:

| | contended | clean | |
|---|---:|---:|---:|
| pie now, TTFT 16,090 | 7.39 s | 7.37 s | −0.3% |
| pie now, TTFT 28,390 | 13.81 s | 13.93 s | +0.9% |
| pie original, TTFT 28,390 | 57.58 s | 57.40 s | −0.3% |
| mlx-lm, TTFT 28,390 | 17.13 s | 17.33 s | +1.2% |
| vLLM-metal, TTFT 28,390 | 37.26 s | 37.15 s | −0.3% |
| pie now, replay | 7.05 s | 7.14 s | +1.3% |

**So a CPU-saturating indexer cost these measurements essentially nothing**,
which is consistent with the memory roof having stayed at 288–296 GB/s
throughout: this workload is GPU- and bandwidth-bound, and one busy CPU core did
not reach it. `require_quiet_gpu`'s own note says the same thing from the other
direction — it was written after a *memory* contender tilted an A/B, and it
warns about CPU separately because the two are not the same hazard.

**And a correction to what I said about the first run.** I attributed the
+11% drift on the longest prompt to the contention. It is not: the clean run has
the same drift.

| first arm vs repeat, 28,390 TTFT | contended | clean |
|---|---:|---:|
| pie_now | 13.81 s | 13.93 s |
| pie_now_repeat (same arm, ~20 min later) | 15.34 s | 15.59 s |
| drift | +11.1% | **+11.9%** |

It reproduces on an idle machine, so it is **thermal** — the machine warms over
a ~20-minute run and the longest prefill, which is the most sustained GPU load,
feels it most. That matters for anyone using this harness: **a four-arm run
carries ~10% of position-dependent drift at the long end regardless of what else
is running**, and the fix is interleaving (as `matched_spec.sh` does), not a
quieter machine. Every conclusion above survives using pie's worst number
against the others' single number.

---

## The honest summary

**Prefill: pie wins**, by 1.06–1.24× over mlx-lm and 1.6–2.7× over vLLM-metal.
**Decode: mlx-lm wins**, by 1.16–1.28×. Which engine is faster end to end
depends on the shape of the turn; on this prefill-heavy replay it is pie by
1.12×.

Against pie two days ago: **2.95–4.12× on prefill, 1.16–1.57× on decode,
2.28× on the replay.**

---

## Prefill composition, re-traced on the current build

Same 23,655-token prompt, `PIE_METAL_DISPATCH_TRACE=1 PIE_METAL_TRACE_STRIDE=8`,
fourth measurement in the series:

| traced wall | pre-NAX | + NAX attn | + exp 3 | **+ NAX GEMMs** |
|---|---:|---:|---:|---:|
| | 66.77 s | 37.72 s | 29.98 s | **17.57 s** |

**3.80× cumulative.**

| kernel | ms | share | previous share |
|---|---:|---:|---:|
| `sdpa_paged_nax` (attention) | 9742 | **56.7%** | 32.2% |
| `affine_qmm_t_routed_nax` | 4073 | 23.7% | 43.6% |
| `affine_qmm_t_nax` | 1990 | 11.6% | 19.5% |
| everything else | ~1370 | 8.0% | 4.1% |

In situ the GEMMs moved **3.17×** (routed, 12898 → 4073 ms) and **2.90×** (dense,
5774 → 1990) — better than the 2.21× / 2.72× measured in isolation, the same
direction attention went and for the same reason: an isolated probe competes
with nothing.

**Attention is the largest term again at 56.7%** — the third rotation of the
ordering (attention → routed GEMM → attention). Every time a term is fixed the
ordering changes, which is why the rule here is to re-trace rather than plan off
a share measured before the last change.

**It is a harder target now.** pie's attention is already 1.435 ms/layer against
MLX's 1.555, so there is no reference implementation left to copy; the remaining
headroom is 15.5 → 32.5 TFLOP/s and finding it means profiling rather than
porting.

## Reproducing

```sh
bash integrations/opencode/tools/four_way.sh
# or, on a machine with a daemon that will finish on its own:
bash integrations/opencode/tools/when_quiet.sh -- bash integrations/opencode/tools/four_way.sh
```

## Prefix caching, and how it changes the reading of everything above

All three engines hit **97–98%** on the 6-turn replay, and it is worth separating
the cold turn from the cached ones because they are different measurements:

| engine | cold turn 1 TTFC | cached turns (mean) | vs pie now |
|---|---:|---:|---|
| **pie now** | **3.345 s** | **0.403 s** | — |
| pie original | 10.499 s | 0.721 s | pie 3.14× / 1.79× |
| mlx-lm | 4.321 s | 0.441 s | pie 1.29× / 1.09× |
| vLLM-metal | 5.999 s | 0.461 s | pie 1.79× / 1.14× |

**Prefix caching compresses the engine differences.** On a cold prompt the three
engines span 3.3–6.0 s; on a cached turn they span 0.403–0.461 s, a 14% spread.
That is the honest context for the headline prefill numbers: `rate_probe` measures
**cold** prefill on every call, which is the worst case, and a real agentic
session is mostly cached turns.

Two things follow, and they pull in opposite directions:

1. **The kernel work matters less than the cold numbers suggest** for a session
   that stays in cache — 1.79× on a cached turn against 3.14× on a cold one.
2. **It still matters on cached turns**, because a 190-token delta is not free:
   those rows still attend the whole 7.4k context through the same attention and
   GEMMs. pie original spends 0.721 s where pie now spends 0.403.

**And a cached turn is dominated by FIXED cost, not by its fresh tokens.** 192
tokens in 0.403 s is 476 tok/s, against 2157 tok/s on the cold turn — the same
kernels, a fifth of the apparent rate. Whatever that fixed term is, it is now a
larger share of a steady agentic turn than the tokens are, and it has not been
attributed. (`results-turn-latency.md` chased something similar once and the
answer was a driver row-count cliff, since steered around; this is a different
residue and is unexamined.)

**vLLM reports `cached_tokens: 0` on every turn** while plainly caching — its
TTFC drops from 5.999 s to ~0.46 s. It does not populate
`prompt_tokens_details`, so its column above is "not reported", not "not cached".
Reading it as zero would be the instrument-silence error this repo keeps
tripping over.
