# pie (now) vs pie (original) vs mlx-lm vs vLLM-metal — 2026-08-16

**One session, four arms, boot → measure → kill → next.** No number here is
quoted from memory: every engine ran today, on this machine, next to the others.
`tools/four_way.sh` is the harness.

**"Original pie" is the same binary** with the three kernels this work added
switched off (`PIE_METAL_SDPA_HSHARE=0 PIE_METAL_SDPA_NAX=0
PIE_METAL_QMM_NAX=0`). That is a stronger control than an old commit — same
build, same config, same server, so the only difference is the kernels — and it
is also the honest test of those switches: a half-applying revert would show up
right here.

## Machine conditions, stated because they were not ideal

A macOS Spotlight indexer (`spotlightknowledged`) held ~96–100% of one CPU core
throughout. The preflight gate warned; the run continued because the thing that
actually corrupts these measurements is *memory* contention, and the streaming
roof stayed at **288–294 GB/s** across all five arms (normal is 294–298).

**The drift control says how much it cost.** The first arm was repeated last:

| | pie_now (first) | pie_now_repeat (last) | drift |
|---|---:|---:|---:|
| TTFT 5,840 | 2.62 s | 2.62 s | 0% |
| TTFT 16,090 | 7.39 s | 7.67 s | +3.8% |
| TTFT 28,390 | 13.81 s | 15.34 s | **+11%** |
| 6-turn replay | 7.05 s | 7.37 s | +4.5% |

So drift grows with prompt length and the 28k column carries ~10% uncertainty.
mlx-lm and vLLM ran in the middle of the session, so they sit between the two
pie measurements in time. **Every conclusion below survives using pie's WORST
number against the others' single number** — that is the test applied before
stating any of them.

---

## TTFT (prefill)

| prompt | **pie now** | pie original | mlx-lm | vLLM-metal |
|---:|---:|---:|---:|---:|
| 5,840 | **2.62 s** | 8.02 | 2.88 | 4.42 |
| 16,090 | **7.39 s** | 27.81 | 8.74 | 17.10 |
| 28,390 | **13.81 s** | 57.58 | 17.13 | 37.26 |

**pie is now the fastest of the three engines on prefill at every size.**

| pie now is faster than | 5,840 | 16,090 | 28,390 |
|---|---:|---:|---:|
| its own original | 3.06× | 3.76× | 4.17× |
| mlx-lm | 1.10× | 1.18× | 1.24× |
| vLLM-metal | 1.69× | 2.31× | 2.70× |

At the worst-case (repeat) pie number, the mlx margin at 28k is 1.12× rather
than 1.24× — still pie's.

## Decode

| prompt | pie now | pie original | **mlx-lm** | vLLM-metal |
|---:|---:|---:|---:|---:|
| 5,840 | 55.1 tok/s | 47.0 | **65.9** | 51.2 |
| 16,090 | 40.3 | 25.7 | **47.3** | 30.1 |
| 28,390 | 28.0 | 19.8 | **35.7** | 16.9 |

**mlx-lm still wins decode**, by 1.20× / 1.17× / 1.28×. pie now beats vLLM-metal
at every size and is 1.17× / 1.57× / 1.41× over its own original.

## The 6-turn canned agentic replay

| | total | turn 1 TTFC (cold 7.2k) |
|---|---:|---:|
| **pie now** | **7.05 s** | 3.279 s |
| pie original | 16.31 s | 11.16 s |
| mlx-lm | 7.94 s | 3.973 s |
| vLLM-metal | 10.17 s | 5.51 s |

**pie is fastest**, 1.13× over mlx-lm and 1.44× over vLLM-metal, and 2.31× over
its own original. Read with the caveat that this replay generates ~10 tokens a
turn and so is prefill-dominated — it weights pie's strength and under-weights
mlx's. A turn that writes a long patch would move toward mlx.

---

## The honest summary

**Prefill: pie wins, by 1.10–1.24× over mlx-lm and 1.7–2.7× over vLLM-metal.**
**Decode: mlx-lm wins, by 1.17–1.28×.** Which engine is faster end to end
therefore depends on the shape of the turn, and on this prefill-heavy replay it
is pie by 1.13×.

Against pie two days ago: **3.06–4.17× on prefill, 1.17–1.57× on decode,
2.31× on the replay.**

---

## Prefill composition, re-traced on the current build

Same 23,655-token prompt, `PIE_METAL_DISPATCH_TRACE=1 PIE_METAL_TRACE_STRIDE=8`,
fourth measurement in the series:

| traced wall | pre-NAX | + NAX attn | + exp 3 | **+ NAX GEMMs** |
|---|---:|---:|---:|---:|
| | 66.77 s | 37.72 s | 29.98 s | **17.57 s** |

**3.80× cumulative**, and the last column was taken with the Spotlight indexer
running, so it is if anything conservative.

| kernel | ms | share | previous share |
|---|---:|---:|---:|
| `sdpa_paged_nax` (attention) | 9742 | **56.7%** | 32.2% |
| `affine_qmm_t_routed_nax` | 4073 | 23.7% | 43.6% |
| `affine_qmm_t_nax` | 1990 | 11.6% | 19.5% |
| everything else | ~1370 | 8.0% | 4.1% |

In situ the GEMMs moved **3.17×** (routed, 12898 → 4073 ms) and **2.90×** (dense,
5774 → 1990) — better than the 2.21× and 2.72× measured in isolation, which is
the same direction attention went and for the same reason: an isolated probe is
not competing for anything.

**Attention is the largest term again, at 56.7%.** That is the third rotation:
attention → routed GEMM → attention. Every time a term is fixed the ordering
changes, which is exactly why the rule here is to re-trace rather than to plan
off a share measured before the last change.

**But it is a harder target now.** pie's attention is already 1.435 ms/layer
against MLX's 1.555, so there is no reference implementation left to copy — the
remaining headroom is the gap from 15.5 TFLOP/s to `matmul2d`'s 32.5, and
finding it means profiling rather than porting.

## Reproducing

```sh
bash integrations/opencode/tools/four_way.sh          # all four arms, one session
python3 integrations/opencode/tools/rate_probe.py --base-url … --model … --label …
```
