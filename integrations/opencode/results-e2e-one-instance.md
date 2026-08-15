# One SWE-bench instance, three engines, decomposed — 2026-08-15

**Machine:** Apple M5 Pro, 48 GB, idle (`roofline_probe` 295.3–297.9 GB/s before
each arm). **Model:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, the
same weights on all three. **Agent:** stock opencode 1.18.18.
**Instance:** `django__django-14373`. **Harness:** `tools/e2e1.sh`, every arm
through `tools/turnlog.py`.

> **CORRECTION, added the same day — the totals below are one sample and the
> variance is larger than most of the differences they are used to argue.**
>
> Running this same instance three times per arm afterwards
> (`results-head-sharing-decode.md`, "A methodological correction") gave pie
> 22 / 48 / 53 s with 3 / 5 / 5 turns, and **one of the three produced a 0-byte
> patch**. The agent loop is not a deterministic function of the engine:
> opencode's own prompt for the same turn varies (7406 / 7408 / 7425 tokens),
> and once it does, greedy decoding diverges and the agent takes a different
> path.
>
> So "all three produced the byte-identical patch in exactly 5 turns" was
> **luck, not a property** — a favourable draw that made the comparison look
> more controlled than it is. What survives is the PER-CALL columns: TTFT
> against fresh prompt tokens, and decode tok/s. Those are properties of a
> single model call and do not depend on how many calls the agent chose to
> make. The wall-clock and in-server TOTALS do not survive, and the attribution
> in "Where pie's 20.9 s go" should be read as indicative rather than measured.
>
> The direction it points is independently confirmed and is not in doubt:
> pie's prefill is 0.32–0.37× of mlx's on the same prompt, per turn, and that
> is a per-call measurement.

## The comparison is clean, and that is the point

All three engines produced the **byte-identical 412-byte patch**
(`sha256 a535bfde25c1…`) in **exactly 5 turns**, from prompts within 0.6% of
each other. The agent did the same work, in the same shape, on each engine.

That matters because on this benchmark wall clock is

    turns x (prefill + decode)

and it has already misled once here: `results-swebench.md`'s vLLM arm looked
faster at 10–39 s per instance when it was an agent giving up after two calls.
With the turn count and the patch pinned, the remaining difference is serving
speed and nothing else.

## Result

| | pie | vLLM-metal | mlx-lm |
|---|---:|---:|---:|
| agent wall | 51 s | 38 s | **29 s** |
| time inside the server | 42.6 s | 39.0 s | **21.7 s** |
| — of which TTFT | 24.9 s | 19.4 s | **9.2 s** |
| — of which decode | 17.7 s | 19.6 s | **12.5 s** |
| output tokens | 641 | 572 | 681 |
| decode rate | 36.2 tok/s | 29.2 tok/s | **54.5 tok/s** |
| patch | 412 B ✓ | 412 B ✓ | 412 B ✓ |

**mlx-lm is the engine to beat, not vLLM.** It is 1.96× pie on server time.
vLLM is only 1.09× pie here — and that margin is entirely its own cold start
(turn 1 cost it 4.5 s of TTFT and 5.3 s to emit 9 tokens; excluding turn 1 it is
29.2 s against pie's 41.3 s, i.e. 1.41×).

## Where pie's 20.9 s go, against mlx

| | pie | mlx | gap |
|---|---:|---:|---:|
| TTFT | 24.9 s | 9.2 s | **15.7 s (75% of the gap)** |
| decode | 17.7 s | 12.5 s | 5.2 s (25%) |

Normalising decode for the differing token counts: pie 27.6 ms/token,
mlx 18.4 ms/token — **1.50×**. Prefill is **2.7×**.

Per turn, with the fresh (uncached) prompt tokens divided by TTFT:

| turn | fresh prompt | pie | vLLM | mlx |
|---:|---:|---:|---:|---:|
| 2 (cold 7.4k) | 7408 | 654 tok/s | 938 | **1762** |
| 4 | ~4200 | 400 tok/s | 654 | **1246** |

pie's prefill is 0.37–0.32× of mlx's on the same prompt. Turn 4 is worth
noticing on its own: pie prefills a 4.2k delta *more slowly per token* than it
prefills a 7.4k cold prompt, which is the chunking, not the context.

**This corrects the emphasis in `docs/HANDOVER.md`.** That document's headline
is a 2.5× throughput deficit at 8-way concurrency, and its first recommendation
is the k-row decode kernel. Neither is what this workload is made of: one agent
is one stream, so the concurrency ceiling in §5 never binds, and decode — where
the k-row kernel and speculation act — is a quarter of the gap. **Three
quarters of it is prefill.**

## Correctness is unchanged and is not what this measures

n = 1 cannot support a correctness claim and is not asked to; all three arms
resolving this instance identically is a property of the instance (it was
chosen as the one both pie and vLLM already solved). The correctness result
stands where it was — `swe3.sh` + `grade.sh` over the known-5, pie 4/5 against
vLLM 1/5 — and this run neither strengthens nor weakens it.

What this run *does* establish is that a speed comparison on this instance is
uncontaminated by correctness: nobody gave up, nobody wrote a different patch.

## Method notes

* **`turnlog.py` is a streaming proxy and must not buffer.** Its first version
  used `HTTPResponse.read(n)`, which blocks until it has n bytes, and so
  reported TTFT == total and a decode rate of 190,000 tok/s. `read1` is what
  returns what has arrived. Proxy tax measured at ~7 ms on an 800 ms call
  (<1%), and it is on every arm equally.
* **`e2e1.sh`'s roofline gate did not gate during this run.** It parsed `$NF`
  from `"… -> 297.9 GB/s"`, getting the unit, and awk compared the string
  `"GB/s"` against 250 lexically — which passes for any machine state. Fixed
  and tested in both directions afterwards. The run itself is still trustworthy:
  `roofline_probe` was run by hand at 297.9 GB/s before it and 295.3 GB/s after,
  and no other process exceeded 2 GB throughout. Recorded because a gate that
  reports OK without checking is the exact failure this repo has hit three
  times (§9.1 of the handover).
* vLLM's `cached` column is empty because it does not populate
  `prompt_tokens_details` in the OpenAI usage payload. Its prefix cache is
  live; its own engine log reports the hit rate.

## Reproducing

```sh
bash integrations/opencode/tools/e2e1.sh django__django-14373
python3 integrations/opencode/tools/turn_summary.py /tmp/e2e1-*/turns-*.jsonl
```

## What this says to do next

1. **Prefill.** 75% of the gap to mlx, and `results-turn-latency.md`'s dispatch
   trace puts attention at 43% of a prefill fire and the routed MoE GEMM at 38%.
   pie's attention runs at 6.32 ms/layer against mlx's 1.31 — and mlx's number
   is essentially its memory roofline, so the difference is that mlx issues the
   attention matmuls on the M5 neural accelerators and pie issues them on the
   simdgroup matrix unit, whose measured ceiling (5.48 TFLOP/s) puts a **4.1
   ms/layer floor under pie's current kernel no matter how it is tuned**.
   Re-verify the shares on THIS workload's fire shapes before building.
2. **Decode**, 25% of the gap, 1.50× at ~10k context.
3. Not the concurrency ceiling, and not — for this workload — speculation.
