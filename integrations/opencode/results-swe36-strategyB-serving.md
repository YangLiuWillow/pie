# Strategy B under the real envelope: 30 SWE-bench instances, serving analysis — 2026-08-18

The pie-B arm of the swe36 four-way (relaunched post §3.1-3.3, post
context-standardization), analyzed the evening it finished, while the
strategy-A arm runs. Serving robustness only — resolve rates need Docker
grading and the cross-engine columns need the other arms.

Run: `/tmp/swe36-20260818`, 30 instances, 1800 s timeout, temp 0,
`enable_thinking=false` pinned and verified (0 dropped). Strategy B at
max_model_len 65536, client limit 61440. Arm wall: ~4.5 h.

## The headline: the failure ledger is clean where it must be

| serving-fault class | count over 30 instances / 1,686 calls |
|---|---|
| driver poison epochs | **0** |
| `launch failed` | **0** |
| inferlet panics / process deaths | **0** |
| degraded (empty) completions | **0** |
| session-killing refusals | **0** — 30 × 503, every one survived |
| mislabeled retains (the §11 guard) | **0** |

Monitors were armed on both engine and shim logs the whole arm (the same
signatures that caught every historical fault) — silence here is
instrumented, not blind.

## The envelope was genuinely traversed

This is the part no prior run exercised — pre-fix, conversations truncated
at ~12.3k:

- **Median turn resumed at 35,487 cached tokens.** p90 52,621; max 57,215.
- **93% of calls hit the cache**, median reuse 0.99 of the prompt when hit.
  Strategy B's entire premise — pay only for the delta — held at 35-57k
  depths across 1,642 archived turns.
- **6 earlier-boundary refusals in the whole arm** (0.2/instance, was
  ~14/instance pre-fix — the context fix measured ~70× fewer). All six are
  the same shape: opencode compacting at its new ~57k threshold back to the
  ~7.5k head, refused into a cold rebuild, exactly as designed.
- **30 × 503** — the gateway's bounded refusal at genuine pool saturation
  (conversations pressing the 65,536 pool). Every one was survivable: 279
  successful calls landed within 120 s after a 503; no session died. This
  is `fcb1dc814` (a refused turn keeps the session) doing its job at scale.

Latency/throughput (from calls-pie.jsonl; run.log's "0 calls" line is an
accounting artifact — the file holds 1,686 records):

- TTFT median 2.85 s, p90 9.62 s, max 168.6 s (the max is a full ~57k cold
  prefill after a compaction — the priced cost of fold irreversibility).
- Decode 60.1 tok/s median (per-call p90 is a small-completion artifact).
- Prompt sizes: median 36,201, max 58,680 tokens.

## What is NOT clean — and is not serving

- **18/30 non-empty patches** (12 empty), 3 × 1800 s timeouts (one with a
  2,247-byte patch that ran out of clock).
- **transcript_health flags looping on 14/30 instances** (worst:
  flask-5014 repeating one command 215×). This is agent/model behavior at
  temp 0, not a serving fault — and the new envelope plausibly AMPLIFIES
  it: pre-fix, compaction at 12k effectively reset a looping agent every
  few turns; now it can loop across 50k tokens before compaction
  intervenes. The other arms (same model, same limits, stateless engines)
  will show whether the loop rate is engine-independent; if it is, this is
  a model/agent-loop property and belongs in the accuracy caveats, not the
  engine comparison.

## Verdict on "bulletproof"

For the failure classes strategy B is RESPONSIBLE for — state correctness,
fault containment, cache honesty — this run is the strongest evidence yet:
zero faults across 1,686 calls at depths 3-4× anything previously served,
with every refusal loud, priced, and survived. The design's promise
("slower, never wrong") held: the costs that remain are one cold prefill
per compaction (~6 per 30 instances) and honest 503s at the pool bound.

Not bulletproof in the absolute: `#31` (silent wrong-output risk, bounded,
unobserved) stays open; poison epochs remain a mechanism with no clearing
path (two driver defects on file can produce them, neither reachable by
single-conversation serving); and the accuracy column awaits grading plus
the A/mlx/vllm/llamacpp arms for attribution.
