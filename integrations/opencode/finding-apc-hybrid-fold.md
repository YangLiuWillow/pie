# Strategy A's prefix cache serves wrong context on hybrid models — 2026-08-19

**CONFIRMED by measurement.** On an APC resume, `chat-completions` prefills
only the suffix (`cached..n`) and builds a FRESH recurrent working set per
request — so the attention layers read the full history from parked KV while
the GDN layers fold only the unparked tail from a zero state. The model that
answers is not the model that would have answered cold.

## The probe (`tools/fold_divergence_probe.py`, 10 minutes)

Same request three times against a fresh strategy-A boot, temp 0, a
9,105-token prompt asking for the most important invariant of a real source
file. Request 1 is cold (fold covers everything, and the turn parks the
history); requests 2 and 3 resume at 9,088 of 9,105 tokens — the fold covers
17 tokens.

```
C1 (cold):    "…every logical page index within a working set is either
               backed by a live physical page ID or is explicitly marked…"
C2 (resumed): "…the `KvStore` struct must always maintain a consistent
               relationship between its `table` … its `pool`…"
C3 (resumed): identical to C2, byte for byte
```

`C1 != C2`, `C2 == C3`: the divergence is systematic, not sampling noise.
Both answers are fluent and on-topic — which is precisely why nothing ever
caught it. No driver check can: the fresh instance's first fire carries a
legitimate reset and self-consistent geometry. Raw transcript:
`fold-divergence-result.txt`.

## Scope

- **Coder-30B (pure attention): unaffected.** KV is the whole state; the APC
  is sound there, which is where it was built and measured (99.9% reuse).
- **Qwen3.6-35B-A3B (hybrid): every APC hit is wrong-context.** The swe36
  benchmark's pie-A arm ran with ~94% hit rate — its accuracy column is
  "A with unfolded resume", labeled as such, and the A-vs-B accuracy delta
  after grading is this defect's measured cost.
- Strategy B is immune by construction: it retains and resumes `{ws, rs}`
  as one unit and refuses any resume the fold cannot honor.

## Fix ladder

1. **Gate (immediate, one line):** skip `apc.resume` when
   `model::pass_kind() != Attention`. Correct everywhere; A reverts to
   full re-prefill on hybrids. Lands right after the benchmark's A arm
   completes (not before — the arm must stay on one build).
2. **Tip-cut fold parking (the real fix):** park the recurrent state
   alongside KV, at exact-tip cuts only — interior cuts are impossible
   because a fold is a running summary, not a list (the same constraint B
   lives under). Needs an engine-level index for RS state.

## The general lesson

The WIT layer already types folds so attention-only ALGORITHMS cannot touch
them. This defect lived below that net: the KV index stores untyped pages,
and grafting pages does not syntactically touch a fold — it just silently
fails to bring one along. Any state-reuse feature must answer "can this
model class's state actually be restored here?" — for a hybrid, KV alone is
not the state.
