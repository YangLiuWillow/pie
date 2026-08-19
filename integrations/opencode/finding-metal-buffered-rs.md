# Metal's buffered recurrent state, measured — 2026-08-18

**Question (handover §3.3):** is B's sequential decode still forced, or does
Metal now support the buffer/discard verbs the interface was designed around?
The engine module doc claims Metal "validates `rs_fold_lens` and then never
reads it again... every fire folded regardless" — written when that was true,
never re-measured since.

**Instrument:** `runtime/engine/tests/inferlets/gdn-foldcommit` — the existing
fold-commit probe suite, previously exercised only against CUDA and the mock —
run mode-by-mode against a real Metal boot of Qwen3.6-35B-A3B via a small
client driver. Wasm at tests/inferlets target, driver in the session
scratchpad; every verdict below is from a clean boot (see the poison caveat).

## Verdicts

| shape | mode | verdict on Metal today |
|---|---|---|
| buffer k tokens WITHOUT folding, then commit the accepted prefix and abandon the rest | `4`/`2`/`0` | **WORKS** — committed=4/2/0, abandoned=0/2/4, greedy tokens consistent across arms |
| fold-through with a buffered tail kept live | `inside` | **WORKS** — `agree=yes` against solo references |
| append to a NON-EMPTY buffer (the buffer READ/replay path) | `chain` | REFUSED — `paged continuation: slot at 6, fire starts at 2`. Same gap as CUDA (`cuda_gdn_foldcommit.rs` pins it there) |
| fold-behind (next window folds previous prefix while writing its own) | `behind` | REFUSED — geometry (`token/position count mismatch or empty span`) |
| commit via a token-free row | `empty` | REFUSED — empty span |
| device-resident fold length (verify+commit fused on device) | `device` | REFUSED — `driver published RETRY at frame settle; retry is not a v14 outcome` |
| fold boundary interior to a fire's own tokens (2R-segment) | `interior` | REFUSED — geometry, its own reference arm carries an empty span |
| two RS rows, one folds while one buffers | `mixed` | **ENGINE BUG** — `launch is missing qo_indptr for a forward-needing member [wire geometry]`, and the driver publishes a **poison epoch**. Reproduced on a clean boot |

## Two corrections to the written record

1. **`engine.rs`'s module claim is stale.** `rs_fold_lens` IS respected on
   Metal now: a `fold_len=0` fire buffers without advancing the fold, and the
   `inside` parity run pins the states as correct, not merely unrefused. The
   buffer-and-discard scheme was abandoned for generate-on-fork on evidence
   that has since expired. (Generate-on-fork is still correct and simpler for
   RETENTION; this finding is about what else is now possible.)
2. **Sequential decode is no longer driver-forced.** The core speculation
   cadence is expressible on Metal today: one buffered verify fire (fold 0,
   k rows) → host accepts m → next fire commits `fold_len=m` and the tail is
   abandoned. The `chain` refusal shapes the cadence — the buffer must be
   committed/emptied before the next window buffers — but does not forbid it.

## Poison caveat (method note)

The first suite run executed all modes against one boot; `mixed`'s poison
epoch landed mid-suite and CONTAMINATED the later verdicts — `inside`
"failed" with disagreeing arms (271 vs 198) under the poison and PASSES on a
clean boot. Same lesson as finding-inferlet-killed §11: one poison epoch
quietly rewrites every measurement after it. Per-mode verdicts require
per-boot isolation once any mode poisons.

## What this does NOT settle

Whether buffered speculation is WORTH it for B's decode on this hardware.
The verify fire is multi-row, and the measured multi-row cost on Metal is
adverse (rows=8 = 4.1× a 1-row fire — liszt-ai-00's per-row KV re-read
finding). The capability boundary is now known; the arithmetic and the
drafting integration live in the decode loop liszt-ai-00 owns. Their 1.45×
adaptive-drafting number is from the attention model; the hybrid's viability
depends on the verify-fire row scaling they are already chasing.

## Open defect

`mixed` (two RS working sets in one fire, per-row persist mask): the engine
emits a launch with no `qo_indptr` and the driver poisons the epoch. Third
member of the poison-epoch-producer family. Single-conversation serving never
fires this shape; multi-request hybrid serving will.

---

## CORRECTION + the number, same day (hybrid-rows-probe)

**The "WORKS" verdicts above are RING-PATH verdicts.** `gdn-foldcommit`'s
prompt is "hello world" — a few tokens — so every mode ran the tiny-context
M=1 ring path. The first buffered fire at a REAL context (paged path,
ctx 7424, `hybrid-rows-probe`) exposed what the suite could not see:

**The paged position record counts buffered rows, and nothing rewinds it.**
`commit_paged_request_state` sets `resident_next_position` from the fire's
last position unconditionally — a `fold_len=0` fire of r rows advances the
record by r even though the fold did not move. `discard_buffered` empties the
store's buffer and never reaches the driver. Measured, one step at a time
(`cadence` mode):

```
window1 buffered(2) @7424   OK        (slot record -> 7426)
window1 commit(1)  @7424    REFUSED   (async: "slot 0 is at position 7426,
                                       this fire starts at 7424")
window2                     dead      (inherits the pipeline failure)
```

So at real contexts the buffer→commit speculation cadence is **unusable on
Metal today**, blocked not by compose.cpp's stated refusals but by paged
continuation bookkeeping that treats buffered rows as folded ones. Second
open defect from this probe (after the `mixed` qo_indptr poison). The fix
direction is driver-side: the record must track the FOLD boundary, or fold
and buffer extents separately, for `validate_linear_sequence_geometry` to
compare against.

**The rows curve exists anyway** — the timing loop advances its position per
sample instead of committing, which fires identical workloads without
tripping the record. ctx=7424, buffered verify fires, median of 10 after 2
warmups per shape:

| rows | median ms | ×fire(1) | beta(r) = (f(r)−f(1))/((r−1)·f(1)) |
|---|---|---|---|
| 1 | 12.07 | 1.00 | — |
| 2 | 27.32 | 2.26 | 1.26 |
| 3 | 31.36 | 2.60 | 0.80 |
| 5 | 38.64 | 3.20 | 0.55 |
| 8 | 50.32 | 4.17 | 0.45 |

Shape: leaving the 1-row path costs ~15 ms flat, then ~3.9 ms per additional
row. rows=2 LOSES to two sequential fires outright (27.3 vs 24.1 ms); only
wide windows can win, and at rows=8 the break-even is ≈52% of the full
window accepted (50.3 vs 8×12.07 ms), before commit costs. The curve's
per-row slope is close to Coder-30B's despite head_dim 256 vs 128; the
difference is the fixed multi-row penalty.

**Bottom line for §3.3:** capability half-open (ring path yes, paged path
blocked by the position record), and even once unblocked, speculation pays
off only at wide windows and ≥~50% window acceptance. Both the blocker and
the arithmetic are now specific enough to act on. Raw transcripts:
`hybrid-cadence-7424.txt`, `hybrid-rows-7424.txt`.
