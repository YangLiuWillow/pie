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
