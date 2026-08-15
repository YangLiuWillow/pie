# Where pie's 2.3x throughput gap comes from

pie 88.2 tok/s against vLLM-metal 206.9 and mlx-lm 203.8, 8 concurrent, 32
requests, `max_tokens=128`. First look, from data already on disk plus one
source trace — no new benchmark run.

## The arithmetic says pie is serializing

`decode-rows-probe` (results-speculation.md §1) fits a decode fire's cost on the
real driver:

    weight half   10.19 + 2.49 * rows   ms
    KV half        0.464 + 0.703 * rows  ms per 1k of context

The throughput bench runs ~0.5k of context, so:

| configuration | predicted |
|---|---:|
| 1 row, x8 serialized | ~106 ms per token per stream |
| 8 rows in ONE fire | ~33 ms for 8 tokens = **241 tok/s** |
| **measured** | 90 ms per token per stream = **88.2 tok/s** |

90 against a serialized 106 and a batched 33. **pie is serializing the eight
concurrent decodes**, give or take. vLLM's 206.9 works out to ~4.8 ms per token,
which is only reachable by batching.

## Why it serializes — and it is not a scheduling accident

`runtime/engine/src/pipeline/fire.rs:1458`:

```rust
req.device_resolved_geometry = decode_envelope.is_some();
```

**Every decode fire carries device-resolved geometry.** And the Metal driver
(`driver/metal/src/context.cpp:1006`) supports at most ONE such program per
batch:

    launch: N device-geometry programs in one batch (at most one is supported)

So two concurrent decodes can never share a fire on this driver. The scheduler
clause added in `LaunchGrouping::accepts()` did not create this ceiling — it
made the runtime honour a constraint the driver was already enforcing, which
until then had been failing *silently* (one token of a 64-token budget,
reported as a clean stop). Before the fix pie completed 8 of 32 requests; after
it, 32 of 32 at 88.2 tok/s. The fix bought correctness and did not cost
throughput, because the batching was never happening.

## The second ceiling, behind the first

Even if the driver allowed co-batching, the same probe says the win would be
modest: a k-row decode does NOT share its KV read. The slope is
`0.464 + 0.703*rows`, so only 40% of one row's KV cost is shared and each extra
row adds a full re-read of the cache — 140 GB/s against the ~203 GB/s the
weight read reaches. An 8-row fire costs 4.08x a 1-row fire at 16k context, so
batching 8 requests would return roughly 2x, not 8x.

At the bench's short context the weight term dominates and batching alone would
be worth ~3.2x, which would close most of the 2.3x gap. At agentic context
lengths, where the KV term takes over, it would not.

## Two fixes, and the second is already on the roadmap

1. **Let more than one device-geometry program share a batch** (driver), or stop
   requiring device-resolved geometry for decode fires (`fire.rs:1458`). This is
   the one that matters for the measured gap. UNEXAMINED: why the constraint
   exists, and what a decode envelope needs the device to resolve. That is the
   first thing to read.

2. **A decode kernel that shares the KV read across k rows.**
   `results-speculation.md` already specifies it: the per-row kernel's
   key-parallel decomposition, with each simdgroup computing all k rows against
   the keys it has already loaded — one KV read, k dot products per key. Decode
   is bandwidth-bound here so the extra arithmetic is close to free.

**The second fix serves both problems.** It is what speculative decoding needs
(a verify fire is few rows, one request, sharing a key span — the shape neither
existing kernel was written for) and it is what concurrent batching needs. One
kernel, two payoffs, and speculation is already implemented and waiting on it.

## CONFIRMED, without a GPU

The serialization claim is structural, so it does not need a serving run to
test — it needs an assertion. `runtime/engine/src/scheduler/worker.rs` now
carries one:

```
two concurrent decode fires MUST NOT co-batch
```

with the control that makes it mean something: two requests identical but for
the `device_resolved_geometry` flag DO co-batch. Without that control the
assertion passes whenever `accepts()` refuses for any reason at all — a full
batch, a token limit — and would look like proof of a ceiling that was not
there.

Both pass. **Two concurrent decodes cannot share a fire; two ordinary fires
can.** The ceiling is where the source trace said it was.

The benchmark confirmation is still owed: `require_quiet_gpu` refuses a 30B
serving run at 12.7 GB free against ~20 needed, with `pie-boN` holding 14.2 GB.
What that run would add is the 15% the cost model does not explain (90 ms
measured against 106 ms predicted), not the ceiling itself.

## Found while confirming: `scheduler::worker` tests are flaky

Not related to the throughput work, and worth someone's attention.

    without any change: 3 of 5 full-module runs FAILED
    with the assertion added: 1 of 5 FAILED

So the assertion is not the cause — the module is flaky at roughly 40% on a
full run. Two tests account for it, both concurrency-shaped by name:

  * `two_pipelines_coalesce_into_one_wave`
  * `close_defers_slot_retirement_until_outstanding_completion_drops`

Each failed once in six runs. This matters beyond tidiness: a suite that fails
40% of the time cannot gate anything, and the natural response to a red run is
to re-run it until it is green — which is how a real regression gets waved
through.

## What this does NOT say

No new measurement was taken; this is arithmetic over existing probe data plus
a source trace. The serialization hypothesis predicts 106 ms per token per
stream against 90 measured — close, but 15% off, so something else is also
moving. Confirming it needs `PIE_METAL_DISPATCH_TRACE` on a concurrent run to
count rows per fire directly, which needs a quiet machine.
