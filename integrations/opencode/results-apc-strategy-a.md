# Automatic prefix caching in Strategy A

Strategy A serves one inferlet instance per request and drops it, so every turn
re-prefilled the whole conversation. On `django__django-13089` with Coder-30B
that cost 33.9 s, 63.3 s and 57.3 s on turns 3–5, `cached=0` on every one,
against 4.5–6.1 s for the equivalent turns under the session arm.

The fix does not need a long-lived process. `WorkingSet::update_index` /
`from_index` are ENGINE-level, so KV parked by one instance is found by the
next. That property was verified on this build before anything was built on it
(`explicit_prefix_index_survives_across_inferlet_instances`), because the
pre-existing test with the matching name publishes and looks up inside ONE
invocation and could not answer it.

## What was measured

Six opencode-shaped prompts, ~9.2k tokens, Qwen3-Coder-30B-A3B-4bit on Metal:

| | |
|---|---|
| reuse | median **99.9%** of prompt tokens (min 99.6%) |
| latency | median **16.5 s cold → 0.9 s warm** |

Reuse is reported as a fraction, never a hit flag. A turn that resumes on a
shallow cut and re-prefills most of the history still reports a hit; both
prefix-cache defects RatioThink shipped did exactly that, one of them behind a
25.7x TTFT regression. `usage.cached_tokens` carries the depth the engine
ACTUALLY reached — the generator refuses a resume whose page count disagrees
with its address and rebuilds, and that refusal reports 0.

Latency is a ratio on one machine and one prompt shape. It is not a throughput
claim, and it is not a correctness claim.

## Is the grafted answer the same answer?

Cold and warm disagreed on 3 of 6 prompts at temperature 0, which is ambiguous
in the worst way: a near-tie argmax flip and a graft onto a prefix that ends in
the wrong place both produce fluent, plausible, different text.

Two runs separate them.

**Floor.** Same six prompts, cold only, no cache, prefill chunking changed from
2048 to 1024: **5/6 byte-identical**. So chunk shape alone flips 1 in 6 at
greedy decoding, with nothing cached.

**The graft itself.** `runtime/engine/tests/inferlets/apc-graft-probe` removes
the chunk-shape difference and keeps only the question. HONEST prefills `0..cut`
in chunks, fires `cut..n`, decodes 32 tokens, and parks pages `0..cut/page`.
GRAFT loads those pages and fires the identical `cut..n`. Both compute the tail
with the same fire shapes over KV that is not merely equivalent but the same
bytes, so a correct graft must produce EQUAL output, not close output.

At n=9216, cut=4608, on the real driver and the real model:

```
APC_GRAFT_PROBE GRAFT_EXACT graft_agrees=33/33 wrong_prefix_agrees=0/33
```

The second number is the point. **Two earlier versions of that probe passed
while being incapable of failing**: pseudo-random token ids made the model emit
filler, and self-similar text made a prefix built from different notes produce
the same 33 tokens. Both looked clean. The probe only became evidence once the
answer lived solely in the parked half — a code planted before the cut, the
question after it — at which point the wrong-prefix arm diverged at token 0.

So the cold-vs-warm disagreement is chunk-shape arithmetic. The graft is exact.

## Park last, and not for style

Parking a slice BEFORE the decode made the decode's write copy-on-write, and
`pie_metal_copy_kv` returns `PIE_STATUS_UNSUPPORTED` for any checkpoint that is
not a GDN hybrid (`driver/metal/src/context.cpp`, `copy_kv_impl`: "this
increment only supports the qwen3.6 (GDN-hybrid) checkpoint geometry").
Qwen3-Coder-30B is pure attention, so the fire was rejected with
`pre-launch KV copy rejected: pie_metal_copy_kv failed with status -3` and the
turn returned **its first token only** — a fluent, plausible, truncated answer
that nothing at the wire level flagged.

Parking after every write leaves no fire after the sharing is introduced, so no
copy is ever planned. This is also why switching to a GDN hybrid to "get"
copy-on-write would be the wrong trade: it is not needed, it would change the
model under an A/B whose other arm ran Coder-30B, and `engine.rs` documents that
`run_ahead` folds speculative fires into recurrent state irreversibly, so saved
hybrid state can sit ahead of its own KV and still generate fluent text.

## Behaviour under pool pressure

The index does NOT evict by count — `store_index_count_does_not_evict_entries`
asserts 257 entries all survive. Parked prefixes accumulate. When the pool
cannot reserve pages the planner calls `reclaim_idle()` →
`drop_unused_cache_leases()` and retries (`planner.rs:2053`), which frees them
all rather than the oldest.

So the expected shape over a long conversation is a sawtooth: fast turns until
the pool fills, one flush, one full-price turn, fast again. Correct throughout —
the worst case is a cold turn, not a failed one.

## Defects found on the way

- **`run_pie_opencode.sh` served a wasm ten hours old.** It took the first live
  build candidate, which assumed `CARGO_TARGET_DIR` was set identically in the
  build shell and in the boot script. It was not. One boot and one probe run
  measured code that was never compiled. It now takes the NEWEST candidate and
  prints the build time on every boot, including the no-op refresh — the case
  where nothing else would tell you.
- **The e2e suite did not compile on this branch** (`ModelConfig` gained a
  `checkpoint` field the harness never set), and `add_and_install` raced when
  two tests shared an inferlet: `program::add(.., replace = true)` re-registers
  the component while a concurrent test is spawning it, giving
  `Component not found for program`. Installs are now once-per-process.
- **An over-long resume is refused, not silently wrong.** A deliberately skewed
  graft was rejected at launch with `a token at position 6144 has no page in its
  request's list`. Worth knowing: the geometry checks that exist are loud.

## End to end, on a real agent run

`django__django-13089`, stock opencode, Coder-30B, after the contention fix:

```
django__django-13089   ok   87 s   894-byte patch      driver faults: 0

[apc] e269a3e8 reuse  0.0% (0/1045)      from  4 cuts; parked 1   (title call)
[apc] b4c0e8dd reuse  0.0% (0/7712)      from 30 cuts; parked 1
[apc] f96cc4ce reuse 69.6% (7680/11040)  from 44 cuts; parked 1
[apc] d93fa111 reuse 95.4% (11008/11536) from 46 cuts; parked 1

prefix reuse 61.7% (18,688 of 30,288 prompt tokens served from parked KV)
model call latency p50/p95 19.7/48.9 s   time split 99% model
```

Each turn resumes exactly where the previous one parked — 7680 is call 2's
history aligned down, 11008 is call 3's. Reuse is bounded by how much the
conversation GREW, not by the cache: call 3 reused 69.6% because a large tool
result landed between the turns, and that is the correct number rather than a
shortfall.

The two independent measurements agree exactly: `ttb_summary.py` reads
`tokens.cache.read` from opencode's own database and totals 18,688, which is
7680 + 11008 from the server's log. Client and server are counting the same
thing.

NOT yet a speed claim against the no-cache arm. The trajectory differs — this
run made 3 model calls where the earlier baseline made 7 — so comparing total
wall clock would compare trajectories, not engines. Per-call latency and
in-call throughput are the comparable quantities, and a like-for-like A/B still
has to be run.

## The wedge, and why the first two diagnoses were wrong

Under real opencode traffic (which issues CONCURRENT requests, unlike every
probe here) the server wedged: every request refused with
`admission rejected: cluster saturated: no healthy worker has KV/seq headroom`,
persisting while the box was idle.

Two diagnoses were wrong before the right one:

1. "Parked prefixes exhaust the pool." Dismissed at the time on an occupancy
   argument that used **the wrong denominator**, and the argument was wrong even
   though the conclusion survived. I divided by `total_pages = 2048` and got
   11%. The driver does not use that number: `context.cpp:245
   effective_total_pages()` clamps the pool to
   `min(total_pages, ceil(max_model_len / kv_page_size))`, which at
   `max_model_len 16384` is **512 pages — 16,384 tokens, a quarter of what the
   config appears to ask for**. Against the real pool a single conversation's
   parked prefix runs 45–72%:

   ```
   parked  7,424 tokens = 232 pages = 45.3% of the pool
   parked 11,712 tokens = 366 pages = 71.5% of the pool
   ```

   So pool pressure was never absurd, and it is only the DRIVER LOG — a fence
   timeout and a poison epoch — that rules it out as the trigger here. This is
   the same clamp that starves the context ring below, and the same one this
   repo has now been caught by three separate times.
2. "A stuck allocation waiter." Closer — the `waiters` floor IS what pins the
   bucket at 240 — but still a symptom.

The driver log had the cause:

```
[pie-driver-metal] member forward rejected: Metal forward timed out before its completion fence
[pie-driver-metal] instance 3 launch failed: Metal M1 host readiness fault 512   (x5)
WARN pie::inferlet: channel is poisoned: driver published poison epoch 1
```

A forward missed its completion fence, the driver published a poison epoch, and
the pressure floor then pinned the gateway shut. `saturated` was the last link,
not the first.

What implicates this module: every `slice` and `update_index` takes the engine's
GLOBAL KV mutex, and parking 8 cuts is 16 exclusive acquisitions per turn on a
lock `fire.rs` keeps a single integer load off. Serial probes never saw it;
concurrent opencode traffic did. At `PUBLISH_MAX = 1` the run above completed
with zero driver faults.

That is one run either side of a one-variable change — suggestive, not proof.
The persistence is a separate fact worth keeping: nothing clears a poison epoch,
so ONE driver-side failure converts into a permanently refusing engine that
reads like a capacity problem.

## The two arms stop being comparable at the first cached call

Measured, not predicted. Same five instances, same source, `APC_OFF=1` for the
control (parks nothing, so nothing can hit):

```
instance 2, call 1:  apc  1333 (cached    0)   ctl  1333   same
            call 2:  apc  7998 (cached    0)   ctl  7998   same
            call 3:  apc  8172 (cached 7968)   ctl  8171   drift <- first resume
            call 4:  apc  8318 (cached 8160)   ctl  8316
            call 5:  apc  8461 (cached 8288)   ctl  8458
            call 6:  apc  8538 (cached 8448)   ctl  8605
            call 7:  apc 23974 (cached 8512)   ctl  8690   different conversations
```

Byte-identical while `cached = 0` in both arms; divergent from the exact call
that first resumes. The gap compounds 1, 2, 3, 67, then unbounded.

This is NOT a bad graft — the graft is bit-exact against an honest chunked
continuation, with a negative control that fails (see above). It is chunk shape:
a resumed turn computes its delta as ONE narrow fire where a cold turn computes
it inside wide even chunks, and greedy decoding flips on near-ties. The
cache-free floor for that effect is already measured — moving prefill chunking
from 2048 to 1024 with no cache at all flips 1 prompt in 6.

**Consequence for the benchmark.** Past the first cached call the arms are
solving different conversations, so a per-case outcome difference at n=5
measures trajectory variance, not the cache. Attributing task outcomes to APC
needs enough instances for that variance to average out; the arms cannot be made
numerically identical, because the cold arm chunks differently by construction.

Instance 1 shows the same thing from the other side: both arms ran identically
throughout and both hit the 20,061-token context wall.

## Stability, which IS answerable at n=5

Five instances, ONE server never restarted, 20 model calls, prefix reuse 57.4%
(128,320 of 223,696 prompt tokens):

```
fence timeouts 0   poison epochs 0   saturation 0   agent-error 0
server answered 200 after the suite
```

Parked prefixes from five different conversations accumulated across the whole
suite and degraded nothing. That is the question the soak was built to answer,
and it is the one claim here that n=5 supports.

## The context ring is now the binding constraint

Two of five instances died mid-trajectory at 20,061 and 23,974 prompt tokens
against a **16,384-token ring** — `max_model_len / kv_page_size` = 512 effective
pages, even though `total_pages = 2048` provides 65,536 tokens of KV. Instance 2
is the clearest: six productive calls at 97–99% reuse, then the wall.

Prefill cost used to be what made long agent conversations expensive. At 98%
reuse it mostly is not, and the ceiling moved to `max_model_len`. Raising it is
a config change, and it has to be raised for BOTH arms before any A/B, or the
comparison measures the ceiling rather than either engine.

## The A/B: same source, `APC_OFF=1` control, five instances each

```
SERVING — what the engine did with the work it was given
                                     APC off      APC on
  prefix reuse                          0.0%       57.4%
  model call latency p50 (s)            22.7        15.6
  model call latency p95 (s)            58.6        38.6
  model time, whole suite (s)          559.8       324.6

TASK — whether the agent got the job done
  non-empty patch rate                   40%         40%
  model calls (mean/case)                4.0         4.0
  agent-error / wall-exhausted          0 / 0       0 / 0
```

Per-instance, the SAME two instances produced patches in both arms, at nearly
the same size (13089: 869 B off / 882 B on; 14373: 412 B both). The other three
produced nothing in either arm.

**42% less time in the model for identical task outcomes.** That is the claim
this pair of runs supports.

Two things it does NOT support:

- `in_call_prompt_tok_s` reads 401.6 (off) against 293.8 (on), which looks like
  a regression and is not comparable: opencode's `input` counts only what the
  server actually processed, so the cached arm divides a much smaller numerator
  by a wall clock that still includes real prefill of the delta and all of the
  decode. Compare `model_time_s`, or compare rates only within an arm.
- `resolved` is null in both — the official grader needs Docker, and `colima`
  cannot start on this machine (`lima not found`). The TASK row is patch
  PRODUCTION. A non-empty patch is a weaker claim than a resolved instance and
  can move independently of it.

**The 2/5 is not the cache's doing.** An earlier pie baseline on these instances
was 4/5 RESOLVED; this pair sits at 2/5 non-empty in BOTH arms, so whatever
moved belongs to the harness, the context ceiling, or the model — not to APC.
Identifying it needs the grader.

## Does the numeric divergence change outcomes? Measured: no

Per-instance trajectory match between the arms:

```
inst 1   3/3 calls   identical throughout
inst 2   7/8 calls   diverges at call 3 (first resume): 8172 vs 8171
inst 3   5/4 calls   diverges at call 3: 7825 vs 11030
inst 4   6/6 calls   diverges at call 6: 11946 vs 12071
inst 5   4/4 calls   identical throughout
```

Divergence is real but PROBABILISTIC — two of five instances tracked exactly
end to end. And where it happened, it did not change the verdict: the same two
instances produced the same patches. So the drift is a reproducibility property,
not a quality one, at this sample size.

That lowers the priority of making the arms bit-identical (canonical
absolute-position chunk grid + resume cuts aligned to it, which
`apc-graft-probe` already demonstrates gives exact agreement). It buys
same-request-same-answer and a tighter A/B, at roughly 1.4 s per turn of
re-prefill; it does not appear to buy task quality.

## Raising the context ring traded one wall for another

`max_model_len` 16384 -> 65536 (the driver's own cap is 131,072 =
`kPhase1bRsSlots(64) * kMetalCtxTokensPerRequest(2048)`, so the 16k was purely
our config). Verified at the driver, not assumed: 20,014- and 28,014-token
prompts both complete with `finish=stop` where 16,384 refused them.

It worked, and it immediately exposed the next limit:

```
django__django-12276   372 s  (was  26 s)
django__django-13028   656 s  (was  61 s)
django__django-13089     2 s  <- died instantly
django__django-14373     2 s  <- died instantly
django__django-15569    21 s  <- died instantly
0/5 patches
```

```
[pie-driver-metal] register_program: Metal M1 program executable cache is full
```

`kMaxProgramCacheEntries = 64` (`driver/metal/src/pipeline/m1_runtime.cpp:68`).
Once full, `register_program` rejects and every later turn degrades to
`finish_reason:"length"`. Neither earlier run hit it — 0 events across 31 and 25
calls — because the ring was killing trajectories before they could compile
enough programs. **Raising the ring did not create this; it removed what was
hiding it.**

### Why one program per turn

The reservation size reaches the traced graph as a SHAPE CONSTANT, through the
decode epilogue's `reshape(&pids, [pool_pages_total])`. Computed exactly, it is
`(n + max_tokens + 2) / kv_page_size` — so 8,174 prompt tokens reserve 384
pages, the next turn's 8,319 reserve 388, the next 393. Three prompt lengths,
three shapes, three compiled programs. One per turn, and 64 turns ends the
server.

Fixed guest-side by quantising:

```rust
const POOL_GRANULARITY: u32 = 256;
let pool_pages = (n + max_tokens as u32 + 2)
    .div_ceil(page_t).next_multiple_of(POOL_GRANULARITY).max(have);
```

384, 388 and 393 all become 512. It costs nothing real — `reserve` is purely
logical, no memory is held until a forward writes, so an over-reservation is a
longer page-id list and nothing else.

**Not provably sufficient.** A prefill chunk's token count also reaches the
trace, and with the cache on, a resumed turn's delta is a different length every
turn. Whether that alone still fills 64 slots is being measured rather than
argued.

**And the durable fix is not in the guest.** A cache that REJECTS when full
instead of evicting converts a capacity limit into a dead server, and no guest
can be written that never needs a 65th shape. The quantisation buys headroom;
the driver wants an eviction policy.

## Still open

- Never exercised against real opencode traffic: every probe here was
  non-streaming and tool-free. Real requests carry tool schemas, tool-call and
  tool-result messages, and `stream: true`.
- `ttb_summary.py` does not yet report reuse fraction, so the run summary
  understates what the cache did.
