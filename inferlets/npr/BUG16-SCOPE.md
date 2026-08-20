# Engine bug 16 — scoping the silent whole-engine deadlock

*Written 2026-08-20 by `pr-split`. **This is a scoping document, not a fix, and
not a reproduction.** Everything below is derived from reading the runtime on
`npr-inferlet`; nothing here has been observed in a controlled run by this
session. Section 6 pre-registers the predictions that would settle it, in the
sense of DESIGN.md's measurement-hygiene item 4 — written down before the run,
so a null is evidence rather than a shrug.*

DESIGN.md §16 records the observation. This document asks the next question:
**what, mechanically, makes the stall permanent?** The answer offered is a
*structural* one — a set of liveness holes that are provable from the code —
plus an explicit statement of which of them actually fired, which is **not yet
known**.

---

## 1. The observation, restated

From DESIGN.md §16 (A40, ~6,700-page pool ≈ 214k tokens, 24 concurrent runs
each growing toward ~28k generated tokens; stalled ~40 minutes in):

- GPU utilization 0%; harness alive; **zero** errors, panics, warns or rejections;
- gateway RPC heartbeats keep flowing — the server looks healthy from outside;
- **a fresh 1-page, 5-token request on a brand-new context hangs forever.**

That last point is the important one. It is not per-context starvation. Whatever
the state is, it denies service to a request that needs almost nothing.

The standing hypothesis in §16 was "classic allocation deadlock in `drain_queues`,
FIFO head-of-line, eviction economics fail to break it". That hypothesis is
**partly right and materially incomplete**: the head-of-line blocking is real
(§3.1), but the reason the economics don't break it is more specific and more
interesting than "evidently do not" (§3.4, §3.5).

---

## 2. The state machine, as built

Three structures matter.

**`alloc_queue`** (`VecDeque<ContextId>`) — contexts with a deferred operation
waiting for GPU pages. Drained FIFO by `drain_queues` phase 1.

**`restore_queue`** (priority heap by bid) — Suspended/Stashed contexts waiting
to come back. Drained by `drain_queues` phase 2.

**`deferred_ops`** (per context) — the closures to replay once pages arrive.

The only place a page shortage is *resolved* rather than *waited on* is
`when_allocated_inner` (`runtime/src/context/sched.rs:717`), which runs once per
allocation request and proceeds: residency check → `num_pages == 0` fast path →
priority gate → free pool → **eviction loop** → park or self-suspend.

---

## 3. Five structural facts

Each is a single-line fact about the code, cited, and each is independently
checkable.

### 3.1 `drain_queues` phase 1 never escalates to eviction

```rust
// runtime/src/context.rs:2031
while let Some(&front_ctx_id) = self.alloc_queue.front() {
    ...
    if n > 0 && self.gpu_stores[driver_idx].available() < n {
        break;
    }
```

The queue head is served **only** if the free pool already covers it. There is
no attempt to make room. `find_eviction_victim` has exactly one call site in the
entire runtime — `sched.rs:795`, inside `when_allocated_inner` — so a context
parked in `alloc_queue` is waiting on a *purely passive* condition: someone
else, elsewhere, must happen to free enough pages. Nothing will ever try to free
them on its behalf.

This also means the classic mitigation for FIFO head-of-line (skip the head and
serve a smaller request behind it) is not merely absent — the `break` makes the
queue strictly non-work-conserving by design.

### 3.2 Restores are gated on the alloc queue being empty

```rust
// runtime/src/context.rs:2058
if !self.alloc_queue.is_empty() {
    return;
}
```

One un-servable head therefore freezes **all** restores, for every context, on
every driver. And a suspended context's `deferred_ops` replay only on restore
(`suspend` at `sched.rs:1175` removes it from `alloc_queue` and leaves the ops
on the context). So a single stuck entry converts the entire suspended
population into a set that can never make progress.

### 3.3 Restores have a second hard gate

```rust
// runtime/src/context.rs:2070
utilization > self.restore_pause_at_utilization
```

Default `0.85` (`bin/pie/src/ops/config/template.rs:101`). Under sustained
oversubscription, utilization sits pinned near capacity, so phase 2 returns
early even when the alloc queue *is* empty. The two gates are independent; the
absence of one does not imply progress.

### 3.4 The escape hatch is powered by the thing that stopped

`find_eviction_victim` (`sched.rs:881`) rejects a candidate unless
`ctx.defaulted || ctx.bid <= requester_bid` (`sched.rs:912`). `defaulted` is the
system's answer to "everyone is holding pages and nobody will yield": a context
whose owner can't pay rent becomes evictable *regardless of bid*.

`defaulted` is written in exactly one place — the market tick
(`sched.rs:545`) — and `Message::Tick` has exactly one sender in the runtime:

```rust
// runtime/src/inference/scheduler.rs:1499  (immediately after execute_batch)
crate::context::tick(driver_idx, latency.as_secs_f64(), batch_ctx_ids);
```

**The market clock advances only when a batch actually fires.** So the moment
the engine reaches a state where no batch can be formed, rent stops being
charged, no balance is ever depleted, nobody is ever flagged `defaulted`, and
the only mechanism that can break a bid standoff is frozen — *by the same
condition it exists to resolve.*

This is the property that makes the state **absorbing** rather than merely slow.
Whatever gets the engine into a no-batch state keeps it there.

### 3.5 Idle page hoarding is free

Even when the clock does tick, rent is charged only to contexts that were in the
batch:

```rust
// runtime/src/context/sched.rs:500
for &ctx_id in batch_ctx_ids {
```

A context holding thousands of pages while parked — not batched, not computing —
**pays nothing**, so its balance never falls, so it never defaults, so
`sched.rs:912` keeps protecting it from any lower-bidding requester. The
economics charge for *compute*, but the scarce resource under contention is
*residency*. Precisely the contexts that cause the deadlock are the ones the
rent never reaches.

### 3.6 (Corollary) `pending_suspend` narrows the victim pool

`find_eviction_victim` skips candidates already marked `pending_suspend`
(`sched.rs:906`), a flag set on a *pinned* victim and cleared only inside
`suspend()` (`sched.rs:1171`), which for a pinned context runs at `unpin`
(`context.rs:2005`). Any context that stays Pinned keeps the flag, and is
permanently invisible to eviction. This is not itself a deadlock — it is a way
for the eligible-victim set to shrink to empty while pages are still held.

---

## 4. How they compose

A candidate absorbing state, consistent with every element of the §16 signature:

1. Sustained oversubscription drives contexts through `when_allocated_inner`'s
   eviction loop. Some park in `alloc_queue` (step 5, `has_deferred`); some
   self-suspend into `restore_queue` (step 6, no victim).
2. At least one `alloc_queue` head needs more pages than the free pool holds.
   By **3.1** nothing will ever try to make room for it.
3. By **3.2** no restore can run while that head sits there, so every suspended
   context's deferred op is frozen too.
4. No forward pass is submitted, so no batch fires → by **3.4** the market clock
   stops → nobody defaults → by **3.5** the parked page-holders were never going
   to default anyway.
5. A *new* request arrives. It reaches the eviction loop, but every remaining
   on-GPU page-holder is either `pending_suspend` (**3.6**) or protected by
   `bid > requester_bid` with defaulting frozen (**3.4**). No victim → step 6 →
   self-suspend → `restore_queue` → never restored (**3.2**/**3.3**).
   **This is the "fresh 1-page request hangs forever" observation.**

Every step is a deferral. No step has an error path. That is why it is silent:
the engine is not failing, it is waiting, correctly, forever.

---

## 5. What is established and what is not

**Established** (code reading, cited above, independently checkable):

- E1. `alloc_queue` has no escalation to eviction; `find_eviction_victim` has one
  call site. — `context.rs:2031`, `sched.rs:795`
- E2. A single un-servable alloc-queue head blocks all restores. — `context.rs:2058`
- E3. Restores are additionally gated at 0.85 utilization. — `context.rs:2070`
- E4. The market clock advances only after `execute_batch`. — `scheduler.rs:1499`
- E5. Rent, and therefore defaulting, applies only to in-batch contexts. — `sched.rs:500`
- E6. `suspend()` always frees GPU pages, whether or not the CPU spill
  succeeds — `free(&working)` at `sched.rs:1143` and `release(&committed_hashes)` at `sched.rs:1160` are both outside the `cpu_offload` guard. *(This refutes an attractive alternative: the
  deadlock is not "suspension failed to free anything".)*

**Not established.** Which of these actually fired on the A40 is **unknown**, and
E1–E5 are individually sufficient to explain a stall, so "they're all true" is
not the same as "here is what happened". Three live candidates:

- **M1 — alloc-queue head starvation.** One entry demands N pages the pool never
  passively reaches; E1+E2 do the rest. Sub-variant worth noting: the eviction
  loop breaks out on `free_now + deferred_pages >= num_pages` using
  `count_reclaimable`, a **snapshot** (`sched.rs:802`, tested at `sched.rs:825`). `count_reclaimable`
  correctly excludes shared prefixes, but NPR forks constantly, so a fork landing
  between the count and the suspend raises a prefix's refcount and those pages
  never arrive. The requester then waits on a number that was true when it was
  taken.
- **M2 — empty eligible-victim set.** Every on-GPU holder is `pending_suspend`
  or bid-protected, so new requesters self-suspend into a restore queue that E2
  or E3 has frozen. Requires at least one context to stay Pinned indefinitely,
  which is itself unexplained.
- **M3 — restore pause latch.** Utilization stays above 0.85 because the
  remaining holders are parked rather than progressing, so phase 2 never runs
  even when the alloc queue drains.

M1/M2/M3 are not exclusive, and E4 says that once *any* of them stops batching,
the others become unrecoverable. The interesting question is which one is the
*entry* condition, because that is what a fix has to target.

**Also unknown:**

- Whether bids are heterogeneous in this workload at all. NPR never calls
  `Context::bid`, so all contexts should carry the default — in which case
  `bid > requester_bid` at `sched.rs:912` is never true and the bid gate is not
  the barrier, leaving `pending_suspend` (**3.6**) as the only way the victim set
  empties. **This is cheap to check and would eliminate a whole branch of the
  tree.** It has not been checked.
- Whether any context was still Pinned at the time of the stall (required by M2).
- Whether the `count_reclaimable` snapshot was ever actually invalidated (M1's
  sub-variant) or is a mechanism-shaped story of the exact species DESIGN.md's
  hygiene list warns about. It is offered as a *candidate*, not a finding.

---

## 6. Discriminating experiments — predictions registered in advance

All of these are **CPU-only, no model, no GPU, no pod**. `runtime/tests/` already
carries a mock-GPU end-to-end harness with real WASM inferlets
(`runtime/tests/contention.rs`: 2 GPUs × 8 pages, `page_size = 16`; config in
`runtime/tests/common/env.rs`). That is the right vehicle — it removes the model
entirely, which the §16 note's "cheap and CPU-portable" plan did not.

The one gap: the `generate` test inferlet hardcodes `max_tokens: 5` and ignores
its input, so it cannot produce the *long-growing* contexts the bug needs. A
`grow` inferlet taking `(prompt_pages, steps)` from its input is the missing
fixture, and building it is the first task, not an afterthought.

| # | Experiment | Prediction if **M1** | Prediction if **M2** | Prediction if **M3** |
|---|---|---|---|---|
| X1 | Instrument (§7) and re-run the wedge in the mock harness. | `alloc_queue` non-empty with a head whose `num_pages > available()`, `restore_queue` non-empty, `available()` stable > 0 | `alloc_queue` **empty**, `restore_queue` non-empty, all on-GPU ctxs `pending_suspend` | both queues non-empty, utilization pinned > 0.85, `restore_rejections` climbing |
| X2 | Raise `restore_pause_at_utilization` to 1.0. | no effect | no effect | **wedge clears or moves** |
| X3 | Make phase 1 work-conserving (skip an un-servable head, serve smaller requests behind it) — *diagnostic only, not a proposed fix*. | **wedge clears** | no effect | no effect |
| X4 | Log `ctx.bid` for all contexts at wedge time. | uniform | uniform ⇒ bid gate innocent, `pending_suspend` is the barrier; non-uniform ⇒ bid gate is live | uniform |
| X5 | Drive `tick` from a timer instead of `execute_batch` — *diagnostic only*. | wedge persists but rent accrues and defaulting eventually frees victims ⇒ **self-heals slowly** | same | same |

X5 is the one that tests §3.4 directly, and its prediction is the same for all
three mechanisms — which is exactly why it is worth running: it separates "what
started the stall" from "what made it permanent". If X5 self-heals, the
permanence is the clock, whatever the entry condition was.

**Pre-registered negative:** if X1 shows `alloc_queue` empty *and* utilization
below 0.85 *and* an eligible victim available, then every mechanism in this
document is wrong and the cause is somewhere I have not looked — most likely in
`inference/scheduler.rs` batch formation rather than in the context actor at all.
That outcome is a result, not a failure, and it should be recorded as one.

---

## 7. The observability gap is itself a finding

`SchedCounters` (`context.rs:1154`) already tracks precisely what X1 needs:
`eviction_suspends`, `priority_gate_suspends`, `no_victim_suspends`, `restores`,
`restore_rejections`, `defaults_flagged`, `eviction_searches`.

**Nothing ever prints them.** The only dump site is commented out:

```rust
// runtime/src/context.rs:2727
// if self.sched_counters.ticks % 1000 == 0 { ... }
```

This is why the 40-minute A40 wedge produced an evidence log with nothing in it
about scheduler state, and it is why §16 had to end at "prime suspect
(unconfirmed)". The minimum change that makes the *next* occurrence
self-diagnosing — worth doing before any fix, and independently of one:

1. A periodic dump of `SchedCounters` plus queue depths, free pages, and
   utilization per driver.
2. A stall detector: if `drain_queues` has run N times with a non-empty
   `alloc_queue` and a head that has not moved, log the head's `ctx_id`,
   `num_pages`, `available()`, and the eligible-victim count **once**, at warn.
   A deadlock that announces itself is a bug report; this one was a mystery for
   40 minutes because it announces nothing.

Note that (2) is *not* a fix and must not be sold as one — it converts a silent
hang into a loud one. That is a large improvement and a small change.

---

## 8. Shape of a fix — deliberately not proposed here

Recorded only so the scope is legible; each has a cost that needs thought, and
none should be written before X1 names the entry condition.

- **Work-conserving phase 1** (X3): serve requests behind an un-servable head.
  Cheap, but changes fairness and may starve large requests indefinitely — the
  standard mitigation is a reservation for the head, which is more machinery
  than it first looks.
- **Escalate from the queue**: let phase 1 attempt eviction for a head that has
  waited too long. Directly closes E1, but re-entrancy into the eviction loop
  from inside a drain needs care.
- **Decouple the market clock from batch execution** (X5): closes E4, and is the
  only candidate that restores the *designed* escape hatch rather than adding a
  new one. Requires deciding what rent means when nothing is computing.
- **Charge residency, not just compute** (E5): the deeper fix, and the largest
  change — the current market prices the wrong resource for this failure.

---

## 9. Status

Mechanism: **structurally characterised, not identified.** Five liveness holes
are established from the code; which one fired is unknown; three candidates and
the experiments that separate them are above, with predictions registered before
any run. No reproduction has been attempted by this session. No patch is
proposed.

The single most valuable next step is **not** a fix — it is §7's instrumentation
plus X1, because every remaining question in §5 is answered by one wedged run
that can talk.
