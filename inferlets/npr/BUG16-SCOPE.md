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

### 3.4 The escape hatch stops when the engine stops

> **Correction, 2026-08-20, after running the harness.** An earlier revision of
> this section claimed the escape hatch was *unreachable* — that rent was
> identically zero because every context sits at `bid = 0.0`. **That was wrong,
> and the first run of the mock harness refuted it in 85 milliseconds.** The
> claim rested on a bad grep: I searched `sdk/rust/inferlet/src/context.rs` for
> `fn bid`, found nothing, and concluded the SDK could not bid. The method is
> `set_bid` (`context.rs:243`), and more importantly **the `Generator` auto-bids
> on every step** — `compute_bid` at `context.rs:51`, called from
> `generation.rs:98` at construction and `generation.rs:447` from the step loop
> (`generation.rs:316`). Measured evidence is in §6.1. The refuted claims are
> retained as **R1** in §5 rather than deleted, because what they got wrong is
> more instructive than what they got right.

What survives is (a) below. What does not is the "rent is always zero" half.


`find_eviction_victim` (`sched.rs:881`) rejects a candidate unless
`ctx.defaulted || ctx.bid <= requester_bid` (`sched.rs:912`). `defaulted` is the
system's answer to "everyone is holding pages and nobody will yield": a context
whose owner can't pay rent becomes evictable *regardless of bid*.

Two independent facts make that flag unreachable for this workload.

**(a) The clock only runs when the engine does.** `defaulted` is written in
exactly one place — the market tick (`sched.rs:545`) — and `Message::Tick` has
exactly one sender in the runtime:

```rust
// runtime/src/inference/scheduler.rs:1499  (immediately after execute_batch)
crate::context::tick(driver_idx, latency.as_secs_f64(), batch_ctx_ids);
```

So the moment no batch can be formed, rent stops being charged and the only
mechanism that can break a standoff is frozen *by the same condition it exists
to resolve*. That is what makes the state **absorbing** rather than merely slow.

**(b) The market is live, and that matters for what a fix can assume.** The
clearing price is the **minimum** bid among GPU-resident contexts on the driver
(`sched.rs:464`, `sched.rs:483`), and bids are neither zero nor uniform: the
`Generator` sets a truthful budget-exhausting bid at construction and recomputes
it every step,

```text
bid = (B/μ + d) / (p + μ(1 + cv²) / (2s))      // sdk/.../context.rs:51
```

with `B` = credit balance, `μ` = expected remaining steps, `p` = pages held,
`s` = page size. Two consequences worth holding onto:

- **The bid falls as balance falls and as pages held rise.** The long-running,
  page-heavy contexts — precisely the ones creating the pressure — drift toward
  the *lowest* bids, which makes them the preferred eviction victims and the
  first to default. The market is aimed correctly at this failure.
- **A parked context stops recomputing.** `recompute_bid` is driven from the
  Generator's step loop (`generation.rs:316`), so a context waiting on pages
  holds a stale bid while everyone still running updates theirs. Whether that
  matters is untested — flagged, not claimed.

So the escape hatch is real and does fire (§6.1 measures six defaulting events
in an 85 ms run). Its weakness is **(a)**: it is clocked by `execute_batch`, so
it works right up until the moment it is needed and then stops.

### 3.5 Idle page hoarding is free

A third gap, which would matter if (b) above were fixed by giving contexts real
bids: rent is charged only to contexts that were in the batch,

```rust
// runtime/src/context/sched.rs:500
for &ctx_id in batch_ctx_ids {
```

so a context holding thousands of pages while parked — not batched, not
computing — pays nothing even under a non-zero clearing price. The economics
charge for *compute*; the scarce resource under contention is *residency*.
Precisely the contexts that cause the deadlock are the ones the rent never
reaches.

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
- E7. The clearing price is the **minimum** resident bid on the driver
  (`sched.rs:464`, `sched.rs:483`), so the most-exhausted context sets the rent
  for every other context sharing the driver.
- E8. Bids are **auto-set and heterogeneous**: `Generator` calls `compute_bid`
  at construction (`generation.rs:98`) and again every step
  (`generation.rs:316` → `:447`), giving
  `bid = (B/μ + d) / (p + μ(1+cv²)/(2s))`. Both bid comparisons are therefore
  live, not vacuous.

**Refuted, and kept on the record:**

- **R1 — "rent is identically zero, so `defaulted` can never be set, so the
  market layer is inert in both directions."** This was E7/E8 in the revision of
  this document committed at `cf204689a`, and it was **wrong**. It rested on a
  grep for `fn bid` in the SDK's `context.rs` that missed `set_bid`
  (`context.rs:243`) and missed the `Generator` auto-bid entirely. The first run
  of the mock harness refuted it in 85 ms: six defaulting events, bids spanning
  ~870× (§6.1). The correction matters beyond bookkeeping — it moves the whole
  weight of the economic argument onto E4, and it puts the bid gate and the
  priority gate *back* into the candidate tree after I had eliminated them.

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
- **M2 — empty eligible-victim set.** New requesters find no victim, self-suspend,
  and land in a restore queue that E2 or E3 has frozen. Three filters can empty
  the set, and **R1 restored the third one after I had wrongly ruled it out**:
  `pending_suspend` (requires a context stuck Pinned — unexplained), `is_off_gpu`,
  and **bid protection** (`ctx.bid > requester_bid`, live per E8). The bid-protection
  route is now the more plausible of the two: a fresh context has a *high* bid
  (full balance, no pages held), so it should evict easily — but a fresh context
  is also the one observed to hang, so if bid protection is the barrier the
  arithmetic has to be checked, not assumed.
- **M3 — restore pause latch.** Utilization stays above 0.85 because the
  remaining holders are parked rather than progressing, so phase 2 never runs
  even when the alloc queue drains.

M1/M2/M3 are not exclusive, and E4 says that once *any* of them stops batching,
the others become unrecoverable. The interesting question is which one is the
*entry* condition, because that is what a fix has to target.

**Also unknown:**

- ~~Whether bids are heterogeneous in this workload.~~ **Settled by measurement,
  in the opposite direction to the first answer.** They are heterogeneous and
  auto-set (E8, R1, §6.1). The grep that said otherwise was wrong; the run that
  said so took 85 ms. Nothing in this document's candidate tree should be pruned
  on a grep again without a run behind it.
- **Whether any context was still Pinned at the time of the stall.** Required by
  M2, and now the load-bearing unknown: with E8 removing bid protection,
  `pending_suspend` is the only way the victim set empties, and it persists only
  while a context stays Pinned. *Why* a context would stay Pinned indefinitely is
  not explained by anything in this document — the answer, if M2 is right, is
  probably in `inference/scheduler.rs` batch formation rather than in the context
  actor.
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
| X4 | Observe `ctx.bid` spread and defaulting events under load. | — | — | — |
| X5 | Drive `tick` from a timer instead of `execute_batch` — *diagnostic only*. | wedge persists; rent keeps accruing, defaulting frees victims ⇒ **self-heals slowly** | same | same |
| X6 | Log `state` for every context at wedge time (Active / Pinned / Suspended / Stashed) and the eligible-victim count. | some Active with pages, none needed | **≥1 context stuck Pinned** *or* every holder bid-protected; eligible-victim count 0 | mixture, utilization > 0.85 |

**X4 has been run — see §6.1. It refuted the answer I had reached by grep**, and
with it E7/E8 as first written (R1). X5's prediction is now back to its original
form, because rent does accrue: a timer-driven clock really would keep defaulting
alive through a stall. That prediction has been rewritten twice, once wrongly, and
both revisions are visible above on purpose.

X6 remains the cheapest high-value experiment, and R1 widened it: it must now
distinguish *two* ways the victim set can empty (a stuck-Pinned context versus
bid protection), where the previous revision had wrongly reduced it to one.

### 6.1 First harness run — no wedge, and one refutation

`runtime/tests/bug16_wedge.rs` on branch `scratch/bug16-repro` (not merged): mock
GPU, no model, 1 driver × 32 pages, `page_size = 16`, 6 concurrent `grow`
contexts each reaching ~32 pages — **6.0× oversubscribed**, the same ratio as the
A40 observation.

**Result: no wedge.** All six branches completed in **82 ms**. Final counters:

```text
ticks=1540  evict_susp=5  prio_gate_susp=0  no_victim_susp=0
restores=5  restore_rej=10  defaults=6  evict_searches=0  drains=5
```

What that establishes and what it does not:

- **The contention machinery engages and resolves at this scale.** Five
  evictions, five restores, ten restore rejections — the paths under suspicion
  are all exercised, and they recover. So the oversubscription *ratio* alone is
  not sufficient; the A40 wedge took ~40 minutes, and this run took 82 ms. Scale,
  duration, or per-step timing is doing work that this parameterisation does not
  capture. **No conclusion about the mechanism can be drawn from this run.**
- **`defaults=6` is what refuted R1.** Instrumenting the defaulting branch printed
  balances, payments and bids directly:

  ```text
  [BUG16-DEFAULT] ctx=5 balance=0.00501 payment=0.03009 clearing_price=0.00094 bid=0.00094
  [BUG16-DEFAULT] ctx=2 balance=0.02087 payment=0.05165 clearing_price=0.00161 bid=0.00161
  [BUG16-DEFAULT] ctx=1 balance=23.98009 payment=26.07440 clearing_price=0.81482 bid=0.81482
  ```

  Bids span ~870× across six contexts in a single run, the clearing price tracks
  the minimum bid exactly, and contexts do default. The market is live.
- **`prio_gate_susp=0`** — the priority gate did not fire here. That is an
  observation about this run, not a property of the system; with heterogeneous
  bids it *can* fire, which is why X6 still has to test for it.
- **`evict_searches=0` alongside `evict_susp=5` — the counter is dead.**
  `sched_counters.eviction_searches` is declared and never incremented anywhere
  in the runtime. Minor, but it is one of the seven counters §7 proposes to dump,
  and a dumped counter that is always zero is worse than no counter at all.

Next parameterisations to try, in order of expected information: many more
concurrent contexts (24, matching the A40), far longer runs, a slower mock
backend so that rent and bid dynamics have real time to evolve between
allocations, and a smaller pool relative to a single context's footprint.

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
- **Decouple the market clock from batch execution** (X5): closes E4, and with
  R1 corrected this is now the *strongest* economic candidate — rent does accrue
  and contexts do default, so a clock that keeps running through a stall keeps
  the escape hatch alive. Requires deciding what rent means when nothing is
  computing.
- **Charge residency, not just compute** (E5/§3.5): a parked page-holder pays
  nothing even under a live market, so the thing being priced is not the thing
  that is scarce. Larger change, deeper fix.
- **Revisit the min-bid clearing price** (E7): the most-exhausted context sets
  the rent for everyone on the driver. Not implicated in any current candidate —
  listed because it surprised me and may surprise the next reader.

---

## 9. Status

Mechanism: **structurally characterised, not identified.** Eight facts (E1–E8)
are established from the code; one earlier claim (R1) was established from the
code and then **refuted by measurement**; which mechanism actually fired is
unknown. Three candidates and the experiments that separate them are above, with
predictions registered before each run.

A reproduction harness now exists (`scratch/bug16-repro`, unmerged): mock GPU, no
model, no CUDA, plus a `debug_snapshot` probe that answers through the actor
mailbox — which works precisely because this class of deadlock leaves the actor
healthy. **It does not yet reproduce the wedge** at 6× oversubscription; §6.1 has
the numbers and the next parameterisations to try. No patch is proposed.

The most useful thing to carry forward is the shape of the error, not the
conclusion:

- I eliminated a whole branch of the candidate tree — the bid gate, the priority
  gate, and defaulting — on the strength of a grep, wrote it up as two
  established facts, and committed it. The first run of the harness refuted it in
  85 milliseconds. The grep was for `fn bid` in the SDK; the method is `set_bid`,
  and the `Generator` auto-bids on every step, so the thing I declared impossible
  happens six times in a tenth of a second.
- DESIGN.md's hygiene list already names this species: an untraced,
  mechanism-shaped story that survives because it is coherent. Being derived from
  code rather than from imagination is **not** the protection it feels like — a
  grep is a measurement, and it has a false-negative rate.
- The practice that caught it was cheap: build the harness, run it, print the
  numbers that the story says cannot exist. Eighty-five milliseconds of engine
  time against a claim that had already been committed to two remotes.

The single most valuable next step is still §7's instrumentation plus a
parameterisation that actually wedges. Every remaining question in §5 is answered
by one wedged run that can talk — and, on the evidence above, by very little
that is not a run.

The single most valuable next step is **not** a fix — it is §7's instrumentation
plus X1, because every remaining question in §5 is answered by one wedged run
that can talk.
