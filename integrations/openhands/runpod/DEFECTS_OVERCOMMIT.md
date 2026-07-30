# Engine defects under KV overcommit — found 2026-07-30, repro in-tree

Both found with `50_run_repro.sh` (~4 min each) after c8 arms A/B/C all
failed 8/8 regardless of inferlet policy. Both are UPSTREAM of bid/suspend
policy. GitHub issues are disabled on this fork, so this file is the record.

## 1. Alloc-path livelock (repro run 1, exit 2)

8 live-context sessions, unique ~28k-token histories, 124k-token pool,
swap_pool=4096. Session 8's FRESH context (`live-fresh`, not a restore)
starves >120 s in its first fill while 7 idle parents sit resident at bid
1e-12 — ideal victims the eviction loop never evicts.

Suspects: `runtime/src/context/sched.rs` eviction loop (steps 2–6) not
victimizing Active idle contexts for a fresh/off-GPU requester, or
alloc_queue deferral with no DrainKick on page-free.
`PIE_SCHED_DEBUG=1 PIE_SCHED_DEBUG_SECS=5` dumps head need-vs-free.

## 2. KV growth race (repro run 2, exit 1)

    GenStep::execute forward: KV_INVARIANT_VIOLATION ctx=15
    total_kv=15649 page_capacity=15648 num_pages=489 kv_len=15648
    num_input=1 page_size=32

Off by one token at a page boundary: a decode step that grows KV across a
page boundary under pool exhaustion EXECUTES without its new page instead
of waiting for the deferred alloc. Nondeterministic vs defect 1 — race.
Log: logs/repro_pie_serve_20260730_214033.log (pod-local).

## Benchmark-scale evidence

c8, H100 80GB, overcommit toml: Arm A (snapshots+swap) 8/8 timeouts;
Arm B (live+wake-bid 1.0) 8/8 after 6–9 healthy calls each; Arm C
(live+idle-suspend) same. All ~900 s ConversationRunError. c1 smoke of
live mode: clean (93.8% reuse, 0 errors).

## RESOLUTION (2026-07-30, ~23:25) — repro passes 12/12 rounds at ~1.9x overcommit

Five fixes, each forced by a repro measurement:
1. SDK counter resync before reservation sizing (defect 2) — b7b872a4.
2. Eviction requires victims bidding STRICTLY below the requester
   (sched.rs): equal-bid eviction produced whole-context evict/replay churn
   per decode step; strict inequality makes mid-turn peers inviolable and
   yields turn-slot round-robin.
3. Live-mode wake bid (1.0) set on ALL requester paths incl. fresh/rebuilt:
   a bid-0.0 fresh fill evicted under pressure could never evict back in.
4. Live mode destroys the per-turn child fork explicitly: plain drop only
   collects at instance exit, which a daemon never reaches (leaked ~1
   context/turn; wedged when the pool no longer fit the rotation).
5. SDK Context::destroy() forgets the raw handle after the WIT destroy:
   the handle's own resource-drop double-deleted host-side and TRAPPED the
   instance (the old phase-2 comment was right).

ENGINE DEBTS still open: suspend() frees working pages without refcounting
(a fork sharing the parent's tail working page loses it — masked now by
park-after-generation ordering, not fixed); the restore reject-loop spins
hot instead of backing off; equal-bid zero-bid legacy flows now cannot
evict each other at all (acceptable: they couldn't before either).
