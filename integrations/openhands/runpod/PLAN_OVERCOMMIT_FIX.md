# Overcommit fix — live plan and state (2026-07-30, ~22:00 UTC)

Written mid-investigation so an SSH drop or context loss costs nothing.
Companion to `DEFECTS_OVERCOMMIT.md` (the defect record) and
`AGENT_HANDOVER_20260729.md` (the prior session). Repo state: everything
through `4fb3d367` is pushed to `origin/openhands-integration-updated`;
the park-after-generation inferlet fix (described below) is built and
installed on the pod but NOT yet committed — commit it once the repro
validates it.

## What the repro is and what it must show

`50_run_repro.sh` / `50_overcommit_repro.py`: 8 concurrent live-context
daemon sessions, UNIQUE ~28k-token histories (unique because the KV trie
is content-addressed — identical fillers dedupe and never overcommit),
124k-token pool + swap_pool=4096, 12 rounds of concurrent turns,
120 s/call wedge threshold. Exit 0 = clean, 2 = wedge (livelock), 1 =
inferlet error (so far: KV_INVARIANT_VIOLATION).

Run instrumented: `PIE_SCHED_DEBUG=1 PIE_SCHED_DEBUG_SECS=5 bash
50_run_repro.sh` — dumps scheduler state (active/pinned/stashed/suspended,
alloc-queue head need-vs-free) every 5 s to the server log
(`logs/repro_pie_serve_<TS>.log`).

**A run is IN FLIGHT right now** (server log timestamp 220004), the first
with the park-after-generation fix. Interpret its outcome:

- **Completes all rounds** → the parked-window was feeding both defects;
  livelock may need the harsher trigger (idle-suspend: add
  `--idle-suspend`) or higher pressure (`--sessions 10`). Try those before
  declaring victory.
- **Exit 2 (wedge)** → defect 1 reproduced CLEANLY. Read the last DumpSched
  lines in the server log: head `need > free` persistently = eviction loop
  never victimizes idle min-bid contexts for the starved requester;
  `need <= free` = stale queue / missing DrainKick. Fix accordingly (below).
- **Exit 1 (KV_INVARIANT again)** → the theft window is wider than
  fork-time; re-read the theory in §"Defect 2 diagnosis" — check whether
  `flush()` of the NEXT request's append can also share working pages, or
  whether the parent's tail page is referenced by the child even after the
  parent was re-parked by a LATER round while a prior child still lives.

## The diagnosis chain (how we got here)

1. c8 benchmark arms all failed 8/8 with 900 s timeouts regardless of
   inferlet policy: A (snapshots+swap), B (live + wake-bid 1.0),
   C (live + idle-suspend). Policy is not the problem.
2. Repro run 1 (exit 2): session 8's FRESH context starved while 7 idle
   parents sat resident at bid 1e-12 — **defect 1: alloc-path
   livelock** (eviction loop never evicts them, or queue never drains).
3. Repro runs 2–3 (exit 1): `KV_INVARIANT_VIOLATION total_kv =
   page_capacity + 1` — **defect 2**: `suspend()` FREES working pages
   without refcounting (`sched.rs` suspend Phase 1), and a fork's child
   references the parent's partial tail WORKING page (prompts aren't
   page-aligned; partial pages can't commit). Evicting a parked parent
   mid-generation steals the generating child's tail page.
4. Inferlet-side fix for defect 2's trigger (BUILT + INSTALLED, uncommitted):
   in `handle_request`, the parent is now held at active priority during
   generation and parked (idle bid / optional suspend) only AFTER the
   child's generation completes. See the two "CRITICAL ORDER" /
   "NOW it is safe" comments in `inferlets/openhands-coder-session/src/lib.rs`.
   The ENGINE bug remains: suspend must not free working pages a live fork
   references (refcount working pages, or copy-on-fork the tail page).

## The fix plan for defect 1 (task: alloc/eviction path)

Priority suspects in `runtime/src/context/sched.rs`:

- The alloc path branches on `requester_off_gpu` → `alloc_queue.push_back`
  ("wait for capacity **without restoring**") — a FRESH context may take a
  wait-only branch instead of the eviction loop (Step 4), so fresh fills
  can never evict anyone. Verify with the DumpSched head data.
- `DrainKick` exists as a message; check every page-free site actually
  sends it — a free with no kick strands alloc_queue/restore_queue waiters.
- If admission-starvation appears instead (need > free with churn):
  implement reserve-toward-head — freed pages accrue to the highest-bid
  waiter instead of returning to the general pool.

Validation ladder after any fix: repro (both policies, exit 0) → c1 smoke
(1 instance, live mode, `ab_h100_pie_overcommit_p32_c1` pattern; expect
93.8%-style reuse, 0 errors) → c8 arm (`armB2_driver.sh` pattern). Then
update fig4 + the writeup `[PENDING]` section, commit everything.

## Machine state

H100 pod, rebuilt 2026-07-30 (~18:40) by `00_setup_h200.sh` after /root
wipe — gate check PASS. Installed inferlet
`/root/.pie/programs/programs/openhands-coder-session/0.1.0.wasm` =
target build md5 `3829b3f50bf059efcc5a1272f3540394` (park-after-generation
fix). GPU must be free before any run (`nvidia-smi`; kill by PID, never
`pkill -f`). Chunk-sweep results and all benchmark rows are in
`predictions/` and summarized in the writeup draft
(`../docs/blog-pie-agent-serving.md`). GitHub token for pushes: ask the
human (earlier PAT is in this session's history only; a second token lives
in `/workspace/.env`).

## Open items beyond defect 1

- Engine fix for defect 2 (working-page refcount / copy-on-fork).
- Commit the park-after-generation inferlet change once repro-validated.
- Re-run c8 arms after fixes; fill writeup fig4 + `[PENDING]`.
- Decode-slope +10% at blk64 (borderline) wants one repeat before the
  writeup quotes 24,120 tok/s as the recommended config.
- Stretch: chat-apc HTTP surface (in `/workspace/pie-vllm-test`) as the
  transport for multi-harness (Codex) arms.
