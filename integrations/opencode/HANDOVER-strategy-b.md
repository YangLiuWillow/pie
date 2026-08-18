# Strategy B handover — 2026-08-18

Branch `liu/opencode-integration`, HEAD `539d49930`. Tree clean, 22 guest tests
green, device released.

**One-line state:** strategy B serves correct patches, survives pool
saturation, and no longer dies silently — but it still poisons a driver
instance now and then, and the harness's retries are what hide it. Do not run
the 30-instance benchmark until §3.1 is settled.

---

## 1. The bug this session opened with, and what it actually was

A conversation past ~55k tokens "killed the inferlet". It never did. The chain:

```
guest rounds its KV reservation up to 2048 of 2048 pages
  -> pool reports 100% occupied
  -> worker's kv_pressure_bucket saturates (>= 240/255)
  -> gateway admission refuses the NEXT turn
  -> ws.rs `break`s out of serve()
  -> handle.close() tears down the session
  -> the worker kills the inferlet and every retained branch
  -> the socket is dropped WITHOUT a close frame
  -> the client sees 1006 and SYNTHESIZES "WebSocket connection closed"
```

That last line is why it took days: the payload everyone read was manufactured
client-side. pie stated its reason on the first refusal and **three layers threw
it away** — the client's `except Exception: pass`, the client processing only
binary frames while the gateway states failures on text, and `ws.rs` returning
`SessionError` into a bare `break`.

## 2. Fixed and measured

| what | commit | evidence |
|---|---|---|
| a refused turn no longer destroys the session | `fcb1dc814` | `gateway/tests/ws_turn_refusal.rs` — **fails on the pre-fix code** at the socket-is-dead assertion |
| guest stopped reserving the whole pool | `9dc2bd785` | occupancy equalled the reservation on every turn; 326 phantom pages at the kill |
| program cache: cross product found, budget derived | `7b599454c`, `0a393ddbf` | 5 fire shapes per pool size; 65-vs-64 overflow reproduced |
| a full program cache no longer fails the fire | `78a10d2f4`, `d35a5cf88` | cap=8 vs cap=64 A/B: 272 s vs 270 s, 0 degraded |
| driver cap aligned to the engine's 256 | `e4178b1cd` | engine registry is a 256-entry LRU; driver was 64 and never evicted |
| pool sized above one conversation | `f809a4870` era | ramp 59,507 -> **88,159**, 0 trims, 0 refusals |
| retention consolidated into ONE rule | `582625657`, `38e50880f`, `fe0b374fb` | replaced 3 mechanisms + 2 dead budgets |
| post-saturation refusal is a 4.5 s window, not a wedge | `f809a4870` | `t+0.0 refused / t+4.5 SERVED` |
| earlier-boundary resume refused on a recurrent model | `1f0aeee3b` | killed the large-gap poison class (162 / 1210 / 4712 token drift) |

**Retention now has one invariant.** The tip is the working set and is kept;
older branches are cache bounded by `retain_tokens`; one exception may take the
tip, when keeping it would leave the pool with no room for another turn.
`cache_budget_tokens` is pure and tested (5 tests, both failure directions).

## 3. Open, ranked

### 3.1 A second fold-drift class poisons driver instances — BLOCKS THE BENCHMARK

```
instance 5126 launch failed: recurrent slot 1 is at position 806, this fire starts at 768
instance 5878 launch failed: recurrent slot 0 is at position 789, this fire starts at 768
```

Nothing clears a poison epoch, so each one can end the process. The agent
harness retries over them, so the only trace is these lines plus
`prefill take: channel is poisoned` in the shim log. **Two landed in a
five-minute run whose three trajectories all reported success.**

I first blamed `RsWorkingSet::fork` not being copy-on-write. **That is probably
wrong** — 4 resumes succeeded and 2 poisoned in the same run, and universal CoW
breakage would fail every resume.

Better fit: **recurrent-state seat pressure.** `requested=8 seated=4`. That run
put three sequential conversations through one guest process, each retaining a
branch holding a slot, and the failures name DIFFERENT slots (0 and 1) at the
SAME position (768) — two conversations, not one drifting.

**The test, ~10 minutes, no new code:**

```sh
# arm A: ONE conversation against a fresh process
PIE_PYTHON=~/.venvs/pie/bin/python tools/boot_pie.sh t1 PIE_STRATEGY=b \
    PIE_MODEL=qwen3.6-35b-a3b PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096
python run_swebench.py --instances django__django-14373 --model pie/qwen3.6-35b-a3b ...
grep -ac "launch failed" /tmp/pie_t1.log

# arm B: THREE conversations against the SAME process (reboot first, then 3 runs)
```

If the failures need more than one conversation it is seats, not fork, and the
fork/fold experiment in `finding-inferlet-killed-at-large-context.md` §10 is not
worth building.

### 3.2 `park()` — unused in the whole crate

`engine.rs:356` opens `Pipeline::new()` per turn and closes it at `:746`.
`submit`'s contract: the first submit of a pass binds seeds, steady-state
resubmits carry only the identity hash. So we pay seed binding **every turn**.
`park` leaves the frame wait-set without running down `submit_deadline` and
keeps the pipeline usable.

Lands in the **TTFT** column, not the per-token rate — which matters because
TTFT is what the A-vs-B comparison is quoted on.

### 3.3 Sequential decode may not be necessary

`engine.rs:24-45` rejects `run_ahead` because overshoot fires fold
irreversibly. That reasoning came from the qwen-code port, which **sealed**;
this crate does not. `discard_buffered` is exactly the rewind the comment says
does not exist. Cost is not the 0.4% recorded — it also forecloses the frame
model's heterogeneous slots. Re-measure the fold-position invariant first.

### 3.4 `#31` — the container binds pool size with token count

Bounded (10 shapes x 5 = 50 of 256), not eliminated. Feasibility established:
`forward.cpp:371-379` checks `kv_page_indptr[r+1] <= kv_pages.size()`, not
equality. Blocked on tracing guest `pages_p` -> engine descriptor -> driver
`kv_pages`. Low priority; the risk is silent wrong output.

### 3.5 Stale comments

`engine.rs:129-133` still describes `RsWorkingSet::fork` minting a new sequence
id as an open driver bug blocking CoW branching. **Fixed** by
`rebase_linear_sequence` in `driver/metal/src/batch/forward.cpp`.

## 4. Config facts worth not rediscovering

- **For a hybrid the KV pool is `ceil(max_model_len / kv_page_size)` EXACTLY**;
  configured `total_pages` is ignored (`context.cpp:255`,
  `rs_cache_required ? ctx_pages : min(...)`). So the pool is one
  maximum-length conversation unless you raise `max_model_len`. Driver cap:
  `kPhase1bRsSlots(64) * kMetalCtxTokensPerRequest(2048)` = **131,072**.
- The gateway refuses at `kv_pressure_bucket >= 240/255` (~94% used), and the
  bucket is **pushed** every `REPORT_INTERVAL = 2s`. Recovery after a flush is
  therefore a few seconds, not instant, and not a wedge.
- `waiters != 0` clamps the bucket to exactly 240 — one queued allocation
  reports full saturation. Harmless now that a refusal is survivable.
- The shim needs the 3.12 venv: `PIE_PYTHON=~/.venvs/pie/bin/python`. System
  `python3` is 3.9 with no `msgpack`, and the boot fails with the server up.
- Guest samples at `DEFAULT_TEMPERATURE = 0.6`. **n=1 cannot separate a
  regression from variance** — one instance measured 412/0/412/412/412 bytes
  across five runs of the same build.

## 5. Tools

- `tools/pie_watchdog.sh` — in the repo now (was `/tmp`). Silence means
  healthy. Checks: program-cache full + headroom, turn refused (**with the
  reason**), retention thrash, pool unrelievable, resume refused, driver fault
  (**searches BOTH logs**), inferlet crash, reuse collapse (**consecutive**,
  not a share), degraded turns, empty-patch run.
- `tools/recovery_window.py` — measures the WINDOW after saturation, prints the
  refusal **body**. Requires `PIE_PROBE_OK=1`; the machine is shared.
- `tools/ramp_context.py` — grows one conversation ~2.2k tokens/turn.

## 6. The lesson worth carrying

**Four instrument bugs against roughly four engine bugs.** The watchdog keyed
on a string my own fix retired; its reuse check cried wolf on every
conversation boundary; its driver-fault check read the wrong log; the recovery
probe recorded a status code and discarded the reason. Every one encoded an
assumption about *why* something fails, and each was wrong in the specific way
that made it silent.

Three of my fixes were also wrong on first attempt — the reservation cap, the
trim threshold, and a KV/RS split that `forward-hybrid.wit` forbids outright.
All three were caught by measurement, none by reasoning.

**Measure the thing, print the reason, and re-run before believing an n=1.**

## 7. Coordination

Peer session `liszt-ai-00` works kernel optimisation on
`liu/qwen3-coder-kernels` (worktree `~/Documents/Liszt_ai/pie-coder-kernels`).
Division: they own `draft.rs` and the decode/speculation path in `engine.rs`;
we own the reservation block, retention and `handler.rs`. **Announce before
taking the GPU.** Their findings that bear on us:

- Adaptive drafting: **37.4 vs 25.8 tok/s at 28k** (1.45x over static).
- B's TTFT advantage over A at 28k is **prefix reuse**, not an A prefill
  defect. The decode gap (21.8 vs 39.3, same kernel) is open.
- Strategy A is ~2x slower than B at 28k in both prefill and decode.
