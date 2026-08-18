# The inferlet process is killed at large context — FIXED — 2026-08-18

Serving `Qwen3.6-35B-A3B` through `opencode-session` (strategy B), a
conversation that grew past a certain size **killed the guest process**. The
engine kept running and logged nothing; the shim reconnected and the run
continued, having silently lost every retained branch.

**Root cause: the gateway refused the turn, and refusing a turn killed the
session.** Nothing was ever wrong with the guest, the driver ABI, or the
session-resume path.

The chain, end to end:

```
guest rounds its KV reservation up to 2048 of 2048 pages
  -> pool reports 100% occupied
  -> worker heartbeat's kv_pressure_bucket saturates (>= 240/255)
  -> gateway admission refuses the NEXT turn
  -> ws.rs `break`s out of serve()
  -> handle.close() tears down the session
  -> the worker kills the inferlet and every working set it retained
  -> the socket is dropped WITHOUT a close frame
  -> the client sees a bare 1006 and synthesizes "WebSocket connection closed"
```

Fixed in `fcb1dc814` (the session-killing half) and `9dc2bd785` (the
saturating reservation). What remains open is a *different* wall, §7.

---

## 1. Why it took so long: three places discarded the reason

The engine stated its reason on the very first refusal. Three separate layers
threw it away, and each had to be opened before the next was even visible.

| layer | what it discarded | fix |
|---|---|---|
| `pie_client.client._listen_to_server` | `except Exception: pass` swallowed the close code | records and logs it, and threads it into the synthesized error |
| the same loop | processed only **binary** frames — and the gateway states failures on **text** (`error_json`) | text frames are parsed; `{"type":"error"}` is surfaced and hooked |
| `gateway/src/ingress/ws.rs` | returned `SessionError` to a `break` with no log | `tracing::warn!("ws: turn refused (session kept)")` |

The payload everyone was reading — `terminal event error: 'WebSocket connection
closed'` — is **not something pie sends**. `_fail_connection_waiters`
manufactures that string client-side whenever the listener loop ends. It says
"the socket died" and nothing else, which is why it looked like a guest death.
A real guest death delivers `ProcessEvent::Error(msg)` carrying a message.

With the first two opened, one ramp named it:

```
admission rejected: cluster saturated: no healthy worker has KV/seq headroom
```

## 2. The reservation was not "purely logical"

`inferlets/opencode-session/src/engine.rs` rounded its reservation up to a
multiple of 256 pages, under a comment asserting this "costs nothing real —
`reserve` is purely logical, no memory is held until a forward writes".

Measured against the pool's own `kv-pool-status`, per turn, that is false —
reported occupancy equalled `pool_pages_want` on **every** row:

| tokens `n` | need | want (rounded) | pool reported | actually written | held for nothing |
|---|---|---|---|---|---|
| 30,855 | 1093 | 1280 | **1280** | 964 | 315 pages |
| 37,467 | 1299 | 1536 | **1536** | 1171 | 365 pages |
| 46,283 | 1575 | 1792 | **1792** | 1447 | 345 pages |
| 55,099 | 1850 | **2048** | **2048/2048 = 100%** | 1722 | 326 pages |

The last row is the kill: 326 of the pages it saturated the pool with were
holding nothing at all.

**This is the whole `max_tokens` dependence.** `max_tokens` enters the
reservation, so it decides which turn crosses into the final 256-page bucket;
the refusal lands on the turn *after*. Both arms fit exactly — 4096 rounds up
at 55,099, 8 at 59,507, each dying one step later.

The old "§6 failed hypothesis" was right about the mechanism and wrong by one
turn, which is precisely why its prediction missed: it timed the turn that
*reserves*, not the turn that is then *refused*.

## 3. Two constants that meet exactly

`kv_pressure_bucket` (`runtime/engine/src/planner.rs`) clamps to **240**
whenever the planner has any queued allocation. `AdmissionConfig`
(`gateway/src/admission.rs`) treats **240** as saturated. They were chosen
independently and collide, so a single momentarily queued allocation reports
the whole cluster as saturated. Left as is — the session now survives a
refusal, which is what made the collision harmless. Worth revisiting if
refusals show up in normal operation.

## 4. What was fixed

**`gateway/src/ingress/ws.rs`** — a refused turn is non-fatal, exactly as the
adjacent arm already treated a malformed frame ("report and keep the
session"). Closing the socket also aborted the session's *other* live turns,
which had not failed at all.

**`gateway/tests/ws_turn_refusal.rs`** — drives the real axum ingress over a
real socket: opens a session with headroom, removes the headroom, and asserts
the refusal arrives **and** the socket still serves, then recovers on the same
socket. It fails on the pre-fix code at exactly the socket-is-dead assertion.

**`inferlets/opencode-session/src/engine.rs`** — the reservation keeps its
coarse 256-page step while there is room, and narrows to 64 pages above 85% of
the pool, so the rounding can never be what saturates it.

**`integrations/opencode/session_shim.py`** — delivers the refusal to the
in-flight turn. It used to be failed as a side effect of the process dying.

## 5. Measured result

| build | last good prompt | how it failed | session |
|---|---|---|---|
| before | 55,099 | 1006, no reason anywhere | **killed**, all retained KV lost |
| gateway fix only | 55,099 | clean HTTP 503, reason stated | **survives** |
| + reservation fix | **57,303** | see §7 | **survives** |

Deterministic: two independent runs of the pre-fix build died at exactly
55,099.

## 6. Reproduce (~8 min, no agent, no Docker, no dataset)

```sh
cd integrations/opencode
PIE_PYTHON=~/.venvs/pie/bin/python tools/boot_pie.sh ramp \
    PIE_STRATEGY=b PIE_MODEL=qwen3.6-35b-a3b \
    PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096
~/.venvs/pie/bin/python tools/ramp_context.py 4096   # or 8
```

`ramp_context.py` grows ONE conversation ~2.2k tokens per turn and prints each
prompt size, so the last good size is the last line before it dies. The guest
now prints pool occupancy per turn, which is the number the gate reads next:

```sh
grep -ao "turn [0-9]* cached=[0-9]* .*pool=[0-9/]* ([0-9]*%)" /tmp/pie_opencode_shim.log
grep -a "pie-driver-metal" /tmp/pie_ramp.log
```

Environment measured on: Apple M5 Pro, 48 GB, Metal 4; `total_pages = 2048`,
`kv_page_size = 32` (pool = 65,536 tokens = exactly `max_model_len`; the pool
is clamped to the context ring, `driver/metal/src/context.cpp:245`).

Note `PIE_PYTHON`: the shim needs the 3.12 venv at `~/.venvs/pie`. System
`python3` is 3.9 and has no `msgpack`, and `boot_pie.sh` then fails with the
server already up.

## 7. STILL OPEN — the next wall: the Metal program cache

At 57,303 tokens the turn now fails a different way — a DEGRADED empty
completion, session intact:

```
register_program: Metal M1 program executable cache is full
(64 entries, cap 64, no eviction)
```

This is the previously unexplained `status -5` from 45k–54k prefills. It says
its own name now (`abi.cpp` logs `what()`; `m1_runtime.cpp` logs the count and
the refused shape).

Measured: **64 entries over 26 turns, ~2.5 new programs per turn**, each
`stages=1 channels=10..11`. So it is a ceiling in **turns**, not tokens, and
`kMaxProgramCacheEntries = 64` in
`driver/metal/src/pipeline/m1_runtime.cpp` never evicts — once full, every new
shape is refused for the rest of the process.

Two things are needed and neither is done:

1. **Find the churn source.** `POOL_GRANULARITY` quantizes the pool-page
   dimension, but something else in the fire container varies per turn — 13
   distinct pool shapes cannot produce 64 registrations. Log the shape inputs
   feeding `program_hash` and diff two consecutive turns.
2. **Then decide eviction vs. churn.** Eviction is a driver-lifetime change
   (entries are `shared_ptr`, so in-flight fires would keep theirs alive), and
   it is the wrong fix if the churn is a bug rather than a cost.

Do not raise the cap without (1): a cap chosen against unmeasured churn just
moves the wall.

## 8. Ruled out, each by experiment

| hypothesis | experiment | result |
|---|---|---|
| the guest process traps / is reaped | a real guest death delivers `ProcessEvent::Error(msg)`; the observed payload is client-synthesized | **falsified** — the guest never died on its own |
| a retained branch pinning the pool | refuse to retain any branch over budget (`72c155a30`) | **falsified**, reverted in `938e7ce11` |
| KV pool starvation | has its own message (`KV pool starved: N pages asked, 0 free of 2048`) | absent |
| hybrid session resume | reuse working immediately before (`cached=53221`, delta 219) | unrelated; fixed separately |
| a fault in the C++ driver ABI | all 9 silent `catch (...)` sites log `what()` | none fired |
| a hard context/model limit at ~53.5k | ramp at `max_tokens=8` | **falsified** — survived 59,507 |
| a websocket size limit | payload at death is ~218 KB against a 1 MiB client limit | not reached |

## 9. Bearing on the four-way benchmark

The unblocking condition is met for the failure this document was opened for:
a saturated pool no longer destroys a session, and pie answers 503 instead of
silently losing its KV and continuing with a worse patch.

**But do not treat the arm as sound yet.** §7 still degrades a turn at ~26
turns of growth on this workload, and a degraded turn is exactly the silent
accuracy loss that made the original bug invisible. `tools/pie_watchdog.sh`
keys on `terminal event error`, which will no longer fire — it should key on
`cache is full` and on DEGRADED turns before the next long run.

Strategy A is still **UNTESTED** against any of this. `ramp_context.py`
against a `PIE_STRATEGY=a` boot would settle it cheaply.
