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

Fixed in `fcb1dc814` (the session-killing half), `9dc2bd785` (the saturating
reservation) and `7b599454c` (the program-cache budget that the second of
those overflowed). Reach: 55,099 -> **59,507** tokens, and the failure at the
end is now the true pool limit rather than a premature one.

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
coarse 256-page step while there is room, and narrows to 128 pages above 85%
of the pool, so the rounding can never be what saturates it. The step is sized
by the driver's program budget, not by taste — see §7.

**`integrations/opencode/session_shim.py`** — delivers the refusal to the
in-flight turn. It used to be failed as a side effect of the process dying.

## 5. Measured result

| build | last good prompt | how it failed | session |
|---|---|---|---|
| before | 55,099 | 1006, no reason anywhere | **killed**, all retained KV lost |
| gateway fix only | 55,099 | clean HTTP 503, reason stated | **survives** |
| + reservation fix (64-page band) | 57,303 | DEGRADED empty completion (§7) | survives |
| + 128-page band | **59,507** | clean 503 at the TRUE limit | **survives** |

At 59,507 the next turn genuinely needs more than the pool holds
(`n + max_tokens > 65,536`), so the refusal is correct rather than premature.
51 of the driver's 64 program cache entries were used, with no cache-full
event.

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

## 7. The second wall: the Metal program cache holds a CROSS PRODUCT

Removing the phantom reservation moved the wall, and the next one was the
driver's program cache — `kMaxProgramCacheEntries = 64` in
`driver/metal/src/pipeline/m1_runtime.cpp`, which **never evicts**. This is the
previously unexplained `status -5` at 45k–54k prefills.

Logging every decoded container at the one funnel every program passes through
(`driver/backend.rs::register_program`) shows what varies, in six turns:

```
Pages chan | fire token-counts registered
       256 | [1024, 152, 5, 7, 1]
       512 | [1024, 152, 3, 7, 1]
       768 | [1024, 152, 3, 7, 1]
```

The container binds the `pages_p` channel length **together with** the fire's
token count, so the cache stores their cross product: five fire shapes
re-register against every distinct pool size. The budget is

```
5 x (distinct pool reservations)   against a non-evicting cap of 64
```

not the "~2.5 new programs per turn" an earlier reading of this guessed.

That indicted the §4 reservation fix itself. Its first tuning used a 64-page
step near the top, producing 13 distinct pool sizes = **65 entries against a
cap of 64** — an overflow by exactly one, surfacing as a DEGRADED empty
completion at 57,303. The step is now 128 pages:

| quantization | pool sizes | programs | |
|---|---|---|---|
| 256 only (pre-fix) | 8 | 40 | fits |
| 64-page band | 13 | **65** | **overflowed — shipped briefly** |
| 128-page band | 10 | 50 | fits |

`pool_shape_budget_fits_the_driver_program_cache` (in
`inferlets/opencode-session/src/engine.rs`) enumerates the reservations one
conversation makes and fails with the arithmetic spelled out, because this is a
whole-process resource with no eviction, spent by a constant in a different
crate, whose overflow appears eight minutes into a run as an empty completion.

**The real repair is still open:** stop binding the pool size into the fire
container, making the factor 1 instead of 5 and the cap a non-issue. That is a
container-layout change; the test holds the line until then.

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

## 8b. The refusal after saturation is BOUNDED, not a wedge

A conversation that fills the pool still gets one 503, and the reflex is to
call that a wedge. Measured with `tools/recovery_window.py`:

```
t+ 0.0s  refused  503 admission rejected: cluster saturated
t+ 4.5s  SERVED
RECOVERY WINDOW: 4.5s
```

That is the coarse-load report interval, not a fault. `gateway/src/admission.rs`
reads only `RoutingTable.coarse_load.kv_pressure_bucket`, which the worker
PUSHES every `REPORT_INTERVAL = 2s` (`worker/src/link/control.rs`), and the
controller advances the gateway epoch only on a bucket cross. So after the
guest frees its pages the gate keeps refusing until the next report lands.

This was written up once as "the wedge is back" on the strength of a probe
fired sub-second after the flush. It was racing the report. The lesson is the
measurement, not the mechanism: served-vs-refused cannot distinguish a bounded
lag from a permanent hold, and only the WINDOW can — which is why the tool
reports elapsed time and prints the refusal body rather than the status code
alone.

The guest log shows the whole cycle: turn 27 flushes to 0%, turn 28 serves
cold (`cached=0`) and the pool returns to 12%.

## 9. Bearing on the four-way benchmark

The unblocking condition is met for the failure this document was opened for:
a saturated pool no longer destroys a session, and pie answers 503 instead of
silently losing its KV and continuing with a worse patch.

Both walls now fail cleanly rather than silently, and the program budget has a
test holding it. **Two things still to do before a long run:**
`tools/pie_watchdog.sh` keys on `terminal event error`, which the fix makes
stop firing — it should key on `cache is full`, on `turn refused`, and on
DEGRADED turns, since a degraded turn is exactly the silent accuracy loss that
made the original bug invisible. And a conversation past ~59.5k tokens still
ends in a 503; that is honest, but the harness must be seen to handle it.

Strategy A is still **UNTESTED** against any of this. `ramp_context.py`
against a `PIE_STRATEGY=a` boot would settle it cheaply.

---

## 10. OPEN: a second fold-drift class — is `RsWorkingSet::fork` copy-on-write?

Two poison epochs survive the earlier-boundary fix (§ commit
`1f0aeee3b`), and they are a different mechanism:

```
instance 5126 launch failed: recurrent slot 1 is at position 806, this fire starts at 768
instance 5878 launch failed: recurrent slot 0 is at position 789, this fire starts at 768
```

Gaps of 38 and 21 tokens against a fire starting at 768. Small, and about the
size of one cue plus one generation (`cue=7`, so 7+31 and 7+14).

**Where the retained fold is supposed to come from.** `engine.rs` no longer
buffers-and-discards — that scheme was CUDA-only and silently no-op'd on Metal.
Generation now FOLDS normally, and retention instead takes a `fork` of the rs
working set *before a single scratch token is written*:

```rust
let retain_rs: Vec<RsWorkingSet> = state.rs.iter().map(|r| r.fork(&pipe))...
```

So the retained fold should stand at `render_len` while the live state folds on
through cue and generation.

**The hypothesis.** `working-set.wit` specifies that fork as lazy
copy-on-write: "the first fold/write on a shared folded or buffered slab copies
the relevant object before mutation". If the parent instead folds IN PLACE on a
shared slab, the child's fold advances with it — and the retained state would
sit exactly one cue+generation ahead of `render_len`, which is what these two
measurements show.

That is a hypothesis with a matching signature, NOT a confirmed cause — and a
later measurement WEAKENS it, so read the alternative below before spending a
test on it.

**Why the fork-CoW reading is probably wrong.** In the same run, 4 resumes
SUCCEEDED and 2 poisoned. If the parent folded in place on a shared slab,
copy-on-write would be broken universally and every resume would fail. It does
not.

**The better fit: RS seat pressure.** The engine caps recurrent-state seats:

```
admission: more lanes than the recurrent-state pool can seat;
requested=8 seated=4 seat_cost=2
```

FOUR seats. The run that produced these failures put three sequential agent
conversations through ONE guest process, each retaining a branch that holds a
slot. The two failures name DIFFERENT slots (0 and 1) at the same position
(768) — two conversations of similar length, not one conversation drifting.
Intermittent, slot-indexed, and cross-conversation all point at a slot being
reused or reclaimed while a retained branch still expects it, rather than at a
fold advancing under its own child.

**So test seat pressure first, and it is cheaper than the fork test.** Run ONE
conversation to completion against a fresh process and check for any
`launch failed`; then run three sequential conversations against the same
process and check again. If the failures need more than one conversation, it is
seats, not fork. Only if a single conversation reproduces it is the fork/fold
experiment below worth building.

**How to test it, without an agent.** Fork an rs working set, fold `n` tokens
into the PARENT, then fire a continuation from the CHILD at the pre-fork
position. If fork is CoW the child continues cleanly; if the parent folded in
place the driver refuses with `recurrent slot ... is at position X, this fire
starts at Y` and `Y + n == X`. `driver/metal/tests/executor_geometry_test.cpp`
already pins the sibling invariant (`rebase_linear_sequence`) and is the
natural home.

**Why it matters more than its frequency suggests.** Nothing clears a poison
epoch. Each occurrence can end the process, and the agent harness retries over
it, so the only trace is a `launch failed` line in the engine log and a
`prefill take: channel is poisoned` in the shim log. Two landed in a
five-minute run whose three trajectories all reported success.
