# The inferlet process is killed at large context — OPEN — 2026-08-18

Serving `Qwen3.6-35B-A3B` through `opencode-session` (strategy B), a
conversation that grows past a certain size **kills the guest process**. The
engine keeps running and logs nothing; the shim reconnects and the run
continues, having silently lost every retained branch.

This is NOT the hybrid session-resume bug (fixed, see
`finding-hybrid-session-resume.md`). Reuse works perfectly right up to the
moment of death.

**Status: mechanism not established.** The reservation sizing is implicated by
a controlled experiment; the exact trigger is not known, and one arithmetic
hypothesis has already been falsified. Do not ship a fix that is not backed by
a new measurement.

---

## 1. Symptom

```text
[opencode-session] turn 206 cached=53221 delta=219 cue=7 gen=74 retained ... (len 53440)
[session-shim] inferlet terminal event error: 'WebSocket connection closed'
[session-shim] reconnect attempt failed: ConnectionError('WebSocket connection closed')
[session-shim] session inferlet up: opencode-session@0.1.0 process=<new uuid>
```

The client sees an HTTP 500, or — inside opencode — a degraded turn. Because
the shim reconnects and the harness continues, the event is invisible in the
timing columns, in the patch count, and in the degraded-turn count. It is
observable ONLY as `terminal event error` in the shim log. That is why
`tools/pie_watchdog.sh` keys on that string.

## 2. Reproduce (~25 min, no agent, no Docker, no dataset)

```sh
cd integrations/opencode
tools/boot_pie.sh ramp PIE_STRATEGY=b PIE_MODEL=qwen3.6-35b-a3b \
    PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096
python3 tools/ramp_context.py 8        # or 4096
```

`tools/ramp_context.py` grows ONE conversation ~2.2k tokens per turn and prints
the prompt size of each, so the last good size is the last line before it dies.

Watch, in three places — all three matter, because two of them stay silent:

```sh
tail -f /tmp/pie_opencode_shim.log | grep -aE "cached=|terminal event"
grep -a "pie-driver-metal" /tmp/pie_ramp.log     # ABI faults (instrumented, see §5)
sed 's/\x1b\[[0-9;]*m//g' /tmp/pie_ramp.log | grep -viE "tarpc|otel\.|rpc\."
```

Environment this was measured on: Apple M5 Pro, 48 GB, Metal 4;
`total_pages = 2048`, `kv_page_size = 32` (pool = 65,536 tokens, which is
exactly `max_model_len` — see `driver/metal/src/context.cpp:245`, the pool is
clamped to the context ring).

## 3. The one thing that IS established

**The failure point moves with `max_tokens`.**

| arm | `max_tokens` | last good prompt | died at |
|---|---|---|---|
| opencode (four-way) | 4096 | — | ~53,440 and ~53,562, two separate runs |
| `ramp_context.py` | 8 | **59,507** | next step, ~61,711 |

`max_tokens` reaches the guest in EXACTLY one place — the pool reservation in
`inferlets/opencode-session/src/engine.rs:401`:

```rust
const POOL_GRANULARITY: u32 = 256;
let pool_pages_want = (n + cfg.max_tokens as u32 + 2)
    .div_ceil(page_t)
    .next_multiple_of(POOL_GRANULARITY);
```

So the reservation sizing is implicated. Equally, this rules out a fixed
context ceiling, a model limit, and a driver limit at ~53.5k: with a smaller
`max_tokens` the same build sails past 53.5k without noticing.

## 4. Ruled out, each by experiment

| hypothesis | experiment | result |
|---|---|---|
| retained branch pinning the pool | refuse to retain any branch over budget (`72c155a30`) | **falsified** — the refusal fires, the kill follows 3 lines later. A conversation that large has its pages resident DURING the turn regardless of retention. Reverted in `938e7ce11`. |
| KV pool starvation | that failure has its own explicit message (`KV pool starved: N pages asked, 0 free of 2048`) | different signature; absent here |
| hybrid session resume | reuse working immediately before (`cached=53221`, delta 219) | unrelated |
| a fault in the C++ driver ABI | instrumented all 9 silent `catch (...)` sites to print `what()` (§5) | **none fired** |
| a hard context/model limit at ~53.5k | ramp at `max_tokens=8` | **falsified** — survived 59,507 |

## 5. Instrumentation already added

`driver/metal/src/abi.cpp` had 11 `catch (...)` blocks, 9 of which discarded
the reason and returned a bare `PIE_STATUS_DRIVER_ERROR`. All 9 now log
`e.what()` first: `register_program`, `register_channel`, `bind_instance`,
`launch`, `copy_kv`, `copy_state`, `resize_pool`, `close_instance`,
`close_channel`.

This is why "the ABI is not the fault" is now a measurement rather than an
assumption. It also remains the best lead for the SEPARATE
`pie_metal_register_program failed with status -5` failures seen earlier at
45k–54k prefills — those predate the instrumentation, so their reason was
discarded; if they recur they will now name themselves.

## 6. A hypothesis that FAILED — do not re-derive it

The rounding above jumps to 2048 pages — the entire pool — at a context that
matches the opencode crashes almost exactly:

```
max_tokens=4096:  ctx 53,240 -> reserve 2048 of 2048 pages   (crashes: 53,440, 53,562)
```

It predicts the ramp at `max_tokens=8` dies above **57,328**. The ramp
**survived 57,303 AND 59,507**, dying only after. So "the reservation rounds up
to the whole pool and that is the trigger" is wrong in detail, even though the
`max_tokens` dependence it predicted did hold. Something about the reservation
matters; this arithmetic is not it.

Related, and worth knowing before touching the constant: `chat-completions`
already replaced this fixed 256-page step with `next_power_of_two()`
(`inferlets/chat-completions/src/engine.rs:363`) after measuring that the flat
granularity cost the server's concurrency ceiling — 8 of 32 concurrent requests
completed, 24 rejected at 0.0 s. Its comment explicitly retracts the "costs
nothing real … a longer page-id list and nothing else" claim that
`opencode-session` still carries verbatim at `engine.rs:396`. Note that
power-of-two does NOT avoid the 2048 case either: 1799 -> 2048.

## 7. Next step

Make the guest's death observable. The process dies with no engine error, no
ABI exception, and no trap in any log, which is the signature of the wasm
instance being torn down rather than returning an error. Find where the engine
reaps an inferlet process and log the reason — trap, fuel exhaustion, memory
growth failure, host-call error — since that is the one piece of evidence
neither side currently carries.

Start at `runtime/engine/src/inferlet/` (process lifecycle) and the terminal
event the shim receives; `grep -rn "terminal" runtime/engine/src` for the
publisher of the event the shim prints.

Only after the reason is known should the reservation be changed.

## 8. Why it blocks the four-way

An instance whose context passes the threshold loses its session mid-run. The
harness does not fail — opencode continues against a fresh process — so the arm
reports an ordinary instance with a worse patch. Any pie accuracy or throughput
number carries that silently. The benchmark is stopped until this is fixed, at
the user's direction: measuring a knowingly-buggy engine is not worth the GPU
time.

Strategy A is very likely affected too — it drives the same prefill sizes
through the same reservation shape — but this is UNTESTED. `ramp_context.py`
against a `PIE_STRATEGY=a` boot would settle it cheaply.
