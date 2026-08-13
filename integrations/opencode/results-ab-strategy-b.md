# Strategy A vs Strategy B — the A/B, run 2026-08-12

*Machine: `Lius-MacBook-Pro`, M-series, 48 GB unified, macOS 26.5.1, Metal.
Model: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` (30B MoE, **attention-only**).
Both arms: same binary, same config, same machine, one at a time.*

## Result

Six turns of a growing agentic conversation, replayed byte-identically to both arms.

| turn | A (frozen) | B (session) | speedup | A ttfc | B ttfc | prompt | B cached |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42.73 s | 46.92 s | **0.9×** | 34.93 | 44.62 | 7268 | 0 |
| 2 | 52.74 s | 4.39 s | 12.0× | 44.80 | 2.12 | 7454 | 7261 |
| 3 | 54.64 s | 4.37 s | 12.5× | 46.70 | 2.14 | 7640 | 7447 |
| 4 | 56.26 s | 4.50 s | 12.5× | 48.26 | 2.16 | 7826 | 7633 |
| 5 | 57.31 s | 4.57 s | 12.5× | 49.25 | 2.20 | 8012 | 7819 |
| 6 | 40.85 s | 4.64 s | 8.8× | 32.92 | 2.23 | 8198 | 8005 |
| **total** | **304.53 s** | **69.40 s** | **4.4×** | | | | |
| turns 2–6 | 261.80 s | 22.48 s | **11.6×** | | | | |

Raw: `/tmp/ab_a.json`, `/tmp/ab_b.json` (regenerate with `bench_ab.py`).

**Read the first row before the last one.** On the cold turn B is *slower* —
46.9 s against 42.7 s, about 10% — because it pays for retention and runs a
host-driven decode loop instead of the device-carried one. Every turn after
that is ~12× faster because the ~7.3k-token history is already resident and
only the ~190-token delta is prefilled. Time to first content is the honest
latency number a user feels: **~48 s → ~2.2 s**.

Turn 6 on arm A (40.85 s) breaks the otherwise monotonic 52→57 s trend on the
*longest* prompt of the run. That is unexplained; it is one sample of machine
variance on a shared laptop, and it is left in rather than smoothed.

## Method — and why it is a replay, not stock opencode

The two arms must see **byte-identical prompts** or the measurement is of the
renderers rather than the servers. That is not a hypothetical: qwen-code's run 2
lost a whole benchmark to exactly this.

Driving stock opencode twice cannot give it. Turn N+1's request contains turn
N's assistant reply, so the moment the arms diverge by one sampled token — and
at any temperature they will — every later turn compares different prompts.

So `bench_ab.py` replays a **canned** transcript built from the real captured
opencode wire fixtures (`tests/inferlets/fixtures/opencode/wire/`, a genuine
`opencode/1.18.16` session: 10 tools, a ~7.3k-token system+user turn, then an
assistant tool call and its result). Each turn is timed, its output discarded,
and the *canned* assistant turn appended before the next. Both arms see the same
bytes at every turn; only the server differs.

The canned reply favours neither arm. A retains nothing regardless. B addresses
retention by the **client's** messages only — no server output enters the
address — so a canned assistant turn resumes exactly as a real one would. (Under
the earlier seal-through-the-generated-turn design it would not have, which is
one more reason that design was wrong.)

Controls: `max_tokens=96` on both arms so the comparison is dominated by
prefill, which is where the strategies differ; `temperature=0`; a throwaway
warm-up request first, because the first request after a boot pays wasm JIT and
would otherwise land entirely in whichever arm ran first.

## What this does NOT show

**It does not show that pie beats vLLM.** Strategy A is our own
`chat-completions` inferlet, which has *no prefix caching at all* — every turn
re-prefills the whole history from zero, which is what `cached=0` in every A row
means. A production serving system does better than that for free.

The OpenHands evaluation on `openhands-integration-updated`
(`integrations/openhands/PIE_VS_VLLM_EVALUATION.md` §1) is explicit about this,
and it is the single most important caveat to carry:

> A multi-turn agentic loop is, turn by turn, an **append-only growing prefix**
> … A growing prefix is exactly what APC reuses: on turn 3 it recognizes turns
> 1–2 as cached and prefills only the new tail — the **same computational
> saving** Pie's session gives. So "multi-turn / agentic" is *not*, by itself, a
> Pie advantage.

Their measured 1.36× over litellm+vLLM came from **decode throughput** and from
removing a self-inflicted 10 s-per-call WebSocket close timeout — *not* from
out-reusing APC.

So the honest claim from this run is bounded:

- ✅ **Strategy B removes the re-prefill that Strategy A pays**, 11.6× on
  steady-state turns, ~22× to first content. The `~81 s → ~21 s` prediction in
  the handover is confirmed in shape and exceeded in degree, against the control
  it named.
- ❌ It says nothing about pie vs vLLM+APC, because APC would capture most of
  this same saving. Establishing that needs a vLLM arm on the same box.

## Scope: the hybrid (GDN) path is NOT covered

This ran on an **attention-only** model. Qwen3.6-35B-A3B (GDN hybrid) does not
work, and the blocker is upstream — three distinct walls, in order:

1. **`copy_kv` names CUDA unconditionally.** `scheduler.rs` builds every
   pre-launch copy-on-write plan with `PIE_MEMORY_DOMAIN_CUDA_DEVICE` hardcoded
   (four sites); Metal refuses any domain but `METAL_SHARED`. So
   `WorkingSet::fork` failed for *every* model on Metal. **Fixed in this branch**
   at the backend boundary (`runtime/engine/src/driver/backend.rs::copy_kv`).
2. **`RsWorkingSet::fork` mints a new sequence id**, and the driver rejects the
   child as a continuation of its parent:
   `recurrent slot 1 holds sequence 2^63, this fire is sequence 2^63+1`.
   Nothing in the guest can work around this. (The qwen-code session hit the
   same wall from the other direction, sealing on a fresh pipeline.)
3. **The fold advances even when told not to.** Generation was rebuilt to
   *buffer* rather than fold (`fold_len: Some(0)`, then `discard_buffered`) —
   the SDK's documented "fold nothing" mode, which is what makes a linear model
   speculatable. Turns then generated and retained correctly, but the next turn
   was refused with `recurrent slot 4 is at position 42, this fire starts at 33`
   — and 42 = 33 (render boundary) + 7 (cue) + 2 (decode fires). The fold had
   advanced over the buffered span anyway, and `discard_buffered` did not rewind
   it.

(3) is where it stands. Either the guest is holding the buffer API wrong or the
Metal hybrid path ignores `fold_len` when a KV binding is present; distinguishing
those needs driver-side knowledge this session does not have. Recorded rather
than guessed at.

Consequence: **on hybrid models Strategy B currently has no resume**, and
Strategy A remains the only working path. The attention-only result above stands
on its own — a 30B MoE is a realistic coding model — but the GDN class pie is
otherwise targeting is unmeasured.

## Reproduce

```sh
export CARGO_TARGET_DIR=~/Documents/Liszt_ai/.cargo-target/pie-opencode
export PIE_PYTHON=<a python3.10+ with the pie client deps>

# arm B
PIE_STRATEGY=b PIE_MODEL=mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit \
  integrations/opencode/run_pie_opencode.sh --serve-only &
python3 integrations/opencode/bench_ab.py --arm b --turns 6 --out /tmp/ab_b.json

# arm A — one server at a time; SIGTERM the first
PIE_STRATEGY=a PIE_MODEL=mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit \
  integrations/opencode/run_pie_opencode.sh --serve-only &
python3 integrations/opencode/bench_ab.py --arm a --turns 6 --out /tmp/ab_a.json
```

Both arms answer on `http://127.0.0.1:8080/v1` — under `b` the shim binds that
port and the gateway moves to `$PIE_ENGINE_PORT`. Nothing client-side changes
between arms, by construction.
