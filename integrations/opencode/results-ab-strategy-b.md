# Strategy A vs Strategy B — the A/B, run 2026-08-12

*Machine: `Lius-MacBook-Pro`, M-series, 48 GB unified, macOS 26.5.1, Metal.
Model: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` (30B MoE, **attention-only**).
Both arms: same binary, same config, same machine, one at a time.*

## Result

Six turns of a growing agentic conversation, replayed byte-identically to both arms.

| turn | A (frozen) | B (session) | speedup | A ttfc | B ttfc | prompt | B cached |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42.73 s | 41.09 s | 1.0× | 34.93 | 39.06 | 7268 | 0 |
| 2 | 52.74 s | 3.89 s | 13.6× | 44.80 | 1.85 | 7454 | 7261 |
| 3 | 54.64 s | 3.93 s | 13.9× | 46.70 | 1.87 | 7640 | 7447 |
| 4 | 56.26 s | 3.98 s | 14.1× | 48.26 | 1.90 | 7826 | 7633 |
| 5 | 57.31 s | 4.03 s | 14.2× | 49.25 | 1.92 | 8012 | 7819 |
| 6 | 40.85 s | 4.08 s | 10.0× | 32.92 | 1.95 | 8198 | 8005 |
| **total** | **304.53 s** | **60.99 s** | **5.0×** | | | | |
| turns 2–6 | 261.80 s | 19.91 s | **13.2×** | | | | |

Raw: `/tmp/ab_a.json`, `/tmp/ab_b2.json` (regenerate with `bench_ab.py`).

Steady-state time to first content — the latency a user actually feels —
is **44.39 s → 1.90 s, 23×**. The cold turn is a wash (41.1 s vs 42.7 s):
B pays for retention and runs a host-driven decode loop, and gets it back
because it no longer canonicalizes the message list per turn.

**An earlier revision of this run measured 69.40 s and a cold turn that was
~10% SLOWER than A.** That version addressed retention by hashing canonicalized
*messages*; the current one hashes rendered *token ids*. Hashing tokens is
strictly more correct (see "What changed" below) and turned out to be faster
too, because canonicalizing every message into a fresh `Vec<CanonItem>` of
cloned `String`s each turn cost more than rendering the history a second time.
The superseded numbers are kept here rather than deleted, since "the safer
design was also the faster one" is not the result anyone predicts.

Turn 6 on arm A (40.85 s) breaks the otherwise monotonic 52→57 s trend on the
*longest* prompt of the run. That is unexplained; it is one sample of machine
variance on a shared laptop, and it is left in rather than smoothed.

## What changed since the first revision

Retention is now addressed by **rendered token ids**, not by canonicalized
messages, following the OpenHands prefix cache
(`openhands-integration-updated:inferlets/openhands-coder-session/src/prefix_cache.rs`).
Three things follow:

1. **Template drift misses cleanly.** A message-level address does not move when
   the chat template, the cue or the tokenizer changes — so a resume would have
   handed the model KV rendered by the *old* template, fluently and
   undetectably. A token-level address moves by construction, and
   `TEMPLATE_MARKER` covers what the ids cannot express.
2. **Every render-unit boundary is a resume candidate**, scanned longest-first
   (cap 8). A byte-identical retry — which opencode does after a transport error
   or a tool failure — now re-hits an earlier boundary instead of rebuilding.
   `test_retry_rehits_an_earlier_boundary` pins it.
3. **Tool schemas need no separate hashing.** `plan_render` folds them into the
   system turn, so they are part of the token stream and therefore part of the
   address for free. Same for the thinking channel.

**And it caught a real bug at bench scale.** Keeping many branches over-committed
the KV pool: 8 branches × ~8k tokens against a pool of `512 pages × 32 = 16,384`
tokens. Over-committing does not degrade — the engine kills the process and the
gateway WebSocket goes with it, so turn 6 completed and retained inferlet-side
and then 500ed on delivery. Retention is now bounded by a **token budget**
(`retain_tokens`, passed at launch), not a branch count, because a branch count
is not a resource bound.

The guest cannot derive that budget: **nothing on the `pie:inferlet` surface
reports the KV pool size.** `kv_page_size()` and `max_embed_length()` exist; a
pool capacity does not. So the launcher — the only party that has read the driver
config — passes it in. That is a gap worth closing upstream: a guest asked to
manage KV residency is not told how much KV it may hold.

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
retention by the token ids of the **client's own render** — no server output
enters the address — so a canned assistant turn resumes exactly as a real one
would. (Under the earlier seal-through-the-generated-turn design it would not
have, which is one more reason that design was wrong. OpenHands measured that
same mistake from the other side: predicting the next boundary from the
inferlet's own output collapsed their hit rate to ~3%, because the host
re-serializes JSON arguments with different bytes.)

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

- ✅ **Strategy B removes the re-prefill that Strategy A pays**, 13.2× on
  steady-state turns, 23× to first content. The `~81 s → ~21 s` prediction in
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
