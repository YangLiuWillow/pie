# pie cannot resume a session on a hybrid-attention model — 2026-08-17

Serving `Qwen3.6-35B-A3B` through the `opencode-session` inferlet (strategy B),
**every turn that resumes from retained state fails**. The first four-way
SWE-bench run produced **24 empty patches out of 24 instances** before it was
stopped.

The same inferlet, same code, same machine, resumes correctly on
`Qwen3-Coder-30B-A3B` — which is what the previous 30-instance comparison ran
on, and why this was never seen.

## Symptom

```
[pie-driver-metal] instance 48 launch failed: paged continuation:
    recurrent slot 2 is at position 531, this fire starts at 493
```

The gap is exactly the turn's scratch span: `cue + generated` (7 + 31 = 38, and
36 = 7 + 29 on the other observed instance). A turn retains at length 493 and
the next turn's prefill against 493 is refused because the driver's recurrent
slot stands at 531.

## Cause — the guest's `discard_buffered` is a no-op on Metal

`engine.rs` generates without folding: it passes `fold_len: Some(0)` so the cue
and the decoded tokens land in the **recurrent buffer** rather than the fold,
then calls `discard_buffered` to abandon them. That leaves the fold at the
render boundary, which is what retention addresses. On CUDA this is exactly the
REJECT half of fold-commit and it works.

**The Metal driver does not implement the buffer.** Traced end to end:

| stage | file | what happens to `fold_len` |
|---|---|---|
| engine → driver | `driver/metal/src/context.cpp:874` | `launch.rs_fold_lens = step.rs_fold_lens` — plumbed in |
| validation | `driver/metal/src/batch/compose.cpp:109-131` | shape-checked; `n == 0` admitted (`n != 0 && n < rows` is the refusal) |
| execution | — | **never read again** |

`grep -rn "rs_fold\|fold_len" driver/metal/` returns those two sites and nothing
else. The buffer arrays go the same way: `rs_buffer_slot_ids` appears only in
residency accounting (`context.cpp:785`) and composition, never to hold tokens
out of the fold. Metal states the limitation directly for the read half
(`compose.cpp:61`):

> The buffer READ path is CUDA-only. Metal has no extended token layout […]
> Refuse instead.

So on Metal every fire that writes the slot **folds its tokens**. The
consequences, in order:

1. The recurrent state genuinely stands at 531 after generation. It is not a
   miscount — the cue and the decoded tokens are *in* the state.
2. `resident_next_position = 531` is therefore **correct**, and the driver is
   right to refuse a fire starting at 493.
3. `discard_buffered(38)` touches only engine-side bookkeeping
   (`store/rs.rs:430` adjusts `occupancy` and `buffer_head`) and emits nothing
   to the driver. On Metal it discards a buffer that was never used, and leaves
   the guest believing the state is at 493 when the device is at 531.

**The driver is right and the guest is wrong.** This is not a driver bug to be
patched by rewinding the counter — doing that would resume from a state
contaminated with the cue and the generated tokens, turning a loud refusal into
silent corruption.

## Why only strategy B, and only this model

Not the inferlet's coding — it implements the documented design faithfully. The
design is CUDA-only and nothing checked that at the Metal boundary.

- **Strategy A** never resumes. One `chat-completions` inferlet per request, a
  fresh sequence each time, so the continuation check never runs.
- **Qwen3-Coder-30B** is attention-only. `state.rs` is empty, there is no fold,
  and KV alone rewinds for free by dropping pages past `render_len`.
- **Qwen3.6** is hybrid: 30 of its 40 layers are Gated DeltaNet. The fold is a
  running summary with no `kv_len` to mask an overshoot, so the scratch span
  that KV discards for free is precisely what the fold cannot give back.

## What does not fix it

- **Rewinding `resident_next_position` in the driver** — unsound, see above.
- **`RsWorkingSet::fork`** — already tried (`engine.rs:129`, `:439`). Fork mints
  a *new* sequence id and the paged check refuses the child as a continuation of
  its parent. The KV half was separately fixed on Metal
  (`driver/backend.rs::copy_kv`) and was still not enough.
- **Retaining through the generated span** — the re-render cannot reproduce the
  generated tokens: `chat.rs:729` strips `<think>…</think>` on replay, and
  Qwen3.6 generates into a think block (`generation_suffix: "<think>\n"`). The
  retained prefix would diverge on the first turn the model reasons.

## The fix: fork the fold, and let a CoW child own its own sequence

Two changes, one in the driver and one in the guest.

**Driver — `MetalExecutor::copy_state` rebases the sequence id.** A recurrent
fire's id is not free-standing metadata: `context.cpp` derives it from the slot,
`(1<<63) | rs_slot_id`. `copy_state` carried the SOURCE's id onto the
destination slot, so a copy-on-write child announced `1<<63|dst` against a
record saying `1<<63|src` and the paged check refused it. That — not the shape
of the design — is what made `RsWorkingSet::fork` unusable, and it is the exact
error `engine.rs` recorded when the "generate on a fork" fix was abandoned.
The rebase now happens in `rebase_linear_sequence`, split out as a pure
function so it is testable without a live executor, and pinned in
`driver/metal/tests/executor_geometry_test.cpp` by four assertions — including
one that reproduces the historical refusal string byte-for-byte when the rebase
is undone.

**Guest — `engine.rs` forks at the scratch boundary.** Immediately after the
delta prefill and before the cue, the recurrent working set is forked. The
generation that follows folds normally and copies the slot on its first write,
so it runs away on a private copy while the child keeps the fold exactly as it
stood at `render_len`. That child is what gets retained. `alloc_buffer` /
`discard_buffered` are gone with the scheme that needed them, and speculation is
now gated on there being a fold at all rather than on the buffer flag.

### Measured, on a clean boot

Four conversations, 17 turns, `Qwen3.6-35B-A3B`, strategy B:

```text
turns completed   17
degraded turns     0        (was: every resume)
driver refusals    0        (grep 'launch failed' /tmp/pie_dbg36.log)
```

Reuse now chains, which it never did on this model before — each turn's
`cached` is the previous turn's retained length:

```text
turn 1 cached=0   delta=38  retained (len 38)
turn 2 cached=38  delta=101 retained (len 139)
turn 3 cached=139 delta=95  retained (len 234)
turn 4 cached=234 delta=72  retained (len 306)
turn 5 cached=306 delta=92  retained (len 398)
turn 6 cached=398 delta=82  retained (len 480)
```

and a new conversation still correctly starts cold (`cached=0`). Latency
follows: first turn 1.6 s, every resume 0.8–1.0 s.

## Still open: a byte-identical re-send alternates

Sending the *same* request twice in a row fails on every second attempt:

```text
send 0: finish='stop'   compl=6  'Red, blue, yellow.'
send 1: finish='length' compl=0  'The server could not complete this turn.'
send 2: finish='stop'   compl=6  'Red, blue, yellow.'
send 3: finish='length' compl=0  'The server could not complete this turn.'
```

```text
[opencode-session] turn 12 cached=0 delta=19 ... retained 82e4313c2cd26b30 (len 19)
prefill take @9: g0 take: channel is poisoned: driver published poison epoch 1
[pie-driver-metal] paged continuation: recurrent slot 3 is at position 19,
    this fire starts at 9
```

The failing turn goes COLD (`cached=0`) rather than resuming, and then its
chunked re-prefill collides with the slot the just-retained state still holds.
So it is a slot-reuse problem on the cold path, not the resume path this commit
fixed — a different bug wearing the same poison signature.

Two things bound it. The successful attempts return byte-identical output, so
resumption itself is deterministic and correct. And it needs a request whose
retained address is already present: a cold start *while another conversation
is retained* is fine (verified — conversations 2 and 3 above both start cold
against live retained state and every turn passes). The opencode workload
appends a new user message every turn and so does not take this path, which is
why the 17-turn run is clean; a client that retries a request verbatim would
hit it.

## Reproduce

```sh
tools/boot_pie.sh dbg PIE_STRATEGY=b PIE_MODEL=qwen3.6-35b-a3b \
    PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096
# any two-turn conversation; the second turn degrades
grep -aE "turn [0-9]|failed|poison" /tmp/pie_opencode_shim.log
```

Reproduces on a plain four-turn chat with **no tools**, in seconds. Swap
`PIE_MODEL=qwen3-coder-30b` for the passing control.

## Status of the benchmark

The four-way run serves pie under **strategy A**, which has no cross-turn reuse
while vLLM runs `--enable-prefix-caching` and llama.cpp keeps its slot cache.
That handicap belongs in the results, not a footnote, and it means the four-way
is **not comparable to the 30B three-way** on pie's side — that run used
strategy B.

## What would have caught it sooner

Nothing in the timing columns: an arm that fails every turn finishes *faster*.
`swe30_four_way.sh` now reports non-empty patch count per arm and says plainly
that an all-empty arm is broken rather than inaccurate.

The deeper gap is that the guest asks for a capability the driver silently
declines to implement. Metal refuses the buffer *read* path loudly; it should
refuse a **zero fold length it is going to ignore** just as loudly, rather than
validating the shape and folding anyway. That one-line refusal in
`compose.cpp` would have failed the very first buffered fire instead of the
next turn's resume.

---

# Open: the inferlet is killed at ~53.5k tokens of context — 2026-08-18

Separate from everything above, and NOT caused by session resume or by the
retention policy.

## Signature

```text
[opencode-session] turn 206 cached=53221 delta=219 cue=7 gen=74 retained ... (len 53440)
[session-shim] inferlet terminal event error: 'WebSocket connection closed'
[session-shim] reconnect attempt failed: ConnectionError('WebSocket connection closed')
[session-shim] session inferlet up: opencode-session@0.1.0 process=<new>
```

The guest process dies. The ENGINE keeps running and logs no reason —
`grep -iE "warn|error|kill|fatal|starv|abort|panic"` over the engine log for
that window returns only the unrelated boot-time rs-slot capping warning. The
shim reconnects and the run CONTINUES, having silently lost every retained
branch, so the event is invisible in the timing columns, in the patch count,
and in the degraded-turn count. `tools/pie_watchdog.sh` keys on the shim's
`terminal event error` for exactly that reason.

## What it is not

| hypothesis | test | result |
|---|---|---|
| retained branch riding at 82% of the pool | refuse to retain over budget | **still crashes** — the refusal fires, the kill follows three lines later |
| KV pool starvation | that failure has its own explicit message | different signature, and the pool had room |
| session resume | reuse was working right up to it (`cached=53221`, delta 219) | unrelated |

The refusal experiment is decisive: a conversation that large has its pages
resident DURING the turn whether or not the branch is retained afterwards, so
retention was never the pressure. That change was reverted.

## What it is

Size, and reproducibly close: **53,440** and **53,562** tokens on two separate
runs. It very likely also covers the `pie_metal_register_program failed with
status -5` failures left unexplained earlier — every one landed at a 45k–54k
prefill, and `abi.cpp` discards the reason in `catch (...)`, so a large-context
fault there would present exactly as an opaque `DRIVER_ERROR`.

Nothing about it is specific to strategy B; strategy A drives the same prefill
sizes and should reach it too. That is untested.

## Why it matters for the four-way

An instance whose context passes ~53.5k loses its session mid-run. The harness
does not fail — opencode keeps going against a fresh process — so the arm
reports a normal instance with a worse patch. Any pie accuracy number carries
that, and the honest thing is to report the crash count beside it.

## Next

1. Surface the swallowed reason: `abi.cpp` wraps `register_program` in
   `catch (...)` and returns `PIE_STATUS_DRIVER_ERROR`, discarding `what()`.
   That is the one piece of evidence that would settle whether the -5s and this
   kill are the same fault.
2. Bisect the threshold with a synthetic single-conversation ramp — it is
   cheap, needs no agent, and 53.4k/53.6k suggests a hard edge rather than
   pressure.
3. Check strategy A at the same context size, to confirm this is not a
   session-inferlet bug.
