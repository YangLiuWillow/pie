# The Silent-Empty Saga

*A two-day debugging story from the `rl-completions` bring-up (2026-08-06/07),
branch `liu/rl-completions`. Companion to `KNOWN_ISSUES.md`, which carries the
resulting bug ledger.*

## The setting

Phase 1b of the pie↔rllm RL integration needed one thing: replay recorded
qwen-code episodes (golden fixtures, ~9–15k-token prompts) against the new
`rl-completions` inferlet and get byte-exact token accounting back. The
inferlet had just passed its contract test — short prompts, real weights,
streaming and non-streaming, all green. The replay should have been a
formality.

It failed. And then it failed the same way for two days, on two machines,
across three models, under six different hypotheses — always with the same
error:

```
GenStep::execute forward: ForwardPass::execute: empty input —
must supply at least one input or speculative token
```

The generator, mid-episode, suddenly had nothing to feed the model. Where did
the tokens go?

## The eliminations

Each hypothesis was falsified by a controlled A/B. In order:

**1. "The inferlet's prefill leaves the generator empty-handed."** Partially
true — the first real bug. The original prefill flushed the *entire* prompt
before generating, so the generator's first forward carried zero tokens, which
the portable driver rejects (`plan: request 0 has zero tokens`). Fixed by
holding the final chunk in the buffer. The replay still failed. (Notably, this
failure mode was originally an *infinite silent retry loop*; the empty-input
guard we added at the API boundary — commit `c1b32675`, implementing a check
that a test inferlet claimed existed but never had — is what turned the whole
saga into something diagnosable.)

**2. "Prefill chunks outlive the scheduler's 120s request timeout."**
Beautiful arithmetic — the failure struck at 600.4s ≈ 5 × 120s, and the
scheduler demonstrably abandons slow requests *and returns an empty output as
if successful*. Shrinking chunks from 1024 to 256 tokens changed nothing.
Raising `request_timeout_secs` to 600 changed nothing. Real bug, wrong ceiling.

**3. "The scheduler's per-context endowment (64 pages = 2048 tokens) is a
length wall."** A probe stepping through 601/2245/3745/6001-token prompts
passed cleanly. Dead.

**4. "It's a context-length wall between 6k and 9.3k."** Extended probe:
7.5k, 9.3k, 11.2k, 13.5k tokens — all passed on the 0.6B. There is no length
wall. Whatever it was, it cared about the *model*, not the prompt.

**5. "The Mac's 8GB of RAM."** Moved everything to a 96-CPU, 503GB RunPod
host. The replay failed at 2221s — within seconds of the Mac's 2205s. Two
wildly different machines, near-identical wall time: the time wasn't compute,
and the failure wasn't memory.

**6. "Qwen3-1.7B is broken on the portable driver."** Supported by a clean
stock-code repro — `text-completion` failed identically on 1.7B while 0.6B
passed — until Qwen3-4B *also* failed, and then 0.6B failed too on the Linux
pod. The model axis was a mirage: the real variable was *how long a forward
takes on that machine for that model*.

(Honorable mention: `helloworld` "passing on the broken model" sent us down a
day of my-code-vs-engine bisection before we noticed helloworld never calls
the model at all. A control that controls nothing.)

## The instrumentation descent

With hypotheses exhausted, we switched from theorizing to tracing, adding
printouts layer by layer down the stack — and each layer said "not me":

- `[dbg-sample]` in the driver's slow-path sampler: never printed.
- `[dbg-greedy]` in the greedy fast path: never printed.
- `[dbg-dispatch]` at the sampler dispatch: **printed** — revealing a *third*
  uninstrumented path, `uniform_top_sample`, was taken.
- `[dbg-utop]` in that path: printed garbage — a "descending-sorted" top-K
  whose first entry had probability 0.0 and second had 1.0, indices misaligned
  with probabilities. A real corruption bug (KNOWN_ISSUES #3) — but disabling
  the fast path entirely and rerunning showed **healthy logits, zero NaNs, a
  valid sampled token** on the slow path… and the same starvation. The driver
  was exonerated: it was producing correct tokens that never arrived.
- `[dbg-bso]` in the runtime's slot builder and a marker in `FutureOutput`'s
  error branch: never printed. The response wasn't being mis-assembled — it
  was never assembled at all.

That left exactly one unexamined segment: the transport between scheduler and
driver.

## The bug

`runtime/src/shmem_ipc.rs`:

```rust
/// … Resolved once at startup from `PIE_SHMEM_TIMEOUT_S` (float seconds),
/// defaulting to 60s …
```

The shmem client abandons any forward that takes longer than **60 seconds**.
On abandonment, `fire_batch` returns an error — whereupon the scheduler did
this:

```rust
Err(e) => {
    tracing::error!(...);                                  // invisible: no subscriber
    for req in requests {
        req.response_tx.send(ForwardPassOutput::default()) // EMPTY, sent as SUCCESS
    }
}
```

An empty output delivered as success. The SDK's generator receives zero
sampled tokens, has nothing to feed the next step, and submits an empty
forward — producing the "empty input" error *one step after and one layer
above* the actual failure. Every observation fits: machines and models whose
forwards fit under 60s passed; everyone else failed; wall times clustered at
multiples of the ceiling; the driver's late-finishing work kept printing
healthy debug lines into the void.

Confirmation: with `PIE_SHMEM_TIMEOUT_S=900`, the exact request that always
starved — 1.7B, the model that "was broken" — completed correctly. It took
8m16s of wall clock and **59.5 CPU-minutes**, which quantified the accomplice:
the portable CPU build is ~1000× slower than it should be (KNOWN_ISSUES #2's
sibling — a tiny request llama.cpp serves in under a second). Without the perf
pathology, nothing would ever have hit the 60s ceiling.

## The fix (commit `dde622bd`)

The principle: **a forward that cannot deliver what was requested must error,
visibly, at the boundary — never fabricate an empty success.** Concretely:

- `fire_batch` failures now *drop* the response channel instead of sending
  `ForwardPassOutput::default()`, and log loudly on stderr.
- A dropped channel leaves `FutureOutput`'s result as `None`, which the WIT
  contract (`get: func() -> option<output>`) carries to the SDK as a proper
  error the inferlet sees at the failing step.
- The shmem default rose 60s → 600s, safe now that expiry is an honest error.

This was the *fifth* member of what KNOWN_ISSUES calls the silent-empty
family: scheduler-abandonment-as-empty, `finish_empty()` on errors, the
never-implemented empty-input rejection, the sampler dropping unparseable
tool calls (found during Phase 1a in rllm's gateway — same disease, different
organism), and this. Every one of them converted a specific, nameable failure
into generic corruption discovered far away.

## Morals

1. **Never convert an error into an empty success.** Every hour of these two
   days traces to some layer doing exactly that. Empty-but-successful is the
   most expensive lie a system can tell, because it passes every cheap check
   and fails only semantic ones downstream.
2. **A control must exercise the thing under test.** helloworld "passing"
   on a "broken" model cost a day. Verify the control fails when the
   hypothesis says it should.
3. **Instrument the dispatch, not the branch.** Three sampler paths, two
   instrumented, the third taken. Printing at the fork beats printing in the
   prongs.
4. **Verify the binary contains your instrumentation** (`strings | grep`)
   before trusting its silence — editable rebuilds no-op more often than
   advertised.
5. **Identical wall times on different hardware mean the clock is a timeout,
   not a computation.** 2205s vs 2221s across a MacBook and a 96-core server
   was the single most informative measurement of the saga, and it was almost
   dismissed as coincidence.
6. **Beware `pkill -f X` inside a shell whose command line contains X** — it
   kills its own session. This one bug in *our tooling* manufactured a false
   mystery (vanishing daemons and logs) inside the real one.

## Epilogue

Fixed and committed: the empty-input guard, the prefill chunking, the
errors-not-empties fix, the timeout default. Documented and open: the ~1000×
CPU perf pathology, the corrupted uniform-top-K packing (fast path disabled as
a workaround), BF16-less CUDA unary kernels in the vendored llama.cpp, the dev
driver's silent spawn death on Linux, and the macOS Metal SIGKILL. The fixture
replay — the test that started all of this — was subsequently taken to GPU
pods, where forwards take seconds and the 60-second ceiling is a memory.
