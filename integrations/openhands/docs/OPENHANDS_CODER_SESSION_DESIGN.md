# openhands-coder-session — OpenHands Accelerated by Pie (Design)

Status: Phases 1+2 implemented (2026-07-17) — see "Implementation notes" at the end.
Phase 2 equivalence run: Slurm job 18709682 (`bench_session_equiv_smoke.sbatch`).
Date: 2026-07-17
Predecessor docs: `TOOL_CALL_HISTORY_REPLAY_DESIGN.md`, `PIE_OPENHANDS_WRITEUP.md`

## Motivation

Pattern A (`inferlets/openhands-agent`) is a *different agent* from OpenHands — custom
system prompt, custom recovery nudges, grammar-constrained decoding, different tools and
condensation. Its accuracy delta vs the litellm baseline is therefore confounded by
harness engineering (measured directly: empty-finish recovery alone was worth
34% → 38% on the 50-problem eval). It cannot support the claim "same accuracy, only
speedup."

This doc designs the missing piece already named in `openhands-completion`'s docstring:
**`openhands-coder-session`** — the stock OpenHands agent, unchanged, with Pie replacing
only the LLM transport, *statefully*. The goal is a defensible claim:

> Identical OpenHands agent, identical trajectories at temperature 0,
> N× less prefill compute per step.

## Core principle

The OpenHands agent loop stays on the host, completely stock: its system prompt, tools,
`LLMSummarizingCondenser`, stuck detector, fake-user responses — untouched. Every token
the model sees must be **byte-identical** to what the litellm baseline would send; what
changes is only how many of those tokens get re-prefilled. That invariant is what makes
"pure speedup" defensible — and it is exactly the invariant Pattern A broke.

The hard fidelity prerequisite (byte-identical replay of tool-call history through the
chat template) is already solved — see `TOOL_CALL_HISTORY_REPLAY_DESIGN.md`.

## Supporting API surface (already exists)

From `sdk/rust/inferlet/src/context.rs`:

- `Context::save(name)` / `open(name)` / `take(name)` / `delete(name)` — named contexts
  that persist across inferlet invocations (`demo-persistent-kv` is the existence proof).
- `Context::fork()` — copy-on-write context copies sharing committed pages
  (`demo-parallel-fork`).
- `Context::snapshot()` — anonymous save.
- `Context::suspend()` — release pages; restoration is bid-driven.
- `Context::idle()` — rent break while waiting (used by Pattern A during tool calls).
- `Context::truncate(n)` — roll back working-page tokens.

Client side: `pie_client` supports per-request calls (current `PieLLM._call_pie`),
`signal_process` for host→process messaging, and `launch_daemon` if a long-lived server
inferlet is ever preferred.

## Architecture

### 1. Persistent per-conversation KV via named contexts

Keep `PieLLM._call_pie` per-request (no long-lived process), but add a `session_id`
to the request contract. The inferlet:

1. `Context::open(model, session_id)` if it exists, else build from scratch and `save`.
2. The host sends the **full** message list every call (semantics unchanged), plus the
   token count/hash of the previous prompt. The inferlet compares:
   - **Pure extension** (common case — new tool-result messages appended; the assistant
     turn's tokens are already in KV because the model generated them there): append
     only the delta. Prefill cost per step drops from O(full history) to
     O(new observation).
   - **Not an extension** (condenser rewrote history, retry mutated messages): rebuild
     from the prefix checkpoint (§2). Rebuild is always semantically safe — same
     tokens, just slower — which is what makes the design low-risk.

Extension detection must be done at the **token** level, not the message level:
replay-render the previous prompt and check it is a prefix of the new one. Renders are
deterministic (per the replay design doc), so a length + rolling-hash comparison
suffices.

### 2. Shared static prefix via fork()

The OpenHands system prompt + tool schemas are identical across all SWE-Bench instances
and run to several thousand tokens. Prefill once, `save("oh-prefix-<model>")`, start
every conversation from a `fork()` of it.

vLLM's automatic prefix caching (APC) does this best-effort and evicts under pressure;
Pie makes residency explicit and guaranteed. This is the first clean demonstration of
programmable KV control the baseline cannot express.

### 3. KV-aware condensation (flagship win)

When `LLMSummarizingCondenser` fires in the baseline, two expensive things happen:
the summarizer call re-prefills the entire history being summarized, and the next agent
step re-prefills the rewritten history (APC misses — the prefix changed). With Pie:

- Run summarization on a `fork()` of the live context — the turns to summarize are
  *already in KV*; append only the summarize instruction and generate. Near-zero
  prefill.
- Rebuild the post-condense context by forking the saved prefix checkpoint and
  appending only `keep_first` events + summary + kept tail.

Wiring: either subclass `LLMSummarizingCondenser` so its summarizer LLM call routes
through the same session (flagged "summarize-on-fork"), or have the inferlet recognize
the condenser's request shape. The condensation **policy and output text** stay
identical to stock OpenHands — only where the FLOPs happen changes.

### 4. suspend() during tool execution

OpenHands tool calls (pytest runs especially) take seconds to minutes while KV sits
resident. In the per-request design the inferlet returns between steps, so use
`suspend()` on the saved context (release pages, bid-driven restore) rather than
`idle()`. This is a **throughput** win, not per-conversation latency: more concurrent
SWE-Bench instances per GPU at matched accuracy. Report as instances/GPU-hour.

### 5. Session lifecycle

- `Context::delete` when the conversation ends or errors.
- TTL sweep for leaked sessions — a killed harness process must not pin KV forever.
- This bookkeeping is the main new failure mode; build it early.

## Fidelity verification (before any benchmark)

Add a `--kv-verify` mode to `PieLLM`: on every call, also render the full prompt from
scratch and assert its token IDs equal the session context's accumulated IDs (compare
`seq_len` + a rolling hash returned by the inferlet). Run several instances end-to-end
with it on.

Known trap: token merging at append boundaries. Appending at message boundaries through
`ctx.user()` / `answer_batch_prefix` (which emit the template's special tokens) keeps
boundaries stable — but verify, don't assume.

Also hold sampling identical between arms: same temperature, **no grammar constraints**,
and tool-call parsing matching the baseline vLLM `qwen3_coder` parser.

## Measurement design (honest baseline)

Compare against vLLM **with automatic prefix caching enabled** (it is on by default —
check the baseline config and state it in the writeup either way). APC already recovers
much of the quadratic tax for a single sequential conversation, so the honest framing of
where Pie wins:

1. **Guaranteed vs best-effort reuse** — under concurrency, APC evicts; Pie sessions
   don't (or degrade controllably via bids). Sweep concurrency; plot
   prefill-tokens-computed-per-step for both arms.
2. **Condensation** — APC structurally cannot reuse across a history rewrite;
   fork-summarize can. Measure the condensation-step cost spike in both arms.
3. **Memory release during tool waits** — instances/GPU-hour.
4. **Accuracy** — report litellm-vs-pie-session resolve rate as an *equivalence check*
   (match within noise), not a delta to brag about. With `--kv-verify` passing and
   temperature 0, trajectories should be near-identical token-for-token — the strongest
   evidence for "pure speedup."

## Phasing

| Phase | Deliverable | Why first |
|-------|-------------|-----------|
| 1 | Session contexts + delta append with rebuild fallback (`save`/`open` + prefix check) | Biggest per-step win, smallest risk |
| 2 | `--kv-verify` mode + 5-instance equivalence run | Locks in the accuracy-neutrality claim |
| 3 | Shared prefix checkpoint via `fork()` | Cheap, demonstrates explicit KV control |
| 4 | Fork-based condensation | Flagship capability APC can't match |
| 5 | `suspend()` during tools + concurrency sweep | Throughput story |

Phases 1+2 alone support the headline sentence: identical OpenHands agent, identical
trajectories at t=0, N× less prefill compute per step.

## Relationship to existing work

- `openhands-completion` (exists): stateless per-request transport; rebuilds the entire
  conversation each call. Stays as the fallback path and the Phase-0 baseline.
- Pattern A `openhands-agent` (exists): reframe in the writeup as a *co-designed
  agent + runtime* contribution, not "OpenHands accelerated." Its accuracy deltas are
  confounded by harness differences (custom prompt, empty-finish/test nudges, stuck
  hints, constrained decoding, different condensation and tools).
- Useful Pattern A ablation: a host-side Python twin of Pattern A (same prompt, same
  JSON protocol via vLLM guided decoding, same nudges) to separate Pie's runtime
  contribution from harness engineering.

## Implementation notes (Phases 1+2, 2026-07-17)

Code:

- `inferlets/openhands-coder-session/` — new crate; same contract as
  `openhands-completion` plus the session protocol (`session_id`,
  `session_prev_len`, `session_prev_hash`, `session_action`, `kv_verify`,
  `use_grammar`), returning a `session {id, mode, len, hash, prefill_tokens}`
  block. Stateless when `session_id` is absent.
- `pie_openhands/llm.py` — `pie_session` / `pie_kv_verify` fields, host-side echo
  of the previous render's (len, hash), `close_pie_session()`, per-call telemetry
  via `pie_session_summary()`.
- `benchmarks/run_swe_bench.py` — `--pie-session`, `--kv-verify`;
  `run_pie_backend.sh` — `BACKEND=pie-session` (+ `KV_VERIFY=1`);
  `benchmarks/compare_equivalence.py` + `bench_session_equiv_smoke.sbatch` — the
  Phase 2 two-arm equivalence run.

Decisions that differ from (or sharpen) the sketch above:

1. **Prompt-only snapshots.** The saved context always equals the *canonical
   render* of the message list, refreshed (delete + save) after prefill but
   before generation. The model's own sampled tokens never enter the snapshot:
   after host-side argument re-serialization/sanitization, the replayed
   assistant turn's bytes can differ from what was sampled, which would poison
   every subsequent extension check. Cost: the previous assistant turn is
   re-prefilled as part of each call's delta — still O(delta), not O(history).
2. **The snapshot excludes the generation cue.** The cue's trailing `"\n"` can
   BPE-merge with the next turn's first content token, so a snapshot ending in
   the cue would fail the prefix check against the next render. Ending on a
   message boundary keeps renders concatenative and the prefix property exact.
3. **The host stores no tokens.** The inferlet returns (len, FNV-1a-64 hash) of
   each render; PieLLM echoes them back. Extension = echoed hash matches the
   new render's prefix hash AND the opened snapshot holds exactly that many
   tokens. Any mismatch (condenser rewrite, lost snapshot, a retry whose
   previous attempt updated the snapshot but never delivered) rebuilds — never
   errors, even under kv-verify, because rebuild is semantically identical.
4. **kv-verify** asserts post-prefill `seq_len == len(render)` and errors on
   violation. Token-content verification is by construction: the context is
   only ever filled with the render itself (there is no runtime API to read
   token IDs back from a context).
5. **Condenser thrash (known, accepted for now).** `build_agent` hands the same
   LLM instance to `LLMSummarizingCondenser`, so a summarizer call interleaves
   with a different prompt lineage and forces two rebuilds per condensation.
   Correct but slow; fixed properly by Phase 4 (fork-based condensation).
6. **Ops note:** `pie serve` caches installed programs by `name@version` —
   after rebuilding the wasm, reinstalling over a live server is not enough;
   restart the server (run_pie_backend.sh already boots a fresh server per run).
