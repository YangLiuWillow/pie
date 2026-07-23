# The Fair vLLM Baseline for Fork Test-Time Scaling

*Design doc — the "big novel piece" of Step 3: measuring what
`vLLM --enable-prefix-caching` must pay for the same mid-trajectory K-way
branch structure that Pie forks at ~0.*

## Why this exists

The Pie fork test-time-scaling run (`bench_fork_bestofk.sbatch`,
`_solve_one_agent_fork` in `benchmarks/swe_bench.py`) forks a live KV context
into K branches with `copy_d2d` — ~0 re-prefill. To claim that as a *win* we
need the honest counterfactual: run the **identical** branch structure through
the LiteLLM→vLLM path with **prefix caching ON** (disabling it is a strawman —
the user's standing constraint), and measure what it costs.

**Decision (user, 2026-07-21): full multi-step continuation.** Not just the
first post-branch LLM call — the entire K-branch continuation to completion, so
the measurement captures the *compounding* re-prefill tax, not just the
one-shot fork-vs-reprefill primitive.

## The honest accounting

A naive "Pie forks at 0, vLLM re-prefills K times" overclaims. Because all K
branches share the **identical** branch-point context, vLLM's prefix cache
*hits* on branches 2…K for that shared prefix. So the win is NOT the initial
branch-point prefill (paid once on both sides). Pie's real, honest advantage
is three things the baseline must actually surface:

1. **Re-transmission + lookup cost.** Each vLLM branch call re-sends the full
   multi-thousand-token branch-point context over HTTP and re-hashes it for
   cache lookup; Pie forks KV in-GPU with zero re-send even on a miss.
2. **Eviction resistance.** Under KV pressure vLLM may evict the shared prefix
   between branches → forced re-prefill; Pie's fork holds refcounted pages.
3. **Compounding continuation (the big one).** As each branch runs *further*
   steps and diverges, prefix caching only ever covers the shared trunk; every
   subsequent step re-sends the growing per-branch context (the O(n²) tax),
   while Pie's persistent per-branch KV stays O(n). This is the same
   Pattern-A persistent-KV win, now multiplied across K branches.

## The enabling primitive: `LocalConversation.fork()`

OpenHands ships a native branch mechanism —
`openhands/sdk/conversation/impl/local_conversation.py:314` `fork()`:

- Deep-copies the event history (source stays immutable), starts the fork in
  `idle`; `run()` resumes from the copied state with full event memory.
- `reset_metrics=True` (default) — **per-branch cost/token stats start fresh**,
  exactly what we want for independent per-branch accounting.
- Auto-generates a new conversation id + persistence dir under the same base.

**Caveat — workspace:** `fork()` reuses the parent's `workspace` object
(line 372); it does not isolate the filesystem. For faithful per-branch
*resolve* quality each branch must edit its own tree, so after forking we
`cp -a --reflink=auto` the trunk workspace → `ws__vfork{k}` and repoint the
branch (mirrors the Pie side's `fork_workspace` in `tool_server.py`). For pure
*timing* the workspace doesn't affect the LLM re-prefill numbers, but we want
the best-of-K resolve number too, so isolate.

## Harness shape (`bench_vllm_fork_baseline.py`, mirrors the Pie fork driver)

Per instance, with the **same** `branch_at_step` / `num_branches` /
`temperature` / `top_p` as the Pie run, against `vllm serve --enable-prefix-
caching`:

1. **Trunk.** Build a LiteLLM agent (`build_llm("litellm", …)`,
   `build_agent`), `Conversation(workspace=trunk_ws, persistence_dir=…,
   max_iteration_per_run=branch_at_step)`, send the SWE-bench prompt, `run()`
   to the branch point. Record trunk prefill/wall.
2. **Branch.** For k in 0…K-1: `trunk.fork(conversation_id=…)`, `cp -a` trunk_ws
   → `ws__vfork{k}`, repoint the fork's workspace/tools, then
   `run_with_stuck_retries()` to completion. Capture each branch's patch
   (`capture_patch(ws_k)`) and its **fresh** metrics
   (`conv.state.stats` / `_extract_metrics`): prompt tokens (= re-prefill
   proxy), completion tokens, TTFT if exposed, wall.
3. **Emit** one Prediction per instance with `_metadata.candidates` = the K
   patches (same schema as the Pie side, so `bestofk.py split/combine` scores
   it unchanged), plus a `_metadata.vllm_fork` block with the per-branch and
   aggregate prefill/wall numbers.

## The comparison

Same instances, same branch point, two runs:

| metric | Pie fork | vLLM+APC fork |
|---|---|---|
| branch-point prefill | once (fork = copy_d2d) | once (cold) + K−1 cache-hits |
| per-branch re-prefill over continuation | **0** (persistent KV) | grows with per-branch context each step |
| context re-transmission | none (in-GPU) | full context per call over HTTP |
| best-of-K resolved | (from 19057751) | scored via same `bestofk` combine |

Report: total prefill tokens, wall, throughput per side; **and** best-of-K
resolve parity (the win must not cost quality). Match kernels/config to the
aligned toml so we measure architecture, not driver-vs-server.

## Open items

- Confirm vLLM exposes a prefix-cache-hit / TTFT metric we can read per request
  (else derive re-prefill from logical prompt_tokens minus measured cache hits,
  or instrument via the `/metrics` Prometheus endpoint).
- Repoint-workspace mechanics on a forked `LocalConversation` (settable
  workspace vs. rebuild agent tools against the new path).
- Whether to run trunk once and fork K, or (fairer to vLLM's `n` param) also
  compare against vLLM native `n`-sampling from the branch point.
