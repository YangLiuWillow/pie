# Fork-based Test-Time Scaling on SWE-bench — Design

**Status:** rollout in progress. Step 1 (workspace isolation in `tool_server.py`)
implemented; steps 2–3 pending.

**Goal.** Demonstrate that Pie's session/context **fork** beats vLLM *even with
`--enable-prefix-caching` on*, using a **coding-agent** workload (OpenHands is a
coding agent). A single SWE-bench trajectory is linear/append-only — the one
reuse shape prefix caching handles optimally — so it neutralizes Pie (we measured
~3.7× slower wall, decode-bound). Test-time scaling adds **mid-trajectory
fan-out**: the agent explores the repo, then **forks K candidate solutions** that
share the exploration KV. That reuse is *not* a re-sent prompt prefix, so prefix
caching can't capture it, and Pie's `Context::fork()` (shared prefix pages) wins.

We build on **Pattern A** (`inferlets/openhands-agent`, full agent loop inside
the inferlet), because running the loop in-runtime also removes the per-call
round-trip overhead that made the hybrid path 3.7× slower (co-located I/O =
paper's Figure-6 mechanism). Pattern A already bridges the real OpenHands toolset
(`FileEditor` + `PersistentBash`) to a sandbox over HTTP (`tool_server.py`), and
uses `ctx.idle()` to yield the GPU during tool I/O.

## Core principle: KV fork and workspace fork happen in lockstep

At the branch point the agent forks **both**:
- **KV** via `Context::fork()` — shares the prefix cache (the Pie win);
- **workspace** — snapshot the trunk's repo working-tree K−1 times, so each
  branch edits files independently *from the same starting state* the branch's
  context "saw".

Both branch from the identical point, keeping each candidate's edits consistent
with its context.

## Components (three files)

### 1. `tool_server.py` — multiplex by `workspace_id`  *(Step 1 — DONE)*

- A `Workspace` bundles `(root, bash, editor)`; a `WorkspaceRegistry` maps
  `workspace_id -> Workspace`, seeded with `"0"` = the initial checkout.
- Requests carry an optional **`workspace_id`** (default `"0"` → fully
  backward-compatible; the single-trajectory path is unchanged).
- Two control actions on `/execute`:
  - `fork_workspace {from_id, to_ids:[...]}` — `cp -a --reflink=auto` the
    `from_id` working-tree into a sibling dir per `to_id`, register a fresh
    `PersistentBash` + `FileEditor` for each. Inferlet-triggered, because the
    branch point is decided at runtime inside the loop.
  - `drop_workspace {workspace_id}` — close bash, remove dir, deregister
    (`"0"` is protected).
- Server is now `ThreadingHTTPServer` so the K branches' tool calls run
  concurrently; each `PersistentBash` already has its own lock, and registry
  mutations take a lock.

Fidelity note: `cp -a` copies the working tree **including uncommitted edits**
(the trunk may have edited before branching), which `git clone` would miss.
`--reflink=auto` makes it cheap on CoW filesystems and falls back to a full copy
otherwise. Final patch per branch = `git -C <ws_id> diff HEAD`.

### 2. `inferlets/openhands-agent/src/lib.rs` — fork the loop  *(Step 2 — TODO)*

- `Input` gains `num_branches: usize` (default 1) and `branch_at`
  (`FirstEdit` | `Step(n)` | `Never`).
- `ToolRequest` + `call_tool_server` gain `workspace_id: &str`, threaded into
  the POST body.
- Loop runs the trunk (`workspace_id="0"`) until `branch_at` fires, then:
  POST `fork_workspace{from_id:"0", to_ids:["1".."K-1"]}`; `ctx.fork()` × (K−1);
  `future::join_all` over K per-branch loops, each tagging tool calls with its
  `workspace_id` and running to its own `finish`.
- Return `{branches:[{workspace_id, patch, steps, metrics}], branch_step}`.

Recommended policy: **`FirstEdit`** — fork right before the first
`edit`/`str_replace`, maximizing the shared (explored) prefix and giving branches
divergent *edit* strategies.

### 3. `benchmarks/swe_bench.py` — collect K candidates + select  *(Step 3 — TODO)*

- Forward `num_branches`/`branch_at` through `_run_agent_inferlet`.
- Iterate `result["branches"]`, write K predictions, **test-select** the resolved
  one (reuse `score_swebench_apptainer.py`) → best-of-K.
- Report best-of-K resolve rate (quality) **and** fork prefill/latency savings
  (the win). Reuse `checked_out_repo` + its bare-cache locking for the trunk.

## Fair vLLM baseline  *(Step 3)*

Replicate the same branch structure via litellm: run trunk to the branch point,
then K continuations that **re-send the branch-point context** (vLLM re-prefills,
or hits its prefix cache if still resident). Compare TTFT/throughput/prefill.
Critical: branch **mid-trajectory from the generated context**, never from the
issue prompt — else prefix caching captures the reuse and the win vanishes.

## Rollout

1. **Step 1 (DONE):** `workspace_id` routing + `fork_workspace`/`drop_workspace`
   in `tool_server.py`; smoke test proving isolation (`test_tool_server.py`).
2. **Step 2:** inferlet fork (K=2 fixed-step smoke → confirm two independent
   diffs on one instance).
3. **Step 3:** best-of-K test-selection + the vLLM baseline + measurement.

## Risks / decisions

- **Disk/mem:** K working trees + K bash procs. SWE-bench checkouts ~100s MB;
  K≈4 is fine. `cp -a --reflink`; `drop_workspace` on branch finish.
- **Uncommitted-state fidelity:** use `cp -a` (not `git clone`) so forks inherit
  the exact dirty tree.
- **Baseline fairness:** branch mid-trajectory, not from the prompt.
- **Solve-rate:** Pattern A's toolset is a subset of OpenHands' full tools, so
  per-branch solve-rate may trail the coder-session track. For a *serving*
  result that's fine — report solve-rate to show quality isn't destroyed; the
  headline is throughput/latency under fork.

---

# Addendum — B2-mask: beating vLLM+prefix-caching via masked condensation

**Why this, not best-of-K:** best-of-K forks from the *trunk prefix*, which is a
clean contiguous prefix — vLLM automatic prefix caching (APC) reuses it too, so
best-of-K is a *quality* win, not a *speed* win over APC-on. To beat APC on speed
the reuse must be something APC structurally cannot hold. **Masked condensation is
that:** a long-horizon agent that DROPS stale middle turns by *masking* their KV
(keeping them resident) instead of rebuilding+re-prefilling. APC's only tool is
prefix reuse; to drop middle context it must rebuild → re-prefill the kept suffix
(positions shift). Pie masks the middle pages in place → **zero re-prefill**, and
positions are UNCHANGED (masking skips attended positions but doesn't move them —
so, unlike `import_kvpage` gather, **no RoPE re-encoding is needed**; that's why
this is buildable on the existing primitive and the gather is not).

**Primitive (exists):** `Forward::attention_mask(&[Vec<u32>])` (sdk forward.rs:217)
— per-query-position BRLE mask over KV positions. For a generated token at position
P, attend to `[0,head_end) ∪ [recent_start, P)`, skipping the masked middle
`[head_end, recent_start)`. BRLE = few runs → cheap. NOTE: the high-level
`generate()`/`GenStep` loop only threads `logit_mask`, not `attention_mask`
(generation.rs:528) — so a masked *generate* needs either (Stage 2) an SDK change
to thread a persistent masked-range set through the Generator, or a manual
Forward decode loop (Stage 1).

**Current condensation re-prefills:** `condense_context` (lib.rs:527) rebuilds a
fresh Context (system+task+summary+replayed kept turns) = full re-prefill. That's
the cost B2-mask eliminates on the Pie side (and that vLLM is stuck paying).

## Staged plan

- **Stage 1 — masked-drop MICROBENCH (no SDK change).** New inferlet with its own
  Forward decode loop: build a long context (system + N filler turns + question);
  (a) MASK a middle range via `attention_mask` and decode; (b) REBUILD (system +
  head + recent, re-prefill) and decode. Measure prefill-tokens + TTFT: mask ≈ 0
  re-prefill vs rebuild = re-prefill the kept suffix. Baseline: vLLM can only do
  (b). Headline structural number. Also sanity-check output coherence under mask.
- **Stage 2 — INTEGRATION into openhands-agent.** Track per-turn token boundaries
  (record seq_len before/after each turn append). When over budget, replace the
  rebuild in `condense_context` with: add the dropped middle turns' token ranges to
  a masked set + keep generating (needs the Stage-0 SDK plumbing to thread
  attention_mask through `generate()`). Run on LONG SWE-bench trajectories that
  actually condense; compare mask-condense (pie) vs rebuild-condense (pie) vs vLLM
  rebuild. Report TTFT/prefill saved + resolve-rate parity (masking drops detail
  like summary-condensation does — must show quality isn't destroyed).

## Risks
- **Quality under masking:** attending to head+recent while skipping masked middle
  = condensation-without-summary. Same info-loss tradeoff as summary condensation;
  needs a resolve-rate check. Could keep a short summary token span UNMASKED to
  hedge.
- **SDK change (Stage 2):** threading a persistent attention mask through GenStep
  touches core generation — do it carefully; Stage 1 avoids it via manual Forward.
- **Position-preserving is the whole point:** masking leaves kept-token positions
  intact (gaps in attention, not in position space) → no RoPE work; verify the
  runtime's attention_mask honors non-contiguous attended ranges correctly.
