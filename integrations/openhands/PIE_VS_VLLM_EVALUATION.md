# Pie as a serving backend: can it outperform vLLM+APC? — Aggregated evaluation

*Final writeup of the OpenEvolve and OpenHands integrations. Model: Qwen3-Coder-30B-A3B-Instruct. Hardware: single RTX PRO 6000 Blackwell (96 GB). Baseline engine: vLLM with automatic prefix caching (APC). 2026-07-21.*

---

## 0. Bottom line

On a **single local GPU with vLLM APC enabled, Pie does not beat vLLM on wallclock** for our workloads. Its best observed case is a **tie**; in the agentic setting it is **~3.65× slower**. The cause is structural:

> **vLLM's APC captures the same KV reuse that Pie's fork/session provides.** When the working set fits in cache, both engines avoid the same prefill, so Pie's headline mechanism is not a differentiator — and Pie then loses on secondary overheads.

Pie's mechanisms *work* — they eliminate 95–97 % of prefill in agentic loops and fork shared prefixes perfectly — but that saving does not convert to wallclock, because the baseline already gets the same reuse for free. The Pie paper's real wins (1.3–3.4× on agentic workflows) are **genuine but conditional**: they need a *small model + a real client↔server network gap + many I/O interactions + enough concurrency to make APC evict + tools that run inside the inferlet*. Our environment removes every one of those conditions.

This document catalogs the integration variants we built, every experiment and hypothesis we tested, and where a real Pie win could still live.

---

## 1. Why the two engines tie on reuse (the through-line)

vLLM's **APC** hashes KV-cache blocks by token content and reuses them *across* requests. A multi-turn agentic loop is, turn by turn, an **append-only growing prefix**:

```
turn 1:  [system + task]
turn 2:  [system + task + assistant_1 + tool_result_1]
turn 3:  [system + task + assistant_1 + tool_result_1 + assistant_2 + tool_result_2]
```

A growing prefix is exactly what APC reuses: on turn 3 it recognizes turns 1–2 as cached and prefills only the new tail — the **same computational saving** Pie's session gives. So "multi-turn / agentic" is *not*, by itself, a Pie advantage. Pie beats APC only when the app does something that is **not** a plain growing prefix (mid-context drop/edit, mid-decode branching, custom attention) — or when the *transport/round-trip* around the reuse dominates (network gap), or when the tool runs *inside* the inferlet (co-location).

---

## 2. The integration variants we built

### 2.1 OpenHands (agentic SWE-bench)

OpenHands drives an agent that reads a GitHub issue, explores the repo, edits source, runs the tests, and submits a patch — a long multi-turn loop with heavy tool use. We wired it to Pie **four ways**, separated by two design axes: *where the agent loop runs* (a Python client vs inside the WASM inferlet), and *how the KV context is carried between turns* (re-established each call vs kept live).

| Variant | Inferlet | Loop runs in | KV across turns | Per-turn cost | Job |
|---|---|---|---|---|---|
| **Baseline** | — (litellm→vLLM) | Python client | vLLM APC (implicit) | re-send full history; APC reuses prefix | 18962143 |
| **"pie" completion** | `openhands-completion` | Python client | generic prefix-hash only | one stateless completion/call | — |
| **Pattern B** — coder-session | `openhands-coder-session` | Python client | named-snapshot session (= APC-equivalent) | **re-open + refresh snapshot every call** | 18951267 |
| **Pattern A** — agent | `openhands-agent` | **the inferlet** | **live in-memory ctx** (never torn down) | append the new turn only | 19080944 |
| **Fork / delegation** (Slice 3) | `openhands-coder-session` (fork) | Python client | fork share of parent's KV | child forks parent; ~0 re-prefill (99.5–99.7% saved) | 19039792 |

**Pattern B vs Pattern A is the load-bearing distinction.** In Pattern B the Python loop calls a *stateless* inferlet once per turn; the inferlet re-opens the session's KV snapshot and refreshes it (`delete + save`) on every call. In Pattern A the whole loop lives inside one long-running inferlet, so the KV is created once and simply appended to — no per-call teardown. Experiments #2/#6/#11 show per-call teardown was most of Pattern B's 3.65× deficit.

#### Anatomy of Pattern A (`openhands-agent`)

One `Context` is created, seeded with system prompt + task, then driven through a fixed step loop. Every step's action is **grammar-constrained to a flat JSON schema** (`constrain_with(JsonSchema)`) — `{thought, action, command, path, old_str, new_str, …}` with `action ∈ {bash, edit, read_file, insert, undo_edit, finish}`:

1. **Condense if needed** — before generating, if the effective context nears `context_token_limit`, either *rebuild* (summarize dropped turns + re-prefill) or *mask* (drop stale-middle KV out of attention, 0 re-prefill); a cooldown prevents condense-every-step.
2. **Generate the action** — decode a grammar-constrained JSON step into the *same* live `ctx` (the last step is forced to a `finish` schema).
3. **Yield the GPU** — `ctx.idle()` releases the KV pages while the tool runs, so a slow bash/test call doesn't hold GPU memory idle.
4. **Run the tool on the host** — POST a `ToolRequest` over HTTP to `tool_server.py`, which executes it against the real repo and returns an observation.
5. **Append & loop** — truncate the observation, append it as the next user turn, record per-step metrics, and repeat until `finish` or `max_steps`.

**What runs where** (the WASM/host split CodeACT avoids):

| Inside the inferlet (WASM) | On the host (`tool_server.py`) |
|---|---|
| The whole agent loop & control flow | `bash` via `PersistentBash` (preserves cwd/env) |
| Grammar-constrained generation + live KV ctx | File edits via OpenHands `FileEditor` (view / str_replace / insert / undo) |
| Condensation, robustness guards, metrics | Running the repo's tests |

**Why the split exists:** OpenHands' real tools need a host shell + Python env + the repo checkout, so they **cannot run inside a WASM inferlet**. Unlike CodeACT — whose tool is Boa-JS `eval` and *does* run in WASM — Pattern A must proxy tools over HTTP. So it keeps the paper's *live-KV / no-re-prefill* benefit but not the *co-location / no-round-trip* benefit (and on localhost the round-trip is ≈ 0 anyway).

Two more capabilities sit on the same structure. **Robustness guards** catch truncated JSON, degenerate "symbol-soup" output, and stuck read↔edit / repeated-bash cycles, nudging or force-finishing rather than looping forever. And a **test-time-scaling fork** (`num_branches` / `branch_at_step`) runs a shared trunk to step *k*, then forks both the live KV context *and* the tool-server workspace into N candidate trajectories that share steps 1…*k* — a branch a stateless client cannot express cleanly.

### 2.2 OpenEvolve (MAP-Elites genetic code evolution)

| Component | File | What it is |
|---|---|---|
| **Generation inferlet** | `openevolve-generation` | B2 **"analyze-then-diverge"**: generate a shared analysis A once, fork N children off `P+A` in-memory, per-leaf steer → N diffs. 3-level named-snapshot key tree (L0 / L1p prompt / L1g +generated-analysis / leaf) for cross-worker reuse. |
| **PieLLM backend** | `openevolve/llm/pie.py` | `LLMInterface` over `PieClient`; `generate_children()` batched path; `compute_topk_sig()`. Drop-in via `init_client=PieLLM`. |
| **Controller wiring** | `ensemble.py` / `process_parallel.py` | Least-invasive: per-worker result stays one child; same-parent workers reuse the shared **L1g** analysis snapshot **cross-process** (`Context::open` by name). |

### 2.3 Supporting / diagnostic

- `parallel-generation`, `tree-of-thought` — fork fan-out templates.
- `decode-bench` — decode-throughput diagnostic (built this session).
- **mask-condense** — drop-middle KV condensation via `mask_range`/`mask_kvpage`; measured quality-neutral (job 19070781). Not yet wired into any agent.

---

## 3. Master table: every experiment & hypothesis

Read the verdicts as: ❌ Pie loses/no-win · 🔬 diagnostic that isolated a cause · ✅ a Pie *mechanism* provably works (but does not become a wallclock win) · — result was informational/dropped. "APC" = vLLM automatic prefix caching (the baseline's implicit reuse).

| # | Experiment & setup | Hypothesis | Result & evidence | Verdict — why | Jobs |
|---|---|---|---|---|---|
| 1 | **Fan-out fork** — one Pie call forks a ~10k-token shared prefix into N children (analyze-then-diverge, conc=1, 64 decode tok each) vs vLLM two-phase+APC, matched prefixes | A single server-side N-fork beats vLLM issuing N separate requests (saves N round-trips + N re-prefills of the shared prefix) | Tie at N=1 (5.5 s = 5.5 s); Pie **10× slower at N=64** (71.5 s vs 7.1 s). Fork *did* save prefill perfectly (`leaves_prefill=0`, **668k tokens saved**) but the N child decodes run **serially** while vLLM continuous-batches them | ❌ **loss** — the prefill saving buys nothing because decode dominates and Pie doesn't batch the forked decodes; even fixed it would only *tie* (APC also reuses the prefix) | 19074533 / 19074534 |
| 2 | **OpenHands SWE-bench head-to-head** — 13 SWE-bench-Verified instances, identical agent/tools/condenser/model; Pie `coder-session` (session KV-retain) vs litellm→vLLM w/ `caching_prompt` (APC), client on localhost | Retaining KV across agent turns makes the agentic loop faster than re-sending each turn | **Pie 3.65× slower** total (7,227 s vs 1,978 s), 3.4×/iter, at ~matched iteration counts (961 vs 906). Yet Pie saved **94.8 %** of prefill (e.g. 33,614 / 641,195 tokens), zero rebuilds | ❌ **loss** — the baseline gets the *same* reuse via APC (append-only turns are a growing prefix), localhost erases the round-trip win, so Pie's per-turn overhead dominates. *Not* divergence (iters match) | 18951267 / 18962143 |
| 3 | **Memory-pressure overcommit** — push the working set past the shared 244,704-token KV cache (~29k-tok prefixes, conc 8→32) | Pie's atomic fork + CPU-offload of cold KV beats vLLM's APC evict-then-recompute under pressure | At conc that fits, vLLM slightly faster; at **1.9× overcommit Pie OOM-crashes** (server dies) while vLLM degrades gracefully (59 s, completes via evict+recompute) | ❌ **loss** — Pie's offload never fires: the vLLM driver has **no CPU-swap backstop**, so Pie hits the same cache wall and crashes before it can exploit APC's weakness | 19068668 / 19068669 |
| 4 | **Admission factor = 1.0** — set Pie's scheduler to refuse/queue beyond capacity instead of oversubscribing 4× | Throttling admission lets Pie *survive* overcommit gracefully like vLLM | conc 8 fine; conc ≥ 16 **still CUDA-OOMs** | ❌ — the admission gate is a *soft* market-weight check that relies on swap as the physical backstop; with no swap on the vLLM driver, no config value prevents OOM | 19070291 |
| 5 | **Native-CUDA-driver offload** — investigate building `--features driver-cuda` + swap pool to give Pie a real offload path | A native driver with a working swap pool lets effective KV exceed GPU RAM → Pie survives *and* wins overcommit | Build succeeds on the cluster and the SwapPool primitives exist, but the offload is **blocked on a missing aux-IPC (PIEA) listener** in the cuda driver; a substantial port. **Abandoned** per decision to stay on the vLLM driver | — **dropped** — real engineering gap, not a quick fix; deferred | — |
| 6 | **Decode-tps vs context length** — `decode-bench` inferlet prefills L tokens, decodes exactly N; slope `(t₅₁₂−t₆₄)/448` cancels prefill to isolate pure decode tok/s; L = 2k/5k/20k/50k | Pie's decode slows down over a long retained context (would explain the #2 per-turn gap) | **REFUTED** — Pie flat **~63 tok/s** across 2k→50k; vLLM flat ~65. Pie decode ≈ vLLM, no collapse | 🔬 **isolates cause** — with prefill (#2) and decode both ruled out, the OpenHands gap must be **per-call session-management overhead**, not raw inference. Corollary: mask-condense can't be a *decode* speedup | 19078674 / 19078675 |
| 7 | **Mask-condense quality** — masked drop-middle KV (sink + keep-recent) vs full rebuild, quality measured by output ratio | Dropping middle KV from attention (no re-prefill) is quality-neutral | **YES** — ratio 1.002 ± 0.021 at P=48k (straddles 1.0), 0 re-prefill, tiny ~2 % mid-context penalty vanishes by 48k | ✅ **enabler** — proves masking is *safe* to use; sets up 7b as the structural lever | 19070781 |
| 7b | **Mask-condense inside Pattern A** — same 13 instances, `condense_mode=mask` (drop stale-middle via `mask_range`, 0 re-prefill, no summary call) vs `rebuild` (summary LLM call + re-prefill) | Doing the mid-context drop APC *cannot* express speeds up the agent | Mask engaged (7 `[condense-mask]` events, 0 summaries, 0 re-prefill), 13/13 patches — but **wash** (2,374 s vs 2,329 s, ~2 % slower) | ❌ **no wallclock win** — the per-condense saving caps at ~3 % of runtime, and engine non-determinism (see #12) swings trajectories far more than that; would only matter under memory pressure | 19089013 vs 19080944 |
| 8 | **Cross-worker L1g reuse** — OpenEvolve 30B analyze-then-diverge; workers on the same parent should reuse the shared generated-analysis snapshot across processes | The generated-analysis node (KV over *generated* tokens) is reused by sibling workers via named snapshot | **YES, organic** — reused 3 / generated 9; later same-parent workers show `l1p=opened l1g=reused` (skip ~1.8k-tok P+A prefill + analysis decode) | ✅ **mechanism works** — cross-process generated-KV reuse is real; it just doesn't become a wallclock win because APC captures the same reuse when it fits | 19065839 |
| 9 | **Fork delegation** (OpenHands Slice 3) — a critic/sub-agent turn forks the parent conversation's KV instead of re-prefilling | A delegated turn can fork the parent's live KV (something litellm/APC can't express as cleanly) | **YES** — fork engaged on all 4 cases, prefill 30k+ → 128 tokens (**99.5–99.7 % saved**), `kv_verify` passed, parent uncorrupted | ✅ **mechanism works** — biggest token saving of any experiment; wall win instance-dependent (decode still dominates) | 19039792 |
| 10 | **circle_packing (~100 iters)** — would the harder OpenEvolve task (full-rewrite, ~8k decode tok/candidate, 100 iters) make Pie win? (analysis, no GPU run) | More iterations amortize Pie's shared-prefix reuse → Pie pulls ahead | **No** — the 100 iters are *independent one-shot* generations (population), so more iters just repeat a per-iteration tie; full-rewrite is decode-heavy (Pie's parity axis), and APC captures the prefix anyway | ❌ **predicted loss** — harder task multiplies a tie, doesn't create a crossover | — (analysis) |
| 11 | **Pattern A: loop-in-inferlet** — same 13 instances via `openhands-agent` (full loop in one live inferlet, tools over HTTP to a host server) vs the per-turn `coder-session` (#2) | Keeping the ctx live across turns removes the per-call session overhead (#6) → back to parity | **Partly** — Pattern A **3.1× faster than coder-session** (2,329 s vs 7,227 s), 34 %/iter (4.99 vs 7.52), 13/13 patches; but still **1.18× slower total** than litellm | 🔬 **confirms the cause** — per-call session teardown *was* most of the #2 loss (mostly self-inflicted, not fundamental); good architecture *approaches* but doesn't beat vLLM on localhost | 19080944 |
| 12 | **Trajectory divergence** — why do Pie agent runs take different/longer trajectories than litellm? Is it the tool-call parser? | The longer/divergent trajectories come from a tool-parser mismatch | **No** — two identical temp-0 cold calls (same prompt/tools) gave different outputs 8/8: the **engine itself is non-deterministic** (`engine_det=False`), independent of parsing | — **root-caused** — sets the noise floor; also why #7b/#11 comparisons carry trajectory-variance caveats | 19039792 |

---

## 4. Key experiments in detail

### 4.1 Fan-out fork (OpenEvolve shape) — #1

Shared prefix ~10k tokens, fork into N children, conc=1, decode 64/child. Pie B2 (single-call fork) vs vLLM two-phase+APC, matched prefixes.

| N | Pie (fork) | vLLM two-phase |
|---:|---:|---:|
| 1 | 5.5 s | 5.5 s |
| 4 | 8.6 s | 5.6 s |
| 16 | 21.7 s | 5.9 s |
| 64 | **71.5 s** | **7.1 s** |

- **N=1 tie** → `pie-serve → vLLM-driver` has no overhead vs standalone vLLM.
- **Pie scales ~linearly, vLLM stays flat.** Fork saved prefill perfectly (N=64: `leaves_prefill=0`, **667,926 tokens saved**) but the N child decodes are **serialized** instead of batched into one forward pass. The huge prefill saving buys zero wallclock. Even fixing the batching would only *tie* (both reuse the prefix).

### 4.2 OpenHands SWE-bench head-to-head — #2

13 SWE-bench-Verified instances, identical agent/tools/condenser/model, client on localhost.

| Metric | Pie coder-session | litellm+vLLM | Ratio |
|---|---:|---:|---|
| Total time | 7,227 s | 1,978 s | **Pie 3.65× slower** |
| Total iters | 961 | 906 | ~matched |
| Per iteration | 7.52 s | 2.18 s | **Pie 3.4× slower** |

- KV-retention *works*: 94.8 % prefill saved (django-12276: 33,614 / 641,195 tokens), **zero rebuilds**, 95–97 % saved across all instances. **Prefill is not the problem.**
- **Not divergence** — iteration counts match, so Pie isn't doing more work.
- The baseline gets the same reuse via APC, and localhost erases the round-trip win. The coder-session uses **no masking** — it is naive *prefix-extend-or-rebuild*, functionally identical to APC. Both win-mechanisms neutralized → Pie loses on per-turn overhead.

### 4.3 Decode throughput vs context length — #6 (the decisive isolation)

A `decode-bench` inferlet prefills an L-token prompt and decodes *exactly* N tokens (no stop). Slope method `(t₅₁₂ − t₆₄)/448` cancels prefill → pure decode tok/s.

| Context L | Pie decode | vLLM decode |
|---:|---:|---:|
| 2 k | 63.1 tok/s | 66.0 tok/s |
| 5 k | 62.8 tok/s | 66.3 tok/s |
| 20 k | 63.3 tok/s | 65.0 tok/s |
| 50 k | 63.1 tok/s | 64.8 tok/s |

**Both flat, within ~4 %.** Pie decode does **not** collapse with context. Combined with §4.2 (prefill saved), this **rules out both raw-inference axes** as the cause of the OpenHands loss. The remaining suspect is the **session-management path** — per-call snapshot open + refresh (`delete + save`), `kv_verify`, or decode-over-a-restored-snapshot. Corollary: **mask-condense would not speed up decode** (already at parity); its value is purely the structural mid-context drop APC can't express.

### 4.4 Pattern A: loop-in-inferlet — #11 (job 19080944)

Same 13 instances, but `openhands-agent`: the whole loop runs in one long-lived inferlet with the **KV context live across all turns** (no per-call session teardown), tools proxied over HTTP to the host tool server. This tests whether §4.3's suspected per-call overhead is the culprit.

| Arm | Total (13 inst.) | Iters | s/iter |
|---|---:|---:|---:|
| litellm + vLLM (baseline) | 1,978 s | 906 | 2.18 |
| **Pattern A** (live ctx) | **2,329 s** | **467** | **4.99** |
| Pattern B (coder-session, per-turn) | 7,227 s | 961 | 7.52 |

**Hypothesis confirmed (partly).** Pattern A is **3.1× faster than the coder-session in total wallclock** and **34 % faster per iteration** (4.99 vs 7.52 s/iter), with 13/13 non-empty patches. So the per-call session open/refresh/save overhead — exactly what §4.3 isolated — was a **real, large** contributor; the coder-session's 3.65× loss was **mostly an architectural artifact of the stateless-per-turn design, not fundamental to Pie**.

**But it still does not beat the baseline** — 1.18× slower total, ~2.3×/iter. Two caveats on the residual: (1) Pattern A is a *different, simpler* agent loop (467 vs 906 iters — it finished in fewer steps, which flatters its total time); (2) since decode is at parity (§4.3), the per-iter residual is most plausibly **tokens-per-iteration** differences (Pattern A's verbose thoughts), not a Pie inference gap. Net: the good architecture nearly closes the gap on localhost but does not cross it — consistent with §1.

### 4.4b Mask-condense inside Pattern A — the structural (APC-impossible) lever

`openhands-agent` supports two condensation strategies: **rebuild** (summarize dropped turns via an LLM call, then re-prefill a fresh context) and **mask** (drop stale-middle turns via `mask_range` — 0 re-prefill, no summary call — a KV operation APC cannot express). Same 13 instances, rebuild (job 19080944) vs mask (job 19089013):

| Arm | Total | Iters | s/iter | Condensation | 13/13 |
|---|---:|---:|---:|---|---|
| Rebuild | 2,329 s | 467 | 4.99 | 11 × (summary LLM call + ~12k re-prefill) | yes |
| Mask | 2,374 s | 441 | 5.38 | 7 × mask-range (0 summary, 0 re-prefill) | yes |

**Mask engaged correctly** (`[condense-mask] masked KV [2547,15065) (25 turns dropped, 12 kept)`, zero summaries, zero re-prefill) and kept 13/13 non-empty patches. **But it is a wash on wallclock (~2 % slower, not faster).** Why:

- The only cost mask removes is per-condense: ~11 × (~6 s summary + ~1.5 s re-prefill) ≈ **~80 s ceiling on a ~2,330 s run → ~3 % at absolute best**.
- The Pie engine is **non-deterministic** (`engine_det=False`), so the two runs took different trajectories (467 vs 441 iters; one instance swung 245 s→679 s). **Trajectory variance dwarfs the ~3 % condense saving.**
- Masking does **not** speed up decode (§4.3 — decode is flat with context length), so shrinking the attended context buys nothing there.

**Conclusion:** even the one structural lever APC cannot match yields no measurable wallclock win on a single GPU — its ceiling is ~3 % and run-to-run variance hides it. It would matter under **memory pressure** (rebuild's re-prefill compounds / APC evicts while mask keeps KV cheap) or with a slow/expensive summarizer — regimes not present here. To isolate the pure condense-cost delta without trajectory noise, replay one fixed trajectory through both condensers (the inferlet's `dump_trajectory` supports this).

### 4.5 Memory-pressure / overcommit — #3–5

Both engines share the same 244,704-token KV cache. At conc that barely fits, vLLM is slightly faster; at 1.9× overcommit, **Pie OOM-crashes** (the vLLM driver has no CPU-swap backstop; admission throttle is soft and relies on swap) while **vLLM degrades gracefully** (APC evict+recompute). The "atomic fork + offload beats APC-evict-recompute" thesis does not hold on this driver — Pie hits the wall at the same working-set size and crashes before it can exploit APC's weakness. Real offload lives only in the native CUDA driver's swap pool, which is **not wired into the embedded/vLLM path** (blocked on porting a PIEA aux-IPC listener into the cuda driver). We chose not to pursue that build.

### 4.6 OpenEvolve mechanism validation — #8

The `openevolve-generation` inferlet + PieLLM backend + controller wiring ran end-to-end on 30B (analyze-then-diverge, job 19065839): 12 valid children, best score → 1.4994, and **cross-worker L1g reuse fired organically** (reused 3 / generated 9). The fork/session mechanisms are correct and the token savings are real — but, as everywhere, they don't convert to a wallclock win because APC captures the same prefix reuse when it fits.

---

## 5. Where the paper's agentic win comes from (code-grounded)

The paper's ReACT/CodeACT/Swarm inferlets share one structure (ReACT shown):

```rust
let mut ctx = Context::new(&model)?;                  // ONE context, created ONCE
ctx.system(SYSTEM_PROMPT); ctx.user("Question: …"); ctx.cue();
for step in 1..=max_steps {                           // 8 (ReACT/CodeACT), 32 (Swarm)
    let raw = ctx.generate(…).await?;                 // decode action — appends to SAME ctx
    let observation = match tool {
        "Calculator" => calculator(arg),              // ← TOOL RUNS IN-PROCESS
        …
    };
    ctx.user("Observation: {observation}…"); ctx.cue(); // append to SAME live ctx
}
```

CodeACT is identical with `js::eval(code)` (embedded Boa JS in WASM); Swarm gives each role-agent its own `ctx` with server-side pub/sub hand-off. The loop lives **inside the server**, the tool runs **in-process between steps**. Versus a Python-client baseline, Pie eliminates three per-interaction costs:

| Cost | Baseline (client + vLLM) | Pie inferlet |
|---|---|---|
| Client↔server round-trip | one per step (tens of ms) | **none** — tool in-process |
| Re-transmit + re-tokenize history | every step | **none** — context server-side |
| Re-prefill history | if APC evicted during the gap | **none** — KV pinned in `ctx` |

These dominate **only** in the paper's setup, and vanish in ours:

| Win condition | Paper | Our runs |
|---|---|---|
| Model | 1B/3B (gen ≈ few ms → RTT is a big fraction) | 30B (gen dominates) |
| Client | remote, campus network (real RTT) | **localhost** (RTT ≈ 0) |
| # I/Os | 8 / 8 / 32 (win scales) | fewer, diluted by long gens |
| Concurrency | up to 128 agents → APC evicts | fits → APC never evicts |
| Tool location | **in-WASM** (Rust fn, Boa JS) | bash/edits → **must be on host** |

---

## 6. Where a real Pie win could still live

None is single-GPU-localhost-wallclock:

1. **A genuine network gap** between workers and the server → vLLM pays re-transmit + round-trip that Pie's server-side session avoids. Paper's dominant small-model mechanism; does not trigger Pie's OOM.
2. **Mask-condense (structural, APC-impossible)** → at condensation points the summarizing baseline emits new text → APC miss → re-prefill; Pie `mask_kvpage`-drops the stale middle and keeps the rest valid → zero re-prefill. Not a decode speedup (§4.3), a re-prefill-avoidance + capability win.
3. **Effective-KV > GPU with working offload** → only regime where fork+restore structurally beats APC-evict-recompute; blocked on the native-driver swap path (§4.5).
4. **Fix decode batching across forks** → turns the fan-out *loss* (§4.1) into a tie, not a win.
5. **Reduce per-call session overhead** (Pattern A, §4.4) → best case parity, removes the 3.65× loss.

---

## 7. Conclusions

- **Pie's KV-reuse mechanisms are real and large** (94.8 % prefill saved in OpenHands; 668k tokens in fan-out; organic cross-worker reuse in OpenEvolve; 99.5 % on fork delegation). They are **not wallclock wins** because vLLM APC captures the same reuse for free when it fits.
- **On a single local GPU, Pie's best case is a tie** (proven by the N=1 fan-out result and decode-parity). Every tie-breaker is either blocked by a Pie engine gap (serial fork-decode; OOM under pressure; per-call session overhead) or matched by APC.
- **Architecture matters a lot for Pie's overhead.** The stateless-per-turn coder-session (Pattern B) was 3.65× slower; the loop-in-inferlet agent (Pattern A) recovered most of that (3.1× faster than B), landing at 1.18× slower than the litellm baseline. So the huge OpenHands deficit was mostly self-inflicted by the per-turn session teardown, not fundamental — but even the good architecture only *approaches* parity on localhost, it does not beat vLLM+APC.
- The **paper's agentic win is genuine but conditional** — small model, real network, many I/Os, concurrency, **in-WASM tools** — and our Python/bash-tool workloads on localhost with a big model satisfy none of them.
- The **honest, defensible contribution** of these integrations is **deterministic prefill/token elimination + programmability** (expressing strategies APC cannot — mid-context drop, mid-decode fork, custom attention), which matters for compute/energy/cost, *not* a wallclock win — unless we build one of the §6 conditions.

---

## Appendix: artifacts & job IDs

| Thing | Files | Jobs |
|---|---|---|
| Fan-out | `bench_fanout_{pie,vllm}.sbatch`, `ab_bench.py` | 19074533 / 19074534 |
| OpenHands head-to-head | `bench_parity_30bcoder_13.sbatch`, `bench_litellm_baseline_30bcoder_13.sbatch`, `logs/{parity,litellm_base}_30bcoder_13_*.out` | 18951267 / 18962143 |
| Decode-tps | `inferlets/decode-bench`, `decode_tps_bench.py`, `bench_decode_tps_{pie,vllm}.sbatch` | 19078674 / 19078675 |
| Pattern A | `bench_agent_30bcoder_13.sbatch`, `inferlets/openhands-agent`, `tool_server.py`, `logs/agent_30bcoder_13_19080944.out` | 19080944 |
| Fork delegation | `bench_slice3_delegation.py/.sbatch` | 19039792 |
| Memory pressure | `ab_pressure_{pie,vllm}.sbatch` | 19068668 / 19068669 |
| OpenEvolve | `inferlets/openevolve-generation`, `openevolve/llm/pie.py`, `run_pie_evolution*.sbatch` | 19063941, 19064585, 19065640, 19065839 |
| Mask-condense | `inferlets/mask-condense-bench`, `bench_mask_condense*.sbatch` | 19070781 |
| Paper | `pie.pdf` (Gim, Ma, Lee, Zhong — SOSP '25) | — |
