# Pie as a serving backend: can it outperform vLLM+APC? — Aggregated evaluation

*Writeup of the OpenEvolve and OpenHands integrations. Model: Qwen3-Coder-30B-A3B-Instruct. Hardware: single RTX PRO 6000 Blackwell (96 GB). Baseline engine: vLLM with automatic prefix caching (APC). First written 2026-07-21; **substantially revised 2026-07-23**.*

> ## ⚠️ Revision notice — the headline result of the 2026-07-21 version was wrong
>
> That version's central claim was that Pie is **3.65× slower** than litellm+vLLM on the agentic workload, and everything downstream ("Pie's best case is a tie", "the honest contribution is token elimination, not wallclock") was built on it.
>
> **The cause was a bug in Pie's WebSocket server, not a property of Pie.** `runtime/src/server.rs` handled an incoming Close frame with `WsMessage::Close(_) => break` and never replied. Because the stream is split, tungstenite's automatic close handshake cannot run — the reader half cannot write. The Python client (`websockets` 16.0, default `close_timeout` 10 s) opens **one connection per LLM call**, so every call sat out its full 10 s close timeout. The signature was unmistakable once measured: close/teardown took **10009.4 ms mean, 10010.4 median, 10010.5 max** across 32 calls. Sub-millisecond variance is a timeout, not work.
>
> Fixed in `e4626e5b` by echoing the frame back through the existing send channel (RFC 6455). One line; fixes every Pie client in any language.
>
> **After the fix, on the same 13 instances, the same model, the same settings: Pie is ~26 % faster than the baseline** — 1,430.4 s of solve time against 1,938.9 s (23 m 50 s vs 32 m 19 s), summed per-instance from each run's own log. The sections below are revised accordingly, and every superseded claim is marked ~~struck~~ rather than deleted — the wrong numbers are part of the record, and the *reason* they were wrong is the most transferable lesson here.

---

## 0. Bottom line

On a single local GPU with vLLM APC enabled, **Pie is faster than vLLM on the agentic workload**: 13 SWE-bench-Verified instances in **1,430.4 s vs the baseline's 1,938.9 s — ~26 % faster, a 1.36× speedup** — at 95 %+ KV reuse and zero `kv_verify` errors.

Two things had to be true for that to show up, and neither was about inference:

1. **The transport had to stop burning 10 s per call** (`e4626e5b`, above). This alone took one instance from 467.6 s to 185.4 s.
2. **Session snapshots had to be released at teardown** (`f1aac6eb`). The APC rewrite began saving dozens of snapshots per conversation under `apc/{sid}/…`, but the inferlet's delete branch still removed only the legacy single-slot `oh-session-{id}` name, so cleanup was a silent no-op. On the native CUDA driver, whose KV is a hard allocation with `swap_pool_size=0`, the cache did not evict — it **blocked forever**. The first A/B run wedged at 4 of 13 instances.

The original conclusion — that Pie's KV reuse is real but never converts to wallclock — was **half right for the wrong reason**. The reuse is real (§1 still holds: APC captures the same prefix reuse when it fits, so prefill saving is not where the win comes from). But Pie was not losing to APC; it was losing to its own dead time. With that removed, Pie leads on **decode throughput**: across the run's 416 calls it generated 97,343 tokens at **90.1 tok/s of pure decode, 80.5 tok/s** once prefill and the whole APC apparatus are included, against a baseline measured at ~64 tok/s per call. That is an axis the earlier analysis had measured as *parity* (§4.3) and therefore stopped examining.

**What is still not established: an accuracy claim.** Pie reproduces 11 of the baseline's 13 resolved instances. That set *is* the baseline's own `resolved_ids`, so the baseline scores 13/13 by construction and parity is the ceiling — the comparison cannot show Pie more accurate, only less. A neutral 50-instance run is in flight (§4.7).

---

## 1. Why the two engines tie on *reuse* (the through-line)

*This section survives the revision unchanged — it was never the flawed part.*

vLLM's **APC** hashes KV-cache blocks by token content and reuses them *across* requests. A multi-turn agentic loop is, turn by turn, an **append-only growing prefix**:

```
turn 1:  [system + task]
turn 2:  [system + task + assistant_1 + tool_result_1]
turn 3:  [system + task + assistant_1 + tool_result_1 + assistant_2 + tool_result_2]
```

A growing prefix is exactly what APC reuses: on turn 3 it recognizes turns 1–2 as cached and prefills only the new tail — the **same computational saving** Pie's session gives.

So "multi-turn / agentic" is *not*, by itself, a Pie advantage, and **the 95 % prefill saving is not what makes Pie faster here.** The revised result does not contradict this; it relocates the win. Pie's margin comes from decode throughput and from no longer paying a self-inflicted per-call tax — not from out-reusing APC.

---

## 2. The integration variants we built

### 2.1 OpenHands (agentic SWE-bench)

OpenHands drives an agent that reads a GitHub issue, explores the repo, edits source, runs the tests, and submits a patch — a long multi-turn loop with heavy tool use. We wired it to Pie **four ways**, separated by two design axes: *where the agent loop runs* (a Python client vs inside the WASM inferlet), and *how the KV context is carried between turns*.

| Variant | Inferlet | Loop runs in | KV across turns | Per-turn cost | Job |
|---|---|---|---|---|---|
| **Baseline** | — (litellm→vLLM) | Python client | vLLM APC (implicit) | re-send full history; APC reuses prefix | 19212031 |
| **"pie" completion** | `openhands-completion` | Python client | generic prefix-hash only | one stateless completion/call | — |
| **coder-session** | `openhands-coder-session` | Python client | content-addressed APC snapshots | render → hash → open longest match → append suffix | **19251912** |
| **Pattern A** — agent | `openhands-agent` | **the inferlet** | **live in-memory ctx** (never torn down) | append the new turn only | 19080944 |
| **Fork / delegation** | `openhands-coder-session` (fork) | Python client | fork share of parent's KV | child forks parent; ~0 re-prefill (99.5–99.7 % saved) | 19039792 |

**coder-session is the load-bearing variant**, and the one all current results use. The reason is not performance but **experimental validity**: it runs the *real* upstream OpenHands agent, so the baseline and the Pie arm differ only in the engine. Pattern A reimplements the agent loop, which confounds any accuracy comparison — a win or loss cannot be attributed to Pie.

> **Superseded:** the earlier version framed this as "Pattern B vs Pattern A is the load-bearing distinction", on the theory that coder-session's per-call teardown caused a 3.65× deficit. That teardown cost was the WebSocket close timeout, not session management. See §4.4.

#### Anatomy of coder-session (`openhands-coder-session`)

The inferlet is **a completion endpoint, not an agent**. Upstream OpenHands runs the loop on the host and issues one `launch_process` per LLM call, resending the *entire* message history each time. Everything the inferlet adds exists to avoid re-prefilling that history.

| On the host (real OpenHands) | Inside the inferlet (WASM) |
|---|---|
| The agent loop, prompts, condenser | Render history → tokens |
| All tools (terminal, file_editor, task_tracker) | Content-addressed KV prefix cache |
| Message history, resent in full per call | Decode + tool-call parsing |

**Per call, in order:**

1. **Render.** `render_prompt` turns the full message list into a token stream byte-identical to the stateless inferlet's, minus the trailing generation cue — omitted so the snapshot ends on a message boundary. It also returns `boundaries`: the token length at each *safe render-unit split* (after the system+tools block, after each user/assistant/system message, after each merged tool batch).
2. **Name.** `content_hash(model_id ‖ TEMPLATE_MARKER ‖ prefix_tokens)` — two FNV-1a lanes rendered as 32 hex chars — under `apc/{session_id}/{compat}/{hash}`. A name match means *identical tokens for the same model*, so a false hit is impossible; any drift in model, chat template, history or tool schemas changes the tokens, changes the hash, and misses cleanly. `compat` is a marker the host bumps to invalidate everything on schema drift.
3. **Look up.** Scan the boundaries **longest-first** (capped at `MAX_OPEN_ATTEMPTS = 8`), calling `Context::open` on `hash(full[..L])`; the most recent saved boundary wins. Each hit is gated on the opened snapshot's `seq_len() == L`, so a name collision or a truncated snapshot is *rejected* rather than trusted. Candidates are slices of the caller's own render, so a candidate is a literal token-prefix by construction — a bad split cannot plant a wrong suffix.
4. **Extend or rebuild.** On a hit, append only `full_tokens[L..]` — the previous assistant turn plus the new tool results, **O(delta) not O(conversation)** → mode `extended`. On a miss: fresh `Context`, full append → `rebuilt` (or `fresh` if the render had no interior boundary at all). A miss is always *semantically* safe, just slower.
5. **Flush and verify.** Materialize the prompt KV. Under `kv_verify`, assert `ctx.seq_len() == full_tokens.len()` and error rather than silently rebuilding.
6. **Save.** Store this call's full render under `hash(full_tokens)`. A duplicate name means identical KV is already saved — benign, and the error is ignored rather than delete-and-resave, which is precisely what lets **distinct boundaries coexist**.
7. **Cue and decode.** `ctx.cue()`, then decode against the stop-token set, with a streaming tool decoder parsing calls as tokens arrive and deduping identical `(name, arguments)` pairs within a turn (looping models emit the same call repeatedly, and executing the copies just burns agent iterations).
8. **Teardown.** On `session_action: "delete"`, remove both the legacy single-slot name *and* the whole `apc/{sid}/` namespace by prefix. Skipping this leaks a conversation's KV per conversation — see §0.

**Why the naming rule is the load-bearing detail.** Both save and lookup name a **full host render**, never a *predicted* assistant reply. An earlier version predicted the next turn's boundary from the inferlet's own text and tool_calls; the host re-serializes JSON arguments with different bytes, so `hash(predicted) ≠ hash(host resend)` and the hit rate collapsed to ~3 %. Naming only full renders — which are append-only, so call *N*'s render reappears as an interior boundary of call *N+1* — restored **96.6 %**.

**What falls out for free.** Because names are content-addressed and many boundaries coexist, a retry, a branch, or a truncation re-hits its still-valid earlier boundary automatically, and a delegated sub-agent that shares a task prefix hits the *parent's* boundary — with no explicit fork protocol and no host-side hints. The host sends nothing but `session_id`.

**What it costs.** Measured end-to-end (§4.3): the entire apparatus is **under 9 ms per call** — render 6.8 ms, hash 0.3, open 0.6 across 1.97 attempts, save 0.6, fork 0.0 — against 92.9 % of the call spent in decode.

A two-phase forced-tool-call path also exists for grammar mode, but `--python-tool-parser` sets `use_grammar: false`, so it is inert on the current path (it fired 0 of 753 calls across the twelve runs since).

#### Anatomy of Pattern A (`openhands-agent`)

One `Context` is created, seeded with system prompt + task, then driven through a fixed step loop. Every step's action is **grammar-constrained to a flat JSON schema** (`constrain_with(JsonSchema)`) — `{thought, action, command, path, old_str, new_str, …}`:

1. **Condense if needed** — *rebuild* (summarize + re-prefill) or *mask* (drop stale-middle KV out of attention, 0 re-prefill).
2. **Generate the action** — decode a grammar-constrained JSON step into the *same* live `ctx`.
3. **Yield the GPU** — `ctx.idle()` releases KV pages while the tool runs.
4. **Run the tool on the host** — POST a `ToolRequest` to `tool_server.py`.
5. **Append & loop** — append the observation as the next user turn, repeat until `finish`.

**What runs where** (the WASM/host split CodeACT avoids):

| Inside the inferlet (WASM) | On the host (`tool_server.py`) |
|---|---|
| The whole agent loop & control flow | `bash` via `PersistentBash` (preserves cwd/env) |
| Grammar-constrained generation + live KV ctx | File edits via OpenHands `FileEditor` |
| Condensation, robustness guards, metrics | Running the repo's tests |

**Why the split exists:** OpenHands' real tools need a host shell + Python env + the repo checkout, so they **cannot run inside a WASM inferlet**. Pattern A keeps the paper's *live-KV* benefit but not its *co-location* benefit.

Pattern A's flat schema also puts `old_str`/`new_str` in top-level `required` on every step, which is why it never hit the `file_editor` oneOf grammar bug that stalled coder-session — but it gave 0-byte patches on Qwen3-Coder because of the XML tool-format gap.

### 2.2 OpenEvolve (MAP-Elites genetic code evolution)

| Component | File | What it is |
|---|---|---|
| **Generation inferlet** | `openevolve-generation` | "analyze-then-diverge": generate a shared analysis A once, fork N children off `P+A` in-memory, per-leaf steer → N diffs. 3-level key tree (L0 / L1p / L1g / leaf) for cross-worker reuse. |
| **PieLLM backend** | `openevolve/llm/pie.py` | `LLMInterface` over `PieClient`; `generate_children()` batched path. Drop-in via `init_client=PieLLM`. |
| **Controller wiring** | `ensemble.py` / `process_parallel.py` | Same-parent workers reuse the shared **L1g** analysis snapshot **cross-process** (`Context::open` by name). |

### 2.3 Supporting / diagnostic

- `parallel-generation`, `tree-of-thought` — fork fan-out templates.
- `decode-bench` — decode-throughput diagnostic. **Note: it has no caching** (fresh `Context` per call); this matters for reading its numbers (§4.3).
- **mask-condense** — drop-middle KV condensation via `mask_range`; measured quality-neutral (job 19070781).

---

## 3. Master table: every experiment & hypothesis

Read the verdicts as: ❌ Pie loses/no-win · ✅ a Pie mechanism provably works · 🔬 diagnostic that isolated a cause · ⭐ Pie wins · ⚠️ **superseded — do not cite**.

| # | Experiment & setup | Hypothesis | Result & evidence | Verdict | Jobs |
|---|---|---|---|---|---|
| 1 | **Fan-out fork** — one Pie call forks a ~10k-token shared prefix into N children vs vLLM two-phase+APC | A single server-side N-fork beats N separate requests | Tie at N=1 (5.5 s); Pie **10× slower at N=64** (71.5 s vs 7.1 s). Fork saved prefill perfectly (`leaves_prefill=0`, 668k tokens) but the N child decodes run **serially** while vLLM continuous-batches | ❌ **loss** — a real Pie engine gap (no batching across forked decodes), unaffected by the transport bug: this harness holds one connection, so it never paid the close timeout | 19074533 / 19074534 |
| 2 | ~~**OpenHands head-to-head** — 13 instances, coder-session vs litellm+vLLM~~ | ~~Retaining KV across turns beats re-sending each turn~~ | ~~**Pie 3.65× slower** (7,227 s vs 1,978 s)~~ | ⚠️ **SUPERSEDED by #13** — the deficit was the unanswered Close frame: **468 calls** (counted from the run's own log) × 10 s ≈ **4,680 s**, i.e. ~89 % of the 5,249 s gap | 18951267 / 18962143 |
| 3 | **Memory-pressure overcommit** — push past the shared 244,704-token KV cache (conc 8→32) | Pie's fork + CPU-offload beats APC evict-then-recompute under pressure | At conc that fits, vLLM slightly faster; at **1.9× overcommit Pie OOM-crashes** while vLLM degrades gracefully (59 s) | ❌ **loss** — Pie's offload never fires; the vLLM driver has no CPU-swap backstop | 19068668 / 19068669 |
| 4 | **Admission factor = 1.0** — refuse/queue beyond capacity instead of oversubscribing 4× | Throttling admission lets Pie survive overcommit | conc ≥ 16 **still CUDA-OOMs** | ❌ — the admission gate is a *soft* market-weight check that relies on swap as the physical backstop | 19070291 |
| 5 | **Native-CUDA driver** — build `--features driver-cuda` for a real offload path | A native driver + swap pool lets effective KV exceed GPU RAM | **Built and now the production backend** (§4.5). Boots 30B in ~11 s, decodes correctly, snapshots work. **But swap is still not wired**: the embedded path has no aux-IPC control-fd, so `swap_pool_size=0` | 🔬 **partly resolved** — the driver landed and everything since runs on it; the *offload* half remains blocked | 19188593, 19203530 |
| 6 | **Decode-tps vs context length** — slope `(t₅₁₂−t₆₄)/448` cancels prefill to isolate pure decode | Pie's decode slows over a long retained context | **REFUTED** — Pie flat ~63 tok/s across 2k→50k; vLLM flat ~65 | 🔬 **valid, and immune to the transport bug** — a constant 10 s per call cancels in a slope. This is *why* decode looked clean while everything else was contaminated | 19078674 / 19078675 |
| 7 | **Mask-condense quality** — masked drop-middle KV vs full rebuild | Dropping middle KV from attention is quality-neutral | **YES** — ratio 1.002 ± 0.021 at P=48k, 0 re-prefill | ✅ **enabler** | 19070781 |
| 7b | **Mask-condense inside Pattern A** — 13 instances, `mask` vs `rebuild` | The mid-context drop APC cannot express speeds up the agent | Mask engaged (7 events, 0 summaries, 0 re-prefill), 13/13 patches, but **wash** (2,374 s vs 2,329 s) | ❌ **no wallclock win** — ceiling is ~3 % of runtime and trajectory variance dwarfs it | 19089013 vs 19080944 |
| 8 | **Cross-worker L1g reuse** — OpenEvolve 30B analyze-then-diverge | Sibling workers reuse the generated-analysis snapshot across processes | **YES, organic** — reused 3 / generated 9; `l1p=opened l1g=reused` | ✅ **mechanism works** | 19065839 |
| 9 | **Fork delegation** — a sub-agent turn forks the parent's KV | A delegated turn can fork the parent's live KV | **YES** — prefill 30k+ → 128 tokens (**99.5–99.7 % saved**), `kv_verify` passed, parent uncorrupted | ✅ **mechanism works** | 19039792 |
| 10 | **circle_packing (~100 iters)** — would a harder OpenEvolve task make Pie win? (analysis only) | More iterations amortize shared-prefix reuse | **No** — the iters are independent one-shot generations, so more iters repeat a per-iteration tie | ❌ **predicted loss** | — |
| 11 | **Pattern A: loop-in-inferlet** — 13 instances via `openhands-agent` | A live ctx removes the per-call session overhead | 2,329 s vs coder-session's 7,227 s and baseline's 1,978 s | ⚠️ **misattributed** — Pattern A connects **once per instance**, so it paid the 10 s close ~13 times (~130 s) instead of ~960 times. It was never "better architecture"; it was less exposure to the bug | 19080944 |
| 12 | **Trajectory divergence** — why do Pie runs take different trajectories at t=0? | A tool-parser mismatch causes it | **No** — two identical t=0 cold calls differed 8/8: the **engine is non-deterministic**, independent of parsing | 🔬 **root-caused** — sets the noise floor; why a failure set can move without a regression | 19039792 |
| 13 | ⭐ **OpenHands head-to-head, corrected** — same 13 instances, cuda_native + self-keyed APC, after the leak and Close-frame fixes | With the dead time removed, does Pie beat the baseline? | **YES — 1,430.4 s vs 1,938.9 s (~26 % faster, 1.36×)**. All 13 complete; every instance `modes={rebuilt:1, extended:N}` (95 %+ reuse); 0 kv-verify errors. 97,343 tokens at 90.1 tok/s decode | ⭐ **win** — supersedes #2 and #11 | **19251912** / 19251913 |
| 14 | 🔬 **Where a call's time actually goes** — `Timings` inside the inferlet + `host_ms` around the round trip | Is the per-call overhead in rendering, hashing, forking, or transport? | Accounting closes to 0.1 ms: **decode 92.9 %, prefill 6.9 %, entire APC apparatus < 9 ms** (render 6.8, hash 0.3, open 0.6 @1.97 attempts, save 0.6, fork 0.0). Transport was **10009.4 ms** — a timeout, not work | 🔬 **found the bug** — killed the re-render and per-call-fork suspicions and located the real cost | 19245724, 19249847, 19251176 |
| 15 | **Neutral 50-instance accuracy** — the full subset the 13 were drawn from, sampled before any Pie result existed | Is Pie's accuracy equal to the baseline's on a set that *can* show a difference? | Baseline already scored: **13/50**. Pie arm in flight | ⏳ **pending** | **19276292** / 19276293 |

---

## 4. Key experiments in detail

### 4.1 Fan-out fork (OpenEvolve shape) — #1

Shared prefix ~10k tokens, fork into N children, conc=1, decode 64/child.

| N | Pie (fork) | vLLM two-phase |
|---:|---:|---:|
| 1 | 5.5 s | 5.5 s |
| 4 | 8.6 s | 5.6 s |
| 16 | 21.7 s | 5.9 s |
| 64 | **71.5 s** | **7.1 s** |

- **N=1 tie** → `pie-serve → vLLM-driver` has no overhead vs standalone vLLM.
- **Pie scales ~linearly, vLLM stays flat.** Fork saved prefill perfectly (N=64: 667,926 tokens saved) but the N child decodes are **serialized** instead of batched into one forward pass.

**This result survives the revision.** The N=1 tie at 5.5 s is itself the proof: had this harness paid the 10 s close timeout, no data point could have come in under 10 s. It holds one connection across the benchmark. So the serial-fork-decode gap is a genuine Pie engine limitation.

### 4.2 OpenHands SWE-bench head-to-head — ~~#2~~ → #13

> **The 2026-07-21 numbers in this section were wrong.** They are kept struck-through because the *shape* of the error is instructive: every mechanism-level measurement said Pie should be fine, and the aggregate said it was 3.65× slower. That contradiction was the signal, and it was misread three times before being measured directly.

~~| Metric | Pie coder-session | litellm+vLLM | Ratio |~~
~~| Total time | 7,227 s | 1,978 s | **Pie 3.65× slower** |~~

**Corrected result** (job 19251912, cuda_native + self-keyed APC, after `f1aac6eb` and `e4626e5b`):

| Metric | Pie coder-session | litellm+vLLM | Ratio |
|---|---:|---:|---|
| Solve time, 13 instances | **1,430.4 s** (23 m 50 s) | 1,938.9 s (32 m 19 s) | **Pie ~26 % faster (1.36×)** |
| django-13028 | 140.6 s / 68 iters | 246.7 s / 88 iters | Pie 1.75× faster |
| Decode throughput | 90.1 tok/s | ~64 tok/s | Pie ~1.4× |
| KV reuse | 95 %+ (`rebuilt:1, extended:N`, all 13) | APC (implicit) | — |
| kv-verify errors | 0 | n/a | — |

Totals are the sum of per-instance solve times from each run's own log, which excludes one-time engine startup (where Pie's ~11 s model load is itself an advantage). For scale, `django-13028` took **524.7 s on Pie before the Close-frame fix** and 140.6 s after — against the baseline's 246.7 s, the same instance went from 2.1× slower to 1.75× faster.

Wallclock is measured independently per arm, so it carries none of the selection bias that affects the accuracy comparison below.

**Accuracy: Pie 11/13, baseline 13/13** — both resolved 11; only-baseline = `pydata__xarray-4966` and `scikit-learn__scikit-learn-10908`; only-Pie = none. Both losses are model-side, not machinery:

- **xarray-4966** — a *wrong fix*. 1089-byte patch applied cleanly, PASS_TO_PASS 21/21, FAIL_TO_PASS 0/4.
- **sklearn-10908** — a **0-byte patch** after 63 calls / 128 iterations. Root cause: at call 19 the model corrupted its own workspace path, dropping `learn-` (`sweb-scikit-learn__scikit-10908` for the real `sweb-scikit-learn__scikit-learn-10908`), then thrashed. Every `str_replace` (calls 41, 49, 58) targeted the nonexistent path and failed; it noticed in prose twice, recovered for one `terminal` call, reverted on the next `file_editor` call, and finally emitted a prose summary via `finish`. All 63 tool calls were well-formed and parsed; `modes={rebuilt:1, extended:62}`; no grammar or KV involvement.

**The 13/13 is by construction.** These 13 instances *are* `baseline_t0_full_50.report.json`'s `resolved_ids`. The baseline cannot score below 13/13 on its own wins, so parity is Pie's ceiling and this set can never demonstrate a Pie accuracy advantage. Honest phrasing: *"reproduces 11 of 13"*, not *"2 worse"*. The failure set also **moved** relative to an earlier 11/13 run (which missed `django-13028` and `sklearn-12973`, both now passing) — trajectory divergence (#12), not regression.

### 4.3 Decode throughput vs context length — #6

| Context L | Pie decode | vLLM decode |
|---:|---:|---:|
| 2 k | 63.1 tok/s | 66.0 tok/s |
| 5 k | 62.8 tok/s | 66.3 tok/s |
| 20 k | 63.3 tok/s | 65.0 tok/s |
| 50 k | 63.1 tok/s | 64.8 tok/s |

**Both flat, within ~4 %.** This measurement is **valid and was never contaminated**, for a reason worth stating: the slope method `(t₅₁₂ − t₆₄)/448` subtracts two timings, so a *constant* per-call cost — including a 10 s close timeout — cancels exactly. Decode looked clean because the metric was constructed to be immune to the thing that was wrong.

**Two claims read off this bench were wrong and are retracted:**

- ~~"Native-cuda decode throughput is the bottleneck"~~ — decode is at **parity**, per the slope above.
- ~~"Pie pays a ~60× per-call context-proportional overhead"~~ — read off the `t₆₄` intercept by misinterpreting `prefill_M64: 0` as a cache hit. It is `ctx.seq_len()` *before* flush, a reporting artifact. `decode-bench` builds a fresh `Context` per call and **never caches**, while vLLM's server APC *was* hitting — so the intercept compared Pie-uncached against vLLM-cached. Pie's 4.94 s at L=50k is an ordinary uncached prefill at ~10.1k tok/s.

Note also that in the corrected head-to-head Pie's *effective* decode throughput is **90.1 tok/s** (97,343 tokens over 1,080.4 s of measured decode across 416 calls) — well above this microbenchmark's flat ~63. The gap is a property of the bench, not the engine: `decode-bench` builds a cold `Context` per call and decodes a fixed short burst, while the real workload decodes into a warm, extended session. Read §4.3 as *"decode does not degrade with context length"*, which is what the slope establishes, and not as an absolute throughput figure.

### 4.4 Pattern A: loop-in-inferlet — #11, reinterpreted

| Arm | Total (13 inst.) | Connections opened | Close-timeout exposure | Total less exposure |
|---|---:|---:|---:|---:|
| litellm + vLLM (baseline) | 1,978 s | n/a | none | 1,978 s |
| Pattern A (live ctx) | 2,329 s | 13 (one per instance) | ~130 s | ~2,199 s |
| coder-session (per-call) | 7,227 s | **468** (one per call) | ~4,680 s | ~2,547 s |

Call counts are from each run's own log (`pie session — N calls`, summed), not estimated.

The original reading was that Pattern A's live context removed a real per-call session-management cost, proving architecture was the issue. **That conclusion was largely an artifact.** The variable that tracked the timings was *how many WebSocket connections each variant opened*, because each one cost a flat 10 s. Pattern A looked good because it connects once per instance, not once per call.

Subtract the exposure and the 3.1× gap between the two Pie variants collapses to roughly 1.16× (~2,547 s vs ~2,199 s) — and even that residual is not architecture, because the two runs did different amounts of work (961 vs 467 iterations, per §4.4b). §4.3's instrumentation settles the question directly: the entire session apparatus — render, hash, snapshot open, save, fork — totals **under 9 ms per call**. There was no meaningful per-call session overhead to remove.

Treat the subtracted column as an estimate: it assumes exactly one full 10 s timeout per connection, which the later direct measurement (10009.4 ms mean, 0.5 ms spread) supports but which was not instrumented during these two runs.

The connection-per-call design remains in coder-session, but now costs ~4 ms. Reusing one `PieClient` per conversation is a tidy-up, not a fix.

### 4.4b Mask-condense inside Pattern A

| Arm | Total | Iters | s/iter | Condensation | 13/13 |
|---|---:|---:|---:|---|---|
| Rebuild | 2,329 s | 467 | 4.99 | 11 × (summary call + ~12k re-prefill) | yes |
| Mask | 2,374 s | 441 | 5.38 | 7 × mask-range (0 summary, 0 re-prefill) | yes |

Mask engaged correctly and kept 13/13 non-empty patches, but is a **wash** (~2 % slower). The ceiling is small — ~11 × (~6 s summary + ~1.5 s re-prefill) ≈ 80 s on a ~2,330 s run, ~3 % at best — and engine non-determinism (#12) swings trajectories far more than that. Masking does not speed up decode (§4.3), so shrinking the attended context buys nothing there.

This conclusion is only mildly affected by the transport fix: both arms are Pattern A, so both carried the same ~130 s, and the comparison between them stays valid. It would matter under **memory pressure**, a regime not present here.

### 4.5 Memory-pressure / overcommit — #3–5

Both engines share the same 244,704-token KV cache. At concurrency that barely fits, vLLM is slightly faster; at 1.9× overcommit, **Pie OOM-crashes** while **vLLM degrades gracefully** (59 s, via APC evict+recompute). Pie oversubscribes ~4× betting on swap, but the vLLM driver has no swap and bootstrap pins `swap_pool_size` to 0, so the bet has no backstop; admission control at 1.0 does not help, being a soft market weight that also relies on swap.

**Update:** the earlier version said this build was *abandoned*. It was not — the **native CUDA driver was built and is now the production backend** for every result in this document (job 19188593; workspace-overflow fix from 64 MiB → 512 MiB; verified coherent decode in job 19203530; snapshots verified working). What remains blocked is specifically **offload**: the embedded driver has no aux-IPC control-fd wiring, so `swap_pool_size` must stay 0 and the swap pool cannot be driven. The overcommit conclusion therefore stands, but for a narrower reason than "we didn't build it".

The native driver's hard KV allocation is also what turned the snapshot leak from a slowdown into a hang (§0): with no eviction path, exhausting the 4096-page budget blocks rather than degrades.

### 4.6 OpenEvolve mechanism validation — #8

The `openevolve-generation` inferlet + PieLLM backend + controller wiring ran end-to-end on 30B (job 19065839): 12 valid children, 0 diff-format misses, best score → 1.4994, and **cross-worker L1g reuse fired organically** (reused 3 / generated 9). The mechanisms are correct and the token savings are real.

Unlike the OpenHands result, this one was **not** rehabilitated by the transport fix: the OpenEvolve A/B found vLLM's APC hitting 96–98 % with no eviction and holding ~8 s flat to concurrency 8, while Pie rose to 13–20 s. That workload is a set of *independent* generations rather than one growing conversation, so it opens many short-lived contexts instead of extending one — the shape where APC is hardest to beat.

### 4.7 Neutral accuracy set — #15 (in flight)

Every accuracy comparison so far ran on the 13 instances that are the baseline's own wins, where parity is the ceiling. The neutral set is the **50-instance subset those 13 were drawn from** — a fixed-seed deterministic sample of SWE-bench Verified, chosen before any Pie result existed.

The baseline half is already scored: **13/50** (job 18397164, 9 h 03 m), same model, temperature 0, 100 iterations, same `qwen3_coder` parser. Only the Pie arm was missing; it is job **19276292**, with scoring **19276293** dependent on it. Instance ids are read out of the baseline's report rather than regenerated, so both arms are provably the same 50.

This is the first comparison on which Pie *can* come out ahead on accuracy.

---

## 5. Where the paper's agentic win comes from (code-grounded)

The paper's ReACT/CodeACT/Swarm inferlets share one structure:

```rust
let mut ctx = Context::new(&model)?;                  // ONE context, created ONCE
ctx.system(SYSTEM_PROMPT); ctx.user("Question: …"); ctx.cue();
for step in 1..=max_steps {
    let raw = ctx.generate(…).await?;                 // decode action — appends to SAME ctx
    let observation = match tool {
        "Calculator" => calculator(arg),              // ← TOOL RUNS IN-PROCESS
        …
    };
    ctx.user("Observation: {observation}…"); ctx.cue();
}
```

Versus a Python-client baseline, Pie eliminates three per-interaction costs:

| Cost | Baseline (client + vLLM) | Pie inferlet |
|---|---|---|
| Client↔server round-trip | one per step (tens of ms) | **none** — tool in-process |
| Re-transmit + re-tokenize history | every step | **none** — context server-side |
| Re-prefill history | if APC evicted during the gap | **none** — KV pinned in `ctx` |

Most of these conditions are absent from our setup:

| Win condition | Paper | Our runs |
|---|---|---|
| Model | 1B/3B (gen ≈ few ms → RTT is a big fraction) | 30B (gen dominates) |
| Client | remote, campus network (real RTT) | **localhost** (RTT ≈ 0) |
| # I/Os | 8 / 8 / 32 (win scales) | fewer, diluted by long gens |
| Concurrency | up to 128 agents → APC evicts | fits → APC never evicts |
| Tool location | **in-WASM** (Rust fn, Boa JS) | bash/edits → **must be on host** |

**And yet Pie now wins anyway (§4.2), on none of these axes.** The margin comes from decode throughput on the native CUDA driver. That is a different mechanism from the paper's, and worth being precise about rather than claiming the paper's result reproduced.

---

## 6. Where a real Pie win could still live

1. ⭐ **Agentic wallclock on the native CUDA driver** — *achieved* (§4.2), driven by decode throughput rather than prefix reuse.
2. **A genuine network gap** between workers and the server → vLLM pays re-transmit + round-trip that Pie's server-side session avoids. The paper's dominant small-model mechanism.
3. **Effective-KV > GPU with working offload** → the only regime where fork+restore structurally beats APC-evict-recompute; still blocked on aux-IPC wiring in the native driver (§4.5).
4. **Fix decode batching across forks** → turns the fan-out *loss* (§4.1) into a tie.
5. **Mask-condense under memory pressure** → its ~3 % ceiling on an unpressured single GPU (§4.4b) should grow where rebuild's re-prefill compounds.
6. **Accuracy on a neutral set** → in flight (§4.7).

---

## 7. Conclusions

- ⭐ **Pie is ~26 % faster than litellm+vLLM on the agentic SWE-bench workload** (1,430.4 s vs 1,938.9 s over 13 instances, a 1.36× speedup), with 95 %+ KV reuse and zero kv-verify errors. This supersedes the previous version's "3.65× slower".
- **The reversal was one unanswered WebSocket Close frame.** A flat 10 s per call, on a connection-per-call client. It is now ~4 ms.
- **The win is not the prefix reuse.** §1 still holds — APC captures the same reuse when the working set fits, and the entire APC apparatus costs under 9 ms per call. The margin comes from **decode throughput** (90.1 tok/s pure decode, 80.5 end-to-end, vs ~64).
- **Pie's KV-reuse mechanisms are real and large** (95 %+ prefill saved in OpenHands, 668k tokens in fan-out, 99.5 % on fork delegation, organic cross-worker reuse in OpenEvolve) — but they are a **compute/energy/cost** argument, not the source of the wallclock win.
- **Genuine Pie limitations remain**: forked decodes are serialized (§4.1), and overcommit OOMs because offload is not wired into the embedded driver (§4.5).
- **No accuracy claim is established yet.** 11/13 on a set whose ceiling is parity says only "reproduces the baseline's wins". §4.7 is the run that can say more.

### Method lessons (the expensive ones)

1. **Measure the layer, don't divide the aggregate.** Three wrong conclusions came from dividing wallclock by a count: "3.65× slower per call", "13.5 s/call vs 3.9" (which silently bundled *tool execution* into "engine time"), and "60× per-call overhead". Each survived because it was arithmetically consistent with a true total.
2. **Sub-millisecond variance across dozens of samples is a timeout, not work.** 10009.4 mean / 10010.4 median / 10010.5 max named the bug before any code was read.
3. **A metric built to cancel a constant will hide a constant bug.** The decode slope (§4.3) was the one clean measurement precisely because it subtracted the 10 s away — which is also why it never pointed at the problem.
4. **Validate at the scale you will run at.** The snapshot leak was invisible across every prior validation because all of them were single-instance.

---

## Appendix: artifacts & job IDs

| Thing | Files | Jobs |
|---|---|---|
| ⭐ **Corrected head-to-head** | `bench_cuda_native_ab_30bcoder_13b.sbatch`, `score_ab_cuda_native_13b.sbatch` | **19251912** / 19251913 |
| **Neutral 50 (pending)** | `bench_cuda_native_neutral_50.sbatch`, `score_cuda_native_neutral_50.sbatch` | **19276292** / 19276293 |
| Baseline on the neutral 50 | `bench_t0_baseline_full.sbatch` | 18397164 |
| Timing instrumentation | coder-session `Timings` → `Output.timings`; `llm.py` `host_ms`; `$PIE_DEBUG_LOG` | 19245724, 19247969, 19249847, 19251176 |
| Close-frame fix | `runtime/src/server.rs` | commit `e4626e5b` |
| Snapshot-leak fix | `runtime/src/context/snapshot.rs` (namespace-prefix delete), `prefix_cache::namespace` | commit `f1aac6eb` |
| Native CUDA driver | `build_pie_cuda.sh`, `tests/fixtures/pie_cuda_native_config_30b_moe.toml` | 19188593, 19203530 |
| Superseded head-to-head | `bench_parity_30bcoder_13.sbatch`, `bench_litellm_baseline_30bcoder_13.sbatch` | 18951267 / 18962143 |
| Fan-out | `bench_fanout_{pie,vllm}.sbatch`, `ab_bench.py` | 19074533 / 19074534 |
| Decode-tps | `inferlets/decode-bench`, `decode_tps_bench.py` | 19078674 / 19078675 |
| Pattern A | `bench_agent_30bcoder_13.sbatch`, `inferlets/openhands-agent`, `tool_server.py` | 19080944 |
| Fork delegation | `bench_slice3_delegation.py/.sbatch` | 19039792 |
| Memory pressure | `ab_pressure_{pie,vllm}.sbatch` | 19068668 / 19068669 |
| OpenEvolve | `inferlets/openevolve-generation`, `openevolve/llm/pie.py`, `run_pie_evolution*.sbatch` | 19063941, 19064585, 19065839 |
| Mask-condense | `inferlets/mask-condense-bench`, `bench_mask_condense*.sbatch` | 19070781 |
| Paper | `pie.pdf` (Gim, Ma, Lee, Zhong — SOSP '25) | — |
