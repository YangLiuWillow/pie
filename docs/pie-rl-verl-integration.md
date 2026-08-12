# Pie as the RL Rollout Server for rLLM/verl

**Status:** draft plan v1 (2026-08-05)
**Repos:** `pie` (this repo), `rllm` (`/Users/yangliu/Desktop/Lin_startup/rllm`), verl 0.8.0 (external, pinned by rllm)
**Prior art in-repo:** branch `liu/pie-rl` (flat-rollout / adapter-score / tokenizer-probe inferlets), `pie-codex` worktree (`codex-responses` inferlet: HTTP daemon + content-addressed KV snapshots)
**Training agent (decided 2026-08-05):** **qwen-code** — speaks OpenAI Chat Completions (required by the gateway's cumulative-token rewrite) and is Qwen's own CLI, so its prompts/tool schemas match the Qwen3-Coder-class policy being trained. Bring-up strategy: debug the first training loops on **mini-swe-agent** (simplest harness), then switch to qwen-code for real runs. Codex (Responses-only) and Claude Code (Anthropic-Messages-only; zero gateway support — requests would 404 at the worker) are not trainable via this path; see §6 R2 and §8.
**qwen-code audit (2026-08-05):** full code audit in `qwen-code-rl-audit.md` (clone at `../qwen-code`, v0.21.6). Verdict: usable but NOT with defaults — auto-compaction, tool-result blanking at 500K chars, and synthetic continuation turns on dropped connections all violate the append-only invariant; git-status/date/folder-snapshot injections break prompt determinism. A launch profile (`--bare --safe-mode` + settings) mitigates all but two hazards, which need a small fork patch (zero the continuation retries; stub git status). The audit's §6 launch config and §1 backend hard-requirements table are inputs to Phase 0/1.

---

## 1. Thesis

Agentic RL rollout generation is dominated by prefill over shared prefixes:

- **Across samples:** GRPO/PPO generates `rollout.n` (typically 8–16) samples per task; all share the task prompt.
- **Across turns:** a multi-turn agent episode re-sends its whole history every turn; turn *k+1*'s prompt is byte-extension of turn *k*'s prompt+completion (rllm's gateway *guarantees* this via its cumulative-token renderer).
- **Across tasks:** system prompts and tool schemas are shared batch-wide.

vLLM's automatic prefix caching captures some of this. Pie's programmable KV (radix-trie page dedup, `fork()`, named snapshots with content addressing) should capture more of it, more controllably. **Claim to test: Pie reduces rollout-phase wall-clock per RL training step by ≥20% vs vLLM-with-APC on multi-turn agentic RL.** A clean negative result (APC already captures the win) is still a publishable artifact, same as the OpenHands project's framing.

---

## 2. Seam decision: gateway worker, not verl-native replica

Two possible seams were mapped:

| | Seam 1: verl-native `RolloutReplica` | **Seam 2: rllm gateway worker (chosen)** |
|---|---|---|
| Transport | Ray actor RPC (`server.generate.remote`), token-in/token-out | HTTP: `/v1/completions` + `/v1/chat/completions` + `/health` |
| Requirements | Register in verl's `RolloutReplicaRegistry`; expose verl `Worker` actors; implement replica lifecycle (`wake_up/sleep/abort_all_requests/release_kv_cache/...`); integrate `CheckpointEngineWorker` (NCCL weight receive); **patch rllm's `verl_engine.py:25-26`** which hard-whitelists `vllm`/`sglang` | Serve a well-documented HTTP contract; register URL with the gateway; weight sync handled out-of-band |
| Works for | verl's native token-level agent loops | **CLI-harness training (claude-code/opencode/qwen-code agents in sandboxes) — the workload we care about** |

**Decision: Seam 2.** Reasons:

1. In rllm's CLI-harness training path, *all* rollout generation flows through the gateway to whatever HTTP workers are registered — verl's own replicas are just the default worker set. Replacing the worker set replaces the rollout server. No verl code changes needed.
2. Pie already serves HTTP from daemon inferlets (`codex-responses` proves the pattern); it has no Ray integration and would need `Worker`-actor shims for Seam 1.
3. Seam 1's weight path assumes NCCL collectives into a vLLM-shaped `BaseRollout.update_weights(generator_of_tensors)` — deeply vLLM-flavored.
4. Seam 2 keeps verl's trainer untouched (FSDP actor, PPO loss, checkpointing all stock).

**Deployment shape:** colocated verl config as-is — verl launches its vLLM replicas, immediately sleeps them (rllm does this at init: `verl_backend.py:208`), and the actor trains on those GPUs. Pie runs on **separate dedicated GPUs** (or a separate host) as the gateway's worker fleet. This sidesteps sleep/wake entirely: Pie never shares a GPU with the trainer, so it never needs to release memory mid-step. (A later optimization can reclaim the vestigial vLLM replicas' resources via a `rollout.nnodes=0`-style config, but it is not on the critical path.)

```
┌────────────── training GPUs ──────────────┐   ┌───── rollout GPUs ─────┐
│ verl: FSDP actor + PPO loss               │   │ pie serve (CUDA driver)│
│ (vLLM replicas slept, unused)             │   │  └ rl-completions      │
└───────────────┬───────────────────────────┘   │    inferlet (HTTP)     │
                │ checkpoint export               └───────▲────────────────┘
                │ + update trigger                        │ /v1/completions
                ▼                                         │ (token-in/token-out)
        weight-sync bridge ──────────────────────► pie admin update
                                                          │
   sandboxes (docker):                        ┌───────────┴───────────┐
   claude-code / opencode / qwen-code  ─────► │ rllm-model-gateway    │
   (ANTHROPIC/OPENAI_BASE_URL)                │ sessions, traces,     │
                                              │ cumulative token mode │
                                              └───────────────────────┘
```

---

## 3. The wire contract Pie must serve (verified against rllm source)

The gateway (`rllm-model-gateway`) mutates and routes every request. Two egress shapes:

### 3.1 `POST /v1/completions` — cumulative token mode (turn ≥ 1; the load-bearing one)

Request body the worker receives (`proxy.py:286-288, 316`):

```jsonc
{
  "prompt": [151644, 8948, ...],   // list[int] — PRE-TOKENIZED full cumulative prompt
  "add_special_tokens": false,
  "model": "<pinned by gateway>",
  "logprobs": true,                 // NOTE: bool, not int — be lenient, treat as 1
  "return_token_ids": true,         // vLLM extension
  "max_tokens": ..., "temperature": ..., "top_p": ...,   // session-injected, authoritative
  "stream": true                    // streaming variant only
}
```

Response fields the gateway parses (`data_process.py`):

```jsonc
{
  "prompt_token_ids": [...],        // root level (or choices[0].prompt_token_ids)
  "weight_version": 3,              // optional, root — stamp what the server is running
  "choices": [{
    "text": "...",
    "token_ids": [...],             // completion token ids — REQUIRED (else EnrichMismatchError)
    "finish_reason": "stop",        // "stop" | "length"
    "logprobs": { "token_logprobs": [ -0.12, ... ] }   // flat completions form
  }],
  "usage": { "prompt_tokens": N, "completion_tokens": M }
}
```

Streaming: first SSE chunk carries `prompt_token_ids`; each chunk carries `choices[0].token_ids` deltas (+ per-chunk logprobs); terminator `data: [DONE]`.

### 3.2 `POST /v1/chat/completions` — turn 0 / passthrough

Same middleware injections, but `messages` in; response is chat-shaped (`choices[0].message`, `logprobs.content[].logprob`) and **must still include** root `prompt_token_ids` and `choices[0].token_ids`. This is where Pie applies the chat template — and it must be renderer-compatible (§6, Risk R1).

### 3.3 `GET /health` → 200

Router health-checks every 10 s; 3 failures ⇒ worker marked dead (`session_router.py:198-241`).

### 3.4 Behavioral requirements

- **Missing `token_ids` ⇒ `EnrichMismatchError` ⇒ whole rollout retried** until the retry budget burns (`agentflow_engine.py:181-183`). Token IDs are not optional.
- **Logprobs ARE optional for phase 1**: rllm's transform pads missing logprobs with 0.0 and only emits `rollout_log_probs` (enabling TIS/bypass) when *every* row has them (`transform.py:206-208, 322-327`). PPO recomputes old-logprobs trainer-side by default. So a logprob-less Pie server trains correctly, just without importance-correction features.
- **Tolerate duplicate requests**: the gateway retries `ConnectError` twice and re-sends streams once on a fresh connection — behavior explicitly built for the weight-update window (`proxy.py:531-539, 747-769`).
- **Multi-step merging invariant**: the trainer merges consecutive steps into one training row only when `prompt_ids[k+1]` literally starts with `prompt_ids[k] + completion_ids[k]` (`transform.py:374-376`). The gateway's renderer guarantees this *if* Pie echoes back exactly the token ids it was given and generated.

---

## 4. Gap analysis (from the Pie-side audit)

| # | Gap | Severity | Where |
|---|---|---|---|
| G1 | **No weight update mechanism at any layer** — no RPC in `pie_bridge` schema, no driver support; base weights load once at `pie serve` boot | blocker for on-policy RL | `driver/bridge/src/schema.rs`, `driver/cuda/src/model/loaded_model.cpp` |
| G2 | No "logprob of sampled token" primitive — `logprob(u32)` needs id up front; alternatives: 600 KB logits/token over WASM boundary, top-K approximation (what `flat-rollout` does, with a fabricated −30.0 floor), or a second teacher-forcing pass | high (needed for TIS; correctness hazard if approximated) | `runtime/wit/core/wit/inference.wit:77-105` |
| G3 | No inferlet accepts pre-tokenized prompts or returns `token_ids + logprobs` over HTTP | medium (new inferlet, all primitives exist) | — |
| G4 | Seeding: only `Multinomial` seedable; SDK mislabels its seed as `draws`; top-p/top-k always use process RNG | medium (reproducibility) | `sdk/rust/inferlet/src/sample.rs:43-44` vs `runtime/src/api/inference.rs:504` |
| G5 | KV pages are not weight-version-aware — after a weight swap, every cached page/snapshot is silently stale | blocker once G1 is solved | `runtime/src/context/pagestore.rs` |
| G6 | Probe row limits (`max_logprob_labels`, `max_prob_rows`) sized for chat, not for a probe-per-decode-step-per-sequence rollout workload; overflow is hard per-request rejection, not backpressure | medium | `runtime/src/driver.rs:47-56`, `scheduler.rs:450-475` |
| G7 | One batch in flight per driver (scheduler is sole synchronous firer) — raw decode throughput will likely trail vLLM's | accepted risk; thesis is prefix reuse, not decode speed | `scheduler.rs:1205-1210` |
| G8 | `liu/pie-rl` is 304 commits behind; its adapter fix superseded; its "no multi-slot" workaround stale; CUDA `load_adapter` is a **silent no-op** | cleanup prerequisite | `driver/cuda/src/service/inproc_service.cpp:128-136` |
| G9 | `Dockerfile.cuda` builds CUDA-only (no portable driver, no Python drivers) | affects deployment image | `Dockerfile.cuda:27` |
| G10 | `scheduler.request_timeout_secs` plumbed but unused; real ceiling is `PIE_SHMEM_HARD_TIMEOUT_S` | small but bites long prefills | `scheduler.rs:1550`, `driver/bridge/src/ipc.rs:180` |

rllm-side gaps (small):

| # | Gap | Where |
|---|---|---|
| R-G1 | `GatewayManager.start(rollout_engine)` auto-registers verl's vLLM addresses; need a config knob to register external (Pie) worker URLs instead | `gateway/manager.py:206-238, 337-340` |
| R-G2 | No hook that exports an HF checkpoint + calls an external server's update endpoint on policy update; `on_batch_end`/`on_policy_updated` currently only drive verl's own `CheckpointEngineManager` | `trainer/verl/verl_backend.py:871-895` |
| R-G3 | Sync (non-async) training loop never advances the gateway's `weight_version` — all traces stamp 0 | `unified_trainer.py:408-410, 818-823` |

---

## 5. Phased plan

**Sequencing principle (revised 2026-08-06):** prove the rllm + qwen-code half against a stock backend *before* Pie enters the loop. The first end-to-end runs should have exactly one unproven half; when Pie is swapped in (Phase 1b) it is a single-variable change and every new failure is attributable to Pie. The stock-backend runs are not throwaway — they are the baseline arm of the Phase 4 A/B and the source of the golden traces the inferlet is built against.

### Phase 0 — Two parallel tracks (~2 weeks)

**Track A — rllm + qwen-code baseline, no Pie (CPU + hosted API).**

1. **Stand up the rllm environment.** venv with `verl==0.8.0` (currently NOT installed), `vllm==0.22.1`, `renderers`; docker sandboxing working locally. Surfaces infra friction before any Pie work stacks on it.
2. **Bake the audit's launch profile into the harness.** Update rllm's `qwen_code.py` (`build_env` + `write_configs`): `QWEN_HOME`/`QWEN_RUNTIME_DIR` isolation, telemetry off, `--bare --safe-mode`, settings.json with compaction/microcompaction disabled (see `qwen-code-rl-audit.md` §6). Pin the qwen-code npm version.
3. **First end-to-end runs:** `rllm eval --agent qwen-code` against a hosted API / LiteLLM upstream. Validates the sandbox install script (nvm bootstrap is a likely first casualty), the launch profile (no compaction firing, no auto-memory side-queries, deterministic prompts — diff two rollouts' `--openai-logging` captures byte-for-byte per §10.3), and basic gateway trace capture. No token IDs on this path — that needs Phase 1a.

**Track B — Pie-side foundations (CPU only, independent of Track A).**

1. **Rebase the RL inferlets.** Cherry-pick `tokenizer-probe`, `adapter-score`, `flat-rollout`, `self-correct-rollout` from `liu/pie-rl` onto current main (new branch, e.g. `liu/rl-rollout-server`). Drop the branch's `runtime/src/adapter.rs` patch (superseded by the `DriverChannel` refactor) and the "one output slot" workaround (portable now supports multi-slot mixing, `driver/portable/src/plan.cpp:408-419`).
2. **Fix G4 (seeding).** Rename the SDK's `Multinomial.draws` → `seed` to match host semantics; add seed fields to `TopP/TopK/MinP/TopKTopP` through WIT → bridge schema → both drivers. Small, but every reproducibility debug session downstream depends on it.
3. **Add the `sampled-logprob` primitive (G2).** New sampler variant: sample, then return `(token_id, log_softmax(logits)[token_id])` in one pass. Per the audit this is ~50 lines per driver (portable + CUDA) plus WIT/SDK plumbing. Collapses `flat-rollout`'s two-phase generate-then-score design and eliminates the top-512 approximation.
4. **Tokenizer/template parity harness.** Extend `tokenizer-probe` into a CI-able check against rllm's renderer: install the PrimeIntellect `renderers` package, and assert that Pie's chat-template rendering (turn 0) is prefix-compatible with `renderer.bridge_to_next_turn` output for a corpus of multi-turn tool-calling conversations (Qwen3 family first). De-risks R1 before any GPU time is spent.

**Exit criteria:** Track A — a clean multi-turn qwen-code eval run through the gateway with the launch profile verified (byte-identical prompt heads across rollouts, zero compaction events). Track B — RL inferlets build and run on current main (CPU); exact per-token logprobs in a single pass; renderer parity suite green for Qwen3.

### Phase 1a — Golden traces through vLLM (first GPU use, ~1 week)

Goal: the full training-relevant path — cumulative token mode, token IDs, enrichment, multi-turn merging — proven end-to-end with qwen-code against a **stock vLLM worker**. Pie is not involved; this validates the §3 contract as *recorded reality* instead of static reading.

1. One GPU on RunPod, `Qwen/Qwen3.6-27B` BF16 on an 80 GB card (A100/H100 — the validated Track A eval stack; see rllm `runpod_eval.py`), `vllm serve`; rllm gateway in training mode (cumulative token mode ON) with the vLLM worker registered. (2026-08-06: upgraded from Qwen3-0.6B/1.7B — Track A already proved the 27B serving path on RunPod. Note the Phase 1b CPU swap re-runs this with the portable driver, where a 27B is impractical: capture a second small-model fixture set (Qwen3-0.6B, A40) with the same runner for the Pie-swap comparison.)
2. Run qwen-code episodes; confirm: episodes fully enriched (every step has `prompt_ids`/`completion_ids`/`logprobs`, zero `EnrichMismatchError`), and `transform_episodes_to_dataproto` produces **merged** multi-turn rows — the append-only invariant holding through the real harness.
3. **Capture golden traces**: gateway sqlite traces + qwen-code `--openai-logging` wire captures + raw vLLM request/response pairs for the `/v1/completions` rewrite. These become the inferlet's test fixtures and answer open question #2 (vLLM's exact `logprobs` envelope) empirically.
4. Keep this setup — it is the nucleus of the Phase 4 baseline arm (harden later with `runpod/10_vllm_serve_fair.sh`).

**Exit criteria:** merged multi-turn training rows from real qwen-code episodes via vLLM; golden-trace fixture set checked into the integration test dir.

### Phase 1b — The `rl-completions` inferlet, built against fixtures (CPU, ~2 weeks; can start once fixtures exist)

Goal: Pie serves the recorded contract; swapping the worker URL from vLLM to Pie reproduces Phase 1a's results.

1. **New inferlet `inferlets/rl-completions`** (HTTP daemon), developed against the Phase 1a fixtures:
   - Shell: port from `pie-codex/inferlets/codex-responses` (`#[wstd::http_server]`, SSE utilities, chunked prefill with `PREFILL_CHUNK`-style bounding).
   - `/v1/completions`: accept `prompt` as `list[int]` (also accept string for debugging), `add_special_tokens:false`, lenient `logprobs` (bool→1), `return_token_ids`. `Context::append(&ids)` directly — no tokenizer in the loop. Respond per §3.1 (validated against fixtures) including root `prompt_token_ids` (echo input) and `weight_version`.
   - `/v1/chat/completions`: template via the `Instruct` trait (reuse `codex-responses`'s replay discipline: position-independent rendering), return chat shape + token ids. Tool-call decode per `qwen3coder_parser.py` reference (audit §8.2 — verify actual emitted format under the pinned tool-call style first).
   - `/health`: 200.
   - Streaming for both (SSE; first chunk `prompt_token_ids`, per-chunk `token_ids` deltas, `[DONE]`).
   - KV reuse from day one: content-addressed snapshot on the completion boundary (the `session.rs` pattern, keyed on a hash of the **token-id prefix** — matches the cumulative-token invariant: turn k+1's prompt starts with turn k's saved prefix).
2. **rllm: external-worker registration (R-G1).** Config path (e.g. `rllm.gateway.external_workers: [url]`) so `GatewayManager` registers given URLs instead of / in addition to engine addresses (eval already has this via `EvalGatewayManager(upstream_url)`).
3. **The swap:** re-run the Phase 1a eval with only the worker URL changed to Pie (CPU, portable driver, same small model). Diff behavior against the vLLM run.

**Exit criteria:** same as Phase 1a's, through Pie — fully enriched episodes, merged multi-turn rows, zero `EnrichMismatchError`; response envelopes match fixtures modulo documented deltas.

### Phase 2 — Training-shaped loop with frozen/restart weights (~2 weeks)

Goal: full `AgentTrainer` loop with Pie generating rollouts; weight updates by the dumbest possible mechanism.

1. Run `AgentTrainer` (unified trainer, verl backend, colocated config) with the gateway pointed at Pie on a dedicated GPU. Small model (Qwen3-0.6B), small task set, `rollout.n = 8`.
2. **Weight update v0 = restart:** hook `on_batch_end`/`on_policy_updated` (R-G2) to (a) export the HF-format checkpoint (verl already writes these), (b) restart `pie serve` against the new snapshot dir, (c) wait for `/health`, (d) `POST /admin/weight_version` to the gateway (fix R-G3 while at it). Use `PIE_CUDA_WEIGHT_CACHE_DIR` to soften reboot cost. Ugly, slow, correct — restart also trivially solves KV staleness (G5).
3. **Correctness gate:** run the identical config with the stock vLLM worker path; compare reward curves and `rollout_log_probs` vs trainer-recomputed old-logprobs (should be near-identical for Pie since decode and scoring share the pass — a *stronger* guarantee than vLLM gives, where rollout logprobs come from a different kernel path than FSDP recompute).

**Exit criteria:** a multi-step RL run (≥50 steps) with Pie rollouts whose reward trajectory matches the vLLM-backend run within seed noise. Logprob agreement: mean |rollout_lp − recomputed_lp| comparable to or better than the vLLM baseline.

### Phase 3 — Native weight sync (~3 weeks, the core Pie engineering)

Goal: replace restart with an in-place update; make KV weight-version-aware.

1. **Bridge + driver: `UpdateWeights` RPC (G1).** New `RequestPayload` variant in `driver/bridge/src/schema.rs`: `{weight_version, source}` where `source` is a safetensors directory path (v0) — the trainer and Pie share a filesystem or the checkpoint is rsynced. Driver-side (CUDA first): pause admission, drain in-flight batches, re-run the storage-program executor's materialization against the existing `WeightStore` **in place** (shapes/dtypes identical across PPO steps — assert this), resume. Target: seconds, not the minutes a cold materialize takes; reuse the artifact-cache staged-H2D path.
2. **KV invalidation (G5).** On update: flush the page store and drop all snapshots (v0 — simple, correct). v1 (only if profiling says cross-update reuse matters): fold `weight_version` into page identity the same way `adapter_seed` already is (`pagestore.rs:986-988`), letting old pages age out naturally.
3. **In-flight semantics:** requests racing an update get aborted; respond with an error the gateway's existing retry machinery absorbs (it already tolerates exactly this window). Stamp `weight_version` in every response so rllm's staleness metrics and `buffer._min_weight_version` work unmodified.
4. **Admin surface:** `pie` control-plane message (msgpack WS, alongside `LaunchDaemon` etc.) or a sidecar admin HTTP endpoint — either is fine; the rllm hook from Phase 2 swaps `restart` for `update`.
5. Optional parallel track (only if full-weights update proves too slow): **LoRA RL** — train adapters only, sync adapter safetensors, requires building the CUDA `AdapterPool` (currently a silent no-op stub) + MLP-projection coverage + per-request adapter selection. Bigger Pie lift, smaller transfer per step. Park unless needed; note `adapter_seed` page-hash correctness is already in place.

**Exit criteria:** weight sync ≤ ~10 s at 7B scale (vs minutes for restart); training run from Phase 2 reproduced with in-place sync; no stale-KV artifacts (verify: post-update generations from a probe prompt match a freshly-booted server exactly).

### Phase 4 — The thesis benchmark (~2–3 weeks)

Goal: measure the claim. Multi-turn agentic RL, Pie vs vLLM-with-APC as the rollout server, everything else identical.

1. **Exploit the prefix structure explicitly:**
   - Cross-turn: the content-addressed snapshot from Phase 1 already gives turn-to-turn KV reuse per session.
   - Cross-sample (`rollout.n`): verify whether the radix trie's content-addressed commit path gives *compute* reuse (prefill skip) for identical prefixes arriving as independent HTTP requests, or only storage dedup. If storage-only, add a shared-prefix snapshot: first request for a given prompt-hash saves it, siblings `Context::open` it (the `fork()`-×-N pattern from `demo-parallel-fork`, spread across requests).
   - Tune G6 limits (`max_logprob_labels` etc.) and `PIE_SHMEM_HARD_TIMEOUT_S` for the rollout shape.
2. **Metrics per training step:** rollout wall-clock, prefill tokens actually computed / total prompt tokens (the reuse ratio), GPU utilization, tokens/s decode. Gateway traces + Pie-side counters.
3. **Baselines:** (a) vLLM 0.22.1 with APC on, same GPU count — the honest comparison; (b) optionally vLLM APC-off, to bound the total prefix-reuse opportunity.
4. **Workload:** the multi-turn coding-agent RL setup from Phase 2 scaled up (Qwen3-Coder class model on H100s, SWE-bench-style or rllm's sandbox task sets, ≥8 turns/episode average).
5. **Optional condition — single-rollout PRPO (`rollout.n=1`):** if the R-sync arm from `realtime-rl-single-rollout.md` survives its Phase-2 check, run it alongside `n=8` GRPO. `n=1` removes the cross-sample sharing leg, so its Pie-vs-vLLM delta isolates the multi-turn-resend leg of the thesis — the conservative bound.

**Exit criteria:** the A/B numbers, whatever they are, with the reuse-ratio telemetry explaining *why*.

### Phase 5 — Writeup (~1 week)

Same artifact discipline as the OpenHands plan: chart (rollout wall-clock per step vs baseline; reuse ratio), method note, honest threats-to-validity (G7: single-batch-in-flight decode throughput; renderer coupling). Feeds the same customer story as the other two integrations, now covering training, not just inference.

---

## 6. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Chat-template drift vs rllm's renderer.** Turn 0 goes through Pie's `Instruct` templating; turns ≥1 through the renderer's `bridge_to_next_turn`. If they disagree, the accumulator resets — training still *works* but multi-turn merging degrades to per-turn rows (weaker credit assignment) and cross-turn KV reuse dies. | Phase 0.4 parity harness; pin one model family (Qwen3) initially; the position-independent rendering discipline from `codex-responses` is the template style to follow. Detect at runtime: accumulator resets are visible in gateway logs. |
| R2 | **Codex and Claude Code harnesses are not trainable via this path.** The gateway's cumulative rewrite only handles `/chat/completions`; Codex speaks Responses-API only, Claude Code speaks Anthropic Messages only (verified: zero `/v1/messages`/anthropic handling anywhere in `rllm-model-gateway` or `trace_converter.py` — requests would 404 at a vLLM/Pie worker before tracing even matters). | **Decided: train with qwen-code** (bring-up on mini-swe-agent). Treat Responses-API / Anthropic-Messages support in the gateway + trace converter as separate future work (§8) — do not couple it to this project. |
| R3 | **Decode throughput gap (G7).** One synchronous batch in flight per driver may lose enough on decode to swamp prefill savings. | Measure early (Phase 2 has the A/B harness). If it dominates: pass-level speculation (`speculation_depth`) is already available; scheduler pipelining is a known, bounded engine improvement; and the writeup can separate prefill-phase vs decode-phase accounting so the thesis result stands on its own. |
| R4 | vLLM's APC already captures most of the win (the negative-result scenario). | Same stance as the OpenHands spec: a rigorous negative result is a valuable artifact. The reuse-ratio telemetry makes it diagnostic either way. |
| R5 | In-place weight materialization is slower or trickier than expected (quant transcode paths, MoE). | Restart-with-artifact-cache (Phase 2) remains the fallback and is already acceptable for small-scale demos; start with a non-quantized dense model to keep the v0 path simple. |
| R6 | GPU access friction (H100 requests are non-trivial). | Phases 0–1 are CPU-only by design; Phase 2 needs one GPU; only Phase 4 needs the real fleet. |

---

## 7. Open questions (resolve during Phase 1–2)

1. Does the radix trie give **prefill-compute** reuse across independent requests, or storage dedup only? (Decides Phase 4.1's design; read `runtime/src/inference/scheduler/chunked.rs` + pagestore lookup path, or just measure with `demo-parallel-fork`-style counters.)
2. Exact response envelope vLLM 0.22.1 emits for `logprobs=true` (bool) on `/v1/completions` — mirror its leniency precisely so the gateway's parser sees identical shapes.
3. Checkpoint transport for Phase 3 when trainer and Pie are on different hosts: shared FS, rsync, or a streamed tensor protocol? (v0: shared FS.)
4. Where to run the gateway's `AutoTokenizer`/renderer when the model is a mid-training checkpoint — confirm it can load tokenizer from the exported checkpoint dir rather than the HF hub name.
5. Does verl's colocated init tolerate `rollout.n_gpus` being minimized to free more GPUs for the actor, given its replicas are never used for generation? (Pure config exploration; not blocking.)

---

## 8. Future work (explicitly out of scope)

- **Anthropic Messages support in the training gateway** — a `/v1/messages` ⇄ chat-completions translation layer plus Messages-aware cumulative rewriting and trace conversion. This is the unlock for training with Claude Code as the harness.
- **Responses-API support in the training gateway** — same shape of work for Codex. Would also let the `codex-responses` inferlet serve training rollouts directly.
- Reclaiming the vestigial colocated vLLM replicas' GPU reservation (config surgery in verl resource pools).
- LoRA-RL track (CUDA `AdapterPool`, MLP-projection LoRA coverage) if full-weight sync proves too slow — see Phase 3.5.
- **OpenHands as a secondary RL harness** — revisit only if the artifact's audience shifts from customers to the research community (OpenHands is the standard harness in RL-for-SWE literature). Costs: rllm has no OpenHands harness (would need building), and OpenHands' condenser raises the same append-only questions we audited for qwen-code. The inference-side artifact already covers OpenHands (§10).
- **"Phase 6": parallel reasoning in agentic RL** — Multiverse/NPR-style native parallel reasoning (Map-Process-Reduce inside each agent turn), trained with PAPO-style RL in the qwen-code harness, served by a Pie inferlet instead of the bespoke SGLang forks both papers required. Full design note: `parallel-reasoning-agent-rl.md`. Gated on Phases 0–3; its inference-only demo (P6.0) needs only Phase 1b's inferlet shell.
- **Real-time RL (single-rollout PRPO + TIS)** — the rLLM team's batch-normalized single-rollout training (`realtime_rl.md` post). Nearly free to try: `prpo` estimator + TIS already exist in our rllm checkout; sync arm is a config change after Phase 2, async arm doubles as the Phase 3 acceptance test. Caveat: the async arm makes real sampled-token logprobs (Phase 0B primitive) and `weight_version` stamping **required**, not optional. Idea note: `realtime-rl-single-rollout.md`.
- **Harness Optimizer × inferlets** — Strands' `strands-harness-optimizer` (GEPA-style contrastive-reflection prompt evolution) paired with Pie: inferlet-side rules injection makes the sealed qwen-code harness optimizable without forking it; snapshot-module prompt layout keeps KV reuse across optimizer iterations (vLLM APC re-prefills); `prompt_version` rides the `weight_version` machinery toward two-level continual learning with the real-time-RL arm. First arm (offline optimizer, no Pie) needs only Phase 0A infra and doubles as the prompt-optimization baseline for the writeup. One integration unknown: whether the gateway's cumulative-token enrichment tolerates server-injected tokens. Idea note: `harness-optimizer-inferlet.md`.

## 9. Working constraints (inherited from the other Pie integration projects)

- CPU-first where possible: one GPU enters at Phase 1a (golden-trace capture) and Phase 2 (training loop); the benchmark fleet only in Phase 4. Phase 0 and Phase 1b are CPU-only.
- Never assume API signatures — read the pinned source (`verl==0.8.0`, `vllm==0.22.1`, installed `renderers`).
- New Pie work on a fresh branch off current main (**not** off `liu/pie-rl` — cherry-pick from it instead; it is 304 commits behind).
- `pie/docs/` is gitignored; this spec is untracked like the others.

---

## 10. Reuse inventory from the OpenHands integration

The OpenHands integration (`integrations/openhands/`, branch `openhands-integration-updated`) is **complete and measured** — its results and tooling feed this project directly. Inventory, mapped to the phases that consume it:

### 10.1 Validated evidence (de-risks the thesis → Phase 4 framing)

`integrations/openhands/docs/pie-vs-litellm-writeup.md`: OpenHands+Pie vs OpenHands+vLLM(APC) on SWE-bench Verified, Qwen3-Coder-30B-A3B, one GPU, serial, temp 0:

- **95.8% of prompt tokens served from KV reuse** (only 4.2% of 12.1 M tokens actually prefilled)
- −26% wall time on 13 shared instances; −23% per agent iteration
- +5/50 accuracy on a neutral instance set
- Caveat recorded in the writeup: the vLLM baseline ran with CUDA graphs disabled and untuned MoE kernels — token-reuse and accuracy numbers are clean; wall-clock is partly a config artifact.

Implication for Phase 4: the reuse ratio is *achievable on this workload class*; the RL question is whether it survives weight-update cache flushes, `rollout.n` parallelism, and the gateway hop. Adopt the writeup's structure: clean metrics separated from config-sensitive ones, explicit fairness section.

### 10.2 Benchmark kit (→ Phase 2 correctness gate, Phase 4 A/B)

- `runpod/10_vllm_serve_fair.sh` + `11_autotune_moe.sh` + `12_validate_moe_config.sh` — the **tuned, fair vLLM baseline** (fixes the exact handicap the first writeup had to disclaim).
- `runpod/30_ab_run.sh`, `summarize_ab.py`, `40–43_*_sweep.*` — A/B methodology, concurrency/context/decode sweeps.
- `benchmarks/compare_equivalence.py` — trajectory-equivalence checking between arms ("same agent behavior, only the serving changed").
- `runpod/pie_cuda_native_config_30b_moe_h100*.toml` / `*_h200_latency.toml` — **tuned CUDA driver configs for Qwen3-Coder-30B-A3B**, the target policy-model class.
- `score_swebench_apptainer.py`, `RUNPOD_SCORING.md` — SWE-bench scoring on the cluster.

### 10.3 The prompt-invariance lesson (→ Phase 0 methodology)

`pie_openhands/tool_desc_invariance.py`: OpenHands appended the per-conversation workspace path to one tool description — **one variable byte-run in the prompt head, and measured cross-conversation snapshot reuse never hit** (2026-07-31: second instance still cold-rebuilt). Fix: strip it at the harness layer, applied identically to both arms. Two rules adopted:

1. Strip per-instance bytes from the prompt head at the harness layer, identically for both benchmark arms (qwen-code equivalents: git status, dates, folder snapshots — `qwen-code-rl-audit.md` H11–H16).
2. **Never assume reuse — measure snapshot hit rates.** A single leaked byte silently zeroes the win while everything still "works."

### 10.4 Code to port into `rl-completions` (→ Phase 1)

- `pie_openhands/qwen3coder_parser.py` — dependency-free port of vLLM's Qwen3-Coder XML tool-call parser. Load-bearing: qwen-code prompts `qwen*-coder` models to emit XML `<function=NAME>` calls; the inferlet must decode that dialect into OpenAI `tool_calls`. Reference implementation for the Rust decode.
- The replay/generation core of `inferlets/openhands-completion` / `openhands-coder-session` — already ported once into `codex-responses` (`replay.rs`); the chat-completions shape needed here is closer to the OpenHands original than Codex's Responses items were.
- `pie_openhands/editor_repair.py` — pattern for a model-specific repair layer; a mid-training policy emits more malformed tool calls than a polished instruct model.
- `tests/` (wire-protocol smoke, session, e2e smoke) — testing shape for the inferlet.

### 10.5 Process patterns

- `docs/OPENHANDS_CODER_SESSION_DESIGN.md` has an explicit **"Fidelity verification (before any benchmark)"** phase — same instinct as our Phase 0 parity harness, validated by experience.
- `docs/CODER_SESSION_EXPLAINER.md` documents the content-addressed prefix-cache design decisions that `codex-responses` refined and `rl-completions` inherits third-hand.

### 10.6 What does NOT transfer

The `PieLLM` Python adapter and WebSocket transport (OpenHands-SDK-specific; our path is the HTTP daemon inferlet), and all condenser handling (OpenHands history compression — for qwen-code we disable compression outright rather than supporting it surgically).

### 10.7 Harness-choice rationale (recorded 2026-08-06)

Why qwen-code and not OpenHands as the RL harness, despite the deeper OpenHands integration: (a) OpenHands is research-segment — standard in RL-for-SWE literature, but consumer mindshare is on CLI agents; the artifact targets customers, and the OpenHands inference artifact already exists, so RL-on-qwen-code broadens coverage instead of concentrating it; (b) rllm ships no OpenHands harness — using it means a net-new harness build plus a fresh append-only/determinism audit of its condenser; (c) among agents the gateway can actually train (chat-completions dialect), qwen-code is the most popular — Claude Code and Codex are excluded by wire protocol, not preference. Revisit trigger: a research-audience artifact (§8).
