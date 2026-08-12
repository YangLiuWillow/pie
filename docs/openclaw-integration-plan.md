# OpenClaw ↔ Pie — detailed execution plan

Task-level companion to `openclaw-integration.md` (the two-strategy spec; read
it first). Progress is logged in `openclaw-integration-progress.md`, one entry
per completed task, same convention as the opencode pair.

**Branch:** `liu/openclaw-integration` (worktree `Lin_startup/pie-openclaw`),
forked from `liu/opencode-integration` @ `1053a7790` — the opencode
infrastructure (P0.1/P0.3/P0.4/PA.2 done, PA.1 milestone 1) is inherited, not
duplicated. Builds use the shared target dir:
`CARGO_TARGET_DIR=$HOME/Desktop/Lin_startup/pie/target` (disk-space note in the
opencode progress log).

**OpenClaw checkout:** `~/Desktop/Lin_startup/openclaw` (`2026.8.1`). OpenClaw
work (the `extensions/pie/` plugin) lives there, not in this repo; this repo
carries fixtures, audit, inferlets, gateway work, and the `integrations/openclaw/`
harness (configs, scripts, acceptance suite).

Naming: pie-side task ids continue the opencode convention — `P0.*` prereqs,
`PA.*` Strategy A, `PB.*` Strategy B — prefixed `oc-` in the progress log's
task board to avoid collision with the opencode board.

---

## Phase 0 — wire audit, fixtures, parity

### P0.1 — source-derived wire audit  *(no OpenClaw execution needed)*

Answer each question by reading OpenClaw source; every finding gets a row in
`tests/inferlets/fixtures/openclaw/AUDIT.md` marked **source-derived** (later
upgraded to **capture-verified** by P0.2). Question → where to look:

1. **Keepalive semantics.** Does the SSE idle watchdog reset on comment bytes
   (`: ping`) or only on parsed events? →
   `packages/ai/src/transports/openai-completions-stream.ts` (line scanner),
   `packages/ai/src/utils/stream-first-event-timeout.ts`, the ~120 s read-idle
   watchdog, guarded-fetch total abort
   (`packages/ai/src/transports/host-policy.ts`). Decides: keep `: ping`
   comments vs switch the ingress per-client to empty-delta chunks. Also pin
   down `timeoutSeconds` propagation (which of the three timers it raises).
2. **Content shape.** Are user/assistant messages serialized as plain strings
   or structured content-part arrays? Multimodal parts on a text-only turn? →
   `packages/ai/src/transports/openai-completions-params.ts:292-460`
   (`buildOpenAICompletionsParams`), `compat.requiresStringContent` handling.
   Decides ingress tolerance vs compat flag in the pie manifest.
3. **Retry / failover discipline.** What does OpenClaw do on 400 vs 429 vs 5xx
   vs network error vs mid-stream abort? Is any retry unbounded (opencode's
   never-500 rule)? → retry settings resolution in
   `src/agents/sessions/sdk.ts:475-500`, failover classifiers
   (`classifyFailoverReason`), and `matchesContextOverflowError` — collect the
   exact strings/shapes OpenClaw matches for context overflow so pie's error
   bodies trigger compaction, never a hard fail.
4. **Tool schemas.** What does the default tool catalog look like on the wire
   (count, size, JSON-Schema keywords)? What do `toolSchemaProfile` /
   `unsupportedToolSchemaKeywords` / `normalizeToolSchemas` strip? Tool-name
   charset/case rules? Deterministic ordering
   (`sortPromptCacheToolsByName`) confirmed on the wire? → tool assembly in
   `src/agents/tools/`, cleaning in the compat layer. Also: lean mode's wire
   difference (`localModelLean` surface vs full).
5. **Cache fields.** With `supportsPromptCacheKey: true` +
   `supportsUsageInStreaming: true`: exact `prompt_cache_key` value (sessionId
   64-char clamp — stable across turns? across restarts?),
   `prompt_cache_retention`, `stream_options`, `store`. Confirm the
   cache-boundary marker is stripped and where it sits in the system prompt. →
   `packages/ai/src/providers/openai-completions.ts:715-760`,
   `openai-prompt-cache.ts`, `system-prompt-cache-boundary.ts`.
6. **Headers & identity.** Full header set on a request (attribution headers,
   session headers, `resolveTransportTurnState` output). Is there a
   session-scoped header usable as the sticky-affinity key, or is
   `prompt_cache_key` (body field) the only session signal? Affects PA.1
   design (header-based vs body-sniff affinity).
7. **Extra model calls.** Enumerate every non-main-turn LLM call and whether it
   shares the provider/model: heartbeat turns (`docs/gateway/heartbeat.md`),
   compaction summarization (`agents.defaults.compaction.model` fallback
   chain), title/utility calls, memory/embedding calls. Each needs a policy in
   Strategy B (must not touch the session working set).
8. **Sampling & limits.** Default `temperature`/`top_p`/`max_tokens` sent for
   an unknown local model id; `maxTokensField` selection; context-window
   clamping math (`contextWindow`/`contextTokens`/`maxTokens` interplay);
   `params.extra_body` passthrough.
9. **Streaming-event expectations.** Which delta shapes the parser accepts
   (`reasoning_content`? `reasoning_details`?), tool-call delta assembly rules
   (first-delta `id`+`name` requirement?), `[DONE]` handling, usage-chunk
   position, empty-choices chunks.

Method note: do the reading in the OpenClaw checkout with targeted agents;
findings land only in AUDIT.md (single source of truth). Where opencode's
AUDIT.md answered the same question, record agreement/divergence explicitly —
the divergence list is what makes the shared ingress robust.

**Deliverable:** `tests/inferlets/fixtures/openclaw/AUDIT.md` (hazard table,
every row sourced with `file:line`). **Acceptance:** all nine questions
answered or explicitly marked capture-blocked.

### P0.2 — real wire captures

1. **Recorder.** Reuse `tests/inferlets/fixtures/opencode/record_server.py`
   (stateful recording proxy). Adapt only if OpenClaw's guarded fetch or
   header expectations require it (loopback origin is trusted, so no TLS
   games). Output convention: `wire/req-00N.json` with request body + response
   summary, same as opencode.
2. **Capture profile.** `integrations/openclaw/openclaw.json`: gateway-scoped
   config with `models.providers.pie` → `baseUrl:
   http://127.0.0.1:<recorder>/v1`, a real upstream behind the recorder
   (any OpenAI-compatible endpoint — LM Studio / the pie ingress itself once
   live / an echo stub for request-only capture), `compat` flags set as PA.2
   will ship them. Keep the workspace fixed and tools pinned so fixture bytes
   are replayable.
3. **Drive turns headlessly.** Preferred: `openclaw agent` / the
   `openclaw infer model run --local` debug path (survey's debug ladder) —
   verify the exact non-interactive command that produces (a) a plain turn,
   (b) a tool-call turn (e.g. ask it to read a file), (c) a follow-up turn
   carrying assistant+tool history replay, (d) whatever utility calls fire
   (title/heartbeat if triggerable). Both full surface and
   `localModelLean: true` variants.
4. **Bank + audit upgrade.** Fixtures → `tests/inferlets/fixtures/openclaw/`
   (`wire/`, `README.md`); upgrade AUDIT.md rows to capture-verified; note
   any behavior the source reading missed.

**Deliverable:** ≥5 fixtures covering plain / tools / history-replay / lean /
utility-call. **Acceptance:** history-replay fixture contains
`assistant`+`tool_calls` and `role:"tool"` turns; AUDIT rows upgraded.
**Risk:** OpenClaw may need onboarding state to run headless — budget time for
a minimal `~/.openclaw` profile; if the gateway insists on a channel, use the
CLI chat channel or the `infer` debug path.

### P0.3 — renderer parity on OpenClaw fixtures

1. Point `integrations/opencode/parity/render-tokens` +
   `check_render.py` at the OpenClaw fixtures (path argument; no code fork —
   if a flag is needed, add it upstream in the opencode harness).
2. Run all fixtures vs HF `apply_chat_template(enable_thinking=False)`,
   Qwen3-0.6B (venv recipe in the opencode progress log; transformers 5.x
   note applies).
3. Record divergences. Expected: none beyond the fixed D1–D4 classes; any new
   class (likely candidates: OpenClaw's larger schema set hitting new
   `tojson` corners, unicode in the system prompt vs the ensure_ascii caveat)
   gets a D-number and a fix in the shared crate/template — render and
   snapshot address change together (response/save unification).

**Deliverable:** parity report in the progress log. **Acceptance:** all
OpenClaw fixtures token-exact, or each divergence has a D-number + owner.

---

## Phase A — bundled provider + live serving

### PA.1 — keyed sticky affinity in the gateway  *(shared with opencode track)*

The PA.2-deferred item, now load-bearing: without stickiness, multi-worker
KV snapshot reuse is defeated. Design: a keyed-affinity variant in
`gateway/src/session.rs` fed by, in priority order, `x-session-id` header
(opencode) → `prompt_cache_key` body field (OpenClaw) → hash of
(identity, system prompt, tools) (fallback). Single-worker standalone mode is
unaffected (everything lands on the one worker). Tests alongside
`gateway/tests/openai_ingress.rs`.

**Acceptance:** two sequential requests with the same key hit the same stub
worker in a two-worker test; existing 6/6 ingress tests stay green.

### PA.2 — `extensions/pie/` in OpenClaw

Per spec §1.2: `index.ts` (`defineSelfHostedOpenAICompatibleProvider`, id
`pie`, baseURL `http://127.0.0.1:8080/v1`, env `PIE_API_KEY`),
`openclaw.plugin.json` (compat flags incl. `supportsPromptCacheKey`,
`supportsStreamingUsage`; pricing external:false; discovery refreshable),
`localService` documented profile (`pie serve`, healthUrl `/health`, long
`readyTimeoutMs`), core touch-ups (overlay-id allowlist, lean-mode
auto-enable, `docs/providers/pie.md`, ui plugin card). Follow the vllm
extension file-for-file; divergences only where the audit says so.
Whether these land upstream or stay a patch branch of OpenClaw is a
publication question — keep the diff minimal and self-contained either way.

**Acceptance:** stock OpenClaw + the extension + a running `pie serve`
completes a plain turn and a tool-call turn end-to-end; `/v1/models`
discovery and onboarding wizard show pie.

### PA.3 — acceptance suite + e2e + A/B

1. Generalize `integrations/opencode/test_acceptance.py` → shared
   `integrations/common/` runner parameterized by fixture dir + hazard
   profile, or a sibling `integrations/openclaw/test_acceptance.py` reusing
   its helpers (decide by diff size after the audit; do not copy-paste 25
   tests). OpenClaw-specific hard rules from AUDIT.md become new tests
   (e.g. watchdog-compatible keepalives, overflow-error shape).
2. `integrations/openclaw/run_pie_openclaw.sh` mirroring the opencode
   launcher (same RAM/`PIE_METAL_ROW_BUDGET_MB` mitigations; the
   `max_model_len` floor must cover OpenClaw's larger prompt — measure in
   P0.3 and set accordingly).
3. E2e: scripted OpenClaw tasks (file read/edit in a sandbox workspace) on
   `pie/qwen3-*`; then A/B vs Ollama and llama.cpp (OpenClaw's incumbents):
   wall clock, prompt tokens, `cached_tokens`, trajectory match — the
   qwen-code §5b protocol.

**Acceptance:** suite green against a live server (model-behavior rules soft
at 0.6B, wire-shape rules hard); A/B numbers logged honestly, incl. the
short-context losses.

---

## Phase B — session inferlet (`openclaw-session`)

Ordering note: B is gated on A only for its *fallback path*; dialect design
(PB.1) can start as soon as P0.1's edit-op inventory exists.

### PB.1 — shared session dialect (design + crate)

One versioned JSON dialect for both `opencode-session` and
`openclaw-session` (spec §2 of both docs): `turn.append/edits/gen`,
`fork/join`, `pin` (cache-boundary), upstream `delta/reasoning/tool_call/
finish/checkpointed`. Deliverable: `inferlets/session-dialect/` (pure
serde crate, native tests, fixture round-trips) + a dialect.md contract doc.
Design review against BOTH harnesses' rewrite inventories (OpenClaw:
compaction, tool-output truncation, message editing, heartbeat, subagent
spawn; opencode: its compaction + truncation) before freezing v0.

### PB.2 — `openclaw-session` inferlet

`inferlets/openclaw-session/`: session loop over `session::receive()`;
state = per-branch `kv-working-set`; render/decode/PTIR via
`pie-openai-serving` (shared with `chat-completions` — any divergence goes
into the shared crate, never forked); `update-index` checkpoint per turn;
`fork` on subagent branch; `discard/slice` on edit ops; degrade path =
rebuild from full history (Strategy A code).

### PB.3 — `createStreamFn` transport in `extensions/pie/`

Native `"pie"` api id in `MODEL_APIS` + provider `createStreamFn`: Node WS
(`ws` pkg, `/v1/ws` path + `x-pie-identity` header — pie's bundled JS client
is not usable as-is), shadow-history diff keyed on `StreamOptions.sessionId`,
delta upload, renegotiate-on-divergence, transparent HTTP fallback. Emits
OpenClaw's stream events (`text_*`, `thinking_*`, `toolcall_*`, `done`).

### PB.4 — measurement

Vs Phase A, per spec §4: bytes/turn, prefill tokens/turn, latency at
32k/128k, subagent spawn with/without fork, compaction with/without B-2,
heartbeat turn cost (B-7). Same honesty rule.

### PB.5 — optional depth

Grammar-forced tool calls behind a driver-capability probe (B-6);
speculative continuation during tool wait (B-4/B-5); embedding serving via
`registerEmbeddingProvider`.

---

## Standing environment notes

- Metal admission needs ~3.2 GiB reclaimable RAM; mitigations
  (`PIE_METAL_ROW_BUDGET_MB`, floor caveat) in the opencode progress log's
  PA.3 entry. OpenClaw's bigger prompt raises the refuse-threshold stakes —
  over-long prompts are refused, not chunked.
- Known-good local config: `~/.pie/config.toml` (metal,
  `Qwen--Qwen3-0.6B-optimized`), `max_model_len` raised to ≥16384; raise
  further if OpenClaw renders past it (measure in P0.3).
- The bare-inferlet-name 500 was fixed on the opencode branch tip we forked
  (`1053a7790`) — verify `chat-completions@0.1.0` resolution on first live run.
- Rebase discipline: this branch tracks `liu/opencode-integration`; rebase
  onto it when the opencode track moves (shared crates live there), and keep
  fixture/audit/integration-dir changes conflict-free by construction
  (separate `openclaw/` dirs).
