# OpenClaw ↔ Pie integration — two strategies

**Date:** 2026-08-11. Target: pie `dev` (`58cb77936`) **plus the
`liu/opencode-integration` branch** (worktree `Lin_startup/pie-opencode`), which
carries the serving infrastructure this plan reuses. OpenClaw `2026.8.1`
(`~/Desktop/Lin_startup/openclaw`). Companions: `opencode-integration.md` (the
two-strategy template and its progress log), `qwen-code-integration-plan.md`
(the validated Option-A precedent), the SOSP'25 paper for the performance model.

**Purpose.** Same phased pair as opencode, adapted to OpenClaw's seams:

- **Strategy A — serve the completion endpoint.** OpenClaw + a bundled `pie`
  provider extension, pointed at the pie-served OpenAI-compatible
  `/v1/chat/completions` already built for opencode. Drop-in serving;
  KV reuse is content-addressed and best-effort.
- **Strategy B — harness-adjacent session inferlet.** Move OpenClaw's
  *context-management layer* server-side as a long-lived `openclaw-session`
  inferlet: persistent working sets, delta-only wire traffic, in-place context
  editing, KV forking for subagents, decode/tool overlap, grammar-forced tool
  calls. OpenClaw's provider layer makes this *easier* than opencode: a fully
  custom transport (`createStreamFn`) is a first-class, non-experimental plugin
  hook — no `file://` npm workaround, no gated V2 stack.

A is the baseline and the fallback path B degrades to. The session-inferlet
wire dialect should be designed once and shared with opencode's Strategy B —
both clients diff full-history prompts against a server shadow and send deltas;
nothing in that contract is harness-specific.

---

## 0. Ground truth

### 0.1 What the opencode branch already provides (reusable as-is)

Status per `opencode-integration-progress.md` (all on `liu/opencode-integration`):

- **P0.1 done** — tool-history replay primitives restored on dev:
  `tools.wit` `equip-after-system` / `assistant-with-tool-calls` /
  `answer-batch`, `Instruct` trait defaults, validated qwen_3 overrides.
- **P0.3 done** — shared `pie-openai-serving` crate (types, chunk framing,
  session canon/FNV addressing, render planning, error discipline,
  `VisibleFilter`, salvage parsers). 43/43 native tests.
- **P0.4 done** — renderer parity harness; all 5 opencode fixtures
  **token-exact** vs HF `apply_chat_template(enable_thinking=False)` on
  Qwen3-0.6B after D1–D4 fixes.
- **PA.2 done** — `gateway/src/ingress/openai.rs`: `POST /v1/chat/completions`,
  `GET /v1/models`, `GET /health`; Bearer→Identity; gateway⇄inferlet envelope
  (`{"status":u16}` first, then verbatim chunk JSON); SSE keepalive comments;
  6/6 integration tests.
- **PA.1 milestone 1** — `inferlets/chat-completions/` guest crate: chunked
  PTIR prefill + device-carried decode, tool-call decoding with atomic first
  deltas, stop discipline, salvage, 400-vs-500 discipline. Pending: KV snapshot
  sessions (`cached_tokens`), grammar-forced calls, Coder XML dialect.
- Known open items: bare `chat-completions` launch name must become
  `chat-completions@0.1.0` (predicted universal-500 bug); sticky affinity
  keyed on a session header is deferred (Ephemeral today); PA.3 live acceptance
  run blocked on machine RAM vs Metal admission margin.

**None of this is opencode-specific except the fixtures and the audit.** For
OpenClaw, Strategy A needs zero new pie-side subsystems — only the OpenClaw
wire audit, OpenClaw fixtures, and whatever hazards that audit surfaces.

### 0.2 OpenClaw's seams (from the provider-layer survey)

- **Zero-code custom provider**: `models.providers.pie` in `openclaw.json` with
  `baseUrl` defaults to `api: "openai-completions"` — Chat Completions SSE via
  the official `openai` SDK (`packages/ai/src/transports/openai-completions-transport.ts:104`).
- **Bundled provider extension**: `extensions/vllm/` is the 27-line template —
  `defineSelfHostedOpenAICompatibleProvider` (`src/plugin-sdk/provider-model-shared.ts:43`)
  wires auth, `/v1/models` discovery, onboarding, model picker. Plus a
  declarative `openclaw.plugin.json` manifest.
- **Fully custom transport, first-class**: `ProviderPlugin.createStreamFn`
  (`src/plugins/provider-plugin.types.ts:300`); the api id is an open string
  union (`packages/llm-core/src/types.ts:6`), auto-registered at first use
  (`src/agents/provider-stream.ts:75`). Ollama (`extensions/ollama/index.ts:851`)
  and llama.cpp (in-process) are the templates. **This is Strategy B's seam** —
  native, ungated, no fork.
- The agent loop re-sends the **full history + system prompt + a large tool
  surface every turn** (`packages/agent-core/src/agent-loop.ts:526-560`);
  stateless chat contract, session state in SQLite.
- Hooks that already anticipate a session-aware backend:
  - `StreamOptions.sessionId` flows to every provider
    (`packages/llm-core/src/types.ts:93`) — the Strategy B process key and the
    Strategy A affinity key.
  - `prompt_cache_key` = sessionId (64-char clamp) is sent iff
    `compat.supportsPromptCacheKey: true`
    (`packages/ai/src/providers/openai-completions.ts:728`) — off by default
    for self-hosted; the pie manifest must set it.
  - `<!-- OPENCLAW_CACHE_BOUNDARY -->` splits stable prefix from dynamic
    suffix (`packages/ai/src/utils/system-prompt-cache-boundary.ts`); the
    completions path strips it (`openai-completions-params.ts:299`) — pie can
    instead consume it as an explicit prefix-pin hint.
  - Prompt-cache stability helpers already sort tools deterministically
    (`packages/ai/src/utils/prompt-cache-stability.ts`) — good for snapshot
    addressing.
- **Local-model machinery**: `localService` auto-spawn/health-check
  (`docs/gateway/local-model-services.md`); `ModelCompatConfig` switchboard
  (`src/config/types.models.ts:82` — `requiresStringContent`,
  `supportsTools`, `maxTokensField`, `toolSchemaProfile`, …); lean mode
  (`localModelLean`) trims the tool surface, auto-enabled for ollama/lmstudio
  (`src/config/local-model-lean-auto.ts:42`) — add pie.
- **Timeout profile** (differs from opencode — audit target): first-stream-event
  timeout + SSE read-idle watchdog ~120 s default + guarded-fetch total abort,
  all raised via provider `timeoutSeconds`. Whether SSE comment bytes reset the
  idle watchdog is **unverified** — the audit must decide `: ping` vs
  empty-delta keepalives.
- **Latency-sensitive extras** (Strategy B upside beyond opencode):
  heartbeat = periodic *full* agent turns, default 30 min
  (`docs/gateway/heartbeat.md`); compaction = an extra summarization LLM call +
  full re-prefill (`src/agents/sessions/agent-session-compaction.ts:130`);
  subagents/swarm = parallel model calls re-prefilling the shared prefix;
  memory search = embedding calls
  (`src/plugins/openai-compatible-embedding-provider.ts`).
- Networking: exact configured `baseUrl` origin is trusted for loopback;
  metrics/identity note — OpenClaw sends no identity header; the pie ingress's
  Bearer→Identity mapping (PA.2) covers it via `apiKey`.

---

## 1. Strategy A — bundled `pie` provider over the existing OpenAI ingress

**Shape.** Reuse the opencode Strategy-A stack unchanged on the pie side;
build the OpenClaw side as a bundled extension.

1. **Pie side: nothing new.** `gateway/src/ingress/openai.rs` +
   `inferlets/chat-completions` + `pie-openai-serving`, with the pending PA.1
   items (sessions/`cached_tokens`, launch-name fix) landing on the opencode
   track. One addition worth pulling forward: **sticky affinity keyed on
   `prompt_cache_key`** (present in every OpenClaw request once the compat flag
   is set) — the same keyed-affinity variant PA.2 deferred; opencode's
   `x-session-id` header and OpenClaw's `prompt_cache_key` should feed one
   mechanism in `gateway/src/session.rs`.
2. **`extensions/pie/` in OpenClaw** (mirror `extensions/vllm/`):
   - `index.ts` — `defineSelfHostedOpenAICompatibleProvider({ id: "pie",
     defaultBaseUrl: "http://127.0.0.1:8080/v1", apiKeyEnvVar: "PIE_API_KEY", … })`.
   - `openclaw.plugin.json` — `providers`, `modelCatalog.discovery.pie:
     "refreshable"`, `providerRequest.providers.pie.openAICompletions:
     { supportsStreamingUsage: true, supportsPromptCacheKey: true }`,
     `modelPricing.providers.pie.external: false`, setup env vars.
   - `localService` default block in docs: `command: pie, args: ["serve"]`,
     `healthUrl: http://127.0.0.1:8080/health` (the PA.2 route), generous
     `readyTimeoutMs` (model load).
   - Core touch-ups: `"pie"` in `BUILT_IN_MODEL_PROVIDER_OVERLAY_IDS`
     (`src/config/zod-schema.core.ts`), lean-mode auto-enable list
     (`src/config/local-model-lean-auto.ts`), `docs/providers/pie.md`.
3. **OpenClaw wire audit + fixtures** (P0.2 analog — the real work). Capture
   stock-OpenClaw traffic through a logging proxy into
   `tests/inferlets/fixtures/openclaw/`, with an `AUDIT.md` hazard table.
   Known question list, from the survey:
   - content shape: does OpenClaw send string content or structured parts
     (drives `requiresStringContent` vs ingress tolerance)?
   - keepalive semantics: do SSE comments reset the ~120 s idle watchdog?
     If not, switch the ingress's per-client keepalive to empty-delta chunks
     (the envelope already keeps wire logic in one place).
   - retry discipline on 4xx/5xx (opencode's never-500 rule came from
     unbounded 5xx retry; verify OpenClaw's `retry` settings and failover
     classifiers — `matchesContextOverflowError` strings must match pie's
     error bodies so compaction triggers instead of hard failure).
   - tool-schema noise under OpenClaw's `toolSchemaProfile` /
     `unsupportedToolSchemaKeywords` cleaning; tool-name case rules.
   - `prompt_cache_key` + `stream_options` actually on the wire once compat
     flags are set; cache-boundary marker stripped as documented.
   - the extra model calls: heartbeat turns, compaction summarization,
     title/utility calls — same-model? separate? (analog of opencode's
     `small_model` finding).
4. **Renderer parity**: fixtures re-run through the existing harness
   (`integrations/opencode/parity/`) — OpenClaw's system prompt and tool
   catalog are much larger than opencode's; this stresses the same D1–D4
   classes at longer lengths, plus lean-mode vs full-surface variants.

**What A buys / costs.** Identical to opencode's analysis: proven shape,
zero-risk client, works for every OpenAI-speaking client; inherits the
stateless tax (full re-upload + re-render + hash per turn, pressure-evictable
snapshots). OpenClaw's default surface makes the tax *bigger* than opencode's
(heavier system prompt + tools), which also makes the Strategy B delta larger.

**Effort.** OpenClaw extension: small (a vllm-clone plus manifest + docs).
Audit + fixtures: the bulk. Pie side: only the shared keyed-affinity item.

---

## 2. Strategy B — `openclaw-session` inferlet (programmable-KV-native)

**Thesis.** Unchanged from opencode's §2: one long-lived session inferlet per
OpenClaw session over the sticky WebSocket, owning the conversation's KV
working set; the client sends deltas, not history. The tool loop stays
client-side (attribution lesson); pie colocates the *context layer*.

**Wire.** No gateway changes — `GET /v1/ws` + `launch_process` +
`signal_process` + `session::send`. **Adopt the same JSON dialect as
opencode's Strategy B** (`opencode-integration.md` §2): `turn.append` /
`turn.edits` / `turn.gen`, `fork.branch_id`, and `delta` / `reasoning` /
`tool_call` / `finish` upstream. Anything OpenClaw needs that opencode
doesn't (e.g. an explicit `pin` edit for the cache-boundary marker) goes into
the shared dialect, versioned once.

**Pie side.** `inferlets/openclaw-session/` sharing the renderer/decoder/PTIR
core with `chat-completions` and `opencode-session` via `pie-openai-serving`
(+ a shared session-dialect crate). Per-branch `kv-working-set`, extended in
place; `update-index` checkpoint at every turn end so WS drop / worker
eviction degrades to a Strategy-A rebuild.

**Client side.** A **native OpenClaw provider transport** — cleaner than
opencode's path:

- Add `"pie"` to `MODEL_APIS` (`src/config/types.models.ts:14`).
- `extensions/pie/` grows a `createStreamFn` implementing the dialect over a
  Node WS client (the `ws` package — pie's bundled JS client neither appends
  `/v1/ws` nor can send headers from a browser; under Node both are fixable,
  or bypass the client and speak msgpack directly).
- The StreamFn keeps a shadow of server-held history keyed by
  `StreamOptions.sessionId`, diffs each incoming full-history `Context`
  against it, sends only the suffix; prefix divergence → renegotiate
  (re-render from divergence point or full rebuild = Strategy A code path).
- OpenClaw-side rewrites that must map to `edits` or force renegotiation:
  compaction rewrites, tool-output truncation, message editing. First profile
  disables auto-compaction (big `contextWindow`), as qwen-code/opencode did;
  B-2 then re-enables long-session viability server-side.
- Reconnect/fallback: on WS loss or process death, the StreamFn falls back to
  the Strategy A HTTP path transparently (same rendered bytes — the
  response/save unification invariant makes the rebuild hit the snapshot).

**What programmability buys, mapped to OpenClaw behaviors** (B-1…B-6 are the
opencode table; B-7/B-8 are OpenClaw-specific):

| # | OpenClaw behavior today | pie-native replacement | mechanism |
|---|---|---|---|
| B-1 | Full history + heavy system prompt + tool catalog re-sent/re-prefilled every turn | turn cost = O(new tokens); KV never leaves the worker | persistent working set |
| B-2 | Auto-compaction: extra summarization LLM call, then full re-prefill of rewritten history (`agent-session-compaction.ts`) | in-place eviction of stale tool outputs / old turns at token granularity; summarize *into* the retained prefix; no re-prefill | `working-set.discard/slice`, masks |
| B-3 | Subagents / delegate / swarm re-prefill the shared parent prefix per spawn | copy-on-write branch; subagent starts warm | `working-set.fork` (O(1)) |
| B-4 | Tool call surfaced when the SSE block closes | push each call the moment its arguments close; keep decoding parallel calls | incremental `tools.decoder` |
| B-5 | Tool result waits for next request's prefill | prefill result tokens as they stream over the socket, hidden behind tool latency | integrated I/O |
| B-6 | Local models emit tool calls as raw text → OpenClaw's documented #1 local pain (`docs/gateway/local-models.md:190`) | grammar-forced call arguments | `tools.format` + `grammar.wit`, behind a driver-capability probe |
| B-7 | Heartbeat: a *full* agent turn every 30 min over an almost-identical prefix | near-free: append the heartbeat prompt to the resident working set, decode, discard the branch | fork + discard on the session set |
| B-8 | Cache-boundary marker stripped; prefix stability is heuristic | explicit pin: everything above the marker published once via `update-index`, shared across sessions of the same agent config | named cross-process prefix cache |

Optional adjacency, not part of B proper: serve OpenClaw's memory-search
embeddings from the same engine (`registerEmbeddingProvider` hook) once an
embedding inferlet exists — keeps the whole local stack on one runtime.

**Risks / open questions** (beyond opencode's list, which all carry over):

- OpenClaw's multi-session concurrency (channels + heartbeats + subagents) ×
  one process per session = more concurrent inferlet processes than the
  opencode dev-box story; worker FCFS termination under pressure needs the
  checkpoint/rebuild path to be genuinely cheap.
- Session lifetime: OpenClaw sessions are long-lived (days); process = WS
  lifetime today. Decide: keep-alive with reattach (`attach_process`) vs
  relaunch-from-checkpoint per burst of activity.
- `sessionId` stability across gateway restarts and `/new` resets — audit how
  OpenClaw rotates ids so shadows don't go stale silently.
- Utility calls (compaction summarizer, titles) must not touch the session
  working set — route them to the plain chat-completions path.

---

## 3. Prerequisites and their status

| item | status | remaining for OpenClaw |
|---|---|---|
| Tool-history replay primitives (P0.1) | done on `liu/opencode-integration` | — |
| Shared `pie-openai-serving` crate (P0.3) | done | possible additions surfaced by the OpenClaw audit |
| Renderer parity harness (P0.4) | done, token-exact ×5 | run OpenClaw fixtures through it |
| Gateway OpenAI ingress (PA.2) | done | keyed sticky affinity (shared item) |
| `chat-completions` inferlet (PA.1) | milestone 1 | sessions/`cached_tokens`, launch-name fix — shared with opencode track |
| OpenClaw wire audit + fixtures | **not started** | the Phase-0 work of this plan |
| Session-inferlet wire dialect | designed in opencode §2, unimplemented | design review with OpenClaw's edit ops; then shared implementation |

**Branch strategy**: the OpenClaw work depends on `liu/opencode-integration`'s
infrastructure. Branch `liu/openclaw-integration` off it (worktree
`Lin_startup/pie-openclaw`, shared `CARGO_TARGET_DIR` per the disk-space note),
or wait until the opencode branch merges to `dev`. Do not fork the shared
crates.

## 4. Phasing and measurement

- **Phase 0** — OpenClaw wire audit + fixtures + parity run (§1.3–1.4).
- **Phase A** — `extensions/pie/` bundled provider + `localService` profile +
  keyed sticky affinity; acceptance = the 25-test suite generalized to the
  OpenClaw fixture set; e2e stock OpenClaw on toy tasks; A/B vs Ollama and
  llama.cpp (OpenClaw's incumbent local backends) on wall clock, prompt
  tokens, `cached_tokens`, and trajectory match.
- **Phase B1** — shared session dialect + `openclaw-session` inferlet +
  `createStreamFn` transport; A is the reconnect/rebuild fallback. Measure
  vs Phase A: bytes/turn, prefill tokens/turn, turn latency at 32k/128k,
  subagent spawn latency with/without fork, compaction cost with/without B-2,
  heartbeat turn cost (B-7 is free measurement — it fires every 30 min).
- **Phase B2 (optional)** — grammar-forced tool calls behind capability probe
  (B-6), speculative continuation during tool wait (B-4/B-5 depth), embedding
  serving.

Success criteria mirror the H200 lesson: no short-context claims; the win
regime is long contexts (OpenClaw's default surface is *always* long), warm
multi-turn sessions, subagent fan-out, and the heartbeat/idle-turn pattern
unique to OpenClaw.
