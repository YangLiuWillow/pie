# OpenClaw wire audit (oc-P0.1 / oc-P0.2) — OpenAI-completions provider path

**Date:** 2026-08-11.
**Source cross-reference:** `/Users/yangliu/Desktop/Lin_startup/openclaw` (repo HEAD,
`2026.8.1`); line numbers refer to it. Uses official `openai` npm SDK `6.49.0`
(`package.json:2057`) — SDK internals could not be read (no `node_modules` in the
checkout); SDK-attributed claims are marked *(SDK, uncited)*.
**Verification status:** §1–§2 request-side claims are **capture-verified** against
`wire/req-002..006.json` where §8 says so; everything else is **source-derived**
(oc-P0.1). Capture client is the published npm CLI `2026.7.1-2` (SDK **6.45.0**),
which diverges from repo HEAD in ways §8 enumerates — pie must tolerate both.

Companion: the opencode audit (`../opencode/AUDIT.md`). §6 lists the divergences —
they are the load-bearing content for the shared gateway ingress.

---

## 0. Which code path runs

For a config-defined custom provider (`api: "openai-completions"`, custom `baseUrl`)
under the embedded agent runner, the wire code is the **managed transport**:
`src/agents/embedded-agent-runner/stream-resolution.ts:210-240` →
`packages/ai/src/transports/openai-completions-transport.ts:173` →
`packages/ai/src/transports/openai-completions-params.ts:292` (`buildOpenAICompletionsParams`).
The near-duplicate `packages/ai/src/providers/openai-completions.ts` ("path B",
AgentSession/TUI `streamSimple`) differs in retries/headers; noted where it matters.

**Endpoint class drives defaults** (`src/agents/provider-attribution.ts:280-315`):
`127.0.0.1`/`localhost`/`::1`/`*.local`/`*.internal` ⇒ `"local"`; any other custom
host ⇒ `"custom"`. Both count as proxy-like (`:521-531`). Consequences below.

## 1. Request wire format

Only endpoint hit for chat: `POST {baseUrl}/chat/completions`. Model discovery
(`GET /models`) fires only via provider-setup flows, not per turn.

### 1a. Top-level body (defaults for a custom provider, no `params` configured)

| Field | Behavior | Source |
|---|---|---|
| `model` | config model id | — |
| `stream` | always `true` (no non-streaming path) | `openai-completions-params.ts:317` |
| `stream_options` | `{"include_usage":true}` **only when** `compat.supportsUsageInStreaming`; default TRUE for endpoint class `local` (loopback), FALSE for remote `custom` | `:319-321`, `openai-completions-compat.ts:136-141` |
| `max_completion_tokens` | **default field** (`compat.maxTokensField` default `"max_completion_tokens"`); value always sent: `model.maxTokens` clamped to `effectiveContextTokens − estimatedInput − 1` for proxy-like endpoints (char/4×1.25 estimator, images 8000 chars) | `:425-448`, `openai-completions-compat.ts:142` |
| `tools` / `tool_choice` | tools name-sorted; `tool_choice: "auto"` forced whenever tools present (proxy-like); both deleted when tools empty | `:367-404`, sort `:266` |
| `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `response_format`, `n`, `parallel_tool_calls` | **all omitted** unless set via config `params` | `:336-366`, `extra-params.ts:471-604` |
| `prompt_cache_key` | only when `compat.supportsPromptCacheKey` (default false; **no api.openai.com fallback on this path**); value = `` `${sessionId}:${boundaryCount}` ``, ≤64 code points | `:325-326`, `run/session-boundary-prompt-cache-key.ts:23`, `openai-prompt-cache.ts:2-14` |
| `prompt_cache_retention` | `"24h"` only when retention `"long"` (`OPENCLAW_CACHE_RETENTION=long`) + `supportsLongCacheRetention` | `:332-333` |
| `store`, `user`, `safety_identifier`, reasoning fields | never for default custom compat (`reasoning_effort` needs `compat.supportsReasoningEffort`, default false for proxy-like) | `:322-324`, `:492-500`, `extra-params.ts:699-728` |
| `params.extra_body` | payload-level `Object.assign` patch, wins over built body (then `store`-strip and `parallel_tool_calls` wrappers still run after it) | `extra-params.ts:794-811, 927-947` |

### 1b. Message shapes

- `system`: **single first message, plain string** (`developer` role needs
  `model.reasoning` + `compat.supportsDeveloperRole`, default false for custom) —
  `openai-completions-messages.ts:86-93`.
- `user`: **content-part array** `[{"type":"text","text":…}]` even for plain text —
  the agent path always normalizes prompts to parts (`packages/agent-core/src/agent.ts:459-463`,
  `openai-completions-messages.ts:121-149`). Flattens to string only with
  `compat.requiresStringContent` (`openai-completions-string-content.ts:5-40`;
  bails out if any non-text part).
- assistant tool-call replay: **`content: null`** (key present) alongside
  `tool_calls:[{id, type:"function", function:{name, arguments:<json-string>}}]`;
  string content when the turn had visible text — `:151-241`. (opencode sent `""` —
  accept both.)
- tool result: `{"role":"tool","content":"<string>","tool_call_id":…}`; empty →
  `"(no output)"`; `name` only with `compat.requiresToolResultName` — `:242-266`.
  Tool-result images hoisted into a following synthetic user message — `:284-292`.
- images: `{"type":"image_url","image_url":{"url":"data:<mime>;base64,…"}}`, never
  remote URLs, no `detail`; non-standard `video_url` for video-capable models —
  `:122-140`.
- `compat.strictMessageKeys` strips everything but `role`+`content` (kills
  `tool_calls` replay — do not set for pie) — `openai-completions-string-content.ts:43-58`.

### 1c. Tools

- Shape `{"type":"function","function":{name,description,parameters}}`; **no
  `strict` key** for custom endpoints (`openai-strict-tool-setting.ts:46-57`).
- **Default surface ≈40 tools** (assembled conditionally: 6 core coding + ~34
  openclaw tools; realistic installs 45–60 with plugins/MCP) — serialized order
  **50–120 KB**. Heavy schemas: `computer`, `sessions_spawn`, `message`, `nodes`,
  `automations` (9 unions, `maxLength:65536`) — `src/agents/openclaw-tools.ts:469-723`,
  `core-coding-tools.ts`, `tools/cron-tool-schema.ts:195`.
- **Lean mode (`localModelLean`) wire list = 9 tools** (+`message` when it owns the
  reply): `apply_patch, edit, exec, process, read, tool_call, tool_describe,
  tool_search, write` — deny list + Tool Search forced on
  (`src/agents/local-model-lean.ts:15-30`, `tool-search-catalog.ts:228-238`).
- Schema hygiene: **no `$schema`**; nothing stripped by default
  (`unsupportedToolSchemaKeywords` default `[]`, `toolSchemaProfile` default unset) —
  nested `anyOf`/`oneOf`, `maximum:1000000`, `minLength`, `pattern`, `format` all go
  out as-is; top-level unions flattened, `$ref` inlined, `nullable` normalized —
  `agent-tools-parameter-schema.ts:801-940`.
- Ordering: name-sorted (raw codepoint) on **every** request — good prefix
  stability — `prompt-cache-stability.ts:10-27`.
- **Mid-session tool churn exists** (KV-prefix hazard for Strategy B): heartbeat
  turns add `heartbeat_respond`; memory-flush turns collapse to `read`+`write`
  (with mutated `write` description); `exec`/`process`/`message` descriptions are
  conditionally dynamic — `agent-tools.ts:111,451-455,849-871`,
  `message-tool-description.ts:12-29`.
- Inbound tool-name repair is aggressive (case-insensitive match, alias map
  `bash→exec`, id-derived recovery, plain-text call promotion) — but emit exact
  lowercase names anyway — `run/attempt-tool-call-name-resolution.ts:190-223`,
  `packages/tool-call-repair/`.

### 1d. System prompt

- One string; the internal stable/volatile split marker
  `\n<!-- OPENCLAW_CACHE_BOUNDARY -->\n` is **stripped** (twice), leaving a
  **three-newline run** at the boundary site — `system-prompt.ts:1430`,
  `openai-completions-params.ts:299-304`, `system-prompt-cache-boundary.ts:10-12`.
- Volatile tail (below the boundary, inlined on the wire): `## Temporal Context`
  (date, midnight-granularity), dynamic project context, `## Conversation Context`,
  channel sections, `## Runtime` line (session/model/capabilities; biggest churn:
  live exec sessions with pid+cwd) — `system-prompt.ts:1433-1609`.
- For KV prefix reuse: the stable prefix is byte-identical across turns but
  undelimited on the wire → longest-common-prefix matching per session is the
  mechanism (pie's trie does this natively). No timestamps finer than the date.

### 1e. Headers

| Header | Value | Source |
|---|---|---|
| `Authorization` | `Bearer <apiKey>` always (SDK) | `openai-completions-transport.ts:105` |
| `Content-Type` | `application/json` (SDK) | |
| `Accept` | `application/json` — **`text/event-stream` is NOT sent; don't require it** | `openai-transport-params.ts:352-355` |
| `User-Agent` + `x-stainless-*` | openai SDK 6.49.0 defaults | transport tests |
| provider/model `headers`, `request.headers` | config passthrough (caller-wins merge; attribution keys protected) | `provider-request-config.ts:711-732` |
| session headers | **NONE by default.** `session_id`/`x-client-request-id`/`x-session-affinity` only with `compat.sendSessionAffinityHeaders: true` (and path B); path A never emits them | `openai-completions.ts:660-668`, `openai-completions-compat.ts:226-241` |
| attribution headers | none for custom/local endpoints | `provider-attribution.ts:552-562` |

⇒ **The only session signal on the default wire is `prompt_cache_key`** (opt-in
via compat). Sticky affinity for pie: key on `prompt_cache_key`, with
`sendSessionAffinityHeaders` as the header-based upgrade.
`resolveTransportTurnState` is Responses-API-only — not usable here.

## 2. Streaming response contract

### 2a. Keepalive — the load-bearing divergence from opencode

**SSE comments (`: ping`) are useless AND invisible**: the sanitizer
(`src/agents/provider-transport-fetch.ts:154-305`, on for every non-openai.com
provider) **drops frames with no readable `data:` payload**, and both watchdogs
reset only on **parsed chunks**:

- First-event timeout: armed before the first SDK chunk; **120 s** (remote) /
  **300 s** (loopback/private/self-hosted) — `stream-first-event-timeout.ts:82-134`,
  `run/llm-idle-timeout.ts:360-411`.
- Idle watchdog: reset per parsed chunk (`notifyLlmRequestActivity`,
  `openai-completions-stream.ts:330`); **120 s** cloud, **300 s** self-hosted
  hostname, **disabled (0)** for loopback/private baseUrls, 60 s cap under cron
  trigger — `run/llm-idle-timeout.ts:22-28, 297-355, 424-612`.
- Transport byte-timer exists but only when `models.providers.<id>.timeoutSeconds`
  is set (`provider-transport-fetch.ts:607-620`); that key also raises/overrides
  the two timers above (`llm-idle-timeout.ts:297-315,394-401`).

⇒ **Keepalive must be an empty-delta chunk** `{"choices":[{"index":0,"delta":{}}]}`
— fully tolerated, emits no events, resets the full idle budget
(`openai-completions-stream.ts:325-343`). Cadence < 60 s covers the cron cap.
(Also safe for opencode, whose watchdog resets on raw bytes ⇒ make empty-delta
chunks the ingress's universal keepalive; drop `: ping`.)

### 2b. Tool-call deltas

Accumulator `openai-completions-stream.ts:426-490` + finalizer
`openai-completions-tool-calls.ts:182-249`:

- First delta need not carry `id`/`name` (filled in later) — but **always send
  `index`** (missing index+id ⇒ every delta becomes a new broken call). Whole
  call in one delta is fine; args may be split (256 KB cap).
- **Finalization is strict**: non-empty `name` AND `arguments` parsing to a JSON
  **object**, else the whole turn errors (`"Provider returned an incomplete or
  malformed tool call"`). **No-arg tools must send `arguments: "{}"`**.
- `finish_reason: "stop"` with a tool call: promoted only if there was **no
  visible text**; with text, tool calls are **silently dropped** — always send
  `finish_reason: "tool_calls"`.
- Missing finish_reason + tool calls: promoted only when `data: [DONE]` was seen
  (overflow-safe detector, lines >1024 chars poisoned) —
  `openai-completions-transport.ts:55-104, 278`.

### 2c. finish_reason / usage / termination

- Allowed values: `stop`, `end`, `length`, `tool_calls`, `function_call`,
  `tool_call`, `content_filter`(→error), `network_error`(→error), null/absent(→stop).
  **Any other string fails the turn** (`Provider finish_reason: <x>` →
  transport throw) — `openai-stop-reason.ts:8-38`, `transport-stream-shared.ts:170-183`.
- Usage: read from last usage-bearing chunk; `choices: []` usage chunk accepted;
  `cacheRead` = `prompt_tokens_details.cached_tokens` (fallback
  `prompt_cache_hit_tokens`), `cache_write_tokens` honored, **`prompt_tokens`
  must be inclusive of cached tokens** (client computes
  `input = prompt − cacheRead − cacheWrite`), `completion_tokens_details.
  reasoning_tokens` honored, provider `total_tokens` ignored —
  `openai-transport-shared.ts:72-103`.
- `data: [DONE]`: optional for plain turns, required for the tool-call promotion
  path — always send it.
- Empty response (no content at all): retried once
  (`DEFAULT_EMPTY_RESPONSE_RETRY_LIMIT = 1`) — pie's non-empty-content floor
  already avoids this.

### 2d. Reasoning

- `<think>` tags in `content` are **stripped, not surfaced** (Markdown-aware
  partitioner; no thinking events) — `openai-completions-stream.ts:288-292`,
  `packages/markdown-core/src/reasoning-tags.ts`.
- Thinking events come only from `reasoning_content` / `reasoning` /
  `reasoning_text` / `reasoning_details[].type=="reasoning.text"` deltas, gated on
  `model.reasoning === true` (catalog flag) — `:529-598`.
  ⇒ pie should emit `reasoning_content` deltas (the inferlet's reasoning decoder
  maps 1:1) and the pie model entry must set `reasoning: true`.
- Replay: for `model.reasoning: true` custom providers, `reasoning_content` on
  replayed assistant messages is preserved (trust rule) — `openai-completions-replay.ts:259-268`.

### 2e. In-stream errors / content-type

- `data: {"error":{…}}` → SDK throws *(SDK, uncited)* → turn error with projected
  message; `event: error` **without a data line is silently dropped** by the
  sanitizer (never signal errors that way).
- Streaming responses must be `text/event-stream` (or JSON/empty, which gets
  sniffed and relabeled/wrapped); other content-types hard-fail
  (`invalid_provider_content_type`) — `provider-transport-fetch.ts:383-437,921-932`.
- Never emit SSE frames whose only `data:` lines are blank — dropped (`:74-84`).

## 3. Errors, retries, context overflow

- **All retries bounded** (divergence from opencode's infinite-5xx): SDK ~2
  retries (path A; path B 0), session retry ×3 (2/4/8 s), transient-HTTP turn
  retry ×1 (2.5 s), overload ×10 (2.5→30 s), same-model rate-limit ×3
  (10/20/30 s), outer run budget 32–160 iterations, then model
  failover — `agent-session-execution.ts:21-95`,
  `agent-runner-error-handler.ts:53-60,271-282,399-413`, `run/helpers.ts:41-134`.
- Retryable = status ∈ {429,500,502,503,504,524} or transient-text evidence
  (`retry-evidence.ts:15-58`); 400 fails fast (`format`) **but must carry a body**
  (empty-body 400/422 classifies null → failover loop) —
  `classification-rules.ts:52-59,301-317`.
- Mid-stream failures are **fully replayed** (no completions-path replay-unsafe
  guard) — make generation idempotent-safe.
- 429: send `Retry-After` (≤60 s to be honored; without the header OpenClaw
  injects `x-should-retry: false` and skips SDK retry) and echo the delay in the
  message text (OpenClaw re-parses it from the body) —
  `provider-transport-fetch.ts:54,256-284,523-541`, `retry-evidence.ts:26-28`.
- **Context overflow recipe** (triggers compaction instead of failure): HTTP 400 +
  ```json
  {"error":{"message":"This model's maximum context length is N tokens. However, your messages resulted in M tokens. Please reduce the length of the messages.","type":"invalid_request_error","code":"context_length_exceeded","param":"messages"}}
  ```
  Matches `failover-explicit` (`overflow.ts:73-98`) + `assistant-error`
  (`overflow.ts:44-71`); message classification survives the 400 status rule
  (`classification-rules.ts:305-307`). **Never include** `rate limit`, `too many
  requests`, `tpm`, `tokens per minute`, or `quota` in that body (veto patterns).
  Plugin hook `matchesContextOverflowError` can make this authoritative later —
  `provider-patterns.ts:103-121`.
- Auth: 401 `"invalid api key"` → one auth refresh then rotate; revoked/disabled
  wording → `auth_permanent` (disables the profile). Keep pie's 401 wording plain.

## 4. HTTP transport

- **HTTP/1.1 only** (`allowH2: false` on all dispatchers) with keep-alive
  connection pooling (16 origins, 60 s idle) —
  `undici-dispatcher-options.ts:129-138`, `provider-transport-dispatcher-pool.ts:10-19`.
- Loopback baseUrl trusted via exact-origin allowlist; metadata/link-local always
  blocked; `redirect: manual` — `provider-transport-fetch.ts:666-720,812-822`.
- Env proxies honored; `models.providers.<id>.request.{proxy,tls}` forces the
  managed transport; `insecureSkipVerify` forbidden.

## 5. Extra model calls (Strategy B session-hygiene inventory)

| Call | Trigger | Model default | Session | Tools | Hits pie? |
|---|---|---|---|---|---|
| Main turn | user msg | session model | main | ~40 (lean 9) | yes |
| Compaction summary (+ split-turn 2nd call) | threshold/overflow/`/compact` | session model (`compaction.model` unset) | same id, 1-msg ctx, **no tools** | none | **yes** |
| Memory flush | pre-compaction (default on) | session model | same session | **2** (`read`,`write`) | **yes** |
| Heartbeat turn | every 30 m | session model | **main session** (or `:heartbeat` if isolated) | ~41 (+`heartbeat_respond`) | **yes** |
| Session title | first dashboard turn | utility → **primary fallback** | none | none | yes |
| Progress narration | long turns | utility → primary fallback | none | none | yes |
| Tool-call titles | Control UI | utility only, no fallback | none | none | only if configured |
| `openclaw` delegate / `sessions_spawn` / `subagents` | model-invoked | default-agent / target-agent route | separate sessions | different surfaces | usually yes |
| Active Memory recall | pre-reply (escalate) | agent model | sub-agent | recall set | yes |
| Embeddings | memory index/search | **openai** default | n/a | n/a | no, unless `openai-compatible`→pie |

Strategy-B policy implications: compaction/title/narration calls must go to the
plain chat-completions path, never the session working set; heartbeat + memory
flush change `tools[]` mid-session → dialect needs a tools-digest field per turn
(renegotiate or branch on change).

## 6. Divergences from the opencode audit (shared-ingress checklist)

| # | Dimension | opencode | OpenClaw |
|---|---|---|---|
| D-1 | Keepalive | raw bytes reset watchdog; `: ping` OK | comments dropped/ignored; **empty-delta chunk required** (§2a) |
| D-2 | user content | plain string | **parts array** |
| D-3 | assistant tool-call `content` | `""` | **`null`** |
| D-4 | max-tokens field | `max_tokens` always | **`max_completion_tokens`** (default compat) |
| D-5 | `stream_options` | always sent | only loopback endpoint class (or compat flag) |
| D-6 | 5xx retry | **unbounded** (never-500 rule) | bounded ladders + failover |
| D-7 | Session headers | `x-session-id`/`x-session-affinity` always | none by default; `prompt_cache_key` (opt-in) / `sendSessionAffinityHeaders` (opt-in) |
| D-8 | First tool-call delta | must carry `id`+`name` (AI SDK throws) | tolerant; strictness moved to finalization (args must parse to object) |
| D-9 | Unknown finish_reason | mapped to `"unknown"`, harmless | **fails the turn** |
| D-10 | Tool schema noise | `$schema` + `maximum:2^53−1` present | no `$schema`; big unions/bounds; nothing stripped by default |
| D-11 | Tools size | 10 tools, small | ~40 tools, 50–120 KB (lean: 9) |
| D-12 | Title side-call | every session, same model | dashboard sessions only; utility-model fallback chain |
| D-13 | `Accept` | `*/*` | `application/json` |
| D-14 | Empty-content response | ends turn quietly | retried once |
| D-15 | `<think>` handling | n/a (surfaced as content) | stripped silently; reasoning must use `reasoning_content` |

Shared-ingress consequences: keepalive = empty-delta chunks universally (safe for
both); accept `content: null|""|absent` on assistant replay; accept both
`max_tokens` and `max_completion_tokens`; accept string **and** parts content;
emit only mapped finish_reason values; report cache hits inside inclusive
`prompt_tokens`.

## 7. Recommended pie provider config (for oc-PA.2)

```json5
// extensions/pie manifest / model compat
compat: {
  supportsPromptCacheKey: true,     // emit prompt_cache_key = sessionId:boundaryCount
  sendSessionAffinityHeaders: true, // session_id / x-session-affinity headers (path B)
  supportsUsageInStreaming: true,   // future-proof for non-loopback hosts
  // leave maxTokensField default; pie ingress accepts max_completion_tokens
  // do NOT set: requiresStringContent, strictMessageKeys (both lossy)
}
// model entry: reasoning: true  (enables reasoning_content surfacing)
// provider: timeoutSeconds for slow cold loads; localService { command: "pie", args: ["serve"], healthUrl: ".../health" }
```

## 8. Capture results (oc-P0.2) — verified rows and version skew

Captured with `openclaw@2026.7.1-2` (npm) / SDK 6.45.0; source audit targets repo
`2026.8.1` / SDK 6.49.0. Inventory in `README.md`.

**Capture-verified (agree with §1–§2):** single string system prompt (33 KB) with
the **three-newline boundary artifact** and no marker; tools name-sorted;
`tool_choice:"auto"`; `max_completion_tokens` (=32000, the model's maxTokens);
`stream_options:{include_usage:true}` on loopback; `Accept: application/json`;
`Authorization: Bearer`; no `prompt_cache_key`/`temperature`/`top_p`/`store` by
default; assistant tool-call replay `content: null` (D-3); tool result = plain
string with `tool_call_id`; empty-turn retry observed (one retry, D-14);
lean mode drastically reduces the surface; recorder's SSE (comment line, split
tool-call deltas, usage-only chunk, `[DONE]`) accepted end-to-end.

**Version-skew divergences (published CLI vs repo-HEAD source):**

| # | Wire item | npm `2026.7.1-2` (fixtures) | repo `2026.8.1` (source audit) |
|---|---|---|---|
| S-1 | user content | **plain string**, with a `[Tue 2026-08-11 18:08 PDT] ` envelope prefix | content-part array (§1b) |
| S-2 | tool `strict` | **`strict: false` present on every tool** | key absent for custom endpoints (§1c) |
| S-3 | lean wire list | **4 tools**: `exec, tool_call, tool_describe, tool_search` | 9 tools incl. read/write/edit/apply_patch/process |
| S-4 | default surface | 34 tools (incl. `cron`, `browser`, `canvas`, `dir_*`, `file_*`, `node_inference`, `memory_*`) | ~40 tools, `automations` (alias `cron`), different roster |
| S-5 | tool-call id replay | normalized: `call_record_001` → `callrecord001` (underscores stripped) — ids don't round-trip verbatim (they stay self-consistent within a request) | `[^a-zA-Z0-9_-]`→`_`, ≤40 chars (§1b) |
| S-6 | SDK / UA | `OpenAI/JS 6.45.0` | 6.49.0 pinned |

Ingress consequences: accept string **and** parts user content (already required
by D-2 vs S-1 — both shapes are in the wild); tolerate `strict: false` on tool
schemas; never key snapshots or matching on tool-call ids surviving verbatim;
the timestamp envelope prefix means the **user message** (not just the system
suffix) has per-turn dynamic content — session-inferlet delta protocol is
unaffected, but content-addressed (Strategy A) snapshot reuse must treat the
trailing user turn as always-fresh (it already does: resume strips the trailing
user/tool suffix).

Pie-side ingress/inferlet action items surfaced by this audit:
1. Keepalive: switch per-client `: ping` to empty-delta chunks (universal).
2. Parser: accept content-part arrays; assistant `content: null`;
   `max_completion_tokens`; absent `stream_options` (usage chunk still fine to send).
3. Emitter: only mapped finish_reason values; `arguments: "{}"` floor for no-arg
   calls; always `[DONE]`; `reasoning_content` for thinking deltas.
4. Errors: context-overflow 400 body per §3 recipe; 429 with Retry-After ≤60 s;
   never empty-body 4xx.
5. Affinity: key sticky routing on `prompt_cache_key` (body) with
   `x-session-affinity`/`session_id` headers as the upgrade path.
6. HTTP/1.1 keep-alive; content-type `text/event-stream`; no blank-data frames.
