# OpenClaw wire audit (first pass) — OpenAI-compatible provider path

**Date:** 2026-08-13.
**Client under test:** OpenClaw `2026.8.1`, branch `liu/pie-provider` @ `e35577861f4`
(tree: `~/Documents/Liszt_ai/openclaw`). Line numbers below refer to it.
**Companion:** `../opencode/AUDIT.md`. Read that one first — this file is written as a
**delta** against it and does not repeat what is identical.

**Verification status: SOURCE-DERIVED THROUGHOUT. Nothing here is capture-verified yet.**
The opencode audit earned its authority from `wire/req-001..005.json` recorded against
`record_server.py`; this one has no captures. Treat every row as a hypothesis with a file
and line behind it, not as a pinned contract. **Step one for whoever picks this up is to
point `record_server.py` at OpenClaw and re-derive §1 from bytes** — the opencode audit
found things in the captures that reading the SDK would not have shown (the `: keepalive`
comment surviving the parser; `content: ""` on the tool-call assistant turn).

Why this cannot inherit the opencode table: opencode's custom-provider path always goes
through the AI SDK (`@ai-sdk/openai-compatible@2.0.41`). OpenClaw does not use the AI SDK
at all — it has its own transport under `packages/ai/src/transports/openai-completions-*`
and its own stream assembler. Different code, different hazards, in both directions.

---

## 0. Bottom line — what changes for pie

Three things pie must newly tolerate, and one thing pie must newly guarantee.

**Must tolerate** (opencode never sends these; OpenClaw does):

1. `store: false` — sent whenever `compat.supportsStore` resolves true
   (`openai-completions-params.ts:322-324`). Default for a self-hosted, non-standard
   endpoint needs confirming, but the code path exists and opencode has no equivalent.
2. `prompt_cache_key` (and possibly `prompt_cache_retention: "24h"`) —
   `openai-completions-params.ts:325-335`, gated on `compat.supportsPromptCacheKey`, which
   `docs/providers/pie.md` recommends turning **on**. Value is
   `` `${sessionId}:${boundaryCount}` `` (`src/agents/embedded-agent-runner/run/session-boundary-prompt-cache-key.ts:23`).
3. `tools: []` — the **empty array**, not an omitted key, when the turn has tool-call
   history but no active tools (`openai-completions-params.ts:375-377, 393-395`). opencode
   omits `tools` entirely on its title side-call. A deserializer that treats `tools: []` as
   "tool mode on, zero tools" will render a degenerate tool prompt.

**Must guarantee** (opencode tolerated its absence; OpenClaw does not):

4. **`data: [DONE]` is load-bearing for tool calls.** `openai-completions-stream.ts:501-511`
   promotes a tool-call-only response only if `sawStopFinishReason ||
   (sawNativeToolCallDelta && sawStreamDONE)`, and the comment is explicit that "EOF without
   [DONE] remains fail-closed". A stream that emits `delta.tool_calls`, no `finish_reason`,
   and then just closes → **OpenClaw drops the tool call**. Under opencode the same stream
   finished cleanly (`../opencode/AUDIT.md` §4). The gateway already sends `[DONE]` on the
   clean path (`gateway/src/ingress/openai.rs`); the risk is the abort path, which emits an
   `error` event instead.

---

## 1. Request wire format — delta vs opencode §1

### 1a. Endpoints hit

**OpenClaw probes `GET {baseUrl}/models`; opencode does not** (opencode §1: "No `/models`
probe, no health check, no preflight"). `discoverOpenAICompatibleLocalModels`
(`src/plugins/provider-self-hosted-setup.ts:180-235`):

- `GET {baseUrl}/models`, `Authorization: Bearer <key>` when a key is configured.
- **5 s timeout**, and the response is read through an SSRF-guarded fetch with a
  self-hosted base-URL policy.
- Expects `{"data": [ {"id": …}, … ]}`. A non-`ok` status is a warn + empty list, not a
  hard failure — discovery degrades quietly rather than blocking.
- Size-capped by `SELF_HOSTED_DISCOVERY_JSON_MAX_BYTES`.

`extensions/pie/openclaw.plugin.json` declares `modelCatalog.discovery.pie = "refreshable"`,
so this fires on catalog refresh, not just onboarding. The gateway's `/v1/models` handler
already exists and serves the single loaded model.

`/health` is hit only if the operator configures `models.providers.pie.localService.healthUrl`
(OpenClaw spawning `pie serve` itself). Not part of the request path.

### 1b. Top-level body fields

Same as opencode unless noted. All from `openai-completions-params.ts`.

| Field | OpenClaw behavior | vs opencode |
|---|---|---|
| `stream` | always `true` | same |
| `stream_options` | `{include_usage: true}` iff `compat.supportsUsageInStreaming` (`:319-321`) | opencode forces it unconditionally |
| `store` | `false` iff `compat.supportsStore` (`:322-324`) | **new** — never sent by opencode |
| `prompt_cache_key` | iff `compat.supportsPromptCacheKey` (`:325-335`) | **new** — opencode only with `setCacheKey:true` (default false) |
| `prompt_cache_retention` | `"24h"` iff long retention **and** `supportsLongCacheRetention` (`:332-334`) | **new** |
| `max_tokens` / `max_completion_tokens` | **one or the other**, chosen by `compat.maxTokensField` (`:442-448`; default rule at `openai-completions-compat.ts:142`) | opencode always `max_tokens`, never `max_completion_tokens` |
| `tools` | present with tools; **`[]`** with tool history and no tools; deleted entirely for proxy-like endpoints when empty (`:367-404`) | opencode: present or absent, never `[]` |
| `tool_choice` | only when the caller set one, **or** for proxy-like endpoints (`:378-392`) | opencode: always `"auto"` when tools present — **OpenClaw may send tools with no `tool_choice`** |
| `temperature`, `top_p` | only when set (`:336-341`) | same shape, different trigger |
| `response_format` | when set and `compat.supportsJsonSchemaResponseFormat` (`:342-354`) | opencode: never |
| `frequency_penalty`, `presence_penalty`, `seed`, `stop` | when set (`:355-366`) | opencode: never |
| reasoning params | family-specific (`applyQwenOpenAICompletionsThinkingParams` etc., `:468-475`) — **Qwen-specific paths exist and pie serves Qwen** | opencode: family-gated too, different families |

**Token-budget hazard, OpenClaw-only:** for proxy-like endpoints OpenClaw *estimates* input
tokens client-side and shrinks `max_tokens` to fit the declared `contextWindow`
(`:425-441`). This is why `docs/providers/pie.md` insists `contextWindow` equal the engine's
`max_model_len`: the handover §3 records that pie **refuses** an over-long prompt rather
than chunking it, so OpenClaw's preflight is the thing that has to fire first.

### 1c. Headers

`packages/ai/src/providers/openai-completions.ts:660-668`, when
`compat.sendSessionAffinityHeaders: true`:

```
session_id: <sessionId>
x-client-request-id: <sessionId>
x-session-affinity: <sessionId>
```

**`x-session-id` is NOT sent on this path** — that name is the OpenRouter branch only
(`:661-662`). This matters directly: `gateway/src/ingress/openai.rs:36-41` plans keyed
affinity on "the client's `x-session-id` header (opencode sends it)". **opencode sends both
`x-session-affinity` and `x-session-id`; OpenClaw sends only `x-session-affinity`.** Keying
on `x-session-affinity` serves both clients; keying on `x-session-id` silently drops
OpenClaw to ephemeral routing — which will look like "KV reuse doesn't help OpenClaw"
rather than like a header mismatch.

### 1d. Side calls

Unknown. opencode's per-session title call (§1d) is a real interleaving hazard and OpenClaw
almost certainly has its own analogue (title/summary/compaction calls). **Not yet traced —
this is the largest hole in this document.** The recorder will show it immediately.

---

## 2. Response requirements — delta vs opencode §3

### 2a. Tool-call deltas: OpenClaw is *more* forgiving than the AI SDK

The AI SDK throws `InvalidResponseDataError` if the first delta for a tool_call index lacks
`id` or `function.name` (opencode §3). OpenClaw's assembler does not
(`openai-completions-stream.ts:434-489`): it opens a block with `id: toolCall.id || ""` and
`name: toolCall.function?.name || ""` and fills either in from any later delta. It
correlates by `index` first, then by `id`.

So the single hardest constraint in the opencode audit **does not bind here**. Do not
relax the emitter on that basis — opencode remains a target client, and the constraint is
cheap to keep.

### 2b. `[DONE]` — see §0.4. This is the inversion: OpenClaw is *stricter*.

### 2c. Retries / 400-vs-500

`gateway/src/ingress/openai.rs:30-33` reserves 500 for real faults because "opencode retries
5xx without bound". **That rationale does not transfer as-stated.** OpenClaw builds its
client with `maxRetries: options?.maxRetries ?? 0`
(`packages/ai/src/providers/openai-completions.ts:163`) — transport retries are off by
default, like opencode's `streamText({maxRetries: 0})`. Whether OpenClaw has an *outer*
session-level retry loop equivalent to opencode's `Effect.retry` was **not found and not
ruled out**; `model-fallback-runner.ts` is the place to look next.

Keep the 400-vs-500 discipline regardless — it is correct on its own merits — but do not
cite the unbounded-retry justification for OpenClaw until someone confirms it.

### 2d. Usage

OpenClaw reads streaming usage when `stream_options.include_usage` was sent. Whether it
reads `prompt_tokens_details.cached_tokens` into a cache-read figure the way the AI SDK does
(opencode §5) is **unverified** — `openai-completions-stream.usage.test.ts` is the file to
read. Emitting `cached_tokens` costs nothing and is already implemented for opencode.

---

## 3. Open items, in the order they should be closed

1. **Record captures.** Point `record_server.py` at OpenClaw
   (`models.providers.pie.baseUrl` → the recorder), run one plain turn and one tool turn,
   and re-derive §1 from bytes. Everything above is a source read.
2. **Trace the side calls** (§1d). Unknown request shapes arriving interleaved on the same
   session is exactly the class of surprise that cost time on the opencode side.
3. **Decide the affinity header** (§1c) before keyed affinity is implemented, not after.
4. **Confirm the outer retry loop** (§2c).
5. **`tools: []` and `store: false` handling** in the `chat-completions` inferlet (§0.1,
   §0.3) — these are cheap to accept and silent when wrong.
