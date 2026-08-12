# opencode wire audit (Phase 0.2) — OpenAI-compatible provider path

**Date:** 2026-08-11.
**Client under test:** installed CLI `opencode-ai@1.18.16` (via `npx`), which bundles
`@ai-sdk/openai-compatible@2.0.41` / `@ai-sdk/provider-utils@4.0.23` — the exact versions
pinned in the source repo (`packages/opencode/package.json:71`) and confirmed by the
`User-Agent` header on the wire.
**Source cross-reference:** `/Users/yangliu/Documents/Liszt_ai/opencode` (repo HEAD; line
numbers below refer to it). SDK source read from `@ai-sdk/openai-compatible@2.0.41` dist.
**Captures:** `wire/req-001..005.json` in this directory, recorded by `record_server.py`
against the config in `./opencode.json` (custom provider `pie`, npm
`@ai-sdk/openai-compatible`, baseURL `http://127.0.0.1:8123/v1`). See `README.md` to re-run.

**Verification status:** everything in §1–§2 and §6–§7 is **capture-verified** unless marked
*(source)*. §3–§5 are **source-derived** (repair/retry/timeout paths are client-internal and
were not forced to fire), except where noted.

Which runtime handles this provider: the native `@opencode-ai/llm` runtime is gated to
`providerID ∈ {openai, anthropic, opencode*}` (`packages/opencode/src/session/llm/native-runtime.ts:54-59`),
so a custom provider id like `pie` **always** goes through the AI SDK `streamText` path
(`packages/opencode/src/session/llm.ts:280`) even though its npm package would qualify.
Capture-verified via the `ai-sdk/provider-utils` User-Agent.

---

## 1. Request wire format

Only endpoint hit: `POST {baseURL}/chat/completions`. No `/models` probe, no health check,
no preflight (captures contain no other paths).

### 1a. Top-level body fields (capture: req-002/004/005)

| Field | Value / behavior | Source |
|---|---|---|
| `model` | config model key (`test-model`) | — |
| `max_tokens` | **always sent** = `min(model.limit.output, 32000)`; `OUTPUT_TOKEN_MAX = 32_000` | `provider/transform.ts:18,1418` |
| `max_completion_tokens` | **never** (chat impl of SDK 2.0.41 emits `max_tokens` only) | SDK dist `index.js:525,1251` |
| `stream` | always `true` on the agent path | — |
| `stream_options` | always `{"include_usage": true}` — forced on for every `@ai-sdk/openai-compatible` provider unless config sets `includeUsage:false` | `provider/provider.ts:1694` |
| `tools` | present when agent has tools (see 1c); **absent** (not `[]`) on the title side-call | capture |
| `tool_choice` | `"auto"` whenever `tools` present; absent otherwise | capture |
| `temperature` / `top_p` | **absent** for this config. Only sent if the agent config sets them or the model id matches a known family (qwen→0.55, glm/minimax→1.0, …). Unknown ids ⇒ `undefined` ⇒ omitted | `provider/transform.ts:526-560` |
| `parallel_tool_calls`, `seed`, `logprobs`, `n`, `response_format`, `stop` | never sent | capture + SDK source |
| reasoning fields (`reasoning_effort`, `thinking`, `chat_template_kwargs`, …) | none for this config; only injected for specific model-id families or `--variant` on reasoning-capable models *(source)* | `provider/transform.ts:1195-1270` |
| `cache_control` / `prompt_cache_key` | none. Anthropic-style caching applies only to claude/anthropic ids; `prompt_cache_key` for openai-compatible only when `options.setCacheKey: true` (default false) | `transform.ts:467-483,1400-1410` |

### 1b. Message shapes (capture: req-005 — the history-replay request)

- `system`: single first message, plain **string** content (~30 KB build-agent prompt).
- `user`: plain **string** when a single text part; array of `{type:"text"|"image_url",...}`
  parts only with attachments *(source: SDK `convertToOpenAICompatibleChatMessages`)*.
- assistant tool-call turn, exactly as replayed:

  ```json
  {"role": "assistant", "content": "",
   "tool_calls": [{"id": "call_record_001", "type": "function",
     "function": {"name": "read", "arguments": "{\"filePath\":\"...\"}"}}]}
  ```

  Note `content` is the **empty string**, not `null`/omitted — a strict deserializer must
  accept `""`. The tool-call `id` is echoed back verbatim.
- tool result:

  ```json
  {"role": "tool", "tool_call_id": "call_record_001", "content": "<path>...</path>\n<type>file</type>\n<content>..."}
  ```

  `content` is a plain string (opencode wraps tool output in pseudo-XML).
- No `name` field on any message; no assistant `reasoning_content` replay on this path
  (reasoning parts are dropped for openai-compatible chat unless the server streamed
  `reasoning_content`, which round-trips via `reasoning_content` on the assistant message
  *(source: SDK dist:200-215)*).

### 1c. Tools schema (capture: req-004)

`{"type":"function","function":{"name","description","parameters"}}` — no `strict` field
(that's only forced for `@ai-sdk/openai`/azure/mantle, `session/llm/request.ts:152-158`).
Tools are **name-sorted** (`request.ts:184`). Default build-agent set (10):
`bash, edit, glob, grep, read, skill, task, todowrite, webfetch, write`.
A hidden `invalid` tool exists client-side but is excluded from the wire via `activeTools`
(`llm.ts:317`).

**Hazard:** `parameters` is a JSON Schema that includes
`"$schema": "https://json-schema.org/draft/2020-12/schema"` and integer bounds like
`"maximum": 9007199254740991` (2^53−1). Strict schema validators (OpenAI strict mode,
some vLLM tool parsers) reject or mangle these — the pie daemon must pass them through.

### 1d. Side calls: title generation (capture: req-001/003)

Every new session fires a **separate** title request to the same model/endpoint, before or
concurrently with the first agent request: no `tools`/`tool_choice`, 3 messages —
title system prompt (~2 KB), `{"role":"user","content":"Generate a title for this conversation:\n"}`,
then the user's prompt **JSON.stringify'd** (`"\"say hello\""`). The daemon will see these
interleaved small requests on the same session headers; don't assume one request per turn.

### 1e. Headers (capture)

```
Authorization: Bearer <options.apiKey>          (always, even dummy values)
Content-Type: application/json
User-Agent: opencode/1.18.16 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14
x-session-affinity: ses_...                     (session id)
x-session-id: ses_...                           (same value)
Accept: */*                                     (NOT text/event-stream)
Accept-Encoding: gzip, deflate, br, zstd
Connection: keep-alive
```

Subagent sessions additionally send `x-parent-session-id` *(source:
`session/llm/request.ts:196-201`)*. `x-session-affinity` is what opencode's own OpenAI
plugin uses for sticky routing — pie can use it the same way for KV-cache affinity.

---

## 2. chunkTimeout / keepalive — the watchdog resets on RAW BYTES

This is the load-bearing answer for pie's keepalive strategy.

- `wrapSSE` (`packages/opencode/src/provider/provider.ts:37-83`) wraps the **raw
  `Response.body` reader inside the custom `fetch`**, upstream of the AI SDK's SSE parser.
  The timer is armed per `reader.read()` call: **any bytes** delivered — SSE comment lines,
  partial events, whitespace — clear it. It does *not* wait for a parsed
  `chat.completion.chunk`.
- The SSE parser (`eventsource-parser` via provider-utils 4.0.23) drops `:`-prefixed
  comment lines silently (`onComment` not wired). **Capture-verified:** our recorder's
  streams began with a `: keepalive` comment and both turns completed normally.
- **Conclusion: periodic SSE comments (`: ping\n\n`) are a valid liveness signal** for
  opencode's chunkTimeout watchdog, and are invisible to the JSON layer. Pie should emit
  them during long prefill/decode gaps.

**Defaults for this provider path (custom provider, `@ai-sdk/openai-compatible`):**

| Timeout | Default | Notes |
|---|---|---|
| `headerTimeout` | **none** | the 300 s default applies only to `providerID === "openai"` (`provider.ts:35,208`); configurable per provider in `opencode.json` `options.headerTimeout` |
| `chunkTimeout` | **none** — wrapSSE not even installed unless `options.chunkTimeout` is a positive number (`provider.ts:1746,1766`) | when set, per-raw-read as above |
| total `timeout` | **none**; Bun's own fetch timeout explicitly disabled (`timeout:false`, `provider.ts:1763`) | `options.timeout` if configured |

So stock opencode with this config waits indefinitely; keepalive matters when users set
`chunkTimeout`, and never hurts. When a chunkTimeout fires, the error message
`"SSE read timed out"` matches the retryable pattern `/\bread (?:timeout|timed out)\b/i`
(`session/retry.ts:36`), so the whole request is **replayed** by the session retry loop (§4).

---

## 3. Tool-name repair — what server-emitted tool_call names must look like *(source)*

`experimental_repairToolCall` (`packages/opencode/src/session/llm.ts:296-312`), invoked by
the AI SDK when a streamed tool call fails validation (unknown name or schema-invalid args):

1. If `toolName.toLowerCase()` differs from `toolName` **and** the lowercase name exists →
   the call is renamed (so `Read`/`READ` → `read` is silently fixed).
2. Anything else (unknown name, or valid name with schema-invalid arguments) is rewritten
   to the hidden `invalid` tool with input `{"tool": <name>, "error": <message>}`; that tool
   returns an error string to the model — a wasted round-trip, not a crash.

**Requirement on pie:** tool-call names must byte-match a name from the request's `tools`
array (case-normalization is the only forgiveness), and `arguments` must validate against
the advertised JSON Schema, or the turn degrades into an `invalid`-tool bounce.

Streaming-shape requirement (SDK dist `index.js:768-784`): the **first delta for a
tool_call index must carry both `id` and `function.name`**, else
`InvalidResponseDataError` kills the stream. `arguments` may be split across deltas;
`index` is required to correlate them. Emitting the whole call in one delta (as our
recorder does after the name delta) is fine.

## 4. finish_reason / empty-content handling, retries *(source; SSE happy path capture-verified)*

- finish_reason mapping (SDK): `stop|length|content_filter|tool_calls|function_call` →
  mapped; anything else/missing → `"unknown"`. **No client-side error for a missing
  finish_reason** (unlike qwen-code's `NO_FINISH_REASON` retry loop).
- **No empty-content retry loop.** `streamText` runs with `maxRetries: 0`
  (`llm.ts:323`); the session-level `Effect.retry` (`processor.ts:660`,
  `retry.ts:77-149`) fires **only on errors** whose status/message matches the retryable
  patterns (429/5xx, rate-limit, network/timeout strings). An empty-but-well-formed stream
  simply produces an empty assistant turn and the loop ends.
- Retry policy when it does fire: **unbounded attempts**, delay `2000·2^(n−1)` ms capped at
  30 s (no headers) or `retry-after`/`retry-after-ms` honored up to 2^31−1 ms. 400s whose
  bodies don't match the patterns fail fast; a 500 is retried **forever** — pie must not
  return 5xx for malformed requests.
- In-stream error objects: a `data: {"error": {...}}` event is surfaced as a stream error
  part; whether it retries again depends on the same message patterns.
- `data: [DONE]` is consumed by the parser; stream close without it also terminates cleanly
  (finish is emitted in `flush`), but send it anyway.
- Usage-only chunk with `"choices": []` is accepted (capture-verified).

## 5. Usage / cached tokens (capture + SDK source)

SDK dist `index.js:86-103`:

```
inputTokens.total     = usage.prompt_tokens
inputTokens.cacheRead = usage.prompt_tokens_details.cached_tokens   (else 0)
inputTokens.noCache   = prompt_tokens − cached_tokens               (computed client-side)
outputTokens.reasoning= usage.completion_tokens_details.reasoning_tokens
```

So pie should report cache hits via **`prompt_tokens_details.cached_tokens`** on the final
usage chunk (with `include_usage` semantics: last chunk, empty `choices`). `cached_tokens`
must be ≤ `prompt_tokens` or `noCache` goes negative. Cache-write tokens have no wire slot.
Capture req-004's response included `cached_tokens: 64` and the turn completed normally.
Missing usage entirely is survivable (all fields become `undefined`).

## 6. Things a strict server might 400 on (must-tolerate list)

| Item | Where seen |
|---|---|
| `stream_options: {"include_usage": true}` | every request |
| `max_tokens` (deprecated on OpenAI proper) always present | every request |
| `tool_choice: "auto"` | tool requests |
| `"$schema"` + `maximum: 9007199254740991` inside tool `parameters` | req-004 |
| assistant message with `content: ""` alongside `tool_calls` | req-005 |
| requests with **no** `tools` key at all (title call) on the same model | req-001/003 |
| user content as JSON-stringified string (leading/trailing quotes are semantic) | req-001 |
| non-standard headers `x-session-affinity`, `x-session-id`, `x-parent-session-id` | every request |
| `Accept: */*` (do NOT require `Accept: text/event-stream` for SSE) | every request |
| `Accept-Encoding: gzip, …` — responding identity-encoded is fine, but don't 406 | every request |
| HTTP keep-alive connection reuse across requests | observed |

## 7. Capture inventory

| File | What it is |
|---|---|
| `wire/req-001.json` | title-generation side-call, run 1 (no tools, 3 messages) |
| `wire/req-002.json` | simple text turn: system+user, 10 tools, `tool_choice:"auto"` |
| `wire/req-003.json` | title side-call, run 2 |
| `wire/req-004.json` | first agent request of tool run (server answered with `read` tool call) |
| `wire/req-005.json` | **follow-up with tool history**: system, user, assistant+`tool_calls`, `role:"tool"` result |

Response side (what the recorder served and opencode accepted): SSE with a leading
`: keepalive` comment, `{"delta":{"role":"assistant"}}`, content / tool_calls deltas,
`finish_reason` chunk, usage-only chunk (`choices: []`,
`prompt_tokens_details.cached_tokens`), `data: [DONE]`. See `record_server.py`.
