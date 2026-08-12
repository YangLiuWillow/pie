# Codex ↔ Pie Integration

**Branch:** `liu/codex-integration`
**Goal:** run OpenAI's Codex CLI against a Pie-served open-weight model, so Pie's
programmable KV-cache primitives are exercised by a real, unmodified coding agent.

This document is the design contract. It is grounded in reads of
`../codex` (commit `bd5b55e403`) and this repo as of `ea36d778`.

---

## 1. The seam: Codex only speaks the Responses API

Codex talks to models through exactly one wire protocol. `wire_api = "chat"` was
removed — the enum has a single variant and deserializing `"chat"` is a hard error:

- `codex-rs/model-provider-info/src/lib.rs:57` — `enum WireApi { Responses }`
- `codex-rs/model-provider-info/src/lib.rs:50` — `CHAT_WIRE_API_REMOVED_ERROR`

So there is no Chat-Completions shim path. **Pie must serve
`POST {base_url}/responses` with SSE.** Everything else (the TUI, sandbox,
apply-patch, rollout, MCP) is untouched — we swap only the provider.

Codex is pointed at us with a config block, no code changes:

```toml
# ~/.codex/config.toml
model = "qwen3-coder-30b"          # arbitrary slug; unknown slugs get sane fallback metadata
model_provider = "pie"

[model_providers.pie]
name = "pie"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
```

Unknown model slugs are fine: `models-manager/src/model_info.rs:123`
(`model_info_from_slug`) supplies fallback metadata with
`supported_in_api: true`, `apply_patch_tool_type: None`,
`use_responses_lite: false`, and `ConfigShellToolType::Default`. That last set of
defaults matters — it means tools arrive in the **top-level `tools` field**
(`core/src/client.rs:855-880`) rather than smuggled in as an `AdditionalTools`
input item, and `shell` arrives as an ordinary function tool.

### 1.1 Exactly what Codex sends

`ResponsesApiRequest` — `codex-rs/codex-api/src/common.rs:252`:

| field | value from Codex | what Pie must do |
|---|---|---|
| `model` | config slug | echo back |
| `instructions` | base instructions (large system prompt) | → system turn |
| `input` | `Vec<ResponseItem>` — full conversation each turn | replay into `Context` |
| `tools` | `Vec<ToolSpec>` | → `tools::equip_prefix` schemas |
| `tool_choice` | always `"auto"` (`client.rs:912`) | ignore |
| `parallel_tool_calls` | bool | may emit >1 `function_call` item |
| `reasoning` | `Some(Reasoning{..})` | ignore for non-reasoning models |
| `store` | `false` for us (`client.rs:915`) | no server-side state required |
| `stream` | always `true` (`client.rs:916`) | SSE mandatory |
| `include` | `["reasoning.encrypted_content"]` | ignore |
| `prompt_cache_key` | **session id** (`client.rs:483-487`) | **KV session key — see §3** |
| `client_metadata`, `text`, `service_tier` | present | ignore |

`ResponseItem` variants that actually appear in `input`
(`codex-rs/protocol/src/models.rs:799`): `Message`, `Reasoning`, `FunctionCall`,
`FunctionCallOutput`, `LocalShellCall`, `ToolSearchCall`, `AdditionalTools`,
`AgentMessage`. Phase 1 must handle `Message` / `FunctionCall` /
`FunctionCallOutput` and tolerate-and-skip the rest.

Tool shapes (`codex-rs/tools/src/tool_spec.rs:19`) are tagged by `type`:
`function`, `custom` (freeform, `{format:{type,syntax,definition}}`), `namespace`,
`tool_search`, `web_search`. A `function` tool is
`{type,name,description,strict,parameters}` — note `parameters` is **not** nested
under a `function` key, unlike Chat Completions.

### 1.2 Exactly what Codex requires back

`process_responses_event` — `codex-rs/codex-api/src/sse/responses.rs:327`. Only
these `type`s are consumed; everything else is trace-logged and dropped:

**Required for a working turn**
- `response.created` — must carry a `response` object (any shape)
- `response.output_item.done` — `item` must deserialize as a `ResponseItem`;
  this is how both assistant text and **function calls** are delivered
- `response.completed` — `response` must parse as `ResponseCompleted`
  (`sse/responses.rs:114`): **`id` is mandatory**, `usage` and `end_turn` optional

**Optional / nice**
- `response.output_text.delta` — live TUI text
- `response.output_item.added`, `response.reasoning_*`, `response.custom_tool_call_input.delta`

**Error channel**
- `response.failed` with `response.error.code` — Codex maps `context_length_exceeded`,
  `rate_limit_exceeded`, `insufficient_quota`, `invalid_prompt` etc. to typed errors
  (`sse/responses.rs:387-422`); an unrecognized code becomes a **retryable** error,
  so misreporting a fatal error makes Codex retry 4× before failing.
- `response.incomplete` → hard stop.

The usage struct is the interesting one:

```
usage.input_tokens
usage.input_tokens_details.cached_tokens      # ← Pie prefix-reuse lands here
usage.input_tokens_details.cache_write_tokens
usage.output_tokens
usage.output_tokens_details.reasoning_tokens
usage.total_tokens
```

`sse/responses.rs:123-147`. Reporting `cached_tokens` honestly means **Codex's own
TUI and rollout logs become our cache-hit telemetry** — no instrumentation fork needed.

`GET /models` is also worth serving (`model-provider/src/models_endpoint.rs:39`);
it is used by `codex doctor` and model listing, not by the turn loop.

---

## 2. What exists on the Pie side, and the gap

### 2.1 `inferlets/openresponses` — right shape, demo-grade content

It already serves `POST /responses` and `POST /v1/responses`
(`src/lib.rs:47`) and emits a plausible SSE sequence (`src/streaming.rs`). But
against Codex it fails immediately:

| gap | where | consequence |
|---|---|---|
| `Tool` enum only accepts `type:"function"` | `src/types.rs:51` | a `custom`/`web_search` tool ⇒ whole request 400s |
| tools parsed but **never used** | `src/handler.rs` | model is never told the tools exist |
| no `function_call` output ever emitted | `handler.rs:164-278` | Codex can never act; agent loop is dead |
| assistant messages dropped | `handler.rs:63` | multi-turn history lost |
| `FunctionCall` input items dropped | `handler.rs:68` | tool-call history lost |
| `FunctionCallOutput` flattened to `"Function result: …"` user msg | `handler.rs:72` | wrong chat template framing |
| only the *last* system/developer msg survives | `handler.rs:57` | `instructions` clobbered |
| `usage: None` | `handler.rs:392` | no cache telemetry |
| no `prompt_cache_key` handling | — | full prefill every turn; Pie's whole value unexercised |

Verdict: **fork it, don't extend it.** Keep the HTTP/SSE shell, replace the body.

### 2.2 `inferlets/openclaw-chat` — the generation core we want

`inferlets/openclaw-chat/src/lib.rs` (467 lines) already does, correctly, the three
hard things:

- **history replay with native tool framing** — `replay_history` / `replay_incremental`
- **streaming native tool-call decode** — via `inferlet::tools::Decoder`
- **KV session pinning** — `Context::open(&model, sid)` on resume,
  `ctx.save(sid)` after generation, with transparent fallback to full replay when
  the snapshot was evicted (`lib.rs:171-200`, `lib.rs:310-315`)

It just speaks a bespoke WebSocket/msgpack protocol instead of HTTP Responses.

### 2.3 The SDK primitives (inherited from `openhands-integration`)

`sdk/rust/inferlet/src/tools.rs` gives us everything needed to drive a native
tool-calling model without hand-rolling templates:

- `equip_after_system_prefix(model, system, schemas)` — folds `instructions` and the
  tool schemas into one system turn (matches Qwen's template)
- `assistant_with_tool_calls_prefix(model, content, calls)` — replay a past
  assistant turn that called tools
- `answer_batch_prefix(model, results)` — replay consecutive tool results as **one**
  merged turn, which is what chat templates expect
- `native_grammar` / `native_matcher` — constrain decoding to well-formed tool calls
- `Decoder` — streaming detector emitting `Event::Call(name, args_json)`

These map 1:1 onto Codex's `ResponseItem` variants. That correspondence is the
core reason this integration is tractable.

### 2.4 Serving mechanics

`pie http` in the openresponses doc comment **does not exist** in the current CLI
(`server/src/cli.rs:50`). HTTP inferlets are served by the daemon path:
`runtime/src/daemon.rs` binds a port and invokes
`wasi:http/incoming-handler@0.2.4`, spawned via the `launch_daemon` RPC
(`runtime/src/server/handler.rs:218`). Startup is therefore: `pie serve` → client
issues `launch_daemon(port, inferlet)`. Confirm the exact client incantation
before writing the runbook; `integrations/openclaw/tests/test_e2e_smoke.mjs` is the
working precedent for booting a real `pie serve` in a test.

---

## 3. Design

### 3.1 New inferlet: `inferlets/codex-responses`

```
Codex CLI (unmodified)                    Pie
┌────────────────────┐                   ┌────────────────────────────┐
│ ModelClient        │  POST /v1/responses│  codex-responses inferlet  │
│  wire_api=responses│──── JSON ─────────▶│                            │
│                    │                    │  1 parse ResponsesApiReq   │
│  consumes:         │◀─── SSE ──────────│  2 key = prompt_cache_key   │
│   output_item.done │                    │  3 Context::open|new       │
│   output_text.delta│                    │  4 replay ResponseItems    │
│   completed{usage} │                    │  5 equip tools + grammar   │
└────────────────────┘                    │  6 stream decode           │
                                          │  7 ctx.save(key)           │
                                          └────────────────────────────┘
```

**Item ⇄ Pie mapping**

| Codex `ResponseItem` (input) | Pie replay call |
|---|---|
| `Message{role:"system"\|"developer"}` + `instructions` | `equip_after_system_prefix` (one turn, with tool schemas) |
| `Message{role:"user"}` | `ctx.user(text)` |
| `Message{role:"assistant"}` (no calls) | `ctx.assistant(text)` |
| `FunctionCall{name,arguments,call_id}` | `assistant_with_tool_calls_prefix` — **coalesce runs** of consecutive calls |
| `FunctionCallOutput{call_id,output}` | `answer_batch_prefix` — **coalesce runs** |
| `Reasoning`, `LocalShellCall`, `ToolSearchCall`, `AdditionalTools` | skip in phase 1 (log once) |

Coalescing matters: Codex emits one item per call/result, but chat templates group
consecutive ones into a single turn. `answer_batch_prefix` exists precisely for this.

**Output**

- text tokens → `response.output_text.delta`, accumulated into an
  `OutputItem::Message` emitted as `response.output_item.done`
- each `Decoder::Event::Call(name, args)` → an
  `OutputItem::FunctionCall{id, call_id, name, arguments, status:"completed"}`
  emitted as its own `response.output_item.done`. `call_id` must be unique and is
  what Codex echoes back in the next turn's `FunctionCallOutput`.
- `response.completed` with `id` + real `usage`

**Correctness traps**
- `parameters` sits at the tool's top level, not under `function` — the
  openclaw-chat schema builder (`lib.rs:160-169`) reads `t.function.parameters`
  and must be adapted.
- Unknown tool `type`s must deserialize, not 400. Use `#[serde(other)]`-style
  tolerance or `Value` + manual dispatch.
- Report `stop`/`length` honestly: hitting `max_output_tokens` mid-tool-call
  yields malformed arguments; prefer `response.incomplete` over emitting garbage.
- Only `context_length_exceeded` should be reported as such — anything else
  triggers Codex's 4× retry.
- **Re-saving a session key: use `take`, never repeated `save`.** The host
  `save` **bails if `(username, name)` already exists**
  (`runtime/src/context/snapshot.rs:179`). So you cannot re-`save` under the same
  `prompt_cache_key` each turn. openclaw-chat does `let _ = ctx.save(sid)` and
  swallows the error (`inferlets/openclaw-chat/src/lib.rs:314`), so its snapshot
  silently ossifies at turn 1 — do **not** copy that. Correct evolving-session
  pattern is `Context::take(key)` → append the new turn → `save(key)`: `take`
  consumes the old snapshot (removes the entry, hands back an owned mutable
  context, `snapshot.rs:279`), which frees the name for the re-save. `take`
  returns `Err("Snapshot not found")` on cold start or after eviction → fall back
  to `Context::new` + full replay.

### 3.2 The Pie value proposition, made measurable

Codex hands us `prompt_cache_key` = session id (`client.rs:483`) on **every**
request, and re-sends the entire conversation each turn. That is the ideal setup
for KV pinning:

- turn *n* prefix ⊃ turn *n−1* prefix, exactly
- the key to pin under arrives in-band, no protocol extension needed
- the cache-hit number has a first-class home in the response (`cached_tokens`)

Phase 2 keeps the context resident keyed by `prompt_cache_key` and appends only
the delta. The comparison story writes itself: **Codex + Pie vs. Codex + vLLM
(automatic prefix caching)**, on the same model, same tasks, TTFT and wall-clock.
As with the OpenHands work, a clean negative result is still a result.

### 3.3 Where evaluation goes

`../test-time-bench` already has `benchmarks/swe-bench-lite-first-20` and
`swe-bench-lite-full-300`, and a frozen per-case serving contract
(`docs/per-case-inferlet-request-contract.md`) whose stated target model is "one
submitted inferlet installed in a continuously running Pie engine, one request per
benchmark case" — with agentic drivers explicitly anticipated above that boundary.
Codex slots in as such a driver: the case request supplies the repo/task, the
driver shells out to `codex exec`, and the inferlet is `codex-responses`.
Identity stays `(run_id, attempt, case_id, sample_index)`; cache isolation comes
free if we derive `prompt_cache_key` scoping per case.

### 3.4 Serving runbook (verified against source)

There is **no `pie http` subcommand** — the openresponses doc comment is stale
(`server/src/cli.rs:50` has only serve/run/config/auth/model/…). An HTTP inferlet
is served by the **daemon** path, over two distinct ports:

1. `pie serve --config <cfg> --port <CONTROL_PORT> --no-auth` — boots the
   control-plane **WebSocket** on `CONTROL_PORT`.
2. A client connects to `ws://127.0.0.1:<CONTROL_PORT>`, uploads the local wasm by
   hash with `install_program(wasm, manifest)` (chunked `add_program`,
   `client/python/src/pie_client/client.py:416`), then calls
   `launch_daemon(inferlet_name, HTTP_PORT, input)`
   (`client.py:565`; also `client/javascript/src/index.js:597`). This binds a
   **separate HTTP port** (`HTTP_PORT`) that serves
   `wasi:http/incoming-handler@0.2.4` (`runtime/src/server/handler.rs:218` →
   `runtime/src/daemon.rs:44`). `launch_daemon` installs the program itself first,
   so pre-`install_program` is only needed to get a local (non-registry) wasm onto
   the server.
3. Point Codex at the daemon: `base_url = "http://127.0.0.1:<HTTP_PORT>/v1"`.

`integrations/openclaw/tests/test_e2e_smoke.mjs` is the working precedent for
booting a real `pie serve` and installing an inferlet in a test (it launches a
*process*, not a daemon — the daemon call is `launchDaemon` instead, otherwise the
setup is identical). A phase-1 smoke harness for `codex-responses` should follow
its structure but drive an actual `codex exec` against the daemon port.

### 3.5 KV snapshots survive across HTTP requests (verified)

The daemon instantiates a **fresh WASM store + component instance per HTTP
request** (`daemon.rs:194,216-219`), so nothing in WASM linear memory persists
between Codex turns. KV persistence instead rides on the **host-side** snapshot
store, and it does survive, because:

- `Context::save/open/take` are host calls keyed by `(model_id, username, name)`
  into the per-model `ContextManager.snapshots: HashMap<(username,name),ContextId>`
  (`runtime/src/api/context.rs:41-146`, `runtime/src/context.rs:768`). That map
  lives in the long-running `pie serve` process, not the instance.
- Every per-request instance is created with the **daemon's `username`**
  (`daemon.rs:219` passes `username` into `linker::instantiate`), so request *N+1*
  sees the snapshot request *N* saved.
- A named snapshot is a *separate* context with `owner: None`
  (`snapshot.rs:217`) and its own refcount on the committed pages (`save` calls
  `gpu_stores.fork(&committed_hashes)`, `snapshot.rs:193`). On instance drop,
  `unregister_process` destroys only the process's **owned** contexts and retains
  snapshots that don't point at them (`sched.rs:207-243`). The snapshot outlives
  the instance that created it.

Bounds worth stating plainly:

- **Lifetime = the `pie serve` process.** Snapshots are in-memory; a server
  restart clears them. A Codex session lives well inside one server lifetime, so
  this is fine for phase 2.
- **Evictable.** Snapshots carry `bid: 0.0` (`snapshot.rs:228`), so under memory
  pressure they can be reclaimed; `open`/`take` then return `Err`. The inferlet
  must treat that as cold-start and full-replay — same fallback openclaw-chat
  already implements.

Net: **phase 2 is viable as designed.** The KV pin lives host-side keyed by
`prompt_cache_key`, survives the per-request instance churn, and degrades safely to
full replay on eviction or server restart.

---

## 4. Phases

**Phase 1 — one turn, end to end.** `codex-responses` inferlet: parse the real
request, replay history, equip tools, stream one assistant message + function
calls, emit `response.completed` with usage. Success = `codex exec "list files
then read README"` completes a shell→result→answer loop on CPU with a small model.
No caching, no perf claims.

**Phase 2 — KV pinning on `prompt_cache_key`.** Resident context per session,
delta-append, honest `cached_tokens`. Success = ≥20% wall-clock improvement on a
multi-turn task vs. phase 1, and `cached_tokens` in Codex's own telemetry matching
Pie's accounting.

**Phase 3 — benchmark + writeup.** Codex driver in test-time-bench, SWE-Bench
Lite, Codex+Pie vs. Codex+vLLM-APC.

Development is CPU-only until phase 3 needs GPUs (same constraint as the
OpenHands work).

## 5. Open questions

**Resolved (see §3.4, §3.5):**
- ~~Exact `launch_daemon` invocation for a `wasi:http` inferlet.~~ → §3.4:
  `pie serve` (WS control port) + `install_program` + `launch_daemon(name, http_port)`
  (separate HTTP port). No `pie http`.
- ~~Does `Context::save`/`open` survive across daemon HTTP requests?~~ → §3.5:
  **yes**, host-side and keyed by `(model_id, username, name)`; survives the
  per-request instance churn for the `pie serve` lifetime, evictable with safe
  full-replay fallback. Note the `take`-not-`save` requirement (§3.1 traps).

**Open:**
1. Can an open-weight coding model actually drive Codex's tool surface? Codex's
   base instructions are long and tuned for GPT-5-class models. Measure the
   malformed-tool-call rate early; `native_grammar` constraining is the mitigation.
2. `apply_patch` is a `custom` (freeform, non-JSON) tool for models that declare it.
   Fallback metadata leaves it off, so Codex uses shell-based patching — cheaper for
   us, but worth confirming behaviourally.
3. Does the dummy-driver CPU path exercise `save`/`take`/`open` faithfully enough
   for a phase-1 smoke test, or does snapshot page-copy (`snapshot.rs:196-213`)
   need a real device? Confirm before relying on the openclaw-style e2e harness.
