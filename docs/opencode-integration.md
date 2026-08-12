# opencode ↔ Pie integration — two strategies

**Date:** 2026-08-11. Target: pie `dev` (`58cb77936`), opencode `1.18.16`
(`~/Documents/Liszt_ai/opencode`). Companions: `qwen-code-integration-plan.md`
(the validated Option-A precedent), `codex-integration.md` (Responses-API precedent),
and the SOSP'25 paper (`~/Desktop/pie.pdf`) for the performance model.

**Purpose.** Plan two integration strategies:

- **Strategy A — serve the completion endpoint.** Stock opencode, pointed at a
  Pie-served OpenAI-compatible `/v1/chat/completions`. Pie is a drop-in serving
  system; KV reuse is content-addressed and best-effort.
- **Strategy B — harness-adjacent session inferlet.** Move the *context-management
  layer* of opencode's harness server-side as a long-lived inferlet, so pie's
  programmable KV cache (the paper's R1) is exercised directly: persistent
  working sets, delta-only wire traffic, in-place context editing, KV forking for
  subagents, decode/tool-execution overlap.

They are phased, not competing: A is the baseline and fallback path B degrades to.

---

## 0. Ground truth (what exists today)

### 0.1 Pie `dev` is a rewrite; the old serving assets are off-branch

- `git ls-files inferlets` on `dev` → empty. `chat-completions`, `openresponses`,
  `codex-responses`, `openhands-completion` exist only on `liu/codex-integration`,
  `liu/rl-completions`, `openhands-integration-updated` — all built against the
  pre-rewrite runtime (`runtime/src/daemon.rs`, `pie:core/inference`,
  `Context::save/open/take`), none of which exist on `dev`.
- The inbound-HTTP-inferlet mechanism (`launch_daemon(port)`) is gone:
  `interface/inferlet/world.wit:26` imports only `wasi:http/client` (outbound).
- The `dev` gateway exposes exactly three routes (`gateway/src/ingress/mod.rs:24-29`,
  `gateway/src/blob.rs:143-145`): `POST /v1/generate` (one-shot, SSE,
  `Affinity::Ephemeral`), `GET /v1/ws` (multi-turn WebSocket, `Affinity::Sticky`,
  HRW routing on session id), `GET /blob/{hash}`. Payloads are
  `ClientMessage`/`ServerMessage` (`interface/client/src/message.rs:21-122`), not
  OpenAI. No `Authorization` handling — identity is a trusted edge header
  `x-pie-identity` (`gateway/src/ingress/identity.rs:26-54`).
- KV persistence across requests on `dev` = `working-set.wit`:
  `update-index(key)` / `from-index(key)` / `remove-index(key)` — named,
  best-effort, pressure-evictable snapshots. This replaces `Context::save/open/take`.
- Long-lived processes exist: `launch_process` + `signal_process` →
  `session::receive()` (`runtime/engine/src/server/handler.rs:543-565`) lets a
  client message a running inferlet mid-run. **Strategy B needs zero gateway
  changes because of this.**
- Chat templating is host-side Rust (`model/src/instruct.rs:20-105`). The `dev`
  `Instruct` trait (`model/common/src/instruct.rs:73-98`) has
  `equip`/`answer`/`cue`/decoders but **no `assistant_with_tool_calls` and no
  `answer_batch`** — the two primitives the old integrations' history replay was
  built on. Restoring them host-side is shared prerequisite work (§3).
- One model per engine (`interface/inferlet/model.wit:3-4`): the OpenAI `model`
  field is echoed, not routed on.

### 0.2 opencode's seams (from the provider-layer survey)

- **Zero-code custom provider**: an `opencode.json` `provider` block with
  `npm: "@ai-sdk/openai-compatible"` + `options.baseURL` gets a Chat-Completions
  SSE client, `include_usage` forced on
  (`packages/opencode/src/provider/provider.ts:1439-1444, 1694-1696`). This is
  exactly how llama.cpp / LM Studio / Ollama are documented
  (`packages/web/src/content/docs/providers.mdx:1351-1692, 2431-2560`).
- **Custom npm provider, no fork**: `npm: "file:///…"` dynamically loads a local
  AI SDK provider package exporting `create*()` → `LanguageModelV3`
  (`provider/provider.ts:1770-1800`). This is Strategy B's client-side seam.
- **V2 native stack** (`packages/llm`, Effect-based, no AI SDK on the wire) is the
  long-term first-class seam: a new protocol is `Protocol.make` + `Route.make`
  (`packages/llm/src/route/client.ts:182-215`) plus one branch in
  `packages/core/src/session/runner/model.ts:131-171`. Gated today behind
  `OPENCODE_EXPERIMENTAL_NATIVE_LLM`.
- The agent loop calls `streamText` exactly once per turn
  (`packages/opencode/src/session/llm.ts:280`); tools are AI SDK `tool()` values;
  full message history is re-sent every turn (stateless chat contract).
- Relevant client behaviors to respect: ID-substring sampling defaults
  (qwen → temp 0.55; `provider/transform.ts:526-573`),
  `maxOutputTokens = min(limit.output, 32_000)` (`transform.ts:18`), inter-chunk
  watchdog `chunkTimeout` + `headerTimeout` (default 300 s,
  `provider/provider.ts:35-83`), tool-name repair middleware
  (`session/llm.ts:286-292`), reasoning streamed from `reasoning_content` for
  openai-compatible (`provider/provider.ts:1485-1487`), client-side compaction
  and tool-output truncation (analogs of qwen-code hazards H1/H2).

---

## 1. Strategy A — OpenAI-compatible endpoint (stock opencode)

**Shape.** Keep opencode 100% stock. Build on pie `dev`:

1. **`gateway/src/ingress/openai.rs`** — new ingress module mounted in
   `ingress/mod.rs`: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.
   - Map `Authorization: Bearer <key>` → `Identity{tenant,user}` (key table in
     gateway config; keep `x-pie-identity` for the trusted-edge path).
   - Translate the OpenAI body into
     `LaunchProcess{inferlet:"chat-completions", input:<raw OpenAI JSON>, capture_outputs:true}`
     and drive it through the existing `Sessions::create` path exactly as
     `http.rs:51-72` does. Affinity: `Sticky` keyed on a hash of
     (identity, system prompt, tools) so consecutive turns of one agent land on
     the warm worker — otherwise HRW/ephemeral spread defeats snapshot reuse.
   - Framing decision: the **inferlet emits ready-made `chat.completion.chunk`
     JSON strings** via `session::send()`; the ingress only wraps them in
     `data:` lines, injects `: ping` keepalives, and closes with `[DONE]` on
     `ProcessEvent::Return`. Keeps all OpenAI wire logic testable in one crate
     and the gateway thin.
2. **`inferlets/chat-completions` ported to `dev`** — re-implement the validated
   qwen-code design (`qwen-code-integration-plan.md` §2: file layout, 9 response
   invariants, content-addressed sessions) on the new API surface:
   - render: `chat.wit` + `tools.wit` (+ the restored history-replay primitives, §3);
   - generate: the `tests/inferlets/chat-completion/src/lib.rs` PTIR core
     (prefill fire + device-carried decode loop) extended with tool-call decoding
     (`tools.decoder`) and the salvage parsers from the qwen-code work;
   - sessions: port `codex-responses/src/session.rs` (canonicalize → FNV-1a-64 ×2
     → named snapshot) onto `working-set update-index/from-index`; resume = strip
     trailing tool/user suffix, `from-index` on hit, rebuild on miss; report hit
     depth as `prompt_tokens_details.cached_tokens`.
   - All nine wire invariants from the qwen-code plan carry over verbatim
     (finish_reason discipline, non-empty content, globally-unique tool-call ids
     from `system::instance-id()`, 400-vs-500, never a context-length 400,
     keepalives, usage chunk).
3. **opencode config** (`integrations/opencode/opencode.json`):

   ```json
   {
     "provider": {
       "pie": {
         "npm": "@ai-sdk/openai-compatible",
         "name": "Pie (local)",
         "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "{env:PIE_API_KEY}" },
         "models": { "qwen3-coder-30b": { "limit": { "context": 262144, "output": 32768 } } }
       }
     },
     "model": "pie/qwen3-coder-30b"
   }
   ```

   Launch profile mirroring the qwen-code §4 discipline: pin the tool list,
   disable client compaction (huge `contextWindowSize`), fixed workspace — so
   snapshot addresses stay stable and reuse is measurable.
4. **opencode wire audit + fixtures** — repeat the qwen-code §1 audit for
   opencode: capture real request bodies (logging proxy in front of the daemon),
   check specifically: does the AI SDK's SSE parser treat `: ping` comments as
   liveness for `chunkTimeout` (if not, emit empty-delta chunks instead); the
   tool-repair middleware's lowercase rule (emit lowercase tool names); cache
   marker fields sent when `setCacheKey`/anthropic-style marks leak into options
   (must be ignored, never 400). Bank captures next to
   `tests/inferlets/fixtures/rl_completions/` (real qwen-code captures already
   there are the bootstrap fixtures — same wire dialect).

**What A buys / costs.** Proven shape (33/33 acceptance, 99.7% local KV reuse,
81.5% on H200 for qwen-code). Zero opencode changes; works for every other
OpenAI-speaking client for free. But it inherits the stateless contract's tax:
full-history re-upload + re-render + hash every turn; reuse is fragile against
any byte divergence (the response/save unification lesson) and snapshots are
pressure-evictable. The H200 A/B says the honest ceiling: at short contexts pie
loses to vLLM batch-1 decode; the win regime is long contexts and multi-tenant.

**Effort.** Gateway ingress ~small (the session plumbing already exists). The
inferlet port is the bulk — the old code targets the pre-rewrite API, so this is
a re-implementation guided by a validated spec, not a rebase. Plus §3 prereqs.

---

## 2. Strategy B — harness-adjacent session inferlet (programmable-KV-native)

> **2026-08-12 — what the SOSP'25 paper says, read after Phase A shipped.**
> Strategy B is not merely the more interesting option; it is the shape the
> system was designed around, and Strategy A runs against its grain. Worth
> stating precisely, because it reframes several Phase A defects as symptoms
> rather than isolated bugs.
>
> - **Agentic workflows are the motivating case, not an application of it.**
>   Of the three requirements the paper derives for next-generation serving,
>   **R3** is *"Agentic workflows and interactions with external systems
>   necessitate tightly coupling token generation with arbitrary computations
>   and I/O **within the generation flow**, without … complex external
>   orchestration."* The headline result is ours: *"1.1×–2.4× lower latency,
>   1.3×–3.4× higher throughput"* on agentic workflows, against 3–12% latency
>   overhead on plain text completion. Pie is *supposed* to be good at this.
>
> - **The unit of service is a long-lived program, not a request.** *"Each
>   inferlet executes within a single-threaded, event-driven runtime.
>   Concurrency within an inferlet is handled through asynchronous,
>   non-blocking API calls, a model well-suited for I/O-bound agentic
>   workflows."* The ILM launches an inferlet and *"users can communicate with
>   inferlets through the ILM after launch"* via `send`/`receive`. Upstream's
>   own `text-completion-bench` is built exactly this way: a `prompts` array
>   plus `batch_concurrency`, N generations concurrent **inside one process**.
>
> - **Our Strategy A is the opposite shape by necessity.** Stock opencode
>   speaks OpenAI over HTTP, so `chat-completions` is one inferlet per request
>   and N concurrent turns are N concurrent *inferlets*. That axis is
>   unbenchmarked upstream, and it is where we found the N≥2 defect.
>
> - **The contention policy is tuned for independent tenants.** *"To handle
>   resource contention, the control layer uses a First Come First Serve
>   (FCFS) policy, terminating the most recently created inferlets until
>   sufficient resources are freed."* Reasonable when concurrent inferlets are
>   separate tenants; exactly inverted when they are N turns of ONE user's
>   agent session, because the newest request — the one the user is waiting on
>   — is the first killed. **Not confirmed as our mechanism** (our failure is
>   a synchronous `PIE_STATUS_INVALID_ARGUMENT` descriptor rejection, not a
>   termination), but it is the designed behaviour in the neighbourhood and
>   should be ruled in or out before the concurrency work is called done.
>
> **Consequence for phasing.** The 2026-08-12 measurement — ~70% of a
> two-tool-call opencode task is re-prefill of a history that changed by a few
> hundred tokens, `~81 s → ~21 s` with resume — is the same conclusion the
> paper argues from first principles. Strategy A remains the right
> compatibility path and is now green; but the performance argument for this
> project lives here, and the concurrency defect is a reason to reach it
> sooner rather than a reason to keep hardening the shim.

**Thesis.** Strategy A treats pie as a vLLM stand-in and reconstructs continuity
by hashing. Strategy B changes the *contract*: one long-lived
**`opencode-session` inferlet per opencode session**, attached over the existing
sticky WebSocket, owning the conversation's KV working set for its whole life.
The client sends **deltas**, not history. This is the paper's Fig-5-right shape
applied to a real coding agent, with the tool loop still client-side (attribution
lesson from the deprecated `openhands-agent` Pattern A: don't rewrite the
harness, colocate its *context layer*).

**Wire.** No gateway changes: `GET /v1/ws` + `launch_process` (session start) +
`signal_process` (each turn) + `session::send` (streamed events) is sufficient
(`interface/client/src/message.rs:59,73,106`). Define a small JSON dialect:

```jsonc
// client → inferlet (signal_process payloads)
{ "turn": { "append": [ {"role":"user"|"tool", ...} ],   // delta only
            "edits":  [ {"truncate_tool_output": {"call_id":"…","keep":2048}} ],
            "gen":    { "max_tokens": 32000, "temperature": 0.55 } } }
{ "fork": { "branch_id": "sub1", "append": [ …subagent prompt… ] } }
// inferlet → client (session::send payloads)
{ "delta": "…" } | { "reasoning": "…" } | { "tool_call": {"id","name","arguments"} }
| { "finish": {"reason":"stop|length|tool_calls", "usage": {…}} }
```

**Pie-side.** `inferlets/opencode-session/`: same renderer/decoder/PTIR core as
Strategy A (shared crate), but state lives in the process: one
`kv-working-set` per branch, extended in place each turn. Checkpoint via
`update-index` at every turn end so a dropped WS or evicted worker degrades to a
Strategy-A-style rebuild instead of data loss.

**Client-side.** Phase B1: an AI SDK provider package
(`sdks/pie-ai-sdk-provider`, loaded via `npm: "file://…"` — no opencode fork).
It presents `LanguageModelV3`; internally it keeps a shadow of what the server
holds, diffs each incoming full-history prompt against it, and sends only the
suffix (prefix divergence → renegotiate: re-render from divergence point or full
rebuild). Phase B2: first-class `packages/llm/src/protocols/pie.ts` route +
`ProviderV2.Api` variant in opencode's native stack — the intended extension
point per `specs/v2/provider-model.md`.

**What programmability buys, feature by feature** (each maps to an opencode
harness behavior and a paper mechanism):

| # | opencode behavior today | pie-native replacement | mechanism |
|---|---|---|---|
| B-1 | Full history re-sent, re-rendered, re-prefilled (or hash-matched) every turn | KV never leaves the worker; turn cost = O(new tokens) | persistent working set; paper R1 |
| B-2 | Client-side compaction/summarization → full re-prefill of the rewritten history | In-place context editing: drop stale tool outputs / old turns at token granularity, no re-prefill; summarize *into* the retained prefix | `working-set.discard/slice`, masks (paper Fig 7 #3 `mask_kvpage`) |
| B-3 | Subagents (task tool) re-prefill the shared parent prefix | Copy-on-write branch: `fork` the working set, subagent starts warm | `working-set.fork(on)` (prefix-tree / SGLang-equivalent, §7.3) |
| B-4 | Tool calls surfaced only when the AI SDK closes the block | Push each call the moment its arguments close, keep decoding subsequent parallel calls; client executes while the model still streams | incremental `tools.decoder` events (paper Fig 7 #2 early fire) |
| B-5 | Tool result waits for the next request's prefill | Prefill tool-result tokens as they arrive over the socket, hidden behind client-side tool latency | integrated I/O (paper R3) |
| B-6 | Malformed tool calls → repair middleware / retry turns | Grammar-constrained call arguments | `tools.format` + `grammar.wit`; **gate on a driver-capability probe** (portable-driver trap from qwen-code §5a) |

B-4/B-5 attack exactly the two factors the paper measured as the agentic-workflow
gains (round-trip elimination and KV retention across interactions, §7.1);
B-2/B-3 are the pie-unique features no OpenAI endpoint can express and the
demo-able "pie makes opencode faster *and* smarter about context" story.

**Risks / open questions.**

- Process lifetime = WS lifetime; worker FCFS termination under pressure kills
  live sessions. Mitigation: per-turn `update-index` checkpoints + client-side
  rebuild path (which is just Strategy A's code — hence phasing).
- Shadow-diff fidelity in the provider package: opencode's own history rewrites
  (compaction, truncation, message editing) must map to `edits` ops or force
  renegotiation. First profile disables opencode compaction (as qwen-code did);
  B-2 then *re-enables* long-session viability server-side.
- `small_model` utility calls (titles, summaries): route to the same pie model or
  an external provider; don't let them touch the session working set.
- One model per engine: fine for the single-agent dev-box story; multi-model is
  a routing-layer question out of scope here.

---

## 3. Shared prerequisite work (both strategies)

1. **Restore tool-history replay primitives host-side**: extend
   `model/common/src/instruct.rs` + `interface/inferlet/{chat,tools}.wit` with
   `assistant_with_tool_calls(content, calls) -> tokens` and
   `answer_batch(results) -> tokens` (per-arch, next to the existing per-arch
   templates). Without these, OpenAI history replay can't be rendered faithfully.
2. **Shared `openai-serving` crate** (types/render/decode/salvage-parsers/session
   canon) consumed by both `chat-completions` and `opencode-session` inferlets —
   the response/save unification invariant lives in exactly one place.
3. **Renderer parity harness** (port of qwen-code C3): rendered token ids vs HF
   `apply_chat_template` for the target models, driven by the banked fixtures.
4. **opencode wire audit** (A.4 above) — the H1–H17-style hazard enumeration for
   opencode, before trusting e2e runs.

## 4. Phasing and measurement

- **Phase 0** — prereqs (§3) + opencode wire audit/fixtures.
- **Phase A** — ingress + `chat-completions` port; acceptance suite (port the
  33-test harness); e2e stock opencode on toy tasks; A/B vs vLLM (same protocol
  as qwen-code §5b: wall clock, prompt tokens, cached_tokens, trajectory match).
- **Phase B1** — `opencode-session` inferlet + `pie-ai-sdk-provider` (file://
  npm); A is the reconnect/rebuild fallback. Measure against Phase A: bytes on
  the wire per turn, prefill tokens per turn, turn latency at 32k/128k contexts,
  subagent spawn latency with/without fork, compaction cost with/without B-2.
- **Phase B2 (optional)** — native `packages/llm` protocol in opencode V2;
  overlap/speculation experiments (B-4/B-5 depth, speculative continuation
  during tool wait).

Success criteria mirror the honest H200 lesson: don't claim wins at short
contexts; target the long-context single-agent regime (B-1/B-2) and the
subagent/multi-tenant regime (B-3), where the paper and the OpenHands result
(~26% e2e at 95% reuse) say the advantage lives.
