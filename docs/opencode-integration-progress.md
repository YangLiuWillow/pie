# opencode ↔ Pie integration — progress log

Companion to `opencode-integration.md` (the two-strategy plan). One entry per
completed task, newest first. Worktree: `Lin_startup/pie-opencode`, branch
`liu/opencode-integration` (from `dev` @ `58cb77936`).

## Task board

| id | task | status |
|---|---|---|
| P0.1 | Tool-history replay primitives (Instruct + WIT + host + SDK) | **done** |
| P0.2 | opencode wire audit + fixture capture | **done** |
| P0.3 | Shared `openai-serving` crate | pending |
| P0.4 | Renderer parity harness | pending |
| PA.1 | `chat-completions` inferlet on dev | pending |
| PA.2 | Gateway OpenAI ingress | pending |
| PA.3 | Acceptance suite + stock-opencode e2e | pending |
| PB.1 | `opencode-session` inferlet + AI SDK provider package | pending |
| PB.2 | Native `packages/llm` protocol in opencode V2 | pending (optional) |

## Log

### 2026-08-11 — P0.1 done: tool-history replay primitives restored on dev

All changes on `liu/opencode-integration` (uncommitted). Tests: 24/24
`pie-model-qwen-3 --features chat`, `pie-model` 7/7, `pie-engine` +
`inferlet` (wasm32-wasip2) compile clean.

- `interface/inferlet/tools.wit`: added `equip-after-system`,
  `assistant-with-tool-calls` (takes `list<tool-call>`), `answer-batch`
  (takes `list<tuple<string,string>>`); synced to both vendored copies via
  `scripts/sync-wit.sh`.
- `model/common/src/instruct.rs`: three new `Instruct` trait methods with the
  old-branch default impls (plain-concat / drop-calls / per-result fold).
- `model/qwen_3/src/chat.rs`: ported the validated implementation from
  `openhands-integration-updated:runtime/src/model/instruct/qwen3.rs` (the
  branch containing the qwen-code H200 work at `f380e990` — NOT
  `liu/codex-integration`, whose copy predates the final fixes): no-newline
  role prefixes, pre-tokenized tool-call fragments (fragments always encoded
  as fixed literals, dynamic parts encoded in isolation — retokenization
  hazard), the three overrides, and the reference tests. Also fixed dev's
  regressed `build_tool_system_prompt`: `"\n# Tools"` preamble (dev had
  `" # Tools"`) + `{"type":"function","function":…}` envelope wrapping.
- Host: `runtime/engine/src/inferlet/host/tools.rs` delegations; SDK:
  `sdk/rust/inferlet/src/lib.rs` now re-exports a `tools` module (it had no
  tools surface at all).
- Deferred to PA.1 (deliberately out of P0.1 scope): `ToolFormat::Coder`
  (Qwen3-Coder `<function=…>` XML dialect) and the two salvage parsers —
  port them from the same `openhands-integration-updated` file when building
  the inferlet.

### 2026-08-11 — P0.2 done: opencode wire audit + real captures

Deliverables in `tests/inferlets/fixtures/opencode/`: `record_server.py`
(stateful recorder), `opencode.json` (capture config), `wire/req-001..005.json`
(real stock `opencode-ai@1.18.16` traffic incl. the history-replay request
with `assistant+tool_calls` and `role:"tool"`), `AUDIT.md` (hazard table,
capture-verified vs source-derived marked), `README.md`.

Load-bearing findings for PA.1/PA.2:

1. **Keepalives: SSE comments work.** opencode's `chunkTimeout` watchdog wraps
   the raw body reader (resets on any bytes, upstream of the SSE parser), and
   custom providers have NO default header/chunk/total timeout at all.
   `: ping` comments are safe (capture-verified).
2. **Tool-history shapes**: assistant replay carries `content: ""` (empty
   string, not null/absent) + `tool_calls[{id, type:"function",
   function:{name, arguments:<json-string>}}]`; tool result is
   `{role:"tool", tool_call_id, content:<string>}`.
3. **First tool_call delta per index must carry both `id` and
   `function.name`** or the AI SDK throws.
4. **Never 500 on malformed input** — opencode retries 5xx forever
   (`maxRetries:0` at SDK level, but session-level retry on 429/5xx/network
   is unbounded). No empty-content retry loop (unlike qwen-code).
5. Always `max_tokens` + `stream_options:{include_usage:true}`; no
   temperature/top_p for unknown model ids; cache reuse read from
   `prompt_tokens_details.cached_tokens` (verified end-to-end).
6. Headers include `x-session-id`/`x-session-affinity` (+
   `x-parent-session-id` on subagents) — a ready-made sticky-affinity key
   for PA.2 and branch key for Strategy B.
7. Tolerate JSON-Schema noise: `$schema` draft-2020-12 keys and
   `maximum: 9007199254740991` appear in tool parameters.
8. Every session also fires a no-tools title-generation request at the same
   model.
9. The V2 native-LLM path is gated by provider id (`openai|anthropic|
   opencode*`), so a custom `pie` provider always uses the AI SDK path —
   Strategy B's B2 phase needs that gate widened or a first-class variant.

### 2026-08-11 — environment note

Disk filled to zero mid-build (killed tool execution). Freed ~19.6 GB by
deleting `~/Library/Caches/vscode-cpptools` (17 GB, IntelliSense cache —
regenerates) and `~/Library/Caches/pip`. To avoid re-duplicating build
artifacts, worktree builds use `CARGO_TARGET_DIR=…/Lin_startup/pie/target`
(same commit as the main checkout → 24 GB of artifacts shared). The
`pie-opencode/target` dir was deleted; keep using the shared target dir.

### 2026-08-11 — project setup
- Two-strategy plan written: `docs/opencode-integration.md` (Strategy A: OpenAI
  endpoint; Strategy B: harness-adjacent session inferlet; phased A → B).
- Worktree `pie-opencode` created on new branch `liu/opencode-integration` from
  `dev` @ `58cb77936`. Note: `docs/` is untracked on `dev`; design docs should be
  committed on this branch.
- Codebase surveys completed (opencode provider layer; pie dev serving surface);
  findings folded into the plan doc §0.
