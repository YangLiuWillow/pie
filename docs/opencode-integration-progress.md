# opencode ↔ Pie integration — progress log

Companion to `opencode-integration.md` (the two-strategy plan). One entry per
completed task, newest first. Worktree: `Lin_startup/pie-opencode`, branch
`liu/opencode-integration` (from `dev` @ `58cb77936`).

## Task board

| id | task | status |
|---|---|---|
| P0.1 | Tool-history replay primitives (Instruct + WIT + host + SDK) | **done** |
| P0.2 | opencode wire audit + fixture capture | **done** |
| P0.3 | Shared `openai-serving` crate | **done** |
| P0.4 | Renderer parity harness | **done** |
| PA.1 | `chat-completions` inferlet on dev | pending |
| PA.2 | Gateway OpenAI ingress | pending |
| PA.3 | Acceptance suite + stock-opencode e2e | pending |
| PB.1 | `opencode-session` inferlet + AI SDK provider package | pending |
| PB.2 | Native `packages/llm` protocol in opencode V2 | pending (optional) |

## Log

### 2026-08-11 — P0.4 follow-up: D2 + D3 fixed; D1 deferred; D4 discovered

Applied the two mechanical template fixes the parity harness identified:

- **D2 fixed** — `model/qwen_3/src/chat.rs build_tool_system_prompt` no longer
  emits a leading `"\n"` before `# Tools` (the old validated branch carried
  this bug; HF renders `content + "\n\n" + "# Tools"`).
- **D3 fixed** — `inferlets/openai-serving/src/types.rs` now serializes tool
  schema envelopes with Python `json.dumps` separators (`", "`/`": "`) via a
  custom `python_json` formatter, matching HF Jinja `tojson`. This string
  feeds the snapshot address too — render and address changed together.
  (ensure_ascii escaping noted as an open caveat; fixtures are ASCII.)
- Re-run verdict: **all 5 fixtures are now char-exact except D1** (the
  `<think>\n\n</think>\n\n` no-think block after the cue — lands with PA.1's
  channel decision).
- **D4 (new)**: one token-boundary divergence inside the tool-call replay
  region — same bytes, different segmentation (pie's pre-tokenized fragment
  joins vs HF's whole-text encode, e.g. `…"arguments": ` + `{"…` splits where
  HF merges `Ġ{"`). Char-parity holds; token-parity doesn't. Decision for
  PA.1: encode each replayed turn's contiguous text in one pass (special
  tokens as separate ids) instead of concatenating isolated fragment
  encodings — deterministic either way, but only whole-text encoding matches
  HF's tokenizer behavior. Tests 24/24 + 28/28 still green after D2/D3.

### 2026-08-11 — P0.4 done: renderer parity harness — 3 divergences found

Harness at `integrations/opencode/parity/`: `render-tokens/` (Rust bin, new
root-workspace member; real serving path `plan_render` → `QwenInstruct` with
the exact "qwen3" `ChatMLConfig` from `model/src/instruct.rs` →
`pie-tokenizer`, fixture path in, token-id JSON array out) + `check_render.py`
(HF `apply_chat_template(…, enable_thinking=False)` on Qwen/Qwen3-0.6B,
tokenizer files only; token first-divergence report + complete grouped
char-level diff) + `README.md`. No prior harness existed to port —
`openhands-integration-updated:integrations/qwen-code/` has no `parity/` dir;
written fresh. Run: all 5 opencode wire fixtures.

**Verdict: no fixture token-exact; exactly 3 systematic divergences, nothing
else across ~7.5k-token prompts.** Do-not-fix-in-harness list for P0.1/PA.1:

1. **D1, all fixtures (incl. req-001/003 title calls, which are otherwise
   token-exact)**: HF `enable_thinking=False` appends `<think>\n\n</think>\n\n`
   (ids 151667,271,151668,271) after `<|im_start|>assistant\n`; pie `cue()`
   doesn't. Deliberate so far (pie's no-think channel = `/no_think` decoration
   at the inferlet, and opencode never sends `chat_template_kwargs`), but the
   channel must be *chosen* at PA.1: empty-think-block cue vs `/no_think`.
2. **D2, tools fixtures**: one extra `\n` — pie
   `…</available_skills>\n\n\n# Tools`, HF `…\n\n# Tools`. Template bug in
   `model/qwen_3/src/chat.rs`: `build_tool_system_prompt` starts `"\n# Tools"`
   while `equip_after_system` merges `{c}\n\n{block}` (HF: `content + '\n\n'`
   + `"# Tools…"`; no-system case `<|im_start|>system\n# Tools…` is also off
   by the same `\n`). Faithful port of the old branch — the bug is inherited,
   the qwen-code e2e validated tool-calling behavior, not byte parity here.
3. **D3, tools fixtures, ×293 (= every JSON separator in the 10 schemas;
   HF 480 spaced separators in `<tools>`, pie 187, Δ293)**: pie's
   `tool_schema_envelopes` serializes compact
   (`{"name":"bash","description":…}`), HF's `tojson` = `json.dumps` default
   separators (`{"name": "bash", "description": …}`, wire key order,
   ensure_ascii=False). Fix belongs in `openai-serving/types.rs` — and the
   envelope string feeds the snapshot address too, so render + address must
   change together (response/save unification).

Confirmed exact: all role scaffolding, replayed
`<tool_call>`/`<tool_response>` turns in req-005 (content:"" ≡ null ≡ absent
under HF's template — probed), tool name-sort, no trailing newline after the
final `<|im_end|>`. Harness notes: transformers 5.x `tokenize=True` return
shape changed — driver tokenizes the rendered text with
`add_special_tokens=False` instead; venv in scratchpad (transformers 5.15.0,
no torch).

### 2026-08-11 — P0.3 done: shared `pie-openai-serving` crate

Pure-logic host crate at `inferlets/openai-serving/` (new workspace member in
the root `Cargo.toml`; serde/serde_json only, zero wasm/WIT deps). Tests:
28/28 native, incl. all 5 opencode wire captures as fixtures; `cargo check
-p pie-engine` still clean.

- `src/types.rs` — ported near-verbatim from
  `openhands-integration-updated:inferlets/chat-completions/src/types.rs`;
  added `ChatMessage::text_opt()` (opencode's assistant `content:""` → None)
  and made `tool_schema_envelopes` **name-sort** the `{name,description,
  parameters}` envelopes (opencode sorts on the wire anyway; makes the
  snapshot address order-independent). `$schema` + `maximum: 2^53−1` schema
  noise round-trips losslessly through `serde_json::Value`.
- `src/streaming.rs` — ported chunk framing, refactored to return
  `serde_json::Value` (framing split into `sse_frame`/`sse_done`/`sse_ping`
  per the plan's "inferlet emits chunk JSON, gateway frames" decision);
  added `completion_response` (non-streaming body from the old handler) and
  the `: ping` comment helper (opencode keepalive). Golden tests pin exact
  chunk shapes (atomic tool-call delta with id+name on first delta).
- `src/session.rs` — canon/FNV-1a-64×2/split ported verbatim minus engine
  calls; `snapshot_name` → `snapshot_address` returning bare 32-hex (caller
  prefixes its namespace — old code hardcoded `qwenchat/`). Response/save
  unification invariant documented at module level.
- `src/render.rs` — reworked engine-free: `RenderOp` enum + `plan_render`.
  Deviations from old render.rs: no `System` op (leading system/developer
  folds into `EquipAfterSystem` even with zero tools — the no-tools title
  call renders as a plain system turn; mid-list system → `MisplacedSystem`
  error, old code rendered it inline); `AnswerBatch` pairs now carry the
  real tool name recovered via `tool_call_id→name` from preceding assistant
  turns (old code passed `""`); `/no_think` decoration + special-token
  `sanitize_messages` deliberately left to the inferlet (tokenizer-touching).
- `src/error.rs` — extracted from handler.rs: OpenAI `{"error":{message,
  type,param:null,code:null}}` body, `invalid_request_error`/`server_error`
  constants, `parse_request` (only 400 rule: bad JSON / empty messages).
  Handler itself not ported (PA.1).
- Surprise from the old code: old `AnswerBatch` genuinely never used tool
  names (Qwen template folds results namelessly), so the name recovery is
  new capability, not a port — harmless for Qwen, needed if a template
  renders names. Also `rl_completions` fixtures don't exist on this branch
  (they live on the old one); the fixture test sweeps them only if present.

### 2026-08-11 — P0.1 done: tool-history replay primitives restored on dev

Tests: 24/24
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
