# qwen-code ↔ Pie integration plan

**Date:** 2026-08-10. Companion to `qwen-code-rl-audit.md` (wire contract, hazards H1–H17) and
`codex-integration.md` (the structural precedent).

**Decision (agreed):** do NOT reimplement qwen-code as an inferlet. Keep stock qwen-code
client-side and serve it from a new **`chat-completions` inferlet** — an OpenAI-compatible
`/v1/chat/completions` SSE daemon with content-addressed KV-session reuse (Option A). A
same-harness server-side port (Route 2, with a wire-replay parity harness) is a later,
optional ablation and is specced in §6 but not built now.

Why this shape wins (short form): the paper's dominant agentic gain for 8B+ models is KV
retention across tool-call turns, and `codex-responses/session.rs` proved that is achievable
with the agent fully client-side via content-addressed snapshots (OpenHands measured 95%+ KV
reuse, ~26% end-to-end win). Tool execution must round-trip to the user's environment in
every design, so server-siding the loop only removes a localhost hop — noise next to shell
latency. Pattern A (`openhands-agent`) was deprecated precisely because rewriting the loop
makes speed/accuracy claims unattributable.

---

## 1. Deliverable overview

| # | Deliverable | Path | Status |
|---|---|---|---|
| 1 | `chat-completions` inferlet (HTTP/SSE daemon) | `inferlets/chat-completions/` | build now |
| 2 | Launch scaffolding + launch profile | `integrations/qwen-code/` | build now |
| 3 | Acceptance tests from audit §1 + fixture replay | `tests/inferlets/` + in-crate tests | build now |
| 4 | Renderer-parity check vs. HF chat template | `integrations/qwen-code/parity/` | build now (script) |
| 5 | Same-harness server-side port + replay harness | — | later (§6) |

## 2. The `chat-completions` inferlet

New Rust inferlet, assembled from four in-repo precedents:

- **HTTP/SSE skeleton** — `inferlets/openresponses/{lib,streaming}.rs` (`#[wstd::http_server]`,
  `BodyForthcoming`, per-event `flush()`).
- **Message-list rendering + native tool calling** — `inferlets/openhands-completion/src/lib.rs`
  (already consumes OpenAI chat dicts: `equip_after_system_prefix`, `assistant_with_tool_calls`,
  `answer_batch`, `tools::Decoder`).
- **KV-session reuse** — `codex-responses/src/session.rs` (branch `liu/codex-integration`):
  canonicalize items → FNV-1a-64 ×2 → named snapshot; strip trailing tool/user suffix on the
  next request and `Context::take` the match. Port, don't re-derive.
- **Token accounting** — `rl-completions/completions.rs` (branch `liu/rl-completions`) for
  usage reporting shape.

### Files

```
inferlets/chat-completions/
  Pie.toml            # empty [parameters] → daemon-launchable
  Cargo.toml
  src/lib.rs          # routing: POST /chat/completions, /v1/chat/completions, GET /, OPTIONS
  src/types.rs        # wire types (request + chunk/response + error), serde with unknown-field tolerance
  src/render.rs       # messages[] + tools[] → Context prefix ops (the ONLY place render order lives)
  src/session.rs      # content-addressed KV snapshots (port of codex session.rs, chat-message canon)
  src/streaming.rs    # chat.completion.chunk SSE emitter + keepalives during prefill
  src/handler.rs      # request orchestration: parse → session resume → generate → decode → emit
```

### Wire contract (from audit §1 — this is the acceptance checklist)

Request: accept `model`, `messages` (system first; assistant with `tool_calls[]` and echoed
`reasoning_content`; `{role:"tool", tool_call_id, content:[parts]}` — content may be string OR
parts array), `tools` (name-sorted function schemas), `max_tokens` (always present), `stream`
(always true from qwen-code; support false for curl debugging), `stream_options:
{include_usage:true}`, optional `temperature`/`top_p`, `chat_template_kwargs:
{enable_thinking:false}` (honor: render no-think), `reasoning` field duplication for qwen3
models (ignore on input). **Unknown fields must be ignored, never 400.**

Response invariants (violations break qwen-code, per audit table):

1. Final chunk of a text turn carries `finish_reason` (`"stop"`/`"length"`); tool-call turns
   use `"tool_calls"`.
2. Text turns must produce non-empty content (empty → qwen-code retry loop). If the model
   emits only whitespace/EOS, still emit the decoded text as-is; never emit zero content
   deltas plus `finish_reason:"stop"` on an empty accumulation without at least one delta.
3. `Content-Type: text/event-stream` on streaming 200.
4. Tool-call `id`s present and **unique across the whole session** — derive from
   `runtime::instance_id()` + counter (fresh WASM instance per request; a static counter
   alone recreates the `call_0` collision bug both prior integrations hit).
5. Never emit `finish_reason:"error_finish"`.
6. Malformed request → **400** with OpenAI error JSON (`{"error":{message,type,param,code}}`);
   500 reserved for genuine server faults (500s trigger 7×-app × 3×-SDK retry storms).
7. Usage chunk (`choices:[]`, `usage:{prompt_tokens, completion_tokens, total_tokens}`) when
   `include_usage`; report KV reuse via `prompt_tokens_details.cached_tokens`.
8. **Never return a context-length 400** — that fires H1 full-history compaction client-side.
   On genuine overflow, emit what fits + `finish_reason:"length"` rather than erroring.
9. SSE keepalive comments (`: ping`) during long prefills — qwen-code aborts after 240 s
   without a chunk (`QWEN_STREAM_IDLE_TIMEOUT_MS`).

Chunk framing: first delta carries `role:"assistant"`; then content deltas; tool calls stream
as `delta.tool_calls[{index, id, function:{name, arguments-fragments}}]`; final chunk carries
`finish_reason`; then optional usage chunk; then `data: [DONE]`.

### Rendering (`render.rs`)

- `equip_after_system_prefix(system, tools)` for turn 0 prefix; subsequent history replay:
  user → `user`, assistant-with-calls → `assistant_with_tool_calls_prefix`, plain assistant →
  `assistant`, tool results → `answer_batch_prefix` (batch consecutive tool messages, keyed by
  `tool_call_id` → name mapping recovered from the preceding assistant turn).
- `reasoning_content` on history assistant turns (H17): with `enable_thinking:false` we render
  the no-think channel and **drop** echoed reasoning_content from replay — parity with what
  vLLM does under the same flag. Must be consistent between generation and replay or prefixes
  self-invalidate.
- Tool-call dialect: use the tool-use trait's native `format() → grammar` + `Decoder` for the
  loaded model. Verify empirically (Phase C) which dialect Qwen3-0.6B's template yields
  (hermes `<tool_call>` JSON vs. qwen3-coder `<function=…>` XML) and that qwen-code's
  `QWEN_CODE_TOOL_CALL_STYLE=general` prompt examples don't contradict it.
  `pie_openhands/qwen3coder_parser.py` is the XML-dialect parity reference if needed.
- Optional `use_grammar` knob (constrained decode of tool calls) mirroring
  `openhands-coder-session` — default off, flag-gated.

### KV sessions (`session.rs`)

Port of the codex design with a chat-message canonical form:

- Canon items: `Sys(text)` folded into address header with tool schemas; `Msg{role,text}`,
  `Call{id,name,args}`, `CallOutput{id,text}`. Reasoning content excluded from the address
  (it is excluded from the rendered stream too — keeps H17 from poisoning addresses).
- Address = FNV-1a-64 two-seed hash over (system, name-sorted tool schemas, canon items) →
  `qwenchat/{hash32}`. qwen-code sends no session key, so the namespace is purely
  content-addressed; no scope needed for correctness. (H14 tool-list growth changes the
  address → clean miss → rebuild: correct, just slower; the §4 launch profile pins the list.)
- Resume: strip trailing `tool`/`user` messages back to the last assistant item; hash the
  remainder; `Context::take` on hit (open+delete keeps ≤1 live snapshot per branch), append
  only the stripped suffix + cue. Miss → full rebuild (always safe).
- Save: after generation, extend canon with the just-produced assistant item, hash, `save(name)`.
- Report hit depth as `cached_tokens`; log mode (`fresh|extended|rebuilt`) to stderr for the
  instrumentation rule below.

### Known engine constraints to respect

- `runtime/src/daemon.rs` instantiates a **fresh WASM instance per HTTP request** — zero
  in-memory cross-request state; snapshots are the only continuity. No statics for ids.
- Snapshot retention: bound the namespace (take-on-hit already caps live snapshots per
  conversation branch; stale branches leak until engine GC — acceptable for now, note in
  README; revisit with the snapshot-retention work from commit `14b74063`).

## 3. Test plan (Phase C)

**C1 — In-crate unit tests** (native target, no WASM): request parsing over the checked-in
fixtures `tests/inferlets/fixtures/rl_completions/wire/episode*/openai-*.json` (real
`--openai-logging` captures = real qwen-code request bodies); canon/hash stability; resume-point
splitting; SSE chunk framing golden tests.

**C2 — Live acceptance script** (`integrations/qwen-code/test_acceptance.py`): boot
`pie serve` + `launch_daemon`, then assert every row of the audit §1 hard-requirements table
with raw HTTP: finish_reason present, SSE content-type, unique tool ids across two sequential
requests, 400-vs-500 semantics, unknown-field tolerance, include_usage chunk, [DONE]
terminator, keepalives.

**C3 — Renderer parity** (`integrations/qwen-code/parity/check_render.py`): for each fixture
request, compare the inferlet's rendered token ids (exposed via a debug `echo_tokens` request
flag, gated to non-stream mode) against HF `tokenizer.apply_chat_template(..., enable_thinking
=False)` for the same model. Byte/token equality or documented, deliberate divergence — the
`rl-completions` chat endpoint was explicitly deferred on this exact verification, so this is
the load-bearing test.

**C4 — End-to-end**: run real qwen-code (clone at `~/Desktop/Lin_startup/qwen-code`, pinned
`3235faf`) with the §4 profile against the daemon on a toy task in a scratch repo; verify turn
completion, tool execution, clean exit; then two identical rollouts + byte-diff of their
`--openai-logging` captures (the `tool_desc_invariance` discipline: a leaked variable byte
fails silently and silently zeroes the reuse win). Instrument snapshot hit rate from day one.

## 4. Launch scaffolding (`integrations/qwen-code/`)

- `launch_daemon.py` — port of `integrations/codex/launch_daemon.py` (`PieClient` →
  `install_program` + `launch_daemon(port=8123)`).
- `run_pie_qwen.sh` — boots `pie serve :18080` + daemon, exports the audit §6 env profile
  (`QWEN_HOME`/`QWEN_RUNTIME_DIR` isolation, `OPENAI_BASE_URL=http://127.0.0.1:8123/v1`,
  `QWEN_USAGE_STATISTICS_ENABLED=false`, `--bare --safe-mode --auth-type openai`, turn/wall
  limits) and writes the §6 `.qwen/settings.json` (compaction off via huge contextWindowSize,
  clearContextOnIdle disabled, skipStartupContext, truncateToolOutputThreshold).
- `README.md` — architecture, hazards actually mitigated vs. accepted (H3 synthetic
  continuation and H11 git-status remain fork-patch items, out of scope here; within-rollout
  reuse survives H11, cross-rollout reuse does not).

## 5. Execution order

- **A. Scaffold + wire types + render** — crate builds for `wasm32-wasip2`; C1 parsing tests
  green against fixtures.
- **B. Generation + SSE + sessions** — handler end-to-end; C2 acceptance green via curl.
- **C. Parity + e2e** — C3 renderer parity, C4 real qwen-code run on M2 Metal with
  Qwen3-0.6B (known-good from the OpenHands work).
- **D. Scaffolding + docs** — `integrations/qwen-code/` complete; results note appended here.

Risks called out up front: (1) C3 may surface chat-template divergence between pie's
`instruct` trait and HF for Qwen3 — that's the known reason chat/completions was deferred;
budget for template fixes or a model-config change rather than hacking render.rs. (2) The
0.6B local model is a functional testbed only; reuse wins are expected to be modest until an
8B+ run (paper §7.1). (3) Tool-dialect mismatch under `TOOL_CALL_STYLE=general` — resolve
empirically in C3 before trusting C4.

## 5a. Results (2026-08-10, first live bring-up)

Phases A–C executed same-day on M2 Metal, Qwen3-0.6B (`integrations/qwen-code/pie_config.toml`):

- **Unit tests: 12/12** (wire-fixture parsing, session hashing incl. save↔echo-back round-trip,
  SSE framing, fence parser).
- **Acceptance: 33/33** (`test_acceptance.py`) — full audit §1 table plus a two-turn echo-back
  conversation with `cached_tokens > 0`.
- **E2E: stock qwen-code v0.21.6** (npx, §6 profile) completed a shell-tool task against the
  daemon: native `run_shell_command` tool call decoded and executed, file created, clean exit,
  zero retries. Replay of a tool-result turn showed **99.7% KV reuse** (cached 8597/8626).

Findings folded back into the design:

1. **Response/save content unification.** The turn's content string must be computed once and
   used for BOTH the wire response and the snapshot address — any divergence (trimming,
   fence-truncation, whitespace fallbacks) breaks every subsequent resume, because qwen-code
   echoes response content back verbatim. In streaming mode the canonical content is exactly
   the emitted delta bytes, untrimmed.
2. **Phase-2 forced tool call removed.** Grammar-constrained decoding traps the guest on the
   portable driver; the trap kills the SSE stream pre-finish and qwen-code burns its 4
   `NO_FINISH_REASON` retries on identical failures. A no-call turn is graceful; a dead stream
   is not. Re-add behind a driver-capability probe only.
3. **Known engine defect under KV pressure.** With the page pool near-full (accumulated
   snapshots), a save-time seal+flush landing on an exact page boundary can fail with
   `KV_INVARIANT_VIOLATION` (`DEFECTS_OVERCOMMIT.md` defect 2 — deferred reserve under
   pressure). Failure is non-fatal by design (next turn rebuilds), but it costs that turn's
   reuse; snapshot retention/GC is the real fix.
4. The stale-binary + stale-config traps: `cargo build -p pie` builds the runtime *lib*, the
   server binary is `-p pie-server`; portable-driver option names changed
   (`max_forward_tokens`/`total_pages`).

## 5b. H200 results (2026-08-10, pod terminated at user request mid-27B-leg)

Setup: RunPod H200 (driver 580.159.04, sm_90), branch at `f380e990`, weights for both
target models. Bring-up potholes fixed for posterity: base image lacks cmake ≥3.23
(pip-install it), the vendored SM90 TMA-WS MoE kernels need the fp8-alpha-gate patch on a
fresh CPM cache — now applied automatically from CMake (`18b86efc`) — and `pie driver
cuda-native doctor` then confirms `hopper TMA-WS launchers: COMPILED`; boot banner
`prefill_decode_plan=on xqa_decode=on` for the 30B MoE.

**Qwen3-Coder-30B-A3B-Instruct, 5-task qwen-code A/B (audited profile, both arms):**

| arm | ok | wall | prompt tok | cached (reuse) | notes |
|---|---|---|---|---|---|
| pie (chat-completions daemon) | 5/5 | 45.1 s | 291 K | 237 K (81.5%) | KV snapshots hit every follow-up turn |
| vLLM 0.25.1 fair tier (APC on, CUDA graphs, tuned MoE, qwen3_coder parser) | 5/5 | 30.7 s | 236 K | (APC internal, not surfaced) | |

Trajectories tool-call-identical on 3/5 tasks; the two divergences trace to prompt-render
differences (pie `/no_think` decoration vs vLLM template kwarg) plus sampling — qwen-code
v0.21.6 has **no settings/env path that puts `temperature` on the wire** for the OpenAI
provider (`samplingParams` is not in the settings schema), so both arms ran server
defaults. On trajectory-matched tasks pie is ~5–15% slower. Honest read: at these context
lengths (~9 K tok/turn, H200 prefill ~ms) explicit KV reuse cannot pay for pie's remaining
batch-1 decode gap vs vLLM + per-request daemon overhead (fresh WASM instance + snapshot
seal/flush/save per turn). The reuse win needs long contexts (the OpenHands SWE-bench
result: ~26% faster at 95% reuse) and/or the multi-tenant regime — single-agent toy tasks
are pie's worst case, and 1.47× overall / ~1.1× matched is the honest number there.

Robustness work the runs forced (all committed): salvage parsers for **bare Coder-XML**
calls (30B emits `<function=…>` without the `<tool_call>` wrapper under general-style
prompt examples) and **unterminated hermes** calls (27B stops at EOS before
`</tool_call>`), both schema/brace-aware, both after the native decoder and fenced-JSON
fallback.

**Qwen3.6-27B: incomplete.** Loads and generates (`model_type=qwen3_5_text`) but pie's
fast attention paths are OFF for this arch (`prefill_decode_plan=off xqa_decode=off`) —
fallback-path speed, and most bench tasks still failed at termination time (captures in
the banked tarball). vLLM arm never ran. Making 27B a fair pie arm needs driver fast-path
support for qwen3_5_text first; benchmarking it today would measure the fallback path
(the A100 lesson from the runpod handovers).

## 6. Later: same-harness server-side port (Route 2) — spec only

Kept for the record; not part of this build. Fidelity is made testable, not asserted:

- **Claim:** given identical tool results, the server-side harness emits byte-identical
  model-facing requests to stock qwen-code v0.21.6.
- **Instrument:** replay harness. Parse an `--openai-logging` episode into
  `(request_i, response_i)` pairs; drive the port with both externals stubbed — intercept its
  assembled request, assert equality with `request_i`, inject recorded `response_i`; tool
  results arrive implicitly in `request_{i+1}`. First divergence localizes the infidelity to
  exact bytes.
- **Normalization budget:** pin environment (fixed workspace, frozen date, seeded ids) over
  canonicalization; every canonicalization rule is a hole in the claim. H1–H17 is the
  pre-enumerated list of where divergence will appear.
- **Coverage:** directed fixture episodes per hazard (force compaction, 500K-char blanking,
  tool-list growth); fault-injection for H3/retry paths, which replay cannot reach.
- **What it measures over Option A:** scheduler integration (`idle`/`bid`) under multi-tenant
  load, and behavior-preserving overlap/speculation — with Option A as the unconfounded
  baseline. Two port routes: componentize-js of `@qwen-code/qwen-code-core` (1–2 day spike to
  size the Node-builtin shim surface) vs. Rust port under the replay harness.
