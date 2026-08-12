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
| PA.1 | `chat-completions` inferlet on dev | **milestone 1 done** (sessions/grammar/coder-dialect pending) |
| PA.2 | Gateway OpenAI ingress | **done** |
| PA.3 | Acceptance suite + stock-opencode e2e | suite + scaffolding **authored**; live run pending (needs a CUDA ≥12.8 GPU box, or ~1.5 GB more free RAM locally) |
| PB.1 | `opencode-session` inferlet + AI SDK provider package | pending |
| PB.2 | Native `packages/llm` protocol in opencode V2 | pending (optional) |

## Log

### 2026-08-11 — GPU bring-up attempt (RunPod H100): blocked on the image's CUDA, pod terminated

Attempted the PA.3 live run on a rented H100 PCIe (sm_90, driver 580.142,
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) because the local
Metal path is RAM-blocked. **No acceptance results — the host build never
finished.** The pod was terminated at the user's request (all account pods
stopped; the H100 alone billed $2.89/h). Everything needed to retry lives in
git, so the loss is build state only.

**The blocker, and the note that makes the next attempt cheap: the stock
CUDA 12.4 image cannot build pie's sm_90 kernels.** The vendored XQA
attention kernels emit Hopper TMA bulk-copy instructions that 12.4's `ptxas`
rejects:

```
ptxas …-6_attention_xqa_gqa8_sm90.ptx, line 1922; error : State space
        incorrect for instruction 'cp.async.bulk.tensor'
```

Dozens of these; `cargo build -p pie-bin --release --features driver-cuda`
died at ~28 min with `BUILD_EXIT=101`. It is a *compiler* limit, not a driver
one (580.142 is fine) — the prior validated H200 bring-up ran a 12.8-era
toolkit, which is why `qwen-code-integration-plan.md` §5b never mentions it.
That doc's bring-up list should gain this next to its cmake-≥3.23 note.

Recipe for the next pod:

- Prefer a **CUDA ≥ 12.8 devel image** — it removes a ~10-min apt step and a
  full kernel recompile. If stuck on 12.4:
  `apt-get install -y cuda-toolkit-12-8` (toolkit only, no driver), then
  `export PATH=/usr/local/cuda-12.8/bin:$PATH CUDACXX=…/nvcc` (plus
  `CMAKE_CUDA_COMPILER` if cmake cached the old one), and delete **only**
  `target/release/build/pie-worker-*/out/cuda` so cmake reconfigures while
  the Rust artifacts survive. Fallback: `.run` installer, `--toolkit --silent`.
- The image ships **no `nvcc` on `PATH`** (it is at `/usr/local/cuda/bin`),
  and **no apt cmake at all** — `pip install cmake ninja`.
- Its NVIDIA apt list served a stale `Packages.gz` ("Mirror sync in
  progress"), which fails `apt-get update` hard: move
  `/etc/apt/sources.list.d/cuda-*.list` aside for the base installs, restore
  it only if you need the 12.8 toolkit.
- Materialize the Rust toolchain ONCE (`cargo --version` inside the repo)
  before launching parallel builds — concurrent first-use races rustup's
  component install (`could not rename 'component' file … File exists`).
- What did work, for timing reference: prereqs + rustup ≈ 3 min; clone 25 s;
  **the wasm inferlet builds clean on Linux in 35.5 s** (604 KB); HF
  `Qwen/Qwen3-0.6B` snapshot ≈ 4 s. The pod also confirmed by inspection that
  `CHAT_INFERLET` is the fixed `chat-completions@0.1.0` on this branch.

### 2026-08-11 — PA.3 first half: acceptance suite + launch scaffolding AUTHORED (not yet run live)

**Blocker for the live half:** machine RAM vs the Metal admission margin —
`pie serve` needs ~3.2 GiB reclaimable for the Metal heap and the machine had
~1.9, so the driver refuses admission. Mitigations documented in the script:
free RAM and/or `PIE_METAL_ROW_BUDGET_MB` (activation-row reservation, driver
default 1024 MB, `driver/metal/src/context.cpp row_budget_bytes()`; don't go
low enough to refuse the ~7.5k-token opencode prompt — over-long prompts are
refused, not chunked).

New in `integrations/opencode/`:

- **`test_acceptance.py`** — **25 tests**, stdlib-only raw HTTP (incl. SSE
  parsing with keepalive-comment/junk-line separation), one test per hard
  requirement from `tests/inferlets/fixtures/opencode/AUDIT.md` + the
  PA.2 ingress contract: health/models; 401-without-Bearer with
  `authentication_error` shape; 400 (never 5xx) on bad JSON / non-object /
  empty messages with OpenAI error bodies; `$schema` + `maximum:2^53−1` +
  unknown-top-level tolerance; streaming (content-type, role-first delta,
  content accumulation, finish stop|length, usage chunk incl.
  `prompt_tokens_details.cached_tokens ≤ prompt_tokens`, `[DONE]`, chunk-id
  consistency, no `{"status":…}` envelope leakage, no stdout leakage);
  non-streaming single-body shape; req-004 verbatim replay (10 real tools)
  + a synthetic forced-tool turn; req-005 tool-history replay; cross-process
  tool-call-id uniqueness over two sequential requests; `max_tokens:1` ⇒
  `length`; keepalive/long-prefill completion (timing soft-logged);
  global never-`error_finish` + no in-stream `error` events sweep.
  Policy: wire-SHAPE assertions hard; 0.6B model-BEHAVIOR assertions soft
  (`[WARN]`, e.g. "did the model actually call the tool") — but whenever
  calls DO appear, the atomic-first-delta `index`+`id`+`function.name` /
  valid-JSON-arguments / `finish_reason:"tool_calls"` shape is hard.
  Fixture bodies replay verbatim except `max_tokens` clamped 32000→1024
  (`PIE_TEST_MAX_TOKENS`) to bound live runtime. Run:
  `PIE_BASE_URL=http://127.0.0.1:8080 python3 integrations/opencode/test_acceptance.py`
  (`--collect-only` / `--only <substr>`; exit 2 = server unreachable).
- **`run_pie_opencode.sh`** — `$PIE_BIN` (default
  `../pie/target/release/pie`, the shared-target-dir release build) `-c
  <config> serve` → wait `/health` (180 s, model load) → suite → clean
  kill; `--serve-only` keeps it up for the stock-opencode e2e. Config = arg
  or a generated trimmed copy of the known-good `~/.pie/config.toml`
  (metal, `Qwen--Qwen3-0.6B-optimized`) with ONE change: `max_model_len`
  4096→16384 (= 512 pages × 32), since req-005 renders at 7473 tokens.
  Also refreshes `$PIE_HOME/programs/chat-completions/0.1.0.{wasm,toml}`
  from the shared target dir when newer, and runs `pie doctor` preflight.
- **`opencode.json`** + **`README.md`** — stock-opencode e2e profile
  (provider `pie`, `@ai-sdk/openai-compatible`, baseURL
  `http://127.0.0.1:8080/v1`, model `pie/qwen3-0.6b`, `limit.output:4096`
  to bound `max_tokens`); README covers acceptance, the
  `opencode run -m pie/qwen3-0.6b …` e2e, blockers, and what green means.

Serverless self-checks (all that can run without the server): `--collect-only`
lists 25/25; helpers unit-probed (fixture load + clamp, SSE parse incl. junk
detection); unreachable-server preflight exits 2; `bash -n` clean; `pie doctor`
on the generated profile: ready (metal compiled, weights artifact found,
config parses).

**Finding while wiring the launch path — predicted first-run failure:**
`gateway/src/ingress/openai.rs` launches `CHAT_INFERLET = "chat-completions"`
(bare), but `ProgramName::parse` (runtime/engine/src/inferlet/program.rs:154)
requires `name@major.minor.patch` — the engine will reject the launch and
every chat request will 500 with `Invalid program identifier
'chat-completions'`. PA.2's 6/6 integration tests missed it because the stub
worker never parses the name. Fix belongs to the live half: constant →
`chat-completions@0.1.0` (or bare-name resolution in `handle_launch_process`)
+ rebuild; the suite will show it as universal 500s until then.

New guest crate `inferlets/chat-completions/` (wasm32-wasip2, own workspace
root via empty `[workspace]` table — same exclusion policy as
`tests/inferlets/*`; path-deps `inferlet` SDK + `pie-openai-serving`).
Serves one OpenAI chat-completions request per process launch on the PA.2
gateway⇄inferlet envelope (`{"status":u16}` first, then verbatim chunk JSON
per message / one unary body). Build (verified):
`cd inferlets/chat-completions && CARGO_TARGET_DIR=…/Lin_startup/pie/target
cargo build --target wasm32-wasip2 --release`.

- **`src/engine.rs`** — generation core ported from
  `tests/inferlets/chat-completion/src/lib.rs` (PTIR prefill + in-graph
  top-p/Gumbel sampling + device-carried decode loop under `run_ahead`),
  two deltas: prefill is CHUNKED via `prefill_chunks` (naive-baseline
  shape — serving prompts exceed `max_embed_length`), and per-token policy
  is a caller-supplied callback. Engine failures never become wire errors:
  `generate` returns the first error and the turn degrades to
  `finish_reason:"length"` (KV overflow mid-decode = "generate what fits").
- **`src/turn.rs`** — orchestration ported from the OLD validated handler
  (`openhands-integration-updated:…/handler.rs`, logic only, not its engine
  API): per token = tool-decoder feed → atomic tool-call delta on `Call`
  (dedup + `call_{instance-id-fragment}_{n}` ids, unique per process) →
  stop-set check (chat stops + `<|im_start|>` anti-loop stop) → chat-decoder
  delta through `VisibleFilter` (partial `<tool_call>`/`<think>` never leaks
  into content) → client stop-strings on the visible tail. ≥1 call ⇒
  `finish_reason:"tool_calls"`; generation runs until the model's own stop
  (old-handler behavior — no cut after the call block). Salvage after the
  loop: fenced-JSON on visible text, unclosed-hermes on raw text.
  `final_content` fallback (raw-minus-think, then `"…"`) guarantees a
  non-empty text turn.
- **`src/lib.rs`** — envelope + 400-vs-500 discipline: bad JSON / empty
  `messages` / misplaced system / unknown role → `{"status":400}` + OpenAI
  error body (never a process error; opencode retries 5xx forever); render
  planned via `plan_render`, mapped 1:1 to WIT (`tools.equip-after-system`,
  `chat.user`, `tools.assistant-with-tool-calls`, `tools.answer-batch`,
  `chat.cue-no-think` — **D1 decided: always no-think this milestone**,
  matching the token-exact parity verdict). Streaming: status → role-first
  chunk → content/tool-call deltas → finish chunk → usage chunk
  (`prompt_tokens_details.cached_tokens: 0` for now) when `include_usage`;
  non-stream: status + one `completion_response` body. Sampling defaults
  from the reference inferlet (t=0.6, top_p=0.95) when the request omits
  them; `max_tokens` via `effective_max_tokens(4096)`.
- **Ported INTO `pie-openai-serving`** (pure logic, native tests):
  `filter.rs` — `VisibleFilter` verbatim from the old branch's `filter.rs`
  + `sanitize_messages` (pure half of old `render::sanitize_messages`;
  caller supplies decoded `model::special_tokens()` strings); `salvage.rs`
  — `parse_fenced_tool_calls` + `parse_hermes_tool_calls` from the old
  handler, with its unit tests adapted. Crate suite **43/43** (was 28; +10
  filter/sanitize, +5 salvage). SDK: `chat::cue_no_think` added to the
  `sdk/rust/inferlet` re-export list (binding existed since PA.2).
- **Deliberately dropped/changed vs the old handler**: coder-XML salvage
  (`parse_coder_xml_calls`) — out of scope with the Coder dialect, seam
  marked in `turn::salvage`; grammar-forced phase-2 call — already absent
  in the old code (traps guests on drivers without grammar support), seam
  in lib.rs module docs; the wstd HTTP daemon shell — replaced by the
  envelope (gateway owns HTTP/SSE/keepalives now); pre-status degrade macro
  — render faults now answer a clean `{"status":500}` *before* the stream
  commits (the old code had already sent SSE headers by then); degraded
  turns emit `final_content` (`"…"` floor) instead of the old literal
  `" "`; fixed rng seeds from the reference inferlet (deterministic per
  request — revisit if per-request variety matters).
- **Open seams** (marked in-code): KV snapshot sessions
  (`split_resume_point`/`snapshot_address` already tested in
  `pie-openai-serving::session`; attach at `build_prompt` + pre-finish save,
  then report `cached_tokens`); grammar-constrained tool calls (behind a
  capability probe via `tools::format`/`create-matcher`); Qwen3-Coder XML
  dialect (decoder/template model-side + salvage slot).
- Not yet exercised on a live worker — that is PA.3's acceptance suite.

### 2026-08-11 — PA.2 done: gateway OpenAI ingress; parity now TOKEN-EXACT (D1+D4 fixed)

**Renderer parity is fully green: all 5 opencode fixtures token-exact** vs HF
`apply_chat_template(enable_thinking=False)`, Qwen3-0.6B — including the
7473-token tool-history replay (req-005).

- **D1 fixed** — new `cue_no_think()` through the full stack (Instruct trait
  default → `chat.wit` `cue-no-think` (synced) → engine host → qwen_3
  override appending `<think>\n\n</think>\n\n`). The parity bin and the
  serving inferlet use it; plain `cue()` unchanged for thinking-mode callers.
- **D4 fixed** — `assistant_with_tool_calls`/`answer_batch` (and `answer`,
  now the single-element batch) build the turn's inner text as ONE string
  and encode it in ONE pass, matching HF's whole-text BPE segmentation.
  The pre-tokenized tool-call fragments are gone. Unit tests byte-test the
  extracted `*_inner_text` builders (the toy vocab has no BPE merges, so
  token-level fidelity is the parity harness's job — documented in-code).
- **PA.2 done** — `gateway/src/ingress/openai.rs`: `POST /v1/chat/completions`
  + `GET /v1/models` + `GET /health`; Bearer→Identity (blake3-keyed user,
  trust-edge semantics preserved, `x-pie-identity` wins when present); the
  gateway⇄inferlet envelope contract (module docs): first message
  `{"status": u16}`, then verbatim chunk JSON per `data:` line / one unary
  body; pre-stream rejection responds plain JSON, not SSE; launch acks and
  stdout/stderr instrumentation filtered; `[DONE]` on clean Eos, SSE `error`
  event on abort; axum keep-alive comments cover the prefill window.
  6/6 integration tests (`gateway/tests/openai_ingress.rs`) drive the real
  listener with a raw HTTP/1.1 client against an envelope-speaking stub
  worker. Gateway suite overall 40+1+6 green.
- Trap for posterity: `bind()` binds but does NOT serve the client edge —
  call `into_handle()`/`serve()`; a raw client against a bound-only listener
  hangs forever (cost ~40 min of hung background test runs to find).
- Affinity: Ephemeral for Phase A (single worker). Multi-worker sticky
  routing on opencode's `x-session-id` header needs a keyed-affinity variant
  in `gateway/src/session.rs` — deliberately deferred, noted in openai.rs.

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
