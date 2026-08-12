# OpenClaw ↔ Pie integration — progress log

Companions: `openclaw-integration.md` (two-strategy spec),
`openclaw-integration-plan.md` (task-level plan), and
`openclaw-integration-handover.md` (**start here on a new machine**: setup,
build flags, the live-run blocker, verification gaps, next steps). One entry
per completed task, newest first. Worktree: `Lin_startup/pie-openclaw`, branch
`liu/openclaw-integration` (from `liu/opencode-integration` @ `1053a7790`).
Builds: `CARGO_TARGET_DIR=$HOME/Desktop/Lin_startup/pie/target`.

## Task board

| id | task | status |
|---|---|---|
| oc-P0.1 | Source-derived OpenClaw wire audit → AUDIT.md | **done** |
| oc-P0.2 | Real wire captures + fixture bank | **done** (gateway-only side-calls + image fixture pending, see README gaps) |
| oc-P0.3 | Renderer parity on OpenClaw fixtures | **done — all 5 token-exact** |
| oc-PA.0 | Audit fixes: keepalive/overflow/fixture-sweep in serving stack | **done** |
| oc-PA.1 | Keyed sticky affinity in gateway (shared w/ opencode track) | **done** |
| oc-PA.2 | `extensions/pie/` bundled provider in OpenClaw | **done** (compile/docs-check pending pnpm install) |
| oc-PA.3 | Acceptance suite + e2e + A/B vs Ollama/llama.cpp | suite + launcher **authored** (29 tests); live run + A/B pending |
| oc-PB.1 | Shared session dialect (design + crate) | pending |
| oc-PB.2 | `openclaw-session` inferlet | pending |
| oc-PB.3 | `createStreamFn` WS transport | pending |
| oc-PB.4 | Strategy B measurement | pending |
| oc-PB.5 | Optional depth (grammar calls, speculation, embeddings) | pending |

## Log

### 2026-08-12 — live run BLOCKED on Metal admission; handover written

Rebuilt the release binary with today's gateway changes — note the build
line: `cargo build --release -p pie-bin --features pie-bin/driver-metal`
(the bin package is `pie-bin`, not `pie`; omitting the driver feature makes
the worker build script panic about `driver-cuda` being Linux-only).

`run_pie_openclaw.sh` never got past boot. Metal's host-side admission
(`driver/metal/src/batch/forward.cpp:855-900`, flat 2 GiB margin + up to a
2 GiB copy window) refused all three profiles tried:

| profile | needs | reclaimable |
|---|---|---|
| 32k ctx / 1024 pages / default row budget | 4.245 GiB | 2.501 GiB |
| 12k ctx / 384 pages / `PIE_METAL_ROW_BUDGET_MB=256` | 2.058 GiB | 2.398 GiB |
| 10k ctx / 320 pages / `PIE_METAL_ROW_BUDGET_MB=192` | 1.839 GiB | 2.486 GiB |

Same class of blocker the opencode track hit, worse here (the 32k profile
doubles the pages). The driver's message notes a wedged prior run can hold
pages until reboot.

Dummy-driver fallback prepared instead (exercises ingress/envelope/inferlet/
SSE without a GPU); `pie doctor` passes on it, but the suite run was
interrupted before finishing, so **no acceptance test has yet met real
server bytes**. Two config traps found while getting there, both recorded in
the handover: `[model.driver]`/`[model.scheduler]` are now
`[driver]`/`[runtime]`, and dummy driver options are flat keys on `[driver]`
(no `options` table) — without `vocab_size`/`arch_name` it tries to read a
`config.json` inside the `.zt` artifact and fails `Not a directory`.

`openclaw-integration-handover.md` written for the machine switch: repo/
branch/remote map, build + venv setup, what is done, the blocker above, the
**five verification gaps** (nothing live-run; keepalive path unproven; keyed
affinity never observed routing; zero OpenClaw-repo gates run;
`context_overflow_body` unwired), and the ordered next steps.

### 2026-08-11 — oc-PA.2 done + oc-PA.3 authored

**PA.2 — `extensions/pie/` in the OpenClaw repo** (branch `liu/pie-provider`
@ `e35577861f4`, off `main`): vllm-pattern bundled provider via
`defineSelfHostedOpenAICompatibleProvider` (id `pie`, default baseUrl
`http://127.0.0.1:8080/v1`, env `PIE_API_KEY`), manifest with
`openAICompletions.supportsStreamingUsage` (the only compat flag the
manifest schema carries — the rest ship as documented config), core
touch-ups (overlay-id allowlist, lean auto-enable set — both existing
hardcoded lists with vllm/lmstudio precedent), `docs/providers/pie.md`
(recommended compat block, `contextWindow` = engine `max_model_len` rule,
`localService` profile), docs nav entry. Deliberately NOT in
`PLUGIN_ART_SLUGS` (no pie.webp; gradient fallback covers it).
**Verification gaps** (openclaw repo has no node_modules; per its AGENTS.md
heavy proof is remote): extension compile, `pnpm docs:check-config-examples`
on pie.md fences (`compat.sendSessionAffinityHeaders` verified present in
`types.models.ts:46`), and the vitest suites — all pending a pnpm install or
CI run when a PR is opened.

**PA.3 — acceptance suite + launcher authored** (`integrations/openclaw/`):
`test_acceptance.py` imports the opencode suite's plumbing + its 22 generic
tests and swaps in 7 OpenClaw-specific ones (fixtures req-003/004/005
replay incl. `content:null` history and lean surface;
`max_completion_tokens`-only budget → `length`; content-part user messages;
empty-delta keepalive chunks with mirrored chunk id; finish_reason ⊆
{stop,length,tool_calls} sweep) — 29 collected. `run_pie_openclaw.sh` =
opencode launcher with `max_model_len 32768` / `total_pages 1024` (parity
measured ~24.2k-token full-surface prompts). `openclaw.json` e2e profile
carries the audit compat block + lean mode. Live run pending the same
Metal-admission RAM constraint as the opencode track (worse here: double
the pages).

### 2026-08-11 — oc-PA.0 + oc-PA.1 done: audit fixes + keyed affinity

**PA.0** (`gateway/src/ingress/openai.rs`, `inferlets/openai-serving/`):
- Keepalive rebuilt per D-1: axum's comment `KeepAlive` removed; the gateway
  now injects an **empty-delta chunk** after 15 s of inferlet silence
  (< OpenClaw's 60 s cron cap), mirroring the stream's chunk id/model from
  the role chunk so opencode's chunk-id-consistency invariant holds. The one
  documented envelope exception where the gateway authors chunk JSON.
- `error::context_overflow_body(max, requested)` — the 400 body whose wording
  OpenClaw classifies as `context_overflow` (triggers compaction, not
  failure); veto-word test pins it. Wiring pre-generation is a seam until a
  WIT context-capacity getter exists; until then the guard is catalog
  `contextWindow` = engine `max_model_len` (PA.2 requirement).
- `sse_ping` demoted to non-keepalive; streaming module docs corrected.
- Serving crate now sweeps the openclaw fixtures: parse + plan_render on all
  5, asserting the divergent shapes (D-3 null content, D-4
  max_completion_tokens, S-2 strict:false). Crate tests 43→45.

**PA.1** (`gateway/src/session.rs`, `openai.rs`): new `Affinity::Keyed(u64)`
— stable HRW on an external client-session key. The OpenAI ingress derives
it: `x-session-affinity` → `x-session-id` → `session_id` headers →
`prompt_cache_key` body field (blake3→u64); no signal ⇒ Ephemeral/p2c as
before (deliberately NOT hash-of-prompt — would herd multi-tenant load; note
in module docs). Tests: extraction priority/stability unit test; session
test pinning Keyed(k) → dispatch key verbatim across two sessions. Gateway
suite 43+1+6 green.

### 2026-08-11 — oc-P0.3 done: renderer parity — ALL 5 OpenClaw fixtures token-exact

The opencode parity harness (`integrations/opencode/parity/`, unmodified —
`--fixtures-dir` pointed at the openclaw wire captures) vs HF
`apply_chat_template(enable_thinking=False)`, Qwen/Qwen3-0.6B,
transformers 5.15.0 (+ jinja2, now a known venv dep):

```
req-002 24133 tok | req-003 24139 | req-004 24236 | req-005 8642 | req-006 8687 — all exact
```

The D1–D4 fixes from the opencode track generalize with **zero new divergence
classes** at 3× the prompt length: the 34-tool/56 KB schema block (nested
unions, big maximums, `strict: false` keys) survives the `python_json`
formatter, and the history-replay turn (assistant `content: null` +
normalized tool-call id + string tool result) renders exactly.

Sizing consequence for oc-PA.3: full-surface OpenClaw prompts render ≈24.2k
tokens (lean ≈8.7k) — the live-serve config needs `max_model_len ≥ 32768`
(1024 pages × 32), double the opencode profile's 16384.

**Phase 0 complete.** Next: oc-PA.1 keyed sticky affinity (gateway),
oc-PA.2 `extensions/pie/` in OpenClaw, oc-PA.3 acceptance + e2e + A/B.

### 2026-08-11 — oc-P0.2 done: real wire captures banked

`tests/inferlets/fixtures/openclaw/wire/req-002..006.json` captured from the
published CLI (`openclaw@2026.7.1-2` via npx, Node 24.19.0 via nvm, SDK 6.45.0
on the wire) against the adapted recorder: plain turn (34 tools / 56 KB,
33 KB system prompt), tool-call turn, **history replay** (assistant
`content: null` + `tool_calls` + `role:"tool"` string result), and two
lean-mode turns (4-tool surface). Driver: `openclaw agent --local
--session-key … --model pie/test-model` with `OPENCLAW_CONFIG_PATH` +
`OPENCLAW_STATE_DIR` isolation (`integrations/openclaw/openclaw.capture.json`;
the repo CLI's `agent exec --config` flags don't exist on the npm release yet).

AUDIT.md upgraded: §8 records capture-verified rows and the **version-skew
table S-1..S-6** — the npm client sends user content as a plain *string* with a
timestamp envelope prefix (repo HEAD sends part arrays → pie must accept both),
puts `strict: false` on every tool, ships a 34-tool default / 4-tool lean
surface, and normalizes tool-call ids (`call_record_001`→`callrecord001`, so
ids must never be assumed to round-trip). Recorder hardened: in tool mode it
now only synthesizes calls for read-like tools (the lean run showed it would
otherwise pick `exec` and run the placeholder as a real command).

Known gaps (README): gateway-only side-calls (title/heartbeat/compaction),
image-part fixture, and repo-HEAD (`2026.8.1`) recapture once released.

### 2026-08-11 — oc-P0.1 done: source-derived wire audit

`tests/inferlets/fixtures/openclaw/AUDIT.md` written from four parallel source
sweeps of the OpenClaw repo (streaming/timeouts, request body, retry/errors/
headers, tools/extra-calls); every claim `file:line`-cited; all rows
source-derived pending oc-P0.2 capture upgrades. Load-bearing findings:

- **Keepalive divergence (D-1)**: OpenClaw's watchdogs reset only on parsed
  chunks and its SSE sanitizer *drops* comment-only frames — the opencode-style
  `: ping` keepalive is invisible. Ingress must switch to empty-delta chunks
  (`choices:[{index:0,delta:{}}]`), which are also safe for opencode ⇒ make it
  universal. Defaults: first-event 120 s (300 s self-hosted), idle 120 s
  (disabled for loopback baseUrls), cron-trigger cap 60 s.
- **Body-shape divergences**: user content is a parts array (D-2); assistant
  tool-call replay has `content: null` vs opencode's `""` (D-3); default
  max-tokens field is `max_completion_tokens` (D-4); `stream_options` only for
  loopback endpoint class (D-5).
- **Strictness moved**: first tool-call delta may lack id/name (D-8), but
  finalization requires args parsing to an object (`"{}"` floor) and
  `finish_reason:"tool_calls"` when text preceded calls; unknown finish_reason
  strings fail the whole turn (D-9); `[DONE]` needed for promotion.
- **Bounded retries** (D-6) unlike opencode; 400s must carry a body; context
  overflow must be a 400 whose message matches the overflow tables (recipe in
  AUDIT §3) with rate-limit wording vetoing the match.
- **Session signal**: no session headers by default — `prompt_cache_key`
  (= `sessionId:boundaryCount`, opt-in compat) is the only affinity key;
  `compat.sendSessionAffinityHeaders: true` upgrades to real headers (oc-PA.1
  affinity design: key on either).
- **Tool surface**: ~40 tools ≈50–120 KB default, 9 in lean mode; name-sorted
  every request; **mid-session churn** on heartbeat (adds `heartbeat_respond`)
  and memory-flush (collapses to `read`+`write`) turns — Strategy B dialect
  needs a per-turn tools digest.
- **Extra calls**: compaction summaries, memory flush, heartbeats, titles,
  narration all hit the same provider by default (AUDIT §5 table) — Strategy B
  must route them to the plain completions path, off the session working set.
- `<think>` tags are stripped client-side; reasoning must stream as
  `reasoning_content` and the model entry needs `reasoning: true` (D-15).

Also: capture rig prepared — `record_server.py` adapted (schema-driven tool-arg
synthesis), headless driver identified (`openclaw agent exec --config … --json`;
`--local-model-lean` variant), Node 24.19.0 installed via nvm for the published
CLI (`openclaw@2026.7.1-2` vs repo `2026.8.1` — skew to note on captures).

### 2026-08-11 — project setup
- Codebase surveys completed (OpenClaw provider layer; pie dev serving
  surface); two-strategy spec written (`openclaw-integration.md`), reusing the
  opencode branch's infrastructure per its progress log.
- Worktree `pie-openclaw` created on new branch `liu/openclaw-integration`
  from `liu/opencode-integration` @ `1053a7790`.
- Detailed plan (`openclaw-integration-plan.md`) + this log committed.
