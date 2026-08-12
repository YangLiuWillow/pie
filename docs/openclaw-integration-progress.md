# OpenClaw ↔ Pie integration — progress log

Companion to `openclaw-integration.md` (two-strategy spec) and
`openclaw-integration-plan.md` (task-level plan). One entry per completed
task, newest first. Worktree: `Lin_startup/pie-openclaw`, branch
`liu/openclaw-integration` (from `liu/opencode-integration` @ `1053a7790`).
Builds: `CARGO_TARGET_DIR=$HOME/Desktop/Lin_startup/pie/target`.

## Task board

| id | task | status |
|---|---|---|
| oc-P0.1 | Source-derived OpenClaw wire audit → AUDIT.md | **done** |
| oc-P0.2 | Real wire captures + fixture bank | **done** (gateway-only side-calls + image fixture pending, see README gaps) |
| oc-P0.3 | Renderer parity on OpenClaw fixtures | **done — all 5 token-exact** |
| oc-PA.1 | Keyed sticky affinity in gateway (shared w/ opencode track) | pending |
| oc-PA.2 | `extensions/pie/` bundled provider in OpenClaw | pending |
| oc-PA.3 | Acceptance suite + e2e + A/B vs Ollama/llama.cpp | pending |
| oc-PB.1 | Shared session dialect (design + crate) | pending |
| oc-PB.2 | `openclaw-session` inferlet | pending |
| oc-PB.3 | `createStreamFn` WS transport | pending |
| oc-PB.4 | Strategy B measurement | pending |
| oc-PB.5 | Optional depth (grammar calls, speculation, embeddings) | pending |

## Log

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
