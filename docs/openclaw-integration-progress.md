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
| oc-P0.2 | Real wire captures + fixture bank | pending |
| oc-P0.3 | Renderer parity on OpenClaw fixtures | pending |
| oc-PA.1 | Keyed sticky affinity in gateway (shared w/ opencode track) | pending |
| oc-PA.2 | `extensions/pie/` bundled provider in OpenClaw | pending |
| oc-PA.3 | Acceptance suite + e2e + A/B vs Ollama/llama.cpp | pending |
| oc-PB.1 | Shared session dialect (design + crate) | pending |
| oc-PB.2 | `openclaw-session` inferlet | pending |
| oc-PB.3 | `createStreamFn` WS transport | pending |
| oc-PB.4 | Strategy B measurement | pending |
| oc-PB.5 | Optional depth (grammar calls, speculation, embeddings) | pending |

## Log

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
