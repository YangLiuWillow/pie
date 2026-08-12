# qwen-code v0.21.6 audit for RL harness use

**Date:** 2026-08-05. Clone at `/Users/yangliu/Desktop/Lin_startup/qwen-code` (commit `3235faf`).
**Purpose:** qwen-code is the chosen training agent for the Pie↔rLLM/verl RL project (see `pie-rl-verl-integration.md`). This audit maps its wire protocol, history handling, and nondeterminism sources against the two invariants the training path depends on: (a) append-only history per turn (gateway cumulative-token mode), (b) prompt determinism (KV-cache reuse, content-addressed snapshots).

**Verdict:** usable, but NOT with defaults. All category-(a) hazards are mitigable via settings except two (H3 needs a fork patch or proxy sentinel handling; H1 needs a giant `contextWindowSize`). `--bare --safe-mode` kills most category-(b) noise except git-status-in-system-prompt (fork patch).

---

## 1. Wire protocol (what the `rl-completions` inferlet must serve)

- SDK: official `openai` npm 5.11.0. Auth path `OPENAI_API_KEY`/`OPENAI_BASE_URL`/`OPENAI_MODEL` is first-class (`AuthType.USE_OPENAI`); a neutral gateway hostname never triggers the DashScope-specific behaviors (no `X-DashScope-*` headers, no `metadata.sessionId`, no cache_control markers).
- Endpoints: **only `POST {base}/chat/completions`** on the agent path. No `/models`, no health probe, no startup ping (preconnect skipped for custom base URLs and when `SANDBOX` is set).
- **Streaming always** in `-p` mode, with `stream_options: {include_usage: true}` always sent.
- Request body: `model`, `messages` (system first, string content; assistant carries `tool_calls[]` and echoes `reasoning_content`; tool results as `{role:'tool', tool_call_id, content:[parts]}`), `tools` (OpenAI function schema, name-sorted), `max_tokens` (**always injected**, up to 64K default for unknown models). `temperature`/`top_p` NOT sent unless configured — gateway's session overrides are compatible. Never sends: `parallel_tool_calls`, `seed`, `logprobs`, `n`, `response_format`.
- For models matching `qwen3`: `reasoning_content` is duplicated into a `reasoning` field on the wire. For `^qwen` models with reasoning disabled on non-DashScope endpoints: `chat_template_kwargs: {enable_thinking: false}` is injected (vLLM/SGLang convention — Pie must accept-and-honor or ignore consistently).

### Hard requirements on the backend (things that break)

| Requirement | If violated |
|---|---|
| `finish_reason` present on final chunk (text turns) | `InvalidStreamError(NO_FINISH_REASON)` → 4 retries → turn failure. Tool-call streams are exempt. |
| Non-empty content for text turns | `NO_RESPONSE_TEXT` / `NO_TOOL_RESULT_PROGRESS` → retry loop |
| SSE `Content-Type: text/event-stream` (or x-ndjson) on streaming 200 | `NonSSEResponseError`, no retry |
| Tool-call `id` present + unique | Call and its response silently dropped from all future requests (`cleanOrphanedToolCalls`) |
| Never emit `finish_reason: "error_finish"` | Interpreted as provider throttle → 10×/60s–5min retry loop |
| Accept `max_tokens`, `stream_options` fields | — |
| 400 ⇒ fail-fast; 429/5xx ⇒ retried (up to 7× app-level × 3× SDK) | A gateway returning 500 for client errors gets hammered |
| No usage chunk | Survivable — compaction thresholds degrade to chars/4 estimates |

Retry behavior on connection drops (our weight-update window): `ECONNRESET`/`EPIPE`/etc. classified retryable. If **no output** was delivered → pure replay (fine). If partial text was delivered → **synthetic continuation** (hazard H3 below). Stream idle watchdog: 240 s without a chunk aborts (tune `QWEN_STREAM_IDLE_TIMEOUT_MS`, 0=off) — relevant for slow CPU-driver runs.

## 2. History management — seven mutation mechanisms

Full conversation IS resent each turn (good), but:

- **H1 — Auto-compaction** (`ChatCompressionService`): at ~85% of `contextWindowSize` (or on a context-overflow 400 from the server), replaces the ENTIRE history with an LLM-generated summary — also firing an extra side-call with a completely different message list. **No boolean disable.** Mitigation: `model.generationConfig.contextWindowSize: 100000000` and ensure the gateway/Pie never returns a context-length 400.
- **H2 — Microcompaction**: after **500 000 cumulative chars** of tool output (default) or 60 min idle, old tool results are blanked in place to `[Old tool result content cleared]`. Very reachable in coding rollouts. Disable: `context.clearContextOnIdle.{toolResultsThresholdMinutes: -1, toolResultsTotalCharsThreshold: -1}`.
- **H3 — Transport-cut continuation**: if an SSE stream dies after partial text (exactly a weight-update drop), the retry request appends two synthetic turns (`{role:'model', partial}` + `{role:'user', "The connection dropped mid-response…"}`) that are then absent from the next turn's history — an append-only violation. Same shape for MAX_TOKENS recovery. **Mitigation options:** (a) fork-patch `maxContinuationRetries: 0` + `MAX_OUTPUT_RECOVERY_ATTEMPTS: 0` (`geminiChat.ts:490-523`) forcing pure replay; (b) teach the gateway to detect the sentinel message text and reset the token accumulator for that session; (c) drain in-flight requests before weight swap (Pie-side, reduces frequency but not MAX_TOKENS case). Recommend (a) — we control the harness install; pin a forked/patched package.
- **H4 — Memory-pressure compaction**: forces microcompaction under RSS pressure regardless of settings. Give the sandbox memory headroom.
- **H5/H6 — Retroactive removals**: orphaned tool calls (cancelled/missing-id) and "invalid" (empty) model turns are dropped from all subsequent requests. Deterministic servers avoid triggering these; accumulator resets absorb them when they happen.
- **H7 — `history[0]` replacement** on startup-context refresh / post-compaction restore (fresh date + folder snapshot).
- **H9 — Side queries**: auto-memory recall (fires EVERY user turn by default!), compaction summarization, `web_fetch` summarization, `agent` subagents — separate conversations interleaved on the same session. Kill via `--bare` + `memory.enableManagedAutoMemory: false`. (Gateway note: the trace enricher matches traces to steps positionally — interleaved side-conversations would corrupt enrichment. `--bare` makes this moot.)

## 3. Prompt determinism

- **System prompt ≈ 22 K chars (~5.5 K tokens)**, plus context files (QWEN.md/AGENTS.md), plus **git status + last 5 commits** (H11 — VOLATILE per rollout and after agent commits; no setting; fork-patch `getRecentGitStatus` → null or replace prompt via `QWEN_SYSTEM_MD`), plus auto-memory tail (H13 — killed by `--safe-mode`/isolated `QWEN_HOME`).
- **`history[0]` startup prelude**: today's date, cwd, **full recursive folder-structure snapshot** (H12). Disable: `model.skipStartupContext: true`. Date-rollover mid-session injects a reminder part — long rollouts crossing midnight differ.
- Tool list is name-sorted (deterministic) but **grows mid-session** via `tool_search` deferred-tool reveal and MCP discovery (H14) — prefix-invalidating. `--bare` pins it to `read_file`, `edit`, `notebook_edit`, `run_shell_command`.
- Tool-call example dialect in the system prompt keys on the **client-side model name** (XML for `qwen*-coder`) — pin `QWEN_CODE_TOOL_CALL_STYLE` so gateway model overrides can't cause drift (H15).
- `SANDBOX` env changes ~600 chars of system prompt — keep constant (H16).
- `reasoning_content` persists in history and is echoed back — Pie's replay/renderer must render the same thinking channel or prefixes diverge (H17). Simplest: run with thinking disabled (`enable_thinking: false` injection) and `/no_think`-style templating consistent with the Pie inferlet.

## 4. Loop/termination/limits (defaults)

- Exits when a turn yields no tool calls. `maxSessionTurns` default unlimited — SET IT (`--max-session-turns`). `--max-wall-time`, `--max-tool-calls` exit with code 55.
- Loop detection heuristics OFF by default (`skipLoopDetection: true`) but always-on tier: 5 consecutive identical tool calls aborts; retry events correctly roll back counters.
- Retries: 4 independent layers (SDK 3×; stream-establish 7× exp backoff; mid-stream rate-limit 10× 60s–5min; transport 2× replay + 3× continuation). `QWEN_CODE_UNATTENDED_RETRY` = unbounded — leave OFF.

## 5. Isolation & telemetry

- `QWEN_HOME` + `QWEN_RUNTIME_DIR` redirect ALL `~/.qwen` state — per-rollout isolation lever (harness `build_env` should set both, like claude_code.py's `CLAUDE_CONFIG_DIR`).
- **Alibaba RUM usage beacon ON by default** — `QWEN_USAGE_STATISTICS_ENABLED=false`.
- `--openai-logging` writes exact wire JSON to `<cwd>/logs/openai` — useful ground truth for the Phase 0 parity harness and for debugging enrichment mismatches.
- Project `.env` files are read (keys set only if unset) — harness must `export` all `OPENAI_*` so a task repo's `.env` can't hijack the endpoint.

## 6. Recommended launch configuration (feeds rllm `qwen_code.py` harness update)

```bash
QWEN_HOME=/tmp/qwen-home QWEN_RUNTIME_DIR=/tmp/qwen-runtime \
QWEN_USAGE_STATISTICS_ENABLED=false QWEN_CODE_SKIP_UPDATE_CHECK_ONCE=1 \
QWEN_CODE_DISABLE_PRECONNECT=1 QWEN_DISABLE_AUTO_TITLE=1 \
QWEN_CODE_TOOL_CALL_STYLE=general \
OPENAI_BASE_URL=<gateway> OPENAI_API_KEY=<token> OPENAI_MODEL=<model> \
qwen --yolo --bare --safe-mode --auth-type openai \
     --max-session-turns 60 --max-wall-time 30m --max-tool-calls 400 \
     --chat-recording false -p "<instruction>"
```

Workspace `.qwen/settings.json`:

```jsonc
{
  "model": {
    "skipStartupContext": true,
    "maxToolCallsPerTurn": 0,
    "generationConfig": { "contextWindowSize": 100000000 }
  },
  "context": { "clearContextOnIdle": {
    "toolResultsThresholdMinutes": -1, "toolResultsTotalCharsThreshold": -1 } },
  "memory": { "enableManagedAutoMemory": false, "enableManagedAutoDream": false,
              "enableAutoSkill": false },
  "privacy": { "usageStatisticsEnabled": false },
  "tools": { "truncateToolOutputThreshold": 30000 }
}
```

Fork patches (pin a patched npm package or install from our fork in the harness):
1. `TRANSPORT_STREAM_RETRY_CONFIG.maxContinuationRetries → 0` and `MAX_OUTPUT_RECOVERY_ATTEMPTS → 0` (`packages/core/src/core/geminiChat.ts:490-523`) — replay instead of synthetic continuation (H3).
2. Stub `getRecentGitStatus` → null (`packages/core/src/core/client.ts:953`) — or accept per-rollout system-prompt variance and rely on within-rollout caching only (H11).

## 7. Plan impact (updated 2026-08-06 for the baseline-first restructure)

- **Phase 0 Track A:** bake this profile into rllm's `qwen_code.py` harness (`build_env` + `write_configs`), pin the npm version, decide fork-vs-sentinel for H3; validate the profile in real eval runs against a hosted API — including byte-for-byte diffing of two rollouts' `--openai-logging` captures (§8.1).
- **Phase 1a:** capture golden traces (gateway + `--openai-logging` + raw vLLM pairs) with this profile active — these become the inferlet's fixtures.
- **Phase 1b:** Pie inferlet acceptance tests derived from §1's hard-requirements table (finish_reason, SSE content-type, tool-call ids, error_finish avoidance, 400-vs-500 semantics, no context-length 400s) plus the Phase 1a fixtures.
- **KV-reuse note:** system prompt + tool schemas ≈ stable multi-K-token shared prefix across ALL rollouts of a batch once git-status/startup-context noise is removed — this widens the Phase 4 prefix-reuse win beyond per-episode reuse.

## 8. Cross-references from the OpenHands integration

The completed OpenHands work (`integrations/openhands/`; reuse inventory in `pie-rl-verl-integration.md` §10) supplies precedent and code for three items in this audit:

1. **H11–H16 mitigation precedent — `pie_openhands/tool_desc_invariance.py`.** The OpenHands equivalent of this audit's determinism hazards was a *single* variable byte-run (a cwd path in one tool description), and the measured effect was total loss of cross-conversation snapshot reuse — the second instance still cold-rebuilt. Two rules carried over: strip the per-instance bytes at the harness layer, **identically in both benchmark arms** (so trajectory equivalence holds); and instrument snapshot hit rates from day one — a leaked byte fails silently, everything "works," and the win is just gone. Apply the same monkeypatch-style verification to the launch profile in §6: after applying it, diff two rollouts' `--openai-logging` captures byte-for-byte before trusting any reuse numbers.
2. **XML tool-call decoding — `pie_openhands/qwen3coder_parser.py`.** §3 noted qwen-code prompts `qwen*-coder` models with XML `<function=NAME>` tool-call examples (and has its own client-side XML recovery parser). On the server side, the `rl-completions` inferlet must decode that native XML dialect into OpenAI `tool_calls` with ids — this file is the dependency-free reference port of vLLM's Qwen3CoderToolParser, already extracted for exactly this reuse. Note the interaction with `QWEN_CODE_TOOL_CALL_STYLE=general` in the §6 profile: if we pin the *general* prompt style, verify which format the model actually emits under it before choosing the decode path.
3. **Parity-harness precedent.** The Phase 0 renderer-parity harness (audit §7) follows the OpenHands "fidelity verification before any benchmark" discipline (`docs/OPENHANDS_CODER_SESSION_DESIGN.md`), and its `benchmarks/compare_equivalence.py` is the model for asserting that Pie-served and vLLM-served qwen-code rollouts are behaviorally equivalent before comparing their speed.
