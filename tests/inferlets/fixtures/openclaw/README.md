# OpenClaw wire fixtures (oc-P0.2)

Real traffic from the **published** OpenClaw CLI against `record_server.py`,
plus the source-derived audit in `AUDIT.md` (repo-HEAD cross-reference).

**Capture client:** `openclaw@2026.7.1-2` via `npx` under Node `24.19.0`
(bundles `openai` SDK **6.45.0** per the wire User-Agent).
**Source cross-ref:** repo checkout `2026.8.1` (`openai` 6.49.0 pinned).
The skew is real and visible on the wire — divergences are listed in
`AUDIT.md` §8; treat the fixtures as ground truth for the shipped client and
the audit as ground truth for repo HEAD, and make pie tolerate both.

## Inventory

| File | What it is |
|---|---|
| `wire/req-002.json` | plain text turn, full default surface: system (33 KB, 3-newline boundary artifact) + user (**plain string** with `[<weekday> <date> <time> <TZ>]` envelope prefix); **34 tools, 56 KB serialized, name-sorted, `strict: false` on each**; `tool_choice:"auto"`, `max_completion_tokens:32000`, `stream_options:{include_usage:true}` |
| `wire/req-003.json` | first request of the tool run (same shape as req-002, different prompt) — recorder answered with a `read` tool call (`arguments: {"path": ...}` — OpenClaw's read arg is `path`, not opencode's `filePath`) |
| `wire/req-004.json` | **history replay**: system, user, assistant with **`content: null`** + `tool_calls` (id normalized by the client: `call_record_001` → `callrecord001` — ids do not round-trip verbatim), then `{"role":"tool","content":"<string>","tool_call_id":...}` |
| `wire/req-005.json` | lean-mode (`agents.defaults.experimental.localModelLean`) first request: **4 tools** (`exec, tool_call, tool_describe, tool_search`; repo HEAD would show 9) — recorder (still in tool mode) answered with a synthetic `exec` call that did not round-trip |
| `wire/req-006.json` | lean-mode retry of the same turn, completed as plain text |

Response side (what the recorder served and the client accepted): SSE with a
leading `: keepalive` comment, role delta, content / tool-call deltas
(name-first then arguments), `finish_reason` chunk, usage-only chunk
(`choices: []`, `prompt_tokens_details.cached_tokens`), `data: [DONE]`.
NOTE: acceptance of the comment line here does NOT validate comments as
keepalive — repo HEAD's sanitizer drops comment-only frames and its watchdogs
reset only on parsed chunks (AUDIT §2a). Keepalives must be empty-delta chunks.

## Re-running the capture

```bash
SCRATCH=<workdir>; mkdir -p $SCRATCH/capture-ws $SCRATCH/openclaw-state
echo "hello from the capture workspace" > $SCRATCH/capture-ws/hello.txt

# 1. recorder (this dir): text or tool mode
RECORD_MODE=tool RECORD_TOOL_FILE=$SCRATCH/capture-ws/hello.txt python3 record_server.py &

# 2. drive turns with the published CLI (needs Node >=24; nvm install 24)
export OPENCLAW_CONFIG_PATH=../../../../integrations/openclaw/openclaw.capture.json
export OPENCLAW_STATE_DIR=$SCRATCH/openclaw-state
npx -y openclaw@latest agent --local --session-key capture-1 \
  -m "Read the file hello.txt in the workspace and tell me its contents." \
  --model pie/test-model --json
```

The capture config (`integrations/openclaw/openclaw.capture.json`) pins the
workspace and declares provider `pie` → `http://127.0.0.1:8123/v1`. The repo
version of the CLI (`2026.8.1`) replaces `agent --local` with
`openclaw agent exec "<msg>" --config <path>`; recapture with that once the
npm release catches up, and re-verify AUDIT §8's skew rows.

Known capture gaps (acceptable for Phase A; close opportunistically):
- no dashboard-title / heartbeat / compaction side-call fixtures (need a
  gateway session, not `agent --local`);
- no image-part fixture (needs a vision-capable model entry + attachment);
- repo-HEAD (`2026.8.1`) fixtures pending an npm release or a monorepo build.
