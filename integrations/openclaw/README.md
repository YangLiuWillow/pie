# OpenClaw ↔ Pie integration harness (oc-PA.3)

Companions: `../../docs/openclaw-integration.md` (two-strategy spec),
`openclaw-integration-plan.md` (tasks), `openclaw-integration-progress.md`
(log), `../../tests/inferlets/fixtures/openclaw/AUDIT.md` (wire contract).

## Files

| File | Purpose |
|---|---|
| `test_acceptance.py` | Live acceptance suite: imports the opencode suite's plumbing + generic wire tests, swaps in the OpenClaw fixtures and client policy (empty-delta keepalives D-1, mapped finish_reason set D-9, `max_completion_tokens` D-4, content-part tolerance D-2, `content:null` replay D-3). `--collect-only` / `--only <substr>`; env `PIE_BASE_URL` etc. per its docstring. |
| `run_pie_openclaw.sh` | Boot `pie serve` (max_model_len **32768** — full-surface OpenClaw prompts render ≈24.2k tokens), health-wait, run the suite, shut down. `--serve-only` keeps it up for the e2e. |
| `openclaw.json` | Stock-OpenClaw e2e profile: provider `pie` at `http://127.0.0.1:8080/v1`, model `pie/qwen3-0.6b` with the audit's compat block (`supportsPromptCacheKey` + `sendSessionAffinityHeaders` — feeds pie's keyed sticky affinity), `contextWindow` = engine `max_model_len`, lean mode on (0.6B). |
| `openclaw.capture.json` | The oc-P0.2 **capture** profile (recorder on :8123) — not for live serving. |

## Acceptance

```bash
./run_pie_openclaw.sh            # boot + suite + shutdown
# or against an already-running server:
PIE_BASE_URL=http://127.0.0.1:8080 python3 test_acceptance.py
```

Wire-SHAPE assertions are hard; 0.6B model-BEHAVIOR expectations are soft
(`[WARN]`). Green means: the gateway/inferlet honor every hard rule in
AUDIT.md §7 that is testable without a client, on real wire bytes.

## Stock-OpenClaw e2e

```bash
./run_pie_openclaw.sh --serve-only
# separate shell (published CLI needs Node >= 24; nvm install 24):
OPENCLAW_CONFIG_PATH=$PWD/openclaw.json OPENCLAW_STATE_DIR=$(mktemp -d) \
  npx -y openclaw@latest agent --local --session-key e2e-1 \
  -m "Read the file README.md in the workspace and tell me its contents." \
  --model pie/qwen3-0.6b --json
```

(The repo CLI `2026.8.1` replaces `agent --local` with
`openclaw agent exec "<msg>" --config <path>`.) A/B protocol vs
Ollama/llama.cpp per the plan's oc-PA.3: wall clock, prompt tokens,
`prompt_tokens_details.cached_tokens`, trajectory match.

## Known blockers

Same machine constraint as the opencode track (see its progress log):
`pie serve` needs ~3.2 GiB reclaimable RAM for the Metal heap — and this
profile doubles `total_pages` (1024) for the 32k context, so headroom needs
are higher still. Mitigations: free RAM, `PIE_METAL_ROW_BUDGET_MB=512`
(don't go low enough to refuse the ~24k-token prompt).
