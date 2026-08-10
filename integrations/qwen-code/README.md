# qwen-code ↔ Pie integration

Serve **stock qwen-code** from Pie with zero client-side changes: the
`chat-completions` inferlet (`inferlets/chat-completions/`) is an
OpenAI-compatible `/v1/chat/completions` SSE daemon with content-addressed
KV-session reuse. qwen-code just gets `OPENAI_BASE_URL` pointed at it.

Design and rationale: `docs/qwen-code-integration-plan.md`. Wire contract and
hazard catalogue: `docs/qwen-code-rl-audit.md`. Structural precedent:
`docs/codex-integration.md` (branch `liu/codex-integration`).

## Architecture

```
qwen-code (Node, unmodified, tools run locally)
    │  POST /v1/chat/completions  (SSE, full history each turn)
    ▼
chat-completions inferlet  ── daemon on :8123, fresh WASM instance/request
    │  render history → resume KV snapshot (hash of message prefix)
    │  → prefill only the new suffix → generate → decode tool calls
    ▼
pie serve :18080 (engine; snapshots live here, named qwenchat/<hash>)
```

Per turn the inferlet hashes the message prefix up to the last assistant
turn; the previous request saved its post-generation KV under exactly that
name, so the follow-up pays prefill only for the new tool results + user
turn. Reuse is reported in `usage.prompt_tokens_details.cached_tokens`.
A miss is always safe (full rebuild).

## Quick start

```bash
# 1. one-time: build the inferlet
cd inferlets/chat-completions && cargo build --target wasm32-wasip2 --release

# 2. boot pie serve + daemon (foreground; Ctrl-C tears down)
integrations/qwen-code/run_pie_qwen.sh          # portable Qwen3-0.6B config

# 3. verify the wire contract (audit §1 hard-requirements table)
python3 integrations/qwen-code/test_acceptance.py

# 4. run qwen-code against it — the launch profile is printed by step 2
```

## Hazard coverage (from the audit)

Mitigated server-side: unique tool-call ids (`instance_id()`-derived — the
fresh-instance-per-request `call_0` bug), 400-vs-500 semantics, no
`error_finish`, empty-delta keepalives during long prefills, non-empty
content on text turns (phase-2 forced tool call + fenced-JSON fallback,
ported from `openhands-completion`), generation faults degraded to
`finish_reason:"length"` (never a context-length 400 → H1 stays dormant),
`enable_thinking:false` honored via position-independent `/no_think`
user-turn decoration (H17), special-token stripping on round-tripped
content.

Mitigated by the launch profile (§6 of the audit): compaction H1/H2,
side-query interleaving H9, startup-context volatility H12/H13, tool-list
growth H14, dialect pinning H15.

Accepted / out of scope here: H3 synthetic continuation on dropped streams
and H11 git-status-in-system-prompt need qwen-code fork patches; H11 only
costs *cross-rollout* prefix reuse — within-session reuse is unaffected.

Snapshot retention: `Context::take`-on-hit keeps ≤1 live snapshot per
conversation branch, but abandoned sessions leak their last snapshot until
engine-side GC — acceptable for dev loops, revisit with the snapshot-
retention work if daemons run long. Under a near-full page pool, save-time
seal+flush can hit the engine's exact-page-boundary reserve defect
(`KV_INVARIANT_VIOLATION`, `DEFECTS_OVERCOMMIT.md` defect 2); the save
fails non-fatally and the next turn rebuilds.

No grammar-forced tool calls: constrained decoding traps the guest on the
portable driver (kills the SSE stream pre-finish → qwen-code retry storm),
so the inferlet relies on native `<tool_call>` decoding plus the fenced-JSON
fallback instead. Status (2026-08-10): 12/12 unit, 33/33 acceptance, e2e
with stock qwen-code v0.21.6 green (native tool call executed, 99.7% KV
reuse on tool-result turns).

## Files

- `run_pie_qwen.sh` — boots the stack, prints the qwen-code launch profile
- `launch_daemon.py` — install + `launch_daemon` via `pie_client`
- `test_acceptance.py` — raw-HTTP assertions of the audit §1 table, plus a
  two-turn echo-back conversation asserting `cached_tokens > 0`
