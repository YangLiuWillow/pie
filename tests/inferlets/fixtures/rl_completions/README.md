# Phase 1a golden-trace fixtures for the `rl-completions` inferlet

Captured 2026-08-06 per `docs/pie-rl-verl-integration.md` §5 Phase 1a: real
qwen-code episodes driven through the rllm training-mode gateway
(`cumulative_token_mode=ON`, renderer family `qwen3.6`, logprobs +
`return_token_ids` injection) against **stock vLLM** serving
`Qwen/Qwen3.6-27B` BF16 on one A100 SXM (RunPod). Task:
`agronholm__exceptiongroup.0b4f4937.combine_file__74zzufuj` from the
rllm-swesmith 10-task slice, 2 attempts. Capture driver:
`rllm/phase1a_golden_traces.py` (rllm repo root, git-excluded there).

These are the behavioral contract the `inferlets/rl-completions` HTTP daemon
must reproduce (Phase 1b): build it against these before any GPU work, then
re-run the same eval with only the worker URL swapped vLLM→Pie.

## Layout

- `episodes/episode{0,1}.json` — enriched rllm episodes (10 and 13 steps).
  Every step has `prompt_ids`, `response_ids`, `logprobs` (23/23 enriched).
  Token-level append-only holds: step k+1's `prompt_ids` starts with
  step k's `prompt_ids + response_ids` for all 21 consecutive pairs.
- `wire/episode{0,1}/openai-*.json` — qwen-code `--openai-logging` captures:
  the exact chat-completions request/response JSON the CLI saw through the
  gateway (turn 0 = chat path via vLLM parsers; turns 1+ = cumulative
  `/v1/completions` rewrite, tool calls parsed gateway-side by the renderer).
  One tool call per non-final turn; message counts grow strictly (no retries).
- `wire/episode*/qwen-code.log` — CLI stdout for the same rollouts.
- `enrichment_report.json` — the capture run's per-step enrichment audit.

## Contract notes discovered during capture (see rllm/progress.md for detail)

- vLLM's `--tool-call-parser`/`--reasoning-parser` run ONLY on the chat
  endpoint. On the pre-tokenized `/v1/completions` path the server returns raw
  text; the gateway parses tool calls / reasoning via the renderer
  (`rllm-model-gateway/src/rllm_model_gateway/proxy.py::_parsed_chat_message`).
  The Pie inferlet serves `/v1/completions`, so it inherits the same division
  of labor: return raw completion tokens + logprobs, parsing stays gateway-side.
- Tool calls with a usable name+arguments are forwarded even when schema
  validation fails (e.g. the model hallucinating an undeclared `glob` tool —
  visible in these very traces); the client's "tool not found" reply is
  training signal. Dropping them causes contentless-turn retries.
- CAVEAT — raw vLLM `/v1/completions` envelopes are NOT in this set. The wire
  captures here are client-side and sanitized (the gateway strips vLLM-only
  fields like `token_ids` before responding), and the gateway sqlite that held
  the raw pairs is deleted post-run by design. The token-level contract
  (`prompt_token_ids` echo, `choices[0].token_ids`, `logprobs.token_logprobs`)
  is evidenced indirectly through the enriched episodes (per-step token ids +
  logprobs came from those fields). Capture raw envelopes during the 0.6B run
  by snapshotting the gateway sqlite mid-run (same watcher pattern as the wire
  captures), or by pointing the capture driver's out dir at a store the engine
  doesn't clean.

The matching small-model fixture set for the Phase 1b CPU swap is at
`../rl_completions_1.7b/` (Qwen3-1.7B — 0.6B was tried and rejected: it never
tool-calls under qwen-code, so the cumulative path is never exercised). That
set also includes per-turn gateway trace records (`raw_envelopes/`) closing
the raw-envelope gap described above.
