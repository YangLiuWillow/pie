# Phase 1b small-model fixtures (Qwen3-1.7B) for the `rl-completions` inferlet

Captured 2026-08-06, same driver and pipeline as `../rl_completions/` (the
Qwen3.6-27B contract set — read that README first): rllm training-mode gateway
(`cumulative_token_mode=ON`, renderer family `qwen3`, logprobs + token-ids
injection) against stock vLLM serving **Qwen/Qwen3-1.7B** (A40, `hermes`
tool parser for the turn-0 chat path). Same swesmith task, 2 attempts.

This is the CPU-swap set: Phase 1b develops the Pie inferlet against these,
then re-runs the same eval with the worker URL flipped vLLM→Pie on CPU, where
a 27B is impractical. **Qwen3-0.6B was tried first and rejected**: it never
emits a tool call under qwen-code's prompt, so episodes end after one
chat-path turn and the cumulative `/v1/completions` path is never exercised.
1.7B tool-calls reliably (66 turns captured; one rollout ran to the 60-turn
session cap — noisy agent behavior, but rich fixture traffic).

## Layout

- `episodes/episode{0,1}.json` — enriched episodes (6 and 60 steps, 66/66
  steps with `prompt_ids`/`response_ids`/`logprobs`; token-level append-only
  verified 5/5 and 59/59 consecutive pairs).
- `raw_envelopes/*.json` — **gateway trace records, one per turn** (66),
  snapshotted from the trace sqlite mid-run (the engine deletes sessions
  post-run). Each carries the vLLM-extracted contract payload: root
  `prompt_token_ids`, `completion_token_ids`, per-token `logprobs`,
  `finish_reason`, plus the chat-form `raw_request` (messages + tools).
  Note the literal `/v1/completions` request/response JSON bodies are not
  retained by the gateway — but the completions prompt equals
  `prompt_token_ids` by construction, so the (token ids in → token ids +
  logprobs out) contract is fully specified. These records are what the
  inferlet's contract tests should assert against.
- `wire_6req/`, `wire_60req/` — client-side qwen-code `--openai-logging`
  captures per rollout (suffix = request count, matching episode step counts)
  plus CLI stdout logs.
- `enrichment_report.json` — the capture run's enrichment audit.
