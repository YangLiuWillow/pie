# pie-openhands

Integration between [Pie](../../) and [OpenHands](https://github.com/All-Hands-AI/OpenHands):
run the OpenHands coding agent against a model served locally by Pie, with a
content-addressed KV prefix cache that keeps the agent from re-prefilling its
conversation on every turn.

**New here? Start with [`docs/TUTORIAL_CODER_SESSION.md`](docs/TUTORIAL_CODER_SESSION.md)** —
a walkthrough from an empty machine to the agent editing your own repo, assuming
no prior knowledge of either project.

## Status

Working end-to-end and benchmarked.

- [x] Package scaffold
- [x] PieLLM (Phase 1)
- [x] `openhands-completion` inferlet (Phase 1)
- [x] SWE-Bench harness wiring (Phase 1)
- [x] `openhands-coder-session` inferlet with self-keyed prefix cache (Phase 2)
- [x] Benchmark writeup (Phase 3) — [`PIE_VS_VLLM_EVALUATION.md`](PIE_VS_VLLM_EVALUATION.md)

Measured on 13 SWE-bench-Verified instances with Qwen3-Coder-30B-A3B, against
the same agent driven by litellm + vLLM: **~26 % faster** (1,430 s vs 1,939 s),
95 %+ KV reuse, zero cache-verification errors. Accuracy on that set is 11/13
against the baseline's 13/13 — but the set *is* the baseline's own resolved
instances, so parity is the ceiling and it cannot show a difference either way;
a neutral run is in flight. `PieConversation` was never needed: the prefix
cache made it redundant.

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/TUTORIAL_CODER_SESSION.md`](docs/TUTORIAL_CODER_SESSION.md) | **Start here.** Build → serve → run the agent on your repo |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Layered by-hand verification, shortest to most integrated |
| [`docs/OPENHANDS_CODER_SESSION_DESIGN.md`](docs/OPENHANDS_CODER_SESSION_DESIGN.md) | Design of the session/prefix-cache protocol |
| [`docs/SDK_INTERNALS.md`](docs/SDK_INTERNALS.md) | Verified internals of `openhands-sdk` |
| [`PIE_VS_VLLM_EVALUATION.md`](PIE_VS_VLLM_EVALUATION.md) | Results vs vLLM + APC, including where Pie loses |

## Dev setup

```bash
cd pie/integrations/openhands
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -e ../../client/python   # pie_client
.venv/bin/pytest -q                            # ~97 passed
```

The tests need no GPU. For a GPU-free check of the cache protocol itself, run
`session_smoke.py` against the dummy driver — see the tutorial's §7.
