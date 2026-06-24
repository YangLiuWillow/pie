# pie-openhands

Integration between [Pie](../../) and [OpenHands](https://github.com/All-Hands-AI/OpenHands).

See [`pie/docs/openhands-integration.md`](../../docs/openhands-integration.md) for the full spec.

## Status

- [x] Package scaffold
- [ ] PieLLM (Phase 1)
- [ ] openhands-completion inferlet (Phase 1)
- [ ] SWE-Bench harness wiring (Phase 1)
- [ ] PieConversation (Phase 2)
- [ ] openhands-coder-session inferlet (Phase 2)
- [ ] Benchmark writeup (Phase 3)

## Dev setup

```bash
cd pie/integrations/openhands
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
pytest
```
