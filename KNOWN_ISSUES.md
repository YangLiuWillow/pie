# Known engine issues found during rl-completions bring-up (2026-08-06/07)

Branch context: `liu/rl-completions` — Phase 1b of the pie↔rllm RL integration.
Each issue below reproduces on stock code from this branch's base (main @ e22915ca)
unless noted. Ordered by severity.

## 1. Qwen3-1.7B generation is broken on the portable driver

**Repro (stock code, ~80s):**
```
python3 tests/inferlets/test_text_completion.py --driver portable \
    --model Qwen/Qwen3-1.7B --device cpu
```
Fails with `GenStep::execute forward: ForwardPass::execute: empty input` (the
message comes from this branch's new API guard, commit c1b32675; on unguarded
main it is an infinite silent retry loop instead). Underlying behavior: the
driver's forward pass completes without error but returns **zero sampled
tokens**, so the SDK generator has nothing to feed the next step.

Established by A/B elimination:
- Qwen3-0.6B works everywhere (tested to 13.5k-token prompts, both samplers,
  string and token-id prompts, streaming and not).
- Qwen3-1.7B fails on 6-token prompts, Argmax and TopP, both prompt forms,
  on an 8GB M2 Mac AND a 96-CPU/503GB RunPod host (rules out memory), with
  scheduler `request_timeout_secs` raised (rules out timeouts).
- `helloworld` passing is NOT a counter-signal — it never touches the model.

Suspected area: per-model config/graph handling in
`driver/portable/src/graph_qwen3*.cpp` (0.6B: hidden 1024; 1.7B: hidden 2048;
q/k head layouts differ). Next step: instrument the sampler/output path for
1.7B and diff against 0.6B; optionally bracket with Qwen3-4B.

## 2. "Silent empty" failure family — errors converted to empty successes

Four sightings of the same pattern, each of which cost real debugging time
because the failure surfaced far from its cause:

1. Scheduler abandons any request exceeding `request_timeout_secs` (default
   120s) and returns an EMPTY output as if successful → contexts with holes.
2. `FutureOutput::ready` maps upstream errors to `finish_empty()` (empty
   output), not an error (openhands branch; main behaves equivalently).
3. Empty-input forwards were never rejected at the API layer (fixed on this
   branch, commit c1b32675 — the check "fix #6" that `empty-forward-test`
   describes had never actually been implemented anywhere).
4. Issue #1 above: a forward that samples nothing reports success.

Proposed principle: a forward that cannot deliver what was requested should
ERROR, visibly, at the API boundary. Empty-success responses poison KV state
and surface as unrelated errors many steps later.

## 3. Portable driver Metal backend: silent SIGKILL at model load (macOS)

`--device auto` (Metal engages; note `cpu` force-disables it even when
compiled in) kills the process at model load with no traceback, no crash
report — Qwen3-0.6B and 1.7B alike, independent of free memory. Portable CPU
on the same build works. Wheel built with `PIE_PORTABLE_METAL=1` on an M2,
macOS (Darwin 25.4).

## 4. pie_driver_dev is Linux-only but fails on Linux pod too

- macOS: `shmem_ipc.py:65` hard-codes `ctypes.CDLL("librt.so.1")` → dev
  driver cannot start at all on Mac.
- Linux (RunPod, torch 2.4.1+cu124): worker subprocess dies with exit 1
  before ready, no traceback surfaced by the launcher. Not yet diagnosed.

## 5. Build/tooling paper cuts

- `sdk/python-server/Cargo.toml` hard-requires the Linux-only `driver-cuda`
  feature → maturin build cannot succeed on macOS at all (worked around on
  this branch, commit 291db390; needs a target gate).
- Portable driver needs cmake ≥ 3.23; Ubuntu 22.04 ships 3.22 (`pip install
  cmake` suffices).
- `pie-server` build needs `libssl-dev` (undeclared).
- `pie_driver_dev` needs `torch` and `torchao` (undeclared deps).
- Scheduler `request_timeout_secs` was not exposed to the test harness
  (added `--request-timeout-secs` to `tests/inferlets/conftest.py` on this
  branch).
