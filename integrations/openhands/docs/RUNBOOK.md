# Phase 1 Runbook — run the integration end-to-end by hand

This is the by-hand recipe for the Pie ↔ OpenHands Phase-1 integration. It is layered shortest → most integrated; each layer should pass before the next is worth attempting.

For the higher-level project plan, see [`pie/docs/openhands-integration.md`](../../../docs/openhands-integration.md). For the verified internal API of `openhands-sdk`, see [`SDK_INTERNALS.md`](SDK_INTERNALS.md). For running a real GPU benchmark on Yale YCRC HPC — either the fast HumanEvalFix smoke test or the full SWE-Bench run — see [`GPU_BENCHMARK.md`](GPU_BENCHMARK.md); this runbook's Layers A–F only cover local CPU/Metal plumbing checks.

---

## 0. One-time setup

```bash
# Toolchain — wstd/wit-bindgen need ≥1.87; the macOS default of 1.85 is too old.
rustup toolchain install stable
# pie/rust-toolchain.toml and inferlets/openhands-completion/rust-toolchain.toml
# already pin "stable" for these two builds, so no global default change is needed.

# Build pie-server (~8 min release build, one-time)
cd /Users/yangliu/Desktop/Lin_startup/pie
cargo build --release -p pie-server
# → target/release/pie (53 MB)

# Build the Phase-1 inferlet to wasm32-wasip2 (~35 s)
cd inferlets/openhands-completion
cargo build --target wasm32-wasip2 --release
# → target/wasm32-wasip2/release/openhands_completion.wasm (190 KB)

# Python venv for the integration
cd /Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -e ../../client/python
# → installs openhands-sdk==1.21.1 + pie-client + pie-openhands
```

If `~/.pie/` is wiped and recreated, the directory `~/.pie/auth/` may come back with insecure perms. This is only consulted when `[auth] enabled = true`. The smoke config sets `enabled = false`, so it's a non-issue for everything in this runbook.

---

## Layer A — unit tests (no Pie, no Agent, ~3 s)

Verifies the `PieLLM` class wiring against a mocked `_call_pie`. Catches Pydantic field bugs, wire-contract regressions, and `ModelResponse`/tool-call construction errors.

```bash
cd /Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands
.venv/bin/pytest tests/test_pie_llm.py -v
# expected: 14 passed
```

---

## Layer B — OpenHands E2E with mocked Pie (~3 s)

Real `Agent` + `Conversation` doing one step through `PieLLM` with a scripted response. Catches integration issues with the SDK's retry / format / event pipeline. **This is the most important test for the OpenHands side of the integration** — if it passes, the SDK plumbing is right.

```bash
.venv/bin/pytest tests/test_e2e_openhands_smoke.py -v
# expected: 4 passed
```

---

## Layer C — Pie one-shot via CLI (~25 s, no Python)

Verifies the inferlet itself runs through a real Pie engine. Uses `pie run --path` to boot an in-process engine, install the wasm, launch it once, then tear down. First run downloads the Qwen3-0.6B tokenizer (~few MB) from HuggingFace.

```bash
PIE=/Users/yangliu/Desktop/Lin_startup/pie/target/release/pie
WASM=/Users/yangliu/Desktop/Lin_startup/pie/inferlets/openhands-completion/target/wasm32-wasip2/release/openhands_completion.wasm
MAN=/Users/yangliu/Desktop/Lin_startup/pie/inferlets/openhands-completion/Pie.toml
CFG=/Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands/tests/fixtures/pie_dummy_config.toml

$PIE run --path "$WASM" --manifest "$MAN" --config "$CFG" \
  --input '{"messages":[{"role":"user","content":"hello world"}],"max_tokens":8,"temperature":0.0}'
```

Expected output (single JSON line):

```json
{"text":"唑-hours_mat diet filing Melanie dessa tether","tool_calls":[],"stop_reason":"length","prompt_tokens":2,"tokens_generated":8}
```

The dummy driver returns random token IDs — content is meaningless, **the *shape* is the smoke signal**: `text` exists, `tool_calls` is a (possibly empty) list, `stop_reason` is one of `stop|length|eos|tool_calls`, the token counts are populated.

---

## Layer D — full stack via PieLLM through `pie serve` (~25 s)

Boots `pie serve`, uploads the inferlet via `install_program`, runs `PieLLM.completion()` against it. This is the full Phase-1 wire path.

```bash
cd /Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands
PIE_BIN=/Users/yangliu/Desktop/Lin_startup/pie/target/release/pie \
PIE_WASM=/Users/yangliu/Desktop/Lin_startup/pie/inferlets/openhands-completion/target/wasm32-wasip2/release/openhands_completion.wasm \
PIE_MANIFEST=/Users/yangliu/Desktop/Lin_startup/pie/inferlets/openhands-completion/Pie.toml \
PIE_CONFIG=/Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands/tests/fixtures/pie_dummy_config.toml \
.venv/bin/pytest tests/test_wire_protocol_smoke.py -v -s
# expected: 1 passed
```

If it hangs at "starting pie serve", inspect `/tmp/pie_serve_smoke.log` — the fixture pipes pie's stdout/stderr there.

---

## Ad-hoc interactive run

When you want to poke at it manually (try a different prompt, time things, etc.):

**Terminal 1** — keep the server up:

```bash
PIE=/Users/yangliu/Desktop/Lin_startup/pie/target/release/pie
CFG=/Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands/tests/fixtures/pie_dummy_config.toml
$PIE serve --config "$CFG" --port 8080 --no-auth
# Ctrl+C to stop
```

**Terminal 2** — install once, then drive PieLLM:

```bash
cd /Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands
.venv/bin/python <<'PY'
import asyncio
from openhands.sdk import Message, TextContent
from pie_client import PieClient
from pie_openhands import PieLLM

WASM = "/Users/yangliu/Desktop/Lin_startup/pie/inferlets/openhands-completion/target/wasm32-wasip2/release/openhands_completion.wasm"
MANIFEST = "/Users/yangliu/Desktop/Lin_startup/pie/inferlets/openhands-completion/Pie.toml"

async def install():
    async with PieClient("ws://127.0.0.1:8080") as c:
        await c.authenticate("local-dev")
        await c.install_program(WASM, MANIFEST, force_overwrite=True)

asyncio.run(install())

llm = PieLLM(
    model="default",
    pie_uri="ws://127.0.0.1:8080",
    pie_inferlet="openhands-completion@0.1.0",
    num_retries=1,
)
resp = llm.completion(
    messages=[Message(role="user", content=[TextContent(text="hello")])],
    max_tokens=16,
)
print("assistant text:", resp.message.content[0].text)
print("usage:", resp.raw_response.usage)
PY
```

---

## Going beyond the dummy driver

The dummy driver returns *random* tokens — perfect for plumbing checks, useless for any real test of model behavior. To run a real (small) model on CPU/Metal for development, swap the config:

```toml
# tests/fixtures/pie_portable_config.toml
[auth]
enabled = false

[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "portable"          # ggml CPU/Metal backend
device = ["metal"]         # or ["cpu"]
activation_dtype = "bfloat16"
```

Then re-run any layer above with `PIE_CONFIG=` pointed at the new file.

**First call downloads ~1.2 GB** of weights to the HF cache, so plan accordingly. Chat-template rendering and tool-call formatting now happen inside the inferlet itself (see `inferlets/openhands-completion/src/lib.rs` and `docs/TOOL_CALL_HISTORY_REPLAY_DESIGN.md`), so there's no `pie_render_strategy` to set on the Python side anymore.

When you graduate to H100 for benchmarking, the driver becomes `cuda_native`. Run `pie doctor` to see what's missing on the host (CUDA toolkit, nvidia-smi visibility, etc.).

---

## Known startup gotchas (and the fixes baked into this runbook)

| Symptom | Underlying issue | Fix |
|---|---|---|
| `Failed to load authorized users: ... insecure permissions: 755` | The runtime checks `~/.pie/auth/` against file-mode rules even though it is a directory. The suggested `chmod 600` is not viable for a directory (no `+x` = can't traverse). | Run with `--no-auth` for `pie serve`, or set `[auth] enabled = false` in the config TOML. Both are already in place here. |
| `Invalid manifest: TOML parse error ... unknown variant 'string[]'` | Pie's manifest `[parameters]` section only accepts scalar types (`string`, `int`, `float`, `bool`); array types are not supported. | Drop the array parameter from `Pie.toml`. The inferlet's `serde` Input struct still validates the JSON at runtime. The Phase-1 `Pie.toml` already does this for the `stop` list. |
| `rustc 1.85.1 is not supported ... requires rustc 1.87.0` | Default macOS rustup channel is older than the wstd / wit-bindgen requirement. | `rustup toolchain install stable`; the repo-local `rust-toolchain.toml` files already pin `stable` for the two crates that need it. |
| `Registry returned error 404 Not Found for std/text-completion` | The public registry has stale inferlets that no longer match the current engine WIT. | Submit inferlets from local source via `pie run --path` or `client.install_program(...)` — both bypass the registry. This runbook uses the local path everywhere. |

---

## Layer E — SWE-Bench harness (driving half)

The harness is split into a library (`benchmarks/swe_bench.py`) and a CLI (`benchmarks/run_swe_bench.py`). It produces a **predictions JSONL** compatible with `python -m swebench.harness.run_evaluation` — that grader half is Docker-gated and deferred.

### Hermetic unit tests (~4 s)

```bash
.venv/bin/pytest tests/test_swe_bench_smoke.py -v
# expected: 6 passed, 1 skipped (network test)
```

Covers: deterministic subset selection, backend factory (`pie`/`litellm`/`test`), predictions JSONL schema matches what the grader expects.

### Single-problem end-to-end with TestLLM (~14 s, requires network)

Clones a real repo from SWE-Bench Verified (`psf/requests`), checks out the base commit, runs an OpenHands agent backed by `TestLLM` (scripted to never edit), captures the empty `git diff HEAD`. This is the load-bearing smoke for the *driver*: workspace setup → agent loop → patch capture → JSONL serialization.

```bash
SWE_BENCH_NETWORK=1 .venv/bin/pytest \
  tests/test_swe_bench_smoke.py::test_drive_one_problem_end_to_end -v
# expected: 1 passed
```

### Drive one problem via the CLI

For real iteration, use `benchmarks/run_swe_bench.py` directly. The `test` backend lets you exercise the full pipeline without a model:

```bash
cd /Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands
.venv/bin/python -m benchmarks.run_swe_bench \
  --backend test \
  --instance-id astropy__astropy-12907 \
  --output /tmp/preds.jsonl
cat /tmp/preds.jsonl
```

### Drive against Pie (real)

Boot `pie serve` and install the inferlet first (see Layer D / ad-hoc section). Then:

```bash
.venv/bin/python -m benchmarks.run_swe_bench \
  --backend pie \
  --pie-uri ws://127.0.0.1:8080 \
  --model default \
  --subset-size 50 \
  --output predictions/pie_qwen3.jsonl \
  --label pie+qwen3-coder-32b
```

> ⚠️ With the **dummy** driver this returns random tokens and you will not get useful patches — the dummy is only useful for plumbing checks. Real benchmarking needs a real model behind Pie (CPU/Metal via `portable`, or H100 via `cuda_native`).

### Drive against vanilla vLLM (the baseline)

```bash
# In another terminal, with GPU:
# python -m vllm.entrypoints.openai.api_server \
#   --model Qwen/Qwen3-Coder-32B-Instruct --enable-prefix-caching --port 8000

.venv/bin/python -m benchmarks.run_swe_bench \
  --backend litellm \
  --model openai/Qwen3-Coder-32B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --subset-size 50 \
  --output predictions/vllm_qwen3.jsonl \
  --label vllm+qwen3-coder-32b
```

### Scoring (deferred — needs Docker)

The `swebench` PyPI package provides the grader:

```bash
# NOT YET WIRED — Docker required. Documenting the shape so it's ready when GPU is.
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path predictions/pie_qwen3.jsonl \
  --run_id pie_qwen3_run_001 \
  --max_workers 4
# writes results to ./evaluation_results/<run_id>/
```

Scoring spins up a Docker container per problem to run the test suite — heavy. We defer this until we have a real model producing real patches.

---

## Layer F — Real-model end-to-end on CPU/Metal (Qwen3-0.6B)

The lowest-effort way to validate that **real generation** flows through the full stack without renting a GPU. Uses the portable (ggml) driver with Qwen3-0.6B; first run downloads ~1.2 GB of safetensors.

### Run

```bash
PIE=/Users/yangliu/Desktop/Lin_startup/pie/target/release/pie
CFG=/Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands/tests/fixtures/pie_portable_config.toml
$PIE serve --config "$CFG" --port 8181 --no-auth > /tmp/pie_portable.log 2>&1 &

# Install the inferlet once (run an ad-hoc PieClient script — see "Ad-hoc interactive run" above)
# Then drive a single problem with the SWE-Bench harness:
cd /Users/yangliu/Desktop/Lin_startup/pie/integrations/openhands
.venv/bin/python -m benchmarks.run_swe_bench \
  --backend pie \
  --pie-uri ws://127.0.0.1:8181 \
  --model Qwen/Qwen3-0.6B \
  --pie-request-timeout-s 1800 \
  --instance-id psf__requests-1142 \
  --max-iterations 3 \
  --output /tmp/preds_cpu_smoke.jsonl \
  --label pie+qwen3-0.6b-portable
```

### Expected outcome (and what it tells you)

| What you'll see | What it means |
|---|---|
| ✓ "inferlet installed", PieLLM completion returns in ~30s for a short prompt | The wire, render, and inferlet paths all work with a real model. |
| ✗ `ConversationRunError` / `TimeoutError` after ~10 min | One agent step (≥5k token prefill + decode) overruns the per-request timeout. **This is expected on a 0.6B CPU model** — increase `--pie-request-timeout-s 3600` to push past it, but each iteration will still take many minutes. The bottleneck is throughput, not correctness. |
| Empty `model_patch` in the predictions JSONL with the error captured cleanly under `_metadata.error` | The driver handled the failure gracefully and produced a well-formed prediction row (with `model_patch=""`, which scores as unresolved). |

This is **not** a Phase-1 acceptance result — the acceptance test for Phase 1 is "PieLLM resolved-rate matches the LiteLLM baseline on 50 problems," and a 0.6B model on CPU will not score above 0% on SWE-Bench Verified. The point of Layer F is to confirm the *plumbing* runs end-to-end with a real model behind it, so when GPU access lands the only thing that needs to change is the model config.

### Path from here to real benchmark numbers

1. Get an H100 (or 2×A100) — see project spec §3.10.
2. Replace `tests/fixtures/pie_portable_config.toml` with a `cuda_native` driver config pointing at `Qwen/Qwen3-Coder-32B-Instruct`.
3. Re-run Layer D (wire smoke) and Layer E single-problem to verify.
4. Run the full 50-problem subset:
   ```bash
   .venv/bin/python -m benchmarks.run_swe_bench \
     --backend pie --pie-uri ws://... --model Qwen/Qwen3-Coder-32B-Instruct \
     --subset-size 50 --output predictions/pie_qwen3.jsonl \
     --label pie+qwen3-coder-32b
   ```
5. Run the vanilla-vLLM baseline (`--backend litellm`).
6. Score both with `python -m swebench.harness.run_evaluation`.

---

## Phase 1 acceptance checklist

Tick when each is green. Layers A–F are now exercised locally as of 2026-05-14 (F hits the expected CPU-throughput ceiling).

- [x] Layer A — `tests/test_pie_llm.py` (13 tests)
- [x] Layer B — `tests/test_e2e_openhands_smoke.py` (4 tests, real `Agent`)
- [x] Layer C — `pie run --path` on the inferlet
- [x] Layer D — `tests/test_wire_protocol_smoke.py` (full stack through `pie serve`)
- [x] Layer E — `tests/test_swe_bench_smoke.py` + driver CLI (clones a real repo, runs agent, captures patch)
- [x] Layer F — full stack with real CPU model (Qwen3-0.6B) — agent loop too slow to complete one step, expected
- [ ] 50-problem SWE-Bench run on a real model (gated on GPU)
- [ ] Docker-based scoring (gated on having predictions to score)
