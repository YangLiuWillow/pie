# Running the SWE-Bench GPU benchmark (Yale YCRC HPC)

This is the recipe for driving a real SWE-Bench run against a GPU-backed model,
either through Pie (`pie serve` + `pie-driver-vllm`) or the vanilla LiteLLM
baseline (direct vLLM `api_server`). It picks up where
[`RUNBOOK.md`](RUNBOOK.md) Layers A–F leave off — those layers validate the
plumbing on CPU/Metal with dummy or tiny models; this doc is for the actual
GPU/HPC path used for Phase 1 numbers.

For the code-level status of this work (what's fixed, what's still an open
problem), see the project memory / prior session notes — the short version:
tool-calling plumbing is fixed and verified end-to-end on GPU, but neither
backend reliably produces a clean patch yet due to a model-capability +
upstream-SDK-stuck-detector gap (see "Known limitations" below).

---

## 0. One-time setup (already done on this cluster, documented for a fresh checkout)

```bash
cd /nfs/roberts/project/pi_ql324/ly337/pie

# Build pie-server
cargo build --release -p pie-server
# -> target/release/pie

# Build the Phase-1 inferlet
cd inferlets/openhands-completion
cargo build --target wasm32-wasip2 --release
# -> target/wasm32-wasip2/release/openhands_completion.wasm

# Python venv for the harness/SDK side
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands
python3 -m venv .venv
PYTHONPATH="" .venv/bin/pip install -e '.[dev]'
PYTHONPATH="" .venv/bin/pip install -e ../../client/python

# Shared vLLM driver venv (separate from the harness venv above — this is
# what pie-server's vllm driver subprocess actually runs in)
cd /nfs/roberts/project/pi_ql324/ly337/pie/driver/vllm
PYTHONPATH="" UV_PROJECT_ENVIRONMENT=/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env \
  uv sync --active --no-build-isolation-package pie-driver-vllm \
           --no-build-isolation-package pie-driver-dev
```

**Cluster gotcha — always clear `PYTHONPATH`.** The OOD-Jupyter / module
environment injects a pile of `/apps/software/.../site-packages` paths that
silently shadow venv packages (this is how a stale `packaging==24.1` got
picked up over a venv's newer copy once). Prefix every `uv`/`pip`/`python`
invocation against either venv with `PYTHONPATH=""`. `run_pie_backend.sh` and
`run_litellm_baseline.sh` already do this.

Do **not** use `pie driver vllm install --run` for this — that CLI recipe uses
plain `uv pip install` and doesn't honor the `[tool.uv.sources]` custom index
needed for `flashinfer-jit-cache`. Use `uv sync` from `driver/vllm/` as above.

---

## 1. Get a GPU

Real model runs need their own Slurm allocation — **not** the OOD-Jupyter
notebook's own interactive session (that's capped at 5GB system RAM regardless
of GPU, which OOM-kills the vLLM worker at the host level independent of GPU
VRAM headroom).

| Partition | Account | QOS | GPU flag | Notes |
|---|---|---|---|---|
| `gpu_h200` | `pi_ql324` | `normal` | `--gpus=h200:1` | Fair-share queue, can sit `PD` a while |
| `priority_gpu` | `prio_ql324` | (implied by account) | `--gpus=<type>:1` (`a100`, `rtx_5000_ada`, `h200`, `rtx_pro_6000_blackwell`, `b200`) | Priority tier, faster scheduling, but **only** with the `prio_ql324` account — it rejects `pi_ql324` and vice versa |

Two ready-made sbatch scripts (2-problem smoke test, Pie backend):

```bash
sbatch integrations/openhands/smoke_test.sbatch           # gpu_h200 / pi_ql324 / h200
sbatch integrations/openhands/smoke_test_priority.sbatch  # priority_gpu / prio_ql324 / rtx_5000_ada
```

Logs land in `integrations/openhands/logs/smoke_<jobid>.out`.

For a real multi-hour 50-problem run, write a new sbatch job (or copy one of
the above) that calls `run_pie_backend.sh` / `run_litellm_baseline.sh` with a
longer `--time` — observed per-problem latency is ~250–900s with PieLLM.

---

## 2. Run the Pie backend

```bash
cd /nfs/roberts/project/pi_ql324/ly337/pie/integrations/openhands
bash run_pie_backend.sh --subset-size 2          # or --instance-id <id>, or no args for --subset-size 50
```

This boots `pie serve` (vllm driver, 7B Qwen2.5-Coder by default), waits for
it to report ready (up to 60 min — cold model downloads take a while),
installs the inferlet, then runs the harness. Predictions land in
`integrations/openhands/predictions/<prefix>_<timestamp>.jsonl` (gitignored —
these are run outputs, not source).

**Switch to the 32B model** via env vars (all default to the 7B values, so
plain `bash run_pie_backend.sh` is unchanged):

```bash
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
LABEL=pie+qwen2.5-coder-32b \
OUTPUT_PREFIX=pie_qwen25_coder_32b \
bash run_pie_backend.sh --subset-size 2
```

32B fits on one H200 (141GB): ~65GB weights + KV cache at
`gpu_memory_utilization=0.85`. Both models' weights land in the shared
`HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache`.

---

## 3. Run the LiteLLM baseline (for comparison)

```bash
bash run_litellm_baseline.sh --subset-size 2
```

Boots a plain vLLM `api_server` (no Pie) against the same 7B model and runs
the harness through `--backend litellm`. This is the thing PieLLM's
resolved-rate needs to match for Phase 1 acceptance.

Its readiness-wait loop has a known-harmless bug: it greps `/health` for
`"{}"` but vLLM's `/health` returns an empty body, so it always falls through
to the fixed 240s timeout instead of detecting readiness early. Wastes a
couple minutes per run; not worth fixing unless this script sees heavier use.

---

## 4. Useful CLI flags (`benchmarks/run_swe_bench.py`)

Both wrapper scripts pass extra args straight through, so any of these work
appended to `run_pie_backend.sh` / `run_litellm_baseline.sh`, or directly via
`python -m benchmarks.run_swe_bench`:

| Flag | Purpose |
|---|---|
| `--backend {pie,litellm,test}` | Which model backend to drive |
| `--subset-size N` / `--instance-id ID` (repeatable) | Deterministic N-problem subset, or specific instance(s) |
| `--max-iterations N` | Cap agent steps per problem (default 50) |
| `--max-stuck-retries N` | How many times to nudge-and-retry when the SDK's `StuckDetector` fires (default 2) |
| `--native-tool-calling` / `--no-native-tool-calling` | Only `PieLLM` hardcodes this off internally — pass explicitly for a fair PieLLM-vs-LiteLLM comparison, since vanilla `litellm` defaults to `native_tool_calling=True` (a different, untested code path requiring `--enable-auto-tool-choice`/`--tool-call-parser` on the vLLM server) |
| `--log-completions DIR` | Dump one JSON file per raw LLM completion to `DIR` for debugging. **Always point this at a path under a directory named `logs/`** (e.g. `logs/completions/`) — that name is gitignored repo-wide, so dumps won't show up as untracked clutter. Don't pass `.` or the repo root. |
| `--output PATH` | Predictions JSONL (required) |
| `--label TEXT` | Free-text label recorded in the JSONL `_metadata` |
| `--verbose` | Stream agent events to stdout |

---

## 5. Known limitations (as of the last debugging session)

Tool-calling plumbing itself is solved and verified live on GPU
(`native_tool_calling` override + the `editor_repair.py` escape-bug fix both
confirmed working). What's still open:

- Neither backend (Pie or LiteLLM baseline) reliably produces a clean,
  correct patch yet — this is a **model-capability** issue (wrong-file
  fixation, eventual token-repetition collapse on long contexts), confirmed
  to affect both backends identically since they share the same OpenHands SDK
  non-native function-calling mock.
- The SDK's own `StuckDetector` requires byte-identical `thought` text between
  repeated actions to flag a loop; Qwen-Coder rephrases its thought each turn
  even while substantively looping, so it often doesn't fire. The
  `run_with_stuck_retries()` nudge-and-retry layer in `swe_bench.py` is wired
  up correctly but only helps when the SDK's detector actually trips.
- Next concrete step identified but not yet implemented: a looser custom
  repetition detector that compares only `tool_name` + parsed action params
  (ignoring `thought` text), plus flagging `str_replace` calls where
  `old_str == new_str` as an immediate no-op.
- The full 50-problem run (either backend) hasn't been done yet — should go
  through `sbatch`, not an interactive session.
- Scoring (`swebench.harness.run_evaluation`) is blocked on Docker access —
  this cluster only has apptainer/singularity.

---

## Paths reference

- Pie config fixtures: `integrations/openhands/tests/fixtures/pie_cuda_vllm_config.toml` (7B), `..._32b.toml` (32B)
- PieLLM: `integrations/openhands/pie_openhands/llm.py`
- Escape-bug repair: `integrations/openhands/pie_openhands/editor_repair.py`
- SWE-Bench harness: `integrations/openhands/benchmarks/swe_bench.py`, `run_swe_bench.py`
- Run scripts: `integrations/openhands/run_pie_backend.sh`, `run_litellm_baseline.sh`
- Sbatch smoke tests: `integrations/openhands/smoke_test.sbatch`, `smoke_test_priority.sbatch`
- Shared vLLM driver venv: `/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env`
- HF cache: `/nfs/roberts/scratch/pi_ql324/ly337/hf_cache`
