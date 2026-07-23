# Tutorial: run a coding agent on your own GPU with OpenHands + Pie

*For someone who has not used either project. No prior Pie or OpenHands knowledge assumed.*

---

## Read this first: is this the thing you want?

Three separate pieces of software are involved, and it helps to keep them straight:

| | What it is | Analogy |
|---|---|---|
| **OpenHands** | The coding *agent*. Reads your issue, explores the repo, edits files, runs tests. | The developer |
| **Pie** | An *inference engine* — it loads a model onto your GPU and generates tokens. | The brain |
| **coder-session** | A small WASM program that runs inside Pie and stops it re-reading the whole conversation every turn. | The brain's short-term memory |

**If you only want to try an open-source coding agent, you do not need Pie at all.** Install OpenHands, point it at any model API, and you are done in five minutes. That is the shortest path and there is no shame in it.

You want *this* setup when you want the model to run **on your own GPU** — no API keys, no per-token cost, nothing leaving your machine — and you want it to be fast. The "fast" part is what coder-session adds, and §6 explains what it actually does.

**Be honest with yourself about the cost.** This is a research integration, not a packaged product. Expect **60–90 minutes** the first time, most of it compiling. You need an NVIDIA GPU. Things will go wrong and §8 lists the ones I hit.

---

## 1. What you need

**A CUDA GPU**, and the model has to fit in its memory alongside the KV cache:

| Model | Weights (bf16) | KV cache | Total | Realistic card |
|---|---|---|---|---|
| Qwen2.5-Coder-7B | ~14 GB | ~4 GB | **~19 GB** | RTX 3090/4090 (24 GB) |
| Qwen3-Coder-30B-A3B | ~60 GB | ~12.5 GB | **~73 GB** | A100 80 GB, RTX PRO 6000 |

**Start with the 7B config.** It is the one most people can actually run. The 30B is what the benchmark numbers come from, but it needs a data-centre card.

Also: **CUDA toolkit 12.x** (`nvcc --version`), **CMake ≥ 3.31**, **Rust stable ≥ 1.87** (`rustup toolchain install stable`), **Python 3.12**, and **~80 GB of disk** for model weights and build artifacts.

> **No GPU?** You can still see the machinery work. Skip to §7 — a dummy driver runs the whole protocol on CPU and prints real cache statistics. It generates nonsense text, but it proves your setup is wired correctly, and it takes two minutes.

---

## 2. Build Pie

Pie does not ship a binary; you compile it. The CUDA driver is not in the default build, so the feature flags matter.

```bash
git clone https://github.com/pie-project/pie      # or your fork
cd pie

# Point the build at your CUDA toolchain.
export CUDACXX=$(which nvcc)
# Your GPU's compute capability WITHOUT the dot:
#   80 = A100 · 86 = RTX 3090 · 89 = RTX 4090 · 90 = H100 · 120 = RTX PRO 6000 Blackwell
export CMAKE_CUDA_ARCHITECTURES=89

cargo build -p pie-server --release --features driver-portable,driver-cuda
```

This takes **10–20 minutes** — it fetches and compiles FlashInfer and CUTLASS through CMake. Setting `CMAKE_CUDA_ARCHITECTURES` wrong is the single most common failure; the build succeeds and the server dies at load time.

Verify the CUDA driver is actually in the binary:

```bash
./target/release/pie driver list
```

`cuda_native` must appear. If you only see `portable` and `dummy`, the feature flags did not take — rebuild.

> On a cluster with environment modules, load `CUDA`, `CMake`, and `NCCL` first. `integrations/openhands/build_pie_cuda.sh` is a worked example.

---

## 3. Build the coder-session inferlet

An **inferlet** is a small program compiled to WebAssembly that runs *inside* the Pie server, next to the KV cache. coder-session is the one that makes the agent fast.

```bash
rustup target add wasm32-wasip2
cd inferlets/openhands-coder-session
cargo build --target wasm32-wasip2 --release
cd ../..
```

Fast — about 30 seconds. Produces `inferlets/openhands-coder-session/target/wasm32-wasip2/release/openhands_coder_session.wasm` (~320 KB).

---

## 4. Set up the Python side

This installs OpenHands' SDK, the Pie client, and the glue package that connects them.

```bash
cd integrations/openhands
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -e ../../client/python
```

Sanity check, no GPU needed:

```bash
.venv/bin/pytest tests/ -q      # expect ~97 passed
```

---

## 5. Start the server and load the model

Pie is configured with a TOML file. A ready-made one for the 7B model lives at `tests/fixtures/pie_cuda_native_config.toml`:

```toml
[auth]
enabled = false                       # local use, no tokens

[[model]]
name = "default"
hf_repo = "Qwen/Qwen2.5-Coder-7B-Instruct"    # downloaded from HuggingFace

[model.driver]
type = "cuda_native"                  # the in-process CUDA engine
device = ["cuda:0"]

[model.driver.options]
kv_page_size = 32
max_num_kv_pages = 2048               # 2048 × 32 = 65,536 KV tokens ≈ 4 GB
weight_dtype = "bfloat16"
swap_pool_size = 0
```

**The one knob to understand:** `max_num_kv_pages × kv_page_size` is your conversation-memory budget in tokens, and the CUDA driver allocates it *up front* as a hard block. It ignores `gpu_mem_utilization`. Too high and you get an out-of-memory error at startup; too low and long agent conversations run out of room. 65,536 tokens is comfortable for the 7B.

Start it (first run downloads ~14 GB of weights):

```bash
cd integrations/openhands
../../target/release/pie serve \
    --config tests/fixtures/pie_cuda_native_config.toml \
    --port 8080 --no-auth
```

Wait for `pie-server serving on`. Leave this running and open a second terminal.

Now **install the inferlet** into the running server — a one-time upload per server start:

```bash
cd integrations/openhands
.venv/bin/python - <<'EOF'
import asyncio
from pie_client import PieClient

WASM = "../../inferlets/openhands-coder-session/target/wasm32-wasip2/release/openhands_coder_session.wasm"
MANIFEST = "../../inferlets/openhands-coder-session/Pie.toml"

async def main():
    async with PieClient("ws://127.0.0.1:8080") as c:
        await c.authenticate("local-dev")
        await c.install_program(WASM, MANIFEST, force_overwrite=True)
        print("coder-session installed")

asyncio.run(main())
EOF
```

---

## 6. Point the agent at your own code

Save this as `run_agent.py` inside `integrations/openhands`. It is the whole integration in about twenty lines.

```python
"""Run the OpenHands agent against a local repo, served by Pie."""
import sys
from pie_openhands import PieLLM
from openhands.sdk import Agent, Conversation
from openhands.tools.preset.default import get_default_tools

repo, task = sys.argv[1], sys.argv[2]

llm = PieLLM(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    pie_uri="ws://127.0.0.1:8080",
    pie_inferlet="openhands-coder-session@0.1.0",
    pie_session=True,             # ← turn ON the KV prefix cache
    pie_python_tool_parser=True,  # ← parse Qwen-Coder's XML tool calls
    temperature=0.0,
)

agent = Agent(
    llm=llm,
    tools=get_default_tools(enable_browser=False),   # terminal, file_editor, task_tracker
    system_prompt_kwargs={"cli_mode": True},
)

conv = Conversation(agent=agent, workspace=repo, max_iteration_per_run=50)
try:
    conv.send_message(task)
    conv.run()
finally:
    llm.close_pie_session()       # ← REQUIRED. See the warning below.
    print(llm.pie_session_summary())
```

Run it:

```bash
.venv/bin/python run_agent.py /path/to/your/repo \
    "The CLI crashes when --output is passed without a value. Find the bug and fix it."
```

The agent explores your repo, edits files, and runs commands **directly on that directory** — it is not sandboxed. Point it at a git checkout with everything committed so `git diff` shows you exactly what it did.

Two lines carry the weight:

- **`pie_session=True`** enables the prefix cache. Without it, every turn re-reads the entire conversation from scratch.
- **`llm.close_pie_session()`** in a `finally` block is **not optional**. It releases the KV snapshots the conversation accumulated. Skip it and each conversation permanently leaks its memory; on a hard-allocated cache, a few back-to-back runs exhaust the budget and the next one *hangs forever* waiting for pages that will never be freed. This exact bug cost us a 4-hour benchmark run.

---

## 7. Check that the cache is actually working

`pie_session_summary()` prints something like:

```python
{'num_calls': 28, 'prompt_tokens_rendered': 804743,
 'prompt_tokens_prefilled': 43516, 'modes': {'rebuilt': 1, 'extended': 27}}
```

Read it like this:

- **`modes: {rebuilt: 1, extended: N}`** is the shape you want. One rebuild (the first call, nothing cached yet) and everything after extending the cache. If you see `rebuilt` climbing, the cache is missing and something is wrong.
- **prefilled ÷ rendered** is the work you avoided. Here 43,516 of 804,743 tokens were actually computed — **95 % saved**. The agent sends the full history every turn; the cache means Pie only processes what is new.

**No GPU? Try this now.** The dummy driver runs the entire protocol on CPU:

```bash
../../target/release/pie serve --config tests/fixtures/pie_dummy_config.toml \
    --port 18099 --no-auth &
.venv/bin/python session_smoke.py
```

It walks the protocol — first call, extension, history rewrite, retry, delete — and asserts the cache behaves correctly at each step. The generated text is gibberish, but the bookkeeping is real, and it is the fastest way to confirm your build works.

---

## 8. When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `pie driver list` shows no `cuda_native` | Built without the feature | Rebuild with `--features driver-portable,driver-cuda` |
| CUDA error at model load | `CMAKE_CUDA_ARCHITECTURES` ≠ your GPU | Rebuild with the right value (§2) |
| OOM at startup, before any request | `max_num_kv_pages` too large | Lower it; remember the driver ignores `gpu_mem_utilization` |
| Everything hangs after a few conversations | `close_pie_session()` not called | Add the `finally` block (§6) |
| `modes` shows `rebuilt` every call | Cache is missing | Confirm `pie_session=True`; changing the model/tools/template invalidates by design |
| Model narrates instead of using tools | Tool format mismatch | Keep `pie_python_tool_parser=True` for Qwen-Coder models |
| Empty patch, agent looped | Usually the model's own limits | Try a bigger model before suspecting the plumbing |

**On what to expect from the model.** A 7B model will fail tasks a 30B solves. In our benchmark the 30B produced correct patches for 11 of 13 real Django/scikit-learn issues; both failures were the *model's* reasoning, not the machinery — one wrote a plausible but wrong fix, the other mistyped its own file path and never recovered. Judge the setup by the cache statistics in §7; judge the model by its patches.

---

## Where to go next

- **`docs/RUNBOOK.md`** — layered by-hand verification, shortest to most integrated
- **`docs/OPENHANDS_CODER_SESSION_DESIGN.md`** — the design behind the prefix cache
- **`PIE_VS_VLLM_EVALUATION.md`** — measured results against vLLM, including where Pie *loses*
- **`inferlets/openhands-coder-session/src/prefix_cache.rs`** — the cache itself; short and commented

To run the SWE-bench benchmark rather than your own repo, `run_pie_backend.sh` wraps all of the above:

```bash
BACKEND=pie-session \
CFG=tests/fixtures/pie_cuda_native_config.toml \
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
OUTPUT=predictions/my_run.jsonl \
  bash run_pie_backend.sh --subset-size 5 --python-tool-parser --temperature 0
```
