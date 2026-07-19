# Pie × OpenHands: A Code Walkthrough

*A file-by-file guide to the integration, for anyone reading the code for the first time.*

---

The integration lives across ~10 files spanning Rust, Python, TOML, and shell. This document walks through each one — what it does, why it exists, and where the interesting decisions are. Read in order for a bottom-up understanding; or jump to whatever file you're staring at.

---

## 1. The Agent Inferlet

**`inferlets/openhands-agent/src/lib.rs`** (~870 lines of Rust → 360KB WASM)

This is the heart of the integration. It's a Rust program compiled to WebAssembly that runs *inside* Pie's serving engine, directly alongside the model. It implements the full agent loop: generate a structured action, execute it via HTTP, observe the result, decide the next step.

### The entry point

```rust
#[inferlet::main]
async fn main(input: Input) -> Result<String> {
```

The `#[inferlet::main]` macro wires this into Pie's WASM component model. The function receives a JSON `Input` (task description, tool server URL, step limits) and returns a JSON string with results and metrics. Everything between is the agent loop.

### The generate-act-observe loop

The core loop (starting around line 616) does five things per step:

1. **Check context size** — if the estimated sequence length approaches the limit, trigger condensation (more on this below).

2. **Generate with a schema constraint** — `ctx.generate(Sampler::Argmax).constrain_with(inferlet::JsonSchema(schema))` forces the model's output to conform to `ACTION_SCHEMA`. The engine masks logits at every token position. The model literally cannot produce invalid JSON.

3. **Parse and dispatch** — the JSON is deserialized into `thought`, `action`, `command`, `path`, `old_str`, `new_str`, `message`. If the action is `"finish"`, the loop ends (with empty-finish and test-verification checks first).

4. **Execute the tool** — `ctx.idle()` drops the context's memory bid, then `call_tool_server()` makes an HTTP POST. When the response arrives, `drop(_idle)` restores priority. The GPU memory is available for other workloads during the entire tool execution.

5. **Append observation** — `ctx.user(&format!("Observation:\n{}", obs))` adds the tool output to the context. `ctx.cue()` signals "your turn, model." The KV cache grows incrementally — nothing is re-encoded.

### Two schemas, switched at the boundary

`ACTION_SCHEMA` allows any of `["bash", "edit", "read_file", "insert", "undo_edit", "finish"]`. On the final step (`step == input.max_steps`), the schema switches to `FINISH_SCHEMA`, where `"action"` is constrained to `const: "finish"`. The model is physically forced to terminate. No runaway loops.

### Stuck detection

`detect_stuck()` (line 241) inspects a rolling window of recent actions for five patterns:

- Same `(action, path)` failing 3 times in a row
- `read_file` → `edit` → fail cycles on the same file
- Identical bash commands repeated 3 times
- Alternating A-B-A-B action patterns with failures
- 5 consecutive actions on the same path with 3+ failures

When detected, a specific, actionable hint is appended to the observation: *"Try a different approach: use `bash` with `sed`..."* This is more useful than OpenHands SDK's built-in stuck detector, which requires byte-identical `thought` text across turns — real models almost always rephrase slightly.

### Context condensation

`condense_context()` (line 504) fires when the estimated sequence length approaches the model's context window. It:

1. Uses the model itself (`summarize_dropped_turns()`) to summarize the oldest turns into a ~400-word summary
2. Rebuilds the context from scratch: system prompt → task → summary → recent turns
3. Targets 70% of the budget (`CONDENSE_TARGET_FRAC`) to leave headroom before the next trigger
4. Has a 5-step cooldown (`CONDENSE_COOLDOWN`) to prevent pathological condense-every-step loops

This is the "agent manages its own memory" primitive. The condensation policy is just code inside the inferlet — different agents could implement different strategies.

### Empty-finish recovery

When the agent calls `finish`, the inferlet checks `git diff` via a `has_diff` tool-server call (line 703). If there's no diff (the agent didn't actually change anything), it nudges: *"Your changes produced no diff — the issue is NOT resolved yet."* Up to `max_empty_finishes` retries. This recovered 2 extra SWE-Bench instances in the 50-problem eval.

### Test verification gate

If the agent finishes with a diff but never ran `pytest` or `unittest` (tracked via `ran_tests_after_edit`, line 609), it gets a one-time nudge to verify its fix before finishing. Prevents the "edit and immediately declare victory" pattern.

### Per-step metrics

`StepMetrics` (line 216) tracks `generate_s`, `tool_s`, `prompt_tokens`, `completion_tokens`, and `seq_len_after` per step. These are returned in the output JSON and extracted by the Python harness into the predictions JSONL. This is how the efficiency breakdown in the writeup will be generated.

### The manifest

**`inferlets/openhands-agent/Pie.toml`** declares the inferlet's parameters (task, tool_server_url, step limits) and its runtime requirements. This is what Pie reads when the inferlet is installed — it tells the runtime what inputs to expect and what capabilities the inferlet needs (like outbound HTTP for tool calls).

---

## 2. The Tool Server

**`integrations/openhands/tool_server.py`** (~530 lines of Python)

The inferlet runs inside a WASM sandbox — no filesystem access, no process spawning. The tool server is the bridge: a lightweight HTTP server that receives action requests and executes them on the host.

### Starting it

```python
server, port = start_tool_server("/path/to/workspace")
```

This spawns a background thread running an `HTTPServer` with an OS-assigned port. Returns the port so the inferlet knows where to send requests.

### The persistent bash

`PersistentBash` (line 51) is a long-lived `bash --norc --noprofile` subprocess. Commands are sent via `stdin` with a random marker to detect completion:

```python
marker = f"___PIE_DONE_{os.urandom(8).hex()}___"
wrapped = f"{command}\n_pie_ec=$?\nprintf '\\n{marker}%d\\n' \"$_pie_ec\"\n"
```

The server reads stdout line-by-line until it sees the marker, then extracts the exit code. This preserves `cwd` and environment across commands — the agent can `cd` into a directory on one step and `ls` on the next. A stateless bash (fresh subprocess per command) is a constant source of agent confusion.

The bash session auto-restarts if the subprocess dies (e.g., from an OOM-killed test suite). Timeout is 120 seconds per command.

### Chunked encoding handling

`_read_chunked()` (line 215) handles HTTP/1.1 chunked transfer encoding. This exists because the `wstd` crate (WASI HTTP) always sends chunked bodies, even when `Content-Length` is set. Python's `BaseHTTPRequestHandler` only reads `Content-Length` bytes by default, so without this, every tool call arrives as an empty body. Twenty lines to bridge a spec mismatch between three layers.

### File operations

The server delegates to OpenHands SDK's `FileEditor` when available (imported at the top with a fallback). This gives proper `view_range`, `insert`, `undo_edit`, and `str_replace` with fuzzy matching. When `FileEditor` isn't installed, built-in fallbacks handle the basics.

Key design decisions in the file operations:

- **Edit size limits** (line 28): `MAX_EDIT_OLD_LINES = 50`, `MAX_EDIT_NEW_LINES = 100`. Prevents the agent from trying to replace an entire function body, which almost never works.
- **Syntax checking** (line 363): after editing a `.py` file, `compile(new_content, path, "exec")` runs and any `SyntaxError` is included in the observation. Immediate feedback vs. discovering the error three steps later.
- **Context in edit confirmations** (line 408): successful edits return surrounding lines with line numbers, so the agent can verify without an extra `read_file` call.
- **Diff check** (`_exec_has_diff`, line 300): a `git diff HEAD` + `git ls-files --others` check used by the inferlet's empty-finish recovery.

---

## 3. The PieLLM Adapter (Pattern B)

**`integrations/openhands/pie_openhands/llm.py`** (~420 lines of Python)

This is the drop-in LLM backend for Pattern B — where the Python SDK runs the agent loop and Pie acts as a completion endpoint. It's a subclass of `openhands.sdk.LLM` that overrides `_transport_call` to route through Pie instead of LiteLLM.

### Why `_transport_call`

The parent class `LLM` has a complex `completion()` method that handles message formatting, prompt caching, retry decoration, tool-call schema injection, and `LLMResponse` construction. `_transport_call` (line 75) is the narrow waist — it receives already-formatted OpenAI-style message dicts and returns a `ModelResponse`. By overriding only this, all the SDK machinery continues to work.

### The Pie round-trip

`_call_pie()` (line 129) connects to the Pie server via WebSocket, launches the `openhands-completion` inferlet with the messages and tools as structured JSON, and collects the `Return` event. The inferlet does its own chat-template rendering and history replay — the Python side just forwards structured data.

### Tool call ID uniqueness

A subtle bug (documented in the comment at line 179): the inferlet numbers tool calls `call_0`, `call_1`, ... starting from zero on every request. Since most turns emit exactly one tool call, nearly every call would get id `"call_0"`. OpenHands SDK's `ObservationUniquenessProperty` deduplicates by `tool_call_id`, so it would silently drop every tool result after the first. The fix: UUID-based ids (`call_{uuid.uuid4().hex[:24]}`).

### Tool argument sanitization

`_sanitize_tool_args()` (line 240) strips keys from tool-call arguments that aren't in the tool's JSON Schema. Small models (especially the 7B) sometimes append garbage key-value pairs after valid arguments — `{"command":"pwd && ls",", ":", "}` — that survive the grammar (any valid JSON) but get rejected by Pydantic `extra='forbid'` on OpenHands' action models, sending the agent into a stuck error loop. This strips the garbage before it reaches the SDK.

### Native few-shot examples

`_inject_native_examples()` (line 385) prepends concrete tool-usage examples to the first user message. These show the exact `<tool_call>{"name":..., "arguments":...}</tool_call>` format the model is expected to produce — including `old_str`/`new_str` for `str_replace`, which smaller models otherwise omit. This mirrors OpenHands' non-native `fn_call_examples.py` but in the native format.

---

## 4. The Benchmark Harness

**`integrations/openhands/benchmarks/swe_bench.py`** (~700 lines of Python)

The orchestration layer. Takes a list of SWE-Bench problems, runs each one through either Pattern A or Pattern B, and writes predictions to a JSONL file compatible with the SWE-Bench grader.

### Two code paths

The harness supports three backends: `"pie"` (Pattern B), `"pie-agent"` (Pattern A), and `"litellm"` (vanilla baseline). The backend is selected in `build_llm()` (line 199), which returns a configured LLM instance for Pattern B, or in `solve_one_agent()` (line 666), which bypasses the LLM entirely and talks to the inferlet directly.

**Pattern B** (`solve_one`, line 481): builds an LLM + Agent + Conversation, sends the problem statement, and runs the SDK's agent loop with `run_with_stuck_retries()`.

**Pattern A** (`solve_one_agent`, line 666): clones the repo, starts a tool server, launches the inferlet via `_run_agent_inferlet()`, and captures `git diff` when it finishes. No Agent, no Conversation, no SDK agent loop — the inferlet handles everything.

### `run_with_stuck_retries()`

This function (line 397) wraps Pattern B's `conv.run()` with two recovery mechanisms:

1. **Fake user responses** — when the agent sends a content-only message (no tool calls), the SDK sets `FINISHED` after one turn. This detects that case and sends a nudge (*"Please continue working on the task..."*), up to `max_fake_responses` (default 10). Ported from the official OpenHands evaluation harness.

2. **Stuck retries** — when the SDK's `StuckDetector` fires, sends `STUCK_NUDGE_MESSAGE` to break out of repeating action-observation loops.

Note: Pattern A doesn't use this function — the inferlet has its own stuck detection and recovery built in. This is only for Pattern B.

### Workspace management

`checked_out_repo()` (line 154) clones the target repo at `base_commit` into a temp directory. With a `cache_dir`, it maintains bare clones and only copies the working tree — much faster when many problems share a repo (e.g., 50 django problems).

### The 8-phase system prompt

`SWE_BENCH_SYSTEM_SUFFIX` (line 240) is a structured prompt guiding the model through: reading → running tests → exploration → test creation → fix analysis → fix implementation → verification → final review. This matches the official OpenHands evaluation template and significantly helps smaller models stay on track.

### `_run_agent_inferlet()`

This async function (line 590) is Pattern A's interface to the Pie runtime. It connects via WebSocket, sends the input payload (task, tool_server_url, max_steps, context_token_limit), and streams `Stdout` events (the inferlet's step-by-step logs) while waiting for the `Return` event. Three timeout layers:

- `idle_timeout_s`: no event for N seconds → generation hung
- `instance_timeout_s`: wall-clock cap per problem → prevents one hard instance from burning the whole allocation
- `timeout_s`: legacy per-recv timeout

### The predictions JSONL

`Prediction` (line 108) captures everything needed for scoring and analysis: `instance_id`, `model_patch` (the `git diff`), plus metadata — `wall_clock_s`, `agent_iterations`, `stuck_retries`, token counts, and (for Pattern A) per-step `generate_s`/`tool_s` breakdowns.

---

## 5. The CLI Entry Point

**`integrations/openhands/benchmarks/run_swe_bench.py`**

Argparse wrapper around `swe_bench.py`. All the flags:

- `--backend pie|pie-agent|litellm|test` — which code path
- `--model`, `--pie-uri`, `--pie-inferlet` — model and endpoint config
- `--subset-size`, `--instance-id` — problem selection
- `--max-iterations`, `--max-steps` — step limits (Pattern B and A respectively)
- `--max-stuck-retries`, `--max-fake-responses` — recovery limits
- `--no-condenser` — disable the LLM summarizing condenser
- `--output`, `--resume` — output file and skip-already-completed support
- `--context-token-limit` — controls when Pattern A triggers condensation
- `--verbose` — sets log level to DEBUG

The `--resume` flag is critical for HPC jobs: if a run is preempted (Slurm timeout) or crashes, re-running with `--resume` skips instances already in the output JSONL.

---

## 6. The Shell Orchestrator

**`integrations/openhands/run_pie_backend.sh`** (~180 lines of bash)

Glues everything together for a real GPU run. Three phases:

1. **Boot pie serve** — starts the Pie server with the model config, waits for the readiness line (`"pie-server serving on"`), hard-fails on timeout.

2. **Install the inferlet** — uploads the WASM binary and manifest via `PieClient`.

3. **Run the benchmark** — launches the Python harness with all the config wired through environment variables.

### Environment variables

The script is parameterized via env vars so the same script works for different models, backends, and harnesses:

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND` | `pie` | `pie` (Pattern B) or `pie-agent` (Pattern A) |
| `CFG` | `pie_cuda_vllm_config.toml` | Pie server config |
| `MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct` | HuggingFace model id |
| `LABEL` | `pie+qwen2.5-coder-7b` | Label in predictions JSONL |
| `HARNESS` | `benchmarks.run_swe_bench` | Python module to run |
| `OUTPUT_PREFIX` | `pie_qwen25_coder_7b` | Predictions filename prefix |

### Auto-restart

The benchmark loop (line 137) handles server crashes gracefully. If the harness exits with code 42 (a convention meaning "server died"), the script restarts `pie serve`, reinstalls the inferlet, and resumes the benchmark with `--resume`. Up to `MAX_SERVER_RESTARTS` (default 3) restarts before giving up. This is essential for long HPC runs where vLLM OOM crashes can kill the server mid-problem.

---

## 7. The Model Config

**`integrations/openhands/tests/fixtures/pie_cuda_vllm_config_32b.toml`**

```toml
[auth]
enabled = false

[server]
verbose = true

[[model]]
name = "default"
hf_repo = "Qwen/Qwen2.5-Coder-32B-Instruct"

[model.driver]
type = "vllm"
device = ["cuda:0"]

[model.driver.options]
venv = "/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env"
enforce_eager = true
gpu_memory_utilization = 0.80
max_model_len = 32768
```

Key settings:

- **`verbose = true`** — required for the readiness-wait grep in `run_pie_backend.sh` (the readiness line is gated behind this flag — bug #8 in the integration history).
- **`enforce_eager = true`** — disables CUDA graph capture, which avoids OOM during warmup on GPUs with tight memory.
- **`gpu_memory_utilization = 0.80`** — lowered from 0.85 after a CUDA OOM crash at step 25 of the 200-problem eval. The SiLU activation layer needed 1.33 GiB that wasn't available at 0.85.
- **`max_model_len = 32768`** — caps the context window, reducing KV cache pre-allocation.
- **`venv`** — path to the shared Python venv with vLLM and its dependencies.

There's also a `pie_cuda_vllm_config.toml` for the 7B model, and a `pie_cuda_vllm_config_30b_moe.toml` for the MoE model. They differ only in `hf_repo` and memory settings.

---

## 8. The Driver-Side Fixes

Three bugs in Pie's driver layer were found and fixed during the integration. They're small in code but large in impact.

### `driver/vllm/src/pie_driver_vllm/engine.py` — Architecture name normalization

The vLLM driver reported HuggingFace's raw architecture string (`"Qwen2ForCausalLM"`) to the Pie runtime, which matches on short lowercase names (`"qwen2"`). The mismatch caused a silent fallback to a generic config with `has_tools: false`, meaning tool schemas never reached the model for the entire life of the integration.

Fix: `_normalize_arch_name()` — lowercase + strip `"forcausallm"` suffix.

### `driver/dev/src/pie_driver_dev/worker.py` — Vocab size resolution

The vLLM driver's worker resolved vocab size via `getattr(config, "vocab_size", 128000)`, but vLLM's `ModelConfig` has a `get_vocab_size()` *method*, not a `vocab_size` attribute. Silent fallback to 128,000. Qwen2.5-Coder's real vocab is 152,064. This made grammar-constrained decoding crash the vLLM subprocess — which looked like a hang because the host loop didn't detect the dead worker.

Fix: `_resolve_vocab_size()` — try `num_vocabs` attr → `get_vocab_size()` method → `vocab_size` attr → fallback.

### `runtime/src/model/tokenizer.rs` — Special token exclusion

The grammar's token-candidate enumeration excluded tokens with empty decoded text but not tokens in `special_token_ids()` that have printable text (like `<|im_end|>`, which decodes to literal `<|im_end|>`). These special tokens leaked into the grammar's "valid completion" set, letting the model emit EOS from inside an incomplete JSON string.

Fix: exclude `special_token_ids` from `sorted_vocab`, not just empty-decoded tokens.

---

## 9. The Test Suite

**`integrations/openhands/tests/`**

- **`test_pie_llm.py`** (~17 tests) — unit tests for `PieLLM`: Pydantic field defaults, `_wrap_as_model_response` construction, `_sanitize_tool_args` stripping, `_flatten_content` normalization. Uses a mocked `_call_pie`.

- **`test_e2e_openhands_smoke.py`** (~4 tests) — real `Agent` + `Conversation` with `PieLLM` and a scripted response. Catches integration issues with the SDK's retry/format/event pipeline.

- **`test_swe_bench_smoke.py`** (~10 tests) — harness plumbing: `run_with_stuck_retries` with a duck-typed `_FakeConversation`, fake user response logic, and Pattern A `_run_agent_inferlet` contract.

- **`test_tool_server.py`** (~13 tests) — tool server actions: bash execution, edit with str_replace, file create, read_file, chunked encoding, diff check.

- **`test_wire_protocol_smoke.py`** — env-gated test that runs the full `PieLLM` → `pie run --path` → inferlet round-trip against a real (dummy-driver) binary. Needs `PIE_BIN` and `PIE_WASM` env vars.

Total: ~58 tests, 3 skipped for env gates. Run with:

```bash
cd integrations/openhands
.venv/bin/pytest tests/ -v
```

---

## How It All Fits Together

A Pattern A SWE-Bench run looks like this, from top to bottom:

1. **`run_pie_backend.sh`** boots `pie serve` with a model config, installs the agent WASM, and launches the Python harness.

2. **`run_swe_bench.py`** parses CLI flags and calls into `swe_bench.py`.

3. **`swe_bench.py::solve_one_agent()`** clones the repo, starts a `tool_server`, and calls `_run_agent_inferlet()`.

4. **`_run_agent_inferlet()`** connects to `pie serve` via WebSocket and launches the inferlet with the task and tool server URL.

5. **`inferlets/openhands-agent/src/lib.rs`** runs the agent loop on the GPU:
   - `ctx.generate()` with `JsonSchema` constraint → structured action
   - `ctx.idle()` → `call_tool_server()` HTTP POST → `tool_server.py` executes the action
   - `ctx.user(observation)` → `ctx.cue()` → next step
   - Context condensation when approaching the limit
   - Stuck detection, empty-finish recovery, test verification

6. **`tool_server.py`** executes bash commands, file edits, and reads against the git worktree.

7. The inferlet returns a JSON result with the agent's message, step count, and per-step metrics.

8. **`swe_bench.py`** captures `git diff HEAD` from the worktree, writes a `Prediction` to the JSONL.

9. Repeat for each problem. If the server crashes (exit code 42), `run_pie_backend.sh` restarts it and resumes.

Pattern B follows the same flow for steps 1-3, but then uses the OpenHands SDK's `Agent` + `Conversation` + `PieLLM` to run the loop in Python, with `PieLLM._transport_call()` making per-step calls to the `openhands-completion` inferlet.
