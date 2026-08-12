# Handover — qwen-code ↔ Pie on the rewritten engine

Written 2026-08-12 for the agent picking this up on a new 48 GB machine.
Read this first, then `docs/qwen-code-dev-port.md` (design + results) and
`PROGRESS.md` (running log). Everything below is either verified or
explicitly marked unverified — please keep that distinction.

---

## 0. Setting up a fresh machine

Prerequisites (Homebrew + Xcode command line tools assumed):

```bash
xcode-select --install 2>/dev/null || true
brew install cmake node jq git
curl -sSf https://sh.rustup.rs | sh -s -- -y      # toolchain + wasm target
source "$HOME/.cargo/env"                          # come from rust-toolchain.toml
```

Clone and build (the wasm target is pinned by `rust-toolchain.toml`, so
rustup installs it on first build):

```bash
mkdir -p ~/Documents/Liszt_ai && cd ~/Documents/Liszt_ai
git clone -b liu/qwen-code-dev https://github.com/YangLiuWillow/pie.git
cd pie
git remote add fork https://github.com/YangLiuWillow/pie.git 2>/dev/null || true

cargo build --release -p pie-bin --features driver-metal        # ~15 min cold
(cd tests/inferlets && cargo build --target wasm32-wasip2 --release -p chat-completions)
```

Python side (a fresh venv — the old machine's venv will not survive the
move; `transformers` is only needed for the parity check):

**The venv needs Python ≥ 3.10, and a bare macOS box does not have one.**
`client/python` uses PEP-604 annotations (`str | Path`), so the stock
CommandLineTools 3.9.6 fails at *import* with `TypeError: unsupported
operand type(s) for |` and the shim never starts. With no Homebrew, `uv`
installs a standalone interpreter without sudo (both gaps hit on the
2026-08-12 bring-up):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 ~/.venvs/pie && source ~/.venvs/pie/bin/activate
uv pip install websockets msgpack blake3 cryptography transformers huggingface_hub jinja2
```

`jinja2` is **required** and is not pulled in by `transformers` — without
it `apply_chat_template` raises and `check_render.py` cannot run at all.

Sanity-check the toolchain before pulling 17 GB of weights:

```bash
(cd tests/inferlets && cargo test -p chat-completions)   # expect 25 passed
python3 integrations/qwen-code/bench/test_summarize.py   # expect PASS
```

## 1. Where things are

| | |
|---|---|
| Repo | `~/Documents/Liszt_ai/pie` (moved 2026-08-12 from `~/Desktop/Lin_startup/pie`) |
| Branch | `liu/qwen-code-dev`, pushed to remote `fork` = `github.com/YangLiuWillow/pie` |
| Upstream | remote `origin` = `pie-project/pie`; branch `dev` is the rewritten engine |
| **Rule** | **Never commit to `dev` or `main`.** Feature branches only (`liu/<topic>`). |
| Secrets | `~/Documents/Liszt_ai/pie-rl/.env` (`RUNPOD_API_KEY`, `HF_TOKEN`, `GH_TOKEN`) |
| Old work | Pre-rewrite integration lives on branch `openhands-integration-updated` |

## 2. What this is

Stock qwen-code (unmodified, v0.21.6) talks OpenAI `/v1/chat/completions`
to a **shim**, which drives a **long-lived inferlet** inside pie:

```
qwen-code ──HTTP/SSE──► integrations/qwen-code/shim.py ──WS /v1/ws──► pie serve
                        (transport only)                              │
                                                    tests/inferlets/chat-completions
                                                    (all OpenAI semantics + KV reuse)
```

The rewritten engine removed in-guest HTTP (`world.wit` imports only
outbound `wasi:http/client`), so the OpenAI surface *must* live client-side.
The inferlet's `run()` is a `session::receive()/send()` loop; the shim
multiplexes HTTP requests onto it by `req_id`.

Envelope contracts (stable, both directions):
- shim → inferlet: `{"req_id", "body": <OpenAI request>, "now": <unix secs>}`
- inferlet → shim: `{"req_id", "event": "chunk"|"response"|"done"|"error", "data"}`

## 3. State of play — verified vs not

**Verified (evidence in `docs/qwen-code-dev-port.md` §8–§11):**
- Port runs end-to-end on GPU: **acceptance 33/33** on an RTX 3090 with
  Qwen3-0.6B, and stock qwen-code completed a shell-tool task with a clean
  exit.
- **KV reuse works on a real agent workload**: 76.9% of prompt tokens served
  from cache on the 30B run (~8.7K-token resume hits per follow-up turn).
- **Renderer parity, hermes dialect: 23/23 byte-exact** vs
  `apply_chat_template` (`parity/check_render.py`).
- Unit tests 25/25 native; wasm builds clean.
- Reporting layer (`bench/summarize.py`) regression-tested offline
  (`bench/test_summarize.py`).
- **Coder-dialect prompt parity: 23/23 exact against a *served* Coder
  model** (2026-08-12, Metal, `pie_config_metal_coder.toml`, referenced to
  `Qwen/Qwen3-Coder-30B-A3B-Instruct`). No renderer changes were needed.
  Confirmed to be genuinely the Coder path, not a pass for the wrong
  reason, by eight dialect markers. Gate 1 of §4 is **closed**. Details and
  caveats: `docs/qwen-code-dev-port.md` §12.

**NOT verified — this is the top of the queue:**
- **Coder tool-calling end-to-end.** §12 covers *prompt bytes only*;
  `echo_tokens` short-circuits before the forward pass. That a served Coder
  model actually emits `tool_calls` rather than prose is still unproven,
  and cannot be proven locally — see the MoE blocker in §5 / port doc §13.
  Needs a CUDA pod.
- **Trajectory equivalence vs vLLM.** Never validly measured (see §4).
- **The `/no_think` divergence class.** No fixture sets
  `enable_thinking:false`, so that path is untested; `check_render.py`
  normalizes-and-reports it as `KNOWN-DIV` when it fires.

## 4. The benchmark: three runs, no valid numbers yet

| run | outcome |
|---|---|
| 1 (H200) | Incomplete — RunPod credit hit $0 mid-run, pods auto-terminated, results lost with the pod. |
| 2 (H200) | Completed but **INVALID**. pie 3/5 @ 123.7 s vs vLLM 5/5 @ 30.3 s, **all five trajectories diverged** — because our renderer served a Coder-tuned model the hermes tool prompt, so it answered in prose and never called a tool. Raw data archived in `bench/results-2026-08-12-run1/`. **Do not quote these numbers.** |
| 3 | Not started. Dialect now fixed; needs the two gates below. |

**Two gates before run 3 produces meaningful numbers:**

1. **Coder-dialect prompt parity must be exact.** Run
   `parity/check_render.py --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct`
   against a pie instance serving a Coder model. If the prompt bytes differ,
   trajectories will differ for reasons that have nothing to do with serving.
2. **Pin greedy decoding on both arms.** This is the subtle one:
   **qwen-code never puts `temperature` on the wire** — run-2 captures carry
   only `model`, `max_tokens`, `stream`, `stream_options`, despite the bench
   `settings.json` setting `samplingParams`. So both arms sample
   stochastically with *different* defaults (pie: `DEFAULT_TEMPERATURE` 0.7 /
   `DEFAULT_TOP_P` 0.8 in `handler.rs`; vLLM: the model's
   `generation_config`, which also sets `top_k`). Under stochastic sampling,
   trajectories differ run-to-run **even when both stacks are correct**, so
   trajectory equality is meaningless until t=0 is forced on both sides:
   - pie: set the inferlet defaults to 0 (or add a launch knob — there is no
     env override today).
   - vLLM: `--override-generation-config '{"temperature":0,"top_p":1}'`.
   Then repeat the battery twice; identical trajectories across repeats is
   the determinism check, and only then is arm-vs-arm comparison valid.

Also note **vLLM never reports `cached_tokens`**, so its APC savings are
invisible to this harness. Report pie's reuse against its own prompt
volume; never present it as an arm-vs-arm reuse comparison.

## 5. The new 48 GB machine — what becomes possible

This is a real unlock: the old 8 GB M2 could not boot *any* model, because
the Metal driver's admission guard needs `want + 2 GiB` **host-reclaimable**
memory and the 2 GiB margin alone exceeded what was free.

- **Metal is 4-bit-only** (every matvec binds `.weight`/`.scales`/`.biases`;
  a bf16 repo imports fine and then fails to bind at load). Use MLX builds:
  - smoke/parity: `mlx-community/Qwen3-0.6B-4bit`
  - **Coder dialect: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`
    (~17 GB)** — this is what makes gate 1 above doable locally. It loads
    and admits fine at `max_model_len = 32768` on 48 GB (weights are mapped
    where they lie, *not* on the heap, so the admission arithmetic is much
    kinder than the raw 17 GB suggests).
  - **But that model's MoE path generates garbage on Metal** — the "if the
    MoE path misbehaves, fall back to a GPU pod" case, now confirmed.
    Deterministic token salad under greedy; a dense `Qwen3-0.6B-4bit`
    control on the same driver and binary is coherent. Evidenced hypothesis:
    the build is mixed-precision (routers 8-bit, everything else 4-bit) and
    `model_facts.cpp:74-82` reads only the global `bits`/`group_size`. Port
    doc §13. **Rendering is unaffected** (it never runs the model), so the
    parity gate is still valid locally — generation work is not.
- `[driver] max_model_len` is the knob that shrinks KV (it sizes the M=1
  ring); `total_pages` is **not** and never was. Default is the driver
  ceiling (~14.6 GiB on a small model) — always set it.
- Check headroom before booting: the guard compares against *reclaimable*
  memory, roughly `free + inactive + purgeable` from `vm_stat`. If it
  refuses, close apps — do **not** bypass the guard: overcommitting Metal
  wedges the GPU until reboot (the driver source documents kernel panics).
- **vLLM does not run on Apple Silicon GPUs.** So the A/B *cannot* be done
  locally at any memory size. Local = correctness, parity, dialect, KV-reuse
  behavior. The pie-vs-vLLM comparison always needs a CUDA pod.
- The **dummy driver caps at 4096-token contexts** (pool derived in
  `worker/src/embedded_driver.rs`; `kv_page_size`/`total_pages`/
  `max_model_len` are rejected knobs). A real qwen-code turn is ~9K tokens,
  so dummy is for transport/acceptance work only.

**Dialect detection depends on the config name.** pie exposes no HF id and
`architecture()` returns `qwen3_moe` for Coder and non-Coder alike, so
`[model] name` must contain `"coder"` for a Coder deployment or the renderer
silently serves the hermes prompt — the exact bug that invalidated run 2.

## 6. Command cheat-sheet

```bash
cd ~/Documents/Liszt_ai/pie

# build (Metal on Apple Silicon; package is pie-bin, NOT pie)
cargo build --release -p pie-bin --features driver-metal
(cd tests/inferlets && cargo build --target wasm32-wasip2 --release -p chat-completions)

# model must be converted to a .zt artifact before serving
./target/release/pie --config <cfg> model import mlx-community/Qwen3-0.6B-4bit
./target/release/pie --config <cfg> doctor          # says whether it can boot

# stack (config via -c/--config or $PIE_CONFIG)
PIE_CONFIG=integrations/qwen-code/pie_config.toml ./target/release/pie serve &
python3 integrations/qwen-code/shim.py --pie ws://127.0.0.1:18080 --port 8123 \
  --wasm tests/inferlets/target/wasm32-wasip2/release/chat_completions.wasm \
  --manifest tests/inferlets/chat-completions/Pie.toml &

# checks
python3 integrations/qwen-code/test_acceptance.py --base http://127.0.0.1:8123
python3 integrations/qwen-code/parity/check_render.py [--hf-model <id>]
(cd tests/inferlets && cargo test -p chat-completions)
python3 integrations/qwen-code/bench/test_summarize.py

# benchmark (one arm at a time; they share the GPU)
bash integrations/qwen-code/bench/start_pie_arm.sh
ARM=pie  BASE_URL=http://127.0.0.1:8123/v1  MODEL=qwen3-coder bash bench/run_arm.sh
bash integrations/qwen-code/bench/start_vllm_arm.sh
ARM=vllm BASE_URL=http://127.0.0.1:18000/v1 MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct bash bench/run_arm.sh
python3 bench/summarize.py results/<pie-dir> [results/<vllm-dir>]   # 1 or 2 arms
```

`parity/check_render.py` needs `transformers`; the only working interpreter
found on the old machine was `integrations/openhands/.venv/bin/python`
(that venv may not survive the move — recreate if needed).

## 7. RunPod playbook (when a GPU is needed)

- Key in `pie-rl/.env`. **The account is shared with other Claude sessions** —
  always `query { myself { pods { … } clientBalance } }` first. Run 1 died
  because four pods across sessions drained the balance to $0 and RunPod
  terminated everything mid-benchmark.
- **vLLM 0.25.1 needs host driver ≥ 580** (CUDA-13 wheels). Do **not**
  downgrade vLLM to dodge this — it changes the baseline. Instead pick a
  machine: **H100 80GB HBM3 / H100 NVL on SECURE cloud reliably had
  580.126.09**; the H200 pool was 570/575 on 16 consecutive tries. H100 is
  also cheaper ($3.29/hr vs $4.59/hr) and fits the 57 GB model fine.
  `start_vllm_arm.sh` fails fast on an old driver rather than after a
  ten-minute install.
- Bring-up is one command: `bash integrations/qwen-code/pod_bootstrap.sh`
  (installs CUDA toolkit ≥ 12.9 — the pie CUDA driver needs it — plus
  rustup, node, builds both binaries, imports the model).
- **Copy results off the pod before terminating.** Run 1 lost everything by
  terminating first. `rsync` the `bench/results/` tree, then terminate.
- Budget: a full two-arm run is ~1 h ≈ $4.

## 8. Gotchas that each cost real time

1. `pie` the binary comes from package **`pie-bin`**; `-p pie` is a different
   crate and fails confusingly.
2. `pie model list` also prints raw HF-cache snapshots, so "is it imported?"
   must check for the `.zt` artifact (this made `pod_bootstrap.sh` skip the
   30B import and waste a boot).
3. The **first request on a big model outlives the default 30 s
   `silence_timeout`**, and the engine's kill takes the gateway WebSocket
   down with the inferlet. Bench config sets 300 s; the shim now reconnects
   with backoff.
4. **The degrade path must not emit whitespace.** qwen-code trims content
   before its non-empty check, so a `" "` delta reads as an empty turn and
   burns the 4× `NO_FINISH_REASON` retry budget.
5. `qwen-code --safe-mode` blocks `write_file` in headless yolo mode: the
   tool call round-trips correctly but execution is refused. Phrase e2e
   tasks as shell commands, or drop the flag.
6. The bench harness needs **node** (it runs `npx qwen-code`); without it
   every task exits 127.
7. The dev-branch python client rejected the engine's "Already
   authenticated" sentinel — patched in `client/python`.
8. Sub-page conversations report `cached_tokens: 0` by design; reuse is
   page-granular (32 tokens).
9. Jinja's `| string` is Python's `str()`, so booleans render `True`/`False`
   inside tool schemas — the Coder golden caught this.
10. Tool schemas and replayed tool-call arguments must be re-serialized
    jinja-`tojson` style (spaced separators, insertion order — hence
    `serde_json/preserve_order`), because that is what a template-driven
    server feeds the model.

## 9. File map

```
tests/inferlets/chat-completions/          the inferlet
  src/types.rs      OpenAI wire types (unknown fields ignored, never 400)
  src/render_text.rs PURE strings: both dialects, tojson, no_think. Natively tested.
  src/render.rs     token assembly (piecewise: pre-encoded seams + encode)
  src/session.rs    canon items + FNV-1a-64 addressing, resume-point split
  src/generation.rs PTIR chunked prefill + device-carried decode loop
  src/handler.rs    orchestration, degrade discipline, KV session retention
  src/salvage.rs    bare Coder-XML + unterminated-hermes call recovery
  src/chunk.rs      chat.completion.chunk assembly + shim envelope
  tests/coder_template_golden.json   generated from the real chat template

integrations/qwen-code/
  shim.py           OpenAI HTTP/SSE ↔ gateway WS, reconnects, keepalives
  pie_config*.toml  metal / cuda / dummy profiles
  pod_bootstrap.sh  idempotent GPU-pod bring-up
  test_acceptance.py  audit §1 hard-requirements (33 rows)
  parity/check_render.py  C3 prompt parity vs apply_chat_template
  bench/            A/B harness, arm boot scripts, summarize + its test
  bench/results-2026-08-12-run1/  archived (INVALID — see §4)
```

## 10. Suggested order of work

1. ~~**Close the Coder-dialect gate locally**~~ — **DONE 2026-08-12:
   `23 exact, 0 known-divergence, 0 mismatched`, no renderer changes
   needed** (port doc §12). The prompt half of this step is closed; the
   "should produce `tool_calls`, not prose" half is **not**, and is blocked
   locally by the Metal MoE bug (§5, port doc §13) — carry it into step 4
   on the pod. The sequence below still reproduces the gate verbatim:

   ```bash
   cd ~/Documents/Liszt_ai/pie
   source ~/.venvs/pie/bin/activate
   CFG=integrations/qwen-code/pie_config_metal_coder.toml

   ./target/release/pie --config $CFG model import \
       mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit      # ~17 GB
   ./target/release/pie --config $CFG doctor                # must not say "cannot boot"

   PIE_CONFIG=$CFG ./target/release/pie serve > /tmp/pie.log 2>&1 &
   grep -q "standalone serving" <(tail -f /tmp/pie.log)     # wait for ready

   python3 integrations/qwen-code/shim.py --pie ws://127.0.0.1:18080 --port 8123 \
     --wasm tests/inferlets/target/wasm32-wasip2/release/chat_completions.wasm \
     --manifest tests/inferlets/chat-completions/Pie.toml > /tmp/shim.log 2>&1 &

   # 1a. does it generate at all? (first request is slow: PTIR compile)
   curl -s -m 600 http://127.0.0.1:8123/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"say hi"}],"max_tokens":16,"stream":false}'

   # 1b. THE GATE — prompt bytes must match the model's own template
   python3 integrations/qwen-code/parity/check_render.py \
       --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct
   ```

   Expected: `23 exact, 0 known-divergence, 0 mismatched`. Anything in the
   MISMATCH column is a rendering bug — the script prints the first
   diverging byte and a diff; fix `render_text.rs` until it is exact.
2. Add a t=0 knob (inferlet + shim passthrough) so decoding can be pinned.
   **Start by checking what already works**: sending `"temperature": 0` in
   the request body demonstrably changed the served output (and made greedy
   repeats identical), so the per-request path may already be wired end to
   end — observed, not yet traced in `handler.rs`. If it is, the remaining
   work is only forcing it on for qwen-code, which never sends the field.
3. Extend `check_render.py` coverage to the `/no_think` class by capturing
   fixtures with `enable_thinking:false`.
4. Only then: GPU pod, run 3, two arms, two repeats, `summarize.py`.
5. Optional: widen the task battery beyond 5 short tasks — trajectory
   equality over 5 tasks is weak evidence.
6. Upstream candidates: CUDA ≥ 12.9 gate in `driver/cuda/src/ops/gemm.cpp`
   (blocks stock cuda-12.4 images), and the python-client auth sentinel.
