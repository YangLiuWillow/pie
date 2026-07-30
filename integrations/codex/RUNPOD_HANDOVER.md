# RunPod handover — Codex ↔ Pie on Qwen3-30B (MoE)

State of the work, what's verified, and the exact path to running the Codex
CLI against Pie on a GPU pod. Written 2026-07-30.

## 0. What exists and what's verified (macOS M2, Qwen3-0.6B, CPU driver)

Branch: **`liu/codex-integration`** in the pie repo (this checkout:
`~/Desktop/Lin_startup/pie-codex`).

- `inferlets/codex-responses/` — WASM HTTP server speaking the OpenAI
  Responses API that Codex uses for custom providers (`wire_api =
  "responses"`). Native Qwen tool calling (template-rendered schemas,
  byte-identical history replay, streamed `function_call` items) plus
  **content-addressed KV-session reuse**: each turn saves its context as an
  engine snapshot named `codex/{prompt_cache_key}/{hash(instructions, tools,
  item-prefix)}`; the next turn `Context::open`s it and prefill only covers
  the new tool outputs.
- `integrations/codex/` — `run_pie_codex.sh` (pie serve :18080 + HTTP daemon
  :8123), `launch_daemon.py`, `pie_config.toml` (0.6B portable), this doc,
  `pie_config_runpod_30b.toml` (30B MoE cuda_native, copied from the proven
  H200 config on `openhands-integration-updated`).
- `~/.codex/config.toml` on the Mac already points at
  `http://127.0.0.1:8123/v1`.

Verified end-to-end on the Mac:
- Codex CLI (0.146.0, `codex exec`) completes turns against the daemon; the
  full wire contract (SSE events, item shapes, ids, usage) is accepted.
- Structured `function_call` emission from `<tool_call>` blocks.
- Two-turn KV resume: turn 2 reported `cached_tokens: 194 / input 215` —
  only the tool output + cue paid prefill.

Known-good but slow: Codex's real system prompt is ~7.2k tokens; CPU prefill
runs ~10 tok/s, which is why a GPU is needed for actual use. Two engine-level
traps were hit and fixed on the branch (keep them in mind if configs change):
- The scheduler's `request_timeout_secs` default (120s) abandons responses
  whose forward outlives it while the driver keeps grinding — every queued
  request then starves. Config now sets 600s, and the inferlet prefills in
  1024-token chunks with an SSE keepalive per chunk (Codex kills streams
  silent > 5 min).
- `pie serve` caches installed programs; after rebuilding the wasm, restart
  the server — reinstalling over a live server does not take effect.

## 1. Pod selection (both requirements, independently)

- **Compute capability ≥ 9.0** (H100/H200, sm_90). Pie's fast attention
  paths are gated to sm_90+; an A100 (sm_80) silently disables them and is
  not a valid Pie arm.
- **Host driver ≥ 580** (CUDA 13). vLLM/torch comparisons need it, and
  `libcuda` is bind-mounted from the host — a wrong driver cannot be fixed
  by changing the container image. Set RunPod's CUDA-version filter to
  13.0+ *before* picking the GPU.

## 2. Storage rules (cost hours when violated)

- `/workspace` is MooseFS: repo, model weights, logs there. **Small-file
  work stalls indefinitely** (no error) — venvs, caches, and the cargo tree
  go on `/root`.
- `export UV_CACHE_DIR=/root/.uv-cache` before any `uv pip install`.
- `/root` is wiped on pod replacement; keep a bootstrap script.
- Reuse `integrations/openhands/runpod/00_setup_h200.sh` and
  `bootstrap_runpod.sh` from the `openhands-integration-updated` branch —
  they encode all of this and hard-fail on wrong driver/GPU.

## 3. Build

```bash
cd /workspace && git clone https://github.com/YangLiuWillow/pie.git pie-codex
cd pie-codex && git checkout liu/codex-integration
export CMAKE_CUDA_ARCHITECTURES=90
CARGO_TARGET_DIR=/root/cargo-target cargo build -p pie-server --release \
    --features driver-portable,driver-cuda
/root/cargo-target/release/pie driver cuda-native doctor   # <1s sanity check
(cd inferlets/codex-responses && cargo build --target wasm32-wasip2 --release)
```

## 4. Cherry-picks from `openhands-integration-updated` (ordered by value)

The codex branch predates the H200 performance/correctness work. For 30B MoE
these matter; hashes are on `openhands-integration-updated`:

1. **`d9aaf9f1`** — flash-decoding fix (`DecodeWorkEstimator` discarded
   FlashInfer's `split_kv` for batch ≤ 512, which is *every* decode). Without
   it, single-stream long-context decode is ~10× slow. Highest value.
2. **`e18b2953`** — OOB GPU write when compact-logits is off: `top_p`/`top_k`
   (which this Responses server sets by default!) trigger it on any prefill
   over ~512 tokens on 30B. **Required for correctness, not just speed.**
3. **`c77d91d9`** (runtime part) — `Instruct::create` takes `model_name` and
   routes `*coder*` → `ToolFormat::Coder`, `has_thinking: false`. Without it,
   Qwen3-**Coder**-30B (model_type `qwen3_moe`) resolves to the JSON
   `<tool_call>` format and **every tool call is unparseable**. Required if
   using the Coder model. Also bumps FlashInfer float workspace 80→512 MiB
   (needed for ~7k-token codex prefills on 30B).
4. `4a76fdbe` + `bdc3bb25` — tensor-core decode for qwen3_5(-MoE), default on
   at GQA ≥ 4 (Qwen3-30B-A3B is GQA 8). +20% decode.
5. `28e0b6c5` — fused QKV for the MoE forward (mind the `ws.qkv_fused`
   sizing guard).
6. `f1aac6eb`/`c77d91d9` (snapshot part) — namespace prefix-delete
   (`Context::delete("codex/{sid}/")` with trailing slash). On `cuda_native`
   with `swap_pool_size = 0`, leaked snapshots are hard allocations: enough
   abandoned conversations will *hang* (not slow) the pool. Until picked,
   restart `pie serve` between long sessions.

If cherry-picking is painful, the alternative is rebasing
`liu/codex-integration`'s inferlet + integration commits onto
`openhands-integration-updated` — the codex work touches almost no runtime
files (three small uncommitted-then-committed SDK/runtime patches), so
conflicts should be minimal.

## 5. Model + config

`pie_config_runpod_30b.toml` here is the proven H200 config
(Qwen3-Coder-30B-A3B-Instruct, cuda_native, `gpu_mem_utilization = 0.90`,
memory_profile auto; requires `PIE_CUDA_KV_PAGE_SIZE=32` in the env per its
header). Add the codex-specific scheduler knob if absent:

```toml
[model.scheduler]
request_timeout_secs = 600
```

On GPU you can raise the inferlet's prefill chunk (it's
`PREFILL_CHUNK: usize = 1024` in
`inferlets/codex-responses/src/handler.rs`) or leave it — chunking is
harmless when each chunk takes milliseconds.

Verify at boot: driver banner must print `prefill_decode_plan=on
xqa_decode=on` — but remember the `model_type=qwen3_moe` banner caveat: those
llama_like flags don't describe the MoE path; trust measurements, not the
banner (see `AGENT_HANDOVER_20260728.md` §3/§8 on the other branch).

## 6. Run + wire Codex

```bash
cd /workspace/pie-codex/integrations/codex
python3 -m venv /root/venv-codex && /root/venv-codex/bin/pip install -e ../../client/python
PIE=/root/cargo-target/release/pie CFG=$PWD/pie_config_runpod_30b.toml \
  VENV=/root/venv-codex bash run_pie_codex.sh
```

(The script's `VENV` default is `./.venv`; override as above or make the venv
in place. First boot downloads ~58 GB of weights — be patient; readiness is
grepped from the log.)

Codex on your laptop through an SSH tunnel:

```bash
ssh -L 8123:127.0.0.1:8123 root@<pod>
# ~/.codex/config.toml already points at http://127.0.0.1:8123/v1
codex   # set model = "qwen3-coder-30b" (name is informational; pie serves its default model)
```

Or install codex on the pod (`npm i -g @openai/codex`) and copy the
`[model_providers.pie]` block from this repo's README.

## 7. Verification sequence (in order, cheap → expensive)

1. `curl -s http://127.0.0.1:8123/` — daemon up.
2. Non-streaming small request (README "Smoke tests") — `output_tokens > 0`.
3. Streaming request with the shell tool — a `function_call` item appears.
4. Re-send with the echoed call + a `function_call_output` —
   `usage.input_tokens_details.cached_tokens` ≈ previous context size, and
   the `debug` field says `resume hit …`.
5. `codex exec --skip-git-repo-check --json "Run ls and tell me what you see"`
   — expect a `command_execution` item and `cached_input_tokens > 0` from
   the second round-trip on.

Assert token counts, not exit codes — multiple past runs returned rc=0 with
zero iterations and plausible-looking logs.

## 8. Open items / known limitations

- Model load happens per HTTP request (fresh WASM instance per request;
  ~ok on GPU, the load is a handle lookup — but the *persistent daemon over
  session::send/receive* pattern from `9d77878e` on the updated branch is
  the right long-term fix and also what enables holding a Context live).
- Non-`function` tools (Codex's `custom` grammar tool `apply_patch`) are
  accepted but not rendered; with the Coder tool-format cherry-pick,
  consider mapping `apply_patch` to a function schema.
- Snapshot GC: one live boundary per conversation branch (previous boundary
  deleted after the next save), but abandoned conversations leak one
  snapshot each until the namespace-delete cherry-pick lands.
- Small-model quirks on 0.6B (repeating the same tool call, parroting) are
  model capability, not protocol — expect them to vanish on 30B.
