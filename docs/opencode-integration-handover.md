# opencode ↔ pie — handover

**For:** a fresh Claude Code session picking this up cold.
**Written:** 2026-08-13, superseding the 2026-08-12 version (that one ends at
"PB.1 / Strategy B is next"; everything after §2 here is newer).
**Branch:** `liu/opencode-integration`, **36 commits ahead of `fork/`, nothing
pushed.** Push is the user's call, not yours.
**Tree:** `~/Documents/Liszt_ai/pie-opencode`.

Read this, then `docs/opencode-integration.md` (design) and
`integrations/opencode/results-*.md` (measurements). Those are durable; this
file is the bridge and is meant to be rewritten.

---

## 0. Orient yourself in five minutes

```sh
cd ~/Documents/Liszt_ai/pie-opencode
git status -sb && git log --oneline -5      # liu/opencode-integration, clean, ahead 36

cargo test -p pie-openai-serving            # 73 passed
cargo test -p pie-model                     # 19 + 3 + 4 + 23 + 4 + 3 passed
cargo test -p pie-model-qwen-3 --all-features   # 30 passed
```

Native driver tests (build dir is stable; rebuild with `cargo build -p pie-bin
--release --features driver-metal` first if it is missing):

```sh
BD=target/release/build/pie-worker-89d6a7186bcf3091/out/metal/build
$BD/bin/llama_pso_test          # 32 passed, 0 failed
$BD/bin/llama_numerics_test     # 51 passed, 18 FAILED  <- expected, see §5
```

**`llama_numerics_test`'s 18 failures are pre-existing and are not a
regression.** It fails dense cases at `rel_l2 34.5` on a family that serves
real checkpoints correctly. Use it **differentially** — same binary, one knob
moved — never as an absolute verdict. That is how it was used to size the
attention accumulator, and there it was decisive.

Live suites need a server (§3):

```sh
PIE_BASE_URL=http://127.0.0.1:8080 python3 integrations/opencode/test_acceptance.py
# 25 passed, 0 failed, 0 warnings — on BOTH the 0.6B and Coder-30B
```

---

## 1. What this is

Make **pie** (programmable serving: inferlets, explicit KV control) the backend
for **opencode** (a coding agent), then benchmark pie vs vLLM-metal on
SWE-bench using opencode as the harness.

Two strategies, both live:
- **A** — stock opencode → gateway OpenAI ingress → one `chat-completions`
  inferlet per request. KV dies with the request. This is what everything below
  was measured on unless stated.
- **B** — `session_shim.py` → one long-lived `opencode-session` inferlet over a
  sticky WebSocket, KV retained across turns. 5/5 resume, 2/2 head sharing.

---

## 2. Where things stand

| | state |
|---|---|
| Prefill vs vLLM-metal (0.6B, matched artifact) | **1.97×** behind, was 4.63× |
| Decode | ~parity (1.12×) |
| Qwen3-Coder-30B | **works** — 25/25, 0 warnings, tool calls, multi-step agent loops |
| SWE-bench, 5 known-solvable | **pie 4/5 resolved**, vLLM 1/5 — read §6 before quoting |
| opencode | installed, 1.18.18, `~/.opencode/bin/opencode` |
| Docker scoring | works on this Mac, §4 |

### What this session changed

1. **Matrix-unit attention for llama-family heads — 2.35× on prefill.**
   `sdpa_paged_mma.metal` existed and was wired for gpt-oss only. Now
   instantiated at D=128 and selected for llama (which serves Qwen3-0.6B *and*
   the Qwen MoEs). The speedup **grows with context** (1.08× at 251 tokens,
   2.29× at 5,052) — a quadratic term fixed, not a constant shaved.
   Needed an unanticipated precision fix: `simdgroup_multiply_accumulate`
   carries its accumulator in the fragment type, so a score is a chain of `D/8`
   half roundings. Capping the chain and folding into float between chunks
   fixed the numerics **and was 36% faster**. d=64 keeps one chunk, so gpt-oss
   is untouched bit for bit.
2. **Three Coder-30B bugs, each hiding the next.** The router was read at the
   wrong quantization width (8-bit tensor, 4-bit kernel → arbitrary expert
   routing → fluent garbage); then an empty `<think>` block silenced a model
   with no thinking channel; then the wrong tool dialect meant it could not
   call anything. All fixed. The model now matches mlx-lm byte for byte on a
   greedy prompt, *including* the reference's own quirk.
3. **SWE-bench end to end**, drive + Docker scoring, on this laptop.

---

## 3. Running a server

Always use the boot helper — see §7 for why:

```sh
integrations/opencode/tools/boot_pie.sh <tag> \
    PIE_MODEL=<artifact> [PIE_MAX_MODEL_LEN=32768] [PIE_KV_TRACE=1]
```

It exists because three things have to be true and two of them bit hard: it
kills by `release/pie -c` (**not** `"pie serve"`, which matches nothing —
§7), waits for `/health`, and then **proves the live process is the one it
started** by matching that boot's own mktemp config path. `/health` answering
`ok` is not proof anybody's server is yours.

Models (`~/.pie/models/`):
- `Qwen--Qwen3-0.6B-optimized` (default), `mlx-community--Qwen3-0.6B-4bit`
- `mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit` ← the agent model
- `mlx-community--Qwen3.6-35B-A3B-4bit` (different driver family, head_dim 256)

Knobs added this session: `PIE_MAX_MODEL_LEN` (default 16384; agents need
32768 — an over-long prompt is **refused**, not chunked) and `PIE_KV_TRACE=1`.

---

## 4. Environment that is now set up

- **opencode 1.18.18** at `~/.opencode/bin/opencode`. Provider config is
  `integrations/opencode/opencode.json` (`pie/…` and `vllm/…` entries). It must
  be **in the cwd** — without it you get `ProviderModelNotFoundError` disguised
  as a generic server error.
- **Docker**: colima + lima + static docker CLI, all user-local, no sudo, no
  Homebrew. `export PATH=$HOME/.local/lima/bin:$HOME/.local/bin:$PATH`, then
  `colima start --cpu 6 --memory 14 --disk 80 --vm-type vz --vz-rosetta`.
- **Python**: `<scratchpad>/venv-shim` (3.12) has `msgpack blake3 websockets
  cryptography typer toml datasets swebench`. The system `python3` is 3.9 and
  **cannot** run the shim (`str | Path` annotations).
- **vLLM-metal**: `~/.venv-vllm-metal/bin/vllm`.

### SWE-bench scoring on Apple Silicon

Three non-obvious things, all needed together:

```sh
colima start … --vz-rosetta                      # x86_64 images via Rosetta
docker pull --platform linux/amd64 <image>       # harness pulls without a platform flag
--dataset_name SWE-bench/SWE-bench_Verified      # NOT princeton-nlp/… (needs the `image` column)
```

Four graded instances ran in **2 min 4 s**. `--cache_level` no longer exists.

---

## 5. Open items, most valuable first

1. **pie wears out under sustained agent load.** After a ~9-minute session,
   every later request to that server returns `completion_tokens: 0` with
   `finish_reason: "length"` — including "Say hello in one word." A fresh
   server answers fine; restart clears it. **The KV-exhaustion hypothesis was
   tested and refuted** (§7): with `PIE_KV_TRACE=1`, `live_ws` returns to 0 and
   the pool returns to full between requests. Cause is open. Two attempts to
   reproduce it under tracing failed — reproducing it reliably is step one.
2. **`total_pages` is a knob that lies.** `simple_family.cpp:313` computes
   `g_.total_pages = kv_max_ctx / kv_page_size`, overwriting the configured
   value. The pool is exactly one max-length sequence (1024 pages at 32k, 512
   at 16k, verified both ways). Not the wear defect, but wrong.
3. **The quantized GEMM** — still ~2.4× behind MLX's `quantized_matmul` on
   these shapes. Largest remaining prefill item now that attention is fixed.
4. **`llama_numerics_test`'s 18 failures** — mostly MoE, one at `rel_l2 14.0`.
   Either real or the test is over-tight; nobody has determined which.
5. **Prompt parity with vLLM** — `parity/check_render_vllm.py` exists and has
   not been re-run since the dialect fix. Do this before any trajectory claim.
6. **Possible opencode session bleed** — during one repro the model referenced
   a path from a *different* workspace with a corrupted UUID. opencode keys
   projects by git remote, so same-repo workspaces may share state. **If true
   it contaminates both benchmark arms.** Unconfirmed: my tee proxy truncates
   payloads at 12 KB and hid the evidence. Raise the limit and re-check.

---

## 6. How to read the SWE-bench number

**pie 4/5, vLLM 1/5** — with four caveats, all of which cut against it:

1. **The arms were not symmetric.** pie ran with `--restart-cmd` (fresh server
   per instance, to isolate the wear defect); vLLM did not. Giving one arm a
   mitigation the other lacks is how a flattering number is made. **A fair
   re-run restarts both.**
2. **n = 5.** One instance moves it 20 points.
3. **The set is deliberately biased** — the first five of the thirteen that
   `litellm+qwen3-coder-30b-a3b-t0` resolved on a neutral 50 (the OpenHands
   branch's `resolved_ids`). That bias is the *point*: on instances the model
   has already solved, a zero is attributable to the stack rather than to the
   task. It is a debugging instrument, **not a resolve rate**.
4. **The vLLM arm hit a tool-call defect** (1/3 schema-valid vs pie's 3/3 on a
   direct probe), so the comparison measures tool-call fidelity at least as
   much as serving. It is *not* a claim about prefill, decode or KV reuse.

Do not quote 4/5 as a SWE-bench score. The honest framing: pie carried 4 of 5
trajectories the model is known to be capable of, against a baseline that
carried 5/5 on the same five.

---

## 7. Traps that cost hours here

Read this section. Every item below was paid for.

- **`pkill -f "pie serve"` matches nothing.** The process is
  `pie -c <config> serve`. Three "restarts" silently lost the port to the first
  server, `/health` answered `ok` throughout, and **both arms of an A/B ran
  against one process**, producing the entirely reasonable conclusion "the new
  kernel is no faster". Kill by `release/pie -c`, and verify the live process
  is *yours*.
- **A throughput number cannot distinguish "no faster" from "never ran".**
  Hence `PIE_METAL_SDPA_TRACE=1`, which prints which attention was chosen and
  which clause decided. Use it whenever a kernel change appears to do nothing.
- **opencode masks every failure** as
  `{"name":"UnknownError","message":"Unexpected server error"}`. Always rerun
  with `--print-logs`. Two unrelated failures wore that mask in one hour; one
  was pie's, one was a missing provider config.
- **`subprocess.run([opencode, …])` fails 100%; `zsh -c` with the same argv,
  cwd and byte-identical env succeeds 100%.** Ruled out: env, stdin, capture,
  `close_fds`, `start_new_session`, PATH, fd limits. Cause unknown. Invoke
  through a shell.
- **Two hypotheses of mine were wrong this session, both the same shape**: a
  plausible mechanism asserted from a partial read, then contradicted by
  looking at the thing itself.
  - *KV exhaustion* — refuted by instrumenting the pool. My first reading of
    the trace "confirmed" it; the numbers were an artifact of tracing at
    retire-**entry** with linearly growing prompts. The line that disproved it
    was on the same object and I had not looked at it.
  - *vLLM lacks schema typing* — refuted by reading one file further.
    `_qwen3_arg_converter` in `vllm/parser/qwen3.py` does store raw strings,
    but `vllm/parser/engine/parser_engine.py` calls `find_tool_properties` and
    `coerce_to_schema_type` one layer up. vLLM types from the schema.
  **Before asserting a mechanism, go and look at it.** Both corrections were
  cheap; both claims had already been written down as findings.
- **Instrumentation that truncates hides the evidence you built it for** — the
  tee proxy's 12 KB cap is why item §5.6 is still unconfirmed.
- **`kv_pages.allocated` / `available` are defined in `telemetry.rs` and
  recorded nowhere.** `PIE_KV_TRACE=1` is the only pool observability.

---

## 8. Suggested next session

Pick one, in this order:

1. **Reproduce the wear defect reliably** (§5.1). Everything about pie under
   agent load is blocked behind it, and it is the only defect here that a user
   would hit.
2. **Fair re-run of SWE-bench** — restart *both* arms per instance, and widen
   to the full 13 known-solvable. That turns a debugging instrument into a
   defensible comparison.
3. **The quantized GEMM** (§5.3) if you want throughput rather than
   correctness.

Whatever you pick: measure first, and when a mechanism suggests itself, open
the file before writing it down.
