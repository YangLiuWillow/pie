# test-time-bench methods, applied to opencode + qwen

`shsym/test-time-bench` (TTB) is a benchmark control plane with a mature answer
to the reporting problems this integration kept hitting by hand. This directory
adopts its **methods**; it does not vendor its code, and nothing here has been
validated by its backend.

Two-step plan, of which this is step 1:

1. **(here)** Bring TTB's contract into this repo and use it for our own runs.
2. Port the result upstream as an opencode agent runner inside TTB.

## What was adopted, and what it fixed here

| TTB method | What it replaced |
|---|---|
| `inferletbench.run_summary.v2` artifact | Numbers living in a markdown table, with no machine record of what was run |
| Cryptographic provenance (`engine_config_sha256`, `pie_version`, image digests) | Boot helpers that prove liveness at run time but leave no trace in the result |
| Budgets declared **and enforced**, or reported null | A `--timeout` that appeared nowhere in the output |
| Exhaustion counters (`wall_time_exhausted_cases`, …) | "empty patch after 21 minutes" — unattributable between agent failure and clock expiry |
| Scorer pinned by **digest** | `docker pull …:latest`, a moving grading target |
| Dataset snapshot pinning the rendered prompt | A `PROMPT` constant in the harness that could change without a version bump |

### The metric that immediately paid for itself

TTB reports `model_calls` and token totals per run. Recomputed over the two
already-graded arms, from opencode's own session store:

| | pie | vLLM-metal |
|---|---:|---:|
| resolved | 4/5 | 1/5 |
| **mean model calls / case** | **8.2** | **2.8** |
| tokens out | 15,744 | 5,959 |
| tokens in | 550,268 | 177,430 |
| mean case latency | 600 s | 58 s |

`results-swebench.md` argued the vLLM arm "gave up" from wall-clock alone —
10–146 s against pie's 309–1265 s. **Mean 8.2 model calls against 2.8** is that
claim as a measurement rather than an inference. The `resolved` column
reproduces the official grader exactly (4 and 1), which is what validates the
join.

## Files

- `config.opencode-qwen3-8b.json` — benchmark definition in TTB's
  `benchmark-config.v3` *shape*. Deliberately labelled `-local-profile`: it has
  never been validated by TTB's backend, and its `execution.agent` is an
  external process, not TTB's `agent_loop.rs`.
- `pin_images.sh` — `write` records grader image digests; `verify` fails if a
  local image has drifted from the pin. Run `verify` before grading a recorded
  run.
- `images/digests.json` — the 11 grader images for the known-solvable set, as
  pulled 2026-08-13.
- `fetch_dataset.sh` — pulls TTB's `swe-bench-lite-first-20` snapshot. Fetched,
  not vendored: it is private and carries its own `LICENSE-DATA`.
- `../ttb_summary.py` — the emitter. `run_swebench.py` now always writes
  `<out>.cases.jsonl`, which this joins against `opencode.db`.

## Step 2 — what the upstream port actually has to be

The load-bearing fact: **TTB's decoder is a pie inferlet**, and the inferlet is
used *only* in the eval stage. Per case, the control plane sends one
`inferletbench.inferlet_case_request.v1` to a persistent decoder-host sitting
beside the GPU, which has loaded the compiled WASM (`decoder.wasm`) and its
`Pie.toml` at startup. Contract:

- `docs/per-case-inferlet-request-contract.md` — normative
- `backend/src/runner/native/one_shot.rs` — the wire contract and per-case loop
- `backend/src/runner/native/agentic.rs` — the agentic decoder factory
- `examples/inferlets/decoder-*-rust/src/lib.rs` — the inferlet sources

The request carries a durable identity `(run_id, attempt, case_id,
sample_index)` echoed by the engine so stale results are rejected, a budget
(`max_model_calls`, `max_output_tokens`, `max_wall_time_s`,
`answer_token_budget`), the case prompt, and KV cache-scoping keys — request
-local by default, with a `shared_scaffold_key` when the driver can prove
sharing is sound.

**This is why opencode cannot be ported as an inferlet.** An inferlet is WASM
running inside the engine; opencode is an external process that drives a
repository through its own tool loop. The seam is on the *runner* side, not the
decoder side: TTB's decoder-host already serves the model, and opencode already
speaks OpenAI to a server. So the port is an **opencode agentic runner** that
executes cases against the decoder-host's OpenAI endpoint, in place of
`agent_loop.rs`, reusing TTB's provisioning, budgets, scoring and publication
unchanged.

Two things that make that tractable, and one that does not:

- This repo already has the decoder half — `inferlets/openai-serving` plus the
  gateway ingress is the OpenAI surface an external agent needs.
- The budgets opencode cannot enforce (`max_model_calls`, agent steps) are
  exactly the ones TTB's own `H1` audit says are already not plumbed, so the
  port does not regress anything that currently works.
- **Unverified**: whether `agentic.rs`'s decoder factory can be bypassed
  cleanly for an external-agent runner, or whether the case executor assumes an
  in-engine decoder throughout. That is the first thing to read before
  promising the shape above — it is inferred from the contract and the audit
  notes, not from having read the executor.

## Honesty notes carried over from TTB's own audits

TTB's `H1-agentic-budget-not-plumbed` finds budgets that are schema-validated
but never plumbed, with the run summary reporting a hardcoded constant as the
declared contract — its shipped SWE-bench config says `max_steps: 1` while runs
use 64 model calls and 600 s.

That failure mode is designed out here rather than inherited: every budget in
`config.opencode-qwen3-8b.json` carries an `enforced` flag, and
`ttb_summary.py` emits `null` for anything unenforced instead of printing the
declared value. **Only two budgets are real** — a wall-clock timeout, and a
per-*call* output cap from `opencode.json`. opencode exposes no step cap and no
model-call cap (`opencode run --help`), so those are declared `null` with the
reason recorded.

`success_rate` is likewise `null` unless a grader report is supplied. It is not
inferred from patch bytes: a non-empty patch is not a resolved instance, and
`nonempty_patch_rate` is reported separately so the two can never be confused.

## Model choice, and how far it is from TTB's

TTB's engine configs are CUDA-only. `qwen3-32b-cuda0.toml` is bf16 across two
80 GB cards (~65.6 GB of weights) and does not fit this box in its configured
dtype; serving it 4-bit would deviate further than changing model size. So
**`qwen3-8b`** is the target.

Identity and hardware are split into `models/qwen3-8b.toml` and
`execution-profiles/local-metal-48gb.toml`, which is TTB's own structure and
exists for a reason it states plainly: *"Execution settings live under
configs/execution-profiles/ so model comparisons do not absorb hardware."* This
repo has already made that mistake once, reading a KV-pool difference as a
kernel result.

**TTB registers `Qwen/Qwen3-8B`** at revision `b968826d…`, `quantization = ""`,
with a weight digest over the safetensors manifest. That digest is re-verifiable
without downloading the weights, and was verified here before being trusted —
sha256 over the sorted `"<file> <lfs-sha256>\n"` manifest of all 5 shards, read
from the HF API, reproduces `sha256:d11fc503…` exactly.

**Its weight precision cannot be reproduced on this hardware.** Not
inconvenient — impossible. pie's Metal llama path binds every projection
through an affine-U4 kernel and refuses a checkpoint with no `.scales`
tensors. `mlx-community/Qwen3-8B-bf16` was downloaded (15.3 GiB) and imported
before the driver said so, which is the cost of a mistake worth naming:
`activation_dtype = "bfloat16"` in TTB's engine cell is the **activation**
dtype, and matching it does not require bf16 **weights** — the Metal driver
runs bf16 activations over int4 weights, which is what every prior measurement
in this repo did.

So this box serves `mlx-community/Qwen3-8B-4bit` (`545dc425…`,
`sha256:690df13b…`, pinned by the same recipe), which is also what vLLM-metal
serves, keeping both arms on the same file.

**A number produced here is therefore not comparable to a TTB leaderboard row
for `qwen3-8b`** — four independent reasons, and the first one dominates:
4-bit weights vs unquantized, different hardware, a different artifact
container, and an external agent in place of `agent_loop.rs`. It is comparable
to another run under this profile.

**This changes what a score means.** The `KNOWN_SOLVABLE_BOTH` set in
`run_swebench.py` is the instance list a *Coder-30B* is known to solve; it is
not a debugging instrument for an 8B, and a zero on it would be unattributable.
That is why this profile targets TTB's own `swe-bench-lite-first-20` instead —
a fixed, model-independent set, where honesty comes from budgets, pinned
grading and reported telemetry rather than from a model-specific instance list.
Expect a low resolve rate from an 8B and read the telemetry, not the score.
