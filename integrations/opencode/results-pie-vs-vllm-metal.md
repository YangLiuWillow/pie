# pie vs vLLM-metal, opencode workload — first measurement, 2026-08-12

*Machine: `Lius-MacBook-Pro`, M-series, 48 GB unified, macOS 26.5.1, Metal.
One server at a time (each model is ~20 GB). Same replayed transcript, same
`max_tokens=96`, `temperature=0`, `max_model_len=16384` on both stacks.*

**Baseline stack:** [`vllm-metal`](https://github.com/vllm-project/vllm-metal)
`0.3.0.dev20260813023306` on vLLM `0.27.0`, a community hardware plugin that runs
MLX checkpoints on Apple Silicon — so both stacks serve **the same
`mlx-community` artifact**, on the same box, with no cross-hardware confound.

> An earlier version of the plan recorded "vLLM does not run on Metal" as a hard
> blocker requiring a rented CUDA box. That was wrong, and the correction is
> what made this measurement possible at all.

## Result — Qwen3-Coder-30B-A3B-4bit, MATCHED CONFIG

Re-run 2026-08-12 after a config-parity check found pie running with a KV pool
12× smaller than the baseline's. Both stacks now at prefill chunk 2048; pie's KV
pool raised 16,384 → 65,536 tokens. Matching was worth **1.50× on arm A and
1.31× on arm B** on its own.

| turn | pie A | pie B | vLLM | pie B ttfc | vLLM ttfc | pie B gen | vLLM gen |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37.13 s | 26.54 s | 6.76 s | 24.52 | 5.95 | 96 | 40 |
| 2 | 27.38 s | 3.88 s | 0.67 s | 1.84 | 0.46 | 96 | 11 |
| 3 | 28.42 s | 3.92 s | 0.64 s | 1.87 | 0.43 | 96 | 11 |
| 4 | 35.59 s | 3.97 s | 0.68 s | 1.90 | 0.47 | 96 | 11 |
| 5 | 42.81 s | 4.02 s | 0.68 s | 1.92 | 0.47 | 96 | 11 |
| 6 | 32.10 s | 4.07 s | 0.69 s | 1.95 | 0.47 | 96 | 11 |
| **total** | **203.44 s** | **46.41 s** | **10.12 s** | | | | |

*(Superseded starved-config totals, kept for the record: pie A 304.53 s,
pie B 60.99 s, vLLM 10.30 s.)*

**Do not read those totals as a 6× ratio.** pie generated 96 tokens every turn
(hitting `max_tokens`); vLLM generated 11. Decode dominates once the prefix is
cached, so the totals compare different amounts of work. The comparable numbers:

| | pie (Strategy B) | vLLM-metal | ratio |
|---|---:|---:|---|
| cold prefill rate | 296 tok/s | **1216 tok/s** | **4.1× vLLM** |
| steady-state ttfc | 1.90 s | **0.46 s** | **4.1× vLLM** |
| steady-state decode | 46.2 tok/s | 51.7 tok/s | 1.12× vLLM (~parity) |
| prefix reuse | ~97.5% cached | 78.9% hit rate | both work |

Strategy B over Strategy A, matched: **4.4×**.

**Both stacks reuse the prefix.** vLLM's APC is on and hitting — 82.2% per its
own logger. So this is not "session inferlet vs uncached baseline"; it is two
caching stacks, and the difference is the kernels underneath.

**The honest summary: on this workload and this machine, vLLM-metal is ~4×
faster than pie at prefill, ties at decode, and Strategy B does not close that
gap.** Strategy B is a 4.4× win *against pie's own uncached baseline* and remains
so; it does not make pie competitive with vLLM here. The per-token deficit is
uniform across dense and MoE and across 50× of model size — see
`results-prefill-profile.md`.

### Qwen3.6-35B-A3B (GDN hybrid)

| | pie A | vLLM |
|---|---:|---:|
| total, 6 turns | 125.21 s | **20.37 s** |
| cold prefill | ~570 tok/s | ~1362 tok/s |

pie Strategy B has **no resume on hybrid models** (blocked upstream — see
`results-ab-strategy-b.md`), so pie A is pie's best available arm here. vLLM's
docs matrix marks Automatic Prefix Cache ❌ for the Qwen3.5/3.6 hybrid row, but
the installed build reports an **80.2% hit rate** on it, so the matrix is behind
the build. pie generated *fewer* tokens than vLLM on every turn here (38/5/14/18/20/40
vs 64/34/50/53/59/49), so it is not losing by doing more decode work.

## What this measurement does NOT establish

1. **The prompts are not byte-identical between stacks.** pie renders through its
   WIT template; vLLM renders the checkpoint's `chat_template.jinja`. Turn 1:
   7268 tokens (pie) vs 7235 (vLLM) on the Coder-30B, 7351 vs 7471 on the 35B —
   0.5% to 2.7% apart. That difference is small in size and evidently large in
   effect: pie ran to `max_tokens` every turn while vLLM stopped after 11 tokens,
   which is a behavioural divergence, not a timing one. **Fixing this is the next
   prerequisite**, and vLLM exposes `/v1/chat/completions/render` for exactly
   that comparison (the trick `parity/capture_vllm_render.py` uses on
   `liu/qwen-code-dev`).
2. **This is a Metal result — but the gap is NOT Metal-only.** An earlier
   version of this section cited the OpenHands evaluation as measuring pie
   **~26% faster** than litellm+vLLM on CUDA, and used it to argue the Metal
   result was an artifact. **That was wrong twice over.** The 26% came from a
   baseline the runpod handover calls *crippled* (`--enforce-eager` plus an
   untuned MoE kernel); their fair-parity rerun puts pie at **parity to ~9%
   behind** on an H100, with **prefill specifically 1.86× behind** (24.1k vs
   44.9k tok/s). So the prefill gap exists on tuned CUDA too. Metal widens it to
   4.4–6.4× (see `results-prefill-profile.md`); it does not create it.
3. **Single run, no repetitions, shared laptop, batch-1, single tenant** — the
   regime the project's own honest performance frame says pie loses in. Nothing
   here touches multi-tenant load or subagent forking, which is where pie's
   programmability is supposed to pay.
4. **No accuracy or trajectory claim.** Nothing here compares outputs.

## Two harness bugs this run exposed

**The warm-up was priming the cache.** `bench_ab.py` warmed with turn 1's own
messages, which on a cache-enabled backend converts turn 1 from a cold prefill
into a hit — and only on arms that *have* a cache. It put vLLM's turn-1 ttfc at
0.338 s, an implausible ~22k tok/s prefill, while pie's turn 1 stayed genuinely
cold (pie excludes its own final boundary as a resume candidate). Two arms, two
different meanings for "turn 1". The warm-up now uses an unrelated short prompt.

**`cached_tokens` is unpopulated by vLLM-metal.** It reports `0` in
`usage.prompt_tokens_details` while its own logger reports an 82.2% hit rate, so
the bench's `cached` column is meaningless for that arm and the server log is the
only source. Reading the usage field alone would have produced "vLLM has no
prefix caching", which is false and would have flattered pie by ~4×.

## Reproduce

```sh
# baseline arm
export PATH="$HOME/.venv-vllm-metal/bin:$PATH"
vllm serve mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --port 8000 --served-model-name coder30b --max-model-len 16384 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder
python3 integrations/opencode/bench_ab.py --arm vllm --turns 6 \
  --base-url http://127.0.0.1:8000 --model coder30b --out /tmp/ab_vllm.json
grep -oE "Prefix cache hit rate: [0-9.]+%" <server log> | tail -1   # NOT usage.cached_tokens

# pie arms: see results-ab-strategy-b.md
```

Tool calling needs `--enable-auto-tool-choice --tool-call-parser <hermes|qwen3_coder>`
on vLLM or every request with `tools` 400s; pie has it built in. Qwen3.6 takes
`hermes`, Qwen3-Coder takes `qwen3_coder`.
