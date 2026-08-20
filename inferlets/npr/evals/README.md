# NPR-on-pie eval harness

avg@8 on AIME 2025, comparing the inferlet's join modes against a sequential
baseline — the headline question of the port: *does user-space NPR reproduce
NPR Engine quality and speed?*

## Files

| File | What |
|---|---|
| `aime25.jsonl` | 30 problems (AIME 2025 I + II) from `math-ai/aime25`, `{id, problem, answer}` |
| `run_eval.py` | concurrent sweep driver over a running `pie serve` |
| `score.py` | avg@k / pass@k plus parallelism + throughput stats |
| `collapse.py` | per-arm `blocks<=1` vs `blocks>=2` split — the mixture the penalty A/B turns on |
| `results/` | JSONL output, one line per run (gitignored except committed summaries) |

## Arms

| Arm | Inferlet input | What it measures |
|---|---|---|
| `refill` | `join_mode=refill` | the faithful join: sibling KV refilled at overlapping positions behind BRLE hole masks — numerically equivalent to NPR Engine's page stitching |
| `textual` | `join_mode=textual` | phase-1 baseline: sibling steps concatenated causally at sequential positions. Off-distribution for the RL'd model (DESIGN.md §6.5); the A/B quantifies how much the faithful join buys |
| `adopt` | `join_mode=adopt` | the same faithful join done as a device-side KV row copy instead of recomputation (`adopt_kv`, DESIGN.md §13). §11 found it identical to `refill` on quality and 2.2x faster on join latency, so it is the default arm for quality work |
| `adopt_nopen` / `refill_nopen` | `rep_penalty=1.0` | ablation of NPR's per-`<step>` repetition penalty 1.02 (DESIGN.md §15), which is otherwise on by default from the first fork onward. Pair either against its penalised namesake to isolate the penalty |
| `sequential` | `max_plans=0` | never forks. The model still emits `<guideline>/<plan>/<step>`, but decodes the whole trace in one causal stream — NPR's own degrade-to-sequential path. The speed reference |

The sequential arm is the same checkpoint and the same schema, so accuracy
differences isolate the *execution* model, not the prompt. Note the arms are not
budget-matched by construction: NPR charges branch tokens `× parallel_degree`
against `max_new_tokens` (`schedule_batch.py:693-697`, reproduced in the
inferlet's ledger), so a 5-way round burns budget 5× faster than sequential
decoding does. `score.py` reports both `gen tok` (real tokens) and `chg tok`
(ledger charge) so the accounting stays visible.

## Why a run stopped

Every result carries `stop_reason` (and the derived boolean `budget_exhausted`)
straight from the inferlet, because the `fmt` column — the share of runs with no
`\boxed{}`, the "unanswered" rate — cannot on its own tell a run that reasoned to
the end and never boxed an answer from one that was cut off mid-thought:

| `stop_reason` | meaning | `budget_exhausted` |
|---|---|---|
| `eos` | the trunk hit chat-template end-of-turn | false |
| `step_end` | the trunk closed a `</step>` at top level | false |
| `budget` | the trunk ran out of global token budget | **true** |
| `branch_terminal` | a branch hit end-of-turn inside a parallel block | false |
| `branch_budget` | a branch ran out of global token budget | **true** |
| `branch_step_cap` | the `max_step_tokens` test hook capped a branch | false |
| `step_cap` | the `max_step_tokens` test hook capped the trunk | false |

`branch_budget` and `branch_step_cap` used to be folded into `branch_terminal`,
so budget starvation was indistinguishable from a genuine finish. `score.py`
prints a `bud` column (share exhausted) plus a per-arm breakdown of the
unanswered runs by `stop_reason`. Results recorded **before** the split score
`bud = nan` and show `branch_terminal`: that label is ambiguous by construction
and must not be read as "not starved".

## Methodology

**Accuracy and speed need different runs.** `elapsed_ms` is per-run wall clock;
under a concurrent sweep the engine batches runs together, so per-run latency
reflects contention, not the parallel-decode speedup. So:

- **accuracy sweep** — high concurrency, all three arms, k=8. Contention does not
  affect which tokens are sampled.
- **speed pass** — `--concurrency 1`, k=1, a subset of problems, `refill` vs
  `sequential`. This is the only number comparable to the paper's 4.6× decode
  speedup.

Sampling is temperature 1.0 / top_p 0.7 (NPR's settings), and the inferlet takes
no seed, so the k samples differ by engine RNG alone.

## Running

```bash
# server already up:  PIE_CONFIG=<cfg> pie serve
cd inferlets/npr
cargo build --release --target wasm32-wasip2

# accuracy: 3 arms x 30 problems x 8 samples = 720 runs
python evals/run_eval.py --arms refill,textual,sequential --k 8 \
    --concurrency 24 --out evals/results/aime25.jsonl

# speed: serial, first 8 problems, one sample each
python evals/run_eval.py --arms refill,sequential --k 1 --limit 8 \
    --concurrency 1 --no-upload --out evals/results/speed.jsonl

python evals/score.py evals/results/aime25.jsonl --by-problem
python evals/score.py evals/results/speed.jsonl
```

For an ablation A/B, score the mixture as well as the mean — an arm's accuracy
is `P(b>=2)·acc|b>=2 + P(b<=1)·acc|b<=1`, and the two halves are ~80% and ~5%
correct respectively (§11 read 5), so a change in the *block mix* and a change
in *conditional quality* are different findings that the mean cannot tell apart:

```bash
python evals/collapse.py evals/results/aime25-pen-ab.jsonl --baseline adopt_nopen
```

`run_eval.py` appends and resumes: re-running skips keys already recorded (pass
`--no-resume` to start over), so an interrupted sweep continues where it stopped
and a killed server costs only the in-flight runs.

## Protocol note

One WebSocket connection **per run**. The gateway's session loop replaces the
in-flight turn whenever a new client frame arrives
(`gateway/src/ingress/ws.rs:96`), so pipelining several `launch_process` frames
down one connection silently drops the earlier runs' event streams.
`launch_processes` (batch) exists but its turn never terminates cleanly — the
worker tracks only the first process (`worker/src/link/gateway.rs::turn_terminal`).
Per-run connections avoid both. The wasm is uploaded once up front; later
connections launch by name.

When a run fails, the guest-visible error is usually not the real one — see
HANDOVER.md §7.1 and `grep -a "future output failed" serve.log`.
