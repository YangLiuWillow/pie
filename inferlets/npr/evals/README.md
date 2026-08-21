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
| `budget_positional` | the trunk hit the engine's **primary** positional cap | **true** |
| `budget_charge` | the trunk hit the **secondary** ×degree charge cap | **true** |
| `branch_terminal` | a branch hit end-of-turn inside a parallel block | false |
| `branch_budget_positional` | a branch hit the positional cap | **true** |
| `branch_budget_charge` | a branch hit the ×degree charge cap | **true** |
| `budget`, `branch_budget` | **pre-split**: budget exhaustion, cap unknown | **true** |
| `branch_step_cap` | the `max_step_tokens` test hook capped a branch | false |
| `step_cap` | the `max_step_tokens` test hook capped the trunk | false |

`branch_budget` and `branch_step_cap` used to be folded into `branch_terminal`,
so budget starvation was indistinguishable from a genuine finish. `score.py`
prints a `bud` column (share exhausted) plus a per-arm breakdown of the
unanswered runs by `stop_reason`. Results recorded **before** the split score
`bud = nan` and show `branch_terminal`: that label is ambiguous by construction
and must not be read as "not starved".

That split paid out on the first sweep that used it. The stranded (blocks≤1)
population had been characterised from **charge ratios** as roughly two
populations — some genuinely starved, some ending early on branch EOS. The labels
say otherwise: **42 of 46 stranded runs are `branch_budget`**, and the proxy
misclassified 15 of them. The collapse is one phenomenon, not two. The reason the
proxy failed is the subject of the next section.

### The two budget caps, and why they are labelled separately

**The prediction table below was pre-registered on 2026-08-20, before this change
was written and before any run read it. It is left exactly as written.**

`decode_segment` breaks to the same budget stop on either of two conditions:

```rust
if sh.ledger.borrow().position_exhausted(cur_pos)          // (a) positional cap
    || degree_charge + 128 >= sh.ledger.borrow().budget    // (b) ×degree charge
{ break Stop::Budget; }
```

(a) is the engine's **primary** check — `right_most_pos - init_input_len >=
max_new_tokens - 128`, metering the longest path through the parallel structure.
(b) is the ×degree charge against the cumulative ledger. Today both emit
`budget` / `branch_budget` and set `budget_exhausted: true`.

This was deliberately *not* split when `branch_terminal` was, because there was no
evidence the two behaved differently.

> **Correction, same day, before any run used this.** An earlier revision of this
> paragraph claimed that runs exhausting at **charge ratios as low as 0.517**
> *are* (a) firing while (b) sits nowhere near its cap. **That does not follow.**
> Those ratios are equally consistent with a charge peak that was refunded before
> the row was written — see the `charge_peak` note below. They are a good reason
> to *build* the discriminator; they are not evidence of its answer. The split is
> the instrument, not a confirmation of something already believed. Left visible
> rather than quietly rewritten, because the distinction between "motivates the
> measurement" and "predicts its result" is the whole reason to pre-register one.

The planned split is
`branch_budget_positional` / `branch_budget_charge` (and `budget_positional` /
`budget_charge` for the trunk), with `budget_exhausted` staying **true** for all
four so nothing downstream changes meaning. `score.py` reports a
`budget exhaustion by cap:` line; rows written before this split count as
`cause-unknown(pre-split)` there rather than being guessed at from charge ratios,
which is the proxy this split exists to replace.

**What each outcome will mean** — written down now so that a constant label is a
result rather than a disappointment:

| observed | conclusion |
|---|---|
| both labels appear | both caps bind, at different times; per-arm counts say which dominates |
| **only `*_positional`** | **the ×degree charge never binds in this workload** — *on the record rather than assumed*. Read together with `charge_peak`: if headroom stayed large, the multiplied ledger is simply not a factor and the positional cap is the only thing stranding runs; if headroom came close, the two caps are in tension and the margin is worth knowing before any budget change |
| only `*_charge` | the positional check never fires, and the reasoning that motivated this split was reading something else entirely; the ledger needs re-examining before any budget conclusion stands |
| neither appears | the re-run eliminated budget exhaustion outright; the stranded population should vanish with it |

A label whose value never varies is only wasted if nobody wrote down in advance
that its constancy was informative. Same discipline as `over_device_capacity`
in `BUG16-SCOPE.md` §7.1.

### `charge_peak`, and why `tokens_charged` could not answer this

`Ledger::refund_join` hands back the `(degree-1)×` multiplier when a parallel
block closes, so the multiplied charge exists only **transiently, inside an open
block**. `tokens_charged` in the result JSON is the **post-refund** value, and the
transient peak was never stored anywhere — it could not be recovered after the
fact by any amount of re-analysis.

That is the real root of the misclassification above, and it is deeper than
"positional versus charge": the proxy was not merely coarse, **the field it
reasoned from had already had the multiplier subtracted out of it.**

`charge_peak` records the pre-refund high-water mark, so
`token_budget - charge_peak` is the ×degree cap's actual headroom (it trips at
`<= 128`). `score.py` prints its min/median/max. With the label split alone you
learn which cap fired; with `charge_peak` you also learn how close the other one
came — which is the difference between one pod and two.

**Note on framing:** this is *not* an investigation of §14's ×degree-versus-
per-sequence budget question. That framing is dead — across all 100 rows of the
penalty A/B, including 54 multi-block runs averaging 5.2 branches,
`tokens_charged == tokens_generated` **exactly, every row**, so the cumulative
×degree starvation §14 diagnosed was already fixed by that refund. The live
question is narrower: **which of the two caps inside the budget stop actually
binds**, and whether our positional semantics match the reference engine's.

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
