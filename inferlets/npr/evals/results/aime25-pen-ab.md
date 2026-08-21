# Repetition penalty 1.02 — AIME 2025 A/B

`adopt` (penalty 1.02, DESIGN.md §15) vs `adopt_nopen` (penalty 1.0), the arm
that produced HANDOVER §11's 0.460. Raw rows are `aime25-pen-ab.jsonl`
(gitignored by convention); this file is the committed summary.

## Baseline decomposition — what the A/B can actually resolve

Computed from §11's raw rows (`aime25-rerun.jsonl`, L40S, 2026-08-17) with
`collapse.py`, not quoted from the §11 prose:

| arm | n | avg | unanswered | share b>=2 | acc \| b<=1 | acc \| b>=2 |
|---|---|---|---|---|---|---|
| adopt | 50 | 0.460 | 0.200 | 0.540 | 0.043 (1/23) | 0.815 (22/27) |
| refill | 50 | 0.460 | 0.260 | 0.620 | 0.000 (0/19) | 0.742 (23/31) |
| textual | 50 | 0.360 | 0.360 | 0.340 | 0.091 | 0.882 |
| sequential | 50 | 0.440 | 0.020 | 0.000 | 0.440 | — |

The mixture is exact for adopt: `0.540 x 0.815 + 0.460 x 0.043 = 0.460`. So the
arm's accuracy is almost entirely *how often a run reaches a second parallel
block*, and barely at all how well it reasons once there.

**This is what the headline question costs.** Holding both conditionals fixed,
reaching the paper's 0.504 needs `share(b>=2)` to move 0.540 -> 0.597: **+2.2
correct runs out of 50.** At n=50 per arm the Wilson interval on `avg` is about
+/-0.13. So a 25 x k=2 sweep **cannot** resolve "does the penalty close
0.460 -> 0.504" through the mean — the target effect is roughly three runs and
sits far inside noise. What this sweep can resolve is the mechanism, where the
signal is much larger: the share of runs that escape the first block, the
unanswered rate, and the token spend of the runs that do not.

### What a *powered* mean would cost

Worth stating precisely, because the obvious next question is "so run more".

Priced as two independent proportions — `(1.96+0.84)^2 x (0.46x0.54 +
0.504x0.496) / 0.044^2` — the answer is **2,021 runs per arm**, and that is the
number to quote if the arms are compared as two marginal accuracies. It is also
the wrong model for this design.

Both arms run the *same* 25 problems, so the comparison is paired, and AIME
difficulty is close to bimodal: from §11's rows the between-problem variance of
latent `p_i` is 0.188 and the problem-level ICC is **0.767**. The model mostly
either solves a problem or doesn't. That makes `p(1-p) = 0.248` — the per-run
noise the unpaired formula charges — an overstatement of what a paired test
fights, which is the *within-problem* variance `E[p(1-p)] = 0.248 - 0.188 =
0.060`, **4.1x smaller**:

| analysis | runs per arm for 80% power at 0.044 |
|---|---|
| unpaired, two marginal proportions | 2,021 |
| unpaired but clustered (ICC 0.767, k=2) | ~3,570 |
| **paired on per-problem differences** | **487** |
| paired, if the effect clips near p=1 (realized 0.0335) | 838 |

So a powered mean costs **~500-1,000 runs per arm** — 5-10x this sweep, roughly
$15-25 of L40S time at the ~90 min/100 rows this sweep ran at. Affordable, not
prohibitive. But it buys less than it looks: **the mean can only ever say the
penalty worked, never which of the two failure populations below moved**, and
that is what picks the next fix. The mechanism route is the better buy on
information per dollar, not merely the fallback for a sweep that came up short.

Two conditions on the larger sweep, if anyone runs it. It must be **analyzed
paired** — 500 runs/arm compared as two independent Wilson intervals will
overlap and be miscalled null, having paid for the data and discarded the
design. And ignoring the pairing is worse than the naive figure suggests, not
better: clustering inflates the k=2 design to ~3,570 runs/arm.

## The collapse has two causes, not one

Every one of adopt's 23 `blocks<=1` runs reported `stop_reason=branch_terminal`,
which on §11's code conflated "a branch hit end-of-turn" with "a branch ran out
of global budget". Their charge-to-budget ratios separate them:

| charged / budget | runs | reading |
|---|---|---|
| >= 0.90 | 14 | starved — ran the ledger out inside the first block |
| 0.46 – 0.75 | 9 | a branch emitted EOS early and terminated the whole run |

By contrast the `blocks>=2` stratum averages 0.455 of budget: runs that get out
of the first block finish with more than half the ledger unspent.

Closing 0.460 -> 0.504 through this channel requires **12.4% of the stranded
runs to convert** to blocks>=2 — a concrete prediction the mechanism read can be
scored against, and one the mean cannot express.

The repetition penalty is a plausible fix for the first group only — §15's
theory is that it shortens repetition-inflated branches. The second group is a
separate defect (one branch's EOS killing a multi-branch run) and no sampling
penalty addresses it. **Ceiling estimate:** converting all 14 starved runs at
the observed `acc|b>=2` of 0.815 would take the arm to ~0.67; converting three
of them reaches the paper's 0.504. The penalty therefore has ample headroom —
the question is whether it converts any.

This is also why the sweep runs on `a35a68ac9`+ rather than §11's code: the
`branch_terminal` / `branch_budget` split lands the starved-vs-EOS distinction
directly in the rows instead of inferring it from a charge ratio.

## CUDA validation of the penalty path (settled)

`e14082eb0` shipped the `TopLogits` raw-logit probe validated on Metal only.
It is now validated on CUDA, on an L40S (46,068 MiB, driver 570.124.06), build
`cf204689a` — whose `inferlets/npr/src/lib.rs` is blob-identical to the audited
`a35a68ac9`:

```
[npr] selftest toplogits: ids_match=true max_dev=0.000000 max_logit=21.875
```

`ids_match` compares the `TopLogits{k:10}` ids against a `Distribution` probe on
the *same* forward pass; `max_dev` is the max |softmax(logits) - prob| over the
renormalised top-10. §12 expected `max_dev <= 1e-3` on CUDA (bf16->f32 widening
on the host vs the softmax kernel) and allowed for drift; it came back exactly
**0.000000**, matching Metal. `max_logit = 21.875` is a real logit magnitude,
not the 0-1 that would mean the sentinel path returned probabilities — so
`compute_dist_slots`' temp-0 `gather_bf16_rows` branch is correct on CUDA.

The full selftest passes alongside it (`selftest_pass: true`):

| equivalence | TV |
|---|---|
| `mask`, `mask_4run`, `positions`, `noise_floor` | 0.0000 (exact) |
| `hole_natural` | 0.0047 |
| `refill_short` | 0.0062 |
| `refill` | 0.0269 |
| `adopt` | 0.0269 |
| `control` (causal, must differ) | 0.5512 |
| `adopt_control` (causal, must differ) | 0.6468 |

`refill`/`adopt` at 0.027 sit inside the 0.001-0.03 CUDA kernel-noise band §3
documents, and the two controls confirm the masks bite (the H200 measured 0.55
on the same control). **`adopt` and `refill` agree to 1.5e-5 of each other**,
which is the phase-3 KV-graft join reproducing the refill join's distribution on
CUDA rather than merely compiling there.

## Pre-registered prediction

**Committed before the sweep's rows landed** (this sweep was at 11/100 rows at
the time of this commit; `git log` on this file establishes the ordering, and
the raw JSONL was not inspected beyond counting lines). Stated so a reader can
tell it was not fitted after the fact.

From the baseline decomposition, if the repetition penalty closes 0.460 -> 0.504
through the collapse channel — i.e. by un-stranding trajectories rather than by
making reasoning better once parallel — then:

> **12.4% of the `blocks<=1` runs must convert to `blocks>=2`**, moving
> `share(b>=2)` from 0.540 to 0.597, while `acc|b<=1` (0.043) and `acc|b>=2`
> (0.815) stay put.

Three outcomes, distinguished in advance:

1. **`share(b>=2)` rises by roughly the predicted amount and the conditionals
   hold** — the penalty works, by the mechanism §15 claims.
2. **`share(b>=2)` is flat** — the penalty is not the lever; the next suspect is
   the budget-exhaustion path, and `budget_exhausted` on these rows says which
   half of the stranded population is which.
3. **A conditional moves instead** — the penalty is changing reasoning quality,
   not the block mix, which is not what §15 predicts and would need explaining.

At n=50/arm, a move of 5.7 points in `share(b>=2)` is ~3 runs and will not be
separable from noise. So the honest reading is directional: the prediction is
falsifiable in *sign and rough size*, not in significance, and the writeup below
says which of the three it landed on without dressing a directional read as a
conclusion.

## Method

- 2 arms x 25 problems (AIME 2025, `--limit 25`) x k=2 = 100 runs, `--concurrency 8`,
  `--prompt-cache`, 30,000-token ledger budget, temperature 1.0 / top_p 0.7,
  `--abort-after 10`. Identical to §11's protocol so the `adopt_nopen` arm is a
  direct replication check on 0.460.
- CUDA `TopLogits` sentinel gated the sweep before it started
  (`pod/chain.sh`): `ids_match=true` against a `Distribution` probe on the same
  forward pass, and `max_logit` a real logit magnitude rather than the 0-1 that
  would mean the sentinel returned probabilities.
- Scored with `score.py` (avg@2 / pass@2 / unanswered / budget share) and
  `collapse.py` (block-stratum split, Wilson 95%, token spend per stratum,
  `stop_reason x blocks`).

## Result

L40S (46,068 MiB, driver 570.124.06), build `cf204689a` (`lib.rs` blob-identical
to the audited `a35a68ac9`), 100/100 rows, **zero engine errors**, 2026-08-21.

| arm | n | avg@2 | pass@2 | unanswered | budget-exhausted | share b>=2 | acc \| b<=1 | acc \| b>=2 | gen tok |
|---|---|---|---|---|---|---|---|---|---|
| `adopt` (penalty 1.02) | 50 | **0.520** | 0.560 | 0.220 | 0.420 | **0.540** | 0.217 | 0.778 | 18,547 |
| `adopt_nopen` (penalty 1.0) | 50 | **0.480** | 0.520 | 0.200 | 0.420 | **0.540** | 0.261 | 0.667 | 19,436 |
| delta (pp) | | +4.0 | +4.0 | +2.0 | **+0.0** | **+0.0** | -4.3 | +11.1 | -889 |

**Replication check first:** `adopt_nopen` is §11's configuration, and it
reproduces §11's `adopt` at 0.460 with 0.480 — inside noise. The baseline holds,
so the comparison below is against a control that behaves as expected.

### Pre-registered prediction: outcome 2 — the share is flat

The prediction required `share(b>=2)` to move 0.540 -> 0.597. It moved
**0.540 -> 0.540**. Not "within noise of no change": the counts are *identical*,
**27/50 in both arms**. So is budget exhaustion — **21/50 in both arms**. On the
metric §15 says the penalty exists to move, the penalty did exactly nothing.

Two further checks, both against the penalty:

* **It did not shorten stranded branches.** §15's mechanism is that repetition
  inflates branch length and burns the x-degree ledger. In the `blocks<=1`
  stratum the penalty arm generated **25,926** mean tokens against the control's
  **25,378** — slightly *more*, not less.
* **The unanswered rate did not fall** (0.220 vs 0.200), and every unanswered run
  in both arms is `branch_budget` — 11 and 10 respectively.

### Where the +4.0 came from, and why it is not a result

Entirely from `acc|b>=2`: 21/27 correct vs 18/27, **+3 runs**. Wilson intervals
[0.59, 0.89] against [0.48, 0.81] overlap across most of their range. `acc|b<=1`
moved the other way (5/23 vs 6/23). At n=50 this is exactly the ~3-run,
inside-noise move the method section said the mean could not resolve — it landed
in the predicted direction and remains unresolvable, which is why the mechanism
was pre-registered as the deciding metric rather than the mean.

So: **does the penalty close 0.460 -> 0.504?** No. The penalty arm reads
**0.520 against a same-sweep control of 0.480** — that pairing is the number,
and neither half should ever be quoted without the other in the same sentence,
because 0.520 alone reads as the penalty clearing the paper, which is the
inverse of the finding. The gap between them is 3 runs, and the channel that
would have to carry a real improvement is provably flat. The honest statement is
that the penalty is **quality-neutral within this sweep's resolution and
mechanistically inert** on the collapse.

**This is answered, not underpowered.** The distinction matters: an
underpowered result is one where more data would resolve it. Here three
independent channels came back flat — `share(b>=2)` identical at 27/50, budget
exhaustion identical at 21/50, and the penalty failing its own stated mechanism
by generating *more* tokens in the stranded stratum. A powered mean sharpens an
estimate; it cannot make a flat channel non-flat. More runs would refine the
+4.0 pp and change nothing about the conclusion.

### Correction: the collapse has one cause, not two

The pre-sweep section above split the `blocks<=1` population into ~14 starved
and ~9 killed by early branch EOS, inferred from charge-to-budget ratios on
§11's rows. **The `stop_reason` split refutes that.** With real labels, of 46
`blocks<=1` runs across both arms: **42 `branch_budget`, 3 `eos`, 1
`branch_terminal`.**

The proxy failed for a reason worth recording: the ledger's primary check is
**positional**, not charge-based (`Ledger::position_exhausted`), so a run can
exhaust its budget positionally while `tokens_charged` sits well below it.
**15 of the 42 `branch_budget` runs had charge ratios under 0.90, as low as
0.517** — every one of which my proxy would have called an early EOS. The
conclusion moves accordingly: the collapse is *one* phenomenon, budget
exhaustion, and the "9 early-EOS runs" were never there.

This is `pr-split`'s `stop_reason` split (`a35a68ac9`) paying for itself
immediately: it turned an inference that was wrong into a measurement, on the
first sweep that carried it.

### What this points at next

The lever is not the sampler. **42 of 46 stranded runs died of budget/positional
exhaustion**, unchanged by the penalty, and §14's second suspect — whether the
paper's 30,000-token evaluation budget is ledger-charged x degree as we
replicate it, or effectively per-sequence in their eval path — is now the
leading explanation for the remaining gap to 0.504. That is a
configuration question answerable far more cheaply than a powered A/B: re-run
one arm at a per-sequence-equivalent budget and see whether the `blocks<=1`
population survives.
