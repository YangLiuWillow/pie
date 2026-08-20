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

Pending — sweep in flight. Numbers land here with the build sha and GPU.
