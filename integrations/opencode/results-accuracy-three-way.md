# Is pie's accuracy at parity with mlx-lm and vLLM-metal? — 2026-08-17

**Yes. Graded 4/10 each.** SWE-bench Verified, ten instances, stock opencode,
`swebench.harness.run_evaluation` in Docker, **every arm restarting its server
per instance**.

| engine | resolved | known-solvable | unseen | which |
|---|---:|---:|---:|---|
| **pie** | **4/10** | 3/5 | 1/5 | 10914, 13089, 14373, 15569 |
| **mlx-lm** | **4/10** | 3/5 | 1/5 | 11099, 12276, 14373, 15569 |
| **vLLM-metal** | **4/10** | 4/5 | 0/5 | 12276, 13089, 14373, 15569 |

Each engine solves a different four. Only 14373 and 15569 fall to all three, and
**no engine solves 11133, 13028 or 13158** — three instances defeat everyone,
which caps the field at 7/10 and is a property of the tasks.

Generation: pie 1641 s, mlx-lm 1160 s, vLLM 2055 s for ten instances each.

## The August result over vLLM does not survive symmetry

`results-swebench.md` reports pie 4/5 against vLLM-metal 1/5 and flags in its own
caveats that the arms were asymmetric: pie got `--restart-cmd` and a fresh server
per instance, vLLM ran one server throughout. It called that "exactly the kind of
asymmetry that produces a flattering number."

It did, and the number was ours. On the same five known-solvable instances, with
both arms restarting per instance, **vLLM scores 4/5** — the same as it scores
here — against pie's 3/5. The original gap was substantially the mitigation, not
the engine.

## Where pie is genuinely behind: one instance, to a loop

**django-12276.** pie produced 0 bytes with a transcript repeating one sentence
45 times. mlx-lm and vLLM both resolved it the same night, and **pie itself
resolved it in August**. That is the only concrete accuracy cost attributable to
pie tonight, and it is n=1 — pie gains 10914 where mlx fails, which is why the
totals tie.

### Degenerate repetition is the checkpoint's, not pie's

Splitting the loops by what is repeating:

| | pie | mlx-lm |
|---|---:|---:|
| prose/decode loops | **3** | **3** |
| tool-retry loops | 1 | 0 |

Exactly tied on decode degeneration. Every prose loop in **both** engines opens
with the same signature — *"Looking at the code more carefully…"* — and 13158
loops in both on near-identical wording. A shared failure text across
independent implementations is close to direct evidence of a model property.

pie's one extra (11099, 132×) is the agent re-issuing an identical failing
`sed`: tool-error handling, not decoding, and it should not be counted with the
others.

## First bisect result: speculation is implicated

| arm | patch | maxrep |
|---|---:|---:|
| speculation **on** (tonight's config) | 0 B | 45 |
| speculation **off** (`SPEC_OFF=1`) | **1333 B** | **4** |

Pending the control arm reproducing the failure, this points at speculation —
and the mechanism was established independently earlier the same day:

> `rows == 1` selects the **split-K** kernel; `rows > 1` selects the
> **head-sharing** kernel (plus `_u4` above the 8192 context gate).

A verify fire carries `DRAFT_K + 1 = 5` rows, so **speculation computes its
tokens with a different attention kernel than the sequential decode it is
assumed equivalent to.** `engine.rs`'s accept test is itself correct, but its
equivalence argument silently assumes matching logits, and on this driver they do
not match. Different rounding → different argmax near ties → a different
trajectory; a prompt-lookup drafter then locks any repetition in, because once a
line recurs it proposes the whole line and greedy verification confirms it in
bulk.

`SPEC_OFF=1` is a one-line mitigation available today, at the cost of
speculation's throughput.

## Instrument gap this exposed

**Every throughput instrument in this repo scores a degenerate loop as a win**,
because looping finishes sooner. pie's arm ran ten instances in 27 minutes
against ~10 minutes *per instance* in August, and four of those runs were
spinning. `tools/transcript_health.py` closes the gap with the cheapest check
that works — the most-repeated line in an agent transcript, engine-neutral so it
runs over pie, mlx and vLLM alike:

    1-3 healthy | 4-9 watch | >=10 degenerate (every such run produced 0 bytes)

## Caveats, all of which cut against reading too much into this

1. **n = 10.** One instance moves any arm by 10 points.
2. **The known-solvable five are the August run's wins**, biased toward
   solvable. Only the unseen five are an unbiased read, and there the scores are
   1/5, 1/5, 0/5 — too small to separate anything.
3. **Quantizations differ.** pie serves its own 4-bit conversion; mlx-lm and
   vLLM serve `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`. These are not
   bit-identical models, so a per-instance difference need not be an engine
   difference.
4. **Nothing here measures agent quality.** A resolved instance is the agent and
   the engine together.

## Reproduce

```sh
integrations/opencode/tools/swe_overnight.sh        # three arms, symmetric
integrations/opencode/tools/swe_grade_overnight.sh  # Docker grading
integrations/opencode/tools/repetition_bisect.sh    # speculation vs kernels
integrations/opencode/tools/transcript_health.py DIR [DIR2]
```

Grading needs `export PATH=$HOME/.local/lima/bin:$PATH` — colima bundles its own
lima there, and without it `colima status` reports `lima not found` and sends you
to a package manager that is not installed, while the VM is already running.

## Loop rate, measured — and a discrepancy that is not explained

Twenty fresh runs on the shipped configuration, five reps each, every run
restarting the server exactly as the graded run did:

| instance | runs | looped | rate | non-empty patches |
|---|---:|---:|---:|---:|
| 11099 | 5 | 0 | 0% | 3 |
| 12276 | 5 | 1 | **20%** | 2 |
| 13158 | 5 | 0 | 0% | 0 |
| **14373** *(control)* | 5 | **0** | **0%** | 4 |
| **pooled** | **20** | **1** | **5%** | |

The control behaves: 14373 never loops and patches 4/5, so the metric is not
firing on everything.

**Tonight's graded run had 4 of 10 instances looping — 40%.** Against a 5%
rate, the chance of 4-or-more loops in ten instances is **0.001**. These two
measurements are not samples from the same distribution, and I cannot say why.

What differs between them, none of it tested:

* the graded arm ran ten instances through ONE `run_swebench.py` invocation
  sharing one workdir; the rate runs each got a fresh workdir;
* the graded arm ran first, after a day of GPU benchmarking; the rate runs
  followed two hours of SWE-bench;
* ordering — the graded arm interleaved ten different instances, the rate runs
  repeated one instance five times.

Until that is resolved, **the loop rate is not a stable property of the engine**,
and the honest use of these numbers is narrow:

1. A single agentic run's loop count says nothing. 12276 loops 20% of the time;
   the graded run caught one of those and I read it as a regression.
2. Failure and degeneration are largely independent. 13158 produced zero patches
   in five runs and never looped once; 12276 produced no patch in three runs of
   which only one looped. "Most failures are loops" was one run's coincidence.
3. Something makes loops CLUSTER within a run. That is worth chasing, because if
   it is the shared workdir it is a harness defect contaminating the graded
   numbers, and if it is machine state it is a confound in every agentic
   measurement this repo takes.
