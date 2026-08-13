# SWE-bench Verified against pie — 2026-08-13

**Officially graded: 4/5 resolved.** pie serving Qwen3-Coder-30B-A3B on Metal,
driven by stock opencode 1.18.18, scored by
`swebench.harness.run_evaluation` in Docker.

| instance | agent | patch | graded |
|---|---:|---:|---|
| django__django-12276 | 545 s | 411 B | **resolved** |
| django__django-13028 | 1265 s | 0 B | empty patch |
| django__django-13089 | 332 s | 915 B | **resolved** |
| django__django-14373 | 309 s | 412 B | **resolved** |
| django__django-15569 | 843 s | 483 B | **resolved** |

Artifacts: `preds-swebench-known5.jsonl`, `report-swebench-known5.json`.

The one miss produced no patch at all after 21 minutes — the agent worked and
committed nothing. Nothing was scored wrong; nothing was scored generously.

## Read the instance set before reading the number

These are **not** a random sample, and 4/5 is not a resolve rate. They are the
first five of the thirteen that `litellm+qwen3-coder-30b-a3b-t0` resolved on a
neutral 50 (the OpenHands branch's `baseline_t0_full_50.report.json`
`resolved_ids`), scored by the same Docker grader.

The set is deliberately biased, and the bias is the point. A seeded random
subset conflates two failures — the serving stack misbehaving, and the model
being unable to do the task — and SWE-bench Verified's base rate for a 30B is
low enough that 0/5 is unremarkable for a *healthy* stack. On instances this
model family has already solved, a zero is attributable. That is what makes
this set a debugging instrument rather than a score.

So: **this measures whether pie can carry a trajectory the model is known to
be capable of.** It cannot be quoted as a SWE-bench result, and the honest
comparison is against the baseline's 5/5 on the same five.

## The run before this one: 0/5, and why

The seeded-random run scored 0/5 for two reasons, only one of them the model's:

1. **pie wore out.** After the first instance's 534-second agent session,
   every subsequent request to that server returned `completion_tokens: 0` —
   including "Say hello in one word.", which the same server had answered
   normally an hour earlier. A freshly booted server on the identical config
   answers correctly. Restart clears it completely.
2. The instances were random, so even a healthy stack would likely have
   scored 0.

This run isolates (1) with `--restart-cmd`, which boots a fresh server before
each instance. That is **isolation, not a fix** — the defect is still there,
and a benchmark that quietly restarts around it would hide it. With the
defect isolated, 4 of 5 land.

### The wear defect, still open

Signature worth keeping:

- `completion_tokens: 0` with `finish_reason: "length"` — self-contradictory:
  nothing generated, yet the stop reason is the length cap.
- Content renders as `'…'` — opencode drawing an empty message, not the model
  emitting an ellipsis.
- Independent of prompt size (14-token prompts fail like 1,340-token ones)
  and of `max_tokens`.
- Cleared by a restart, every time.

Working hypothesis is KV/working-set exhaustion: the long session ran at 32k
context against a 4,096-page pool (131,072 tokens of KV), and if retained
sets are not released at session end the pool fills and later requests get no
room to decode. **Hypothesis, not finding** — the pool was not instrumented,
and the worn server logged no allocation warning. Next step: instrument the
pool, run one long session, then one trivial request. If pages are not
returned, the bug is in session teardown rather than the kernels.

**Why no replay found it.** Every other suite here is short and bounded: five
captured requests, five turns, five swept prompts. None runs a real agent for
nine minutes and then asks the server for one more token.

## Scoring on Apple Silicon

Docker on this box is colima + lima + the static docker CLI, all installed
user-local without sudo or Homebrew.

The one thing that does not work out of the box: SWE-bench's published images
are **x86_64 only** (`swebench/sweb.eval.x86_64.…`), and the harness pulls
without a platform flag, so Docker refuses with *"no matching manifest for
linux/arm64/v8"*. Two things fix it together:

```sh
colima start --cpu 6 --memory 14 --disk 80 --vm-type vz --vz-rosetta
docker pull --platform linux/amd64 <each image>   # then the harness finds them locally
```

Rosetta translation makes those containers cheap — the four graded instances
ran in 2 min 4 s total. Note also that SWE-bench 5.0 needs the
`SWE-bench/SWE-bench_Verified` dataset, not `princeton-nlp/…`: the harness
reads an `image` column the older copy does not have, and dropped
`--cache_level`.

## Reproduce

```sh
# drive (fresh server per instance isolates the wear defect)
python3 integrations/opencode/run_swebench.py --known-solvable --n 5 \
    --timeout 1500 --out preds.jsonl --restart-cmd '<boot pie>'

# score
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path preds.jsonl --max_workers 2 --run_id pie_known5
```
