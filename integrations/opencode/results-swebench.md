# SWE-bench Verified, 5 instances — first run, 2026-08-13

**Result: 0/5 patches, and the reason is a pie defect, not the model.**

Harness: `integrations/opencode/run_swebench.py` (drive half), stock
opencode 1.18.18, pie serving `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`
at `max_model_len=32768`, `total_pages=4096`. Instances are the seeded subset
(seed 1234) of `princeton-nlp/SWE-bench_Verified`.

| instance | state | secs | patch |
|---|---|---:|---:|
| astropy__astropy-13398 | ran | 534 | 0 B |
| django__django-11749 | no-op | 2 | 0 B |
| django__django-16333 | no-op | 2 | 0 B |
| sphinx-doc__sphinx-8035 | no-op | 1 | 0 B |
| sympy__sympy-24066 | no-op | 1 | 0 B |

Only the first instance ran at all. The other four returned in ~1 s having
generated nothing.

## The defect: pie degrades to zero-token completions

After the first instance's 534-second agent session, **every subsequent
request to that server returned `completion_tokens: 0`** — including
requests that had nothing to do with the benchmark:

```
prompt="Say hello in one word."   gen=0  finish=length  content='…'
```

The same server had answered that prompt normally an hour earlier. A
freshly booted server on the identical config answers it correctly
(`'Hello!'`, 3 tokens). So the server does not fail — it **wears out**.

The signature is specific and worth keeping:

- `completion_tokens: 0` with `finish_reason: "length"`, which is
  self-contradictory: nothing was generated, yet the stop reason is the
  length cap.
- Visible content is `'…'`, which is opencode rendering an empty message,
  not the model emitting an ellipsis.
- Independent of prompt size (14-token prompts fail the same as 1,340-token
  ones) and of `max_tokens` (80 and 300 behave identically).
- Cleared completely by a restart.

**Working hypothesis: KV/working-set exhaustion.** The first instance was a
long multi-turn agent session at 32k context against a 4,096-page pool
(131,072 tokens of KV). If retained working sets are not released when a
session ends, the pool fills and later requests get no room to decode. That
matches "everything after the big session fails, restart fixes it", but it is
a hypothesis — the pool was not instrumented during the run, and the degraded
server's log carried no admission or allocation warning.

## Why no replay found this

Every suite in this integration is short and bounded: the acceptance suite
replays five captured requests, the resume suite five turns, the profiler
sweeps five prompts and exits. **Nothing before this ran a real agent for
nine minutes and then asked the server for one more token.** That is the
whole value of the benchmark harness independent of any score it produces.

## What is NOT the problem

Ruled out by measurement during this run:

- **The model.** A fresh server answers correctly, and the same checkpoint
  passes 25/25 acceptance with zero warnings.
- **Tool calling.** Instance 1's agent ran 534 s of real tool use.
- **Prompt size or content.** All five prompts fail identically on a worn
  server and all five are well under the context ceiling.
- **opencode.** Its generic `{"name":"UnknownError","message":"Unexpected
  server error"}` masks the real error; `--print-logs` shows the true cause
  each time. Two distinct failures wore that same mask here, and only one of
  them was pie's (see below).

## A second, unrelated bug found on the way

`ProviderModelNotFoundError: Model not found: pie/qwen3-coder-30b` — the
harness copies `opencode.json` into each workspace and `capture_patch`
deletes it afterwards, so any *re-run* in a used workspace has no provider
config. Harmless to the benchmark (the copy happens before every attempt)
but it cost an hour of misdiagnosis, because opencode reports it as the same
"Unexpected server error" as everything else.

## Next

1. **Instrument the KV pool** and re-run instance 1 followed by a trivial
   request. If pages are not returned, that is the bug and it is in session
   teardown, not the kernels.
2. Until then, a per-instance server restart would make the benchmark
   *run*, but it would also hide the defect, so it is deliberately not done
   here.
3. Scoring still needs Docker, which this machine does not have:
   ```sh
   python -m swebench.harness.run_evaluation \
       --dataset_name princeton-nlp/SWE-bench_Verified \
       --predictions_path preds.jsonl --max_workers 4 --run_id pie-opencode
   ```
