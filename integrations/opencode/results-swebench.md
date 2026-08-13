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

## vLLM-metal on the same five — 1/5

Same agent (opencode), same checkpoint, same prompts, same instances, same
machine, graded by the same Docker harness.

| instance | pie | vLLM-metal |
|---|---|---|
| django-12276 | **resolved** (545 s) | unresolved (146 s) |
| django-13028 | empty patch (1265 s) | empty patch (88 s) |
| django-13089 | **resolved** (332 s) | empty patch (10 s) |
| django-14373 | **resolved** (309 s) | **resolved** (39 s) |
| django-15569 | **resolved** (843 s) | empty patch (11 s) |
| **resolved** | **4/5** | **1/5** |

**Read this with three caveats, all of which cut against the headline.**

1. **The arms were not symmetric, and that is my doing.** pie ran with
   `--restart-cmd`, so it got a fresh server per instance; vLLM ran on one
   server throughout. The restart exists to isolate pie's own wear defect,
   and giving one arm a mitigation the other did not get is exactly the kind
   of asymmetry that produces a flattering number. A fair re-run restarts
   both.
2. **n = 5.** One instance either way moves this by 20 points.
3. **The set is the baseline's wins**, so it is biased toward "solvable",
   not toward either stack.

**What is real regardless of the score** is *how* the vLLM arm failed. Its
runs are an order of magnitude shorter — 10, 11, 39, 88, 146 s against pie's
309–1265 s — which is an agent giving up, not an agent working faster. And
instance 1 said why: `Edit … failed: The edit tool was called with invalid
arguments: SchemaError`.

That was a guess when first written. It has since been measured directly,
same checkpoint and the same `edit` schema opencode actually sends, three
identical prompts exercising the boolean `replaceAll`:

| trial | pie | vLLM-metal |
|---|---|---|
| 1 | `filePath, oldString, newString, replaceAll=true` ✓ | `{"path": …}` — 3 required missing |
| 2 | ✓ | ✓ (`replaceAll=true`) |
| 3 | ✓ | `{}` — empty arguments, 3 required missing |
| **schema-valid** | **3/3** | **1/3** |

**Retraction.** I wrote that this was "the same class of bug pie fixed —
arguments must be typed from the schema". I then read vLLM's parser, and that
attribution is wrong. The parser is:

- `vllm/tool_parsers/qwen3_engine_tool_parser.py` — an 8-line adapter;
- `vllm/parser/qwen3.py` — the XML grammar and state machine
  (`<tool_call>` / `<function=` / `<parameter=`), whose `_qwen3_arg_converter`
  does store every value as a raw string;
- `vllm/parser/engine/parser_engine.py` — the engine, which then calls
  `find_tool_properties(self._tools, func_name)` and `coerce_to_schema_type`
  on the result.

So vLLM **does** type arguments from the tool schema; it just does it one
layer up from the converter I first read. Its trial-2 `replaceAll=true` is
that coercion working. pie has no advantage here, and the sentence claiming
one is withdrawn.

**What the measured failures actually are**, re-read with that in mind:
`{"path": …}` is a *wrong parameter name*, and `{}` is *nothing extracted* —
neither is a typing error, and schema coercion cannot repair either. A wrong
name is the model's; an empty extraction is either the model emitting
something unparseable or the engine's segmentation. **I did not isolate
which**, and n=3 cannot carry the attribution.

**Why this was not "fixed".** vLLM's parser is a compiled Rust extension
(`_rust_tool_parser.abi3.so`); this build exposes no `--tool-parser-plugin`,
so the checkpoint's own Python reference parser cannot be loaded in its
place, and `qwen3_coder` / `qwen3_xml` are two names for the same adapter.
More decisive than either: **vLLM is the baseline.** A vLLM patched by us is
not the thing the comparison is against, and the honest place for this fix is
upstream. What belongs here is a reproduction, and that is what the table is.

Read together with the run above: the vLLM arm is handicapped by a tool-call
defect, so 4/5 vs 1/5 is measuring tool-call fidelity at least as much as it
is measuring serving. That is a real difference and it matters to an agent —
but it is not a claim about prefill, decode, or KV reuse.

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

**The KV-exhaustion hypothesis was tested and does not survive.** The pool is
now instrumented (`PIE_KV_TRACE=1`), which it never was — `kv_pages.allocated`
and `kv_pages.available` are defined in `telemetry.rs` and recorded nowhere.
Across sequential requests the trace reads:

```
install_ws   avail=1024/1024  live_ws=1  pending_recycle=0
retire_idle  avail= 974/1024  live_ws=0  pending_recycle=50
install_ws   avail=1024/1024  live_ws=1  pending_recycle=0   <- fully recovered
```

`live_ws` returns to 0 and the pool returns to full between requests. Working
sets are released and their pages do come back, so "exhaustion at session
teardown" is wrong as stated. Two attempts to reproduce the degradation under
tracing both failed — the server kept generating — so the cause is still open.

A correction worth keeping, because the shape of the error recurs: the
`retire_idle` numbers appeared to decline monotonically (974, 925, 877) and I
first read that as the leak. It was an artifact of two things I had chosen —
the trace sits at retire-*entry*, before retirement runs, and my probe prompts
grew linearly, so each line was the pool minus that request's own pages. The
`install_ws` line, on the same object, said the opposite; I had not looked at
it.

**Found while instrumenting, and it is real**: `simple_family.cpp:313` computes
`g_.total_pages = kv_max_ctx / kv_page_size`, silently overwriting the
configured `total_pages`. The KV pool is therefore sized to *exactly one
max-length sequence* — 1024 pages at `max_model_len` 32768, 512 at 16384,
verified both ways — and the config knob does nothing. That is not this
degradation (two concurrent 9k-token requests both completed), but it is a
knob that lies, on the path an agent workload stresses hardest.

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
