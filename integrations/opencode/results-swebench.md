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

## Pre-flight for the fair re-run — three blockers, all cleared 2026-08-13

**1. Prompt parity: passes.** `parity/check_render_vllm.py`, both stacks on the
same Coder-30B artifact: **2 of 5 fixtures byte-identical, 3 differing only by
JSON separator whitespace** in the embedded schema blobs of the XML tool
definitions (`{"type":"object",…}` vs `{"type": "object", …}`) — +21/+22 tokens
on ~7,200, 0.3%, no structural difference. The arms see the same prompt.

Two harness bugs had to be fixed first, both the same drift: `render-tokens`
hardcoded its `ChatMLConfig` rather than deriving it as the server does. It no
longer compiled against `tool_dialect`, and `has_thinking: true` made every
Coder render carry an empty `<think></think>` cue the server never emits — 4
tokens that presented as a pie-vs-vLLM divergence and were entirely ours. Both
now come from `pie_model::instruct::is_coder_lineage`.

**2. opencode session bleed: refuted**, from the previous runs' own state store.
The `project` row *is* shared across every same-repo workspace in both arms, but
sessions are per-invocation and directory-scoped, and no foreign path appears in
any user or system part. Every cross-workspace reference is the model's own
tool-call argument, token-corrupted mid-string (`Lis另一边-ai`,
`claudeETwitter-501`, `/private entrevista-501/`). The "corrupted UUID" from the
original report is `swe-vllm-10318` — a garbled `swe-vllm-103910`, not another
directory.

What it turned into is a measured asymmetry: a corrupted path lands outside the
workspace, opencode's permission gate fires, and unattended it auto-rejects.

| arm | tool calls | errored | permission-rejected |
|---|---:|---:|---:|
| pie (known5) | 32 | 1 | **0** |
| vLLM (103910) | 45 | 15 | **10** |

All 10 were out-of-workspace; zero in-workspace calls were ever gated, so the
gate is downstream of the corruption rather than a confound of its own. Same
tool-call defect as the schema failures above, one layer further out. n = 1 run
per arm — re-count it in the fair re-run.

**3. Grader images: all 11 pulled** for the full `KNOWN_SOLVABLE_BOTH` set
(`docker pull --platform linux/amd64`, names read from the dataset's `image`
column), so the widened run will not stall on an arm64 manifest mid-grade.

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

### Booting the vLLM arm

Use `tools/boot_vllm.sh <tag>`, the sibling of `boot_pie.sh`. It wraps exactly
this:

```sh
~/.venv-vllm-metal/bin/vllm serve \
    mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
    --port 8000 --served-model-name qwen3-coder-30b coder30b \
    --max-model-len 32768 --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

and then refuses to exit 0 until the process **listening on :8000 is the one it
started** (by pid and process group) and `/v1/models` names
`qwen3-coder-30b`. It also kills the previous server's whole *session*, not
just its launcher: vLLM's V1 engine runs a separate `VLLM::EngineCore` holding
the ~17 GB of weights, and an orphaned one keeps both the memory and the port.
That is why the server is started under `os.setsid()` (macOS ships no
`setsid(1)`). It refuses to boot on top of a live pie server — the box does not
hold two 30B servers, and the failure mode is swapping, not an error.

Both arms restarting per instance is the whole point; a restart you cannot
prove happened is worse than none, which is the lesson pie already paid for.

Three details in the command, each of which produces a confusing failure rather
than a clear one if you get it wrong:

- **`--served-model-name qwen3-coder-30b`** — opencode sends the model id from
  `opencode.json`, which is `qwen3-coder-30b`. Serving only `coder30b` (as the
  A/B reproduce block in `results-pie-vs-vllm-metal.md` does, for the direct
  bench scripts) makes every agent request fail as opencode's generic
  `{"name":"UnknownError","message":"Unexpected server error"}`, with nothing
  naming the model. The flag takes a list, so serving both names satisfies
  opencode *and* `bench_ab.py` / `check_render_vllm.py`, which pass
  `--model coder30b`. Gate on `curl -s :8000/v1/models` before driving
  anything.
- **`--max-model-len 32768`** — the agent arm's number, matched to pie's
  `PIE_MAX_MODEL_LEN=32768`; 16384 is the *A/B bench's* matched number and is
  not enough for an agent (opencode declares a 32768 context for this model in
  `opencode.json`, and an over-long prompt is refused, not chunked). If vLLM
  cannot fit 32768 on this box, drop **both** arms to 16384 — a mismatch here
  is a confound, not a detail.
- **`--enable-auto-tool-choice --tool-call-parser qwen3_coder`** — without
  these, every request carrying `tools` 400s, which is every opencode request.

`--enable-prefix-caching` is on because pie reuses KV too; note that
vLLM-metal reports `usage.prompt_tokens_details.cached_tokens: 0` regardless,
so the server log is the only place its hit rate is visible.

Measured on this box, 2026-08-13: **cold boot 19 s, restart over a live server
22 s** (both to a served, identity-proven `/v1/models`). Per-instance restart
is therefore ~2% of a 1500 s instance budget, so symmetry costs nothing.

### An asymmetry the boot log exposes, still open

vLLM reports `GPU KV cache size: 193,440 tokens` at `--max-model-len 32768`.
pie's pool at the same context is **32,768 tokens** — exactly one max-length
sequence. That is a **~5.9× KV asymmetry** between the arms.

**Where it comes from** (not `simple_family.cpp:313`, which is what an earlier
version of this file and the handover's §5.2 both say — that line is
`Gemma4Engine`'s geometry, the llama engine serving Qwen3-Coder has its own
copy at :1371, and neither is the policy). The policy is
`context.cpp:245 effective_total_pages()`:

```cpp
ctx_pages = ceil(effective_max_ctx_tokens(cfg) / kv_page_size);
return rs_cache_required ? ctx_pages : std::min(cfg.batching.total_pages, ctx_pages);
```

and it is **deliberate**: the pool is what the runtime's physical page ids
index, so a pool larger than the ring is a runtime addressing pages that were
never allocated. The engine lines then derive the same number from `max_ctx`,
so geometry and advertised caps agree by construction. Nothing is being
overwritten by accident.

What is true is narrower and still worth knowing: **`total_pages` can only ever
lower the pool, never raise it past the context ring**, silently. The codebase
says so itself, at `BatchingConfig::max_model_len` — "`total_pages` looks like
the one to reach for and is not … setting it changed nothing and said nothing."

The structural part is that **`max_model_len` sizes both the per-sequence
ceiling and the fleet-wide ring**, and the pool is clamped to the ring — so
`pool == exactly one max-length sequence`, always. No configuration expresses
"let one sequence reach 32k, and give me six sequences of pool." To get 65,536
tokens of pool you must set `max_model_len = 65536` and pay the per-sequence
KV region for it.

### The SWE-bench config quartered the pool — and it does not matter

Same arithmetic, applied to this repo's own history:

| commit | date | `max_model_len` | `total_pages` cfg | **effective pool** |
|---|---|---:|---:|---:|
| `5c3114995` (matched-config re-run) | 08-12 | *not emitted* → 0 → ring 131,072 | 2048 | **2048 pages** (65,536 tok) |
| `a878d5e16` (SWE-bench headroom) | 08-13 | 16384 | 2048 | **512 pages** (16,384 tok) |
| …with `PIE_MAX_MODEL_LEN=32768` | 08-13 | 32768 | 2048 | **1024 pages** (32,768 tok) |

The 08-12 config emitted no `max_model_len`, so the ring defaulted to
`kPhase1bRsSlots(64) × kMetalCtxTokensPerRequest(2048)` = 131,072 tokens and
nothing clamped the configured 2048 pages. Adding the context headroom the next
day put the pool back to **512 pages**, and to 1024 at the 32768 the agent runs
use — a 4× swing that nothing in any output mentions.

**Measured before assuming it cost anything, and it did not.** Chunk held at
2048, only `max_model_len` moved, Coder-30B, `--repeat 3`, pool read from
`PIE_KV_TRACE` rather than derived:

| `max_model_len` | pool | marginal prefill | naive @5055 tok |
|---:|---:|---:|---:|
| 65536 | 2048 pages | 597.2 tok/s | 737.4 |
| 32768 | 1024 pages | 593.9 tok/s | 738.5 |
| 16384 | 512 pages | **598.0 tok/s** | **739.6** |

**0.7% spread across a 4× pool, non-monotonic, with the "starved" arm nominally
fastest.** So there is **no prefill reason to raise `max_model_len` for the
fair re-run**; keep it at 32768, which is set for agent context, not throughput.

This also retires a claim that was propagating: `results-prefill-profile.md`
attributed **1.73× on prefill** to this pool raise. That change moved two knobs
at once — chunk 1024→2048 *and* pool 512→2048 — and only the pair was measured.
The pool half is now measured alone and is worth nothing here. See the
correction in that file; the 1.73× should not be re-quoted until someone
re-derives it one knob at a time.

**Scope of the null result**: concurrency 1, prompts ≤ 5,055 tokens (159 pages,
fits even the 512-page arm). It says nothing about concurrency, or about
strategy B, where retained KV and the live turn share the pool — and where the
`PIE_RETAIN_TOKENS` over-commit below actually bites.

`run_pie_opencode.sh:140` said the pool was 2048; corrected 08-13. The pool
sizes above are confirmed three ways: the code, the commit order, and
`PIE_KV_TRACE=1`'s `avail=` read at each boot.

### A third divisor that turned out not to apply — `executor.max_clients`

Worth recording because it looks alarming and isn't.
`worker/src/executor/mod.rs:1410` computes
`pages_per_client = capabilities.total_pages / max_clients` with
`base_page: slot * pages_per_client`; `max_clients` defaults to **4** and the
generated pie config never sets it. Read naively, a conversation would get 256
pages / 8,192 tokens, and the gap against vLLM would be 23.6× rather than 5.9×.

**It does not bind this path**, on four independent counts:

- The division sits inside `fn hello`, the *remote client* handshake, and
  `worker/src/config.rs:38` scopes the whole struct: "Limits on remote clients
  leasing this worker's KV space." The in-process inferlet path takes no lease.
- `PIE_KV_TRACE=1` reads `install_ws avail=1024/1024` at `max_model_len`
  32768 — the full pool, not a quarter of it.
- Two concurrent **9k-token** requests both completed (§the wear defect). 9k
  tokens is 282 pages, which does not fit one 256-page lease, let alone two.
- The OpenHands session reached the same conclusion from the other side: with
  `max_clients` at its default 4, a single conversation drew on its driver's
  entire 256-page pool. A lease would have given it 64.

So **5.9× stands.** A cheap confirmation remains available if a number is ever
quoted per-conversation: trace on, one prompt over 8,192 tokens, check it
serves and read `avail`.

Related, and also **not** ours: that session's unexplained 256-page dummy pool
resolved to `worker/src/embedded_driver.rs:609-632`, where serving a single
`.zt` makes `read_hf_config_defaults` fail (it wants a model *directory*), so
`max_model_len` silently takes a 4096 fallback and every term of the
`total_pages` `max()` chain lands on 256. That synthesis is in
`dummy_native_options` and is dummy-only: `write_metal_startup_toml:422` writes
the operator's configured `total_pages` straight into the driver TOML, and the
`context.cpp:245` clamp is the only thing that touches it afterwards. We serve
`.zt` artifacts too, so this was worth checking rather than assuming.

### A second consequence, on strategy B only

`PIE_RETAIN_TOKENS` defaults to `PIE_TOTAL_PAGES × kv_page_size / 2` = 32,768
tokens, commented as "about half the pool". It is computed from the
**configured** 2048 pages, not the **effective** clamped pool — so it is 100%
of the pool at `max_model_len` 32768 and 200% of it at 16384. The comment
beside it notes that over-committing "kills the process rather than evicting".

**This is not the §5.1 wear defect.** That was observed on strategy A, which
does not use `retain_tokens` at all. Stated as arithmetic, not as a mechanism.

**External corroboration that this is worth fixing before strategy B is
measured** — from the OpenHands session, on their own retained-KV design and
their own driver, so it is context rather than our data:

- With a pool sized for the workload, prefix reuse held **flat at 83.6% from c4
  to c12** with one rebuild per conversation. On a 10× smaller pool the same
  workload fell to **21.3% at c8 and 16.2% at c12**. They had previously read
  that collapse as "concurrency degrades reuse" and have retracted it — it was
  the pool.
- When *live* demand exceeds the pool the failure is **binary, not gradual**: a
  computed refusal, not a slowdown. No eviction policy funds concurrent
  contexts that cannot fit at once.
- A single long conversation **wedged the pool for everyone else** until
  restart — retained entries with nothing to reclaim into.

Which is the complement to our null above rather than a contradiction of it:
the pool is not a throughput knob, but for a *retained-KV* design it decides how
much reuse survives, and reuse is prefill work not done. At concurrency 1 with a
159-page prompt there is nothing to evict and nothing to lose, so our regime
cannot see it. For strategy B, "one max-length sequence" of pool means retained
working set **and** live turn share it — which is exactly what the
`PIE_RETAIN_TOKENS` arithmetic above over-commits.

### What the asymmetry does and does not mean

Strategy A kills KV with the request, so pie needs the pool only for the
in-flight request, and 32,768 tokens is exactly enough for one 32k request —
with zero headroom. The gap is therefore not "pie is starved per request"; it
is that vLLM carries ~5.9 max-length sequences of **cross-request prefix-cache
headroom** (its own logger measured an 82.2% hit rate) where pie's arm A has
none by design. Quote it that way, or cap vLLM with
`--num-gpu-blocks-override` to match — that is the cheap direction and it does
not touch pie.
