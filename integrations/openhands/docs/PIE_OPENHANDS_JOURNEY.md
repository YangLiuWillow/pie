# Teaching a Coding Agent to Fork: A Field Report on Integrating OpenHands with Pie

*How I spent two months gluing a real software-engineering agent onto a
programmable inference engine — the bugs, the dead ends, the honest
benchmarks, and the one architectural idea that finally made the whole thing
worth it.*

---

## Prologue: two systems that had never met

There are two pieces of software at the center of this story.

**Pie** is a programmable LLM inference engine. Where a normal serving stack
(vLLM, SGLang, TGI) hands you a `/v1/chat/completions` endpoint and hides the
KV cache behind it, Pie flips the model inside-out: you write a small program
— an *inferlet*, compiled to WebAssembly — that runs *inside* the engine and
drives generation directly. From inside an inferlet you can fill a context,
sample tokens, **fork** a KV cache, **mask** arbitrary positions out of
attention, snapshot and restore state, and yield the GPU while you wait on
I/O. It is, in effect, a KV-cache operating system with a WASI syscall
surface.

**OpenHands** is one of the better open-source software-engineering agents —
the kind of system that reads a GitHub issue, explores a repository, edits
files, runs tests, and iterates until the bug is fixed. It ships as a Python
SDK with a rich agent loop: tool calling, a stuck-detector, an
LLM-summarizing condenser, sub-agent delegation, critics.

The thesis that kicked this off was simple to state and, it turned out,
genuinely hard to prove:

> A coding agent is a *long-horizon, KV-cache-heavy* workload. If a
> programmable engine like Pie is ever going to beat a well-tuned
> `vLLM --enable-prefix-caching`, an agent is where it should happen.

This is the story of finding out whether that's true. Spoiler: the first,
obvious way of measuring it says *no*. The interesting part is why, and what
you have to do differently to make the answer *yes*.

---

## The map: where does Pie even plug in?

OpenHands' SDK funnels **every** model call through a single method —
`LLM._transport_call` (in the installed SDK at `llm/llm.py:1128`). Everything
above it — tool dispatch, the stuck detector, condensation, delegation,
critics — is provider-agnostic Python that just wants a `ModelResponse`
back.

That single seam is the whole integration surface. Our `PieLLM`
(`integrations/openhands/pie_openhands/llm.py`) subclasses the SDK's `LLM` and
overrides `_transport_call` to route the request into Pie instead of an HTTP
provider. Because we sit *below* all the SDK machinery, every OpenHands
feature — delegation, critics, parallel execution, pause/resume — is
*reachable* through Pie without touching the SDK. Whether it's *useful* is a
different question we'll come back to.

Two architectures grew out of that seam, and the tension between them is a
running theme:

- **Pattern B — the completion adapter.** Inferlets `openhands-completion`
  and later `openhands-coder-session` behave like a smart
  `/chat/completions`: the Python SDK owns the agent loop, and each turn ships
  the whole conversation down to the inferlet, which renders it, generates one
  reply, and returns. Faithful to stock OpenHands; every SDK feature works
  unmodified.

- **Pattern A — the whole agent in WASM.** The inferlet `openhands-agent`
  *is* the agent loop. It keeps one persistent KV context alive across every
  step, constrains output to a JSON schema, and proxies tool execution out to
  a Python `tool_server.py` over HTTP. Maximal Pie leverage; we reimplement the
  loop.

Hold onto that split. It's the fork in the road that everything else hangs
off of.

---

## Act I — The naive integration, and the long tail of tool-calling bugs

The first milestone was embarrassingly humble: get *any* SWE-bench instance to
run end to end. It exposed a cascade of bugs that, in retrospect, form a
perfect tour of everything that can silently go wrong when you wire an agent
to a new backend.

### The bug that looked like the model giving up

Every SWE-bench problem "succeeded" in three iterations with a **0-byte
patch**. No exception, no traceback — it just looked like the model instantly
surrendering.

The culprit: OpenHands' base `LLM` class defaults `native_tool_calling=True`.
That flips a switch (`should_mock_tool_calls()`) that makes the SDK assume the
*provider* parses tool calls and hands them back structured. Our early
`PieLLM` always returned `tool_calls=None`, so the SDK's own text-based
tool-call parser was skipped entirely and **every tool call the model emitted
was silently dropped.** The fix was one field — `native_tool_calling: bool =
Field(default=False)` — and runs went from 3-iteration/0-byte to real
12–85-iteration tool-calling loops.

That fix bought us into the *real* problem, which turned out to be a saga.

### The tool-call format saga

The single hardest thread in this whole project was getting a Qwen-Coder model
to emit tool calls in a form Pie could parse — reliably, under real agent
contexts, without collapsing. It took, honestly, weeks and produced a chain of
distinct root causes, each hiding behind the last:

1. **The engine was lying about its architecture.** Pie's `vllm` driver
   reported the *raw* HuggingFace arch string (`"Qwen2ForCausalLM"`) up to the
   runtime, but `runtime/src/model/instruct.rs::create()` matches on short
   lowercase names (`"qwen2"`). Anything unmatched silently falls back to a
   generic config with `has_tools: false`. Net effect: **for the entire life
   of the integration, tool schemas had never reached the model** through the
   vllm driver — `equip()` returned an empty vector every time. It was
   invisible because the old *non-native* mock path did its tool prompting in
   Python and never touched Pie's `equip`. Fix:
   `_normalize_arch_name()` in `driver/vllm/src/pie_driver_vllm/engine.py`,
   mirroring the Rust heuristic (lowercase, strip `forcausallm`). Prompt token
   count jumped 41→177 as the `<tools>` block finally appeared.

2. **Grammar-constrained decoding "hung."** Pie can *force* well-formed tool
   calls with a grammar (`QwenInstruct::tool_call_grammar` in
   `runtime/src/model/instruct/qwen3.rs`, compiled through
   `runtime/src/inference/structured/`). First attempt: a 90-second hang. It
   was never a grammar bug. The dev/vllm worker resolved vocab size via
   `getattr(model_config, "vocab_size", 128000)` — but vLLM's `ModelConfig`
   exposes neither `num_vocabs` nor `vocab_size`, only a `get_vocab_size()`
   *method*. So it built a sampling mask sized `128000` for a `152064`-vocab
   model, the tensor-size mismatch crashed the worker, and **pie-server didn't
   notice the dead worker and kept resubmitting forward passes forever.** The
   "hang" was a crash-loop. Fix: `_resolve_vocab_size()` in `worker.py`.

3. **Special tokens leaked into the grammar's candidate set.** With decoding
   finally running, generation truncated mid-JSON-string. `<|im_end|>` decodes
   to *printable text*, which is legal inside an open JSON string, and
   `Tokenizer::sorted_vocab` only excluded *empty*-decoding tokens, not
   special ones. So the model could sample a real EOS out of an incomplete
   grammar state. Fix: exclude `special_token_ids` from `sorted_vocab` in
   `runtime/src/model/tokenizer.rs`.

4. **The grammar was simultaneously too strict and too loose.** `json-pair`
   forbade whitespace after `:`, so the model's natural ` "` continuation
   (logit 27) was masked in favor of noise-floor tokens (logit 7); meanwhile
   omitting per-tool schema constraints let grammar-legal garbage
   (`"}}\n`) win coin-flips inside argument strings. This is the two-sided
   nature of structured decoding: constrain the syntax but not the schema and
   the model wanders; constrain the property *order* and the model — which
   emits `command` before `security_risk` — gets rejected and falls back to
   the generic rule. We iterated through: adding whitespace tolerance,
   schema-per-tool grammars, then *removing* the per-tool schema layer again
   and instead injecting native-format few-shot examples
   (`_inject_native_examples` in `llm.py`).

5. **Forcing a tool call from token 0 lobotomized the model.** The killer
   subtlety: a grammar of `root ::= tool-call ("\n" tool-call)*` forces a tool
   call as the *very first token*, which means **zero reasoning text**, ever.
   The model, denied its chain-of-thought, spammed duplicate calls and looped.
   The fix (commit `c3426c76`) was elegant: make the root a *right-linear
   prefix automaton of the literal `"<tool_call>"`* — free text is completely
   unconstrained (so reasoning and content-only turns are legal and stop
   tokens are reachable), and the schema only binds *once the model has
   actually started* a tool call. Structured output that respects the model's
   need to think first.

6. **Qwen3-Coder speaks a different dialect entirely.** After all that: the
   30B and 32B *Qwen3-Coder* models use an **XML** tool-call format
   (`<function=name><parameter=key>value</parameter></function>`), while Pie's
   `Instruct` renders, constrains, and decodes the **ChatML-JSON** format.
   Through the completion inferlets, a Qwen3-Coder model *never* produces a
   parseable call — every reply looks content-only, the harness burns its ten
   fake-user nudges, and the patch is empty. The pragmatic fix was **Option
   B**: a dependency-free Python port of vLLM's lenient `qwen3_coder` parser
   (`pie_openhands/qwen3coder_parser.py`), re-parsing the raw output host-side.
   That single change flipped `django-13028` from 0-byte to *resolved* (a
   correct 1039-byte fix to `query.py::check_filterable`) — **and** cache reuse
   held at ~97%, because parsing happens *after* generation on the newest turn,
   which is always in the freshly-prefilled suffix, never the reused prefix.

There's a recurring shape to all six. Each was an **unexercised code path** —
grammar-constrained decoding, `equip` on the vllm driver, the sampling-mask
tensor — that had simply never been run before an agent leaned on it. And each
was invisible from the outside: a hang, a 0-byte patch, an instant "success."
The tooling that cracked them was mundane but essential: a `--log-completions`
flag dumping every turn's `raw_response`, host-side `eprintln!` in the runtime
(guest-side WASM `println!` does *not* surface during `ctx.idle()` — a gotcha
that cost real hours), and decoding individual BRLE mask bits inline to ask
"is token *X* actually allowed here?"

### Two model-capability truths, separated from infrastructure

Once the plumbing was honest, two failure modes remained that were *not* our
bug:

- **The literal-`\n` quirk.** Qwen2.5-Coder-7B emits real newlines inside
  `create`'s `file_text` but degrades to the two characters `\` `n` inside
  `str_replace`'s `new_str`. We proved it was the model (not a JSON round-trip)
  by decoding `raw_response` bytes, and confirmed it hit the LiteLLM baseline
  *identically* — same stuck-loop, same 0-byte patch, just faster. That's the
  bar: *matches baseline*. We added `editor_repair.py` to un-escape it anyway.

- **The `/workspace` prompt bug.** The user prompt (inherited from the
  dockerized official harness) claimed the repo lived at `/workspace/<repo>`
  while our checkout was a tempdir. The 7B model would `ls /workspace/django`,
  get exit 2, and greedily repeat that same command for the entire run. Pattern
  A was immune because it used `use_cwd=True`. This one bug had been quietly
  suppressing the resolved-rate of *every* stock-agent run.

The lesson I keep relearning: **with a weak model at temperature 0, a single
wrong token early is unrecoverable, and it will look exactly like a
capability limit until you read the raw trace.**

---

## Act II — Moving the whole agent into the engine

Pattern B has a structural problem that no amount of bug-fixing removes: the
Python SDK replays the *entire* conversation from scratch every step. That's
O(n²) prompt tokens over a trajectory, and every turn pays a fresh
render+transport round-trip.

So we built **Pattern A**: `inferlets/openhands-agent/` — the full agent loop
living inside a single WASM inferlet against one persistent KV context.

The design (see the module header in `inferlets/openhands-agent/src/lib.rs`):

- **`constrain_with(JsonSchema)` every step.** The model *must* emit
  `{thought, action, command, path, old_str, new_str, message}`. This sidesteps
  the entire tool-call-format saga — there's no XML-vs-JSON dialect problem
  when *you* define the schema and force it. `old_str`/`new_str` are always in
  the flat top-level `required` set, so the omission bug that plagued the
  multiplexed `file_editor` schema can't happen.
- **Per-step schema switching** — a `FINISH_SCHEMA` on the last step.
- **KV continuity** — the context persists across all steps: O(n), not O(n²).
- **`ctx.idle()`** — during an HTTP tool call the inferlet yields its GPU
  pages so other work can use the device.
- **Context condensation** — a checkpoint after the system prompt, rebuilt
  from when the context nears the model's limit.

Tool execution proxies out over HTTP to `integrations/openhands/tool_server.py`
— a lightweight server wrapping OpenHands' *real* `PersistentBash` and
`FileEditor` (so the sandbox behavior is faithful to stock OpenHands, not a
homegrown approximation). Getting that bridge working surfaced a lovely WASI
gotcha:

> **WASI HTTP always uses `transfer-encoding: chunked` for outgoing request
> bodies**, even when `Content-Length` is known (the `wstd` crate has a TODO
> about it). Python's `BaseHTTPRequestHandler` reads `Content-Length` bytes
> and only that — so every request body arrived *empty*. The fix was a manual
> chunked-decoder (`_read_chunked()`) walking `hex-length\r\n data\r\n … 0\r\n`
> on the server side.

Pattern A verified end-to-end on GPU (read a file, identify a bug, edit,
run verification, finish — four clean steps) and, on the 30B MoE, showed the
first real signal that the persistent-KV approach *did* something: **16/50
(32%) vs the baseline's 13/50 (26%), with a ~2.1× speedup.**

That felt like the win. It wasn't — or rather, it was the wrong win, measured
the wrong way.

---

## The honest benchmark that said "no"

Here's the uncomfortable result, and the most important one in the whole
project.

We aligned the configurations carefully — same vLLM 0.16.0, same bf16, same
FLASH_ATTN kernels, same MoE Triton path, same seed, matched `max_model_len`
and chunk sizes — and asked the real question: **is Pie faster than
`vLLM --enable-prefix-caching`, with prefix caching left ON?** (The user
refused to disable it, correctly: disabling it would be measuring a strawman.)

The answer on a **single SWE-bench trajectory** was a clean, humbling **no**.
Pie ran ~3.7× *slower* on wall-clock (64 vs ~18 tok/s), decode-bound.

The diagnosis is worth internalizing because it's a general law, not a Pie
flaw:

> A single coding-agent trajectory is **linear, append-only, and
> monotonically growing**. That is *precisely* the one reuse shape prefix
> caching handles optimally. Both engines hit ~95%+ prefill savings. Once the
> prefill advantage is a wash, what's left is raw decode throughput — and
> Pie's layered path (Python ↔ WebSocket ↔ host ↔ WASM ↔ vllm-driver, roughly
> 1.6 inference + 1.5 control-API calls per token) is simply longer than
> vLLM's in-process loop.

This matches Pie's own paper: 3–12% *slower* on plain text completion; faster
*only* on agentic/reasoning workloads with a high I/O-to-token ratio **and**
application-level KV optimizations, measured on *equalized* kernels. The task
alone doesn't create the advantage. **The mechanism does.**

So the strategy crystallized:

> Prefix caching can only reuse a **contiguous shared prefix**. To beat it,
> find the reuse that is **not** prefix-shaped — and exploit programmability
> where a fixed serving engine structurally cannot follow.

Three such shapes, each a Pie primitive that vLLM/SGLang don't expose:

- **Fork** a live KV cache into K children at ~0 prefill (branch, don't
  re-prefill).
- **Mask** the stale middle of a context out of attention, in place (drop
  without rebuilding).
- **Assemble** a new context from cached chunks at arbitrary positions
  (modular caching).

The rest of the project is about building agent workloads that *have* those
shapes.

---

## Act III — Exploiting the mechanism

### 3.1 Session forking on delegation

The first non-prefix shape lives in OpenHands' **sub-agent delegation**. When
a parent agent delegates, the SDK builds the child's LLM via
`parent.agent.llm.model_copy(update=...)` (`tools/task/manager.py`). Because
`PieLLM` overrides only `_transport_call`, that copy path is *already* wired
through us — the child can **fork the parent's KV prefix** instead of
cold-starting. LiteLLM-over-HTTP architecturally cannot do this: there is no
handle to the parent's cache to fork from.

The mechanics (memory `[[project_openhands_fork_integration]]`):

- **Inferlet side** (`openhands-coder-session/src/lib.rs`): a fork branch in
  `build_context` — a token-prefix test against the *parent's* snapshot
  (content-hashed), which opens the parent (refcount-shared KV), appends the
  child's suffix, and saves under the child's name leaving the parent
  untouched. A pure optimization: mismatch falls through to a scratch rebuild.
- **Python side** (`pie_openhands/llm.py`): the sharp edge was that Pydantic's
  `model_copy` copies private attributes, so a naive child would inherit the
  parent's `_pie_session_id` and **corrupt the parent's snapshot by extending
  it.** We override `model_copy` to reset the child's session identity and
  stamp the fork source — idempotent across the SDK's double-copy.

A precursor microbench measured **88.3% prefill saved** on a continuation
delegation. The full **critic-continue** slice (job 19039792) passed 4/4: fork
engaged on every instance, prefill collapsing from 30k–45k tokens down to a
constant **128** (the critic-turn suffix) — **99.5–99.7% saved** — with the
parent verified uncorrupted afterward. Wall-clock savings were
instance-dependent (3–35%; decode dominates on long outputs), but the prefill
win is structural and total.

That slice also delivered a load-bearing side-finding that reshaped how we
talk about "equivalence" (below).

### 3.2 Fork for test-time scaling (best-of-K)

The headline application. A coding agent builds a long *generated*
repo-exploration context, then wants to try **K candidate patches**. The naive
way re-prefills the shared context K times. The Pie way **forks the live
session KV → K children at ~0 prefill.**

The critical design constraint — and the reason the honest benchmark earlier
matters — is this: you must branch **mid-trajectory**, from the generated
context, *not* from the initial issue prompt. vLLM's prefix caching plus its
`n` sampling parameter already share a *static* prompt prefix; if you branch
from the issue text, the baseline captures the reuse and the Pie advantage
vanishes. The win only exists where the shared prefix is *itself generated* and
*divergent downstream* — exactly what a static prefix cache can't anticipate.

Pattern A grew a `run_branch` extraction, an `Input` with
`num_branches`/`branch_at_step`/`temperature`, `base_ctx.fork()` K-1 times, and
`futures::join_all` running the branches concurrently. The `tool_server.py`
grew a **workspace registry** — `fork_workspace` does `cp -a --reflink=auto` of
the working tree (including uncommitted edits) so each branch edits an isolated
filesystem — and switched to a threading server for concurrent branch tool
calls.

The validation arc was instructive:

- **Mechanism** (job 19051340): 2 branches on `django-14373`, each ran on its
  own workspace, produced the correct fix. Concurrency confirmed (151s vs ~289s
  serial). *But both diffs were byte-identical* — because it's a trivial
  one-line fix, so any trajectory converges. Inconclusive on divergence.
- **Divergence** (job 19052176): `scikit-learn-12973` — a problem with *two*
  plausible fix sites — 3 branches at temp 0.7 gave **3 distinct diffs** with
  different step counts (30/26/21). Genuine trajectory divergence; best-of-K is
  *meaningful*. Bonus: all three branches edited the *correct* site
  (`_preprocess_data`), where an earlier single-trajectory run had edited the
  wrong one. ~2.65× concurrency speedup.

The remaining, honest piece of work is the **fair vLLM baseline** for this: you
have to reimplement the mid-trajectory branch structure on the LiteLLM side
(re-sending the branch-point context K times, letting vLLM re-prefill or hit
its cache) and measure TTFT/throughput against Pie's fork-at-~0. OpenHands has
no native fork, so this is net-new harness code. That's the load-bearing
comparison the paper needs, and it's designed but not yet built.

### 3.3 Mask-condensation (the experiment running right now)

The second non-prefix shape, and the one with the cleanest structural
argument. Long-horizon agents overflow their context and must **condense**.
The standard move — what a stock agent *and* vLLM+prefix-caching must do — is:
summarize the dropped middle turns, then **rebuild** a fresh context and
**re-prefill** the kept suffix. Two costs: the re-prefill, *and* an extra LLM
call to generate the summary.

Pie can instead **mask** the stale middle turns' KV out of attention in place.
Zero re-prefill, no summary call, and — because masking skips *attended*
positions without *moving* them — no RoPE re-encoding and no position shift.
It's buildable *today*, unlike true modular-caching gather (which needs an
`import_kvpage` primitive the SDK doesn't yet expose).

- **Stage 1 microbench** (`inferlets/mask-condense-bench/`, job 19057750,
  COMPLETED ✅): on the 30B, history=8000, sweeping the kept window:

  | window kept | mask re-prefill | **rebuild re-prefill** | mask decode | rebuild decode |
  |---:|---:|---:|---:|---:|
  | 576  | 0 ms | 64.6 ms  | 2059 ms | 2062 ms |
  | 2112 | 0 ms | 92.3 ms  | 2048 ms | 2054 ms |
  | 4160 | 0 ms | 156.6 ms | 2052 ms | 2055 ms |

  Two facts fall out. **(1)** Masking costs **0** re-prefill; the rebuild path
  (what APC *must* pay to drop a middle) costs 64–157 ms and **grows with the
  kept-window size**. **(2)** `mask_decode ≈ rebuild_decode` — the masked
  middle is genuinely *trimmed from the kernel* (Pie's page-trim), so keeping
  the full KV resident costs nothing at decode time, and it stays
  *recoverable* (you can unmask it later). The honest magnitude: ~3–8% of a
  128-token turn on fast Blackwell hardware — but it **compounds** over many
  condensations in a long run, grows with window size, and is **purely
  structural**: APC cannot avoid it.

- **Stage 2 A/B** (job **19059956**, COMPLETED): this required a real SDK
  change — threading an `attention_mask` through the high-level `generate()`
  path (`sdk/rust/inferlet`: `Context::mask_range`, `brle_causal_minus`, masks
  applied in both prefill flush and decode, inherited across `fork()`). Pattern
  A gained an opt-in `condense_mode` (`"rebuild"` default, or `"mask"`) with a
  `mask_condense()` masking `[turn_starts[0], turn_starts[n-keep])`, keeping the
  system+task prefix and the last K turns. The A/B driver
  (`mask_condense_ab.py`) runs one SWE-bench instance (`sympy-19346`) twice —
  rebuild then mask — at a low `CONTEXT_LIMIT` to *force* condensations.

  **The result was a clean negative, and it's instructive:**

  | mode | finished | steps | wall_s | gen_s | prompt_tok | patch |
  |---|---|---|---|---|---|---|
  | rebuild | no (hit 50 cap) | 50 | 152.2 | 83.9 | 151,061 | 0 B |
  | mask | yes | 34 | 1308.8 | 98.0 | 532,662 | 0 B |

  On its face mask looks ~8× *slower*. But read the columns: **generation time
  was comparable** (84 vs 98 s — the mask mechanism did *not* blow up decode,
  consistent with Stage 1's page-trim). The ~1200 s wall gap is almost entirely
  **tool-execution time**, and **both arms produced 0-byte patches** — the
  instance was solved by neither. The two runs simply took *completely
  different trajectories* (34 steps/finished vs 50 steps/capped), one of which
  happened to run expensive tools many times.

  This is the same **layer-B nondeterminism** biting the experiment design: the
  engine is not reproducible, so running one instance twice compares **two
  different trajectories, not two condensation strategies.** A full agent loop
  can't isolate the mechanism this way — the trajectory variance and tool-time
  dwarf the prefill delta the mask is supposed to save. The honest, clean proof
  of the mask mechanism remains **Stage 1's controlled microbench** (0 vs
  64–157 ms re-prefill, equal decode). To get a *meaningful* integrated number,
  the next iteration needs (a) an instance both arms actually resolve, so
  there's a quality signal, and (b) an apples-to-apples control — e.g. replay a
  *fixed, pre-recorded* trajectory through both condensers, or aggregate over
  many instances so trajectory noise averages out and the per-condensation
  prefill saving becomes visible above it. As designed, this A/B is confounded
  and inconclusive — a real result worth reporting precisely because it shows
  how easily a nondeterministic engine sabotages a two-run comparison.

---

## An unexpected sub-plot: why the trajectories diverge

Along the way a nagging observation demanded explanation: Pie-glued OpenHands
*sometimes* runs longer trajectories than LiteLLM OpenHands. The tempting
culprit is the tool-call parser — but the user's argument ruled it out
cleanly: both sides run the *same* `qwen3_coder` parser (vLLM server-side vs our
Python port), and a parser can only *classify* a raw output, not change the raw
tokens. Same parser + different trajectory ⟹ the divergence is **upstream** of
parsing.

We decomposed the pipeline — `messages+tools →render(C)→ tokens →sample(B)→ raw
→serialize(D)→ next prompt →parse(A)→ tool_call` — and tested each layer:

- **Parser (A): exonerated.** 360/360 raw outputs parsed identically by our
  port and the real vLLM parser (offline).
- **Render (C): exonerated.** The inferlet's own Rust `render_prompt` and
  vLLM's Jinja template differ by at most one token (the generation cue).
- **Engine/sampler (B): the culprit.** The fork slice had already proven it:
  two *identical* cold temperature-0 calls give *different* outputs, 8/8. Pie's
  MoE/batch-numerics path is **not bit-reproducible.**

This has a sharp downstream consequence for the fork work: **exact
`hash(fork_output) == hash(cold_output)` is an unachievable neutrality metric
on this engine.** The correct claim is the *weak* form — a forked branch
diverges no more than the engine's own noise floor (cold-vs-cold). A bit-exact
KV copy still yields identical *logits*; it's the sampler/reduction order that
introduces noise, fork or no fork. Worth knowing before you write "provably
identical" in a paper.

And an empirical correction to our own folklore: the "consistently longer"
premise was *false*. Across 13 head-to-head instances Pie was longer 6, shorter
4, equal 3 (three byte-identical) — a mild ~6% aggregate lean, not a systematic
bias. Symmetric nondeterministic divergence with a slight positive skew.

---

## What I'd tell the next person

**The single seam is a gift.** Everything routes through `_transport_call`. It
means every OpenHands feature is reachable without forking the SDK — but it also
means the *interesting* Pie wins (fork, mask) live one layer *below* where the
SDK thinks about the problem, so you're often implementing a capability the SDK
has no vocabulary for.

**Two patterns, two philosophies.** `openhands-completion` /
`openhands-coder-session` (Pattern B) stay faithful and let the Python loop
drive — best when you want stock-agent behavior with KV reuse bolted on.
`openhands-agent` (Pattern A) moves the loop into WASM — best when you want
maximal KV control (fork, mask, `ctx.idle`) and are willing to reimplement the
loop. Pattern A is also what kills the 3.7× round-trip overhead, because
generation and control stop crossing the process boundary every token.

**Where the real KV-fork opportunities are** (from reading the SDK,
`[[project_sdk_branch_points]]`) — and where they *aren't*:
- `agent/parallel_executor.py` is **not inference at all**: it runs
  *already-generated* actions in parallel threads. Nothing to fork.
- `subagent/` is a **new session** (its own system prompt) — a *batching*
  opportunity (parent+child decodes on one device), not a fork.
- `critic/impl/api/critic.py` is a **different model** — prefix-cache reuse at
  best, only if Pie hosts it.
- The genuine same-model fork lives in the **decode step**: two-phase tool
  calling (built) and N-best next-action sampling (our test-time-scaling work).
  Forking is a *narrower* opportunity in this SDK than it first looks;
  **batching across concurrent sessions is the bigger, cheaper throughput
  lever** — the benchmark driver runs instances strictly serially
  (`swe_bench.py`), firing the batch scheduler at batch≈1, i.e. a few percent
  of the GPU.

**Benchmark the mechanism, not the task.** SWE-bench-single *neutralized* Pie
because a linear trajectory is the perfect case for the incumbent. RepoBench
would do the same to the current glue (it's completion, not agentic — no
session to reuse). The task never creates the advantage. Pick — or build — the
reuse *shape* that prefix caching can't express, and match kernels so you're
measuring architecture, not driver-vs-server.

**Weak models fail like infrastructure bugs.** A huge fraction of this project
was separating "our plumbing is broken" from "the 7B fixated on the wrong file
at token 0." The tools that made that tractable — `--log-completions`,
host-side `eprintln!`, per-token/per-mask BRLE inspection — were worth every
minute spent building them.

---

## The code, annotated

For the next person walking in, here's where the bodies are buried.

**Inferlets** (`inferlets/`):
- `openhands-agent/src/lib.rs` — Pattern A: full agent loop, JSON-schema
  constrained, `ctx.idle()`, fork-based branching, mask-condensation.
- `openhands-coder-session/src/lib.rs` — Pattern B with persistent
  prompt-token KV snapshots, host-echoed len+hash extension check, rebuild
  fallback, and the `session_fork_from` delegation branch.
- `openhands-completion/src/lib.rs` — the original stateless completion
  adapter with history replay.
- `mask-condense-bench/src/lib.rs` — the Stage-1 mask-vs-rebuild microbench
  (its own `Forward` decode loop, no SDK change).

**The Python bridge** (`integrations/openhands/`):
- `pie_openhands/llm.py` — `PieLLM`; the `_transport_call` seam; session/fork
  private attrs and the `model_copy` override; `_sanitize_tool_args`;
  `qwen3coder_parser` re-parse hook.
- `tool_server.py` — the sandbox bridge (real `PersistentBash` + `FileEditor`),
  chunked-encoding handler, workspace registry for fork isolation.
- `benchmarks/swe_bench.py` — the harness: `solve_one_agent`,
  `_run_agent_inferlet`, fake-user-response loop, condenser wiring,
  fork best-of-K dispatch.
- `pie_openhands/qwen3coder_parser.py` — the lenient XML-format tool-call port.

**The runtime** (`runtime/src/`, `sdk/rust/inferlet/`):
- `model/instruct/qwen3.rs` — tool-call rendering, `build_tool_call_grammar`
  (the free-text-prefix automaton), decoder.
- `model/instruct.rs` — the `Instruct` trait and the arch-name dispatch that
  once silently disabled tools.
- `inference/structured/` — grammar compilation, `GrammarMatcher`.
- `model/tokenizer.rs` — the `sorted_vocab` special-token exclusion.
- `sdk/rust/inferlet/src/context.rs` / `generation.rs` — chunked prefill
  (`MAX_FILL_CHUNK`), the `mask_range`/`attention_mask` plumbing, `fork()`.

**Design docs** (`integrations/openhands/docs/`):
`OPENHANDS_CODER_SESSION_DESIGN.md`, `FORK_TEST_TIME_SCALING_DESIGN.md`,
`SDK_INTERNALS.md`, `GPU_BENCHMARK.md`, `RUNBOOK.md`.

---

## Where it stands

The plumbing is solved and honest. A weak open model resolves real SWE-bench
instances end to end, through both patterns, with tool calling that actually
parses. Fork-on-delegation saves ~99% of branch prefill and is GPU-verified.
Fork-based test-time scaling produces genuinely divergent candidates, so
best-of-K means something. Mask-condensation kills the re-prefill *and* the
summary call that prefix caching structurally cannot avoid — Stage 1's
controlled microbench proved the mechanism cleanly, while Stage 2's naive
in-the-loop A/B came back confounded (two nondeterministic trajectories, both
0-byte, tool-time-dominated) and needs a controlled-trajectory or many-instance
redesign to yield a trustworthy integrated number.

The thesis we started with — *a coding agent is where a programmable engine
should beat a fixed one* — turned out to need a correction that is really the
whole point:

> Not the agent. The **agent's non-prefix-shaped reuse.** Build *that*, match
> the kernels, leave the incumbent's best trick switched on — and only then do
> you get to measure architecture instead of luck.

*— written the afternoon job 19059956 came back with a confounded negative,
which is exactly the kind of result that teaches you something.*
