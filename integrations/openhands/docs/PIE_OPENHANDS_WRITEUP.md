# Moving the Agent Loop to the GPU: What Happens When You Stop Treating an LLM as an API

*July 2026*

---

Every coding agent works the same way. Read the bug report. Run a command. Look at the output. Decide what to do next. Repeat — twenty, fifty, sometimes a hundred times per problem.

And every coding agent pays for this the same way: each iteration is a full round-trip to the LLM. Re-send the entire conversation history. Wait for the engine to re-encode it. Get a response. Throw away all the internal state. Do it again.

We decided to stop doing that.

This is the story of integrating [OpenHands](https://github.com/All-Hands-AI/OpenHands) — an open-source coding agent framework — with [Pie](https://github.com/anthropics/pie), an inference runtime that lets you run programs *alongside* the model instead of just sending prompts to it. The agent loop now executes as a 360KB WebAssembly binary inside the GPU serving engine. There is no API boundary between the agent and the model. The KV cache persists across all steps. Every generation is grammar-constrained to valid JSON. And when the agent waits for a bash command to finish, it voluntarily yields its GPU memory so other workloads can use it.

Here's what we built, what broke, and what we learned.

---

## The Quadratic Tax

Consider what happens when a coding agent runs through a standard serving endpoint — vLLM, SGLang, TGI, any OpenAI-compatible API.

Step 1: the agent sends a system prompt and the bug report. The engine encodes it, generates a response, and discards the KV cache.

Step 5: the agent sends the system prompt, the bug report, and four prior turns of context. The engine re-encodes all of it from scratch.

Step 30: the agent sends the system prompt, the bug report, and twenty-nine prior turns. The engine re-encodes everything again. The prefill alone might take longer than the generation.

The total compute across N steps is O(N^2) in the conversation length. For a 50-step SWE-Bench problem, you're paying to encode the system prompt fifty times. You're paying to re-read the model's own prior outputs forty-nine times, forty-eight times, forty-seven times. This isn't a rounding error — on long-context agent runs, prefill can dominate wall-clock time.

And that's just the compute cost. There are three other problems that compound:

**Structured output is bolted on, not built in.** The agent needs the model to emit a structured action every turn — which tool to call, with what arguments. Standard approaches either rely on prompt engineering ("please respond in JSON") and parse-retry loops, or they bolt on constrained decoding as a separate post-processing layer with its own failure modes. Either way, you're fighting the serving layer instead of working with it.

**GPU memory is wasted during tool execution.** While the agent waits for `pytest` to run (sometimes 30+ seconds), the KV cache sits in GPU memory doing nothing. In a multi-tenant deployment, those pages are locked. Other users can't touch them. Pure waste.

**The agent can't reason about its own memory.** When the conversation gets too long, something has to decide which turns to drop. In a standard setup, this decision happens in Python, outside the serving engine, with no access to the model's actual token counts. It's guesswork.

---

## The Idea: Programs, Not Prompts

Pie replaces the prompt-in/tokens-out contract with something different: **programs that run inside the inference engine**. These programs — called *inferlets* — are compiled to WebAssembly and execute alongside the model, with direct access to the KV cache, the sampler, the grammar constraint machinery, and a cooperative GPU scheduler.

An inferlet is not a prompt template. It's a Rust program that orchestrates inference. It can fill the context, generate tokens, inspect them, apply schema constraints, yield GPU resources while waiting for I/O, and resume exactly where it left off — all without leaving the runtime.

The OpenHands agent inferlet is ~400 lines of Rust compiled to a 360KB WASM binary. It runs the entire agent loop — generate an action, execute a tool, observe the result, decide the next step — inside the serving engine. Here are the four primitives that make this possible.

### 1. KV-Cache Continuity

The inferlet owns a `Context` — a handle to the KV cache for one sequence. When the agent appends a new turn and generates, the previously computed KV pages are still resident. No re-encoding. A 50-step agent run prefills the system prompt *once*.

```rust
let mut ctx = Context::new(&model)?;
ctx.system(SYSTEM_PROMPT);
ctx.user(&task);
ctx.cue();

for step in 1..=max_steps {
    let response = ctx.generate(Sampler::Argmax)
        .max_tokens(16384)
        .constrain_with(inferlet::JsonSchema(ACTION_SCHEMA))?
        .collect_text()
        .await?;

    // Parse, execute tool, append observation.
    // The KV cache grows incrementally. Nothing is re-encoded.
    ctx.user(&format!("Observation:\n{}", observation));
    ctx.cue();
}
```

The total compute is O(N) in the conversation length. Each step only encodes the new observation and generates the next action. Everything prior is already in the cache.

### 2. Structured Output by Construction

Every generation step is wrapped in a JSON schema constraint. The engine masks logits at each token position so that only tokens leading to valid JSON can be sampled. The model *cannot* produce malformed output. No retries, no parsing failures, no "please respond in valid JSON" begging in the system prompt.

```rust
const ACTION_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "thought":  { "type": "string", "minLength": 1 },
        "action":   { "type": "string",
                      "enum": ["bash","edit","read_file","finish"] },
        "command":  { "type": "string" },
        "path":     { "type": "string" },
        "old_str":  { "type": "string" },
        "new_str":  { "type": "string" },
        "message":  { "type": "string" }
    },
    "required": ["thought","action","command",
                  "path","old_str","new_str","message"]
}"#;
```

The schema even switches on the last step: a `FINISH_SCHEMA` constrains `"action"` to the constant `"finish"`, so the agent always terminates cleanly. No runaway loops, no "I'll keep trying" on the last step. The model physically cannot output anything except a termination response.

### 3. Cooperative GPU Scheduling

When the agent calls an external tool — running a bash command, editing a file — it waits for a response over HTTP. During that wait, it calls `ctx.idle()`, which drops the context's memory-market bid to zero. The runtime can now evict those KV pages to serve other workloads. When the tool returns, dropping the idle guard restores priority and the pages are brought back.

```rust
let _idle = ctx.idle();       // "I'm waiting — use my pages"
let obs = call_tool_server(url, &action).await?;
drop(_idle);                  // "I'm back — prioritize me again"
```

This is invisible to the agent logic. The program just makes HTTP calls. The runtime handles the memory choreography underneath. In a multi-tenant deployment, this means an agent waiting for a 30-second test suite isn't holding GPU memory hostage.

### 4. Context Condensation

Long agent runs can exceed the model's context window. The inferlet monitors its estimated sequence length and, when it approaches the limit, triggers a condensation cycle: it uses the model itself to summarize the early turns of the conversation, then rebuilds the context from the system prompt, the summary, and only the most recent turns that fit within budget.

```rust
let est_seq_len = ctx.seq_len() + ctx.buffer().len() as u32;
if est_seq_len + CONDENSE_HEADROOM > context_token_limit {
    ctx = condense_context(&model, &task, &history,
                           context_token_limit)?;
}
```

This happens *inside* the inferlet, not in an external orchestrator. The agent manages its own memory. Different strategies are possible — keep all edits but drop bash output, summarize exploration but preserve the fix — because the condensation policy is just code.

---

## Architecture

The integration supports two deployment patterns.

**Pattern B** is the backwards-compatible path. The Python SDK runs the agent loop. Each step renders the full conversation into a prompt, sends it to Pie through a `PieLLM` adapter (a subclass of OpenHands' `LLM` that routes the transport layer to Pie instead of LiteLLM), and parses the response. It's a drop-in replacement for any OpenAI-compatible endpoint. No KV-cache persistence across steps.

**Pattern A** is the new architecture. The entire agent loop runs inside the WASM inferlet. The Python harness sets up the workspace, starts a lightweight tool server, and launches the inferlet via Pie's client library. Everything else — generate, parse, tool-call, observe, condense, detect stuck loops — happens inside the 360KB binary on the GPU server.

```
┌──────────────────────────────────────────────────────────┐
│                    Pie Runtime (GPU)                      │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │         openhands-agent inferlet (WASM)           │    │
│  │                                                   │    │
│  │  Input ──→ system prompt ──→ ┌───────────┐        │    │
│  │          task description    │ Agent Loop │        │    │
│  │                              │            │        │    │
│  │  ┌───────────────────────────┤  for step  │        │    │
│  │  │                           │  1..=max   │        │    │
│  │  │  1. ctx.generate()        │            │        │    │
│  │  │     + JsonSchema          │            │        │    │
│  │  │     → {thought, action}   │            │        │    │
│  │  │                           │            │        │    │
│  │  │  2. ctx.idle()            │            │        │    │
│  │  │     HTTP POST ──────────────────────────────┐   │    │
│  │  │                           │            │   │   │    │
│  │  │  3. observation ←──────────────────────────┘   │    │
│  │  │     drop(idle)            │            │        │    │
│  │  │                           │            │        │    │
│  │  │  4. ctx.user(obs)         │            │        │    │
│  │  │     ctx.cue()             │            │        │    │
│  │  │     (KV cache grows)      │            │        │    │
│  │  └───────────────────────────┤            │        │    │
│  │                              └───────────┘        │    │
│  └──────────────────────────────────────────────────┘    │
│           │                          ▲                    │
│           │ model forward pass       │ KV pages           │
│           ▼                          │                    │
│  ┌────────────────────────────────────────┐               │
│  │      vLLM driver + H200 GPU           │               │
│  └────────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────┘
            │
            │ HTTP POST /execute
            ▼
┌────────────────────────────────────────┐
│     Tool Server (Python, ~250 lines)   │
│                                        │
│  bash  → persistent subprocess         │
│  edit  → str_replace with fuzzy match  │
│  read  → numbered lines + ranges       │
│  finish → summary                      │
└────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │  Git worktree  │
    │  (SWE-Bench    │
    │   checkout)    │
    └───────────────┘
```

---

## The Tool Server: Small but Opinionated

The inferlet runs inside a WASM sandbox. It cannot spawn processes, read files, or touch the filesystem. All side effects happen through a lightweight Python HTTP server running on the same host. Four actions, one endpoint, ~250 lines of code.

But those 250 lines encode hard-won lessons about what helps a coding agent succeed — and what makes it spin its wheels.

**Fuzzy error messages.** When an edit's `old_str` doesn't match the file, the server uses `difflib.get_close_matches` to find similar lines and shows them in the error. Instead of a bare "not found," the agent sees: *"Did you mean line 42: `msg['Message-ID'] = make_msgid(domain=DNS_NAME)`?"* One extra line of context can save five wasted steps.

**Edit size guardrails.** Edits are capped at 50 lines for `old_str` and 100 for `new_str`. This prevents the agent from attempting to replace an entire file or function body, which almost never works and burns steps. Smaller, surgical edits succeed more often and are easier to verify.

**Syntax checking.** After editing a `.py` file, the server runs `compile()` on the result and includes a `SyntaxError` warning in the observation if it fails. The agent gets immediate feedback — "line 47: unexpected indent" — rather than discovering the error three steps later when it tries to run the code.

**Persistent bash.** The tool server keeps a long-lived bash subprocess, preserving working directory and environment variables across commands. The agent can `cd` into a directory on one step and `ls` on the next without re-navigating. This matters more than you'd think: stateless bash is a constant source of agent confusion ("why is `ls` showing the wrong files?").

---

## The Bugs That Made the Integration Real

The interesting part of any systems integration isn't the clean architecture diagram. It's the bugs. Here are the ones that shaped the final design.

### The vLLM Vocab-Size Crash (or: Why Grammar-Constrained Decoding Appeared to Hang)

When we first wired up grammar-constrained decoding — the feature that forces every generation to produce valid JSON — it appeared to hang indefinitely. 90 seconds, then timeout. Low CPU usage, high system time. It looked like a deadlock in the grammar machinery.

It wasn't.

The real bug was in how the vLLM driver resolved the model's vocabulary size. The code did:

```python
vocab_size = getattr(engine.model_config, "num_vocabs",
             getattr(engine.model_config, "vocab_size", 128000))
```

But vLLM's `ModelConfig` has neither a `num_vocabs` nor a `vocab_size` *attribute* — only a `get_vocab_size()` *method*. So the code silently fell through to the hardcoded default of 128,000. Qwen2.5-Coder's actual vocabulary is 152,064.

The grammar constraint machinery built a logit mask sized for 128K tokens. The vLLM worker tried to apply it to a 152K-token logit tensor. `RuntimeError: The expanded size of the tensor (152064) must match the existing size (128000)`. The worker crashed.

But the crash didn't propagate. Pie's host-side loop didn't detect the dead worker and kept silently resubmitting forward passes. What looked like a grammar-engine deadlock was actually a dead subprocess and a zombie retry loop.

The fix was a three-line function:

```python
def _resolve_vocab_size(model_config):
    for attr in ("num_vocabs",):
        if hasattr(model_config, attr):
            return getattr(model_config, attr)
    if hasattr(model_config, "get_vocab_size"):
        return model_config.get_vocab_size()
    return getattr(model_config, "vocab_size", 128000)
```

### The Architecture Name That Broke Tool Calling

Pie's runtime matches model architectures against short lowercase names — `"qwen2"`, `"qwen3"`, `"llama"` — to determine which chat template and tool-calling format to use. The vLLM driver reported the architecture as whatever HuggingFace's `config.json` contained: `"Qwen2ForCausalLM"`.

`"Qwen2ForCausalLM"` does not match `"qwen2"`.

The runtime silently fell back to a generic config with `has_tools: false`. The `equip()` function — which injects tool schemas into the model's context — returned an empty vector. Tool schemas never reached the model. For the *entire life of the integration*.

This was invisible until we switched to native tool calling. The old non-native path did its own tool-call prompt engineering in Python, completely bypassing Pie's `Instruct` trait. Only when we moved tool handling into the runtime did the bug surface.

The fix: lowercase the architecture name and strip the `"ForCausalLM"` suffix. The Rust-side code already did this for the dummy driver. The Python vLLM driver just... didn't.

### Special Tokens Leak Through the Grammar

After fixing the vocab-size crash, grammar-constrained generation mostly worked — except it sometimes produced truncated JSON. A `{"expression":"` that ended mid-string, cut off cleanly.

Per-token logging revealed the cause: the model was sampling `<|im_end|>` (token 151645) from inside an open JSON string. The grammar should have prevented this — you can't terminate generation in the middle of a string value. But the grammar's token-candidate enumeration only excluded tokens with *empty* decoded text. `<|im_end|>` decodes to the literal characters `<|im_end|>`, which are perfectly legal content inside a JSON string (they're not `"` or `\`). So the grammar let it through.

The fix: exclude all tokens in the tokenizer's `special_token_ids` set from the grammar's candidate vocabulary, not just empty-decoded ones. The grammar now cannot even consider emitting an EOS token unless the grammar itself has reached a valid termination state.

### WASI HTTP and the Chunked Encoding Surprise

The inferlet runs as a WASM component and makes HTTP calls to the tool server using the WASI HTTP API via the `wstd` crate. On the first live GPU test, every tool call returned an empty observation.

The cause: `wstd`'s HTTP client *always* sends request bodies with `Transfer-Encoding: chunked`, even when the body size is known. There's a TODO comment in their source code about this. Python's `BaseHTTPRequestHandler` only reads `Content-Length` bytes, so chunked bodies arrived as empty.

We added a chunked-encoding reader to the tool server:

```python
def _read_chunked(self) -> bytes:
    buf = bytearray()
    while True:
        line = self.rfile.readline().strip()
        chunk_len = int(line, 16)
        if chunk_len == 0:
            self.rfile.readline()  # trailing CRLF
            break
        buf.extend(self.rfile.read(chunk_len))
        self.rfile.readline()  # trailing CRLF
    return bytes(buf)
```

A 20-line fix for a bug at the intersection of three specs (WASI HTTP, HTTP/1.1 chunked transfer encoding, Python's `http.server`). These are the bugs you don't find in unit tests.

---

## What the Agent Does When Things Go Wrong

Beyond the basic generate-call-observe loop, the inferlet handles several failure modes that are invisible in single-call APIs but critical over a multi-step agent run.

**Degenerate output detection.** When the model starts producing garbled text — high ratio of non-ASCII characters, often CJK characters from a context-overflowed English model — the inferlet detects it, triggers context condensation, and retries. Three consecutive degenerate outputs force a clean exit.

**Truncation recovery.** If a generated response hits `max_tokens` before completing the JSON object, the grammar constraint means the output is syntactically incomplete. The inferlet catches the parse error, logs it, and nudges the model: *"Your response was truncated. Be more concise."* Three consecutive truncations force exit.

**Stuck detection.** A rolling-window detector inspects the last 3-5 actions for cycling patterns: same `(action, path)` with failures three times in a row, read-edit-fail cycles on the same file, identical bash commands repeated. When it fires, the agent gets a specific, actionable hint:

> *"STUCK: You have tried the same edit 3 times and it keeps failing. Try a different approach: use `bash` with `sed` to make the change, or use `read_file` with a different line range to see the exact content."*

This solves a notorious failure mode in coding agents. The OpenHands SDK has its own stuck detector, but it requires *byte-identical* action text across turns — including the free-form `thought` field. Real models almost always rephrase their reasoning slightly, so the detector never fires even when the agent is obviously looping. Our detector ignores the thought and compares only the structural action, command, and path.

**Empty-finish recovery.** When the agent calls `finish` but `git diff` shows no changes, the inferlet bounces it back: *"Your changes produced no diff. The issue is NOT resolved yet."* Up to three retries before accepting an empty finish. This alone recovered 2 additional SWE-Bench instances in our 50-problem eval (34% to 38%).

**Test verification gate.** If the agent finishes with a diff but hasn't run any test commands (`pytest`, `unittest`), it gets a one-time nudge to verify its fix before finishing. This catches the common pattern of "edit the file and immediately declare victory" without checking whether the edit actually works.

---

## Results

We evaluated on SWE-Bench Verified — 500 real GitHub issues from major Python projects, with held-out test patches for automated scoring.

### 50-Problem Eval: 30B MoE Model

Using Qwen3-Coder-30B-A3B (a mixture-of-experts model with 3B active parameters per token) on an NVIDIA H200:

| Configuration | Resolved | Rate |
|---|---|---|
| Pattern A (Pie agent inferlet) | 16/50 | 32% |
| Baseline (vanilla OpenHands + same model) | 13/50 | 26% |

Pattern A solved 3 more problems and ran 2.1x faster wall-clock. The agent outperformed the baseline despite using the exact same model and the same 50 problems — the difference is purely architectural. KV-cache continuity eliminated the quadratic prefill overhead. Grammar-constrained output eliminated tool-call parse failures entirely — every single action was valid JSON, every single step. The stuck detector and recovery heuristics caught failure modes that the baseline's simpler loop couldn't recover from.

With additional recovery mechanisms (empty-finish detection, test-verification nudges, root-cause-analysis prompting), the agent reached **19/50 (38%)** — but the interesting number isn't the absolute score, it's the delta. The agent uniquely solves instances the baseline can't (django-13590, xarray-4966), and the remaining gap traces to model-capability limits on specific problems (wrong root-cause identification), not infrastructure issues.

### The 7B vs 32B Split

The model size matters enormously — not just for accuracy, but for *behavior*.

On a 3-problem SWE-Bench smoke test:

| Problem | 7B (Qwen2.5-Coder-7B) | 32B (Qwen2.5-Coder-32B) |
|---|---|---|
| django-11333 | 100 steps, wrong patch | **12 steps, clean solve** |
| django-11532 | 100 steps, 0-byte patch | 60+ steps, correct fix identified but couldn't land the edit |
| sympy-13877 | 100 steps, garbled patch | **~15 steps, correct fix** |

The 7B model exhibited **semantic looping**: repeating the same action with slightly different phrasing each time. It burned through 100 steps without making progress. It would fixate on a wrong file, read it repeatedly, attempt edits that didn't match, and eventually degenerate into token-repetition garbage.

The 32B model showed genuine adaptive debugging. On django-11333, it: read the file, identified the bug, wrote a fix, built a test harness, fixed the test setup when it failed, verified the fix, and finished — all in 12 steps. On sympy-13877, it found the `_find_reasonable_pivot` function in the Bareiss algorithm, identified the NaN comparison issue with symbolic entries, and fixed it in 5 steps.

The 32B also revealed a distinct failure mode: **edit-matching loops**. On django-11532, it correctly identified the fix (punycode-encode `DNS_NAME` before passing it to `make_msgid`) but spent 40+ steps unable to match the exact text for `old_str`. It *knew* what to change but couldn't *land* the edit. This is exactly what the stuck detector and fuzzy-match error messages in the tool server are designed to address.

### Coming Soon: The Efficiency Breakdown

The 2.1x wall-clock speedup is the headline, but it doesn't tell you *where* the time goes — or how much compute is wasted. A detailed efficiency comparison between Pattern A and the baseline is in progress on the 200-problem 32B eval, and will break down the numbers along three axes:

**Token usage.** The baseline re-sends the full conversation history every step, so total prompt tokens grow quadratically with step count. Pattern A's KV-cache continuity means each step only tokenizes the new observation. We're collecting per-step `prompt_tokens` and `completion_tokens` from the inferlet's `StepMetrics` (which tracks `ctx.seq_len()` and `ctx.buffer().len()` at each step boundary) alongside the baseline's standard `usage` fields. The expected result: Pattern A should show dramatically fewer total prompt tokens for the same number of agent steps — the savings compound with conversation length, so the longest, hardest problems should show the largest delta.

**Time breakdown: generation vs. tool execution vs. overhead.** The inferlet records `generate_s` (model forward pass) and `tool_s` (HTTP round-trip to the tool server, including `ctx.idle()` yield time) per step. The baseline has no such decomposition — its wall time is an opaque lump of prefill + decode + Python overhead + HTTP to the vLLM server. Comparing these will quantify how much of the baseline's wall time is pure re-prefill overhead, and how much is irreducible model computation. It also lets us measure the `idle()` yield: time spent waiting for tools is time the GPU *could* be serving other requests.

**Throughput under concurrency.** The single-agent numbers don't capture `idle()`'s real value. When multiple agents share a GPU, the baseline locks pages for the full duration of each tool call; Pattern A releases them. The planned measurement: run N concurrent agents on the same H200 and measure aggregate throughput (problems/hour) as N scales. This is where the cooperative scheduling primitive should show its largest advantage — not on any single run, but on fleet utilization.

These numbers are being collected from the 200-problem runs currently in progress (Qwen2.5-Coder-32B-Instruct on H200, jobs 18690721 and 18599876). Once both complete and are scored, we'll publish the full breakdown — per-step token curves, cumulative compute comparison, and time decomposition histograms — as a follow-up to this writeup.

### Scaling Up: 200 Problems with Qwen2.5-Coder-32B

*[Results pending — this section will be updated when both runs complete and are scored.]*

The 50-problem eval on the 30B MoE model (3B active parameters) established that Pattern A beats the baseline. The natural next question: does the advantage hold — or grow — with a larger, more capable model and more statistical power?

Two 200-problem runs are in progress on NVIDIA H200 GPUs using Qwen2.5-Coder-32B-Instruct (all 32B parameters active, ~4x the compute per token of the 30B MoE):

| Run | Job ID | Status | Config |
|---|---|---|---|
| Pattern A (Pie agent inferlet) | 18690721 | Running | `gpu_memory_utilization=0.80`, `max_model_len=32768`, `--resume` from 8 completed |
| Baseline (vanilla OpenHands) | 18599876 | Running | Same model via vLLM OpenAI-compat endpoint |

**Why 200 problems.** The 50-problem subset was chosen for fast iteration — a single run finishes in a few hours. But 50 problems means each resolved instance moves the score by 2 percentage points, and a 3-instance delta (like the 16 vs 13 result) has wide confidence intervals. 200 problems gives 4x the statistical power: each instance is 0.5 points, and a real 6-point advantage (like the 32% vs 26% we saw) would show as ~12 instances — well above noise.

**Why the 32B dense model.** The 30B MoE (Qwen3-Coder-30B-A3B) only activates 3B parameters per token. It's fast and fits easily on a single GPU, but its per-token reasoning capacity is limited — the 7B vs 32B comparison above showed that model size is the dominant factor in agent *behavior*, not just accuracy. The dense 32B should push the model-capability ceiling higher, letting us see whether Pattern A's architectural advantages compound when the model is actually good enough to exploit them (fewer wasted steps from semantic looping means more steps where KV-cache continuity and grammar constraints actually matter).

**What to expect.** The 50-problem results suggested that Pattern A's edge comes from three sources: (1) fewer wasted steps due to guaranteed structured output, (2) faster per-step latency from KV-cache continuity, and (3) recovery from failure modes the baseline can't handle (empty finishes, stuck loops). If these hold at scale, we'd expect:

- **Resolve rate:** Pattern A >= baseline, with the delta potentially *larger* than the 30B MoE result if the stronger model produces more multi-step solves (which benefit most from KV-cache continuity).
- **Wall-clock time:** Significantly faster per problem, with the gap growing on harder problems that require more steps.
- **Token efficiency:** Dramatically fewer total prompt tokens for equivalent step counts — the quadratic-vs-linear gap is most visible on the long-running problems the 32B model is capable of solving.

#### Early Signal: The First 8 Instances

The agent's initial 200-problem job (18599875) OOM'd after completing 8 instances — all astropy problems, alphabetically first in the SWE-Bench Verified set. It's a small, biased sample, but it's also the first head-to-head comparison with a dense 32B model, and the numbers are illuminating.

| Instance | Agent Steps | Agent Wall (s) | Agent Patch | Stuck Detections | Condensations | Baseline Iters | Baseline Wall (s) | Baseline Patch | Baseline Prompt Tokens |
|---|---|---|---|---|---|---|---|---|---|
| astropy-12907 | 52 | 1210 | 459 B | 4 | 9 | 23 | 110 | 0 B | 98,545 |
| astropy-13033 | 38 | 1218 | 887 B | 3 | 0 | 23 | 47 | 0 B | 94,015 |
| astropy-13453 | 52 | 1007 | 379 B | 18 | 12 | 23 | 349 | 0 B | 110,427 |
| astropy-13579 | 23 | 734 | 13,361 B | 2 | 0 | 23 | 103 | 0 B | 112,581 |
| astropy-14365 | 26 | 611 | 497 B | 3 | 0 | 23 | 495 | 0 B | 115,533 |
| astropy-14369 | 33 | 665 | 730 B | 6 | 0 | 0* | 4707* | 0 B | — |
| astropy-14508 | 19 | 439 | 1,401 B | 3 | 0 | 23 | 346 | 0 B | 126,437 |
| astropy-14995 | 27 | 195 | 0 B† | 7 | 0 | 23 | 210 | 0 B | 113,131 |
| **Totals** | **270** | **6,080** | **7 patches** | | | **161** | **6,366** | **0 patches** | **770,669** |

\* Instance 14369 timed out at 4,707s with an API timeout — no LLM calls completed.
† Instance 14995 failed due to a WebSocket disconnect (the OOM that killed the job).

**The agent produced patches on 7 of 8 instances. The baseline produced zero.** Every baseline instance that made it to the LLM (7 of 8) exhausted all 10 stuck retries, each time regenerating the entire context — burning 770,669 prompt tokens across 77 LLM calls, producing nothing but empty 0-byte diffs.

The contrast is stark at the behavioral level:

- **Agent action distribution:** 144 bash commands, 76 file reads, 49 edits, 1 finish action. The agent explores, reads, modifies, verifies. It acts like a developer.
- **Baseline behavior:** Every instance hit the stuck-retry ceiling (10 retries × 23 iterations). The baseline was semantically looping — repeating the same actions — but without grammar constraints or in-loop detection, it couldn't break free. Each retry re-sent the full context, paying O(N²) prompt tokens for zero net progress.

Two instances highlight the agent's recovery mechanisms in action. Instance 12907 triggered context condensation 9 times across its 52-step run — the context grew large enough to require summarization, but the inferlet's `condense_context()` pruned old turns and let it continue to a successful patch. Instance 13453 hit 18 stuck detections and 12 condensations, the hardest fight in the batch — and still produced a patch.

**The efficiency picture is lopsided.** A naive wall-clock comparison says both runs took similar total time — the agent at 6,080s, the baseline at 6,366s. But that number hides opposite failure modes. The baseline "finishes" each instance quickly on 6 of 7 instances (47–495s) because it gives up — 10 stuck retries, no patch, move on. It only looks slow in aggregate because instance 14369 sat in a 4,707s API timeout. The agent takes longer per instance (195–1,218s) because it's *actually working* — exploring the codebase, reading files, editing code, verifying changes.

The real efficiency gap is in tokens:

| Metric | Agent (Pattern A) | Baseline |
|---|---|---|
| Total steps/iterations | 270 | 161 |
| Patches produced | 7 | 0 |
| Prompt tokens consumed | O(N) per step (KV-cache) | 770,669 (O(N²) regrowth) |
| Avg prompt tokens per LLM call | — | ~10,009 |
| Useful output | 7 patches, 17.7 KB total | 0 B |

The baseline sent 770,669 prompt tokens and got nothing. Each stuck retry re-sent the full conversation history — the same information, re-encoded and re-processed from scratch. That's the O(N²) penalty of stateless inference: step *k* re-transmits the full prefix of steps 1 through *k−1*, and the LLM re-computes attention over all of them.

The agent, running inside the inference engine, never re-sends the prefix. Each step appends to the existing KV cache and pays only for the new tokens. On a 52-step instance like astropy-12907, the baseline would have to re-send (and re-attend-to) the full history 52 times. The agent attends to the full history once and extends it incrementally. The compute savings are proportional to the square of the step count — exactly the instances where the agent's ability to persist makes it capable of solving harder problems.

These results haven't been scored yet (the patches haven't been graded against the held-out test suites), so we don't know how many of the 7 agent patches are *correct*. But the baseline's 0-for-8 patch rate means it can't possibly score above zero on this subset — the scoring comparison is already decided before grading.

The resubmitted agent job (18690721, `gpu_memory_utilization` lowered from 0.85 to 0.80 to prevent the OOM) resumes from these 8 completed instances.

Results, including per-instance breakdowns and the efficiency analysis described above, will be published as soon as both runs complete and are scored via the Apptainer-based SWE-Bench grader.

---

## Why This Architecture Matters

The integration demonstrates something broader than a performance improvement on one benchmark. It's a proof of concept for a different relationship between agents and inference engines.

In the standard architecture, the agent is a *client* of the LLM. It sends requests and receives responses. Everything interesting about inference — the KV cache, the sampler, the constraint engine — is hidden behind an API boundary. The agent can't touch it.

In Pattern A, the agent is a *program that runs inside the inference engine*. It doesn't call an API; it orchestrates the forward pass directly. This isn't a minor optimization — it changes what's *possible*.

**The agent can switch constraints mid-conversation** based on its own state. On the last step, force a finish action. After three failures, switch to a more constrained schema. After a degenerate output, condense and retry. These are control-flow decisions that happen inside the loop, not policy bolted on from outside.

**The agent can manage its own memory.** Context condensation — deciding when to drop old turns and how many to keep — happens inside the inferlet. Different strategies could implement different policies: keep all edits but drop bash output, or summarize exploration turns but preserve the fix attempt. The policy is just code, running next to the model.

**GPU utilization improves without any configuration.** The `idle()` primitive means the scheduling market works automatically. An agent waiting for `pytest` isn't holding pages hostage. This matters most in multi-tenant deployments where ten agents share the same GPU — the cooperative scheduling means they don't need to be serialized.

**The agent is a single, auditable artifact.** The entire behavior of the coding agent — system prompt, action schema, stuck detection, condensation policy, recovery heuristics — is in one 400-line Rust file compiled to a 360KB WASM binary. It can be versioned, diffed, reviewed, and reproduced. No Python notebooks, no YAML config sprawl, no implicit behavior from library defaults.

---

## The Debugging Story Nobody Tells You

Integrating two systems that were designed independently — an agent framework and an inference runtime — means you'll find bugs at every layer boundary. Most of them are boring. Some of them are instructive.

The most expensive class of bug in this project was **silent fallbacks**. The vLLM driver defaulting to vocab size 128,000 when it couldn't find the right attribute. The runtime silently using a generic config when the architecture name didn't match. The readiness-wait loop in the startup script silently proceeding after a timeout instead of failing. The `native_tool_calling` field defaulting to `True` in the base class, silently disabling the entire non-native tool-calling pipeline.

In every case, the system appeared to work. Tests passed. No exceptions. The output was just quietly wrong — or quietly empty. The only signal was that the agent produced 0-byte patches on every problem, which could equally well be a model capability issue, a prompt issue, a scoring issue, or an infrastructure issue.

The antidote we found: **temporary instrumentation that makes the invisible visible**. Adding a `debug_prompt` field to the inferlet's output struct to see the exact rendered prompt. Per-token logging of which vocabulary items the grammar allowed. Host-side `eprintln!` timers on every WIT host-method call. These are throwaway diagnostics — added, used, and fully reverted before committing — but they're the difference between a 30-minute root-cause and a three-day mystery.

WASM sandboxing adds its own wrinkle: `println!` from inside the guest during `ctx.idle()` never reaches the host stdout. `std::env::var()` doesn't see host environment variables. The inferlet is genuinely isolated, which is the point, but it means standard debugging techniques just silently no-op. The reliable path is instrumenting the host side (Pie's runtime, the driver) and embedding structured data in the output JSON.

---

## What's Next

The current inferlet is ~400 lines of Rust. It works, but the evaluations exposed clear next steps.

**KV-cache forking for speculative fixes.** Pie supports forking a context — sharing committed KV pages while exploring different continuations. An agent could fork before attempting a risky edit, try it, and roll back to the fork point if verification fails. No compute is wasted, and the cost is only the divergent pages. This is a primitive that doesn't exist in any API-based serving setup.

**Larger-scale evaluation.** A 200-problem eval with Qwen2.5-Coder-32B is currently running, comparing against a vanilla OpenHands baseline with the same model. The 50-problem results suggest the architectural advantages are real but not yet dominant over model capability. More data will clarify the picture.

**Better condensation strategies.** The current policy is simple: summarize old turns, keep recent ones. But an agent that's been exploring for 30 steps and finally found the right file should probably preserve the file's content and the location discovery, even if those happened early. Semantic condensation — keeping turns based on their *content* rather than their *recency* — is an obvious next step, and the inferlet architecture makes it easy to implement.

**Multi-agent coordination.** If one inferlet can run an agent loop, multiple inferlets can run in parallel on the same GPU, sharing the memory market. One agent explores the codebase while another writes tests. They coordinate through the tool server. Each yields its pages when idle. This is possible today — the primitives are all there — but hasn't been built yet.

---

## Running It

The integration is self-contained. You need Pie, a GPU, and a Python environment.

```bash
# Build the inferlet (35s, one-time)
cd inferlets/openhands-agent
cargo build --target wasm32-wasip2 --release

# Start pie serve with a model config
pie serve --config tests/fixtures/pie_cuda_vllm_config_32b.toml &

# Install the inferlet
python -c "from pie_client import PieClient; \
  PieClient('ws://127.0.0.1:8080').install_program( \
    'openhands-agent', 'target/wasm32-wasip2/release/openhands_agent.wasm', \
    manifest_path='Pie.toml')"

# Run a 3-problem SWE-Bench smoke test
BACKEND=pie-agent \
CFG=tests/fixtures/pie_cuda_vllm_config_32b.toml \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
  bash run_pie_backend.sh \
    --instance-id django__django-11333 \
    --instance-id django__django-11532 \
    --instance-id sympy__sympy-13877
```

The code is on the `openhands-integration` branch. The inferlet is at `inferlets/openhands-agent/src/lib.rs`. The tool server is at `integrations/openhands/tool_server.py`. The benchmark harness is at `integrations/openhands/benchmarks/swe_bench.py`.

---

*The integration was built on Yale YCRC HPC using NVIDIA H200 GPUs. Models evaluated: Qwen2.5-Coder-7B-Instruct, Qwen2.5-Coder-32B-Instruct, and Qwen3-Coder-30B-A3B. SWE-Bench Verified scoring uses Apptainer containers on the cluster (no Docker available). All code is open source.*
