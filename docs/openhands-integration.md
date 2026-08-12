# Pie ⇄ OpenHands Integration Specification

**Status:** Draft v1.0
**Owner:** TBD
**Last updated:** 2026-05-14

---

## 0. One-paragraph summary

We will integrate the [Pie](../README.md) inference runtime with the [OpenHands](https://github.com/All-Hands-AI/OpenHands) coding agent so that OpenHands can drive an agent loop against a Pie-backed model instead of against a LiteLLM-fronted endpoint. The integration is delivered in three phases of escalating ambition: **(1)** a drop-in `PieLLM` subclass that proves correctness end-to-end on SWE-Bench Verified, **(2)** a custom Pie inferlet (`coder-session`) plus a `PieConversation` wrapper that keeps the agent's KV cache pinned across turns and surgically handles condenser events, and **(3)** a stretch "branching speculation" inferlet that samples and scores N candidate first-actions in parallel sharing a prompt prefix. The deliverable is a writeup with a chart comparing wall-clock task time and token economics on SWE-Bench Verified between OpenHands+Pie and OpenHands+vLLM-with-prefix-caching, plus open-source inferlet code.

The hidden value of this project is that *any* outcome ≥ Phase 1 is a real result. A clear negative ("vLLM's prefix caching has already eaten most of the available win") is an artifact Lin can show customers; it is not a failure.

---

## 1. Background and key codebase facts

Read this section first — the rest of the document assumes the facts here.

### 1.1 Pie

Pie is a programmable LLM serving runtime. The full architectural map is in [pie/README.md](../README.md) and the WIT interfaces in [`runtime/wit/`](../runtime/wit/). For this project, the load-bearing primitives are:

| Concept | Where it lives | Why we need it |
|---|---|---|
| **Inferlet** | WASM module loaded by `pie serve`, written in Rust / Python / TS via the SDKs in [`sdk/`](../sdk/). | User code runs co-located with the KV cache; this is the seam where the agent's per-turn logic can be made cache-aware. |
| **`Context`** | [`sdk/python/src/inferlet/context.py`](../sdk/python/src/inferlet/context.py) lines 65–229 / `sdk/rust/inferlet/src/context.rs`. | The KV state holder. Owns committed (immutable) and working (mutable) GPU pages. Forkable, snapshottable. |
| **Named snapshots** | `ctx.save(name) / Context.open(model, name) / Context.take / Context.delete`, ibid lines 102–149. | Survive across inferlet invocations. This is the only feature that makes Phase 2 possible. |
| **Client RPC** | [`client/python/src/pie_client/client.py`](../client/python/src/pie_client/client.py) — `PieClient.launch_process` (`launch_process` returns a `Process`; `Process.recv()` yields `(Event, value)` tuples until `Event.Return` or `Event.Error`). | This is how OpenHands' Python process will talk to Pie. |
| **Driver layer** | [`driver/{portable,cuda,vllm,sglang,dummy}/`](../driver/) | Pluggable inference backends. For benchmarking we want CUDA or vLLM driver; for development we can use `dummy`/`dev`. |
| **Server CLI** | `pie serve --config ~/.pie/config.toml --port 8080` (entry: [`server/src/main.rs`](../server/src/main.rs)). | How we start the runtime. |

Pie's tokenization is **server-side** — tokenizers are loaded by the driver and exposed across the WIT boundary. The client never tokenizes.

### 1.2 OpenHands V1

The local checkout at [`/Users/yangliu/Desktop/Lin_startup/OpenHands/`](../../OpenHands/) is the **app server / orchestration monorepo** (version `1.7.0`, see `OpenHands/pyproject.toml` line 143). The agent loop and the `LLM` class live in a separate PyPI package:

```toml
# OpenHands/pyproject.toml
"openhands-sdk==1.21.1"
"openhands-agent-server==1.21.1"
"openhands-tools==1.21.1"
```

The upstream source for `openhands-sdk` is the **All-Hands-AI/agent-sdk** repo on GitHub (the V1 SDK split released in late 2025). To read `LLM` source you must either (a) `pip install openhands-sdk==1.21.1` and inspect `site-packages/openhands/sdk/llm/llm.py`, or (b) clone the agent-sdk repo separately. **This spec assumes (a) for development and a frozen subclass-in-user-code approach for the deliverable — no fork of agent-sdk is required for Phase 1.**

Critical OpenHands facts:

| Fact | Source |
|---|---|
| `LLM` is a **Pydantic model** (subclassable; the app server already subclasses it as `StrictLLM`). | `OpenHands/openhands/app_server/settings/llm_profiles.py:67-76` |
| `LLM` is instantiated by name in `_configure_llm()` and handed to `AgentSettings.create_agent()`. | `OpenHands/openhands/app_server/app_conversation/live_status_app_conversation_service.py:965-992`, `:1380-1397` |
| The Agent step loop, prompt assembly, tool-call parsing, and `Condenser` all live in `openhands.sdk` (i.e. inside the PyPI package, not the local monorepo). | — |
| The condenser is `openhands.sdk.LLMSummarizingCondenser`; defaults `max_size=240`, `keep_first=2`. | `OpenHands/openhands/app_server/app_conversation/app_conversation_service_base.py:459-493` |
| **SWE-Bench harness is NOT in this monorepo.** The README mentions an external `OpenHands/benchmarks` repo. We will need to clone it separately. | `OpenHands/README.md` line ~80 |
| LiteLLM constraint: `>=1.83.14, !=1.64.4, !=1.67.*`. OpenAI SDK pinned `2.24.0`. | `OpenHands/pyproject.toml` lines 162–163 |

**Subclassing-vs-fork decision:** Because `LLM` is a Pydantic model and the app server already demonstrates subclassing, **Phase 1 does not require a fork of openhands-sdk**. We deliver `PieLLM` as a small Python package the user imports. Phase 2 *may* require subclassing or wrapping the Agent class as well; that decision is deferred to Phase 2 design (§4.4).

### 1.3 What's already done for us by vLLM

vLLM ≥ 0.5 has **automatic prefix caching** enabled by default. This means: even the dumb baseline (OpenHands → LiteLLM → vLLM) already gets KV reuse *within* a single conversation, *as long as the prompt prefix is identical across turns*. The win Pie has to deliver in Phase 2 must come from somewhere vLLM cannot help, namely:

1. **KV that survives across what vLLM treats as separate requests** — vLLM's prefix cache is shared across requests but evictable; Pie's named snapshots are pinned by user code.
2. **Surgical edits to cached state when the prompt prefix changes** — vLLM throws the cache away when the prefix changes (condenser firing). A Pie inferlet can rebuild only the changed region.
3. **Cross-trajectory sharing** — vLLM has no concept of "branch this KV state into N candidates and score them."

This framing should be revisited in Week 1 (§7.1).

---

## 2. Repository layout for the deliverable

All new code lives in a new directory we will create:

```
pie/
  inferlets/
    openhands-coder-session/       ← Phase 2: the real inferlet (Rust)
      Pie.toml
      Cargo.toml
      src/
        lib.rs                     ← entry point + RPC loop
        protocol.rs                ← message types
        state.rs                   ← per-turn KV management
    openhands-branch-spec/         ← Phase 3 (stretch)
      Pie.toml
      Cargo.toml
      src/lib.rs
  integrations/                    ← NEW top-level directory
    openhands/
      pyproject.toml
      pie_openhands/
        __init__.py
        llm.py                     ← Phase 1: PieLLM subclass
        conversation.py            ← Phase 2: PieConversation wrapper
        tokenization.py            ← Phase 1: token <-> message translation
        client.py                  ← thin wrapper around pie_client.PieClient
      tests/
        test_pie_llm.py
        test_pie_conversation.py
        test_e2e_swe_bench_smoke.py
      benchmarks/
        run_swe_bench.py
        baseline_vllm.sh
        baseline_pie.sh
        analysis.ipynb
      docs/
        QUICKSTART.md
        BENCHMARK_RESULTS.md       ← Phase 3 deliverable
```

The OpenHands repo is **not modified** in Phase 1. We import from `openhands.sdk` as a third-party library.

---

## 3. Phase 1 — `PieLLM` (Weeks 1–3, "make it work")

**Goal:** OpenHands solves a handful of SWE-Bench problems end-to-end through Pie, with SWE-Bench Verified scores matching the LiteLLM baseline. **No** performance claims yet.

### 3.1 Acceptance criteria

1. `pip install -e pie/integrations/openhands` succeeds in an env with `openhands-sdk==1.21.1` installed.
2. The unit test suite in `pie_openhands/tests/test_pie_llm.py` passes against a `pie serve` instance using the `dummy` driver.
3. The smoke test (`test_e2e_swe_bench_smoke.py`) solves at least 1 SWE-Bench Verified problem end-to-end through `PieLLM`, on the same model, with the same final patch as the LiteLLM path.
4. A 50-problem SWE-Bench Verified run: `PieLLM` resolved-rate is within ±1 problem of the LiteLLM baseline (i.e. they agree on n=50).
5. Baseline vLLM benchmark numbers are captured into a CSV in `benchmarks/` for later comparison.

### 3.2 Architectural sketch

```
┌─────────────────────────────────────────┐
│  OpenHands agent loop (openhands.sdk)   │
│  - builds messages list each step       │
│  - calls llm.completion(messages, ...)  │
└────────────────┬────────────────────────┘
                 │  (in-process, Pydantic field on Agent)
┌────────────────▼────────────────────────┐
│  PieLLM(LLM)                            │
│  - serialize messages -> prompt string  │
│  - PieClient.launch_process(            │
│      "openhands-completion",            │
│      {"prompt": ..., "max_tokens": ...} │
│    )                                    │
│  - await Process.recv() loop            │
│  - return LLMResponse                   │
└────────────────┬────────────────────────┘
                 │  WebSocket RPC
┌────────────────▼────────────────────────┐
│  pie serve  (Pie runtime)               │
│   └─ inferlets/raw-completion or a      │
│      new inferlets/openhands-completion │
│      (Rust, wasm32-wasip2)              │
└─────────────────────────────────────────┘
```

In Phase 1 we do **not** keep a session open across turns. Each `llm.completion()` call launches a fresh inferlet process. This is wasteful but proves correctness.

### 3.3 What the SDK `LLM` actually exposes

Before writing `PieLLM` you must read the installed SDK source to confirm the method signatures. The exploration covered the *consumers* of `LLM` but not its internals (the source ships only via PyPI). The minimum reading list, after `pip install openhands-sdk==1.21.1`:

```
site-packages/openhands/sdk/llm/llm.py          # the LLM class
site-packages/openhands/sdk/llm/message.py      # Message types
site-packages/openhands/sdk/llm/llm_response.py # response shape
site-packages/openhands/sdk/agent/agent.py      # how Agent calls .completion()
site-packages/openhands/sdk/conversation/       # conversation classes (Phase 2)
```

The method we need to override is almost certainly named `completion(...)` (the LiteLLM convention) and returns a structured response that includes `choices[0].message.content` plus optional `tool_calls`. **Confirm this in Week 1** before writing code beyond a stub — the rest of Phase 1 hinges on getting this signature right.

### 3.4 `PieLLM` skeleton

`pie/integrations/openhands/pie_openhands/llm.py`:

```python
"""PieLLM: an openhands.sdk.LLM that talks to a Pie runtime over WebSocket.

This is the Phase 1 ("dumb") integration: every completion() call launches a
fresh inferlet process. We do not pin KV state across turns yet. The point is
correctness — same SWE-Bench score as the LiteLLM path.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from openhands.sdk.llm import LLM            # confirm exact import path in Week 1
from openhands.sdk.llm.message import Message
from openhands.sdk.llm.llm_response import LLMResponse  # name TBD — confirm
from pie_client import Event, PieClient
from pydantic import Field

from .tokenization import messages_to_prompt, parse_tool_calls


class PieLLM(LLM):
    """openhands.sdk.LLM subclass that routes completions to a Pie backend.

    Configuration is via the standard LLM Pydantic fields plus our two extras.
    Set ``model`` to the model name registered in the Pie server's config.toml
    (the same string Pie reports from ``runtime.models()``).
    """

    # ------- Pie-specific config (extra fields on top of LLM) ----------
    pie_uri: str = Field(default="ws://127.0.0.1:8080",
                         description="Pie server WebSocket URI")
    pie_username: str = Field(default="local-dev",
                              description="Pie auth username (see `pie auth`)")
    pie_inferlet: str = Field(default="openhands-completion",
                              description="Inferlet name registered with `pie serve`")
    pie_request_timeout_s: float = Field(default=600.0)

    # The base LLM has model_config extra='ignore'; that is fine for us.

    # ------------------------ Core override ---------------------------
    # NOTE: signature MUST match the parent. Read site-packages source first.
    def completion(
        self,
        messages: list[Message] | list[dict[str, Any]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return asyncio.run(self._async_completion(messages, tools, **kwargs))

    async def _async_completion(self, messages, tools, **kwargs) -> LLMResponse:
        prompt = messages_to_prompt(messages, tools=tools, model=self.model)
        max_tokens = int(kwargs.get("max_tokens") or 2048)
        temperature = float(kwargs.get("temperature") or self.temperature or 0.0)
        top_p = float(kwargs.get("top_p") or 0.95)
        stop = kwargs.get("stop") or []

        async with PieClient(self.pie_uri) as client:
            await client.authenticate(self.pie_username)
            proc = await client.launch_process(
                self.pie_inferlet,
                input={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop,
                    "model": self.model,
                },
            )
            chunks: list[str] = []
            final_payload: dict | None = None
            while True:
                event, value = await asyncio.wait_for(
                    proc.recv(), timeout=self.pie_request_timeout_s
                )
                if event == Event.Stdout:
                    chunks.append(value)
                elif event == Event.Return:
                    final_payload = json.loads(value) if isinstance(value, str) else value
                    break
                elif event == Event.Error:
                    raise RuntimeError(f"Pie inferlet error: {value}")
                # Event.Stderr, Event.Message, Event.File: ignore for now

        text = (final_payload or {}).get("text") or "".join(chunks)
        tool_calls = parse_tool_calls(text, model=self.model)
        return _build_llm_response(text, tool_calls)


def _build_llm_response(text: str, tool_calls: list[dict] | None):
    """Construct an LLMResponse mirroring what LiteLLM returns.

    The exact constructor differs per openhands-sdk version. Pin this once
    you've read the installed source.
    """
    raise NotImplementedError("Wire to openhands.sdk.llm.llm_response in Week 1")
```

### 3.5 Tokenization / message rendering

`pie/integrations/openhands/pie_openhands/tokenization.py`:

```python
"""Render OpenHands Message lists into a single prompt string for the Pie inferlet.

The inferlet on the server side will tokenize using the model's tokenizer.
We render to the model's *native chat template* — Qwen3-Coder and DeepSeek-V4
both use ChatML-style templates with model-specific tool-call grammars.

For Phase 1 we lean on transformers' AutoTokenizer.apply_chat_template if
available; otherwise we provide hand-rolled renderers for the two target
models.
"""

from typing import Any
from openhands.sdk.llm.message import Message  # confirm in Week 1


def messages_to_prompt(messages, *, tools, model: str) -> str:
    ...


def parse_tool_calls(generated_text: str, *, model: str) -> list[dict]:
    """Extract structured tool calls from raw model text.

    For Qwen3-Coder this means parsing ``<tool_call>...</tool_call>`` blocks.
    For DeepSeek-V4, follow its native function-calling grammar.
    """
    ...
```

**Open question (resolve Week 2):** Can we get the inferlet to do native function-call grammar enforcement via Pie's `grammar` primitive (JSON-schema-constrained sampling, see [`sdk/python/src/inferlet/__init__.py`](../sdk/python/src/inferlet/__init__.py) and the `constrained-decoding` inferlet)? If yes, this hardens the integration considerably. If no, fall back to text-mode parsing as above.

### 3.6 The Phase 1 inferlet

A new Rust inferlet at `pie/inferlets/openhands-completion/`. It is mostly a copy of [`inferlets/raw-completion/src/lib.rs`](../inferlets/raw-completion/src/lib.rs) (which we already include as a reference example in §3.7) with three changes:

1. Accept a richer input struct (`prompt`, `max_tokens`, `temperature`, `top_p`, `stop` list, `model` name).
2. Stop-string handling: after each token batch, check the decoded text for any of the `stop` strings and terminate.
3. Return JSON `{"text": ..., "stop_reason": "length"|"stop"|"eos", "tokens_generated": N}`.

This keeps Phase 1 self-contained. It does **not** use `Context.save`/`open` yet — that lands in Phase 2.

### 3.7 Reference: existing raw-completion inferlet

The Pie repo already has [`inferlets/raw-completion/src/lib.rs`](../inferlets/raw-completion/src/lib.rs). Phase 1 inferlet is a small modification of this file. The structure (verified during exploration):

```rust
#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let model = Model::load(runtime::models().first().ok_or("no models")?)?;
    let tokenizer = model.tokenizer();
    let prompt_tokens = tokenizer.encode(&input.prompt);

    let mut ctx = Context::new(&model)?;
    ctx.append(&prompt_tokens);

    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut g = ctx.generate(Sampler::TopP { temperature, p: top_p })
                   .max_tokens(input.max_tokens);
    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        generated.extend_from_slice(&out.tokens);
        // Phase 1 addition: stop-string check
        if hit_stop(&tokenizer.decode(&generated)?, &input.stop) { break; }
    }
    Ok(Output { text: tokenizer.decode(&generated)?, /* ... */ })
}
```

### 3.8 Baseline: OpenHands + vLLM with prefix caching

In parallel with PieLLM, set up the comparison baseline.

```bash
# baseline_vllm.sh
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Coder-32B-Instruct \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --port 8000
```

OpenHands points at this via the standard LiteLLM `openai/Qwen3-Coder-32B-Instruct` model name with `base_url=http://localhost:8000/v1`.

**Capture per-call**:

- Wall-clock start/end timestamps (turn boundaries).
- `prompt_tokens`, `completion_tokens` from the OpenAI-format response.
- vLLM exposes `/metrics` (Prometheus). Scrape: `vllm:gpu_cache_usage_perc`, `vllm:num_requests_running`, `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`. **Prefix-cache hits are exposed as `vllm:prefix_cache_hit_rate`** (confirm — this metric name has shifted across vLLM versions; verify against the version we pin).
- GPU utilization: `nvidia-smi dmon -s u -c 1` sampled at 1Hz alongside the run.

Output: `benchmarks/baseline_vllm_<date>.csv`, one row per (problem, turn).

### 3.9 SWE-Bench Verified subset selection

Use the official 500-problem set (downloadable from `princeton-nlp/SWE-bench_Verified` on HF). For benchmarking we use a deterministic 50-problem subset:

```python
# benchmarks/run_swe_bench.py (excerpt)
import datasets, random
ds = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
rng = random.Random(20251201)
indices = sorted(rng.sample(range(len(ds)), 50))
subset = ds.select(indices)
```

We freeze the seed and indices in a JSON file checked into the repo so every run uses the same 50 problems.

The actual harness — the part that applies the predicted patch to the repo and runs the test suite — is **not in the local OpenHands monorepo**. The README points at an external `OpenHands/benchmarks` repo. **Action item, Week 1:** locate that repo (likely `All-Hands-AI/OpenHands` org on GitHub; or it may have moved into the SWE-Bench upstream's `swebench/harness`). Pin a commit; do not chase main.

### 3.10 Choice of open-weight coding model

| Model | VRAM (bf16) | Approx VRAM (fp8) | Notes |
|---|---|---|---|
| **Qwen3-Coder-32B-Instruct** | ~64 GB | ~36 GB | Strong SWE-Bench numbers, well-supported by vLLM and SGLang. **Recommended default.** |
| **DeepSeek-V4-Coder** (when released) | TBD | TBD | If V4 is out and benched on SWE-Bench Verified by release, prefer it. |
| **Qwen3-Coder-7B-Instruct** | ~16 GB | ~9 GB | Use only if GPU access is limited; SWE-Bench resolved-rate will be much lower and signal-to-noise on a 50-problem subset worsens. |

**Recommendation:** Qwen3-Coder-32B-Instruct, fp8 weights, on a single H100-80GB. If only A100-40GB ×2 is available, run bf16 with TP=2 (slightly slower but valid).

**Action item, Week 1:** Confirm with Lin what GPU is actually available. Without an H100 or 2×A100 the project's compute budget is the real bottleneck.

### 3.11 Week-by-week breakdown (Phase 1)

**Week 1:**
- Day 1–2: Read installed `openhands-sdk` source for `LLM`, `Agent`, `Message`, `LLMResponse`, `Conversation`. Write a 1-page "internals notes" doc; *every signature below* in this spec needs to be updated to match what you find.
- Day 1: Verify GPU access and choose model. Tell Lin if there's a problem.
- Day 3: Spin up `pie serve` with the CUDA or vLLM driver against the chosen model. Run `pie run text-completion --prompt "hello"` end-to-end successfully.
- Day 3: Run vanilla vLLM server (baseline) end-to-end with one OpenHands trajectory on a single SWE-Bench problem; capture timing.
- Day 4–5: Write the `openhands-completion` inferlet (modify `raw-completion`). Test directly with `pie run`.

**Week 2:**
- Day 1–3: Implement `PieLLM` skeleton (§3.4) and `tokenization.py`. Unit-test against the inferlet via the `dummy` driver.
- Day 4: Plumb `PieLLM` into an OpenHands conversation. Solve one SWE-Bench problem end-to-end.
- Day 5: Compare PieLLM trajectory to LiteLLM trajectory on the same problem — should produce identical patches if the sampler is deterministic (temperature=0).

**Week 3:**
- Day 1–2: Spend 50 GPU-hours running both PieLLM and LiteLLM-vLLM on the 50-problem subset. Compare resolved rates.
- Day 3–4: Capture baseline metrics (vLLM `prefix_cache_hit_rate`, GPU util, per-turn wall-clock). Write `BASELINE.md`.
- Day 5: Slack Lin with results. Demo Phase 1.

---

## 4. Phase 2 — `PieConversation` + persistent KV inferlet (Weeks 4–7, "make it fast")

**Goal:** ≥ 20% wall-clock improvement on multi-turn SWE-Bench tasks (≥10 agent steps) at equivalent quality (resolved-rate within ±1 problem of baseline on the 50-problem subset).

This is the core technical contribution. **Do not shortcut the condenser** (§4.4); it is harder than it looks and getting it wrong reintroduces the problem we are trying to solve.

### 4.1 The shape of the win

In Phase 1, every OpenHands `agent.step()` does the following on the server:

1. Receive the full message history as a prompt string (potentially 20k+ tokens after 15 turns).
2. Tokenize (cheap).
3. **Prefill**: forward-pass every token in the prompt to populate KV cache.
4. Decode N output tokens.
5. Return.

vLLM's prefix caching gives us a free pass on step 3 *only when the prompt prefix is byte-identical to a recent request*. In a coding agent trajectory the prefix is *usually* identical from one turn to the next (we just append a new user message containing tool output), so vLLM's automatic prefix caching already wins a lot. **This is the first thing to measure in Phase 1 — `vllm:prefix_cache_hit_rate` will tell us what's left on the table.**

The remaining wins Pie can capture:

- **(a) Persistent KV across vLLM eviction.** vLLM's prefix cache is a finite LRU pool; under concurrent load or after a long pause, cached prefixes get evicted and we pay re-prefill. Pie's named snapshots are pinned by user code and only evicted if the inferlet explicitly releases them or if the runtime's bidding system overrides (see [`sdk/python/src/inferlet/context.py`](../sdk/python/src/inferlet/context.py) lines 45–57). **This is the *most reliable* source of win.**
- **(b) Surgical handling of the condenser.** When the condenser fires (every ~240 messages in `LLMSummarizingCondenser`), the prompt prefix changes mid-trajectory — the first half of the conversation is replaced with a summary message. vLLM throws the cache away from that point onward. Pie can keep the unchanged *suffix* (the summary itself + everything after) cached and only re-prefill the changed *prefix* — or, if the summary is short enough, just rebuild the whole thing more cheaply because the entire model state is local.
- **(c) Same-turn forking** (Phase 3): Sample N first-action candidates sharing the prompt prefix. vLLM cannot do this within a single trajectory.

Phase 2 targets **(a) and (b)**.

### 4.2 Architecture

```
OpenHands AgentSettings
       │
       │  agent.llm = PieLLM(...)
       │  agent.conversation = PieConversation(...)   ← NEW
       │
       └─► PieConversation
              │  - owns a unique session_id (per OpenHands conversation)
              │  - on first call: launch_process("coder-session", ...) and keep proc alive
              │  - on subsequent .step() calls: send a Message over proc.signal()
              │      containing just the new turn's tokens
              │  - receive completion tokens back as Stdout events
              │  - on Condenser fire: send a "REWRITE" command
              │
              ▼  WebSocket (proc.signal / proc.recv)
       inferlets/openhands-coder-session/   ← long-lived inferlet
              │  - loads model once
              │  - maintains a Context with named snapshot "session-<uuid>"
              │  - on each NEW_TURN message:
              │      • ctx.append(new_tokens)
              │      • ctx.generate(...) until stop
              │      • ctx.save("session-<uuid>")
              │  - on each REWRITE message:
              │      • truncate ctx back to the post-system position
              │      • append the new (post-condenser) prompt suffix
              │      • ctx.save("session-<uuid>")
              │  - on TEARDOWN: Context.delete(model, "session-<uuid>")
```

### 4.3 Why one inferlet process per OpenHands conversation

Two design alternatives were considered:

- **Alt A — Named snapshots only, fresh inferlet per turn**: each `agent.step()` launches a new inferlet that does `Context.open(model, "session-<uuid>")`, appends the new turn, generates, saves, exits. Simple but pays inferlet startup cost every turn. *Acceptable as a fallback if the long-lived design is buggy.*
- **Alt B — Long-lived inferlet, owns its own Context** (chosen): the inferlet runs for the duration of the OpenHands conversation, receives messages via `session.receive()`, holds the Context in WASM memory across turns. *Saves startup cost; matches the project's "the KV cache *is* the conversation" thesis.*

We choose **Alt B**, with **Alt A as the test fallback** for cases where the long-lived process is killed by an external event (OOM, deploy).

### 4.4 The condenser, in detail

This is the part of the spec that took the most thought during planning. Make sure you have read this section twice before writing code.

**What the condenser does (verify by reading `openhands.sdk.condenser` source):**
1. Triggered when `len(messages) > max_size` (default 240).
2. Keeps the first `keep_first` messages (default 2 — typically system prompt and the initial user request).
3. Sends messages `[keep_first : -keep_recent]` (where `keep_recent` is some tail buffer) to a separate LLM call (the "condenser LLM", which is a copy of the agent LLM with `usage_id='condenser'`) to be summarized.
4. Replaces those messages with a single summary message.
5. The resulting message list is: `[first_N_messages, summary_msg, recent_messages]`.

**Implication for the KV cache:** the prompt's *prefix* is unchanged (first N messages). The *middle* changes (raw messages → summary). The *suffix* is unchanged (recent messages). vLLM's prefix cache: HIT on the prefix, MISS from the summary forward.

**Pie's surgical handling — Option (a), rebuild-on-fire:**

```
On REWRITE event from PieConversation:
  - figure out the token boundary where keep_first ends
  - call ctx.truncate_working_page_tokens(...) and/or release pages back to that boundary
  - append the new tokens (summary + recent messages)
  - save snapshot
```

This is **safe** but pays a full re-prefill of (summary + recent) once per condenser fire. Since the condenser fires rarely (every ~240 messages, while a SWE-Bench trajectory averages 15–40 messages), in practice it fires 0–1 times per task. So the rebuild cost is amortized.

**Pie's surgical handling — Option (b), region-rewrite via fork:**

A more aggressive design: maintain two snapshots, `session-prefix` (just the kept prefix) and `session-full` (the running conversation). When condenser fires:

```
new_ctx = Context.open(model, "session-prefix")        # immutable fork
new_ctx.append(tokenize(summary + recent))             # only re-prefill the changed region
new_ctx.save("session-full")
```

This is what the project's overview meant by "rewriting a region of cached state." It is **strictly better** than (a) only if maintaining `session-prefix` is cheap, which it is — once.

**Decision:** Implement Option (b) from the start. The code is only marginally more complex and the conceptual story is much stronger for the writeup.

**Trap to avoid:** the kept-prefix boundary is *not* a fixed token count. It depends on the runtime length of the `keep_first` messages, which can vary if the agent context (`system_message_suffix`, secrets, etc., see `OpenHands/openhands/app_server/.../live_status_app_conversation_service.py` line 1383) changes. **At session start, after the first prefill, record the exact token offset of the kept-prefix boundary** and save the snapshot there. Do not derive the boundary later by re-tokenizing — there will be off-by-one bugs.

### 4.5 PieConversation skeleton

`pie/integrations/openhands/pie_openhands/conversation.py`:

```python
"""PieConversation — wraps an openhands.sdk.Conversation to route through a
long-lived Pie inferlet session.

Phase 2 deliverable.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from openhands.sdk.conversation import Conversation        # confirm path in Wk 1
from openhands.sdk.condenser import LLMSummarizingCondenser
from pie_client import Event, PieClient
from pydantic import Field

from .llm import PieLLM


class PieConversation(Conversation):
    """A Conversation that keeps a Pie inferlet session alive across turns.

    Replaces the default behavior of re-sending the full message history each
    step. The Pie session holds the KV state in GPU pages; we only ship deltas.
    """

    pie_session_id: str = Field(default_factory=lambda: f"session-{uuid.uuid4()}")
    _client: PieClient | None = None
    _proc: Any | None = None   # pie_client.Process
    _kept_prefix_boundary: int | None = None
    _last_known_message_count: int = 0

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    async def _ensure_session(self, llm: PieLLM):
        if self._proc is not None:
            return
        self._client = PieClient(llm.pie_uri)
        await self._client.__aenter__()
        await self._client.authenticate(llm.pie_username)
        self._proc = await self._client.launch_process(
            "openhands-coder-session",
            input={
                "model": llm.model,
                "session_id": self.pie_session_id,
            },
        )
        # spawn a background reader for events
        self._reader_task = asyncio.create_task(self._read_events())

    async def close(self):
        if self._proc is not None:
            await self._proc.signal(json.dumps({"type": "TEARDOWN"}))
            await self._proc.terminate()
            await self._client.__aexit__(None, None, None)

    # ----------------------------------------------------------------
    # The hook: intercept Conversation.step (or the equivalent in the SDK)
    # ----------------------------------------------------------------
    async def step(self, *args, **kwargs):
        await self._ensure_session(self.agent.llm)
        # 1. Did the condenser fire since last step?
        if self._condenser_fired_since_last_step():
            new_summary, recent = self._extract_condenser_output()
            await self._send_rewrite(new_summary, recent)
        # 2. Send only the new turn's user/tool messages
        delta = self._extract_new_messages_since_last_step()
        await self._send_new_turn(delta)
        # 3. Receive completion
        return await self._await_completion()

    # ----------------------------------------------------------------
    # Internals (sketch only — fill in once SDK signatures are known)
    # ----------------------------------------------------------------
    def _condenser_fired_since_last_step(self) -> bool: ...
    def _extract_condenser_output(self): ...
    def _extract_new_messages_since_last_step(self): ...
    async def _send_new_turn(self, delta): ...
    async def _send_rewrite(self, summary_text, recent): ...
    async def _await_completion(self): ...
    async def _read_events(self): ...
```

The key design choice is **where in OpenHands' Conversation/Agent lifecycle we hook**. There are three plausible hook points:

1. **Override `Conversation.step()`** — clean but couples us tightly to the SDK's internal step shape.
2. **Override `Agent.step()` directly** — finer-grained but more invasive.
3. **Stay at the LLM layer (override `PieLLM.completion`) but maintain process-wide state keyed by some identifier in `messages`** — least invasive (works without subclassing Conversation) but requires inferring conversation identity from prompt content, which is fragile.

**Choose 1.** Confirm the exact method signature in Week 4 day 1 (reading installed `openhands.sdk.conversation` source).

### 4.6 The `openhands-coder-session` inferlet

This is the centerpiece of Phase 2. Rust, in `pie/inferlets/openhands-coder-session/`.

`pie/inferlets/openhands-coder-session/src/lib.rs`:

```rust
//! Long-lived inferlet that owns a single OpenHands agent conversation's KV state.
//!
//! Receives JSON messages via session.receive() of the following shapes:
//!
//!     {"type": "INIT",  "prompt": "<full initial prompt incl. system + first user>",
//!                       "kept_prefix_token_count": <int, optional, default = whole>}
//!     {"type": "TURN",  "user_tokens_text": "<new user/tool message rendered>",
//!                       "max_tokens": int, "temperature": float, "top_p": float,
//!                       "stop": ["<str>", ...]}
//!     {"type": "REWRITE", "summary_tokens_text": "<summary>",
//!                         "recent_tokens_text": "<all recent messages>"}
//!     {"type": "TEARDOWN"}
//!
//! Each TURN response is streamed back as session.send() chunks, followed by
//! session.send(json!({"type": "DONE", "stop_reason": ..., "tool_calls": ...})).

use inferlet::{Context, Result, model::Model, runtime, sample::Sampler, session};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct InitInput {
    model: String,
    session_id: String,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Message {
    INIT { prompt: String, kept_prefix_token_count: Option<usize> },
    TURN { user_tokens_text: String, max_tokens: usize, temperature: f32, top_p: f32, stop: Vec<String> },
    REWRITE { summary_tokens_text: String, recent_tokens_text: String },
    TEARDOWN,
}

#[derive(Serialize)]
struct DoneMsg { r#type: &'static str, stop_reason: String, tokens_generated: usize }

#[inferlet::main]
async fn main(input: InitInput) -> Result<()> {
    let model = Model::load(&input.model)?;
    let tok = model.tokenizer();

    // We hold *two* contexts: the prefix-only snapshot, and the live working context.
    let snapshot_full = format!("{}-full",   input.session_id);
    let snapshot_prefix = format!("{}-prefix", input.session_id);

    // Try to attach to an existing session (e.g. inferlet restart) ...
    let mut ctx = Context::open(&model, &snapshot_full)
        .unwrap_or_else(|| Context::new(&model).expect("ctx"));

    // Main message loop
    loop {
        let raw = session::receive().await;
        let msg: Message = serde_json::from_str(&raw)?;
        match msg {
            Message::INIT { prompt, kept_prefix_token_count } => {
                let toks = tok.encode(&prompt);
                ctx = Context::new(&model)?;
                ctx.append(&toks);
                ctx.flush().await?;
                let boundary = kept_prefix_token_count.unwrap_or(toks.len());
                save_prefix_snapshot(&model, &ctx, boundary, &snapshot_prefix).await?;
                ctx.save(&snapshot_full)?;
            }
            Message::TURN { user_tokens_text, max_tokens, temperature, top_p, stop } => {
                let new_toks = tok.encode(&user_tokens_text);
                ctx.append(&new_toks);
                let mut generated = Vec::new();
                let mut g = ctx.generate(Sampler::TopP { temperature, p: top_p })
                               .max_tokens(max_tokens);
                let mut stop_reason = "length".to_string();
                while let Some(step) = g.next()? {
                    let out = step.execute().await?;
                    for &t in &out.tokens {
                        generated.push(t);
                        // stream chunk back
                        if let Ok(text) = tok.decode(&[t]) { session::send(&text); }
                    }
                    if generated.len() >= max_tokens { break; }
                    if let Ok(text_so_far) = tok.decode(&generated) {
                        if stop.iter().any(|s| text_so_far.ends_with(s)) {
                            stop_reason = "stop".into(); break;
                        }
                    }
                }
                ctx.save(&snapshot_full)?;
                session::send(serde_json::to_string(&DoneMsg {
                    r#type: "DONE", stop_reason, tokens_generated: generated.len(),
                })?);
            }
            Message::REWRITE { summary_tokens_text, recent_tokens_text } => {
                // Surgical: re-open the prefix-only snapshot, append summary+recent.
                ctx = Context::open(&model, &snapshot_prefix)
                    .ok_or("missing prefix snapshot")?;
                let combined = format!("{}{}", summary_tokens_text, recent_tokens_text);
                let toks = tok.encode(&combined);
                ctx.append(&toks);
                ctx.flush().await?;
                ctx.save(&snapshot_full)?;
            }
            Message::TEARDOWN => {
                Context::delete(&model, &snapshot_full);
                Context::delete(&model, &snapshot_prefix);
                return Ok(());
            }
        }
    }
}

async fn save_prefix_snapshot(
    model: &Model, ctx: &Context, boundary: usize, name: &str,
) -> Result<()> {
    // Fork an empty ctx, append only the first `boundary` tokens of `ctx`, save.
    // Implementation depends on whether the SDK exposes a "fork-truncated" op or
    // we manually re-prefill the boundary tokens into a fresh Context.
    // The latter is fine — it happens once per session.
    todo!()
}
```

Open issues flagged with `todo!()` will be resolved during implementation. Most likely we have to do an explicit re-prefill of the prefix into a fresh Context to make the prefix-only snapshot, because [`context.rs`](../sdk/rust/inferlet/src/context.rs) does not currently expose a "truncate-at-N-tokens-and-save" primitive. **Verify this assumption in Week 4 day 2** before committing to the design.

### 4.7 Tool calls in Phase 2

Two options:

- **Stream the inferlet's textual output, parse tool calls in `PieConversation` as before.** Simpler.
- **Use Pie's `grammar` primitive (`grammar::from_json_schema`) inside the inferlet to enforce the tool-call schema at the logits level.** Fewer parse failures; constrained generation is part of Pie's value proposition.

**Recommendation:** start with option 1 in Phase 1, then in Phase 2 day 2 evaluate switching to option 2. If the model is well-trained on its native tool-call format (Qwen3-Coder is), option 1 may already have a parse-failure rate of <0.5% and option 2 is not worth the complexity. Measure first.

### 4.8 Benchmarking methodology

The benchmark must be **paired and randomized at the problem level**:

```
For each problem in the 50-problem subset:
  - Coin flip (seeded) decides whether to run Pie or vLLM first.
  - Both backends serve the same model snapshot.
  - Same temperature, top_p, max_tokens, system prompt.
  - Same OpenHands version (1.7.0), same SDK version (1.21.1).
  - Capture: per-turn wall-clock, prompt_tokens, completion_tokens.
  - Discard the first-of-pair runs only if there is an obvious cold-cache effect
    (e.g. > 3σ outlier in turn-1 latency).
```

**Primary metric:** median wall-clock time per problem on multi-turn tasks (define: ≥ 10 agent steps). Use the Wilcoxon signed-rank test on the paired differences. Report effect size (relative speedup) with 95% bootstrap CI.

**Secondary metrics:** total wall-clock time per problem (all tasks), GPU-second cost per problem (wall-clock × number of GPUs × utilization), prefill-token cost per problem (sum of prompt_tokens across turns), generated-token cost per problem.

**Quality metric:** resolved-rate on SWE-Bench Verified (50 problems). Must be within ±1 problem of vLLM baseline to count.

### 4.9 Week-by-week (Phase 2)

**Week 4:**
- Day 1: Read installed `openhands.sdk.conversation` and `openhands.sdk.condenser` sources. Update §4.4 and §4.5 with verified signatures.
- Day 2: Verify Context fork/snapshot primitives can do what §4.6 needs. If not, file a Pie issue and design around.
- Day 3–5: Implement the `openhands-coder-session` inferlet, no condenser logic. Unit-test with a dummy driver.

**Week 5:**
- Day 1–3: Implement `PieConversation`. Plumb through one SWE-Bench problem end-to-end. Verify match-with-Phase-1 trajectories at temperature=0.
- Day 4–5: Implement condenser handling (Option (b), §4.4). Synthetic test: a conversation long enough to trigger condenser.

**Week 6:**
- Day 1–3: Hardening. Failure modes: snapshot eviction, inferlet crash, client disconnect/reconnect, condenser firing mid-generation.
- Day 4–5: First benchmark pass on the 50-problem subset. Look at the numbers honestly.

**Week 7:**
- Day 1–3: Iterate on findings from Week 6. If win is <10%, instrument and find out why. Most likely culprits: vLLM is already winning on prefix-cache (then we have a negative result — write it up), inferlet startup is dominating (move to per-turn snapshot model), or sampler mismatch is causing extra retries.
- Day 4–5: Re-run benchmark. Lock in Phase 2 numbers.

### 4.10 What "5–10% improvement" means and how to write it up

If the win is < 15%, **do not publish a Phase 2 success claim**. Write up the negative result honestly:

> We integrated Pie's named-snapshot KV-pinning and condenser-aware region rewriting into OpenHands V1 on SWE-Bench Verified (n=50, paired) against vLLM 0.17 with automatic prefix caching enabled. Measured median wall-clock improvement on multi-turn tasks (≥10 steps): **<X>% (95% CI [a, b]), p=<p-value>**. We attribute the modest gap to vLLM's automatic prefix caching capturing the majority of the within-trajectory KV reuse opportunity. The remaining headroom is concentrated in (1) cross-request snapshot pinning under high concurrency, where vLLM's LRU evicts and Pie does not, and (2) condenser-fire boundaries, where vLLM's cache invalidates and Pie's region-rewrite avoids a full re-prefill. Both effects are real but their aggregate contribution on SWE-Bench Verified single-user trajectories is small.

This is publishable. Don't pad it.

---

## 5. Phase 3 — Branching speculation + writeup (Weeks 8–10)

**Goal A (always do this):** clean technical writeup with chart, plus open-source inferlet code well-documented enough that an All Hands AI engineer can read it in an afternoon.

**Goal B (stretch):** a branching-speculation inferlet that improves SWE-Bench Verified score by 1–2 percentage points at fixed compute, or holds score with 30%+ less compute.

### 5.1 Writeup (Goal A) — non-negotiable

Sections, roughly:

1. **What we did** (one paragraph).
2. **Why this is interesting** (KV state as user-programmable infrastructure; coding agents as a workload where prefix reuse dominates).
3. **Phase 1: the integration.** Architecture diagram. Code link.
4. **Phase 2: the inferlet.** Diagram showing the three context states (prefix, full, post-condenser). Walk through one trajectory.
5. **Benchmark methodology.** §4.8 verbatim.
6. **Results.** One main chart: paired wall-clock per problem, scatter plot, with the y=x line. Inset: distribution of per-problem speedups. Three secondary tables: resolved-rate, GPU-second cost, prefix-cache hit rates.
7. **Where the win comes from / does not come from.** This is the most important section if the number is modest.
8. **What this implies for Pie's positioning.** Hand-off to Lin.
9. **Limitations.** Single-user; one model; benchmark; we didn't measure tail latency under concurrent load (where Pie's pinning should win more).
10. **Code.** Link to GitHub.

Deliverable file: `pie/integrations/openhands/docs/BENCHMARK_RESULTS.md`. Charts via `analysis.ipynb` exported as PNG/SVG.

### 5.2 Branching speculation inferlet (Goal B)

The idea, concretely:

```
At a "planning step" (defined: turn where the agent is choosing
between several plausible high-level actions — heuristic detector
or explicit signal from the agent):

  1. Compute the prompt up through the planning prefix.
  2. ctx.save("plan-prefix").
  3. Fork N times (Context.open or Context.fork).
  4. In parallel, sample one continuation in each fork with high temperature.
  5. Each fork generates until it produces a structured action
     (e.g. a tool call).
  6. Score the N candidates with a cheap in-engine heuristic
     (logprob of the chosen action under the original model,
     or value-head if we have one — for Phase 3 we use logprob).
  7. Commit to the highest-scoring candidate by selecting its fork
     as the working Context; discard the others.
```

`pie/inferlets/openhands-branch-spec/src/lib.rs`. Rust. New inferlet.

This is *speculation* in the broader sense (sampling many futures), not the narrow KV-speculation sense. It is harder to get right than Phase 2 because:

- The "planning step" detector must not fire too often (cost) or too rarely (no benefit).
- N must be tuned (we expect N=4 to N=8).
- Scoring must be cheap and not require a separate model.

**Honest assessment:** this works *if and only if* SWE-Bench trajectories actually have decision points where the first action is high-variance and the right one is much better. If trajectories are well-anchored after the system+user prompt (which Qwen3-Coder largely is), branching helps little. **Treat Phase 3 Goal B as a research bet, not a planned deliverable.** Spend ≤ 1 week. If it doesn't show signal by end of Week 9, kill it cleanly and put the time into writeup polish.

### 5.3 Week-by-week (Phase 3)

**Week 8:**
- Day 1–3: Write the writeup outline and produce first-pass charts from Phase 2 data.
- Day 4–5: If pursuing Goal B: implement the branching inferlet against a synthetic prompt.

**Week 9:**
- Goal B: integrate into PieConversation. Run a partial benchmark (20 problems). Decide go/no-go on Week 10 day 1.

**Week 10:**
- Polish writeup. Lock in code (tag a release in this repo + the inferlets). Hand the artifact to Lin.

---

## 6. Setup / install / runbook

### 6.1 Repos to clone

```bash
mkdir -p ~/Lin_startup && cd ~/Lin_startup
# Already present:
#   pie/
#   OpenHands/
# Also clone:
git clone https://github.com/All-Hands-AI/agent-sdk.git    # for reading + (maybe) forking
# (Confirm exact URL — see §1.2.)
git clone <openhands-benchmarks-repo>   # for SWE-Bench harness; URL TBD
```

### 6.2 Python env

Use `uv` or `pip-tools` to keep `openhands-sdk==1.21.1` and `litellm` in sync with OpenHands' constraints. A minimal `pyproject.toml` for `pie/integrations/openhands`:

```toml
[project]
name = "pie-openhands"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "openhands-sdk==1.21.1",
  "pie-client",                  # install -e ../../client/python
  "pydantic>=2.7",
  "transformers>=4.46",          # for chat-template rendering
  "datasets>=2.20",              # for SWE-Bench loading
  "tenacity>=8.0",               # retry logic
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff"]
bench = ["pandas", "matplotlib", "scipy"]
```

### 6.3 Bringing up Pie for development

```bash
# Build the runtime + drivers once
cd ~/Lin_startup/pie
cargo build --release -p pie-server
./scripts/install_drivers.sh    # if it exists; otherwise see driver/*/README

# Build inferlets
cd inferlets/openhands-completion
cargo build --target wasm32-wasip2 --release

# Configure
cat > ~/.pie/config.toml <<'EOF'
[[model]]
name = "qwen3-coder-32b"
hf_repo = "Qwen/Qwen3-Coder-32B-Instruct"

[model.driver]
type = "cuda"   # or "vllm"
device = ["cuda:0"]
activation_dtype = "bfloat16"
EOF

# Run
pie serve --config ~/.pie/config.toml --port 8080
```

### 6.4 Bringing up the vLLM baseline

```bash
pip install "vllm>=0.17"   # confirm latest stable
bash benchmarks/baseline_vllm.sh   # see §3.8
```

### 6.5 Smoke test

```bash
cd ~/Lin_startup/pie/integrations/openhands
pip install -e .
pytest tests/test_pie_llm.py -k smoke
python benchmarks/run_swe_bench.py --backend pie --n 1
```

---

## 7. Risks and open questions

### 7.1 Top-of-list risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| vLLM's prefix caching already eats most of the available win on SWE-Bench. | **High** | The Phase 2 result is modest (5–10%). | Spend Week 1 day 1 measuring `vllm:prefix_cache_hit_rate` on the baseline. If it's already > 90%, redesign Phase 2 around cross-request pinning and condenser-fire rebuilding (the two things vLLM cannot do) and lower the expected effect size accordingly. Write up a negative result honestly. |
| The condenser is harder than it looks. | **High** | Phase 2 slips 1–2 weeks. | Read the SDK source for `LLMSummarizingCondenser` *before* implementing PieConversation. Build a synthetic test that triggers condenser. Implement Option (b) only after Option (a) works. |
| openhands-sdk API drifts. | Medium | Code rewrites. | Pin `openhands-sdk==1.21.1` and OpenHands `1.7.0`. Do not update during the project. Note version in every benchmark CSV. |
| GPU access. | Medium (depends on what Lin has) | Project does not start. | Confirm in Week 1 day 1. Fallback model: Qwen3-Coder-7B if 32B is infeasible — acknowledge in writeup. |
| The SWE-Bench harness lives in an external repo that has shifted. | Medium | Phase 1 benchmarking slips. | Locate and pin the harness in Week 1. Worst case, vendor a copy. |
| `LLMSummarizingCondenser` invokes its LLM via the *same* LLM object (`condenser` usage_id). So our `PieLLM.completion()` will be called by the condenser too, with a very different prompt shape. | Medium | Condenser calls bypass `PieConversation` entirely. | OK as long as PieLLM's stateless path works (Phase 1 design). The condenser call is a one-shot summarization — Phase 1's launch-fresh-inferlet path is fine for it. |
| Sampler determinism mismatch: Pie's `Sampler::TopP` and vLLM's top-p may differ at the seed/RNG level even at temperature=0. | Low | Trajectories diverge; can't directly compare patches. | Compare resolved-rate at population level, not patches at trajectory level. Use temperature=0 with greedy sampling (no top-p) for the comparability test. |
| Pie's `vllm` driver may have a different prefix-cache implementation than vanilla vLLM, contaminating the comparison. | Medium | Apples vs oranges. | **Do not run the baseline through Pie's vllm driver.** The vLLM baseline must be a vanilla `vllm.entrypoints.openai.api_server` process, separate from Pie. |
| Inferlet startup cost dominates per-turn cost on short conversations. | Low (Phase 2 design avoids this) | Per-turn-launch design (Alt A in §4.3) has bad latency. | We chose the long-lived inferlet design (Alt B). Document the startup cost in the writeup. |
| Pie's snapshot eviction (the bidding system) silently drops the session under memory pressure. | Medium | Mysterious quality regressions. | Detect missing snapshots and fall back to full re-prefill from the message history; emit a warning. |

### 7.2 Open questions that must be answered in Week 1

1. **What is the exact `LLM.completion(...)` signature in openhands-sdk 1.21.1?** Without this, every code block in this spec is wrong.
2. **Is `Conversation.step()` (or its equivalent) overridable in a subclass, or does the agent framework own the loop in a way that makes subclassing infeasible?** If infeasible, switch to wrapping at the Agent level.
3. **Does Pie's Rust `Context` SDK currently expose a primitive to "fork the first N tokens of a Context into a new Context"?** If not, the prefix snapshot in §4.6 has to be built by re-prefilling — file a feature request with Lin's team.
4. **Where is the SWE-Bench harness today (in code)?** Lock down the repo + commit before Week 1 ends.
5. **What GPU does the team actually have?** This determines the model.

### 7.3 Decisions deferred

- **Grammar-constrained tool calls** (§4.7): defer to Week 5 day 2.
- **Goal B (branching speculation)** (§5.2): defer go/no-go to Week 8 day 5.

---

## 8. Definition of done

A reviewer should be able to read this checklist top-to-bottom and tick each box.

- [ ] `pie/integrations/openhands/` exists with `pip install -e .` working.
- [ ] `pie/inferlets/openhands-completion/` builds to `wasm32-wasip2`.
- [ ] `pie/inferlets/openhands-coder-session/` builds to `wasm32-wasip2`.
- [ ] CI runs the integration's unit tests against the `dummy` driver.
- [ ] A SWE-Bench Verified 50-problem benchmark CSV is checked in for both Pie and vLLM with matching timestamps.
- [ ] `BENCHMARK_RESULTS.md` exists with the chart, the methodology, and an honest interpretation.
- [ ] Code is documented well enough that an engineer at All Hands AI can read it in an afternoon.
- [ ] If Goal B was pursued, `pie/inferlets/openhands-branch-spec/` is shipped or explicitly marked killed-in-Week-9.

---

## 9. Pointers (for the implementing engineer)

Files to read first, in order:

1. [`pie/README.md`](../README.md)
2. [`pie/sdk/python/src/inferlet/context.py`](../sdk/python/src/inferlet/context.py) — to internalize the Context API.
3. [`pie/inferlets/raw-completion/src/lib.rs`](../inferlets/raw-completion/src/lib.rs) — to internalize the inferlet pattern.
4. [`pie/inferlets/demo-persistent-kv/src/lib.rs`](../inferlets/demo-persistent-kv/src/lib.rs) — to see snapshot reuse working.
5. [`pie/client/python/src/pie_client/client.py`](../client/python/src/pie_client/client.py) (lines 470–540) — to see `launch_process`/`Process.recv`.
6. `site-packages/openhands/sdk/llm/llm.py` — the LLM class to subclass.
7. `site-packages/openhands/sdk/agent/agent.py` — how the agent calls LLM.
8. `site-packages/openhands/sdk/conversation/*.py` — what to subclass in Phase 2.
9. `site-packages/openhands/sdk/condenser/llm_summarizing_condenser.py` — the condenser behavior.
10. `OpenHands/openhands/app_server/app_conversation/live_status_app_conversation_service.py` (lines 965–1397) — how OpenHands constructs an agent from an LLM. Useful for understanding what we're replacing.

Files to write, in order:

1. `pie/integrations/openhands/pyproject.toml`
2. `pie/integrations/openhands/pie_openhands/__init__.py`
3. `pie/inferlets/openhands-completion/{Pie.toml,Cargo.toml,src/lib.rs}`
4. `pie/integrations/openhands/pie_openhands/tokenization.py`
5. `pie/integrations/openhands/pie_openhands/llm.py`
6. `pie/integrations/openhands/tests/test_pie_llm.py`
7. `pie/integrations/openhands/benchmarks/run_swe_bench.py`
8. `pie/integrations/openhands/benchmarks/baseline_vllm.sh`
9. `pie/inferlets/openhands-coder-session/{Pie.toml,Cargo.toml,src/lib.rs}` (Phase 2)
10. `pie/integrations/openhands/pie_openhands/conversation.py` (Phase 2)
11. `pie/integrations/openhands/docs/BENCHMARK_RESULTS.md` (Phase 3)
12. `pie/inferlets/openhands-branch-spec/...` (Phase 3 stretch)

---

*End of specification. This document is intended to be edited as Week 1 findings come in — particularly §3.4, §4.5, and §4.6, all of which depend on SDK source that has not been read at the time of writing.*
