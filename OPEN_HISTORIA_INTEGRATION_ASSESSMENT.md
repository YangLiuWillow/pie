# Pie ↔ Open Historia — Integration Feasibility Assessment

*Date: 2026-07-24 · Repo studied: [github.com/Open-Historia/open-historia](https://github.com/Open-Historia/open-historia)*

---

## TL;DR

- **Integration feasibility: HIGH.** Open Historia is browser-side and deliberately
  provider-agnostic — it already ships an `openai-compatible` provider. The only
  missing piece is that **Pie has no OpenAI HTTP server** (native interface is
  WebSocket + WASM inferlets). Integration reduces to building **one thin
  OpenAI-compatible HTTP adapter** over `pie_client`, reusing the
  `openhands-completion` inferlet's message/tool rendering. No game code changes.
- **Drop-in single-player *performance* win: UNLIKELY.** The default workload is
  single-player, batch ~1, serial, linear — the "linear = prefix-cache-perfect"
  regime where Pie does **not** beat vLLM/Ollama + automatic prefix caching (APC).
- **The compelling version — *capability* win via harness-in-inferlet: STRONG.**
  Put a reasoning inferlet (CoT/ToT/GoT + branch scoring) behind the unchanged
  OpenAI endpoint. The game's contract is untouched; effective model capability
  rises. This is the generalizable form of the project thesis: **the reasoning
  harness lives server-side where the KV is hot.**

---

## 1. What Open Historia is

A turn-based, AI-driven grand-strategy game (open-source Pax Historia alternative).

- **React 19 SPA** (Vite), MapLibre GL + PMTiles map, OpenLayers editor behind a flag.
- **Three interchangeable `/api` backends**, selected at compile time: local Express
  (desktop download), IndexedDB `fetch` interceptor (hosted web build), embedded
  nodejs-mobile (Android). The `src/` client is byte-identical across all three.
- **All AI runs browser-side** under `src/Game/AI/`. Two entry points on a
  provider-dispatch transport:
  - `callAI` — free-form advisor / diplomacy chat.
  - `runJsonTask` — schema-validated structured tasks that mutate world state
    (~13 task types: timeline jumps, catalysts, GM commands, stat sheets, idle
    diplomacy, action suggestions, event consolidation, …).

### The decisive fact

**The browser calls the LLM provider directly.** Provider config lives in
`localStorage`. Built-in providers include `openai-compatible` — explicitly the
catch-all for "Ollama, LM Studio, OpenRouter, vLLM, and other gateways speaking
`/chat/completions`" (default endpoint `http://localhost:11434/v1`) — and
`anthropic-compatible`. So **swapping in a new backend needs zero game code
changes, provided the backend speaks the OpenAI HTTP API.**

---

## 2. The one gap: Pie has no OpenAI HTTP server

Pie's native interface is a **WebSocket server** (`ws://127.0.0.1:8080`) driven by
WASM inferlets, with a Python client (`pie_client`). The OpenHands integration's
`PieLLM` hooks in at the Python `LLM._transport_call` layer — it never exposes an
HTTP endpoint. A browser cannot speak that protocol.

➡️ **Integration = build one OpenAI-compatible HTTP adapter in front of Pie**,
reusing the existing `openhands-completion` inferlet (which already does
OpenAI-shaped message + tool-history replay → generation).

### Adapter contract (from `src/Game/AI/main.jsx:623`, `callOpenAIStyleChatCompletions`)

| Requirement | Difficulty | Notes |
|---|---|---|
| `POST /chat/completions` (buffered JSON) | Low | Core path. |
| Forced tool call `tool_choice:"required"` (**string** form) → return `tool_calls` | Medium | Every structured task forces one tool. But a fallback ladder (`tool → json_schema → json_object → text_json`) plus the tolerant `extractJsonPayload` parser means even a plain-JSON-in-prose model works. Lowers the bar substantially. |
| `GET /models` (discovery) | Trivial / optional | Only used when no model is set; user can just type the model name. |
| SSE `stream: true` | Medium, optional | Open Historia streams **only for local endpoints**, purely to make Cancel physical. Buffered works; you just lose instant cancel. Pie inferlets can stream. |
| CORS | Low | The desktop/self-hosted build has a same-origin `/api/ai/relay` (`server/server.js:565`) that forwards to the endpoint **server-side**, defeating CORS. So a self-hosted OH + local Pie adapter Just Works. Only a *hosted* website hitting a *local* Pie needs permissive CORS headers. |

---

## 3. The advantage question — first read (and why it's incomplete)

The default workload is **single-player, single-browser → batch ~1, serial, linear**:

- Tasks are serialized behind a `beginSimulation()/endSimulation()` busy lock; no
  tree-search, no rollout, no speculative branching.
- The shared prefix is only *partial*: `buildTemplateVariables()` re-renders the
  **mutating** world state (region ownership, appended events) into context every
  turn. The instruction preamble is stable; the world-state body is not. (Advisor /
  diplomacy chat *is* append-only, so those reuse well.)

This is the **"linear = prefix-cache-perfect"** regime already characterized in
prior experiments (see the OpenEvolve and SWE-bench-single results): Pie does **not**
beat vLLM/Ollama + APC on a single GPU here, and a plain Ollama endpoint already
works out-of-the-box. **Conclusion of the first read: integration is clean, but a
naive backend swap is a working demo, not a performance win.**

---

## 4. The reframe — harness-in-inferlet (the real case)

> *The value isn't faster same-workload inference. By integrating Pie we shift the
> responsibility of writing the harness from the user to the model: a fixed set of
> inferlets lets the model do CoT / ToT / GoT / complex branching server-side,
> raising its effective capability.*

**The OpenAI `/chat/completions` shape is a keyhole.** The game hands in
`(system prompt, history, one forced tool + schema)` and expects `(one tool call
back)`. What happens *between* is invisible to the client. Today that's a single
forward pass + one corrective retry. Replace it with a **reasoning inferlet** that
forks the world-state KV, runs CoT/ToT/GoT internally, self-scores the branches,
and returns the single winning tool call. **The game's code, schema, and contract
are untouched.** The branching Pie's fork was built for — which the game never
provided — the inferlet now *manufactures*.

This flips the first read: we're not measuring Pie against Ollama on the same
single-shot workload; we're **changing the workload the endpoint performs while
keeping the workload the client issues identical.**

### Why Open Historia is an unusually good venue

1. **Turn-based → it can afford depth.** Jumps run with *no timeout by default* —
   the player already waits for the model. Spending 10–30 s of hot-KV branching for
   a markedly better world-update is a trade the UX already accepts. A
   latency-critical chat app could not.

2. **It ships a verifier for free.** ToT/GoT only beats single-shot when branches
   can be *ranked*. Open Historia already has a hard discriminator the inferlet can
   optimize against with **no extra model calls**:
   - `validateGeneratedWorldChanges` — resolves a region's plain name → real map id
     (`DEU.2_1`); currently *silently drops* transfers it can't resolve.
   - **Reluctance guard** — fails a payload whose narration uses capture language
     while shipping zero `regionTransfers` (narrative/map disagreement).
   - Schema + date-clamp validation.

   Today these run once and salvage/drop whatever's broken. A ToT inferlet runs them
   as a **fitness function over branches**: generate N candidate jumps, keep the one
   with the fewest dropped transfers and no consistency violation. Measurable win —
   fewer fallbacks-to-canned-events, fewer phantom-dropped region changes — and
   **no judge model needed.**

### The "fixed set of inferlets" is small

The tool schema arrives *in the request body*, so a single **parametric** inferlet
covers all 13 structured tasks:

```
reason-structured(system, history, tool_schema,
                  strategy = cot | tot | got,
                  width, depth,
                  verifier = schema+game-rules | self-critique)
    → forks, branches, scores against the passed schema, returns one valid tool call

reason-chat(system, history, strategy)
    → CoT / self-refine, returns text (advisor / diplomacy)
```

Two inferlets = the "fixed set." The game still sees a vanilla OpenAI endpoint.
This is the same thesis as OpenEvolve (a GoT search already hosted in Pie) and the
OpenHands critic-continue / best-of-k work.

---

## 5. Honest boundary — where to stay skeptical

- **The ToT uplift is engine-agnostic.** You could run the same search against
  vLLM + APC. So "ToT makes the game smarter" is not, by itself, a *Pie* argument.
  The Pie-specific moat is narrower and worth naming exactly:
  1. **Harness-as-a-deployable-artifact behind a standard endpoint.** With vLLM the
     ToT controller must live *somewhere else* — in the browser (N HTTP round-trips,
     client re-orchestration, the game author has to write it) or in a bolted-on
     proxy (at which point you've reinvented an inferlet, minus the hot-KV coupling).
     Pie unifies *reasoning harness + KV + standard endpoint* into one shippable
     thing. This generalizes past this one game.
  2. **Fork-KV efficiency only becomes decisive at depth/width/concurrency/pressure.**
     In the unpressured single-GPU regime, APC already reuses the shared prefix, so
     Pie's re-prefill savings on *shallow* branching are marginal. Shallow ToT ≈ APC;
     deep/wide GoT or multi-player self-hosting is where Pie separates (same
     conditions as the prior memory-pressure result).
- **Verifier quality caps everything.** The game-rule verifier raises *validity*
  (provable). Raising *quality* ("more interesting / plausible history") needs a
  judge pass — noisier, and you must show the judge beats random or you've spent 4×
  tokens for nothing.

---

## 6. Where Pie wins beyond single-player (independent of the reframe)

1. **Multi-player self-hosted server** — several players on one scenario share the
   prompt-pack + scenario prefix → batching + cross-session content-addressed KV
   cache (already built for OpenHands). Fits Open Historia's self-hosting ethos.
2. **A fork-native "explore N futures" *game feature*** — the one place Pie is
   uniquely strong as a user-visible feature: fork the shared world-state KV and
   diverge N ways to preview alternate timelines / catalyst outcomes / Monte-Carlo
   what-ifs. Needs game-side work.
3. **Graceful degradation on cheap hardware** — Pie's CPU-offload/swap survives
   memory overcommit where vLLM OOMs. Great for "self-host on a home GPU serving
   your friends" — **but** only with the native-CUDA driver + `swap_pool_size > 0`
   (swap is off by default / the embedded driver lacks the wiring).
4. **Translator batching** — the live UI translator fires up to 3 concurrent
   batches; the one naturally concurrent workload. Minor.

---

## 7. Proof experiment

A single, tight A/B/C on the same jump + GM tasks:

| Arm | Setup |
|---|---|
| **A** | Current single-shot `runJsonTask`. |
| **B** | `reason-structured` ToT inferlet, width 4, verifier = game-rules. |
| **C** | Same ToT over vLLM + APC. |

**Metrics (hard, no judge):** fallback-to-canned rate, dropped-transfer count,
consistency-guard violations — plus tokens & wall-clock.

- **A vs B** = capability lift from the harness.
- **B vs C** = whether the *fork* pays, or just the *search*.

If A→B shows a lift and B→C shows Pie separating only at depth/concurrency, you have
the exact, defensible claim.

---

## 8. Recommended next steps

1. **Prototype the OpenAI-compatible adapter** over `pie_client` (small; de-risks
   everything else). Point Open Historia's "OpenAI Compatible" provider at it.
2. **Build the parametric `reason-structured` inferlet** (schema-in → branch →
   verify-against-game-rules → best tool call out).
3. **Run the A/B/C** above.

---

## Appendix — key source references (Open Historia)

| Path | Role |
|---|---|
| `src/Game/AI/main.jsx` | Transport; `callAI` dispatch; `callOpenAIStyleChatCompletions` (`:623`); provider callers; streaming reassembly; chat. |
| `src/Game/AI/gameplay.js` | `runJsonTask` (`:382`); every structured task; validation/salvage; apply-to-world. |
| `src/Game/AI/gameplaySchemas.js` | JSON schemas, tool defs, `getGameplayTool`, `validateGameplayPayload`. |
| `src/Game/AI/providerConfig.js` | Provider registry; `openai-compatible` default `http://localhost:11434/v1`; reasoning toggle. |
| `server/server.js:565` | `/api/ai/relay` — server-side forward that defeats CORS for self-hosted builds. |
| `docs/ai-overview.md`, `docs/architecture.md`, `docs/runtime-services.md` | Primary sources for this assessment. |
