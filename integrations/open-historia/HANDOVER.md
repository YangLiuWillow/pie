# Open Historia ↔ Pie — Session Handover

*Last updated: 2026-07-24. Read this first next session.*

## Goal

Integrate [Pie](../../) as an AI backend for
[Open Historia](https://github.com/Open-Historia/open-historia) — an
open-source, AI-driven grand-strategy game (Pax Historia alternative).

## One-paragraph orientation

Open Historia's AI runs **in the browser** and is **provider-agnostic**: it
already ships an "OpenAI Compatible" provider (`fetch(endpoint + "/chat/completions")`).
Pie has **no OpenAI HTTP server** — it speaks WebSocket + WASM inferlets. So the
integration = **one OpenAI-compatible HTTP adapter in front of Pie** (built, see
below). The *interesting* thesis beyond a plain backend swap: put a **reasoning
inferlet** (CoT/ToT/GoT + branch scoring) behind the unchanged endpoint so the
model gets better at the game with zero client changes — scored against Open
Historia's own built-in validation rules, which are a free verifier.

## Where things live

| What | Path |
|---|---|
| Pie repo (working dir) | `/nfs/roberts/project/pi_ql324/ly337/pie` |
| Open Historia full clone (durable, for reference) | `/nfs/roberts/project/pi_ql324/ly337/open-historia` |
| Working branch | `liu/open-historia` (in the Pie repo) |
| Feasibility assessment (full) | `pie/OPEN_HISTORIA_INTEGRATION_ASSESSMENT.md` |
| The adapter (this dir) | `pie/integrations/open-historia/` |
| Memory pointer | `project_openhistoria_integration_assessment.md` (+ MEMORY.md index) |

Open Historia's AI source of truth: its `docs/ai-overview.md`,
`docs/architecture.md`, `docs/runtime-services.md`, and
`src/Game/AI/main.jsx` / `gameplay.js` / `gameplaySchemas.js`.

## Git state (as of this handover)

Branch `liu/open-historia` = `main` (550de0f3) + 3 commits:

| Commit | Pushed? | Contents |
|---|---|---|
| `6222a58b` | ✅ pushed | `OPEN_HISTORIA_INTEGRATION_ASSESSMENT.md` |
| `d9c4ae95` | ✅ pushed | 36 curated infra files from `openhands-integration-updated` (engine/driver fixes, tool-use plumbing, both inferlets, pie_client) — **excludes** the whole `integrations/openhands/` tree + its vendor submodule (`.gitmodules`) |
| `1066b78c` | ❌ **local only** | The adapter scaffold (`integrations/open-historia/`) |

Remote `origin/liu/open-historia` is at `d9c4ae95`. **The scaffold commit and this
handover are not yet pushed.** (User said don't push without asking.)

Note: `openhands-integration-updated` and `oh-int-merged` are checked out in other
git worktrees (shown with `+` in `git branch`) — read their files via
`git show <branch>:<path>`, don't try to `git checkout` them.

## What's DONE

1. **Feasibility assessment** — integration is HIGH feasibility; a naive
   single-player backend swap is *not* a perf win (it's the
   "linear = prefix-cache-perfect" regime); the win is multi-player self-host
   or the harness-in-inferlet reframe. Full detail + A/B/C proof experiment in
   the assessment doc.
2. **Clean branch base** — rebased onto `main`, brought only reusable infra.
3. **Adapter scaffold** — `pie-openhistoria`, an aiohttp OpenAI-compat server:
   - `translate.py` — pure OpenAI ⇄ inferlet mapping. **11 unit tests pass**
     (`python3 -m pytest tests/test_translate.py -q`), no GPU/server needed.
   - `backend.py` — `pie_client` round trip (connect → auth → `launch_process`
     → drain to `Event.Return`), ported from `PieLLM._call_pie`.
   - `server.py` — routes `/v1/chat/completions` (+ bare), `/v1/models`,
     `/health`; wide CORS; buffered + SSE responses.
   - `__main__.py` — CLI/env config. `README.md` — setup + contract + roadmap.

## What's NOT done (pick up here)

1. **Live end-to-end smoke test (highest priority).** Nothing has touched a real
   Pie server + GPU yet. Steps:
   - `pip install -e '.[dev]'` + `pip install -e ../../client/python` + ensure
     `aiohttp` present.
   - Build the `openhands-completion@0.1.0` inferlet (bakery step — see
     `git show openhands-integration-updated:integrations/openhands/docs/RUNBOOK.md`).
   - Start a Pie server with an **instruct** model (Qwen3 / Qwen2.5 — the game is
     NL-heavy: diplomacy prose, event narration, multilingual; not a pure coder).
   - `python -m pie_openhistoria --pie-uri ws://127.0.0.1:8080 --port 8000`.
   - `curl` a forced-tool request (one tool, `tool_choice:"required"`) and a
     plain chat request; confirm the ChatCompletion shape + `tool_calls`.
   - Then point the real game at `http://localhost:8000/v1` (Settings → AI →
     OpenAI Compatible) and run a timeline jump.
   - **Write this as a GPU-gated smoke script** (mirror
     `integrations/openhands/session_smoke.py`).
2. **Real token streaming** — currently buffered-then-replayed, so Cancel isn't
   physical. Needs the inferlet to emit tokens on Stdout + client-disconnect →
   `terminate_process`.
3. **Cross-turn KV pinning** — each request is a fresh process. The
   `openhands-coder-session` inferlet + its prefix-cache protocol is the
   substrate; port/adapt it. This is also the substrate for #4.
4. **The reasoning inferlet (`reason-structured`)** — the actual differentiator.
   Parametric over the incoming tool schema: fork world-state KV → branch
   (CoT/ToT/GoT) → score branches against Open Historia's own validators
   (region-name→id resolution that currently *drops* unresolved transfers; the
   reluctance/consistency guard; schema/date-clamp) → return the single best
   tool call. Then run the A/B/C in the assessment doc.

## Key technical facts (don't re-derive these)

- **Inferlet contract is `messages`/`tools`** (per the doc-comment atop
  `inferlets/openhands-completion/src/lib.rs`), NOT the stale `prompt`-based
  `Pie.toml [parameters]` block. Input: `{messages, tools?, max_tokens?,
  temperature?, top_p?, stop?, model?}`. Output: `{text, tool_calls:[{id,name,
  arguments}], stop_reason, prompt_tokens, tokens_generated}` (`arguments` is a
  JSON-encoded string).
- **pie_client invocation** (from `PieLLM._call_pie`, branch
  `openhands-integration-updated`): `async with PieClient(uri) as c:
  await c.authenticate(username); proc = await c.launch_process(inferlet,
  input=payload)`; loop `proc.recv()` → `Event.Stdout` accumulate /
  `Event.Return` = JSON result / `Event.Error` = raise.
- **What Open Historia asks of the endpoint** (`src/Game/AI/main.jsx`,
  `callOpenAIStyleChatCompletions`):
  - Forced single tool call: `tool_choice: "required"` (STRING form — object
    form breaks llama.cpp servers).
  - Fallback ladder `tool → json_schema → json_object → text_json` +
    tolerant `extractJsonPayload` ⇒ even non-tool-calling models work.
  - `max_tokens` floored at 8192 for structured tasks.
  - `GET /models` only when no model is configured.
  - Streaming ONLY for local endpoints, to make Cancel physical.
  - Desktop build reaches the endpoint through its own `/api/ai/relay`
    (`server.js:565`), which defeats CORS server-side; hosted-web → local-Pie
    needs the CORS headers the adapter sends.
- **13 structured task types** run through `runJsonTask` (`gameplay.js:382`),
  each a forced tool with its own schema; free-form chat (advisor/diplomacy) via
  `callAI` with no tool. Tasks are serialized (a `beginSimulation()` busy lock),
  so the default workload is batch~1/linear.

## Open decisions / watch-outs

- Adapter lives in the **Pie repo** (`integrations/open-historia/`), not the game
  clone — decided with the user (it's Python, uses pie_client + the inferlet).
- Untracked working files still in the tree from prior work:
  `integrations/openhands/{HANDOVER.md, *.sbatch, stress_concurrency.py}` — left
  alone, not part of this branch.
- The included infra depends as a set (inferlets need the tool-use plumbing +
  CUDA/concurrency fixes) — that's why Option A brought all 36 files, not a
  subset. Build should therefore be self-contained on this branch, but it has
  **not been compiled** here (native build is several minutes).
- Model choice matters: instruct/chat model, not a coder — the game is NL-heavy.
