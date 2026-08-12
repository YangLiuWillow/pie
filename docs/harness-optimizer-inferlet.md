# Harness Optimizer × Inferlets: server-side prompt evolution for sealed harnesses

**Status:** idea note v1 (2026-08-06) — experiment arm for `pie-rl-verl-integration.md`, not a scope change. Sibling notes: `realtime-rl-single-rollout.md` (composes with this one), `parallel-reasoning-agent-rl.md`.
**Source:** Strands Agents post "Introducing Harness Optimizer" — https://strandsagents.com/blog/introducing-harness-optimizer/. Package `strands-harness-optimizer` (PyPI v0.0.1, Apache 2.0); repo verified 2026-08-06: https://github.com/strands-labs/harness-optimizer (labs org, not `strands-agents`; homepage + README confirm it's the Strands project and the pip name). Same AWS ecosystem as the AgentCore/Strands migrationbench cookbook already in our rllm checkout.

---

## 1. The idea in one paragraph

Harness Optimizer treats agent scaffolding as trainable parameters without touching weights: a `Formula` exposes `process()/get_tunable_params()/update_params()`, a `Trainer.fit()` loop runs rollouts, scores them with a `RewardFunction`, and the default `ContrastiveReflectionOptimizer` has an LLM compare win-vs-loss traces and append distilled rules to the system prompt (a `MultiAgentOptimizer` variant shards traces across sub-agents to avoid fixating on one failure mode). Reported gains: AppWorld 72.6%→95.8%, WebShop 56.5%→67.5%, tau-bench improvements across the board. Currently system-prompt-only; skills/memory/tool-description formulas are roadmap. It is GEPA/DSPy-shaped prompt evolution, **not RL** — no gradients, no weight sync, inference-only.

## 2. The honest scoping: what needs Pie and what doesn't

The optimizer core (reflection LLM, Trainer loop) is plain Python and pairs with *any* inference stack — running it against our rllm+qwen-code+swesmith setup needs no Pie at all, just a thin adapter (rllm episodes in, rules text out; we already have traces + rewards from Phase 0A). The **inferlet pairing** is specifically about *serving* the evolving harness, and there it buys three real things:

### 2a. Formula injection into a sealed harness (the headline pairing)

Their integration story is `apply_formulas_on_strands_agent(agent, [formula])` — in-process Python attachment. Our harness is qwen-code: a sealed TypeScript CLI we deliberately pinned and audited for prompt determinism; there is no in-process attachment point, and forking deeper than the audit's minimal patches is off the table. But **every prompt already flows through the gateway → Pie inferlet**, so the inferlet can *be* the Formula's `process()`: deterministically inject the current learned-rules block into the prompt head server-side, per request. That makes any OpenAI-compatible agent optimizable without touching its binary — a more general artifact claim than the Strands adapter, and inferlet-native (the injection is program logic inside inference, not a proxy hack).

### 2b. Cache-stable prompt evolution (the thesis-relevant pairing)

The `tool_desc_invariance.py` lesson: one changed byte in the prompt head = zero snapshot reuse downstream of it. Harness Optimizer *mutates the system prompt every optimizer step* — naively that invalidates the fleet's KV each iteration, which for a continual/production optimizer means paying full prefill on every episode after every update. Mitigations, weakest to strongest:
- **Append-only rules placement** (works on vLLM APC too): if the rules block sits at the very end of the prompt head, everything before it stays cache-valid. But for Qwen3-Coder-class templates the tool descriptions serialize *inside* the system region — a rules block that must precede tool descs (or multiple independently-evolving formula sites, per their roadmap) breaks the trick.
- **Inferlet snapshot modules**: keep [base system prompt] and [tool descs] as stable snapshots, maintain the rules block as its own versioned segment, and let the inferlet compose them with explicit KV layout (the fork/pagestore/position-override toolkit from the Phase 6 note, used for a much simpler linear case). Evolving one module re-prefills only that module. This is a differentiated Pie capability and a measurable claim: **prefill cost per optimizer iteration, Pie modules vs APC re-prefill**.

### 2c. `prompt_version` rides the `weight_version` machinery

The gateway contract already carries optional root `weight_version` for RL staleness. A harness/rules version is the same shape: stamp responses with `prompt_version`, freeze rules per episode (mid-episode updates would violate append-only), and the trainer/optimizer knows exactly which behavior configuration produced each trace. This composes directly with `realtime-rl-single-rollout.md` into a **two-level continual-learning story**: fast, interpretable prompt-rule updates (reflection, no GPUs) + slow weight updates (PRPO) — both served by Pie, both versioned, one telemetry stream. That combined framing is a stronger writeup headline than either alone.

## 3. Traps and open questions

- **Cumulative-token-mode consistency (must resolve first).** rllm's gateway builds turn-N prompts from prior turns' echoed token ids and enriches traces expecting exact prefix extension (`EnrichMismatchError` ⇒ retry burn). Server-side injected tokens appear in `prompt_token_ids` but were never sent by the harness — need to verify whether the gateway's cumulative rewrite tolerates the divergence (echo is authoritative?) or chokes. If it chokes, injection must happen *gateway-side* (a formula-aware proxy tweak, like our Phase 1a `parse_response` fix) rather than inferlet-side for training runs; inferlet-side stays clean for eval/serving. This is the one genuine integration unknown.
- **Determinism audit interaction.** Injection must be a pure function of (episode id, prompt_version) — re-applied byte-identically on every turn's resend — or it breaks the append-only invariant we validated in Phase 0A. Forbid mid-episode rules updates; version bump = new episodes only.
- **Reward sparsity, same as PRPO's caveat.** Contrastive reflection needs both wins and losses in a batch of traces; swesmith slices where everything fails give the reflector nothing to contrast. Same task-slice-selection discipline as the R-sync arm.
- **Their Trainer wants to run rollouts itself** (`AgentRolloutEngine` / `LocalRolloutEngine`). We'd bypass it: keep rllm as the rollout engine, feed episodes + rewards into the optimizer as a custom engine or offline batch. Thin adapter, but it's ours to write and the trace format is undocumented — read the source, don't assume.
- **Very early-stage library**: single PyPI release (0.0.1), 24 GitHub stars, last push 2026-07-17. Expect API churn and undocumented trace formats — vendor or pin the exact commit on first use; treat the Formula/Trainer interfaces as read-the-source, not stable contracts.

## 4. Experiment plan (GPU-light; can start before the RL phases finish)

**Arm H0 — optimizer without Pie (no gating; needs only Phase 0A infra):** run ContrastiveReflectionOptimizer offline over existing + newly collected qwen-code/swesmith eval traces, injecting rules by editing the system prompt at the *harness* config level (no server tricks). Exit: does prompt evolution move accuracy on our slice at all (their +11–23pt is on different benches)? Cost: inference-only eval runs + reflection LLM calls. This also produces the **prompt-optimization baseline for the RL writeup** — "how much of the RL gain does cheap prompt evolution capture?" is an ablation reviewers will ask for.

**Arm H1 — inferlet injection (after Phase 1b):** move the winning rules block from H0 into inferlet-side injection; resolve the cumulative-token-mode question (§3 first bullet); assert append-only + enrichment still pass on injected episodes.

**Arm H2 — cache-stable modules benchmark (with/after Phase 4):** measure prefill-per-optimizer-iteration, inferlet snapshot modules vs vLLM APC full re-prefill, under a realistic update cadence. This is the differentiated Pie number (§2b).

**Arm H3 — two-level continual learning (after Phase 3 + R-async):** rules updates + PRPO weight updates in one live loop, `prompt_version`/`weight_version` both stamped. Stretch goal; only if R-async and H1 both survive.
