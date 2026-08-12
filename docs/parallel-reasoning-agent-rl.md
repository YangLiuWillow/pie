# Parallel Reasoning in Agentic RL on Pie ("Phase 6" design note)

**Status:** design note v1 (2026-08-06) — follow-on to `pie-rl-verl-integration.md`; NOT part of that project's scope.
**Papers:** Multiverse (arXiv 2506.09991, CMU) · Native Parallel Reasoner / NPR (arXiv 2512.07461, BIGAI, ICML 2026). PDFs in session scratchpad; both open-source their stacks (Multiverse: github.com/Multiverse4FM).

---

## 1. The idea in one paragraph

Multiverse showed LLMs can natively generate in a MapReduce topology (decompose → parallel branches → lossless KV merge) with SFT only; NPR showed the capability can be *self-taught and amplified with RL* (PAPO), reaching +24.5% accuracy and 4.6× decode speedup on math benchmarks — but both are single-turn, tool-free reasoners, and both had to fork SGLang to get controlled branching. The open niche: **parallel reasoning inside each turn of a tool-using agent, trained with RL in a real harness** (qwen-code + rllm/verl), served by Pie — where branch/merge topologies are inferlet programs rather than engine forks. This composes the two theses of the parent project: Pie's programmable KV is the substrate both papers hand-built, and the parent project's RL loop is the missing training infrastructure both papers lack for agentic settings.

## 2. What the papers established (mechanics we inherit)

### Multiverse (2506.09991)
- **Model**: Map stage (sequential plan with `<Parallel><Goal><Outline>` tags) → Process stage (`<Path>` branches decode independently, sharing prefix KV) → Reduce stage (`<Conclusion>` conditioned on ALL branches via lossless KV concatenation, no summarization loss).
- **Multiverse Attention**: branches mutually invisible via attention mask; all branches start from the same position id; Reduce restarts at max position reached by any branch — trains like causal attention (few thousand examples suffice to convert an AR model).
- **Data**: Multiverse Curator — LLM pipeline rewriting sequential CoT into the parallel schema (5 stages + edit-distance/grammar checks). Finding: >98% of s1K long-CoT traces contain parallelizable branches (79% collective/subtask, 19% selective/path-exploration), but AR models can't explicitly enforce them (probing classifier ≈ random).
- **Engine**: SGLang fork — interpreter dispatches on control tags, Map spawns paths sharing radix-cache prefix, finished paths enter "zombie" state, Reduce concatenates KV indices without copying, then continues sequentially.
- **Results**: Multiverse-32B (SFT on 1K examples, 3 h) ≈ AR-32B quality; ~2× per-token speedup at parallelism ~1.2–2; better accuracy-per-context-length.
- **Stated limitation: no RL** — "requires a more robust engine."

### NPR (2512.07461)
- **Three stages, teacher-free**: (1) NPR-Zero — format-follow RL (DAPO; format reward 0/-2, accuracy ±1) makes the base model *discover* a leaner schema (`<guideline>/<plan>/<step>/<takeaway>`, Map-Process-Reduce) while still decoding sequentially ("simulated parallelism"); (2) NPR-Beta — rejection-sample NPR-Zero's outputs (correct ∧ well-formatted), parallel-SFT with Multiverse-style masks/positions — self-distillation beats Multiverse's teacher-distilled corpus by ~10 pts; (3) NPR — **PAPO** RL natively in the parallel execution graph.
- **PAPO specifics** (all load-bearing for us):
  - Rollouts via a strict parallel engine so every trajectory obeys the schema; **schema-level filtering** (mask/position-encoding-level, not regex — text checkers "always miss rare corner cases") drops malformed rollouts pre-optimization → reward reduces to accuracy only.
  - **Batch-level advantage normalization** (Lite-PPO style) instead of group-level — filtering collapses group variance.
  - **Preserve gradients on branch/merge special tokens** — clip-masking them breaks the learned structure.
  - **Strict on-policy, no importance sampling**: objective uses stop-gradient ratio `π/sg[π]`; removing clip-masking made IS ratios unstable, so they dropped IS entirely. (Convenient for us: strict on-policy is also our simplest logprob story.)
- **NPR Engine war stories** (= our test checklist): KV-cache double-free in shared radix paths under heavy branching; global token budget under-counting parallel branches (runs blowing past `max_new_tokens`); undefined states from illegal branch layouts (fixed by cheap pre-branch structural validators); local repetition inside `<step>` blocks (selective repetition penalty, 1.02, inside steps only).
- **Results** (Qwen3-4B backbones): avg 65.0 vs 62.0 sequential-RL; up to +24.5 over base; TPS speedup 2.9–4.6× (larger on harder benchmarks); 100% parallel trigger rate vs Multiverse's 45.8–76% (30%+ AR fallback in prior parallel models).

## 3. Why our stack is the natural home

1. **Both teams forked SGLang; Pie does this natively.** Fork-with-shared-prefix = `Context::fork()` + refcounted radix pagestore. Caller-supplied positions = `Forward::input_at`. Custom attention masks = per-request mask support (scheduler `max_custom_mask_bytes`). Branch-aware budgets/flow control = inferlet code. Existing inferlets (`demo-parallel-fork`, `best-of-n`, `graph-of-thought`, `agent-swarm`) are primitive versions of the topology. NPR's engine bugs are precisely the failure modes Pie's page-refcount design targets — and precisely what we must stress-test.
2. **The parent project builds the missing RL substrate**: token-level rollouts through a Pie worker (Phase 1b), weight sync (Phase 3), exact sampled-token logprobs (Phase 0 Track B), deterministic qwen-code harness (Phase 0 Track A), verifiable sandbox rewards (rllm tasks). Neither paper has tool use, multi-turn, or a harness.
3. **The niche is genuinely open**: parallel reasoning × agentic RL × real harness appears in neither paper nor (per NPR's related-work sweep: APR, Parallel-R1, ParaThinker, Parallelsearch) anywhere else yet. Speedups grow with task difficulty (NPR: 4.6× on AIME25 vs 2.9× on AMC) — agent turns are hard problems with many viable exploration paths.

## 4. The experiment (candidate design)

**Shape:** parallel reasoning *within* each agent turn, sequential at the harness boundary. The model thinks in Map-Process-Reduce inside its completion (explore k hypotheses about the bug / candidate fixes / candidate commands in parallel branches), the Reduce block commits to a decision, then a normal sequential tool call is emitted. The wire protocol (chat completions, tool_calls) is untouched — parallelism is confined inside a single completion, invisible to qwen-code and the gateway.

**Stages (mirroring NPR's curriculum, adapted):**
1. **P6.0 — Inference-only demo (no RL, cheap, high demo value):** run an existing open checkpoint (Multiverse-32B or NPR-4B — both Qwen-family) through a new `parallel-completions` inferlet implementing the tag-interpreter loop on Pie primitives. Deliverable: qwen-code (or plain chat) driving a natively-parallel model on Pie, with per-token latency vs sequential measured. This alone is a "Pie replaces their bespoke engines" artifact.
2. **P6.1 — Format induction on agent traces:** NPR-Zero-style format-follow RL (or curator-style SFT from our own rollout corpus) teaching the policy the parallel schema *in the agent context* (tool outputs in history, decision-oriented Reduce). Sequential engine; format+accuracy reward; reuses the parent project's Phase 2 loop unchanged.
3. **P6.2 — Parallel SFT:** rejection-sample P6.1, train with parallel masks/positions (extend `transform.py` — see §5.2).
4. **P6.3 — PAPO in the harness:** parallel rollouts served by the inferlet; schema-level filtering; batch-level advantage norm; gradients preserved on control tokens; strict on-policy. Reward = task verifier (unchanged).
5. **P6.4 — Measure:** accuracy on held-out tasks, wall-clock per episode, parallel trigger rate (NPR's metric), reuse ratio — vs the sequential-RL policy from the parent project's Phase 4 (which becomes this experiment's baseline arm).

## 5. Hard problems (ranked)

### 5.1 The linear-sequence assumption in the training path — biggest lift
rllm's trace converter, the gateway's cumulative-token accumulator, and `transform.py`'s prefix-extension merging all model a rollout as an append-only token chain; a parallel block is a DAG (`P(ŷ|q) = ∏ P(s_t | Pa(s_t), q)`). Required: (a) trace schema extension carrying branch topology (branch spans + parent map) from the inferlet through the gateway to the trainer — likely as an rLLM-extension field on the completions response, mirroring how `routing_matrices` already flows; (b) `transform.py` emitting Multiverse-style attention masks + position ids into the DataProto (NPR Alg. 1–2 are the reference construction); (c) verl's actor forward honoring custom masks/positions (verify — NPR built theirs on verl-adjacent infra, so this is proven feasible on vLLM-era stacks). Cumulative-token mode interaction: within-turn parallelism keeps turn boundaries linear, so the accumulator's cross-turn invariant survives; only the within-completion structure needs the new schema.

### 5.2 Reduce-stage KV in Pie — needs a spike
Fork is native; the question is the merge. Multiverse's Reduce = one sequence whose KV is [prefix ‖ branch₁ ‖ … ‖ branchₖ] with realigned positions. Pie options: (i) forward requests referencing sibling contexts' committed pages (`max_page_refs`) with `input_at`-style position override — lossless, the Multiverse way; needs verification that page refs can span contexts and that position metadata is per-request not per-page; (ii) v0 fallback: replay branch outputs into the merged context (correct, costs one prefill pass over branch tokens — bounded by branch length, and chunked-prefill machinery already exists). Spike deliverable: a `parallel-fork-merge` micro-inferlet proving (i) or quantifying (ii)'s overhead. Note: position-id realignment interacts with the pagestore hash (pages keyed by token chain — do merged-topology pages dedup correctly, or must they be treated like `adapter_seed`-tagged pages?).

### 5.3 Schema × harness coexistence
The parallel tags must not collide with qwen-code's tool-call format or the `<think>` channel. Keeping parallelism inside the reasoning segment (before any tool_call emission) and choosing tag vocab disjoint from Qwen's chat-template specials should suffice; the audit's `filter.rs`-style streaming filter strips structural tags from user-visible deltas. The XML tool-call dialect question (audit §8.2) must be settled first.

### 5.4 Engine hardening under branching RL load
Adopt NPR's bug list as a Pie stress-test suite: branch-heavy sustained load hunting refcount/double-free issues in the pagestore; branch-aware token-budget accounting in the inferlet (global ledger across branches, not longest-branch); pre-branch structural validation; per-`<step>` repetition penalty (Pie samplers support per-request penalties — verify repetition_penalty exists or add).

### 5.5 Probe/batch limits
A k-branch turn multiplies concurrent decode streams and (if logging logprobs) probe rows. `max_logprob_labels`/`max_prob_rows`/`max_forward_requests` sizing from the parent project's G6 gets multiplied by average branching factor.

## 6. Dependencies on the parent project

| Needs | From |
|---|---|
| Deterministic qwen-code harness + launch profile | Phase 0 Track A |
| `sampled-logprob` primitive, seeding fixes | Phase 0 Track B |
| `rl-completions` inferlet (HTTP shell, token-in/out, streaming, snapshots) — `parallel-completions` extends it | Phase 1b |
| Golden-trace fixture workflow | Phase 1a |
| Training loop + weight sync | Phases 2–3 |
| Sequential-RL policy as baseline arm | Phase 4 |

P6.0 (inference demo) only needs Phase 1b's shell and can run early as a side quest; P6.1+ should not start before Phase 3 is stable.

## 7. Risks specific to this follow-on

- **Small-model ceiling in agentic settings**: NPR used Qwen3-4B on math; whether a 4–8B policy can learn *useful* parallel decomposition of coding-agent turns (vs degenerate always-1-branch or redundant-branch behavior) is the core scientific risk. Mitigation: NPR's trigger-rate + branch-diversity metrics from day one; P6.1's format stage will reveal degenerate collapse early.
- **Parallelism may not bind on agent turns**: agent turns are shorter than AIME solutions; if median useful branching is ~1, speedup evaporates. P6.0's inference demo on real tasks gives an early read cheaply.
- **Compounding schedule risk**: this stacks three research-grade components (parallel training batches, KV merge, PAPO). Strict gating on parent-project phases; every stage has a standalone artifact so partial completion still yields value (P6.0 demo; P6.1 formatted-agent corpus; P6.2 parallel-SFT policy).

## 8. References

- Multiverse: Xinyu Yang et al., *Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation*, arXiv:2506.09991. Code/data/weights open (Multiverse-32B, Multiverse-1K, SGLang engine fork).
- NPR: Tong Wu, Yang Liu, Jun Bai et al., *Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning*, arXiv:2512.07461, ICML 2026. Qwen3-4B checkpoints; PAPO; NPR-Engine on SGLang.
- Related sweep (from NPR §B): APR (2504.15466), Parallel-R1 (2509.07980), ParaThinker (2509.04475), Parallelsearch (2508.09303) — none agentic/harness-based.
