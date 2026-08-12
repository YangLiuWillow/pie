# Real-Time RL (single-rollout, batch-normalized) on the Pie Rollout Server

**Status:** idea note v1 (2026-08-06) — experiment arm for `pie-rl-verl-integration.md`, not a scope change. One of several ideas being evaluated in parallel this period (see also `parallel-reasoning-agent-rl.md`).
**Source:** rLLM project post "Continual Learning via Real-Time RL for Agents" — https://rllm-project.com/post.html?post=realtime_rl.md (content served from `/posts/realtime_rl.md`; companion post `continual_learning.md` introduces the PRPO estimator). Reference impl: `rllm/cookbooks/migrationbench/` (AgentCore/Strands harness — harness-specific parts do NOT transfer; the algorithm does).

---

## 1. The idea in one paragraph

GRPO needs k rollouts per task to form a within-group baseline; deployed agents see each production interaction exactly once. The rLLM post shows single-rollout learning works if the advantage is **batch-normalized across different prompts**: `A_i = (R_i − batch_mean) / batch_std` over a training batch (~128). On MigrationBench (Qwen3-Coder-30B, Java 8→17): baseline 43% → REINFORCE 48.2% → batch-normalized 59.2%, matching GRPO at equal rollout budget. Their explanation: with sparse rewards, 30–50% of plain-REINFORCE rollouts have zero advantage; batch normalization makes nearly every rollout carry signal. In the **asynchronous** variant (rollout workers keep generating while the trainer updates weights), truncated importance sampling is load-bearing: without TIS training collapses to 4%; with it, 51.8%.

## 2. Why this is nearly free for us to try

The algorithm is already in our rllm checkout — this is a **trainer-side config change, orthogonal to the Pie gateway seam** (the rollout server never sees whether a task is sampled once or 16 times):

- **Estimator**: `rLLMAdvantageEstimator.PRPO` (`rllm/rllm/trainer/algorithms/config.py:251`), implemented at `rllm/rllm/trainer/algorithms/advantage.py:114` — flattens rewards across groups, centers by batch mean, normalizes by batch std. Its docstring cites this post series. Note: the `base.yaml` comment for `adv_estimator` lists only `[grpo, reinforce, reinforce_plus_plus_baseline, rloo]` — the comment is stale; `prpo` is registered and dispatchable.
- **TIS**: `rllm.algorithm.rollout_correction.{tis_mode: token|sequence, tis_cap: 2.0, bypass_mode}` (`rllm/trainer/config/rllm/base.yaml`), applied in `rllm/trainer/verl/verl_backend.py:669`. **verl-only** (our backend) — on tinker it's a documented silent no-op (`docs/training/capability-matrix.mdx`).
- **Config delta** vs our planned GRPO run:
  - `algorithm.adv_estimator=prpo` (replaces `grpo` + `norm_adv_by_std_in_grpo`)
  - `actor_rollout_ref.rollout.n=1` (their reference uses 16; batch of 32–128 distinct tasks per step)
  - async arm only: `rollout_correction.tis_mode=token`, real rollout logprobs (see §4)

## 3. Interactions with the Pie thesis (§1 of the parent spec)

1. **`n=1` removes one of the two prefix-sharing legs.** The parent thesis rests on (a) `rollout.n` siblings sharing a task prefix and (b) multi-turn history resend. Single-rollout kills (a). But the OpenHands result (95.8% KV reuse) came almost entirely from (b) — within-trajectory resend — so the dominant leg survives. Framing for the writeup: **real-time RL is the setting where per-episode KV reuse matters most**, because there is no sibling batch to amortize prefill against. A Pie-vs-vLLM-APC delta measured under `n=1` is the *conservative* version of the benchmark.
2. **Async real-time RL is the intended workload for Phase 3.** "Rollout workers generate while the trainer updates weights" is exactly the in-place UpdateWeights + KV-invalidation machinery. The gateway contract's optional root `weight_version` (§3.1 of the parent spec) is precisely the staleness signal the trainer needs to compute TIS ratios against the right behavior policy — in this arm it stops being optional metadata.
3. **Continual-learning story for the artifact.** The post's pitch (learn from one-off production interactions) is customer-shaped, which matches the §10.7 rationale for qwen-code over OpenHands. If the arm works, "Pie as the serving substrate for continual/real-time RL" is a stronger artifact headline than "faster GRPO rollouts."

## 4. Prerequisite upgrade: sampled-token logprobs become REQUIRED (async arm)

Parent-spec stance is "logprobs optional for training v1; `transform.py` pads 0.0 and PPO recomputes π_old trainer-side." That is fine for the **sync** arm (strict on-policy, ratios computed trainer-side). It is **silently corrupting** for the async arm: TIS divides by rollout-time behavior logprobs (`transform.py:205` forwards them when present), and zero-padded values would produce garbage importance weights with no error. Consequences:

- Phase 0 Track B's sampled-token-logprob driver primitive (~50-line add, already planned) is a **hard blocker** for the async arm — schedule it before, not opportunistically.
- The `rl-completions` inferlet must return `logprobs.token_logprobs` from actual sampling, not probe-recomputed values, and stamp `weight_version` on every response.
- Add a fixture assertion in Phase 1b: logprobs in the response are finite, non-zero-padded, and length-matched to `choices[0].token_ids`.

## 5. Experiment plan (two arms, no reordering of parent phases)

**Arm R-sync — after Phase 2** (restart-based weight sync suffices):
- qwen-code harness on rllm-swesmith (our existing baseline slice), Qwen3.6-27B or 0.6B smoke first.
- Three configs at equal rollout budget: GRPO (`n=8/16`) vs REINFORCE (`n=1`) vs PRPO (`n=1`), batch ≥ 32 distinct tasks.
- Exit: reproduce the post's ordering (PRPO ≈ GRPO ≫ REINFORCE) on our task slice. If PRPO ≪ GRPO here, the arm dies cheaply and Phase 4 stays GRPO-only.
- Watch: PRPO normalizes across *tasks*, so a batch with skewed difficulty mix gives biased advantages — log per-batch reward mean/std; if std ≈ 0 (all-fail batches on a hard slice), advantages vanish — pick the task slice for reward diversity.

**Arm R-async — after Phase 3** (needs in-place UpdateWeights + §4 logprobs):
- Rollout workers generate continuously; trainer updates in place; staleness = `weight_version` gap.
- A/B: TIS on vs off at fixed staleness (expect the post's collapse-without-TIS signature; if we don't see it, our staleness is too low to matter — increase update frequency before concluding TIS is unnecessary).
- This arm doubles as the Phase 3 acceptance test under realistic load.

**Phase 4 placement:** if R-sync survives, add `n=1` PRPO as a benchmark condition alongside `n=8` GRPO in the Pie-vs-vLLM A/B — it isolates the multi-turn-resend leg of the thesis (§3.1).

## 6. Open questions

- Does PRPO's cross-task normalization interact badly with rllm's `stepwise_advantage` broadcast mode for multi-step agent episodes? (The post's agent is single-episode-reward; so is ours — broadcast should be a no-op difference, verify once in R-sync.)
- Post uses batch 128 and mentions PPO clipping even in the sync path — check whether `eps_clip=0.2` default matches their setup or needs the asymmetric `eps_clip_high`.
- `bypass_mode` (π_old := π_rollout) vs TIS: mutually exclusive framings of the same logprobs (`tis_mode` requires `bypass_mode=false` per fireworks docs) — decide per-arm, don't mix.
- MigrationBench rewards are relatively dense (build/tests/test-count); swesmith pass/fail may be sparser — if all-zero batches are common, PRPO's advantage over GRPO shrinks. Measure reward density in the Phase 2 loop before judging arm results.
