# Masked KV Condensation for OpenHands/Pie — Results Write-up

**Date:** 2026-07-21
**Model:** Qwen3-Coder-30B-A3B-Instruct (config `tests/fixtures/pie_cuda_vllm_config_30b_moe.toml`)
**Binary:** `target-grammarfix/release/pie`
**Inferlet:** `inferlets/mask-condense-bench` (`mask_condense_bench@0.1.0`)

---

## 1. Thesis

Long agent trajectories (OpenHands, SWE-bench) eventually overflow the context
window and must be **condensed** — the middle of the history is dropped, keeping
a head/sink and a recent window. The stock approach (and what APC-style prefix
caching effectively forces) is to **rebuild**: throw away the dropped KV, then
**re-prefill** the surviving tokens re-positioned compactly. That re-prefill is
pure recompute, and it compounds every time the agent condenses again.

Pie can do better because it owns the KV cache directly. Instead of rebuilding,
it can **keep the KV pages resident and simply mask attention** to everything
except an attention **sink** (`sink_size`) plus a **keep-recent** window. The
surviving tokens stay at their **original positions**; nothing is recomputed.

> **Claim under test:** masked condensation matches (or beats) rebuild on
> *quality*, while costing **zero** re-prefill.

Two independent experiments test this: a **synthetic variance** study (tight
mean ± SEM over many corpus windows) and a **fixed real-trajectory replay**
(confound-free, on an actual agent run).

---

## 2. Mechanism

At each condensation the three arms are compared on **identical probe tokens**
(the real continuation), so any perplexity gap is purely the condensation
mechanism:

| Arm | KV of dropped middle | Positions of survivors | Recompute |
|-----|----------------------|------------------------|-----------|
| **FULL** | kept (no condense) | original | none — reference floor |
| **MASK** (Pie) | kept, attention-masked | **original (gapped)** | **0 ms** |
| **REBUILD** (stock/APC) | discarded | **recompacted** | re-prefill survivors |

FULL is the upper-bound reference (whole history resident). MASK and REBUILD
drop the same middle and attend to the same surviving tokens — the **only**
difference between them is positional: MASK leaves survivors at their original
(now gapped) position ids; REBUILD renumbers them contiguously and pays to
re-prefill. `mask_reprefill_ms` is **definitionally 0** (`src/lib.rs:288`).

---

## 3. Result A — Synthetic variance study (job 19070781)

Fixed prefix `P`, `NQ=256` gold continuation tokens, `sink_size=64`,
`keep_recent=256`; the mask/rebuild perplexity ratio is averaged over many
corpus-offset windows for a tight mean ± SEM.
Out: `logs/mask_quality_var_19070781.out`.

| P (tokens) | windows | full_ppl | **mask/rebuild ratio** | mask better | reading |
|-----------:|--------:|---------:|-----------------------:|:-----------:|---------|
| 24 000 | 16 | 2.45 | **1.0222 ± 0.0152** | 6/16 | tiny ~2% mid-context penalty |
| 48 000 | 10 | 2.36 | **1.0020 ± 0.0205** | 4/10 | **quality-neutral** (SEM straddles 1.0) |

**Reading:** ratio ± SEM straddling 1.0 ⇒ mask **== rebuild** in quality, and
mask is **free**. The small (~2%) mid-context penalty at P=24k **vanishes** by
P=48k — it converges to neutral as context grows, which is the regime that
matters (condensation only fires on long contexts). Both mask and rebuild sit
slightly above the full-context floor, as any condensation must.

---

## 4. Result B — Fixed real-trajectory replay (job 19073215) — the headline

The earlier live A/B (job 19059956) was **confounded**: it ran the agent twice,
producing two *different* trajectories via layer-B engine nondeterminism, so any
delta measured trajectory variance, not the condenser. This experiment removes
the confound in two phases:

1. **Capture (once, GPU):** run `openhands-agent` on one real SWE-bench instance
   with a high context limit so it does **not** condense — a clean, linear
   transcript (`dump_trajectory=True`). Cached to
   `logs/traj_capture_<id>.json` and reused.
2. **Replay (GPU, cheap, deterministic):** tokenize that fixed transcript once
   into `token_ids` + per-turn offsets, then feed it to the `traj_replay` mode,
   which imposes a **low** context limit and, at every turn that overflows,
   probes the model's prediction of the **real** continuation three ways
   (FULL / MASK / REBUILD).

Because both condensers see **byte-identical** tokens, the mask-vs-rebuild gap is
the pure mechanism.

**Instance:** `django__django-14373` — 27 turns, 13 057 tokens.
**Replay config:** `context_limit=5000`, `keep_recent_turns=6`, `probe_tokens=48`
→ 9 condensation probes.
Out: `logs/mask_traj_replay_19073215.out`,
`logs/traj_capture_django__django-14373_replay.json`.

| turn | histlen | full_ppl | mask_ppl | rebld_ppl | **m/r** | reprefill_ms |
|-----:|--------:|---------:|---------:|----------:|--------:|-------------:|
|  9 |  5081 | 1.3503 | 1.4065 | 1.4688 | 0.958 | 231.1 |
| 11 |  5392 | 1.2086 | 1.2365 | 1.5622 | **0.792** | 187.9 |
| 13 |  6697 | 1.4664 | 1.4694 | 1.5922 | 0.923 | 143.7 |
| 15 |  7644 | 1.8280 | 1.7105 | 2.1031 | **0.813** | 217.2 |
| 17 |  8469 | 1.3683 | 1.4259 | 1.5695 | 0.909 | 232.2 |
| 19 | 10234 | 1.9827 | 2.0889 | 2.1598 | 0.967 | 244.6 |
| 21 | 11084 | 1.5123 | 1.6181 | 1.6234 | 0.997 | 240.7 |
| 23 | 11791 | 2.5321 | 2.4636 | 2.6443 | 0.932 | 237.7 |
| 25 | 12445 | 1.9831 | 1.8404 | 2.0512 | 0.897 | 174.0 |

**Headline:**

- **mean mask/rebuild perplexity ratio = 0.9096 ± 0.0214 — mask BETTER in 9/9 probes**
- **rebuild re-prefill = 1909 ms total across 9 condensations; MASK = 0 ms** (structural)

On a **real** agent trajectory, masked condensation does not merely tie
rebuild — it **wins** on quality *and* is free. The 1909 ms of re-prefill is the
compounding cost APC pays to drop the middle; mask avoids it entirely.

**Why mask beats rebuild *on this instance* (interpretation):** both arms attend
to the same surviving tokens, so the only difference is position encoding.
Keeping survivors at their **original (gapped)** positions can preserve the
model's learned positional relationships between the recent window and the query
better than recompacting them. **However, this win did not generalize** — see
§6, where a longer instance lands back at neutral. The robust cross-instance
claim is **quality-neutral**, not "mask wins"; the django result is best read as
the favorable tail of a neutral distribution.

---

## 5. Caveats & scope

- **FULL and MASK keep the whole `[0, p)` KV resident**, so probes are gated to
  `p ≤ MAX_HISTORY = 60000` (`src/lib.rs:301`), staying inside the model's
  trained/positional range. Beyond ~65k, mask's growing gapped positions leave
  the trained range and rebuild's re-positioning becomes necessary — masked
  condensation is a **within-window** technique.
- **The 0.91 "mask better" on django is instance-specific.** A second, longer
  instance (§6) returns to neutral, so the defensible cross-instance claim is
  quality-neutral, not "mask wins."
- The replay defaults (`REPLAY_CONTEXT_LIMIT=16000`, `KEEP_RECENT_TURNS=12`) do
  **not** overflow a 13k-token trajectory (that is why the first submission, job
  19072344, errored "no turn overflows" after caching the capture). The headline
  run needs the `context_limit=5000`, `keep_recent_turns=6` overrides.

---

## 6. Generality validation — job 19074738 (matplotlib-22719)

Capture + replay `matplotlib__matplotlib-22719` — a much longer trajectory to
test whether django's mask-win generalizes. Same overrides (`context_limit=5000`,
`keep_recent=6`, `max_steps=100`). **Instance:** 49 turns, 34 261 tokens →
14 condensation probes. Out: `logs/mask_traj_replay_19074738.out`.

| turn | histlen | full_ppl | mask_ppl | rebld_ppl | **m/r** | reprefill_ms |
|-----:|--------:|---------:|---------:|----------:|--------:|-------------:|
|  7 |  7384 | 1.8982 | 1.7134 | 1.5109 | 1.134 | 302.5 |
| 10 |  8110 | 2.0335 | 2.0822 | 2.1915 | 0.950 | 317.4 |
| 13 |  8520 | 2.0739 | 2.2425 | 3.1556 | **0.711** | 255.3 |
| 16 |  9839 | 1.5652 | 2.1334 | 2.0364 | 1.048 | 270.5 |
| 19 | 11126 | 1.4415 | 1.7363 | 1.5683 | 1.107 | 302.0 |
| 22 | 15063 | 1.4966 | 1.8298 | 1.3798 | 1.326 | 469.2 |
| 25 | 19908 | 1.8572 | 1.8475 | 1.6735 | 1.104 | 714.8 |
| 28 | 20687 | 1.7859 | 1.9418 | 1.7544 | 1.107 | 499.7 |
| 31 | 21386 | 2.7878 | 2.5377 | 2.4865 | 1.021 | 262.6 |
| 34 | 22471 | 1.5443 | 1.7905 | 1.8119 | 0.988 | 271.3 |
| 37 | 23536 | 2.4249 | 3.5084 | 2.8181 | 1.245 | 285.7 |
| 40 | 29350 | 1.8162 | 1.7400 | 2.1217 | **0.820** | 557.6 |
| 43 | 30533 | 1.9964 | 2.0381 | 1.8079 | 1.127 | 557.6 |
| 46 | 32522 | 2.1342 | 2.0085 | 2.4502 | **0.820** | 333.5 |

- **mean mask/rebuild ratio = 1.0362 ± 0.0434** → band ≈ [0.993, 1.079],
  **straddles 1.0 ⇒ quality-neutral** (leaning a hair toward rebuild), mask
  better in 5/14.
- **rebuild re-prefill = 5400 ms total; MASK = 0 ms.**

**Conclusion:** django's 0.91 mask-win **did not generalize** — the longer
instance is neutral, matching the synthetic study (1.002). The robust
cross-instance claim is **quality-neutral + free**. Note the free-ness *scales*:
the longer trajectory pushed rebuild's compounding re-prefill from 1909 ms
(django) to **5400 ms** (matplotlib), while mask stays 0 ms — the benefit grows
with trajectory length.

**Three-point summary:**

| experiment | context | mask/rebuild ratio | verdict |
|------------|---------|--------------------|---------|
| synthetic variance (19070781) | P=48k | 1.0020 ± 0.0205 | neutral |
| django-14373 replay (19073215) | 13k, 9 probes | 0.9096 ± 0.0214 | mask better (instance-specific) |
| matplotlib-22719 replay (19074738) | 34k, 14 probes | 1.0362 ± 0.0434 | neutral |

## 7. Next steps

- **Compounding/drift run:** repeated mask cycles on one trajectory to check
  whether quality degrades as masks stack (the realistic production pattern).
- **Production wiring:** wire the mask path into `openhands-coder-session` so
  condensation keeps KV instead of rebuilding — the "make it real" step.
- **Cleanup:** `src/lib.rs:447` has a harmless unused-`mut` warning
  (`mask_reprefill_ms`).

---

## 8. Job lineage

`mask-condense-bench` 19057750 → `mask-condense-ab` 19059956 (confounded) →
compounding 19065660/19065819 → `mask-quality` 19066058 (failed) → 19068615 →
19068808 → hardening 19069519 (noisy) → **variance 19070781 (Result A)** →
traj-replay 19072344 (capture only, replay errored) →
**traj-replay 19073215 (django, Result B)** →
**traj-replay 19074738 (matplotlib, generality check, §6)**.

---

## 9. Bottom line

Across a synthetic study and two real agent trajectories, masked KV condensation
is **quality-neutral versus rebuild, at zero re-prefill cost** — with a single
instance (django) landing on the favorable tail (mask better) and the others
neutral. The value proposition is therefore **"same quality, free"**, not "better
quality": a condensation primitive that is at least as good as rebuild within the
model's context window while eliminating rebuild's compounding re-prefill (which
grew from 1.9 s to 5.4 s as the trajectory lengthened) — and one that only Pie,
which owns the KV cache, can offer. It is the "condensation" half of the Pie
advantage story (the "fork" half being test-time-scaling branch reuse); see
`docs/FORK_TEST_TIME_SCALING_DESIGN.md` and the eval-plan scorecard.
