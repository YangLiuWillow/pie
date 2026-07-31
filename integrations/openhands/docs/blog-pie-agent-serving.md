# I Benchmarked My Coding Agent Against Two LLM Serving Engines. Here's What Actually Matters.

*A hands-on tour of what happens between "the agent calls the LLM" and "tokens
come back" — with real numbers from a week of benchmarking
[Pie](https://pie-project.org) against vLLM under a live SWE-bench coding agent,
several embarrassing bugs, and instructions for running your own local coding
agent on Pie at the end.*

*Draft — living document. Every number traces to a predictions file or sweep log
in `integrations/openhands/`; hardware named next to each number.*

---
I bumped into Dr. Sebastian Raschka's article titled [Using Local Coding Agents](https://magazine.sebastianraschka.com/p/using-local-coding-agents) in June and greatly enjoyed it! So I am writing a little tutorial on how to run a local coding agent, and make it *faster*! 

If you've played with local coding agents, you've probably had this experience:
you point your agent at a local model server, it works, and then you wonder —
*is this thing actually fast? Compared to what?*

I spent the past week finding out properly. The setup: an
[OpenHands](https://github.com/OpenHands/software-agent-sdk) coding agent
solving real SWE-bench tasks, backed by two different serving engines on the
same GPU — [vLLM](https://github.com/vllm-project/vllm), the de-facto standard,
and [Pie](https://pie-project.org), a research system from Gim et al.'s SOSP '25
paper ([PDF](https://ingim.org/papers/gim2025pie.pdf),
[DOI](https://doi.org/10.1145/3731569.3764814)) that treats each *conversation*
— not each request — as the thing being served.

Along the way I learned more than I expected about what an agent workload
actually does to a serving engine. That's the real subject of this post.

## Why agents are a weird workload

Here's the thing about a coding agent that most serving benchmarks miss: **an
agent doesn't send requests, it grows one giant conversation.** Every step, the
agent re-sends the *entire* history — system prompt, task, every tool call and
result so far — plus one new tool output. Over a SWE-bench task that's 30–60 LLM
calls, with prompts swelling to ~47,000 tokens. And between calls, the GPU sits
idle while pytest runs.

Two consequences, and they drive everything below:

1. **Prefill dominates, but only if you're naive about it.** Re-processing 47k
   tokens on every call would be catastrophic. Both engines avoid it with
   *prefix caching*: the KV cache (the model's per-token attention state) from
   the previous call is reused, and only the new suffix is computed. Measured on
   this workload, both engines reuse ~96% of prompt tokens — vLLM automatically
   (hash the prefix, look it up), Pie explicitly (the conversation *is* a
   server-side object you append to).
2. **Memory is the real currency.** A 47k-token conversation holds ~4.5 GB of
   KV cache on this model. Eight concurrent agents want ~30 GB of a pool that,
   after 60 GB of weights, is ~12 GB. Somebody has to give. How an engine
   handles that moment turns out to matter more than raw speed.

**The setup for all numbers below:** Qwen3-Coder-30B-A3B (a mixture-of-experts
model: 128 experts, 8 active per token), one H100 80GB, vLLM 0.25.1 pinned,
13 SWE-bench instances, temperature 0, both engines capped at 2048 output
tokens. And a fairness rule learned the hard way — an earlier internal
comparison flattered Pie by ~26% because vLLM ran with CUDA graphs off and an
untuned MoE kernel. The rule now: *each engine gets the configuration a
competent user reaches with that project's own documented tooling*, and every
run's server banner is archived as proof. (A second lesson from that era:
never compare raw wall time — agents walk different trajectories even at
temperature 0, so we report seconds-per-iteration and per-call latency.)

## Round 1: single conversation. vLLM wins, and the margin is instructive.

| metric | Pie | vLLM | ratio |
|---|---|---|---|
| s/iter | 1.204 | 1.061 | 1.14× |
| median call latency | 1.06 s | 0.85 s | 1.25× |
| in-call throughput | 144 tok/s | 160 tok/s | **1.11×** |
| prefix reuse | 95.7% | 96.9–97.1% | ≈ tie |

An 11% gap. But *where* the gap lives is the educational part. We split
per-token decode cost into a context-independent part (reading weights, running
the MoE) and a context-dependent part (reading the KV cache):

![Decode parity: per-token cost and KV-read bandwidth, Pie vs vLLM](figs/fig2_decode.png)

Two things I didn't expect. First, **Pie's attention kernel beats
FlashAttention-3** on KV bandwidth (2,988 vs 2,829 GB/s — 89% vs 84% of the
H100's theoretical peak). Second, both engines read MoE expert weights at a
miserable ~30% of peak during decode — and that's not sloppiness, it's physics:
at batch size 1, each generated token reads 8 *scattered* experts' weights
(~5 GB) to do almost no math with them. There is no headroom there for either
engine. Decode, it turns out, is a solved problem.

## Round 2: prefill, where I found an actual bug and two free levers

Prefill was different: Pie started **8.6× slower** than vLLM. That number
decomposed into three findings, in escalating order of "wish I'd known this
Monday":

**Finding 1 — a per-layer sync was strangling everything.** Pie's MoE prefill
did a device-to-host copy plus a full stream synchronize *per layer* — 48
pipeline stalls and ~43,000 kernel launches per forward pass, regardless of
token count. Routing it through an on-device path that already existed (a
one-line threshold change): 2,890 → 8,610 tok/s.

**Finding 2 — chunk size is secretly a bandwidth knob.** Here's the arithmetic
that made this click for me. In any decently sized batch of tokens, an MoE
layer activates *essentially all* 128 experts — so every forward pass reads the
full ~58 GB of expert weights, *no matter how many tokens it carries*. Pie
prefilled in 512-token chunks: 58 GB ÷ 512 tokens ≈ 113 MB of weight-reads per
token, which caps prefill at ~30k tok/s on bandwidth alone. vLLM batches 8,192
tokens per step — 16× better amortization, which is most of its prefill
advantage. Forcing Pie's planner to 2,048-token chunks (one env var): **16,634
tok/s.** (Why not 8,192? On an 80 GB card the activation workspace for a chunk
that big eats the KV pool — 124k → 28k tokens. Remember "memory is the
currency"; it'll be back.)

**Finding 3 — a "dead" lever came back to life.** Bigger GEMM tiles had been
measured useless at 512-token chunks (the padding overhead ate the gains). At
2,048-token chunks the padding math changes completely and 64-row tiles gave
another +45%: **24,120 tok/s.**

![Prefill throughput progression: bug fix, chunk size, tile size, vs vLLM](figs/fig1_prefill.png)

Net: prefill went from 8.6× behind to **1.86× behind, with zero kernels
written** — a bug fix and two environment variables. The remaining 1.86× is
kernel-generation shape: we built and measured the obvious CUTLASS fix (proper
variable-size grouped GEMM), and it *lost* to the simple approach by 6% —
because the compiled CUTLASS kernels are Ampere-era and can't use the H100's
TMA hardware, while vLLM's Triton kernels are H100-native. The lesson I keep
re-learning: **measure before building — plausible-on-paper lost to
"embarrassingly simple" every time this week.** (Compiling the Hopper-native
kernels is the one open item; it's a build project, not a mystery.)

## Round 3: eight agents, one GPU — where the engines stop being comparable

At concurrency 8, the KV demand (~240k tokens) is double the pool. This is the
regime where architecture, not kernels, decides what happens — and where this
project got genuinely interesting.

**vLLM's answer** is built-in: an admission queue. It runs the ~4 conversations
that fit and makes the rest wait; preempted work is recomputed later (cheap,
because its prefill is fast). Boring, robust, works: 8/8 instances completed.

**Pie's answer** is a genuinely different design: the engine has an internal
*market*. Each conversation's server-side context bids for GPU residency;
under pressure, low bidders are evicted to a host-RAM swap pool and restored —
at PCIe speed, no recompute — when they bid higher. The policy (who bids what,
when) lives in a small WebAssembly program you can edit.

What I can now tell you from experience: **the market works, after five bug
fixes that each taught me something about the design.** The first c8 run failed
0-for-8 — not because the machinery was missing, but because the benchmark's
original request pattern never created the live objects the market manages.
Building the policy (~40 lines: park cheap when idle, bid high when serving,
generate on a fork so the reusable state stays clean) surfaced real bugs — a
stale-counter off-by-one that only fired at exact page boundaries under
pressure, an eviction rule that let equal bidders evict each other in circles
(~7 whole-conversation evictions *per second*, throughput cratered), a
context-per-turn leak. Each was found by a 4-minute repro script, fixed, and
re-measured. That loop — *policy bug, measured; engine bug, measured; fix,
measured* — is the actual argument for programmable serving, more than any
single benchmark number.

The result, matched instances under identical 8-way pressure:

| instance | Pie s/iter | vLLM s/iter | ratio |
|---|---|---|---|
| django-12276 | 4.26 | 4.09 | 1.04× |
| django-13089 | 4.68 | 4.68 | **1.00×** |
| django-15569 | 4.45 | 4.14 | 1.08× |
| matplotlib-22719* | 5.88 | 3.90 | 1.51× |

*\*divergent trajectories (69 vs 96 iterations, completely different patches) —
not a clean speed comparison.*

**Parity, on cleanly-matched instances.** Both engines degrade ~4× from their
single-conversation numbers — the cost is the 8-way rotation itself, which both
reach by different roads.

![c8 overcommit completion rates](figs/fig4_overcommit.png)

And one final twist that ties the whole post together: I tried stacking the
prefill lever (2,048-token chunks) onto the c8 run, expecting the best of both.
It *collapsed* — the lever's 15% KV cost shrank the pool from ~4 concurrent
seats to ~2.5, and six conversations queued behind whole generations. **Speed
levers and capacity levers trade against each other, and under overcommit,
capacity wins.** Config-per-regime (fast chunks solo, lean chunks under load)
is the operating discipline — and in Pie, that could literally be policy code.

## What I'd tell you to take away

1. **For agent workloads, prefix caching is table stakes and both engines
   nail it** (~96%). If your serving setup doesn't, fix that before anything.
2. **Decode speed is a red herring** at batch 1 — everyone's at the same
   physical floor. Prefill and memory policy are where engines differ.
3. **Chunk size × KV pool is the trade to understand** on small GPUs. It shows
   up twice above, once as a speed win and once as a capacity collapse.
4. Pie's honest scorecard today: ~11% behind at c1, parity at c8, fork/branch
   and programmable-policy capabilities vLLM doesn't have, and a younger
   engine's rough edges (we hit real bugs; all fixed-and-measured in a day —
   which is itself informative about the codebase).

We'll extend this to Codex and other harnesses next — likely through Pie's
OpenAI-compatible HTTP inferlet, so any `base_url`-configurable agent works
unmodified. `[PENDING: multi-harness arms]`

## Appendix: run your own coding agent on Pie

Everything below assumes one CUDA GPU (sm_90+ gets the fast attention paths;
80 GB fits the 30B MoE in bf16) and the
[`openhands-integration-updated` branch](https://github.com/YangLiuWillow/pie).

```bash
# 1. Build the engine (Rust + CUDA; ~30-60 min cold)
git clone -b openhands-integration-updated https://github.com/YangLiuWillow/pie
cd pie && cargo build -p pie-server --release --features driver-portable,driver-cuda

# 2. Build the coding-session inferlet (the per-conversation wasm program)
(cd inferlets/openhands-coder-session && cargo build --target wasm32-wasip2 --release)

# 3. Serve. The toml picks the model + memory profile; kv_page_size=32
#    keeps the fast decode kernel on.
export PIE_CUDA_KV_PAGE_SIZE=32
./target/release/pie serve \
  --config integrations/openhands/runpod/pie_cuda_native_config_30b_moe_h100_overcommit.toml \
  --port 18097 --no-auth
# Watch the banner: you want `prefill_decode_plan=on xqa_decode=on`.

# 4. Speed levers for single-user use (see "Round 2"; skip under heavy concurrency):
export PIE_CUDA_PREFILL_TOKENS=2048 PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK=64

# 5. Point OpenHands at it. The integration ships a LiteLLM-compatible model
#    class that speaks Pie's session protocol (persistent contexts, prefix
#    reuse, live-context mode):
cd integrations/openhands
uv venv && uv pip install --exclude-newer 2026-05-15 -e .
python -m benchmarks.run_swe_bench --backend pie --pie-session \
  --pie-uri ws://127.0.0.1:18097 \
  --pie-inferlet openhands-coder-session@0.1.0 \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --instance-id django__django-14373 --temperature 0
# Or use pie_openhands.PieLLM directly in any OpenHands SDK script.

# 6. Optional: the pressure-policy experiment from Round 3
export PIE_LIVE_CONTEXT=1        # live contexts + bid policy
export PIE_LIVE_IDLE_SUSPEND=1   # aggressive variant: park to host RAM every turn
```

Two practical warnings from the trenches: read the server banner every time
(a silently-off fast path cost this project a full invalid benchmark run), and
if you share the GPU with anything else, 60 GB engines do not co-reside —
serialize, don't hope.

*Repro for every number: `integrations/openhands/runpod/` — arms, banners,
sweep logs, and the defect chronicle in `DEFECTS_OVERCOMMIT.md`.*
