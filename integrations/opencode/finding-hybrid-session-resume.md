# pie cannot resume a session on a hybrid-attention model — 2026-08-17

Serving `Qwen3.6-35B-A3B` through the `opencode-session` inferlet (strategy B),
**every turn that resumes from retained KV fails**. The first four-way SWE-bench
run produced **24 empty patches out of 24 instances** before it was stopped.

The same inferlet, same code, same machine, resumes correctly on
`Qwen3-Coder-30B-A3B` — which is what the previous 30-instance comparison ran
on, and why this was never seen.

## Symptom

Each instance made three calls and wrote nothing:

```
  #  prompt  compl      finish    ttft   total  nmsg
  0     825      9        stop  2.6705  2.8065     3
  1    7874     50  tool_calls 13.8104 14.4721     2
  2       0      0      length  0.0357   0.036     4     <-- 36 ms, zero tokens
```

The transcript ends:

```
✱ Glob "**/separable*"  1 match
The server could not complete this turn.
```

That string is `handler.rs::degrade()`, which answers a turn that died before
producing anything with `finish_reason:"length"` and zero usage — deliberately
never a 5xx, because opencode retries 5xx without bound.

## Cause

From the inferlet's stderr:

```
[opencode-session] turn 2 cached=0 delta=493 cue=7 gen=31 retained 043deee9 (len 493)
[opencode-session] generation setup failed:
prefill take @493: g0 take: channel is poisoned: driver published poison epoch 1
```

The address is exact: a turn retains its KV at length N, and the next turn's
prefill against N is refused by the driver. The turn after that re-prefills from
scratch (`cached=0`), succeeds, retains — and the turn after *that* fails again.
A clean alternation of retain / poisoned-resume.

## What it is not

Ruled out by direct experiment, each against a live server:

| hypothesis | test | result |
|---|---|---|
| the tool dialect change | plain 4-turn chat, **no tools at all** | fails identically on every resume |
| prompt size | swept 500 → 38,000 prompt tokens | degrades at *every* size |
| tool-result rendering | cold request carrying a tool result | **works** (339 tokens, `finish=stop`) |
| model size / MoE | `Qwen3-Coder-30B-A3B`, same strategy, same probe | **all turns pass** |

The control is the decisive one:

```
Qwen3-Coder-30B (paged KV)   turns 1-4 OK,  cached=13 -> 28 -> 43   (reuse working)
Qwen3.6-35B     (hybrid)     every resume poisoned, cached=0 throughout
```

Note `cached=0` even on Qwen3.6's *successful* turns: the session never achieves
any reuse on this model. The turns that appear to work are full re-prefills.

So it is the **hybrid family's session-resume path** — not size, not the
renderer, not the tool dialect.

Qwen3.6 is 40 layers of which only 10 are full attention; the other 30 are
Gated DeltaNet, carrying a recurrent state rather than paged KV. `engine.rs`
already knows retention must cover both — *"on a recurrent-state model, the
folded state belonging to the same prefix ... these two cannot be retained
separately"* — and its module docs describe an adjacent hazard in the same
words as the failure: *"a free-standing `seal()` cannot bind a fold on a fresh
pipeline (poison epoch)"*. The resume path evidently does not reconstitute the
fold for this family.

## Workaround used for the benchmark

The four-way run serves pie under **strategy A**: one `chat-completions`
inferlet per request, KV dying with the request, so there is no resume to
poison. Verified clean on a four-turn chat and a two-turn tool loop.

**This is a handicap and belongs in the results, not a footnote.** Strategy A
has no cross-turn prefix reuse, while vLLM runs `--enable-prefix-caching` and
llama.cpp keeps its slot cache. On an agentic loop that re-sends the whole
transcript every turn, pie pays full prefill each time and the engines it is
measured against do not. It lands on TTFT first and throughput second.

It also means the four-way is **not comparable to the 30B three-way** on pie's
side: that run used strategy B, and strategy B is what produced 42 tok/s at 12.3
calls per instance.

## Reproduce

```sh
tools/boot_pie.sh dbg PIE_STRATEGY=b PIE_MODEL=qwen3.6-35b-a3b \
    PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096
# then any two-turn conversation; the second turn degrades
grep -aE "turn [0-9]|failed|poison" /tmp/pie_opencode_shim.log
```

Swap `PIE_MODEL=qwen3-coder-30b` for the passing control.

## What would have caught it sooner

Nothing in the timing columns. An arm that fails every turn finishes *faster* —
24 instances in about twelve minutes, every one marked `ok`, every patch empty.
The three-way harness only counted predictions, and 30 empty predictions counts
as 30.

`swe30_four_way.sh` now reports non-empty patch count per arm and says plainly
that an all-empty arm is broken rather than inaccurate. Patch bytes are still
not accuracy — only the Docker grade is — but all-empty is not a low score.
