"""Fork-eviction microbench — does vLLM+APC evict the SHARED TRUNK under KV
pressure, forcing a re-prefill that Pie's refcounted fork avoids?

This is the controlled counterpart to `bench_vllm_fork_baseline.py`. That full-
agent run showed APC hitting 94.6% at 10-18% KV usage (no pressure) → fork and
APC at parity. The fork's structural edge over APC is *eviction resistance*, which
only appears under pressure. This isolates exactly that, with NO agent loop / tool
time / trajectory-nondeterminism / workspace confounds (same philosophy as the
Stage-1 mask microbench).

Structure (mirrors K-way fork of one long generated trunk):
  - Build ONE long shared TRUNK prompt (~TRUNK_TOKENS), identical across branches
    = the generated branch-point context all K children share.
  - For each of K branches, run ROUNDS continuation calls (trunk + a growing,
    branch-unique suffix + decode) — the multi-step divergence.
  - vLLM+APC caches the trunk ONCE (shared blocks). Under enough pressure (long
    trunk × many branches × small KV) the trunk gets evicted between uses and
    must be RE-PREFILLED. We detect that per call via the OpenAI usage field
    `prompt_tokens_details.cached_tokens`: trunk cached ⇒ cached≈trunk_len;
    trunk evicted ⇒ cached≈0.

The decisive number: **trunk re-prefill events** = branch calls whose cached
prefix fell below the trunk length ⇒ APC paid to rebuild the shared trunk. Pie's
fork pays the trunk ONCE regardless of K ⇒ its trunk-re-prefill count is 0 by
construction. As K rises / KV shrinks, APC's count should climb — that is the
fork win, quantified.

Env: MODEL, BASE_URL, TRUNK_TOKENS, K, ROUNDS, DECODE_TOKENS, SUFFIX_TOKENS,
     CONCURRENCY (1 = sequential, the eviction-prone order), OUT_JSON.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:18000")
TRUNK_TOKENS = int(os.environ.get("TRUNK_TOKENS", "12000"))
K = int(os.environ.get("K", "16"))
ROUNDS = int(os.environ.get("ROUNDS", "4"))
DECODE_TOKENS = int(os.environ.get("DECODE_TOKENS", "64"))
SUFFIX_TOKENS = int(os.environ.get("SUFFIX_TOKENS", "400"))  # per-round suffix growth
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))
OUT_JSON = os.environ.get("OUT_JSON", "logs/fork_eviction_result.json")

# ~4 chars/token; use varied, code-like filler so it tokenizes densely and the
# trunk is a genuine block-aligned shared prefix.
_WORDS = ("def compute_gradient(matrix, weights, bias, learning_rate): "
          "result = [] # accumulate partial sums across the feature dimension ")


def _filler(n_tokens: int, seed: str) -> str:
    """Deterministic ~n_tokens of text. `seed` makes branch suffixes unique."""
    chunk = f"/* {seed} */ " + _WORDS
    reps = max(1, (n_tokens * 4) // len(chunk) + 1)
    return (chunk * reps)[: n_tokens * 4]


TRUNK = "# SHARED REPO EXPLORATION CONTEXT (branch point)\n" + _filler(TRUNK_TOKENS, "trunk")


def _post_completion(prompt: str) -> dict:
    body = json.dumps({
        "model": MODEL, "prompt": prompt,
        "max_tokens": DECODE_TOKENS, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    latency = time.monotonic() - t0
    usage = resp.get("usage", {}) or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "cached_tokens": (details.get("cached_tokens") if details else None),
        "completion_tokens": usage.get("completion_tokens", 0),
        "latency_s": latency,
    }


def _scrape_metrics() -> dict:
    """Pull prefix-cache + preemption counters from vLLM's /metrics."""
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=30) as r:
            text = r.read().decode()
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, val = line.partition(" ")
        base = name.split("{")[0]
        if base in (
            "vllm:gpu_prefix_cache_hit_rate",
            "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_hits_total",
            "vllm:num_preemptions_total",
        ):
            try:
                out[base] = out.get(base, 0.0) + float(val)
            except ValueError:
                pass
    return out


def _run_branch(k: int) -> list[dict]:
    calls = []
    suffix = ""
    for r in range(ROUNDS):
        suffix += _filler(SUFFIX_TOKENS, f"branch{k}_round{r}")
        prompt = f"{TRUNK}\n\n## BRANCH {k} — divergent work, step {r}\n{suffix}\nContinue the fix:\n"
        c = _post_completion(prompt)
        c.update(branch=k, round=r)
        calls.append(c)
    return calls


def main() -> int:
    print(f"=== fork-eviction: K={K}, trunk≈{TRUNK_TOKENS}tok, rounds={ROUNDS}, "
          f"decode={DECODE_TOKENS}, suffix/round≈{SUFFIX_TOKENS}, conc={CONCURRENCY} ===")
    # Warm the trunk once (paid by both sides; Pie via fork, vLLM via first prefill).
    print("[warm] priming trunk cache...")
    _post_completion(TRUNK + "\nWarm.\n")
    m0 = _scrape_metrics()

    t0 = time.monotonic()
    all_calls: list[dict] = []
    if CONCURRENCY <= 1:
        for k in range(K):
            all_calls.extend(_run_branch(k))
    else:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            for calls in ex.map(_run_branch, range(K)):
                all_calls.extend(calls)
    wall = time.monotonic() - t0
    m1 = _scrape_metrics()

    have_cached = any(c["cached_tokens"] is not None for c in all_calls)
    # A "trunk re-prefill" = a call whose cached prefix fell below 90% of the
    # trunk ⇒ APC had to recompute (part of) the shared trunk.
    thresh = int(TRUNK_TOKENS * 0.9)
    trunk_reprefills = sum(
        1 for c in all_calls
        if c["cached_tokens"] is not None and c["cached_tokens"] < thresh
    )
    first_calls = [c for c in all_calls if c["round"] == 0]
    first_call_reprefills = sum(
        1 for c in first_calls
        if c["cached_tokens"] is not None and c["cached_tokens"] < thresh
    )
    total_prompt = sum(c["prompt_tokens"] for c in all_calls)
    total_cached = sum((c["cached_tokens"] or 0) for c in all_calls)
    real_prefill = total_prompt - total_cached
    # Pie-fork counterfactual: trunk prefilled ONCE; every branch call reuses the
    # forked KV for the trunk ⇒ 0 trunk re-prefills, real prefill = new tokens only
    # (prompt_tokens - trunk_tokens per call, i.e. the divergent suffix/decode).
    pie_real_prefill = sum(max(0, c["prompt_tokens"] - TRUNK_TOKENS) for c in all_calls)

    def _pre(m):
        q = m.get("vllm:gpu_prefix_cache_queries_total", m.get("vllm:prefix_cache_queries_total", 0))
        h = m.get("vllm:gpu_prefix_cache_hits_total", m.get("vllm:prefix_cache_hits_total", 0))
        return q, h

    q0, h0 = _pre(m0); q1, h1 = _pre(m1)
    dq, dh = q1 - q0, h1 - h0
    hit_rate = (dh / dq) if dq else None
    preempt = m1.get("vllm:num_preemptions_total", 0) - m0.get("vllm:num_preemptions_total", 0)

    result = {
        "config": {"K": K, "trunk_tokens": TRUNK_TOKENS, "rounds": ROUNDS,
                   "decode_tokens": DECODE_TOKENS, "suffix_tokens": SUFFIX_TOKENS,
                   "concurrency": CONCURRENCY, "model": MODEL},
        "calls": len(all_calls), "wall_s": round(wall, 2),
        "have_cached_tokens": have_cached,
        "trunk_reprefills": trunk_reprefills,
        "first_call_trunk_reprefills": first_call_reprefills,
        "first_calls": len(first_calls),
        "total_prompt_tokens": total_prompt, "total_cached_tokens": total_cached,
        "vllm_real_prefill_tokens": real_prefill,
        "pie_fork_real_prefill_tokens": pie_real_prefill,
        "prefill_ratio_vllm_over_pie": (round(real_prefill / pie_real_prefill, 2)
                                        if pie_real_prefill else None),
        "prefix_cache_hit_rate_delta": (round(hit_rate, 4) if hit_rate is not None else None),
        "preemptions_delta": preempt,
    }
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    print(f"\nInterpretation: vLLM+APC re-prefilled the shared trunk on "
          f"{trunk_reprefills}/{len(all_calls)} calls "
          f"({first_call_reprefills}/{len(first_calls)} branch first-calls). "
          f"Pie fork = 0 by construction. vLLM real prefill {real_prefill} vs "
          f"Pie-fork {pie_real_prefill} tokens"
          + (f" ({result['prefill_ratio_vllm_over_pie']}x)." if result['prefill_ratio_vllm_over_pie'] else "."))
    if not have_cached:
        print("WARNING: server did not return prompt_tokens_details.cached_tokens — "
              "rely on prefix_cache_hit_rate_delta + preemptions_delta instead.")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
