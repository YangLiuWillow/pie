#!/usr/bin/env python3
"""Print an A/B of two `inferletbench.run_summary.v2` files as TWO tables.

Two tables, not one, on purpose. An agent's wall clock is model time plus tool
time — running pytest, editing files, waiting on a build — so a single
tokens-per-second number lets a task that happens to trigger a slow test suite
masquerade as a slow serving engine. And a serving change that makes turns
cheaper can still make the AGENT worse, because a resumed turn computes its
delta with different kernel shapes and greedy decoding can flip on that.

So: serving quantities on one side, task outcomes on the other, and no combined
score. If one improves and the other regresses, that is the finding, and a
blended number would hide it.

  python3 ab_report.py <baseline.json> <candidate.json>
"""
import json
import sys


def get(summary, key, default=None):
    return summary.get("metrics", {}).get(key, default)


def row(label, a, b, fmt="{}", better=None):
    va, vb = fmt.format(a) if a is not None else "—", fmt.format(b) if b is not None else "—"
    mark = ""
    if better and isinstance(a, (int, float)) and isinstance(b, (int, float)) and a != b:
        improved = (b > a) if better == "up" else (b < a)
        mark = "  ✓" if improved else "  ✗"
    print(f"  {label:<34} {va:>12} {vb:>12}{mark}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a = json.loads(open(sys.argv[1]).read())
    b = json.loads(open(sys.argv[2]).read())
    la = a.get("label", sys.argv[1])
    lb = b.get("label", sys.argv[2])

    print(f"\n{'':36}{la:>12} {lb:>12}\n")
    print("SERVING — what the engine did with the work it was given")
    row("prefix reuse", get(a, "prompt_reuse_fraction"), get(b, "prompt_reuse_fraction"),
        "{:.1%}", "up")
    row("model call latency p50 (s)", get(a, "model_call_latency_p50_s"),
        get(b, "model_call_latency_p50_s"), "{:.1f}", "down")
    row("model call latency p95 (s)", get(a, "model_call_latency_p95_s"),
        get(b, "model_call_latency_p95_s"), "{:.1f}", "down")
    row("in-call prompt tok/s", get(a, "in_call_prompt_tok_s"),
        get(b, "in_call_prompt_tok_s"), "{}")
    row("model time (s)", get(a, "model_time_s"), get(b, "model_time_s"), "{}", "down")

    print("\nTASK — whether the agent actually got the job done")
    row("resolved", get(a, "resolved"), get(b, "resolved"), "{}", "up")
    row("non-empty patch rate", get(a, "nonempty_patch_rate"),
        get(b, "nonempty_patch_rate"), "{:.0%}", "up")
    row("model calls (mean/case)", get(a, "mean_model_calls"),
        get(b, "mean_model_calls"), "{:.1f}")
    row("agent-error cases", get(a, "agent_error_cases"), get(b, "agent_error_cases"),
        "{}", "down")
    row("wall-time-exhausted cases", get(a, "wall_time_exhausted_cases"),
        get(b, "wall_time_exhausted_cases"), "{}", "down")

    if get(a, "resolved") is None or get(b, "resolved") is None:
        print("\n  NOTE: `resolved` is null — the official grader has not run, so the")
        print("  TASK table is patch PRODUCTION, not task resolution. A non-empty")
        print("  patch is a weaker claim and can move independently of correctness.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
