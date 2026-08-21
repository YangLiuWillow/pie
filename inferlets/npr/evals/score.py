"""Score an NPR eval sweep: avg@k / pass@k plus parallelism and throughput stats.

Answer matching follows NPR's `evals/evaluate.py` semantics restricted to what
AIME needs — the gold labels are integers 0-999, so normalization (strip LaTeX
wrappers, separators, trailing zeros) plus a numeric compare is exact; there is
no need for the symbolic `math_equal` fallback here. A run whose trajectory has
no `\\boxed{}` scores as wrong and is counted separately as a format failure.

Usage:
    python score.py results/aime25.jsonl
    python score.py results/*.jsonl --by-problem
"""

import argparse
import collections
import json
import math
import pathlib
import re
import statistics
import sys

_STRIP = [
    (re.compile(r"\\(?:left|right|!|,|;|:|\s)"), ""),
    (re.compile(r"\\text(?:bf|rm|it)?\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\$"), ""),
    (re.compile(r"\s+"), ""),
]


def normalize(ans) -> str | None:
    if ans is None:
        return None
    s = str(ans).strip()
    for pat, rep in _STRIP:
        s = pat.sub(rep, s)
    s = s.rstrip(".").replace(",", "")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s or None


def equal(pred, gold) -> bool:
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        return math.isclose(float(p), float(g), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return False


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else float("nan")


# Stop reasons that mean "ran out of global token budget" rather than "the
# model stopped on its own". Mirrors `is_budget_stop` in src/lib.rs.
#
# `budget` / `branch_budget` are the pre-split labels: still unambiguously
# budget exhaustion, but silent about WHICH cap bound. The `_positional` /
# `_charge` pair separates the engine's primary positional check from the
# secondary x-degree charge — they bind at different times (runs exhaust
# positionally at charge ratios as low as 0.517), which is why a charge-ratio
# proxy misread 15 of 46 stranded runs.
BUDGET_STOPS = {
    "budget",
    "branch_budget",
    "budget_positional",
    "budget_charge",
    "branch_budget_positional",
    "branch_budget_charge",
}

# Which cap bound, for rows that say. None = the row predates the cause split.
BUDGET_CAUSE = {
    "budget_positional": "positional",
    "branch_budget_positional": "positional",
    "budget_charge": "charge",
    "branch_budget_charge": "charge",
}


def budget_cause(r: dict) -> str | None:
    """Which budget cap ended this run, or None if the row cannot say.

    Returns None for pre-split rows (`budget` / `branch_budget`) as well as for
    non-budget stops — the same refusal-to-guess as `budget_exhausted`. Do not
    substitute a charge-ratio estimate here; that is the proxy this split exists
    to replace.
    """
    return BUDGET_CAUSE.get(r.get("stop_reason") or "")


def budget_exhausted(r: dict) -> bool | None:
    """Did this run stop because it ran out of tokens?

    Prefers the inferlet's own `budget_exhausted` flag; falls back to the
    `stop_reason` label. Returns None for records written before the split,
    whose `branch_terminal` conflates EOS with budget exhaustion and so
    cannot answer the question at all.
    """
    if r.get("budget_exhausted") is not None:
        return bool(r["budget_exhausted"])
    stop = r.get("stop_reason")
    if stop is None:
        return None
    if stop == "branch_terminal":
        return None  # pre-split label: ambiguous by construction
    return stop in BUDGET_STOPS


def report(records: list[dict], by_problem: bool) -> None:
    arms = sorted({r["arm"] for r in records})
    print(
        f"{'arm':<11} {'n':>4} {'avg@k':>7} {'pass@k':>7} {'fmt':>6} {'bud':>6} "
        f"{'err':>5} {'par%':>6} {'brnch':>6} {'gen tok':>8} {'chg tok':>8} {'wall s':>7} {'tok/s':>7}"
    )
    print("-" * 103)
    for arm in arms:
        rs = [r for r in records if r["arm"] == arm]
        per_problem: dict[str, list[bool]] = collections.defaultdict(list)
        for r in rs:
            per_problem[r["problem_id"]].append(equal(r.get("answer"), r["gold"]))
        avg_at_k = mean([statistics.fmean(v) for v in per_problem.values()])
        pass_at_k = mean([1.0 if any(v) else 0.0 for v in per_problem.values()])
        fmt_fail = mean([1.0 if r.get("answer") in (None, "") else 0.0 for r in rs])
        bud = mean([1.0 if budget_exhausted(r) else 0.0 for r in rs if budget_exhausted(r) is not None])
        errs = sum(1 for r in rs if r.get("error"))
        good = [r for r in rs if not r.get("error")]
        par = mean([1.0 if (r.get("parallel_blocks") or 0) > 0 else 0.0 for r in good])
        branches = mean([r.get("branches_total") for r in good])
        gen = mean([r.get("tokens_generated") for r in good])
        chg = mean([r.get("tokens_charged") for r in good])
        wall = mean([(r.get("elapsed_ms") or 0) / 1000 for r in good])
        toks = mean(
            [
                (r["tokens_generated"] / (r["elapsed_ms"] / 1000))
                for r in good
                if r.get("elapsed_ms") and r.get("tokens_generated")
            ]
        )
        print(
            f"{arm:<11} {len(rs):>4} {avg_at_k:>7.3f} {pass_at_k:>7.3f} {fmt_fail:>6.3f} {bud:>6.3f} "
            f"{errs:>5} {par:>6.3f} {branches:>6.1f} {gen:>8.0f} {chg:>8.0f} {wall:>7.1f} {toks:>7.1f}"
        )
    print(
        "\navg@k = mean per-problem accuracy over k samples; pass@k = any-correct."
        "\nfmt = share with no \\boxed{} (the 'unanswered' column); par% = share that forked at least once."
        "\nbud = share that ran out of token budget (stop_reason budget/branch_budget);"
        "\n      nan on pre-split results, whose branch_terminal conflates EOS with exhaustion."
        "\nchg tok = ledger charge (branch tokens x parallel degree, NPR accounting)."
        "\nwall/tok-s are only comparable across arms when the sweep ran at concurrency 1."
    )

    # Why the unanswered runs stopped. `fmt` alone cannot distinguish a run
    # that reasoned to the end and never boxed an answer from one that was cut
    # off mid-thought; the stop_reason split is what separates them, and it is
    # the lever on the 20-26% unanswered rate the parallel arms carry.
    # Which budget cap bound, across every run that says. Pre-registered in
    # evals/README.md: a column that never varies is a result here, not a dud —
    # "only positional" means the x-degree charge never binds in this workload.
    causes = collections.Counter(
        budget_cause(r) for r in records if not r.get("error") and budget_cause(r)
    )
    unknown = sum(
        1
        for r in records
        if not r.get("error") and budget_exhausted(r) and not budget_cause(r)
    )
    if causes or unknown:
        detail = "  ".join(f"{k}={v}" for k, v in causes.most_common())
        if unknown:
            detail += f"  cause-unknown(pre-split)={unknown}"
        print(f"\nbudget exhaustion by cap: {detail}")

    print("\nunanswered runs (no \\boxed{}) by stop_reason")
    print(f"{'arm':<11} {'n':>4}  reasons")
    for arm in arms:
        unans = [
            r
            for r in records
            if r["arm"] == arm and not r.get("error") and r.get("answer") in (None, "")
        ]
        if not unans:
            print(f"{arm:<11} {0:>4}  -")
            continue
        counts = collections.Counter(r.get("stop_reason") or "unknown" for r in unans)
        detail = "  ".join(f"{k}={v}" for k, v in counts.most_common())
        print(f"{arm:<11} {len(unans):>4}  {detail}")

    if by_problem:
        print("\nper-problem accuracy")
        pids = sorted({r["problem_id"] for r in records})
        print(f"{'problem':<12} " + " ".join(f"{a:>10}" for a in arms) + "   gold")
        for pid in pids:
            cells = []
            gold = ""
            for arm in arms:
                rs = [r for r in records if r["arm"] == arm and r["problem_id"] == pid]
                gold = rs[0]["gold"] if rs else gold
                cells.append(
                    f"{statistics.fmean([equal(r.get('answer'), r['gold']) for r in rs]):>10.2f}"
                    if rs
                    else f"{'-':>10}"
                )
            print(f"{pid:<12} " + " ".join(cells) + f"   {gold}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by-problem", action="store_true")
    args = ap.parse_args()

    records: dict[str, dict] = {}
    for f in args.files:
        for line in pathlib.Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                records[r["key"]] = r  # later lines win (retried runs)
    if not records:
        print("no records", file=sys.stderr)
        sys.exit(1)
    report(list(records.values()), args.by_problem)


if __name__ == "__main__":
    main()
