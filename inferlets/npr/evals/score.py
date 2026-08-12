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


def report(records: list[dict], by_problem: bool) -> None:
    arms = sorted({r["arm"] for r in records})
    print(
        f"{'arm':<11} {'n':>4} {'avg@k':>7} {'pass@k':>7} {'fmt':>6} "
        f"{'err':>5} {'par%':>6} {'brnch':>6} {'gen tok':>8} {'chg tok':>8} {'wall s':>7} {'tok/s':>7}"
    )
    print("-" * 96)
    for arm in arms:
        rs = [r for r in records if r["arm"] == arm]
        per_problem: dict[str, list[bool]] = collections.defaultdict(list)
        for r in rs:
            per_problem[r["problem_id"]].append(equal(r.get("answer"), r["gold"]))
        avg_at_k = mean([statistics.fmean(v) for v in per_problem.values()])
        pass_at_k = mean([1.0 if any(v) else 0.0 for v in per_problem.values()])
        fmt_fail = mean([1.0 if r.get("answer") in (None, "") else 0.0 for r in rs])
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
            f"{arm:<11} {len(rs):>4} {avg_at_k:>7.3f} {pass_at_k:>7.3f} {fmt_fail:>6.3f} "
            f"{errs:>5} {par:>6.3f} {branches:>6.1f} {gen:>8.0f} {chg:>8.0f} {wall:>7.1f} {toks:>7.1f}"
        )
    print(
        "\navg@k = mean per-problem accuracy over k samples; pass@k = any-correct."
        "\nfmt = share with no \\boxed{}; par% = share that forked at least once."
        "\nchg tok = ledger charge (branch tokens x parallel degree, NPR accounting)."
        "\nwall/tok-s are only comparable across arms when the sweep ran at concurrency 1."
    )

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
