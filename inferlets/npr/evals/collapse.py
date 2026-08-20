"""Per-arm collapse split: the metric the repetition-penalty A/B actually turns on.

HANDOVER.md §11 read 5 / §14: adopt runs that reach >=2 parallel blocks are
~80% correct; runs stranded at <=1 block are ~5% and account for essentially
the whole 20-26% unanswered rate. So an arm's mean accuracy is a mixture of two
very different populations, and a penalty that helps by un-stranding
trajectories shows up as a *shift in the block mix*, not necessarily as a
uniform accuracy lift. This prints both halves so the two effects can be told
apart:

    share  = P(blocks>=2)          -- did the penalty stop trajectories looping
                                      inside their first block?
    acc|.. = P(correct | blocks)   -- conditional quality within each stratum

A penalty that raises `avg` purely by raising `share`, with the conditionals
flat, is doing exactly the job §15 predicts. A penalty that moves the
conditionals is doing something else, and that difference is the finding.

Usage:
    python collapse.py results/aime25-pen-ab.jsonl
    python collapse.py results/aime25-pen-ab.jsonl --baseline adopt_nopen
"""

import argparse
import collections
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score import equal  # noqa: E402  -- same answer-matching semantics as score.py


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Binomial, so it stays sane at k=0 or k=n, which a
    normal-approximation interval does not -- and the blocks<=1 stratum is
    usually near 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def frac(xs, pred) -> tuple[int, int]:
    k = sum(1 for x in xs if pred(x))
    return k, len(xs)


def fmt(k: int, n: int) -> str:
    if n == 0:
        return f"{'--':>17}"
    lo, hi = wilson(k, n)
    return f"{k / n:>5.3f} [{lo:.2f},{hi:.2f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument(
        "--baseline",
        default="adopt_nopen",
        help="arm to difference the others against (default: the penalty-off arm)",
    )
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

    by = collections.defaultdict(list)
    for r in records.values():
        by[r["arm"]].append(r)

    ok = lambda r: equal(r.get("answer"), r["gold"])  # noqa: E731
    blocks = lambda r: r.get("parallel_blocks") or 0  # noqa: E731

    print(
        f"{'arm':<13} {'n':>4} {'avg':>17} {'unans':>17} "
        f"{'share b>=2':>17} {'acc|b<=1':>17} {'acc|b>=2':>17}"
    )
    print("-" * 108)
    stats = {}
    for arm in sorted(by):
        rs = by[arm]
        lo = [r for r in rs if blocks(r) <= 1]
        hi = [r for r in rs if blocks(r) >= 2]
        a = frac(rs, ok)
        u = frac(rs, lambda r: r.get("answer") in (None, ""))
        s = (len(hi), len(rs))
        cl = frac(lo, ok)
        ch = frac(hi, ok)
        stats[arm] = {"avg": a, "unans": u, "share": s, "lo": cl, "hi": ch}
        print(
            f"{arm:<13} {len(rs):>4} {fmt(*a)} {fmt(*u)} {fmt(*s)} {fmt(*cl)} {fmt(*ch)}"
        )

    base = args.baseline
    if base in stats and len(stats) > 1:
        print(f"\ndeltas vs {base} (percentage points):")
        b = stats[base]
        for arm, s in stats.items():
            if arm == base:
                continue
            d = {
                k: 100 * ((s[k][0] / s[k][1] if s[k][1] else float("nan"))
                          - (b[k][0] / b[k][1] if b[k][1] else float("nan")))
                for k in ("avg", "unans", "share", "lo", "hi")
            }
            print(
                f"  {arm:<13} avg {d['avg']:+6.1f}  unans {d['unans']:+6.1f}  "
                f"share(b>=2) {d['share']:+6.1f}  acc|b<=1 {d['lo']:+6.1f}  "
                f"acc|b>=2 {d['hi']:+6.1f}"
            )

    print(
        "\nIntervals are Wilson 95% on the run count (k=2 per problem, so runs are"
        "\nnot independent across a problem pair -- read them as scale, not as a test)."
        "\nAt n=50 per arm a 10-point avg move is roughly the resolution limit."
    )

    # Token spend by stratum. §14's decomposition said the quality gap is token
    # starvation under the x-degree ledger, and §15's theory of the penalty is
    # that it shortens repetition-inflated branches. If that is the mechanism,
    # the penalty arm spends fewer generated tokens per run, not just more of
    # them well.
    print("\nmean tokens_generated / charged (runs without an error):")
    for arm in sorted(by):
        good = [r for r in by[arm] if not r.get("error")]
        for label, sel in (("b<=1", lambda r: blocks(r) <= 1), ("b>=2", lambda r: blocks(r) >= 2)):
            xs = [r for r in good if sel(r)]
            g = [r.get("tokens_generated") for r in xs if r.get("tokens_generated")]
            c = [r.get("tokens_charged") for r in xs if r.get("tokens_charged")]
            gm = sum(g) / len(g) if g else float("nan")
            cm = sum(c) / len(c) if c else float("nan")
            print(f"  {arm:<13} {label}  n={len(xs):>3}  gen={gm:>8.0f}  charged={cm:>8.0f}")

    # stop_reason x blocks, the mechanism behind the collapse
    print("\nstop_reason x parallel_blocks (count, correct):")
    for arm in sorted(by):
        tally = collections.Counter()
        good = collections.Counter()
        for r in by[arm]:
            cell = (r.get("stop_reason") or "?", "b>=2" if blocks(r) >= 2 else "b<=1")
            tally[cell] += 1
            good[cell] += ok(r)
        cells = ", ".join(
            f"{sr}/{bk}: {n} ({good[(sr, bk)]} ok)" for (sr, bk), n in sorted(tally.items())
        )
        print(f"  {arm:<13} {cells}")


if __name__ == "__main__":
    main()
