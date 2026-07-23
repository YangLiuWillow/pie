"""Aggregate ab_bench.jsonl files into a comparison table.

Usage: python ab_bench_report.py logs/ab_bench_*.jsonl
Groups per-batch records by (arm, N, concurrency) and reports median per-batch
wallclock plus reuse metrics (Pie: shared_prefill_saved, kv_avoided; vLLM: APC
cache-hit-rate = cached/prompt). The concurrency sweep is the headline: watch
vLLM's cache-hit-rate fall and per-batch latency climb while Pie stays flat.
"""
import glob
import json
import statistics
import sys
from collections import defaultdict


def median(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return statistics.median(xs) if xs else float("nan")


def main(paths):
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    # key -> list of per-batch dicts (with wallclock + telemetry)
    cells = defaultdict(list)
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["arm"], rec["N"], rec["concurrency"])
                for b in rec.get("per_batch", []):
                    cells[key].append(b)

    print(f"\n{'arm':14s} {'N':>2s} {'conc':>4s} {'batches':>7s} "
          f"{'med_batch_s':>11s} {'saved_tok':>9s} {'kv_avoid':>8s} "
          f"{'apc_hit%':>8s} {'prefill_recomp':>14s}")
    print("-" * 96)
    for key in sorted(cells):
        arm, N, conc = key
        bs = cells[key]
        wall = median([b["wallclock_s"] for b in bs])
        saved = kv_avoid = apc = recomp = float("nan")
        if arm == "pie_batch":
            saved = median([(b.get("pie") or {}).get("shared_prefill_saved", 0) for b in bs])
            kv_avoid = median([(b.get("pie") or {}).get("kv_avoided", 0) or 0 for b in bs])
            recomp = median([
                ((b.get("pie") or {}).get("l1p_prefill_tokens", 0) or 0)
                + ((b.get("pie") or {}).get("l1g_decode_tokens", 0) or 0)
                + ((b.get("pie") or {}).get("leaves_prefill_tokens", 0) or 0)
                for b in bs])
        else:
            hits, prompts, recomps = [], [], []
            for b in bs:
                v = b.get("vllm") or {}
                pr = v.get("phase1_prompt", 0) + v.get("phase2_prompt", 0)
                ca = v.get("phase1_cached", 0) + v.get("phase2_cached", 0)
                de = v.get("phase1_completion", 0) + v.get("phase2_completion", 0)
                if pr:
                    hits.append(100.0 * ca / pr)
                prompts.append(pr)
                recomps.append((pr - ca) + de)  # uncached prefill + all decode
            apc = median(hits) if hits else float("nan")
            recomp = median(recomps)
        def f(x, w=9, p=0):
            return ("{:>%d.%df}" % (w, p)).format(x) if x == x else " " * (w - 3) + "n/a"
        print(f"{arm:14s} {N:>2d} {conc:>4d} {len(bs):>7d} "
              f"{f(wall,11,2)} {f(saved)} {f(kv_avoid,8)} {f(apc,8,1)} {f(recomp,14)}")
    print("\nNotes: med_batch_s = median wallclock to produce N children for one parent.")
    print("  saved_tok/kv_avoid: Pie KV-token compute avoided by fork/snapshot reuse.")
    print("  apc_hit%: vLLM prompt tokens served from prefix cache (higher=more reuse).")
    print("  prefill_recomp: KV-token compute actually done (lower=better). Watch it rise")
    print("  with concurrency for vLLM (APC eviction) but stay flat for Pie.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["logs/ab_bench_*.jsonl"])
