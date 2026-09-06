#!/usr/bin/env python3
"""
compare.py -- diff two .npz tensor dumps produced by mini_dit_ref.py / *_golden.py
(or by a pie parity inferlet that writes the same keys).

    python compare.py A.npz B.npz
    python compare.py A.npz B.npz --tol 2e-2 --rel-tol 5e-3 --cos-tol 0.9999
    python compare.py A.npz B.npz --keys 'b0.*' --keys velocity --sort-by rel
    python compare.py A.npz B.npz --quiet          # only failures + summary

Per tensor it prints max-abs error, relative error (||a-b|| / ||b||), max relative
elementwise error over the elements above --floor, and cosine similarity.
Exit status is 1 if any gate fails or the key sets disagree (unless --allow-missing).
"""

from __future__ import annotations

import argparse
import fnmatch
import sys

import numpy as np


def load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: np.asarray(z[k], dtype=np.float64) for k in z.files}


def stats(a: np.ndarray, b: np.ndarray, floor: float) -> dict:
    d = a - b
    max_abs = float(np.abs(d).max()) if d.size else 0.0
    nb = float(np.linalg.norm(b))
    rel = float(np.linalg.norm(d) / nb) if nb > 0 else (0.0 if max_abs == 0 else float("inf"))
    m = np.abs(b) > floor
    max_rel_el = float((np.abs(d[m]) / np.abs(b[m])).max()) if m.any() else 0.0
    af, bf = a.ravel(), b.ravel()
    na, nbv = np.linalg.norm(af), np.linalg.norm(bf)
    cos = float(af @ bf / (na * nbv)) if na > 0 and nbv > 0 else 1.0
    return dict(max_abs=max_abs, rel=rel, max_rel_el=max_rel_el, cos=cos,
                shape=a.shape, n=a.size,
                amax=float(np.abs(a).max()) if a.size else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--keys", action="append", default=None,
                    help="glob(s) selecting which keys to compare (repeatable)")
    ap.add_argument("--tol", type=float, default=None, help="gate on max-abs error")
    ap.add_argument("--rel-tol", type=float, default=None, help="gate on ||a-b||/||b||")
    ap.add_argument("--cos-tol", type=float, default=None, help="gate on cosine similarity (min)")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="absolute floor: ignore |B| below this in max-rel-el")
    ap.add_argument("--floor-frac", type=float, default=1e-3,
                    help="relative floor: also ignore |B| below floor_frac * max|B| "
                         "(keeps max-rel-el from being dominated by near-zero elements)")
    ap.add_argument("--sort-by", choices=["key", "abs", "rel", "cos"], default="key")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    ka, kb = set(A), set(B)
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    keys = sorted(ka & kb)
    if args.keys:
        keys = [k for k in keys if any(fnmatch.fnmatch(k, g) for g in args.keys)]

    print(f"A = {args.a}   ({len(A)} tensors)")
    print(f"B = {args.b}   ({len(B)} tensors)")
    if only_a:
        print(f"  only in A ({len(only_a)}): {', '.join(only_a[:12])}{' ...' if len(only_a) > 12 else ''}")
    if only_b:
        print(f"  only in B ({len(only_b)}): {', '.join(only_b[:12])}{' ...' if len(only_b) > 12 else ''}")

    rows, bad_shape = [], []
    for k in keys:
        if A[k].shape != B[k].shape:
            bad_shape.append((k, A[k].shape, B[k].shape))
            continue
        fl = max(args.floor, args.floor_frac * float(np.abs(B[k]).max()))
        rows.append((k, stats(A[k], B[k], fl)))

    order = {"key": lambda r: r[0], "abs": lambda r: -r[1]["max_abs"],
             "rel": lambda r: -r[1]["rel"], "cos": lambda r: r[1]["cos"]}[args.sort_by]
    rows.sort(key=order)

    fails = []
    hdr = f"{'tensor':<38} {'shape':<20} {'|a|max':>10} {'max-abs':>11} {'rel':>11} {'max-rel-el':>11} {'cos':>12}"
    if not args.quiet:
        print("\n" + hdr); print("-" * len(hdr))
    for k, s in rows:
        f = []
        if args.tol is not None and s["max_abs"] > args.tol: f.append("ABS")
        if args.rel_tol is not None and s["rel"] > args.rel_tol: f.append("REL")
        if args.cos_tol is not None and s["cos"] < args.cos_tol: f.append("COS")
        if f: fails.append((k, s, f))
        if not args.quiet or f:
            mark = ("  FAIL:" + ",".join(f)) if f else ""
            print(f"{k:<38} {str(s['shape']):<20} {s['amax']:10.4g} {s['max_abs']:11.4g} "
                  f"{s['rel']:11.4g} {s['max_rel_el']:11.4g} {s['cos']:12.9f}{mark}")

    print()
    if rows:
        print(f"compared {len(rows)} tensors   "
              f"worst max-abs {max(r[1]['max_abs'] for r in rows):.4g}   "
              f"worst rel {max(r[1]['rel'] for r in rows):.4g}   "
              f"worst cos {min(r[1]['cos'] for r in rows):.9f}")
    for k, sa, sb in bad_shape:
        print(f"SHAPE MISMATCH {k}: {sa} vs {sb}")

    rc = 0
    if fails:
        print(f"FAILED {len(fails)} / {len(rows)} tensors"); rc = 1
    if bad_shape:
        rc = 1
    if (only_a or only_b) and not args.allow_missing:
        print("FAILED: key sets differ (pass --allow-missing to ignore)"); rc = 1
    if rc == 0:
        print("PASS")
    return rc


if __name__ == "__main__":
    sys.exit(main())
