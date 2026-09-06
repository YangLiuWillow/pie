#!/usr/bin/env python3
"""
wan22_parity.py -- drive pie's `wan22-mini-*` rows against the M3 miniature golden.

The reference and its dump come from `wan22_golden.py --mini` (see README).
This script is the other half: it turns the golden's *inputs* into the case
JSON the `wan2-parity` inferlet takes, runs it, turns its JSON answer back
into an `.npz` under the golden's own key names, and diffs the two with
`compare.py`.

    # 1. the case the inferlet reads (scalar timestep, or TI2V's per-token one)
    python wan22_parity.py case  --out /tmp/wan22-parity
    python wan22_parity.py case  --out /tmp/wan22-parity --pertoken

    # 2. run it (the config's `[model] model` is the imported artifact:
    #    `pie model import <dir with wan22_mini_d128.safetensors> --sku wan22-mini-d128-bf16-kv-bf16`)
    python wan22_parity.py run   --out /tmp/wan22-parity --config ~/.pie/config.wan22-mini.toml

    # 3. the pie-side npz, then the diff
    python wan22_parity.py collect --out /tmp/wan22-parity
    python wan22_parity.py compare --out /tmp/wan22-parity

    # or all four
    python wan22_parity.py all --out /tmp/wan22-parity [--pertoken]

Nothing here imports torch: the reference numbers are already on disk, and
patchify is a reshape.

THE GUEST'S SHAPE
-----------------
One step is a context lane plus one video lane (scalar timestep) or two
video lanes (`--pertoken`: the first latent frame's tokens at timestep 0,
the rest at `t` -- `wan_2`'s per-lane form of TI2V's per-token timestep),
all in one attention group, each on ITS OWN pipeline.

LAYOUTS
-------
Patch rows are `(c, ph, pw)`, the transformer's `patch_embedding` input
order, and the family hands its velocity back in the same order (`proj_out`
is permuted at import), so `unpatchify` here is the inverse of `patchify`.
Token order is `t * (H/2) * (W/2) + h * (W/2) + w`; positions are `(t, h, w)`
in patch units.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

DEFAULT_GOLDEN = os.path.join(
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "wan22"
)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# The golden is fp32 (`wan22_golden.py --mini` runs on CPU in float32); pie
# runs the same weights cast to bf16 with bf16 activations: the bf16 drift of
# a two-block trunk, the same order as mini-dit's gate (README §3).
TOLERANCES = ["--tol", "0.1", "--rel-tol", "0.02", "--cos-tol", "0.9999"]


# ----------------------------------------------------------------------------
# patchify / unpatchify -- (1, 2, 2) patches, feature order (c, ph, pw)
# ----------------------------------------------------------------------------

def patchify(latent: np.ndarray, p: int) -> np.ndarray:
    """[B,C,T,H,W] -> [B, T*(H/p)*(W/p), C*p*p], feature order (c, ph, pw)."""
    b, c, t, h, w = latent.shape
    x = latent.reshape(b, c, t, h // p, p, w // p, p)
    x = x.transpose(0, 2, 3, 5, 1, 4, 6)
    return np.ascontiguousarray(x.reshape(b, t * (h // p) * (w // p), c * p * p))


def unpatchify(tokens: np.ndarray, c: int, t: int, h: int, w: int, p: int) -> np.ndarray:
    b = tokens.shape[0]
    x = tokens.reshape(b, t, h // p, w // p, c, p, p)
    x = x.transpose(0, 4, 1, 2, 5, 3, 6)
    return np.ascontiguousarray(x.reshape(b, c, t, h, w))


def positions(t: int, hp: int, wp: int) -> np.ndarray:
    """`(t, h, w)` per token, in token order."""
    tt, hh, ww = np.meshgrid(np.arange(t), np.arange(hp), np.arange(wp), indexing="ij")
    return np.stack([tt, hh, ww], axis=-1).reshape(-1, 3).astype(np.float32)


# ----------------------------------------------------------------------------
# the case
# ----------------------------------------------------------------------------

def suffix(args) -> str:
    return "_pertoken" if args.pertoken else ""


def numbered(out: str, stem: str, tail: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(stem)}{re.escape(tail)}_(\d+)\.json$")
    found = []
    for name in os.listdir(out):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(out, name)))
    return [path for _, path in sorted(found)]


def config(golden: str) -> dict:
    with open(os.path.join(golden, "wan22_mini_config.json")) as f:
        return json.load(f)


def cases(args) -> list[str]:
    """Write one `case[_pertoken]_{b}.json` per batch element; answer their paths."""
    cfg = config(args.golden)
    dump = np.load(os.path.join(args.golden, "wan22_mini.npz"))
    v = args.variant
    p = cfg["variants"][v]["config"]["patch_size"]
    assert p == [1, 2, 2], p
    hs = dump[f"mini.{v}.in.hidden_states"]           # [B, C, T, H, W]
    ctx = dump[f"mini.{v}.in.encoder_hidden_states"]  # [B, L, text_dim]
    ts = dump[f"mini.{v}.in.timestep"]                # [B]
    b, c, t, h, w = hs.shape
    tokens = patchify(hs, 2)
    pos = positions(t, h // 2, w // 2)
    assert tokens.shape[1] == pos.shape[0]

    os.makedirs(args.out, exist_ok=True)
    written = []
    for i in range(b):
        cond_rows = 0
        timestep = float(ts[i])
        if args.pertoken:
            pt = dump[f"mini.{v}.in.timestep_pertoken"][i]   # [S]
            zeros = np.flatnonzero(pt == 0.0)
            cond_rows = int(zeros.size)
            assert np.array_equal(zeros, np.arange(cond_rows)), "the zeros are a prefix"
            rest = pt[cond_rows:]
            assert rest.size and np.all(rest == rest[0]), "one timestep past the prefix"
            timestep = float(rest[0])
        case = {
            "latents": tokens[i].reshape(-1).astype(np.float32).tolist(),
            "rows": int(tokens.shape[1]),
            "patch_features": int(tokens.shape[2]),
            "cond_rows": cond_rows,
            "context": ctx[i].reshape(-1).astype(np.float32).tolist(),
            "context_rows": int(ctx.shape[1]),
            "context_width": int(ctx.shape[2]),
            "positions": pos.reshape(-1).tolist(),
            "timestep": timestep,
        }
        path = os.path.join(args.out, f"case{suffix(args)}_{i}.json")
        with open(path, "w") as f:
            json.dump(case, f)
        written.append(path)
    print(f"[case] {len(written)} batch element(s), {tokens.shape[1]} rows "
          f"({'cond ' + str(cond_rows) + ' + rest' if args.pertoken else 'one lane'}) -> {args.out}")
    return written


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

def wasm(inferlet: str) -> str:
    """The newest `.wasm` a build left for `inferlet`, building one first."""
    name = os.path.basename(os.path.normpath(inferlet))
    stem = name.replace("-", "_")
    workspace = os.path.dirname(os.path.normpath(inferlet))
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        done = subprocess.run(
            ["cargo", "build", "-p", name, "--target", "wasm32-wasip2"],
            cwd=workspace, capture_output=True, text=True,
        )
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            raise SystemExit(f"building {name} for wasm32-wasip2 failed")
    candidates = [
        os.path.join(workspace, "target/wasm32-wasip2/release", f"{stem}.wasm"),
        os.path.join(workspace, "target/wasm32-wasip2/debug", f"{stem}.wasm"),
    ]
    present = [path for path in candidates if os.path.exists(path)]
    if not present:
        raise SystemExit(f"no wasm for {name}; tried {', '.join(candidates)}")
    return max(present, key=os.path.getmtime)


def run(args) -> None:
    paths = numbered(args.out, "case", suffix(args))
    if not paths:
        raise SystemExit(f"{args.out}: no case JSON; run `case` first")
    pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")
    if not os.path.exists(pie):
        raise SystemExit(
            f"{pie}: no pie binary. Build one with "
            f"`cargo build -p pie --features cuda`, or pass --pie."
        )
    binary = wasm(args.inferlet)
    manifest = os.path.join(args.inferlet, "Pie.toml")
    for b, case in enumerate(paths):
        out = os.path.join(args.out, f"pie{suffix(args)}_{b}.json")
        cmd = [pie]
        if args.config:
            cmd += ["--config", args.config]
        cmd += ["run", "--path", binary, "--manifest", manifest, "--"]
        text = ""
        if args.case_file:
            cmd += ["--case_file", os.path.basename(case)]
        else:
            text = open(case).read()
            n = 8
            step = -(-len(text) // n)
            for i in range(n):
                cmd += [f"--case_{i}", text[i * step:(i + 1) * step]]
        print(f"[run] {' '.join(cmd[:8])} ... ({len(text)} bytes of case)")
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        with open(out[:-5] + ".stderr", "w") as f:
            f.write(done.stderr)
        if done.returncode != 0:
            sys.stderr.write(done.stdout)
            sys.stderr.write(done.stderr)
            raise SystemExit(f"pie run failed ({done.returncode})")
        with open(out, "w") as f:
            f.write(done.stdout)
        print(f"[run] batch {b} -> {out}")


# ----------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------

def document(path: str) -> dict:
    """`pie run` prints a human header before the document; take the JSON."""
    lines = [line for line in open(path).read().splitlines() if line.startswith("{")]
    if not lines:
        raise SystemExit(f"{path}: no JSON document (did the run fail?)")
    doc = json.loads(lines[-1])
    if "result" in doc and isinstance(doc["result"], (dict, str)):
        doc = doc["result"]
    if isinstance(doc, str):
        doc = json.loads(doc)
    return doc


def collect(args) -> str:
    v = args.variant
    dump = np.load(os.path.join(args.golden, "wan22_mini.npz"))
    hs = dump[f"mini.{v}.in.hidden_states"]
    _, c, t, h, w = hs.shape
    paths = numbered(args.out, "pie", suffix(args))
    if not paths:
        raise SystemExit(f"{args.out}: no pie answer; run `run` first")
    docs = [document(path) for path in paths]
    rows, width = docs[0]["rows"], docs[0]["patch_features"]
    tokens = np.stack(
        [np.asarray(doc["velocity"], dtype=np.float32).reshape(rows, width) for doc in docs]
    )
    key = f"mini.{v}.{'out_pertoken' if args.pertoken else 'out'}.0"
    mine = unpatchify(tokens, c, t, h, w, 2)
    path = os.path.join(args.out, f"wan22_mini_pie{suffix(args)}.npz")
    np.savez(path, **{key: mine})
    theirs = dump[key][: len(docs)]
    target = os.path.join(args.out, f"wan22_mini_target{suffix(args)}.npz")
    np.savez(target, **{key: theirs.astype(np.float32)})
    print(f"[collect] {mine.shape} from {len(docs)} batch element(s) -> {path}; golden -> {target}")
    return path


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------

def compare(args) -> int:
    mine = os.path.join(args.out, f"wan22_mini_pie{suffix(args)}.npz")
    theirs = os.path.join(args.out, f"wan22_mini_target{suffix(args)}.npz")
    for path in (mine, theirs):
        if not os.path.exists(path):
            raise SystemExit(f"{path}: missing; run `collect` first")
    cmd = [
        sys.executable, os.path.join(HERE, "compare.py"), mine, theirs,
        "--sort-by", "rel", *TOLERANCES,
    ]
    print(f"[compare] {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["case", "run", "collect", "compare", "all"])
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--out", default="/tmp/wan22-parity")
    ap.add_argument("--variant", choices=["d128", "nano"], default="d128")
    ap.add_argument("--pertoken", action="store_true",
                    help="the TI2V per-token-timestep forward: two video lanes")
    ap.add_argument("--inferlet", default=os.path.join(REPO, "tests/inferlets/wan2-parity"))
    ap.add_argument("--config", default=None,
                    help="the serving config; its `[model] model` must be the imported miniature")
    ap.add_argument("--pie", default=None, help="the pie binary (default: PATH, else target/debug)")
    ap.add_argument("--case_file", action="store_true",
                    help="pass the case as a scratch file instead of argv pieces")
    args = ap.parse_args()

    if args.cmd == "case":
        cases(args)
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "collect":
        collect(args)
    elif args.cmd == "compare":
        return compare(args)
    else:
        cases(args)
        run(args)
        collect(args)
        return compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
