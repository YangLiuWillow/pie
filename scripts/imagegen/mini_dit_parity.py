#!/usr/bin/env python3
"""
mini_dit_parity.py -- drive pie's `mini-dit` row against the M0 golden.

The reference and its dumps come from `mini_dit_ref.py` (see README §3).  This
script is the other half: it turns the golden's *inputs* into the case JSON the
`mini-dit-parity` inferlet takes, runs it, turns its JSON answer back into an
`.npz` with the golden's own key names, and diffs the two with `compare.py`.

    # 1. the case the inferlet reads (one file per batch element)
    python mini_dit_parity.py case  --out /tmp/mini-dit-parity
    python mini_dit_parity.py case  --out /tmp/mini-dit-parity --euler

    # 2. run it (the config's `[model] model` is the imported artifact; the
    #    case files must be reachable as /scratch/<name> inside the sandbox)
    python mini_dit_parity.py run   --out /tmp/mini-dit-parity --config ~/.pie/config.toml

    # 3. the pie-side npz, then the diff
    python mini_dit_parity.py collect --out /tmp/mini-dit-parity
    python mini_dit_parity.py compare --out /tmp/mini-dit-parity

    # or all four
    python mini_dit_parity.py all --out /tmp/mini-dit-parity

Nothing here imports torch: the reference numbers are already on disk, and
patchify is a reshape.

WHAT THIS CANNOT DO YET
-----------------------
The guest side is complete: the `reading` / `input` / `stream` / `group` verbs
and the `velocity()` intrinsic have landed.  What `run` still needs is the
CUDA dispatch arms for `attention.ragged`, `layout.{pack,unpack}_rows` and
`elementwise.{modulate,gated_residual_add,sinusoid,silu,rope_axes}`, which
`engine-cuda` refuses by name today.  `case`, `collect` and `compare` are
useful on their own the moment a pie-side answer exists, however it was
produced, and `run` says what failed rather than failing obscurely.
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
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "mini-dit"
)
DEFAULT_SKU = "mini-dit-bf16-kv-bf16"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# The bf16 drift the reference measured on these inputs (README §3): worst
# max-abs 0.088, worst relative 9.7e-3, worst cosine 0.99995.  A pie-side bf16
# run is a different bf16 run, so the gate is the same order and no tighter.
TOLERANCES = ["--tol", "0.1", "--rel-tol", "0.02", "--cos-tol", "0.9999"]


# ----------------------------------------------------------------------------
# patchify / unpatchify -- the reference's exact index algebra, in numpy
# ----------------------------------------------------------------------------

def patchify(latent: np.ndarray, p: int) -> np.ndarray:
    """[B,C,H,W] -> [B, (H/p)*(W/p), C*p*p], feature order (c, ph, pw)."""
    b, c, hs, ws = latent.shape
    x = latent.reshape(b, c, hs // p, p, ws // p, p)
    x = x.transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(x.reshape(b, (hs // p) * (ws // p), c * p * p))


def unpatchify(tokens: np.ndarray, c: int, hs: int, ws: int, p: int) -> np.ndarray:
    b = tokens.shape[0]
    x = tokens.reshape(b, hs // p, ws // p, c, p, p)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    return np.ascontiguousarray(x.reshape(b, c, hs, ws))


# ----------------------------------------------------------------------------
# the case
# ----------------------------------------------------------------------------

def numbered(out: str, stem: str, euler: bool) -> list[str]:
    """`<out>/<stem>[_euler]_<b>.json` for every batch element `b`, in order.

    Matched by pattern and not by prefix: `pie_euler_0.json` starts with
    `pie_`, so a prefix test folds the four-step run's files into the
    one-step run's and stacks a batch of four against a golden of two.
    """
    pattern = re.compile(rf"^{re.escape(stem)}{'_euler' if euler else ''}_(\d+)\.json$")
    found = []
    for name in os.listdir(out):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(out, name)))
    return [path for _, path in sorted(found)]


def config(golden: str) -> dict:
    with open(os.path.join(golden, "config.json")) as f:
        return json.load(f)


def cases(args) -> list[str]:
    """Write one `case_{b}.json` per batch element; answer their paths."""
    cfg = config(args.golden)
    dump = np.load(os.path.join(args.golden, f"mini_dit_dump_{args.policy}.npz"))
    p = cfg["patch_size"]
    c, hs, ws = cfg["latent_shape"]

    text = dump["in.text"]
    context = dump["in.context"]
    timestep = dump["in.timestep"]
    txt_pos = dump["in.txt_pos"]
    img_pos = dump["in.img_pos"]
    # The reference dumps `patches` already; recomputing it from `in.latent`
    # is what proves this file's index algebra is the reference's.
    patches = patchify(dump["in.latent"], p)
    assert np.array_equal(patches, dump["patches"]), "patchify disagrees with the reference"

    if args.euler:
        euler = np.load(os.path.join(args.golden, f"mini_dit_euler_{args.policy}.npz"))
        latents = patchify(euler["euler.x_init"], p)
        sigmas = euler["euler.sigmas"].tolist()
    else:
        latents = patches
        sigmas = []

    os.makedirs(args.out, exist_ok=True)
    written = []
    for b in range(latents.shape[0]):
        case = {
            "latents": latents[b].reshape(-1).astype(np.float32).tolist(),
            "image_rows": int(latents.shape[1]),
            "patch_features": int(latents.shape[2]),
            "text": text[b].reshape(-1).astype(np.float32).tolist(),
            "text_rows": int(text.shape[1]),
            "text_width": int(text.shape[2]),
            "context": context[b].reshape(-1).astype(np.float32).tolist(),
            "context_rows": int(context.shape[1]),
            "context_width": int(context.shape[2]),
            "text_positions": txt_pos.reshape(-1).astype(np.float32).tolist(),
            "image_positions": img_pos.reshape(-1).astype(np.float32).tolist(),
            "timestep": float(timestep[b]),
            "sigmas": [float(s) for s in sigmas],
            "t_scale": float(cfg["euler_t_scale"]),
            "steps": int(cfg["euler_steps"]) if args.euler else 0,
        }
        path = os.path.join(args.out, f"case{'_euler' if args.euler else ''}_{b}.json")
        with open(path, "w") as f:
            json.dump(case, f)
        written.append(path)
    print(f"[case] {len(written)} batch element(s), {latents.shape[1]} image rows -> {args.out}")
    return written


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

def wasm(inferlet: str) -> str:
    """The newest `.wasm` a build left for `inferlet`, building one first.

    Newest wins, not first, for `tests/inferlets/conftest.py`'s reason: a
    release artifact built once by hand would otherwise shadow every debug
    rebuild afterwards, silently.
    """
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
        os.path.join(inferlet, "target/wasm32-wasip2/release", f"{stem}.wasm"),
        os.path.join(inferlet, "target/wasm32-wasip2/debug", f"{stem}.wasm"),
    ]
    present = [path for path in candidates if os.path.exists(path)]
    if not present:
        raise SystemExit(f"no wasm for {name}; tried {', '.join(candidates)}")
    return max(present, key=os.path.getmtime)


def run(args) -> None:
    paths = numbered(args.out, "case", args.euler)
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
    # `pie run` takes no `--model`: the row it serves comes from the config,
    # which has to point `[model] model` at the artifact this row imported
    # (`pie model import ... --sku mini-dit-bf16-kv-bf16`). The sandbox needs
    # `allow_fs` and a scratch dir holding the case files, since a case is
    # far past one command-line argument.
    for b, case in enumerate(paths):
        out = os.path.join(args.out, f"pie{'_euler' if args.euler else ''}_{b}.json")
        cmd = [pie]
        if args.config:
            cmd += ["--config", args.config]
        cmd += ["run", "--path", binary, "--manifest", manifest, "--"]
        if args.case_file:
            # the case as a file under the sandbox's per-process scratch dir
            # (needs `[sandbox] allow_fs` and the file placed there by hand)
            cmd += ["--case_file", os.path.basename(case)]
        else:
            # the case as eight argv pieces (`case_0..7`), each well under the
            # kernel's 128 KiB single-argument ceiling; no sandbox fs needed
            text = open(case).read()
            n = 8
            step = -(-len(text) // n)
            for i in range(n):
                cmd += [f"--case_{i}", text[i * step:(i + 1) * step]]
        if args.euler:
            cmd += ["--euler", "true"]
        print(f"[run] {' '.join(cmd)}")
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        if done.returncode != 0:
            sys.stderr.write(done.stdout)
            sys.stderr.write(done.stderr)
            raise SystemExit(
                f"pie run failed ({done.returncode}). If it refused an op by name, the "
                f"engine arms for that op are not wired yet — see this file's header."
            )
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
    cfg = config(args.golden)
    p = cfg["patch_size"]
    c, hs, ws = cfg["latent_shape"]
    paths = numbered(args.out, "pie", args.euler)
    if not paths:
        raise SystemExit(f"{args.out}: no pie answer; run `run` first")
    docs = [document(path) for path in paths]
    rows, width = docs[0]["image_rows"], docs[0]["patch_features"]

    def stack(key: str, at: int | None = None) -> np.ndarray:
        tokens = np.stack(
            [
                np.asarray(doc[key] if at is None else doc[key][at], dtype=np.float32)
                .reshape(rows, width)
                for doc in docs
            ]
        )
        return unpatchify(tokens, c, hs, ws, p)

    out: dict[str, np.ndarray] = {}
    if args.euler:
        steps = len(docs[0]["euler_v"])
        for i in range(steps):
            out[f"euler.v{i}"] = stack("euler_v", i)
            out[f"euler.x{i + 1}"] = stack("euler_x", i)
        out["euler.latent"] = out[f"euler.x{steps}"]
    else:
        out["velocity"] = stack("velocity")
        # The head's own rectangle too, so a mismatch says whether it is the
        # arithmetic or the unpatchify.
        out["final.tokens"] = np.stack(
            [np.asarray(doc["velocity"], dtype=np.float32).reshape(rows, width) for doc in docs]
        )

    path = os.path.join(args.out, f"mini_dit_pie{'_euler' if args.euler else ''}.npz")
    np.savez(path, **out)
    print(f"[collect] {len(out)} tensors from {len(docs)} batch element(s) -> {path}")
    return path


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------

def compare(args) -> int:
    mine = os.path.join(args.out, f"mini_dit_pie{'_euler' if args.euler else ''}.npz")
    if not os.path.exists(mine):
        raise SystemExit(f"{mine}: no pie npz; run `collect` first")
    theirs = os.path.join(
        args.golden,
        f"mini_dit_{'euler' if args.euler else 'dump'}_{args.policy}.npz",
    )
    cmd = [
        sys.executable, os.path.join(HERE, "compare.py"), mine, theirs,
        "--allow-missing", "--sort-by", "rel", *TOLERANCES,
    ]
    if args.keys:
        for glob in args.keys:
            cmd += ["--keys", glob]
    print(f"[compare] {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["case", "run", "collect", "compare", "all"])
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--out", default="/tmp/mini-dit-parity")
    ap.add_argument("--policy", choices=["bf16", "fp32"], default="bf16",
                    help="which emulation of the reference to diff against")
    ap.add_argument("--euler", action="store_true",
                    help="the four-step schedule instead of one step")
    ap.add_argument("--inferlet", default=os.path.join(REPO, "tests/inferlets/mini-dit-parity"))
    ap.add_argument("--config", default=None,
                    help=f"the serving config; its `[model] model` must be the artifact "
                         f"`{DEFAULT_SKU}` imported")
    ap.add_argument("--pie", default=None, help="the pie binary (default: PATH, else target/debug)")
    ap.add_argument("--case_file", action="store_true", help="pass the case as a scratch file instead of argv pieces")
    ap.add_argument("--keys", action="append", default=None)
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
