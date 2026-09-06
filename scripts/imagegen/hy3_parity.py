#!/usr/bin/env python3
"""
hy3_parity.py -- drive pie's `hunyuanimage3-mini` row against the M6 miniature golden.

The reference and its dump come from `hy3_golden.py --mini`. This script is
the other half: it turns the golden's *inputs* into the case JSON the
`hy3-parity` inferlet takes, runs it, turns its JSON answer back into an
`.npz` under the golden's own key names, and diffs the two with `compare.py`.

    # 0. the artifact (once)
    pie model import /root/.cache/pie-imagegen/golden/hy3 \\
        --sku hunyuanimage3-mini-bf16-kv-bf16

    # 1. the case the inferlet reads
    python hy3_parity.py case --out /tmp/hy3-parity

    # 2. run it (give the run its OWN config file and its own [server] port,
    #    and set [engine] max_model_len >= the sequence)
    python hy3_parity.py run --out /tmp/hy3-parity --config ~/.pie/config.hy3-mini.toml

    # 3. the pie-side npz, then the diff
    python hy3_parity.py collect --out /tmp/hy3-parity
    python hy3_parity.py compare --out /tmp/hy3-parity

    # or all four
    python hy3_parity.py all --out /tmp/hy3-parity

Nothing here imports torch.

WHAT IS COMPARED
----------------
`denoise.hidden`   the trunk's rows for the canvas — the `h*w` image rows in
                   raster order and then the `<timestep>` row — with `ln_f`
                   NOT applied, which is what `final_layer` consumes.
`encode.max`       the causal text pass's winning logit per row, and the
                   argmax as an agreement COUNT beside it. (The full
                   `[L, 133120]` plane is 5 MB of JSON, and two token ids do
                   not subtract.)

MEASURED (2026-09-06, `hunyuanimage3-mini-bf16-kv-bf16` on one Blackwell,
fp32 golden vs bf16 weights and bf16 activations):

    denoise.hidden.image        (64, 256)   cos 0.999996   max-abs 0.0013
    denoise.hidden.timestep_row (256,)      cos 0.999993   max-abs 0.00046
    encode.max                  (9,)        cos 0.999998   max-abs 5.8e-05
    prefill argmax                          8 / 9 rows agree

WHAT IS NOT
-----------
The two voxel arms (`image.in` = `patch_embed`, `image.out` = `final_layer`).
The SDK has no channel-fed `Voxels` port and no `pixels()` intrinsic yet, so
the case hands pie the golden's OWN `patch_embed` rows and stops before
`final_layer`. When those two verbs land, the same case grows two more fires
and the comparison reaches the velocity.

THE ROTARY POSITIONS
--------------------
The family's `positions` port is `(y, x')` with `x' = x * theta^(-2/d)`:
`elementwise.rope_axes` gives each axis a contiguous channel block and
restarts the frequency ladder inside it, while the reference interleaves the
two axes over ONE ladder, so `x`'s rung offset is folded into the coordinate.
`crates/models/src/hunyuan_image_3/model.rs::rope_x_scale` is the same number.
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
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "hy3"
)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# The golden runs fp32; pie runs the same weights cast to bf16 with bf16
# activations, through a two-layer MoE trunk whose router is a hard top-k --
# one flipped expert on one row is a big elementwise error and a tiny cosine
# one, which is why the cosine gate carries the weight here.
TOLERANCES = ["--tol", "0.2", "--rel-tol", "0.05", "--cos-tol", "0.999"]


def rope_x_scale(head_dim: int, theta: float = 10000.0) -> float:
    """`theta^(-2/d)` — `model.rs::rope_x_scale`."""
    return float(theta ** (-2.0 / head_dim))


def config(golden: str) -> dict:
    with open(os.path.join(golden, "hy3_mini_config.json")) as f:
        return json.load(f)


def numbered(out: str, stem: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.json$")
    found = []
    for name in os.listdir(out):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(out, name)))
    return [path for _, path in sorted(found)]


# ----------------------------------------------------------------------------
# the case
# ----------------------------------------------------------------------------

def cases(args) -> list[str]:
    cfg = config(args.golden)
    dump = np.load(os.path.join(args.golden, "hy3_mini.npz"))
    layout = cfg["layout"]
    head_dim = int(cfg["config"]["attention_head_dim"])
    hidden = int(cfg["config"]["hidden_size"])
    scale = rope_x_scale(head_dim, float(cfg["config"]["rope_theta"]))

    ids = layout["ids"]
    t_at, img_at, n = layout["timestep_row"], layout["image_row0"], layout["image_rows"]
    prefix = ids[:t_at]
    pos = dump["rope.positions"].astype(np.float64)     # [seq, 2] integer (y, x)

    def scaled(rows: np.ndarray) -> list[float]:
        out = rows.copy()
        out[:, 1] *= scale
        return out.reshape(-1).astype(np.float32).tolist()

    # The canvas lane's row order: the `<timestep>` row, then the image rows
    # in raster order — which is the SEQUENCE's own run, because the CUDA
    # fire path derives a lane's KV positions as `held .. held + rows` and
    # refuses an explicit list.
    canvas_seq = [t_at] + list(range(img_at, img_at + n))
    canvas_pos = pos[canvas_seq]

    case = {
        "prefix": [int(i) for i in prefix],
        "prefix_positions": scaled(pos[:t_at]),
        "img_id": int(ids[img_at]),
        "timestep_id": int(ids[t_at]),
        "image_rows": int(n),
        "canvas_positions": scaled(canvas_pos),
        "canvas_sequence": [int(p) for p in canvas_seq],
        "rows": dump["image_in.rows"].reshape(-1).astype(np.float32).tolist(),
        "hidden": hidden,
        "timestep": float(layout["timestep"]),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "case_0.json")
    with open(path, "w") as f:
        json.dump(case, f)
    print(
        f"[case] prefix {len(prefix)} rows, canvas {n} + 1 rows at hidden {hidden}, "
        f"x-scale {scale:.6f} -> {path}"
    )
    return [path]


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

def wasm(inferlet: str) -> str:
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


def split(text: str, n: int) -> list[str]:
    """`n` pieces, none of which starts with `-`.

    The CLI reads a parameter as `--case_i <value>` and a value that begins
    with `-` is read as the next FLAG, which leaves `case_i` a bare boolean
    and the document unparseable. Nudging each boundary forward past a
    minus sign costs nothing and cannot fail: a JSON number is never all
    minus signs.
    """
    step = -(-len(text) // n)
    cuts = [0]
    for i in range(1, n):
        at = min(i * step, len(text))
        while at < len(text) and text[at] == "-":
            at += 1
        cuts.append(at)
    cuts.append(len(text))
    return [text[a:b] for a, b in zip(cuts, cuts[1:])]


def run(args) -> None:
    paths = numbered(args.out, "case")
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
        out = os.path.join(args.out, f"pie_{b}.json")
        cmd = [pie]
        if args.config:
            cmd += ["--config", args.config]
        cmd += ["run", "--path", binary, "--manifest", manifest, "--"]
        text = ""
        if args.case_file:
            cmd += ["--case_file", os.path.basename(case)]
        else:
            text = open(case).read()
            pieces = split(text, 8)
            for i in range(len(pieces)):
                cmd += [f"--case_{i}", pieces[i]]
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
        print(f"[run] -> {out}")


# ----------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------

def document(path: str) -> dict:
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
    layout = cfg["layout"]
    dump = np.load(os.path.join(args.golden, "hy3_mini.npz"))
    paths = numbered(args.out, "pie")
    if not paths:
        raise SystemExit(f"{args.out}: no pie answer; run `run` first")
    doc = document(paths[0])
    n, hidden = int(doc["image_rows"]), int(doc["hidden"])
    rows = np.asarray(doc["canvas_hidden"], dtype=np.float32).reshape(n + 1, hidden)

    logits = dump["encode.logits"]
    # The argmax is NOT compared as a tensor: it is a token id, and the
    # distance between two ids says nothing (a flipped near-tie is a whole
    # vocabulary apart and a rounding error in the logits). It is reported
    # as an agreement COUNT beside the winning logit, which is comparable.
    mine = {
        "denoise.hidden.image": rows[1:],
        "denoise.hidden.timestep_row": rows[0],
        "encode.max": np.asarray(doc["encode_max"], dtype=np.float32),
    }
    theirs = {
        "denoise.hidden.image": dump["denoise.hidden.image"],
        "denoise.hidden.timestep_row": dump["denoise.hidden.timestep_row"],
        "encode.max": logits.max(axis=-1).astype(np.float32),
    }
    a = os.path.join(args.out, "hy3_mini_pie.npz")
    b = os.path.join(args.out, "hy3_mini_target.npz")
    np.savez(a, **mine)
    np.savez(b, **theirs)
    theirs_argmax = logits.argmax(axis=-1)
    agree = int((np.asarray(doc["encode_argmax"]) == theirs_argmax).sum())
    print(
        f"[collect] canvas {rows.shape} (image rows {n}, one <timestep> row), "
        f"prefill argmax {agree}/{len(theirs_argmax)} agree "
        f"({layout['token_h']}x{layout['token_w']} grid) -> {a}; golden -> {b}"
    )
    return a


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------

def compare(args) -> int:
    mine = os.path.join(args.out, "hy3_mini_pie.npz")
    theirs = os.path.join(args.out, "hy3_mini_target.npz")
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
    ap.add_argument("--out", default="/tmp/hy3-parity")
    ap.add_argument("--inferlet", default=os.path.join(REPO, "tests/inferlets/hy3-parity"))
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
