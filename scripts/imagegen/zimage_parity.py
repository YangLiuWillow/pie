#!/usr/bin/env python3
"""
zimage_parity.py -- drive pie's `z-image` rows against the diffusers goldens.

The reference and its dumps come from `zimage_golden.py` (README §4).  This
script is the other half: it turns a golden's *inputs* into the case JSON the
`zimage-parity` inferlet takes, runs it, turns its JSON answer back into an
`.npz` under the golden's own key names, and diffs the two with `compare.py`.

Two modes, one per row:

    --mini  (default)  the miniature `z-image-mini-bf16-kv-bf16` row against
                       `zimage_mini.npz` (one forward of the random-init
                       transformer at dim 256: `mini.in.*` -> `mini.out.0`)
    --pad              the same row against `zimage_mini_pad.npz`: rows that
                       NEED padding (48 -> 64 image rows, 40 -> 64 caption
                       rows) at a pipeline-realistic `t = 0.5`
    --turbo            the flagship `z-image-turbo-bf16-kv-bf16` row against
                       `zimage_golden.npz`'s step-0 transformer call
                       (`dit.step0.in.*` -> `dit.step0.out.0`; the prompt
                       embeds are the golden's own, so the `text` reading is
                       not exercised here)
    --text             the flagship's `text` reading alone: the golden's prompt
                       through the family template -> `prompt_embeds.0`
    --chain            the flagship end to end, text -> refine -> denoise,
                       against the same `dit.step0.out.0` (and the embeds)

    python zimage_parity.py case    --out /tmp/zimage-parity
    python zimage_parity.py run     --out /tmp/zimage-parity --config ~/.pie/config.zimage-mini.toml
    python zimage_parity.py collect --out /tmp/zimage-parity
    python zimage_parity.py compare --out /tmp/zimage-parity
    python zimage_parity.py all     --out /tmp/zimage-parity [--turbo]

`run` needs the config's `[model] model` to be the row's artifact (`pie model
import <golden dir> --sku z-image-mini-bf16-kv-bf16`, or the Z-Image-Turbo
snapshot under `--sku z-image-turbo-bf16-kv-bf16`), and — for a case too large
for argv — `[sandbox] allow_fs = true` with `fs_scratch_dir` equal to `--out`,
since the case is then read as `/scratch/<name>` inside the sandbox.

Nothing here imports torch: the reference numbers are already on disk, and
patchify is a reshape.

WHAT THE HARNESS CONVERTS
-------------------------
* patchify is diffusers' `_patchify_image` exactly (feature order
  `(pf, ph, pw, c)`, NOT mini-dit's `(c, ph, pw)`), and its inverse;
* the caption is padded to a multiple of 32 rows and flagged; caption row `j`
  sits at rotary `(1 + j, 0, 0)` — pads included, which is what diffusers'
  `_pad_with_ids` does for a caption (its grid spans the padded length);
  image patch `(a, b)` sits at `(L32 + 1, a, b)`, image pads at `(0, 0, 0)`;
* the golden holds the TRANSFORMER's `t` (multiplied by `t_scale = 1000`
  inside the model); the family's port takes the SCHEDULER timestep and flips
  it itself (`u = 1000 - t`), so the case carries `1000 - 1000 * t`;
* the family hands back `-model_out` (the pipeline's `noise_pred = -noise_pred`
  folded in); the golden is the raw model output, so `collect` negates back.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

DEFAULT_GOLDEN = os.path.join(
    os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden"), "z-image"
)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SEQ_MULTIPLE = 32
PATCH = 2
T_SCALE = 1000.0
T_FLIP = 1000.0

# The gates. The mini row is a 6-block bf16 run against an fp32 CPU reference:
# the same order as the mini-dit gate. The Turbo row is 34 bf16 blocks against
# a bf16 CUDA reference whose reductions differ from ours: looser.
TOLERANCES = {
    "mini": ["--tol", "0.1", "--rel-tol", "0.02", "--cos-tol", "0.9999"],
    "mini_pad": ["--tol", "0.1", "--rel-tol", "0.02", "--cos-tol", "0.9999"],
    "turbo": ["--tol", "1.0", "--rel-tol", "0.05", "--cos-tol", "0.999"],
    # the encoder's rows are O(10..100) wide in value; the gate is the cosine
    "text": ["--rel-tol", "0.05", "--cos-tol", "0.999"],
    "chain": ["--rel-tol", "0.05", "--cos-tol", "0.999"],
}
PROMPT_KEY = "prompt_embeds.0"

def keyed(file: str, prefix: str, out: str, sku: str, refined: str | None = None) -> dict:
    """A mode: the golden file, its input keys (`<prefix>x.0`, `<prefix>cap.0`,
    `<prefix>t`), the output key the velocity lands under, the row's SKU and —
    for a tapped golden — the key the refined caption lands under."""
    return dict(file=file, x=f"{prefix}x.0", cap=f"{prefix}cap.0", t=f"{prefix}t",
                out=out, sku=sku, refined=refined)


MINI_SKU = "z-image-mini-bf16-kv-bf16"
TURBO_SKU = "z-image-turbo-bf16-kv-bf16"
MODES = {
    "mini": keyed("zimage_mini.npz", "mini.in.", "mini.out.0", MINI_SKU),
    # the same weights over rows that NEED padding (16 image pads, 24 caption
    # pads) at a pipeline-realistic t: `zimage_golden.py --mini-pad`
    "mini_pad": keyed("zimage_mini_pad.npz", "mini_pad.in.", "mini_pad.out.0", MINI_SKU),
    "turbo": keyed("zimage_golden.npz", "dit.step0.in.", "dit.step0.out.0", TURBO_SKU),
    # the Turbo row's `text` reading alone (the golden's prompt through the
    # family template -> `prompt_embeds.0`), and the whole chain text ->
    # refine -> denoise against the same step-0 velocity
    "text": keyed("zimage_golden.npz", "dit.step0.in.", PROMPT_KEY, TURBO_SKU),
    "chain": keyed("zimage_golden.npz", "dit.step0.in.", "dit.step0.out.0", TURBO_SKU),
    # the tapped goldens (`zimage_golden.py --taps`): the Turbo transformer
    # alone over the step-0 inputs (`full`) and over their 256-row crop
    # (`crop`), with the refined caption to diff first
    "full": keyed("zimage_taps.npz", "taps.full.in.", "taps.full.out.0", TURBO_SKU,
                  refined="taps.full.cap.refined"),
    "crop": keyed("zimage_taps.npz", "taps.crop.in.", "taps.crop.out.0", TURBO_SKU,
                  refined="taps.crop.cap.refined"),
    "tiny": keyed("zimage_taps.npz", "taps.tiny.in.", "taps.tiny.out.0", TURBO_SKU,
                  refined="taps.tiny.cap.refined"),
}
# the pipeline golden's step-0 keys are `arg0.0` / `arg2.0` / `arg1`, not `x` / `cap` / `t`
for _mode in ("turbo", "text", "chain"):
    MODES[_mode].update(x="dit.step0.in.arg0.0", cap="dit.step0.in.arg2.0", t="dit.step0.in.arg1")
TOLERANCES.update(full=TOLERANCES["turbo"], crop=TOLERANCES["turbo"], tiny=TOLERANCES["turbo"])


def floats(arr: np.ndarray) -> list[str]:
    """Every value as the shortest decimal that reads back to the same f32:
    a third of `json.dump`'s double repr, which is what keeps the Turbo case
    inside a few dozen argv pieces."""
    return [np.format_float_positional(v, unique=True, trim="-") for v in arr.astype(np.float32).ravel()]


def dump_case(doc: dict, path: str) -> None:
    """`json.dump` with the float lists written by `floats`."""
    parts = []
    for key, value in doc.items():
        if isinstance(value, np.ndarray):
            parts.append(f'"{key}":[{",".join(floats(value))}]')
        else:
            parts.append(f'"{key}":{json.dumps(value)}')
    with open(path, "w") as f:
        f.write("{" + ",".join(parts) + "}")


# ----------------------------------------------------------------------------
# patchify / unpatchify -- diffusers' `ZImageTransformer2DModel` index algebra
# ----------------------------------------------------------------------------

def patchify(image: np.ndarray, p: int = PATCH, pf: int = 1) -> tuple[np.ndarray, tuple]:
    """`[C, F, H, W]` -> `[(F/pf)(H/p)(W/p), pf*p*p*C]`, feature order (pf, ph, pw, c)."""
    c, f, h, w = image.shape
    ft, ht, wt = f // pf, h // p, w // p
    x = image.reshape(c, ft, pf, ht, p, wt, p)
    x = x.transpose(1, 3, 5, 2, 4, 6, 0)
    return np.ascontiguousarray(x.reshape(ft * ht * wt, pf * p * p * c)), (ft, ht, wt)


def unpatchify(tokens: np.ndarray, c: int, f: int, h: int, w: int, p: int = PATCH, pf: int = 1) -> np.ndarray:
    ft, ht, wt = f // pf, h // p, w // p
    n = ft * ht * wt
    x = tokens[:n].reshape(ft, ht, wt, pf, p, p, c)
    x = x.transpose(6, 0, 3, 1, 4, 2, 5)
    return np.ascontiguousarray(x.reshape(c, f, h, w))


def padded(n: int) -> int:
    return n + (-n) % SEQ_MULTIPLE


# ----------------------------------------------------------------------------
# the case
# ----------------------------------------------------------------------------

def mode_of(args) -> str:
    if args.mode:
        return args.mode
    if args.text:
        return "text"
    if args.chain:
        return "chain"
    if args.turbo:
        return "turbo"
    return "mini_pad" if args.pad else "mini"


def golden_of(args) -> np.lib.npyio.NpzFile:
    return np.load(os.path.join(args.golden, MODES[mode_of(args)]["file"]))


def case(args) -> str:
    mode = mode_of(args)
    keys = MODES[mode]
    dump = golden_of(args)
    image = dump[keys["x"]].astype(np.float32)          # [C, F, H, W]
    caption = dump[keys["cap"]].astype(np.float32)      # [L, cap_width]
    t_model = float(np.asarray(dump[keys["t"]]).reshape(-1)[0])

    # the caption: padded rows (diffusers repeats the last row; the plan
    # overwrites them with `cap_pad_token` anyway), flags, positions
    l_real = caption.shape[0]
    l32 = padded(l_real)
    cap_rows = np.concatenate([caption, np.repeat(caption[-1:], l32 - l_real, axis=0)], axis=0)
    cap_pad = np.zeros(l32, np.float32)
    cap_pad[l_real:] = 1.0
    cap_pos = np.zeros((l32, 3), np.float32)
    cap_pos[:, 0] = 1.0 + np.arange(l32)

    # the image: patch rows, padded, flags, positions `(L32 + 1 + f, a, b)`
    patches, (ft, ht, wt) = patchify(image)
    n_real = patches.shape[0]
    n32 = padded(n_real)
    img_rows = np.concatenate([patches, np.repeat(patches[-1:], n32 - n_real, axis=0)], axis=0)
    img_pad = np.zeros(n32, np.float32)
    img_pad[n_real:] = 1.0
    grid = np.stack(np.meshgrid(np.arange(ft), np.arange(ht), np.arange(wt), indexing="ij"), axis=-1)
    img_pos = np.zeros((n32, 3), np.float32)
    img_pos[:n_real] = grid.reshape(-1, 3).astype(np.float32)
    img_pos[:n_real, 0] += l32 + 1

    timestep = T_FLIP - T_SCALE * t_model
    doc = {
        "latents": img_rows,
        "image_rows": int(n32),
        "patch_features": int(img_rows.shape[1]),
        "image_pad": img_pad,
        "image_positions": img_pos,
        "caption": cap_rows,
        "caption_rows": int(l32),
        "caption_width": int(cap_rows.shape[1]),
        "caption_pad": cap_pad,
        "caption_positions": cap_pos,
        "timestep": float(timestep),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"case_{mode}.json")
    dump_case(doc, path)
    print(f"[case] {mode}: {n_real}->{n32} image rows x {img_rows.shape[1]}, "
          f"{l_real}->{l32} caption rows x {cap_rows.shape[1]}, t_model {t_model} -> "
          f"port {timestep} ({os.path.getsize(path)} bytes) -> {path}")
    return path


def prompt_of(args) -> str:
    """The prompt the full golden was rendered from (`MANIFEST.json`)."""
    with open(os.path.join(args.golden, "MANIFEST.json")) as f:
        return json.load(f)["prompt"]


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
        os.path.join(inferlet, "target/wasm32-wasip2/release", f"{stem}.wasm"),
        os.path.join(inferlet, "target/wasm32-wasip2/debug", f"{stem}.wasm"),
    ]
    present = [path for path in candidates if os.path.exists(path)]
    if not present:
        raise SystemExit(f"no wasm for {name}; tried {', '.join(candidates)}")
    return max(present, key=os.path.getmtime)


# The case travels as argv pieces `case_0..N`, each under the kernel's 128 KiB
# single-argument ceiling, up to the count the guest's Pie.toml declares. The
# total is past the default 2 MiB `ARG_MAX` for the Turbo case, so the child
# runs under an unlimited stack rlimit (Linux sizes argv at a quarter of it).
PIECE = 120 * 1024
MAX_PIECES = 32


def unlimited_stack() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def run(args) -> None:
    mode = mode_of(args)
    pie = args.pie or shutil.which("pie") or os.path.join(REPO, "target/debug/pie")
    if not os.path.exists(pie):
        raise SystemExit(f"{pie}: no pie binary. Build one with `cargo build -p pie --features cuda`, or pass --pie.")
    binary = wasm(args.inferlet)
    manifest = os.path.join(args.inferlet, "Pie.toml")
    cmd = [pie]
    if args.config:
        cmd += ["--config", args.config]
    cmd += ["run", "--path", binary, "--manifest", manifest, "--"]
    if mode == "text":
        cmd += ["--prompt", prompt_of(args), "--text_only", "true"]
    else:
        path = os.path.join(args.out, f"case_{mode}.json")
        if not os.path.exists(path):
            raise SystemExit(f"{path}: no case JSON; run `case` first")
        text = open(path).read()
        if args.case_file:
            # the case as a file under the sandbox's PER-PROCESS scratch dir
            # (`[sandbox] allow_fs = true`; the dir is created at process start
            # and deleted at exit, so this needs a hand on the other side)
            cmd += ["--case_file", os.path.basename(path)]
        else:
            # No piece may start with `-`: `pie run` would read it as the
            # next flag and hand the guest `true`. A cut that lands before a
            # minus sign slides past it (the concatenation is unchanged).
            pieces, at = [], 0
            while at < len(text):
                cut = min(at + PIECE, len(text))
                while cut < len(text) and text[cut] == "-":
                    cut += 1
                pieces.append(text[at:cut])
                at = cut
            if len(pieces) > MAX_PIECES:
                raise SystemExit(f"{path}: {len(text)} bytes is more than {MAX_PIECES} pieces of {PIECE}")
            for i, piece in enumerate(pieces):
                cmd += [f"--case_{i}", piece]
        if mode == "chain":
            cmd += ["--prompt", prompt_of(args)]
        if args.refine_only:
            cmd += ["--refine_only", "true"]
        if args.ctx_tap:
            cmd += ["--ctx_tap", "true"]
    out = os.path.join(args.out, f"pie_{mode}.json")
    print(f"[run] {' '.join(c if len(c) < 80 else c[:40] + '...' for c in cmd)}")
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, preexec_fn=unlimited_stack)
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
    mode = mode_of(args)
    keys = MODES[mode]
    path = os.path.join(args.out, f"pie_{mode}.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path}: no pie answer; run `run` first")
    doc = document(path)
    image = golden_of(args)[keys["x"]]
    c, f, h, w = image.shape
    out: dict[str, np.ndarray] = {}
    if doc.get("text"):
        out[PROMPT_KEY] = np.asarray(doc["text"], np.float32).reshape(doc["text_rows"], doc["text_width"])
    if doc["refined"]:
        out["refined"] = np.asarray(doc["refined"], np.float32).reshape(doc["caption_rows"], doc["dim"])
        if keys["refined"]:
            out[keys["refined"]] = out["refined"]
    if doc["velocity"]:
        rows = np.asarray(doc["velocity"], np.float32).reshape(doc["image_rows"], doc["patch_features"])
        out["velocity.rows"] = rows
        # the family's velocity is `-model_out`; the golden is `model_out`
        # (a tapped family reads back an intermediate instead: rows only)
        if rows.shape[1] == c * PATCH * PATCH:
            out[keys["out"]] = unpatchify(-rows, c, f, h, w)
    if doc.get("ctx"):
        rows = len(doc["ctx"]) // max(int(doc["caption_rows"]), 1)
        out["ctx.rows"] = np.asarray(doc["ctx"], np.float32).reshape(doc["caption_rows"], rows)
    npz = os.path.join(args.out, f"zimage_pie_{mode}.npz")
    np.savez(npz, **out)
    print(f"[collect] {len(out)} tensors -> {npz}")
    return npz


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------

def compare(args) -> int:
    mode = mode_of(args)
    mine = os.path.join(args.out, f"zimage_pie_{mode}.npz")
    if not os.path.exists(mine):
        raise SystemExit(f"{mine}: no pie npz; run `collect` first")
    theirs = os.path.join(args.golden, MODES[mode]["file"])
    cmd = [
        sys.executable, os.path.join(HERE, "compare.py"), mine, theirs,
        "--allow-missing", "--sort-by", "rel", *TOLERANCES[mode],
    ]
    keys = args.keys or [MODES[mode]["out"]]
    if not args.keys:
        if mode == "chain":
            keys.append(PROMPT_KEY)
        if MODES[mode]["refined"]:
            keys.append(MODES[mode]["refined"])
    for glob in keys:
        cmd += ["--keys", glob]
    print(f"[compare] {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["case", "run", "collect", "compare", "all"])
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--out", default="/tmp/zimage-parity")
    ap.add_argument("--mode", choices=sorted(MODES), default=None,
                    help="the case by name (the flags below are shortcuts; `full`/`crop` "
                         "are the tapped Turbo goldens of `zimage_golden.py --taps`)")
    ap.add_argument("--mini", action="store_true", help="the miniature row (default)")
    ap.add_argument("--pad", action="store_true",
                    help="the miniature row over rows that need padding (zimage_mini_pad.npz)")
    ap.add_argument("--turbo", action="store_true", help="the Turbo row's step-0 denoise instead")
    ap.add_argument("--text", action="store_true",
                    help="the Turbo row's text reading alone, against prompt_embeds.0")
    ap.add_argument("--chain", action="store_true",
                    help="the Turbo row end to end: text -> refine -> denoise against dit.step0.out.0")
    ap.add_argument("--inferlet", default=os.path.join(REPO, "tests/inferlets/zimage-parity"))
    ap.add_argument("--config", default=None,
                    help="the serving config; its `[model] model` must be the row's artifact")
    ap.add_argument("--pie", default=None, help="the pie binary (default: PATH, else target/debug)")
    ap.add_argument("--case_file", action="store_true",
                    help="pass the case as a scratch file even when it fits argv")
    ap.add_argument("--refine_only", action="store_true", help="stop after the refine reading")
    ap.add_argument("--ctx_tap", action="store_true",
                    help="read the denoise readout off the context lane too (bisect); "
                         "lands as `ctx.rows` in the npz")
    ap.add_argument("--keys", action="append", default=None)
    args = ap.parse_args()

    if args.cmd == "case":
        case(args)
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "collect":
        collect(args)
    elif args.cmd == "compare":
        return compare(args)
    else:
        case(args)
        run(args)
        collect(args)
        return compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
