#!/usr/bin/env python3
"""Shared plumbing for the *_golden.py reference dumps.

Everything lands under $PIE_IMAGEGEN_GOLDEN/<model>/ (default
/root/.cache/pie-imagegen/golden/<model>/) -- never inside the repo.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterator, Tuple

import numpy as np
import torch

GOLDEN_ROOT = os.environ.get("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden")


def outdir(model: str) -> str:
    d = os.path.join(GOLDEN_ROOT, model)
    os.makedirs(d, exist_ok=True)
    return d


class Tap:
    """Collects named tensors, saves one .npz.  Everything upcast to fp32."""

    def __init__(self) -> None:
        self.d: dict[str, np.ndarray] = {}

    def put(self, key: str, val: Any) -> None:
        if isinstance(val, torch.Tensor):
            self.d[key] = val.detach().to(torch.float32).cpu().numpy()
        elif isinstance(val, np.ndarray):
            self.d[key] = val.astype(np.float32) if val.dtype.kind == "f" else val
        elif isinstance(val, (int, float, bool)):
            self.d[key] = np.asarray(val, dtype=np.float32)
        elif isinstance(val, (list, tuple)) and val and isinstance(val[0], (int, float)):
            self.d[key] = np.asarray(val, dtype=np.float32)

    def put_tree(self, prefix: str, obj: Any) -> None:
        for suffix, t in walk(obj):
            self.put(prefix + suffix, t)

    def save(self, path: str) -> str:
        np.savez(path, **self.d)
        print(f"  [npz] {len(self.d):4d} tensors -> {path} ({os.path.getsize(path)/1e6:.1f} MB)")
        return path

    def __len__(self) -> int:
        return len(self.d)


def walk(obj: Any, suffix: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield (dotted-suffix, leaf) for tensors nested in lists/tuples/dicts."""
    if isinstance(obj, (torch.Tensor, np.ndarray, int, float, bool)):
        yield suffix, obj
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, f"{suffix}.{i}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{suffix}.{k}")


def hook_transformer(module: torch.nn.Module, tap: Tap, prefix: str, steps=(0,)):
    """Wrap `module.forward`; on the listed call indices record every tensor argument
    and the return value.  Returns a restore() callable."""
    orig = module.forward
    ctr = {"i": 0}

    def wrapped(*args, **kwargs):
        i = ctr["i"]
        ctr["i"] = i + 1
        out = orig(*args, **kwargs)
        if i in steps:
            p = f"{prefix}.step{i}"
            for j, a in enumerate(args):
                tap.put_tree(f"{p}.in.arg{j}", a)
            for k, v in kwargs.items():
                if k in ("return_dict", "attention_kwargs", "joint_attention_kwargs"):
                    continue
                tap.put_tree(f"{p}.in.{k}", v)
            o = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
            tap.put_tree(f"{p}.out", o)
        return out

    module.forward = wrapped
    return lambda: setattr(module, "forward", orig)


def hook_scheduler(pipe, tap: Tap, prefix: str = "sched"):
    """Record every scheduler.step() output; the last one is the final latent."""
    sched = pipe.scheduler
    orig = sched.step
    ctr = {"i": 0}

    def wrapped(*args, **kwargs):
        i = ctr["i"]; ctr["i"] = i + 1
        out = orig(*args, **kwargs)
        s = out[0] if isinstance(out, tuple) else getattr(out, "prev_sample", out)
        tap.put(f"{prefix}.x{i+1}", s)
        tap.put(f"{prefix}.nsteps", i + 1)
        return out

    sched.step = wrapped
    return lambda: setattr(sched, "step", orig)


def hook_prepare_latents(pipe, tap: Tap, key: str = "noise.init"):
    orig = pipe.prepare_latents
    done = {"v": False}

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        if not done["v"]:
            done["v"] = True
            tap.put_tree(key, out)
        return out

    pipe.prepare_latents = wrapped
    return lambda: setattr(pipe, "prepare_latents", orig)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(d: str, extra: dict | None = None) -> str:
    import diffusers, transformers
    rows = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p) and name != "MANIFEST.json":
            rows.append({"file": name, "bytes": os.path.getsize(p), "md5": md5(p)})
    m = {"dir": d, "torch": torch.__version__, "diffusers": diffusers.__version__,
         "transformers": transformers.__version__, "files": rows}
    if extra:
        m.update(extra)
    path = os.path.join(d, "MANIFEST.json")
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"  [manifest] {path}")
    for r in rows:
        print(f"    {r['file']:<34} {r['bytes']:>12,}  {r['md5']}")
    return path


def npz_keys(tap: Tap, limit: int = 400) -> None:
    for k in sorted(tap.d)[:limit]:
        print(f"    {k:<58} {tap.d[k].shape}")
