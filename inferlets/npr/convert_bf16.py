"""Cast a fp32 safetensors checkpoint to bf16 for pie's weight loaders.

The published NPR-4B is fp32 (FSDP-merged) and pie does not cast at load time,
so serving it directly fails with
`gemm_act_x_w: unsupported dtype combo (act=bf16, w=fp32, y=bf16)`.

    python convert_bf16.py <snapshot-dir> <out-dir>
    python convert_bf16.py            # resolve the NPR-4B snapshot from the HF cache

Non-tensor files (tokenizer, chat template, added_tokens) are copied verbatim;
`config.json:torch_dtype` and the index's `total_size` are rewritten to match.
"""

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

DEFAULT_REPO = "bigai-NPR/NPR-4B"


def default_snapshot() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(DEFAULT_REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", nargs="?", help="resolved HF snapshot dir (fp32)")
    ap.add_argument("out", nargs="?", help="destination dir for the bf16 copy")
    args = ap.parse_args()

    snap = args.snapshot or default_snapshot()
    out = args.out or (os.path.expanduser("~") + "/NPR-4B-bf16")
    if os.path.abspath(snap) == os.path.abspath(out):
        sys.exit("refusing to convert in place: snapshot and out are the same dir")
    os.makedirs(out, exist_ok=True)
    print(f"{snap}\n  -> {out}", flush=True)

    total = 0
    for f in sorted(os.listdir(snap)):
        src = os.path.join(snap, f)
        if f.endswith(".safetensors"):
            sd = load_file(src)
            sd = {
                k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v)
                for k, v in sd.items()
            }
            total += sum(v.numel() * v.element_size() for v in sd.values())
            save_file(sd, os.path.join(out, f), metadata={"format": "pt"})
            print("converted", f, flush=True)
        elif os.path.isfile(src):
            shutil.copy(src, os.path.join(out, f))

    cfgp = os.path.join(out, "config.json")
    cfg = json.load(open(cfgp))
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(cfgp, "w"), indent=2)

    idxp = os.path.join(out, "model.safetensors.index.json")
    if os.path.exists(idxp):
        idx = json.load(open(idxp))
        idx.setdefault("metadata", {})["total_size"] = total
        json.dump(idx, open(idxp, "w"))
    print("DONE total bytes:", total)


if __name__ == "__main__":
    main()
