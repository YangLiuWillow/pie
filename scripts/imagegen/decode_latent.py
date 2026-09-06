#!/usr/bin/env python3
"""
decode_latent.py -- turn the `text-to-image` inferlet's raw latent into a PNG.

The generic `text-to-image` guest exits one of two ways (imagegen design D11):
through a `vae.decode` reading, whose pixels the runtime encodes and streams
out without ever entering WASM memory; or -- while no family declares that
reading -- as the FINAL LATENT, raw little-endian f32, plus a JSON sidecar
that is the guest's own report. This script is the second half of that second
exit: it reads the sidecar, un-patchifies the rows the way the family's VAE
expects, decodes with the diffusers autoencoder of the same checkpoint, and
writes a PNG.

    CUDA_VISIBLE_DEVICES=3 python decode_latent.py \\
        --latent  /tmp/t2i/image.latent.f32 \\
        --sidecar /tmp/t2i/image.json \\
        --model-dir ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/ \\
        --out /tmp/t2i/image.png

WHAT THE SIDECAR ANSWERS AND WHAT THIS SCRIPT STILL KNOWS. The sidecar carries
the geometry: `grid_h`/`grid_w` (the latent-row grid), `row_width` (one row's
values), `latent_channels`, `patch_h`/`patch_w` and `spatial_compression` as
`model.latent()` states them. What it cannot carry is the VAE's own contract --
how a denoiser row maps onto VAE channels, and the normalisation between them --
because that is the `vae.decode` arm's business and the denoise facts
deliberately stop short of it. So this script is family-aware where the guest is
not: it reads `--family` (default `flux2`), and each family is a dozen lines
below. That asymmetry is the point of the exercise: everything the GUEST does is
fact-driven, and only this reference-side script needs the rest.

FLUX.2: a denoiser row is 128 values at /16, which is a 2x2 patch of 32-channel
VAE cells at /8. `_unpack_latents_with_ids` puts the rows back on the (h, w)
grid as [128, h, w]; the VAE's own BatchNorm running stats denormalise it;
`_unpatchify_latents` opens the 2x2 into [32, 2h, 2w]; `vae.decode` does the
rest. Same order, same numbers as `Flux2KleinPipeline.__call__`'s tail.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------

def decode_flux2(latent: torch.Tensor, meta: dict, model_dir: str, device: str,
                 dtype: torch.dtype) -> np.ndarray:
    """`[grid_h, grid_w, row_width]` -> `[H, W, 3]` uint8, FLUX.2's way."""
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(model_dir, subfolder="vae",
                                             torch_dtype=dtype).to(device)
    vae.eval()

    # rows -> [1, C, h, w]; the guest wrote them h-major, which is the packed
    # order `_unpack_latents_with_ids` scatters into.
    x = latent.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    # The VAE's BatchNorm is the packed-space normalisation the pipeline
    # applies BEFORE unpatchifying, so its 128 channels line up with a row.
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(x.device, x.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1)
                     + vae.config.batch_norm_eps).to(x.device, x.dtype)
    x = x * std + mean

    # 2x2 unpatchify: [1, 4c, h, w] -> [1, c, 2h, 2w].
    b, c, h, w = x.shape
    x = x.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(b, c // 4, h * 2, w * 2)

    with torch.no_grad():
        image = vae.decode(x, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)[0].float().cpu().numpy()
    return (image.transpose(1, 2, 0) * 255).round().astype(np.uint8)


FAMILIES = {"flux2": decode_flux2}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def resolve(path: str) -> str:
    """Expand `~` and a single glob (a HuggingFace `snapshots/*/` path)."""
    path = os.path.expanduser(path)
    if any(ch in path for ch in "*?["):
        hits = sorted(glob.glob(path))
        if not hits:
            raise SystemExit(f"{path}: matches nothing")
        return hits[0]
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent", required=True, help="the raw f32 blob the inferlet sent")
    ap.add_argument("--sidecar", required=True, help="its JSON report")
    ap.add_argument("--model-dir", required=True,
                    help="the diffusers folder the artifact was imported from (its `vae/`)")
    ap.add_argument("--out", required=True, help="PNG to write")
    ap.add_argument("--family", default="flux2", choices=sorted(FAMILIES),
                    help="whose VAE contract to apply (default: flux2)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    with open(resolve(args.sidecar)) as f:
        meta = json.load(f)
    grid_h, grid_w = int(meta["grid_h"]), int(meta["grid_w"])
    row_width = int(meta["row_width"])

    raw = np.fromfile(resolve(args.latent), dtype="<f4")
    want = grid_h * grid_w * row_width
    if raw.size != want:
        raise SystemExit(
            f"{args.latent}: {raw.size} floats, but the sidecar says "
            f"{grid_h}x{grid_w}x{row_width} = {want}"
        )
    if not np.isfinite(raw).all():
        print(f"warning: {(~np.isfinite(raw)).sum()} non-finite values in the latent",
              file=sys.stderr)
    latent = torch.from_numpy(raw.reshape(grid_h, grid_w, row_width).copy())

    print(f"latent  {grid_h}x{grid_w}x{row_width}  "
          f"mean {raw.mean():+.4f}  std {raw.std():.4f}")
    print(f"prompt  {meta.get('prompt')!r}  steps {meta.get('steps')}  "
          f"seed {meta.get('seed')}")

    dtype = getattr(torch, args.dtype)
    rgb = FAMILIES[args.family](latent, meta, resolve(args.model_dir), args.device, dtype)

    from PIL import Image

    out = resolve(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    Image.fromarray(rgb).save(out)
    print(f"wrote   {out}  {rgb.shape[1]}x{rgb.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
