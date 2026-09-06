#!/usr/bin/env python3
"""
zimage_golden.py -- reference dump for Z-Image-Turbo (M1).

Full run (needs ~25 GB VRAM, ~1 min):
    CUDA_VISIBLE_DEVICES=0 python zimage_golden.py --full

Miniature (CPU or GPU, seconds; no weights needed beyond the repo config):
    python zimage_golden.py --mini

Outputs -> $PIE_IMAGEGEN_GOLDEN/z-image/  (default /root/.cache/pie-imagegen/golden/z-image)
    zimage_golden.npz    prompt embeds (Qwen3 layer -2, unpadded), initial noise,
                         step-0 transformer inputs + velocity, per-step latents, final latent
    zimage_golden.png    decoded 1024x1024 image
    zimage_mini.npz      one random-init forward of a tiny ZImageTransformer2DModel
    zimage_mini.safetensors  the tiny weights (seed 0)

Settings are the official Turbo defaults (Z-Image repo `inference.py`): 8 steps,
guidance 0.0, cfg_normalization False, max_sequence_length 512, shift 3.0 from the
shipped scheduler_config.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from golden_common import (Tap, hook_prepare_latents, hook_scheduler, hook_transformer,
                           manifest, npz_keys, outdir)

REPO = "Tongyi-MAI/Z-Image-Turbo"
MODEL = "z-image"
PROMPT = "a red bicycle leaning on a blue wall"
SEED = 0
STEPS = 8
GUIDANCE = 0.0
SIZE = 1024

# miniature: 2 noise-refiner + 2 context-refiner + 2 joint layers, dim 256.
# head_dim must equal sum(axes_dims) and dim must stay >= 256 (adaLN width trap, study D.2).
MINI_CFG = dict(
    all_patch_size=(2,), all_f_patch_size=(1,),
    in_channels=16, dim=256, n_layers=2, n_refiner_layers=2,
    n_heads=4, n_kv_heads=4, norm_eps=1e-5, qk_norm=True,
    cap_feat_dim=64, siglip_feat_dim=None,
    rope_theta=256.0, t_scale=1000.0,
    axes_dims=[16, 24, 24], axes_lens=[256, 64, 64],
)


def run_full(d: str, dtype=torch.bfloat16):
    from diffusers import ZImagePipeline

    tap = Tap()
    pipe = ZImagePipeline.from_pretrained(REPO, torch_dtype=dtype).to("cuda")
    pipe.set_progress_bar_config(disable=False)

    cfgs = {"transformer": dict(pipe.transformer.config),
            "scheduler": dict(pipe.scheduler.config),
            "vae": dict(pipe.vae.config)}
    with open(os.path.join(d, "zimage_config.json"), "w") as f:
        json.dump(cfgs, f, indent=2, default=str)

    # ---- text stage: Qwen3 hidden_states[-2], unpadded, one tensor per prompt --------
    with torch.no_grad():
        embeds = pipe.encode_prompt(PROMPT, device=torch.device("cuda"),
                                    do_classifier_free_guidance=False,
                                    max_sequence_length=512)
    pos = embeds[0] if isinstance(embeds, tuple) else embeds
    tap.put_tree("prompt_embeds", pos)
    lens = [t.shape[0] for t in pos] if isinstance(pos, list) else [pos.shape[-2]]
    tap.put("prompt_embeds.lengths", lens)
    print(f"  prompt embeds: {lens} x {pos[0].shape[-1] if isinstance(pos, list) else pos.shape[-1]}")

    # ---- denoise --------------------------------------------------------------------
    r1 = hook_prepare_latents(pipe, tap)
    r2 = hook_transformer(pipe.transformer, tap, "dit", steps=(0,))
    r3 = hook_scheduler(pipe, tap)
    g = torch.Generator("cpu").manual_seed(SEED)
    out = pipe(prompt=PROMPT, height=SIZE, width=SIZE, num_inference_steps=STEPS,
               guidance_scale=GUIDANCE, generator=g, max_sequence_length=512,
               output_type="pil")
    r3(); r2(); r1()

    tap.put("sigmas", np.asarray(pipe.scheduler.sigmas.float().cpu()))
    tap.put("timesteps", np.asarray(pipe.scheduler.timesteps.float().cpu()))
    final = max((k for k in tap.d if k.startswith("sched.x")), key=lambda k: int(k[7:]))
    tap.d["latent.final"] = tap.d[final]

    img = out.images[0]
    img.save(os.path.join(d, "zimage_golden.png"))
    tap.put("image.rgb", np.asarray(img).astype(np.float32))
    tap.save(os.path.join(d, "zimage_golden.npz"))
    print("  keys:"); npz_keys(tap)


def run_mini(d: str, device="cpu", dtype=torch.float32):
    from diffusers import ZImageTransformer2DModel
    from safetensors.torch import save_file

    torch.manual_seed(0)
    m = ZImageTransformer2DModel(**MINI_CFG).to(device=device, dtype=dtype).eval()
    g = torch.Generator().manual_seed(0)
    for _, p in sorted(m.named_parameters()):
        p.data = (0.02 * torch.randn(p.shape, generator=g)).to(device=device, dtype=dtype)
    sd = {k: v.detach().contiguous().float().cpu() for k, v in m.state_dict().items()}
    save_file(sd, os.path.join(d, "zimage_mini.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(d, "zimage_mini_config.json"), "w") as f:
        json.dump({"config": MINI_CFG,
                   "num_parameters": int(sum(v.numel() for v in sd.values())),
                   "tensors": {k: list(v.shape) for k, v in sorted(sd.items())}}, f, indent=2)

    tap = Tap()
    gi = torch.Generator().manual_seed(1234)
    x = [torch.randn(16, 1, 16, 16, generator=gi).to(device=device, dtype=dtype)]     # C,F,H,W
    cap = [torch.randn(8, MINI_CFG["cap_feat_dim"], generator=gi).to(device=device, dtype=dtype)]
    t = torch.tensor([500.0], device=device, dtype=dtype)
    tap.put_tree("mini.in.x", x); tap.put_tree("mini.in.cap", cap); tap.put("mini.in.t", t)
    with torch.no_grad():
        out = m(x, t, cap, return_dict=False)[0]
    tap.put_tree("mini.out", out)
    tap.save(os.path.join(d, "zimage_mini.npz"))
    print(f"  mini params: {sum(v.numel() for v in sd.values())}")
    npz_keys(tap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if not (a.full or a.mini):
        a.full = a.mini = True
    d = outdir(MODEL)
    torch.set_grad_enabled(False)
    if a.mini:
        print("== mini =="); run_mini(d, a.device)
    if a.full:
        print("== full =="); run_full(d)
    manifest(d, {"repo": REPO, "prompt": PROMPT, "seed": SEED,
                 "steps": STEPS, "guidance": GUIDANCE, "size": SIZE,
                 "mini_config": MINI_CFG})


if __name__ == "__main__":
    main()
