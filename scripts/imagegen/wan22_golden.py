#!/usr/bin/env python3
"""
wan22_golden.py -- reference dump for Wan 2.2 TI2V-5B (M3).

    CUDA_VISIBLE_DEVICES=0 python wan22_golden.py --full     # ~30 GB VRAM, minutes
    python wan22_golden.py --mini                            # CPU, seconds
    CUDA_VISIBLE_DEVICES=0 python wan22_golden.py --vae      # the VAE alone, fp32

Outputs -> $PIE_IMAGEGEN_GOLDEN/wan22/
    wan22_golden.npz   umT5 prompt embeds (truncate-then-zero-pad to 512), initial noise,
                       step-0 transformer inputs (incl. the per-token `timestep [B,S]`
                       that TI2V's expand_timesteps produces) and velocity, per-step
                       latents, final latent, VAE-decode input/output
    wan22_golden.mp4 / wan22_frames.npy
    wan22_mini.npz     two random-init WanTransformer3DModel forwards:
                         `nano`  head_dim 24 -> rope split [8,8,8]   (catches d-4*(d//6))
                         `d128`  head_dim 128 -> rope split [44,42,42] (the real split)
    wan22_vae/*.f32    --vae: the DiT-space latent and the fp32 decode of it, as raw
                       little-endian f32 in the row-per-voxel `(t, h, w)` layout pie's
                       voxel axis reads, PLUS the per-chunk boundaries the decoder's own
                       frame-by-frame loop produces (`shapes.json`)

TI2V-5B specifics: VAE stride (4,16,16), z=48, in/out channels 48, expand_timesteps=True,
single backbone (boundary_ratio null), UniPCMultistepScheduler.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from golden_common import (Tap, hook_prepare_latents, hook_scheduler, hook_transformer,
                           manifest, npz_keys, outdir)

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
MODEL = "wan22"
PROMPT = "a red bicycle leaning on a blue wall"
NEGATIVE = ""
SEED = 0
STEPS = 8
HEIGHT, WIDTH, FRAMES = 480, 832, 17

MINI_CFGS = {
    # study wan22.md D.3 "wan22-nano": deliberately a different rope split than 128
    "nano": dict(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=24,
                 in_channels=16, out_channels=16, text_dim=64, freq_dim=32, ffn_dim=128,
                 num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                 eps=1e-6, image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=1024),
    # the real head_dim, so the [44,42,42] split is exercised too
    "d128": dict(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=128,
                 in_channels=16, out_channels=16, text_dim=64, freq_dim=256, ffn_dim=512,
                 num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                 eps=1e-6, image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=1024),
}
MINI_LATENT = (16, 5, 16, 16)     # C, T, H, W -> S = 5*8*8 = 320 tokens
MINI_CTX_LEN = 32


def run_full(d: str, dtype=torch.bfloat16):
    from diffusers import WanPipeline

    tap = Tap()
    pipe = WanPipeline.from_pretrained(REPO, torch_dtype=dtype)
    pipe.vae.to(torch.float32)                     # fp32 decode is required for parity
    pipe.to("cuda")
    with open(os.path.join(d, "wan22_config.json"), "w") as f:
        json.dump({"transformer": dict(pipe.transformer.config),
                   "scheduler": dict(pipe.scheduler.config),
                   "vae": dict(pipe.vae.config),
                   "pipeline": {k: v for k, v in dict(pipe.config).items()
                                if not k.startswith("_")}}, f, indent=2, default=str)

    with torch.no_grad():
        pe, npe = pipe.encode_prompt(prompt=PROMPT, negative_prompt=NEGATIVE,
                                     do_classifier_free_guidance=True,
                                     device=torch.device("cuda"), max_sequence_length=512)
    tap.put("prompt_embeds", pe)
    if npe is not None:
        tap.put("negative_prompt_embeds", npe)
    nz = (pe[0].abs().sum(-1) > 0).sum().item()
    tap.put("prompt_embeds.nonzero_rows", nz)
    print(f"  prompt embeds {tuple(pe.shape)}  nonzero rows {nz}")

    # VAE decode tap
    vtap = {}
    vorig = pipe.vae.decode

    def vdec(z, *a, **kw):
        out = vorig(z, *a, **kw)
        vtap.setdefault("in", z)
        vtap.setdefault("out", out[0] if isinstance(out, tuple) else getattr(out, "sample", out))
        return out
    pipe.vae.decode = vdec

    r1 = hook_prepare_latents(pipe, tap)
    r2 = hook_transformer(pipe.transformer, tap, "dit", steps=(0,))
    r3 = hook_scheduler(pipe, tap)
    g = torch.Generator("cpu").manual_seed(SEED)
    out = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH,
               num_frames=FRAMES, num_inference_steps=STEPS, generator=g,
               max_sequence_length=512, output_type="np")
    r3(); r2(); r1(); pipe.vae.decode = vorig

    tap.put("sigmas", torch.as_tensor(np.asarray(pipe.scheduler.sigmas, dtype=np.float32)))
    tap.put("timesteps", pipe.scheduler.timesteps.float().cpu())
    final = max((k for k in tap.d if k.startswith("sched.x")), key=lambda k: int(k[7:]))
    tap.d["latent.final"] = tap.d[final]
    for k, v in vtap.items():
        tap.put("vae.decode." + k, v)

    frames = out.frames[0]                       # (F, H, W, 3) float in [0,1]
    u8 = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    np.save(os.path.join(d, "wan22_frames.npy"), u8)
    try:
        import imageio.v3 as iio
        iio.imwrite(os.path.join(d, "wan22_golden.mp4"), u8, fps=24, codec="libx264")
    except Exception as e:                        # pragma: no cover
        print(f"  [warn] mp4 encode failed: {e}")
    from PIL import Image
    Image.fromarray(u8[0]).save(os.path.join(d, "wan22_frame0.png"))
    tap.put("frames.u8_shape", list(u8.shape))
    tap.save(os.path.join(d, "wan22_golden.npz"))
    npz_keys(tap)


def run_mini(d: str, device="cpu", dtype=torch.float32):
    from diffusers import WanTransformer3DModel
    from safetensors.torch import save_file

    tap = Tap()
    meta = {}
    for tag, cfg in MINI_CFGS.items():
        torch.manual_seed(0)
        m = WanTransformer3DModel(**cfg).to(device=device, dtype=dtype).eval()
        g = torch.Generator().manual_seed(0)
        for _, p in sorted(m.named_parameters()):
            p.data = (0.02 * torch.randn(p.shape, generator=g)).to(device=device, dtype=dtype)
        sd = {k: v.detach().contiguous().float().cpu() for k, v in m.state_dict().items()}
        save_file(sd, os.path.join(d, f"wan22_mini_{tag}.safetensors"), metadata={"format": "pt"})
        meta[tag] = {"config": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                     "num_parameters": int(sum(v.numel() for v in sd.values())),
                     "tensors": {k: list(v.shape) for k, v in sorted(sd.items())}}

        c, t, h, w = MINI_LATENT
        gi = torch.Generator().manual_seed(1234)
        hs = torch.randn(1, c, t, h, w, generator=gi).to(device, dtype)
        ctx = torch.randn(1, MINI_CTX_LEN, cfg["text_dim"], generator=gi).to(device, dtype)
        ts = torch.tensor([500.0], device=device, dtype=dtype)
        tap.put(f"mini.{tag}.in.hidden_states", hs)
        tap.put(f"mini.{tag}.in.encoder_hidden_states", ctx)
        tap.put(f"mini.{tag}.in.timestep", ts)
        with torch.no_grad():
            o = m(hidden_states=hs, timestep=ts, encoder_hidden_states=ctx, return_dict=False)
        tap.put_tree(f"mini.{tag}.out", o)

        # TI2V's per-token timestep path: timestep is [B, S] instead of [B]
        s = t * (h // 2) * (w // 2)
        pt = torch.full((1, s), 500.0, device=device, dtype=dtype)
        pt[:, : (h // 2) * (w // 2)] = 0.0        # first latent frame is the conditioning image
        tap.put(f"mini.{tag}.in.timestep_pertoken", pt)
        try:
            with torch.no_grad():
                o2 = m(hidden_states=hs, timestep=pt, encoder_hidden_states=ctx, return_dict=False)
            tap.put_tree(f"mini.{tag}.out_pertoken", o2)
        except Exception as e:
            print(f"  [warn] per-token timestep forward failed for {tag}: {type(e).__name__}: {e}")
        print(f"  mini/{tag}: {meta[tag]['num_parameters']} params, S = {s}")

    with open(os.path.join(d, "wan22_mini_config.json"), "w") as f:
        json.dump({"latent": list(MINI_LATENT), "ctx_len": MINI_CTX_LEN, "variants": meta},
                  f, indent=2)
    tap.save(os.path.join(d, "wan22_mini.npz"))
    npz_keys(tap)


def run_vae(d: str, device="cuda"):
    """`AutoencoderKLWan` alone, fp32, over the full run's final latent.

    The DiT works in a NORMALISED latent space and the pipeline undoes that
    before the decode (`latents * latents_std + latents_mean`); pie's
    `vae.decode` arms undo it themselves, so what is dumped as `latent.f32`
    is the DiT-space latent — exactly the rows the denoise reading answers —
    and `denorm.f32` is the decoder's own input beside it, for a bisect.

    `_decode` is a LOOP: it clears the per-conv frame caches, runs
    `post_quant_conv` over the whole latent, and then calls the decoder once
    per latent frame carrying the caches across. Latent frame 0 lands ONE
    output frame, every later one lands FOUR, so `F = 4*T - 3`. Both halves
    are dumped: the whole clip's pixels, and the frame boundary each chunk
    ends at, so a port can be scored chunk by chunk and say WHICH fire drifted.
    """
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae",
                                           torch_dtype=torch.float32).to(device).eval()
    cfg = vae.config
    full = os.path.join(d, "wan22_golden.npz")
    if not os.path.exists(full):
        raise SystemExit(f"{full} is missing; run `--full` first (the latent is a real one)")
    z = np.load(full)["latent.final"]            # [1, 48, T, H, W], DiT space
    z = torch.from_numpy(np.ascontiguousarray(z.reshape(z.shape[-4:]))).to(
        device=device, dtype=torch.float32)[None]

    mean = torch.tensor(cfg.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(cfg.latents_std, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    denorm = z * std + mean

    with torch.no_grad():
        x = vae.decode(denorm, return_dict=False)[0]          # [1, 3, 4T-3, 16H, 16W]

    t_lat = int(z.shape[2])
    frames = int(x.shape[2])
    assert frames == 4 * t_lat - 3, f"{t_lat} latent frames should land {4 * t_lat - 3}, not {frames}"

    raw = os.path.join(d, "wan22_vae")
    os.makedirs(raw, exist_ok=True)
    shapes = {}
    for key, t in (("latent", z[0]), ("denorm", denorm[0]), ("pixels", x[0])):
        cthw = t.detach().float().cpu().numpy()               # [C, T, H, W]
        rows = np.ascontiguousarray(cthw.transpose(1, 2, 3, 0)).astype("<f4")
        rows.tofile(os.path.join(raw, f"{key}.f32"))
        shapes[key] = {"t": int(cthw.shape[1]), "h": int(cthw.shape[2]),
                       "w": int(cthw.shape[3]), "channels": int(cthw.shape[0])}
    # Chunk `k` of the decode loop is latent frame `k`; it lands output
    # frames `[chunks[k], chunks[k+1])`.
    shapes["chunks"] = [0] + [1 + 4 * k for k in range(t_lat)]
    shapes["latents_mean"] = [float(v) for v in cfg.latents_mean]
    shapes["latents_std"] = [float(v) for v in cfg.latents_std]
    shapes["clip_output"] = bool(cfg.clip_output)
    shapes["source"] = "latent.final of wan22_golden.npz"
    with open(os.path.join(raw, "shapes.json"), "w") as f:
        json.dump(shapes, f, indent=2)

    from PIL import Image
    for k in (0, frames // 2):
        u8 = (np.clip(x[0, :, k].float().cpu().numpy().transpose(1, 2, 0) + 1, 0, 2) * 127.5)
        Image.fromarray(u8.astype("uint8")).save(os.path.join(raw, f"frame{k:03d}.png"))
    print(f"  vae: latent {tuple(z.shape)} -> pixels {tuple(x.shape)} "
          f"[{float(x.min()):.3f}, {float(x.max()):.3f}]; chunks {shapes['chunks']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--vae", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if not (a.full or a.mini or a.vae):
        a.full = a.mini = a.vae = True
    d = outdir(MODEL)
    torch.set_grad_enabled(False)
    if a.mini:
        print("== mini =="); run_mini(d, a.device)
    if a.full:
        print("== full =="); run_full(d)
    if a.vae:
        print("== vae =="); run_vae(d, "cuda" if torch.cuda.is_available() else "cpu")
    manifest(d, {"repo": REPO, "prompt": PROMPT, "seed": SEED, "steps": STEPS,
                 "height": HEIGHT, "width": WIDTH, "frames": FRAMES})


if __name__ == "__main__":
    main()
