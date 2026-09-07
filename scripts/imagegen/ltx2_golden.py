#!/usr/bin/env python3
"""
ltx2_golden.py -- reference dump for LTX-2.5 (M4).

    python ltx2_golden.py --mini                # CPU, seconds
    CUDA_VISIBLE_DEVICES=1 python ltx2_golden.py --vae   # the video VAE decoder alone, fp32

Outputs -> $PIE_IMAGEGEN_GOLDEN/ltx25/
    ltx2_mini.safetensors   a random-init miniature: the DiT under `dit.` and
                            the two text connectors under `connectors.`, the
                            prefixes `checkpoint::file::diffusers` gives the
                            shipped pipeline's `transformer/` and
                            `connectors/` folders, so `pie model import` reads
                            it with the family's ONE reading
    ltx2_mini.npz           one joint video+audio denoise step: the packed
                            latents in, the two text contexts, the timesteps,
                            the ALREADY-NORMALISED rope coordinates the pie
                            port takes, and the two velocities out; plus one
                            connector pass (packed trunk rows in, the two
                            contexts out)
    ltx2_mini_config.json   the config and the tensor list
    ltx2_vae/*.f32          --vae: a fixed random DiT-space latent clip and the
                            fp32 `AutoencoderKLLTX2Video.decode` of it, as raw
                            rows of voxels (see `run_vae`), plus shapes.json

WHY A VENDORED REFERENCE. `import sglang` needs the whole serving stack
(starlette, orjson, ...) which this box does not have, so the reference
classes cannot be imported. `vendor/ltx_2/modeling.py` is a self-contained
transcription of them, with its provenance stated at the top of the file and
the HUGGING FACE checkpoint's names on every module, so the same
`crates/models/src/ltx_2/import.rs` reads this miniature and the shipped
`Lightricks/LTX-2.5-Diffusers`.

The flagship DiT dump (`--full`) is not implemented: it needs the 201 GB
snapshot and the serving environment. The miniature is what the DiT parity
gate drives. The VAE dump (`--vae`) IS the real thing: diffusers 0.40's
`AutoencoderKLLTX2Video` over the shipped `vae/` folder, which is 1.4 GB and
present in the HuggingFace cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golden_common import Tap, md5, npz_keys, outdir  # noqa: E402
from vendor.ltx_2.modeling import (  # noqa: E402
    LTX2Config,
    LTX2ConnectorConfig,
    LTX2TextConnectors,
    LTX2VideoTransformer3DModel,
)

MODEL = "ltx25"
REPO = "Lightricks/LTX-2.5-Diffusers"

# The miniature: two blocks, two heads a side at the REAL head widths (the
# ragged attention kernel is stamped at 64/128/256), the real 128-channel
# latents, a 16-wide caption.
MINI = LTX2Config(
    num_layers=2,
    num_attention_heads=2,
    attention_head_dim=128,
    audio_num_attention_heads=2,
    audio_attention_head_dim=64,
    in_channels=128,
    out_channels=128,
    audio_in_channels=128,
    audio_out_channels=128,
    cross_attention_dim=256,
    audio_cross_attention_dim=128,
)
MINI_CONN = LTX2ConnectorConfig(
    caption_channels=16,
    text_proj_in_factor=49,
    video_heads=2,
    video_head_dim=128,
    audio_heads=2,
    audio_head_dim=64,
    video_layers=1,
    audio_layers=1,
)

# One job: a 3 x 4 x 6 latent grid (72 video rows), 8 audio rows, 16 text rows.
LATENT_FRAMES, LATENT_H, LATENT_W = 3, 4, 6
AUDIO_FRAMES = 8
TEXT_ROWS = 16
FPS = 24.0
# The step the golden takes, in scheduler units (`sigma * 1000`).
SIGMA = 0.909375
AUDIO_SIGMA = 0.909375


def seeded(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """A fixture initialisation that DISCRIMINATES, re-drawn from one seed.

    Not `0.02 * randn` over every parameter, which is what the other goldens
    do: LTX's modulation is `scale_shift_table + adaLN(t)` and its attention
    answer is scaled by `2 sigmoid(W x)`, so flattening every parameter to a
    tiny normal drives every gate to a half of a half and every scale and
    shift to nothing — the fixture would then be nearly `proj_out(norm(
    proj_in(x)))` and would not discriminate a modulation bug from a typo.
    Torch's own `reset_parameters` (and the `randn / sqrt(dim)` tables the
    reference states) keep the gates near one and the modulation alive."""
    g = torch.Generator().manual_seed(seed)
    for name, p in sorted(module.named_parameters()):
        if p.dim() >= 2 and "scale_shift_table" not in name and "registers" not in name:
            # `nn.Linear`'s own kaiming-uniform bound.
            bound = math.sqrt(1.0 / p.shape[-1])
            p.data = ((torch.rand(p.shape, generator=g) * 2 - 1) * bound).to(dtype=p.dtype)
        elif "norm_q" in name or "norm_k" in name:
            # An across-heads RMS gain sits at one; a gain of 0.06 would make
            # every attention logit vanish and every softmax uniform, and the
            # gate could then not tell a wrong rope from a right one.
            p.data = (1.0 + 0.02 * torch.randn(p.shape, generator=g)).to(dtype=p.dtype)
        elif "scale_shift_table" in name or "registers" in name:
            # The modulation tables at unit scale, NOT the reference's
            # `randn / sqrt(dim)`: every gate is then O(1) and a dropped or
            # mis-sliced fold moves the answer instead of hiding under the
            # tolerance. The trained tables are O(1) too.
            p.data = torch.randn(p.shape, generator=g).to(dtype=p.dtype)
        else:
            p.data = (torch.randn(p.shape, generator=g) / math.sqrt(p.shape[-1])).to(
                dtype=p.dtype
            )
    return module


def normalised(coords: torch.Tensor, maxima) -> np.ndarray:
    """The rope's own arithmetic, stopped one step early: the MIDPOINT of the
    latent cell in physical units, over its maximum, mapped to `[-1, 1]` and
    scaled by `pi/2`. That product is what pie's `positions` port takes, and
    what `RopeForm::SplitLadder` multiplies by `theta^(f/(F-1))`."""
    start, end = coords.chunk(2, dim=-1)
    mid = ((start + end) / 2.0).squeeze(-1)  # [B, axes, T]
    axes = mid.shape[1]
    out = torch.stack(
        [(2.0 * mid[:, i] / float(maxima[i]) - 1.0) * (math.pi / 2.0) for i in range(axes)],
        dim=-1,
    )
    return out.float().numpy()  # [B, T, axes]


def connector_positions(rows: int, base_seq_len: int) -> np.ndarray:
    i = np.arange(rows, dtype=np.float64) / float(base_seq_len)
    return ((2.0 * i - 1.0) * (math.pi / 2.0)).astype(np.float32).reshape(rows, 1)


def run_mini(d: str, device="cpu", dtype=torch.float32) -> None:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    dit = seeded(LTX2VideoTransformer3DModel(MINI).to(device=device, dtype=dtype), 0).eval()
    conn = seeded(LTX2TextConnectors(MINI_CONN).to(device=device, dtype=dtype), 1).eval()

    state = {f"dit.{k}": v.detach().contiguous().float().cpu() for k, v in dit.state_dict().items()}
    state.update(
        {
            f"connectors.{k}": v.detach().contiguous().float().cpu()
            for k, v in conn.state_dict().items()
        }
    )
    save_file(state, os.path.join(d, "ltx2_mini.safetensors"), metadata={"format": "pt"})

    tap = Tap()
    g = torch.Generator().manual_seed(1234)
    rows = LATENT_FRAMES * LATENT_H * LATENT_W
    x_v = torch.randn(1, rows, MINI.in_channels, generator=g).to(device, dtype)
    x_a = torch.randn(1, AUDIO_FRAMES, MINI.audio_in_channels, generator=g).to(device, dtype)

    # --- the connector pass: packed trunk rows in, the two contexts out ----
    stack = torch.randn(
        1, TEXT_ROWS, MINI_CONN.caption_channels * MINI_CONN.text_proj_in_factor, generator=g
    ).to(device, dtype)
    with torch.no_grad():
        video_ctx, audio_ctx = conn(stack)
    tap.put("mini.conn.in.text", stack)
    tap.put("mini.conn.in.positions", torch.from_numpy(
        connector_positions(TEXT_ROWS, MINI_CONN.rope_base_seq_len)
    ))
    tap.put("mini.conn.out.video", video_ctx)
    tap.put("mini.conn.out.audio", audio_ctx)

    # --- the joint denoise step -------------------------------------------
    t_v = torch.full((1,), SIGMA * 1000.0, device=device, dtype=dtype)
    t_a = torch.full((1,), AUDIO_SIGMA * 1000.0, device=device, dtype=dtype)
    with torch.no_grad():
        v_v, v_a = dit(
            hidden_states=x_v,
            audio_hidden_states=x_a,
            encoder_hidden_states=video_ctx,
            audio_encoder_hidden_states=audio_ctx,
            timestep=t_v,
            audio_timestep=t_a,
            num_frames=LATENT_FRAMES,
            height=LATENT_H,
            width=LATENT_W,
            fps=FPS,
            audio_num_frames=AUDIO_FRAMES,
        )

    video_coords = dit.rope.prepare_video_coords(
        1, LATENT_FRAMES, LATENT_H, LATENT_W, torch.device(device), fps=FPS
    )
    audio_coords = dit.audio_rope.prepare_audio_coords(1, AUDIO_FRAMES, torch.device(device))
    tap.put("mini.dit.in.latents", x_v)
    tap.put("mini.dit.in.audio_latents", x_a)
    tap.put("mini.dit.in.context", video_ctx)
    tap.put("mini.dit.in.audio_context", audio_ctx)
    tap.put("mini.dit.in.timestep", t_v)
    tap.put("mini.dit.in.audio_timestep", t_a)
    tap.put(
        "mini.dit.in.positions",
        torch.from_numpy(normalised(video_coords, (MINI.pos_embed_max_pos, MINI.base_height, MINI.base_width))),
    )
    tap.put(
        "mini.dit.in.audio_positions",
        torch.from_numpy(normalised(audio_coords, (MINI.audio_pos_embed_max_pos,))),
    )
    tap.put("mini.dit.out.velocity", v_v)
    tap.put("mini.dit.out.audio_velocity", v_a)

    with open(os.path.join(d, "ltx2_mini_config.json"), "w") as f:
        json.dump(
            {
                "dit": MINI.__dict__ | {"vae_scale_factors": list(MINI.vae_scale_factors)},
                "connectors": MINI_CONN.__dict__,
                "job": {
                    "latent_frames": LATENT_FRAMES,
                    "latent_height": LATENT_H,
                    "latent_width": LATENT_W,
                    "audio_frames": AUDIO_FRAMES,
                    "text_rows": TEXT_ROWS,
                    "fps": FPS,
                    "sigma": SIGMA,
                    "audio_sigma": AUDIO_SIGMA,
                },
                "num_parameters": {
                    "dit": int(sum(v.numel() for v in dit.state_dict().values())),
                    "connectors": int(sum(v.numel() for v in conn.state_dict().values())),
                },
                "tensors": {k: list(v.shape) for k, v in sorted(state.items())},
            },
            f,
            indent=2,
            default=str,
        )
    tap.save(os.path.join(d, "ltx2_mini.npz"))
    npz_keys(tap)
    print(
        f"  mini: {rows} video rows, {AUDIO_FRAMES} audio rows, {TEXT_ROWS} text rows; "
        f"dit {sum(v.numel() for v in dit.state_dict().values())} params"
    )


# The VAE golden's latent clip: 3 latent frames of 8 x 12 cells, which is
# 17 frames of 256 x 384 pixels — small enough to keep the dump at 20 MB and
# the parity fire under a second, big enough that every up block's shuffle,
# every temporal trim and the un-patchify land on more than one cell.
VAE_LATENT_FRAMES, VAE_LATENT_H, VAE_LATENT_W = 3, 8, 12
VAE_SEED = 7


def run_vae(d: str, device="cuda", shape=(VAE_LATENT_FRAMES, VAE_LATENT_H, VAE_LATENT_W)) -> None:
    """`AutoencoderKLLTX2Video` alone, fp32, over a fixed random latent.

    The DiT works in a NORMALISED latent space and the pipeline undoes that
    before the decode (`_denormalize_latents`: `z * latents_std + latents_mean`,
    `scaling_factor` 1.0); pie's `vae.decode` arm undoes it itself, so what
    is dumped as `latent.f32` is the DiT-space latent — a unit normal, which
    is what a denoised latent in that space looks like — and `denorm.f32` is
    the decoder's own input beside it, for a bisect.

    The decoder is NON-causal (`decoder_causal: False`): `decode` is one call
    over the whole clip, every conv padding its time axis with the clip's own
    first and last frames, and each of the three temporal upsamplers drops
    the first frame after its shuffle, so `F = 8 * (T - 1) + 1`. There is no
    per-frame loop and no cache, so there are no chunk boundaries to dump.
    `timestep_conditioning` is off, so the `temb` argument is not passed.
    """
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        REPO, subfolder="vae", torch_dtype=torch.float32
    ).to(device).eval()
    cfg = vae.config
    assert not cfg.decoder_causal, "this dump states the non-causal decoder"
    assert not cfg.timestep_conditioning, "this dump passes no temb"
    t_lat, h_lat, w_lat = shape

    g = torch.Generator().manual_seed(VAE_SEED)
    z = torch.randn(1, cfg.latent_channels, t_lat, h_lat, w_lat, generator=g).to(
        device=device, dtype=torch.float32
    )
    mean = vae.latents_mean.to(device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = vae.latents_std.to(device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    denorm = z * std / cfg.scaling_factor + mean

    with torch.no_grad():
        x = vae.decode(denorm, return_dict=False)[0]        # [1, 3, 8T-7, 32H, 32W]

    frames = int(x.shape[2])
    assert frames == 8 * t_lat - 7, f"{t_lat} latent frames should land {8 * t_lat - 7}, not {frames}"
    assert tuple(x.shape[-2:]) == (32 * h_lat, 32 * w_lat), tuple(x.shape)

    # The default shape is THE gate's golden; another shape lands beside it
    # under its own name, so `the_ltx_2_vae_answers_the_reference` can be
    # pointed at it (`PIE_LTX2_VAE_GOLDEN=ltx2_vae_TxHxW`) without
    # disturbing the row `gates.py` measures.
    default = shape == (VAE_LATENT_FRAMES, VAE_LATENT_H, VAE_LATENT_W)
    raw = os.path.join(d, "ltx2_vae" if default else f"ltx2_vae_{t_lat}x{h_lat}x{w_lat}")
    os.makedirs(raw, exist_ok=True)
    shapes = {}
    for key, t in (("latent", z[0]), ("denorm", denorm[0]), ("pixels", x[0])):
        cthw = t.detach().float().cpu().numpy()             # [C, T, H, W]
        rows = np.ascontiguousarray(cthw.transpose(1, 2, 3, 0)).astype("<f4")
        rows.tofile(os.path.join(raw, f"{key}.f32"))
        shapes[key] = {"t": int(cthw.shape[1]), "h": int(cthw.shape[2]),
                       "w": int(cthw.shape[3]), "channels": int(cthw.shape[0])}
    shapes["latents_mean"] = [float(v) for v in vae.latents_mean.float()]
    shapes["latents_std"] = [float(v) for v in vae.latents_std.float()]
    shapes["scaling_factor"] = float(cfg.scaling_factor)
    shapes["decoder_causal"] = bool(cfg.decoder_causal)
    shapes["seed"] = VAE_SEED
    shapes["source"] = f"torch.randn(1, {cfg.latent_channels}, {t_lat}, {h_lat}, {w_lat}) at seed {VAE_SEED}"
    with open(os.path.join(raw, "shapes.json"), "w") as f:
        json.dump(shapes, f, indent=2)

    from PIL import Image
    for k in (0, frames // 2, frames - 1):
        u8 = (np.clip(x[0, :, k].float().cpu().numpy().transpose(1, 2, 0) + 1, 0, 2) * 127.5)
        Image.fromarray(u8.astype("uint8")).save(os.path.join(raw, f"frame{k:03d}.png"))
    print(f"  vae: latent {tuple(z.shape)} -> pixels {tuple(x.shape)} "
          f"[{float(x.min()):.3f}, {float(x.max()):.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--vae", action="store_true")
    ap.add_argument("--vae-shape", default=None,
                    help="T,H,W of the latent clip (default 3,8,12)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if not (a.mini or a.vae):
        a.mini = True
    d = outdir(MODEL)
    torch.set_grad_enabled(False)
    if a.vae:
        print("== vae ==")
        shape = tuple(int(v) for v in a.vae_shape.split(",")) if a.vae_shape else (
            VAE_LATENT_FRAMES, VAE_LATENT_H, VAE_LATENT_W)
        run_vae(d, "cuda" if torch.cuda.is_available() else "cpu", shape)
        if not a.mini:
            return
    print("== mini ==")
    run_mini(d, a.device)
    # `golden_common.manifest` asks diffusers and transformers for their
    # versions and this dump needs neither, so it writes its own.
    rows = [
        {"file": name, "bytes": os.path.getsize(os.path.join(d, name)), "md5": md5(os.path.join(d, name))}
        for name in sorted(os.listdir(d))
        if os.path.isfile(os.path.join(d, name)) and name != "MANIFEST.json"
    ]
    with open(os.path.join(d, "MANIFEST.json"), "w") as f:
        json.dump(
            {
                "dir": d,
                "torch": torch.__version__,
                "repo": REPO,
                "reference": "scripts/imagegen/vendor/ltx_2/modeling.py (transcribed from sglang)",
                "sigma": SIGMA,
                "files": rows,
            },
            f,
            indent=2,
        )
    for r in rows:
        print(f"    {r['file']:<34} {r['bytes']:>12,}  {r['md5']}")


if __name__ == "__main__":
    main()
