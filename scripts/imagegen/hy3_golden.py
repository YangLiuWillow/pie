#!/usr/bin/env python3
"""
hy3_golden.py -- reference dump for HunyuanImage 3.0 (M6).

    CUDA_VISIBLE_DEVICES=1 python hy3_golden.py --mini          # seconds, ~1 GB
    CUDA_VISIBLE_DEVICES=1 python hy3_golden.py --mini --cpu

Outputs -> $PIE_IMAGEGEN_GOLDEN/hy3/
    hy3_mini.safetensors    a random-init miniature in the CHECKPOINT's own
                            spelling (`model.layers.N.*`, `patch_embed.model.*`,
                            `final_layer.model.*`, `time*_emb*`), trunk and image
                            head only -- what `models::hunyuan_image_3`'s import
                            reads.
    hy3_mini_config.json    the config the miniature was built from, plus the
                            sequence layout the dump used.
    hy3_mini.npz            one PREFILL (the causal text pass, logits) and one
                            DENOISE step, tapped at every seam pie reads back:
                              image.in   patch_embed(x_t, time_embed(t))
                              denoise    the trunk's rows for the image span and
                                         for the <timestep> row (NO ln_f)
                              image.out  final_layer(rows, time_embed_2(t))
                            beside the inputs (ids, 2-D rope positions and the
                            cos/sin they build, the generalized causal mask, the
                            noisy latent, the scheduler timestep).

WHAT HAD TO BE PATCHED to get here (all inside this script, none in the
reference):

  * `HunyuanImage3ForCausalMM(config)` builds the VAE and the SigLIP2 tower
    too; at the flagship's `vae`/`vit` blocks that is 1.7 B parameters for a
    dump that touches neither. The miniature's config shrinks both to toys and
    the dump drops their tensors from the safetensors.
  * The reference reaches the trunk through `generate_image()`, which needs the
    real tokenizer, the resolution group, a system prompt and three `generate`
    calls. This script assembles the sequence by hand (the layout is
    `tokenization_hunyuan_image_3.py:899-938`) and calls `model.model(...)`
    directly with `inputs_embeds`, the bool mask and the 2-D `custom_pos_emb`
    the reference builds -- no tokenizer, no pipeline, no `flash_attn`.
  * `attn_implementation` is left at the default (`HunyuanImage3SDPAAttention`
    is BOTH the "eager" and the "sdpa" row), so the mask is exact and no
    flash-attention wheel is needed. `flashinfer` is optional in the reference
    and absent here, so the MoE takes its eager per-expert loop.
  * `pad_token_id`/`image_token_id` are checkpoint ids in the 128k range, so the
    miniature keeps the REAL vocabulary (133 120 rows) even at hidden 256:
    `nn.Embedding` refuses a padding index outside the table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

from golden_common import Tap, manifest, npz_keys, outdir

MODEL = "hy3"

# The GitHub package (byte-identical to the Instruct repo's modeling file); the
# base HF repo ships the same classes under `hunyuan.py`.
HY3_SRC = os.environ.get(
    "HY3_SRC",
    "/tmp/claude-0/-root-Workspace-pie/85b561db-ffb2-4c11-af67-9e924fa46755/scratchpad/hy3/gh",
)
HY3_CONFIG = os.environ.get("HY3_CONFIG", os.path.join(HY3_SRC, "..", "config.json"))

# The miniature: two layers, eight experts top-2 plus the shared one, a 64-wide
# head (the 2-D rope wants head_dim % 4 == 0), the image head's waist at 64.
MINI = dict(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    attention_head_dim=64,
    head_dim=64,
    num_experts=8,
    moe_topk=[2, 2],
    num_shared_expert=[1, 1],
    moe_intermediate_size=[256, 256],
    intermediate_size=256,
    max_position_embeddings=4096,
    patch_embed_hidden_dim=64,
    image_base_size=256,
)
# 128 x 128 pixels at stride 16 and patch 1 — small enough that the
# whole case fits argv (the sandbox's `/scratch` is per-process, so a
# harness cannot place a file in it).
TOKEN_H, TOKEN_W = 8, 8
# `<bos>` + five text ids + `<boi> <img_size> <img_ratio>` then `<timestep>`.
# `<|startoftext|>` + five ordinary pieces + `<boi> <img_size_1024> <img_ratio_0>`.
TEXT_IDS = [127958, 100, 200, 300, 400, 500, 128000, 128037, 128044]
# `<timestep>` in the base repo's tokenizer (128 010 is `<cfg>`).
TIMESTEP_ID = 128017
IMG_ID = 128006
EOI_ID = 128001
# `<cfg>`: what the unconditional branch writes over every prompt token.
CFG_ID = 128010
# `sigma' * 1000` at the first of eight shifted steps (shift 3).
TIMESTEP = 750.0
SEED = 0

DROP = ("vae.", "vision_model.", "vision_aligner.")


def build(device: str, dtype: torch.dtype):
    sys.path.insert(0, HY3_SRC)
    from hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config
    import hunyuan_image_3.modeling_hunyuan_image_3 as M

    base = json.load(open(HY3_CONFIG))
    cfg = dict(base)
    cfg.update(MINI)
    # Toy VAE / ViT: built and never called, dropped from the dump.
    cfg["vae"] = dict(
        cfg["vae"],
        block_out_channels=[32, 64, 64, 64, 64],
        layers_per_block=1,
        sample_size=64,
        sample_tsize=4,
    )
    cfg["vit"] = dict(
        cfg["vit"],
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        num_patches=64,
    )
    cfg["vit_aligner"] = dict(cfg["vit_aligner"], input_dim=64, n_embed=256)
    cfg.pop("architectures", None)
    cfg.pop("auto_map", None)
    config = HunyuanImage3Config(**cfg)

    torch.manual_seed(SEED)
    model = M.HunyuanImage3ForCausalMM(config)
    # Small, well-conditioned weights: the reference's `_init_weights` only
    # touches Linear/Embedding, and a random-init GroupNorm/Conv stack at
    # PyTorch defaults is fine to diff but noisy to read.
    gen = torch.Generator().manual_seed(SEED + 1)
    for name, p in sorted(model.named_parameters()):
        if name.startswith(DROP):
            continue
        if p.dim() >= 2:
            p.data = 0.02 * torch.randn(p.shape, generator=gen)
        else:
            p.data = 0.01 * torch.randn(p.shape, generator=gen)
    model = model.to(device=device, dtype=dtype).eval()
    return M, config, model


def sequence():
    """The T2I pretrain layout, by hand.

    `<bos> text <boi> <img_size> <img_ratio> <timestep> <img> x h*w <eoi>`
    """
    n = TOKEN_H * TOKEN_W
    ids = list(TEXT_IDS) + [TIMESTEP_ID] + [IMG_ID] * n + [EOI_ID]
    t_at = len(TEXT_IDS)          # the `<timestep>` row
    img_at = t_at + 1             # the first `<img>` row
    return ids, t_at, img_at, n


def run_mini(d: str, device: str, dtype: torch.dtype):
    M, config, model = build(device, dtype)
    tap = Tap()

    ids, t_at, img_at, n = sequence()
    seq = len(ids)
    head_dim = config.attention_head_dim

    # ---- the generalized causal mask (model:2859-2884) --------------------
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    mask[img_at : img_at + n, img_at : img_at + n] = True
    tap.put("mask", mask.to(torch.float32))

    # ---- the 2-D rope, and the (y, x) grid it is built from ---------------
    cos, sin, pos = M.build_2d_rope(
        seq,
        head_dim,
        image_infos=[(slice(img_at, img_at + n), (TOKEN_H, TOKEN_W))],
        base=config.rope_theta,
        return_all_pos=True,
    )
    # `apply_rotary_pos_emb` unsqueezes at dim 1, so cos/sin arrive `[B, S, d]`
    # (what `build_batch_2d_rope` stacks).
    cos = cos[None].to(device=device, dtype=dtype)
    sin = sin[None].to(device=device, dtype=dtype)
    tap.put("rope.cos", cos)
    tap.put("rope.sin", sin)
    tap.put("rope.positions", pos.reshape(seq, 2))   # (y, x), integer

    # ---- the noisy latent -------------------------------------------------
    g = torch.Generator().manual_seed(SEED + 2)
    x = torch.randn(1, config.vae["latent_channels"], TOKEN_H, TOKEN_W, generator=g)
    x = x.to(device=device, dtype=dtype)
    t = torch.tensor([TIMESTEP], device=device, dtype=dtype)
    tap.put("latent", x)
    tap.put("timestep", t)
    tap.put("ids", torch.tensor(ids, dtype=torch.int32))
    tap.put("layout", torch.tensor([seq, t_at, img_at, n, TOKEN_H, TOKEN_W], dtype=torch.int32))

    with torch.no_grad():
        # ---- image.in: patch_embed(x_t, time_embed(t)) --------------------
        t_freq = M.timestep_embedding(t, config.patch_embed_hidden_dim * 0 + 256).to(dtype)
        temb_in = model.time_embed(t)
        temb_out = model.time_embed_2(t)
        temb_tok = model.timestep_emb(t)
        tap.put("tfreq", t_freq)
        tap.put("time_embed", temb_in)
        tap.put("time_embed_2", temb_out)
        tap.put("timestep_emb", temb_tok)

        rows, th, tw = model.patch_embed(x, temb_in)
        assert (th, tw) == (TOKEN_H, TOKEN_W), (th, tw)
        tap.put("image_in.rows", rows[0])            # [n, hidden]

        # ---- the sequence's input rows ------------------------------------
        h = model.model.wte(torch.tensor([ids], device=device))
        h = h.to(dtype).clone()
        h[:, img_at : img_at + n] = rows
        h[:, t_at] = temb_tok
        tap.put("trunk.in", h[0])

        # ---- denoise: the trunk over the whole sequence --------------------
        out = model.model(
            inputs_embeds=h,
            attention_mask=mask[None, None].to(device),
            custom_pos_emb=(cos, sin),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.last_hidden_state
        tap.put("denoise.hidden", hs[0])
        tap.put("denoise.hidden.image", hs[0, img_at : img_at + n])
        tap.put("denoise.hidden.timestep_row", hs[0, t_at])
        for i, layer in enumerate(out.hidden_states):
            tap.put(f"denoise.layer{i}", layer[0])

        # ---- image.out: final_layer(rows, time_embed_2(t)) ------------------
        v = model.final_layer(hs[:, img_at : img_at + n], temb_out, th, tw)
        tap.put("image_out.velocity", v[0])          # [C, h, w]
        tap.put(
            "image_out.velocity.rows",
            v[0].reshape(v.shape[1], -1).transpose(0, 1),   # [n, C], raster order
        )

        # ---- THE UNCONDITIONAL BRANCH -------------------------------------
        # The same denoise step over the reference's own uncond prefix:
        # every PROMPT token replaced by `<cfg>`, the `<bos>` and the three
        # image-meta tokens kept, so the length, the mask and the rotary
        # positions are identical (`tokenization:738-739`). This is the
        # reference's own measurement of how much the canvas is
        # CONDITIONED, and the parity gate reads it rather than guessing a
        # constant: a denoise fire whose image rows attended only
        # themselves would match the conditional dump inside tolerance on a
        # random-init miniature.
        alt = list(ids)
        for i in range(1, t_at - 3):
            alt[i] = CFG_ID
        h_alt = model.model.wte(torch.tensor([alt], device=device)).to(dtype).clone()
        h_alt[:, img_at : img_at + n] = rows
        h_alt[:, t_at] = temb_tok
        out_alt = model.model(
            inputs_embeds=h_alt,
            attention_mask=mask[None, None].to(device),
            custom_pos_emb=(cos, sin),
            use_cache=False,
            return_dict=True,
        )
        hs_alt = out_alt.last_hidden_state
        tap.put("uncond.ids", torch.tensor(alt, dtype=torch.int32))
        tap.put("uncond.hidden.image", hs_alt[0, img_at : img_at + n])
        tap.put("uncond.hidden.timestep_row", hs_alt[0, t_at])
        moved = (hs_alt[0] - hs[0]).norm() / hs[0].norm()
        tap.put("uncond.moved", float(moved))
        print(f"  the prefix conditions the canvas: <cfg> moves it rel {float(moved):.4f}")

        # ---- encode: the causal text pass and its logits --------------------
        text_ids = torch.tensor([ids[: t_at]], device=device)
        tmask = torch.tril(torch.ones(t_at, t_at, dtype=torch.bool))
        tcos, tsin, tpos = M.build_2d_rope(
            t_at, head_dim, image_infos=None, base=config.rope_theta, return_all_pos=True
        )
        enc = model.model(
            input_ids=text_ids,
            attention_mask=tmask[None, None].to(device),
            custom_pos_emb=(
                tcos[None].to(device=device, dtype=dtype),
                tsin[None].to(device=device, dtype=dtype),
            ),
            use_cache=False,
            return_dict=True,
        )
        normed = model.model.ln_f(enc.last_hidden_state)
        tap.put("encode.hidden", enc.last_hidden_state[0])
        tap.put("encode.logits", model.lm_head(normed)[0])
        tap.put("encode.positions", tpos.reshape(t_at, 2))

    # ---- the artifact pie imports -----------------------------------------
    from safetensors.torch import save_file

    sd = {
        k: v.detach().to(torch.bfloat16).contiguous().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith(DROP)
    }
    path = os.path.join(d, "hy3_mini.safetensors")
    save_file(sd, path, metadata={"format": "pt"})
    print(f"  [safetensors] {len(sd)} tensors -> {path}")

    with open(os.path.join(d, "hy3_mini_config.json"), "w") as f:
        json.dump(
            {
                "config": {k: v for k, v in config.to_dict().items() if k != "auto_map"},
                "layout": {
                    "ids": ids,
                    "seq": seq,
                    "timestep_row": t_at,
                    "image_row0": img_at,
                    "image_rows": n,
                    "token_h": TOKEN_H,
                    "token_w": TOKEN_W,
                    "timestep": TIMESTEP,
                    "seed": SEED,
                },
                "tensors": {k: list(v.shape) for k, v in sorted(sd.items())},
            },
            f,
            indent=2,
            default=str,
        )
    stage(d, path, config)
    tap.save(os.path.join(d, "hy3_mini.npz"))
    npz_keys(tap, limit=64)


def stage(d: str, weights: str, config) -> str:
    """The directory `pie model import` reads: the weights, a `config.json`
    (the artifact carries it as `model/config`, and boot refuses one without),
    and the REAL tokenizer (the row's contract pins `<boi>`/`<eoi>`/`<img>` at
    the checkpoint's own ids, so no borrowed vocabulary will do)."""
    out = os.path.join(d, "artifact")
    os.makedirs(out, exist_ok=True)
    link = os.path.join(out, "model.safetensors")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.relpath(weights, out), link)
    cfg = {k: v for k, v in config.to_dict().items() if k != "auto_map"}
    cfg["architectures"] = ["HunyuanImage3ForCausalMM"]
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)
    src = os.environ.get("HY3_TOKENIZER", tokenizer_dir())
    copied = []
    for name in ("tokenizer.json", "tokenizer_config.json"):
        at = os.path.join(src, name) if src else None
        if at and os.path.exists(at):
            shutil.copyfile(at, os.path.join(out, name))
            copied.append(name)
    print(f"  [artifact] {out} ({', '.join(copied) if copied else 'NO TOKENIZER'})")
    if not copied:
        print("    serving needs one: set $HY3_TOKENIZER to a HunyuanImage-3 snapshot")
    return out


def tokenizer_dir() -> str | None:
    """The base repo's snapshot in the HF cache, if it is there."""
    root = os.path.expanduser(
        "~/.cache/huggingface/hub/models--tencent--HunyuanImage-3.0/snapshots"
    )
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        at = os.path.join(root, name)
        if os.path.exists(os.path.join(at, "tokenizer.json")):
            return at
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true", default=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    dtype = getattr(torch, a.dtype)
    d = outdir(MODEL)
    torch.set_grad_enabled(False)
    print(f"== mini == device={device} dtype={dtype}")
    run_mini(d, device, dtype)
    manifest(d, {"source": HY3_SRC, "mini": MINI, "timestep": TIMESTEP, "seed": SEED})


if __name__ == "__main__":
    main()
