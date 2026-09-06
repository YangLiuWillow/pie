#!/usr/bin/env python3
"""
h3_golden.py -- reference dump for MiniMax H3 (M5).

    python h3_golden.py --mini      # CPU, seconds, no checkpoint

Outputs -> $PIE_IMAGEGEN_GOLDEN/minimax_h3/
    h3_mini.safetensors    a random-init MiniMaxH3DiT at the study's D.3
                           miniature config, written with the OFFICIAL tensor
                           spellings (and the official per-head-interleaved
                           `qkv_proj`), so `crates/models/src/minimax_h3/
                           import.rs` reads it by the same path it reads the
                           66 GB flagship
    h3_mini.npz            the step-0 inputs (video / audio / reference latent
                           rows, the raw text hidden rows, the `(t, h, w)`
                           position table, the modality tags, the per-row
                           timestep index and the step's four unique
                           timesteps) and the two velocities it answers
    h3_mini_config.json    the config, the row layout, the parameter count

WHY THE REFERENCE IS VENDORED
-----------------------------
`sglang.multimodal_gen` cannot be imported here (its `__init__` pulls in
starlette and the rest of the serving stack, none of which is installed),
so `vendor/minimax_h3/modeling.py` transcribes the eager arithmetic from
sglang commit 6cee9285a3dd43e1f0270818aef7bf01a0568863 with line anchors
per class. That file's header is the audit trail.

THE ROW LAYOUT THIS GOLDEN USES
-------------------------------
The reference packs `[text | keyframes/refs | audio | video | pad]`; pie
submits the same rows as four lanes of one attention group, which pack by
stream code as `[text | video | audio | reference]` and carry no pad. The
attention is unmasked inside the group and every row's rotary coordinates
travel with it, so the two orders answer identically — which this script
CHECKS (`--mini` runs both orders and reports the cosine) rather than
assumes, and then dumps pie's order so the parity harness compares row for
row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from golden_common import Tap, manifest, npz_keys, outdir  # noqa: E402
from vendor.minimax_h3 import MINI, MiniMaxH3DiT, interleave_qkv  # noqa: E402

MODEL = "minimax_h3"
REPO = "MiniMaxAI/MiniMax-H3"
SEED = 0

# The miniature's job: a three-row prompt, a 2 x 4 x 4 video latent grid
# under the (1, 2, 2) patch (2 x 2 x 2 = 8 video rows), three audio latent
# ticks in stereo (6 rows), and one keyframe (4 rows: one latent frame of
# the same 2 x 2 patch grid).
TEXT_ROWS = 3
LATENT_T, LATENT_H, LATENT_W = 2, 4, 4
AUDIO_T = 3
KEYFRAMES = 1

# The four timesteps of one step, in the slot order pie's `timestep` port
# states: video, the visual condition rows' pinned max(t, 0.999), audio,
# and the ref2va audio-reference slot no FL2VA lane claims.
UNIQUE_TIMESTEPS = [0.35, 0.999, 0.62, 1.0]
# The modality tag of each row class (presentation.py:26-27): visual 0,
# text 1, audio 2.
TAGS = {"text": 1, "video": 0, "audio": 2, "reference": 0}
# Which unique timestep each row class runs at (denoise_loop.py:369-418).
SLOTS = {"text": 0, "video": 0, "audio": 2, "reference": 1}
# pie's packed order; the reference's is `("text", "reference", "audio", "video")`.
PIE_ORDER = ("text", "video", "audio", "reference")
REF_ORDER = ("text", "reference", "audio", "video")


def rows_of(arch) -> dict[str, int]:
    ph, pw = arch.patch_size[1], arch.patch_size[2]
    video = LATENT_T * (LATENT_H // ph) * (LATENT_W // pw)
    return {
        "text": TEXT_ROWS,
        "video": video,
        "audio": 2 * AUDIO_T,
        "reference": KEYFRAMES * (LATENT_H // ph) * (LATENT_W // pw),
    }


def positions(arch) -> dict[str, np.ndarray]:
    """The `(t, h, w)` table, one block per row class.

    A faithful miniature of study §C.2 without its fp64 aspect grids: text
    is a 1-D prefix on `t`; a video patch of latent frame `k` sits at
    `TEXT_ROWS + 5/3 · Σ FRAME_PER_TOKEN` on a small integer `h`/`w` grid;
    an audio latent at tick `i` sits at `TEXT_ROWS + i` with `h = 0` and
    `w` pinned to the left or right extreme of that grid; the keyframe
    shares the target's grid at the FIRST video token's `t`.
    """
    ph, pw = arch.patch_size[1], arch.patch_size[2]
    hg, wg = LATENT_H // ph, LATENT_W // pw
    frame_per_token = (1, 4, 4, 4, 4)
    spans, cursor = [], float(TEXT_ROWS)
    for k in range(LATENT_T):
        spans.append(cursor)
        cursor += 5.0 / 3.0 * frame_per_token[k % 5]

    text = np.stack(
        [np.arange(TEXT_ROWS, dtype=np.float32), np.zeros(TEXT_ROWS), np.zeros(TEXT_ROWS)],
        axis=-1,
    ).astype(np.float32)

    video = np.array(
        [[spans[k], float(h), float(w)] for k in range(LATENT_T) for h in range(hg) for w in range(wg)],
        dtype=np.float32,
    )
    # Channel-major stereo (packed_tokens.py:73-97): every left-channel row
    # first, then every right-channel row; `w` is the pinned extreme.
    audio = np.array(
        [
            [float(TEXT_ROWS + i), 0.0, float(0 if c == 0 else wg - 1)]
            for c in range(2)
            for i in range(AUDIO_T)
        ],
        dtype=np.float32,
    )
    reference = np.array(
        [[spans[0], float(h), float(w)] for _ in range(KEYFRAMES) for h in range(hg) for w in range(wg)],
        dtype=np.float32,
    )
    return {"text": text, "video": video, "audio": audio, "reference": reference}


def table(order, per_class: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([per_class[name] for name in order], axis=0)


def run_mini(d: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
    arch = MINI
    torch.manual_seed(SEED)
    model = MiniMaxH3DiT(arch).to(device=device, dtype=dtype).eval()
    g = torch.Generator().manual_seed(SEED)
    for _, p in sorted(model.named_parameters()):
        p.data = (0.02 * torch.randn(p.shape, generator=g)).to(device=device, dtype=dtype)

    counts = rows_of(arch)
    pos = positions(arch)
    gi = torch.Generator().manual_seed(1234)

    def randn(*shape):
        return torch.randn(*shape, generator=gi).to(device=device, dtype=dtype)

    text_hidden = randn(counts["text"], arch.text_dim)
    video_rows = randn(counts["video"], arch.video_patch_dim)
    audio_rows = randn(counts["audio"], arch.audio_latents_dim)
    reference_rows = randn(counts["reference"], arch.video_patch_dim)
    unique = torch.tensor(UNIQUE_TIMESTEPS, device=device, dtype=dtype)

    with torch.no_grad():
        refined = model.refine_prompt_embeds(text_hidden)

    answers = {}
    for tag, order in (("pie", PIE_ORDER), ("ref", REF_ORDER)):
        position_ids = torch.from_numpy(table(order, pos)).to(device=device, dtype=dtype)[None]
        tags = torch.tensor(
            [TAGS[name] for name in order for _ in range(counts[name])],
            device=device,
            dtype=torch.long,
        )
        inverse = torch.tensor(
            [SLOTS[name] for name in order for _ in range(counts[name])],
            device=device,
            dtype=torch.long,
        )
        with torch.no_grad():
            video_v, audio_v = model(
                refined_text=refined,
                video_rows=video_rows,
                audio_rows=audio_rows,
                reference_rows=reference_rows,
                position_ids=position_ids,
                token_tags=tags,
                inverse_indices=inverse,
                unique_timesteps=unique,
                order=order,
            )
        answers[tag] = (video_v, audio_v, position_ids[0], tags, inverse)

    # The claim `crates/models/src/minimax_h3/forward.rs` makes about the
    # row order, checked rather than assumed.
    for which, index in (("video", 0), ("audio", 1)):
        a = answers["pie"][index].flatten().double()
        b = answers["ref"][index].flatten().double()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        print(f"  row order {which}: cos(pie order, reference order) = {cos:.8f}")
        assert cos > 1 - 1e-6, f"{which}: the packed order changed the answer"

    tap = Tap()
    video_v, audio_v, position_ids, tags, inverse = answers["pie"]
    tap.put("mini.in.text_hidden", text_hidden)
    tap.put("mini.in.refined_text", refined)
    tap.put("mini.in.video_rows", video_rows)
    tap.put("mini.in.audio_rows", audio_rows)
    tap.put("mini.in.reference_rows", reference_rows)
    tap.put("mini.in.unique_timesteps", unique)
    tap.put("mini.in.positions", position_ids)
    tap.put("mini.in.token_tags", tags.to(torch.float32))
    tap.put("mini.in.inverse_indices", inverse.to(torch.float32))
    tap.put("mini.out.video", video_v)
    tap.put("mini.out.audio", audio_v)
    tap.save(os.path.join(d, "h3_mini.npz"))
    npz_keys(tap)

    # The checkpoint, in the OFFICIAL spellings: `qkv_proj` re-interleaved
    # per head so pie's import runs its de-interleave, everything else
    # named as `FL2VA/transformer/`'s index names it.
    from safetensors.torch import save_file

    state = {}
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().contiguous().float().cpu()
        if name.endswith("attn.qkv_proj.weight"):
            value = interleave_qkv(
                value, heads=arch.num_attention_heads, head_dim=arch.attention_head_dim
            )
        state[name] = value
    path = os.path.join(d, "h3_mini.safetensors")
    save_file(state, path, metadata={"format": "pt"})
    print(f"  [safetensors] {len(state)} tensors -> {path}")

    layout = {
        name: {
            "rows": counts[name],
            "tag": TAGS[name],
            "timestep_slot": SLOTS[name],
        }
        for name in PIE_ORDER
    }
    with open(os.path.join(d, "h3_mini_config.json"), "w") as f:
        json.dump(
            {
                "arch": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in vars(arch).items()
                },
                "order": list(PIE_ORDER),
                "reference_order": list(REF_ORDER),
                "layout": layout,
                "unique_timesteps": UNIQUE_TIMESTEPS,
                "latent": {"t": LATENT_T, "h": LATENT_H, "w": LATENT_W, "audio_t": AUDIO_T},
                "num_parameters": int(sum(v.numel() for v in state.values())),
                "tensors": {k: list(v.shape) for k, v in state.items()},
            },
            f,
            indent=2,
        )
    total = sum(counts.values())
    print(f"  mini: {sum(v.numel() for v in state.values())} params, {total} packed rows {counts}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mini", action="store_true", default=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    d = outdir(MODEL)
    torch.set_grad_enabled(False)
    print("== mini ==")
    run_mini(d, args.device)
    manifest(
        d,
        {
            "repo": REPO,
            "partition": "FL2VA",
            "seed": SEED,
            "reference": "sglang 6cee9285a3dd43e1f0270818aef7bf01a0568863, vendored",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
