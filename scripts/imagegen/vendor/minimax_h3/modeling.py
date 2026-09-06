"""A dependency-free transcription of MiniMax H3's DiT, for the miniature golden.

SOURCE
------
`sglang` at commit **6cee9285a3dd43e1f0270818aef7bf01a0568863**, files

    python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py
    python/sglang/multimodal_gen/configs/models/dits/minimax_h3.py

Importing that package needs sglang's full serving environment (starlette,
its kernels, its distributed layers) which is not installed in this
checkout, so the arithmetic every path below reproduces is transcribed
rather than imported. Only the EAGER branches are kept: no tensor
parallelism (`get_tp_world_size() == 1`), no quantization, no fused CUDA
kernels, no adaLN cache, no sparse/ring/Ulysses attention, no MPS
chunking, no Cache-DiT. Every one of those is a fast path the reference
falls back FROM, so what is left is the reference's own formula.

Line-anchored correspondences (upstream file : line):

    _norm                     minimax_h3.py:353-357
    _rotate_half              minimax_h3.py:361-363
    _modulate_scale_shift     minimax_h3.py:366-386   (eager tail)
    _modulate_gate            minimax_h3.py:389-410   (eager tail)
    _silu_mul                 minimax_h3.py:413-424   (eager tail)
    MiniMaxH3Rope             minimax_h3.py:457-483
    _rope_cos_sin_cache       minimax_h3.py:485-496
    _apply_rope               minimax_h3.py:540-549
    TimeEmbedder              minimax_h3.py:552-604
    Attention (eager)         minimax_h3.py:736-1124
    MLP                       minimax_h3.py:1127-1178
    AdalnProj                 minimax_h3.py:1181-1238
    TokenRefinerBlock         minimax_h3.py:1240-1320
    DiTBlock                  minimax_h3.py:1323-1423
    FinalLayer                minimax_h3.py:1426-1532

The packed-sequence layout (`packed_sequence.py`) is NOT transcribed: the
golden builds its own rows, positions and tags, because that table is
guest data on pie's side (`crates/models/src/minimax_h3/forward.rs`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

MODALITY_NUM = 3  # MINIMAX_H3_ADALN_MODALITY_NUM
ADALN_SLICES = 6
FINAL_SLICES = 2


@dataclass
class Arch:
    """`MiniMaxH3DiTArchConfig`, the fields the DiT reads."""

    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.latents_dim * pt * ph * pw

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def adaln_out_features(self) -> int:
        return ADALN_SLICES * self.hidden_size * MODALITY_NUM

    @property
    def final_adaln_out_features(self) -> int:
        return FINAL_SLICES * self.hidden_size

    @property
    def rope_dim(self) -> int:
        return 6 * self.rope_inv_freq_len


# The study's D.3 miniature: two blocks, one refiner, 64-wide heads,
# eight frequencies per axis (48 of 64 rotated).
MINI = Arch(
    hidden_size=128,
    num_layers=2,
    token_refiner_num_layers=1,
    num_attention_heads=2,
    attention_head_dim=64,
    ffn_hidden_size=256,
    text_dim=64,
    timestep_input_dim=32,
    time_embed_hidden_size=128,
    time_embed_dim=64,
    rope_inv_freq_len=8,
)


def _norm(size: int, *, eps: float) -> nn.RMSNorm:
    """minimax_h3.py:353-357 — affine RMSNorm, fp32 accumulation."""
    return nn.RMSNorm(size, eps=eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """`x · (1 + scale[idx]) + shift[idx]` (minimax_h3.py:366-386)."""
    return x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """`x + gate[idx] · other` (minimax_h3.py:389-410)."""
    return x + gate.index_select(0, indices) * other


def _silu_mul(hidden: torch.Tensor) -> torch.Tensor:
    gate, up = hidden.chunk(2, dim=-1)
    return nn.functional.silu(gate) * up


class Rope(nn.Module):
    """minimax_h3.py:457-483 — 3-D rope over `(t, h, w)`.

    `inv_freq[i] = base^(-2i / (2·len))`; the shipped checkpoint's buffer
    is `10000^(-2i/32)` to seven digits, which is what `base` defaults to.
    """

    def __init__(self, inv_freq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        i = torch.arange(0, 2 * inv_freq_len, 2, dtype=torch.float32)
        self.register_buffer("inv_freq", base ** (-i / (2 * inv_freq_len)), persistent=True)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """`[1, S, 3]` -> `[S, 6·len]`."""
        pos = position_ids[0].to(torch.float32)
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)


def rope_cos_sin_cache(freqs: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """minimax_h3.py:485-496 — `[cos(first half) | sin(first half)]`."""
    half = freqs.shape[-1] // 2
    return torch.cat(
        (torch.cos(freqs[:, :half]), torch.sin(freqs[:, :half])), dim=-1
    ).to(dtype=dtype)


def apply_rope(x: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    """minimax_h3.py:540-549 — rotate the cached prefix, pass the tail."""
    half = cos_sin_cache.shape[-1] // 2
    cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


class TimeEmbedder(nn.Module):
    """minimax_h3.py:552-604 — `[cos | sin]` sinusoid then a two-layer MLP."""

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = nn.Linear(arch.timestep_input_dim, arch.time_embed_hidden_size, bias=True)
        self.proj_out = nn.Linear(arch.time_embed_hidden_size, arch.time_embed_dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.to(torch.float32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(nn.functional.silu(self.proj_in(t_freq)))


class AdalnProj(nn.Module):
    """minimax_h3.py:1181-1238 — `[M, t_dim] -> [M·modalities, ratio·H]`."""

    def __init__(self, arch: Arch, out_features: int, *, expand_ratio: int, modality_num: int):
        super().__init__()
        assert out_features == expand_ratio * arch.hidden_size * modality_num
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.linear = nn.Linear(arch.time_embed_dim, out_features, bias=True)

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.linear(adaln_input)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MLP(nn.Module):
    """minimax_h3.py:1127-1178 — `fc1` lands `[gate | up]`, no biases."""

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        self.fc1 = nn.Linear(arch.hidden_size, 2 * arch.ffn_hidden_size, bias=False)
        self.fc2 = nn.Linear(arch.ffn_hidden_size, arch.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_silu_mul(self.fc1(x)))


class Attention(nn.Module):
    """minimax_h3.py:736-1124, the eager path.

    Fused `qkv_proj` (no bias), per-head RMSNorm on q and k, the rope where
    there is a cache, non-causal varlen attention over the `cu_seqlens`
    segments, `out_proj` (no bias).
    """

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        self.inner_dim = arch.inner_dim
        self.softmax_scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(arch.hidden_size, 3 * self.inner_dim, bias=False)
        self.q_norm = _norm(self.head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(self.head_dim, eps=arch.qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, arch.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor | None,
        cu_seqlens: list[int],
    ) -> torch.Tensor:
        rows = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.inner_dim, dim=-1)
        shape = (rows, self.num_heads, self.head_dim)
        q = self.q_norm(q.view(shape))
        k = self.k_norm(k.view(shape))
        if rope_cache is not None:
            q = apply_rope(q, rope_cache)
            k = apply_rope(k, rope_cache)
        v = v.view(shape)
        out = torch.empty_like(q)
        for lo, hi in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            if lo == hi:
                continue
            # `[1, heads, span, head_dim]`, non-causal, no mask.
            qs = q[lo:hi].transpose(0, 1).unsqueeze(0)
            ks = k[lo:hi].transpose(0, 1).unsqueeze(0)
            vs = v[lo:hi].transpose(0, 1).unsqueeze(0)
            o = nn.functional.scaled_dot_product_attention(
                qs, ks, vs, is_causal=False, scale=self.softmax_scale
            )
            out[lo:hi] = o[0].transpose(0, 1)
        return self.out_proj(out.reshape(rows, self.inner_dim))


class TokenRefinerBlock(nn.Module):
    """minimax_h3.py:1240-1280 — pre-norm, no adaLN, no rope."""

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = Attention(arch)
        self.mlp = MLP(arch)

    def forward(self, x: torch.Tensor, *, cu_seqlens: list[int]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cache=None, cu_seqlens=cu_seqlens)
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):
    """minimax_h3.py:1282-1320."""

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TokenRefinerBlock(arch) for _ in range(arch.token_refiner_num_layers)
        )
        self.final_norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)

    def forward(self, x: torch.Tensor, *, cu_seqlens: list[int]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    """minimax_h3.py:1323-1423."""

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = Attention(arch)
        self.mlp = MLP(arch)
        self.adaln_proj = AdalnProj(
            arch,
            arch.adaln_out_features,
            expand_ratio=ADALN_SLICES,
            modality_num=MODALITY_NUM,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: torch.Tensor,
        cu_seqlens: list[int],
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(
            adaln_input
        )
        residual = x
        h = _modulate_scale_shift(self.norm1(x), shift_msa, scale_msa, combined_indices)
        h = self.attn(h, rope_cache=rope_cache, cu_seqlens=cu_seqlens)
        x = _modulate_gate(residual, gate_msa, h, combined_indices)

        residual = x
        h = _modulate_scale_shift(self.norm2(x), shift_mlp, scale_mlp, combined_indices)
        h = self.mlp(h)
        return _modulate_gate(residual, gate_mlp, h, combined_indices)


class FinalLayer(nn.Module):
    """minimax_h3.py:1426-1532 — one modality, both heads on every row."""

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        self.norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = AdalnProj(
            arch, arch.final_adaln_out_features, expand_ratio=FINAL_SLICES, modality_num=1
        )
        self.video_out = nn.Linear(arch.hidden_size, arch.video_patch_dim, bias=True)
        self.audio_out = nn.Linear(arch.hidden_size, arch.audio_latents_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        h = _modulate_scale_shift(self.norm(x), shift, scale, inverse_indices)
        return self.video_out(h), self.audio_out(h)


class MiniMaxH3DiT(nn.Module):
    """The whole transformer over ONE packed row sequence.

    `forward` takes the rows already placed: the caller owns the layout
    (which is where pie puts it too — the guest builds the positions, the
    tags and the row order).
    """

    def __init__(self, arch: Arch) -> None:
        super().__init__()
        self.arch = arch
        self.video_patch_proj = nn.Linear(arch.video_patch_dim, arch.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(arch.audio_latents_dim, arch.hidden_size, bias=True)
        self.condition_proj = nn.Linear(arch.text_dim, arch.hidden_size, bias=True)
        self.time_embedder = TimeEmbedder(arch)
        self.rope = Rope(arch.rope_inv_freq_len)
        self.token_refiner = TokenRefiner(arch)
        self.blocks = nn.ModuleList(DiTBlock(arch) for _ in range(arch.num_layers))
        self.final_layer = FinalLayer(arch)

    def refine_prompt_embeds(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """minimax_h3.py:2077-2110 — run ONCE per request."""
        rows = text_hidden.shape[0]
        return self.token_refiner(self.condition_proj(text_hidden), cu_seqlens=[0, rows])

    def forward(
        self,
        *,
        refined_text: torch.Tensor,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        reference_rows: torch.Tensor,
        position_ids: torch.Tensor,
        token_tags: torch.Tensor,
        inverse_indices: torch.Tensor,
        unique_timesteps: torch.Tensor,
        order: tuple[str, ...] = ("text", "video", "audio", "reference"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Answer `(video velocity, audio velocity)` on the rows that keep them.

        `order` names the packed row order; `position_ids`, `token_tags`
        and `inverse_indices` are already in it.
        """
        arch = self.arch
        pieces = {
            "text": refined_text,
            "video": self.video_patch_proj(video_rows),
            "audio": self.audio_patch_proj(audio_rows),
            "reference": self.video_patch_proj(reference_rows),
        }
        x = torch.cat([pieces[name] for name in order], dim=0)
        spans, at = {}, 0
        for name in order:
            spans[name] = (at, at + pieces[name].shape[0])
            at += pieces[name].shape[0]
        rows = x.shape[0]
        assert position_ids.shape[1] == rows, (position_ids.shape, rows)

        adaln_input = nn.functional.silu(self.time_embedder(unique_timesteps))
        combined = inverse_indices * MODALITY_NUM + token_tags
        rope_cache = rope_cos_sin_cache(self.rope(position_ids), dtype=x.dtype)
        for block in self.blocks:
            x = block(
                x,
                adaln_input=adaln_input,
                combined_indices=combined,
                rope_cache=rope_cache,
                cu_seqlens=[0, rows],
            )
        video, audio = self.final_layer(
            x, adaln_input=adaln_input, inverse_indices=inverse_indices
        )
        vlo, vhi = spans["video"]
        alo, ahi = spans["audio"]
        _ = arch
        return video[vlo:vhi], audio[alo:ahi]


def interleave_qkv(weight: torch.Tensor, *, heads: int, head_dim: int) -> torch.Tensor:
    """`[Q | K | V]` back into the official `[q_h | k_h | v_h]` per head.

    The inverse of `_reorder_grouped_qkv_to_qkv` (minimax_h3.py:223-252) at
    `heads_per_group = 1`: the shipped safetensors interleave each head's
    three blocks, and every loader — pie's `import.rs` included — undoes it.
    """
    rest = weight.shape[1:]
    q, k, v = weight.split(heads * head_dim, dim=0)
    grouped = torch.stack(
        [
            q.reshape(heads, head_dim, *rest),
            k.reshape(heads, head_dim, *rest),
            v.reshape(heads, head_dim, *rest),
        ],
        dim=1,
    )
    return grouped.reshape(3 * heads * head_dim, *rest).contiguous()
