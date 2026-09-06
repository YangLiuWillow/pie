"""
Vendored LTX-2.5 reference modules — a self-contained torch transcription.

SOURCE. Every class here is transcribed from the local sglang-diffusion
reference, which is the authority for this family:

    /root/sglang/python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py
        LTX2AudioVideoRotaryPosEmbed, apply_split_rotary_emb, LTX2Attention,
        LTX2FeedForward, LTX2TimestepEmbedder, LTX2AdaLayerNormSingle,
        LTX2TransformerBlock, LTX2VideoTransformer3DModel
    /root/sglang/python/sglang/multimodal_gen/runtime/models/adapter/ltx_2_connector.py
        LTX2Attention (1d), LTX2RotaryPosEmbed1d, LTX2TransformerBlock1d,
        LTX2ConnectorTransformer1d, LTX2TextConnectors
    /root/sglang/python/sglang/multimodal_gen/runtime/layers/visual_embedding.py
        timestep_embedding
    configs/models/dits/{ltx_2,ltx_2_5}.py, configs/models/adapter/ltx_2_connector.py

WHY VENDORED. `import sglang` pulls the whole serving stack (starlette,
orjson, msgspec, ...) and this box has none of it, so the reference classes
cannot be imported (the study says as much: §K "Miniature feasibility").
This file is the same arithmetic with the parallel layers replaced by
`nn.Linear`, the fused CUDA/Triton fast paths replaced by the eager chains
they self-verify against, and the SP/TP/perturbation/cache-dit machinery
dropped. It is NOT a general LTX-2 implementation: it serves the miniature
golden and nothing else.

NAMES. The module tree spells the SHIPPED CHECKPOINT's names, verified
against `Lightricks/LTX-2.5-Diffusers`'s own safetensors index, so that
`state_dict()` here and the shipped `transformer/` + `connectors/` folders
read under one `crates/models/src/ltx_2/import.rs`: `proj_in`, `time_embed`,
`transformer_blocks.N.attn1.to_q`, `ff.net.0.proj`, `video_text_proj_in`.
The checkpoint is not consistent about which side of
`LTX2_PARAM_NAMES_MAPPING` it stands on — the per-block cross-modal tables
carry sglang's names (`video_a2v_cross_attn_scale_shift_table`) while the
four global adaLN heads carry ltx-core's (`av_cross_attn_video_scale_shift`)
— so both spellings appear below, each as the index has it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000, dtype=torch.float32
) -> torch.Tensor:
    """`visual_embedding.timestep_embedding`: `[cos | sin]`, fp32."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=dtype, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def apply_split_rotary_emb(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """`x = [x1 | x2]` halves of the head; `cos`/`sin` are `[B, H, T, R]`."""
    cos, sin = freqs
    x_dtype = x.dtype
    b = x.shape[0]
    _, h, t, _ = cos.shape
    x = x.reshape(b, t, h, -1).transpose(1, 2)

    last = x.shape[-1]
    r = last // 2
    split_x = x.reshape(*x.shape[:-1], 2, r)
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]
    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)
    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]
    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)
    out = out.reshape(*out.shape[:-2], last)
    out = out.transpose(1, 2).reshape(b, t, -1)
    return out.to(dtype=x_dtype)


class LTX2AudioVideoRotaryPosEmbed(nn.Module):
    """The split-form rope, video and audio, transcribed for one batch."""

    def __init__(
        self,
        dim: int,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: Tuple[int, ...] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        num_attention_heads: int = 32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.base_num_frames = int(base_num_frames)
        self.base_height = int(base_height)
        self.base_width = int(base_width)
        self.sampling_rate = int(sampling_rate)
        self.hop_length = int(hop_length)
        self.scale_factors = tuple(int(x) for x in scale_factors)
        self.theta = float(theta)
        self.causal_offset = int(causal_offset)
        self.modality = modality
        self.double_precision = bool(double_precision)
        self.num_attention_heads = int(num_attention_heads)

    def prepare_video_coords(
        self, batch_size, num_frames, height, width, device, fps=24.0
    ) -> torch.Tensor:
        grid_f = torch.arange(0, num_frames, 1, dtype=torch.float32, device=device)
        grid_h = torch.arange(0, height, 1, dtype=torch.float32, device=device)
        grid_w = torch.arange(0, width, 1, dtype=torch.float32, device=device)
        grid = torch.stack(torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij"), dim=0)
        delta = torch.tensor((1, 1, 1), dtype=grid.dtype, device=device)
        patch_ends = grid + delta.view(3, 1, 1, 1)
        latent = torch.stack([grid, patch_ends], dim=-1).flatten(1, 3)
        latent = latent.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        scale = torch.tensor(self.scale_factors, device=device)
        shape = [1] * latent.ndim
        shape[1] = -1
        pixel = latent * scale.view(*shape)
        pixel[:, 0, ...] = (
            pixel[:, 0, ...] + self.causal_offset - self.scale_factors[0]
        ).clamp(min=0)
        pixel[:, 0, ...] = pixel[:, 0, ...] / fps
        return pixel

    def prepare_audio_coords(self, batch_size, num_frames, device) -> torch.Tensor:
        grid_f = torch.arange(0, num_frames, 1, dtype=torch.float32, device=device)
        s = self.scale_factors[0]
        start = (grid_f * s + self.causal_offset - s).clip(min=0)
        end = ((grid_f + 1) * s + self.causal_offset - s).clip(min=0)
        start = start * self.hop_length / self.sampling_rate
        end = end * self.hop_length / self.sampling_rate
        coords = torch.stack([start, end], dim=-1)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)
        return coords

    def forward(self, coords: torch.Tensor, out_dtype=None):
        device = coords.device
        out_dtype = out_dtype or coords.dtype
        num_pos_dims = coords.shape[1]
        if coords.ndim == 4:
            start, end = coords.chunk(2, dim=-1)
            coords = ((start + end) / 2.0).squeeze(-1)
        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        else:
            max_positions = (self.base_num_frames,)
        grid = torch.stack(
            [coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1
        ).to(device)

        n_elem = num_pos_dims * 2
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(
                0.0, 1.0, self.dim // n_elem, dtype=freqs_dtype, device=device
            ),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)
        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs
        freqs = freqs.transpose(-1, -2).flatten(2)

        expected = self.dim // 2
        pad = expected - freqs.shape[-1]
        cos_freq, sin_freq = freqs.cos(), freqs.sin()
        if pad != 0:
            cos_freq = torch.cat([torch.ones_like(cos_freq[:, :, :pad]), cos_freq], -1)
            sin_freq = torch.cat([torch.zeros_like(sin_freq[:, :, :pad]), sin_freq], -1)
        b, t = cos_freq.shape[0], cos_freq.shape[1]
        cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
        sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)
        return (
            torch.swapaxes(cos_freq, 1, 2).to(out_dtype),
            torch.swapaxes(sin_freq, 1, 2).to(out_dtype),
        )


class LTX2RotaryPosEmbed1d(nn.Module):
    """The connectors' 1-D rope: row `i` at `i / base_seq_len`."""

    def __init__(
        self,
        dim: int,
        base_seq_len: int = 4096,
        theta: float = 10000.0,
        double_precision: bool = True,
        num_attention_heads: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.base_seq_len = base_seq_len
        self.theta = theta
        self.double_precision = double_precision
        self.num_attention_heads = num_attention_heads

    def forward(self, batch_size: int, pos: int, device, dtype=None):
        grid_1d = torch.arange(pos, dtype=torch.float32, device=device)
        grid_1d = grid_1d / self.base_seq_len
        grid = grid_1d.unsqueeze(0).repeat(batch_size, 1)
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(0.0, 1.0, self.dim // 2, dtype=freqs_dtype, device=device),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)
        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs
        cos_freq, sin_freq = freqs.cos(), freqs.sin()
        b, t = cos_freq.shape[0], cos_freq.shape[1]
        cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
        sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)
        cos, sin = torch.swapaxes(cos_freq, 1, 2), torch.swapaxes(sin_freq, 1, 2)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# attention and feed-forward
# ---------------------------------------------------------------------------


class LTX2Attention(nn.Module):
    """`to_q|to_k|to_v`, RMSNorm ACROSS heads with a gain, optional split
    rope with separate query and key tables, SDPA, a per-head sigmoid gate
    (`out · 2σ(W·x)`), `to_out.0`."""

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.context_dim = query_dim if context_dim is None else context_dim
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(self.context_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(self.context_dim, self.inner_dim, bias=True)
        self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)
        self.norm_q = nn.RMSNorm(self.inner_dim, eps=norm_eps)
        self.norm_k = nn.RMSNorm(self.inner_dim, eps=norm_eps)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=True)])

    def forward(self, x, context=None, pe=None, k_pe=None, mask=None):
        gate_input = x
        ctx = x if context is None else context
        q = self.to_q(x)
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        q = self.norm_q(q).to(dtype=x.dtype)
        k = self.norm_k(k).to(dtype=x.dtype)
        if pe is not None:
            q = apply_split_rotary_emb(q, pe)
            k = apply_split_rotary_emb(k, pe if k_pe is None else k_pe)
        q = q.unflatten(2, (self.heads, -1)).transpose(1, 2)
        k = k.unflatten(2, (self.heads, -1)).transpose(1, 2)
        v = v.unflatten(2, (self.heads, -1)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        out = out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        gate = self.to_gate_logits(gate_input)
        b, t = out.shape[:2]
        out = out.view(b, t, self.heads, self.dim_head)
        out = out * (2.0 * torch.sigmoid(gate).unsqueeze(-1))
        out = out.view(b, t, self.heads * self.dim_head)
        return self.to_out[0](out)


class GELUProj(nn.Module):
    """diffusers' `GELU` module: a `proj` linear and a tanh gelu."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        return F.gelu(self.proj(x), approximate="tanh")


class LTX2FeedForward(nn.Module):
    """`net.0.proj` up, GELU (tanh), `net.2` down — diffusers' layout."""

    def __init__(self, dim: int, mult: int = 4, bias: bool = True) -> None:
        super().__init__()
        inner = dim * mult
        self.net = nn.ModuleList(
            [GELUProj(dim, inner, bias=bias), nn.Identity(), nn.Linear(inner, dim, bias=bias)]
        )

    def forward(self, x):
        return self.net[2](self.net[0](x))


# ---------------------------------------------------------------------------
# timestep modulation
# ---------------------------------------------------------------------------


class LTX2TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim: int, in_channels: int = 256) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, embedding_dim, bias=True)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, t_emb):
        return self.linear_2(F.silu(self.linear_1(t_emb)))


class CombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = LTX2TimestepEmbedder(embedding_dim, in_channels=256)

    def forward(self, timestep, hidden_dtype=None):
        t = timestep.reshape(-1).to(dtype=torch.float32)
        emb = timestep_embedding(t, dim=256, max_period=10000, dtype=torch.float32)
        if hidden_dtype is not None:
            emb = emb.to(dtype=hidden_dtype)
        return self.timestep_embedder(emb)


class LTX2AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6) -> None:
        super().__init__()
        self.emb = CombinedTimestepSizeEmbeddings(embedding_dim)
        self.linear = nn.Linear(
            embedding_dim, embedding_coefficient * embedding_dim, bias=True
        )

    def forward(self, timestep, hidden_dtype=None):
        embedded = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(F.silu(embedded)), embedded


def rms_no_weight(x, eps):
    return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=eps)


def modulate(x, scale, shift):
    return x * (1 + scale) + shift


def residual_gate_add(residual, update, gate):
    return residual + update * gate


# ---------------------------------------------------------------------------
# the transformer
# ---------------------------------------------------------------------------


@dataclass
class LTX2Config:
    num_layers: int = 48
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    in_channels: int = 128
    out_channels: int = 128
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    cross_attention_dim: int = 4096
    audio_cross_attention_dim: int = 2048
    norm_eps: float = 1e-6
    ff_bias: bool = False
    audio_ff_bias: bool = True
    rope_theta: float = 10000.0
    pos_embed_max_pos: int = 20
    base_height: int = 2048
    base_width: int = 2048
    audio_pos_embed_max_pos: int = 20
    causal_offset: int = 1
    vae_scale_factors: Tuple[int, int, int] = (8, 32, 32)
    audio_scale_factor: int = 4
    audio_sampling_rate: int = 16000
    audio_hop_length: int = 160
    rope_double_precision: bool = True

    @property
    def hidden_size(self):
        return self.num_attention_heads * self.attention_head_dim

    @property
    def audio_hidden_size(self):
        return self.audio_num_attention_heads * self.audio_attention_head_dim


class LTX2TransformerBlock(nn.Module):
    def __init__(self, cfg: LTX2Config) -> None:
        super().__init__()
        d, a = cfg.hidden_size, cfg.audio_hidden_size
        self.norm_eps = cfg.norm_eps
        self.attn1 = LTX2Attention(
            d, None, cfg.num_attention_heads, cfg.attention_head_dim, cfg.norm_eps
        )
        self.audio_attn1 = LTX2Attention(
            a,
            None,
            cfg.audio_num_attention_heads,
            cfg.audio_attention_head_dim,
            cfg.norm_eps,
        )
        self.attn2 = LTX2Attention(
            d,
            cfg.cross_attention_dim,
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            cfg.norm_eps,
        )
        self.audio_attn2 = LTX2Attention(
            a,
            cfg.audio_cross_attention_dim,
            cfg.audio_num_attention_heads,
            cfg.audio_attention_head_dim,
            cfg.norm_eps,
        )
        self.audio_to_video_attn = LTX2Attention(
            d,
            a,
            cfg.audio_num_attention_heads,
            cfg.audio_attention_head_dim,
            cfg.norm_eps,
        )
        self.video_to_audio_attn = LTX2Attention(
            a,
            d,
            cfg.audio_num_attention_heads,
            cfg.audio_attention_head_dim,
            cfg.norm_eps,
        )
        self.ff = LTX2FeedForward(d, 4, bias=cfg.ff_bias)
        self.audio_ff = LTX2FeedForward(a, 4, bias=cfg.audio_ff_bias)

        self.scale_shift_table = nn.Parameter(torch.randn(9, d) / d**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(9, a) / a**0.5)
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, d))
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, a))
        self.prompt_scale_shift_table = nn.Parameter(torch.randn(2, d))
        self.audio_prompt_scale_shift_table = nn.Parameter(torch.randn(2, a))

    @staticmethod
    def _ada(table, batch_size, timestep, indices):
        n = int(table.shape[0])
        values = (
            table[indices].unsqueeze(0).unsqueeze(0).to(timestep)
            + timestep.reshape(batch_size, timestep.shape[1], n, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return [t.squeeze(2) if t.ndim > 3 else t for t in values]

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        temb,
        temb_audio,
        temb_prompt,
        temb_audio_prompt,
        temb_ca_scale_shift,
        temb_ca_audio_scale_shift,
        temb_ca_gate,
        temb_ca_audio_gate,
        video_rotary_emb,
        audio_rotary_emb,
        ca_video_rotary_emb,
        ca_audio_rotary_emb,
    ):
        b = hidden_states.size(0)
        eps = self.norm_eps

        # 1. self-attentions
        vshift, vscale, vgate = self._ada(self.scale_shift_table, b, temb, slice(0, 3))
        h = modulate(rms_no_weight(hidden_states, eps), vscale, vshift)
        hidden_states = residual_gate_add(
            hidden_states, self.attn1(h, pe=video_rotary_emb), vgate
        )

        ashift, ascale, agate = self._ada(
            self.audio_scale_shift_table, b, temb_audio, slice(0, 3)
        )
        h = modulate(rms_no_weight(audio_hidden_states, eps), ascale, ashift)
        audio_hidden_states = residual_gate_add(
            audio_hidden_states, self.audio_attn1(h, pe=audio_rotary_emb), agate
        )

        # 2. prompt cross-attentions, with the context modulated too
        vshift_q, vscale_q, vgate_q = self._ada(
            self.scale_shift_table, b, temb, slice(6, 9)
        )
        p_shift, p_scale = self._ada(
            self.prompt_scale_shift_table, b, temb_prompt, slice(None)
        )
        h = modulate(rms_no_weight(hidden_states, eps), vscale_q, vshift_q)
        c = modulate(encoder_hidden_states, p_scale, p_shift)
        hidden_states = residual_gate_add(hidden_states, self.attn2(h, c), vgate_q)

        ashift_q, ascale_q, agate_q = self._ada(
            self.audio_scale_shift_table, b, temb_audio, slice(6, 9)
        )
        ap_shift, ap_scale = self._ada(
            self.audio_prompt_scale_shift_table, b, temb_audio_prompt, slice(None)
        )
        h = modulate(rms_no_weight(audio_hidden_states, eps), ascale_q, ashift_q)
        c = modulate(audio_encoder_hidden_states, ap_scale, ap_shift)
        audio_hidden_states = residual_gate_add(
            audio_hidden_states, self.audio_attn2(h, c), agate_q
        )

        # 3. the cross-modal pair, both norms taken before either fold
        nv = rms_no_weight(hidden_states, eps)
        na = rms_no_weight(audio_hidden_states, eps)

        vt = self.video_a2v_cross_attn_scale_shift_table
        at = self.audio_a2v_cross_attn_scale_shift_table
        v_ss = (
            vt[:4][None, None] .to(temb_ca_scale_shift)
            + temb_ca_scale_shift.reshape(b, temb_ca_scale_shift.shape[1], 4, -1)
        ).unbind(dim=2)
        v_gate = (
            vt[4:][None, None].to(temb_ca_gate)
            + temb_ca_gate.reshape(b, temb_ca_gate.shape[1], 1, -1)
        ).unbind(dim=2)[0]
        a_ss = (
            at[:4][None, None].to(temb_ca_audio_scale_shift)
            + temb_ca_audio_scale_shift.reshape(
                b, temb_ca_audio_scale_shift.shape[1], 4, -1
            )
        ).unbind(dim=2)
        a_gate = (
            at[4:][None, None].to(temb_ca_audio_gate)
            + temb_ca_audio_gate.reshape(b, temb_ca_audio_gate.shape[1], 1, -1)
        ).unbind(dim=2)[0]
        v_a2v_scale, v_a2v_shift, v_v2a_scale, v_v2a_shift = v_ss
        a_a2v_scale, a_a2v_shift, a_v2a_scale, a_v2a_shift = a_ss

        q_in = modulate(nv, v_a2v_scale, v_a2v_shift)
        kv_in = modulate(na, a_a2v_scale, a_a2v_shift)
        hidden_states = residual_gate_add(
            hidden_states,
            self.audio_to_video_attn(
                q_in, kv_in, pe=ca_video_rotary_emb, k_pe=ca_audio_rotary_emb
            ),
            v_gate,
        )

        q_in = modulate(na, a_v2a_scale, a_v2a_shift)
        kv_in = modulate(nv, v_v2a_scale, v_v2a_shift)
        audio_hidden_states = residual_gate_add(
            audio_hidden_states,
            self.video_to_audio_attn(
                q_in, kv_in, pe=ca_audio_rotary_emb, k_pe=ca_video_rotary_emb
            ),
            a_gate,
        )

        # 4. feed-forwards
        vshift_m, vscale_m, vgate_m = self._ada(
            self.scale_shift_table, b, temb, slice(3, 6)
        )
        h = modulate(rms_no_weight(hidden_states, eps), vscale_m, vshift_m)
        hidden_states = residual_gate_add(hidden_states, self.ff(h), vgate_m)

        ashift_m, ascale_m, agate_m = self._ada(
            self.audio_scale_shift_table, b, temb_audio, slice(3, 6)
        )
        h = modulate(rms_no_weight(audio_hidden_states, eps), ascale_m, ashift_m)
        audio_hidden_states = residual_gate_add(
            audio_hidden_states, self.audio_ff(h), agate_m
        )
        return hidden_states, audio_hidden_states


class LTX2VideoTransformer3DModel(nn.Module):
    def __init__(self, cfg: LTX2Config) -> None:
        super().__init__()
        self.config = cfg
        d, a = cfg.hidden_size, cfg.audio_hidden_size
        self.proj_in = nn.Linear(cfg.in_channels, d, bias=True)
        self.audio_proj_in = nn.Linear(cfg.audio_in_channels, a, bias=True)
        self.time_embed = LTX2AdaLayerNormSingle(d, 9)
        self.audio_time_embed = LTX2AdaLayerNormSingle(a, 9)
        self.prompt_adaln = LTX2AdaLayerNormSingle(d, 2)
        self.audio_prompt_adaln = LTX2AdaLayerNormSingle(a, 2)
        self.av_cross_attn_video_scale_shift = LTX2AdaLayerNormSingle(d, 4)
        self.av_cross_attn_video_a2v_gate = LTX2AdaLayerNormSingle(d, 1)
        self.av_cross_attn_audio_scale_shift = LTX2AdaLayerNormSingle(a, 4)
        self.av_cross_attn_audio_v2a_gate = LTX2AdaLayerNormSingle(a, 1)
        self.scale_shift_table = nn.Parameter(torch.randn(2, d) / d**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, a) / a**0.5)
        self.transformer_blocks = nn.ModuleList(
            [LTX2TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm_out = nn.LayerNorm(d, eps=cfg.norm_eps, elementwise_affine=False)
        self.audio_norm_out = nn.LayerNorm(a, eps=cfg.norm_eps, elementwise_affine=False)
        self.proj_out = nn.Linear(d, cfg.out_channels, bias=True)
        self.audio_proj_out = nn.Linear(a, cfg.audio_out_channels, bias=True)

        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=d,
            base_num_frames=cfg.pos_embed_max_pos,
            base_height=cfg.base_height,
            base_width=cfg.base_width,
            scale_factors=cfg.vae_scale_factors,
            theta=cfg.rope_theta,
            causal_offset=cfg.causal_offset,
            modality="video",
            double_precision=cfg.rope_double_precision,
            num_attention_heads=cfg.num_attention_heads,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=a,
            base_num_frames=cfg.audio_pos_embed_max_pos,
            sampling_rate=cfg.audio_sampling_rate,
            hop_length=cfg.audio_hop_length,
            scale_factors=(cfg.audio_scale_factor,),
            theta=cfg.rope_theta,
            causal_offset=cfg.causal_offset,
            modality="audio",
            double_precision=cfg.rope_double_precision,
            num_attention_heads=cfg.audio_num_attention_heads,
        )
        cross_max = max(cfg.pos_embed_max_pos, cfg.audio_pos_embed_max_pos)
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=cfg.audio_cross_attention_dim,
            base_num_frames=cross_max,
            base_height=cfg.base_height,
            base_width=cfg.base_width,
            theta=cfg.rope_theta,
            causal_offset=cfg.causal_offset,
            modality="video",
            double_precision=cfg.rope_double_precision,
            num_attention_heads=cfg.num_attention_heads,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=cfg.audio_cross_attention_dim,
            base_num_frames=cross_max,
            sampling_rate=cfg.audio_sampling_rate,
            hop_length=cfg.audio_hop_length,
            scale_factors=(cfg.audio_scale_factor,),
            theta=cfg.rope_theta,
            causal_offset=cfg.causal_offset,
            modality="audio",
            double_precision=cfg.rope_double_precision,
            num_attention_heads=cfg.audio_num_attention_heads,
        )

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        audio_timestep=None,
        num_frames=None,
        height=None,
        width=None,
        fps=24.0,
        audio_num_frames=None,
    ):
        b = hidden_states.size(0)
        audio_timestep = timestep if audio_timestep is None else audio_timestep
        device = hidden_states.device

        video_coords = self.rope.prepare_video_coords(
            b, num_frames, height, width, device, fps=fps
        )
        audio_coords = self.audio_rope.prepare_audio_coords(b, audio_num_frames, device)
        dt = hidden_states.dtype
        video_rotary_emb = self.rope(video_coords, out_dtype=dt)
        audio_rotary_emb = self.audio_rope(audio_coords, out_dtype=dt)
        ca_video_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :], out_dtype=dt)
        ca_audio_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], out_dtype=dt
        )

        hidden_states = self.proj_in(hidden_states)
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)

        temb, embedded = self.time_embed(timestep.flatten(), hidden_dtype=dt)
        temb = temb.view(b, -1, temb.size(-1))
        embedded = embedded.view(b, -1, embedded.size(-1))
        temb_audio, audio_embedded = self.audio_time_embed(
            audio_timestep.flatten(), hidden_dtype=dt
        )
        temb_audio = temb_audio.view(b, -1, temb_audio.size(-1))
        audio_embedded = audio_embedded.view(b, -1, audio_embedded.size(-1))

        prompt_timestep = timestep if timestep.ndim <= 1 else timestep.amax(dim=1)
        audio_prompt_timestep = (
            audio_timestep if audio_timestep.ndim <= 1 else audio_timestep.amax(dim=1)
        )
        temb_prompt, _ = self.prompt_adaln(prompt_timestep.flatten(), hidden_dtype=dt)
        temb_prompt = temb_prompt.view(b, -1, temb_prompt.size(-1))
        temb_audio_prompt, _ = self.audio_prompt_adaln(
            audio_prompt_timestep.flatten(), hidden_dtype=dt
        )
        temb_audio_prompt = temb_audio_prompt.view(b, -1, temb_audio_prompt.size(-1))

        ca_ss, _ = self.av_cross_attn_video_scale_shift(
            timestep.flatten(), hidden_dtype=dt
        )
        ca_ss = ca_ss.view(b, -1, ca_ss.shape[-1])
        ca_gate, _ = self.av_cross_attn_video_a2v_gate(
            timestep.flatten(), hidden_dtype=dt
        )
        ca_gate = ca_gate.view(b, -1, ca_gate.shape[-1])
        ca_a_ss, _ = self.av_cross_attn_audio_scale_shift(
            audio_timestep.flatten(), hidden_dtype=dt
        )
        ca_a_ss = ca_a_ss.view(b, -1, ca_a_ss.shape[-1])
        ca_a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            audio_timestep.flatten(), hidden_dtype=dt
        )
        ca_a_gate = ca_a_gate.view(b, -1, ca_a_gate.shape[-1])

        for block in self.transformer_blocks:
            hidden_states, audio_hidden_states = block(
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                temb,
                temb_audio,
                temb_prompt,
                temb_audio_prompt,
                ca_ss,
                ca_a_ss,
                ca_gate,
                ca_a_gate,
                video_rotary_emb,
                audio_rotary_emb,
                ca_video_rotary_emb,
                ca_audio_rotary_emb,
            )

        values = self.scale_shift_table[None, None].to(hidden_states) + embedded[
            :, :, None
        ].to(hidden_states.dtype)
        shift, scale = values[:, :, 0], values[:, :, 1]
        hidden_states = modulate(self.norm_out(hidden_states.float()).to(hidden_states.dtype), scale, shift)
        hidden_states = self.proj_out(hidden_states)

        a_values = self.audio_scale_shift_table[None, None].to(
            audio_hidden_states
        ) + audio_embedded[:, :, None].to(audio_hidden_states.dtype)
        a_shift, a_scale = a_values[:, :, 0], a_values[:, :, 1]
        audio_hidden_states = modulate(
            self.audio_norm_out(audio_hidden_states.float()).to(audio_hidden_states.dtype),
            a_scale,
            a_shift,
        )
        audio_hidden_states = self.audio_proj_out(audio_hidden_states)
        return hidden_states, audio_hidden_states


# ---------------------------------------------------------------------------
# the connectors
# ---------------------------------------------------------------------------


class LTX2TransformerBlock1d(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.attn1 = LTX2Attention(dim, None, heads, head_dim, eps)
        self.ff = LTX2FeedForward(dim, 4, bias=True)

    def forward(self, x, rotary_emb=None):
        x = x + self.attn1(rms_no_weight(x, self.eps), pe=rotary_emb)
        x = x + self.ff(rms_no_weight(x, self.eps))
        return x


class LTX2ConnectorTransformer1d(nn.Module):
    def __init__(
        self,
        heads: int,
        head_dim: int,
        num_layers: int,
        num_registers: int = 128,
        rope_base_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.inner_dim = heads * head_dim
        self.eps = eps
        self.learnable_registers = nn.Parameter(
            torch.rand(num_registers, self.inner_dim) * 2.0 - 1.0
        )
        self.rope = LTX2RotaryPosEmbed1d(
            self.inner_dim,
            base_seq_len=rope_base_seq_len,
            theta=rope_theta,
            num_attention_heads=heads,
        )
        self.transformer_blocks = nn.ModuleList(
            [LTX2TransformerBlock1d(self.inner_dim, heads, head_dim, eps) for _ in range(num_layers)]
        )

    def forward(self, hidden_states):
        """No register substitution: the golden hands a full window, which is
        the reference's own path when nothing is padded (`forward.rs`)."""
        b, seq_len, _ = hidden_states.shape
        rotary = self.rope(b, seq_len, hidden_states.device, hidden_states.dtype)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb=rotary)
        return rms_no_weight(hidden_states, self.eps)


@dataclass
class LTX2ConnectorConfig:
    caption_channels: int = 3840
    text_proj_in_factor: int = 49
    video_heads: int = 32
    video_head_dim: int = 128
    video_layers: int = 8
    audio_heads: int = 32
    audio_head_dim: int = 64
    audio_layers: int = 8
    num_registers: int = 128
    rope_base_seq_len: int = 4096
    rope_theta: float = 10000.0
    eps: float = 1e-6


class LTX2TextConnectors(nn.Module):
    def __init__(self, cfg: LTX2ConnectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        feature_in = cfg.caption_channels * cfg.text_proj_in_factor
        video_dim = cfg.video_heads * cfg.video_head_dim
        audio_dim = cfg.audio_heads * cfg.audio_head_dim
        self.video_text_proj_in = nn.Linear(feature_in, video_dim, bias=True)
        self.audio_text_proj_in = nn.Linear(feature_in, audio_dim, bias=True)
        self.video_connector = LTX2ConnectorTransformer1d(
            cfg.video_heads,
            cfg.video_head_dim,
            cfg.video_layers,
            cfg.num_registers,
            cfg.rope_base_seq_len,
            cfg.rope_theta,
            cfg.eps,
        )
        self.audio_connector = LTX2ConnectorTransformer1d(
            cfg.audio_heads,
            cfg.audio_head_dim,
            cfg.audio_layers,
            cfg.num_registers,
            cfg.rope_base_seq_len,
            cfg.rope_theta,
            cfg.eps,
        )

    @staticmethod
    def _rescale(x, target_dim, source_dim):
        return x * math.sqrt(target_dim / source_dim)

    def forward(self, text_hidden_states):
        source = self.cfg.caption_channels
        v = self._rescale(
            text_hidden_states, self.video_text_proj_in.out_features, source
        )
        a = self._rescale(
            text_hidden_states, self.audio_text_proj_in.out_features, source
        )
        v = self.video_text_proj_in(v)
        a = self.audio_text_proj_in(a)
        return self.video_connector(v), self.audio_connector(a)


def pack_text_embeds_v2(text_hidden_states: torch.Tensor, eps: float = 1e-6):
    """`[B, L, hidden, layers] -> [B, L, hidden*layers]`, per-token per-layer
    RMS-normalised then flattened LAYER-FASTEST."""
    variance = torch.mean(text_hidden_states**2, dim=2, keepdim=True)
    return (text_hidden_states * torch.rsqrt(variance + eps)).flatten(2)
