//! The `wan_2` declaration: every dimension a Rust constant or a [`Dims`]
//! field, every weight named in the plan's own scheme.
//!
//! Three components under one plan (design D5): the umT5-xxl encoder
//! ([`TextEncoder`], `te.`), the Wan 2.2 transformer ([`Dit`], `dit.`) and
//! the Wan 2.2 VAE decoder ([`Vae`], `vae.`). The numbers are the
//! `Wan-AI/Wan2.2-TI2V-5B-Diffusers` snapshot's `transformer/config.json`,
//! `text_encoder/config.json` and `vae/config.json` (study
//! `.wiki/imagegen/study/wan22.md` §C, §D.1), restated here because a
//! family's dims are Rust constants and a `config.json` is carried, never
//! read. The miniatures are `scripts/imagegen/wan22_golden.py`'s
//! `MINI_CFGS`: the transformer alone, two heads, two blocks, random
//! weights — `d128` at the real 128-wide head (rope split `[44, 42, 42]`)
//! and `nano` at a 24-wide head (`[8, 8, 8]`, a different split on
//! purpose).
//!
//! # Numerics contract
//!
//! * Banks are bf16 (the snapshot's transformer and VAE ship fp32 and are
//!   cast at import; the encoder ships bf16). Token-row activations are
//!   bf16; matmuls accumulate fp32 and round once.
//! * The three `FP32LayerNorm`s of a block and the head's are
//!   `elementwise.layernorm{_no_scale}`: fp32 statistics, one rounding at
//!   the store; the modulation that follows reads the bf16 row and an f32
//!   vector and rounds once more (the fused `NormModulate` arm of the CUDA
//!   shell folds the two into one rounding, which is the reference's).
//! * The QK norms are `torch.nn.RMSNorm` over the WHOLE `heads·head_dim`
//!   row (`qk_norm = "rms_norm_across_heads"`): `elementwise.rmsnorm` with
//!   a `[dim]` gain, fp32 sum, one rounding.
//! * The timestep chain (sinusoid → `time_embedder` → `time_proj`) is a
//!   lane-vector chain and stays fp32 end to end; the reference runs the
//!   sinusoid and `time_embedder` in fp32 (`_keep_in_fp32_modules`) and
//!   `time_proj` in bf16, so a port is the more precise side. The
//!   `scale_shift_table`s are added in fp32, as the reference does.
//! * The encoder's `wo` runs bf16 here where HF keeps it fp32
//!   (`_keep_in_fp32_modules = ["wo"]`): one bf16 rounding of the gated
//!   activation before the GEMM.
//! * The VAE decodes in bf16 rows (fp32 accumulation) where the reference
//!   pipeline forces fp32 (study §A.4); `spatial.conv3d` is bf16-in by
//!   contract (`IMAGEGEN_CONTRACT.md` §6). Its RMS norms (`WanRMS_norm`:
//!   `F.normalize(x, dim=C) · √C · γ`) are `elementwise.rmsnorm` over the
//!   channel row at [`VAE_EPS`].

use model_dsl::{Dtype, Weight};

/// The DiT patch: one token is `1 × 2 × 2` latent cells.
pub const PATCH_T: u32 = 1;
pub const PATCH_H: u32 = 2;
pub const PATCH_W: u32 = 2;
/// `PATCH_T · PATCH_H · PATCH_W`.
pub const PATCH_VOL: u32 = PATCH_T * PATCH_H * PATCH_W;

/// Pixels per latent cell along height and width (the Wan 2.2 VAE's
/// stride, the 2×2 pixel patchify inside it included), and frames per
/// latent cell along time.
pub const VAE_SPATIAL_COMPRESSION: u32 = 16;
pub const VAE_TEMPORAL_COMPRESSION: u32 = 4;

/// The sinusoidal timestep embedding: diffusers' `Timesteps(freq_dim,
/// flip_sin_to_cos=True, downscale_freq_shift=0)` is `[cos | sin]` at
/// `max_period` 10 000; the scheduler timestep (`σ·1000`) goes in raw.
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;

/// `eps` of every norm in the transformer (the three `FP32LayerNorm`s, the
/// across-heads QK RMSNorms, the head's norm).
pub const NORM_EPS: f32 = 1e-6;
/// Three rotary axes `(t, h, w)` at θ 10 000, adjacent pairs
/// (`RopeForm::Interleaved`: diffusers' `apply_rotary_emb` over
/// `unflatten(-1, (-1, 2))`).
pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_AXES: u8 = 3;

/// How many `[dim]` slices a block's modulation vector carries:
/// `(shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa,
/// c_gate_msa)` in the checkpoint; the plan's order swaps each
/// `(shift, scale)` pair.
pub const MOD_SLICES: u32 = 6;
/// The head's `(shift, scale)` table.
pub const HEAD_SLICES: u32 = 2;

/// `UniPCMultistepScheduler(flow_shift=5.0, num_train_timesteps=1000,
/// use_flow_sigmas=True)`: TI2V-5B's `scheduler_config.json`.
pub const TRAIN_STEPS: u32 = 1000;
pub const SHIFT_TI2V: f32 = 5.0;

/// The text conditioning is always 512 rows: the encoder's rows truncated
/// to the prompt's real length and zero-padded to 512, attended WITHOUT a
/// mask (study §C.6, §I.9). The `context` port takes exactly what the
/// reference hands the transformer, so a guest allocates 512 rows and
/// zero-fills past the `text` readout's length (`forward.rs`).
pub const CONTEXT_LEN: u32 = 512;

/// The text encoder's numbers (`text_encoder/config.json`, umT5-xxl,
/// encoder only).
pub const TE_HIDDEN: u32 = 4096;
pub const TE_VOCAB: u32 = 256_384;
pub const TE_HEADS: u32 = 64;
pub const TE_HEAD_DIM: u32 = 64;
pub const TE_INTER: u32 = 10_240;
pub const TE_LAYERS: u32 = 24;
pub const TE_EPS: f32 = 1e-6;
/// `relative_attention_num_buckets` / `relative_attention_max_distance`:
/// every umT5 layer owns its own bucket embedding (T5 shares layer 0's).
pub const TE_BUCKETS: u32 = 32;
pub const TE_MAX_DISTANCE: f32 = 128.0;
/// `max_sequence_length`: the truncation bound of a prompt, and the width
/// the relative-bias tables answer exactly.
pub const TE_MAX_TOKENS: u32 = 512;

/// `AutoencoderKLWan` 2.2 (`vae/config.json`): `z_dim` 48, 12 pixel
/// channels (RGB through a 2×2 space-to-depth), `decoder_base_dim` 256
/// under `dim_mult [1, 2, 4, 4]`, `num_res_blocks` 2 (three resnets per up
/// block), `is_residual` (the `DupUp3D` shortcuts), temporal upsampling on
/// the first two up blocks (`temperal_downsample [F, T, T]` reversed).
pub const VAE_Z: u32 = 48;
pub const VAE_PIX_CHANNELS: u32 = 12;
pub const VAE_RGB: u32 = 3;
pub const VAE_PATCH: u32 = 2;
/// `dims = [256·4, 256·4, 256·4, 256·2, 256·1]`: the decoder's widths, in
/// to out.
pub const VAE_DECODER_DIMS: [u32; 5] = [1024, 1024, 1024, 512, 256];
pub const VAE_RESNETS: u32 = 3;
pub const VAE_TEMPORAL_UP: [bool; 4] = [true, true, false, false];
/// `dims = [160·1, 160·1, 160·2, 160·4, 160·4]`: the ENCODER's widths
/// (`base_dim` 160 under the same `dim_mult`), in to out. The encoder is
/// narrower than the decoder — `decoder_base_dim` is 256 and `base_dim`
/// 160 — so its mid block attends a 640-wide row where the decoder's
/// attends 1024.
pub const VAE_ENCODER_DIMS: [u32; 5] = [160, 160, 320, 640, 640];
/// `num_res_blocks`: two resnets per `WanResidualDownBlock` (the decoder's
/// up blocks run `num_res_blocks + 1`).
pub const VAE_ENC_RESNETS: u32 = 2;
/// `temperal_downsample`, as the encoder reads it: down block `i` halves
/// time iff this says so, and the LAST block downsamples neither time nor
/// space (`down_flag = i != len(dim_mult) - 1`).
pub const VAE_TEMPORAL_DOWN: [bool; 4] = [false, true, true, false];
/// `F.normalize` clamps the norm at 1e-12; the RMS form adds it to the mean
/// square, which is the same number for any live activation.
pub const VAE_EPS: f32 = 1e-12;
/// `latents_mean` / `latents_std` (`vae/config.json`): the DiT works in a
/// NORMALISED latent space, `(z_vae − mean)/std`, and the reference
/// pipeline undoes that before `vae.decode` (`latents/(1/std) + mean`).
/// The decoder arm owns it (`forward::vae_decode`), the way `z_image`'s
/// owns its `scaling_factor`/`shift_factor`, so a guest hands the arm the
/// denoiser's own latent and never spells a family's numbers.
pub const VAE_LATENTS_MEAN: [f32; VAE_Z as usize] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];
pub const VAE_LATENTS_STD: [f32; VAE_Z as usize] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

/// The widest latent plane (`h·w` at latent resolution) one clip may
/// carry: 1280×704 at stride 16 is 44×80. The causal convs' frame caches
/// are sized from it (`CacheRow::State` slabs of `[2·plane, C_in]` per
/// slot, planes growing ×4 per spatial upsample), so a deployment sizes
/// its state slots against it.
pub const VAE_MAX_LATENT_PLANE: u64 = 44 * 80;

/// The float ports this text reads, by index within their kind and
/// reading. A port index is the family's own (`RuntimeInput::Latents {
/// port, .. }` carries it) and is the position among ports of one kind in
/// the reading's `ReadingFact::ports`, which is how the runtime resolves
/// `input(name)`.
pub mod port {
    /// `denoise`: a video lane's patch rows, `[rows, C·4]` bf16, feature
    /// order `(c, ph, pw)`.
    pub const LATENTS: u8 = 0;
    /// `denoise`: the context lane's 512 umT5 rows, `[rows, 4096]` bf16.
    pub const CONTEXT: u8 = 0;
    /// `denoise`: the scheduler timestep `σ·1000`, `[lanes, 1]`, bound by
    /// every VIDEO lane (TI2V's clean first frame is a second video lane
    /// at timestep 0 — `forward.rs`).
    pub const TIMESTEP: u8 = 0;
    /// `denoise`: the three rotary coordinates `(t, h, w)` per video row,
    /// in patch units, `[rows, 3]`.
    pub const POSITIONS: u8 = 0;
    /// `vae.decode.head` / `vae.decode`: ONE latent frame's voxels,
    /// `[h·w, 48]` bf16, in the DENOISER's space — the arm denormalises
    /// (`z·std + mean`, [`super::VAE_LATENTS_STD`]) itself, so the guest
    /// hands over exactly the rows the denoise reading answered.
    pub const VOXELS: u8 = 0;
    /// `vae.encode.head` / `vae.encode`: a PIXEL chunk's voxels,
    /// `[t·H·W, 3]` bf16 in `[-1, 1]`. Voxel index ONE, because the engine
    /// seats one rectangle per `(kind, index)` for the whole plan and the
    /// decode arms' latent clip is 48 wide at index 0 (`z_image`'s
    /// `vae.encode` states its own for the same reason).
    pub const PIXEL_VOXELS: u8 = 1;
}

/// One row's transformer shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    /// `num_attention_heads · attention_head_dim`.
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub ffn: u32,
    pub layers: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    /// `text_dim`: the width of the context rows.
    pub text_dim: u32,
    /// `freq_dim`: the sinusoid's width.
    pub freq_dim: u32,
}

impl Dims {
    /// `Wan-AI/Wan2.2-TI2V-5B-Diffusers`'s `transformer/config.json`.
    #[must_use]
    pub const fn ti2v_5b() -> Dims {
        Dims {
            dim: 3072,
            heads: 24,
            head_dim: 128,
            ffn: 14_336,
            layers: 30,
            in_channels: 48,
            out_channels: 48,
            text_dim: 4096,
            freq_dim: 256,
        }
    }

    /// `wan22_golden.py`'s `d128`: the real head, two of them.
    #[must_use]
    pub const fn mini_d128() -> Dims {
        Dims {
            dim: 256,
            heads: 2,
            head_dim: 128,
            ffn: 512,
            layers: 2,
            in_channels: 16,
            out_channels: 16,
            text_dim: 64,
            freq_dim: 256,
        }
    }

    /// `wan22_golden.py`'s `nano`: a 24-wide head, so `d − 4⌊d/6⌋` lands a
    /// split other than `[44, 42, 42]`.
    #[must_use]
    pub const fn mini_nano() -> Dims {
        Dims {
            dim: 48,
            heads: 2,
            head_dim: 24,
            ffn: 128,
            layers: 2,
            in_channels: 16,
            out_channels: 16,
            text_dim: 64,
            freq_dim: 32,
        }
    }

    /// The rotary split `[d − 4⌊d/6⌋, 2⌊d/6⌋, 2⌊d/6⌋, 0]` (`WanRotaryPosEmbed`):
    /// `[44, 42, 42]` at 128, `[8, 8, 8]` at 24. Sums to the head.
    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        let hw = 2 * (self.head_dim / 6);
        [self.head_dim - 2 * hw, hw, hw, 0]
    }

    /// `head_dim^-0.5`.
    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    /// The width of one patch row in: `in_channels · PATCH_VOL`.
    #[must_use]
    pub const fn patch_in(&self) -> u32 {
        self.in_channels * PATCH_VOL
    }

    /// The width of one patch row out: `out_channels · PATCH_VOL`.
    #[must_use]
    pub const fn patch_out(&self) -> u32 {
        self.out_channels * PATCH_VOL
    }
}

/// One `nn.Linear` with a bias — every projection in the transformer is one.
pub struct Linear {
    pub w: Weight,
    pub bias: Weight,
}

impl Linear {
    fn at(name: &str, out: u32, in_: u32, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [u64::from(out), u64::from(in_)], banks),
            bias: Weight::sym(
                format!("{name}.bias"),
                [u64::from(out)],
                crate::dense(banks),
            ),
        }
    }

    fn packed(name: &str, seams: &[u32], in_: u32, banks: Dtype) -> Linear {
        let out: u64 = seams.iter().map(|&s| u64::from(s)).sum();
        let seams: Vec<u64> = seams.iter().map(|&s| u64::from(s)).collect();
        Linear {
            w: Weight::sym(name, [out, u64::from(in_)], banks).packed(seams.clone()),
            bias: Weight::sym(format!("{name}.bias"), [out], crate::dense(banks)).packed(seams),
        }
    }
}

/// `attn1`: one packed `q|k|v`, the two across-heads RMS gains (`[dim]`
/// each), the output projection.
pub struct SelfAttn {
    pub qkv: Linear,
    pub norm_q: Weight,
    pub norm_k: Weight,
    pub out: Linear,
}

/// `attn2`: queries off the video rows, one packed `k|v` off the embedded
/// context rows, the same across-heads gains, the output projection. No
/// rope, no modulation, an ungated residual (study §C.1).
pub struct CrossAttn {
    pub q: Linear,
    pub kv: Linear,
    pub norm_q: Weight,
    pub norm_k: Weight,
    pub out: Linear,
}

/// `FeedForward(activation_fn="gelu-approximate")`: `net.0.proj` up, GELU
/// (tanh), `net.2` down.
pub struct Ffn {
    pub up: Linear,
    pub down: Linear,
}

/// One `WanTransformerBlock`.
pub struct Block {
    /// `scale_shift_table` `[1, 6, dim]`, read as one `[6·dim]` f32 bias
    /// over the shared `time_proj` vector, in the plan's slice order
    /// `[scale_msa | shift_msa | gate_msa | c_scale | c_shift | c_gate]`.
    pub table: Weight,
    pub self_attn: SelfAttn,
    /// `norm2`: the affine `FP32LayerNorm` before the cross-attention.
    pub norm2: Weight,
    pub norm2_bias: Weight,
    pub cross: CrossAttn,
    pub ffn: Ffn,
}

impl Block {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Block {
        let dense = crate::dense(banks);
        let dim = d.dim;
        let n = |s: &str| format!("{prefix}.{s}");
        let gain = |s: &str| Weight::sym(n(s), [u64::from(dim)], dense);
        Block {
            table: Weight::sym(n("table"), [u64::from(MOD_SLICES * dim)], Dtype::F32),
            self_attn: SelfAttn {
                qkv: Linear::packed(&n("self.qkv"), &[dim, dim, dim], dim, banks),
                norm_q: gain("self.norm_q"),
                norm_k: gain("self.norm_k"),
                out: Linear::at(&n("self.out"), dim, dim, banks),
            },
            norm2: gain("norm2"),
            norm2_bias: gain("norm2.bias"),
            cross: CrossAttn {
                q: Linear::at(&n("cross.q"), dim, dim, banks),
                kv: Linear::packed(&n("cross.kv"), &[dim, dim], dim, banks),
                norm_q: gain("cross.norm_q"),
                norm_k: gain("cross.norm_k"),
                out: Linear::at(&n("cross.out"), dim, dim, banks),
            },
            ffn: Ffn {
                up: Linear::at(&n("ffn.up"), d.ffn, dim, banks),
                down: Linear::at(&n("ffn.down"), dim, d.ffn, banks),
            },
        }
    }
}

/// A two-layer MLP: `linear_2(act(linear_1(x)))` — `TimestepEmbedding`
/// (SiLU) and `PixArtAlphaTextProjection` (GELU tanh) share the shape.
pub struct Embedder {
    pub linear_1: Linear,
    pub linear_2: Linear,
}

impl Embedder {
    fn at(prefix: &str, in_: u32, dim: u32, banks: Dtype) -> Embedder {
        Embedder {
            linear_1: Linear::at(&format!("{prefix}.1"), dim, in_, banks),
            linear_2: Linear::at(&format!("{prefix}.2"), dim, dim, banks),
        }
    }
}

/// The transformer.
pub struct Dit {
    /// `patch_embedding`: `Conv3d(C, dim, (1, 2, 2), stride (1, 2, 2))` is
    /// one `Linear(C·4 → dim)` over patch rows laid out `(c, ph, pw)`.
    pub patch_embed: Linear,
    /// `condition_embedder.text_embedder`: `Linear(text_dim → dim)`, GELU
    /// (tanh), `Linear(dim → dim)`, over the context rows once per fire.
    pub text_embed: Embedder,
    /// `condition_embedder.time_embedder`: the SiLU MLP over the sinusoid.
    pub time_embed: Embedder,
    /// `condition_embedder.time_proj`: `Linear(dim → 6·dim)` over
    /// `silu(temb)`, in the plan's slice order.
    pub time_proj: Linear,
    /// `time_embedder.linear_2` stacked twice, `[2·dim, dim]`, so the head
    /// reads `[temb | temb]` off `silu(linear_1(sinusoid))` in one
    /// projection: this IR has no column concatenation, and the head adds
    /// `temb` to BOTH slices of its table.
    pub head_proj: Linear,
    pub blocks: Vec<Block>,
    /// The model-level `scale_shift_table` `[1, 2, dim]` as one `[2·dim]`
    /// f32 bias in the plan's `[scale | shift]` order.
    pub head_table: Weight,
    /// `proj_out`: `Linear(dim → C_out·4)`, its rows permuted at import so
    /// the velocity row is laid out `(c, ph, pw)` like the latent row in.
    pub proj_out: Linear,
}

/// One umT5 encoder block: the pre-norm relative-bias self-attention and
/// the pre-norm gated-GELU MLP, every layer with its own bucket embedding.
pub struct TeLayer {
    pub attn_norm: Weight,
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub o: Weight,
    /// `relative_attention_bias.weight` `[num_buckets, heads]`.
    pub rel_bias: Weight,
    pub ffn_norm: Weight,
    pub wi_0: Weight,
    pub wi_1: Weight,
    pub wo: Weight,
}

/// `UMT5EncoderModel`: the shared embedding, 24 blocks, the final norm.
/// Bidirectional and cacheless: no kv space, no rope, the T5 relative
/// position bias on every layer at `sm_scale = 1`.
pub struct TextEncoder {
    pub embed: Weight,
    pub layers: Vec<TeLayer>,
    pub final_norm: Weight,
}

impl TextEncoder {
    fn umt5_xxl(banks: Dtype) -> TextEncoder {
        let dense = crate::dense(banks);
        let hidden = u64::from(TE_HIDDEN);
        let inner = u64::from(TE_HEADS * TE_HEAD_DIM);
        let inter = u64::from(TE_INTER);
        let layers = (0..TE_LAYERS)
            .map(|l| {
                let n = |s: &str| format!("te.layer.{l}.{s}");
                TeLayer {
                    attn_norm: Weight::sym(n("attn_norm"), [hidden], dense),
                    q: Weight::sym(n("q"), [inner, hidden], banks),
                    k: Weight::sym(n("k"), [inner, hidden], banks),
                    v: Weight::sym(n("v"), [inner, hidden], banks),
                    o: Weight::sym(n("o"), [hidden, inner], banks),
                    rel_bias: Weight::sym(
                        n("rel_bias"),
                        [u64::from(TE_BUCKETS), u64::from(TE_HEADS)],
                        dense,
                    ),
                    ffn_norm: Weight::sym(n("ffn_norm"), [hidden], dense),
                    wi_0: Weight::sym(n("wi_0"), [inter, hidden], banks),
                    wi_1: Weight::sym(n("wi_1"), [inter, hidden], banks),
                    wo: Weight::sym(n("wo"), [hidden, inter], banks),
                }
            })
            .collect();
        TextEncoder {
            embed: Weight::sym("te.embed", [u64::from(TE_VOCAB), hidden], banks),
            layers,
            final_norm: Weight::sym("te.final_norm", [hidden], dense),
        }
    }
}

/// One convolution of the VAE: the kernel as the checkpoint stores it
/// (`[C_out, C_in·kt·kh·kw]`, declared tap-major), its f32 bias, and — for
/// a causal 3-D kernel with `kt > 1` — the `CacheRow::State` slab holding
/// the clip's last [`front`](Conv::front) input frames between tiles
/// (`[front·plane, C_in]` bf16 per slot, `plane` the widest `h·w` this
/// conv sees).
pub struct Conv {
    pub w: Weight,
    pub bias: Weight,
    pub c_in: u32,
    pub c_out: u32,
    pub k: [u32; 3],
    /// How many frames of the previous tile this convolution pads with —
    /// its causal FRONT PAD, and so the height of its cache slab. `kt − 1`
    /// for a `same`-padded causal convolution (`WanCausalConv3d(.., 3,
    /// padding=1)` pads `2·padding = 2` frames in front), and ONE for the
    /// encoder's `downsample3d` time convolution, which pads nothing of
    /// its own and is handed `cat([last frame, x])` instead.
    pub front: u32,
    /// The frame cache's state row, `Some` iff `k[0] > 1`.
    pub cache: Option<String>,
    pub plane: u64,
}

impl Conv {
    fn at(name: &str, c_out: u32, c_in: u32, k: [u32; 3], plane: u64, banks: Dtype) -> Conv {
        let taps = k[0] * k[1] * k[2];
        Conv {
            w: Weight::sym(
                name,
                [u64::from(c_out), u64::from(c_in) * u64::from(taps)],
                banks,
            )
            .conv_taps_major(c_in, taps),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c_out)], Dtype::F32),
            c_in,
            c_out,
            k,
            front: k[0].saturating_sub(1),
            cache: (k[0] > 1).then(|| format!("{name}.frames")),
            plane,
        }
    }

    /// The same convolution padding `front` frames in front instead of
    /// `kt − 1` — the encoder's stride-2 time convolution, which keeps one
    /// frame of history and no padding.
    #[must_use]
    fn fronting(mut self, front: u32) -> Conv {
        self.front = front;
        self
    }

    /// The slab one slot of this conv's cache holds: `[front·plane, C_in]`.
    #[must_use]
    pub fn slab(&self) -> [u64; 2] {
        [u64::from(self.front) * self.plane, u64::from(self.c_in)]
    }
}

/// `WanResidualBlock`: `norm1 → silu → conv1 → norm2 → silu → conv2`, added
/// to the input (through a 1×1×1 `conv_shortcut` when the channels
/// change). The norms are channel RMS norms with a `[C]` gain.
pub struct Resnet {
    pub norm1: Weight,
    pub conv1: Conv,
    pub norm2: Weight,
    pub conv2: Conv,
    pub shortcut: Option<Conv>,
}

impl Resnet {
    fn at(prefix: &str, c_in: u32, c_out: u32, plane: u64, banks: Dtype) -> Resnet {
        let dense = crate::dense(banks);
        Resnet {
            norm1: Weight::sym(format!("{prefix}.norm1"), [u64::from(c_in)], dense),
            conv1: Conv::at(
                &format!("{prefix}.conv1"),
                c_out,
                c_in,
                [3, 3, 3],
                plane,
                banks,
            ),
            norm2: Weight::sym(format!("{prefix}.norm2"), [u64::from(c_out)], dense),
            conv2: Conv::at(
                &format!("{prefix}.conv2"),
                c_out,
                c_out,
                [3, 3, 3],
                plane,
                banks,
            ),
            shortcut: (c_in != c_out).then(|| {
                Conv::at(
                    &format!("{prefix}.shortcut"),
                    c_out,
                    c_in,
                    [1, 1, 1],
                    plane,
                    banks,
                )
            }),
        }
    }
}

/// `WanAttentionBlock`: a single-head (`head_dim = C`) self-attention over
/// every position of ONE FRAME, `to_qkv` and `proj` as 1×1 convs.
///
/// Traced (`forward::mid_attention`) through `spatial::attention` — the
/// conv VAE's own attention arm, one head as wide as the row, per CLIP, at
/// the stamped widths 256/512/1024 — and never `attention.ragged`, whose
/// kernel is stamped at head widths 64/128/256 over a token-axis CSR.
/// Per clip IS per frame here, because a decode arm takes exactly ONE
/// latent frame ([`super::forward`]'s chunking contract): the reference
/// decodes frame by frame through its per-conv caches and this arm is that
/// loop's body.
pub struct MidAttention {
    pub norm: Weight,
    pub qkv: Linear,
    pub proj: Linear,
}

/// `WanResample("upsample3d" | "upsample2d")`: for the 3-D kind a causal
/// `(3, 1, 1)` conv doubling the channels, read as two frames; then a
/// nearest 2× spatial upsample and a 3×3 conv, per frame.
pub struct Upsampler {
    pub time_conv: Option<Conv>,
    pub resample: Conv,
}

/// The `DupUp3D` residual shortcut of a `WanResidualUpBlock`, worked out
/// per block from `repeat_interleave` + the `(ft, fs, fs)` view: at equal
/// widths it is a nearest `(2, 2, 2)` upsample of every channel; halving
/// the width, channel `2c + b` lands at `h`-subposition `b` of channel `c`
/// (a `(1, 2, 1)` pixel shuffle) and the `w` side duplicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shortcut {
    /// `factor_t = 2, factor_s = 2`, `C → C`.
    Nearest222,
    /// `factor_t = 1, factor_s = 2`, `C → C/2`.
    ShuffleH,
}

/// One `WanResidualUpBlock`.
pub struct UpBlock {
    pub resnets: Vec<Resnet>,
    pub upsampler: Option<Upsampler>,
    pub shortcut: Option<Shortcut>,
}

/// `AvgDown3D(in_dim, out_dim, factor_t, factor_s)`, the residual shortcut
/// of a `WanResidualDownBlock`, as `spatial::avg_down` states it: the
/// block `factor` and the size of the contiguous run of widened channels
/// the mean folds. `group_size = in·factor / out` is the reference's own
/// formula, and on every Wan block it comes out `factor_s²` — a 2×2
/// spatial average pool that keeps the time block as `factor_t` channels
/// per input channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AvgDown {
    pub factor: [u32; 3],
    pub group: u32,
}

impl AvgDown {
    /// The shortcut a block with these widths and factors states.
    #[must_use]
    pub fn at(c_in: u32, c_out: u32, factor_t: u32, factor_s: u32) -> AvgDown {
        let factor = [factor_t, factor_s, factor_s];
        let volume = factor_t * factor_s * factor_s;
        assert_eq!(
            c_in * volume % c_out,
            0,
            "an AvgDown3D's widened channels must fold into whole groups"
        );
        AvgDown {
            factor,
            group: c_in * volume / c_out,
        }
    }
}

/// `WanResample("downsample2d" | "downsample3d")`: a per-frame 3×3
/// stride-2 convolution behind `nn.ZeroPad2d((0, 1, 0, 1))`, and — for the
/// 3-D kind — a causal `(3, 1, 1)` STRIDE-2 time convolution after it.
///
/// That time convolution is not a `same`-padded causal one: the reference
/// gives it `padding=0` and hands it `cat([the previous chunk's last
/// frame, x])`, so it pads ONE frame and halves the frame count. On the
/// FIRST chunk it does not run at all — the reference only stores the
/// frames (`forward::vae_encode`'s `first` arm, `spatial::store_frames`).
pub struct Downsampler {
    pub resample: Conv,
    pub time_conv: Option<Conv>,
}

/// One `WanResidualDownBlock`: two resnets, an optional resampler, and the
/// `AvgDown3D` shortcut added to the result — `x_out = f(x) +
/// avg_shortcut(x)`, the shortcut reading the block's INPUT.
pub struct DownBlock {
    pub resnets: Vec<Resnet>,
    pub downsampler: Option<Downsampler>,
    pub shortcut: AvgDown,
}

/// The Wan 2.2 VAE ENCODER: the 2×2 space-to-depth, `conv_in`, four
/// residual down blocks, the mid block, the head, `quant_conv`, and the
/// normalisation into the denoiser's latent space.
pub struct VaeEncoder {
    pub conv_in: Conv,
    pub down: Vec<DownBlock>,
    pub mid_res0: Resnet,
    pub mid_attn: MidAttention,
    pub mid_res1: Resnet,
    pub norm_out: Weight,
    /// `640 → 2·z_dim`: `DiagonalGaussianDistribution`'s
    /// `[mean | logvar]`, both halves, because `quant_conv` mixes all 96
    /// channels before the mean is taken.
    pub conv_out: Conv,
    /// `quant_conv` `96 → 96`, imported as its FIRST 48 output rows alone:
    /// the posterior mean is all this arm answers and the logvar's rows
    /// are never computed.
    pub quant: Conv,
    /// `mean` and `1/std` as `[48]` rows: the `(z − bias)·scale` form of
    /// `(z_vae − mean)/std`, which is what the denoise reading takes. The
    /// mirror of the decoder's [`Vae::denorm_bias`]/[`Vae::denorm_scale`],
    /// so a guest hands `vae.encode`'s answer straight to `denoise`.
    pub norm_bias: Weight,
    pub norm_scale: Weight,
}

impl VaeEncoder {
    fn wan22(banks: Dtype) -> VaeEncoder {
        let dense = crate::dense(banks);
        let dims = VAE_ENCODER_DIMS;
        // The encoder's planes go the other way from the decoder's: the
        // patchified pixel clip is 8× the latent plane on each axis, and
        // every down block quarters it.
        let p0 = VAE_MAX_LATENT_PLANE;
        let mut plane = 64 * p0;
        let mut down = Vec::new();
        for i in 0..4 {
            let (c_in, c_out) = (dims[i], dims[i + 1]);
            let prefix = format!("vae.enc.down.{i}");
            let resnets = (0..VAE_ENC_RESNETS)
                .map(|r| {
                    Resnet::at(
                        &format!("{prefix}.res.{r}"),
                        if r == 0 { c_in } else { c_out },
                        c_out,
                        plane,
                        banks,
                    )
                })
                .collect();
            let down_flag = i != 3;
            let temporal = VAE_TEMPORAL_DOWN[i];
            let downsampler = down_flag.then(|| Downsampler {
                resample: Conv::at(
                    &format!("{prefix}.resample"),
                    c_out,
                    c_out,
                    [1, 3, 3],
                    plane,
                    banks,
                ),
                // The time convolution sees the RESAMPLED plane, a quarter
                // of the block's, and keeps ONE frame of history.
                time_conv: temporal.then(|| {
                    Conv::at(
                        &format!("{prefix}.time_conv"),
                        c_out,
                        c_out,
                        [3, 1, 1],
                        plane / 4,
                        banks,
                    )
                    .fronting(1)
                }),
            });
            down.push(DownBlock {
                resnets,
                downsampler,
                shortcut: AvgDown::at(
                    c_in,
                    c_out,
                    if temporal { 2 } else { 1 },
                    if down_flag { 2 } else { 1 },
                ),
            });
            if down_flag {
                plane /= 4;
            }
        }
        let top = dims[4];
        VaeEncoder {
            conv_in: Conv::at(
                "vae.enc.conv_in",
                dims[0],
                VAE_PIX_CHANNELS,
                [3, 3, 3],
                64 * p0,
                banks,
            ),
            down,
            mid_res0: Resnet::at("vae.enc.mid.res.0", top, top, plane, banks),
            mid_attn: MidAttention {
                norm: Weight::sym("vae.enc.mid.attn.norm", [u64::from(top)], dense),
                qkv: Linear::at("vae.enc.mid.attn.qkv", 3 * top, top, banks),
                proj: Linear::at("vae.enc.mid.attn.proj", top, top, banks),
            },
            mid_res1: Resnet::at("vae.enc.mid.res.1", top, top, plane, banks),
            norm_out: Weight::sym("vae.enc.norm_out", [u64::from(top)], dense),
            conv_out: Conv::at("vae.enc.conv_out", 2 * VAE_Z, top, [3, 3, 3], plane, banks),
            quant: Conv::at("vae.enc.quant", VAE_Z, 2 * VAE_Z, [1, 1, 1], plane, banks),
            norm_bias: Weight::sym("vae.enc.norm_bias", [u64::from(VAE_Z)], dense),
            norm_scale: Weight::sym("vae.enc.norm_scale", [u64::from(VAE_Z)], dense),
        }
    }

    /// Every conv with a frame cache, in trace order.
    pub fn cached_convs(&self) -> impl Iterator<Item = &Conv> + '_ {
        let mut out: Vec<&Conv> = vec![&self.conv_in];
        for block in &self.down {
            for r in &block.resnets {
                out.push(&r.conv1);
                out.push(&r.conv2);
            }
            if let Some(Downsampler {
                time_conv: Some(tc),
                ..
            }) = &block.downsampler
            {
                out.push(tc);
            }
        }
        for r in [&self.mid_res0, &self.mid_res1] {
            out.push(&r.conv1);
            out.push(&r.conv2);
        }
        out.push(&self.conv_out);
        out.into_iter().filter(|c| c.cache.is_some())
    }
}

/// The Wan 2.2 VAE decoder: `post_quant_conv`, `conv_in`, the mid block,
/// four up blocks, the head, and the 2×2 depth-to-space at the end — with
/// the [`encoder`](Vae::enc) beside it.
pub struct Vae {
    /// `−mean/std` and `std` as `[48]` rows in the trunk's dtype: the
    /// `(z − bias)·scale` form of `z·std + mean`, which is what
    /// `elementwise.standardize` computes. Derived at import from
    /// [`VAE_LATENTS_MEAN`]/[`VAE_LATENTS_STD`] — the IR has no constant
    /// op, so a config number reaches a plan as a filled plane.
    pub denorm_bias: Weight,
    pub denorm_scale: Weight,
    pub post_quant: Conv,
    pub conv_in: Conv,
    pub mid_res0: Resnet,
    pub mid_attn: MidAttention,
    pub mid_res1: Resnet,
    pub up: Vec<UpBlock>,
    pub norm_out: Weight,
    pub conv_out: Conv,
    /// The encoder side: `vae.encode.head` / `vae.encode`.
    pub enc: VaeEncoder,
}

impl Vae {
    fn wan22(banks: Dtype) -> Vae {
        let dense = crate::dense(banks);
        let dims = VAE_DECODER_DIMS;
        let top = dims[0];
        let p0 = VAE_MAX_LATENT_PLANE;
        let mut plane = p0;
        let mut up = Vec::new();
        for i in 0..4 {
            let (c_in, c_out) = (dims[i], dims[i + 1]);
            let prefix = format!("vae.up.{i}");
            let resnets = (0..VAE_RESNETS)
                .map(|r| {
                    Resnet::at(
                        &format!("{prefix}.res.{r}"),
                        if r == 0 { c_in } else { c_out },
                        c_out,
                        plane,
                        banks,
                    )
                })
                .collect();
            let up_flag = i != 3;
            let upsampler = up_flag.then(|| Upsampler {
                time_conv: VAE_TEMPORAL_UP[i].then(|| {
                    Conv::at(
                        &format!("{prefix}.time_conv"),
                        2 * c_out,
                        c_out,
                        [3, 1, 1],
                        plane,
                        banks,
                    )
                }),
                resample: Conv::at(
                    &format!("{prefix}.resample"),
                    c_out,
                    c_out,
                    [1, 3, 3],
                    4 * plane,
                    banks,
                ),
            });
            let shortcut = up_flag.then(|| {
                if VAE_TEMPORAL_UP[i] {
                    assert_eq!(c_in, c_out, "a (2, 2, 2) DupUp3D keeps the width");
                    Shortcut::Nearest222
                } else {
                    assert_eq!(c_in, 2 * c_out, "a (1, 2, 2) DupUp3D halves the width");
                    Shortcut::ShuffleH
                }
            });
            up.push(UpBlock {
                resnets,
                upsampler,
                shortcut,
            });
            if up_flag {
                plane *= 4;
            }
        }
        let last = dims[4];
        Vae {
            denorm_bias: Weight::sym("vae.denorm_bias", [u64::from(VAE_Z)], dense),
            denorm_scale: Weight::sym("vae.denorm_scale", [u64::from(VAE_Z)], dense),
            post_quant: Conv::at("vae.post_quant", VAE_Z, VAE_Z, [1, 1, 1], p0, banks),
            conv_in: Conv::at("vae.conv_in", top, VAE_Z, [3, 3, 3], p0, banks),
            mid_res0: Resnet::at("vae.mid.res.0", top, top, p0, banks),
            mid_attn: MidAttention {
                norm: Weight::sym("vae.mid.attn.norm", [u64::from(top)], dense),
                qkv: Linear::at("vae.mid.attn.qkv", 3 * top, top, banks),
                proj: Linear::at("vae.mid.attn.proj", top, top, banks),
            },
            mid_res1: Resnet::at("vae.mid.res.1", top, top, p0, banks),
            up,
            norm_out: Weight::sym("vae.norm_out", [u64::from(last)], dense),
            conv_out: Conv::at(
                "vae.conv_out",
                VAE_PIX_CHANNELS,
                last,
                [3, 3, 3],
                plane,
                banks,
            ),
            enc: VaeEncoder::wan22(banks),
        }
    }

    /// Every conv with a frame cache, in trace order.
    pub fn cached_convs(&self) -> impl Iterator<Item = &Conv> + '_ {
        let mut out: Vec<&Conv> = vec![&self.conv_in];
        for r in [&self.mid_res0, &self.mid_res1] {
            out.push(&r.conv1);
            out.push(&r.conv2);
        }
        for block in &self.up {
            for r in &block.resnets {
                out.push(&r.conv1);
                out.push(&r.conv2);
            }
            if let Some(Upsampler {
                time_conv: Some(tc),
                ..
            }) = &block.upsampler
            {
                out.push(tc);
            }
        }
        out.push(&self.conv_out);
        out.into_iter().filter(|c| c.cache.is_some())
    }
}

/// The whole text.
pub struct Model {
    pub tp: u32,
    /// The dtype the banks are stored in — `Bf16` on every row today.
    pub banks: Dtype,
    pub dims: Dims,
    pub dit: Dit,
    /// `None` on the miniatures: their checkpoint is the transformer alone
    /// and their context rows are random, so they declare no `text`
    /// reading.
    pub te: Option<TextEncoder>,
    /// `None` on the miniatures, for the same reason.
    pub vae: Option<Vae>,
    /// The row's static flow shift.
    pub shift: f32,
}

impl Model {
    /// `Wan-AI/Wan2.2-TI2V-5B-Diffusers`: the 5 B single-backbone TI2V
    /// transformer behind umT5-xxl, the Wan 2.2 VAE, flow shift 5.
    #[must_use]
    pub fn ti2v_5b(banks: Dtype, tp: u32) -> Model {
        Model::new(
            banks,
            tp,
            Dims::ti2v_5b(),
            Some(TextEncoder::umt5_xxl(banks)),
            Some(Vae::wan22(banks)),
            SHIFT_TI2V,
        )
    }

    /// The `d128` miniature `wan22_golden.py --mini` writes. What the
    /// parity harness drives.
    #[must_use]
    pub fn mini_d128(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::mini_d128(), None, None, SHIFT_TI2V)
    }

    /// The `nano` miniature: a 24-wide head, traced and baked for the
    /// rope split alone (`attention.ragged` serves 64/128/256-wide heads
    /// on CUDA, so it does not fire).
    #[must_use]
    pub fn mini_nano(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::mini_nano(), None, None, SHIFT_TI2V)
    }

    fn new(
        banks: Dtype,
        tp: u32,
        d: Dims,
        te: Option<TextEncoder>,
        vae: Option<Vae>,
        shift: f32,
    ) -> Model {
        assert_eq!(
            tp, 1,
            "this text ships one-rank rows; tp {tp} is not a world it states"
        );
        assert_eq!(
            d.heads * d.head_dim,
            d.dim,
            "plain MHA: heads × head_dim is the width"
        );
        assert_eq!(
            d.rope_dims().iter().sum::<u32>(),
            d.head_dim,
            "the three rotary axes cover the whole head"
        );
        if te.is_some() {
            assert_eq!(d.text_dim, TE_HIDDEN, "the context rows are the encoder's");
        }
        if vae.is_some() {
            assert_eq!(d.in_channels, VAE_Z, "the latent is the VAE's");
        }
        let dim = d.dim;
        let dit = Dit {
            patch_embed: Linear::at("dit.patch_embed", dim, d.patch_in(), banks),
            text_embed: Embedder::at("dit.text_embed", d.text_dim, dim, banks),
            time_embed: Embedder::at("dit.time_embed", d.freq_dim, dim, banks),
            time_proj: Linear::at("dit.time_proj", MOD_SLICES * dim, dim, banks),
            head_proj: Linear::at("dit.head_proj", HEAD_SLICES * dim, dim, banks),
            blocks: (0..d.layers)
                .map(|i| Block::at(&format!("dit.block.{i}"), &d, banks))
                .collect(),
            head_table: Weight::sym("dit.head_table", [u64::from(HEAD_SLICES * dim)], Dtype::F32),
            proj_out: Linear::at("dit.proj_out", d.patch_out(), dim, banks),
        };
        Model {
            tp,
            banks,
            dims: d,
            dit,
            te,
            vae,
            shift,
        }
    }
}
