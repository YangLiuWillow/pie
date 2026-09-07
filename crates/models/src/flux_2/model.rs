//! The `flux_2` declaration: every dimension a Rust constant or a
//! [`Dims`] field, every weight named in the plan's own scheme.
//!
//! Three components under one plan (design D5): the Qwen3-4B text encoder
//! ([`TextEncoder`], `te.`), the FLUX.2 transformer ([`Dit`], `dit.`) and
//! the `AutoencoderKLFlux2` ([`super::vae::Vae`], `vae.`). The numbers are the
//! snapshot's `transformer/config.json`, `text_encoder/config.json` and
//! `vae/config.json` (study §C.2, §C.8, §C.9), restated here because a
//! family's dims are Rust constants and a `config.json` is carried, never
//! read. The miniature is `scripts/imagegen/flux2_golden.py`'s `MINI_CFG`:
//! the transformer alone, two heads, two blocks of each kind, a guidance
//! embedder (the flagship has none), random weights.

use model_dsl::{Dtype, Weight};

/// `in_channels`: one DiT token is a 2×2 block of the VAE's 32 latent
/// channels — the pixel-unshuffle the VAE wrapper performs, so the
/// transformer's `patch_size` is 1 and it never sees a grid.
pub const IN_CHANNELS: u32 = 128;
/// The VAE's latent channels before the 2×2 packing (`latent_channels`).
pub const VAE_CHANNELS: u32 = 32;
/// The packing factor along each spatial axis between the VAE latent
/// (`/8`) and the DiT token (`/16`).
pub const PACK: u32 = 2;
/// Pixels per VAE latent cell along each axis.
pub const VAE_COMPRESSION: u32 = 8;
/// Pixels per DiT token along each axis: `VAE_COMPRESSION · PACK`.
pub const TOKEN_COMPRESSION: u32 = VAE_COMPRESSION * PACK;

/// The attention head width. Pinned by `sum(axes_dims_rope) == 128`: every
/// row of this family, the miniature included, has 128-wide heads.
pub const HEAD_DIM: u32 = 128;
/// The four rotary axes `(T, H, W, L)`, 32 channels each — the whole head
/// turns. `T` is the reference index axis (`10·(i+1)` for reference `i`,
/// 0 for the target and the text), `L` the text position.
pub const ROPE_DIMS: [u32; 4] = [32, 32, 32, 32];
pub const ROPE_THETA: f32 = 2000.0;
pub const ROPE_AXES: u8 = 4;
/// **THE REFERENCE LANE'S ROTARY STRIDE ON `T`**: reference `i`'s tokens
/// sit at `T = REFERENCE_TIME_STRIDE·(i + 1)`, so the first reference
/// clears the target grid's `T = 0` and each further one clears the last
/// (`_prepare_image_ids`, which offsets by `10*(i+1)`). Published to
/// guests as `PositionConvention::reference_stride`, which is the only way
/// a family-blind guest can place a reference lane without spelling this
/// family's number.
pub const REFERENCE_TIME_STRIDE: u32 = 10;

/// The sinusoidal timestep/guidance embedding's width
/// (`timestep_guidance_channels`), `max_period` and layout: diffusers'
/// `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)` is
/// `[cos | sin]`.
pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
/// The reference multiplies its `[0, 1]` timestep by 1000 before the
/// sinusoid. The `timestep` port takes the SCHEDULER timestep `σ·1000`
/// already (what a `FlowMatchEuler` guest holds), so the plan scales by 1.
pub const T_SCALE: f32 = 1.0;
/// The guidance scale enters raw (`4.0`) and the reference multiplies it by
/// 1000 before the same sinusoid; the plan does that scaling itself.
pub const GUIDANCE_SCALE: f32 = 1000.0;

/// Every LayerNorm in the trunk is `elementwise_affine=False` at this eps;
/// the QK RMSNorms share it (`config.eps`).
pub const NORM_EPS: f32 = 1e-6;
/// `head_dim^-0.5`.
pub const SM_SCALE: f32 = 0.088_388_35;

/// How many `[dim]` slices the double-stream modulation carries per
/// stream: `(shift, scale, gate)` for attention, then for the MLP.
pub const DOUBLE_MOD_SLICES: u32 = 6;
/// The single-stream (parallel) block's one `(shift, scale, gate)` set.
pub const SINGLE_MOD_SLICES: u32 = 3;

/// `mlp_ratio` — 3.0 on every row.
pub const MLP_RATIO: u32 = 3;

/// The schedule the family was trained under: `FlowMatchEulerDiscrete`,
/// 1000 training steps. The dynamic shift is the EMPIRICAL mu fit of
/// `pipeline_flux2_klein.py::compute_empirical_mu` — see
/// [`super::forward::empirical_mu`] — which depends on the step count as
/// well as the token count, and so has no place in a `ScheduleFact`;
/// the fact pins the 1024² / 4-step sigmas the golden was run under.
pub const TRAIN_STEPS: u32 = 1000;

/// The text encoder's numbers (`text_encoder/config.json`, Qwen3-4B).
pub const TE_HIDDEN: u32 = 2560;
pub const TE_VOCAB: u32 = 151_936;
pub const TE_Q_HEADS: u32 = 32;
pub const TE_KV_HEADS: u32 = 8;
pub const TE_HEAD_DIM: u32 = 128;
pub const TE_INTER: u32 = 9728;
pub const TE_THETA: f32 = 1_000_000.0;
pub const TE_EPS: f32 = 1e-6;
/// `num_hidden_layers`.
pub const TE_DEPTH: u32 = 36;
/// The conditioning is `hidden_states[k]` for `k` in [`TE_TAPS`] — the
/// residual LEAVING decoder layer `k − 1` (index 0 is the embedding) —
/// concatenated on the channel axis: `3 · 2560 = 7680 =
/// joint_attention_dim`. Nothing past the last tap is computed, so this
/// plan runs [`TE_LAYERS`] of the 36 layers and never reads the final norm.
pub const TE_TAPS: [u32; 3] = [9, 18, 27];
pub const TE_LAYERS: u32 = 27;
/// `max_sequence_length`: the truncation bound of a prompt.
pub const TE_MAX_TOKENS: u32 = 512;
/// `joint_attention_dim` of the flagship: the three taps side by side.
pub const TE_CONTEXT_WIDTH: u32 = 3 * TE_HIDDEN;

/// The float ports this text reads, by index within their kind and
/// reading. A port index is the family's own (`RuntimeInput::Latents {
/// port, .. }` carries it) and is the position among ports of one kind in
/// the reading's `ReadingFact::ports`, which is how the runtime resolves
/// `input(name)`.
pub mod port {
    /// `denoise`: the image AND reference lanes' token rows, `[rows, 128]`
    /// bf16 — the same port on both streams, since a reference is more of
    /// the same tokens (study §I.1).
    pub const LATENTS: u8 = 0;
    /// `denoise`: the text lane's rows. On the flagship these are the
    /// `text` reading's readout, already through `context_embedder`
    /// (`[rows, dim]`); on the miniature the raw `[rows,
    /// joint_attention_dim]` stack, embedded in this arm.
    pub const CONTEXT: u8 = 0;
    /// `denoise`: the scheduler timestep `σ·1000`, `[lanes, 1]`.
    pub const TIMESTEP: u8 = 0;
    /// `denoise`, rows with `guidance_embeds`: the guidance scale,
    /// `[lanes, 1]`.
    pub const GUIDANCE: u8 = 1;
    /// `denoise`: the four rotary coordinates per row, `[rows, 4]`.
    pub const POSITIONS: u8 = 0;
    /// `vae.decode`: the packed latent clip, `[voxels, 128]` bf16 at token
    /// resolution (`/16`), BatchNorm-normalised as the denoiser holds it.
    pub const VOXELS: u8 = 0;
    /// `vae.encode`: the pixel clip, `[voxels, 3]` in `[-1, 1]`. A second
    /// voxel index because the engine seats one rectangle per `(kind,
    /// index)` for the whole plan and the two clips are different widths.
    pub const PIXEL_VOXELS: u8 = 1;
}

/// One row's shape: the numbers that differ between the shipped
/// transformer and the miniature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    /// `num_attention_heads · attention_head_dim`.
    pub dim: u32,
    pub heads: u32,
    /// `int(dim · mlp_ratio)`.
    pub inter: u32,
    /// `joint_attention_dim`: the width of the raw text conditioning.
    pub context_in: u32,
    pub double_blocks: u32,
    pub single_blocks: u32,
    /// `guidance_embeds`: a second sinusoid + MLP added to the timestep
    /// embedding, and a `guidance` port on the `denoise` reading.
    pub guidance_embeds: bool,
}

impl Dims {
    /// `black-forest-labs/FLUX.2-klein-4B`'s `transformer/config.json`.
    #[must_use]
    pub const fn klein_4b() -> Dims {
        Dims {
            dim: 3072,
            heads: 24,
            inter: 3072 * MLP_RATIO,
            context_in: TE_CONTEXT_WIDTH,
            double_blocks: 5,
            single_blocks: 20,
            guidance_embeds: false,
        }
    }

    /// `flux2_golden.py`'s `MINI_CFG`.
    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            dim: 256,
            heads: 2,
            inter: 256 * MLP_RATIO,
            context_in: 192,
            double_blocks: 2,
            single_blocks: 2,
            guidance_embeds: true,
        }
    }
}

/// One bias-free `nn.Linear` — every projection in the transformer is one.
pub type Linear = Weight;

/// A self-attention's projections: one packed `q|k|v`, the per-head QK
/// RMS gains, the output projection.
pub struct Attn {
    pub qkv: Linear,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Linear,
}

impl Attn {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Attn {
        let dense = crate::dense(banks);
        let dim = u64::from(d.dim);
        Attn {
            qkv: Weight::sym(format!("{prefix}.qkv"), [3 * dim, dim], banks)
                .packed([dim, dim, dim]),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out: Weight::sym(format!("{prefix}.out"), [dim, dim], banks),
        }
    }
}

/// `Flux2FeedForward`: `linear_in` lands `[gate | up]` in one plane
/// (`Flux2SwiGLU` is parameter-free), `linear_out` brings it back.
pub struct Swiglu {
    pub linear_in: Linear,
    pub linear_out: Linear,
}

impl Swiglu {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Swiglu {
        let (dim, inter) = (u64::from(d.dim), u64::from(d.inter));
        Swiglu {
            linear_in: Weight::sym(format!("{prefix}.in"), [2 * inter, dim], banks)
                .packed([inter, inter]),
            linear_out: Weight::sym(format!("{prefix}.out"), [dim, inter], banks),
        }
    }
}

/// One side of a double-stream block: its own attention projections and
/// its own MLP. The modulation is NOT here — the three modulation linears
/// are shared by every block (study §C.5) and live on [`Dit`].
pub struct Side {
    pub attn: Attn,
    pub ff: Swiglu,
}

impl Side {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Side {
        Side {
            attn: Attn::at(&format!("{prefix}.attn"), d, banks),
            ff: Swiglu::at(&format!("{prefix}.ff"), d, banks),
        }
    }
}

/// `Flux2TransformerBlock`: two sides over one joint attention.
pub struct DoubleBlock {
    pub img: Side,
    pub txt: Side,
}

/// `Flux2SingleTransformerBlock`: the ViT-22B parallel block. One
/// in-projection lands `[q | k | v | gate | up]`; the out-projection reads
/// `[attn | swiglu]`, and is declared as its two column blocks because
/// this IR concatenates rows, not columns — `out(cat(a, m)) = out_a(a) +
/// out_m(m)`, exactly.
pub struct SingleBlock {
    pub in_proj: Linear,
    pub q_norm: Weight,
    pub k_norm: Weight,
    /// `to_out[:, :dim]`.
    pub out_attn: Linear,
    /// `to_out[:, dim:]`.
    pub out_mlp: Linear,
}

impl SingleBlock {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> SingleBlock {
        let dense = crate::dense(banks);
        let (dim, inter) = (u64::from(d.dim), u64::from(d.inter));
        SingleBlock {
            in_proj: Weight::sym(format!("{prefix}.in"), [3 * dim + 2 * inter, dim], banks)
                .packed([dim, dim, dim, inter, inter]),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out_attn: Weight::sym(format!("{prefix}.out_attn"), [dim, dim], banks),
            out_mlp: Weight::sym(format!("{prefix}.out_mlp"), [dim, inter], banks),
        }
    }
}

/// `TimestepEmbedding`: `linear_2(silu(linear_1(sinusoid)))`.
pub struct Embedder {
    pub linear_1: Linear,
    pub linear_2: Linear,
}

impl Embedder {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Embedder {
        let dim = u64::from(d.dim);
        Embedder {
            linear_1: Weight::sym(format!("{prefix}.1"), [dim, u64::from(T_FREQ_DIM)], banks),
            linear_2: Weight::sym(format!("{prefix}.2"), [dim, dim], banks),
        }
    }
}

/// The transformer.
pub struct Dit {
    pub x_embed: Linear,
    /// `context_embedder`, applied in the `denoise` arm over the raw
    /// `[rows, context_in]` conditioning — the miniature's way. `None` on
    /// the flagship, whose encoder applies it as three column blocks
    /// ([`TextEncoder::context_embed`]).
    pub context_embed: Option<Linear>,
    pub t_embed: Embedder,
    pub g_embed: Option<Embedder>,
    /// The three shared modulation linears (study §C.5), in the plan's
    /// slice order `[scale | shift | gate]` per set (the checkpoint's is
    /// `[shift | scale | gate]`; `import.rs` swaps).
    pub mod_img: Linear,
    pub mod_txt: Linear,
    pub mod_single: Linear,
    pub double: Vec<DoubleBlock>,
    pub single: Vec<SingleBlock>,
    /// `norm_out.linear`: `[scale | shift]`, the checkpoint's own order.
    pub norm_out: Linear,
    pub proj_out: Linear,
}

/// One Qwen3 decoder layer of the text encoder: plain (ungated) GQA
/// attention with per-head QK RMSNorm and full neox rotary, a SwiGLU MLP.
pub struct TeLayer {
    pub attn_norm: Weight,
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub o: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub mlp_norm: Weight,
    pub gate_up: Weight,
    pub down: Weight,
    /// This layer's kv row.
    pub kv: String,
}

/// The text encoder: `Qwen3ForCausalLM` (Qwen3-4B), headless, read at
/// three intermediate residuals.
///
/// Why this is its own declaration and not `qwen_3::model::Model`: the
/// `qwen_3` family in this catalog is Qwen3.5/3.6 — a sigmoid-gated
/// attention projection, Gated DeltaNet on three layers in four, partial
/// rotary at θ 1e7 — none of which `Qwen3ForCausalLM` has. The `z_image`
/// family carries the same declaration for the same encoder; folding both
/// into `qwen_3` once it grows a gate-less all-attention row is the
/// recorded follow-up.
pub struct TextEncoder {
    pub hidden: u32,
    pub vocab: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub theta: f32,
    pub eps: f32,
    pub sm_scale: f32,
    pub embed: Weight,
    pub layers: Vec<TeLayer>,
    /// `dit.context_embedder.weight` `[dim, 3·hidden]` as its three column
    /// blocks, one per tap: `embed(cat(h9, h18, h27)) = W0·h9 + W1·h18 +
    /// W2·h27`, which is how the `text` reading exports the conditioning
    /// already embedded (`[rows, dim]`) — this IR has no column
    /// concatenation, and one `hidden` readout per reading is what the
    /// runtime reads back.
    pub context_embed: [Linear; 3],
}

impl TextEncoder {
    fn qwen3_4b(d: &Dims, banks: Dtype) -> TextEncoder {
        let dense = crate::dense(banks);
        let hidden = u64::from(TE_HIDDEN);
        let hd = u64::from(TE_HEAD_DIM);
        let inter = u64::from(TE_INTER);
        let layers = (0..TE_LAYERS)
            .map(|l| {
                let n = |s: &str| format!("te.layer.{l}.{s}");
                TeLayer {
                    attn_norm: Weight::sym(n("attn_norm"), [hidden], dense),
                    q: Weight::sym(n("q"), [u64::from(TE_Q_HEADS) * hd, hidden], banks),
                    k: Weight::sym(n("k"), [u64::from(TE_KV_HEADS) * hd, hidden], banks),
                    v: Weight::sym(n("v"), [u64::from(TE_KV_HEADS) * hd, hidden], banks),
                    o: Weight::sym(n("o"), [hidden, u64::from(TE_Q_HEADS) * hd], banks),
                    q_norm: Weight::sym(n("q_norm"), [hd], dense),
                    k_norm: Weight::sym(n("k_norm"), [hd], dense),
                    mlp_norm: Weight::sym(n("mlp_norm"), [hidden], dense),
                    gate_up: Weight::sym(n("gate_up"), [2 * inter, hidden], banks)
                        .packed([inter, inter]),
                    down: Weight::sym(n("down"), [hidden, inter], banks),
                    kv: format!("te.kv.{l}"),
                }
            })
            .collect();
        let dim = u64::from(d.dim);
        let tap = |i: usize| Weight::sym(format!("dit.context_embed.{i}"), [dim, hidden], banks);
        TextEncoder {
            hidden: TE_HIDDEN,
            vocab: TE_VOCAB,
            q_heads: TE_Q_HEADS,
            kv_heads: TE_KV_HEADS,
            head_dim: TE_HEAD_DIM,
            inter: TE_INTER,
            theta: TE_THETA,
            eps: TE_EPS,
            sm_scale: (TE_HEAD_DIM as f32).sqrt().recip(),
            embed: Weight::sym("te.embed", [u64::from(TE_VOCAB), hidden], banks),
            layers,
            context_embed: [tap(0), tap(1), tap(2)],
        }
    }
}

/// The whole text.
pub struct Model {
    pub tp: u32,
    /// The dtype the banks are stored in — `Bf16` on every row today.
    pub banks: Dtype,
    /// The kv dtype of the encoder's cache.
    pub kv: Dtype,
    pub dims: Dims,
    pub dit: Dit,
    /// `None` on the miniature: its checkpoint is the transformer alone
    /// and its conditioning rows are random, so it declares no `text`
    /// reading.
    pub te: Option<TextEncoder>,
    /// The `AutoencoderKLFlux2` ([`super::vae`]), `None` on the miniature
    /// for the same reason: the `vae.decode` / `vae.encode` readings exist
    /// iff this does.
    pub vae: Option<super::vae::Vae>,
}

impl Model {
    /// `black-forest-labs/FLUX.2-klein-4B`: the 3.9 B transformer behind
    /// Qwen3-4B, the shared VAE, four distilled steps, no guidance embedder.
    #[must_use]
    pub fn klein_4b(banks: Dtype, tp: u32) -> Model {
        let d = Dims::klein_4b();
        Model::new(
            banks,
            tp,
            d,
            Some(TextEncoder::qwen3_4b(&d, banks)),
            Some(super::vae::Vae::flux2(banks)),
        )
    }

    /// The miniature `flux2_golden.py --mini` writes: the transformer at
    /// [`Dims::mini`], with a guidance embedder, no encoder, no VAE. What
    /// the parity harness drives.
    #[must_use]
    pub fn mini(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::mini(), None, None)
    }

    fn new(
        banks: Dtype,
        tp: u32,
        d: Dims,
        te: Option<TextEncoder>,
        vae: Option<super::vae::Vae>,
    ) -> Model {
        assert_eq!(
            tp, 1,
            "this text ships one-rank rows; tp {tp} is not a world it states"
        );
        assert_eq!(
            d.heads * HEAD_DIM,
            d.dim,
            "plain MHA over 128-wide heads: heads × 128 is the width"
        );
        assert_eq!(
            ROPE_DIMS.iter().sum::<u32>(),
            HEAD_DIM,
            "the four rotary axes cover the whole head"
        );
        let dim = u64::from(d.dim);
        let dit = Dit {
            x_embed: Weight::sym("dit.x_embed", [dim, u64::from(IN_CHANNELS)], banks),
            context_embed: te
                .is_none()
                .then(|| Weight::sym("dit.context_embed", [dim, u64::from(d.context_in)], banks)),
            t_embed: Embedder::at("dit.t_embed", &d, banks),
            g_embed: d
                .guidance_embeds
                .then(|| Embedder::at("dit.g_embed", &d, banks)),
            mod_img: Weight::sym(
                "dit.mod_img",
                [u64::from(DOUBLE_MOD_SLICES) * dim, dim],
                banks,
            ),
            mod_txt: Weight::sym(
                "dit.mod_txt",
                [u64::from(DOUBLE_MOD_SLICES) * dim, dim],
                banks,
            ),
            mod_single: Weight::sym(
                "dit.mod_single",
                [u64::from(SINGLE_MOD_SLICES) * dim, dim],
                banks,
            ),
            double: (0..d.double_blocks)
                .map(|i| DoubleBlock {
                    img: Side::at(&format!("dit.double.{i}.img"), &d, banks),
                    txt: Side::at(&format!("dit.double.{i}.txt"), &d, banks),
                })
                .collect(),
            single: (0..d.single_blocks)
                .map(|i| SingleBlock::at(&format!("dit.single.{i}"), &d, banks))
                .collect(),
            norm_out: Weight::sym("dit.norm_out", [2 * dim, dim], banks),
            proj_out: Weight::sym("dit.proj_out", [u64::from(IN_CHANNELS), dim], banks),
        };
        Model {
            tp,
            banks,
            kv: Dtype::Bf16,
            dims: d,
            dit,
            te,
            vae,
        }
    }
}
