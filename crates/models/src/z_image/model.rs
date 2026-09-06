//! The Z-Image declaration: every dimension a Rust constant (per row, in a
//! [`Dims`]), every weight named in the plan's own scheme.
//!
//! Z-Image (Tongyi-MAI; `.wiki/imagegen/study/z-image.md`) is a single-stream
//! DiT — image patch rows and caption rows in ONE sequence, image first — of
//! 2 modulated noise-refiner blocks over the image rows, 2 unmodulated
//! context-refiner blocks over the caption rows, and 30 modulated joint
//! blocks over both, fed by a Qwen3-4B text encoder read at its second-last
//! hidden state and decoded by the FLUX 16-channel VAE. The numbers below
//! are `transformer/config.json`'s and `text_encoder/config.json`'s,
//! restated here because a family's dims are Rust constants and a
//! `config.json` is carried, never read.
//!
//! One `Model` carries up to three components under one plan (design D5):
//! `dit` (always), `te` (the text encoder, `None` on the miniature whose
//! checkpoint is the transformer alone) and — not yet — the VAE, which
//! waits on the voxel axis (`IMAGEGEN_CONTRACT.md` §6) being read by a
//! family text; see [`super::forward`]'s `vae` stub.
//!
//! # Numerics contract
//!
//! * Banks are bf16 (the Turbo checkpoint ships fp32 and is cast at
//!   import; the base ships bf16). Activations along the token rows are
//!   bf16; matmuls accumulate fp32 and round once.
//! * Every RMSNorm (the four sandwich norms per block, the QK norms, the
//!   caption embedder's) sums in fp32 and rounds once at the store —
//!   what `elementwise.rmsnorm` does on every shell. diffusers' `RMSNorm`
//!   takes its variance in fp32 too and rounds twice (once after the
//!   rsqrt, once after the gain), so a port matches the reference to one
//!   bf16 rounding of the gain multiply. The *sglang* "bf16-native"
//!   reduction tree the study calls a correctness contract (§C.2) is NOT
//!   reproduced: this IR has no norm variant that accumulates in bf16, and
//!   the goldens (`scripts/imagegen/zimage_golden.py`) come from diffusers,
//!   not sglang. A `Rmsnorm { accumulate: Bf16 }` variant is the follow-up
//!   if a parity gate ever demands the sglang tree.
//! * The timestep chain (`1000 − t` → sinusoid → MLP → the adaLN vectors)
//!   is fp32 end to end, as a lane-vector chain is (contract §3); the
//!   reference runs it in bf16 under `torch_dtype=bf16`, so a port is the
//!   more precise side. Gates go through `tanh` in fp32, scales are
//!   `1 + s` in fp32, both applied to bf16 rows with one rounding.
//! * The velocity leaves as bf16 rows (the reference's transformer output
//!   is bf16 before the pipeline's `.float()`); the sampler is the guest's
//!   fp32 epilogue.

use model_dsl::{Dtype, Weight};

/// VAE latent channels (FLUX 16-channel `AutoencoderKL`).
pub const CHANNELS: u32 = 16;
/// Spatial patch: one image row is `PATCH × PATCH` latent cells.
pub const PATCH: u32 = 2;
/// `CHANNELS · PATCH²`: the width of one image row in and out.
pub const PATCH_FEATURES: u32 = CHANNELS * PATCH * PATCH;
/// Pixels per latent cell (the VAE's 8× stride); with [`PATCH`], 16 px per row side.
pub const SPATIAL_COMPRESSION: u32 = 8;

/// The adaLN bottleneck (`ADALN_EMBED_DIM`): every block's modulation is a
/// `Linear(256 → 4·dim)` off this one vector.
pub const ADALN_DIM: u32 = 256;
/// The timestep MLP's middle width (`TimestepEmbedder(mid_size=1024)`).
pub const T_MID: u32 = 1024;
/// The sinusoidal embedding's width (`frequency_embedding_size`).
pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
/// diffusers' `TimestepEmbedder` concatenates `[cos | sin]`, which is the
/// `flip_sin_cos` reading of `elementwise.sinusoid`.
pub const T_FLIP_SIN_COS: bool = true;
/// `t_scale`: the transformer's `t ∈ [0, 1)` is multiplied by 1000 before
/// the sinusoid. The plan takes the SCHEDULER timestep `σ·1000` on its port
/// and computes `1000 − t` itself (the reference pipeline's
/// `(1000 − t) / 1000`, times `t_scale`), so this constant is folded into
/// [`T_FLIP`] and the sinusoid runs at scale 1.
pub const T_SCALE: f32 = 1000.0;
/// The constant the timestep is subtracted from: `u = T_FLIP − t`.
pub const T_FLIP: f32 = 1000.0;
/// Training steps the schedule's sigma axis is scaled by.
pub const TRAIN_STEPS: u32 = 1000;

/// Every row count in this model is padded to a multiple of this
/// (`SEQ_MULTI_OF`): image rows and caption rows alike, with learned pad
/// rows that ARE attended (study §C.3). A guest supplies rows already
/// padded and flags the pad rows; see [`super::forward`].
pub const SEQ_MULTIPLE: u32 = 32;

/// Block / QK RMSNorm epsilon (`norm_eps`).
pub const NORM_EPS: f32 = 1e-5;
/// The final layer's LayerNorm epsilon.
pub const FINAL_LN_EPS: f32 = 1e-6;
/// Three rotary axes `(t, h, w)`, θ 256, interleaved pairs.
pub const ROPE_AXES: u8 = 3;
pub const ROPE_THETA: f32 = 256.0;
/// How many `[dim]` slices an adaLN vector carries: `scale_msa`,
/// `gate_msa`, `scale_mlp`, `gate_mlp` — two scales and two gates, no
/// shift (study §C.2).
pub const MOD_SLICES: u32 = 4;

/// The text encoder's numbers (`text_encoder/config.json`, Qwen3-4B).
pub const TE_HIDDEN: u32 = 2560;
pub const TE_VOCAB: u32 = 151_936;
pub const TE_Q_HEADS: u32 = 32;
pub const TE_KV_HEADS: u32 = 8;
pub const TE_HEAD_DIM: u32 = 128;
pub const TE_INTER: u32 = 9728;
pub const TE_THETA: f32 = 1_000_000.0;
pub const TE_EPS: f32 = 1e-6;
/// The encoder's depth (`num_hidden_layers`).
pub const TE_DEPTH: u32 = 36;
/// How many of its layers this plan RUNS: the caption is
/// `hidden_states[-2]`, the residual entering the last block, so the last
/// block and the final norm are never computed and their weights never
/// read (~110 M parameters the reference loads for nothing).
pub const TE_LAYERS: u32 = TE_DEPTH - 1;
/// The most caption tokens a prompt renders to (`max_sequence_length`).
pub const TE_MAX_TOKENS: u32 = 512;

/// The float ports this text reads, by index within their kind and reading.
/// A port index is the family's own (`RuntimeInput::Latents { port, .. }`
/// carries it) and is the position among ports of one kind in the reading's
/// `ReadingFact::ports`, which is how the runtime resolves `input(name)`.
pub mod port {
    /// `denoise`: the image lane's patch rows, `[rows, PATCH_FEATURES]` bf16.
    pub const LATENTS: u8 = 0;
    /// `denoise`: the image lane's pad flags, `[rows, 1]` (`0` real, `1`
    /// pad), the second latents port of the reading.
    pub const PAD_IMAGE: u8 = 1;
    /// `refine`: the caption lane's pad flags, `[rows, 1]`, its only
    /// latents port.
    pub const PAD_CAPTION: u8 = 0;
    /// `refine`: the raw caption rows, `[rows, cap_width]` (Qwen3 layer −2);
    /// `denoise`: the refined caption rows, `[rows, dim]`. Both the first
    /// context port of their reading.
    pub const CONTEXT: u8 = 0;
    /// `denoise`: the scheduler timestep `σ·1000`, `[lanes, 1]`.
    pub const TIMESTEP: u8 = 0;
    /// `refine`, `denoise`: the three rotary coordinates per row, `[rows, 3]`.
    pub const POSITIONS: u8 = 0;
}

/// One row's shape, the numbers that differ between the shipped transformer
/// and the miniature `scripts/imagegen/zimage_golden.py --mini` writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
    /// SwiGLU intermediate, `int(dim / 3 · 8)`.
    pub inter: u32,
    pub joint_layers: u32,
    pub refiner_layers: u32,
    /// The caption feature width (`cap_feat_dim`): the encoder's hidden.
    pub cap_width: u32,
    /// Channels per rotary axis, summing to `head_dim`.
    pub rope_dims: [u32; 4],
}

impl Dims {
    /// `Tongyi-MAI/Z-Image-Turbo` (and `Z-Image`): `transformer/config.json`.
    #[must_use]
    pub const fn turbo() -> Dims {
        Dims {
            dim: 3840,
            heads: 30,
            head_dim: 128,
            inter: 10_240,
            joint_layers: 30,
            refiner_layers: 2,
            cap_width: TE_HIDDEN,
            rope_dims: [32, 48, 48, 0],
        }
    }

    /// `zimage_golden.py --mini`'s `MINI_CFG`: dim 256 (kept ≥ 256 so the
    /// adaLN width stays 256), 4 heads of 64, 2+2+2 blocks, captions 64
    /// wide, rope `[16, 24, 24]`. `int(256 / 3 · 8) = 682`.
    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            dim: 256,
            heads: 4,
            head_dim: 64,
            inter: 682,
            joint_layers: 2,
            refiner_layers: 2,
            cap_width: 64,
            rope_dims: [16, 24, 24, 0],
        }
    }

    /// `head_dim^-0.5`.
    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }
}

/// One `nn.Linear` with a bias.
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
}

/// A block's attention: one packed bias-free `q|k|v`, the per-head QK
/// gains, the bias-free output projection.
pub struct Attn {
    pub qkv: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Weight,
}

/// The SwiGLU: `w2(silu(w1 x) · w3 x)`, `w1|w3` packed, all bias-free.
pub struct Swiglu {
    pub gate_up: Weight,
    pub down: Weight,
}

/// One `ZImageTransformerBlock`: sandwich RMSNorms around each sublayer,
/// and — for the modulated kind — the `Linear(256 → 4·dim)` adaLN
/// projection (no activation before it; only the final layer has one).
pub struct Block {
    /// `None` for a context-refiner block, which takes no timestep at all.
    pub ada: Option<Linear>,
    pub attn_norm1: Weight,
    pub attn_norm2: Weight,
    pub ffn_norm1: Weight,
    pub ffn_norm2: Weight,
    pub attn: Attn,
    pub mlp: Swiglu,
}

impl Block {
    fn at(prefix: &str, d: &Dims, modulated: bool, banks: Dtype) -> Block {
        let dense = crate::dense(banks);
        let dim = u64::from(d.dim);
        let hd = u64::from(d.head_dim);
        let inter = u64::from(d.inter);
        let n = |s: &str| format!("{prefix}.{s}");
        let norm = |s: &str, width: u64| Weight::sym(n(s), [width], dense);
        Block {
            ada: modulated.then(|| Linear::at(&n("ada"), MOD_SLICES * d.dim, ADALN_DIM, banks)),
            attn_norm1: norm("attn_norm1", dim),
            attn_norm2: norm("attn_norm2", dim),
            ffn_norm1: norm("ffn_norm1", dim),
            ffn_norm2: norm("ffn_norm2", dim),
            attn: Attn {
                qkv: Weight::sym(n("qkv"), [3 * dim, dim], banks).packed([dim, dim, dim]),
                q_norm: norm("q_norm", hd),
                k_norm: norm("k_norm", hd),
                out: Weight::sym(n("out"), [dim, dim], banks),
            },
            mlp: Swiglu {
                gate_up: Weight::sym(n("gate_up"), [2 * inter, dim], banks).packed([inter, inter]),
                down: Weight::sym(n("down"), [dim, inter], banks),
            },
        }
    }
}

/// The transformer.
pub struct Dit {
    /// `all_x_embedder["2-1"]`: `Linear(64 → dim)`.
    pub x_embed: Linear,
    /// The image pad rows as a scale-shift projection of the pad flag: a
    /// `[2·dim, 1]` bank `[−1 × dim | x_pad_token]ᵀ`, so a flag row `f`
    /// projects to `[−f | f·x_pad_token]` and one `elementwise.modulate`
    /// lands `x·(1−f) + f·x_pad_token` — the row itself at `f = 0`, the
    /// learned token at `f = 1`. `import.rs` states it from the stored
    /// `x_pad_token`.
    pub x_pad_mod: Weight,
    /// `cap_embedder.0`: `RMSNorm(cap_width)`.
    pub cap_norm: Weight,
    /// `cap_embedder.1`: `Linear(cap_width → dim)`.
    pub cap_embed: Linear,
    /// As [`x_pad_mod`](Dit::x_pad_mod), from `cap_pad_token`.
    pub cap_pad_mod: Weight,
    /// `t_embedder.mlp.0`: `Linear(256 → 1024)`.
    pub t_mlp0: Linear,
    /// `t_embedder.mlp.2`: `Linear(1024 → 256)`.
    pub t_mlp1: Linear,
    /// The `[1]` f32 constant [`T_FLIP`], the addend of the time reversal.
    /// A weight because the IR has no scalar-constant op; derived at import.
    pub t_flip: Weight,
    pub noise_refiner: Vec<Block>,
    pub context_refiner: Vec<Block>,
    pub layers: Vec<Block>,
    /// `all_final_layer["2-1"].adaLN_modulation.1`: `Linear(256 → dim)`
    /// behind a SiLU, a scale and nothing else.
    pub final_ada: Linear,
    /// `all_final_layer["2-1"].linear`: `Linear(dim → 64)`.
    pub final_linear: Linear,
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

/// The text encoder: `Qwen3Model` (Qwen3-4B), headless, read at the
/// residual entering its last block.
///
/// Why this is its own declaration and not `qwen_3::model::Model`: the
/// `qwen_3` family in this catalog is Qwen3.5/3.6 — a sigmoid-gated
/// attention projection (`qg_proj`), Gated DeltaNet on three layers in four,
/// partial rotary (64 of 256) at θ 1e7 and `rmsnorm_plus_one` norms —
/// none of which `Qwen3ForCausalLM` has. Lifting a gate-less, all-attention,
/// full-rotary variant into that family means touching its `Attn`, its
/// forward and its import, shared by a dozen rows; the encoder here is ~40
/// lines of declaration and ~60 of forward, and the follow-up to fold it
/// back is recorded in `crates/models/src/z_image.rs`.
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
}

impl TextEncoder {
    fn qwen3_4b(banks: Dtype) -> TextEncoder {
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
    /// `None` on the miniature: its checkpoint is the transformer alone and
    /// its captions are random rows, so it declares no `text` reading.
    pub te: Option<TextEncoder>,
    /// The Turbo checkpoint's scheduler shift (`scheduler_config.json`).
    pub shift: f32,
}

impl Model {
    /// `Tongyi-MAI/Z-Image-Turbo`: the 6.15 B transformer plus Qwen3-4B,
    /// static shift 3.0. The base `Tongyi-MAI/Z-Image` is the same text at
    /// shift 6.0 with CFG; a row for it is a second constructor here.
    #[must_use]
    pub fn turbo(banks: Dtype, tp: u32) -> Model {
        Model::new(
            banks,
            tp,
            Dims::turbo(),
            Some(TextEncoder::qwen3_4b(banks)),
            3.0,
        )
    }

    /// The miniature `zimage_golden.py --mini` writes: the transformer at
    /// [`Dims::mini`], no encoder. What the parity harness drives.
    #[must_use]
    pub fn mini(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::mini(), None, 3.0)
    }

    fn new(banks: Dtype, tp: u32, d: Dims, te: Option<TextEncoder>, shift: f32) -> Model {
        assert_eq!(
            tp, 1,
            "this text ships one-rank rows; tp {tp} is not a world it states"
        );
        assert_eq!(
            d.rope_dims.iter().sum::<u32>(),
            d.head_dim,
            "the rotary axes cover the whole head"
        );
        assert_eq!(
            d.heads * d.head_dim,
            d.dim,
            "plain MHA: heads × head_dim is the width"
        );
        let dense = crate::dense(banks);
        let dim = d.dim;
        let blocks = |stem: &str, count: u32, modulated: bool| -> Vec<Block> {
            (0..count)
                .map(|i| Block::at(&format!("dit.{stem}.{i}"), &d, modulated, banks))
                .collect()
        };
        let dit = Dit {
            x_embed: Linear::at("dit.x_embed", dim, PATCH_FEATURES, banks),
            x_pad_mod: Weight::sym("dit.x_pad_mod", [u64::from(2 * dim), 1], banks),
            cap_norm: Weight::sym("dit.cap_norm", [u64::from(d.cap_width)], dense),
            cap_embed: Linear::at("dit.cap_embed", dim, d.cap_width, banks),
            cap_pad_mod: Weight::sym("dit.cap_pad_mod", [u64::from(2 * dim), 1], banks),
            t_mlp0: Linear::at("dit.t_mlp0", T_MID, T_FREQ_DIM, banks),
            t_mlp1: Linear::at("dit.t_mlp1", ADALN_DIM, T_MID, banks),
            t_flip: Weight::sym("dit.t_flip", [1], Dtype::F32),
            noise_refiner: blocks("noise", d.refiner_layers, true),
            context_refiner: blocks("context", d.refiner_layers, false),
            layers: blocks("layer", d.joint_layers, true),
            final_ada: Linear::at("dit.final_ada", dim, ADALN_DIM, banks),
            final_linear: Linear::at("dit.final", PATCH_FEATURES, dim, banks),
        };
        Model {
            tp,
            banks,
            kv: Dtype::Bf16,
            dims: d,
            dit,
            te,
            shift,
        }
    }
}
