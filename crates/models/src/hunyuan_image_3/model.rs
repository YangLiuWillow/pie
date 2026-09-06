//! The `hunyuan_image_3` declaration: every dimension a Rust constant or a
//! [`Dims`] field, every weight named in the plan's own scheme.
//!
//! Three components under one plan (design D5, D10): the Hunyuan-A13B MoE
//! trunk ([`Layer`], `layer.*`), the conv image head ([`UNetDown`],
//! [`UNetUp`], `patch_embed.*` / `final_layer.*`) and the three timestep
//! embedders (`timestep_emb`, `time_embed`, `time_embed_2`). The numbers
//! are `tencent/HunyuanImage-3.0`'s `config.json` (study
//! `.wiki/imagegen/study/hunyuanimage3.md` §C, §D), restated here because a
//! family's dims are Rust constants and a `config.json` is carried, never
//! read. The miniature is `scripts/imagegen/hy3_golden.py --mini`'s config:
//! two layers, eight experts, a 64-wide head, random weights.
//!
//! # Numerics contract
//!
//! * The dense banks are bf16 and the routed expert banks are U8g64 on the
//!   flagship (design D10: 77 B routed parameters at 8 bits, ≈20 GB a rank
//!   at `tp = 4`); the miniature is bf16 throughout. Token-row activations
//!   are bf16; every matmul accumulates fp32 and rounds once.
//! * Every norm in the trunk is `HunyuanRMSNorm` at [`NORM_EPS`]: fp32
//!   statistics, the gain applied in the input dtype, one rounding.
//! * **RoPE runs BEFORE the QK norms** (the Hunyuan-A13B order, study
//!   §C.1), so `q`/`k` are turned and then normalised per head.
//! * The router is a `[experts, hidden]` `nn.Linear` the reference keeps in
//!   **fp32** and this text declares at the dense dtype: the softmax itself
//!   is fp32 inside `linear.moe_topk_softmax`, the logits that feed it are
//!   not. Stated here as the one known numeric deviation of the trunk.
//! * The three `TimestepEmbedder`s use `nn.GELU` (the erf form); this text
//!   traces `elementwise.gelu{tanh: true}`, the only GELU the CUDA shell
//!   serves. |Δ| ≤ 1e-3 of the activation, on a 256→hidden MLP whose output
//!   is one token row and two adaGN vectors.
//! * The image head's `GroupNorm(32)` runs fp32 Welford per clip per group
//!   (`spatial.group_norm`), its convolutions bf16-in with fp32
//!   accumulation (`IMAGEGEN_CONTRACT.md` §6), and the adaptive
//!   `h·(1 + scale) + shift` is one `elementwise.modulate{ScaleShift}`.

use model_dsl::{Dtype, Weight};

/// `num_train_timesteps` and the static flow shift of
/// `FlowMatchDiscreteScheduler(shift=3.0, reverse=True, solver="euler")`
/// (`generation_config.json`).
pub const TRAIN_STEPS: u32 = 1000;
pub const FLOW_SHIFT: f32 = 3.0;

/// `rms_norm_eps`.
pub const NORM_EPS: f32 = 1e-5;

/// `rope_theta`. The 2-D rope's two axes share it — see [`rope_x_scale`].
pub const ROPE_THETA: f32 = 10_000.0;
/// The rotary space is `(y, x)`.
pub const ROPE_AXES: u8 = 2;

/// **WHY THE `x` COORDINATE IS PRE-SCALED.**
///
/// `build_2d_rope` builds `theta[k] = base^(-2k/d)` for `k = 0..d/2`,
/// reshapes it `[d/4, 2]` and pairs frequency `theta[2j]` with `y` and
/// `theta[2j+1]` with `x`; the `d/2` angles are then `repeat(2)`-ed and
/// applied with `rotate_half`. So the two axes do NOT share one frequency
/// ladder: `x`'s is `y`'s shifted by one rung, `base^(-2/d)`.
///
/// `elementwise.rope_axes` gives each axis a CONTIGUOUS channel block and
/// restarts the ladder inside it (`positions[a] · thetas[a]^(-2i/dims[a])`,
/// `IMAGEGEN_CONTRACT.md` §4). Two things make that exact here:
///
/// 1. `import.rs` permutes each head's `q`/`k` channels (and the QK-norm
///    gains with them) so the `y` pairs occupy `[0, d/2)` and the `x` pairs
///    `[d/2, d)`, in `RopeForm::Split`'s own `(b + i, b + dims/2 + i)`
///    pairing — which turns the reference's one `rotate_half` over
///    `(k, k + d/2)` into two half-width rotate-halves.
/// 2. The half-rung offset is folded into the POSITION: an angle is
///    `position · frequency`, so `x' = x · base^(-2/d)` reproduces
///    `x · theta[2j+1]` exactly. A guest hands `(y, x')`, which is why this
///    family states no [`crate::PositionConvention`].
#[must_use]
pub fn rope_x_scale(head_dim: u32) -> f32 {
    ROPE_THETA.powf(-2.0 / head_dim as f32)
}

/// The sinusoidal timestep embedding: `timestep_embedding(t, 256,
/// max_period = 1e4)` is `[cos | sin]` over `exp(-ln(1e4)·i/half)`, the
/// scheduler timestep (`σ·1000`) going in raw.
pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;

/// `nn.GroupNorm(32, C)` — the image head's every norm.
pub const GN_GROUPS: u32 = 32;
/// `nn.GroupNorm`'s default eps.
pub const GN_EPS: f32 = 1e-5;

/// `patch_size` 1 on the 16× VAE latent: one image token is one latent
/// cell, so a 1024² image is 64 × 64 = 4096 rows.
pub const PATCH: u32 = 1;
/// `vae.latent_channels`.
pub const LATENT_CHANNELS: u32 = 32;
/// `vae_downsample_factor`.
pub const SPATIAL_COMPRESSION: u32 = 16;
/// `scaling_factor` of the 3-D VAE; the guest denormalises with it.
pub const VAE_SCALING: f32 = 0.562_679_2;

/// A 3×3 convolution over a still is a `[1, 3, 3]` `spatial.conv3d`.
pub const CONV3: [u32; 3] = [1, 3, 3];
pub const CONV1: [u32; 3] = [1, 1, 1];

/// The float ports this text reads, by index within their kind. A port
/// index is the family's own (`RuntimeInput::Latents { port, .. }` carries
/// it) and the runtime resolves `input(name)` through
/// [`crate::ReadingFact::ports_indexed`].
pub mod port {
    /// `denoise`: the canvas rows already through [`super::UNetDown`],
    /// `[rows, hidden]` bf16 — the `image.in` reading's readout.
    pub const ROWS: u8 = 0;
    /// `denoise`: `1.0` on the lane's `<timestep>` row and `0.0` on every
    /// image row, `[rows, 1]`. The IR has no row-level class, so the one
    /// special row of the canvas lane is selected by a flag the guest
    /// fills — z_image's `pad` port, used to PICK rather than to blank.
    pub const SPECIAL: u8 = 1;
    /// `denoise`: the scheduler timestep `σ·1000`, `[lanes, 1]`.
    pub const TIMESTEP: u8 = 0;
    /// `encode`, `denoise`: the two rotary coordinates `(y, x')` per row —
    /// `x'` pre-scaled, see [`rope_x_scale`].
    pub const POSITIONS: u8 = 0;
    /// `image.in`: the noisy latent WITH the timestep's sinusoid beside it,
    /// `[h·w, 32 + 256]` on the voxel axis, split by the trace.
    ///
    /// **TWO THINGS FORCE ONE PACKED PORT.** The sinusoid rides per VOXEL
    /// because a voxel-axis lane broadcast does not exist:
    /// `elementwise.modulate` takes a `[Lanes, ·]` vector only through a
    /// `[Tokens]` lane map, and `linear.matmul` takes bf16 activations
    /// only, so the adaGN vector can be neither computed once per lane and
    /// broadcast onto voxel rows nor derived there from an f32 sinusoid.
    /// And it rides in the SAME rectangle because the CUDA shell seats one
    /// voxel width a fire (`IMAGEGEN_CONTRACT.md` §6, "M0: one width a
    /// fire") and refuses a lane that feeds two. Cost: 4096 × 256 f32 =
    /// 4 MB a fire, and the two `Linear`s stay bf16 on the voxel axis with
    /// exactly the reference's arithmetic.
    pub const LATENT_VOXELS: u8 = 0;
    /// `image.out`: the trunk's hidden rows for the image span with the
    /// same sinusoid beside them, `[h·w, hidden + 256]`. Its own index:
    /// the engine seats one rectangle per `(kind, index)` for the whole
    /// plan, and this one is another width.
    pub const ROW_VOXELS: u8 = 1;
}

/// One row's transformer shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    /// `num_experts` / `moe_topk` / `moe_intermediate_size`.
    pub experts: u32,
    pub top_k: u32,
    pub moe_inter: u32,
    /// `intermediate_size · num_shared_expert`.
    pub shared_inter: u32,
    /// `patch_embed_hidden_dim`: the image head's waist.
    pub head_hidden: u32,
}

impl Dims {
    /// `tencent/HunyuanImage-3.0`'s `config.json`.
    #[must_use]
    pub const fn flagship() -> Dims {
        Dims {
            hidden: 4096,
            layers: 32,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            vocab: 133_120,
            experts: 64,
            top_k: 8,
            moe_inter: 3072,
            shared_inter: 3072,
            head_hidden: 1024,
        }
    }

    /// `hy3_golden.py --mini`: two layers, eight experts, a 64-wide head.
    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            hidden: 256,
            layers: 2,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            // The REAL vocabulary even at hidden 256: the checkpoint's
            // `pad_token_id` is 128 009 and `nn.Embedding` refuses a padding
            // index outside its table, so `hy3_golden.py --mini` cannot
            // shrink it.
            vocab: 133_120,
            experts: 8,
            top_k: 2,
            moe_inter: 256,
            shared_inter: 256,
            head_hidden: 64,
        }
    }

    /// `attention_head_dim^-0.5`.
    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    /// The `(y, x)` channel split of one head: half each, the whole head
    /// rotated. See [`rope_x_scale`] for why two equal blocks are exact.
    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        [self.head_dim / 2, self.head_dim / 2, 0, 0]
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

/// One convolution of the image head, declared as the checkpoint stores it
/// (`[C_out, C_in·kh·kw]`, `weight.reshape(C_out, -1)`) and served
/// tap-major by `spatial.conv3d`.
pub struct Conv {
    pub w: Weight,
    pub bias: Weight,
    pub k: [u32; 3],
}

impl Conv {
    fn at(name: &str, c_out: u32, c_in: u32, k: [u32; 3], banks: Dtype) -> Conv {
        let taps = k[0] * k[1] * k[2];
        Conv {
            w: Weight::sym(
                name,
                [u64::from(c_out), u64::from(c_in) * u64::from(taps)],
                banks,
            )
            .conv_taps_major(c_in, taps),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c_out)], Dtype::F32),
            k,
        }
    }
}

/// A `[C]` affine group norm — `nn.GroupNorm(32, C)`'s weight and bias,
/// both f32 as `spatial.group_norm` demands.
pub struct GroupNorm {
    pub weight: Weight,
    pub bias: Weight,
}

impl GroupNorm {
    fn at(name: &str, c: u32) -> GroupNorm {
        GroupNorm {
            weight: Weight::sym(name, [u64::from(c)], Dtype::F32),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c)], Dtype::F32),
        }
    }
}

/// One `ResBlock` of the image head, at `patch_size = 1` (no up/down
/// sampling): `in_layers` (GN → SiLU → 3×3 conv), the adaptive group norm
/// off `emb_layers(SiLU(t))`, `out_layers` (GN → modulate → SiLU → 3×3
/// conv) and the 1×1 skip.
pub struct ResBlock {
    pub norm_in: GroupNorm,
    pub conv_in: Conv,
    /// `emb_layers.1`: `Linear(emb → 2·out)`, the `[scale | shift]` pair in
    /// the reference's own `chunk(2, dim=1)` order.
    pub emb: Linear,
    pub norm_out: GroupNorm,
    pub conv_out: Conv,
    /// `skip_connection`: a 1×1 conv when the width changes, absent when it
    /// does not (`nn.Identity`). Both this family's blocks change it.
    pub skip: Option<Conv>,
}

impl ResBlock {
    fn at(prefix: &str, c_in: u32, c_out: u32, emb: u32, banks: Dtype) -> ResBlock {
        let n = |s: &str| format!("{prefix}.{s}");
        ResBlock {
            norm_in: GroupNorm::at(&n("norm_in"), c_in),
            conv_in: Conv::at(&n("conv_in"), c_out, c_in, CONV3, banks),
            emb: Linear::at(&n("emb"), 2 * c_out, emb, banks),
            norm_out: GroupNorm::at(&n("norm_out"), c_out),
            conv_out: Conv::at(&n("conv_out"), c_out, c_out, CONV3, banks),
            skip: (c_in != c_out).then(|| Conv::at(&n("skip"), c_out, c_in, CONV1, banks)),
        }
    }
}

/// `patch_embed = UNetDown(patch_size=1, in=32, hidden=1024, emb=hidden,
/// out=hidden)`: a 3×3 conv into the waist, then one [`ResBlock`] out to
/// the trunk's width.
pub struct UNetDown {
    pub conv_in: Conv,
    pub res: ResBlock,
}

/// `final_layer = UNetUp(patch_size=1, in=hidden, hidden=1024, out=32,
/// out_norm=True)`: one [`ResBlock`] down to the waist, then GN → SiLU →
/// 3×3 conv to the latent channels.
pub struct UNetUp {
    pub res: ResBlock,
    pub norm_out: GroupNorm,
    pub conv_out: Conv,
}

/// A `TimestepEmbedder`: `Linear(256 → hidden)`, GELU, `Linear(hidden →
/// out)`. `timestep_emb` lands the `<timestep>` TOKEN (and so declares its
/// second linear stacked twice — see [`Model::timestep_emb`]); `time_embed`
/// and `time_embed_2` land the two adaGN conditions.
pub struct Embedder {
    pub mlp_in: Linear,
    pub mlp_out: Linear,
}

impl Embedder {
    fn at(prefix: &str, hidden: u32, out: u32, banks: Dtype) -> Embedder {
        Embedder {
            mlp_in: Linear::at(&format!("{prefix}.in"), hidden, T_FREQ_DIM, banks),
            mlp_out: Linear::at(&format!("{prefix}.out"), out, hidden, banks),
        }
    }
}

/// One `HunyuanImage3DecoderLayer`: pre-norm attention with a fused
/// interleaved `qkv_proj` and per-head QK norms, then pre-norm MoE.
pub struct Layer {
    pub attn_norm: Weight,
    /// `qkv_proj` de-interleaved at import into `[q | k | v]`.
    pub qkv: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub o_proj: Weight,
    pub kv: String,
    pub mlp_norm: Weight,
    /// `mlp.gate.wg`: `[experts, hidden]`.
    pub router: Weight,
    /// `mlp.experts.*.gate_and_up_proj` stacked, halves SWAPPED at import
    /// (the reference is `down(x1 · silu(x2))`, first half UP, and
    /// `linear.mlp_swiglu` is `silu(first) · second`).
    pub experts_gate_up: Weight,
    pub experts_down: Weight,
    pub shared_gate_up: Weight,
    pub shared_down: Weight,
}

/// The whole text.
pub struct Model {
    pub tp: u32,
    /// The dtype the dense banks are stored in.
    pub banks: Dtype,
    /// The dtype the routed expert banks are stored in (`U8g64` on the
    /// flagship, design D10).
    pub expert_banks: Dtype,
    pub kv_dtype: Dtype,
    pub dims: Dims,
    /// Per-rank head counts (`Dims` states the model-wide ones).
    pub q_heads: u32,
    pub kv_heads: u32,
    pub moe_inter: u32,
    pub shared_inter: u32,
    pub embed: Weight,
    pub head: Weight,
    pub final_norm: Weight,
    pub layers: Vec<Layer>,
    /// `timestep_emb`, with its second linear stacked twice so the plan
    /// reads `[t_emb | t_emb]` in one projection: the `<timestep>` row is
    /// landed as `modulate(0, [t_emb | t_emb], ScaleShift) = 0·(1 + t_emb)
    /// + t_emb`, this IR having no lane→row broadcast of its own.
    pub timestep_emb: Embedder,
    pub time_embed: Embedder,
    pub time_embed_2: Embedder,
    pub patch_embed: UNetDown,
    pub final_layer: UNetUp,
    /// A `[2·hidden, 1]` bank of `[+1 | -1]`: the canvas lane's one-column
    /// `special` flag broadcast to a row in both signs at once, so the
    /// `<timestep>` row can be picked out of the canvas without a
    /// row-level class — and so their SUM is a fresh rectangle of zeros,
    /// which is what the lane's timestep vector is broadcast over.
    pub ones: Weight,
}

impl Model {
    /// `tencent/HunyuanImage-3.0`: the 80 B / 13 B-active trunk with
    /// `U8g64` routed experts and bf16 everywhere else (design D10).
    #[must_use]
    pub fn flagship(banks: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(banks, experts, kv, tp, Dims::flagship())
    }

    /// The miniature `hy3_golden.py --mini` writes: the parity fixture.
    #[must_use]
    pub fn mini(banks: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(banks, banks, kv, tp, Dims::mini())
    }

    fn new(banks: Dtype, expert_banks: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        assert!(
            d.head_dim.is_multiple_of(4),
            "the 2-D rope splits a head into two even blocks; {} is not a multiple of 4",
            d.head_dim
        );
        assert!(
            d.q_heads.is_multiple_of(d.kv_heads),
            "GQA: {} query heads do not group into {} kv heads",
            d.q_heads,
            d.kv_heads
        );
        assert!(
            d.head_hidden.is_multiple_of(GN_GROUPS) && d.hidden.is_multiple_of(GN_GROUPS),
            "the image head group-norms {GN_GROUPS} ways"
        );
        let q_heads = d.q_heads / tp;
        let kv_heads = d.kv_heads / tp;
        let moe_inter = d.moe_inter / tp;
        let shared_inter = d.shared_inter / tp;
        assert!(
            q_heads > 0 && kv_heads > 0 && moe_inter > 0,
            "tp {tp} cuts this row past its heads and experts"
        );

        let dense = crate::dense(banks);
        let hidden = u64::from(d.hidden);
        let hd = u64::from(d.head_dim);
        let q_w = u64::from(q_heads) * hd;
        let kv_w = u64::from(kv_heads) * hd;
        let iw = u64::from(moe_inter);
        let sw = u64::from(shared_inter);
        let n_experts = u64::from(d.experts);

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                Layer {
                    attn_norm: Weight::sym(n("attn_norm"), [hidden], dense),
                    qkv: Weight::sym(n("qkv"), [q_w + 2 * kv_w, hidden], banks)
                        .packed([q_w, kv_w, kv_w]),
                    q_norm: Weight::sym(n("q_norm"), [hd], dense),
                    k_norm: Weight::sym(n("k_norm"), [hd], dense),
                    o_proj: Weight::sym(n("o_proj"), [hidden, q_w], banks).rows(),
                    kv: format!("kv.{l}"),
                    mlp_norm: Weight::sym(n("mlp_norm"), [hidden], dense),
                    router: Weight::sym(n("router"), [n_experts, hidden], dense),
                    experts_gate_up: Weight::sym(
                        n("experts_gate_up"),
                        [n_experts, 2 * iw, hidden],
                        expert_banks,
                    )
                    .bank([iw, iw]),
                    experts_down: Weight::sym(
                        n("experts_down"),
                        [n_experts, hidden, iw],
                        expert_banks,
                    )
                    .rows(),
                    shared_gate_up: Weight::sym(n("shared_gate_up"), [2 * sw, hidden], banks)
                        .packed([sw, sw]),
                    shared_down: Weight::sym(n("shared_down"), [hidden, sw], banks).rows(),
                }
            })
            .collect();

        let hw = d.head_hidden;
        Model {
            tp,
            banks,
            expert_banks,
            kv_dtype: kv,
            dims: d,
            q_heads,
            kv_heads,
            moe_inter,
            shared_inter,
            embed: Weight::sym("wte", [u64::from(d.vocab), hidden], banks),
            head: Weight::sym("lm_head", [u64::from(d.vocab), hidden], banks),
            final_norm: Weight::sym("ln_f", [hidden], dense),
            layers,
            // Stacked twice: the plan reads `[t_emb | t_emb]`.
            timestep_emb: Embedder {
                mlp_in: Linear::at("timestep_emb.in", d.hidden, T_FREQ_DIM, banks),
                mlp_out: Linear::at("timestep_emb.out", 2 * d.hidden, d.hidden, banks),
            },
            time_embed: Embedder::at("time_embed", d.hidden, d.hidden, banks),
            time_embed_2: Embedder::at("time_embed_2", d.hidden, d.hidden, banks),
            patch_embed: UNetDown {
                conv_in: Conv::at("patch_embed.conv", hw, LATENT_CHANNELS, CONV3, banks),
                res: ResBlock::at("patch_embed.res", hw, d.hidden, d.hidden, banks),
            },
            final_layer: UNetUp {
                res: ResBlock::at("final_layer.res", d.hidden, hw, d.hidden, banks),
                norm_out: GroupNorm::at("final_layer.norm_out", hw),
                conv_out: Conv::at("final_layer.conv", LATENT_CHANNELS, hw, CONV3, banks),
            },
            // REPLICATED: the flag it spreads modulates a full-width
            // port rectangle, so every rank wants the whole column.
            ones: Weight::sym("special.ones", [2 * hidden, 1], banks),
        }
    }

    /// The width of this rank's `q` rectangle.
    #[must_use]
    pub fn q_width(&self) -> u32 {
        self.q_heads * self.dims.head_dim
    }

    /// The width of this rank's `k` (and `v`) rectangle.
    #[must_use]
    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.dims.head_dim
    }
}
