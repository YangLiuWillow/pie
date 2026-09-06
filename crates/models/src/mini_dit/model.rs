//! The `mini-dit` declaration: every dimension a Rust constant, every weight
//! named in the plan's own scheme.
//!
//! This is the synthetic family the image/video substrate (design D2, D3, D6,
//! D7) is verified against — the smallest text that names a single-stream
//! joint block, an MM-DiT double-stream block and a Wan-style cross-attention
//! block at once. Its reference is `scripts/imagegen/mini_dit_ref.py`; the
//! numbers below are that file's `CONFIG`, restated here because a family's
//! dims are Rust constants and a `config.json` is carried, never read.

use model_dsl::{Dtype, Weight};

/// The trunk width every rectangle in this text is stated in.
pub const HIDDEN: u32 = 256;
/// Query heads. `HEADS * HEAD_DIM == HIDDEN`, so there is no GQA here.
pub const HEADS: u32 = 4;
pub const HEAD_DIM: u32 = 64;
/// SwiGLU intermediate: `HIDDEN * mlp_ratio` at ratio 2.
pub const INTER: u32 = 512;
/// The latent's channels, and the patch extent along each spatial axis.
pub const CHANNELS: u32 = 16;
pub const PATCH: u32 = 2;
/// `CHANNELS * PATCH²` — the width of one patch row, in and out.
pub const PATCH_FEATURES: u32 = CHANNELS * PATCH * PATCH;
/// The caption stream's width: this model's text tokens arrive already at
/// trunk width, so the text port needs no projection.
pub const TEXT_WIDTH: u32 = 256;
/// The cross-attention context's width (block 2 only).
pub const CONTEXT_WIDTH: u32 = 512;
/// The sinusoidal timestep embedding's width.
pub const TIMESTEP_DIM: u32 = 256;
pub const TIMESTEP_MAX_PERIOD: f32 = 10_000.0;
/// `[sin | cos]`, not `[cos | sin]`, and no scaling of the raw timestep.
pub const TIMESTEP_FLIP_SIN_COS: bool = false;
pub const TIMESTEP_SCALE: f32 = 1.0;

/// The three rotary axes `(t, h, w)`, in channels per axis; they sum to
/// [`HEAD_DIM`], so every channel of every head turns.
pub const ROPE_DIMS: [u32; 4] = [16, 24, 24, 0];
pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_AXES: u8 = 3;

/// LayerNorm-no-affine, and the affine cross-attention norm.
pub const LN_EPS: f32 = 1e-6;
/// The QK-norms.
pub const RMS_EPS: f32 = 1e-6;

/// `head_dim^-0.5`, stated once because it is a trace constant.
pub const SM_SCALE: f32 = 0.125;

/// How many `[HIDDEN]` slices an adaLN-Zero vector carries: shift, scale and
/// gate for the attention sublayer, then the same three for the MLP.
pub const MOD_SLICES: u32 = 6;

/// The float ports this text reads, by index. A port index is the family's
/// own — `RuntimeInput::Latents { port, .. }` and friends carry it — and the
/// runtime binds one guest channel per port through
/// [`super::forward::READINGS`].
pub mod port {
    /// The image lane's patch rows, `[rows, PATCH_FEATURES]`.
    pub const LATENTS: u8 = 0;
    /// The caption rows, `[rows, TEXT_WIDTH]`.
    pub const TEXT: u8 = 0;
    /// The cross-attention context rows, `[rows, CONTEXT_WIDTH]`.
    pub const CONTEXT: u8 = 1;
    /// The per-lane timestep, `[lanes, 1]`.
    pub const TIMESTEP: u8 = 0;
    /// The three rotary coordinates per row, `[rows, 3]`.
    pub const POSITIONS: u8 = 0;
}

/// One `nn.Linear`: a bank and the bias beside it. Every projection in this
/// text is biased, which is what makes it a useful exercise of `add_bias`.
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

/// A self-attention sublayer: one packed `q|k|v`, a per-head RMS gain on each
/// of q and k, and the output projection.
pub struct SelfAttn {
    pub qkv: Linear,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Linear,
}

impl SelfAttn {
    fn at(prefix: &str, banks: Dtype) -> SelfAttn {
        let dense = crate::dense(banks);
        SelfAttn {
            qkv: Linear::at(&format!("{prefix}.qkv"), 3 * HIDDEN, HIDDEN, banks),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out: Linear::at(&format!("{prefix}.o"), HIDDEN, HIDDEN, banks),
        }
    }
}

/// The cross-attention sublayer: queries off the image rectangle, keys and
/// values off a wider context rectangle through one packed `k|v`.
pub struct CrossAttn {
    pub q: Linear,
    pub kv: Linear,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Linear,
}

impl CrossAttn {
    fn at(prefix: &str, banks: Dtype) -> CrossAttn {
        let dense = crate::dense(banks);
        CrossAttn {
            q: Linear::at(&format!("{prefix}.q"), HIDDEN, HIDDEN, banks),
            kv: Linear::at(&format!("{prefix}.kv"), 2 * HIDDEN, CONTEXT_WIDTH, banks),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out: Linear::at(&format!("{prefix}.o"), HIDDEN, HIDDEN, banks),
        }
    }
}

/// The gated MLP: `down(silu(gate) · up)`, the two halves stored packed so
/// one matmul and one `linear.mlp_swiglu` serve them.
pub struct Swiglu {
    pub gate_up: Linear,
    pub down: Linear,
}

impl Swiglu {
    fn at(prefix: &str, banks: Dtype) -> Swiglu {
        // `.packed` states the seam between the two halves: the checkpoint
        // ships them as two tensors and `Builder::read_concat` fuses them
        // along it. At one rank it cuts nothing; it is the seam that is the
        // fact, not the width.
        let name = format!("{prefix}.gate_up");
        let seams = [u64::from(INTER), u64::from(INTER)];
        Swiglu {
            gate_up: Linear {
                w: Weight::sym(&name, [u64::from(2 * INTER), u64::from(HIDDEN)], banks)
                    .packed(seams),
                bias: Weight::sym(
                    format!("{name}.bias"),
                    [u64::from(2 * INTER)],
                    crate::dense(banks),
                )
                .packed(seams),
            },
            down: Linear::at(&format!("{prefix}.down"), HIDDEN, INTER, banks),
        }
    }
}

/// Block 0: text and image in ONE sequence, one set of weights, one adaLN.
pub struct Single {
    pub ada: Linear,
    pub attn: SelfAttn,
    pub mlp: Swiglu,
}

/// One side of block 1: an MM-DiT stream has its own modulation, its own
/// projections and its own MLP, and shares only the attention itself.
pub struct Side {
    pub ada: Linear,
    pub attn: SelfAttn,
    pub mlp: Swiglu,
}

impl Side {
    fn at(prefix: &str, banks: Dtype) -> Side {
        Side {
            ada: Linear::at(&format!("{prefix}.ada"), MOD_SLICES * HIDDEN, HIDDEN, banks),
            attn: SelfAttn::at(&format!("{prefix}.attn"), banks),
            mlp: Swiglu::at(&format!("{prefix}.mlp"), banks),
        }
    }
}

/// Block 1: the double-stream block, two sides over one joint attention.
pub struct Double {
    pub img: Side,
    pub txt: Side,
}

/// Block 2: Wan's shape — self-attention, then cross-attention into the
/// context lane, then the FFN, with the modulation read off a learned table
/// PLUS the timestep projection.
pub struct Cross {
    /// Wan's `scale_shift_table`, stored `[6, HIDDEN]` and read as one
    /// `[MOD_SLICES * HIDDEN]` bias over the projected vector.
    pub mod_table: Weight,
    pub ada: Linear,
    pub self_attn: SelfAttn,
    /// The pre-cross LayerNorm, which here has affine parameters and takes
    /// no modulation at all.
    pub norm: Weight,
    pub norm_bias: Weight,
    pub cross: CrossAttn,
    pub mlp: Swiglu,
}

/// The whole text.
pub struct Model {
    pub tp: u32,
    /// The dtype the banks are stored in — one row, `Bf16`, today.
    pub banks: Dtype,
    pub x_embed: Linear,
    pub single: Single,
    pub double: Double,
    pub cross: Cross,
    /// The head's two-slice modulation, in the plan's `[scale | shift]`
    /// order.
    pub final_ada: Linear,
    pub final_proj: Linear,
    /// The parity harness's bisection knob: the golden dump key of an
    /// intermediate to plant [`model_dsl::seam::VELOCITY`] on INSTEAD of the
    /// head's output (`forward::Tap`). `None` is the model; a key is a
    /// probe, and the readout width follows it.
    pub tap: Option<String>,
}

impl Model {
    /// The one shape this family ships. `tp` is a column for the catalog's
    /// sake: every rectangle here is 256 wide and nothing is worth cutting,
    /// so this text ships one-rank rows only and every weight is replicated.
    #[must_use]
    pub fn mini(banks: Dtype, tp: u32) -> Model {
        let dense = crate::dense(banks);
        Model {
            tp,
            banks,
            x_embed: Linear::at("x_embed", HIDDEN, PATCH_FEATURES, banks),
            single: Single {
                ada: Linear::at("single.ada", MOD_SLICES * HIDDEN, HIDDEN, banks),
                attn: SelfAttn::at("single.attn", banks),
                mlp: Swiglu::at("single.mlp", banks),
            },
            double: Double {
                img: Side::at("double.img", banks),
                txt: Side::at("double.txt", banks),
            },
            cross: Cross {
                mod_table: Weight::sym("cross.mod_table", [u64::from(MOD_SLICES * HIDDEN)], dense),
                ada: Linear::at("cross.ada", MOD_SLICES * HIDDEN, HIDDEN, banks),
                self_attn: SelfAttn::at("cross.self", banks),
                norm: Weight::sym("cross.norm", [u64::from(HIDDEN)], dense),
                norm_bias: Weight::sym("cross.norm.bias", [u64::from(HIDDEN)], dense),
                cross: CrossAttn::at("cross.x", banks),
                mlp: Swiglu::at("cross.mlp", banks),
            },
            final_ada: Linear::at("final.ada", 2 * HIDDEN, HIDDEN, banks),
            final_proj: Linear::at("final.proj", PATCH_FEATURES, HIDDEN, banks),
            tap: None,
        }
    }

    /// The same text, with one intermediate exported in the velocity's
    /// place. See [`super::forward::Tap`].
    #[must_use]
    pub fn tapped(mut self, tap: Option<String>) -> Model {
        self.tap = tap;
        self
    }
}
