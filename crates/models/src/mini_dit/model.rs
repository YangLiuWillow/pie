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

    /// Cut along the OUTPUT axis: each rank lands a column block of the
    /// weight and the matching block of the bias (the projection's answer
    /// is that rank's heads / its slice of the intermediate).
    fn columns(self) -> Linear {
        Linear {
            w: self.w.columns(),
            bias: self.bias.columns(),
        }
    }

    /// Cut along the REDUCTION axis: each rank lands a row block of the
    /// weight; the partial products meet in an `all_reduce` and the bias —
    /// replicated — is added once, after it (`forward::linear_reduced`).
    fn rows(self) -> Linear {
        Linear {
            w: self.w.rows(),
            bias: self.bias,
        }
    }

    /// Cut axis 0 at the stated seams, weight and bias alike — a fused
    /// projection (`qkv`, `kv`) whose halves must not straddle ranks.
    fn packed(self, seams: impl IntoIterator<Item = u64> + Clone) -> Linear {
        Linear {
            w: self.w.packed(seams.clone()),
            bias: self.bias.packed(seams),
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
    /// A declared shape is the rank's OWN band (`tp` of them make the
    /// checkpoint's plane along the cut axis): the fused projection is cut
    /// at its three seams (each rank lands its `HIDDEN / tp` of q, k and v);
    /// the QK-norm gains are per head width and stay whole; the output
    /// projection reduces across ranks.
    fn at(prefix: &str, banks: Dtype, tp: u32) -> SelfAttn {
        let dense = crate::dense(banks);
        let mine = HIDDEN / tp;
        let seams = [u64::from(mine); 3];
        SelfAttn {
            qkv: Linear::at(&format!("{prefix}.qkv"), 3 * mine, HIDDEN, banks).packed(seams),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out: Linear::at(&format!("{prefix}.o"), HIDDEN, mine, banks).rows(),
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
    fn at(prefix: &str, banks: Dtype, tp: u32) -> CrossAttn {
        let dense = crate::dense(banks);
        let mine = HIDDEN / tp;
        CrossAttn {
            q: Linear::at(&format!("{prefix}.q"), mine, HIDDEN, banks).columns(),
            kv: Linear::at(&format!("{prefix}.kv"), 2 * mine, CONTEXT_WIDTH, banks)
                .packed([u64::from(mine); 2]),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out: Linear::at(&format!("{prefix}.o"), HIDDEN, mine, banks).rows(),
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
    fn at(prefix: &str, banks: Dtype, tp: u32) -> Swiglu {
        // `.packed` states the seam between the two halves: the checkpoint
        // ships them as two tensors and `Builder::read_concat` fuses them
        // along it. At one rank it cuts nothing; it is the seam that is the
        // fact, not the width.
        let name = format!("{prefix}.gate_up");
        let mine = INTER / tp;
        let seams = [u64::from(mine), u64::from(mine)];
        Swiglu {
            gate_up: Linear {
                w: Weight::sym(&name, [u64::from(2 * mine), u64::from(HIDDEN)], banks)
                    .packed(seams),
                bias: Weight::sym(
                    format!("{name}.bias"),
                    [u64::from(2 * mine)],
                    crate::dense(banks),
                )
                .packed(seams),
            },
            down: Linear::at(&format!("{prefix}.down"), HIDDEN, mine, banks).rows(),
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
    fn at(prefix: &str, banks: Dtype, tp: u32) -> Side {
        Side {
            ada: Linear::at(&format!("{prefix}.ada"), MOD_SLICES * HIDDEN, HIDDEN, banks),
            attn: SelfAttn::at(&format!("{prefix}.attn"), banks, tp),
            mlp: Swiglu::at(&format!("{prefix}.mlp"), banks, tp),
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
    /// The one shape this family ships, at one rank or several.
    ///
    /// **The `tp` convention (design D14).** The plan is SPMD; a rank's
    /// share is what its cut weights land, and the widths the forward
    /// spells are per rank:
    ///
    /// - every self- and cross-attention projection is cut BY HEADS —
    ///   `qkv` at its three seams, cross `q` by columns, cross `kv` at its
    ///   two seams — so a rank holds `HEADS / tp` heads of each and its
    ///   `attention.ragged` runs over them with no collective (the softmax
    ///   is per head);
    /// - every projection back into the residual (`attn.o`, `cross.x.o`,
    ///   `mlp.down`) is cut BY ROWS: the partial products `all_reduce`, and
    ///   the bias, replicated, is added once after the reduction;
    /// - `mlp.gate_up` is cut at its gate/up seam (each rank its `INTER /
    ///   tp` of both halves, so the SwiGLU is local);
    /// - the adaLN linears, `x_embed`, the head (`final.*`), every norm gain
    ///   and the QK-norm gains are replicated (per-lane vectors and `[HEAD_DIM]`
    ///   gains are not worth cutting; `final.proj` reads the reduced residual).
    ///
    /// `tp` must divide `HEADS` (4) and `INTER` (512): 1, 2 and 4. An
    /// artifact imported at one rank serves every width, each rank reading
    /// its band at load.
    #[must_use]
    pub fn mini(banks: Dtype, tp: u32) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4),
            "tp {tp} does not divide mini-dit's {HEADS} heads"
        );
        let dense = crate::dense(banks);
        Model {
            tp,
            banks,
            x_embed: Linear::at("x_embed", HIDDEN, PATCH_FEATURES, banks),
            single: Single {
                ada: Linear::at("single.ada", MOD_SLICES * HIDDEN, HIDDEN, banks),
                attn: SelfAttn::at("single.attn", banks, tp),
                mlp: Swiglu::at("single.mlp", banks, tp),
            },
            double: Double {
                img: Side::at("double.img", banks, tp),
                txt: Side::at("double.txt", banks, tp),
            },
            cross: Cross {
                mod_table: Weight::sym("cross.mod_table", [u64::from(MOD_SLICES * HIDDEN)], dense),
                ada: Linear::at("cross.ada", MOD_SLICES * HIDDEN, HIDDEN, banks),
                self_attn: SelfAttn::at("cross.self", banks, tp),
                norm: Weight::sym("cross.norm", [u64::from(HIDDEN)], dense),
                norm_bias: Weight::sym("cross.norm.bias", [u64::from(HIDDEN)], dense),
                cross: CrossAttn::at("cross.x", banks, tp),
                mlp: Swiglu::at("cross.mlp", banks, tp),
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
