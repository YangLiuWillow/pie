//! Z-Image's traced arithmetic: three readings of one plan, selected per
//! lane by the reading bits of the fact word (design D1, D5).
//!
//! | reading | lanes (stream) | binds | reads back |
//! |---|---|---|---|
//! | `text` | one, `Text` | `embed(ids)`, `attention(kv)` | `hidden` `[L, 2560]`: Qwen3 layer −2 |
//! | `refine` | one, `Context` | `caption` `[L32, 2560]`, `pad` `[L32, 1]`, `positions` `[L32, 3]` | `hidden` `[L32, 3840]`: the refined caption |
//! | `denoise` | `Image` + `Context`, one group | image: `latents` `[N32, 64]`, `pad` `[N32, 1]`, `positions`, `timestep`; context: `context` `[L32, 3840]`, `positions`, `timestep` | `velocity` `[N32, 64]` on the image lane |
//!
//! `L32`/`N32` are the caption / image row counts padded up to a multiple of
//! [`super::model::SEQ_MULTIPLE`]. **The pad rows are the guest's to
//! allocate and this text's to fill**: the IR cannot grow a lane, so a lane
//! arrives already padded, its `pad` port flags each row (`0.0` real, `1.0`
//! pad), and the plan overwrites every flagged row with the learned
//! `x_pad_token` / `cap_pad_token` after its embedder (the reference's
//! `torch.where(mask, pad_token, feats)`), at rotary position `(0, 0, 0)` —
//! which the guest's positions must state. The pad rows are attended (study
//! §C.3); the velocity rows they produce are discarded by the guest. The
//! `denoise` context lane binds no `pad`: its rows are the `refine`
//! readout, pads included. The `timestep` is bound by BOTH denoise lanes
//! (the same cell): the joint trunk modulates every row by its own lane's
//! vector, caption rows included.
//!
//! Positions (`AxisPositions`, `[rows, 3]` f32, `(t, h, w)`): caption row
//! `j` is `(1 + j, 0, 0)`; image patch `(a, b)` is `(L32 + 1, a, b)` — the
//! image's temporal index depends on the caption's padded length (study
//! §C.5); pad rows `(0, 0, 0)`.
//!
//! The timestep port takes the SCHEDULER timestep `σ · 1000` (what a generic
//! `FlowMatchEuler` guest holds); the plan performs the reference's time
//! reversal `u = 1000 − t` itself, and negates the velocity it hands back
//! (the reference pipeline's `noise_pred = -noise_pred`), so `x ← x +
//! (σ' − σ) · velocity` is the guest's whole step.
//!
//! The three readouts live in one plan: `velocity` on the denoise arm, one
//! `hidden` on the text arm (layer mark `TE_LAYERS − 1`) and one on the
//! refine arm (layer mark `refiner_layers − 1`). Each export is live in its
//! own class only (`engine_cuda::exports::Export::classes`), so a lane's
//! readback must be chosen by its class — the shell that picks one readout
//! seam per plan reads a text lane's rows off the velocity plane.
//!
//! Every attention here is `attention.ragged` over packed rows: the
//! refiners over one lane's rows (`lane_indptr`), the joint trunk over the
//! request's group (`group_indptr`), whose packed order — by stream code,
//! Image (1) before Context (4) — is the reference's `[image ‖ caption]`.

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    Generative, LatentSpace, PortFact, PortKind, ReadingFact, ReadoutKind, ScheduleFact,
    ScheduleKind,
};

use super::model::{
    ADALN_DIM, Block, Dims, Dit, FINAL_LN_EPS, Linear, MOD_SLICES, Model, NORM_EPS, PATCH_FEATURES,
    ROPE_AXES, ROPE_THETA, T_FLIP_SIN_COS, T_FREQ_DIM, T_MAX_PERIOD, TE_HIDDEN, TE_LAYERS,
    TE_MAX_TOKENS, TRAIN_STEPS, TextEncoder, port,
};
use super::model::{CHANNELS, PATCH, SPATIAL_COMPRESSION};

/// The bit the one-hot stream facts start at (D2): bits 0..6 are the six
/// streams, of which this text names Text, Image and Context.
pub const STREAM_BASE: u8 = 0;

/// The two bits the reading index lives in, as a plain binary code: bit
/// [`READING_LO`] is its low bit, [`READING_HI`] its high bit. Four codes,
/// three readings and one reserved for the VAE.
pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;

/// Which reading index means what, per row. The word packs the index the
/// runtime stamps (`Request::reading`) and nothing else — `Classify::of` has
/// no model to ask — so the *meaning* of a code is the row's: the flagship
/// runs `text` at 0, the miniature (no encoder) runs `refine` at 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    pub text: Option<u8>,
    pub refine: u8,
    pub denoise: u8,
    /// Reserved: the VAE decode reading, once the voxel axis is read by
    /// this text. No node is guarded on it today.
    pub vae_decode: u8,
}

impl Model {
    /// This row's reading codes, dense from 0 in `Generative::readings`
    /// order (what `validate_generative` demands).
    #[must_use]
    pub fn readings(&self) -> Readings {
        match self.te {
            Some(_) => Readings {
                text: Some(0),
                refine: 1,
                denoise: 2,
                vae_decode: 3,
            },
            None => Readings {
                text: None,
                refine: 0,
                denoise: 1,
                vae_decode: 2,
            },
        }
    }

    /// This row's generative facts (design D12).
    #[must_use]
    pub fn generative(&self) -> Generative {
        let d = &self.dims;
        let codes = self.readings();
        let port = |name, kind, width, streams: &[Stream]| PortFact {
            name,
            kind,
            width,
            streams: streams.to_vec(),
        };
        let mut readings = Vec::new();
        if let (Some(index), Some(te)) = (codes.text, &self.te) {
            readings.push(ReadingFact {
                name: "text",
                index,
                has_kv: true,
                takes_tokens: true,
                streams: vec![Stream::Text],
                // A sequence lane: ids and kv, no float port.
                ports: vec![],
                readout: ReadoutKind::Hidden,
                readout_width: te.hidden,
            });
        }
        let axes = u32::from(ROPE_AXES);
        // Port ORDER is load-bearing: a port's index is its position among
        // its kind, and `model::port` numbers them so.
        readings.push(ReadingFact {
            name: "refine",
            index: codes.refine,
            has_kv: false,
            takes_tokens: false,
            streams: vec![Stream::Context],
            ports: vec![
                port("pad", PortKind::Latents, 1, &[Stream::Context]),
                port(
                    "caption",
                    PortKind::Context,
                    d.cap_width,
                    &[Stream::Context],
                ),
                port(
                    "positions",
                    PortKind::AxisPositions,
                    axes,
                    &[Stream::Context],
                ),
            ],
            readout: ReadoutKind::Hidden,
            readout_width: d.dim,
        });
        readings.push(ReadingFact {
            name: "denoise",
            index: codes.denoise,
            has_kv: false,
            takes_tokens: false,
            streams: vec![Stream::Image, Stream::Context],
            ports: vec![
                port(
                    "latents",
                    PortKind::Latents,
                    PATCH_FEATURES,
                    &[Stream::Image],
                ),
                port("pad", PortKind::Latents, 1, &[Stream::Image]),
                port("context", PortKind::Context, d.dim, &[Stream::Context]),
                port(
                    "timestep",
                    PortKind::LaneVector,
                    1,
                    &[Stream::Image, Stream::Context],
                ),
                port(
                    "positions",
                    PortKind::AxisPositions,
                    axes,
                    &[Stream::Image, Stream::Context],
                ),
            ],
            readout: ReadoutKind::Velocity,
            readout_width: PATCH_FEATURES,
        });
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: CHANNELS,
                patch_t: 1,
                patch_h: PATCH,
                patch_w: PATCH,
                spatial_compression: SPATIAL_COMPRESSION,
                temporal_compression: 1,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: self.shift,
                train_steps: TRAIN_STEPS,
                boundary: None,
                pinned_sigmas: turbo_sigmas(self.shift),
            }),
            // 2048² at 16 px per row is 16 384 image rows, plus the widest
            // caption; the miniature's reference grid is 8 × 8.
            max_rows: match self.te {
                Some(_) => 16_384 + TE_MAX_TOKENS,
                None => 4096,
            },
        }
    }
}

/// The Turbo checkpoint's own eight-step schedule: `linspace(1, 1/8, 8)`
/// through the static shift `σ' = s·σ / (1 + (s − 1)·σ)`
/// (`use_dynamic_shifting: false`). Pinned because the checkpoint is
/// DMD-distilled against these sigmas (study §L.6); a guest asking for
/// another step count gets a resampling of them, never a dynamic shift.
#[must_use]
pub fn turbo_sigmas(shift: f32) -> Vec<f32> {
    (0..8)
        .map(|i| 1.0 - i as f32 / 8.0)
        .map(|sigma| shift * sigma / (1.0 + (shift - 1.0) * sigma))
        .collect()
}

/// The per-lane facts: which stream the lane's rows are, and which reading
/// its pass runs.
pub struct Facts {
    pub stream: Stream,
    /// The reading index, `0..4` (a wider index is truncated to two bits).
    pub reading: u8,
}

impl Facts {
    #[must_use]
    pub fn text() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Text)
    }

    #[must_use]
    pub fn image() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Image)
    }

    #[must_use]
    pub fn context() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Context)
    }

    /// The low reading bit.
    #[must_use]
    pub fn reading_lo() -> Predicate {
        Predicate::fact(READING_LO)
    }

    /// The high reading bit.
    #[must_use]
    pub fn reading_hi() -> Predicate {
        Predicate::fact(READING_HI)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            stream: r.stream(),
            reading: r.reading() & 3,
        }
    }

    fn word(&self) -> u64 {
        self.stream.word(STREAM_BASE) | (u64::from(self.reading & 3) << READING_LO)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    /// The encoder's kv space, one row per layer it runs; nothing else is
    /// held between fires. The miniature declares no cache at all.
    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        if let Some(te) = &self.te {
            let kv = c.kv_space(self.kv);
            let plane = u64::from(te.kv_heads) * u64::from(te.head_dim);
            for layer in &te.layers {
                c.kv(kv, layer.kv.clone(), [plane, plane]);
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let codes = self.readings();
        // Four arms by reading code, each a conjunction of the two reading
        // literals (so every arm names a `Selection` the host can pack).
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (c3, c2) = hi.split(&Facts::reading_lo());
        let (c1, c0) = lo.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3];
        let arm = |code: u8| &arms[usize::from(code)];

        if let (Some(code), Some(te)) = (codes.text, &self.te) {
            text_encode(arm(code), te);
        }
        refine(arm(codes.refine), &self.dims, &self.dit);
        let velocity = denoise(arm(codes.denoise), &self.dims, &self.dit);
        vae_decode(arm(codes.vae_decode));
        velocity
    }
}

/// The `text` reading: Qwen3-4B, prefill only, causal over the paged kv,
/// `hidden` planted on the residual leaving layer `TE_LAYERS − 1` — the
/// reference's `hidden_states[-2]`, before any final norm. No head.
fn text_encode(arm: &Input<Facts>, te: &TextEncoder) {
    let plan = ops::attn::plan_prefill(arm, te.q_heads, te.kv_heads, te.head_dim, None);
    let ids = arm.tokens();
    let positions = arm.positions();
    let mut y = ops::layout::embed(&ids, &te.embed, te.vocab);
    let last = te.layers.len() - 1;
    debug_assert_eq!(te.layers.len(), TE_LAYERS as usize);
    for (l, w) in arm.walk_layers(&te.layers) {
        let pages = arm.kv(&w.kv);
        let x = ops::elemwise::rmsnorm(&y, &w.attn_norm, te.eps);
        let q = ops::linear::matmul(&x, &w.q);
        let k = ops::linear::matmul(&x, &w.k);
        let v = ops::linear::matmul(&x, &w.v);
        let q = ops::elemwise::rmsnorm_per_head(&q, &w.q_norm, te.head_dim, te.eps);
        let k = ops::elemwise::rmsnorm_per_head(&k, &w.k_norm, te.head_dim, te.eps);
        // HF `rotate_half` over the whole head: the neox pairing.
        let (q, k) = ops::elemwise::rope_full(&q, &k, &positions, te.head_dim, te.theta, false);
        ops::attn::kv_append(
            &k,
            &v,
            pages,
            &arm.write_page(&w.kv),
            &arm.write_offset(&w.kv),
        );
        let o = ops::attn::prefill(
            &q,
            &plan,
            pages,
            None,
            te.head_dim,
            te.kv_heads,
            te.sm_scale,
        );
        let o = ops::linear::matmul(&o, &w.o);
        y = ops::elemwise::residual_add(&o, &y);

        let x = ops::elemwise::rmsnorm(&y, &w.mlp_norm, te.eps);
        let f = ops::linear::matmul(
            &ops::linear::mlp_swiglu(&ops::linear::matmul(&x, &w.gate_up), te.inter),
            &w.down,
        );
        y = ops::elemwise::residual_add(&f, &y);
        if l as usize == last {
            seam::at(seam::HIDDEN, &[&y]);
        }
    }
}

/// The `refine` reading: `cap_embedder` (RMSNorm → Linear) over the raw
/// caption rows, the pad rows overwritten with `cap_pad_token`, then the
/// unmodulated context-refiner blocks over the lane; `hidden` planted on
/// the result inside the last block's layer mark.
fn refine(arm: &Input<Facts>, d: &Dims, m: &Dit) {
    let c = arm.context(port::CONTEXT, d.cap_width);
    let c = ops::elemwise::rmsnorm(&c, &m.cap_norm, NORM_EPS);
    let c = linear(&m.cap_embed, &c);
    let c = pad_rows(
        &c,
        &arm.latents(port::PAD_CAPTION, 1, Dtype::Bf16),
        &m.cap_pad_mod,
    );
    let geom = Geom {
        positions: arm.axis_positions(port::POSITIONS, ROPE_AXES),
        perm: arm.row_permutation(),
        csr: arm.lane_indptr(),
        mask: RaggedMask::None,
    };
    let mut c = c;
    let last = m.context_refiner.len() - 1;
    for (l, block) in arm.walk_layers(&m.context_refiner) {
        c = run_block(&c, block, None, d, &geom);
        if l as usize == last {
            seam::at(seam::HIDDEN, &[&c]);
        }
    }
}

/// The `denoise` reading: the image lane through `x_embedder` and the
/// modulated noise refiners, joined with the refined caption lane for the
/// joint trunk, read out through the final layer on the image rows.
fn denoise(arm: &Input<Facts>, d: &Dims, m: &Dit) -> Value {
    let (img, ctx) = arm.split(&Facts::image());

    // Reading-wide tables, read once under the reading's guard.
    let lanes = arm.request_of_token();
    let positions = arm.axis_positions(port::POSITIONS, ROPE_AXES);
    let joint = Geom {
        positions: positions.clone(),
        perm: arm.row_permutation(),
        csr: arm.group_indptr(),
        mask: RaggedMask::GroupBlockDiagonal,
    };

    // The timestep: `u = 1000 − t`, `[cos | sin]` sinusoid, the two-layer
    // MLP — all f32, once per lane.
    let t = arm.lane_vector(port::TIMESTEP, 1);
    // `add` is the one fresh copy of a `[Lanes, 1]` row this IR has; the
    // two in-place steps after it never touch the port's own cell.
    let u = ops::elemwise::mul_scalar(-0.5, &ops::elemwise::add(&t, &t));
    let u = ops::elemwise::add_bias(&m.t_flip, &u);
    let temb = ops::elemwise::sinusoid(&u, T_FREQ_DIM, T_MAX_PERIOD, T_FLIP_SIN_COS, 1.0);
    let temb = linear(&m.t_mlp1, &ops::elemwise::silu(&linear(&m.t_mlp0, &temb)));
    debug_assert_eq!(temb.width(), u64::from(ADALN_DIM));

    // ---- the image lane: embed, pad, refine ------------------------------
    let (img_lanes, _) = lanes.split(&Facts::image());
    let (img_positions, _) = positions.split(&Facts::image());
    let (temb_img, _) = temb.split(&Facts::image());
    let own = Geom {
        positions: img_positions,
        perm: img.row_permutation(),
        csr: img.lane_indptr(),
        mask: RaggedMask::None,
    };
    let x = img.latents(port::LATENTS, PATCH_FEATURES, Dtype::Bf16);
    let x = linear(&m.x_embed, &x);
    let mut x = pad_rows(
        &x,
        &img.latents(port::PAD_IMAGE, 1, Dtype::Bf16),
        &m.x_pad_mod,
    );
    for (_, block) in img.walk_layers(&m.noise_refiner) {
        let mods = adaln4(
            &linear(
                block.ada.as_ref().expect("a noise refiner is modulated"),
                &temb_img,
            ),
            d.dim,
        );
        x = run_block(&x, block, Some((&mods, &img_lanes)), d, &own);
    }

    // ---- the caption lane: already refined, pads included ---------------
    let c = ctx.context(port::CONTEXT, d.dim);

    // ---- the joint trunk over `[image ‖ caption]` ------------------------
    let mut u = Value::merge(vec![x, c]);
    for (_, block) in arm.walk_layers(&m.layers) {
        let mods = adaln4(
            &linear(
                block.ada.as_ref().expect("a joint block is modulated"),
                &temb,
            ),
            d.dim,
        );
        u = run_block(&u, block, Some((&mods, &lanes)), d, &joint);
    }

    // ---- the final layer, image rows only ---------------------------------
    // `SiLU → Linear(256 → dim)`, a scale and nothing else, over a
    // non-affine LayerNorm; then the projection back to patch rows.
    let scale = linear(&m.final_ada, &ops::elemwise::silu(&temb));
    let (scale_img, _) = scale.split(&Facts::image());
    let (ui, _) = u.split(&Facts::image());
    let h = ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(&ui, FINAL_LN_EPS),
        &scale_img,
        Some(&img_lanes),
        ModulateForm::Scale,
    );
    let v = linear(&m.final_linear, &h);
    // The reference pipeline's `noise_pred = -noise_pred`.
    let velocity = ops::elemwise::mul_scalar(-1.0, &v);
    seam::at(seam::VELOCITY, &[&velocity]);
    velocity
}

/// The `vae.decode` reading: NOT WIRED. The FLUX `AutoencoderKL` decoder is
/// a voxel-axis text (`IMAGEGEN_CONTRACT.md` §6: `spatial::conv3d`,
/// `group_norm`, `upsample_nearest`, the mid-block attention) that this
/// family will state under this arm once the `vae.` tensors are declared
/// (`import.rs` lists them). Until then a lane stamped with this code runs
/// no node at all; the reading is reserved in [`Readings`] and absent from
/// the facts, so a guest cannot name it.
fn vae_decode(_arm: &Input<Facts>) {}

/// The tables one attention sublayer needs: the rows' rotary coordinates,
/// the arm's row permutation, the CSR its segments pair by, and the mask.
struct Geom {
    positions: Value,
    perm: Value,
    csr: Value,
    mask: RaggedMask,
}

/// One biased projection.
fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

/// Overwrite the flagged rows with the learned pad token: the `[rows, 1]`
/// flag projected through the `[2·dim, 1]` bank to a per-row `[−f | f·pad]`
/// scale-shift, applied with one modulate. A real row (`f = 0`) is
/// `x·(1+0)+0`; a pad row (`f = 1`) is `x·(1−1)+pad`.
fn pad_rows(x: &Value, flag: &Value, bank: &Weight) -> Value {
    let m = ops::linear::matmul(flag, bank);
    ops::elemwise::modulate(x, &m, None, ModulateForm::ScaleShift)
}

/// The four adaLN slices a modulated block applies, in the checkpoint's own
/// `[scale_msa | gate_msa | scale_mlp | gate_mlp]` order, the gates already
/// through `tanh` (once per lane, here, rather than once per row inside
/// the fold).
struct Mods {
    scale_msa: Value,
    gate_msa: Value,
    scale_mlp: Value,
    gate_mlp: Value,
}

fn adaln4(m: &Value, dim: u32) -> Mods {
    debug_assert_eq!(m.width(), u64::from(MOD_SLICES * dim));
    let (scale_msa, rest) = ops::layout::split_rows(m, dim);
    let (gate_msa, rest) = ops::layout::split_rows(&rest, dim);
    let (scale_mlp, gate_mlp) = ops::layout::split_rows(&rest, dim);
    Mods {
        scale_msa,
        gate_msa: ops::elemwise::tanh(&gate_msa),
        scale_mlp,
        gate_mlp: ops::elemwise::tanh(&gate_mlp),
    }
}

/// One `ZImageTransformerBlock`:
///
/// ```text
/// h = attn(attention_norm1(x) · (1 + s_msa))
/// x = x + tanh(g_msa) · attention_norm2(h)
/// h = ffn(ffn_norm1(x) · (1 + s_mlp))
/// x = x + tanh(g_mlp) · ffn_norm2(h)
/// ```
///
/// with the scales and gates dropped for the unmodulated kind. Attention is
/// `to_qkv` → per-head QK RMSNorm → three-axis interleaved rope → ragged
/// attention over the packed segments → `to_out`; the MLP is SwiGLU.
fn run_block(x: &Value, b: &Block, mods: Option<(&Mods, &Value)>, d: &Dims, g: &Geom) -> Value {
    let dim = d.dim;
    let hd = d.head_dim;
    let scaled = |normed: &Value, s: &Value, lanes: &Value| {
        ops::elemwise::modulate(normed, s, Some(lanes), ModulateForm::Scale)
    };

    let h = ops::elemwise::rmsnorm(x, &b.attn_norm1, NORM_EPS);
    let h = match mods {
        Some((m, lanes)) => scaled(&h, &m.scale_msa, lanes),
        None => h,
    };
    let (q, k, v) = ops::layout::split_qkv(&ops::linear::matmul(&h, &b.attn.qkv), dim, dim);
    let turn = |x: &Value, gain: &Weight| {
        ops::elemwise::rope_axes(
            &ops::elemwise::rmsnorm_per_head(x, gain, hd, NORM_EPS),
            &g.positions,
            d.rope_dims,
            [ROPE_THETA; 4],
            RopeForm::Interleaved,
            hd,
            hd,
        )
    };
    let (q, k) = (turn(&q, &b.attn.q_norm), turn(&k, &b.attn.k_norm));
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&q, &g.perm),
        &ops::layout::pack_rows(&k, &g.perm),
        &ops::layout::pack_rows(&v, &g.perm),
        &g.csr,
        &g.csr,
        hd,
        d.sm_scale(),
        g.mask,
    );
    let o = ops::layout::unpack_rows(&o, &g.perm);
    let o = ops::linear::matmul(&o, &b.attn.out);
    let o = ops::elemwise::rmsnorm(&o, &b.attn_norm2, NORM_EPS);
    let x = match mods {
        Some((m, lanes)) => ops::elemwise::gated_residual_add(x, &m.gate_msa, &o, Some(lanes)),
        None => ops::elemwise::residual_add(&o, x),
    };

    let h = ops::elemwise::rmsnorm(&x, &b.ffn_norm1, NORM_EPS);
    let h = match mods {
        Some((m, lanes)) => scaled(&h, &m.scale_mlp, lanes),
        None => h,
    };
    let f = ops::linear::matmul(
        &ops::linear::mlp_swiglu(&ops::linear::matmul(&h, &b.mlp.gate_up), d.inter),
        &b.mlp.down,
    );
    let f = ops::elemwise::rmsnorm(&f, &b.ffn_norm2, NORM_EPS);
    match mods {
        Some((m, lanes)) => ops::elemwise::gated_residual_add(&x, &m.gate_mlp, &f, Some(lanes)),
        None => ops::elemwise::residual_add(&f, &x),
    }
}

// `TE_HIDDEN` is the caption width the flagship's `refine` port states;
// named here so the readings table and the model agree by construction.
const _: () = assert!(Dims::turbo().cap_width == TE_HIDDEN);
