//! `mini-dit`'s traced arithmetic: one denoise reading over three lanes.
//!
//! The fire a guest submits for one denoise step carries three lanes of one
//! request, and therefore of one attention group (D2):
//!
//! | lane | stream | rows | what feeds it |
//! |---|---|---|---|
//! | caption | [`Stream::Text`] | 8 | `Context` port 0, `[rows, 256]` |
//! | image | [`Stream::Image`] | 64 | `Latents` port 0, `[rows, 64]` |
//! | context | [`Stream::Context`] | 16 | `Context` port 1, `[rows, 512]` |
//!
//! plus one `LaneVector` port (the timestep) and one `AxisPositions` port
//! (the three rotary coordinates per row). The readout is
//! [`seam::VELOCITY`], `[image rows, 64]` — this text has no logits and
//! declares no kv space at all, so `trace_hybrid` plants no `out`.
//!
//! The three blocks are written out rather than walked, because they are
//! three different shapes and a `walk_layers` over them would be a lie about
//! what a layer of this model is.

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    AxisRole, Generative, LatentSpace, PortFact, PortKind, PositionConvention, ReadingFact,
    ReadoutKind, ScheduleFact, ScheduleKind,
};

use super::model::{
    CrossAttn, HEAD_DIM, INTER, LN_EPS, Linear, MOD_SLICES, Model, RMS_EPS, ROPE_AXES, ROPE_DIMS,
    ROPE_THETA, SM_SCALE, SelfAttn, Side, Swiglu, TIMESTEP_DIM, TIMESTEP_FLIP_SIN_COS,
    TIMESTEP_MAX_PERIOD, TIMESTEP_SCALE, port,
};

/// The bit the one-hot stream facts start at (D2). Bits 0..6 are the six
/// streams; this text names three of them.
pub const STREAM_BASE: u8 = 0;

/// The bit that says a lane runs the `denoise` reading. There is one reading
/// today, so no node is guarded on it; the bit exists because the fact word
/// a runtime stamps must say which arm a lane is in, and a second reading
/// (`vae.decode`, a refiner) would be a second value here.
pub const DENOISE_BIT: u8 = 6;

/// Which reading index the runtime stamps a denoise lane with — the position
/// of `"denoise"` in [`readings`].
pub const DENOISE_READING: u8 = 0;

/// **THE PARITY HARNESS'S BISECTION KNOB.** A synthetic fixture exists to be
/// compared with its reference, and a mismatch at the head says nothing
/// about WHERE the two diverged. `PIE_MINI_DIT_TAP=<dump key>` — one of the
/// intermediates `scripts/imagegen/mini_dit_ref.py` dumps, spelled the way
/// the `.npz` spells it — makes [`Model::forward`] plant [`seam::VELOCITY`]
/// on that rectangle and return there, so the guest's `velocity(width)`
/// reads the intermediate off its own lane's rows and
/// `scripts/imagegen/mini_dit_parity.py collect --tap` names it by the key.
///
/// This is the one place a family reads the environment, and it is read
/// once, at catalog time, for a row nothing real is served by. A real
/// family states its readout in its text and nowhere else.
pub struct Tap;

impl Tap {
    /// The environment variable, read at catalog time.
    pub const ENV: &'static str = "PIE_MINI_DIT_TAP";

    /// The requested tap, or `None` for the model as it is.
    #[must_use]
    pub fn from_env() -> Option<String> {
        std::env::var(Self::ENV).ok().filter(|key| !key.is_empty())
    }

    /// The width of the rectangle a tap exports: the patch features at the
    /// head's own two rectangles, the trunk width everywhere else.
    #[must_use]
    pub fn width(key: Option<&str>) -> u32 {
        match key {
            None | Some("final.tokens") => super::model::PATCH_FEATURES,
            Some(_) => super::model::HIDDEN,
        }
    }
}

/// This family's generative facts (design D12): what a guest sizes a job
/// from and what `forward-pass.reading` / `input` resolve against.
///
/// The row is a denoiser and nothing else, so there is one reading. Its
/// latent space is the identity — `mini-dit` has no VAE, and its "pixels"
/// are its latent cells — and its schedule is the rectified flow the
/// reference's four-step Euler run uses.
#[must_use]
pub fn generative(tap: Option<&str>) -> Generative {
    Generative {
        readings: vec![denoise_reading(tap)],
        latent: Some(LatentSpace {
            channels: super::model::CHANNELS,
            patch_t: 1,
            patch_h: super::model::PATCH,
            patch_w: super::model::PATCH,
            // No VAE: one latent cell IS one cell of the reference's
            // `[C, H, W]` array, so both compressions are the identity.
            spatial_compression: 1,
            temporal_compression: 1,
        }),
        schedule: Some(ScheduleFact {
            kind: ScheduleKind::Flow,
            shift: 1.0,
            train_steps: 1000,
            boundary: None,
            // `mini_dit_ref.py`'s `euler_sigmas`, without its trailing zero
            // (the schedule appends that itself).
            pinned_sigmas: vec![1.0, 0.75, 0.5, 0.25],
            // One backbone, one schedule: every lane takes `shift`.
            stream_shifts: vec![],
        }),
        // A synthetic row: the reference's grid is 8 x 8 patch rows, and a
        // pass big enough for a 128 x 128 latent at patch 2 covers anything
        // this family will be asked for.
        max_rows: 4096,
    }
}

/// The one reading. Port ORDER is load-bearing: a port's index is its
/// position among the ports of its own kind, so `text` before `context` is
/// what makes them `Input::context(0, ..)` and `Input::context(1, ..)` —
/// the numbering `model::port` states and the trace reads.
///
/// `streams` lists every lane the request submits. It is the whole set, not
/// one port's: a row port fills one stream's rows, but the `timestep` lane
/// vector is read once per lane and every class that modulates needs it,
/// while the context lane's cell is written and never read (its class runs
/// one projection and no modulation).
fn denoise_reading(tap: Option<&str>) -> ReadingFact {
    let port = |name, kind, width, streams: &[Stream]| PortFact {
        name,
        kind,
        width,
        streams: streams.to_vec(),
                at: None,
        };
    ReadingFact {
        name: "denoise",
        index: DENOISE_READING,
        // A denoise pass holds nothing between fires and embeds no tokens:
        // both `attention` and `embed` are refused by name on it, and the
        // image lane's row count comes from the latents channel.
        has_kv: false,
        takes_tokens: false,
        streams: vec![Stream::Text, Stream::Image, Stream::Context],
        ports: vec![
            port(
                "latents",
                PortKind::Latents,
                super::model::PATCH_FEATURES,
                &[Stream::Image],
            ),
            port(
                "text",
                PortKind::Context,
                super::model::TEXT_WIDTH,
                &[Stream::Text],
            ),
            port(
                "context",
                PortKind::Context,
                super::model::CONTEXT_WIDTH,
                &[Stream::Context],
            ),
            // Read once per lane by every class that modulates; the context
            // lane's class runs one projection and no modulation.
            port(
                "timestep",
                PortKind::LaneVector,
                1,
                &[Stream::Text, Stream::Image],
            ),
            port(
                "positions",
                PortKind::AxisPositions,
                u32::from(ROPE_AXES),
                &[Stream::Text, Stream::Image],
            ),
        ],
        // `(t, h, w)`: caption row `j` at `(j, 0, 0)`, image patch
        // `(h, w)` at `(0, h, w)` — `mini_dit_ref.py`'s `text_positions` /
        // `image_positions`.
        positions: Some(PositionConvention {
            axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
            text_axis: 0,
            text_origin: 0,
            image_follows_text: false,
        }),
        readout: ReadoutKind::Velocity,
        readout_width: Tap::width(tap),
    }
}

/// The per-lane facts: which stream the lane's rows are, and which reading
/// its pass runs.
pub struct Facts {
    pub stream: Stream,
    pub denoise: bool,
}

impl Facts {
    /// The caption lane.
    #[must_use]
    pub fn text() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Text)
    }

    /// The image lane — the only one the head reads out.
    #[must_use]
    pub fn image() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Image)
    }

    /// The cross-attention context lane: keys and values only, never
    /// queries, and no rope.
    #[must_use]
    pub fn context() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Context)
    }

    /// The reading. Unguarded today (one reading), stated so a runtime and a
    /// second reading find the bit already named.
    #[must_use]
    pub fn denoise() -> Predicate {
        Predicate::fact(DENOISE_BIT)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            stream: r.stream(),
            denoise: r.reading() == DENOISE_READING,
        }
    }

    fn word(&self) -> u64 {
        self.stream.word(STREAM_BASE) | (u64::from(self.denoise) << DENOISE_BIT)
    }
}

/// `tap!(m, "key", &value)`: under [`Tap`] `key`, plant the velocity seam on
/// `value` and return it as the plan's readout. A no-op otherwise, so the
/// model's text reads through it unchanged.
macro_rules! tap {
    ($m:expr, $key:literal, $v:expr) => {
        if let Some(v) = tapped($m, "", $key, $v) {
            return v;
        }
    };
}

/// A helper's tap hit, handed up: `Err(v)` is the exported rectangle, and
/// `forward` returns it.
macro_rules! unwrap_tap {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(tapped) => return tapped,
        }
    };
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    /// A denoise reading holds nothing between fires: no kv space, no state
    /// slab. Every table the joint attention needs (the group CSRs, the row
    /// permutations, the token→lane map) is readable with no kv space
    /// declared.
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        // The context lane on one side; the two lanes that share a sequence
        // on the other. Splitting the joint pair OUT OF the joint arm is
        // what lets `Value::merge` recover `!context` as the merge's guard,
        // so a merged rectangle and the joint arm's own CSR read as the same
        // arm (`Guard::common`, `record.rs`).
        let (ctx, joint) = inputs.split(&Facts::context());
        let (txt_in, img_in) = joint.split(&Facts::text());

        // Whole-fire reads: unguarded, so they meet every arm.
        let lanes = inputs.request_of_token();
        let positions = inputs.axis_positions(port::POSITIONS, ROPE_AXES);

        // sinusoid → SiLU, once per fire. The chain lands f32 (a lane vector
        // is f32 and `linear.matmul` answers in its activation's dtype), which
        // is the dtype `elementwise.modulate` takes its vector in.
        let t = inputs.lane_vector(port::TIMESTEP, 1);
        let temb = ops::elemwise::sinusoid(
            &t,
            TIMESTEP_DIM,
            TIMESTEP_MAX_PERIOD,
            TIMESTEP_FLIP_SIN_COS,
            TIMESTEP_SCALE,
        );
        let temb = ops::elemwise::silu(&temb);

        // The packing of the joint sequence: `[caption ‖ image]` per group,
        // which is the reference's own order (packed order is (group, stream
        // code, lane), and Text's code is below Image's).
        let joint_perm = joint.row_permutation();
        let joint_csr = joint.group_indptr();
        // The image rectangle alone, for block 2's self-attention and for the
        // cross-attention's queries.
        let img_perm = img_in.row_permutation();
        let img_csr = img_in.group_indptr();

        // ---- the lanes' rows ------------------------------------------
        let txt = txt_in.context(port::TEXT, super::model::TEXT_WIDTH);
        let patches = img_in.latents(port::LATENTS, super::model::PATCH_FEATURES, Dtype::Bf16);
        // Block 2's context rows, read here with the other ports so every
        // port the reading declares is in the plan whatever a tap cuts.
        let ctx_rows = ctx.context(port::CONTEXT, super::model::CONTEXT_WIDTH);
        // A port read is a binding, not a rectangle, so the probe is a norm
        // of it (zero in, zero out — it still says whether the rows landed).
        tap!(
            m,
            "in.text",
            &ops::elemwise::layernorm_no_scale(&txt, LN_EPS)
        );
        let img = linear(&m.x_embed, &patches);
        tap!(m, "x_embed", &img);

        // ================================================== block 0: single
        // One sequence, one set of weights, one modulation.
        let x = Value::merge(vec![txt, img]);
        tap!(m, "b0.in", &x);
        let (msa, gate_a, mmlp, gate_m) = adaln6(&linear(&m.single.ada, &temb));
        let x = unwrap_tap!(attn_sublayer(
            m,
            "b0",
            &x,
            &m.single.attn,
            &msa,
            &gate_a,
            &lanes,
            &positions,
            &joint_perm,
            &joint_csr,
        ));
        tap!(m, "b0.x_after_attn", &x);
        let x = unwrap_tap!(mlp_sublayer(
            m,
            "b0",
            &x,
            &m.single.mlp,
            &mmlp,
            &gate_m,
            &lanes
        ));
        tap!(m, "b0.out", &x);

        // ================================================== block 1: double
        // The same joint attention, but each stream brings its own
        // modulation, projections and MLP; only the softmax is shared.
        let (txt, img) = x.split(&Facts::text());
        let (txt_mod, img_mod) = (
            adaln6(&linear(&m.double.txt.ada, &temb)),
            adaln6(&linear(&m.double.img.ada, &temb)),
        );
        let (tq, tk, tv) = unwrap_tap!(qkv(
            m,
            "b1.txt_",
            &txt,
            &m.double.txt,
            &txt_mod.0,
            &lanes,
            &positions
        ));
        let (iq, ik, iv) = unwrap_tap!(qkv(
            m,
            "b1.img_",
            &img,
            &m.double.img,
            &img_mod.0,
            &lanes,
            &positions
        ));
        let o = joint_attention(
            &Value::merge(vec![tq, iq]),
            &Value::merge(vec![tk, ik]),
            &Value::merge(vec![tv, iv]),
            &joint_perm,
            &joint_csr,
        );
        tap!(m, "b1.joint_attn_heads", &o);
        let (o_txt, o_img) = o.split(&Facts::text());
        let ta = linear(&m.double.txt.attn.out, &o_txt);
        tap!(m, "b1.txt_attn_out", &ta);
        let ia = linear(&m.double.img.attn.out, &o_img);
        tap!(m, "b1.img_attn_out", &ia);
        let txt = ops::elemwise::gated_residual_add(&txt, &txt_mod.1, &ta, Some(&lanes));
        tap!(m, "b1.txt_after_attn", &txt);
        let img = ops::elemwise::gated_residual_add(&img, &img_mod.1, &ia, Some(&lanes));
        tap!(m, "b1.img_after_attn", &img);
        // The caption stream ends here: block 2 and the head are image-only.
        // Its MLP is still emitted — the reference computes it, the checkpoint
        // ships its weights, and the class sweep simply never roots it.
        let txt = unwrap_tap!(mlp_sublayer(
            m,
            "b1.txt_",
            &txt,
            &m.double.txt.mlp,
            &txt_mod.2,
            &txt_mod.3,
            &lanes
        ));
        tap!(m, "b1.out_txt", &txt);
        let x = unwrap_tap!(mlp_sublayer(
            m,
            "b1.img_",
            &img,
            &m.double.img.mlp,
            &img_mod.2,
            &img_mod.3,
            &lanes
        ));
        tap!(m, "b1.out_img", &x);

        // =================================================== block 2: cross
        // Wan's modulation: a learned table plus the timestep projection.
        let mod2 = ops::elemwise::add_bias(&m.cross.mod_table, &linear(&m.cross.ada, &temb));
        let (msa, gate_a, mffn, gate_f) = adaln6(&mod2);
        let x = unwrap_tap!(attn_sublayer(
            m,
            "b2",
            &x,
            &m.cross.self_attn,
            &msa,
            &gate_a,
            &lanes,
            &positions,
            &img_perm,
            &img_csr,
        ));
        tap!(m, "b2.x_after_self", &x);

        // Cross-attention: LayerNorm WITH affine, no modulation, no rope, and
        // an UNGATED residual — the Wan contract. The queries are the image
        // arm's; the keys and values are the context lane's, and
        // `attention.ragged` is the one op whose operands may come from two
        // arms.
        let hc = ops::elemwise::layernorm(&x, &m.cross.norm, &m.cross.norm_bias, LN_EPS);
        tap!(m, "b2.cross_norm_out", &hc);
        let cq = ops::elemwise::rmsnorm_per_head(
            &linear(&m.cross.cross.q, &hc),
            &m.cross.cross.q_norm,
            HEAD_DIM,
            RMS_EPS,
        );
        tap!(m, "b2.cross_q", &cq);
        let (ck, cv) = context_kv(&ctx_rows, &m.cross.cross);
        let ctx_perm = ctx.row_permutation();
        let ctx_csr = ctx.group_indptr();
        let ca = ops::attn::ragged(
            &ops::layout::pack_rows(&cq, &img_perm),
            &ops::layout::pack_rows(&ck, &ctx_perm),
            &ops::layout::pack_rows(&cv, &ctx_perm),
            &img_csr,
            &ctx_csr,
            HEAD_DIM,
            SM_SCALE,
            RaggedMask::GroupBlockDiagonal,
        );
        let ca = ops::layout::unpack_rows(&ca, &img_perm);
        tap!(m, "b2.cross_attn_heads", &ca);
        let ca = linear(&m.cross.cross.out, &ca);
        tap!(m, "b2.cross_attn_out", &ca);
        let x = ops::elemwise::residual_add(&ca, &x);
        tap!(m, "b2.x_after_cross", &x);
        let x = unwrap_tap!(mlp_sublayer(
            m,
            "b2",
            &x,
            &m.cross.mlp,
            &mffn,
            &gate_f,
            &lanes
        ));
        tap!(m, "b2.out", &x);

        // ========================================================== head
        // LayerNorm-no-affine → scale/shift → Linear back to C·p·p. The
        // guest unpatchifies; the plan hands back patch rows.
        let scale_shift = linear(&m.final_ada, &temb);
        let hf = ops::elemwise::modulate(
            &ops::elemwise::layernorm_no_scale(&x, LN_EPS),
            &scale_shift,
            Some(&lanes),
            ModulateForm::ScaleShift,
        );
        tap!(m, "final.norm_out", &hf);
        let velocity = linear(&m.final_proj, &hf);
        seam::at(seam::VELOCITY, &[&velocity]);
        velocity
    }
}

/// One biased projection.
fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

/// An adaLN-Zero vector cut into the four things a block applies: the
/// attention `[scale | shift]` pair, the attention gate, the MLP
/// `[scale | shift]` pair, the MLP gate.
///
/// The plan's own slice order is `[s_a | b_a | g_a | s_m | b_m | g_m]`, so
/// `elementwise.modulate`'s `[s | b]` pair is a prefix and no reordering op
/// is needed. `import.rs` is where the checkpoint's `[shift, scale, gate]`
/// order becomes this one.
fn adaln6(m: &Value) -> (Value, Value, Value, Value) {
    let width = super::model::HIDDEN;
    debug_assert_eq!(m.width(), u64::from(MOD_SLICES * width));
    let (msa, rest) = ops::layout::split_rows(m, 2 * width);
    let (gate_a, rest) = ops::layout::split_rows(&rest, width);
    let (mmlp, gate_m) = ops::layout::split_rows(&rest, 2 * width);
    (msa, gate_a, mmlp, gate_m)
}

/// Modulate, project, QK-norm, rope: the front half of a self-attention
/// sublayer, up to the three head rectangles.
///
/// `stem` is the golden's key prefix for this sublayer (`"b0"`,
/// `"b1.img_"`, …), for [`Tap`]: a helper that meets its tap plants the seam
/// and answers `Err(rectangle)`, which `forward` returns.
#[allow(clippy::too_many_arguments)]
fn qkv(
    model: &Model,
    stem: &str,
    x: &Value,
    side: &Side,
    m: &Value,
    lanes: &Value,
    positions: &Value,
) -> Result<(Value, Value, Value), Value> {
    heads(model, stem, x, &side.attn, m, lanes, positions)
}

/// The rectangle a tap point exports, if this is its key: the seam is
/// planted and the value handed back for `forward` to return. `None` is the
/// untapped path. `stem` prefixes `tail` with a dot (`b0.norm1_out`), or
/// bare when it already ends in the golden's underscore (`b1.img_norm1_out`);
/// an empty stem is the whole key.
fn tapped(model: &Model, stem: &str, tail: &str, v: &Value) -> Option<Value> {
    let key = if stem.is_empty() || stem.ends_with('_') {
        format!("{stem}{tail}")
    } else {
        format!("{stem}.{tail}")
    };
    if model.tap.as_deref() == Some(key.as_str()) {
        seam::at(seam::VELOCITY, &[v]);
        Some(v.clone())
    } else {
        None
    }
}

/// `tapped`, as the `?`-able form a helper propagates.
fn tap_at(model: &Model, stem: &str, tail: &str, v: &Value) -> Result<(), Value> {
    match tapped(model, stem, tail, v) {
        Some(v) => Err(v),
        None => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
fn heads(
    model: &Model,
    stem: &str,
    x: &Value,
    attn: &SelfAttn,
    m: &Value,
    lanes: &Value,
    positions: &Value,
) -> Result<(Value, Value, Value), Value> {
    let h = ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(x, LN_EPS),
        m,
        Some(lanes),
        ModulateForm::ScaleShift,
    );
    tap_at(model, stem, "norm1_out", &h)?;
    let (q, k, v) = ops::layout::split_qkv(
        &linear(&attn.qkv, &h),
        super::model::HIDDEN,
        super::model::HIDDEN,
    );
    // Block 2 spells its self-attention heads `self_q_rope`, `self_v`, ….
    let hp = if stem == "b2" { "self_" } else { "" };
    tap_at(model, stem, &format!("{hp}q_raw"), &q)?;
    tap_at(model, stem, &format!("{hp}k_raw"), &k)?;
    tap_at(model, stem, &format!("{hp}v"), &v)?;
    let turn = |x: &Value, gain: &Weight, tail: &str| -> Result<Value, Value> {
        let n = ops::elemwise::rmsnorm_per_head(x, gain, HEAD_DIM, RMS_EPS);
        tap_at(model, stem, &format!("{hp}{tail}_qknorm"), &n)?;
        let r = ops::elemwise::rope_axes(
            &n,
            positions,
            ROPE_DIMS,
            [ROPE_THETA; 4],
            RopeForm::Interleaved,
            HEAD_DIM,
            HEAD_DIM,
        );
        tap_at(model, stem, &format!("{hp}{tail}_rope"), &r)?;
        Ok(r)
    };
    Ok((
        turn(&q, &attn.q_norm, "q")?,
        turn(&k, &attn.k_norm, "k")?,
        v,
    ))
}

/// The joint attention itself: pack by the arm's row order, one ragged read
/// over the group CSR, unpack back onto the fire's rows.
fn joint_attention(q: &Value, k: &Value, v: &Value, perm: &Value, csr: &Value) -> Value {
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(q, perm),
        &ops::layout::pack_rows(k, perm),
        &ops::layout::pack_rows(v, perm),
        csr,
        csr,
        HEAD_DIM,
        SM_SCALE,
        RaggedMask::GroupBlockDiagonal,
    );
    ops::layout::unpack_rows(&o, perm)
}

/// A whole single-stream attention sublayer: `x += gate · out(attn(mod(x)))`.
#[allow(clippy::too_many_arguments)]
fn attn_sublayer(
    model: &Model,
    stem: &str,
    x: &Value,
    attn: &SelfAttn,
    m: &Value,
    gate: &Value,
    lanes: &Value,
    positions: &Value,
    perm: &Value,
    csr: &Value,
) -> Result<Value, Value> {
    let (q, k, v) = heads(model, stem, x, attn, m, lanes, positions)?;
    let o = joint_attention(&q, &k, &v, perm, csr);
    // Block 2 spells its self-attention keys apart from its cross-attention's.
    let (heads_key, out_key) = if stem == "b2" {
        ("self_attn_heads", "self_attn_out")
    } else {
        ("attn_heads", "attn_out")
    };
    tap_at(model, stem, heads_key, &o)?;
    let o = linear(&attn.out, &o);
    tap_at(model, stem, out_key, &o)?;
    Ok(ops::elemwise::gated_residual_add(x, gate, &o, Some(lanes)))
}

/// `x += gate · down(swiglu(gate_up(mod(x))))`.
#[allow(clippy::too_many_arguments)]
fn mlp_sublayer(
    model: &Model,
    stem: &str,
    x: &Value,
    mlp: &Swiglu,
    m: &Value,
    gate: &Value,
    lanes: &Value,
) -> Result<Value, Value> {
    let h = ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(x, LN_EPS),
        m,
        Some(lanes),
        ModulateForm::ScaleShift,
    );
    // Block 2's MLP norm is its third; every other block's is its second.
    let norm_key = if stem == "b2" {
        "norm3_out"
    } else {
        "norm2_out"
    };
    tap_at(model, stem, norm_key, &h)?;
    let h = ops::linear::mlp_swiglu(&linear(&mlp.gate_up, &h), INTER);
    let y = linear(&mlp.down, &h);
    tap_at(model, stem, "mlp_out", &y)?;
    Ok(ops::elemwise::gated_residual_add(x, gate, &y, Some(lanes)))
}

/// The context lane's keys and values: one packed projection off the wider
/// rectangle, split, and the keys QK-normed. No rope — the Wan contract.
fn context_kv(c: &Value, cross: &CrossAttn) -> (Value, Value) {
    let (k, v) = ops::layout::split_rows(&linear(&cross.kv, c), super::model::HIDDEN);
    (
        ops::elemwise::rmsnorm_per_head(&k, &cross.k_norm, HEAD_DIM, RMS_EPS),
        v,
    )
}
