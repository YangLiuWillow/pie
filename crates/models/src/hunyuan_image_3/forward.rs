//! HunyuanImage 3's traced arithmetic: four readings of one plan, selected
//! per lane by the reading bits of the fact word (design D1, D5, D10).
//!
//! | reading | lanes (stream) | binds | mask | reads back |
//! |---|---|---|---|---|
//! | `encode` | one, `Text` | `embed(ids)`, `attention(kv)`, `positions` `[L, 2]` | causal (prefill / decode) | `out` `[L, vocab]` |
//! | `denoise` | one, `Image` | `embed(ids)`, `attention(kv)`, `latents` `[N+1, hidden]`, `special` `[N+1, 1]`, `timestep`, `positions` `[N+1, 2]` | the guest's `[rows, kv]` slab: causal prefix ∪ the canvas block | `hidden` `[N+1, hidden]` |
//! | `image.in` | one, `Image`, one clip `{1, h, w}` | `latent` `[h·w, 32 + 256]` on the voxel axis — the noisy latent BESIDE the timestep's sinusoid | — | `pixels` `[h·w, hidden]` |
//! | `image.out` | one, `Image`, one clip `{1, h, w}` | `rows` `[h·w, hidden + 256]`, likewise | — | `pixels` `[h·w, 32]` — the velocity |
//!
//! # Why a denoise step is three fires
//!
//! The reference's step is `patch_embed → trunk → final_layer`, and the
//! two ends are 3×3 convolutions. Convolutions live on the VOXEL axis
//! (design D8) and the trunk lives on the TOKEN axis, and
//! `model_compiler::unit::partition` refuses a plan whose axes alternate
//! (`Error::UnitsInterleave`): `[Voxels, Tokens, Voxels]` is not a plan.
//! So the image head is two readings of its own, each a single-unit voxel
//! plan, and the guest carries `[h·w, hidden]` between them on device
//! channels — exactly the `image_in` / `trunk_step` / `image_out` split the
//! model study proposes (§J). Per step: `image.in`, `denoise`, `image.out`,
//! three fires and ≈2.5 ms of seam against tens of ms of trunk.
//!
//! # Why the `<timestep>` row rides in the canvas lane
//!
//! The reference recomputes, every step, the `<timestep>` token row beside
//! the `h·w` image rows, and the image rows attend its fresh K/V. Putting
//! it on a lane of its own would make that K/V a cross-class read inside
//! one fire, whose order nothing states; putting it in the canvas lane
//! makes it one `attention.kv_append` before one `attention.masked`, which
//! is ordered by construction. The lane is therefore `h·w + 1` rows, the
//! special row LAST (a lane's row order is not its sequence order — the
//! guest's `positions`, `w-slot` and `w-off` channels place every row), so
//! the guest writes the `image.in` readout into the leading rows of the
//! `latents` cell and leaves the tail alone.
//!
//! Which row is special is a `[rows, 1]` flag the guest fills
//! (`port::SPECIAL`): this IR has no row-level class, and a flag column
//! projected through a bank is z_image's own `pad_rows`. The row it flags
//! is overwritten with `timestep_emb(t)`, broadcast from the lane vector
//! through [`super::model::Model::timestep_emb`]'s doubled second linear.
//!
//! # Positions
//!
//! `AxisPositions`, `[rows, 2]` f32, `(y, x')`. A text token at sequence
//! index `p` is `(p, p·s)`; image token `(r, c)` of an `h × w` span
//! starting at `L` is `(L + (wh − h)/2 + r, (L + (wh − w)/2 + c)·s)` with
//! the halves truncated as the reference's `.long()` does, `s` being
//! [`super::model::rope_x_scale`]. The scale is why this family states no
//! [`crate::PositionConvention`].

use model_dsl::ops::spatial;
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, Predicate, Request,
    RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    Generative, LatentSpace, PortFact, PortKind, ReadingFact, ReadoutKind, ScheduleFact,
    ScheduleKind,
};

use super::model::{
    Conv, Dims, Embedder, GN_EPS, GN_GROUPS, LATENT_CHANNELS, Linear, Model, NORM_EPS, PATCH,
    ROPE_AXES, ROPE_THETA, ResBlock, SPATIAL_COMPRESSION, T_FLIP_SIN_COS, T_FREQ_DIM, T_MAX_PERIOD,
    T_SCALE, TRAIN_STEPS, port,
};

/// The bit the one-hot stream facts start at (D2): bits 0..6 are the six
/// streams, of which this text names Text and Image.
pub const STREAM_BASE: u8 = 0;

/// The two bits the reading index lives in, as a plain binary code.
pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;
/// The AR decode bit: one query row against the whole prefix. Read on the
/// `encode` reading alone (a canvas fire is never one row).
pub const QO_ONE: u8 = 8;

/// The reading codes, dense from 0 in `Generative::readings` order.
pub const ENCODE: u8 = 0;
pub const DENOISE: u8 = 1;
pub const IMAGE_IN: u8 = 2;
pub const IMAGE_OUT: u8 = 3;

impl Model {
    /// This row's generative facts (design D12).
    #[must_use]
    pub fn generative(&self) -> Generative {
        let d = &self.dims;
        let port = |name, kind, width, at| PortFact {
            name,
            kind,
            width,
            streams: vec![],
            at,
            rows: None,
        };
        let readings = vec![
            ReadingFact {
                name: "encode",
                index: ENCODE,
                has_kv: true,
                takes_tokens: true,
                streams: vec![Stream::Text],
                ports: vec![port(
                    "positions",
                    PortKind::AxisPositions,
                    u32::from(ROPE_AXES),
                    None,
                )],
                positions: None,
                readout: ReadoutKind::Logits,
                readout_width: d.vocab,
            },
            ReadingFact {
                name: "denoise",
                index: DENOISE,
                has_kv: true,
                takes_tokens: true,
                streams: vec![Stream::Image],
                ports: vec![
                    port("latents", PortKind::Latents, d.hidden, None),
                    port("special", PortKind::Latents, 1, None),
                    port("timestep", PortKind::LaneVector, 1, None),
                    port(
                        "positions",
                        PortKind::AxisPositions,
                        u32::from(ROPE_AXES),
                        None,
                    ),
                ],
                positions: None,
                // The trunk's rows for the canvas, `ln_f` deliberately NOT
                // applied (the reference skips it for image rows).
                readout: ReadoutKind::Hidden,
                readout_width: d.hidden,
            },
            ReadingFact {
                name: "image.in",
                index: IMAGE_IN,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Image],
                ports: vec![port(
                    "latent",
                    PortKind::Voxels,
                    LATENT_CHANNELS + T_FREQ_DIM,
                    Some(port::LATENT_VOXELS),
                )],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: d.hidden,
            },
            ReadingFact {
                name: "image.out",
                index: IMAGE_OUT,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Image],
                ports: vec![port(
                    "rows",
                    PortKind::Voxels,
                    d.hidden + T_FREQ_DIM,
                    Some(port::ROW_VOXELS),
                )],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: LATENT_CHANNELS,
            },
        ];
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: LATENT_CHANNELS,
                patch_t: 1,
                patch_h: PATCH,
                patch_w: PATCH,
                spatial_compression: SPATIAL_COMPRESSION,
                temporal_compression: 1,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: super::model::FLOW_SHIFT,
                train_steps: TRAIN_STEPS,
                boundary: None,
                // `FlowMatchDiscreteScheduler` builds `linspace(1, 0, N+1)`
                // through the static shift; nothing is pinned, and one
                // stream carries the whole schedule.
                pinned_sigmas: vec![],
                stream_shifts: vec![],
            }),
            // Every entry of the resolution group is ≈4096 image tokens
            // plus the `<timestep>` row; the prompt prefix and any
            // reference images are prefix PAGES, not canvas rows.
            max_rows: if d.layers > 4 { 8192 } else { 1024 },
        }
    }
}

/// The per-lane facts: which stream the lane's rows are, which reading its
/// pass runs, and whether it is a one-row AR step.
pub struct Facts {
    pub stream: Stream,
    /// The reading index, `0..4` (a wider index is truncated to two bits).
    pub reading: u8,
    pub qo_one: bool,
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
    pub fn reading_lo() -> Predicate {
        Predicate::fact(READING_LO)
    }

    #[must_use]
    pub fn reading_hi() -> Predicate {
        Predicate::fact(READING_HI)
    }

    #[must_use]
    pub fn qo_one() -> Predicate {
        Predicate::fact(QO_ONE)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            stream: r.stream(),
            reading: r.reading() & 3,
            qo_one: r.query_len() == 1,
        }
    }

    fn word(&self) -> u64 {
        self.stream.word(STREAM_BASE)
            | (u64::from(self.reading & 3) << READING_LO)
            | (u64::from(self.qo_one) << QO_ONE)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    /// One kv space, one `[kv_heads·head_dim]` pair of planes a layer.
    /// The prefix pages and the canvas slots live in it together: what
    /// makes a step cheap is that the guest hands the same readable span
    /// and the same writable slots every fire (design D10).
    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv_dtype);
        let plane = u64::from(self.kv_width());
        for w in &self.layers {
            c.kv(kv, w.kv.clone(), [plane, plane]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (image_out_arm, image_in_arm) = hi.split(&Facts::reading_lo());
        let (denoise_arm, encode_arm) = lo.split(&Facts::reading_lo());

        // The two voxel units first: a plan's units are ordered by first
        // appearance, and the trunk's token unit must not be reopened.
        image_in(&image_in_arm, self);
        image_out(&image_out_arm, self);

        let _ = &encode_arm;
        trunk(&lo, &denoise_arm, self)
    }
}

/// One biased projection.
fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

/// A `TimestepEmbedder` over an already-built sinusoid rectangle:
/// `linear_2(GELU(linear_1(t_freq)))`. The reference's GELU is the erf
/// form; this is the tanh one (see the model's numerics contract).
fn embedder(e: &Embedder, t_freq: &Value) -> Value {
    linear(
        &e.mlp_out,
        &ops::elemwise::gelu(&linear(&e.mlp_in, t_freq), true),
    )
}

// ---------------------------------------------------------------- the trunk

/// The `encode` and `denoise` readings: one stack of 32 MoE decoder
/// layers, three attention arms (the canvas's masked bidirectional read,
/// the AR decode, the AR prefill), one head each.
fn trunk(all: &Input<Facts>, den: &Input<Facts>, m: &Model) -> Value {
    let d: &Dims = &m.dims;
    let hd = d.head_dim;
    let sm = d.sm_scale();
    // One class list, cut once off the trunk's own arm and reused for the
    // plans and for every layer's q: a plan and the query that consumes it
    // must say the same guard, word for word.
    let classes = [Facts::reading_lo(), Facts::qo_one(), Predicate::rest()];
    let [canvas_in, ar_decode, ar_prefill] = all.split(classes.clone());

    let plan_den = ops::attn::plan_prefill(&canvas_in, m.q_heads, m.kv_heads, hd, None);
    let plan_dec = ops::attn::plan_decode(&ar_decode, m.q_heads, m.kv_heads, hd, None);
    let plan_pre = ops::attn::plan_prefill(&ar_prefill, m.q_heads, m.kv_heads, hd, None);
    let mask = canvas_in.mask();

    let positions = all.axis_positions(port::POSITIONS, ROPE_AXES);
    let ids = all.tokens();
    let y = ops::layout::embed(&ids, &m.embed, d.vocab);
    // The canvas lane's rows are not its ids: they are the image head's
    // output beside one `timestep_emb(t)` row.
    let (_, encoded) = y.split(&Facts::reading_lo());
    let mut y = Value::merge(vec![canvas_rows(den, m), encoded]);

    for (l, w) in all.walk_layers(&m.layers) {
        let n = ops::elemwise::rmsnorm(&y, &w.attn_norm, NORM_EPS);
        let (q, k, v) =
            ops::layout::split_qkv(&ops::linear::matmul(&n, &w.qkv), m.q_width(), m.kv_width());
        // **ROPE FIRST, THEN THE QK NORMS** (study §C.1).
        let turn = |x: &Value| {
            ops::elemwise::rope_axes(
                x,
                &positions,
                d.rope_dims(),
                [ROPE_THETA; 4],
                RopeForm::Split,
                hd,
                hd,
            )
        };
        let q = ops::elemwise::rmsnorm_per_head(&turn(&q), &w.q_norm, hd, NORM_EPS);
        let k = ops::elemwise::rmsnorm_per_head(&turn(&k), &w.k_norm, hd, NORM_EPS);

        let pages = all.kv(&w.kv);
        ops::attn::kv_append(
            &k,
            &v,
            pages,
            &all.write_page(&w.kv),
            &all.write_offset(&w.kv),
        );

        let [dq, aq, pq] = q.split(classes.clone());
        let a = Value::merge(vec![
            // The generalized causal mask: the guest's slab says which of
            // `[prefix | canvas]` each row sees, and the causal bound is
            // lifted (the canvas block is bidirectional).
            ops::attn::masked(
                &dq, &plan_den, &mask, pages, None, hd, m.kv_heads, false, sm,
            ),
            ops::attn::decode(&aq, &plan_dec, pages, None, hd, sm),
            ops::attn::prefill(&pq, &plan_pre, pages, None, hd, m.kv_heads, sm),
        ]);
        let o = ops::linear::matmul(&a, &w.o_proj);
        let o = if m.tp > 1 {
            ops::collective::all_reduce(&o)
        } else {
            o
        };
        // **THE FIRST FOLD IS FRESH, EVERY OTHER ONE IS IN PLACE.**
        // `residual_add` writes through its residual, and the class walk
        // follows an in-place write backwards through the classes it does
        // not run (`check::classes::passes_through`). Chained from the
        // last layer that walk reaches the trunk's HEAD — the merge that
        // picks the canvas's rows over the text's — inside the two VOXEL
        // classes, where neither arm holds and the merge reads
        // `Uncovered`. One fresh sum ends the chain at a value the walk
        // cannot pass through, for one extra `[rows, hidden]` rectangle.
        y = if l == 0 {
            ops::elemwise::add(&o, &y)
        } else {
            ops::elemwise::residual_add(&o, &y)
        };

        let n = ops::elemwise::rmsnorm(&y, &w.mlp_norm, NORM_EPS);
        let f = moe(&n, w, m);
        let f = if m.tp > 1 {
            ops::collective::all_reduce(&f)
        } else {
            f
        };
        y = ops::elemwise::residual_add(&f, &y);
    }

    // `model.ln_f` is applied to the TEXT rows only; the canvas's rows go
    // to `final_layer` raw (study §C.2).
    //
    // Both readouts are planted HERE, and the value handed back is the
    // canvas's: `trace_hybrid` plants `out` on a returned value that is
    // not already under a float readout, and an `out` planted on a value
    // whose guard is the whole plan would demand this chain in the two
    // VOXEL classes as well — where the trunk writes nothing.
    let (canvas, text) = y.split(&Facts::reading_lo());
    seam::at(seam::HIDDEN, &[&canvas]);
    let x = ops::elemwise::rmsnorm(&text, &m.final_norm, NORM_EPS);
    let logits = ops::linear::lm_head(&x, &m.head);
    seam::at(seam::OUT, &[&logits]);
    canvas
}

/// 64 routed experts top-8 with a renormalised softmax, plus the always-on
/// shared expert. `gate_and_up_proj`'s halves are swapped at import so
/// `linear.mlp_swiglu`'s `silu(first) · second` is the reference's
/// `x1 · silu(x2)`.
fn moe(x: &Value, w: &super::model::Layer, m: &Model) -> Value {
    let d = &m.dims;
    let (routes, weights) =
        ops::linear::moe_topk_softmax(&ops::linear::matmul(x, &w.router), d.experts, d.top_k);
    let select = |act: &Value, bank: &Weight| {
        if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
            ops::linear::moe_matmul_select(act, bank, &routes, d.top_k)
        } else {
            ops::linear::moe_matmul_select_quant(act, bank, &routes, d.top_k)
        }
    };
    let act = ops::linear::mlp_swiglu(&select(x, &w.experts_gate_up), m.moe_inter);
    let routed = ops::linear::moe_weighted_sum(&select(&act, &w.experts_down), &weights);
    let shared = ops::linear::matmul(
        &ops::linear::mlp_swiglu(&ops::linear::matmul(x, &w.shared_gate_up), m.shared_inter),
        &w.shared_down,
    );
    ops::elemwise::residual_add(&shared, &routed)
}

/// The canvas lane's input rows: the `image.in` readout everywhere, the
/// `<timestep>` token's embedding on the one row the `special` flag names.
///
/// `f` is the flag (`0` on an image row, `1` on the special one), `ones` a
/// constant `[hidden, 1]` bank that broadcasts it to a row. Then
/// `x = u·(1 − f) + t_emb·f`, in three elementwise ops and no row-level
/// class.
fn canvas_rows(arm: &Input<Facts>, m: &Model) -> Value {
    let d = &m.dims;
    let u = arm.latents(port::ROWS, d.hidden, Dtype::Bf16);
    let flag = arm.latents(port::SPECIAL, 1, Dtype::Bf16);
    let lanes = arm.request_of_token();

    // `timestep_emb(σ·1000)` once per lane, in fp32 (`Sinusoid` and the
    // lane GEMMs below are the f32 lane-vector chain of §3), doubled into
    // `[t_emb | t_emb]` so one `ScaleShift` over a zero row lands it:
    // `0·(1 + t_emb) + t_emb`.
    let t = arm.lane_vector(port::TIMESTEP, 1);
    let freqs = ops::elemwise::sinusoid(&t, T_FREQ_DIM, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE);
    let doubled = embedder(&m.timestep_emb, &freqs);

    // `[+f | -f]`, both signs of the flag broadcast to a row in one
    // projection; their sum is a FRESH rectangle of zeros, which is what
    // the lane's `[t_emb | t_emb]` is landed over (a `mul_scalar(0.0)`
    // would write in place through the port's own rectangle, which is not
    // the arena's to write).
    let spread = ops::linear::matmul(&flag, &m.ones);
    let (pos, neg) = ops::layout::split_rows(&spread, d.hidden);
    let zero = ops::elemwise::add(&pos, &neg);
    let special = ops::elemwise::modulate(&zero, &doubled, Some(&lanes), ModulateForm::ScaleShift);

    // `u·(1 − f)` on every row, then `t_emb` on the flagged one.
    let kept = ops::elemwise::modulate(&u, &neg, None, ModulateForm::Scale);
    ops::elemwise::add(&kept, &ops::elemwise::mul(&special, &pos))
}

// ----------------------------------------------------------- the image head

/// One convolution of the image head: `k` as declared, stride 1, `same`
/// spatial padding, no time padding (a still is one frame).
fn conv(x: &Value, g: &Value, c: &Conv) -> (Value, Value) {
    let shape = spatial::Conv::conv3d(c.k, [1, 1, 1], [0, c.k[1] / 2, c.k[2] / 2]);
    spatial::conv3d(x, g, &c.w, Some(&c.bias), shape, None)
}

/// One `ResBlock` at `patch_size = 1`: `GN → SiLU → conv`, the adaptive
/// group norm `GN(h)·(1 + scale) + shift` off `emb_layers(SiLU(t_emb))`,
/// `SiLU → conv`, and the 1×1 skip.
fn resblock(x: &Value, g: &Value, r: &ResBlock, temb: &Value) -> Value {
    let h = spatial::group_norm(
        x,
        g,
        GN_GROUPS,
        &r.norm_in.weight,
        &r.norm_in.bias,
        GN_EPS,
        true,
    );
    let (h, g1) = conv(&h, g, &r.conv_in);
    // `[scale | shift]`, per VOXEL — see `model::port::TFREQ_VOXELS` on why
    // this chain is not a lane vector.
    let m = linear(&r.emb, &ops::elemwise::silu(temb));
    let h = spatial::group_norm(
        &h,
        &g1,
        GN_GROUPS,
        &r.norm_out.weight,
        &r.norm_out.bias,
        GN_EPS,
        false,
    );
    let h = ops::elemwise::modulate(&h, &m, None, ModulateForm::ScaleShift);
    let h = ops::elemwise::silu(&h);
    let (h, _) = conv(&h, &g1, &r.conv_out);
    let skip = match &r.skip {
        Some(c) => conv(x, g, c).0,
        None => x.clone(),
    };
    ops::elemwise::add(&skip, &h)
}

/// The `image.in` reading: `patch_embed` — `Conv2d(32 → 1024, 3×3)` then
/// one `ResBlock(1024 → hidden)` conditioned on `time_embed(t)`. The
/// `rearrange('b c h w -> b (h w) c')` that follows in the reference is
/// this axis's own layout, so nothing is traced for it.
fn image_in(arm: &Input<Facts>, m: &Model) {
    let g = arm.grid();
    let clip = arm.voxels(
        port::LATENT_VOXELS,
        LATENT_CHANNELS + T_FREQ_DIM,
        Dtype::Bf16,
    );
    let (z, freqs) = ops::layout::split_rows(&clip, LATENT_CHANNELS);
    let temb = embedder(&m.time_embed, &freqs);
    let (h, g1) = conv(&z, &g, &m.patch_embed.conv_in);
    let x = resblock(&h, &g1, &m.patch_embed.res, &temb);
    seam::at(seam::PIXELS, &[&x, &g1]);
}

/// The `image.out` reading: `final_layer` — one `ResBlock(hidden → 1024)`
/// conditioned on `time_embed_2(t)`, then `GN → SiLU → Conv2d(1024 → 32,
/// 3×3)`. The rows it lands are the flow-matching velocity.
fn image_out(arm: &Input<Facts>, m: &Model) {
    let g = arm.grid();
    let clip = arm.voxels(port::ROW_VOXELS, m.dims.hidden + T_FREQ_DIM, Dtype::Bf16);
    let (rows, freqs) = ops::layout::split_rows(&clip, m.dims.hidden);
    let temb = embedder(&m.time_embed_2, &freqs);
    let x = resblock(&rows, &g, &m.final_layer.res, &temb);
    let h = spatial::group_norm(
        &x,
        &g,
        GN_GROUPS,
        &m.final_layer.norm_out.weight,
        &m.final_layer.norm_out.bias,
        GN_EPS,
        true,
    );
    let (v, gv) = conv(&h, &g, &m.final_layer.conv_out);
    seam::at(seam::PIXELS, &[&v, &gv]);
}

/// Nothing in this family is platform-conditional; the import is, and
/// `Platform` reaches it there.
#[allow(dead_code)]
fn platform_is_stated(_: Platform) {}
