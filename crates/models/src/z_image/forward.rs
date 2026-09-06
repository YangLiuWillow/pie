//! Z-Image's traced arithmetic: five readings of one plan, selected per
//! lane by the reading bits of the fact word (design D1, D5).
//!
//! | reading | lanes (stream) | binds | reads back |
//! |---|---|---|---|
//! | `text` | one, `Text` | `embed(ids)`, `attention(kv)` | `hidden` `[L, 2560]`: Qwen3 layer −2 |
//! | `refine` | one, `Context` | `caption` `[L32, 2560]`, `pad` `[L32, 1]`, `positions` `[L32, 3]` | `hidden` `[L32, 3840]`: the refined caption |
//! | `denoise` | `Image` + `Context`, one group | image: `pad` `[N32, 1]`, `latents` `[N32, 64]`, `positions`, `timestep`; context: `context` `[L32, 3840]`, `positions`, `timestep` | `velocity` `[N32, 64]` on the image lane |
//! | `vae.decode` | one, `Image`, one clip `{1, h, w}` | `latent` `[h·w, 16]` on the voxel axis | `pixels` `[8h·8w, 3]` in `[-1, 1]` |
//! | `vae.encode` | one, `Image`, one clip `{1, H, W}` | `pixels` `[H·W, 3]` on the voxel axis | `pixels` `[H/8·W/8, 16]`: the posterior mean |
//!
//! (Port ORDER within a reading is load-bearing — an index is a port's
//! position among its kind — and `(kind, index)` is seated once per plan
//! at one width; `super::model::port` says how the readings share.)
//!
//! The two VAE readings ([`super::vae`]) exist on the flagship only (the
//! miniature has no VAE) and run on the voxel axis: a lane submits one
//! clip, its `Voxels` port's channel is `[h, w, C]`, and it reads its
//! pixels back with the clip's output box.
//!
//! `L32`/`N32` are the caption / image row counts padded up to a multiple of
//! [`super::model::SEQ_MULTIPLE`]. **The pad rows are the guest's to
//! allocate and this text's to fill**: the IR cannot grow a lane, so a lane
//! arrives already padded, its `pad` port flags each row (`0.0` real, `1.0`
//! pad), and the plan overwrites every flagged row with the learned
//! `x_pad_token` / `cap_pad_token` after its embedder (the reference's
//! `torch.where(mask, pad_token, feats)`). The pad rows are attended (study
//! §C.3); the velocity rows they produce are discarded by the guest. The
//! `denoise` context lane binds no `pad`: its rows are the `refine`
//! readout, pads included. The `timestep` is bound by BOTH denoise lanes
//! (the same cell): the joint trunk modulates every row by its own lane's
//! vector, caption rows included.
//!
//! Positions (`AxisPositions`, `[rows, 3]` f32, `(t, h, w)`), as the
//! reference's `_pad_with_ids` states them: caption row `j` is
//! `(1 + j, 0, 0)` for EVERY row `j < L32`, pads included (the caption's
//! coordinate grid spans its padded length); image patch `(a, b)` is
//! `(L32 + 1, a, b)` — the image's temporal index depends on the caption's
//! padded length (study §C.5) — and an image pad row is `(0, 0, 0)`.
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
    AxisRole, Generative, LatentSpace, PortFact, PortKind, PositionConvention, ReadingFact,
    ReadoutKind, ScheduleFact, ScheduleKind,
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

/// The three bits the reading index lives in, as a plain binary code: bit
/// [`READING_LO`] is its low bit, [`READING_MID`] the middle one,
/// [`READING_HI`] its high bit. Eight codes, five readings on the flagship.
pub const READING_LO: u8 = 6;
pub const READING_MID: u8 = 7;
pub const READING_HI: u8 = 8;

/// Which reading index means what, per row. The word packs the index the
/// runtime stamps (`Request::reading`) and nothing else — `Classify::of` has
/// no model to ask — so the *meaning* of a code is the row's: the flagship
/// runs `text` at 0, the miniature (no encoder) runs `refine` at 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    pub text: Option<u8>,
    pub refine: u8,
    pub denoise: u8,
    /// The VAE readings, on a row that carries the VAE.
    pub vae_decode: Option<u8>,
    pub vae_encode: Option<u8>,
}

impl Model {
    /// This row's reading codes, dense from 0 in `Generative::readings`
    /// order (what `validate_generative` demands).
    #[must_use]
    pub fn readings(&self) -> Readings {
        let mut next = 0u8;
        let mut take = |present: bool| {
            present.then(|| {
                next += 1;
                next - 1
            })
        };
        let text = take(self.te.is_some());
        let refine = take(true).unwrap_or(0);
        let denoise = take(true).unwrap_or(0);
        let vae_decode = take(self.vae.is_some());
        let vae_encode = take(self.vae.is_some());
        Readings {
            text,
            refine,
            denoise,
            vae_decode,
            vae_encode,
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
            at: None,
            rows: None,
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
                positions: None,
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
            // `(t, h, w)`, one lane: caption row `j` at `(1 + j, 0, 0)`,
            // a pad row at the origin (study §C.5).
            positions: Some(PositionConvention {
                axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
                text_axis: 0,
                text_origin: 1,
                image_follows_text: false,
            }),
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
                port("pad", PortKind::Latents, 1, &[Stream::Image]),
                port(
                    "latents",
                    PortKind::Latents,
                    PATCH_FEATURES,
                    &[Stream::Image],
                ),
                // A Latents port (index 2), not a Context one: the plan's
                // Context 0 is the raw caption at `cap_width`, and a
                // `(kind, index)` pair is seated once per plan.
                port("context", PortKind::Latents, d.dim, &[Stream::Context]),
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
            // `(t, h, w)`: the caption rides the TIME axis ahead of the
            // image — caption row `j` at `(1 + j, 0, 0)`, image patch
            // `(a, b)` at `(L32 + 1, a, b)` — so the image's time index
            // follows the caption's padded length (study §C.5). Pad rows
            // sit at the origin, which the guest's grid states.
            positions: Some(PositionConvention {
                axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
                text_axis: 0,
                text_origin: 1,
                image_follows_text: true,
            }),
            readout: ReadoutKind::Velocity,
            readout_width: Tap::width(Tap::from_env().as_deref(), d),
        });
        if let (Some(decode), Some(encode), Some(_)) =
            (codes.vae_decode, codes.vae_encode, &self.vae)
        {
            readings.push(ReadingFact {
                name: "vae.decode",
                index: decode,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Image],
                ports: vec![port("latent", PortKind::Voxels, CHANNELS, &[Stream::Image])],
                // A VAE tile is a box on the voxel axis, not rows in a
                // rotary space: it takes no positions and states no
                // convention.
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: super::vae::RGB,
            });
            readings.push(ReadingFact {
                name: "vae.encode",
                index: encode,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Image],
                // Voxel index ONE: the engine seats one rectangle per
                // `(kind, index)` for the whole plan, and `vae.decode`'s
                // latent clip is 16 wide at index 0.
                ports: vec![PortFact {
                    name: "pixels",
                    kind: PortKind::Voxels,
                    width: super::vae::RGB,
                    streams: vec![Stream::Image],
                    at: Some(super::model::port::PIXEL_VOXELS),
                    rows: None,
                }],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: CHANNELS,
            });
        }
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
                // One backbone, one schedule: every lane takes `shift`.
                stream_shifts: vec![],
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

/// **THE PARITY HARNESS'S BISECTION KNOB**, the `mini_dit::forward::Tap`
/// idiom spelled for this row. Thirty-four blocks between a caption and a
/// velocity say nothing about WHERE two runs diverged, so
/// `PIE_Z_IMAGE_TAP=<key>` makes the `denoise` reading plant
/// [`seam::VELOCITY`] on ONE intermediate and stop there; the guest's
/// `velocity(width)` then reads that rectangle off its own lane's rows and
/// `scripts/imagegen/zimage_parity.py` diffs it against the matching
/// `zimage_golden.py --taps` key.
///
/// The keys, in the order the arm computes them: `latents` (the port as it
/// landed), `x_linear` (`x_embedder`, the golden's `x.embed`), `x_embed`
/// (its pad rows substituted), `normed0` / `scaled0` and `b0.{q,k,v,attn,
/// out,norm2,res1,ffn,ffn_norm2,res2}` inside noise refiner 0, `refiner{l}`,
/// `x_refined` (the golden's `x.refined`), `layer{l}` (joint block `l`, the
/// golden's `layer{l}.out` — this one carries BOTH lanes, so a guest that
/// reads the context lane back too gets its caption half), `final_norm` and
/// `final_linear` (the golden's `out.0`, before the sign flip).
///
/// **A tap TRUNCATES the plan.** The seam is an export that runs at the end
/// of the plan, so a seam planted on a live intermediate while the rest of
/// the arm still runs reads whatever recycled that buffer — the tap must be
/// the last thing the arm computes. Truncating drops the ports the rest of
/// the arm would have read, so a tapped arm reads the context port up front
/// and merges the tapped rectangle with it: the plan keeps declaring the
/// port the context lane feeds, and the merge covers both classes.
///
/// This is the one place this family reads the environment, and it is read
/// at catalog time, for a row nothing real is served by.
pub struct Tap;

impl Tap {
    /// The environment variable, read at catalog time.
    pub const ENV: &'static str = "PIE_Z_IMAGE_TAP";

    /// The requested tap, or `None` for the model as it is.
    #[must_use]
    pub fn from_env() -> Option<String> {
        std::env::var(Self::ENV).ok().filter(|key| !key.is_empty())
    }

    /// The width of the rectangle a tap exports: the patch features at the
    /// arm's own two rectangles, the trunk width everywhere else.
    #[must_use]
    pub fn width(key: Option<&str>, d: &Dims) -> u32 {
        match key {
            None | Some("latents") | Some("final_linear") => PATCH_FEATURES,
            Some(_) => d.dim,
        }
    }
}

/// The per-lane facts: which stream the lane's rows are, and which reading
/// its pass runs.
pub struct Facts {
    pub stream: Stream,
    /// The reading index, `0..8` (a wider index is truncated to three bits).
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

    /// The middle reading bit.
    #[must_use]
    pub fn reading_mid() -> Predicate {
        Predicate::fact(READING_MID)
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
            reading: r.reading() & 7,
        }
    }

    fn word(&self) -> u64 {
        self.stream.word(STREAM_BASE) | (u64::from(self.reading & 7) << READING_LO)
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
        // Eight arms by reading code, each a conjunction of the three
        // reading literals (so every arm names a `Selection` the host can
        // pack); a code no reading claims runs no node.
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (hi_mid, hi_low) = hi.split(&Facts::reading_mid());
        let (lo_mid, lo_low) = lo.split(&Facts::reading_mid());
        let (c7, c6) = hi_mid.split(&Facts::reading_lo());
        let (c5, c4) = hi_low.split(&Facts::reading_lo());
        let (c3, c2) = lo_mid.split(&Facts::reading_lo());
        let (c1, c0) = lo_low.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3, c4, c5, c6, c7];
        let arm = |code: u8| &arms[usize::from(code)];

        if let (Some(code), Some(te)) = (codes.text, &self.te) {
            text_encode(arm(code), te);
        }
        refine(arm(codes.refine), &self.dims, &self.dit);
        let velocity = denoise(arm(codes.denoise), &self.dims, &self.dit);
        if let (Some(decode), Some(encode), Some(vae)) =
            (codes.vae_decode, codes.vae_encode, &self.vae)
        {
            super::vae::decode(arm(decode), vae);
            super::vae::encode(arm(encode), vae);
        }
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
    let c = arm.context(port::CAPTION, d.cap_width);
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
    // [`Tap`]: the bisect's context rows, read up front so a TRUNCATED plan
    // still declares the port the context lane feeds.
    let tap = Tap::from_env().unwrap_or_default();
    let c_early = (!tap.is_empty()).then(|| ctx.latents(port::CONTEXT_REFINED, d.dim, Dtype::Bf16));
    // `tap!(key, value)`: under [`Tap`] `key`, seam the image rectangle
    // merged with the context lane's own (narrowed to the same width, so
    // the merge is one rectangle) and return there.
    macro_rules! tap {
        ($name:expr, $v:expr) => {
            if tap == $name {
                return seam_tapped($v, c_early.expect("a tapped arm reads the context port"));
            }
        };
    }
    tap!("latents", x.clone());
    let x = linear(&m.x_embed, &x);
    tap!("x_linear", x.clone());
    let flag = img.latents(port::PAD_IMAGE, 1, Dtype::Bf16);
    let mut x = pad_rows(&x, &flag, &m.x_pad_mod);
    tap!("x_embed", x.clone());
    for (l, block) in img.walk_layers(&m.noise_refiner) {
        let mods = adaln4(
            &linear(
                block.ada.as_ref().expect("a noise refiner is modulated"),
                &temb_img,
            ),
            d.dim,
        );
        if l == 0 {
            tap!(
                "normed0",
                ops::elemwise::rmsnorm(&x, &block.attn_norm1, NORM_EPS)
            );
            tap!("scaled0", {
                let h = ops::elemwise::rmsnorm(&x, &block.attn_norm1, NORM_EPS);
                ops::elemwise::modulate(&h, &mods.scale_msa, Some(&img_lanes), ModulateForm::Scale)
            });
        }
        let want = if l == 0 { tap.as_str() } else { "" };
        let (next, hit) = run_block_tapped(&x, block, Some((&mods, &img_lanes)), d, &own, want);
        if let Some(hit) = hit {
            return seam_tapped(hit, c_early.expect("a tapped arm reads the context port"));
        }
        x = next;
        tap!(format!("refiner{l}"), x.clone());
    }

    tap!("x_refined", x.clone());
    // ---- the caption lane: already refined, pads included ---------------
    let c = c_early
        .clone()
        .unwrap_or_else(|| ctx.latents(port::CONTEXT_REFINED, d.dim, Dtype::Bf16));

    // ---- the joint trunk over `[image ‖ caption]` ------------------------
    let mut u = Value::merge(vec![x, c]);
    for (l, block) in arm.walk_layers(&m.layers) {
        let mods = adaln4(
            &linear(
                block.ada.as_ref().expect("a joint block is modulated"),
                &temb,
            ),
            d.dim,
        );
        u = run_block(&u, block, Some((&mods, &lanes)), d, &joint);
        if tap == format!("layer{l}") {
            // Both lanes already: the merge is `u`'s own two classes, so
            // the context lane reads its caption half of the same seam.
            let (ui, ci) = u.split(&Facts::image());
            let both = Value::merge(vec![ui, ci]);
            seam::at(seam::VELOCITY, &[&both]);
            return both;
        }
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
    tap!("final_norm", h.clone());
    let v = linear(&m.final_linear, &h);
    tap!("final_linear", v.clone());
    // The reference pipeline's `noise_pred = -noise_pred`.
    let velocity = ops::elemwise::mul_scalar(-1.0, &v);
    seam::at(seam::VELOCITY, &[&velocity]);
    velocity
}

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

/// [`Tap`]'s seam: the tapped image rectangle merged with the context
/// lane's rows narrowed to the same width — the plan then declares the port
/// that lane feeds (a truncated arm never reaches its own read of it) and
/// every class of the merge is covered.
fn seam_tapped(image: Value, context: Value) -> Value {
    let width = u32::try_from(image.width()).unwrap_or(u32::MAX);
    let context = if u64::from(width) < context.width() {
        ops::layout::split_rows(&context, width).0
    } else {
        context
    };
    let both = Value::merge(vec![image, context]);
    seam::at(seam::VELOCITY, &[&both]);
    both
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
    run_block_tapped(x, b, mods, d, g, "").0
}

/// [`run_block`] with a [`Tap`] hook: the second half of the answer is the
/// tapped intermediate, for the caller to seam and return there (a seam
/// planted while the rest of the arm still runs reads a recycled buffer).
fn run_block_tapped(
    x: &Value,
    b: &Block,
    mods: Option<(&Mods, &Value)>,
    d: &Dims,
    g: &Geom,
    tap: &str,
) -> (Value, Option<Value>) {
    let mut hit: Option<Value> = None;
    macro_rules! tap {
        ($name:expr, $v:expr) => {
            if tap == $name {
                hit = Some($v);
            }
        };
    }
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
    tap!("b0.q", q.clone());
    tap!("b0.k", k.clone());
    tap!("b0.v", v.clone());
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
    tap!("b0.attn", o.clone());
    let o = ops::linear::matmul(&o, &b.attn.out);
    tap!("b0.out", o.clone());
    let o = ops::elemwise::rmsnorm(&o, &b.attn_norm2, NORM_EPS);
    tap!("b0.norm2", o.clone());
    let x = match mods {
        Some((m, lanes)) => ops::elemwise::gated_residual_add(x, &m.gate_msa, &o, Some(lanes)),
        None => ops::elemwise::residual_add(&o, x),
    };
    tap!("b0.res1", x.clone());

    let h = ops::elemwise::rmsnorm(&x, &b.ffn_norm1, NORM_EPS);
    let h = match mods {
        Some((m, lanes)) => scaled(&h, &m.scale_mlp, lanes),
        None => h,
    };
    let f = ops::linear::matmul(
        &ops::linear::mlp_swiglu(&ops::linear::matmul(&h, &b.mlp.gate_up), d.inter),
        &b.mlp.down,
    );
    tap!("b0.ffn", f.clone());
    let f = ops::elemwise::rmsnorm(&f, &b.ffn_norm2, NORM_EPS);
    tap!("b0.ffn_norm2", f.clone());
    let out = match mods {
        Some((m, lanes)) => ops::elemwise::gated_residual_add(&x, &m.gate_mlp, &f, Some(lanes)),
        None => ops::elemwise::residual_add(&f, &x),
    };
    tap!("b0.res2", out.clone());
    (out, hit)
}

// `TE_HIDDEN` is the caption width the flagship's `refine` port states;
// named here so the readings table and the model agree by construction.
const _: () = assert!(Dims::turbo().cap_width == TE_HIDDEN);
