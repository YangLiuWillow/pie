//! Wan 2.2's traced arithmetic: four arms of one plan, selected per lane
//! by the reading bits of the fact word (design D1, D5).
//!
//! | reading | lanes (stream) | binds | reads back |
//! |---|---|---|---|
//! | `text` | one, `Text` | `embed(ids)` — NO `attention`: umT5 is bidirectional and cacheless | `hidden` `[L, 4096]`: `last_hidden_state` after `final_layer_norm` |
//! | `denoise` | one or two `Video` + one `Context`, one group | video: `latents` `[N, C·4]`, `positions` `[N, 3]`, `timestep`; context: `context` `[512, 4096]` | `velocity` `[N, C·4]` on every video lane |
//! | `vae.decode.head`, `vae.decode` (arms only, see below) | one, `Video` | `voxels` `[V, 48]` + the grid | `pixels` `[64·V, 3]` (head: `16·V`) |
//!
//! **The timestep is per lane, and TI2V's per-token timestep is two
//! lanes.** The reference's TI2V path (`expand_timesteps`) hands the
//! transformer a `[B, S]` timestep in which the first latent frame's
//! tokens carry `0` (the conditioning image, clean) and every other token
//! carries `t`; the modulation then runs per token (`[B, S, 6, dim]`).
//! Every token of one frame shares its value, so this text keeps the
//! per-LANE modulation of `IMAGEGEN_CONTRACT.md` §3 (an f32 lane-vector
//! chain, `Modulate` broadcast by `request_of_token`) and a TI2V step
//! submits the video as TWO `Stream::Video` lanes of one group — frame 0's
//! tokens at `timestep = 0`, the rest at `t` — each with its own
//! `timestep` cell and its own `positions`; the self-attention packs the
//! group's video lanes into one sequence (`GroupBlockDiagonal` over the
//! video selection's group CSR), which with explicit rotary positions is
//! the reference's one sequence in another row order. A T2V step is one
//! video lane. A per-token `[rows, 1]` timestep port would need an f32
//! token-axis GEMM chain this shell has no arm for; the two-lane form
//! needs nothing new and is exactly the reference's numbers.
//!
//! **Positions** (`AxisPositions`, `[rows, 3]` f32, `(t, h, w)` in PATCH
//! units): token `(t, h, w)` of the `T' × H/2 × W/2` grid, whatever lane
//! it is on. The context lane carries none: cross-attention has no rope.
//!
//! **The context is 512 zero-padded rows, attended without a mask.** The
//! `text` reading answers the prompt's `L` rows; the pipeline truncates
//! to `L` and zero-pads to 512 (study §C.6), and the transformer attends
//! every one of the 512 keys. The `context` port takes those 512 rows
//! as the reference builds them — the guest allocates `[512, 4096]`
//! zero-filled and writes the `L` real rows in front. (The IR cannot grow
//! a lane; an SDK `encode_text` helper owning this pad is the follow-up.)
//! The miniatures' goldens hand a 32-row random context; nothing here
//! pins 512.
//!
//! **The latent row layout** is `(c, ph, pw)` — `patch_embedding`'s own
//! `Conv3d` input order — on the way in AND on the way out: `proj_out`'s
//! rows are permuted at import (the checkpoint's `(ph, pw, c)`), so a
//! guest's Euler step is elementwise over one layout.
//!
//! **The two `vae.decode` arms are traced, imported, and NOT declared in
//! the facts.** `models::PortKind`/`ReadoutKind` have no `Voxels`/`Pixels`
//! (the facts vocabulary stops at the engine, as for `flux_2`), and the
//! mid block's single-head attention (`head_dim = 1024`, per frame) has
//! neither a kernel arm (`attention.ragged` serves 64/128/256) nor a
//! voxel-axis indptr to segment by, so the arms below are the decoder
//! MINUS that attention and their pixels are not the reference's yet.
//! Two arms because the reference decodes one latent frame at a time
//! through per-conv frame caches and treats the FIRST frame apart: the
//! `upsample3d` time conv is skipped (`"Rep"`) and `DupUp3D` emits frame
//! 0 once. `vae.decode.head` takes exactly ONE latent frame per clip (its
//! time convs are absent, its `(2, 2, 2)` shortcuts are `(1, 2, 2)`);
//! `vae.decode` takes any later frame range with every causal conv
//! reading its `CacheRow::State` slab, which `Shell::open` zeroes — the
//! zero front padding the reference gives its first chunk. The encoder is
//! not traced (its `AvgDown3D` mean over a `(2, 2, 2)` window has no
//! `Spatial` member).

use model_dsl::ops::spatial;
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};
use model_ir::TimePad;

use crate::{
    Generative, LatentSpace, PortFact, PortKind, ReadingFact, ReadoutKind, ScheduleFact,
    ScheduleKind,
};

use super::model::{
    Block, CONTEXT_LEN, Conv, Dims, Dit, Linear, MOD_SLICES, Model, NORM_EPS, PATCH_H, PATCH_T,
    PATCH_W, ROPE_AXES, ROPE_THETA, Resnet, Shortcut, T_FLIP_SIN_COS, T_MAX_PERIOD, T_SCALE,
    TE_BUCKETS, TE_EPS, TE_HEAD_DIM, TE_HIDDEN, TE_MAX_DISTANCE, TE_MAX_TOKENS, TE_VOCAB,
    TRAIN_STEPS, TextEncoder, VAE_EPS, VAE_PATCH, VAE_SPATIAL_COMPRESSION,
    VAE_TEMPORAL_COMPRESSION, VAE_Z, Vae, port,
};

/// The bit the one-hot stream facts start at (D2): bits 0..6 are the six
/// streams, of which this text names Text, Video and Context.
pub const STREAM_BASE: u8 = 0;

/// The two bits the reading index lives in, as a plain binary code: bit
/// [`READING_LO`] is its low bit, [`READING_HI`] its high bit. Four codes:
/// `text`, `denoise`, `vae.decode.head`, `vae.decode`; a `vae.encode`
/// or a second backbone (`denoise.low`, D9) is a third bit away.
pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;

/// Which reading code means what, per row: the flagship runs `text` at 0,
/// the miniatures (no encoder) run `denoise` at 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    pub text: Option<u8>,
    pub denoise: u8,
    /// The first-frame decoder arm, on a row that carries a VAE.
    pub vae_decode_head: Option<u8>,
    /// The later-frames decoder arm.
    pub vae_decode: Option<u8>,
}

impl Model {
    /// This row's reading codes, dense from 0 in `Generative::readings`
    /// order for the declared ones (what `validate_generative` demands).
    #[must_use]
    pub fn readings(&self) -> Readings {
        let mut next = 0u8;
        let mut take = || {
            let code = next;
            next += 1;
            code
        };
        let text = self.te.as_ref().map(|_| take());
        let denoise = take();
        let vae_decode_head = self.vae.as_ref().map(|_| take());
        let vae_decode = self.vae.as_ref().map(|_| take());
        Readings {
            text,
            denoise,
            vae_decode_head,
            vae_decode,
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
        };
        let mut readings = Vec::new();
        if let (Some(index), Some(_)) = (codes.text, &self.te) {
            readings.push(ReadingFact {
                name: "text",
                index,
                // Bidirectional and cacheless: ids in, no kv space.
                has_kv: false,
                takes_tokens: true,
                streams: vec![Stream::Text],
                ports: vec![],
                positions: None,
                readout: ReadoutKind::Hidden,
                readout_width: TE_HIDDEN,
            });
        }
        readings.push(ReadingFact {
            name: "denoise",
            index: codes.denoise,
            has_kv: false,
            takes_tokens: false,
            streams: vec![Stream::Video, Stream::Context],
            ports: vec![
                port("latents", PortKind::Latents, d.patch_in(), &[Stream::Video]),
                port("context", PortKind::Context, d.text_dim, &[Stream::Context]),
                port("timestep", PortKind::LaneVector, 1, &[Stream::Video]),
                port(
                    "positions",
                    PortKind::AxisPositions,
                    u32::from(ROPE_AXES),
                    &[Stream::Video],
                ),
            ],
            // NO CONVENTION STATED. `PositionConvention`'s image half is a
            // 2-D `(h, w)` grid and a Wan token is `(t, h, w)` over a
            // `T' x H/2 x W/2` volume; its text half does not apply either
            // (the context lane binds no positions — cross-attention has no
            // rope). A video family fits once the convention grows a
            // temporal extent; until then this is `none` and a guest of
            // this row builds its own grid, which is what `none` means.
            positions: None,
            readout: ReadoutKind::Velocity,
            readout_width: d.patch_out(),
        });
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: d.in_channels,
                patch_t: PATCH_T,
                patch_h: PATCH_H,
                patch_w: PATCH_W,
                spatial_compression: VAE_SPATIAL_COMPRESSION,
                temporal_compression: VAE_TEMPORAL_COMPRESSION,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: self.shift,
                train_steps: TRAIN_STEPS,
                boundary: None,
                // `use_dynamic_shifting: false`: the guest builds
                // `linspace` through the static shift; nothing is pinned.
                pinned_sigmas: vec![],
                // One backbone, one schedule: every lane takes `shift`.
                stream_shifts: vec![],
            }),
            // 1280×704 at 121 frames is 31 × 22 × 40 = 27 280 tokens plus
            // the 512-row context; the miniatures' reference grid is
            // 5 × 8 × 8.
            max_rows: match self.te {
                Some(_) => 32_768 + CONTEXT_LEN,
                None => 4096,
            },
        }
    }
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
    pub fn video() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Video)
    }

    #[must_use]
    pub fn context() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Context)
    }

    #[must_use]
    pub fn reading_lo() -> Predicate {
        Predicate::fact(READING_LO)
    }

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

    /// No kv space anywhere: the encoder is cacheless and the denoiser
    /// holds nothing between fires. The VAE's causal convs each hold their
    /// last two input frames per slot (`model::Conv::slab`).
    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        if let Some(vae) = &self.vae {
            for conv in vae.cached_convs() {
                let name = conv.cache.as_ref().expect("a cached conv names its slab");
                c.state(name.clone(), conv.slab(), Dtype::Bf16);
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
        let velocity = denoise(arm(codes.denoise), self);
        if let (Some(head), Some(rest), Some(vae)) =
            (codes.vae_decode_head, codes.vae_decode, &self.vae)
        {
            vae_decode(arm(head), vae, true);
            vae_decode(arm(rest), vae, false);
        }
        velocity
    }
}

/// The `text` reading: `UMT5EncoderModel` over the prompt's ids, unpadded
/// — pre-norm relative-bias attention over each lane's own rows at
/// `sm_scale = 1` (T5 folds the scale into its weights), a gated-GELU MLP,
/// `final_layer_norm`, `hidden` planted on the result. No positions, no
/// kv, no head.
fn text_encode(arm: &Input<Facts>, te: &TextEncoder) {
    let ids = arm.tokens();
    let perm = arm.row_permutation();
    let csr = arm.lane_indptr();
    let mut y = ops::layout::embed(&ids, &te.embed, TE_VOCAB);
    for (_, w) in arm.walk_layers(&te.layers) {
        // Every umT5 layer owns its bucket embedding: one table per layer.
        let table = ops::elemwise::relative_bucket_bias(
            arm.recorder(),
            &w.rel_bias,
            TE_MAX_TOKENS,
            TE_BUCKETS,
            TE_MAX_DISTANCE,
            true,
        );
        let x = ops::elemwise::rmsnorm(&y, &w.attn_norm, TE_EPS);
        let q = ops::linear::matmul(&x, &w.q);
        let k = ops::linear::matmul(&x, &w.k);
        let v = ops::linear::matmul(&x, &w.v);
        let o = ops::attn::ragged(
            &ops::layout::pack_rows(&q, &perm),
            &ops::layout::pack_rows(&k, &perm),
            &ops::layout::pack_rows(&v, &perm),
            &csr,
            &csr,
            TE_HEAD_DIM,
            1.0,
            ops::attn::relative_bias(&table, TE_MAX_TOKENS),
        );
        let o = ops::layout::unpack_rows(&o, &perm);
        y = ops::elemwise::residual_add(&ops::linear::matmul(&o, &w.o), &y);

        let x = ops::elemwise::rmsnorm(&y, &w.ffn_norm, TE_EPS);
        let gate = ops::linear::matmul(&x, &w.wi_0);
        let up = ops::linear::matmul(&x, &w.wi_1);
        // `gelu_new` is the tanh approximation.
        let act = ops::linear::mlp_geglu_tanh(&gate, &up);
        y = ops::elemwise::residual_add(&ops::linear::matmul(&act, &w.wo), &y);
    }
    let out = ops::elemwise::rmsnorm(&y, &te.final_norm, TE_EPS);
    seam::at(seam::HIDDEN, &[&out]);
}

/// One biased projection.
fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

/// The tables the video side's attentions need: the token→lane map, the
/// rotary coordinates, the video selection's permutation and group CSR.
struct VideoGeom {
    lanes: Value,
    positions: Value,
    perm: Value,
    csr: Value,
}

/// The context selection's permutation and group CSR: the key side of
/// every cross-attention.
struct ContextGeom {
    perm: Value,
    csr: Value,
}

/// A block's modulation vector `[Lanes, 6·dim]` f32 cut into what it
/// applies: the attention `[scale | shift]` pair and gate, the FFN pair
/// and gate — the plan's own slice order, which `import.rs` makes.
struct Mods {
    attn_ss: Value,
    attn_gate: Value,
    ffn_ss: Value,
    ffn_gate: Value,
}

fn adaln6(e: &Value, dim: u32) -> Mods {
    debug_assert_eq!(e.width(), u64::from(MOD_SLICES * dim));
    let (attn_ss, rest) = ops::layout::split_rows(e, 2 * dim);
    let (attn_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (ffn_ss, ffn_gate) = ops::layout::split_rows(&rest, 2 * dim);
    Mods {
        attn_ss,
        attn_gate,
        ffn_ss,
        ffn_gate,
    }
}

/// `FP32LayerNorm(affine=False)` then `x·(1+scale)+shift`, per lane.
fn norm_modulate(x: &Value, scale_shift: &Value, lanes: &Value) -> Value {
    ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(x, NORM_EPS),
        scale_shift,
        Some(lanes),
        ModulateForm::ScaleShift,
    )
}

/// One `WanTransformerBlock` over the video rows `x`, reading the embedded
/// context rows `c` in its cross-attention.
fn block(
    x: &Value,
    c: &Value,
    b: &Block,
    e: &Value,
    d: &Dims,
    vg: &VideoGeom,
    cg: &ContextGeom,
) -> Value {
    let dim = d.dim;
    let hd = d.head_dim;
    let m = adaln6(e, dim);

    // 1. Self-attention: across-heads QK RMSNorm, three-axis interleaved
    //    rope, one ragged read over the group's video rows, a gated fold.
    let h = norm_modulate(x, &m.attn_ss, &vg.lanes);
    let (q, k, v) = ops::layout::split_qkv(&linear(&b.self_attn.qkv, &h), dim, dim);
    let turn = |x: &Value, gain: &Weight| {
        ops::elemwise::rope_axes(
            &ops::elemwise::rmsnorm(x, gain, NORM_EPS),
            &vg.positions,
            d.rope_dims(),
            [ROPE_THETA; 4],
            RopeForm::Interleaved,
            hd,
            hd,
        )
    };
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&turn(&q, &b.self_attn.norm_q), &vg.perm),
        &ops::layout::pack_rows(&turn(&k, &b.self_attn.norm_k), &vg.perm),
        &ops::layout::pack_rows(&v, &vg.perm),
        &vg.csr,
        &vg.csr,
        hd,
        d.sm_scale(),
        RaggedMask::GroupBlockDiagonal,
    );
    let o = ops::layout::unpack_rows(&o, &vg.perm);
    let x = ops::elemwise::gated_residual_add(
        x,
        &m.attn_gate,
        &linear(&b.self_attn.out, &o),
        Some(&vg.lanes),
    );

    // 2. Cross-attention: the affine norm, no modulation, no rope, the
    //    queries off the video arm and the keys off the context arm — the
    //    one op whose operands may come from two arms — and an ungated
    //    residual.
    let hc = ops::elemwise::layernorm(&x, &b.norm2, &b.norm2_bias, NORM_EPS);
    let cq = ops::elemwise::rmsnorm(&linear(&b.cross.q, &hc), &b.cross.norm_q, NORM_EPS);
    let (ck, cv) = ops::layout::split_rows(&linear(&b.cross.kv, c), dim);
    let ck = ops::elemwise::rmsnorm(&ck, &b.cross.norm_k, NORM_EPS);
    let ca = ops::attn::ragged(
        &ops::layout::pack_rows(&cq, &vg.perm),
        &ops::layout::pack_rows(&ck, &cg.perm),
        &ops::layout::pack_rows(&cv, &cg.perm),
        &vg.csr,
        &cg.csr,
        hd,
        d.sm_scale(),
        RaggedMask::GroupBlockDiagonal,
    );
    let ca = ops::layout::unpack_rows(&ca, &vg.perm);
    let x = ops::elemwise::residual_add(&linear(&b.cross.out, &ca), &x);

    // 3. The FFN: modulated norm, `Linear → GELU(tanh) → Linear`, a gated
    //    fold.
    let hf = norm_modulate(&x, &m.ffn_ss, &vg.lanes);
    let f = linear(
        &b.ffn.down,
        &ops::elemwise::gelu(&linear(&b.ffn.up, &hf), true),
    );
    ops::elemwise::gated_residual_add(&x, &m.ffn_gate, &f, Some(&vg.lanes))
}

/// The `denoise` reading.
fn denoise(arm: &Input<Facts>, m: &Model) -> Value {
    let d = &m.dims;
    let dit: &Dit = &m.dit;

    // The context lane on one side; the video lane(s) on the other.
    let (ctx, vid) = arm.split(&Facts::context());
    let vg = VideoGeom {
        lanes: vid.request_of_token(),
        positions: vid.axis_positions(port::POSITIONS, ROPE_AXES),
        perm: vid.row_permutation(),
        csr: vid.group_indptr(),
    };
    let cg = ContextGeom {
        perm: ctx.row_permutation(),
        csr: ctx.group_indptr(),
    };

    // ---- the conditioning vectors, once per video lane -------------------
    // `[cos | sin]` sinusoid of the scheduler timestep; `time_embedder`
    // (`linear_2(silu(linear_1(·)))`); `time_proj(silu(temb))` shared by
    // every block; and the head's `[temb | temb]` off the same hidden.
    let t = vid.lane_vector(port::TIMESTEP, 1);
    let h_t = ops::elemwise::silu(&linear(
        &dit.time_embed.linear_1,
        &ops::elemwise::sinusoid(&t, d.freq_dim, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE),
    ));
    let temb = linear(&dit.time_embed.linear_2, &h_t);
    let proj = linear(&dit.time_proj, &ops::elemwise::silu(&temb));
    let head_mod = ops::elemwise::add_bias(&dit.head_table, &linear(&dit.head_proj, &h_t));

    // ---- the context rows: `text_embedder`, once per fire ----------------
    let c = ctx.context(port::CONTEXT, d.text_dim);
    let c = linear(
        &dit.text_embed.linear_2,
        &ops::elemwise::gelu(&linear(&dit.text_embed.linear_1, &c), true),
    );

    // ---- the video rows: `patch_embedding` as a linear over patch rows --
    let mut x = linear(
        &dit.patch_embed,
        &vid.latents(port::LATENTS, d.patch_in(), Dtype::Bf16),
    );

    for (_, b) in arm.walk_layers(&dit.blocks) {
        // `scale_shift_table + timestep_proj`, in fp32, per lane.
        let e = ops::elemwise::add_bias(&b.table, &proj);
        x = block(&x, &c, b, &e, d, &vg, &cg);
    }

    // ---- the head: `norm_out · (1 + scale) + shift`, `proj_out` -----------
    let h = norm_modulate(&x, &head_mod, &vg.lanes);
    let velocity = linear(&dit.proj_out, &h);
    seam::at(seam::VELOCITY, &[&velocity]);
    velocity
}

/// One convolution of the decoder: `k` as declared, stride 1, `same`
/// spatial padding, and — for `kt > 1` — causal time padding read from the
/// conv's frame cache.
fn conv(x: &Value, g: &Value, c: &Conv, arm: &Input<Facts>) -> (Value, Value) {
    let shape = spatial::Conv::conv3d(c.k, [1, 1, 1], [0, c.k[1] / 2, c.k[2] / 2]);
    let (shape, cache) = match &c.cache {
        Some(name) => (shape.causal(TimePad::Zero), Some(arm.state(name))),
        None => (shape, None),
    };
    spatial::conv3d(x, g, &c.w, Some(&c.bias), shape, cache)
}

/// `WanRMS_norm → SiLU`: the channel RMS norm with its gain, then SiLU in
/// place on the fresh rows.
fn norm_silu(x: &Value, gain: &Weight) -> Value {
    ops::elemwise::silu(&ops::elemwise::rmsnorm(x, gain, VAE_EPS))
}

/// `WanResidualBlock`, box-keeping: the grid in is the grid out.
fn resnet(x: &Value, g: &Value, r: &Resnet, arm: &Input<Facts>) -> Value {
    let h = norm_silu(x, &r.norm1);
    let (h, _) = conv(&h, g, &r.conv1, arm);
    let h = norm_silu(&h, &r.norm2);
    let (h, _) = conv(&h, g, &r.conv2, arm);
    let skip = match &r.shortcut {
        Some(c) => conv(x, g, c, arm).0,
        None => x.clone(),
    };
    ops::elemwise::add(&skip, &h)
}

/// A `vae.decode` arm: `post_quant_conv` → `conv_in` → the mid block (its
/// attention unserved, see the module doc) → four up blocks → `norm_out`,
/// SiLU, `conv_out` → the 2×2 depth-to-space → `clamp(−1, 1)` → `pixels`.
/// `first` is the first-frame arm.
fn vae_decode(arm: &Input<Facts>, vae: &Vae, first: bool) {
    let g0 = arm.grid();
    let z = arm.voxels(port::VOXELS, VAE_Z, Dtype::Bf16);
    let (z, g) = conv(&z, &g0, &vae.post_quant, arm);
    let (mut x, _) = conv(&z, &g, &vae.conv_in, arm);
    let mut g = g;

    x = resnet(&x, &g, &vae.mid_res0, arm);
    // UNSERVED: `vae.mid_attn` — see the module doc.
    x = resnet(&x, &g, &vae.mid_res1, arm);

    for up in &vae.up {
        let (x_in, g_in) = (x.clone(), g.clone());
        for r in &up.resnets {
            x = resnet(&x, &g, r, arm);
        }
        if let Some(u) = &up.upsampler {
            if let (Some(tc), false) = (&u.time_conv, first) {
                // The causal `(3, 1, 1)` conv doubles the channels; its rows
                // are permuted at import so the depth-to-space along time
                // reads `(c, r1)`: channel `2c` is the even frame, `2c + 1`
                // the odd.
                let (y, gy) = conv(&x, &g, tc, arm);
                let (y, gy) = spatial::pixel_shuffle(&y, &gy, [2, 1, 1]);
                x = y;
                g = gy;
            }
            let (y, gy) = spatial::upsample_nearest(&x, &g, [1, 2, 2], false);
            let (y, gy) = conv(&y, &gy, &u.resample, arm);
            x = y;
            g = gy;
        }
        if let Some(shortcut) = up.shortcut {
            let s = match shortcut {
                // `DupUp3D(2, 2, 2)`: on the first frame, emitted once —
                // for a one-frame tile, a plain spatial doubling.
                Shortcut::Nearest222 => {
                    let factor = if first { [1, 2, 2] } else { [2, 2, 2] };
                    spatial::upsample_nearest(&x_in, &g_in, factor, false).0
                }
                // `DupUp3D(1, 2, 2)` at half the width: channel `2c + b`
                // to `h`-subposition `b` of channel `c`, then `w` doubled.
                Shortcut::ShuffleH => {
                    let (s, gs) = spatial::pixel_shuffle(&x_in, &g_in, [1, 2, 1]);
                    spatial::upsample_nearest(&s, &gs, [1, 1, 2], false).0
                }
            };
            x = ops::elemwise::add(&x, &s);
        }
    }

    let x = norm_silu(&x, &vae.norm_out);
    let (y, gy) = conv(&x, &g, &vae.conv_out, arm);
    // The 2×2 space-to-depth the VAE wraps its conv stack in, undone:
    // `conv_out`'s rows are permuted at import into the shuffle's `(c, ph,
    // pw)` order (the checkpoint's is `(c, pw, ph)`).
    let (pixels, gp) = spatial::pixel_shuffle(&y, &gy, [1, VAE_PATCH, VAE_PATCH]);
    // diffusers clamps regardless of `clip_output`.
    let pixels = ops::elemwise::clamp(&pixels, -1.0, 1.0);
    seam::at(seam::PIXELS, &[&pixels, &gp]);
}
