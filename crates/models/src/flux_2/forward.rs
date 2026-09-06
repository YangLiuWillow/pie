//! FLUX.2's traced arithmetic: three arms of one plan, selected per lane by
//! the reading bits of the fact word (design D1, D5).
//!
//! | reading | lanes (stream) | binds | reads back |
//! |---|---|---|---|
//! | `text` | one, `Text` | `embed(ids)`, `attention(kv)` | `hidden` `[L, dim]`: the layer-{9,18,27} stack, through `context_embedder` |
//! | `denoise` | `Text` + `Image` (+ `Reference`), one group | text: `context`, `positions`, `timestep` (+ `guidance`); image, reference: `latents`, `positions`, `timestep` (+ `guidance`) | `velocity` `[N, 128]` on the image lane |
//! | `vae.decode` | one, `Image`, one clip `{1, h, w}` | `latent` `[h·w, 128]` on the voxel axis | `pixels` `[16h·16w, 3]` in `[-1, 1]` |
//! | `vae.encode` | one, `Image`, one clip `{1, H, W}` | `pixels` `[H·W, 3]` on the voxel axis | `pixels` `[H/16·W/16, 128]`: the normalised posterior MEAN |
//!
//! The two VAE readings ([`super::vae`]) exist on the flagship only (the
//! miniature's checkpoint is the transformer alone). Each runs ONE clip a
//! lane, its `Voxels` port's channel is `[h, w, C]`, and it reads its
//! pixels back off the `pixels` seam beside the output grid. The port is
//! the DiT's own 128-wide `/16` grid on both arms — the 2×2 pixel shuffle
//! and the frozen BatchNorm live INSIDE the plan, so a guest hands the
//! denoiser's rows over and gets the denoiser's rows back ([`super::vae`]
//! states why that is the boundary).
//!
//! **Sequence layout.** The joint attention packs a group's lanes by
//! stream code — Text (0), Image (1), Reference (5) — which is the
//! reference's own `[txt ‖ target ‖ refs]` (study §E.1), unmasked: the
//! `dev`/diffusers layout, which klein-4B's `Flux2KleinPipeline` runs
//! (`RaggedMask::GroupBlockDiagonal`). The KV-cached `[txt ‖ refs ‖ img]`
//! layout with refs self-attending only is `klein-9b-kv`'s and is not this
//! row's (study §I.2, §L.1); `ReferenceSelfOnly` is one enum away when it
//! is. A reference lane carries every reference's tokens concatenated
//! (`cat(refs)`), at rotary `T = 10·(i+1)`; it runs the image side's
//! weights (it IS image tokens) and reads out nothing — the head runs on
//! the target lane alone (`pred[:, :S_img]`).
//!
//! **Positions** (`AxisPositions`, `[rows, 4]` f32, `(T, H, W, L)`): text
//! row `j` is `(0, 0, 0, j)`; target token `(h, w)` is `(0, h, w, 0)`;
//! reference `i`'s token `(h, w)` is `(10·(i+1), h, w, 0)` — `_prepare_
//! {text,latent,image}_ids`.
//!
//! **Timestep and guidance.** The `timestep` port takes the SCHEDULER
//! timestep `σ·1000` (the reference's `timestep · 1000`); `guidance` takes
//! the raw scale (`4.0`) and the plan multiplies by 1000 before the
//! sinusoid, as the reference does. Both are `[Lanes, 1]` and are bound by
//! EVERY lane of the reading: the three shared modulation vectors are per
//! lane, and each stream's class applies its own.
//!
//! **Text conditioning.** The reference stacks `hidden_states[9|18|27]`
//! of Qwen3 on the channel axis (`[L, 7680]`) and `context_embedder`
//! projects it to `dim`. This IR has no column concatenation and one
//! `hidden` readout per reading, so the `text` arm folds the embedder in:
//! `W·cat(h9, h18, h27) = W0·h9 + W1·h18 + W2·h27` with `W = [W0 | W1 |
//! W2]` cut at import — the same numbers, two more bf16 roundings — and
//! exports `[L, dim]`. The miniature has no encoder and takes the raw
//! `[L, joint_attention_dim]` stack on its `context` port, embedding it in
//! the `denoise` arm (which is what `flux2_golden.py --mini` feeds).
//!
//! **The text encoder runs the prompt UNPADDED.** The reference pads every
//! prompt to 512 with `<|endoftext|>` and masks the pad keys; the pad
//! rows still enter the DiT as text tokens. A guest may pad the same way
//! (each pad row then attends its predecessors, pads included, since a
//! prefill has no key mask), or hand the native length; the family states
//! the truncation bound (`TE_MAX_TOKENS`), not a pad target.
//!
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    AxisRole, Generative, LatentSpace, PortFact, PortKind, PositionConvention, ReadingFact,
    ReadoutKind, ScheduleFact, ScheduleKind,
};

use super::model::{
    Attn, DOUBLE_MOD_SLICES, Dit, Embedder, GUIDANCE_SCALE, HEAD_DIM, IN_CHANNELS, Model, NORM_EPS,
    ROPE_AXES, ROPE_DIMS, ROPE_THETA, SINGLE_MOD_SLICES, SM_SCALE, Swiglu, T_FLIP_SIN_COS,
    T_FREQ_DIM, T_MAX_PERIOD, T_SCALE, TE_LAYERS, TE_MAX_TOKENS, TE_TAPS, TOKEN_COMPRESSION,
    TRAIN_STEPS, TextEncoder, port,
};

/// The bit the one-hot stream facts start at (D2): bits 0..6 are the six
/// streams, of which this text names Text, Image, Context and Reference.
pub const STREAM_BASE: u8 = 0;

/// The two bits the reading index lives in, as a plain binary code: bit
/// [`READING_LO`] is its low bit, [`READING_HI`] its high bit. Four codes:
/// `text`, `denoise`, `vae.decode`, `vae.encode`.
pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;

/// Which reading code means what, per row. The word packs the index the
/// runtime stamps (`Request::reading`) and nothing else — `Classify::of`
/// has no model to ask — so the *meaning* of a code is the row's: the
/// flagship runs `text` at 0, the miniature (no encoder) runs `denoise`
/// at 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    pub text: Option<u8>,
    pub denoise: u8,
    /// The two VAE arms' codes, on a row that carries a VAE.
    pub vae_decode: Option<u8>,
    pub vae_encode: Option<u8>,
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
        let vae_decode = self.vae.as_ref().map(|_| take());
        let vae_encode = self.vae.as_ref().map(|_| take());
        Readings {
            text,
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
        if let (Some(index), Some(_)) = (codes.text, &self.te) {
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
                readout_width: d.dim,
            });
        }
        let image_side = [Stream::Image, Stream::Reference];
        let every = [Stream::Text, Stream::Image, Stream::Reference];
        // Port ORDER is load-bearing: a port's index is its position among
        // its kind, and `model::port` numbers them so — `timestep` before
        // `guidance` is what makes them lane vectors 0 and 1.
        let mut ports = vec![
            port("latents", PortKind::Latents, IN_CHANNELS, &image_side),
            port(
                "context",
                PortKind::Context,
                if self.te.is_some() {
                    d.dim
                } else {
                    d.context_in
                },
                &[Stream::Text],
            ),
            port("timestep", PortKind::LaneVector, 1, &every),
        ];
        if d.guidance_embeds {
            ports.push(port("guidance", PortKind::LaneVector, 1, &every));
        }
        ports.push(port(
            "positions",
            PortKind::AxisPositions,
            u32::from(ROPE_AXES),
            &every,
        ));
        readings.push(ReadingFact {
            name: "denoise",
            index: codes.denoise,
            has_kv: false,
            takes_tokens: false,
            streams: every.to_vec(),
            ports,
            // `(T, H, W, L)`: the target grid on `(h, w)` at `T = 0`, a
            // text row `j` at `(0, 0, 0, j)` — `_prepare_{text,latent}_ids`.
            // (A reference lane's `T = 10·(i + 1)` is the family's, not the
            // convention's; a guest that binds references states it.)
            positions: Some(PositionConvention {
                axes: vec![
                    AxisRole::Time,
                    AxisRole::Height,
                    AxisRole::Width,
                    AxisRole::Index,
                ],
                text_axis: 3,
                text_origin: 0,
                image_follows_text: false,
            }),
            readout: ReadoutKind::Velocity,
            readout_width: IN_CHANNELS,
        });
        // The two voxel readings, on a row that carries the autoencoder.
        // Both put the port at the DiT's own 128-wide `/16` grid
        // ([`super::vae`]); the readout is the other side of the same
        // `pixels` seam.
        if let (Some(decode), Some(encode), Some(_)) =
            (codes.vae_decode, codes.vae_encode, &self.vae)
        {
            readings.push(ReadingFact {
                name: "vae.decode",
                index: decode,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Image],
                ports: vec![port(
                    "latent",
                    PortKind::Voxels,
                    IN_CHANNELS,
                    &[Stream::Image],
                )],
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
                // packed latent clip is 128 wide at index 0.
                ports: vec![PortFact {
                    name: "pixels",
                    kind: PortKind::Voxels,
                    width: super::vae::RGB,
                    streams: vec![Stream::Image],
                    at: Some(port::PIXEL_VOXELS),
                    rows: None,
                }],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: IN_CHANNELS,
            });
        }
        Generative {
            readings,
            // Stated as the denoiser holds it: a token is 128 channels at
            // /16, one cell, no further patching. The VAE's own 32 channels
            // at /8 are the VAE arms' business (`model::VAE_CHANNELS`,
            // `model::PACK`) and never leave them.
            latent: Some(LatentSpace {
                channels: IN_CHANNELS,
                patch_t: 1,
                patch_h: 1,
                patch_w: 1,
                spatial_compression: TOKEN_COMPRESSION,
                temporal_compression: 1,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                // The exponential shift at the flagship's default job: the
                // empirical mu at 1024² (4096 tokens), four steps. Stated
                // beside the pinned sigmas it produced, for the record; a
                // guest at another size or step count wants
                // [`sigmas`] and not this number.
                shift: empirical_mu(4096, 4).exp(),
                train_steps: TRAIN_STEPS,
                boundary: None,
                // klein is distilled to four steps; these are its sigmas at
                // 1024² (the golden's `sigmas[:-1]`).
                pinned_sigmas: sigmas(4096, 4),
                // One backbone, one schedule: every lane takes `shift`.
                stream_shifts: vec![],
            }),
            // 1024² target + four 1024² references (the API's klein cap) +
            // the 512-token prompt; the miniature's job is 64 + 128 + 32.
            max_rows: match self.te {
                Some(_) => 5 * 4096 + TE_MAX_TOKENS,
                None => 4096,
            },
        }
    }
}

/// `pipeline_flux2_klein.py::compute_empirical_mu`: the exponential
/// time-shift's `mu`, fit empirically by BFL over the TARGET token count
/// (references excluded) and the step count (study §F).
#[must_use]
pub fn empirical_mu(image_rows: u32, steps: u32) -> f32 {
    const A1: f64 = 8.738_095_24e-5;
    const B1: f64 = 1.898_333_33;
    const A2: f64 = 0.000_169_27;
    const B2: f64 = 0.456_666_66;
    let rows = f64::from(image_rows);
    if image_rows > 4300 {
        return (A2 * rows + B2) as f32;
    }
    let m_200 = A2 * rows + B2;
    let m_10 = A1 * rows + B1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    (a * f64::from(steps) + b) as f32
}

/// The diffusers sigma grid for `steps` steps at `image_rows` target
/// tokens: `linspace(1, 1/steps, steps)` through the exponential shift
/// `σ' = e^mu / (e^mu + 1/σ − 1)`, descending, WITHOUT the trailing zero
/// the scheduler appends. `sigmas(4096, 4)` is the golden's.
#[must_use]
pub fn sigmas(image_rows: u32, steps: u32) -> Vec<f32> {
    let steps = steps.max(1);
    let shift = f64::from(empirical_mu(image_rows, steps)).exp();
    (0..steps)
        .map(|i| {
            let n = f64::from(steps);
            let sigma = 1.0 - f64::from(i) * (1.0 - 1.0 / n) / (n - 1.0).max(1.0);
            (shift / (shift + 1.0 / sigma - 1.0)) as f32
        })
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
    pub fn reference() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Reference)
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
        let velocity = denoise(arm(codes.denoise), self);
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
/// the residual leaving layers 9, 18 and 27 (`hidden_states[k]`) each
/// through its column block of `context_embedder` and summed; `hidden`
/// planted on the sum inside the last tap's layer mark. No final norm, no
/// head, nothing past layer 27.
fn text_encode(arm: &Input<Facts>, te: &TextEncoder) {
    let plan = ops::attn::plan_prefill(arm, te.q_heads, te.kv_heads, te.head_dim, None);
    let ids = arm.tokens();
    let positions = arm.positions();
    let mut y = ops::layout::embed(&ids, &te.embed, te.vocab);
    let mut ctx: Option<Value> = None;
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

        // `hidden_states[k]` is the residual leaving layer `k − 1`.
        if let Some(tap) = TE_TAPS.iter().position(|&k| k == l + 1) {
            let part = ops::linear::matmul(&y, &te.context_embed[tap]);
            let sum = match ctx.take() {
                None => part,
                Some(acc) => ops::elemwise::residual_add(&part, &acc),
            };
            if tap + 1 == TE_TAPS.len() {
                seam::at(seam::HIDDEN, &[&sum]);
            }
            ctx = Some(sum);
        }
    }
}

/// The three things a modulated sublayer applies: the `[scale | shift]`
/// pair and the gate — the plan's slice order, which `import.rs` makes.
struct Mod {
    scale_shift: Value,
    gate: Value,
}

/// A double-stream side's two sets: attention, then MLP.
struct DoubleMod {
    attn: Mod,
    mlp: Mod,
}

fn adaln6(m: &Value, dim: u32) -> DoubleMod {
    debug_assert_eq!(m.width(), u64::from(DOUBLE_MOD_SLICES * dim));
    let (a_ss, rest) = ops::layout::split_rows(m, 2 * dim);
    let (a_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (m_ss, m_gate) = ops::layout::split_rows(&rest, 2 * dim);
    DoubleMod {
        attn: Mod {
            scale_shift: a_ss,
            gate: a_gate,
        },
        mlp: Mod {
            scale_shift: m_ss,
            gate: m_gate,
        },
    }
}

fn adaln3(m: &Value, dim: u32) -> Mod {
    debug_assert_eq!(m.width(), u64::from(SINGLE_MOD_SLICES * dim));
    let (scale_shift, gate) = ops::layout::split_rows(m, 2 * dim);
    Mod { scale_shift, gate }
}

/// The tables one joint attention needs: the arm's row permutation and
/// the group CSR its segments pair by.
struct Joint {
    perm: Value,
    csr: Value,
}

/// `TimestepEmbedding`: `linear_2(silu(linear_1(x)))`, a lane-shaped f32
/// chain.
fn embed(e: &Embedder, x: &Value) -> Value {
    let h = ops::elemwise::silu(&ops::linear::matmul(x, &e.linear_1));
    ops::linear::matmul(&h, &e.linear_2)
}

/// `LayerNorm(affine=False)` then `x·(1+scale)+shift`, per lane.
fn norm_modulate(x: &Value, scale_shift: &Value, lanes: &Value) -> Value {
    ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(x, NORM_EPS),
        scale_shift,
        Some(lanes),
        ModulateForm::ScaleShift,
    )
}

/// Per-head QK RMSNorm and the four-axis interleaved rope.
fn turn(x: &Value, gain: &Weight, positions: &Value) -> Value {
    ops::elemwise::rope_axes(
        &ops::elemwise::rmsnorm_per_head(x, gain, HEAD_DIM, NORM_EPS),
        positions,
        ROPE_DIMS,
        [ROPE_THETA; 4],
        RopeForm::Interleaved,
        HEAD_DIM,
        HEAD_DIM,
    )
}

/// One side's queries, keys and values: modulate, project, QK-norm, rope.
fn heads(
    x: &Value,
    attn: &Attn,
    m: &Mod,
    dim: u32,
    lanes: &Value,
    positions: &Value,
) -> (Value, Value, Value) {
    let h = norm_modulate(x, &m.scale_shift, lanes);
    let (q, k, v) = ops::layout::split_qkv(&ops::linear::matmul(&h, &attn.qkv), dim, dim);
    (
        turn(&q, &attn.q_norm, positions),
        turn(&k, &attn.k_norm, positions),
        v,
    )
}

/// The joint attention itself: pack by the arm's row order, one ragged
/// read over the group CSR, unpack back onto the fire's rows.
fn joint_attention(q: &Value, k: &Value, v: &Value, j: &Joint) -> Value {
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(q, &j.perm),
        &ops::layout::pack_rows(k, &j.perm),
        &ops::layout::pack_rows(v, &j.perm),
        &j.csr,
        &j.csr,
        HEAD_DIM,
        SM_SCALE,
        RaggedMask::GroupBlockDiagonal,
    );
    ops::layout::unpack_rows(&o, &j.perm)
}

/// `x += gate · linear_out(swiglu(linear_in(mod(x))))`.
fn ff_sublayer(x: &Value, ff: &Swiglu, m: &Mod, inter: u32, lanes: &Value) -> Value {
    let h = norm_modulate(x, &m.scale_shift, lanes);
    let h = ops::linear::mlp_swiglu(&ops::linear::matmul(&h, &ff.linear_in), inter);
    ops::elemwise::gated_residual_add(
        x,
        &m.gate,
        &ops::linear::matmul(&h, &ff.linear_out),
        Some(lanes),
    )
}

/// The `denoise` reading.
fn denoise(arm: &Input<Facts>, m: &Model) -> Value {
    let d = &m.dims;
    let dit: &Dit = &m.dit;
    let dim = d.dim;

    // The text lane on one side; the image and reference lanes — one
    // rectangle of image tokens, one set of weights — on the other.
    let (txt_in, img_in) = arm.split(&Facts::text());

    // Reading-wide tables, read once under the reading's guard. The joint
    // attention packs the whole group, so its permutation and CSR are the
    // arm's own (`IMAGEGEN_CONTRACT.md` §7, the pack_rows window rule).
    let lanes = arm.request_of_token();
    let positions = arm.axis_positions(port::POSITIONS, ROPE_AXES);
    let joint = Joint {
        perm: arm.row_permutation(),
        csr: arm.group_indptr(),
    };

    // ---- the conditioning vector, once per lane ---------------------------
    // `[cos | sin]` sinusoid of the scheduler timestep, the two-layer MLP;
    // with `guidance_embeds`, the same over `guidance · 1000`, added.
    let t = arm.lane_vector(port::TIMESTEP, 1);
    let temb = embed(
        &dit.t_embed,
        &ops::elemwise::sinusoid(&t, T_FREQ_DIM, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE),
    );
    let temb = match &dit.g_embed {
        Some(g_embed) => {
            let g = arm.lane_vector(port::GUIDANCE, 1);
            let gemb = embed(
                g_embed,
                &ops::elemwise::sinusoid(
                    &g,
                    T_FREQ_DIM,
                    T_MAX_PERIOD,
                    T_FLIP_SIN_COS,
                    GUIDANCE_SCALE,
                ),
            );
            ops::elemwise::add(&temb, &gemb)
        }
        None => temb,
    };
    // Every consumer of `temb` reads it through a SiLU (the three
    // modulation linears and `norm_out`), so it is taken once, in place.
    let stemb = ops::elemwise::silu(&temb);
    // A node's operands come from one arm (`record.rs`), so each stream's
    // vectors are split onto its arm before the blocks read them: the text
    // side's onto the text lanes, the image side's onto image AND
    // reference lanes, the single blocks' onto the whole reading.
    let (mod_txt, _) = ops::linear::matmul(&stemb, &dit.mod_txt).split(&Facts::text());
    let (_, mod_img) = ops::linear::matmul(&stemb, &dit.mod_img).split(&Facts::text());
    let mod_txt = adaln6(&mod_txt, dim);
    let mod_img = adaln6(&mod_img, dim);
    let mod_single = adaln3(&ops::linear::matmul(&stemb, &dit.mod_single), dim);
    // `AdaLayerNormContinuous`: `[scale | shift]`, the plan's own order —
    // read by the head on the target lane alone.
    let mod_out = ops::linear::matmul(&stemb, &dit.norm_out);
    let (lanes_txt, lanes_img) = lanes.split(&Facts::text());
    let (pos_txt, pos_img) = positions.split(&Facts::text());

    // ---- the two streams' rows --------------------------------------------
    let mut txt = match &dit.context_embed {
        Some(w) => ops::linear::matmul(&txt_in.context(port::CONTEXT, d.context_in), w),
        None => {
            // Already embedded by the `text` arm. The first block folds its
            // residual IN PLACE on the text stream, and a port's cell is not
            // an arena rectangle to fold into (`arena::fold_in_place`), so
            // the rows are landed first: packed by the text arm's own
            // permutation and unpacked straight back, the one exact copy
            // of a token rectangle this IR has.
            let c = txt_in.context(port::CONTEXT, dim);
            let perm = txt_in.row_permutation();
            ops::layout::unpack_rows(&ops::layout::pack_rows(&c, &perm), &perm)
        }
    };
    let mut img = ops::linear::matmul(
        &img_in.latents(port::LATENTS, IN_CHANNELS, Dtype::Bf16),
        &dit.x_embed,
    );

    // ---- the double-stream blocks -----------------------------------------
    // Each stream brings its own projections and MLP under the shared
    // modulation of its side; only the softmax is joint.
    for (_, block) in arm.walk_layers(&dit.double) {
        let (tq, tk, tv) = heads(
            &txt,
            &block.txt.attn,
            &mod_txt.attn,
            dim,
            &lanes_txt,
            &pos_txt,
        );
        let (iq, ik, iv) = heads(
            &img,
            &block.img.attn,
            &mod_img.attn,
            dim,
            &lanes_img,
            &pos_img,
        );
        let o = joint_attention(
            &Value::merge(vec![tq, iq]),
            &Value::merge(vec![tk, ik]),
            &Value::merge(vec![tv, iv]),
            &joint,
        );
        let (o_txt, o_img) = o.split(&Facts::text());
        txt = ops::elemwise::gated_residual_add(
            &txt,
            &mod_txt.attn.gate,
            &ops::linear::matmul(&o_txt, &block.txt.attn.out),
            Some(&lanes_txt),
        );
        img = ops::elemwise::gated_residual_add(
            &img,
            &mod_img.attn.gate,
            &ops::linear::matmul(&o_img, &block.img.attn.out),
            Some(&lanes_img),
        );
        txt = ff_sublayer(&txt, &block.txt.ff, &mod_txt.mlp, d.inter, &lanes_txt);
        img = ff_sublayer(&img, &block.img.ff, &mod_img.mlp, d.inter, &lanes_img);
    }

    // ---- the single-stream blocks over `[txt ‖ img ‖ refs]` ---------------
    // The parallel block: one in-projection lands `[q | k | v | gate | up]`,
    // attention and SwiGLU run side by side, and the out-projection over
    // `[attn | mlp]` is its two column blocks summed — one gate for both.
    let mut x = Value::merge(vec![txt, img]);
    for (_, block) in arm.walk_layers(&dit.single) {
        let h = norm_modulate(&x, &mod_single.scale_shift, &lanes);
        let proj = ops::linear::matmul(&h, &block.in_proj);
        let (qkv, mlp) = ops::layout::split_rows(&proj, 3 * dim);
        let (q, k, v) = ops::layout::split_qkv(&qkv, dim, dim);
        let o = joint_attention(
            &turn(&q, &block.q_norm, &positions),
            &turn(&k, &block.k_norm, &positions),
            &v,
            &joint,
        );
        let a = ops::linear::matmul(&o, &block.out_attn);
        let f = ops::linear::matmul(&ops::linear::mlp_swiglu(&mlp, d.inter), &block.out_mlp);
        x = ops::elemwise::gated_residual_add(
            &x,
            &mod_single.gate,
            &ops::elemwise::residual_add(&a, &f),
            Some(&lanes),
        );
    }

    // ---- the head, target rows only ---------------------------------------
    // `pred[:, :S_img]`: the text rows are dropped and the reference rows'
    // predictions are never computed (the reference discards them).
    let (_, img_all) = x.split(&Facts::text());
    let (target, _refs) = img_all.split(&Facts::image());
    let (mod_out, _) = mod_out.split(&Facts::text()).1.split(&Facts::image());
    let (lanes_target, _) = lanes_img.split(&Facts::image());
    let h = norm_modulate(&target, &mod_out, &lanes_target);
    let velocity = ops::linear::matmul(&h, &dit.proj_out);
    seam::at(seam::VELOCITY, &[&velocity]);
    velocity
}
