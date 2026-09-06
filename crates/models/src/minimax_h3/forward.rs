//! MiniMax H3's traced arithmetic: three readings of one plan, selected
//! per lane by the reading bits of the fact word (design D1, D5).
//!
//! | reading | lanes (stream) | binds | reads back |
//! |---|---|---|---|
//! | `text` | one, `Text` | `embed(ids)`, `attention(kv)` | `hidden` `[L, 5120]`: Qwen3-VL-32B's residual leaving layer 49 |
//! | `refine` | one, `Text` | `caption` `[L, 5120]` | `hidden` `[L, 5376]`: `condition_proj` + the two token-refiner blocks + `final_norm` |
//! | `denoise` | `Text` + `Video` + `Audio` + `Reference`, ONE group | text: `context`; video: `latents`; audio: `audio`; reference: `reference`; every lane: `timestep` `[4]`, `positions` `[3]` | `velocity` `[N, 96]` on the video lane, `hidden` `[2A, 32]` on the audio lane |
//!
//! # One packed row sequence, four lanes, one attention
//!
//! The reference builds ONE row sequence `[text | keyframes/refs | audio |
//! video | pad]` and reads it with a two-segment `cu_seqlens = [0, used,
//! seq_len]` so the 64-row alignment pad is a second document that never
//! attends the real rows (study §C.1, §E). This text submits the same rows
//! as FOUR lanes of one attention group — `Stream::Text` (the refined
//! conditioning), `Stream::Reference` (the clean, noise-augmented keyframe
//! and reference latent rows), `Stream::Audio`, `Stream::Video` — and the
//! joint attention is `attn::ragged` over the group's CSR
//! (`RaggedMask::GroupBlockDiagonal`). Two consequences, both intended:
//!
//! * **there is no pad.** A lane carries exactly its rows, so the padding
//!   document has nothing to hold and the second `cu_seqlens` segment
//!   disappears. The answer on the real rows is the reference's, since a
//!   pad row was never attended;
//! * **the packed order is the lanes' order, not the reference's.** Rows
//!   pack by stream code — Text (0), Video (2), Audio (3), Reference (5) —
//!   which is a permutation of `[text | refs | audio | video]`. The
//!   attention is unmasked inside the group and every row's rotary
//!   coordinates travel with it (`positions`), so the two orders answer
//!   identically; only the row order of the readback differs, and each
//!   lane reads back its own rows.
//!
//! # Positions (`AxisPositions`, `[rows, 3]` f32, `(t, h, w)`)
//!
//! Guest data, as they are for every family here — and this family states
//! **no `PositionConvention`**: H3's `t` axis is a shared timeline in
//! 1/40 s audio ticks along which text is a 1-D prefix (`t = i`, `h = w =
//! 0`), a video patch of latent frame `k` sits at `text_len + Σ_{j<k}
//! 5/3·FRAME_PER_TOKEN[j mod 5]` with aspect-normalised `h`/`w` on a
//! `[0, 32)` grid, an audio latent at tick `t` sits at `text_len + t` with
//! `h = 0` and `w` pinned to the LEFT or RIGHT extreme of that grid (the
//! stereo channel), and reference blocks run at their own advancing
//! cursor (study §C.2). None of that is the two-axis image convention the
//! fact vocabulary states, so `positions: None` and the guest owns the
//! table — which is exactly where the reference puts it
//! (`packed_sequence.py`, host fp64).
//!
//! # The timestep is a `[lanes, 4]` vector and the modulation is a gather
//!
//! Up to four distinct timesteps run in one step (video, audio, the
//! visual condition rows' `max(t_video, 0.999)`, ref2va's audio
//! references' `1.0`) and every row picks its modulation by `combined =
//! 3·timestep_index + modality_tag`. Both indices are constant over a
//! lane, so the `timestep` port carries the step's whole
//! `unique_timesteps[≤4]` and each stream's arm slices ITS column
//! ([`super::model::timestep_slot`]) and multiplies ITS third of the
//! adaLN bank ([`super::model::modality`]); the four `[Lanes_s, 6·dim]`
//! answers merge back into one `[Lanes, 6·dim]` rectangle and the fifty
//! blocks run on one arm with one weight set. See
//! [`super::model`]'s header for why that is the whole gather and why it
//! needs no new op.
//!
//! # The two autoencoders are not here
//!
//! Neither `vae.decode` nor `vae.encode` is declared, for two different
//! reasons.
//!
//! The **video VAE** — a causal 3-D CNN encoder (f16 t4 c24) paired with a
//! 36-block ViT decoder whose every token emits a 4×16×16 pixel block —
//! is within the `Spatial` vocabulary as it stands: the encoder is the
//! causal `Conv3d` + `GroupNorm` ladder `IMAGEGEN_CONTRACT.md` §6 already
//! serves, and the decoder is `spatial::patchify` at `p = (1, 1, 1)` off
//! the latent, `attn::ragged` over the clip's tokens under a three-axis
//! rope (θ 100, angles pre-multiplied by 2π — `use_angle`, which is the
//! guest's position table's business), and `spatial::unpatchify` at
//! `p = (4, 16, 16)`. What stops it is the CHECKPOINT: the partition
//! keeps its weights at `video_vae/source/model.safetensors`, a nested
//! folder `checkpoint::file::diffusers::weight_files` does not descend
//! into (design D5's subfolder recursion is unbuilt), and the file had
//! not finished downloading while this text was written. So the reading
//! is a follow-up with a known shape, not an open question.
//!
//! The **audio VAE** — a DAC encoder with a BigVGAN vocoder — is not:
//! seven `ConvTranspose1d` upsamplers (×800 together), weight-norm
//! reparameterisation (`g·v/‖v‖`), `Snake`/`SnakeBeta`
//! (`x + α⁻¹·sin²(αx)`) and anti-aliased sinc up/down-sampling are four
//! ops the `Spatial` family does not state. `Elementwise::Snake` is the
//! small one; the transposed convolution is the real work.

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    Generative, LatentSpace, PortFact, PortKind, ReadingFact, ReadoutKind, ScheduleFact,
    ScheduleKind,
};

use super::model::{
    ADALN_SLICES, AUDIO_CHANNELS, AUDIO_SHIFT, Attn, Block, CONDITION_TIMESTEP, Dims, Dit,
    FINAL_SLICES, Linear, Mlp, Model, NORM_EPS, PATCH_H, PATCH_T, PATCH_W, ROPE_AXES, ROPE_THETA,
    Refiner, SPATIAL_COMPRESSION, STEPS, T_FLIP_SIN_COS, T_MAX_PERIOD, T_SCALE, TE_HIDDEN,
    TE_LAYERS, TE_MAX_TOKENS, TEMPORAL_COMPRESSION, TIMESTEP_SLOTS, TRAIN_STEPS, TextEncoder,
    VIDEO_FEATURES, VIDEO_SHIFT, modality, port, timestep_slot,
};

/// The bit the one-hot stream facts start at (D2): bits 0..6 are the six
/// streams, of which this text names Text, Video, Audio and Reference.
pub const STREAM_BASE: u8 = 0;

/// The two bits the reading index lives in, as a plain binary code. Four
/// codes; three readings on the flagship, and the fourth is where a
/// `vae.decode` arm lands when the `Spatial` vocabulary can hold one.
pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;

/// Which reading code means what, per row. The word packs the index the
/// runtime stamps (`Request::reading`) and nothing else — `Classify::of`
/// has no model to ask — so the *meaning* of a code is the row's: the
/// flagship runs `text` at 0, the miniature (no encoder) runs `refine`
/// at 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    pub text: Option<u8>,
    pub refine: u8,
    pub denoise: u8,
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
        let refine = take();
        let denoise = take();
        Readings {
            text,
            refine,
            denoise,
        }
    }

    /// This row's generative facts (design D12).
    #[must_use]
    pub fn generative(&self) -> Generative {
        let d = &self.dims;
        let codes = self.readings();
        let port = |name, kind, at, width, streams: &[Stream]| PortFact {
            name,
            kind,
            width,
            streams: streams.to_vec(),
            at: Some(at),
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
        readings.push(ReadingFact {
            name: "refine",
            index: codes.refine,
            has_kv: false,
            takes_tokens: false,
            streams: vec![Stream::Text],
            ports: vec![port(
                "caption",
                PortKind::Context,
                port::CAPTION,
                d.text_dim,
                &[Stream::Text],
            )],
            positions: None,
            readout: ReadoutKind::Hidden,
            readout_width: d.dim,
        });
        let every = [
            Stream::Text,
            Stream::Video,
            Stream::Audio,
            Stream::Reference,
        ];
        readings.push(ReadingFact {
            name: "denoise",
            index: codes.denoise,
            has_kv: false,
            takes_tokens: false,
            streams: every.to_vec(),
            ports: vec![
                port(
                    "latents",
                    PortKind::Latents,
                    port::LATENTS,
                    VIDEO_FEATURES,
                    &[Stream::Video],
                ),
                port(
                    "reference",
                    PortKind::Latents,
                    port::REFERENCE,
                    VIDEO_FEATURES,
                    &[Stream::Reference],
                ),
                port(
                    "audio",
                    PortKind::Latents,
                    port::AUDIO,
                    AUDIO_CHANNELS,
                    &[Stream::Audio],
                ),
                port(
                    "context",
                    PortKind::Latents,
                    port::CONTEXT,
                    d.dim,
                    &[Stream::Text],
                ),
                port(
                    "timestep",
                    PortKind::LaneVector,
                    port::TIMESTEP,
                    TIMESTEP_SLOTS,
                    &every,
                ),
                port(
                    "positions",
                    PortKind::AxisPositions,
                    port::POSITIONS,
                    u32::from(ROPE_AXES),
                    &every,
                ),
            ],
            // See the module header: H3's `t` axis is a shared audio-tick
            // timeline with a stereo-pinned `w`, which the convention
            // vocabulary does not state.
            positions: None,
            readout: ReadoutKind::Velocity,
            readout_width: VIDEO_FEATURES,
        });
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: super::model::LATENT_CHANNELS,
                patch_t: PATCH_T,
                patch_h: PATCH_H,
                patch_w: PATCH_W,
                spatial_compression: SPATIAL_COMPRESSION,
                temporal_compression: TEMPORAL_COMPRESSION,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: VIDEO_SHIFT,
                train_steps: TRAIN_STEPS,
                boundary: None,
                // `unique_consecutive` over `linspace(1, 0, 50)` through
                // the shift keeps all fifty points, and the loop runs the
                // first forty-nine: the sigma list is the schedule, and
                // the family is CFG-distilled against it.
                pinned_sigmas: shifted_sigmas(VIDEO_SHIFT, STEPS),
                stream_shifts: vec![
                    (Stream::Video, VIDEO_SHIFT),
                    (Stream::Audio, AUDIO_SHIFT),
                    // The condition rows do not follow a schedule at all —
                    // they are pinned — but a guest sizing its four
                    // timestep slots wants the number, and 1.0 is "no
                    // shift" for the one it never advances.
                    (Stream::Reference, 1.0),
                ],
            }),
            // 1344×768 at 15 s is 107 856 video rows plus 1206 audio rows
            // plus the prompt (study §D.2); the miniature's job is a few
            // hundred.
            max_rows: match self.te {
                Some(_) => 131_072,
                None => 4096,
            },
        }
    }
}

/// The flow schedule's sigma grid at one shift: `base = linspace(1, 0, n)`
/// through `σ = s·base / (1 + (s − 1)·base)`, descending, WITHOUT the
/// trailing zero (`time_request.py:32-59`). The video grid at `s = 12` and
/// the audio grid at `s = 3` have the same length and are consumed in
/// lock-step, one Euler step per modality.
#[must_use]
pub fn shifted_sigmas(shift: f32, steps: u32) -> Vec<f32> {
    let n = steps.max(2);
    (0..n - 1)
        .map(|i| {
            let base = 1.0 - f64::from(i) / f64::from(n - 1);
            let s = f64::from(shift);
            (s * base / (1.0 + (s - 1.0) * base)) as f32
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
    pub fn video() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Video)
    }

    #[must_use]
    pub fn audio() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Audio)
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
        // literals (so every arm names a `Selection` the host can pack);
        // a code no reading claims runs no node.
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (c3, c2) = hi.split(&Facts::reading_lo());
        let (c1, c0) = lo.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3];
        let arm = |code: u8| &arms[usize::from(code)];

        if let (Some(code), Some(te)) = (codes.text, &self.te) {
            text_encode(arm(code), te);
        }
        refine(arm(codes.refine), &self.dims, &self.dit);
        denoise(arm(codes.denoise), &self.dims, &self.dit)
    }
}

/// The `text` reading: Qwen3-VL-32B's language half, prefill only, causal
/// over the paged kv, `hidden` planted on the residual leaving layer
/// `TE_LAYERS − 1`. No final norm (the reference replaces it with
/// `nn.Identity`), no head, nothing past layer 49.
///
/// **The rotary is plain neox at `rope_theta`.** The shipped encoder is
/// M-RoPE with interleaved sections `[24, 20, 20]`; on a TEXT-ONLY
/// sequence all three sections carry the same position, so every channel
/// turns by `pos · θ^(−2i/head_dim)` whatever section it belongs to —
/// which is `rope_full`, angle for angle. An image or video token would
/// break that identity, which is why this row is text-only.
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

/// The tables one attention sublayer reads: the arm's row permutation,
/// the CSR its segments pair by, and the mask.
struct Geom {
    perm: Value,
    csr: Value,
    mask: RaggedMask,
}

/// One biased projection.
fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

/// Per-head QK RMSNorm, then the three-axis neox rope over the rotated
/// prefix (`rope_axes` assigns an ANGLE to an axis and pairs
/// `(p, p + rotary_dim/2)`, which is the reference's
/// `cat(t·f, h·f, w·f, t·f, h·f, w·f)` cache with `rotate_half`,
/// channel for channel).
fn turn(x: &Value, gain: &Weight, positions: &Value, d: &Dims) -> Value {
    ops::elemwise::rope_axes(
        &ops::elemwise::rmsnorm_per_head(x, gain, d.head_dim, NORM_EPS),
        positions,
        d.rope_dims(),
        [ROPE_THETA; 4],
        RopeForm::Neox,
        d.rotary_dim(),
        d.head_dim,
    )
}

/// The attention body shared by the trunk and the token refiner: fused
/// `qkv`, per-head QK RMSNorm, the rope where there is one, one ragged
/// read over the packed segments, `out_proj`.
fn attention(x: &Value, a: &Attn, d: &Dims, g: &Geom, positions: Option<&Value>) -> Value {
    let inner = d.inner();
    let (q, k, v) = ops::layout::split_qkv(&ops::linear::matmul(x, &a.qkv), inner, inner);
    let (q, k) = match positions {
        Some(p) => (turn(&q, &a.q_norm, p, d), turn(&k, &a.k_norm, p, d)),
        None => (
            ops::elemwise::rmsnorm_per_head(&q, &a.q_norm, d.head_dim, NORM_EPS),
            ops::elemwise::rmsnorm_per_head(&k, &a.k_norm, d.head_dim, NORM_EPS),
        ),
    };
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&q, &g.perm),
        &ops::layout::pack_rows(&k, &g.perm),
        &ops::layout::pack_rows(&v, &g.perm),
        &g.csr,
        &g.csr,
        d.head_dim,
        d.sm_scale(),
        g.mask,
    );
    ops::linear::matmul(&ops::layout::unpack_rows(&o, &g.perm), &a.out)
}

/// `fc2(silu(gate) · up)`.
fn mlp(x: &Value, m: &Mlp, d: &Dims) -> Value {
    ops::linear::matmul(
        &ops::linear::mlp_swiglu(&ops::linear::matmul(x, &m.fc1), d.inter),
        &m.fc2,
    )
}

/// The `refine` reading: `condition_proj` over the encoder's rows, the two
/// unmodulated token-refiner blocks (no adaLN, no rope, attention over the
/// lane's own rows), `final_norm`; `hidden` planted on the result inside
/// the last block's layer mark. The reference runs this ONCE per request
/// (`minimax_h3.py:2077-2110`), which is why it is a reading of its own
/// and not a prologue of `denoise`.
fn refine(arm: &Input<Facts>, d: &Dims, m: &Dit) {
    let mut x = linear(&m.condition, &arm.context(port::CAPTION, d.text_dim));
    let geom = Geom {
        perm: arm.row_permutation(),
        csr: arm.lane_indptr(),
        mask: RaggedMask::None,
    };
    let last = m.refine.len() - 1;
    for (l, block) in arm.walk_layers(&m.refine) {
        x = refiner_block(&x, block, d, &geom);
        if l as usize == last {
            let y = ops::elemwise::rmsnorm(&x, &m.refine_norm, NORM_EPS);
            seam::at(seam::HIDDEN, &[&y]);
        }
    }
}

/// `MiniMaxH3TokenRefinerBlock`: pre-norm, no modulation.
fn refiner_block(x: &Value, b: &Refiner, d: &Dims, g: &Geom) -> Value {
    let h = ops::elemwise::rmsnorm(x, &b.norm1, NORM_EPS);
    let x = ops::elemwise::residual_add(&attention(&h, &b.attn, d, g, None), x);
    let h = ops::elemwise::rmsnorm(&x, &b.norm2, NORM_EPS);
    ops::elemwise::residual_add(&mlp(&h, &b.mlp, d), &x)
}

/// The six `[dim]` slices one block's modulation carries, in the plan's
/// own `[scale | shift]` pair order (the checkpoint's is `[shift |
/// scale]`; `import.rs` swaps), so `ModulateForm::ScaleShift` reads a
/// pair whole.
struct Mods {
    attn: Value,
    attn_gate: Value,
    mlp: Value,
    mlp_gate: Value,
}

fn adaln6(m: &Value, dim: u32) -> Mods {
    debug_assert_eq!(m.width(), u64::from(ADALN_SLICES * dim));
    let (attn, rest) = ops::layout::split_rows(m, 2 * dim);
    let (attn_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (mlp, mlp_gate) = ops::layout::split_rows(&rest, 2 * dim);
    Mods {
        attn,
        attn_gate,
        mlp,
        mlp_gate,
    }
}

/// Column `slot` of a `[Lanes, total]` vector.
fn column(v: &Value, slot: u32, total: u32) -> Value {
    debug_assert!(slot < total);
    let tail = if slot == 0 {
        v.clone()
    } else {
        ops::layout::split_rows(v, slot).1
    };
    if slot + 1 == total {
        tail
    } else {
        ops::layout::split_rows(&tail, 1).0
    }
}

/// One stream's arm of the denoise reading: its rows, and the post-SiLU
/// timestep embedding its modulation is projected from.
struct Side {
    arm: Input<Facts>,
    stream: Stream,
    /// `silu(time_embedder(t_slot))` — `adaln_input`, `[Lanes_s, t_dim]`.
    stemb: Value,
}

impl Side {
    /// `adaln_input` through the modality's row block of a block's bank.
    fn modulation(&self, block: &Block) -> Value {
        linear(&block.adaln[modality(self.stream)], &self.stemb)
    }
}

/// The `denoise` reading.
fn denoise(arm: &Input<Facts>, d: &Dims, m: &Dit) -> Value {
    // The four lanes, cut as a COVERING binary tree so their join is the
    // reading's own guard and the merges below come back on this arm
    // (`record.rs::merge`). The last arm is every non-text, non-video,
    // non-audio lane of the reading — the reference lane, which is the
    // only other stream the facts declare.
    let (text, rest) = arm.split(&Facts::text());
    let (video, rest) = rest.split(&Facts::video());
    let (audio, reference) = rest.split(&Facts::audio());

    // Reading-wide tables, read once under the reading's guard. The joint
    // attention packs the whole group, so its permutation and CSR are the
    // arm's own (`IMAGEGEN_CONTRACT.md` §7, the pack_rows window rule).
    let lanes = arm.request_of_token();
    let positions = arm.axis_positions(port::POSITIONS, ROPE_AXES);
    let joint = Geom {
        perm: arm.row_permutation(),
        csr: arm.group_indptr(),
        mask: RaggedMask::GroupBlockDiagonal,
    };

    // ---- the four sides, each with its own timestep column -------------
    let side = |arm: Input<Facts>, stream: Stream| {
        let t = column(
            &arm.lane_vector(port::TIMESTEP, TIMESTEP_SLOTS),
            timestep_slot(stream),
            TIMESTEP_SLOTS,
        );
        // `TimeEmbedder`: a `[cos | sin]` sinusoid at base 10 000, then
        // `proj_out(silu(proj_in(·)))`; `adaln_input` is one more SiLU.
        let e = ops::elemwise::sinusoid(&t, d.t_freq, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE);
        let e = linear(&m.t_out, &ops::elemwise::silu(&linear(&m.t_in, &e)));
        Side {
            arm,
            stream,
            stemb: ops::elemwise::silu(&e),
        }
    };
    let sides = [
        side(text, Stream::Text),
        side(video, Stream::Video),
        side(audio, Stream::Audio),
        side(reference, Stream::Reference),
    ];

    // ---- the rows -------------------------------------------------------
    // The text rows arrive already refined (`refine`'s readout). The first
    // block folds its residual IN PLACE, and a port's cell is not an arena
    // rectangle to fold into, so they are landed first: packed by the text
    // arm's own permutation and unpacked straight back, the one exact copy
    // of a token rectangle this IR has (the `flux_2` precedent).
    let ctx = sides[0].arm.latents(port::CONTEXT, d.dim, Dtype::Bf16);
    let ctx_perm = sides[0].arm.row_permutation();
    let text_rows = ops::layout::unpack_rows(&ops::layout::pack_rows(&ctx, &ctx_perm), &ctx_perm);
    let video_rows = linear(
        &m.video_patch,
        &sides[1]
            .arm
            .latents(port::LATENTS, VIDEO_FEATURES, Dtype::Bf16),
    );
    let audio_rows = linear(
        &m.audio_patch,
        &sides[2]
            .arm
            .latents(port::AUDIO, AUDIO_CHANNELS, Dtype::Bf16),
    );
    // A reference row is a video patch and runs the video projection.
    let reference_rows = linear(
        &m.video_patch,
        &sides[3]
            .arm
            .latents(port::REFERENCE, VIDEO_FEATURES, Dtype::Bf16),
    );
    let mut x = Value::merge(vec![text_rows, video_rows, audio_rows, reference_rows]);

    // ---- the fifty blocks ------------------------------------------------
    for (_, block) in arm.walk_layers(&m.blocks) {
        // The gather: each side's own third of the bank over its own
        // timestep, merged back into one `[Lanes, 6·dim]` rectangle.
        let mods = adaln6(
            &Value::merge(sides.iter().map(|s| s.modulation(block)).collect()),
            d.dim,
        );
        let h = ops::elemwise::modulate(
            &ops::elemwise::rmsnorm(&x, &block.norm1, NORM_EPS),
            &mods.attn,
            Some(&lanes),
            ModulateForm::ScaleShift,
        );
        x = ops::elemwise::gated_residual_add(
            &x,
            &mods.attn_gate,
            &attention(&h, &block.attn, d, &joint, Some(&positions)),
            Some(&lanes),
        );
        let h = ops::elemwise::modulate(
            &ops::elemwise::rmsnorm(&x, &block.norm2, NORM_EPS),
            &mods.mlp,
            Some(&lanes),
            ModulateForm::ScaleShift,
        );
        x = ops::elemwise::gated_residual_add(
            &x,
            &mods.mlp_gate,
            &mlp(&h, &block.mlp, d),
            Some(&lanes),
        );
    }

    // ---- the final layer and the two heads --------------------------------
    // One modality, indexed by the timestep alone: every side projects its
    // own `[scale | shift]` pair from its own `adaln_input`.
    let final_mod = Value::merge(
        sides
            .iter()
            .map(|s| linear(&m.final_adaln, &s.stemb))
            .collect(),
    );
    debug_assert_eq!(final_mod.width(), u64::from(FINAL_SLICES * d.dim));
    let h = ops::elemwise::modulate(
        &ops::elemwise::rmsnorm(&x, &m.final_norm, NORM_EPS),
        &final_mod,
        Some(&lanes),
        ModulateForm::ScaleShift,
    );
    // The reference runs BOTH heads over EVERY row and selects afterwards
    // (`minimax_h3.py:2545-2596`); this text runs each head on the rows
    // that keep its answer, which is the same numbers over fewer rows.
    // Cut the SAME nested tree the lanes were cut with, so each head's
    // guard is exactly its arm's: a head split straight off the reading
    // (`h.split(video)`) would make `text ∧ video` — a word no request can
    // carry, since a stream fact is one-hot — a class of its own, which
    // the arming pass then cannot find a representative for.
    let (_, rest) = h.split(&Facts::text());
    let (h_video, rest) = rest.split(&Facts::video());
    let (h_audio, _) = rest.split(&Facts::audio());
    let velocity = linear(&m.video_out, &h_video);
    seam::at(seam::VELOCITY, &[&velocity]);
    // The audio head is 32 wide and the video head 96, and a plan carries
    // ONE velocity export (`engine_cuda::exports::Exports::velocity` keeps
    // the first planting; `ModelProfile::velocity_width` is read off it),
    // so the audio prediction goes out on the `hidden` seam — which is the
    // export a lane whose class writes no velocity reads back
    // (`Exports::readout_for`), and the audio lane's class is exactly
    // that. The two-velocity-width export is the recorded follow-up.
    let audio_velocity = linear(&m.audio_out, &h_audio);
    seam::at(seam::HIDDEN, &[&audio_velocity]);
    velocity
}

// The flagship's caption width is the encoder's hidden width; named here
// so the readings table and the model agree by construction.
const _: () = assert!(Dims::h3(1).text_dim == TE_HIDDEN);
// A prompt longer than the encoder's window cannot reach the trunk.
const _: () = assert!(TE_MAX_TOKENS > 0);
// The condition rows' pinned timestep is a fact of the guest's schedule,
// not of the plan; it is stated so a guest and this text spell one number.
const _: () = assert!(CONDITION_TIMESTEP > 0.0);
