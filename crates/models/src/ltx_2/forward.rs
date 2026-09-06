//! LTX-2.5's traced arithmetic: three arms of one plan, selected per lane by
//! the reading bits of the fact word (design D1, D5).
//!
//! | reading | lanes (stream) | binds | reads back |
//! |---|---|---|---|
//! | `denoise` | `Video` + `Audio` + `Context` + `Reference`, one group | video: `latents` `[S, 128]`, `positions` `[S, 3]`, `timestep`; audio: `latents` `[L, 128]`, `audio_positions` `[L, 1]`, `timestep`; context: `context` `[1024, 4096]`, `timestep`; reference: `audio_context` `[1024, 2048]`, `timestep` | `velocity` `[S + L, 128]` on the video AND audio lanes |
//! | `refine.video` | one, `Text` | `text` `[1024, caption·49]`, `text_positions` `[1024, 1]` | `hidden` `[1024, 4096]` |
//! | `refine.audio` | one, `Text` | the same two ports | `hidden` `[1024, 2048]` |
//!
//! # FOUR STREAMS, SIX ATTENTIONS, ONE FIRE
//!
//! LTX-2 is not MM-DiT: nothing is concatenated, the two modalities keep
//! separate widths (video 4096, audio 2048) and separate weights for
//! everything, and they meet only in two cross-attentions. That is exactly
//! D2's picture — a lane per stream, one group, per-stream weights as
//! guarded arms — and it needs no merge at all except at the readout,
//! because every attention's answer comes back under its QUERY's guard
//! (`IMAGEGEN_CONTRACT.md` §1).
//!
//! The four lanes of a step:
//!
//! * `Stream::Video` — the video latent rows, 4096 wide inside.
//! * `Stream::Audio` — the audio latent rows, 2048 wide inside.
//! * `Stream::Context` — the VIDEO text context (the video connector's
//!   1024 rows at 4096).
//! * `Stream::Reference` — the AUDIO text context (1024 rows at 2048). It
//!   is a context lane in everything but name; the stream vocabulary has one
//!   `Context` and this text needs two, at two widths, in two classes.
//!
//! Per block, in the reference's order: video self-attention, audio
//! self-attention, video→text cross-attention, audio→text cross-attention,
//! then the cross-modal PAIR (a2v: video queries over audio keys; v2a: the
//! other way), both reading norms of the two streams taken BEFORE either
//! fold, then the two feed-forwards.
//!
//! # THE TIMESTEP IS PER LANE, AND A PER-TOKEN TIMESTEP IS SEVERAL LANES
//!
//! The reference's I2V path hands the transformer a `[B, S]` timestep in
//! which the conditioning image's tokens carry `0` and every other token
//! carries `t`, and every modulation then runs per token. Every token of
//! one conditioning span shares its value, so this text keeps the per-LANE
//! modulation of `IMAGEGEN_CONTRACT.md` §3 and an I2V step submits the
//! video as SEVERAL `Stream::Video` lanes of one group — the clean tokens
//! at `timestep = 0`, the rest at `t` — each with its own `timestep` cell
//! and its own rotary coordinates. The self-attention packs the group's
//! video lanes into one sequence (`GroupBlockDiagonal` over the video
//! selection's group CSR), which with explicit coordinates is the
//! reference's one sequence in another row order. This is `wan_2`'s
//! answer to the same question, for the same reason: a per-token `[rows, 1]`
//! timestep port would need an f32 token-axis GEMM chain this shell has no
//! arm for.
//!
//! The two context lanes carry a `timestep` cell too, and for a reason: the
//! reference modulates the text context itself
//! (`prompt_scale_shift_table + prompt_adaln_single(t_prompt)`), and a
//! modulation vector must be computed on the arm whose rows it modulates.
//! `t_prompt` is the reference's `amax` over the (possibly per-token)
//! timestep, which for a scalar step is `t` and for an I2V step is the
//! noisy lanes' `t`; the guest hands that number.
//!
//! # POSITIONS ARE PHYSICAL, FRACTIONAL AND ALREADY NORMALISED
//!
//! LTX's rope does not turn by an index. It turns by a coordinate in
//! SECONDS and PIXELS, taken at the MIDPOINT of the latent cell, mapped to
//! `[-1, 1]` against a fixed maximum, and multiplied by `π/2` — and its
//! frequency ladder runs ACROSS the whole row with the axes handed out
//! round-robin along it, which is [`RopeForm::SplitLadder`]. So the
//! `positions` port takes the finished angle scale, not the grid:
//!
//! ```text
//! video row (f, h, w) of an F_l x H_l x W_l latent grid, at `fps`:
//!   t_start = max(f·8 + 1 − 8, 0) / fps        t_end = max((f+1)·8 + 1 − 8, 0) / fps
//!   h_start = h·32                             h_end = (h+1)·32
//!   w_start = w·32                             w_end = (w+1)·32
//!   coord_a = (start_a + end_a) / 2            (`use_middle_indices_grid`)
//!   positions[a] = (2·coord_a / max_a − 1) · π/2,  max = (20 s, 2048 px, 2048 px)
//!
//! audio row f:  mel_start = max(f·4 + 1 − 4, 0),  mel_end = max((f+1)·4 + 1 − 4, 0)
//!   coord = (mel_start + mel_end)/2 · 160/16000  seconds
//!   audio_positions[0] = (2·coord/20 − 1) · π/2
//!
//! connector row i:  text_positions[0] = (2·(i/4096) − 1) · π/2
//! ```
//!
//! The cross-modal pair's rope is the video row's TIME coordinate on one
//! side and the audio row's on the other — the same absolute-seconds axis,
//! normalised by the same 20 s (`max(pos_embed_max_pos,
//! audio_pos_embed_max_pos)`), which is why the video lane's `positions[0]`
//! serves both its own three-axis rope and its half of the pair, and why
//! this text splits the first column off rather than taking a second port.
//! **Q-rope ≠ K-rope**: the a2v attention turns its queries by video time
//! and its keys by audio time, in two `rope_axes` calls on two rectangles.
//!
//! # WHAT THIS ARM DOES NOT DO
//!
//! * The connector's **learnable registers**. The reference compacts the
//!   unpadded rows to the front of the window and fills the tail with tiles
//!   of a `[128, dim]` learned table, then attends everything; that is a
//!   data-dependent gather with no `Layout` member. `refine.*` runs the
//!   blocks over exactly the rows its lane carries and attends all of them,
//!   which is the reference for a prompt that fills its window, and differs
//!   on the padded tail otherwise. A guest hands the rows it wants attended.
//! * The **CFG / STG / modality-guidance passes** of the dev variant: those
//!   are extra lanes and guest arithmetic (D4), not text.
//!
//! [`RopeForm::SplitLadder`]: model_dsl::RopeForm::SplitLadder

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    Generative, LatentSpace, PortFact, PortKind, ReadingFact, ReadoutKind, ScheduleFact,
    ScheduleKind,
};

use super::model::{
    AV_GATE_TIMESTEP_SCALE, AV_SS_SLICES, AdaLn, Attn, Block, Connector, DISTILLED_SIGMAS, Dims,
    Dit, Ffn, GATE_SCALE, Linear, MOD_SLICES, Model, NORM_EPS, PATCH_H, PATCH_T, PATCH_W,
    ROPE_AXES, ROPE_THETA, Side, Stream as StreamHeads, T_FLIP_SIN_COS, T_FREQ_DIM, T_MAX_PERIOD,
    T_SCALE, TEXT_LEN, TRAIN_STEPS, VAE_SPATIAL_COMPRESSION, VAE_TEMPORAL_COMPRESSION, port,
};

/// The bit the one-hot stream facts start at (D2): bits 0..6 are the six
/// streams, of which this text names Video, Audio, Context, Reference and
/// Text.
pub const STREAM_BASE: u8 = 0;

/// The two bits the reading index lives in, as a plain binary code: bit
/// [`READING_LO`] is its low bit, [`READING_HI`] its high bit. Three codes
/// are used of the four — `denoise`, `refine.video`, `refine.audio`; the
/// fourth is where `vae.decode` lands when the decoders do.
pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;

/// Which reading code means what. Every row of this family carries both
/// connectors, so the codes are the same on all of them.
pub const DENOISE: u8 = 0;
pub const REFINE_VIDEO: u8 = 1;
pub const REFINE_AUDIO: u8 = 2;

impl Model {
    /// This row's generative facts (design D12).
    #[must_use]
    pub fn generative(&self) -> Generative {
        let d = &self.dims;
        let port = |name, kind, width, streams: &[Stream], at| PortFact {
            name,
            kind,
            width,
            streams: streams.to_vec(),
            at,
            rows: None,
        };
        // Both connectors read one rectangle of packed trunk rows, and both
        // hand a text lane one coordinate column.
        let refine_ports = || {
            vec![
                port(
                    "text",
                    PortKind::Latents,
                    d.text_in(),
                    &[Stream::Text],
                    Some(port::TEXT),
                ),
                port(
                    "text_positions",
                    PortKind::AxisPositions,
                    1,
                    &[Stream::Text],
                    Some(port::TIME_POSITIONS),
                ),
            ]
        };
        let readings = vec![
            ReadingFact {
                name: "denoise",
                index: DENOISE,
                has_kv: false,
                takes_tokens: false,
                streams: vec![
                    Stream::Video,
                    Stream::Audio,
                    Stream::Context,
                    Stream::Reference,
                ],
                ports: vec![
                    port(
                        "latents",
                        PortKind::Latents,
                        d.channels,
                        &[Stream::Video, Stream::Audio],
                        Some(port::LATENTS),
                    ),
                    port(
                        "context",
                        PortKind::Context,
                        d.cross_dim,
                        &[Stream::Context],
                        Some(port::CONTEXT),
                    ),
                    port(
                        "audio_context",
                        PortKind::Context,
                        d.audio_cross_dim,
                        &[Stream::Reference],
                        Some(port::AUDIO_CONTEXT),
                    ),
                    // Every lane of the reading: the two context lanes
                    // modulate their own rows from the prompt timestep.
                    port(
                        "timestep",
                        PortKind::LaneVector,
                        1,
                        &[],
                        Some(port::TIMESTEP),
                    ),
                    port(
                        "positions",
                        PortKind::AxisPositions,
                        u32::from(ROPE_AXES),
                        &[Stream::Video],
                        Some(port::POSITIONS),
                    ),
                    port(
                        "audio_positions",
                        PortKind::AxisPositions,
                        1,
                        &[Stream::Audio],
                        Some(port::TIME_POSITIONS),
                    ),
                ],
                positions: None,
                readout: ReadoutKind::Velocity,
                readout_width: d.channels,
            },
            ReadingFact {
                name: "refine.video",
                index: REFINE_VIDEO,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Text],
                ports: refine_ports(),
                positions: None,
                readout: ReadoutKind::Hidden,
                readout_width: d.cross_dim,
            },
            ReadingFact {
                name: "refine.audio",
                index: REFINE_AUDIO,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Text],
                ports: refine_ports(),
                positions: None,
                readout: ReadoutKind::Hidden,
                readout_width: d.audio_cross_dim,
            },
        ];
        Generative {
            readings,
            // One token is one latent cell: patch (1, 1, 1) at 128 channels,
            // /32 in space and /8 in time.
            latent: Some(LatentSpace {
                channels: d.channels,
                patch_t: PATCH_T,
                patch_h: PATCH_H,
                patch_w: PATCH_W,
                spatial_compression: VAE_SPATIAL_COMPRESSION,
                temporal_compression: VAE_TEMPORAL_COMPRESSION,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                // The distilled row shifts nothing: its sigmas are pinned.
                // (The dev row's resolution-dependent `mu` — base 0.95 at
                // 1024 tokens, max 2.05 at 4096 — is a second checkpoint's,
                // not this one's.)
                shift: 1.0,
                train_steps: TRAIN_STEPS,
                boundary: None,
                pinned_sigmas: DISTILLED_SIGMAS.to_vec(),
                // The audio stream keeps its OWN scheduler cursor and runs
                // the same eight sigmas on it (`denoising.py:1584` clones
                // the scheduler), so one list states both and no stream
                // takes a shift of its own.
                stream_shifts: Vec::new(),
            }),
            // 960x544x121 is 8160 video + 126 audio + two 1024-row contexts;
            // 1920x1088 is 32640 video. The miniature's job is far smaller,
            // and the connector's rectangle is the flagship's real cost:
            // a `[max_rows, caption·49]` bf16 seat.
            max_rows: match d.layers {
                48 => 32_768 + 2 * TEXT_LEN,
                _ => 4096,
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
    pub fn video() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Video)
    }

    #[must_use]
    pub fn audio() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Audio)
    }

    /// The VIDEO text context lane. `Stream::Text` and not `Context` for
    /// the ORDER it puts the lane in: see the module doc.
    #[must_use]
    pub fn context() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Context)
    }

    /// The AUDIO text context lane, on `Stream::Context` for the same
    /// reason.
    #[must_use]
    pub fn audio_context() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Reference)
    }

    #[must_use]
    pub fn text() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Text)
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

    /// No kv space and no state anywhere: the denoiser holds nothing between
    /// fires and the connectors are one pass each.
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        // Four arms by reading code, each a conjunction of the two reading
        // literals (so every arm names a `Selection` the host can pack).
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (c3, c2) = hi.split(&Facts::reading_lo());
        let (c1, c0) = lo.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3];
        let arm = |code: u8| &arms[usize::from(code)];

        let velocity = denoise(arm(DENOISE), self);
        let (text_in, caption) = (self.dims.text_in(), self.dims.caption);
        refine(
            arm(REFINE_VIDEO),
            &self.connectors.0,
            text_in,
            self.connectors.0.rescale(caption),
        );
        refine(
            arm(REFINE_AUDIO),
            &self.connectors.1,
            text_in,
            self.connectors.1.rescale(caption),
        );
        velocity
    }
}

/// One projection, biased where the checkpoint has a bias.
fn linear(w: &Linear, x: &Value) -> Value {
    let y = ops::linear::matmul(x, &w.w);
    match &w.bias {
        Some(bias) => ops::elemwise::add_bias(bias, &y),
        None => y,
    }
}

/// `RMSNormNoWeight` over the whole row: `x · rsqrt(mean(x²) + eps)`.
fn rms(x: &Value) -> Value {
    let width = u32::try_from(x.width()).expect("a row narrower than 4 G");
    ops::elemwise::rmsnorm_no_scale(x, width, NORM_EPS)
}

/// `x · (1 + scale) + shift`, per lane.
fn modulate(x: &Value, scale_shift: &Value, lanes: &Value) -> Value {
    ops::elemwise::modulate(x, scale_shift, Some(lanes), ModulateForm::ScaleShift)
}

/// The scale-free RMS norm and the modulation that follows it.
fn norm_modulate(x: &Value, scale_shift: &Value, lanes: &Value) -> Value {
    modulate(&rms(x), scale_shift, lanes)
}

/// The tables one stream's attentions need: the token→lane map, the rotary
/// coordinates (three columns on the video lane, one on the audio lane and
/// on a text lane), and the selection's permutation and group CSR.
struct Geom {
    lanes: Value,
    positions: Value,
    perm: Value,
    csr: Value,
}

/// A key side's tables: its permutation, its group CSR, and — for a context
/// lane, which modulates its own rows — its token→lane map.
struct KeyGeom {
    lanes: Value,
    perm: Value,
    csr: Value,
}

/// One stream's nine modulation rows cut into what they apply: the
/// self-attention pair and gate, the FFN pair and gate, the prompt
/// cross-attention pair and gate — the plan's own slice order, which
/// `import.rs` makes.
struct Mods {
    msa_ss: Value,
    msa_gate: Value,
    mlp_ss: Value,
    mlp_gate: Value,
    q_ss: Value,
    q_gate: Value,
}

fn adaln9(e: &Value, dim: u32) -> Mods {
    debug_assert_eq!(e.width(), u64::from(MOD_SLICES * dim));
    let (msa_ss, rest) = ops::layout::split_rows(e, 2 * dim);
    let (msa_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (mlp_ss, rest) = ops::layout::split_rows(&rest, 2 * dim);
    let (mlp_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (q_ss, q_gate) = ops::layout::split_rows(&rest, 2 * dim);
    Mods {
        msa_ss,
        msa_gate,
        mlp_ss,
        mlp_gate,
        q_ss,
        q_gate,
    }
}

/// One stream's four cross-modal rows: the a2v pair and the v2a pair. The
/// checkpoint stores them scale-first already, so no exchange is needed.
struct AvMods {
    a2v_ss: Value,
    v2a_ss: Value,
}

fn adaln_av(e: &Value, dim: u32) -> AvMods {
    debug_assert_eq!(e.width(), u64::from(AV_SS_SLICES * dim));
    let (a2v_ss, v2a_ss) = ops::layout::split_rows(e, 2 * dim);
    AvMods { a2v_ss, v2a_ss }
}

/// One `LTX2AdaLayerNormSingle` over a lane's timestep: the sinusoid, the
/// two-layer embedder, and `linear(silu(·))`. The embedder's hidden is
/// handed back beside the answer because the model head reads it (as
/// `embedded_timestep`, through the doubled `head_proj`).
fn adaln(head: &AdaLn, sinusoid: &Value) -> (Value, Value) {
    let h = ops::elemwise::silu(&linear(&head.embed.linear_1, sinusoid));
    let emb = linear(&head.embed.linear_2, &h);
    (linear(&head.proj, &ops::elemwise::silu(&emb)), h)
}

/// `timestep_embedding(t, 256, max_period=10000)` — `[sin | cos]`, fp32.
fn sinusoid(t: &Value) -> Value {
    ops::elemwise::sinusoid(t, T_FREQ_DIM, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE)
}

/// Every modulation vector one stream's lanes carry, per block-invariant
/// piece: the nine rows, the four cross-modal rows, the one gate row, and
/// the head's `[temb | temb] + table`.
struct StreamMods {
    proj9: Value,
    av_ss: Value,
    av_gate: Value,
    head: Value,
}

fn stream_mods(heads: &StreamHeads, t: &Value) -> StreamMods {
    let sin = sinusoid(t);
    let (proj9, hidden) = adaln(&heads.adaln, &sin);
    let (av_ss, _) = adaln(&heads.av_ss, &sin);
    // The cross-modal GATE reads its own timestep scale; for LTX-2.5 the
    // reference's factor is 1, so the same sinusoid serves.
    debug_assert_eq!(AV_GATE_TIMESTEP_SCALE, 1.0);
    let (av_gate, _) = adaln(&heads.av_gate, &sin);
    let head = ops::elemwise::add_bias(&heads.head_table, &linear(&heads.head_proj, &hidden));
    StreamMods {
        proj9,
        av_ss,
        av_gate,
        head,
    }
}

/// The attention's QK norm: `torch.nn.RMSNorm(inner_dim)` — one gain over
/// the whole `heads·head_dim` row, not per head.
fn qk_norm(x: &Value, gain: &Weight) -> Value {
    ops::elemwise::rmsnorm(x, gain, NORM_EPS)
}

/// One rope call in LTX's own form: one ladder across the row, the axes
/// round-robin along it, the pairing rotate-half within a head.
fn turn(x: &Value, positions: &Value, dims: [u32; 4], head_dim: u32) -> Value {
    ops::elemwise::rope_axes(
        x,
        positions,
        dims,
        [ROPE_THETA; 4],
        RopeForm::SplitLadder,
        head_dim,
        head_dim,
    )
}

/// The gated fold every LTX attention ends with: `out · 2σ(W_gate · x_norm)`
/// per head, then the output projection.
fn gate_out(o: &Value, h: &Value, a: &Attn) -> Value {
    let logits = linear(&a.gate, h);
    let gated = ops::elemwise::gate_sigmoid_mul_heads(o, &logits, a.head_dim, GATE_SCALE);
    linear(&a.out, &gated)
}

/// A self-attention over one selection's rows: packed `q|k|v`, the
/// across-heads QK norms, one rope per side, one ragged read over the
/// selection's group CSR, the per-head gate, the output projection.
fn self_attention(h: &Value, a: &Attn, dims: [u32; 4], g: &Geom) -> Value {
    let inner = a.inner();
    let (q, k, v) = ops::layout::split_qkv(&linear(&a.qkv, h), inner, inner);
    let q = turn(&qk_norm(&q, &a.q_norm), &g.positions, dims, a.head_dim);
    let k = turn(&qk_norm(&k, &a.k_norm), &g.positions, dims, a.head_dim);
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&q, &g.perm),
        &ops::layout::pack_rows(&k, &g.perm),
        &ops::layout::pack_rows(&v, &g.perm),
        &g.csr,
        &g.csr,
        a.head_dim,
        a.sm_scale(),
        RaggedMask::GroupBlockDiagonal,
    );
    gate_out(&ops::layout::unpack_rows(&o, &g.perm), h, a)
}

/// A cross-attention: queries off `h` (with `qg`'s tables), keys and values
/// off `ctx` (with `kg`'s), each side turned by its OWN coordinates — which
/// is what makes the cross-modal pair a clock and not a mixer. `rope` is
/// `None` for the text cross-attentions, which turn nothing.
#[allow(clippy::too_many_arguments)]
fn cross_attention(
    h: &Value,
    ctx: &Value,
    a: &Attn,
    rope: Option<(&Value, &Value, [u32; 4])>,
    qg: &Geom,
    kg: &KeyGeom,
) -> Value {
    let inner = a.inner();
    let kv =
        a.kv.as_ref()
            .expect("a cross-attention keeps its k|v apart");
    let q = qk_norm(&linear(&a.qkv, h), &a.q_norm);
    let (k, v) = ops::layout::split_rows(&linear(kv, ctx), inner);
    let k = qk_norm(&k, &a.k_norm);
    let (q, k) = match rope {
        Some((q_pos, k_pos, dims)) => (
            turn(&q, q_pos, dims, a.head_dim),
            turn(&k, k_pos, dims, a.head_dim),
        ),
        None => (q, k),
    };
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&q, &qg.perm),
        &ops::layout::pack_rows(&k, &kg.perm),
        &ops::layout::pack_rows(&v, &kg.perm),
        &qg.csr,
        &kg.csr,
        a.head_dim,
        a.sm_scale(),
        RaggedMask::GroupBlockDiagonal,
    );
    gate_out(&ops::layout::unpack_rows(&o, &qg.perm), h, a)
}

/// `x += gate · down(gelu_tanh(up(mod(rms(x)))))`.
fn ff_sublayer(x: &Value, ff: &Ffn, ss: &Value, gate: &Value, lanes: &Value) -> Value {
    let h = norm_modulate(x, ss, lanes);
    let f = linear(&ff.down, &ops::elemwise::gelu(&linear(&ff.up, &h), true));
    ops::elemwise::gated_residual_add(x, gate, &f, Some(lanes))
}

/// One stream's per-block modulation, table plus global vector.
struct BlockMods {
    m: Mods,
    av: AvMods,
    av_gate: Value,
    prompt_ss: Value,
}

/// `table + vector`, on a COPY of the vector. `elementwise.add_bias` folds
/// its bias in place (the IR aliases `out_out` onto `out`), and every table
/// below is added to a vector the WHOLE STACK shares — one adaLN head serves
/// all 48 blocks — so each block must add its table to a copy or the second
/// block reads the first block's table as well as its own.
fn table_add(table: &Weight, v: &Value) -> Value {
    ops::elemwise::add_bias(table, &ops::elemwise::copy(v))
}

fn block_mods(side: &Side, s: &StreamMods, prompt: &Value, dim: u32) -> BlockMods {
    BlockMods {
        m: adaln9(&table_add(&side.table, &s.proj9), dim),
        av: adaln_av(&table_add(&side.av_ss_table, &s.av_ss), dim),
        av_gate: table_add(&side.av_gate_table, &s.av_gate),
        prompt_ss: table_add(&side.prompt_table, prompt),
    }
}

/// The two streams' rows through one `LTX2TransformerBlock`.
#[allow(clippy::too_many_arguments)]
fn block(
    xv: &Value,
    xa: &Value,
    b: &Block,
    d: &Dims,
    mv: &BlockMods,
    ma: &BlockMods,
    ctx: &Value,
    actx: &Value,
    vg: &Geom,
    ag: &Geom,
    v_time: &Value,
    cg: &KeyGeom,
    acg: &KeyGeom,
) -> (Value, Value) {
    // 1-2. The two self-attentions, each over its own stream's rows.
    let hv = norm_modulate(xv, &mv.m.msa_ss, &vg.lanes);
    let ov = self_attention(&hv, &b.video.self_attn, d.rope_dims(), vg);
    let xv = ops::elemwise::gated_residual_add(xv, &mv.m.msa_gate, &ov, Some(&vg.lanes));

    let ha = norm_modulate(xa, &ma.m.msa_ss, &ag.lanes);
    let oa = self_attention(&ha, &b.audio.self_attn, d.audio_rope_dims(), ag);
    let xa = ops::elemwise::gated_residual_add(xa, &ma.m.msa_gate, &oa, Some(&ag.lanes));

    // 3-4. The two text cross-attentions. The CONTEXT is modulated too,
    //      on its own arm, from its own lane's prompt timestep.
    let hv = norm_modulate(&xv, &mv.m.q_ss, &vg.lanes);
    let c = modulate(ctx, &mv.prompt_ss, &cg.lanes);
    let ov = cross_attention(&hv, &c, &b.video.cross, None, vg, cg);
    let xv = ops::elemwise::gated_residual_add(&xv, &mv.m.q_gate, &ov, Some(&vg.lanes));

    let ha = norm_modulate(&xa, &ma.m.q_ss, &ag.lanes);
    let ac = modulate(actx, &ma.prompt_ss, &acg.lanes);
    let oa = cross_attention(&ha, &ac, &b.audio.cross, None, ag, acg);
    let xa = ops::elemwise::gated_residual_add(&xa, &ma.m.q_gate, &oa, Some(&ag.lanes));

    // 5. The cross-modal pair. BOTH norms are taken before EITHER fold —
    //    v2a reads the video rows as a2v found them, not as a2v left them.
    let nv = rms(&xv);
    let na = rms(&xa);
    let av_dims = d.av_rope_dims();

    let q_in = modulate(&nv, &mv.av.a2v_ss, &vg.lanes);
    let kv_in = modulate(&na, &ma.av.a2v_ss, &ag.lanes);
    let o = cross_attention(
        &q_in,
        &kv_in,
        &b.a2v,
        Some((v_time, &ag.positions, av_dims)),
        vg,
        &KeyGeom {
            lanes: ag.lanes.clone(),
            perm: ag.perm.clone(),
            csr: ag.csr.clone(),
        },
    );
    let xv = ops::elemwise::gated_residual_add(&xv, &mv.av_gate, &o, Some(&vg.lanes));

    let q_in = modulate(&na, &ma.av.v2a_ss, &ag.lanes);
    let kv_in = modulate(&nv, &mv.av.v2a_ss, &vg.lanes);
    let o = cross_attention(
        &q_in,
        &kv_in,
        &b.v2a,
        Some((&ag.positions, v_time, av_dims)),
        ag,
        &KeyGeom {
            lanes: vg.lanes.clone(),
            perm: vg.perm.clone(),
            csr: vg.csr.clone(),
        },
    );
    let xa = ops::elemwise::gated_residual_add(&xa, &ma.av_gate, &o, Some(&ag.lanes));

    // 6. The two feed-forwards.
    let xv = ff_sublayer(&xv, &b.video.ffn, &mv.m.mlp_ss, &mv.m.mlp_gate, &vg.lanes);
    let xa = ff_sublayer(&xa, &b.audio.ffn, &ma.m.mlp_ss, &ma.m.mlp_gate, &ag.lanes);
    (xv, xa)
}

/// The `denoise` reading.
fn denoise(arm: &Input<Facts>, m: &Model) -> Value {
    let d = &m.dims;
    let dit: &Dit = &m.dit;

    // The four lanes: the two text contexts peeled off first, so that what
    // is left is EXACTLY the two modalities — the guard the merged velocity
    // is planted under, and the one every class of it must be covered by.
    let (ctx, rest) = arm.split(&Facts::context());
    let (actx, media) = rest.split(&Facts::audio_context());
    let (vid, aud) = media.split(&Facts::video());

    let vg = Geom {
        lanes: vid.request_of_token(),
        positions: vid.axis_positions(port::POSITIONS, ROPE_AXES),
        perm: vid.row_permutation(),
        csr: vid.group_indptr(),
    };
    let ag = Geom {
        lanes: aud.request_of_token(),
        positions: aud.axis_positions(port::TIME_POSITIONS, 1),
        perm: aud.row_permutation(),
        csr: aud.group_indptr(),
    };
    let cg = KeyGeom {
        lanes: ctx.request_of_token(),
        perm: ctx.row_permutation(),
        csr: ctx.group_indptr(),
    };
    let acg = KeyGeom {
        lanes: actx.request_of_token(),
        perm: actx.row_permutation(),
        csr: actx.group_indptr(),
    };
    // The cross-modal pair turns the video rows by their TIME coordinate
    // alone, normalised by the same 20 s the three-axis rope uses, so the
    // first column of the video positions IS the pair's video half.
    let (v_time, _) = ops::layout::split_rows(&vg.positions, 1);

    // ---- the conditioning vectors, once per lane --------------------------
    let vt = vid.lane_vector(port::TIMESTEP, 1);
    let at = aud.lane_vector(port::TIMESTEP, 1);
    let mods_v = stream_mods(&dit.video, &vt);
    let mods_a = stream_mods(&dit.audio, &at);
    // The two prompt vectors live on the CONTEXT arms: they modulate those
    // lanes' rows, and a modulation vector is computed where it is applied.
    let (prompt_v, _) = adaln(&dit.prompt, &sinusoid(&ctx.lane_vector(port::TIMESTEP, 1)));
    let (prompt_a, _) = adaln(
        &dit.audio_prompt,
        &sinusoid(&actx.lane_vector(port::TIMESTEP, 1)),
    );

    // ---- the rows in ------------------------------------------------------
    let mut xv = linear(
        &dit.video.patchify,
        &vid.latents(port::LATENTS, d.channels, Dtype::Bf16),
    );
    let mut xa = linear(
        &dit.audio.patchify,
        &aud.latents(port::LATENTS, d.channels, Dtype::Bf16),
    );
    let ctx_rows = ctx.context(port::CONTEXT, d.cross_dim);
    let actx_rows = actx.context(port::AUDIO_CONTEXT, d.audio_cross_dim);

    for (_, b) in arm.walk_layers(&dit.blocks) {
        let mv = block_mods(&b.video, &mods_v, &prompt_v, d.dim());
        let ma = block_mods(&b.audio, &mods_a, &prompt_a, d.audio_dim());
        let (v, a) = block(
            &xv, &xa, b, d, &mv, &ma, &ctx_rows, &actx_rows, &vg, &ag, &v_time, &cg, &acg,
        );
        xv = v;
        xa = a;
    }

    // ---- the two heads ----------------------------------------------------
    // `LayerNorm(affine=False)` in fp32, then `·(1 + scale) + shift` from
    // `scale_shift_table + embedded_timestep`, then `proj_out`.
    let head = |x: &Value, s: &StreamHeads, mods: &StreamMods, g: &Geom| {
        let h = ops::elemwise::modulate(
            &ops::elemwise::layernorm_no_scale(x, NORM_EPS),
            &mods.head,
            Some(&g.lanes),
            ModulateForm::ScaleShift,
        );
        linear(&s.proj_out, &h)
    };
    let vv = head(&xv, &dit.video, &mods_v, &vg);
    let va = head(&xa, &dit.audio, &mods_a, &ag);
    // One velocity plane over both streams' rows: the two are 128 wide
    // apiece and their rows are disjoint, so a merge is the whole readout
    // and each lane reads back its own.
    let velocity = Value::merge(vec![vv, va]);
    seam::at(seam::VELOCITY, &[&velocity]);
    velocity
}

/// A `refine.*` reading: one connector transformer over the packed trunk
/// rows a text lane carries.
fn refine(arm: &Input<Facts>, conn: &Connector, text_in: u32, rescale: f32) {
    let g = Geom {
        lanes: arm.request_of_token(),
        positions: arm.axis_positions(port::TIME_POSITIONS, 1),
        perm: arm.row_permutation(),
        csr: arm.lane_indptr(),
    };
    // `W·(s·x) + b` with the reference's `sqrt(dim / caption_channels)`
    // rescale moved to the far side of the projection, where it is one
    // in-place scale of a fresh rectangle instead of a copy of the
    // `caption·49`-wide port cell.
    let x = arm.latents(port::TEXT, text_in, Dtype::Bf16);
    let mut h = ops::elemwise::mul_scalar(rescale, &ops::linear::matmul(&x, &conn.aggregate.w));
    if let Some(bias) = &conn.aggregate.bias {
        h = ops::elemwise::add_bias(bias, &h);
    }
    let dims = conn.rope_dims();
    for (_, b) in arm.walk_layers(&conn.blocks) {
        let n = rms(&h);
        let o = self_attention(&n, &b.attn, dims, &g);
        h = ops::elemwise::residual_add(&o, &h);
        let n = rms(&h);
        let f = linear(
            &b.ffn.down,
            &ops::elemwise::gelu(&linear(&b.ffn.up, &n), true),
        );
        h = ops::elemwise::residual_add(&f, &h);
    }
    let out = rms(&h);
    seam::at(seam::HIDDEN, &[&out]);
}
