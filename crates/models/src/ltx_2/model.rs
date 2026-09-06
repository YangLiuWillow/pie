//! The `ltx_2` declaration: every dimension a Rust constant or a [`Dims`]
//! field, every weight named in the plan's own scheme.
//!
//! Two components under one plan (design D5): the two text connectors
//! ([`Connector`], `connectors.`) and the audio-video transformer
//! ([`Dit`], `dit.`). The numbers are `Lightricks/LTX-2.5-Diffusers`'s
//! `transformer/config.json` and `connectors/config.json`, read off the
//! snapshot and restated here because a family's dims are Rust constants
//! and a `config.json` is carried, never read (study
//! `.wiki/imagegen/study/ltx25.md` §C, §D.1).
//!
//! The miniature is `scripts/imagegen/ltx2_golden.py --mini`'s: two blocks,
//! two heads a side, the REAL head widths (video 128, audio 64 —
//! `attention.ragged` is stamped at 64/128/256) and the real 128-channel
//! latents, random weights.
//!
//! # What this text does not declare, and why
//!
//! * The **Gemma-4-12B text trunk** (`text_encoder/`). Its 49 hidden states
//!   are the connectors' input, and the two `refine` readings take them on
//!   a float port instead. The trunk is `gemma4-12b-ltx-v1`
//!   (`text_encoder/config.json`): 48 layers, hidden 3840, 16 heads at
//!   head_dim 256 with a 512-wide GLOBAL head every sixth layer, 8 kv heads
//!   (1 global), `attention_k_eq_v`, and a `proportional` rope at
//!   `partial_rotary_factor 0.25` — three features this tree has read no
//!   reference for. A `text` reading folding the two `aggregate_embed`
//!   projections into its tail (49 matmuls summed, the IR having no width
//!   concat) is the natural follow-up and needs no new op: it would hand
//!   `[L, cross_dim + audio_cross_dim]` back on one `hidden` seam and the
//!   `refine` readings would split it.
//! * The **video and audio VAE decoders**. Two things stopped them the
//!   pass this text was written in, and NEITHER STOPS THEM NOW:
//!   `vae/config.json` had not arrived in the snapshot (it has), and the
//!   video decoder's temporal upsampler drops the first `s_t − 1` frames
//!   after its pixel shuffle as the causal anchor, which no `Spatial`
//!   member stated — `GridRule::Shuffle` now carries `trim_t` and
//!   `spatial::pixel_shuffle_trimming` writes it (`Upsample
//!   { keep_first_frame }` produces the same `1 + (t−1)·s_t` extent by
//!   REPLICATION, not by shuffling, which is why the trim had to be its
//!   own statement). What is left is the text itself: four up blocks of
//!   `[2, 4, 6, 4]` resnets over channels `[1024, 512, 512, 256]`, whose
//!   `PerChannelRMSNorm` is `elemwise::rmsnorm_no_scale(x, C, 1e-8)` on a
//!   voxel row (a row IS a location and its width IS the channels, so the
//!   reference's RMS across the channel dim is a row norm, not a
//!   `Spatial::GroupNorm`) and whose channel-change shortcut is
//!   `elemwise::layernorm_no_scale` plus its affine. The audio side maps
//!   cleanly onto the voxel axis otherwise (time on the grid's `t` and
//!   causal, mel on its `h`, `w = 1`, `[voxels, 8]` in), and is the
//!   smaller of the two.
//! * The **diffusion decoder**, the **latent upsampler**, the **duration
//!   head** and the **48 kHz vocoder**. The vocoder is the only one blocked
//!   by the op vocabulary rather than by time: its BigVGAN stack is DILATED
//!   `Conv1d` and `ConvTranspose1d` behind anti-aliased sinc filters, and
//!   `Spatial::Conv3d` carries neither dilation nor a transpose.
//!   `Elementwise::Snake` was therefore NOT added — nothing this text
//!   traces needs SnakeBeta.
//!
//! # Numerics contract
//!
//! * Banks are bf16 (the snapshot ships bf16). Token-row activations are
//!   bf16; matmuls accumulate fp32 and round once.
//! * The pre-attention norms are `RMSNormNoWeight` over the WHOLE row
//!   (`elementwise.rmsnorm_no_scale` at `head_dim = width`): fp32
//!   statistics, one rounding at the store; the modulation that follows
//!   reads the bf16 row and an f32 vector and rounds once more.
//! * The QK norms are `torch.nn.RMSNorm(inner_dim)` — across ALL heads,
//!   with a gain — which is `elementwise.rmsnorm` over the whole
//!   `heads·head_dim` row, fp32 sum, one rounding. The reference runs them
//!   with autocast disabled, in the activation dtype, which is this.
//! * The timestep chains (sinusoid → `timestep_embedder` → SiLU → the
//!   adaLN linear) are lane-vector chains and stay fp32 end to end; the
//!   reference runs the sinusoid in fp32 and the two MLPs in bf16, so a
//!   port is the more precise side. The `scale_shift_table`s are added in
//!   fp32, as the reference does.
//! * The rope is [`RopeForm::SplitLadder`] over f32 positions the guest
//!   hands already normalised (`forward.rs` states the arithmetic to the
//!   letter). The reference builds its ladder in float64 and rounds to f32
//!   before the outer product; a `powf` ladder in f32 is those numbers to
//!   about one ulp, and the kernel's `sincosf` matches torch's reduction at
//!   the ~1.6e4-radian angles this rope reaches.
//! * The output norms are `LayerNorm(elementwise_affine=False)`, which the
//!   reference runs with autocast off (fp32 statistics), then a per-lane
//!   modulation.
//!
//! [`RopeForm::SplitLadder`]: model_dsl::RopeForm::SplitLadder

use model_dsl::{Dtype, Weight};

/// The DiT patch: one token is one latent cell (`patch_size 1`,
/// `patch_size_t 1`).
pub const PATCH_T: u32 = 1;
pub const PATCH_H: u32 = 1;
pub const PATCH_W: u32 = 1;

/// `vae_scale_factors`: pixels per latent cell along height and width, and
/// frames per latent cell along time (`F` frames map to `(F − 1)/8 + 1`).
pub const VAE_SPATIAL_COMPRESSION: u32 = 32;
pub const VAE_TEMPORAL_COMPRESSION: u32 = 8;
/// `latent_channels` of the video VAE: the width of one latent row.
pub const VAE_Z: u32 = 128;

/// The sinusoidal timestep embedding: sglang's `timestep_embedding(t, 256,
/// max_period=10000)` concatenates `[cos | sin]` at `downscale_freq_shift =
/// 0`, which is `Elementwise::Sinusoid` under `flip_sin_cos`; the scheduler
/// timestep (`σ·1000`) goes in raw.
pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;

/// `norm_eps` of every norm in the transformer and the connectors.
pub const NORM_EPS: f32 = 1e-6;

/// `rope_theta`, and the maxima each axis is normalised by (`pos_embed_max_pos`,
/// `base_height`, `base_width`, `audio_pos_embed_max_pos`). The cross-modal
/// pair takes `max(video, audio)`, which is the same 20 s.
pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_MAX_POS: [f32; 3] = [20.0, 2048.0, 2048.0];
pub const AUDIO_ROPE_MAX_POS: f32 = 20.0;
pub const CROSS_ROPE_MAX_POS: f32 = 20.0;
/// The rope's axis counts: three for the video sequence, one for audio and
/// for both halves of the cross-modal pair.
pub const ROPE_AXES: u8 = 3;
pub const AUDIO_ROPE_AXES: u8 = 1;

/// `vae_scale_factors` / `audio_scale_factor` / `causal_offset`: how a
/// latent cell becomes a physical coordinate. The guest builds the
/// coordinates; `forward.rs` states the arithmetic.
pub const VIDEO_SCALE: [f32; 3] = [8.0, 32.0, 32.0];
pub const AUDIO_SCALE: f32 = 4.0;
pub const CAUSAL_OFFSET: f32 = 1.0;
/// `audio_sampling_rate` / `audio_hop_length`: mel frames per second, and
/// so `sampling_rate / hop / audio_scale` audio latent frames per second.
pub const AUDIO_SAMPLING_RATE: f32 = 16_000.0;
pub const AUDIO_HOP: f32 = 160.0;

/// `gated_attn` / `audio_gated_attn`: every attention multiplies its answer
/// by `2σ(W_gate · x_norm)`, one logit per head.
pub const GATE_SCALE: f32 = 2.0;

/// How many `[dim]` rows a block's own tables carry: nine per stream
/// (`shift/scale/gate` for the self-attention, the FFN and the prompt
/// cross-attention), four plus one per stream for the cross-modal pair
/// (two `scale/shift` pairs and one gate, which come from two DIFFERENT
/// global adaLN heads and so are two tables here), and two per stream for
/// the text context. The head's is two more.
pub const MOD_SLICES: u32 = 9;
pub const AV_SS_SLICES: u32 = 4;
pub const AV_GATE_SLICES: u32 = 1;
pub const PROMPT_SLICES: u32 = 2;
pub const HEAD_SLICES: u32 = 2;

/// `cross_attn_timestep_scale_multiplier` is 1000 in the checkpoint and the
/// reference does not read it: for a non-2.3 variant sglang's
/// `_get_av_ca_gate_timestep_factor` answers its own default,
/// `av_ca_timestep_scale_multiplier = 1`. So the cross-modal GATE reads the
/// same timestep as everything else, and this text does too — the golden is
/// the reference's, and the checkpoint key is an open question the study
/// should carry.
pub const AV_GATE_TIMESTEP_SCALE: f32 = 1.0;

/// `num_train_timesteps`, and the distilled sigma list `LTX25Config` pins
/// (`configs/pipeline_configs/ltx_2_5.py`), descending, without the
/// trailing zero the scheduler appends.
pub const TRAIN_STEPS: u32 = 1000;
pub const DISTILLED_SIGMAS: [f32; 8] = [
    1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875,
];
/// The second stage's three sigmas, after the ×2 latent upsample and the
/// renoise at 0.909375 (study §B.2). Stated for the record; the guest
/// drives the stage handoff.
pub const STAGE2_SIGMAS: [f32; 3] = [0.909_375, 0.725, 0.421_875];

/// `text_proj_in_factor`: how many of the trunk's hidden states the
/// connectors read (the embedding output plus 48 layers). One packed text
/// row is `caption · TEXT_LAYERS` wide, laid out `(hidden, layer)` — layer
/// fastest, which is `pack_text_embeds_v2`'s `flatten(2)` of
/// `[.., hidden, 49]`.
pub const TEXT_LAYERS: u32 = 49;
/// The tokenizer's `max_length`, and so the row count a connector lane
/// takes at the flagship.
pub const TEXT_LEN: u32 = 1024;
/// `*_connector_num_learnable_registers`: the connector replaces padded
/// positions with tiles of this table. Not declared — see [`Connector`].
pub const CONN_REGISTERS: u32 = 128;
/// `connector_rope_base_seq_len`: row `i` sits at `i / 4096`.
pub const CONN_ROPE_BASE: f32 = 4096.0;
/// The connectors' feed-forward multiplier (diffusers' `FeedForward`
/// default).
pub const CONN_FF_MULT: u32 = 4;

/// The float ports this text reads, by index within their kind. A port
/// index is the family's own and is what the runtime resolves `input(name)`
/// to; two readings that read one kind at DIFFERENT widths take different
/// indices (`PortFact::at`), because the engine seats one rectangle per
/// `(kind, index)` for the whole plan.
pub mod port {
    /// `denoise`: a video OR audio lane's latent rows, `[rows, 128]` bf16.
    /// ONE rectangle for both streams: their rows are disjoint, and both
    /// latents are 128 channels wide.
    pub const LATENTS: u8 = 0;
    /// `denoise`: the video text context, `[1024, 4096]` bf16.
    pub const CONTEXT: u8 = 0;
    /// `denoise`: the audio text context, `[1024, 2048]` bf16.
    pub const AUDIO_CONTEXT: u8 = 1;
    /// `refine.*`: the packed trunk stack, `[rows, caption·49]` bf16. A
    /// LATENTS port and not a context one, at an index of its own: a
    /// token-less reading states its lane's row count through its latents
    /// port, and the runtime refuses one that does not
    /// (`runtime::validate_generative`).
    pub const TEXT: u8 = 1;
    /// The scheduler timestep `σ·1000`, `[lanes, 1]`, bound by EVERY lane
    /// of the `denoise` reading — the two context lanes modulate their own
    /// rows from the prompt timestep, so they carry a cell too.
    pub const TIMESTEP: u8 = 0;
    /// `denoise`: the video rows' three normalised rope coordinates.
    pub const POSITIONS: u8 = 0;
    /// `denoise`: the audio rows' one normalised coordinate, and
    /// `refine.*`: the text rows' one.
    pub const TIME_POSITIONS: u8 = 1;
}

/// One row's transformer shape. The audio side is a second, narrower set of
/// numbers over the same block count — LTX-2 is asymmetric, not shared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub layers: u32,
    /// Video: `num_attention_heads`, `attention_head_dim`.
    pub heads: u32,
    pub head_dim: u32,
    /// Audio: `audio_num_attention_heads`, `audio_attention_head_dim`. The
    /// cross-modal pair runs at these too, whichever way it points.
    pub audio_heads: u32,
    pub audio_head_dim: u32,
    /// `in_channels` / `out_channels`, video and audio — 128 everywhere.
    pub channels: u32,
    /// `cross_attention_dim`: the width of the video text context (the
    /// video connector's output).
    pub cross_dim: u32,
    /// `audio_cross_attention_dim`: the audio text context's width.
    pub audio_cross_dim: u32,
    /// `mult` of both feed-forwards.
    pub ff_mult: u32,
    /// `caption_channels`: the trunk's hidden width, one of the 49 stacked
    /// planes a packed text row carries.
    pub caption: u32,
    /// `*_connector_num_layers`.
    pub conn_layers: u32,
}

impl Dims {
    /// `Lightricks/LTX-2.5-Diffusers`'s `transformer/config.json` and
    /// `connectors/config.json`.
    #[must_use]
    pub const fn ltx_2_5() -> Dims {
        Dims {
            layers: 48,
            heads: 32,
            head_dim: 128,
            audio_heads: 32,
            audio_head_dim: 64,
            channels: 128,
            cross_dim: 4096,
            audio_cross_dim: 2048,
            ff_mult: 4,
            caption: 3840,
            conn_layers: 8,
        }
    }

    /// `ltx2_golden.py`'s `mini`: two blocks, two heads a side at the REAL
    /// head widths, the real 128-channel latents, a 16-wide caption. The
    /// video rope's `dim/6` ladder becomes 42 frequencies an axis and keeps
    /// the two-slot identity pad, which is the thing to exercise.
    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            layers: 2,
            heads: 2,
            head_dim: 128,
            audio_heads: 2,
            audio_head_dim: 64,
            channels: 128,
            cross_dim: 256,
            audio_cross_dim: 128,
            ff_mult: 4,
            caption: 16,
            conn_layers: 1,
        }
    }

    /// The video stream's width.
    #[must_use]
    pub const fn dim(&self) -> u32 {
        self.heads * self.head_dim
    }

    /// The audio stream's width.
    #[must_use]
    pub const fn audio_dim(&self) -> u32 {
        self.audio_heads * self.audio_head_dim
    }

    /// The cross-modal attentions' inner width: audio heads at audio head
    /// width, whichever stream the queries come from.
    #[must_use]
    pub const fn av_inner(&self) -> u32 {
        self.audio_heads * self.audio_head_dim
    }

    /// One packed text row: `caption_channels · text_proj_in_factor`.
    #[must_use]
    pub const fn text_in(&self) -> u32 {
        self.caption * TEXT_LAYERS
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub fn audio_sm_scale(&self) -> f32 {
        (self.audio_head_dim as f32).sqrt().recip()
    }

    /// The video rope's per-axis channel count over the WHOLE row
    /// (`RopeForm::SplitLadder` reads `dims` that way): the ladder holds
    /// `dim / (2·axes)` frequencies an axis, and what the row has left over
    /// is the identity pad the reference concatenates in front.
    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        let f = self.dim() / (2 * ROPE_AXES as u32);
        [2 * f, 2 * f, 2 * f, 0]
    }

    /// The audio rope's: one axis, `audio_dim/2` frequencies, no pad.
    #[must_use]
    pub const fn audio_rope_dims(&self) -> [u32; 4] {
        [self.audio_dim(), 0, 0, 0]
    }

    /// The cross-modal pair's, on both sides: one axis over
    /// `audio_cross_attention_dim`, which is the a2v/v2a inner width.
    #[must_use]
    pub const fn av_rope_dims(&self) -> [u32; 4] {
        [self.av_inner(), 0, 0, 0]
    }
}

/// One `nn.Linear`, its bias declared only where the checkpoint has one
/// (LTX-2.5 drops the VIDEO feed-forward's and keeps the audio one).
pub struct Linear {
    pub w: Weight,
    pub bias: Option<Weight>,
}

impl Linear {
    fn at(name: &str, out: u32, in_: u32, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [u64::from(out), u64::from(in_)], banks),
            bias: Some(Weight::sym(
                format!("{name}.bias"),
                [u64::from(out)],
                crate::dense(banks),
            )),
        }
    }

    fn plain(name: &str, out: u32, in_: u32, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [u64::from(out), u64::from(in_)], banks),
            bias: None,
        }
    }

    fn packed(name: &str, seams: &[u32], in_: u32, banks: Dtype) -> Linear {
        let out: u64 = seams.iter().map(|&s| u64::from(s)).sum();
        let seams: Vec<u64> = seams.iter().map(|&s| u64::from(s)).collect();
        Linear {
            w: Weight::sym(name, [out, u64::from(in_)], banks).packed(seams.clone()),
            bias: Some(
                Weight::sym(format!("{name}.bias"), [out], crate::dense(banks)).packed(seams),
            ),
        }
    }
}

/// One `LTX2Attention`: `to_q`/`to_k`/`to_v` with biases, the two
/// across-heads RMS gains (`[heads·head_dim]` each —
/// `torch.nn.RMSNorm(inner_dim)`, NOT per head), the per-head gate logits,
/// and the output projection.
///
/// A self-attention packs `q|k|v` into one bank ([`Attn::kv`] `None`); a
/// cross-attention keeps `q` alone and packs `k|v` off the other
/// rectangle, because the two read different widths.
pub struct Attn {
    /// `to_q`, or the packed `to_q|to_k|to_v` of a self-attention.
    pub qkv: Linear,
    /// The packed `to_k|to_v` of a cross-attention.
    pub kv: Option<Linear>,
    pub q_norm: Weight,
    pub k_norm: Weight,
    /// `to_gate_logits`: `[heads, query_dim]` — one logit per head.
    pub gate: Linear,
    pub out: Linear,
    pub heads: u32,
    pub head_dim: u32,
}

impl Attn {
    /// A self-attention over a `dim`-wide rectangle.
    fn own(prefix: &str, dim: u32, heads: u32, head_dim: u32, banks: Dtype) -> Attn {
        let inner = heads * head_dim;
        Attn {
            qkv: Linear::packed(&format!("{prefix}.qkv"), &[inner, inner, inner], dim, banks),
            kv: None,
            q_norm: gain(&format!("{prefix}.q_norm"), inner, banks),
            k_norm: gain(&format!("{prefix}.k_norm"), inner, banks),
            gate: Linear::at(&format!("{prefix}.gate"), heads, dim, banks),
            out: Linear::at(&format!("{prefix}.out"), dim, inner, banks),
            heads,
            head_dim,
        }
    }

    /// A cross-attention: queries off a `dim`-wide rectangle, keys and
    /// values off a `ctx`-wide one, the answer projected back to `dim`.
    fn cross(prefix: &str, dim: u32, ctx: u32, heads: u32, head_dim: u32, banks: Dtype) -> Attn {
        let inner = heads * head_dim;
        Attn {
            qkv: Linear::at(&format!("{prefix}.q"), inner, dim, banks),
            kv: Some(Linear::packed(
                &format!("{prefix}.kv"),
                &[inner, inner],
                ctx,
                banks,
            )),
            q_norm: gain(&format!("{prefix}.q_norm"), inner, banks),
            k_norm: gain(&format!("{prefix}.k_norm"), inner, banks),
            gate: Linear::at(&format!("{prefix}.gate"), heads, dim, banks),
            out: Linear::at(&format!("{prefix}.out"), dim, inner, banks),
            heads,
            head_dim,
        }
    }

    /// `head_dim^-0.5` — `LTX2Attention` passes no `softmax_scale`.
    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    /// This attention's inner width.
    #[must_use]
    pub const fn inner(&self) -> u32 {
        self.heads * self.head_dim
    }
}

fn gain(name: &str, width: u32, banks: Dtype) -> Weight {
    Weight::sym(name, [u64::from(width)], crate::dense(banks))
}

/// `LTX2FeedForward`: `proj_in` up, GELU (tanh), `proj_out` down.
pub struct Ffn {
    pub up: Linear,
    pub down: Linear,
}

impl Ffn {
    fn at(prefix: &str, dim: u32, mult: u32, bias: bool, banks: Dtype) -> Ffn {
        let inner = dim * mult;
        let make = |name: String, out, in_| {
            if bias {
                Linear::at(&name, out, in_, banks)
            } else {
                Linear::plain(&name, out, in_, banks)
            }
        };
        Ffn {
            up: make(format!("{prefix}.up"), inner, dim),
            down: make(format!("{prefix}.down"), dim, inner),
        }
    }
}

/// One stream's side of a `LTX2TransformerBlock`.
pub struct Side {
    /// `scale_shift_table` `[9, dim]` as one `[9·dim]` f32 bias, in the
    /// plan's slice order (every `(shift, scale)` pair exchanged, so
    /// `elementwise.modulate` reads the `[s | b]` it wants).
    pub table: Weight,
    /// The first four rows of `*_a2v_cross_attn_scale_shift_table`
    /// `[5, dim]`: the a2v and v2a `(scale, shift)` pairs, which the
    /// checkpoint already stores scale-first.
    pub av_ss_table: Weight,
    /// Its fifth row: the cross-modal gate, which adds a DIFFERENT global
    /// adaLN head's vector and so is its own plane here.
    pub av_gate_table: Weight,
    /// `prompt_scale_shift_table` `[2, dim]`, the pair exchanged.
    pub prompt_table: Weight,
    pub self_attn: Attn,
    /// The prompt cross-attention (`attn2`).
    pub cross: Attn,
    pub ffn: Ffn,
}

/// One `LTX2TransformerBlock`: two streams' sides plus the two cross-modal
/// attentions they share.
pub struct Block {
    pub video: Side,
    pub audio: Side,
    /// `audio_to_video_attn`: queries off the video rows, keys and values
    /// off the audio rows, the answer folded into the video stream.
    pub a2v: Attn,
    /// `video_to_audio_attn`: the other way about.
    pub v2a: Attn,
}

impl Block {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Block {
        let (dim, adim) = (d.dim(), d.audio_dim());
        let table = |name: String, slices: u32, width: u32| {
            Weight::sym(name, [u64::from(slices * width)], Dtype::F32)
        };
        let side =
            |stem: String, width: u32, heads: u32, head_dim: u32, ctx: u32, bias: bool| Side {
                table: table(format!("{stem}.table"), MOD_SLICES, width),
                av_ss_table: table(format!("{stem}.av_ss_table"), AV_SS_SLICES, width),
                av_gate_table: table(format!("{stem}.av_gate_table"), AV_GATE_SLICES, width),
                prompt_table: table(format!("{stem}.prompt_table"), PROMPT_SLICES, width),
                self_attn: Attn::own(&format!("{stem}.self"), width, heads, head_dim, banks),
                cross: Attn::cross(&format!("{stem}.cross"), width, ctx, heads, head_dim, banks),
                ffn: Ffn::at(&format!("{stem}.ffn"), width, d.ff_mult, bias, banks),
            };
        Block {
            video: side(
                format!("{prefix}.video"),
                dim,
                d.heads,
                d.head_dim,
                d.cross_dim,
                false,
            ),
            audio: side(
                format!("{prefix}.audio"),
                adim,
                d.audio_heads,
                d.audio_head_dim,
                d.audio_cross_dim,
                true,
            ),
            a2v: Attn::cross(
                &format!("{prefix}.a2v"),
                dim,
                adim,
                d.audio_heads,
                d.audio_head_dim,
                banks,
            ),
            v2a: Attn::cross(
                &format!("{prefix}.v2a"),
                adim,
                dim,
                d.audio_heads,
                d.audio_head_dim,
                banks,
            ),
        }
    }
}

/// A two-layer MLP: `linear_2(silu(linear_1(x)))` — every
/// `LTX2TimestepEmbedder`.
pub struct Embedder {
    pub linear_1: Linear,
    pub linear_2: Linear,
}

impl Embedder {
    fn at(prefix: &str, in_: u32, dim: u32, banks: Dtype) -> Embedder {
        Embedder {
            linear_1: Linear::at(&format!("{prefix}.1"), dim, in_, banks),
            linear_2: Linear::at(&format!("{prefix}.2"), dim, dim, banks),
        }
    }
}

/// One `LTX2AdaLayerNormSingle`: the sinusoid's MLP, then
/// `linear(silu(·))` into `slices` `dim`-wide rows. Eight of them stand
/// outside the blocks, one per modulation family per stream.
pub struct AdaLn {
    pub embed: Embedder,
    pub proj: Linear,
    pub slices: u32,
}

impl AdaLn {
    fn at(prefix: &str, dim: u32, slices: u32, banks: Dtype) -> AdaLn {
        AdaLn {
            embed: Embedder::at(&format!("{prefix}.emb"), T_FREQ_DIM, dim, banks),
            proj: Linear::at(&format!("{prefix}.proj"), slices * dim, dim, banks),
            slices,
        }
    }
}

/// One stream's own heads: the patchify projection, the nine-row adaLN, the
/// four-row cross-modal scale/shift, the one-row cross-modal gate, the
/// head's `[temb | temb]` projection and its table, and `proj_out`.
pub struct Stream {
    pub patchify: Linear,
    pub adaln: AdaLn,
    pub av_ss: AdaLn,
    pub av_gate: AdaLn,
    /// `adaln.embed.linear_2` stacked twice, `[2·dim, dim]`: this IR has no
    /// column concatenation and the head adds `embedded_timestep` to BOTH
    /// slices of its table.
    pub head_proj: Linear,
    /// The model-level `scale_shift_table` `[2, dim]` as `[2·dim]` f32, the
    /// pair exchanged.
    pub head_table: Weight,
    pub proj_out: Linear,
}

impl Stream {
    fn at(prefix: &str, channels: u32, dim: u32, banks: Dtype) -> Stream {
        Stream {
            patchify: Linear::at(&format!("{prefix}.patchify"), dim, channels, banks),
            adaln: AdaLn::at(&format!("{prefix}.adaln"), dim, MOD_SLICES, banks),
            av_ss: AdaLn::at(&format!("{prefix}.av_ss"), dim, AV_SS_SLICES, banks),
            av_gate: AdaLn::at(&format!("{prefix}.av_gate"), dim, AV_GATE_SLICES, banks),
            head_proj: Linear::at(
                &format!("{prefix}.head_proj"),
                HEAD_SLICES * dim,
                dim,
                banks,
            ),
            head_table: Weight::sym(
                format!("{prefix}.head_table"),
                [u64::from(HEAD_SLICES * dim)],
                Dtype::F32,
            ),
            proj_out: Linear::at(&format!("{prefix}.proj_out"), channels, dim, banks),
        }
    }
}

/// The transformer.
pub struct Dit {
    pub video: Stream,
    pub audio: Stream,
    /// `prompt_adaln_single`: the two rows that modulate the VIDEO text
    /// context. It runs on the context lane, off that lane's own timestep
    /// cell (`forward.rs`).
    pub prompt: AdaLn,
    pub audio_prompt: AdaLn,
    pub blocks: Vec<Block>,
}

/// One `LTX2TransformerBlock1d` of a connector: a scale-free RMSNorm, a
/// gated self-attention with an across-heads QK norm and a one-axis rope,
/// then a scale-free RMSNorm and a GELU-tanh feed-forward — both residual,
/// neither modulated.
pub struct ConnBlock {
    pub attn: Attn,
    pub ffn: Ffn,
}

/// One `LTX2ConnectorTransformer1d` with its input projection.
///
/// The `learnable_registers` table is NOT declared: the reference compacts
/// the unpadded rows to the front and fills the tail with tiles of it, a
/// data-dependent gather the IR has no member for (`forward.rs` states what
/// this reading does instead).
pub struct Connector {
    /// `video_aggregate_embed` / `audio_aggregate_embed`: `[dim, caption·49]`.
    /// The reference scales its input by `sqrt(dim / caption_channels)`
    /// first; that constant is folded into the bank at import, so the
    /// projection here reads the packed row raw.
    pub aggregate: Linear,
    pub blocks: Vec<ConnBlock>,
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
}

impl Connector {
    fn at(
        prefix: &str,
        text_in: u32,
        dim: u32,
        heads: u32,
        layers: u32,
        banks: Dtype,
    ) -> Connector {
        let head_dim = dim / heads;
        Connector {
            aggregate: Linear::at(&format!("{prefix}.aggregate"), dim, text_in, banks),
            blocks: (0..layers)
                .map(|l| ConnBlock {
                    attn: Attn::own(
                        &format!("{prefix}.block.{l}.attn"),
                        dim,
                        heads,
                        head_dim,
                        banks,
                    ),
                    ffn: Ffn::at(
                        &format!("{prefix}.block.{l}.ffn"),
                        dim,
                        CONN_FF_MULT,
                        true,
                        banks,
                    ),
                })
                .collect(),
            dim,
            heads,
            head_dim,
        }
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    /// The one-axis ladder over the whole row.
    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        [self.dim, 0, 0, 0]
    }

    /// `sqrt(dim / caption_channels)`, the rescale the reference applies to
    /// the packed row before this projection — folded into the bank.
    #[must_use]
    pub fn rescale(&self, caption: u32) -> f32 {
        (f64::from(self.dim) / f64::from(caption)).sqrt() as f32
    }
}

/// The whole text.
pub struct Model {
    pub tp: u32,
    /// The dtype the banks are stored in — `Bf16` on every row today.
    pub banks: Dtype,
    pub dims: Dims,
    pub dit: Dit,
    /// The video connector and the audio connector, in that order.
    pub connectors: (Connector, Connector),
}

impl Model {
    /// `Lightricks/LTX-2.5-Diffusers`: the 19 B distilled transformer and
    /// the 3.2 B connectors.
    #[must_use]
    pub fn ltx_2_5(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::ltx_2_5())
    }

    /// The miniature `ltx2_golden.py --mini` writes. What the parity
    /// harness drives.
    #[must_use]
    pub fn mini(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::mini())
    }

    fn new(banks: Dtype, tp: u32, d: Dims) -> Model {
        assert_eq!(
            tp, 1,
            "this text ships one-rank rows; tp {tp} is not a world it states"
        );
        assert_eq!(
            d.rope_dims().iter().sum::<u32>() + 2 * rope_pad(d.dim(), ROPE_AXES),
            d.dim(),
            "the video ladder and its identity pad cover the row"
        );
        assert_eq!(
            rope_pad(d.audio_dim(), AUDIO_ROPE_AXES),
            0,
            "a one-axis ladder pads nothing"
        );
        assert_eq!(
            d.av_inner(),
            d.audio_cross_dim,
            "the cross-modal rope is built at `audio_cross_attention_dim`, which is the \
             pair's inner width"
        );
        let dit = Dit {
            video: Stream::at("dit.video", d.channels, d.dim(), banks),
            audio: Stream::at("dit.audio", d.channels, d.audio_dim(), banks),
            prompt: AdaLn::at("dit.prompt", d.dim(), PROMPT_SLICES, banks),
            audio_prompt: AdaLn::at("dit.audio_prompt", d.audio_dim(), PROMPT_SLICES, banks),
            blocks: (0..d.layers)
                .map(|i| Block::at(&format!("dit.block.{i}"), &d, banks))
                .collect(),
        };
        Model {
            tp,
            banks,
            dims: d,
            dit,
            connectors: (
                Connector::at(
                    "connectors.video",
                    d.text_in(),
                    d.cross_dim,
                    d.heads,
                    d.conn_layers,
                    banks,
                ),
                Connector::at(
                    "connectors.audio",
                    d.text_in(),
                    d.audio_cross_dim,
                    d.audio_heads,
                    d.conn_layers,
                    banks,
                ),
            ),
        }
    }
}

/// How many identity slots the reference concatenates in front of a
/// `dim`-wide ladder over `axes` axes: `dim/2 − axes·(dim/(2·axes))`. Two
/// for a 4096-wide row over three axes, none for one axis.
#[must_use]
pub const fn rope_pad(dim: u32, axes: u8) -> u32 {
    let axes = axes as u32;
    dim / 2 - axes * (dim / (2 * axes))
}
