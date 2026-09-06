//! The `minimax_h3` declaration: every dimension a Rust constant or a
//! [`Dims`] field, every weight named in the plan's own scheme.
//!
//! Three components under one plan (design D5): the Qwen3-VL-32B text
//! encoder ([`TextEncoder`], `te.`), the H3 transformer ([`Dit`], `dit.`)
//! and — declared but not yet traced — the two autoencoders. The numbers
//! are the snapshot's `FL2VA/transformer/config.json` and
//! `FL2VA/text_encoder/config.json` (study §D.1, §A.2), restated here
//! because a family's dims are Rust constants and a `config.json` is
//! carried, never read. The miniature is the study's §D.3 configuration,
//! which `scripts/imagegen/h3_golden.py --mini` builds and dumps.
//!
//! # The 13 B adaLN bank, and how this text says the gather
//!
//! The reference's modulation is ONE `Linear(2688 -> 18·5376)` per block
//! whose `[M, 18·H]` answer is viewed `[3M, 6H]` and gathered per ROW by
//! `combined = 3·timestep_index + modality_tag`
//! (`runtime/models/dits/minimax_h3.py:1181-1238, 2491-2497`). Both halves
//! of that index are constant over a LANE in this IR, so neither needs a
//! new op (design D2 rejected a per-row modality selector):
//!
//! * the **modality** picks which THIRD of the bank's rows a lane's arm
//!   multiplies — `view(M·3, 6H)` cuts `[18H, 2688]` into three
//!   `[6H, 2688]` row blocks, and a row block is a weight. So
//!   [`Block::adaln`] is three biased linears, one per modality
//!   ([`modality`]), and each stream's arm reads its own;
//! * the **timestep index** picks which column of the `[Lanes, 4]`
//!   `timestep` port a lane's arm slices before the sinusoid
//!   ([`timestep_slot`]) — the port carries the step's up-to-four unique
//!   timesteps (`MINIMAX_H3_ADALN_MAX_PLAN_WIDTH = 4`), which is what the
//!   reference hands `TimeEmbedder` as `unique_timesteps[M]`.
//!
//! The four `[Lanes_s, 6H]` answers are `Value::merge`d back into one
//! `[Lanes, 6H]` rectangle under the reading's guard, so the fifty blocks
//! run on ONE arm with one weight set and one `Modulate` per sublayer,
//! broadcast by `request_of_token`. That merge IS the gather.

use model_dsl::{Dtype, Stream, Weight};

/// `latents_dim`: the video VAE's latent channels.
pub const LATENT_CHANNELS: u32 = 24;
/// `audio_latents_dim`: one audio row's width, a 40 Hz stereo latent.
pub const AUDIO_CHANNELS: u32 = 32;
/// `patch_size` — `(1, 2, 2)` over the video latent.
pub const PATCH_T: u32 = 1;
pub const PATCH_H: u32 = 2;
pub const PATCH_W: u32 = 2;
/// One video row's width: `latents_dim · patch_t · patch_h · patch_w`.
pub const VIDEO_FEATURES: u32 = LATENT_CHANNELS * PATCH_T * PATCH_H * PATCH_W;
/// The video VAE's compression: `f16 t4`.
pub const SPATIAL_COMPRESSION: u32 = 16;
pub const TEMPORAL_COMPRESSION: u32 = 4;

/// The attention head width on every row of this family, miniature
/// included in spirit but not in fact — the miniature halves it so the
/// rope's 3·(2·`rope_freqs`) span still fits (`attention.ragged` serves
/// 64/128/256).
pub const HEAD_DIM: u32 = 128;

/// `norm_eps` = `qk_norm_eps` = `final_norm_eps`.
pub const NORM_EPS: f32 = 1e-5;

/// The sinusoidal timestep embedding: `timestep_input_dim` wide, base
/// 10000, `[cos | sin]` (`minimax_h3.py:583-590`) — diffusers'
/// `flip_sin_to_cos = True` at `downscale_freq_shift = 0`. The timestep
/// reaching the port is `t = 1 − σ ∈ [0, 1]`, so the plan scales by 1.
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;

/// The rotary base read off `rope.inv_freq` in the shipped checkpoint:
/// `inv_freq[i] = 10000^(-2i/32)` to seven digits (study §L.1 asked; the
/// snapshot answers).
pub const ROPE_THETA: f32 = 10_000.0;
/// Three axes `(t, h, w)`; the fourth `RopeAxes` slot is unused.
pub const ROPE_AXES: u8 = 3;

/// How many `[dim]` slices one block's modulation carries per modality:
/// `(shift, scale, gate)` for attention, then for the MLP — six, chunked
/// out of a `6·H` row (`MiniMaxH3AdalnProj.split_output`).
pub const ADALN_SLICES: u32 = 6;
/// Modalities the adaLN bank is cut into: visual, text, audio
/// (`MINIMAX_H3_ADALN_MODALITY_NUM`).
pub const MODALITIES: u32 = 3;
/// The final layer's `(shift, scale)` — one modality, indexed by the
/// timestep alone.
pub const FINAL_SLICES: u32 = 2;

/// The most distinct timesteps one step carries, and so the width of the
/// `timestep` port: video, audio, the visual condition rows' pinned
/// `max(t_video, 0.999)`, and (ref2va) the audio references' `1.0`
/// (`MINIMAX_H3_ADALN_MAX_PLAN_WIDTH`).
pub const TIMESTEP_SLOTS: u32 = 4;

/// The training-step scale of the timestep axis. H3 feeds the model
/// `t = 1 − σ` in `[0, 1]` directly — there is no `·1000` anywhere in
/// `denoise_loop.py` — so the schedule fact states 1.
pub const TRAIN_STEPS: u32 = 1;

/// The two sigma shifts one request runs at once
/// (`model_index.json::_minimax_h3.sigma_shift_scales`).
pub const VIDEO_SHIFT: f32 = 12.0;
pub const AUDIO_SHIFT: f32 = 3.0;
/// `num_inference_steps` — the grid has this many points and the loop
/// runs one fewer evaluation.
pub const STEPS: u32 = 50;
/// The timestep the noise-augmented keyframe / reference latent rows are
/// pinned at every step (`denoise_loop.py:30-32, :437`).
pub const CONDITION_TIMESTEP: f32 = 0.999;

/// The text encoder's numbers (`text_encoder/config.json`, the
/// `text_config` half of Qwen3-VL-32B).
pub const TE_HIDDEN: u32 = 5120;
pub const TE_VOCAB: u32 = 151_936;
pub const TE_Q_HEADS: u32 = 64;
pub const TE_KV_HEADS: u32 = 8;
pub const TE_HEAD_DIM: u32 = 128;
pub const TE_INTER: u32 = 25_600;
pub const TE_THETA: f32 = 5_000_000.0;
pub const TE_EPS: f32 = 1e-6;
/// `num_hidden_layers` of the shipped encoder.
pub const TE_DEPTH: u32 = 64;
/// The conditioning is the residual LEAVING layer 49 — sglang builds the
/// encoder with `num_hidden_layers = 50` and replaces the final norm with
/// `nn.Identity` (`encoders/minimax_h3_qwen3vl.py:46-52, 281`), so this
/// plan runs fifty of the sixty-four layers and reads no norm and no head.
pub const TE_LAYERS: u32 = 50;
/// `max_position_embeddings`; the truncation bound of a prompt.
pub const TE_MAX_TOKENS: u32 = 262_144;

/// The float ports this text reads, by index within their kind and
/// reading. A port index is the family's own and is the position among
/// ports of one kind in the reading's `ReadingFact::ports`, which is how
/// the runtime resolves `input(name)`; every index here is stated on the
/// `PortFact` too (`at`), so the two readings of a kind cannot drift.
pub mod port {
    /// `denoise`: the target video rows, `[rows, 96]` bf16.
    pub const LATENTS: u8 = 0;
    /// `denoise`: the clean keyframe / reference latent rows, `[rows, 96]`
    /// bf16 — the same width as [`LATENTS`] and a rectangle of its own,
    /// because the two streams' rows are two channels a guest feeds
    /// independently.
    pub const REFERENCE: u8 = 1;
    /// `denoise`: the audio rows, `[rows, 32]` bf16.
    pub const AUDIO: u8 = 2;
    /// `denoise`: the refined text rows, `[rows, dim]` bf16 — the
    /// `refine` reading's readout. A Latents port, not a Context one: the
    /// plan's Context 0 is `refine`'s RAW `[rows, 5120]` caption and a
    /// `(kind, index)` pair is seated once per plan at one width
    /// (`IMAGEGEN_CONTRACT.md` §7).
    pub const CONTEXT: u8 = 3;
    /// `refine`: the encoder's `[rows, 5120]` hidden rows.
    pub const CAPTION: u8 = 0;
    /// `denoise`: the step's unique timesteps, `[lanes, 4]` f32.
    pub const TIMESTEP: u8 = 0;
    /// `denoise`: the `(t, h, w)` rotary coordinates, `[rows, 3]` f32.
    pub const POSITIONS: u8 = 0;
}

/// The adaLN modality tag a stream's rows carry
/// (`presentation.py:26-27`, `packed_sequence.py`): visual 0, text 1,
/// audio 2. Every row of a lane shares it, which is what lets a row block
/// of the bank stand in for the gather.
#[must_use]
pub const fn modality(stream: Stream) -> usize {
    match stream {
        // The target video rows and the clean condition rows are both
        // "visual" to the modulation.
        Stream::Video | Stream::Reference | Stream::Image => 0,
        Stream::Audio => 2,
        // Text and context rows take the text vectors.
        Stream::Text | Stream::Context => 1,
    }
}

/// Which column of the `[Lanes, 4]` `timestep` port a stream's lane
/// slices — the reference's `inverse_indices` for that row class
/// (`denoise_loop.py:369-418`): the target video rows and the text rows
/// share the video timestep, the condition rows take the pinned
/// `max(t_video, 0.999)`, the audio rows their own schedule's.
///
/// Slot 3 is the ref2va audio-reference timestep (`1.0`); no stream of
/// the FL2VA partition claims it, and a second audio lane wanting it is
/// the recorded follow-up (a lane cannot be told from its stream mate
/// here, so ref2va needs either a fifth stream or a per-lane slot port).
#[must_use]
pub const fn timestep_slot(stream: Stream) -> u32 {
    match stream {
        Stream::Video | Stream::Text | Stream::Context => 0,
        Stream::Reference | Stream::Image => 1,
        Stream::Audio => 2,
    }
}

/// One row's shape: the numbers that differ between the shipped
/// transformer and the miniature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    /// `hidden_size`.
    pub dim: u32,
    /// `num_attention_heads`, ALREADY divided by the world width.
    pub heads: u32,
    /// `attention_head_dim`.
    pub head_dim: u32,
    /// `ffn_hidden_size`, already divided by the world width.
    pub inter: u32,
    /// `num_layers`.
    pub blocks: u32,
    /// `token_refiner_num_layers`.
    pub refiners: u32,
    /// `text_dim`: the width of the encoder's conditioning rows.
    pub text_dim: u32,
    /// `timestep_input_dim`: the sinusoid's width.
    pub t_freq: u32,
    /// `time_embed_hidden_size`: the timestep MLP's inner width.
    pub t_hidden: u32,
    /// `time_embed_dim`: the adaLN input's width.
    pub t_dim: u32,
    /// `rope_inv_freq_len`: frequencies per rotary axis. Each axis owns
    /// `2 ·` this many CHANNELS, and the three of them cover
    /// `6 ·` this many of `head_dim`.
    pub rope_freqs: u32,
}

impl Dims {
    /// `MiniMaxAI/MiniMax-H3`'s `FL2VA/transformer/config.json`, at a
    /// world width of `tp` ranks.
    #[must_use]
    pub const fn h3(tp: u32) -> Dims {
        Dims {
            dim: 5376,
            heads: 56 / tp,
            head_dim: HEAD_DIM,
            inter: 14336 / tp,
            blocks: 50,
            refiners: 2,
            text_dim: TE_HIDDEN,
            t_freq: 256,
            t_hidden: 5376,
            t_dim: 2688,
            rope_freqs: 16,
        }
    }

    /// The study's §D.3 miniature, which `h3_golden.py --mini` builds:
    /// two blocks, one refiner, 64-wide heads and eight frequencies per
    /// axis (`6 · 8 = 48` of 64 rotated).
    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            dim: 128,
            heads: 2,
            head_dim: 64,
            inter: 256,
            blocks: 2,
            refiners: 1,
            text_dim: 64,
            t_freq: 32,
            t_hidden: 128,
            t_dim: 64,
            rope_freqs: 8,
        }
    }

    /// The inner attention width, `heads · head_dim` (per rank).
    #[must_use]
    pub const fn inner(&self) -> u32 {
        self.heads * self.head_dim
    }

    /// `head_dim^-0.5` — the reference's `softmax_scale`.
    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    /// The rotary channel count per axis, and the whole rotated span.
    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        let per = 2 * self.rope_freqs;
        [per, per, per, 0]
    }

    #[must_use]
    pub const fn rotary_dim(&self) -> u32 {
        6 * self.rope_freqs
    }

    /// One block's adaLN row block: the six `[dim]` slices of one modality.
    #[must_use]
    pub const fn adaln_width(&self) -> u32 {
        ADALN_SLICES * self.dim
    }
}

/// One `nn.Linear` with a bias.
pub struct Linear {
    pub w: Weight,
    pub bias: Weight,
}

impl Linear {
    fn at(name: &str, out: u64, inp: u64, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [out, inp], banks),
            bias: Weight::sym(format!("{name}.bias"), [out], crate::dense(banks)),
        }
    }
}

/// A self-attention's projections: one packed `q|k|v` (the checkpoint's
/// per-head interleave undone at import), the per-head QK RMS gains, the
/// output projection. No bias anywhere.
pub struct Attn {
    pub qkv: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Weight,
}

impl Attn {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Attn {
        let dense = crate::dense(banks);
        let (dim, inner) = (u64::from(d.dim), u64::from(d.inner()));
        Attn {
            qkv: Weight::sym(format!("{prefix}.qkv"), [3 * inner, dim], banks)
                .packed([inner, inner, inner]),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(d.head_dim)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(d.head_dim)], dense),
            out: Weight::sym(format!("{prefix}.out"), [dim, inner], banks),
        }
    }
}

/// `MiniMaxH3MLP`: `fc1` lands `[gate | up]` in one plane, `fc2` brings it
/// back. No biases.
pub struct Mlp {
    pub fc1: Weight,
    pub fc2: Weight,
}

impl Mlp {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Mlp {
        let (dim, inter) = (u64::from(d.dim), u64::from(d.inter));
        Mlp {
            fc1: Weight::sym(format!("{prefix}.fc1"), [2 * inter, dim], banks)
                .packed([inter, inter]),
            fc2: Weight::sym(format!("{prefix}.fc2"), [dim, inter], banks),
        }
    }
}

/// One `MiniMaxH3DiTBlock`: two RMSNorms, the joint attention, the MLP,
/// and the adaLN bank as its three modality row blocks (see the module
/// header). Each block's slice order is the plan's `[scale | shift]` per
/// pair (the checkpoint's is `[shift | scale]`; `import.rs` swaps), so
/// `ModulateForm::ScaleShift` reads it whole.
pub struct Block {
    pub norm1: Weight,
    pub norm2: Weight,
    pub attn: Attn,
    pub mlp: Mlp,
    /// Indexed by [`modality`]: visual, text, audio.
    pub adaln: [Linear; MODALITIES as usize],
}

impl Block {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Block {
        let width = u64::from(d.adaln_width());
        let t_dim = u64::from(d.t_dim);
        Block {
            norm1: Weight::sym(
                format!("{prefix}.norm1"),
                [u64::from(d.dim)],
                crate::dense(banks),
            ),
            norm2: Weight::sym(
                format!("{prefix}.norm2"),
                [u64::from(d.dim)],
                crate::dense(banks),
            ),
            attn: Attn::at(&format!("{prefix}.attn"), d, banks),
            mlp: Mlp::at(&format!("{prefix}.mlp"), d, banks),
            adaln: std::array::from_fn(|m| {
                Linear::at(&format!("{prefix}.adaln.{m}"), width, t_dim, banks)
            }),
        }
    }
}

/// One `MiniMaxH3TokenRefinerBlock`: the same block without adaLN and
/// without rope, over the text rows alone.
pub struct Refiner {
    pub norm1: Weight,
    pub norm2: Weight,
    pub attn: Attn,
    pub mlp: Mlp,
}

impl Refiner {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Refiner {
        Refiner {
            norm1: Weight::sym(
                format!("{prefix}.norm1"),
                [u64::from(d.dim)],
                crate::dense(banks),
            ),
            norm2: Weight::sym(
                format!("{prefix}.norm2"),
                [u64::from(d.dim)],
                crate::dense(banks),
            ),
            attn: Attn::at(&format!("{prefix}.attn"), d, banks),
            mlp: Mlp::at(&format!("{prefix}.mlp"), d, banks),
        }
    }
}

/// The transformer.
pub struct Dit {
    /// `video_patch_proj`: `[dim, 96]`, the (1,2,2) patch of the latent.
    pub video_patch: Linear,
    /// `audio_patch_proj`: `[dim, 32]`.
    pub audio_patch: Linear,
    /// `condition_proj`: `[dim, text_dim]`, the encoder's rows into the
    /// trunk's width, ahead of the refiner blocks.
    pub condition: Linear,
    /// `time_embedder.proj_in` / `.proj_out`.
    pub t_in: Linear,
    pub t_out: Linear,
    pub refine: Vec<Refiner>,
    /// `token_refiner.final_norm`.
    pub refine_norm: Weight,
    pub blocks: Vec<Block>,
    /// `final_layer.norm`.
    pub final_norm: Weight,
    /// `final_layer.adaln_proj.linear`: `[2·dim, t_dim]`, one modality,
    /// in the plan's `[scale | shift]` order.
    pub final_adaln: Linear,
    /// `final_layer.video_out`: `[96, dim]`.
    pub video_out: Linear,
    /// `final_layer.audio_out`: `[32, dim]`.
    pub audio_out: Linear,
}

/// One Qwen3-VL text decoder layer: plain (ungated) GQA attention with
/// per-head QK RMSNorm and full neox rotary, a SwiGLU MLP.
pub struct TeLayer {
    pub attn_norm: Weight,
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub o: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub mlp_norm: Weight,
    pub gate_up: Weight,
    pub down: Weight,
    /// This layer's kv row.
    pub kv: String,
}

/// The text encoder: Qwen3-VL-32B's LANGUAGE half, headless, cut after
/// layer 49 and read before any final norm.
///
/// **The vision tower is not here.** The shipped encoder is a VLM whose
/// image and video tokens enter the same sequence (and take the visual
/// adaLN tag inside the DiT); this row runs the text-only path, where the
/// three M-RoPE sections carry one and the same position and the rotary
/// is therefore plain neox at `rope_theta` over the whole head — which is
/// what [`super::forward::text_encode`] traces. Wiring
/// `qwen_3::media`'s tower under this prefix, and splitting the text lane
/// so the vision-token rows take modality 0, is the recorded follow-up.
pub struct TextEncoder {
    pub hidden: u32,
    pub vocab: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub theta: f32,
    pub eps: f32,
    pub sm_scale: f32,
    pub embed: Weight,
    pub layers: Vec<TeLayer>,
}

impl TextEncoder {
    fn qwen3_vl_32b(banks: Dtype, tp: u32) -> TextEncoder {
        let dense = crate::dense(banks);
        let hidden = u64::from(TE_HIDDEN);
        let hd = u64::from(TE_HEAD_DIM);
        let inter = u64::from(TE_INTER / tp);
        let q_heads = TE_Q_HEADS / tp;
        let kv_heads = TE_KV_HEADS / tp;
        let layers = (0..TE_LAYERS)
            .map(|l| {
                let n = |s: &str| format!("te.layer.{l}.{s}");
                TeLayer {
                    attn_norm: Weight::sym(n("attn_norm"), [hidden], dense),
                    q: Weight::sym(n("q"), [u64::from(q_heads) * hd, hidden], banks),
                    k: Weight::sym(n("k"), [u64::from(kv_heads) * hd, hidden], banks),
                    v: Weight::sym(n("v"), [u64::from(kv_heads) * hd, hidden], banks),
                    o: Weight::sym(n("o"), [hidden, u64::from(q_heads) * hd], banks),
                    q_norm: Weight::sym(n("q_norm"), [hd], dense),
                    k_norm: Weight::sym(n("k_norm"), [hd], dense),
                    mlp_norm: Weight::sym(n("mlp_norm"), [hidden], dense),
                    gate_up: Weight::sym(n("gate_up"), [2 * inter, hidden], banks)
                        .packed([inter, inter]),
                    down: Weight::sym(n("down"), [hidden, inter], banks),
                    kv: format!("te.kv.{l}"),
                }
            })
            .collect();
        TextEncoder {
            hidden: TE_HIDDEN,
            vocab: TE_VOCAB,
            q_heads,
            kv_heads,
            head_dim: TE_HEAD_DIM,
            inter: TE_INTER / tp,
            theta: TE_THETA,
            eps: TE_EPS,
            sm_scale: (TE_HEAD_DIM as f32).sqrt().recip(),
            embed: Weight::sym("te.embed", [u64::from(TE_VOCAB), hidden], banks),
            layers,
        }
    }
}

/// The whole text.
pub struct Model {
    pub tp: u32,
    /// The dtype the banks are stored in — `Bf16` on every row today.
    pub banks: Dtype,
    /// The kv dtype of the encoder's cache.
    pub kv: Dtype,
    pub dims: Dims,
    pub dit: Dit,
    /// `None` on the miniature: its checkpoint is the transformer alone
    /// and its conditioning rows are random, so it declares no `text`
    /// reading.
    pub te: Option<TextEncoder>,
}

impl Model {
    /// `MiniMaxAI/MiniMax-H3`, the `FL2VA/` partition: the 33.1 B
    /// transformer behind Qwen3-VL-32B's first fifty layers.
    #[must_use]
    pub fn fl2va(banks: Dtype, tp: u32) -> Model {
        Model::new(
            banks,
            tp,
            Dims::h3(tp),
            Some(TextEncoder::qwen3_vl_32b(banks, tp)),
        )
    }

    /// The miniature `h3_golden.py --mini` writes: the transformer at
    /// [`Dims::mini`], random weights, no encoder. What the parity
    /// harness drives.
    #[must_use]
    pub fn mini(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::mini(), None)
    }

    fn new(banks: Dtype, tp: u32, d: Dims, te: Option<TextEncoder>) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4),
            "tp {tp} is not a world this text ships"
        );
        assert_eq!(
            d.rotary_dim() % 2,
            0,
            "each rotary axis owns an even channel count"
        );
        assert!(
            d.rotary_dim() <= d.head_dim,
            "the three rotary axes cover {} of a {}-wide head",
            d.rotary_dim(),
            d.head_dim
        );
        let dense = crate::dense(banks);
        let dim = u64::from(d.dim);
        let dit = Dit {
            video_patch: Linear::at("dit.video_patch", dim, u64::from(VIDEO_FEATURES), banks),
            audio_patch: Linear::at("dit.audio_patch", dim, u64::from(AUDIO_CHANNELS), banks),
            condition: Linear::at("dit.condition", dim, u64::from(d.text_dim), banks),
            t_in: Linear::at(
                "dit.t_in",
                u64::from(d.t_hidden),
                u64::from(d.t_freq),
                banks,
            ),
            t_out: Linear::at(
                "dit.t_out",
                u64::from(d.t_dim),
                u64::from(d.t_hidden),
                banks,
            ),
            refine: (0..d.refiners)
                .map(|i| Refiner::at(&format!("dit.refine.{i}"), &d, banks))
                .collect(),
            refine_norm: Weight::sym("dit.refine_norm", [dim], dense),
            blocks: (0..d.blocks)
                .map(|i| Block::at(&format!("dit.block.{i}"), &d, banks))
                .collect(),
            final_norm: Weight::sym("dit.final_norm", [dim], dense),
            final_adaln: Linear::at(
                "dit.final_adaln",
                u64::from(FINAL_SLICES) * dim,
                u64::from(d.t_dim),
                banks,
            ),
            video_out: Linear::at("dit.video_out", u64::from(VIDEO_FEATURES), dim, banks),
            audio_out: Linear::at("dit.audio_out", u64::from(AUDIO_CHANNELS), dim, banks),
        };
        Model {
            tp,
            banks,
            kv: Dtype::Bf16,
            dims: d,
            dit,
            te,
        }
    }
}
