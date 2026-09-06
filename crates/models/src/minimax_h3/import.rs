//! MiniMax H3's reading of a checkpoint: the only place the checkpoint's
//! own tensor spellings appear.
//!
//! # The repository is a container of two pipelines
//!
//! `MiniMaxAI/MiniMax-H3` is **not** a diffusers pipeline at its root: the
//! root holds `FL2VA/` and `Ref2VA/`, and each of THOSE is a pipeline
//! (its own `model_index.json`, its own `transformer/`, `text_encoder/`,
//! `video_vae/`, `audio_vae/`, `tokenizer/`, `processor/`, and a
//! `_minimax_h3` block naming the partition, its tasks and its two sigma
//! shifts). The import target is therefore the PARTITION directory, and
//! this row reads `FL2VA/` — the `t2va`/`fl2va` half. `Ref2VA/` is the
//! same text over a second 66 GB transformer and is one more constructor
//! away.
//!
//! # The component prefixes
//!
//! `checkpoint::file::diffusers::prefix_of` maps a component FOLDER to a
//! ROLE prefix, and three of this pipeline's four weight-bearing folders
//! were already in that vocabulary: `transformer/` → `dit.`,
//! `text_encoder/` → `te.`, `audio_vae/` → `avae.`. The fourth,
//! `video_vae/`, is this family's addition — mapped to `vae.`, the role
//! prefix the table already gives "the latent autoencoder", because that
//! is exactly what it is (H3's only latent autoencoder; the audio codec
//! has its own role and its own prefix). `processor/` carries no
//! safetensors and is skipped by discovery.
//!
//! # What is not a plain read
//!
//! * **the fused `qkv_proj` is interleaved per head.** The official
//!   safetensors store `[q_h | k_h | v_h]` for each of the 56 heads in
//!   turn (`minimax_h3.py:223-252, 856-921` reorders on load); this text
//!   declares `[Q | K | V]`, so the read strides the checkpoint's
//!   `[3·heads, head_dim·in]` view three times and concatenates —
//!   `_reorder_grouped_qkv_to_qkv` at `heads_per_group = 1`, stated in
//!   the contract algebra;
//! * **the adaLN bank is cut into its three modalities and its pairs are
//!   swapped.** `blocks.N.adaln_proj.linear` is `[18·H, 2688]` whose
//!   `view(3M, 6H)` row blocks ARE the modalities (`model.rs`'s header),
//!   so each of the three `[6H, 2688]` blocks is read as six `[H, 2688]`
//!   row slices in the plan's order: the checkpoint chunks
//!   `(shift, scale, gate)` twice and `ModulateForm::ScaleShift` reads
//!   `[scale | shift]`, so slices 0↔1 and 3↔4 trade places. The final
//!   layer's `[2H, 2688]` swaps its one pair the same way;
//! * **`ff.net.0.proj` order.** The NATIVE `mlp.fc1` is `[gate | up]`,
//!   which is what `mlp_swiglu` reads, so it is a plain read; only a
//!   diffusers-aliased checkpoint would store `[value, gate]`, and this
//!   row does not read one;
//! * **the fp32 islands are cast.** `video_patch_proj`,
//!   `audio_patch_proj`, `time_embedder.*`, `final_layer.video_out` and
//!   `final_layer.audio_out` ship fp32 (the reference keeps them fp32 at
//!   run time); every bank here is declared bf16, so each is a cast. This
//!   is the one numerics decision of this import, and the parity harness
//!   is run against a reference at the same width;
//! * **`rope.inv_freq` is not read.** It is a `[16]` fp32 buffer holding
//!   `10000^(-2i/32)` to seven digits, and this text states the base as a
//!   constant ([`super::model::ROPE_THETA`]) because `RopeAxes` takes a
//!   theta, not a table. `the_minimax_h3_import_reads_the_fl2va_index`
//!   checks the buffer is present and that no plane goes unread for any
//!   other reason.
//! * **the encoder is cut at layer 50 and has no head.** Layers 50…63,
//!   `model.language_model.norm` and `lm_head` are never materialised
//!   (`encoders/minimax_h3_qwen3vl.py:46-52`), and neither is the vision
//!   tower (`model.visual.*`) — this row is text-only.

use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint::types::Encoding;
use checkpoint_dsl::{Builder, Error, extents, stored_encoding};
use model_dsl::{Platform, Weight};

use super::model::{
    ADALN_SLICES, Attn, Block, Dit, Linear, MODALITIES, Mlp, Model, Refiner, TextEncoder,
};

/// Where a checkpoint puts the components.
#[derive(Clone, Copy)]
enum Layout {
    /// One partition of the repository read through
    /// `checkpoint::file::diffusers`: `dit.<transformer name>`,
    /// `te.<text_encoder name>`.
    Diffusers,
    /// The transformer's `state_dict` alone, at the root — the
    /// miniature's one file. No encoder can be read under it.
    Bare,
}

impl Layout {
    fn spelling(self) -> &'static str {
        match self {
            Self::Diffusers => "a MiniMax H3 partition (`dit.`/`te.` prefixes)",
            Self::Bare => "a bare transformer state_dict",
        }
    }

    fn dit(self, tail: &str) -> String {
        match self {
            Self::Diffusers => format!("dit.{tail}"),
            Self::Bare => tail.to_string(),
        }
    }

    fn te(self, tail: &str) -> Option<String> {
        match self {
            Self::Diffusers => Some(format!("te.model.language_model.{tail}")),
            Self::Bare => None,
        }
    }
}

impl Model {
    /// # Errors
    ///
    /// [`Error::Illegible`] when no layout lands every plane this row
    /// declares.
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut refusals: Vec<String> = Vec::new();
        let layouts: &[Layout] = match self.te {
            Some(_) => &[Layout::Diffusers],
            None => &[Layout::Bare, Layout::Diffusers],
        };
        for layout in layouts {
            match self.import_from(src, platform, *layout) {
                Ok(contract) => return Ok(contract),
                Err(why) => refusals.push(format!("as {}, {why}", layout.spelling())),
            }
        }
        Err(Error::Illegible {
            name: "minimax_h3".to_string(),
            detail: format!(
                "no reading of this checkpoint lands every plane this family declares — {}",
                refusals.join("; ")
            ),
        })
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        dit(&mut b, src, &self.dims, &self.dit, layout)?;
        if let Some(te) = &self.te {
            text_encoder(&mut b, te, layout)?;
        }
        Ok(b.build())
    }
}

fn dit(
    b: &mut Builder,
    src: &ztensor::Source,
    d: &super::model::Dims,
    m: &Dit,
    layout: Layout,
) -> Result<(), Error> {
    let at = |tail: &str| layout.dit(tail);

    biased(b, &m.video_patch, &at("video_patch_proj"))?;
    biased(b, &m.audio_patch, &at("audio_patch_proj"))?;
    biased(b, &m.condition, &at("condition_proj"))?;
    biased(b, &m.t_in, &at("time_embedder.proj_in"))?;
    biased(b, &m.t_out, &at("time_embedder.proj_out"))?;

    for (i, block) in m.refine.iter().enumerate() {
        refiner(b, src, d, block, &at(&format!("token_refiner.blocks.{i}")))?;
    }
    b.read(&m.refine_norm, at("token_refiner.final_norm.weight"))?;

    for (i, block) in m.blocks.iter().enumerate() {
        dit_block(b, src, d, block, &at(&format!("blocks.{i}")))?;
    }

    b.read(&m.final_norm, at("final_layer.norm.weight"))?;
    // One `(shift, scale)` pair, read as `[scale | shift]`.
    pairs(
        b,
        &m.final_adaln,
        &at("final_layer.adaln_proj.linear"),
        &[1, 0],
        i64::from(d.dim),
    )?;
    biased(b, &m.video_out, &at("final_layer.video_out"))?;
    biased(b, &m.audio_out, &at("final_layer.audio_out"))?;
    Ok(())
}

/// One `MiniMaxH3DiTBlock`: two norms, the attention, the MLP, and the
/// three modality row blocks of the adaLN bank.
fn dit_block(
    b: &mut Builder,
    src: &ztensor::Source,
    d: &super::model::Dims,
    block: &Block,
    stem: &str,
) -> Result<(), Error> {
    b.read(&block.norm1, format!("{stem}.norm1.weight"))?;
    b.read(&block.norm2, format!("{stem}.norm2.weight"))?;
    attention(b, src, d, &block.attn, &format!("{stem}.attn"))?;
    feedforward(b, &block.mlp, &format!("{stem}.mlp"))?;
    let dim = i64::from(d.dim);
    let bank = format!("{stem}.adaln_proj.linear");
    for (m, w) in block.adaln.iter().enumerate() {
        // `view(M·3, 6H)` cuts the bank's rows into `MODALITIES` blocks of
        // `ADALN_SLICES` slices; within one block the checkpoint chunks
        // `(shift, scale, gate)` twice and the plan wants `[scale |
        // shift]` per pair.
        let base = i64::try_from(m).unwrap_or(0) * i64::from(ADALN_SLICES);
        let order = [base + 1, base, base + 2, base + 4, base + 3, base + 5];
        pairs(b, w, &bank, &order, dim)?;
    }
    debug_assert_eq!(block.adaln.len(), MODALITIES as usize);
    Ok(())
}

/// One `MiniMaxH3TokenRefinerBlock`: the same, without the adaLN bank.
fn refiner(
    b: &mut Builder,
    src: &ztensor::Source,
    d: &super::model::Dims,
    block: &Refiner,
    stem: &str,
) -> Result<(), Error> {
    b.read(&block.norm1, format!("{stem}.norm1.weight"))?;
    b.read(&block.norm2, format!("{stem}.norm2.weight"))?;
    attention(b, src, d, &block.attn, &format!("{stem}.attn"))?;
    feedforward(b, &block.mlp, &format!("{stem}.mlp"))
}

/// The attention's four planes. `qkv_proj` is de-interleaved (see the
/// module header); the rest are plain reads with no bias anywhere.
fn attention(
    b: &mut Builder,
    src: &ztensor::Source,
    d: &super::model::Dims,
    a: &Attn,
    stem: &str,
) -> Result<(), Error> {
    let name = format!("{stem}.qkv_proj.weight");
    let dtype = raw_dtype(src, &name, &a.qkv)?;
    let heads = i64::from(d.heads);
    let group = i64::from(d.head_dim) * i64::from(d.dim);
    let want = extents(&a.qkv);
    b.read_over(&a.qkv, name, |e| {
        // `[3·heads, head_dim · in]`: one row per head's Q, K or V block,
        // in the checkpoint's `[q_h | k_h | v_h]` order.
        let grouped = e.transmute(TensorType::raw(vec![3 * heads, group], dtype));
        Expr::concat(
            0,
            (0..3)
                .map(|which| grouped.clone().stride(0, which, heads, 3))
                .collect(),
        )
        .transmute(TensorType::raw(want, dtype))
    })?;
    b.read(&a.q_norm, format!("{stem}.q_norm.weight"))?;
    b.read(&a.k_norm, format!("{stem}.k_norm.weight"))?;
    b.read(&a.out, format!("{stem}.out_proj.weight"))
}

/// `fc1` lands `[gate | up]` as the native checkpoint stores it; `fc2`
/// brings it back. No biases.
fn feedforward(b: &mut Builder, m: &Mlp, stem: &str) -> Result<(), Error> {
    b.read(&m.fc1, format!("{stem}.fc1.weight"))?;
    b.read(&m.fc2, format!("{stem}.fc2.weight"))
}

/// A biased `nn.Linear`: `<stem>.weight` and `<stem>.bias`.
fn biased(b: &mut Builder, w: &Linear, stem: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{stem}.weight"))?;
    b.read(&w.bias, format!("{stem}.bias"))
}

/// A biased `nn.Linear` whose `[k·block, ·]` rows are re-ordered into
/// `order` — the modulation banks, whose slices the plan spells in
/// another order than the checkpoint chunks them.
///
/// The re-order runs at the checkpoint's own width, under an internal
/// tensor, with the dtype adaptation above it
/// ([`Builder::read_over`]'s contract), so a fp32 bank cast to bf16
/// re-orders before it narrows.
fn pairs(b: &mut Builder, w: &Linear, stem: &str, order: &[i64], block: i64) -> Result<(), Error> {
    let rows = |e: &Expr| {
        Expr::concat(
            0,
            order
                .iter()
                .map(|&slice| e.clone().slice(0, slice * block, block))
                .collect(),
        )
    };
    b.read_over(&w.w, format!("{stem}.weight"), |e| rows(&e))?;
    b.read_over(&w.bias, format!("{stem}.bias"), |e| rows(&e))
}

/// The raw dtype `name` is stored in, refusing a quantized plane by name.
fn raw_dtype(
    src: &ztensor::Source,
    name: &str,
    w: &Weight,
) -> Result<checkpoint::types::DType, Error> {
    match stored_encoding(src, name)? {
        Encoding::Raw(dtype) => Ok(dtype),
        other => Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{name}` is stored {other:?}; a fused qkv bank is a raw plane"),
        }),
    }
}

/// The encoder: Qwen3-VL's `model.language_model.*` names under `te.`, the
/// first `TE_LAYERS` layers only, no final norm, no head, no vision tower.
fn text_encoder(b: &mut Builder, te: &TextEncoder, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| -> Result<String, Error> {
        layout.te(tail).ok_or_else(|| Error::Illegible {
            name: format!("te.{tail}"),
            detail: format!(
                "{} carries no text encoder, and this row declares one",
                layout.spelling()
            ),
        })
    };
    b.read(&te.embed, at("embed_tokens.weight")?)?;
    for (l, w) in te.layers.iter().enumerate() {
        let n = |s: &str| at(&format!("layers.{l}.{s}"));
        b.read(&w.attn_norm, n("input_layernorm.weight")?)?;
        b.read(&w.q, n("self_attn.q_proj.weight")?)?;
        b.read(&w.k, n("self_attn.k_proj.weight")?)?;
        b.read(&w.v, n("self_attn.v_proj.weight")?)?;
        b.read(&w.o, n("self_attn.o_proj.weight")?)?;
        b.read(&w.q_norm, n("self_attn.q_norm.weight")?)?;
        b.read(&w.k_norm, n("self_attn.k_norm.weight")?)?;
        b.read(&w.mlp_norm, n("post_attention_layernorm.weight")?)?;
        b.read_concat(
            &w.gate_up,
            [n("mlp.gate_proj.weight")?, n("mlp.up_proj.weight")?],
        )?;
        b.read(&w.down, n("mlp.down_proj.weight")?)?;
    }
    Ok(())
}
