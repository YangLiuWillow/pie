//! LTX-2.5's reading of a checkpoint: the only place the checkpoint's own
//! tensor spellings appear.
//!
//! The shipped checkpoint is a diffusers pipeline folder
//! (`Lightricks/LTX-2.5-Diffusers`), which `checkpoint::file::diffusers`
//! opens as ONE name space with a component prefix per folder:
//! `transformer/` → `dit.` (8 bf16 shards, 19 B) and `connectors/` →
//! `connectors.` (2 shards, 3.2 B; the folder is not in the prefix
//! vocabulary, so it keeps its own name). Below the prefix the names are
//! the components' own `state_dict` names, read off the snapshot's
//! safetensors headers. The miniature `ltx2_golden.py --mini` writes those
//! same prefixed names into one file, so there is ONE reading of a
//! checkpoint here and not two.
//!
//! What is not a plain read:
//!
//! * `attn*.to_{q,k,v}` fuse into one packed `qkv` for a SELF-attention
//!   (weights and biases); a cross-attention keeps `to_q` apart and packs
//!   `to_k|to_v`, because the two read different widths;
//! * every `(shift, scale)` pair is exchanged into the `[scale | shift]`
//!   that `elementwise.modulate` reads — the nine-row block tables and the
//!   nine-slice `time_embed.linear`, the two-row prompt tables and their
//!   two-slice linears, and the model-level head tables. The cross-modal
//!   `scale_shift_table_a2v_ca_*` is NOT exchanged: the checkpoint already
//!   stores it scale-first (`video_a2v_ca_scale, video_a2v_ca_shift,
//!   video_v2a_ca_scale, video_v2a_ca_shift, gate`), and its fifth row —
//!   the gate — is split off into a plane of its own, because it adds a
//!   different global adaLN head's vector than the first four do;
//! * `time_embed.emb.timestep_embedder.linear_2` is stacked twice into
//!   `head_proj` (`model::Stream::head_proj`), so the head reads
//!   `[embedded_timestep | embedded_timestep]` in one projection: this IR
//!   has no column concatenation and the head adds that vector to BOTH
//!   slices of its table;
//! * the connectors' `*_text_proj_in` banks absorb the `sqrt(dim /
//!   caption_channels)` the reference multiplies their INPUT by (the bias
//!   is left alone — `W·(s·x) + b`);
//! * every table is declared f32 and read from its stored bf16 with a cast;
//! * the video VAE decoder (`vae/` → `vae.`, 84 bf16 tensors plus the two
//!   `latents_mean`/`latents_std` buffers; the flagship only): every conv
//!   kernel `[C_out, C_in, 3, 3, 3]` is read as the `[C_out, C_in·27]`
//!   rectangle `Weight::conv_taps_major` declares (a transmute; the shell
//!   relabels at load), conv biases and the two buffers are cast to f32,
//!   `conv_out`'s 48 rows are permuted from the checkpoint's `(c, pw, ph)`
//!   to the shuffle's `(c, ph, pw)` ([`conv_out_rows`]), and the
//!   upsamplers' rows are NOT permuted (the reference's
//!   `reshape(B, -1, s_t, s_h, s_w, ..)` is already the shuffle's order).
//!   One `[128]` zero row is stated for the denormalisation's
//!   `standardize` ([`zero_row`]).
//!
//! Not read at all: `keyframes_abs_pos_embedding` (allocated upstream,
//! zero-initialised, and never consumed by the denoising forward), the
//! connectors' `learnable_registers` (the substitution is not traced —
//! `forward.rs`), `vae.encoder.*` (no encode arm — `model.rs`), and every
//! component this text does not declare (`text_encoder/`, `audio_vae/`,
//! `diffusion_decoder/`, `latent_upsampler/`, `duration_head/`, `vocoder/`,
//! `prompt_enhancer/`).

use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::Encoding;
use checkpoint_dsl::{Builder, Error, extents, stored_encoding};
use model_dsl::{Platform, Weight};

use super::model::{
    AV_GATE_SLICES, AV_SS_SLICES, AdaLn, Attn, Block, Connector, Dims, Dit, Ffn, HEAD_SLICES,
    Linear, MOD_SLICES, Model, PROMPT_SLICES, Side, Stream, VAE_PATCH, VAE_RGB, Vae, VaeConv,
    VaeResnet,
};

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        dit(&mut b, src, &self.dit, &self.dims)?;
        connector(&mut b, &self.connectors.0, "video", self.dims.caption)?;
        connector(&mut b, &self.connectors.1, "audio", self.dims.caption)?;
        if let Some(v) = &self.vae {
            vae(&mut b, src, v)?;
        }
        Ok(b.build())
    }

    /// The VAE decoder's planes alone, for a harness that loads one arm out
    /// of the snapshot (`engine-cuda`'s
    /// `the_ltx_2_vae_answers_the_reference`) without touching the 19 B
    /// transformer beside it.
    pub fn import_vae(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let v = self.vae.as_ref().ok_or_else(|| Error::Illegible {
            name: "vae".to_string(),
            detail: "this row declares no VAE".to_string(),
        })?;
        let mut b = Builder::new(src, self.tp, platform);
        vae(&mut b, src, v)?;
        Ok(b.build())
    }
}

/// The video VAE's decoder side: `AutoencoderKLLTX2Video`'s own names
/// under `vae.` — `latents_mean`/`latents_std`, `decoder.conv_in`, the mid
/// block's resnets, the four up blocks (upsampler first, then resnets),
/// `decoder.conv_out`. Every conv is `<stem>.conv.weight`/`.bias`
/// (`LTX2VideoCausalConv3d` wraps an `nn.Conv3d` called `conv`).
fn vae(b: &mut Builder, src: &ztensor::Source, v: &Vae) -> Result<(), Error> {
    let at = |tail: &str| format!("vae.{tail}");
    b.read(&v.latents_mean, at("latents_mean"))?;
    b.read(&v.latents_std, at("latents_std"))?;
    zero_row(b, &v.zero)?;

    vae_conv(b, src, &v.conv_in, &at("decoder.conv_in"), None)?;
    for (r, res) in v.mid.iter().enumerate() {
        vae_resnet(b, src, res, &at(&format!("decoder.mid_block.resnets.{r}")))?;
    }
    for (i, up) in v.up.iter().enumerate() {
        let stem = at(&format!("decoder.up_blocks.{i}"));
        // `LTX2VideoUpsampler3d.conv` is itself an `LTX2VideoCausalConv3d`, so the
        // kernel sits one `.conv` deeper than a resnet's.
        vae_conv(b, src, &up.upsampler, &format!("{stem}.upsamplers.0.conv"), None)?;
        for (r, res) in up.resnets.iter().enumerate() {
            vae_resnet(b, src, res, &format!("{stem}.resnets.{r}"))?;
        }
    }
    vae_conv(
        b,
        src,
        &v.conv_out,
        &at("decoder.conv_out"),
        Some(conv_out_rows()),
    )
}

fn vae_resnet(
    b: &mut Builder,
    src: &ztensor::Source,
    r: &VaeResnet,
    stem: &str,
) -> Result<(), Error> {
    vae_conv(b, src, &r.conv1, &format!("{stem}.conv1"), None)?;
    vae_conv(b, src, &r.conv2, &format!("{stem}.conv2"), None)
}

/// A decoder conv: the kernel `[C_out, C_in, 3, 3, 3]` transmuted to the
/// declared `[C_out, C_in·27]` (rows gathered first where `rows` says so),
/// and its bias (gathered the same way, cast to f32 by the read).
fn vae_conv(
    b: &mut Builder,
    src: &ztensor::Source,
    c: &VaeConv,
    stem: &str,
    rows: Option<Vec<i64>>,
) -> Result<(), Error> {
    let name = format!("{stem}.conv.weight");
    let stored = stored_encoding(src, &name)?;
    let shape = TensorType::new(extents(&c.w), stored);
    let gathered = rows.clone();
    b.read_over(&c.w, name, move |e| {
        let kernel = match gathered {
            Some(rows) => e.gather(0, rows),
            None => e,
        };
        kernel.transmute(shape)
    })?;
    let bias = format!("{stem}.conv.bias");
    match rows {
        Some(rows) => b.read_over(&c.bias, bias, move |e| e.gather(0, rows)),
        None => b.read(&c.bias, bias),
    }
}

/// `conv_out`'s 48 rows in the shuffle's `(c, ph, pw)` order. The
/// reference's un-patchify is `reshape(B, -1, p_t, p, p, T, H, W)` then
/// `permute(0, 1, 5, 2, 6, 4, 7, 3)`: the LAST `p` (dim 4) lands beside `H`
/// and the first (dim 3) beside `W`, so checkpoint channel `c·16 + a·4 + b`
/// is `(c, pw = a, ph = b)`, and plan row `(c, ph, pw)` is checkpoint row
/// `c·16 + pw·4 + ph`.
fn conv_out_rows() -> Vec<i64> {
    let p = i64::from(VAE_PATCH);
    let mut rows = Vec::with_capacity((VAE_RGB * VAE_PATCH * VAE_PATCH) as usize);
    for c in 0..i64::from(VAE_RGB) {
        for ph in 0..p {
            for pw in 0..p {
                rows.push(c * p * p + pw * p + ph);
            }
        }
    }
    rows
}

/// A `[n]` row of zeros at the declared dtype, stated rather than read: the decoder's
/// `(z − 0)·latents_std` needs a bias plane and the checkpoint has none to
/// offer. A `Fill` is one storage instruction on every serving backend
/// (`z_image`'s `vae.shift` row is built the same way and loads on CUDA).
fn zero_row(b: &mut Builder, w: &Weight) -> Result<(), Error> {
    let want = checkpoint_dsl::encoding(w.dtype);
    let Encoding::Raw(dtype) = want.clone() else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("declared {want:?}; a stated zero row wants a raw dtype"),
        });
    };
    b.push(TensorContract::new(
        w.name.clone(),
        Expr::fill(0.0, TensorType::raw(extents(w), dtype)),
        extents(w),
        want,
    ));
    Ok(())
}

fn dit(b: &mut Builder, src: &ztensor::Source, m: &Dit, d: &Dims) -> Result<(), Error> {
    let at = |tail: &str| format!("dit.{tail}");

    stream(
        b,
        src,
        &m.video,
        &at("proj_in"),
        &at("time_embed"),
        "",
        d.dim(),
    )?;
    stream(
        b,
        src,
        &m.audio,
        &at("audio_proj_in"),
        &at("audio_time_embed"),
        "audio_",
        d.audio_dim(),
    )?;
    // The two prompt heads: two slices, the pair exchanged.
    adaln(b, &m.prompt, &at("prompt_adaln"), d.dim(), &[1, 0])?;
    adaln(
        b,
        &m.audio_prompt,
        &at("audio_prompt_adaln"),
        d.audio_dim(),
        &[1, 0],
    )?;

    for (i, block) in m.blocks.iter().enumerate() {
        transformer_block(b, src, block, &at(&format!("transformer_blocks.{i}")), d)?;
    }
    Ok(())
}

/// One stream's own heads: `proj_in`, the nine-slice adaLN with its
/// doubled head projection and its `[2, dim]` table, the two cross-modal
/// adaLN heads, and `proj_out`.
fn stream(
    b: &mut Builder,
    src: &ztensor::Source,
    s: &Stream,
    proj_in: &str,
    time_embed: &str,
    prefix: &str,
    dim: u32,
) -> Result<(), Error> {
    biased(b, &s.patchify, proj_in)?;
    // The nine slices, `(shift, scale)` exchanged in each of the three sets.
    adaln(b, &s.adaln, time_embed, dim, &[1, 0, 2, 4, 3, 5, 7, 6, 8])?;
    // `linear_2` twice over, for the head's `[temb | temb]`. Stated as ONE
    // gather of its rows twice rather than a concatenation of the tensor
    // with itself: a `Concat` whose two parts are the same whole `Src` does
    // not lower (the executor walks it as a two-row run).
    let l2 = format!("{time_embed}.emb.timestep_embedder.linear_2");
    let rows: Vec<i64> = (0..i64::from(dim)).chain(0..i64::from(dim)).collect();
    let twice = |tail: &str| Expr::src(format!("{l2}.{tail}")).gather(0, rows.clone());
    b.read_expr(&s.head_proj.w, twice("weight"))?;
    b.read_expr(
        s.head_proj
            .bias
            .as_ref()
            .expect("the head projection is biased"),
        twice("bias"),
    )?;
    table(
        b,
        src,
        &s.head_table,
        &format!("dit.{prefix}scale_shift_table"),
        HEAD_SLICES,
        &[1, 0],
    )?;
    // The cross-modal scale/shift and gate heads: no exchange, the
    // checkpoint stores them scale-first.
    adaln(
        b,
        &s.av_ss,
        &format!(
            "dit.av_cross_attn_{stem}_scale_shift",
            stem = av_stem(prefix)
        ),
        dim,
        &[0, 1, 2, 3],
    )?;
    adaln(
        b,
        &s.av_gate,
        &format!(
            "dit.av_cross_attn_{stem}_{gate}_gate",
            stem = av_stem(prefix),
            gate = av_gate(prefix)
        ),
        dim,
        &[0],
    )?;
    biased(b, &s.proj_out, &format!("dit.{prefix}proj_out"))
}

/// The checkpoint's word for a stream in the cross-modal head names.
fn av_stem(prefix: &str) -> &'static str {
    if prefix.is_empty() { "video" } else { "audio" }
}

/// Which direction's gate a stream owns: the video side carries the a2v
/// gate (it is the one folded into the video rows), the audio side v2a.
fn av_gate(prefix: &str) -> &'static str {
    if prefix.is_empty() { "a2v" } else { "v2a" }
}

/// One `LTX2AdaLayerNormSingle`: the two-layer embedder, then the projection
/// whose `dim`-wide slices are reordered into the plan's.
fn adaln(b: &mut Builder, head: &AdaLn, stem: &str, dim: u32, order: &[i64]) -> Result<(), Error> {
    debug_assert_eq!(order.len(), head.slices as usize);
    let emb = format!("{stem}.emb.timestep_embedder");
    biased(b, &head.embed.linear_1, &format!("{emb}.linear_1"))?;
    biased(b, &head.embed.linear_2, &format!("{emb}.linear_2"))?;
    let proj = format!("{stem}.linear");
    b.read_expr(
        &head.proj.w,
        reordered(&format!("{proj}.weight"), order, i64::from(dim), 0),
    )?;
    b.read_expr(
        head.proj.bias.as_ref().expect("an adaLN linear is biased"),
        reordered(&format!("{proj}.bias"), order, i64::from(dim), 0),
    )
}

/// One `LTX2TransformerBlock`'s planes under `stem`.
fn transformer_block(
    b: &mut Builder,
    src: &ztensor::Source,
    block: &Block,
    stem: &str,
    d: &Dims,
) -> Result<(), Error> {
    side(b, src, &block.video, stem, "", d.dim())?;
    side(b, src, &block.audio, stem, "audio_", d.audio_dim())?;
    attention(b, &block.a2v, &format!("{stem}.audio_to_video_attn"))?;
    attention(b, &block.v2a, &format!("{stem}.video_to_audio_attn"))
}

/// One stream's side of a block: its four tables and its three sublayers.
fn side(
    b: &mut Builder,
    src: &ztensor::Source,
    s: &Side,
    stem: &str,
    prefix: &str,
    dim: u32,
) -> Result<(), Error> {
    table(
        b,
        src,
        &s.table,
        &format!("{stem}.{prefix}scale_shift_table"),
        MOD_SLICES,
        &[1, 0, 2, 4, 3, 5, 7, 6, 8],
    )?;
    // `[5, dim]`: the four scale/shift rows and the gate, split into two
    // planes because they add two different global vectors. The shipped
    // checkpoint spells this table the way sglang's module tree does
    // (`<stream>_a2v_cross_attn_scale_shift_table`) and NOT the way the
    // upstream ltx-core one does (`scale_shift_table_a2v_ca_<stream>`,
    // which `LTX2_PARAM_NAMES_MAPPING` rewrites); the four global adaLN
    // heads keep the ltx-core spelling. Read off the snapshot's index.
    let av = format!(
        "{stem}.{stream}_a2v_cross_attn_scale_shift_table",
        stream = av_stem(prefix)
    );
    banded(b, src, &s.av_ss_table, &av, dim, 0, AV_SS_SLICES)?;
    banded(
        b,
        src,
        &s.av_gate_table,
        &av,
        dim,
        i64::from(AV_SS_SLICES),
        AV_GATE_SLICES,
    )?;
    table(
        b,
        src,
        &s.prompt_table,
        &format!("{stem}.{prefix}prompt_scale_shift_table"),
        PROMPT_SLICES,
        &[1, 0],
    )?;
    attention(b, &s.self_attn, &format!("{stem}.{prefix}attn1"))?;
    attention(b, &s.cross, &format!("{stem}.{prefix}attn2"))?;
    feed_forward(b, &s.ffn, &format!("{stem}.{prefix}ff"))
}

/// One `LTX2Attention`: the packed projections, the two across-heads gains,
/// the per-head gate logits, and `to_out.0`.
fn attention(b: &mut Builder, a: &Attn, stem: &str) -> Result<(), Error> {
    match &a.kv {
        None => packed(
            b,
            &a.qkv,
            &[
                format!("{stem}.to_q"),
                format!("{stem}.to_k"),
                format!("{stem}.to_v"),
            ],
        )?,
        Some(kv) => {
            biased(b, &a.qkv, &format!("{stem}.to_q"))?;
            packed(b, kv, &[format!("{stem}.to_k"), format!("{stem}.to_v")])?;
        }
    }
    b.read(&a.q_norm, format!("{stem}.norm_q.weight"))?;
    b.read(&a.k_norm, format!("{stem}.norm_k.weight"))?;
    biased(b, &a.gate, &format!("{stem}.to_gate_logits"))?;
    biased(b, &a.out, &format!("{stem}.to_out.0"))
}

/// `FeedForward`: `net.0.proj` up, `net.2` down.
fn feed_forward(b: &mut Builder, ff: &Ffn, stem: &str) -> Result<(), Error> {
    biased(b, &ff.up, &format!("{stem}.net.0.proj"))?;
    biased(b, &ff.down, &format!("{stem}.net.2"))
}

/// One connector: its input projection (with the reference's input rescale
/// folded in) and its blocks.
fn connector(b: &mut Builder, conn: &Connector, stem: &str, caption: u32) -> Result<(), Error> {
    let _ = caption;
    // A plain read: the reference's `sqrt(dim / caption_channels)` rescale of
    // the INPUT is `s·(W·x) + b` on the other side of the projection, and
    // the plan applies it there (`forward.rs`) — `Expr::scale` needs a
    // kernel the import path cannot lower.
    biased(
        b,
        &conn.aggregate,
        &format!("connectors.{stem}_text_proj_in"),
    )?;
    for (l, block) in conn.blocks.iter().enumerate() {
        let at = format!("connectors.{stem}_connector.transformer_blocks.{l}");
        attention(b, &block.attn, &format!("{at}.attn1"))?;
        feed_forward(b, &block.ffn, &format!("{at}.ff"))?;
    }
    Ok(())
}

/// A `nn.Linear`: `<stem>.weight` and, where the plan declares one,
/// `<stem>.bias`.
fn biased(b: &mut Builder, w: &Linear, stem: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{stem}.weight"))?;
    match &w.bias {
        Some(bias) => b.read(bias, format!("{stem}.bias")),
        None => Ok(()),
    }
}

/// Several `nn.Linear`s fused into one packed bank, weights and biases.
fn packed(b: &mut Builder, w: &Linear, stems: &[String]) -> Result<(), Error> {
    b.read_concat(&w.w, stems.iter().map(|s| format!("{s}.weight")))?;
    b.read_concat(
        w.bias.as_ref().expect("a packed projection is biased"),
        stems.iter().map(|s| format!("{s}.bias")),
    )
}

/// A `scale_shift_table` `[k, dim]` as the `[k·dim]` f32 bias the plan
/// reads, its rows permuted into the plan's slice order.
fn table(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    name: &str,
    slices: u32,
    order: &[i64],
) -> Result<(), Error> {
    debug_assert_eq!(order.len(), slices as usize);
    let stored = stored_encoding(src, name)?;
    b.read_expr(
        w,
        reordered(name, order, 1, 0).transmute(TensorType::new(extents(w), stored)),
    )
}

/// A contiguous band of a `[k, dim]` table as a `[len·dim]` f32 bias, its
/// rows kept in the checkpoint's order.
fn banded(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    name: &str,
    _dim: u32,
    start: i64,
    len: u32,
) -> Result<(), Error> {
    let stored = stored_encoding(src, name)?;
    b.read_expr(
        w,
        Expr::src(name.to_string())
            .slice(0, start, i64::from(len))
            .transmute(TensorType::new(extents(w), stored)),
    )
}

/// The permutation itself: `order.len()` consecutive `width`-wide bands of
/// `axis`, concatenated in the plan's order. An order that permutes nothing
/// is the tensor (a one-slice "cut" would be a slice covering its whole
/// axis, which the contract refuses as a way of saying its operand).
fn reordered(from: &str, order: &[i64], width: i64, axis: u8) -> Expr {
    if order.iter().enumerate().all(|(at, &i)| i as usize == at) {
        return Expr::src(from.to_string());
    }
    let take = |i: i64| Expr::src(from.to_string()).slice(axis, i * width, width);
    Expr::concat(axis, order.iter().map(|&i| take(i)).collect())
}
