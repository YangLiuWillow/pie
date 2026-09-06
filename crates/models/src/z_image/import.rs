//! Z-Image's reading of a checkpoint: the only place the checkpoint's own
//! tensor spellings appear.
//!
//! The shipped checkpoint is a diffusers pipeline folder
//! (`Tongyi-MAI/Z-Image-Turbo`), which `checkpoint::file::diffusers` opens as
//! ONE name space with a component prefix per folder: `transformer/` →
//! `dit.`, `text_encoder/` → `te.`, `vae/` → `vae.`. Below the prefix the
//! names are the components' own `state_dict` names, verified against the
//! snapshot's `diffusion_pytorch_model.safetensors.index.json` and
//! `model.safetensors.index.json` (`tests/the_z_image_import_reads_every_dit_tensor_once.rs`
//! re-reads them). The miniature `zimage_golden.py --mini` writes the
//! transformer's `state_dict` alone, unprefixed, in one file:
//! [`Layout::Bare`].
//!
//! What is not a plain read:
//!
//! * `to_q`/`to_k`/`to_v` are fused into one packed `qkv` bank, `w1`/`w3`
//!   into one `gate_up` (the `read_concat` every family uses);
//! * the Turbo transformer ships **fp32** (24.6 GB) and every bank here is
//!   declared bf16, so each read is a cast — the one dtype decision of this
//!   import. The modulation projections are cast too: the reference runs
//!   them in bf16 under `torch_dtype=bf16`, and their activations stay f32
//!   on this side regardless (a lane-vector chain lands f32);
//! * the two learned pad tokens become `[2·dim, 1]` flag projections (see
//!   `model::Dit::x_pad_mod`), and the `1000` of the time reversal becomes
//!   a `[1]` f32 weight: both stated by the contract algebra (`Fill`,
//!   `Bias`, `Concat`) in the checkpoint's own dtype, since the IR has no
//!   constant op and a `registered` plane is zero-filled;
//! * the encoder's last layer and final norm are not read at all
//!   (`model::TE_LAYERS`).
//!
//! * the VAE (244 bf16 tensors, `AutoencoderKL`'s own names under `vae.`):
//!   a conv kernel `[C_out, C_in, kh, kw]` is read as the natural
//!   `[C_out, C_in·kh·kw]` rectangle (`weight.reshape(C_out, -1)`, a
//!   transmute of the same bytes) under `Weight::conv_taps_major`, which
//!   the CUDA shell relabels tap-major at load (`IMAGEGEN_CONTRACT.md`
//!   §6); conv biases and GroupNorm affines are cast bf16 → f32 (the
//!   kernels take fp32 per-channel planes); the attention projections are
//!   plain biased linears; and `shift_factor` becomes a `[16]` row like
//!   the `1000` above.

use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use checkpoint::types::Encoding;
use checkpoint_dsl::{Builder, Error, extents, stored_encoding};
use model_dsl::{Platform, Weight};

use super::model::{Block, Dit, Linear, Model, T_FLIP, TextEncoder};
use super::vae::{ConvW, Mid, Norm, ResBlock, SHIFT_FACTOR, Vae};

/// Where a checkpoint puts the components.
#[derive(Clone, Copy)]
enum Layout {
    /// A diffusers pipeline folder read through `checkpoint::file::diffusers`:
    /// `dit.<transformer name>`, `te.<text_encoder name>`.
    Diffusers,
    /// The transformer's `state_dict` alone, at the root — the miniature's
    /// one file. No encoder can be read under it.
    Bare,
}

impl Layout {
    fn spelling(self) -> &'static str {
        match self {
            Self::Diffusers => "a diffusers pipeline (`dit.`/`te.` prefixes)",
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
            Self::Diffusers => Some(format!("te.model.{tail}")),
            Self::Bare => None,
        }
    }

    fn vae(self, tail: &str) -> Option<String> {
        match self {
            Self::Diffusers => Some(format!("vae.{tail}")),
            Self::Bare => None,
        }
    }
}

impl Model {
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
            name: "z_image".to_string(),
            detail: format!(
                "no reading of this checkpoint lands every plane this family declares — {}",
                refusals.join("; ")
            ),
        })
    }

    /// The VAE's contract alone, over a diffusers pipeline name space
    /// (`vae.` prefix): what a load of the VAE by itself — the parity gate
    /// `engine-cuda/tests/the_z_image_vae_answers_the_reference` — reads,
    /// with the same reads the whole-model import states.
    ///
    /// # Errors
    ///
    /// [`Error::Illegible`] for a row with no VAE, or any read's refusal.
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
        vae(&mut b, src, v, Layout::Diffusers)?;
        Ok(b.build())
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        dit(&mut b, src, &self.dit, layout)?;
        if let Some(te) = &self.te {
            text_encoder(&mut b, te, layout)?;
        }
        if let Some(v) = &self.vae {
            vae(&mut b, src, v, layout)?;
        }
        Ok(b.build())
    }
}

fn dit(b: &mut Builder, src: &ztensor::Source, m: &Dit, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| layout.dit(tail);

    biased(b, &m.x_embed, &at("all_x_embedder.2-1"))?;
    pad_table(b, src, &m.x_pad_mod, &at("x_pad_token"))?;
    b.read(&m.cap_norm, at("cap_embedder.0.weight"))?;
    biased(b, &m.cap_embed, &at("cap_embedder.1"))?;
    pad_table(b, src, &m.cap_pad_mod, &at("cap_pad_token"))?;
    biased(b, &m.t_mlp0, &at("t_embedder.mlp.0"))?;
    biased(b, &m.t_mlp1, &at("t_embedder.mlp.2"))?;
    // The `[1]` constant `1000`, seeded from one element of a stored plane.
    constant(b, src, &m.t_flip, &at("t_embedder.mlp.0.bias"), T_FLIP)?;

    for (stem, blocks) in [
        ("noise_refiner", &m.noise_refiner),
        ("context_refiner", &m.context_refiner),
        ("layers", &m.layers),
    ] {
        for (i, block) in blocks.iter().enumerate() {
            transformer_block(b, block, &at(&format!("{stem}.{i}")))?;
        }
    }

    biased(
        b,
        &m.final_ada,
        &at("all_final_layer.2-1.adaLN_modulation.1"),
    )?;
    biased(b, &m.final_linear, &at("all_final_layer.2-1.linear"))?;
    Ok(())
}

/// One `ZImageTransformerBlock`'s planes under `stem`.
fn transformer_block(b: &mut Builder, block: &Block, stem: &str) -> Result<(), Error> {
    let n = |s: &str| format!("{stem}.{s}");
    if let Some(ada) = &block.ada {
        biased(b, ada, &n("adaLN_modulation.0"))?;
    }
    b.read(&block.attn_norm1, n("attention_norm1.weight"))?;
    b.read(&block.attn_norm2, n("attention_norm2.weight"))?;
    b.read(&block.ffn_norm1, n("ffn_norm1.weight"))?;
    b.read(&block.ffn_norm2, n("ffn_norm2.weight"))?;
    b.read_concat(
        &block.attn.qkv,
        [
            n("attention.to_q.weight"),
            n("attention.to_k.weight"),
            n("attention.to_v.weight"),
        ],
    )?;
    b.read(&block.attn.q_norm, n("attention.norm_q.weight"))?;
    b.read(&block.attn.k_norm, n("attention.norm_k.weight"))?;
    b.read(&block.attn.out, n("attention.to_out.0.weight"))?;
    b.read_concat(
        &block.mlp.gate_up,
        [n("feed_forward.w1.weight"), n("feed_forward.w3.weight")],
    )?;
    b.read(&block.mlp.down, n("feed_forward.w2.weight"))
}

/// The encoder: `Qwen3Model`'s `model.*` names under `te.`, the first
/// `TE_LAYERS` layers only, no final norm, no head (the checkpoint ties the
/// embedding and ships none).
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

/// The FLUX VAE: `AutoencoderKL`'s `state_dict` under `vae.`, every one
/// of its 244 tensors read once.
fn vae(b: &mut Builder, src: &ztensor::Source, v: &Vae, layout: Layout) -> Result<(), Error> {
    let at = |tail: &str| -> Result<String, Error> {
        layout.vae(tail).ok_or_else(|| Error::Illegible {
            name: format!("vae.{tail}"),
            detail: format!(
                "{} carries no VAE, and this row declares one",
                layout.spelling()
            ),
        })
    };
    // `shift_factor` as a row, seeded from a stored plane's dtype.
    constant(b, src, &v.shift, &at("decoder.conv_in.bias")?, SHIFT_FACTOR)?;

    let d = &v.decoder;
    conv(b, src, &d.conv_in, &at("decoder.conv_in")?)?;
    mid_block(b, src, &d.mid, &at("decoder.mid_block")?)?;
    for (i, block) in d.up.iter().enumerate() {
        for (r, res) in block.resnets.iter().enumerate() {
            resnet(
                b,
                src,
                res,
                &at(&format!("decoder.up_blocks.{i}.resnets.{r}"))?,
            )?;
        }
        if let Some(up) = &block.upsample {
            conv(
                b,
                src,
                up,
                &at(&format!("decoder.up_blocks.{i}.upsamplers.0.conv"))?,
            )?;
        }
    }
    norm(b, &d.norm_out, &at("decoder.conv_norm_out")?)?;
    conv(b, src, &d.conv_out, &at("decoder.conv_out")?)?;

    let e = &v.encoder;
    conv(b, src, &e.conv_in, &at("encoder.conv_in")?)?;
    for (i, block) in e.down.iter().enumerate() {
        for (r, res) in block.resnets.iter().enumerate() {
            resnet(
                b,
                src,
                res,
                &at(&format!("encoder.down_blocks.{i}.resnets.{r}"))?,
            )?;
        }
        if let Some(down) = &block.downsample {
            conv(
                b,
                src,
                down,
                &at(&format!("encoder.down_blocks.{i}.downsamplers.0.conv"))?,
            )?;
        }
    }
    mid_block(b, src, &e.mid, &at("encoder.mid_block")?)?;
    norm(b, &e.norm_out, &at("encoder.conv_norm_out")?)?;
    // `[mean | logvar]` stored; the mean's output rows declared.
    conv_head(
        b,
        src,
        &e.conv_out,
        &at("encoder.conv_out")?,
        v.encoder_out_stored,
    )
}

/// `UNetMidBlock2D`: `resnets.0`, `attentions.0`, `resnets.1`.
fn mid_block(b: &mut Builder, src: &ztensor::Source, m: &Mid, stem: &str) -> Result<(), Error> {
    resnet(b, src, &m.res0, &format!("{stem}.resnets.0"))?;
    let a = &m.attn;
    let n = |s: &str| format!("{stem}.attentions.0.{s}");
    norm(b, &a.norm, &n("group_norm"))?;
    biased(b, &a.q, &n("to_q"))?;
    biased(b, &a.k, &n("to_k"))?;
    biased(b, &a.v, &n("to_v"))?;
    biased(b, &a.out, &n("to_out.0"))?;
    resnet(b, src, &m.res1, &format!("{stem}.resnets.1"))
}

/// `ResnetBlock2D`: two norms, two convs, the `conv_shortcut` where the
/// width changes.
fn resnet(b: &mut Builder, src: &ztensor::Source, r: &ResBlock, stem: &str) -> Result<(), Error> {
    norm(b, &r.norm1, &format!("{stem}.norm1"))?;
    conv(b, src, &r.conv1, &format!("{stem}.conv1"))?;
    norm(b, &r.norm2, &format!("{stem}.norm2"))?;
    conv(b, src, &r.conv2, &format!("{stem}.conv2"))?;
    if let Some(shortcut) = &r.shortcut {
        conv(b, src, shortcut, &format!("{stem}.conv_shortcut"))?;
    }
    Ok(())
}

/// A `GroupNorm`'s affine planes, cast to the f32 the kernel reads.
fn norm(b: &mut Builder, n: &Norm, stem: &str) -> Result<(), Error> {
    b.read(&n.weight, format!("{stem}.weight"))?;
    b.read(&n.bias, format!("{stem}.bias"))
}

/// A `Conv2d`: the kernel `[C_out, C_in, kh, kw]` transmuted to the
/// natural `[C_out, C_in·kh·kw]` rectangle in its stored dtype (the
/// relabelling to tap-major order is the shell's, at load), the bias cast
/// to f32.
fn conv(b: &mut Builder, src: &ztensor::Source, c: &ConvW, stem: &str) -> Result<(), Error> {
    let kernel = format!("{stem}.weight");
    let stored = stored_encoding(src, &kernel)?;
    let Encoding::Raw(dtype) = stored else {
        return Err(Error::Illegible {
            name: c.w.name.clone(),
            detail: format!("`{kernel}` is stored {stored:?}; a conv kernel is a raw plane"),
        });
    };
    b.read_expr(
        &c.w,
        Expr::src(&kernel).transmute(TensorType::raw(extents(&c.w), dtype)),
    )?;
    b.read(&c.bias, format!("{stem}.bias"))
}

/// A `Conv2d` of which the plan declares the FIRST `c.c_out` of `stored`
/// output channels: the kernel transmuted to `[stored, C_in·kh·kw]` and
/// sliced down axis 0, the bias sliced the same way, both cast as
/// [`conv`] casts.
fn conv_head(
    b: &mut Builder,
    src: &ztensor::Source,
    c: &ConvW,
    stem: &str,
    stored: u32,
) -> Result<(), Error> {
    let kernel = format!("{stem}.weight");
    let encoding = stored_encoding(src, &kernel)?;
    let Encoding::Raw(dtype) = encoding else {
        return Err(Error::Illegible {
            name: c.w.name.clone(),
            detail: format!("`{kernel}` is stored {encoding:?}; a conv kernel is a raw plane"),
        });
    };
    let mut natural = extents(&c.w);
    natural[0] = i64::from(stored);
    let rows = i64::from(c.c_out);
    b.read_expr(
        &c.w,
        Expr::src(&kernel)
            .transmute(TensorType::raw(natural, dtype))
            .slice(0, 0, rows),
    )?;
    // `Cast` is a root-only kernel in the contract algebra (a slice under
    // it counts its elements off the whole source), so the bias's slice is
    // its own internal step in the stored dtype and the cast sits over it.
    let bias = format!("{stem}.bias");
    let stored = stored_encoding(src, &bias)?;
    let want = checkpoint_dsl::encoding(c.bias.dtype);
    let head = format!("{}.head", c.bias.name);
    b.push(
        TensorContract::new(
            head.clone(),
            Expr::src(&bias).slice(0, 0, rows),
            extents(&c.bias),
            stored.clone(),
        )
        .internal(),
    );
    let expr = if want == stored {
        Expr::out(head)
    } else {
        Expr::out(head).cast(want.clone())
    };
    b.push(TensorContract::new(
        c.bias.name.clone(),
        expr,
        extents(&c.bias),
        want,
    ));
    Ok(())
}

/// A `nn.Linear`: `<stem>.weight` and `<stem>.bias`.
fn biased(b: &mut Builder, w: &Linear, stem: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{stem}.weight"))?;
    b.read(&w.bias, format!("{stem}.bias"))
}

/// The learned pad token `[1, dim]` as the `[2·dim, 1]` flag projection
/// `model::Dit::x_pad_mod` describes: `−1` down the first `dim` rows, the
/// token down the next `dim`.
///
/// `Bias` is a root-only kernel in the contract algebra, so the `−1` block
/// is its own internal step (a zero fill plus `−1`) and the bank is one
/// affine concatenation over it and the stored token, cast to the declared
/// dtype. The token's `[1, dim]` bytes ARE its `[dim, 1]` bytes.
fn pad_table(b: &mut Builder, src: &ztensor::Source, w: &Weight, token: &str) -> Result<(), Error> {
    let stored = stored_encoding(src, token)?;
    let Encoding::Raw(dtype) = stored.clone() else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{token}` is stored {stored:?}; a pad token is a raw float row"),
        });
    };
    let shape = extents(w);
    let dim = shape[0] / 2;
    let neg = format!("{}.neg", w.name);
    b.push(
        TensorContract::new(
            neg.clone(),
            Expr::fill(0.0, TensorType::raw(vec![dim, 1], dtype)).bias(-1.0),
            vec![dim, 1],
            stored,
        )
        .internal(),
    );
    b.read_expr(
        w,
        Expr::concat(
            0,
            vec![
                Expr::out(neg),
                Expr::src(token).transmute(TensorType::raw(vec![dim, 1], dtype)),
            ],
        ),
    )
}

/// A constant row of `value` at the declared extents and dtype: a zero
/// fill plus `value`, stated in the dtype the checkpoint stores `seed` in
/// (so the base and the Turbo checkpoint derive it the same way), cast
/// where the row wants another.
fn constant(
    b: &mut Builder,
    src: &ztensor::Source,
    w: &Weight,
    seed: &str,
    value: f32,
) -> Result<(), Error> {
    let stored = stored_encoding(src, seed)?;
    let Encoding::Raw(dtype) = stored.clone() else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{seed}` is stored {stored:?}; a constant is stated in a raw dtype"),
        });
    };
    let raw = format!("{}.raw", w.name);
    b.push(
        TensorContract::new(
            raw.clone(),
            Expr::fill(0.0, TensorType::raw(extents(w), dtype)).bias(value),
            extents(w),
            stored.clone(),
        )
        .internal(),
    );
    let want = checkpoint_dsl::encoding(w.dtype);
    let expr = if want == stored {
        Expr::out(raw)
    } else {
        Expr::out(raw).cast(want.clone())
    };
    b.push(TensorContract::new(w.name.clone(), expr, extents(w), want));
    Ok(())
}
