//! `AutoencoderKLFlux2` — FLUX.2's autoencoder, as two readings on the
//! voxel axis (design D8, `IMAGEGEN_CONTRACT.md` §6): `vae.decode` takes
//! the DiT's own packed latent and lands pixels, `vae.encode` takes pixels
//! and lands that same packed latent. `vae/config.json` restated as
//! constants: `block_out_channels [128, 256, 512, 512]`,
//! `layers_per_block 2`, `norm_num_groups 32`, `latent_channels 32`,
//! `patch_size (2, 2)`, `batch_norm_eps 1e-4`, `use_quant_conv` and
//! `use_post_quant_conv` both true, `mid_block_add_attention` true, SiLU
//! everywhere. There is **no** `scaling_factor`/`shift_factor` on this
//! family: a frozen `BatchNorm2d(128)` normalises the latent instead.
//!
//! | reading | port (`Voxels`, clip `{1, h, w}`) | lands (`pixels` seam) |
//! |---|---|---|
//! | `vae.decode` | `latent` `[h·w, 128]`, the DiT-space token grid at `/16` | `[16h·16w, 3]` f32 in `[-1, 1]` (clamping and `uint8` are the frames layer's) |
//! | `vae.encode` | `pixels` `[H·W, 3]` in `[-1, 1]` | `[H/16·W/16, 128]`, the BatchNorm-normalised posterior MEAN |
//!
//! # Where the port cuts the model (the 128-wide grid, not the 32-wide one)
//!
//! `AutoencoderKLFlux2` codes at 32 channels on a `/8` grid, and the
//! pipeline packs a 2×2 block of those cells into one 128-channel cell at
//! `/16` — `_patchify_latents` / `_unpatchify_latents`, a plain
//! pixel-unshuffle / pixel-shuffle — and normalises THAT by the frozen
//! `bn`. The transformer's `in_channels` is 128 and its `patch_size` is 1,
//! so the 128-wide `/16` grid is what the denoiser emits and consumes; the
//! 32-wide `/8` latent never leaves the autoencoder. Both readings
//! therefore put the port at the 128-wide grid and keep the shuffle and
//! the BatchNorm INSIDE the plan. A guest hands the denoiser's own rows to
//! `vae.decode` and gets the denoiser's own rows back from `vae.encode`,
//! with no normalisation of its own to state (this is where FLUX.2 differs
//! from Z-Image, whose guest applies `(mean − shift)·scaling` itself).
//!
//! The order is the reference's, and it is not symmetric on the page:
//!
//! ```text
//! encode: pixels → encoder → conv_out 512→64 → quant_conv 64→64
//!         → mean = rows [0, 32) → pixel_unshuffle 2 → 128 ch at /16
//!         → (x − running_mean) / √(running_var + eps)
//! decode: 128 ch at /16 → x·√(running_var + eps) + running_mean
//!         → pixel_shuffle 2 → 32 ch at /8 → post_quant_conv → decoder
//! ```
//!
//! Both readings run over ONE clip per lane — an image, `t = 1` — and the
//! grid is the port's.
//!
//! # Structure (diffusers `vae.py`, the same `Encoder`/`Decoder` Z-Image's
//! `AutoencoderKL` uses)
//!
//! ```text
//! decode: conv_in 32→512 · mid(res, attn, res) · up0(res×3 @512, ↑2 conv)
//!         · up1(res×3 @512, ↑2 conv) · up2(res 512→256, res×2, ↑2 conv)
//!         · up3(res 256→128, res×2) · GroupNorm+SiLU · conv_out 128→3
//! encode: conv_in 3→128 · down0(res×2 @128, ↓2) · down1(res 128→256, res, ↓2)
//!         · down2(res 256→512, res, ↓2) · down3(res×2 @512) · mid
//!         · GroupNorm+SiLU · conv_out 512→64 = [mean | logvar]
//! res:    x + conv2(silu(gn2(conv1(silu(gn1(x)))))), a 1×1 conv on the
//!         skip where the width changes
//! attn:   x + to_out(softmax(q·kᵀ/√512)·v), q/k/v off gn(x), one head
//! ↓2:     F.pad(x, (0, 1, 0, 1)) then conv 3×3 stride 2 (pad_back)
//! ↑2:     nearest ×2 then conv 3×3
//! ```
//!
//! The posterior's `logvar` is never computed: `quant_conv` mixes all 64
//! stored channels, so the slice down to the mean is taken on ITS output
//! rows (the plan declares 32 of the stored 64) rather than on
//! `encoder.conv_out`'s, which stays whole.
//!
//! # Numerics contract
//!
//! Conv weights are bf16 (as stored), activations bf16 between launches,
//! every convolution and projection accumulating fp32 with one rounding at
//! the store; GroupNorm statistics fp32 Welford; the attention fp32
//! scores, softmax and accumulation (`upcast_softmax`). The reference runs
//! the VAE in fp32 (`force_upcast`), so a port differs by one bf16
//! rounding per launch — the parity gate
//! (`scripts/imagegen/flux2_golden.py --vae`) holds at decode `cos ≥
//! 0.999`, encode `cos ≥ 0.9995`.

use model_dsl::ops::spatial::{self, Conv};
use model_dsl::{Dtype, Input, Value, Weight, ops, seam};

use super::forward::Facts;
use super::model::{IN_CHANNELS, Linear, PACK, VAE_CHANNELS, port};

/// `block_out_channels`.
pub const BLOCK_CHANNELS: [u32; 4] = [128, 256, 512, 512];
/// `norm_num_groups`.
pub const GN_GROUPS: u32 = 32;
/// Every GroupNorm's epsilon (`resnet_eps`, and the attention's `eps`).
pub const GN_EPS: f32 = 1e-6;
/// `batch_norm_eps`: the frozen `BatchNorm2d(128)`'s.
pub const BN_EPS: f32 = 1e-4;
/// `layers_per_block`: resnets per encoder block; a decoder block has one more.
pub const LAYERS_PER_BLOCK: u32 = 2;
/// Resnets per decoder block (`layers_per_block + 1`).
pub const DECODER_RESNETS: u32 = LAYERS_PER_BLOCK + 1;
/// `in_channels` / `out_channels`: RGB.
pub const RGB: u32 = 3;
/// `kh·kw` of every 3×3 convolution.
const TAPS3: u32 = 9;
/// What the encoder head stores: `[mean | logvar]`, `2·latent_channels`
/// wide, which `quant_conv` maps to itself before the mean is taken.
pub const POSTERIOR_STORED: u32 = 2 * VAE_CHANNELS;

/// A convolution's plane (`[C_out, C_in·taps]`, tap-major at load) and its
/// `[C_out]` f32 bias.
pub struct ConvW {
    pub w: Weight,
    pub bias: Weight,
    pub c_in: u32,
    pub c_out: u32,
    pub taps: u32,
}

impl ConvW {
    fn at(name: &str, c_out: u32, c_in: u32, taps: u32, banks: Dtype) -> ConvW {
        ConvW {
            w: Weight::sym(
                name,
                [u64::from(c_out), u64::from(c_in) * u64::from(taps)],
                banks,
            )
            .conv_taps_major(c_in, taps),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c_out)], Dtype::F32),
            c_in,
            c_out,
            taps,
        }
    }
}

/// `torch.nn.GroupNorm(32, C)`'s affine planes, `[C]` f32.
pub struct Norm {
    pub weight: Weight,
    pub bias: Weight,
}

impl Norm {
    fn at(name: &str, c: u32) -> Norm {
        Norm {
            weight: Weight::sym(format!("{name}.weight"), [u64::from(c)], Dtype::F32),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c)], Dtype::F32),
        }
    }
}

/// A biased `nn.Linear` — the mid block's four projections, the only
/// biased matmuls this family has (its transformer's are all bias-free).
pub struct Proj {
    pub w: Linear,
    pub bias: Weight,
}

impl Proj {
    fn at(name: &str, c: u32, banks: Dtype, dense: Dtype) -> Proj {
        let c = u64::from(c);
        Proj {
            w: Weight::sym(name.to_string(), [c, c], banks),
            bias: Weight::sym(format!("{name}.bias"), [c], dense),
        }
    }
}

/// `ResnetBlock2D`: `conv_shortcut` only where the width changes.
pub struct ResBlock {
    pub norm1: Norm,
    pub conv1: ConvW,
    pub norm2: Norm,
    pub conv2: ConvW,
    pub shortcut: Option<ConvW>,
}

impl ResBlock {
    fn at(name: &str, c_in: u32, c_out: u32, banks: Dtype) -> ResBlock {
        ResBlock {
            norm1: Norm::at(&format!("{name}.norm1"), c_in),
            conv1: ConvW::at(&format!("{name}.conv1"), c_out, c_in, TAPS3, banks),
            norm2: Norm::at(&format!("{name}.norm2"), c_out),
            conv2: ConvW::at(&format!("{name}.conv2"), c_out, c_out, TAPS3, banks),
            shortcut: (c_in != c_out)
                .then(|| ConvW::at(&format!("{name}.shortcut"), c_out, c_in, 1, banks)),
        }
    }
}

/// The mid block's `Attention`: one head as wide as the row, biased
/// projections, a residual.
pub struct AttnBlock {
    pub norm: Norm,
    pub q: Proj,
    pub k: Proj,
    pub v: Proj,
    pub out: Proj,
    pub width: u32,
}

impl AttnBlock {
    fn at(name: &str, c: u32, banks: Dtype, dense: Dtype) -> AttnBlock {
        AttnBlock {
            norm: Norm::at(&format!("{name}.norm"), c),
            q: Proj::at(&format!("{name}.q"), c, banks, dense),
            k: Proj::at(&format!("{name}.k"), c, banks, dense),
            v: Proj::at(&format!("{name}.v"), c, banks, dense),
            out: Proj::at(&format!("{name}.out"), c, banks, dense),
            width: c,
        }
    }
}

/// `UNetMidBlock2D`: resnet, attention, resnet.
pub struct Mid {
    pub res0: ResBlock,
    pub attn: AttnBlock,
    pub res1: ResBlock,
}

impl Mid {
    fn at(name: &str, c: u32, banks: Dtype, dense: Dtype) -> Mid {
        Mid {
            res0: ResBlock::at(&format!("{name}.res0"), c, c, banks),
            attn: AttnBlock::at(&format!("{name}.attn"), c, banks, dense),
            res1: ResBlock::at(&format!("{name}.res1"), c, c, banks),
        }
    }
}

/// `UpDecoderBlock2D`: resnets, then `Upsample2D` (nearest ×2 + conv) on
/// every block but the last.
pub struct UpBlock {
    pub resnets: Vec<ResBlock>,
    pub upsample: Option<ConvW>,
}

/// `DownEncoderBlock2D`: resnets, then `Downsample2D` (pad + stride-2
/// conv) on every block but the last.
pub struct DownBlock {
    pub resnets: Vec<ResBlock>,
    pub downsample: Option<ConvW>,
}

pub struct Decoder {
    pub conv_in: ConvW,
    pub mid: Mid,
    pub up: Vec<UpBlock>,
    pub norm_out: Norm,
    pub conv_out: ConvW,
}

pub struct Encoder {
    pub conv_in: ConvW,
    pub down: Vec<DownBlock>,
    pub mid: Mid,
    pub norm_out: Norm,
    /// `[2·latent_channels, 512·9]`: `[mean | logvar]` whole, because
    /// `quant_conv` mixes both halves before either is used.
    pub conv_out: ConvW,
}

/// The whole autoencoder.
pub struct Vae {
    /// The BatchNorm denormalisation `z·√(var+eps) + mean` as two ops read
    /// it: `standardize` (`(x − bias)·scale`) with a ZERO bias and `scale
    /// = √(var+eps)`, then `add_bias` of `mean` — all `[128]`, the
    /// deviation derived at import from `bn.running_var`, the zero a fill.
    /// (The one-op form `bias = −mean/√(var+eps)` wants a quotient of two
    /// planes the contract algebra does not state.)
    pub bn_zero: Weight,
    pub bn_scale: Weight,
    /// The same BatchNorm the other way: `standardize(x, bn_mean,
    /// bn_rscale)` IS `(x − mean)/√(var+eps)`, one launch, with
    /// `bn_rscale = (var + eps)^(-1/2)` derived at import
    /// (`UnaryOp::Rsqrt`).
    pub bn_rscale: Weight,
    pub bn_mean: Weight,
    /// `quant_conv`, declared at the mean's 32 output rows of the stored
    /// [`POSTERIOR_STORED`] (sliced at import); its 64 INPUT channels are
    /// the encoder head's whole `[mean | logvar]`.
    pub quant_conv: ConvW,
    pub post_quant_conv: ConvW,
    pub decoder: Decoder,
    pub encoder: Encoder,
}

impl Vae {
    /// `AutoencoderKLFlux2` under `vae.`, its projections in `banks`.
    #[must_use]
    pub fn flux2(banks: Dtype) -> Vae {
        let dense = crate::dense(banks);
        let top = BLOCK_CHANNELS[BLOCK_CHANNELS.len() - 1];
        // The decoder walks the channel list reversed: 512, 512, 256, 128.
        let mut up = Vec::new();
        let mut c_prev = top;
        for (i, &c) in BLOCK_CHANNELS.iter().rev().enumerate() {
            let name = format!("vae.dec.up{i}");
            let resnets = (0..DECODER_RESNETS)
                .map(|r| {
                    let block = ResBlock::at(&format!("{name}.res{r}"), c_prev, c, banks);
                    c_prev = c;
                    block
                })
                .collect();
            let last = i + 1 == BLOCK_CHANNELS.len();
            up.push(UpBlock {
                resnets,
                upsample: (!last)
                    .then(|| ConvW::at(&format!("{name}.upsample"), c, c, TAPS3, banks)),
            });
        }
        let mut down = Vec::new();
        let mut c_prev = BLOCK_CHANNELS[0];
        for (i, &c) in BLOCK_CHANNELS.iter().enumerate() {
            let name = format!("vae.enc.down{i}");
            let resnets = (0..LAYERS_PER_BLOCK)
                .map(|r| {
                    let block = ResBlock::at(&format!("{name}.res{r}"), c_prev, c, banks);
                    c_prev = c;
                    block
                })
                .collect();
            let last = i + 1 == BLOCK_CHANNELS.len();
            down.push(DownBlock {
                resnets,
                downsample: (!last)
                    .then(|| ConvW::at(&format!("{name}.downsample"), c, c, TAPS3, banks)),
            });
        }
        let bn =
            |tail: &str| Weight::sym(format!("vae.bn.{tail}"), [u64::from(IN_CHANNELS)], dense);
        Vae {
            bn_zero: bn("zero"),
            bn_scale: bn("scale"),
            bn_rscale: bn("rscale"),
            bn_mean: bn("mean"),
            quant_conv: ConvW::at("vae.quant", VAE_CHANNELS, POSTERIOR_STORED, 1, banks),
            post_quant_conv: ConvW::at("vae.post_quant", VAE_CHANNELS, VAE_CHANNELS, 1, banks),
            decoder: Decoder {
                conv_in: ConvW::at("vae.dec.conv_in", top, VAE_CHANNELS, TAPS3, banks),
                mid: Mid::at("vae.dec.mid", top, banks, dense),
                up,
                norm_out: Norm::at("vae.dec.norm_out", BLOCK_CHANNELS[0]),
                conv_out: ConvW::at("vae.dec.conv_out", RGB, BLOCK_CHANNELS[0], TAPS3, banks),
            },
            encoder: Encoder {
                conv_in: ConvW::at("vae.enc.conv_in", BLOCK_CHANNELS[0], RGB, TAPS3, banks),
                down,
                mid: Mid::at("vae.enc.mid", top, banks, dense),
                norm_out: Norm::at("vae.enc.norm_out", top),
                conv_out: ConvW::at("vae.enc.conv_out", POSTERIOR_STORED, top, TAPS3, banks),
            },
        }
    }
}

/// The `vae.decode` reading: the denoiser's own `[h·w, 128]` token grid on
/// the `Voxels` port, denormalised by the frozen BatchNorm, unpacked to
/// the autoencoder's 32 channels at `/8`, through the decoder to
/// `[16h·16w, 3]` pixels in `[-1, 1]` on the `pixels` seam beside their
/// grid.
pub fn decode(arm: &Input<Facts>, vae: &Vae) -> Value {
    let d = &vae.decoder;
    let g0 = arm.grid();
    let z = arm.voxels(port::VOXELS, IN_CHANNELS, Dtype::Bf16);
    // `standardize` runs in place and a port's cell is never overwritten,
    // so the rows are copied first — a factor-1 nearest upsample is the
    // one fresh box-keeping copy the voxel axis has.
    let (z, g0) = spatial::upsample_nearest(&z, &g0, [1, 1, 1], false);
    // `latents · √(running_var + eps) + running_mean`, the pipeline's step
    // before `_unpatchify_latents`.
    let z = ops::elemwise::standardize(&z, &vae.bn_zero, &vae.bn_scale);
    let z = ops::elemwise::add_bias(&vae.bn_mean, &z);
    let (z, mut grid) = spatial::pixel_shuffle(&z, &g0, [1, PACK, PACK]);
    let z = conv(&z, &grid, &vae.post_quant_conv);
    let mut h = conv(&z, &grid, &d.conv_in);
    h = mid(&h, &grid, &d.mid);
    for block in &d.up {
        for res in &block.resnets {
            h = resnet(&h, &grid, res);
        }
        if let Some(c) = &block.upsample {
            let (up, up_grid) = spatial::upsample_nearest(&h, &grid, [1, 2, 2], false);
            grid = up_grid;
            h = conv(&up, &grid, c);
        }
    }
    let h = group_norm(&h, &grid, &d.norm_out, true);
    let y = conv(&h, &grid, &d.conv_out);
    seam::at(seam::PIXELS, &[&y, &grid]);
    y
}

/// The `vae.encode` reading: `[H·W, 3]` pixels in `[-1, 1]` on the
/// `Voxels` port through the encoder to the posterior mean at `/8`, packed
/// 2×2 into the denoiser's `[H/16·W/16, 128]` grid and normalised by the
/// frozen BatchNorm, on the `pixels` seam beside its grid.
pub fn encode(arm: &Input<Facts>, vae: &Vae) -> Value {
    let e = &vae.encoder;
    let mut grid = arm.grid();
    let x = arm.voxels(port::PIXEL_VOXELS, RGB, Dtype::Bf16);
    let mut h = conv(&x, &grid, &e.conv_in);
    for block in &e.down {
        for res in &block.resnets {
            h = resnet(&h, &grid, res);
        }
        if let Some(c) = &block.downsample {
            // `Downsample2D`: zeros behind the box on h and w, then 3×3 at
            // stride 2 with no padding of its own.
            let (y, y_grid) = spatial::conv3d(
                &h,
                &grid,
                &c.w,
                Some(&c.bias),
                Conv::conv2d([3, 3], [2, 2], [0, 0]).pad_back([0, 1, 1]),
                None,
            );
            h = y;
            grid = y_grid;
        }
    }
    h = mid(&h, &grid, &e.mid);
    let h = group_norm(&h, &grid, &e.norm_out, true);
    // `[mean | logvar]` through `quant_conv`, of which the plan declares
    // the mean's 32 output rows alone (the logvar is never computed, and
    // FLUX.2 takes the mean, never a sample).
    let h = conv(&h, &grid, &e.conv_out);
    let mean = conv(&h, &grid, &vae.quant_conv);
    // `_patchify_latents` then `(x − running_mean)/√(running_var + eps)`.
    let (packed, grid) = spatial::pixel_unshuffle(&mean, &grid, [1, PACK, PACK]);
    let z = ops::elemwise::standardize(&packed, &vae.bn_mean, &vae.bn_rscale);
    seam::at(seam::PIXELS, &[&z, &grid]);
    z
}

/// A box-keeping 3×3 (or 1×1) convolution with its bias.
fn conv(x: &Value, grid: &Value, c: &ConvW) -> Value {
    let shape = match c.taps {
        1 => Conv::conv2d([1, 1], [1, 1], [0, 0]),
        _ => Conv::conv2d([3, 3], [1, 1], [1, 1]),
    };
    spatial::conv3d(x, grid, &c.w, Some(&c.bias), shape, None).0
}

fn group_norm(x: &Value, grid: &Value, norm: &Norm, silu: bool) -> Value {
    spatial::group_norm(x, grid, GN_GROUPS, &norm.weight, &norm.bias, GN_EPS, silu)
}

/// `ResnetBlock2D`, `output_scale_factor 1`, no time embedding, no dropout.
fn resnet(x: &Value, grid: &Value, r: &ResBlock) -> Value {
    let h = group_norm(x, grid, &r.norm1, true);
    let h = conv(&h, grid, &r.conv1);
    let h = group_norm(&h, grid, &r.norm2, true);
    let h = conv(&h, grid, &r.conv2);
    let skip = match &r.shortcut {
        Some(c) => conv(x, grid, c),
        None => x.clone(),
    };
    ops::elemwise::add(&skip, &h)
}

/// `UNetMidBlock2D`.
fn mid(x: &Value, grid: &Value, m: &Mid) -> Value {
    let h = resnet(x, grid, &m.res0);
    let h = attention(&h, grid, &m.attn);
    resnet(&h, grid, &m.res1)
}

/// The mid block's `Attention(heads=1, dim_head=C, residual_connection=True,
/// rescale_output_factor=1, upcast_softmax=True)` behind its own GroupNorm.
fn attention(x: &Value, grid: &Value, a: &AttnBlock) -> Value {
    let h = group_norm(x, grid, &a.norm, false);
    let linear =
        |p: &Proj, x: &Value| ops::elemwise::add_bias(&p.bias, &ops::linear::matmul(x, &p.w));
    let q = linear(&a.q, &h);
    let k = linear(&a.k, &h);
    let v = linear(&a.v, &h);
    let o = spatial::attention(&q, &k, &v, grid, (a.width as f32).sqrt().recip());
    let o = linear(&a.out, &o);
    ops::elemwise::add(x, &o)
}
