//! The FLUX 16-channel `AutoencoderKL` (Z-Image's VAE) as two readings on
//! the voxel axis (design D8, `IMAGEGEN_CONTRACT.md` §6): `vae.decode`
//! takes the DiT's latent and lands pixels, `vae.encode` takes pixels and
//! lands the posterior mean. `vae/config.json` restated as constants:
//! `block_out_channels [128, 256, 512, 512]`, `layers_per_block 2`,
//! `norm_num_groups 32`, `latent_channels 16`, `scaling_factor 0.3611`,
//! `shift_factor 0.1159`, no `quant_conv`/`post_quant_conv`, SiLU
//! everywhere, attention in the mid blocks.
//!
//! | reading | port (`Voxels`, clip `{1, h, w}`) | lands (`pixels` seam) |
//! |---|---|---|
//! | `vae.decode` | `latent` `[h·w, 16]`, the DiT-space latent | `[8h·8w, 3]` f32 in `[-1, 1]` (clamping and `uint8` are the frames layer's) |
//! | `vae.encode` | `pixels` `[H·W, 3]` in `[-1, 1]` | `[H/8·W/8, 16]`, the posterior MEAN (`conv_out`'s first 16 channels) |
//!
//! The decoder denormalises itself (`z / scaling_factor + shift_factor`,
//! the reference pipeline's step before `vae.decode`); the encoder hands
//! the raw mean back and the guest applies `(mean − shift) · scaling`
//! (FLUX and Z-Image take the mean, never a sample). Both readings run
//! over ONE clip per lane — an image, `t = 1` — and the grid is the port's.
//!
//! # Structure (diffusers `vae.py`)
//!
//! ```text
//! decode: conv_in 16→512 · mid(res, attn, res) · up0(res×3 @512, ↑2 conv)
//!         · up1(res×3 @512, ↑2 conv) · up2(res 512→256, res×2, ↑2 conv)
//!         · up3(res 256→128, res×2) · GroupNorm+SiLU · conv_out 128→3
//! encode: conv_in 3→128 · down0(res×2 @128, ↓2) · down1(res 128→256, res, ↓2)
//!         · down2(res 256→512, res, ↓2) · down3(res×2 @512) · mid
//!         · GroupNorm+SiLU · conv_out 512→32 = [mean | logvar], of which
//!         only the mean's 16 output channels are declared and read
//! res:    x + conv2(silu(gn2(conv1(silu(gn1(x)))))), a 1×1 conv on the
//!         skip where the width changes
//! attn:   x + to_out(softmax(q·kᵀ/√512)·v), q/k/v off gn(x), one head
//! ↓2:     F.pad(x, (0, 1, 0, 1)) then conv 3×3 stride 2 (pad_back)
//! ↑2:     nearest ×2 then conv 3×3
//! ```
//!
//! # Numerics contract
//!
//! Conv weights are bf16 (as stored), activations bf16 between launches,
//! every convolution and projection accumulating fp32 with one rounding at
//! the store; GroupNorm statistics fp32 Welford; the attention fp32 scores,
//! softmax and accumulation (`upcast_softmax`). The reference runs the VAE
//! in fp32 (`force_upcast`), so a port differs by one bf16 rounding per
//! launch — the parity gate (`scripts/imagegen/zimage_golden.py --vae`)
//! holds at `cos ≥ 0.999`, `max |err| ≤ 0.05` on pixels in `[-1, 1]`.

use model_dsl::ops::spatial::{self, Conv};
use model_dsl::{Dtype, Input, Value, Weight, ops, seam};

use super::forward::Facts;
use super::model::{CHANNELS, Linear, port};

/// `scaling_factor`: the DiT's latent is the posterior mean times this.
pub const SCALING_FACTOR: f32 = 0.3611;
/// `shift_factor`: subtracted from the mean before the scaling.
pub const SHIFT_FACTOR: f32 = 0.1159;
/// `norm_num_groups`.
pub const GN_GROUPS: u32 = 32;
/// Every GroupNorm's epsilon (`resnet_eps`, and the attention's `eps`).
pub const GN_EPS: f32 = 1e-6;
/// `block_out_channels`.
pub const BLOCK_CHANNELS: [u32; 4] = [128, 256, 512, 512];
/// `layers_per_block`: resnets per encoder block; a decoder block has one more.
pub const LAYERS_PER_BLOCK: u32 = 2;
/// `in_channels` / `out_channels`: RGB.
pub const RGB: u32 = 3;
/// `kh·kw` of every 3×3 convolution.
const TAPS3: u32 = 9;

/// A convolution's plane (`[C_out, C_in·taps]` bf16, tap-major at load) and
/// its `[C_out]` f32 bias.
pub struct ConvW {
    pub w: Weight,
    pub bias: Weight,
    pub c_in: u32,
    pub c_out: u32,
    pub taps: u32,
}

impl ConvW {
    fn at(name: &str, c_out: u32, c_in: u32, taps: u32) -> ConvW {
        ConvW {
            w: Weight::sym(
                name,
                [u64::from(c_out), u64::from(c_in) * u64::from(taps)],
                Dtype::Bf16,
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

/// `ResnetBlock2D`: `conv_shortcut` only where the width changes.
pub struct ResBlock {
    pub norm1: Norm,
    pub conv1: ConvW,
    pub norm2: Norm,
    pub conv2: ConvW,
    pub shortcut: Option<ConvW>,
}

impl ResBlock {
    fn at(name: &str, c_in: u32, c_out: u32) -> ResBlock {
        ResBlock {
            norm1: Norm::at(&format!("{name}.norm1"), c_in),
            conv1: ConvW::at(&format!("{name}.conv1"), c_out, c_in, TAPS3),
            norm2: Norm::at(&format!("{name}.norm2"), c_out),
            conv2: ConvW::at(&format!("{name}.conv2"), c_out, c_out, TAPS3),
            shortcut: (c_in != c_out)
                .then(|| ConvW::at(&format!("{name}.shortcut"), c_out, c_in, 1)),
        }
    }
}

/// The mid block's `Attention`: one head as wide as the row, biased
/// projections, a residual.
pub struct AttnBlock {
    pub norm: Norm,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub width: u32,
}

impl AttnBlock {
    fn at(name: &str, c: u32, banks: Dtype) -> AttnBlock {
        AttnBlock {
            norm: Norm::at(&format!("{name}.norm"), c),
            q: Linear::at(&format!("{name}.q"), c, c, banks),
            k: Linear::at(&format!("{name}.k"), c, c, banks),
            v: Linear::at(&format!("{name}.v"), c, c, banks),
            out: Linear::at(&format!("{name}.out"), c, c, banks),
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
    fn at(name: &str, c: u32, banks: Dtype) -> Mid {
        Mid {
            res0: ResBlock::at(&format!("{name}.res0"), c, c),
            attn: AttnBlock::at(&format!("{name}.attn"), c, banks),
            res1: ResBlock::at(&format!("{name}.res1"), c, c),
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
    pub conv_out: ConvW,
}

/// The whole VAE.
pub struct Vae {
    /// The `[CHANNELS]` row of [`SHIFT_FACTOR`], in the trunk's dtype — a
    /// weight because the IR has no scalar-add; derived at import.
    pub shift: Weight,
    /// The encoder's `conv_out` is stored `[2·CHANNELS, 512, 3, 3]` (`[mean
    /// | logvar]`); the plan declares its first `CHANNELS` output rows only
    /// (the mean is the latent FLUX and Z-Image take), sliced at import.
    pub encoder_out_stored: u32,
    pub decoder: Decoder,
    pub encoder: Encoder,
}

impl Vae {
    /// The FLUX VAE under `vae.`, its projections in `banks`.
    #[must_use]
    pub fn flux(banks: Dtype) -> Vae {
        let top = BLOCK_CHANNELS[BLOCK_CHANNELS.len() - 1];
        // The decoder walks the channel list reversed: 512, 512, 256, 128.
        let mut up = Vec::new();
        let mut c_prev = top;
        for (i, &c) in BLOCK_CHANNELS.iter().rev().enumerate() {
            let name = format!("vae.dec.up{i}");
            let resnets = (0..=LAYERS_PER_BLOCK)
                .map(|r| {
                    let block = ResBlock::at(&format!("{name}.res{r}"), c_prev, c);
                    c_prev = c;
                    block
                })
                .collect();
            let last = i + 1 == BLOCK_CHANNELS.len();
            up.push(UpBlock {
                resnets,
                upsample: (!last).then(|| ConvW::at(&format!("{name}.upsample"), c, c, TAPS3)),
            });
        }
        let mut down = Vec::new();
        let mut c_prev = BLOCK_CHANNELS[0];
        for (i, &c) in BLOCK_CHANNELS.iter().enumerate() {
            let name = format!("vae.enc.down{i}");
            let resnets = (0..LAYERS_PER_BLOCK)
                .map(|r| {
                    let block = ResBlock::at(&format!("{name}.res{r}"), c_prev, c);
                    c_prev = c;
                    block
                })
                .collect();
            let last = i + 1 == BLOCK_CHANNELS.len();
            down.push(DownBlock {
                resnets,
                downsample: (!last).then(|| ConvW::at(&format!("{name}.downsample"), c, c, TAPS3)),
            });
        }
        Vae {
            shift: Weight::sym("vae.shift", [u64::from(CHANNELS)], crate::dense(banks)),
            encoder_out_stored: 2 * CHANNELS,
            decoder: Decoder {
                conv_in: ConvW::at("vae.dec.conv_in", top, CHANNELS, TAPS3),
                mid: Mid::at("vae.dec.mid", top, banks),
                up,
                norm_out: Norm::at("vae.dec.norm_out", BLOCK_CHANNELS[0]),
                conv_out: ConvW::at("vae.dec.conv_out", RGB, BLOCK_CHANNELS[0], TAPS3),
            },
            encoder: Encoder {
                conv_in: ConvW::at("vae.enc.conv_in", BLOCK_CHANNELS[0], RGB, TAPS3),
                down,
                mid: Mid::at("vae.enc.mid", top, banks),
                norm_out: Norm::at("vae.enc.norm_out", top),
                conv_out: ConvW::at("vae.enc.conv_out", CHANNELS, top, TAPS3),
            },
        }
    }
}

/// The `vae.decode` reading: the DiT-space latent on the `Voxels` port,
/// denormalised, through the decoder to `[8h·8w, 3]` pixels in `[-1, 1]`
/// on the `pixels` seam beside their grid.
pub fn decode(arm: &Input<Facts>, vae: &Vae) -> Value {
    let d = &vae.decoder;
    let mut grid = arm.grid();
    let z = arm.voxels(port::VOXELS, CHANNELS, Dtype::Bf16);
    // `z / scaling_factor + shift_factor`. `add` is the one fresh copy of a
    // port rectangle this IR has (`2z`); the scale and the shift after it
    // run in place on the copy, never on the port's own cell.
    let z = ops::elemwise::add(&z, &z);
    let z = ops::elemwise::add_bias(
        &vae.shift,
        &ops::elemwise::mul_scalar(0.5 * SCALING_FACTOR.recip(), &z),
    );
    let mut h = conv3(&z, &grid, &d.conv_in);
    h = mid(&h, &grid, &d.mid);
    for block in &d.up {
        for res in &block.resnets {
            h = resnet(&h, &grid, res);
        }
        if let Some(conv) = &block.upsample {
            let (up, up_grid) = spatial::upsample_nearest(&h, &grid, [1, 2, 2], false);
            grid = up_grid;
            h = conv3(&up, &grid, conv);
        }
    }
    let h = group_norm(&h, &grid, &d.norm_out, true);
    let y = conv3(&h, &grid, &d.conv_out);
    seam::at(seam::PIXELS, &[&y, &grid]);
    y
}

/// The `vae.encode` reading: `[H·W, 3]` pixels in `[-1, 1]` on the
/// `Voxels` port through the encoder to the posterior mean `[H/8·W/8, 16]`
/// on the `pixels` seam beside its grid.
pub fn encode(arm: &Input<Facts>, vae: &Vae) -> Value {
    let e = &vae.encoder;
    let mut grid = arm.grid();
    let x = arm.voxels(port::PIXEL_VOXELS, RGB, Dtype::Bf16);
    let mut h = conv3(&x, &grid, &e.conv_in);
    for block in &e.down {
        for res in &block.resnets {
            h = resnet(&h, &grid, res);
        }
        if let Some(conv) = &block.downsample {
            // `Downsample2D`: zeros behind the box on h and w, then 3x3 at
            // stride 2 with no padding of its own.
            let (y, y_grid) = spatial::conv3d(
                &h,
                &grid,
                &conv.w,
                Some(&conv.bias),
                Conv::conv2d([3, 3], [2, 2], [0, 0]).pad_back([0, 1, 1]),
                None,
            );
            h = y;
            grid = y_grid;
        }
    }
    h = mid(&h, &grid, &e.mid);
    let h = group_norm(&h, &grid, &e.norm_out, true);
    // `DiagonalGaussianDistribution`'s `[mean | logvar]`, of which the
    // plan's `conv_out` is the mean's rows alone (the latent FLUX and
    // Z-Image take; the logvar is never computed).
    let mean = conv3(&h, &grid, &e.conv_out);
    seam::at(seam::PIXELS, &[&mean, &grid]);
    mean
}

/// A box-keeping 3×3 (or 1×1) convolution with its bias.
fn conv3(x: &Value, grid: &Value, conv: &ConvW) -> Value {
    let shape = match conv.taps {
        1 => Conv::conv2d([1, 1], [1, 1], [0, 0]),
        _ => Conv::conv2d([3, 3], [1, 1], [1, 1]),
    };
    spatial::conv3d(x, grid, &conv.w, Some(&conv.bias), shape, None).0
}

fn group_norm(x: &Value, grid: &Value, norm: &Norm, silu: bool) -> Value {
    spatial::group_norm(x, grid, GN_GROUPS, &norm.weight, &norm.bias, GN_EPS, silu)
}

/// `ResnetBlock2D`, `output_scale_factor 1`, no time embedding, no dropout.
fn resnet(x: &Value, grid: &Value, r: &ResBlock) -> Value {
    let h = group_norm(x, grid, &r.norm1, true);
    let h = conv3(&h, grid, &r.conv1);
    let h = group_norm(&h, grid, &r.norm2, true);
    let h = conv3(&h, grid, &r.conv2);
    let skip = match &r.shortcut {
        Some(conv) => conv3(x, grid, conv),
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
        |w: &Linear, x: &Value| ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w));
    let q = linear(&a.q, &h);
    let k = linear(&a.k, &h);
    let v = linear(&a.v, &h);
    let o = spatial::attention(&q, &k, &v, grid, (a.width as f32).sqrt().recip());
    let o = linear(&a.out, &o);
    ops::elemwise::add(x, &o)
}
