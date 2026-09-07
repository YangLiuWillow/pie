pub mod adapter;
pub mod deepseek_v4;
pub mod drafter;
pub mod flux_2;
pub mod gemma_4;
pub mod gemma_4_diffusion;
pub mod glm_5;
pub mod glm_5_next;
pub mod gpt_oss;
pub mod hunyuan_image_3;
pub mod inkling;
pub mod kimi_k3;
pub mod ltx_2;
pub mod media;
pub mod mini_dit;
pub mod minimax_h3;
pub mod muse_glimmer;
pub mod published;
pub mod qwen_3;
pub mod qwen_4;
pub mod template;
pub mod tokenizer;
pub mod wan_2;
pub mod z_image;

use std::sync::LazyLock;

use checkpoint::contract::ModelContract;
use model_dsl::Dtype;

pub use model_dsl::{ClassifyFn, Platform, Request, Stream, biases_name, scales_name};

/// What a SKU is: the text it serves, the numeric forms its weight banks are
/// stored in (dense first, then routed experts when they differ), the kv
/// dtype, and the tensor-parallel width it is traced and imported at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recipe {
    pub text: &'static str,
    pub weights: &'static [Dtype],
    pub kv: Dtype,
    pub tp: u32,
}

impl Recipe {
    /// `<text>-<weights..>-kv-<kv>[-tp<n>]`: the name a deployment keys on.
    #[must_use]
    pub fn name(&self) -> String {
        let mut name = self.text.to_string();
        for dtype in self.weights {
            name.push('-');
            name.push_str(&word(*dtype));
        }
        name.push_str("-kv-");
        name.push_str(&word(self.kv));
        if self.tp > 1 {
            name.push_str(&format!("-tp{}", self.tp));
        }
        name
    }
}

/// A dtype as a SKU name spells it: the variant, lowercased.
#[must_use]
pub fn word(dtype: Dtype) -> String {
    format!("{dtype:?}").to_lowercase()
}

/// One shipping SKU, stated once: its recipe and how it traces, classifies,
/// imports, chats and tokenizes.
pub struct Sku {
    pub name: String,
    pub recipe: Recipe,
    pub trace: model_dsl::TraceFn,
    pub classify: ClassifyFn,
    pub import: ImportFn,
    pub template:
        fn(std::sync::Arc<::tokenizer::Tokenizer>) -> std::sync::Arc<dyn template::Instruct>,
    pub tokenizer: &'static tokenizer::Contract,
    /// The canvas a block-diffusion text denoises; `None` for every
    /// autoregressive row. What `model.pass-kind()` reads `diffusion` off,
    /// stated by the family beside its rows rather than derived from a
    /// name.
    pub diffusion: Option<Diffusion>,
    /// What a generative family (a DiT, a VAE, an encoder-plus-denoiser
    /// composite) states about its readings, latent space and schedule —
    /// the facts `model.readings()`/`latent()`/`schedule()` answer (design
    /// D12). `None` for every text row, which has one implicit reading and
    /// no latent. Set beside `diffusion`, never derived from a name.
    pub generative: Option<Generative>,
}

/// A generative family's guest-facing facts (design D12): the readings its
/// plan carries as guarded arms, the latent space its denoiser works in,
/// the schedule it was trained under, and the widest latent row count one
/// pass may carry. Everything a guest sizes a job from, so it never parses
/// `architecture()`.
#[derive(Debug, Clone, PartialEq)]
pub struct Generative {
    /// The plan's readings, in `index` order: index `i` is `Request::in_reading(i)`,
    /// the arm a lane stamped `reading = i` runs. `0` is the family's
    /// default arm. Names are the family's (`"text"`, `"denoise"`,
    /// `"denoise.low"`, `"vae.decode"`, ...); a guest names one per pass.
    pub readings: Vec<ReadingFact>,
    /// The latent space the denoise/VAE readings work in; `None` for a
    /// family with no latent (an encoder-only row).
    pub latent: Option<LatentSpace>,
    /// The noise schedule the denoiser was trained under; `None` when no
    /// reading denoises.
    pub schedule: Option<ScheduleFact>,
    /// The most latent rows one pass may carry (`Budget::max_tokens` for a
    /// float lane, D13): the guest sizes its latent channels under this.
    pub max_rows: u32,
}

/// One reading of a generative plan: which streams a request submits lanes
/// on, what it binds, what it reads back.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadingFact {
    /// The name a guest passes to `forward-pass.reading`.
    pub name: &'static str,
    /// `Request::in_reading`'s index — the arm's fact value. Unique per
    /// family; the position in `Generative::readings` must agree.
    pub index: u8,
    /// This reading declares a KV space: `attention(kv, geom)` is required
    /// on its pass. `false` refuses it by name (a DiT's denoise, a VAE).
    pub has_kv: bool,
    /// This reading embeds token rows: `embed(tokens, indptr)` is required
    /// on its pass. `false` refuses it; the lane's row count then comes
    /// from the latents port's channel shape.
    pub takes_tokens: bool,
    /// Which lane streams a request submits — one pass per stream when
    /// there are several (`forward-pass.stream`), all in one attention
    /// group. Empty means `Text` only.
    pub streams: Vec<Stream>,
    /// The float input ports, in the order the trace numbers them within
    /// each kind: the `n`th `Latents` port here is `Input::latents(n, ..)`.
    pub ports: Vec<PortFact>,
    /// Where this reading's lanes place their rows in the rotary space
    /// (design D7/D12), when it declares an `AxisPositions` port and its
    /// layout fits [`PositionConvention`]. `None` otherwise.
    pub positions: Option<PositionConvention>,
    /// Which export seam the epilogue reads (`logits()`, `velocity()`,
    /// `hidden()`).
    pub readout: ReadoutKind,
    /// The readout row's width: the vocabulary for logits, `C·p^k` for a
    /// velocity, the hidden width for a hidden readout.
    pub readout_width: u32,
}

impl ReadingFact {
    /// The port named `name`, with its index among ports of its kind —
    /// the `(PortKind, port)` pair `RuntimeInput` reads it by.
    #[must_use]
    pub fn port(&self, name: &str) -> Option<(u8, &PortFact)> {
        // Through `ports_indexed`, so a port that STATES its index
        // (`PortFact::at`) resolves to the index the trace reads it at and
        // not to its position — the two differ exactly when two readings
        // read one kind at different widths.
        self.ports_indexed().find(|(_, port)| port.name == name)
    }

    /// Every port with its kind-relative index, in declaration order.
    pub fn ports_indexed(&self) -> impl Iterator<Item = (u8, &PortFact)> + '_ {
        let mut seen = [0u8; 5];
        self.ports.iter().map(move |port| {
            let slot = match port.kind {
                PortKind::Latents => 0,
                PortKind::LaneVector => 1,
                PortKind::Context => 2,
                PortKind::AxisPositions => 3,
                PortKind::Voxels => 4,
            };
            let index = port.at.unwrap_or(seen[slot]);
            seen[slot] = index.saturating_add(1);
            (index, port)
        })
    }
}

/// One float input port a reading declares (design D3), as the guest binds
/// it with `forward-pass.input(name, channel)`. The channel is always `f32`
/// on the guest side (`types.dtype` has no bf16); the model's own element
/// type is the engine's marshal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortFact {
    /// The name the family states (`"latents"`, `"timestep"`, `"context"`,
    /// `"positions"`, `"guidance"`, ...).
    pub name: &'static str,
    /// Which `RuntimeInput` kind feeds it, and so what shape the channel is.
    pub kind: PortKind,
    /// The row width: `channels` for latents/context/lane-vector, the axis
    /// count for axis positions. Latents and context channels are
    /// `[rows, width]`; a lane vector is `[width]` or `[1, width]`.
    pub width: u32,
    /// Which streams' lanes carry this port. A row port belongs to the one
    /// stream whose rows it fills; a lane vector (a timestep) is read once
    /// per lane by every class that modulates. EMPTY means every lane of the
    /// reading. A pass on stream `s` must bind exactly the ports that list
    /// `s` (or list nothing).
    pub streams: Vec<Stream>,
    /// The `RuntimeInput` index this port is read at, when it is not the
    /// positional one. A port's index is normally its position among the
    /// ports of its own kind IN THIS READING, which is what a family wants
    /// when every reading reads the same rectangle. Two readings that read
    /// one kind at DIFFERENT WIDTHS need different indices, because the
    /// engine seats one rectangle per `(kind, index)` for the whole plan —
    /// z_image's `vae.encode` takes its pixel clip at voxel index 1 so that
    /// `vae.decode`'s 16-wide latent keeps index 0. Stating an index also
    /// moves the positional counter past it.
    pub at: Option<u8>,
    /// **THE ROW COUNT THIS PORT'S CHANNEL MUST CARRY, when the family
    /// fixes it.** `None` — nearly always — means the lane's own rows: a
    /// latents port is as tall as the picture's grid, a context port as
    /// tall as the prompt.
    ///
    /// A few families PAD instead. Wan 2.2 zero-pads its umT5 rows to 512
    /// and the transformer attends every one of the 512 keys, so a context
    /// lane of the prompt's real length is a DIFFERENT model: the pad rows
    /// go through `text_embedder` into a nonzero constant that carries real
    /// attention mass. The IR cannot grow a lane, so the pad is the
    /// GUEST's — and this is the fact that lets a family-blind guest build
    /// it without spelling 512 (design D13, "text padding is a model
    /// contract").
    pub rows: Option<u32>,
}

/// Which `RuntimeInput` kind a port is (mirrors `engine::fire::PortKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortKind {
    /// `RuntimeInput::Latents`: `[rows, width]`, the lane's own rows.
    Latents,
    /// `RuntimeInput::LaneVector`: one `[width]` row per lane (timestep,
    /// guidance, per-modality sigma).
    LaneVector,
    /// `RuntimeInput::Context`: `[rows, width]`, a context lane's rows.
    Context,
    /// `RuntimeInput::AxisPositions`: `[rows, axes]` f32, `1..=4` axes.
    AxisPositions,
    /// `RuntimeInput::Voxels` (design D8): `[t·h·w, channels]` on the voxel
    /// axis, one clip per lane — a VAE's tile. The channel is
    /// `[t, h, w, channels]` (or `[h, w, channels]` for a still), which is
    /// how the clip's box reaches the engine beside its rows.
    Voxels,
}

/// What one axis of an `AxisPositions` port means. The rotary space a DiT
/// was trained in is 1..=4 axes wide and every family orders them
/// differently; this names them so a guest can fill the grid without
/// knowing the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisRole {
    /// Frame / reference / sequence index (FLUX's `T`, Z-Image's `t`).
    Time,
    /// The latent grid's row.
    Height,
    /// The latent grid's column.
    Width,
    /// A plain row-ordinal axis carried by one lane's rows alone (FLUX's
    /// fourth `L` axis, which numbers the text rows).
    Index,
}

/// **WHERE A READING'S LANES SIT IN THE ROTARY SPACE.**
///
/// A denoise reading's `AxisPositions` port takes one coordinate vector per
/// row, and which coordinate goes where is a family contract the guest must
/// not spell (`flux_2`'s `(0, h, w, j)`, `z_image`'s `(L + 1, a, b)`). This
/// states it once, so one model-agnostic builder
/// (`inferlet::latent::positions_for`) fills every family's grid:
///
/// - a `Text`/`Context` lane's row `j` sits at `text_origin + j` on axis
///   `text_axis` and at 0 on every other axis;
/// - an `Image` lane's patch `(a, b)` sits at `a` on the `Height` axis and
///   `b` on the `Width` axis; on `text_axis` it sits at `text_origin +
///   text_rows` when `image_follows_text`, else 0; on every remaining
///   axis, 0.
/// - a `Reference` lane's patch `(a, b)` of reference `i` sits where the
///   image lane's would, except on the `Time` axis, where it sits at
///   `reference_stride · (i + 1)` — the offset that keeps each reference
///   picture in a rotary neighbourhood of its own, away from the target
///   grid at 0.
///
/// `axes` has exactly the port's `width` entries. A family whose positions
/// do not fit this shape states `None` and its guests build their own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionConvention {
    /// One role per axis of the positions port, in the port's own order.
    pub axes: Vec<AxisRole>,
    /// Which axis a text/context lane numbers its rows on.
    pub text_axis: u32,
    /// The coordinate that lane's FIRST row sits at.
    pub text_origin: u32,
    /// The image lane's coordinate on `text_axis` is `text_origin +
    /// text_rows` (Z-Image) rather than 0 (FLUX.2, mini-dit).
    pub image_follows_text: bool,
    /// **WHERE REFERENCE `i` SITS ON THE `Time` AXIS**: at
    /// `reference_stride · (i + 1)`, so the first reference clears the
    /// target grid's `T = 0` and each further one clears the last
    /// (FLUX.2's `10·(i + 1)`, `_prepare_image_ids`).
    ///
    /// `None` — the answer for every family that declares no `Reference`
    /// stream — means this row states no reference convention, and a guest
    /// that would bind a reference lane must refuse rather than invent an
    /// offset. It is the one number a reference lane needs that the target
    /// grid's rules do not already give.
    pub reference_stride: Option<u32>,
}

/// Which export seam a reading's epilogue reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadoutKind {
    /// `seam::OUT`, `intrinsics::logits()`.
    Logits,
    /// `seam::VELOCITY`, `intrinsics::velocity(width)`.
    Velocity,
    /// `seam::HIDDEN`, `intrinsics::hidden(width)`.
    Hidden,
    /// `seam::PIXELS` (design D8): one row per output voxel of the lane's
    /// clip, `[t'·h'·w', width]`, read back with the clip's output box
    /// (`engine::fire::ReadoutSeam::Pixels`).
    Pixels,
}

/// The latent space a family's denoiser works in: what one latent row is
/// and how it maps back to pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentSpace {
    /// VAE latent channels.
    pub channels: u32,
    /// Patch extents: one latent row is `patch_t × patch_h × patch_w`
    /// latent cells, so a row carries `channels · patch_t · patch_h ·
    /// patch_w` values.
    pub patch_t: u32,
    pub patch_h: u32,
    pub patch_w: u32,
    /// Pixels per latent cell along height and width.
    pub spatial_compression: u32,
    /// Frames per latent cell along time (1 for an image model).
    pub temporal_compression: u32,
}

/// The noise schedule a denoiser was trained under.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleFact {
    pub kind: ScheduleKind,
    /// The trained timestep shift (`1.0` = none); a flow model's `mu`
    /// base when it shifts dynamically by token count.
    pub shift: f32,
    /// Training steps the timestep axis is scaled by (1000 for the
    /// diffusers families).
    pub train_steps: u32,
    /// The sigma at which a two-backbone family hands over from its high-
    /// noise arm to its low-noise arm; `None` for one backbone.
    pub boundary: Option<f32>,
    /// A distilled model's pinned sigma list, descending, `1.0 -> 0.0`
    /// exclusive of the final zero; empty when the guest builds its own.
    pub pinned_sigmas: Vec<f32>,
    /// The shift each STREAM's own grid is built at, for a family whose
    /// modalities advance on different schedules inside one step (MiniMax
    /// H3 runs video at 12 and audio at 3 in the same evaluation). Empty
    /// when [`shift`](ScheduleFact::shift) serves every lane, which is
    /// every other family; a stream absent from the list takes `shift`.
    pub stream_shifts: Vec<(Stream, f32)>,
}

/// Which prediction target the schedule's velocity is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleKind {
    /// Rectified flow: `v = x1 - x0`, `x <- x + (sigma' - sigma) · v`.
    Flow,
    /// DDPM-style epsilon prediction.
    Epsilon,
    /// v-prediction.
    V,
}

/// A block-diffusion row's canvas: how many tokens one block is, and the
/// trunk's hidden width (the row width of a self-conditioning signal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Diffusion {
    pub canvas: u32,
    pub hidden: u32,
    /// How many `(id, weight)` taps per canvas row the self-conditioning
    /// input takes — the width of the two vectors a guest hands
    /// `forward-diffusion.self-conditioning`.
    pub self_cond_taps: u32,
}

impl Sku {
    /// This SKU's reading of a checkpoint, at its own width.
    pub fn contract(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, checkpoint_dsl::Error> {
        (self.import)(src, self.recipe.tp, platform)
    }
}

/// A family's reading of a checkpoint, for a stated world width and platform.
pub type ImportFn =
    fn(&ztensor::Source, u32, Platform) -> Result<ModelContract, checkpoint_dsl::Error>;

/// One family's SKU table. `$m` is `|tp: u32| Model::…(…, tp)`; the same
/// closure traces, classifies and imports the row, so the three cannot
/// disagree about the model.
#[macro_export]
macro_rules! skus {
    ($( ($text:literal, $tp:literal, [$($w:expr),+ $(,)?], $kv:expr, $trace:path, $template:expr, $tokenizer:expr, $m:expr $(,)?) ),+ $(,)?) => {
        vec![ $( {
            const RECIPE: $crate::Recipe = $crate::Recipe {
                text: $text,
                weights: &[$($w),+],
                kv: $kv,
                tp: $tp,
            };
            $crate::Sku {
                name: RECIPE.name(),
                recipe: RECIPE,
                trace: |platform: $crate::Platform| {
                    $trace(&RECIPE.name(), &($m)(RECIPE.tp), platform)
                },
                classify: |request: &$crate::Request| {
                    model_dsl::word_of(|| ($m)(RECIPE.tp), request)
                },
                import: |src: &ztensor::Source, tp: u32, platform: $crate::Platform| {
                    ($m)(tp).import(src, platform)
                },
                template: $template,
                tokenizer: $tokenizer,
                diffusion: None,
                generative: None,
            }
        } ),+ ]
    };
}

static SKUS: LazyLock<Vec<Sku>> = LazyLock::new(|| {
    [
        deepseek_v4::skus(),
        flux_2::skus(),
        gemma_4::skus(),
        gemma_4_diffusion::skus(),
        glm_5::skus(),
        glm_5_next::skus(),
        gpt_oss::skus(),
        hunyuan_image_3::skus(),
        inkling::skus(),
        kimi_k3::skus(),
        muse_glimmer::skus(),
        qwen_3::skus(),
        qwen_4::skus(),
        // A diffusers pipeline reads under `dit.`/`te.` prefixes no text row
        // spells, so the generative rows identify nothing above them.
        z_image::skus(),
        wan_2::skus(),
        minimax_h3::skus(),
        ltx_2::skus(),
        // Last: the synthetic parity row identifies nothing an operator
        // ships, and identification is catalog order.
        mini_dit::skus(),
    ]
    .into_iter()
    .flatten()
    .collect()
});

/// Every SKU this build ships, in identification order.
pub fn skus() -> impl Iterator<Item = &'static Sku> {
    SKUS.iter()
}

#[must_use]
pub fn sku(name: &str) -> Option<&'static Sku> {
    skus().find(|sku| sku.name == name)
}

/// Every one-rank SKU's reading of `src`, in identification order: the
/// first that reads is what the checkpoint is.
pub fn fits<'a>(
    src: &'a ztensor::Source,
    platform: Platform,
) -> impl Iterator<Item = (&'static Sku, Result<ModelContract, checkpoint_dsl::Error>)> + 'a {
    skus()
        .filter(|sku| sku.recipe.tp == 1)
        .map(move |sku| (sku, sku.contract(src, platform)))
}

/// The dtype the planes beside a bank of `banks` are stated in (e.g. a
/// layernorm next to a quantized weight, which is never itself quantized).
pub(crate) fn dense(banks: Dtype) -> Dtype {
    model_dsl::compute_dtype(banks)
        .unwrap_or_else(|| panic!("`{banks:?}` is not a weight representation a family declares"))
}

pub fn identify(src: &ztensor::Source, platform: Platform) -> Result<&'static str, Unmatched> {
    let mut misses: Vec<(&'static str, String)> = Vec::new();
    for (sku, read) in fits(src, platform) {
        match read {
            // A row that reads the checkpoint only by quantizing its codes a
            // second time (an 8-bit conversion under a 4-bit row) is a
            // choice, never an identification: `--sku` pins it.
            Ok(contract) => match requantizes(&contract) {
                None => return Ok(&sku.name),
                Some(plane) => misses.push((
                    &sku.name,
                    format!(
                        "reads this checkpoint only by re-quantizing `{plane}` from the form \
                         it is stored in; a second quantization is taken by `--sku`, not by \
                         identification"
                    ),
                )),
            },
            Err(why) => misses.push((&sku.name, why.to_string())),
        }
    }
    Err(Unmatched { misses })
}

/// The first plane `contract` publishes quantized out of a stored quantized
/// form (`checkpoint_dsl`'s `<name>.stored` road), if any. A stored form
/// decoded to a RAW plane is not one: the values are what the row asked for.
pub fn requantizes(contract: &checkpoint::contract::ModelContract) -> Option<String> {
    use checkpoint::types::Encoding;
    contract.tensors.iter().find_map(|stored| {
        let name = stored.name.strip_suffix(".stored")?;
        if !matches!(stored.encoding, Encoding::Quant(_)) {
            return None;
        }
        let published = contract.tensors.iter().find(|t| t.name == name)?;
        matches!(published.encoding, Encoding::Quant(_)).then(|| name.to_string())
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unmatched {
    pub misses: Vec<(&'static str, String)>,
}

impl std::fmt::Display for Unmatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "this checkpoint matches no SKU this build ships")?;
        for (sku, why) in &self.misses {
            write!(f, "\n  {sku}: {why}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Unmatched {}
