pub mod adapter;
pub mod deepseek_v4;
pub mod drafter;
pub mod gemma_4;
pub mod gemma_4_diffusion;
pub mod glm_5;
pub mod glm_5_next;
pub mod gpt_oss;
pub mod kimi_k3;
pub mod media;
pub mod mini_dit;
pub mod published;
pub mod qwen_3;
pub mod qwen_4;
pub mod template;
pub mod z_image;
pub mod tokenizer;

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
    pub template: fn(std::sync::Arc<::tokenizer::Tokenizer>) -> std::sync::Arc<dyn template::Instruct>,
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
        let port = self.ports.iter().find(|port| port.name == name)?;
        let index = self
            .ports
            .iter()
            .take_while(|p| p.name != name)
            .filter(|p| p.kind == port.kind)
            .count();
        Some((u8::try_from(index).unwrap_or(u8::MAX), port))
    }

    /// Every port with its kind-relative index, in declaration order.
    pub fn ports_indexed(&self) -> impl Iterator<Item = (u8, &PortFact)> + '_ {
        let mut seen = [0u8; 4];
        self.ports.iter().map(move |port| {
            let slot = match port.kind {
                PortKind::Latents => 0,
                PortKind::LaneVector => 1,
                PortKind::Context => 2,
                PortKind::AxisPositions => 3,
            };
            let index = seen[slot];
            seen[slot] = seen[slot].saturating_add(1);
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
        gemma_4::skus(),
        gemma_4_diffusion::skus(),
        glm_5::skus(),
        glm_5_next::skus(),
        gpt_oss::skus(),
        kimi_k3::skus(),
        qwen_3::skus(),
        qwen_4::skus(),
        // A diffusers pipeline reads under `dit.`/`te.` prefixes no text row
        // spells, so the generative rows identify nothing above them.
        z_image::skus(),
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
