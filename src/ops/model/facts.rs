//! **WHAT A GENERATIVE ROW CAN DO**, read off the catalog rather than off the
//! weights (imagegen design D12).
//!
//! `pie model list` and `pie model info` answered one question about an
//! artifact -- how big it is and where it came from -- which is the whole
//! answer for a text row and half of it for a generative one. A person
//! holding a freshly imported FLUX.2 has no way to find out, short of running
//! something, whether it takes a prompt at all, what resolutions its latent
//! grid divides into, or how many steps its schedule pins. Those are exactly
//! the facts `model.readings()` / `latent()` / `schedule()` answer a guest
//! with, and they are **static per SKU**: [`models::sku`] holds them beside
//! the trace, so this reads them from the artifact's own serving stamp
//! without opening a single tensor.
//!
//! An unstamped `.zt` (a checkpoint rather than an artifact) and a text row
//! both report nothing here, which is the truth in both cases and not a
//! failure.

/// The generative half of one artifact's facts, or `None` when the row it was
/// imported as is not a generative one.
#[derive(serde::Serialize)]
pub struct GenerativeFacts {
    /// The plan's readings, in index order.
    pub readings: Vec<Reading>,
    /// The latent space the denoise/VAE readings work in.
    pub latent: Option<Latent>,
    /// The schedule the denoiser was trained under.
    pub schedule: Option<Schedule>,
    /// The most latent rows one pass may carry.
    pub max_latent_rows: u32,
}

/// One reading, as a listing needs it: the roles a guest finds it by, and
/// what it binds.
#[derive(serde::Serialize)]
pub struct Reading {
    pub name: String,
    pub index: u8,
    /// Whether `attention(kv, geom)` is required on its pass.
    pub kv: bool,
    /// Whether `embed(tokens, indptr)` is required on its pass.
    pub tokens: bool,
    pub streams: Vec<String>,
    /// `name[width]` per float port, in declaration order.
    pub ports: Vec<String>,
    pub readout: String,
    pub readout_width: u32,
}

/// The latent space, verbatim.
#[derive(serde::Serialize)]
pub struct Latent {
    pub channels: u32,
    pub patch_t: u32,
    pub patch_h: u32,
    pub patch_w: u32,
    pub spatial_compression: u32,
    pub temporal_compression: u32,
}

/// The schedule, verbatim.
#[derive(serde::Serialize)]
pub struct Schedule {
    pub kind: String,
    pub shift: f32,
    pub train_steps: u32,
    pub boundary: Option<f32>,
    pub pinned_sigmas: Vec<f32>,
}

/// The facts of the catalog row `sku` names, when it is a generative one.
///
/// `None` covers three different situations that need no distinguishing in a
/// listing: no stamp on the artifact, a stamp naming a row this build does not
/// ship, and a row that is not generative.
pub fn of(sku: Option<&str>) -> Option<GenerativeFacts> {
    let generative = models::sku(sku?)?.generative.as_ref()?;
    Some(GenerativeFacts {
        readings: generative.readings.iter().map(reading).collect(),
        latent: generative.latent.map(|l| Latent {
            channels: l.channels,
            patch_t: l.patch_t,
            patch_h: l.patch_h,
            patch_w: l.patch_w,
            spatial_compression: l.spatial_compression,
            temporal_compression: l.temporal_compression,
        }),
        schedule: generative.schedule.as_ref().map(|s| Schedule {
            kind: match s.kind {
                models::ScheduleKind::Flow => "flow",
                models::ScheduleKind::Epsilon => "epsilon",
                models::ScheduleKind::V => "v-prediction",
            }
            .to_string(),
            shift: s.shift,
            train_steps: s.train_steps,
            boundary: s.boundary,
            pinned_sigmas: s.pinned_sigmas.clone(),
        }),
        max_latent_rows: generative.max_rows,
    })
}

fn reading(fact: &models::ReadingFact) -> Reading {
    Reading {
        name: fact.name.to_string(),
        index: fact.index,
        kv: fact.has_kv,
        tokens: fact.takes_tokens,
        streams: fact
            .streams
            .iter()
            .map(|s| stream(*s).to_string())
            .collect(),
        ports: fact
            .ports
            .iter()
            .map(|port| format!("{}[{}]", port.name, port.width))
            .collect(),
        readout: match fact.readout {
            models::ReadoutKind::Logits => "logits",
            models::ReadoutKind::Velocity => "velocity",
            models::ReadoutKind::Hidden => "hidden",
            models::ReadoutKind::Pixels => "pixels",
        }
        .to_string(),
        readout_width: fact.readout_width,
    }
}

fn stream(stream: models::Stream) -> &'static str {
    match stream {
        models::Stream::Text => "text",
        models::Stream::Image => "image",
        models::Stream::Video => "video",
        models::Stream::Audio => "audio",
        models::Stream::Context => "context",
        models::Stream::Reference => "reference",
    }
}

impl GenerativeFacts {
    /// The one-line form for `pie model list`: what a person scanning a store
    /// needs to tell a generative row from a text one, and one generative row
    /// from another.
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("{} readings", self.readings.len())];
        if let Some(latent) = &self.latent {
            parts.push(format!("latent {}", latent.line()));
        }
        if let Some(schedule) = &self.schedule {
            parts.push(schedule.line());
        }
        parts.join(" · ")
    }

    /// Whether this row can be told what to draw -- a text reading and a
    /// denoise reading, which is what `text-to-image` looks for.
    pub fn text_to_image(&self) -> bool {
        self.readings
            .iter()
            .any(|r| r.tokens && r.readout == "hidden")
            && self
                .readings
                .iter()
                .any(|r| !r.tokens && r.readout == "velocity")
    }
}

impl Latent {
    /// `16 ch, patch 2x2, /8 px` -- one latent row in VAE terms, and what
    /// the pixel size a job asks for gets rounded to.
    pub fn line(&self) -> String {
        let patch = if self.patch_t > 1 {
            format!("patch {}x{}x{}", self.patch_t, self.patch_h, self.patch_w)
        } else {
            format!("patch {}x{}", self.patch_h, self.patch_w)
        };
        let temporal = if self.temporal_compression > 1 {
            format!(", /{} frames", self.temporal_compression)
        } else {
            String::new()
        };
        format!(
            "{} ch, {}, /{} px{temporal}",
            self.channels, patch, self.spatial_compression
        )
    }

    /// The pixel grid one latent row covers, which is what makes a size
    /// legal: a job's width and height round down to a multiple of this.
    pub fn pixels_per_row(&self) -> (u32, u32) {
        (
            self.patch_w * self.spatial_compression,
            self.patch_h * self.spatial_compression,
        )
    }
}

impl Schedule {
    /// `flow, shift 1.0, 4 pinned sigmas` -- enough to know whether a step
    /// count is a choice or a fact.
    pub fn line(&self) -> String {
        let mut line = format!("{}, shift {:.2}", self.kind, self.shift);
        if !self.pinned_sigmas.is_empty() {
            line.push_str(&format!(", {} pinned sigmas", self.pinned_sigmas.len()));
        }
        if let Some(boundary) = self.boundary {
            line.push_str(&format!(", handover at sigma {boundary:.3}"));
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalog row every image instruction in the tree names.
    const FLUX2_KLEIN: &str = "flux2-klein-4b-bf16-kv-bf16";

    #[test]
    fn a_generative_row_reports_what_it_can_be_asked_to_draw() {
        // The claim `pie model list`'s new line rests on: the facts are on
        // the SKU, so a listing can state them without opening a tensor.
        let facts = of(Some(FLUX2_KLEIN)).expect("flux2-klein-4b is a generative row");
        assert!(
            facts.text_to_image(),
            "klein-4B has a text reading and a denoise reading: {:?}",
            facts
                .readings
                .iter()
                .map(|r| (&r.name, &r.readout, r.tokens))
                .collect::<Vec<_>>()
        );
        let latent = facts.latent.expect("a denoiser has a latent space");
        // 16 channels through a 2x2 patch over an /8 VAE is the FLUX family's
        // arithmetic, and it is what makes 1024 a legal width: one row is
        // 16 pixels a side.
        assert_eq!(latent.pixels_per_row(), (16, 16));
        let schedule = facts.schedule.expect("a denoiser has a schedule");
        assert_eq!(schedule.kind, "flow");
        assert!(facts.max_latent_rows > 0);
    }

    #[test]
    fn a_text_row_reports_nothing_here() {
        // Not an empty block, no block: a language model's one implicit
        // reading is not a fact worth a line on every listing.
        let text = models::skus()
            .find(|sku| sku.generative.is_none())
            .expect("the catalog ships text rows");
        assert!(of(Some(&text.name)).is_none());
        assert!(of(None).is_none());
        assert!(of(Some("a row this build does not ship")).is_none());
    }
}
