//! **THE MODEL-AGNOSTIC `text-to-video` INFERLET** (imagegen design D1, D4,
//! D8, D11, D12): a prompt in, an mp4 out, with nothing about any family
//! spelled here. The sibling of `text-to-image`, and deliberately its
//! sibling rather than a flag on it — three things differ, and each is a
//! fact the host answers:
//!
//! | | image | video |
//! |---|---|---|
//! | the latent | an `h x w` GRID | a `t x h x w` VOLUME (`LaneRows::Volume`, `latent().temporal-compression > 1`) |
//! | the decode | ONE fire over one clip | ONE FIRE PER LATENT FRAME, in order, down one pipeline |
//! | the way out | a still through `take_frames` | a clip assembled from the fires, `frames.from-rgb8`, `mp4-h264` |
//!
//! # HOW IT FINDS ITS READINGS
//!
//! | role | the fact that says so |
//! |---|---|
//! | text encoder | `takes_tokens && readout == Hidden` |
//! | denoiser | `!takes_tokens && readout == Velocity` |
//! | VAE decode | `readout == Pixels`, name `vae.decode*`, not `vae.encode` |
//!
//! # THE FRAME ARITHMETIC IS THE FAMILY'S, AND IT REFUSES BY NAME
//!
//! A causal video VAE decodes ONE latent frame at a time through per-conv
//! frame caches and treats the FIRST apart: latent frame 0 lands ONE output
//! frame and every later one lands `temporal-compression`. So `T` latent
//! frames are `1 + tc*(T - 1)` output frames and nothing else — 1, 5, 9,
//! 13, 17, ..., 49, ... at `tc = 4`. A `frames` that is not on that lattice
//! is REFUSED with the two nearest that are, rather than silently rounded:
//! a caller who asked for 48 frames and got 49 has a clip whose length is
//! not the one their timeline expects.
//!
//! A family that declares ONE decode reading (no `.head` arm) is an image
//! VAE or a non-causal one; this program then fires it once over the whole
//! clip and the arithmetic is `tc * T`.
//!
//! # THE DECODE FIRES ARE A SEQUENCE, AND THE ORDER IS LOAD-BEARING
//!
//! Every causal convolution of the decoder holds its last input frames in a
//! `CacheRow::State` slab that the slot carries BETWEEN fires. So the fires
//! are not independent: frame `k`'s pixels depend on frame `k-1`'s having
//! been fired into the same slot first. Two rules follow, and this program
//! keeps both:
//!
//! * ONE PIPELINE for the whole decode. A pipeline is serial, which here is
//!   the point — the opposite of the denoise loop, where one pipeline per
//!   lane is what lets a group compose.
//! * The head arm's pass is closed BEFORE the later arm's is opened. A
//!   float pass is seated on a scratch working set the host mints at bind,
//!   and a seat comes back to the book when its pass closes — so closing
//!   the head pass first is what puts the later arm on the slot the head
//!   arm warmed. Firing them the other way round, or holding both open,
//!   gives the later arm a slot whose slabs are zero, which is the pixels
//!   of a decoder with no history (measured: cos 0.98 and mean |err| 0.125
//!   on `[-1, 1]` against the reference, versus 0.99999 warm).
//!
//! # THE PIXELS DO CROSS, AND THAT IS SAID RATHER THAN HIDDEN
//!
//! `text-to-image` never lets its picture into linear memory: one fire, one
//! channel, `Channel::take_frames` hands the cell to the host's encoders.
//! A clip is several fires and there is no host-side verb that concatenates
//! their cells, so this program takes each chunk with `take_host` and builds
//! ONE handle with `frames.from-rgb8`. The cost is real (a 480x832x17 clip
//! is 81 MB of f32 through the boundary and 20 MB of RGB8 back) and the fix
//! is a seam that appends rather than replaces, not a change here.
//!
//! # CFG
//!
//! A model with no `guidance` port takes a negative prompt as a second lane
//! PAIR, and the two velocities combine `u + s(c - u)` ON THE DEVICE. The
//! branches are two attention GROUPS of one fire — they must not attend
//! each other — and each names the other with `ForwardPass::peer`, which is
//! also what gathers them into one cohort so they seal into one fire. BOTH
//! video lanes compute the same combine from their own side (the
//! conditional one reads `(own = c, peer = u)`, the other
//! `(own = u, peer = c)`), so both latents step in lockstep and neither
//! crosses the host between fires.
//!
//! This path used to fail at the first readback with "no cell available",
//! because the host combine took both branches' `out` cells in one turn and
//! one had none committed yet. Moving the combine onto the device removed
//! the turn, and with it the failure.
//!
//! # THE PROMPT, AND THE ROW WHOSE TOKENIZER PIE CANNOT SPELL
//!
//! `--prompt` renders the bound model's template and tokenizes with the
//! bound model's vocabulary. Wan 2.2's text encoder is umT5, whose
//! tokenizer is a SentencePiece **Unigram** model whereas `crates/tokenizer`
//! compiles BPE pipelines alone (`models::wan_2::tokenizer` states this),
//! so that row's artifact carries somebody else's vocabulary and `--prompt`
//! on it would condition the DiT on ids it has never seen. `--prompt-ids`
//! is the door for that: the caller tokenizes with the reference tokenizer
//! and hands the ids over, and the ENCODER still runs inside pie. The
//! report says which door was used, so a green run can never be mistaken
//! for a tokenizer that works.

use inferlet::frames::{Frames, ImageFormat};
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    prompt_ids: Option<String>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    negative_prompt_ids: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    frames: Option<u32>,
    #[serde(default)]
    fps: Option<f32>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    out: Option<String>,
}

#[derive(Serialize)]
struct Output {
    model: String,
    architecture: String,
    reading: String,
    prompt: String,
    /// `"prompt"` (the bound tokenizer) or `"prompt-ids"` (handed in).
    prompt_source: String,
    width: u32,
    height: u32,
    frames: u32,
    fps: f32,
    /// The latent-row volume and one row's width.
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    rows: u32,
    row_width: u32,
    /// `model::latent()`, verbatim.
    latent_channels: u32,
    patch_t: u32,
    patch_h: u32,
    patch_w: u32,
    spatial_compression: u32,
    temporal_compression: u32,
    /// The latent frames the decode fired, and the arms it used.
    latent_frames: u32,
    decode_fires: u32,
    decode_readings: Vec<String>,
    steps: u32,
    seed: u32,
    guidance: f32,
    cfg: bool,
    sigmas: Vec<f32>,
    /// The rows the context lane carries, and how many of them are the
    /// prompt's own (the rest are the family's stated zero pad).
    context_rows: u32,
    prompt_rows: u32,
    /// What the client was sent, and in what format.
    file: String,
    format: String,
    /// Health of the final latent — a loop that diverged says so here.
    non_finite: u32,
    mean: f32,
    std: f32,
}

/// The readings a text-to-video job needs, found by role.
struct Roles {
    text: model::ReadingFact,
    denoise: model::ReadingFact,
    /// The first-frame arm, when the family splits its causal decoder in
    /// two (`vae.decode.head`); `None` for a one-arm decoder.
    head: Option<model::ReadingFact>,
    /// The arm every later latent frame goes through, or the only arm.
    decode: model::ReadingFact,
}

fn roles() -> Result<Roles> {
    // **NOT `pass_kind()`.** That verb is derived from `rs_state_size() > 0`,
    // and a video row carries a causal-conv VAE whose every convolution owns
    // a `CacheRow::State` slab — so `wan22-ti2v-5b` answers `recurrent`
    // while a transformer-only miniature answers `attention`. The
    // difference is in the DECODER arms and says nothing about a denoise
    // pass. What a denoise step actually needs is stated on the reading,
    // and is checked below: it binds no kv space and takes no tokens.
    let readings = model::readings();
    let text = readings
        .iter()
        .find(|r| r.takes_tokens && r.readout == model::ReadoutKind::Hidden)
        .cloned()
        .ok_or("this model has no text reading")?;
    let denoise = readings
        .iter()
        .find(|r| !r.takes_tokens && r.readout == model::ReadoutKind::Velocity)
        .cloned()
        .ok_or("this model declares no token-less reading with a velocity readout")?;
    if denoise.has_kv {
        return Err(format!(
            "reading `{}` binds a kv space; a denoise pass binds none",
            denoise.name
        )
        .into());
    }
    // A decode arm lands PIXELS. The encoder lands pixels too — its pixels
    // are its INPUT — and the name is what separates them, in the design's
    // own reading vocabulary and not a family's.
    let pixels: Vec<model::ReadingFact> = readings
        .iter()
        .filter(|r| r.readout == model::ReadoutKind::Pixels && r.name != "vae.encode")
        .cloned()
        .collect();
    let head = pixels.iter().find(|r| r.name.ends_with(".head")).cloned();
    let decode = pixels
        .iter()
        .find(|r| !r.name.ends_with(".head"))
        .cloned()
        .ok_or(
            "this model declares no `vae.decode` reading, so there is no way to turn its \
             latent into frames from inside pie",
        )?;
    Ok(Roles {
        text,
        denoise,
        head,
        decode,
    })
}

/// The denoise reading's ports, sorted into the roles a sampler binds.
struct Ports {
    latents: model::PortFact,
    context: model::PortFact,
    timestep: String,
    guidance: Option<String>,
    convention: model::PositionConvention,
    video_stream: model::LaneStream,
    context_stream: model::LaneStream,
}

fn stream_of(
    streams: &[model::LaneStream],
    want: model::LaneStream,
    fallback: model::LaneStream,
) -> model::LaneStream {
    if streams.iter().any(|s| *s == want) {
        return want;
    }
    streams.first().copied().unwrap_or(fallback)
}

fn ports(reading: &model::ReadingFact) -> Result<Ports> {
    let of_kind = |kind: model::PortKind| reading.ports.iter().find(|p| p.kind == kind).cloned();
    let latents = of_kind(model::PortKind::Latents)
        .ok_or("this denoise reading declares no latents port; there is nothing to denoise")?;
    let context = of_kind(model::PortKind::Context)
        .ok_or("this denoise reading declares no context port for the prompt rows")?;
    let positions = of_kind(model::PortKind::AxisPositions)
        .ok_or("this denoise reading declares no axis-positions port")?;
    let convention = reading.positions.clone().ok_or_else(|| {
        format!(
            "reading `{}` takes positions but states no position convention, so a \
             family-blind guest cannot say where its rows sit",
            reading.name
        )
    })?;
    if convention.axes.len() as u32 != positions.width {
        return Err(format!(
            "reading `{}` states {} axis roles for a {}-wide positions port",
            reading.name,
            convention.axes.len(),
            positions.width
        )
        .into());
    }
    // A video family's rows sit on a TIME axis; without one there is no
    // coordinate for a frame index and this is the wrong driver.
    if !convention.axes.iter().any(|r| *r == model::AxisRole::Time) {
        return Err(format!(
            "reading `{}` states no `time` axis in its position convention; a clip's rows \
             have nowhere to put a frame index",
            reading.name
        )
        .into());
    }
    let guidance = reading
        .ports
        .iter()
        .find(|p| p.kind == model::PortKind::LaneVector && p.name == "guidance")
        .map(|p| p.name.clone());
    let timestep = reading
        .ports
        .iter()
        .find(|p| p.kind == model::PortKind::LaneVector && Some(&p.name) != guidance.as_ref())
        .map(|p| p.name.clone())
        .ok_or_else(|| {
            format!(
                "reading `{}` declares no lane-vector port for the timestep",
                reading.name
            )
        })?;
    let video_stream = stream_of(
        &latents.streams,
        model::LaneStream::Video,
        model::LaneStream::Image,
    );
    let context_stream = stream_of(
        &context.streams,
        model::LaneStream::Context,
        model::LaneStream::Text,
    );
    Ok(Ports {
        latents,
        context,
        timestep,
        guidance,
        convention,
        video_stream,
        context_stream,
    })
}

/// Everything one lane binds, kept alive for the loop's lifetime.
struct Lane {
    pass: ForwardPass,
    #[allow(dead_code)]
    held: Vec<Channel>,
}

#[allow(clippy::too_many_arguments)]
fn lane(
    reading: &model::ReadingFact,
    ports: &Ports,
    stream: model::LaneStream,
    group: u32,
    // The OTHER guidance branch's group, on the CFG path. Named by EVERY
    // lane of both branches, not only the ones that read: naming is what
    // gathers the two groups into one cohort, and a peer in another fire is
    // not on this fire's velocity plane at all.
    peer: Option<u32>,
    rows: u32,
    latents: Option<&Channel>,
    context: Option<&Channel>,
    positions: &[f32],
    clock: &LaneClock,
    guidance: f32,
    tag: &str,
) -> Result<Lane> {
    let pass = ForwardPass::new();
    pass.reading(&reading.name)?;
    pass.stream(stream)?;
    pass.group(group)?;
    if let Some(peer) = peer {
        pass.peer(peer)?;
    }
    let mut held = Vec::new();
    let mut latents_bound = false;
    let mut context_bound = false;
    let mut timestep_bound = false;
    for port in &reading.ports {
        if !(port.streams.is_empty() || port.streams.iter().any(|s| *s == stream)) {
            continue;
        }
        let name = format!("{tag}_{}", port.name);
        let ch = match port.kind {
            model::PortKind::Latents if !latents_bound && latents.is_some() => {
                latents_bound = true;
                latents.unwrap().clone()
            }
            model::PortKind::Context if !context_bound && context.is_some() => {
                context_bound = true;
                context.unwrap().clone()
            }
            model::PortKind::Latents | model::PortKind::Context => {
                Channel::from_shaped([rows, port.width], vec![0f32; (rows * port.width) as usize])
                    .named(&name)
            }
            model::PortKind::AxisPositions => {
                Channel::from_shaped([rows, port.width], positions.to_vec()).named(&name)
            }
            model::PortKind::LaneVector if port.name == ports.timestep && !timestep_bound => {
                timestep_bound = true;
                clock.timestep.clone()
            }
            model::PortKind::LaneVector if Some(&port.name) == ports.guidance.as_ref() => {
                Channel::from(vec![guidance; port.width as usize]).named(&name)
            }
            model::PortKind::LaneVector => {
                Channel::from(vec![1.0f32; port.width as usize]).named(&name)
            }
            model::PortKind::Voxels => {
                return Err(format!(
                    "reading `{}` declares a voxel port `{}`; this sampler drives latent rows, \
                     not a VAE tile",
                    reading.name, port.name
                )
                .into());
            }
        };
        pass.input(&port.name, &ch)?;
        held.push(ch);
    }
    Ok(Lane { pass, held })
}

/// Unpatchify a denoise reading's final latent into the clip volume a
/// `vae.decode` port takes: `[rows, C*pt*ph*pw]` in `(c, pt, ph, pw)`
/// feature order becomes `[grid_t*pt, grid_h*ph, grid_w*pw, C]` row-major
/// — the `[t, h, w, C]` shape a `Voxels` channel declares (design D8).
///
/// The inverse of the patchify every DiT does at its embed, written out
/// rather than called, because the guest is the only place both halves of
/// the pair are visible: the trunk states the patch and the VAE states the
/// clip, and nothing between them holds the index algebra.
#[allow(clippy::too_many_arguments)]
fn unpatchify(rows: &[f32], grid: [u32; 3], patch: [u32; 3], channels: u32) -> Vec<f32> {
    let (gt, gh, gw) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
    let (pt, ph, pw) = (patch[0] as usize, patch[1] as usize, patch[2] as usize);
    let c = channels as usize;
    let (t, h, w) = (gt * pt, gh * ph, gw * pw);
    let row_width = c * pt * ph * pw;
    let mut out = vec![0f32; t * h * w * c];
    for a in 0..gt {
        for b in 0..gh {
            for d in 0..gw {
                let row = ((a * gh + b) * gw + d) * row_width;
                for ci in 0..c {
                    for z in 0..pt {
                        for y in 0..ph {
                            for x in 0..pw {
                                let src = row + ((ci * pt + z) * ph + y) * pw + x;
                                let dst =
                                    (((a * pt + z) * h + (b * ph + y)) * w + (d * pw + x)) * c + ci;
                                out[dst] = rows[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Fire ONE decode arm over ONE latent frame and take its pixels back.
///
/// The clip's box IS the channel's shape (`[h, w, C]` for one frame), so
/// the geometry the voxel axis needs travels with the numbers. `frames` is
/// how many output frames this arm lands, which the caller knows from the
/// family's `temporal-compression` and which arm this is.
async fn decode_frame(
    reading: &model::ReadingFact,
    clip: &[f32],
    clip_h: u32,
    clip_w: u32,
    frames: u32,
    pixel_h: u32,
    pixel_w: u32,
    pipe: &Pipeline,
) -> Result<Vec<f32>> {
    let port = reading
        .ports
        .iter()
        .find(|p| p.kind == model::PortKind::Voxels)
        .ok_or_else(|| {
            format!(
                "reading `{}` lands pixels but reads no voxel port; there is no clip to hand it",
                reading.name
            )
        })?;
    let want = (clip_h as usize) * (clip_w as usize) * (port.width as usize);
    if clip.len() != want {
        return Err(format!(
            "one latent frame is {} numbers and port `{}` wants {clip_h}x{clip_w}x{} = {want}",
            clip.len(),
            port.name,
            port.width
        )
        .into());
    }
    let rows = frames * pixel_h * pixel_w;
    let width = reading.readout_width;
    let pass = ForwardPass::new();
    pass.reading(&reading.name)?;
    pass.stream(model::LaneStream::Video)?;
    let cell =
        Channel::from_shaped([clip_h, clip_w, port.width], clip.to_vec()).named("vae_latent");
    pass.input(&port.name, &cell)?;
    let out = Channel::new([rows, width], dtype::f32).named("vae_pixels");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::pixels(rows, width));
    });
    pass.submit(pipe)
        .with_context(|| format!("the `{}` fire", reading.name))?;
    out.take_host::<Vec<f32>>().await.map_err(Into::into)
}

/// `[-1, 1]` f32 in `(t, h, w, 3)` order to interleaved RGB8, clamped.
fn rgb8(pixels: &[f32]) -> Vec<u8> {
    pixels
        .iter()
        .map(|v| (((v + 1.0) * 127.5).clamp(0.0, 255.0)) as u8)
        .collect()
}

/// **THE CONTEXT PAD, WHERE THE FAMILY STATES ONE.**
///
/// `port-fact.rows` is a family saying "my context lane is exactly this
/// tall, whatever the prompt was". Wan 2.2 says 512: its reference
/// truncates umT5's answer to the prompt's real length and zero-pads the
/// EMBEDS back to 512, and the transformer attends every one of those keys
/// — the pad rows are not masked away, they go through `text_embedder`
/// into a nonzero constant with real attention mass. Handing such a model a
/// context lane of the prompt's own height is a different model, quietly.
///
/// `None` (FLUX.2, Z-Image) means the encoder's rows are the lane's rows
/// and this hands the channel straight back. A prompt LONGER than the
/// stated height is refused rather than truncated: the encoder already
/// truncated at `max-embed-length` if it had to, and silently dropping rows
/// here would drop the end of a caption.
fn pad_context(mut rows: Vec<f32>, width: u32, port: &model::PortFact) -> Result<Channel> {
    if width != port.width {
        return Err(format!(
            "the text reading hands back {width}-wide rows and port `{}` takes {}",
            port.name, port.width
        )
        .into());
    }
    let have = u32::try_from(rows.len() / width.max(1) as usize).unwrap_or(0);
    let want = port.rows.unwrap_or(have);
    if have > want {
        return Err(format!(
            "port `{}` takes exactly {want} context rows and the prompt encoded to {have}",
            port.name
        )
        .into());
    }
    // Zero rows, not pad-token rows: the reference truncates the encoder's
    // answer and pads the EMBEDS, and zeros are what the DiT was
    // conditioned on. There is no device-side verb that grows a rectangle,
    // and these rows already crossed the host coming out of the encoder.
    rows.resize((want * width) as usize, 0.0);
    Ok(Channel::from_shaped([want, width], rows).named("context"))
}

/// The prompt as ids, through the BOUND MODEL's own template and
/// tokenizer: one user turn plus the generation cue, which is what every
/// instruct-encoder pipeline renders before it tokenizes. Only reached on
/// the `--prompt` door; `--prompt-ids` skips both.
fn template_ids(prompt: &str) -> Vec<u32> {
    let mut ids = inferlet::chat::first_user(prompt);
    ids.extend(inferlet::chat::cue());
    ids
}

/// `"1,2,3"` to ids. Refuses a piece that is not a number, by name.
fn ids_of(text: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for piece in text.split(|c: char| c == ',' || c.is_whitespace()) {
        if piece.is_empty() {
            continue;
        }
        out.push(
            piece
                .parse::<u32>()
                .map_err(|_| format!("`{piece}` in the id list is not a token id"))?,
        );
    }
    if out.is_empty() {
        return Err("the id list is empty".into());
    }
    Ok(out)
}

/// One prompt's two lanes: the encoder rows and the latent they condition.
struct Branch {
    context: Lane,
    image: Lane,
    out: Channel,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let roles = roles()?;
    let ports = ports(&roles.denoise)?;

    // ---- the latent volume, out of `model::latent()` ----------------------
    let space = model::latent().ok_or(
        "this model states no latent space, so its denoise reading cannot be sized from pixels",
    )?;
    let sc = space.spatial_compression.max(1);
    let tc = space.temporal_compression.max(1);
    let (pt, ph, pw) = (
        space.patch_t.max(1),
        space.patch_h.max(1),
        space.patch_w.max(1),
    );
    if tc == 1 {
        return Err(
            "this model's latent space has no temporal compression: it is an image family, and \
             `text-to-image` is its driver"
                .into(),
        );
    }
    let cell_h = sc * ph;
    let cell_w = sc * pw;
    let height = (input.height.unwrap_or(480) / cell_h).max(1) * cell_h;
    let width = (input.width.unwrap_or(832) / cell_w).max(1) * cell_w;
    let grid_h = height / cell_h;
    let grid_w = width / cell_w;

    // THE FRAME LATTICE, REFUSED BY NAME. A causal decoder with a head arm
    // lands `1 + tc*(T - 1)` frames; a one-arm decoder lands `tc*T`.
    let asked = input.frames.unwrap_or(17).max(1);
    let latent_frames = if roles.head.is_some() {
        if (asked - 1) % tc != 0 {
            let below = 1 + tc * ((asked - 1) / tc);
            return Err(format!(
                "this family's VAE lands 1 + {tc}*(T - 1) frames, so {asked} is not a clip it \
                 has: the nearest are {below} and {}",
                below + tc
            )
            .into());
        }
        (asked - 1) / tc + 1
    } else {
        if asked % tc != 0 {
            return Err(format!(
                "this family's VAE lands {tc}*T frames, so {asked} is not a clip it has: the \
                 nearest are {} and {}",
                (asked / tc).max(1) * tc,
                (asked / tc + 1) * tc
            )
            .into());
        }
        asked / tc
    };
    let frames = asked;
    if latent_frames % pt != 0 {
        return Err(format!(
            "{latent_frames} latent frames do not divide into this model's {pt}-deep patches"
        )
        .into());
    }
    let grid_t = latent_frames / pt;
    let rows = grid_t * grid_h * grid_w;
    let row_width = ports.latents.width;
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && rows > max_rows {
        return Err(format!(
            "{width}x{height} at {frames} frames is {rows} latent rows and this model carries \
             {max_rows} a pass"
        )
        .into());
    }

    // ---- the schedule -----------------------------------------------------
    let fact = model::schedule().ok_or("this model states no schedule; nothing here denoises")?;
    let steps = match input.steps.filter(|s| *s > 0) {
        Some(steps) => steps,
        None if !fact.pinned_sigmas.is_empty() => {
            u32::try_from(fact.pinned_sigmas.len()).unwrap_or(8)
        }
        None => 8,
    };
    // **NO DYNAMIC SHIFT.** `FlowMatchEuler`'s `rows` argument turns the
    // stated shift into a base `mu` and bends it by the latent's row count
    // — a resolution heuristic FLUX and Z-Image were trained with. A video
    // family's rows count FRAMES as well as pixels, and neither video row
    // in the tree rescales by them (Wan's scheduler says
    // `use_dynamic_shifting: false`, and its 5.0 is the shift itself), so
    // bending by 1950 rows would put nineteen of twenty steps above sigma
    // 0.58 and leave the last one to do the denoising. `None` takes the
    // stated shift as the fixed one it is.
    let sched = FlowMatchEuler::from_schedule(&fact, steps, None)?;

    // ---- the prompt, through whichever door the caller opened -------------
    let guidance = input.guidance.unwrap_or(1.0);
    let (encoded, prompt, prompt_source) = match (&input.prompt_ids, &input.prompt) {
        (Some(ids), _) => {
            let ids = ids_of(ids)?;
            let shown = format!("<{} ids>", ids.len());
            (
                encode_ids_rows(&ids, &roles.text.name).await?,
                shown,
                "prompt-ids".to_string(),
            )
        }
        (None, Some(text)) if !text.trim().is_empty() => (
            encode_ids_rows(&template_ids(text), &roles.text.name).await?,
            text.clone(),
            "prompt".to_string(),
        ),
        _ => {
            return Err(
                "pass `prompt` or `prompt_ids`: this program films what it is told to".into(),
            );
        }
    };
    let (hidden, hidden_width) = encoded;
    let prompt_rows = u32::try_from(hidden.len() / hidden_width.max(1) as usize).unwrap_or(0);
    let context = pad_context(hidden, hidden_width, &ports.context)?;
    let context_rows = context.shape().dims()[0];
    let negative = match (&input.negative_prompt_ids, &input.negative_prompt) {
        (Some(ids), _) => {
            let (anti, anti_width) = encode_ids_rows(&ids_of(ids)?, &roles.text.name).await?;
            Some(pad_context(anti, anti_width, &ports.context)?)
        }
        (None, Some(text)) if !text.trim().is_empty() => {
            let (anti, anti_width) = encode_ids_rows(&template_ids(text), &roles.text.name).await?;
            Some(pad_context(anti, anti_width, &ports.context)?)
        }
        _ => None,
    };
    if negative.is_some() && ports.guidance.is_some() {
        return Err(format!(
            "reading `{}` declares a `guidance` port, so this row is guidance-distilled and a \
             negative prompt has no lane to ride",
            roles.denoise.name
        )
        .into());
    }
    let cfg = negative.is_some() && guidance > 1.0;
    let uncond = if cfg { negative } else { None };

    // ---- the loop ---------------------------------------------------------
    let mut loops = DenoiseLoop::new(&sched);
    let velocity_width = roles.denoise.readout_width;
    let cells = (rows * row_width) as usize;
    let shape = [rows, row_width];
    let seed = input.seed.unwrap_or(0);
    let volume = LaneRows::Volume {
        t: grid_t,
        h: grid_h,
        w: grid_w,
    };

    let mut branches: Vec<Branch> = Vec::new();
    for (group, ctx) in [Some(&context), uncond.as_ref()]
        .into_iter()
        .flatten()
        .enumerate()
    {
        let group = group as u32;
        let tag = if group == 0 { "cond" } else { "uncond" };
        let ctx_rows = ctx.shape().dims()[0];
        let text_clock = loops.lane(&format!("{tag}_ctx"));
        let image_clock = loops.lane(&format!("{tag}_vid"));
        let context_lane = lane(
            &roles.denoise,
            &ports,
            ports.context_stream,
            group,
            cfg.then(|| 1 - group),
            ctx_rows,
            None,
            Some(ctx),
            &positions_for(&ports.convention, LaneRows::Sequence(ctx_rows), ctx_rows),
            &text_clock,
            guidance,
            &format!("{tag}_ctx"),
        )?;
        let latent = Channel::from_shaped(shape, vec![0f32; cells]).named(&format!("{tag}_x"));
        let image_lane = lane(
            &roles.denoise,
            &ports,
            ports.video_stream,
            group,
            cfg.then(|| 1 - group),
            rows,
            Some(&latent),
            None,
            &positions_for(&ports.convention, volume, ctx_rows),
            &image_clock,
            guidance,
            &format!("{tag}_vid"),
        )?;
        let out = Channel::new(shape, dtype::f32)
            .capacity(channel_capacity() as u32)
            .named(&format!("{tag}_out"));
        text_clock.drive(&context_lane.pass, |_| {});
        let dts = loops.dts(tag);
        let rng = Channel::from(rng_state(seed)).named(&format!("{tag}_rng"));
        let x = latent.clone();
        let readback = out.clone();
        image_clock.drive(&image_lane.pass, move |k| {
            // On the CFG path BOTH branches step with the SAME combine, each
            // computed from its own side: the conditional lane reads
            // `(own = c, peer = u)` and the unconditional one
            // `(own = u, peer = c)`, and `u + s(c - u)` is the same number
            // either way. So the two latents stay in step without either one
            // crossing the host between fires.
            let v = if cfg {
                guided_velocity(velocity_width, guidance, group == 0)
            } else {
                intrinsics::velocity(velocity_width)
            };
            seed_or_step(k, &x, &v, &dts, &rng, shape, Some(&readback));
        });
        branches.push(Branch {
            context: context_lane,
            image: image_lane,
            out,
        });
    }

    // Both paths are the same loop now: the device seeds its own latent and
    // integrates its own Euler step, guided or not. The guided branches read
    // each other's velocity off the fire's own plane, so nothing crosses the
    // host between steps — which is also why the four-lane path runs at all
    // now: it no longer takes two branches' cells in one turn.
    let mut last: Vec<f32> = Vec::new();
    for fire in 0..loops.fires() {
        let passes: Vec<&ForwardPass> = branches
            .iter()
            .flat_map(|b| [&b.context.pass, &b.image.pass])
            .collect();
        loops
            .fire(&passes)
            .with_context(|| format!("fire {fire}"))?;
        // The conditional branch's latent is the answer; the unconditional
        // one steps in lockstep with it (same combine, computed from its own
        // side) and is read only to keep its channel's cursor moving.
        last = branches[0]
            .out
            .take_host::<Vec<f32>>()
            .await
            .with_context(|| format!("readback after fire {fire}"))?;
        if cfg {
            let _ = branches[1].out.take_host::<Vec<f32>>().await?;
        }
    }
    loops.close();
    // EVERY DENOISE PASS IS CLOSED BEFORE THE FIRST DECODE FIRE. Each holds
    // a seat, and the decode's two arms need the ones this releases.
    drop(branches);

    let n = last.len().max(1) as f32;
    let mean = last.iter().sum::<f32>() / n;
    let var = last.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    if last.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "the denoise loop left {} non-finite values in the latent; there is nothing to \
             decode",
            last.iter().filter(|v| !v.is_finite()).count()
        )
        .into());
    }

    // ---- the decode: one fire per latent frame, in order ------------------
    let clip = unpatchify(
        &last,
        [grid_t, grid_h, grid_w],
        [pt, ph, pw],
        space.channels,
    );
    let clip_h = grid_h * ph;
    let clip_w = grid_w * pw;
    let plane = (clip_h * clip_w * space.channels) as usize;
    let mut pixels: Vec<f32> = Vec::with_capacity((frames * height * width * 3) as usize);
    let mut used: Vec<String> = Vec::new();

    // ONE pipeline for the whole decode: a pipeline is serial, and the
    // frame caches make these fires a sequence.
    let vae_pipe = Pipeline::new();
    let mut first_frame = 0u32;
    if let Some(head) = &roles.head {
        let out = decode_frame(
            head,
            &clip[..plane],
            clip_h,
            clip_w,
            1,
            height,
            width,
            &vae_pipe,
        )
        .await?;
        pixels.extend_from_slice(&out);
        used.push(head.name.clone());
        first_frame = 1;
    }
    // The head arm's pass is closed by now (`decode_frame` owns it and it
    // died with the call), so the later arm binds onto the seat it warmed.
    for k in first_frame..latent_frames {
        let at = k as usize * plane;
        let out = decode_frame(
            &roles.decode,
            &clip[at..at + plane],
            clip_h,
            clip_w,
            tc,
            height,
            width,
            &vae_pipe,
        )
        .await?;
        pixels.extend_from_slice(&out);
    }
    used.push(roles.decode.name.clone());
    vae_pipe.close();

    let want = (frames as usize) * (height as usize) * (width as usize) * 3;
    if pixels.len() != want {
        return Err(format!(
            "the decode landed {} values and a {frames}x{height}x{width} clip is {want}",
            pixels.len()
        )
        .into());
    }

    // ---- the way out ------------------------------------------------------
    let fps = input.fps.unwrap_or(24.0);
    let name = input.out.unwrap_or_else(|| "video".to_string());
    let asked_format = input.format.as_deref().unwrap_or("mp4");
    let (format, ext) = match asked_format {
        "mp4" | "mp4-h264" => (ImageFormat::Mp4H264, "mp4"),
        "y4m" => (ImageFormat::Y4m, "y4m"),
        other => return Err(format!("unknown video format {other:?}; try 'mp4' or 'y4m'").into()),
    };
    let handle = Frames::from_rgb8(&rgb8(&pixels), width, height, frames, fps)
        .map_err(|why| format!("frames.from-rgb8: {why}"))?;
    let file = format!("{name}.{ext}");
    inferlet::session::send_frames(&handle, format, &file)
        .map_err(|why| format!("session.send-frames({asked_format}): {why}"))?;

    Ok(Output {
        model: model::name(),
        architecture: model::architecture(),
        reading: roles.denoise.name.clone(),
        prompt,
        prompt_source,
        width,
        height,
        frames,
        fps,
        grid_t,
        grid_h,
        grid_w,
        rows,
        row_width,
        latent_channels: space.channels,
        patch_t: pt,
        patch_h: ph,
        patch_w: pw,
        spatial_compression: sc,
        temporal_compression: tc,
        latent_frames,
        decode_fires: latent_frames,
        decode_readings: used,
        steps: sched.steps(),
        seed,
        guidance,
        cfg,
        sigmas: sched.sigmas.clone(),
        context_rows,
        prompt_rows,
        file,
        format: asked_format.to_string(),
        non_finite: 0,
        mean,
        std: var.sqrt(),
    })
}

