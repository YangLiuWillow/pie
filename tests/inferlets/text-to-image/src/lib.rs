//! **THE MODEL-AGNOSTIC `text-to-image` INFERLET** (imagegen design D1, D4,
//! D11, D12): a prompt in, an image out, with nothing about any family
//! spelled here. Everything this program needs to size and drive the job is
//! a host-answered fact — `model::readings()`, `model::latent()`,
//! `model::schedule()`, `model::max_latent_rows()` — and every port and
//! reading is found by ROLE, never by name and never by `architecture()`.
//!
//! # HOW IT FINDS ITS THREE READINGS
//!
//! | role | the fact that says so |
//! |---|---|
//! | text encoder | `takes_tokens && readout == Hidden` |
//! | denoiser | `!takes_tokens && readout == Velocity` |
//! | VAE decode | a reading named `vae.decode` |
//!
//! A model with no text reading is refused BY NAME ("this model has no text
//! reading") — that is the right answer for the `mini-dit` fixture, whose
//! caption rows are random embeddings and which therefore cannot be told
//! what to draw.
//!
//! # THE STEP LOOP
//!
//! Two lanes of one request (D2), each on its own pipeline, submitted back
//! to back so the wait-all seal composes them into ONE fire: the prompt
//! lane (the stream the reading's `Context` port lists — `Text` for FLUX.2,
//! `Context` for Z-Image) carrying the encoder's rows, and the image lane
//! carrying the latent. Neither `embed` nor `attention` is called: a
//! denoise reading declares no tokens and no KV space.
//!
//! The sampler is an EPILOGUE (D4). Fire 0 seeds the latent with a keyed
//! Normal draw on the device — no 2 MB `Channel::from` through WASM — and
//! every later fire integrates one Euler step `x <- x + (sigma' - sigma).v`
//! over `velocity()`, advancing the loop-carried timestep and step cells.
//! `inferlet::latent::DenoiseLoop` owns the per-lane pipelines and clocks;
//! `seed_or_step` is the epilogue body.
//!
//! # POSITIONS COME FROM A FACT, NOT FROM THE FAMILY
//!
//! Where a lane's rows sit in the rotary space is a family contract — FLUX.2
//! puts the target grid at `(0, h, w, 0)` and text row `j` at `(0, 0, 0, j)`,
//! Z-Image puts the caption at `(1 + j, 0, 0)` and the image BEHIND it at
//! `(L + 1, a, b)` — so a guest that spelled either would be a guest for one
//! family. The reading states it instead (`reading-fact.positions`, a
//! `position-convention` of per-axis roles plus the text axis, its origin
//! and whether the image follows the caption on it), and
//! `latent::positions_for` fills both grids from the same code.
//!
//! # THE WAY OUT
//!
//! Two exits, and which one a model gets is its own fact.
//!
//! The intended one is D11: a `vae.decode` reading, whose pixels land in a
//! host-held `frames` handle that `session::send_frames` encodes and
//! streams to the client without ever entering linear memory. **No family
//! declares that reading on `dev`** — `flux_2` traces its decoder but keeps
//! it out of the facts, because `PortKind`/`ReadoutKind` carry no
//! voxel/pixel seam yet — so there is no honest way to drive it from here
//! and this program says so by name rather than guessing at a shape that
//! does not exist. The branch is written that way ON PURPOSE: the day the
//! VAE arm lands, this is the one place that changes.
//!
//! The exit every model has today is the second: the final latent as raw
//! little-endian f32 through `session::send_file`, with the geometry a
//! decode needs returned as the JSON report — its sidecar.
//! `scripts/imagegen/decode_latent.py` finishes the job with the diffusers
//! VAE of the same checkpoint.
//!
//! # CFG
//!
//! When the reading declares a `guidance` port the model is guidance-
//! distilled and the scale is just another lane vector; `--negative-prompt`
//! is then refused as meaningless. Without that port, a negative prompt at
//! `guidance > 1` runs a second lane PAIR (its own group, its own prompt
//! rows, its own latent), and the two velocities combine as
//! `u + s(c - u)`. That combine is HOST-side here, and the reason is worth
//! stating: an epilogue is per lane and reads its own lane's `velocity()`
//! only, so the device form (`latent::cfg_combine`) would need one lane's
//! epilogue to see another lane's epilogue write WITHIN one fire, which the
//! channel contract does not promise. The arithmetic is
//! `latent::cfg_combine_host`, the same formula stated once. No model in
//! the tree exercises this path (klein-4B and Z-Image Turbo are both
//! distilled), so it is the shape of a CFG loop and not a verified one.

use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    out: Option<String>,
}

/// The report, which doubles as the SIDECAR of the raw-latent exit: every
/// number `scripts/imagegen/decode_latent.py` needs to turn the `.f32` blob
/// back into pixels, plus what the run actually did.
#[derive(Serialize)]
struct Output {
    model: String,
    architecture: String,
    reading: String,
    prompt: String,
    /// The pixel size actually rendered, after rounding to the grid.
    width: u32,
    height: u32,
    /// The latent-row grid and one row's width.
    grid_h: u32,
    grid_w: u32,
    rows: u32,
    row_width: u32,
    /// `model::latent()`, verbatim: what a row is in VAE terms.
    latent_channels: u32,
    patch_h: u32,
    patch_w: u32,
    spatial_compression: u32,
    steps: u32,
    seed: u32,
    guidance: f32,
    cfg: bool,
    sigmas: Vec<f32>,
    /// The prompt's row count out of the text reading.
    context_rows: u32,
    /// Whether the pixels went out as an encoded image (a `vae.decode`
    /// reading) or the latent went out raw.
    decoded: bool,
    /// The name the client should give what was sent.
    file: String,
    bytes: u32,
    /// Health of the final latent — a loop that diverged says so here.
    non_finite: u32,
    mean: f32,
    std: f32,
}

/// The readings a text-to-image job needs, found by role.
struct Roles {
    text: model::ReadingFact,
    denoise: model::ReadingFact,
    decode: Option<model::ReadingFact>,
}

fn roles() -> Result<Roles> {
    if model::pass_kind() != model::ForwardKind::Attention {
        return Err(
            "a DiT's readings are attention-kind passes with no kv bound; this model's \
                    pass kind is not attention"
                .into(),
        );
    }
    let readings = model::readings();
    if readings.is_empty() {
        return Err("this model declares no readings; there is nothing here to draw with".into());
    }
    let text = readings
        .iter()
        .find(|r| r.takes_tokens && r.readout == model::ReadoutKind::Hidden)
        .cloned()
        .ok_or("this model has no text reading")?;
    let denoise = readings
        .iter()
        .find(|r| !r.takes_tokens && r.readout == model::ReadoutKind::Velocity)
        .cloned()
        .ok_or(
            "this model declares no token-less reading with a velocity readout, so nothing \
                here denoises",
        )?;
    if denoise.has_kv {
        return Err(format!(
            "reading `{}` binds a kv space; a denoise pass binds none",
            denoise.name
        )
        .into());
    }
    let decode = readings.iter().find(|r| r.name == "vae.decode").cloned();
    Ok(Roles {
        text,
        denoise,
        decode,
    })
}

/// The denoise reading's ports, sorted into the roles a sampler binds. Kind
/// is what says which is which (`PortKind` is the facts vocabulary); the
/// one name read here is `guidance`, which the design lists as a port role
/// of its own and which decides whether the model wants a scale or a second
/// lane pair.
struct Ports {
    latents: model::PortFact,
    context: model::PortFact,
    timestep: String,
    guidance: Option<String>,
    /// The convention `positions_for` fills the grids from.
    convention: model::PositionConvention,
    /// Which stream carries the latent, and which carries the prompt rows.
    image_stream: model::LaneStream,
    context_stream: model::LaneStream,
}

/// The first stream of `streams`, preferring `want` when it is listed;
/// `fallback` when the port lists none (a port every lane binds).
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
    let context = of_kind(model::PortKind::Context).ok_or(
        "this denoise reading declares no context port; a text-conditioned sampler has nowhere \
         to put the prompt",
    )?;
    let positions = of_kind(model::PortKind::AxisPositions).ok_or(
        "this denoise reading declares no axis-positions port; this sampler builds rotary \
         coordinates and the model takes none",
    )?;
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
    let image_stream = stream_of(
        &latents.streams,
        model::LaneStream::Image,
        model::LaneStream::Image,
    );
    let context_stream = stream_of(
        &context.streams,
        model::LaneStream::Text,
        model::LaneStream::Context,
    );
    Ok(Ports {
        latents,
        context,
        timestep,
        guidance,
        convention,
        image_stream,
        context_stream,
    })
}

/// Everything one lane binds, kept alive for the loop's lifetime: a channel
/// dropped while its pass is live is a port with nothing feeding it.
struct Lane {
    pass: ForwardPass,
    #[allow(dead_code)]
    held: Vec<Channel>,
}

/// Build one lane of a denoise group: state its stream and group, then bind
/// EXACTLY the ports that list its stream (or list none), each from its
/// kind. `latents` is the lane's own rows when it is the image lane;
/// `context` is the encoder's rows when it is the prompt lane. A port of a
/// kind this lane has no value for — Z-Image's `pad` flags, a second
/// context — is bound to zeros of its declared width, which is what "no
/// pad rows / no extra conditioning" means.
#[allow(clippy::too_many_arguments)]
fn lane(
    reading: &model::ReadingFact,
    ports: &Ports,
    stream: model::LaneStream,
    group: u32,
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
        };
        pass.input(&port.name, &ch)?;
        held.push(ch);
    }
    Ok(Lane { pass, held })
}

/// One prompt's two lanes: the encoder rows and the latent they condition.
struct Branch {
    context: Lane,
    image: Lane,
    /// The loop-carried latent, advanced by the image lane's epilogue (or
    /// `set` by the host on the CFG path).
    latent: Channel,
    /// `[rows, width]` published once a fire: the latent after this fire on
    /// the plain path, this fire's velocity on the CFG path.
    out: Channel,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let prompt = input
        .prompt
        .clone()
        .filter(|p| !p.trim().is_empty())
        .ok_or("pass `prompt`: this program draws what it is told to")?;
    let roles = roles()?;
    let ports = ports(&roles.denoise)?;

    // ---- the latent grid, out of `model::latent()` ------------------------
    let space = model::latent().ok_or(
        "this model states no latent space, so its denoise reading cannot be sized from pixels",
    )?;
    let cell_h = space.spatial_compression.max(1) * space.patch_h.max(1);
    let cell_w = space.spatial_compression.max(1) * space.patch_w.max(1);
    let height = (input.height.unwrap_or(1024) / cell_h).max(1) * cell_h;
    let width = (input.width.unwrap_or(1024) / cell_w).max(1) * cell_w;
    let grid_h = height / cell_h;
    let grid_w = width / cell_w;
    let rows = grid_h * grid_w;
    let row_width = ports.latents.width;
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && rows > max_rows {
        return Err(format!(
            "{width}x{height} is {rows} latent rows and this model carries {max_rows} a pass"
        )
        .into());
    }

    // ---- the schedule -----------------------------------------------------
    let fact = model::schedule().ok_or("this model states no schedule; nothing here denoises")?;
    // A DISTILLED ROW STATES ITS OWN STEP COUNT. `pinned_sigmas` is the
    // trajectory the model was distilled onto, so its length is the default;
    // an undistilled row gets eight, which is a policy and not a fact.
    let steps = match input.steps.filter(|s| *s > 0) {
        Some(steps) => steps,
        None if !fact.pinned_sigmas.is_empty() => {
            u32::try_from(fact.pinned_sigmas.len()).unwrap_or(8)
        }
        None => 8,
    };
    let sched = FlowMatchEuler::from_schedule(&fact, steps, Some(rows))?;

    // ---- guidance: a port, or a second lane pair --------------------------
    let guidance = input.guidance.unwrap_or(1.0);
    let negative = input
        .negative_prompt
        .clone()
        .filter(|p| !p.trim().is_empty());
    if negative.is_some() && ports.guidance.is_some() {
        return Err(format!(
            "reading `{}` declares a `guidance` port, so this row is guidance-distilled and a \
             negative prompt has no lane to ride; pass `guidance` alone",
            roles.denoise.name
        )
        .into());
    }
    let cfg = negative.is_some() && guidance > 1.0;

    // ---- the prompt -------------------------------------------------------
    let context = encode_text(&prompt, &roles.text.name).await?;
    let context_rows = context.shape().dims()[0];
    if context.shape().dims()[1] != ports.context.width {
        return Err(format!(
            "the text reading hands back {}-wide rows and the denoise context port takes {}",
            context.shape().dims()[1],
            ports.context.width
        )
        .into());
    }
    let uncond = match &negative {
        Some(text) if cfg => Some(encode_text(text, &roles.text.name).await?),
        _ => None,
    };

    // ---- the loop ---------------------------------------------------------
    let mut loops = DenoiseLoop::new(&sched);
    let velocity_width = roles.denoise.readout_width;
    let cells = (rows * row_width) as usize;
    let shape = [rows, row_width];
    let seed = input.seed.unwrap_or(0);

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
        let image_clock = loops.lane(&format!("{tag}_img"));
        let context_lane = lane(
            &roles.denoise,
            &ports,
            ports.context_stream,
            group,
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
            ports.image_stream,
            group,
            rows,
            Some(&latent),
            None,
            &positions_for(
                &ports.convention,
                LaneRows::Grid {
                    h: grid_h,
                    w: grid_w,
                },
                ctx_rows,
            ),
            &image_clock,
            guidance,
            &format!("{tag}_img"),
        )?;
        let out = Channel::new(shape, dtype::f32)
            .capacity(channel_capacity() as u32)
            .named(&format!("{tag}_out"));

        // A lane that only modulates advances its clock and nothing else.
        text_clock.drive(&context_lane.pass, |_| {});
        if cfg {
            // The CFG branches publish their velocity and leave the latent
            // alone; the host combines and `set`s both cells.
            let readback = out.clone();
            image_clock.drive(&image_lane.pass, move |_| {
                readback.put(intrinsics::velocity(velocity_width));
            });
        } else {
            let dts = loops.dts(tag);
            let rng = Channel::from(rng_state(seed)).named(&format!("{tag}_rng"));
            let x = latent.clone();
            let readback = out.clone();
            image_clock.drive(&image_lane.pass, move |k| {
                seed_or_step(
                    k,
                    &x,
                    &intrinsics::velocity(velocity_width),
                    &dts,
                    &rng,
                    shape,
                    Some(&readback),
                );
            });
        }
        branches.push(Branch {
            context: context_lane,
            image: image_lane,
            latent,
            out,
        });
    }

    // On the CFG path the host owns the trajectory, so it owns the draw too:
    // the same keyed formula the device would have used is not reachable
    // from here, so this is a plain seeded normal and the two paths do not
    // produce the same image from the same seed. Said here rather than
    // discovered later.
    let mut x: Vec<f32> = if cfg {
        host_normal(seed, cells)
    } else {
        Vec::new()
    };
    if cfg {
        for branch in &branches {
            branch.latent.set(x.as_slice())?;
        }
    }

    let mut last: Vec<f32> = Vec::new();
    for fire in 0..loops.fires() {
        let passes: Vec<&ForwardPass> = branches
            .iter()
            .flat_map(|b| [&b.context.pass, &b.image.pass])
            .collect();
        loops
            .fire(&passes)
            .with_context(|| format!("fire {fire}"))?;
        if cfg {
            // Fire `k >= 1` integrates step `k - 1`; fire 0 only carries the
            // seed the host already wrote.
            let cond = branches[0].out.take_host::<Vec<f32>>().await?;
            let uncond = branches[1].out.take_host::<Vec<f32>>().await?;
            if fire > 0 {
                let v = cfg_combine_host(&cond, &uncond, guidance);
                let dt = sched.dt(fire - 1);
                for (xi, vi) in x.iter_mut().zip(&v) {
                    *xi += dt * vi;
                }
                for branch in &branches {
                    branch.latent.set(x.as_slice())?;
                }
            }
            last = x.clone();
        } else {
            last = branches[0]
                .out
                .take_host::<Vec<f32>>()
                .await
                .with_context(|| format!("readback after fire {fire}"))?;
        }
    }
    loops.close();

    // ---- the way out ------------------------------------------------------
    let n = last.len().max(1) as f32;
    let mean = last.iter().sum::<f32>() / n;
    let var = last.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let name = input.out.unwrap_or_else(|| "image".to_string());
    let (file, decoded, bytes) = match &roles.decode {
        // The pixels never enter linear memory: the decode reading's frames
        // handle is streamed out by the host (D11). No family declares this
        // reading on `dev`; the arm is here so the exit exists the day one
        // does, and it refuses loudly rather than pretending.
        Some(decode) => {
            return Err(format!(
                "this model declares a `{}` reading, but its pixels seam is not in the facts \
                 vocabulary yet (`ReadoutKind` has no `Pixels`), so this program cannot drive \
                 it; re-run once the VAE arm lands",
                decode.name
            )
            .into());
        }
        None => {
            let mut bytes = Vec::with_capacity(last.len() * 4);
            for value in &last {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
            inferlet::session::send_file(&bytes);
            (format!("{name}.latent.f32"), false, len)
        }
    };

    Ok(Output {
        model: model::name(),
        architecture: model::architecture(),
        reading: roles.denoise.name.clone(),
        prompt,
        width,
        height,
        grid_h,
        grid_w,
        rows,
        row_width,
        latent_channels: space.channels,
        patch_h: space.patch_h,
        patch_w: space.patch_w,
        spatial_compression: space.spatial_compression,
        steps: sched.steps(),
        seed,
        guidance,
        cfg,
        sigmas: sched.sigmas.clone(),
        context_rows,
        decoded,
        file,
        bytes,
        non_finite: last.iter().filter(|v| !v.is_finite()).count() as u32,
        mean,
        std: var.sqrt(),
    })
}

/// `uncond + s.(cond - uncond)`, on the host: the twin of the device
/// `latent::cfg_combine`, stated so the CFG path has one formula and not
/// two.
fn cfg_combine_host(cond: &[f32], uncond: &[f32], s: f32) -> Vec<f32> {
    cond.iter()
        .zip(uncond)
        .map(|(c, u)| u + s * (c - u))
        .collect()
}

/// `n` standard-normal values from `seed`: splitmix64 for the uniforms and
/// Box-Muller for the shape. Only the CFG path uses it — the device draw
/// (`latent::noise`) is what a normal run seeds with, and the two do not
/// agree, which the caller is told.
fn host_normal(seed: u32, n: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(seed);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next().max(1e-12);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        if out.len() < n {
            out.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
        }
    }
    out
}
