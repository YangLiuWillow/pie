//! The pie half of the `z-image` goldens. Hands the family exactly what
//! `scripts/imagegen/zimage_golden.py` handed diffusers — the caption rows,
//! the patchified latent, the timestep and the three rotary coordinates of
//! every row — and reads back the refined caption and the velocity, as
//! JSON `scripts/imagegen/zimage_parity.py` turns into an `.npz` under the
//! golden's own key names.
//!
//! Two fires per case, in the order the reference's forward runs them:
//!
//! 1. **`refine`** — one `Context` lane carrying `caption` (`[L32, cap]`),
//!    `pad` (`[L32, 1]`) and `positions` (`[L32, 3]`); its epilogue reads
//!    `hidden(dim)`, the refined caption with its pad rows.
//! 2. **`denoise`** — an `Image` lane (`latents`, `pad`, `positions`,
//!    `timestep`) and a `Context` lane (`context` = the refine readout,
//!    `positions`, `timestep`) in one group; the image lane's epilogue
//!    reads `velocity(64)`.
//!
//! The refine readout goes through the host between the two: a parity
//! harness wants both numbers on disk anyway (the refined caption is the
//! first thing to diff when the velocity disagrees), and a seeded channel
//! attaches to one pass only, so the second fire seeds its own.
//!
//! Row counts, pad flags and positions are the CASE's: the family cannot
//! grow a lane, so the harness pads to the 32-row multiple and states the
//! reference's own coordinates (caption row `j` at `(1 + j, 0, 0)`, pads
//! included; image patch `(a, b)` at `(L32 + 1, a, b)`; image pads at
//! `(0, 0, 0)`). The timestep is the family's port convention — the
//! scheduler's `σ·1000`, which the plan flips to the reference's `1000 −
//! t` — so the harness converts the golden's transformer-side `t` first.
use inferlet::latent::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    case: Option<String>,
    #[serde(default)]
    case_file: Option<String>,
    #[serde(default)]
    case_0: Option<String>,
    #[serde(default)]
    case_1: Option<String>,
    #[serde(default)]
    case_2: Option<String>,
    #[serde(default)]
    case_3: Option<String>,
    #[serde(default)]
    case_4: Option<String>,
    #[serde(default)]
    case_5: Option<String>,
    #[serde(default)]
    case_6: Option<String>,
    #[serde(default)]
    case_7: Option<String>,
    #[serde(default)]
    refine_only: bool,
}

/// The reference's fixed inputs, flattened row-major and already padded.
#[derive(Deserialize)]
struct Case {
    /// `[image_rows, patch_features]`: the patchified latent, pad rows
    /// appended (their content is irrelevant; the plan overwrites them).
    latents: Vec<f32>,
    image_rows: u32,
    patch_features: u32,
    /// `[image_rows]`: `0.0` real, `1.0` pad.
    image_pad: Vec<f32>,
    /// `[image_rows, 3]`.
    image_positions: Vec<f32>,
    /// `[caption_rows, caption_width]`, pad rows appended.
    caption: Vec<f32>,
    caption_rows: u32,
    caption_width: u32,
    /// `[caption_rows]`.
    caption_pad: Vec<f32>,
    /// `[caption_rows, 3]`.
    caption_positions: Vec<f32>,
    /// The SCHEDULER timestep `σ·1000` the family's port takes.
    timestep: f32,
}

#[derive(Serialize)]
struct Output {
    /// `[caption_rows, dim]`: the refine readout, pads included.
    refined: Vec<f32>,
    caption_rows: u32,
    dim: u32,
    /// `[image_rows, patch_features]`: the velocity, pads included, in the
    /// family's sign (the reference's `-model_out`).
    velocity: Vec<f32>,
    image_rows: u32,
    patch_features: u32,
}

struct Reading {
    name: String,
    ports: Vec<String>,
    axes: u32,
    readout_width: u32,
}

/// The reading's port names read off `model::readings()`, so a renamed port
/// fails here with the model's own vocabulary instead of at the host.
fn reading(readout: model::ReadoutKind, wants: &[&str]) -> Result<Reading> {
    let fact = model::readings()
        .into_iter()
        .find(|reading| !reading.takes_tokens && reading.readout == readout)
        .ok_or_else(|| {
            format!("this model declares no token-less reading with a {readout:?} readout")
        })?;
    if fact.has_kv {
        return Err(format!(
            "reading `{}` binds a kv space; the parity pass binds none",
            fact.name
        )
        .into());
    }
    let mut ports = Vec::new();
    for want in wants {
        let port = fact
            .ports
            .iter()
            .find(|port| port.name == *want)
            .ok_or_else(|| format!("reading `{}` declares no port `{want}`", fact.name))?;
        ports.push(port.name.clone());
    }
    let axes = fact
        .ports
        .iter()
        .find(|port| port.name == "positions")
        .map(|port| port.width)
        .unwrap_or(3);
    Ok(Reading {
        name: fact.name.clone(),
        ports,
        axes,
        readout_width: fact.readout_width,
    })
}

/// The `refine` reading: one context lane, the refined caption back.
async fn refine(case: &Case, pipe: &Pipeline) -> Result<(Vec<f32>, u32)> {
    let r = reading(model::ReadoutKind::Hidden, &["caption", "pad", "positions"])?;
    let rows = case.caption_rows;
    let pass = ForwardPass::new();
    pass.reading(&r.name)?;
    pass.stream(LaneStream::Context)?;
    let caption =
        Channel::from_shaped([rows, case.caption_width], case.caption.as_slice()).named("caption");
    let pad = Channel::from_shaped([rows, 1], case.caption_pad.as_slice()).named("cap_pad");
    let positions =
        Channel::from_shaped([rows, r.axes], case.caption_positions.as_slice()).named("cap_pos");
    pass.input(&r.ports[0], &caption)?;
    pass.input(&r.ports[1], &pad)?;
    pass.input(&r.ports[2], &positions)?;
    let width = r.readout_width;
    let out = Channel::new([rows, width], dtype::f32).named("refined");
    let readback = out.clone();
    pass.epilogue(move || {
        readback.put(intrinsics::hidden(width));
    });
    pass.submit(pipe).context("refine lane")?;
    let rows_out: Vec<f32> = out.take_host().await?;
    Ok((rows_out, width))
}

/// The `denoise` reading: the image lane and the refined-caption lane in
/// one group, the velocity back off the image lane.
async fn denoise(case: &Case, refined: &[f32], dim: u32, pipe: &Pipeline) -> Result<Vec<f32>> {
    let r = reading(
        model::ReadoutKind::Velocity,
        &["latents", "pad", "context", "timestep", "positions"],
    )?;
    if r.readout_width != case.patch_features {
        return Err(format!(
            "the model reads a {}-wide velocity and the case carries {}-wide patch rows",
            r.readout_width, case.patch_features
        )
        .into());
    }
    let (p_latents, p_pad, p_context, p_timestep, p_positions) = (
        &r.ports[0],
        &r.ports[1],
        &r.ports[2],
        &r.ports[3],
        &r.ports[4],
    );
    let group = 0;

    // One timestep cell PER PASS: a seeded channel attaches to one pass only.
    let t_img = Channel::from([case.timestep]).named("t_img");
    let t_ctx = Channel::from([case.timestep]).named("t_ctx");

    // The caption lane: the refine readout, pads included, no pad port.
    let context = ForwardPass::new();
    context.reading(&r.name)?;
    context.stream(LaneStream::Context)?;
    context.group(group)?;
    let ctx = Channel::from_shaped([case.caption_rows, dim], refined).named("context");
    let ctx_pos = Channel::from_shaped(
        [case.caption_rows, r.axes],
        case.caption_positions.as_slice(),
    )
    .named("ctx_pos");
    context.input(p_context, &ctx)?;
    context.input(p_positions, &ctx_pos)?;
    context.input(p_timestep, &t_ctx)?;

    // The image lane: the latents in, the velocity out.
    let rows = case.image_rows;
    let width = case.patch_features;
    let image = ForwardPass::new();
    image.reading(&r.name)?;
    image.stream(LaneStream::Image)?;
    image.group(group)?;
    let x = Channel::from_shaped([rows, width], case.latents.as_slice()).named("latents");
    let pad = Channel::from_shaped([rows, 1], case.image_pad.as_slice()).named("img_pad");
    let img_pos =
        Channel::from_shaped([rows, r.axes], case.image_positions.as_slice()).named("img_pos");
    image.input(p_latents, &x)?;
    image.input(p_pad, &pad)?;
    image.input(p_positions, &img_pos)?;
    image.input(p_timestep, &t_img)?;
    let out = Channel::new([rows, width], dtype::f32).named("velocity");
    let readback = out.clone();
    image.epilogue(move || {
        readback.put(intrinsics::velocity(width));
    });

    context.submit(pipe).context("context lane")?;
    image.submit(pipe).context("image lane")?;
    out.take_host::<Vec<f32>>().await.map_err(Into::into)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Attention {
        return Err("z-image is an attention-kind pass with no kv bound".into());
    }
    let pieces: String = [
        &input.case_0,
        &input.case_1,
        &input.case_2,
        &input.case_3,
        &input.case_4,
        &input.case_5,
        &input.case_6,
        &input.case_7,
    ]
    .into_iter()
    .flatten()
    .map(String::as_str)
    .collect();
    let text = match (&input.case, &input.case_file) {
        (Some(text), _) => text.clone(),
        (None, Some(name)) => std::fs::read_to_string(format!("/scratch/{name}"))
            .map_err(|why| format!("reading /scratch/{name}: {why}"))?,
        (None, None) if !pieces.is_empty() => pieces,
        (None, None) => {
            return Err("pass `case` (json), `case_0..7` (its pieces) or `case_file`".into());
        }
    };
    let case: Case =
        inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}"))?;
    let check = |what: &str, have: usize, want: u32| -> Result<()> {
        if have != want as usize {
            return Err(
                format!("`{what}` carries {have} numbers, the case's shape wants {want}").into(),
            );
        }
        Ok(())
    };
    check(
        "latents",
        case.latents.len(),
        case.image_rows * case.patch_features,
    )?;
    check("image_pad", case.image_pad.len(), case.image_rows)?;
    check(
        "image_positions",
        case.image_positions.len(),
        case.image_rows * 3,
    )?;
    check(
        "caption",
        case.caption.len(),
        case.caption_rows * case.caption_width,
    )?;
    check("caption_pad", case.caption_pad.len(), case.caption_rows)?;
    check(
        "caption_positions",
        case.caption_positions.len(),
        case.caption_rows * 3,
    )?;
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && case.image_rows + case.caption_rows > max_rows {
        return Err(format!(
            "{} rows exceed the model's {max_rows} rows a pass",
            case.image_rows + case.caption_rows
        )
        .into());
    }

    let pipe = Pipeline::new();
    let (refined, dim) = refine(&case, &pipe).await?;
    let velocity = if input.refine_only {
        Vec::new()
    } else {
        denoise(&case, &refined, dim, &pipe).await?
    };
    pipe.close();
    Ok(Output {
        refined,
        caption_rows: case.caption_rows,
        dim,
        velocity,
        image_rows: case.image_rows,
        patch_features: case.patch_features,
    })
}
