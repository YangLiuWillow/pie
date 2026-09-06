//! The pie half of the `mini-dit` golden. Hands the model exactly what
//! `scripts/imagegen/mini_dit_ref.py` handed PyTorch — the patch rows, the
//! caption rows, the cross-attention context, the timestep and the three
//! rotary coordinates per row — and reads back the velocity the head
//! predicts, as JSON `scripts/imagegen/mini_dit_parity.py` turns into an
//! `.npz` under the golden's own key names.
//!
//! Two modes: one step against `mini_dit_dump_bf16.npz`'s `velocity`, and
//! the four-step Euler schedule against `mini_dit_euler_bf16.npz`'s
//! `euler.v{0..3}` / `euler.x{1..4}`.
//!
//! # THREE LANES, ONE GROUP, ONE FIRE
//!
//! `mini-dit`'s `denoise` reading declares three streams (D2), so one step
//! is three passes — caption, image, context — each stating its
//! [`stream`](inferlet::eta::attention::ForwardPass::stream) and all in one
//! [`group`](inferlet::eta::attention::ForwardPass::group), which is what
//! lets the caption rows join the image rows' attention and the context
//! rows serve as block 2's keys. Neither `attention` nor `embed` is called
//! on any of them: a denoise reading declares no kv space and no tokens,
//! and the image lane's row count comes from its latents channel.
//!
//! Only the image lane reads out — the caption stream ends after block 1,
//! and the head is image-only — so only it carries an epilogue.
//!
//! # WHY THE EULER LOOP IS ON THE HOST
//!
//! `latent-probe` integrates on the device, which is what a real sampler
//! does (D4) and what a real guest should copy. A parity harness wants the
//! opposite: every step's velocity AND every step's latent, at the
//! reference's own numbers, with nothing folded together. So this program
//! takes the velocity after each fire and steps the latent itself. The
//! arithmetic is `FlowMatchEuler`'s and the sigmas are the model's own
//! (`model::schedule()`), so what is checked is still the schedule the
//! family declares.
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
    euler: bool,
}

/// One batch element of the reference's fixed inputs, flattened row-major.
/// The harness writes the same numbers `mini_dit_ref.py` fed torch: the
/// patchified latent (never the `[C, H, W]` array), the caption rows, the
/// context rows, one timestep, and the `(t, h, w)` coordinates of every row.
#[derive(Deserialize)]
struct Case {
    /// `[image_rows, patch_features]`, row-major. The reference's `patches`.
    latents: Vec<f32>,
    image_rows: u32,
    patch_features: u32,
    /// `[text_rows, text_width]`.
    text: Vec<f32>,
    text_rows: u32,
    text_width: u32,
    /// `[context_rows, context_width]`.
    context: Vec<f32>,
    context_rows: u32,
    context_width: u32,
    /// `[rows, 3]` each, in the packed order the plan reads (`Stream::Text`
    /// before `Stream::Image`).
    text_positions: Vec<f32>,
    image_positions: Vec<f32>,
    /// One step's timestep, or — under `--euler` — the schedule.
    timestep: f32,
    #[serde(default)]
    sigmas: Vec<f32>,
    #[serde(default)]
    t_scale: f32,
    #[serde(default)]
    steps: u32,
}

#[derive(Serialize)]
struct Output {
    /// `[image_rows, patch_features]` — the reference's `final.tokens`
    /// before its unpatchify. The harness unpatchifies.
    velocity: Vec<f32>,
    image_rows: u32,
    patch_features: u32,
    /// The sigmas actually stepped, so a schedule mismatch reads as one.
    #[serde(default)]
    sigmas: Vec<f32>,
    /// `--euler`: the velocity of every step, and the latent after each.
    #[serde(default)]
    euler_v: Vec<Vec<f32>>,
    #[serde(default)]
    euler_x: Vec<Vec<f32>>,
}

/// The port names this family's `denoise` reading declares. Read off
/// `model::readings()` rather than typed in, so a renamed port fails here
/// with the model's own vocabulary instead of at the host.
struct Ports {
    latents: String,
    text: String,
    context: String,
    timestep: String,
    positions: String,
    axes: u32,
    velocity_width: u32,
}

fn ports(reading: &model::ReadingFact) -> Result<Ports> {
    let named = |name: &str| -> Result<&model::PortFact> {
        reading
            .ports
            .iter()
            .find(|port| port.name == name)
            .ok_or_else(|| format!("reading `{}` declares no port `{name}`", reading.name).into())
    };
    Ok(Ports {
        latents: named("latents")?.name.clone(),
        text: named("text")?.name.clone(),
        context: named("context")?.name.clone(),
        timestep: named("timestep")?.name.clone(),
        positions: named("positions")?.name.clone(),
        axes: named("positions")?.width,
        velocity_width: reading.readout_width,
    })
}

/// One denoise step: three lanes, one group, one fire, one velocity.
async fn step(
    case: &Case,
    ports: &Ports,
    reading: &str,
    pipe: &Pipeline,
    group: u32,
    latents: &[f32],
    timestep: f32,
) -> Result<Vec<f32>> {
    let tag = |what: &str| format!("{what}_g{group}");
    let rows = case.image_rows;
    let width = case.patch_features;
    // One timestep cell PER PASS: a seeded channel attaches to one pass
    // only (the runtime's channel-role rule), so the two modulating lanes
    // each carry their own copy of the same scalar.
    let t_txt = Channel::from([timestep]).named(&tag("t_txt"));
    let t_img = Channel::from([timestep]).named(&tag("t_img"));

    // The caption lane. Its rows are fixed random 256-wide embeddings —
    // this family has no text encoder, which is what makes it an M0 fixture
    // and not a model. It modulates, so it carries the timestep too.
    let caption = ForwardPass::new();
    caption.reading(reading)?;
    caption.stream(LaneStream::Text)?;
    caption.group(group)?;
    let txt = Channel::from_shaped([case.text_rows, case.text_width], case.text.as_slice())
        .named(&tag("text"));
    let txt_pos =
        Channel::from_shaped([case.text_rows, ports.axes], case.text_positions.as_slice())
            .named(&tag("txt_pos"));
    caption.input(&ports.text, &txt)?;
    caption.input(&ports.positions, &txt_pos)?;
    caption.input(&ports.timestep, &t_txt)?;

    // The context lane: block 2's cross-attention keys and values, and
    // nothing else. No positions — the Wan contract gives cross-attention
    // no rope — and no timestep, since its class never modulates.
    let context = ForwardPass::new();
    context.reading(reading)?;
    context.stream(LaneStream::Context)?;
    context.group(group)?;
    let ctx = Channel::from_shaped(
        [case.context_rows, case.context_width],
        case.context.as_slice(),
    )
    .named(&tag("ctx"));
    context.input(&ports.context, &ctx)?;

    // The image lane: the latents in, the velocity out.
    let out = Channel::new([rows, width], dtype::f32).named(&tag("velocity"));
    let image = ForwardPass::new();
    image.reading(reading)?;
    image.stream(LaneStream::Image)?;
    image.group(group)?;
    let x = Channel::from_shaped([rows, width], latents).named(&tag("latents"));
    let img_pos = Channel::from_shaped([rows, ports.axes], case.image_positions.as_slice())
        .named(&tag("img_pos"));
    image.input(&ports.latents, &x)?;
    image.input(&ports.positions, &img_pos)?;
    image.input(&ports.timestep, &t_img)?;
    let velocity_width = ports.velocity_width;
    let readback = out.clone();
    image.epilogue(move || {
        // The denoise reading's `logits()`: `[image rows, C·p²]`.
        readback.put(intrinsics::velocity(velocity_width));
    });

    caption.submit(pipe).context("caption lane")?;
    context.submit(pipe).context("context lane")?;
    image.submit(pipe).context("image lane")?;
    out.take_host::<Vec<f32>>().await.map_err(Into::into)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Attention {
        return Err("mini-dit is an attention-kind pass with no kv bound".into());
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

    let reading = model::readings()
        .into_iter()
        .find(|reading| !reading.takes_tokens && reading.readout == model::ReadoutKind::Velocity)
        .ok_or("this model declares no token-less reading with a velocity readout")?;
    if reading.has_kv {
        return Err(format!(
            "reading `{}` binds a kv space; the parity pass binds none",
            reading.name
        )
        .into());
    }
    if reading.readout_width != case.patch_features {
        return Err(format!(
            "the model reads a {}-wide velocity and the case carries {}-wide patch rows",
            reading.readout_width, case.patch_features
        )
        .into());
    }
    let max_rows = model::max_latent_rows();
    if max_rows > 0 && case.image_rows > max_rows {
        return Err(format!(
            "{} rows exceed the model's {max_rows} latent rows a pass",
            case.image_rows
        )
        .into());
    }
    let ports = ports(&reading)?;
    let pipe = Pipeline::new();
    let mut out = Output {
        velocity: Vec::new(),
        image_rows: case.image_rows,
        patch_features: case.patch_features,
        sigmas: Vec::new(),
        euler_v: Vec::new(),
        euler_x: Vec::new(),
    };

    if !input.euler {
        out.velocity = step(
            &case,
            &ports,
            &reading.name,
            &pipe,
            0,
            &case.latents,
            case.timestep,
        )
        .await?;
        pipe.close();
        return Ok(out);
    }

    // The schedule the model declares, checked against the one the golden
    // was generated under: a silent disagreement here would look like a
    // numerics bug in the trunk.
    let steps = case.steps.max(1);
    let sched = match model::schedule() {
        Some(fact) => FlowMatchEuler::from_schedule(&fact, steps, Some(case.image_rows))?,
        None => FlowMatchEuler::from_sigmas(case.sigmas.clone(), case.t_scale.max(1.0) as u32),
    };
    if !case.sigmas.is_empty() {
        let drift = sched
            .sigmas
            .iter()
            .zip(&case.sigmas)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if sched.sigmas.len() != case.sigmas.len() || drift > 1e-6 {
            return Err(format!(
                "the model's schedule is {:?} and the golden's is {:?}",
                sched.sigmas, case.sigmas
            )
            .into());
        }
    }
    out.sigmas = sched.sigmas.clone();

    let mut x = case.latents.clone();
    for i in 0..steps {
        let v = step(
            &case,
            &ports,
            &reading.name,
            &pipe,
            i,
            &x,
            sched.timestep(i),
        )
        .await?;
        let dt = sched.dt(i);
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi += dt * vi;
        }
        out.euler_v.push(v);
        out.euler_x.push(x.clone());
    }
    out.velocity = out.euler_v.last().cloned().unwrap_or_default();
    pipe.close();
    Ok(out)
}
