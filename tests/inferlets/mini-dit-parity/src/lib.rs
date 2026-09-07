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
//! **ONE PIPELINE PER LANE.** A group is a fact about a FIRE: the three
//! passes join one attention only when they are members of one step. The
//! scheduler seals a frame when every live pipeline has submitted and never
//! seats two passes of one pipeline in one step, so three passes down one
//! pipeline are three fires — each lane attending alone, the context lane's
//! keys never seen. Three pipelines, one pass each, submitted back to back,
//! are what the wait-all seal composes into one fire.
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
    /// The family's `PIE_MINI_DIT_TAP` bisection knob is set: the readout
    /// is an intermediate, at whatever width the model declares.
    #[serde(default)]
    tap: bool,
    /// Under `tap`, a tap whose rectangle spans the caption rows too: read
    /// the caption lane out as well, into `text_tap`.
    #[serde(default)]
    tap_text: bool,
    /// Classifier-free guidance, ON THE DEVICE. Runs the step as SIX lanes
    /// in one fire — two attention groups of three, group 0 holding the
    /// case's context and group 1 holding a zeroed one — and combines the
    /// two branches' velocities inside group 0's epilogue with
    /// `intrinsics::peer_velocity`. Reports the combine at `cfg_scale`
    /// alongside each branch's own answer, so the caller can check the two
    /// identities that need no golden: at `s = 1` the combine IS the
    /// conditional branch, at `s = 0` it IS the unconditional one.
    #[serde(default)]
    cfg: bool,
    /// The guidance scale the `cfg` combine runs at. Default 1.0, the
    /// identity on the conditional branch.
    #[serde(default)]
    cfg_scale: Option<f32>,
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
    /// `--tap_text`: the caption lane's rows of the tapped rectangle,
    /// `[text_rows, width]`.
    #[serde(default)]
    text_tap: Vec<f32>,
    #[serde(default)]
    text_rows: u32,
    /// `--cfg`: the guided velocity group 0's epilogue computed on the
    /// device, `[image_rows, patch_features]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cfg_guided: Vec<f32>,
    /// `--cfg`: the conditional branch's own velocity (group 0), for the
    /// `s = 1` identity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cfg_cond: Vec<f32>,
    /// `--cfg`: the unconditional branch's own velocity (group 1), for the
    /// `s = 0` identity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cfg_uncond: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cfg_scale: Option<f32>,
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

/// The three lanes' pipelines, one each, so the three passes of a step seal
/// into one fire.
struct Pipes {
    caption: Pipeline,
    context: Pipeline,
    image: Pipeline,
}

impl Pipes {
    fn new() -> Pipes {
        Pipes {
            caption: Pipeline::new(),
            context: Pipeline::new(),
            image: Pipeline::new(),
        }
    }

    fn close(&self) {
        self.caption.close();
        self.context.close();
        self.image.close();
    }
}

/// One BRANCH of a guided step: the three lanes of one attention group,
/// built like `step`'s but with the group's own context rows. Returns the
/// three passes and the channel the image lane's epilogue publishes into —
/// `None` for a branch whose velocity is only ever read by its peer, which
/// needs no epilogue of its own.
#[allow(clippy::too_many_arguments)]
fn branch(
    case: &Case,
    ports: &Ports,
    reading: &str,
    group: u32,
    context: &[f32],
    latents: &[f32],
    timestep: f32,
    other: u32,
    guide: Option<(u32, f32, bool)>,
) -> Result<(ForwardPass, ForwardPass, ForwardPass, Channel)> {
    let tag = |what: &str| format!("{what}_b{group}");
    let rows = case.image_rows;
    let width = ports.velocity_width;

    // GUIDANCE IS MUTUAL. Every lane of a branch names the other branch, so
    // the two groups gather under one cohort key and seal into ONE fire — a
    // peer in another fire is not on this fire's velocity plane at all. Only
    // the guided lane READS its peer; naming is what makes them one step.
    let partner = other;
    let caption = ForwardPass::new();
    caption.reading(reading)?;
    caption.stream(LaneStream::Text)?;
    caption.group(group)?;
    caption.peer(partner)?;
    let txt = Channel::from_shaped([case.text_rows, case.text_width], case.text.as_slice())
        .named(&tag("text"));
    let txt_pos =
        Channel::from_shaped([case.text_rows, ports.axes], case.text_positions.as_slice())
            .named(&tag("txt_pos"));
    let t_txt = Channel::from([timestep]).named(&tag("t_txt"));
    caption.input(&ports.text, &txt)?;
    caption.input(&ports.positions, &txt_pos)?;
    caption.input(&ports.timestep, &t_txt)?;

    // The branch's own context: the case's rows for the conditional branch,
    // zeros for the unconditional one. This is the ONLY difference between
    // the two branches, which is what makes the guidance real rather than a
    // combine of one answer with itself.
    let ctx_pass = ForwardPass::new();
    ctx_pass.reading(reading)?;
    ctx_pass.stream(LaneStream::Context)?;
    ctx_pass.group(group)?;
    ctx_pass.peer(partner)?;
    let ctx =
        Channel::from_shaped([case.context_rows, case.context_width], context).named(&tag("ctx"));
    ctx_pass.input(&ports.context, &ctx)?;

    let image = ForwardPass::new();
    image.reading(reading)?;
    image.stream(LaneStream::Image)?;
    image.group(group)?;
    image.peer(partner)?;
    let x = Channel::from_shaped([rows, case.patch_features], latents).named(&tag("latents"));
    let img_pos = Channel::from_shaped([rows, ports.axes], case.image_positions.as_slice())
        .named(&tag("img_pos"));
    let t_img = Channel::from([timestep]).named(&tag("t_img"));
    image.input(&ports.latents, &x)?;
    image.input(&ports.positions, &img_pos)?;
    image.input(&ports.timestep, &t_img)?;
    let out = Channel::new([rows, width], dtype::f32).named(&tag("velocity"));
    let readback = out.clone();
    match guide {
        // The guided branch: its epilogue holds BOTH predictions — its own
        // off `velocity()`, its peer's off `peer_velocity()` — and publishes
        // the combine. No host round trip, no second fire.
        Some((_, scale, conditional)) => image.epilogue(move || {
            readback.put(&guided_velocity(width, scale, conditional));
        }),
        None => image.epilogue(move || {
            readback.put(intrinsics::velocity(width));
        }),
    }
    Ok((caption, ctx_pass, image, out))
}

/// One guided denoise step: SIX lanes, two attention groups, ONE fire.
///
/// Group 0 carries the case's context, group 1 a zeroed one, and group 0's
/// image lane names group 1 as its peer — so its epilogue reads both
/// branches' velocities off the fire-wide velocity plane and combines them
/// at `scale`. The two groups do not attend each other, which is what makes
/// them two independent denoisings rather than one wider one.
///
/// Also publishes each branch's own velocity, so the caller can check the
/// two identities that need no golden: `s = 1` is the conditional branch
/// exactly, `s = 0` is the unconditional one exactly.
async fn guided_step(
    case: &Case,
    ports: &Ports,
    reading: &str,
    scale: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let zeros = vec![0.0f32; case.context.len()];
    let (c_txt, c_ctx, c_img, guided) = branch(
        case,
        ports,
        reading,
        0,
        case.context.as_slice(),
        &case.latents,
        case.timestep,
        1,
        Some((1, scale, true)),
    )?;
    let (u_txt, u_ctx, u_img, uncond) = branch(
        case,
        ports,
        reading,
        1,
        zeros.as_slice(),
        &case.latents,
        case.timestep,
        0,
        None,
    )?;
    // A seventh lane would be cheaper, but the conditional branch's own
    // velocity is not readable beside its combine — one epilogue publishes
    // one thing. So the unguided answer comes from `step`, which is the
    // same three lanes at the same timestep, and the identity it proves is
    // the stronger one for it.
    let pipes: Vec<Pipeline> = (0..6).map(|_| Pipeline::new()).collect();
    // ONE PIPELINE PER LANE, all six submitted before any fires: a pipeline
    // is serial, and the frame seals when every live one has submitted.
    c_txt.submit(&pipes[0]).context("cond caption")?;
    c_ctx.submit(&pipes[1]).context("cond context")?;
    c_img.submit(&pipes[2]).context("cond image")?;
    u_txt.submit(&pipes[3]).context("uncond caption")?;
    u_ctx.submit(&pipes[4]).context("uncond context")?;
    u_img.submit(&pipes[5]).context("uncond image")?;
    let guided = guided.take_host::<Vec<f32>>().await?;
    let uncond = uncond.take_host::<Vec<f32>>().await?;
    for pipe in &pipes {
        pipe.close();
    }
    Ok((guided, Vec::new(), uncond))
}

/// One denoise step: three lanes, one group, one fire, one velocity.
async fn step(
    case: &Case,
    ports: &Ports,
    reading: &str,
    pipes: &Pipes,
    group: u32,
    latents: &[f32],
    timestep: f32,
    text_tap: Option<&mut Vec<f32>>,
) -> Result<Vec<f32>> {
    let tag = |what: &str| format!("{what}_g{group}");
    let rows = case.image_rows;
    // The readout's width is the reading's, which is the patch features —
    // or, under the family's tap knob, the tapped rectangle's.
    let width = ports.velocity_width;
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
    // The bisection knob's caption readout: the same intrinsic, off the
    // caption lane's own rows of the tapped rectangle.
    let text_out = text_tap.as_ref().map(|_| {
        let out = Channel::new([case.text_rows, ports.velocity_width], dtype::f32)
            .named(&tag("text_tap"));
        let readback = out.clone();
        let width = ports.velocity_width;
        caption.epilogue(move || {
            readback.put(intrinsics::velocity(width));
        });
        out
    });

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
    let x = Channel::from_shaped([rows, case.patch_features], latents).named(&tag("latents"));
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

    caption.submit(&pipes.caption).context("caption lane")?;
    context.submit(&pipes.context).context("context lane")?;
    image.submit(&pipes.image).context("image lane")?;
    let velocity = out.take_host::<Vec<f32>>().await?;
    if let (Some(sink), Some(text_out)) = (text_tap, text_out) {
        *sink = text_out.take_host::<Vec<f32>>().await?;
    }
    Ok(velocity)
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
    if !(input.tap || input.tap_text) && reading.readout_width != case.patch_features {
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
    let pipes = Pipes::new();
    let mut out = Output {
        velocity: Vec::new(),
        image_rows: case.image_rows,
        patch_features: case.patch_features,
        sigmas: Vec::new(),
        euler_v: Vec::new(),
        euler_x: Vec::new(),
        text_tap: Vec::new(),
        text_rows: case.text_rows,
        cfg_guided: Vec::new(),
        cfg_cond: Vec::new(),
        cfg_uncond: Vec::new(),
        cfg_scale: None,
    };

    // Guidance runs one step, six lanes, two groups — never a schedule.
    if input.cfg {
        let scale = input.cfg_scale.unwrap_or(1.0);
        // The conditional branch alone FIRST, as the unguided three-lane
        // step: the same rows at the same timestep with the same context.
        // Before the guided step, and with its pipelines closed, because a
        // frame seals over the pipelines that are LIVE — three idle ones
        // would seal a six-lane fire down to whichever group was ready, and
        // the peer would not be in it.
        let mut text_tap = Vec::new();
        let cond = step(
            &case,
            &ports,
            &reading.name,
            &pipes,
            0,
            &case.latents,
            case.timestep,
            input.tap_text.then_some(&mut text_tap),
        )
        .await?;
        pipes.close();
        let (guided, _, uncond) = guided_step(&case, &ports, &reading.name, scale).await?;
        out.velocity = cond.clone();
        out.cfg_guided = guided;
        out.cfg_cond = cond;
        out.cfg_uncond = uncond;
        out.cfg_scale = Some(scale);
        return Ok(out);
    }

    if !input.euler {
        let mut text_tap = Vec::new();
        out.velocity = step(
            &case,
            &ports,
            &reading.name,
            &pipes,
            0,
            &case.latents,
            case.timestep,
            input.tap_text.then_some(&mut text_tap),
        )
        .await?;
        out.text_tap = text_tap;
        pipes.close();
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
            &pipes,
            i,
            &x,
            sched.timestep(i),
            None,
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
    pipes.close();
    Ok(out)
}
