//! The pie half of the `mini-dit` golden. Hands the model exactly what
//! `scripts/imagegen/mini_dit_ref.py` handed PyTorch — the patch rows, the
//! caption rows, the cross-attention context, the timestep and the three
//! rotary coordinates per row — and reads back the velocity the head
//! predicts, as JSON the harness turns into an `.npz`.
//!
//! Two modes: one step against `mini_dit_dump_bf16.npz`'s `velocity`, and
//! the four-step Euler schedule against `mini_dit_euler_bf16.npz`'s
//! `euler.v{0..3}` / `euler.x{1..4}`.
//!
//! # THIS PROGRAM IS WRITTEN AGAINST A WIT THAT HAS NOT LANDED
//!
//! Design D1 adds two verbs to `pie:inferlet/forward`, and D2 two more:
//!
//! ```text
//! /// Which of the family's declared readings this pass runs.
//! reading: func(name: string) -> result<_, error>;
//! /// Bind a channel to one of the reading's declared input ports, read
//! /// at every submit.
//! input:   func(port: string, ch: borrow<channel>) -> result<_, error>;
//! /// Which stream this pass's rows are (D2: a lane is (request, stream)).
//! stream:  func(s: stream) -> result<_, error>;
//! /// Which attention group they join — the lanes of one request.
//! group:   func(id: u32) -> result<_, error>;
//! ```
//!
//! Until the runtime carries them the body below is behind the
//! `imagegen-wit` feature (off by default), so the fixture workspace still
//! builds. What it assumes, and what the runtime agent should check against
//! the WIT it actually lands:
//!
//! * **one `ForwardPass` per lane**, three per step (caption, image,
//!   context), tied into one fire by a shared `group` — the reading of D2
//!   under which "a request submits several lanes, one per stream";
//! * **a lane's row count comes from its ports' channel shapes**, since a
//!   denoise reading binds no `embed`;
//! * **`intrinsics::velocity()`** is the image lane's readout, gated by the
//!   reading's export seams exactly as `logits()` is by `has_logits`;
//! * **`attention(..)` is not called at all** — a denoise reading declares
//!   no kv space and must refuse it by name.
//!
//! The Euler update is done on the HOST here (take the velocity, step the
//! latent, feed it back) rather than in the epilogue. D4 puts it in the
//! epilogue once `velocity()` and the latent cell are both reachable from a
//! program; a host loop over four steps is the same arithmetic and is what
//! a parity harness wants to see anyway.
#![cfg(feature = "imagegen-wit")]

use inferlet::eta::attention::prelude::*;
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
/// patchified latent (never the `[C, H, W]` latent), the caption rows, the
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
    /// The `(t, h, w)` of every caption row then every image row, in the
    /// packed order the plan reads (`Stream::Text` before `Stream::Image`).
    text_positions: Vec<f32>,
    image_positions: Vec<f32>,
    /// One step's timestep, or — in `--euler` — the schedule.
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
    /// `[image_rows, patch_features]` — one step's velocity, the reference's
    /// `final.tokens` before its unpatchify. The harness unpatchifies.
    velocity: Vec<f32>,
    image_rows: u32,
    patch_features: u32,
    /// `--euler`: the velocity of every step, and the latent after each.
    #[serde(default)]
    euler_v: Vec<Vec<f32>>,
    #[serde(default)]
    euler_x: Vec<Vec<f32>>,
}

/// The three lanes of one denoise step, submitted as one group.
struct Step<'a> {
    case: &'a Case,
    pipe: &'a Pipeline,
    group: u32,
}

impl Step<'_> {
    /// One fire; answers the `[image_rows * patch_features]` velocity.
    async fn run(&self, latents: &[f32], timestep: f32) -> Result<Vec<f32>> {
        let case = self.case;
        let tag = |what: &str| format!("{what}_g{}", self.group);
        let rows = case.image_rows;
        let width = case.patch_features;

        // The caption lane: `Context` port 0, 256 wide. Its rows carry the
        // fixed random embeddings the reference used — this model has no
        // text encoder, which is exactly why it is a useful M0 fixture.
        let caption = ForwardPass::new();
        caption.reading("denoise")?;
        caption.stream(Stream::Text)?;
        caption.group(self.group)?;
        caption.input(
            "text",
            &Channel::from_shaped([case.text_rows, case.text_width], case.text.as_slice())
                .named(&tag("text")),
        )?;
        caption.input(
            "positions",
            &Channel::from_shaped([case.text_rows, 3u32], case.text_positions.as_slice())
                .named(&tag("txt_pos")),
        )?;
        caption.input(
            "timestep",
            &Channel::from([timestep]).named(&tag("txt_t")),
        )?;

        // The context lane: `Context` port 1, 512 wide, keys and values for
        // block 2's cross-attention and nothing else. No positions: the Wan
        // contract gives cross-attention no rope.
        let context = ForwardPass::new();
        context.reading("denoise")?;
        context.stream(Stream::Context)?;
        context.group(self.group)?;
        context.input(
            "context",
            &Channel::from_shaped(
                [case.context_rows, case.context_width],
                case.context.as_slice(),
            )
            .named(&tag("ctx")),
        )?;

        // The image lane: the latents in, the velocity out.
        let velocity = Channel::new([rows, width], dtype::f32).named(&tag("velocity"));
        let image = ForwardPass::new();
        image.reading("denoise")?;
        image.stream(Stream::Image)?;
        image.group(self.group)?;
        image.input(
            "latents",
            &Channel::from_shaped([rows, width], latents).named(&tag("latents")),
        )?;
        image.input(
            "positions",
            &Channel::from_shaped([rows, 3u32], case.image_positions.as_slice())
                .named(&tag("img_pos")),
        )?;
        image.input("timestep", &Channel::from([timestep]).named(&tag("img_t")))?;
        let out = velocity.clone();
        image.epilogue(move || {
            // The denoise reading's `logits()`: `[image rows, C·p²]`.
            out.put(intrinsics::velocity());
        });

        // One group, one fire: the lanes are submitted together and the
        // group table is what lets the caption rows join the image rows'
        // attention.
        caption.submit(self.pipe).context("caption lane")?;
        context.submit(self.pipe).context("context lane")?;
        image.submit(self.pipe).context("image lane")?;
        velocity.take_host::<Vec<f32>>().await.map_err(Into::into)
    }
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

    let pipe = Pipeline::new();
    let mut out = Output {
        velocity: Vec::new(),
        image_rows: case.image_rows,
        patch_features: case.patch_features,
        euler_v: Vec::new(),
        euler_x: Vec::new(),
    };

    if !input.euler {
        let step = Step {
            case: &case,
            pipe: &pipe,
            group: 0,
        };
        out.velocity = step.run(&case.latents, case.timestep).await?;
        pipe.close();
        return Ok(out);
    }

    // The reference's flow-matching schedule: `t = sigma * t_scale`,
    // `x += (sigma[i+1] - sigma[i]) * v`. Host arithmetic, one group per
    // step, because a step's latents are the previous step's answer.
    if case.sigmas.len() < case.steps as usize + 1 {
        return Err("an Euler run wants steps + 1 sigmas".into());
    }
    let mut x = case.latents.clone();
    for step in 0..case.steps {
        let sigma = case.sigmas[step as usize];
        let next = case.sigmas[step as usize + 1];
        let v = Step {
            case: &case,
            pipe: &pipe,
            group: step,
        }
        .run(&x, sigma * case.t_scale)
        .await?;
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi += (next - sigma) * vi;
        }
        out.euler_v.push(v);
        out.euler_x.push(x.clone());
    }
    out.velocity = out.euler_v.last().cloned().unwrap_or_default();
    pipe.close();
    Ok(out)
}
