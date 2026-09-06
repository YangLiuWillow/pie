//! The pie half of the HunyuanImage 3 miniature golden: the two TOKEN-AXIS
//! readings of `hunyuan_image_3`, driven exactly as
//! `scripts/imagegen/hy3_golden.py --mini` drove the reference.
//!
//! # What runs here, and what cannot yet
//!
//! A denoise step of this family is THREE fires (`crates/models/src/
//! hunyuan_image_3/forward.rs` says why): `image.in` (the conv `patch_embed`
//! on the voxel axis), `denoise` (the trunk), `image.out` (the conv
//! `final_layer`). **The two voxel arms are not drivable from a guest
//! today**: the SDK has no channel-fed `Voxels` port and no `pixels()`
//! intrinsic to read a voxel-axis seam back with — the same gap
//! `tests/inferlets/text-to-image` records for `z_image`'s `vae.decode`. So
//! this program drives the two token-axis readings and takes the image
//! head's rows from the golden:
//!
//! ```text
//! encode    the prompt prefix, causal, its K/V ARE the sequence  -> logits
//! denoise   the canvas: h*w image rows whose input is the golden's
//!           patch_embed output, plus ONE <timestep> row the plan lands
//!           from the lane's timestep vector, bidirectional over
//!           [prefix | canvas] against the pages `encode` wrote  -> hidden
//! ```
//!
//! # The canvas lane
//!
//! `h*w + 1` rows, the special row LAST — a lane's row order is not its
//! sequence order, and every row's place is stated three times over: its
//! rotary coordinates (`positions`, the float port), its KV position (the
//! geometry's `positions`) and its write slot (`w_slot`/`w_off`). The
//! `special` port flags the one row the plan overwrites with
//! `timestep_emb(t)`; every other row takes its `latents` row.
//!
//! The mask is the reference's generalized causal attention, built here as
//! the dense `[rows, keys]` bool rectangle the engine's `AttnMask` port
//! takes: an image row sees every key of `[prefix | <timestep> | canvas]`,
//! the `<timestep>` row sees the prefix and itself. Nothing past the canvas
//! is in the KV at all, so the reference's trailing `<eoi>` needs no
//! masking off.
//!
//! # The prefix is frozen
//!
//! `encode` runs once and its pages are never written again; the denoise
//! pass declares them readable and writes only the canvas's own slots. That
//! is the whole of design D10's "prefix KV frozen across denoise steps" —
//! resubmitting this pass with a new timestep and new latents is a step.

use inferlet::eta::diffusion::prelude::*;
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
}

/// The golden's inputs, flattened row-major.
#[derive(Deserialize)]
struct Case {
    /// The text prefix's ids: `<bos> … <boi> <img_size> <img_ratio>`.
    prefix: Vec<i32>,
    /// `[prefix.len(), 2]` — `(y, x')`, the `x` already through the
    /// family's rotary scale.
    prefix_positions: Vec<f32>,
    /// The `<img>` id and the `<timestep>` id.
    img_id: i32,
    timestep_id: i32,
    /// `h·w`.
    image_rows: u32,
    /// `[image_rows + 1, 2]`, the canvas lane's row order: the image rows
    /// in raster order, then the `<timestep>` row.
    canvas_positions: Vec<f32>,
    /// `[image_rows + 1]`: the SEQUENCE position of each canvas row.
    canvas_sequence: Vec<u32>,
    /// `[image_rows, hidden]` — `patch_embed(x_t, time_embed(t))`, which
    /// the `image.in` reading would have produced.
    rows: Vec<f32>,
    hidden: u32,
    /// The scheduler timestep `σ·1000`.
    timestep: f32,
}

#[derive(Serialize)]
struct Output {
    /// `[prefix.len()]`: the argmax of the causal pass's logits, and the
    /// logit it won with.
    encode_argmax: Vec<i32>,
    encode_max: Vec<f32>,
    /// `[image_rows + 1, hidden]` in the canvas lane's row order — the
    /// trunk's rows with `ln_f` NOT applied, which is what `final_layer`
    /// consumes.
    canvas_hidden: Vec<f32>,
    image_rows: u32,
    hidden: u32,
}

fn reading(name: &'static str) -> Result<model::ReadingFact> {
    model::readings()
        .into_iter()
        .find(|reading| reading.name == name)
        .ok_or_else(|| format!("this model declares no `{name}` reading").into())
}

fn port_of(reading: &model::ReadingFact, name: &str) -> Result<String> {
    reading
        .ports
        .iter()
        .find(|port| port.name == name)
        .map(|port| port.name.clone())
        .ok_or_else(|| format!("reading `{}` declares no port `{name}`", reading.name).into())
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Diffusion {
        return Err("hunyuan_image_3 is a `forward-diffusion` row; the bound model is not".into());
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

    let encode = reading("encode")?;
    let denoise = reading("denoise")?;
    if !(encode.has_kv && denoise.has_kv) {
        return Err("both token-axis readings of this family bind a kv space".into());
    }
    if denoise.readout_width != case.hidden {
        return Err(format!(
            "the model reads a {}-wide canvas row and the case carries {}",
            denoise.readout_width, case.hidden
        )
        .into());
    }
    let axes = encode
        .ports
        .iter()
        .find(|port| port.name == "positions")
        .map(|port| port.width)
        .ok_or("the `encode` reading declares no `positions` port")?;

    let prefix_len = u32::try_from(case.prefix.len()).map_err(|_| "a prefix too long")?;
    let canvas_len = case.image_rows + 1;
    if case.canvas_sequence.len() as u32 != canvas_len {
        return Err("one sequence position per canvas row".into());
    }
    // The KV holds the prefix, the `<timestep>` row and the canvas — and
    // nothing after them, which is what makes the reference's trailing
    // `<eoi>` invisible to the image rows without a mask column.
    let kv_len = case
        .canvas_sequence
        .iter()
        .copied()
        .max()
        .ok_or("an empty canvas")?
        + 1;
    let page_size = kv_page_size();
    let max_pages = kv_len.div_ceil(page_size).max(1);

    let ws = WorkingSet::new();
    ws.reserve(max_pages).context("reserve KV")?;
    let pipe = Pipeline::new();

    // ---- the prefix: one causal `encode` fire, whose pages then freeze ---
    let toks = Channel::from(case.prefix.clone()).named("prefix_ids");
    let indptr = Channel::from([0u32, prefix_len]).named("prefix_indptr");
    let e_pos = Channel::from_iter(0..prefix_len).named("prefix_kv_positions");
    let e_rope = Channel::from_shaped([prefix_len, axes], case.prefix_positions.as_slice())
        .named("prefix_positions");
    let pages = Channel::from_iter(0..max_pages).named("pages");
    let e_indptr =
        Channel::from([0u32, prefix_len.div_ceil(page_size)]).named("prefix_page_indptr");
    let e_slot = Channel::from_iter((0..prefix_len).map(|p| p / page_size)).named("prefix_w_slot");
    let e_off = Channel::from_iter((0..prefix_len).map(|p| p % page_size)).named("prefix_w_off");
    let e_len = Channel::from([prefix_len]).named("prefix_kv_len");
    let e_argmax = Channel::new([prefix_len], dtype::i32).named("encode_argmax");
    let e_max = Channel::new([prefix_len], dtype::f32).named("encode_max");

    let prefill = ForwardPass::new();
    prefill.reading(&encode.name)?;
    prefill.canvas(Mode::Encode)?;
    prefill.embed(&toks, &indptr)?;
    prefill.input(&port_of(&encode, "positions")?, &e_rope)?;
    prefill.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &e_len,
            pages: &pages,
            page_indptr: &e_indptr,
            w_slot: &e_slot,
            w_off: &e_off,
            positions: &e_pos,
            mask: None,
        },
    )?;
    let (argmax_out, max_out) = (e_argmax.clone(), e_max.clone());
    prefill.epilogue(move || {
        let logits = intrinsics::logits();
        argmax_out.put(cast(reduce_argmax(&logits), dtype::i32));
        max_out.put(reduce_max(&logits));
    });
    prefill.submit(&pipe).context("the prefix prefill")?;
    let encode_argmax = e_argmax.take_host::<Vec<i32>>().await?;
    let encode_max = e_max.take_host::<Vec<f32>>().await?;

    // ---- the canvas: one `denoise` fire over the frozen prefix -----------
    let mut ids: Vec<i32> = vec![case.img_id; case.image_rows as usize];
    ids.push(case.timestep_id);
    let special: Vec<f32> = (0..canvas_len)
        .map(|row| f32::from(row == case.image_rows))
        .collect();
    // `[canvas_len, hidden]`: the image head's rows, then one row the plan
    // overwrites (its bytes are never read).
    let mut rows = case.rows.clone();
    rows.resize((canvas_len * case.hidden) as usize, 0.0);
    // The generalized causal mask, dense `[rows, keys]` row-major: an image
    // row sees every key, the `<timestep>` row sees itself and the prefix.
    let t_at = case.canvas_sequence[case.image_rows as usize];
    let mask: Vec<bool> = (0..canvas_len)
        .flat_map(|row| {
            let bound = if row == case.image_rows { t_at } else { kv_len - 1 };
            (0..kv_len).map(move |key| key <= bound)
        })
        .collect();

    let d_toks = Channel::from(ids).named("canvas_ids");
    let d_indptr = Channel::from([0u32, canvas_len]).named("canvas_indptr");
    let d_rows = Channel::from_shaped([canvas_len, case.hidden], rows.as_slice()).named("canvas_rows");
    let d_special = Channel::from_shaped([canvas_len, 1], special.as_slice()).named("canvas_special");
    let d_t = Channel::from([case.timestep]).named("canvas_timestep");
    let d_rope = Channel::from_shaped([canvas_len, axes], case.canvas_positions.as_slice())
        .named("canvas_positions");
    let d_pos = Channel::from(case.canvas_sequence.clone()).named("canvas_kv_positions");
    let d_indptr_pages = Channel::from([0u32, kv_len.div_ceil(page_size)]).named("canvas_page_indptr");
    let d_slot = Channel::from_iter(case.canvas_sequence.iter().map(|p| p / page_size))
        .named("canvas_w_slot");
    let d_off = Channel::from_iter(case.canvas_sequence.iter().map(|p| p % page_size))
        .named("canvas_w_off");
    let d_len = Channel::from([kv_len]).named("canvas_kv_len");
    let d_mask = Channel::from_shaped([canvas_len, kv_len], mask.as_slice()).named("canvas_mask");
    let out = Channel::new([canvas_len, case.hidden], dtype::f32).named("canvas_hidden");

    let step = ForwardPass::new();
    step.reading(&denoise.name)?;
    step.canvas(Mode::Denoise)?;
    step.stream(LaneStream::Image)?;
    step.embed(&d_toks, &d_indptr)?;
    step.input(&port_of(&denoise, "latents")?, &d_rows)?;
    step.input(&port_of(&denoise, "special")?, &d_special)?;
    step.input(&port_of(&denoise, "timestep")?, &d_t)?;
    step.input(&port_of(&denoise, "positions")?, &d_rope)?;
    step.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &d_len,
            pages: &pages,
            page_indptr: &d_indptr_pages,
            w_slot: &d_slot,
            w_off: &d_off,
            positions: &d_pos,
            mask: Some(&d_mask),
        },
    )?;
    let hidden = case.hidden;
    let readback = out.clone();
    step.epilogue(move || {
        readback.put(intrinsics::hidden(hidden));
    });
    step.submit(&pipe).context("the denoise step")?;
    let canvas_hidden = out.take_host::<Vec<f32>>().await?;
    pipe.close();

    Ok(Output {
        encode_argmax,
        encode_max,
        canvas_hidden,
        image_rows: case.image_rows,
        hidden: case.hidden,
    })
}
