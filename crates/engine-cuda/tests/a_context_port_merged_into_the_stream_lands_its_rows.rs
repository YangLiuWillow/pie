//! **A FLOAT PORT MERGED STRAIGHT INTO A STREAM (`Value::merge(vec![ctx_port,
//! image])`, the Wan/mini-dit shape) LANDS ITS LANES' ROWS IN THE MERGED
//! COLUMN, AND AN f32 LANE CHAIN WITH `add_bias` AND `split_rows` ARMS
//! BODIES THAT ANSWER THEIR EAGER WALK.**
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<n> cargo test -p engine-cuda --features cuda \
//!   --test a_context_port_merged_into_the_stream_lands_its_rows
//! ```
//!
//! Two things this plan states that the double-block miniature does not: a
//! merge whose arm is a runtime input (no node writes that arm's rows, so
//! the engine lands the port's rows in the merged column before the walk,
//! zeros for a lane that fed nothing — the arming synthetics — so two walks
//! of one composition read the same bytes and the golden holds), and the
//! adaLN chain `sinusoid → silu → matmul → add_bias → split_rows →
//! modulate / gated_residual_add` kept in f32 on the lane axis. Loads under
//! the default knobs (bodies armed and golden-checked), fires two requests
//! against a host f32 reference. Skips when no device is present.

#![cfg(feature = "cuda")]

mod common_dit;

use common_dit::{
    Lcg, NAME, Rig, StreamFacts, WIDTH, Weights, assert_close, attach, bf, frame, lane,
};
use engine::Engine;
use engine::fire::LaneStream;
use model_dsl::{
    Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, Stream, Trace, Value, Weight,
    ops, seam, trace_hybrid,
};

const FREQ: u32 = 16;

struct MergedBlock;

impl ForwardHybrid for MergedBlock {
    type Facts = StreamFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<StreamFacts>) -> Value {
        let (txt, img) = inputs.split(&StreamFacts::on(Stream::Text));
        // The text lanes' rows come straight off a context port; the image
        // lanes' off an embedded latent — one merged stream.
        let x_txt = txt.context(0, WIDTH);
        let embed = Weight::sym("embed", [u64::from(WIDTH), u64::from(WIDTH)], Dtype::Bf16);
        let x_img = ops::linear::matmul(&img.latents(0, WIDTH, Dtype::Bf16), &embed);
        let x = Value::merge(vec![x_txt, x_img]);
        // adaLN in f32 on the lane axis, biased and split.
        let t = inputs.lane_vector(0, 1);
        let emb = ops::elemwise::silu(&ops::elemwise::sinusoid(&t, FREQ, 10_000.0, true, 1.0));
        let ada = Weight::sym("ada", [3 * u64::from(WIDTH), u64::from(FREQ)], Dtype::Bf16);
        let ada_bias = Weight::sym("ada_bias", [3 * u64::from(WIDTH)], Dtype::Bf16);
        let m = ops::elemwise::add_bias(&ada_bias, &ops::linear::matmul(&emb, &ada));
        let (ss, gate) = ops::layout::split_rows(&m, 2 * WIDTH);
        let lanes = inputs.request_of_token();
        let h = ops::elemwise::modulate(
            &ops::elemwise::layernorm_no_scale(&x, 1e-6),
            &ss,
            Some(&lanes),
            ModulateForm::ScaleShift,
        );
        let w1 = Weight::sym("w1", [u64::from(WIDTH), u64::from(WIDTH)], Dtype::Bf16);
        let y = ops::linear::matmul(&h, &w1);
        let r = ops::elemwise::gated_residual_add(&x, &gate, &y, Some(&lanes));
        seam::at(seam::VELOCITY, &[&r]);
        r
    }
}

fn plan() -> Trace {
    trace_hybrid(NAME, &MergedBlock, Platform::Cuda)
}

/// One lane's velocity rows on the host: `x` is the lane's rows as the
/// merged column holds them (the context cell, or `latent · embedᵀ`).
fn reference(weights: &Weights, x: &[f32], rows: usize, timestep: f32) -> Vec<f32> {
    let w = WIDTH as usize;
    let half = (FREQ / 2) as usize;
    let angles: Vec<f32> = (0..half)
        .map(|i| timestep * (-10_000f32.ln() * i as f32 / half as f32).exp())
        .collect();
    let mut emb: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    emb.extend(angles.iter().map(|a| a.sin()));
    let emb: Vec<f32> = emb.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    let ada = weights.get("ada");
    let bias = weights.get("ada_bias");
    let m: Vec<f32> = (0..3 * w)
        .map(|c| {
            let mut acc = 0f32;
            for i in 0..FREQ as usize {
                acc = emb[i].mul_add(ada[c * FREQ as usize + i], acc);
            }
            acc + bias[c]
        })
        .collect();
    let w1 = weights.get("w1");
    let mut out = vec![0f32; rows * w];
    for r in 0..rows {
        let row = &x[r * w..(r + 1) * w];
        let mean = row.iter().sum::<f32>() / w as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        let h: Vec<f32> = (0..w)
            .map(|c| bf(((row[c] - mean) * inv).mul_add(1.0 + m[c], m[w + c])))
            .collect();
        for c in 0..w {
            let mut acc = 0f32;
            for i in 0..w {
                acc = h[i].mul_add(w1[c * w + i], acc);
            }
            let y = bf(acc);
            out[r * w + c] = bf(m[2 * w + c].mul_add(y, row[c]));
        }
    }
    out
}

#[test]
fn the_merged_context_rows_and_the_f32_lane_chain_land_the_reference() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let plan = plan();
    let weights = Weights::random(&plan, 29);
    let mut rig = Rig::load_plan(plan, &weights, 64, vec![16, 64]);
    let w = WIDTH as usize;
    let mut rng = Lcg::seeded(31);
    let (text_rows, image_rows) = (3usize, 6usize);
    let mut lanes = Vec::new();
    let mut attachments = Vec::new();
    let mut want = Vec::new();
    for req in 0..2u32 {
        let timestep = 0.3 + 0.2 * req as f32;
        let text: Vec<f32> = (0..text_rows * w).map(|_| bf(rng.unit())).collect();
        let latent: Vec<f32> = (0..image_rows * w).map(|_| bf(rng.unit())).collect();
        // The image rows as the merged column holds them.
        let embed = weights.get("embed");
        let mut image = vec![0f32; image_rows * w];
        for r in 0..image_rows {
            for c in 0..w {
                let mut acc = 0f32;
                for i in 0..w {
                    acc = latent[r * w + i].mul_add(embed[c * w + i], acc);
                }
                image[r * w + c] = bf(acc);
            }
        }
        let t = rig.lane(text_rows as u32);
        let i = rig.lane(image_rows as u32);
        for (handles, rows, cell) in [(&t, text_rows, &text), (&i, image_rows, &latent)] {
            rig.publish(handles.instance, 0, cell);
            rig.publish(handles.instance, 1, &[timestep]);
            rig.publish(handles.instance, 2, &vec![0.0; rows * 2]);
        }
        let slot = 2 * req;
        let mut text_lane = lane(slot, &t, LaneStream::Text, req);
        // The text lane feeds the context port (this plan's text input);
        // `lane` wires port 0 of the latents kind, so retarget it.
        for feed in &mut text_lane.ports {
            if feed.kind == engine::fire::PortKind::Latents {
                feed.kind = engine::fire::PortKind::Context;
            }
        }
        let mut image_lane = lane(slot + 1, &i, LaneStream::Image, req);
        for feed in &mut image_lane.ports {
            if feed.kind == engine::fire::PortKind::Latents {
                feed.port = 0;
            }
        }
        // This plan reads no positions port: drop that feed.
        text_lane
            .ports
            .retain(|f| f.kind != engine::fire::PortKind::AxisPositions);
        image_lane
            .ports
            .retain(|f| f.kind != engine::fire::PortKind::AxisPositions);
        attachments.push(attach(lanes.len() as u32, &t));
        lanes.push(text_lane);
        attachments.push(attach(lanes.len() as u32, &i));
        lanes.push(image_lane);
        want.push(reference(&weights, &text, text_rows, timestep));
        want.push(reference(&weights, &image, image_rows, timestep));
    }
    let mut ticket = rig
        .engine
        .submit(&frame(lanes, attachments))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    for (at, readout) in ticket.steps[0].readouts.iter().enumerate() {
        assert_close(&readout.values, &want[at], &format!("lane {at} velocity"));
    }
}
