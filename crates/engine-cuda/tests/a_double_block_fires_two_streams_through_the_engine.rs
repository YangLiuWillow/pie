//! **A HAND-WRITTEN MM-DiT MINIATURE — two streams per request, one joint
//! `attention.ragged` over merged per-stream q/k/v, adaLN from a lane-vector
//! timestep, `rope_axes`, `pack_rows`/`unpack_rows`, a `velocity` export —
//! FIRES THROUGH THE REAL `Engine` API WITH TWO REQUESTS (FOUR LANES) AND
//! LANDS WHAT A HOST f32 REFERENCE COMPUTES, ON BOTH READBACK ROADS.**
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<n> cargo test -p engine-cuda --features cuda \
//!   --test a_double_block_fires_two_streams_through_the_engine
//! ```
//!
//! The plan is `model_dsl` text (no catalog family); its weights are random
//! and written to a serving artifact the engine loads under its default
//! knobs, so the load's arming pass records bodies and diffs each against
//! its eager walk (the golden) before the first fire. Every lane feeds its
//! latents, its timestep and its rope positions from channel cells; the
//! velocity comes back as `LaneReadout { seam: Velocity }` rows through
//! `settle_frame`, and as `latent + velocity` through an eta epilogue that
//! reads the `velocity()` intrinsic. Skips when no device is present.

#![cfg(feature = "cuda")]

mod common_dit;

use common_dit::{
    HostRequest, Lcg, Rig, WIDTH, Weights, assert_close, attach, bf, frame, lane, reference,
};
use engine::Engine;
use engine::fire::{LaneStream, ReadoutSeam};

/// One request's host-side inputs, random.
fn request(rng: &mut Lcg, text_rows: usize, image_rows: usize) -> HostRequest {
    let w = WIDTH as usize;
    let rows = text_rows + image_rows;
    HostRequest {
        text: (0..text_rows * w).map(|_| bf(rng.unit())).collect(),
        image: (0..image_rows * w).map(|_| bf(rng.unit())).collect(),
        text_rows,
        image_rows,
        timestep: 0.5 + 0.5 * rng.unit(),
        positions: (0..rows)
            .map(|r| [r as f32, (r % 3) as f32 + 0.25 * rng.unit()])
            .collect(),
    }
}

#[test]
fn the_double_block_lands_the_host_reference_on_four_lanes() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&common_dit::trace(), 7);
    let mut rig = Rig::load(&weights, 64, vec![16, 32, 64]);
    assert!(
        rig.profile().has_velocity,
        "the plan plants a velocity seam"
    );
    assert_eq!(rig.profile().velocity_width, WIDTH);

    let mut rng = Lcg::seeded(3);
    let requests = [request(&mut rng, 3, 5), request(&mut rng, 4, 6)];
    let want: Vec<(Vec<f32>, Vec<f32>)> = requests.iter().map(|r| reference(&weights, r)).collect();

    // One instance per lane; the text lane and the image lane of a request
    // share a group.
    let mut lanes = Vec::new();
    let mut attachments = Vec::new();
    let mut handles = Vec::new();
    for (at, req) in requests.iter().enumerate() {
        let text = rig.lane(req.text_rows as u32);
        let image = rig.lane(req.image_rows as u32);
        rig.publish(text.instance, 0, &req.text);
        rig.publish(text.instance, 1, &[req.timestep]);
        rig.publish(
            text.instance,
            2,
            &req.positions[..req.text_rows]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<f32>>(),
        );
        rig.publish(image.instance, 0, &req.image);
        rig.publish(image.instance, 1, &[req.timestep]);
        rig.publish(
            image.instance,
            2,
            &req.positions[req.text_rows..]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<f32>>(),
        );
        let slot = (2 * at) as u32;
        // Submitted image-first for one request, text-first for the other:
        // the packed order is the group's, not the submission's.
        if at == 0 {
            lanes.push(lane(slot, &image, LaneStream::Image, at as u32));
            lanes.push(lane(slot + 1, &text, LaneStream::Text, at as u32));
            attachments.push(attach(lanes.len() as u32 - 2, &image));
            attachments.push(attach(lanes.len() as u32 - 1, &text));
            handles.push((image, text));
        } else {
            lanes.push(lane(slot, &text, LaneStream::Text, at as u32));
            lanes.push(lane(slot + 1, &image, LaneStream::Image, at as u32));
            attachments.push(attach(lanes.len() as u32 - 2, &text));
            attachments.push(attach(lanes.len() as u32 - 1, &image));
            handles.push((text, image));
        }
    }
    let mut ticket = rig
        .engine
        .submit(&frame(lanes, attachments))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts.len(), 4);
    for readout in readouts {
        assert_eq!(readout.seam, ReadoutSeam::Velocity);
        assert_eq!(readout.width, WIDTH);
    }
    // Submission order: [r0.image, r0.text, r1.text, r1.image].
    assert_close(&readouts[0].values, &want[0].1, "request 0 image velocity");
    assert_close(&readouts[1].values, &want[0].0, "request 0 text velocity");
    assert_close(&readouts[2].values, &want[1].0, "request 1 text velocity");
    assert_close(&readouts[3].values, &want[1].1, "request 1 image velocity");

    // The epilogue's road: `latent + velocity` on each lane's out channel.
    let (r0_image, r0_text) = &handles[0];
    let (r1_text, r1_image) = &handles[1];
    let stepped = |values: &[f32], latent: &[f32]| -> Vec<f32> {
        values.iter().zip(latent).map(|(v, x)| v + x).collect()
    };
    let out = rig.take(r0_image.instance, 3);
    assert_close(
        &out,
        &stepped(&want[0].1, &requests[0].image),
        "request 0 image epilogue",
    );
    let out = rig.take(r0_text.instance, 3);
    assert_close(
        &out,
        &stepped(&want[0].0, &requests[0].text),
        "request 0 text epilogue",
    );
    let out = rig.take(r1_text.instance, 3);
    assert_close(
        &out,
        &stepped(&want[1].0, &requests[1].text),
        "request 1 text epilogue",
    );
    let out = rig.take(r1_image.instance, 3);
    assert_close(
        &out,
        &stepped(&want[1].1, &requests[1].image),
        "request 1 image epilogue",
    );
}
