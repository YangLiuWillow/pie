//! **A HAND-WRITTEN BIDIRECTIONAL ENCODER LAYER — per-layer bucket
//! embedding, `elementwise.relative_bucket_bias`, one `attention.ragged`
//! over per-lane CSRs under `RaggedMask::RelativeBias`, a `hidden` export —
//! FIRES THROUGH THE REAL `Engine` API WITH THREE LANES OF UNEQUAL LENGTH
//! AND LANDS WHAT A HOST f32 REFERENCE (THE T5 BUCKET FUNCTION INCLUDED)
//! COMPUTES.**
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<n> cargo test -p engine-cuda --features cuda \
//!   --test a_bidirectional_attention_adds_its_relative_bias_through_the_engine
//! ```
//!
//! The plan is `model_dsl` text (no catalog family); its weights are random
//! and written to a serving artifact the engine loads. The bias table is a
//! `[heads, 2·max_len − 1]` plan constant computed on the device from the
//! bf16 bucket embedding and handed whole to the attention. Every lane
//! feeds its rows from a latent channel cell; the hidden rows come back as
//! `LaneReadout { seam: Hidden }` through `settle_frame`. Skips when no
//! device is present.
//!
//! The first test loads with `Graphs::Shaped` — eager, on graph-shaped
//! schedules. The second is the same fire under the load's default knobs
//! (bodies armed at load and replayed from the bucket-64 body): a kv-less
//! plan's per-lane tables (the packing CSRs, the group and slot ids) are
//! staged at the body's lane ceiling, padded with empty segments, rather
//! than at the fire's own lane count — a shorter staging left the tail as
//! the last arming fire's bounds, which the replayed ragged kernel read as
//! live groups over rows 16..32.

#![cfg(feature = "cuda")]

mod common_encoder;

use common_encoder::{Lcg, Rig, WIDTH, Weights, assert_close, attach, bf, frame, lane, reference};
use engine::Engine;
use engine::fire::ReadoutSeam;
use engine_cuda::Graphs;

/// Three lanes: one shorter than the bucket function's exact band, one past
/// it, one past the bucket doubling, so both bucket regimes are exercised
/// and the fire (47 rows) crosses the 16- and 32-row buckets.
const ROWS: [usize; 3] = [5, 13, 29];

fn fire_and_compare(graphs: Graphs) {
    let weights = Weights::random(&common_encoder::trace(), 13);
    let mut rig = Rig::load(&weights, 64, vec![16, 32, 64], graphs);

    let mut rng = Lcg::seeded(17);
    let inputs: Vec<Vec<f32>> = ROWS
        .iter()
        .map(|&n| (0..n * WIDTH as usize).map(|_| bf(rng.unit())).collect())
        .collect();
    let want: Vec<Vec<f32>> = ROWS
        .iter()
        .zip(&inputs)
        .map(|(&n, x)| reference(&weights, x, n))
        .collect();

    let mut lanes = Vec::new();
    let mut attachments = Vec::new();
    for (at, (&n, x)) in ROWS.iter().zip(&inputs).enumerate() {
        let handles = rig.lane(n as u32);
        rig.publish(handles.instance, 0, x);
        lanes.push(lane(at as u32, &handles, at as u32));
        attachments.push(attach(at as u32, &handles));
    }
    let mut ticket = rig
        .engine
        .submit(&frame(lanes, attachments))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts.len(), 3);
    for (at, readout) in readouts.iter().enumerate() {
        assert_eq!(readout.seam, ReadoutSeam::Hidden);
        assert_eq!(readout.width, WIDTH);
        assert_close(
            &readout.values,
            &want[at],
            &format!("lane {at} hidden rows"),
        );
    }
}

#[test]
fn three_lanes_of_unequal_length_land_the_host_reference() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    fire_and_compare(Graphs::Shaped);
}

#[test]
fn the_same_fire_lands_the_host_reference_from_an_armed_body() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    fire_and_compare(Graphs::On);
}
