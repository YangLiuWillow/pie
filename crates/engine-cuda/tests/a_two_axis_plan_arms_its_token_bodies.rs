//! **A PLAN THAT STATES VOXEL ROWS STILL ARMS ITS TOKEN BODIES: ARMING IS
//! PER AXIS.** (design D8)
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<n> cargo test -p engine-cuda --features cuda \
//!   --test a_two_axis_plan_arms_its_token_bodies -- --nocapture
//! ```
//!
//! `serve::load` used to downgrade `bodies` wholesale for any plan with a
//! voxel axis — which served a flagship's whole DiT eagerly, at ~470 kernel
//! launches of host time a step, for no reason but the VAE standing beside it
//! in the same artifact. The eagerness belongs to the voxel REGIONS: the
//! arming pass fires synthetics that carry no clip, so a voxel window it sees
//! has zero rows and would read as capturable, and a spatial launch reads no
//! window seat, so no replay could retire its padding. `Windows::admit_axes`
//! says exactly that and no more — every region on `RowAxis::Voxels` is an
//! island, every token region is judged as it always was.
//!
//! The plan here is the smallest thing that can tell the two apart: one DiT
//! block on the token axis under reading 0, one convolution on the voxel axis
//! under reading 1. The claim is that the load arms bodies at all (it armed
//! none before), and that both arms still answer — the token lane through a
//! body, the voxel lane eagerly, in the same load.
//!
//! Skipped at run time with no device, as the other device gates are.

#![cfg(feature = "cuda")]

mod common_two_axis;

use common_two_axis::{
    C_IN, C_OUT, Lcg, Rig, WIDTH, Weights, assert_close, attach, bf, conv_reference, dit_lane,
    frame, pixel_epilogue, trace, vae_lane,
};
use engine::Engine;
use engine::fire::{ReadoutSeam, StepVoxels};
use eta_ir::container::HostRole;

const CLIP: [u32; 3] = [1, 4, 6];
const ROWS: u32 = 8;

const fn voxels() -> u32 {
    CLIP[0] * CLIP[1] * CLIP[2]
}

#[test]
fn the_token_regions_arm_while_the_voxel_regions_stay_eager() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&trace(), 0x2b);
    let mut rig = Rig::load(&weights, 32, vec![16, 32], voxels() + 8);

    // THE CLAIM. The arming pass ran at load and recorded bodies; before
    // arming was per axis this was zero for any plan with a voxel axis.
    let stats = rig
        .engine
        .shell()
        .expect("the load holds a shell")
        .body_stats();
    println!("[two-axis] {stats}");
    assert!(
        stats.census.bodies > 0,
        "a plan stating voxel rows armed no token body: {stats}"
    );
    assert!(
        stats.tally.armed_at_load > 0,
        "the arming pass recorded nothing at load: {stats}"
    );

    // And the two arms still answer, in one load. The voxel arm first, since
    // it is the one the bodies must not have swallowed.
    let program = rig.register(pixel_epilogue(CLIP, voxels()), 1);
    let cell = rig.channel(vec![CLIP[1], CLIP[2], C_IN], HostRole::Writer);
    let back = rig.channel(vec![voxels(), C_OUT], HostRole::Reader);
    let vae = rig.bind(program, vec![cell, back], voxels());

    let mut rng = Lcg::seeded(9);
    let clip: Vec<f32> = (0..voxels() as usize * C_IN as usize)
        .map(|_| bf(rng.unit()))
        .collect();
    rig.publish(vae, 0, &clip);
    let mut ticket = rig
        .engine
        .submit(&frame(
            vec![vae_lane(0, cell)],
            vec![attach(0, vae)],
            vec![StepVoxels {
                lane: 0,
                clips: vec![CLIP],
                payload: Vec::new(),
            }],
        ))
        .expect("the voxel arm fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the voxel fire settles");
    assert_eq!(ticket.steps[0].readouts[0].seam, ReadoutSeam::Pixels);
    assert_close(
        &rig.take(vae, 1),
        &conv_reference(&weights, CLIP, &clip),
        "the voxel arm's pixels",
    );

    // The token arm, through the same load: its answer is whatever the DiT
    // block computes, and what is asked here is that it computes SOMETHING
    // finite over every row — a body replayed at the wrong geometry lands
    // zeros or garbage, and the load's own golden check (which runs at every
    // armed key) is what proves the numbers.
    let velocity = rig.register(common_two_axis::velocity_epilogue(ROWS), 2);
    let latent = rig.channel(vec![ROWS, WIDTH], HostRole::Writer);
    let timestep = rig.channel(vec![1, 1], HostRole::Writer);
    let positions = rig.channel(vec![ROWS, 2], HostRole::Writer);
    let out = rig.channel(vec![ROWS, WIDTH], HostRole::Reader);
    let dit = rig.bind(velocity, vec![latent, timestep, positions, out], ROWS);
    rig.publish(
        dit,
        0,
        &(0..ROWS as usize * WIDTH as usize)
            .map(|_| bf(rng.unit()))
            .collect::<Vec<f32>>(),
    );
    rig.publish(dit, 1, &[0.7]);
    rig.publish(
        dit,
        2,
        &(0..ROWS)
            .flat_map(|r| [r as f32, 0.5 * r as f32])
            .collect::<Vec<f32>>(),
    );
    let mut ticket = rig
        .engine
        .submit(&frame(
            vec![dit_lane(0, ROWS, latent, timestep, positions)],
            vec![attach(0, dit)],
            Vec::new(),
        ))
        .expect("the token arm fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the token fire settles");
    let answered = rig.take(dit, 3);
    assert_eq!(answered.len(), (ROWS * WIDTH) as usize);
    assert!(
        answered.iter().all(|v| v.is_finite()) && answered.iter().any(|v| v.abs() > 1e-6),
        "the token arm answered nothing"
    );

    // The bodies were actually used: a hit is a replay, and the token fire is
    // the only thing here that could have produced one.
    let after = rig
        .engine
        .shell()
        .expect("the load holds a shell")
        .body_stats();
    println!("[two-axis, after] {after}");
    assert!(
        after.tally.hits > stats.tally.hits,
        "no fire replayed a body: {after}"
    );
}
