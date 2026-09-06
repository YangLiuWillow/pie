//! **A `PortKind::Voxels` PORT IS FED FROM ITS CHANNEL'S COMMITTED CELL, AND
//! THE GUEST READS THE PIXELS BACK THROUGH THE `pixels()` INTRINSIC.**
//! (design D8, `IMAGEGEN_CONTRACT.md` §6)
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=<n> cargo test -p engine-cuda --features cuda \
//!   --test a_channel_fed_voxel_port_lands_the_committed_cell
//! ```
//!
//! The voxel port had one feed: a payload the caller handed the shell beside
//! its clips, which crosses the host bus and which no guest can reach. This
//! is the other one — the road a guest actually has. The lane names its port
//! in `Lane::ports`, the clip's box travels beside it in `StepVoxels::clips`
//! (a channel cell carries no grid, so the shape of the channel IS the box),
//! and the shell copies the committed cell into the voxel payload device to
//! device, casting the f32 ring master into the plan's bf16 port on the way.
//!
//! The answer is checked twice over: through the `pixels` seam's readout, and
//! through the `pixels()` intrinsic the attached epilogue reads — which is
//! the one that matters, since a guest never sees a `LaneReadout`.
//!
//! Fired twice with two different cells, so a feed that read a stale
//! rectangle would land the first answer again and fail.
//!
//! Skipped at run time with no device, as the other device gates are.

#![cfg(feature = "cuda")]

mod common_two_axis;

use common_two_axis::{
    C_IN, C_OUT, Lcg, Rig, Weights, assert_close, attach, bf, conv_reference, frame,
    pixel_epilogue, trace, vae_lane,
};
use engine::Engine;
use engine::fire::{ReadoutSeam, StepVoxels};

/// One still, `t = 1`.
const CLIP: [u32; 3] = [1, 5, 7];

const fn voxels() -> u32 {
    CLIP[0] * CLIP[1] * CLIP[2]
}

#[test]
fn the_cell_the_channel_holds_is_the_clip_the_convolution_reads() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&trace(), 0x5a);
    let mut rig = Rig::load(&weights, 32, vec![16, 32], voxels() + 8);
    assert!(
        rig.profile().has_pixels,
        "a plan planting `seam::PIXELS` states the `pixels()` gate"
    );

    // Channel 0 is the port cell — its shape IS the clip's box (D8) — and
    // channel 1 is where the epilogue hands the pixels back.
    let program = rig.register(pixel_epilogue(CLIP, voxels()), 1);
    let cell = rig.channel(
        vec![CLIP[1], CLIP[2], C_IN],
        eta_ir::container::HostRole::Writer,
    );
    let back = rig.channel(vec![voxels(), C_OUT], eta_ir::container::HostRole::Reader);
    let instance = rig.bind(program, vec![cell, back], voxels());

    let mut rng = Lcg::seeded(3);
    let mut draw = || -> Vec<f32> {
        (0..voxels() as usize * C_IN as usize)
            .map(|_| bf(rng.unit()))
            .collect()
    };
    let first = draw();
    let second = draw();

    let fire = |rig: &mut Rig, clip: &[f32]| -> (Vec<f32>, Vec<[u32; 3]>, Vec<f32>) {
        rig.publish(instance, 0, clip);
        let mut ticket = rig
            .engine
            .submit(&frame(
                vec![vae_lane(0, cell)],
                vec![attach(0, instance)],
                // No payload: the port is channel-fed, so what travels is
                // the geometry a channel cell cannot carry.
                vec![StepVoxels {
                    lane: 0,
                    clips: vec![CLIP],
                    payload: Vec::new(),
                }],
            ))
            .expect("the frame fires");
        rig.engine
            .settle_frame(&mut ticket)
            .expect("the frame settles");
        let readout = ticket.steps[0].readouts[0].clone();
        assert_eq!(
            readout.seam,
            ReadoutSeam::Pixels,
            "a VAE lane answers pixels"
        );
        (readout.values, readout.clips, rig.take(instance, 1))
    };

    let (seam_first, boxes, guest_first) = fire(&mut rig, &first);
    assert_eq!(boxes, vec![CLIP], "a `same3` convolution keeps the box");
    let want_first = conv_reference(&weights, CLIP, &first);
    assert_close(&seam_first, &want_first, "the first fire's pixels seam");
    assert_close(&guest_first, &want_first, "the first fire's `pixels()`");

    let (seam_second, _, guest_second) = fire(&mut rig, &second);
    let want_second = conv_reference(&weights, CLIP, &second);
    assert_close(&seam_second, &want_second, "the second fire's pixels seam");
    assert_close(&guest_second, &want_second, "the second fire's `pixels()`");

    // And the two really differ: a feed that re-read the first cell would
    // have landed the first answer twice and passed every assertion above.
    let moved = guest_first
        .iter()
        .zip(&guest_second)
        .any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(moved, "the second fire re-read the first fire's cell");
}
