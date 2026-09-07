//! **THE VOXEL-AXIS RESHAPES MOVE EVERY ELEMENT WHERE THE REFERENCE PUTS
//! IT**: `pixel_shuffle` lands the `torch.pixel_shuffle` ordering on a
//! hand-computed example, `pixel_unshuffle` inverts it, `patchify` and
//! `unpatchify` round-trip, and `upsample_nearest` matches the host
//! reference with and without the causal first-frame rule, and `avg_down`
//! answers `AvgDown3D` — on two lanes of different boxes.
//!
//! `CUDA_VISIBLE_DEVICES=<n> cargo test -p kernels-cuda --features cuda --test the_voxel_reshapes_land_the_reference_ordering`

#![cfg(feature = "cuda")]

mod common;

use common::spatial::{
    Box3, avg_down_ref, pixel_shuffle_ref, pixel_unshuffle_ref, table, upsample_ref,
};
use common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::spatial::{
    avg_down, patchify, pixel_shuffle, pixel_unshuffle, unpatchify, upsample_nearest,
};
use kernels_cuda::tensor::Tensor;

/// `torch.pixel_shuffle(x, 2)` on `x = arange(8).reshape(1, 4, 1, 2)`:
///
/// ```text
/// x[c][0][w] = 2c + w          out[0][ho][wo] = x[(ho % 2) * 2 + wo % 2][0][wo / 2]
/// out[0][0] = [x0[0], x1[0], x0[1], x1[1]] = [0, 2, 1, 3]
/// out[0][1] = [x2[0], x3[0], x2[1], x3[1]] = [4, 6, 5, 7]
/// ```
///
/// In the voxel-row layout the input is two rows (`w = 0, 1`) of four
/// channels, `[[0, 2, 4, 6], [1, 3, 5, 7]]`, and the output eight rows of
/// one channel in `(h, w)` order.
#[test]
fn pixel_shuffle_lands_torchs_ordering_on_the_hand_computed_example() {
    let x: Vec<u16> = [0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]
        .map(to_bf16)
        .to_vec();
    let (grid, rows) = table(&[Box3::new(1, 1, 2)]);
    let (o_grid, rows_out) = table(&[Box3::new(1, 2, 4)]);
    assert_eq!((rows, rows_out), (2, 8));
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let o_at = gpu.zeros(rows_out * 2);
    let ctx = gpu.ctx();
    let mut o = Tensor::new(o_at, rows_out as u32, 1, Dtype::Bf16);
    pixel_shuffle(
        &ctx,
        Tensor::new(x_at, rows as u32, 4, Dtype::Bf16),
        Tensor::new(grid_at, 1, 4, Dtype::I32),
        [1, 2, 2],
        0,
        &mut o,
        Tensor::new(o_grid_at, 1, 4, Dtype::I32),
    )
    .expect("the shuffle fires");
    gpu.sync();
    let got: Vec<f32> = gpu
        .down::<u16>(o_at, rows_out)
        .into_iter()
        .map(from_bf16)
        .collect();
    assert_eq!(got, [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
}

const BOXES: [Box3; 2] = [Box3::new(2, 4, 6), Box3::new(4, 2, 4)];

const R: [u32; 3] = [2, 2, 2];

#[test]
fn pixel_unshuffle_inverts_pixel_shuffle_and_both_match_the_reference() {
    let c = 3usize;
    let vol = 8usize;
    let (grid, rows) = table(&BOXES);
    let mut lcg = Lcg::seeded(0x5a);
    let (x_raw, x) = lcg.row(rows * c * vol);
    let (want_up, up_boxes) = pixel_shuffle_ref(&x, &BOXES, c, [2, 2, 2]);
    let (o_grid, rows_up) = table(&up_boxes);
    let (want_back, back_boxes) = pixel_unshuffle_ref(&want_up, &up_boxes, c, [2, 2, 2]);
    assert_eq!(back_boxes, BOXES.to_vec());
    assert_eq!(want_back, x, "the host references invert each other");

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let up_at = gpu.zeros(rows_up * c * 2);
    let back_at = gpu.zeros(rows * c * vol * 2);
    let ctx = gpu.ctx();
    let x_t = Tensor::new(x_at, rows as u32, (c * vol) as u32, Dtype::Bf16);
    let g = Tensor::new(grid_at, 2, 4, Dtype::I32);
    let og = Tensor::new(o_grid_at, 2, 4, Dtype::I32);
    let mut up = Tensor::new(up_at, rows_up as u32, c as u32, Dtype::Bf16);
    pixel_shuffle(&ctx, x_t, g, R, 0, &mut up, og).expect("the shuffle fires");
    let mut back = Tensor::new(back_at, rows as u32, (c * vol) as u32, Dtype::Bf16);
    pixel_unshuffle(&ctx, up, og, R, &mut back, g).expect("the unshuffle fires");
    gpu.sync();
    let got_up: Vec<u16> = gpu.down(up_at, rows_up * c);
    let got_back: Vec<u16> = gpu.down(back_at, rows * c * vol);
    let want_up: Vec<u16> = want_up.iter().map(|&v| to_bf16(v)).collect();
    assert_eq!(got_up, want_up, "the shuffle's ordering");
    assert_eq!(got_back, x_raw, "the round trip");
}

#[test]
fn patchify_and_unpatchify_round_trip_over_the_token_box() {
    let c = 4usize;
    let p = [1u32, 2, 2];
    let (grid, rows) = table(&BOXES);
    let mut lcg = Lcg::seeded(0x9a7);
    let (x_raw, x) = lcg.row(rows * c);
    let (want_tokens, token_boxes) = pixel_unshuffle_ref(&x, &BOXES, c, [1, 2, 2]);
    let (t_grid, tokens) = table(&token_boxes);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let t_grid_at = gpu.up(&t_grid);
    let tok_at = gpu.zeros(tokens * c * 4 * 2);
    let back_at = gpu.zeros(rows * c * 2);
    let ctx = gpu.ctx();
    let g = Tensor::new(grid_at, 2, 4, Dtype::I32);
    let tg = Tensor::new(t_grid_at, 2, 4, Dtype::I32);
    let mut tok = Tensor::new(tok_at, tokens as u32, (c * 4) as u32, Dtype::Bf16);
    patchify(
        &ctx,
        Tensor::new(x_at, rows as u32, c as u32, Dtype::Bf16),
        g,
        p,
        &mut tok,
        tg,
    )
    .expect("patchify fires");
    let mut back = Tensor::new(back_at, rows as u32, c as u32, Dtype::Bf16);
    unpatchify(&ctx, tok, tg, p, &mut back, g).expect("unpatchify fires");
    gpu.sync();
    let got_tok: Vec<u16> = gpu.down(tok_at, tokens * c * 4);
    let got_back: Vec<u16> = gpu.down(back_at, rows * c);
    let want_tok: Vec<u16> = want_tokens.iter().map(|&v| to_bf16(v)).collect();
    assert_eq!(got_tok, want_tok, "`(c pt ph pw)` token channels");
    assert_eq!(got_back, x_raw, "the round trip");
}

fn upsample(keep_first: bool) {
    let c = 8usize;
    let f = [2usize, 2, 3];
    let (grid, rows) = table(&BOXES);
    let mut lcg = Lcg::seeded(0x0b);
    let (x_raw, x) = lcg.row(rows * c);
    let (want, out_boxes) = upsample_ref(&x, &BOXES, c, f, keep_first);
    let (o_grid, rows_out) = table(&out_boxes);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let o_at = gpu.zeros(rows_out * c * 2);
    let ctx = gpu.ctx();
    let mut o = Tensor::new(o_at, rows_out as u32, c as u32, Dtype::Bf16);
    upsample_nearest(
        &ctx,
        Tensor::new(x_at, rows as u32, c as u32, Dtype::Bf16),
        Tensor::new(grid_at, 2, 4, Dtype::I32),
        [2, 2, 3],
        keep_first,
        &mut o,
        Tensor::new(o_grid_at, 2, 4, Dtype::I32),
    )
    .expect("the upsample fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, rows_out * c);
    let want: Vec<u16> = want.iter().map(|&v| to_bf16(v)).collect();
    assert_eq!(got, want, "keep_first {keep_first}");
}

#[test]
fn upsample_nearest_repeats_every_frame() {
    upsample(false);
}

#[test]
fn upsample_nearest_keeps_the_first_frame_single() {
    upsample(true);
}

/// `AvgDown3D` against the module it is: the same channel-major block
/// `pixel_unshuffle` lays out, averaged in runs of `group`, with the TIME
/// axis zero-padded IN FRONT where the box does not fill the block.
///
/// Three shapes, all of them Wan 2.2's: `(1, 2, 2)` with `group = 4` (the
/// first down block's shortcut — a plain 2x2 spatial pool), `(2, 2, 2)`
/// with `group = 4` (the temporal ones — a spatial pool that keeps the
/// time block as two channels per input channel), and `(1, 1, 1)` with
/// `group = 1`, which the reference builds for the last down block and
/// which must be the identity.
fn avg_down_case(boxes: &[Box3], c: usize, r: [u32; 3], group: u32) {
    let (grid, rows) = table(boxes);
    let ru = r.map(|v| v as usize);
    let mut lcg = Lcg::seeded(0xa7 ^ u64::from(r[0] * 31 + r[1] * 7 + r[2]) ^ (group as u64) << 8);
    let (x_raw, x) = lcg.row(rows * c);
    let (want, out_boxes) = avg_down_ref(&x, boxes, c, ru, group as usize);
    let (o_grid, rows_out) = table(&out_boxes);
    let c_out = c * ru[0] * ru[1] * ru[2] / group as usize;

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let o_at = gpu.zeros(rows_out * c_out * 2);
    let ctx = gpu.ctx();
    let lanes = boxes.len() as u32;
    let mut o = Tensor::new(o_at, rows_out as u32, c_out as u32, Dtype::Bf16);
    avg_down(
        &ctx,
        Tensor::new(x_at, rows as u32, c as u32, Dtype::Bf16),
        Tensor::new(grid_at, lanes, 4, Dtype::I32),
        r,
        group,
        &mut o,
        Tensor::new(o_grid_at, lanes, 4, Dtype::I32),
    )
    .unwrap_or_else(|why| panic!("{r:?} group {group}: {why}"));
    gpu.sync();
    let got: Vec<f32> = gpu
        .down::<u16>(o_at, rows_out * c_out)
        .into_iter()
        .map(from_bf16)
        .collect();
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        worst = worst.max((g - w).abs());
        assert!(
            (g - w).abs() <= 4e-3 + 8e-3 * w.abs(),
            "{r:?} group {group}: row {} channel {}: got {g}, want {w}",
            i / c_out,
            i % c_out
        );
    }
    eprintln!("avg_down {r:?} group {group}: {rows} -> {rows_out} rows, max |err| {worst:.5}");
}

#[test]
fn avg_down_answers_the_avgdown3d_reference() {
    // Boxes that DIVIDE the time block and one that does not: `t = 1` under
    // `factor_t = 2` is the encoder head chunk, where the front pad is a
    // whole frame of zeros and halves every channel it reaches.
    const EVEN: [Box3; 2] = [Box3::new(4, 4, 6), Box3::new(2, 2, 4)];
    const ODD: [Box3; 2] = [Box3::new(1, 4, 6), Box3::new(3, 2, 4)];
    avg_down_case(&EVEN, 5, [1, 2, 2], 4);
    avg_down_case(&EVEN, 5, [2, 2, 2], 4);
    avg_down_case(&ODD, 5, [2, 2, 2], 4);
    avg_down_case(&ODD, 3, [2, 2, 2], 8);
    avg_down_case(&ODD, 3, [1, 1, 1], 1);
}

/// The front pad, stated on its own: one frame under a factor-2 time block
/// leaves the EVEN widened channels reading a zero frame, so a `group` of
/// `fh·fw` (which never crosses the time block) lands exactly half of the
/// spatial pool on those channels and the pool itself on the odd ones.
#[test]
fn avg_down_pads_a_short_chunks_time_axis_in_front() {
    let boxes = [Box3::new(1, 2, 2)];
    let (grid, rows) = table(&boxes);
    // One channel, four voxels, values 1..4: the 2x2 spatial pool is 2.5.
    let x: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0].map(to_bf16).to_vec();
    let (o_grid, rows_out) = table(&[Box3::new(1, 1, 1)]);
    assert_eq!((rows, rows_out), (4, 1));
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let o_at = gpu.zeros(2 * 2);
    let ctx = gpu.ctx();
    let mut o = Tensor::new(o_at, 1, 2, Dtype::Bf16);
    avg_down(
        &ctx,
        Tensor::new(x_at, 4, 1, Dtype::Bf16),
        Tensor::new(grid_at, 1, 4, Dtype::I32),
        [2, 2, 2],
        4,
        &mut o,
        Tensor::new(o_grid_at, 1, 4, Dtype::I32),
    )
    .expect("the avg-down fires");
    gpu.sync();
    let got: Vec<f32> = gpu
        .down::<u16>(o_at, 2)
        .into_iter()
        .map(from_bf16)
        .collect();
    // Channel 0 is the padded (zero) frame's pool, channel 1 the real one.
    assert_eq!(got, [0.0, 2.5]);
}
