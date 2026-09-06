//! **THE CONVOLUTION'S RATE, PRINTED**: TFLOP/s of both kernels on a
//! video-VAE-decoder-like shape (`C_in = C_out = 384`, `3x3x3`, one clip of
//! `16 x 90 x 160` voxels, causal time) and an image-VAE-like shape
//! (`C = 512`, `3x3`, one `256 x 256` image). Ignored by default: it is a
//! measurement, not a claim.
//!
//! `CUDA_VISIBLE_DEVICES=<n> cargo test -p kernels-cuda --features cuda --release --test the_conv3d_is_priced_in_tflops -- --ignored --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::time::Instant;

use common::spatial::{Box3, out_boxes, table};
use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::spatial::{Conv3d, ConvPath, TimePad, conv3d_on};
use kernels_cuda::tensor::Tensor;

fn price(name: &str, boxes: &[Box3], c: usize, conv: Conv3d) {
    let taps = conv.taps() as usize;
    let (grid, rows) = table(boxes);
    let outs = out_boxes(&conv, boxes);
    let (o_grid, rows_out) = table(&outs);
    let mut lcg = Lcg::seeded(0x7f0);
    let (x_raw, _) = lcg.row(rows * c);
    let (w_raw, _) = lcg.row(c * c * taps);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let w_at = gpu.up(&w_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let o_at = gpu.zeros(rows_out * c * 2);
    let ctx = gpu.ctx();
    let x = Tensor::new(x_at, rows as u32, c as u32, Dtype::Bf16);
    let w = Tensor::new(w_at, c as u32, (c * taps) as u32, Dtype::Bf16);
    let g = Tensor::new(grid_at, boxes.len() as u32, 4, Dtype::I32);
    let og = Tensor::new(o_grid_at, boxes.len() as u32, 4, Dtype::I32);
    let mut o = Tensor::new(o_at, rows_out as u32, c as u32, Dtype::Bf16);

    let flops = 2.0 * rows_out as f64 * c as f64 * (taps * c) as f64;
    for path in [ConvPath::Direct, ConvPath::TensorCore] {
        let fire = |o: &mut Tensor| {
            conv3d_on(&ctx, path, x, g, w, None, conv, None, o, og).expect("the convolution fires");
        };
        fire(&mut o);
        gpu.sync();
        let reps = 5;
        let start = Instant::now();
        for _ in 0..reps {
            fire(&mut o);
        }
        gpu.sync();
        let seconds = start.elapsed().as_secs_f64() / f64::from(reps);
        eprintln!(
            "{name}: {path:?} {:.2} ms, {:.1} TFLOP/s ({:.3} TFLOP per fire)",
            seconds * 1e3,
            flops / seconds / 1e12,
            flops / 1e12
        );
    }
}

#[test]
#[ignore = "a measurement: run with --ignored --nocapture --release"]
fn the_video_decoder_shape() {
    price(
        "video 384ch 3x3x3 causal over 16x90x160",
        &[Box3::new(16, 90, 160)],
        384,
        Conv3d {
            k: [3, 3, 3],
            stride: [1, 1, 1],
            pad: [2, 1, 1],
            causal_t: true,
            time_pad: TimePad::Replicate,
        },
    );
}

#[test]
#[ignore = "a measurement: run with --ignored --nocapture --release"]
fn the_image_decoder_shape() {
    price(
        "image 512ch 3x3 over 256x256",
        &[Box3::new(1, 256, 256)],
        512,
        Conv3d::conv2d([3, 3], [1, 1], [1, 1]),
    );
}
