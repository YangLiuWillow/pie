//! The wide expert select (`moe_matmul_select_quant` from 256 routes on:
//! the grouped tensor-core kernel) priced at gemma-4-26B-A4B's own shapes
//! — 128 experts, top-8 over a 256-row canvas (2048 routes, routed with a
//! realistic skew), the `[1408, 2816]` up leg read by token and the
//! `[2816, 704]` down leg read by route — against the bytes it must read.
//! A denoise step fires each leg thirty times, so a millisecond here is
//! thirty on the step. Prints the price; asserts only that it answers.
//!
//!   cargo test --release -p kernels-cuda --features cuda \
//!     --test the_expert_select_is_priced_by_its_bytes -- --nocapture

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::linear::moe::{GroupSeat, matmul_select_quant};
use kernels_cuda::tensor::Tensor;
use std::time::Instant;

const GROUP: usize = 64;
const EXPERTS: usize = 128;
/// One canvas by default; `PIE_BENCH_TOKENS` prices a batch of canvases.
fn tokens() -> usize {
    std::env::var("PIE_BENCH_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(256)
}
const TOP_K: usize = 8;

/// Routes with a skew like a trained router's: token `t`'s eight experts
/// are drawn from a distribution where a few experts are popular.
fn routes(lcg: &mut Lcg) -> Vec<i32> {
    let mut out = Vec::with_capacity(tokens() * TOP_K);
    for _ in 0..tokens() {
        let mut picked: Vec<i32> = Vec::with_capacity(TOP_K);
        while picked.len() < TOP_K {
            // Square of a uniform: the low experts are picked far more often.
            let u = lcg.unit().abs();
            let e = if std::env::var_os("PIE_BENCH_UNIFORM").is_some() {
                (u * EXPERTS as f32) as i32 % EXPERTS as i32
            } else {
                ((u * u) * EXPERTS as f32) as i32 % EXPERTS as i32
            };
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        out.extend(picked);
    }
    out
}

fn price(bits: usize, n: usize, k: usize, by_token: bool, label: &str) {
    let mut lcg = Lcg::seeded(0x77 + bits as u64 + n as u64);
    let groups = k / GROUP;
    let code_bytes = EXPERTS * n * k * bits / 8;
    let codes: Vec<u8> = (0..code_bytes).map(|i| ((i * 2654435761) >> 7) as u8).collect();
    let scales: Vec<u16> = (0..EXPERTS * n * groups).map(|_| common::to_bf16(lcg.unit() * 0.01)).collect();
    let biases: Vec<u16> = (0..EXPERTS * n * groups).map(|_| common::to_bf16(lcg.unit() * 0.1)).collect();
    let routes = routes(&mut lcg);
    let act_rows = if by_token { tokens() } else { tokens() * TOP_K };
    let (x_raw, _) = lcg.row(act_rows * k);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let codes_at = gpu.up(&codes);
    let scales_at = gpu.up(&scales);
    let biases_at = gpu.up(&biases);
    let routes_at = gpu.up(&routes);
    let y_at = gpu.zeros(tokens() * TOP_K * n * 2);
    let ctx = gpu.ctx();
    let x_t = Tensor::new(x_at, act_rows as u32, k as u32, Dtype::Bf16);
    let codes_t = Tensor::new(codes_at, EXPERTS as u32, (n * k * bits / 8) as u32, Dtype::U8);
    let scales_t = Tensor::new(scales_at, EXPERTS as u32, (n * groups * 2) as u32, Dtype::U8);
    let biases_t = Tensor::new(biases_at, EXPERTS as u32, (n * groups * 2) as u32, Dtype::U8);
    let routes_t = Tensor::new(routes_at, tokens() as u32, TOP_K as u32, Dtype::I32);
    let mut y_t = Tensor::new(y_at, (tokens() * TOP_K) as u32, n as u32, Dtype::Bf16);
    let fire = |y_t: &mut Tensor| {
        matmul_select_quant(&ctx, x_t, codes_t, scales_t, Some(biases_t), routes_t, y_t, GroupSeat::RESIDENT)
            .expect("the select fires")
    };
    // Ramp, then a timed run of many launches; the sync sits outside.
    for _ in 0..5 {
        fire(&mut y_t);
    }
    gpu.sync();
    const REPS: usize = 40;
    let t0 = Instant::now();
    for _ in 0..REPS {
        fire(&mut y_t);
    }
    gpu.sync();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
    let bytes = code_bytes + 2 * 2 * EXPERTS * n * groups;
    let gbs = bytes as f64 / ms / 1e6;
    let flop = 2.0 * (tokens() * TOP_K) as f64 * n as f64 * k as f64;
    eprintln!(
        "{label:9} u{bits} n={n:4} k={k:4}: {ms:6.3} ms  {:6.1} MB  {gbs:6.0} GB/s  {:5.1} TFLOP/s",
        bytes as f64 / 1e6,
        flop / ms / 1e9
    );
}

#[test]
fn the_up_and_down_legs_at_four_and_eight_bits() {
    for bits in [4usize, 8] {
        price(bits, 1408, 2816, true, "up leg");
        price(bits, 2816, 704, false, "down leg");
    }
}
