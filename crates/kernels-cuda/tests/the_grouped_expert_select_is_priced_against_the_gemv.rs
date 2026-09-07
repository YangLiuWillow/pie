//! **WHAT GROUPING THE ROUTED MATMUL IS WORTH**, at both catalog bf16 MoE
//! shapes over a 256-row canvas: gemma-4-26B-A4B (128 experts, top-8, the
//! `[1408, 2816]` up leg by token and `[2816, 704]` down by route) and
//! qwen a3b (256 experts, top-8, `[1024, 2048]` up and `[2048, 512]` down).
//!
//! The two are not the same bet. Grouping pads every expert's rows up to a
//! whole block, so twice the experts over the same canvas halves the routes
//! each one gets and pays more padding for them: qwen is where the win is
//! thinnest, and it is the one worth watching.
//!
//! The per-route GEMV reads a whole expert bank back for every route that
//! names it, so its weight traffic is `routes * N * K` where the work needs
//! `experts * N * K`. Grouping sorts the routes into per-expert blocks and
//! reads each bank once per block. At this shape that is 2048 reads against
//! 256, and the print below is where that lands.
//!
//! Prints the price; asserts only that both legs answer. Run it twice to
//! read the comparison — the second run puts the GEMV back:
//!
//!   cargo test --release -p kernels-cuda --features cuda \
//!     --test the_grouped_expert_select_is_priced_against_the_gemv \
//!     -- --nocapture --test-threads=1
//!   PIE_NO_MOE_GROUP=1 cargo test --release ... -- --nocapture --test-threads=1
//!
//! `--test-threads=1` is not optional for the numbers: the two shapes are
//! two `#[test]`s, cargo runs them on threads, and two fires contending for
//! one device price each other at nearly twice their own cost.

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::linear::moe::{ExpertTable, matmul_select};
use kernels_cuda::tensor::Tensor;
use std::time::Instant;

/// One canvas by default; `PIE_BENCH_TOKENS` prices a batch of canvases.
fn tokens() -> usize {
    std::env::var("PIE_BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256)
}

/// Routes with a trained router's skew: a few experts are picked far more
/// often than the rest. An even spread would flatter grouping, which pads
/// every expert's rows up to a whole block.
fn routes(lcg: &mut Lcg, experts: usize, top_k: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(tokens() * top_k);
    for _ in 0..tokens() {
        let mut picked: Vec<i32> = Vec::with_capacity(top_k);
        while picked.len() < top_k {
            let u = lcg.unit().abs();
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let e = ((u * u) * experts as f32) as i32 % experts as i32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        out.extend(picked);
    }
    out
}

fn price(experts: usize, top_k: usize, n: usize, k: usize, by_token: bool, label: &str) {
    let mut lcg = Lcg::seeded(0x9c + n as u64);
    let route_list = routes(&mut lcg, experts, top_k);
    let act_rows = if by_token { tokens() } else { tokens() * top_k };
    let (x_raw, _) = lcg.row(act_rows * k);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    // The bank is left zeroed rather than filled: it is a gigabyte at this
    // shape, and neither leg's price depends on what it reads — both walk
    // every byte they walk regardless, and bf16 zero carries no denormal.
    let bank_at = gpu.zeros(experts * n * k * 2);
    let routes_at = gpu.up(&route_list);
    let y_at = gpu.zeros(tokens() * top_k * n * 2);

    let ctx = gpu.ctx();
    let x_t = Tensor::new(x_at, act_rows as u32, k as u32, Dtype::Bf16);
    // One row per expert, the row its whole plane — see the correctness test.
    let bank_t = Tensor::new(bank_at, experts as u32, (n * k) as u32, Dtype::Bf16);
    let routes_t = Tensor::new(routes_at, tokens() as u32, top_k as u32, Dtype::I32);
    let mut y_t = Tensor::new(y_at, (tokens() * top_k) as u32, n as u32, Dtype::Bf16);

    let fire = |y_t: &mut Tensor| {
        matmul_select(&ctx, x_t, bank_t, routes_t, y_t, ExpertTable::RESIDENT)
            .expect("the routed select fires");
    };
    // Ramp, then a timed run of many launches; the sync sits outside.
    for _ in 0..5 {
        fire(&mut y_t);
    }
    gpu.sync();
    const REPS: usize = 20;
    let t0 = Instant::now();
    for _ in 0..REPS {
        fire(&mut y_t);
    }
    gpu.sync();

    let ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
    let plane = (n * k * 2) as f64;
    let per_route = route_list.len() as f64 * plane;
    let leg = if std::env::var_os("PIE_NO_MOE_GROUP").is_some() {
        "gemv"
    } else {
        "grouped"
    };
    let flop = 2.0 * (tokens() * top_k) as f64 * n as f64 * k as f64;
    eprintln!(
        "{leg:8} {label:14} e={experts:3} n={n:4} k={k:4}: {ms:7.3} ms  {:5.1} TFLOP/s  \
         (a per-route read would be {:6.1} GB, one per expert {:5.1} GB)",
        flop / ms / 1e9,
        per_route / 1e9,
        experts as f64 * plane / 1e9,
    );
}

#[test]
fn the_up_and_down_legs_at_gemmas_shapes() {
    price(128, 8, 1408, 2816, true, "gemma up");
    price(128, 8, 2816, 704, false, "gemma down");
}

/// Twice the experts over the same canvas: eight routes each rather than
/// sixteen, and a block's padding spread over twice as many banks. If
/// grouping ever stops paying, it stops here first.
#[test]
fn the_up_and_down_legs_at_qwen_a3bs_shapes() {
    price(256, 8, 1024, 2048, true, "qwen a3b up");
    price(256, 8, 2048, 512, false, "qwen a3b down");
}
