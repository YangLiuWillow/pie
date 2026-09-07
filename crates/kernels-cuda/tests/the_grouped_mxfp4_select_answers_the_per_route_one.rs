//! **THE GROUPED mxfp4 SELECT AGAINST THE ONE IT REPLACES.**
//!
//! `matmul_select_quant` split on the presence of zero points: an affine
//! bank got `moe_route_order` and a grouped kernel — one block per expert,
//! the plane decoded once per K chunk and applied to sixteen routes — while
//! an mxfp4 bank got a block per route. On an L40S at gpt-oss's shapes that
//! was 4.8 TFLOP/s against the affine path's 34.
//!
//! The grouped mxfp4 kernel is the affine one with the decode swapped, so
//! what has to be checked is that the swap is faithful. The per-route select
//! is the reference: it ships, it is what every mxfp4 fire has been
//! answering with, and `PIE_NO_MXFP4_GROUP` still reaches it. A host
//! reference would mean transcribing e2m1 by hand and checking my
//! transcription instead of the kernel.
//!
//! The two do not agree bit for bit and must not be asked to: the per-route
//! select accumulates a 32-code block and scales the sum, the grouped one
//! scales each weight and accumulates. Same arithmetic, different rounding.
//!
//!   cargo test --release -p kernels-cuda --features cuda \
//!     --test the_grouped_mxfp4_select_answers_the_per_route_one

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{GroupSeat, matmul_select_bias, matmul_select_quant};
use kernels_cuda::tensor::Tensor;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

/// **ONE ARM AT A TIME IN THIS PROCESS.** Every test here fires both legs by
/// toggling `PIE_NO_MXFP4_GROUP`, which is process-wide: two of them on
/// cargo's threads would read each other's setting and compare a leg
/// against itself. The prices want the serialisation anyway — two fires
/// contending for one device price each other.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A variable set for the length of a scope, **restored even if the scope
/// unwinds**. Restoring it on the happy path only is how a failing arm
/// leaves its setting behind for every test after it: an assertion inside
/// `check_fp32_form` used to skip the removal, and the tensor-core arms
/// that ran next took the fp32 kernel and failed for a reason that was not
/// theirs. That is a mutation test reporting six failures for one defect.
struct Env(&'static str);

impl Env {
    fn set(name: &'static str) -> Env {
        // SAFETY: every caller holds `serialized`, so no other arm reads it.
        unsafe { std::env::set_var(name, "1") };
        Env(name)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        // SAFETY: as `set`.
        unsafe { std::env::remove_var(self.0) };
    }
}

const EXPERTS: usize = 8;
const TOP_K: usize = 4;

/// mxfp4's block: 32 codes to one e8m0 scale byte.
const BLOCK: usize = 32;

/// A deterministic byte stream — the two arms must read the same bank.
fn bytes(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as u8
        })
        .collect()
}

/// Block scales as e8m0 exponents: `mxfp4_block_scale` reads the byte as a
/// float's exponent field, so 127 is 1.0. Drawn near it, because a bank
/// spanning 2^±127 would make every comparison a comparison of infinities.
fn scale_bytes(seed: u64, count: usize) -> Vec<u8> {
    bytes(seed, count)
        .into_iter()
        .map(|b| 124 + (b % 7))
        .collect()
}

fn routes_of(tokens: usize) -> Vec<i32> {
    (0..tokens)
        .flat_map(|t| {
            (0..TOP_K).map(move |j| {
                i32::try_from((t * 3 + j * 5 + t / 5) % EXPERTS).expect("an expert id in i32")
            })
        })
        .collect()
}

/// One shape, both arms, compared.
fn check(tokens: usize, n: usize, k: usize, by_token: bool, biased: bool) {
    let _one = serialized();
    check_held(tokens, n, k, by_token, biased);
}

/// [`check`]'s body, with the serialisation already taken. Split out
/// because `check_fp32_form` has to set its variable INSIDE the lock: it
/// used to set it and then block on the mutex, so whichever test held the
/// lock ran the fp32 kernel while claiming to test the tensor-core one —
/// which a mutation of the fp32 decode found by failing tests that should
/// not have noticed it.
fn check_held(tokens: usize, n: usize, k: usize, by_token: bool, biased: bool) {
    assert_eq!(k % BLOCK, 0, "K is whole mxfp4 blocks");
    let groups = k / BLOCK;
    let routes = routes_of(tokens);
    let route_count = routes.len();
    let act_rows = if by_token { tokens } else { route_count };

    let codes = bytes(0x11 + n as u64, EXPERTS * n * k / 2);
    let scales = scale_bytes(0x22 + k as u64, EXPERTS * n * groups);
    let act: Vec<u16> = bytes(0x33, act_rows * k * 2)
        .chunks(2)
        .map(|p| to_bf16((f32::from(p[0]) / 128.0 - 1.0) * 0.5))
        .collect();
    // **BIG ENOUGH TO SEE.** The dots here land around ±280 and `close` is a
    // 3 % relative check, so a bias of ±0.25 hides inside the tolerance: an
    // epilogue that dropped it entirely still passed. Scaled to the same
    // order as the result, dropping it fails.
    let bias: Vec<u16> = bytes(0x44, EXPERTS * n)
        .into_iter()
        .map(|b| to_bf16((f32::from(b) / 128.0 - 1.0) * 120.0))
        .collect();

    let mut gpu = Gpu::open();
    let act_at = gpu.up(&act);
    let codes_at = gpu.up(&codes);
    let scales_at = gpu.up(&scales);
    let bias_at = gpu.up(&bias);
    let routes_at = gpu.up(&routes);
    let grouped_at = gpu.zeros(route_count * n * 2);
    let per_route_at = gpu.zeros(route_count * n * 2);

    let ctx = gpu.ctx();
    let x = Tensor::new(act_at, act_rows as u32, k as u32, Dtype::Bf16);
    // One row per expert: the codes row is the whole plane at four bits an
    // element, the scales row one byte per 32-code block.
    let codes_t = Tensor::new(codes_at, EXPERTS as u32, (n * k / 2) as u32, Dtype::U8);
    let scales_t = Tensor::new(scales_at, EXPERTS as u32, (n * groups) as u32, Dtype::U8);
    let bias_t = Tensor::new(bias_at, EXPERTS as u32, n as u32, Dtype::Bf16);
    let routes_t = Tensor::new(routes_at, tokens as u32, TOP_K as u32, Dtype::I32);
    // **THE TWO DOORS ARE NOT ONE DOOR WITH A FLAG.** `matmul_select_quant`
    // reads the presence of a companion plane as the SCHEME — `Some` is an
    // affine bank's zero points and lands on the affine kernel entirely.
    // mxfp4's own output bias goes through `matmul_select_bias`, which is
    // what the IR's `MoeMatmulSelectBias` dispatches to.
    let mut fire = |into: u64| {
        let mut y = Tensor::new(into, route_count as u32, n as u32, Dtype::Bf16);
        if biased {
            matmul_select_bias(
                &ctx, x, codes_t, scales_t, bias_t, routes_t, &mut y, GroupSeat::RESIDENT,
            )
        } else {
            matmul_select_quant(
                &ctx, x, codes_t, scales_t, None, routes_t, &mut y, GroupSeat::RESIDENT,
            )
        }
        .expect("the mxfp4 select fires");
    };
    fire(grouped_at);
    {
        let _bare = Env::set("PIE_NO_MXFP4_GROUP");
        fire(per_route_at);
    }
    gpu.sync();

    let grouped: Vec<u16> = gpu.down(grouped_at, route_count * n);
    let per_route: Vec<u16> = gpu.down(per_route_at, route_count * n);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, p)) in grouped.iter().zip(&per_route).enumerate() {
        let miss = (from_bf16(*g) - from_bf16(*p)).abs();
        if miss > worst {
            worst = miss;
            at = i;
        }
    }
    assert!(
        close(from_bf16(grouped[at]), from_bf16(per_route[at])),
        "route {} column {} of a {} {} fire: grouped {} where the per-route select says {}",
        at / n,
        at % n,
        if by_token { "token-read" } else { "route-read" },
        if biased { "biased" } else { "bare" },
        from_bf16(grouped[at]),
        from_bf16(per_route[at]),
    );
}

/// The up leg: one activation row per token, a bias per (expert, row), and
/// a rectangle that divides both tiles — two K chunks by two row tiles.
#[test]
fn the_token_read_leg_with_a_bias_agrees() {
    check(64, 256, 256, true, true);
}

/// The down leg: one activation row per route already, and no bias.
#[test]
fn the_route_read_leg_agrees() {
    check(64, 256, 256, false, false);
}

/// The same two readings with the bias the other way round, because the
/// bias is read once per row pair OUTSIDE the batch loop and a leg that
/// read it inside, or per route, would still pass the two above.
#[test]
fn the_token_read_leg_without_a_bias_agrees() {
    check(64, 256, 256, true, false);
}

#[test]
fn the_route_read_leg_with_a_bias_agrees() {
    check(64, 256, 256, false, true);
}

/// **THE TAILS.** `n = 200` leaves a row tile part-full, so the grouped
/// kernel's clamped decode rows must not be written back; `k = 160` is five
/// mxfp4 blocks, so the last K chunk is part-full and the rest of the tile
/// has to read as zero. Both guards are the kind that pass every aligned
/// shape and fail every real one.
#[test]
fn a_rectangle_that_divides_neither_tile_agrees() {
    check(48, 200, 160, true, true);
}

/// **WHAT THE GROUPING IS WORTH**, at gpt-oss-20b's own mxfp4 shapes: 32
/// experts, top-4 over a 256-row canvas, the `[5760, 2880]` gate/up leg
/// read by token and the `[2880, 2880]` down leg read by route.
///
/// Prints the price; asserts only that the grouped leg is the faster one.
/// Run with `--test-threads=1` — two fires contending for one device price
/// each other, and this file's arms toggle a process-wide variable besides.
fn price(n: usize, k: usize, by_token: bool, label: &str) {
    let _one = serialized();
    let tokens = 256usize;
    let groups = k / BLOCK;
    let routes = routes_of(tokens);
    let route_count = routes.len();
    let act_rows = if by_token { tokens } else { route_count };

    let mut gpu = Gpu::open();
    let act: Vec<u16> = bytes(0x33, act_rows * k * 2)
        .chunks(2)
        .map(|p| to_bf16((f32::from(p[0]) / 128.0 - 1.0) * 0.5))
        .collect();
    let act_at = gpu.up(&act);
    // The bank is left zeroed: neither leg's price depends on what it
    // reads, and filling a quarter of a gigabyte per shape costs more than
    // the measurement.
    let codes_at = gpu.zeros(EXPERTS * n * k / 2);
    let scales_at = gpu.zeros(EXPERTS * n * groups);
    let routes_at = gpu.up(&routes);
    let y_at = gpu.zeros(route_count * n * 2);

    let ctx = gpu.ctx();
    let x = Tensor::new(act_at, act_rows as u32, k as u32, Dtype::Bf16);
    let codes_t = Tensor::new(codes_at, EXPERTS as u32, (n * k / 2) as u32, Dtype::U8);
    let scales_t = Tensor::new(scales_at, EXPERTS as u32, (n * groups) as u32, Dtype::U8);
    let routes_t = Tensor::new(routes_at, tokens as u32, TOP_K as u32, Dtype::I32);
    let mut y = Tensor::new(y_at, route_count as u32, n as u32, Dtype::Bf16);

    let mut run = || {
        for _ in 0..3 {
            matmul_select_quant(
                &ctx, x, codes_t, scales_t, None, routes_t, &mut y, GroupSeat::RESIDENT,
            )
            .expect("the mxfp4 select fires");
        }
        gpu.sync();
        const REPS: usize = 10;
        let t0 = Instant::now();
        for _ in 0..REPS {
            matmul_select_quant(
                &ctx, x, codes_t, scales_t, None, routes_t, &mut y, GroupSeat::RESIDENT,
            )
            .expect("the mxfp4 select fires");
        }
        gpu.sync();
        t0.elapsed().as_secs_f64() * 1e3 / REPS as f64
    };
    let grouped_ms = run();
    let per_route_ms = {
        let _bare = Env::set("PIE_NO_MXFP4_GROUP");
        run()
    };

    let flop = 2.0 * route_count as f64 * n as f64 * k as f64;
    eprintln!(
        "{label:9} n={n:5} k={k:5}: grouped {grouped_ms:7.3} ms ({:5.1} TFLOP/s)  \
         per-route {per_route_ms:7.3} ms ({:5.1} TFLOP/s)  {:.1}x",
        flop / grouped_ms / 1e9,
        flop / per_route_ms / 1e9,
        per_route_ms / grouped_ms,
    );
    assert!(
        grouped_ms < per_route_ms,
        "{label}: grouping took {grouped_ms:.3} ms against the per-route select's \
         {per_route_ms:.3}; this is the shape it exists for"
    );
}

#[test]
fn the_gpt_oss_legs_are_priced() {
    price(5760, 2880, true, "gate/up");
    price(2880, 2880, false, "down");
}

/// **THE fp32 FORM, WHICH NOTHING ELSE REACHES.** With a bf16 activation
/// the dispatch always takes the tensor-core kernel, so the fp32 grouped
/// one — the first form written, and the one an f16 row would take — would
/// ship having never run. `PIE_MXFP4_NO_WMMA` is what makes it reachable,
/// and this is what makes it checked: the same four shapes, same reference.
fn check_fp32_form(tokens: usize, n: usize, k: usize, by_token: bool, biased: bool) {
    let _one = serialized();
    let _fp32 = Env::set("PIE_MXFP4_NO_WMMA");
    check_held(tokens, n, k, by_token, biased);
}

#[test]
fn the_fp32_form_agrees_at_both_readings() {
    check_fp32_form(64, 256, 256, true, true);
    check_fp32_form(64, 256, 256, false, false);
}

#[test]
fn the_fp32_form_agrees_on_a_ragged_rectangle() {
    check_fp32_form(48, 200, 160, true, true);
}
