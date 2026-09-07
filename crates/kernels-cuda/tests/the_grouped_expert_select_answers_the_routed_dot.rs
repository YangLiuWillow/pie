//! **THE GROUPED ROUTED MATMUL, AGAINST THE DOT IT CLAIMS TO COMPUTE.**
//!
//! `matmul_select` fired the per-route GEMV at every width. A prefill names
//! each expert many times over, and the GEMV re-reads the whole bank once
//! per route that names it, so the wide fire it was serving cost an order
//! of magnitude more bandwidth than the work needs. The grouped leg sorts
//! the routes by expert, gathers the activations behind that permutation,
//! and hands cuBLAS one batched GEMM whose every block reads its bank once.
//!
//! Reordering a sum is not free of consequence — the batched GEMM
//! accumulates in a different order from the GEMV and lands different last
//! bits — so this reads both against the host's own dot rather than against
//! each other.
//!
//!   cargo test --release -p kernels-cuda --features cuda \
//!     --test the_grouped_expert_select_answers_the_routed_dot

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{ExpertTable, matmul_select};
use kernels_cuda::tensor::Tensor;

/// How a fire's routes are spread over the experts.
#[derive(Clone, Copy)]
enum Spread {
    /// Deterministic and skewed — an even spread would hide a block whose
    /// expert id was read off by one.
    Mixed,
    /// **THE BLOCK BUDGET'S WORST CASE**: one expert takes half the fire
    /// while the rest still hold rows, so the dominant expert spans several
    /// blocks AND every other expert still claims a partial one. That sum —
    /// not `experts`, and not `routes / block` — is what the budget has to
    /// cover. Note that routing EVERYTHING to one expert is the easy case,
    /// not the hard one: it needs fewer blocks, not more.
    Dominant,
}

/// The routing a token's `j`th pick lands on.
fn expert_of(token: usize, j: usize, experts: usize, spread: Spread) -> i32 {
    let e = match spread {
        Spread::Mixed => (token * 3 + j * 5 + token / 7) % experts,
        Spread::Dominant if j == 0 => 0,
        Spread::Dominant => 1 + token % (experts - 1),
    };
    i32::try_from(e).expect("an expert id inside i32")
}

/// `y[route] = x[row(route)] . bank[expert(route)]`, in f32, on the host.
/// The one definition both device legs answer to.
fn routed_dot(
    x: &[f32],
    bank: &[f32],
    routes: &[i32],
    top_k: usize,
    k: usize,
    n: usize,
    by_token: bool,
) -> Vec<f32> {
    let mut y = vec![0f32; routes.len() * n];
    for (route, &expert) in routes.iter().enumerate() {
        let row = if by_token { route / top_k } else { route };
        let expert = expert as usize;
        for col in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[row * k + i] * bank[(expert * n + col) * k + i];
            }
            y[route * n + col] = acc;
        }
    }
    y
}

/// One fire, checked against [`routed_dot`]. `x_rows` states the reading:
/// one row per token is the up leg, one per route the down leg.
fn check(
    experts: usize,
    tokens: usize,
    top_k: usize,
    k: usize,
    n: usize,
    by_token: bool,
    spread: Spread,
) {
    let routes: Vec<i32> = (0..tokens)
        .flat_map(|t| (0..top_k).map(move |j| expert_of(t, j, experts, spread)))
        .collect();
    let route_count = routes.len();
    let x_rows = if by_token { tokens } else { route_count };

    let mut rng = Lcg::seeded(0x5eed_1234);
    let (x_raw, x_exact) = rng.row(x_rows * k);
    let (bank_raw, bank_exact) = rng.row(experts * n * k);
    let want = routed_dot(&x_exact, &bank_exact, &routes, top_k, k, n, by_token);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let bank_at = gpu.up(&bank_raw);
    let routes_at = gpu.up(&routes);
    let y_at = gpu.zeros(route_count * n * 2);

    let ctx = gpu.ctx();
    let x = Tensor::new(
        x_at,
        u32::try_from(x_rows).unwrap(),
        u32::try_from(k).unwrap(),
        Dtype::Bf16,
    );
    // `[experts, N, K]` as the engine flattens it: `experts * N` rows of K.
    let bank = Tensor::new(
        bank_at,
        u32::try_from(experts * n).unwrap(),
        u32::try_from(k).unwrap(),
        Dtype::Bf16,
    );
    let routes_t = Tensor::new(
        routes_at,
        u32::try_from(tokens).unwrap(),
        u32::try_from(top_k).unwrap(),
        Dtype::I32,
    );
    let mut y = Tensor::new(
        y_at,
        u32::try_from(route_count).unwrap(),
        u32::try_from(n).unwrap(),
        Dtype::Bf16,
    );

    matmul_select(&ctx, x, bank, routes_t, &mut y, ExpertTable::RESIDENT)
        .expect("the routed select answers this shape");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, route_count * n);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&raw, &want)) in got.iter().zip(want.iter()).enumerate() {
        let miss = (from_bf16(raw) - want).abs();
        if miss > worst {
            worst = miss;
            at = i;
        }
    }
    assert!(
        close(from_bf16(got[at]), want[at]),
        "route {} column {} of a {} fire answered {} where the dot says {}",
        at / n,
        at % n,
        if by_token { "token-read" } else { "route-read" },
        from_bf16(got[at]),
        want[at],
    );
}

/// The up leg: one activation row per token, read once per route.
#[test]
fn the_token_read_leg_answers_the_dot() {
    check(8, 48, 2, 64, 96, true, Spread::Mixed);
}

/// The down leg: one activation row per route already.
#[test]
fn the_route_read_leg_answers_the_dot() {
    check(8, 48, 2, 64, 96, false, Spread::Mixed);
}

/// **THE WIDTH THE GEMV REFUSES.** Its route run rides the grid's y axis,
/// which stops at 65535, so a fire past that is one the per-route leg
/// cannot serve at all: passing here is the proof that the grouped leg —
/// not a silent fallback — computed this.
#[test]
fn a_fire_wider_than_the_gemvs_grid_is_still_answered() {
    check(8, 33_000, 2, 32, 32, true, Spread::Mixed);
}

/// **THE WIDTH THAT MUST NOT GROUP.** A decode fire names each expert about
/// once, so sorting and gathering would buy nothing and the per-route GEMV
/// is already the right shape. This is the fallback still answering — and
/// the same dot answering it.
#[test]
fn a_decode_width_fire_is_still_served() {
    check(8, 1, 2, 64, 96, true, Spread::Mixed);
}

/// **THE BLOCK BUDGET UNDER ITS WORST CASE.** A dominant expert spanning
/// several blocks while every other expert still claims a partial one sums
/// past `experts`, so the budget has to be `experts + ceil(routes/block)`.
/// Too tight and the alignment drops the routes past it — silently, since
/// the kernel guards that write rather than reporting it.
#[test]
fn a_dominant_expert_still_fits_the_block_budget() {
    check(8, 48, 2, 64, 96, true, Spread::Dominant);
}

/// An expert count that is not a power of two, and a fan-out that does not
/// divide it: the alignment's per-expert prefix sum walks a ragged tail.
#[test]
fn a_ragged_expert_count_still_lands() {
    check(48, 40, 3, 64, 96, true, Spread::Mixed);
}
