//! **`RngKind::Normal` IS N(0, 1), NOT SOMETHING NEAR IT.** A diffusion
//! sampler's whole first step is `x = sigma_max * z`, so a draw whose
//! variance is off by a few percent — a Box-Muller written with the wrong
//! angle constant, a pairing that reuses one uniform, an inverse-CDF
//! approximation — produces a latent of the wrong scale and an image that
//! is merely a bit wrong. Nothing downstream would say so. This measures the
//! first four moments of a million draws and pins the shape of the
//! distribution rather than the plausibility of one number.
//!
//! ```text
//! cargo test -p eta-ir --test the_normal_kind_draws_a_standard_normal
//! ```
//!
//! The tolerances are the standard errors of the estimators at `N = 1e6`
//! (mean 1/sqrt(N) = 1e-3, variance sqrt(2/N) = 1.4e-3) widened ~5x, which
//! is loose enough that a correct draw never trips it and tight enough that
//! a 1% scale error always does.

use eta_ir::rng::{NORMAL_PAIR_STRIDE, NORMAL_TWO_PI, hash_normal, hash_uniform, keyed_seed};

const N: usize = 1_000_000;

fn draws(key: u32, counter: u32) -> Vec<f32> {
    let seed = keyed_seed(key, counter);
    (0..N as u32).map(|i| hash_normal(seed, i)).collect()
}

#[test]
fn a_million_draws_have_the_moments_of_a_standard_normal() {
    let z = draws(0x7ce1, 0);

    let mean = z.iter().map(|&x| f64::from(x)).sum::<f64>() / N as f64;
    let variance = z
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / N as f64;
    let deviation = variance.sqrt();
    let skew = z
        .iter()
        .map(|&x| ((f64::from(x) - mean) / deviation).powi(3))
        .sum::<f64>()
        / N as f64;
    let kurtosis = z
        .iter()
        .map(|&x| ((f64::from(x) - mean) / deviation).powi(4))
        .sum::<f64>()
        / N as f64;

    assert!(mean.abs() < 5e-3, "mean {mean} is not zero");
    assert!(
        (variance - 1.0).abs() < 8e-3,
        "variance {variance} is not one"
    );
    assert!(skew.abs() < 2e-2, "skew {skew} is not zero");
    assert!(
        (kurtosis - 3.0).abs() < 6e-2,
        "kurtosis {kurtosis} is not three"
    );

    // The tails are the half a moment test can miss: a truncated or clamped
    // draw keeps its mean and variance and loses its shape.
    let beyond = |t: f32| z.iter().filter(|&&x| x.abs() > t).count() as f64 / N as f64;
    assert!(
        (beyond(1.0) - 0.317_310).abs() < 3e-3,
        "|z| > 1 mass {} is not 0.3173",
        beyond(1.0)
    );
    assert!(
        (beyond(2.0) - 0.045_500).abs() < 2e-3,
        "|z| > 2 mass {} is not 0.0455",
        beyond(2.0)
    );
    assert!(
        (beyond(3.0) - 0.002_700).abs() < 6e-4,
        "|z| > 3 mass {} is not 0.0027",
        beyond(3.0)
    );
    assert!(z.iter().all(|x| x.is_finite()), "a draw was not finite");
}

#[test]
fn the_draw_is_the_pair_transform_the_contract_states() {
    // Not "some normal": the exact expression every backend projects, over
    // the exact two lanes the stride names.
    let seed = keyed_seed(11, 3);
    for index in [0u32, 1, 2, 97, 65_535] {
        let lane = index * NORMAL_PAIR_STRIDE;
        let u0 = hash_uniform(seed, lane);
        let u1 = hash_uniform(seed, lane + 1);
        let want = (-2.0f32 * u0.ln()).sqrt() * (NORMAL_TWO_PI * u1).cos();
        assert_eq!(
            hash_normal(seed, index).to_bits(),
            want.to_bits(),
            "lane {index}"
        );
    }
}

#[test]
fn two_counters_of_one_key_draw_independent_noise() {
    // The denoise loop advances `ctr` a step and expects fresh noise; a
    // state word folded in weakly would leave the two draws correlated.
    let a = draws(0x51ee, 0);
    let b = draws(0x51ee, 1);
    let mean_a = a.iter().map(|&x| f64::from(x)).sum::<f64>() / N as f64;
    let mean_b = b.iter().map(|&x| f64::from(x)).sum::<f64>() / N as f64;
    let covariance = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| (f64::from(x) - mean_a) * (f64::from(y) - mean_b))
        .sum::<f64>()
        / N as f64;
    assert!(
        covariance.abs() < 5e-3,
        "counters 0 and 1 draw correlated noise (covariance {covariance})"
    );
    assert!(
        a.iter().zip(&b).any(|(x, y)| x != y),
        "two counters drew the same stream"
    );
}
