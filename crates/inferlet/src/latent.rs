//! `inferlet::latent` — the sampler prelude for the generative families
//! (imagegen design D4). The sampler is an eta EPILOGUE: per step the guest
//! submits, the epilogue reads `velocity()`, the latent cell and the
//! control cell, writes the next latent and advances the control word, and
//! no bytes cross to the host. What lives here is the arithmetic that
//! epilogue spells, the schedule it steps along, the noise it starts from,
//! and the one host-side helper every text-conditioned model wants
//! ([`encode_text`]).
//!
//! Model facts come from `model::schedule()` / `model::latent()` /
//! `model::readings()`, never from `architecture()`.
//!
//! ```ignore
//! use inferlet::latent::prelude::*;
//!
//! let sched = FlowMatchEuler::from_model(steps, None)?;   // sigmas + fixed shift
//! let x = Channel::from_shaped([rows, width], vec![0f32; (rows * width) as usize]);
//! let t = Channel::from([sched.timestep(0)]);
//! let step = Channel::from([0u32]);
//! let denoise = ForwardPass::new();
//! denoise.reading("denoise")?;
//! denoise.input("latents", &x)?;
//! denoise.input("timestep", &t)?;
//! denoise.prologue(move || { x.take(); x.put(noise([rows, width], seed)); });
//! denoise.epilogue(move || {
//!     let i = step.take();
//!     let v = intrinsics::velocity(width);
//!     x.put(euler_step(&x.take(), &v, dt.gather(i)));
//!     ...
//! });
//! ```

use crate::eta::{
    Channel, KvGeometry, Pipeline, WorkingSet, broadcast, gather, intrinsics, max_elem, min_elem,
    normal, reduce_sum, reshape, sqrt,
};
use crate::model::{ReadoutKind, ScheduleFact, ScheduleKind};
use eta_dsl::Tensor;

/// The glob-import surface for a sampler author: everything here plus the
/// attention pass prelude (a DiT's passes are `forward` passes).
pub mod prelude {
    pub use super::{
        FlowMatchEuler, apg, cfg_combine, dynamic_shift, encode_text, euler_step, noise,
        positions_grid, rng_state,
    };
    pub use crate::eta::attention::prelude::*;
}

/// A rectified-flow Euler schedule: `sigmas` descending from 1 towards 0,
/// `steps + 1` of them (the last is 0), so step `i` integrates from
/// `sigmas[i]` to `sigmas[i + 1]` with `dt(i) = sigmas[i + 1] - sigmas[i]`
/// (negative: the latent moves against the velocity).
///
/// Built from the model's [`ScheduleFact`]: a distilled model's pinned
/// sigmas are taken as-is (resampled to `steps` when they differ), an
/// undistilled one's are `linspace(1, 1/steps)` under the trained time
/// shift `sigma' = shift·sigma / (1 + (shift − 1)·sigma)`.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowMatchEuler {
    /// `steps + 1` sigmas, descending, ending in 0.
    pub sigmas: Vec<f32>,
    /// The timestep axis scale (`train-steps`; 1000 for the diffusers
    /// families): `timestep(i) = sigmas[i] · train_steps`.
    pub train_steps: u32,
}

impl FlowMatchEuler {
    /// The bound model's schedule for `steps` steps. `rows` — the latent
    /// row count — turns on the dynamic shift (see [`dynamic_shift`]) for
    /// a family whose stated `shift` is its base `mu` (FLUX, Z-Image);
    /// `None` takes the stated shift as a fixed one (Wan, LTX). Which a
    /// family is, its report says; the fact does not. Refuses a model with
    /// no schedule, or one that is not a flow.
    pub fn from_model(steps: u32, rows: Option<u32>) -> Result<FlowMatchEuler, String> {
        let Some(schedule) = crate::model::schedule() else {
            return Err("this model states no schedule; nothing here denoises".to_string());
        };
        FlowMatchEuler::from_schedule(&schedule, steps, rows)
    }

    /// The schedule `fact` describes, at `steps` steps. `rows` enables the
    /// dynamic shift (see [`dynamic_shift`]) for a flow model whose stated
    /// shift is its base `mu`; `None` uses the stated shift as-is.
    pub fn from_schedule(
        fact: &ScheduleFact,
        steps: u32,
        rows: Option<u32>,
    ) -> Result<FlowMatchEuler, String> {
        if fact.kind != ScheduleKind::Flow {
            return Err(format!(
                "FlowMatchEuler integrates a flow schedule; this model's is {:?}",
                fact.kind
            ));
        }
        let steps = steps.max(1);
        let mut sigmas: Vec<f32> = if fact.pinned_sigmas.is_empty() {
            let raw: Vec<f32> = (0..steps).map(|i| 1.0 - i as f32 / steps as f32).collect();
            let shift = match rows {
                Some(rows) if fact.shift > 0.0 => dynamic_shift(rows, fact.shift),
                _ => fact.shift,
            };
            raw.into_iter()
                .map(|sigma| shift_sigma(sigma, shift))
                .collect()
        } else {
            resample(&fact.pinned_sigmas, steps as usize)
        };
        sigmas.push(0.0);
        Ok(FlowMatchEuler {
            sigmas,
            train_steps: fact.train_steps.max(1),
        })
    }

    /// One schedule from sigmas stated outright (a guest's own).
    #[must_use]
    pub fn from_sigmas(mut sigmas: Vec<f32>, train_steps: u32) -> FlowMatchEuler {
        if sigmas.last().is_none_or(|&last| last != 0.0) {
            sigmas.push(0.0);
        }
        FlowMatchEuler {
            sigmas,
            train_steps: train_steps.max(1),
        }
    }

    /// How many steps this schedule takes.
    #[must_use]
    pub fn steps(&self) -> u32 {
        u32::try_from(self.sigmas.len().saturating_sub(1)).unwrap_or(u32::MAX)
    }

    /// `sigmas[i + 1] − sigmas[i]`, the signed step every Euler update
    /// integrates over; 0 past the last step.
    #[must_use]
    pub fn dt(&self, i: u32) -> f32 {
        let i = i as usize;
        match (self.sigmas.get(i), self.sigmas.get(i + 1)) {
            (Some(&from), Some(&to)) => to - from,
            _ => 0.0,
        }
    }

    /// Every step's `dt`, for a `[steps]` control channel the epilogue
    /// gathers by step index.
    #[must_use]
    pub fn dts(&self) -> Vec<f32> {
        (0..self.steps()).map(|i| self.dt(i)).collect()
    }

    /// The timestep the model reads at step `i`: `sigma · train_steps`.
    #[must_use]
    pub fn timestep(&self, i: u32) -> f32 {
        self.sigmas.get(i as usize).copied().unwrap_or(0.0) * self.train_steps as f32
    }

    /// Every step's timestep, for a `[steps]` control channel.
    #[must_use]
    pub fn timesteps(&self) -> Vec<f32> {
        (0..self.steps()).map(|i| self.timestep(i)).collect()
    }

    /// The first step whose sigma is at or below `boundary` — where a
    /// two-backbone family (`schedule-fact.boundary`) hands over from its
    /// high-noise arm to its low-noise one. `steps()` when none is.
    #[must_use]
    pub fn handover(&self, boundary: f32) -> u32 {
        self.sigmas[..self.sigmas.len().saturating_sub(1)]
            .iter()
            .position(|&sigma| sigma <= boundary)
            .map_or(self.steps(), |i| i as u32)
    }
}

/// diffusers' time shift: `shift·σ / (1 + (shift − 1)·σ)`.
fn shift_sigma(sigma: f32, shift: f32) -> f32 {
    if shift <= 0.0 || shift == 1.0 {
        return sigma;
    }
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

/// The dynamic time shift a flow model resolves from its token count
/// (diffusers' `calculate_shift`): `mu` interpolates linearly from
/// `base_mu` at 256 rows to 1.15 at 4096 rows, and the shift is `exp(mu)`.
/// `base_mu` is the family's stated `shift` when it is a base `mu` (FLUX's
/// 0.5); a family that states a fixed shift passes `None` rows instead.
#[must_use]
pub fn dynamic_shift(rows: u32, base_mu: f32) -> f32 {
    const BASE_ROWS: f32 = 256.0;
    const MAX_ROWS: f32 = 4096.0;
    const MAX_MU: f32 = 1.15;
    let slope = (MAX_MU - base_mu) / (MAX_ROWS - BASE_ROWS);
    let mu = slope * rows as f32 + (base_mu - slope * BASE_ROWS);
    mu.exp()
}

/// Linear resampling of a pinned sigma list to `steps` entries.
fn resample(pinned: &[f32], steps: usize) -> Vec<f32> {
    if pinned.len() == steps || pinned.len() < 2 {
        return pinned.to_vec();
    }
    (0..steps)
        .map(|i| {
            let at = i as f32 * (pinned.len() - 1) as f32 / (steps.max(2) - 1) as f32;
            let lo = at.floor() as usize;
            let hi = (lo + 1).min(pinned.len() - 1);
            let frac = at - lo as f32;
            pinned[lo] * (1.0 - frac) + pinned[hi] * frac
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Epilogue arithmetic
// ---------------------------------------------------------------------------

/// `x + dt · v`: one Euler step of the flow. `x`/`v` are `[rows, width]`
/// f32; `dt` is a scalar tensor (a `[1]` control cell's value, or a
/// constant) broadcast over every element.
pub fn euler_step(x: &Tensor, v: &Tensor, dt: &Tensor) -> Tensor {
    let shape = x.shape();
    x + &(v * &broadcast(dt, shape))
}

/// Classifier-free guidance: `uncond + s · (cond − uncond)`. `cond` and
/// `uncond` are the two lanes' velocity rows (`[rows, width]` each); `s`
/// is a scalar tensor or constant.
pub fn cfg_combine(cond: &Tensor, uncond: &Tensor, s: &Tensor) -> Tensor {
    let shape = cond.shape();
    uncond + &(&(cond - uncond) * &broadcast(s, shape))
}

/// Adaptive projected guidance (Sadat et al., 2410.02416): the guidance
/// difference `cond − uncond` is rescaled to at most `norm_threshold` per
/// row (0 disables), split into the component parallel to `cond` and the
/// one orthogonal to it, and recombined as `orth + eta · parallel`;
/// the result is `cond + (s − 1) · guidance`. Row-wise over `[rows, width]`.
pub fn apg(cond: &Tensor, uncond: &Tensor, s: f32, eta: f32, norm_threshold: f32) -> Tensor {
    let shape = cond.shape();
    let rows = shape.dims()[0];
    let per_row = |x: &Tensor| broadcast(reshape(x, [rows, 1]), shape);
    let mut diff = cond - uncond;
    if norm_threshold > 0.0 {
        let norm = sqrt(reduce_sum(&(&diff * &diff)));
        let scale = min_elem(
            &(&Tensor::constant(norm_threshold) / &max_elem(&norm, 1e-12f32)),
            1.0f32,
        );
        diff = &diff * &per_row(&scale);
    }
    let dot = reduce_sum(&(&diff * cond));
    let cond_sq = max_elem(reduce_sum(&(cond * cond)), 1e-12f32);
    let parallel = cond * &per_row(&(&dot / &cond_sq));
    let orthogonal = &diff - &parallel;
    let guidance = &orthogonal + &(&parallel * eta);
    cond + &(&guidance * (s - 1.0))
}

/// The `[2] u32` keyed-RNG state `[key, counter]` for `seed`: what
/// [`noise`] draws from, and what a loop-carried channel advances.
#[must_use]
pub fn rng_state(seed: u32) -> [u32; 2] {
    [seed, 0]
}

/// A seeded standard-normal draw of `shape`, bit-identical across backends
/// (`RngKind::Normal` over the keyed formula). `seed` is a `[2] u32` state
/// tensor ([`rng_state`] as a constant, or a channel's value); the caller
/// advances the counter for the next draw. A prologue that `put`s this into
/// the latent channel fills the initial latent on the device — no bytes
/// through WASM.
pub fn noise(shape: impl eta_dsl::IntoShape, seed: &Tensor) -> Tensor {
    normal(seed, shape)
}

/// Gather the scalar at `index` of a `[n]` control vector (a `dts` or
/// `timesteps` channel), as a `[1]` tensor.
pub fn at(vector: &Tensor, index: &Tensor) -> Tensor {
    gather(vector, reshape(index, [1]))
}

// ---------------------------------------------------------------------------
// Host-side builders
// ---------------------------------------------------------------------------

/// The `[t·h·w, 3]` f32 axis-position grid for an `AxisPositions` port:
/// row `(i, j, k)` in `t`-major order carries `[i + offsets[0], j +
/// offsets[1], k + offsets[2]]`. An image is `t = 1`; a reference image at
/// FLUX's `T = 10·(n+1)` passes that as `offsets[0]`.
#[must_use]
pub fn positions_grid(t: u32, h: u32, w: u32, offsets: [f32; 3]) -> Vec<f32> {
    let mut grid = Vec::with_capacity((t * h * w * 3) as usize);
    for i in 0..t {
        for j in 0..h {
            for k in 0..w {
                grid.push(i as f32 + offsets[0]);
                grid.push(j as f32 + offsets[1]);
                grid.push(k as f32 + offsets[2]);
            }
        }
    }
    grid
}

/// Encode `prompt` with the family's text reading: tokenize, run one
/// `forward` pass in `reading` (`"text"` for every family that has one)
/// with `embed` + `attention` over a fresh working set, read `hidden()` at
/// every row, and hand the rows back as a SEEDED `[rows, width]` f32
/// channel the denoise pass binds with `input("context", ..)`. The family's
/// template and padding contract live in its encode arm, so the guest
/// tokenizes plain text. The rows cross the host once per prompt (the
/// channel roles cannot chain a terminal output device-to-device yet).
pub async fn encode_text(prompt: &str, reading: &str) -> Result<Channel, String> {
    let Some(fact) = crate::model::reading(reading) else {
        return Err(format!("this model declares no reading `{reading}`"));
    };
    if !(fact.has_kv && fact.takes_tokens) {
        return Err(format!(
            "reading `{reading}` is not a text encoder (it needs tokens and a KV space)"
        ));
    }
    if fact.readout != ReadoutKind::Hidden {
        return Err(format!(
            "reading `{reading}` reads out {:?}, not the hidden rows an encoder hands over",
            fact.readout
        ));
    }
    let width = fact.readout_width;
    let ids: Vec<i32> = crate::model::encode(prompt)
        .into_iter()
        .map(|id| i32::try_from(id).unwrap_or(0))
        .collect();
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    if len == 0 {
        return Err("the prompt tokenizes to nothing".to_string());
    }
    let page_size = crate::eta::kv_page_size().max(1);
    let pages = len.div_ceil(page_size);
    let ws = WorkingSet::new();
    ws.reserve(pages)?;
    let pipe = Pipeline::new();

    let toks = Channel::from(ids);
    let embed_indptr = Channel::from([0u32, len]);
    let positions = Channel::from_iter(0..len);
    let page_ids = Channel::from_iter(0..pages);
    let page_indptr = Channel::from([0u32, pages]);
    let w_slot = Channel::from_iter((0..len).map(|p| p / page_size));
    let w_off = Channel::from_iter((0..len).map(|p| p % page_size));
    let kv_len = Channel::from([len]);
    let readout = Channel::from_iter(0..len);
    let out = Channel::new([len, width], crate::eta::Dtype::F32);

    let pass = crate::eta::attention::ForwardPass::new();
    pass.reading(reading)?;
    pass.embed(&toks, &embed_indptr)?;
    pass.readout(&readout)?;
    pass.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len,
            pages: &page_ids,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    pass.epilogue(move || {
        out.put(intrinsics::hidden(width));
    });
    pass.submit(&pipe)?;
    let rows: Vec<f32> = out.take_host().await?;
    pipe.close();
    Ok(Channel::from_shaped([len, width], rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flow_schedule_ends_at_zero_and_steps_downhill() {
        let fact = ScheduleFact {
            kind: ScheduleKind::Flow,
            shift: 1.0,
            train_steps: 1000,
            boundary: None,
            pinned_sigmas: Vec::new(),
        };
        let sched = FlowMatchEuler::from_schedule(&fact, 4, None).unwrap();
        assert_eq!(sched.sigmas, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
        assert_eq!(sched.steps(), 4);
        assert_eq!(sched.dt(0), -0.25);
        assert_eq!(sched.timestep(0), 1000.0);
        assert_eq!(sched.dts().len(), 4);
    }

    #[test]
    fn a_shift_bends_the_sigmas_up() {
        let fact = ScheduleFact {
            kind: ScheduleKind::Flow,
            shift: 3.0,
            train_steps: 1000,
            boundary: Some(0.875),
            pinned_sigmas: Vec::new(),
        };
        let sched = FlowMatchEuler::from_schedule(&fact, 4, None).unwrap();
        assert!(sched.sigmas[1] > 0.75, "{:?}", sched.sigmas);
        assert_eq!(sched.sigmas[0], 1.0);
        assert_eq!(*sched.sigmas.last().unwrap(), 0.0);
        assert!(sched.handover(0.875) >= 1);
    }

    #[test]
    fn pinned_sigmas_are_taken_and_resampled() {
        let fact = ScheduleFact {
            kind: ScheduleKind::Flow,
            shift: 1.0,
            train_steps: 1000,
            boundary: None,
            pinned_sigmas: vec![1.0, 0.5],
        };
        let sched = FlowMatchEuler::from_schedule(&fact, 2, None).unwrap();
        assert_eq!(sched.sigmas, vec![1.0, 0.5, 0.0]);
        let sched = FlowMatchEuler::from_schedule(&fact, 3, None).unwrap();
        assert_eq!(sched.sigmas, vec![1.0, 0.75, 0.5, 0.0]);
    }

    #[test]
    fn the_dynamic_shift_interpolates_mu() {
        assert!((dynamic_shift(256, 0.5) - 0.5f32.exp()).abs() < 1e-5);
        assert!((dynamic_shift(4096, 0.5) - 1.15f32.exp()).abs() < 1e-5);
    }

    #[test]
    fn a_positions_grid_is_t_major_with_offsets() {
        let grid = positions_grid(1, 2, 2, [10.0, 0.0, 0.0]);
        assert_eq!(
            grid,
            vec![
                10.0, 0.0, 0.0, 10.0, 0.0, 1.0, 10.0, 1.0, 0.0, 10.0, 1.0, 1.0
            ]
        );
    }
}
