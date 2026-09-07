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
    Channel, KvGeometry, Pipeline, WorkingSet, attention::ForwardPass, broadcast, eq, gather,
    intrinsics, max_elem, min_elem, normal, reduce_sum, reshape, select, sqrt,
};
use crate::model::{AxisRole, PositionConvention, ReadoutKind, ScheduleFact, ScheduleKind};
use eta_dsl::Tensor;

/// The glob-import surface for a sampler author: everything here plus the
/// attention pass prelude (a DiT's passes are `forward` passes).
pub mod prelude {
    pub use super::{
        DenoiseLoop, FlowMatchEuler, LaneClock, LaneRows, apg, at, cfg_combine, dynamic_shift,
        encode_ids, encode_ids_rows, encode_text, euler_step, guided_velocity, noise,
        positions_for, positions_grid, rng_state, seed_or_step,
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
    /// The schedule ONE stream's lanes advance on, for a family that runs
    /// several inside one evaluation (`schedule-fact.stream-shifts`:
    /// MiniMax H3's video grid is built at shift 12 and its audio grid at
    /// 3, and one step advances both). A stream the fact does not name
    /// takes the family-wide `shift`, so this is
    /// [`from_schedule`](FlowMatchEuler::from_schedule) for every other
    /// family.
    ///
    /// # Errors
    ///
    /// The model's schedule is not a flow schedule.
    pub fn for_stream(
        fact: &ScheduleFact,
        lane: crate::model::LaneStream,
        steps: u32,
        rows: Option<u32>,
    ) -> Result<FlowMatchEuler, String> {
        let mut fact = fact.clone();
        if let Some(found) = fact
            .stream_shifts
            .iter()
            .find(|shift| shift.lane == lane)
            .map(|shift| shift.shift)
        {
            fact.shift = found;
            // A pinned list is the family's own grid at the family-wide
            // shift; a per-stream shift replaces the grid, not scales it.
            fact.pinned_sigmas.clear();
        }
        FlowMatchEuler::from_schedule(&fact, steps, rows)
    }

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
///
/// Both rows are readable inside one epilogue: the lane reads its own with
/// [`intrinsics::velocity`] and the other lane's with
/// [`intrinsics::peer_velocity`], after naming it with `ForwardPass::peer`.
/// The two lanes must share an attention group, and the group's cohort is
/// what makes them one fire. See [`guided_velocity`] for the whole shape.
pub fn cfg_combine(cond: &Tensor, uncond: &Tensor, s: &Tensor) -> Tensor {
    let shape = cond.shape();
    uncond + &(&(cond - uncond) * &broadcast(s, shape))
}

/// The guided velocity of a lane that named a peer: this lane's own
/// prediction combined with its peer's at guidance `s`, all on the device,
/// inside one epilogue.
///
/// Which of the two lanes is `cond` and which is `uncond` is the guest's to
/// say, because only the guest knows which context it fed each lane:
/// `conditional = true` means THIS lane took the prompt and its peer took
/// the negative one. At `s == 1.0` the combine is the identity on this
/// lane's own rows, which is what makes a distilled row's guidance-1 path
/// exactly its ungated one — a useful thing to be able to check.
///
/// The two branches are two attention GROUPS of one fire, not two lanes of
/// one group: they are independent denoisings and must not attend each
/// other's rows. Each group carries its own context lane — the prompt for
/// one, the negative prompt for the other — and its own image lane.
///
/// ```ignore
/// // Group 0 is the conditional branch, group 1 the unconditional one.
/// cond_image.group(0)?;
/// cond_image.peer(1)?;                   // the branch to guide against
/// cond_image.epilogue(move || {
///     let v = guided_velocity(width, guidance, true);
///     seed_or_step(&k, &x, &v, &dts, &rng, shape, Some(&out));
/// });
/// // The uncond image lane needs no epilogue of its own: its velocity is
/// // read by its peer, off the plane the forward walk already wrote.
/// ```
#[must_use]
pub fn guided_velocity(width: u32, s: f32, conditional: bool) -> Tensor {
    let own = intrinsics::velocity(width);
    let peer = intrinsics::peer_velocity(width);
    let scale = Tensor::constant([s]);
    if conditional {
        cfg_combine(&own, &peer, &scale)
    } else {
        cfg_combine(&peer, &own, &scale)
    }
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

/// Which lane's rows a [`positions_for`] grid is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneRows {
    /// A text or context lane of `rows` rows, numbered along the
    /// convention's `text_axis`.
    Sequence(u32),
    /// An image lane over an `h × w` latent-row grid, `h`-major (the packed
    /// order every family's `positions` port reads).
    Grid { h: u32, w: u32 },
    /// **A VIDEO lane over a `t × h × w` latent-row volume**, `t`-major then
    /// `h`-major — the order a video family's `latents` port packs its rows
    /// in, and the reason this is a variant and not `Grid { h: t·h, w }`:
    /// the temporal coordinate is its OWN axis of the rotary space (Wan's
    /// `(t, h, w)`), so folding it into the height would place every frame
    /// of a clip at a different height and none of them at a different
    /// time.
    ///
    /// `Grid { h, w }` is `Volume { t: 1, h, w }` with the `Time` axis left
    /// at 0 — an image is one frame — and is kept separate because an image
    /// family may put something else on that axis (FLUX.2's reference
    /// index).
    Volume { t: u32, h: u32, w: u32 },
}

/// The `[rows, axes]` f32 grid one lane binds to a reading's
/// `AxisPositions` port, built from the family's own
/// [`PositionConvention`](crate::model::PositionConvention) so the guest
/// never spells a rotary layout.
///
/// `text_rows` is how many rows the request's TEXT lane carries — the image
/// grid needs it when the family stacks the image behind the caption on one
/// axis (`image_follows_text`); pass 0 for a lane set with no text lane.
///
/// Rows come out in the port's own order: `Sequence` numbers row `j` at
/// `text_origin + j` on `text_axis` and 0 elsewhere; `Grid`/`Volume` put
/// `i` on the `Time` axis, `a` on the `Height` axis, `b` on the `Width`
/// axis, the image's caption offset on `text_axis` WHEN the family stacks
/// the image behind the caption, and 0 on everything else.
///
/// `text_axis` only overrides a role under `image_follows_text` — a video
/// family numbers its frames on the `Time` axis and has no caption offset
/// to put there (Wan's context lane binds no positions at all: its
/// cross-attention has no rope), so a convention that says
/// `image_follows_text: false` leaves every axis to its role.
#[must_use]
pub fn positions_for(convention: &PositionConvention, lane: LaneRows, text_rows: u32) -> Vec<f32> {
    let axes = convention.axes.len();
    let text_axis = convention.text_axis as usize;
    let origin = convention.text_origin as f32;
    let (t, h, w) = match lane {
        LaneRows::Sequence(rows) => {
            let mut grid = vec![0f32; rows as usize * axes];
            for j in 0..rows as usize {
                if let Some(cell) = grid.get_mut(j * axes + text_axis) {
                    *cell = origin + j as f32;
                }
            }
            return grid;
        }
        LaneRows::Grid { h, w } => (1, h, w),
        LaneRows::Volume { t, h, w } => (t, h, w),
    };
    let follow = origin + text_rows as f32;
    let mut grid = Vec::with_capacity((t * h * w) as usize * axes);
    for i in 0..t {
        for a in 0..h {
            for b in 0..w {
                for (axis, role) in convention.axes.iter().enumerate() {
                    grid.push(if axis == text_axis && convention.image_follows_text {
                        follow
                    } else {
                        match role {
                            AxisRole::Time => i as f32,
                            AxisRole::Height => a as f32,
                            AxisRole::Width => b as f32,
                            AxisRole::Index => 0.0,
                        }
                    });
                }
            }
        }
    }
    grid
}

// ---------------------------------------------------------------------------
// The step loop
// ---------------------------------------------------------------------------

/// The loop-carried clock of ONE denoise lane: which fire it is on, the
/// `timestep` cell its pass binds, and the schedule's timestep vector it
/// gathers the next one out of.
///
/// **EVERY LANE OF A STEP NEEDS ITS OWN.** A seeded channel attaches to one
/// pass only (the runtime's channel-role rule), so the lanes of one denoise
/// group cannot share a timestep cell; each carries a copy and each advances
/// it in its own epilogue, which is what keeps the group on one fire index.
///
/// Fire 0 is the SEED — its velocity is discarded and its `dt` is 0 — and
/// step `i` is fire `i + 1`.
#[derive(Clone)]
pub struct LaneClock {
    /// `[1] u32`: the fire index this lane's next epilogue reads.
    pub step: Channel,
    /// `[1] f32`: the cell the `timestep` port is bound to.
    pub timestep: Channel,
    /// `[fires + 1] f32`: every fire's timestep, with one trailing 0 so the
    /// last fire's advance gathers something.
    ts: Channel,
}

impl LaneClock {
    /// Install this lane's epilogue: `body` runs with the fire index `k`
    /// this fire is on, then the clock advances to `k + 1`. A lane that only
    /// modulates (a text lane) passes a body that does nothing; the image
    /// lane's body is the Euler update.
    pub fn drive(&self, pass: &ForwardPass, body: impl Fn(&Tensor) + 'static) {
        // `Channel` is a Copy handle; these are the epilogue closure's own
        // copies of the same three cells.
        let (step, timestep, ts) = (self.step, self.timestep, self.ts);
        pass.epilogue(move || {
            let k = step.take();
            body(&k);
            let next = &k + 1u32;
            timestep.take();
            timestep.put(at(&ts.read(), &next));
            step.put(&next);
        });
    }
}

/// **ONE DENOISE LOOP: ONE PIPELINE PER LANE, ONE CLOCK PER LANE.**
///
/// A denoise step is one fire carrying every lane of one request (D2), and a
/// group composes only when its passes are members of that one step: the
/// scheduler never seats two passes of one pipeline in one step, so `n` lanes
/// down one pipeline are `n` fires, each attending alone. `n` pipelines with
/// one pass each, submitted back to back, are what the wait-all seal composes
/// into one fire — and the runtime holds a fresh group's first frame for the
/// cohort its passes declare, so the first step needs no priming.
///
/// This owns those pipelines and the per-lane clocks and nothing else: the
/// passes, their ports and their epilogue arithmetic stay the guest's.
///
/// ```ignore
/// let mut loops = DenoiseLoop::new(&sched);
/// let text_clock = loops.lane("text");
/// let image_clock = loops.lane("image");
/// // ... build `text` and `image` passes, binding `clock.timestep` ...
/// text_clock.drive(&text, |_| {});
/// image_clock.drive(&image, move |k| {
///     seed_or_step(k, &x, &intrinsics::velocity(w), &dts, &rng, shape, Some(&out));
/// });
/// for _ in 0..loops.fires() {
///     loops.fire(&[&text, &image])?;
/// }
/// loops.close();
/// ```
pub struct DenoiseLoop {
    pipes: Vec<Pipeline>,
    /// Fire `k`'s Euler `dt`: 0 on the seed fire, one trailing 0 past the
    /// end. The integrating lane gathers this by fire index.
    dts: Vec<f32>,
    ts: Vec<f32>,
    fires: u32,
}

impl DenoiseLoop {
    /// The loop `sched` describes, with a seed fire ahead of its steps:
    /// `sched.steps() + 1` fires in all.
    #[must_use]
    pub fn new(sched: &FlowMatchEuler) -> DenoiseLoop {
        let mut ts = vec![sched.timestep(0)];
        ts.extend(sched.timesteps());
        ts.push(0.0);
        let mut dts = vec![0.0];
        dts.extend(sched.dts());
        dts.push(0.0);
        DenoiseLoop {
            pipes: Vec::new(),
            dts,
            ts,
            fires: sched.steps() + 1,
        }
    }

    /// How many fires the loop runs: one seed plus one per step.
    #[must_use]
    pub fn fires(&self) -> u32 {
        self.fires
    }

    /// Add a lane, in the order its pass is submitted, and hand back its
    /// clock. `tag` names the lane's channels in a trace.
    pub fn lane(&mut self, tag: &str) -> LaneClock {
        self.pipes.push(Pipeline::new());
        LaneClock {
            step: Channel::from([0u32]).named(&format!("{tag}_step")),
            timestep: Channel::from([self.ts[0]]).named(&format!("{tag}_t")),
            ts: Channel::from(self.ts.clone()).named(&format!("{tag}_ts")),
        }
    }

    /// A `[fires + 1] f32` channel of every fire's Euler `dt`, for the lane
    /// that integrates. One per epilogue that gathers it, for the same
    /// channel-role reason [`LaneClock`] states.
    #[must_use]
    pub fn dts(&self, tag: &str) -> Channel {
        Channel::from(self.dts.clone()).named(&format!("{tag}_dts"))
    }

    /// Submit one fire: `passes[i]` rides lane `i`'s pipeline, in the order
    /// the lanes were added. Refuses a count that is not the lane count — a
    /// missing lane is a group that never composes.
    pub fn fire(&self, passes: &[&ForwardPass]) -> Result<(), String> {
        if passes.len() != self.pipes.len() {
            return Err(format!(
                "this loop has {} lanes and {} passes were submitted; every lane of a group must \
                 fire in the same step",
                self.pipes.len(),
                passes.len()
            ));
        }
        for (i, pass) in passes.iter().enumerate() {
            pass.submit(&self.pipes[i])
                .map_err(|why| format!("lane {i}: {why}"))?;
        }
        Ok(())
    }

    /// Close every lane's pipeline.
    pub fn close(&self) {
        for pipe in &self.pipes {
            pipe.close();
        }
    }
}

/// The Euler epilogue body of an image lane: on fire 0 SEED the latent from
/// the keyed RNG state (the model's velocity over the empty cell is
/// discarded), on every later fire integrate `x ← x + dt(k)·v`, and — when
/// `out` is given — publish the result for the host to read.
///
/// `velocity` is what this fire predicts: the lane's own
/// `intrinsics::velocity(width)`, or a CFG-combined one.
pub fn seed_or_step(
    k: &Tensor,
    x: &Channel,
    velocity: &Tensor,
    dts: &Channel,
    rng: &Channel,
    shape: [u32; 2],
    out: Option<&Channel>,
) {
    let current = x.take();
    let stepped = euler_step(&current, velocity, &at(&dts.read(), k));
    let state = rng.take();
    let fresh = noise(shape, &state);
    let first = broadcast(reshape(eq(k, 0u32), [1, 1]), shape);
    let next = select(&first, &fresh, &stepped);
    x.put(&next);
    if let Some(out) = out {
        out.put(&next);
    }
    rng.put(&state + &Tensor::constant([0u32, 1u32]));
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
    if !fact.takes_tokens {
        return Err(format!(
            "reading `{reading}` is not a text encoder (it embeds no tokens)"
        ));
    }
    // THE TEMPLATE IS THE FAMILY'S, AND THE `chat` SURFACE IS WHERE IT
    // LIVES. Every text encoder in the generative zoo is an instruct model
    // whose reference pipeline renders one user turn plus the generation
    // cue (`apply_chat_template(add_generation_prompt=True)`) before it
    // tokenizes — FLUX.2 klein's Qwen3 with the empty think block,
    // Z-Image's Qwen3 without it. A guest that hands the bare prompt gets
    // an embedding the DiT was never conditioned on, so the template is not
    // optional and it is not the guest's to spell: `first_user` + `cue` are
    // the bound model's own rendering. Padding stays the family's (the
    // encode arm's), so nothing is added here.
    let mut ids = crate::chat::first_user(prompt);
    ids.extend(crate::chat::cue());
    if ids.is_empty() {
        return Err("the prompt tokenizes to nothing".to_string());
    }
    encode_ids(&ids, reading).await
}

/// The other door into a text reading: the ids themselves.
///
/// [`encode_text`] renders the bound model's template and tokenizes with
/// the bound model's tokenizer, which is the right thing whenever the
/// artifact carries the encoder's own vocabulary. Some do not — Wan 2.2's
/// umT5 is a SentencePiece **Unigram** model and `crates/tokenizer`
/// compiles BPE pipelines alone (`models::wan_2::tokenizer`), so that row's
/// artifact carries somebody else's vocabulary and `encode_text` on it
/// would condition the DiT on ids it has never seen. A caller that HAS the
/// right ids hands them over here instead, and the encoder still runs
/// inside pie.
///
/// **A CACHELESS ENCODER BINDS NO ATTENTION.** `has_kv` is the fact that
/// says which: a causal encoder (Qwen3 behind FLUX.2 and Z-Image) needs a
/// working set and a `KvGeometry`; a bidirectional one (umT5) declares
/// `has_kv: false`, which REFUSES `attention` on its pass by name, and its
/// rows attend each other inside the arm over the lane's own indptr. Both
/// land `hidden()` at every row.
pub async fn encode_ids(ids: &[u32], reading: &str) -> Result<Channel, String> {
    let (rows, width) = encode_ids_rows(ids, reading).await?;
    let len = u32::try_from(rows.len() / width.max(1) as usize).unwrap_or(0);
    Ok(Channel::from_shaped([len, width], rows))
}

/// [`encode_ids`] one step lower: the encoder's rows as host values plus
/// their width, before they are put on a channel.
///
/// A guest that must RESHAPE the rows before the denoiser sees them needs
/// this: a family whose context lane is a fixed height
/// (`PortFact::rows` — Wan 2.2's 512) has the guest zero-pad the encoder's
/// answer, and building a channel only to take it apart again would cross
/// the host twice.
pub async fn encode_ids_rows(ids: &[u32], reading: &str) -> Result<(Vec<f32>, u32), String> {
    let Some(fact) = crate::model::reading(reading) else {
        return Err(format!("this model declares no reading `{reading}`"));
    };
    if !fact.takes_tokens {
        return Err(format!(
            "reading `{reading}` embeds no tokens, so there is nothing to hand ids to"
        ));
    }
    if fact.readout != ReadoutKind::Hidden {
        return Err(format!(
            "reading `{reading}` reads out {:?}, not the hidden rows an encoder hands over",
            fact.readout
        ));
    }
    let width = fact.readout_width;
    let cap = crate::eta::max_embed_length().max(1);
    let ids: Vec<i32> = ids
        .iter()
        .take(cap)
        .map(|id| i32::try_from(*id).unwrap_or(0))
        .collect();
    let len = u32::try_from(ids.len()).map_err(|_| "the prompt is too long")?;
    if len == 0 {
        return Err("no ids to encode".to_string());
    }
    let pipe = Pipeline::new();
    let toks = Channel::from(ids);
    let embed_indptr = Channel::from([0u32, len]);
    let readout = Channel::from_iter(0..len);
    let out = Channel::new([len, width], crate::eta::Dtype::F32);

    let pass = crate::eta::attention::ForwardPass::new();
    pass.reading(reading)?;
    pass.embed(&toks, &embed_indptr)?;
    pass.readout(&readout)?;
    // Held past the `if` so the working set outlives the submit on the
    // cached path; a cacheless reading reserves no pages at all.
    let ws = WorkingSet::new();
    if fact.has_kv {
        let page_size = crate::eta::kv_page_size().max(1);
        let pages = len.div_ceil(page_size);
        ws.reserve(pages)?;
        let positions = Channel::from_iter(0..len);
        let page_ids = Channel::from_iter(0..pages);
        let page_indptr = Channel::from([0u32, pages]);
        let w_slot = Channel::from_iter((0..len).map(|p| p / page_size));
        let w_off = Channel::from_iter((0..len).map(|p| p % page_size));
        let kv_len = Channel::from([len]);
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
    }
    let readback = out;
    pass.epilogue(move || {
        readback.put(intrinsics::hidden(width));
    });
    pass.submit(&pipe)?;
    let rows: Vec<f32> = out.take_host().await?;
    pipe.close();
    Ok((rows, width))
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
            stream_shifts: Vec::new(),
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
            stream_shifts: Vec::new(),
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
            stream_shifts: Vec::new(),
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

    /// FLUX.2's `(T, H, W, L)`: the text rows number the fourth axis, the
    /// image grid rides `(h, w)` at `T = 0` and `L = 0`.
    #[test]
    fn a_grid_then_index_convention_numbers_text_on_its_own_axis() {
        let flux = PositionConvention {
            axes: vec![
                AxisRole::Time,
                AxisRole::Height,
                AxisRole::Width,
                AxisRole::Index,
            ],
            text_axis: 3,
            text_origin: 0,
            image_follows_text: false,
        };
        assert_eq!(
            positions_for(&flux, LaneRows::Sequence(2), 2),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            positions_for(&flux, LaneRows::Grid { h: 2, w: 2 }, 2),
            vec![
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 0.0,
            ]
        );
    }

    /// Z-Image's `(t, h, w)`: caption row `j` at `(1 + j, 0, 0)` and the
    /// image behind it at `(text_rows + 1, a, b)`.
    #[test]
    fn a_text_time_prefix_convention_stacks_the_image_behind_the_caption() {
        let z = PositionConvention {
            axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
            text_axis: 0,
            text_origin: 1,
            image_follows_text: true,
        };
        assert_eq!(
            positions_for(&z, LaneRows::Sequence(2), 2),
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0]
        );
        assert_eq!(
            positions_for(&z, LaneRows::Grid { h: 1, w: 2 }, 32),
            vec![33.0, 0.0, 0.0, 33.0, 0.0, 1.0]
        );
    }

    /// mini-dit's `(t, h, w)`: caption row `j` at `(j, 0, 0)`, image patch
    /// `(h, w)` at `(0, h, w)` — the same builder, no offsets.
    #[test]
    fn a_grid_only_convention_leaves_the_time_axis_at_zero() {
        let mini = PositionConvention {
            axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
            text_axis: 0,
            text_origin: 0,
            image_follows_text: false,
        };
        assert_eq!(
            positions_for(&mini, LaneRows::Sequence(2), 2),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            positions_for(&mini, LaneRows::Grid { h: 2, w: 1 }, 2),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
    }
}
