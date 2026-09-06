//! Frame scheduling: seals a frame only once every awaited lane's next
//! frame (k waves submitted as one unit) is fully submitted, then dispatches
//! its waves in slot order.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::stats::SchedulerStats;
use crate::scheduler::ProcessId;

/// Default dispatch depth (`[model.scheduler] frame_dispatch_depth`): how
/// far ahead of the device the runtime runs; also sizes the engine's staging
/// ring (`engine::runahead::Runahead`). Validated with `frame_size` by
/// `RuntimeConfig::validate()`.
const DEFAULT_DISPATCH_DEPTH: usize = 2;

pub(super) fn configured_dispatch_depth() -> usize {
    match DISPATCH_DEPTH.load(Ordering::Relaxed) {
        0 => DEFAULT_DISPATCH_DEPTH,
        depth => depth,
    }
}

/// Install the configured dispatch depth at bootstrap.
pub(crate) fn set_dispatch_depth(depth: usize) {
    DISPATCH_DEPTH.store(depth, Ordering::Relaxed);
}

/// `0` = never installed; see `crate::scheduler::reconfigure`.
static DISPATCH_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// `PIE_SEAL_MODE=strict` (default): hold the boundary for every awaited
/// lane. `PIE_SEAL_MODE=ready`: when idle, open it with whichever lanes are
/// arrival-complete instead; late lanes join later partitions.
/// Threshold for the `[idle-gap]` dump, in microseconds. `u64::MAX` (never)
/// unless `PIE_IDLE_DUMP_US` names one.
/// Seat floor for the `[device-idle]` dump. `0` unless
/// `PIE_IDLE_DUMP_MIN_SEATS` names one.
fn idle_dump_min_seats() -> usize {
    static SEATS: OnceLock<usize> = OnceLock::new();
    *SEATS.get_or_init(|| {
        std::env::var("PIE_IDLE_DUMP_MIN_SEATS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

pub(super) fn idle_dump_threshold_us() -> u64 {
    static THRESHOLD: OnceLock<u64> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("PIE_IDLE_DUMP_US")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(u64::MAX)
    })
}

/// **THE SEAL MODE'S DEFAULT IS STRICT — WAIT-ALL — ON EVERY PLATFORM.**
/// Ready mode was measured on CUDA at ~240 run-ahead lanes, where it opened
/// the boundary earlier without narrowing the batch (+1-2%), and taken as
/// the CUDA default on that. Metal at eight lanes showed the opposite (a
/// fire is 12-66 ms and a lane's host round trip is a visible fraction of
/// that, so the arrival-complete subset was two or three lanes; strict read
/// 1.7-2.3x). Then CUDA at 64 HOST-DRIVEN lanes (text-completion: one host
/// round trip per token, no submit-ahead) showed the same failure in a
/// steadier form: ready mode opens the boundary the instant the device is
/// free, 50-300 µs before the lanes it just answered have re-submitted, so
/// the population settles into two cohorts (41/23, 45/19) that fire
/// ALTERNATELY — every fire half-width, every lane a token per two device
/// steps. Strict wait-all read 1690 -> 3000 tok/s there, and the run-ahead
/// program 2690 -> 2990, at the same 19.5 ms per 64-wide fire. Strict is
/// the principle and now the default; `PIE_SEAL_MODE=ready` opts back in.
static SEAL_DEFAULT_READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_seal_default_ready(ready: bool) {
    SEAL_DEFAULT_READY.store(ready, Ordering::Relaxed);
}

/// **HOW LONG A READY-MODE BOUNDARY WAITS FOR THE REST BEFORE IT OPENS
/// WITH WHO IS THERE.** Ready mode opens the boundary from the
/// arrival-complete lanes after ONE quiet poll (`GATHER_POLL_US`); on a Metal
/// box with eight lanes whose readbacks return a fraction of a millisecond
/// apart that opened it with two or three of them every time (2.2 lanes a
/// batch of eight offered), and strict sealing read 1.7-2.3x. A window says:
/// once a candidate exists, keep gathering until the boundary has been open
/// this long OR every awaited lane arrived — so lanes that land within the
/// window batch as strict would, and a lane that stalls holds the others for
/// the window, not the 50 ms leash. Zero is ready mode as it was.
/// `PIE_SEAL_COALESCE_US` overrides the platform's default
/// (`set_seal_coalesce_default`).
static SEAL_COALESCE_DEFAULT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn set_seal_coalesce_default(window: Duration) {
    SEAL_COALESCE_DEFAULT_US.store(window.as_micros() as u64, Ordering::Relaxed);
}

fn seal_coalesce() -> Duration {
    static CONFIGURED: OnceLock<Duration> = OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        std::env::var("PIE_SEAL_COALESCE_US")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .map_or_else(
                || Duration::from_micros(SEAL_COALESCE_DEFAULT_US.load(Ordering::Relaxed)),
                Duration::from_micros,
            )
    })
}

fn seal_mode_ready() -> bool {
    static CONFIGURED: OnceLock<bool> = OnceLock::new();
    *CONFIGURED.get_or_init(|| match std::env::var("PIE_SEAL_MODE") {
        Ok(value) => match value.trim() {
            "strict" => false,
            "ready" => true,
            // An unrecognised value defaults to ready, not strict.
            other => {
                tracing::warn!(
                    value = other,
                    "PIE_SEAL_MODE must be \"ready\" or \"strict\"; \
                     ignoring and keeping the default (ready)"
                );
                true
            }
        },
        Err(_) => SEAL_DEFAULT_READY.load(Ordering::Relaxed),
    })
}

/// Default OFF — the strict wait-all rule; `PIE_GATE_CONTRIBUTED=1` turns
/// the relaxation on. On, while a frame executes a lane the runtime owes a
/// result with nothing else queued is not counted missing (see
/// [`LaneState::gate_verdict`]), so the gate "holds" the moment every
/// OTHER lane has re-submitted. For host-driven lanes that is a permanent
/// split into two alternating half-width cohorts (see
/// [`SEAL_DEFAULT_READY`]); a run-ahead lane in flight has already
/// submitted its next frame and never needed the relaxation.
fn gate_contributed() -> bool {
    static CONFIGURED: OnceLock<bool> = OnceLock::new();
    *CONFIGURED
        .get_or_init(|| std::env::var("PIE_GATE_CONTRIBUTED").is_ok_and(|value| value == "1"))
}

/// The frame identity one fire carries from `forward.submit`: which lane
/// (pipeline scope), which frame of that lane, which wave slot, and how many
/// fires the whole frame holds (so arrival completeness is decidable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStamp {
    pub lane: ProcessId,
    pub seq: u64,
    pub slot: u32,
    pub fires: u32,
}

/// Liveness watchdog for a blocked gather. Report-only: never removes a
/// member and never fires a narrow epoch — an unresponsive lane leaves only
/// through close/terminate.
const STRICT_WATCHDOG_US: u64 = 1_000_000;

/// How long a blocked gather waits before looking again. Distinct from the
/// watchdog above, which is a reporting cadence, not a wake interval.
const GATHER_POLL_US: u64 = 500;

struct ArrivedFire {
    slot: u32,
    /// `None` when the fire was rejected at scheduler admission — it counts
    /// toward arrival completeness but never dispatches.
    fire_id: Option<u64>,
    tokens: usize,
    rows: usize,
}

/// One live fire measured into a wave while a partition is assembled
/// ([`FramePolicy::seal`]) — the arrival's fields with its wave resolved.
struct PlacedFire {
    wave: usize,
    fire_id: u64,
    rows: usize,
    tokens: usize,
}

struct PendingFrame {
    seq: u64,
    /// Fires this frame is declared to hold; lowered on a mid-frame submit
    /// failure via [`FramePolicy::on_frame_truncated`].
    expected: u32,
    /// Cut short at what had arrived, so it is sealable as-is. Once set,
    /// later arrivals keep the frame self-completing instead of re-raising
    /// `expected`.
    truncated: bool,
    /// `forward.park()`: a wait-set exit ordered like a submit. Carries no
    /// fires and is complete on arrival, landing once every frame the guest
    /// submitted before it has sealed.
    park: bool,
    fires: Vec<ArrivedFire>,
    /// When this frame first became complete. Diagnostic only.
    complete_at: Option<Instant>,
}

impl PendingFrame {
    fn is_complete(&self) -> bool {
        self.park || self.fires.len() >= self.expected as usize
    }
}

struct LaneState {
    owner: Option<ProcessId>,
    /// Wait-set membership: joined on the lane's first stamped fire,
    /// released by close/terminate, park, or the leash below.
    awaited: bool,
    /// Whether this lane has ever been sealed into a partition. Diagnostic
    /// only.
    served: bool,
    /// When this lane's last frame retired. Diagnostic only.
    retired_at: Option<Instant>,
    /// Left the wait-set through `forward.park()`. Kept distinct from a
    /// cleared `awaited` so rejoin can be implicit (next accepted fire).
    parked: bool,
    /// Parked by the leash (silence timeout) rather than the guest. Rejoin
    /// is identical to a voluntary park, but the silence clock keeps running.
    leashed: bool,
    /// Start of this lane's silence (first seen blocking with nothing
    /// owed). Cleared by any accepted fire or debt owed. `None` = not timed.
    clock_from: Option<Instant>,
    frames: VecDeque<PendingFrame>,
    /// **THE ATTENTION GROUP THIS LANE JOINS** (design D2), as its fires
    /// stated it. Owner-scoped: the key is `(owner, group)`. `None` for a
    /// lane in no group — the overwhelming majority, and the reason the
    /// group rules below cost an ungrouped fleet one `Option` test.
    ///
    /// Set by any fire that states a group and never cleared, because the
    /// two mistakes are not symmetric: a stale membership costs a hold that
    /// the silence timeout bounds, while forgetting one costs a fire that
    /// attends half its group and answers plausibly. A pipeline is a lane
    /// of one group for its life anyway — `latent::DenoiseLoop` owns one
    /// pipeline per lane and closes it, which purges the lane.
    group: Option<u32>,
    /// Fired into the current boundary, or written off by
    /// [`FramePolicy::close_boundary`]. Cleared on every new boundary; a
    /// lane joining mid-boundary starts fired (waits for the next gate).
    fired_this_boundary: bool,
}

/// One attention group's first-frame gathering: which pipelines have
/// arrived under it, out of how many its passes declared.
#[derive(Debug)]
struct Cohort {
    expected: u32,
    arrived: BTreeSet<ProcessId>,
    since: Instant,
    /// Already reported as breached, so the verdict is handed to the worker
    /// once. The entry stays until the process's leave purges it — it must
    /// keep holding its lanes out of every seal while the kill is in
    /// flight, or the partial group would fire on the way out.
    condemned: bool,
}

impl Cohort {
    fn complete(&self) -> bool {
        self.arrived.len() as u64 >= u64::from(self.expected)
    }
}

/// Why the policy is asking the worker to end a process. The worker turns
/// this into the guest-visible error, so it says plainly what the guest did
/// wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Doom {
    /// Silent past the silence timeout with nothing owed and no
    /// `forward.park()`.
    Abandoned,
    /// A STATED attention-group cohort (design D2) that never completed: a
    /// pass of the group never reached the runtime on a lane of its own, so
    /// the group could never compose. See [`FramePolicy::cohort_verdict`].
    CohortNeverComplete {
        group: u32,
        expected: u32,
        arrived: u32,
    },
}

impl std::fmt::Display for Doom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Abandoned => f.write_str(
                "pipeline abandoned: silent past the silence timeout without submitting \
                 and without calling forward.park()",
            ),
            Self::CohortNeverComplete {
                group,
                expected,
                arrived,
            } => write!(
                f,
                "attention group {group} never composed: {expected} live forward passes name \
                 the group and only {arrived} of them reached the runtime on a lane of its \
                 own before the silence timeout. A group attends inside ONE fire, so the \
                 runtime will not fire it short — every forward pass that names a group must \
                 submit into the group's next frame, each down a pipeline of its own (a \
                 pipeline is serial, so two of a group's passes submitted to ONE pipeline \
                 arrive as one lane and count once; a pass that means to sit out must not \
                 name the group, and dropping it is enough to leave)"
            ),
        }
    }
}

/// The wait-all gate's verdict for one lane — [`LaneState::gate_verdict`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateVerdict {
    /// Not a member, or its next frame is fully submitted: the gate is free
    /// to seal as far as this lane is concerned.
    Satisfied,
    /// A member whose next frame is not fully submitted. The boundary holds.
    Blocking,
    /// A member with nothing queued that the runtime still owes a result, so
    /// its next frame cannot exist until the device answers (see
    /// [`gate_contributed`]). Not counted missing.
    SkippedOwed,
}

impl LaneState {
    /// The attention group this lane joins, owner-scoped — the key both
    /// [`FramePolicy::cohorts`] and the group rules in `seal` use. Group
    /// ids are per-process, so an unowned lane joins nothing.
    fn group_key(&self) -> Option<(ProcessId, u32)> {
        self.owner.zip(self.group)
    }

    /// Whether this lane still owes its attention group a fire: it is a
    /// member (awaited, or dropped from the boundary by the submit leash
    /// and so still expected) whose next frame is not fully submitted. A
    /// lane that called `forward.park()` said it is sitting out and owes
    /// nothing.
    fn owes_its_group(&self) -> bool {
        (self.awaited || self.leashed)
            && !self.frames.front().is_some_and(PendingFrame::is_complete)
    }

    /// Does this lane deny the wait-all gate? Blocks while a member whose
    /// next frame is not complete. `relax` is [`gate_contributed`] AND the
    /// device busy; `owes` is the runtime's debt to this lane.
    fn gate_verdict(&self, relax: bool, owes: bool) -> GateVerdict {
        if !self.awaited || self.frames.front().is_some_and(PendingFrame::is_complete) {
            return GateVerdict::Satisfied;
        }
        // Already in-flight with nothing queued: waiting for the device,
        // not the guest.
        if relax && owes && self.frames.is_empty() {
            return GateVerdict::SkippedOwed;
        }
        GateVerdict::Blocking
    }
}

/// One sealed (immutable) frame, dispatched whole: per-wave fire-id lists
/// in slot order. Leaves this queue once posted; the policy tracks nothing
/// after.
struct SealedFrame {
    waves: Vec<Vec<u64>>,
    /// Member lanes — read by the dispatch gate to hold a frame whose lane
    /// has a pre-launch copy barrier queued.
    members: BTreeSet<ProcessId>,
}

/// What the worker should do next for frame-managed launches.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum FramePlan {
    /// Post this whole sealed frame now: per-wave fire ids in slot order.
    /// The frame has left the policy — the worker owns it from here.
    Dispatch(Vec<Vec<u64>>),
    /// Nothing dispatchable yet; re-decide after the bootstrap hold or at
    /// the blocked-gather watchdog deadline.
    Hold(Duration),
    /// No sealed work and no seal candidates: park until an arrival.
    Park,
    /// These processes cannot be served any longer: each is named with the
    /// reason the guest is told. The worker terminates them.
    Terminate(Vec<(ProcessId, Doom)>),
}

/// The stamped fire ids still in the worker queue, sorted (sorted-push +
/// binary search; ids arrive near-ascending, so the sort is usually a no-op).
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct QueuedFireIds {
    ids: Vec<u64>,
}

impl QueuedFireIds {
    pub fn clear(&mut self) {
        self.ids.clear();
    }

    pub fn push(&mut self, fire_id: u64) {
        self.ids.push(fire_id);
    }

    /// Restore the sorted invariant `contains` relies on. Retries and
    /// re-admissions can put a lower id behind a higher one, so this cannot
    /// assume queue order.
    pub fn seal(&mut self) {
        if !self.ids.windows(2).all(|pair| pair[0] <= pair[1]) {
            self.ids.sort_unstable();
        }
    }

    pub fn contains(&self, fire_id: &u64) -> bool {
        self.ids.binary_search(fire_id).is_ok()
    }
}

impl FromIterator<u64> for QueuedFireIds {
    fn from_iter<T: IntoIterator<Item = u64>>(iter: T) -> Self {
        let mut out = Self {
            ids: iter.into_iter().collect(),
        };
        out.seal();
        out
    }
}

pub(super) struct FramePolicy {
    k: usize,
    max_wave_tokens: usize,
    max_wave_rows: usize,
    /// The wait-set plus each lane's queued frames. BTreeMap for
    /// deterministic admission order.
    lanes: BTreeMap<ProcessId, LaneState>,
    sealed: VecDeque<SealedFrame>,
    /// The worker's in-flight signal as of the current `plan_dispatch` call.
    /// Diagnostic only.
    executing_now: bool,
    /// When the most recent frame retired. Diagnostic only.
    last_retire_at: Option<Instant>,
    /// Whether the current device-idle episode has already been dumped, so
    /// `PIE_IDLE_DUMP_US` prints one census per episode. Diagnostic only.
    idle_dumped: bool,
    /// Bind controls accepted but not yet completed. Binds don't hold the
    /// seal — a live rebinder is covered by its own lane membership.
    pending_binds: BTreeMap<ProcessId, usize>,
    /// **ATTENTION-GROUP COHORTS STILL GATHERING** (design D2), keyed by
    /// `(owner, group)`. A group composes only when its passes are members
    /// of one step, and the gate can await only pipelines it has seen: the
    /// first lane of a fresh group would seal alone. So an arrival under a
    /// group holds ITS OWN LANES out of every seal (see
    /// [`FramePolicy::group_short`]) until the cohort its request declares
    /// has arrived — never the gate, so a slow sibling costs its own
    /// request and not the fleet, and never a partial fire, so the stated
    /// group is kept whole or the request dies by name
    /// ([`FramePolicy::cohort_verdict`]). A cohort is dropped the moment it
    /// completes, so this map is empty except between a group's first and
    /// last submit of one frame; in that window `group_short` and wait-all
    /// say the same thing, and outside it the map costs nothing.
    cohorts: BTreeMap<(ProcessId, u32), Cohort>,
    /// Successor pool: bring-up processes whose lane hasn't fired yet.
    /// While `pending_slots > 0`, one of these is about to take a slot,
    /// opening the cohort-boundary window.
    staged: BTreeSet<ProcessId>,
    /// Released execution slots not yet re-consumed: +1 on release,
    /// saturating -1 on consume. Positive with a non-empty `staged` pool
    /// means a successor's admission is imminent.
    pending_slots: u64,
    /// Identity-paired in-flight joins: a parked process that acquired its
    /// execution permit but whose first stamped fire has not arrived yet.
    joins_in_flight: BTreeSet<ProcessId>,
    /// Processes blocked on an execution permit, in the order the FIFO-fair
    /// semaphore will hand them one.
    admission_queue: VecDeque<ProcessId>,
    /// Processes that consumed an execution slot and still hold it.
    slotted: BTreeSet<ProcessId>,
    /// Slot holders between their Terminate leave and their release
    /// broadcast; without this the seal could close on a partial cohort.
    departing: BTreeSet<ProcessId>,
    /// Processes the planner has suspended and not yet resumed. While
    /// marked here their lanes never (re-)enter the wait-set.
    suspended: BTreeSet<ProcessId>,
    /// Per-lane: the frame seq cut short when it last left the wait-set
    /// mid-frame, so a late slot from that seq isn't re-formed. Kept outside
    /// `lanes` since a truncated lane is usually dropped first.
    truncated_seqs: BTreeMap<ProcessId, u64>,
    /// Liveness-only deadline for the current blocked-gather episode.
    strict_watchdog_deadline: Option<Instant>,
    /// Count of accepted fires landed in a lane's front frame — the
    /// ready-mode quiescence signal: the gate opens early only after one
    /// full hold cycle in which this didn't move.
    gather_seq: u64,
    /// `gather_seq` at the last held gate evaluation with a seal
    /// candidate. Matching `gather_seq` at the next evaluation means
    /// nothing landed since: quiesced.
    quiesce_mark: Option<u64>,
    /// Lanes with fires dispatched but not yet retired: the runtime owes
    /// them a result, so the submit deadline must not run against them.
    in_flight_lanes: BTreeSet<ProcessId>,
    /// When each in-flight lane's debt was incurred, so the debt itself gets
    /// a deadline even if the frame never retires.
    in_flight_since: BTreeMap<ProcessId, Instant>,
    /// How long a silent member may hold the boundary before the leash
    /// drops it (not a failure — its next fire rejoins). See
    /// [`crate::scheduler::configured_submit_deadline`].
    submit_deadline: Duration,
    /// How long a lane may stay silent in total before the process is
    /// terminated. A pipeline that intends to stop must call
    /// `forward.park()`, which is never killed.
    silence_timeout: Duration,
    /// [`gate_contributed`], read once so the lever is fixed for the
    /// policy's life. `false` is the strict wait-all rule.
    gate_contributed: bool,
    /// [`seal_mode_ready`], read once, for the same reason. `false` is the
    /// strict wait-all rule.
    seal_mode_ready: bool,
    /// [`seal_coalesce`]: how long a ready-mode boundary with a candidate
    /// keeps gathering before it opens with who is there.
    seal_coalesce: Duration,
    /// When the current ready-mode hold began — the first evaluation that
    /// found a candidate and a missing lane with the device idle. Cleared
    /// with `quiesce_mark`.
    coalesce_since: Option<Instant>,
    /// Probe sink (`profile-fire` wave counters); `None` in unit tests.
    stats: Option<Arc<SchedulerStats>>,
}

impl FramePolicy {
    /// Override the submit deadline. Tests only.
    #[cfg(test)]
    fn with_submit_deadline(mut self, deadline: Duration) -> Self {
        self.submit_deadline = deadline;
        // Out of the way unless a test asks for it, to avoid the leash
        // firing on the same driven clock.
        self.silence_timeout = Duration::from_secs(86_400);
        self
    }

    /// Override the silence timeout — the VERDICT clock. Tests only.
    #[cfg(test)]
    fn with_silence_timeout(mut self, timeout: Duration) -> Self {
        self.silence_timeout = timeout;
        self
    }

    /// Pin the seal mode for one policy. Tests only.
    #[cfg(test)]
    fn with_seal_mode_ready(mut self, on: bool) -> Self {
        self.seal_mode_ready = on;
        self
    }

    /// Pin the coalescing window for one policy. Tests only.
    #[cfg(test)]
    fn with_seal_coalesce(mut self, window: Duration) -> Self {
        self.seal_coalesce = window;
        self
    }

    pub fn new(
        k: usize,
        max_wave_rows: usize,
        max_wave_tokens: usize,
        stats: Option<Arc<SchedulerStats>>,
    ) -> Self {
        Self {
            k,
            max_wave_tokens,
            max_wave_rows,
            lanes: BTreeMap::new(),
            sealed: VecDeque::new(),
            executing_now: false,
            last_retire_at: None,
            idle_dumped: false,
            pending_binds: BTreeMap::new(),
            cohorts: BTreeMap::new(),
            staged: BTreeSet::new(),
            pending_slots: 0,
            joins_in_flight: BTreeSet::new(),
            admission_queue: VecDeque::new(),
            slotted: BTreeSet::new(),
            departing: BTreeSet::new(),
            suspended: BTreeSet::new(),
            truncated_seqs: BTreeMap::new(),
            strict_watchdog_deadline: None,
            gather_seq: 0,
            quiesce_mark: None,
            in_flight_lanes: BTreeSet::new(),
            in_flight_since: BTreeMap::new(),
            submit_deadline: crate::scheduler::configured_submit_deadline(),
            silence_timeout: crate::scheduler::configured_silence_timeout(),
            gate_contributed: gate_contributed(),
            seal_mode_ready: seal_mode_ready(),
            seal_coalesce: seal_coalesce(),
            coalesce_since: None,
            stats,
        }
    }

    /// Whether this deployment runs 1-slot frames (k = 1): a frame is a
    /// wave, and the worker synthesizes a per-fire stamp at admission.
    pub fn single_slot(&self) -> bool {
        self.k == 1
    }

    /// A stamped fire was accepted into the scheduler queue.
    pub fn on_fire_enqueued(
        &mut self,
        stamp: FrameStamp,
        owner: Option<ProcessId>,
        fire_id: u64,
        tokens: usize,
        rows: usize,
        cohort: Option<(u32, u32)>,
    ) {
        let accept_began = self.stats.is_some().then(Instant::now);
        self.gather_cohort(stamp.lane, owner, cohort);
        self.record_arrival(
            stamp,
            owner,
            ArrivedFire {
                slot: stamp.slot,
                fire_id: Some(fire_id),
                tokens,
                rows,
            },
        );
        // The group rides every fire of the lane, not just its first: it is
        // what keeps the group whole in steady state, long after the cohort
        // that gathered its first frame was forgotten.
        if let Some((group, _)) = cohort
            && let Some(lane) = self.lanes.get_mut(&stamp.lane)
        {
            lane.group = Some(group);
        }
        if let (Some(began), Some(stats)) = (accept_began, &self.stats) {
            use std::sync::atomic::Ordering::Relaxed;
            stats
                .fire
                .quorum
                .accept_us
                .fetch_add(began.elapsed().as_micros() as u64, Relaxed);
            stats.fire.quorum.accept_calls.fetch_add(1, Relaxed);
        }
    }

    /// A fire rejected at admission still counts toward its frame's arrival
    /// completeness, so the frame can seal with its surviving fires.
    pub fn on_fire_rejected_at_admission(&mut self, stamp: FrameStamp, owner: Option<ProcessId>) {
        self.record_arrival(
            stamp,
            owner,
            ArrivedFire {
                slot: stamp.slot,
                fire_id: None,
                tokens: 0,
                rows: 0,
            },
        );
    }

    /// A lane arrived under attention group `cohort = (group, expected)`:
    /// it joins the group's cohort, which holds the group's own lanes out
    /// of every seal until all `expected` members have arrived.
    ///
    /// **EVERY MEMBER'S ARRIVAL COUNTS, INCLUDING ONE ON A LANE THE GATE
    /// ALREADY KNOWS.** It used to skip a lane already in the wait-set, on
    /// the theory that wait-all covers it and only an unseen pipeline needs
    /// gathering. That made the tally asymmetric: `expected` counts every
    /// pass of the group (see `pipeline::instance::cohort_of`) while
    /// `arrived` counted only members whose pipeline was new to the gate —
    /// so a guest that runs earlier UNGROUPED work down a pipeline and then
    /// reuses it for a group member could never complete its cohort, and
    /// the request died with "stated a cohort of 2 … only 1 submitted".
    /// That is z-image's exact shape (a `refine` fire, then a two-lane
    /// `denoise` group with the image lane back on the refine pipeline),
    /// and reusing a pipeline is not a guest error. Counting every arrival
    /// makes the two halves of the tally the same population, and is order
    /// independent: which member the gate happened to have seen before no
    /// longer decides whether the group can compose.
    ///
    /// The cost is that a cohort now forms on every frame of a group rather
    /// than only its first, and is dropped again the moment it completes.
    /// That is not extra policy: in steady state `group_short` already
    /// holds a group whose members have not all submitted, so the cohort
    /// agrees with the gate instead of duplicating it — and it closes two
    /// holes the first-frame-only scoping left open, a group that grows a
    /// lane after its first composition, and a member that calls
    /// `forward.park()` mid-group (which used to let the rest of the group
    /// fire short and silently attend without it).
    fn gather_cohort(
        &mut self,
        lane: ProcessId,
        owner: Option<ProcessId>,
        cohort: Option<(u32, u32)>,
    ) {
        let (Some(owner), Some((group, expected))) = (owner, cohort) else {
            return;
        };
        let entry = self
            .cohorts
            .entry((owner, group))
            .or_insert_with(|| Cohort {
                expected,
                arrived: BTreeSet::new(),
                since: Instant::now(),
                condemned: false,
            });
        // A later member may count more siblings than the first did (one
        // named the group after the first submitted): the largest claim is
        // the cohort's.
        entry.expected = entry.expected.max(expected);
        entry.arrived.insert(lane);
        if entry.complete() {
            self.cohorts.remove(&(owner, group));
        }
    }

    /// **A STATED COHORT IS A PROMISE THE RUNTIME KEEPS, NOT A PREFERENCE.**
    /// A cohort still short past the SILENCE TIMEOUT ends its process, by
    /// name; it never ends in a fire over whoever showed up.
    ///
    /// It used to end in one. The cohort was leashed by `submit_deadline`
    /// (50 ms) like a silent lane, and expiry dropped the entry and let the
    /// boundary seal — so a 1024² job whose first submit carries ~6 MB of
    /// seeded `context` plus ~2 MB of `latents` sealed the denoise group's
    /// first frame with the IMAGE lane alone, ran an unconditioned denoise,
    /// and reported success (cos 0.535 against the golden, 0.9999 against a
    /// deliberately text-free reference — silently, plausibly wrong).
    ///
    /// The two clocks are not interchangeable, and that is the whole fix:
    /// `submit_deadline` is a LEASH, which drops a slow lane from a
    /// boundary so the fleet keeps moving and costs it only latency;
    /// `silence_timeout` is a VERDICT, generous by construction, and a
    /// verdict is the only thing a stated cohort may end in. A leash may
    /// never be spent on a cohort, because dropping a member of a group
    /// changes the ANSWER, not the schedule. What makes the leash
    /// unnecessary here is that a gathering cohort no longer holds the gate
    /// at all — it holds only its own lanes out of `seal` — so a sibling
    /// that is merely slow costs its own request a few milliseconds and the
    /// fleet nothing.
    fn cohort_verdict(&mut self, now: Instant) -> Vec<(ProcessId, Doom)> {
        if self.cohorts.is_empty() {
            return Vec::new();
        }
        let timeout = self.silence_timeout;
        let mut doomed = Vec::new();
        for ((owner, group), cohort) in self.cohorts.iter_mut() {
            if cohort.condemned || now.saturating_duration_since(cohort.since) < timeout {
                continue;
            }
            cohort.condemned = true;
            doomed.push((
                *owner,
                Doom::CohortNeverComplete {
                    group: *group,
                    expected: cohort.expected,
                    arrived: cohort.arrived.len() as u32,
                },
            ));
        }
        doomed
    }

    /// Whether a stated attention group is still assembling somewhere: a
    /// cohort gathering its first frame, or a lane holding a frame it
    /// cannot seal until its groupmates arrive. The policy must not park in
    /// that state — nothing else would wake it, and the verdict above is
    /// only ever read from an evaluation.
    fn group_assembling(&self) -> bool {
        !self.cohorts.is_empty()
            || self
                .lanes
                .values()
                .any(|lane| !lane.frames.is_empty() && self.group_short(lane))
    }

    /// **WHETHER THIS LANE'S ATTENTION GROUP WOULD FIRE SHORT** (design D2).
    /// A group composes only when its passes are members of ONE step, so a
    /// lane whose group is not all here may not seal — in either of the two
    /// ways a group can be incomplete:
    ///
    /// - its frame's cohort is still gathering (the gate can await only
    ///   pipelines it has seen, and the group's others have not fired
    ///   yet), or
    /// - a sibling the gate does await owes this boundary a fire — INCLUDING
    ///   one the submit leash just dropped from the wait-set, which is
    ///   precisely when a group would otherwise fire half of itself.
    ///
    /// This is the rule that makes a partial group impossible by
    /// construction rather than by timing: it is asked on every seal path
    /// (strict, ready-mode, mid-boundary partition, budget partition), so
    /// there is no route by which some lanes of a group reach a fire
    /// without the rest.
    fn group_short(&self, lane: &LaneState) -> bool {
        let Some(key) = lane.group_key() else {
            return false;
        };
        if self.cohorts.contains_key(&key) {
            return true;
        }
        self.lanes
            .values()
            .any(|other| other.group_key() == Some(key) && other.owes_its_group())
    }

    fn record_arrival(&mut self, stamp: FrameStamp, owner: Option<ProcessId>, fire: ArrivedFire) {
        // Staged -> live promotion, keyed by owner: `staged`/`joins_in_flight`
        // are process-scoped while `stamp.lane` is the pipeline scope id.
        if let Some(owner) = owner {
            self.staged.remove(&owner);
            self.joins_in_flight.remove(&owner);
        }
        // A fire can overtake the planner's suspend broadcast: while the
        // owner is suspended, its lane records fires without joining the
        // wait-set.
        let lane_owner = owner.or_else(|| self.lanes.get(&stamp.lane).and_then(|lane| lane.owner));
        let suspended = lane_owner.is_some_and(|owner| self.suspended.contains(&owner));
        // A slot under a seq this lane already truncated is the tail of an
        // already-sealed frame; a newer seq means the cut is spent.
        let late = match self.truncated_seqs.get(&stamp.lane).copied() {
            Some(seq) if seq == stamp.seq => true,
            Some(seq) if seq < stamp.seq => {
                self.truncated_seqs.remove(&stamp.lane);
                false
            }
            _ => false,
        };
        let lane = self.lanes.entry(stamp.lane).or_insert_with(|| LaneState {
            owner,
            awaited: !suspended,
            served: false,
            retired_at: None,
            // Becomes a member on its first post-restore fire, like any
            // other park-shaped exit's implicit rejoin.
            parked: suspended,
            leashed: false,
            clock_from: None,
            frames: VecDeque::new(),
            group: None,
            // A lane arriving mid-boundary is not a member of it; one
            // arriving between boundaries is picked up by the next gate.
            fired_this_boundary: true,
        });
        if lane.owner.is_none() {
            lane.owner = owner;
        }
        // Implicit rejoin, atomic with the fire: a race with the suspend
        // broadcast must not consume the latch, stranding the lane un-awaited.
        if lane.parked && !suspended {
            lane.parked = false;
            lane.awaited = true;
        }
        // Any accepted arrival is proof of life.
        lane.leashed = false;
        lane.clock_from = None;
        // Is this fire for the frame the gather is actually held by? A lane
        // with no frames yet is about to have this one as its front.
        let for_front_frame = lane
            .frames
            .front()
            .is_none_or(|front| front.seq == stamp.seq);
        let frame = match lane.frames.iter_mut().find(|frame| frame.seq == stamp.seq) {
            Some(frame) => frame,
            None => {
                lane.frames.push_back(PendingFrame {
                    seq: stamp.seq,
                    expected: stamp.fires,
                    truncated: false,
                    park: false,
                    fires: Vec::with_capacity(stamp.fires as usize),
                    complete_at: None,
                });
                lane.frames.back_mut().expect("frame just pushed")
            }
        };
        frame.truncated |= late || suspended;
        frame.fires.push(fire);
        if for_front_frame {
            self.gather_seq = self.gather_seq.wrapping_add(1);
        }
        frame.expected = if frame.truncated {
            frame.fires.len() as u32
        } else {
            frame.expected.max(stamp.fires)
        };
        // Cut below what the guest declared: remember the seq so later
        // slots for an already-sealed frame are recognized as late.
        if frame.complete_at.is_none() && frame.is_complete() {
            frame.complete_at = Some(Instant::now());
        }
        let cut = frame.truncated && frame.expected < stamp.fires;
        if cut {
            self.truncated_seqs.insert(stamp.lane, stamp.seq);
        }
    }

    /// The host failed a frame mid-submit: only `submitted` fires exist.
    pub fn on_frame_truncated(&mut self, lane: ProcessId, seq: u64, submitted: u32) {
        if let Some(lane) = self.lanes.get_mut(&lane)
            && let Some(frame) = lane.frames.iter_mut().find(|frame| frame.seq == seq)
        {
            frame.expected = submitted;
            frame.truncated = true;
        }
    }

    /// `forward.park()`: the guest won't submit again until it fires, so the
    /// seal stops waiting for it. Treated as a submit (ordered, not queued
    /// behind the held dispatch); a lane with nothing queued parks immediately.
    pub fn on_lane_park(&mut self, lane: ProcessId, seq: u64) {
        let Some(state) = self.lanes.get_mut(&lane) else {
            // Never fired: no wait-set membership to release.
            return;
        };
        if state.frames.iter().any(|frame| frame.park) {
            // Parking twice with no fire in between is a no-op.
            return;
        }
        // Ordered by seq, not appended: a park can be dequeued after a fire
        // the guest submitted later.
        let at = state
            .frames
            .iter()
            .position(|frame| frame.seq > seq)
            .unwrap_or(state.frames.len());
        state.frames.insert(
            at,
            PendingFrame {
                seq,
                expected: 0,
                truncated: true,
                park: true,
                fires: Vec::new(),
                complete_at: Some(Instant::now()),
            },
        );
    }

    /// A dispatched batch settled: the lanes' submit deadline restarts from
    /// now with the full budget, since the guest must never be charged for
    /// the wave it was waiting on. Clears rather than decrements, since a
    /// frame can retire in more than one batch.
    pub fn on_frame_retired(&mut self, lanes: impl IntoIterator<Item = ProcessId>) {
        let now = Instant::now();
        self.last_retire_at = Some(now);
        for lane in lanes {
            self.in_flight_lanes.remove(&lane);
            self.in_flight_since.remove(&lane);
            if let Some(state) = self.lanes.get_mut(&lane) {
                state.clock_from = None;
                state.retired_at = Some(now);
            }
        }
    }

    fn retire_parks(&mut self) {
        for state in self.lanes.values_mut() {
            while state.frames.front().is_some_and(|frame| frame.park) {
                state.frames.pop_front();
                // Only an exit with nothing behind it removes the lane: a
                // frame queued after the park is a later submit and rejoins.
                if state.frames.is_empty() {
                    state.awaited = false;
                    state.parked = true;
                    state.clock_from = None;
                }
            }
        }
    }

    /// A bind control entered the scheduler. On its own it means nothing to
    /// the boundary: a bring-up process (no lane yet) enters `staged`.
    pub fn on_bind_enqueued(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid {
            *self.pending_binds.entry(pid).or_default() += 1;
            // Only a process with no lane yet is a successor worth waiting
            // for; one that already fired is held through its lane already.
            if !self.lanes.values().any(|lane| lane.owner == Some(pid)) {
                self.staged.insert(pid);
            }
        }
    }

    /// Bootstrap: seed the slot balance with initial free capacity, so the
    /// cold start's first seal waits for the whole co-launched fleet.
    pub fn preload_free_slots(&mut self, slots: usize) {
        self.pending_slots = slots as u64;
    }

    /// A retiring process's deferred teardown dropped its execution permit.
    /// While unconsumed with a successor staged, the cohort-boundary window is open.
    pub fn on_execution_slot_released(&mut self, pid: ProcessId) {
        self.departing.remove(&pid);
        self.pending_slots += 1;
    }

    /// A process acquired its execution permit; the seal now waits for
    /// `pid` itself. Guarded on `staged`.
    pub fn on_execution_slot_consumed(&mut self, pid: ProcessId) {
        self.pending_slots = self.pending_slots.saturating_sub(1);
        self.admission_queue.retain(|queued| *queued != pid);
        self.slotted.insert(pid);
        if self.staged.contains(&pid) {
            self.joins_in_flight.insert(pid);
        }
    }

    /// A process began waiting for an execution permit.
    pub fn on_admission_queued(&mut self, pid: ProcessId) {
        if !self.admission_queue.contains(&pid) {
            self.admission_queue.push_back(pid);
        }
    }

    /// It took the permit, or was cancelled before it could.
    pub fn on_admission_dequeued(&mut self, pid: ProcessId) {
        self.admission_queue.retain(|queued| *queued != pid);
    }

    /// A slot holder's Terminate leave arrived; its release broadcast is
    /// now in flight. The seal treats the imminent slot like a freed one.
    pub fn on_slotted_terminate(&mut self, pid: ProcessId) {
        if self.slotted.remove(&pid) {
            self.departing.insert(pid);
        }
    }

    /// Whether any lane's bind is still in assembly (the seal is
    /// bind-held).
    pub fn has_pending_binds(&self) -> bool {
        !self.pending_binds.is_empty()
    }

    /// The processes earmarked to fill freeing slots: `admission_queue` is
    /// FIFO-fair, so matching its front by position against the free-slot
    /// count names the takers. A joiner is not a wait-set member, so
    /// sealing without it excludes nobody.
    pub fn earmarked(&self) -> impl Iterator<Item = ProcessId> + '_ {
        let slots = self.pending_slots as usize + self.departing.len();
        self.admission_queue
            .iter()
            .filter(|pid| self.staged.contains(pid))
            .take(slots)
            .copied()
    }

    pub fn is_joining(&self) -> bool {
        !self.joins_in_flight.is_empty() || self.earmarked().next().is_some()
    }

    pub fn on_bind_completed(&mut self, pid: Option<ProcessId>) {
        if let Some(pid) = pid
            && let Some(count) = self.pending_binds.get_mut(&pid)
        {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.pending_binds.remove(&pid);
            }
        }
    }

    /// A pipeline scope left. `purge_queued` for Terminate/Suspend (queued
    /// fires rejected); graceful Close releases the lane immediately but
    /// keeps queued frames draining to settlement. `owner` is required
    /// whenever the leaver has no lane yet to recover one from.
    pub fn on_lane_leave(&mut self, lane: ProcessId, owner: Option<ProcessId>, purge_queued: bool) {
        // Recover the owner from the lane while it still exists; an explicit
        // `owner` wins, since a laneless leaver can only be identified that way.
        let owner = owner.or_else(|| self.lanes.get(&lane).and_then(|state| state.owner));
        if purge_queued {
            self.lanes.remove(&lane);
            self.truncated_seqs.remove(&lane);
            // A terminated lane's fires may be cancelled rather than
            // retired, so this can't rely on `on_frame_retired`; purge it here.
            self.in_flight_lanes.remove(&lane);
        } else if let Some(state) = self.lanes.get_mut(&lane) {
            state.awaited = false;
            // A leave is a park, not a verdict: rejoin is implicit on the
            // next accepted fire, which needs `parked` latched here too.
            state.parked = true;
            self.truncate_incomplete(lane);
            let drained = self
                .lanes
                .get(&lane)
                .is_some_and(|state| state.frames.is_empty());
            if drained {
                self.lanes.remove(&lane);
            }
        }
        if let Some(owner) = owner {
            self.pending_binds.remove(&owner);
            self.cohorts.retain(|(who, _), _| *who != owner);
            self.forget_staged(owner);
        }
        self.maybe_reset_episode();
    }

    /// Every scope owned by `owner` left (process terminate/suspend).
    pub fn on_process_leave(&mut self, owner: ProcessId) {
        let owned: Vec<ProcessId> = self
            .lanes
            .iter()
            .filter(|(_, lane)| lane.owner == Some(owner))
            .map(|(id, _)| *id)
            .collect();
        for id in owned {
            self.truncated_seqs.remove(&id);
            // The process is gone: no retirement is owed. Not done in
            // `on_process_suspend`, where the debt is still real.
            self.in_flight_lanes.remove(&id);
        }
        self.lanes.retain(|_, lane| lane.owner != Some(owner));
        self.pending_binds.remove(&owner);
        self.cohorts.retain(|(who, _), _| *who != owner);
        self.suspended.remove(&owner);
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

    /// The planner is evicting `owner`: its lanes stop being awaited so
    /// boundaries seal without them, but already-submitted frames still
    /// drain. Rejoin is implicit on the next post-restore arrival.
    pub fn on_process_suspend(&mut self, owner: ProcessId) {
        self.suspended.insert(owner);
        let owned: Vec<ProcessId> = self
            .lanes
            .iter()
            .filter(|(_, lane)| lane.owner == Some(owner))
            .map(|(id, _)| *id)
            .collect();
        for lane_id in owned {
            if let Some(lane) = self.lanes.get_mut(&lane_id) {
                lane.awaited = false;
                // Same rejoin contract as the park-shaped leave: the
                // `parked` latch makes `record_arrival` re-enter it on the
                // first post-restore fire.
                lane.parked = true;
            }
            self.truncate_incomplete(lane_id);
            let drained = self
                .lanes
                .get(&lane_id)
                .is_some_and(|lane| lane.frames.is_empty());
            if drained {
                self.lanes.remove(&lane_id);
            }
        }
        self.pending_binds.remove(&owner);
        self.cohorts.retain(|(who, _), _| *who != owner);
        self.forget_staged(owner);
        self.maybe_reset_episode();
    }

    /// The planner concluded `owner` is runnable again. Clearing the
    /// suspend mark lets its lanes rejoin the wait-set.
    pub fn on_process_resume(&mut self, owner: ProcessId) {
        self.suspended.remove(&owner);
    }

    /// Makes every unfinished frame on `lane` sealable at what has arrived
    /// (a lane leaving mid-frame still needs its slots to seal and drain,
    /// releasing the fire leases an eviction's quiescence wait blocks on).
    fn truncate_incomplete(&mut self, lane_id: ProcessId) {
        let Some(lane) = self.lanes.get_mut(&lane_id) else {
            return;
        };
        let mut cut = None;
        for frame in &mut lane.frames {
            if frame.is_complete() {
                continue;
            }
            frame.expected = frame.fires.len() as u32;
            frame.truncated = true;
            // Only the newest frame can be unfinished, so one seq covers
            // the cut.
            cut = Some(frame.seq);
        }
        if let Some(seq) = cut {
            self.truncated_seqs.insert(lane_id, seq);
        }
    }

    /// A staged or joining successor departed before its first fire: the
    /// seal must never wait for a lane that cannot arrive.
    fn forget_staged(&mut self, pid: ProcessId) {
        self.staged.remove(&pid);
        self.joins_in_flight.remove(&pid);
        self.in_flight_since.remove(&pid);
    }

    /// Mirror of the quorum's empty-wait-set re-arm: when the last awaited
    /// lane leaves, the next fleet starts a fresh episode.
    fn maybe_reset_episode(&mut self) {
        if self.lanes.values().any(|lane| lane.awaited) {
            return;
        }
        self.strict_watchdog_deadline = None;
        self.idle_dumped = false;
    }

    /// A lane whose front frame could actually be sealed now. A lane whose
    /// attention group would fire short is NOT a candidate: counting it as
    /// one would only spin the ready-mode boundary against a seal that
    /// refuses it.
    fn have_seal_candidate(&self) -> bool {
        self.lanes.values().any(|lane| {
            lane.frames.front().is_some_and(PendingFrame::is_complete) && !self.group_short(lane)
        })
    }

    /// Whether a boundary is open: some awaited lane hasn't fired into it
    /// yet. While open, the wait-all gate isn't re-evaluated.
    fn boundary_open(&self) -> bool {
        self.lanes
            .values()
            .any(|lane| lane.awaited && !lane.fired_this_boundary)
    }

    /// Open a fresh boundary: every awaited lane owes it one frame.
    fn open_boundary(&mut self) {
        for lane in self.lanes.values_mut() {
            lane.fired_this_boundary = false;
        }
    }

    /// Close the open boundary without serving the rest: no owing lane can
    /// seal, so control returns to the wait-all gate rather than a narrow
    /// partition.
    fn close_boundary(&mut self) {
        for lane in self.lanes.values_mut() {
            lane.fired_this_boundary = true;
        }
    }

    /// Seals one partition of the open boundary — first-fit against the
    /// per-wave row/token budgets: lowest-id owing lane first (progress
    /// guarantee), then resubmitted served lanes, then the rest. One
    /// partition per call (not the whole boundary) is what lets a
    /// prefill-heavy boundary produce mixed prefill+decode launches.
    fn seal(&mut self) -> Option<FramePlan> {
        if !self.have_seal_candidate() {
            return None;
        }
        // Only opens a boundary if it actually seals; a call that finds
        // nothing sealable leaves the policy untouched.
        let mid_boundary = self.boundary_open();

        // One partition per call: unfired lanes first (lane-id order),
        // then already-fired-and-resubmitted lanes, then the rest.
        let k = self.k;
        let max_wave_rows = self.max_wave_rows;
        let max_wave_tokens = self.max_wave_tokens;
        loop {
            let mut fresh: Vec<ProcessId> = Vec::new();
            let mut continuing: Vec<ProcessId> = Vec::new();
            for (lane_id, lane) in self.lanes.iter() {
                if !lane.frames.front().is_some_and(PendingFrame::is_complete) {
                    continue;
                }
                // Its stated group is not all here: this lane waits for its
                // groupmates, not for a partition. Asked HERE, on the one
                // path every seal takes, so no mode and no timing can route
                // around it.
                if self.group_short(lane) {
                    continue;
                }
                if mid_boundary && lane.fired_this_boundary {
                    continuing.push(*lane_id);
                } else {
                    fresh.push(*lane_id);
                }
            }
            if mid_boundary && fresh.is_empty() {
                // Nothing left that advances this boundary; sealing only
                // continuation lanes would be a narrow epoch.
                return None;
            }
            let mut order: Vec<ProcessId> = Vec::with_capacity(fresh.len() + continuing.len());
            let mut rest = fresh.as_slice();
            if let Some((first, tail)) = fresh.split_first() {
                order.push(*first);
                rest = tail;
            }
            order.extend_from_slice(&continuing);
            order.extend_from_slice(rest);

            let mut waves: Vec<Vec<u64>> = vec![Vec::new(); k];
            let mut fire_waves = HashMap::new();
            let mut wave_tokens = vec![0usize; k];
            let mut wave_rows = vec![0usize; k];
            let mut members: HashSet<ProcessId> = HashSet::new();
            let mut dropped_empty = false;
            // **UNITS, NOT LANES.** A stated attention group's lanes attend
            // each other inside ONE fire, so they enter a partition
            // together or not at all. Without this the first-fit below
            // would defer one lane of a group over the row/token budget and
            // fire the rest without it — the same silent wrongness as
            // sealing a cohort short, one boundary later. An ungrouped lane
            // is a unit of one, which is the fleet's whole population.
            let mut units: Vec<Vec<ProcessId>> = Vec::with_capacity(order.len());
            let mut placed: BTreeSet<ProcessId> = BTreeSet::new();
            for lane_id in &order {
                if !placed.insert(*lane_id) {
                    continue;
                }
                match self.lanes.get(lane_id).and_then(LaneState::group_key) {
                    None => units.push(vec![*lane_id]),
                    Some(key) => {
                        let unit: Vec<ProcessId> = order
                            .iter()
                            .copied()
                            .filter(|id| {
                                self.lanes.get(id).and_then(LaneState::group_key) == Some(key)
                            })
                            .collect();
                        placed.extend(unit.iter().copied());
                        units.push(unit);
                    }
                }
            }
            for unit in units {
                // Measure the unit whole, then commit it whole.
                let mut unit_fires: Vec<(ProcessId, Vec<PlacedFire>)> =
                    Vec::with_capacity(unit.len());
                let mut unit_rows = vec![0usize; k];
                let mut unit_tokens = vec![0usize; k];
                let mut short = false;
                for lane_id in &unit {
                    let Some(lane) = self.lanes.get_mut(lane_id) else {
                        short = true;
                        continue;
                    };
                    let Some(front) = lane.frames.front() else {
                        short = true;
                        continue;
                    };
                    let live: Vec<PlacedFire> = front
                        .fires
                        .iter()
                        .filter_map(|fire| {
                            fire.fire_id.map(|fire_id| PlacedFire {
                                wave: (fire.slot as usize).min(k - 1),
                                fire_id,
                                rows: fire.rows.max(1),
                                tokens: fire.tokens,
                            })
                        })
                        .collect();
                    if live.is_empty() {
                        // Every fire was rejected at admission: drop the
                        // frame, the lane's next front may be sealable.
                        lane.frames.pop_front();
                        dropped_empty = true;
                        short = true;
                        continue;
                    }
                    // The budget test is per fire against what is already
                    // in the partition PLUS what this unit's earlier lanes
                    // put there — for a unit of one, exactly the first-fit
                    // rule as it always was.
                    let fits = live.iter().all(|fire| {
                        wave_rows[fire.wave] + unit_rows[fire.wave] + fire.rows <= max_wave_rows
                            && wave_tokens[fire.wave] + unit_tokens[fire.wave] + fire.tokens
                                <= max_wave_tokens
                    });
                    if !fits {
                        short = true;
                        continue;
                    }
                    for fire in &live {
                        unit_rows[fire.wave] += fire.rows;
                        unit_tokens[fire.wave] += fire.tokens;
                    }
                    unit_fires.push((*lane_id, live));
                }
                if short {
                    // Over budget, or a groupmate that cannot join: the
                    // whole unit seals into a later partition.
                    continue;
                }
                for (lane_id, live) in unit_fires {
                    for fire in &live {
                        wave_rows[fire.wave] += fire.rows;
                        wave_tokens[fire.wave] += fire.tokens;
                        waves[fire.wave].push(fire.fire_id);
                        fire_waves.insert(fire.fire_id, fire.wave);
                    }
                    members.insert(lane_id);
                    let lane = self.lanes.get_mut(&lane_id).expect("measured lane exists");
                    let frame_complete_at = lane.frames.front().and_then(|front| front.complete_at);
                    // Turnaround sample: last retirement to this completion
                    // is the whole result -> guest -> resubmit round trip.
                    if let (Some(done), Some(from)) = (frame_complete_at, lane.retired_at) {
                        let us = done.saturating_duration_since(from).as_micros() as u64;
                        if let Some(stats) = &self.stats {
                            use std::sync::atomic::Ordering::Relaxed;
                            stats.fire.quorum.turnaround_sum_us.fetch_add(us, Relaxed);
                            stats.fire.quorum.turnaround_n.fetch_add(1, Relaxed);
                            stats.fire.quorum.turnaround_max_us.fetch_max(us, Relaxed);
                        }
                    }
                    lane.served = true;
                    lane.frames.pop_front();
                }
            }
            if fire_waves.is_empty() {
                // Nothing sealable; retry only if a frame with no live fire
                // has been dropped, since the lane's next front may be sealable.
                if dropped_empty {
                    continue;
                }
                self.lanes
                    .retain(|_, lane| lane.awaited || lane.leashed || !lane.frames.is_empty());
                return None;
            }
            if super::worker::wave_trace() {
                let complete = self
                    .lanes
                    .values()
                    .filter(|lane| lane.frames.front().is_some_and(PendingFrame::is_complete))
                    .count();
                let awaited = self.lanes.values().filter(|lane| lane.awaited).count();
                super::worker::wave_trace_emit(format!(
                    "[wave-trace] t={}us seal partition={} fresh={} continuing={} complete_left={} awaited={} lanes={} in_flight={} mid_boundary={mid_boundary} executing={}",
                    super::worker::wave_trace_us(),
                    members.len(),
                    fresh.len(),
                    continuing.len(),
                    complete,
                    awaited,
                    self.lanes.len(),
                    self.in_flight_lanes.len(),
                    self.executing_now
                ));
            }
            if !mid_boundary {
                self.open_boundary();
            }
            for member in &members {
                if let Some(lane) = self.lanes.get_mut(member) {
                    lane.fired_this_boundary = true;
                }
            }
            self.record_sealed_waves(waves.iter().filter(|wave| !wave.is_empty()).count());
            self.record_seal_engagement();
            let _ = &fire_waves;
            self.sealed.push_back(SealedFrame {
                waves,
                members: members.iter().copied().collect(),
            });
            // A leashed lane is kept even with nothing queued: its silence
            // clock is the only thing that can still terminate an abandoned
            // pipeline.
            self.lanes
                .retain(|_, lane| lane.awaited || lane.leashed || !lane.frames.is_empty());
            return Some(FramePlan::Dispatch(Vec::new()));
        }
    }

    /// An unstamped rider batch posted outside the sealed waves: still one
    /// wave fire for the density counters.
    pub fn record_rider_wave(&self) {
        self.record_sealed_waves(1);
    }

    /// Wave-density probe: `avg_active = wave_active_sum / wave_fires`
    /// discriminates a persistent wait-set from one that empties between fires.
    fn record_sealed_waves(&self, wave_count: usize) {
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            let awaited = self.lanes.values().filter(|lane| lane.awaited).count() as u64;
            let waves = wave_count as u64;
            stats.fire.quorum.wave_fires.fetch_add(waves, Relaxed);
            stats
                .fire
                .quorum
                .wave_active_sum
                .fetch_add(awaited * waves, Relaxed);
        }
    }

    /// Chain engagement, one sample per sealed partition: was the device
    /// still working when this boundary was assembled?
    fn record_seal_engagement(&self) {
        if let Some(stats) = &self.stats {
            use std::sync::atomic::Ordering::Relaxed;
            stats.fire.quorum.seal_events.fetch_add(1, Relaxed);
            if self.executing_now {
                stats.fire.quorum.seal_while_executing.fetch_add(1, Relaxed);
            }
        }
    }

    /// The next sealed frame the worker should post whole, if any.
    /// `still_queued` resolves sealed ids that vanished (rejected/cancelled).
    /// `blocked_lanes` holds a frame whose lane has a pre-launch copy
    /// barrier queued, until the copy retires. `executing` is the worker's
    /// in-flight signal. Frames overlap on-stream: posting is globally
    /// ordered (seal order, then slot order within a frame), with no
    /// launch-time barrier.
    pub fn plan_dispatch(
        &mut self,
        still_queued: &QueuedFireIds,
        blocked_lanes: &HashSet<ProcessId>,
        executing: bool,
        now: Instant,
    ) -> FramePlan {
        self.executing_now = executing;
        // Device-idle census, taken at the top so it catches every way the
        // policy can leave the device starved. `PIE_IDLE_DUMP_US` arms it.
        let threshold = idle_dump_threshold_us();
        if threshold != u64::MAX
            && !executing
            && let Some(since) = self.last_retire_at
        {
            let idle = now.saturating_duration_since(since).as_micros() as u64;
            // `PIE_IDLE_DUMP_MIN_SEATS` (default 0) is a seat floor so a
            // near-empty startup wait-set can be excluded from the census.
            let seated = self.lanes.values().filter(|lane| lane.awaited).count();
            if idle >= threshold && !self.idle_dumped && seated >= idle_dump_min_seats() {
                self.idle_dumped = true;
                let (mut ready, mut empty_owed, mut empty_unowed, mut partial) = (0, 0, 0, 0);
                let (mut unowed_fresh, mut unowed_between) = (0, 0);
                // Turnaround age of the lanes actually denying this gate.
                let (mut turn_max, mut turn_sum, mut turn_n) = (0u64, 0u64, 0u64);
                let mut oldest_ready_us = 0u64;
                let mut newest_ready_us = u64::MAX;
                for (lane_id, lane) in &self.lanes {
                    if !lane.awaited {
                        continue;
                    }
                    if lane.frames.front().is_some_and(PendingFrame::is_complete) {
                        ready += 1;
                        if let Some(at) = lane.frames.front().and_then(|f| f.complete_at) {
                            let age = now.saturating_duration_since(at).as_micros() as u64;
                            oldest_ready_us = oldest_ready_us.max(age);
                            newest_ready_us = newest_ready_us.min(age);
                        }
                    } else if lane.frames.is_empty() {
                        let owes = self.in_flight_lanes.contains(lane_id)
                            || lane
                                .owner
                                .is_some_and(|owner| self.pending_binds.contains_key(&owner));
                        if owes {
                            empty_owed += 1;
                        } else {
                            empty_unowed += 1;
                            if lane.served {
                                unowed_between += 1;
                                if let Some(at) = lane.retired_at {
                                    let age = now.saturating_duration_since(at).as_micros() as u64;
                                    turn_max = turn_max.max(age);
                                    turn_sum += age;
                                    turn_n += 1;
                                }
                            } else {
                                unowed_fresh += 1;
                            }
                        }
                    } else {
                        partial += 1;
                    }
                }
                // `fresh == 0` with a boundary open is the one state that
                // makes `seal` return None with work everywhere.
                let open = self.boundary_open();
                let (mut fresh, mut continuing) = (0, 0);
                for lane in self.lanes.values() {
                    if lane.frames.front().is_some_and(PendingFrame::is_complete) {
                        if open && lane.fired_this_boundary {
                            continuing += 1;
                        } else {
                            fresh += 1;
                        }
                    }
                }
                println!(
                    "[device-idle] {idle}us awaited={} ready={ready} empty+owed={empty_owed} \
empty+unowed={empty_unowed}(fresh={unowed_fresh},between={unowed_between}) \
partial_front={partial} sealed={} staged={} \
pending_slots={} joins={} binds={} boundary_open={open} fresh={fresh} \
continuing={continuing} turnaround_max={turn_max}us \
turnaround_mean={}us ready_age_oldest={oldest_ready_us}us \
ready_age_newest={}us",
                    ready + empty_owed + empty_unowed + partial,
                    self.sealed.len(),
                    self.staged.len(),
                    self.pending_slots,
                    self.joins_in_flight.len(),
                    self.pending_binds.values().sum::<usize>(),
                    turn_sum.checked_div(turn_n).unwrap_or(0),
                    if newest_ready_us == u64::MAX {
                        0
                    } else {
                        newest_ready_us
                    },
                );
            }
        }
        loop {
            // A guest that parked leaves the wait-set before anything is
            // counted missing, so the gather below never sees it.
            self.retire_parks();
            // Resolve sealed ids that left the queue without posting.
            for frame in &mut self.sealed {
                for wave in &mut frame.waves {
                    wave.retain(|fire_id| still_queued.contains(fire_id));
                }
            }
            while self
                .sealed
                .front()
                .is_some_and(|frame| frame.waves.iter().all(Vec::is_empty))
            {
                self.sealed.pop_front();
            }
            if let Some(front) = self.sealed.front() {
                if front
                    .members
                    .iter()
                    .any(|member| blocked_lanes.contains(member))
                {
                    // The copy's retirement re-decides through the scheduler
                    // channel; the hold is only a liveness backstop.
                    if let Some(stats) = &self.stats {
                        stats
                            .fire
                            .quorum
                            .dispatch_blocked_holds
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    return FramePlan::Hold(Duration::from_micros(500));
                }
                let frame = self.sealed.pop_front().expect("front frame exists");
                // The runtime now owes every member a result; their deadline
                // clocks stop here and restart only at `on_frame_retired`.
                self.in_flight_lanes.extend(frame.members.iter().copied());
                // Stamped once per lane, not re-stamped, so a lane already
                // in debt is measured from when the debt began.
                for member in frame.members.iter().copied() {
                    self.in_flight_since.entry(member).or_insert(now);
                }
                // Nothing was executing when this frame was posted, so the
                // device has been idle since the last retirement.
                let mut starved_us = 0u64;
                if !self.executing_now
                    && let Some(since) = self.last_retire_at
                    && let Some(stats) = &self.stats
                {
                    use std::sync::atomic::Ordering::Relaxed;
                    let idle = now.saturating_duration_since(since).as_micros() as u64;
                    stats.fire.quorum.device_idle_us.fetch_add(idle, Relaxed);
                    stats.fire.quorum.device_idle_gaps.fetch_add(1, Relaxed);
                    stats.record_bubble_us(idle);
                    starved_us = idle;
                }
                // `PIE_IDLE_DUMP_US=<us>`: dump the fleet state across a
                // starvation gap longer than the threshold. Off by default.
                if starved_us >= idle_dump_threshold_us() {
                    println!("[idle-gap] {starved_us}us  {}", self.debug_summary());
                }
                return FramePlan::Dispatch(frame.waves);
            }
            // Boundary: seal only once every awaited lane's next frame is
            // fully submitted; the watchdog below reports a stalled gather
            // but never evicts. Once open, every owing lane was ready when
            // the gate held, so the next partition can't be narrow.
            if self.boundary_open() {
                if let Some(plan) = self.seal() {
                    match plan {
                        FramePlan::Dispatch(_) => continue,
                        plan => return plan,
                    }
                }
                // Every remaining owing lane is unsealable: close the
                // boundary and re-gate.
                self.close_boundary();
            }
            if !self.lanes.values().any(|lane| !lane.frames.is_empty()) {
                // Nothing queued anywhere: no gather episode is running.
                self.strict_watchdog_deadline = None;
                self.idle_dumped = false;
                return FramePlan::Park;
            }
            // A stated cohort that never completed: the request dies here,
            // by name. It is never released into a fire over whoever showed
            // up — see `cohort_verdict`.
            let doomed = self.cohort_verdict(now);
            if !doomed.is_empty() {
                return FramePlan::Terminate(doomed);
            }
            // `missing` counts awaited lanes whose next frame isn't fully
            // submitted; an idle member is waited for like any other. A
            // gathering cohort is NOT counted: it holds its own lanes out
            // of `seal` and lets the rest of the fleet through.
            let mut missing = 0;
            // The submit deadline's clock is armed here per lane: it stops
            // only for debts owed to this lane, and isn't gated on
            // `executing` — a silent member must be found even while
            // others are busy.
            let mut expired: Vec<ProcessId> = Vec::new();
            let leash = self.submit_deadline;
            let silence = self.silence_timeout;
            // The relaxation applies only while a frame is executing: with
            // the device idle there's no chain to protect, so dense
            // gathering is right.
            let contributed_relax = self.gate_contributed && executing;
            // Reached only for a closed boundary, where `fired_this_boundary`
            // is true of every lane — so `gate_verdict` doesn't test it.
            debug_assert!(
                !self.boundary_open(),
                "the wait-all gate is evaluated only against a closed boundary"
            );
            for (lane_id, lane) in self.lanes.iter_mut() {
                let owes = self.in_flight_lanes.contains(lane_id)
                    || lane
                        .owner
                        .is_some_and(|owner| self.pending_binds.contains_key(&owner));
                // The debt's own deadline (twice the silence timeout)
                // backstops a runtime that never answers; past it, ordinary
                // expiry judges the lane.
                let debt_since = if owes {
                    self.in_flight_since.get(lane_id).copied()
                } else {
                    None
                };
                let owes_forever = debt_since
                    .is_some_and(|from| now.saturating_duration_since(from) >= silence * 2);
                // Leashed: already dropped by the leash below and still
                // silent — blocks nobody, but the clock keeps running.
                let blocking = match lane.gate_verdict(contributed_relax && !owes_forever, owes) {
                    GateVerdict::Blocking => true,
                    GateVerdict::SkippedOwed | GateVerdict::Satisfied => false,
                };
                if !blocking && !lane.leashed {
                    lane.clock_from = None;
                    continue;
                }
                if owes && !owes_forever {
                    lane.clock_from = None;
                } else {
                    // Seed the silence clock from when the debt began, not
                    // from now, or the backstop would take three timeouts.
                    if owes_forever {
                        lane.clock_from = debt_since;
                    }
                    match lane.clock_from {
                        Some(from) if now.saturating_duration_since(from) >= silence => {
                            // Silent through the whole timeout with nothing
                            // owed: `forward.park()` is the honest exit; skipping it is the contract breach.
                            lane.awaited = false;
                            lane.leashed = false;
                            lane.clock_from = None;
                            expired.push(lane.owner.unwrap_or(*lane_id));
                            continue;
                        }
                        Some(from) if blocking && now.saturating_duration_since(from) >= leash => {
                            // Leash, not a verdict: the lane stops being
                            // waited on so the boundary seals; its next fire
                            // rejoins through the ordinary parked path.
                            lane.awaited = false;
                            lane.parked = true;
                            lane.leashed = true;
                            continue;
                        }
                        Some(_) => {}
                        None => lane.clock_from = Some(now),
                    }
                }
                if blocking {
                    missing += 1;
                }
            }
            if !expired.is_empty() {
                // One process can own several lanes, so the same owner can
                // breach on more than one of them in a single pass.
                expired.sort_unstable();
                expired.dedup();
                return FramePlan::Terminate(
                    expired
                        .into_iter()
                        .map(|pid| (pid, Doom::Abandoned))
                        .collect(),
                );
            }
            if super::worker::wave_trace() {
                super::worker::wave_trace_emit(format!(
                    "[wave-trace] t={}us gate missing={missing} executing={executing} candidate={} in_flight={} awaited={} lanes={}",
                    super::worker::wave_trace_us(),
                    self.have_seal_candidate(),
                    self.in_flight_lanes.len(),
                    self.lanes.values().filter(|lane| lane.awaited).count(),
                    self.lanes.len()
                ));
            }
            // A joiner never holds the seal: it is not a wait-set member
            // yet, so sealing without it excludes nobody.
            if missing > 0 {
                if executing {
                    // An epoch is executing: retirements re-decide and the
                    // gather continues. Drop the quiescence mark; this exit
                    // isn't a hold cycle.
                    self.quiesce_mark = None;
                    self.coalesce_since = None;
                    return FramePlan::Park;
                }
                // The ready-mode hold's remaining window, `None` outside one.
                let mut window_left: Option<Duration> = None;
                if self.seal_mode_ready && self.have_seal_candidate() {
                    // Ready mode: open the boundary from the
                    // arrival-complete subset only after one full hold
                    // cycle with no new arrival, so missing lanes are
                    // genuinely slow, not mid-burst — and only once the
                    // boundary has been open for the coalescing window, so
                    // lanes a fraction of a millisecond apart share it
                    // (`seal_coalesce`).
                    let since = *self.coalesce_since.get_or_insert(now);
                    let open_for = now.saturating_duration_since(since);
                    let quiesced = self.quiesce_mark == Some(self.gather_seq);
                    if quiesced && open_for >= self.seal_coalesce {
                        self.quiesce_mark = None;
                        self.coalesce_since = None;
                        self.strict_watchdog_deadline = None;
                        self.idle_dumped = false;
                        match self.seal() {
                            Some(FramePlan::Dispatch(_)) => continue,
                            Some(plan) => return plan,
                            // Candidates exist but none sealed (e.g. every
                            // ready lane is capacity-deferred): fall through
                            // to the ordinary hold.
                            None => {}
                        }
                    } else {
                        if !quiesced {
                            self.quiesce_mark = Some(self.gather_seq);
                        }
                        window_left = Some(self.seal_coalesce.saturating_sub(open_for));
                    }
                }
                let deadline = self
                    .strict_watchdog_deadline
                    .get_or_insert(now + Duration::from_micros(STRICT_WATCHDOG_US));
                if now >= *deadline {
                    *deadline = now + Duration::from_micros(STRICT_WATCHDOG_US);
                }
                let mut hold = deadline
                    .saturating_duration_since(now)
                    .min(Duration::from_micros(GATHER_POLL_US));
                // Inside a window, re-look when it closes if that is sooner
                // than the poll (never sooner than the quiescence check
                // needs: a zero window is one poll, as before).
                if let Some(left) = window_left
                    && left > Duration::ZERO
                {
                    hold = hold.min(left);
                }
                return FramePlan::Hold(hold);
            }
            self.strict_watchdog_deadline = None;
            self.idle_dumped = false;
            // Nothing is missing, so this evaluation is not a hold cycle
            // either: retire the mark.
            self.quiesce_mark = None;
            self.coalesce_since = None;
            // Early seal: the gate held, so seal now — normally while the
            // previous frame still executes.
            match self.seal() {
                Some(FramePlan::Dispatch(_)) => continue,
                Some(plan) => return plan,
                // Ready lanes exist but none can seal (all busy in an
                // executing round partition): retirements re-decide.
                None if self.group_assembling() => {
                    // Except when a group is assembling: nothing would wake
                    // the policy from a park (its groupmate's arrival is
                    // the only event, and a group that will never assemble
                    // has no event at all), and `cohort_verdict` above is
                    // only read from an evaluation. Poll instead.
                    return FramePlan::Hold(Duration::from_micros(GATHER_POLL_US));
                }
                None => return FramePlan::Park,
            }
        }
    }

    /// Probe/diagnostic summary line.
    /// Whether any lane holds a queued frame.
    pub fn has_queued_frames(&self) -> bool {
        self.lanes.values().any(|lane| !lane.frames.is_empty())
    }

    pub fn debug_summary(&self) -> String {
        use std::fmt::Write as _;
        let mut out = format!(
            "frame k={} lanes={} awaited={} sealed={} \
pending_binds={} staged={} joins_in_flight={} departing={} suspended={} \
pending_slots={} watchdog={:?}",
            self.k,
            self.lanes.len(),
            self.lanes.values().filter(|lane| lane.awaited).count(),
            self.sealed.len(),
            self.pending_binds.values().sum::<usize>(),
            self.staged.len(),
            self.joins_in_flight.len(),
            self.departing.len(),
            self.suspended.len(),
            self.pending_slots,
            self.strict_watchdog_deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now())),
        );
        let _ = write!(
            out,
            "\n  pending_bind_pids=[{}]",
            self.pending_binds
                .keys()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        for (pid, lane) in &self.lanes {
            let front_complete = lane.frames.front().is_some_and(PendingFrame::is_complete);
            let _ = write!(
                out,
                "\n  lane {pid}: owner={:?} awaited={} queued_frames={} front_complete={front_complete}",
                lane.owner,
                lane.awaited,
                lane.frames.len(),
            );
        }
        for (index, frame) in self.sealed.iter().enumerate() {
            let _ = write!(
                out,
                "\n  sealed[{index}]: waves={} fires={} members={}",
                frame.waves.len(),
                frame.waves.iter().map(Vec::len).sum::<usize>(),
                frame.members.len(),
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid() -> ProcessId {
        ProcessId::new_v4()
    }

    fn stamp(lane: ProcessId, seq: u64, slot: u32, fires: u32) -> FrameStamp {
        FrameStamp {
            lane,
            seq,
            slot,
            fires,
        }
    }

    fn plan(policy: &mut FramePolicy, queued: &QueuedFireIds, now: Instant) -> FramePlan {
        policy.plan_dispatch(queued, &HashSet::new(), false, now)
    }

    /// Flatten a whole-frame dispatch for order-insensitive membership
    /// asserts.
    fn fires(plan: &FramePlan) -> Vec<u64> {
        match plan {
            FramePlan::Dispatch(waves) => waves.iter().flatten().copied().collect(),
            plan => panic!("expected a frame dispatch, got {plan:?}"),
        }
    }

    #[test]
    fn seals_complete_lanes_and_orders_waves_by_slot() {
        let mut policy = FramePolicy::new(4, 64, 4096, None);
        let (a, b) = (pid(), pid());
        // Lane a: full decode frame (4 fires). Lane b: chunk in slot 0 only.
        for slot in 0..4 {
            policy.on_fire_enqueued(stamp(a, 0, slot, 4), Some(a), 100 + slot as u64, 1, 1, None);
        }
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 200, 37, 1, None);

        let queued: QueuedFireIds = [100, 101, 102, 103, 200].into_iter().collect();
        let sealed = plan(&mut policy, &queued, Instant::now());
        // One whole frame: wave 0 = both lanes' slot-0 fires (lane-id order
        // preserved), later slots in slot order.
        let FramePlan::Dispatch(waves) = sealed else {
            panic!("expected a whole-frame dispatch");
        };
        assert_eq!(waves.len(), 4);
        assert_eq!(waves[0].len(), 2);
        assert!(waves[0].contains(&100) && waves[0].contains(&200));
        assert_eq!(waves[1], vec![101]);
        assert_eq!(waves[2], vec![102]);
        assert_eq!(waves[3], vec![103]);
    }

    /// **A GROUP'S FIRST FRAME WAITS FOR ITS COHORT** (design D2): three
    /// passes of one process name attention group 7 and each declares a
    /// cohort of three; the gate has never seen the other two pipelines, so
    /// without the cohort the first would seal alone and attend alone. It
    /// holds instead, until the third arrives — then one frame, one wave,
    /// all three. Steady state re-gathers the same way and says the same
    /// thing as wait-all, and a complete cohort is forgotten either way.
    #[test]
    fn a_grouped_lanes_first_frame_waits_for_its_cohort() {
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_secs(86_400));
        let owner = pid();
        let (caption, context, image) = (pid(), pid(), pid());
        let now = Instant::now();

        policy.on_fire_enqueued(stamp(caption, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 3)));
        let queued: QueuedFireIds = [1].into_iter().collect();
        assert!(
            matches!(plan(&mut policy, &queued, now), FramePlan::Hold(_)),
            "the first lane of a cohort of three must not seal alone"
        );
        policy.on_fire_enqueued(stamp(context, 0, 0, 1), Some(owner), 2, 16, 1, Some((7, 3)));
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        assert!(
            matches!(plan(&mut policy, &queued, now), FramePlan::Hold(_)),
            "two of three still hold"
        );
        policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 3, 64, 1, Some((7, 3)));
        let queued: QueuedFireIds = [1, 2, 3].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, now));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![1, 2, 3], "the whole cohort seals into one frame");
        assert!(policy.cohorts.is_empty(), "a complete cohort is forgotten");

        // Steady state: the three are members now, so wait-all would hold
        // for the slow one on its own; the cohort re-gathers and agrees.
        policy.on_fire_enqueued(stamp(caption, 1, 0, 1), Some(owner), 4, 8, 1, Some((7, 3)));
        policy.on_fire_enqueued(stamp(image, 1, 0, 1), Some(owner), 5, 64, 1, Some((7, 3)));
        let queued: QueuedFireIds = [4, 5].into_iter().collect();
        assert!(matches!(plan(&mut policy, &queued, now), FramePlan::Hold(_)));
        policy.on_fire_enqueued(stamp(context, 1, 0, 1), Some(owner), 6, 16, 1, Some((7, 3)));
        assert!(policy.cohorts.is_empty(), "a complete cohort is forgotten");
        let queued: QueuedFireIds = [4, 5, 6].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, now));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![4, 5, 6]);
    }

    /// **A GROUP COMPOSES ON A PIPELINE THE GATE HAS ALREADY SEEN.** A
    /// guest may do ungrouped work down a pipeline and then reuse it for a
    /// member of an attention group — z-image's parity guest runs `refine`
    /// on one pipeline, then puts the `denoise` group's image lane back on
    /// it and its context lane on a fresh one. The cohort's two halves have
    /// to count the same population for that to work: `expected` counts
    /// every pass naming the group, so `arrived` must count every member's
    /// fire and not only the ones whose pipeline was new to the gate. It
    /// used to skip the known lane, so this shape stated a cohort of two,
    /// gathered one, and died with `CohortNeverComplete` no matter what the
    /// guest did. Both submit orders compose here — which member the gate
    /// happened to have seen must not decide the answer.
    #[test]
    fn a_group_composes_on_a_pipeline_that_already_fired_ungrouped() {
        for reuse_first in [false, true] {
            let mut policy = FramePolicy::new(1, 64, 4096, None)
                .with_seal_mode_ready(false)
                .with_submit_deadline(Duration::from_millis(50));
            let owner = pid();
            let (image, context) = (pid(), pid());
            let now = Instant::now();

            // `refine`: an UNGROUPED fire down the image lane's pipeline,
            // which puts that pipeline in the wait-set for good.
            policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 1, 8, 1, None);
            let queued: QueuedFireIds = [1].into_iter().collect();
            assert_eq!(fires(&plan(&mut policy, &queued, now)), vec![1]);
            policy.on_frame_retired([image]);

            // `denoise`: both lanes name group 0 and both count two passes.
            let (first, second) = if reuse_first {
                ((image, 1, 2u64), (context, 0, 3u64))
            } else {
                ((context, 0, 2u64), (image, 1, 3u64))
            };
            policy.on_fire_enqueued(
                stamp(first.0, first.1, 0, 1),
                Some(owner),
                first.2,
                8,
                1,
                Some((0, 2)),
            );
            let queued: QueuedFireIds = [first.2].into_iter().collect();
            assert!(
                matches!(plan(&mut policy, &queued, now), FramePlan::Hold(_)),
                "one lane of a cohort of two must not seal alone (reuse_first={reuse_first})"
            );
            policy.on_fire_enqueued(
                stamp(second.0, second.1, 0, 1),
                Some(owner),
                second.2,
                16,
                1,
                Some((0, 2)),
            );
            let queued: QueuedFireIds = [first.2, second.2].into_iter().collect();
            let mut sealed = fires(&plan(&mut policy, &queued, now));
            sealed.sort_unstable();
            assert_eq!(
                sealed,
                vec![2, 3],
                "the reused pipeline's fire completes the cohort (reuse_first={reuse_first})"
            );
            assert!(policy.cohorts.is_empty(), "a complete cohort is forgotten");
        }
    }

    /// **A STATED COHORT IS NEVER SEALED SHORT.** A guest that states a
    /// cohort of two and whose second lane is LATE — later than the submit
    /// deadline, which is what a 1024² first submit carrying ~6 MB of seeded
    /// `context` plus ~2 MB of `latents` actually is under load — gets its
    /// group whole and late, never a fire over the first lane alone. The
    /// leash used to release the cohort here and the fire that followed was
    /// an unconditioned denoise reported as a success (cos 0.535 against the
    /// golden, 0.9999 against a text-free reference).
    #[test]
    fn a_stated_cohort_is_never_sealed_short() {
        let leash = Duration::from_millis(50);
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(leash);
        let owner = pid();
        let (image, text) = (pid(), pid());
        let now = Instant::now();
        policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 2)));
        let queued: QueuedFireIds = [1].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, now),
            FramePlan::Hold(_)
        ));
        // Past the leash, and past it again: the deadline that drops a
        // silent LANE from a boundary may never drop a stated group's
        // member, because that changes the answer and not the schedule.
        for late in [leash + Duration::from_millis(1), leash * 20] {
            match plan(&mut policy, &queued, now + late) {
                FramePlan::Hold(_) => {}
                plan => panic!("the image lane must not fire without its group, got {plan:?}"),
            }
        }
        // The straggler lands, and the whole group fires in one frame.
        let landed = now + leash * 20;
        policy.on_fire_enqueued(stamp(text, 0, 0, 1), Some(owner), 2, 16, 1, Some((7, 2)));
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, landed));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![1, 2], "the late cohort still seals whole");
    }

    /// A cohort that never completes is a guest bug, and the runtime says so
    /// by name: past the SILENCE TIMEOUT (the verdict clock — the submit
    /// deadline is a leash and can only cost latency) the whole request is
    /// terminated with what the guest got wrong, and the lane that did
    /// arrive never fires.
    #[test]
    fn a_cohort_that_never_completes_kills_its_request_by_name() {
        let silence = Duration::from_secs(30);
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_millis(50))
            .with_silence_timeout(silence);
        let owner = pid();
        let lone = pid();
        let now = Instant::now();
        policy.on_fire_enqueued(stamp(lone, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 2)));
        let queued: QueuedFireIds = [1].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, now),
            FramePlan::Hold(_)
        ));

        // A hair past the timeout, which is measured from the arrival and
        // not from the test's clock.
        let verdict = plan(&mut policy, &queued, now + silence + Duration::from_secs(1));
        let FramePlan::Terminate(doomed) = &verdict else {
            panic!("a cohort short past the silence timeout ends the request, got {verdict:?}");
        };
        assert_eq!(
            doomed,
            &vec![(
                owner,
                Doom::CohortNeverComplete {
                    group: 7,
                    expected: 2,
                    arrived: 1,
                },
            )]
        );
        let said = doomed[0].1.to_string();
        assert!(
            said.contains("attention group 7")
                && said.contains("2 live forward passes")
                && said.contains("only 1")
                && said.contains("must submit into the group's next frame"),
            "the error names the group, the tally, and the guest's mistake: {said}"
        );
        // Condemned once, and still never fired: the kill is in flight, and
        // the lane must not slip into a partition on its way out.
        match plan(&mut policy, &queued, now + silence * 3) {
            FramePlan::Hold(_) => {}
            plan => panic!("a condemned cohort neither repeats nor fires, got {plan:?}"),
        }
    }

    /// **THE SUBMIT LEASH MUST NOT SPLIT A GROUP IN STEADY STATE EITHER.**
    /// The cohort is forgotten once it completes, and the leash — whose
    /// whole job is to drop a silent member so the boundary can seal — is
    /// exactly the thing that would fire half a group on the frame after.
    /// Both of `group_short`'s rules refuse: the second frame re-gathers a
    /// cohort, and the leashed sibling still owes its group. It drops the
    /// lane from the boundary (the fleet keeps moving) and the groupmate
    /// still does not fire without it.
    #[test]
    fn a_group_never_fires_short_when_the_leash_drops_a_member() {
        let leash = Duration::from_millis(50);
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(leash);
        let owner = pid();
        let (image, text) = (pid(), pid());
        let now = Instant::now();
        // First frame: the cohort gathers and the pair fires.
        policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 2)));
        policy.on_fire_enqueued(stamp(text, 0, 0, 1), Some(owner), 2, 16, 1, Some((7, 2)));
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        assert_eq!(fires(&plan(&mut policy, &queued, now)).len(), 2);
        policy.on_frame_retired([image, text]);
        assert!(policy.cohorts.is_empty(), "steady state keeps no cohort");

        // Second frame: only the image lane resubmits, and stays alone past
        // the leash.
        policy.on_fire_enqueued(stamp(image, 1, 0, 1), Some(owner), 3, 8, 1, Some((7, 2)));
        let queued: QueuedFireIds = [3].into_iter().collect();
        for late in [Duration::ZERO, leash + Duration::from_millis(1), leash * 10] {
            match plan(&mut policy, &queued, now + late) {
                FramePlan::Hold(_) => {}
                plan => panic!("half a group must never fire, got {plan:?}"),
            }
        }
        policy.on_fire_enqueued(stamp(text, 1, 0, 1), Some(owner), 4, 16, 1, Some((7, 2)));
        let queued: QueuedFireIds = [3, 4].into_iter().collect();
        let mut sealed = fires(&plan(&mut policy, &queued, now + leash * 10));
        sealed.sort_unstable();
        assert_eq!(sealed, vec![3, 4]);
    }

    /// A gathering cohort holds ITS OWN lanes, never the gate: an unrelated
    /// lane fires while the group is still assembling, so a guest's late
    /// sibling costs its own request and not the fleet. (This is what makes
    /// the leash unnecessary — and therefore removable — above.)
    #[test]
    fn a_gathering_cohort_holds_its_own_lanes_and_not_the_fleet() {
        let mut policy = FramePolicy::new(1, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_millis(50));
        let owner = pid();
        let grouped = pid();
        let stranger = pid();
        let now = Instant::now();
        policy.on_fire_enqueued(stamp(grouped, 0, 0, 1), Some(owner), 1, 8, 1, Some((7, 2)));
        policy.on_fire_enqueued(stamp(stranger, 0, 0, 1), Some(stranger), 2, 8, 1, None);
        let queued: QueuedFireIds = [1, 2].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, now)),
            vec![2],
            "the ungrouped lane fires; the grouped one waits for its group"
        );
    }

    /// The row/token budgets partition a boundary, and a group must not be
    /// what they cut: both lanes of a group wait for a partition that holds
    /// them both rather than firing one now and one next.
    #[test]
    fn a_group_is_never_split_across_two_partitions() {
        // First-fit runs in lane-id order, so the ids are pinned: the
        // stranger's 2 rows are placed first, and then only ONE of the
        // group's two 4-row lanes fits the 8-row wave. First-fit used to
        // take it and defer the other.
        let mut policy = FramePolicy::new(1, 8, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_millis(50));
        let owner = ProcessId::from_u128(9);
        let stranger = ProcessId::from_u128(1);
        let (image, text) = (ProcessId::from_u128(2), ProcessId::from_u128(3));
        let now = Instant::now();
        policy.on_fire_enqueued(stamp(stranger, 0, 0, 1), Some(stranger), 1, 2, 2, None);
        policy.on_fire_enqueued(stamp(image, 0, 0, 1), Some(owner), 2, 4, 4, Some((7, 2)));
        policy.on_fire_enqueued(stamp(text, 0, 0, 1), Some(owner), 3, 4, 4, Some((7, 2)));
        let queued: QueuedFireIds = [1, 2, 3].into_iter().collect();
        assert_eq!(
            fires(&plan(&mut policy, &queued, now)),
            vec![1],
            "the group does not fit beside the stranger, so it takes the next partition whole"
        );
        let mut group_frame = fires(&plan(&mut policy, &queued, now));
        group_frame.sort_unstable();
        assert_eq!(group_frame, vec![2, 3], "the group's lanes ride one frame");
    }

    /// Wait-all regression: an incomplete lane blocks the seal (the
    /// watchdog reports, never evicts); once it completes, the epoch seals dense.
    #[test]
    fn incomplete_lane_holds_the_seal_until_it_completes() {
        // Pins the strict rule (the default; stated so the test does not
        // depend on it).
        let mut policy = FramePolicy::new(2, 64, 4096, None)
            .with_seal_mode_ready(false)
            .with_submit_deadline(Duration::from_secs(86_400));
        let (fast, slow) = (pid(), pid());
        policy.on_fire_enqueued(stamp(fast, 0, 0, 2), Some(fast), 1, 1, 1, None);
        policy.on_fire_enqueued(stamp(fast, 0, 1, 2), Some(fast), 2, 1, 1, None);
        // `slow` declared 2 fires but only one arrived: a missing member.
        policy.on_fire_enqueued(stamp(slow, 0, 0, 2), Some(slow), 3, 1, 1, None);

        let queued: QueuedFireIds = [1, 2, 3].into_iter().collect();
        let t0 = Instant::now();
        match plan(&mut policy, &queued, t0) {
            // A blocked gather re-looks at the poll cadence, not the
            // watchdog's (report-only, an order of magnitude further out).
            FramePlan::Hold(hold) => {
                assert_eq!(hold, Duration::from_micros(GATHER_POLL_US));
                assert!(hold < Duration::from_micros(STRICT_WATCHDOG_US));
            }
            plan => panic!("wait-all must hold for the incomplete lane, got {plan:?}"),
        }
        // Long past the watchdog: it re-arms and reports; it never fires.
        match plan(&mut policy, &queued, t0 + Duration::from_secs(60)) {
            FramePlan::Hold(_) => {}
            plan => panic!("the watchdog reports, it must not fire: got {plan:?}"),
        }

        // The straggler completes: one dense epoch, both lanes' slot-0.
        policy.on_fire_enqueued(stamp(slow, 0, 1, 2), Some(slow), 4, 1, 1, None);
        let queued: QueuedFireIds = [1, 2, 3, 4].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("all lanes ready: the epoch must seal");
        };
        assert_eq!(waves[0].len(), 2, "dense wave 0 holds BOTH lanes");
        assert!(waves[0].contains(&1) && waves[0].contains(&3));
    }

    /// A sealed frame dispatches whole — every wave in slot order in one
    /// plan — and the policy tracks nothing afterwards.
    #[test]
    fn sealed_frame_dispatches_whole_and_frames_overlap() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 2), Some(a), 50, 1, 1, None);
        policy.on_fire_enqueued(stamp(a, 0, 1, 2), Some(a), 51, 1, 1, None);
        let queued: QueuedFireIds = [50, 51].into_iter().collect();
        let FramePlan::Dispatch(frame0) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("expected lane a's whole frame");
        };
        assert_eq!(frame0, vec![vec![50], vec![51]]);
        // Mid-execution (worker in flight), a straggler submits its first
        // frame and lane a its next: the wait-all gate holds, so f+1 seals
        // now and dispatches whole behind the executing frame.
        policy.on_fire_enqueued(stamp(b, 0, 0, 2), Some(b), 60, 1, 1, None);
        policy.on_fire_enqueued(stamp(b, 0, 1, 2), Some(b), 61, 1, 1, None);
        policy.on_fire_enqueued(stamp(a, 1, 0, 2), Some(a), 52, 1, 1, None);
        policy.on_fire_enqueued(stamp(a, 1, 1, 2), Some(a), 53, 1, 1, None);
        let queued: QueuedFireIds = [52, 53, 60, 61].into_iter().collect();
        let FramePlan::Dispatch(merged) =
            policy.plan_dispatch(&queued, &HashSet::new(), true, Instant::now())
        else {
            panic!("the overlapped next frame must seal and dispatch whole");
        };
        assert_eq!(merged[0].len(), 2, "wave 0 must hold BOTH lanes");
        assert!(merged[0].contains(&52) && merged[0].contains(&60));
        assert_eq!(merged[1].len(), 2);
        assert!(merged[1].contains(&53) && merged[1].contains(&61));
    }

    /// A fire can overtake the planner's suspend broadcast: the lane must
    /// not rejoin, but its already-arrived slot must still seal (its lease
    /// depends on it).
    #[test]
    fn a_fire_racing_the_suspend_seals_alone_without_rejoining_the_wait_set() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (victim, healthy) = {
            let (x, y) = (pid(), pid());
            if x < y { (x, y) } else { (y, x) }
        };
        policy.on_fire_enqueued(stamp(victim, 0, 0, 2), Some(victim), 100, 1, 1, None);
        policy.on_fire_enqueued(stamp(victim, 0, 1, 2), Some(victim), 101, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 0, 0, 2), Some(healthy), 102, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 0, 1, 2), Some(healthy), 103, 1, 1, None);
        let queued: QueuedFireIds = [100, 101, 102, 103].into_iter().collect();
        assert!(matches!(
            plan(&mut policy, &queued, Instant::now()),
            FramePlan::Dispatch(_)
        ));

        // The planner evicts the victim, then its slot 0 for the next frame
        // lands (already past the eviction fence); slot 1 cannot follow.
        policy.on_process_suspend(victim);
        policy.on_fire_enqueued(stamp(victim, 1, 0, 2), Some(victim), 200, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 1, 0, 2), Some(healthy), 300, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 1, 1, 2), Some(healthy), 301, 1, 1, None);
        assert!(
            !policy.lanes[&victim].awaited,
            "a suspended owner's arrival must not rejoin the wait-set"
        );

        let queued: QueuedFireIds = [200, 300, 301].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("the boundary must seal without waiting for the victim");
        };
        assert_eq!(
            waves[0],
            vec![200, 300],
            "the victim's stranded slot seals too — that lease has to drain"
        );
        assert_eq!(waves[1], vec![301]);

        // Post-restore, the late slot stands alone rather than re-forming an
        // unsatisfiable 2-slot frame under a seq that already sealed.
        policy.on_process_resume(victim);
        policy.on_fire_enqueued(stamp(victim, 1, 1, 2), Some(victim), 201, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 2, 0, 2), Some(healthy), 302, 1, 1, None);
        policy.on_fire_enqueued(stamp(healthy, 2, 1, 2), Some(healthy), 303, 1, 1, None);
        let queued: QueuedFireIds = [201, 302, 303].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("the late slot must seal");
        };
        assert_eq!(waves[1], vec![201, 303], "the late slot keeps its wave");

        // Fully rejoined: the next boundary waits for the victim again.
        policy.on_fire_enqueued(stamp(victim, 2, 0, 2), Some(victim), 400, 1, 1, None);
        policy.on_fire_enqueued(stamp(victim, 2, 1, 2), Some(victim), 401, 1, 1, None);
        assert!(
            policy.lanes[&victim].awaited,
            "a resumed process rejoins on its next frame"
        );
    }

    /// The same stranding without the planner: a KV allocation park closes
    /// a lane mid-frame, and the submitted slot must still seal.
    #[test]
    fn a_lane_parked_mid_frame_seals_what_it_submitted() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 2), Some(lane), 10, 1, 1, None);
        policy.on_lane_leave(lane, Some(lane), false);
        let queued: QueuedFireIds = [10].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("a parked lane's submitted slot must still seal");
        };
        assert_eq!(waves[0], vec![10]);
        assert!(waves[1].is_empty());
    }

    #[test]
    fn truncated_frame_seals_with_submitted_fires_only() {
        let mut policy = FramePolicy::new(4, 64, 4096, None);
        let lane = pid();
        policy.on_fire_enqueued(stamp(lane, 0, 0, 4), Some(lane), 30, 1, 1, None);
        policy.on_fire_enqueued(stamp(lane, 0, 1, 4), Some(lane), 31, 1, 1, None);
        // Host submit failed after 2 slots.
        policy.on_frame_truncated(lane, 0, 2);
        let queued: QueuedFireIds = [30, 31].into_iter().collect();
        let FramePlan::Dispatch(waves) = plan(&mut policy, &queued, Instant::now()) else {
            panic!("truncated frame must still seal");
        };
        assert_eq!(waves[0], vec![30]);
        assert_eq!(waves[1], vec![31]);
        assert!(waves[2].is_empty() && waves[3].is_empty());
    }

    /// Graceful close is the only way a straggler stops being awaited: the
    /// lane leaves the wait-set immediately and the fleet seals without it.
    #[test]
    fn graceful_close_releases_the_wait() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let (a, b) = (pid(), pid());
        policy.on_fire_enqueued(stamp(a, 0, 0, 1), Some(a), 90, 1, 1, None);
        policy.on_fire_enqueued(stamp(b, 0, 0, 1), Some(b), 91, 1, 1, None);
        let queued: QueuedFireIds = [90, 91].into_iter().collect();
        let bootstrap = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&bootstrap).len(), 2);

        // b resubmits; a does not, so the gather blocks on a.
        policy.on_fire_enqueued(stamp(b, 1, 0, 1), Some(b), 92, 1, 1, None);
        let queued: QueuedFireIds = [92].into_iter().collect();
        match plan(&mut policy, &queued, Instant::now()) {
            FramePlan::Hold(_) => {}
            plan => panic!("the gather must block on lane a, got {plan:?}"),
        }
        policy.on_lane_leave(a, None, false);
        let next = plan(&mut policy, &queued, Instant::now());
        assert_eq!(fires(&next), vec![92]);
    }

    /// Regression (fleet-wide stall): every admission notifies now, so a
    /// release consumed by anyone drains the balance and a later bystander
    /// inherits no phantom hold.
    #[test]
    fn consumed_release_leaves_no_phantom_hold_for_bystander() {
        let mut policy = FramePolicy::new(2, 64, 4096, None);
        let executing = pid();
        let bystander = pid();
        policy.on_bind_enqueued(Some(executing));
        policy.on_bind_completed(Some(executing));
        policy.on_execution_slot_consumed(executing);
        policy.on_fire_enqueued(stamp(executing, 0, 0, 1), Some(executing), 95, 1, 1, None);
        // A retirement's release is consumed by an uncontended admission
        // elsewhere, while a later process binds and stages.
        policy.on_execution_slot_released(pid());
        policy.on_execution_slot_consumed(executing);
        policy.on_bind_enqueued(Some(bystander));
        let queued: QueuedFireIds = [95].into_iter().collect();
        assert!(
            matches!(
                plan(&mut policy, &queued, Instant::now()),
                FramePlan::Dispatch(_)
            ),
            "a drained release must not hold for a staged bystander"
        );
    }

    // The contributed-and-owed gate relaxation (`PIE_GATE_CONTRIBUTED`).
    // Pins `gate_verdict`: `awaited && frames.is_empty() && owes`.
}
