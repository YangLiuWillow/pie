//! `Group`: one tensor-parallel deployment behind [`Engine`], made of one
//! [`Cuda`] shell per rank.
//!
//! Traces are SPMD — every rank runs the same plan over its own band of the
//! weights and meets the others in the plan's collectives — so a group is
//! the same verbs, issued to every rank at once. Each verb runs on one
//! thread per rank (a rank's device is bound per thread), which is what lets
//! a fire's collectives complete: rank 0's launch would otherwise wait on a
//! peer that has not been asked yet. Rank 0 answers; the followers' answers
//! are checked for errors and for agreeing on the identities they mint.
//!
//! The runtime sees one engine: one load, one frame numbering, one
//! completion sink (rank 0's; a follower's completions are its own
//! business), one set of channels read from rank 0.
//!
//! # A rank that fails does not strand its peers
//!
//! A collective blocks until every rank arrives. A rank that refuses a verb
//! before its collective (a bad address, a refused dispatch) never arrives,
//! and its peers would sit in NCCL forever — the open item of
//! `.wiki/tp-verification.md`. So every verb is a bounded wait: the ranks
//! answer on a channel, the first refusal ABORTS every communicator
//! (`ncclCommAbort` makes the peers' pending collectives return), and a rank
//! that answers nothing within the verb's wait is given up on the same way.
//! Either way the group is POISONED: the refusal is the answer, every later
//! verb refuses by name, and teardown leaves the stuck ranks to the process
//! (a destroy would wait on them). The communicators are opened with the
//! same bound.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::DeviceFacts;
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::MediaEncode;
use engine::fire::{FrameSubmission, FrameTicket, Step};
use engine::load::{LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, KvHandle, StateCopy};
use eta_ir::container::HostRole;

use crate::api::{ClassifyFor, ContractFor, Cuda, DeviceBoot, World};
use crate::comm::{Comm, Id};

/// How long the communicators may take to find each other. Rank threads
/// past it are given up on (they hold nothing but their own device).
const INIT_WAIT: Duration = Duration::from_secs(120);
/// How long a load may take on every rank — the arming pass fires every
/// rung of the lattice, a matter of minutes for a wide model.
const LOAD_WAIT: Duration = Duration::from_secs(3600);
/// How long any other verb may take on every rank.
const VERB_WAIT: Duration = Duration::from_secs(600);
/// After an abort, how long the aborted ranks get to come back before they
/// are left to the process.
const GRACE: Duration = Duration::from_secs(30);

/// The ranks of one tensor-parallel group, as one engine.
pub struct Group {
    /// One shell per rank, each behind its own lock so a verb's rank thread
    /// owns it for the verb — and a rank that never came back keeps it
    /// locked, which is how a poisoned group refuses fast.
    ranks: Vec<Arc<Mutex<Cuda>>>,
    ordinals: Vec<i32>,
    /// Every rank's communicator, for the abort that unblocks the peers of
    /// a rank that refused.
    comms: Vec<Arc<Comm>>,
    /// Set by the first refusal or timeout; every later verb refuses by name.
    poisoned: Arc<Mutex<Option<String>>>,
    /// Rank 0's device facts, read once at open. Kept beside the shells
    /// rather than through them: [`Engine::device_facts`] borrows for the
    /// group's life, and a shell held by a verb's rank thread cannot lend
    /// one out. The facts are "what the machine is. Stable for the life of
    /// the process", so one read is the whole answer.
    facts: Option<DeviceFacts>,
}

/// Opens one CUDA shell per boot as a tensor-parallel group: rank `i` is
/// `boots[i]`, its width is `boots.len()`, and every rank's communicator is
/// opened here, together, before any shell exists.
///
/// # Errors
///
/// Fewer than two boots (one rank is [`open`](crate::open)), a knob out of
/// range, a communicator NCCL refused to open, or a rank that did not join
/// the group within [`INIT_WAIT`].
pub fn open_group(
    boots: Vec<DeviceBoot>,
    contract_for: ContractFor,
    classify_for: ClassifyFor,
) -> Result<Group, String> {
    let size = boots.len();
    if size < 2 {
        return Err(format!(
            "a tensor-parallel group is two or more ranks; {size} boot(s) were given \
             (one device opens through `open`)"
        ));
    }
    let size32 = u32::try_from(size).map_err(|_| "more ranks than a u32 counts".to_string())?;
    // Rank 0's word: every rank of a group is booted from one `[engine]`
    // table, and NCCL's environment is the process's, not a rank's.
    let id = Id::new(boots[0].knobs.nccl_transport).map_err(|fault| fault.to_string())?;
    // Every rank opens its communicator on a thread bound to its own device;
    // `ncclCommInitRank` returns only once the whole group has arrived, so
    // the opens run concurrently — and a rank that never arrives would hold
    // the rest in NCCL, so the opens are waited on with a bound and a thread
    // past it is left to the process.
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<Comm, String>)>();
    for (rank, boot) in boots.iter().enumerate() {
        let tx = tx.clone();
        let id = id.clone();
        let ordinal = boot.ordinal;
        std::thread::Builder::new()
            .name(format!("nccl-init-{rank}"))
            .spawn(move || {
                let opened = crate::device::ctx::bind_thread(ordinal)
                    .map_err(|fault| fault.to_string())
                    .and_then(|()| {
                        Comm::open(&id, rank as u32, size32).map_err(|fault| fault.to_string())
                    });
                let _ = tx.send((rank, opened));
            })
            .map_err(|why| format!("rank {rank}: no thread to open its communicator on: {why}"))?;
    }
    drop(tx);
    let mut comms: Vec<Option<Comm>> = (0..size).map(|_| None).collect();
    let deadline = std::time::Instant::now() + INIT_WAIT;
    let mut refusal: Option<String> = None;
    for _ in 0..size {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match rx.recv_timeout(left) {
            Ok((rank, Ok(comm))) => comms[rank] = Some(comm),
            Ok((rank, Err(why))) => {
                refusal.get_or_insert(format!("rank {rank} (cuda:{}): {why}", boots[rank].ordinal));
            }
            Err(_) => {
                let missing: Vec<String> = comms
                    .iter()
                    .enumerate()
                    .filter(|(_, comm)| comm.is_none())
                    .map(|(rank, _)| format!("rank {rank} (cuda:{})", boots[rank].ordinal))
                    .collect();
                refusal.get_or_insert(format!(
                    "{} did not join the NCCL group within {}s; the group is {size} \
                     rank(s) over devices {:?} and every rank must open at once",
                    missing.join(", "),
                    INIT_WAIT.as_secs(),
                    boots.iter().map(|boot| boot.ordinal).collect::<Vec<_>>()
                ));
                break;
            }
        }
    }
    if let Some(why) = refusal {
        // The ranks that did open are aborted so a peer still inside
        // `ncclCommInitRank` returns rather than waits for one that never comes.
        for comm in comms.iter().flatten() {
            comm.abort();
        }
        return Err(why);
    }
    let mut ranks = Vec::with_capacity(size);
    let mut ordinals = Vec::with_capacity(size);
    let mut held = Vec::with_capacity(size);
    for (rank, (mut boot, comm)) in boots.into_iter().zip(comms).enumerate() {
        let comm = Arc::new(comm.expect("every rank answered above"));
        boot.world = World {
            rank: rank as u32,
            size: size32,
        };
        boot.comm = Some(Arc::clone(&comm));
        ordinals.push(boot.ordinal);
        held.push(comm);
        ranks.push(Arc::new(Mutex::new(crate::boot::open(
            boot,
            contract_for,
            classify_for,
        )?)));
    }
    let facts = ranks[0]
        .lock()
        .ok()
        .and_then(|rank| rank.device_facts().cloned());
    Ok(Group {
        ranks,
        ordinals,
        comms: held,
        poisoned: Arc::new(Mutex::new(None)),
        facts,
    })
}

impl Group {
    /// How many ranks this group is.
    #[must_use]
    pub fn size(&self) -> usize {
        self.ranks.len()
    }

    /// The refusal that poisoned this group, if one did.
    fn poison(&self) -> Option<String> {
        self.poisoned
            .lock()
            .map(|held| held.clone())
            .unwrap_or(None)
    }

    /// Poison the group with `why` and abort every communicator, so a rank
    /// parked in a collective returns.
    fn poison_with(&self, why: &str) {
        if let Ok(mut held) = self.poisoned.lock()
            && held.is_none()
        {
            *held = Some(why.to_string());
        }
        for comm in &self.comms {
            comm.abort();
        }
    }

    /// Runs `verb` on every rank at once, each on a thread bound to that
    /// rank's device, and returns every rank's answer in rank order. The
    /// first rank that refused speaks for the group — and poisons it: its
    /// peers, parked in the collective it never reached, are aborted out
    /// of NCCL and their answers discarded. A rank that answers nothing
    /// within `wait` is treated the same way, its thread left holding its
    /// shell.
    fn each_within<R, F>(&mut self, wait: Duration, verb: F) -> EngineResult<Vec<R>>
    where
        R: Send + 'static,
        F: Fn(&mut Cuda) -> EngineResult<R> + Send + Sync + 'static,
    {
        if let Some(why) = self.poison() {
            return Err(Error::Device(format!(
                "this tensor-parallel group is poisoned: {why}"
            )));
        }
        let verb = Arc::new(verb);
        let (tx, rx) = std::sync::mpsc::channel::<(usize, EngineResult<R>)>();
        for (rank, (shell, &ordinal)) in self.ranks.iter().zip(&self.ordinals).enumerate() {
            let shell = Arc::clone(shell);
            let verb = Arc::clone(&verb);
            let tx = tx.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("tp-rank-{rank}"))
                .spawn(move || {
                    let answer = (|| {
                        crate::device::ctx::bind_thread(ordinal)
                            .map_err(|fault| Error::Device(fault.to_string()))?;
                        let mut shell = shell.try_lock().map_err(|_| {
                            Error::Device(format!(
                                "rank {rank} is still held by a verb that never came back"
                            ))
                        })?;
                        shell.bind_thread()?;
                        verb(&mut shell)
                    })();
                    let _ = tx.send((rank, answer));
                });
            if let Err(why) = spawned {
                self.poison_with(&format!("rank {rank}: no thread to run it on: {why}"));
                return Err(Error::Device(format!(
                    "rank {rank}: no thread to run the verb on: {why}"
                )));
            }
        }
        drop(tx);
        let size = self.ranks.len();
        let mut answers: Vec<Option<R>> = (0..size).map(|_| None).collect();
        let mut refused: Option<Error> = None;
        let deadline = std::time::Instant::now() + wait;
        let mut heard = 0usize;
        while heard < size {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            match rx.recv_timeout(left) {
                Ok((rank, Ok(value))) => {
                    answers[rank] = Some(value);
                    heard += 1;
                }
                Ok((rank, Err(error))) => {
                    heard += 1;
                    // Said here as well as returned: the group's teardown
                    // waits on every rank, so a refusal is otherwise the
                    // last thing read.
                    eprintln!("engine-cuda: tensor-parallel rank {rank} refused: {error}");
                    if refused.is_none() {
                        self.poison_with(&format!("rank {rank} refused: {error}"));
                        refused = Some(error);
                        // The peers come back through the abort; give them
                        // the grace, not the whole wait.
                        let grace = std::time::Instant::now() + GRACE;
                        while heard < size {
                            let left = grace.saturating_duration_since(std::time::Instant::now());
                            match rx.recv_timeout(left) {
                                Ok(_) => heard += 1,
                                Err(_) => break,
                            }
                        }
                        break;
                    }
                }
                Err(_) => {
                    let missing: Vec<usize> = answers
                        .iter()
                        .enumerate()
                        .filter(|(_, answer)| answer.is_none())
                        .map(|(rank, _)| rank)
                        .collect();
                    let why = format!(
                        "rank(s) {missing:?} of the tensor-parallel group answered nothing \
                         within {}s; the group is aborted and poisoned",
                        wait.as_secs()
                    );
                    eprintln!("engine-cuda: {why}");
                    self.poison_with(&why);
                    refused = Some(Error::Device(why));
                    break;
                }
            }
        }
        if let Some(error) = refused {
            return Err(error);
        }
        Ok(answers
            .into_iter()
            .map(|answer| answer.expect("every rank answered"))
            .collect())
    }

    /// [`each_within`](Self::each_within) at the ordinary verb's wait.
    fn each<R, F>(&mut self, verb: F) -> EngineResult<Vec<R>>
    where
        R: Send + 'static,
        F: Fn(&mut Cuda) -> EngineResult<R> + Send + Sync + 'static,
    {
        self.each_within(VERB_WAIT, verb)
    }

    /// Runs `verb` on rank `rank` alone, on this thread bound to its device.
    fn on<R, F>(&mut self, rank: usize, verb: F) -> EngineResult<R>
    where
        F: FnOnce(&mut Cuda) -> EngineResult<R>,
    {
        if let Some(why) = self.poison() {
            return Err(Error::Device(format!(
                "this tensor-parallel group is poisoned: {why}"
            )));
        }
        crate::device::ctx::bind_thread(self.ordinals[rank])
            .map_err(|fault| Error::Device(fault.to_string()))?;
        let mut shell = self.ranks[rank].try_lock().map_err(|_| {
            Error::Device(format!(
                "rank {rank} is still held by a verb that never came back"
            ))
        })?;
        shell.bind_thread()?;
        verb(&mut shell).inspect_err(|error| {
            eprintln!("engine-cuda: tensor-parallel rank {rank} refused: {error}");
        })
    }

    /// [`each`](Self::each), answering with rank 0's value.
    fn lead<R, F>(&mut self, verb: F) -> EngineResult<R>
    where
        R: Send + 'static,
        F: Fn(&mut Cuda) -> EngineResult<R> + Send + Sync + 'static,
    {
        let mut answers = self.each(verb)?;
        Ok(answers.swap_remove(0))
    }
}

impl Drop for Group {
    /// Every rank is torn down at once, each on a thread bound to its
    /// device: `ncclCommDestroy` waits for the rest of the clique, so
    /// dropping the ranks one after another would hang on the first. A
    /// poisoned group is left to the process: some rank may still be parked
    /// in a verb, and a teardown that waited on it would never end.
    fn drop(&mut self) {
        let ranks = std::mem::take(&mut self.ranks);
        let ordinals = std::mem::take(&mut self.ordinals);
        if self.poison().is_some() {
            for rank in ranks {
                std::mem::forget(rank);
            }
            return;
        }
        std::thread::scope(|scope| {
            for (rank, ordinal) in ranks.into_iter().zip(ordinals) {
                scope.spawn(move || {
                    let _ = crate::device::ctx::bind_thread(ordinal);
                    drop(rank);
                });
            }
        });
    }
}

impl Engine for Group {
    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        // Rank 0's, read once at open: a shell a verb's rank thread is
        // holding could not lend one out, and the machine does not change.
        self.facts.as_ref()
    }

    fn export_kv_handle(&self) -> Option<KvHandle> {
        // A rank's pages hold its band of the heads; a transfer plane that
        // reads one rank's handle would move half a cache.
        None
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        // Each rank binds its own device inside `each`.
        Ok(())
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        // Every rank loads at once: the warm-up fires inside a load carry
        // the plan's collectives, which need the whole group present.
        let mut answers = self.each_within(LOAD_WAIT, move |rank| rank.load(request.clone()))?;
        Ok(answers.swap_remove(0))
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        // Every rank runs the guest: a PIPELINED decode step's input token is
        // sampled by the prior step's epilogue and injected by this step's
        // prologue, so a follower without the guest would decode a placeholder
        // token. Sampling (`argmax`) is deterministic over the full,
        // replicated logits, so every rank samples the same token; only the
        // host-facing streaming (take_channel) is read from rank 0.
        let registration = registration.clone();
        self.lead(move |rank| rank.register_program(&registration))
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        // Rank 0 owns the host end and the runtime pumps its mirror. A
        // follower shares rank 0's endpoint for a host-facing ring — its
        // shadow session pulls the guest's cells out of the same pinned
        // mirror and never writes a word or cell of it — but registers its
        // OWN device-only ring (e.g. the decode `tok_in` handoff), which
        // lives on the follower's device and carries no host end to share.
        let registered = self.on(0, |rank| rank.register_channel(registration))?;
        if registration.host_role == HostRole::None {
            for rank in 1..self.ranks.len() {
                self.on(rank, |shell| {
                    shell.register_channel(registration).map(|_| ())
                })?;
            }
        } else {
            let endpoint = self
                .on(0, |rank| Ok(rank.endpoint(registration.id)))?
                .expect("rank 0 just registered this channel");
            let id = registration.id;
            for rank in 1..self.ranks.len() {
                let endpoint = endpoint.clone();
                self.on(rank, move |shell| {
                    shell.adopt_channel(id, endpoint).map(|_| ())
                })?;
            }
        }
        Ok(registered)
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        // Every rank binds its own session over the same instance id (each
        // shell's counter is deterministic); a follower binds as a shadow of
        // rank 0's (`Plane::set_shadow`, armed at load).
        let binding = binding.clone();
        self.lead(move |rank| rank.bind_instance(&binding))
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        self.lead(move |rank| rank.close_instance(id))
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.lead(move |rank| rank.close_channel(id))
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.on(0, |rank| rank.publish_channel(instance, channel, cell))
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.on(0, |rank| rank.take_channel(instance, channel))
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        // The adapter is a weight bank sharded like the rest of the stack, so
        // every rank lands its band.
        let registration = registration.clone();
        self.lead(move |rank| rank.register_adapter(&registration))
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        // Every rank runs the whole fire — the SPMD model forward, its
        // collectives, AND the guest boundaries attached to it. A follower's
        // guest is a shadow: it samples the same token off its own replicated
        // logits and feeds its own device-only `tok_in` for the next
        // pipelined step, but its host-facing rings are rank 0's, pulled and
        // never written. Rank 0's ticket is the group's; a follower's
        // host-facing readouts are never taken.
        let frame = frame.clone();
        self.each(move |rank| rank.submit(&frame))
            .map(|mut tickets| tickets.swap_remove(0))
    }

    fn settles_asynchronously(&self) -> bool {
        self.ranks[0]
            .try_lock()
            .map(|rank| rank.settles_asynchronously())
            .unwrap_or(true)
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        // One completion per step reaches the runtime: rank 0's. A follower
        // still settles its own fires; it just tells no one.
        let silent: engine::CompletionSink = Arc::new(|_, _| {});
        for (rank, shell) in self.ranks.iter().enumerate() {
            if let Ok(mut shell) = shell.try_lock() {
                shell.on_complete(if rank == 0 {
                    sink.clone()
                } else {
                    silent.clone()
                });
            }
        }
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        // Rank 0's ticket carries the readouts back; the followers settle a
        // copy so their pending frame retires with it.
        let template = ticket.clone();
        let mut settled = self.each(move |rank| {
            let mut own = template.clone();
            rank.settle_frame(&mut own)?;
            Ok(own)
        })?;
        *ticket = settled.swap_remove(0);
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        for rank in &self.ranks {
            if let Ok(mut rank) = rank.try_lock() {
                rank.expect_fire(submission);
            }
        }
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        let copy = copy.clone();
        self.lead(move |rank| rank.copy_kv(&copy))
    }

    fn copy_state(&mut self, copy: &StateCopy) -> EngineResult<()> {
        let copy = copy.clone();
        self.lead(move |rank| rank.copy_state(&copy))
    }

    fn encode(&mut self, plan: &mut MediaEncode) -> EngineResult<()> {
        // A media tower is a plan of its own, run by rank 0 alone: its
        // rows land in a channel the group reads from rank 0 anyway. This
        // shell carries no encoder today, so rank 0's answer is its own
        // refusal by name rather than the group's.
        self.on(0, |rank| rank.encode(plan))
    }

    fn disconnect(&self, message: &str) {
        for rank in &self.ranks {
            if let Ok(rank) = rank.try_lock() {
                rank.disconnect(message);
            }
        }
    }
}
