//! The NCCL communicator a tensor-parallel rank fires its collectives on.
//!
//! One [`Comm`] per rank, opened together by [`open_group`](crate::open_group)
//! and carried onto that rank's kernel context
//! (`kernels_cuda::jit::Ctx::with_comm`). The collectives themselves live in
//! `kernels-cuda`; this module only owns the handle's life: opened once, on a
//! thread bound to the rank's device, and destroyed idle — never aborted,
//! since an abort followed by a destroy was observed to spin forever.

use core::ffi::c_void;
use std::fmt;

use crate::error::{Fault, Result};

/// Which transport a group's ranks talk over — `[engine] nccl_transport`.
///
/// NCCL takes this as an environment variable and there is no other door into
/// it, so pie WRITES `NCCL_P2P_DISABLE` on its behalf. What pie may not do is
/// READ it: a shell that decided its own default from the environment took a
/// knob from nowhere (article 9), so the policy is stated here and the write
/// follows from the word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transport {
    /// Shared memory: pie writes `NCCL_P2P_DISABLE=1`. **The default**, and
    /// not NCCL's — on PCIe boxes NCCL's P2P transport can wedge before it
    /// falls back (observed on a 2×L40S pair and on a 4×RTX PRO 6000 one).
    #[default]
    Shm,
    /// Peer-to-peer: pie writes `NCCL_P2P_DISABLE=0`. For a box whose links
    /// are real (NVLink), where P2P is the whole point.
    Peer,
    /// Whatever NCCL decides for itself: pie writes nothing, and NCCL reads
    /// its own environment. The door for a deployment that states an
    /// `NCCL_*` policy in the shell it launches pie from — before this word
    /// existed, an already-set `NCCL_P2P_DISABLE` was deferred to silently,
    /// which is the same intent said out loud.
    Nccl,
}

impl std::str::FromStr for Transport {
    type Err = String;

    /// `shm`, `peer`, or `nccl`; anything else refuses by name.
    fn from_str(word: &str) -> std::result::Result<Transport, String> {
        match word {
            "shm" => Ok(Transport::Shm),
            "peer" => Ok(Transport::Peer),
            "nccl" => Ok(Transport::Nccl),
            other => Err(format!(
                "`{other}` does not name an NCCL transport; the spellings are \
                 `shm` (P2P off, the default), `peer` (P2P on), and `nccl` \
                 (whatever NCCL's own environment says)"
            )),
        }
    }
}

/// `ncclUniqueId`: the 128 bytes every rank of one group opens with.
#[derive(Clone)]
pub struct Id(pub [u8; 128]);

impl Id {
    /// A fresh group identity, from rank 0's process, over `transport`.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] with no runtime, [`Fault::Device`] when NCCL
    /// refused.
    pub fn new(transport: Transport) -> Result<Id> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
            // Here rather than in `open`: this runs once, on the opener's
            // own thread, before a rank thread exists — and `set_var` from
            // the rank threads, which open concurrently, would be a data
            // race on the environment.
            transport_defaults(transport);
            let mut id = nccl::ncclUniqueId { internal: [0; 128] };
            // SAFETY: a live out-parameter of the exact type NCCL writes.
            let code = unsafe { nccl::ncclGetUniqueId(&raw mut id) };
            answered("ncclGetUniqueId", code)?;
            Ok(Id(id.internal.map(|byte| byte as u8)))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = transport;
            Err(Fault::Runtimeless)
        }
    }
}

/// One rank's open communicator.
pub struct Comm {
    raw: *mut c_void,
    rank: u32,
    size: u32,
}

// SAFETY: NCCL communicators are used from any thread as long as calls on one
// communicator are not concurrent, which the group guarantees by driving each
// rank from one thread at a time.
unsafe impl Send for Comm {}
unsafe impl Sync for Comm {}

impl Comm {
    /// Opens rank `rank` of a `size`-wide group. Collective: every rank must
    /// call this at the same time with the same `id`, from a thread whose
    /// current device is that rank's — `ncclCommInitRank` blocks until the
    /// whole group has arrived.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] with no runtime, [`Fault::Device`] when NCCL
    /// refused.
    pub fn open(id: &Id, rank: u32, size: u32) -> Result<Comm> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
            let unique = nccl::ncclUniqueId {
                internal: id.0.map(|byte| byte as core::ffi::c_char),
            };
            let mut raw: nccl::ncclComm_t = core::ptr::null_mut();
            // SAFETY: a live out-parameter; `unique` is by value, as the
            // binding declares it.
            let code = unsafe {
                nccl::ncclCommInitRank(
                    &raw mut raw,
                    size as core::ffi::c_int,
                    unique,
                    rank as core::ffi::c_int,
                )
            };
            answered("ncclCommInitRank", code)?;
            Ok(Comm {
                raw: raw.cast(),
                rank,
                size,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (id, rank, size);
            Err(Fault::Runtimeless)
        }
    }

    /// The `ncclComm_t`, for the kernel context.
    #[must_use]
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// Abort the communicator: every collective pending or later issued on
    /// it returns an error instead of waiting for peers that will never
    /// arrive. The group's answer to a rank that refused before its
    /// collective — its peers come back out of NCCL with an error rather
    /// than sitting there forever. Idempotent; a closed handle is left
    /// alone. The communicator is never destroyed after this (see `Drop`).
    pub fn abort(&self) {
        #[cfg(feature = "cuda")]
        {
            use cudarc::nccl::sys as nccl;
            if self.raw.is_null() {
                return;
            }
            // SAFETY: a live communicator this group opened; NCCL allows an
            // abort from any thread while other threads are inside calls on
            // the same communicator — that is what it is for.
            let _ = unsafe { nccl::ncclCommAbort(self.raw.cast()) };
        }
    }

    #[must_use]
    pub fn rank(&self) -> u32 {
        self.rank
    }

    #[must_use]
    pub fn size(&self) -> u32 {
        self.size
    }
}

impl Drop for Comm {
    /// The communicator is left to the process. `ncclCommDestroy` waits for
    /// the whole clique and for anything still queued on the rank's stream;
    /// a group torn down after a refused boot has ranks whose peers never
    /// arrived, and the destroy then spins forever (observed on both this
    /// engine and the driver before it). A deployment opens one group for
    /// its life, so the leak is bounded by the process.
    fn drop(&mut self) {
        self.raw = core::ptr::null_mut();
    }
}

impl fmt::Debug for Comm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Comm")
            .field("rank", &self.rank)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Comm {
    fn eq(&self, other: &Comm) -> bool {
        self.raw == other.raw
    }
}

/// State the group's transport in the only place NCCL reads one: its own
/// environment. `[engine] nccl_transport` decides what is written, and
/// [`Transport::Nccl`] writes nothing at all.
///
/// **This is a write, not a read.** Article 9 is about a shell taking a knob
/// from the environment; handing one to a library that has no other door is
/// the opposite direction, and the word itself came off the boot document.
/// What used to stand here read `NCCL_P2P_DISABLE` first and deferred to a
/// stated one — a deployment that wants that keeps it by stating
/// `nccl_transport = "nccl"`, which says the same thing where a reader can
/// see it.
///
/// Called from [`Id::new`] alone, which is the group's first NCCL call and
/// runs on the opener's thread before any rank thread starts: writing the
/// environment from the rank threads, which open concurrently, would be a
/// race.
#[cfg(feature = "cuda")]
fn transport_defaults(transport: Transport) {
    let disable_p2p = match transport {
        Transport::Shm => "1",
        Transport::Peer => "0",
        // Not our environment to write.
        Transport::Nccl => return,
    };
    // SAFETY: called before any communicator exists, from the group opener,
    // on the thread that starts the rank threads.
    unsafe { std::env::set_var("NCCL_P2P_DISABLE", disable_p2p) };
}

#[cfg(feature = "cuda")]
fn answered(call: &'static str, code: cudarc::nccl::sys::ncclResult_t) -> Result<()> {
    if code == cudarc::nccl::sys::ncclResult_t::ncclSuccess {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: code as i32,
        })
    }
}
