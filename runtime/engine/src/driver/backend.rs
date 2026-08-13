//! Driver specs, backend storage (the `DriverId` registry), and concrete
//! backend dispatch. Scheduler-handle lookup lives in the scheduler layer;
//! this module keeps only what the driver ABI itself owns: `DriverSpec`
//! plus the optional `DriverBackend` it's paired with.

use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};

#[cfg(feature = "driver-cuda")]
mod cuda;
mod dummy;
#[cfg(feature = "driver-metal")]
mod metal;
mod remote;

#[cfg(feature = "driver-cuda")]
pub use cuda::CudaDriver;
pub use dummy::DummyDriver;
#[cfg(feature = "driver-metal")]
pub use metal::MetalDriver;
pub use remote::{RemoteDisconnectHandle, RemoteDriver};

use crate::driver::channel::RegisteredChannel;
use crate::driver::command::{
    ChannelRegistrationPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan, ProgramRegistration,
    StateCopyPlan,
};
use crate::driver::completion::SubmissionCompletion;
use crate::driver::instance::{BoundInstance, InstanceBindingPlan};
use crate::driver::submission::FrameSubmission;

#[derive(Debug, Clone, Copy)]
pub struct SchedulerLimits {
    pub max_forward_requests: usize,
    pub max_forward_tokens: usize,
    pub max_page_refs: usize,
}

#[derive(Debug, Clone)]
pub struct DriverSpec {
    pub num_kv_pages: usize,
    pub limits: SchedulerLimits,
    pub device_geometry_port_mask: u32,
}

impl DriverSpec {
    pub fn scheduler_limits(&self) -> SchedulerLimits {
        self.limits
    }
}

/// Outcome of a frame launch post: admission is folded into the launch call
/// (ABI v14), so a post either enters the driver with one completion, or
/// reports why it cannot right now.
pub enum FrameLaunchOutcome {
    /// The frame was admitted and posted; one completion settles it.
    Launched(SubmissionCompletion),
    /// Admission is full right now; the engine re-posts later.
    Exhausted,
    /// The frame can never fit within the driver's physical budget ceiling.
    Impossible,
}

/// Rewrite a copy plan's device-side domains to `device`, or `None` when the
/// plan already agrees and no clone is needed.
///
/// A *device* domain names "wherever this backend's pages live"; the concrete
/// tag only means something to the driver that receives it. `HOST_PINNED` is
/// left alone — it names host memory on every backend, and rewriting it would
/// silently turn an offload's device↔host transfer into a same-domain copy.
#[cfg(feature = "driver-metal")]
fn localize_device_domain(
    plan: &KvCopyPlan,
    device: pie_driver_abi::PieMemoryDomain,
) -> Option<KvCopyPlan> {
    let is_device = |d: pie_driver_abi::PieMemoryDomain| {
        d != pie_driver_abi::PIE_MEMORY_DOMAIN_HOST_PINNED
    };
    let src = if is_device(plan.src_domain) { device } else { plan.src_domain };
    let dst = if is_device(plan.dst_domain) { device } else { plan.dst_domain };
    if src == plan.src_domain && dst == plan.dst_domain {
        return None;
    }
    Some(KvCopyPlan {
        src_domain: src,
        dst_domain: dst,
        src_device_ordinal: plan.src_device_ordinal,
        dst_device_ordinal: plan.dst_device_ordinal,
        src_page_ids: plan.src_page_ids.clone(),
        dst_page_ids: plan.dst_page_ids.clone(),
        cells: plan.cells.clone(),
    })
}

pub enum DriverBackend {
    Dummy(DummyDriver),
    #[cfg(feature = "driver-cuda")]
    Cuda(CudaDriver),
    #[cfg(feature = "driver-metal")]
    Metal(MetalDriver),
    Remote(RemoteDriver),
}

impl DriverBackend {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Dummy(_) => "dummy",
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(_) => "cuda",
            #[cfg(feature = "driver-metal")]
            Self::Metal(_) => "metal",
            Self::Remote(_) => "remote",
        }
    }

    pub fn dummy(
        options: pie_driver_dummy_lib::DummyDriverOptions,
    ) -> Result<(Self, pie_driver_abi::DeviceFacts)> {
        let driver = DummyDriver::new(options);
        let facts = driver.device_facts().clone();
        Ok((Self::Dummy(driver), facts))
    }

    #[cfg(feature = "driver-cuda")]
    pub fn cuda_create(config_bytes: &[u8]) -> Result<(Self, pie_driver_abi::DeviceFacts)> {
        let (driver, facts) = CudaDriver::create(config_bytes)?;
        Ok((Self::Cuda(driver), facts))
    }

    #[cfg(feature = "driver-cuda")]
    pub fn cuda_group_create(
        config_blobs: Vec<Vec<u8>>,
    ) -> Result<(Self, Vec<pie_driver_abi::DeviceFacts>)> {
        let (driver, facts) = CudaDriver::create_group(config_blobs)?;
        Ok((Self::Cuda(driver), facts))
    }

    #[cfg(feature = "driver-metal")]
    pub fn metal_create(config_bytes: &[u8]) -> Result<(Self, pie_driver_abi::DeviceFacts)> {
        let (driver, facts) = MetalDriver::create(config_bytes)?;
        Ok((Self::Metal(driver), facts))
    }

    pub fn load_model(
        &mut self,
        descs: Vec<pie_driver_abi::ModelLoadDesc>,
    ) -> Result<pie_driver_abi::DriverCapabilities> {
        match self {
            Self::Dummy(driver) => {
                let [desc] = descs.as_slice() else {
                    return Err(anyhow!(
                        "dummy model load requires exactly one descriptor, got {}",
                        descs.len()
                    ));
                };
                driver.load_model(desc)
            }
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.load_model(descs),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => {
                let [desc] = descs.as_slice() else {
                    return Err(anyhow!(
                        "metal model load requires exactly one descriptor, got {}",
                        descs.len()
                    ));
                };
                driver.load_model(desc)
            }
            Self::Remote(driver) => driver.load_model(descs),
        }
    }

    /// The backend whose kernels this driver wants the host to generate, or
    /// `None` when it generates its own (or needs none). The variant already
    /// says which native backend it is, so this needs no capability round-trip.
    fn codegen_backend(&self) -> Option<&'static str> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(_) => Some("cuda"),
            #[cfg(feature = "driver-metal")]
            Self::Metal(_) => Some("metal"),
            // The dummy driver interprets PTIR, and a remote driver's own
            // backend does its generation on the far side.
            _ => None,
        }
    }

    pub fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        // Attach whatever this driver reads and the caller did not already
        // supply. Generation is memoised per program per backend, so a
        // re-registration costs a lookup.
        let registered = crate::pipeline::program::lookup(desc.program_hash);
        let codegen_backend = self.codegen_backend();

        // The driver no longer carries an emitter, so a fused region with no
        // host source is a registration failure rather than a slower path.
        let emitted = codegen_backend
            .filter(|_| desc.emitted_kernels.is_empty())
            .and_then(|backend| {
                registered
                    .as_ref()
                    .and_then(|program| program.emitted(backend))
            });

        // The region analysis is the other half of the CUDA emitter's own
        // contract -- which regions bind, and how the kernel's intrinsic side
        // tables are laid out -- so it only means anything to a driver running
        // those kernels.
        let region_analysis = if desc.region_analysis.is_empty() && codegen_backend == Some("cuda")
        {
            registered
                .as_ref()
                .map(|program| program.region_analysis())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let owned;
        let desc = if emitted.is_some() || !region_analysis.is_empty() {
            let mut next = desc.clone();
            if let Some(emitted) = emitted {
                next.emitter_version = emitted.emitter_version;
                next.emitted_kernels = emitted
                    .kernels
                    .iter()
                    .map(|kernel| pie_driver_abi::EmittedKernel {
                        kind: kernel.kind,
                        stage_index: kernel.stage_index,
                        region_index: kernel.region_index,
                        entry_name: kernel.entry_name.clone(),
                        source: kernel.source.clone(),
                        error: kernel.error.clone(),
                    })
                    .collect();
            }
            if !region_analysis.is_empty() {
                next.region_analysis = region_analysis;
            }
            owned = next;
            &owned
        } else {
            desc
        };
        match self {
            Self::Dummy(driver) => driver.register_program(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.register_program(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.register_program(desc),
            Self::Remote(driver) => driver.register_program(desc),
        }
    }

    pub fn register_channel(
        &mut self,
        desc: &ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        match self {
            Self::Dummy(driver) => driver.register_channel(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.register_channel(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.register_channel(desc),
            Self::Remote(driver) => driver.register_channel(desc),
        }
    }

    pub fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        match self {
            Self::Dummy(driver) => driver.bind_instance(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.bind_instance(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.bind_instance(desc),
            Self::Remote(driver) => driver.bind_instance(desc),
        }
    }

    /// Post one sealed frame. Admission is folded into the call: the driver
    /// evaluates the frame-union demand and either admits (one completion
    /// settles the whole frame) or reports Exhausted/Impossible without side
    /// effects.
    pub fn launch(&mut self, desc: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        match self {
            Self::Dummy(driver) => driver.launch(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.launch(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.launch(desc),
            Self::Remote(driver) => driver.launch(desc),
        }
    }

    pub fn encode(&mut self, plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.encode(plan),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.encode(plan),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.encode(plan),
            Self::Remote(driver) => driver.encode(plan),
        }
    }

    /// Issue a KV copy, normalizing the plan's *device* memory domain to this
    /// backend's own.
    ///
    /// The scheduler builds every pre-launch copy-on-write plan with
    /// `PIE_MEMORY_DOMAIN_CUDA_DEVICE` hardcoded
    /// (`scheduler.rs`, four sites). On Metal the driver refuses it on its
    /// second guard:
    ///
    /// ```text
    /// [pie-driver-metal] copy_kv: UNSUPPORTED — only same-domain
    ///   (PIE_MEMORY_DOMAIN_METAL_SHARED) copies are supported; there is no
    ///   host-pinned swap pool in this build
    /// ```
    ///
    /// which makes `WorkingSet::fork` fail for EVERY model on Metal — the KV
    /// branching primitive the whole programmable-cache story rests on. The
    /// failure is close to undiagnosable from a guest: the fork is ordered on
    /// the pipeline and only materializes when a later fire declares a shared
    /// page writable, so the inferlet sees a poisoned channel at an unrelated
    /// prefill take and nothing anywhere names the fork.
    ///
    /// The device domain is a property of the backend, not of the caller, so it
    /// is resolved here rather than at each construction site — one place, and
    /// no call site can be missed. Only device domains are rewritten:
    /// `HOST_PINNED` is meaningful on every backend and the offload path
    /// (`scheduler/dispatch.rs`) depends on it staying put.
    pub fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.copy_kv(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_kv(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => {
                let localized;
                let desc = match localize_device_domain(
                    desc,
                    pie_driver_abi::PIE_MEMORY_DOMAIN_METAL_SHARED,
                ) {
                    Some(fixed) => {
                        localized = fixed;
                        &localized
                    }
                    None => desc,
                };
                driver.copy_kv(desc)
            }
            Self::Remote(driver) => driver.copy_kv(desc),
        }
    }

    pub fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.copy_state(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_state(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.copy_state(desc),
            Self::Remote(driver) => driver.copy_state(desc),
        }
    }

    pub fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.resize_pool(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.resize_pool(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.resize_pool(desc),
            Self::Remote(driver) => driver.resize_pool(desc),
        }
    }

    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        match self {
            Self::Dummy(driver) => driver.close_instance(id),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_instance(id),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.close_instance(id),
            Self::Remote(driver) => driver.close_instance(id),
        }
    }

    pub fn close_channel(&mut self, id: u64) -> Result<()> {
        match self {
            Self::Dummy(driver) => driver.close_channel(id),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_channel(id),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.close_channel(id),
            Self::Remote(driver) => driver.close_channel(id),
        }
    }

    pub fn export_kv_handle(&self) -> Option<pie_driver_abi::KvHandle> {
        match self {
            Self::Dummy(driver) => driver.export_kv_handle(),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.export_kv_handle(),
            #[cfg(feature = "driver-metal")]
            Self::Metal(_) => None,
            Self::Remote(_) => None,
        }
    }

    pub fn disconnect(&self, message: impl Into<String>) {
        if let Self::Remote(driver) = self {
            driver.disconnect(message);
        }
    }
}

struct DriverRegistration {
    spec: DriverSpec,
    backend: Option<DriverBackend>,
}

fn registry() -> &'static RwLock<Vec<Option<DriverRegistration>>> {
    static REGISTRY: OnceLock<RwLock<Vec<Option<DriverRegistration>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn register_driver(spec: DriverSpec) -> usize {
    let mut drivers = registry().write().unwrap();
    let id = drivers.len();
    drivers.push(Some(DriverRegistration {
        spec,
        backend: None,
    }));
    id
}

pub fn register_driver_backend(spec: DriverSpec, backend: DriverBackend) -> usize {
    let mut drivers = registry().write().unwrap();
    let id = drivers.len();
    drivers.push(Some(DriverRegistration {
        spec,
        backend: Some(backend),
    }));
    id
}

pub fn get_spec(driver_id: usize) -> Result<DriverSpec> {
    registry()
        .read()
        .unwrap()
        .get(driver_id)
        .and_then(|d| d.as_ref().map(|r| r.spec.clone()))
        .ok_or_else(|| anyhow!("unknown driver {driver_id}"))
}

pub fn take_driver_backend(driver_id: usize) -> Result<DriverBackend> {
    let mut drivers = registry().write().unwrap();
    let Some(Some(driver)) = drivers.get_mut(driver_id) else {
        return Err(anyhow!("unknown driver {driver_id}"));
    };
    driver
        .backend
        .take()
        .ok_or_else(|| anyhow!("driver {driver_id} has no backend installed"))
}

pub fn unregister_driver(driver_id: usize) -> Result<()> {
    let mut drivers = registry().write().unwrap();
    let Some(slot) = drivers.get_mut(driver_id) else {
        return Err(anyhow!("unknown driver {driver_id}"));
    };
    slot.take();
    Ok(())
}
