//! `DispatchCustomCuda for Run<'_>`: refuses cuda-plane custom ops on the metal backend.

use model_exec::{DispatchCustomCuda, KernelError};
use model_ir::{CustomCuda, Operands};

use crate::run::Run;

impl DispatchCustomCuda for Run<'_> {
    /// Cuda-plane ops never resolve on the metal backend.
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: op.name() })
    }
}

/// M0 (design D8): the voxel axis's family is CUDA-only this phase; this
/// shell refuses every member by name rather than folding.
impl model_exec::DispatchSpatial for Run<'_> {
    fn dispatch(&mut self, op: &model_ir::Spatial) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: op.name() })
    }
}

impl model_exec::DispatchProbe for Run<'_> {}
