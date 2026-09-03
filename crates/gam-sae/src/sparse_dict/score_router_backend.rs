//! The CUDA backend shared by the atom-lane and block-lane score routers: one
//! context/stream, a per-`P` compiled-module family, and the device's shared
//! memory limit. Each router keeps its own process-wide slot (they probe under
//! different labels), but the backend they build is one type built one way.

use gam_gpu::backend_probe::CudaBackendParts;
use gam_gpu::device_cache::KeyedPtxModuleCache;
use gam_gpu::gpu_error::GpuError;

use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

pub(super) struct ScoreRouterBackend {
    pub(super) ctx: Arc<CudaContext>,
    pub(super) stream: Arc<CudaStream>,
    /// One compiled module per row width `P`, which the kernel source bakes in.
    pub(super) modules: KeyedPtxModuleCache<usize>,
    pub(super) max_shared_mem_per_block: usize,
}

impl ScoreRouterBackend {
    /// Build the backend from a probe's parts; the builder handed to
    /// `CachedBackend::get_or_probe`.
    pub(super) fn from_parts(parts: CudaBackendParts) -> Result<Self, GpuError> {
        Ok(Self {
            ctx: parts.ctx,
            stream: parts.stream,
            modules: KeyedPtxModuleCache::new(),
            max_shared_mem_per_block: gam_gpu::GpuRuntime::require()?
                .selected_device()
                .max_shared_mem_per_block,
        })
    }
}
