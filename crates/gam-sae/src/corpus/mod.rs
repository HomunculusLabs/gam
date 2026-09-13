//! Streaming / out-of-core corpus driver for the SAE term (#973).
//!
//! This module is the **driver** that lets a sparse-autoencoder term fit on a
//! corpus of activations far larger than RAM, without ever materializing the
//! activations as a dense `f64` matrix. It sits *behind* the SAE term: the
//! term consumes the seam this module exposes, this module owns the streaming,
//! warm-state, scheduling and mixed-precision machinery.
//!
//! # The four pieces
//!
//! * [`shard_reader`] — an mmap-backed, bounded-prefetch reader over one or
//!   many on-disk activation shards. It defines the `v1` shard format
//!   (fixed 32-byte header + row-major `f32` payload) and yields `f64`-upcast
//!   row batches in a **deterministic global order** with stable `row_id`s,
//!   independent of OS readahead.
//! * [`warm_state`] — a disk-backed (mmap/LRU) per-row warm-state cache keyed
//!   by `(row_id, TermCollectionSpec structural hash)`. It persists each row's
//!   inner-solve seed (latent coords + active set) so re-solving the same row
//!   across outer ρ passes (or across a SIGKILL-resume) costs ~3 inner
//!   iterations instead of ~30. The structural hash is computed the same way
//!   the existing warm-start cache does (#869,
//!   `TermCollectionSpec::write_structural_shape_hash`), so distinct topologies
//!   never cross-seed.
//! * `kernels` — fused mixed-precision kernels (`dot`, `gram`, `gemv`,
//!   `gemv_t`, `cross`) that **read `f32` rows and accumulate in `f64`**, the
//!   numerical contract that keeps the streaming sums deterministic and
//!   precise despite `f32` on-disk storage.
//!
//! # The seam
//!
//! The SAE term (owned by another track, `sae_manifold.rs`) consumes exactly
//! two traits from here — re-exported below as the public seam:
//!
//! * [`CorpusRowSource`] — "give me the next deterministic batch of rows /
//!   rewind for the next ρ pass", and
//! * [`RowWarmCache`] — "give me / take back this row's inner-solve warm
//!   start".
//!
//! Together with the `kernels` (how to accumulate a batch's contribution),
//! these let the term run a full streaming, warm-started,
//! mixed-precision REML fit over an out-of-core corpus while keeping the
//! determinism and crash-resume guarantees the rest of #973 established.
//!
//! Nothing in this module references `sae_manifold.rs`; the term wires these
//! pieces in on its side of the seam.

pub mod shard_reader;
pub mod warm_state;

// ---------------------------------------------------------------------------
// The driver seam consumed by the SAE term.
// ---------------------------------------------------------------------------

/// Deterministic, restartable source of activation row batches (seam half 1).
pub use shard_reader::{
    CorpusRowSource, DTYPE_F32, HEADER_LEN, MmapShardSource, RowBatch, SHARD_MAGIC, ShardError,
};

/// Per-row inner-solve warm-state cache (seam half 2).
pub use warm_state::{DiskRowWarmCache, RowWarmCache, RowWarmState};

