//! Shared helpers for the large-`K` scaling examples (`scale_k`, `tiered_gpu_scale`,
//! `tiered_k2000_measure`). Cargo does not build this directory as an example target
//! because it has no `main.rs`; each example pulls it in with `mod common;`.

/// splitmix64 mixing step — the self-contained deterministic RNG the scaling
/// examples use to synthesise seeds/coordinates without pulling in an RNG crate.
/// This is the same mixing function the library seeds its coordinate-partition
/// dictionary with (`sparse_dict::coordinate_partition_frames`); kept here as one
/// example-side copy instead of a paste in every scaling example.
pub fn splitmix64(x: u64) -> u64 {
    gam_linalg::utils::splitmix64_hash(x)
}
