//! Solver GPU kernels and dispatch wiring.
//!
//! Process-wide GPU policy and direct device selection live in [`crate::gpu`];
//! the domain compute kernels live in [`crate::gpu_kernels`]. This module owns
//! the solver-side device dispatch glue: the PIRLS CUDA kernel bodies
//! ([`pirls_gpu`], `pirls_dispatch_wire`), their PIRLS-side host admission
//! shim (`pirls_host_dispatch`, folded here from `solver/pirls/gpu_dispatch.rs`),
//! and the REML/arrow-Schur device entry points.

pub(crate) mod pirls_dispatch_wire;
pub mod pirls_gpu;
pub(crate) mod pirls_host_dispatch;
pub(crate) mod reml_outer;
