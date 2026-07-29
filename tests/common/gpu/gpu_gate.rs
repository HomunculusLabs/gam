//! Structured GPU-skip gate for tests that require CUDA.
//!
//! Every test that is meaningful only on a CUDA host calls `gpu_gate` at the
//! top of its body and early-returns when the result is `GpuGate::Skip`.
//! Unlike a bare `if !cuda_selected() { return; }`, the skip is counted and
//! asserted, so the early return is no longer a test that reported `ok` having
//! executed nothing.
//!
//! ## Usage
//!
//! ```rust,ignore
//! mod common;
//! use common::gpu_gate::{GpuGate, gpu_gate};
//!
//! #[test]
//! fn my_gpu_test() {
//!     if let GpuGate::Skip = gpu_gate("my_gpu_test") { return; }
//!     // ... test body ...
//! }
//! ```
//!
//! ## CI enforcement
//!
//! A skip is counted, not just printed: the gate delegates to
//! `gam::gpu::test_gate`, which increments a process counter and emits the one
//! shared `SKIPPED(no-cuda):` marker, and this gate then asserts the count
//! moved. So a gated test that declines still executes an assertion, and a
//! green run can be scraped for an inventory of what did not run instead of
//! implying that every `ok` verified something (#2422).
//!
//! On a GPU runner nothing here skips, and a device that is present but not
//! *selected* is a hard failure rather than a skip — see `gpu_gate`. The
//! companion `gpu_required_tests_did_not_skip` asserts the same implication at
//! suite level, against the counter rather than against `cuda_selected()`.

use gam::gpu::cuda_selected;
use gam::gpu::test_gate::{GpuTestGate, assert_absent_device_was_counted, gpu_for_test};

/// Result of the GPU gate check.
pub enum GpuGate {
    /// CUDA is selected — the test body should execute.
    Run,
    /// No CUDA device on this host. The skip has been COUNTED by
    /// `gam::gpu::test_gate` and announced with the shared marker, and the
    /// count was asserted, so the caller's `return` no longer leaves a test
    /// that executed zero assertions.
    Skip,
}

/// Check whether the test should run on this host.
///
/// Three outcomes, because the callers of this gate need two different things
/// to be true and conflating them produced a pass that verified nothing:
///
/// 1. **No device** → counted skip. [`gpu_for_test`] records it against the
///    shared counter and prints the one greppable marker, and this asserts the
///    count moved, so the skip path executes a real assertion (#2422).
/// 2. **Device present and CUDA selected** → `Run`.
/// 3. **Device present but CUDA *not* selected** → panic.
///
/// Case 3 is the one that was silently wrong. Every caller of this gate is a
/// GPU-vs-CPU parity or routing test, so it needs the production dispatch to
/// actually reach the device — `cuda_selected()`, not merely a device being
/// installed. But `cuda_selected()` consults the process-wide policy, which
/// [`gam::gpu::configure_global_policy`] fixes first-writer-wins, and
/// `backend_status_and_policy_dispatch_are_consistent` sets it to
/// `GpuPolicy::Off` **in this same test binary**. Whenever that sibling won the
/// race on a GPU host, this gate reported "cuda not selected" and every parity
/// test skipped — or worse, had they not skipped, compared the CPU path against
/// itself and passed while measuring nothing.
///
/// So a device that is present but not selected is a defect in this suite's own
/// setup, not a reason to skip, and it fails here naming the policy that caused
/// it rather than disappearing into a `SKIP` line.
pub fn gpu_gate(test_name: &str) -> GpuGate {
    let floor = gam::gpu::test_gate::skipped_for_absent_device();
    match gpu_for_test(test_name) {
        GpuTestGate::AbsentDevice => {
            assert_absent_device_was_counted(floor);
            GpuGate::Skip
        }
        GpuTestGate::Ready(runtime) => {
            let selected = cuda_selected().unwrap_or_else(|error| {
                panic!("GPU probe fault while gating {test_name}: {error}")
            });
            assert!(
                selected,
                "{test_name}: this host HAS a CUDA device ({}), yet the unified GPU policy \
                 did not select CUDA (process policy = {:?}). Skipping here would report a \
                 pass for a parity test that never reached the device, and running would \
                 compare the CPU path against itself. The process policy is a \
                 first-writer-wins OnceLock, so the usual cause is a sibling test in this \
                 binary calling configure_global_policy(Off) before this one ran (#2422).",
                runtime.selected_device().name,
                gam::gpu::global_policy(),
            );
            GpuGate::Run
        }
    }
}
