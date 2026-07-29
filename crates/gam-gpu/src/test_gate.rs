//! One gate for every GPU-conditional test, so an absent device is RECORDED
//! rather than reported as a pass (#2422).
//!
//! 41 tests could report `1 passed` having executed zero assertions, 37 of them
//! because a CPU-only runner takes a bare `return;` before the first assertion.
//! `sphere_gpu_end_to_end_fit_hill_climb_10x_vs_cpu` was green for its entire
//! life and had never appeared in `MASTER_FAILURES`; on a real A10 it misses its
//! `>=10x` gate at **0.51x**. A test that passes having verified nothing is
//! worse than a missing test, because a reader counts it as coverage.
//!
//! Five different skip idioms had grown up across fifteen files, and they
//! disagreed on the two things that matter:
//!
//! * **A device FAULT is not an absent device.** Twenty-one sites wrote
//!   `let Some(rt) = GpuRuntime::resolve(..) else { return; }`, and that `else`
//!   also swallows `Err(..)`. A machine WITH a GPU whose driver is broken
//!   therefore reported the same silent pass as a machine without one. This
//!   gate panics on `Err` unconditionally: a fault is a failure, always.
//! * **A lane that demanded a GPU must not skip.** [`crate::global_policy`]
//!   already carries that demand, so [`GpuPolicy::Required`] turns an absent
//!   device into a panic here. A GPU lane sets the policy once and every gated
//!   test in the process becomes mandatory — no per-test opt-in, and no
//!   environment variable (`env::var` is banned in this tree).
//!
//! What remains — `Auto`/`Off` with no device — is a real skip, and it is
//! counted. [`skipped_for_absent_device`] lets a suite assert how many gated
//! tests declined to run, so "37 passed" can no longer hide "37 verified
//! nothing".
use crate::device_runtime::GpuRuntime;
use crate::{GpuPolicy, global_policy};
use std::sync::atomic::{AtomicU64, Ordering};

static SKIPPED_FOR_ABSENT_DEVICE: AtomicU64 = AtomicU64::new(0);

/// How many gated tests have declined to run in this process for want of a
/// device.
///
/// The point of a counter rather than a log line: a log line is only evidence
/// if somebody reads it, and nothing did for 37 tests. This is a value a test
/// can assert against.
pub fn skipped_for_absent_device() -> u64 {
    SKIPPED_FOR_ABSENT_DEVICE.load(Ordering::Relaxed)
}

/// The outcome of asking for a device in a test.
#[derive(Debug)]
pub enum GpuTestGate {
    /// A runtime resolved; the test body must proceed and assert.
    Ready(&'static GpuRuntime),
    /// No device on this host, under a policy that permits running without
    /// one. Counted by [`skipped_for_absent_device`] and announced on stderr.
    AbsentDevice,
}

impl GpuTestGate {
    /// The runtime, or `None` when the host has no device.
    ///
    /// Deliberately NOT `Option`-shaped at the call site by default: a caller
    /// that writes `let Some(rt) = gate.runtime() else { return }` is back to
    /// the idiom this module exists to remove. Prefer matching on the gate so
    /// the absent arm is written out and visible in review.
    pub fn runtime(&self) -> Option<&'static GpuRuntime> {
        match self {
            Self::Ready(runtime) => Some(runtime),
            Self::AbsentDevice => None,
        }
    }
}

/// Resolve a runtime for a GPU-conditional test under the process policy.
///
/// Panics when the device is faulted (`Err`) or when the policy is
/// [`GpuPolicy::Required`] and no device is present. Returns
/// [`GpuTestGate::AbsentDevice`] only for a genuinely device-free host under a
/// policy that tolerates one.
pub fn gpu_for_test(label: &str) -> GpuTestGate {
    let policy = global_policy();
    match GpuRuntime::resolve(policy) {
        Ok(Some(runtime)) => GpuTestGate::Ready(runtime),
        Ok(None) => {
            if matches!(policy, GpuPolicy::Required) {
                // SAFETY: aborting is the contract. Under `GpuPolicy::Required`
                // the caller has declared that a device MUST be present, so the
                // only alternatives are to abort or to return a gate the test
                // reads as "skip" -- and a skip here prints `ok` for a test that
                // verified nothing, which is the #2422 defect this gate exists
                // to remove. Reachable only from a `#[test]` under an explicit
                // Required policy.
                panic!(
                    "[gpu-test] {label} REQUIRES a device: the process policy is \
                     GpuPolicy::Required and no CUDA runtime resolved. Skipping here \
                     would report a pass for a test that verified nothing (#2422)."
                );
            }
            SKIPPED_FOR_ABSENT_DEVICE.fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "[gpu-test] SKIPPED (no CUDA device, policy={policy:?}): {label} \
                 -- this test asserted NOTHING; libtest will still print `ok`"
            );
            GpuTestGate::AbsentDevice
        }
        // SAFETY: aborting is the contract. A FAULTED device is not an absent
        // one: resolution reached the runtime and it errored, so continuing
        // would run the test against a broken device or silently skip it. The
        // twenty-one `let Some(..) = resolve(..) else { return }` sites this
        // replaced swallowed exactly this case (#2422). Reachable only from a
        // `#[test]`.
        Err(error) => panic!(
            "[gpu-test] {label}: CUDA resolution FAULTED: {error}. A faulted device is \
             not an absent one, and must never be skipped (#2422) -- twenty-one sites \
             used `let Some(..) = resolve(..) else {{ return }}`, whose else-arm \
             swallowed exactly this."
        ),
    }
}

#[cfg(test)]
mod tests_gpu_test_gate_2422 {
    use super::*;

    /// The counter is the whole point: an absent device must leave a trace a
    /// test can assert on, not just a line on stderr that nothing reads.
    ///
    /// This test is itself device-conditional, and it says so honestly: on a
    /// host WITH a device it checks that nothing was counted as skipped, and on
    /// a host without one it checks that the skip was counted. Both arms
    /// assert, which is the property #2422 is about.
    #[test]
    fn an_absent_device_is_counted_not_silent_2422() {
        let before = skipped_for_absent_device();
        match gpu_for_test("gate self-test") {
            GpuTestGate::Ready(runtime) => {
                assert_eq!(
                    skipped_for_absent_device(),
                    before,
                    "a resolved runtime must not count as a skip"
                );
                assert!(
                    !runtime.selected_device().name.is_empty(),
                    "a Ready gate must carry a runtime with a selected device"
                );
            }
            GpuTestGate::AbsentDevice => {
                assert_eq!(
                    skipped_for_absent_device(),
                    before + 1,
                    "an absent device must increment the skip counter exactly once"
                );
            }
        }
    }

    /// `Ready` carries a runtime and `AbsentDevice` does not — the projection
    /// every call site reads.
    #[test]
    fn the_gate_projects_to_an_option_only_at_the_call_site_2422() {
        let gate = gpu_for_test("gate projection self-test");
        assert_eq!(
            gate.runtime().is_some(),
            matches!(gate, GpuTestGate::Ready(_)),
            "runtime() must agree with the variant"
        );
    }
}
