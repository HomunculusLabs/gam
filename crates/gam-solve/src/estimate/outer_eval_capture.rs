//! Structured capture of outer-objective evidence for integration tests.
//!
//! The raw-evaluation window serves flexible-link measurements (#1876). The
//! finite-difference record serves end-to-end gradient gates (#2460): when
//! explicitly enabled, the generic outer runner compares the analytic gradient
//! at its first bounded seed with a finite difference of that same objective.
//! Tests consume typed arrays rather than scraping formatted production logs.
//!
//! Both channels are disabled by default and gated by relaxed atomic loads.

use ndarray::Array1;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// One captured outer evaluation: the outer coordinate `theta = (ρ ‖ link)`, the
/// scalar cost, and the analytic outer gradient in the same layout.
#[derive(Clone, Debug)]
pub struct OuterEvalRecord {
    pub theta: Array1<f64>,
    pub cost: f64,
    pub gradient: Array1<f64>,
}

/// Analytic-vs-finite-difference evidence at one real outer seed.
#[derive(Clone, Debug)]
pub struct OuterGradientFdRecord {
    pub theta: Array1<f64>,
    pub cost: f64,
    pub analytic_gradient: Array1<f64>,
    pub finite_difference_gradient: Array1<f64>,
    pub steps: Array1<f64>,
}

/// Maximum evaluations retained per capture window (opening iterates only).
const MAX_CAPTURED: usize = 8;

static ENABLED: AtomicBool = AtomicBool::new(false);
static FD_ENABLED: AtomicBool = AtomicBool::new(false);

fn buffer() -> &'static Mutex<Vec<OuterEvalRecord>> {
    static BUFFER: OnceLock<Mutex<Vec<OuterEvalRecord>>> = OnceLock::new();
    BUFFER.get_or_init(|| Mutex::new(Vec::new()))
}

fn fd_buffer() -> &'static Mutex<Option<OuterGradientFdRecord>> {
    static BUFFER: OnceLock<Mutex<Option<OuterGradientFdRecord>>> = OnceLock::new();
    BUFFER.get_or_init(|| Mutex::new(None))
}

/// Start capturing outer evaluations, clearing any prior window.
pub fn enable_outer_eval_capture() {
    buffer().lock().expect("outer-eval capture buffer").clear();
    ENABLED.store(true, Ordering::Relaxed);
}

/// Stop capturing and drain the recorded opening evaluations (in eval order).
pub fn take_outer_eval_capture() -> Vec<OuterEvalRecord> {
    ENABLED.store(false, Ordering::Relaxed);
    std::mem::take(&mut *buffer().lock().expect("outer-eval capture buffer"))
}

/// Request one structured analytic-vs-FD audit at the next outer seed.
pub fn enable_outer_gradient_fd_capture() {
    *fd_buffer().lock().expect("outer-gradient FD buffer") = None;
    FD_ENABLED.store(true, Ordering::Relaxed);
}

/// Stop the audit window and take its single record.
pub fn take_outer_gradient_fd_capture() -> Option<OuterGradientFdRecord> {
    FD_ENABLED.store(false, Ordering::Relaxed);
    fd_buffer()
        .lock()
        .expect("outer-gradient FD buffer")
        .take()
}

pub(crate) fn outer_gradient_fd_capture_enabled() -> bool {
    FD_ENABLED.load(Ordering::Relaxed)
}

pub(crate) fn record_outer_gradient_fd(record: OuterGradientFdRecord) {
    if !FD_ENABLED.swap(false, Ordering::Relaxed) {
        return;
    }
    *fd_buffer().lock().expect("outer-gradient FD buffer") = Some(record);
}

/// Record one outer evaluation when capture is enabled (no-op otherwise). Only
/// the first [`MAX_CAPTURED`] evaluations of a window are retained.
pub(crate) fn record_outer_eval(theta: &Array1<f64>, cost: f64, gradient: &Array1<f64>) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let mut b = buffer().lock().expect("outer-eval capture buffer");
    if b.len() < MAX_CAPTURED {
        b.push(OuterEvalRecord {
            theta: theta.clone(),
            cost,
            gradient: gradient.clone(),
        });
    }
}
