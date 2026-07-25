//! Structured capture of outer-objective evidence for integration tests.
//!
//! The raw-evaluation window serves flexible-link measurements (#1876). The
//! finite-difference record serves end-to-end gradient gates (#2460): when
//! explicitly enabled, the generic outer runner compares the analytic gradient
//! at its first bounded seed with a finite difference of that same objective.
//! Tests consume typed arrays rather than scraping formatted production logs.
//!
//! Both channels are disabled by default. The raw window is process-global
//! because its flexible-link measurements intentionally span helper calls. The
//! finite-difference request is thread-local: a parallel integration test can
//! neither consume nor overwrite another test's one-shot audit.

use ndarray::Array1;
use std::cell::RefCell;
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

/// Analytic-vs-finite-difference evidence for the ψ block at one real outer
/// seed.
///
/// `theta` retains the complete outer seed and `rho_dim` locates the ψ block in
/// that seed. The three gradient/stencil arrays contain exactly `psi_dim`
/// entries in ψ-local order. Smoothing-parameter ρ coordinates are deliberately
/// excluded: the κ/geometry gates that request this record do not grade them,
/// and each unnecessary finite-difference coordinate costs two complete inner
/// profiles.
#[derive(Clone, Debug)]
pub struct OuterGradientFdRecord {
    pub theta: Array1<f64>,
    pub rho_dim: usize,
    pub psi_dim: usize,
    pub cost: f64,
    pub analytic_psi_gradient: Array1<f64>,
    pub finite_difference_psi_gradient: Array1<f64>,
    pub psi_steps: Array1<f64>,
    pub fixed_beta_psi_gradient: Array1<f64>,
    pub logdet_h_psi_gradient: Array1<f64>,
    pub logdet_s_psi_gradient: Array1<f64>,
}

/// Maximum evaluations retained per capture window (opening iterates only).
const MAX_CAPTURED: usize = 8;

static ENABLED: AtomicBool = AtomicBool::new(false);

struct OuterGradientFdCapture {
    min_psi_dim: usize,
    record: Option<OuterGradientFdRecord>,
    components: Vec<(f64, f64, f64)>,
}

thread_local! {
    static FD_CAPTURE: RefCell<Option<OuterGradientFdCapture>> = const { RefCell::new(None) };
}

fn buffer() -> &'static Mutex<Vec<OuterEvalRecord>> {
    static BUFFER: OnceLock<Mutex<Vec<OuterEvalRecord>>> = OnceLock::new();
    BUFFER.get_or_init(|| Mutex::new(Vec::new()))
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

/// Request one structured audit at the next outer seed with enough ψ axes.
pub fn enable_outer_gradient_fd_capture(min_psi_dim: usize) {
    FD_CAPTURE.with(|capture| {
        *capture.borrow_mut() = Some(OuterGradientFdCapture {
            min_psi_dim,
            record: None,
            components: Vec::new(),
        });
    });
}

pub(crate) fn begin_outer_gradient_component_capture() {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut() {
            state.components.clear();
        }
    });
}

pub(crate) fn record_outer_gradient_component(
    fixed_beta: f64,
    logdet_h: f64,
    logdet_s: f64,
) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
        {
            state.components.push((fixed_beta, logdet_h, logdet_s));
        }
    });
}

pub(crate) fn take_outer_gradient_components() -> Vec<(f64, f64, f64)> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .as_mut()
            .map_or_else(Vec::new, |state| std::mem::take(&mut state.components))
    })
}

/// Stop the audit window and take its single record.
pub fn take_outer_gradient_fd_capture() -> Option<OuterGradientFdRecord> {
    FD_CAPTURE.with(|capture| capture.borrow_mut().take().and_then(|state| state.record))
}

pub(crate) fn outer_gradient_fd_capture_enabled(psi_dim: usize) -> bool {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow()
            .as_ref()
            .is_some_and(|state| state.record.is_none() && psi_dim >= state.min_psi_dim)
    })
}

pub(crate) fn record_outer_gradient_fd(record: OuterGradientFdRecord) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
            && record.psi_dim >= state.min_psi_dim
        {
            state.record = Some(record);
        }
    });
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
