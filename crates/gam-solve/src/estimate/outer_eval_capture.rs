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

use ndarray::{Array1, Array2};
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
/// that seed. Every gradient and scalar-stencil array contains exactly
/// `psi_dim` entries in ψ-local order. Smoothing-parameter ρ coordinates are
/// deliberately excluded: the κ/geometry gates that request this record do not
/// grade them, and each unnecessary finite-difference coordinate costs two
/// complete inner profiles.
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
    pub frozen_logdet_h_psi_gradient: Array1<f64>,
    pub mode_response_logdet_h_psi_gradient: Array1<f64>,
    pub analytic_mode_response_norm: Array1<f64>,
    pub finite_difference_mode_response_norm: Array1<f64>,
    pub mode_response_relative_error: Array1<f64>,
    pub mode_response_max_abs_error: Array1<f64>,
    pub logdet_s_psi_gradient: Array1<f64>,
    pub kkt_psi_gradient: Array1<f64>,
    pub finite_difference_fixed_beta_psi_gradient: Array1<f64>,
    pub finite_difference_logdet_h_psi_gradient: Array1<f64>,
    pub finite_difference_logdet_s_psi_gradient: Array1<f64>,
    pub finite_difference_kkt_psi_gradient: Array1<f64>,
}

/// Maximum evaluations retained per capture window (opening iterates only).
const MAX_CAPTURED: usize = 8;

static ENABLED: AtomicBool = AtomicBool::new(false);

struct OuterGradientFdCapture {
    min_psi_dim: usize,
    record: Option<OuterGradientFdRecord>,
    components: Vec<(f64, f64, f64, f64, f64, f64)>,
    criterion_components: Option<(f64, [f64; 4])>,
    selected_mode: Option<(Array1<f64>, Option<Array2<f64>>)>,
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
            criterion_components: None,
            selected_mode: None,
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

pub(crate) fn outer_gradient_component_capture_enabled() -> bool {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow()
            .as_ref()
            .is_some_and(|state| state.record.is_none())
    })
}

pub(crate) fn record_outer_gradient_component(
    fixed_beta: f64,
    logdet_h: f64,
    frozen_logdet_h: f64,
    mode_response_logdet_h: f64,
    logdet_s: f64,
    kkt: f64,
) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
        {
            state.components.push((
                fixed_beta,
                logdet_h,
                frozen_logdet_h,
                mode_response_logdet_h,
                logdet_s,
                kkt,
            ));
        }
    });
}

pub(crate) fn take_outer_gradient_components() -> Vec<(f64, f64, f64, f64, f64, f64)> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .as_mut()
            .map_or_else(Vec::new, |state| std::mem::take(&mut state.components))
    })
}

pub(crate) fn begin_outer_criterion_component_capture() {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut() {
            state.criterion_components = None;
            state.selected_mode = None;
        }
    });
}

/// Retain the final selected scalar-criterion decomposition for an armed
/// outer-gradient audit.
///
/// This is public only so sibling workspace evaluators can report through the
/// same typed sink after their own nonconvex mode selection. It is a no-op
/// unless [`enable_outer_gradient_fd_capture`] armed the calling thread.
pub fn record_outer_criterion_components(cost: f64, components: [f64; 4]) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
        {
            state.criterion_components = Some((cost, components));
        }
    });
}

pub(crate) fn take_outer_criterion_components() -> Option<(f64, [f64; 4])> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .as_mut()
            .and_then(|state| state.criterion_components.take())
    })
}

/// Retain the selected coefficient mode and its analytic extended-coordinate
/// response columns for an armed finite-difference audit.
///
/// Sibling workspace evaluators call this only after nonconvex candidate
/// selection, beside [`record_outer_criterion_components`]. Value-only
/// evaluations pass no response columns but still retain their selected
/// coefficients for the scalar stencil.
pub fn record_outer_selected_mode(
    beta: Array1<f64>,
    ext_mode_response_cols: Option<Array2<f64>>,
) {
    FD_CAPTURE.with(|capture| {
        if let Some(state) = capture.borrow_mut().as_mut()
            && state.record.is_none()
        {
            state.selected_mode = Some((beta, ext_mode_response_cols));
        }
    });
}

pub(crate) fn take_outer_selected_mode() -> Option<(Array1<f64>, Option<Array2<f64>>)> {
    FD_CAPTURE.with(|capture| {
        capture
            .borrow_mut()
            .as_mut()
            .and_then(|state| state.selected_mode.take())
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

// ═══════════════════════════════════════════════════════════════════════════
//  ρ-block outer audit (#2454)
// ═══════════════════════════════════════════════════════════════════════════
//
// The ψ block has carried a typed analytic-vs-FD record since #2460; the ρ
// block had none, so every large-λ smoothing-gradient investigation had to
// scrape `log::trace!` lines or bolt an environment-gated instrument onto the
// evaluator. This channel closes that asymmetry: it emits, per outer
// evaluation, the SAME four-way additive decomposition the criterion VALUE
// carries (`RemlCriterionComponents`) but for each ρ coordinate's analytic
// gradient — so a caller can finite-difference each criterion component and
// grade the gradient part that owns it, instead of grading only their sum.
//
// Thread-local and disabled by default, matching the ψ channel: a parallel
// integration test can neither consume nor overwrite another test's audit.

/// One ρ coordinate's analytic gradient, split into the additive parts that
/// match the criterion-value components of [`RemlCriterionComponents`].
///
/// `fixed_beta + logdet_h + logdet_s` is the envelope gradient entry as
/// assembled; `total` additionally carries any IFT/KKT correction folded in
/// afterwards, so `total − (fixed_beta + logdet_h + logdet_s)` is the `kkt`
/// part. `lambda` and `block_quadratic` are the two raw inputs the
/// `fixed_beta` part is built from (`½·λ_k·q_k`, scaled by the dispersion
/// channel), retained because a defect that is proportional to `λ_k` is only
/// diagnosable against the `λ_k` it was multiplied by.
#[derive(Clone, Copy, Debug)]
pub struct RhoGradientParts {
    pub index: usize,
    pub lambda: f64,
    pub block_quadratic: f64,
    pub fixed_beta: f64,
    pub logdet_h: f64,
    pub logdet_s: f64,
    pub total: f64,
}

/// One outer evaluation's ρ-block audit: the criterion value decomposition and
/// the per-coordinate analytic gradient decomposition that pairs with it.
#[derive(Clone, Debug, Default)]
pub struct RhoOuterAudit {
    /// `(cost, [fixed_beta, logdet_h, logdet_s, kkt])` for the criterion VALUE.
    pub criterion: Option<(f64, [f64; 4])>,
    /// Per-ρ-coordinate analytic gradient parts, in coordinate order.
    pub parts: Vec<RhoGradientParts>,
}

thread_local! {
    static RHO_AUDIT: RefCell<Option<RhoOuterAudit>> = const { RefCell::new(None) };
}

/// Arm the ρ-block audit on this thread, discarding any previous window.
///
/// Every subsequent outer evaluation on this thread overwrites the window, so
/// the caller reads the audit for the LAST evaluation it triggered — which is
/// the contract a probe wants when it evaluates at one θ at a time.
pub fn enable_rho_outer_audit() {
    RHO_AUDIT.with(|audit| *audit.borrow_mut() = Some(RhoOuterAudit::default()));
}

/// Disarm the ρ-block audit and take the last evaluation's window.
pub fn take_rho_outer_audit() -> Option<RhoOuterAudit> {
    RHO_AUDIT.with(|audit| audit.borrow_mut().take())
}

pub(crate) fn rho_outer_audit_enabled() -> bool {
    RHO_AUDIT.with(|audit| audit.borrow().is_some())
}

/// Start a fresh window for one outer evaluation (no-op when disarmed).
pub(crate) fn begin_rho_outer_audit_eval() {
    RHO_AUDIT.with(|audit| {
        if let Some(state) = audit.borrow_mut().as_mut() {
            *state = RhoOuterAudit::default();
        }
    });
}

pub(crate) fn record_rho_outer_criterion(cost: f64, components: [f64; 4]) {
    RHO_AUDIT.with(|audit| {
        if let Some(state) = audit.borrow_mut().as_mut() {
            state.criterion = Some((cost, components));
        }
    });
}

pub(crate) fn record_rho_gradient_parts(parts: Vec<RhoGradientParts>) {
    RHO_AUDIT.with(|audit| {
        if let Some(state) = audit.borrow_mut().as_mut() {
            state.parts = parts;
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
