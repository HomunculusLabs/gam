//! Regression for #989 from a *different angle than the sibling repro*: the
//! binding-coefficient-constraint KKT gate must be SCALE-INVARIANT, not merely
//! within an order of magnitude of a fixed absolute tolerance.
//!
//! The original repro (`bug_hunt_nonnegative_constraint_kkt_abort_with_free_term`)
//! trips the gate by ~one order of magnitude (`stat≈5.78e-5` vs the `5e-6` gate)
//! on an `n=400`, O(1)-response dataset. A patch that merely *loosened* the
//! absolute gate (say to `1e-4`) would make that case pass while leaving the
//! true bug — a relative-vs-absolute mismatch between the inner active-set
//! solver and the outer validation gate — intact: the least-squares /
//! profiled-REML gradient is O(n) in magnitude even at a genuine stationary
//! point (issue #879), so the absolute stationarity residual scales with the
//! data and overruns *any* fixed absolute floor for large enough data.
//!
//! This test pins the scale-invariance directly. Same low-discrepancy surface
//! as the sibling, but with `n = 1500` rows and the response multiplied by
//! `C = 1000`:
//!
//! ```text
//!   y = 1000 · ( sin(2π·x) + 0.5·x2 ).
//! ```
//!
//! The OLS slope of `sin(2π·x)` against `x` on `[0,1]` is negative, so
//! `β_x ≥ 0` still BINDS at a MAP of `β_x = 0`. But the inner solve now drives
//! the gradient from `‖g‖₀ ≈ 8.6e5` down to a stationarity residual
//! `‖g−Aᵀλ‖∞ ≈ 0.22` — over
//! four orders of magnitude ABOVE the `5e-6` absolute gate, while the relative
//! ratio `0.22 / 8.6e5 ≈ 2.6e-7` is the same as on the unscaled data. Under a
//! bare absolute gate (or one merely loosened to `1e-4`) this fit aborts hard;
//! under the scale-invariant gate it succeeds and persists that boundary MAP.
//! The separately reported/default-prediction coefficient is the strictly
//! interior truncated posterior mean required by SPEC. That five-orders-of-
//! magnitude headroom is what makes
//! this a genuine scale-invariance test rather than a tolerance-tweak test.

use gam::inference::model::FittedModel;
use std::path::{Path, PathBuf};
use std::process::Command;

const N: usize = 1500;
/// Response magnitude multiplier. Large enough that the absolute stationarity
/// residual at the optimum (≈0.22) clears the `5e-6` gate by >4 orders of
/// magnitude, so only a relative (scale-invariant) criterion can admit the fit.
const SCALE: f64 = 1000.0;

/// Deterministic low-discrepancy training data, response scaled by `SCALE`.
fn write_training_csv(path: &Path) {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let sqrt2 = 2.0_f64.sqrt();
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "x2", "y"]).expect("write header");
    for i in 0..N {
        let k = (i + 1) as f64;
        let x = (k * phi).fract();
        let x2 = (k * sqrt2).fract();
        let y = SCALE * ((2.0 * std::f64::consts::PI * x).sin() + 0.5 * x2);
        writer
            .write_record([format!("{x:.12}"), format!("{x2:.12}"), format!("{y:.12}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// Four probe rows: a pair isolating the `x` slope (x2 held at 0.4) and a pair
/// isolating the `x2` slope (x held at 0.5).
fn write_probe_csv(path: &Path) {
    let mut writer = csv::Writer::from_path(path).expect("create probe csv");
    writer.write_record(["x", "x2", "y"]).expect("write header");
    for &(x, x2) in &[(0.0_f64, 0.4_f64), (0.8, 0.4), (0.5, 0.0), (0.5, 1.0)] {
        writer
            .write_record([format!("{x:.12}"), format!("{x2:.12}"), "0.0".to_string()])
            .expect("write probe row");
    }
    writer.flush().expect("flush probe csv");
}

fn read_means(path: &Path) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open predictions csv");
    let headers = reader.headers().expect("predict csv headers").clone();
    let mean_idx = headers
        .iter()
        .position(|h| h == "posterior_mean")
        .expect("standard predict csv has a posterior_mean column");
    reader
        .records()
        .map(|rec| {
            rec.expect("predict csv row")[mean_idx]
                .parse::<f64>()
                .expect("numeric prediction")
        })
        .collect()
}

fn constrained_x_estimands(model_path: &Path) -> (f64, f64) {
    let model = FittedModel::load_from_path(model_path).expect("load constrained saved model");
    let fit = model.unified().expect("saved model carries unified fit");
    let posterior = fit
        .geometry
        .as_ref()
        .and_then(|geometry| geometry.constrained_posterior.as_ref())
        .expect("nonnegative coefficient persists its constrained posterior");
    (posterior.mode[1], fit.beta[1])
}

fn try_fit(dir: &Path, label: &str, formula: &str) -> (bool, PathBuf, String) {
    let train = dir.join("train.csv");
    let model = dir.join(format!("model_{label}.json"));
    write_training_csv(&train);
    let out = Command::new(gam::gam_binary!())
        .arg("fit")
        .arg(&train)
        .arg(formula)
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(&model)
        .output()
        .expect("spawn gam fit");
    let ok = out.status.success() && model.is_file();
    (ok, model, String::from_utf8_lossy(&out.stderr).into_owned())
}

fn probe_slopes(dir: &Path, label: &str, model: &Path) -> (f64, f64) {
    let probe = dir.join("probe.csv");
    let out_csv = dir.join(format!("pred_{label}.csv"));
    write_probe_csv(&probe);
    let status = Command::new(gam::gam_binary!())
        .arg("predict")
        .arg(model)
        .arg(&probe)
        .arg("--out")
        .arg(&out_csv)
        .status()
        .expect("spawn gam predict");
    assert!(status.success(), "gam predict `{label}` failed");
    let m = read_means(&out_csv);
    assert_eq!(m.len(), 4, "expected four probe predictions for `{label}`");
    let x_slope = (m[1] - m[0]) / 0.8; // Δx = 0.8 at x2 = 0.4
    let x2_slope = m[3] - m[2]; // Δx2 = 1.0 at x = 0.5
    (x_slope, x2_slope)
}

#[test]
fn nonnegative_constraint_kkt_gate_is_scale_invariant() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let dir = tmp.path();

    // THE BUG, at extreme scale: the absolute stationarity residual at this
    // optimum is ≈0.22 — >4 orders of magnitude above the 5e-6 gate — so a bare
    // absolute gate (or one merely loosened to 1e-4) aborts. The scale-invariant
    // gate admits it, because the *relative* residual is ≈2.6e-7.
    let (nn_ok, nnmodel, nn_err) = try_fit(dir, "nonneg", "y ~ nonnegative(x) + x2");
    assert!(
        nn_ok,
        "BUG (#989 scale-invariance): `y ~ nonnegative(x) + x2` aborted on scaled data, where \
         the absolute KKT stationarity residual (~0.22) overruns the 5e-6 gate by >4 orders of \
         magnitude even though the relative residual (~2.6e-7) is tiny. The gate must certify stationarity relative to the \
         O(n) gradient scale, not against a fixed absolute floor. gam fit stderr:\n{nn_err}"
    );

    let (mode_xslope, reported_xslope) = constrained_x_estimands(&nnmodel);
    assert!(
        mode_xslope.abs() < 1e-5,
        "scaled nonnegative coefficient MAP must bind at 0, got {mode_xslope:.12}"
    );
    assert!(
        reported_xslope >= 0.0 && reported_xslope > mode_xslope,
        "reported coefficient {reported_xslope:.12} must be the strictly-interior \
         truncated posterior mean in [0, ∞), above its boundary MAP {mode_xslope:.12}"
    );

    let (nn_xslope, nn_x2slope) = probe_slopes(dir, "nonneg", &nnmodel);
    assert!(
        (nn_xslope - reported_xslope).abs() <= 1e-9 * reported_xslope.abs().max(1.0),
        "default prediction slope {nn_xslope:.12} must equal the persisted posterior \
         mean {reported_xslope:.12}"
    );
    assert!(
        nn_x2slope > 0.1 * SCALE,
        "the free x2 effect should remain a large positive scaled effect, got {nn_x2slope:.6}"
    );
}
