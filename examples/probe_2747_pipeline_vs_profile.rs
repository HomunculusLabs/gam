//! gam#2747: is the range optimized against the model that gets FITTED?
//!
//! Restoring the pinned-κ/free-range enrollment moves
//! `kappa_one_fit_recovers_planted_spherical_signal` from `ℓ̂ = 1.73` (the auto
//! heuristic) to `ℓ̂ = 1.2e8` (the geodesic-distance face), and its R² against
//! the planted truth from **0.8910 to 0.8686**. The criterion is now honest
//! everywhere in the chart — that is measured, `probe_2747_range_normalization`
//! — so a range that MINIMIZES the evidence and LOSES accuracy is either
//!
//!   (a) an ordinary REML-vs-R² disagreement on a basis this fixture is
//!       capacity-starved for (`measure_2687_...` records `matern` at 885× less
//!       residual on the same data at the same centers), or
//!   (b) the profile scoring a different model than the fit realizes.
//!
//! (b) is the one worth killing first, because it would make every range
//! estimate wrong rather than this one unlucky. The profile assembles
//! `[1 | K z]` with a block penalty and calls `gaussian_reml_multi_closed_form`
//! directly; the fit goes through the whole term-collection pipeline. If those
//! two disagree about which ℓ is best, the profile is optimizing a surrogate.
//!
//! So: the EXACT fixture data (same `StdRng`, same draw order), swept over a
//! pinned `length_scale`, scored both ways at each point.
//!
//! Run: `cargo run --release --example probe_2747_pipeline_vs_profile`

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam::matrix::LinearOperator;
use gam_terms::analytic_penalties::PenaltyOp;
use gam_terms::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    build_constant_curvature_basis,
};
use ndarray::{Array1, Array2, s};

const KAPPA: f64 = 1.0;
const CENTERS: usize = 30;

/// `kappa_one_fit_recovers_planted_spherical_signal`, reproduced draw for draw.
fn fixture() -> Vec<(f64, f64, f64, f64)> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(945);
    (0..400)
        .map(|_| {
            let r = 0.9 * rng.random::<f64>().sqrt();
            let th = std::f64::consts::TAU * rng.random::<f64>();
            let x1 = r * th.cos();
            let x2 = r * th.sin();
            let r2 = x1 * x1 + x2 * x2;
            let pz = (1.0 - r2) / (1.0 + r2);
            let px = 2.0 * x1 / (1.0 + r2);
            let truth = pz + 0.5 * px;
            let y = truth + 0.05 * (rng.random::<f64>() - 0.5);
            (x1, x2, truth, y)
        })
        .collect()
}

fn r2_against(truth: &[f64], pred: &Array1<f64>) -> f64 {
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    let ss_res: f64 = truth
        .iter()
        .zip(pred.iter())
        .map(|(t, p)| (t - p) * (t - p))
        .sum();
    let ss_tot: f64 = truth.iter().map(|t| (t - mean) * (t - mean)).sum();
    1.0 - ss_res / ss_tot
}

/// The PIPELINE arm: the whole `fit_from_formula` route, scored exactly as the
/// fixture scores it.
fn pipeline_arm(rows: &[(f64, f64, f64, f64)], formula: &str) -> Option<(f64, f64, f64, f64, f64)> {
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let records: Vec<StringRecord> = rows
        .iter()
        .map(|(x1, x2, _, y)| StringRecord::from(vec![x1.to_string(), x2.to_string(), y.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, records).ok()?;
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) = fit_from_formula(formula, &data, &cfg).ok()? else {
        return None;
    };
    let mut m = Array2::<f64>::zeros((rows.len(), 3));
    for (i, (x1, x2, _, _)) in rows.iter().enumerate() {
        m[(i, 0)] = *x1;
        m[(i, 1)] = *x2;
    }
    let design = gam::smooth::build_term_collection_design(m.view(), &fit.resolvedspec).ok()?;
    let pred = design.design.apply(&fit.fit.beta);
    let truth: Vec<f64> = rows.iter().map(|(_, _, t, _)| *t).collect();
    let realized = fit.resolvedspec.smooth_terms.iter().find_map(|term| {
        if let gam::smooth::SmoothBasisSpec::ConstantCurvature { spec, .. } = &term.basis {
            Some(spec.length_scale)
        } else {
            None
        }
    })?;
    let truth_for_span: Vec<f64> = rows.iter().map(|(_, _, t, _)| *t).collect();
    let dense = design.design.to_dense();
    eprintln!(
        "    [pipeline] design {}x{}  span_ceiling(truth) = {:.6}  penalties = {}",
        dense.nrows(),
        dense.ncols(),
        span_ceiling(&dense, &truth_for_span),
        fit.fit.lambdas.len(),
    );
    let lambda = fit.fit.lambdas.iter().cloned().next().unwrap_or(f64::NAN);
    let reml = fit.fit.reml_score().unwrap_or(f64::NAN);
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    Some((r2_against(&truth, &pred), realized, lambda.ln(), edf, reml))
}

/// The PROFILE arm: exactly what `constant_curvature_psi_profile_value`
/// assembles — `[1 | K z]`, a block-diagonal penalty, one closed-form Gaussian
/// REML — plus the R² of the fit it lands on.
fn profile_arm(rows: &[(f64, f64, f64, f64)], ell: f64) -> Option<(f64, f64, f64, f64)> {
    let n = rows.len();
    let mut feats = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for (i, (x1, x2, _, yy)) in rows.iter().enumerate() {
        feats[(i, 0)] = *x1;
        feats[(i, 1)] = *x2;
        y[i] = *yy;
    }
    let spec = ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: CENTERS,
        },
        kappa: KAPPA,
        kappa_fixed: true,
        length_scale: ell,
        length_scale_fixed: false,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    };
    let basis = build_constant_curvature_basis(feats.view(), &spec).ok()?;
    let smooth = basis.design.to_dense();
    let (_, p) = smooth.dim();
    let mut design = Array2::<f64>::ones((n, p + 1));
    design.slice_mut(s![.., 1..]).assign(&smooth);
    let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
    penalty
        .slice_mut(s![1.., 1..])
        .assign(&basis.active_penalties[0].matrix.as_dense());
    let truth_for_span: Vec<f64> = rows.iter().map(|(_, _, t, _)| *t).collect();
    eprintln!(
        "    [profile ] design {}x{}  span_ceiling(truth) = {:.6}",
        design.nrows(),
        design.ncols(),
        span_ceiling(&design, &truth_for_span),
    );
    let response = y.clone().insert_axis(ndarray::Axis(1));
    let fit = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
        design.view(),
        response.view(),
        penalty.view(),
        None,
        None,
    )
    .ok()?;
    let pred = fit.fitted.column(0).to_owned();
    let truth: Vec<f64> = rows.iter().map(|(_, _, t, _)| *t).collect();
    Some((
        fit.reml_score,
        r2_against(&truth, &pred),
        fit.rho,
        fit.edf,
    ))
}

/// The best any fit could do on a given design: the orthogonal projection of the
/// PLANTED TRUTH onto its column span, scored as R². If this is already low the
/// span is deficient and no smoothing parameter can rescue it.
fn span_ceiling(design: &Array2<f64>, truth: &[f64]) -> f64 {
    use gam_linalg::faer_ndarray::FaerEigh;
    let t = Array1::from_vec(truth.to_vec());
    let gram = design.t().dot(design);
    let rhs = design.t().dot(&t);
    let (evals, evecs) = FaerEigh::eigh(&gram, faer::Side::Lower).expect("gram spectrum");
    let largest = evals.iter().cloned().fold(0.0_f64, f64::max);
    let projected = evecs.t().dot(&rhs);
    let mut solved = Array1::<f64>::zeros(projected.len());
    for i in 0..projected.len() {
        if evals[i] > 1.0e-10 * largest {
            solved[i] = projected[i] / evals[i];
        }
    }
    let pred = design.dot(&evecs.dot(&solved));
    r2_against(truth, &pred)
}

/// The profile's own assembly at a FORCED `ρ`, so the smoothing parameter can be
/// held equal to the pipeline's while everything else stays the profile's.
/// `β̂ = (XᵀX + λS)⁻¹Xᵀy` through the symmetric eigenbasis, which needs no
/// factorization to succeed on a nearly-singular Gram.
fn profile_at_fixed_rho(rows: &[(f64, f64, f64, f64)], ell: f64, rho: f64) -> Option<f64> {
    use gam_linalg::faer_ndarray::FaerEigh;
    let n = rows.len();
    let mut feats = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for (i, (x1, x2, _, yy)) in rows.iter().enumerate() {
        feats[(i, 0)] = *x1;
        feats[(i, 1)] = *x2;
        y[i] = *yy;
    }
    let spec = ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: CENTERS,
        },
        kappa: KAPPA,
        kappa_fixed: true,
        length_scale: ell,
        length_scale_fixed: false,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    };
    let basis = build_constant_curvature_basis(feats.view(), &spec).ok()?;
    let smooth = basis.design.to_dense();
    let (_, p) = smooth.dim();
    let mut design = Array2::<f64>::ones((n, p + 1));
    design.slice_mut(s![.., 1..]).assign(&smooth);
    let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
    penalty
        .slice_mut(s![1.., 1..])
        .assign(&basis.active_penalties[0].matrix.as_dense());
    let gram = design.t().dot(&design) + &penalty.mapv(|v| v * rho.exp());
    let rhs = design.t().dot(&y);
    let (evals, evecs) = FaerEigh::eigh(&gram, faer::Side::Lower).ok()?;
    let projected = evecs.t().dot(&rhs);
    let mut solved = Array1::<f64>::zeros(projected.len());
    for i in 0..projected.len() {
        if evals[i] > 1.0e-12 * evals.iter().cloned().fold(0.0_f64, f64::max) {
            solved[i] = projected[i] / evals[i];
        }
    }
    let beta = evecs.dot(&solved);
    let pred = design.dot(&beta);
    let truth: Vec<f64> = rows.iter().map(|(_, _, t, _)| *t).collect();
    Some(r2_against(&truth, &pred))
}

fn main() {
    let rows = fixture();
    println!(
        "{:>10}  {:>12} {:>8} {:>8} {:>7} | {:>12} {:>8} {:>8} {:>7}",
        "ell", "V_profile", "rho_p", "R2_prof", "edf_p", "V_pipe", "rho_pipe", "R2_pipe", "edf_pipe"
    );
    for ell in [
        0.1_f64, 0.2, 0.3, 0.5, 0.8, 1.2, 1.7345, 3.0, 10.0, 100.0, 1.0e4, 1.0e6, 1.2e8,
    ] {
        let prof = profile_arm(&rows, ell);
        let formula = format!("y ~ curv(x1, x2, kappa=1, centers={CENTERS}, length_scale={ell})");
        let pipe = pipeline_arm(&rows, &formula);
        match (prof, pipe) {
            (Some((v, r2p, rho, edf)), Some((r2, _realized, rho_pipe, edf_pipe, reml_pipe))) => {
                println!(
                    "{ell:>10.3e}  {v:>12.4} {rho:>8.4} {r2p:>8.4} {edf:>7.3} |                      {reml_pipe:>12.4} {rho_pipe:>8.4} {r2:>8.4} {edf_pipe:>7.3}"
                )
            }
            (prof, pipe) => println!("{ell:>10.3e}  profile={:?} pipeline={:?}", prof.is_some(), pipe.is_some()),
        }
    }
    println!(
        "\nprofile assembly at a FORCED rho, ell = 1.7345 (the pipeline lands at rho = -0.7352):"
    );
    println!("{:>8}  {:>8}", "rho", "R2");
    for rho in [-6.0_f64, -5.0339, -4.0, -3.0, -2.0, -1.0, -0.7352, 0.0, 1.0] {
        match profile_at_fixed_rho(&rows, 1.7345, rho) {
            Some(r2) => println!("{rho:>8.4}  {r2:>8.4}"),
            None => println!("{rho:>8.4}  refused"),
        }
    }

    println!("\nfree-range (the shipped fixture, range estimated):");
    let formula = format!("y ~ curv(x1, x2, kappa=1, centers={CENTERS})");
    match pipeline_arm(&rows, &formula) {
        Some((r2, realized, rho, edf, reml)) => println!(
            "  R2 = {r2:.6}   realized ell = {realized:.6e}   rho = {rho:.4}  edf = {edf:.3}  V = {reml:.4}"
        ),
        None => println!("  refused"),
    }
}
