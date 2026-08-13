//! gam#2747: did `76a520c45` leave the kernel class ORTHOGONAL to its parametric
//! block, or only un-deleted?
//!
//! `apply_global_smooth_identifiability` exists to enforce one invariant on the
//! kernel/radial class — the realized smooth block is orthogonal to
//! `[1 | overlapping linear columns]` — and it enforced it by DELETING one
//! coefficient direction per parametric direction. `76a520c45` established that
//! the deletion is licensed only under containment, and made the gate a
//! containment test; the branch it left for a non-contained direction is to do
//! nothing at all.
//!
//! `probe_2747_containment_registry` measures that the whole Matérn class is
//! non-contained, so the whole Matérn class now takes that branch. This probe
//! asks the consequence: with the deletion correctly withheld, is the invariant
//! still there, and does anything downstream notice that it is not?
//!
//! Measured through the REAL pipeline (`build_term_collection_design`), on the
//! designs a fit actually consumes:
//!
//!   * `||X_smooth^T C|| / (||X_smooth|| * ||C||)` — the identical statistic
//!     `orthogonality_relative_residual_for_design` gates at `1e-8` when a
//!     transform IS applied, and which nothing evaluates when it is not;
//!   * the model's own span dimension, so an orthogonality gain that costs a
//!     dimension is visible as such;
//!   * for a formula with an OVERLAPPING linear term, the same two numbers
//!     against `[1 | x1]`, where the estimand split between the linear term and
//!     the smooth is the thing at stake.
//!
//! Run: `cargo run --release --example probe_2747_parametric_orthogonality`

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use ndarray::{Array1, Array2};

const N: usize = 400;

/// A deterministic 2-D cloud in the unit disk plus a response with both a linear
/// and a curved component, so a formula that names `x1` explicitly has something
/// to identify.
fn rows() -> Vec<(f64, f64, f64)> {
    let mut state = 0x2747_0000_0000_0002_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(N);
    while out.len() < N {
        let a = 2.0 * next() - 1.0;
        let b = 2.0 * next() - 1.0;
        if a * a + b * b <= 1.0 {
            let signal = 1.3 * a + (3.0 * (a * a + b * b)).sin();
            let noise = 0.1 * (2.0 * next() - 1.0);
            out.push((a, b, signal + noise));
        }
    }
    out
}

/// `||A^T B|| / (||A|| ||B||)` — the shipped orthogonality statistic.
fn relative_cross(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let cross = a.t().dot(b);
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let a_norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    num / (a_norm * b_norm).max(1e-300)
}

/// `X − C(CᵀC)⁻¹CᵀX`, through `C`'s own truncated spectrum so a rank-deficient
/// constraint block is handled rather than factorized into a failure.
fn residualize(design: &Array2<f64>, constraint: &Array2<f64>) -> Array2<f64> {
    use gam_linalg::faer_ndarray::FaerEigh;
    let gram = constraint.t().dot(constraint);
    let cross = constraint.t().dot(design);
    let (evals, evecs) = FaerEigh::eigh(&gram, faer::Side::Lower).expect("constraint gram");
    let top = evals.iter().cloned().fold(0.0_f64, f64::max);
    let mut solved = evecs.t().dot(&cross);
    for i in 0..evals.len() {
        let scale = if evals[i] > top * (evals.len() as f64) * f64::EPSILON {
            1.0 / evals[i]
        } else {
            0.0
        };
        for j in 0..solved.ncols() {
            solved[[i, j]] *= scale;
        }
    }
    design - &constraint.dot(&evecs.dot(&solved))
}

/// The OWNER-block arm: the constraint block is another smooth's REALIZED
/// columns, not the intercept and not a raw data column.
///
/// `analyze_smooth_ownership` makes a narrower smooth own its subspace and
/// residualizes the broader one against it, which is what stops the two fitting
/// the same structure twice. That block is not contained in any kernel span
/// either, so the containment gate withholds it wholesale rather than for one
/// direction — the whole hierarchy, not a centering convention.
fn measure_owner(formula: &str, dependent_contains: &str) {
    let data = rows();
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let records: Vec<StringRecord> = data
        .iter()
        .map(|(x1, x2, y)| StringRecord::from(vec![x1.to_string(), x2.to_string(), y.to_string()]))
        .collect();
    let encoded = match encode_recordswith_inferred_schema(headers, records) {
        Ok(encoded) => encoded,
        Err(err) => {
            println!("{formula}: encode refused: {err}");
            return;
        }
    };
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let fit = match fit_from_formula(formula, &encoded, &cfg) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => {
            println!("{formula}: not a standard fit");
            return;
        }
        Err(err) => {
            println!("{formula}: refused: {err}");
            return;
        }
    };
    let mut frame = Array2::<f64>::zeros((data.len(), 3));
    for (i, (x1, x2, y)) in data.iter().enumerate() {
        frame[(i, 0)] = *x1;
        frame[(i, 1)] = *x2;
        frame[(i, 2)] = *y;
    }
    let design = match gam::smooth::build_term_collection_design(frame.view(), &fit.resolvedspec) {
        Ok(design) => design,
        Err(err) => {
            println!("{formula}: design rebuild refused: {err}");
            return;
        }
    };
    let dense = design.design.to_dense();
    let offset = smooth_block_offset(&design);
    let mut owner = None;
    let mut dependent = None;
    for built in &design.smooth.terms {
        let block = dense
            .slice(ndarray::s![
                ..,
                offset + built.coeff_range.start..offset + built.coeff_range.end
            ])
            .to_owned();
        if built.name.contains(dependent_contains) {
            dependent = Some((built.name.clone(), block));
        } else {
            owner = Some((built.name.clone(), block));
        }
    }
    match (owner, dependent) {
        (Some((owner_name, owner_block)), Some((dependent_name, dependent_block))) => {
            let residualized = residualize(&dependent_block, &owner_block);
            println!(
                "{formula}\n    owner '{owner_name}' {}x{}   dependent '{dependent_name}' {}x{}\n\
                 \x20       shipped        ||X_dep' X_own||/(norms) = {:.6e}\n\
                 \x20       residualized   ||X_dep' X_own||/(norms) = {:.6e}   cols = {}",
                owner_block.nrows(),
                owner_block.ncols(),
                dependent_block.nrows(),
                dependent_block.ncols(),
                relative_cross(&dependent_block, &owner_block),
                relative_cross(&residualized, &owner_block),
                residualized.ncols(),
            );
        }
        _ => println!("{formula}: expected exactly one owner and one dependent smooth"),
    }
}

/// Where the smooth block starts in the FULL design: `SmoothTerm::coeff_range`
/// is smooth-block-local, and the design carries the intercept, linear and
/// random-effect blocks ahead of it.
fn smooth_block_offset(design: &gam::smooth::TermCollectionDesign) -> usize {
    design
        .intercept_range
        .end
        .max(
            design
                .linear_ranges
                .iter()
                .map(|(_, range)| range.end)
                .max()
                .unwrap_or(0),
        )
        .max(
            design
                .random_effect_ranges
                .iter()
                .map(|(_, range)| range.end)
                .max()
                .unwrap_or(0),
        )
}

fn measure(formula: &str, parametric_cols: &[usize]) {
    let data = rows();
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let records: Vec<StringRecord> = data
        .iter()
        .map(|(x1, x2, y)| {
            StringRecord::from(vec![x1.to_string(), x2.to_string(), y.to_string()])
        })
        .collect();
    let encoded = match encode_recordswith_inferred_schema(headers, records) {
        Ok(encoded) => encoded,
        Err(err) => {
            println!("{formula}: encode refused: {err}");
            return;
        }
    };
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let fit = match fit_from_formula(formula, &encoded, &cfg) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => {
            println!("{formula}: not a standard fit");
            return;
        }
        Err(err) => {
            println!("{formula}: refused: {err}");
            return;
        }
    };

    let mut frame = Array2::<f64>::zeros((data.len(), 3));
    for (i, (x1, x2, y)) in data.iter().enumerate() {
        frame[(i, 0)] = *x1;
        frame[(i, 1)] = *x2;
        frame[(i, 2)] = *y;
    }
    let design = match gam::smooth::build_term_collection_design(frame.view(), &fit.resolvedspec) {
        Ok(design) => design,
        Err(err) => {
            println!("{formula}: design rebuild refused: {err}");
            return;
        }
    };
    let dense = design.design.to_dense();

    // The parametric block this smooth is supposed to be orthogonal to:
    // the intercept, plus whichever raw feature columns the caller names.
    let mut constraint = Array2::<f64>::ones((data.len(), 1 + parametric_cols.len()));
    for (slot, &col) in parametric_cols.iter().enumerate() {
        for i in 0..data.len() {
            constraint[(i, 1 + slot)] = frame[(i, col)];
        }
    }

    // `SmoothTerm::coeff_range` is SMOOTH-BLOCK-LOCAL; the full design carries
    // the intercept, the linear terms and the random-effect blocks ahead of it.
    // Slicing the full design with the local range silently puts the intercept
    // column inside the "smooth block", which reads as a catastrophic
    // orthogonality failure on every basis including the contained ones.
    let smooth_offset = smooth_block_offset(&design);
    for term in &fit.resolvedspec.smooth_terms {
        let Some(realized) = design
            .smooth
            .terms
            .iter()
            .find(|built| built.name == term.name)
        else {
            continue;
        };
        let block = dense
            .slice(ndarray::s![
                ..,
                smooth_offset + realized.coeff_range.start
                    ..smooth_offset + realized.coeff_range.end
            ])
            .to_owned();
        // The two constructions this issue is choosing between, on the SAME
        // realized block, so the comparison is controlled rather than a claim
        // about a previous commit:
        //
        //   deleted      = X·Z, Z spanning null((XᵀC)ᵀ) — what the pipeline
        //                  applied unconditionally before `76a520c45`, and still
        //                  applies to a contained direction;
        //   residualized = X − C(CᵀC)⁻¹CᵀX — span-preserving, always available.
        let operator = gam_linalg::matrix::DesignMatrix::from(block.clone());
        let (deleted_cols, deleted_cross) =
            match gam_terms::basis::orthogonality_transform_for_design(
                &operator,
                constraint.view(),
                None,
            ) {
                Ok(z) => {
                    let constrained = block.dot(&z);
                    (
                        constrained.ncols() as i64,
                        relative_cross(&constrained, &constraint),
                    )
                }
                Err(_) => (-1, f64::NAN),
            };
        let residualized = residualize(&block, &constraint);
        println!(
            "{formula}\n    term '{}'  block {}x{}   full design {}x{}\n\
             \x20       shipped        ||X'C||/(||X||||C||) = {:.6e}   cols = {}\n\
             \x20       deleted   (Z)  ||X'C||/(||X||||C||) = {:.6e}   cols = {}\n\
             \x20       residualized   ||X'C||/(||X||||C||) = {:.6e}   cols = {}",
            term.name,
            block.nrows(),
            block.ncols(),
            dense.nrows(),
            dense.ncols(),
            relative_cross(&block, &constraint),
            block.ncols(),
            deleted_cross,
            deleted_cols,
            relative_cross(&residualized, &constraint),
            residualized.ncols(),
        );
    }
    let fitted = design.design.apply(&fit.fit.beta);
    let truth: Array1<f64> = data.iter().map(|(_, _, y)| *y).collect();
    let residual = &truth - &fitted;
    println!(
        "    residual ss = {:.6e}   edf = {:?}",
        residual.dot(&residual),
        fit.fit.edf_total(),
    );
}

fn main() {
    println!(
        "the shipped gate on this statistic, WHEN a transform is applied, is 1e-8\n\
         (`ORTHOGONALITY_REL_RESIDUAL_TOL` in apply_global_smooth_identifiability)\n"
    );
    // Intercept only.
    measure("y ~ matern(x1, x2, centers=24)", &[]);
    measure("y ~ tps(x1, x2, centers=24)", &[]);
    measure("y ~ duchon(x1, x2, centers=24)", &[]);
    measure("y ~ curv(x1, x2, centers=24)", &[]);
    // Intercept PLUS an overlapping linear term: the estimand split is at stake.
    measure("y ~ x1 + matern(x1, x2, centers=24)", &[0]);
    measure("y ~ x1 + tps(x1, x2, centers=24)", &[0]);
    measure("y ~ x1 + duchon(x1, x2, centers=24)", &[0]);
    measure("y ~ x1 + curv(x1, x2, centers=24)", &[0]);
    // An OWNER smooth rather than a linear term: `analyze_smooth_ownership` makes
    // `s(x1)` own its subspace and residualizes the broader smooth against the
    // owner's REALIZED columns. That block is not the intercept and is not
    // contained in any kernel span either, so the containment gate withholds it
    // wholesale -- the hierarchy is the thing at stake here, not a convention.
    measure_owner("y ~ s(x1) + matern(x1, x2, centers=24)", "matern");
    measure_owner("y ~ s(x1) + curv(x1, x2, centers=24)", "curv");
    measure_owner("y ~ s(x1) + tps(x1, x2, centers=24)", "tps");
}
