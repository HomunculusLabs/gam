#![cfg(test)]
//! #2273 measurement probe: the outer REML criterion of an exactly-separated
//! binomial `smooth(x)` fit, evaluated as a pure function of ρ.
//!
//! The `y ~ smooth(x)` arm of #2273 fails at its SEED: `railed=[]`,
//! `after 0 outer iteration(s)`, `line_search=StepSizeTooSmall after 50
//! attempt(s)` at `rho_checkpoint = [1.0, 1.0]`. `StepSizeTooSmall` means the
//! direction WAS a descent direction and no step improved the objective, which
//! is only possible if the criterion the line search sees is not the
//! differentiable function the gradient claims — the #2519 genus.
//!
//! This probe builds the SAME design the production fit builds (formula →
//! `build_termspec` → `build_term_collection_design`), then evaluates the outer
//! criterion through `gam_solve::estimate::evaluate_externalcost_andridge` /
//! `evaluate_externalgradient`, each of which constructs a FRESH `RemlState`,
//! so every value here is the criterion at ρ alone with no carried warm start,
//! LM hint, or IFT extrapolation. What it reports:
//!
//!   1. the realized design (p, per-column norms, zero columns, penalty blocks)
//!   2. a ρ grid map of the criterion, marking infeasible/erroring points
//!   3. cost + analytic gradient at the seed ρ = (1, 1), against central FD
//!   4. the optimizer's own α ladder along −g/‖g‖ with the Armijo test
//!
//! It is a MEASUREMENT, not a contract: it asserts only what it has confirmed,
//! and prints the rest.

#![cfg(test)]

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam_terms::inference::formula_dsl::parse_formula;
use gam_terms::smooth::{TermCollectionDesign, build_term_collection_design};
use gam_terms::term_builder::build_termspec;
use ndarray::{Array1, Array2};

/// The issue's exact-separation fixture, identical to the one the regression
/// tests use: `n/2` rows of class 0 at `x = 1.0, 1.1, …` and `n/2` of class 1
/// at `x = 10.0, 10.1, …`, so the two class supports are separated by a gap.
fn separated_dataset(n: usize) -> gam_data::EncodedDataset {
    assert_eq!(n % 2, 0);
    let half = n / 2;
    let headers: Vec<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
    let mut rows = Vec::with_capacity(n);
    for i in 0..half {
        let x = 1.0 + 0.1 * i as f64;
        rows.push(StringRecord::from(vec![x.to_string(), "0".to_string()]));
    }
    for i in 0..half {
        let x = 10.0 + 0.1 * i as f64;
        rows.push(StringRecord::from(vec![x.to_string(), "1".to_string()]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Rebuild the production design for `y ~ smooth(x)` on the fixture.
fn separated_smooth_design(n: usize) -> (TermCollectionDesign, Array1<f64>, Array2<f64>) {
    let ds = separated_dataset(n);
    let parsed = parse_formula("y ~ smooth(x)").expect("the formula parses");
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let spec = build_termspec(&parsed.terms, &ds, &col_map, &mut notes, &policy)
        .expect("the term spec builds");
    let data = ds.values.clone();
    let design = build_term_collection_design(data.view(), &spec).expect("the design builds");
    let y_col = col_map["y"];
    let y = data.column(y_col).to_owned();
    (design, y, data)
}

fn logit_external_options(nullspace_dims: Vec<usize>) -> ExternalOptimOptions {
    binomial_external_options(nullspace_dims, StandardLink::Logit, false)
}

fn binomial_external_options(
    nullspace_dims: Vec<usize>,
    link: StandardLink,
    firth: bool,
) -> ExternalOptimOptions {
    // Mirrors `canonical_standard_fit_options` (`entry.rs`): the formula path's
    // outer tolerance is 1e-10 and its penalty shrinkage floor 1e-6.
    ExternalOptimOptions {
        family: LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link)),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: true,
        max_iter: 200,
        tol: 1e-10,
        nullspace_dims,
        linear_constraints: None,
        firth_bias_reduction: Some(firth),
        penalty_shrinkage_floor: Some(1e-6),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

/// #2273 / #2519: measure the exactly-separated `smooth(x)` outer criterion.
#[test]
fn separated_smooth_outer_criterion_probe_2273() {
    const N: usize = 60;
    let (design, y, data) = separated_smooth_design(N);
    let p = design.design.ncols();
    let weights = Array1::<f64>::ones(N);
    let offset = design
        .compose_offset(Array1::<f64>::zeros(N).view(), "probe")
        .expect("offset composes");

    // --- 1. the realized design -------------------------------------------
    let dense = design
        .design
        .try_to_dense_by_chunks("#2273 separated-smooth probe")
        .expect("the probe design materializes densely");
    let mut design_report = format!(
        "design: n={} p={p} penalties={} nullspace_dims={:?}\n  affine_offset|max|={:.3e}",
        design.design.nrows(),
        design.penalties.len(),
        design.nullspace_dims,
        design
            .affine_offset
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs())),
    );
    for col in 0..p {
        let column = dense.column(col);
        let norm = column.iter().map(|v| v * v).sum::<f64>().sqrt();
        let max_abs = column.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        design_report.push_str(&format!(
            "\n  col {col:>2}: ||.||={norm:.6e}  max|.|={max_abs:.6e}{}",
            if norm <= 1e-300 { "   <-- EXACTLY ZERO" } else { "" }
        ));
    }
    for (idx, penalty) in design.penalties.iter().enumerate() {
        let block = &penalty.local;
        let trace = (0..block.nrows().min(block.ncols()))
            .map(|i| block[[i, i]])
            .sum::<f64>();
        design_report.push_str(&format!(
            "\n  penalty {idx}: cols {:?} dim {}x{} trace={trace:.6e}",
            penalty.col_range,
            block.nrows(),
            block.ncols(),
        ));
    }
    eprintln!("#2273 probe {design_report}");

    let opts = logit_external_options(design.nullspace_dims.clone());
    let rho_dim = design.nullspace_dims.len();
    let cost_at = |rho: &Array1<f64>| -> (Option<f64>, Option<f64>, String) {
        match evaluate_externalcost_andridge(
            y.view(),
            weights.view(),
            design.design.clone(),
            offset.view(),
            &design.penalties,
            &opts,
            rho,
        ) {
            Ok((cost, ridge)) if cost.is_finite() => {
                (Some(cost), Some(ridge), format!("{cost:.12e}"))
            }
            Ok((cost, ridge)) => (None, Some(ridge), format!("NON-FINITE {cost}")),
            Err(err) => (None, None, format!("ERR {err}")),
        }
    };
    let grad_at = |rho: &Array1<f64>| -> Result<Array1<f64>, String> {
        evaluate_externalgradient(
            y.view(),
            weights.view(),
            design.design.clone(),
            offset.view(),
            &design.penalties,
            &opts,
            rho,
        )
        .map_err(|err| err.to_string())
    };

    // --- 2. the rho grid ---------------------------------------------------
    assert_eq!(
        rho_dim, 2,
        "the probe's ladder assumes the (range, null-space) double penalty"
    );
    let axis = [-12.0_f64, -8.0, -4.0, -2.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    let mut grid = String::from("\n  rho1 \\ rho2");
    for r2 in axis {
        grid.push_str(&format!("  {r2:>+8.1}"));
    }
    for r1 in axis {
        grid.push_str(&format!("\n  {r1:>+8.1}   "));
        for r2 in axis {
            let (cost, _, _) = cost_at(&Array1::from(vec![r1, r2]));
            match cost {
                Some(value) => grid.push_str(&format!(" {value:>+9.3}"),),
                None => grid.push_str("       inf"),
            }
        }
    }
    eprintln!("#2273 probe outer criterion on a rho grid (fresh state per call):{grid}");

    // --- 3. the seed: cost, gradient, FD -----------------------------------
    let rho0 = Array1::from(vec![1.0_f64, 1.0]);
    let (seed_cost, seed_ridge, seed_label) = cost_at(&rho0);
    let g = grad_at(&rho0);
    let mut seed_report = format!(
        "seed rho0=[1,1]: cost={seed_label} ridge={:?} gradient={:?}",
        seed_ridge,
        g.as_ref().map(|g| format!("{g:.9e}")),
    );
    if let (Some(f0), Ok(g)) = (seed_cost, g.as_ref()) {
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        seed_report.push_str(&format!("\n  |g|={gnorm:.9e}"));
        for coord in 0..rho_dim {
            for h in [1.0e-3_f64, 1.0e-4, 1.0e-5] {
                let mut plus = rho0.clone();
                let mut minus = rho0.clone();
                plus[coord] += h;
                minus[coord] -= h;
                let (fp, _, fp_label) = cost_at(&plus);
                let (fm, _, fm_label) = cost_at(&minus);
                let text = match (fp, fm) {
                    (Some(fp), Some(fm)) => {
                        let fd = (fp - fm) / (2.0 * h);
                        format!("fd={fd:+.9e}  fd/analytic={:+.6}", fd / g[coord])
                    }
                    _ => format!("fd=n/a (+: {fp_label}; -: {fm_label})"),
                };
                seed_report.push_str(&format!(
                    "\n  d/drho[{coord}] h={h:.0e}  analytic={:+.9e}  {text}",
                    g[coord]
                ));
            }
        }

        // --- 4. the optimizer's own alpha ladder ---------------------------
        let d = g.mapv(|v| -v / gnorm);
        let c1 = 1.0e-4_f64;
        let mut accepted_alpha: Option<f64> = None;
        let mut finite = 0usize;
        for alpha in [
            1.0e0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9,
            1.0e-10, 1.0e-11, 1.0e-12,
        ] {
            let trial = &rho0 + &d.mapv(|v| v * alpha);
            let (cost, ridge, label) = cost_at(&trial);
            let target = -c1 * alpha * gnorm;
            let (delta_text, accept) = match cost {
                Some(f) => {
                    finite += 1;
                    let delta = f - f0;
                    (format!("{delta:+.6e}"), delta <= target)
                }
                None => ("n/a".to_string(), false),
            };
            if accept && accepted_alpha.is_none() {
                accepted_alpha = Some(alpha);
            }
            seed_report.push_str(&format!(
                "\n  alpha={alpha:.0e}  f={label}  ridge={ridge:?}  f-f0={delta_text}  \
                 armijo<={target:+.6e}  accept={accept}"
            ));
        }
        seed_report.push_str(&format!(
            "\n  finite ladder probes: {finite}/13; first accepting alpha: {accepted_alpha:?}"
        ));
    }
    eprintln!("#2273 probe {seed_report}");

    // --- 5. what the production path does with the same data ---------------
    let ds = separated_dataset(N);
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula("y ~ smooth(x)", &ds, &cfg) {
        Ok(FitResult::Standard(StandardFitResult { fit, .. })) => eprintln!(
            "#2273 probe production fit MINTED edf={:.4} |g|={:.3e}",
            fit.edf_total().unwrap_or(f64::NAN),
            fit.outer_gradient_norm.unwrap_or(f64::NAN),
        ),
        Ok(_) => eprintln!("#2273 probe production fit returned a non-standard result"),
        Err(err) => eprintln!("#2273 probe production fit REFUSED: {err}"),
    }

    // The one thing this probe asserts: the seed the production fit reported as
    // its checkpoint must be evaluable at all. Everything else is printed.
    assert!(
        seed_cost.is_some(),
        "the seed rho the production fit carried as its checkpoint is not even \
         evaluable from a fresh state: {seed_label}"
    );
    assert_eq!(
        data.nrows(),
        N,
        "the probe design was built from a frame that dropped rows"
    );
}

/// Evaluate one arm's criterion: a ρ grid, then cost/analytic-gradient/central-FD
/// and the optimizer's own Armijo ladder at `rho_of_interest`.
fn report_criterion_arm(
    label: &str,
    design: &TermCollectionDesign,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    offset: &Array1<f64>,
    opts: &ExternalOptimOptions,
    rho_axis: &[f64],
    rho_of_interest: &Array1<f64>,
) {
    let cost_at = |rho: &Array1<f64>| -> (Option<f64>, Option<f64>, String) {
        match evaluate_externalcost_andridge(
            y.view(),
            weights.view(),
            design.design.clone(),
            offset.view(),
            &design.penalties,
            opts,
            rho,
        ) {
            Ok((cost, ridge)) if cost.is_finite() => {
                (Some(cost), Some(ridge), format!("{cost:.12e}"))
            }
            Ok((cost, ridge)) => (None, Some(ridge), format!("NON-FINITE {cost}")),
            Err(err) => (None, None, format!("ERR {err}")),
        }
    };
    let grad_at = |rho: &Array1<f64>| -> Result<Array1<f64>, String> {
        evaluate_externalgradient(
            y.view(),
            weights.view(),
            design.design.clone(),
            offset.view(),
            &design.penalties,
            opts,
            rho,
        )
        .map_err(|err| err.to_string())
    };

    let mut grid = String::from("\n  rho1 \\ rho2");
    for r2 in rho_axis {
        grid.push_str(&format!("  {r2:>+8.1}"));
    }
    let mut best: Option<(f64, f64, f64)> = None;
    for &r1 in rho_axis {
        grid.push_str(&format!("\n  {r1:>+8.1}   "));
        for &r2 in rho_axis {
            let (cost, _, _) = cost_at(&Array1::from(vec![r1, r2]));
            match cost {
                Some(value) => {
                    grid.push_str(&format!(" {value:>+9.3}"));
                    if best.is_none_or(|(b, _, _)| value < b) {
                        best = Some((value, r1, r2));
                    }
                }
                None => grid.push_str("       inf"),
            }
        }
    }
    eprintln!("#2273 probe [{label}] criterion grid:{grid}\n  grid minimum: {best:?}");

    let (f0, ridge0, label0) = cost_at(rho_of_interest);
    let g = grad_at(rho_of_interest);
    let mut report = format!(
        "rho={rho_of_interest:.9e}: cost={label0} ridge={ridge0:?} g={:?}",
        g.as_ref().map(|g| format!("{g:.9e}")),
    );
    if let (Some(f0), Ok(g)) = (f0, g.as_ref()) {
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        report.push_str(&format!("\n  |g|={gnorm:.9e}"));
        for coord in 0..rho_of_interest.len() {
            for h in [1.0e-2_f64, 1.0e-3, 1.0e-4, 1.0e-5] {
                let mut plus = rho_of_interest.clone();
                let mut minus = rho_of_interest.clone();
                plus[coord] += h;
                minus[coord] -= h;
                let (fp, _, fp_label) = cost_at(&plus);
                let (fm, _, fm_label) = cost_at(&minus);
                let text = match (fp, fm) {
                    (Some(fp), Some(fm)) => {
                        let fd = (fp - fm) / (2.0 * h);
                        format!(
                            "fd={fd:+.9e}  fd-analytic={:+.3e}  fd/analytic={:+.6}",
                            fd - g[coord],
                            fd / g[coord]
                        )
                    }
                    _ => format!("fd=n/a (+: {fp_label}; -: {fm_label})"),
                };
                report.push_str(&format!(
                    "\n  d/drho[{coord}] h={h:.0e}  analytic={:+.9e}  {text}",
                    g[coord]
                ));
            }
        }
        let d = g.mapv(|v| -v / gnorm);
        let c1 = 1.0e-4_f64;
        let mut accepted: Option<f64> = None;
        for alpha in [
            1.0e0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-10,
            1.0e-12,
        ] {
            let trial = rho_of_interest + &d.mapv(|v| v * alpha);
            let (cost, _, label) = cost_at(&trial);
            let target = -c1 * alpha * gnorm;
            let (delta_text, accept) = match cost {
                Some(f) => {
                    let delta = f - f0;
                    (format!("{delta:+.6e}"), delta <= target)
                }
                None => ("n/a".to_string(), false),
            };
            if accept && accepted.is_none() {
                accepted = Some(alpha);
            }
            report.push_str(&format!(
                "\n  alpha={alpha:.0e}  f={label}  f-f0={delta_text}  \
                 armijo<={target:+.6e}  accept={accept}"
            ));
        }
        report.push_str(&format!("\n  first accepting alpha: {accepted:?}"));
    }
    eprintln!("#2273 probe [{label}] {report}");
}

/// #2273: three arms of the SAME separated `smooth(x)` design.
///
/// `detect_logit_instability` (`pirls/loop_driver.rs`) returns `false` unless the
/// link is logit and Firth is off, so the three arms differ in exactly whether
/// that detector runs:
///
/// | arm | detector runs | criterion |
/// |---|---|---|
/// | binomial/logit | YES | plain LAML |
/// | binomial/probit | no (link gate) | plain LAML |
/// | binomial/logit + Firth | no (firth gate) | Firth-penalized LAML |
///
/// If the logit arm's `+inf` wall is the DETECTOR rather than a genuinely
/// unsolvable inner problem, the probit arm — numerically the same shape of
/// problem — is finite where logit is infinite. That is the control this test
/// exists to run.
#[test]
fn separated_smooth_criterion_arms_probe_2273() {
    const N: usize = 60;
    let (design, y, _) = separated_smooth_design(N);
    let weights = Array1::<f64>::ones(N);
    let offset = design
        .compose_offset(Array1::<f64>::zeros(N).view(), "probe")
        .expect("offset composes");
    let dims = design.nullspace_dims.clone();
    let axis = [-12.0_f64, -8.0, -4.0, -2.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    let seed = Array1::from(vec![1.0_f64, 1.0]);

    report_criterion_arm(
        "logit (detector ON)",
        &design,
        &y,
        &weights,
        &offset,
        &binomial_external_options(dims.clone(), StandardLink::Logit, false),
        &axis,
        &seed,
    );
    report_criterion_arm(
        "probit (detector OFF via link gate)",
        &design,
        &y,
        &weights,
        &offset,
        &binomial_external_options(dims.clone(), StandardLink::Probit, false),
        &axis,
        &seed,
    );
    // The Firth arm's reported checkpoint from the production refusal, so the
    // FD/ladder columns are measured exactly where the fit gave up.
    let firth_checkpoint = Array1::from(vec![-3.999035662000272_f64, 25.906659992355834]);
    report_criterion_arm(
        "logit + Firth (detector OFF via firth gate)",
        &design,
        &y,
        &weights,
        &offset,
        &binomial_external_options(dims, StandardLink::Logit, true),
        &[-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 8.0, 16.0, 24.0, 28.0],
        &firth_checkpoint,
    );
}

/// #2273 arm 2: the exactly-separated **probit** `y ~ x` fit at n=6, whose only
/// route is the Firth rescue.
///
/// The refusal is `inner status StalledAtValidMinimum, outer status outer
/// evidence was not considered because the inner mode did not report
/// convergence, after 0 outer iteration(s); final objective 2.729456e0;
/// stationarity residual 0.000e0 against bound 8.941e-8`. `y ~ x` carries no
/// penalty, so ρ is empty and "0 outer iterations" is not a stall — there is no
/// outer problem at all. The whole failure is ONE Firth-probit P-IRLS solve on 6
/// rows and 2 coefficients.
///
/// An independent reference (scipy Nelder-Mead on `ℓ + ½log|XᵀWX|`, W the probit
/// Fisher weight) puts that solve's answer at
///
/// ```text
///   beta = (0.000, 1.3468)   max|eta| = 1.377   w = 0.310 .. 0.330
///   eig(I) = (1.9183, 1.9201)  cond(I) = 1.0009   0.5*log|I| = 0.65191
/// ```
///
/// so the target is as well-conditioned as a 2-parameter problem gets: no
/// underflow, no saturation, condition number one. Whatever makes P-IRLS run its
/// iteration budget out here is not the data.
///
/// This probe forwards the engine's `[PIRLS] iter …` debug lines so the
/// iteration history is visible, and prints the refusal verbatim.
#[test]
fn separated_probit_firth_inner_solve_probe_2273() {
    struct ForwardingTestLogger;
    impl log::Log for ForwardingTestLogger {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::max_level()
        }
        fn log(&self, record: &log::Record<'_>) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
        fn flush(&self) {}
    }
    static FORWARDING_TEST_LOGGER: ForwardingTestLogger = ForwardingTestLogger;
    // Ignore the error when another test already installed a global logger.
    if log::set_logger(&FORWARDING_TEST_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Debug);
    }

    let ds = separated_dataset(6);
    for (formula, firth) in [
        ("y ~ x + link(type=probit)", true),
        ("y ~ x + link(type=cloglog)", true),
        ("y ~ x + link(type=cloglog)", false),
    ] {
        let cfg = FitConfig {
            family: Some("binomial".to_string()),
            firth,
            ..FitConfig::default()
        };
        eprintln!("#2273 arm-2 probe ===== {formula} n=6 firth={firth} =====");
        match fit_from_formula(formula, &ds, &cfg) {
            Ok(FitResult::Standard(StandardFitResult { fit, .. })) => eprintln!(
                "#2273 arm-2 probe MINTED edf={:.6} beta={:?}",
                fit.edf_total().unwrap_or(f64::NAN),
                fit.beta.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>(),
            ),
            Ok(_) => eprintln!("#2273 arm-2 probe non-standard result"),
            Err(err) => eprintln!("#2273 arm-2 probe REFUSED: {err}"),
        }
    }
}

/// #2273: the premise the in-loop separation retreat's removal rests on —
/// `S(λ)` is positive definite on every column it touches.
///
/// `detect_logit_instability` no longer refuses a penalized binomial fit for
/// saturating, because under a coercive penalty `β̂(λ)` is finite and unique even
/// under exact separation, so saturation is the answer rather than a divergence.
/// Coercivity holds iff no direction of recession of `−ℓ` lies in `null(S(λ))`,
/// and for these designs `null(S(λ))` is exactly the span of the columns NO
/// penalty touches — the intercept and any bare parametric term, where a
/// separating direction is caught up front by `reject_prefit_binomial_separation`
/// (which restricts itself to those same columns, and is right to).
///
/// That equality is a property of the DESIGN CONSTRUCTION, not of this issue: a
/// smooth whose penalty set left its own linear direction unpenalized would put
/// a separating direction back into `null(S(λ))` without any prefit column being
/// implicated, and the retreat's removal would then be unsound for that basis.
/// So it is asserted here rather than assumed: for every penalized column block,
/// the summed penalty at λ = 1 must be strictly positive definite.
#[test]
fn penalized_blocks_are_coercive_for_every_smooth_2273() {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;

    for formula in [
        "y ~ smooth(x)",
        "y ~ x",
        "y ~ x + smooth(x)",
        "y ~ smooth(x) + smooth(z)",
        "y ~ te(x, z)",
    ] {
        let ds = coercivity_dataset(60);
        let parsed = parse_formula(formula).expect("the formula parses");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let spec = build_termspec(&parsed.terms, &ds, &col_map, &mut notes, &policy)
            .expect("the term spec builds");
        let design = build_term_collection_design(ds.values.view(), &spec)
            .expect("the design builds");
        let p = design.design.ncols();

        // S(1) = Σ_k S_k, embedded into the full coefficient space.
        let mut s_total = Array2::<f64>::zeros((p, p));
        for penalty in &design.penalties {
            for (local_row, global_row) in penalty.col_range.clone().enumerate() {
                for (local_col, global_col) in penalty.col_range.clone().enumerate() {
                    s_total[[global_row, global_col]] += penalty.local[[local_row, local_col]];
                }
            }
        }

        // A column is "touched" when the penalty sum gives it any curvature at
        // all. `null(S)` must be exactly the complement.
        let scale = s_total
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(1.0);
        let touched: Vec<usize> = (0..p)
            .filter(|&col| s_total[[col, col]].abs() > 1e-12 * scale)
            .collect();
        assert!(
            !touched.is_empty() || formula == "y ~ x",
            "#2273: {formula} produced no penalized column at all (p={p})"
        );
        if touched.is_empty() {
            continue;
        }

        let block = {
            let mut block = Array2::<f64>::zeros((touched.len(), touched.len()));
            for (i, &row) in touched.iter().enumerate() {
                for (j, &col) in touched.iter().enumerate() {
                    block[[i, j]] = s_total[[row, col]];
                }
            }
            block
        };
        assert!(
            block.cholesky(Side::Lower).is_ok(),
            "#2273: {formula}'s summed penalty is NOT positive definite on the {} columns \
             it touches, so `null(S(λ))` is larger than the unpenalized columns the prefit \
             separation check inspects, and a penalized binomial fit on separable data can \
             diverge along a direction nothing refuses. p={p}, touched={touched:?}",
            touched.len(),
        );
    }
}

/// Two numeric covariates so the multi-smooth and tensor formulas above have
/// something to build on; `y` stays the separated response.
fn coercivity_dataset(n: usize) -> gam_data::EncodedDataset {
    assert_eq!(n % 2, 0);
    let half = n / 2;
    let headers: Vec<String> = ["x", "z", "y"].iter().map(|s| s.to_string()).collect();
    let mut rows = Vec::with_capacity(n);
    for i in 0..half {
        let x = 1.0 + 0.1 * i as f64;
        rows.push(StringRecord::from(vec![
            x.to_string(),
            (0.5 * x).to_string(),
            "0".to_string(),
        ]));
    }
    for i in 0..half {
        let x = 10.0 + 0.1 * i as f64;
        rows.push(StringRecord::from(vec![
            x.to_string(),
            (0.5 * x - 3.0).to_string(),
            "1".to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// #2273: the outer criterion of an exactly-separated penalized binomial fit must
/// be FINITE wherever the penalized problem is coercive.
///
/// This is the arm-1 fix asserted on the criterion itself rather than through the
/// optimizer, so it stays meaningful if the outer search is ever re-planned. The
/// grid deliberately covers the region the retreat used to delete — every
/// `ρ_range ≤ −2` and every `ρ_null ≤ 0` returned `+∞` before, including the
/// nodes carrying the interior minimum.
///
/// `ρ_range = −8` and below is excluded, and legitimately: there only the rank-1
/// null-space penalty is left on an 11-dimensional smooth block, so the objective
/// really is unbounded along the separating direction and `+∞` is the honest
/// reply. The boundary between the two regimes is the point of the test.
#[test]
fn separated_smooth_criterion_is_finite_where_the_penalty_is_coercive_2273() {
    for n in [40usize, 60, 80, 100] {
        let (design, y, _) = separated_smooth_design(n);
        let weights = Array1::<f64>::ones(n);
        let offset = design
            .compose_offset(Array1::<f64>::zeros(n).view(), "coercive-criterion guard")
            .expect("offset composes");
        let opts = logit_external_options(design.nullspace_dims.clone());
        let mut infinite = Vec::new();
        for &rho_range in &[-4.0_f64, -2.0, 0.0, 1.0, 2.0, 4.0] {
            for &rho_null in &[-8.0_f64, -4.0, -2.0, 0.0, 1.0, 2.0] {
                let rho = Array1::from(vec![rho_range, rho_null]);
                let value = evaluate_externalcost_andridge(
                    y.view(),
                    weights.view(),
                    design.design.clone(),
                    offset.view(),
                    &design.penalties,
                    &opts,
                    &rho,
                );
                match value {
                    Ok((cost, _)) if cost.is_finite() => {}
                    Ok((cost, _)) => infinite.push(format!("({rho_range},{rho_null})->{cost}")),
                    Err(err) => infinite.push(format!("({rho_range},{rho_null})->ERR {err}")),
                }
            }
        }
        assert!(
            infinite.is_empty(),
            "#2273: the exactly-separated `smooth(x)` criterion is not a finite function of rho \
             at n={n} on the coercive part of the box, so the optimum is unreachable: {}",
            infinite.join(", "),
        );
    }
}
