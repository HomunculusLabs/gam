//! gam#2894 bar 2: the survival marginal-slope outer criterion prices `½·log|Zᵀ M_true Z|` on
//! the inner mode's active face, and within one active set its analytic ρ-gradient and ρ-Hessian
//! must be the derivatives of that value.
//!
//! The fixture is the #979 repro arm the defect was measured on: 160 observations, a Duchon
//! smooth over three PCs with six centers, a linear baseline. The seed probe evaluates the real
//! criterion at the seed, halfway to the returned optimum and at the optimum, and grades every
//! ρ coordinate's analytic gradient against a central difference of the criterion value itself.
//! At the optimum it also grades every column of the analytic outer Hessian, the curvature the
//! certificate reads (#2905), against a central difference of the analytic gradient. The
//! differences are formed here, because production takes no finite difference (SPEC rule 2).
//!
//! A stencil whose points straddle a change of active face differences two criteria, not one.
//! Its halving ladder does not settle, since successive estimates disagree by far more than
//! roundoff. That coordinate is reported and not graded, and each point must still grade most
//! of its coordinates.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_solve::estimate::outer_eval_capture::{
    OuterSeedOrder, OuterSeedProbe, observe_next_outer_seed,
};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

const N: usize = 160;
const CENTERS: usize = 6;
const N_PCS: usize = 3;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The `repro979_survival_margslope` generating process, bit for bit.
fn build_dataset() -> gam_data::EncodedDataset {
    let mut headers = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "prs_z".to_string(),
    ];
    for i in 0..N_PCS {
        headers.push(format!("PC{}", i + 1));
    }
    headers.push("sex".to_string());
    let mut state: u64 = 0xD0E1_2345_6789_ABCD;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    for _ in 0..N {
        let pcs: Vec<f64> = (0..N_PCS).map(|_| next_gauss(&mut state) * 0.5).collect();
        let prs = next_gauss(&mut state);
        let sex = if next_unit(&mut state) < 0.5 { 1.0 } else { 0.0 };
        let entry = 40.0 + 5.0 * next_unit(&mut state);
        let followup = 0.5 + 8.0 * next_unit(&mut state);
        let exit = entry + followup;
        let score = 0.3 * prs + 0.4 * pcs[0] - 0.3 * pcs[1] + 0.2 * pcs[2]
            + 0.15 * sex
            + 0.2 * next_gauss(&mut state);
        let event = if score > 0.0 { 1 } else { 0 };
        let mut record = vec![
            entry.to_string(),
            exit.to_string(),
            event.to_string(),
            prs.to_string(),
        ];
        for pc in &pcs {
            record.push(pc.to_string());
        }
        record.push(sex.to_string());
        rows.push(StringRecord::from(record));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #979 repro dataset")
}

fn fit_config() -> (String, FitConfig) {
    let duchon = format!("duchon(PC1, PC2, PC3, centers={CENTERS}, order=1)");
    let formula = format!("Surv(entry_age, exit_age, event) ~ {duchon} + sex");
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("prs_z".to_string()),
        slope_formula: Some(duchon),
        baseline_target: "linear".to_string(),
        ..FitConfig::default()
    };
    (formula, config)
}

/// One ρ coordinate's analytic gradient against its central difference.
#[derive(Debug)]
struct CoordinateGrade {
    point: &'static str,
    coordinate: usize,
    analytic: f64,
    difference: f64,
    step: f64,
    /// `|D(h) − D(h/2)|` at the accepted rung: the ladder's own disagreement.
    settle: f64,
}

/// One column of the analytic outer Hessian against a central difference of the analytic
/// gradient along that column's coordinate.
#[derive(Debug)]
struct HessianColumnGrade {
    column: usize,
    analytic: Array1<f64>,
    difference: Array1<f64>,
    step: f64,
    /// `max_i |D(h)_i − D(h/2)_i|` at the accepted rung.
    settle: f64,
}

/// The first rung of a central-difference ladder along coordinate `j`: one hundredth of the
/// coordinate's scale, and never more than half its room inside the seed box.
fn initial_step(probe: &dyn OuterSeedProbe, theta: &Array1<f64>, j: usize) -> Result<f64, String> {
    let layout = probe.layout();
    let room = (theta[j] - layout.lower[j])
        .min(layout.upper[j] - theta[j])
        .max(0.0);
    let step = (1.0e-2 * (1.0 + theta[j].abs())).min(0.5 * room);
    if !(step > 0.0) {
        return Err(format!("coordinate {j} sits on a face of the seed box"));
    }
    Ok(step)
}

/// Central differences of the criterion along coordinate `j` on a halving ladder, returning
/// the two finest estimates that agree best, the step, and their disagreement.
fn central_difference(
    probe: &mut dyn OuterSeedProbe,
    theta: &Array1<f64>,
    j: usize,
) -> Result<(f64, f64, f64), String> {
    let mut step = initial_step(probe, theta, j)?;
    let mut value_at = |offset: f64| -> Result<f64, String> {
        let mut displaced = theta.clone();
        displaced[j] += offset;
        probe
            .evaluate(&displaced, OuterSeedOrder::Value)
            .map(|evaluation| evaluation.cost)
            .map_err(|error| format!("theta[{j}] displaced by {offset:e}: {error}"))
    };
    let mut estimates: Vec<(f64, f64)> = Vec::new();
    for _ in 0..6 {
        let plus = value_at(step)?;
        let minus = value_at(-step)?;
        estimates.push((step, (plus - minus) / (2.0 * step)));
        step *= 0.5;
    }
    let (index, settle) = (1..estimates.len())
        .map(|i| (i, (estimates[i].1 - estimates[i - 1].1).abs()))
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .expect("the ladder has six rungs");
    Ok((estimates[index].1, estimates[index].0, settle))
}

/// Central differences of the analytic gradient along coordinate `j` on a halving ladder,
/// returning the column estimate of the rung that agrees best with its predecessor, the step,
/// and that rung's largest entrywise disagreement.
fn gradient_central_difference(
    probe: &mut dyn OuterSeedProbe,
    theta: &Array1<f64>,
    j: usize,
) -> Result<(Array1<f64>, f64, f64), String> {
    let mut step = initial_step(probe, theta, j)?;
    let mut gradient_at = |offset: f64| -> Result<Array1<f64>, String> {
        let mut displaced = theta.clone();
        displaced[j] += offset;
        probe
            .evaluate(&displaced, OuterSeedOrder::ValueAndGradient)
            .map_err(|error| format!("theta[{j}] displaced by {offset:e}: {error}"))?
            .gradient
            .ok_or_else(|| format!("theta[{j}] displaced by {offset:e}: no gradient published"))
    };
    let mut estimates: Vec<(f64, Array1<f64>)> = Vec::new();
    for _ in 0..6 {
        let plus = gradient_at(step)?;
        let minus = gradient_at(-step)?;
        estimates.push((step, (&plus - &minus) / (2.0 * step)));
        step *= 0.5;
    }
    let (index, settle) = (1..estimates.len())
        .map(|i| {
            let disagreement = (&estimates[i].1 - &estimates[i - 1].1)
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            (i, disagreement)
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .expect("the ladder has six rungs");
    Ok((estimates[index].1.clone(), estimates[index].0, settle))
}

fn grade_point(
    probe: &mut dyn OuterSeedProbe,
    point: &'static str,
    theta: &Array1<f64>,
) -> Result<Vec<CoordinateGrade>, String> {
    let evaluation = probe
        .evaluate(theta, OuterSeedOrder::ValueAndGradient)
        .map_err(|error| format!("{point}: analytic evaluation: {error}"))?;
    if !evaluation.cost.is_finite() {
        return Err(format!("{point}: the criterion is not finite: {}", evaluation.cost));
    }
    let gradient = evaluation
        .gradient
        .ok_or_else(|| format!("{point}: the gradient evaluation published no gradient"))?;
    let mut grades = Vec::with_capacity(gradient.len());
    for j in 0..gradient.len() {
        let (difference, step, settle) = central_difference(probe, theta, j)?;
        grades.push(CoordinateGrade {
            point,
            coordinate: j,
            analytic: gradient[j],
            difference,
            step,
            settle,
        });
    }
    Ok(grades)
}

fn grade_hessian(
    probe: &mut dyn OuterSeedProbe,
    theta: &Array1<f64>,
) -> Result<Vec<HessianColumnGrade>, String> {
    let evaluation = probe
        .evaluate(theta, OuterSeedOrder::ValueGradientHessian)
        .map_err(|error| format!("optimum: analytic Hessian evaluation: {error}"))?;
    let hessian = evaluation
        .hessian
        .ok_or_else(|| "optimum: the objective served no analytic outer Hessian".to_string())?;
    let dim = theta.len();
    if hessian.dim() != (dim, dim) {
        return Err(format!(
            "optimum: the analytic Hessian has shape {:?}, expected ({dim}, {dim})",
            hessian.dim()
        ));
    }
    let mut grades = Vec::with_capacity(dim);
    for column in 0..dim {
        let (difference, step, settle) = gradient_central_difference(probe, theta, column)?;
        grades.push(HessianColumnGrade {
            column,
            analytic: hessian.column(column).to_owned(),
            difference,
            step,
            settle,
        });
    }
    Ok(grades)
}

#[test]
fn survival_marginal_slope_face_criterion_derivatives_match_central_differences_2894() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    let data = build_dataset();
    let (formula, config) = fit_config();

    let optimum = match fit_from_formula(&formula, &data, &config) {
        Ok(FitResult::SurvivalMarginalSlope(result)) => result.fit.log_lambdas.clone(),
        Ok(_) => panic!("a marginal-slope survival formula returned another model class"),
        Err(error) => panic!("the 160x6 repro fit refused: {error}"),
    };

    type Grades = (Vec<CoordinateGrade>, Vec<HessianColumnGrade>);
    let captured: Rc<RefCell<Option<Result<Grades, String>>>> = Rc::new(RefCell::new(None));
    let sink = Rc::clone(&captured);
    observe_next_outer_seed(
        0,
        Box::new(
            move |probe: &mut dyn OuterSeedProbe| -> Result<(), gam_solve::estimate::EstimationError> {
                let layout = probe.layout().clone();
                let outcome = (|| -> Result<Grades, String> {
                    if optimum.len() != layout.seed.len() {
                        return Err(format!(
                            "the returned optimum has {} coordinates, the seed {}",
                            optimum.len(),
                            layout.seed.len()
                        ));
                    }
                    let clamp = |theta: Array1<f64>| {
                        Array1::from_shape_fn(theta.len(), |i| {
                            theta[i].clamp(layout.lower[i], layout.upper[i])
                        })
                    };
                    let seed = layout.seed.clone();
                    let halfway = clamp((&seed + &optimum) * 0.5);
                    let returned = clamp(optimum.clone());
                    let mut grades = Vec::new();
                    for (point, theta) in
                        [("seed", seed), ("halfway", halfway), ("optimum", returned.clone())]
                    {
                        grades.extend(grade_point(probe, point, &theta)?);
                    }
                    let hessian_grades = grade_hessian(probe, &returned)?;
                    Ok((grades, hessian_grades))
                })();
                *sink.borrow_mut() = Some(outcome);
                Ok(())
            },
        ),
    );
    let refit = fit_from_formula(&formula, &data, &config);
    let (grades, hessian_grades) = captured
        .borrow_mut()
        .take()
        .unwrap_or_else(|| panic!("the outer runner lent no seed probe: {:?}", refit.err()))
        .unwrap_or_else(|reason| panic!("the seed probe refused: {reason}"));

    let mut graded_per_point = std::collections::BTreeMap::<&'static str, usize>::new();
    let mut failures = Vec::new();
    for grade in &grades {
        let scale = grade.analytic.abs().max(1.0);
        let settled = grade.settle <= 1.0e-4 * scale;
        eprintln!(
            "[2894-FD] point={} coordinate={} analytic={:.9e} difference={:.9e} step={:e} \
             settle={:e} graded={settled}",
            grade.point, grade.coordinate, grade.analytic, grade.difference, grade.step, grade.settle,
        );
        if !settled {
            continue;
        }
        *graded_per_point.entry(grade.point).or_default() += 1;
        if (grade.analytic - grade.difference).abs() > 1.0e-4 * scale + 10.0 * grade.settle {
            failures.push(format!(
                "{} coordinate {}: analytic {:e} vs central difference {:e}",
                grade.point, grade.coordinate, grade.analytic, grade.difference
            ));
        }
    }
    let mut settled_columns = 0usize;
    for grade in &hessian_grades {
        let scale = grade
            .analytic
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let settled = grade.settle <= 1.0e-4 * scale;
        for row in 0..grade.analytic.len() {
            eprintln!(
                "[2894-FD-HESSIAN] row={row} column={} analytic={:.9e} difference={:.9e} \
                 step={:e} settle={:e} graded={settled}",
                grade.column, grade.analytic[row], grade.difference[row], grade.step, grade.settle,
            );
        }
        if !settled {
            continue;
        }
        settled_columns += 1;
        for row in 0..grade.analytic.len() {
            let entry_scale = grade.analytic[row].abs().max(1.0);
            if (grade.analytic[row] - grade.difference[row]).abs()
                > 1.0e-4 * entry_scale + 10.0 * grade.settle
            {
                failures.push(format!(
                    "optimum Hessian [{row}, {}]: analytic {:e} vs central difference {:e}",
                    grade.column, grade.analytic[row], grade.difference[row]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "derivatives disagree with the criterion: {failures:?}");
    for point in ["seed", "halfway", "optimum"] {
        let graded = graded_per_point.get(point).copied().unwrap_or(0);
        assert!(
            graded >= 4,
            "{point}: only {graded} of six coordinates had a settled central difference"
        );
    }
    assert!(
        settled_columns >= 4,
        "optimum: only {settled_columns} of {} Hessian columns had a settled central difference",
        hessian_grades.len()
    );
}
