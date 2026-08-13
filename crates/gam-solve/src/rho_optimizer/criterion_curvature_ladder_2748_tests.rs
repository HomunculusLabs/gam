//! #2748 — the certificate MEASURES the Hessian it disputes, instead of only
//! failing to falsify it.
//!
//! # The defect, as a property rather than an instance
//!
//! `adjudicate_negative_curvature` evaluates the criterion on both sides of the
//! certified point along the disputed eigenvector. That is a symmetric probe
//! ladder — the exact instrument `gam_linalg::curvature_resolution`'s header
//! says `ε_f` and `M₄` "come free from" — and its evaluations were being spent
//! on one boolean and discarded.
//!
//! What was discarded is the only *"how wrong is this MATRIX?"* measurement
//! anywhere on the path. Downstream,
//! `estimate::smoothing_correction::invert_identified_rho_hessian` judges the
//! SAME matrix at the SAME point and, absent that measurement, judges it
//! against an eigensolver's backward error — a bound on the DECOMPOSITION,
//! which the module doc says must not be handed to a site asking about the
//! assembly. Measured on `geo_disease_matern`: `2.396439e-16` against an
//! intrinsic curvature of `-2.47e-8` that decided the whole verdict, and the
//! two subsystems then reached opposite verdicts on one matrix at one point.
//!
//! Every fixture here plants the criterion FIRST and derives the Hessian the
//! adjudication is handed from it, so "the ladder recovers the curvature" is a
//! statement about a known number rather than about a recorded run.

use super::*;
use ndarray::{Array1, Array2, array};

/// A criterion that is exactly `V(θ) = V₀ + g·θ + ½θᵀCθ + (M₄/24)(v·θ)⁴`, and
/// an analytic Hessian the adjudication is handed that may DISAGREE with `C`.
///
/// The quartic is along the probed direction only, which is what gives the
/// ladder an `M₄` to find; without it the fit's slope column is exactly zero
/// and the two-parameter design is degenerate — a property the ladder measure
/// itself refuses on, and one this fixture must therefore not present by
/// accident.
struct PlantedCriterion {
    baseline: f64,
    gradient: Array1<f64>,
    criterion_curvature: Array2<f64>,
    quartic_direction: Array1<f64>,
    fourth_derivative: f64,
    evaluations: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl PlantedCriterion {
    fn value(&self, theta: &Array1<f64>) -> f64 {
        let projection = self.quartic_direction.dot(theta);
        self.baseline
            + self.gradient.dot(theta)
            + 0.5 * theta.dot(&self.criterion_curvature.dot(theta))
            + self.fourth_derivative * projection.powi(4) / 24.0
    }
}

impl OuterObjective for PlantedCriterion {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Dense,
            n_params: self.gradient.len(),
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        }
    }
    fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
        *self
            .evaluations
            .lock()
            .expect("the evaluation counter is not poisoned") += 1;
        Ok(self.value(theta))
    }
    fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        Ok(OuterEval {
            cost: self.value(theta),
            gradient: &self.gradient + &self.criterion_curvature.dot(theta),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        })
    }
    fn eval_with_order(
        &mut self,
        theta: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        // A planted criterion has closed-form derivatives at every order, so
        // the value-only lane must agree with the derivative-bearing one at the
        // same theta. Honouring the order rather than ignoring it is what makes
        // that agreement a property of the fixture instead of an accident.
        let full = self.eval(theta)?;
        Ok(match order {
            OuterEvalOrder::Value => OuterEval {
                cost: full.cost,
                gradient: Array1::<f64>::zeros(theta.len()),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            },
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => full,
        })
    }
    fn reset(&mut self) {
        // The planted criterion is a pure function of theta: there is no warm
        // state to re-baseline, and saying so is part of the fixture's claim
        // that every evaluation at one theta returns one value.
    }
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "the planted criterion was offered a non-finite inner seed".to_string(),
            ));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

/// The configuration every test below varies: a 3-coordinate criterion whose
/// curvature along `v = (1,0,-1)/√2` is `criterion_vv`, handed an analytic
/// Hessian claiming `analytic_vv` there and agreeing everywhere else.
struct Planted {
    criterion_vv: f64,
    analytic_vv: f64,
    fourth_derivative: f64,
    gradient_scale: f64,
}

/// Build the (objective, θ̂, g, H_analytic, v) the adjudication is handed.
fn planted(spec: Planted) -> (PlantedCriterion, Array1<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let root_half = 0.5_f64.sqrt();
    let v = array![root_half, 0.0, -root_half];
    let u1 = array![root_half, 0.0, root_half];
    let u2 = array![0.0_f64, 1.0, 0.0];
    // Well-curved off the probed direction, so `v` is unambiguously the
    // minimum-curvature eigenvector of both matrices.
    let (a, b) = (1.0_f64, 0.5_f64);
    let build = |vv: f64| {
        let mut matrix = Array2::<f64>::zeros((3, 3));
        for r in 0..3 {
            for c in 0..3 {
                matrix[[r, c]] = a * u1[r] * u1[c] + b * u2[r] * u2[c] + vv * v[r] * v[c];
            }
        }
        matrix
    };
    // A residual gradient of one sign on every coordinate, so the chain-rule
    // floor `Σ|g_k| v_k²` is unambiguous and equals `|g|` on this `v`.
    let gradient = Array1::from(vec![-spec.gradient_scale; 3]);
    let objective = PlantedCriterion {
        baseline: 2.224446222e3,
        gradient: gradient.clone(),
        criterion_curvature: build(spec.criterion_vv),
        quartic_direction: v.clone(),
        fourth_derivative: spec.fourth_derivative,
        evaluations: std::sync::Arc::new(std::sync::Mutex::new(0)),
    };
    (
        objective,
        Array1::<f64>::zeros(3),
        gradient,
        build(spec.analytic_vv),
        v,
    )
}

fn wide_bounds() -> (Array1<f64>, Array1<f64>) {
    (Array1::from(vec![-30.0; 3]), Array1::from(vec![30.0; 3]))
}

/// `geo_disease_matern`'s own numbers, planted.
///
/// At the refusing point the site reported `σ = -6.404082e-6` with a chain-rule
/// term of `-6.379352e-6`, i.e. an intrinsic `-2.47e-8`, and refused against a
/// resolution of `2.396439e-16`. The measured criterion curvature there was
/// `+8.153228e-5` — the OPPOSITE SIGN and thirteen times the magnitude. This
/// plants that disagreement and asserts the ladder finds it.
#[test]
fn the_ladder_recovers_a_criterion_curvature_the_analytic_hessian_got_wrong_2748() {
    let criterion_vv = 8.153228e-5_f64;
    let analytic_vv = -6.404082e-6_f64;
    let (mut objective, theta, gradient, hessian, _direction) = planted(Planted {
        criterion_vv,
        analytic_vv,
        fourth_derivative: 1.094027e-2,
        gradient_scale: 6.379352e-6,
    });
    let counter = std::sync::Arc::clone(&objective.evaluations);
    let baseline = objective.value(&theta);
    let bounds = wide_bounds();

    let verdict = adjudicate_negative_curvature(
        &mut objective,
        &theta,
        &gradient,
        &hessian,
        &[],
        None,
        baseline,
        // The criterion's declared resolution, as the outer loop supplies it:
        // large enough that the claim is unfalsifiable by descent alone, which
        // is exactly the regime the measurement exists for.
        2.225e-4,
        &bounds,
        "planted #2748 fixture",
    );

    let SaddleAdjudication::Contradicted {
        criterion_curvature,
        ..
    } = verdict
    else {
        panic!("a claim whose predicted descent is below the criterion's resolution cannot be falsified by descent, so the adjudication must reach Contradicted: {verdict:?}");
    };
    let measured =
        criterion_curvature.expect("the extended ladder must determine a fit on a smooth planted criterion");

    assert!(
        (measured.ladder.curvature - criterion_vv).abs() <= 1.0e-2 * criterion_vv.abs(),
        "the ladder must recover the PLANTED criterion curvature: got {:.6e}, planted \
         {criterion_vv:.6e}",
        measured.ladder.curvature
    );
    // The eigensolver's own backward error, not a bitwise identity: the claim
    // carried forward is the eigenvalue as DECOMPOSED, which is the number the
    // downstream gate would have refused on.
    // The bar is the eigensolver's own backward error, which is ABSOLUTE in
    // `‖H‖` (here 1.0) and therefore enormous RELATIVE to a `6e-6` eigenvalue.
    // Asserting a relative agreement on a near-null eigenvalue would be
    // asserting a precision Weyl does not offer -- the same category error this
    // whole issue is about, one level down.
    let eigen_backward_error = 64.0 * 3.0 * f64::EPSILON;
    assert!(
        (measured.analytic_curvature - analytic_vv).abs() <= eigen_backward_error,
        "the disagreement must be reported against the analytic claim the gate would see:          {:.17e} vs {analytic_vv:.17e} at backward error {eigen_backward_error:.3e}",
        measured.analytic_curvature
    );
    // The measured `‖δH‖₂`: the disagreement, net of the ladder's own error bar.
    let disagreement = (analytic_vv - criterion_vv).abs();
    let reported = measured.hessian_error_2norm();
    assert!(
        reported > 0.5 * disagreement && reported <= disagreement,
        "the measured ||dH||_2 must be the disagreement net of the ladder's uncertainty: \
         reported {reported:.6e}, disagreement {disagreement:.6e}, uncertainty {:.6e}",
        measured.ladder.curvature_uncertainty
    );
    // And it is enough to make the downstream gate stop refusing, which is the
    // whole point of carrying it (#2428).
    assert!(
        reported > analytic_vv.abs(),
        "the measured assembly error must cover the eigenvalue the gate would refuse on: \
         {reported:.6e} vs {:.6e}",
        analytic_vv.abs()
    );
    // The extension is not free and must not be silent about it. One rung is
    // two evaluations; the falsifiability ladder alone would have been 2.
    assert!(
        *counter.lock().expect("counter") > 4,
        "the ladder must have been EXTENDED past the single falsifiability rung"
    );
}

/// NEGATIVE CONTROL, and the one that keeps this from being a licence: when the
/// analytic Hessian is HONEST, the ladder agrees and measures exactly nothing.
///
/// Without this the routine could report a magnitude on every adjudicated fit
/// and silently widen every downstream curvature gate. In production the same
/// control fired on `geo_disease_matern`'s iso-kappa joint arm:
/// `c_criterion = -1.060226e-6` against `λ_min = -1.060914e-6`, a measured
/// `‖δH‖₂` of `6.3e-10` — six orders below the chain-rule term, so it moved
/// nothing.
#[test]
fn an_honest_analytic_hessian_measures_no_error_at_all_2748() {
    let curvature = -1.060914e-6_f64;
    let (mut objective, theta, gradient, hessian, _direction) = planted(Planted {
        criterion_vv: curvature,
        analytic_vv: curvature,
        fourth_derivative: 3.0e-3,
        gradient_scale: 1.06e-6,
    });
    let baseline = objective.value(&theta);
    let bounds = wide_bounds();

    let verdict = adjudicate_negative_curvature(
        &mut objective,
        &theta,
        &gradient,
        &hessian,
        &[],
        None,
        baseline,
        2.095e-3,
        &bounds,
        "planted #2748 honest-Hessian control",
    );
    let SaddleAdjudication::Contradicted {
        criterion_curvature,
        ..
    } = verdict
    else {
        panic!("the planted descent is below the criterion's declared resolution, so no trial can descend: {verdict:?}");
    };
    let measured = criterion_curvature.expect("the ladder must determine a fit here too");
    assert!(
        (measured.ladder.curvature - curvature).abs() <= 1.0e-2 * curvature.abs(),
        "the ladder must recover the planted curvature on the honest fixture too: got {:.6e}",
        measured.ladder.curvature
    );
    assert_eq!(
        measured.hessian_error_2norm(),
        0.0,
        "an honest Hessian must measure NO assembly error, or every gate downstream widens \
         on every adjudicated fit"
    );
}

/// A REAL saddle is confirmed, not excused. The criterion genuinely descends
/// along `v`, so the adjudication never reaches the measurement at all — it
/// mints the escape reseed, exactly as before.
///
/// This is the assertion that the ladder cannot convert a genuine saddle into
/// an unresolvable direction: the descent search runs first and unchanged, and
/// a claim the criterion confirms leaves by a different door.
#[test]
fn a_genuine_saddle_still_descends_and_never_reaches_the_ladder_2748() {
    let (mut objective, theta, gradient, hessian, _direction) = planted(Planted {
        criterion_vv: -1.6e3,
        analytic_vv: -1.6e3,
        fourth_derivative: 1.0,
        gradient_scale: 1.0e-9,
    });
    let counter = std::sync::Arc::clone(&objective.evaluations);
    let baseline = objective.value(&theta);
    let bounds = wide_bounds();

    let verdict = adjudicate_negative_curvature(
        &mut objective,
        &theta,
        &gradient,
        &hessian,
        &[],
        None,
        baseline,
        1.0e-6,
        &bounds,
        "planted #2748 real-saddle control",
    );
    let SaddleAdjudication::Descended(point) = verdict else {
        panic!("a -1.6e3 curvature at a stationary point descends within one e-fold: {verdict:?}");
    };
    assert!(
        objective.value(&point) < baseline,
        "the minted reseed must be a strictly lower point"
    );
    // The escape found its descent on the first rung of the first sign, then
    // the restore. No ladder extension ran.
    assert!(
        *counter.lock().expect("counter") <= 3,
        "a confirmed saddle must not pay for a measurement it does not need; \
         {} evaluations",
        *counter.lock().expect("counter")
    );
}

/// The extension's END is derived, and this pins the derivation rather than the
/// count it produces on one fixture.
///
/// `α_end = sqrt(roundoff_floor/|λ_min|)` is where the claim's own predicted
/// numerator `|λ_min|·α²` reaches the objective's arithmetic floor. Halving
/// from the falsifiability ladder's smallest step to two rungs past `α_end`
/// therefore spans exactly the range in which the second difference carries
/// signal, plus the plateau `ε_f` is read from. A ladder that stopped at
/// `α_end` would have no plateau and no `ε_f`; one that stopped before it would
/// be fitting `M₄` to a range the quartic term dominates.
#[test]
fn the_ladder_spans_signal_and_plateau_on_both_sides_of_the_derived_end_2748() {
    let lambda_min = -6.404082e-6_f64;
    let baseline_cost = 2.224446222e3_f64;
    let roundoff_floor = baseline_cost.abs().max(1.0) * (16.0 * f64::EPSILON);
    let alpha_end = (roundoff_floor / lambda_min.abs()).sqrt();

    // At `α_end` the claim predicts exactly the objective's own roundoff.
    let predicted_at_end = lambda_min.abs() * alpha_end * alpha_end;
    assert!(
        (predicted_at_end - roundoff_floor).abs() <= 1.0e-12 * roundoff_floor,
        "alpha_end must be where the claim's predicted numerator equals the roundoff floor: \
         {predicted_at_end:.6e} vs {roundoff_floor:.6e}"
    );
    // It is strictly inside the falsifiability ladder's range, so the extension
    // is a real extension on this fixture and not a no-op.
    assert!(
        alpha_end < 1.0,
        "on the #2748 fixture the derived end must be inside the box: {alpha_end:.6e}"
    );
    // And the rungs are the halvings between: enough for the two-parameter fit
    // plus residual degrees of freedom.
    let rungs = (1.0_f64 / alpha_end).log2().ceil() as usize + 2;
    assert!(
        rungs >= 3,
        "the derived ladder must carry at least the three rungs the fit needs: {rungs}"
    );
}
