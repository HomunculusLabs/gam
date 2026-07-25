//! The λ→∞ limit of the REML criterion, built analytically (#2348 Inc 5).
//!
//! [`RemlState::rail_face_limit`] answers one question exactly: *what is this
//! fit when the smoothing parameters on a rail face are literally infinite?*
//! It does not evaluate the criterion at a huge finite λ — at ρ ≈ 30 the
//! logdet pair `½log|H| − ½log|S_λ|₊` is cancelling `r_F·log λ ≈ 400` against
//! itself and its ρ-derivative is instrument noise. Instead it forms the limit
//! objects directly:
//!
//! * the **limit fit** — the model restricted to `N = ⋂_{j∈F} null(S_j)`,
//!   solved with the surviving penalties. This is the λ=∞ model, and its
//!   criterion value is `V_∞` (the divergent `log|QᵀS_FQ|` cancels exactly
//!   between the two logdets, so the limit is finite);
//! * the **limit score** `g_c = XᵀW(y − Xβ̂_∞) − S_Rβ̂_∞`, the Lagrange force
//!   the constraint carries, computed from `O(1)` quantities rather than read
//!   off a `λ⁻¹`-scale coefficient tail;
//! * the **first-order form** `C = Schur_Z(XᵀWX + S_R) − Schur_Z(S_R) −
//!   g_Qg_Qᵀ/φ̂` on the released subspace, whose positive definiteness is the
//!   whole face certificate (see [`crate::rho_optimizer::rail_face`] for the
//!   derivation and what `C ≻ 0` proves).
//!
//! The Schur complements ARE the analytic λ→∞ limits of the two logdet terms'
//! first-order content: `tr((QᵀS_FQ)⁻¹·Schur_Z(K))` is what `½log|H|` leaves
//! behind after its divergence is removed, and `tr((QᵀS_FQ)⁻¹·Schur_Z(S_R))`
//! is what `−½log|S_λ|₊` leaves behind. Nothing here is differenced, probed,
//! or extrapolated.
//!
//! # Scope
//!
//! Gaussian-identity, dense design, flat ρ-prior, no coefficient constraints.
//! For every other configuration the working weights depend on `β̂`, so the
//! logdet terms acquire a third-derivative contribution at the same order and
//! this closed form is no longer the whole first-order story; the method
//! returns `None` and the measured-tail certificate remains the authority.

use super::*;
use crate::rho_optimizer::rail_face::{RailFaceLimit, gaussian_rail_face_limit};

impl RemlState<'_> {
    /// Build the analytic λ→∞ limit data for the rail face `face` (ρ-block
    /// indices), at the smoothing parameters `rho`.
    ///
    /// This is the gate; the algebra is [`gaussian_rail_face_limit`], which any
    /// caller holding the same parts can use directly. `Ok(None)` — never an
    /// error — whenever this fit's configuration is outside the exact closed
    /// form, or the limit itself is not available. A refusal is not a fit
    /// failure; it just means this certificate has nothing to say and the
    /// caller keeps whatever authority it already had.
    pub(crate) fn rail_face_limit(
        &self,
        rho: &Array1<f64>,
        face: &[usize],
    ) -> Result<Option<RailFaceLimit>, EstimationError> {
        // The closed form is exact only where the working weights do not move
        // with β̂ and the criterion is the plain profiled-Gaussian REML.
        if !reml_is_gaussian_identity(&self.config.likelihood)
            || self.linear_constraints.is_some()
            || self.coefficient_lower_bounds.is_some()
            || self.runtime_mixture_link_state.is_some()
            || self.runtime_sas_link_state.is_some()
            || self.penalty_shrinkage_floor.is_some()
            || !matches!(self.rho_prior, RhoPrior::Flat)
            || !matches!(self.x, DesignMatrix::Dense(_))
        {
            return Ok(None);
        }
        let design = self
            .x
            .try_to_dense_by_chunks("rail face limit")
            .map_err(EstimationError::RemlOptimizationFailed)?;
        if design.ncols() != self.p {
            return Ok(None);
        }
        // The builder wants the response already net of any offset.
        let mut response = self.y.to_owned();
        if self.offset.len() == response.len() {
            response -= &self.offset;
        }
        Ok(gaussian_rail_face_limit(
            design.view(),
            response.view(),
            self.weights,
            self.canonical_penalties.as_slice(),
            rho,
            face,
        ))
    }
}

#[cfg(test)]
mod rail_face_limit_tests {
    use super::*;
    use crate::rho_optimizer::OuterEvalOrder;
    use crate::rho_optimizer::rail_face::{RailFaceVerdict, certify_rail_face};

    /// A cubic design `[1, t, t², t³]` whose truth is linear. Penalty 0 hits the
    /// quadratic+cubic columns (the block that can rail to λ=∞), penalty 1 hits
    /// the linear column (the survivor that stays finite).
    ///
    /// `curvature` sets how much genuine quadratic signal the data carries —
    /// the whole point of the face certificate is that it separates "the
    /// released directions are not worth their Occam cost" from "they are".
    /// The residual is a deterministic Nyquist alternation, which no cubic can
    /// chase, so the dispersion is real and `curvature` alone controls the
    /// score for releasing the face.
    /// Self-adjoint eigendecomposition of the small design gram, used only to
    /// project the fixture's residual out of the design's column space.
    fn design_gram_eigh(gram: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;
        let n = gram.nrows();
        let mut sym = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                sym[[i, j]] = 0.5 * (gram[[i, j]] + gram[[j, i]]);
            }
        }
        sym.eigh(Side::Lower)
            .expect("the cubic design gram is well posed")
    }

    fn cubic_fixture(curvature: f64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        cubic_fixture_sized(curvature, 96)
    }

    fn cubic_fixture_sized(
        curvature: f64,
        n: usize,
    ) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut x = Array2::<f64>::zeros((n, 4));
        let mut raw_noise = Array1::<f64>::zeros(n);
        let mut signal = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            x[[i, 0]] = 1.0;
            x[[i, 1]] = t;
            x[[i, 2]] = t * t;
            x[[i, 3]] = t * t * t;
            raw_noise[i] = if i % 2 == 0 { 0.35 } else { -0.35 };
            // NO true slope: a penalized survivor coordinate shrinks a real
            // slope, and that shrinkage bias is itself signal the released
            // directions can absorb — a genuine reason to come off the face,
            // and not the one this fixture is about.
            signal[i] = 0.8 + curvature * t * t;
        }
        // Project the deterministic residual out of the design's column space,
        // so the limit fit's score in the released directions is EXACTLY the
        // curvature the fixture asked for — the quantity the certificate weighs
        // against those directions' Occam cost.
        let gram = x.t().dot(&x);
        let rhs = x.t().dot(&raw_noise);
        let (values, vectors) = design_gram_eigh(&gram);
        let rotated = vectors.t().dot(&rhs);
        let mut scaled = Array1::<f64>::zeros(values.len());
        for i in 0..values.len() {
            scaled[i] = rotated[i] / values[i];
        }
        let residual = &raw_noise - &x.dot(&vectors.dot(&scaled));
        let y = &signal + &residual;
        (y, Array1::<f64>::ones(n), x)
    }

    /// A truth with no material curvature: releasing the bend block buys
    /// almost no fit against a real Occam cost, so `C ≻ 0` and λ=∞ is the
    /// optimum.
    const NULL_CURVATURE: f64 = 0.004;
    /// A truth the released directions genuinely explain: the score for
    /// releasing them beats their cost, `C` is indefinite, and the face must be
    /// refused.
    const SIGNAL_CURVATURE: f64 = 6.0;

    fn cubic_penalties() -> Vec<gam_terms::construction::CanonicalPenalty> {
        let p = 4usize;
        let mut bend = Array2::<f64>::zeros((p, p));
        bend[[2, 2]] = 1.0;
        bend[[3, 3]] = 2.0;
        let mut slope = Array2::<f64>::zeros((p, p));
        slope[[1, 1]] = 1.0;
        gam_terms::construction::canonicalize_penalty_specs(
            &[
                crate::estimate::PenaltySpec::Dense(bend),
                crate::estimate::PenaltySpec::Dense(slope),
            ],
            &[2, 3],
            p,
            "rail_face_limit_fixture",
        )
        .map(|(canonical, _)| canonical)
        .expect("canonicalize the rail-face fixture penalties")
    }

    fn gaussian_config() -> RemlConfig {
        RemlConfig::external(
            GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            )),
            1e-10,
            false,
        )
    }

    /// THE VALIDATION. The analytic pencil constant is a prediction about the
    /// PRODUCTION criterion: on the tail, `∂V/∂ρ_j = −c_j·e^{−ρ_j}` with
    /// `c_j = ½tr((QᵀS_jQ)⁻¹QᵀCQ)` — a number built entirely from limit
    /// objects, with no probe, no finite difference, and no evaluation at a
    /// large λ anywhere in it.
    ///
    /// Measure the real `∂V/∂ρ_0` at a MODERATE `ρ_0 = 12` (deep enough that
    /// the `O(e^{−2ρ})` correction is ~1e-5 relative, shallow enough that the
    /// logdet pair has not started cancelling) and require the analytic
    /// constant to reproduce it. If the closed form had the wrong dispersion
    /// convention, a missing Schur term, or a sign error, this test would miss
    /// by orders of magnitude rather than by a rounding.
    /// One `(n, ρ_0)` row: the analytic constant against the production one.
    fn measured_against_analytic(n: usize, rho_0: f64) -> (f64, f64) {
        let (y, weights, x) = cubic_fixture_sized(NULL_CURVATURE, n);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            cubic_penalties(),
            4,
            &config,
            Some(vec![2, 3]),
            None,
            None,
        )
        .expect("build the rail-face fixture state");

        let rho = Array1::from(vec![rho_0, 1.0]);
        let limit = state
            .rail_face_limit(&rho, &[0])
            .expect("the analytic face limit must not error")
            .expect("a Gaussian dense fixture is inside the closed form");
        let analytic = match certify_rail_face(&limit) {
            RailFaceVerdict::Certified(proof) => proof.tail_constants[0],
            RailFaceVerdict::Refused { reason } => {
                panic!("the fixture face should certify at n={n}, rho_0={rho_0}: {reason}")
            }
        };
        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production REML gradient must evaluate");
        (analytic, -(rho_0.exp()) * eval.gradient[0])
    }

    #[test]
    fn analytic_face_constant_reproduces_the_production_rho_gradient() {
        let (analytic_shallow, measured_shallow) = measured_against_analytic(96, 10.0);
        let (analytic_deep, measured_deep) = measured_against_analytic(96, 14.0);
        let (analytic_wide, measured_wide) = measured_against_analytic(384, 10.0);
        for (label, measured) in [
            ("n=96 rho_0=10", measured_shallow),
            ("n=96 rho_0=14", measured_deep),
            ("n=384 rho_0=10", measured_wide),
        ] {
            assert!(
                measured > 0.0,
                "{label}: the fixture must be on an upper tail, measured pencil {measured:.6e}"
            );
        }

        // Where the production gradient is still a trustworthy instrument, the
        // analytic constant reproduces it.
        let shallow = (analytic_shallow - measured_shallow).abs() / measured_shallow;
        let wide = (analytic_wide - measured_wide).abs() / measured_wide;
        assert!(
            shallow < 5.0e-3 && wide < 5.0e-3,
            "the analytic face constant must reproduce the production tail law: \
             n=96 analytic={analytic_shallow:.9e} measured={measured_shallow:.9e} \
             rel={shallow:.4e}; n=384 analytic={analytic_wide:.9e} \
             measured={measured_wide:.9e} rel={wide:.4e}"
        );

        // The analytic constant is a property of the LIMIT, so it does not
        // depend on where the coordinate happens to sit — bitwise, not to a
        // tolerance: nothing in it is evaluated at ρ.
        assert_eq!(
            analytic_shallow, analytic_deep,
            "the analytic pencil constant must not depend on rho at all"
        );

        // The measured one does depend on ρ, and that is the defect this
        // certificate removes: the pencil is `e^rho` times a gradient whose
        // assembly error is fixed, so the instrument's error grows by the same
        // factor while the law does not move. Asserting the divergence keeps
        // the fixture honest — if the production gradient ever became exact at
        // depth, this test would say so instead of silently passing.
        let deep = (analytic_deep - measured_deep).abs() / measured_deep;
        assert!(
            deep > 4.0 * shallow,
            "the measured pencil should degrade with depth while the analytic \
             one does not: rel at rho_0=10 is {shallow:.4e}, at rho_0=14 is {deep:.4e} \
             (measured {measured_shallow:.9e} then {measured_deep:.9e} against a \
             constant analytic {analytic_shallow:.9e})"
        );
    }

    /// The value gap is the criterion improvement still available by running
    /// the face to λ=∞. Its prediction is testable directly against production
    /// VALUES — no derivative, no cancellation: `V(ρ_0) − V(ρ_0 + Δ)` must
    /// equal `c·(e^{−ρ_0} − e^{−ρ_0−Δ})`.
    #[test]
    fn analytic_value_gap_predicts_the_measured_criterion_drop() {
        let (y, weights, x) = cubic_fixture(NULL_CURVATURE);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            cubic_penalties(),
            4,
            &config,
            Some(vec![2, 3]),
            None,
            None,
        )
        .expect("build the rail-face fixture state");

        let near = Array1::from(vec![10.0, 1.0]);
        let far = Array1::from(vec![14.0, 1.0]);
        let limit = state
            .rail_face_limit(&near, &[0])
            .expect("the analytic face limit must not error")
            .expect("a Gaussian dense fixture is inside the closed form");
        let near_gap = match certify_rail_face(&limit) {
            RailFaceVerdict::Certified(proof) => proof.value_gap,
            RailFaceVerdict::Refused { reason } => panic!("face should certify: {reason}"),
        };
        let far_limit = state
            .rail_face_limit(&far, &[0])
            .expect("the analytic face limit must not error")
            .expect("a Gaussian dense fixture is inside the closed form");
        let far_gap = match certify_rail_face(&far_limit) {
            RailFaceVerdict::Certified(proof) => proof.value_gap,
            RailFaceVerdict::Refused { reason } => panic!("face should certify: {reason}"),
        };

        let near_value = state.compute_cost(&near).expect("criterion at rho_0=10");
        let far_value = state.compute_cost(&far).expect("criterion at rho_0=14");
        let measured_drop = near_value - far_value;
        let predicted_drop = near_gap - far_gap;

        assert!(
            predicted_drop > 0.0,
            "running the face outward must lower the criterion; predicted {predicted_drop:.6e}"
        );
        let relative = (predicted_drop - measured_drop).abs() / measured_drop.abs();
        assert!(
            relative < 1.0e-2,
            "the analytic value gap must predict the measured criterion drop: \
             predicted={predicted_drop:.9e} measured={measured_drop:.9e} rel={relative:.3e}"
        );
    }

    /// THE REFUSAL IS ALSO A MEASUREMENT. Give the same design a truth the
    /// released directions genuinely explain. The face form goes indefinite and
    /// the certificate refuses — and it is RIGHT to: the production criterion
    /// at the same point is still *rising* in `ρ_0`, so the descent runs INWARD
    /// and λ=∞ is not the optimum at all. A certificate that minted here would
    /// be shipping a fit the optimizer was still trying to leave.
    #[test]
    fn face_refuses_when_the_released_directions_earn_their_cost() {
        let (y, weights, x) = cubic_fixture(SIGNAL_CURVATURE);
        let offset = Array1::<f64>::zeros(y.len());
        let config = gaussian_config();
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            cubic_penalties(),
            4,
            &config,
            Some(vec![2, 3]),
            None,
            None,
        )
        .expect("build the signal-bearing fixture state");

        let rho = Array1::from(vec![12.0, 1.0]);
        let limit = state
            .rail_face_limit(&rho, &[0])
            .expect("the analytic face limit must not error")
            .expect("a Gaussian dense fixture is inside the closed form");
        match certify_rail_face(&limit) {
            RailFaceVerdict::Refused { reason } => assert!(
                reason.contains("not positive definite"),
                "the refusal must name the failed curvature gate: {reason}"
            ),
            RailFaceVerdict::Certified(proof) => panic!(
                "a face the data wants released must NOT certify; got λ_min(C)={:.3e}",
                proof.min_curvature
            ),
        }

        let eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the production REML gradient must evaluate");
        assert!(
            eval.gradient[0] > 0.0,
            "the refusal must agree with the criterion: dV/drho_0={:.6e} should be POSITIVE \
             (descent runs inward, away from the rail)",
            eval.gradient[0]
        );
    }

    /// The gate is a gate: a non-Gaussian configuration is outside the closed
    /// form and must decline rather than return a wrong limit.
    #[test]
    fn non_gaussian_configuration_declines_the_closed_form() {
        let (y01, weights, x) = cubic_fixture(NULL_CURVATURE);
        let y: Array1<f64> = y01.mapv(|v| if v > 1.4 { 1.0 } else { 0.0 });
        let offset = Array1::<f64>::zeros(y.len());
        let config = RemlConfig::external(
            GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            )),
            1e-10,
            false,
        );
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            weights.view(),
            offset.view(),
            cubic_penalties(),
            4,
            &config,
            Some(vec![2, 3]),
            None,
            None,
        )
        .expect("build the binomial fixture state");
        let rho = Array1::from(vec![12.0, 1.0]);
        assert!(
            state
                .rail_face_limit(&rho, &[0])
                .expect("the gate must decline, not error")
                .is_none(),
            "a binomial fit is outside the Gaussian closed form and must decline"
        );
    }
}
