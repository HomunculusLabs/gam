//! Total derivatives of the reference-normalised likelihood.
use super::*;
use crate::scalar::Mixed;

impl EventHistoryFamily {
    fn reference_values<S: JetField>(
        &self, beta: &[S], loadings: &[S], rates: &[S],
    ) -> Result<crate::preserve::Normalisers<S>, EventHistoryError> {
        let tables = self.reference.as_ref().ok_or_else(|| EventHistoryError::InvalidInput {
            reason: "this family has no reference population".to_string(),
        })?;
        let marks = self.marks();
        let nodes = tables.grid.len();
        let offsets = self.block_offsets();
        let mut normalisers = Vec::new();
        let mut risk_mass = Vec::new();
        let mut masks = 0;
        for s in 0..tables.strata {
            let mut eta0 = Vec::with_capacity(nodes * marks);
            for n in 0..nodes {
                for d in 0..marks {
                    let row = s * nodes + n;
                    let mut value = beta[0].constant_like(tables.offsets[d][row]);
                    for (j, x) in tables.designs[d].row(row).iter().enumerate() {
                        value = value.add(&beta[offsets[d] + j].scale(*x));
                    }
                    eta0.push(value);
                }
            }
            let out = stratum_normalisers(&tables.grid, &eta0, loadings, rates,
                self.time_scale, &self.gh, &tables.kinds, self.atoms)?;
            masks = out.masks;
            normalisers.extend(out.log_normaliser);
            risk_mass.extend(out.log_risk_mass);
        }
        Ok(crate::preserve::Normalisers {
            log_normaliser: normalisers, log_risk_mass: risk_mass, masks,
        })
    }

    pub(super) fn computed_reference(&self, states: &[ParameterBlockState]) -> Result<RiskSetCentring, String> {
        self.validate_states(states)?;
        let beta: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
        let latent_offset = self.block_offsets()[self.marks()];
        let latent = Array1::from(beta[latent_offset..].to_vec());
        let rates = self.atom_rates(&latent);
        let out = self.reference_values(&beta,
            &beta[latent_offset..latent_offset + self.marks() * self.atoms], &rates)?;
        let tables = self.reference.as_ref()
            .ok_or_else(|| "reference centring requires reference tables".to_string())?;
        let (_, mask_of_mark) = crate::preserve::killing_masks(&tables.kinds);
        Ok(RiskSetCentring { grid: tables.grid.clone(), profiles: tables.profiles.clone(),
            coefficients: beta, node_stratum: tables.node_stratum.clone(),
            log_normaliser: out.log_normaliser,
            log_risk_mass: out.log_risk_mass, masks: out.masks, mask_of_mark })
    }

    /// The reference law is evaluated with the same jet as the subject
    /// likelihood, before interpolation. Grid adaptation is differentiated too.
    pub(super) fn path_value<S: JetField + Send + Sync>(
        &self, states: &[ParameterBlockState], beta: &[S],
    ) -> Result<S, EventHistoryError> {
        let marks = self.marks();
        let offsets = self.block_offsets();
        let latent_offset = offsets[marks];
        let loadings = &beta[latent_offset..latent_offset + marks * self.atoms];
        let rates: Vec<S> = self.free_rate_slots().iter().enumerate().map(|(k, slot)| {
            match slot {
                Some(slot) => rate_from_chart(self.rate_band, &beta[latent_offset + slot]),
                None => beta[0].constant_like(self.held_rates[k]
                    .expect("a non-free atom rate has a held value")),
            }
        }).collect();
        let normalisers = if let (Some(tables), true) = (self.reference.as_ref(), self.atoms > 0) {
            let values = self.reference_values(beta, loadings, &rates)?;
            Some(tables.carry_to_nodes(&values.log_normaliser, marks, self.nodes.total_nodes)?)
        } else { None };
        let results: Result<Vec<S>, EventHistoryError> = self.nodes.subjects.par_iter().map(|subject| {
            let first = subject.first_row;
            let mut eta0 = Vec::with_capacity(subject.len() * marks);
            for row in first..first + subject.len() {
                for d in 0..marks {
                    let mut eta = beta[0].constant_like(0.0);
                    for (j, x) in self.designs[d].row(row).iter().enumerate() {
                        eta = eta.add(&beta[offsets[d] + j].scale(*x));
                    }
                    eta0.push(eta.with_value(states[d].eta[row]));
                }
            }
            let inputs = SubjectInputs {
                nodes: subject, eta0: &eta0, loadings, rates: &rates,
                time_scale: self.time_scale, gh: &self.gh, continuation_gap: 0.0,
                designs: None,
                log_normaliser: normalisers.as_ref().map(|values|
                    &values[first * marks..(first + subject.len()) * marks]),
            };
            subject_marginal(&inputs, false).map(|result| result.loglik)
        }).collect();
        Ok(pairwise_sum(&results?, &beta[0].constant_like(0.0)))
    }

    pub(super) fn computed_joint<S: Directional>(
        &self, states: &[ParameterBlockState], u: Option<&Array1<f64>>,
        v: Option<&Array1<f64>>, derivatives: bool,
    ) -> Result<(S, Vec<S>, Vec<S>), String> {
        let values: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
        let total = values.len();
        for direction in [u, v].into_iter().flatten() {
            if direction.len() != total || direction.iter().any(|x| !x.is_finite()) {
                return Err("invalid event-history derivative direction".to_string());
            }
        }
        let beta: Vec<S> = values.iter().enumerate().map(|(q, value)|
            S::seeded(*value, u.map_or(0.0, |x| x[q]), v.map_or(0.0, |x| x[q]))).collect();
        if !derivatives {
            return Ok((self.path_value(states, &beta)?, Vec::new(), Vec::new()));
        }
        let zero = beta[0].constant_like(0.0);
        let mut value = zero.clone();
        let mut gradient = vec![zero.clone(); total];
        let mut hessian = vec![zero; total * total];
        for i in 0..total {
            for j in 0..=i {
                let seeded: Vec<Mixed<S>> = beta.iter().enumerate().map(|(q, b)|
                    Mixed::seed(b.clone(), f64::from(q == i), f64::from(q == j))).collect();
                let result = self.path_value(states, &seeded)?;
                value = result.base;
                gradient[i] = result.u;
                hessian[i * total + j] = result.uv.clone();
                hessian[j * total + i] = result.uv;
            }
        }
        Ok((value, gradient, hessian))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cohort::{CovariateSegment, Event, SubjectHistory};
    use ndarray::array;

    fn single_event() -> (EventHistoryFamily, Vec<ParameterBlockState>) {
        let mut cohort = EventHistoryCohort {
            mark_names: vec!["disease".to_string()], mark_kinds: vec![MarkKind::Once],
            covariate_names: Vec::new(), covariate_levels: Vec::new(),
            covariates: Array2::zeros((1, 0)),
            subjects: vec![SubjectHistory {
                id: "one".to_string(), entry: 0.0, exit: 6.0,
                events: vec![Event { time: 6.0, mark: 0 }],
                segments: vec![CovariateSegment { start: 0.0, row: 0 }],
            }],
        };
        cohort.validate().unwrap();
        let nodes = Arc::new(expand_nodes(&cohort, 9, 3).unwrap());
        let intervals = 144;
        let times: Vec<f64> = (0..=intervals).map(|n| 6.0 * n as f64 / intervals as f64).collect();
        let grid = ReferenceGrid { gaps: times.windows(2).map(|w| w[1] - w[0]).collect(), times };
        let locations: Vec<(usize, f64)> = nodes.subjects[0].times.iter().map(|&t| grid.locate(t).unwrap()).collect();
        let tables = ReferenceTables {
            designs: vec![Arc::new(Array2::ones((grid.len(), 1)))],
            offsets: vec![Array1::zeros(grid.len())], kinds: vec![MarkKind::Once],
            profiles: Array2::zeros((1, 0)), strata: 1,
            node_stratum: vec![0; nodes.total_nodes],
            node_lower: locations.iter().map(|x| x.0).collect(),
            node_weight: locations.iter().map(|x| x.1).collect(), grid,
        };
        let states = vec![
            ParameterBlockState { beta: array![-1.2], eta: Array1::from_elem(nodes.total_nodes, -1.2) },
            ParameterBlockState { beta: array![0.9], eta: Array1::zeros(nodes.total_nodes) },
        ];
        let family = EventHistoryFamily::new(nodes.clone(), vec![Arc::new(Array2::ones((nodes.total_nodes, 1)))],
            1, 15, 1.0, vec![Some(1e-8)]).unwrap().with_reference(Some(Arc::new(tables)));
        (family, states)
    }

    fn moved(states: &[ParameterBlockState], slot: usize, change: f64) -> Vec<ParameterBlockState> {
        let mut out = states.to_vec();
        out[slot].beta[0] += change;
        if slot == 0 { out[0].eta += change; }
        out
    }

    #[test]
    fn normalised_objective_differentiates_the_reference_law() {
        let (family, states) = single_event();
        let joint = family.joint_evaluation(&states).unwrap();
        let exact = -1.2 - 6.0 * (-1.2_f64).exp();
        assert!((joint.log_likelihood - exact).abs() < 2e-4, "{} vs {exact}", joint.log_likelihood);
        assert!(joint.gradient[1].abs() < 5e-4, "unidentified frailty score: {}", joint.gradient[1]);
        for slot in 0..2 {
            let h = 1e-4;
            let plus = moved(&states, slot, h);
            let minus = moved(&states, slot, -h);
            let fd = (family.log_likelihood(&plus).unwrap() - family.log_likelihood(&minus).unwrap()) / (2.0 * h);
            assert!((fd - joint.gradient[slot]).abs() < 1e-6, "gradient {slot}: {fd} vs {}", joint.gradient[slot]);
            let gp = family.exact_gradient(&plus).unwrap();
            let gm = family.exact_gradient(&minus).unwrap();
            for j in 0..2 {
                let fd = -(gp[j] - gm[j]) / (2.0 * h);
                assert!((fd - joint.hessian[[j, slot]]).abs() < 2e-5,
                    "Hessian {j},{slot}: {fd} vs {}", joint.hessian[[j, slot]]);
            }
        }
        let reference = family.refresh_normaliser(&states).unwrap();
        let encoded = serde_json::to_string(&reference).unwrap();
        let restored: RiskSetCentring = serde_json::from_str(&encoded).unwrap();
        assert_eq!(reference.coefficients, restored.coefficients);
        assert_eq!(reference.log_normaliser, restored.log_normaliser);
        assert_eq!(reference.log_risk_mass, restored.log_risk_mass);
        assert_eq!(reference.grid.times, restored.grid.times);
        assert_eq!(reference.profiles, restored.profiles);
        assert_eq!(reference.node_stratum, restored.node_stratum);
        assert_eq!(reference.mask_of_mark, restored.mask_of_mark);
        assert!(reference.log_normaliser.last().unwrap() < &reference.log_normaliser[0]);
        assert!((reference.log_risk_mass.last().unwrap() + 6.0 * (-1.2_f64).exp()).abs() < 2e-4);
    }

    #[test]
    fn normalised_hessian_directional_derivatives_follow_the_same_objective() {
        let (family, states) = single_event();
        let direction = array![0.0, 1.0];
        let first = family.directional_hessian(&states, &direction).unwrap();
        let second = family.second_directional_hessian(&states, &direction, &direction).unwrap();
        let h = 1e-3;
        let plus = moved(&states, 1, h);
        let minus = moved(&states, 1, -h);
        let hp = family.joint_evaluation(&plus).unwrap();
        let hm = family.joint_evaluation(&minus).unwrap();
        let dp = family.directional_hessian(&plus, &direction).unwrap();
        let dm = family.directional_hessian(&minus, &direction).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((first[[i, j]] - (hp.hessian[[i, j]] - hm.hessian[[i, j]]) / (2.0 * h)).abs() < 2e-4);
                assert!((second[[i, j]] - (dp[[i, j]] - dm[[i, j]]) / (2.0 * h)).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn reference_refinement_detects_latent_error_at_fixed_parameters() {
        let (mut family, mut states) = single_event();
        family.held_rates = vec![Some(0.0)];
        family.gh = Arc::new(GaussHermite::new(9).unwrap());
        states[1].beta[0] = 2.0;
        let coarse = family.refresh_normaliser(&states).unwrap();
        family.gh = Arc::new(GaussHermite::new(33).unwrap());
        let fine = family.refresh_normaliser(&states).unwrap();
        let gap = coarse.discrepancy(&fine, 1).unwrap();
        assert!(gap > 1e-4, "unresolved latent reference integral: {gap}");
        assert_eq!(fine.discrepancy(&fine, 1).unwrap(), 0.0);
        let mut different = fine.clone();
        different.coefficients[0] += 0.1;
        assert!(coarse.discrepancy(&different, 1).is_err());
        different = fine.clone();
        different.log_normaliser[0] = f64::NAN;
        assert!(coarse.discrepancy(&different, 1).is_err());
    }

    fn recurrent_family(event_time: f64, rates: Vec<Option<f64>>, order: usize) -> (EventHistoryFamily, Vec<ParameterBlockState>) {
        let mut cohort = EventHistoryCohort {
            mark_names: vec!["event".to_string()], mark_kinds: vec![MarkKind::Recurrent],
            covariate_names: Vec::new(), covariate_levels: Vec::new(), covariates: Array2::zeros((1, 0)),
            subjects: vec![SubjectHistory { id: "one".to_string(), entry: 0.0, exit: 1.0,
                events: vec![Event { time: event_time, mark: 0 }],
                segments: vec![CovariateSegment { start: 0.0, row: 0 }] }],
        };
        cohort.validate().unwrap();
        let nodes = Arc::new(expand_nodes(&cohort, 9, 0).unwrap());
        let states = vec![
            ParameterBlockState { beta: array![0.0], eta: Array1::zeros(nodes.total_nodes) },
            ParameterBlockState { beta: array![2.0, 0.0], eta: Array1::zeros(nodes.total_nodes) },
        ];
        let family = EventHistoryFamily::new(nodes.clone(), vec![Arc::new(Array2::ones((nodes.total_nodes, 1)))],
            2, order, 1.0, rates).unwrap();
        (family, states)
    }

    #[test]
    fn static_factor_curvature_matches_the_time_invariant_integral() {
        let points = 4001;
        let mut mass = 0.0;
        let mut second = 0.0;
        for i in 0..points {
            let z = -10.0 + 20.0 * i as f64 / (points - 1) as f64;
            let r = (2.0 * z - 2.0).exp();
            let w = (-0.5 * z * z).exp() * r * (-r).exp();
            mass += w;
            second += w * (r * r - 2.0 * r);
        }
        let expected = second / mass;
        assert!(expected < -0.3);
        let mut curvatures = Vec::new();
        for event_time in [0.1, 0.5, 0.9] {
            let (family, states) = recurrent_family(event_time, vec![Some(0.0), Some(0.0)], 33);
            let beta = vec![Mixed::seed(0.0, 0.0, 0.0), Mixed::seed(2.0, 0.0, 0.0), Mixed::seed(0.0, 1.0, 1.0)];
            let curvature = family.path_value(&states, &beta).unwrap().uv;
            let h = 1e-3;
            let eval = |a: f64| family.path_value(&states, &[0.0, 2.0, a]).unwrap();
            let finite = (eval(h) + eval(-h) - 2.0 * eval(0.0)) / (h * h);
            eprintln!("event {event_time}: AD={curvature}, finite={finite}, target={expected}");
            assert!((curvature - finite).abs() < 2e-5);
            assert!((curvature - expected).abs() < 1e-4, "{curvature} vs {expected}");
            assert!(added_factor_curvature_pair(&family, &states, EventHistorySpec::new(Vec::new()).quadrature_tolerance)
                .unwrap()
                .is_some());
            curvatures.push(curvature);
        }
        assert!(curvatures.iter().all(|c| (c - curvatures[0]).abs() < 1e-12));
    }

    /// The added-factor curvature of an in-band dynamic rate is read at its
    /// order and one ladder rung up (`2·order − 1`) while that order is
    /// certifiable, up to the ladder's top certifiable rung, and is not formed
    /// above it: that is where the rank search stops with growth unresolved.
    ///
    /// This replaces `unresolved_near_static_curvature_is_rejected`, whose rates
    /// of `1e-10` sit below `ν_min`: there the OU kernel is one to double
    /// precision, so the fixture is the static model evaluated through the
    /// dynamic interpolant. Its `is_err()` passed on the positivity loss inside
    /// the curvature, not on the gap check it named. A rate below `ν_min` belongs
    /// to the static face.
    #[test]
    fn added_factor_curvature_is_checked_up_to_the_top_certifiable_rung() {
        let tolerance = EventHistorySpec::new(Vec::new()).quadrature_tolerance;
        let rates = vec![Some(0.5), Some(0.7)];
        let (base, _) = recurrent_family(0.1, rates.clone(), 9);
        let nodes = base.nodes.max_subject_nodes();
        let mut top = 9;
        while let Some(next) = positivity_raise(top, nodes, tolerance) {
            top = next;
        }
        assert!(top > 9, "a certifiable rung must remain above order 9 over {nodes} nodes");
        assert!(certifiable(top, nodes, tolerance) && !certifiable(2 * top - 1, nodes, tolerance));
        let (family, states) = recurrent_family(0.1, rates.clone(), top);
        let pair = added_factor_curvature_pair(&family, &states, tolerance)
            .unwrap()
            .expect("the top certifiable rung checks its curvature");
        assert_eq!(pair.next_order, 2 * top - 1);
        assert!(pair.coarse.iter().chain(pair.refined.iter()).all(|x| x.is_finite()));
        let (above, above_states) = recurrent_family(0.1, rates, 2 * top - 1);
        assert!(
            added_factor_curvature_pair(&above, &above_states, tolerance).unwrap().is_none(),
            "order {} is above the top certifiable rung {top}, so its curvature cannot be checked",
            2 * top - 1
        );
    }

    /// One predicate decides every rung: at 449 nodes order 11 is certifiable,
    /// order 21 is not, so the raise from 11 refuses rather than landing on a
    /// rung whose own certificate cannot be checked (job 1150580 raised to 21
    /// and then refused at order 41, Lebesgue constant 1.154e13).
    #[test]
    fn at_449_nodes_order_11_is_the_top_certifiable_rung() {
        let tolerance = EventHistorySpec::new(Vec::new()).quadrature_tolerance;
        assert!(certifiable(11, 449, tolerance));
        assert!(!certifiable(21, 449, tolerance));
        assert_eq!(positivity_raise(11, 449, tolerance), None);
    }

    /// The start shift is the one-unit bar's numerator: each term against its
    /// closed form, sign-aligned eigenvectors moving nothing, a refused
    /// proposal moving nothing, and a spread that cannot price anything
    /// refusing rather than passing.
    #[test]
    fn proposal_start_shift_prices_the_rung_in_posterior_sd() {
        let values = array![3.0, 1.0];
        let identity = array![[1.0, 0.0], [0.0, 1.0]];
        let spreads = [0.1, 0.5];
        let mode_scale = 2.0;
        let flipped = array![[-1.0, 0.0], [0.0, 1.0]];
        assert_eq!(proposal_start_shift((&values, &identity), (&values, &flipped), mode_scale, &spreads), 0.0);

        let raised = array![3.5, 1.0];
        let shift = proposal_start_shift((&values, &identity), (&raised, &identity), mode_scale, &spreads);
        assert!((shift - 0.5 * mode_scale * spreads[0]).abs() < 1e-15, "{shift}");

        let theta = 0.01_f64;
        let rotated = array![[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]];
        let shift = proposal_start_shift((&values, &identity), (&values, &rotated), mode_scale, &spreads);
        let along_second = mode_scale * theta.sin() / spreads[1];
        let along_top = mode_scale * (1.0 - theta.cos()) / spreads[0];
        assert!(along_top < along_second);
        assert!((shift - along_second).abs() < 1e-15, "{shift} vs {along_second}");

        assert_eq!(proposal_start_shift((&values, &identity), (&raised, &rotated), 0.0, &spreads), 0.0);
        assert_eq!(
            proposal_start_shift((&values, &identity), (&values, &identity), mode_scale, &[0.1, f64::NAN]),
            f64::INFINITY
        );
        assert_eq!(
            proposal_start_shift((&values, &identity), (&values, &identity), mode_scale, &[0.1]),
            f64::INFINITY
        );
    }

    /// `mode_spread` against the Gaussian it must reproduce: `1/√(a + λ)` about
    /// a maximiser at zero, and the same about a maximiser six sd out, where
    /// the reflection's location is not spread.
    #[test]
    fn mode_spread_is_the_posterior_sd_about_the_mode() {
        let profile = |slope0: f64, curvature: f64| {
            let points: Vec<f64> = (0..=400).map(|i| 0.025 * i as f64).collect();
            DirectionProfile {
                values: points.iter().map(|t| slope0 * t - 0.5 * curvature * t * t).collect(),
                slopes: points.iter().map(|t| slope0 - curvature * t).collect(),
                points,
            }
        };
        let (curvature, lambda) = (3.0_f64, 1.0_f64);
        let sd = 1.0 / (curvature + lambda).sqrt();
        let centred = profile(0.0, curvature).mode_spread(lambda);
        assert!((centred - sd).abs() < 1e-6 * sd, "{centred} vs {sd}");
        let shifted = profile(12.0, curvature).mode_spread(lambda);
        assert!((shifted - sd).abs() < 1e-4 * sd, "{shifted} vs {sd}");
    }

    #[test]
    fn mixed_static_and_dynamic_factors_have_finite_total_derivatives() {
        let (family, mut states) = recurrent_family(0.5, vec![Some(0.0), Some(0.7)], 9);
        states[1].beta = array![0.3, 0.2];
        let joint = family.joint_evaluation(&states).unwrap();
        for slot in 0..2 {
            let h = 1e-4;
            let mut plus = states.clone();
            let mut minus = states.clone();
            plus[1].beta[slot] += h;
            minus[1].beta[slot] -= h;
            let gp = family.exact_gradient(&plus).unwrap();
            let gm = family.exact_gradient(&minus).unwrap();
            for j in 0..3 {
                let fd = -(gp[j] - gm[j]) / (2.0 * h);
                assert!((fd - joint.hessian[[j, slot + 1]]).abs() < 1e-5);
            }
        }
    }
}
