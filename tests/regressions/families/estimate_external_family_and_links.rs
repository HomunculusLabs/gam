use gam::estimate::{FitOptions, fit_gamwith_heuristic_lambdas};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};

fn base_opts() -> FitOptions {
    FitOptions {
        compute_inference: true,
        max_iter: 40,
        nullspace_dims: vec![0],
        ..FitOptions::default()
    }
}

fn tiny_problem() -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Vec<BlockwisePenalty>,
) {
    let x = array![[1.0, -1.0], [1.0, -0.2], [1.0, 0.4], [1.0, 1.2]];
    let y = array![0.0, 0.0, 1.0, 1.0];
    let w = Array1::ones(4);
    let offset = Array1::zeros(4);
    let s = vec![BlockwisePenalty::new(0..2, Array2::eye(2))];
    (x, y, w, offset, s)
}

#[test]
fn heuristic_rho_seed_produces_a_finite_optimized_reml_fit() {
    gam_runtime::test_support::install_diagnostic_logger();
    let (x, y, w, offset, s) = tiny_problem();
    let opts = base_opts();
    let family = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    );
    let fit = fit_gamwith_heuristic_lambdas(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s,
        Some(&[2.5]),
        family,
        &opts,
    )
    .expect("fit should succeed");

    // This public API returns the REML optimum, not the candidate from which the
    // search started. The exact seed-ordering contract is owned by
    // `seeding::tests::uses_full_heuristicvector_as_primary_anchor`; at this
    // boundary the observable contract is that the seeded search produces a
    // valid optimized fit, while remaining free to move rho away from 2.5.
    assert_eq!(
        fit.log_lambdas.len(),
        1,
        "one penalty must produce one optimized log-lambda"
    );
    assert!(
        fit.log_lambdas[0].is_finite(),
        "seeded REML must publish a finite optimized log-lambda, got {}",
        fit.log_lambdas[0]
    );
    assert!(
        fit.deviance.is_finite(),
        "seeded REML must publish a finite deviance, got {}",
        fit.deviance
    );
}

/// #2158: the binomial `loglog` and `cauchit` links are ordinary state-less
/// probability links (full 5-jet Fisher weights via `fisher_weight_jet5`), so
/// the external-design route — the very function whose allow-list was missing
/// them — must fit them exactly like probit/cloglog. This exercises
/// `resolve_external_family` directly (unlike the formula-level regression),
/// pinning the fix at the machinery it lives in. Each must fit and report the
/// requested standard link back in `likelihood_family` (a silent fallback to
/// logit would report the wrong link here).
#[test]
fn resolve_external_family_fits_binomial_loglog_and_cauchit_2158() {
    for link in [StandardLink::LogLog, StandardLink::Cauchit] {
        let (x, y, w, offset, s) = tiny_problem();
        let opts = base_opts();
        let family = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link));
        let fit = fit_gamwith_heuristic_lambdas(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s,
            None,
            family.clone(),
            &opts,
        )
        .unwrap_or_else(|e| {
            panic!("binomial {link:?} must fit on the external route (#2158): {e}")
        });

        assert!(
            fit.deviance.is_finite(),
            "binomial {link:?}: deviance must be finite, got {}",
            fit.deviance
        );
        assert!(
            matches!(
                fit.likelihood_family.as_ref().map(|f| &f.link),
                Some(InverseLink::Standard(l)) if *l == link
            ),
            "binomial {link:?}: fit must report the requested standard link in likelihood_family, \
             got {:?} — a silent fallback would be caught here",
            fit.likelihood_family.as_ref().map(|f| &f.link)
        );
    }
}

#[test]
fn firth_accepted_for_binomial_loglog_and_cauchit_2158() {
    let (x, y, w, offset, penalties) = tiny_problem();
    let mut opts = base_opts();
    opts.firth_bias_reduction = true;
    for link in [StandardLink::LogLog, StandardLink::Cauchit] {
        let fit = fit_gamwith_heuristic_lambdas(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &penalties,
            Some(&[2.5]),
            LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link)),
            &opts,
        )
        .unwrap_or_else(|error| panic!("Firth {link:?} must fit with inference enabled: {error}"));
        assert!(fit.deviance.is_finite(), "{link:?}: non-finite deviance");
        assert!(
            fit.beta_flat().iter().all(|value| value.is_finite()),
            "{link:?}: non-finite coefficient"
        );
        let covariance = fit
            .beta_covariance()
            .expect("Firth fit must carry covariance");
        assert_eq!(covariance.dim(), (x.ncols(), x.ncols()));
        assert!(covariance.iter().all(|value| value.is_finite()));
        assert!(covariance.diag().iter().all(|value| *value > 0.0));
    }
}
