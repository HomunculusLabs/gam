//! gam#2600, from the estimator's side: a conditional-transformation-normal fit
//! must RECOVER the transformation it is named for.
//!
//! The defect this issue was closed on is that the CTN row density carried a
//! renormalizer by the standard-normal mass between two FITTED support
//! endpoints, `φ(h)·h' / [Φ(h_hi) − Φ(h_lo)]`. That divisor destroyed the two
//! properties a most-likely-transformation model needs — concavity (`log Z` is
//! concave in the endpoints by Prékopa, and the endpoints are linear in `β`) and
//! coercivity (the `−½Σh²` that would punish an escaping location column is
//! divided out) — so the MLE did not exist at any smoothing strength and the
//! inner solve correctly refused to find a mode.
//!
//! Two unit pins already carry the two destroyed properties directly
//! (`ctn_penalized_objective_is_coercive_in_the_location_column_2600` and
//! `ctn_observed_information_is_positive_semidefinite_2600`). Both are
//! statements about the SHAPE of the objective at hand-built coefficient
//! vectors. This one is the other end of the same claim and shares none of its
//! machinery: it draws from a law whose true transformation is known in closed
//! form, fits, and asks whether the returned transformation is that one.
//!
//! `Y = exp(Z)`, `Z ~ N(0,1)`, so `F(y) = Φ(ln y)` and the model's own
//! definition `F(y) = Φ(h(y))` pins `h(y) = ln y` EXACTLY — there is no location
//! or scale freedom left to quotient out, because `h` is the whole model. That
//! makes the fitted score `h(y_i)` directly comparable to the latent `z_i` that
//! generated the row, with no alignment step that could hide a bias.
//!
//! The accuracy bar is not a chosen number. It is the nonparametric estimator
//! the same data already supports: the rank (plotting-position) transform
//! `ĥ(y_(r)) = Φ⁻¹(r / (n+1))`, which uses exactly the information a monotone
//! transformation model has and none of the smoothness. A penalized monotone
//! spline exists to beat that baseline; a fit whose likelihood is the wrong
//! probability model — a renormalized one, say — does not, because its mode is a
//! mode of some other law. So the assertion is `RMSE(fit) < RMSE(rank)` against a
//! baseline computed inside the test, plus the calibration statement (the PIT of
//! the fitted model is uniform at the standard 95% Kolmogorov bar) and the
//! structural one (the map is monotone).

use gam::probability::{normal_cdf, standard_normal_quantile};
use gam::test_support::synthetic::SplitMixNormalRng;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

/// Sample size. Large enough that the rank baseline is a genuinely good
/// estimator (so beating it is a statement), small enough that the fit is a
/// few-second intercept-only solve.
const N: usize = 256;

/// Root-mean-square deviation of an estimated transform from the truth.
fn rmse(estimate: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(estimate.len(), truth.len());
    let n = estimate.len() as f64;
    (estimate
        .iter()
        .zip(truth)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / n)
        .sqrt()
}

/// One-sample Kolmogorov–Smirnov distance of `values` from `U(0,1)`.
fn ks_distance_from_uniform(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite PIT values sort"));
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(index, &u)| {
            let upper = ((index + 1) as f64) / n - u;
            let lower = u - (index as f64) / n;
            upper.max(lower)
        })
        .fold(0.0_f64, f64::max)
}

#[test]
fn ctn_recovers_the_true_lognormal_transform_2600() {
    init_parallelism();

    // ---- the law whose transformation is known in closed form ---------------
    let mut rng = SplitMixNormalRng::new(0x2600_C7Du64);
    let z: Vec<f64> = (0..N).map(|_| rng.standard_normal()).collect();
    let y: Vec<f64> = z.iter().map(|value| value.exp()).collect();
    assert!(
        y.iter().all(|value| value.is_finite() && *value > 0.0),
        "the lognormal fixture must be strictly positive and finite"
    );

    // ---- fit the intercept-only conditional transformation model ------------
    let headers = vec!["y".to_string()];
    let rows: Vec<csv::StringRecord> = y
        .iter()
        .map(|value| csv::StringRecord::from(vec![format!("{value:.17e}")]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode lognormal column");
    let cfg = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ 1", &data, &cfg).expect(
        "the CTN penalized likelihood must have a mode: under the endpoint-renormalized \
         density it had none at any smoothing strength and this fit refused at seed validation",
    );
    let FitResult::TransformationNormal(fit) = result else {
        panic!("expected a TransformationNormal fit result for transformation_normal=true");
    };
    let block = fit
        .fit
        .block_states
        .first()
        .expect("transformation-normal fit must have one coefficient block");
    let fitted: Vec<f64> = block.eta.to_vec();
    assert_eq!(fitted.len(), N, "one score per observation");
    assert!(
        fitted.iter().all(|value| value.is_finite()),
        "fitted scores must be finite"
    );

    // ---- STRUCTURE: the fitted map is a transformation of y ------------------
    let mut order: Vec<usize> = (0..N).collect();
    order.sort_by(|&i, &j| y[i].partial_cmp(&y[j]).expect("finite responses sort"));
    for window in order.windows(2) {
        let (lo, hi) = (window[0], window[1]);
        assert!(
            fitted[hi] - fitted[lo] >= -1.0e-9,
            "the fitted transformation is not monotone in y: y={:.6e} -> h={:.6e}, \
             y={:.6e} -> h={:.6e}",
            y[lo],
            fitted[lo],
            y[hi],
            fitted[hi]
        );
    }

    // ---- the nonparametric baseline the same data already supports ----------
    // `ĥ(y_(r)) = Φ⁻¹(r/(n+1))`: the rank transform. It is consistent, uses the
    // same information a monotone transformation model has, and carries no
    // smoothness — which is the whole of what the penalized spline adds.
    let mut rank_transform = vec![0.0_f64; N];
    for (rank_zero_based, &row) in order.iter().enumerate() {
        let plotting_position = ((rank_zero_based + 1) as f64) / ((N + 1) as f64);
        rank_transform[row] =
            standard_normal_quantile(plotting_position).expect("plotting position is interior");
    }

    let fitted_rmse = rmse(&fitted, &z);
    let rank_rmse = rmse(&rank_transform, &z);

    // ---- CALIBRATION: the fitted model's own PIT is uniform ------------------
    // `F(y|x) = Φ(h)` is the model's CDF, so `Φ(h_i)` is its PIT. The bar is the
    // standard asymptotic 95% Kolmogorov critical value `1.358/√n` — a
    // distributional statement, not a tuned tolerance.
    let pit: Vec<f64> = fitted
        .iter()
        .map(|&h| normal_cdf(h))
        .collect();
    let ks = ks_distance_from_uniform(&pit);
    let ks_bar = 1.358 / (N as f64).sqrt();

    eprintln!(
        "#2600 CTN transform recovery: n={N} rmse(fit vs ln y)={fitted_rmse:.6e} \
         rmse(rank plug-in)={rank_rmse:.6e} ratio={:.4} KS(PIT,U)={ks:.6e} bar={ks_bar:.6e}",
        fitted_rmse / rank_rmse
    );

    assert!(
        ks <= ks_bar,
        "the fitted model's own PIT is not uniform: KS={ks:.6e} > {ks_bar:.6e} \
         (asymptotic 95% Kolmogorov bar at n={N})"
    );
    assert!(
        fitted_rmse < rank_rmse,
        "the penalized monotone transformation estimates ln y no better than the raw \
         rank plug-in it is supposed to improve on: rmse(fit)={fitted_rmse:.6e} vs \
         rmse(rank)={rank_rmse:.6e}. A likelihood that is not this model's own density \
         has its mode somewhere else, and this is where that shows."
    );
}
