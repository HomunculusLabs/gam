//! End-to-end CAPABILITY test: gam must FIT interval-censored survival data
//! THROUGH THE USER-FACING FORMULA FIT PATH, SAVE the fitted model, and RECOVER
//! the known generating marginal survival curve from that saved model, matching
//! or beating `lifelines` on the same brackets.
//!
//! Interval censoring is a first-class survival capability in every mature tool
//! (`lifelines.*Fitter.fit_interval_censoring`, flexsurv `Surv(L, R, type="interval2")`,
//! R `icenReg`): the exact event time `T` is never observed, only a bracket
//! `T ∈ (L, R]` produced by a discrete inspection schedule. The correct row
//! contribution is the interval mass
//!     ℓ_i = log[ S(L_i | x_i) − S(R_i | x_i) ],
//! not a point-density (exact event) nor a single-sided survival
//! (right-censoring). gam's latent-survival family implements exactly this
//! contribution, and this test exercises it END-TO-END: a user writes the
//! formula `SurvInterval(L, R, event) ~ 1`, gam parses the dedicated
//! interval-censored response, materializes the time basis at BOTH boundaries
//! `L` and `R`, fits through `fit_latent_survival_terms`, and saves the model
//! through `fit_formula_to_payload`, the one save path the CLI and Python share.
//!
//! DATA-GENERATING PROCESS (truth is known exactly).
//!   Latent log-frailty  U ~ Normal(μ*, σ*).
//!   Conditional cumulative hazard  Λ(t | U) = B(t) · exp(U),  with a KNOWN
//!   baseline cumulative hazard B(t) = t (unit Weibull/exponential clock), so
//!   conditional survival  S(t | U) = exp(−B(t) e^U).
//!   Marginal survival     S(t) = E_U[exp(−B(t) e^U)] = K_{0, B(t)}(μ*, σ*),
//!   the kernel's order-0 mass integral. Each subject's true event time T is
//!   drawn from this marginal, then only the bracketing pair (L, R] from a fixed
//!   inspection grid is retained — the exact T is discarded. This is exactly how
//!   interval-censored data arises in practice.
//!
//! OBJECTIVE METRIC (truth recovery of the identified estimand).
//!   Interval brackets from an intercept-only model identify the MARGINAL
//!   survival curve at the inspection times and nothing finer: the likelihood is
//!   a function of S at the bracket endpoints alone. The frailty spread σ is not
//!   separately identified against gam's flexible monotone (I-spline) baseline.
//!   For any σ, S_σ(B) = E_U[exp(−B e^U)] is continuous and strictly decreasing
//!   in B, so a monotone B_σ = S_σ⁻¹(S) reproduces the same marginal. The
//!   spline's finite span only weakly separates the candidates. The primary claim
//!   is therefore that gam's OWN fitted marginal curve, read from the saved model
//!   by the production latent-window predictor, recovers the true marginal S(t):
//!   relative L2 error on the inspection grid below a finite-sample tolerance.
//!   The fitted spread σ̂ is printed as context, not asserted. At 113cf2d3a the
//!   earlier σ bar read σ̂ = 0.1688 against σ* = 0.6000 (MSI job 604258).
//!
//! BASELINE TO MATCH-OR-BEAT:
//!   `lifelines.WeibullFitter().fit_interval_censoring(L, R)` is fit on the
//!   IDENTICAL (L, R] brackets, and its fitted marginal survival curve is scored
//!   against the SAME truth on the same grid. gam's fitted curve must be no worse
//!   by more than 10%. Both engines are scored on their own fitted curves.
//!
//! There is no skip path: a missing `lifelines` is a real failure.
use csv::StringRecord;
use gam::families::survival::lognormal_kernel::{
    FrailtyScale, FrailtySpec, HazardLoading, LatentSurvivalRow, LatentSurvivalRowJet,
};
use gam::quadrature::QuadratureContext;
use gam::test_support::reference::{Column, run_python};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::predict::{
    SurvivalPredictEstimand, SurvivalPredictRequest, predict_latent_window_survival,
};
use ndarray::{Array1, Array2};

/// Marginal survival S(t) = K_{0, t}(μ, σ) through the gam kernel, used only to
/// build the synthetic truth.
fn marginal_survival(ctx: &QuadratureContext, b_t: f64, mu: f64, sigma: f64) -> f64 {
    let row = LatentSurvivalRow::right_censored(0.0, b_t, 0.0, 0.0);
    let jet = LatentSurvivalRowJet::evaluate(ctx, &row, mu, sigma)
        .expect("right-censored survival mass must evaluate");
    jet.log_lik.exp()
}

#[test]
fn gam_recovers_interval_censored_latent_truth_match_or_beat_lifelines() {
    init_parallelism();
    let ctx = QuadratureContext::new();

    // ---- truth ------------------------------------------------------------
    let mu_true = -0.4_f64;
    let sigma_true = 0.6_f64;
    let n = 240usize;

    // Fixed inspection grid (discrete clinic visits). B(t) = t (unit clock).
    let grid: Vec<f64> = (0..=12).map(|k| 0.5 * k as f64).collect();

    // Draw true event times from the MARGINAL S(t) via inverse-CDF on a fine
    // grid (deterministic low-discrepancy quantiles -> reproducible, no RNG
    // dependence). Then bracket each by the inspection grid to make it interval
    // data; the exact time is discarded.
    let mut fine_t = Vec::new();
    let mut fine_s = Vec::new();
    {
        let mut t = 1e-4;
        while t <= 7.0 {
            fine_t.push(t);
            fine_s.push(marginal_survival(&ctx, t, mu_true, sigma_true));
            t += 0.01;
        }
    }
    let invert_survival = |u: f64| -> f64 {
        for w in fine_s.windows(2).enumerate() {
            let (i, pair) = w;
            if pair[0] >= u && u > pair[1] {
                let frac = (pair[0] - u) / (pair[0] - pair[1]).max(1e-12);
                return fine_t[i] + frac * (fine_t[i + 1] - fine_t[i]);
            }
        }
        *fine_t.last().unwrap()
    };

    let mut left_times = Vec::with_capacity(n);
    let mut right_times = Vec::with_capacity(n);
    for i in 0..n {
        // van der Corput low-discrepancy quantile in (0,1)
        let mut q = 0.0_f64;
        let mut base = 0.5_f64;
        let mut k = i + 1;
        while k > 0 {
            if k & 1 == 1 {
                q += base;
            }
            base *= 0.5;
            k >>= 1;
        }
        let u = 1.0 - (q * 0.998 + 0.001); // survival prob in (0,1)
        let t_true = invert_survival(u);

        // bracket by the inspection grid
        let mut l = 0.0_f64;
        let mut r = f64::INFINITY;
        for win in grid.windows(2) {
            if t_true > win[0] && t_true <= win[1] {
                l = win[0];
                r = win[1];
                break;
            }
        }
        if !r.is_finite() {
            // beyond last visit: bracket to the fine horizon (still an interval).
            l = *grid.last().unwrap();
            r = 7.0;
        }
        // The interval kernel requires a strictly positive left boundary (B(L)
        // is a cumulative hazard; B(0) = 0 collapses S(L) = 1, which is the
        // correct "alive at study start" contribution — gam clips the time floor
        // internally, but we keep L strictly inside the support for a crisp
        // bracket). Left-clip exactly like the lifelines reference body.
        left_times.push(l.max(1e-6));
        right_times.push(r);
    }

    // ---- gam interval-censored fit THROUGH THE FORMULA FIT PATH, saved -----
    // A user writes `SurvInterval(L, R, event) ~ 1`. `event = 1` marks every row
    // as bracketed (the exact time lies in (L, R]); the latent family integrates
    // the lognormal frailty over a flexible monotone baseline. The saved model
    // persists the fitted spread as a fixed hazard multiplier.
    let headers = vec!["L".to_string(), "R".to_string(), "event".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                left_times[i].to_string(),
                right_times[i].to_string(),
                "1".to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode interval-censored survival data");

    let cfg = FitConfig {
        survival_likelihood: Some("latent".to_string()),
        baseline_target: "weibull".to_string(),
        time_basis: "ispline".to_string(),
        frailty: FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Learned { initial_sigma: 0.5 },
            loading: HazardLoading::Full,
        },
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("SurvInterval(L, R, event) ~ 1".to_string(), &data, &cfg)
        .expect("gam interval-censored latent fit must route through the formula fit path and save");
    let sigma_hat = match payload.family_state.frailty() {
        Some(FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Fixed { sigma },
            ..
        }) => *sigma,
        _ => panic!("a saved latent survival fit must persist its fitted hazard-multiplier spread"),
    };
    let model = FittedModel::from_payload(payload);

    // ---- gam's own fitted marginal survival curve, from the saved model -----
    // A SurvInterval fit saves L as its exit column and no entry column, so the
    // saved latent law's time grid over one row reads P(T > t) = S(t) at every
    // evaluation time (8bbc815e8). R and event are carried only because the saved
    // schema names them; the grid reads neither.
    let eval_t: Vec<f64> = (1..=12).map(|k| 0.5 * k as f64).collect();
    let columns = data.column_map();
    let mut frame = Array2::<f64>::zeros((1, data.headers.len()));
    frame[[0, columns["L"]]] = eval_t[0];
    frame[[0, columns["R"]]] = eval_t[0] + 0.5;
    frame[[0, columns["event"]]] = 1.0;
    let offset = Array1::<f64>::zeros(1);
    let gam_curve: Vec<f64> = predict_latent_window_survival(SurvivalPredictRequest {
        model: &model,
        data: frame.view(),
        col_map: &columns,
        training_headers: Some(&data.headers),
        primary_offset: &offset,
        noise_offset: &offset,
        time_grid: Some(eval_t.as_slice()),
        with_uncertainty: false,
        estimand: SurvivalPredictEstimand::Plugin,
    })
    .expect("the saved latent survival model predicts its marginal survival on the grid")
    .grid_survival
    .expect("a requested time grid returns grid survival")
    .row(0)
    .to_vec();

    // ---- lifelines baseline on the IDENTICAL brackets --------------------
    // lifelines fits the same interval brackets; we read its fitted marginal
    // survival curve on the same evaluation grid.
    let ref_res = run_python(
        &[
            Column::new("L", &left_times),
            Column::new("R", &right_times),
        ],
        r#"
import numpy as np
from lifelines import WeibullFitter
L = np.asarray(df["L"], dtype=float)
R = np.asarray(df["R"], dtype=float)
# Avoid exact-zero left bound for the parametric interval fit.
L = np.clip(L, 1e-6, None)
wf = WeibullFitter()
wf.fit_interval_censoring(L, R)
# Recovered marginal survival on the evaluation grid.
eval_t = np.array([0.5*k for k in range(1,13)], dtype=float)
S = wf.survival_function_at_times(eval_t).values.ravel()
emit("S_ref", list(S))
"#,
    );
    let s_ref = ref_res.vector("S_ref");
    assert_eq!(
        s_ref.len(),
        eval_t.len(),
        "lifelines survival curve length mismatch"
    );

    // ---- both fitted curves against the true marginal ---------------------
    let mut gam_curve_sse = 0.0;
    let mut ref_curve_sse = 0.0;
    let mut truth_norm = 0.0;
    for (idx, &t) in eval_t.iter().enumerate() {
        let s_true = marginal_survival(&ctx, t, mu_true, sigma_true);
        gam_curve_sse += (gam_curve[idx] - s_true).powi(2);
        ref_curve_sse += (s_ref[idx] - s_true).powi(2);
        truth_norm += s_true.powi(2);
    }
    let gam_curve_rell2 = (gam_curve_sse / truth_norm).sqrt();
    let ref_curve_rell2 = (ref_curve_sse / truth_norm).sqrt();

    eprintln!(
        "interval-censored recovery (formula fit path, saved model): fitted marginal curve \
         relL2 gam={:.4} lifelines={:.4} | context: gam sigma_hat={:.4} truth sigma*={:.4} \
         (sigma is not separately identified from intercept-only brackets) | gam S={:?}",
        gam_curve_rell2,
        ref_curve_rell2,
        sigma_hat,
        sigma_true,
        gam_curve
    );

    // ---- truth-recovery assertion on the identified estimand -------------
    assert!(
        gam_curve_rell2 < 0.06,
        "gam's fitted marginal survival curve is too far from the truth: relL2={:.4}",
        gam_curve_rell2
    );

    // ---- match-or-beat lifelines -----------------------------------------
    assert!(
        gam_curve_rell2 <= ref_curve_rell2 * 1.10 + 1e-3,
        "gam interval-censored survival-curve recovery ({:.4}) is worse than lifelines ({:.4}) by >10%",
        gam_curve_rell2,
        ref_curve_rell2
    );
}
