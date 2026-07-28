//! Regression guard for #561, from a different angle than the R-backed
//! `quality_vs_vgam_multinomial_softmax` quality arm: this test needs no R and
//! pins the STRUCTURAL invariants the fused-λ / dead-selection bugs violate.
//!
//! The #561 family of defects all left the multinomial outer REML/LAML loop
//! unable to actually SELECT smoothing parameters — either fusing every smooth
//! term into one shared λ, or (after the #1587 joint-penalty refactor) leaving
//! the per-term λ pinned at their seed because the custom-family outer path
//! rejected every ρ-evaluation (dimension contract), reused a stale operator
//! (cache fingerprint), dropped `½log|H|` from the cost (projected logdet), or
//! folded a phantom KKT-residual correction (inner residual omitted the joint
//! penalty). In every case the fit came back essentially UNPENALIZED (EDF near
//! the coefficient count) with `λ ≡ init`, and the recovered probability surface
//! sat far past the truth.
//!
//! This test synthesizes the same known softmax surface as the quality arm
//! (cubic in x1, sigmoid in x2, per-class linear x3), fits the penalized
//! multinomial GAM, and asserts three things that hold ONLY when per-term REML
//! selection genuinely runs:
//!   1. truth recovery — RMSE(P_gam, P_true) is small;
//!   2. penalization is active — per-class EDF is well below the coefficient
//!      count (a near-unpenalized overfit is the dead-selection signature);
//!   3. per-term λ are genuinely selected and differ — a fused/dead selector
//!      returns near-equal λ (or all equal to the seed), so the within-class
//!      spread collapses.
//! It also fits from two very different seeds and asserts the selected λ do not
//! merely echo `init_lambda` (the smoking gun that selection never moved).

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
    predict_multinomial_formula_plugin,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const N: usize = 300;
const K: usize = 3;

fn true_eta(x1: f64, x2: f64, x3: f64) -> [f64; K] {
    let cubic = 2.0 * x1.powi(3) - 1.0 * x1;
    let sigmoid = 3.0 / (1.0 + (-6.0 * (x2 - 0.5)).exp()) - 1.5;
    let eta0 = 0.6 + cubic + 0.5 * sigmoid + 1.5 * x3;
    let eta1 = -0.4 - 0.5 * cubic + sigmoid - 0.8 * x3;
    [eta0, eta1, 0.0]
}

fn softmax(eta: &[f64; K]) -> [f64; K] {
    let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut ex = [0.0; K];
    let mut s = 0.0;
    for k in 0..K {
        ex[k] = (eta[k] - m).exp();
        s += ex[k];
    }
    for k in 0..K {
        ex[k] /= s;
    }
    ex
}

fn synth() -> (gam_data::EncodedDataset, Vec<[f64; K]>) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).unwrap();
    let u01 = Uniform::new(0.0_f64, 1.0_f64).unwrap();
    let ux3 = Uniform::new(-1.5_f64, 1.5_f64).unwrap();
    let udraw = Uniform::new(0.0_f64, 1.0_f64).unwrap();
    let mut rows = Vec::with_capacity(N);
    let mut truth = Vec::with_capacity(N);
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let b = u01.sample(&mut rng);
        let c = ux3.sample(&mut rng);
        let p = softmax(&true_eta(a, b, c));
        let u = udraw.sample(&mut rng);
        let mut acc = 0.0;
        let mut chosen = K - 1;
        for k in 0..K {
            acc += p[k];
            if u <= acc {
                chosen = k;
                break;
            }
        }
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            c.to_string(),
            format!("c{chosen}"),
        ]));
        truth.push(p);
    }
    let headers: Vec<String> = ["x1", "x2", "x3", "y"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        truth,
    )
}

fn rmse_vs_truth(
    model: &gam_models::multinomial::MultinomialSavedModel,
    ds: &gam_data::EncodedDataset,
    truth: &[[f64; K]],
) -> f64 {
    let probs = predict_multinomial_formula(model, ds).expect("predict");
    let col_code: Vec<usize> = model
        .class_levels
        .iter()
        .map(|l| l.trim_start_matches('c').parse::<usize>().unwrap())
        .collect();
    let mut se = 0.0;
    for k in 0..K {
        for i in 0..N {
            let d = probs[[i, k]] - truth[i][col_code[k]];
            se += d * d;
        }
    }
    (se / (N * K) as f64).sqrt()
}

#[test]
fn multinomial_outer_reml_selects_per_term_lambda_and_recovers_truth() {
    let (ds, truth) = synth();
    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 40,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&ds, "y ~ s(x1, k=6) + s(x2, k=6) + x3", &cfg)
    })
    .expect("gam multinomial fit");

    // (1) Truth recovery. The fused-λ driver measured ≥ 0.13 and the pinned-λ
    // (dead-selection) driver ≥ 0.07 on this DGP; a working per-term REML fit
    // recovers to a few percent.
    let rmse = rmse_vs_truth(&model, &ds, &truth);
    assert!(
        rmse < 0.065,
        "multinomial fit did not recover the true simplex: RMSE={rmse:.5} (>= 0.065 \
         indicates fused-λ or a stalled outer smoothing selection)"
    );

    // (2) Penalization is active. Per-class EDF must sit well below the
    // per-class coefficient count; a near-unpenalized fit (EDF ≈ p) is the
    // dead-selection signature (λ pinned at the seed).
    let p_per_class = model.p_per_class as f64;
    let edf = model
        .edf_per_class
        .as_ref()
        .expect("REML fit must report per-class EDF");
    assert_eq!(edf.len(), K - 1, "one EDF entry per active class");
    for (a, &e) in edf.iter().enumerate() {
        assert!(
            e.is_finite() && e > 0.0 && e < 0.75 * p_per_class,
            "class {a} EDF={e:.3} is not in (0, 0.75·p={:.2}): the fit is \
             near-unpenalized, so REML never selected a smoothing parameter",
            0.75 * p_per_class
        );
    }

    // (3) Per-term λ are genuinely selected and DIFFER. With #561 fixed the
    // rough cubic term and the smoother sigmoid/null-space terms take very
    // different λ; a fused or dead selector returns near-equal λ (or all == the
    // seed 1.0). Check the within-class span.
    let per_block = &model.lambdas_per_block;
    assert!(!per_block.is_empty(), "must report per-class λ block sizes");
    let n0 = per_block[0];
    assert!(
        n0 >= 2,
        "class 0 must carry ≥2 penalty components, got {n0}"
    );
    let class0 = &model.lambdas[..n0];
    let lam_max = class0.iter().cloned().fold(f64::MIN, f64::max);
    let lam_min = class0.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        lam_max / lam_min > 5.0,
        "within-class per-term λ barely differ (max={lam_max:.4} min={lam_min:.4}); \
         REML is not selecting independent per-term smoothing (fused-λ regression)"
    );

    // (4) Selection is not a passthrough of the seed. Fit from a very different
    // seed and assert the recovered λ are NOT all ≈ init (the exact
    // dead-selection fingerprint: λ ≡ init for every component).
    let model_hi = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 50.0,
        max_iter: 40,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&ds, "y ~ s(x1, k=6) + s(x2, k=6) + x3", &cfg)
    })
    .expect("gam multinomial fit (init=50)");
    let echoes_seed = model_hi.lambdas.iter().all(|&l| (l - 50.0).abs() < 1e-6);
    assert!(
        !echoes_seed,
        "every selected λ equals the init seed 50.0 — the outer smoothing search \
         never moved (dead selection): {:?}",
        model_hi.lambdas
    );
}

/// #2612: the plug-in predictor reads the SAME fitted model at its mode.
///
/// `predict_multinomial_formula` publishes `E[softmax(η)]` and
/// `predict_multinomial_formula_plugin` publishes `softmax(x'β̂)`. Two estimands,
/// one model — so the plug-in has to clear every structural bar the integrated
/// one does, and it has to recover the SAME data-generating truth. What it must
/// NOT do is agree numerically: the difference between them is the whole reason
/// the function exists, and a test that pinned them together would be satisfied
/// by a plug-in that silently returned the integrated answer.
///
/// The class-column convention is the real hazard for a new predictor. The
/// reference class carries `η = 0` and sits LAST, and getting that wrong rotates
/// every column while leaving each row a perfectly valid simplex — invisible to a
/// shape check, fatal to a log-loss. Truth recovery is what catches it: the truth
/// table is indexed by the DGP's own class codes, so a rotated predictor scores
/// nothing like the fit it came from.
#[test]
fn the_plug_in_predictor_reads_the_same_model_at_its_mode() {
    let (ds, truth) = synth();
    let cfg = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 40,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&ds, "y ~ s(x1, k=6) + s(x2, k=6) + x3", &cfg)
    })
    .expect("gam multinomial fit");

    let plugin = predict_multinomial_formula_plugin(&model, &ds).expect("plug-in predict");
    assert_eq!(
        plugin.dim(),
        (N, K),
        "plug-in prediction must be one row per observation and one column per class"
    );

    // STRUCTURE: every row is a valid simplex.
    for i in 0..N {
        let mut row_sum = 0.0;
        for k in 0..K {
            let p = plugin[[i, k]];
            assert!(
                p.is_finite() && (-0.0..=1.0 + 1e-12).contains(&p),
                "plug-in probability out of [0, 1] at row {i} class {k}: {p}"
            );
            row_sum += p;
        }
        assert!(
            (row_sum - 1.0).abs() < 1e-9,
            "plug-in probabilities for row {i} sum to {row_sum}, not 1"
        );
    }

    // COLUMN CONVENTION + QUALITY: score the plug-in against the DGP truth with
    // the same class-code mapping the integrated predictor is scored under. A
    // rotated reference class still forms a simplex but cannot recover this.
    let col_code: Vec<usize> = model
        .class_levels
        .iter()
        .map(|l| {
            l.trim_start_matches('c')
                .parse::<usize>()
                .expect("synthetic class levels are c0/c1/c2")
        })
        .collect();
    let mut squared_error = 0.0;
    for k in 0..K {
        for i in 0..N {
            let d = plugin[[i, k]] - truth[i][col_code[k]];
            squared_error += d * d;
        }
    }
    let plugin_rmse = (squared_error / (N * K) as f64).sqrt();
    assert!(
        plugin_rmse < 0.065,
        "plug-in prediction did not recover the true simplex: RMSE={plugin_rmse:.5}; \
         the same bar the integrated predictor clears on this fit"
    );

    // The two estimands are CLOSE but need not be equal, and the direction of
    // the difference is not asserted: integrating a softmax over posterior width
    // moves probability toward the centre of the simplex for a confident row and
    // can move it the other way near the middle, so a per-row inequality is not a
    // theorem. What IS required is that both read the same model — a plug-in that
    // had lost the fit entirely would not land within a probability of it.
    let integrated = predict_multinomial_formula(&model, &ds).expect("integrated predict");
    let mut max_gap: f64 = 0.0;
    for i in 0..N {
        for k in 0..K {
            max_gap = max_gap.max((plugin[[i, k]] - integrated[[i, k]]).abs());
        }
    }
    assert!(
        max_gap < 0.5,
        "plug-in and integrated predictions differ by {max_gap:.4} somewhere; they \
         are different estimands of the SAME fit, not different fits"
    );
}
