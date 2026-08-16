//! #2676 probe: is the `geo_disease_*_matern` penalty-map redundancy EXACT, or
//! only close?
//!
//! The whole #2676 deflation apparatus is licensed by one premise — that the
//! penalty map carries an EXACT linear redundancy `sum_i w_i S_i = 0`, so the
//! criterion is exactly constant along `lambda + s w` and the rho-curvature
//! there is the chain-rule term and nothing else. That premise entered the
//! thread from a diagnostic that prints `cos` to six decimals, which cannot
//! tell `1 - 1e-16` from `1 - 1e-9` (#2748 already said so about the printing;
//! nobody re-derived the premise).
//!
//! This probe measures the defect itself, at a swept length scale, and prints
//! the ONE discriminator that separates the two readings:
//!
//! * if `delta(kappa) = min_c ||S_2 - c S_0||_F / ||S_0||_F` is FLAT in kappa
//!   at the construction's noise level, the two operators are the same object
//!   and the defect is arithmetic — an exact invariance that the Gram's rank
//!   test is missing because it judges the Gram's entries at the EIGENSOLVER's
//!   backward error rather than at the error those entries were built with;
//! * if `delta(kappa)` MOVES with kappa — and in particular if it is far above
//!   any plausible construction error away from the operating point — then the
//!   operators are genuinely distinct and the redundancy is a near-degeneracy
//!   of that geometry, not an invariance. The certificate then has an
//!   approximate flat direction to reason about and no exact one, and the
//!   deflation cannot be what fixes it.
//!
//! Run: `cargo run --release --example probe2676_penalty_map_defect --`
//!      `[centers] [n] [n_pcs]`

use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternLengthScale, MaternNu,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitRequest, FitResult, StandardFitRequest};
use ndarray::{Array1, Array2};

fn spec_at(n_pcs: usize, centers: usize, length_scale: MaternLengthScale) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "geo".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..n_pcs).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::EqualMassCovarRepresentative {
                        num_centers: centers,
                    },
                    periodic: None,
                    length_scale,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::default(),
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

/// `min_c ||B - c A||_F / ||A||_F`, and the minimizing `c`. This is the defect
/// of the proportionality claim in the only norm the Gram's rank test sees.
fn proportionality_defect(a: &Array2<f64>, b: &Array2<f64>) -> (f64, f64) {
    let aa: f64 = a.iter().map(|v| v * v).sum();
    let ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    if aa <= 0.0 {
        return (f64::INFINITY, f64::NAN);
    }
    let c = ab / aa;
    let residual: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (y - c * x) * (y - c * x))
        .sum::<f64>()
        .sqrt();
    (residual / aa.sqrt(), c)
}

fn canonical_of(
    design: &gam::smooth::TermCollectionDesign,
) -> (Vec<gam::terms::construction::CanonicalPenalty>, usize) {
    let specs: Vec<gam::terms::PenaltySpec> = design
        .penalties
        .iter()
        .map(|penalty| gam::terms::PenaltySpec::Block {
            local: penalty.local.clone(),
            col_range: penalty.col_range.clone(),
            prior_mean: penalty.prior_mean.clone(),
            structure_hint: penalty.structure_hint.clone(),
            op: penalty.op.clone(),
        })
        .collect();
    let p = design.design.ncols();
    let (canonical, _) = gam::terms::construction::canonicalize_penalty_specs(
        &specs,
        &vec![0usize; specs.len()],
        p,
        "#2676 probe",
    )
    .expect("canonicalization must succeed");
    (canonical, p)
}

/// The Matern length scale the build actually realized, read off the term's
/// basis metadata rather than off the spec (an `Auto` spec carries `None`
/// until the planner resolves it).
fn cold_length_scale(design: &gam::smooth::TermCollectionDesign) -> f64 {
    for term in &design.smooth.terms {
        if let gam::terms::basis::BasisMetadata::Matern { length_scale, .. } = &term.metadata {
            return length_scale.original_value();
        }
    }
    f64::NAN
}

fn report(label: &str, design: &gam::smooth::TermCollectionDesign) {
    let (canonical, p) = canonical_of(design);
    let k = canonical.len();
    let invariance = gam::solver::penalty_invariance::PenaltyMapInvariance::from_canonical_penalties(
        &canonical, p,
    );
    let (dimension, resolution) = match &invariance {
        Ok(value) => (value.dimension(), value.resolution()),
        Err(error) => {
            eprintln!("[probe2676] {label}: invariance unavailable: {error}");
            (usize::MAX, f64::NAN)
        }
    };
    eprintln!(
        "[probe2676] {label}: k={k} p={p} certified_nullity={dimension} gram_resolution={resolution:.3e} \
         sources={:?} ranks={:?}",
        design
            .penaltyinfo
            .iter()
            .map(|info| format!("{:?}", info.penalty.source))
            .collect::<Vec<_>>(),
        canonical.iter().map(|c| c.rank()).collect::<Vec<_>>(),
    );
    for i in 0..k {
        for j in (i + 1)..k {
            if canonical[i].col_range != canonical[j].col_range {
                continue;
            }
            let (defect, c) = proportionality_defect(&canonical[i].local, &canonical[j].local);
            eprintln!(
                "[probe2676] {label}:   pair=({i},{j}) relative_defect={defect:.6e} best_c={c:.9} \
                 |S_i|_F={:.6e} |S_j|_F={:.6e}",
                canonical[i]
                    .local
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt(),
                canonical[j]
                    .local
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt(),
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let centers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let n_pcs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Error);

    let (x, y) = gam::test_support::synthetic::geo_disease_columns(n, 20260226);
    let x = if x.ncols() <= n_pcs {
        x
    } else {
        x.slice(ndarray::s![.., ..n_pcs]).to_owned()
    };
    eprintln!("[probe2676] centers={centers} n={n} n_pcs={n_pcs}");

    // ── 1. The cold design, i.e. what `Auto` resolves to before any fit ──
    let cold = gam::smooth::build_term_collection_design(
        x.view(),
        &spec_at(n_pcs, centers, MaternLengthScale::auto()),
    )
    .expect("cold design builds");
    report("cold(auto)", &cold);
    let resolved = cold_length_scale(&cold);
    eprintln!("[probe2676] cold auto length_scale = {resolved:.12e}");

    // ── 2. The defect as a FUNCTION of the length scale ──
    //
    // The discriminator. A defect that is flat across four decades of kappa is
    // arithmetic; one that moves is geometry.
    for exponent in -6..=6 {
        let scale = resolved * 2.0_f64.powi(exponent);
        let design = match gam::smooth::build_term_collection_design(
            x.view(),
            &spec_at(n_pcs, centers, MaternLengthScale::fixed(scale)),
        ) {
            Ok(design) => design,
            Err(error) => {
                eprintln!("[probe2676] length_scale={scale:.6e}: build failed: {error}");
                continue;
            }
        };
        report(&format!("ls={scale:.6e}"), &design);
    }

    // ── 3. The design the FIT realizes, which is what the acceptance reads ──
    let n_rows = y.len();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(Array1::ones(n_rows)),
        offset: std::sync::Arc::new(Array1::zeros(n_rows)),
        spec: spec_at(n_pcs, centers, MaternLengthScale::auto()),
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        estimate_tweedie_p: false,
        options: FitOptions {
            compute_inference: true,
            ..FitOptions::default()
        },
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
    }));
    match result {
        Ok(FitResult::Standard(fit)) => {
            report("realized", &fit.design);
            eprintln!(
                "[probe2676] realized length_scale = {:.12e}",
                cold_length_scale(&fit.design)
            );
        }
        Ok(_) => eprintln!("[probe2676] non-standard result"),
        Err(error) => eprintln!("[probe2676] FIT FAILED: {error}"),
    }
}
