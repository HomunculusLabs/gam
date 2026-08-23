// #2774: the per-smooth basis-adequacy report — the driver that turns a fitted
// model into "is each smooth's realized basis rich enough for the function it
// was asked to represent?".
//
// `include!`d into `drivers/mod.rs` like the other self-contained inference
// subsystems, so it shares the driver's flat namespace and import surface.
//
// The statistic itself lives in `gam_terms::inference::basis_adequacy` and is
// basis-agnostic: hand it an enrichment `Z`, the design, the weights and the
// working score and it returns a lack-of-fit p-value. This file owns the two
// decisions that statistic deliberately does not make — *what* to enrich with,
// and *when* the evidence is missing rather than negative.

/// Why a term's adequacy check produced (or failed to produce) a verdict.
///
/// This is a REASON, not a status code: every non-`Tested` value names the
/// specific piece of evidence that was absent, so a reader of an undetermined
/// row never has to guess whether the check passed, failed, or never ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BasisAdequacyProvenance {
    /// The lack-of-fit test ran against a radial enrichment of the term's own
    /// covariates. `p_value` is meaningful.
    RadialEnrichment,
    /// The term has no continuous structural covariates to enrich over — a
    /// pure random effect, a factor-level block, or a `by=` gate with no
    /// smooth axis.
    NoContinuousCovariates,
    /// Every structural covariate of the term is constant on the training rows,
    /// so no enrichment of it can carry information.
    DegenerateCovariates,
    /// The enrichment budget (set by `n`, the term's dimension and the memory
    /// ceiling) does not admit a basis WIDER than the one the term already
    /// realized. Testing a lower-resolution alternative than the fit already
    /// uses would answer a different question, so nothing is reported.
    EnrichmentBudgetBelowRealizedWidth,
    /// The enrichment basis could not be built (degenerate center geometry,
    /// an ill-conditioned kernel chart).
    EnrichmentBuildFailed,
    /// The fit did not retain the IRLS row state (weights, working response,
    /// linear predictor) the score needs. Saved-model and warm-start replays
    /// can land here.
    NoIrlsRowState,
    /// The design could not be materialized densely under the process memory
    /// governor, and this diagnostic will not evict the fit to run.
    DesignNotMaterializable,
    /// The test itself declined: no estimable enrichment direction survived the
    /// projection, or the assembled quadratic form was not finite.
    StatisticUnavailable,
}

impl BasisAdequacyProvenance {
    /// Serialized label carried into the model payload and the Python surface.
    pub fn label(self) -> &'static str {
        match self {
            Self::RadialEnrichment => "radial_enrichment",
            Self::NoContinuousCovariates => "no_continuous_covariates",
            Self::DegenerateCovariates => "degenerate_covariates",
            Self::EnrichmentBudgetBelowRealizedWidth => "enrichment_budget_below_realized_width",
            Self::EnrichmentBuildFailed => "enrichment_build_failed",
            Self::NoIrlsRowState => "no_irls_row_state",
            Self::DesignNotMaterializable => "design_not_materializable",
            Self::StatisticUnavailable => "statistic_unavailable",
        }
    }
}

/// One smooth term's basis-adequacy row.
///
/// `p_value` is `None` exactly when `provenance != RadialEnrichment`; the two
/// are emitted together so an absent verdict never appears without its reason.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BasisAdequacyRow {
    pub name: String,
    pub term_idx: usize,
    /// Realized coefficient width of the term — the `k'` a caller compares EDF
    /// against.
    pub basis_dim: usize,
    /// Dimension of the term's joint penalty null space. Those columns are
    /// never shrunk, so `basis_dim − nullspace_dim` is the penalizable capacity
    /// and the part of `k'` that EDF saturation is actually about.
    pub nullspace_dim: usize,
    /// The term's effective degrees of freedom — the influence-matrix trace over
    /// its coefficient block, the same number `summary().smooth_terms` reports.
    ///
    /// Carried HERE, beside `basis_dim` and `nullspace_dim`, because the three
    /// only mean anything together. The natural reading of a summary table is
    /// `edf` against `basis_dim`, and on a 16-D radial smooth that reads 20.9 of
    /// 24 — 87%, "saturated" — while the PENALIZED part is 3.9 of a capacity of
    /// 7. The null space is `d + 1` columns that are unpenalized and therefore
    /// always fully used; comparing against `basis_dim` counts them as evidence
    /// of saturation when they are evidence of nothing (#2774).
    ///
    /// `None` when the term owns no penalty block whose EDF could be traced.
    #[serde(default)]
    pub edf: Option<f64>,
    /// Realized column width of the higher-resolution alternative the residuals
    /// were tested against (equal to its center count — the spec-chart builder's
    /// width is a function of the centers alone, so the reference d.f. of the
    /// test cannot depend on which rows the basis was evaluated at).
    #[serde(default)]
    pub enrichment_dim: Option<usize>,
    /// Estimable enrichment directions left after the design was projected out
    /// — the reference d.f. of the test, and a direct measure of how much NEW
    /// resolution the alternative carried.
    #[serde(default)]
    pub enrichment_rank: Option<usize>,
    #[serde(default)]
    pub statistic: Option<f64>,
    #[serde(default)]
    pub p_value: Option<f64>,
    pub provenance: BasisAdequacyProvenance,
}

impl BasisAdequacyRow {
    /// Whether this row is evidence that the term's basis is too small at the
    /// given family-wise level.
    ///
    /// `None` — never `false` — when no test ran: "not measured" and "measured
    /// and adequate" are different states and a caller must be able to tell
    /// them apart.
    pub fn is_inadequate_at(&self, level: f64) -> Option<bool> {
        self.p_value.map(|p| p < level)
    }
}

/// Resolution multiple the enrichment aims for, relative to the term's realized
/// width. The alternative has to be a genuinely finer representation of the same
/// covariates; 4× is the smallest multiple that showed usable power on the #2774
/// fixture (a 2× enrichment of a 24-column 16-D Duchon reads p ≈ 2e-3 where 4×
/// reads p ≈ 1e-8), and larger multiples buy little because the reference d.f.
/// grows with them.
const ENRICHMENT_WIDTH_MULTIPLE: usize = 4;

/// Floor on the enrichment width. Below ~32 directions the χ² reference is
/// coarse enough that moderate lack of fit hides inside it.
const ENRICHMENT_WIDTH_MIN: usize = 32;

/// Ceiling on the enrichment width, independent of `n`. Past a few hundred
/// directions the reference d.f. grows faster than the non-centrality for any
/// realistic alternative, so a wider enrichment is strictly worse AND more
/// expensive.
const ENRICHMENT_WIDTH_MAX: usize = 256;

/// Flop budget for the `O(n·q²)` Gram accumulation, in multiply-adds. At
/// `n = 200_000` this caps `q` at 100; the diagnostic then costs ~1% of a fit
/// that already takes tens of seconds at that size.
const ENRICHMENT_FLOP_BUDGET: f64 = 2.0e9;

/// Byte budget for the materialized `n × q` enrichment. Keeps the diagnostic
/// from being the peak-memory term of a fit at very large `n`.
const ENRICHMENT_BYTE_BUDGET: f64 = 2.56e8;

/// Rows per center: never ask for a kernel chart the data cannot condition.
const ENRICHMENT_ROWS_PER_CENTER: usize = 8;

/// Byte budget for materializing an operator-backed design `X` (`n × p`).
///
/// Same ceiling as [`ENRICHMENT_BYTE_BUDGET`] and for the same reason: this is
/// the other `n`-tall matrix the report needs in one piece, and a diagnostic
/// that runs on every fit may not be the term that decides that fit's peak
/// residency. A design past the cap yields
/// [`BasisAdequacyProvenance::DesignNotMaterializable`] — a stated absence of
/// evidence, which is the contract this whole module is built on.
const DESIGN_BYTE_BUDGET: usize = 268_435_456;

/// The enrichment width for one term, or `None` when the budget cannot beat the
/// realized width.
fn enrichment_width(realized_width: usize, n_rows: usize) -> Option<usize> {
    let target = realized_width
        .saturating_mul(ENRICHMENT_WIDTH_MULTIPLE)
        .clamp(ENRICHMENT_WIDTH_MIN, ENRICHMENT_WIDTH_MAX);
    let flop_cap = (ENRICHMENT_FLOP_BUDGET / (n_rows.max(1) as f64))
        .sqrt()
        .floor();
    let byte_cap = ENRICHMENT_BYTE_BUDGET / (8.0 * n_rows.max(1) as f64);
    let cap = flop_cap
        .min(byte_cap)
        .min(ENRICHMENT_WIDTH_MAX as f64)
        .max(0.0) as usize;
    let width = target.min(cap).min(n_rows / ENRICHMENT_ROWS_PER_CENTER);
    (width > realized_width).then_some(width)
}

/// Standardized copy of the term's structural covariate columns.
///
/// Per-axis standardization is what makes ONE isotropic radial enrichment a
/// sensible alternative for every smooth kind: it removes the units and the
/// relative scales of the covariates, which the fitted term already handles its
/// own way (an input scale, per-axis anisotropy, a tensor product's marginals).
/// Constant columns carry no information and are dropped rather than producing a
/// zero-variance kernel axis.
fn standardized_covariates(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
) -> Option<Array2<f64>> {
    let n = data.nrows();
    if n == 0 {
        return None;
    }
    let mut kept: Vec<Array1<f64>> = Vec::new();
    for &col in feature_cols {
        if col >= data.ncols() {
            return None;
        }
        let column = data.column(col);
        if column.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mean = column.sum() / n as f64;
        let variance = column
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (n.max(2) - 1) as f64;
        let sd = variance.sqrt();
        if !(sd.is_finite() && sd > 0.0) {
            continue;
        }
        kept.push(column.mapv(|value| (value - mean) / sd));
    }
    if kept.is_empty() {
        return None;
    }
    let mut out = Array2::<f64>::zeros((n, kept.len()));
    for (index, column) in kept.iter().enumerate() {
        out.column_mut(index).assign(column);
    }
    Some(out)
}

/// The reference high-resolution alternative for a smooth over `covariates`.
///
/// A Duchon kernel at `centers` equal-mass centers, at the same
/// `(nullspace_order, power)` the formula front end resolves by default for that
/// dimension. Deliberately NOT the term's own basis kind rebuilt at a larger
/// `k`: the question a basis-adequacy check should answer is "is there ANY
/// smooth structure in these covariates the fit cannot represent", and a smooth
/// whose *kind* is wrong for the data (an isotropic Matérn with a badly-learned
/// range, say) would be invisible to an enrichment drawn from that same wrong
/// kind. Building one canonical alternative also means every term type — 1-D
/// spline, tensor product, radial, manifold — is tested by the same object with
/// the same null, instead of by a per-kind rebuild whose calibration would have
/// to be established separately for each.
///
/// The spec-chart entry point is used rather than `build_duchon_basis` because
/// its realized width is a function of the centers alone: a data-metric chart
/// would make the enrichment's dimension depend on the rows it is evaluated at,
/// and the reference d.f. of a test may not depend on that.
fn radial_enrichment(covariates: ArrayView2<'_, f64>, centers: usize) -> Option<Array2<f64>> {
    let dimension = covariates.ncols();
    if dimension == 0 || centers < 2 {
        return None;
    }
    let (nullspace_order, power) = gam_terms::basis::duchon_cubic_default(dimension);
    let spec = gam_terms::basis::DuchonBasisSpec {
        center_strategy: CenterStrategy::EqualMass {
            num_centers: centers,
        },
        periodic: None,
        length_scale: None,
        power,
        nullspace_order,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: gam_terms::basis::OneDimensionalBoundary::Open,
        radial_reparam: None,
    };
    let built = gam_terms::basis::build_duchon_basis_spec_chart(covariates, &spec).ok()?;
    // `as_dense_cow` PANICS on an operator-backed design; a diagnostic on the
    // ordinary fit path may not take the process down over its own scratch
    // basis, so take the fallible route and treat a refusal as "no enrichment".
    let dense = match built.design.as_dense_ref() {
        Some(matrix) => matrix.clone(),
        None => built
            .design
            .try_to_dense_by_chunks("basis_adequacy enrichment")
            .ok()?,
    };
    (dense.ncols() > 0 && dense.iter().all(|value| value.is_finite())).then_some(dense)
}

/// The IRLS row state the score test needs, read off the fit's retained P-IRLS
/// result.
struct ScoreRowState {
    hessian_weights: Array1<f64>,
    score_weights: Array1<f64>,
    score: Array1<f64>,
}

fn score_row_state(fit: &UnifiedFitResult, n_rows: usize) -> Option<ScoreRowState> {
    let pirls = fit.artifacts.pirls.as_ref()?;
    if pirls.finalweights.len() != n_rows
        || pirls.solveweights.len() != n_rows
        || pirls.solveworking_response.len() != n_rows
        || pirls.final_eta.len() != n_rows
    {
        return None;
    }
    // The working score `s = W_F ⊙ (z − η̂)`: the same object the outer
    // REML/LAML gradient calls its working residual (`reml/hyper.rs`), so the
    // score this test differentiates is the score the fit itself solved to zero.
    let score = &pirls.solveweights * &(&pirls.solveworking_response - &pirls.final_eta);
    let hessian_weights = pirls.finalweights.to_owned();
    let score_weights = pirls.solveweights.to_owned();
    if hessian_weights.iter().any(|w| !w.is_finite())
        || score_weights.iter().any(|w| !(w.is_finite() && *w >= 0.0))
        || score.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    Some(ScoreRowState {
        hessian_weights,
        score_weights,
        score,
    })
}

/// The per-smooth basis-adequacy report for a fitted standard GAM.
///
/// Never fails: a term whose evidence is missing gets an `Undetermined` row
/// carrying the reason. This runs on the ordinary fit path, so it may not be
/// able to refuse a fit, evict memory, or raise.
///
/// `data` is the materialized numeric feature matrix the design was built from
/// — the frame `SmoothBasisSpec::structural_feature_cols` indexes. Nothing in
/// this computation touches the coefficient frame: `X`, the weights and the
/// score are all row-space objects and the Gram is formed from `X` itself, so
/// there is no frame to get wrong.
pub fn basis_adequacy_report(
    data: ArrayView2<'_, f64>,
    design: &gam_terms::smooth::TermCollectionDesign,
    spec: &gam_terms::smooth::TermCollectionSpec,
    fit: &UnifiedFitResult,
) -> Vec<BasisAdequacyRow> {
    let term_count = design.smooth.terms.len();
    if term_count == 0 || term_count != spec.smooth_terms.len() {
        return Vec::new();
    }
    let n_rows = design.design.nrows();
    if n_rows == 0 || data.nrows() != n_rows {
        return Vec::new();
    }

    // The term's EDF, read through the design's OWN penalty-range accessor
    // rather than by re-deriving the flat-layout cursor walk. That walk has been
    // a filed defect five times (#1219, #1277, #1360, #1368, #1372) and
    // `smooth_term_penalty_range` exists so a consumer never has to redo it.
    let per_term_edf = |idx: usize| -> Option<f64> {
        let realized = design.smooth.terms.get(idx)?;
        let penalty_range = design.smooth_term_penalty_range(idx).ok().flatten()?;
        let smooth_start = design
            .design
            .ncols()
            .saturating_sub(design.smooth.total_smooth_cols());
        let global_range =
            (smooth_start + realized.coeff_range.start)..(smooth_start + realized.coeff_range.end);
        let edf = fit.per_term_edf(global_range, penalty_range.start, penalty_range.len());
        edf.is_finite().then_some(edf)
    };
    let undetermined = |idx: usize, reason: BasisAdequacyProvenance| BasisAdequacyRow {
        name: design.smooth.terms[idx].name.clone(),
        term_idx: idx,
        basis_dim: design.smooth.terms[idx].coeff_range.len(),
        nullspace_dim: design.smooth.terms[idx].wald_unpenalized_dim(),
        edf: per_term_edf(idx),
        enrichment_dim: None,
        enrichment_rank: None,
        statistic: None,
        p_value: None,
        provenance: reason,
    };

    let Some(rows_state) = score_row_state(fit, n_rows) else {
        return (0..term_count)
            .map(|idx| undetermined(idx, BasisAdequacyProvenance::NoIrlsRowState))
            .collect();
    };
    // `as_dense_ref` is `Some` ONLY for `Dense(Materialized)`. A reparameterized
    // smooth ships `Dense(Lazy(op))` — `X·Qs` held as an operator — and that is
    // what every radial/Duchon term the fit path reparameterizes becomes,
    // INCLUDING the 16-D `duchon(pc1..pc16, centers=24)` fixture this issue was
    // filed on. Taking `as_dense_ref` as the whole answer made the report return
    // `design_not_materializable` for exactly the fits it exists to diagnose.
    //
    // The chunked route is the one `radial_enrichment` already takes for the
    // enrichment above, and `DesignNotMaterializable`'s own documentation
    // already reads "could not be materialized densely under the process memory
    // governor" — the budgeted call is what makes that sentence true rather than
    // universal. This is an observability-only diagnostic on the ordinary fit
    // path, so it refuses BEFORE allocating instead of becoming the peak
    // residency term of somebody's fit.
    let materialized_design;
    let dense_design = match design.design.as_dense_ref() {
        Some(matrix) => matrix,
        None => match design
            .design
            .try_to_dense_by_chunks_budgeted("basis_adequacy design", DESIGN_BYTE_BUDGET)
        {
            Ok(matrix) => {
                materialized_design = matrix;
                &materialized_design
            }
            Err(_) => {
                return (0..term_count)
                    .map(|idx| undetermined(idx, BasisAdequacyProvenance::DesignNotMaterializable))
                    .collect();
            }
        },
    };
    // `G = XᵀW_H X`, factored ONCE for the whole model: the projection is applied
    // per smooth term but `G` is a property of the design and the weights, and
    // re-factoring it per term would charge `O(p³)` per smooth on a fit that
    // runs only a few dozen IRLS iterations in total.
    let Some(design_gram) =
        gam_linalg::matrix::LinearOperator::diag_xtw_x(&design.design, &rows_state.hessian_weights)
            .ok()
            .as_ref()
            .and_then(|gram| {
                gam_terms::inference::basis_adequacy::DesignGramFactor::new(gram.view())
            })
    else {
        return (0..term_count)
            .map(|idx| undetermined(idx, BasisAdequacyProvenance::DesignNotMaterializable))
            .collect();
    };
    // The dispersion that scales the score's variance is the same multiplier the
    // fit publishes on its coefficient covariance (`1` for every family carrying
    // its dispersion inside the IRLS weight, `φ̂` for the profiled Gaussian).
    let dispersion = fit
        .coefficient_covariance_scale()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    let scale = if fit.likelihood_scale.wald_scale_is_estimated() {
        gam_terms::inference::smooth_test::SmoothTestScale::Estimated
    } else {
        gam_terms::inference::smooth_test::SmoothTestScale::Known
    };
    let residual_df = fit.wald_residual_degrees_of_freedom();

    (0..term_count)
        .map(|idx| {
            let realized = &design.smooth.terms[idx];
            let realized_width = realized.coeff_range.len();
            let feature_cols = spec.smooth_terms[idx].basis.structural_feature_cols();
            if feature_cols.is_empty() {
                return undetermined(idx, BasisAdequacyProvenance::NoContinuousCovariates);
            }
            let Some(covariates) = standardized_covariates(data, &feature_cols) else {
                return undetermined(idx, BasisAdequacyProvenance::DegenerateCovariates);
            };
            let Some(centers) = enrichment_width(realized_width, n_rows) else {
                return undetermined(
                    idx,
                    BasisAdequacyProvenance::EnrichmentBudgetBelowRealizedWidth,
                );
            };
            let Some(enrichment) = radial_enrichment(covariates.view(), centers) else {
                return undetermined(idx, BasisAdequacyProvenance::EnrichmentBuildFailed);
            };
            let outcome = gam_terms::inference::basis_adequacy::basis_adequacy_score_test(
                gam_terms::inference::basis_adequacy::BasisAdequacyInput {
                    enrichment: enrichment.view(),
                    design: dense_design.view(),
                    hessian_weights: rows_state.hessian_weights.view(),
                    score_weights: rows_state.score_weights.view(),
                    score: rows_state.score.view(),
                    design_gram: &design_gram,
                    dispersion,
                    residual_df,
                    scale,
                },
            );
            match outcome {
                Some(result) => BasisAdequacyRow {
                    name: realized.name.clone(),
                    term_idx: idx,
                    basis_dim: realized_width,
                    nullspace_dim: realized.wald_unpenalized_dim(),
                    edf: per_term_edf(idx),
                    enrichment_dim: Some(enrichment.ncols()),
                    enrichment_rank: Some(result.rank),
                    statistic: Some(result.statistic),
                    p_value: Some(result.p_value),
                    provenance: BasisAdequacyProvenance::RadialEnrichment,
                },
                None => undetermined(idx, BasisAdequacyProvenance::StatisticUnavailable),
            }
        })
        .collect()
}

/// Family-wise level at which a fit-time note is raised.
///
/// A per-term 0.1% Bonferroni-corrected over the model's smooth terms. The
/// threshold is stringent on purpose: this note fires on the ordinary fit path
/// for every model, so its false-alarm rate is a property of the whole engine's
/// output, not of one report a user asked for. Measured on the #2774 fixture at
/// the small (`n = 3000`) tier, a correctly specified 16-D Duchon fit clears
/// `p < 0.001` in 0 of 20 replicates while the underfitted one trips it in 18 of
/// 20 — so the corrected level keeps the note essentially silent on adequate
/// fits without giving up the detection it exists for. Callers wanting the
/// ordinary 5% read `p_value` themselves.
pub const BASIS_ADEQUACY_NOTE_LEVEL: f64 = 1.0e-3;

/// The user-facing advisories a basis-adequacy report produces, in the same
/// `inference_notes` channel as the mgcv-style basis-reduction notes.
///
/// Only `Inadequate` terms produce a note. An `Undetermined` row is not an
/// advisory — it is an absence of evidence, and saying so at fit time on every
/// random-effect term would drown the channel it shares.
pub fn basis_adequacy_notes(rows: &[BasisAdequacyRow]) -> Vec<String> {
    let tested = rows
        .iter()
        .filter(|row| row.p_value.is_some())
        .count()
        .max(1);
    let level = BASIS_ADEQUACY_NOTE_LEVEL / tested as f64;
    rows.iter()
        .filter(|row| row.is_inadequate_at(level).unwrap_or(false))
        .map(|row| {
            let p_value = row.p_value.unwrap_or(f64::NAN);
            let rank = row.enrichment_rank.unwrap_or(0);
            // The penalized reading, stated rather than left to be derived: a
            // reader who compares total EDF against the column count sees a
            // 16-D radial smooth as 87% saturated at 65% of its real capacity,
            // because `nullspace_dim` of those columns are unpenalized and are
            // always fully used.
            let capacity = row.basis_dim.saturating_sub(row.nullspace_dim);
            let occupancy = match row.edf {
                Some(edf) if capacity > 0 => format!(
                    ", using {:.2} of its {capacity} penalizable dimensions",
                    (edf - row.nullspace_dim as f64).clamp(0.0, capacity as f64)
                ),
                _ => String::new(),
            };
            format!(
                "basis adequacy: smooth '{}' has {} coefficient columns ({} of them the \
                 unpenalized null space{occupancy}), and the fit's residuals still carry \
                 structure in its covariates that this basis cannot represent (lack-of-fit \
                 p = {p_value:.3e} against {rank} higher-resolution directions). Refit with a \
                 larger basis for this term and compare; the reported convergence certificate \
                 covers the optimizer only, not the adequacy of the basis it converged on. \
                 See gam#2774.",
                row.name, row.basis_dim, row.nullspace_dim,
            )
        })
        .collect()
}
