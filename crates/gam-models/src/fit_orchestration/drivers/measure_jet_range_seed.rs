// #2750 — the measure-jet representer range is SCREENED against the response
// before the outer ψ search refines it.
//
// `include!`d into `drivers/mod.rs` exactly like `constant_curvature_profile.rs`,
// whose shape this file deliberately copies: a cheap value-only bracket that
// picks the basin, then the existing exact machinery refines inside it.
//
// ## The defect this closes
//
// `ℓ` is the ONE design-moving coordinate of a measure-jet term: it decides
// which span the representers occupy, and λ cannot move a span. #2761 made it
// REML-selected, but the selection is a local descent seeded at a pure geometry
// heuristic (the median nearest-node spacing), and the profiled criterion in
// `ln ℓ` is not unimodal — as `ℓ` grows past a few node spacings the Gaussian
// columns become collinear, the rank-revealing identifiability section drops
// columns, and the criterion steps. Measured end-to-end on
// `measure_jet_formula_fit_robustness_sweep` seed 1:
//
// ```text
//   ℓ (orig)   profiled V     held-out RMSE
//   0.0204      -234.5           0.0173     <- auto seed: LOCAL minimum
//   0.0345      -231.1           0.0185     <- barrier (+3.4)
//   0.0757      -236.4           0.0161
//   0.2163      -246.4           0.0110
//   0.8030      -256.3           0.0084     <- GLOBAL minimum, 21.7 deeper
//   1.0438      -198.5           0.0538     <- past the diameter: block collapses
//   s(x, bs="tp")  -247.4        0.0123
// ```
//
// The free search lands at `V = -234.6`: it never leaves the first basin. The
// criterion's ranking tracks the truth at every node, so the criterion is right
// and the search is caged — and the λ that comes back is a faithful readout of
// a range nothing could move. That is the "1-D fits select a too-large λ" of
// gam#2750, and the same cage is what leaves the term behind a same-size
// Matérn in gam#2761.
//
// ## What this does, and what it deliberately does NOT do
//
// The screen replaces a heuristic SEED with a data-chosen one. It does not
// replace the outer search, does not add a coordinate, and does not fire when
// the user pinned `length_scale=` (an explicit range is a request, not a seed —
// the same mgcv-`sp=` convention the range already follows).
//
// The screening criterion is the closed-form profiled Gaussian REML of the term
// ALONE against the response, with the double-penalty component off — the exact
// object `constant_curvature_psi_profile_value` screens κ with, for the same
// reason: the bracket needs a ranking, not a fit, and a per-node full-collection
// multi-ρ solve would multiply the cost of every measure-jet fit by the node
// count. Two consequences are accepted openly:
//
//   * on a multi-term formula the screen ranks the term's own fit to `y`, not
//     its fit to the partial residual. It is a seed; the joint ψ/ρ search that
//     follows is the estimator.
//   * on a non-Gaussian family the screen is Gaussian-REML on the response
//     scale. Still strictly more informed than the geometry-only heuristic it
//     replaces, which never looks at `y` at all.

/// One screening evaluation: the profiled Gaussian REML of `[1 | X(ℓ)]` with
/// the term's single jet-energy penalty, at one candidate range.
///
/// `data` is the term's feature columns in the STANDARDIZED frame the basis is
/// realized in, and `ell` is a standardized range, so this never has to reason
/// about the input-frame conversion the term-collection builder owns.
///
/// Returns `None` (never an error) when the candidate cannot be realized or
/// scored: a bracket node that refuses is simply not a candidate, and one
/// unbuildable range must not fail an otherwise healthy fit.
fn measure_jet_range_screen_value(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    spec: &gam_terms::basis::MeasureJetBasisSpec,
    ell: f64,
) -> Option<f64> {
    if !(ell.is_finite() && ell > 0.0) {
        return None;
    }
    let mut screen = spec.clone();
    screen.length_scale = ell;
    // The screen ranks SPANS. The null-component candidate is a second REML
    // coordinate, not a property of the span, and carrying it would make every
    // node a multi-ρ solve; the shipped fit still gets it.
    screen.double_penalty = false;
    screen.learn_length_scale = false;
    let basis = gam_terms::basis::build_measure_jet_basis(data, &screen).ok()?;
    if basis.active_penalties.len() != 1 {
        return None;
    }
    let smooth_design = basis.design.to_dense();
    let (n, p) = smooth_design.dim();
    if n != y.len() || p == 0 {
        return None;
    }
    let mut design = Array2::<f64>::ones((n, p + 1));
    design
        .slice_mut(ndarray::s![.., 1..])
        .assign(&smooth_design);
    let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
    penalty
        .slice_mut(ndarray::s![1.., 1..])
        .assign(&basis.active_penalties[0].matrix);
    // Rank-reveal, exactly as the term collection does at fit time. A Gaussian
    // representer design LOSES COLUMNS as the range grows — that is the whole
    // reason the range is worth screening — and a criterion that keeps the
    // dependent columns does not merely score them badly, it REFUSES: the
    // profiled evaluator classifies the penalty in the `XᵀWX` metric, and a
    // singular `X` turns an exactly-PSD penalty into one with a negative
    // eigenvalue. A screen that refuses at every long range would report the
    // seed basin as the optimum for the second time (gam#2750), which is the
    // defect, not a measurement of it.
    let (design, penalty) = whiten_to_identifiable_subspace(&design, &penalty)?;
    let response = y.insert_axis(ndarray::Axis(1));
    let fit = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
        design.view(),
        response,
        penalty.view(),
        weights,
        None,
    )
    .ok()?;
    fit.reml_score.is_finite().then_some(fit.reml_score)
}

/// Restrict `(design, penalty)` to the subspace the design actually identifies,
/// in a chart where the Gram is the identity.
///
/// ## Why this is needed at all
///
/// A Gaussian representer design LOSES RANK as the range grows — that is the
/// whole reason the range is worth screening — and the profiled evaluator
/// classifies the penalty in the `XᵀWX` metric (`L⁻¹ S L⁻ᵀ`, `L` the Cholesky
/// of the Gram). On a rank-deficient design that congruence turns an
/// exactly-PSD penalty into one with a negative eigenvalue and the evaluation
/// REFUSES: measured on the gam#2750 fixture, `-8.9e-8` at `ℓ = 0.166` and
/// `-2.8e-7` at `ℓ = 0.298`. A screen that refuses at exactly the ranges it
/// exists to reach would report the seed basin as the optimum for the second
/// time, which is the defect rather than a measurement of it.
///
/// ## Why it is free
///
/// For an INVERTIBLE `T`, `X → XT`, `S → TᵀST` leaves the profiled criterion
/// exactly unchanged: `log|Tᵀ(XᵀWX + λS)T|` and `log|λTᵀST|₊` both pick up
/// `2 ln|det T|` and the deviance is invariant, so the `2 ln|det T|` cancels in
/// the difference. Whitening is therefore a free change of chart, not a change
/// of model — and it makes the Gram exactly `I`, so the congruence above is the
/// identity and the evaluator sees the penalty as it was assembled.
///
/// The map is only non-invertible where it drops directions, and dropping is
/// the honest reading there: the collection's own realization drops columns at
/// the same ranges. The cut is at `√ε` of the leading Gram eigenvalue — the
/// half-mantissa bar, i.e. the point past which a direction cannot survive
/// being squared into a Gram and inverted back out with any significant digits.
fn whiten_to_identifiable_subspace(
    design: &Array2<f64>,
    penalty: &Array2<f64>,
) -> Option<(Array2<f64>, Array2<f64>)> {
    let p = design.ncols();
    if p == 0 || penalty.nrows() != p || penalty.ncols() != p {
        return None;
    }
    let gram = gam_linalg::faer_ndarray::fast_ata(design);
    let (values, vectors) =
        gam_linalg::faer_ndarray::strict_symmetric_eigh(&gram, faer::Side::Lower).ok()?;
    let leading = values.iter().copied().fold(0.0_f64, |a, v| a.max(v));
    if !(leading.is_finite() && leading > 0.0) {
        return None;
    }
    let cut = leading * f64::EPSILON.sqrt();
    let kept: Vec<usize> = (0..p).filter(|&i| values[i] > cut).collect();
    if kept.is_empty() {
        return None;
    }
    let mut transform = Array2::<f64>::zeros((p, kept.len()));
    for (column, &index) in kept.iter().enumerate() {
        let inverse_root = values[index].sqrt().recip();
        for row in 0..p {
            transform[(row, column)] = vectors[(row, index)] * inverse_root;
        }
    }
    let whitened_design = gam_linalg::faer_ndarray::fast_ab(design, &transform);
    let half = gam_linalg::faer_ndarray::fast_ab(penalty, &transform);
    let mut whitened_penalty = gam_linalg::faer_ndarray::fast_atb(&transform, &half);
    // The congruence is symmetric in exact arithmetic; make it so in floating
    // point as well, because the evaluator's spectral classification refuses a
    // matrix that is not exactly self-adjoint.
    let transposed = whitened_penalty.t().to_owned();
    whitened_penalty += &transposed;
    whitened_penalty *= 0.5;
    Some((whitened_design, whitened_penalty))
}

/// The screened range for ONE measure-jet term, in standardized units, or
/// `None` when no candidate scored.
///
/// Walks the term's realized scale band (see
/// [`gam_terms::basis::measure_jet_range_bracket`]), extends geometrically at
/// the band's own log step while an endpoint keeps improving and the node cloud
/// still admits distinct representers, and finishes with one parabolic step
/// through the three points around the argmin.
fn screen_measure_jet_range(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    spec: &gam_terms::basis::MeasureJetBasisSpec,
) -> Option<f64> {
    let bracket = gam_terms::basis::measure_jet_range_bracket(data, spec).ok()?;
    if bracket.nodes.len() < 2 || !(bracket.log_step.is_finite() && bracket.log_step > 0.0) {
        return None;
    }
    // `(ln ell, value)`, kept sorted by ln ell so the parabolic step below can
    // read its three points off neighbouring entries.
    let mut scored: Vec<(f64, f64)> = bracket
        .nodes
        .iter()
        .filter_map(|&ell| {
            measure_jet_range_screen_value(data, y, weights, spec, ell).map(|v| (ell.ln(), v))
        })
        .collect();
    if scored.is_empty() {
        return None;
    }
    // Endpoint walk. The criterion is still descending at a band end whenever
    // the band's resolution is coarser than the basin, and refusing to look is
    // how a bracket silently reports its own edge as an optimum.
    //
    // The walk is UPWARD ONLY, and that is a statement about the coordinate, not
    // a simplification. The band's bottom node IS the physical floor — the median
    // nearest-node spacing, below which neighbouring representers stop
    // overlapping, the design stops being a partition of unity, and rows between
    // nodes fall outside every representer's support — and the band's bottom node
    // is already scored. There is nowhere below it to walk to.
    //
    // It used to walk downward as well, under a guard that could not fire
    // (gam#2750): `next_ln < floor_ln - log_step * scored.len()` recedes by one
    // log step for every node the walk pushes, exactly as fast as `next_ln`
    // descends, so the comparison is false at every iteration for any bracket
    // with two or more nodes. The only stops were "the criterion stopped
    // improving" and "the basis refused to build", i.e. the documented floor was
    // not enforced at all and the screen could seed a range below the one the
    // outer search's own window (`measure_jet_ln_range_window`) is floored at —
    // which the #2454 incumbent-containment rule would then have widened that
    // window to admit, reintroducing exactly the region the floor excludes.
    //
    // The cap is the bracket's own ceiling, so the walk still introduces no
    // length of its own.
    let ceiling_ln = bracket.ceiling.max(bracket.nodes[0]).ln();
    loop {
        let (best_ln, best_value) =
            scored
                .iter()
                .copied()
                .fold((f64::NAN, f64::INFINITY), |acc, node| {
                    if node.1 < acc.1 { node } else { acc }
                });
        let edge = scored.last().copied()?;
        if edge.0 != best_ln || edge.1 != best_value {
            break;
        }
        let next_ln = edge.0 + bracket.log_step;
        if next_ln > ceiling_ln {
            break;
        }
        let Some(value) = measure_jet_range_screen_value(data, y, weights, spec, next_ln.exp())
        else {
            break;
        };
        scored.push((next_ln, value));
        if value >= edge.1 {
            break;
        }
    }
    let argmin = (0..scored.len()).fold(0usize, |best, idx| {
        if scored[idx].1 < scored[best].1 {
            idx
        } else {
            best
        }
    });
    let mut chosen = scored[argmin];
    // One parabolic step through the bracketing triple. Cheap, deterministic,
    // and kept only if it actually scores better than the node it refines.
    if argmin > 0 && argmin + 1 < scored.len() {
        let (x0, f0) = scored[argmin - 1];
        let (x1, f1) = scored[argmin];
        let (x2, f2) = scored[argmin + 1];
        let denominator = (x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0);
        if denominator.abs() > f64::EPSILON {
            let numerator = (x1 - x0) * (x1 - x0) * (f1 - f2) - (x1 - x2) * (x1 - x2) * (f1 - f0);
            let vertex = x1 - 0.5 * numerator / denominator;
            if vertex.is_finite() && vertex > x0 && vertex < x2 {
                if let Some(value) =
                    measure_jet_range_screen_value(data, y, weights, spec, vertex.exp())
                    && value < chosen.1
                {
                    chosen = (vertex, value);
                }
            }
        }
    }
    Some(chosen.0.exp())
}

/// Screen every AUTO measure-jet representer range in `spec` against the
/// response and write the winner back, in the spec's own ORIGINAL input units.
///
/// Returns the number of terms whose range moved. A term is eligible only when
/// its range is the auto sentinel (`length_scale == 0.0`) and its quadrature is
/// not frozen: an explicit `length_scale=` is a request, and a frozen term is a
/// replay with nothing left to seed.
///
/// Failure to screen is never an error. Every refusal path leaves the term at
/// the geometry heuristic, which is exactly the pre-#2750 behaviour.
pub(crate) fn seed_measure_jet_auto_ranges(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    spec: &mut TermCollectionSpec,
) -> usize {
    let n = data.nrows();
    if y.len() != n || weights.len() != n || n == 0 {
        return 0;
    }
    let positive_weights = weights.iter().all(|w| w.is_finite() && *w > 0.0);
    let mut seeded = 0usize;
    for term in spec.smooth_terms.iter_mut() {
        let SmoothBasisSpec::MeasureJet {
            feature_cols,
            spec: mj,
            input_scale,
        } = &mut term.basis
        else {
            continue;
        };
        if mj.length_scale != 0.0 || mj.frozen_quadrature.is_some() || input_scale.is_some() {
            continue;
        }
        let Ok(columns) = select_columns(data, feature_cols) else {
            continue;
        };
        // The basis is realized in the auto-standardized frame; screen there,
        // then hand the winner back in original units because a FRESH spec's
        // `length_scale` is an original-units request (the scale contract's
        // asymmetric fresh/replay rule).
        let Ok(scale) =
            gam_terms::smooth::input_standardization::estimate_isotropic_scale(columns.view())
        else {
            continue;
        };
        let mut standardized = columns;
        scale.standardize(&mut standardized);
        let mut screen_spec = mj.clone();
        // Center selection happens in the standardized frame here, so a
        // resolved (already-standardized-by-the-builder) strategy would be
        // double-converted. Only auto strategies reach this path; anything
        // carrying explicit coordinates is left to the builder.
        if matches!(
            screen_spec.center_strategy,
            gam_terms::basis::CenterStrategy::UserProvided(_)
        ) {
            continue;
        }
        screen_spec.identifiability = gam_terms::basis::MeasureJetIdentifiability::CenterSumToZero;
        let screened = screen_measure_jet_range(
            standardized.view(),
            y,
            positive_weights.then_some(weights),
            &screen_spec,
        );
        let Some(ell) = screened else {
            continue;
        };
        let original = scale
            .to_original_units(gam_terms::StandardizedUnits::new(ell))
            .original_value();
        if original.is_finite() && original > 0.0 {
            mj.length_scale = original;
            seeded += 1;
        }
    }
    seeded
}
