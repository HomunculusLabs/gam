// The measure-jet term's outer-ψ plumbing: which coordinates a term enrolls,
// what they are seeded at, what box they are searched over, and how an optimized
// ψ is written back.
//
// `include!`d into `smooth/mod.rs` immediately after `term_specs.rs`, in the same
// flat module, so every path this exposes is unchanged. It is carved out of that
// file because it is one coherent contract with one invariant — the coordinate
// ORDER (`ln ℓ` first when enrolled, then the multiscale penalty dials) must be
// identical in `measure_jet_psi_dim`, `measure_jet_psi_seed`,
// `measure_jet_psi_bound_values`, `apply_measure_jet_psi` and the producer
// `build_measure_jet_basis_psi_derivatives`. Five functions that must agree
// coordinate-for-coordinate belong in one file where a reader can check them
// against each other, not scattered through a 10k-line specification module.


/// The measure-jet term's spec, when `term_idx` is a measure-jet smooth.
/// Single accessor for every dial-plumbing dispatch below.
pub fn measure_jet_term_spec(
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<&crate::basis::MeasureJetBasisSpec> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec, .. } => Some(spec),
            _ => None,
        })
}

/// Single source for measure-jet outer-ψ enrollment: the lnτ dial is
/// undefined in the τ = 0 pseudo-inverse oracle mode (see
/// `build_measure_jet_basis_psi_derivatives`), so only a positive ridge
/// enrolls the dial group. `spatial_term_supports_hyper_optimization` and
/// `spatial_term_uses_per_axis_psi` both defer here so the θ-layout
/// sources cannot disagree.
pub fn measure_jet_enrolls_psi(mj: &crate::basis::MeasureJetBasisSpec) -> bool {
    // Two independent enrollment sources (#1116), both explicit:
    //   * the design-moving representer length-scale ℓ (`learn_length_scale`),
    //     available in every mode when the spec opts in;
    //   * the multiscale penalty dials (s, α, lnτ): the per-scale spectral
    //     split's (α, lnτ) ride the explicit `multiscale` opt-in, and the lnτ
    //     channel additionally needs a positive ridge (τ = 0 is the
    //     pseudo-inverse oracle mode where lnτ is undefined).
    // A term enrolls if EITHER source is active.
    measure_jet_learns_length_scale(mj)
        || (mj.tau0 > 0.0 && crate::basis::measure_jet_multiscale_mode(mj))
}

/// Whether the design-moving ℓ dial is enrolled for this term. ℓ is fixed by
/// default and learnable in every mode only when `learn_length_scale = true`.
pub fn measure_jet_learns_length_scale(mj: &crate::basis::MeasureJetBasisSpec) -> bool {
    mj.learn_length_scale
}

pub fn freeze_measure_jet_length_scale_learning(spec: &mut TermCollectionSpec) -> usize {
    let mut frozen = 0;
    for term in spec.smooth_terms.iter_mut() {
        if let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &mut term.basis
            && mj.learn_length_scale
        {
            mj.learn_length_scale = false;
            frozen += 1;
        }
    }
    frozen
}

/// Measure-jet ψ dial boxes. The dials are NOT log-kernel-scales, so the
/// κ-window machinery never applies: `α` spans density-weighted (0) through
/// past-Coifman–Lafon (>1) normalization, and `lnτ` covers the ridge from
/// numerically-exact-projection to heavy noise-floor damping. (The energy
/// order `s` is the pinned explicit value or absorbed by the REML-learned
/// per-scale amplitudes — see `measure_jet_penalty_psi_dim` — so it carries no
/// dial box.)
pub const MEASURE_JET_PSI_ALPHA_BOUNDS: (f64, f64) = (-1.0, 3.0);

pub const MEASURE_JET_PSI_LN_TAU_BOUNDS: (f64, f64) = (-18.420680743952367, 4.605170185988092);

/// Number of multiscale PENALTY dials (excluding the design-moving ℓ):
/// multiscale (per-scale spectral) mode carries (α, lnτ) = 2 — the order is
/// either the pinned explicit `s` or absorbed by the REML-learned per-scale
/// amplitudes, so it is NOT a dial; single-scale (the default) carries none.
/// MUST agree with the penalty-coordinate layout of
/// `build_measure_jet_basis_psi_derivatives` (its `per_level` branch always
/// emits exactly the (α, lnτ) coordinate pair).
pub fn measure_jet_penalty_psi_dim(mj: &crate::basis::MeasureJetBasisSpec) -> usize {
    if crate::basis::measure_jet_multiscale_mode(mj) {
        2
    } else {
        0
    }
}

/// ψ dimension of a measure-jet term. The design-moving ℓ dial (when enrolled)
/// is coordinate 0; the multiscale penalty dials follow. MUST agree with the
/// coordinate layout of `build_measure_jet_basis_psi_derivatives` (ℓ first).
pub fn measure_jet_psi_dim(mj: &crate::basis::MeasureJetBasisSpec) -> usize {
    usize::from(measure_jet_learns_length_scale(mj)) + measure_jet_penalty_psi_dim(mj)
}

/// Seed ψ from the term's realized dials, in producer coordinate order: ℓ first
/// (when enrolled), then the multiscale penalty dials. The ℓ seed is the
/// realized representer range `ln(length_scale)` (the resolved spec carries the
/// concrete auto value after the design build/freeze).
pub fn measure_jet_psi_seed(mj: &crate::basis::MeasureJetBasisSpec) -> Vec<f64> {
    let mut seed = Vec::with_capacity(measure_jet_psi_dim(mj));
    if measure_jet_learns_length_scale(mj) {
        // length_scale > 0 after resolution; the 0.0 sentinel (pre-resolution)
        // falls back to the centre of the log-ℓ box so the optimizer still
        // starts feasible and the first data-aware reseed corrects it.
        let ell = if mj.length_scale > 0.0 {
            mj.length_scale
        } else {
            1.0
        };
        seed.push(ell.ln());
    }
    if measure_jet_penalty_psi_dim(mj) > 0 {
        // Multiscale penalty dials, producer order: (α, lnτ).
        let ln_tau = mj.tau0.max(f64::MIN_POSITIVE).ln();
        seed.extend_from_slice(&[mj.alpha, ln_tau]);
    }
    seed
}

/// One end of the per-coordinate dial boxes, in producer coordinate order
/// (ℓ first when enrolled, then the multiscale penalty dials).
///
/// The two PENALTY dials are dimensionless — `α` selects a density
/// normalization exponent and `ln τ` a ridge on the local projection — so
/// nothing in the data's geometry bounds them and their boxes are the fixed
/// intervals above. The design-moving `ln ℓ` dial is the opposite case: it is a
/// LENGTH in the chart the basis is realized in, and its window is the term's
/// own [`crate::basis::measure_jet_ln_range_window`] — the node-spacing floor
/// and the node-diameter ceiling the range bracket already derives (gam#2750).
/// The window is WIDENED, never narrowed, to contain the incumbent range, the
/// same feasible-set rule [`spatial_term_psi_search_box`] applies to the other
/// spatial families (#2454): a box that excludes the incumbent turns a
/// monotonicity contract into a contradiction.
pub fn measure_jet_psi_bound_values(
    data: ArrayView2<'_, f64>,
    term: &SmoothBasisSpec,
    upper: bool,
) -> Result<Vec<f64>, BasisError> {
    let SmoothBasisSpec::MeasureJet {
        feature_cols,
        spec: mj,
        input_scale,
    } = term
    else {
        crate::bail_invalid_basis!(
            "measure-jet ψ bounds requested for a {} term",
            term.structural_kind()
        );
    };
    let pick = |b: (f64, f64)| if upper { b.1 } else { b.0 };
    let mut bounds = Vec::with_capacity(measure_jet_psi_dim(mj));
    if measure_jet_learns_length_scale(mj) {
        let mut columns = select_columns(data, feature_cols)?;
        // A term that carries an input scale is realized in the standardized
        // frame, and so is the `length_scale` the ψ seed reads; convert the
        // view before measuring lengths in it.
        if let Some(scale) = input_scale {
            scale.standardize(&mut columns);
        }
        let (mut lo, mut hi) = crate::basis::measure_jet_ln_range_window(columns.view(), mj)?;
        if mj.length_scale > 0.0 {
            let incumbent = mj.length_scale.ln();
            if incumbent.is_finite() {
                lo = lo.min(incumbent);
                hi = hi.max(incumbent);
            }
        }
        bounds.push(if upper { hi } else { lo });
    }
    if measure_jet_penalty_psi_dim(mj) > 0 {
        // Multiscale penalty dials, producer order: (α, lnτ).
        bounds.push(pick(MEASURE_JET_PSI_ALPHA_BOUNDS));
        bounds.push(pick(MEASURE_JET_PSI_LN_TAU_BOUNDS));
    }
    Ok(bounds)
}

/// Write optimized ψ dials back into a measure-jet spec. Returns `true` when
/// any dial actually moved. The geometry (centers, masses, band, ℓ, z) is
/// ψ-FIXED by contract — only the dials change, so frozen-quadrature
/// rebuilds reproduce the identical penalty layout at the new dials.
pub fn apply_measure_jet_psi(
    mj: &mut crate::basis::MeasureJetBasisSpec,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    if psi.len() != measure_jet_psi_dim(mj) {
        crate::bail_invalid_estim!(
            "measure-jet ψ write-back dimension mismatch: got {} values for a {}-dial term",
            psi.len(),
            measure_jet_psi_dim(mj)
        );
    }
    let mut changed = false;
    // Coordinate 0 (when enrolled) is the design-moving ln(ℓ); the multiscale
    // penalty dials follow. Same order as `measure_jet_psi_seed` and the
    // producer (`build_measure_jet_basis_psi_derivatives`).
    let mut cursor = 0usize;
    if measure_jet_learns_length_scale(mj) {
        let next_ell = psi[cursor].exp();
        cursor += 1;
        if !(next_ell.is_finite() && next_ell > 0.0) {
            crate::bail_invalid_estim!(
                "measure-jet ψ write-back produced a non-finite/non-positive length_scale (ℓ={next_ell})"
            );
        }
        if next_ell != mj.length_scale {
            mj.length_scale = next_ell;
            changed = true;
        }
    }
    if measure_jet_penalty_psi_dim(mj) > 0 {
        // Multiscale penalty dials, producer order: (α, lnτ). The order `s` is
        // not a dial (pinned explicit or absorbed by the per-scale amplitudes).
        let next_alpha = psi[cursor];
        let next_tau = psi[cursor + 1].exp();
        if !(next_alpha.is_finite() && next_tau.is_finite() && next_tau > 0.0) {
            crate::bail_invalid_estim!(
                "measure-jet ψ write-back produced non-finite dials (alpha={next_alpha}, tau={next_tau})"
            );
        }
        if next_alpha != mj.alpha {
            mj.alpha = next_alpha;
            changed = true;
        }
        if next_tau != mj.tau0 {
            mj.tau0 = next_tau;
            changed = true;
        }
    }
    Ok(changed)
}

/// Collection-level measure-jet dial write-back (the `apply_tospec` /
/// realizer-side entry). Returns whether anything moved.
pub fn set_measure_jet_psi_dials(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        crate::bail_invalid_estim!("measure-jet ψ write-back: term index {term_idx} out of range");
    };
    set_single_term_measure_jet_psi_dials(term, psi)
}

/// Single-term dial write-back: the shared match+apply core, also used
/// directly on the cached per-trial build spec (whose caller has already
/// change-checked at the collection level and rebuilds regardless of the
/// moved flag).
pub fn set_single_term_measure_jet_psi_dials(
    term: &mut SmoothTermSpec,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &mut term.basis else {
        crate::bail_invalid_estim!("measure-jet ψ write-back targeted a non-measure-jet term");
    };
    apply_measure_jet_psi(mj, psi)
}
