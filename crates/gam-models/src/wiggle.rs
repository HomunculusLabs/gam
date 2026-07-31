use crate::parameter_block::ParameterBlockInput;
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_solve::pirls::LinearInequalityConstraints;
use gam_terms::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_ispline_derivative_dense,
    ispline_function_penalties,
};
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct WiggleBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub double_penalty: bool,
}

/// Semantic identity of one canonical I-spline penalty block.
///
/// The order of these values is the smoothing-parameter order. Persisting the
/// topology prevents inference code from guessing a derivative order from a
/// lambda index or inventing a zero block when the guess is invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum WigglePenaltyBlockKind {
    Roughness { derivative_order: usize },
    NullspaceShrinkage { derivative_order: usize },
}

/// Complete semantic description of a realized monotone-wiggle penalty list.
///
/// `derivative_orders` is already canonicalized into the exact roughness-block
/// order used by fitting: primary first, followed by deduplicated additional
/// orders. `blocks` additionally records whether the primary roughness emitted
/// a function-metric nullspace shrinkage coordinate. For example, an order-one
/// anchored I-spline roughness is full rank, so `double_penalty=true` emits no
/// synthetic ridge block.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WigglePenaltyMetadata {
    pub derivative_orders: Vec<usize>,
    pub double_penalty: bool,
    pub blocks: Vec<WigglePenaltyBlockKind>,
}

/// Exact matrices and nullities accompanying [`WigglePenaltyMetadata`].
#[derive(Clone, Debug)]
pub struct CanonicalWigglePenaltySet {
    pub metadata: WigglePenaltyMetadata,
    pub matrices: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct SelectedWiggleBasis {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub block: ParameterBlockInput,
    pub penalty_metadata: WigglePenaltyMetadata,
}

// #1521: relocated DOWN into `gam_terms::basis` (was a gamlss/wiggle helper).
// The knot-generation primitive carries no model-family type, so family modules
// and this module's own callers consume it from the basis layer via this
// re-export — keeping every `crate::wiggle::initializewiggle_knots_from_seed`
// call site (gamlss / bms / transformation-normal) resolving unchanged.
pub(crate) use gam_terms::basis::initializewiggle_knots_from_seed;

#[inline]
pub(crate) fn monotone_wiggle_internal_degree(degree: usize) -> Result<usize, String> {
    // Public monotone-wiggle degree refers to the value basis. The low-level
    // I-spline builder integrates a degree-`internal_degree` specification
    // into a degree-`internal_degree + 1` value basis, so we subtract one here
    // to keep the public degree and the per-span value degree aligned.
    degree
        .checked_sub(1)
        .filter(|&internal_degree| internal_degree >= 1)
        .ok_or_else(|| "monotone wiggle degree must be >= 2".to_string())
}

/// Build the exact ordered function-space penalty set for an anchored
/// I-spline monotone wiggle.
///
/// `derivative_orders` must already be in the fitting order and contain no
/// duplicates. The first order is the primary roughness; only its structural
/// null space is eligible for the separate double-penalty coordinate. Every
/// matrix comes from the canonical `C^T S_B C` function Gram, never a
/// coefficient difference or identity metric.
///
/// # The assembled set always has a trivial joint null space (gam#2647)
///
/// A monotone link warp is not an ordinary smooth. It is composed onto a free
/// index — `q = q₀ + w(q₀)` with `q₀ = −η_t·e^{−η_ls}` (binomial location-scale)
/// or `q = η + w(η)` (binomial mean) — and the index carries its own free
/// scale. That makes the warp's LINEAR direction a gauge rather than a shape:
/// for any `s > 0`,
///
/// ```text
///   (β_index, β_w)  ↦  (β_index / s,  β_w + (s−1)·ℓ),      B·ℓ = (u − left)
/// ```
///
/// reproduces the same `q` on every row inside the knot hull, because
/// `q₀/s + (s−1)(q₀/s − left) + w(q₀/s)` is `q₀` up to the constant the warp can
/// itself absorb. The likelihood is exactly invariant along that orbit. The
/// PENALTY is not: the index block is penalized, so `½βᵀSβ` falls like `1/s²`
/// there. If `ℓ` is unpenalized the penalized objective therefore decreases
/// monotonically in `s` and has **no minimiser** — the inner solve walks the
/// orbit forever, `‖β‖∞` diverging while `½βᵀSβ` falls exactly like `‖β‖⁻²`,
/// which is the measured signature on gam#2647.
///
/// An order-`k` anchored I-spline roughness has structural nullity `k − 1`, so
/// every set whose smallest order exceeds one leaves `ℓ` free unless something
/// closes it. `double_penalty` is a user knob and cannot be the thing that
/// decides whether the criterion is bounded below, so the closure below is
/// unconditional: after assembling the requested roughness blocks, the JOINT
/// null space of the set (at unit smoothing) is computed in the function metric
/// and, when non-trivial, one shrinkage coordinate spanning it is appended.
///
/// This is the same treatment — and the same argument — the binomial
/// location-scale log-σ block already receives unconditionally in
/// `build_binomial_threshold_and_scale_blocks`, where `(β_t, β_ls) ↦ (c·β_t,
/// β_ls + ln c)` is the exactly analogous index-scale gauge and an identity
/// shrinkage penalty is appended that the caller never asked for and cannot
/// switch off. The wiggle block simply never received it.
///
/// It is a **no-op on every already-well-posed configuration**: the shipped
/// default (`orders = [1, 2, 3]`) contains the order-one roughness, which is
/// full rank on the anchored basis, so the joint null space is already trivial
/// and nothing is appended; likewise whenever `double_penalty` already closed a
/// single-order set. Only configurations whose criterion is otherwise unbounded
/// gain a coordinate, and that coordinate's strength is chosen by REML like any
/// other.
pub fn canonical_wiggle_function_penalties(
    knots: &Array1<f64>,
    degree: usize,
    derivative_orders: &[usize],
    double_penalty: bool,
) -> Result<CanonicalWigglePenaltySet, String> {
    if derivative_orders.is_empty() {
        return Err("wiggle penalty metadata requires at least one derivative order".to_string());
    }
    if derivative_orders.contains(&0) {
        return Err("wiggle penalty derivative orders must all be positive".to_string());
    }
    for (index, &order) in derivative_orders.iter().enumerate() {
        if derivative_orders[..index].contains(&order) {
            return Err(format!(
                "wiggle penalty derivative order {order} is duplicated in canonical metadata"
            ));
        }
    }

    let internal_degree = monotone_wiggle_internal_degree(degree)?;
    let mut blocks = Vec::new();
    let mut matrices = Vec::new();
    let mut nullspace_dims = Vec::new();
    for (index, &derivative_order) in derivative_orders.iter().enumerate() {
        let penalties = ispline_function_penalties(
            knots.view(),
            internal_degree,
            derivative_order,
            index == 0 && double_penalty,
        )
        .map_err(|error| error.to_string())?;
        blocks.push(WigglePenaltyBlockKind::Roughness { derivative_order });
        matrices.push(penalties.roughness);
        nullspace_dims.push(penalties.roughness_nullspace_dim);
        if let Some(nullspace_shrinkage) = penalties.nullspace_shrinkage {
            blocks.push(WigglePenaltyBlockKind::NullspaceShrinkage { derivative_order });
            matrices.push(nullspace_shrinkage);
            nullspace_dims.push(0);
        }
    }

    // Gauge closure (gam#2647) — see the type-level note above. The joint null
    // space is read off the SUM of the assembled blocks at unit smoothing, which
    // is `null(Σ S_j) = ⋂_j null(S_j)` exactly because every `S_j` is PSD, so a
    // direction survives only when NO requested block penalizes it. Reading the
    // sum (rather than the primary alone) is what makes this both complete —
    // a multi-order set is judged by what it collectively leaves free — and
    // idempotent: when `double_penalty` already emitted a shrinkage coordinate,
    // that coordinate is inside the sum, the intersection is empty, and nothing
    // is appended. Tagged with the PRIMARY derivative order, so a single-order
    // set with `double_penalty = true` and one with `double_penalty = false`
    // produce the same topology, which is the point.
    //
    // Each block enters the sum divided by its OWN mean diagonal. Without that
    // the test would not be "does any block penalize this direction" but "does
    // any block penalize it comparably to the stiffest block present": an
    // order-3 roughness has eigenvalues orders above an order-1 roughness on the
    // same knots, so on the shipped default (`orders = [1, 2, 3]`) the order-1
    // block — which is the one that closes the gauge — would sit inside the
    // rank tolerance of the order-3 block and the sum would report a null space
    // that does not exist, appending a coordinate to a configuration that never
    // needed one. A per-block scale is the only thing that makes the
    // intersection `⋂_j null(S_j)` the quantity actually computed, and it is
    // derived from the matrices rather than chosen.
    let primary_order = derivative_orders[0];
    let joint_dim = matrices.first().map_or(0, |m| m.nrows());
    if joint_dim > 0 {
        let mut joint = Array2::<f64>::zeros((joint_dim, joint_dim));
        for matrix in &matrices {
            let mean_diagonal =
                (0..joint_dim).map(|i| matrix[[i, i]].abs()).sum::<f64>() / joint_dim as f64;
            if !(mean_diagonal > 0.0) || !mean_diagonal.is_finite() {
                continue;
            }
            joint.scaled_add(1.0 / mean_diagonal, matrix);
        }
        let function_gram =
            gam_terms::basis::ispline_function_gram(knots.view(), internal_degree)
                .map_err(|error| error.to_string())?;
        if let Some(gauge_shrinkage) =
            gam_terms::basis::function_space_nullspace_shrinkage(&joint, &function_gram)
                .map_err(|error| error.to_string())?
        {
            blocks.push(WigglePenaltyBlockKind::NullspaceShrinkage {
                derivative_order: primary_order,
            });
            matrices.push(gauge_shrinkage);
            nullspace_dims.push(0);
        }
    }

    Ok(CanonicalWigglePenaltySet {
        metadata: WigglePenaltyMetadata {
            derivative_orders: derivative_orders.to_vec(),
            double_penalty,
            blocks,
        },
        matrices,
        nullspace_dims,
    })
}

fn buildwiggle_block_input_from_canonical_penalties(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    canonical: &CanonicalWigglePenaltySet,
) -> Result<ParameterBlockInput, String> {
    let design = monotone_wiggle_basis_from_knots(seed, knots, degree)?;
    let p = design.ncols();
    if p == 0 {
        return Err("wiggle basis has no free monotone columns".to_string());
    }
    if canonical.matrices.len() != canonical.nullspace_dims.len()
        || canonical.matrices.len() != canonical.metadata.blocks.len()
    {
        return Err(
            "canonical wiggle penalty matrices, nullities, and topology disagree".to_string(),
        );
    }
    for (index, matrix) in canonical.matrices.iter().enumerate() {
        if matrix.dim() != (p, p) {
            return Err(format!(
                "canonical I-spline penalty block {index} is {}x{} but wiggle design has {p} columns",
                matrix.nrows(),
                matrix.ncols(),
            ));
        }
    }
    Ok(ParameterBlockInput {
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::zeros(seed.len()),
        penalties: canonical
            .matrices
            .iter()
            .cloned()
            .map(crate::model_types::PenaltySpec::Dense)
            .collect(),
        nullspace_dims: canonical.nullspace_dims.clone(),
        initial_log_lambdas: None,
        initial_beta: Some(Array1::zeros(p)),
    })
}

pub fn buildwiggle_block_input_from_knots(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    penalty_order: usize,
    double_penalty: bool,
) -> Result<ParameterBlockInput, String> {
    buildwiggle_block_input_from_orders(seed, knots, degree, &[penalty_order], double_penalty)
}

/// Build a monotone I-spline block carrying the COMPLETE requested penalty set.
///
/// Callers that want several derivative orders must come through here rather
/// than building a primary-order block and appending the rest: the gauge closure
/// in [`canonical_wiggle_function_penalties`] is a property of the assembled set
/// (it asks what the set collectively leaves unpenalized), so it can only be
/// decided once, on the final list. Assembling in two stages would judge the
/// primary order alone and could both add a coordinate the later orders made
/// unnecessary and miss one they left open.
///
/// The emitted order is unchanged from the previous two-stage assembly —
/// primary roughness, its optional double-penalty coordinate, then the extra
/// orders in the order given — so persisted penalty topologies are unaffected.
pub fn buildwiggle_block_input_from_orders(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    derivative_orders: &[usize],
    double_penalty: bool,
) -> Result<ParameterBlockInput, String> {
    let canonical =
        canonical_wiggle_function_penalties(knots, degree, derivative_orders, double_penalty)?;
    buildwiggle_block_input_from_canonical_penalties(seed, knots, degree, &canonical)
}

pub fn buildwiggle_block_input_from_seed(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
) -> Result<(ParameterBlockInput, Array1<f64>), String> {
    let knots = initializewiggle_knots_from_seed(seed, cfg.degree, cfg.num_internal_knots)?;
    let block = buildwiggle_block_input_from_knots(
        seed,
        &knots,
        cfg.degree,
        cfg.penalty_order,
        cfg.double_penalty,
    )?;
    Ok((block, knots))
}

pub(crate) fn monotone_wiggle_basis_from_knots(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, String> {
    let internal_degree = monotone_wiggle_internal_degree(degree)?;
    let (basis, _) = create_basis::<Dense>(
        seed,
        KnotSource::Provided(knots.view()),
        internal_degree,
        BasisOptions::i_spline(),
    )
    .map_err(|e| e.to_string())?;
    Ok(basis.as_ref().clone())
}

pub fn monotone_wiggle_basis_with_derivative_order(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    derivative_order: usize,
) -> Result<Array2<f64>, String> {
    if derivative_order == 0 {
        return monotone_wiggle_basis_from_knots(seed, knots, degree);
    }
    let internal_degree = monotone_wiggle_internal_degree(degree)?;
    create_ispline_derivative_dense(seed, knots, internal_degree, derivative_order)
        .map_err(|e| e.to_string())
}

pub(crate) fn monotone_wiggle_nonnegative_constraints(
    beta_dim: usize,
) -> Option<gam_solve::pirls::ConstraintSet> {
    if beta_dim == 0 {
        return None;
    }
    let mut a = Array2::<f64>::zeros((beta_dim, beta_dim));
    for i in 0..beta_dim {
        a[[i, i]] = 1.0;
    }
    Some(gam_solve::pirls::ConstraintSet::Dense(
        LinearInequalityConstraints {
            a,
            b: Array1::zeros(beta_dim),
        },
    ))
}

pub(crate) fn validate_monotone_wiggle_beta_nonnegative<'a>(
    beta: impl IntoIterator<Item = &'a f64>,
    context: &str,
) -> Result<(), String> {
    for (idx, &value) in beta.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{context} coefficient {idx} is non-finite"));
        }
        if value < -1e-12 {
            return Err(format!(
                "{context} coefficient {idx} is negative ({value:.3e}); monotone wiggle coefficients must be non-negative"
            ));
        }
    }
    Ok(())
}

/// Slack tolerance for the `beta >= 0` monotone-wiggle inequality constraints.
///
/// The constrained inner Newton/QP holds a binding coordinate at the boundary
/// only up to its own KKT tolerance, so an accepted step can leave the active
/// coordinate a few ULPs below zero (e.g. `-2e-9`). That is feasibility within
/// the solver tolerance, not a genuine sign violation, so the post-update hook
/// projects such coordinates back onto the non-negative cone (clamps them to
/// exactly `0`) rather than failing the fit. The band matches the constrained
/// blockwise solver's KKT tolerances (`1e-6 * scale + 1e-10`,
/// `1e-10 * (1 + scale)`); anything more negative survives the projection and
/// is rejected by [`validate_monotone_wiggle_beta_nonnegative`].
pub(crate) const MONOTONE_WIGGLE_ACTIVE_SET_TOL: f64 = 1e-6;

/// Project a monotone-wiggle coefficient vector onto the non-negative cone the
/// `beta >= 0` constraints define, clamping coordinates the constrained solve
/// left slightly negative (within [`MONOTONE_WIGGLE_ACTIVE_SET_TOL`]) to exactly
/// `0`. Coordinates more negative than the tolerance are left untouched so the
/// subsequent [`validate_monotone_wiggle_beta_nonnegative`] still rejects
/// genuine sign violations.
pub(crate) fn project_monotone_wiggle_beta_nonnegative(mut beta: Array1<f64>) -> Array1<f64> {
    for value in beta.iter_mut() {
        if *value < 0.0 && *value >= -MONOTONE_WIGGLE_ACTIVE_SET_TOL {
            *value = 0.0;
        }
    }
    beta
}

/// Resolve a requested wiggle penalty-order set into:
///
/// - the primary derivative order used by the monotone I-spline function
///   roughness, and
/// - the remaining function-derivative orders to append on the same basis.
///
/// The primary order is the smallest requested order. If the list is empty,
/// `default_primary` is used. Zero is never silently dropped: it is not a
/// roughness derivative and is therefore a typed configuration error. Extra
/// orders are returned in original order, deduplicated, and exclude primary.
pub fn split_wiggle_penalty_orders(
    default_primary: usize,
    penalty_orders: &[usize],
) -> Result<(usize, Vec<usize>), String> {
    if default_primary == 0 {
        return Err("default wiggle penalty derivative order must be positive".to_string());
    }
    if penalty_orders.contains(&0) {
        return Err("wiggle penalty derivative orders must all be positive".to_string());
    }
    let primary_order = penalty_orders
        .iter()
        .copied()
        .min()
        .unwrap_or(default_primary);
    let mut extras = Vec::new();
    for &order in penalty_orders {
        if order == primary_order || extras.contains(&order) {
            continue;
        }
        extras.push(order);
    }
    Ok((primary_order, extras))
}

pub(crate) fn select_wiggle_basis_from_seed(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
    penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let (primary_order, extra_orders) =
        split_wiggle_penalty_orders(cfg.penalty_order, penalty_orders)?;
    let mut derivative_orders = Vec::with_capacity(1 + extra_orders.len());
    derivative_orders.push(primary_order);
    derivative_orders.extend(extra_orders);
    let knots = initializewiggle_knots_from_seed(seed, cfg.degree, cfg.num_internal_knots)?;
    let canonical = canonical_wiggle_function_penalties(
        &knots,
        cfg.degree,
        &derivative_orders,
        cfg.double_penalty,
    )?;
    let block =
        buildwiggle_block_input_from_canonical_penalties(seed, &knots, cfg.degree, &canonical)?;
    Ok(SelectedWiggleBasis {
        knots,
        degree: cfg.degree,
        block,
        penalty_metadata: canonical.metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_types::PenaltySpec;
    use ndarray::Array1;

    fn dense_penalty(spec: &PenaltySpec) -> &Array2<f64> {
        match spec {
            PenaltySpec::Dense(m) => m,
            other => panic!("expected Dense penalty, got {other:?}"),
        }
    }

    fn is_symmetric(m: &Array2<f64>) -> bool {
        let n = m.nrows();
        if m.ncols() != n {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                if (m[[i, j]] - m[[j, i]]).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    // ---- monotone_wiggle_internal_degree ----

    #[test]
    fn internal_degree_rejects_degree_below_two() {
        // degree 0 and 1 yield internal_degree < 1 -> Err with the documented message.
        for d in [0usize, 1] {
            let err = monotone_wiggle_internal_degree(d).unwrap_err();
            assert_eq!(err, "monotone wiggle degree must be >= 2");
        }
    }

    #[test]
    fn internal_degree_is_degree_minus_one_for_valid_degrees() {
        // degree >= 2 -> Ok(degree - 1): the per-span value degree aligned to the
        // public value-basis degree.
        assert_eq!(monotone_wiggle_internal_degree(2).unwrap(), 1);
        assert_eq!(monotone_wiggle_internal_degree(3).unwrap(), 2);
        assert_eq!(monotone_wiggle_internal_degree(10).unwrap(), 9);
    }

    // ---- buildwiggle_block_input_from_knots (driven via seed for valid knots) ----

    fn build(double_penalty: bool, penalty_order: usize) -> (ParameterBlockInput, usize) {
        // A spread-out seed so knot generation yields several monotone columns.
        let seed = Array1::linspace(0.0, 1.0, 40);
        let cfg = WiggleBlockConfig {
            degree: 3,
            num_internal_knots: 5,
            penalty_order,
            double_penalty,
        };
        let knots =
            initializewiggle_knots_from_seed(seed.view(), cfg.degree, cfg.num_internal_knots)
                .expect("knot init");
        let block = buildwiggle_block_input_from_knots(
            seed.view(),
            &knots,
            cfg.degree,
            cfg.penalty_order,
            cfg.double_penalty,
        )
        .expect("build block");
        let p = block.design.ncols();
        (block, p)
    }

    #[test]
    fn single_penalty_block_shapes_and_invariants() {
        let (block, p) = build(false, 2);
        assert!(p >= 2, "expected multiple monotone columns, got p={p}");
        // Offset is zeros with length = seed length.
        assert_eq!(block.offset.len(), 40);
        assert!(block.offset.iter().all(|&v| v == 0.0));
        // initial_beta is Some(zeros(p)).
        let beta = block.initial_beta.as_ref().expect("initial_beta");
        assert_eq!(beta.len(), p);
        assert!(beta.iter().all(|&v| v == 0.0));
        // One ROUGHNESS penalty, plus the unconditional gauge-closure
        // coordinate (gam#2647): an order-two roughness leaves the linear warp
        // free, and the linear warp is the index scale, not a shape. This
        // assertion used to read `== 1`, which is exactly the shape of the
        // defect — a warp block shipped with an unpenalized reparameterization
        // direction, so the penalized criterion had no minimiser.
        assert_eq!(block.penalties.len(), 2);
        assert_eq!(block.nullspace_dims.len(), 2);
        // The exact function-derivative Gram is p x p and symmetric.
        let s = dense_penalty(&block.penalties[0]);
        assert_eq!(s.dim(), (p, p));
        assert!(is_symmetric(s));
        // The anchored I-spline excludes the constant polynomial, so the
        // order-two derivative null space contains only the linear direction.
        assert_eq!(block.nullspace_dims[0], 1);
        // The closure coordinate penalizes a null space of its own dimension 0.
        assert_eq!(block.nullspace_dims[1], 0);
    }

    /// Smallest generalized eigenvalue of `Σ_j S_j` against the I-spline
    /// function Gram, relative to the largest — i.e. how close the assembled
    /// penalty set comes to leaving a whole function direction free.
    fn relative_joint_nullity_margin(
        knots: &Array1<f64>,
        degree: usize,
        orders: &[usize],
        double_penalty: bool,
    ) -> f64 {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;

        let canonical = canonical_wiggle_function_penalties(knots, degree, orders, double_penalty)
            .expect("canonical wiggle penalties");
        let dim = canonical.matrices[0].nrows();
        let mut joint = Array2::<f64>::zeros((dim, dim));
        for matrix in &canonical.matrices {
            joint += matrix;
        }
        let internal_degree = monotone_wiggle_internal_degree(degree).expect("internal degree");
        let gram = gam_terms::basis::ispline_function_gram(knots.view(), internal_degree)
            .expect("I-spline function Gram");
        // Whiten by the function metric so the ratio is a statement about
        // FUNCTIONS, not about the coefficient chart: `G^{-1/2} S G^{-1/2}`.
        let (gvals, gvecs) = gram.eigh(Side::Lower).expect("Gram eigh");
        let gmax = gvals.iter().copied().fold(0.0_f64, f64::max);
        let mut g_inv_sqrt = Array2::<f64>::zeros((dim, dim));
        for k in 0..dim {
            let lam = gvals[k].max(1e-14 * gmax);
            let scale = 1.0 / lam.sqrt();
            let vk = gvecs.column(k);
            for i in 0..dim {
                for j in 0..dim {
                    g_inv_sqrt[[i, j]] += scale * vk[i] * vk[j];
                }
            }
        }
        let whitened = g_inv_sqrt.dot(&joint).dot(&g_inv_sqrt);
        let (svals, _) = whitened.eigh(Side::Lower).expect("whitened penalty eigh");
        let hi = svals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lo = svals.iter().copied().fold(f64::INFINITY, f64::min);
        lo / hi.max(f64::MIN_POSITIVE)
    }

    /// gam#2647, stated as the invariant rather than as one fixture.
    ///
    /// A monotone link warp is composed onto a free index, so any function
    /// direction the assembled penalty set leaves unpenalized is a
    /// reparameterization the index can absorb for free — the penalized
    /// criterion then has no minimiser and the inner solve diverges along the
    /// orbit. The invariant is therefore: **for every configuration this crate
    /// can emit, the assembled set has a trivial joint null space.** Checked in
    /// the function metric so the verdict cannot be manufactured by a coefficient
    /// rescale.
    ///
    /// Before the fix, every `double_penalty = false` row here (and every
    /// `[2]`/`[2,3]` row regardless) had a joint null space of dimension ≥ 1,
    /// i.e. a margin at machine zero.
    #[test]
    fn wiggle_penalty_set_always_closes_its_own_null_space_2647() {
        let seed = Array1::linspace(0.0, 1.0, 60);
        for degree in [2usize, 3, 4] {
            let knots = initializewiggle_knots_from_seed(seed.view(), degree, 5)
                .expect("knot init for the invariant sweep");
            let max_order = degree; // value degree = degree, so order <= degree is represented
            let mut order_sets: Vec<Vec<usize>> = Vec::new();
            for primary in 1..=max_order {
                order_sets.push(vec![primary]);
            }
            if max_order >= 2 {
                order_sets.push((1..=max_order).collect());
                order_sets.push((2..=max_order).collect());
            }
            for orders in &order_sets {
                for double_penalty in [false, true] {
                    let margin =
                        relative_joint_nullity_margin(&knots, degree, orders, double_penalty);
                    assert!(
                        margin > 1e-10,
                        "degree {degree}, orders {orders:?}, double_penalty={double_penalty}: \
                         the assembled wiggle penalty set leaves a function direction free \
                         (smallest/largest whitened penalty eigenvalue = {margin:.6e}). That \
                         direction is a reparameterization of the index the warp is composed \
                         onto, so the penalized criterion is unbounded below along it (gam#2647)."
                    );
                }
            }
        }
    }

    /// The concrete gauge, named: an order-two roughness leaves the LINEAR warp
    /// free, and the linear warp is exactly the index rescale
    /// `(β_index, β_w) ↦ (β_index/s, β_w + (s−1)ℓ)`. The closure must charge for
    /// it while the roughness alone does not.
    #[test]
    fn linear_warp_direction_is_free_under_roughness_and_charged_after_closure_2647() {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;

        let seed = Array1::linspace(0.0, 1.0, 60);
        let degree = 3usize;
        let knots = initializewiggle_knots_from_seed(seed.view(), degree, 5).expect("knot init");
        let canonical = canonical_wiggle_function_penalties(&knots, degree, &[2], false)
            .expect("order-two canonical set");
        assert_eq!(
            canonical.matrices.len(),
            2,
            "order-two roughness must be accompanied by its gauge closure"
        );
        let roughness = &canonical.matrices[0];
        let closure = &canonical.matrices[1];
        let dim = roughness.nrows();

        // ℓ: the coefficient vector of the linear warp, recovered as the
        // roughness null direction (the anchored basis excludes constants, so
        // the order-two null space is exactly the linear ramp).
        let (rvals, rvecs) = roughness.eigh(Side::Lower).expect("roughness eigh");
        let rmax = rvals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut k_min = 0usize;
        for k in 0..dim {
            if rvals[k] < rvals[k_min] {
                k_min = k;
            }
        }
        let ell = rvecs.column(k_min).to_owned();
        let rough_energy = ell.dot(&roughness.dot(&ell));
        let closure_energy = ell.dot(&closure.dot(&ell));
        assert!(
            rough_energy <= 1e-10 * rmax,
            "the linear warp must be free under an order-two roughness: ℓᵀSℓ = {rough_energy:.6e} \
             against λ_max = {rmax:.6e}"
        );
        assert!(
            closure_energy > 1e-8 * rmax.max(1.0),
            "the gauge closure must charge for the linear warp: ℓᵀRℓ = {closure_energy:.6e}"
        );
        // And it must charge ONLY for it: the closure is a second REML
        // coordinate on the null space, never a re-penalization of curvature.
        let range_energy = (0..dim)
            .filter(|&k| rvals[k] > 1e-8 * rmax)
            .map(|k| {
                let v = rvecs.column(k);
                v.dot(&closure.dot(&v)).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            range_energy <= 1e-8 * closure_energy,
            "the closure leaked onto the roughness range: max range energy {range_energy:.6e} \
             against null energy {closure_energy:.6e}"
        );
    }

    #[test]
    fn double_penalty_appends_nullspace_only_function_ridge() {
        let (block, p) = build(true, 2);
        assert!(p >= 2);
        // Order two has one structural null direction, so double penalty emits
        // one separate function-space shrinkage block.
        assert_eq!(block.penalties.len(), 2);
        assert_eq!(block.nullspace_dims.len(), 2);
        let ridge = dense_penalty(&block.penalties[1]);
        assert_eq!(ridge.dim(), (p, p));
        assert!(is_symmetric(ridge));
        assert!(
            (0..p).any(|i| (0..p).any(|j| i != j && ridge[[i, j]].abs() > 1e-12)),
            "function-metric null shrinkage must not collapse to eye(p)"
        );
        assert_eq!(block.nullspace_dims[1], 0);
    }

    #[test]
    fn order_one_has_no_nullspace_ridge() {
        let (block, _) = build(true, 1);
        assert_eq!(block.penalties.len(), 1);
        assert_eq!(block.nullspace_dims, vec![0]);
    }

    #[test]
    fn unsupported_derivative_order_is_rejected_not_clamped() {
        let seed = Array1::linspace(0.0, 1.0, 40);
        let knots = initializewiggle_knots_from_seed(seed.view(), 3, 5).expect("knot init");
        let error = match buildwiggle_block_input_from_knots(seed.view(), &knots, 3, 4, false) {
            Ok(_) => panic!("order above represented value degree must be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("derivative"), "unexpected error: {error}");
    }

    #[test]
    fn explicit_zero_penalty_order_is_rejected() {
        let error = split_wiggle_penalty_orders(2, &[0, 2]).unwrap_err();
        assert_eq!(
            error,
            "wiggle penalty derivative orders must all be positive"
        );
    }
}
