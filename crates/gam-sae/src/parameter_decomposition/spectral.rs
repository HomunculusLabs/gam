//! Spectral recovery of rotation planes from a frozen matrix (#2951 P3).
//!
//! # The result
//!
//! A rotation acting in orthogonal planes is
//!
//! ```text
//! W = I + sum_k U_k (R_{alpha_k} - I) U_k^T,     U_k^T U_l = delta_kl I_2.
//! ```
//!
//! For any orthogonal `W` the symmetric part `S = (W + W^T)/2` has the plane
//! projectors `P_k = U_k U_k^T` as eigenspaces with eigenvalue `cos alpha_k`, and
//! the skew part `K = (W - W^T)/2` satisfies `P_k K P_k = sin alpha_k J_k` with
//! `J_k^2 = -P_k`. So
//!
//! ```text
//! W = I + sum_k [(cos alpha_k - 1) P_k + sin alpha_k J_k]
//! ```
//!
//! is read off one symmetric eigendecomposition and one compression of `K` per
//! eigenspace. Eigenvalue `+1` is the fixed space and `-1` the half-turn space;
//! every other eigenvalue has even multiplicity.
//!
//! # What the matrix does not determine (reported, never guessed)
//!
//! * **Repeated cosines.** An eigenspace of dimension `2p`, `p >= 2`, carries its
//!   projector and its complex structure `J`, but every `J`-invariant split into
//!   planes is equally valid, so the individual planes are not identified.
//! * **Half-turn.** At `cos alpha = -1`, `sin alpha = 0` and `J` is undefined: a
//!   2-dimensional `-1` space is a plane with no orientation. An odd-dimensional
//!   `-1` space contains a reflection.
//! * **Identity.** No plane at all. Only rotations below a derived angle can hide.
//! * **Winding.** `alpha` and `alpha + 2 pi k` give the same matrix, as do
//!   `(alpha, J)` and `(-alpha, -J)`. The reported angle is the representative in
//!   `(0, pi)` with the orientation carried by `J`; choosing the shortest path is
//!   a convention.
//!
//! # Grouping is derived, not a constant
//!
//! A supplied matrix is not exactly orthogonal. Its 2-norm distance to the
//! orthogonal group is `rho = max_i |sigma_i(W) - 1|`, attained by the polar
//! factor `O`, so every claim here is about `O`, and `S(W)` differs from `S(O)` by
//! at most `rho`. Adding the rounding of forming `S` and the eigensolver's backward
//! error gives `beta >= ||S_computed - S(O)||_2`, and by Weyl each computed
//! eigenvalue is within `beta` of its true one. Neighbouring computed eigenvalues
//! further apart than `2 beta` belong to provably distinct true eigenvalues; nearer
//! ones are not resolved and stay in one cluster. A cluster's projector carries the
//! Davis–Kahan bar `beta / (gap - beta)` of [`projector_error_bar`], `gap` being its
//! measured separation from the rest of the spectrum.
//!
//! The route is dense: `O(d^3)` time and `d x d` workspace for a `d x d` matrix.

use faer::Side;
use gam_linalg::decision::projector_error_bar;
use gam_linalg::faer_ndarray::{FaerLinalgError, FaerSvd, strict_symmetric_eigh};
use gam_linalg::roundoff::{
    accumulation_growth, factor_singular_band, symmetric_spectrum_rounding_band,
};
use ndarray::{Array2, ArrayView2};
use std::f64::consts::SQRT_2;

/// Why a matrix admits no plane-rotation recovery.
#[derive(Debug)]
pub enum PlaneRotationError {
    /// The matrix is empty or not square.
    NotSquare { rows: usize, cols: usize },
    /// An entry is not finite.
    NonFinite { row: usize, col: usize },
    /// The singular value or symmetric eigendecomposition failed.
    Linalg(FaerLinalgError),
    /// A cluster whose cosine interval excludes both `+1` and `-1` has odd
    /// dimension. The eigenvalues of `S(O)` inside `(-1, 1)` come in pairs, and a
    /// valid `beta` keeps every cluster a union of whole true eigenvalue groups, so
    /// an odd count means the derived bound was violated and no claim stands.
    OddInteriorCluster { cluster: usize, dimension: usize },
}

impl std::fmt::Display for PlaneRotationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSquare { rows, cols } => write!(
                formatter,
                "plane-rotation recovery needs a non-empty square matrix, got {rows}x{cols}"
            ),
            Self::NonFinite { row, col } => write!(
                formatter,
                "plane-rotation recovery needs finite entries; entry ({row}, {col}) is not finite"
            ),
            Self::Linalg(error) => {
                write!(formatter, "plane-rotation recovery decomposition failed: {error}")
            }
            Self::OddInteriorCluster { cluster, dimension } => write!(
                formatter,
                "plane-rotation recovery: interior cluster {cluster} has odd dimension \
                 {dimension}, so the derived perturbation bound was violated"
            ),
        }
    }
}

impl std::error::Error for PlaneRotationError {}

/// What one cluster of the symmetric part's spectrum is.
#[derive(Clone, Debug)]
pub enum RotationClusterKind {
    /// The cosine interval admits `+1`: directions the rotation fixes. A rotation
    /// by at most `max_hidden_angle` would be indistinguishable from fixing them.
    Fixed { max_hidden_angle: f64 },
    /// The cosine interval excludes `+1` and `-1`: `planes` planes whose cosines
    /// are not resolved from one another. `planes > 1` is the repeated-cosine case.
    Rotation {
        planes: usize,
        /// `atan2(s, c)`, with `c` the cluster's mean computed cosine and `s` the
        /// root-mean-square singular value of the compressed skew part.
        angle: f64,
        /// `J` in the cluster basis (`m x m` with `m = 2 planes`, skew, `J^2 = -I`),
        /// or `None` when the orientation is not certified (see
        /// [`recover_plane_rotations`]).
        complex_structure: Option<Array2<f64>>,
    },
    /// The cosine interval admits `-1`: half-turn planes without orientation, and a
    /// reflection when the dimension is odd. A rotation by at least
    /// `min_hidden_angle` would be indistinguishable from a half-turn.
    HalfTurn { min_hidden_angle: f64 },
    /// The cosine interval admits both `+1` and `-1`: no structure is resolved.
    Unresolved,
}

/// One cluster of the symmetric part's spectrum, provably separated from the rest.
#[derive(Clone, Debug)]
pub struct RotationCluster {
    /// Orthonormal basis of the cluster's computed eigenspace (`d x m`).
    pub basis: Array2<f64>,
    /// Interval containing every true cosine of the cluster:
    /// `[lowest computed - beta, highest computed + beta]`.
    pub cosine_interval: (f64, f64),
    /// Measured distance from the cluster's computed eigenvalues to the nearest
    /// other computed eigenvalue; infinite when the cluster is the whole spectrum.
    pub separation: f64,
    /// Davis–Kahan bound on the 2-norm distance between the projector onto `basis`
    /// and the true spectral projector of `S(O)`. Zero for the whole spectrum, whose
    /// projector is the identity.
    pub projector_bar: f64,
    pub kind: RotationClusterKind,
}

/// A structural fact about the recovered rotation that the matrix does not decide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RotationAmbiguity {
    /// Every cluster is fixed: no plane at all.
    Identity,
    /// Cluster `cluster` holds `planes >= 2` planes with unresolved cosines.
    RepeatedCosine { cluster: usize, planes: usize },
    /// Cluster `cluster` admits cosine `-1`: no orientation.
    HalfTurn { cluster: usize },
    /// Cluster `cluster` admits both `+1` and `-1`.
    Unresolved { cluster: usize },
    /// Angles are fixed only modulo `2 pi`; the reported representative is a
    /// convention.
    Winding,
}

/// Plane-rotation structure of a frozen square matrix.
#[derive(Clone, Debug)]
pub struct PlaneRotationRecovery {
    /// Certified bound on the 2-norm distance from the matrix to the orthogonal
    /// group, `rho`.
    pub orthogonality_defect: f64,
    /// `beta`, the bound on the distance between the decomposed symmetric part and
    /// `S(O)`.
    pub perturbation_bound: f64,
    /// Clusters in increasing cosine order.
    pub clusters: Vec<RotationCluster>,
}

impl PlaneRotationRecovery {
    /// The ambiguities the clusters carry, in cluster order, with
    /// [`RotationAmbiguity::Winding`] last whenever any rotation or half-turn exists.
    pub fn ambiguities(&self) -> Vec<RotationAmbiguity> {
        let mut ambiguities = Vec::new();
        if self
            .clusters
            .iter()
            .all(|cluster| matches!(cluster.kind, RotationClusterKind::Fixed { .. }))
        {
            ambiguities.push(RotationAmbiguity::Identity);
        }
        let mut winding = false;
        for (index, cluster) in self.clusters.iter().enumerate() {
            match &cluster.kind {
                RotationClusterKind::Rotation { planes, .. } => {
                    winding = true;
                    if *planes > 1 {
                        ambiguities.push(RotationAmbiguity::RepeatedCosine {
                            cluster: index,
                            planes: *planes,
                        });
                    }
                }
                RotationClusterKind::HalfTurn { .. } => {
                    winding = true;
                    ambiguities.push(RotationAmbiguity::HalfTurn { cluster: index });
                }
                RotationClusterKind::Unresolved => {
                    ambiguities.push(RotationAmbiguity::Unresolved { cluster: index });
                }
                RotationClusterKind::Fixed { .. } => {}
            }
        }
        if winding {
            ambiguities.push(RotationAmbiguity::Winding);
        }
        ambiguities
    }
}

/// Recover the rotation planes of a frozen square matrix.
///
/// Clusters the spectrum of `S = (W + W^T)/2` at the derived resolution `2 beta`
/// (module documentation) and classifies each cluster by whether its cosine
/// interval admits `+1` or `-1`. An interior cluster of dimension `m` compresses
/// the skew part to `G = V^T K V`.
///
/// # Orientation certificate
///
/// With `V Q` the true eigenbasis for the Procrustes-optimal `Q`,
/// `||V - V Q||_2 <= sqrt(2) bar` and `||K(O)||_2 <= 1`, so
///
/// ```text
/// ||G - Q^T V^T K(O) V Q||_2 <= ||K_computed - K(O)||_2 + 2 sqrt(2) bar + rounding.
/// ```
///
/// The true compression is `sin alpha` times a complex structure, with
/// `sin alpha >= sqrt(1 - max c^2)` over the cosine interval. The opposite
/// orientation lies `2 sin alpha` away, so when that sine floor exceeds the error
/// above, `J = G / s` carries the true orientation and is reported; otherwise it is
/// `None`.
pub fn recover_plane_rotations(
    matrix: ArrayView2<'_, f64>,
) -> Result<PlaneRotationRecovery, PlaneRotationError> {
    let (rows, cols) = matrix.dim();
    if rows == 0 || rows != cols {
        return Err(PlaneRotationError::NotSquare { rows, cols });
    }
    if let Some(((row, col), _)) = matrix.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(PlaneRotationError::NonFinite { row, col });
    }
    let dimension = rows;
    let (_, singular_values, _) = matrix
        .svd(false, false)
        .map_err(PlaneRotationError::Linalg)?;
    let sigma_max = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value));
    let orthogonality_defect = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max((value - 1.0).abs()))
        + factor_singular_band(dimension, dimension, sigma_max);

    // `a + b` and `b + a` round identically, so `symmetric` is exactly symmetric.
    let mut symmetric = Array2::<f64>::zeros((dimension, dimension));
    let mut skew = Array2::<f64>::zeros((dimension, dimension));
    for row in 0..dimension {
        for col in 0..dimension {
            symmetric[[row, col]] = 0.5 * (matrix[[row, col]] + matrix[[col, row]]);
            skew[[row, col]] = 0.5 * (matrix[[row, col]] - matrix[[col, row]]);
        }
    }
    // One rounded addition per entry, halved exactly: `u (|w_ij| + |w_ji|) / 2`,
    // whose Frobenius norm is at most `u ||W||_F`. The same bound holds for `K`.
    let formation_band = accumulation_growth(1) * frobenius_norm(matrix);
    let (values, vectors) =
        strict_symmetric_eigh(&symmetric, Side::Lower).map_err(PlaneRotationError::Linalg)?;
    let mut order: Vec<usize> = (0..dimension).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let cosines: Vec<f64> = order.iter().map(|&index| values[index]).collect();
    let perturbation_bound =
        orthogonality_defect + formation_band + symmetric_spectrum_rounding_band(&cosines);

    let resolution = 2.0 * perturbation_bound;
    let mut starts = vec![0_usize];
    for index in 1..dimension {
        if cosines[index] - cosines[index - 1] > resolution {
            starts.push(index);
        }
    }
    let mut clusters = Vec::with_capacity(starts.len());
    for (cluster_index, &start) in starts.iter().enumerate() {
        let end = starts
            .get(cluster_index + 1)
            .copied()
            .unwrap_or(dimension);
        let width = end - start;
        let lowest = cosines[start];
        let highest = cosines[end - 1];
        let below = if start > 0 {
            lowest - cosines[start - 1]
        } else {
            f64::INFINITY
        };
        let above = if end < dimension {
            cosines[end] - highest
        } else {
            f64::INFINITY
        };
        let separation = below.min(above);
        let projector_bar = if separation.is_finite() {
            projector_error_bar(separation, perturbation_bound)
        } else {
            0.0
        };
        let mut basis = Array2::<f64>::zeros((dimension, width));
        for (column, &index) in order[start..end].iter().enumerate() {
            basis.column_mut(column).assign(&vectors.column(index));
        }
        let cosine_interval = (lowest - perturbation_bound, highest + perturbation_bound);
        let admits_fixed = cosine_interval.1 >= 1.0;
        let admits_half_turn = cosine_interval.0 <= -1.0;
        // A true cosine lies in the interval and in `[-1, 1]`.
        let kind = match (admits_fixed, admits_half_turn) {
            (true, true) => RotationClusterKind::Unresolved,
            (true, false) => RotationClusterKind::Fixed {
                max_hidden_angle: cosine_interval.0.min(1.0).acos(),
            },
            (false, true) => RotationClusterKind::HalfTurn {
                min_hidden_angle: cosine_interval.1.max(-1.0).acos(),
            },
            (false, false) => {
                if width % 2 != 0 {
                    return Err(PlaneRotationError::OddInteriorCluster {
                        cluster: cluster_index,
                        dimension: width,
                    });
                }
                interior_rotation(
                    &skew,
                    &basis,
                    &cosines[start..end],
                    cosine_interval,
                    orthogonality_defect + formation_band + 2.0 * SQRT_2 * projector_bar,
                )
            }
        };
        clusters.push(RotationCluster {
            basis,
            cosine_interval,
            separation,
            projector_bar,
            kind,
        });
    }
    Ok(PlaneRotationRecovery {
        orthogonality_defect,
        perturbation_bound,
        clusters,
    })
}

/// Angle and certified orientation of an interior cluster. `skew_error` bounds
/// `||K_computed - K(O)||_2 + 2 sqrt(2) bar`; the compression's own rounding is
/// added here.
fn interior_rotation(
    skew: &Array2<f64>,
    basis: &Array2<f64>,
    cosines: &[f64],
    cosine_interval: (f64, f64),
    skew_error: f64,
) -> RotationClusterKind {
    let (dimension, width) = basis.dim();
    let compressed = basis.t().dot(&skew.dot(basis));
    // Two nested length-`d` accumulations per entry: at most `gamma_{2d}` times the
    // absolute sum of the terms, `(|V|^T |K| |V|)_ab`.
    let absolute_basis = basis.mapv(f64::abs);
    let absolute_compressed = absolute_basis
        .t()
        .dot(&skew.mapv(f64::abs).dot(&absolute_basis));
    let compression_band =
        accumulation_growth(2 * dimension) * frobenius_norm(absolute_compressed.view());
    let sine = frobenius_norm(compressed.view()) / (width as f64).sqrt();
    let mean_cosine = cosines.iter().sum::<f64>() / width as f64;
    let magnitude = cosine_interval.0.abs().max(cosine_interval.1.abs());
    let sine_floor = (1.0 - magnitude * magnitude).sqrt();
    let complex_structure = if sine_floor > skew_error + compression_band {
        Some(compressed.mapv(|value| value / sine))
    } else {
        None
    };
    RotationClusterKind::Rotation {
        planes: width / 2,
        angle: sine.atan2(mean_cosine),
        complex_structure,
    }
}

fn frobenius_norm(matrix: ArrayView2<'_, f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::faer_ndarray::{FaerEigh, FaerQr};
    use ndarray::s;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::f64::consts::PI;
    use std::ops::Range;

    const DIMENSION: usize = 8;

    /// A random orthonormal basis: Householder QR of a uniform draw.
    fn hidden_basis(dimension: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let draw =
            Array2::<f64>::from_shape_fn((dimension, dimension), |_| rng.random_range(-1.0..1.0));
        let (q, _) = draw.qr().expect("Householder QR of a uniform draw");
        q
    }

    /// Block-diagonal normal form: `[[c, -s], [s, c]]` per angle, then
    /// `negative_axes` entries `-1`, then `+1`.
    fn normal_form(dimension: usize, angles: &[f64], negative_axes: usize) -> Array2<f64> {
        let mut form = Array2::<f64>::eye(dimension);
        for (plane, &angle) in angles.iter().enumerate() {
            let (sine, cosine) = angle.sin_cos();
            let offset = 2 * plane;
            form[[offset, offset]] = cosine;
            form[[offset, offset + 1]] = -sine;
            form[[offset + 1, offset]] = sine;
            form[[offset + 1, offset + 1]] = cosine;
        }
        for axis in 0..negative_axes {
            let index = 2 * angles.len() + axis;
            form[[index, index]] = -1.0;
        }
        form
    }

    /// The planted matrix `W = Q B Q^T` and bounds on how far its float factors are
    /// from their exactly orthogonal polar factors `Q_o`, `B_o`.
    struct Planted {
        basis: Array2<f64>,
        matrix: Array2<f64>,
        /// Bound on `||Q - Q_o||_2`.
        basis_defect: f64,
        /// Bound on `||B - B_o||_2`.
        form_defect: f64,
        /// Bound on `||W - Q_o B_o Q_o^T||_2`.
        matrix_defect: f64,
    }

    /// `||X - X_o||_2 = max |sigma_i - 1| <= ||X^T X - I||_F`, plus the rounding of
    /// the Gram: `gamma_n` times each entry's absolute term sum.
    fn orthogonality_bound(factor: &Array2<f64>) -> f64 {
        let columns = factor.ncols();
        let gram = factor.t().dot(factor) - Array2::<f64>::eye(columns);
        let absolute = factor.mapv(f64::abs);
        frobenius_norm(gram.view())
            + accumulation_growth(columns) * frobenius_norm(absolute.t().dot(&absolute).view())
    }

    fn plant(dimension: usize, angles: &[f64], negative_axes: usize, seed: u64) -> Planted {
        let basis = hidden_basis(dimension, seed);
        let form = normal_form(dimension, angles, negative_axes);
        let matrix = basis.dot(&form).dot(&basis.t());
        let basis_defect = orthogonality_bound(&basis);
        let form_defect = orthogonality_bound(&form);
        let absolute_basis = basis.mapv(f64::abs);
        let formation = accumulation_growth(2 * dimension)
            * frobenius_norm(
                absolute_basis
                    .dot(&form.mapv(f64::abs))
                    .dot(&absolute_basis.t())
                    .view(),
            );
        // ||Q B Q^T - Q_o B_o Q_o^T|| <= ||Q - Q_o|| ||B|| ||Q||
        //   + ||Q_o|| ||B - B_o|| ||Q|| + ||Q_o|| ||B_o|| ||Q - Q_o||.
        let matrix_defect = basis_defect * (1.0 + form_defect) * (1.0 + basis_defect)
            + form_defect * (1.0 + basis_defect)
            + basis_defect
            + formation;
        Planted {
            basis,
            matrix,
            basis_defect,
            form_defect,
            matrix_defect,
        }
    }

    /// `||V_a V_a^T - V_b V_b^T||_2` and the rounding band of that measurement.
    fn projector_distance(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> (f64, f64) {
        let dimension = left.nrows();
        let left_projector = left.dot(&left.t());
        let right_projector = right.dot(&right.t());
        let mut difference = Array2::<f64>::zeros((dimension, dimension));
        for row in 0..dimension {
            for col in 0..dimension {
                difference[[row, col]] = 0.5
                    * ((left_projector[[row, col]] - right_projector[[row, col]])
                        + (left_projector[[col, row]] - right_projector[[col, row]]));
            }
        }
        let (values, _) = difference
            .eigh(Side::Lower)
            .expect("projector difference eigendecomposition");
        let distance = values.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let absolute_left = left.mapv(f64::abs);
        let absolute_right = right.mapv(f64::abs);
        let band = accumulation_growth(left.ncols())
            * frobenius_norm(absolute_left.dot(&absolute_left.t()).view())
            + accumulation_growth(right.ncols())
                * frobenius_norm(absolute_right.dot(&absolute_right.t()).view())
            + accumulation_growth(3)
                * (frobenius_norm(left_projector.view()) + frobenius_norm(right_projector.view()))
            + symmetric_spectrum_rounding_band(&values.to_vec());
        (distance, band)
    }

    /// The cluster spans the planted columns: within the Davis–Kahan bar about
    /// `Q_o B_o Q_o^T` (its `beta` with `rho` replaced by the planted defect), plus
    /// `||Q_k Q_k^T - Q_o,k Q_o,k^T|| <= eta_Q (2 + eta_Q)` and the measurement band.
    fn assert_spans_planted(
        recovery: &PlaneRotationRecovery,
        cluster: &RotationCluster,
        planted: &Planted,
        columns: Range<usize>,
    ) {
        let truth_bound =
            recovery.perturbation_bound - recovery.orthogonality_defect + planted.matrix_defect;
        let bar = if cluster.separation.is_finite() {
            projector_error_bar(cluster.separation, truth_bound)
        } else {
            0.0
        };
        let column_defect = planted.basis_defect * (2.0 + planted.basis_defect);
        let (distance, band) = projector_distance(
            cluster.basis.view(),
            planted.basis.slice(s![.., columns.clone()]),
        );
        assert!(
            distance <= bar + column_defect + band,
            "planted columns {columns:?}: projector distance {distance:.3e} exceeds bar \
             {bar:.3e} + column defect {column_defect:.3e} + band {band:.3e}"
        );
    }

    /// The planted cosine lies in the cluster's interval widened by the planted
    /// defects: `beta_truth <= beta + eta_W`, and `|c - c/r| <= eta_B`.
    fn assert_cosine_admitted(cluster: &RotationCluster, planted: &Planted, angle: f64) {
        let widen = planted.matrix_defect + planted.form_defect;
        let cosine = angle.sin_cos().1;
        assert!(
            cluster.cosine_interval.0 - widen <= cosine
                && cosine <= cluster.cosine_interval.1 + widen,
            "planted cosine {cosine} outside {:?} widened by {widen:.3e}",
            cluster.cosine_interval
        );
    }

    /// `<J_recovered, J_planted>_F / 2`, which is `+1` for the planted orientation and
    /// `-1` for the reverse. `J_planted = q_2 q_1^T - q_1 q_2^T` from `[[0, -1], [1, 0]]`.
    fn orientation_alignment(cluster: &RotationCluster, planted: &Planted, first: usize) -> f64 {
        let structure = match &cluster.kind {
            RotationClusterKind::Rotation {
                complex_structure: Some(structure),
                ..
            } => structure,
            other => panic!("expected a certified rotation, got {other:?}"),
        };
        let recovered = cluster.basis.dot(structure).dot(&cluster.basis.t());
        let (u, v) = (planted.basis.column(first), planted.basis.column(first + 1));
        let dimension = planted.basis.nrows();
        let planted_structure =
            Array2::<f64>::from_shape_fn((dimension, dimension), |(row, col)| {
                v[row] * u[col] - u[row] * v[col]
            });
        (&recovered * &planted_structure).sum() / 2.0
    }

    fn rotation_planes(cluster: &RotationCluster) -> usize {
        match &cluster.kind {
            RotationClusterKind::Rotation { planes, angle, .. } => {
                assert!(*angle > 0.0 && *angle < PI, "angle {angle} outside (0, pi)");
                *planes
            }
            other => panic!("expected a rotation cluster, got {other:?}"),
        }
    }

    #[test]
    fn planted_planes_hidden_by_a_random_basis_are_recovered_2951() {
        let angles = [0.35, 1.2, 2.6];
        let planted = plant(DIMENSION, &angles, 0, 0x2951_0001);
        let recovery = recover_plane_rotations(planted.matrix.view()).expect("recovery");
        assert_eq!(recovery.clusters.len(), 4, "{recovery:?}");
        // Increasing cosine: 2.6, 1.2, 0.35, then the fixed space.
        for (cluster, (angle, first)) in recovery
            .clusters
            .iter()
            .zip([(2.6, 4_usize), (1.2, 2), (0.35, 0)])
        {
            assert_eq!(rotation_planes(cluster), 1, "{cluster:?}");
            assert_cosine_admitted(cluster, &planted, angle);
            assert_spans_planted(&recovery, cluster, &planted, first..first + 2);
            let alignment = orientation_alignment(cluster, &planted, first);
            assert!(alignment > 0.0, "angle {angle}: alignment {alignment}");
        }
        let fixed = &recovery.clusters[3];
        assert!(
            matches!(fixed.kind, RotationClusterKind::Fixed { .. }),
            "{fixed:?}"
        );
        assert_eq!(fixed.basis.ncols(), 2);
        assert_spans_planted(&recovery, fixed, &planted, 6..8);
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
    }

    #[test]
    fn repeated_cosines_report_the_invariant_subspace_not_the_planes_2951() {
        let planted = plant(DIMENSION, &[0.9, 0.9, 2.0], 0, 0x2951_0002);
        let recovery = recover_plane_rotations(planted.matrix.view()).expect("recovery");
        assert_eq!(recovery.clusters.len(), 3, "{recovery:?}");
        let repeated = &recovery.clusters[1];
        assert_eq!(rotation_planes(repeated), 2, "{repeated:?}");
        assert_cosine_admitted(repeated, &planted, 0.9);
        assert_spans_planted(&recovery, repeated, &planted, 0..4);
        assert_eq!(
            recovery.ambiguities(),
            vec![
                RotationAmbiguity::RepeatedCosine {
                    cluster: 1,
                    planes: 2
                },
                RotationAmbiguity::Winding
            ]
        );

        // Positive control: distinct cosines are separated into identified planes.
        let distinct = plant(DIMENSION, &[0.9, 1.0, 2.0], 0, 0x2951_0002);
        let recovery = recover_plane_rotations(distinct.matrix.view()).expect("recovery");
        assert_eq!(recovery.clusters.len(), 4, "{recovery:?}");
        for (cluster, (angle, first)) in recovery
            .clusters
            .iter()
            .zip([(2.0, 4_usize), (1.0, 2), (0.9, 0)])
        {
            assert_eq!(rotation_planes(cluster), 1, "{cluster:?}");
            assert_spans_planted(&recovery, cluster, &distinct, first..first + 2);
            assert_cosine_admitted(cluster, &distinct, angle);
        }
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
    }

    #[test]
    fn non_orthogonality_widens_the_grouping_until_distinct_cosines_merge_2951() {
        let planted = plant(DIMENSION, &[0.9, 1.0], 0, 0x2951_0003);
        let clean = recover_plane_rotations(planted.matrix.view()).expect("recovery");
        assert_eq!(clean.clusters.len(), 3, "{clean:?}");
        assert_eq!(clean.ambiguities(), vec![RotationAmbiguity::Winding]);

        // Scaling by 1.05 moves every singular value to 1.05, so rho >= 0.05 and the
        // resolution 2 beta exceeds the scaled cosine gap 1.05 (cos 0.9 - cos 1.0).
        let scale = 1.05;
        let scaled = planted.matrix.mapv(|value| scale * value);
        let widened = recover_plane_rotations(scaled.view()).expect("recovery");
        let scaled_gap = scale * (0.9_f64.cos() - 1.0_f64.cos());
        assert!(
            2.0 * widened.perturbation_bound > scaled_gap,
            "resolution {} does not exceed the scaled gap {scaled_gap}",
            2.0 * widened.perturbation_bound
        );
        assert_eq!(widened.clusters.len(), 2, "{widened:?}");
        assert_eq!(rotation_planes(&widened.clusters[0]), 2);
        assert!(matches!(
            widened.clusters[1].kind,
            RotationClusterKind::Fixed { .. }
        ));
        assert_eq!(
            widened.ambiguities(),
            vec![
                RotationAmbiguity::RepeatedCosine {
                    cluster: 0,
                    planes: 2
                },
                RotationAmbiguity::Winding
            ]
        );
    }

    #[test]
    fn half_turns_and_reflections_report_no_orientation_2951() {
        let planted = plant(DIMENSION, &[PI, 1.1], 0, 0x2951_0004);
        let recovery = recover_plane_rotations(planted.matrix.view()).expect("recovery");
        assert_eq!(recovery.clusters.len(), 3, "{recovery:?}");
        let half_turn = &recovery.clusters[0];
        assert!(
            matches!(half_turn.kind, RotationClusterKind::HalfTurn { .. }),
            "{half_turn:?}"
        );
        assert_eq!(half_turn.basis.ncols(), 2);
        assert_spans_planted(&recovery, half_turn, &planted, 0..2);
        assert_eq!(rotation_planes(&recovery.clusters[1]), 1);
        assert_eq!(
            recovery.ambiguities(),
            vec![
                RotationAmbiguity::HalfTurn { cluster: 0 },
                RotationAmbiguity::Winding
            ]
        );

        // A single -1 axis is a reflection: an odd-dimensional half-turn space.
        let reflection = plant(DIMENSION, &[1.1], 1, 0x2951_0004);
        let recovery = recover_plane_rotations(reflection.matrix.view()).expect("recovery");
        assert_eq!(recovery.clusters.len(), 3, "{recovery:?}");
        let axis = &recovery.clusters[0];
        assert!(
            matches!(axis.kind, RotationClusterKind::HalfTurn { .. }),
            "{axis:?}"
        );
        assert_eq!(axis.basis.ncols(), 1);
        assert_spans_planted(&recovery, axis, &reflection, 2..3);
    }

    #[test]
    fn identity_reports_no_plane_and_the_largest_angle_it_cannot_exclude_2951() {
        let identity = Array2::<f64>::eye(DIMENSION);
        let recovery = recover_plane_rotations(identity.view()).expect("recovery");
        assert_eq!(recovery.clusters.len(), 1, "{recovery:?}");
        let cluster = &recovery.clusters[0];
        assert!(cluster.separation.is_infinite());
        assert_eq!(cluster.projector_bar, 0.0);
        let max_hidden_angle = match cluster.kind {
            RotationClusterKind::Fixed { max_hidden_angle } => max_hidden_angle,
            ref other => panic!("expected a fixed cluster, got {other:?}"),
        };
        // `1 - cos(hidden) = beta + (1 - lowest computed eigenvalue) <= 2 beta`.
        assert!(max_hidden_angle > 0.0);
        assert!(1.0 - max_hidden_angle.cos() <= 2.0 * recovery.perturbation_bound);
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Identity]);

        // Positive control: a rotation well below that angle is reported as identity,
        // while a resolvable one is a plane.
        let hidden = plant(DIMENSION, &[max_hidden_angle / 4.0], 0, 0x2951_0005);
        let recovery = recover_plane_rotations(hidden.matrix.view()).expect("recovery");
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Identity]);
        let visible = plant(DIMENSION, &[0.3], 0, 0x2951_0005);
        let recovery = recover_plane_rotations(visible.matrix.view()).expect("recovery");
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
    }

    #[test]
    fn winding_and_orientation_resolve_to_the_principal_representative_2951() {
        for (angle, sign) in [(1.1, 1.0), (1.1 + 2.0 * PI, 1.0), (-1.1, -1.0)] {
            let planted = plant(DIMENSION, &[angle], 0, 0x2951_0006);
            let recovery = recover_plane_rotations(planted.matrix.view()).expect("recovery");
            assert_eq!(recovery.clusters.len(), 2, "{recovery:?}");
            let plane = &recovery.clusters[0];
            assert_eq!(rotation_planes(plane), 1);
            assert_cosine_admitted(plane, &planted, angle);
            assert_spans_planted(&recovery, plane, &planted, 0..2);
            let alignment = orientation_alignment(plane, &planted, 0);
            assert!(
                sign * alignment > 0.0,
                "angle {angle}: alignment {alignment} has the wrong sign"
            );
            assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
        }
    }

    #[test]
    fn orientation_is_withheld_when_the_skew_part_is_not_resolved_2951() {
        let dimension = 4;
        let angles = [0.6_f64.acos(), (-0.6_f64).acos()];
        let planted = plant(dimension, &angles, 0, 0x2951_0007);
        let exact = recover_plane_rotations(planted.matrix.view()).expect("recovery");
        assert_eq!(exact.clusters.len(), 2, "{exact:?}");
        for cluster in &exact.clusters {
            assert!(
                matches!(
                    cluster.kind,
                    RotationClusterKind::Rotation {
                        complex_structure: Some(_),
                        ..
                    }
                ),
                "{cluster:?}"
            );
        }

        // Shrinking the skew part to a tenth keeps the symmetric part, but moves the
        // matrix 1 - sqrt(0.36 + 0.0064) ~ 0.395 from the orthogonal group. The
        // cosine intervals still exclude +-1, yet their sine floor sqrt(1 - 0.995^2)
        // is below that defect, so no orientation is certified.
        let skew_scale = 0.1;
        let shrunk = Array2::<f64>::from_shape_fn((dimension, dimension), |(row, col)| {
            let (forward, backward) = (planted.matrix[[row, col]], planted.matrix[[col, row]]);
            0.5 * (forward + backward) + skew_scale * 0.5 * (forward - backward)
        });
        let withheld = recover_plane_rotations(shrunk.view()).expect("recovery");
        assert_eq!(withheld.clusters.len(), 2, "{withheld:?}");
        for cluster in &withheld.clusters {
            assert!(
                matches!(
                    cluster.kind,
                    RotationClusterKind::Rotation {
                        complex_structure: None,
                        ..
                    }
                ),
                "{cluster:?}"
            );
        }
    }
}
