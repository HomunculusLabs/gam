//! Invariant planes of a frozen non-orthogonal operator, each with a certified
//! perturbation bound (#2951).
//!
//! # Why P3 does not apply
//!
//! `spectral` reads rotation planes off the symmetric part `(W + W^T)/2`. That is
//! valid only for an orthogonal `W`. A conjugated rotation `T = I + U (R - I) U^+`
//! with a non-orthonormal `U` (mpd-modadd's shift operator `T_s`), or a trained
//! weight matrix, is not orthogonal, and the eigenspaces of its symmetric part are
//! not its planes. What survives is the spectral structure of `T` itself. On an
//! invariant subspace `V` (`T V ⊆ V`) where `T` has the complex pair `r e^{±iα}`,
//!
//! ```text
//! T|_V = r (cos α I + sin α J),     J^2 = -I on V,
//! ```
//!
//! so `T U = U r R_α` for `U = [x, J x]` and any nonzero `x ∈ V`. The plane `V`,
//! `r`, `α` and `J` are determined by `T`. The basis `U` is determined only up to
//! the commutant `{a I + b J}`, so no `U` is reported.
//!
//! # Certificate (Stewart–Sun, Theorem V.2.1)
//!
//! Let `Q = [Q_1 Q_2]` be orthogonal, `Q_1` spanning a candidate subspace, and
//! `Q^T T Q = [[L_1, H], [G, L_2]]`. Take `γ = ||G||_F`, `η = ||H||_F` and
//! `δ = sep_F(L_1, L_2) = σ_min(I ⊗ L_1 - L_2^T ⊗ I)`. If `γ η / δ^2 < 1/4`, some
//! `P` with `||P||_F < 2γ/δ` makes `Q_1 + Q_2 P` span an invariant subspace of `T`,
//! on which `T` acts by `L_1 + H P`. The projector onto `Q_1` is then within `2γ/δ`
//! of the true one (`sin Θ <= tan Θ`), and the restriction is within `2γη/δ` of
//! `L_1`.
//!
//! The certificate is a posteriori: it holds for any candidate, however that
//! candidate was produced, so an inaccurate candidate is refused, never reported.
//! The computed `Q` is orthogonal only up to `η_Q >= ||Q^T Q - I||_2`. Every claim
//! is therefore about its polar factor `Q_o`, with `||Q - Q_o||_2 <= η_Q` and
//! `||Q^T T Q - Q_o^T T Q_o||_F <= η_Q (2 + η_Q) ||T||_F`, plus the Wilkinson band
//! of forming `Q^T T Q`. That similarity error adds to `γ` and `η`, and is
//! subtracted from `δ` once per block, since `sep_F` is 1-Lipschitz in each. The SVD
//! owner's backward band and the rounding of forming the Kronecker operator are
//! subtracted from `δ` as well.
//!
//! # Classifying a certified plane
//!
//! A 2x2 restriction with trace `t` and determinant `d` carries a complex pair iff
//! `t^2 - 4d < 0`. A perturbation `||E||_F <= ε` moves `t` by at most `sqrt(2) ε`
//! and `d` by at most `||L||_F ε + ε^2/2`, so the sign of the discriminant is either
//! certified or not:
//!
//! * certified negative: a rotation-scaling, with modulus and cosine intervals and
//!   `J`;
//! * certified positive: two distinct real eigenvalues and no rotation;
//! * neither: a defective or unresolved double eigenvalue. No action is claimed.
//!
//! A block of dimension above two gets its projector and no planes. Repeated
//! eigenvalues do not identify planes (P3's repeated-cosine case), and nothing here
//! certifies that such a block is semisimple.
//!
//! # Recovery
//!
//! [`recover_invariant_blocks`] proposes, certifies and merges. Eigenvalue estimates
//! come from gam-linalg's certified [`real_general_spectrum`]. It certifies each
//! eigenvalue by its measured backward error: the eigenvector residual, or where
//! faer's vector is no evidence (an exactly repeated semisimple pair), σ_min(A − λI).
//! It refuses an unconverged or moved eigenvalue with a typed refusal. faer 0.24 keeps
//! its Schur vectors private, so the estimates only seed proposals and no claim rests
//! on them: an estimate the owner's certificate cannot rule out yields proposals that
//! fail certification and merge.
//! Each real estimate and each conjugate pair starts as its own cluster.
//!
//! A cluster of `k` estimates proposes the `k` smallest right singular vectors of
//! `p(T)^m`, for `m = 1, 2, …` until `m deg p >= k`. Here `p(x) = x - μ` with `μ`
//! the mean estimate when every estimate is real, and `p(x) = x^2 - 2 a x + b`
//! otherwise, with `a` the mean real part and `b` the mean `|λ|^2`. The powers
//! reach generalized eigenspaces of defective blocks.
//!
//! A cluster that no proposal certifies absorbs its nearest cluster, and every
//! cluster within `2 sqrt(ε_sim η)`. That is the separation Stewart's condition
//! asks of a proposal invariant up to the unavoidable similarity error `ε_sim`, and
//! `sep_F(L_1, L_2)` never exceeds the distance between the spectra of `L_1` and
//! `L_2`. Each merge removes a cluster, and a cluster covering the whole space
//! certifies with no separation needed, so the loop ends.
//!
//! Every bar here is a uniform bound that includes numerical error, over the stated
//! hypotheses and nothing wider (#2946 fr-census overclaim audit, comment
//! 5716123817).
//!
//! The route is dense. One proposal costs `O(n^3)` time with `n x n` workspace;
//! certifying a `k`-dimensional cluster adds `O((k (n - k))^3)` time and a
//! `k (n - k)` square workspace.

use gam_linalg::faer_ndarray::{FaerLinalgError, FaerQr, FaerSvd, real_general_spectrum};
use gam_linalg::roundoff::{accumulation_band, accumulation_growth, factor_singular_band};
use ndarray::{Array2, ArrayView2, s};
use std::f64::consts::SQRT_2;

/// Why no invariant-subspace claim could be attempted.
#[derive(Debug)]
pub enum InvariantSubspaceError {
    /// The operator is empty or not square.
    NotSquare { rows: usize, cols: usize },
    /// An operator entry is not finite.
    NonFinite { row: usize, col: usize },
    /// A candidate basis needs the operator's row count and between 1 and that
    /// many columns.
    BasisShape {
        rows: usize,
        cols: usize,
        dimension: usize,
    },
    /// A candidate basis entry is not finite.
    NonFiniteBasis { row: usize, col: usize },
    /// The candidate basis resolves fewer columns than it has, above the SVD
    /// owner's backward band.
    RankDeficientBasis { resolved: usize, columns: usize },
    /// A singular value, QR or certified general eigendecomposition failed or refused.
    Linalg(FaerLinalgError),
    /// A complex eigenvalue estimate whose neighbour is not its conjugate.
    UnpairedComplexEigenvalue { index: usize },
}

impl std::fmt::Display for InvariantSubspaceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSquare { rows, cols } => write!(
                formatter,
                "invariant-subspace recovery needs a non-empty square operator, got {rows}x{cols}"
            ),
            Self::NonFinite { row, col } => write!(
                formatter,
                "invariant-subspace recovery needs finite entries; entry ({row}, {col}) is not finite"
            ),
            Self::BasisShape { rows, cols, dimension } => write!(
                formatter,
                "a candidate basis needs {dimension} rows and 1 to {dimension} columns, got {rows}x{cols}"
            ),
            Self::NonFiniteBasis { row, col } => write!(
                formatter,
                "candidate basis entry ({row}, {col}) is not finite"
            ),
            Self::RankDeficientBasis { resolved, columns } => write!(
                formatter,
                "the candidate basis resolves {resolved} of its {columns} columns above the SVD band"
            ),
            Self::Linalg(error) => {
                write!(formatter, "invariant-subspace decomposition failed: {error}")
            }
            Self::UnpairedComplexEigenvalue { index } => write!(
                formatter,
                "eigenvalue estimate {index} is complex but its neighbour is not its conjugate"
            ),
        }
    }
}

impl std::error::Error for InvariantSubspaceError {}

/// Whether Stewart's condition certified the candidate subspace.
#[derive(Clone, Debug)]
pub enum SubspaceVerdict {
    /// An invariant subspace `V` of the operator exists, with
    /// `||Q_1 Q_1^T - P_V||_2 <= projector_bar`. The operator restricted to `V`,
    /// in some basis of `V`, is within `restriction_error` in Frobenius norm of
    /// `restriction`.
    Certified {
        projector_bar: f64,
        restriction_error: f64,
    },
    /// `4 γ η >= δ^2` or `δ <= 0`: no claim. `required_separation = 2 sqrt(γ η)`
    /// is the separation the condition would have needed.
    NotSeparated { required_separation: f64 },
}

/// The a posteriori certificate of one candidate subspace (module documentation).
#[derive(Clone, Debug)]
pub struct InvariantSubspaceCertificate {
    /// `Q_1`, orthonormal up to `frame_defect` (`n x k`).
    pub basis: Array2<f64>,
    /// `Q_1^T T Q_1` as computed (`k x k`).
    pub restriction: Array2<f64>,
    /// Upper bound on `γ`, the Frobenius norm of the exact `G` block.
    pub residual: f64,
    /// Upper bound on `η`, the Frobenius norm of the exact `H` block.
    pub coupling: f64,
    /// Lower bound on `sep_F(L_1, L_2)`; infinite when `k = n`.
    pub separation: f64,
    /// `η_Q`, the bound on `||Q - Q_o||_2`.
    pub frame_defect: f64,
    /// `ε_sim`, the bound on `||Q^T T Q - Q_o^T T Q_o||_F` including formation.
    pub similarity_error: f64,
    pub verdict: SubspaceVerdict,
}

impl InvariantSubspaceCertificate {
    /// The action of the operator on the certified subspace, or `None` when the
    /// subspace was not certified.
    pub fn kind(&self) -> Option<InvariantBlockKind> {
        match self.verdict {
            SubspaceVerdict::Certified {
                restriction_error, ..
            } => Some(block_kind(&self.restriction, restriction_error)),
            SubspaceVerdict::NotSeparated { .. } => None,
        }
    }
}

/// What the operator does on a certified invariant subspace.
#[derive(Clone, Debug)]
pub enum InvariantBlockKind {
    /// One real eigenvalue, inside `eigenvalue_interval`.
    Real { eigenvalue_interval: (f64, f64) },
    /// A complex pair `r e^{±iα}`: `T|_V = r (cos α I + sin α J)`.
    RotationScaling {
        /// Interval containing `r`.
        modulus_interval: (f64, f64),
        /// Interval containing `cos α`.
        cosine_interval: (f64, f64),
        /// `α` in `(0, π)` from the computed restriction. The orientation is
        /// carried by `J`, and `α + 2πk` and `(-α, -J)` give the same operator.
        angle: f64,
        /// `J` in the coordinates of the certificate's `basis`, computed as
        /// `(L - t/2 I) / sqrt(d - t^2/4)`.
        complex_structure: Array2<f64>,
    },
    /// Two distinct real eigenvalues, with their computed estimates: a real
    /// two-dimensional block with no rotation.
    RealPair { eigenvalues: (f64, f64) },
    /// The discriminant's sign is not certified: a defective or unresolved double
    /// eigenvalue. No action is claimed.
    Unresolved,
    /// A block of dimension above two. Its planes are not identified.
    Repeated { dimension: usize },
}

/// A structural fact about the recovered operator that its matrix does not decide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvariantAmbiguity {
    /// Block `block` has dimension above two: planes not identified.
    Repeated { block: usize, dimension: usize },
    /// Block `block` has an unresolved or defective double eigenvalue.
    Unresolved { block: usize },
    /// Angles are fixed only modulo `2π`, and the representative is a convention.
    Winding,
}

/// One certified invariant block.
#[derive(Clone, Debug)]
pub struct InvariantBlock {
    /// The computed eigenvalue estimates `(re, im)` that proposed the block. They
    /// are not certified.
    pub eigenvalue_estimates: Vec<(f64, f64)>,
    pub certificate: InvariantSubspaceCertificate,
    pub kind: InvariantBlockKind,
}

/// A partition of the whole space into certified invariant blocks.
#[derive(Clone, Debug)]
pub struct InvariantBlockRecovery {
    /// Blocks in increasing order of the mean real part of their estimates.
    pub blocks: Vec<InvariantBlock>,
}

impl InvariantBlockRecovery {
    /// The ambiguities in block order, with [`InvariantAmbiguity::Winding`] last
    /// whenever any rotation-scaling exists.
    pub fn ambiguities(&self) -> Vec<InvariantAmbiguity> {
        let mut ambiguities = Vec::new();
        let mut winding = false;
        for (index, block) in self.blocks.iter().enumerate() {
            match &block.kind {
                InvariantBlockKind::RotationScaling { .. } => winding = true,
                InvariantBlockKind::Unresolved => {
                    ambiguities.push(InvariantAmbiguity::Unresolved { block: index });
                }
                InvariantBlockKind::Repeated { dimension } => {
                    ambiguities.push(InvariantAmbiguity::Repeated {
                        block: index,
                        dimension: *dimension,
                    });
                }
                InvariantBlockKind::Real { .. } | InvariantBlockKind::RealPair { .. } => {}
            }
        }
        if winding {
            ambiguities.push(InvariantAmbiguity::Winding);
        }
        ambiguities
    }
}

/// Certify that `basis` (`n x k`, any full-rank basis) spans an invariant subspace
/// of `matrix` (module documentation).
pub fn certify_invariant_subspace(
    matrix: ArrayView2<'_, f64>,
    basis: ArrayView2<'_, f64>,
) -> Result<InvariantSubspaceCertificate, InvariantSubspaceError> {
    validate_operator(matrix)?;
    let dimension = matrix.nrows();
    let (rows, cols) = basis.dim();
    if rows != dimension || cols == 0 || cols > dimension {
        return Err(InvariantSubspaceError::BasisShape {
            rows,
            cols,
            dimension,
        });
    }
    if let Some(((row, col), _)) = basis.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(InvariantSubspaceError::NonFiniteBasis { row, col });
    }
    let (_, singular_values, _) = basis
        .svd(false, false)
        .map_err(InvariantSubspaceError::Linalg)?;
    let sigma_max = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value));
    let band = factor_singular_band(rows, cols, sigma_max);
    let resolved = singular_values.iter().filter(|&&value| value > band).count();
    if resolved < cols {
        return Err(InvariantSubspaceError::RankDeficientBasis {
            resolved,
            columns: cols,
        });
    }
    // Householder QR of `[basis, e_1, ..., e_{n-k}]` is orthogonal whatever the
    // trailing columns are, and its leading `k` columns span `basis`.
    let mut completed = Array2::<f64>::zeros((dimension, dimension));
    completed.slice_mut(s![.., ..cols]).assign(&basis);
    for column in cols..dimension {
        completed[[column - cols, column]] = 1.0;
    }
    let (frame, _) = completed.qr().map_err(InvariantSubspaceError::Linalg)?;
    certify_frame(matrix, &frame, cols)
}

/// Partition the whole space into certified invariant blocks (module
/// documentation).
pub fn recover_invariant_blocks(
    matrix: ArrayView2<'_, f64>,
) -> Result<InvariantBlockRecovery, InvariantSubspaceError> {
    validate_operator(matrix)?;
    let estimates = eigenvalue_estimates(matrix)?;
    let mut clusters = initial_clusters(&estimates)?;
    let mut accepted: Vec<Option<(InvariantSubspaceCertificate, InvariantBlockKind)>> =
        vec![None; clusters.len()];
    while let Some(position) = accepted.iter().position(Option::is_none) {
        let (certified, reach_floor) =
            propose_and_certify(matrix, &estimates, &clusters[position])?;
        if certified.is_some() {
            accepted[position] = certified;
            continue;
        }
        // A cluster covering the whole space always certifies, so a failing
        // cluster has neighbours and the nearest one is absorbed.
        let distances: Vec<f64> = (0..clusters.len())
            .map(|other| {
                if other == position {
                    f64::INFINITY
                } else {
                    cluster_distance(&estimates, &clusters[position], &clusters[other])
                }
            })
            .collect();
        let nearest = distances.iter().copied().fold(f64::INFINITY, f64::min);
        let reach = reach_floor.max(nearest);
        let mut merged = Vec::new();
        let mut kept_clusters = Vec::with_capacity(clusters.len());
        let mut kept_accepted = Vec::with_capacity(clusters.len());
        for (index, (members, certificate)) in clusters.into_iter().zip(accepted).enumerate() {
            if index == position || distances[index] <= reach {
                merged.extend(members);
            } else {
                kept_clusters.push(members);
                kept_accepted.push(certificate);
            }
        }
        merged.sort_unstable();
        kept_clusters.insert(0, merged);
        kept_accepted.insert(0, None);
        clusters = kept_clusters;
        accepted = kept_accepted;
    }
    let mut blocks: Vec<InvariantBlock> = clusters
        .into_iter()
        .zip(accepted)
        .filter_map(|(members, certified)| {
            certified.map(|(certificate, kind)| InvariantBlock {
                eigenvalue_estimates: members.iter().map(|&index| estimates[index]).collect(),
                certificate,
                kind,
            })
        })
        .collect();
    blocks.sort_by(|left, right| mean_real_part(left).total_cmp(&mean_real_part(right)));
    Ok(InvariantBlockRecovery { blocks })
}

fn validate_operator(matrix: ArrayView2<'_, f64>) -> Result<(), InvariantSubspaceError> {
    let (rows, cols) = matrix.dim();
    if rows == 0 || rows != cols {
        return Err(InvariantSubspaceError::NotSquare { rows, cols });
    }
    if let Some(((row, col), _)) = matrix.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(InvariantSubspaceError::NonFinite { row, col });
    }
    Ok(())
}

/// Eigenvalue estimates `(re, im)` from gam-linalg's certified general
/// eigendecomposition, conjugate pairs adjacent with the positive imaginary part
/// first. Non-finite input is refused before faer, and a refusal of the owner's
/// certificate arrives as [`InvariantSubspaceError::Linalg`].
fn eigenvalue_estimates(
    matrix: ArrayView2<'_, f64>,
) -> Result<Vec<(f64, f64)>, InvariantSubspaceError> {
    let spectrum = real_general_spectrum(&matrix).map_err(InvariantSubspaceError::Linalg)?;
    Ok(spectrum
        .re
        .iter()
        .copied()
        .zip(spectrum.im.iter().copied())
        .collect())
}

fn initial_clusters(estimates: &[(f64, f64)]) -> Result<Vec<Vec<usize>>, InvariantSubspaceError> {
    let mut clusters = Vec::new();
    let mut index = 0;
    while index < estimates.len() {
        let (real, imaginary) = estimates[index];
        if imaginary == 0.0 {
            clusters.push(vec![index]);
            index += 1;
            continue;
        }
        match estimates.get(index + 1) {
            Some(&(partner_real, partner_imaginary))
                if partner_real == real && partner_imaginary == -imaginary =>
            {
                clusters.push(vec![index, index + 1]);
                index += 2;
            }
            _ => return Err(InvariantSubspaceError::UnpairedComplexEigenvalue { index }),
        }
    }
    Ok(clusters)
}

/// Try the proposals `p(T)^m` of one cluster. Returns the first certified one with
/// its kind, or `None` with the smallest `2 sqrt(ε_sim η)` over the attempts.
fn propose_and_certify(
    matrix: ArrayView2<'_, f64>,
    estimates: &[(f64, f64)],
    cluster: &[usize],
) -> Result<(Option<(InvariantSubspaceCertificate, InvariantBlockKind)>, f64), InvariantSubspaceError>
{
    let dimension = matrix.nrows();
    let columns = cluster.len();
    let count = columns as f64;
    let all_real = cluster.iter().all(|&index| estimates[index].1 == 0.0);
    let mean_real = cluster.iter().map(|&index| estimates[index].0).sum::<f64>() / count;
    let (polynomial, degree) = if all_real {
        let mut shifted = matrix.to_owned();
        for index in 0..dimension {
            shifted[[index, index]] -= mean_real;
        }
        (shifted, 1)
    } else {
        let mean_square = cluster
            .iter()
            .map(|&index| estimates[index].0.powi(2) + estimates[index].1.powi(2))
            .sum::<f64>()
            / count;
        let mut quadratic = matrix.dot(&matrix) - &matrix.mapv(|value| 2.0 * mean_real * value);
        for index in 0..dimension {
            quadratic[[index, index]] += mean_square;
        }
        (quadratic, 2)
    };
    let mut power = polynomial.clone();
    let mut exponent = 1;
    let mut reach_floor = f64::INFINITY;
    loop {
        let frame = proposal_frame(&power)?;
        let certificate = certify_frame(matrix, &frame, columns)?;
        match certificate.verdict {
            SubspaceVerdict::Certified {
                restriction_error, ..
            } => {
                let kind = block_kind(&certificate.restriction, restriction_error);
                return Ok((Some((certificate, kind)), 0.0));
            }
            SubspaceVerdict::NotSeparated { .. } => {
                reach_floor = reach_floor
                    .min(2.0 * (certificate.similarity_error * certificate.coupling).sqrt());
            }
        }
        if exponent * degree >= columns {
            return Ok((None, reach_floor));
        }
        power = power.dot(&polynomial);
        exponent += 1;
    }
}

/// The right singular vectors of `power` in increasing singular-value order, as the
/// columns of an `n x n` frame. A zero `power` has every vector in its null space,
/// and its Householder coefficients are undefined, so it proposes the identity. A
/// non-finite frame is refused, never certified.
fn proposal_frame(power: &Array2<f64>) -> Result<Array2<f64>, InvariantSubspaceError> {
    let dimension = power.nrows();
    if power.iter().all(|&value| value == 0.0) {
        return Ok(Array2::<f64>::eye(dimension));
    }
    let (_, singular_values, right) = power
        .svd(false, true)
        .map_err(InvariantSubspaceError::Linalg)?;
    let right = right.ok_or(InvariantSubspaceError::Linalg(
        FaerLinalgError::SvdNoConvergence {
            context: "invariant-subspace proposal: right singular vectors",
        },
    ))?;
    if right.iter().any(|value| !value.is_finite()) {
        return Err(InvariantSubspaceError::Linalg(
            FaerLinalgError::SvdNoConvergence {
                context: "invariant-subspace proposal: non-finite right singular vectors",
            },
        ));
    }
    let mut order: Vec<usize> = (0..singular_values.len()).collect();
    order.sort_by(|&left, &right_index| {
        singular_values[left].total_cmp(&singular_values[right_index])
    });
    let mut frame = Array2::<f64>::zeros((dimension, dimension));
    for (column, &index) in order.iter().enumerate() {
        frame.column_mut(column).assign(&right.row(index));
    }
    Ok(frame)
}

/// Stewart's certificate for the leading `columns` columns of an orthogonal-up-to-
/// roundoff `frame` (`n x n`).
fn certify_frame(
    matrix: ArrayView2<'_, f64>,
    frame: &Array2<f64>,
    columns: usize,
) -> Result<InvariantSubspaceCertificate, InvariantSubspaceError> {
    let dimension = matrix.nrows();
    let absolute_frame = frame.mapv(f64::abs);
    // `||Q - Q_o||_2 = max |σ_i - 1| <= ||Q^T Q - I||_F`, plus the rounding of the
    // Gram and of subtracting the identity: `γ_{n+1}` times each entry's absolute
    // term sum.
    let gram_defect = frame.t().dot(frame) - Array2::<f64>::eye(dimension);
    let frame_defect = frobenius_norm(gram_defect.view())
        + accumulation_growth(dimension + 1)
            * frobenius_norm(absolute_frame.t().dot(&absolute_frame).view());
    let reduced = frame.t().dot(&matrix.dot(frame));
    // Two nested length-`n` accumulations per entry of `Q^T T Q`.
    let formation_band = accumulation_growth(2 * dimension)
        * frobenius_norm(
            absolute_frame
                .t()
                .dot(&matrix.mapv(f64::abs).dot(&absolute_frame))
                .view(),
        );
    // `||Q^T T Q - Q_o^T T Q_o||_F <= ||Q - Q_o||_2 ||T||_F (||Q||_2 + ||Q_o||_2)`.
    let similarity_error =
        frame_defect * (2.0 + frame_defect) * frobenius_norm(matrix) + formation_band;
    let restriction = reduced.slice(s![..columns, ..columns]).to_owned();
    let residual = frobenius_norm(reduced.slice(s![columns.., ..columns])) + similarity_error;
    let coupling = frobenius_norm(reduced.slice(s![..columns, columns..])) + similarity_error;
    // `||Q_1 Q_1^T - Q_o1 Q_o1^T||_2 <= ||Q_1 - Q_o1||_2 (||Q_1||_2 + ||Q_o1||_2)`.
    let frame_projector_error = frame_defect * (2.0 + frame_defect);
    let basis = frame.slice(s![.., ..columns]).to_owned();
    if columns == dimension {
        return Ok(InvariantSubspaceCertificate {
            basis,
            restriction,
            residual,
            coupling,
            separation: f64::INFINITY,
            frame_defect,
            similarity_error,
            verdict: SubspaceVerdict::Certified {
                projector_bar: frame_projector_error,
                restriction_error: similarity_error,
            },
        });
    }
    let separation =
        kronecker_separation(&reduced, columns)? - 2.0 * similarity_error;
    let verdict = if separation > 0.0 && 4.0 * residual * coupling < separation * separation {
        let tangent_bar = 2.0 * residual / separation;
        SubspaceVerdict::Certified {
            projector_bar: tangent_bar + frame_projector_error,
            restriction_error: similarity_error + coupling * tangent_bar,
        }
    } else {
        SubspaceVerdict::NotSeparated {
            required_separation: 2.0 * (residual * coupling).sqrt(),
        }
    };
    Ok(InvariantSubspaceCertificate {
        basis,
        restriction,
        residual,
        coupling,
        separation,
        frame_defect,
        similarity_error,
        verdict,
    })
}

/// Lower bound on `sep_F(M_11, M_22)` of the computed blocks: the smallest singular
/// value of `I_r ⊗ M_11 - M_22^T ⊗ I_k` (column-major `vec`), less the SVD owner's
/// band and the rounding of the diagonal entries `m_11[a, a] - m_22[j, j]`. Every
/// other entry is a copy or a negation, which is exact.
///
/// The SVD's backward error is gam-linalg's `factor_singular_band`,
/// `max(m, n) ε σ_max`: the LAPACK convention, not a bound counted for faer 0.24.
/// This bound, and so every `Certified` verdict, rests on that declared assumption
/// (#2951, mpd-verify batch 22 NOTE 2 and batch 23's SPEC-tension note). A counted
/// band or an a posteriori SVD certificate is an owner change in gam-linalg.
fn kronecker_separation(reduced: &Array2<f64>, columns: usize) -> Result<f64, InvariantSubspaceError> {
    let dimension = reduced.nrows();
    let complement = dimension - columns;
    let size = columns * complement;
    let mut kronecker = Array2::<f64>::zeros((size, size));
    let mut diagonal_band_squared = 0.0;
    for col_j in 0..complement {
        for row_a in 0..columns {
            let target = row_a + columns * col_j;
            for row_b in 0..columns {
                if row_b != row_a {
                    kronecker[[target, row_b + columns * col_j]] = reduced[[row_a, row_b]];
                }
            }
            for col_l in 0..complement {
                if col_l != col_j {
                    kronecker[[target, row_a + columns * col_l]] =
                        -reduced[[columns + col_l, columns + col_j]];
                }
            }
            let (leading, trailing) = (
                reduced[[row_a, row_a]],
                reduced[[columns + col_j, columns + col_j]],
            );
            kronecker[[target, target]] = leading - trailing;
            diagonal_band_squared += (leading.abs() + trailing.abs()).powi(2);
        }
    }
    let (_, singular_values, _) = kronecker
        .svd(false, false)
        .map_err(InvariantSubspaceError::Linalg)?;
    let sigma_min = singular_values
        .iter()
        .fold(f64::INFINITY, |acc, &value| acc.min(value));
    let sigma_max = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value));
    Ok(sigma_min
        - factor_singular_band(size, size, sigma_max)
        - accumulation_growth(1) * diagonal_band_squared.sqrt())
}

fn block_kind(restriction: &Array2<f64>, restriction_error: f64) -> InvariantBlockKind {
    match restriction.nrows() {
        1 => {
            let value = restriction[[0, 0]];
            let absolute_sum = value.abs() + restriction_error;
            InvariantBlockKind::Real {
                eigenvalue_interval: (
                    round_down(value - restriction_error, absolute_sum, 1),
                    round_up(value + restriction_error, absolute_sum, 1),
                ),
            }
        }
        2 => plane_kind(restriction, restriction_error),
        dimension => InvariantBlockKind::Repeated { dimension },
    }
}

/// The certified sign of a 2x2 restriction's discriminant (module documentation).
fn plane_kind(restriction: &Array2<f64>, restriction_error: f64) -> InvariantBlockKind {
    let (first, upper, lower, second) = (
        restriction[[0, 0]],
        restriction[[0, 1]],
        restriction[[1, 0]],
        restriction[[1, 1]],
    );
    let trace = first + second;
    let determinant = first * second - upper * lower;
    // `|tr E| <= sqrt(2) ||E||_F`; `|det(L + E) - det L| <= ||adj L||_F ||E||_F +
    // ||E||_F^2 / 2` with `||adj L||_F = ||L||_F`; plus the rounding of forming each.
    // Every bound below is rounded outward from exact-arithmetic bounds, so a certified
    // endpoint or sign holds for the exact restriction, not only to within an ulp.
    // `SQRT_2` exceeds `sqrt(2)`.
    let trace_error = {
        let value = SQRT_2 * restriction_error + accumulation_band(1, first.abs() + second.abs());
        // `sqrt(2) ε`, the band's `γ_1`, its absolute sum and product, and the sum.
        round_up(value, value, 6)
    };
    let determinant_error = {
        let value = frobenius_norm(restriction.view()) * restriction_error
            + 0.5 * restriction_error * restriction_error
            + accumulation_band(2, (first * second).abs() + (upper * lower).abs());
        // The Frobenius norm (4 squares, 3 sums, 1 root), its product with `ε`, `ε^2`,
        // the band's `γ_2`, two products, their sum and the band product, and two sums.
        round_up(value, value, 18)
    };
    let magnitude = trace.abs();
    let trace_high = round_up(magnitude + trace_error, magnitude + trace_error, 1);
    let trace_low = round_down(magnitude - trace_error, magnitude + trace_error, 1).max(0.0);
    let determinant_absolute_sum = determinant.abs() + determinant_error;
    let determinant_low = round_down(
        determinant - determinant_error,
        determinant_absolute_sum,
        1,
    );
    let determinant_high = round_up(
        determinant + determinant_error,
        determinant_absolute_sum,
        1,
    );
    let square_high = round_up(trace_high * trace_high, trace_high * trace_high, 2);
    let square_low = round_down(trace_low * trace_low, trace_low * trace_low, 2);
    // `t^2 - 4 d` over `|t| in [trace_low, trace_high]`, `d in [determinant_low,
    // determinant_high]`; `4 d` scales exactly.
    let discriminant_upper = round_up(
        square_high - 4.0 * determinant_low,
        square_high + 4.0 * determinant_low.abs(),
        1,
    );
    let discriminant_lower = round_down(
        square_low - 4.0 * determinant_high,
        square_low + 4.0 * determinant_high.abs(),
        1,
    );
    if discriminant_upper < 0.0 {
        // `discriminant_upper < 0` gives `4 determinant_low > trace_high^2 >= 0`, so both
        // square roots are of positive numbers; a correctly rounded root moves by at most
        // half an ulp.
        let modulus_interval = (
            determinant_low.sqrt().next_down(),
            determinant_high.sqrt().next_up(),
        );
        // `t / (2 sqrt d)` is monotone in `t` and, for a fixed sign of `t`, in `d`, so its
        // extremes over the outward box are at its corners. Each corner rounds in the
        // root and the division (`2 sqrt d` doubles exactly), and its band is taken on the
        // rounded quotient, which costs one more counted operation.
        let corner_traces = [
            round_down(trace - trace_error, magnitude + trace_error, 1),
            round_up(trace + trace_error, magnitude + trace_error, 1),
        ];
        let mut cosine_low = f64::INFINITY;
        let mut cosine_high = f64::NEG_INFINITY;
        for corner_trace in corner_traces {
            for corner_determinant in [determinant_low, determinant_high] {
                let cosine = corner_trace / (2.0 * corner_determinant.sqrt());
                cosine_low = cosine_low.min(round_down(cosine, cosine.abs(), 3));
                cosine_high = cosine_high.max(round_up(cosine, cosine.abs(), 3));
            }
        }
        let half_trace = 0.5 * trace;
        let sine_part = 0.5 * (4.0 * determinant - trace * trace).sqrt();
        let mut complex_structure = restriction.clone();
        complex_structure[[0, 0]] -= half_trace;
        complex_structure[[1, 1]] -= half_trace;
        complex_structure.mapv_inplace(|value| value / sine_part);
        InvariantBlockKind::RotationScaling {
            modulus_interval,
            cosine_interval: (cosine_low.max(-1.0), cosine_high.min(1.0)),
            angle: sine_part.atan2(half_trace),
            complex_structure,
        }
    } else if discriminant_lower > 0.0 {
        let root = 0.5 * (trace * trace - 4.0 * determinant).sqrt();
        InvariantBlockKind::RealPair {
            eigenvalues: (0.5 * trace - root, 0.5 * trace + root),
        }
    } else {
        InvariantBlockKind::Unresolved
    }
}

/// `value` less the counted rounding of the `operations` round-to-nearest steps that
/// formed it, `γ_k Σ|terms|` (Higham, Lemma 3.1), then one ulp further down for the
/// subtraction itself. `absolute_sum` majorizes the exact terms.
fn round_down(value: f64, absolute_sum: f64, operations: usize) -> f64 {
    (value - accumulation_band(operations, absolute_sum)).next_down()
}

/// The upward counterpart of [`round_down`].
fn round_up(value: f64, absolute_sum: f64, operations: usize) -> f64 {
    (value + accumulation_band(operations, absolute_sum)).next_up()
}

fn cluster_distance(estimates: &[(f64, f64)], left: &[usize], right: &[usize]) -> f64 {
    left.iter()
        .flat_map(|&first| {
            right.iter().map(move |&second| {
                (estimates[first].0 - estimates[second].0)
                    .hypot(estimates[first].1 - estimates[second].1)
            })
        })
        .fold(f64::INFINITY, f64::min)
}

fn mean_real_part(block: &InvariantBlock) -> f64 {
    block
        .eigenvalue_estimates
        .iter()
        .map(|estimate| estimate.0)
        .sum::<f64>()
        / block.eigenvalue_estimates.len() as f64
}

fn frobenius_norm(matrix: ArrayView2<'_, f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const DIMENSION: usize = 8;

    /// `G = I + N`, with ones on the first superdiagonal of `N` and minus ones on the
    /// second, and its inverse by integer back-substitution.
    fn unipotent_frame() -> (Array2<f64>, Array2<f64>) {
        let mut frame = Array2::<f64>::eye(DIMENSION);
        for index in 0..DIMENSION {
            if index + 1 < DIMENSION {
                frame[[index, index + 1]] = 1.0;
            }
            if index + 2 < DIMENSION {
                frame[[index, index + 2]] = -1.0;
            }
        }
        let mut inverse = Array2::<f64>::zeros((DIMENSION, DIMENSION));
        for col in 0..DIMENSION {
            for row in (0..DIMENSION).rev() {
                let mut value = if row == col { 1.0 } else { 0.0 };
                for later in row + 1..DIMENSION {
                    value -= frame[[row, later]] * inverse[[later, col]];
                }
                inverse[[row, col]] = value;
            }
        }
        (frame, inverse)
    }

    /// `T = G B G^{-1}`. Every entry of `G`, `G^{-1}` and `B` is a small integer or
    /// a multiple of 1/8, so every product and sum is exact and `T` is the planted
    /// operator itself: `G G^{-1} = I` and `T G = G B` are pinned bitwise.
    fn plant(form: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let (frame, inverse) = unipotent_frame();
        assert_eq!(frame.dot(&inverse), Array2::<f64>::eye(DIMENSION));
        let operator = frame.dot(form).dot(&inverse);
        assert_eq!(operator.dot(&frame), frame.dot(form));
        (operator, frame)
    }

    /// `[[real, -imaginary], [imaginary, real]]` at `offset`.
    fn set_plane(form: &mut Array2<f64>, offset: usize, real: f64, imaginary: f64) {
        form[[offset, offset]] = real;
        form[[offset, offset + 1]] = -imaginary;
        form[[offset + 1, offset]] = imaginary;
        form[[offset + 1, offset + 1]] = real;
    }

    /// `||(I - Q_1 Q_1^T) G_k||_F` less its formation band, against `bar ||G_k||_F`.
    /// When `span G_k` is the certified subspace `V`, `P_V G_k = G_k` and
    /// `||Q_1 Q_1^T - P_V||_2 <= bar` make the first at most the second.
    fn span_excess(
        certificate: &InvariantSubspaceCertificate,
        truth: ArrayView2<'_, f64>,
    ) -> (f64, f64) {
        let bar = match certificate.verdict {
            SubspaceVerdict::Certified { projector_bar, .. } => projector_bar,
            ref other => panic!("expected a certified subspace, got {other:?}"),
        };
        let basis = &certificate.basis;
        let residual = &truth - &basis.dot(&basis.t().dot(&truth));
        let absolute_basis = basis.mapv(f64::abs);
        let absolute_truth = truth.mapv(f64::abs);
        // `n` products summed, then `k`, then one subtraction per entry.
        let band = accumulation_growth(basis.nrows() + basis.ncols() + 1)
            * frobenius_norm(
                (absolute_basis.dot(&absolute_basis.t().dot(&absolute_truth)) + &absolute_truth)
                    .view(),
            );
        (
            frobenius_norm(residual.view()) - band,
            bar * frobenius_norm(truth),
        )
    }

    fn assert_spans(certificate: &InvariantSubspaceCertificate, truth: ArrayView2<'_, f64>) {
        let (measured, allowed) = span_excess(certificate, truth);
        assert!(
            measured <= allowed,
            "planted columns leave the certified subspace by {measured:.3e}, above {allowed:.3e}"
        );
    }

    /// A certified rotation-scaling whose intervals admit `real ± i imaginary`, which
    /// spans the planted columns `first..first + 2`, and whose `J` carries the planted
    /// orientation.
    fn assert_rotation(
        certificate: &InvariantSubspaceCertificate,
        kind: &InvariantBlockKind,
        frame: &Array2<f64>,
        first: usize,
        real: f64,
        imaginary: f64,
    ) {
        let (modulus_interval, cosine_interval, angle, structure) = match kind {
            InvariantBlockKind::RotationScaling {
                modulus_interval,
                cosine_interval,
                angle,
                complex_structure,
            } => (*modulus_interval, *cosine_interval, *angle, complex_structure),
            other => panic!("expected a rotation-scaling, got {other:?}"),
        };
        // `real^2 + imaginary^2` is exact, and the correctly rounded root is within half
        // an ulp of the exact modulus. The cosine carries the root and the division, with
        // its band taken on the rounded quotient. Each truth enclosure must lie inside its
        // reported interval, which proves the exact value does; no slack is added to the
        // intervals themselves.
        let modulus = (real * real + imaginary * imaginary).sqrt();
        let modulus_truth = (modulus.next_down(), modulus.next_up());
        let cosine = real / modulus;
        let cosine_truth = (
            round_down(cosine, cosine.abs(), 3),
            round_up(cosine, cosine.abs(), 3),
        );
        assert!(
            modulus_interval.0 <= modulus_truth.0 && modulus_truth.1 <= modulus_interval.1,
            "modulus enclosure {modulus_truth:?} outside {modulus_interval:?}"
        );
        assert!(
            cosine_interval.0 <= cosine_truth.0 && cosine_truth.1 <= cosine_interval.1,
            "cosine enclosure {cosine_truth:?} outside {cosine_interval:?}"
        );
        assert!(angle > 0.0 && angle < PI, "angle {angle} outside (0, pi)");
        assert_spans(certificate, frame.slice(s![.., first..first + 2]));
        // `T G_k = G_k (real I + imaginary J_std)` with `imaginary > 0`, so the
        // intrinsic `J` maps the first planted column to the second.
        let basis = &certificate.basis;
        let mapped = basis.dot(&structure.dot(&basis.t().dot(&frame.column(first))));
        let alignment = mapped.dot(&frame.column(first + 1));
        assert!(alignment > 0.0, "orientation alignment {alignment}");
    }

    fn two_planes_fixed_space_and_an_axis() -> Array2<f64> {
        let mut form = Array2::<f64>::zeros((DIMENSION, DIMENSION));
        set_plane(&mut form, 0, 0.5, 0.75);
        set_plane(&mut form, 2, -0.625, 0.25);
        for axis in 4..7 {
            form[[axis, axis]] = 1.0;
        }
        form[[7, 7]] = -1.5;
        form
    }

    #[test]
    fn planted_non_orthogonal_planes_are_certified_with_angle_and_orientation_2951() {
        let (operator, frame) = plant(&two_planes_fixed_space_and_an_axis());
        // Far from orthogonal, so P3's symmetric-part route does not apply.
        let gram_defect = operator.t().dot(&operator) - Array2::<f64>::eye(DIMENSION);
        assert!(frobenius_norm(gram_defect.view()) > 1.0);

        let recovery = recover_invariant_blocks(operator.view()).expect("recovery");
        assert_eq!(recovery.blocks.len(), 4, "{recovery:?}");
        let axis = &recovery.blocks[0];
        match axis.kind {
            InvariantBlockKind::Real {
                eigenvalue_interval,
            } => assert!(
                eigenvalue_interval.0 <= -1.5 && -1.5 <= eigenvalue_interval.1,
                "eigenvalue -1.5 outside {eigenvalue_interval:?}"
            ),
            ref other => panic!("expected a real block, got {other:?}"),
        }
        assert_spans(&axis.certificate, frame.slice(s![.., 7..8]));
        let (slow, fast) = (&recovery.blocks[1], &recovery.blocks[2]);
        assert_rotation(&slow.certificate, &slow.kind, &frame, 2, -0.625, 0.25);
        assert_rotation(&fast.certificate, &fast.kind, &frame, 0, 0.5, 0.75);
        let fixed = &recovery.blocks[3];
        assert!(
            matches!(fixed.kind, InvariantBlockKind::Repeated { dimension: 3 }),
            "{fixed:?}"
        );
        assert_spans(&fixed.certificate, frame.slice(s![.., 4..7]));
        assert_eq!(
            recovery.ambiguities(),
            vec![
                InvariantAmbiguity::Repeated {
                    block: 3,
                    dimension: 3
                },
                InvariantAmbiguity::Winding
            ]
        );
    }

    #[test]
    fn a_defective_block_is_refused_where_a_complex_pair_is_certified_2951() {
        let mut form = Array2::<f64>::zeros((DIMENSION, DIMENSION));
        form[[0, 0]] = 0.5;
        form[[0, 1]] = 1.0;
        form[[1, 1]] = 0.5;
        form[[2, 2]] = 2.0;
        form[[3, 3]] = -1.0;
        set_plane(&mut form, 4, 0.25, 0.5);
        form[[6, 6]] = 3.0;
        form[[7, 7]] = -2.5;
        let (operator, frame) = plant(&form);
        let recovery = recover_invariant_blocks(operator.view()).expect("recovery");
        assert_eq!(recovery.blocks.len(), 6, "{recovery:?}");
        let jordan = &recovery.blocks[3];
        assert!(
            matches!(jordan.kind, InvariantBlockKind::Unresolved),
            "{jordan:?}"
        );
        assert_spans(&jordan.certificate, frame.slice(s![.., 0..2]));
        let plane = &recovery.blocks[2];
        assert_rotation(&plane.certificate, &plane.kind, &frame, 4, 0.25, 0.5);
        assert_eq!(
            recovery.ambiguities(),
            vec![
                InvariantAmbiguity::Unresolved { block: 3 },
                InvariantAmbiguity::Winding
            ]
        );

        // Positive control: the same position holding `[[0.5, 1], [-0.25, 0.5]]`,
        // the pair 0.5 ± 0.5i, is a certified rotation-scaling.
        let mut paired = form.clone();
        paired[[1, 0]] = -0.25;
        let (operator, frame) = plant(&paired);
        let recovery = recover_invariant_blocks(operator.view()).expect("recovery");
        assert_eq!(recovery.blocks.len(), 6, "{recovery:?}");
        let pair = &recovery.blocks[3];
        match &pair.kind {
            InvariantBlockKind::RotationScaling {
                modulus_interval,
                cosine_interval,
                ..
            } => {
                // `0.5^2 + 0.5^2 = 0.5` exactly. Enclose the exact modulus and cosine as in
                // `assert_rotation`, and require the enclosures inside the intervals.
                let modulus = 0.5_f64.sqrt();
                let modulus_truth = (modulus.next_down(), modulus.next_up());
                let cosine = 0.5 / modulus;
                let cosine_truth = (
                    round_down(cosine, cosine.abs(), 3),
                    round_up(cosine, cosine.abs(), 3),
                );
                assert!(
                    modulus_interval.0 <= modulus_truth.0 && modulus_truth.1 <= modulus_interval.1,
                    "modulus enclosure {modulus_truth:?} outside {modulus_interval:?}"
                );
                assert!(
                    cosine_interval.0 <= cosine_truth.0 && cosine_truth.1 <= cosine_interval.1,
                    "cosine enclosure {cosine_truth:?} outside {cosine_interval:?}"
                );
            }
            other => panic!("expected a rotation-scaling, got {other:?}"),
        }
        assert_spans(&pair.certificate, frame.slice(s![.., 0..2]));
        assert_eq!(recovery.ambiguities(), vec![InvariantAmbiguity::Winding]);
    }

    #[test]
    fn repeated_planes_report_the_invariant_subspace_not_the_planes_2951() {
        let mut form = Array2::<f64>::zeros((DIMENSION, DIMENSION));
        set_plane(&mut form, 0, 0.5, 0.75);
        set_plane(&mut form, 2, 0.5, 0.75);
        form[[4, 4]] = 2.0;
        form[[5, 5]] = -1.0;
        set_plane(&mut form, 6, -0.625, 0.25);
        let (operator, frame) = plant(&form);
        let recovery = recover_invariant_blocks(operator.view()).expect("recovery");
        assert_eq!(recovery.blocks.len(), 4, "{recovery:?}");
        let repeated = &recovery.blocks[2];
        assert!(
            matches!(repeated.kind, InvariantBlockKind::Repeated { dimension: 4 }),
            "{repeated:?}"
        );
        assert_spans(&repeated.certificate, frame.slice(s![.., 0..4]));
        let plane = &recovery.blocks[1];
        assert_rotation(&plane.certificate, &plane.kind, &frame, 6, -0.625, 0.25);
        assert_eq!(
            recovery.ambiguities(),
            vec![
                InvariantAmbiguity::Repeated {
                    block: 2,
                    dimension: 4
                },
                InvariantAmbiguity::Winding
            ]
        );

        // The identity has no plane: one block spanning the whole space.
        let identity = Array2::<f64>::eye(DIMENSION);
        let recovery = recover_invariant_blocks(identity.view()).expect("recovery");
        assert_eq!(recovery.blocks.len(), 1, "{recovery:?}");
        let whole = &recovery.blocks[0].certificate;
        assert!(whole.separation.is_infinite());
        match whole.verdict {
            SubspaceVerdict::Certified {
                projector_bar,
                restriction_error,
            } => assert!(
                projector_bar.is_finite() && restriction_error.is_finite(),
                "whole-space bars {projector_bar}, {restriction_error}"
            ),
            ref other => panic!("expected a certified whole space, got {other:?}"),
        }
        assert_eq!(
            recovery.ambiguities(),
            vec![InvariantAmbiguity::Repeated {
                block: 0,
                dimension: DIMENSION
            }]
        );
    }

    #[test]
    fn a_candidate_plane_is_certified_only_when_it_is_invariant_2951() {
        let (operator, frame) = plant(&two_planes_fixed_space_and_an_axis());
        let certificate = certify_invariant_subspace(operator.view(), frame.slice(s![.., 0..2]))
            .expect("certificate");
        let kind = certificate.kind().expect("the planted plane is certified");
        assert_rotation(&certificate, &kind, &frame, 0, 0.5, 0.75);
        // Negative control of the span check: the other plane is not in this one.
        let (measured, allowed) = span_excess(&certificate, frame.slice(s![.., 2..4]));
        assert!(
            measured > allowed,
            "the other plane passes the span check: {measured:.3e} <= {allowed:.3e}"
        );

        // A plane mixing the two planted planes is not invariant.
        let mut mixed = Array2::<f64>::zeros((DIMENSION, 2));
        mixed.column_mut(0).assign(&frame.column(0));
        mixed.column_mut(1).assign(&frame.column(2));
        let refused =
            certify_invariant_subspace(operator.view(), mixed.view()).expect("certificate");
        assert!(
            matches!(refused.verdict, SubspaceVerdict::NotSeparated { .. }),
            "{refused:?}"
        );
        assert!(refused.kind().is_none());

        // A rank-deficient candidate is refused before any claim.
        let mut dependent = mixed.clone();
        dependent
            .column_mut(1)
            .assign(&frame.column(0).mapv(|value| 2.0 * value));
        assert!(matches!(
            certify_invariant_subspace(operator.view(), dependent.view()),
            Err(InvariantSubspaceError::RankDeficientBasis {
                resolved: 1,
                columns: 2
            })
        ));
    }

    #[test]
    fn outward_rounding_keeps_an_endpoint_that_round_to_nearest_cuts_off_2951() {
        // `2^-60` is below half an ulp of 1.0, so round-to-nearest returns exactly 1.0 for
        // `1 - e` and `1 + e`: both nearest endpoints cut off the exact bounds.
        let e = 2.0_f64.powi(-60);
        assert_eq!(1.0 - e, 1.0);
        assert_eq!(1.0 + e, 1.0);
        // Outward rounding reaches the neighbouring floats, which enclose `1 ± e`.
        let lower = round_down(1.0 - e, 1.0 + e, 1);
        let upper = round_up(1.0 + e, 1.0 + e, 1);
        assert!(lower < 1.0 && upper > 1.0, "outward endpoints {lower}, {upper}");
        // A 1x1 restriction [1.0] with restriction error 2^-60.
        match block_kind(&Array2::<f64>::from_elem((1, 1), 1.0), e) {
            InvariantBlockKind::Real {
                eigenvalue_interval,
            } => assert!(
                eigenvalue_interval.0 < 1.0 && eigenvalue_interval.1 > 1.0,
                "interval {eigenvalue_interval:?} cuts off 1 ± 2^-60"
            ),
            other => panic!("expected a real block, got {other:?}"),
        }
    }
}
