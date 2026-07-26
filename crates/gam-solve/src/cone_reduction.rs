//! Geometry of a cone-truncated Laplace posterior whose reduced precision is
//! INDEFINITE.
//!
//! # The object
//!
//! After the `#2442` reparameterization a location-scale fit with an indefinite
//! ambient Hessian leaves
//!
//! ```text
//! π(w) ∝ exp(−½ wᵀ M w) · 1{w ≥ ℓ},     In(M) = (n−1, 0, 1)
//! ```
//!
//! `M` is not a precision matrix: it has exactly one negative eigenvalue, so
//! `M⁻¹` is not a covariance and this law is NOT a truncated Gaussian. It is
//! normalizable exactly when `M` is strictly copositive on the nonnegative
//! orthant, because along every feasible ray `d ≥ 0` the exponent grows like
//! `½ t² dᵀMd` and copositivity is the statement that `dᵀMd > 0` there.
//!
//! # Why the origin has to move
//!
//! Everything downstream is expressed as an offset from the CONSTRAINED MODE,
//! not from `w = 0` or from `w = ℓ`. That is not presentation. On the fixture
//! this module was built against, the ambient centre lies outside the feasible
//! set and the integrand's peak over that set is `exp(−513.82)`; carried in the
//! reduction's natural origin, every downstream conditional sits about thirty
//! posterior standard deviations outside the feasible region, and a cubature
//! asked for such a probability returns a number that climbs monotonically with
//! its node count instead of converging (measured: +74 log units from `2¹⁰` to
//! `2¹⁸` nodes, still climbing). Re-centred, the same quantities are ordinary.
//!
//! # Everything here is exact
//!
//! Both searches are finite face enumerations rather than iterative solves:
//!
//! * `min wᵀMw` over the simplex is attained at a stationary point in the
//!   relative interior of some face, so enumerating all `2ⁿ − 1` supports plus
//!   the vertices decides copositivity exactly;
//! * the constrained minimiser of `½xᵀMx + cᵀx` over `x ≥ 0` satisfies, on its
//!   free set `F`, `M_FF x_F = −c_F` with `x_F ≥ 0`, `(Mx + c)_A ≥ 0` on the
//!   active set, and `M_FF ⪰ 0`; enumerating supports and keeping the feasible
//!   KKT point of least value is therefore exact.
//!
//! Exactness is worth the `2ⁿ` because `n` is the number of RETAINED constraint
//! rows — six on the motivating fixture — and because the multiplier vector
//! `g = Mx* + c` is consumed downstream, where an optimiser's tolerance would
//! become the quadrature's error floor.

use ndarray::{Array1, Array2, ArrayView2};

/// Inertia `(positive, zero, negative)` of a symmetric matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inertia {
    pub positive: usize,
    pub zero: usize,
    pub negative: usize,
}

/// The constrained mode of `½xᵀMx + cᵀx` over the nonnegative orthant.
#[derive(Clone, Debug)]
pub struct ConeMode {
    /// The minimiser `x*`.
    pub point: Array1<f64>,
    /// `φ* = ½x*ᵀMx* + cᵀx*`.
    pub value: f64,
    /// `g = Mx* + c`. Zero on the free set and non-negative on the active set —
    /// these are the KKT multipliers, and the downstream quadrature integrates
    /// `exp(−½dᵀMd − gᵀd)` over `{d ≥ −x*}`.
    pub gradient: Array1<f64>,
    /// Indices with `x*_j > 0`.
    pub free: Vec<usize>,
}

/// Symmetric inertia by `LDLᵀ` with symmetric pivoting.
///
/// Sylvester's law of inertia makes the pivot signs the inertia, so this needs
/// no eigensolver. Diagonal pivoting keeps it well posed for the indefinite
/// case, which is the case this module exists for.
pub fn symmetric_inertia(matrix: ArrayView2<'_, f64>, tolerance: f64) -> Result<Inertia, String> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(format!(
            "inertia needs a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut work = matrix.to_owned();
    let scale = work
        .iter()
        .fold(0.0f64, |worst, value| worst.max(value.abs()))
        .max(1.0);
    let floor = tolerance * scale;
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut inertia = Inertia {
        positive: 0,
        zero: 0,
        negative: 0,
    };
    while !remaining.is_empty() {
        // Pivot on the largest-magnitude remaining diagonal entry.
        let (position, &pivot_index) = remaining
            .iter()
            .enumerate()
            .max_by(|left, right| {
                work[[*left.1, *left.1]]
                    .abs()
                    .partial_cmp(&work[[*right.1, *right.1]].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| "inertia pivot selection found no candidate".to_string())?;
        let pivot = work[[pivot_index, pivot_index]];
        if !pivot.is_finite() {
            return Err(format!("inertia pivot {pivot_index} is not finite"));
        }
        if pivot.abs() <= floor {
            // The whole remaining block is numerically zero on its diagonal. A
            // nonzero off-diagonal here would be a 2x2 block; refuse rather
            // than guess, since callers of this module treat the inertia as a
            // certificate.
            for &i in &remaining {
                for &j in &remaining {
                    if i != j && work[[i, j]].abs() > floor {
                        return Err(format!(
                            "inertia needs a 2x2 pivot at ({i},{j}); the matrix is not \
                             diagonally pivotable at tolerance {tolerance:.3e}"
                        ));
                    }
                }
            }
            inertia.zero += remaining.len();
            break;
        }
        if pivot > 0.0 {
            inertia.positive += 1;
        } else {
            inertia.negative += 1;
        }
        remaining.remove(position);
        let rest = remaining.clone();
        for &i in &rest {
            let factor = work[[i, pivot_index]] / pivot;
            if factor == 0.0 {
                continue;
            }
            for &j in &rest {
                work[[i, j]] -= factor * work[[pivot_index, j]];
            }
        }
        for &i in &rest {
            work[[i, pivot_index]] = 0.0;
            work[[pivot_index, i]] = 0.0;
        }
    }
    Ok(inertia)
}

/// Solve `A y = b` for a small dense `A` by Gaussian elimination with partial
/// pivoting. Returns `None` when a pivot falls below the floor, which the
/// callers read as "this face is degenerate, skip it" rather than as an error —
/// a singular face carries no isolated stationary point to compare.
fn symmetric_solve(a: &Array2<f64>, b: &Array1<f64>, floor: f64) -> Option<Array1<f64>> {
    let n = a.nrows();
    let mut work = a.clone();
    let mut rhs = b.clone();
    for column in 0..n {
        let mut pivot_row = column;
        let mut best = work[[column, column]].abs();
        for row in (column + 1)..n {
            let candidate = work[[row, column]].abs();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if !best.is_finite() || best <= floor {
            return None;
        }
        if pivot_row != column {
            for j in 0..n {
                let swap = work[[column, j]];
                work[[column, j]] = work[[pivot_row, j]];
                work[[pivot_row, j]] = swap;
            }
            rhs.swap(column, pivot_row);
        }
        let pivot = work[[column, column]];
        for row in (column + 1)..n {
            let factor = work[[row, column]] / pivot;
            if factor == 0.0 {
                continue;
            }
            for j in column..n {
                work[[row, j]] -= factor * work[[column, j]];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = Array1::<f64>::zeros(n);
    for row in (0..n).rev() {
        let mut total = rhs[row];
        for column in (row + 1)..n {
            total -= work[[row, column]] * solution[column];
        }
        solution[row] = total / work[[row, row]];
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(solution)
}

/// Is `A` positive semidefinite, to the given relative floor?
fn is_positive_semidefinite(a: &Array2<f64>, tolerance: f64) -> bool {
    match symmetric_inertia(a.view(), tolerance) {
        Ok(inertia) => inertia.negative == 0,
        Err(_) => false,
    }
}

/// Exact minimum of `wᵀMw` over the unit simplex `{w ≥ 0, 1ᵀw = 1}`.
///
/// Strictly positive iff `M` is strictly copositive, which is exactly the
/// condition for `exp(−½wᵀMw)` to be normalizable on a shifted orthant. On the
/// face with support `S` the stationary value is `1/(1ᵀM_SS⁻¹1)`, so enumerating
/// all `2ⁿ − 1` supports and the vertices `M_jj` decides it — no nonconvex QP,
/// and a non-positive answer is a PROOF of impropriety rather than an
/// inconclusive bound.
pub fn copositive_simplex_minimum(
    matrix: ArrayView2<'_, f64>,
) -> Result<(f64, Array1<f64>), String> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(format!(
            "copositivity needs a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if n == 0 || n > 20 {
        return Err(format!(
            "exact copositivity enumerates 2^n faces and is meant for a retained \
             constraint face; n = {n} is out of range"
        ));
    }
    let owned = matrix.to_owned();
    let scale = owned
        .iter()
        .fold(0.0f64, |worst, value| worst.max(value.abs()))
        .max(1.0);
    let floor = 1e-12 * scale;
    let mut best = f64::INFINITY;
    let mut best_point = Array1::<f64>::zeros(n);
    for mask in 1u32..(1u32 << n) {
        let support: Vec<usize> = (0..n).filter(|j| mask & (1 << j) != 0).collect();
        let size = support.len();
        let mut block = Array2::<f64>::zeros((size, size));
        for (i, &row) in support.iter().enumerate() {
            for (j, &column) in support.iter().enumerate() {
                block[[i, j]] = owned[[row, column]];
            }
        }
        let ones = Array1::<f64>::ones(size);
        let Some(solution) = symmetric_solve(&block, &ones, floor) else {
            continue;
        };
        let total: f64 = solution.sum();
        if !total.is_finite() || total.abs() <= floor {
            continue;
        }
        let weights = &solution / total;
        if weights.iter().any(|value| *value <= 0.0) {
            continue;
        }
        let value = weights.dot(&block.dot(&weights));
        if value.is_finite() && value < best {
            best = value;
            best_point = Array1::zeros(n);
            for (i, &row) in support.iter().enumerate() {
                best_point[row] = weights[i];
            }
        }
    }
    for j in 0..n {
        if owned[[j, j]] < best {
            best = owned[[j, j]];
            best_point = Array1::zeros(n);
            best_point[j] = 1.0;
        }
    }
    if !best.is_finite() {
        return Err("copositivity enumeration produced no finite face value".to_string());
    }
    Ok((best, best_point))
}

/// Exact constrained mode of `φ(x) = ½xᵀMx + cᵀx` over `x ≥ 0`.
///
/// Requires `M` strictly copositive, which makes `φ` coercive on the orthant
/// (`xᵀMx ≥ μ|x|²` there with `μ > 0`) and therefore guarantees the minimum is
/// attained. Refuses otherwise, because without copositivity there is no
/// minimum to report and the cone-truncated law is improper.
pub fn constrained_cone_mode(
    matrix: ArrayView2<'_, f64>,
    linear: &Array1<f64>,
) -> Result<ConeMode, String> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(format!(
            "cone mode needs a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if linear.len() != n {
        return Err(format!(
            "cone mode: matrix is {n}x{n} but the linear term has length {}",
            linear.len()
        ));
    }
    if n > 20 {
        return Err(format!(
            "exact cone mode enumerates 2^n faces and is meant for a retained \
             constraint face; n = {n} is out of range"
        ));
    }
    let (copositive_minimum, _) = copositive_simplex_minimum(matrix)?;
    if copositive_minimum <= 0.0 {
        return Err(format!(
            "the reduced precision is not strictly copositive (min wᵀMw over the simplex \
             = {copositive_minimum:.9e}); the cone-truncated posterior is improper and has \
             no mode"
        ));
    }
    let owned = matrix.to_owned();
    let scale = owned
        .iter()
        .fold(0.0f64, |worst, value| worst.max(value.abs()))
        .max(1.0);
    let floor = 1e-12 * scale;
    let feasibility = 1e-9 * scale.sqrt().max(1.0);
    let gradient_floor = 1e-7
        * linear
            .iter()
            .fold(0.0f64, |worst, value| worst.max(value.abs()))
            .max(1.0);

    let mut best: Option<ConeMode> = None;
    for mask in 0u32..(1u32 << n) {
        let free: Vec<usize> = (0..n).filter(|j| mask & (1 << j) != 0).collect();
        let mut point = Array1::<f64>::zeros(n);
        if !free.is_empty() {
            let size = free.len();
            let mut block = Array2::<f64>::zeros((size, size));
            for (i, &row) in free.iter().enumerate() {
                for (j, &column) in free.iter().enumerate() {
                    block[[i, j]] = owned[[row, column]];
                }
            }
            // Second-order necessary condition on the face. A face whose block
            // has a negative direction carries no minimiser, and admitting it
            // would let a saddle win the enumeration.
            if !is_positive_semidefinite(&block, 1e-12) {
                continue;
            }
            let mut rhs = Array1::<f64>::zeros(size);
            for (i, &row) in free.iter().enumerate() {
                rhs[i] = -linear[row];
            }
            let Some(solution) = symmetric_solve(&block, &rhs, floor) else {
                continue;
            };
            if solution.iter().any(|value| *value < -feasibility) {
                continue;
            }
            for (i, &row) in free.iter().enumerate() {
                point[row] = solution[i].max(0.0);
            }
        }
        let gradient = owned.dot(&point) + linear;
        if (0..n)
            .filter(|j| mask & (1 << j) == 0)
            .any(|j| gradient[j] < -gradient_floor)
        {
            continue;
        }
        let value = 0.5 * point.dot(&owned.dot(&point)) + linear.dot(&point);
        if !value.is_finite() {
            continue;
        }
        if best.as_ref().is_none_or(|current| value < current.value) {
            best = Some(ConeMode {
                point,
                value,
                gradient,
                free: free.clone(),
            });
        }
    }
    best.ok_or_else(|| {
        "no feasible KKT point on any face; the cone mode enumeration found nothing".to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The reduced precision `M` of the refusing location-scale fixture on
    /// #2529, taken from the probe dump at `e23e674633b` (`p = 9`, `m = 6`,
    /// blocks `MU ⊕ LOG_SIGMA ⊕ WIGGLE`). Every constant asserted against it
    /// below was produced twice by two lanes on two independent methods.
    const FIXTURE_M: [[f64; 6]; 6] = [
        [2144.265169679624, 1715.134178122592, 1747.5745584612605, 935.098928788, -2.7864165543774675, -0.20985105745649374],
        [1715.134178122592, 2085.4964662263064, 1875.9766836439958, 759.8968208234021, -39.68741458861115, -0.2501447114808116],
        [1747.5745584612605, 1875.9766836439958, 1822.1414523216163, 1123.054947333127, 109.2026621607369, -0.17598452900630168],
        [935.098928788, 759.8968208234021, 1123.054947333127, 938.363436676176, 106.59121181068619, -4.890252117554146],
        [-2.7864165543774675, -39.68741458861115, 109.2026621607369, 106.59121181068619, 23.64370794528972, -21.728482419069984],
        [-0.20985105745649374, -0.2501447114808116, -0.17598452900630168, -4.890252117554146, -21.728482419069984, 57.945174065326796],
    ];
    const FIXTURE_ELL: [f64; 6] = [
        0.41517285129090653,
        -1.8692500719946608,
        2.765160237666297,
        -3.8165670131467633,
        6.59422728766729,
        4.190338688011645,
    ];

    fn fixture() -> (Array2<f64>, Array1<f64>) {
        let mut matrix = Array2::<f64>::zeros((6, 6));
        for (i, row) in FIXTURE_M.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                matrix[[i, j]] = *value;
            }
        }
        (matrix, Array1::from_vec(FIXTURE_ELL.to_vec()))
    }

    #[test]
    fn inertia_counts_pivot_signs_rather_than_solving_an_eigenproblem() {
        // Diagonal: the inertia is read straight off.
        let diagonal = array![[3.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 5.0]];
        assert_eq!(
            symmetric_inertia(diagonal.view(), 1e-12).expect("diagonal inertia"),
            Inertia { positive: 2, zero: 0, negative: 1 }
        );
        // A congruence transform must leave the inertia alone — that is
        // Sylvester's law, and it is the whole reason pivot signs are a
        // certificate. `C A Cᵀ` with `C` invertible.
        let c = array![[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]];
        let congruent = c.dot(&diagonal).dot(&c.t());
        assert_eq!(
            symmetric_inertia(congruent.view(), 1e-12).expect("congruent inertia"),
            Inertia { positive: 2, zero: 0, negative: 1 },
            "congruence preserves inertia"
        );
    }

    #[test]
    fn the_fixture_reduced_precision_has_exactly_one_negative_direction() {
        let (matrix, _) = fixture();
        assert_eq!(
            symmetric_inertia(matrix.view(), 1e-12).expect("fixture inertia"),
            Inertia { positive: 5, zero: 0, negative: 1 },
            "In(M) = (5,0,1) is what makes this a cone problem rather than a truncated Gaussian"
        );
    }

    #[test]
    fn copositivity_is_decided_exactly_and_matches_the_published_minimum() {
        let (matrix, _) = fixture();
        let (minimum, point) =
            copositive_simplex_minimum(matrix.view()).expect("simplex minimum");
        // Published on #2529 step 1 by the constrained-posterior lane
        // (face enumeration cross-checked by 4000-start projected gradient to
        // 3.02e-13 relative); reproduced here by an independent enumeration.
        assert!(
            (minimum - 6.683215003061817).abs() < 1e-9,
            "min wᵀMw over the simplex was {minimum:.15e}, expected 6.683215003061817"
        );
        assert!(minimum > 0.0, "strict copositivity ⇒ the cone-truncated law is proper");
        let total: f64 = point.sum();
        assert!((total - 1.0).abs() < 1e-9, "the argmin lies on the simplex, sum was {total}");
        assert!(point.iter().all(|value| *value >= 0.0), "the argmin is nonnegative");
    }

    #[test]
    fn a_matrix_with_a_negative_entry_pattern_is_refused_as_improper() {
        // Copositive matrices need `x'Mx > 0` on the ORTHANT only, so a negative
        // off-diagonal is not disqualifying by itself — but one large enough to
        // beat the diagonal is. This pins the direction of the test: it must
        // fail for the right reason.
        let improper = array![[1.0, -3.0], [-3.0, 1.0]];
        let (minimum, _) = copositive_simplex_minimum(improper.view()).expect("2x2 minimum");
        assert!(minimum < 0.0, "min was {minimum}, expected a negative certificate");
        let mode = constrained_cone_mode(improper.view(), &array![1.0, 1.0]);
        let message = mode.expect_err("an improper cone has no mode").to_string();
        assert!(
            message.contains("improper"),
            "the refusal must name impropriety, got: {message}"
        );
        // ... and the copositive sibling with the SAME sparsity is accepted, so
        // the test above is not passing on the sign of an off-diagonal alone.
        let proper = array![[1.0, 3.0], [3.0, 1.0]];
        let (proper_minimum, _) =
            copositive_simplex_minimum(proper.view()).expect("copositive 2x2");
        assert!(proper_minimum > 0.0, "b > 0 with positive diagonal is copositive");
    }

    #[test]
    fn the_fixture_mode_reproduces_both_lanes_to_nine_digits() {
        let (matrix, ell) = fixture();
        let linear = matrix.dot(&ell);
        let mode = constrained_cone_mode(matrix.view(), &linear).expect("cone mode");

        assert_eq!(mode.free, vec![2, 3], "the free set is coordinates 2 and 3");
        assert!(
            (mode.value - (-471.2169566469144)).abs() < 1e-7,
            "phi* was {:.12} , expected -471.2169566469144",
            mode.value
        );
        // `½ ell'M ell + phi*` is the exponent at the constrained mode, i.e. the
        // integrand's peak over the feasible set. constrained-posterior measured
        // 513.820370 by projected gradient; this is the same number by exact
        // enumeration.
        let peak = 0.5 * ell.dot(&matrix.dot(&ell)) + mode.value;
        assert!(
            (peak - 513.8203701213133).abs() < 1e-6,
            "min ½wᵀMw over the feasible set was {peak:.10}, expected 513.8203701213133"
        );

        // KKT, asserted rather than assumed: the gradient vanishes on the free
        // set and is non-negative on the active set. These ARE the multipliers
        // the quadrature integrates against, so a loose mode would become the
        // quadrature's error floor.
        for &j in &mode.free {
            assert!(
                mode.gradient[j].abs() < 1e-6,
                "gradient[{j}] = {:.3e} should vanish on the free set",
                mode.gradient[j]
            );
        }
        for j in 0..6 {
            if !mode.free.contains(&j) {
                assert!(
                    mode.gradient[j] > 0.0,
                    "multiplier {j} = {:.6} must be positive on an active wall",
                    mode.gradient[j]
                );
                assert!(
                    mode.point[j].abs() < 1e-12,
                    "an active coordinate sits on its wall, got {}",
                    mode.point[j]
                );
            }
        }
        assert!(
            (mode.point[2] - 0.6719500839935457).abs() < 1e-9
                && (mode.point[3] - 0.07574657998897513).abs() < 1e-9,
            "x* was {:?}",
            mode.point
        );
    }

    #[test]
    fn the_mode_beats_every_face_it_did_not_choose() {
        // The enumeration's claim is a GLOBAL minimum, so the reported value
        // must be no worse than the value at any feasible point — including the
        // vertices and the interior points of the faces it rejected. Without
        // this the test above would pass on a local minimum.
        let (matrix, ell) = fixture();
        let linear = matrix.dot(&ell);
        let mode = constrained_cone_mode(matrix.view(), &linear).expect("cone mode");
        let objective = |x: &Array1<f64>| 0.5 * x.dot(&matrix.dot(x)) + linear.dot(x);
        assert!((objective(&mode.point) - mode.value).abs() < 1e-9);

        // deterministic sweep over the orthant: scaled unit directions and the
        // mode perturbed toward each wall
        for j in 0..6 {
            for scale in [0.0, 0.05, 0.25, 1.0, 4.0, 16.0] {
                let mut probe = Array1::<f64>::zeros(6);
                probe[j] = scale;
                assert!(
                    objective(&probe) >= mode.value - 1e-7,
                    "axis probe ({j}, {scale}) scored {:.6} below phi* {:.6}",
                    objective(&probe),
                    mode.value
                );
                let mut nudged = mode.point.clone();
                nudged[j] = (nudged[j] + scale).max(0.0);
                assert!(
                    objective(&nudged) >= mode.value - 1e-7,
                    "nudge ({j}, {scale}) scored below phi*"
                );
            }
        }
    }

    #[test]
    fn a_mode_on_an_interior_optimum_keeps_every_coordinate_free() {
        // Positive definite with a linear term pulling well inside: the answer
        // is the unconstrained solve and no wall is active. This is the branch
        // the fixture never exercises, and it is where an off-by-one in the
        // support mask would show up.
        let matrix = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]];
        let linear = array![-10.0, -12.0, -20.0];
        let mode = constrained_cone_mode(matrix.view(), &linear).expect("interior mode");
        assert_eq!(mode.free, vec![0, 1, 2], "an interior optimum activates no wall");
        assert!(
            mode.gradient.iter().all(|value| value.abs() < 1e-9),
            "an interior optimum has a vanishing gradient, got {:?}",
            mode.gradient
        );
        let residual = matrix.dot(&mode.point) + &linear;
        assert!(residual.iter().all(|value| value.abs() < 1e-9));
    }

    #[test]
    fn the_mode_of_a_pinned_problem_sits_on_the_corner() {
        // Linear term pushing outward on every coordinate: the mode is the
        // origin, every wall active, every multiplier positive.
        let matrix = array![[2.0, 0.5], [0.5, 2.0]];
        let linear = array![3.0, 7.0];
        let mode = constrained_cone_mode(matrix.view(), &linear).expect("corner mode");
        assert!(mode.free.is_empty(), "no coordinate is free at a full corner");
        // These are exact by construction (the mode is the origin, so the
        // objective and its gradient are evaluated on a zero vector), but they
        // are asserted with a tolerance rather than by equality so the test does
        // not depend on that remaining true of the arithmetic.
        assert!(mode.point.iter().all(|value| value.abs() < 1e-12));
        assert!(mode.value.abs() < 1e-12, "phi* at a full corner is zero, got {}", mode.value);
        for j in 0..2 {
            assert!(
                (mode.gradient[j] - linear[j]).abs() < 1e-12,
                "the multipliers are the linear term itself at a full corner"
            );
        }
    }
}
