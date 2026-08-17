//! The fitted CTN conditional transform, tabulated — **with the affine tails it
//! actually has** outside the tabulated range.
//!
//! A conditional-transformation-normal model is `F(y | x) = Φ(h(y | x))` with
//! `h(·|x)` strictly increasing, so every response-scale question a fitted CTN
//! can be asked — the conditional mean `E[Y|x] = E_Z[h⁻¹(Z|x)]`, the predictive
//! quantile ladder `h⁻¹(Φ⁻¹(p)|x)`, an inverse-transform draw `h⁻¹(Z|x)` — is
//! one function, `h⁻¹`, evaluated at different latent arguments. This type is
//! that function, and it is the ONLY place it is implemented.
//!
//! # Why the tails are part of the object
//!
//! `h` is tabulated on the fitted response support `[y_lo, y_hi]`, which is a
//! bounded interval; the latent `Z` is not bounded. The tabulated rows therefore
//! never cover the whole latent axis: `h(y_lo|x)` and `h(y_hi|x)` are finite
//! numbers `L(x)`, `U(x)`, typically around `∓Φ⁻¹(1/(n+1))`, and every latent
//! target outside `[L, U]` — which is `Φ(L) + 1 − Φ(U)` of the predictive mass,
//! *by the model's own reckoning* — lands off the end of the table.
//!
//! Both inverters this type replaces answered such a target with the support
//! endpoint. That is a truncation the fitted likelihood does not perform: it
//! makes `y_lo` and `y_hi` atoms of the predictive law, pins every observation
//! band at the training range no matter how extreme the level, and biases
//! `E[Y|x]` inward. gam#2600.
//!
//! Since the CTN transformation is **affine beyond the boundary knots**
//! (`ctn_response_bases_at` continues the I-spline value basis linearly at its
//! own boundary derivative), the inverse outside the table is not a modelling
//! choice made here — it is arithmetic:
//!
//! ```text
//! z < L(x):  h⁻¹(z|x) = y_lo + (z − L(x)) / h'(y_lo|x)
//! z > U(x):  h⁻¹(z|x) = y_hi + (z − U(x)) / h'(y_hi|x)
//! ```
//!
//! with `h'(y_lo|x)`, `h'(y_hi|x) ≥ ε > 0` structurally, from the same chart
//! evaluation that produced the tabulated row. Carrying those two slopes
//! alongside the table is what makes the tails impossible for a consumer to
//! forget: there is no way to hold this object and not hold them.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A per-row tabulation of the fitted CTN transform `h(·|x_i)` on a shared
/// response grid, plus the slopes of its two affine tails.
///
/// Invariants, all checked by [`CtnTransformTable::new`]:
/// * `grid_y` has `g ≥ 2` finite, strictly increasing entries;
/// * `h` is `n × g` and strictly increasing along every row;
/// * both tail slopes are finite and strictly positive on every row.
///
/// Together these make `invert` a total function of the latent argument: the
/// tabulated part is a monotone bracket-and-interpolate, and the two tails are
/// exact affine inverses.
#[derive(Clone, Debug)]
pub struct CtnTransformTable {
    grid_y: Array1<f64>,
    h: Array2<f64>,
    tail_slope_lower: Array1<f64>,
    tail_slope_upper: Array1<f64>,
}

impl CtnTransformTable {
    /// Assemble a table, validating every invariant `invert` relies on.
    ///
    /// The validation is not defensive decoration: a non-monotone row makes the
    /// bracketing search meaningless, and a zero or negative tail slope makes
    /// the affine inverse point the wrong way (or to infinity). Both are
    /// structurally impossible for a feasible CTN fit — `h' = ε + Σ_k M_k α_k`
    /// with `α ≥ 0` on the monotonicity cone — so either one signals a corrupt
    /// coefficient block, and the caller should hear about it here rather than
    /// receive a silently wrong quantile.
    pub fn new(
        grid_y: Array1<f64>,
        h: Array2<f64>,
        tail_slope_lower: Array1<f64>,
        tail_slope_upper: Array1<f64>,
    ) -> Result<Self, String> {
        let g = grid_y.len();
        if g < 2 {
            return Err(format!(
                "CTN transform table needs at least two response grid nodes, got {g}"
            ));
        }
        for k in 0..g {
            if !grid_y[k].is_finite() {
                return Err(format!(
                    "CTN transform table response grid node {k} is not finite: {}",
                    grid_y[k]
                ));
            }
            if k > 0 && !(grid_y[k] > grid_y[k - 1]) {
                return Err(format!(
                    "CTN transform table response grid is not strictly increasing at node {k}: \
                     {:.17e} -> {:.17e}",
                    grid_y[k - 1],
                    grid_y[k]
                ));
            }
        }
        let n = h.nrows();
        if h.ncols() != g {
            return Err(format!(
                "CTN transform table has {} latent columns but {g} response grid nodes",
                h.ncols()
            ));
        }
        if tail_slope_lower.len() != n || tail_slope_upper.len() != n {
            return Err(format!(
                "CTN transform table tail slopes have lengths ({}, {}) but the table has {n} rows",
                tail_slope_lower.len(),
                tail_slope_upper.len()
            ));
        }
        for i in 0..n {
            for k in 0..g {
                if !h[[i, k]].is_finite() {
                    return Err(format!(
                        "CTN transform table entry (row {i}, node {k}) is not finite: {}",
                        h[[i, k]]
                    ));
                }
                if k > 0 && !(h[[i, k]] > h[[i, k - 1]]) {
                    return Err(format!(
                        "CTN transform table row {i} is not strictly increasing between nodes \
                         {} and {k}: {:.17e} -> {:.17e}",
                        k - 1,
                        h[[i, k - 1]],
                        h[[i, k]]
                    ));
                }
            }
            for (name, slope) in [
                ("lower", tail_slope_lower[i]),
                ("upper", tail_slope_upper[i]),
            ] {
                if !(slope.is_finite() && slope > 0.0) {
                    return Err(format!(
                        "CTN transform table {name} tail slope at row {i} is {slope:.6e}; the \
                         transform is affine beyond the fitted support with slope h' = ε + \
                         Σ_k M_k·α_k, which is structurally positive on the monotonicity cone"
                    ));
                }
            }
        }
        Ok(Self {
            grid_y,
            h,
            tail_slope_lower,
            tail_slope_upper,
        })
    }

    /// Number of covariate rows the table carries.
    pub fn nrows(&self) -> usize {
        self.h.nrows()
    }

    /// The shared, strictly increasing response grid the transform is tabulated
    /// on. Its first and last entries are the fitted support `[y_lo, y_hi]`, the
    /// two points the affine tails are anchored at.
    pub fn grid_y(&self) -> ArrayView1<'_, f64> {
        self.grid_y.view()
    }

    /// `h[[i, k]] = h(grid_y[k] | x_i)` — the model's own latent, on the scale
    /// the standard normal is compared against.
    pub fn latent(&self) -> ArrayView2<'_, f64> {
        self.h.view()
    }

    /// `(h'(y_lo | x_i), h'(y_hi | x_i))` — the two tail slopes of row `i`.
    pub fn tail_slopes(&self, row: usize) -> (f64, f64) {
        (self.tail_slope_lower[row], self.tail_slope_upper[row])
    }

    /// `h⁻¹(target | x_row)` — the response value whose latent is `target`.
    ///
    /// Inside the tabulated range this brackets and linearly interpolates the
    /// row; outside it the transform is affine, so the inverse is the exact
    /// affine inverse rather than the support endpoint. The two branches agree
    /// at the endpoints by construction (`target == h[0]` returns `grid_y[0]`
    /// from either side), so the returned quantile function is continuous and
    /// strictly increasing in `target` on the whole real line.
    pub fn invert(&self, row: usize, target: f64) -> f64 {
        let g = self.grid_y.len();
        let h = self.h.row(row);
        if target <= h[0] {
            return self.grid_y[0] + (target - h[0]) / self.tail_slope_lower[row];
        }
        if target >= h[g - 1] {
            return self.grid_y[g - 1] + (target - h[g - 1]) / self.tail_slope_upper[row];
        }
        let mut lo = 0usize;
        let mut hi = g - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if h[mid] <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (target - h[lo]) / (h[hi] - h[lo]);
        self.grid_y[lo] + t * (self.grid_y[hi] - self.grid_y[lo])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn table() -> CtnTransformTable {
        // h(y) = 2y on [-1, 1], so L = -2, U = +2, and both tails have slope 2.
        let grid_y = Array1::from_vec(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        let h = Array2::from_shape_fn((2, 5), |(i, k)| {
            let slope = if i == 0 { 2.0 } else { 4.0 };
            slope * grid_y[k]
        });
        CtnTransformTable::new(grid_y, h, array![2.0, 4.0], array![2.0, 4.0]).expect("valid table")
    }

    #[test]
    fn inverse_is_exact_inside_the_table() {
        let t = table();
        for &z in &[-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
            assert!((t.invert(0, z) - 0.5 * z).abs() < 1e-12, "z={z}");
            assert!((t.invert(1, z) - 0.25 * z).abs() < 1e-12, "z={z}");
        }
    }

    #[test]
    fn inverse_extends_through_the_affine_tails_rather_than_clamping() {
        // The whole of gam#2600's predictive residual in one assertion: a latent
        // target past the tabulated range is not the support endpoint.
        let t = table();
        for &z in &[-8.0_f64, -4.0, 4.0, 8.0] {
            let y = t.invert(0, z);
            assert!(
                (y - 0.5 * z).abs() < 1e-12,
                "tail inverse at z={z} is {y}, expected {}",
                0.5 * z
            );
            assert!(
                y.abs() > 1.0,
                "tail inverse at z={z} clamped back inside the tabulated support"
            );
        }
    }

    #[test]
    fn inverse_is_continuous_at_both_table_ends() {
        let t = table();
        for &(z, expected) in &[(-2.0_f64, -1.0_f64), (2.0, 1.0)] {
            let inside = t.invert(0, z * (1.0 - 1e-12));
            let outside = t.invert(0, z * (1.0 + 1e-12));
            assert!((inside - expected).abs() < 1e-9, "inside {inside}");
            assert!((outside - expected).abs() < 1e-9, "outside {outside}");
        }
    }

    #[test]
    fn a_non_monotone_row_is_refused() {
        let grid_y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let h = Array2::from_shape_vec((1, 3), vec![0.0, 0.5, 0.5]).expect("shape");
        let error = CtnTransformTable::new(grid_y, h, array![1.0], array![1.0])
            .expect_err("a flat row must be refused");
        assert!(error.contains("strictly increasing"), "{error}");
    }

    #[test]
    fn a_non_positive_tail_slope_is_refused() {
        let grid_y = Array1::from_vec(vec![0.0, 1.0]);
        let h = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).expect("shape");
        let error = CtnTransformTable::new(grid_y, h, array![0.0], array![1.0])
            .expect_err("a zero tail slope must be refused");
        assert!(error.contains("tail slope"), "{error}");
    }
}
