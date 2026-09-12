//! GLM fixed-design sufficient-statistic reuse for #1033 mechanism (a),
//! extending the n-free lever from the Gaussian lane to non-Gaussian families.
//!
//! Scope: single-scale-mode measure jets where `dX/dpsi == 0`, i.e. the design matrix
//! `X` is theta-invariant across the lambda/rho outer loop AND across the inner
//! IRLS/PIRLS iterations. For non-Gaussian families the scalar working-weight
//! diagonal `W` (and the working response `z`) genuinely change every IRLS step,
//! so the Gaussian constant-Gram cache in `FixedDesignGramCache` does not apply.
//! What is reusable is the design itself.
//!
//! This module owns the fixed `X` rows once and exposes the two weighted
//! contractions the GLM normal equations need each iteration:
//!
//!   * `weighted_gram(w)`     = X' diag(w) X        (the IRLS Hessian block)
//!   * `weighted_xty(w, z)`   = X' diag(w) z        (the IRLS working RHS)
//!
//! These match the PIRLS semantics formed in `solver::reml::assembly`
//! (`xt_diag_x_dense_into`) and `linalg::faer_ndarray` (`fast_xt_diag_x`,
//! `fast_xt_diag_y`), routing through the very same weighted-contraction
//! primitives so values agree bit-for-bit with the runtime recompute path.
//!
//! What is SAVED across trials: the O(n p) construction of the n-row measure-jet
//! design. The n-row basis kernel is evaluated exactly once, at construction,
//! and never again as the outer lambda loop sweeps or as IRLS reweights. The
//! stored rows are immutable, so the cache never touches the n-row design on a
//! query.
//!
//! What is NOT saved (and cannot be, when `W` moves): the O(n p^2) weighted
//! contraction `X' diag(w) X` and the O(n p) `X' diag(w) z`. Those are the
//! irreducible IRLS work and run every iteration over the cached rows.
//!
//! This is distinct from `measure_jet_gram_cache::FixedDesignRowCache`, which
//! exposes recompute accessors; this module owns the design privately and every
//! query borrows it immutably, so no query rebuilds or mutates the n-row design.

use gam_linalg::faer_ndarray::{fast_xt_diag_x, fast_xt_diag_y};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// GLM fixed-design sufficient-statistic provider.
///
/// Holds the theta-invariant design `X` (n x p) once. Every IRLS iteration and
/// every outer lambda trial reuses these stored rows for the weighted
/// contractions, so the n-row measure-jet basis is built exactly once per fit.
pub struct GlmFixedDesignSufficient {
    x: Array2<f64>,
    n: usize,
    p: usize,
}

impl GlmFixedDesignSufficient {
    /// Cache a finite, non-empty fixed design.
    ///
    /// The n-row work happens here, once. Subsequent `weighted_gram` /
    /// `weighted_xty` calls never rebuild the design.
    pub fn build(x: ArrayView2<'_, f64>) -> Result<Self, String> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(format!(
                "design must be non-empty, got shape {}x{}",
                x.nrows(),
                x.ncols()
            ));
        }
        validate_finite_matrix("x", x)?;
        let n = x.nrows();
        let p = x.ncols();
        let x_owned = x.to_owned();
        Ok(Self {
            x: x_owned,
            n,
            p,
        })
    }

    /// Row count `n` of the fixed design.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Column count `p` of the fixed design.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Borrow the stored, immutable design rows.
    pub fn design(&self) -> ArrayView2<'_, f64> {
        self.x.view()
    }

    /// Weighted Gram `X' diag(w) X` over the stored rows.
    ///
    /// This is the GLM IRLS Hessian block. It is recomputed each iteration
    /// because `w` moves, but it reuses the cached design: the O(n p^2)
    /// contraction runs, while the n-row design construction does not repeat.
    /// Routed through `fast_xt_diag_x` to match the runtime PIRLS recompute path
    /// exactly.
    pub fn weighted_gram(&self, w: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.validate_weights(w)?;
        Ok(fast_xt_diag_x(&self.x, &w))
    }

    /// Weighted cross product `X' diag(w) z` over the stored rows.
    ///
    /// This is the GLM IRLS working right-hand side for working response `z`.
    /// Recomputed each iteration (`w` and `z` move) but over the cached design.
    /// Routed through `fast_xt_diag_y` to match the runtime PIRLS recompute path.
    pub fn weighted_xty(
        &self,
        w: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        self.validate_weights(w)?;
        if z.len() != self.n {
            return Err(format!(
                "z length {} must match design row count {}",
                z.len(),
                self.n
            ));
        }
        validate_finite_vector("z", z)?;
        let z2 = z.insert_axis(ndarray::Axis(1));
        let xtwz_mat = fast_xt_diag_y(&self.x, &w, &z2);
        Ok(xtwz_mat.column(0).to_owned())
    }

    fn validate_weights(&self, weights: ArrayView1<'_, f64>) -> Result<(), String> {
        if weights.len() != self.n {
            return Err(format!(
                "weights length {} must match design row count {}",
                weights.len(),
                self.n
            ));
        }
        validate_finite_vector("weights", weights)
    }
}

fn validate_finite_matrix(name: &str, matrix: ArrayView2<'_, f64>) -> Result<(), String> {
    for ((row, col), value) in matrix.indexed_iter() {
        if !(*value).is_finite() {
            return Err(format!("{name}[{row},{col}] must be finite"));
        }
    }
    Ok(())
}

fn validate_finite_vector(name: &str, vector: ArrayView1<'_, f64>) -> Result<(), String> {
    for (index, value) in vector.iter().enumerate() {
        if !(*value).is_finite() {
            return Err(format!("{name}[{index}] must be finite"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::GlmFixedDesignSufficient;
    use ndarray::Array2;

    fn deterministic_design(n: usize, p: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, p), |(i, j)| {
            let row = i as f64 + 1.0;
            let col = j as f64 + 1.0;
            ((row * 0.17 + col * 0.31).sin()) + row * col * 0.002
        })
    }

    #[test]
    fn build_rejects_empty_and_nonfinite() {
        let empty = Array2::<f64>::zeros((0, 3));
        assert!(GlmFixedDesignSufficient::build(empty.view()).is_err());

        let mut x = deterministic_design(10, 3);
        x[[4, 1]] = f64::NAN;
        assert!(GlmFixedDesignSufficient::build(x.view()).is_err());
    }

}
