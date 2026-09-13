//! The monotone link-wiggle basis shared by every GAMLSS family that carries
//! a `linkwiggle(...)` block: `q(q0) = q0 + B(q0)·β`, with `B` the monotone
//! wiggle basis on the family's knots.
//!
//! A family declares only where its knots and degree live; the basis value
//! and the first- to fourth-derivative contractions with the wiggle
//! coefficients are provided here once. Three families used to carry private
//! copies of these seven methods (#2470).

use super::*;

pub(crate) trait MonotoneWiggleFamily {
    /// Knot vector of the monotone wiggle basis.
    fn wiggle_knots(&self) -> &Array1<f64>;

    /// Spline degree of the monotone wiggle basis.
    fn wiggle_degree(&self) -> usize;

    fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            self.wiggle_knots(),
            self.wiggle_degree(),
            options.derivative_order,
        )
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    /// `dq/dq0 = 1 + B'(q0)·β`.
    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d1 = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        wiggle_contraction_columns_match("derivative", d1.ncols(), beta_link_wiggle.len())?;
        Ok(d1.dot(&beta_link_wiggle) + 1.0)
    }

    /// `d²q/dq0² = B''(q0)·β`.
    fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2 = self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        wiggle_contraction_columns_match("second-derivative", d2.ncols(), beta_link_wiggle.len())?;
        Ok(d2.dot(&beta_link_wiggle))
    }

    /// The third-derivative basis `B⁽³⁾(q0)`.
    fn wiggle_d3basis_constrained(
        &self,
        q0: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            self.wiggle_knots(),
            self.wiggle_degree(),
            3,
        )
    }

    /// `d³q/dq0³ = B⁽³⁾(q0)·β`.
    fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3 = self.wiggle_d3basis_constrained(q0)?;
        wiggle_contraction_columns_match("third-derivative", d3.ncols(), beta_link_wiggle.len())?;
        Ok(d3.dot(&beta_link_wiggle))
    }

    /// `d⁴q/dq0⁴ = B⁽⁴⁾(q0)·β`.
    fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            self.wiggle_knots(),
            self.wiggle_degree(),
            4,
        )?;
        wiggle_contraction_columns_match("fourth-derivative", d4.ncols(), beta_link_wiggle.len())?;
        Ok(d4.dot(&beta_link_wiggle))
    }
}

/// The wiggle coefficient vector must have one entry per basis column.
fn wiggle_contraction_columns_match(
    which: &str,
    basis_columns: usize,
    coefficients: usize,
) -> Result<(), String> {
    if basis_columns == coefficients {
        return Ok(());
    }
    Err(GamlssError::DimensionMismatch {
        reason: format!(
            "wiggle {which}/beta mismatch: basis has {basis_columns} columns but \
             beta_link_wiggle has {coefficients} coefficients"
        ),
    }
    .into())
}
