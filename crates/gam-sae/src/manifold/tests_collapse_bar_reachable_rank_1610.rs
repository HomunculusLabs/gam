//! Causal regression oracles for #1610/#2498.
//!
//! Training explained variance is not a decoder-disappearance proof. These
//! tests retain the useful projection mathematics as explicit test-only
//! oracles, while preventing it from drifting back into a hard production
//! verdict.

use super::tests::periodic_basis;
use super::*;
use ndarray::{Array2, ArrayView2, array, s};

fn concatenated_design_rank(
    designs: &[ArrayView2<'_, f64>],
    n_rows: usize,
) -> Result<usize, String> {
    let mut total_cols = 0usize;
    for (atom, design) in designs.iter().enumerate() {
        if design.nrows() != n_rows {
            return Err(format!(
                "atom {atom} design has {} rows; expected {n_rows}",
                design.nrows()
            ));
        }
        if design.iter().any(|value| !value.is_finite()) {
            return Err(format!("atom {atom} design is non-finite"));
        }
        total_cols = total_cols
            .checked_add(design.ncols())
            .ok_or_else(|| "concatenated design width overflowed usize".to_string())?;
    }
    if n_rows == 0 || total_cols == 0 {
        return Ok(0);
    }
    let mut concatenated = Array2::<f64>::zeros((n_rows, total_cols));
    let mut first_col = 0usize;
    for design in designs {
        let past_last_col = first_col + design.ncols();
        concatenated
            .slice_mut(s![.., first_col..past_last_col])
            .assign(design);
        first_col = past_last_col;
    }
    let (_, singular_values, _) = concatenated
        .svd(false, false)
        .map_err(|error| format!("concatenated design SVD failed: {error}"))?;
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    if largest == 0.0 {
        return Ok(0);
    }
    let cutoff = SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF * largest;
    Ok(singular_values
        .iter()
        .filter(|&&value| value > cutoff)
        .count())
}

fn centered_free_projection_null_ev(
    n_rows: usize,
    total_rank_including_intercept: usize,
) -> Result<f64, String> {
    if n_rows < 2 {
        return Err("centered null EV requires at least two rows".to_string());
    }
    if total_rank_including_intercept == 0 || total_rank_including_intercept > n_rows {
        return Err(format!(
            "centered null EV rank must lie in 1..={n_rows}; got \
             {total_rank_including_intercept}"
        ));
    }
    Ok((total_rank_including_intercept - 1) as f64 / (n_rows - 1) as f64)
}

#[test]
fn concatenated_chart_rank_uses_one_consistent_row_space_1610() {
    let n = 8usize;
    let coords_a = array![
        [0.0_f64],
        [0.125],
        [0.25],
        [0.375],
        [0.5],
        [0.625],
        [0.75],
        [0.875],
    ];
    let coords_b = Array2::<f64>::from_elem((n, 1), 0.3);
    let (phi_a, _) = periodic_basis(&coords_a);
    let (phi_b, _) = periodic_basis(&coords_b);
    assert_eq!(concatenated_design_rank(&[phi_a.view()], n).unwrap(), 3);
    assert_eq!(concatenated_design_rank(&[phi_b.view()], n).unwrap(), 1);

    let union_rank =
        concatenated_design_rank(&[phi_a.view(), phi_b.view()], n).expect("valid union rank");
    assert_eq!(
        union_rank, 3,
        "the repeated chart contributes only the constant direction already in the full chart"
    );
    assert_eq!(
        concatenated_design_rank(&[phi_a.view(), phi_a.view()], n).unwrap(),
        3,
        "identical atom designs must not double-count a shared row subspace"
    );

    let short = phi_a.slice(s![..n - 1, ..]);
    assert!(
        concatenated_design_rank(&[short, phi_b.view()], n).is_err(),
        "a row-space mismatch must be rejected, never replaced by a summed-rank fallback"
    );
}

#[test]
fn centered_projection_null_is_four_ninths_not_five_tenths() {
    let null_ev = centered_free_projection_null_ev(10, 5).unwrap();
    assert!((null_ev - 4.0 / 9.0).abs() < 1.0e-15);
    for observed in [0.4495_f64, 0.4887] {
        assert!(
            observed > null_ev,
            "reported EV {observed} exceeds the exact centered free-projection null {null_ev}"
        );
    }
}

#[test]
fn one_eigen_smoother_has_three_distinct_null_quantities() {
    let smoother_eigenvalue = 0.6_f64;
    let expected_ev = 2.0 * smoother_eigenvalue - smoother_eigenvalue.powi(2);
    let trace_influence = smoother_eigenvalue;
    let expected_output_energy = smoother_eigenvalue.powi(2);

    assert!(expected_ev > trace_influence);
    assert!(trace_influence > expected_output_energy);
    assert!(
        (expected_ev - trace_influence).abs() > f64::EPSILON
            && (trace_influence - expected_output_energy).abs() > f64::EPSILON,
        "one arbitrary q/n-style scalar cannot calibrate EV and output energy for a smoother"
    );
}
