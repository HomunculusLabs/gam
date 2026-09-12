use ndarray::ArrayView2;

/// Gauge-companion loss for equivariant terms.
///
/// Penalizes the fitted angular field `theta` against the auxiliary targets
/// `aux_values` (each component in `[0, 1)`). The first companion wraps onto the
/// circle as `1 - cos(theta_0 - 2*pi*aux_0)`; any further companions
/// (`k = 1, 2`, gated by `d_aux` and the available columns) are squared
/// cosine-deviation terms `(cos(theta_k) - (2*aux_k - 1))^2`. Each companion's
/// per-row mean is averaged and scaled by `weight`. Returns `0.0` for an empty
/// batch. This is the single source of truth shared by the CLI/core and the
/// `equivariant_gauge_companion_loss` FFI shim.
pub fn gauge_companion_loss(
    aux_values: ArrayView2<'_, f64>,
    theta: ArrayView2<'_, f64>,
    d_aux: usize,
    weight: f64,
) -> Result<f64, String> {
    if !weight.is_finite() {
        return Err("gauge companion weight must be finite".to_string());
    }
    if aux_values.ncols() < 1 {
        return Err("aux_values must have at least one column".to_string());
    }
    if theta.nrows() != aux_values.nrows() {
        return Err("aux_values and theta must agree in row count".to_string());
    }
    let n = aux_values.nrows();
    if n == 0 {
        return Ok(0.0);
    }
    let n_f = n as f64;
    let mut terms: Vec<f64> = Vec::new();
    let two_pi = std::f64::consts::TAU;
    let mut term0 = 0.0_f64;
    for row in 0..n {
        let h_rad = aux_values[[row, 0]] * two_pi;
        term0 += 1.0 - (theta[[row, 0]] - h_rad).cos();
    }
    terms.push(term0 / n_f);
    if d_aux >= 2 && theta.ncols() >= 2 && aux_values.ncols() >= 2 {
        let mut term1 = 0.0_f64;
        for row in 0..n {
            let diff = theta[[row, 1]].cos() - (2.0 * aux_values[[row, 1]] - 1.0);
            term1 += diff * diff;
        }
        terms.push(term1 / n_f);
    }
    if d_aux >= 3 && theta.ncols() >= 3 && aux_values.ncols() >= 3 {
        let mut term2 = 0.0_f64;
        for row in 0..n {
            let diff = theta[[row, 2]].cos() - (2.0 * aux_values[[row, 2]] - 1.0);
            term2 += diff * diff;
        }
        terms.push(term2 / n_f);
    }
    let total: f64 = terms.iter().sum();
    Ok(weight * total / (terms.len() as f64))
}
