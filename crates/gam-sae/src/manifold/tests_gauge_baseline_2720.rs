//! #2720 baseline re-measurement: are the chart-gauge orbit directional
//! derivatives of the penalized objective still above tolerance at current main?
//!
//! ## Background
//!
//! The issue measured `|gᵀv₁| = 4.592e-3` (8.24× tol) and `|gᵀv₂| = 5.823e-3`
//! (10.45× tol) at a stall state on the 2336 fixture. Those numbers were
//! produced by a solver that no longer exists — `a386c1e8b` deleted
//! `exact_hessian_physical_complement`, which made the Newton step minimum-norm
//! on a complement holding 87–93% of the gradient.
//!
//! 18 of #2674's 19 fixtures flipped green on that change. So the stall state
//! that produced `8.24×` and `10.45×` may no longer be reachable.
//!
//! This test takes the measurement again at current `main`. It does NOT assert
//! any particular value — it prints the measurement so we can compare against
//! the pre-`a386c1e8b` baseline.
//!
//! ## Method
//!
//! 1. Create the `ard_saddle_state` from the #2336 fixture (same geometry the
//!    original measurement was taken on).
//! 2. Run `penalized_quasi_laplace_criterion_with_cache` to reach the inner
//!    solve state.
//! 3. Assemble the Arrow-Schur system at the resulting inner state.
//! 4. Extract the joint KKT gradient `[g_t (per row); g_β]`.
//! 5. Get the chart-gauge basis from `joint_chart_gauge_basis_for_arrow_layout`.
//! 6. Project: `|gᵀvᵢ|` for each gauge direction.
//! 7. Report against `SAE_MANIFOLD_INNER_GRAD_REL_TOL * iterate_scale`.

#![cfg(test)]
use super::*;
use crate::manifold::tests::gamma_fd_tiny_fixture;

/// Reproduce the ARD saddle state from #2336: the same planted circle fixture
/// with phase-shifted sinusoidal perturbation that puts some latent coordinates
/// in the ARD periodic prior's concave half.
fn ard_saddle_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (term, mut target, mut rho) = gamma_fd_tiny_fixture();
    let (n, p) = (target.nrows(), target.ncols());
    for row in 0..n {
        for col in 0..p {
            let phase = (row as f64 + 0.35) / n as f64;
            let theta = std::f64::consts::TAU * phase;
            target[[row, col]] += 0.6 * (3.0 * theta + 0.5 * col as f64).sin();
        }
    }
    rho.log_lambda_sparse = -0.5;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -0.5;
        }
    }
    (term, target, rho)
}

#[test]
fn measure_chart_gauge_orbit_gradient_projection_2720() {
    let (mut term, target, rho) = ard_saddle_state();

    // Run the inner solve to reach the quasi-Laplace criterion state.
    let result = term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );

    let (criterion_value, loss, _cache) = match result {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[2720-baseline] inner solve returned Err: {e:?}");
            eprintln!("[2720-baseline] the inner solve REFUSED — the stall state");
            eprintln!("[2720-baseline] this itself is informative: the solver either");
            eprintln!("[2720-baseline] cannot reach a stationary point, or refuses on");
            eprintln!("[2720-baseline] a gate that did not exist when 8.24×/10.45× were measured");
            return;
        }
    };

    eprintln!("[2720-baseline] criterion value = {criterion_value:.6e}");
    eprintln!(
        "[2720-baseline] loss components: data_fit={:.6e}, sparsity={:.6e}, smoothness={:.6e}, ard={:.6e}",
        loss.data_fit, loss.assignment_sparsity, loss.smoothness, loss.ard
    );

    // Assemble the Arrow-Schur system at the inner solve state to extract
    // the KKT gradient.
    let sys = match term.assemble_arrow_schur(target.view(), &rho, None) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[2720-baseline] assemble_arrow_schur failed: {e}");
            return;
        }
    };

    // Extract the joint gradient: [g_t (per row, flattened); g_β]
    let n_rows = sys.rows.len();
    let coord_dim = sys.rows.first().map(|r| r.gt.len()).unwrap_or(0);
    let border_dim = sys.gb.len();
    let full_len: usize = sys.rows.iter().map(|r| r.gt.len()).sum::<usize>() + border_dim;

    let mut grad = Array1::<f64>::zeros(full_len);
    let mut offset = 0usize;
    for row in &sys.rows {
        for (i, &v) in row.gt.iter().enumerate() {
            grad[offset + i] = v;
        }
        offset += row.gt.len();
    }
    for (i, &v) in sys.gb.iter().enumerate() {
        grad[offset + i] = v;
    }

    let grad_norm = grad.dot(&grad).sqrt();
    eprintln!(
        "[2720-baseline] n_rows={n_rows}, coord_dim={coord_dim}, border_dim={border_dim}"
    );
    eprintln!("[2720-baseline] full gradient length = {full_len}");
    eprintln!("[2720-baseline] ‖g‖ = {grad_norm:.6e}");

    // Get the chart-gauge basis vectors.
    let row_offsets: Vec<usize> = {
        let mut offsets = Vec::with_capacity(n_rows + 1);
        offsets.push(0);
        let mut acc = 0;
        for row in &sys.rows {
            acc += row.gt.len();
            offsets.push(acc);
        }
        offsets
    };

    let gauge_basis = match term.joint_chart_gauge_basis_for_arrow_layout(
        &row_offsets,
        border_dim,
        "2720-baseline chart gauge projection",
    ) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[2720-baseline] joint_chart_gauge_basis failed: {e}");
            return;
        }
    };

    eprintln!(
        "[2720-baseline] chart-gauge basis: {} directions",
        gauge_basis.len()
    );

    if gauge_basis.is_empty() {
        eprintln!("[2720-baseline] no gauge directions found — measurement not applicable");
        eprintln!("[2720-baseline] at this state/fixture combination");
        return;
    }

    // Tolerance: same as the acceptance gate.
    let iterate_scale = term.inner_iterate_scale();
    let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * iterate_scale;
    eprintln!(
        "[2720-baseline] iterate_scale = {iterate_scale:.6e}, tolerance = {tolerance:.6e}"
    );

    // Project gradient onto each gauge direction.
    let mut max_ratio = 0.0f64;
    for (i, v) in gauge_basis.iter().enumerate() {
        let proj = grad.dot(v);
        let ratio = if tolerance > 0.0 {
            proj.abs() / tolerance
        } else {
            f64::INFINITY
        };
        max_ratio = max_ratio.max(ratio);
        eprintln!(
            "[2720-baseline] v_{i}: |gᵀv| = {:.6e}  ({:.2}× tolerance)",
            proj.abs(),
            ratio
        );
    }

    eprintln!("[2720-baseline] ────────────────────────────────────────");
    eprintln!("[2720-baseline] max |gᵀv|/tol = {max_ratio:.2}×");

    // Compare against the pre-a386c1e8b baseline.
    if max_ratio <= 1.0 {
        eprintln!(
            "[2720-baseline] AT OR BELOW TOLERANCE — the orbit derivatives are now"
        );
        eprintln!(
            "[2720-baseline] within the acceptance gate. The witness state for"
        );
        eprintln!(
            "[2720-baseline] 8.24×/10.45× may be gone; the modelling fix's"
        );
        eprintln!(
            "[2720-baseline] acceptance criteria need a new fixture."
        );
    } else {
        eprintln!(
            "[2720-baseline] ABOVE TOLERANCE — the orbit derivatives are still"
        );
        eprintln!(
            "[2720-baseline] materially above the convergence tolerance."
        );
        eprintln!(
            "[2720-baseline] The modelling question is live and the original"
        );
        eprintln!(
            "[2720-baseline] acceptance criteria may still be applicable."
        );
    }
}
