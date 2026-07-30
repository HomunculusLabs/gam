//! SCRATCH probe for #2638 — bisect the analytic `dS/dpsi` chain on the
//! `no_ident` fixture. Not for landing.

#![cfg(test)]

use ndarray::{Array2, s};

use super::*;

fn fixture() -> (Array2<f64>, DuchonBasisSpec) {
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    (data, spec)
}

fn fro(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Rebuild the intermediate quantities of
/// `build_duchon_native_penalty_psi_derivatives` at a given psi and return
/// (omega, floored.value, primary_embedded, primary_normalized, trend_value).
fn stages(
    centers: ndarray::ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    psi: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let mut workspace = BasisWorkspace::default();
    let base_ls = spec.length_scale.expect("ls");
    let kappa = (1.0 / base_ls) * psi.exp();
    let length_scale = 1.0 / kappa;
    let effective_nullspace_order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power_as_usize();
    let dim = centers.ncols();
    let z = kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)
        .expect("z");
    let kernel_cols = z.ncols();
    let poly_cols = polynomial_block_from_order(centers, effective_nullspace_order).ncols();
    let total_cols = kernel_cols + poly_cols;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale.max(1e-300));
    let kernel_amp = duchon_kernel_amplification(
        centers,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        None,
        Some(&coeffs),
        None,
    );
    let n_centers = centers.nrows();
    let mut kernel = Array2::<f64>::zeros((n_centers, n_centers));
    for i in 0..n_centers {
        for j in i..n_centers {
            let r = euclidean_distance_rows(centers, i, centers, j);
            let core = duchon_radial_core_psi_triplet(r, length_scale, p_order, s_order, dim, &coeffs)
                .expect("core");
            kernel[[i, j]] = core.phi.value;
            kernel[[j, i]] = core.phi.value;
        }
    }
    let amp2 = kernel_amp * kernel_amp;
    let kernel_gauge = gam_problem::Gauge::from_block_transforms(&[z.clone()]);
    let omega = kernel_gauge.restrict_penalty(&kernel).mapv(|v| v * amp2);
    let floored = duchon_range_floor_curvature(&omega, total_cols).expect("floor");

    let embed = |block: &Array2<f64>, ridge: f64| {
        let mut out = Array2::<f64>::zeros((total_cols, total_cols));
        out.slice_mut(s![..kernel_cols, ..kernel_cols]).assign(block);
        if poly_cols > 1 {
            for col in (kernel_cols + 1)..total_cols {
                out[[col, col]] = ridge;
            }
        }
        symmetrize(&out)
    };
    let mean_diag = |m: &Array2<f64>| -> f64 {
        (0..kernel_cols).map(|i| m[[i, i]]).sum::<f64>() / kernel_cols as f64
    };
    let curvature_scale = mean_diag(&floored).abs();
    let ridge = if poly_cols > 1 {
        if curvature_scale > 0.0 {
            DUCHON_AFFINE_NATIVE_RIDGE_REL * curvature_scale
        } else {
            DUCHON_AFFINE_NATIVE_RIDGE_REL
        }
    } else {
        0.0
    };
    let primary = embed(&floored, ridge);
    let c = fro(&primary);
    let primary_norm = primary.mapv(|v| v / c);
    (omega, floored, primary, primary_norm)
}

#[test]
fn zz_probe_2638_bisect() {
    let (data, spec) = fixture();
    let centers = select_centers_by_strategy(data.view(), &spec.center_strategy).expect("centers");
    let eps = 1e-5_f64;

    let derivative =
        build_duchon_basis_log_kappa_derivatives(data.view(), &spec).expect("analytic");
    eprintln!(
        "[2638] analytic penalties_derivative len = {}",
        derivative.first.penalties_derivative.len()
    );

    // Which sources?
    let mut workspace = BasisWorkspace::default();
    let effective = duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let z = kernel_constraint_nullspace(centers.view(), effective, &mut workspace.cache).expect("z");
    let candidates = duchon_native_penalty_candidates(
        centers.view(),
        spec.length_scale,
        spec.power,
        effective,
        None,
        &z,
        None,
    )
    .expect("candidates");
    eprintln!(
        "[2638] raw candidate sources = {:?}",
        candidates.iter().map(|c| c.source.clone()).collect::<Vec<_>>()
    );
    let filtered = filter_penalty_candidates(candidates).expect("filter");
    eprintln!(
        "[2638] filtered.active sources = {:?}",
        filtered
            .active
            .iter()
            .map(|c| c.info.source.clone())
            .collect::<Vec<_>>()
    );

    let (om0, fl0, pr0, prn0) = stages(centers.view(), &spec, 0.0);
    let (omp, flp, prp, prnp) = stages(centers.view(), &spec, eps);
    let (omm, flm, prm, prnm) = stages(centers.view(), &spec, -eps);
    eprintln!(
        "[2638] norms at psi=0: omega={:.6e} floored={:.6e} primary={:.6e} primary_norm={:.6e}",
        fro(&om0),
        fro(&fl0),
        fro(&pr0),
        fro(&prn0)
    );
    let fd_om = (&omp - &omm) / (2.0 * eps);
    let fd_fl = (&flp - &flm) / (2.0 * eps);
    let fd_pr = (&prp - &prm) / (2.0 * eps);
    let fd_prn = (&prnp - &prnm) / (2.0 * eps);
    eprintln!(
        "[2638] FD norms: d omega={:.6e} d floored={:.6e} d primary={:.6e} d primary_norm={:.6e}",
        fro(&fd_om),
        fro(&fd_fl),
        fro(&fd_pr),
        fro(&fd_prn)
    );

    // Now the analytic intermediates.
    let mut ws2 = BasisWorkspace::default();
    let (sources, first, _second) =
        build_duchon_native_penalty_psi_derivatives(centers.view(), &spec, None, &mut ws2)
            .expect("native psi derivs");
    eprintln!("[2638] native derivative sources = {sources:?}");
    for (i, f) in first.iter().enumerate() {
        eprintln!("[2638]   analytic first[{i}] norm = {:.6e}", fro(f));
    }

    // Compare the analytic Primary against the FD of the normalized primary.
    let err = fro(&(&first[0] - &fd_prn));
    eprintln!(
        "[2638] || analytic_first[0] - FD(primary_norm) || = {:.6e}  (FD norm {:.6e})",
        err,
        fro(&fd_prn)
    );

    // What does the shipped forward penalty (through the full builder) look like?
    let ls_plus = 1.0 / (1.0 * eps.exp());
    let ls_minus = 1.0 / (1.0 * (-eps).exp());
    let mut sp = spec.clone();
    let mut sm = spec.clone();
    sp.length_scale = Some(ls_plus);
    sm.length_scale = Some(ls_minus);
    let plus = build_duchon_basis(data.view(), &sp).expect("plus");
    let minus = build_duchon_basis(data.view(), &sm).expect("minus");
    let zero = build_duchon_basis(data.view(), &spec).expect("zero");
    eprintln!(
        "[2638] forward active_penalties: zero={} plus={} minus={}",
        zero.active_penalties.len(),
        plus.active_penalties.len(),
        minus.active_penalties.len()
    );
    for (i, p) in zero.active_penalties.iter().enumerate() {
        eprintln!(
            "[2638]   forward penalty[{i}] source={:?} dim={:?} fro={:.6e}",
            p.info.source.clone(),
            p.matrix.dim(),
            fro(&p.matrix)
        );
    }
    let fd_fwd = (&plus.active_penalties[0].matrix - &minus.active_penalties[0].matrix) / (2.0 * eps);
    eprintln!("[2638] FD(forward penalty 0) norm = {:.6e}", fro(&fd_fwd));
    eprintln!(
        "[2638] || FD(forward penalty0) - FD(primary_norm probe) || = {:.6e}",
        fro(&(&fd_fwd - &fd_prn))
    );
    eprintln!(
        "[2638] || forward penalty0(psi=0) - probe primary_norm || = {:.6e}",
        fro(&(&zero.active_penalties[0].matrix - &prn0))
    );
}

/// Freeze the FULL chart (centers + V + T) from a cold build, exactly as the
/// passing `_frozen` sibling does.
fn freeze_chart(data: ndarray::ArrayView2<'_, f64>, spec: &mut DuchonBasisSpec) -> BasisBuildResult {
    let base = build_duchon_basis(data, spec).expect("cold base build");
    if let BasisMetadata::Duchon {
        centers,
        identifiability_transform,
        radial_reparam,
        ..
    } = &base.metadata
    {
        spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
        spec.radial_reparam = radial_reparam.clone();
        spec.identifiability = match identifiability_transform {
            Some(t) => SpatialIdentifiability::FrozenTransform {
                transform: t.clone(),
            },
            None => SpatialIdentifiability::None,
        };
    } else {
        panic!("expected Duchon metadata");
    }
    base
}

#[test]
fn zz_probe_2638_chart_motion() {
    let (data, spec) = fixture();
    let eps = 1e-5_f64;

    let mut spec_f = spec.clone();
    let base_f = freeze_chart(data.view(), &mut spec_f);
    eprintln!(
        "[2638b] resolved chart: V={:?} T={:?}",
        spec_f.radial_reparam.as_ref().map(|v| v.dim()),
        match &spec_f.identifiability {
            SpatialIdentifiability::FrozenTransform { transform } => Some(transform.dim()),
            _ => None,
        }
    );

    // Cold (moving-chart) ±eps
    let mut sp = spec.clone();
    let mut sm = spec.clone();
    sp.length_scale = Some(1.0 / eps.exp());
    sm.length_scale = Some(1.0 / (-eps).exp());
    let cold_p = build_duchon_basis(data.view(), &sp).expect("cold +");
    let cold_m = build_duchon_basis(data.view(), &sm).expect("cold -");

    // Frozen-chart ±eps
    let mut fp = spec_f.clone();
    let mut fm = spec_f.clone();
    fp.length_scale = Some(1.0 / eps.exp());
    fm.length_scale = Some(1.0 / (-eps).exp());
    let frz_p = build_duchon_basis(data.view(), &fp).expect("frozen +");
    let frz_m = build_duchon_basis(data.view(), &fm).expect("frozen -");

    let analytic_raw = build_duchon_basis_log_kappa_derivatives(data.view(), &spec)
        .expect("analytic raw");
    let analytic_frz = build_duchon_basis_log_kappa_derivatives(data.view(), &spec_f)
        .expect("analytic frozen");

    for idx in 0..analytic_raw.first.penalties_derivative.len() {
        let fd_cold = (&cold_p.active_penalties[idx].matrix
            - &cold_m.active_penalties[idx].matrix)
            / (2.0 * eps);
        let fd_frz =
            (&frz_p.active_penalties[idx].matrix - &frz_m.active_penalties[idx].matrix)
                / (2.0 * eps);
        let a_raw = &analytic_raw.first.penalties_derivative[idx];
        let a_frz = &analytic_frz.first.penalties_derivative[idx];
        eprintln!(
            "[2638b] penalty {idx} ({:?}):\n  \
             |A_raw|={:.4e} |A_frz|={:.4e} |FD_cold|={:.4e} |FD_frz|={:.4e}\n  \
             |A_raw-FD_cold|={:.4e}  |A_frz-FD_frz|={:.4e}  |FD_cold-FD_frz|={:.4e}\n  \
             |S_cold(0)-S_frz(0)|={:.4e}",
            base_f.active_penalties[idx].info.source.clone(),
            fro(a_raw),
            fro(a_frz),
            fro(&fd_cold),
            fro(&fd_frz),
            fro(&(a_raw - &fd_cold)),
            fro(&(a_frz - &fd_frz)),
            fro(&(&fd_cold - &fd_frz)),
            fro(
                &(&build_duchon_basis(data.view(), &spec)
                    .expect("cold 0")
                    .active_penalties[idx]
                    .matrix
                    - &base_f.active_penalties[idx].matrix)
            ),
        );
    }
}
