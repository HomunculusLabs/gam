use super::*;
use ndarray::{Array1, Array2};

/// The #2660 production shape: 240 rows, five active latent coordinates per
/// row, and the composed narrow co-fit's 35 decoder coefficients.
///
/// The declared non-axis beta direction is an exact orbit of the row cross
/// blocks (`H_tbeta q = 0`) and the shared block is isotropic, while `g_beta`
/// deliberately contains a large orbit component. An unquotiented solve
/// therefore returns a visible `q^T delta_beta`; the Faddeev--Popov solve must
/// erase it while preserving the identifiable complement.
fn production_shaped_gauge_system() -> (ArrowSchurSystem, Array1<f64>) {
    let n = 240usize;
    let d = 5usize;
    let k = 35usize;
    let raw_direction = Array1::from_shape_fn(k, |index| 0.5 + ((index as f64 + 1.0) * 0.37).sin());
    let quotient =
        ArrowBetaGaugeQuotient::new(vec![raw_direction]).expect("independent beta gauge");
    let direction = quotient.directions[0].clone();

    let mut sys = ArrowSchurSystem::new(n, d, k);
    for (row_index, row) in sys.rows.iter_mut().enumerate() {
        let mut htt = Array2::<f64>::zeros((d, d));
        for axis in 0..d {
            htt[[axis, axis]] = 3.0 + 0.2 * axis as f64;
            for other in 0..axis {
                let coupling = 0.01 / (1.0 + (axis - other) as f64);
                htt[[axis, other]] = coupling;
                htt[[other, axis]] = coupling;
            }
            row.gt[axis] = 0.03 * (((row_index + 1) * (axis + 2)) as f64 * 0.017).sin();
        }
        row.htt = htt;

        for axis in 0..d {
            let mut cross = Array1::from_shape_fn(k, |beta| {
                0.002 * (((row_index + 3) * (axis + 5) * (beta + 7)) as f64 * 0.0013).cos()
            });
            let orbit_component = cross.dot(&direction);
            cross.scaled_add(-orbit_component, &direction);
            for beta in 0..k {
                row.htbeta[[axis, beta]] = cross[beta];
            }
            assert!(
                row.htbeta.row(axis).dot(&direction).abs() <= 2.0e-16,
                "fixture cross row must annihilate the declared beta orbit"
            );
        }
    }
    sys.hbb = Array2::from_diag(&Array1::from_elem(k, 12.0));

    let raw_identifiable =
        Array1::from_shape_fn(k, |index| 0.04 * ((index as f64 + 1.0) * 0.11).cos());
    let mut identifiable = quotient.project_complement(raw_identifiable.view());
    identifiable.scaled_add(0.5, &direction);
    sys.gb = identifiable;
    assert!(
        direction.dot(&sys.gb).abs() >= 0.49,
        "fixture needs a discriminating raw gauge-gradient component"
    );
    sys.set_beta_gauge_quotient(quotient)
        .expect("quotient width matches production border");
    (sys, direction)
}

fn assert_quotient_device_parity(
    label: &str,
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    gauge_direction: &Array1<f64>,
    cpu_t: &Array1<f64>,
    cpu_beta: &Array1<f64>,
    device_t: &Array1<f64>,
    device_beta: &Array1<f64>,
) {
    let orbit_step = gauge_direction.dot(device_beta).abs();
    assert!(
        orbit_step <= 1.0e-12,
        "#2660 {label} leaked beta-gauge motion: |q^T delta_beta|={orbit_step:e}"
    );

    let max_t_error = device_t
        .iter()
        .zip(cpu_t.iter())
        .map(|(device, cpu)| (device - cpu).abs() / (1.0 + cpu.abs()))
        .fold(0.0_f64, f64::max);
    let max_beta_error = device_beta
        .iter()
        .zip(cpu_beta.iter())
        .map(|(device, cpu)| (device - cpu).abs() / (1.0 + cpu.abs()))
        .fold(0.0_f64, f64::max);
    assert!(
        max_t_error <= 1.0e-9 && max_beta_error <= 1.0e-9,
        "#2660 {label} != CPU quotient Direct: max_rel_t={max_t_error:e}, \
         max_rel_beta={max_beta_error:e}"
    );

    let backward_error = arrow_quotient_backward_error_certificate(
        sys,
        ridge_t,
        ridge_beta,
        device_t.view(),
        device_beta.view(),
    )
    .unwrap_or_else(|error| panic!("#2660 {label} backward-error certificate failed: {error}"));
    assert!(
        backward_error <= 1.0e-10,
        "#2660 {label} quotient backward error is {backward_error:e}"
    );
}

#[test]
fn device_direct_applies_beta_gauge_quotient_at_composed_cofit_shape_2660() {
    let runtime = gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto)
        .unwrap_or_else(|error| panic!("#2660 CUDA runtime probe failed: {error}"));
    if runtime.is_none() {
        eprintln!("[#2660] no CUDA runtime; skipping device quotient parity");
        return;
    }

    let (sys, gauge_direction) = production_shaped_gauge_system();
    let ridge_t = 1.0e-7;
    let ridge_beta = 1.0e-6;

    let cpu_options = ArrowSolveOptions::direct().with_gpu_policy(gam_gpu::GpuPolicy::Off);
    let (cpu_t, cpu_beta, cpu_diag) =
        solve_arrow_newton_step_core(&sys, ridge_t, ridge_beta, &cpu_options)
            .expect("CPU quotient Direct solve");
    assert!(
        !cpu_diag.used_device_arrow,
        "GpuPolicy::Off CPU oracle must not report device execution"
    );

    let device_options = ArrowSolveOptions::direct().with_gpu_policy(gam_gpu::GpuPolicy::Required);
    let (device_t, device_beta, device_diag) =
        solve_arrow_newton_step_core(&sys, ridge_t, ridge_beta, &device_options)
            .expect("device quotient Direct solve");
    assert!(
        device_diag.used_device_arrow,
        "#2660 production shape cleared Device Direct but telemetry says it ran on the host"
    );
    assert_quotient_device_parity(
        "re-upload/fused Direct",
        &sys,
        ridge_t,
        ridge_beta,
        &gauge_direction,
        &cpu_t,
        &cpu_beta,
        &device_t,
        &device_beta,
    );

    let cpu_backward_error = arrow_quotient_backward_error_certificate(
        &sys,
        ridge_t,
        ridge_beta,
        cpu_t.view(),
        cpu_beta.view(),
    )
    .expect("CPU quotient backward-error certificate");
    assert!(
        device_diag.final_relative_residual <= 1.0e-10 && cpu_backward_error <= 1.0e-10,
        "#2660 quotient residual failed: device={:e}, cpu={cpu_backward_error:e}",
        device_diag.final_relative_residual,
    );

    let g_t: Vec<f64> = sys
        .rows
        .iter()
        .flat_map(|row| row.gt.iter().copied())
        .collect();
    let g_beta = sys.gb.to_vec();

    let fixed_frame =
        crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle::new(&sys, ridge_t, ridge_beta)
            .expect("ridge-fixed resident quotient frame");
    let fixed = fixed_frame
        .solve_gradient(&g_t, &g_beta)
        .expect("ridge-fixed resident quotient solve");
    assert_quotient_device_parity(
        "ridge-fixed resident Direct",
        &sys,
        ridge_t,
        ridge_beta,
        &gauge_direction,
        &cpu_t,
        &cpu_beta,
        &fixed.delta_t,
        &fixed.delta_beta,
    );

    let base_frame = crate::gpu_kernels::arrow_schur::ResidentBaseArrowFrameHandle::new(&sys)
        .expect("ridge-keyed resident quotient base frame");
    let keyed = base_frame
        .refactor_and_solve_with_gradient(ridge_t, ridge_beta, &g_t, &g_beta)
        .expect("ridge-keyed resident quotient solve");
    assert_quotient_device_parity(
        "ridge-keyed resident Direct",
        &sys,
        ridge_t,
        ridge_beta,
        &gauge_direction,
        &cpu_t,
        &cpu_beta,
        &keyed.delta_t,
        &keyed.delta_beta,
    );
}
