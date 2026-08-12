//! gam#2747 probe: is the range coordinate's runaway a property of the DATA or
//! of the PARAMETRIZATION?
//!
//! `20bde053f` reverted the pinned-κ/free-range enrollment with this
//! measurement: *"on the capacity-starved kappa=1 sphere fixture the criterion
//! is monotone in ell all the way to its asymptote — the design converges to
//! its linear-in-distance limit, V goes flat, and ell-hat ran to 1.5e6, a
//! readout of the box rather than of the data."*
//!
//! The shipped realized design is `X = K_{κ,ℓ}(data, C)·z` with
//! `K = exp(−d_κ/ℓ)` and `z` summing to zero over centers, so constants are
//! annihilated and
//!
//! ```text
//!   X = (exp(−d/ℓ) − 1)z = −(1/ℓ)·D z + O(1/ℓ²),
//!   S = zᵀ K z          = −(1/ℓ)·zᵀD z + O(1/ℓ²).
//! ```
//!
//! Both blocks collapse like `1/ℓ`. The model's prior on the fitted function is
//! `(1/λ)·X S⁻ Xᵀ`, which is invariant under `(X, S, λ) → (cX, cS, cλ)`, so a
//! range move is EXACTLY compensated by `ρ̂ → ρ̂ + ln c` — the profile file
//! already records the symptom (`ρ̂ ≈ const − ln ℓ`, "each ×100 in ℓ costs 4.6
//! in ρ̂") and works around it by refusing points whose `ρ̂` railed.
//!
//! Three arms, one dataset, so the claim is separable:
//!
//! * **RAW** — the shipped kernel. Expect `ρ̂` to fall one-for-one with `ln ℓ`
//!   and to reach `RHO_LOWER = −30`.
//! * **NORM** — the same model with `(X, S)` both multiplied by `ℓ`, built
//!   STABLY as `−ℓ·expm1(−d/ℓ)` rather than by scaling the cancelled `K z`.
//!   This is a pure reparametrization: the REML value must be bit-comparable to
//!   RAW wherever RAW's `ρ̂` is interior, and `ρ̂` must stop drifting.
//! * **LIMIT** — the `ℓ → ∞` face, `X = −D(data,C) z`, `S = −zᵀD(C,C)z`, the
//!   geodesic-distance (conditionally negative definite) kernel. If NORM
//!   converges to LIMIT then the asymptote is a MODEL, not a numerical wall.
//!
//! Run: `cargo run --release --example probe_2747_range_normalization`

use gam_geometry::manifolds::constant_curvature::ConstantCurvature;
use gam_terms::basis::{
    BasisMetadata, CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    build_constant_curvature_basis, constant_curvature_length_scale_bounds,
    constant_curvature_realized_centers, realized_constant_curvature_length_scale,
};
use gam_terms::analytic_penalties::PenaltyOp;
use ndarray::{Array1, Array2, ArrayView2, s};

const KAPPA: f64 = 1.0;
const CENTERS: usize = 30;
const ROWS: usize = 400;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// The `kappa_one_fit_recovers_planted_spherical_signal` fixture's geometry:
/// 400 chart points in the radius-0.9 disk, response `p_z + 0.5·p_x` on the
/// embedded sphere plus a little noise.
fn fixture() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut state = 945_u64;
    let mut feats = Array2::<f64>::zeros((ROWS, 2));
    let mut truth = Array1::<f64>::zeros(ROWS);
    let mut y = Array1::<f64>::zeros(ROWS);
    for i in 0..ROWS {
        let r = 0.9 * unit(&mut state).sqrt();
        let th = std::f64::consts::TAU * unit(&mut state);
        let x1 = r * th.cos();
        let x2 = r * th.sin();
        let r2 = x1 * x1 + x2 * x2;
        let pz = (1.0 - r2) / (1.0 + r2);
        let px = 2.0 * x1 / (1.0 + r2);
        feats[(i, 0)] = x1;
        feats[(i, 1)] = x2;
        truth[i] = pz + 0.5 * px;
        y[i] = truth[i] + 0.05 * (unit(&mut state) - 0.5);
    }
    (feats, truth, y)
}

fn spec_at(length_scale: f64) -> ConstantCurvatureBasisSpec {
    ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: CENTERS,
        },
        kappa: KAPPA,
        kappa_fixed: true,
        length_scale,
        length_scale_fixed: false,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    }
}

/// Pairwise geodesic distances, `rows(a) × rows(b)`.
fn distances(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    let manifold = ConstantCurvature::new(a.ncols(), KAPPA);
    let mut out = Array2::<f64>::zeros((a.nrows(), b.nrows()));
    for i in 0..a.nrows() {
        for j in 0..b.nrows() {
            out[(i, j)] = manifold
                .distance(a.row(i), b.row(j))
                .expect("fixture points are inside the κ = 1 chart");
        }
    }
    out
}

struct Arm {
    value: f64,
    rho: f64,
    r2: f64,
    edf: f64,
}

/// One REML fit of `[1 | X]` against `y` with the smooth block penalized by
/// `S`, exactly as `constant_curvature_psi_profile_value` assembles it.
fn reml(x: &Array2<f64>, s: &Array2<f64>, y: &Array1<f64>, truth: &Array1<f64>) -> Option<Arm> {
    let (n, p) = x.dim();
    let mut design = Array2::<f64>::ones((n, p + 1));
    design.slice_mut(s![.., 1..]).assign(x);
    let mut penalty = Array2::<f64>::zeros((p + 1, p + 1));
    penalty.slice_mut(s![1.., 1..]).assign(s);
    let response = y.clone().insert_axis(ndarray::Axis(1));
    let fit = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
        design.view(),
        response.view(),
        penalty.view(),
        None,
        None,
    )
    .ok()?;
    let pred = fit.fitted.column(0).to_owned();
    let mean = truth.sum() / truth.len() as f64;
    let ss_res: f64 = truth
        .iter()
        .zip(pred.iter())
        .map(|(t, p)| (t - p) * (t - p))
        .sum();
    let ss_tot: f64 = truth.iter().map(|t| (t - mean) * (t - mean)).sum();
    Some(Arm {
        value: fit.reml_score,
        rho: fit.rho,
        r2: 1.0 - ss_res / ss_tot,
        edf: fit.edf,
    })
}

fn main() {
    let (feats, truth, y) = fixture();
    let spec = spec_at(0.0);
    let centers = constant_curvature_realized_centers(feats.view(), &spec).expect("centers");
    let ell_ref =
        realized_constant_curvature_length_scale(centers.view(), 0.0).expect("auto length scale");
    let (lo, hi) =
        constant_curvature_length_scale_bounds(feats.view(), centers.view()).expect("range box");
    println!("centers={CENTERS} rows={ROWS} kappa={KAPPA}");
    println!("auto ell_ref = {ell_ref:.6}   derived box = [{lo:.6e}, {hi:.6e}]");
    println!(
        "                                 rho box = [{}, {}]",
        gam_solve::gaussian_reml::RHO_LOWER,
        gam_solve::gaussian_reml::RHO_UPPER
    );

    // The constraint transform the builder itself realizes, so every arm below
    // is the SAME model in the same gauge.
    let built = build_constant_curvature_basis(feats.view(), &spec_at(ell_ref)).expect("build");
    let z = match &built.metadata {
        BasisMetadata::ConstantCurvature {
            constraint_transform: Some(z),
            ..
        } => z.clone(),
        _ => panic!("the constant-curvature builder publishes its constraint transform"),
    };

    let d_dc = distances(feats.view(), centers.view());
    let d_cc = distances(centers.view(), centers.view());

    // LIMIT: the ℓ → ∞ face. `−d` is conditionally positive definite on the
    // sum-to-zero frame, so `S` is genuinely PD here.
    let x_limit = d_dc.mapv(|v| -v).dot(&z);
    let s_limit_raw = z.t().dot(&d_cc.mapv(|v| -v)).dot(&z);
    let s_limit = (&s_limit_raw + &s_limit_raw.t()) * 0.5;
    let limit = reml(&x_limit, &s_limit, &y, &truth);

    println!(
        "\n{:>12}  {:>12} {:>9} {:>7} {:>7} {:>4}   {:>12} {:>9} {:>7} {:>7}  {:>9}",
        "ell", "V_raw", "rho_raw", "R2_raw", "edf_raw", "rail", "V_norm", "rho_norm", "R2_norm",
        "edf_norm", "cancel"
    );
    let grid: Vec<f64> = {
        let mut g = Vec::new();
        let mut e = 1.0e-2_f64;
        while e <= 1.0e9 {
            g.push(e);
            g.push(e * 3.0);
            e *= 10.0;
        }
        g.push(hi);
        g.sort_by(|a, b| a.partial_cmp(b).expect("finite grid"));
        g
    };
    for ell in grid {
        let raw_blocks = build_constant_curvature_basis(feats.view(), &spec_at(ell))
            .ok()
            .map(|b| {
                (
                    b.design.to_dense(),
                    b.active_penalties[0].matrix.as_dense().to_owned(),
                )
            });
        let raw = raw_blocks
            .as_ref()
            .and_then(|(x, s)| reml(x, s, &y, &truth));

        // NORM: the same kernel with the `1/ℓ` factor removed, built stably.
        // `−ℓ·expm1(−d/ℓ) → d` as `ℓ → ∞` with no cancellation.
        let g_dc = d_dc.mapv(|d| -ell * (-d / ell).exp_m1());
        let g_cc = d_cc.mapv(|d| -ell * (-d / ell).exp_m1());
        let x_norm = g_dc.mapv(|v| -v).dot(&z);
        let s_norm_raw = z.t().dot(&g_cc.mapv(|v| -v)).dot(&z);
        let s_norm = (&s_norm_raw + &s_norm_raw.t()) * 0.5;
        let norm = reml(&x_norm, &s_norm, &y, &truth);

        let rail = raw.as_ref().is_some_and(|a| {
            (a.rho - gam_solve::gaussian_reml::RHO_LOWER).abs() <= 1e-9
                || (a.rho - gam_solve::gaussian_reml::RHO_UPPER).abs() <= 1e-9
        });
        let fmt = |a: &Option<Arm>| match a {
            Some(a) => format!(
                "{:>12.5} {:>9.4} {:>7.4} {:>7.3}",
                a.value, a.rho, a.r2, a.edf
            ),
            None => format!("{:>12} {:>9} {:>7} {:>7}", "refused", "-", "-", "-"),
        };
        // How much of `X_raw·ℓ` survives? The raw path forms `exp(−d/ℓ)` whose
        // entries all approach 1, then annihilates the constant with `z` — so
        // the signal is what is left after a cancellation of relative size
        // `d/ℓ`. The normalized path never forms the difference.
        let scaled = x_norm.mapv(|v| v);
        let cancellation = match &raw_blocks {
            Some((x_raw, _)) => {
                let mut num = 0.0_f64;
                let mut den = 0.0_f64;
                for (a, b) in x_raw.iter().zip(scaled.iter()) {
                    num += (a * ell - b) * (a * ell - b);
                    den += b * b;
                }
                (num / den).sqrt()
            }
            None => f64::NAN,
        };
        println!(
            "{ell:>12.4e}  {} {:>4}   {}  {cancellation:>9.2e}",
            fmt(&raw),
            if rail { "YES" } else { "" },
            fmt(&norm)
        );
    }

    println!(
        "\n{:>12}  {:>12} {:>9} {:>7} {:>7}",
        "ell -> inf", "V", "rho", "R2", "edf"
    );
    match limit {
        Some(a) => println!(
            "{:>12}  {:>12.5} {:>9.4} {:>7.4} {:>7.3}",
            "d-kernel", a.value, a.rho, a.r2, a.edf
        ),
        None => println!("{:>12}  refused", "d-kernel"),
    }
}
