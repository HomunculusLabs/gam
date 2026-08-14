//! gam#2747: WHICH shipped bases actually contain the constant they are centered against?
//!
//! `smooth_requires_parametric_orthogonality` names five basis classes and
//! justifies all of them with one sentence — *"their realized column span
//! contains the constant … so without this step the smooth and the parametric
//! intercept fight over the same direction — a structural rank-1 collision"*.
//! `76a520c45` measured that sentence to be FALSE for the constant-curvature
//! geodesic kernel and made the gate a per-direction containment test, which
//! silently changes the model for every other member of the class whose span
//! also fails to contain the constant.
//!
//! That blast radius was never measured: `76a520c45`'s evidence is
//! `cargo test -p gam-terms --lib`, which contains no realized-design
//! containment measurement for any of these bases.
//!
//! This probe measures it directly, one number per basis:
//!
//! ```text
//! containment residual = ||1 - P_X 1|| / ||1||
//! ```
//!
//! zero iff the constant is exactly in the realized span. The shipped gate keeps
//! a direction when that residual is `<= sqrt(eps) = 1.49e-8`.
//!
//! Run: `cargo run --release --example probe_2747_containment_registry`

use gam_terms::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability, DuchonBasisSpec,
    DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec, MaternIdentifiability,
    MaternLengthScale, MaternNu, OneDimensionalBoundary, SpatialIdentifiability, ThinPlateBasisSpec,
    build_constant_curvature_basis, build_duchon_basis, build_matern_basis, build_thin_plate_basis,
};
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2};

const N: usize = 400;
const CENTERS: usize = 24;

/// A deterministic 2-D cloud in the unit disk — the shape every spatial fixture
/// in this lane uses.
fn cloud() -> Array2<f64> {
    let mut state = 0x2747_0000_0000_0001_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Array2::<f64>::zeros((N, 2));
    let mut filled = 0usize;
    while filled < N {
        let a = 2.0 * next() - 1.0;
        let b = 2.0 * next() - 1.0;
        if a * a + b * b <= 1.0 {
            out[(filled, 0)] = a;
            out[(filled, 1)] = b;
            filled += 1;
        }
    }
    out
}

/// `||1 - P_X 1|| / ||1||` on the realized design `X`, through the design Gram's
/// own truncated spectrum so a rank-deficient `X` is handled rather than
/// factorized into a failure.
fn containment_residual(design: &Array2<f64>) -> f64 {
    let n = design.nrows();
    let ones = Array1::<f64>::ones(n);
    let gram = design.t().dot(design);
    let rhs = design.t().dot(&ones);
    let (evals, evecs) = FaerEigh::eigh(&gram, faer::Side::Lower).expect("design gram spectrum");
    let top = evals.iter().cloned().fold(0.0_f64, f64::max);
    let projected = evecs.t().dot(&rhs);
    let mut solved = Array1::<f64>::zeros(projected.len());
    for i in 0..projected.len() {
        if evals[i] > top * (design.ncols() as f64) * f64::EPSILON {
            solved[i] = projected[i] / evals[i];
        }
    }
    let fitted = design.dot(&evecs.dot(&solved));
    let residual = &ones - &fitted;
    residual.dot(&residual).sqrt() / ones.dot(&ones).sqrt()
}

fn report(label: &str, design: &Array2<f64>) {
    let residual = containment_residual(design);
    let bar = f64::EPSILON.sqrt();
    println!(
        "{label:<44} {:>4}x{:<4}  residual = {residual:.6e}   {}",
        design.nrows(),
        design.ncols(),
        if residual <= bar {
            "CONTAINED   (deletion is free)"
        } else {
            "NOT contained (deletion loses a function)"
        }
    );
}

fn main() {
    let data = cloud();
    println!(
        "containment bar = sqrt(eps) = {:.6e}\n",
        f64::EPSILON.sqrt()
    );

    for identifiability in [
        MaternIdentifiability::CenterSumToZero,
        MaternIdentifiability::CenterLinearOrthogonal,
    ] {
        let name = format!("{identifiability:?}");
        for nu in [MaternNu::ThreeHalves, MaternNu::FiveHalves] {
            for length_scale in [0.2_f64, 1.0, 10.0] {
                let spec = MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: CENTERS,
                    },
                    periodic: None,
                    length_scale: MaternLengthScale::fixed(length_scale),
                    nu,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: identifiability.clone(),
                    aniso_log_scales: None,
                };
                match build_matern_basis(data.view(), &spec) {
                    Ok(basis) => report(
                        &format!("matern({nu:?}, ell={length_scale}, {name})"),
                        &basis.design.to_dense(),
                    ),
                    Err(err) => {
                        println!("matern({nu:?}, ell={length_scale}, {name}): refused: {err}")
                    }
                }
            }
        }
    }

    let tps = ThinPlateBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: CENTERS,
        },
        periodic: None,
        length_scale: 1.0,
        double_penalty: false,
        identifiability: SpatialIdentifiability::None,
        radial_reparam: None,
    };
    match build_thin_plate_basis(data.view(), &tps) {
        Ok(basis) => report("thinplate(OrthogonalToParametric off)", &basis.design.to_dense()),
        Err(err) => println!("thinplate: refused: {err}"),
    }

    let duchon = DuchonBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: CENTERS,
        },
        periodic: None,
        length_scale: None,
        power: 0.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
        radial_reparam: None,
    };
    match build_duchon_basis(data.view(), &duchon) {
        Ok(basis) => report("duchon(OrthogonalToParametric off)", &basis.design.to_dense()),
        Err(err) => println!("duchon: refused: {err}"),
    }

    for length_scale in [0.2_f64, 0.5, 1.0, 3.0, 100.0] {
        for kappa in [-1.0_f64, 0.0, 1.0] {
            let spec = ConstantCurvatureBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint {
                    num_centers: CENTERS,
                },
                kappa,
                kappa_fixed: true,
                length_scale,
                length_scale_fixed: true,
                double_penalty: false,
                identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
            };
            match build_constant_curvature_basis(data.view(), &spec) {
                Ok(basis) => report(
                    &format!("curv(kappa={kappa:+.1}, ell={length_scale})"),
                    &basis.design.to_dense(),
                ),
                Err(err) => println!("curv(kappa={kappa:+.1}, ell={length_scale}): refused: {err}"),
            }
        }
    }
}
