//! #2933 F25 — the ARD coordinate prior must descend to the quotient each
//! quotient atom's cover chart represents.
//!
//! A density on a quotient takes one value on every deck orbit. The decoder bases
//! already respect the deck identifications, and their own tests pin that. These
//! pins ask the same of the whole objective the fit minimizes (data fit plus
//! coordinate prior) on the dense and the support-sparse routes, and of the
//! prior's normalization over the quotient:
//!
//! * Möbius band, cover `Circle{2} × [-1, 1]`: `(s, w) ~ (s + 1, -w)`.
//! * Klein bottle, cover `T²`: `(theta, phi) ~ (theta + 1/2, -phi)`.
//! * `RP²` on the `(lat, lon)` cover: `(lat, lon) ~ (-lat, lon + π)`.
//! * `RP²` on the ambient cover: `u ~ -u`. Its ambient quadratic is already even,
//!   so this case is the control a repair must not disturb.
//!
//! The deck maps are written out here from the audit's statement rather than read
//! from `SaeAtomBasisKind::deck_generator`, so the pins are free to disagree with
//! the implementation. Each case also carries a representative of a DIFFERENT
//! point, and the prior must tell that one apart, so an invariance that holds
//! because the prior went flat does not pass.
#![cfg(test)]
use super::*;
use crate::assignment::AssignmentMode;
use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};
use gam_problem::LatentRetractionRegistry;
use ndarray::{Array1, Array2, array};
use std::f64::consts::PI;
use std::sync::Arc;

const OUTPUT_DIM: usize = 3;

/// One quotient atom on its cover chart.
struct QuotientCase {
    name: &'static str,
    kind: SaeAtomBasisKind,
    latent_dim: usize,
    evaluator: Arc<dyn SaeBasisSecondJet>,
    coords: Array2<f64>,
    /// The deck generator per axis as `(sign, shift, wrap period)`.
    deck: Vec<(f64, f64, Option<f64>)>,
    /// `coords` with row 0 moved to a point NOT deck-equivalent to it. Moving one
    /// row keeps the control's prior change from cancelling across rows.
    other_points: Array2<f64>,
    log_ard: Array1<f64>,
}

impl QuotientCase {
    fn twin(&self) -> Array2<f64> {
        let mut out = self.coords.clone();
        for row in 0..out.nrows() {
            for (axis, &(sign, shift, period)) in self.deck.iter().enumerate() {
                let moved = sign * self.coords[[row, axis]] + shift;
                out[[row, axis]] = period.map_or(moved, |p| moved.rem_euclid(p));
            }
        }
        out
    }

    fn sign(&self, axis: usize) -> f64 {
        self.deck[axis].0
    }
}

fn quotient_cases() -> Vec<QuotientCase> {
    vec![
        QuotientCase {
            name: "mobius",
            kind: SaeAtomBasisKind::Mobius,
            latent_dim: 2,
            evaluator: Arc::new(MobiusHarmonicEvaluator::new(2, 2).expect("mobius basis")),
            coords: array![[0.13, 0.41], [0.62, -0.27], [1.37, 0.08], [0.91, -0.66]],
            deck: vec![(1.0, 1.0, Some(2.0)), (-1.0, 0.0, None)],
            other_points: array![[0.5, 0.41], [0.62, -0.27], [1.37, 0.08], [0.91, -0.66]],
            log_ard: array![3.1_f64.ln(), 1.4_f64.ln()],
        },
        QuotientCase {
            name: "klein_bottle",
            kind: SaeAtomBasisKind::KleinBottle,
            latent_dim: 2,
            evaluator: Arc::new(QuotientSpectralEvaluator::klein_bottle(2).expect("klein basis")),
            coords: array![[0.17, 0.23], [0.44, 0.71], [0.08, 0.52], [0.36, 0.94]],
            deck: vec![(1.0, 0.5, Some(1.0)), (-1.0, 0.0, Some(1.0))],
            other_points: array![[0.42, 0.23], [0.44, 0.71], [0.08, 0.52], [0.36, 0.94]],
            log_ard: array![40.0_f64.ln(), 2.5_f64.ln()],
        },
        QuotientCase {
            name: "projective_plane_chart",
            kind: SaeAtomBasisKind::ProjectivePlane,
            latent_dim: 2,
            evaluator: Arc::new(
                QuotientSpectralEvaluator::projective_plane(1).expect("rp2 chart basis"),
            ),
            coords: array![[0.31, 0.7], [-0.52, 2.4], [0.94, 4.1], [-0.18, 5.6]],
            deck: vec![(-1.0, 0.0, None), (1.0, PI, Some(2.0 * PI))],
            other_points: array![[0.31, 0.7 + 0.5 * PI], [-0.52, 2.4], [0.94, 4.1], [-0.18, 5.6]],
            log_ard: array![2.2_f64.ln(), 1.9_f64.ln()],
        },
        QuotientCase {
            name: "projective_plane_ambient",
            kind: SaeAtomBasisKind::ProjectivePlane,
            latent_dim: 3,
            evaluator: Arc::new(
                QuotientSpectralEvaluator::projective_plane_ambient(1).expect("rp2 ambient basis"),
            ),
            coords: array![
                [0.36, -0.48, 0.8],
                [0.6, 0.0, 0.8],
                [-0.28, 0.96, 0.0],
                [0.0, 0.6, -0.8]
            ],
            deck: vec![(-1.0, 0.0, None); 3],
            // Row 0 cyclically permuted: a unit vector at a different point, which
            // the anisotropic precisions below charge differently.
            other_points: array![
                [-0.48, 0.8, 0.36],
                [0.6, 0.0, 0.8],
                [-0.28, 0.96, 0.0],
                [0.0, 0.6, -0.8]
            ],
            log_ard: array![1.3_f64.ln(), 2.1_f64.ln(), 0.7_f64.ln()],
        },
    ]
}

fn decoder(width: usize) -> Array2<f64> {
    Array2::from_shape_fn((width, OUTPUT_DIM), |(basis, out)| {
        0.6 * (0.37 * (7 * basis + 3 * out + 1) as f64).sin()
    })
}

fn target(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, OUTPUT_DIM), |(row, out)| {
        0.3 * ((row + 2 * out) as f64).cos()
    })
}

fn quotient_atom(case: &QuotientCase, coords: &Array2<f64>) -> SaeManifoldAtom {
    let (phi, jet) = case.evaluator.evaluate(coords.view()).expect("evaluate cover basis");
    let width = phi.ncols();
    SaeManifoldAtom::new_with_provided_function_gram(
        case.name,
        case.kind.clone(),
        case.latent_dim,
        phi,
        jet,
        decoder(width),
        Array2::<f64>::eye(width),
    )
    .expect("quotient atom")
    .with_basis_second_jet(case.evaluator.clone())
}

fn dense_term(case: &QuotientCase, coords: &Array2<f64>) -> SaeManifoldTerm {
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::from_shape_fn((coords.nrows(), 1), |(row, _)| 0.2 * row as f64 - 0.3),
        vec![coords.clone()],
        vec![case.kind.latent_manifold(case.latent_dim)],
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, true),
    )
    .expect("dense assignment");
    SaeManifoldTerm::new(vec![quotient_atom(case, coords)], assignment).expect("dense term")
}

fn support_term(case: &QuotientCase, coords: &Array2<f64>) -> SaeSupportSparseTerm {
    let n = coords.nrows();
    let spec = SaeAssignmentAtomSpec {
        latent_dim: case.latent_dim,
        manifold: case.kind.latent_manifold(case.latent_dim),
        retraction: LatentRetractionRegistry::all_euclidean(),
    };
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        n,
        1,
        1,
        vec![spec],
        vec![vec![0]; n],
        vec![vec![1.0]; n],
        (0..n).map(|row| coords.row(row).to_vec()).collect(),
    )
    .expect("support state");
    SaeSupportSparseTerm::new(vec![quotient_atom(case, coords)], state).expect("support term")
}

fn assert_close(label: &str, twin: f64, expected: f64) {
    let scale = 1.0 + twin.abs().max(expected.abs());
    assert!(
        (twin - expected).abs() <= 1.0e-9 * scale,
        "{label}: deck twin {twin:.17e} vs cover representative {expected:.17e}"
    );
}

/// Compare two assembled Arrow systems at deck-equivalent rows. The deck acts on
/// coordinate slot `a` by `sign_a`, so the transformed gradient is `sign_a·g_a`,
/// the Hessian action is `sign_a·sign_b·H_ab`, and the coordinate/decoder cross
/// block is `sign_a·H_aβ`. Leading gate slots are untouched by the deck.
fn assert_systems_transform(
    label: &str,
    case: &QuotientCase,
    cover: &ArrowSchurSystem,
    twin: &ArrowSchurSystem,
) {
    assert_eq!(cover.rows.len(), twin.rows.len());
    for (row, (c, t)) in cover.rows.iter().zip(twin.rows.iter()).enumerate() {
        let q = c.gt.len();
        assert_eq!(t.gt.len(), q);
        assert!(q >= case.latent_dim);
        let gate_width = q - case.latent_dim;
        let sign = |slot: usize| {
            if slot < gate_width {
                1.0
            } else {
                case.sign(slot - gate_width)
            }
        };
        for a in 0..q {
            assert_close(
                &format!("{label} {} row {row} gradient[{a}]", case.name),
                t.gt[a],
                sign(a) * c.gt[a],
            );
            for b in 0..q {
                assert_close(
                    &format!("{label} {} row {row} Hessian[{a},{b}]", case.name),
                    t.htt[[a, b]],
                    sign(a) * sign(b) * c.htt[[a, b]],
                );
            }
            for beta in 0..c.htbeta.ncols() {
                assert_close(
                    &format!("{label} {} row {row} cross[{a},{beta}]", case.name),
                    t.htbeta[[a, beta]],
                    sign(a) * c.htbeta[[a, beta]],
                );
            }
        }
    }
}

/// Dense route: reconstruction, prior value and total agree at deck-equivalent
/// representatives on every quotient cover. The transformed gradient and Hessian
/// action agree on every cover the dense assembly admits, and the charted `RP²` it
/// does not admit is refused with its typed error.
#[test]
fn dense_objective_is_deck_invariant_on_every_quotient_2933() {
    for case in quotient_cases() {
        let n = case.coords.nrows();
        let target = target(n);
        let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![case.log_ard.clone()]);
        let mut cover = dense_term(&case, &case.coords);
        let mut twin = dense_term(&case, &case.twin());

        let cover_loss = cover.loss(target.view(), &rho).expect("cover loss");
        let twin_loss = twin.loss(target.view(), &rho).expect("twin loss");
        assert_close(
            &format!("{} data fit", case.name),
            twin_loss.data_fit,
            cover_loss.data_fit,
        );
        assert_close(&format!("{} ARD prior", case.name), twin_loss.ard, cover_loss.ard);
        assert_close(
            &format!("{} total objective", case.name),
            twin_loss.total(),
            cover_loss.total(),
        );

        let other_loss = dense_term(&case, &case.other_points)
            .loss(target.view(), &rho)
            .expect("other-point loss");
        assert!(
            (other_loss.ard - cover_loss.ard).abs() > 1.0e-2,
            "{}: the prior must distinguish a non-equivalent point (cover {}, other {})",
            case.name,
            cover_loss.ard,
            other_loss.ard
        );

        // The dense Arrow route takes a spherical kind only on its ambient cover:
        // `push_atom_row_gauge_deflations` refuses a charted `RP²` with a typed
        // error. Pin that refusal at both representatives, so a dense route that
        // admits the chart, or panics on it, fails here and brings the assembly
        // comparison back. The chart's assembly is compared on the support-sparse
        // route.
        if case.kind == SaeAtomBasisKind::ProjectivePlane && case.latent_dim != 3 {
            for term in [&mut cover, &mut twin] {
                let refusal = term
                    .assemble_arrow_schur(target.view(), &rho, None)
                    .err()
                    .expect("dense assembly must refuse a charted RP²");
                assert!(
                    refusal.contains("requires latent dimension 3"),
                    "{}: the dense assembly refusal names another cause: {refusal}",
                    case.name
                );
            }
            continue;
        }
        let cover_system = cover
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("cover assembly");
        let twin_system = twin
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("twin assembly");
        assert_systems_transform("dense", &case, &cover_system, &twin_system);
    }
}

/// Support-sparse route: the same invariance through its own prior cache.
#[test]
fn support_objective_is_deck_invariant_on_every_quotient_2933() {
    for case in quotient_cases() {
        let n = case.coords.nrows();
        let target = target(n);
        let lambda_smooth = [0.8];
        let ard = vec![case.log_ard.iter().map(|log_alpha| log_alpha.exp()).collect::<Vec<_>>()];
        let cover = support_term(&case, &case.coords);
        let twin = support_term(&case, &case.twin());

        let cover_value = cover
            .penalized_objective(target.view(), &lambda_smooth, &ard)
            .expect("cover objective");
        let twin_value = twin
            .penalized_objective(target.view(), &lambda_smooth, &ard)
            .expect("twin objective");
        assert_close(&format!("support {} objective", case.name), twin_value, cover_value);

        let other_value = support_term(&case, &case.other_points)
            .penalized_objective(target.view(), &lambda_smooth, &ard)
            .expect("other-point objective");
        assert!(
            (other_value - cover_value).abs() > 1.0e-2,
            "support {}: the objective must distinguish a non-equivalent point (cover {cover_value}, other {other_value})",
            case.name
        );

        let cover_system = cover
            .assemble_arrow_schur(target.view(), &lambda_smooth, &ard)
            .expect("cover support assembly");
        let twin_system = twin
            .assemble_arrow_schur(target.view(), &lambda_smooth, &ard)
            .expect("twin support assembly");
        assert_systems_transform("support", &case, &cover_system, &twin_system);
    }
}

/// Trapezoid mass of `exp(-ard_value)` over the `T²` cover grid
/// `theta ∈ [0, theta_span)`, `phi ∈ [0, 1)`. Returns the mass and the range of
/// the negative log density it saw.
fn torus_cover_prior_mass(case: &QuotientCase, rho: &SaeManifoldRho, theta_span: f64) -> (f64, f64) {
    let nodes = 64usize;
    let manifold = case.kind.latent_manifold(case.latent_dim);
    let mut term = dense_term(case, &array![[0.0, 0.0]]);
    let mut mass = 0.0_f64;
    let mut min_neg_log = f64::INFINITY;
    let mut max_neg_log = f64::NEG_INFINITY;
    for i in 0..nodes {
        for j in 0..nodes {
            let point = array![[
                theta_span * i as f64 / nodes as f64,
                j as f64 / nodes as f64
            ]];
            term.assignment.coords[0] = LatentCoordValues::from_matrix_with_manifold(
                point.view(),
                LatentIdMode::None,
                manifold.clone(),
            );
            let neg_log_density = term.ard_value(rho).expect("ard value");
            min_neg_log = min_neg_log.min(neg_log_density);
            max_neg_log = max_neg_log.max(neg_log_density);
            mass += (-neg_log_density).exp();
        }
    }
    mass *= (theta_span / nodes as f64) * (1.0 / nodes as f64);
    (mass, max_neg_log - min_neg_log)
}

/// Normalization over the quotient. `exp(-ard_value)` on one row is the prior
/// density at that row up to the per-tangent-dimension constant the normalizer
/// convention pairs with the Laplace term (#2933 F26). The Klein bottle's
/// fundamental domain is `theta ∈ [0, 1/2)`, `phi ∈ [0, 1)`. A torus atom with the
/// same precisions uses the same circle normalizer family over the whole of `T²`,
/// so the Klein prior's mass over its fundamental domain must equal the torus
/// prior's mass over `T²`, and the convention constant cancels. A prior normalized
/// over the cover puts half its mass on the other sheet and carries exactly half
/// here. Both integrands are periodic over their grids, so the trapezoid rule is
/// spectrally accurate.
#[test]
fn klein_bottle_ard_prior_carries_the_cover_mass_over_one_fundamental_domain_2933() {
    let klein = quotient_cases()
        .into_iter()
        .find(|case| case.name == "klein_bottle")
        .expect("klein case");
    let torus = QuotientCase {
        name: "torus",
        kind: SaeAtomBasisKind::Torus,
        latent_dim: 2,
        evaluator: Arc::new(TorusHarmonicEvaluator::new(2, 2).expect("torus basis")),
        coords: klein.coords.clone(),
        deck: Vec::new(),
        other_points: klein.other_points.clone(),
        log_ard: klein.log_ard.clone(),
    };
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![60.0_f64.ln(), 30.0_f64.ln()]]);
    let (torus_mass, torus_range) = torus_cover_prior_mass(&torus, &rho, 1.0);
    let (klein_mass, _) = torus_cover_prior_mass(&klein, &rho, 0.5);
    assert!(
        torus_range > 0.5,
        "the prior must be materially non-uniform over T², else normalization is untested: \
         range {torus_range}"
    );
    assert!(
        (klein_mass - torus_mass).abs() <= 1.0e-10 * torus_mass,
        "Klein-bottle prior mass over one fundamental domain is {klein_mass:.15}, but the \
         torus prior's mass over T² is {torus_mass:.15}"
    );
}

/// Product-rule mass of `exp(-ard_value)` over ambient `S²` points in surface
/// measure `dS = dz·dφ`: midpoint rule in `z` on `2·z_cells` cells of `[-1, 1]`,
/// trapezoid in `φ`. `upper_half_only` keeps exactly the cells with `z > 0`.
/// Returns the mass and the range of the negative log density it saw.
fn ambient_sphere_prior_mass(
    case: &QuotientCase,
    rho: &SaeManifoldRho,
    upper_half_only: bool,
) -> (f64, f64) {
    let z_cells = 48usize;
    let phi_nodes = 32usize;
    let manifold = case.kind.latent_manifold(case.latent_dim);
    let mut term = dense_term(case, &array![[0.0, 0.0, 1.0]]);
    let first_cell = if upper_half_only { z_cells } else { 0 };
    let mut mass = 0.0_f64;
    let mut min_neg_log = f64::INFINITY;
    let mut max_neg_log = f64::NEG_INFINITY;
    for cell in first_cell..2 * z_cells {
        let z = -1.0 + (cell as f64 + 0.5) / z_cells as f64;
        let radius = (1.0 - z * z).sqrt();
        for node in 0..phi_nodes {
            let phi = std::f64::consts::TAU * node as f64 / phi_nodes as f64;
            let point = array![[radius * phi.cos(), radius * phi.sin(), z]];
            term.assignment.coords[0] = LatentCoordValues::from_matrix_with_manifold(
                point.view(),
                LatentIdMode::None,
                manifold.clone(),
            );
            let neg_log_density = term.ard_value(rho).expect("ard value");
            min_neg_log = min_neg_log.min(neg_log_density);
            max_neg_log = max_neg_log.max(neg_log_density);
            mass += (-neg_log_density).exp();
        }
    }
    mass *= (1.0 / z_cells as f64) * (std::f64::consts::TAU / phi_nodes as f64);
    (mass, max_neg_log - min_neg_log)
}

/// Normalization over the quotient for a reflection-only deck group. `RP²` on its
/// ambient cover has the antipodal deck `u ~ -u` and no half-turned axis, so its
/// prior keeps the sphere's energy and partition family, and one fundamental domain
/// is a hemisphere. The quotient prior must carry over that hemisphere the same mass
/// the `S²` prior carries over the whole sphere; a prior normalized over the cover
/// carries only half of it. The hemisphere's nodes are exactly the upper half of the
/// sphere's and the energy is even in `z`, so the comparison is free of quadrature
/// error and of whatever per-factor constant convention the sphere partition uses.
#[test]
fn projective_plane_ambient_prior_carries_the_quotient_sheet_count_2933() {
    let rp2 = quotient_cases()
        .into_iter()
        .find(|case| case.name == "projective_plane_ambient")
        .expect("ambient rp2 case");
    let sphere = QuotientCase {
        name: "sphere",
        kind: SaeAtomBasisKind::Sphere,
        latent_dim: 3,
        evaluator: Arc::new(AmbientSphereHarmonicEvaluator::new(2).expect("sphere basis")),
        coords: rp2.coords.clone(),
        deck: Vec::new(),
        other_points: rp2.other_points.clone(),
        log_ard: rp2.log_ard.clone(),
    };
    let rho = SaeManifoldRho::new(
        0.0,
        0.0,
        vec![array![2.6_f64.ln(), 0.9_f64.ln(), 4.3_f64.ln()]],
    );
    let (sphere_mass, sphere_range) = ambient_sphere_prior_mass(&sphere, &rho, false);
    let (quotient_mass, _) = ambient_sphere_prior_mass(&rp2, &rho, true);
    assert!(
        sphere_range > 0.5,
        "the anisotropic prior must be materially non-uniform on S², else the sheet count \
         is untested: range {sphere_range}"
    );
    assert!(
        (quotient_mass - sphere_mass).abs() <= 1.0e-12 * sphere_mass,
        "RP² prior mass over a hemisphere is {quotient_mass:.15}, but the S² prior's mass \
         over the sphere is {sphere_mass:.15}"
    );
}
