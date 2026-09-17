#![cfg(test)]
//! #2933 F27: a hard-TopK dense fit prices the ARD coordinate prior on the slots
//! its row blocks hold.
//!
//! A TopK row block holds only the selected atoms' coordinates
//! (`SaeRowLayout::from_topk_gates`). So `½·log|A|` integrates a coordinate only on
//! the rows that select its atom, and the inner solve never moves the others.
//! `ard_value`, its log-precision derivative and the MacKay statistic summed the
//! energy and the log partition over every row of every atom. An unselected
//! coordinate kept whatever value it last held and still paid `E(t) + log Z(α)`, with
//! no `½·log|H_tt|` to pair with. The support-sparse route has no such coordinate,
//! so the two routes priced different priors at one state.
//!
//! The references are independent of the production kernels: a trapezoid rule on
//! the period-one von Mises partition (spectrally exact for a smooth periodic
//! integrand) and the Gaussian integral on the line.

use super::*;
use ndarray::array;
use std::f64::consts::TAU;

const N: usize = 6;
const CIRCLE_ROWS: usize = 3;
const CIRCLE_ACTIVE: [f64; 3] = [0.08, -0.21, 0.33];
const PLANE_ACTIVE: [[f64; 2]; 3] = [[0.4, -0.7], [-0.25, 0.15], [0.9, 0.05]];
const LOG_ALPHA_CIRCLE: f64 = 0.5306282510621704; // ln 1.7
const LOG_ALPHA_PLANE: [f64; 2] = [1.0986122886681098, 0.4054651081081644]; // ln 3, ln 1.5
const NODES: usize = 512;

/// Two atoms under top-1 support: a period-one circle selected on rows 0..3 and
/// a two-axis plane selected on rows 3..6. `inactive` fills every unselected
/// coordinate, which the model does not hold.
fn topk_term(inactive: f64) -> SaeManifoldTerm {
    let mut circle = Array2::<f64>::from_elem((N, 1), inactive);
    let mut plane = Array2::<f64>::from_elem((N, 2), -1.5 * inactive);
    let mut logits = Array2::<f64>::zeros((N, 2));
    for row in 0..N {
        if row < CIRCLE_ROWS {
            circle[[row, 0]] = CIRCLE_ACTIVE[row];
            logits[[row, 0]] = 1.0;
        } else {
            plane[[row, 0]] = PLANE_ACTIVE[row - CIRCLE_ROWS][0];
            plane[[row, 1]] = PLANE_ACTIVE[row - CIRCLE_ROWS][1];
            logits[[row, 1]] = 1.0;
        }
    }
    let atom = |name: &'static str, d: usize| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::EuclideanPatch,
            d,
            Array2::<f64>::ones((N, 2)),
            Array3::<f64>::zeros((N, 2, d)),
            Array2::<f64>::zeros((2, 3)),
            Array2::<f64>::eye(2),
        )
        .expect("atom fixture: basis, jet, decoder and Gram shapes agree by construction")
    };
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![circle, plane],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Product(vec![LatentManifold::Euclidean; 2]),
        ],
        AssignmentMode::top_k_support(1),
    )
    .expect("assignment fixture: one coordinate block per manifold");
    SaeManifoldTerm::new(vec![atom("circle", 1), atom("plane", 2)], assignment)
        .expect("term fixture: two atoms, two blocks")
}

fn rho(log_alpha_circle: f64, log_alpha_plane: [f64; 2]) -> SaeManifoldRho {
    SaeManifoldRho::new(
        0.0,
        0.0,
        vec![array![log_alpha_circle], array![log_alpha_plane[0], log_alpha_plane[1]]],
    )
    .for_assignment(AssignmentMode::top_k_support(1))
}

/// `(1 − cos κt)/κ²` on the period-one circle: the von Mises energy per unit precision.
fn circle_shape(t: f64) -> f64 {
    (1.0 - (TAU * t).cos()) / (TAU * TAU)
}

/// Trapezoid `log ∫₀¹ exp[−α·shape(t)] dt − ½·log 2π` and its `∂/∂log α`,
/// `−α·E[shape]` under the normalized prior.
fn circle_log_partition(alpha: f64) -> (f64, f64) {
    let mut mass = 0.0_f64;
    let mut moment = 0.0_f64;
    for node in 0..NODES {
        let shape = circle_shape(node as f64 / NODES as f64);
        let weight = (-alpha * shape).exp();
        mass += weight;
        moment += weight * shape;
    }
    (
        (mass / NODES as f64).ln() - 0.5 * TAU.ln(),
        -alpha * moment / mass,
    )
}

/// The prior over the slots the model holds, per atom: circle energy and log
/// partition, then the plane's two line axes.
struct SlotReference {
    value: f64,
    log_precision_gradient: [f64; 3],
    sumsq: [f64; 3],
}

fn slot_reference(alpha_circle: f64, alpha_plane: [f64; 2]) -> SlotReference {
    let slots = CIRCLE_ROWS as f64;
    let (log_z_circle, log_z_circle_gradient) = circle_log_partition(alpha_circle);
    let circle_energy = CIRCLE_ACTIVE
        .iter()
        .map(|&t| alpha_circle * circle_shape(t))
        .sum::<f64>();
    let plane_sq = [0, 1].map(|axis| PLANE_ACTIVE.iter().map(|row| row[axis] * row[axis]).sum::<f64>());
    let plane_energy = [0, 1].map(|axis| 0.5 * alpha_plane[axis] * plane_sq[axis]);
    SlotReference {
        value: circle_energy
            + slots * log_z_circle
            + plane_energy[0]
            + plane_energy[1]
            - 0.5 * slots * (alpha_plane[0].ln() + alpha_plane[1].ln()),
        log_precision_gradient: [
            circle_energy + slots * log_z_circle_gradient,
            plane_energy[0] - 0.5 * slots,
            plane_energy[1] - 0.5 * slots,
        ],
        sumsq: [
            2.0 * CIRCLE_ACTIVE.iter().map(|&t| circle_shape(t)).sum::<f64>(),
            plane_sq[0],
            plane_sq[1],
        ],
    }
}

#[test]
fn topk_ard_prices_only_the_coordinates_the_row_blocks_hold_2933_f27() {
    let alpha_circle = LOG_ALPHA_CIRCLE.exp();
    let alpha_plane = LOG_ALPHA_PLANE.map(f64::exp);
    let reference = slot_reference(alpha_circle, alpha_plane);
    let inactive = 0.37;
    let term = topk_term(inactive);
    let base = rho(LOG_ALPHA_CIRCLE, LOG_ALPHA_PLANE);

    // The fixture discriminates: pricing every row adds the unselected coordinates'
    // energy and three more normalizers per atom.
    let (log_z_circle, _) = circle_log_partition(alpha_circle);
    let plane_inactive = -1.5 * inactive;
    let every_row = reference.value
        + CIRCLE_ROWS as f64 * (alpha_circle * circle_shape(inactive) + log_z_circle)
        + CIRCLE_ROWS as f64
            * (0.5 * (alpha_plane[0] + alpha_plane[1]) * plane_inactive * plane_inactive
                - 0.5 * (alpha_plane[0].ln() + alpha_plane[1].ln()));
    assert!(
        (every_row - reference.value).abs() > 0.1,
        "fixture must separate the slot prior {:.6e} from the every-row prior {every_row:.6e}",
        reference.value
    );

    let value = term.ard_value(&base).expect("ARD value");
    eprintln!(
        "[#2933 F27 dense TopK ARD] value={value:.12e} slot_reference={:.12e} \
         every_row_reference={every_row:.12e}",
        reference.value
    );
    assert!(
        (value - reference.value).abs() <= 1.0e-10 * (1.0 + reference.value.abs()),
        "TopK ARD value {value:.12e} != the prior on held slots {:.12e} (every row: {every_row:.12e})",
        reference.value
    );

    // A coordinate the model does not hold moves nothing.
    let moved = topk_term(-0.52).ard_value(&base).expect("ARD value, moved inactive coordinates");
    assert!(
        (moved - value).abs() <= 1.0e-12 * (1.0 + value.abs()),
        "unselected coordinates changed the ARD value: {value:.12e} -> {moved:.12e}"
    );

    // The explicit log-precision derivative, against the independent reference and
    // against a central difference of the production value.
    let derivatives = term
        .ard_log_precision_explicit_derivatives(&base)
        .expect("ARD log-precision derivatives");
    let analytic = [derivatives[0][0], derivatives[1][0], derivatives[1][1]];
    let step = 1.0e-5;
    let perturbed = |axis: usize, delta: f64| {
        let mut circle = LOG_ALPHA_CIRCLE;
        let mut plane = LOG_ALPHA_PLANE;
        match axis {
            0 => circle += delta,
            _ => plane[axis - 1] += delta,
        }
        term.ard_value(&rho(circle, plane)).expect("perturbed ARD value")
    };
    for axis in 0..3 {
        let expected = reference.log_precision_gradient[axis];
        assert!(
            (analytic[axis] - expected).abs() <= 1.0e-9 * (1.0 + expected.abs()),
            "axis {axis}: ∂ARD/∂log α {:.12e} != slot reference {expected:.12e}",
            analytic[axis]
        );
        let central = (perturbed(axis, step) - perturbed(axis, -step)) / (2.0 * step);
        assert!(
            (analytic[axis] - central).abs() <= 1.0e-6 * (1.0 + central.abs()),
            "axis {axis}: ∂ARD/∂log α {:.12e} != central difference {central:.12e}",
            analytic[axis]
        );
    }

    // The MacKay statistic sums the same slots `tr H⁻¹` does.
    let sumsq = term.ard_coord_sumsq().expect("ARD sufficient statistic");
    let statistic = [sumsq[0][0], sumsq[1][0], sumsq[1][1]];
    for axis in 0..3 {
        assert!(
            (statistic[axis] - reference.sumsq[axis]).abs()
                <= 1.0e-12 * (1.0 + reference.sumsq[axis]),
            "axis {axis}: MacKay statistic {:.12e} != held-slot sum {:.12e}",
            statistic[axis],
            reference.sumsq[axis]
        );
    }
}
