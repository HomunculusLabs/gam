//! #2545 — the KKT floor a saturating barrier leaves at an upper rail, and
//! the exact removal that clears it.
//!
//! These assert on the two functions the certificate composes at its
//! projection site: [`gradient_with_rail_barrier_removed`] then
//! [`project_gradient_vector`]. The numbers are the measured #2450/#2545
//! fixture: at ρ=30 the criterion's ρ-gradient was `1.332521e-7`, of which
//! the barrier was `1.332439e-7` (99.999%) and the λ=∞ face tail underneath
//! it `8.185450e-12 = 87.51·e^{−30}`.
use super::{Array1, gradient_with_rail_barrier_removed, project_gradient_vector};

/// `w·a·tanh(a·ρ)` at the measured saturation, `w = 1e-6`, `s = 4`, `b = 30`.
const BARRIER_AT_RHO_30: f64 = 1.332_439e-7;
/// The measured criterion ρ-gradient at ρ=30 on the #2450 fixture.
const CRITERION_AT_RHO_30: f64 = 1.332_521e-7;
/// `87.51·e^{−30}`: the λ=∞ face tail, the whole residual that should
/// survive the removal.
const FACE_TAIL_AT_RHO_30: f64 = 8.185_45e-12;

fn box_30(dim: usize) -> (Array1<f64>, Array1<f64>) {
    (
        Array1::from_elem(dim, -30.0),
        Array1::from_elem(dim, 30.0),
    )
}

/// The defect, stated as an assertion: with the barrier still in the
/// certificate's view, an upper rail carries `|Pg| ≈ w·a` no matter how
/// completely the criterion's own tail has decayed — because the projection
/// KEEPS the positive part and the barrier's contribution is positive.
#[test]
fn an_upper_rail_carries_the_barrier_as_a_standing_kkt_residual() {
    let rho = Array1::from_vec(vec![30.0]);
    let bounds = box_30(1);
    // Sign convention at an upper rail: the face tail pulls INWARD
    // (∂V/∂ρ < 0, which the projection discards) while the barrier pulls
    // outward-positive, so the total is dominated by the barrier.
    let gradient = Array1::from_vec(vec![BARRIER_AT_RHO_30 - FACE_TAIL_AT_RHO_30]);
    let projected = project_gradient_vector(&rho, &gradient, Some(&bounds));
    assert!(
        projected[0] > 1.0e-7,
        "the pre-#2545 residual at an upper rail must be the saturated barrier \
         (~1.33e-7), got {:.6e}",
        projected[0]
    );
}

/// The fix: remove the published barrier on the railed coordinate and the
/// projection returns exactly zero, because what is left points INWARD and
/// `gi.max(0.0)` discards a feasible-descent pull at an upper bound.
#[test]
fn removing_the_barrier_lets_a_clean_upper_rail_certify_at_the_face_tail() {
    let rho = Array1::from_vec(vec![30.0]);
    let bounds = box_30(1);
    let gradient = Array1::from_vec(vec![BARRIER_AT_RHO_30 - FACE_TAIL_AT_RHO_30]);
    let guard = Array1::from_vec(vec![BARRIER_AT_RHO_30]);
    let view =
        gradient_with_rail_barrier_removed(&rho, &gradient, &bounds, Some(&guard));
    assert!(
        (view[0] + FACE_TAIL_AT_RHO_30).abs() <= 1.0e-18,
        "the removal must be EXACT — what is left is the face tail alone, \
         {:.6e} expected {:.6e}",
        view[0],
        -FACE_TAIL_AT_RHO_30
    );
    let projected = project_gradient_vector(&rho, &view, Some(&bounds));
    assert_eq!(
        projected[0], 0.0,
        "a clean lambda=infinity face must project to exactly 0 once the \
         barrier is out of the certificate's view (#2545)"
    );
    // And the residual the certificate now judges is the face tail's own
    // magnitude — the acceptance number on the issue, five orders below the
    // 1.332521e-7 the barrier was pinning it at.
    assert!(
        FACE_TAIL_AT_RHO_30 < CRITERION_AT_RHO_30 * 1.0e-4,
        "the face tail must be four or more orders below the barrier-bearing \
         residual, else this fixture is not the measured one"
    );
}

/// The line the removal must not cross: an INTERIOR coordinate keeps the
/// barrier, because there the optimizer descends criterion-plus-barrier and
/// halts where their SUM vanishes. Subtracting it would make the certificate
/// judge a function nothing optimized.
#[test]
fn an_interior_coordinate_keeps_the_barrier_in_the_certificates_view() {
    let rho = Array1::from_vec(vec![30.0, 2.5]);
    let bounds = box_30(2);
    let gradient = Array1::from_vec(vec![BARRIER_AT_RHO_30, 3.0e-3]);
    let guard = Array1::from_vec(vec![BARRIER_AT_RHO_30, 4.4e-8]);
    let view =
        gradient_with_rail_barrier_removed(&rho, &gradient, &bounds, Some(&guard));
    assert_eq!(
        view[0], 0.0,
        "the railed coordinate 0 must have its barrier removed"
    );
    assert_eq!(
        view[1], gradient[1],
        "the interior coordinate 1 must keep the criterion gradient the \
         optimizer actually descended, barrier included"
    );
}

/// #2629 — on an objective whose outer coordinate is
/// `θ = [ρ (rho_dim), ψ/link (psi_dim)]`, the publication is θ-length with
/// the barrier in the leading ρ block and EXACT zeros in the trailing one.
///
/// This is the gate the issue asks for, and it is on the seam rather than
/// on a call site because that is where the layout is now applied. The
/// mixture/SAS objective installs the same closure standard REML does — a
/// closure that speaks ρ — and `ClosureObjective` embeds it from the
/// declared `psi_dim`. The failure this refuses is specifically invisible:
/// every coordinate's barrier is the same order of magnitude, so a
/// publication shifted by one slot has the same norm as a correct one and
/// shows up only as a coordinate that never certifies.
#[test]
fn a_psi_bearing_objective_publishes_the_barrier_in_the_rho_block_and_zeros_elsewhere_2629() {
    use super::{Derivative, HessianValue, OuterEval, OuterProblem};
    // 2 ρ coordinates + 2 link coordinates, the shape a one-smooth SAS fit
    // presents (`θ_dim = k + sas_dim`).
    let problem = OuterProblem::new(4)
        .with_gradient(Derivative::Analytic)
        .with_psi_dim(2);
    // The hook is handed the ρ BLOCK and answers over it. Distinct values so
    // a slot shift cannot be mistaken for a correct answer.
    let mut obj = problem
        .build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * theta.dot(theta),
                    gradient: theta.clone(),
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<super::EfsEval, super::EstimationError>>,
        )
        .with_soft_rho_guard_gradient(|_: &mut (), rho: &Array1<f64>| {
            assert_eq!(
                rho.len(),
                2,
                "the hook must receive the RHO block, not the full theta"
            );
            Array1::from_vec(vec![BARRIER_AT_RHO_30, 0.5 * BARRIER_AT_RHO_30])
        });

    let theta = Array1::from_vec(vec![30.0, 30.0, 0.25, -0.25]);
    let published = super::OuterObjective::soft_rho_guard_gradient(&mut obj, &theta)
        .expect("an objective that installed the hook must publish");
    assert_eq!(
        published.len(),
        theta.len(),
        "the publication is indexed by OUTER coordinate at every consumer, so it \
         must be theta-length"
    );
    assert_eq!(published[0], BARRIER_AT_RHO_30);
    assert_eq!(published[1], 0.5 * BARRIER_AT_RHO_30);
    // Exact zeros, not "small": the barrier acts on ρ only, and a consumer
    // subtracts this from a railed coordinate's gradient verbatim.
    assert_eq!(
        published[2], 0.0,
        "the link slots carry no barrier and must publish EXACTLY zero"
    );
    assert_eq!(published[3], 0.0);

    // ...and composed with the removal, a railed LINK coordinate keeps its
    // gradient bit for bit while the railed ρ coordinates lose theirs. This
    // is the property a hand-written θ publication would have broken.
    let bounds = box_30(4);
    let at_bounds = Array1::from_vec(vec![30.0, 30.0, 30.0, -30.0]);
    let gradient = Array1::from_vec(vec![
        BARRIER_AT_RHO_30,
        0.5 * BARRIER_AT_RHO_30,
        7.5e-3,
        -2.5e-3,
    ]);
    let view =
        gradient_with_rail_barrier_removed(&at_bounds, &gradient, &bounds, Some(&published));
    assert_eq!(view[0], 0.0, "railed rho coordinate 0 loses its barrier");
    assert_eq!(view[1], 0.0, "railed rho coordinate 1 loses its barrier");
    assert_eq!(
        view[2], gradient[2],
        "a railed LINK coordinate carries no barrier and must be untouched — \
         its box is a real constraint, not a proxy for a limit"
    );
    assert_eq!(view[3], gradient[3]);
}

/// A missing or mis-shaped publication is an ABSENCE, never a partial
/// subtraction: the pre-#2545 behavior, not a coordinate-shifted one.
#[test]
fn an_unpublished_or_misshaped_barrier_leaves_the_gradient_untouched() {
    let rho = Array1::from_vec(vec![30.0, 30.0]);
    let bounds = box_30(2);
    let gradient = Array1::from_vec(vec![BARRIER_AT_RHO_30, BARRIER_AT_RHO_30]);
    assert_eq!(
        gradient_with_rail_barrier_removed(&rho, &gradient, &bounds, None),
        gradient,
        "an objective with no barrier must see the pre-#2545 gradient exactly"
    );
    let short = Array1::from_vec(vec![BARRIER_AT_RHO_30]);
    assert_eq!(
        gradient_with_rail_barrier_removed(&rho, &gradient, &bounds, Some(&short)),
        gradient,
        "a length mismatch must degrade to no removal rather than subtract \
         one coordinate's barrier from another's gradient"
    );
}
