//! #2730: the shared-dispersion REML penalty gradient must be built from the
//! range of `S` that its own cache classified, not from every eigenvalue that
//! happens to be `> 0.0`.
//!
//! `GaussianRemlEigenCache::penalty_rank` is defined by a relative,
//! scale-invariant threshold.  The pseudoinverse feeding
//! `gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit` used to ask
//! the same question with an absolute `delta > 0.0`, and the eigensolver returns
//! numerically null directions as small POSITIVE numbers — the cache's cleanup
//! loop only zeroes negative ones.  The reciprocal of such a value then lands in
//! the returned gradient.
//!
//! Measured on this fixture before the repair: `penalty_rank = 6` while seven
//! eigenvalues pass `delta > 0.0`, the seventh being `3.20001575162645240e-18`;
//! removing the range pseudoinverse from the returned gradient leaves
//! `1.6182871562151468e15` against a range contribution of `1.4682801591665569e0`.
//!
//! The chart here is deliberately NOT near-interpolating — its pooled deviance
//! sits about ten orders of magnitude above its own roundoff — so neither test
//! can pass or fail for a reason belonging to the near-interpolating regime.
//!
//! The two tests are separate `#[test]` functions on purpose: as one function
//! the first assertion aborts before the second property is ever exercised, and
//! an arm that never runs is not an arm.

use gam_solve::gaussian_reml::{
    GaussianRemlMultiResult, gaussian_reml_multi_shared_dispersion_closed_form,
    gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit,
};
use ndarray::{Array1, Array2};

const ROWS: usize = 46;
const COEFFICIENTS: usize = 8;

fn design() -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((ROWS, COEFFICIENTS));
    for row in 0..ROWS {
        let t = (row as f64) / ((ROWS - 1) as f64);
        for col in 0..COEFFICIENTS {
            x[[row, col]] = (std::f64::consts::PI * (col as f64) * t).cos();
        }
    }
    x
}

/// Second-difference penalty: rank 6, nullity 2 on eight coefficients.
fn penalty() -> Array2<f64> {
    let bands = COEFFICIENTS - 2;
    let mut difference = Array2::<f64>::zeros((bands, COEFFICIENTS));
    for band in 0..bands {
        difference[[band, band]] = 1.0;
        difference[[band, band + 1]] = -2.0;
        difference[[band, band + 2]] = 1.0;
    }
    let mut s = Array2::<f64>::zeros((COEFFICIENTS, COEFFICIENTS));
    for row in 0..COEFFICIENTS {
        for col in 0..COEFFICIENTS {
            let mut acc = 0.0;
            for band in 0..bands {
                acc += difference[[band, row]] * difference[[band, col]];
            }
            s[[row, col]] = acc;
        }
    }
    s
}

/// A comfortably resolved response, so nothing below is decided by the
/// near-interpolating regime.
fn response(x: &Array2<f64>) -> Array2<f64> {
    let mut beta = Array1::<f64>::zeros(COEFFICIENTS);
    for index in 0..COEFFICIENTS {
        beta[index] = 0.4 * (-(index as f64) * 0.35).exp() * if index % 2 == 0 { 1.0 } else { -1.0 };
    }
    let mut y = Array2::<f64>::zeros((ROWS, 1));
    for row in 0..ROWS {
        let mut mean = 0.0;
        for col in 0..COEFFICIENTS {
            mean += x[[row, col]] * beta[col];
        }
        y[[row, 0]] = mean + 1.0e-3 * ((row as f64) * 2.399_963_229_728_653).sin();
    }
    y
}

fn max_abs(matrix: &Array2<f64>) -> f64 {
    matrix.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

struct Fixture {
    x: Array2<f64>,
    s: Array2<f64>,
    y: Array2<f64>,
    fit: GaussianRemlMultiResult,
    /// Indices of the `penalty_rank` largest eigenvalues: the range the cache
    /// classified, in the cache's own storage order.
    retained: Vec<usize>,
    gradient: Array2<f64>,
}

/// The fixture together with the preconditions BOTH tests depend on.  A green in
/// either test is mute unless the spectrum actually carries a positive-but-null
/// direction, so those checks live here and fail loudly rather than being
/// assumed.
fn fixture() -> Fixture {
    let x = design();
    let s = penalty();
    let y = response(&x);
    let fit =
        gaussian_reml_multi_shared_dispersion_closed_form(x.view(), y.view(), s.view(), None, None)
            .expect("healthy shared-dispersion forward fit");

    let rank = fit.cache.penalty_rank;
    let positive = fit
        .cache
        .penalty_eigenvalues
        .iter()
        .filter(|delta| **delta > 0.0)
        .count();
    assert_eq!(rank, COEFFICIENTS - 2, "fixture must have rank 6");
    assert_eq!(fit.cache.nullity, 2, "fixture must have nullity 2");
    assert!(
        positive > rank,
        "precondition unmet: the spectrum carries no positive-but-null direction \
         (rank={rank}, eigenvalues passing `> 0.0`={positive}); this fixture cannot \
         exercise the branch under test and a pass here would be mute"
    );

    let shared_nu = (ROWS - fit.cache.nullity) as f64;
    let pooled_deviance = fit.sigma2[0] * shared_nu;
    let response_energy = y.iter().map(|value| value * value).sum::<f64>();
    assert!(
        pooled_deviance > 1.0e6 * f64::EPSILON * response_energy,
        "precondition unmet: this fixture is supposed to be a HEALTHY chart, but \
         pooled_deviance={pooled_deviance:e} is close to the roundoff of the \
         response energy {response_energy:e}"
    );

    let mut retained: Vec<usize> = (0..COEFFICIENTS).collect();
    retained.sort_by(|a, b| {
        fit.cache.penalty_eigenvalues[*b]
            .partial_cmp(&fit.cache.penalty_eigenvalues[*a])
            .expect("finite eigenvalues")
    });
    retained.truncate(rank);

    let gradient = gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit(
        x.view(),
        y.view(),
        s.view(),
        None,
        &fit,
    )
    .expect("penalty gradient on a healthy fit");

    Fixture {
        x,
        s,
        y,
        fit,
        retained,
        gradient,
    }
}

/// The returned gradient is `0.5·d·(λ·H⁻¹ − S⁺) + deviance_scale·Σ β β'`.
/// Rebuild the pseudoinverse over the range the cache classified, remove it, and
/// what remains is the other two terms — which cannot legitimately exceed the
/// range pseudoinverse's own contribution.  The bound is therefore read off the
/// fit, not chosen.  A null direction entering as `1/roundoff` breaks it by
/// fifteen orders of magnitude.
#[test]
fn penalty_gradient_carries_no_term_the_classified_range_cannot_explain() {
    let case = fixture();
    let rank = case.fit.cache.penalty_rank;

    let mut range_pseudoinverse = Array2::<f64>::zeros((COEFFICIENTS, COEFFICIENTS));
    for row in 0..COEFFICIENTS {
        for col in 0..COEFFICIENTS {
            let mut acc = 0.0;
            for eig in case.retained.iter().copied() {
                acc += case.fit.cache.coefficient_basis[[row, eig]]
                    * case.fit.cache.coefficient_basis[[col, eig]]
                    / case.fit.cache.penalty_eigenvalues[eig];
            }
            range_pseudoinverse[[row, col]] = acc;
        }
    }
    let mut residue = case.gradient.clone();
    for row in 0..COEFFICIENTS {
        for col in 0..COEFFICIENTS {
            residue[[row, col]] += 0.5 * range_pseudoinverse[[row, col]];
        }
    }
    let bound = 0.5 * max_abs(&range_pseudoinverse);
    assert!(
        max_abs(&residue) <= bound,
        "the penalty gradient carries a term the classified range cannot explain: \
         removing 0.5·S⁺ over the {rank} range directions leaves {:e}, which exceeds \
         the range pseudoinverse's own contribution {bound:e}",
        max_abs(&residue)
    );
}

/// A direction the cache classified as NULL must not influence the gradient at
/// all.  Rescale that eigenvalue — it stays null under the relative threshold
/// either way — and the returned gradient must be bit-identical.  `λ·δ`
/// underflows to zero in `1/(1 + λ·δ)` for both values, so the inverse Hessian is
/// untouched and this isolates the pseudoinverse.  Under an absolute `δ > 0.0`
/// test the gradient moves by the rescaling factor.
///
/// This fails for a reason independent of the magnitude bound above: it would
/// still fire if the spurious term were small enough to sit under that bound.
#[test]
fn penalty_gradient_does_not_read_an_eigenvalue_the_cache_classified_as_null() {
    let case = fixture();
    let rank = case.fit.cache.penalty_rank;

    let null_index = (0..COEFFICIENTS)
        .find(|index| {
            !case.retained.contains(index) && case.fit.cache.penalty_eigenvalues[*index] > 0.0
        })
        .expect("the fixture precondition established a positive null direction");

    let mut perturbed = case.fit.clone();
    perturbed.cache.penalty_eigenvalues[null_index] *= 1.0e-3;
    let perturbed_gradient = gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit(
        case.x.view(),
        case.y.view(),
        case.s.view(),
        None,
        &perturbed,
    )
    .expect("penalty gradient after rescaling a null eigenvalue");

    for row in 0..COEFFICIENTS {
        for col in 0..COEFFICIENTS {
            assert_eq!(
                case.gradient[[row, col]].to_bits(),
                perturbed_gradient[[row, col]].to_bits(),
                "entry ({row},{col}) moved from {} to {} when eigenvalue {null_index} — \
                 classified null by penalty_rank={rank} — was scaled by 1e-3; the gradient \
                 is reading a direction the cache calls null",
                case.gradient[[row, col]],
                perturbed_gradient[[row, col]]
            );
        }
    }
}
