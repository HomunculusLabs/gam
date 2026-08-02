//! #2761, from the basis layer: the measure-jet representer range ℓ moves the
//! design's SPAN, so it is not a quantity a smoothing parameter can stand in
//! for.
//!
//! The end-to-end gate for #2761 is
//! `measure_jet_perf_parity::measure_jet_single_scale_mode_accuracy_parity`,
//! which compares a fitted measure-jet against a fitted Matérn on one Gaussian
//! fixture. That test can go green for the wrong reason — a lucky λ, a lucky
//! seed, a comparator that got worse — and it cannot say *why* ℓ has to be a
//! REML coordinate. This one asserts the mechanism directly and with no fit in
//! it at all:
//!
//!   **span floor(ℓ)** = the least-squares projection residual of the NOISELESS
//!   truth onto the realized design's column span.
//!
//! That is the bound *no* choice of λ can beat, because λ only shrinks a
//! coefficient vector inside the span; it never moves the span. If the floor
//! were flat in ℓ, freezing ℓ at a geometric heuristic would cost nothing and
//! `#1041`'s revert would have been harmless. It is not flat: on a 1-D curve
//! embedded in 3-D — the geometry measure-jet exists for — the floor at the
//! auto seed is more than two orders of magnitude above the floor a longer
//! range reaches, at identical centers, identical rank and identical `p`.
//!
//! The fixture here is deliberately NOT the perf-parity one: different
//! embedding, different target, deterministic low-discrepancy sampling instead
//! of an RNG draw. A regression that only reappears on one seed's center layout
//! should not be able to hide from both tests at once.

use gam_terms::basis::{
    CenterStrategy, MeasureJetBasisSpec, MeasureJetIdentifiability, build_measure_jet_basis,
    measure_jet_quadrature_nodes, realized_measure_jet_length_scale, select_centers_by_strategy,
};
use ndarray::{Array1, Array2};

const N: usize = 1_200;
const CENTERS: usize = 16;

/// A 1-D curve in 3-D whose ambient speed varies by ~3x along its length, so an
/// ambient-maximin center set is deliberately NOT uniform in the intrinsic
/// coordinate and the median nearest-center spacing is a poor summary of the
/// resolution the target needs. That mismatch is the geometry measure-jet
/// exists for and the one its auto seed is weakest on.
fn latent_to_coords(t: f64) -> [f64; 3] {
    [
        t,
        0.45 * (2.0 * std::f64::consts::PI * t).sin(),
        0.45 * t * t * t,
    ]
}

/// Three intrinsic cycles: smooth, but finer than one center spacing's worth of
/// ambient resolution, which is exactly where the range matters.
fn truth(t: f64) -> f64 {
    (6.0 * std::f64::consts::PI * t).sin() + 0.4 * (2.0 * std::f64::consts::PI * t).cos()
}

/// Golden-ratio additive recurrence: deterministic and equidistributed, so the
/// realized center layout is a property of the geometry rather than of a draw.
fn latents(n: usize) -> Vec<f64> {
    let phi = 0.618_033_988_749_894_9_f64;
    (0..n).map(|i| ((i as f64 + 0.5) * phi).fract()).collect()
}

/// Residual RMSE of the least-squares projection of `y` onto
/// `span(1, columns of x)`, by twice-reorthogonalized modified Gram–Schmidt.
/// The intercept is included because the model carries one.
fn span_floor(x: &Array2<f64>, y: &[f64]) -> (f64, usize) {
    let n = x.nrows();
    let mut basis: Vec<Array1<f64>> = Vec::new();
    let absorb = |mut v: Array1<f64>, basis: &mut Vec<Array1<f64>>| {
        let raw = v.dot(&v).sqrt();
        for _ in 0..2 {
            for q in basis.iter() {
                let c = q.dot(&v);
                v.scaled_add(-c, q);
            }
        }
        let nrm = v.dot(&v).sqrt();
        if nrm > 1.0e-10 * raw.max(1.0) {
            v.mapv_inplace(|z| z / nrm);
            basis.push(v);
        }
    };
    absorb(Array1::<f64>::ones(n), &mut basis);
    for j in 0..x.ncols() {
        absorb(x.column(j).to_owned(), &mut basis);
    }
    let yv = Array1::from_vec(y.to_vec());
    let mut resid = yv.clone();
    for q in basis.iter() {
        let c = q.dot(&yv);
        resid.scaled_add(-c, q);
    }
    ((resid.dot(&resid) / n as f64).sqrt(), basis.len())
}

fn spec_at(length_scale: f64) -> MeasureJetBasisSpec {
    MeasureJetBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: CENTERS,
        },
        length_scale,
        identifiability: MeasureJetIdentifiability::CenterSumToZero,
        ..MeasureJetBasisSpec::default()
    }
}

#[test]
fn representer_range_moves_the_span_so_lambda_cannot_stand_in_for_it_2761() {
    let ts = latents(N);
    let mut data = Array2::<f64>::zeros((N, 3));
    for (r, &t) in ts.iter().enumerate() {
        let c = latent_to_coords(t);
        for k in 0..3 {
            data[[r, k]] = c[k];
        }
    }
    let y: Vec<f64> = ts.iter().map(|&t| truth(t)).collect();
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let flat = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt();

    // The range the `0.0` auto sentinel realizes, through the SAME helpers the
    // basis builder calls, so this is the seed production uses and not a
    // re-derivation of it.
    let seeds = select_centers_by_strategy(
        data.view(),
        &CenterStrategy::FarthestPoint {
            num_centers: CENTERS,
        },
    )
    .expect("farthest-point seeds");
    let (nodes, _masses) =
        measure_jet_quadrature_nodes(data.view(), seeds.view()).expect("quadrature nodes");
    let seed_range = realized_measure_jet_length_scale(nodes.view(), 0.0).expect("auto range");
    assert!(
        seed_range.is_finite() && seed_range > 0.0,
        "the auto sentinel must realize a positive range"
    );

    let measure = |ell: f64| -> (f64, usize, usize) {
        let built = build_measure_jet_basis(data.view(), &spec_at(ell))
            .unwrap_or_else(|e| panic!("measure-jet build at ell={ell}: {e}"));
        let dense = built.design.to_dense();
        let (floor, rank) = span_floor(&dense, &y);
        (floor, rank, dense.ncols())
    };

    let (seed_floor, seed_rank, seed_p) = measure(seed_range);

    // A range REML can reach: the ψ box is ln ℓ ∈ [ln 1e-3, ln 1e2], so 8x the
    // seed is deep inside it and needs no extrapolation of the claim.
    let long_range = 8.0 * seed_range;
    let (long_floor, long_rank, long_p) = measure(long_range);

    println!(
        "[2761] seed_range={seed_range:.6} floor={seed_floor:.3e} (p={seed_p}, rank={seed_rank})  \
         long_range={long_range:.6} floor={long_floor:.3e} (p={long_p}, rank={long_rank})  \
         flat_fit_floor={flat:.6}"
    );

    // Same centers, same masses, same band, same penalty, same width: the ONLY
    // thing that differs between these two designs is the representer range.
    assert_eq!(
        seed_p, long_p,
        "the two ranges must produce the same basis width, or the comparison is \
         about dimension rather than about span alignment"
    );
    assert_eq!(
        seed_rank, long_rank,
        "the Gaussian kernel is strictly PD for every ell > 0, so neither range may \
         lose numerical rank; a rank drop would make this a conditioning test"
    );

    // Anti-vacuity: the seed range's shortfall has to be a bias that MATTERS,
    // not a rounding artifact that the ratio below could then inflate for free.
    // The bar is derived rather than chosen — a fit of this width at the noise
    // level the measure-jet Gaussian fixtures use carries sampling error
    // `sigma * sqrt(p/n)`, so requiring the seed floor to exceed twice that
    // says the approximation bias dominates the variance, which is exactly the
    // regime where a smoothing parameter cannot rescue it.
    const REFERENCE_SIGMA: f64 = 0.10;
    let sampling_noise = REFERENCE_SIGMA * (seed_p as f64 / N as f64).sqrt();
    assert!(
        seed_floor > 2.0 * sampling_noise,
        "precondition: the auto seed range must be the hard case this test is about \
         (seed_floor={seed_floor:.3e} against 2x the sampling error {:.3e} of a p={seed_p} \
         fit at sigma={REFERENCE_SIGMA}, flat-fit residual {flat:.6}); if the seed already \
         spans the truth, this fixture has stopped exercising #2761 and needs a finer target",
        2.0 * sampling_noise
    );

    // ...and a longer range does, by more than two orders of magnitude. This is
    // the whole argument: no lambda applied to the seed-range design can reach
    // what the long-range design reaches unpenalized.
    assert!(
        long_floor * 100.0 < seed_floor,
        "the representer range moves the design SPAN, so it cannot be frozen at a \
         geometric heuristic and left to the smoothing parameter: seed range \
         {seed_range:.6} floors at {seed_floor:.3e} while {long_range:.6} floors at \
         {long_floor:.3e} on identical centers at identical width. lambda shrinks \
         inside a span; it never moves one (#2761)"
    );
}
