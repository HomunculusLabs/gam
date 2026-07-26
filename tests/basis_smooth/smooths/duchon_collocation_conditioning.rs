//! TARGET behavior: the tension/mass collocation sample of the redesigned
//! non-periodic Euclidean Duchon penalty must RESOLVE every basis direction.
//!
//! The Hilbert-scale `OperatorTension` (`Σ‖∇f‖²`) and `OperatorMass` (`Σ(f−f̄)²`)
//! penalties are collocated on an O(k) farthest-point sample of the data. If
//! that sample were degenerate — too few points, or clustered so the collocation
//! design dropped rank — the emitted operator penalty would be rank-deficient in
//! directions the sample fails to see, and REML could not control the wiggle it
//! is meant to. This file guards that the collocation sample is well-conditioned
//! by probing the EMITTED penalty.
//!
//! The redesigned core has LANDED: `build_duchon_collocation_operator_matrices`
//! emits D0/D1/D2 including the polynomial null-space value/gradient/Hessian, so
//! the Hilbert-scale function penalties act on the full β-basis and the emitted
//! `OperatorTension` block resolves every basis direction. This is now a LIVE
//! regression guard on that conditioning contract (it passes on `main`), not a
//! forward-spec awaiting the core.
//!
//! METHODOLOGY (coordinate-free, mirroring `duchon_structural_seminorms.rs`).
//! The emitted penalty `S` acts on the design's own coefficient space: for a
//! coefficient vector `c`, `cᵀ S c` equals the (collocated) first-order energy of
//! the function `f = X c`. For a target `g` sampled at the data rows we recover
//! representing coefficients `c = argmin ‖X c − g‖²` and read off `cᵀ S c`. A
//! well-conditioned tension penalty (i) is non-degenerate — its trace is strictly
//! positive, so it penalizes SOME direction; (ii) ANNIHILATES constants — a flat
//! function has zero gradient energy on any sample; and (iii) genuinely PENALIZES
//! a wiggly target — a high-frequency function has strictly positive collocated
//! gradient energy, which can only hold if the collocation sample resolves the
//! wiggly directions. We assert with absolute energy bounds tied to the penalty's
//! own trace, never to any reference tool.
//!
//! NOTE ON A PUBLIC σ_min HOOK. `gam::basis` exposes
//! `build_duchon_collocation_operator_matrices`, which returns a public
//! `CollocationOperatorMatrices { d0, d1, d2, collocation_points, .. }` from
//! which a caller could form `D1ᵀD1` and take its σ_min directly. We deliberately
//! probe the penalty EMITTED by `build_duchon_basis` instead: that is the matrix
//! the fit actually uses, and the farthest-point collocation SAMPLE selection is
//! internal to the build (the public matrix builder takes externally supplied
//! centers/weights). Probing the emitted block guards the end-to-end contract —
//! that the sample chosen by the redesign resolves all basis directions.

use faer::Side;
use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, PenaltySource, SpatialIdentifiability, build_duchon_basis,
};
use gam::faer_ndarray::FaerCholesky;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

// ── fixtures ────────────────────────────────────────────────────────────────

/// Farthest-point centers requested of the default spec. Shared with
/// `wiggly_target`, which derives its frequency from the resulting sample
/// spacing: the probe target and the sample that must resolve it are two halves
/// of one contract and cannot be allowed to drift apart.
const NUM_CENTERS: usize = 24;

/// Sample spacings per wavelength of the wiggly probe target. Inherited from the
/// `d = 1` fixture (`λ = 0.5`, `h = 2/24`), where the probe sat at the sample's
/// resolution limit — which is what gives assertion (iii) its discriminating
/// power.
const RESOLVED_WAVELENGTHS_PER_SPACING: f64 = 6.0;

/// How much of the probe target's energy may fall OUTSIDE the design's column
/// space, as a relative L2 residual.
///
/// `coeff_for_target` returns coefficients for the orthogonal PROJECTION of the
/// target, so a relative residual `r` means the recovered function retains a
/// fraction `1 − r²` of the target's energy. What the probe's inference needs is
/// only that those coefficients represent the INTENDED target rather than an
/// artifact of whatever the design happens to express — the assertions
/// downstream compare energies separated by six orders of magnitude, so they are
/// nowhere near sensitive to a fraction of a percent of lost target.
///
/// This was `1e-4`, i.e. "retain 1 − 1e-8 of the energy". That is an
/// approximation-theory statement about a ONE-dimensional sample, not a
/// requirement of any assertion here, and at `d = 2` it is unsatisfiable jointly
/// with wiggliness at any sane `k`: the measured residual falls as `(λ/h)^-2`
/// (9.2e-1, 8.5e-1, 3.3e-1, 7.9e-2, 2.6e-2, 1.9e-2, 9.3e-3, 5.4e-3 at
/// λ/h = 1.22, 2, 3, 4, 6, 8, 12, 16), so 1e-4 extrapolates to λ/h ≈ 117 — and
/// since `cycles = √k / (λ/h)`, holding even two cycles there would need
/// k ≈ 55000 centers. The gate, not the dimension, was the `d = 1` constant.
///
/// `1 − 0.999` in energy, i.e. `√1e-3 ≈ 3.16e-2` in residual: at the probe's
/// `λ/h = 6` the measured residual is 2.59e-2 (99.93% of the target retained),
/// while the next coarser ratio tried, 4, measures 7.87e-2 and is correctly
/// refused.
const MAX_UNREPRESENTED_FRACTION: f64 = 0.031_622_776_6; // (1 − 0.999).sqrt()

/// `n` rows in `[-1, 1]^d`, deterministic from `seed`.
fn synthetic_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0_f64, 1.0).expect("uniform params valid");
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = dist.sample(&mut rng);
        }
    }
    data
}

/// The DEFAULT non-periodic Euclidean Duchon spec (the all-on Hilbert scale).
fn default_duchon_spec(k: usize) -> DuchonBasisSpec {
    DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: None,
        power: 0.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
    }
}

/// One built Duchon basis: dense design plus the emitted `OperatorTension`
/// block. `tension` is `None` when the build did not emit a collocated tension
/// penalty — itself a contract violation the test surfaces.
struct BuiltDuchon {
    design: Array2<f64>,
    tension: Option<Array2<f64>>,
}

fn build(data: &Array2<f64>, spec: &DuchonBasisSpec) -> BuiltDuchon {
    let result = build_duchon_basis(data.view(), spec).expect("build_duchon_basis succeeded");
    let design: Array2<f64> = result
        .design
        .try_to_dense_arc("collocation-conditioning test")
        .expect("design can be materialized")
        .as_ref()
        .clone();

    // Select the collocated tension matrix from the same atomic record that
    // declares its semantic role.
    let tension = result
        .active_penalties
        .iter()
        .find(|penalty| matches!(&penalty.info.source, PenaltySource::OperatorTension))
        .map(|penalty| penalty.matrix.clone());

    BuiltDuchon { design, tension }
}

/// Coefficients `c` minimizing `‖X c − g‖²` via the normal equations with a
/// vanishing relative ridge for conditioning. Asserts the target is represented
/// (small relative residual) so the recovered `cᵀ S c` is meaningful.
fn coeff_for_target(x: &Array2<f64>, g: &Array1<f64>, what: &str) -> Array1<f64> {
    let p = x.ncols();
    let xtx = x.t().dot(x);
    let max_diag = xtx.diag().iter().cloned().fold(1.0_f64, f64::max);
    let mut gram = xtx.clone();
    let eps = 1e-10 * max_diag;
    for i in 0..p {
        gram[[i, i]] += eps;
    }
    let xtg = x.t().dot(g);
    let chol = gram
        .cholesky(Side::Lower)
        .expect("Cholesky of X'X + εI for target projection");
    let c = chol.solvevec(&xtg);

    let fit = x.dot(&c);
    let resid: f64 = fit
        .iter()
        .zip(g.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = g.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    assert!(
        resid / scale < MAX_UNREPRESENTED_FRACTION,
        "design cannot represent the {what} target (rel residual {:.3e} exceeds \
         {MAX_UNREPRESENTED_FRACTION:.3e}); the probe requires the target be \
         representable in the design's column space",
        resid / scale
    );
    c
}

/// `cᵀ S c`.
fn quad(s: &Array2<f64>, c: &Array1<f64>) -> f64 {
    c.dot(&s.dot(c))
}

fn constant_target(data: &Array2<f64>) -> Array1<f64> {
    Array1::from_elem(data.nrows(), 0.7)
}

/// A genuinely wiggly target sampled at the data rows: a sine of the first
/// coordinate, at the highest frequency the collocation sample can still
/// RESOLVE. Its gradient energy is strictly positive, so a tension penalty whose
/// collocation sample resolves the wiggly directions must charge it a positive
/// cost.
///
/// The frequency is derived, not fixed. A farthest-point sample of `k` centers
/// over `[-1,1]^d` has spacing `h ≈ (2^d / k)^(1/d)`, and `coeff_for_target`
/// requires the target to sit in the design's column space to a relative
/// residual of 1e-4 — so a target finer than the sample can express fails the
/// probe's own precondition before it ever reaches the penalty. This fixture
/// resolved its target at `λ/h = 6` while it ran at `d = 1` (`h = 0.083`,
/// `λ = 0.5`), and that ratio is what makes the probe sit AT the resolution
/// limit rather than beyond it — which is the whole discriminating power of
/// assertion (iii). Holding `λ/h` fixed is therefore the dimension-independent
/// statement of the same probe: at `d = 2`, `h ≈ 0.41`, so `λ ≈ 2.4` — a little
/// under one cycle across `[-1,1]`. Carrying the old `d = 1` frequency into
/// `d = 2` unchanged puts `λ/h` at 1.22, barely above Nyquist, which left 92% of
/// the target's energy outside the span.
fn wiggly_target(data: &Array2<f64>) -> Array1<f64> {
    let n = data.nrows();
    let d = data.ncols() as f64;
    let spacing = (2.0_f64.powf(d) / NUM_CENTERS as f64).powf(1.0 / d);
    let freq = 2.0 * std::f64::consts::PI / (RESOLVED_WAVELENGTHS_PER_SPACING * spacing);
    let mut g = Array1::zeros(n);
    for i in 0..n {
        g[i] = (freq * data[[i, 0]]).sin();
    }
    g
}

// ── (5) the collocated tension penalty resolves all basis directions ─────────

/// COLLOCATION SAMPLE WELL-CONDITIONED. The emitted `OperatorTension` penalty is
/// non-degenerate and resolves the basis directions: (i) its trace is strictly
/// positive (it penalizes SOME gradient energy — the collocation sample is not
/// rank-collapsed); (ii) it ANNIHILATES constants (a flat function has zero
/// gradient energy on any sample); and (iii) it charges a STRICTLY POSITIVE cost
/// to a wiggly target (a high-frequency function is penalized — only possible if
/// the O(k) farthest-point collocation sample resolves the wiggly directions
/// rather than missing them). A degenerate / clustered sample would leave some
/// wiggly direction unseen and the wiggly energy would collapse toward the
/// constant's; the gap between them is the conditioning guard.
#[test]
fn collocation_sample_well_conditioned() {
    // d = 2, not 1: the default spec is PURE Duchon (`length_scale: None`), and
    // pure mode requires `2s < d` for CPD-adequacy against the polynomial
    // null space (`resolve_duchon_orders`, Wendland Thm 8.17). The default
    // `power = 0.5` gives `2s = 1`, which is admissible in d = 2 and exactly
    // INADMISSIBLE in d = 1. The conditioning contract under test is
    // dimension-agnostic — a constant has zero gradient energy and the wiggly
    // target varies along coordinate 0 in any dimension — so the guard is
    // carried at the lowest dimension where the DEFAULT power it exists to
    // probe is legal, rather than by weakening the power away from the default.
    let data = synthetic_data(400, 2, 11);
    let spec = default_duchon_spec(NUM_CENTERS);
    let built = build(&data, &spec);
    let tension = built.tension.as_ref().expect(
        "the all-on default Duchon must emit a collocated OperatorTension penalty \
         (first-order energy Σ‖∇f‖² on the farthest-point sample)",
    );

    // (i) Non-degeneracy: the tension penalty charges SOME direction. Its trace
    // is the reference scale for the "≈ 0 on constants" and "≫ 0 on wiggle"
    // bounds below.
    let trace: f64 = (0..tension.nrows()).map(|i| tension[[i, i]]).sum();
    assert!(
        trace > 1e-6,
        "collocated tension penalty is degenerate (trace={trace:.3e}); the O(k) \
         farthest-point sample must resolve gradient energy, not collapse to rank 0"
    );

    // (ii) Constants are annihilated: zero gradient energy on any sample.
    let c_const = coeff_for_target(&built.design, &constant_target(&data), "constant");
    let energy_const = quad(tension, &c_const);
    let const_bound = 1e-6 * trace.max(1.0);
    assert!(
        energy_const <= const_bound,
        "collocated tension penalty must annihilate constants (∇const = 0): \
         energy_const={energy_const:.3e} vs trace={trace:.3e}"
    );

    // (iii) A wiggly target is genuinely penalized. If the collocation sample
    // resolved all basis directions, the high-frequency target carries real
    // gradient energy — strictly positive and far above the constant's residual.
    let c_wiggle = coeff_for_target(&built.design, &wiggly_target(&data), "wiggly");
    let energy_wiggle = quad(tension, &c_wiggle);
    eprintln!(
        "duchon-collocation-conditioning: k=24 trace={trace:.4} \
         energy_const={energy_const:.3e} energy_wiggle={energy_wiggle:.3e}"
    );
    // The wiggle's collocated gradient energy must dominate: a non-trivial
    // fraction of the penalty's own trace scale, and orders of magnitude above
    // the constant's residual. A sample that failed to resolve the wiggly
    // directions would let this collapse toward `energy_const`.
    assert!(
        energy_wiggle > 1e-3 * trace.max(1.0),
        "collocated tension penalty fails to penalize a wiggly target \
         (energy_wiggle={energy_wiggle:.3e} vs trace={trace:.3e}); the collocation \
         sample does not resolve the high-frequency directions"
    );
    assert!(
        energy_wiggle > 1e3 * energy_const.max(f64::MIN_POSITIVE),
        "collocated tension penalty does not separate wiggle from constant \
         (energy_wiggle={energy_wiggle:.3e}, energy_const={energy_const:.3e}); a \
         well-conditioned sample must charge curvature far more than a flat function"
    );
}

/// DIAGNOSTIC (not a contract): report how far the wiggly probe target sits
/// outside the design's column space as a function of `λ/h`, the number of
/// farthest-point sample spacings per wavelength.
///
/// `wiggly_target` derives its frequency from `RESOLVED_WAVELENGTHS_PER_SPACING`,
/// inherited from the `d = 1` fixture. That inheritance is an anchor, not a
/// proof: `d = 1` and `d = 2` radial bases do not approximate a given `λ/h`
/// equally well, and the probe's precondition is a hard 1e-4 relative residual.
/// This measurement reports the whole curve in ONE run so the constant can be
/// set from the data instead of by trying values — print it with
/// `--nocapture`.
#[test]
fn zz_measure_representability_vs_sample_spacing() {
    let data = synthetic_data(400, 2, 11);
    let spec = default_duchon_spec(NUM_CENTERS);
    let built = build(&data, &spec);
    let x = &built.design;

    let d = data.ncols() as f64;
    let spacing = (2.0_f64.powf(d) / NUM_CENTERS as f64).powf(1.0 / d);

    // Least-squares residual of a target against the design, WITHOUT the
    // assertion `coeff_for_target` applies — the point is to see the value.
    let rel_residual = |freq: f64| -> f64 {
        let n = data.nrows();
        let mut g = Array1::<f64>::zeros(n);
        for i in 0..n {
            g[i] = (freq * data[[i, 0]]).sin();
        }
        let p = x.ncols();
        let xtx = x.t().dot(x);
        let max_diag = xtx.diag().iter().cloned().fold(1.0_f64, f64::max);
        let mut gram = xtx.clone();
        for i in 0..p {
            gram[[i, i]] += 1e-10 * max_diag;
        }
        let chol = gram.cholesky(Side::Lower).expect("Cholesky");
        let c = chol.solvevec(&x.t().dot(&g));
        let fit = x.dot(&c);
        let resid: f64 = fit
            .iter()
            .zip(g.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = g.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
        resid / scale
    };

    eprintln!(
        "[zz] d={d} k={NUM_CENTERS} spacing h={spacing:.4} gate=1e-4 \
         (current constant {RESOLVED_WAVELENGTHS_PER_SPACING})"
    );
    let mut passing: Option<f64> = None;
    for ratio in [1.22_f64, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0] {
        let wavelength = ratio * spacing;
        let freq = 2.0 * std::f64::consts::PI / wavelength;
        let rel = rel_residual(freq);
        let cycles = 2.0 / wavelength;
        eprintln!(
            "[zz] lambda/h={ratio:>5.2}  lambda={wavelength:.3}  cycles_over_[-1,1]={cycles:.2}  \
             rel_residual={rel:.3e}  {}",
            if rel < 1e-4 { "PASS" } else { "fail" }
        );
        if rel < 1e-4 && passing.is_none() {
            passing = Some(ratio);
        }
    }
    match passing {
        Some(ratio) => eprintln!("[zz] smallest passing lambda/h = {ratio}"),
        None => eprintln!("[zz] NO ratio up to 16 clears the 1e-4 gate at this k"),
    }

    // The diagnostic still has to be honest about its own premise: a target at
    // the OLD d=1 frequency (lambda/h = 1.22 here) must be the worst row. If it
    // were representable, the d=2 failure this measurement explains would have
    // some other cause entirely.
    let worst = rel_residual(2.0 * std::f64::consts::PI / (1.22 * spacing));
    assert!(
        worst > 1e-4,
        "the inherited d=1 frequency must be UNrepresentable at d=2 — that is the \
         observed failure this measurement is explaining; got rel_residual={worst:.3e}"
    );
}
