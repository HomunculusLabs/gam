//! #938 escalation tiers over the smoothing-parameter posterior `π(ρ|y)`.
//!
//! Reference-as-truth: every assertion here is against self-constructed
//! closed-form truth. On an exactly Gaussian criterion
//! `−log π(ρ|y) = ½(ρ−ρ̂)ᵀH(ρ−ρ̂)` the Laplace posterior IS the truth:
//! Tier-1 Gauss-Hermite quadrature must reproduce its moments to quadrature
//! precision, and Tier-2 NUTS (exact gradient, Hessian-whitened) must recover
//! them within Monte-Carlo error under a fixed seed. The auto-selection seam
//! must route an `Escalate` verdict to the right tier by `K`, and the
//! mixture-corrected coefficient covariance must reduce to the plug-in
//! `Vb(ρ̂)` when all mixture weight concentrates at `ρ̂`.

use gam::inference::rho_posterior::{
    ESCALATE_K_HAT, PLUG_IN_CERTIFIED_K_HAT, RhoCertificate, RhoPosteriorCertificate,
    RhoPosteriorEscalation, TIER1_MAX_DIM, TIER2_MAX_DIM, escalate_rho_posterior,
    mixture_coefficient_covariance, rho_posterior_certificate, rho_posterior_nuts,
    rho_posterior_quadrature,
};
use gam::psis;
use gam_math::probability::{normal_cdf, normal_pdf};
use ndarray::{Array1, Array2, array};

/// How many standard errors of the tail-shape estimator must separate the truth
/// from the cutoff a verdict is read against.
///
/// The certificate is deterministic (fixed-seed splitmix64 + Box–Muller), so
/// this is not a per-run confidence level; it is the distance at which the
/// verdict survives an incidental change in the draw stream, the draw count, or
/// the estimator's internals. Four standard errors leave a two-sided Gaussian
/// budget of `2Φ(−4) ≈ 6.3e-5` for a re-seeded fixture to flip.
const RESOLUTION_SIGMAS: f64 = 4.0;

/// Known tail index for the heavy arm.
///
/// `pareto_tail_criterion` builds a proper density exactly when its index is
/// below `1` (the tail decays like `exp(−(1−k)·h·ρ²/2)`), so `1` is the
/// supremum of what this construction can express, and the required draw budget
/// falls as the index rises away from `ESCALATE_K_HAT`. `0.95` is the trade:
/// unambiguously inside the escalate regime, still an integrable posterior, and
/// resolvable at a budget the fixture can afford. Nothing downstream depends on
/// the particular value — the draw count and the tolerance are both derived
/// FROM it.
const HEAVY_TAIL_INDEX: f64 = 0.95;

/// A criterion `−log π(ρ|y)` whose Tier-0 importance weights are EXACTLY Pareto
/// with tail index `k`.
///
/// The certificate proposes `ρ = ρ̂ + z/√h` with `z ~ N(0, 1)` and forms
/// `log w = −criterion(ρ) + criterion(ρ̂) + ½z²`. Taking
///
/// ```text
/// criterion(ρ) = ½·h·ρ² + k·log Φ(√h·ρ)
/// ```
///
/// (with `ρ̂ = 0`) the quadratic cancels exactly against `½z²` and leaves
/// `w ∝ Φ(z)^(−k)`. Since `U = Φ(z)` is standard uniform under the proposal,
/// `P(w > t) = P(U < t^(−1/k)) = t^(−1/k)` — an exact Pareto upper tail of index
/// `k`, with NO slowly varying factor to bias the fit. That is the ground truth
/// the assertions below are scored against: unlike "this target looks heavy",
/// the tail index here is a number the construction fixes.
fn pareto_tail_criterion(k: f64, h: f64, rho: f64) -> f64 {
    let z = h.sqrt() * rho;
    0.5 * h * rho * rho + k * normal_cdf(z).ln()
}

/// `d/dρ` of [`pareto_tail_criterion`]: `h·ρ + k·√h·φ(z)/Φ(z)`.
fn pareto_tail_criterion_grad(k: f64, h: f64, rho: f64) -> f64 {
    let root_h = h.sqrt();
    let z = root_h * rho;
    h * rho + k * root_h * normal_pdf(z) / normal_cdf(z)
}

/// Smallest proposal draw count `M` at which the certificate's shape estimator
/// can actually place a tail of true index `k_true` on the correct side of
/// `cutoff`.
///
/// This is the fix for the failure this fixture used to have: `k̂` is fitted to
/// `psis::tail_count(M)` excesses only and is then shrunk toward `0.5`, so a
/// bare `k̂ > cutoff` assertion at a hand-picked `M` is a coin flip whenever the
/// truth sits inside the estimator's own resolution. Here the budget is DERIVED
/// — the smallest tail sample at which `RESOLUTION_SIGMAS` standard errors fit
/// between the expected report and the cutoff. Because `tail_count(M) = ⌈√M⌉`
/// over this range, candidate budgets are the perfect squares.
fn draws_resolving(k_true: f64, cutoff: f64) -> usize {
    let side = (k_true - cutoff).signum();
    assert!(side != 0.0, "a truth ON the cutoff is not resolvable");
    for n in psis::MIN_TAIL_COUNT..100_000 {
        let m = n * n;
        if psis::tail_count(m) != n {
            continue;
        }
        let gap = (psis::expected_reported_shape(k_true, n) - cutoff) * side;
        if gap >= RESOLUTION_SIGMAS * psis::shape_resolution(k_true, n) {
            return m;
        }
    }
    panic!("no draw budget below 10^10 resolves tail index {k_true} against cutoff {cutoff}");
}

/// The reported shape must land within the estimator's own resolution of the
/// value the construction guarantees. This is what makes the verdict assertions
/// non-vacuous: a certificate that reported a constant, or a PSIS fit that lost
/// the tail, would miss the derived band even when the routing happened to come
/// out right.
fn assert_resolved_shape(cert: &RhoPosteriorCertificate, k_true: f64) {
    let n_tail = psis::tail_count(cert.n_samples);
    let expected = psis::expected_reported_shape(k_true, n_tail);
    let resolution = psis::shape_resolution(k_true, n_tail);
    let deviation = (cert.k_hat - expected).abs();
    // Printed unconditionally: the fixture's whole subject is how finely `k_hat`
    // can be resolved, and that is unreadable if it only surfaces on failure.
    println!(
        "rho-certificate: k_true = {k_true}, M = {}, tail = {n_tail}, k_hat = {}, \
         expected {expected} +/- {resolution}, deviation = {} sigma, verdict = {:?}",
        cert.n_samples,
        cert.k_hat,
        deviation / resolution,
        cert.certificate,
    );
    assert!(
        deviation <= RESOLUTION_SIGMAS * resolution,
        "k_hat = {} must recover the construction's tail index {k_true} within the \
         estimator's resolution: expected {expected} +/- {RESOLUTION_SIGMAS} x {resolution} \
         at M = {} draws (tail sample {n_tail}), deviation {deviation}",
        cert.k_hat,
        cert.n_samples,
    );
}

/// `½ (ρ−ρ̂)ᵀ H (ρ−ρ̂)` — the criterion whose exact posterior is `N(ρ̂, H⁻¹)`.
fn gaussian_quadratic(rho: &Array1<f64>, rho_hat: &Array1<f64>, h: &Array2<f64>) -> f64 {
    let d = rho - rho_hat;
    let mut q = 0.0;
    for i in 0..d.len() {
        for j in 0..d.len() {
            q += d[i] * h[[i, j]] * d[j];
        }
    }
    0.5 * q
}

/// `∇ = H (ρ−ρ̂)` for the quadratic criterion above.
fn gaussian_quadratic_grad(
    rho: &Array1<f64>,
    rho_hat: &Array1<f64>,
    h: &Array2<f64>,
) -> Array1<f64> {
    let d = rho - rho_hat;
    let k = d.len();
    Array1::from_shape_fn(k, |i| (0..k).map(|j| h[[i, j]] * d[j]).sum())
}

fn inverse_2x2(h: &Array2<f64>) -> Array2<f64> {
    let det = h[[0, 0]] * h[[1, 1]] - h[[0, 1]] * h[[1, 0]];
    array![
        [h[[1, 1]] / det, -h[[0, 1]] / det],
        [-h[[1, 0]] / det, h[[0, 0]] / det]
    ]
}

/// (a) Tier-1 quadrature on an exact Gaussian quadratic criterion reproduces
/// the Laplace posterior moments to quadrature precision: mean = ρ̂ and
/// covariance = H⁻¹ (the GH rule integrates degree-2 polynomials exactly, and
/// the importance reweighting is identically the GH weight for this target).
#[test]
fn quadrature_reproduces_laplace_moments_on_gaussian_quadratic() {
    let rho_hat = array![0.4, -1.2];
    let h = array![[2.0, 0.5], [0.5, 1.5]];
    let truth_cov = inverse_2x2(&h);

    let mixture = rho_posterior_quadrature(
        &rho_hat,
        &h,
        |rho| Some(gaussian_quadratic(rho, &rho_hat, &h)),
        None,
    )
    .expect("tier-1 quadrature on a Gaussian quadratic must succeed");

    // K = 2 auto-selects 5 nodes per axis -> 25 nodes.
    assert_eq!(mixture.nodes.len(), 25);
    let total: f64 = mixture.nodes.iter().map(|n| n.weight).sum();
    assert!(
        (total - 1.0).abs() < 1e-10,
        "mixture weights must sum to 1, got {total}"
    );
    for i in 0..2 {
        assert!(
            (mixture.mean[i] - rho_hat[i]).abs() < 1e-8,
            "posterior mean component {i} must equal rho_hat: {} vs {}",
            mixture.mean[i],
            rho_hat[i]
        );
        for j in 0..2 {
            assert!(
                (mixture.covariance[[i, j]] - truth_cov[[i, j]]).abs() < 1e-8,
                "posterior covariance [{i},{j}] must equal H^-1: {} vs {}",
                mixture.covariance[[i, j]],
                truth_cov[[i, j]]
            );
        }
    }
    // The exact-Gaussian target makes the importance correction a no-op, so
    // the node weights ARE the GH weights and the ESS is near the node count.
    assert!(
        mixture.effective_sample_size > 5.0,
        "Gaussian target must keep a healthy quadrature ESS, got {}",
        mixture.effective_sample_size
    );
}

/// (b) Tier-2 NUTS on the same Gaussian quadratic recovers mean and covariance
/// within Monte-Carlo error with a fixed seed, and the run is deterministic
/// (same seed -> bit-identical moments).
#[test]
fn nuts_recovers_gaussian_quadratic_moments_with_fixed_seed() {
    let rho_hat = array![0.4, -1.2];
    let h = array![[2.0, 0.5], [0.5, 1.5]];
    let truth_cov = inverse_2x2(&h);
    let seed = 0x938_0002_u64;

    let run = || {
        rho_posterior_nuts(
            &rho_hat,
            &h,
            |rho: &Array1<f64>| {
                Some((
                    gaussian_quadratic(rho, &rho_hat, &h),
                    gaussian_quadratic_grad(rho, &rho_hat, &h),
                ))
            },
            512,
            seed,
        )
        .expect("tier-2 NUTS on a Gaussian quadratic must succeed")
    };
    let samples = run();

    assert!(samples.converged, "rhat = {} must be < 1.1", samples.rhat);
    assert!(samples.samples.nrows() >= 512);
    for i in 0..2 {
        assert!(
            (samples.mean[i] - rho_hat[i]).abs() < 0.12,
            "NUTS mean component {i} outside MC error: {} vs {}",
            samples.mean[i],
            rho_hat[i]
        );
        for j in 0..2 {
            assert!(
                (samples.covariance[[i, j]] - truth_cov[[i, j]]).abs() < 0.2,
                "NUTS covariance [{i},{j}] outside MC error: {} vs {}",
                samples.covariance[[i, j]],
                truth_cov[[i, j]]
            );
        }
    }

    // Deterministic seeding: a second run with the same seed is bit-identical.
    let again = run();
    for i in 0..2 {
        assert_eq!(
            samples.mean[i].to_bits(),
            again.mean[i].to_bits(),
            "fixed-seed NUTS must be deterministic in mean[{i}]"
        );
    }
}

/// (c) The escalation seam routes an `Escalate` verdict to the right tier by
/// `K`: quadrature for `K <= 4`, NUTS for `K <= 16`, and an honest
/// `Unavailable` beyond.
#[test]
fn escalation_routes_by_dimension() {
    // K = 1: a target whose importance weights have a KNOWN Pareto tail index
    // above the escalate cutoff, at a draw budget derived from the estimator's
    // own resolution (see `pareto_tail_criterion` / `draws_resolving`), makes
    // the Tier-0 certificate refuse the plug-in, and the escalation must land
    // on quadrature.
    let rho_hat = array![0.0];
    let h = array![[4.0]];
    let heavy_k = HEAVY_TAIL_INDEX;
    let heavy_draws = draws_resolving(heavy_k, ESCALATE_K_HAT);
    let heavy = |rho: &Array1<f64>| Some(pareto_tail_criterion(heavy_k, h[[0, 0]], rho[0]));
    let cert = rho_posterior_certificate(&rho_hat, &h, heavy, Some(heavy_draws))
        .expect("certificate on the heavy-tailed target");
    assert_resolved_shape(&cert, heavy_k);
    assert_eq!(
        cert.certificate,
        RhoCertificate::Escalate,
        "a target whose importance weights are exactly Pareto with tail index \
         {heavy_k} > {ESCALATE_K_HAT} must escalate; k_hat = {} at M = {} draws \
         (tail sample {}, resolution {})",
        cert.k_hat,
        cert.n_samples,
        psis::tail_count(cert.n_samples),
        psis::shape_resolution(heavy_k, psis::tail_count(cert.n_samples)),
    );

    // Negative control on the SAME construction: dial the known tail index into
    // the plug-in regime and the certificate must decline to escalate. Without
    // this a certificate that escalated unconditionally would satisfy the
    // assertion above.
    let light_k = PLUG_IN_CERTIFIED_K_HAT / 2.0;
    let light_draws = draws_resolving(light_k, PLUG_IN_CERTIFIED_K_HAT);
    let light_cert = rho_posterior_certificate(
        &rho_hat,
        &h,
        |rho: &Array1<f64>| Some(pareto_tail_criterion(light_k, h[[0, 0]], rho[0])),
        Some(light_draws),
    )
    .expect("certificate on the light-tailed control");
    assert_resolved_shape(&light_cert, light_k);
    assert_eq!(
        light_cert.certificate,
        RhoCertificate::PlugInCertified,
        "the identical construction at tail index {light_k} < {PLUG_IN_CERTIFIED_K_HAT} \
         must certify the plug-in, not escalate; k_hat = {} at M = {} draws",
        light_cert.k_hat,
        light_cert.n_samples,
    );

    let escalated = escalate_rho_posterior(&rho_hat, &h, heavy, |rho: &Array1<f64>| {
        Some((
            pareto_tail_criterion(heavy_k, h[[0, 0]], rho[0]),
            Array1::from_vec(vec![pareto_tail_criterion_grad(heavy_k, h[[0, 0]], rho[0])]),
        ))
    });
    assert!(
        matches!(escalated, RhoPosteriorEscalation::Quadrature(_)),
        "K=1 <= {TIER1_MAX_DIM} must route to tier-1 quadrature"
    );

    // K = 6 (> TIER1_MAX_DIM, <= TIER2_MAX_DIM): must route to NUTS.
    let k = 6;
    let rho_hat6 = Array1::<f64>::zeros(k);
    let h6 = Array2::<f64>::eye(k);
    let rh = rho_hat6.clone();
    let h6c = h6.clone();
    let rh2 = rho_hat6.clone();
    let h6c2 = h6.clone();
    let escalated6 = escalate_rho_posterior(
        &rho_hat6,
        &h6,
        move |rho: &Array1<f64>| Some(gaussian_quadratic(rho, &rh, &h6c)),
        move |rho: &Array1<f64>| {
            Some((
                gaussian_quadratic(rho, &rh2, &h6c2),
                gaussian_quadratic_grad(rho, &rh2, &h6c2),
            ))
        },
    );
    match escalated6 {
        RhoPosteriorEscalation::Nuts(samples) => {
            assert_eq!(samples.mean.len(), k);
            for i in 0..k {
                assert!(
                    samples.mean[i].abs() < 0.25,
                    "tier-2 mean[{i}] must be near 0, got {}",
                    samples.mean[i]
                );
            }
        }
        other => panic!("K=6 must route to tier-2 NUTS, got {other:?}"),
    }

    // K = 20 (> TIER2_MAX_DIM): honest Unavailable, criterion never consulted.
    let k_big = TIER2_MAX_DIM + 4;
    let rho_big = Array1::<f64>::zeros(k_big);
    let h_big = Array2::<f64>::eye(k_big);
    let escalated_big = escalate_rho_posterior(
        &rho_big,
        &h_big,
        |rho: &Array1<f64>| {
            panic!(
                "criterion must not be consulted beyond the NUTS cap (rho len {})",
                rho.len()
            )
        },
        |rho: &Array1<f64>| {
            panic!(
                "gradient must not be consulted beyond the NUTS cap (rho len {})",
                rho.len()
            )
        },
    );
    match escalated_big {
        RhoPosteriorEscalation::Unavailable { n_params, reason } => {
            assert_eq!(n_params, k_big);
            assert!(
                reason.contains("unavailable"),
                "the report must say escalation is unavailable: {reason}"
            );
        }
        other => panic!("K={k_big} must report Unavailable, got {other:?}"),
    }
}

/// (d) The mixture-corrected coefficient covariance reduces to the plug-in
/// `Vb(rho_hat)` when all mixture weight concentrates at `rho_hat`: a criterion
/// vastly sharper than the proposal kills every off-center node, the center
/// node IS `rho_hat`, and the spread term vanishes even though `beta(rho)`
/// varies with `rho`.
#[test]
fn mixture_covariance_reduces_to_plug_in_when_weight_concentrates() {
    let rho_hat = array![0.7];
    let h = array![[1.0]];
    // Criterion 1000x sharper than the proposal quadratic: off-center GH nodes
    // (|z| >= 1.35) carry weight ~ exp(-900) ~ 0.
    let mixture = rho_posterior_quadrature(
        &rho_hat,
        &h,
        |rho| {
            let d = rho[0] - rho_hat[0];
            Some(500.0 * d * d)
        },
        None,
    )
    .expect("concentrated quadrature must succeed");
    let center_weight: f64 = mixture
        .nodes
        .iter()
        .filter(|n| (n.rho[0] - rho_hat[0]).abs() < 1e-12)
        .map(|n| n.weight)
        .sum();
    assert!(
        center_weight > 1.0 - 1e-12,
        "all weight must concentrate at rho_hat, center weight = {center_weight}"
    );

    let vb0 = array![[0.3, 0.1], [0.1, 0.2]];
    // beta varies with rho, so a spread-out mixture WOULD inflate the
    // covariance; with concentrated weight it must not.
    let result = mixture_coefficient_covariance(&mixture, |rho| {
        Ok((array![rho[0], 2.0 * rho[0]], vb0.clone()))
    })
    .expect("mixture coefficient covariance");

    let beta_hat = array![rho_hat[0], 2.0 * rho_hat[0]];
    for i in 0..2 {
        assert!(
            (result.beta_bar[i] - beta_hat[i]).abs() < 1e-9,
            "beta_bar[{i}] must reduce to beta(rho_hat): {} vs {}",
            result.beta_bar[i],
            beta_hat[i]
        );
        for j in 0..2 {
            assert!(
                (result.covariance[[i, j]] - vb0[[i, j]]).abs() < 1e-9,
                "V_beta_marginal[[{i},{j}]] must reduce to Vb(rho_hat): {} vs {}",
                result.covariance[[i, j]],
                vb0[[i, j]]
            );
        }
    }
}
