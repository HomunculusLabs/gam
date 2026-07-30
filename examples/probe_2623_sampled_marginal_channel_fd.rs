//! #2623 probe: make the #784 block-local sampled-marginalization splice ENGAGE
//! on a small binomial fixture, then finite-difference EACH of its gradient
//! channels separately against the value channel it belongs to.
//!
//! This is the measurement driver behind the #2623 channel verdict, kept in the
//! tree so the verdict is reproducible rather than quoted. It is a reporter, not
//! a gate: it asserts nothing and prints what it measures.
//!
//! It exists because the sign of channels (b), (c) and (d) in the splice assembly
//! cannot be settled by reading the source. The type contract, the assembly
//! comment and the shipped line disagree, and the two self-consistent readings
//! differ by `2*(trace + mode)`. Two things have to be established before any
//! comparison means anything, and both are printed:
//!
//!  * a fixture where the splice ENGAGES at all — the search reports `max|gamma|`
//!    against `tau(n_eff) = sqrt(4.8/n_eff)` for every candidate, so a declining
//!    cell says how far it was from engaging instead of just failing;
//!  * that the sampler RESOLVES its own `Delta_b`, since the FD reference for the
//!    spliced channel is a difference of two `Delta_b` estimates.

use gam::estimate::outer_eval_capture::{
    RhoOuterAudit, enable_rho_outer_audit, take_rho_outer_audit,
};
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};

fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

struct Fixture {
    y: Array1<f64>,
    w: Array1<f64>,
    x: Array2<f64>,
    offset: Array1<f64>,
    penalties: Vec<BlockwisePenalty>,
    opts: ExternalOptimOptions,
}

/// Which design the fixture uses. This is not a knob for taste: the two designs
/// answer different objections, and a verdict that only holds on one of them is
/// not a verdict about the assembly.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Design {
    /// The design shape of the existing `glm-reml/binomial-noncanonical-outer`
    /// gate row: raw powers of `z` in one block, and a SECOND block that is the
    /// same powers perturbed by `1e-3*sin`. Deliberately near-collinear, which
    /// drives the smallest `H_pen` eigenvalue toward zero. Whitened draws are
    /// `t_r = z_r/sqrt(lambda_r)`, so a tiny `lambda_r` makes `t` enormous and
    /// pushes `Delta_b` and the moments into an extreme regime.
    NearCollinearPowers,
    /// Two DISTINCT harmonic blocks over two DISTINCT covariates. Well
    /// conditioned by construction, so an engagement here cannot be blamed on a
    /// near-singular `H_pen` or on runaway whitened draws.
    DistinctHarmonics,
}

/// A binomial fixture whose linear predictor is scaled by `amp`. At `amp` near 1
/// the probabilities sit in a moderate band and the Laplace summary is good; at
/// large `amp` both tails saturate, the working weights collapse, the smallest
/// `H` eigenvalues fall, and the standardized cubic `gamma_r = T[v,v,v] /
/// lambda_r^{3/2}` rises past `tau(n_eff) = sqrt(4.8/n_eff)`.
fn fixture(n: usize, k: usize, amp: f64, link: StandardLink, design: Design) -> Fixture {
    let p = 1 + 2 * k;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    let inv_phi = 2.0 / (1.0 + 5.0_f64.sqrt());
    let half_pi = 0.5 * std::f64::consts::PI;
    for i in 0..n {
        x[[i, 0]] = 1.0;
        let z = -1.0 + 2.0 * i as f64 / (n as f64 - 1.0);
        // A second covariate that is deterministic, equidistributed and not a
        // function of `z` at any polynomial order.
        let z2 = -1.0 + 2.0 * (0.25 + (i as f64) * inv_phi).fract();
        match design {
            Design::NearCollinearPowers => {
                let mut acc = 1.0;
                for j in 0..k {
                    acc *= z;
                    x[[i, 1 + j]] = acc;
                    x[[i, 1 + k + j]] = acc + 1.0e-3 * ((i + j) as f64).sin();
                }
            }
            Design::DistinctHarmonics => {
                for j in 0..k {
                    let order = (j + 1) as f64;
                    x[[i, 1 + j]] = (order * half_pi * (z + 1.0)).sin();
                    x[[i, 1 + k + j]] = (order * half_pi * (z2 + 1.0)).cos();
                }
            }
        }
        let signal = match design {
            Design::NearCollinearPowers => (std::f64::consts::PI * z).sin() * z.cos(),
            Design::DistinctHarmonics => {
                0.7 * (std::f64::consts::PI * z).sin() + 0.3 * (2.0 * half_pi * z2).cos()
            }
        };
        let prob = 1.0 / (1.0 + (-amp * signal).exp());
        let u = (0.5 + (i as f64) * inv_phi).fract();
        y[i] = if u < prob { 1.0 } else { 0.0 };
    }
    let penalties = vec![
        BlockwisePenalty::new(1..(1 + k), second_difference_penalty(k)),
        BlockwisePenalty::new((1 + k)..p, second_difference_penalty(k)),
    ];
    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link)),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1.0e-12,
        nullspace_dims: vec![2, 2],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    };
    Fixture {
        y,
        w: Array1::ones(n),
        x,
        offset: Array1::zeros(n),
        penalties,
        opts,
    }
}

/// One audited value evaluation at `rho`.
fn audited_cost(fix: &Fixture, rho: &Array1<f64>) -> (f64, RhoOuterAudit) {
    enable_rho_outer_audit();
    let cost = evaluate_externalcost_andridge(
        fix.y.view(),
        fix.w.view(),
        fix.x.clone(),
        fix.offset.view(),
        &fix.penalties,
        &fix.opts,
        rho,
    )
    .expect("cost evaluation")
    .0;
    (cost, take_rho_outer_audit().expect("audit armed"))
}

/// One audited value+gradient evaluation at `rho`.
fn audited_gradient(fix: &Fixture, rho: &Array1<f64>) -> (Array1<f64>, RhoOuterAudit) {
    enable_rho_outer_audit();
    let grad = evaluate_externalgradient(
        fix.y.view(),
        fix.w.view(),
        fix.x.clone(),
        fix.offset.view(),
        &fix.penalties,
        &fix.opts,
        rho,
    )
    .expect("gradient evaluation");
    (grad, take_rho_outer_audit().expect("audit armed"))
}

fn main() {
    gam::init_parallelism();

    // ── Phase 1: engagement search ───────────────────────────────────────────
    //
    // Report the activation evidence itself for each candidate, not just
    // engaged/declined: max|gamma| against tau(n_eff) says HOW FAR a declining
    // candidate is from engaging, which a bare verdict does not.
    println!("== engagement search ==");
    println!(
        "design               link       n    k  amp   rho      engaged  max|gamma|  tau        m  \
         Delta_b      se     se/|Db|   ESS/S"
    );
    let mut engaged: Vec<(f64, StandardLink, usize, usize, f64, f64, Design)> = Vec::new();
    for design in [Design::DistinctHarmonics, Design::NearCollinearPowers] {
        for link in [StandardLink::Logit, StandardLink::Probit] {
            for &n in &[240usize, 960, 2400] {
                for &k in &[6usize] {
                    for &amp in &[3.0_f64, 6.0, 10.0, 16.0] {
                        for &r in &[-2.0_f64, -1.0, 0.0, 1.0] {
                        let fix = fixture(n, k, amp, link, design);
                        let rho = Array1::from(vec![r, r + 0.05]);
                        let (_, audit) = audited_cost(&fix, &rho);
                        let (did, mg, tau, m, db, se, ess, ndraws) =
                            match audit.sampled_marginal.as_ref() {
                                Some(s) => (
                                    audit.sampled_marginal_engaged,
                                    s.max_abs_skewness,
                                    s.skewness_threshold,
                                    s.block_cols.len(),
                                    s.delta_b,
                                    s.standard_error,
                                    s.importance_ess,
                                    s.n_draws,
                                ),
                                None => (
                                    false,
                                    f64::NAN,
                                    f64::NAN,
                                    0,
                                    f64::NAN,
                                    f64::NAN,
                                    f64::NAN,
                                    0,
                                ),
                            };
                        let rel_se = se / db.abs();
                        let name = if matches!(link, StandardLink::Logit) {
                            "logit "
                        } else {
                            "probit"
                        };
                        let dname = if design == Design::DistinctHarmonics {
                            "distinct-harmonics  "
                        } else {
                            "near-collinear-power"
                        };
                        println!(
                            "{dname} {name} {n:5} {k:2} {amp:5.1} {r:6.2}   {did:5}   \
                             {mg:10.4}  {tau:9.4} {m:2}  {db:11.4e} {se:9.3e} \
                             {rel_se:9.3e} {ess:6.1}/{ndraws}"
                        );
                        // Rank by EFFECTIVE SAMPLE SIZE, not by se/|Delta_b|.
                        // The two disagree sharply here and the second is the
                        // wrong key: when the importance weights collapse to one
                        // draw, |Delta_b| inflates to 10^2..10^4 nats and
                        // se/|Delta_b| goes SMALL, so ranking on it selects the
                        // most degenerate cells. ESS/S names the thing that has
                        // to be large for `Delta_b` to be an average at all.
                        // Stored negated so the ascending sort puts the best
                        // first.
                        if did && ess.is_finite() && ndraws > 0 {
                            engaged.push((
                                -(ess / ndraws as f64),
                                link,
                                n,
                                k,
                                amp,
                                r,
                                design,
                            ));
                        }
                        }
                    }
                }
            }
        }
    }

    if engaged.is_empty() {
        println!("\nNO CANDIDATE ENGAGED THE SPLICE — the channel FD cannot be taken.");
        return;
    }
    // Take the cells with the LARGEST effective sample size: a channel FD is
    // only as meaningful as the estimator it differentiates.
    engaged.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite resolutions"));
    // Run the best-resolved cell of EACH design, so the verdict is not read off
    // one conditioning regime.
    for design in [Design::DistinctHarmonics, Design::NearCollinearPowers] {
        let cells: Vec<_> = engaged
            .iter()
            .filter(|cell| cell.6 == design)
            .take(2)
            .copied()
            .collect();
        if cells.is_empty() {
            println!("\nno engaged cell for this design");
        }
        for (neg_ess_fraction, link, n, k, amp, r, _) in cells {
            println!(
                "\n######## highest-ESS cell, ESS/S = {:.4e} ########",
                -neg_ess_fraction
            );
            channel_fd(link, n, k, amp, r, design);
        }
    }
}

fn channel_fd(link: StandardLink, n: usize, k: usize, amp: f64, r: f64, design: Design) {
    println!(
        "\n== per-channel FD: design={} link={} n={n} k={k} amp={amp} rho0={r} ==",
        if design == Design::DistinctHarmonics {
            "distinct-harmonics"
        } else {
            "near-collinear-powers"
        },
        if matches!(link, StandardLink::Logit) {
            "logit"
        } else {
            "probit"
        }
    );

    let fix = fixture(n, k, amp, link, design);
    let rho = Array1::from(vec![r, r + 0.05]);
    let (analytic, audit) = audited_gradient(&fix, &rho);
    let sampled = audit
        .sampled_marginal
        .as_ref()
        .expect("engaged config must carry the channel split");
    let (cost0, comp0) = audit.criterion.expect("criterion recorded");
    println!(
        "engaged={} m={} block_cols={:?} Delta_b={:.8e} se={:.3e} ESS={:.1}/{} max|gamma|={:.4} \
         tau={:.4}",
        audit.sampled_marginal_engaged,
        sampled.block_cols.len(),
        sampled.block_cols,
        sampled.delta_b,
        sampled.standard_error,
        sampled.importance_ess,
        sampled.n_draws,
        sampled.max_abs_skewness,
        sampled.skewness_threshold,
    );
    println!(
        "cost={cost0:.10e} components fixed_beta={:.8e} logdet_h={:.8e} logdet_s={:.8e} \
         kkt={:.8e}",
        comp0[0], comp0[1], comp0[2], comp0[3],
    );
    println!("analytic total gradient = {analytic:.8e}");

    for &h in &[1.0e-2_f64, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4] {
        println!("\n-- FD step h={h:.1e} --");
        for j in 0..rho.len() {
            let mut plus = rho.clone();
            plus[j] += h;
            let mut minus = rho.clone();
            minus[j] -= h;
            let (cost_p, audit_p) = audited_cost(&fix, &plus);
            let (cost_m, audit_m) = audited_cost(&fix, &minus);
            let Some(sp) = audit_p.sampled_marginal.as_ref() else {
                println!("j={j}: splice DECLINED at rho+h; the stencil is not comparable");
                continue;
            };
            let Some(sm) = audit_m.sampled_marginal.as_ref() else {
                println!("j={j}: splice DECLINED at rho-h; the stencil is not comparable");
                continue;
            };
            if sp.block_cols != sampled.block_cols || sm.block_cols != sampled.block_cols {
                println!(
                    "j={j}: BLOCK MEMBERSHIP MOVED across the stencil ({:?} / {:?} / {:?}) — the \
                     quotient is not a derivative",
                    sm.block_cols, sampled.block_cols, sp.block_cols,
                );
                continue;
            }
            let (_, cp) = audit_p.criterion.expect("criterion at rho+h");
            let (_, cm) = audit_m.criterion.expect("criterion at rho-h");

            let fd_total = (cost_p - cost_m) / (2.0 * h);
            let fd_delta_b = (sp.delta_b - sm.delta_b) / (2.0 * h);
            // The criterion folds `-Delta_b` into `fixed_beta`, so adding
            // `Delta_b` back recovers the envelope channel the audit's
            // `parts[j].fixed_beta` is the derivative of.
            let fd_fixed_beta = ((cp[0] + sp.delta_b) - (cm[0] + sm.delta_b)) / (2.0 * h);
            let fd_logdet_h = (cp[1] - cm[1]) / (2.0 * h);
            let fd_logdet_s = (cp[2] - cm[2]) / (2.0 * h);

            let a = sampled.explicit_a[j];
            let trace = sampled.trace_bc[j];
            let mode = sampled.mode_d[j];
            let spliced = sampled.spliced[j];
            // The criterion carries `-Delta_b`, so the spliced gradient entry
            // must be `-d(Delta_b)/d(rho_j)`. Under the Delta_b-side reading of
            // all four channels that is `-(a + trace + mode)`; under the shipped
            // line it is `-a + trace + mode`, which asserts instead that
            // `d(Delta_b)/d(rho_j) = a - trace - mode`.
            let flipped = -(a + trace + mode);
            println!(
                "j={j}  channels a={a:+.6e} trace={trace:+.6e} mode={mode:+.6e}  \
                 spliced(shipped)={spliced:+.6e} flipped={flipped:+.6e}"
            );
            // Every sign assignment of the three channels, against the measured
            // total. If the question were a sign, exactly one row would match.
            println!("      FD(Delta_b)={fd_delta_b:+.6e}   candidate sums:");
            for (label, cand) in [
                ("+a+trace+mode", a + trace + mode),
                ("+a-trace-mode", a - trace - mode),
                ("-a+trace+mode", -a + trace + mode),
                ("-a-trace-mode", -a - trace - mode),
                ("+a+trace-mode", a + trace - mode),
                ("+a-trace+mode", a - trace + mode),
                ("+a          ", a),
                ("   trace+mode", trace + mode),
            ] {
                println!(
                    "        {label} = {cand:+.6e}   ratio to FD = {:+.4e}",
                    cand / fd_delta_b,
                );
            }
            if let Some(part) = audit.parts.iter().find(|p| p.index == j) {
                println!(
                    "      envelope fixed_beta analytic={:+.6e} FD={fd_fixed_beta:+.6e} | \
                     logdet_h analytic={:+.6e} FD={fd_logdet_h:+.6e} | logdet_s \
                     analytic={:+.6e} FD={fd_logdet_s:+.6e}",
                    part.fixed_beta, part.logdet_h, part.logdet_s,
                );
            }
            println!(
                "      total analytic={:+.6e} FD={fd_total:+.6e}  rel={:.3e}",
                analytic[j],
                (analytic[j] - fd_total).abs() / analytic[j].abs().max(fd_total.abs()).max(1e-12),
            );
        }
    }
}
