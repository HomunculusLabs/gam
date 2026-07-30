//! #2623 probe: grade EACH #784 gradient channel against a finite difference of
//! `Δ_b` taken in the ONE input that channel is the derivative with respect to,
//! with every other input frozen.
//!
//! This is the measurement the issue thread is blocked on. The existing probe
//! (`probe_2623_sampled_marginal_channel_fd`) differences `Δ_b` in `ρ`, which
//! moves ALL FOUR dependencies at once — the explicit penalty score, the block
//! curvatures `λ_r`, the block frame `u_r`, and the mode `β̂`. Against a
//! near-cancelling total, that FD can say "the assembled sum is wrong" and
//! nothing more; the published verdict ("no sign assignment and no fixed
//! reweighting of the three channels reproduces the FD, so at least one
//! channel's VALUE is wrong") is exactly as far as it can go.
//!
//! Each channel is a derivative in a DIFFERENT input, and the sampler's draws
//! `z_s` are a hash of `(block_dim, rho_dim)` alone — so those inputs can be
//! perturbed one at a time against byte-identical draws. That turns one
//! four-way-confounded comparison into four independent scalar identities:
//!
//! ```text
//!   (a)  ∂Δ_b/∂ρ_k                     =  rho_gradient[k]
//!   (b)  ∂Δ_b/∂λ_r                     = −M_r / λ_r
//!   (c)  ∂Δ_b/∂u_r  along w            = −wᵀ R[:,r]
//!   (d)  ∂Δ_b/∂β̂   along d            = −g_dᵀ d
//! ```
//!
//! `M_r`, `R` and `g_d` are recomputed here EXACTLY as
//! `gam-solve/src/reml/gradient_hessian.rs` assembles them from
//! `BlockSampledMoments`, so a failing row indicts that assembly's moment
//! contraction, not this probe's algebra.
//!
//! Note what these four identities do NOT depend on: none of them uses the
//! eigen-perturbation formulas `dλ_r/dρ_j = u_rᵀ Ḣ_j u_r` and
//! `du_r/dρ_j = Σ_q u_q (u_qᵀ Ḣ_j u_r)/(λ_r − σ_q)`, nor the IFT mode response
//! `dβ̂/dρ_j = −H⁻¹λ_j S_j β̂`. So the block frame here need NOT be an
//! eigenframe of anything — it is an arbitrary orthonormal basis, and the
//! curvatures are its Rayleigh quotients. That is deliberate: it removes the
//! eigensolver from the measurement entirely and grades only the moment→
//! derivative contraction, which is the part the thread has narrowed to.
//!
//! The target mirrors `Gam784BlockTarget` (`state_caches.rs:1713`) for the
//! binomial-logit φ=1 case, where the row oracle's half-deviance and score are
//! elementary: `D_i/2 = log(1+e^η) − y_i η`, `∂(D_i/2)/∂η = μ_i − y_i`,
//! `W_i = μ_i(1−μ_i)`, `c_i = W_i(1−2μ_i)`. Every field the real target derives
//! from `β̂` — `η̂`, `W`, the base half-deviance, the base score, and the penalty
//! scores `S_kβ̂` — is rederived here whenever `β̂` moves, so the (d) row
//! differences the same function of `β̂` the real assembly does.
//!
//! It asserts nothing and prints what it measures.

use gam::inference::hmc_io::block_sampled_marginal_correction;
use gam_problem::laplace_sampler_contract::BlockExcessTarget;
use ndarray::{Array1, Array2, Axis};

// ───────────────────────────── fixture ──────────────────────────────────────

fn second_difference_penalty(p: usize, cols: std::ops::Range<usize>) -> Array2<f64> {
    let k = cols.len();
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    let block = d.t().dot(&d);
    let mut full = Array2::<f64>::zeros((p, p));
    for (a, ca) in cols.clone().enumerate() {
        for (b, cb) in cols.clone().enumerate() {
            full[[ca, cb]] = block[[a, b]];
        }
    }
    full
}

/// Everything the excess needs, split into the inputs the four channels
/// differentiate and the fields DERIVED from them. `rebuild` is what makes the
/// `β̂` row honest: it re-derives every β̂-dependent field, exactly as a moved
/// mode does inside a real outer evaluation.
struct Probe {
    x: Array2<f64>,
    y: Array1<f64>,
    penalties: Vec<Array2<f64>>,
    // ── the four differentiated inputs ──
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    block_vecs: Array2<f64>,
    block_lambdas: Array1<f64>,
    // ── derived from `beta` ──
    eta_hat: Array1<f64>,
    weights_obs: Array1<f64>,
    c_weights: Array1<f64>,
    base_half: f64,
    base_ngs: Array1<f64>,
    penalty_scores: Vec<Array1<f64>>,
}

fn logistic(e: f64) -> f64 {
    if e >= 0.0 {
        1.0 / (1.0 + (-e).exp())
    } else {
        let z = e.exp();
        z / (1.0 + z)
    }
}

/// `log(1 + e^η)`, evaluated on the stable side of zero.
fn softplus(e: f64) -> f64 {
    if e >= 0.0 {
        e + (-e).exp().ln_1p()
    } else {
        e.exp().ln_1p()
    }
}

impl Probe {
    /// Scaled half-deviance and per-row score at an arbitrary `η`, the mirror of
    /// `Gam784BlockTarget::likelihood_surface_at` at φ = 1.
    fn surface(&self, eta: &Array1<f64>) -> (f64, Array1<f64>) {
        let mut half = 0.0_f64;
        let mut score = Array1::<f64>::zeros(eta.len());
        for i in 0..eta.len() {
            half += softplus(eta[i]) - self.y[i] * eta[i];
            score[i] = logistic(eta[i]) - self.y[i];
        }
        (half, score)
    }

    fn rebuild(&mut self) {
        self.eta_hat = self.x.dot(&self.beta);
        let (half, ngs) = self.surface(&self.eta_hat);
        self.base_half = half;
        self.base_ngs = ngs;
        self.weights_obs = self.eta_hat.mapv(|e| {
            let mu = logistic(e);
            mu * (1.0 - mu)
        });
        self.c_weights = self.eta_hat.mapv(|e| {
            let mu = logistic(e);
            mu * (1.0 - mu) * (1.0 - 2.0 * mu)
        });
        self.penalty_scores = self.penalties.iter().map(|s| s.dot(&self.beta)).collect();
    }

    /// Penalized Hessian `H = XᵀWX + Σ_k λ_k S_k` at the current `β̂`.
    fn hessian(&self) -> Array2<f64> {
        let p = self.x.ncols();
        let mut xw = self.x.clone();
        for i in 0..self.x.nrows() {
            let w = self.weights_obs[i];
            xw.row_mut(i).mapv_inplace(|v| v * w);
        }
        let mut h = self.x.t().dot(&xw);
        for (s, &lam) in self.penalties.iter().zip(self.lambdas.iter()) {
            h.scaled_add(lam, s);
        }
        for a in 0..p {
            h[[a, a]] += 1.0e-10;
        }
        h
    }

    /// Penalized Newton to the mode, so the fixture sits where a real outer
    /// evaluation would call the splice. The identities graded below hold at any
    /// `β̂`; being at the mode only makes the regime representative.
    fn fit(&mut self) {
        for _ in 0..60 {
            self.rebuild();
            let mut grad = self.x.t().dot(&self.base_ngs);
            for (s, &lam) in self.penalty_scores.iter().zip(self.lambdas.iter()) {
                grad.scaled_add(lam, s);
            }
            let step = cholesky_solve(&self.hessian(), &grad);
            let norm = step.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            self.beta = &self.beta - &step;
            if norm < 1.0e-13 {
                break;
            }
        }
        self.rebuild();
    }
}

/// Dense Cholesky solve for the small `p × p` systems here.
fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = sum.max(1.0e-300).sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    let mut yv = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * yv[k];
        }
        yv[i] = sum / l[[i, i]];
    }
    let mut xv = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = yv[i];
        for k in (i + 1)..n {
            sum -= l[[k, i]] * xv[k];
        }
        xv[i] = sum / l[[i, i]];
    }
    xv
}

impl BlockExcessTarget for Probe {
    fn block_dim(&self) -> usize {
        self.block_lambdas.len()
    }

    fn rho_dim(&self) -> usize {
        self.lambdas.len()
    }

    fn block_curvatures(&self) -> &Array1<f64> {
        &self.block_lambdas
    }

    fn excess(&self, t: &Array1<f64>) -> f64 {
        let delta = self.block_vecs.dot(t);
        let s = self.x.dot(&delta);
        let eta = &self.eta_hat + &s;
        let (half, _) = self.surface(&eta);
        let mut penalty_term = 0.0_f64;
        for (score, &lam) in self.penalty_scores.iter().zip(self.lambdas.iter()) {
            penalty_term += lam * score.dot(&delta);
        }
        let mut curv = 0.0_f64;
        for i in 0..s.len() {
            curv += self.weights_obs[i] * s[i] * s[i];
        }
        (half - self.base_half) + penalty_term - 0.5 * curv
    }

    fn excess_rho_gradient(&self, t: &Array1<f64>) -> Array1<f64> {
        let delta = self.block_vecs.dot(t);
        let mut grad = Array1::<f64>::zeros(self.lambdas.len());
        for (k, (score, &lam)) in self
            .penalty_scores
            .iter()
            .zip(self.lambdas.iter())
            .enumerate()
        {
            grad[k] = lam * score.dot(&delta);
        }
        grad
    }

    fn displaced_neg_score(&self, t: &Array1<f64>) -> Result<Array1<f64>, String> {
        let delta = self.block_vecs.dot(t);
        let s = self.x.dot(&delta);
        let eta = &self.eta_hat + &s;
        Ok(self.surface(&eta).1)
    }

    fn base_neg_score(&self) -> Result<Array1<f64>, String> {
        Ok(self.base_ngs.clone())
    }
}

// ───────────────────── the assembly's own channel algebra ───────────────────

/// `M_r`, `R` and `g_d` recomputed EXACTLY as `gradient_hessian.rs` forms them
/// from the sampler's moments. Nothing here is a re-derivation: the three
/// expressions are transcribed from the assembly so the FD grades that code's
/// contraction rather than an independent one.
struct Channels {
    m_vec: Array1<f64>,
    r_mat: Array2<f64>,
    g_d: Array1<f64>,
}

fn assemble_channels(
    probe: &Probe,
    moments: &gam_problem::laplace_sampler_contract::BlockSampledMoments,
) -> Channels {
    let x = &probe.x;
    let n_rows = x.nrows();
    let p = x.ncols();
    let m = probe.block_lambdas.len();

    let xv = x.dot(&probe.block_vecs); // n × m
    let xv_ett = xv.dot(&moments.e_tt); // n × m
    let sigma2 = (&xv_ett * &xv).sum_axis(Axis(1)); // n
    let mut w_xv_ett = xv_ett.clone();
    for i in 0..n_rows {
        let w_i = probe.weights_obs[i];
        w_xv_ett.row_mut(i).mapv_inplace(|v| v * w_i);
    }

    // Channel (d) moment.
    let delta_mean = probe.block_vecs.dot(&moments.e_t);
    let mut g_d = x.t().dot(&(&moments.e_neg_score - &probe.base_ngs));
    for (pen, &lam) in probe.penalties.iter().zip(probe.lambdas.iter()) {
        g_d.scaled_add(lam, &pen.dot(&delta_mean));
    }
    g_d.scaled_add(-0.5, &x.t().dot(&(&probe.c_weights * &sigma2)));

    // Channel (c) moment.
    let mut pen_score_total = Array1::<f64>::zeros(p);
    for (score, &lam) in probe.penalty_scores.iter().zip(probe.lambdas.iter()) {
        pen_score_total.scaled_add(lam, score);
    }
    let mut r_mat = x.t().dot(&moments.e_t_neg_score); // p × m
    for r in 0..m {
        r_mat
            .column_mut(r)
            .scaled_add(moments.e_t[r], &pen_score_total);
    }
    r_mat -= &x.t().dot(&w_xv_ett);

    // Channel (b) moment.
    let xvt_etngs = xv.t().dot(&moments.e_t_neg_score);
    let pterm = probe.block_vecs.t().dot(&pen_score_total);
    let xvt_w_xv_ett = xv.t().dot(&w_xv_ett);
    let mut m_vec = Array1::<f64>::zeros(m);
    for r in 0..m {
        m_vec[r] = -0.5 * (xvt_etngs[(r, r)] + pterm[r] * moments.e_t[r] - xvt_w_xv_ett[(r, r)]);
    }

    Channels { m_vec, r_mat, g_d }
}

// ─────────────────────────────── driver ─────────────────────────────────────

fn build(n: usize, amp: f64, rho: f64) -> Probe {
    let k = 5usize;
    let p = 2 * k;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    let inv_phi = 2.0 / (1.0 + 5.0_f64.sqrt());
    let half_pi = 0.5 * std::f64::consts::PI;
    for i in 0..n {
        let z = -1.0 + 2.0 * i as f64 / (n as f64 - 1.0);
        let z2 = -1.0 + 2.0 * (0.25 + (i as f64) * inv_phi).fract();
        for j in 0..k {
            let order = (j + 1) as f64;
            x[[i, j]] = (order * half_pi * (z + 1.0)).sin();
            x[[i, k + j]] = (order * half_pi * (z2 + 1.0)).cos();
        }
        let signal = 0.7 * (std::f64::consts::PI * z).sin() + 0.3 * (2.0 * half_pi * z2).cos();
        let prob = logistic(amp * signal);
        let u = (0.5 + (i as f64) * inv_phi).fract();
        y[i] = if u < prob { 1.0 } else { 0.0 };
    }
    let penalties = vec![
        second_difference_penalty(p, 0..k),
        second_difference_penalty(p, k..p),
    ];
    let lambdas = Array1::from(vec![rho.exp(), (rho + 0.05).exp()]);

    let mut probe = Probe {
        x,
        y,
        penalties,
        beta: Array1::zeros(p),
        lambdas,
        block_vecs: Array2::zeros((p, 0)),
        block_lambdas: Array1::zeros(0),
        eta_hat: Array1::zeros(n),
        weights_obs: Array1::zeros(n),
        c_weights: Array1::zeros(n),
        base_half: 0.0,
        base_ngs: Array1::zeros(n),
        penalty_scores: Vec::new(),
    };
    probe.fit();

    // An arbitrary orthonormal block frame. The four identities graded below do
    // not use the eigen-perturbation formulas, so this need not diagonalize
    // anything; the curvatures are the frame's Rayleigh quotients, which is what
    // makes the whitened proposal the right scale.
    let m = 2usize;
    let mut vecs = Array2::<f64>::zeros((p, m));
    for r in 0..m {
        for a in 0..p {
            vecs[[a, r]] = ((a * 7 + r * 13 + 3) as f64).sin() + 0.3 * (r as f64 + 1.0);
        }
    }
    // Modified Gram-Schmidt.
    for r in 0..m {
        for q in 0..r {
            let proj = vecs.column(q).dot(&vecs.column(r));
            let col_q = vecs.column(q).to_owned();
            vecs.column_mut(r).scaled_add(-proj, &col_q);
        }
        let norm = vecs.column(r).dot(&vecs.column(r)).sqrt();
        vecs.column_mut(r).mapv_inplace(|v| v / norm);
    }
    let h = probe.hessian();
    let mut curvatures = Array1::<f64>::zeros(m);
    for r in 0..m {
        let u = vecs.column(r).to_owned();
        curvatures[r] = u.dot(&h.dot(&u));
    }
    probe.block_vecs = vecs;
    probe.block_lambdas = curvatures;
    probe
}

/// `Δ_b` at the probe's current inputs, against the same fixed draws every time.
fn delta_b(probe: &Probe) -> f64 {
    block_sampled_marginal_correction(probe)
        .expect("correction")
        .value
}

fn grade(label: &str, analytic: f64, fd: f64) {
    let scale = analytic.abs().max(fd.abs()).max(1.0e-300);
    let rel = (analytic - fd).abs() / scale;
    let verdict = if rel < 1.0e-5 {
        "AGREE"
    } else if rel < 1.0e-2 {
        "near "
    } else {
        "WRONG"
    };
    println!("  {label:<34} analytic={analytic:+.10e}  FD={fd:+.10e}  rel={rel:.3e}  {verdict}");
}

fn main() {
    gam::init_parallelism();

    for &(n, amp, rho) in &[
        (300usize, 6.0_f64, -1.0_f64),
        (300, 10.0, 0.0),
        (900, 6.0, -1.0),
        (900, 12.0, 1.0),
    ] {
        println!("\n======== n={n} amp={amp} rho0={rho} ========");
        let base = build(n, amp, rho);
        let out = block_sampled_marginal_correction(&base).expect("correction");
        let Some(moments) = out.moments.as_ref() else {
            println!("  no moments (every draw carried zero weight)");
            continue;
        };
        let ch = assemble_channels(&base, moments);
        println!(
            "  Delta_b={:+.10e}  ESS={:.1}/{}  se={:.3e}  block_lambdas={:?}",
            out.value,
            out.importance_ess,
            out.n_draws,
            out.standard_error,
            base.block_lambdas.to_vec(),
        );

        // Only report on a cell whose importance sampler is essentially exact:
        // the FD reference is a difference of two Δ_b estimates, so a collapsed
        // weight set makes every row below noise, not evidence.
        if out.importance_ess < 0.9 * out.n_draws as f64 {
            println!("  ESS below 0.9*S — the FD reference is not resolved; skipping the rows");
            continue;
        }

        let m = base.block_lambdas.len();
        let p = base.x.ncols();

        for &h in &[1.0e-4_f64, 1.0e-5] {
            println!("\n  -- central FD step h={h:.1e} --");

            // (a) explicit ρ channel: only `lambdas` moves.
            for kk in 0..base.lambdas.len() {
                // ρ_k = log λ_k is what is differenced, so λ(ρ±h) = λ e^{±h}.
                let mut plus = build(n, amp, rho);
                let mut minus = build(n, amp, rho);
                let lam0 = base.lambdas[kk];
                plus.lambdas[kk] = lam0 * h.exp();
                minus.lambdas[kk] = lam0 * (-h).exp();
                let fd = (delta_b(&plus) - delta_b(&minus)) / (2.0 * h);
                grade(
                    &format!("(a) dDelta_b/drho_{kk}"),
                    out.rho_gradient[kk],
                    fd,
                );
            }

            // (b) block curvature channel: only `block_lambdas[r]` moves.
            for r in 0..m {
                let mut plus = build(n, amp, rho);
                let mut minus = build(n, amp, rho);
                let lam_r = base.block_lambdas[r];
                let step = h * lam_r;
                plus.block_lambdas[r] = lam_r + step;
                minus.block_lambdas[r] = lam_r - step;
                let fd = (delta_b(&plus) - delta_b(&minus)) / (2.0 * step);
                grade(
                    &format!("(b) dDelta_b/dlambda_{r}"),
                    -ch.m_vec[r] / lam_r,
                    fd,
                );
            }

            // (c) block frame channel: only `block_vecs[:,r]` moves, along a
            // fixed direction `w`. The frame is NOT renormalized: `R[:,r]` is
            // the derivative of the unconstrained column, which is what the
            // eigen-perturbation formula feeds it.
            for r in 0..m {
                let mut w = Array1::<f64>::zeros(p);
                for a in 0..p {
                    w[a] = ((a * 11 + r * 5 + 1) as f64).cos();
                }
                let mut plus = build(n, amp, rho);
                let mut minus = build(n, amp, rho);
                for a in 0..p {
                    plus.block_vecs[[a, r]] += h * w[a];
                    minus.block_vecs[[a, r]] -= h * w[a];
                }
                let fd = (delta_b(&plus) - delta_b(&minus)) / (2.0 * h);
                grade(
                    &format!("(c) dDelta_b/du_{r} along w"),
                    -w.dot(&ch.r_mat.column(r)),
                    fd,
                );
            }

            // (d) mode channel: only `β̂` moves, along a fixed direction `d`,
            // with every β̂-derived field rebuilt.
            {
                let mut dvec = Array1::<f64>::zeros(p);
                for a in 0..p {
                    dvec[a] = ((a * 3 + 2) as f64).sin();
                }
                let mut plus = build(n, amp, rho);
                let mut minus = build(n, amp, rho);
                plus.beta = &plus.beta + &(&dvec * h);
                plus.rebuild();
                minus.beta = &minus.beta - &(&dvec * h);
                minus.rebuild();
                let fd = (delta_b(&plus) - delta_b(&minus)) / (2.0 * h);
                grade("(d) dDelta_b/dbeta along d", -ch.g_d.dot(&dvec), fd);
            }
        }
    }
}
