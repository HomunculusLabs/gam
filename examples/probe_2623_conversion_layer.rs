//! #2623 probe, second half: grade the layer that converts the #784 channel
//! moments into ρ-derivatives.
//!
//! `probe_2623_channel_isolation` established that the four channel VALUES are
//! exact — `rho_gradient`, `M_r`, `R̃` and `g_d` each reproduce a finite
//! difference of `Δ_b` in the one input they are the derivative with respect to,
//! 28 of 28 rows to 3.1e-7. That leaves exactly one layer unmeasured: the
//! identities that turn those four derivatives into `d(Δ_b)/dρ_j`.
//!
//! ```text
//!   dlambda_r/drho_j = u_r^T Hdot_j u_r
//!   du_r/drho_j      = sum_{q != r} u_q (u_q^T Hdot_j u_r) / (lambda_r - sigma_q)
//!   dbeta/drho_j     = -H^{-1} lambda_j S_j beta                (the IFT response)
//!   Hdot_j           = lambda_j S_j - C[v_j],  C[v] = X^T diag(c * Xv) X
//! ```
//!
//! Each is finite-differenced here against the object it claims to be the
//! derivative of, on a fixture where the eigenframe IS the eigenframe of the
//! penalized Hessian and the mode IS the penalized mode — the two things the
//! isolation probe deliberately took out of the measurement.
//!
//! The fifth row is the composition: `d(Δ_b)/dρ_j` against `a_j − trace_j −
//! mode_j`, the propagation the four measured channel signs force. On this
//! fixture every field is built from one penalty frame and one `H`, so a row
//! that passes here and fails in a real fit indicts the plumbing (which penalty
//! frame `H` carries versus which one the assembly reads), not the calculus.
//!
//! It asserts nothing and prints what it measures.

use gam::inference::hmc_io::block_sampled_marginal_correction;
use gam_problem::laplace_sampler_contract::BlockExcessTarget;
use ndarray::{Array1, Array2, Axis};

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

fn logistic(e: f64) -> f64 {
    if e >= 0.0 {
        1.0 / (1.0 + (-e).exp())
    } else {
        let z = e.exp();
        z / (1.0 + z)
    }
}

fn softplus(e: f64) -> f64 {
    if e >= 0.0 {
        e + (-e).exp().ln_1p()
    } else {
        e.exp().ln_1p()
    }
}

/// Cyclic Jacobi eigendecomposition of a small symmetric matrix, returned with
/// eigenvalues ASCENDING. Chosen over a library call so the probe has no
/// dependency on which LAPACK the workspace links, and because at `p = 10` the
/// sweep is exact to machine precision — which the FD rows below need, since an
/// eigenvector differenced at `h = 1e-5` cannot be graded by a solver whose own
/// error is larger.
fn jacobi_eigh(input: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = input.nrows();
    let mut a = input.clone();
    let mut v = Array2::<f64>::eye(n);
    for _sweep in 0..100 {
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[[i, j]] * a[[i, j]];
            }
        }
        if off.sqrt() < 1.0e-15 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[[p, q]].abs() < 1.0e-300 {
                    continue;
                }
                let theta = (a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[[k, p]];
                    let akq = a[[k, q]];
                    a[[k, p]] = c * akp - s * akq;
                    a[[k, q]] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[[p, k]];
                    let aqk = a[[q, k]];
                    a[[p, k]] = c * apk - s * aqk;
                    a[[q, k]] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[[i, i]].partial_cmp(&a[[j, j]]).expect("finite spectrum"));
    let mut evals = Array1::<f64>::zeros(n);
    let mut evecs = Array2::<f64>::zeros((n, n));
    for (slot, &src) in order.iter().enumerate() {
        evals[slot] = a[[src, src]];
        for k in 0..n {
            evecs[[k, slot]] = v[[k, src]];
        }
    }
    (evals, evecs)
}

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

struct Probe {
    x: Array2<f64>,
    y: Array1<f64>,
    penalties: Vec<Array2<f64>>,
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    block_vecs: Array2<f64>,
    block_lambdas: Array1<f64>,
    eta_hat: Array1<f64>,
    weights_obs: Array1<f64>,
    c_weights: Array1<f64>,
    base_half: f64,
    base_ngs: Array1<f64>,
    penalty_scores: Vec<Array1<f64>>,
}

impl Probe {
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

    /// `H = XᵀWX + Σ_k λ_k S_k`, with NO stabilization ridge: the IFT response
    /// and the eigen-perturbation identities are statements about this matrix,
    /// and adding a ridge here would grade a different one.
    fn hessian(&self) -> Array2<f64> {
        let mut xw = self.x.clone();
        for i in 0..self.x.nrows() {
            let w = self.weights_obs[i];
            xw.row_mut(i).mapv_inplace(|v| v * w);
        }
        let mut h = self.x.t().dot(&xw);
        for (s, &lam) in self.penalties.iter().zip(self.lambdas.iter()) {
            h.scaled_add(lam, s);
        }
        h
    }

    /// `C[v] = Xᵀ diag(c ⊙ Xv) X`.
    fn curvature_drift(&self, v: &Array1<f64>) -> Array2<f64> {
        let xv = self.x.dot(v);
        let mut xd = self.x.clone();
        for i in 0..self.x.nrows() {
            let d = self.c_weights[i] * xv[i];
            xd.row_mut(i).mapv_inplace(|value| value * d);
        }
        self.x.t().dot(&xd)
    }

    fn fit(&mut self) {
        for _ in 0..200 {
            self.rebuild();
            let mut grad = self.x.t().dot(&self.base_ngs);
            for (s, &lam) in self.penalty_scores.iter().zip(self.lambdas.iter()) {
                grad.scaled_add(lam, s);
            }
            let step = cholesky_solve(&self.hessian(), &grad);
            let norm = step.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            self.beta = &self.beta - &step;
            if norm < 1.0e-14 {
                break;
            }
        }
        self.rebuild();
    }
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

/// The fixture at a given `ρ`, with the mode refitted and the block frame taken
/// from `block_cols` of the penalized Hessian's spectrum, sign-aligned to
/// `reference` when one is supplied. Sign alignment matters for every row: the
/// draws `z_s` are fixed, so flipping an eigenvector's sign changes `Δ_b`.
fn build_at(n: usize, amp: f64, rho: &[f64], block_cols: &[usize], reference: Option<&Array2<f64>>) -> Probe {
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
    let mut probe = Probe {
        x,
        y,
        penalties,
        beta: Array1::zeros(p),
        lambdas: Array1::from(rho.iter().map(|r| r.exp()).collect::<Vec<_>>()),
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
    let (evals, evecs) = jacobi_eigh(&probe.hessian());
    let m = block_cols.len();
    let mut vecs = Array2::<f64>::zeros((p, m));
    let mut curvatures = Array1::<f64>::zeros(m);
    for (slot, &col) in block_cols.iter().enumerate() {
        let mut sign = 1.0_f64;
        if let Some(reference) = reference
            && evecs.column(col).dot(&reference.column(slot)) < 0.0
        {
            sign = -1.0;
        }
        for a in 0..p {
            vecs[[a, slot]] = sign * evecs[[a, col]];
        }
        curvatures[slot] = evals[col];
    }
    probe.block_vecs = vecs;
    probe.block_lambdas = curvatures;
    probe
}

/// The full spectrum and frame at the probe's current state, sign-aligned to a
/// reference frame so an FD of an eigenvector is a difference of the SAME
/// eigenvector.
fn spectrum(probe: &Probe, reference: Option<&Array2<f64>>) -> (Array1<f64>, Array2<f64>) {
    let (evals, mut evecs) = jacobi_eigh(&probe.hessian());
    if let Some(reference) = reference {
        for col in 0..evecs.ncols() {
            if evecs.column(col).dot(&reference.column(col)) < 0.0 {
                let flipped = evecs.column(col).mapv(|v| -v);
                evecs.column_mut(col).assign(&flipped);
            }
        }
    }
    (evals, evecs)
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
    println!("  {label:<40} analytic={analytic:+.10e}  FD={fd:+.10e}  rel={rel:.3e}  {verdict}");
}

fn main() {
    gam::init_parallelism();

    let block_cols = [1usize, 3];
    for &(n, amp, r0) in &[
        (300usize, 6.0_f64, -1.0_f64),
        (300, 10.0, 0.0),
        (900, 8.0, -0.5),
    ] {
        let rho = vec![r0, r0 + 0.05];
        println!("\n======== n={n} amp={amp} rho={rho:?} block_cols={block_cols:?} ========");
        let base = build_at(n, amp, &rho, &block_cols, None);
        let p = base.x.ncols();
        let m = block_cols.len();
        let h = base.hessian();
        let (evals, evecs) = spectrum(&base, None);
        println!("  spectrum = {:?}", evals.mapv(|v| (v * 1.0e4).round() / 1.0e4).to_vec());

        let out = block_sampled_marginal_correction(&base).expect("correction");
        let Some(moments) = out.moments.as_ref() else {
            println!("  no moments");
            continue;
        };
        println!(
            "  Delta_b={:+.8e}  ESS={:.1}/{}",
            out.value, out.importance_ess, out.n_draws
        );
        if out.importance_ess < 0.9 * out.n_draws as f64 {
            println!("  ESS below 0.9*S — the FD reference is not resolved; skipping");
            continue;
        }

        // ── the assembly's Q and g_d, transcribed from gradient_hessian.rs ──
        let x = &base.x;
        let xv = x.dot(&base.block_vecs);
        let xv_ett = xv.dot(&moments.e_tt);
        let sigma2 = (&xv_ett * &xv).sum_axis(Axis(1));
        let mut w_xv_ett = xv_ett.clone();
        for i in 0..x.nrows() {
            let w_i = base.weights_obs[i];
            w_xv_ett.row_mut(i).mapv_inplace(|v| v * w_i);
        }
        let delta_mean = base.block_vecs.dot(&moments.e_t);
        let mut g_d = x.t().dot(&(&moments.e_neg_score - &base.base_ngs));
        for (pen, &lam) in base.penalties.iter().zip(base.lambdas.iter()) {
            g_d.scaled_add(lam, &pen.dot(&delta_mean));
        }
        g_d.scaled_add(-0.5, &x.t().dot(&(&base.c_weights * &sigma2)));
        let mut pen_score_total = Array1::<f64>::zeros(p);
        for (score, &lam) in base.penalty_scores.iter().zip(base.lambdas.iter()) {
            pen_score_total.scaled_add(lam, score);
        }
        let mut r_mat = x.t().dot(&moments.e_t_neg_score);
        for r in 0..m {
            r_mat
                .column_mut(r)
                .scaled_add(moments.e_t[r], &pen_score_total);
        }
        r_mat -= &x.t().dot(&w_xv_ett);
        let xvt_etngs = xv.t().dot(&moments.e_t_neg_score);
        let pterm = base.block_vecs.t().dot(&pen_score_total);
        let xvt_w_xv_ett = xv.t().dot(&w_xv_ett);
        let mut m_vec = Array1::<f64>::zeros(m);
        for r in 0..m {
            m_vec[r] =
                -0.5 * (xvt_etngs[(r, r)] + pterm[r] * moments.e_t[r] - xvt_w_xv_ett[(r, r)]);
        }
        let r_tilde = evecs.t().dot(&r_mat);
        let mut g_mat = Array2::<f64>::zeros((p, m));
        for (jr, &col_r) in block_cols.iter().enumerate() {
            let lam_r = base.block_lambdas[jr];
            for q in 0..p {
                if q == col_r {
                    continue;
                }
                g_mat[(q, jr)] = r_tilde[(q, jr)] / (lam_r - evals[q]);
            }
        }
        let q_c_raw = evecs.dot(&g_mat).dot(&base.block_vecs.t());
        let mut q_mat = 0.5 * (&q_c_raw + &q_c_raw.t());
        for jr in 0..m {
            let u_r = base.block_vecs.column(jr);
            let scale = m_vec[jr] / base.block_lambdas[jr];
            for a in 0..p {
                for b in 0..p {
                    q_mat[(a, b)] += scale * u_r[a] * u_r[b];
                }
            }
        }

        for &step in &[1.0e-5_f64, 1.0e-6] {
            println!("\n  -- central FD step h={step:.1e} --");
            for j in 0..base.lambdas.len() {
                let mut plus_rho = rho.clone();
                plus_rho[j] += step;
                let mut minus_rho = rho.clone();
                minus_rho[j] -= step;
                let plus = build_at(n, amp, &plus_rho, &block_cols, Some(&base.block_vecs));
                let minus = build_at(n, amp, &minus_rho, &block_cols, Some(&base.block_vecs));

                let lam_j = base.lambdas[j];
                let a_j = base.penalty_scores[j].mapv(|v| lam_j * v);
                let v_j = cholesky_solve(&h, &a_j);
                let h_dot = &(&base.penalties[j] * lam_j) - &base.curvature_drift(&v_j);

                // (i) the IFT mode response.
                let fd_beta = (&plus.beta - &minus.beta) / (2.0 * step);
                let mut probe_dir = Array1::<f64>::zeros(p);
                for a in 0..p {
                    probe_dir[a] = ((a * 3 + 2) as f64).sin();
                }
                grade(
                    &format!("(i)   dbeta/drho_{j} along d"),
                    -v_j.dot(&probe_dir),
                    fd_beta.dot(&probe_dir),
                );

                let (evals_p, evecs_p) = spectrum(&plus, Some(&evecs));
                let (evals_m, evecs_m) = spectrum(&minus, Some(&evecs));

                // (ii) the eigenvalue perturbation identity.
                for (slot, &col) in block_cols.iter().enumerate() {
                    let u_r = base.block_vecs.column(slot).to_owned();
                    grade(
                        &format!("(ii)  dlambda_{col}/drho_{j}"),
                        u_r.dot(&h_dot.dot(&u_r)),
                        (evals_p[col] - evals_m[col]) / (2.0 * step),
                    );
                }

                // (iii) the eigenvector perturbation identity.
                for (slot, &col_r) in block_cols.iter().enumerate() {
                    let u_r = base.block_vecs.column(slot).to_owned();
                    let mut w = Array1::<f64>::zeros(p);
                    for a in 0..p {
                        w[a] = ((a * 11 + slot * 5 + 1) as f64).cos();
                    }
                    let mut analytic = 0.0_f64;
                    for q in 0..p {
                        if q == col_r {
                            continue;
                        }
                        let u_q = evecs.column(q).to_owned();
                        analytic +=
                            u_q.dot(&w) * u_q.dot(&h_dot.dot(&u_r)) / (base.block_lambdas[slot] - evals[q]);
                    }
                    let fd = (evecs_p.column(col_r).dot(&w) - evecs_m.column(col_r).dot(&w))
                        / (2.0 * step);
                    grade(&format!("(iii) du_{col_r}/drho_{j} along w"), analytic, fd);
                }

                // (iv) the composition: everything moves at once.
                let mut tr_hq = 0.0_f64;
                for a in 0..p {
                    for b in 0..p {
                        tr_hq += h_dot[[a, b]] * q_mat[[b, a]];
                    }
                }
                let trace_j = tr_hq;
                let mode_j = -v_j.dot(&g_d);
                let fd_total = (block_sampled_marginal_correction(&plus)
                    .expect("plus")
                    .value
                    - block_sampled_marginal_correction(&minus)
                        .expect("minus")
                        .value)
                    / (2.0 * step);
                println!(
                    "        a={:+.6e} trace={:+.6e} mode={:+.6e}",
                    out.rho_gradient[j], trace_j, mode_j
                );
                grade(
                    &format!("(iv)  dDelta_b/drho_{j} TOTAL"),
                    out.rho_gradient[j] - trace_j - mode_j,
                    fd_total,
                );
            }
        }
    }
}
