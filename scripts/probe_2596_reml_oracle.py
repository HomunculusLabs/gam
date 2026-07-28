#!/usr/bin/env python3
"""#2596 independent oracle: what SHOULD a REML-selected penalized lognormal AFT
recover on the fixture of `quality_vs_survival_location_scale_lognormal`?

This is a from-scratch reference implementation (numpy/scipy only) of

    log T = eta(x, z) + sigma * eps,      eps ~ N(0, 1)
    eta(x, z) = b0 + bx * x + f(z),       f a k=10 thin-plate regression spline
    right-censoring at an independent time

fitted by penalized maximum likelihood with the smoothing parameter(s) chosen by
LAML (the Laplace-approximate marginal likelihood mgcv/gam both call "REML" for
non-Gaussian likelihoods).

It answers three questions the Rust side cannot answer about itself:

  1. Is the generative surface recoverable at all at n=300 / 46% censoring?
     -> report the truth-RMSE at the ORACLE lambda (the lambda minimizing
        RMSE against the known truth).
  2. Does a CORRECT LAML criterion select a lambda near that oracle?
     -> report lambda_hat, its EDF, and the truth-RMSE there.
  3. Does adding a null-space (double) penalty change the answer?
     -> both single- and double-penalty variants are reported.

Run: python3 scripts/probe_2596_reml_oracle.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# --------------------------------------------------------------------------
# The fixture: byte-identical draw order to the Rust test (NumPy legacy MT19937
# with seed 2471, `random_sample` and `standard_normal` in the same sequence).
# --------------------------------------------------------------------------


def z_effect_truth(z):
    return np.sin(np.pi * z) + 0.5 * np.sin(3.0 * np.pi * z)


def fixture(n=300, seed=2471):
    rng = np.random.RandomState(seed)
    g2 = -np.log(rng.random_sample()) - np.log(rng.random_sample())
    sigma_true = np.sqrt(1.0 / g2)
    x = 2.0 * rng.random_sample(n) - 1.0
    z = 2.0 * rng.random_sample(n) - 1.0
    eps = rng.standard_normal(n)
    cens_u = rng.random_sample(n)
    eta = -0.5 + 0.8 * x + z_effect_truth(z)
    t_event = np.exp(eta + eps * sigma_true)
    c = -np.log(cens_u) * 0.9
    event = (t_event <= c).astype(float)
    t = np.where(event > 0.5, t_event, np.maximum(c, 1e-6))
    return t, event, x, z, sigma_true


# --------------------------------------------------------------------------
# 1-D thin-plate regression spline (mgcv `bs="tp"`), eigen-truncated to k basis
# functions, then sum-to-zero constrained.
# --------------------------------------------------------------------------


def tprs_basis(z, k=10):
    """Return (X, S_range, S_null) for a centred 1-D TPRS with `k` columns.

    m = 2, d = 1 => eta(r) = |r|^3 / 12, null space T = [1, z].
    """
    n = len(z)
    r = np.abs(z[:, None] - z[None, :])
    E = (r ** 3) / 12.0
    T = np.column_stack([np.ones(n), z])
    evals, evecs = np.linalg.eigh(E)
    order = np.argsort(-np.abs(evals))
    keep = order[: k - T.shape[1]]
    Uk = evecs[:, keep]
    Dk = evals[keep]
    Xr = Uk * Dk  # E @ Uk = Uk * Dk
    X = np.column_stack([Xr, T])
    p = X.shape[1]
    S_range = np.zeros((p, p))
    S_range[: len(Dk), : len(Dk)] = np.diag(Dk)
    # The wiggliness penalty must be PSD; flip the sign convention so the
    # quadratic form is delta' Dk delta with Dk >= 0 (mgcv absorbs the sign of
    # the negative eigenvalues into the basis).
    sgn = np.sign(Dk)
    sgn[sgn == 0] = 1.0
    Xr = Xr * sgn
    X = np.column_stack([Xr, T])
    S_range = np.zeros((p, p))
    S_range[: len(Dk), : len(Dk)] = np.diag(np.abs(Dk))
    # Null-space (double) penalty: identity on the T columns only.
    S_null = np.zeros((p, p))
    S_null[len(Dk):, len(Dk):] = np.eye(T.shape[1])

    # Sum-to-zero identifiability constraint on the smooth (absorb into the basis).
    csum = X.sum(axis=0)
    Q, _ = np.linalg.qr(csum[:, None], mode="complete")
    Z = Q[:, 1:]  # p x (p-1)
    Xc = X @ Z
    Sr = Z.T @ S_range @ Z
    Sn = Z.T @ S_null @ Z
    return Xc, Sr, Sn


# --------------------------------------------------------------------------
# Penalized censored-lognormal-AFT likelihood.
# theta = [beta (p_total), log_sigma]
# --------------------------------------------------------------------------


def make_model(t, event, x, z, k=10):
    Xs, Sr, Sn = tprs_basis(z, k=k)
    # Parametric part: intercept + x, unpenalized.
    Xp = np.column_stack([np.ones(len(x)), x])
    X = np.column_stack([Xp, Xs])
    npar = X.shape[1]
    nsmooth = Xs.shape[1]

    def embed(S):
        out = np.zeros((npar, npar))
        out[Xp.shape[1]:, Xp.shape[1]:] = S
        return out

    return X, embed(Sr), embed(Sn), nsmooth, Xp.shape[1]


LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def nll_and_derivs(theta, X, logt, event, need_hess=True):
    """Negative log-likelihood, gradient and Hessian in (beta, log_sigma)."""
    beta = theta[:-1]
    ls = theta[-1]
    sigma = np.exp(ls)
    eta = X @ beta
    u = (logt - eta) / sigma
    ev = event > 0.5
    ce = ~ev

    # Event rows: -log f = ls + log t + log sqrt(2pi) + u^2/2
    # Censored rows: -log S = -log Phi(-u)
    lam = np.zeros_like(u)  # hazard-like ratio phi(u)/Phi(-u) on censored rows
    if ce.any():
        uc = u[ce]
        logsf = norm.logsf(uc)
        lam[ce] = np.exp(norm.logpdf(uc) - logsf)
    nll = 0.0
    if ev.any():
        nll += np.sum(ls + LOG_SQRT_2PI + 0.5 * u[ev] ** 2)
    if ce.any():
        nll += -np.sum(norm.logsf(u[ce]))

    # d nll / d u  (per row)
    dnll_du = np.zeros_like(u)
    dnll_du[ev] = u[ev]
    dnll_du[ce] = lam[ce]
    # d2 nll / du2
    d2 = np.zeros_like(u)
    d2[ev] = 1.0
    d2[ce] = lam[ce] * (lam[ce] - u[ce])

    # u = (logt - X beta) * exp(-ls)
    du_dbeta = -X / sigma  # n x p
    du_dls = -u  # n

    g_beta = du_dbeta.T @ dnll_du
    g_ls = np.sum(dnll_du * du_dls) + np.sum(ev)  # + d/dls of (ls) on event rows
    grad = np.concatenate([g_beta, [g_ls]])

    if not need_hess:
        return nll, grad, None

    # Second derivatives. Let a = dnll_du, b = d2nll_du2.
    # d2 nll/dbeta dbeta = X' diag(b/sigma^2) X   (du/dbeta = -X/sigma, and
    #                       d2u/dbeta2 = 0)
    W = d2 / sigma ** 2
    H_bb = X.T @ (X * W[:, None])
    # d2u/(dbeta dls) = X/sigma  => cross = X'(b * du_dls * (-1/sigma)) + a * (X/sigma)
    cross = X.T @ (d2 * du_dls * (-1.0 / sigma) + dnll_du * (1.0 / sigma))
    # d2u/dls2 = u  => H_ll = sum(b * u^2 + a * u)
    H_ll = np.sum(d2 * u ** 2 + dnll_du * u)
    H = np.zeros((len(theta), len(theta)))
    H[:-1, :-1] = H_bb
    H[:-1, -1] = cross
    H[-1, :-1] = cross
    H[-1, -1] = H_ll
    return nll, grad, H


def fit_penalized(X, Slam, logt, event, theta0=None):
    """Damped Newton on the penalized negative log-likelihood.

    Exact analytic gradient and Hessian, Levenberg damping when the penalized
    Hessian is not positive definite, Armijo backtracking. Converges in ~10
    iterations from a cold start and 2-4 from a warm one.
    """
    npar = X.shape[1]
    if theta0 is None:
        th = np.zeros(npar + 1)
        th[0] = np.mean(logt)
        th[-1] = np.log(np.std(logt) + 1e-3)
    else:
        th = np.array(theta0, dtype=float)

    P = np.zeros((npar + 1, npar + 1))
    P[:npar, :npar] = Slam
    eye = np.eye(npar + 1)

    def pobj(v):
        nll, _, _ = nll_and_derivs(v, X, logt, event, need_hess=False)
        return nll + 0.5 * v @ (P @ v)

    for _ in range(200):
        nll, g, H = nll_and_derivs(th, X, logt, event)
        gp = g + P @ th
        Hp = H + P
        if np.max(np.abs(gp)) < 1e-9:
            break
        damp = 0.0
        for _ in range(40):
            try:
                L = np.linalg.cholesky(Hp + damp * eye)
                step = np.linalg.solve(L.T, np.linalg.solve(L, gp))
                break
            except np.linalg.LinAlgError:
                damp = max(1e-8, damp * 10.0)
        else:
            break
        f0 = nll + 0.5 * th @ (P @ th)
        a = 1.0
        for _ in range(60):
            cand = th - a * step
            if pobj(cand) <= f0:
                th = cand
                break
            a *= 0.5
        else:
            break
    nll, g, H = nll_and_derivs(th, X, logt, event)
    return th, nll, H + P


def pseudo_logdet(S, tol_rel=1e-10):
    ev = np.linalg.eigvalsh(S)
    m = np.max(np.abs(ev))
    keep = ev > tol_rel * max(m, 1e-300)
    return float(np.sum(np.log(ev[keep]))), int(np.sum(keep))


def laml(rho, X, Ss, logt, event, warm=None):
    """LAML criterion (to MINIMIZE) at log-lambda vector rho."""
    Slam = sum(np.exp(r) * S for r, S in zip(rho, Ss))
    th, nll, Hp = fit_penalized(X, Slam, logt, event, theta0=warm)
    npar = X.shape[1]
    beta = th[:npar]
    pen = 0.5 * beta @ (Slam @ beta)
    ld_s, rank = pseudo_logdet(Slam)
    sign, ld_h = np.linalg.slogdet(Hp)
    if sign <= 0:
        return np.inf, th, np.nan
    # V = -(l - pen) - 0.5 log|S|+ + 0.5 log|H| ; minimize
    V = nll + pen - 0.5 * ld_s + 0.5 * ld_h
    # EDF = trace(Hp^-1 H_unpen)
    _, _, H = nll_and_derivs(th, X, logt, event)
    edf = float(np.trace(np.linalg.solve(Hp, H)))
    return V, th, edf


def main():
    t, event, x, z, sigma_true = fixture()
    n = len(t)
    logt = np.log(t)
    print(f"n={n} cens={1 - event.mean():.3f} sigma_true={sigma_true:.4f} "
          f"log_sigma_true={np.log(sigma_true):.4f}")

    truth = 0.8 * x + z_effect_truth(z)
    truth_c = truth - truth.mean()
    signal_rms = np.sqrt(np.mean(truth_c ** 2))
    print(f"signal_rms={signal_rms:.4f}")

    X, Sr, Sn, nsmooth, nparam = make_model(t, event, x, z, k=10)
    npar = X.shape[1]

    def report(tag, rho, Ss):
        V, th, edf = laml(rho, X, Ss, logt, event)
        mu = X @ th[:npar]
        mu_c = mu - mu.mean()
        r = np.sqrt(np.mean((mu_c - truth_c) ** 2))
        print(f"  {tag:28s} rho={np.round(rho, 3)} V={V:.4f} edf={edf:.2f} "
              f"truth_rmse={r:.4f} log_sigma={th[-1]:+.4f}")
        return V, r, edf

    print("\n--- SINGLE penalty (range only), lambda sweep ---")
    best, oracle = None, None
    for rho in np.arange(-12, 13, 0.5):
        V, r, edf = report(f"rho={rho:+.1f}", [rho], [Sr])
        if best is None or V < best[0]:
            best = (V, rho, r, edf)
        if oracle is None or r < oracle[0]:
            oracle = (r, rho, edf)
    print(f"  LAML-optimal on grid: rho={best[1]:+.1f} V={best[0]:.4f} "
          f"truth_rmse={best[2]:.4f} edf={best[3]:.2f}")
    print(f"  ORACLE (min truth RMSE): rho={oracle[1]:+.1f} "
          f"truth_rmse={oracle[0]:.4f} edf={oracle[2]:.2f}")

    print("\n--- DOUBLE penalty (range + null space) ---")
    grid = np.arange(-10, 13, 2.0)
    best2 = None
    for r1 in grid:
        for r2 in grid:
            V, th, edf = laml([r1, r2], X, [Sr, Sn], logt, event)
            mu = X @ th[:npar]
            mu_c = mu - mu.mean()
            rr = np.sqrt(np.mean((mu_c - truth_c) ** 2))
            if best2 is None or V < best2[0]:
                best2 = (V, r1, r2, rr, edf)
    print(f"  LAML-optimal on grid: rho=({best2[1]:+.0f},{best2[2]:+.0f}) "
          f"V={best2[0]:.4f} truth_rmse={best2[3]:.4f} edf={best2[4]:.2f}")

    print("\n--- degenerate references ---")
    # f(z) == 0 (smooth fully shrunk), x slope free.
    Xp = X[:, :nparam]
    th0, nll0, Hp0 = fit_penalized(Xp, np.zeros((nparam, nparam)), logt, event)
    mu0 = Xp @ th0[:nparam]
    mu0_c = mu0 - mu0.mean()
    print(f"  no smooth at all:      truth_rmse="
          f"{np.sqrt(np.mean((mu0_c - truth_c) ** 2)):.4f} "
          f"log_sigma={th0[-1]:+.4f}")
    print(f"  mu identically zero:   truth_rmse={signal_rms:.4f}")


if __name__ == "__main__":
    main()
