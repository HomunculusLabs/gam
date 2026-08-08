"""#2612 study: on a quasi-separated logistic fit, which of the three candidate
repairs actually moves the PUBLISHED estimand toward calibration?

The witness fixture is a penalized multinomial GAM and one fit of it costs
minutes, so the candidate repairs cannot be compared there.  This reduces the
geometry to its smallest honest form -- a two-parameter logistic regression on
quasi-separated data with the SAME nearly-flat penalty the multinomial formula
path puts on its penalty-null directions -- and compares, at a grid of test
points:

  * plug-in                sigma(eta_hat)
  * Laplace posterior mean E[sigma(eta)] with eta ~ N(eta_hat, x' H^-1 x)
  * EXACT posterior mean   E[sigma(eta)] under the actual posterior,
                           by dense 2-D quadrature over (a, b)

under three models: the nearly-flat penalty alone, a Jeffreys/Firth term added
to it, and a proper prior strong enough to identify the separating direction.

The question it answers is which object is wrong -- the model, or the Gaussian
approximation to it.  Nothing here reads the Rust tree; it is a statement about
the mathematics both implement.
"""

import numpy as np

RNG = np.random.default_rng(0)


def make_data(n=200, overlap=0.04):
    """Quasi-separated design: y = 1{x > 0} with a thin band of flipped rows."""
    x = np.linspace(-1.0, 1.0, n)
    y = (x > 0).astype(float)
    flip = np.abs(x) < overlap
    y[flip] = 1.0 - y[flip]
    return x, y


def design(x):
    return np.column_stack([np.ones_like(x), x])


def loglik(beta, X, y):
    eta = X @ beta
    # log sigma(eta) for y=1, log(1-sigma) for y=0, stable form
    return np.sum(y * eta - np.logaddexp(0.0, eta))


def fisher(beta, X):
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    w = p * (1.0 - p)
    return X.T @ (w[:, None] * X)


def log_posterior(beta, X, y, lam, firth):
    value = loglik(beta, X, y) - 0.5 * lam * float(beta @ beta)
    if firth:
        info = fisher(beta, X)
        sign, logdet = np.linalg.slogdet(info)
        if sign <= 0:
            return -np.inf
        value += 0.5 * logdet
    return value


def mode_and_hessian(X, y, lam, firth):
    """Maximise the log-posterior on a dense grid, then refine by Newton on a
    numerical gradient/Hessian.  Small enough that brute force is exact."""
    from scipy.optimize import minimize

    obj = lambda b: -log_posterior(b, X, y, lam, firth)
    best = None
    for a0 in (-5.0, 0.0, 5.0):
        for b0 in (0.5, 5.0, 30.0):
            res = minimize(obj, np.array([a0, b0]), method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 200000,
                                    "maxfev": 200000})
            if best is None or res.fun < best.fun:
                best = res
    beta = best.x
    # Numerical Hessian of the NEGATIVE log-posterior = posterior precision.
    h = np.zeros((2, 2))
    step = 1e-4 * np.maximum(1.0, np.abs(beta))
    for i in range(2):
        for j in range(2):
            ei = np.zeros(2); ei[i] = step[i]
            ej = np.zeros(2); ej[j] = step[j]
            f_pp = -log_posterior(beta + ei + ej, X, y, lam, firth)
            f_pm = -log_posterior(beta + ei - ej, X, y, lam, firth)
            f_mp = -log_posterior(beta - ei + ej, X, y, lam, firth)
            f_mm = -log_posterior(beta - ei - ej, X, y, lam, firth)
            h[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step[i] * step[j])
    h = 0.5 * (h + h.T)
    return beta, h


def exact_posterior_grid(X, y, lam, firth, beta_hat, cov, n=1201, widths=14.0):
    """Dense (a, b) quadrature of the actual posterior.

    The window is scaled per coordinate by the Laplace standard deviation, so
    the same routine resolves a posterior with `sd(b) = 0.6` and one with
    `sd(b) = 10` without changing the step-to-scale ratio.  The tail check in
    `report` verifies the window actually contains the mass.
    """
    span_a = widths * float(np.sqrt(max(cov[0, 0], 1e-12)))
    span_b = widths * float(np.sqrt(max(cov[1, 1], 1e-12)))
    a = np.linspace(beta_hat[0] - span_a, beta_hat[0] + span_a, n)
    b = np.linspace(beta_hat[1] - span_b, beta_hat[1] + span_b, n)
    A, B = np.meshgrid(a, b, indexing="ij")
    # Vectorised over the grid, accumulated over rows: the row sums are the only
    # thing the log-posterior needs, and they cost one (n_grid x n_grid) pass
    # each rather than one grid pass per row.
    xs = X[:, 1]
    logp = A * float(np.sum(y)) + B * float(np.sum(y * xs))
    s_w = np.zeros_like(A)
    s_wx = np.zeros_like(A)
    s_wxx = np.zeros_like(A)
    for xi, _yi in zip(xs, y):
        eta = A + B * xi
        logp -= np.logaddexp(0.0, eta)
        if firth:
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0)))
            wi = p * (1.0 - p)
            s_w += wi
            s_wx += wi * xi
            s_wxx += wi * xi * xi
    logp -= 0.5 * lam * (A * A + B * B)
    if firth:
        det = s_w * s_wxx - s_wx * s_wx
        logp += 0.5 * np.log(np.maximum(det, 1e-300))
    logp -= np.max(logp)
    w = np.exp(logp)
    w /= np.sum(w)
    return A, B, w


def predictions(x_test, beta_hat, cov, A, B, w):
    """plug-in, Laplace-Gaussian posterior mean, exact posterior mean."""
    out = []
    # Gauss-Hermite for the 1-D Gaussian in eta.
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / np.sum(weights)
    for x0 in x_test:
        xv = np.array([1.0, x0])
        eta_hat = float(xv @ beta_hat)
        sd = float(np.sqrt(max(xv @ cov @ xv, 0.0)))
        plug = 1.0 / (1.0 + np.exp(-eta_hat))
        eta_nodes = eta_hat + sd * nodes
        gauss = float(np.sum(weights / (1.0 + np.exp(-eta_nodes))))
        eta_grid = A + B * x0
        exact = float(np.sum(w / (1.0 + np.exp(-eta_grid))))
        out.append((x0, eta_hat, sd, plug, gauss, exact))
    return out


def held_out_log_loss(x_hold, y_hold, beta_hat, cov, A, B, w):
    """Mean binary log-loss of each estimand on held-out rows drawn from the
    same law -- the issue's own metric, with no reference implementation."""
    rows = predictions(x_hold, beta_hat, cov, A, B, w)
    totals = np.zeros(3)
    for (x0, _, _, plug, gauss, exact), yy in zip(rows, y_hold):
        for k, p in enumerate((plug, gauss, exact)):
            p = min(max(p, 1e-15), 1.0 - 1e-15)
            totals[k] -= yy * np.log(p) + (1.0 - yy) * np.log(1.0 - p)
    return totals / len(y_hold)


def report(name, X, y, lam, firth, x_test, hold):
    beta_hat, precision = mode_and_hessian(X, y, lam, firth)
    cov = np.linalg.inv(precision)
    A, B, w = exact_posterior_grid(X, y, lam, firth, beta_hat, cov)
    print(f"\n=== {name} ===")
    print(f"  mode = ({beta_hat[0]:+.4f}, {beta_hat[1]:+.4f})   "
          f"Laplace sd(a)={np.sqrt(cov[0,0]):.4g} sd(b)={np.sqrt(cov[1,1]):.4g}")
    ea = float(np.sum(w * A)); eb = float(np.sum(w * B))
    va = float(np.sum(w * (A - ea) ** 2)); vb = float(np.sum(w * (B - eb) ** 2))
    edge = float(w[0, :].sum() + w[-1, :].sum() + w[:, 0].sum() + w[:, -1].sum())
    print(f"  EXACT posterior mean = ({ea:+.4f}, {eb:+.4f})   "
          f"exact sd(a)={np.sqrt(va):.4g} sd(b)={np.sqrt(vb):.4g}   "
          f"grid-edge mass={edge:.2e}")
    print("     x     eta_hat     sd(eta)     plug-in    Laplace-E[p]   exact-E[p]")
    for x0, eta_hat, sd, plug, gauss, exact in predictions(x_test, beta_hat, cov, A, B, w):
        print(f"  {x0:+.2f}  {eta_hat:+10.4f}  {sd:10.4f}  {plug:10.6f}  "
              f"{gauss:12.6f}  {exact:11.6f}")
    x_hold, y_hold = hold
    losses = held_out_log_loss(x_hold, y_hold, beta_hat, cov, A, B, w)
    print(f"  held-out log-loss:  plug-in={losses[0]:.6f}  "
          f"Laplace-E[p]={losses[1]:.6f}  exact-E[p]={losses[2]:.6f}")


def main():
    x, y = make_data()
    X = design(x)
    x_test = np.array([-0.8, -0.3, 0.3, 0.8])
    # Held-out rows from the same law, on the SAME quasi-separated geometry.
    x_hold, y_hold = make_data(n=101, overlap=0.04)
    hold = (x_hold, y_hold)
    # The multinomial formula path's own nearly-flat wall, in the same units:
    # lambda = MULTINOMIAL_FORMULA_PRIOR_PSEUDO_OBS * I1 * (n_ref / n_c).
    report("nearly-flat penalty (lambda = 2.17e-4), no Firth", X, y, 2.173913043e-4, False,
           x_test, hold)
    report("nearly-flat penalty + Jeffreys/Firth", X, y, 2.173913043e-4, True, x_test, hold)
    report("proper prior (lambda = 1.0), no Firth", X, y, 1.0, False, x_test, hold)


if __name__ == "__main__":
    main()
