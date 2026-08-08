"""#2612 study: does the Laplace-vs-exact posterior-mean gap grow with the
number of near-null coefficient directions?

The two-parameter reduction in `probe_2612_laplace_vs_exact.py` establishes the
SIGN of the defect (the Gaussian posterior mean is under-confident where the
exact posterior mean is not) but only about a tenth of the MAGNITUDE the
penguins arm shows.  The obvious candidate for the amplification is dimension:
a penalized GAM basis contributes one nearly-unconstrained direction per
penalty-null coordinate, each adding prior variance to `Var(x'beta)` while the
likelihood plateau constrains the exact posterior in a way that does not grow
the same way.

This measures that directly.  Same quasi-separated binary geometry, a
polynomial basis of growing width `p`, the same nearly-flat wall on the
null-space block, and the exact posterior obtained by importance sampling from
a heavy-tailed proposal centred on the mode.  Effective sample size is reported
with every row, because an importance-sampling number without one is not a
measurement.
"""

import numpy as np
from scipy.optimize import minimize

SEED = 20260808


def make_data(n, overlap=0.04):
    x = np.linspace(-1.0, 1.0, n)
    y = (x > 0).astype(float)
    flip = np.abs(x) < overlap
    y[flip] = 1.0 - y[flip]
    return x, y


def design(x, p):
    """Orthonormalised polynomial basis of width `p` (column 0 the intercept)."""
    raw = np.vander(x, p, increasing=True)
    q, _ = np.linalg.qr(raw)
    return q * np.sqrt(len(x))


def penalty(p, lam_null, lam_wiggle):
    """Two-block penalty in the same shape the GAM path uses: a nearly-flat wall
    on the null space (intercept + linear) and a real wiggliness penalty on the
    rest."""
    s = np.zeros((p, p))
    for j in range(p):
        s[j, j] = lam_null if j < 2 else lam_wiggle
    return s


def log_posterior_many(beta, X, y, s_lambda):
    """`beta` is (S, p); returns (S,)."""
    eta = beta @ X.T
    ll = eta @ y - np.logaddexp(0.0, eta).sum(axis=1)
    quad = 0.5 * np.einsum("sp,pq,sq->s", beta, s_lambda, beta)
    return ll - quad


def fit_mode(X, y, s_lambda):
    p = X.shape[1]
    obj = lambda b: -float(log_posterior_many(b[None, :], X, y, s_lambda)[0])

    def grad(b):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
        return -(X.T @ (y - mu)) + s_lambda @ b

    res = minimize(obj, np.zeros(p), jac=grad, method="L-BFGS-B",
                   options={"maxiter": 200000, "ftol": 1e-16, "gtol": 1e-12})
    beta = res.x
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
    w = mu * (1.0 - mu)
    precision = X.T @ (w[:, None] * X) + s_lambda
    return beta, precision


def exact_moments(X, y, s_lambda, beta_hat, cov, x_rows, draws=400_000, df=4.0, inflate=2.0):
    """Importance sampling with a multivariate-t proposal centred at the mode."""
    rng = np.random.default_rng(SEED)
    p = X.shape[1]
    scale = np.linalg.cholesky(inflate * cov + 1e-14 * np.eye(p))
    z = rng.standard_normal((draws, p))
    chi = rng.chisquare(df, draws) / df
    beta = beta_hat + (z / np.sqrt(chi)[:, None]) @ scale.T
    # log proposal density up to a constant
    delta = beta - beta_hat
    solved = np.linalg.solve(inflate * cov, delta.T).T
    maha = np.einsum("sp,sp->s", delta, solved)
    log_q = -0.5 * (df + p) * np.log1p(maha / df)
    log_w = log_posterior_many(beta, X, y, s_lambda) - log_q
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    ess = 1.0 / np.sum(w * w)
    eta_rows = beta @ x_rows.T                       # (S, R)
    probs = 1.0 / (1.0 + np.exp(-np.clip(eta_rows, -700, 700)))
    return w @ probs, ess


def gaussian_moments(beta_hat, cov, x_rows, order=120):
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    weights = weights / weights.sum()
    eta_hat = x_rows @ beta_hat
    sd = np.sqrt(np.maximum(np.einsum("rp,pq,rq->r", x_rows, cov, x_rows), 0.0))
    grid = eta_hat[:, None] + sd[:, None] * nodes[None, :]
    probs = 1.0 / (1.0 + np.exp(-np.clip(grid, -700, 700)))
    return probs @ weights, sd


def predictive_by_marginal_ratio(X, y, s_lambda, beta_hat, precision, x_rows):
    """Candidate repair: the posterior predictive as a RATIO OF NORMALISING
    CONSTANTS, each Laplace-approximated at its own mode.

        E[sigma(x'beta)] = Z(D + one row (x, 1)) / Z(D)

    because the extra row's likelihood factor IS the quantity being averaged.
    Laplace on both sides gives

        E[sigma] ~= exp(L+(beta+) - L(beta)) * sqrt(det H / det H+)

    with `beta+` the mode of the augmented posterior.  Unlike integrating a
    Gaussian in eta, this respects the likelihood plateau: on separated data a
    CONFIRMING extra row barely moves the mode (ratio -> 1) while a
    CONTRADICTING one costs real log-likelihood (ratio -> 0), which is the
    behaviour the exact posterior has and the Gaussian does not.

    Returns the class-1 probability per row, and the (1-normalised) sum of the
    two class ratios as a self-check.
    """
    sign, logdet_h = np.linalg.slogdet(precision)
    if sign <= 0:
        raise RuntimeError("posterior precision is not positive definite")
    base = float(log_posterior_many(beta_hat[None, :], X, y, s_lambda)[0])
    out = np.empty(len(x_rows))
    checksum = np.empty(len(x_rows))
    for r, xr in enumerate(x_rows):
        ratios = []
        for label in (0.0, 1.0):
            x_aug = np.vstack([X, xr[None, :]])
            y_aug = np.concatenate([y, [label]])
            beta_plus, precision_plus = fit_mode_warm(x_aug, y_aug, s_lambda, beta_hat)
            top = float(log_posterior_many(beta_plus[None, :], x_aug, y_aug, s_lambda)[0])
            sign_plus, logdet_plus = np.linalg.slogdet(precision_plus)
            if sign_plus <= 0:
                raise RuntimeError("augmented precision is not positive definite")
            ratios.append(np.exp(top - base + 0.5 * (logdet_h - logdet_plus)))
        total = ratios[0] + ratios[1]
        checksum[r] = total
        out[r] = ratios[1] / total
    return out, checksum


def fit_mode_warm(X, y, s_lambda, start):
    p = X.shape[1]
    obj = lambda b: -float(log_posterior_many(b[None, :], X, y, s_lambda)[0])

    def grad(b):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
        return -(X.T @ (y - mu)) + s_lambda @ b

    res = minimize(obj, start, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 200000, "ftol": 1e-16, "gtol": 1e-12})
    beta = res.x
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
    w = mu * (1.0 - mu)
    return beta, X.T @ (w[:, None] * X) + s_lambda


def log_loss(probs, labels):
    q = np.clip(probs, 1e-15, 1 - 1e-15)
    return float(-np.mean(labels * np.log(q) + (1 - labels) * np.log(1 - q)))


def main():
    n_train = 228
    x, y = make_data(n_train)
    x_hold, y_hold = make_data(115)
    lam_null = 2.173913043e-4
    lam_wiggle = 1.0

    print(f"{'p':>3}  {'max sd(eta)':>11}  {'plug-in':>9}  {'Laplace':>9}  "
          f"{'exact':>9}  {'ratio-fix':>9}  {'L/exact':>8}  {'R/exact':>8}  "
          f"{'ESS':>9}  {'chk':>7}")
    for p in (2, 4, 6, 8, 10, 12, 16):
        X = design(x, p)
        # The held-out design must be the SAME map, so build it jointly and split.
        both = design(np.concatenate([x, x_hold]), p)
        X = both[:n_train]
        X_hold = both[n_train:]
        s_lambda = penalty(p, lam_null, lam_wiggle)
        beta_hat, precision = fit_mode(X, y, s_lambda)
        cov = np.linalg.inv(precision)
        gauss, sd = gaussian_moments(beta_hat, cov, X_hold)
        exact, ess = exact_moments(X, y, s_lambda, beta_hat, cov, X_hold)
        plug = 1.0 / (1.0 + np.exp(-np.clip(X_hold @ beta_hat, -700, 700)))
        ratio_fix, checksum = predictive_by_marginal_ratio(
            X, y, s_lambda, beta_hat, precision, X_hold
        )
        ll_plug = log_loss(plug, y_hold)
        ll_gauss = log_loss(gauss, y_hold)
        ll_exact = log_loss(exact, y_hold)
        ll_ratio = log_loss(ratio_fix, y_hold)
        print(f"{p:3d}  {sd.max():11.4f}  {ll_plug:9.5f}  {ll_gauss:9.5f}  "
              f"{ll_exact:9.5f}  {ll_ratio:9.5f}  "
              f"{ll_gauss / max(ll_exact, 1e-12):8.3f}  "
              f"{ll_ratio / max(ll_exact, 1e-12):8.3f}  {ess:9.1f}  "
              f"{np.abs(checksum - 1.0).max():7.1e}")


if __name__ == "__main__":
    main()
