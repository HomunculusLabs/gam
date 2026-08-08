"""#2612: the marginal-likelihood-ratio predictive, checked in the SHAPE the
production path actually has -- a reference-coded K-class softmax, asymmetric
data, and the SECOND moments the standard-error surface consumes.

The binary study established the estimator; this one establishes that the
generalisation is the mechanical one and that the same machinery supplies every
raw moment the current Smolyak integrator produces:

    E[p_c]        = Z(D + (x, c)) / Z(D)
    E[p_c * p_d]  = Z(D + (x, c) + (x, d)) / Z(D)

with `Z` the posterior normalising constant, each side Laplace-approximated at
its own mode.  Both are ratios of positive quantities, which is exactly the
condition under which the fully-exponential (Tierney-Kadane) Laplace is
O(n^-2) rather than O(n^-1).

Truth is MCMC (parallel random-walk Metropolis in the Laplace metric), reported
with its acceptance rate and a split-half reproducibility check.  An importance
sampler was tried first and rejected: in this 20-dimensional skewed posterior it
returns ESS ~ 800 of 200000 draws, a Monte Carlo error LARGER than the accuracy
being certified, so it cannot adjudicate the comparison it exists for.
"""

import numpy as np
from scipy.optimize import minimize

SEED = 20260808
K = 3


def make_data(n, seed):
    """Asymmetric three-class data along one covariate: unequal class widths and
    unequal class sizes, so no symmetry can make the mode coincide with the
    posterior mean by accident."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    # Boundaries at -0.55 and +0.15: class sizes ~ 22% / 35% / 43%.
    y = np.where(x < -0.55, 0, np.where(x < 0.15, 1, 2))
    # A thin quasi-separation band: flip every 9th row inside 0.05 of a boundary.
    near = (np.abs(x + 0.55) < 0.05) | (np.abs(x - 0.15) < 0.05)
    idx = np.where(near)[0][::9]
    y[idx] = (y[idx] + 1) % K
    return x, y


def design(x, p):
    raw = np.vander(x, p, increasing=True)
    q, _ = np.linalg.qr(raw)
    return q * np.sqrt(len(x))


def penalty(p, lam_null, lam_wiggle):
    s = np.zeros((p, p))
    for j in range(p):
        s[j, j] = lam_null if j < 2 else lam_wiggle
    return s


def log_posterior(beta_flat, X, y, s_lambda, weights):
    """`beta_flat` is (S, p*(K-1)); reference class K-1 has eta == 0."""
    single = beta_flat.ndim == 1
    b = np.atleast_2d(beta_flat)
    p = X.shape[1]
    s, m = b.shape[0], K - 1
    beta = b.reshape(s, m, p)
    eta = np.einsum("smp,np->snm", beta, X)                 # (S, n, K-1)
    full = np.concatenate([eta, np.zeros((s, eta.shape[1], 1))], axis=2)
    lse = np.log(np.exp(full - full.max(axis=2, keepdims=True)).sum(axis=2)) \
        + full.max(axis=2)
    picked = np.take_along_axis(full, y[None, :, None], axis=2)[:, :, 0]
    ll = ((picked - lse) * weights[None, :]).sum(axis=1)
    quad = 0.5 * np.einsum("smp,pq,smq->s", beta, s_lambda, beta)
    value = ll - quad
    return value[0] if single else value


def gradient_and_hessian(beta_flat, X, y, s_lambda, weights):
    p = X.shape[1]
    m = K - 1
    beta = beta_flat.reshape(m, p)
    eta = X @ beta.T                                        # (n, K-1)
    full = np.concatenate([eta, np.zeros((len(X), 1))], axis=1)
    full = full - full.max(axis=1, keepdims=True)
    e = np.exp(full)
    probs = e / e.sum(axis=1, keepdims=True)                # (n, K)
    onehot = np.eye(K)[y]
    resid = (onehot[:, :m] - probs[:, :m]) * weights[:, None]
    grad = -(resid.T @ X).ravel() + s_lambda_full(s_lambda, m) @ beta_flat
    hess = np.zeros((m * p, m * p))
    for a in range(m):
        for c in range(m):
            w = weights * (probs[:, a] * ((a == c) * 1.0 - probs[:, c]))
            block = X.T @ (w[:, None] * X)
            hess[a * p:(a + 1) * p, c * p:(c + 1) * p] = block
    hess += s_lambda_full(s_lambda, m)
    return grad, hess


def s_lambda_full(s_lambda, m):
    p = s_lambda.shape[0]
    out = np.zeros((m * p, m * p))
    for a in range(m):
        out[a * p:(a + 1) * p, a * p:(a + 1) * p] = s_lambda
    return out


def fit_mode(X, y, s_lambda, weights, start):
    obj = lambda b: -log_posterior(b, X, y, s_lambda, weights)
    jac = lambda b: gradient_and_hessian(b, X, y, s_lambda, weights)[0]
    res = minimize(obj, start, jac=jac, method="L-BFGS-B",
                   options={"maxiter": 400000, "ftol": 1e-18, "gtol": 1e-13})
    beta = res.x
    return beta, gradient_and_hessian(beta, X, y, s_lambda, weights)[1]


def softmax_rows(eta):
    full = np.concatenate([eta, np.zeros(eta.shape[:-1] + (1,))], axis=-1)
    full = full - full.max(axis=-1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(axis=-1, keepdims=True)


def gaussian_moments(beta_hat, cov, x_rows, draws=200_000):
    """The estimand as currently implemented: E[softmax(eta)] with eta Gaussian.
    Evaluated by a large deterministic-seed Gaussian sample, which is accurate
    enough for a comparison at the 1e-3 level and avoids reimplementing the
    Smolyak rule."""
    rng = np.random.default_rng(SEED + 1)
    p = len(beta_hat) // (K - 1)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(len(beta_hat)))
    draws_beta = beta_hat + rng.standard_normal((draws, len(beta_hat))) @ chol.T
    beta = draws_beta.reshape(draws, K - 1, p)
    eta = np.einsum("smp,rp->srm", beta, x_rows)
    return softmax_rows(eta).mean(axis=0)


def exact_moments(X, y, s_lambda, weights, beta_hat, cov, x_rows,
                  chains=48, steps=60_000, burn=15_000, thin=25):
    """Truth by MCMC, not importance sampling.

    A t-proposal importance sampler centred on the mode returns ESS ~ 800 of
    200000 in this 20-dimensional, strongly skewed posterior -- a Monte Carlo
    error around 1e-2 on a probability, which is LARGER than the accuracy being
    certified. Random-walk Metropolis with the Laplace covariance as the
    proposal metric does not have that failure mode: it is not reweighting a
    mismatched proposal, and its error is controlled by chain length. Reported
    with the acceptance rate and a split-half reproducibility check, which is
    the MCMC analogue of quoting ESS.
    """
    rng = np.random.default_rng(SEED)
    d = len(beta_hat)
    scale = 2.38 / np.sqrt(d)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(d))
    state = beta_hat + 0.5 * rng.standard_normal((chains, d)) @ chol.T
    logp = log_posterior(state, X, y, s_lambda, weights)
    kept = []
    accepted = 0
    proposed = 0
    for step in range(steps):
        prop = state + scale * rng.standard_normal((chains, d)) @ chol.T
        lp = log_posterior(prop, X, y, s_lambda, weights)
        take = np.log(rng.random(chains)) < (lp - logp)
        state = np.where(take[:, None], prop, state)
        logp = np.where(take, lp, logp)
        accepted += int(take.sum())
        proposed += chains
        if step >= burn and (step - burn) % thin == 0:
            kept.append(state.copy())
    beta = np.concatenate(kept, axis=0)
    p = X.shape[1]

    def moments(sample):
        beta3 = sample.reshape(len(sample), K - 1, p)
        eta = np.einsum("smp,rp->srm", beta3, x_rows)
        probs = softmax_rows(eta)
        return probs.mean(axis=0), np.einsum("srk,srl->rkl", probs, probs) / len(sample)

    mean, second = moments(beta)
    half = len(beta) // 2
    first_mean, _ = moments(beta[:half])
    second_mean, _ = moments(beta[half:])
    split = float(np.abs(first_mean - second_mean).max())
    return mean, second, (accepted / proposed, len(beta), split)


def ratio_moments(X, y, s_lambda, weights, beta_hat, precision, x_rows):
    """Tierney-Kadane: every raw moment is a ratio of normalising constants."""
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        raise RuntimeError("precision not PD")
    base = log_posterior(beta_hat, X, y, s_lambda, weights)
    r = len(x_rows)
    mean = np.zeros((r, K))
    second = np.zeros((r, K, K))

    def augmented(extra_rows, extra_labels, extra_weights):
        x_aug = np.vstack([X, np.array(extra_rows)])
        y_aug = np.concatenate([y, np.array(extra_labels, dtype=int)])
        w_aug = np.concatenate([weights, np.array(extra_weights)])
        beta_plus, precision_plus = fit_mode(x_aug, y_aug, s_lambda, w_aug, beta_hat)
        top = log_posterior(beta_plus, x_aug, y_aug, s_lambda, w_aug)
        sign_plus, logdet_plus = np.linalg.slogdet(precision_plus)
        if sign_plus <= 0:
            raise RuntimeError("augmented precision not PD")
        return np.exp(top - base + 0.5 * (logdet - logdet_plus))

    for i, xr in enumerate(x_rows):
        for c in range(K):
            mean[i, c] = augmented([xr], [c], [1.0])
        for c in range(K):
            for d in range(c, K):
                if c == d:
                    value = augmented([xr, xr], [c, c], [1.0, 1.0])
                else:
                    value = augmented([xr, xr], [c, d], [1.0, 1.0])
                second[i, c, d] = value
                second[i, d, c] = value
    return mean, second


def main():
    n_train, p = 240, 10
    x, y = make_data(n_train, SEED)
    x_hold, y_hold = make_data(60, SEED + 99)
    both = design(np.concatenate([x, x_hold]), p)
    X, X_hold = both[:n_train], both[n_train:]
    weights = np.ones(n_train)
    s_lambda = penalty(p, 2.173913043e-4, 1.0)

    beta_hat, precision = fit_mode(X, y, s_lambda, weights, np.zeros((K - 1) * p))
    cov = np.linalg.inv(precision)

    rows = np.arange(0, len(X_hold), 6)          # a spread of held-out rows
    sample = X_hold[rows]
    exact_mean, exact_second, diag = exact_moments(
        X, y, s_lambda, weights, beta_hat, cov, sample
    )
    gauss_mean = gaussian_moments(beta_hat, cov, sample)
    ratio_mean, ratio_second = ratio_moments(
        X, y, s_lambda, weights, beta_hat, precision, sample
    )
    plug = softmax_rows(np.einsum("mp,rp->rm", beta_hat.reshape(K - 1, p), sample))

    print(f"K={K} p={p} n_train={n_train}  MCMC: acceptance={diag[0]:.3f} "
          f"draws={diag[1]} split-half max|dp|={diag[2]:.2e}")
    print(f"{'row':>4} {'class':>5}   {'plug-in':>9} {'Gaussian':>9} {'ratio':>9} "
          f"{'exact':>9}   {'|G-E|':>9} {'|R-E|':>9}")
    for i in range(len(rows)):
        for c in range(K):
            print(f"{rows[i]:4d} {c:5d}   {plug[i, c]:9.6f} {gauss_mean[i, c]:9.6f} "
                  f"{ratio_mean[i, c] / ratio_mean[i].sum():9.6f} {exact_mean[i, c]:9.6f}   "
                  f"{abs(gauss_mean[i, c] - exact_mean[i, c]):9.2e} "
                  f"{abs(ratio_mean[i, c] / ratio_mean[i].sum() - exact_mean[i, c]):9.2e}")
    print(f"\nratio mass check (sum_c should be 1 before normalising): "
          f"max|sum-1| = {np.abs(ratio_mean.sum(axis=1) - 1.0).max():.3e}")
    gnorm = np.abs(gauss_mean - exact_mean).max()
    rnorm = np.abs(ratio_mean / ratio_mean.sum(axis=1, keepdims=True) - exact_mean).max()
    print(f"max |Gaussian - exact| = {gnorm:.3e}    max |ratio - exact| = {rnorm:.3e}")

    # Second moments: the standard-error surface's input.
    ratio_second_norm = ratio_second / ratio_second.sum(axis=(1, 2), keepdims=True)
    exact_second_norm = exact_second / exact_second.sum(axis=(1, 2), keepdims=True)
    print(f"max |ratio E[p_c p_d] - exact| (row-normalised) = "
          f"{np.abs(ratio_second_norm - exact_second_norm).max():.3e}")


if __name__ == "__main__":
    main()
