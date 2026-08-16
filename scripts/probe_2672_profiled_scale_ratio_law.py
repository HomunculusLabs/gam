"""Offline check of the #2672 profiled-Gaussian reference law.

Fixed design, FIXED lambda, null-true smooth: the experiment that isolates the
profiled scale from lambda selection.  Three things are measured separately.

  1. Is the CONDITIONAL reference right?   Q = (RSS_0 - RSS_f)/sigma^2 against
     sum_j w_j chi2_1,  w = 1 - p_j^2 on the tested block.  This uses the TRUE
     sigma, so it is the known-scale test and the scale plays no part.
  2. Is the RESIDUAL law right?            V = RSS_f/sigma^2 against
     sum_i v_i chi2_1,  v = {p_i^2 over the whole model} U {1 x (n - p)}.
  3. Does the RATIO reference fix the size? W (built exactly as gam builds it
     from the profiled Gaussian log-likelihood) scored against
        P(sum_j w_j chi2_1 - c(W) * sum_i v_i chi2_1 > 0),
        c(W) = exp((W - B)/n) - 1,  B = n log(nu_f/nu_0) + (nu_0 - nu_f),
     versus the SHIPPED P(sum_j w_j chi2_1 > W).
"""
import numpy as np
from scipy import linalg, stats
from scipy.interpolate import BSpline

NODES, WTS = np.polynomial.legendre.leggauss(16)


def imhof_sf(weights, dofs, x, tol=1e-11, panels=1 << 21):
    """P(sum_j lam_j chi2_{h_j} > x) by Imhof inversion; signed lam allowed."""
    weights = np.asarray(weights, float)
    dofs = np.asarray(dofs, float)
    keep = weights != 0.0
    weights, dofs = weights[keep], dofs[keep]
    if weights.size == 0:
        return 1.0 if x < 0 else 0.0
    if np.all(weights > 0) and x <= 0:
        return 1.0
    if np.all(weights < 0) and x >= 0:
        return 0.0
    rate = 0.5 * np.sum(dofs * np.abs(weights)) + 0.5 * abs(x)
    panel = 4.0 * np.pi / max(rate, 1e-300)
    total, lo = 0.0, 0.0
    for _ in range(panels):
        hi = lo + panel
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        u = mid + half * NODES
        lu = np.outer(u, weights)
        theta = 0.5 * (np.arctan(lu) * dofs).sum(axis=1) - 0.5 * x * u
        logrho = 0.25 * (np.log1p(lu ** 2) * dofs).sum(axis=1)
        total += half * np.sum(WTS * np.sin(theta) / (u * np.exp(logrho)))
        lo = hi
        lr = 0.25 * float((np.log1p((lo * weights) ** 2) * dofs).sum())
        bound = np.inf
        active = np.abs(weights) * lo >= 1.0
        Ha = float(dofs[active].sum())
        if Ha > 0:
            bound = 4.0 / (Ha * np.exp(lr))
        if x > 0:
            slack = float((0.5 * dofs * np.abs(weights)
                           / (1.0 + (lo * weights) ** 2)).sum())
            if slack <= 0.25 * x:
                bound = min(bound, 16.0 / (x * lo * np.exp(lr)))
        if bound <= tol:
            break
    return float(np.clip(0.5 + total / np.pi, 0.0, 1.0))


def smooth_block(z, k):
    """Cubic B-spline basis, 2nd-difference penalty, sum-to-zero constrained."""
    degree = 3
    n_inner = k - degree - 1
    inner = np.linspace(0, 1, n_inner + 2)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])
    B = np.empty((len(z), k))
    for j in range(k):
        c = np.zeros(k)
        c[j] = 1.0
        B[:, j] = BSpline(knots, c, degree, extrapolate=False)(np.clip(z, 0, 1))
    B = np.nan_to_num(B)
    D = np.diff(np.eye(k), n=2, axis=0)
    S = D.T @ D
    # Sum-to-zero identifiability: reparameterize onto the orthogonal
    # complement of the column means, exactly as mgcv does.
    m = B.mean(axis=0)
    Q = linalg.qr(m.reshape(-1, 1), mode="full")[0][:, 1:]   # k x (k-1)
    return B @ Q, Q.T @ S @ Q


def shares(Hinv_blk, S_blk):
    """p = eig(B^{1/2} S B^{1/2}) in [0,1], the penalty shares."""
    ev, U = linalg.eigh(Hinv_blk)
    ev = np.clip(ev, 0.0, None)
    root = U @ np.diag(np.sqrt(ev)) @ U.T
    return np.clip(linalg.eigvalsh(root @ S_blk @ root), 0.0, 1.0)


def run(n=30, k=12, lam=1.0e2, reps=2000, seed=0, sigma=0.5):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    Bz, Sz = smooth_block(z, k)
    X0 = np.column_stack([np.ones(n), x])
    X = np.column_stack([X0, Bz])
    p, p0 = X.shape[1], X0.shape[1]
    S = np.zeros((p, p))
    S[p0:, p0:] = Sz
    H = X.T @ X + lam * S
    Hinv = linalg.inv(H)
    A = X @ Hinv @ X.T
    edf_f, nu_f, nu_0 = np.trace(A), n - np.trace(A), n - p0
    IA, IA0 = np.eye(n) - A, np.eye(n) - X0 @ linalg.inv(X0.T @ X0) @ X0.T

    blk = slice(p0, p)
    w = 1.0 - shares(Hinv[blk, blk], lam * S[blk, blk]) ** 2      # Q's spectrum
    v = shares(Hinv, lam * S) ** 2                                # V's, plus 1s
    v_all = np.concatenate([v, np.ones(n - p)])
    B_det = n * np.log(nu_f / nu_0) + (nu_0 - nu_f)

    mu = X0 @ np.array([0.3, 0.8])
    Qs, Vs, Ws = [], [], []
    p_ship, p_ratio = [], []
    for _ in range(reps):
        y = mu + sigma * rng.standard_normal(n)
        rss_f = float(y @ IA @ IA @ y)
        rss_0 = float(y @ IA0 @ y)
        Q, V = (rss_0 - rss_f) / sigma ** 2, rss_f / sigma ** 2
        W = n * np.log(rss_0 / rss_f) + B_det
        Qs.append(Q); Vs.append(V); Ws.append(W)
        p_ship.append(imhof_sf(w, np.ones_like(w), W))
        c = np.exp((W - B_det) / n) - 1.0
        lamv = np.concatenate([w, -c * v_all])
        dofv = np.ones(len(w) + len(v_all))
        p_ratio.append(imhof_sf(lamv, dofv, 0.0))
    Qs, Vs = np.array(Qs), np.array(Vs)

    print(f"n={n} k={k} lam={lam:g} p={p} edf_f={edf_f:.3f} nu_f={nu_f:.2f} B={B_det:+.4f}")
    print(f"   [1] Q: mean {Qs.mean():.4f} vs sum_w {w.sum():.4f} | "
          f"var {Qs.var():.4f} vs 2*sum_w2 {2*(w**2).sum():.4f} | "
          f"KS(exact) {stats.kstest(np.array([imhof_sf(w, np.ones_like(w), q) for q in Qs[:400]]), 'uniform').statistic:.4f}")
    print(f"   [2] V: mean {Vs.mean():.4f} vs sum_v {v_all.sum():.4f} | "
          f"var {Vs.var():.4f} vs 2*sum_v2 {2*(v_all**2).sum():.4f}")
    for name, pv in (("shipped", np.array(p_ship)), ("ratio", np.array(p_ratio))):
        print(f"   [3] {name:<8} size@.10={np.mean(pv<=0.10):.4f} "
              f"size@.05={np.mean(pv<=0.05):.4f} size@.01={np.mean(pv<=0.01):.4f} "
              f"KS={stats.kstest(pv,'uniform').statistic:.4f}")
    print(f"       MC s.e. @.05 = {np.sqrt(0.05*0.95/reps):.4f}")


if __name__ == "__main__":
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    for n in (30, 50, 100, 200):
        for lam in (1.0e2, 1.0):
            run(n=n, k=12, lam=lam, reps=reps, seed=11 + n)
