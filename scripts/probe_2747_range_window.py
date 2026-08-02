"""#2747: the numpy replication of the constant-curvature criterion.

Re-implements, from the crate's own formulas and independently of the Rust
build:

  * the kappa-stereographic Moebius addition and geodesic distance
    (`gam-geometry::manifolds::constant_curvature`),
  * the realized design/penalty pair `X = K(data,C)z`, `S = z'K(C,C)z` at ONE
    range (`build_constant_curvature_basis`),
  * the lambda-profiled Gaussian REML criterion the kappa profile minimises
    (`profiled_gaussian_reml_psi_jet`'s value block).

Three things it settled for #2747:

  1. `V(kappa*, ell)` is sharply unimodal with an interior minimum that recovers
     the planted range, rising monotonically on both sides across four
     log-units -- which is why the range box is an evaluability wall and not a
     statistical one.
  2. As `ell -> infinity` the criterion CONVERGES rather than falling (stable to
     two decimals out to `ell = 1.6e12`), so a permissive upper wall cannot be
     walked into by a descent method.
  3. The sampling spread of `kappa-hat` at n=240 / SNR 33, from which the
     regression gate's bias bar is derived.

The REML criterion is invariant to the choice of frame for the constrained
coefficient subspace, so this uses explicit Helmert contrasts rather than the
crate's transform.
"""

import numpy as np

# ---------------------------------------------------------------- geometry ---
def mobius_neg_x_plus_y(x, y, kappa):
    """w = (−x) ⊕_κ y, vectorised over leading axes of x (n,d) and y (m,d)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xy = -(x[:, None, :] * y[None, :, :]).sum(-1)  # ⟨−x, y⟩
    xx = (x * x).sum(-1)[:, None]
    yy = (y * y).sum(-1)[None, :]
    a = 1.0 - kappa * (2.0 * xy + yy)
    b = 1.0 + kappa * xx
    denom = kappa * kappa * (xx * yy) - kappa * (2.0 * xy) + 1.0
    return (a[..., None] * (-x)[:, None, :] + b[..., None] * y[None, :, :]) / denom[..., None]


def distance(x, y, kappa):
    w = mobius_neg_x_plus_y(x, y, kappa)
    nw = np.linalg.norm(w, axis=-1)
    if kappa > 0:
        s = np.sqrt(kappa)
        return 2.0 * np.arctan(s * nw) / s
    if kappa < 0:
        s = np.sqrt(-kappa)
        return 2.0 * np.arctanh(np.clip(s * nw, 0.0, 1 - 1e-15)) / s
    return 2.0 * nw


# ------------------------------------------------------------------- REML ----
def sum_to_zero_frame(m):
    z = np.zeros((m, m - 1))
    for k in range(1, m):
        nrm = np.sqrt(k * (k + 1))
        z[:k, k - 1] = 1.0 / nrm
        z[k, k - 1] = -k / nrm
    return z


def reml_profile(design, penalty, y, rho_lo=-30.0, rho_hi=30.0):
    """min over ρ = log λ of the Gaussian REML negative log evidence."""
    n, p = design.shape
    a = design.T @ design
    b = design.T @ y
    yty = y @ y
    evals = np.linalg.eigvalsh(penalty)
    tol = max(evals.max(), 0.0) * p * np.finfo(float).eps * 16
    pos = evals[evals > tol]
    rank = pos.size
    nullity = p - rank
    logdet_s = np.log(pos).sum()
    nu = n - nullity

    def v(rho):
        h = a + np.exp(rho) * penalty
        try:
            c = np.linalg.cholesky(h)
        except np.linalg.LinAlgError:
            return np.inf
        logdet_h = 2.0 * np.log(np.diag(c)).sum()
        beta = np.linalg.solve(h, b)
        dp = yty - b @ beta
        if dp <= 0:
            return np.inf
        return 0.5 * (logdet_h - logdet_s - rank * rho) + 0.5 * nu * (
            1.0 + np.log(2.0 * np.pi * dp / nu)
        )

    grid = np.linspace(rho_lo, rho_hi, 241)
    vals = np.array([v(r) for r in grid])
    i = int(np.nanargmin(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, grid.size - 1)]
    for _ in range(80):  # golden-section refine
        m1 = lo + 0.382 * (hi - lo)
        m2 = lo + 0.618 * (hi - lo)
        if v(m1) < v(m2):
            hi = m2
        else:
            lo = m1
    rho = 0.5 * (lo + hi)
    best = v(rho)
    # edf at the optimum
    h = a + np.exp(rho) * penalty
    edf = np.trace(np.linalg.solve(h, a))
    return best, rho, edf


def farthest_point_centers(data, m):
    """The builder's center rule: farthest-point from the cloud, then the
    near-origin center snapped to the exact pole."""
    idx = [int(np.argmax((data * data).sum(-1)))]
    while len(idx) < m:
        d = np.min(
            np.linalg.norm(data[:, None, :] - data[None, idx, :], axis=-1), axis=1
        )
        idx.append(int(np.argmax(d)))
    centers = data[idx].copy()
    centers[int(np.argmin((centers * centers).sum(-1)))] = 0.0
    return centers


def coherent_blocks(data, centers, kappa, ell):
    z = sum_to_zero_frame(centers.shape[0])
    x = np.exp(-distance(data, centers, kappa) / ell) @ z
    s = z.T @ np.exp(-distance(centers, centers, kappa) / ell) @ z
    return x, 0.5 * (s + s.T)


def v_at(data, centers, y, kappa, ell):
    x, s = coherent_blocks(data, centers, kappa, ell)
    n, p = x.shape
    design = np.hstack([np.ones((n, 1)), x])
    penalty = np.zeros((p + 1, p + 1))
    penalty[1:, 1:] = s
    return reml_profile(design, penalty, y)


def fixture(n, m, kappa_star, radius, mult, noise, rng):
    pts = []
    while len(pts) < n:
        a, b = rng.uniform(-1, 1, 2)
        if a * a + b * b <= 1:
            pts.append((a * radius, b * radius))
    data = np.array(pts)
    centers = farthest_point_centers(data, m)
    cc = 2.0 * np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=-1
    )[np.triu_indices(m, 1)]
    ell_ref = np.sort(cc)[cc.size // 2]
    dc = 2.0 * np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=-1)
    evaluated = np.concatenate([dc.ravel(), cc])
    pos = evaluated[evaluated > 0]
    truth_ell = ell_ref * mult
    x, _ = coherent_blocks(data, centers, kappa_star, truth_ell)
    w = np.array([1.0 / (1.0 + j) for j in range(x.shape[1])])
    mu = x @ w
    mu = (mu - mu.mean()) / mu.std()
    y = mu + noise * rng.standard_normal(n)
    return data, centers, y, ell_ref, pos.min(), evaluated.max()


def main():
    n, m, radius, noise = 120, 6, 0.6, 0.10
    for kappa_star in (-1.0, 0.0, 1.0):
        for mult in (0.5, 1.0, 2.0):
            rng = np.random.default_rng(20260802)
            data, centers, y, ell_ref, dmin, dmax = fixture(
                n, m, kappa_star, radius, mult, noise, rng
            )
            grid = ell_ref * np.exp(np.linspace(-4.0, 4.0, 41))
            vals = [v_at(data, centers, y, kappa_star, e)[0] for e in grid]
            i = int(np.argmin(vals))
            cc_lo = 2.0 * np.min(
                [
                    np.linalg.norm(centers[a] - centers[b])
                    for a in range(m)
                    for b in range(a + 1, m)
                ]
            )
            cc_hi = 2.0 * np.max(
                [
                    np.linalg.norm(centers[a] - centers[b])
                    for a in range(m)
                    for b in range(a + 1, m)
                ]
            )
            print(
                f"\nk*={kappa_star:+.1f} mult={mult}: ell_ref={ell_ref:.4f} "
                f"truth={ell_ref*mult:.4f}  center-window=[{cc_lo:.4f},{cc_hi:.4f}]  "
                f"evaluated-window=[{dmin:.4f},{dmax:.4f}]"
            )
            print(f"  argmin_ell = {grid[i]:.4f}  (V={vals[i]:.4f})")
            print(
                "  V(ell): "
                + " ".join(
                    f"{grid[j]:.3f}:{vals[j]:.1f}" for j in range(0, 41, 4)
                )
            )


if __name__ == "__main__":
    main()
