"""#2747 independent replication of the κ-profile criterion in numpy.

Re-implements, from the crate's own formulas:

  * the κ-stereographic Möbius addition and geodesic distance
    (`gam-geometry::manifolds::constant_curvature`),
  * the fill-invariant effective length L(κ) solving g(L,κ) = g(ℓ_ref, 0)
    (`constant_curvature_effective_length_jet`), for the design (data→center)
    and for the penalty (center→center) separately,
  * the λ-profiled Gaussian REML criterion the κ profile minimises
    (`profiled_gaussian_reml_value_kappa_jet`).

and then evaluates V on the (κ, ℓ) PLANE so the shipped one-dimensional fill
slice can be read against the range-profiled criterion min_ℓ V(κ, ℓ).

Independent of the Rust build; agreement with it is checked separately.
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


# ------------------------------------------------------------- fill length ---
def reference_fill(data, centers, ell_ref):
    d0 = 2.0 * np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=-1)
    return np.exp(-d0 / ell_ref).mean()


def effective_length(data, centers, ell_ref, kappa):
    """Newton solve of g(L, κ) = g(ℓ_ref, 0), warm-started at ℓ_ref."""
    target = reference_fill(data, centers, ell_ref)
    d = distance(data, centers, kappa)
    ell = ell_ref
    for _ in range(200):
        k = np.exp(-d / ell)
        g = k.mean()
        g_l = (k * d / (ell * ell)).mean()
        step = (g - target) / g_l
        ell -= step
        if ell <= 0 or not np.isfinite(ell):
            return np.nan
        if abs(step) <= 1e-13 * ell:
            return ell
    return np.nan


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


def model_shipped(data, centers, y, kappa, ell_ref):
    """Design at the data→center L(κ); penalty at the center→center L_S(κ)."""
    ell_d = effective_length(data, centers, ell_ref, kappa)
    ell_s = effective_length(centers, centers, ell_ref, kappa)
    if not (np.isfinite(ell_d) and np.isfinite(ell_s)):
        return None
    z = sum_to_zero_frame(centers.shape[0])
    x = np.exp(-distance(data, centers, kappa) / ell_d) @ z
    s = z.T @ np.exp(-distance(centers, centers, kappa) / ell_s) @ z
    return assemble(x, 0.5 * (s + s.T), y) + (ell_d,)


def model_coherent(data, centers, y, kappa, ell):
    z = sum_to_zero_frame(centers.shape[0])
    x = np.exp(-distance(data, centers, kappa) / ell) @ z
    s = z.T @ np.exp(-distance(centers, centers, kappa) / ell) @ z
    return assemble(x, 0.5 * (s + s.T), y) + (ell,)


def assemble(x, s, y):
    n, p = x.shape
    design = np.hstack([np.ones((n, 1)), x])
    penalty = np.zeros((p + 1, p + 1))
    penalty[1:, 1:] = s
    return reml_profile(design, penalty, y)


# ---------------------------------------------------------------- fixture ----
def fixture(n, m, kappa_star, radius, truth_ell_mult, noise_sd, rng):
    pts = []
    while len(pts) < n:
        a, b = rng.uniform(-1, 1, 2)
        if a * a + b * b <= 1:
            pts.append((a * radius, b * radius))
    data = np.array(pts)
    centers = np.zeros((m, 2))
    for k in range(1, m):
        th = 2 * np.pi * (k - 1) / (m - 1)
        centers[k] = (0.667 * radius * np.cos(th), 0.667 * radius * np.sin(th))
    d0 = 2.0 * np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=-1
    )[np.triu_indices(m, 1)]
    ell_ref = np.sort(d0)[d0.size // 2]
    truth_ell = ell_ref * truth_ell_mult
    dk = distance(data, centers, kappa_star)
    w = np.array([(-1.0) ** k / (1.0 + k) for k in range(m)])
    mu = np.exp(-dk / truth_ell) @ w
    mu = (mu - mu.mean()) / mu.std()
    y = mu + noise_sd * rng.standard_normal(n)
    max_r2 = max(
        (data * data).sum(-1).max(), (centers * centers).sum(-1).max()
    )
    return data, centers, y, ell_ref, 0.5 / max_r2


def main():
    n, m, radius, noise = 200, 7, 0.6, 0.05
    ell_grid = np.exp(np.linspace(-1.8, 1.8, 49))
    for kappa_star in (-1.0, 0.0, 1.0):
        for mult in (0.5, 1.0, 2.0):
            rng = np.random.default_rng(20260802)
            data, centers, y, ell_ref, cap = fixture(
                n, m, kappa_star, radius, mult, noise, rng
            )
            print(
                f"\n=== k*={kappa_star:+.2f} truth_l={ell_ref*mult:.4f} "
                f"({mult}x l_ref={ell_ref:.4f}) cap=+-{cap:.4f} ==="
            )
            print("  kappa    | F@lref      F-prof      l_F     L(k)   "
                  "| C@lref      C-prof      l_C     edf_C")
            argf = [(np.inf, np.nan), (np.inf, np.nan)]
            argc = [(np.inf, np.nan), (np.inf, np.nan)]
            lhat = [np.nan, np.nan]
            for i in range(25):
                kappa = -cap + 2 * cap * i / 24
                rf = model_shipped(data, centers, y, kappa, ell_ref)
                rc = model_coherent(data, centers, y, kappa, ell_ref)
                vf0 = rf[0] if rf else np.nan
                vc0 = rc[0] if rc else np.nan
                lk = rf[3] if rf else np.nan
                bf, bc = (np.inf, np.nan), (np.inf, np.nan, np.nan)
                for ell in ell_ref * ell_grid:
                    r = model_shipped(data, centers, y, kappa, ell)
                    if r and r[0] < bf[0]:
                        bf = (r[0], ell)
                    r = model_coherent(data, centers, y, kappa, ell)
                    if r and r[0] < bc[0]:
                        bc = (r[0], ell, r[2])
                for slot, val in ((0, vf0), (1, bf[0])):
                    if val < argf[slot][0]:
                        argf[slot] = (val, kappa)
                        if slot == 1:
                            lhat[0] = bf[1]
                for slot, val in ((0, vc0), (1, bc[0])):
                    if val < argc[slot][0]:
                        argc[slot] = (val, kappa)
                        if slot == 1:
                            lhat[1] = bc[1]
                print(
                    f"  {kappa:+8.4f} | {vf0:11.5f} {bf[0]:11.5f} {bf[1]:7.4f} "
                    f"{lk:6.4f} | {vc0:11.5f} {bc[0]:11.5f} {bc[1]:7.4f} {bc[2]:7.3f}"
                )
            ins = lambda k: abs(k) < cap * 0.999
            print(
                f"  -> F@lref argmin {argf[0][1]:+.4f} (int={ins(argf[0][1])})   "
                f"F-prof argmin {argf[1][1]:+.4f} (int={ins(argf[1][1])}, l={lhat[0]:.4f})"
            )
            print(
                f"  -> C@lref argmin {argc[0][1]:+.4f} (int={ins(argc[0][1])})   "
                f"C-prof argmin {argc[1][1]:+.4f} (int={ins(argc[1][1])}, l={lhat[1]:.4f})"
                f"   [truth {kappa_star:+.2f}]"
            )


if __name__ == "__main__":
    main()
