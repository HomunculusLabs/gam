"""#2442 — truncated-posterior moments when the ambient precision is indefinite.

Run with:

    uv run --no-project --with numpy --with scipy \
        python artifacts/issue_2442_copositive_orthant_moments.py

This script settles, on the EXACT geometry of the fit that #2442 reports as
refused, the two questions the issue leaves open, and then validates the
algorithm that replaces the refusal.  It shares no code with `gam`: the
geometry is a recorded fixture and every number below is recomputed here from
`numpy`/`scipy` alone, so it cannot re-derive what it checks.

The fixture is the terminal geometry of
`gaussian_location_scale_engine_matches_reference_flow` (p = 9, blocks
[2 location, 1 scale, 6 link-wiggle], A = [0 | I₆] i.e. γ ≥ 0 on the wiggle
block).  `H` carries one eigenvalue at −2.35e1 against `max|λ| = 9.40e4`.

--------------------------------------------------------------------------
1.  Is the truncated posterior proper?
--------------------------------------------------------------------------
Properness needs `dᵀHd > 0` only on the recession cone `K = {d : Ad ≥ 0}` —
copositivity — not `H ≻ 0`.  Two independent instruments agree that it holds
STRICTLY here:

  * numerically, `min{dᵀHd : d ∈ K, ‖d‖ = 1} = 7.00`;
  * and a certificate, below, that needs no optimizer at all.

--------------------------------------------------------------------------
2.  How are the moments computed without inverting `H`?
--------------------------------------------------------------------------
Split `d = Zz + Gu` with `AZ = 0`, `AG = I`, `ZᵀHG = 0` (the `H`-orthogonal
lift).  Then `dᵀHd = zᵀMz + uᵀSu` with `M = ZᵀHZ` and `S = GᵀHG`, so the two
blocks are independent and only `u = Aβ − b ≥ 0` is truncated:

    E_π[β]   = β̂ − Z M⁻¹ Zᵀ g + G (E[u] − c)
    Cov_π[β] = Z M⁻¹ Zᵀ       + G Cov[u] Gᵀ                c = A β̂ − b

Nothing here inverts `H`.  Properness is exactly `M ≻ 0` plus strict
copositivity of `S` on the orthant, and Haynsworth inertia additivity says
`S` inherits every negative eigenvalue of `H` — so the indefinite orthant
integral is unavoidable, not an artefact of the frame.

--------------------------------------------------------------------------
3.  The orthant law, and what makes it tractable
--------------------------------------------------------------------------
The `u`-law is `∝ exp(−½(u−c)ᵀS(u−c) − g_uᵀ(u−c))` on `u ≥ 0`, with `S`
indefinite.  Genz separation-of-variables needs a positive-definite
covariance, so it cannot be run on `S` directly.  The scheme validated here
keeps ONE cubature and moves the whole difficulty into a per-node weight:

    proposal   precision S₀ = S + E,  E = τ·diag(1/σ_i²),  τ ≥ 0 minimal
    location   μ₀ = c − S₀⁻¹g_u          (preserves the constrained mode)
    weight     exp(½ (u−c)ᵀ E (u−c))     (exactly cancels the inflation)

`σ_i` is the standard deviation of the one-dimensional truncated normal with
precision `S_ii`, mean `c_i − g_{u,i}/S_ii`, restricted to `[0, ∞)` — the
posterior's own per-coordinate scale, computed in closed form and WITHOUT any
tightness predicate.  `τ = max(0, −λ_min(diag(σ) S diag(σ)))·margin`, so the
inflation is measured in the metric the posterior actually lives in.

Two properties make this the right choice rather than one of many:

  * `S ⪰ 0` in that metric ⟹ `τ = 0` ⟹ `E = 0` ⟹ the weight is identically
    one and the scheme IS the existing cubature, unchanged;
  * the weight varies as `½‖(u−c)/σ‖²·τ`, i.e. `τ/2` per standardized unit of
    posterior spread — so the efficiency loss is governed by how indefinite
    `S` is *relative to the posterior*, not in absolute units.  On this
    fixture `λ_min(diag(σ)Sdiag(σ)) = −0.0174`: the indefiniteness that
    refuses the fit is a 1.7% perturbation of the standardized curvature.

--------------------------------------------------------------------------
4.  The certificate
--------------------------------------------------------------------------
`S = P + N` with `N` the entrywise-positive part of the off-diagonal and
`P = S − N`.  If `P ≻ 0` then for every `u ≥ 0`, `uᵀSu ≥ uᵀPu ≥ λ_min(P)‖u‖²`,
which is strict copositivity with an explicit modulus and no optimizer in the
loop.  Here `λ_min(P) = 12.48` against a measured copositive minimum of 12.76
— the certificate is within 2% of tight.

--------------------------------------------------------------------------
5.  The reference
--------------------------------------------------------------------------
The moments are checked against a Gibbs sampler on the same orthant law.
Every full conditional is a one-dimensional truncated normal with precision
`S_ii > 0` (itself implied by strict copositivity), so Gibbs is exact and
needs no positive-definite `S` — it shares no formula with the cubature under
test.
"""

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtri_exp
from scipy.stats import norm

SEED = 20260725

# --------------------------------------------------------------------------
# Fixture: the recorded terminal geometry of the refused fit.
# --------------------------------------------------------------------------
# H is the joint posterior precision H + S_lambda + H_Phi in the flattened
# coefficient frame; A/b the joint inequality system; mode the certified
# constrained optimum; grad the penalized gradient at the mode.
P_DIM = 9
BLOCK_WIDTHS = (2, 1, 6)


def _load_fixture(path="artifacts/issue_2442_fixture.txt"):
    """Full 9x9 fixture, kept beside this script so the numbers are auditable."""
    rows = {"H": [], "A": []}
    scalars = {}
    with open(path) as handle:
        for line in handle:
            key, _, rest = line.partition(" ")
            if key in rows:
                rows[key].append([float(v) for v in rest.split()])
            elif key in ("b", "mode", "pgrad"):
                scalars[key] = np.array([float(v) for v in rest.split()])
    return (
        np.array(rows["H"]),
        np.array(rows["A"]),
        scalars["b"],
        scalars["mode"],
        scalars["pgrad"],
    )


# --------------------------------------------------------------------------
# 1. The frame: (Z, G) with no inverse of H anywhere.
# --------------------------------------------------------------------------
def constrained_frame(h_matrix, a_matrix):
    q = a_matrix.shape[0]
    u_left, singular, v_t = np.linalg.svd(a_matrix)
    tangent = v_t[q:].T
    right_inverse = v_t[:q].T @ np.diag(1.0 / singular) @ u_left.T
    m = tangent.T @ h_matrix @ tangent
    lift = right_inverse - tangent @ np.linalg.solve(m, tangent.T @ h_matrix @ right_inverse)
    s = lift.T @ h_matrix @ lift
    return tangent, lift, m, 0.5 * (s + s.T)


# --------------------------------------------------------------------------
# 2. Copositivity: a certificate, and an independent numerical minimum.
# --------------------------------------------------------------------------
def copositivity_certificate(s):
    """`S = P + N`, `N ≥ 0` entrywise off-diagonal. `P ≻ 0` certifies strictly
    copositive with modulus `λ_min(P)`."""
    off = s - np.diag(np.diag(s))
    n = np.where(off > 0.0, off, 0.0)
    p = s - n
    return p, n, np.linalg.eigvalsh(p)[0]


def copositive_minimum(s, restarts=4000, seed=SEED):
    """min xᵀSx over the unit simplex (positive iff strictly copositive)."""
    rng = np.random.default_rng(seed)
    dim = s.shape[0]
    best = None
    for _ in range(restarts):
        start = rng.random(dim)
        start /= start.sum()
        res = minimize(
            lambda x: x @ s @ x,
            start,
            jac=lambda x: 2 * s @ x,
            bounds=[(0, None)] * dim,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1,
                          "jac": lambda x: np.ones(dim)}],
            method="SLSQP",
        )
        if res.success and (best is None or res.fun < best[0]):
            best = (res.fun, res.x)
    value, point = best
    return value / max(point @ point, np.finfo(float).tiny), point


def exact_standard_qp_minimum(s):
    """Exact min of xᵀSx on the unit sphere ∩ orthant by face enumeration.

    Every KKT point of the standard quadratic program has, on its support J,
    `S_JJ x_J ∝ 1`; enumerating supports therefore encloses the global
    minimum.  Exponential in q and used here only as an audit of the
    multistart figure above.
    """
    dim = s.shape[0]
    best = np.inf
    for size in range(1, dim + 1):
        for support in itertools.combinations(range(dim), size):
            block = s[np.ix_(support, support)]
            try:
                x = np.linalg.solve(block, np.ones(size))
            except np.linalg.LinAlgError:
                continue
            if np.any(x <= 0):
                continue
            x = x / np.linalg.norm(x)
            best = min(best, x @ block @ x)
    return best


# --------------------------------------------------------------------------
# 3. The orthant law and the two instruments.
# --------------------------------------------------------------------------
def scalar_truncated(mean, variance):
    sd = np.sqrt(variance)
    alpha = -mean / sd
    mills = np.exp(norm.logpdf(alpha) - norm.logsf(alpha))
    return mean + sd * mills, max(variance * (1.0 + alpha * mills - mills * mills), 0.0)


def marginal_scales(s, c, g_u):
    """Per-coordinate posterior scale proxy: the 1-D truncated normal with
    precision `S_ii`, mean `c_i − g_i/S_ii`, restricted to `[0, ∞)`."""
    sigma = np.empty(len(c))
    for i in range(len(c)):
        _, var = scalar_truncated(c[i] - g_u[i] / s[i, i], 1.0 / s[i, i])
        sigma[i] = np.sqrt(var)
    return sigma


def kronecker_generator(dim):
    primes, candidate = [], 2
    while len(primes) < dim:
        if all(candidate % k for k in range(2, int(candidate**0.5) + 1)):
            primes.append(candidate)
        candidate += 1
    return np.array([np.sqrt(v) % 1 for v in primes])


def cubature_moments(s, c, g_u, points, margin=2.0):
    """The scheme under test: one Genz cubature, inflated proposal, exact
    cancelling weight."""
    sigma = marginal_scales(s, c, g_u)
    standardized = np.diag(sigma) @ s @ np.diag(sigma)
    tau = max(0.0, -np.linalg.eigvalsh(standardized)[0]) * margin
    excess = tau * np.diag(1.0 / sigma**2)
    s0 = s + excess
    cov0 = np.linalg.inv(s0)
    mean0 = c - cov0 @ g_u

    factor = np.linalg.cholesky(cov0)
    dim = len(c)
    gen = kronecker_generator(dim)
    index = np.arange(points) + 0.5
    lattice = 1.0 - np.abs(((index[:, None] * gen[None, :]) % 1.0) * 2 - 1)
    z = np.zeros((points, dim))
    log_w = np.zeros(points)
    for i in range(dim):
        lower = (-mean0[i] - z[:, :i] @ factor[i, :i]) / factor[i, i]
        log_tail = log_ndtr(-lower)
        log_w += log_tail
        log_fraction = np.log(np.maximum(1.0 - lattice[:, i], np.finfo(float).tiny))
        z[:, i] = -ndtri_exp(np.minimum(log_fraction + log_tail, -1e-308))
    pts = mean0 + z @ factor.T
    centered = pts - c
    log_w = log_w + 0.5 * np.einsum("ij,jk,ik->i", centered, excess, centered)
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    mean = w @ pts
    cov = (pts * w[:, None]).T @ pts - np.outer(mean, mean)
    return mean, cov, 1.0 / np.sum(w**2), tau, sigma


def _standard_truncated_normal(lower, rng):
    if lower <= 1.0:
        base = norm.cdf(lower)
        return norm.ppf(min(base + (1.0 - base) * rng.random(), 1.0 - 1e-16))
    alpha = (lower + np.sqrt(lower * lower + 4.0)) / 2.0
    while True:
        candidate = lower + rng.exponential(1.0 / alpha)
        if rng.random() <= np.exp(-((candidate - alpha) ** 2) / 2.0):
            return candidate


def gibbs_moments(s, c, g_u, sweeps=200_000, burn=5_000, seed=SEED):
    """Independent reference. Needs `S_ii > 0` only, never `S ≻ 0`."""
    rng = np.random.default_rng(seed)
    dim = len(c)
    h_tilde = g_u - s @ c
    u = c + 0.01
    sd = 1.0 / np.sqrt(np.diag(s))
    first = np.zeros(dim)
    second = np.zeros((dim, dim))
    kept = 0
    for sweep in range(sweeps):
        for i in range(dim):
            mean = -(h_tilde[i] + s[i] @ u - s[i, i] * u[i]) / s[i, i]
            u[i] = mean + sd[i] * _standard_truncated_normal(-mean / sd[i], rng)
        if sweep >= burn:
            first += u
            second += np.outer(u, u)
            kept += 1
    mean = first / kept
    return mean, second / kept - np.outer(mean, mean)


def main():
    h, a, b, mode, grad = _load_fixture()
    q = a.shape[0]
    print(f"fixture: p={h.shape[0]} q={q} blocks={BLOCK_WIDTHS}")
    eig_h = np.linalg.eigvalsh(h)
    print(f"eig(H): min={eig_h[0]:.6e}  max={eig_h[-1]:.6e}  "
          f"negative={int((eig_h < 0).sum())}")

    tangent, lift, m, s = constrained_frame(h, a)
    print(f"\nframe: ‖AZ‖={np.abs(a @ tangent).max():.1e} "
          f"‖AG−I‖={np.abs(a @ lift - np.eye(q)).max():.1e} "
          f"‖ZᵀHG‖={np.abs(tangent.T @ h @ lift).max():.1e}")
    print(f"eig(M = ZᵀHZ): {np.linalg.eigvalsh(m)}")
    print(f"eig(S = GᵀHG): {np.linalg.eigvalsh(s)}")
    print("Haynsworth: #neg(H) =", int((eig_h < 0).sum()),
          "= #neg(M) + #neg(S) =", int((np.linalg.eigvalsh(m) < 0).sum()), "+",
          int((np.linalg.eigvalsh(s) < 0).sum()))

    p_mat, _, modulus = copositivity_certificate(s)
    measured, _ = copositive_minimum(s)
    exact = exact_standard_qp_minimum(s)
    print(f"\ncopositivity: certificate λ_min(P) = {modulus:.4f}")
    print(f"              multistart minimum   = {measured:.4f}")
    print(f"              face-enumeration min = {exact:.4f}   (strict ⟹ proper)")

    c = a @ mode - b
    g_u = lift.T @ grad
    print(f"\nslack at the mode c = {c}")

    ref_mean, ref_cov = gibbs_moments(s, c, g_u)
    ref_sd = np.sqrt(np.diag(ref_cov))
    print(f"\ngibbs   E[u] = {ref_mean}")
    print(f"gibbs   sd   = {ref_sd}")

    sigma = marginal_scales(s, c, g_u)
    standardized_min = np.linalg.eigvalsh(np.diag(sigma) @ s @ np.diag(sigma))[0]
    print(f"\nstandardized λ_min = {standardized_min:.6f}  "
          f"(the indefiniteness, measured against the posterior's own scale)")

    for points in (2**11, 2**13, 2**15, 2**17):
        mean, cov, ess, tau, _ = cubature_moments(s, c, g_u, points)
        sd = np.sqrt(np.diag(cov))
        print(f"cubature n={points:>7} τ={tau:.5f} ESS={100 * ess / points:5.1f}%  "
              f"Δmean={np.abs(mean - ref_mean).max():.2e}  "
              f"Δsd={np.abs(sd - ref_sd).max():.2e}")
    print(f"cubature E[u] = {mean}")
    print(f"cubature sd   = {sd}")

    # The reported posterior, assembled without ever inverting H.
    m_inv_zt = np.linalg.solve(m, tangent.T)
    tangent_cov = tangent @ m_inv_zt
    post_mean = mode - tangent_cov @ grad + lift @ (mean - c)
    post_cov = tangent_cov + lift @ cov @ lift.T
    print(f"\nposterior mean = {post_mean}")
    print(f"posterior sd   = {np.sqrt(np.diag(post_cov))}")
    print(f"eig(Σ_π) min   = {np.linalg.eigvalsh(post_cov)[0]:.6e}  (must be ≥ 0)")


if __name__ == "__main__":
    main()
