"""#2390: does the regression gate actually discriminate the bug it was written for?

Independent check of the gate added in `99c3b053a`
(`near_wall_wiggle_coordinate_keeps_cross_covariance_in_moments_2390`). It uses
NO gam code, so it cannot re-derive the thing it is testing — it re-implements
the locscale response-moment integral's conditional-mean structure in numpy and
measures the quantity the gate asserts on.

THE GATE. Negating the whole link-wiggle <-> (h, threshold, log sigma) cross
block is the congruence `S' = D S D`, `D = diag(I, -I)`. So `S'` is PSD, its
(h, threshold, log sigma) projection is unchanged, and the conditional
covariance `cov_ww - R R^T` is unchanged (`R -> -R`). The ONLY thing that
changes is the SIGN of the conditional-mean displacement — the correlation
between the realized warp and the realized predictor. If the displacement has
been frozen, both covariances give the same moments and the gap collapses.

WHAT IS MEASURED. That gap, under both clip rules, with COMMON RANDOM NUMBERS
(the same latent draws for `S` and `S'`, so the difference is signal and not
Monte Carlo noise — with independent streams the buggy gap is buried under a
~5e-5 standard error and the comparison looks falsely alive):

  * "per-coordinate" — the landed rule. alpha_j = min(1, max(bhat_j,0)/|d_j|),
    applied coordinatewise, because the cone `A = I, b = 0` is a Cartesian
    product of independent half-lines.
  * "global"         — the rule it replaced. One alpha = min_j over all
    coordinates, the fraction-to-boundary rule for a step along ONE RAY of a
    coupled polytope (correct for #2375's cubature nodes, wrong here).

Two fixtures. The near-wall one is the gate's own: coordinate 1 at 1e-9, just
outside the 1e-10 active-face tightness band, so it is SLACK and keeps a
full-width covariance row while its `max(bhat,0)/|d|` ratio is ~1e-8. The
interior control has no coordinate near a wall, where both rules must agree —
without it, a difference between the rules would not be evidence that the
NEAR-WALL regime is what the gate catches.

Run: uv run --no-project --with numpy --with matplotlib python <this file>
"""

import numpy as np

# --- the fixture, verbatim from the gate ------------------------------------
G = np.array([
    [0.18, 0.00, 0.00, 0.00],
    [0.05, 0.16, 0.00, 0.00],
    [0.00, 0.07, 0.15, 0.00],
    [0.04, 0.00, 0.13, 0.00],
    [0.00, 0.06, 0.00, 0.14],
    [0.03, 0.00, 0.05, 0.12],
    [0.12, 0.10, 0.09, 0.08],
    [0.09, 0.11, 0.07, 0.10],
])
SIGMA = G @ G.T
FLIP = np.diag([1.0, 1, 1, 1, 1, 1, -1, -1])
SIGMA_FLIPPED = FLIP @ SIGMA @ FLIP

BETA_TIME = np.array([0.4, -0.1])
BETA_THRESHOLD = np.array([0.2, 0.3])
BETA_LOG_SIGMA = np.array([-0.5, 0.1])
A_H = np.array([1.0, 0.5])
A_T = np.array([1.0, -0.2])
A_LS = np.array([1.0, 0.3])
MU = np.array([
    A_H @ BETA_TIME + 0.2,
    A_T @ BETA_THRESHOLD + 0.7,
    A_LS @ BETA_LOG_SIGMA + 0.4,
])
GATE = 1.0e-8


def q0_from_eta(eta_t, eta_ls):
    return -eta_t * np.exp(-eta_ls)


# Two staggered non-negative cumulative columns standing in for the I-spline
# value basis, recentred on the nominal q0 exactly as the gate recentres its
# knots. Only non-negativity and the staggering matter for what is measured.
_CENTRE = q0_from_eta(MU[1], MU[2])
_LO, _HI = _CENTRE - 2.0, _CENTRE + 2.0


def basis(x):
    u = np.clip((x - _LO) / (_HI - _LO), 0.0, 1.0)
    return np.stack([np.clip(u / 0.7, 0, 1), np.clip((u - 0.3) / 0.7, 0, 1)], axis=-1)


def conditional_parts(sigma):
    """Reproduce the production decomposition: project to (h, t, ls), regress
    the wiggle block on it, and form the conditional covariance."""
    a = np.zeros((3, 8))
    a[0, [0, 1]] = A_H
    a[1, [2, 3]] = A_T
    a[2, [4, 5]] = A_LS
    cov3 = a @ sigma @ a.T
    cov_wy = sigma[np.ix_([6, 7], range(8))] @ a.T
    ev, vec = np.linalg.eigh(cov3)
    keep = ev > 1e-14 * ev.max()
    factor = vec[:, keep] * np.sqrt(ev[keep])
    regression = cov_wy @ vec[:, keep] / np.sqrt(ev[keep])
    cond = sigma[np.ix_([6, 7], [6, 7])] - regression @ regression.T
    return factor, regression, 0.5 * (cond + cond.T)


def sign_flip_gap(beta_w, rule, n=8_000_000, seed=7):
    rng = np.random.default_rng(seed)
    factor, reg, cond = conditional_parts(SIGMA)
    factor_f, reg_f, cond_f = conditional_parts(SIGMA_FLIPPED)
    # The congruence's whole point: everything except the displacement is fixed.
    assert np.allclose(factor, factor_f) and np.allclose(cond, cond_f)

    latent = rng.standard_normal((n, factor.shape[1]))
    noise = rng.standard_normal(n)
    x = MU + latent @ factor.T
    q0 = q0_from_eta(x[:, 1], x[:, 2])
    b = basis(q0)
    sd = np.sqrt(np.einsum("ij,jk,ik->i", b, cond, b).clip(0.0))
    wall = np.maximum(beta_w, 0.0)

    moments = []
    for regression in (reg, reg_f):
        d = latent @ regression.T
        if rule == "per-coordinate":
            clipped = np.where(
                np.abs(d) > wall, np.copysign(np.broadcast_to(wall, d.shape), d), d
            )
        else:
            alpha = np.minimum(
                1.0, wall / np.maximum(np.abs(d), 1e-300)
            ).min(axis=1, keepdims=True)
            clipped = alpha * d
        # `[0, inf)` feasible image of the cone under a non-negative basis row.
        w = np.maximum((b * (beta_w + clipped)).sum(axis=1) + sd * noise, 0.0)
        surv = 0.5 * (1.0 - np.tanh((x[:, 0] + q0 + w) / 1.7))
        moments.append((surv.mean(), (surv**2).mean()))
    (m1, s1), (m2, s2) = moments
    return abs(m1 - m2), abs(s1 - s2)


FIXTURES = [
    ("near-wall\nbeta_w = [0.30, 1e-9]", np.array([0.30, 1.0e-9])),
    ("interior control\nbeta_w = [0.30, 0.25]", np.array([0.30, 0.25])),
]
RULES = ["per-coordinate", "global"]

results = {}
for label, beta_w in FIXTURES:
    for rule in RULES:
        results[(label, rule)] = sign_flip_gap(beta_w, rule)
        print(f"{label.splitlines()[0]:18s} {rule:14s} "
              f"E[S] gap={results[(label, rule)][0]:.3e}  "
              f"E[S^2] gap={results[(label, rule)][1]:.3e}")

# --- figure -----------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
SERIES = {"per-coordinate": "#2a78d6", "global": "#eb6834"}

fig, ax = plt.subplots(figsize=(9.6, 4.4))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

labels = [f[0] for f in FIXTURES]
y_base = np.arange(len(labels))[::-1] * 1.0
height = 0.2

for offset, rule in zip((height / 2 + 0.03, -height / 2 - 0.03), RULES):
    values = [max(results[(lab, rule)][0], 1e-13) for lab in labels]
    ax.barh(y_base + offset, values, height=height, color=SERIES[rule],
            label=rule, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    for y, v in zip(y_base + offset, values):
        ax.text(v * 1.4, y, f"{v:.2e}", va="center", ha="left",
                fontsize=9, color=INK_2, zorder=4)

ax.axvline(GATE, color=MUTED, lw=1.6, ls=(0, (5, 3)), zorder=2)
ax.text(GATE * 1.6, np.mean(y_base), "gate\n1e-8", fontsize=8.6,
        color=MUTED, ha="left", va="center", linespacing=1.3)

ax.set_xscale("log")
ax.set_xlim(1e-13, 3e-1)
ax.set_ylim(y_base[-1] - 0.5, y_base[0] + 0.5)
ax.set_yticks(y_base)
ax.set_yticklabels(labels, fontsize=9.5, color=INK)
ax.set_xlabel("|E[S] under $\\Sigma$  −  E[S] under $D\\Sigma D$|   "
              "(cross-covariance signal retained)", fontsize=9.5, color=INK_2)
ax.set_title("#2390: the per-coordinate cone clip keeps the cross-covariance "
             "a global clip erases",
             fontsize=12, color=INK, pad=34, loc="left")

ax.grid(axis="x", color="#e6e5e0", lw=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#d8d7d1")
ax.tick_params(axis="x", colors=INK_2, labelsize=9)
ax.tick_params(axis="y", length=0)

leg = ax.legend(title="clip rule", frameon=False, fontsize=9.5, ncol=2,
                loc="lower left", bbox_to_anchor=(0.0, 1.005),
                title_fontsize=9.5, handlelength=1.1, handleheight=0.9,
                columnspacing=1.6, borderpad=0.0)
leg.get_title().set_color(INK_2)
for text in leg.get_texts():
    text.set_color(INK_2)

fig.text(0.012, -0.015,
         "8M common-random-number draws per bar. Near-wall: the landed rule "
         "retains 1.3e-3 of signal, the global rule 4.4e-12 — the gate sits ~4 "
         "orders above the\nbug and ~5 below the fix. Interior control: both "
         "rules agree, so the discrimination is specific to the near-wall "
         "regime. No gam code is used.",
         fontsize=8.2, color=MUTED, ha="left", va="top")

fig.tight_layout()
fig.savefig("artifacts/issue_2390_cone_clip_discrimination.png", dpi=170,
            facecolor=SURFACE, bbox_inches="tight")
print("wrote artifacts/issue_2390_cone_clip_discrimination.png")
