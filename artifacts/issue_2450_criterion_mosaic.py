"""#2450 — the shipped criterion is a mosaic, measured.

Panel A  paired displacement of rho-hat, MAP (Normal{0,3}) minus REML (Flat),
         with paired standard errors. `ps` is identically zero to the bit
         because the driver overrides the caller's prior; `matern` is not.
Panel B  paired MISE difference against the noise-free truth, same pairing.
Panel C  why it matters for the rail certificate: c-hat = -e^rho dV/drho,
         which every rail path tests for CONSTANCY. Under REML it is a
         constant; under Normal{0,3} the measured dV/drho -> rho/9 makes it
         diverge, so no lambda=infinity face exists to certify at any box width.

All numbers from `zz_measure_2450_rho_prior_criterion_bias_ladder`
(gam-models --lib, e22a01f9f, A10, 8 paired replicates per cell, n=200,
sigma=0.2, truth = sin(2*pi*f*x)); panel C's dV/drho values are the ladder
already recorded on the issue.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PS = [  # freq, d_rho, se, d_mise, se_mise
    (0.5, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0, 0.0, 0.0),
]
MATERN = [
    (0.5, +0.5744, 0.0576, -1.9829e-5, 2.0582e-5),
    (1.0, -0.0277, 0.0111, +1.9348e-5, 3.7440e-6),
    (2.0, +0.1933, 0.0099, +1.7021e-5, 7.9959e-6),
]

BLUE = "#3b6ea5"
RED = "#b5442f"
GREY = "#8a8a8a"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

# ---------------------------------------------------------------- panel A
ax = axes[0]
w = 0.34
xs_ps = np.arange(len(PS))
xs_mt = np.arange(len(MATERN)) + len(PS) + 0.6
ax.bar(
    xs_ps,
    [r[1] for r in PS],
    w * 2,
    yerr=[r[2] for r in PS],
    color=GREY,
    label="ps (BSpline1D) — prior overridden",
    capsize=3,
)
ax.bar(
    xs_mt,
    [r[1] for r in MATERN],
    w * 2,
    yerr=[r[2] for r in MATERN],
    color=RED,
    label="matern — prior survives",
    capsize=3,
)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(list(xs_ps) + list(xs_mt))
ax.set_xticklabels([f"{r[0]:g}" for r in PS] + [f"{r[0]:g}" for r in MATERN])
ax.set_xlabel("truth frequency  (sin 2πfx)")
ax.set_ylabel(r"$\hat\rho_{\rm MAP}-\hat\rho_{\rm REML}$  (log $\lambda$)")
ax.set_title(
    "A. paired displacement of the selected $\\log\\lambda$\n"
    "ps: 0.0000 to the BIT, even under an absurd prior",
    fontsize=10.5,
)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.25, axis="y")

# ---------------------------------------------------------------- panel B
ax = axes[1]
ax.bar(
    xs_ps,
    [r[3] for r in PS],
    w * 2,
    yerr=[r[4] for r in PS],
    color=GREY,
    capsize=3,
)
ax.bar(
    xs_mt,
    [r[3] for r in MATERN],
    w * 2,
    yerr=[r[4] for r in MATERN],
    color=RED,
    capsize=3,
)
for x, r in zip(xs_mt, MATERN):
    t = r[3] / r[4]
    ax.annotate(
        f"t={t:+.1f}",
        (x, r[3] + np.sign(r[3]) * (r[4] + 3e-6)),
        ha="center",
        fontsize=8.5,
        color="k",
    )
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(list(xs_ps) + list(xs_mt))
ax.set_xticklabels([f"{r[0]:g}" for r in PS] + [f"{r[0]:g}" for r in MATERN])
ax.set_xlabel("truth frequency")
ax.set_ylabel(r"MISE$_{\rm MAP}$ $-$ MISE$_{\rm REML}$")
ax.set_title(
    "B. paired truth-recovery cost of the MAP criterion\n"
    "above zero = the shipped default fits worse",
    fontsize=10.5,
)
ax.grid(alpha=0.25, axis="y")

# ---------------------------------------------------------------- panel C
ax = axes[2]
rho = np.linspace(6, 16, 400)
# REML: dV/drho = -c*exp(-rho)  =>  c-hat = c, constant. Take c = 1 (the scale
# is irrelevant; constancy is what the certificate tests).
chat_reml = np.ones_like(rho)
# Normal{0, 3}: the criterion carries +rho^2/18, so dV/drho -> rho/9 and
# c-hat = -e^rho * rho/9.
chat_map = np.abs(-np.exp(rho) * rho / 9.0)
ax.semilogy(rho, chat_reml, color=BLUE, lw=2.2, label=r"REML (Flat): $\hat c$ constant")
ax.semilogy(
    rho,
    chat_map,
    color=RED,
    lw=2.2,
    label=r"REML + $\rho^2/18$: $|\hat c| = e^{\rho}\rho/9$",
)
for r0, g in [(9.0, 1.0108), (12.0, 1.3339), (15.0, 1.6667)]:
    ax.plot(r0, np.exp(r0) * g, "o", color="k", ms=5, zorder=5)
ax.annotate(
    "measured $\\partial V/\\partial\\rho$\n(1.0108, 1.3339, 1.6667\nvs $\\rho/9$ = 1.0000, 1.3333, 1.6667)",
    xy=(12.0, np.exp(12.0) * 1.3339),
    xytext=(6.6, 1e5),
    fontsize=8.5,
    arrowprops=dict(arrowstyle="->", lw=0.9),
)
ax.axvline(12.0, color=GREY, ls=":", lw=1.2)
ax.text(12.1, 2e-1, "JOINT_RHO_BOUND", rotation=90, fontsize=8, color=GREY, va="bottom")
ax.set_xlabel(r"$\rho = \log\lambda$")
ax.set_ylabel(r"$|\hat c| = |{-}e^{\rho}\,\partial V/\partial\rho|$")
ax.set_title(
    "C. the law every rail path decides by\n"
    "constant under REML; unbounded under the shipped prior",
    fontsize=10.5,
)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.25, which="both")

fig.suptitle(
    "#2450  the shipped deterministic criterion is not one criterion  —  paired A/B/C at e22a01f9f, gam-models --lib",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = "/private/tmp/claude-501/-Users-user-gam/9d7b5e0c-3ed5-40db-938b-d889da5b5056/scratchpad/issue_2450_criterion_mosaic.png"
fig.savefig(out, dpi=150)
print(out)
