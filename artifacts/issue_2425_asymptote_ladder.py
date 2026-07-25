import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Probe ladder printed by the DECLINING asymptote-rail certificate for
# spatial_length_scale_optimization_monotone_improves_or_keeps_score_for_matern_two_feature
# coordinate k=1, at checkpoint rho=[0.00177, 12.0, 12.0], psi=-0.258.
coarse = [(11,-1.584,9.484e4),(10,-0.9326,2.054e4),(9,-0.4957,4.017e3),(8,-0.3138,9.353e2),
          (7,-0.2913,3.195e2),(6,-0.3347,1.350e2),(5,-0.3834,5.691e1),(4,-0.3961,2.163e1),
          (3,-0.3565,7.161),(2,-0.2880,2.128),(1,-0.2189,0.5950),(0,-0.1768,0.1768),
          (-1,-0.1903,0.0700),(-2,-0.2540,0.03437),(-3,-0.3454,0.01720),(-4,-0.4489,0.008223),
          (-5,-0.5572,0.003755),(-6,-0.6673,0.001654)]
local  = [(11.5,-1.919,1.895e5),(11,-1.584,9.484e4),(10.5,-1.243,4.514e4),(10,-0.9326,2.054e4),
          (9.5,-0.6797,9.080e3),(9,-0.4957,4.017e3)]

c = np.array(sorted(coarse)); l = np.array(sorted(local))
rho, g, chat = c[:,0], c[:,1], c[:,2]
lrho, lg, lchat = l[:,0], l[:,1], l[:,2]

INK="#1f2328"; MUT="#6b7280"; GRID="#e5e7eb"
BLUE="#2563eb"; RED="#dc2626"; GREEN="#059669"

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
fig.patch.set_facecolor("white")

# Panel 1: |dV/drho|
a = ax[0]
a.plot(rho, np.abs(g), color=BLUE, lw=2, marker="o", ms=5, label=r"measured $|\partial V/\partial\rho|$")
a.plot(lrho, np.abs(lg), color=BLUE, lw=0, marker="o", ms=5, mfc="white", mew=1.6)
tail = np.linspace(6, 12, 50)
a.plot(tail, 0.30*np.exp(-(tail-6)), color=GREEN, lw=2, ls="--",
       label=r"what a genuine $\lambda\!\to\!\infty$ face looks like")
a.axvline(12, color=RED, lw=1.6)
a.text(11.85, 2.6, "JOINT_RHO_BOUND = 12", rotation=90, va="top", ha="right",
       color=RED, fontsize=8.5)
a.set_yscale("log"); a.set_xlabel(r"$\rho_1=\log\lambda_1$"); a.set_ylabel(r"$|\partial V/\partial\rho_1|$")
a.set_title("The gradient GROWS toward the rail", fontsize=11, color=INK, loc="left")
a.legend(fontsize=8.5, frameon=False, loc="lower left")

# Panel 2: chat
a = ax[1]
a.plot(rho, chat, color=BLUE, lw=2, marker="o", ms=5, label=r"measured $\hat c=-e^{\rho}\,\partial V/\partial\rho$")
a.plot(lrho, lchat, color=BLUE, lw=0, marker="o", ms=5, mfc="white", mew=1.6)
a.axhline(0.30, color=GREEN, lw=2, ls="--", label=r"certificate requires $\hat c$ CONSTANT")
a.axvline(12, color=RED, lw=1.6)
a.set_yscale("log"); a.set_xlabel(r"$\rho_1=\log\lambda_1$"); a.set_ylabel(r"$\hat c$")
a.set_title(r"$\hat c$ tracks $e^{\rho}$ over 17.5 e-folds — the tail never begins",
            fontsize=11, color=INK, loc="left")
a.legend(fontsize=8.5, frameon=False, loc="upper left")

for a in ax:
    a.grid(True, color=GRID, lw=0.7); a.set_axisbelow(True)
    for s in ("top","right"): a.spines[s].set_visible(False)
    for s in ("left","bottom"): a.spines[s].set_color(MUT)
    a.tick_params(colors=MUT, labelsize=9)

fig.suptitle("#2425 — why the iso-κ joint Matérn fit can never be certified",
             fontsize=13, color=INK, x=0.008, ha="left", y=0.99)
fig.tight_layout(rect=[0,0,1,0.94])
out="/Users/user/gam/docs/images/issue2425_asymptote_ladder.png"
fig.savefig(out, dpi=170, facecolor="white")
print("wrote", out)
