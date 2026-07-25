"""#2366: why the profiled REML criterion was not a function of rho.

Reproduces the landed gate fixture in closed form: the tilted double well
   w(b) = (b^2 - 1)^2 + c*b,     c = 0.3
penalized by 0.5 * lambda * b^2, lambda = exp(rho).

Left  : the penalized objective at three rho, with both modes marked.
Right : mode-vs-rho. The anchor at the rho-box ceiling has ONE mode; following
        it down is the continuation. A caller's seed on the far side of the
        barrier lands on the other branch, and that is the branch the outer
        search was silently descending before the fix.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

TILT = 0.3
CEILING = 12.0   # EFFECTIVE_DF_CEILING: the anchor
FLOOR = -10.0    # rho_lower_bound

def pen_obj(b, rho):
    return (b**2 - 1.0)**2 + TILT * b + 0.5 * np.exp(rho) * b**2

def local_mode(rho, start):
    """Newton-ish descent to the local mode from `start` (the inner solve)."""
    b = float(start)
    for _ in range(400):
        g = 4.0 * b**3 - 4.0 * b + TILT + np.exp(rho) * b
        h = 12.0 * b**2 - 4.0 + np.exp(rho)
        step = g / h if h > 1e-9 else g          # ridge when indefinite
        b_new = b - np.clip(step, -0.5, 0.5)
        if abs(b_new - b) < 1e-14:
            return b_new
        b = b_new
    return b

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ---- left: the objective at three smoothing levels -----------------------
bs = np.linspace(-1.6, 1.6, 900)
# each curve shifted to its own minimum: the SHAPE is the point, and the
# lambda=e^4 curve would otherwise dominate the scale by two orders.
for rho, colour, lab in [(4.0, "#4C6EF5", r"$\rho=4$  (anchor side: one mode)"),
                         (1.2, "#7048E8", r"$\rho=1.2$  (barrier appearing)"),
                         (-6.0, "#E8590C", r"$\rho=-6$  (two modes)")]:
    v = pen_obj(bs, rho)
    ax0.plot(bs, v - v.min(), color=colour, lw=2.2, label=lab)
ax0.set_ylim(-0.05, 1.5)
deep = local_mode(-6.0, -2.0)
shallow = local_mode(-6.0, 2.0)
v6 = pen_obj(bs, -6.0); base = v6.min()
ax0.plot([deep], [pen_obj(deep, -6.0) - base], "o", ms=11, color="#2B8A3E", zorder=5,
         label=f"deep mode  $\\beta$={deep:.3f}")
ax0.plot([shallow], [pen_obj(shallow, -6.0) - base], "s", ms=10, color="#C92A2A", zorder=5,
         label=f"shallow mode  $\\beta$={shallow:.3f}  (+{pen_obj(shallow,-6.0)-pen_obj(deep,-6.0):.3f})")
ax0.set_xlabel(r"coefficient $\beta$")
ax0.set_ylabel(r"$\ell_p(\beta,\rho)\;-\;\min_\beta \ell_p$")
ax0.set_title("The inner problem has TWO modes at one $\\rho$", fontsize=12)
ax0.legend(fontsize=8.5, loc="upper center", framealpha=0.95)
ax0.grid(alpha=0.25)

# ---- right: branch diagram ----------------------------------------------
rhos = np.linspace(CEILING, FLOOR, 400)

# continuation: start at the anchor, carry the mode down
cont = []
b = local_mode(CEILING, 0.0)
for r in rhos:
    b = local_mode(r, b)
    cont.append(b)

# cold-direct from two arbitrary caller seeds
cold_pos = [local_mode(r, 2.0) for r in rhos]
cold_neg = [local_mode(r, -2.0) for r in rhos]

ax1.plot(rhos, cont, color="#2B8A3E", lw=3.0, zorder=4,
         label="anchored continuation (the definition)")
ax1.plot(rhos, cold_pos, color="#C92A2A", lw=2.0, ls="--",
         label=r"cold-direct from caller seed $\beta_0=+2$")
ax1.plot(rhos, np.array(cold_neg) + 0.035, color="#F08C00", lw=2.0, ls=":",
         label=r"cold-direct from $\beta_0=-2$ (offset +0.035; it coincides)")
ax1.axvline(CEILING, color="#495057", lw=1.2)
ax1.annotate("anchor $\\rho_A$\n(term on its penalty nullspace:\nthe mode is UNIQUE here)",
             xy=(CEILING - 0.3, 0.02), xytext=(10.6, 0.55), fontsize=9,
             ha="left", va="center", color="#495057",
             arrowprops=dict(arrowstyle="->", color="#495057", lw=1.1))
ax1.annotate("the branch point:\nthe two seeds part company",
             xy=(0.9, 0.0), xytext=(4.2, -0.72), fontsize=9, ha="left",
             color="#495057",
             arrowprops=dict(arrowstyle="->", color="#495057", lw=1.1))
ax1.axhline(0.0, color="#adb5bd", lw=0.8)
ax1.set_xlim(CEILING, FLOOR)
ax1.set_xlabel(r"$\rho = \log\lambda$   (anchor on the left, unpenalized on the right)")
ax1.set_ylabel(r"selected mode  $\hat\beta(\rho)$")
ax1.set_title("Which mode you get depended on the seed, not on $\\rho$", fontsize=12)
ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
ax1.grid(alpha=0.25)

fig.suptitle("#2366  $V(\\rho)=\\ell_p(\\hat\\theta(\\rho),\\rho)$ is a function of $\\rho$ only once a mode-selection rule is fixed",
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = "/private/tmp/claude-501/-Users-user-gam/05d0692f-b15d-4ac0-87e3-8351b744c410/scratchpad/issue2366_branches.png"
fig.savefig(out, dpi=155)
print("wrote", out)
print(f"at rho=-6: deep={deep:.6f} (obj {pen_obj(deep,-6.0):.6f}), "
      f"shallow={shallow:.6f} (obj {pen_obj(shallow,-6.0):.6f}), "
      f"gap={pen_obj(shallow,-6.0)-pen_obj(deep,-6.0):.6f}")
