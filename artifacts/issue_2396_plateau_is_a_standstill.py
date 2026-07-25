"""#2396: the sparse_dict fixed-point contract never closed because it was
asking the wrong question.

Every number here is MEASURED in CI, not modelled.

Left  : the production budget->residual trajectory of the inner alternation,
        from `zz_measure_2396_open_arm_budget_ev_trace` (gam-sae, focused-rust-
        proof run 30141422908, K=64 s=2 N=400 P=16). Sweeping the epoch budget
        enumerates the trajectory: each budget too short to confirm a plateau
        reports, in its typed non-convergence, the residuals it had reached.
        The EV residual decays smoothly and stalls 17x above the 1e-9 target;
        the ROUTING residual jumps five orders of magnitude at scattered rounds
        as single rows flip between near-equivalent atoms. That is a LIMIT
        CYCLE, not a slow solve, and no budget or tolerance closes it.

Right : the three REML-schedule fits whose admission depended on the defect.
        Measured at budget exhaustion in run 30139426473, the run that landed
        the strict repair (|dEV| against the achieved climb) and returned
        100 passed / 4 failed. Under the OLD rule these were admitted, but only
        through its downhill branch: it scored each round's UPWARD share
        max(dEV, 0) / (next_ev - entry_ev), which is identically zero for any
        round that moved down and whose denominator is <= 0 for any round below
        the entry EV. A monotonically degrading fit therefore reported a plateau
        every round. The landed rule certifies the running MAXIMUM instead and
        returns the iterate attaining it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOL = 1.0e-9

# --- measured: focused-rust-proof run 30141422908 ---------------------------
# budget, EV, EV residual, decoder fixed-point residual, routing residual
TRACE = [
    (2,  0.999962686852, 2.982571e-6, 9.001671e-5, 1.196960e-5),
    (3,  0.999963820543, 1.133692e-6, 4.448199e-5, 3.471203e-2),
    (4,  0.999964312755, 4.922117e-7, 2.001720e-5, 2.301250e-2),
    (5,  0.999964539363, 2.266080e-7, 8.406530e-6, 2.312827e-2),
    (6,  0.999964654449, 1.150859e-7, 3.375675e-6, 6.682908e-7),
    (7,  0.999964721863, 6.741351e-8, 1.318580e-6, 4.529481e-7),
    (8,  0.999964767386, 4.552290e-8, 9.173518e-7, 5.335155e-7),
    (9,  0.999964801988, 3.460251e-8, 7.787434e-7, 2.737675e-7),
    (10, 0.999964829947, 2.795939e-8, 6.588777e-7, 2.502223e-2),
    (11, 0.999964853327, 2.337980e-8, 5.557389e-7, 1.912854e-7),
    (12, 0.999964873226, 1.989880e-8, 4.674100e-7, 1.627196e-7),
    (13, 0.999964873226, 1.710689e-8, 3.920864e-7, 1.392723e-7),  # returned, open
]
RETURNED_AT = 13

# --- measured: focused-rust-proof run 30139426473 ---------------------------
# name, epochs, EV, EV residual, decoder residual, routing residual
FITS = [
    ("shared_rho_fixed_point",       80, 0.930716, 3.5540e-4, 2.2706e-2, 1.8046e-2),
    ("reml_schedule_held_out_ev",    60, 0.877095, 4.4795e-4, 5.8600e-3, 2.0993e-3),
    ("reml_noise_floored_interior",  80, 0.845150, 1.5162e-4, 3.8334e-4, 6.1454e-4),
]

EV_C, ROUTE_C, DEC_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#2b2b2b", "#7d7c76"

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.0, 5.8))

# ---- left: the measured trajectory ----------------------------------------
b = np.array([r[0] for r in TRACE])
ev_r = np.array([r[2] for r in TRACE])
dec_r = np.array([r[3] for r in TRACE])
route_r = np.array([r[4] for r in TRACE])

ax0.semilogy(b, ev_r, "-o", color=EV_C, lw=2.2, ms=6, label="EV residual  $|\\Delta EV|$")
ax0.semilogy(b, route_r, "-o", color=ROUTE_C, lw=2.2, ms=6, label="routing residual")
ax0.semilogy(b, dec_r, "-o", color=DEC_C, lw=2.2, ms=6, label="decoder residual  $\\sin^2\\theta$")
ax0.axhline(TOL, color=MUTED, lw=2.0, ls="--")

ax0.axvline(RETURNED_AT, color=MUTED, lw=1.0, ls=":")
# Direct end labels rather than a legend callout, so nothing sits on a series.
for value, colour in ((ev_r[-1], EV_C), (route_r[-1], ROUTE_C), (dec_r[-1], DEC_C)):
    ax0.text(13.25, value, f"{value:.2e}", fontsize=8.6, color=colour,
             ha="left", va="center")
ax0.text(13.25, TOL * 1.7, "tolerance 1e-9", fontsize=8.6, color=MUTED, ha="left", va="center")

# Annotations live in the empty upper-right quadrant; neither crosses a series.
ax0.annotate("the routing residual jumps FIVE decades and comes back:\n"
             "single rows flipping between two near-equivalent atoms.\n"
             "No budget and no tolerance closes that.",
             xy=(10.0, 2.502223e-2), xytext=(5.9, 1.3e-1), fontsize=8.8,
             ha="left", va="center", color=INK,
             arrowprops=dict(arrowstyle="->", color=ROUTE_C, lw=1.3))
ax0.annotate("returned: open certificate,\nbirths 0, support saturated",
             xy=(13.0, 1.392723e-7), xytext=(10.1, 5.5e-6), fontsize=8.8,
             ha="center", va="center", color=INK,
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))

ax0.set_xlim(1.7, 15.6)
ax0.set_ylim(4e-10, 8e-1)
ax0.set_xticks(b)
ax0.set_xlabel("epoch budget  (each point is that budget's typed non-convergence evidence)")
ax0.set_ylabel("fixed-point residual")
ax0.set_title("The alternation is a LIMIT CYCLE, not a slow solve", fontsize=12)
ax0.legend(fontsize=9, loc="lower left", framealpha=0.95)
ax0.grid(alpha=0.25, which="both")

# ---- right: the fits that depended on the defect ---------------------------
ys = np.arange(len(FITS))[::-1]
for y, (name, epochs, ev, e_r, d_r, r_r) in zip(ys, FITS):
    lo, hi = min(e_r, d_r, r_r), max(e_r, d_r, r_r)
    ax1.plot([lo, hi], [y, y], color="#d8d8d3", lw=7, solid_capstyle="round", zorder=1)
    # Draw the widest-separated last so a coincident pair still reads.
    for value, colour in sorted(((e_r, EV_C), (r_r, ROUTE_C), (d_r, DEC_C)),
                                key=lambda t: -t[0]):
        ax1.plot([value], [y], "o", ms=13, color=colour, mec="white", mew=1.6, zorder=3)
    # Label the two ends of the row. When they are closer than a third of a
    # decade the labels would collide, so drop the far one below the row.
    far, far_c = (r_r, ROUTE_C) if r_r >= d_r else (d_r, DEC_C)
    ax1.annotate(f"{e_r:.2e}", xy=(e_r, y), xytext=(0, 16), textcoords="offset points",
                 fontsize=8.2, color=EV_C, ha="center")
    apart = abs(np.log10(far) - np.log10(e_r))
    ax1.annotate(f"{far:.2e}", xy=(far, y),
                 xytext=(0, 16 if apart > 1.0 else -24), textcoords="offset points",
                 fontsize=8.2, color=far_c, ha="center")

ax1.axvline(TOL, color=MUTED, lw=2.0, ls="--")
ax1.text(TOL * 1.25, len(FITS) - 0.35, "tolerance 1e-9", fontsize=9, color=MUTED,
         ha="left", va="center")
ax1.set_xscale("log")
ax1.set_xlim(6e-10, 2e-1)
ax1.set_ylim(-0.55, len(FITS) - 0.25)
ax1.set_yticks(ys)
ax1.set_yticklabels([f"{name}\n{epochs} epochs · EV {ev:.6f}" for name, epochs, ev, *_ in FITS],
                    fontsize=9)
for lab in ax1.get_yticklabels():
    lab.set_color(INK)
ax1.tick_params(axis="y", length=0)
ax1.set_xlabel("residual at budget exhaustion")
ax1.set_title("Five to seven decades from the target, and admitted anyway", fontsize=12)
ax1.grid(alpha=0.25, axis="x", which="both")

handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c, label=l)
           for c, l in ((EV_C, "EV residual"), (ROUTE_C, "routing residual"),
                        (DEC_C, "decoder residual"))]
ax1.legend(handles=handles, fontsize=9, loc="lower left", framealpha=0.95)

fig.suptitle("#2396   a plateau is a STANDSTILL, not a climb "
             "— and certifying a running maximum obliges returning the iterate that attains it",
             fontsize=13)
fig.text(0.5, 0.015,
         "OLD rule admitted the three fits on the right through its downhill branch: it scored "
         "max($\\Delta$EV, 0) / (ev $-$ entry), which is 0 for any round that moved DOWN and whose "
         "denominator is $\\leq$ 0 below the entry EV,\nso a monotonically degrading fit reported a plateau "
         "every round.   Measured: replacing it with $|\\Delta$EV$|$ against the achieved climb turned exactly "
         "these three red (100 passed / 4 failed).\nThe LANDED rule certifies the running MAXIMUM and returns "
         "the iterate that attains it — 107 passed / 0 failed.   All values measured in CI; none modelled.",
         fontsize=8.8, color=INK, ha="center", va="bottom", linespacing=1.6)
fig.tight_layout(rect=(0, 0.115, 1, 0.945))
out = "artifacts/issue_2396_plateau_is_a_standstill.png"
fig.savefig(out, dpi=155)
print("wrote", out)
print(f"EV residual stalls at {ev_r[-1]:.3e} = {ev_r[-1]/TOL:.1f}x the 1e-9 target")
print(f"routing residual range over the SAME trajectory: "
      f"{route_r.min():.3e} .. {route_r.max():.3e} "
      f"({np.log10(route_r.max()/route_r.min()):.1f} decades)")
