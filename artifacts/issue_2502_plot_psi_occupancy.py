"""Curvature-atoms section-5 test: loss is smooth through 90 deg; occupancy is not."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#1a1a19", "#6b6a63"

rows = sorted((json.loads(l) for l in open("psi_sweep_b0occ.log") if l.startswith("{")),
              key=lambda r: r["psi"])
psi = [r["psi"] for r in rows]
lt = [r["loss_triple"] for r in rows]
occ = [r["occupancy_triple"] for r in rows]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.6), dpi=160, sharex=True)
for ax in (ax1, ax2):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d8d7ce")
    ax.grid(axis="y", color="#eceae2", linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.axvline(90.0, color="#c9c7bc", linewidth=1.2, linestyle="--", zorder=1)

ax1.plot(psi, lt, color=BLUE, linewidth=2, marker="o", markersize=4, zorder=3)
ax1.set_ylabel("triple-active loss (frontier, b=0)", color=MUTED, fontsize=9.5)
ax1.set_title(
    "The half-space law reads out in OCCUPANCY, not loss (b=0 frontier, 3 features at ±ψ)\n"
    "loss is C¹-smooth through 90°; the triple co-activation pattern dies exactly there",
    fontsize=10.5, color=INK, loc="left")
ax1.annotate("smooth through 90°", (90, lt[len(lt) // 2]), textcoords="offset points",
             xytext=(10, 22), fontsize=9, color=INK)

ax2.plot(psi, occ, color=ORANGE, linewidth=2, marker="o", markersize=4, zorder=3)
ax2.set_ylabel("occupancy: frontier decode fires\nall 3 units on triple-active rows",
               color=MUTED, fontsize=9.5)
ax2.set_xlabel("ψ (degrees) — features at −ψ, 0, +ψ", color=MUTED, fontsize=10)
ax2.annotate("exactly 0 for ψ ≥ 90°", (91, 0.0), textcoords="offset points",
             xytext=(8, 12), fontsize=9, color=INK)
fig.tight_layout()
fig.savefig("issue_2502_psi_occupancy.png", bbox_inches="tight", facecolor="white")
print("saved")
