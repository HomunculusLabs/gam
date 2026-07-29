"""#2502 figure: the L0 rate frontier at fixed 1.347M decoder parameters."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#1a1a19", "#6b6a63"

scalars = [4, 8, 16, 32]
flat = [0.79276, 0.84923, 0.89834, 0.96908]     # L0 = scalars
curved = [0.72916, 0.81535, 0.87459, 0.93191]   # L0 = scalars/2

fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
ax.set_facecolor("white")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#d8d7ce")
ax.grid(axis="y", color="#eceae2", linewidth=0.8, zorder=0)
ax.tick_params(colors=MUTED, labelsize=9)

ax.plot(scalars, flat, color=BLUE, linewidth=2, marker="o", markersize=6,
        zorder=3, label="flat TopK (K=10525), L0 = scalar budget")
ax.plot(scalars, curved, color=ORANGE, linewidth=2, marker="o", markersize=6,
        zorder=3, label="curved a·(U+tV) (K=5262), L0 = budget/2")
for x, y in zip(scalars, flat):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8.5, color=INK)
for x, y in zip(scalars, curved):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8.5, color=INK)

ax.set_xscale("log", base=2)
ax.set_xticks(scalars)
ax.set_xticklabels(["4", "8", "16", "32"])
ax.minorticks_off()
ax.set_xlabel("scalars per token (amplitudes + coordinates)", color=MUTED, fontsize=10)
ax.set_ylabel("held-out chart EV", color=MUTED, fontsize=10)
ax.set_title(
    "The rate frontier at fixed 1.347M decoder parameters:\n"
    "flat wins at every matched-scalar rung; the curved win is per-atom, not per-scalar",
    fontsize=10.5, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
fig.tight_layout()
fig.savefig("issue_2502_rate_frontier.png", bbox_inches="tight", facecolor="white")
print("saved")
