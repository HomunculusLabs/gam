"""#2502 figure: curved vs flat at matched decoder parameters, one eval."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#1a1a19", "#6b6a63"

params = [0.512, 1.347, 2.694]
flat8 = [0.83508, 0.84923, 0.85977]
curved8 = [0.86328, 0.87459, 0.88281]

fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
ax.set_facecolor("white")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#d8d7ce")
ax.grid(axis="y", color="#eceae2", linewidth=0.8, zorder=0)
ax.tick_params(colors=MUTED, labelsize=9)

ax.plot(params, flat8, color=BLUE, linewidth=2, marker="o", markersize=6,
        zorder=3, label="flat TopK SAE, 8 atoms/token")
ax.plot(params, curved8, color=ORANGE, linewidth=2, marker="o", markersize=6,
        zorder=3, label="curved atoms a·(U+tV), 8 atoms/token")
ax.plot([0.512, 1.347], [0.83063, 0.84543], color=AQUA, linewidth=2,
        marker="o", markersize=6, zorder=3,
        label="gated-offset B+aU, 8 atoms/token")
ax.annotate("0.8306", (0.512, 0.83063), textcoords="offset points",
            xytext=(0, -14), ha="center", fontsize=8.5, color=INK)
ax.annotate("0.8454", (1.347, 0.84543), textcoords="offset points",
            xytext=(0, -14), ha="center", fontsize=8.5, color=INK)
ax.scatter([1.347], [0.89834], color="#4a3aa7", s=52, zorder=4, marker="D",
           label="flat, 16 atoms/token")
ax.scatter([1.347], [0.81535], color=YELLOW, s=52, zorder=4, marker="s",
           label="curved, 4 atoms/token")
ax.scatter([1.6], [0.85437], color=MUTED, s=40, zorder=4, marker="x",
           label="REML/fixed-point lane (best)")

for x, y, dy in zip(params, curved8, (0.004, 0.004, 0.004)):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8.5, color=INK)
for x, y, off in zip(params, flat8, [(0, -14), (16, 2), (0, -14)]):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=off,
                ha="center" if off[0] == 0 else "left", fontsize=8.5, color=INK)
ax.annotate("0.8983", (1.347, 0.89834), textcoords="offset points",
            xytext=(8, 2), fontsize=8.5, color=INK)
ax.annotate("0.8153", (1.347, 0.81535), textcoords="offset points",
            xytext=(-42, -3), fontsize=8.5, color=INK)

ax.set_xscale("log")
ax.set_xticks(params)
ax.set_xticklabels(["0.512M", "1.347M", "2.694M"])
ax.minorticks_off()
ax.set_xlabel("decoder parameters", color=MUTED, fontsize=10)
ax.set_ylabel("held-out chart EV (one greedy+LS eval for every arm)",
              color=MUTED, fontsize=10)
ax.set_title(
    "The curve is the active ingredient: curved > flat > gated-offset at matched decoder params\n"
    "Qwen3.5-4B L16 residual chart, doc split, identical Gao-et-al recipe; "
    "seeds agree to ±0.0004",
    fontsize=10.5, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK, bbox_to_anchor=(0.02, 0.97))
fig.tight_layout()
fig.savefig("issue_2502_curved_parity.png", bbox_inches="tight",
            facecolor="white")
print("saved")
