"""First-light figure for the curvature tomograph (plain-language captions)."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#1a1a19", "#6b6a63"
LC = [BLUE, ORANGE, AQUA, YELLOW]

d = np.load("tomograph_small.npz")
layer, t, u, ray = d["layer"], d["t"], d["u_norm"], d["ray"]

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), dpi=160)
for ax in axes:
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d8d7ce")
    ax.tick_params(colors=MUTED, labelsize=8.5)

ax = axes[0]
rays = np.unique(ray)
for li in range(4):
    m = layer == li
    ax.scatter(t[m], ray[m] + (li - 1.5) * 0.12, s=3, color=LC[li],
               label=f"layer {li}", rasterized=True)
ax.set_yticks(rays)
ax.set_xlabel("position along the ray (0 = first character, 1 = second)",
              color=MUTED, fontsize=9)
ax.set_ylabel("ray (one real 64-char context each)", color=MUTED, fontsize=9)
ax.set_title("Where the model changes its mind:\nevery ReLU flip along 6 straight paths",
             fontsize=9.5, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5, markerscale=3, loc="upper right")

ax = axes[1]
counts = [(layer == li).sum() / len(rays) for li in range(4)]
ax.bar(range(4), counts, color=[LC[i] for i in range(4)], width=0.62)
for i, c in enumerate(counts):
    ax.annotate(f"{c:.0f}", (i, c), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(4))
ax.set_xlabel("transformer layer", color=MUTED, fontsize=9)
ax.set_ylabel("decision boundaries crossed per ray", color=MUTED, fontsize=9)
ax.set_title("Which layers do the deciding", fontsize=9.5, color=INK, loc="left")

ax = axes[2]
for li in range(4):
    m = layer == li
    if m.sum() > 5:
        ax.hist(np.log10(u[m]), bins=40, histtype="step", linewidth=1.6,
                color=LC[li], label=f"layer {li}")
ax.set_xlabel("log10 ‖u‖ — how hard the flip writes into the output",
              color=MUTED, fontsize=9)
ax.set_ylabel("count", color=MUTED, fontsize=9)
ax.set_title("Most flips write softly; a heavy tail writes hard",
             fontsize=9.5, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig("issue_2502_tomograph_firstlight.png", bbox_inches="tight",
            facecolor="white")
print("saved", len(t), "atoms plotted")
