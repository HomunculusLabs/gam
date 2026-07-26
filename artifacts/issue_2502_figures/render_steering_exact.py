"""A literal, uncluttered comparison of flat and manifold steering."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


OUT = "/private/tmp/claude-scratch-2502/results/steering_exact.png"
INK = "#070b16"
TEXT = "#eaf0ff"
RED = "#ff6b78"
GREEN = "#5fe0ad"
CMAP = plt.get_cmap("twilight_shifted")

fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=200)
fig.patch.set_facecolor(INK)
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.07, top=0.90, wspace=0.08)

rx, ry = 2.10, 1.58
theta = np.linspace(0, 2 * np.pi, 720)
day_theta = np.pi / 2 - np.arange(7) * 2 * np.pi / 7
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
tue_i, wed_i = 1, 2


def xy(angle: float, radius: float = 1.0) -> np.ndarray:
    return np.array([radius * rx * np.cos(angle), radius * ry * np.sin(angle)])


def arrow(ax, start, end, color, width=3.4, mutation=24, style="-|>", z=9):
    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle=style,
            mutation_scale=mutation,
            linewidth=width,
            color=color,
            shrinkA=7,
            shrinkB=7,
            zorder=z,
        )
    )


for ax in axes:
    ax.set_facecolor(INK)
    ax.set_aspect("equal")
    ax.set_xlim(-3.6, 4.3)
    ax.set_ylim(-2.65, 2.55)
    ax.axis("off")
    loop = np.column_stack([rx * np.cos(theta), ry * np.sin(theta)])
    for width, alpha in ((30, 0.035), (18, 0.07), (9, 0.11)):
        ax.plot(loop[:, 0], loop[:, 1], color="#6179bf", lw=width, alpha=alpha)
    for i in range(len(loop) - 1):
        ax.plot(
            loop[i : i + 2, 0],
            loop[i : i + 2, 1],
            color=CMAP(i / (len(loop) - 1)),
            lw=3.2,
            solid_capstyle="round",
            zorder=3,
        )
    for i, angle in enumerate(day_theta):
        p = xy(angle)
        ax.scatter(
            *p,
            s=125,
            color=CMAP(i / 7),
            edgecolors="#f1f5ff",
            linewidths=1.1,
            zorder=6,
        )
        label = xy(angle, 1.14)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        horizontal = "left" if cosine > 0.25 else ("right" if cosine < -0.25 else "center")
        vertical = "bottom" if sine > 0.25 else ("top" if sine < -0.25 else "center")
        ax.text(
            *label,
            day_names[i],
            color=TEXT,
            fontsize=9.5,
            ha=horizontal,
            va=vertical,
            alpha=0.88,
        )
fig.suptitle(
    "Steering towards Wednesday",
    color=TEXT,
    fontsize=22,
    weight="bold",
    y=0.955,
)

tue = xy(day_theta[tue_i])
wed = xy(day_theta[wed_i])

# Flat SAE: add the Wednesday decoder vector to the unchanged Tuesday state.
left = axes[0]
left.text(
    0.0,
    2.30,
    "Linear",
    color=TEXT,
    fontsize=16,
    weight="bold",
    ha="center",
    va="center",
)
left.scatter(*tue, s=360, color=CMAP(tue_i / 7), edgecolors="white", lw=2.0, zorder=8)
left.scatter(*wed, s=310, facecolors="none", edgecolors=TEXT, lw=1.8, zorder=8)
origin = np.array([0.0, 0.0])
w_wed = wed - origin
sum_state = tue + w_wed

arrow(left, tue, sum_state, RED, width=4.0, mutation=26)
left.scatter(*sum_state, s=440, color=RED, alpha=0.94, edgecolors="#ffdfe2", lw=2.0, zorder=10)

# Manifold SAE: subtract the current point and add the target point.
right = axes[1]
right.text(
    0.0,
    2.30,
    "Manifold",
    color=TEXT,
    fontsize=16,
    weight="bold",
    ha="center",
    va="center",
)
right.scatter(*tue, s=360, color=CMAP(tue_i / 7), edgecolors="white", lw=2.0, zorder=8)
right.scatter(*wed, s=430, color=CMAP(wed_i / 7), edgecolors="#c6ffe9", lw=2.2, zorder=10)
arrow(right, tue, wed, GREEN, width=4.2, mutation=27)

# Why this edit is available: inference locates the current activation on the
# learned coordinate loop, target examples locate Wednesday, and the chart
# supplies the path between those coordinates.  The solid chord above is the
# corresponding edit in activation space.
arc_theta = np.linspace(day_theta[tue_i], day_theta[wed_i], 90)
arc = np.column_stack(
    [1.10 * rx * np.cos(arc_theta), 1.10 * ry * np.sin(arc_theta)]
)
right.plot(
    arc[:, 0],
    arc[:, 1],
    color=GREEN,
    lw=2.0,
    ls=(0, (4, 3)),
    alpha=0.82,
    zorder=7,
)
arrow(right, arc[42], arc[50], GREEN, width=1.8, mutation=15, z=8)
right.text(
    2.72,
    0.16,
    "learned coordinate path",
    color="#baffdf",
    fontsize=10.5,
    ha="left",
    va="center",
)
right.annotate(
    "encode current activation",
    xy=tue,
    xytext=(-1.45, 0.72),
    color=TEXT,
    fontsize=10.5,
    ha="center",
    va="center",
    bbox={"boxstyle": "round,pad=0.30", "fc": "#111a2f", "ec": "#7085b8", "alpha": 0.94},
    arrowprops={"arrowstyle": "->", "color": "#9eb0d8", "lw": 1.4},
    zorder=12,
)
right.annotate(
    "Wednesday examples\nlocate target coordinate",
    xy=wed,
    xytext=(-1.38, -0.62),
    color=TEXT,
    fontsize=10.5,
    ha="center",
    va="center",
    bbox={"boxstyle": "round,pad=0.32", "fc": "#10261f", "ec": GREEN, "alpha": 0.94},
    arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.4},
    zorder=12,
)

fig.savefig(OUT, facecolor=INK, dpi=200)
print(OUT)
