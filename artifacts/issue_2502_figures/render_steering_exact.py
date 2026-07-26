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
MUTED = "#8ea0c9"
RED = "#ff6b78"
GREEN = "#5fe0ad"
CMAP = plt.get_cmap("twilight_shifted")

fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=200)
fig.patch.set_facecolor(INK)
fig.subplots_adjust(left=0.025, right=0.975, bottom=0.07, top=0.90, wspace=0.08)

rx, ry = 2.10, 1.58
theta = np.linspace(0, 2 * np.pi, 720)
day_theta = np.pi / 2 - np.arange(7) * 2 * np.pi / 7
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
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
    ax.set_xlim(-3.2, 4.3)
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
        ax.text(
            *(xy(angle, 1.20)),
            names[i],
            color=TEXT if i in (tue_i, wed_i) else MUTED,
            fontsize=10.5,
            ha="center",
            va="center",
            weight="semibold" if i in (tue_i, wed_i) else "normal",
        )

tue = xy(day_theta[tue_i])
wed = xy(day_theta[wed_i])

# Flat SAE: add the Wednesday decoder vector to the unchanged Tuesday state.
left = axes[0]
left.set_title("LINEAR SAE", color=TEXT, fontsize=16, weight="bold", pad=6)
left.scatter(*tue, s=360, color=CMAP(tue_i / 7), edgecolors="white", lw=2.0, zorder=8)
left.scatter(*wed, s=310, facecolors="none", edgecolors=TEXT, lw=1.8, zorder=8)
origin = np.array([0.0, 0.0])
w_wed = wed - origin
sum_state = tue + w_wed

arrow(left, origin, wed, RED, width=2.0, mutation=18, z=7)
left.text(*(0.52 * wed + np.array([0.10, 0.12])), r"$\alpha w_{\rm Wed}$",
          color="#ffb1b8", fontsize=13, ha="left", va="center")
left.plot(
    [wed[0], sum_state[0]],
    [wed[1], sum_state[1]],
    color=RED,
    lw=1.2,
    alpha=0.55,
    ls=(0, (3, 3)),
)
left.plot(
    [tue[0], sum_state[0]],
    [tue[1], sum_state[1]],
    color=RED,
    lw=1.2,
    alpha=0.55,
    ls=(0, (3, 3)),
)
arrow(left, tue, sum_state, RED, width=4.0, mutation=26)
left.scatter(*sum_state, s=440, color=RED, alpha=0.94, edgecolors="#ffdfe2", lw=2.0, zorder=10)
left.text(
    *(sum_state + np.array([0.18, 0.18])),
    r"$x_{\rm Tue}+\alpha w_{\rm Wed}$",
    color="#ffdfe2",
    fontsize=14,
    ha="left",
    va="bottom",
)
left.text(
    *(sum_state + np.array([0.20, -0.18])),
    "off the learned loop",
    color="#ff9ca6",
    fontsize=11,
    ha="left",
    va="top",
)

# Manifold SAE: subtract the current point and add the target point.
right = axes[1]
right.set_title("MANIFOLD SAE", color=TEXT, fontsize=16, weight="bold", pad=6)
right.scatter(*tue, s=360, color=CMAP(tue_i / 7), edgecolors="white", lw=2.0, zorder=8)
right.scatter(*wed, s=430, color=CMAP(wed_i / 7), edgecolors="#c6ffe9", lw=2.2, zorder=10)
arrow(right, tue, wed, GREEN, width=4.2, mutation=27)
arc_theta = np.linspace(day_theta[tue_i], day_theta[wed_i], 120)
arc = np.column_stack(
    [1.22 * rx * np.cos(arc_theta), 1.22 * ry * np.sin(arc_theta)]
)
right.plot(arc[:, 0], arc[:, 1], color=GREEN, lw=1.8, ls=(0, (4, 3)), alpha=0.9)
right.text(
    *(arc[len(arc) // 2] + np.array([0.28, 0.06])),
    r"$\Delta t=1\ {\rm day}$",
    color="#b7ffe0",
    fontsize=12,
    ha="center",
    va="bottom",
)
right.text(
    0,
    -2.40,
    r"$x' = x + a\,[\gamma(t_{\rm Wed})-\gamma(t_{\rm Tue})]$",
    color=TEXT,
    fontsize=15,
    ha="center",
    va="center",
)

fig.savefig(OUT, facecolor=INK, dpi=200)
print(OUT)
