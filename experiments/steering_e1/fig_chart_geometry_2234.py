#!/usr/bin/env python3
"""Fit-free chart geometry for gam#2234 — what is in the cloud before anything is fitted.

Reads the ambient activation clouds dumped by
``probe_chart_2234.py harvest`` and draws the geometry that decides whether E1
can measure anything at all:

  (a) the weekday cloud's top-2 plane, RAW — head identity owns it,
  (b) the same cloud's top-2 plane after per-head centering — the day-of-week
      circle, with its seven centroids in calendar order,
  (c) the ordinal cloud's PC1 against rank after the same centering — a
      monotone line, no periodicity anywhere,
  (d) the recovery bars: raw vs centered vs the per-label-centroid ceiling.

Pure numpy/matplotlib, no `gamfit` and no model: this is the BAR a fitted chart
has to clear, and it is cheap enough to re-run anywhere.

    python3 experiments/steering_e1/fig_chart_geometry_2234.py \
        --weekday-npz big_weekday.npz --ordinal-npz big_ordinal.npz \
        --out experiments/steering_e1/plots/issue_2234_chart_geometry.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

TAU = 2.0 * math.pi
GRID = "#d7dce3"
INK = "#1b1f24"
RAW = "#8a94a6"
CENTERED = "#1f6feb"
CEILING = "#2f9e6b"
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# Colour by calendar position, so a genuine circle shows as a colour wheel and a
# scrambled one does not. Perceptually cyclic by construction.
WHEEL = ["#e4572e", "#f3a712", "#a8c686", "#2f9e6b", "#1f6feb", "#6a4c93", "#d64550"]


def load(path: Path):
    data = np.load(path, allow_pickle=False)
    return data["X"].astype(np.float64), data["label_index"], data["template_index"]


def center_per_head(X, template_index):
    out = X.copy()
    for ti in np.unique(template_index):
        rows = np.flatnonzero(template_index == ti)
        out[rows] -= out[rows].mean(0, keepdims=True)
    return out


def plane(X):
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def circular_r2(angle, label_index, n_labels):
    truth = np.exp(1j * TAU * label_index.astype(np.float64) / n_labels)
    chart = np.exp(1j * angle)
    return float(max(abs(np.mean(truth * np.conj(chart))), abs(np.mean(truth * chart))) ** 2)


def linear_r2(signal, label_index):
    x = label_index.astype(np.float64)
    d = np.column_stack([np.ones(x.size), x])
    coef, *_ = np.linalg.lstsq(d, signal, rcond=None)
    resid = signal - d @ coef
    return 1.0 - float(np.sum(resid ** 2)) / max(float(np.sum((signal - signal.mean()) ** 2)), 1e-30)


def centroid_r2(X, label_index, n_labels, cyclic):
    cent = np.stack([X[label_index == li].mean(0) for li in range(n_labels)])
    p = plane(cent)
    if cyclic:
        return circular_r2(np.arctan2(p[:, 1], p[:, 0]), np.arange(n_labels), n_labels)
    return linear_r2(p[:, 0], np.arange(n_labels))


def style(ax):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9, length=3, color=GRID)


def scatter_plane(ax, scores, label_index, n_labels, title, cyclic, labels=None):
    for li in range(n_labels):
        rows = label_index == li
        ax.scatter(scores[rows, 0], scores[rows, 1], s=22, alpha=0.75, linewidths=0,
                   color=WHEEL[li % len(WHEEL)],
                   label=(labels[li] if labels is not None else str(li)))
    cent = np.stack([scores[label_index == li].mean(0) for li in range(n_labels)])
    if cyclic:
        ring = np.vstack([cent, cent[:1]])
        ax.plot(ring[:, 0], ring[:, 1], color=INK, linewidth=1.4, alpha=0.55, zorder=4)
    ax.scatter(cent[:, 0], cent[:, 1], s=90, marker="D", zorder=5,
               facecolors="white", edgecolors=INK, linewidths=1.3)
    style(ax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("PC 1", fontsize=9, color=INK)
    ax.set_ylabel("PC 2", fontsize=9, color=INK)


def build(weekday_npz: Path, ordinal_npz: Path, out_path: Path, model: str, layer: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Xw, lw, tw = load(weekday_npz)
    Xo, lo, to = load(ordinal_npz)
    nw, no = int(lw.max()) + 1, int(lo.max()) + 1
    Xwc, Xoc = center_per_head(Xw, tw), center_per_head(Xo, to)

    pw_raw, pw_cen = plane(Xw), plane(Xwc)
    po_raw, po_cen = plane(Xo), plane(Xoc)
    r2 = {
        "weekday_raw": circular_r2(np.arctan2(pw_raw[:, 1], pw_raw[:, 0]), lw, nw),
        "weekday_centered": circular_r2(np.arctan2(pw_cen[:, 1], pw_cen[:, 0]), lw, nw),
        "weekday_ceiling": centroid_r2(Xwc, lw, nw, True),
        "ordinal_raw": linear_r2(po_raw[:, 0], lo),
        "ordinal_centered": linear_r2(po_cen[:, 0], lo),
        "ordinal_ceiling": centroid_r2(Xoc, lo, no, False),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.4))
    fig.patch.set_facecolor("white")

    scatter_plane(axes[0][0], pw_raw, lw, nw,
                  f"(a) weekday cloud, RAW top-2 plane — circular R² = {r2['weekday_raw']:.4f}",
                  True, WEEKDAYS)
    scatter_plane(axes[0][1], pw_cen, lw, nw,
                  f"(b) same cloud, per-head centered — circular R² = "
                  f"{r2['weekday_centered']:.4f}", True, WEEKDAYS)
    axes[0][1].legend(fontsize=7.5, frameon=False, ncol=2, loc="best")

    ax = axes[1][0]
    jitter = (np.random.default_rng(0).random(lo.size) - 0.5) * 0.25
    for li in range(no):
        rows = lo == li
        ax.scatter(lo[rows] + jitter[rows], po_cen[rows, 0], s=20, alpha=0.7, linewidths=0,
                   color=CENTERED)
    cent = np.asarray([po_cen[lo == li, 0].mean() for li in range(no)])
    ax.plot(np.arange(no), cent, color=INK, linewidth=1.6, marker="D", markersize=6,
            markerfacecolor="white", markeredgecolor=INK, zorder=5)
    style(ax)
    ax.set_title(f"(c) ordinal cloud, per-head centered — PC1 vs rank, linear R² = "
                 f"{r2['ordinal_centered']:.4f}\n(no periodicity: the ladder never wraps)",
                 fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("ordinal rank (` one` … ` twelve`)", fontsize=9, color=INK)
    ax.set_ylabel("PC 1", fontsize=9, color=INK)

    ax = axes[1][1]
    groups = ("weekday (circle)", "ordinal (line)")
    x = np.arange(len(groups))
    width = 0.26
    for off, key, color, name in (
        (-width, "raw", RAW, "raw cloud"),
        (0.0, "centered", CENTERED, "per-head centered"),
        (width, "ceiling", CEILING, "per-label centroid ceiling"),
    ):
        vals = [r2[f"weekday_{key}"], r2[f"ordinal_{key}"]]
        bars = ax.bar(x + off, vals, width=width, color=color, label=name, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", fontsize=8, color=INK)
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("structure recovery R² (fit-free)", fontsize=9, color=INK)
    ax.set_title("(d) the bar a FITTED chart has to clear", fontsize=10.5, color=INK,
                 loc="left", pad=8)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    fig.suptitle(
        f"gam#2234 — the generator is in the cloud, but only after the context head is removed\n"
        f"{model}, block {layer}, label-token capture; "
        f"{Xw.shape[0]} weekday rows / {Xo.shape[0]} ordinal rows, no fit anywhere",
        fontsize=13, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor="white")
    print(f"wrote {out_path}")
    for k, v in r2.items():
        print(f"{k} = {v:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weekday-npz", required=True)
    ap.add_argument("--ordinal-npz", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--out", default="experiments/steering_e1/plots/issue_2234_chart_geometry.png")
    args = ap.parse_args()
    build(Path(args.weekday_npz), Path(args.ordinal_npz), Path(args.out), args.model, args.layer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
