"""Figure for the #2502 inner-solver wall: the fit is not budget-starved.

Left: the raw KKT residual `solve_fixed_point` reports against the iteration budget
it was given. Middle: the same residual when the activations are simply rescaled.
Right: the same residual against the number of rows fit. The acceptance bar is a
constant, so the two right-hand panels are the currency problem and the left panel
is the stall.

Consumes the jsonl written by probe_n_iter_ladder.py and probe_kkt_scale.py.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e2"
SERIES = "#2a78d6"
BAR = "#eb6834"
TOLERANCE = 1.0e-6


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d9d8d4")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_facecolor(SURFACE)


def main():
    ladder = sorted(load(sys.argv[1]), key=lambda r: r["n_iter"])
    scale = load(sys.argv[2])
    out = sys.argv[3]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7))
    fig.patch.set_facecolor(SURFACE)
    ax, ax2, ax3 = axes

    xs = [r["n_iter"] for r in ladder]
    ys = [r["raw_kkt_max"] for r in ladder]
    ax.plot(xs, ys, "-o", color=SERIES, lw=2.0, ms=7, mec=SURFACE, mew=1.6, zorder=3)
    ax.axhline(TOLERANCE, color=BAR, lw=2.0, ls=(0, (5, 3)), zorder=2)
    ax.text(2.4, TOLERANCE * 1.8, "acceptance tolerance, 1e−6", color=BAR, fontsize=9.5)
    ax.annotate(
        "flat from 256 to 2048:\n8× the budget buys nothing",
        xy=(1024, 1.6), xytext=(150, 60), fontsize=9.5, color=INK_2,
        arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.2),
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(3e-7, 3e3)
    style(ax, "The inner solve is not budget-starved",
          "inner fixed-point cycles allowed (n_iter)",
          "raw KKT residual at termination")

    sc = sorted([r for r in scale if r["arm"] == "scale"], key=lambda r: r["scale"])
    ax2.plot([r["rms_row_norm"] for r in sc], [r["raw_kkt_max"] for r in sc],
             "-o", color=SERIES, lw=2.0, ms=7, mec=SURFACE, mew=1.6, zorder=3)
    ax2.axhline(TOLERANCE, color=BAR, lw=2.0, ls=(0, (5, 3)), zorder=2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(3e-7, 3e3)
    style(ax2, "…but the residual carries the data's units",
          "RMS row norm of the activations\n(same rows, same model, rescaled)",
          "raw KKT residual at termination")
    ax2.annotate(
        "a 1000× rescale moves the\nresidual by 160,000×",
        xy=(0.061, 0.0032), xytext=(0.09, 40), fontsize=9.5, color=INK_2,
        arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.2),
    )

    rw = sorted([r for r in scale if r["arm"] == "rows"], key=lambda r: r["rows"])
    ax3.plot([r["rows"] for r in rw], [r["raw_kkt_max"] for r in rw],
             "-o", color=SERIES, lw=2.0, ms=7, mec=SURFACE, mew=1.6, zorder=3)
    ax3.axhline(TOLERANCE, color=BAR, lw=2.0, ls=(0, (5, 3)), zorder=2)
    ax3.set_xscale("log", base=2)
    ax3.set_yscale("log")
    ax3.set_ylim(3e-7, 3e3)
    ax3.set_xticks([500, 1000, 2000, 4000, 8000])
    ax3.set_xticklabels(["500", "1k", "2k", "4k", "8k"])
    style(ax3, "…and is not invariant to the row count",
          "rows fit (same activations, same model)",
          "raw KKT residual at termination")

    fig.text(
        0.006, 0.02,
        "Qwen3.5-4B-Base L16 PCA-64 chart, K=128 circle atoms, top_k=4, gpu=off, "
        "gamfit 0.1.259 · middle and right panels at n_iter=32\n"
        "The three solver files involved are blob-identical between that wheel and "
        "current main, so this is HEAD behaviour.",
        fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
