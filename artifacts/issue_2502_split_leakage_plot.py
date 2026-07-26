"""Figure for the #2502 split contract: what the row-level split was actually worth.

Left: the change in held-out explained variance when the train/test boundary moves
from a random row cut to the declared document cut, per estimator. Right: the
mechanism — how often a held-out row's nearest training row came from its own
document, and how much closer that made it.

Consumes the jsonl written by issue_2502_split_leakage.py; the measurement stays
the source of truth.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e2"
FALLS = "#eb6834"   # held-out score falls under the document split
RISES = "#2a78d6"   # held-out score rises
NEUTRAL = "#b9b8b3"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "split_leakage.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "issue_2502_split_leakage.png"
    recs = [json.loads(line) for line in open(src) if line.strip()]
    manifest = next(r for r in recs if r["record"] == "split_manifest")
    geom = {r["arm"]: r for r in recs if r["record"] == "arm_geometry"}

    by = {}
    for r in recs:
        if r["record"].startswith("pca_M"):
            by.setdefault(r["record"], {})[r["arm"]] = [r["test_ev_ambient"]]
        elif r["record"] == "topk_sae":
            by.setdefault("topk_sae", {}).setdefault(r["arm"], []).append(
                r["test_ev_ambient"]
            )

    labels = {
        "pca_M8": "PCA-8",
        "pca_M16": "PCA-16",
        "pca_M64": "PCA-64",
        "topk_sae": "TopK SAE\nK = 32,000, L0 = 8",
    }
    order = ["pca_M8", "pca_M16", "pca_M64", "topk_sae"]

    deltas, errs, names = [], [], []
    for key in order:
        row = np.array(by[key]["row"])
        doc = np.array(by[key]["doc"])
        deltas.append(100.0 * (doc.mean() - row.mean()))
        # Seed spread of the difference, when there is more than one seed.
        errs.append(
            100.0 * np.hypot(row.std(ddof=1), doc.std(ddof=1)) if len(row) > 1 else 0.0
        )
        names.append(labels[key])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 4.9), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    fig.patch.set_facecolor(SURFACE)

    y = np.arange(len(order))
    colors = [FALLS if d < 0 else RISES for d in deltas]
    ax.barh(y, deltas, height=0.5, color=colors, zorder=3,
            xerr=errs, error_kw=dict(ecolor=INK_2, elinewidth=1.4, capsize=4))
    ax.axvline(0, color=NEUTRAL, lw=1.4, zorder=2)
    for i, (d, e) in enumerate(zip(deltas, errs)):
        tip = d - e if d < 0 else d + e
        off = -0.018 if d < 0 else 0.018
        ax.text(tip + off, i, f"{d:+.2f}", va="center",
                ha="right" if d < 0 else "left", fontsize=10.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(-0.50, 0.18)
    ax.set_xlabel("change in held-out explained variance, in EV points\n"
                  "(document split minus random row split; error bars are the "
                  "seed spread over 3 fits per arm)",
                  color=INK_2, fontsize=9.5)
    ax.set_title("The row split was leaking, but only into the overcomplete dictionary",
                 color=INK, fontsize=12.5, loc="left", pad=12)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d4")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_facecolor(SURFACE)

    frac = [100 * geom[a]["frac_nn_same_document"] for a in ("row", "doc")]
    ax2.bar([0, 1], frac, width=0.46, color=[FALLS, RISES], zorder=3)
    for i, v in enumerate(frac):
        ax2.text(i, v + 0.12, f"{v:.1f}%", ha="center", fontsize=11, color=INK_2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["random row split", "document split"], fontsize=10, color=INK_2)
    ax2.set_ylim(0, 6.6)
    ax2.set_ylabel("held-out rows whose nearest training\nrow came from the SAME document",
                   color=INK_2, fontsize=9.5)
    ax2.set_title("How much contamination there was to remove",
                  color=INK, fontsize=12.5, loc="left", pad=12)
    ax2.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color("#d9d8d4")
    ax2.tick_params(colors=INK_2, labelsize=9)
    ax2.set_facecolor(SURFACE)

    fig.text(
        0.615, 0.075,
        "Median distance to the nearest training row, as a share of the RMS\n"
        f"row norm: {100 * geom['row']['median_nn_distance_relative']:.1f}% under the "
        f"row split, {100 * geom['doc']['median_nn_distance_relative']:.1f}% under the "
        "document split.\nHeld-out rows sit no further from the training set once\n"
        "documents are disjoint — which is why removing the\ncontamination costs so "
        "little.",
        ha="left", va="bottom", fontsize=9, color=INK_2,
    )
    fig.text(
        0.006, 0.012,
        "Qwen3.5-4B-Base L16, wikitext-103 · arms identical except the split rule "
        "(50,000 fit / ~10,000 held-out rows, train-only PCA-128 chart, ambient "
        f"2560-d scoring)\nsplit_hash {manifest['split_hash']} · "
        f"{manifest['n_docs_test']} of {manifest['n_docs']} documents held out",
        fontsize=8, color=INK_2,
    )
    fig.tight_layout(rect=[0, 0.26, 1, 1])
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
