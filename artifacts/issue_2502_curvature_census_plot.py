"""Figure + statistics for the #2502 local-curvature census of Qwen3.5-4B."""

from __future__ import annotations

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYER_COLOR = {8: "#3b6ea5", 16: "#c1683b", 22: "#4f8a5b"}


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Difference in means, its standard error, and the Welch t statistic."""
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(diff), float(se), float(diff / se) if se > 0 else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--out-table", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    dims = sorted({r["tangent_dim"] for r in rows})
    nbs = sorted({r["n_neighbours"] for r in rows})
    layers = sorted({r["layer"] for r in rows})

    table = []
    for r in rows:
        real = np.array(r["per_landmark_real"])
        null = np.array(r["per_landmark_null"])
        diff, se, t = welch(real, null)
        table.append(
            {
                "layer": r["layer"],
                "n_neighbours": r["n_neighbours"],
                "tangent_dim": r["tangent_dim"],
                "real_mean": float(real.mean()),
                "real_se": float(real.std(ddof=1) / math.sqrt(len(real))),
                "null_mean": float(null.mean()),
                "null_se": float(null.std(ddof=1) / math.sqrt(len(null))),
                "excess": diff,
                "excess_se": se,
                "welch_t": t,
                "n_landmarks": int(len(real)),
                "frac_landmarks_real_positive": float((real > 0).mean()),
                "frac_landmarks_null_positive": float((null > 0).mean()),
            }
        )
    with open(args.out_table, "w") as fh:
        for t in table:
            fh.write(json.dumps(t) + "\n")

    fig, axes = plt.subplots(1, len(dims), figsize=(4.1 * len(dims), 4.0), sharey=True)
    if len(dims) == 1:
        axes = [axes]
    for ax, d in zip(axes, dims):
        ax.axhline(0.0, color="#999999", lw=0.9, zorder=1)
        for layer in layers:
            sel = [t for t in table if t["tangent_dim"] == d and t["layer"] == layer]
            sel.sort(key=lambda t: t["n_neighbours"])
            x = [t["n_neighbours"] for t in sel]
            c = LAYER_COLOR.get(layer, "#666666")
            ax.errorbar(
                x, [t["real_mean"] for t in sel], yerr=[t["real_se"] for t in sel],
                marker="o", color=c, lw=2.0, capsize=3, label=f"L{layer} real", zorder=3,
            )
            ax.errorbar(
                x, [t["null_mean"] for t in sel], yerr=[t["null_se"] for t in sel],
                marker="s", color=c, lw=1.4, ls="--", alpha=0.65, capsize=3,
                label=f"L{layer} Gaussian null", zorder=2, markerfacecolor="white",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(nbs)
        ax.set_xticklabels([str(n) for n in nbs])
        ax.set_xlabel("neighbourhood size")
        ax.set_title(f"tangent dim d = {d}")
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("held-out $R^2$ of the quadratic normal fit")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(layers) * 2, frameon=False,
               bbox_to_anchor=(0.5, -0.015), fontsize=9)
    fig.suptitle(
        "Qwen3.5-4B-Base residual stream is locally CURVED: neighbourhoods bend away from\n"
        "their own tangent plane out-of-sample; a covariance-matched Gaussian does not",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    fig.savefig(args.out_fig, dpi=170, bbox_inches="tight")

    best = max(table, key=lambda t: t["excess"])
    print(f"cells={len(table)} all_positive_excess={all(t['excess'] > 0 for t in table)}")
    print(f"min excess t = {min(t['welch_t'] for t in table):.2f}")
    print("strongest cell:", json.dumps(best))
    print(f"WROTE {args.out_fig} {args.out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
