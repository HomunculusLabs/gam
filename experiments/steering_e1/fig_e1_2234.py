#!/usr/bin/env python3
"""Figure for gam#2234 E1/E2: chart-coordinate steering on TWO chart topologies.

Reads the `e1_records.jsonl` / `e2_collateral.json` ledgers produced by
`run_e1.py` (cyclic: day-of-week circle) and `run_e1_ordinal.py` (non-cyclic:
ordinal interval) and draws, per structure:

  top row     dose-response — mean full-softmax probability mass moved onto the
              intended target token vs the fractional chart dose, on-manifold
              arm vs the matched-L2-norm flat-SAE control,
  bottom row  the E2 collateral frontier `g(e) = min collateral over recorded
              interventions with effect >= e`, i.e. the honest matched-EFFECT
              comparison (the two arms reach different effects at the same dose).

Pure numpy/matplotlib over the recorded ledgers — no model, no fit, so it
re-runs anywhere in a second.

    python3 experiments/steering_e1/fig_e1_2234.py \
        --cyclic-dir experiments/steering_e1/out_cyclic \
        --ordinal-dir experiments/steering_e1/out_ordinal \
        --out experiments/steering_e1/plots/issue_2234_e1_e2.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# One palette for the whole figure: the two arms keep the same colour in every
# panel, so a reader never re-learns the legend.
MANIFOLD = "#1f6feb"   # on-manifold chart-coordinate move
FLAT = "#d1495b"       # matched-norm flat-SAE direction
GRID = "#d7dce3"
INK = "#1b1f24"
ARMS = ("manifold", "flat")
ARM_COLOR = {"manifold": MANIFOLD, "flat": FLAT}
ARM_LABEL = {
    "manifold": "on-manifold (chart coordinate)",
    "flat": "flat SAE, matched L2 norm",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def dose_curve(records, arm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean +/- standard error of effect at each dose fraction, for one arm."""
    doses = sorted({float(r["dose_fraction"]) for r in records})
    mean, sem = [], []
    for d in doses:
        vals = np.asarray([
            float(r["target_probability_mass_moved"])
            for r in records
            if r["arm"] == arm and float(r["dose_fraction"]) == d
        ])
        mean.append(float(vals.mean()) if vals.size else np.nan)
        sem.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
    return np.asarray(doses), np.asarray(mean), np.asarray(sem)


def style(ax) -> None:
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9, length=3, color=GRID)


def panel_dose(ax, records, title: str, dose_unit: str) -> None:
    for arm in ARMS:
        d, m, s = dose_curve(records, arm)
        ax.plot(d, m, marker="o", markersize=4.5, linewidth=2.0,
                color=ARM_COLOR[arm], label=ARM_LABEL[arm], zorder=3)
        ax.fill_between(d, m - s, m + s, color=ARM_COLOR[arm], alpha=0.16, linewidth=0)
    ax.axhline(0.0, color=INK, linewidth=0.8, alpha=0.35, zorder=2)
    style(ax)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel(f"dose (fraction of the target move, {dose_unit})", fontsize=9, color=INK)
    ax.set_ylabel("target probability mass moved", fontsize=9, color=INK)


def panel_frontier(ax, e2: dict[str, Any], title: str) -> None:
    curve = e2["curve"]
    e = np.asarray([row["effect"] for row in curve])
    for arm in ARMS:
        y = np.asarray([
            np.nan if row[f"{arm}_collateral"] is None else float(row[f"{arm}_collateral"])
            for row in curve
        ])
        ax.plot(e, y, linewidth=2.2, color=ARM_COLOR[arm], label=ARM_LABEL[arm], zorder=3)
    eff = e2["collateral_efficiency"]
    verdict = "manifold DOMINATES" if e2["manifold_dominates"] else "no strict dominance"
    ax.text(
        0.02, 0.97,
        f"collateral per unit effect\n"
        f"  manifold {eff['manifold']:.4g}\n"
        f"  flat     {eff['flat']:.4g}\n"
        f"dominance fraction {e2['frontier_dominance_fraction']:.2f}\n"
        f"{verdict}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        color=INK, family="monospace",
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.4"},
    )
    style(ax)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel("achieved effect e (target mass moved)", fontsize=9, color=INK)
    ax.set_ylabel("min collateral KL to reach $\\geq e$", fontsize=9, color=INK)


def build(cyclic_dir: Path, ordinal_dir: Path, out_path: Path, model: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for label, directory, unit in (
        ("cyclic — day-of-week circle", cyclic_dir, "days round the circle"),
        ("non-cyclic — ordinal interval", ordinal_dir, "ranks along the line"),
    ):
        records = load_records(directory / "e1_records.jsonl")
        e2 = json.loads((directory / "e2_collateral.json").read_text())
        meta = json.loads((directory / "e1_summary.json").read_text())["meta"]
        panels.append((label, records, e2, meta, unit))

    fig, axes = plt.subplots(2, len(panels), figsize=(6.4 * len(panels), 8.4))
    fig.patch.set_facecolor("white")
    for col, (label, records, e2, meta, unit) in enumerate(panels):
        panel_dose(axes[0][col], records, f"({'ab'[col]}) dose-response — {label}", unit)
        panel_frontier(axes[1][col], e2, f"({'cd'[col]}) E2 collateral frontier — {label}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        f"gam#2234 E1/E2 — chart-coordinate steering vs a matched-norm flat SAE\n"
        f"{model}, block {panels[0][3]['layer_index']}, PCA-{panels[0][3].get('pca_dim', 64)} chart, "
        f"softmax assignment",
        fontsize=13, color=INK, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor="white")
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cyclic-dir", default="experiments/steering_e1/out_cyclic")
    ap.add_argument("--ordinal-dir", default="experiments/steering_e1/out_ordinal")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--out", default="experiments/steering_e1/plots/issue_2234_e1_e2.png")
    args = ap.parse_args()
    build(Path(args.cyclic_dir), Path(args.ordinal_dir), Path(args.out), args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
