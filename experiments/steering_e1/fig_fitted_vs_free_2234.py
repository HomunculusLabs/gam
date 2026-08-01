#!/usr/bin/env python3
"""gam#2234 — what the FITTED chart recovers, against the fit-free bar.

Consumes the per-dimension ledgers written by
``probe_chart_2234.py sweep --out dim<D>_<structure>.json`` and the planted
control written by ``probe_chart_2234.py planted --out planted_<structure>.json``
and draws the comparison this issue turns on:

  (a,b) structure-recovery R² of `sae_manifold_fit`'s circle/interval coordinate
        as a function of the chart's PCA dimension, against the fit-free top-2
        recovery on the SAME chart. Refusals are marked on the axis rather than
        dropped — a refusal is a datum.
  (c)   the fitted coordinate's standard deviation on a log axis. This is what
        separates "the coordinate disagrees with the generator" from "the
        coordinate collapsed"; both score R² ≈ 0.
  (d)   the planted control: the same fit on the measured chart, on a surrogate
        with the measured label geometry and Gaussian within-label scatter, and
        on an ideal circle/line of the measured radius and noise.

Pure numpy/matplotlib over the ledgers — no model, no fit.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

GRID = "#d7dce3"
INK = "#1b1f24"
FITTED = "#d1495b"
FREE = "#1f6feb"
REFUSED = "#8a94a6"
ARMS = ("real", "surrogate", "ideal")
ARM_COLOR = {"real": "#d1495b", "surrogate": "#f3a712", "ideal": "#2f9e6b"}


def style(ax):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9, length=3, color=GRID)


def coord_std(status: str) -> float:
    m = re.search(r"coord_std=([0-9eE.+-]+)", status or "")
    return float(m.group(1)) if m else float("nan")


def load_scan(ledger_dir: Path, structure: str):
    """(dims, fitted R², coord std, refused dims, fit-free bar) for one structure."""
    dims, fitted, stds, refused = [], [], [], []
    free = float("nan")
    for path in sorted(ledger_dir.glob(f"dim*_{structure}.json")):
        for row in json.loads(path.read_text()):
            if row["k_atoms"] == 0:
                free = float(row["structure_recovery_r2"])
                continue
            d = int(row["pca_dim"])
            r2 = row["structure_recovery_r2"]
            if r2 is None or not np.isfinite(r2):
                refused.append(d)
                continue
            dims.append(d)
            fitted.append(float(r2))
            stds.append(coord_std(row.get("status", "")))
    order = np.argsort(dims)
    return (np.asarray(dims)[order], np.asarray(fitted)[order],
            np.asarray(stds)[order], sorted(set(refused)), free)


def panel_scan(ax, structure, label, ledger_dir):
    dims, fitted, _, refused, free = load_scan(ledger_dir, structure)
    ax.axhline(free, color=FREE, linewidth=2.4, zorder=3,
               label=f"fit-free top-2 plane ({free:.3f})")
    ax.plot(dims, fitted, marker="o", markersize=6, linewidth=2.2, color=FITTED,
            zorder=4, label="sae_manifold_fit coordinate")
    for d in refused:
        ax.plot([d], [0.0], marker="x", markersize=11, markeredgewidth=2.4,
                color=REFUSED, zorder=5)
        ax.annotate("REFUSED", (d, 0.0), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color=REFUSED)
    if fitted.size:
        best = int(np.argmax(fitted))
        ax.annotate(f"best {fitted[best]:.3f}\n(dim {dims[best]})",
                    (dims[best], fitted[best]), textcoords="offset points",
                    xytext=(8, 14), fontsize=8.5, color=FITTED)
    style(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(list(dims) + refused)))
    ax.get_xaxis().set_major_formatter(__import__("matplotlib").ticker.ScalarFormatter())
    ax.set_title(label, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("chart PCA dimension", fontsize=9, color=INK)
    ax.set_ylabel("structure recovery R²", fontsize=9, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="center right")


def panel_std(ax, ledger_dir):
    for structure, marker, color in (("weekday", "o", "#1f6feb"), ("ordinal", "s", "#2f9e6b")):
        dims, _, stds, _, _ = load_scan(ledger_dir, structure)
        keep = np.isfinite(stds) & (stds > 0)
        if keep.any():
            ax.plot(dims[keep], stds[keep], marker=marker, markersize=6, linewidth=2.0,
                    color=color, label=structure)
    ax.axhline(1e-6, color=INK, linewidth=1.0, linestyle="--", alpha=0.5)
    ax.annotate("below here the coordinate is a CONSTANT\n(R² ≈ 0 for a numerical reason)",
                (0.02, 1e-6), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, -22), fontsize=8, color=INK)
    style(ax)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(__import__("matplotlib").ticker.ScalarFormatter())
    ax.set_title("(c) the fitted coordinate's spread — collapse vs disagreement",
                 fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("chart PCA dimension", fontsize=9, color=INK)
    ax.set_ylabel("std of the fitted chart coordinate", fontsize=9, color=INK)
    ax.legend(fontsize=8.5, frameon=False)


def panel_planted(ax, ledger_dir):
    groups, width = ("weekday", "ordinal"), 0.26
    x = np.arange(len(groups))
    any_row = False
    for i, arm in enumerate(ARMS):
        vals, free = [], []
        for structure in groups:
            path = ledger_dir / f"planted_{structure}.json"
            row = None
            if path.exists():
                rows = {r["arm"]: r for r in json.loads(path.read_text())}
                row = rows.get(arm)
            vals.append(float(row["structure_recovery_r2"])
                        if row and row["structure_recovery_r2"] is not None
                        and np.isfinite(row["structure_recovery_r2"]) else np.nan)
            free.append(float(row["fit_free_r2"]) if row else np.nan)
        any_row = any_row or np.any(np.isfinite(vals))
        off = (i - 1) * width
        bars = ax.bar(x + off, np.nan_to_num(vals), width=width, color=ARM_COLOR[arm],
                      label=arm, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, (0 if not np.isfinite(v) else v) + 0.02,
                    "REFUSED" if not np.isfinite(v) else f"{v:.3f}",
                    ha="center", fontsize=7.5, color=INK, rotation=90 if not np.isfinite(v) else 0)
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("structure recovery R² of the FITTED coordinate", fontsize=9, color=INK)
    ax.set_title("(d) planted control — same call, same chart dimension,\n"
                 "increasingly idealised data", fontsize=10.5, color=INK, loc="left", pad=8)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    if not any_row:
        ax.text(0.5, 0.5, "planted control not available", transform=ax.transAxes,
                ha="center", fontsize=10, color=REFUSED)


def build(ledger_dir: Path, out_path: Path, model: str, layer: int, rows: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6))
    fig.patch.set_facecolor("white")
    panel_scan(axes[0][0], "weekday",
               "(a) weekday — the day-of-week CIRCLE", ledger_dir)
    panel_scan(axes[0][1], "ordinal",
               "(b) ordinal — the NON-CYCLIC rank line", ledger_dir)
    panel_std(axes[1][0], ledger_dir)
    panel_planted(axes[1][1], ledger_dir)
    fig.suptitle(
        "gam#2234 — the chart is in the cloud; the fitted chart does not find it\n"
        f"{model}, block {layer}, {rows}, K=1, d_atom=1, softmax assignment, "
        "per-head-centered chart",
        fontsize=13, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor="white")
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--rows", default="10 context heads")
    ap.add_argument("--out",
                    default="experiments/steering_e1/plots/issue_2234_fitted_vs_free.png")
    args = ap.parse_args()
    build(Path(args.ledger_dir), Path(args.out), args.model, args.layer, args.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
