"""Render the #2283/#2441 decoder-refresh A/B from `sae_decoder_refresh_scaling` output.

Consumes the `[refresh-epoch]` / `[refresh-config]` CSV lines the Rust bench
prints (analysis only — no production math lives here, SPEC line 8) and draws:

  1. the per-epoch refresh curve for both arms at one shape, with the block-CG
     column-tile width on a twin axis, because the tile is the carrier;
  2. refresh wall vs ambient dimension P for both arms, which is the scaling law
     the single production data point could not show.

Usage:
    python bench/plot_2283_decoder_refresh_tile.py \
        --base sweep_base.txt --fix sweep_fix.txt --out bench/figures/refresh_2283.png
"""

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_FIELDS = [
    "n", "p", "k", "s", "epoch", "refresh_s", "route_s", "refresh_over_route",
    "births", "cg_columns", "cg_iterations", "recycled_rank", "tile_columns",
    "max_component", "max_component_nnz", "operator_build_s", "kappa_bound", "ev",
]
CONFIG_FIELDS = [
    "n", "p", "k", "s", "epochs_run", "fit_s", "refresh_total_s", "route_total_s",
    "refresh_frac", "first_refresh_s", "last_refresh_s", "growth",
    "first_cg_iterations", "last_cg_iterations", "cg_growth",
    "first_tile", "last_tile", "first_nnz", "last_nnz", "first_kappa", "last_kappa",
]


def _rows(path, tag, fields):
    """Parse one bench output file's `[tag] ...` CSV rows into dicts."""
    prefix = f"[{tag}] "
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(prefix):
                continue
            body = line[len(prefix):].strip()
            if body.startswith(fields[0] + ","):
                continue  # the header row the bench prints once
            values = body.split(",")
            if len(values) != len(fields):
                raise ValueError(f"{path}: {tag} row has {len(values)} fields, want {len(fields)}: {body}")
            out.append({name: float(value) for name, value in zip(fields, values)})
    return out


def _shape(row):
    return (int(row["n"]), int(row["p"]), int(row["k"]), int(row["s"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="sweep output with the pre-fix rank bound")
    parser.add_argument("--fix", required=True, help="sweep output with the tile-reserving rank bound")
    parser.add_argument("--curve-shape", default="", help="n,p,k,s to draw the epoch curve for (default: slowest base shape)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_epochs = _rows(args.base, "refresh-epoch", EPOCH_FIELDS)
    fix_epochs = _rows(args.fix, "refresh-epoch", EPOCH_FIELDS)
    base_configs = _rows(args.base, "refresh-config", CONFIG_FIELDS)
    fix_configs = _rows(args.fix, "refresh-config", CONFIG_FIELDS)

    if args.curve_shape:
        target = tuple(int(v) for v in args.curve_shape.split(","))
    else:
        target = _shape(max(base_configs, key=lambda r: r["refresh_total_s"]))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # --- panel 1: the epoch curve, and the tile width that explains it --------
    for rows, label, color in ((base_epochs, "before", "#c0392b"), (fix_epochs, "after", "#1f77b4")):
        curve = [r for r in rows if _shape(r) == target]
        if not curve:
            continue
        left.plot([r["epoch"] for r in curve], [r["refresh_s"] for r in curve],
                  marker="o", color=color, label=f"refresh_s ({label})")
    tile_axis = left.twinx()
    for rows, label, color in ((base_epochs, "before", "#c0392b"), (fix_epochs, "after", "#1f77b4")):
        curve = [r for r in rows if _shape(r) == target]
        if not curve:
            continue
        tile_axis.plot([r["epoch"] for r in curve], [r["tile_columns"] for r in curve],
                       linestyle="--", color=color, alpha=0.55, label=f"tile_columns ({label})")
    tile_axis.set_ylabel("block-CG column tile width")
    left.set_xlabel("refresh (epoch index within one fit)")
    left.set_ylabel("decoder refresh, seconds")
    left.set_title(f"#2283 decoder refresh per epoch\nn={target[0]} p={target[1]} K={target[2]} s={target[3]}")
    handles, labels = left.get_legend_handles_labels()
    extra_handles, extra_labels = tile_axis.get_legend_handles_labels()
    left.legend(handles + extra_handles, labels + extra_labels, fontsize=8, loc="upper left")
    left.grid(alpha=0.3)

    # --- panel 2: the P scaling law ------------------------------------------
    for configs, label, color in ((base_configs, "before", "#c0392b"), (fix_configs, "after", "#1f77b4")):
        line = sorted(
            (r for r in configs if (int(r["n"]), int(r["k"]), int(r["s"])) == (target[0], target[2], target[3])),
            key=lambda r: r["p"],
        )
        if len(line) < 2:
            continue
        # Mean seconds per refresh, not the fit total: the ladder's arms stop
        # at different epoch counts, and a total would read that difference as
        # a cost difference.
        right.plot([r["p"] for r in line],
                   [r["refresh_total_s"] / r["epochs_run"] for r in line],
                   marker="s", color=color, label=label)
    right.set_xscale("log", base=2)
    right.set_yscale("log")
    right.set_xlabel("ambient dimension P (decoder columns)")
    right.set_ylabel("mean seconds per decoder refresh")
    right.set_title(f"refresh wall vs P\nn={target[0]} K={target[2]} s={target[3]}")
    right.legend(fontsize=9)
    right.grid(alpha=0.3, which="both")

    figure.tight_layout()
    figure.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
