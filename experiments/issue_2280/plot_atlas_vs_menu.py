"""#2280 — figure for the atlas-vs-fixed-menu calibration on the planted zoo.

Analysis and plotting only; the measurement itself is the Rust test
`structure_harvest::tests_atlas_prior_2280::atlas_versus_fixed_menu_on_the_planted_zoo_2280`,
which prints the table this script draws. No math here decides anything — the
rows below are transcribed from a recorded run so the figure can be regenerated
without a compute allocation.

Provenance of the transcribed rows
----------------------------------
MSI job 14612942, partition msismall, `cargo test -p gam-sae --release --lib
tests_atlas_prior_2280 -- --nocapture`, source
`2a43603ee9e73e6cb041992085d6edda57ae1cd0` (a verification branch whose only
delta against `origin/main` at `930a423f7` is the measurement itself).

Usage
-----
    python experiments/issue_2280/plot_atlas_vs_menu.py --out <path>.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

JOB = "MSI 14612942"
SOURCE = "2a43603ee9e73e6cb041992085d6edda57ae1cd0"


@dataclass(frozen=True)
class Row:
    fixture: str
    d: int
    truth: str
    atlas: str
    menu: str

    @property
    def atlas_right(self) -> bool:
        return _names(self.atlas, self.truth)

    @property
    def menu_right(self) -> bool:
        return _names(self.menu, self.truth)


def _names(verdict: str, truth: str) -> bool:
    """`ConstantCurvature` counts as naming `Euclidean`.

    The #944 curvature fusion deliberately subsumes the flat patch into the
    fitted-kappa candidate, so a flat truth can only ever surface under the fused
    name. Scoring them apart would mark the race wrong for a reason that is not
    about topology. This mirrors `names_truth` in the Rust test exactly.
    """
    if verdict == truth:
        return True
    return truth == "Euclidean" and verdict == "ConstantCurvature"


# Transcribed verbatim from the job's stdout table.
ROWS = [
    Row("circle", 1, "Circle", "Circle", "Euclidean"),
    Row("trefoil", 1, "Circle", "Circle", "Euclidean"),
    Row("open_arc", 1, "Euclidean", "Euclidean", "Euclidean"),
    Row("plane", 2, "Euclidean", "Euclidean", "ConstantCurvature"),
    Row("swiss_roll", 2, "Euclidean", "REFUSED", "ConstantCurvature"),
    Row("cylinder", 2, "Cylinder", "Cylinder", "ConstantCurvature"),
    Row("mobius", 2, "Mobius", "Mobius", "ConstantCurvature"),
    Row("torus", 2, "Torus", "REFUSED", "ConstantCurvature"),
    Row("sphere", 2, "Sphere", "Sphere", "ConstantCurvature"),
]

RIGHT = "#2f7d4f"
WRONG = "#b03030"
ABSTAIN = "#8a8a8a"


def _cell_color(verdict: str, right: bool) -> str:
    if right:
        return RIGHT
    return ABSTAIN if verdict == "REFUSED" else WRONG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [2.5, 1.0]}
    )

    n = len(ROWS)
    for i, row in enumerate(ROWS):
        y = n - 1 - i
        ax.text(-0.06, y, f"{row.fixture}  (d={row.d})", ha="right", va="center", fontsize=11)
        ax.text(0.30, y, row.truth, ha="center", va="center", fontsize=10, style="italic")
        for x, (verdict, right) in (
            (1.05, (row.atlas, row.atlas_right)),
            (1.95, (row.menu, row.menu_right)),
        ):
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x - 0.42, y - 0.34),
                    0.84,
                    0.68,
                    boxstyle="round,pad=0.02",
                    linewidth=0,
                    facecolor=_cell_color(verdict, right),
                    alpha=0.88,
                )
            )
            ax.text(x, y, verdict, ha="center", va="center", fontsize=9.5, color="white")

    ax.text(0.30, n - 0.35, "planted truth", ha="center", fontsize=11, fontweight="bold")
    ax.text(1.05, n - 0.35, "atlas (charts)", ha="center", fontsize=11, fontweight="bold")
    ax.text(
        1.95,
        n - 0.35,
        "global-linear seed\n+ fixed menu",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(-1.05, 2.55)
    ax.set_ylim(-0.7, n + 0.35)
    ax.axis("off")
    ax.set_title(
        "#2280  What names the planted manifold?",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )

    atlas_right = sum(r.atlas_right for r in ROWS)
    menu_right = sum(r.menu_right for r in ROWS)
    ax_bar.bar(
        ["atlas", "menu-race"],
        [atlas_right, menu_right],
        color=[RIGHT, WRONG],
        alpha=0.88,
        width=0.55,
    )
    for x, value in enumerate([atlas_right, menu_right]):
        ax_bar.text(x, value + 0.12, f"{value}/{n}", ha="center", fontsize=13, fontweight="bold")
    ax_bar.set_ylim(0, n + 0.9)
    ax_bar.set_ylabel("fixtures whose planted truth was named")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.set_title(
        "the menu-race's verdict is CONSTANT\nwithin each d "
        "(Euclidean at every d=1,\nConstantCurvature at every d=2)",
        fontsize=9.5,
        loc="left",
    )

    fig.text(
        0.01,
        0.015,
        f"planted zoo, {JOB}, source {SOURCE[:9]}   "
        "grey = refusal (abstention, not a misnaming)",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
