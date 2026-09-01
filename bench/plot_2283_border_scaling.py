#!/usr/bin/env python3
"""Plot the #2283 curved-tier border-scaling A/B.

Reads the stdout of ``bench/sae_curved_border_scaling_2283.sh`` (both arms
concatenated) and draws the two sweeps side by side: elapsed time against ``p``
at fixed chart count, and elapsed time against chart count at fixed ``p`` and
fixed ``top_k``.

The second panel is the one that carries the argument. ``top_k`` is fixed, so
the amount of *live* work is constant along it; anything that grows is work
spent on decoder borders whose atom is inactive on the row, i.e. on structural
zeros. A flat fixed-arm curve there is the claim being made.

Analysis only -- no production math lives here (SPEC line 8). All numbers come
from the Rust example's own ``[2283-scaling]`` line and from ``/usr/bin/time``.

Usage::

    python bench/plot_2283_border_scaling.py sweep.out --out bits.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

POINT = re.compile(
    r"arm=(?P<arm>\S+).*?"
    r"n=(?P<n>\d+) p=(?P<p>\d+) charts=(?P<charts>\d+) top_k=(?P<top_k>\d+)"
)
# The reduced shapes terminate in the inner-solve refusal rather than a fitted
# model, so a point's shape may only be recoverable from the environment the
# runner echoed. Parse the timing fields independently of the shape fields.
TIMING = re.compile(
    r"elapsed_s=(?P<elapsed>[\d.]+) cpu_percent=(?P<cpu>\d+)% max_rss_kb=(?P<rss>\d+)"
)


def parse(path: Path) -> list[dict]:
    rows: list[dict] = []
    arm = None
    fixed: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if line.startswith("== #2283 curved-tier border scaling:"):
            arm = line.rsplit(":", 1)[1].strip()
            continue
        if line.startswith("== fixed "):
            fixed = {
                key: int(value)
                for key, value in re.findall(r"(\w+)=(\d+)", line)
            }
            continue
        timing = TIMING.search(line)
        if timing is None:
            continue
        shape = POINT.search(line)
        row = {
            "arm": arm,
            "elapsed_s": float(timing.group("elapsed")),
            "cpu_percent": int(timing.group("cpu")),
            "max_rss_kb": int(timing.group("rss")),
        }
        if shape is not None:
            row.update(
                {
                    "n": int(shape.group("n")),
                    "p": int(shape.group("p")),
                    "charts": int(shape.group("charts")),
                    "top_k": int(shape.group("top_k")),
                }
            )
        else:
            row.update({"n": fixed.get("n"), "top_k": fixed.get("top_k")})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path, nargs="+", help="sweep stdout file(s)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sweep-a-p",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="the p values of sweep A, in the order the runner emitted them",
    )
    parser.add_argument(
        "--sweep-b-charts",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="the chart counts of sweep B, in the order the runner emitted them",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.sweep:
        rows.extend(parse(path))

    arms: dict[str, list[dict]] = {}
    for row in rows:
        arms.setdefault(row["arm"], []).append(row)

    n_a = len(args.sweep_a_p)
    n_b = len(args.sweep_b_charts)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"baseline": "#c0392b", "fixed": "#1f77b4"}

    for arm, points in arms.items():
        if len(points) < n_a + n_b:
            raise SystemExit(
                f"arm {arm!r} has {len(points)} points, expected {n_a + n_b}; "
                "the runner did not finish, so no plot is drawn"
            )
        sweep_a = points[:n_a]
        sweep_b = points[n_a : n_a + n_b]
        color = colors.get(arm)
        left.plot(
            args.sweep_a_p,
            [row["elapsed_s"] for row in sweep_a],
            marker="o",
            label=arm,
            color=color,
        )
        right.plot(
            args.sweep_b_charts,
            [row["elapsed_s"] for row in sweep_b],
            marker="o",
            label=arm,
            color=color,
        )

    left.set_xscale("log", base=2)
    left.set_xlabel("ambient output dimension p")
    left.set_ylabel("elapsed, seconds")
    left.set_title("sweep A: p at 8 charts")
    left.legend()
    left.grid(alpha=0.3)

    right.set_xscale("log", base=2)
    right.set_xlabel("charts (top_k fixed at 2)")
    right.set_ylabel("elapsed, seconds")
    right.set_title("sweep B: charts at p=128, top_k fixed\n(live work is constant along this axis)")
    right.legend()
    right.grid(alpha=0.3)

    figure.suptitle("#2283 curved tier: cost of decoder borders whose atom is inactive")
    figure.tight_layout()
    figure.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
