#!/usr/bin/env python3
"""#2234 — on-manifold vs off-manifold steering, measured on real activations.

Three interventions at the SAME ambient step length, so the only difference is
the direction's relation to the fitted chart:

  on_chart   rotate the row's chart coordinate by dtheta radians and step to the
             new point on the circle -- the intervention the issue proposes
  radial     step the same distance INSIDE the chart plane but along the radius,
             changing r and not theta -- isolates "on the plane" from "along the
             curve"
  off_chart  step the same distance in a random direction ORTHOGONAL to the
             chart plane -- the off-manifold control

`kl_target` is the intended effect (exact KL at the edited position).
Collateral is measured only at positions AFTER the edit, because the model is
causal and a window ending at the edit would report zero collateral by
construction.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("on_chart", "#2ca02c", "on-chart (dose in radians)"),
        ("radial", "#ff7f0e", "radial (same plane, off the curve)"),
        ("off_chart", "#d62728", "off-chart (orthogonal, matched norm)")]


def load(path):
    rows, fit = [], None
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("record") == "chart_dose":
                rows.append(r)
            elif r.get("record") == "chart_fit":
                fit = r
    return rows, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rows, fit = load(args.chart)
    if not rows:
        raise SystemExit("no chart_dose records")
    dthetas = sorted({r["dtheta"] for r in rows})
    n_rows = len({r["row"] for r in rows})

    def series(arm, field):
        out = []
        for d in dthetas:
            v = [r["arms"][arm][field] for r in rows if r["dtheta"] == d]
            out.append((np.median(v), np.percentile(v, 25), np.percentile(v, 75)))
        return np.array(out)

    summary = {"n_rows": n_rows, "n_records": len(rows), "dthetas": dthetas,
               "chart_fit": fit, "provenance": rows[0].get("provenance")}
    for arm, _c, _l in ARMS:
        summary[arm] = {
            "kl_target_median": series(arm, "kl_target")[:, 0].tolist(),
            "kl_other_mean_median": series(arm, "kl_other_mean")[:, 0].tolist(),
            "class_logprob_shift_median": series(arm, "class_logprob_shift")[:, 0].tolist(),
        }
    # collateral per unit of intended effect, pooled over doses
    ratios = {}
    for arm, _c, _l in ARMS:
        num = np.array([r["arms"][arm]["kl_other_mean"] for r in rows])
        den = np.array([r["arms"][arm]["kl_target"] for r in rows])
        m = den > 0
        ratios[arm] = float(np.median(num[m] / den[m]))
    summary["collateral_per_unit_effect_median"] = ratios
    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "provenance"}, indent=2, sort_keys=True)[:2500])

    fig, axes = plt.subplots(1, 4, figsize=(21.5, 5.1))

    ax = axes[0]
    for arm, col, lab in ARMS:
        s = series(arm, "kl_target")
        ax.plot(dthetas, s[:, 0], color=col, lw=2, marker="o", ms=4, label=lab)
        ax.fill_between(dthetas, s[:, 1], s[:, 2], color=col, alpha=0.15)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"chart dose $\Delta\theta$ (radians)")
    ax.set_ylabel("exact KL at the edited position (nats)")
    ax.set_title("A. Dose in radians: intended effect\n"
                 "all three arms at MATCHED ambient step length")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for arm, col, lab in ARMS:
        s = series(arm, "kl_other_mean")
        ax.plot(dthetas, s[:, 0], color=col, lw=2, marker="o", ms=4, label=lab)
        ax.fill_between(dthetas, s[:, 1], s[:, 2], color=col, alpha=0.15)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"chart dose $\Delta\theta$ (radians)")
    ax.set_ylabel("mean KL at the FOLLOWING positions (nats)")
    ax.set_title("B. Collateral damage\n"
                 "(positions after the edit; causality forbids earlier ones)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for arm, col, lab in ARMS:
        x = np.array([r["arms"][arm]["kl_target"] for r in rows])
        y = np.array([r["arms"][arm]["kl_other_mean"] for r in rows])
        m = (x > 0) & (y > 0)
        ax.scatter(x[m], y[m], s=13, color=col, alpha=0.45, label=lab)
    lim = [1e-8, 30]
    ax.plot(lim, lim, "k--", lw=1.2, label="collateral = intended effect")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_xlabel("intended effect: KL at the edited position (nats)")
    ax.set_ylabel("collateral: mean KL downstream (nats)")
    ax.set_title("C. The steering trade-off curve\n"
                 "lower-right is better: more effect per unit damage")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3, which="both")

    ax = axes[3]
    for arm, col, lab in ARMS:
        s = series(arm, "class_logprob_shift")
        ax.plot(dthetas, s[:, 0], color=col, lw=2, marker="o", ms=4, label=lab)
        ax.fill_between(dthetas, s[:, 1], s[:, 2], color=col, alpha=0.15)
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"chart dose $\Delta\theta$ (radians)")
    ax.set_ylabel("mean log-prob shift on the chart's own token class")
    ax.set_title("D. Specificity\n"
                 "does moving along the chart move ITS class?")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    fig.suptitle("gam#2234 — chart-coordinate steering on real activations "
                 "(Qwen3.5-4B-Base L16, month-token circle chart, declared doc "
                 "split 6e3dbf8dcb8164bebc9d58eaacb56067, A10)",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=125, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
