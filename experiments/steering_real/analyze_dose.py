#!/usr/bin/env python3
"""#2263 — read the real-activation dose ladder into the two radii the issue asks for."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np


def load(path):
    out = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("record") == "dose":
                out.append(r)
    return out


def summarize(recs):
    by_fam = defaultdict(list)
    for r in recs:
        by_fam[r["family"]].append(r)
    lines = []
    for fam, rs in sorted(by_fam.items()):
        wins = [r["quadratic_window_rho"] for r in rs if r.get("quadratic_window_rho")]
        cs = [r["c_hat"] for r in rs if r.get("c_hat")]
        n_none = sum(1 for r in rs if not r.get("quadratic_window_rho"))
        if wins:
            lo = np.array([w[0] for w in wins])
            hi = np.array([w[1] for w in wins])
            dec = np.log10(hi / lo)
        else:
            lo = hi = dec = np.array([np.nan])
        cs = np.array(cs) if cs else np.array([np.nan])
        lines.append({
            "family": fam, "n": len(rs), "n_no_window": n_none,
            "rho_lo_median": float(np.median(lo)),
            "rho_lo_q1": float(np.percentile(lo, 25)),
            "rho_lo_q3": float(np.percentile(lo, 75)),
            "rho_hi_median": float(np.median(hi)),
            "decades_median": float(np.median(dec)),
            "c_median": float(np.median(cs)),
            "c_min": float(np.min(cs)), "c_max": float(np.max(cs)),
            "c_spread_ratio": float(np.max(cs) / np.min(cs)) if np.min(cs) > 0 else float("nan"),
        })
    return lines


def kl_at(r, rho):
    """Measured KL at a given rho for one record."""
    for j, rr in enumerate(r["rhos"]):
        if abs(rr - rho) < 1e-12:
            return r["kl"][j]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    recs = load(args.dose)
    if not recs:
        raise SystemExit("no dose records")
    summary = summarize(recs)
    rhos = recs[0]["rhos"]

    # readout-KL radius in NATS: the KL at the top of each record's quadratic
    # window is the largest dose whose cost the quadratic predictor still prices
    # correctly. That is the number an acceptance gate should be stated in.
    nats = []
    for r in recs:
        w = r.get("quadratic_window_rho")
        if not w:
            continue
        v = kl_at(r, w[1])
        if v:
            nats.append(v)
    nats = np.array(nats) if nats else np.array([np.nan])

    # identical-pair floor: is the arithmetic floor really zero?
    floors = np.array([r.get("identical_pair_kl", np.nan) for r in recs],
                      dtype=float)

    payload = {
        "n_records": len(recs),
        "rhos": rhos,
        "by_family": summary,
        "readout_kl_radius_nats": {
            "median": float(np.median(nats)), "min": float(np.min(nats)),
            "max": float(np.max(nats)), "n": int(nats.size),
        },
        "identical_pair_kl": {
            "max": float(np.nanmax(floors)), "n_nonzero":
                int(np.sum(np.nan_to_num(floors) > 0)),
        },
        "provenance": recs[0].get("provenance"),
    }
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
        ax = axes[0]
        colors = {"pca": "#1f77b4", "random": "#d62728"}
        for r in recs:
            k = np.array(r["kl"], dtype=float)
            m = k > 0
            ax.plot(np.array(rhos)[m], k[m], color=colors.get(r["family"], "k"),
                    alpha=0.22, lw=0.9)
        # quadratic reference through the median curve's top point
        allk = np.array([r["kl"] for r in recs], dtype=float)
        ref_hi = np.median(allk[:, -1])
        ref = ref_hi * (np.array(rhos) / rhos[-1]) ** 2
        ax.plot(rhos, ref, "k--", lw=2, label=r"exact quadratic law  $\propto \rho^2$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"dose  $\rho = \|\delta\| / \|x\|$")
        ax.set_ylabel("exact readout KL (nats)")
        ax.set_title("Real Qwen3.5-4B-Base L16: measured dose response\n"
                     "(blue = train-only PCA directions, red = isotropic)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3, which="both")

        ax = axes[1]
        for fam, col in colors.items():
            sl = np.array([r["loglog_slopes"] for r in recs
                           if r["family"] == fam], dtype=float)
            if sl.size == 0:
                continue
            mid = np.sqrt(np.array(rhos[:-1]) * np.array(rhos[1:]))
            med = np.nanmedian(sl, axis=0)
            q1 = np.nanpercentile(sl, 25, axis=0)
            q3 = np.nanpercentile(sl, 75, axis=0)
            ax.plot(mid, med, color=col, lw=2, label=f"{fam} (median)")
            ax.fill_between(mid, q1, q3, color=col, alpha=0.18)
        ax.axhline(2.0, color="k", ls="--", lw=2, label="quadratic (slope 2)")
        ax.axhspan(1.85, 2.15, color="k", alpha=0.07)
        ax.set_xscale("log")
        ax.set_ylim(-1, 4.5)
        ax.set_xlabel(r"dose $\rho$ (interval midpoint)")
        ax.set_ylabel(r"local slope  $d\log \mathrm{KL} / d\log \rho$")
        ax.set_title("The quadratic window is BOUNDED BELOW\n"
                     "by the residual stream's own arithmetic")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

        ax = axes[2]
        for fam, col in colors.items():
            cs = np.array([r["c_hat"] for r in recs
                           if r["family"] == fam and r.get("c_hat")],
                          dtype=float)
            if cs.size == 0:
                continue
            ax.hist(np.log10(cs), bins=18, color=col, alpha=0.55,
                    label=f"{fam}  (n={cs.size})")
        ax.set_xlabel(r"$\log_{10}$ dose coefficient  $c = d^{\top} F_h d$")
        ax.set_ylabel("count")
        ax.set_title("One dose does NOT mean one KL:\n"
                     "c varies by orders of magnitude across rows")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(args.out_png, dpi=130)
        print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
