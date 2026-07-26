#!/usr/bin/env python3
"""#2283 — turn the readability ladder into the one number the hybrid row needs.

The crossover corollary predicts a support-term advantage of ~14-20 bits at
K=32768 / top_k=32.  This reads, from the measured rows, how many bits the
SAME arm moves for reasons the theorem does not speak about: the training seed,
the training budget, and the estimation subsample.  A margin smaller than those
cannot be read off a single paired row.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

TARGET = "bits_at_r2_0.99"


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            recs.append(json.loads(line))
    return recs


def bits(rec, key=TARGET):
    return rec["bits"][key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    recs = load(args.main)
    seeds = [r for r in recs if r.get("record") == "seed_row"]
    ladder = [r for r in recs if r.get("record") == "ladder_row"]
    alphas = [r for r in recs if r.get("record") == "alpha_row"]
    whole = [r for r in recs if r.get("record") == "whole_split_row"]
    theorem = [r for r in recs if r.get("record") == "theorem_margin"]

    out = {"provenance": recs[0].get("provenance"),
           "data_identity": recs[0].get("data_identity"),
           "config": recs[0].get("config")}

    if theorem:
        out["theorem_margin_bits"] = theorem[0]["margin_by_charts"]
        out["support_bits_external"] = theorem[0]["support_bits_external"]

    if seeds:
        b = np.array([bits(r) for r in seeds])
        ev = np.array([r["ev_whole_split"] for r in seeds])
        out["seed_rows"] = [
            {"seed": r["seed"], "ev_whole_split": r["ev_whole_split"],
             "ev_bits_rows": r.get("ev_bits_rows"),
             "bits_at_r2_0.99": bits(r),
             "dictionary": r["bits"].get("dictionary_bits"),
             "support": r["bits"].get("support_bits"),
             "code": r["bits"].get("code_bits_at_r2_0.99"),
             "resid": r["bits"].get("resid_bits_at_r2_0.99"),
             "fit_seconds": r.get("fit_seconds")}
            for r in seeds]
        out["seed_spread"] = {
            "n": int(b.size), "min": float(b.min()), "max": float(b.max()),
            "range_bits": float(b.max() - b.min()),
            "sd_bits": float(b.std(ddof=1)) if b.size > 1 else None,
            "ev_range": float(ev.max() - ev.min()),
        }

    if ladder:
        rows = sorted(ladder, key=lambda r: r["steps"])
        out["ladder_rows"] = [
            {"steps": r["steps"], "ev_whole_split": r["ev_whole_split"],
             "ev_bits_rows": r.get("ev_bits_rows"),
             "bits_at_r2_0.99": bits(r),
             "resid": r["bits"].get("resid_bits_at_r2_0.99"),
             "code": r["bits"].get("code_bits_at_r2_0.99")}
            for r in rows]

    # d(bits)/d(EV) from the REAL refits: pool the ladder with the seed rows
    # (all the same architecture and dictionary size, differing only in how
    # good the fit is).
    pool = [(r.get("ev_bits_rows"), bits(r)) for r in ladder + seeds
            if r.get("ev_bits_rows") is not None]
    if len(pool) >= 3:
        x = np.array([p[0] for p in pool])
        y = np.array([p[1] for p in pool])
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        out["bits_vs_ev"] = {
            "n": len(pool),
            "slope_bits_per_ev": float(slope),
            "ev_span": [float(x.min()), float(x.max())],
            "bits_span": [float(y.min()), float(y.max())],
            "rms_residual_bits": float(np.sqrt(np.mean(resid ** 2))),
            "points": [{"ev": float(a), "bits": float(b_)} for a, b_ in pool],
        }
        if theorem:
            for charts, margin in theorem[0]["margin_by_charts"].items():
                out.setdefault("ev_parity_required", {})[charts] = {
                    "margin_bits": margin,
                    "delta_ev_worth_the_margin": float(abs(margin / slope)),
                }

    if alphas:
        rows = sorted(alphas, key=lambda r: r["alpha"])
        out["alpha_rows"] = [
            {"alpha": r["alpha"], "ev_bits_rows": r["ev_bits_rows"],
             "bits_at_r2_0.99": bits(r),
             "resid": r["bits"].get("resid_bits_at_r2_0.99"),
             "code": r["bits"].get("code_bits_at_r2_0.99"),
             "support": r["bits"].get("support_bits"),
             "dictionary": r["bits"].get("dictionary_bits")}
            for r in rows]
        b1 = [r for r in rows if abs(r["alpha"] - 1.0) < 1e-12]
        if b1:
            base = bits(b1[0])
            out["alpha_rows_delta_bits"] = [
                {"alpha": r["alpha"], "delta_bits": bits(r) - base}
                for r in rows]

    if whole and seeds:
        s0 = [r for r in seeds if r["seed"] == whole[0]["seed"]]
        if s0:
            out["estimation_sample_effect"] = {
                "seed": whole[0]["seed"],
                "bits_rows_subsample": s0[0]["bits_rows"],
                "bits_subsample": bits(s0[0]),
                "bits_rows_whole_split": whole[0]["bits_rows"],
                "bits_whole_split": bits(whole[0]),
                "delta_bits": bits(whole[0]) - bits(s0[0]),
                "dictionary_identical": (
                    whole[0]["bits"].get("dictionary_bits")
                    == s0[0]["bits"].get("dictionary_bits")),
                "support_identical": (
                    whole[0]["bits"].get("support_bits")
                    == s0[0]["bits"].get("support_bits")),
            }

    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))

    if args.out_png and "bits_vs_ev" in out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        ax = axes[0]
        pts = out["bits_vs_ev"]["points"]
        x = np.array([p["ev"] for p in pts])
        y = np.array([p["bits"] for p in pts])
        ax.scatter(x, y, s=64, color="#1f77b4", zorder=3, label="real refits")
        xs = np.linspace(x.min(), x.max(), 50)
        sl = out["bits_vs_ev"]["slope_bits_per_ev"]
        ic = y.mean() - sl * x.mean()
        ax.plot(xs, sl * xs + ic, "k--", lw=1.5,
                label=f"{sl:,.0f} bits per EV point")
        if "seed_spread" in out:
            sr = out["seed_spread"]["range_bits"]
            ax.annotate(f"seed spread at fixed budget: {sr:,.0f} bits",
                        xy=(0.03, 0.9), xycoords="axes fraction", fontsize=10)
        ax.set_xlabel("held-out EV (bits rows)")
        ax.set_ylabel(r"bits at $R^2=0.99$")
        ax.set_title("#2283: the Eq-4 scoreboard vs fit quality\n"
                     "Qwen3.5-4B-Base L16, declared doc split")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        ax = axes[1]
        labels, values = [], []
        if "seed_spread" in out:
            labels.append("seed spread\n(same config)")
            values.append(out["seed_spread"]["range_bits"])
        if "estimation_sample_effect" in out:
            labels.append("estimation sample\n(24k vs whole split)")
            values.append(abs(out["estimation_sample_effect"]["delta_bits"]))
        if "ladder_rows" in out:
            lb = [r["bits_at_r2_0.99"] for r in out["ladder_rows"]]
            sb = [r["bits_at_r2_0.99"] for r in out.get("seed_rows", [])]
            allb = lb + sb
            labels.append("training budget\n(1k vs 8k steps)")
            values.append(max(allb) - min(allb))
        if theorem:
            for charts in ("8", "32", "256"):
                if charts in out["theorem_margin_bits"]:
                    labels.append(f"THEOREM margin\n({charts} charts)")
                    values.append(out["theorem_margin_bits"][charts])
        colors = ["#d62728"] * (len(values) - 3) + ["#2ca02c"] * 3
        ax.barh(labels, values, color=colors[:len(values)])
        ax.set_xscale("log")
        ax.set_xlabel("bits")
        ax.set_title("What the row must resolve vs what it actually moves\n"
                     "(green = the effect being tested; red = everything else)")
        ax.grid(alpha=0.3, axis="x", which="both")
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=130)
        print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
