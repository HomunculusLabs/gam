#!/usr/bin/env python3
"""#2263 figure: the real-activation dose law, and the arithmetic that hides it."""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    out = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("record") == "dose":
                out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    arms = {"bf16 (8-bit mantissa)": (load(args.bf16), "#d62728"),
            "fp16 (11-bit)": (load(args.fp16), "#ff7f0e"),
            "fp32 (24-bit)": (load(args.fp32), "#1f77b4")}

    fig, axes = plt.subplots(1, 4, figsize=(21.5, 5.1))

    # ---- A: the clean fp32 dose curves -----------------------------------
    ax = axes[0]
    recs32 = arms["fp32 (24-bit)"][0]
    rhos = np.array(recs32[0]["rhos"], dtype=float)
    for r in recs32:
        k = np.array(r["kl"], dtype=float)
        m = k > 0
        ax.plot(rhos[m], k[m],
                color="#1f77b4" if r["family"] == "pca" else "#9467bd",
                alpha=0.25, lw=0.9)
    allk = np.array([r["kl"] for r in recs32], dtype=float)
    ref = np.median(allk[:, -1]) * (rhos / rhos[-1]) ** 2
    ax.plot(rhos, ref, "k--", lw=2.2, label=r"$\mathrm{KL}\propto\rho^{2}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"dose  $\rho=\|\delta\|/\|x\|$")
    ax.set_ylabel("exact readout KL (nats)")
    ax.set_title("A. Real Qwen3.5-4B-Base L16, fp32\n"
                 "exact softmax-to-softmax KL, 48 rows x 8 directions")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3, which="both")

    # ---- B: local slope, all three mantissa widths ------------------------
    ax = axes[1]
    mid = np.sqrt(rhos[:-1] * rhos[1:])
    for label, (recs, col) in arms.items():
        sl = np.array([r["loglog_slopes"] for r in recs], dtype=float)
        med = np.nanmedian(sl, axis=0)
        q1 = np.nanpercentile(sl, 25, axis=0)
        q3 = np.nanpercentile(sl, 75, axis=0)
        ax.plot(mid, med, color=col, lw=2.2, label=label)
        ax.fill_between(mid, q1, q3, color=col, alpha=0.15)
    ax.axhline(2.0, color="k", ls="--", lw=2)
    ax.axhspan(1.85, 2.15, color="k", alpha=0.07)
    ax.text(mid[0] * 1.1, 2.25, "quadratic law", fontsize=9)
    ax.set_xscale("log"); ax.set_ylim(-1.2, 4.2)
    ax.set_xlabel(r"dose $\rho$")
    ax.set_ylabel(r"$d\log \mathrm{KL}\,/\,d\log\rho$")
    ax.set_title("B. The small-dose floor is ARITHMETIC\n"
                 "the slope leaves 2 at a dose set by the mantissa")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3, which="both")

    # ---- C: lower edge of the quadratic window vs mantissa ----------------
    ax = axes[2]
    labels, lows, nones = [], [], []
    for label, (recs, col) in arms.items():
        w = [r["quadratic_window_rho"][0] for r in recs
             if r.get("quadratic_window_rho")]
        labels.append(label.split(" ")[0])
        lows.append(np.median(w) if w else np.nan)
        nones.append(100.0 * sum(1 for r in recs
                                 if not r.get("quadratic_window_rho")) / len(recs))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, lows, width=0.55,
                  color=["#d62728", "#ff7f0e", "#1f77b4"])
    ax.set_yscale("log")
    ax.set_ylim(min(lows) / 3.0, max(lows) * 6.0)
    for x, lo, nn in zip(xs, lows, nones):
        ax.text(x, lo * 1.25, f"{lo:g}", ha="center", fontsize=11,
                fontweight="bold")
        ax.text(x, lo * 2.6, f"{nn:.0f}% of directions\nhave NO window",
                ha="center", fontsize=8.5)
    ax.set_ylabel(r"lower edge of the quadratic window, $\rho_{\mathrm{lo}}$"
                  "   (median)")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("model dtype (mantissa bits: 8 / 11 / 24)")
    ax.set_title("C. 50x in the window's floor, 45% -> 0% unusable\n"
                 "same rows, same directions, only the dtype changes")
    ax.grid(alpha=0.3, axis="y", which="both")

    # ---- D: the dose coefficient is not one number -----------------------
    ax = axes[3]
    for fam, col in (("pca", "#1f77b4"), ("random", "#9467bd")):
        cs = np.array([r["c_hat"] for r in recs32
                       if r["family"] == fam and r.get("c_hat")], dtype=float)
        if cs.size:
            ax.hist(np.log10(cs), bins=16, color=col, alpha=0.6,
                    label=f"{fam}  (n={cs.size}, spread "
                          f"{cs.max()/cs.min():.0f}x)")
    ax.set_xlabel(r"$\log_{10}$  dose coefficient  $c=d^{\top}F_h d$")
    ax.set_ylabel("count")
    ax.set_title("D. One dose is not one KL\n"
                 "c spans orders of magnitude across rows (fp32)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("gam#2263 — steering dosimetry measured on real activations "
                 "(Qwen3.5-4B-Base L16, declared doc split "
                 "6e3dbf8dcb8164bebc9d58eaacb56067, A10 CUDA_VISIBLE_DEVICES=0)",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=125, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
