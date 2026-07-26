"""Plots for E1 weekday-circle steering (#2502): dose-response + collateral.

Reads e1_records.jsonl / e1_summary.json, writes PNGs into --out-dir.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_COLOR = {"manifold": "#0072B2", "flat": "#D55E00"}
ARM_LABEL = {"manifold": "on-manifold (circle atom)", "flat": "flat SAE direction (matched norm)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="e1_out")
    args = ap.parse_args()
    out = Path(args.out_dir)
    records = [json.loads(x) for x in (out / "e1_records.jsonl").read_text().splitlines()]
    summary = json.loads((out / "e1_summary.json").read_text())
    meta = summary["meta"]

    doses = sorted({r["dose_fraction"] for r in records})
    shifts = sorted({r["target_shift_days"] for r in records})

    # 1. dose-response: mean target-token probability vs dose, per arm
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for arm in ("manifold", "flat"):
        m, c = [], []
        for d in doses:
            rs = [r for r in records if r["arm"] == arm and r["dose_fraction"] == d]
            m.append(np.mean([r["target_token_probability"] for r in rs]))
            c.append(np.mean([r["collateral_kl_model_to_base_non_target"] for r in rs]))
        axes[0].plot(doses, m, "o-", color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        axes[1].plot(doses, c, "o-", color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    axes[0].set_xlabel("dose fraction of target phase shift")
    axes[0].set_ylabel("target-day token probability (full softmax)")
    axes[0].set_title("steering moves mass to the target weekday")
    axes[1].set_xlabel("dose fraction of target phase shift")
    axes[1].set_ylabel("collateral KL(patched ‖ base), target excluded")
    axes[1].set_title("collateral damage")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{meta['model']} L{meta['layer_index']} — weekday circle steering "
                 f"(fit EV {meta['fit_ev']:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "e1_dose_response.png", dpi=160)

    # 2. efficiency frontier: mass moved vs collateral (endpoint doses, all shifts)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for arm in ("manifold", "flat"):
        xs, ys = [], []
        for s in shifts:
            for d in doses:
                if d == 0.0:
                    continue
                rs = [r for r in records if r["arm"] == arm
                      and r["dose_fraction"] == d and r["target_shift_days"] == s]
                xs.append(np.mean([r["target_probability_mass_moved"] for r in rs]))
                ys.append(np.mean([r["collateral_kl_model_to_base_non_target"] for r in rs]))
        ax.scatter(xs, ys, s=22, color=ARM_COLOR[arm], label=ARM_LABEL[arm], alpha=0.8)
    ax.set_xlabel("target probability mass moved")
    ax.set_ylabel("collateral KL (target excluded)")
    ax.set_title("achieved effect vs collateral, per (shift, dose)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "e1_frontier.png", dpi=160)

    # 3. rotation matrix: endpoint realized-top-weekday vs target (manifold arm)
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for j, arm in enumerate(("manifold", "flat")):
        M = np.zeros((7, 7))
        cnt = np.zeros((7, 7))
        for r in records:
            if r["arm"] != arm or r["dose_fraction"] != 1.0:
                continue
            M[r["target_day_index"], r["realized_top_weekday_index"]] += 1
            cnt[r["target_day_index"], :] += 1 / 7
        with np.errstate(invalid="ignore"):
            M = M / np.maximum(cnt.sum(1, keepdims=True) * 7 / 7, 1e-9)
        im = axs[j].imshow(M, cmap="Blues", vmin=0, vmax=1)
        axs[j].set_xticks(range(7), ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
        axs[j].set_yticks(range(7), ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
        axs[j].set_xlabel("realized top weekday")
        axs[j].set_ylabel("target weekday")
        axs[j].set_title(ARM_LABEL[arm], fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.8, label="fraction of contexts")
    fig.suptitle("full-dose steering: where does the model's top weekday land?", fontsize=10)
    fig.savefig(out / "e1_rotation.png", dpi=160)
    print("wrote", out / "e1_dose_response.png", out / "e1_frontier.png",
          out / "e1_rotation.png")


if __name__ == "__main__":
    main()
