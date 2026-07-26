"""#2502 steering figures from stage-B records (per cycle)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLAG = os.path.expanduser("~/i2502/flagship")
ARM_COLOR = {"manifold": "#0072B2", "flat": "#D55E00"}
ARM_LABEL = {"manifold": "on-manifold (circle-atom phase move)",
             "flat": "torch TopK-SAE latent direction (matched norm)"}
WEEK = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
MON = ["Ja", "Fe", "Mr", "Ap", "My", "Jn", "Jl", "Au", "Se", "Oc", "No", "De"]

records = [json.loads(x) for x in open(f"{FLAG}/steer_records.jsonl")]
sm = json.load(open(f"{FLAG}/steer_meta.json"))

for cyc, n, ticks in (("week", 7, WEEK), ("month", 12, MON)):
    rs_all = [r for r in records if r["cycle"] == cyc]
    if not rs_all:
        continue
    doses = sorted({r["dose_fraction"] for r in rs_all})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm in ("manifold", "flat"):
        m, c = [], []
        for d in doses:
            rs = [r for r in rs_all if r["arm"] == arm and r["dose_fraction"] == d]
            m.append(np.mean([r["target_token_probability"] for r in rs]))
            c.append(np.mean([r["collateral_kl_model_to_base_non_target"] for r in rs]))
        axes[0].plot(doses, m, "o-", color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        axes[1].plot(doses, c, "o-", color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    axes[0].set_xlabel("dose fraction of target phase shift")
    axes[0].set_ylabel("target token probability (full softmax)")
    axes[0].set_title(f"{cyc} steering: mass moves to the target", fontsize=10)
    axes[1].set_xlabel("dose fraction of target phase shift")
    axes[1].set_ylabel("collateral KL(patched ‖ base), target excluded")
    axes[1].set_title("collateral damage", fontsize=10)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    meta = sm[cyc]
    fig.suptitle(f"Qwen3.5-4B-Base L16 — steering the unsupervised {cyc} atom "
                 f"{meta['atom']} (circular R²={meta['r2']:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FLAG}/fig_steer_{cyc}_dose.png", dpi=160)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for arm in ("manifold", "flat"):
        xs = [r["target_probability_mass_moved"] for r in rs_all
              if r["arm"] == arm and r["dose_fraction"] > 0]
        ys = [r["collateral_kl_model_to_base_non_target"] for r in rs_all
              if r["arm"] == arm and r["dose_fraction"] > 0]
        ax.scatter(xs, ys, s=14, alpha=0.5, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.set_xlabel("target probability mass moved")
    ax.set_ylabel("collateral KL (target excluded)")
    ax.set_title(f"{cyc}: achieved effect vs collateral (all doses/shifts)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FLAG}/fig_steer_{cyc}_frontier.png", dpi=160)

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for j, arm in enumerate(("manifold", "flat")):
        M = np.zeros((n, n))
        C = np.zeros((n, n))
        for r in rs_all:
            if r["arm"] != arm or r["dose_fraction"] != 1.0:
                continue
            M[r["target_day_index"], r["realized_top_weekday_index"]] += 1
            C[r["target_day_index"], :] += 1.0 / n
        M = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
        im = axs[j].imshow(M, cmap="Blues", vmin=0, vmax=1)
        axs[j].set_xticks(range(n), ticks, fontsize=7)
        axs[j].set_yticks(range(n), ticks, fontsize=7)
        axs[j].set_xlabel("realized top candidate")
        axs[j].set_ylabel("steering target")
        axs[j].set_title(ARM_LABEL[arm], fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.8, label="fraction of contexts")
    fig.suptitle(f"{cyc}: full-dose steering — where the model's top prediction lands",
                 fontsize=10)
    fig.savefig(f"{FLAG}/fig_steer_{cyc}_rotation.png", dpi=160)
print("steer figs done")
