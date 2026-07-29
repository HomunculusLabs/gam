import re, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2 = os.path.expanduser("~/i2502v2")

def alpha_med(log):
    out = []
    pat = re.compile(r"alpha min/med/max = [0-9.e+-]+/([0-9.e+-]+)/")
    for line in open(os.path.join(V2, log), errors="ignore"):
        if "REML round" in line and (m := pat.search(line)):
            out.append(float(m.group(1)))
    return out

GOOD, BAD, REF = "#1a7a3a", "#c0392b", "#7f8c8d"

# ---- Fig 1 v2: the ladder story, with the verdict IN the picture ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
for log, label, color in [("g808.log", "old: guarded rails", BAD),
                          ("q808.log", "attempt: pooled prior", "#e67e22"),
                          ("q2_808.log", "fix: EM second moment", GOOD)]:
    ys = alpha_med(log)
    ax1.semilogy(range(1, len(ys) + 1), ys, "o-", color=color, lw=2.5, label=label)
ax1.axhspan(50, 2e3, color=BAD, alpha=0.08)
ax1.text(1.1, 300, "RUNAWAY ZONE: prior strangles the\ncoordinates, held-out fit dies",
         fontsize=9, color=BAD)
ax1.axhspan(1e-5, 10, color=GOOD, alpha=0.07)
ax1.text(2.6, 3e-4, "HEALTHY: prior stays weaker than\nthe data evidence", fontsize=9, color=GOOD)
ax1.set_xlabel("training round"); ax1.set_ylabel("prior strength alpha (median, log)")
ax1.set_title("WHAT WENT WRONG: the prior-strength update\nfed on its own shrinkage (up = worse)")
ax1.legend(loc="center right", fontsize=9)
names = ["old\nguarded", "pooled\nattempt", "no ladder\n(reference)", "EM fix"]
vals = [0.3558, 0.5247, 0.7717, 0.7839]
cols = [BAD, "#e67e22", REF, GOOD]
bars = ax2.bar(names, vals, color=cols)
ax2.axhline(0.7717, color=REF, ls="--", lw=1)
ax2.text(3.45, 0.757, "reference", fontsize=8, color=REF, ha="right")
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=11)
ax2.annotate("only the EM fix BEATS\nthe no-ladder reference", xy=(3, 0.7839),
             xytext=(1.6, 0.86), fontsize=10, color=GOOD,
             arrowprops=dict(arrowstyle="->", color=GOOD))
ax2.set_ylim(0, 0.95); ax2.set_ylabel("held-out variance explained (higher = better)")
ax2.set_title("THE VERDICT: same data, same size,\nonly the update rule differs")
fig.tight_layout(); fig.savefig(f"{V2}/ladder_story.png", dpi=160); print("fig1")

# ---- Fig 2 v2: two families, the good corner shaded ----
ours = [(0.8530, 0.00934), (0.8563, 0.01051), (0.8573, 0.01124), (0.8330, 0.01199)]
steel = [(0.8598, 0.01097), (0.8598, 0.01235), (0.8598, 0.00928),
         (0.8492, 0.01495), (0.8492, 0.01566), (0.8492, 0.01214)]
broken = [(0.6294, 0.06167), (0.6243, 0.04999), (0.6140, 0.05157), (0.5907, 0.12916)]
fig, ax = plt.subplots(figsize=(9.5, 7))
ax.axvspan(0.82, 0.88, ymin=0, ymax=0.45, color=GOOD, alpha=0.06)
ax.text(0.845, 0.0075, "GOOD CORNER:\nreconstructs well AND\nbarely disturbs the model",
        fontsize=10, color=GOOD, ha="center")
ax.scatter(*zip(*ours), s=90, color=GOOD, label="manifold SAE (ours)", zorder=4)
ax.scatter(*zip(*steel), s=90, marker="^", color="#555555", label="tuned flat SAE (steelman)", zorder=4)
ax.scatter(*zip(*broken), s=70, marker="x", color=BAD, label="our broken/superseded arms", zorder=3)
ax.annotate("our K=8096: ties the BEST steelman\nseed causally, beats the other five",
            xy=(0.8530, 0.00934), xytext=(0.70, 0.006), fontsize=9, color=GOOD,
            arrowprops=dict(arrowstyle="->", color=GOOD))
ax.annotate("typical steelman", xy=(0.8492, 0.01495), xytext=(0.87, 0.02),
            fontsize=9, color="#555555", arrowprops=dict(arrowstyle="->", color="#555555"))
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_xlabel("held-out variance explained (right = better)")
ax.set_ylabel("causal damage when spliced into the model (up = better)")
ax.set_title("Every dictionary we measured: the manifold family\nsits on the gentle frontier")
ax.legend(loc="lower left", fontsize=10); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(f"{V2}/ev_vs_causal.png", dpi=160); print("fig2")
