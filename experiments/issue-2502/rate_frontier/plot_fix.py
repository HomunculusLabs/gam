import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUT, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig = plt.figure(figsize=(13.5, 9.2))
gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.42, left=0.07, right=0.97, top=0.90, bottom=0.07)

# A: depth curves, uniform inference
axA = fig.add_subplot(gs[0, 0])
d = [40, 60, 80, 300]
flat = {40: [0.851124, 0.851831, 0.851996], 60: [0.855672, 0.856407, 0.856695],
        80: [0.857131, 0.857461, 0.857857], 300: [0.849063, 0.848574, 0.848084]}
field = {40: [0.852297, 0.851895, 0.851513], 60: [0.856049, 0.856007, 0.855304],
         80: [0.856183, 0.855426, 0.855504], 300: [0.847866, 0.846948, 0.847964]}
# d40 numbers are trainer-protocol; d60+ uniform -- keep one protocol per point set:
d = [60, 80, 300]
xpos = [0, 1, 2]
for arr, c, lbl in ((flat, BLUE, "flat TopK"), (field, ORANGE, "curvature field (fixed)")):
    means = [np.mean(arr[k]) for k in d]
    for i, k in enumerate(d):
        axA.plot([xpos[i]] * 3, arr[k], "o", ms=5, color=c, alpha=0.45, zorder=2)
    axA.plot(xpos, means, "-o", ms=8, color=c, label=lbl, zorder=3)
axA.set_xticks(xpos, ["60 ep", "80 ep", "300 ep"])
axA.set_ylabel("chart EV, uniform inference")
axA.legend(frameon=False, loc="lower left")
axA.set_title("A — the depth frontier: both peak near 80 epochs,\n300 epochs over-trains both (cosine horizon)", fontsize=11, loc="left")

# B: paired diffs under uniform inference
axB = fig.add_subplot(gs[0, 1])
diffs = {"60": [-0.000377, 0.000399, 0.001391], "80": [0.000948, 0.002035, 0.002353],
         "300": [0.001196, 0.001625, 0.000120]}
for i, (k, v) in enumerate(diffs.items()):
    axB.plot([x * 1e3 for x in v], [i] * 3, "o", ms=8, color=INK, alpha=0.8)
    axB.plot([np.mean(v) * 1e3], [i], "D", ms=10, color=BLUE, zorder=3)
axB.axvline(0, color=MUT, lw=1)
axB.set_yticks(range(3), ["60 ep", "80 ep", "300 ep"])
axB.set_xlabel("flat − field  (×10⁻³ EV), identical inference for both")
axB.set_title("B — flat's edge survives protocol-uniform eval:\n8/9 pairs positive, ~+0.001 mean (diamond)", fontsize=11, loc="left")

# C: causal splice
axC = fig.add_subplot(gs[1, 0])
fl = [0.041954, 0.042059, 0.042082]
ff = [0.040373, 0.042035, 0.042282]
for s in range(3):
    axC.plot([0, 1], [fl[s] * 1e3, ff[s] * 1e3], "-", color=MUT, lw=1, alpha=0.6)
    axC.plot([0], [fl[s] * 1e3], "o", ms=9, color=BLUE)
    axC.plot([1], [ff[s] * 1e3], "o", ms=9, color=ORANGE)
axC.set_xticks([0, 1], ["flat", "field"])
axC.set_xlim(-0.5, 1.5)
axC.set_ylabel("splice ΔCE  (×10⁻³ nats)")
axC.set_title("C — the causal court: statistical tie\n(identity control exactly 0.0; 71,424 held-out tokens)", fontsize=11, loc="left")

# D: gamma ablation
axD = fig.add_subplot(gs[1, 1])
lab = ["flat", "field", "field, γ→0\n(curvature ablated)"]
vals = [np.mean([0.800215, 0.800903, 0.800860]), np.mean([0.797907, 0.798124, 0.797247]),
        np.mean([0.750238, 0.749503, 0.742293])]
cols = [BLUE, ORANGE, MUT]
for i, (v, c) in enumerate(zip(vals, cols)):
    axD.bar([i], [v - 0.70], bottom=0.70, width=0.55, color=c, edgecolor=SURF, linewidth=2)
    axD.annotate(f"{v:.4f}", (i, v + 0.002), ha="center", fontsize=10, color=INK)
axD.set_xticks(range(3), lab)
axD.set_ylim(0.70, 0.82)
axD.set_ylabel("chart EV (encoder protocol)")
axD.set_title("D — the curvature is load-bearing: ablating γ\ncosts 0.05 EV; 98.9% of atoms elect curvature", fontsize=11, loc="left")

fig.suptitle("The fix campaign — defects repaired (γ-init, dead atoms, eval asymmetry), and what remains is real",
             fontsize=13.5, x=0.07, ha="left", fontweight="bold")
fig.savefig("fix_campaign.png", dpi=160)
print("FIG_OK")
