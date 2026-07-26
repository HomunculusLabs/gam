"""Word-free 5:2 image from REAL fitted atoms (#2502).

Geometry is real: curves are the Rust model's decoded γ_k(t) sampled on a
grid; dots are the actual Qwen3.5-4B L16 chart rows that use each atom,
projected into that atom's own top-2 plane (fit on the curve); the background
starfield is the real chart cloud under a global PCA-2. Layout/glow are the
only aesthetic choices.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUMP = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/i2502/dump_k256")
CHART = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/i2502/prep_L16_p128/train.npy")
NROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
OUT = sys.argv[4] if len(sys.argv) > 4 else os.path.expanduser("~/i2502/manifold_real.png")

man = json.load(open(f"{DUMP}/manifest.json"))
X = np.load(CHART)[:NROWS]
mean = np.fromfile(f"{DUMP}/mean.bin")
P = len(mean)
Xc = X - mean

INK = "#0a0f1e"
fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
fig.patch.set_facecolor(INK)
ax.set_facecolor(INK)
ax.set_xlim(0, 15)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

rng = np.random.default_rng(3)

# real starfield: global PCA-2 of the actual chart cloud
U, S, Vt = np.linalg.svd(Xc[rng.choice(len(Xc), 1500, replace=False)], full_matrices=False)
star = U[:, :2] * S[:2]
star = star / np.abs(star).max(0)
ax.scatter(7.5 + 7.2 * star[:, 0], 3.0 + 2.8 * star[:, 1], color="#93a5cf",
           s=1.8, alpha=0.10, lw=0, zorder=1)


def glow(x, y, color, lw=2.2, zo=5, a=0.95):
    for w, al in ((lw * 7, 0.035), (lw * 4, 0.07), (lw * 2.2, 0.16), (lw, a)):
        ax.plot(x, y, color=color, lw=w, alpha=al, solid_capstyle="round", zorder=zo)


KIND_STYLE = {
    "linear": (["#69d2e7", "#5fd7a7"], "winter"),
    "euclidean": (["#a8e05f", "#ffd166"], "summer"),
    "periodic": (["#c9a0ff", "#ff6b9d"], "twilight"),
}

# panel slots across the 5:2 canvas (cx, cy, half-width)
slots = [(1.55, 4.2, 1.15), (1.75, 1.35, 1.15), (4.55, 2.9, 1.35),
         (7.5, 4.1, 1.35), (7.5, 1.35, 1.35), (10.45, 2.9, 1.35),
         (13.3, 4.2, 1.15), (13.3, 1.35, 1.15)]

# order panels: mix kinds across the canvas, most-used first within kind
by_kind = {}
for entry in man:
    by_kind.setdefault(entry["kind"], []).append(entry)
ordered = []
while any(by_kind.values()):
    for kind in ("periodic", "euclidean", "linear"):
        if by_kind.get(kind):
            ordered.append(by_kind[kind].pop(0))
ordered = ordered[: len(slots)]

for slot_idx, entry in enumerate(ordered):
    cx, cy, half = slots[slot_idx]
    idx = entry["idx"]
    kind = entry["kind"]
    curve = np.fromfile(f"{DUMP}/curve_{idx}.bin").reshape(-1, P)
    toks = np.fromfile(f"{DUMP}/tokens_{idx}.bin").reshape(-1, 3)
    rows = toks[:, 0].astype(int)
    tvals = toks[:, 1]
    cap = min(len(rows), 140)
    keep = rng.choice(len(rows), cap, replace=False)
    rows, tvals = rows[keep], tvals[keep]

    # atom-local plane: top-2 PCA of the decoded curve
    c0 = curve.mean(0)
    _, _, vt = np.linalg.svd(curve - c0, full_matrices=False)
    plane = vt[:2] if vt.shape[0] > 1 else np.vstack([vt[0], rng.normal(0, 1, P)])
    crv2 = (curve - c0) @ plane.T
    pts2 = (Xc[rows] - c0) @ plane.T

    # scale panel to slot
    span = max(np.abs(np.vstack([crv2, pts2])).max(), 1e-9)
    scale = half / span
    crv2, pts2 = crv2 * scale, pts2 * scale

    colors, cmap = KIND_STYLE[kind]
    if kind == "periodic":
        ph = tvals % 1.0
        cols = plt.get_cmap("twilight")(np.linspace(0, 1, len(crv2)))
        ax.plot(cx + crv2[:, 0], cy + crv2[:, 1], color=colors[0], lw=12,
                alpha=0.05, zorder=4)
        for j in range(len(crv2) - 1):
            ax.plot(cx + crv2[j:j + 2, 0], cy + crv2[j:j + 2, 1],
                    color=cols[j], lw=2.6, alpha=0.95, zorder=5)
        cvals = ph
    else:
        glow(cx + crv2[:, 0], cy + crv2[:, 1], colors[slot_idx % 2], lw=2.4)
        cvals = tvals
    ax.scatter(cx + pts2[:, 0], cy + pts2[:, 1], c=cvals, cmap=cmap, s=20,
               alpha=0.9, lw=0.45, edgecolors=INK, zorder=7)

fig.savefig(OUT, facecolor=INK, dpi=200)
print("saved", OUT, "panels:", [(e["kind"], e["usage"]) for e in ordered])
