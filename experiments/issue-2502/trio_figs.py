"""Three new interpretability figures from the fitted dictionary.

A. best_fit.png    -- the atom that TRULY fits: highest local R^2 on its own
                      partial residuals; drawn with that data cloud.
B. loop_walk.png   -- a walk around the strongest closed loop: tokens
                      annotated at phases around the ellipse, i.e. what the
                      loop coordinate MEANS.
C. safety.png      -- the atom whose routed tokens concentrate conflict
                      vocabulary (wikitext is full of military history):
                      a safety-relevant subspace, drawn with its charged
                      tokens annotated.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2 = os.path.expanduser("~/i2502v2")
d = sys.argv[1] if len(sys.argv) > 1 else f"{V2}/b_mix1818"
man = json.load(open(os.path.join(d, "manifest.json")))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", trust_remote_code=True)
seqs = np.load(f"{V2}/seqs.npy")
seq_pos = np.load(f"{V2}/train_seq_pos.npy")


def word_at(row):
    s_i, s_p = int(seq_pos[row, 0]), int(seq_pos[row, 1])
    return tok.decode(seqs[s_i][s_p:s_p + 1]).strip() or "SPACE"


def local_r2(atom):
    pp = os.path.join(d, f"partial_{atom['idx']}.bin")
    if not os.path.exists(pp):
        return None
    part = np.fromfile(pp).reshape(-1, 129)
    rows, data = part[:, 0].astype(int), part[:, 1:]
    n = int(atom.get("grid_n", 0)) if atom.get("dim", 1) == 2 else 0
    curve = np.fromfile(os.path.join(d, f"curve_{atom['idx']}.bin")).reshape(-1, 128)
    # nearest decoded point as the atom's prediction for each token
    # (exact would re-decode at token coords; nearest-grid is a tight bound)
    from numpy.linalg import norm
    pred = curve[np.argmin(
        ((data[:, None, :] - curve[None, :, :]) ** 2).sum(-1), axis=1)] \
        if len(curve) * len(data) < 4e7 else None
    if pred is None:
        idx = np.random.default_rng(0).choice(len(data), 400, replace=False)
        data = data[idx]; rows = rows[idx]
        pred = curve[np.argmin(
            ((data[:, None, :] - curve[None, :, :]) ** 2).sum(-1), axis=1)]
    ss_res = ((data - pred) ** 2).sum()
    ss_tot = ((data - data.mean(0)) ** 2).sum()
    amb = 1 - ss_res / max(ss_tot, 1e-30)
    # in-frame R^2: the same quantities inside the surface's own 3-PC frame,
    # i.e. the variance the FIGURE can actually show
    mean = curve.mean(0)
    _, _, vt = np.linalg.svd(curve - mean, full_matrices=False)
    df = (data - mean) @ vt[:3].T
    pf = (pred - mean) @ vt[:3].T
    ss_res_f = ((df - pf) ** 2).sum()
    ss_tot_f = ((df - df.mean(0)) ** 2).sum()
    inf = 1 - ss_res_f / max(ss_tot_f, 1e-30)
    return amb, inf, rows, data, curve


# ---- A: best fitter (champion by IN-FRAME R^2 so picture matches claim) ----
best = None
for atom in man:
    out = local_r2(atom)
    if out is None:
        continue
    if best is None or out[1] > best[0]:
        best = (out[1], atom, out)
_, atom, (amb_r2, inframe_r2, rows, data, curve) = best
c = np.vstack([curve, data])
mean = curve.mean(0)
_, s, vt = np.linalg.svd(curve - mean, full_matrices=False)
pc = (curve - mean) @ vt[:3].T
pd = (data - mean) @ vt[:3].T
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
if atom.get("dim", 1) == 2:
    n = int(atom["grid_n"])
    g = pc.reshape(n, n, 3)
    ax.plot_surface(g[:, :, 0], g[:, :, 1], g[:, :, 2], cmap="GnBu",
                    alpha=0.45, linewidth=0.05, edgecolor="#4477aa")
else:
    ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], color="#225588", lw=2.5)
sub = np.random.default_rng(0).choice(len(pd), min(500, len(pd)), replace=False)
ax.scatter(pd[sub, 0], pd[sub, 1], pd[sub, 2], s=6, color="#b02020",
           alpha=0.5, depthshade=True)
spans = np.vstack([pc, pd]).max(0) - np.vstack([pc, pd]).min(0)
ax.set_box_aspect(tuple(spans)); ax.set_axis_off()
ax.set_title(
    f"atom {atom['atom']} ({atom['kind']}): in-frame R² = {inframe_r2:.2f} "
    f"(variance shown in this 3-PC view) - ambient 128-d R² = {amb_r2:.2f}",
    fontsize=9)
fig.tight_layout(pad=0)
fig.savefig(f"{V2}/best_fit.png", dpi=160)
print(f"A: atom {atom['atom']} ({atom['kind']}) inframe={inframe_r2:.3f} ambient={amb_r2:.3f}")

# ---- B: loop walk ----
per = [a for a in man if a["kind"] == "periodic"]
per.sort(key=lambda a: -a["usage"])
la = per[0]
curve = np.fromfile(os.path.join(d, f"curve_{la['idx']}.bin")).reshape(-1, 128)
mean = curve.mean(0)
_, s, vt = np.linalg.svd(curve - mean, full_matrices=False)
p2 = (curve - mean) @ vt[:2].T
toks = np.fromfile(os.path.join(d, f"tokens_{la['idx']}.bin")).reshape(-1, 3)
trows, tt_raw = toks[:, 0].astype(int), toks[:, 1]
lo, hi = float(la.get("grid_lo", 0.0)), float(la.get("grid_hi", 1.0))
tt = (tt_raw - lo) / max(hi - lo, 1e-12)   # token coords into curve-fraction
# whiten the 2-PC frame so the (eccentric) ellipse is readable as a loop
p2w = p2 / np.maximum(p2.std(0), 1e-12)
n = len(curve)
NB = 40
hist, edges = np.histogram(tt, bins=NB, range=(0, 1))
occupied = hist > 0
fig, ax = plt.subplots(figsize=(10, 9))
# draw the loop: solid where tokens live, dotted grey where NO token routes
for b in range(NB):
    j0 = int(edges[b] * (n - 1)); j1 = max(j0 + 1, int(edges[b + 1] * (n - 1)))
    seg = p2w[j0:j1 + 1]
    if occupied[b]:
        ax.plot(seg[:, 0], seg[:, 1], color="#334466", lw=3.0,
                solid_capstyle="round", zorder=3)
    else:
        ax.plot(seg[:, 0], seg[:, 1], color="#bbbbbb", lw=1.2, ls=":", zorder=2)
# label the top-occupancy bins with their most common words, stacked clear
rng = np.random.default_rng(2)
top_bins = np.argsort(-hist)[:8]
top_bins = np.array([b for b in top_bins if hist[b] > 0])
lab = []
for b in sorted(top_bins):
    mid = 0.5 * (edges[b] + edges[b + 1])
    members = np.flatnonzero((tt >= edges[b]) & (tt < edges[b + 1]))
    pick = rng.choice(members, min(3, len(members)), replace=False)
    words = [word_at(trows[k]) for k in pick]
    j = int(mid * (n - 1))
    lab.append((p2w[j], " ".join(repr(w) for w in words), hist[b]))
# stack labels down the right margin with leader lines (no overlap ever)
xr = p2w[:, 0].max() + 0.9
ys = np.linspace(p2w[:, 1].max(), p2w[:, 1].min(), max(len(lab), 2))
for (pt, text, cnt), y in zip(lab, ys):
    ax.scatter([pt[0]], [pt[1]], s=30, color="#111111", zorder=5)
    ax.annotate(f"{text}  (n={cnt})", xy=pt, xytext=(xr, y), fontsize=9,
                ha="left", va="center", zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888"))
occ_pct = 100.0 * occupied.mean()
ax.text(0.01, 0.01,
        f"solid: token-occupied arc ({occ_pct:.0f}% of loop) - dotted: no tokens routed (extrapolated shape)",
        transform=ax.transAxes, fontsize=8, color="#555555")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout()
fig.savefig(f"{V2}/loop_walk.png", dpi=160)
print(f"B: loop atom {la['atom']} usage {la['usage']} occupancy {occ_pct:.0f}%")

# ---- C: safety-relevant concentration ----
charged = {"war", "battle", "attack", "killed", "weapon", "gun", "bomb",
           "troops", "army", "assault", "casualties", "invasion", "artillery",
           "regiment", "combat", "enemy", "fire", "dead", "wounded", "siege"}
best_c = None
for atom in man:
    tp = os.path.join(d, f"tokens_{atom['idx']}.bin")
    if not os.path.exists(tp):
        continue
    t = np.fromfile(tp).reshape(-1, 3)
    rws = t[:, 0].astype(int)
    samp = rws[:: max(1, len(rws) // 300)][:300]
    words = [word_at(r).lower() for r in samp]
    frac = sum(w in charged for w in words) / max(len(words), 1)
    if best_c is None or frac > best_c[0]:
        best_c = (frac, atom, samp, words)
frac, atom, samp, words = best_c
curve = np.fromfile(os.path.join(d, f"curve_{atom['idx']}.bin")).reshape(-1, 128)
mean = curve.mean(0)
_, s, vt = np.linalg.svd(curve - mean, full_matrices=False)
p3 = (curve - mean) @ vt[:3].T
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
if atom.get("dim", 1) == 2:
    n = int(atom["grid_n"])
    g = p3.reshape(n, n, 3)
    ax.plot_surface(g[:, :, 0], g[:, :, 1], g[:, :, 2], cmap="Reds",
                    alpha=0.4, linewidth=0.05, edgecolor="#993333")
else:
    ax.plot(p3[:, 0], p3[:, 1], p3[:, 2], color="#992222", lw=2.5)
# draw ALL sampled tokens (grey) and the charged ones (red, annotated) at
# their positions on the fitted manifold via their own coordinates
tks = np.fromfile(os.path.join(d, f"tokens_{atom['idx']}.bin")).reshape(-1, 3)
lo, hi = float(atom.get("grid_lo", 0.0)), float(atom.get("grid_hi", 1.0))
frac_pos = np.clip((tks[:, 1] - lo) / max(hi - lo, 1e-12), 0, 1)
if atom.get("dim", 1) == 2:
    pos_idx = (frac_pos * (len(p3) - 1)).astype(int)
else:
    pos_idx = (frac_pos * (len(p3) - 1)).astype(int)
samp_i = np.arange(len(tks))[:: max(1, len(tks) // 300)][:300]
grey = p3[pos_idx[samp_i]]
jit = np.random.default_rng(3).normal(0, 0.01 * np.abs(p3).max(), grey.shape)
ax.scatter(grey[:, 0] + jit[:, 0], grey[:, 1] + jit[:, 1], grey[:, 2] + jit[:, 2],
           s=5, color="#777777", alpha=0.4, depthshade=True)
n_ann = 0
for k, w in zip(samp_i, words):
    if w in charged and n_ann < 10:
        pt = p3[pos_idx[k]]
        ax.scatter([pt[0]], [pt[1]], [pt[2]], s=34, color="#b01515", depthshade=False)
        ax.text(pt[0], pt[1], pt[2], f" {w!r}", fontsize=9, color="#7a0e0e")
        n_ann += 1
ax.set_title(
    f"atom {atom['atom']} ({atom['kind']}): its fitted 1-D manifold in the "
    f"residual stream\nred curve = decoded atom - grey = routed tokens at "
    f"their fitted coordinate - conflict-vocab share {frac*100:.0f}%",
    fontsize=9)
ax.set_xlabel("PC1 of decoded curve"); ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
spans = p3.max(0) - p3.min(0)
ax.set_box_aspect(tuple(spans))
fig.tight_layout(pad=0)
fig.savefig(f"{V2}/safety.png", dpi=160)
print(f"C: atom {atom['atom']} ({atom['kind']}) conflict share {frac:.2f}")
