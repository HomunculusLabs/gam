import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2 = os.path.expanduser("~/i2502v2")
d = os.path.join(V2, "f_eu8096x")
man = json.load(open(d + "/manifest.json"))

def load(atom):
    curve = np.fromfile(d + "/curve_%d.bin" % atom["idx"]).reshape(-1, 128)
    pp = d + "/partial_%d.bin" % atom["idx"]
    part = np.fromfile(pp).reshape(-1, 129)[:, 1:] if os.path.exists(pp) else None
    toks = np.fromfile(d + "/tokens_%d.bin" % atom["idx"]).reshape(-1, 3)
    return curve, part, toks

def r2s(curve, data):
    if data is None or len(data) < 20:
        return None
    sub = data[np.random.default_rng(0).choice(len(data), min(400, len(data)), replace=False)]
    pred = curve[np.argmin(((sub[:, None, :] - curve[None, :, :]) ** 2).sum(-1), axis=1)]
    amb = 1 - ((sub - pred) ** 2).sum() / max(((sub - sub.mean(0)) ** 2).sum(), 1e-30)
    mean = curve.mean(0)
    _, _, vt = np.linalg.svd(curve - mean, full_matrices=False)
    df, pf = (sub - mean) @ vt[:3].T, (pred - mean) @ vt[:3].T
    inf = 1 - ((df - pf) ** 2).sum() / max(((df - df.mean(0)) ** 2).sum(), 1e-30)
    return amb, inf, sub

scored = []
for a in man:
    curve, part, toks = load(a)
    out = r2s(curve, part)
    if out:
        scored.append((out[1], out[0], a, curve, out[2], toks))
scored.sort(key=lambda t: -t[0])
picks = [scored[0], scored[len(scored) // 2], scored[-1]]
titles = ["BEST atom", "MEDIAN atom", "WORST dumped atom"]

fig = plt.figure(figsize=(15, 9.5))
for col, (pick, tt) in enumerate(zip(picks, titles)):
    inf, amb, a, curve, sub, toks = pick
    mean = curve.mean(0)
    _, _, vt = np.linalg.svd(curve - mean, full_matrices=False)
    pc, pd3 = (curve - mean) @ vt[:3].T, (sub - mean) @ vt[:3].T
    ax = fig.add_subplot(2, 3, col + 1, projection="3d")
    ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], color="#1a7a3a", lw=3, zorder=5)
    ax.scatter(pd3[:, 0], pd3[:, 1], pd3[:, 2], s=5, color="#c0392b", alpha=0.45)
    spans = np.vstack([pc, pd3]).max(0) - np.vstack([pc, pd3]).min(0)
    ax.set_box_aspect(tuple(np.maximum(spans, 1e-9)))
    ax.set_axis_off()
    ax.set_title("%s (#%d)\ngreen = fitted curve, red = the raw data this atom owns\nfit quality here: %.0f%%  (full 128-D: %.0f%%)"
                 % (tt, a["atom"], 100 * inf, 100 * amb), fontsize=9)
    ax2 = fig.add_subplot(2, 3, col + 4)
    ax2.hist(toks[:, 1], bins=40, color="#2060b0", alpha=0.8)
    ax2.set_title("raw data: where %d tokens sit on the coordinate" % len(toks), fontsize=9)
    ax2.set_xlabel("position along the curve")
    ax2.set_ylabel("tokens")
fig.suptitle("The causal champion (K=8096, held-out EV 0.853): actual fitted curves against their raw data\neach red cloud is the residual signal THIS atom is responsible for; green is what it learned",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(V2 + "/fits_gallery.png", dpi=150)
print("gallery: best=%.3f median=%.3f worst=%.3f (in-frame)"
      % (picks[0][0], picks[1][0], picks[2][0]))
