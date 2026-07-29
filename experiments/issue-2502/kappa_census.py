"""Shattered-manifold census on a trained flat SAE (curl.rs witness statistics).

A mean-zero circle's cone IS its 2-plane: two linear atoms reconstruct it
exactly, the residual vanishes, and residual-driven training is structurally
blind to the circle. The witness is the joint amplitude law of a co-active
pair: with r^2 = a^2 + b^2 over co-firings,
    kappa = m4/m2^2 of r
      ~ 1  -> ring (a shattered circle),
      ~ 2  -> Gaussian fill (an ordinary 2-D blob),
      >> 2 -> gated spike.
Plus the phase test: a true ring spreads theta = atan2(b, a) broadly instead
of hugging the axes (axis-hugging = two genuinely separate features).
"""
import os
import numpy as np

V2 = os.path.expanduser("~/i2502v2")
chart = np.fromfile(V2 + "/doc_chart.bin").reshape(-1, 128)[:100000]
import sys as _sys
if len(_sys.argv) > 1 and _sys.argv[1] == "null":
    # NULL: Gaussian data matched to the real rows' mean/covariance diagonal
    # (independent coordinates, no manifolds), through the SAME OMP encoder.
    rng0 = np.random.default_rng(7)
    chart = rng0.normal(0, 1, chart.shape) * chart.std(0) + chart.mean(0)
blob = np.load(V2 + "/baseline_k5333_s2.npz")
W = blob["W_dec"].astype(np.float64)
b = blob["b_pre"].astype(np.float64)
k_act = int(blob["k_act"]) if "k_act" in blob else 8
K = len(W)
norms = (W * W).sum(1)

# OMP top-k encode (identical to the splice harness)
R = chart - b
taken = np.zeros((len(chart), K), dtype=bool)
picks = np.zeros((len(chart), k_act), dtype=np.int64)
coefs = np.zeros((len(chart), k_act))
for s in range(k_act):
    g = 2.0 * (R @ W.T) - norms
    g[taken] = -np.inf
    p = g.argmax(1)
    picks[:, s] = p
    taken[np.arange(len(chart)), p] = True
    c = np.maximum((R * W[p]).sum(1) / np.maximum(norms[p], 1e-30), 0.0)
    coefs[:, s] = c
    R = R - c[:, None] * W[p]

# co-firing table
from collections import defaultdict
pair_rows = defaultdict(list)
for i in range(len(chart)):
    fired = [(picks[i, s], coefs[i, s]) for s in range(k_act) if coefs[i, s] > 1e-8]
    fired.sort()
    for x in range(len(fired)):
        for y in range(x + 1, len(fired)):
            pair_rows[(fired[x][0], fired[y][0])].append((fired[x][1], fired[y][1]))

MIN_CO = 60
results = []
for (i, j), ab in pair_rows.items():
    if len(ab) < MIN_CO:
        continue
    ab = np.array(ab)
    a, bb = ab[:, 0], ab[:, 1]
    r2 = a * a + bb * bb
    m2 = r2.mean()
    m4 = (r2 * r2).mean()
    kappa = m4 / max(m2 * m2, 1e-30)
    theta = np.arctan2(bb, a)
    # phase spread: circular resultant length of 4*theta (axis-hug detector);
    # low spread4 => angles cluster at axes; high entropy-like spread => ring-ish
    spread4 = 1.0 - np.abs(np.exp(4j * theta).mean())
    cosang = abs(float(W[i] @ W[j]) / np.sqrt(norms[i] * norms[j]))
    results.append((kappa, spread4, cosang, len(ab), i, j))

results.sort(key=lambda t: t[0])
arr = np.array([(r[0], r[1], r[2]) for r in results])
n = len(results)
print("co-active pairs with >=%d co-firings: %d (of %d atoms)" % (MIN_CO, n, K))
if n:
    print("kappa quantiles p5/p25/p50/p75/p95: " +
          "/".join("%.2f" % q for q in np.percentile(arr[:, 0], [5, 25, 50, 75, 95])))
    ringish = (arr[:, 0] < 1.35) & (arr[:, 1] > 0.55)
    gauss = (arr[:, 0] > 1.7) & (arr[:, 0] < 2.4)
    print("RING-LIKE (kappa<1.35 AND broad phase): %d pairs (%.1f%%)" % (ringish.sum(), 100 * ringish.mean()))
    print("Gaussian-fill band (1.7<kappa<2.4): %d (%.1f%%)" % (gauss.sum(), 100 * gauss.mean()))
    print("spike band (kappa>3): %d (%.1f%%)" % ((arr[:, 0] > 3).sum(), 100 * (arr[:, 0] > 3).mean()))
    print("top 12 ring candidates (kappa, phase-spread, |cos(dec angle)|, n, i, j):")
    shown = 0
    for kappa, s4, ca, m, i, j in results:
        if kappa < 1.35 and s4 > 0.55:
            print("  k=%.2f spread=%.2f cos=%.2f n=%d atoms=(%d,%d)" % (kappa, s4, ca, m, i, j))
            shown += 1
            if shown >= 12:
                break
np.save(V2 + ("/kappa_null.npy" if len(_sys.argv) > 1 else "/kappa_census.npy"), np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in results]))
print("DONE")
