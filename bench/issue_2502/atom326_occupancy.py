import json, os
import numpy as np
V2 = os.path.expanduser("~/i2502v2"); d = os.path.join(V2, "b_mix1818")
man = json.load(open(d + "/manifest.json"))
a = sorted([x for x in man if x["kind"] == "periodic"], key=lambda x: -x["usage"])[0]
print("atom %d  kind=%s  usage=%d  grid=[%.4f, %.4f]" % (
    a["atom"], a["kind"], a["usage"], a.get("grid_lo", 0), a.get("grid_hi", 1)))
t = np.fromfile(d + "/tokens_%d.bin" % a["idx"]).reshape(-1, 3)
lo, hi = float(a.get("grid_lo", 0.0)), float(a.get("grid_hi", 1.0))
frac = np.clip((t[:, 1] - lo) / max(hi - lo, 1e-12), 0, 1)
print("routed phase: n=%d  min=%.4f max=%.4f" % (len(frac), frac.min(), frac.max()))

# occupancy by 36 bins (10 degrees each)
h, _ = np.histogram(frac, bins=36, range=(0, 1))
occ = (h > 0).sum()
print("occupied bins: %d of 36 (%.0f%% of the circle)" % (occ, 100*occ/36))
print("bin counts:", " ".join("%d" % c for c in h))
top = np.sort(h)[::-1]
print("mass in the densest 4 bins (40 deg): %.1f%%" % (100*top[:4].sum()/h.sum()))
print("mass in the densest 8 bins (80 deg): %.1f%%" % (100*top[:8].sum()/h.sum()))

# exact largest-gap test, the same one convert_underoccupied_loops uses
s = np.sort(frac); n = len(s)
gaps = np.diff(np.concatenate([s, [s[0] + 1.0]]))
g = gaps.max()
refuted = (n - 1) * np.log(1 - g) <= -2 * np.log(n) if g < 1 else True
print()
print("largest circular gap: %.4f of the period (%.1f degrees)" % (g, g*360))
print("uniform-closure test  (n-1)*ln(1-g) <= -2*ln(n):  %.3f <= %.3f  -> %s" % (
    (n-1)*np.log(1-g), -2*np.log(n), "REFUTED (should unroll)" if refuted else "not refuted"))
