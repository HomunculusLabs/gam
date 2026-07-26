"""Word-free 5:2 illustration, 2D composition: seven typed manifold atoms with
tokens living on them, floating in activation space."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
INK = "#0a0f1e"
W, H = 15.0, 6.0
fig, ax = plt.subplots(figsize=(W, H), dpi=200)
fig.patch.set_facecolor(INK)
ax.set_facecolor(INK)
ax.set_xlim(0, 15)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


def glow(x, y, color, lw=2.2, zo=5, alpha_main=0.95):
    for w, a in ((lw * 7, 0.035), (lw * 4, 0.07), (lw * 2.2, 0.16), (lw, alpha_main)):
        ax.plot(x, y, color=color, lw=w, alpha=a, solid_capstyle="round", zorder=zo)


def tokens(px, py, cvals, cmap, s=26, zo=8, noise=0.045):
    ax.scatter(px + rng.normal(0, noise, len(px)), py + rng.normal(0, noise, len(px)),
               c=cvals, cmap=cmap, s=s, alpha=0.96, lw=0.5, edgecolors=INK, zorder=zo)


# starfield
ax.scatter(rng.uniform(0, 15, 700), rng.uniform(0, 6, 700), color="#93a5cf",
           s=rng.uniform(0.6, 3.2, 700), alpha=0.16, lw=0, zorder=1)

# 1. line atom
t = np.linspace(0, 1, 60)
x, y = 0.55 + 1.5 * t, 1.15 + 1.15 * t
glow(x, y, "#69d2e7", lw=2.6)
i = rng.choice(60, 22)
tokens(x[i], y[i], t[i], "winter")

# 2. open curve atom
t = np.linspace(0, 1, 160)
x = 1.15 + 2.1 * t
y = 4.45 + 0.75 * np.sin(5.2 * t) * (1 - 0.35 * t)
glow(x, y, "#a8e05f", lw=2.6)
i = rng.choice(160, 30)
tokens(x[i], y[i], t[i], "summer")

# 3. circle atom, phase-colored (the calendar loop)
th = np.linspace(0, 2 * np.pi, 240)
cx, cy, r = 4.55, 2.05, 1.05
x, y = cx + r * np.cos(th), cy + 0.78 * r * np.sin(th)
cols = plt.get_cmap("twilight")(th / (2 * np.pi))
ax.plot(x, y, color="#caa9ff", lw=13, alpha=0.05, zorder=4)
for j in range(239):
    ax.plot(x[j:j + 2], y[j:j + 2], color=cols[j], lw=3.0, alpha=0.95, zorder=5)
i = rng.choice(240, 40)
tokens(x[i], y[i], th[i], "twilight", s=30)

# 4. sphere atom (longitude/latitude ellipses)
cx, cy, r = 7.0, 4.3, 1.0
th = np.linspace(0, 2 * np.pi, 120)
glow(cx + r * np.cos(th), cy + r * np.sin(th), "#ffb46b", lw=1.6, zo=3)
for k in (-0.62, -0.25, 0.25, 0.62):
    rr = r * np.sqrt(1 - k**2)
    glow(cx + rr * np.cos(th), cy + k * r + 0.22 * rr * np.sin(th), "#ffb46b",
         lw=0.9, zo=3, alpha_main=0.55)
for k in (-0.55, 0.0, 0.55):
    glow(cx + abs(k) * r + 0.30 * r * np.sqrt(1 - k**2) * np.cos(th) * np.sign(k + 1e-9) * 0
         + 0.30 * r * np.sqrt(1 - k**2) * np.cos(th),
         cy + r * np.sqrt(1 - k**2) * np.sin(th), "#ffb46b", lw=0.0001, zo=3, alpha_main=0.0)
for k in (-0.5, 0.0, 0.5):
    glow(cx + k * r + 0.25 * r * np.cos(th), cy + r * np.sqrt(max(1 - k**2, 0.05)) * np.sin(th),
         "#ffb46b", lw=0.9, zo=3, alpha_main=0.5)
u = rng.uniform(0, 2 * np.pi, 42)
v = np.arccos(rng.uniform(-1, 1, 42))
tokens(cx + r * np.sin(v) * np.cos(u), cy + r * np.cos(v) + 0.0 * np.sin(u),
       u, "autumn", s=22, noise=0.03)

# 5. torus atom (two circle families)
cx, cy, R, rr = 9.6, 1.75, 1.12, 0.42
th = np.linspace(0, 2 * np.pi, 160)
glow(cx + (R + rr) * np.cos(th), cy + 0.55 * (R + rr) * np.sin(th), "#ff6b9d", lw=1.6, zo=3)
glow(cx + (R - rr) * np.cos(th), cy + 0.55 * (R - rr) * np.sin(th), "#ff6b9d", lw=1.4, zo=3)
for a in np.linspace(0, 2 * np.pi, 12)[:-1]:
    mx, my = cx + R * np.cos(a), cy + 0.55 * R * np.sin(a)
    glow(mx + rr * np.cos(th) * 0.85, my + rr * np.sin(th) * 0.6, "#ff8fb3",
         lw=0.8, zo=3, alpha_main=0.5)
u = rng.uniform(0, 2 * np.pi, 46)
v = rng.uniform(0, 2 * np.pi, 46)
tokens(cx + (R + rr * 0.85 * np.cos(v)) * np.cos(u),
       cy + 0.55 * (R + rr * 0.85 * np.cos(v)) * np.sin(u) + 0.6 * rr * np.sin(v) * 0.3,
       u, "spring", s=20, noise=0.02)

# 6. Möbius atom (band with a half twist, width -> crossing)
u = np.linspace(0, 2 * np.pi, 300)
cx, cy = 12.15, 4.15
for w in np.linspace(-1, 1, 9):
    x = cx + (1.0 + 0.30 * w * np.cos(u / 2)) * np.cos(u)
    y = cy + 0.62 * (1.0 + 0.30 * w * np.cos(u / 2)) * np.sin(u) + 0.22 * w * np.sin(u / 2)
    glow(x, y, "#b78cff", lw=0.85, zo=3, alpha_main=0.55)
ww = rng.uniform(-1, 1, 34)
uu = rng.uniform(0, 2 * np.pi, 34)
tokens(cx + (1.0 + 0.30 * ww * np.cos(uu / 2)) * np.cos(uu),
       cy + 0.62 * (1.0 + 0.30 * ww * np.cos(uu / 2)) * np.sin(uu)
       + 0.22 * ww * np.sin(uu / 2), uu, "cool", s=20, noise=0.02)

# 7. tree/graph atom
root = (13.55, 0.85)
nodes = [root]
edges = []
frontier = [root]
for depth in range(1, 4):
    nxt = []
    for px, py in frontier:
        k = 2 if depth < 3 else rng.integers(1, 3)
        for j in range(k):
            q = (px + rng.uniform(0.25, 0.55),
                 py + rng.uniform(0.45, 0.85) * (1 if (len(nodes) + j) % 2 else -0.55) / depth
                 + 0.55 / depth * (1 if j else -1))
            nodes.append(q)
            edges.append(((px, py), q))
            nxt.append(q)
    frontier = nxt
for (xa, ya), (xb, yb) in edges:
    glow(np.array([xa, xb]), np.array([ya, yb]), "#5fd7a7", lw=1.5, zo=4, alpha_main=0.8)
px = np.array([p[0] for p in nodes])
py = np.array([p[1] for p in nodes])
tokens(px, py, py, "GnBu", s=42, noise=0.012)

fig.savefig("/home/ubuntu/i2502/manifold_dictionary_art.png", facecolor=INK, dpi=200)
print("saved 3000x1200")
