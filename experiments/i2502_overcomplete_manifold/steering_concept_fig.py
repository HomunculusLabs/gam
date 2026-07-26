import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

INK = "#0a0f1e"
TW = plt.get_cmap("twilight")
fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
ax.set_xlim(0, 15); ax.set_ylim(0, 6); ax.set_aspect("equal"); ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
rng = np.random.default_rng(4)
ax.scatter(rng.uniform(0,15,520), rng.uniform(0,6,520), color="#93a5cf",
           s=rng.uniform(.6,2.6,520), alpha=.12, lw=0)

R, SQ = 2.05, 0.72
def scene(cx, cy):
    th = np.linspace(0, 2*np.pi, 500)
    x = cx + R*np.cos(th); y = cy + SQ*R*np.sin(th)
    for w, a in ((46,.05),(30,.09),(17,.13)):
        ax.plot(x, y, color="#3d548f", lw=w, alpha=a, zorder=2)
    cols = TW(th/(2*np.pi))
    for j in range(499):
        ax.plot(x[j:j+2], y[j:j+2], color=cols[j], lw=3.6, alpha=.98, zorder=4)
    days = np.arange(7)/7*2*np.pi + np.pi/2
    dx = cx + R*np.cos(days); dy = cy + SQ*R*np.sin(days)
    ax.scatter(dx, dy, s=150, c=(days/(2*np.pi))%1, cmap="twilight", vmin=0, vmax=1,
               lw=1.3, edgecolors="#dfe6f7", zorder=6)
    return days, dx, dy

TUE, WED = 5, 6
def mark_start_target(days, dx, dy):
    ax.scatter([dx[TUE]],[dy[TUE]], s=430, color=TW((days[TUE]/(2*np.pi))%1),
               lw=2.6, edgecolors="#ffffff", zorder=8)
    ax.scatter([dx[WED]],[dy[WED]], s=430, facecolors="none",
               edgecolors="#ffffff", lw=1.6, ls=(0,(2,2)), zorder=8)

# ================= LEFT: flat SAE =================
cx, cy = 3.85, 3.0
days, dx, dy = scene(cx, cy)
mark_start_target(days, dx, dy)
wdir = np.array([dx[WED]-cx, dy[WED]-cy]); wdir /= np.linalg.norm(wdir)
start = np.array([dx[TUE], dy[TUE]]); end = start + 2.45*wdir
ax.add_patch(FancyArrowPatch(tuple(start), tuple(end), arrowstyle="-|>",
             mutation_scale=30, lw=4.2, color="#ff6b6b", zorder=9))
# superposition at the endpoint: Tuesday color UNDER Wednesday color, misaligned
ax.scatter([end[0]+.13],[end[1]+.10], s=520, color=TW((days[TUE]/(2*np.pi))%1),
           alpha=.85, lw=1.6, edgecolors="#ffd7d7", zorder=10)
ax.scatter([end[0]-.13],[end[1]-.13], s=520, color=TW((days[WED]/(2*np.pi))%1),
           alpha=.85, lw=1.6, edgecolors="#ffd7d7", zorder=11)
sp = end[:,None] + np.array([[.55,-.42,.66,-.30,.18,-.60,.42],[.42,.60,-.18,-.55,.72,.16,-.44]])
ax.scatter(sp[0], sp[1], s=34, color="#ffab91", alpha=.85, marker="x", lw=1.6, zorder=9)

# ================= RIGHT: manifold =================
cx2, cy2 = 11.15, 3.0
days2, dx2, dy2 = scene(cx2, cy2)
mark_start_target(days2, dx2, dy2)
tt = np.linspace(days2[TUE], days2[TUE] + 2*np.pi/7, 80)
px = cx2 + 1.62*R/2.05*2.05*np.cos(tt)*1.13/1.13; py = cy2 + SQ*R*np.sin(tt)
px = cx2 + (R+0.28)*np.cos(tt); py = cy2 + SQ*(R+0.28)*np.sin(tt)
ax.plot(px[:-8], py[:-8], color="#5fd7a7", lw=4.6, zorder=9, solid_capstyle="round")
ax.add_patch(FancyArrowPatch(tuple([px[-9],py[-9]]), tuple([px[-1],py[-1]]),
             arrowstyle="-|>", mutation_scale=30, lw=4.6, color="#5fd7a7", zorder=9))
# arrival: clean Wednesday, filled now
ax.scatter([dx2[WED]],[dy2[WED]], s=430, color=TW((days2[WED]/(2*np.pi))%1),
           lw=2.6, edgecolors="#b9ffe1", zorder=10)
for s_, a_ in ((1150,.10),(800,.16)):
    ax.scatter([dx2[WED]],[dy2[WED]], s=s_, facecolors="none", edgecolors="#5fd7a7",
               lw=2.2, alpha=a_, zorder=7)
fig.savefig("/home/ubuntu/i2502/steering_concept.png", facecolor=INK, dpi=200)
print("saved v2")
