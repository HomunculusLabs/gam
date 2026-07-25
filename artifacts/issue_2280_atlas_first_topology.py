"""#2280: atlas-first manifold discovery — the recovered atlas and its verdict.

Renders what `manifold::atlas_topology::observe_atlas_topology` actually built on
four fixtures of KNOWN topology: the patch centers it elected, the nerve edges
that carry a certified transition, and the sign of each transition's orientation
class. Red edges are the orientation-REVERSING ones — on the Mobius band they are
the half-twist, and they are the only thing that distinguishes it from the
cylinder, whose GF(2) homology and Euler characteristic are identical.

The topology is NOT plotted from the fixture's known answer; every number in the
figure is the readout's own. Rust is the single source of truth, so this script
plots a dump rather than recomputing anything. Produce the dump with:

    cargo test -p gam-sae --lib zz_measure_atlas_plot_dump_2280 \
        -- --nocapture --test-threads=1 2> dump.log

then

    python3 artifacts/issue_2280_atlas_first_topology.py dump.log \
        artifacts/issue_2280_atlas_first_topology.png
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

LOG = sys.argv[1]
OUT = sys.argv[2]

CLOUD = "#c9ccd1"
CENTER = "#1f4e79"
KEEP = "#4a90d9"
FLIP = "#d1341c"


def parse(path):
    data = {}
    for raw in open(path, "r", errors="replace"):
        if not raw.startswith("PLOT "):
            continue
        parts = raw.split()
        label, kind = parts[1], parts[2]
        d = data.setdefault(label, {"points": {}, "centers": {}, "edges": [], "verdict": ""})
        if kind == "POINT":
            d["points"][int(parts[3])] = (float(parts[4]), float(parts[5]), float(parts[6]))
        elif kind == "CENTER":
            d["centers"][int(parts[3])] = int(parts[4])
        elif kind == "EDGE":
            d["edges"].append((int(parts[3]), int(parts[4]), int(parts[5])))
        elif kind == "VERDICT":
            d["verdict"] = raw.split("VERDICT", 1)[1].strip()
    for d in data.values():
        arr = np.zeros((max(d["points"]) + 1, 3))
        for i, xyz in d["points"].items():
            arr[i] = xyz
        d["xyz"] = arr
    return data


def field(verdict, key):
    for token in verdict.split():
        if token.startswith(key + "="):
            return token.split("=", 1)[1]
    return "?"


def headline(verdict):
    return verdict.split(":")[0].replace("atlas observes ", "").replace("atlas ", "")


def draw_atlas(ax, d, title):
    xyz = d["xyz"]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, c=CLOUD, alpha=0.6, linewidths=0)
    centers = sorted(d["centers"].values())
    cxyz = xyz[centers]
    keep, flip = [], []
    for a, b, sign in d["edges"]:
        (flip if sign < 0 else keep).append([tuple(xyz[a]), tuple(xyz[b])])
    if keep:
        ax.add_collection3d(Line3DCollection(keep, colors=KEEP, linewidths=0.8, alpha=0.7))
    if flip:
        ax.add_collection3d(Line3DCollection(flip, colors=FLIP, linewidths=2.2, alpha=0.95))
    ax.scatter(
        cxyz[:, 0], cxyz[:, 1], cxyz[:, 2],
        s=22, c=CENTER, depthshade=False, edgecolors="white", linewidths=0.4, zorder=6,
    )
    v = d["verdict"]
    ax.set_title(
        "%s\n-> %s     b1=%s  b2=%s  chi=%s\n%s"
        % (
            title,
            headline(v),
            field(v, "b1"),
            field(v, "b2"),
            field(v, "chi"),
            "w1 = 0  (no edge resists the gauge)"
            if not flip
            else "w1 != 0  (%d edge%s no gauge can fix)"
            % (len(flip), "" if len(flip) == 1 else "s"),
        ),
        fontsize=9.5,
        pad=0,
    )
    for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
        setter([])
    ax.grid(False)
    lim = np.abs(xyz).max() * 0.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


data = parse(LOG)
fig = plt.figure(figsize=(15.5, 10.2))
fig.suptitle(
    "#2280  atlas-first manifold discovery: the topology is READ OFF the local charts "
    "and their transition holonomy, never chosen from a menu",
    fontsize=14,
    y=0.985,
)

order = [
    ("trefoil", "trefoil knot   d = 1"),
    ("cylinder", "cylinder   d = 2"),
    ("mobius", "Mobius band   d = 2"),
    ("sphere", "closed 2-sphere   d = 2"),
]
positions = [1, 3, 4, 5]
for (key, title), pos in zip(order, positions):
    ax = fig.add_subplot(2, 3, pos, projection="3d")
    draw_atlas(ax, data[key], title)

# What a global-linear seed sees on the same trefoil.
ax2 = fig.add_subplot(2, 3, 2)
xyz = data["trefoil"]["xyz"]
centered = xyz - xyz.mean(axis=0)
_, s, vt = np.linalg.svd(centered, full_matrices=False)
proj = centered @ vt[:2].T
ax2.plot(proj[:, 0], proj[:, 1], "-", lw=1.0, color="#8b8f95")
ax2.scatter(proj[:, 0], proj[:, 1], s=3, c="#5c6066", linewidths=0)
ax2.set_title(
    "the same trefoil under its best 2-D LINEAR seed\n"
    "singular values %.1f / %.1f / %.1f  (sigma3/sigma1 = %.2f)\n"
    "the shadow self-crosses, so the seed is not injective:\n"
    "distinct points of the loop share one coordinate"
    % (s[0], s[1], s[2], s[2] / s[0]),
    fontsize=9.5,
)
ax2.set_aspect("equal")
ax2.set_xticks([])
ax2.set_yticks([])
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")
circle_v = data["circle"]["verdict"]
trefoil_v = data["trefoil"]["verdict"]
ax6.text(
    0.0,
    1.0,
    "THE CONSTRUCTION\n"
    "  charts   local-PCA frame per patch, certified injective\n"
    "  nerve    every set of patches with a common row, ALL orders\n"
    "  w1       sign of det(transition) as a GF(2) 1-cochain; its\n"
    "           class in H1(nerve; GF(2)) is the Stiefel-Whitney class\n"
    "  verdict  (d, b0, b1, b2, chi, w1) -> the classification of\n"
    "           compact surfaces.  A table lookup, not a race.\n\n"
    "CYLINDER vs MOBIUS\n"
    "  identical b1, b2, chi.  Separated only by the red edges.\n\n"
    "SPHERE\n"
    "  simply connected, so EVERY loop is contractible and holonomy\n"
    "  is blind; chi = 2 is what carries this case.\n\n"
    "TREFOIL vs ROUND CIRCLE\n"
    "  round  : %s\n"
    "  knot   : %s\n"
    "  Same verdict, invariant for invariant. Every input is a\n"
    "  transition between charts, and a transition is intrinsic."
    % (
        headline(circle_v) + "  " + " ".join(circle_v.split()[-5:-1]),
        headline(trefoil_v) + "  " + " ".join(trefoil_v.split()[-5:-1]),
    ),
    va="top",
    ha="left",
    fontsize=9,
    family="monospace",
    transform=ax6.transAxes,
)

fig.subplots_adjust(
    left=0.01, right=0.99, top=0.87, bottom=0.03, wspace=0.06, hspace=0.22
)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
