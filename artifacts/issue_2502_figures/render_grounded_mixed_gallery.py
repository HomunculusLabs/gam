"""Grounded 5:2 gallery: certified SAE curves plus real fitted d=2 Qwen charts.

The four curves are decoded from the certified K=256 manifold-SAE fit.  The
two surfaces are local quadratic d=2 charts fitted directly to neighborhoods
of the same Qwen3.5 layer-16 activation matrix.  No synthetic geometry enters.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/private/tmp/claude-scratch-2502")
DUMP = ROOT / "real_dump_local"
CHART = ROOT / "micro_chart.bin"
OUT = ROOT / "results" / "qwen_grounded_mixed_manifolds.png"
PROVENANCE = ROOT / "results" / "qwen_grounded_mixed_manifolds.json"
INK = "#070b16"
RNG = np.random.default_rng(2502)


def pca(values: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    centered = values - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:n_components].T, mean, vt[:n_components]


def configure(ax) -> None:
    ax.set_facecolor(INK)
    ax.set_axis_off()
    ax.set_box_aspect((1.7, 1.1, 0.75), zoom=1.22)
    ax.view_init(elev=23, azim=-58)
    ax.set_proj_type("persp", focal_length=0.9)


manifest = json.loads((DUMP / "manifest.json").read_text())
ambient_dim = np.fromfile(DUMP / "mean.bin", dtype="<f8").size
by_kind: dict[str, list[dict]] = {}
for item in manifest:
    by_kind.setdefault(item["kind"], []).append(item)

# Strongly used real fitted curves; no straight-line panel because the user
# asked the hero image to emphasize nontrivial geometry.
curve_entries = [
    by_kind["periodic"][0],
    by_kind["euclidean"][0],
    by_kind["periodic"][1],
    by_kind["euclidean"][2],
]

# Real Qwen activation chart.  Neighborhood discovery is done in its top-24
# global PCs, then each neighborhood gets its own d=2 coordinate chart.
x = np.fromfile(CHART, dtype="<f8").reshape(3000, 128)
global_scores, _, _ = pca(x, 24)

# Deterministic candidate anchors spread across the activation cloud.
candidate_ids = [int(np.argmax(np.linalg.norm(global_scores, axis=1)))]
min_dist = np.full(len(x), np.inf)
for _ in range(31):
    last = global_scores[candidate_ids[-1]]
    min_dist = np.minimum(min_dist, np.sum((global_scores - last) ** 2, axis=1))
    candidate_ids.append(int(np.argmax(min_dist)))


def fit_chart(anchor_id: int, neighbors: int = 180) -> dict:
    distance = np.sum((global_scores - global_scores[anchor_id]) ** 2, axis=1)
    rows = np.argsort(distance)[:neighbors]
    local = x[rows]
    uv, mean, tangent = pca(local, 2)
    scale = np.std(uv, axis=0)
    u = uv / np.maximum(scale, 1e-12)
    design = np.column_stack(
        [
            np.ones(len(u)),
            u[:, 0],
            u[:, 1],
            u[:, 0] ** 2,
            u[:, 0] * u[:, 1],
            u[:, 1] ** 2,
        ]
    )
    ridge = np.diag([0.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5])
    beta = np.linalg.solve(design.T @ design + ridge, design.T @ local)
    fitted = design @ beta
    plane_design = design[:, :3]
    plane_beta = np.linalg.lstsq(plane_design, local, rcond=None)[0]
    plane = plane_design @ plane_beta
    quadratic_gain = float(
        np.sum((local - plane) ** 2) - np.sum((local - fitted) ** 2)
    )
    curvature = float(np.linalg.norm(beta[3:]) / max(np.linalg.norm(beta[1:3]), 1e-12))
    return {
        "anchor": anchor_id,
        "rows": rows,
        "u": u,
        "beta": beta,
        "fitted": fitted,
        "score": quadratic_gain * curvature,
        "gain": quadratic_gain,
        "curvature": curvature,
    }


candidates = [fit_chart(anchor) for anchor in candidate_ids]
candidates.sort(key=lambda item: item["score"], reverse=True)
surfaces: list[dict] = []
used_rows: set[int] = set()
for item in candidates:
    overlap = len(used_rows.intersection(map(int, item["rows"]))) / len(item["rows"])
    if overlap < 0.25:
        surfaces.append(item)
        used_rows.update(map(int, item["rows"]))
    if len(surfaces) == 2:
        break

fig = plt.figure(figsize=(15, 6), dpi=200, facecolor=INK)
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.015, top=0.985, wspace=0.0, hspace=0.0)
curve_axes = [fig.add_subplot(2, 3, i + 1) for i in range(4)]
surface_axes = [
    fig.add_subplot(2, 3, 5, projection="3d"),
    fig.add_subplot(2, 3, 6, projection="3d"),
]
axes = curve_axes + surface_axes
for ax in curve_axes:
    ax.set_facecolor(INK)
    ax.set_aspect("equal")
    ax.axis("off")
for ax in surface_axes:
    configure(ax)

cmaps = ["twilight_shifted", "viridis", "twilight_shifted", "plasma"]
for ax, item, cmap_name in zip(axes[:4], curve_entries, cmaps):
    idx = item["idx"]
    curve = np.fromfile(DUMP / f"curve_{idx}.bin", dtype="<f8").reshape(-1, ambient_dim)
    tokens = np.fromfile(DUMP / f"tokens_{idx}.bin", dtype="<f8").reshape(-1, 3)
    curve2, mean, basis = pca(curve, 2)
    t = tokens[:, 1]
    grid = np.linspace(float(item["grid_lo"]), float(item["grid_hi"]), len(curve))
    beads2 = np.column_stack(
        [np.interp(t, grid, curve2[:, axis]) for axis in range(2)]
    )
    cmap = plt.get_cmap(cmap_name)
    phase = np.linspace(0, 1, len(curve2))
    for width, alpha in ((16, 0.025), (9, 0.055)):
        ax.plot(curve2[:, 0], curve2[:, 1], color=cmap(0.55), lw=width, alpha=alpha)
    for j in range(len(curve2) - 1):
        ax.plot(
            curve2[j : j + 2, 0],
            curve2[j : j + 2, 1],
            color=cmap(phase[j]),
            lw=3.0,
            alpha=0.98,
            solid_capstyle="round",
        )
    keep = RNG.choice(len(beads2), min(70, len(beads2)), replace=False)
    bead_phase = (t - grid[0]) / max(grid[-1] - grid[0], 1e-12)
    ax.scatter(
        beads2[keep, 0],
        beads2[keep, 1],
        c=bead_phase[keep],
        cmap=cmap,
        s=18,
        edgecolors="#f2f6ff",
        linewidths=0.45,
        alpha=0.98,
    )
    extent = np.ptp(np.vstack([curve2, beads2]), axis=0)
    center = np.mean(np.vstack([curve2, beads2]), axis=0)
    radius = max(float(extent.max()) * 0.62, 1e-9)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)

surface_meta = []
for ax, item, cmap_name in zip(surface_axes, surfaces, ["viridis", "magma"]):
    q = np.linspace(-1.75, 1.75, 29)
    uu, vv = np.meshgrid(q, q)
    mask = uu**2 + vv**2 <= 2.9
    design = np.column_stack(
        [
            np.ones(uu.size),
            uu.ravel(),
            vv.ravel(),
            uu.ravel() ** 2,
            (uu * vv).ravel(),
            vv.ravel() ** 2,
        ]
    )
    mesh = (design @ item["beta"]).reshape(*uu.shape, -1)
    all_points = np.vstack([mesh[mask], item["fitted"]])
    display, display_mean, display_basis = pca(all_points, 3)
    mesh3 = ((mesh.reshape(-1, 128) - display_mean) @ display_basis.T).reshape(
        *uu.shape, 3
    )
    bead3 = (item["fitted"] - display_mean) @ display_basis.T
    for axis in range(3):
        layer = mesh3[:, :, axis]
        layer[~mask] = np.nan
    color_value = np.arctan2(vv, uu)
    colors = plt.get_cmap(cmap_name)(
        (color_value - np.nanmin(color_value)) / np.ptp(color_value)
    )
    colors[..., 3] = np.where(mask, 0.82, 0.0)
    ax.plot_surface(
        mesh3[:, :, 0],
        mesh3[:, :, 1],
        mesh3[:, :, 2],
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.88,
    )
    keep = RNG.choice(len(bead3), min(75, len(bead3)), replace=False)
    phase = np.arctan2(item["u"][:, 1], item["u"][:, 0])
    ax.scatter(
        bead3[keep, 0],
        bead3[keep, 1],
        bead3[keep, 2],
        c=phase[keep],
        cmap=cmap_name,
        s=14,
        edgecolors="#f2f6ff",
        linewidths=0.35,
        alpha=0.94,
        depthshade=False,
    )
    surface_meta.append(
        {
            "anchor_row": int(item["anchor"]),
            "neighbor_rows": [int(v) for v in item["rows"]],
            "intrinsic_dim": 2,
            "quadratic_sse_gain_over_plane": item["gain"],
            "curvature_score": item["curvature"],
        }
    )

fig.savefig(OUT, dpi=200, facecolor=INK)
PROVENANCE.write_text(
    json.dumps(
        {
            "source": "Qwen3.5-4B-Base layer-16 activation chart",
            "curve_source": "certified K=256 manifold-SAE dump; train EV=0.7146; 109 cycles",
            "surface_source": "local quadratic d=2 charts fitted to real activation neighborhoods",
            "curves": [
                {
                    "atom": int(item["atom"]),
                    "kind": item["kind"],
                    "usage": int(item["usage"]),
                }
                for item in curve_entries
            ],
            "surfaces": surface_meta,
            "synthetic_geometry": False,
        },
        indent=2,
    )
)
print(OUT)
print(PROVENANCE)
