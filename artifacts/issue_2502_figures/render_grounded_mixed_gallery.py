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
    ax.set_box_aspect((1.7, 1.1, 0.75), zoom=1.38)
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
    singular = np.linalg.svd(fitted - fitted.mean(axis=0), compute_uv=False)
    sheet_ratio = float(singular[1] / max(singular[0], 1e-12))
    bend_ratio = float(singular[2] / max(singular[1], 1e-12))
    return {
        "anchor": anchor_id,
        "rows": rows,
        "u": u,
        "beta": beta,
        "fitted": fitted,
        "score": quadratic_gain * curvature * sheet_ratio,
        "gain": quadratic_gain,
        "curvature": curvature,
        "sheet_ratio": sheet_ratio,
        "bend_ratio": bend_ratio,
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
fig.subplots_adjust(left=0.015, right=0.985, bottom=0.01, top=0.99, wspace=0.02, hspace=0.0)
grid = fig.add_gridspec(2, 6, height_ratios=[0.92, 1.28])
curve_axes = [
    fig.add_subplot(grid[0, 0:2]),
    fig.add_subplot(grid[0, 2:4]),
    fig.add_subplot(grid[0, 4:6]),
]
surface_axes = [
    fig.add_subplot(grid[1, 0:3], projection="3d", computed_zorder=False),
    fig.add_subplot(grid[1, 3:6], projection="3d", computed_zorder=False),
]
axes = curve_axes + surface_axes
for ax in curve_axes:
    ax.set_facecolor(INK)
    ax.set_aspect("equal")
    ax.axis("off")
for ax in surface_axes:
    configure(ax)

cmaps = ["twilight_shifted", "viridis", "twilight_shifted"]
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
    for width, alpha in ((24, 0.025), (14, 0.060), (8, 0.10)):
        ax.plot(curve2[:, 0], curve2[:, 1], color=cmap(0.55), lw=width, alpha=alpha)
    for j in range(len(curve2) - 1):
        ax.plot(
            curve2[j : j + 2, 0],
            curve2[j : j + 2, 1],
            color=cmap(phase[j]),
            lw=4.2,
            alpha=0.98,
            solid_capstyle="round",
        )
    keep = RNG.choice(len(beads2), min(90, len(beads2)), replace=False)
    bead_phase = (t - grid[0]) / max(grid[-1] - grid[0], 1e-12)
    ax.scatter(
        beads2[keep, 0],
        beads2[keep, 1],
        c=bead_phase[keep],
        cmap=cmap,
        s=42,
        edgecolors="#f2f6ff",
        linewidths=0.9,
        alpha=0.98,
    )
    extent = np.ptp(np.vstack([curve2, beads2]), axis=0)
    center = np.mean(np.vstack([curve2, beads2]), axis=0)
    radius = max(float(extent.max()) * 0.56, 1e-9)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)

surface_meta = []
for surface_index, (ax, item, cmap_name) in enumerate(
    zip(surface_axes, surfaces, ["viridis", "magma"])
):
    # A smooth, hole-free domain in the atom's actual intrinsic coordinates.
    # Its radial envelope is the smoothed 90th-percentile data extent in each
    # direction, enlarged by 8%: enough context around the observations,
    # without the old far-field extrapolation.
    intrinsic = item["u"]
    token_angle = np.arctan2(intrinsic[:, 1], intrinsic[:, 0])
    token_radius = np.linalg.norm(intrinsic, axis=1)
    n_bins = 72
    bin_width = 2 * np.pi / n_bins
    bin_id = np.floor((token_angle + np.pi) / bin_width).astype(int) % n_bins
    radial_profile = np.full(n_bins, np.nan)
    for bin_index in range(n_bins):
        values = token_radius[bin_id == bin_index]
        if len(values):
            radial_profile[bin_index] = np.quantile(values, 0.90)
    valid = np.flatnonzero(np.isfinite(radial_profile))
    extended_index = np.concatenate([valid - n_bins, valid, valid + n_bins])
    extended_values = np.tile(radial_profile[valid], 3)
    radial_profile = np.interp(np.arange(n_bins), extended_index, extended_values)
    offsets = np.arange(-6, 7)
    weights = np.exp(-0.5 * (offsets / 2.7) ** 2)
    weights /= weights.sum()
    radial_profile = sum(
        weight * np.roll(radial_profile, int(offset))
        for weight, offset in zip(weights, offsets)
    )
    radial_profile *= 1.08

    bin_centers = -np.pi + (np.arange(n_bins) + 0.5) * bin_width
    extended_centers = np.concatenate(
        [bin_centers - 2 * np.pi, bin_centers, bin_centers + 2 * np.pi]
    )
    extended_profile = np.tile(radial_profile, 3)
    theta = np.linspace(-np.pi, np.pi, 97)
    boundary = np.interp(theta, extended_centers, extended_profile)
    radial_fraction = np.linspace(0.0, 1.0, 30)
    uu = radial_fraction[:, None] * boundary[None, :] * np.cos(theta)[None, :]
    vv = radial_fraction[:, None] * boundary[None, :] * np.sin(theta)[None, :]
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
    _, display_mean, display_basis = pca(
        np.vstack([mesh.reshape(-1, 128), item["fitted"]]),
        3,
    )
    mesh3 = ((mesh.reshape(-1, 128) - display_mean) @ display_basis.T).reshape(
        *uu.shape, 3
    )
    bead3 = (item["fitted"] - display_mean) @ display_basis.T
    # These fitted d=2 neighborhoods are Euclidean disk-like charts, not
    # periodic surfaces.  Color therefore follows the first intrinsic chart
    # coordinate continuously; using atan2 here would introduce a false
    # circular branch cut and visually imply the wrong topology.
    color_min = float(min(uu.min(), item["u"][:, 0].min()))
    color_max = float(max(uu.max(), item["u"][:, 0].max()))
    color_span = max(color_max - color_min, 1e-12)
    facecolors = plt.get_cmap(cmap_name)((uu - color_min) / color_span)
    facecolors[..., 3] = 0.82
    ax.plot_surface(
        mesh3[:, :, 0],
        mesh3[:, :, 1],
        mesh3[:, :, 2],
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.86,
        zorder=2,
    )

    # Sparse coordinate lines show the actual regular (u, v) chart rather than
    # an arbitrary triangulation of display-space points.
    ax.plot_wireframe(
        mesh3[:, :, 0],
        mesh3[:, :, 1],
        mesh3[:, :, 2],
        rstride=5,
        cstride=8,
        color="#d9e7ff",
        linewidth=0.34,
        alpha=0.14,
        zorder=4,
    )
    token_boundary = np.interp(token_angle, extended_centers, extended_profile)
    supported_vertices = np.flatnonzero(token_radius <= token_boundary * 1.01)
    keep = RNG.choice(
        supported_vertices,
        min(105, len(supported_vertices)),
        replace=False,
    )
    bead_coordinate = item["u"][:, 0]
    ax.scatter(
        bead3[keep, 0],
        bead3[keep, 1],
        bead3[keep, 2],
        c=bead_coordinate[keep],
        cmap=cmap_name,
        vmin=color_min,
        vmax=color_max,
        s=34,
        edgecolors="#f2f6ff",
        linewidths=0.75,
        alpha=0.98,
        depthshade=False,
        zorder=10,
    )
    if surface_index == 0:
        ax.view_init(elev=29, azim=-54)
    else:
        ax.view_init(elev=31, azim=34)
    surface_meta.append(
        {
            "anchor_row": int(item["anchor"]),
            "neighbor_rows": [int(v) for v in item["rows"]],
            "intrinsic_dim": 2,
            "quadratic_sse_gain_over_plane": item["gain"],
            "curvature_score": item["curvature"],
            "sheet_ratio": item["sheet_ratio"],
            "bend_ratio": item["bend_ratio"],
            "display_domain": "smooth 90th-percentile radial envelope of observed intrinsic coordinates with an 8% margin",
            "color_coordinate": "first non-periodic intrinsic chart coordinate",
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
