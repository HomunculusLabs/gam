"""Embedded-sphere atlas: each fitted S^2 atom's decoded surface with the
routed tokens drawn ON it.

FIX (2026-07-29): the previous version computed fractional lattice indices and
then did `astype(int)`, snapping every token to a mesh VERTEX. With ~600 dots
over an n x n lattice that renders as a grid of clusters -- an artifact of the
quantisation, not a property of the fit. Its docstring claimed "bilinear
lookup"; no interpolation was implemented. This version actually interpolates,
and wraps in longitude so the seam does not clamp.

A token's position is now a convex combination of the four surrounding lattice
samples, so a dot can land anywhere on the rendered surface.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latlon(u):
    lat = np.arcsin(np.clip(u[:, 2], -1, 1))
    lon = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
    return lat, lon


def bilinear(grid3, lat, lon, n):
    """Interpolate the PCA-3 surface at continuous (lat, lon).

    Rows are latitude and do not wrap -- the lattice is open at the poles, so
    the row index is clamped. Columns are longitude and DO wrap: a token at
    lon just under 2*pi sits between the last column and the first, not
    against a wall.
    """
    fi = np.clip((lat + np.pi / 2) / np.pi * (n - 1), 0, n - 1)
    fj = np.mod(lon / (2 * np.pi) * n, n)
    i0 = np.floor(fi).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    j0 = np.floor(fj).astype(int) % n
    j1 = (j0 + 1) % n
    wi = (fi - i0)[:, None]
    wj = (fj - np.floor(fj))[:, None]
    return ((1 - wi) * ((1 - wj) * grid3[i0, j0] + wj * grid3[i0, j1])
            + wi * ((1 - wj) * grid3[i1, j0] + wj * grid3[i1, j1]))


def main() -> int:
    d, dest = sys.argv[1], sys.argv[2]
    cols = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    man = json.load(open(os.path.join(d, "manifest.json")))
    spheres = [a for a in man if a["kind"] == "sphere"][:6]
    if not spheres:
        print("no sphere atoms in manifest")
        return 1
    fig = plt.figure(figsize=(13, 8.5))
    for slot, atom in enumerate(spheres):
        n = int(atom["grid_n"])
        surf = np.fromfile(os.path.join(d, f"curve_{atom['idx']}.bin")).reshape(n, n, cols)
        flat = surf.reshape(-1, cols)
        mean = flat.mean(0)
        _, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
        grid3 = ((flat - mean) @ vt[:3].T).reshape(n, n, 3)
        ax = fig.add_subplot(2, 3, slot + 1, projection="3d")
        ax.plot_surface(grid3[:, :, 0], grid3[:, :, 1], grid3[:, :, 2],
                        rstride=1, cstride=1, color="#88aacc", alpha=0.30,
                        linewidth=0.15, edgecolor="#446688", shade=True)
        distinct = None
        tok_path = os.path.join(d, f"tokens3_{atom['idx']}.bin")
        if os.path.exists(tok_path):
            toks = np.fromfile(tok_path).reshape(-1, 4)
            u = toks[:, 1:4]
            u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
            lat, lon = latlon(u)
            pts = bilinear(grid3, lat, lon, n)
            # How much of the old picture was quantisation? Count the distinct
            # VERTICES the old code would have collapsed these tokens onto.
            oi = np.clip(((lat + np.pi / 2) / np.pi * n - 0.5), 0, n - 1).astype(int)
            oj = np.clip(lon / (2 * np.pi) * (n - 1), 0, n - 1).astype(int)
            distinct = len(set(zip(oi.tolist(), oj.tolist())))
            sub = np.random.default_rng(0).choice(
                len(pts), min(600, len(pts)), replace=False)
            ax.scatter(pts[sub, 0], pts[sub, 1], pts[sub, 2],
                       s=3, color="#b03030", alpha=0.6, depthshade=False)
        spans = grid3.reshape(-1, 3).max(0) - grid3.reshape(-1, 3).min(0)
        ax.set_box_aspect(tuple(spans))
        extra = ("" if distinct is None
                 else f"  ·  {atom['usage']} tokens → {distinct} vertices (old)")
        ax.set_title(f"usage {atom['usage']}  ·  3-PC var "
                     f"{float((s[:3]**2).sum()/(s**2).sum())*100:.0f}%{extra}",
                     fontsize=8)
        ax.set_axis_off()
    fig.suptitle("Embedded S² atoms, routed tokens bilinearly placed "
                 "(was: snapped to mesh vertices)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(dest, dpi=150)
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
