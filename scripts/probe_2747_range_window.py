"""#2747 follow-up: what does `V(κ⋆, ℓ)` actually look like, and how wide does
the range window have to be to contain its minimum?

Uses the COHERENT single-range model (which is now the shipped basis) and the
same λ-profiled Gaussian REML criterion, re-implemented in numpy from the
crate's own formulas so it is independent of the Rust build.
"""

import numpy as np

from probe_2747_kappa_range import (  # noqa: E402
    distance,
    reml_profile,
    sum_to_zero_frame,
)


def farthest_point_centers(data, m):
    """The builder's center rule: farthest-point from the cloud, then the
    near-origin center snapped to the exact pole."""
    idx = [int(np.argmax((data * data).sum(-1)))]
    while len(idx) < m:
        d = np.min(
            np.linalg.norm(data[:, None, :] - data[None, idx, :], axis=-1), axis=1
        )
        idx.append(int(np.argmax(d)))
    centers = data[idx].copy()
    centers[int(np.argmin((centers * centers).sum(-1)))] = 0.0
    return centers


def coherent_blocks(data, centers, kappa, ell):
    z = sum_to_zero_frame(centers.shape[0])
    x = np.exp(-distance(data, centers, kappa) / ell) @ z
    s = z.T @ np.exp(-distance(centers, centers, kappa) / ell) @ z
    return x, 0.5 * (s + s.T)


def v_at(data, centers, y, kappa, ell):
    x, s = coherent_blocks(data, centers, kappa, ell)
    n, p = x.shape
    design = np.hstack([np.ones((n, 1)), x])
    penalty = np.zeros((p + 1, p + 1))
    penalty[1:, 1:] = s
    return reml_profile(design, penalty, y)


def fixture(n, m, kappa_star, radius, mult, noise, rng):
    pts = []
    while len(pts) < n:
        a, b = rng.uniform(-1, 1, 2)
        if a * a + b * b <= 1:
            pts.append((a * radius, b * radius))
    data = np.array(pts)
    centers = farthest_point_centers(data, m)
    cc = 2.0 * np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=-1
    )[np.triu_indices(m, 1)]
    ell_ref = np.sort(cc)[cc.size // 2]
    dc = 2.0 * np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=-1)
    evaluated = np.concatenate([dc.ravel(), cc])
    pos = evaluated[evaluated > 0]
    truth_ell = ell_ref * mult
    x, _ = coherent_blocks(data, centers, kappa_star, truth_ell)
    w = np.array([1.0 / (1.0 + j) for j in range(x.shape[1])])
    mu = x @ w
    mu = (mu - mu.mean()) / mu.std()
    y = mu + noise * rng.standard_normal(n)
    return data, centers, y, ell_ref, pos.min(), evaluated.max()


def main():
    n, m, radius, noise = 120, 6, 0.6, 0.10
    for kappa_star in (-1.0, 0.0, 1.0):
        for mult in (0.5, 1.0, 2.0):
            rng = np.random.default_rng(20260802)
            data, centers, y, ell_ref, dmin, dmax = fixture(
                n, m, kappa_star, radius, mult, noise, rng
            )
            grid = ell_ref * np.exp(np.linspace(-4.0, 4.0, 41))
            vals = [v_at(data, centers, y, kappa_star, e)[0] for e in grid]
            i = int(np.argmin(vals))
            cc_lo = 2.0 * np.min(
                [
                    np.linalg.norm(centers[a] - centers[b])
                    for a in range(m)
                    for b in range(a + 1, m)
                ]
            )
            cc_hi = 2.0 * np.max(
                [
                    np.linalg.norm(centers[a] - centers[b])
                    for a in range(m)
                    for b in range(a + 1, m)
                ]
            )
            print(
                f"\nk*={kappa_star:+.1f} mult={mult}: ell_ref={ell_ref:.4f} "
                f"truth={ell_ref*mult:.4f}  center-window=[{cc_lo:.4f},{cc_hi:.4f}]  "
                f"evaluated-window=[{dmin:.4f},{dmax:.4f}]"
            )
            print(f"  argmin_ell = {grid[i]:.4f}  (V={vals[i]:.4f})")
            print(
                "  V(ell): "
                + " ".join(
                    f"{grid[j]:.3f}:{vals[j]:.1f}" for j in range(0, 41, 4)
                )
            )


if __name__ == "__main__":
    main()
