"""Is Qwen3.5-4B's residual stream locally CURVED? -- the fit-free premise test for #2502.

A curved dictionary is only warranted if local neighbourhoods of the activation
cloud bend away from their own tangent plane by more than a flat model predicts.
This measures exactly that, and prices it against the null that makes the
question non-vacuous.

Per landmark neighbourhood:
  1. split the neighbours into a fit half and a held-out half (paired across arms);
  2. fit the local tangent frame on the fit half: centre + top-d local PCA basis;
  3. project the held-out neighbours; regress their NORMAL components on
     QUADRATIC monomials of their d tangent coordinates, coefficients estimated
     on the fit half only;
  4. report the held-out R^2 of that quadratic normal fit -- the second
     fundamental form's out-of-sample explanatory power.

A locally flat manifold with any amount of isotropic noise scores 0 in
expectation, because the quadratic is fit on rows it is not scored on.

The null arm is a Gaussian with the SAME empirical covariance, run through the
identical pipeline on the identical landmark/neighbour bookkeeping. A Gaussian
is locally flat by construction, so whatever curvature score it produces is the
score's own finite-sample floor. The real arm has to beat that floor, not zero.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch


def farthest_point_landmarks(x: torch.Tensor, n_landmarks: int, seed: int) -> torch.Tensor:
    """Deterministic farthest-point sampling (the #2023/#2280 Tier-1 seeder rule)."""
    n = x.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(0, n, (1,), generator=g).item())
    picked = [first]
    d2 = ((x - x[first]) ** 2).sum(1)
    for _ in range(n_landmarks - 1):
        nxt = int(torch.argmax(d2).item())
        picked.append(nxt)
        d2 = torch.minimum(d2, ((x - x[nxt]) ** 2).sum(1))
    return torch.tensor(picked, dtype=torch.long, device=x.device)


def quadratic_features(t: torch.Tensor) -> torch.Tensor:
    """[1, t_i, t_i t_j (i<=j)] -- affine part plus the second-order monomials."""
    m, d = t.shape
    cols = [torch.ones(m, 1, device=t.device, dtype=t.dtype), t]
    for i in range(d):
        for j in range(i, d):
            cols.append((t[:, i] * t[:, j]).unsqueeze(1))
    return torch.cat(cols, dim=1)


def affine_features(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.ones(t.shape[0], 1, device=t.device, dtype=t.dtype), t], dim=1)


def ridge_solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Least squares via pivoted QR-free lstsq; the design is well conditioned by
    construction (orthonormal tangent coordinates, monomials of order <= 2)."""
    return torch.linalg.lstsq(a, b).solution


def neighbourhood_curvature(
    xs: torch.Tensor, d: int, seed: int
) -> tuple[float, float, float]:
    """Return (held-out quadratic R^2 on normal components, held-out affine R^2,
    tangent participation ratio) for one neighbourhood."""
    m = xs.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(m, generator=g).to(xs.device)
    half = m // 2
    fit_i, ho_i = perm[:half], perm[half:]

    xf = xs[fit_i]
    mu = xf.mean(0, keepdim=True)
    xf_c = xf - mu
    # Local frame from the FIT half only.
    _, s, vh = torch.linalg.svd(xf_c, full_matrices=False)
    dd = min(d, vh.shape[0])
    tangent = vh[:dd]
    ev = (s**2)
    part_ratio = float((ev.sum() ** 2 / (ev**2).sum()).item())

    xh_c = xs[ho_i] - mu
    # Tangent coordinates, and the normal residual we are trying to explain.
    t_fit = xf_c @ tangent.T
    t_ho = xh_c @ tangent.T
    nrm_fit = xf_c - t_fit @ tangent
    nrm_ho = xh_c - t_ho @ tangent

    # Scale tangent coordinates by the fit half's own spread so the quadratic
    # monomials are dimensionless -- no absolute length enters the design.
    scale = t_fit.std(0, keepdim=True).clamp_min(1e-12)
    tf, th = t_fit / scale, t_ho / scale

    denom = float((nrm_ho**2).sum().item())
    if denom <= 0.0:
        return 0.0, 0.0, part_ratio

    coef_q = ridge_solve(quadratic_features(tf), nrm_fit)
    pred_q = quadratic_features(th) @ coef_q
    r2_q = 1.0 - float(((nrm_ho - pred_q) ** 2).sum().item()) / denom

    coef_a = ridge_solve(affine_features(tf), nrm_fit)
    pred_a = affine_features(th) @ coef_a
    r2_a = 1.0 - float(((nrm_ho - pred_a) ** 2).sum().item()) / denom

    return r2_q, r2_a, part_ratio


def run_arm(
    x: torch.Tensor, landmarks: torch.Tensor, n_nb: int, d: int, seed: int
) -> dict:
    r2q, r2a, pr = [], [], []
    for li, lm in enumerate(landmarks.tolist()):
        d2 = ((x - x[lm]) ** 2).sum(1)
        nb = torch.topk(d2, n_nb, largest=False).indices
        q, a, p = neighbourhood_curvature(x[nb], d, seed + li)
        r2q.append(q)
        r2a.append(a)
        pr.append(p)
    r2q_a, r2a_a, pr_a = np.array(r2q), np.array(r2a), np.array(pr)
    return {
        "curv_r2_mean": float(r2q_a.mean()),
        "curv_r2_median": float(np.median(r2q_a)),
        "curv_r2_q10": float(np.quantile(r2q_a, 0.10)),
        "curv_r2_q90": float(np.quantile(r2q_a, 0.90)),
        "affine_r2_mean": float(r2a_a.mean()),
        "curv_gain_over_affine_mean": float((r2q_a - r2a_a).mean()),
        "participation_ratio_mean": float(pr_a.mean()),
        "n_landmarks": int(len(r2q)),
        "per_landmark_curv_r2": r2q_a.tolist(),
        "per_landmark_affine_r2": r2a_a.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 22])
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--landmarks", type=int, default=64)
    ap.add_argument("--neighbours", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--tangent-dims", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(2)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    rows = []
    t0 = time.time()

    for layer in args.layers:
        raw = np.load(f"{args.harvest}/resid_L{layer}.npy", mmap_mode="r")
        idx = np.sort(rng.choice(raw.shape[0], size=min(args.rows, raw.shape[0]), replace=False))
        xs = torch.from_numpy(np.asarray(raw[idx], dtype=np.float32)).to(dev)
        xs = xs - xs.mean(0, keepdim=True)
        # PCA chart via the p x p covariance eigendecomposition: same subspace as
        # the thin SVD, but the workspace is p x p rather than n x p, which is
        # what keeps this arm off the GPU memory the fit lane is holding.
        cov = (xs.T @ xs).double()
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order], evecs[:, order]
        r = min(args.pca, evecs.shape[1])
        basis = evecs[:, :r].T.float()
        real = xs @ basis.T
        ev_frac = float((evals[:r].sum() / evals.clamp_min(0).sum()).item())

        # Covariance-matched Gaussian null in the SAME chart: locally flat by
        # construction, so its score is the statistic's finite-sample floor.
        gen = torch.Generator(device=dev).manual_seed(args.seed + 977 * layer)
        null = torch.randn(real.shape, generator=gen, device=dev) * real.std(0, keepdim=True)

        landmarks = farthest_point_landmarks(real, args.landmarks, args.seed + layer)
        # The null arm reuses the SAME landmark ordinals so the two arms are paired.
        for n_nb in args.neighbours:
            for d in args.tangent_dims:
                n_params = 1 + d + d * (d + 1) // 2
                if n_nb // 2 <= n_params + 2:
                    continue
                real_stats = run_arm(real, landmarks, n_nb, d, args.seed)
                null_lm = farthest_point_landmarks(null, args.landmarks, args.seed + layer)
                null_stats = run_arm(null, null_lm, n_nb, d, args.seed)
                row = {
                    "layer": layer,
                    "pca_dim": r,
                    "pca_ev_frac": ev_frac,
                    "n_rows": int(len(idx)),
                    "n_neighbours": n_nb,
                    "tangent_dim": d,
                    "quad_params": n_params,
                    "real": {k: v for k, v in real_stats.items() if not k.startswith("per_landmark")},
                    "null": {k: v for k, v in null_stats.items() if not k.startswith("per_landmark")},
                    "excess_curv_r2": real_stats["curv_r2_mean"] - null_stats["curv_r2_mean"],
                    "per_landmark_real": real_stats["per_landmark_curv_r2"],
                    "per_landmark_null": null_stats["per_landmark_curv_r2"],
                }
                rows.append(row)
                print(
                    f"L{layer} nb={n_nb} d={d}: real curv_r2={real_stats['curv_r2_mean']:+.4f} "
                    f"null={null_stats['curv_r2_mean']:+.4f} excess={row['excess_curv_r2']:+.4f} "
                    f"[{time.time()-t0:.0f}s]",
                    flush=True,
                )
        del xs, real, null
        torch.cuda.empty_cache()

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"WROTE {args.out} rows={len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
