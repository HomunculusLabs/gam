"""Which chart directions, if any, predict the cross-entropy effect?

The per-row correlation between total reconstruction advantage and CE effect is
-0.05 -- they do not covary. That raises the obvious question the dissociation
leaves open: is there ANY component of the reconstruction difference that does
predict functional effect, or is the whole reconstruction signal orthogonal to
what the model uses?

This splits the per-row squared-error difference by chart band and correlates
each band separately with the CE difference on the same rows. If the top
variance directions predict and the tail does not (or vice versa), that
localises where functional relevance lives. If none predict, the independence is
not about WHICH directions are reconstructed but about reconstruction as such.
"""

import glob
import os

import numpy as np

V2 = os.path.expanduser("~/i2502v2")


def main() -> int:
    d = np.load(f"{V2}/per_position_nll.npz")
    rows = d["scored_rows"]
    manifold = np.mean([d[k] for k in d if k.startswith("manifold")], axis=0)
    baseline = np.mean([d[k] for k in d if "k10525" in k], axis=0)
    ce = manifold - baseline

    chart = np.load(f"{V2}/test_chart.npy")
    recon_m = np.frombuffer(
        open(sorted(glob.glob(f"{V2}/a8_s*/heldout_recon.bin"))[0], "rb").read(),
        dtype=np.float64,
    ).reshape(-1, 128)

    blob = np.load(f"{V2}/baseline_k10525_s0.npz")
    W = blob["W_dec"].astype(np.float64)
    bp = blob["b_pre"].astype(np.float64)
    norms = (W * W).sum(1)
    residual = chart - bp
    taken = np.zeros((len(chart), len(W)), dtype=bool)
    picks = np.zeros((len(chart), 8), dtype=np.int64)
    for step in range(8):
        gain = 2.0 * (residual @ W.T) - norms
        gain[taken] = -np.inf
        pick = gain.argmax(1)
        picks[:, step] = pick
        taken[np.arange(len(chart)), pick] = True
        coef = np.maximum(
            (residual * W[pick]).sum(1) / np.maximum(norms[pick], 1e-30), 0.0
        )
        residual = residual - coef[:, None] * W[pick]
    recon_b = np.empty_like(chart)
    for i in range(len(chart)):
        A = W[picks[i]].T
        c, *_ = np.linalg.lstsq(A, chart[i] - bp, rcond=None)
        recon_b[i] = A @ c
    recon_b = recon_b + bp

    # Positive means the manifold reconstructs that band better on that row.
    advantage = ((chart - recon_b) ** 2 - (chart - recon_m) ** 2)[rows]
    var = (chart ** 2).sum(0)

    print("does the manifold's advantage in a CHART BAND predict its CE effect?")
    print(f"{'band':<16}{'var share':>11}{'mean adv':>11}{'corr w/ CE':>13}")
    for lo, hi in ((0, 8), (8, 32), (32, 64), (64, 128)):
        band = advantage[:, lo:hi].sum(1)
        share = var[lo:hi].sum() / var.sum()
        r = np.corrcoef(band, ce)[0, 1]
        print(f"  dims {lo:>3}-{hi:<8}{share:>11.4f}{band.mean():>11.4f}{r:>13.4f}")

    print()
    print(f"total advantage      corr with CE = {np.corrcoef(advantage.sum(1), ce)[0, 1]:+.4f}")
    norm_gap = np.linalg.norm(recon_m[rows], axis=1) - np.linalg.norm(chart[rows], axis=1)
    print(f"reconstruction norm gap corr with CE = {np.corrcoef(norm_gap, ce)[0, 1]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
