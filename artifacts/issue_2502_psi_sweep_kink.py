"""Pre-registered psi-sweep kink test for the face-combinatorics account of
superposition (curvature-atoms proposal, section 5).

Setup: the classic toy autoencoder x_hat = ReLU(W^T W x + b) with n=3 features
in d=2, feature directions FIXED at angles (-psi, 0, +psi), features sparse
Bernoulli(p) * U[0,1]. Two arms per psi:

  * FRONTIER: W fixed by psi; only b and a per-feature output gain are
    optimized (Adam, many restarts, huge fixed sample). This is the achievable
    loss of the geometry itself, no SGD-finding-it confound.
  * TRAINED: same, but from random inits with fewer steps — does training
    reach the frontier near the threshold?

Prediction registered BEFORE running: the face-lattice account says the
co-activation pattern {1,2,3} is realizable without structural interference
iff the three directions lie in an open half-space, i.e. iff psi < 90 deg.
Loss restricted to triple-active events should therefore show a KINK at
psi = 90 (a discontinuity in slope), while every pairwise/Gram statistic of W
varies smoothly through 90. A smooth loss-in-psi refutes the combinatorial
account; a kink at exactly 90 refutes any purely Gram-based account.
"""

import json
import sys

import numpy as np
import torch


def run_arm(psi_deg, seed, steps, n_samples, p_active, dev="cpu"):
    g = torch.Generator(device=dev).manual_seed(seed)
    psi = np.deg2rad(psi_deg)
    angles = torch.tensor([-psi, 0.0, psi], dtype=torch.float64)
    W = torch.stack([torch.cos(angles), torch.sin(angles)])  # 2 x 3, fixed
    # One fixed evaluation sample per (psi, seed) so arms differ only in psi.
    active = (torch.rand(n_samples, 3, generator=g, device=dev, dtype=torch.float64) < p_active)
    mag = torch.rand(n_samples, 3, generator=g, device=dev, dtype=torch.float64)
    x = active.double() * mag
    gram = W.t() @ W  # 3 x 3
    b = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    a = torch.ones(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([b, a], lr=3e-2)
    for s in range(steps):
        opt.zero_grad()
        pre = x @ gram + b
        xh = a * torch.relu(pre)
        loss = ((x - xh) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pre = x @ gram + b
        xh = a * torch.relu(pre)
        per_row = ((x - xh) ** 2).sum(1)
        total = float(per_row.mean())
        triple = active.all(1)
        pair12 = active[:, 0] & active[:, 1] & ~active[:, 2]
        loss_triple = float(per_row[triple].mean()) if int(triple.sum()) else float("nan")
        loss_pair = float(per_row[pair12].mean()) if int(pair12.sum()) else float("nan")
    return total, loss_triple, loss_pair, float(triple.double().mean())


def main():
    p_active = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    out = []
    psis = np.round(np.arange(60.0, 121.0, 2.0), 1).tolist() + [88.0, 89.0, 91.0, 92.0]
    psis = sorted(set(psis))
    for psi in psis:
        best = None
        for seed in range(6):
            r = run_arm(psi, seed, steps=4000, n_samples=200_000, p_active=p_active)
            if best is None or r[0] < best[0]:
                best = r
        total, ltrip, lpair, ftrip = best
        rec = {"psi": psi, "loss": total, "loss_triple": ltrip,
               "loss_pair12": lpair, "frac_triple": ftrip, "p_active": p_active}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    with open("psi_sweep_frontier.jsonl", "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
