"""Baseline arms for the #2502 confirmation run: three seeds, both budgets.

Reports chart EV and AMBIENT EV. Chart EV asks how well 128 PCA directions are
reconstructed; ambient EV asks how well the 2560-d residual stream is, which is
the space the model actually computes in. A method can win the first and be
marginal in the second, so both are reported and never mixed.
"""

import json
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")


def ev(x, r):
    return 1.0 - float(((x - r) ** 2).sum()) / float((x ** 2).sum())


def main() -> int:
    import torch
    K = int(sys.argv[1])
    seed = int(sys.argv[2])
    k_act = 8
    Xtr = np.load(f"{V2}/train_chart.npy")
    Xte = np.load(f"{V2}/test_chart.npy")
    lift = np.load(f"{V2}/lift.npy")
    c0 = np.load(f"{V2}/c0.npy")
    amb = np.load(f"{V2}/test_ambient.npy").astype(np.float64)
    P = Xtr.shape[1]
    dev = "cuda:0"
    t = torch.tensor(Xtr, dtype=torch.float32, device=dev)

    g = torch.Generator(device=dev).manual_seed(seed)
    W_enc = torch.nn.Parameter(torch.randn(P, K, generator=g, device=dev) / P ** 0.5)
    W_dec = torch.nn.Parameter(torch.randn(K, P, generator=g, device=dev) / K ** 0.5)
    b_enc = torch.nn.Parameter(torch.zeros(K, device=dev))
    b_pre = torch.nn.Parameter(t.mean(0).clone())
    opt = torch.optim.Adam([W_enc, W_dec, b_enc, b_pre], lr=3e-4)

    def fwd(x):
        pre = (x - b_pre) @ W_enc + b_enc
        val, idx = torch.topk(pre, k_act, dim=1)
        z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val))
        return z @ W_dec + b_pre

    n = len(t); bs = 4096
    for _ep in range(150):
        perm = torch.randperm(n, generator=g, device=dev)
        for s in range(0, n, bs):
            xb = t[perm[s:s + bs]]
            loss = ((fwd(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        Wd = W_dec.detach().float()
        bp = b_pre.detach().float()
        norms = (Wd * Wd).sum(1)
        x = torch.tensor(Xte, dtype=torch.float32, device=dev)
        xc = x - bp
        R = xc.clone()
        taken = torch.zeros(x.shape[0], K, dtype=torch.bool, device=dev)
        picks = torch.zeros(x.shape[0], k_act, dtype=torch.long, device=dev)
        for s in range(k_act):
            gain = 2.0 * (R @ Wd.T) - norms
            gain = gain.masked_fill(taken, float("-inf"))
            p = gain.argmax(1)
            picks[:, s] = p
            taken.scatter_(1, p.unsqueeze(1), True)
            coef = ((R * Wd[p]).sum(1) / norms[p].clamp_min(1e-30)).clamp_min(0.0)
            R = R - coef.unsqueeze(1) * Wd[p]
        A = Wd[picks].transpose(1, 2)
        sol = torch.linalg.lstsq(A, xc.unsqueeze(-1)).solution
        rec_chart = ((A @ sol).squeeze(-1) + bp).cpu().numpy().astype(np.float64)
        marg = ev(Xte, fwd(x).cpu().numpy().astype(np.float64))

    rec_amb = rec_chart @ lift + c0
    out = {
        "arm": "baseline", "K": K, "seed": seed, "L0": k_act,
        "chart_ev_marginal": marg,
        "chart_ev_omp_ls": ev(Xte, rec_chart),
        "ambient_ev_omp_ls": ev(amb, rec_amb),
    }
    np.savez(f"{V2}/baseline_k{K}_s{seed}.npz",
             W_dec=W_dec.detach().cpu().numpy(), b_pre=b_pre.detach().cpu().numpy(),
             W_enc=W_enc.detach().cpu().numpy(), b_enc=b_enc.detach().cpu().numpy(),
             k_act=k_act)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
