"""Evaluate a saved flat steelman npz under the same greedy+LS convention the
curved arms use (optimal-amplitude gain, joint LS), so cross-arm numbers share
one eval. For flat atoms the coordinate machinery is inert, so this is the
V=0 special case of the curved eval."""

import json
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")


def ev(x, r):
    return 1.0 - ((x - r) ** 2).sum() / (x ** 2).sum()


def main() -> int:
    import torch

    path = sys.argv[1]
    k_act = 8
    d = np.load(path)
    Xte = np.load(f"{V2}/test_chart.npy")
    lift = np.load(f"{V2}/lift.npy")
    c0 = np.load(f"{V2}/c0.npy")
    amb = np.load(f"{V2}/test_ambient.npy").astype(np.float64)
    dev = "cuda:0"
    Wd = torch.tensor(d["W_dec"], dtype=torch.float32, device=dev)
    bp = torch.tensor(d["b_pre"], dtype=torch.float32, device=dev)
    K = Wd.shape[0]
    norms = (Wd * Wd).sum(1)
    x = torch.tensor(Xte, dtype=torch.float32, device=dev)
    xc = x - bp
    R = xc.clone()
    N = x.shape[0]
    taken = torch.zeros(N, K, dtype=torch.bool, device=dev)
    picks = torch.zeros(N, k_act, dtype=torch.long, device=dev)
    with torch.no_grad():
        for s in range(k_act):
            proj = R @ Wd.T
            gain = torch.where(proj > 0, proj ** 2 / norms.clamp_min(1e-30), torch.zeros_like(proj))
            gain = gain.masked_fill(taken, float("-inf"))
            p = gain.argmax(1)
            picks[:, s] = p
            taken.scatter_(1, p.unsqueeze(1), True)
            a = ((R * Wd[p]).sum(1) / norms[p].clamp_min(1e-30)).clamp_min(0.0)
            R = R - a.unsqueeze(1) * Wd[p]
        A = Wd[picks].transpose(1, 2)
        sol = torch.linalg.lstsq(A, xc.unsqueeze(-1)).solution
        rec_chart = ((A @ sol).squeeze(-1) + bp).cpu().numpy().astype(np.float64)
        alive = int(torch.unique(picks).numel())
    rec_amb = rec_chart @ lift + c0
    print(json.dumps({
        "arm": "flat_myconv", "npz": os.path.basename(path), "K": K, "L0": k_act,
        "decoder_params": int(K * Wd.shape[1]),
        "alive_on_test_routing": alive,
        "chart_ev_omp_ls": float(ev(Xte, rec_chart)),
        "ambient_ev": float(ev(amb, rec_amb)),
    }, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
