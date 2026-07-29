"""The per-feature-bias TopK baseline the critique names as decisive.

The affine rung's reconstruction is exactly  x_hat = sum_{k in S} b_k
+ sum_{k in S} t_k w_k : an unweighted sum of selected offsets (a pure
combinatorial code fired by the gate) plus an ordinary sparse linear code. If
a TopK SAE with a per-feature GATED BIAS at identical decoder parameter count
matches the curved/affine arms, the "manifold" result is a bias result; if it
does not, the remaining difference really is the coordinate.

Atom k, when selected: contributes  B_k + a_k * U_k  with a_k >= 0 from the
TopK ReLU head and B_k a free per-feature offset vector fired at weight 1 by
selection alone. Decoder cost 2P per atom — same as a curved atom, so parity
arms match curved K for K. Training recipe, split, and eval conventions are
byte-identical to curved_steel.py / baseline_steel.py (Gao et al. 2024:
tied init, unit-norm U, AuxK on dead latents, Adam + warmup + cosine).

Eval greedy gain with an offset is exact and closed-form:
  gain_k = 2<R,B_k> - ||B_k||^2 + max(<R - B_k, U_k>, 0)^2   (U unit-norm),
then joint LS on the amplitudes against  xc - sum_selected B_k.
"""

import json
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")


def ev(x, r):
    return 1.0 - ((x - r) ** 2).sum() / (x ** 2).sum()


def main() -> int:
    import torch

    K = int(sys.argv[1])
    seed = int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    k_act = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    k_aux = 512
    aux_weight = 1.0 / 32.0

    Xtr = np.load(f"{V2}/train_chart.npy")
    Xte = np.load(f"{V2}/test_chart.npy")
    lift = np.load(f"{V2}/lift.npy")
    c0 = np.load(f"{V2}/c0.npy")
    amb = np.load(f"{V2}/test_ambient.npy").astype(np.float64)
    P = Xtr.shape[1]
    dev = "cuda:0"
    t_data = torch.tensor(Xtr, dtype=torch.float32, device=dev)

    g = torch.Generator(device=dev).manual_seed(seed)
    U = torch.randn(K, P, generator=g, device=dev)
    U /= U.norm(dim=1, keepdim=True)
    U = torch.nn.Parameter(U)
    B = torch.nn.Parameter(0.01 * torch.randn(K, P, generator=g, device=dev))
    W_enc = torch.nn.Parameter(U.detach().t().clone())
    b_enc = torch.nn.Parameter(torch.zeros(K, device=dev))
    b_pre = torch.nn.Parameter(t_data.mean(0).clone())
    opt = torch.optim.Adam([U, B, W_enc, b_enc, b_pre], lr=1e-3, betas=(0.9, 0.999))

    n = len(t_data)
    bs = 4096
    steps_per_epoch = (n + bs - 1) // bs
    total = epochs * steps_per_epoch
    warm = max(1, total // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1.0 + np.cos(np.pi * (s - warm) / max(1, total - warm))),
    )

    last_fired = torch.zeros(K, dtype=torch.long, device=dev)

    for _ep in range(epochs):
        perm = torch.randperm(n, generator=g, device=dev)
        for s in range(0, n, bs):
            xb = t_data[perm[s:s + bs]]
            xc = xb - b_pre
            pre = xc @ W_enc + b_enc
            val, idx = torch.topk(pre, k_act, dim=1)
            z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val))
            gate = torch.zeros_like(pre).scatter_(1, idx, torch.ones_like(val))
            recon = z @ U + gate @ B + b_pre
            residual = xb - recon
            loss = (residual ** 2).mean()

            with torch.no_grad():
                fired = torch.zeros(K, dtype=torch.bool, device=dev)
                fired.scatter_(0, idx.reshape(-1), True)
                last_fired += 1
                last_fired[fired] = 0
                dead_mask = last_fired > 8 * steps_per_epoch
            if dead_mask.any():
                dead_pre = pre.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
                kk = int(min(k_aux, int(dead_mask.sum().item())))
                if kk > 0:
                    aval, aidx = torch.topk(dead_pre, kk, dim=1)
                    az = torch.zeros_like(pre).scatter_(1, aidx, torch.relu(aval))
                    agate = torch.zeros_like(pre).scatter_(1, aidx, torch.ones_like(aval))
                    aux_recon = az @ U + agate @ B
                    loss = loss + aux_weight * ((residual.detach() - aux_recon) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                U.data /= U.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    with torch.no_grad():
        Ud = U.detach().float()
        Bd = B.detach().float()
        bp = b_pre.detach().float()
        bb = (Bd * Bd).sum(1)
        x = torch.tensor(Xte, dtype=torch.float32, device=dev)
        xc = x - bp
        R = xc.clone()
        N = x.shape[0]
        taken = torch.zeros(N, K, dtype=torch.bool, device=dev)
        picks = torch.zeros(N, k_act, dtype=torch.long, device=dev)
        for s in range(k_act):
            rb = R @ Bd.T
            ru = R @ Ud.T
            bu = (Bd * Ud).sum(1)
            amp = (ru - bu.unsqueeze(0)).clamp_min(0.0)
            gain = 2.0 * rb - bb.unsqueeze(0) + amp ** 2
            gain = gain.masked_fill(taken, float("-inf"))
            p = gain.argmax(1)
            picks[:, s] = p
            taken.scatter_(1, p.unsqueeze(1), True)
            a = ((R - Bd[p]) * Ud[p]).sum(1).clamp_min(0.0)
            R = R - Bd[p] - a.unsqueeze(1) * Ud[p]
        offsets = Bd[picks].sum(1)
        A = Ud[picks].transpose(1, 2)
        rhs = (xc - offsets).unsqueeze(-1)
        sol = torch.linalg.lstsq(A, rhs).solution
        rec_chart = ((A @ sol).squeeze(-1) + offsets + bp).cpu().numpy().astype(np.float64)
        alive = int(torch.unique(picks).numel())

    rec_amb = rec_chart @ lift + c0
    out = {
        "arm": "bias_topk_steelman", "K": K, "seed": seed, "L0": k_act,
        "epochs": epochs, "decoder_params": int(K * 2 * P),
        "alive_on_test_routing": alive, "dead": K - alive, "dead_frac": (K - alive) / K,
        "B_norm_median": float(Bd.norm(dim=1).median()),
        "chart_ev_omp_ls": float(ev(Xte, rec_chart)),
        "ambient_ev": float(ev(amb, rec_amb)),
    }
    print(json.dumps(out, indent=1), flush=True)
    np.savez(f"{V2}/biask_k{K}_s{seed}_l{k_act}.npz",
             U=Ud.cpu().numpy(), B=Bd.cpu().numpy(), b_pre=bp.cpu().numpy(),
             W_enc=W_enc.detach().cpu().numpy(), b_enc=b_enc.detach().cpu().numpy(),
             k_act=k_act)
    return 0


if __name__ == "__main__":
    sys.exit(main())
