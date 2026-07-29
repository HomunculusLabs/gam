"""Adaptive mixed dictionary: per-atom curvature that PAYS RENT or leaves.

The measured frontier says flat wins per-scalar (a coordinate buys less than an
extra atom firing) while curved wins per-parameter. A dictionary should not
have to choose globally: group sparsity on each atom's curvature vector V_k
lets every atom decide — atoms whose data bends keep V (2 scalars/firing),
the rest shed it (1 scalar/firing, exactly a flat atom). At equal decoder
parameters this family contains both pure lanes, so trained well it must
match or beat both at their own operating points.

Same recipe as curved_steel.py + one term: lambda_gl * mean_k ||V_k||_2 (group
lasso over atoms). Eval reports the live-curvature census and the effective
scalars/token (1 + share of picks whose atom kept its V).
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
    lambda_gl = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0e-4
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
    V = torch.randn(K, P, generator=g, device=dev)
    V.mul_(0.1 / V.norm(dim=1, keepdim=True))
    V = torch.nn.Parameter(V)
    W_enc = torch.nn.Parameter(U.detach().t().clone())
    b_enc = torch.nn.Parameter(torch.zeros(K, device=dev))
    W_t = torch.nn.Parameter(0.01 * torch.randn(P, K, generator=g, device=dev))
    b_t = torch.nn.Parameter(torch.zeros(K, device=dev))
    b_pre = torch.nn.Parameter(t_data.mean(0).clone())
    opt = torch.optim.Adam([U, V, W_enc, b_enc, W_t, b_t, b_pre], lr=1e-3)

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
            tcoord = torch.tanh(xc @ W_t + b_t)
            val, idx = torch.topk(pre, k_act, dim=1)
            z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val))
            zt = z * tcoord
            recon = z @ U + zt @ V + b_pre
            residual = xb - recon
            loss = (residual ** 2).mean() + lambda_gl * V.norm(dim=1).mean()

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
                    aux_recon = az @ U + (az * tcoord) @ V
                    loss = loss + aux_weight * ((residual.detach() - aux_recon) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                U.data /= U.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    with torch.no_grad():
        Ud, Vd = U.detach().float(), V.detach().float()
        bp = b_pre.detach().float()
        vnorm = Vd.norm(dim=1)
        # An atom's curvature is LIVE if it survived the group penalty at a
        # level the decode can see: ||V|| above f64-noise relative to ||U||=1.
        live = vnorm > 1.0e-3
        Vd = Vd * live.unsqueeze(1)
        cUV = (Ud * Vd).sum(1)
        vV = (Vd * Vd).sum(1)
        x = torch.tensor(Xte, dtype=torch.float32, device=dev)
        xc = x - bp
        N = x.shape[0]
        R = xc.clone()
        picks = torch.zeros(N, k_act, dtype=torch.long, device=dev)
        tpick = torch.zeros(N, k_act, device=dev)
        taken = torch.zeros(N, K, dtype=torch.bool, device=dev)
        for s in range(k_act):
            alpha = R @ Ud.T
            beta = R @ Vd.T
            tstar = ((beta - alpha * cUV) / (alpha * vV - beta * cUV + 1e-30)).clamp(-1, 1)
            best_g = torch.full((N, K), float("-inf"), device=dev)
            best_t = torch.zeros(N, K, device=dev)
            for tc in (tstar, torch.full_like(tstar, -1.0), torch.full_like(tstar, 1.0)):
                num = alpha + beta * tc
                den = (1.0 + 2.0 * cUV * tc + vV * tc ** 2).clamp_min(1e-12)
                gain = torch.where(num > 0, num ** 2 / den, torch.zeros_like(num))
                better = gain > best_g
                best_g = torch.where(better, gain, best_g)
                best_t = torch.where(better, tc, best_t)
            best_g = best_g.masked_fill(taken, float("-inf"))
            p = best_g.argmax(1)
            picks[:, s] = p
            taken.scatter_(1, p.unsqueeze(1), True)
            tp = best_t.gather(1, p.unsqueeze(1)).squeeze(1)
            tpick[:, s] = tp
            d = Ud[p] + tp.unsqueeze(1) * Vd[p]
            dn = (d * d).sum(1).clamp_min(1e-30)
            a = ((R * d).sum(1) / dn).clamp_min(0.0)
            R = R - a.unsqueeze(1) * d
        A = (Ud[picks] + tpick.unsqueeze(-1) * Vd[picks]).transpose(1, 2)
        sol = torch.linalg.lstsq(A, xc.unsqueeze(-1)).solution
        rec_chart = ((A @ sol).squeeze(-1) + bp).cpu().numpy().astype(np.float64)
        alive = int(torch.unique(picks).numel())
        curved_pick_share = float(live[picks].float().mean())

    rec_amb = rec_chart @ lift + c0
    out = {
        "arm": "mixed_adaptive", "K": K, "seed": seed, "L0": k_act,
        "lambda_gl": lambda_gl, "epochs": epochs,
        "decoder_params": int(K * 2 * P),
        "alive_on_test_routing": alive, "dead": K - alive,
        "atoms_with_live_curvature": int(live.sum()),
        "live_curvature_frac": float(live.float().mean()),
        "curved_pick_share": curved_pick_share,
        "effective_scalars_per_token": float(k_act * (1.0 + curved_pick_share)),
        "V_norm_median_live": float(vnorm[live].median()) if int(live.sum()) else 0.0,
        "chart_ev_omp_ls": float(ev(Xte, rec_chart)),
        "ambient_ev": float(ev(amb, rec_amb)),
    }
    print(json.dumps(out, indent=1), flush=True)
    np.savez(f"{V2}/mixed_k{K}_s{seed}_l{k_act}_gl{lambda_gl}.npz",
             U=Ud.cpu().numpy(), V=Vd.cpu().numpy(), b_pre=bp.cpu().numpy(),
             k_act=k_act, lambda_gl=lambda_gl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
