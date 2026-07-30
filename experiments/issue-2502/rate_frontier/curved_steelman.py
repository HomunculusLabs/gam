"""Curved atoms under the steelman's own training recipe.

Every curved-atom number in this campaign so far came out of the Rust
REML/fixed-point lane, which is measured to stall (raw KKT plateau), strand
capacity at high K (12-15% never-routed vs the steelman's 0%), and lose its
scaling slope above ~1M params. So "curvature loses" has only ever been
measured WITH the fitter tax included.

This removes the tax: the identical modern TopK-SAE recipe (Gao et al. 2024 --
tied init, unit-norm decoder, AuxK dead-latent loss, Adam + warmup + cosine,
same epochs, same batch, same split, same eval-time greedy + joint LS), with
exactly one change: each atom is a 1-D curve segment

    contribution_j(x) = a_j * (U_j + t_j * V_j),   a_j >= 0,  t_j in (-1, 1)

instead of a ray a_j * U_j. Amplitude a_j comes from the same TopK ReLU head;
the coordinate t_j comes from a second linear head through tanh. V_j == 0
recovers the flat steelman exactly, which is the validation arm.

A curved atom's decoder costs 2P parameters, so parity arms are
curved K = flat K/2 (e.g. 5262 vs 10525 -> 1.347M decoder params each).

Eval-time greedy gain per atom is maximized over t in closed form: with
alpha = <R,U>, beta = <R,V>, c = <U,V>, v = ||V||^2, the stationarity of
<R, U+tV>^2 / ||U+tV||^2 reduces to the LINEAR condition
t* = (beta - alpha*c) / (alpha*v - beta*c); the maximum over [-1,1] is at
clamp(t*) or an endpoint, all three of which are just evaluated.
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
    flat_validation = len(sys.argv) > 5 and sys.argv[5] == "flat"
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
    if flat_validation:
        V.zero_()
    V = torch.nn.Parameter(V)
    W_enc = torch.nn.Parameter(U.detach().t().clone())
    b_enc = torch.nn.Parameter(torch.zeros(K, device=dev))
    W_t = torch.nn.Parameter(0.01 * torch.randn(P, K, generator=g, device=dev))
    b_t = torch.nn.Parameter(torch.zeros(K, device=dev))
    b_pre = torch.nn.Parameter(t_data.mean(0).clone())
    params = ([U, W_enc, b_enc, b_pre] if flat_validation
              else [U, V, W_enc, b_enc, W_t, b_t, b_pre])
    opt = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))

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
                    azt = az * tcoord
                    aux_recon = az @ U + azt @ V
                    loss = loss + aux_weight * ((residual.detach() - aux_recon) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                U.data /= U.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    # ---- evaluation: greedy with closed-form optimal t, then joint LS over the
    # picked curve directions, then one coordinate-polish round + re-LS ----
    with torch.no_grad():
        Ud = U.detach().float()
        Vd = V.detach().float()
        bp = b_pre.detach().float()
        cUV = (Ud * Vd).sum(1)                     # <U_j, V_j>, ||U_j|| == 1
        vV = (Vd * Vd).sum(1)
        x = torch.tensor(Xte, dtype=torch.float32, device=dev)
        xc = x - bp
        N = x.shape[0]
        R = xc.clone()
        picks = torch.zeros(N, k_act, dtype=torch.long, device=dev)
        tpick = torch.zeros(N, k_act, device=dev)
        taken = torch.zeros(N, K, dtype=torch.bool, device=dev)

        def best_t_gain(alpha, beta):
            # gain(t) = <R,d>^2/||d||^2 with a >= 0 clamp folded in afterwards
            tstar = (beta - alpha * cUV) / (alpha * vV - beta * cUV + 1e-30)
            cands = torch.stack(
                [tstar.clamp(-1.0, 1.0),
                 torch.full_like(tstar, -1.0),
                 torch.full_like(tstar, 1.0)], dim=0)
            num = (alpha.unsqueeze(0) + beta.unsqueeze(0) * cands)
            den = (1.0 + 2.0 * cUV * cands + vV * cands ** 2).clamp_min(1e-12)
            gains = torch.where(num > 0, num ** 2 / den, torch.zeros_like(num))
            gbest, which = gains.max(dim=0)
            tbest = cands.gather(0, which.unsqueeze(0)).squeeze(0)
            return gbest, tbest

        for s in range(k_act):
            alpha = R @ Ud.T
            beta = R @ Vd.T
            gain, tb = best_t_gain(alpha, beta)
            gain = gain.masked_fill(taken, float("-inf"))
            p = gain.argmax(1)
            picks[:, s] = p
            taken.scatter_(1, p.unsqueeze(1), True)
            tp = tb.gather(1, p.unsqueeze(1)).squeeze(1)
            tpick[:, s] = tp
            d = Ud[p] + tp.unsqueeze(1) * Vd[p]
            dn = (d * d).sum(1).clamp_min(1e-30)
            a = ((R * d).sum(1) / dn).clamp_min(0.0)
            R = R - a.unsqueeze(1) * d

        def joint_ls(picks, tpick):
            A = (Ud[picks] + tpick.unsqueeze(-1) * Vd[picks]).transpose(1, 2)
            sol = torch.linalg.lstsq(A, xc.unsqueeze(-1)).solution
            rec = (A @ sol).squeeze(-1)
            return rec, sol.squeeze(-1)

        rec0, amps = joint_ls(picks, tpick)

        # one polish round: re-optimize each t_j against its own partial residual
        for s in range(k_act):
            p = picks[:, s]
            d = Ud[p] + tpick[:, s].unsqueeze(1) * Vd[p]
            partial = xc - rec0 + amps[:, s].unsqueeze(1) * d
            al = (partial * Ud[p]).sum(1)
            be = (partial * Vd[p]).sum(1)
            c_p, v_p = cUV[p], vV[p]
            tstar = ((be - al * c_p) / (al * v_p - be * c_p + 1e-30)).clamp(-1.0, 1.0)
            cands = torch.stack([tstar, tpick[:, s]], dim=0)
            num = al.unsqueeze(0) + be.unsqueeze(0) * cands
            den = (1.0 + 2.0 * c_p * cands + v_p * cands ** 2).clamp_min(1e-12)
            gains = torch.where(num > 0, num ** 2 / den, torch.zeros_like(num))
            which = gains.argmax(0)
            tpick[:, s] = cands.gather(0, which.unsqueeze(0)).squeeze(0)
        rec1, _ = joint_ls(picks, tpick)

        # keep the better of pre/post polish per the aggregate (polish must not lose)
        rec_chart_t = rec1 if ev(Xte, (rec1 + bp).cpu().numpy()) >= ev(Xte, (rec0 + bp).cpu().numpy()) else rec0
        rec_chart = (rec_chart_t + bp).cpu().numpy().astype(np.float64)
        alive = int(torch.unique(picks).numel())
        curv_share = float((tpick.abs() > 1e-3).float().mean())
        vnorm = Vd.norm(dim=1)

    rec_amb = rec_chart @ lift + c0
    out = {
        "arm": "curved_steelman" + ("_flatval" if flat_validation else ""),
        "K": K, "seed": seed, "L0": k_act, "epochs": epochs,
        "decoder_params": int(K * 2 * P),
        "alive_on_test_routing": alive, "dead": K - alive, "dead_frac": (K - alive) / K,
        "curved_pick_share": curv_share,
        "V_norm_median": float(vnorm.median()), "V_norm_p90": float(vnorm.quantile(0.9)),
        "chart_ev_preLS_polish0": float(ev(Xte, (rec0 + bp).cpu().numpy())),
        "chart_ev_polish1": float(ev(Xte, (rec1 + bp).cpu().numpy())),
        "chart_ev_omp_ls": float(ev(Xte, rec_chart)),
        "ambient_ev": float(ev(amb, rec_amb)),
    }
    wout = os.environ.get("WEIGHTS_OUT", "")
    if wout:
        _w = {"U": U.detach().cpu().numpy(), "W_enc": W_enc.detach().cpu().numpy(),
              "b_enc": b_enc.detach().cpu().numpy(), "b_pre": b_pre.detach().cpu().numpy()}
        for _n in ("V", "gamma"):
            if _n in dir() and locals().get(_n) is not None:
                _w[_n] = locals()[_n].detach().cpu().numpy()
        np.savez(wout, **_w)

    print(json.dumps(out, indent=1), flush=True)
    tag = "flatval" if flat_validation else "curved"
    np.savez(f"{V2}/{tag}_k{K}_s{seed}_l{k_act}.npz",
             U=Ud.cpu().numpy(), V=Vd.cpu().numpy(), b_pre=bp.cpu().numpy(),
             W_enc=W_enc.detach().cpu().numpy(), b_enc=b_enc.detach().cpu().numpy(),
             W_t=W_t.detach().cpu().numpy(), b_t=b_t.detach().cpu().numpy(),
             k_act=k_act)
    return 0


if __name__ == "__main__":
    sys.exit(main())
