"""Principled adaptive curvature: MDL pricing, zero knobs.

The group-lasso arm needed a lambda nobody can justify. This does not: an atom
keeps its curvature vector V_k iff the training SSE its coordinate removes
exceeds the description-length cost of carrying it, in the SAME convention the
campaign's Rust router declares for DoF-priced admission:

  cost(V_k) [SSE units] = 2 sigma^2 ln2 * bits(V_k)
  bits(V_k) = P * (1/2) log2(N_k)          # the vector's parameters
            + N_k * (1/2) log2(N_k)        # one coordinate scalar per firing,
                                           # priced at estimation precision,
                                           # the same rule as parameters
  sigma^2   = the trained model's own per-element training residual variance
  N_k       = rows routed to atom k by the model's own training-style routing

Everything is measured; nothing is swept. Prune = exact zero. The pruned
dictionary is then scored with the standard held-out greedy+LS eval, reporting
EV and the honest effective scalars per token.

Usage: python3 mdl_prune.py curved_k5262_s0_l8.npz [n_price_rows]
"""

import json
import os
import sys

import numpy as np
import torch

V2 = os.path.expanduser("~/i2502v2")


def ev(x, r):
    return 1.0 - ((x - r) ** 2).sum() / (x ** 2).sum()


def greedy(Ud, Vd, xc, k_act, dev):
    cUV = (Ud * Vd).sum(1)
    vV = (Vd * Vd).sum(1)
    N, K = xc.shape[0], Ud.shape[0]
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
    return picks, tpick


def joint_ls(Ud, Vd, picks, tpick, xc):
    A = (Ud[picks] + tpick.unsqueeze(-1) * Vd[picks]).transpose(1, 2)
    sol = torch.linalg.lstsq(A, xc.unsqueeze(-1)).solution
    return (A @ sol).squeeze(-1)


def main():
    name = sys.argv[1]
    n_price = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    blob = np.load(f"{V2}/{name}")
    Ud = torch.tensor(blob["U"], dtype=torch.float32, device=dev)
    Vd = torch.tensor(blob["V"], dtype=torch.float32, device=dev)
    bp = torch.tensor(blob["b_pre"], dtype=torch.float32, device=dev)
    k_act = int(blob["k_act"])
    K, P = Ud.shape

    Xtr = np.load(f"{V2}/train_chart.npy")[:n_price]
    Xte = np.load(f"{V2}/test_chart.npy")
    lift = np.load(f"{V2}/lift.npy")
    c0 = np.load(f"{V2}/c0.npy")
    amb = np.load(f"{V2}/test_ambient.npy").astype(np.float64)

    xtr = torch.tensor(Xtr, dtype=torch.float32, device=dev) - bp
    with torch.no_grad():
        picks, tpick = greedy(Ud, Vd, xtr, k_act, dev)
        rec = joint_ls(Ud, Vd, picks, tpick, xtr)
        sigma2 = float(((xtr - rec) ** 2).mean())

        # Per-atom SSE with the coordinate vs with the coordinate FROZEN AT
        # ZERO on the same support and the same joint solve — the exact price
        # of the coordinate, no re-routing (pricing is a decoder-side decision;
        # the router's own preference is re-measured after pruning).
        rec_flat = joint_ls(Ud, torch.zeros_like(Vd), picks,
                            torch.zeros_like(tpick), xtr)
        err_curved = ((xtr - rec) ** 2).sum(1)
        err_flat = ((xtr - rec_flat) ** 2).sum(1)
        gain_rows = err_flat - err_curved          # SSE the coordinates remove, per row
        # Attribute each row's coordinate gain to its picked atoms equally at
        # the atom level via counts: exact per-atom attribution would need a
        # leave-one-atom-out solve per (row, slot); the row-level gain divided
        # over its k_act picks is the unbiased even split of a jointly-earned
        # quantity (Shapley over exchangeable slots).
        firings = torch.zeros(K, device=dev)
        gain_atom = torch.zeros(K, device=dev)
        flat_picks = picks.reshape(-1)
        firings.scatter_add_(0, flat_picks, torch.ones_like(flat_picks, dtype=torch.float32))
        share = (gain_rows / k_act).unsqueeze(1).expand(-1, k_act).reshape(-1)
        gain_atom.scatter_add_(0, flat_picks, share)

        n_k = firings.clamp_min(1.0)
        bits = (P * 0.5 + n_k * 0.5) * torch.log2(n_k.clamp_min(2.0))
        cost = 2.0 * sigma2 * float(np.log(2.0)) * bits
        keep = gain_atom > cost
        Vp = Vd * keep.unsqueeze(1)

        # Held-out score of the priced dictionary.
        xte = torch.tensor(Xte, dtype=torch.float32, device=dev) - bp
        picks_te, tpick_te = greedy(Ud, Vp, xte, k_act, dev)
        rec_te = joint_ls(Ud, Vp, picks_te, tpick_te, xte)
        rec_chart = (rec_te + bp).cpu().numpy().astype(np.float64)
        curved_share = float(keep[picks_te].float().mean())

    rec_amb = rec_chart @ lift + c0
    out = {
        "arm": "mdl_priced_curvature", "source": name, "K": K, "L0": k_act,
        "sigma2_train": sigma2, "price_rows": len(Xtr),
        "atoms_kept_curved": int(keep.sum()), "kept_frac": float(keep.float().mean()),
        "curved_pick_share_test": curved_share,
        "effective_scalars_per_token": float(k_act * (1.0 + curved_share)),
        "chart_ev_omp_ls": float(ev(Xte, rec_chart)),
        "ambient_ev": float(ev(amb, rec_amb)),
    }
    print(json.dumps(out, indent=1), flush=True)
    print("MDL_PRUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
