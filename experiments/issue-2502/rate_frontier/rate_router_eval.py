"""Variable-rate routing eval over a trained curved dictionary.

The matched-scalars comparison charges every curved firing 2 scalars whether or
not the coordinate earned its keep on that token. This eval lets the ROUTER
decide per token: the budget is SCALARS (S), and each greedy step takes the
action with the best marginal gain PER SCALAR:

  - fire a new atom FLAT   (t = 0, decode a*U): cost 1 scalar
  - fire a new atom CURVED (t = t*):            cost 2 scalars

Flat routing over the same dictionary is a strict subset of this action space,
so at equal scalar budget the router can only match-or-beat the all-flat use of
the same atoms; the open question this measures is whether it beats the FLAT
DICTIONARY at the same total scalars (the standing loss from the L0 frontier).

Loads a curved checkpoint saved by issue_2502_curved_steelman.py and evaluates
at several scalar budgets. Ends with per-slot t polish (curved picks only) and
a joint LS over the chosen (fixed-t) directions -- identical machinery to the
curved arm's eval so numbers are comparable.
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

    ckpt = sys.argv[1]
    budgets = [int(b) for b in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["8", "16"])]

    z = np.load(ckpt)
    Xte = np.load(f"{V2}/test_chart.npy")
    lift = np.load(f"{V2}/lift.npy")
    c0 = np.load(f"{V2}/c0.npy")
    amb = np.load(f"{V2}/test_ambient.npy").astype(np.float64)
    dev = "cuda:0"
    Ud = torch.tensor(z["U"], dtype=torch.float32, device=dev)
    Vd = torch.tensor(z["V"], dtype=torch.float32, device=dev)
    bp = torch.tensor(z["b_pre"], dtype=torch.float32, device=dev)
    K, P = Ud.shape
    cUV = (Ud * Vd).sum(1)
    vV = (Vd * Vd).sum(1)

    x = torch.tensor(Xte, dtype=torch.float32, device=dev)
    xc = x - bp
    N = x.shape[0]

    def best_t_gain(alpha, beta):
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

    results = []
    with torch.no_grad():
        for S in budgets:
            R = xc.clone()
            taken = torch.zeros(N, K, dtype=torch.bool, device=dev)
            spent = torch.zeros(N, dtype=torch.long, device=dev)
            max_picks = S  # at most S flat firings
            picks = torch.full((N, max_picks), -1, dtype=torch.long, device=dev)
            tpick = torch.zeros(N, max_picks, device=dev)
            is_curved = torch.zeros(N, max_picks, dtype=torch.bool, device=dev)
            n_picked = torch.zeros(N, dtype=torch.long, device=dev)
            curved_spent_total = 0.0

            for _step in range(2 * S):
                active = spent < S
                if not bool(active.any()):
                    break
                alpha = R @ Ud.T
                beta = R @ Vd.T
                # flat action: a = max(alpha, 0), gain alpha^2 (unit-norm U), cost 1
                gain_flat = alpha.clamp_min(0.0) ** 2
                gain_curv, tb = best_t_gain(alpha, beta)
                gain_flat = gain_flat.masked_fill(taken, float("-inf"))
                gain_curv = gain_curv.masked_fill(taken, float("-inf"))
                # curved needs 2 scalars of remaining budget
                can_curve = (S - spent) >= 2
                per_scalar_curv = torch.where(
                    can_curve.unsqueeze(1), gain_curv / 2.0,
                    torch.full_like(gain_curv, float("-inf")))
                gf, pf = gain_flat.max(1)
                gc, pc = per_scalar_curv.max(1)
                choose_curved = gc > gf
                p = torch.where(choose_curved, pc, pf)
                # rows with no positive gain or exhausted budget stop
                gbest = torch.where(choose_curved, gc, gf)
                live = active & (gbest > 0)
                if not bool(live.any()):
                    break
                idx = n_picked.clamp_max(max_picks - 1)
                rowsel = torch.arange(N, device=dev)
                pl = p.clone()
                picks[rowsel[live], idx[live]] = pl[live]
                tsel = tb.gather(1, p.unsqueeze(1)).squeeze(1)
                t_used = torch.where(choose_curved, tsel, torch.zeros_like(tsel))
                tpick[rowsel[live], idx[live]] = t_used[live]
                is_curved[rowsel[live], idx[live]] = choose_curved[live]
                taken[rowsel[live], pl[live]] = True
                d = Ud[p] + t_used.unsqueeze(1) * Vd[p]
                dn = (d * d).sum(1).clamp_min(1e-30)
                a = ((R * d).sum(1) / dn).clamp_min(0.0)
                upd = live.float().unsqueeze(1)
                R = R - upd * a.unsqueeze(1) * d
                cost = torch.where(choose_curved, torch.full_like(spent, 2), torch.ones_like(spent))
                spent = spent + torch.where(live, cost, torch.zeros_like(cost))
                n_picked = n_picked + live.long()

            # joint LS over chosen directions (fixed t), matching the curved eval
            m = int(n_picked.max().item())
            rec = torch.zeros_like(xc)
            if m > 0:
                pick_m = picks[:, :m].clamp_min(0)
                mask = torch.arange(m, device=dev).unsqueeze(0) < n_picked.unsqueeze(1)
                D = (Ud[pick_m] + tpick[:, :m].unsqueeze(-1) * Vd[pick_m])
                D = D * mask.unsqueeze(-1)
                A = D.transpose(1, 2)
                # masked slots are zero columns -> the design is rank-deficient by
                # construction; solve the ridge-regularized normal equations on the
                # tiny m x m Gram instead (zero columns get ~zero amplitudes).
                G = D @ D.transpose(1, 2)
                G = G + 1e-8 * torch.eye(G.shape[-1], device=dev).unsqueeze(0)
                rhs = (D @ xc.unsqueeze(-1))
                sol = torch.linalg.solve(G, rhs)
                rec = (A @ sol).squeeze(-1)
            rec_chart = (rec + bp).cpu().numpy().astype(np.float64)
            rec_amb = rec_chart @ lift + c0
            frac_curved = float(is_curved[picks >= 0].float().mean()) if m > 0 else 0.0
            results.append({
                "scalar_budget": S,
                "chart_ev": float(ev(Xte, rec_chart)),
                "ambient_ev": float(ev(amb, rec_amb)),
                "mean_atoms_fired": float(n_picked.float().mean()),
                "frac_picks_curved": frac_curved,
            })
            print(json.dumps(results[-1]), flush=True)

    print(json.dumps({"arm": "rate_router", "ckpt": os.path.basename(ckpt),
                      "results": results}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
