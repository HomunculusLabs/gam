"""One-scalar-per-firing atom FAMILIES, one trainer.

The ambitious core of the manifold SAE is that each atom is a low-dimensional
geometric object and the router pays per scalar. This screens the 1-D family
ladder at IDENTICAL rate (1 scalar per firing = L0 = scalars/token, exactly
like flat) and matched decoder parameters:

  form=ray    f(a) = a*U + g2*a^2*V2                (2P+1 params/atom)
  form=cubic  f(a) = a*U + g2*a^2*V2 + g3*a^3*V3    (3P+2 params/atom)
  form=offset f(a) = B + a*U  (fires -> adds offset) (2P   params/atom)

flat (f = a*U, P params) and the 2-scalar segment a*(U+t*V) live in
curved_steelman.py; together the five forms are the geometry ladder
constant -> linear -> quadratic -> cubic in the single firing scalar.

Same recipe throughout (tied init, unit-norm U, AuxK, Adam+warmup+cosine).
Eval: greedy argmax of g(a) = 2<R,f(a)> - ||f(a)||^2 with per-form exact or
Newton amplitude solve + flat-solution fallback, then 2 polish rounds.
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

    form = sys.argv[1]
    K = int(sys.argv[2])
    seed = int(sys.argv[3])
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    k_act = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    assert form in ("ray", "cubic", "offset", "field")
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
    W_enc = torch.nn.Parameter(U.detach().t().clone())
    b_enc = torch.nn.Parameter(torch.zeros(K, device=dev))
    b_pre = torch.nn.Parameter(t_data.mean(0).clone())
    params = [U, W_enc, b_enc, b_pre]

    V2p = V3p = g2 = g3 = B = MA = MB = None
    if form == "field":
        # one global low-rank curvature field: V_k = (MA @ MB^T) U_k.
        r = int(os.environ.get("FIELD_RANK", "16"))
        MA = torch.nn.Parameter(0.1 * torch.randn(P, r, generator=g, device=dev))
        MB = torch.nn.Parameter(0.1 * torch.randn(P, r, generator=g, device=dev))
        g2 = torch.nn.Parameter(0.05 * torch.ones(K, device=dev))
        params += [MA, MB, g2]
    if form in ("ray", "cubic"):
        V2p = torch.randn(K, P, generator=g, device=dev)
        V2p.mul_(0.1 / V2p.norm(dim=1, keepdim=True))
        V2p = torch.nn.Parameter(V2p)
        g2 = torch.nn.Parameter(0.05 * torch.ones(K, device=dev))
        params += [V2p, g2]
    if form == "cubic":
        V3p = torch.randn(K, P, generator=g, device=dev)
        V3p.mul_(0.1 / V3p.norm(dim=1, keepdim=True))
        V3p = torch.nn.Parameter(V3p)
        g3 = torch.nn.Parameter(0.02 * torch.ones(K, device=dev))
        params += [V3p, g3]
    if form == "offset":
        B = torch.nn.Parameter(torch.zeros(K, P, device=dev))
        params += [B]

    curv_decay = float(os.environ.get("CURV_DECAY", "0.0"))
    if form == "field" and curv_decay > 0.0:
        # Gaussian prior on the curvature field only (decoupled weight decay on
        # MA/MB/gamma) -- the SGD surrogate of REML-chosen effective capacity:
        # ranks are nested under shrinkage, so capacity is data-selected, not a
        # hand knob. U/encoder/bias stay unpenalized exactly as in flat.
        opt = torch.optim.AdamW([
            {"params": [U, W_enc, b_enc, b_pre], "weight_decay": 0.0},
            {"params": [MA, MB, g2], "weight_decay": curv_decay},
        ], lr=1e-3, betas=(0.9, 0.999))
    else:
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

    def decode(z):
        rec = z @ U
        if form == "field":
            V_eff = U @ MA @ MB.t()
            rec = rec + ((z * z) * g2) @ V_eff
        if form in ("ray", "cubic"):
            rec = rec + ((z * z) * g2) @ V2p
        if form == "cubic":
            rec = rec + ((z * z * z) * g3) @ V3p
        if form == "offset":
            rec = rec + (z > 0).float() @ B
        return rec

    for _ep in range(epochs):
        perm = torch.randperm(n, generator=g, device=dev)
        for s in range(0, n, bs):
            xb = t_data[perm[s:s + bs]]
            xc = xb - b_pre
            pre = xc @ W_enc + b_enc
            val, idx = torch.topk(pre, k_act, dim=1)
            z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val))
            recon = decode(z) + b_pre
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
                    loss = loss + aux_weight * ((residual.detach() - decode(az)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                U.data /= U.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    # ---------------- eval ----------------
    with torch.no_grad():
        Ud = U.detach().float()
        bp = b_pre.detach().float()
        x = torch.tensor(Xte, dtype=torch.float32, device=dev)
        xc = x - bp
        N = x.shape[0]
        R = xc.clone()
        picks = torch.zeros(N, k_act, dtype=torch.long, device=dev)
        apick = torch.zeros(N, k_act, device=dev)
        taken = torch.zeros(N, K, dtype=torch.bool, device=dev)

        if form == "field":
            V2d = (Ud @ MA.detach().float() @ MB.detach().float().t())
            g2d = g2.detach().float()
            c2 = (Ud * V2d).sum(1)
            v2 = (V2d * V2d).sum(1)
        if form in ("ray", "cubic"):
            V2d = V2p.detach().float()
            g2d = g2.detach().float()
            c2 = (Ud * V2d).sum(1)
            v2 = (V2d * V2d).sum(1)
        if form == "cubic":
            V3d = V3p.detach().float()
            g3d = g3.detach().float()
            c3 = (Ud * V3d).sum(1)
            v3 = (V3d * V3d).sum(1)
            c23 = (V2d * V3d).sum(1)
        if form == "offset":
            Bd = B.detach().float()
            bU = (Bd * Ud).sum(1)
            bb = (Bd * Bd).sum(1)

        def g_of_a(a, al, idx2):
            # g(a) = 2<R,f(a)> - ||f(a)||^2 for the row-selected atoms idx2
            if form == "offset":
                return (2 * (al + a * 0) - 0)  # unused; offset handled separately
            f_dot_R = a * al
            f_norm = a ** 2
            if form in ("ray", "cubic"):
                pass
            return 2 * f_dot_R - f_norm

        def gain_and_a(al, be2=None, be3=None, gm2=None, gm3=None,
                       cc2=None, vv2=None, cc3=None, vv3=None, cc23=None):
            a = al.clamp_min(0.0)
            if form in ("ray", "field"):
                for _ in range(3):
                    gp = 2 * al + (4 * gm2 * be2 - 2.0) * a - 6 * gm2 * cc2 * a ** 2 \
                        - 4 * gm2 ** 2 * vv2 * a ** 3
                    gpp = (4 * gm2 * be2 - 2.0) - 12 * gm2 * cc2 * a - 12 * gm2 ** 2 * vv2 * a ** 2
                    a = (a - gp / gpp.clamp(max=-1e-6)).clamp_min(0.0)

                def gval(av):
                    return (2 * av * al + 2 * gm2 * av ** 2 * be2 - av ** 2
                            - 2 * gm2 * cc2 * av ** 3 - gm2 ** 2 * vv2 * av ** 4)
            elif form == "cubic":
                for _ in range(4):
                    # d/da of 2(a al + g2 a^2 be2 + g3 a^3 be3)
                    #        - (a^2 + 2 g2 c2 a^3 + (g2^2 v2 + 2 g3 c3) a^4
                    #           + 2 g2 g3 c23 a^5 + g3^2 v3 a^6)
                    gp = (2 * al + (4 * gm2 * be2 - 2.0) * a
                          + (6 * gm3 * be3 - 6 * gm2 * cc2) * a ** 2
                          - 4 * (gm2 ** 2 * vv2 + 2 * gm3 * cc3) * a ** 3
                          - 10 * gm2 * gm3 * cc23 * a ** 4
                          - 6 * gm3 ** 2 * vv3 * a ** 5)
                    gpp = ((4 * gm2 * be2 - 2.0)
                           + 2 * (6 * gm3 * be3 - 6 * gm2 * cc2) * a
                           - 12 * (gm2 ** 2 * vv2 + 2 * gm3 * cc3) * a ** 2
                           - 40 * gm2 * gm3 * cc23 * a ** 3
                           - 30 * gm3 ** 2 * vv3 * a ** 4)
                    a = (a - gp / gpp.clamp(max=-1e-6)).clamp_min(0.0)

                def gval(av):
                    return (2 * (av * al + gm2 * av ** 2 * be2 + gm3 * av ** 3 * be3)
                            - (av ** 2 + 2 * gm2 * cc2 * av ** 3
                               + (gm2 ** 2 * vv2 + 2 * gm3 * cc3) * av ** 4
                               + 2 * gm2 * gm3 * cc23 * av ** 5
                               + gm3 ** 2 * vv3 * av ** 6))
            a0 = al.clamp_min(0.0)
            gn, gf = gval(a), gval(a0)
            better = gn >= gf
            return torch.where(better, gn, gf).clamp_min(0.0), torch.where(better, a, a0)

        for s in range(k_act):
            al = R @ Ud.T
            if form in ("ray", "field"):
                be2 = R @ V2d.T
                gain, ab = gain_and_a(al, be2=be2, gm2=g2d.unsqueeze(0),
                                      cc2=c2.unsqueeze(0), vv2=v2.unsqueeze(0))
            elif form == "cubic":
                be2 = R @ V2d.T
                be3 = R @ V3d.T
                gain, ab = gain_and_a(al, be2=be2, be3=be3,
                                      gm2=g2d.unsqueeze(0), gm3=g3d.unsqueeze(0),
                                      cc2=c2.unsqueeze(0), vv2=v2.unsqueeze(0),
                                      cc3=c3.unsqueeze(0), vv3=v3.unsqueeze(0),
                                      cc23=c23.unsqueeze(0))
            else:  # offset: f = B + aU, a* = al - <B,U>; g = 2(rB + a al) - (bb + 2 a bU + a^2)
                rB = R @ Bd.T
                a_star = (al - bU.unsqueeze(0)).clamp_min(0.0)
                gain = (2 * (rB + a_star * al)
                        - (bb.unsqueeze(0) + 2 * a_star * bU.unsqueeze(0) + a_star ** 2)).clamp_min(0.0)
                ab = a_star
            gain = gain.masked_fill(taken, float("-inf"))
            p = gain.argmax(1)
            picks[:, s] = p
            taken.scatter_(1, p.unsqueeze(1), True)
            ap = ab.gather(1, p.unsqueeze(1)).squeeze(1)
            apick[:, s] = ap
            if form in ("ray", "field"):
                f = ap.unsqueeze(1) * Ud[p] + (g2d[p] * ap ** 2).unsqueeze(1) * V2d[p]
            elif form == "cubic":
                f = (ap.unsqueeze(1) * Ud[p] + (g2d[p] * ap ** 2).unsqueeze(1) * V2d[p]
                     + (g3d[p] * ap ** 3).unsqueeze(1) * V3d[p])
            else:
                f = Bd[p] + ap.unsqueeze(1) * Ud[p]
            R = R - f

        def f_of(p, a):
            if form in ("ray", "field"):
                return a.unsqueeze(1) * Ud[p] + (g2d[p] * a ** 2).unsqueeze(1) * V2d[p]
            if form == "cubic":
                return (a.unsqueeze(1) * Ud[p] + (g2d[p] * a ** 2).unsqueeze(1) * V2d[p]
                        + (g3d[p] * a ** 3).unsqueeze(1) * V3d[p])
            return Bd[p] + a.unsqueeze(1) * Ud[p]

        rec = torch.zeros_like(xc)
        for s in range(k_act):
            rec += f_of(picks[:, s], apick[:, s])

        for _round in range(2):
            for s in range(k_act):
                p = picks[:, s]
                a_old = apick[:, s]
                f_old = f_of(p, a_old)
                partial = xc - rec + f_old
                al = (partial * Ud[p]).sum(1)
                if form in ("ray", "field"):
                    be2 = (partial * V2d[p]).sum(1)
                    gn, an = gain_and_a(al, be2=be2, gm2=g2d[p], cc2=c2[p], vv2=v2[p])
                elif form == "cubic":
                    be2 = (partial * V2d[p]).sum(1)
                    be3 = (partial * V3d[p]).sum(1)
                    gn, an = gain_and_a(al, be2=be2, be3=be3, gm2=g2d[p], gm3=g3d[p],
                                        cc2=c2[p], vv2=v2[p], cc3=c3[p], vv3=v3[p], cc23=c23[p])
                else:
                    rB = (partial * Bd[p]).sum(1)
                    an = (al - bU[p]).clamp_min(0.0)
                    gn = (2 * (rB + an * al) - (bb[p] + 2 * an * bU[p] + an ** 2)).clamp_min(0.0)
                f_new = f_of(p, an)
                old_gain = 2 * (partial * f_old).sum(1) - (f_old * f_old).sum(1)
                keep = gn >= old_gain
                a_fin = torch.where(keep, an, a_old)
                apick[:, s] = a_fin
                rec = rec - f_old + f_of(p, a_fin)

        rec_chart = (rec + bp).cpu().numpy().astype(np.float64)
        alive = int(torch.unique(picks).numel())

    rec_amb = rec_chart @ lift + c0
    ppa = {"ray": 2 * P + 1, "cubic": 3 * P + 2, "offset": 2 * P, "field": P + 2}[form]
    out = {
        "arm": f"form_{form}",
        "K": K, "seed": seed, "L0_equals_scalars": k_act, "epochs": epochs, "field_rank": int(os.environ.get("FIELD_RANK", "16")) if form == "field" else None, "curv_decay": curv_decay,
        "decoder_params": int(K * ppa),
        "alive_on_test_routing": alive, "dead_frac": (K - alive) / K,
        "chart_ev": float(ev(Xte, rec_chart)),
        "ambient_ev": float(ev(amb, rec_amb)),
    }
    print(json.dumps(out, indent=1), flush=True)
    save = {"U": Ud.cpu().numpy(), "b_pre": bp.cpu().numpy(),
            "W_enc": W_enc.detach().cpu().numpy(), "b_enc": b_enc.detach().cpu().numpy(),
            "k_act": k_act}
    if form in ("ray", "cubic", "field"):
        save.update(V2=V2d.cpu().numpy(), g2=g2d.cpu().numpy())
    if form == "cubic":
        save.update(V3=V3d.cpu().numpy(), g3=g3d.cpu().numpy())
    if form == "offset":
        save.update(B=Bd.cpu().numpy())
    np.savez(f"{V2}/{form}_k{K}_s{seed}_l{k_act}_e{epochs}.npz", **save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
