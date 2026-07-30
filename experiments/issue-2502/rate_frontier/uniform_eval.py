"""One inference procedure for both arms: greedy pursuit with per-atom optimal
amplitude, then R rounds of coordinate polish. No joint LS for anyone.
Field with gamma=0 reduces exactly to the flat case, so the procedure is
form-uniform by construction. EV convention matches the trainers (uncentered)."""
import json, sys
import numpy as np
import torch

dev = "cuda:0"
Xte_np = np.load("/home/ubuntu/i2502/test_chart.npy")
xall = torch.tensor(Xte_np, dtype=torch.float32, device=dev)

def ev_unc(x, r):
    return 1.0 - float(((x - r) ** 2).sum()) / float((x ** 2).sum())

def run(w, form, L0=8, rounds=2):
    U = torch.tensor(w["U"], dtype=torch.float32, device=dev)
    b_pre = torch.tensor(w["b_pre"], dtype=torch.float32, device=dev)
    K = U.shape[0]
    if form == "field":
        MA = torch.tensor(w["MA"], dtype=torch.float32, device=dev)
        MB = torch.tensor(w["MB"], dtype=torch.float32, device=dev)
        g2 = torch.tensor(w["g2"], dtype=torch.float32, device=dev)
        V = U @ MA @ MB.t()
    else:
        V = torch.zeros_like(U)
        g2 = torch.zeros(K, device=dev)
    uu = (U * U).sum(1); uv = (U * V).sum(1); vv = (V * V).sum(1)

    def best_a(al, be, g2v, uuv, uvv, vvv):
        # minimize ||r - aU - g a^2 V||^2 over a>=0 : quartic; solve by Newton from a0=al/uu, few steps + candidates
        a = (al / uuv.clamp_min(1e-9)).clamp_min(0.0)
        for _ in range(8):
            # d/da of -2(a al + g a^2 be) + a^2 uu + 2 g a^3 uv + g^2 a^4 vv
            gr = -2*al - 4*g2v*a*be + 2*a*uuv + 6*g2v*a**2*uvv + 4*g2v**2*a**3*vvv
            h = -4*g2v*be + 2*uuv + 12*g2v*a*uvv + 12*g2v**2*a**2*vvv
            a = (a - gr / h.clamp_min(1e-6)).clamp_min(0.0)
        gain = 2*(a*al + g2v*a**2*be) - (a**2*uuv + 2*g2v*a**3*uvv + g2v**2*a**4*vvv)
        return gain.clamp_min(0.0), a

    recs = []
    B = 4096
    for s0 in range(0, xall.shape[0], B):
        xc = xall[s0:s0+B] - b_pre
        R = xc.clone()
        picks = torch.zeros(xc.shape[0], L0, dtype=torch.long, device=dev)
        apick = torch.zeros(xc.shape[0], L0, device=dev)
        taken = torch.zeros(xc.shape[0], K, dtype=torch.bool, device=dev)
        for s in range(L0):
            al = R @ U.T; be = R @ V.T
            gain, a = best_a(al, be, g2.unsqueeze(0), uu.unsqueeze(0), uv.unsqueeze(0), vv.unsqueeze(0))
            gain = gain.masked_fill(taken, float("-inf"))
            p = gain.argmax(1)
            picks[:, s] = p; taken.scatter_(1, p.unsqueeze(1), True)
            ap = a.gather(1, p.unsqueeze(1)).squeeze(1)
            apick[:, s] = ap
            R = R - (ap.unsqueeze(1) * U[p] + (g2[p] * ap**2).unsqueeze(1) * V[p])
        rec = xc - R
        for _r in range(rounds):
            for s in range(L0):
                p = picks[:, s]; a_old = apick[:, s]
                f_old = a_old.unsqueeze(1) * U[p] + (g2[p] * a_old**2).unsqueeze(1) * V[p]
                partial = xc - rec + f_old
                al = (partial * U[p]).sum(1); be = (partial * V[p]).sum(1)
                gn, an = best_a(al, be, g2[p], uu[p], uv[p], vv[p])
                old_gain = 2*(partial*f_old).sum(1) - (f_old*f_old).sum(1)
                a_fin = torch.where(gn >= old_gain, an, a_old)
                apick[:, s] = a_fin
                rec = rec - f_old + (a_fin.unsqueeze(1) * U[p] + (g2[p] * a_fin**2).unsqueeze(1) * V[p])
        recs.append(rec + b_pre)
    return ev_unc(xall, torch.cat(recs))

out = {}
for d in ("60", "80", "300"):
    for s in range(3):
        fk = f"/home/ubuntu/w_d{d}_flat_s{s}.npz" if d != "300" else f"/home/ubuntu/w_flat_s{s}.npz"
        gk = f"/home/ubuntu/w_d{d}_ff_s{s}.npz" if d != "300" else f"/home/ubuntu/w_ff_s{s}.npz"
        try:
            wf = dict(np.load(fk)); wg = dict(np.load(gk))
        except FileNotFoundError:
            continue
        f_ev = run(wf, "flat"); g_ev = run(wg, "field")
        out[f"d{d}_s{s}"] = {"flat": f_ev, "field": g_ev, "diff": f_ev - g_ev}
        print(f"d{d}_s{s} flat={f_ev:.6f} field={g_ev:.6f} diff={f_ev-g_ev:+.6f}", flush=True)
print(json.dumps(out))
