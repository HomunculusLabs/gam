"""Curvature tomography on a pure-ReLU language model — batched MVP.

WHAT THIS MEASURES, IN PLAIN TERMS. A network built from ReLUs computes a
piecewise-linear function: input space is carved into cells, and inside each
cell the model is exactly linear. All of the model's "decisions" live on the
walls between cells — each wall is where one specific ReLU unit switches
on/off. Crossing a wall changes the model's Jacobian by a rank-one jump
u·nᵀ, where n (the "read" direction) is the input direction that flips the
unit and u (the "write" direction) is what the flip adds to the model's
output. The curvature-atoms program treats these (n, u) pairs — "atoms" —
as the model's canonical feature content.

A "ray" here is a straight line segment through input space: we take a real
64-character context from the corpus, hold every position fixed except the
last, and slide that last character's embedding vector linearly from one real
character's embedding to another's. Along this segment we locate every point
where any MLP ReLU (any layer, any hidden unit — "neuron") flips sign, and
record the atom at each flip with its provenance (which layer, which unit,
where along the segment).

MECHANICS: sign profile of every ReLU pre-activation at 257 points along the
segment (one batched forward), bisection on each detected flip to pin the
crossing (all crossings refined together, one batched forward per bisection
step), then n = input-gradient of the flipping unit's pre-activation (one
batched backward), and u = the second difference of the model output across
the crossing (three batched forwards), which isolates the jump in the
directional derivative that the flip causes.
"""

import json
import os
import sys

import numpy as np
import torch


def load_model(dev):
    ck = torch.load(os.path.expanduser("~/relu_lm_ckpt.pt"), map_location="cpu",
                    weights_only=False)
    sys.path.insert(0, os.path.expanduser("~"))
    from relu_lm import ReluLM
    model = ReluLM(len(ck["vocab_bytes"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model.to(dev).double(), ck["vocab_bytes"]


def main():
    n_rays = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, vocab_bytes = load_model(dev)
    text = open(os.path.expanduser("~/wikitext103_train.txt"), "rb").read(2_000_000)
    lut = {b: i for i, b in enumerate(vocab_bytes)}
    ids_all = np.array([lut[b] for b in text if b in lut], dtype=np.int64)

    ctx = 64
    rng = np.random.default_rng(0)
    rec = {"layer": [], "neuron": [], "t": [], "u_norm": [], "ray": []}
    n_list, u_list = [], []
    emb = model.emb.weight.detach()

    def forward_batch(x_last, base, pos_ids, want_pres=True):
        """Run the model on B copies of one context whose LAST position's
        embedding is replaced by each row of x_last (B, d). Returns the final
        residual at the last position (B, d) and, if asked, every MLP
        pre-activation there (list of (B, 4d))."""
        B = x_last.shape[0]
        h = base.expand(B, -1, -1).clone()
        h[:, -1, :] = x_last
        h = h + model.pos(pos_ids)
        t_len = h.shape[1]
        mask = torch.triu(torch.full((t_len, t_len), float("-inf"),
                                     device=h.device, dtype=h.dtype), 1)
        pres = []
        for blk in model.blocks:
            hn = blk.n1(h)
            a, _ = blk.attn(hn, hn, hn, attn_mask=mask, need_weights=False)
            h = h + a
            pre = blk.up(blk.n2(h))
            if want_pres:
                pres.append(pre[:, -1, :])
            h = h + blk.down(torch.relu(pre))
        return model.norm(h)[:, -1, :], pres

    for ray_i in range(n_rays):
        start = int(rng.integers(0, len(ids_all) - ctx - 2))
        seq = torch.tensor(ids_all[start:start + ctx], device=dev)
        j, k = rng.integers(0, len(emb), size=2)
        e0, e1 = emb[int(j)].double(), emb[int(k)].double()
        base = model.emb(seq.unsqueeze(0)).double().detach().detach()
        pos_ids = torch.arange(ctx, device=dev)

        ts = torch.linspace(0.0, 1.0, 257, device=dev, dtype=torch.float64)
        with torch.no_grad():
            xs = (1 - ts).unsqueeze(1) * e0 + ts.unsqueeze(1) * e1
            _, pres = forward_batch(xs, base, pos_ids)
            signs = torch.cat([p.sign() for p in pres], dim=1)   # (257, total)
        flips = (signs[1:] * signs[:-1] < 0).nonzero()
        if flips.numel() == 0:
            continue
        sizes = [p.shape[1] for p in pres]
        offsets = np.cumsum([0] + sizes)
        seg = flips[:, 0]
        unit = flips[:, 1]
        layer = torch.from_numpy(
            np.searchsorted(offsets, unit.cpu().numpy(), side="right") - 1).to(dev)
        local = unit - torch.from_numpy(offsets[:-1]).to(dev)[layer]

        lo = ts[seg].clone()
        hi = ts[seg + 1].clone()
        hi_sign = signs[seg + 1, unit]
        with torch.no_grad():
            # ~50 halvings take the bracket to f64 resolution on [0,1]; the
            # loop is over bisection STEPS, so each step is one batched forward
            # over all crossings at once.
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                xs = (1 - mid).unsqueeze(1) * e0 + mid.unsqueeze(1) * e1
                _, pres_m = forward_batch(xs, base, pos_ids)
                pre_all = torch.cat(pres_m, dim=1)
                val = pre_all[torch.arange(len(mid), device=dev), unit]
                go_hi = (val.sign() == hi_sign) | (val == 0)
                hi = torch.where(go_hi, mid, hi)
                lo = torch.where(go_hi, lo, mid)
        tstar = 0.5 * (lo + hi)

        # n: input-gradient of each flipping unit's pre-activation, all
        # crossings in one backward (rows of the batch are independent).
        xs = ((1 - tstar).unsqueeze(1) * e0 + tstar.unsqueeze(1) * e1)
        xs = xs.detach().requires_grad_(True)
        _, pres_g = forward_batch(xs, base, pos_ids)
        pre_all = torch.cat(pres_g, dim=1)
        sel = pre_all[torch.arange(len(tstar), device=dev), unit].sum()
        sel.backward()
        n_vecs = xs.grad.detach()
        n_unit = n_vecs / n_vecs.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # u: second difference of the output across each crossing along its
        # own n direction — (f(x+εn) + f(x−εn) − 2f(x))/ε cancels the linear
        # part on both sides and leaves the derivative JUMP the wall causes.
        # ε at the f64 second-difference sweet spot, scaled to the embedding.
        eps = float(np.cbrt(np.finfo(np.float64).eps)) * float(emb.norm(dim=1).median())
        with torch.no_grad():
            x0 = ((1 - tstar).unsqueeze(1) * e0 + tstar.unsqueeze(1) * e1)
            out0, _ = forward_batch(x0, base, pos_ids, want_pres=False)
            outp, _ = forward_batch(x0 + eps * n_unit, base, pos_ids, want_pres=False)
            outm, _ = forward_batch(x0 - eps * n_unit, base, pos_ids, want_pres=False)
            u_vecs = (outp + outm - 2 * out0) / eps
            u_norms = u_vecs.norm(dim=1)
        good = torch.isfinite(u_norms) & (u_norms > 1e-9)
        for i in torch.nonzero(good).flatten().tolist():
            rec["layer"].append(int(layer[i]))
            rec["neuron"].append(int(local[i]))
            rec["t"].append(float(tstar[i]))
            rec["u_norm"].append(float(u_norms[i]))
            rec["ray"].append(ray_i)
            n_list.append(n_unit[i].cpu().numpy())
            u_list.append((u_vecs[i] / u_norms[i]).cpu().numpy())
        if ray_i % 10 == 0:
            print(json.dumps({"ray": ray_i, "atoms": len(rec["t"])}), flush=True)

    np.savez(os.path.expanduser("~/tomograph_atoms.npz"),
             layer=np.array(rec["layer"]), neuron=np.array(rec["neuron"]),
             t=np.array(rec["t"]), u_norm=np.array(rec["u_norm"]),
             ray=np.array(rec["ray"]),
             n=np.stack(n_list) if n_list else np.zeros((0, 1)),
             u=np.stack(u_list) if u_list else np.zeros((0, 1)))
    print(json.dumps({"total_atoms": len(rec["t"]), "rays": n_rays}), flush=True)
    print("TOMOGRAPH_DONE", flush=True)


if __name__ == "__main__":
    main()
