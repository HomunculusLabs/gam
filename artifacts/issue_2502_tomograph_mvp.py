"""Curvature tomography MVP on the pure-ReLU LM (curvature-atoms program).

Rays through embedding space; every ReLU flip along the ray is a facet
crossing carrying an exact rank-one Jacobian jump u_F n_F^T (Theorem 1). This
MVP harvests atoms with provenance from the trained relu_lm_ckpt.pt:

  * ray: interpolate the LAST position's token embedding between two real
    tokens inside a real context (attention sees a smooth path; every MLP
    ReLU's pre-activation is a piecewise-smooth function of t whose sign
    changes are located by dense sampling + bisection);
  * per crossing: (layer, neuron, t*), n_F = grad_{x}(preact) at t* (VJP),
    u_F = effect of the unit's post-ReLU bump on the final residual (JVP
    through the frozen downstream network).

Outputs a .npz of atoms: layer, neuron, t, ||u||, n-vectors and u-vectors
(unit-normalized) — the raw material for clustering and the SAE duality test.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def load_model():
    ck = torch.load(os.path.expanduser("~/relu_lm_ckpt.pt"), map_location="cpu",
                    weights_only=False)
    sys.path.insert(0, os.path.expanduser("~"))
    from relu_lm import ReluLM
    vocab = len(ck["vocab_bytes"])
    model = ReluLM(vocab)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["vocab_bytes"]


def main():
    n_rays = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, vocab_bytes = load_model()
    model = model.to(dev).double()
    text = open(os.path.expanduser("~/wikitext103_train.txt"), "rb").read(2_000_000)
    lut = {b: i for i, b in enumerate(vocab_bytes)}
    ids_all = np.array([lut[b] for b in text if b in lut], dtype=np.int64)

    ctx = 64
    rng = np.random.default_rng(0)
    atoms = {"layer": [], "neuron": [], "t": [], "u_norm": [], "ray": []}
    n_list, u_list = [], []

    emb = model.emb.weight.detach()

    def forward_from_embed(x_last, base_embed, pos_ids):
        # base_embed: (1, ctx, d) with the last position REPLACED by x_last.
        h = base_embed.clone()
        h[0, -1, :] = x_last
        h = h + model.pos(pos_ids)
        t_len = h.shape[1]
        mask = torch.triu(torch.full((t_len, t_len), float("-inf"),
                                     device=h.device, dtype=h.dtype), 1)
        pres = []
        for blk in model.blocks:
            hn = blk.n1(h)
            a, _ = blk.attn(hn, hn, hn, attn_mask=mask, need_weights=False)
            h = h + a
            pre = blk.up(blk.n2(h))          # (1, ctx, 4d) pre-ReLU
            pres.append(pre[0, -1, :])       # last position only
            h = h + blk.down(torch.relu(pre))
        return model.norm(h)[0, -1, :], pres

    for ray_i in range(n_rays):
        start = int(rng.integers(0, len(ids_all) - ctx - 2))
        seq = torch.tensor(ids_all[start:start + ctx], device=dev)
        j, k = rng.integers(0, len(emb), size=2)
        e0, e1 = emb[int(j)].double(), emb[int(k)].double()
        base = model.emb(seq.unsqueeze(0)).double()
        pos_ids = torch.arange(ctx, device=dev)

        # Dense sign profile of every MLP pre-activation at the last position.
        ts = torch.linspace(0.0, 1.0, 257, device=dev, dtype=torch.float64)
        signs = []
        with torch.no_grad():
            for t in ts:
                x = (1 - t) * e0 + t * e1
                _, pres = forward_from_embed(x, base, pos_ids)
                signs.append(torch.cat([p.sign() for p in pres]))
        signs = torch.stack(signs)          # (T, total_units)
        flips = (signs[1:] * signs[:-1] < 0).nonzero()
        sizes = [p.numel() for p in
                 forward_from_embed(e0, base, pos_ids)[1]]
        offsets = np.cumsum([0] + sizes)

        for seg, unit in flips.tolist():
            lo, hi = float(ts[seg]), float(ts[seg + 1])
            layer = int(np.searchsorted(offsets, unit, side="right") - 1)
            local = unit - int(offsets[layer])
            # Bisection to the crossing.
            with torch.no_grad():
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    x = (1 - mid) * e0 + mid * e1
                    _, pres = forward_from_embed(x, base, pos_ids)
                    if float(pres[layer][local]) > 0.0:
                        hi_pos = float(signs[seg + 1, unit]) > 0
                        (lo, hi) = (lo, mid) if hi_pos else (mid, hi)
                    else:
                        hi_pos = float(signs[seg + 1, unit]) > 0
                        (lo, hi) = (mid, hi) if hi_pos else (lo, mid)
            tstar = 0.5 * (lo + hi)
            # n_F: pullback of the unit's preact to the ray input.
            x = ((1 - tstar) * e0 + tstar * e1).detach().requires_grad_(True)
            _, pres = forward_from_embed(x, base, pos_ids)
            pres[layer][local].backward()
            n_vec = x.grad.detach().clone()
            # u_F: JVP of a unit bump through the downstream network — finite
            # push of the flipping unit's post-ReLU output along its down-proj
            # row, propagated by the network's own linearity downstream of the
            # flip within the current cell (exact for CPWL blocks; attention
            # of DEEPER layers sees it smoothly — first-order push).
            with torch.no_grad():
                x0 = ((1 - tstar) * e0 + tstar * e1)
                out0, _ = forward_from_embed(x0, base, pos_ids)
                eps = 1e-4
                xp = x0 + eps * (n_vec / n_vec.norm().clamp_min(1e-12))
                outp, _ = forward_from_embed(xp, base, pos_ids)
                xm = x0 - eps * (n_vec / n_vec.norm().clamp_min(1e-12))
                outm, _ = forward_from_embed(xm, base, pos_ids)
                # Second difference across the crease isolates the JUMP in the
                # directional derivative — the u_F direction times the gap.
                u_vec = (outp + outm - 2 * out0) / eps
            un = float(u_vec.norm())
            if not np.isfinite(un) or un < 1e-9:
                continue
            atoms["layer"].append(layer)
            atoms["neuron"].append(int(local))
            atoms["t"].append(tstar)
            atoms["u_norm"].append(un)
            atoms["ray"].append(ray_i)
            n_list.append((n_vec / n_vec.norm().clamp_min(1e-12)).cpu().numpy())
            u_list.append((u_vec / un).cpu().numpy())
        if ray_i % 20 == 0:
            print(json.dumps({"ray": ray_i, "atoms": len(atoms["t"])}), flush=True)

    np.savez(os.path.expanduser("~/tomograph_atoms.npz"),
             layer=np.array(atoms["layer"]), neuron=np.array(atoms["neuron"]),
             t=np.array(atoms["t"]), u_norm=np.array(atoms["u_norm"]),
             ray=np.array(atoms["ray"]),
             n=np.stack(n_list) if n_list else np.zeros((0, 1)),
             u=np.stack(u_list) if u_list else np.zeros((0, 1)))
    print(json.dumps({"total_atoms": len(atoms["t"]), "rays": n_rays}), flush=True)
    print("TOMOGRAPH_DONE", flush=True)


if __name__ == "__main__":
    main()
