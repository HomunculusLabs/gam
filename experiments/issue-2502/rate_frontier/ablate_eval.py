import json, sys
import numpy as np
import torch

dev = "cuda:0"
Xte = torch.tensor(np.load("/home/ubuntu/i2502/test_chart.npy"), dtype=torch.float32, device=dev)
tv = float(((Xte - Xte.mean(0)) ** 2).sum())

def ev(w, form, L0=8, zero_gamma=False):
    U = torch.tensor(w["U"], dtype=torch.float32, device=dev)
    W_enc = torch.tensor(w["W_enc"], dtype=torch.float32, device=dev)
    b_enc = torch.tensor(w["b_enc"], dtype=torch.float32, device=dev)
    b_pre = torch.tensor(w["b_pre"], dtype=torch.float32, device=dev)
    pre = (Xte - b_pre) @ W_enc + b_enc
    val, idx = torch.topk(pre, L0, dim=1)
    z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(val))
    rec = z @ U
    if form == "field" and not zero_gamma:
        MA = torch.tensor(w["MA"], dtype=torch.float32, device=dev)
        MB = torch.tensor(w["MB"], dtype=torch.float32, device=dev)
        g2 = torch.tensor(w["g2"], dtype=torch.float32, device=dev)
        rec = rec + ((z * z) * g2) @ (U @ MA @ MB.t())
    rec = rec + b_pre
    return 1.0 - float(((Xte - rec) ** 2).sum()) / tv

out = {}
for s in range(3):
    wf = dict(np.load(f"/home/ubuntu/w_flat_s{s}.npz"))
    wg = dict(np.load(f"/home/ubuntu/w_ff_s{s}.npz"))
    out[f"s{s}"] = {"flat": ev(wf, "flat"), "field": ev(wg, "field"),
                    "field_gamma0": ev(wg, "field", zero_gamma=True)}
print(json.dumps(out, indent=1))
