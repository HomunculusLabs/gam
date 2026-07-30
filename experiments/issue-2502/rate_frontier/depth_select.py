"""Validation-selected depth per arm; report on disjoint test half.
Held-out rows split by row parity: even rows = val (depth selection),
odd rows = test (reported). Depth is the only selected hyperparameter."""
import json
import numpy as np
import torch

dev = "cuda:0"
Z = torch.tensor(np.load("/home/ubuntu/i2502/test_chart.npy"), dtype=torch.float32, device=dev)
val, test = Z[0::2], Z[1::2]

def ev(part, w, form, L0=8):
    U = torch.tensor(w["U"], dtype=torch.float32, device=dev)
    W_enc = torch.tensor(w["W_enc"], dtype=torch.float32, device=dev)
    b_enc = torch.tensor(w["b_enc"], dtype=torch.float32, device=dev)
    b_pre = torch.tensor(w["b_pre"], dtype=torch.float32, device=dev)
    pre = (part - b_pre) @ W_enc + b_enc
    v, idx = torch.topk(pre, L0, dim=1)
    z = torch.zeros_like(pre).scatter_(1, idx, torch.relu(v))
    rec = z @ U
    if form == "field":
        MA = torch.tensor(w["MA"], dtype=torch.float32, device=dev)
        MB = torch.tensor(w["MB"], dtype=torch.float32, device=dev)
        g2 = torch.tensor(w["g2"], dtype=torch.float32, device=dev)
        rec = rec + ((z * z) * g2) @ (U @ MA @ MB.t())
    rec = rec + b_pre
    tv = float(((part - part.mean(0)) ** 2).sum())
    return 1.0 - float(((part - rec) ** 2).sum()) / tv

out = {}
for s in range(3):
    row = {}
    for arm, form in (("flat", "flat"), ("ff", "field")):
        best = None
        for d in (40, 60, 80):
            try:
                w = dict(np.load(f"/home/ubuntu/w_d{d}_{arm}_s{s}.npz"))
            except FileNotFoundError:
                continue
            v = ev(val, w, form)
            if best is None or v > best[1]:
                best = (d, v, ev(test, w, form))
        row[arm] = {"depth": best[0], "val_ev": best[1], "test_ev": best[2]}
    row["test_diff_flat_minus_ff"] = row["flat"]["test_ev"] - row["ff"]["test_ev"]
    out[f"s{s}"] = row
print(json.dumps(out, indent=1))
