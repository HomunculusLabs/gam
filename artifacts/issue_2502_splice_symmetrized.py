"""Symmetrized (+/-delta) splice with an uncontaminated single-position mode.

Two repairs to splice_paired.py, both aimed at the estimator rather than the
estimand:

1. SYMMETRIZATION. Writing ДCE_i = g_i^T d_i + 0.5 d_i^T H_i d_i + O(|d|^3),
   the mean damage is the quadratic term but the per-row VARIANCE is dominated
   by the signed linear term, which carries no information about
   reconstruction quality. Splicing +d and -d and averaging cancels the linear
   term exactly to third order; the odd half (CE+ - CE-)/2 is reported too,
   as the noise floor that was previously inside every number.

2. SINGLE-POSITION MODE. The multi-position design perturbs all of a
   sequence's held-out positions in one forward, so with causal attention the
   CE at position p mixes the direct effect of d_p with indirect effects of
   every earlier perturbed position -- and sequences were CHOSEN in descending
   held-out count, maximizing that contamination. Mode "single" perturbs
   exactly one position per forward (the sequence's median held-out position),
   over more sequences.

Arms: identity chart round-trip, mean floor, manifold heldout_recon, flat
baseline npzs, and the NEW curved a-(U+tV) npzs (closed-form optimal t greedy
+ joint LS, ported to numpy).

Usage: python3 splice_sym.py [multi|single]
"""

import glob
import json
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")
LAYER = 16
P = 128


def ce_for(model, torch, block, seqs, groups, deltas, sign=1.0):
    inject = {"pos": None, "vec": None}

    def hook(_m, _i, output):
        if inject["pos"] is None:
            return output
        tup = isinstance(output, tuple)
        h = output[0] if tup else output
        h = h.clone()
        idx = torch.tensor(inject["pos"], dtype=torch.long, device=h.device)
        h[0, idx, :] += torch.tensor(
            sign * inject["vec"], dtype=h.dtype, device=h.device)
        return (h,) + output[1:] if tup else h

    handle = block.register_forward_hook(hook)
    total, count, per_pos = 0.0, 0, []
    with torch.inference_mode():
        for s_i, positions in groups:
            pos = np.asarray(positions, dtype=np.int64)
            keep = pos < seqs.shape[1] - 1
            pos = pos[keep]
            if len(pos) == 0:
                continue
            ids = torch.tensor(seqs[s_i][None, :], dtype=torch.long, device="cuda:0")
            if deltas is None:
                inject["pos"], inject["vec"] = None, None
            else:
                inject["pos"] = pos
                inject["vec"] = deltas[(s_i, tuple(pos))]
            logits = model(input_ids=ids, use_cache=False).logits.float()
            lp = torch.log_softmax(logits[0, :-1], -1)
            tgt = ids[0, 1:]
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            per_pos.append(nll[pos].detach().float().cpu().numpy())
            total += float(nll[pos].sum())
            count += len(pos)
    handle.remove()
    return total / max(count, 1), count, np.concatenate(per_pos) if per_pos else np.zeros(0)


def curved_recon(path, chart):
    """Closed-form optimal-t greedy + joint LS for a curved npz, in numpy."""
    blob = np.load(path)
    U, V, b = blob["U"].astype(np.float64), blob["V"].astype(np.float64), \
        blob["b_pre"].astype(np.float64)
    k_act = int(blob["k_act"]) if "k_act" in blob else 8
    c = (U * V).sum(1)
    v = (V * V).sum(1)
    xc = chart - b
    R = xc.copy()
    N = len(chart)
    K = len(U)
    taken = np.zeros((N, K), dtype=bool)
    picks = np.zeros((N, k_act), dtype=np.int64)
    tpick = np.zeros((N, k_act))
    for s in range(k_act):
        alpha = R @ U.T
        beta = R @ V.T
        tstar = np.clip((beta - alpha * c) / (alpha * v - beta * c + 1e-30), -1, 1)
        best_gain = np.full((N, K), -np.inf)
        best_t = np.zeros((N, K))
        for tc in (tstar, np.full_like(tstar, -1.0), np.full_like(tstar, 1.0)):
            num = alpha + beta * tc
            den = np.maximum(1.0 + 2.0 * c * tc + v * tc ** 2, 1e-12)
            gain = np.where(num > 0, num ** 2 / den, 0.0)
            better = gain > best_gain
            best_gain = np.where(better, gain, best_gain)
            best_t = np.where(better, tc, best_t)
        best_gain[taken] = -np.inf
        p = best_gain.argmax(1)
        picks[:, s] = p
        tpick[:, s] = best_t[np.arange(N), p]
        taken[np.arange(N), p] = True
        d = U[p] + tpick[:, s][:, None] * V[p]
        dn = np.maximum((d * d).sum(1), 1e-30)
        a = np.maximum((R * d).sum(1) / dn, 0.0)
        R = R - a[:, None] * d
    rec = np.empty_like(chart)
    for i in range(N):
        A = (U[picks[i]] + tpick[i][:, None] * V[picks[i]]).T
        coef, *_ = np.linalg.lstsq(A, xc[i], rcond=None)
        rec[i] = A @ coef
    return rec + b


def flat_recon(path, chart):
    blob = np.load(path)
    W = blob["W_dec"].astype(np.float64)
    b = blob["b_pre"].astype(np.float64)
    k_act = int(blob["k_act"]) if "k_act" in blob else 8
    norms = (W * W).sum(1)
    R = chart - b
    N = len(chart)
    taken = np.zeros((N, len(W)), dtype=bool)
    picks = np.zeros((N, k_act), dtype=np.int64)
    for s in range(k_act):
        g = 2.0 * (R @ W.T) - norms
        g[taken] = -np.inf
        p = g.argmax(1)
        picks[:, s] = p
        taken[np.arange(N), p] = True
        coef = np.maximum((R * W[p]).sum(1) / np.maximum(norms[p], 1e-30), 0.0)
        R = R - coef[:, None] * W[p]
    rec = np.empty_like(chart)
    for i in range(N):
        A = W[picks[i]].T
        coef, *_ = np.linalg.lstsq(A, chart[i] - b, rcond=None)
        rec[i] = A @ coef
    return rec + b


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    mode = sys.argv[1] if len(sys.argv) > 1 else "multi"
    max_seq = 160 if mode == "multi" else 600

    lift = np.load(f"{V2}/lift.npy")
    chart = np.load(f"{V2}/test_chart.npy")
    seqs = np.load(f"{V2}/seqs.npy")
    seq_pos = np.load(f"{V2}/test_seq_pos.npy")
    clean_mask = np.load(f"{V2}/test_clean_mask.npy")

    rows = np.flatnonzero(clean_mask)
    by_seq = {}
    for r in rows:
        s_i, s_p = int(seq_pos[r, 0]), int(seq_pos[r, 1])
        by_seq.setdefault(s_i, []).append((s_p, r))
    chosen = sorted(by_seq.items(), key=lambda kv: -len(kv[1]))[:max_seq]
    if mode == "single":
        # One position per sequence: the median held-out position, so the
        # forward carries exactly one perturbation and the scored CE is the
        # direct effect only.
        chosen = [(s_i, [sorted(v)[len(v) // 2]]) for s_i, v in chosen]
        chosen = [(s_i, v) for s_i, v in chosen]
        groups = [(s_i, [p for p, _ in v]) for s_i, v in chosen]
        row_of = {s_i: [r for _, r in v] for s_i, v in chosen}
    else:
        groups = [(s_i, [p for p, _ in sorted(v)]) for s_i, v in chosen]
        row_of = {s_i: [r for _, r in sorted(v)] for s_i, v in chosen}
    n_pos = sum(len(p) for _, p in groups)
    print(f"mode={mode}: {len(groups)} sequences, {n_pos} positions", flush=True)

    name = "Qwen/Qwen3.5-4B-Base"
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
    model.eval()
    layers = None
    for _n, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (layers is None or len(mod) > len(layers)):
            layers = mod
    block = layers[LAYER]

    def pack(recon_chart):
        out = {}
        for s_i, positions in groups:
            pos = np.asarray(positions, dtype=np.int64)
            keep = pos < seqs.shape[1] - 1
            idx = [row_of[s_i][k] for k in range(len(positions)) if keep[k]]
            if not idx:
                continue
            out[(s_i, tuple(pos[keep]))] = (recon_chart[idx] - chart[idx]) @ lift
        return out

    arms = {"identity_chart_roundtrip": chart,
            "mean_floor": np.zeros_like(chart) + chart.mean(0)}
    for path in sorted(glob.glob(f"{V2}/a8_s*/heldout_recon.bin")):
        seed = path.split("a8_s")[1].split("/")[0]
        arms[f"manifold_s{seed}"] = np.frombuffer(
            open(path, "rb").read(), dtype=np.float64).reshape(-1, P)
    for path in sorted(glob.glob(f"{V2}/baseline_k*_s*.npz")) + \
            sorted(glob.glob(f"{V2}/steel_k*_s*.npz")):
        arms[os.path.basename(path).replace(".npz", "")] = flat_recon(path, chart)
    for path in sorted(glob.glob(f"{V2}/curved_k*_s*_l8.npz")):
        arms[os.path.basename(path).replace(".npz", "")] = curved_recon(path, chart)

    base_ce, n, base_pp = ce_for(model, torch, block, seqs, groups, None)
    print(f"clean CE = {base_ce:.5f} over {n} positions", flush=True)
    results = {"clean": {"ce": base_ce}}
    per_position = {"clean": base_pp}
    for label, rec in arms.items():
        deltas = pack(rec)
        ce_p, _, pp_p = ce_for(model, torch, block, seqs, groups, deltas, +1.0)
        ce_m, _, pp_m = ce_for(model, torch, block, seqs, groups, deltas, -1.0)
        even = 0.5 * (ce_p + ce_m) - base_ce
        odd = 0.5 * (ce_p - ce_m)
        pp_even = 0.5 * (pp_p + pp_m) - base_pp
        results[label] = {
            "delta_ce_plus": ce_p - base_ce, "delta_ce_minus": ce_m - base_ce,
            "delta_ce_sym": even, "odd_half": odd,
            "sym_se": float(pp_even.std() / np.sqrt(max(len(pp_even), 1))),
        }
        per_position[label] = pp_even
        print(json.dumps({label: results[label]}), flush=True)

    json.dump(results, open(f"{V2}/splice_sym_{mode}.json", "w"), indent=1)
    np.savez(f"{V2}/per_position_sym_{mode}.npz", **per_position)
    print("SPLICE_SYM_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
