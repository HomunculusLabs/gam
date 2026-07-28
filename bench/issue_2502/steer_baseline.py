"""Causal steering along a TopK SAE atom — the baseline arm of the steering test.

Steering has only ever been run on manifold atoms in this campaign, so its
"atom-specific" verdict has never been compared against the dictionary it is
supposed to be better than. This runs the identical three-arm protocol on the
parameter-matched TopK SAE, so the two are measured the same way:

  atom      move along the SAE atom's own decoder direction w_k
  random    isotropic direction rescaled to the SAME residual-space norm
  foreign   a different SAE atom's direction at the same norm

A linear atom's "curve" is the ray `a * w_k`, so the steering move that
corresponds to traversing a manifold atom's coordinate range is a change in
amplitude along `w_k`. The move size is matched to the manifold protocol by
using the same span of the atom's OWN fitted amplitude distribution — the
10th to 90th percentile of the amplitudes it actually takes on real rows —
rather than an arbitrary multiple, so neither arm is handed a larger
perturbation than the other.

Atoms are chosen the way the manifold dump chooses them: the most-used ones, so
both arms are reporting on atoms that carry real traffic.
"""

import json
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")
LAYER = 16
P = 128
N_ROWS = 24
TOP = 8
N_ATOMS = 12


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    blob = np.load(f"{V2}/baseline_k10525_s0.npz")
    W = blob["W_dec"].astype(np.float64)
    bp = blob["b_pre"].astype(np.float64)
    k_act = int(blob["k_act"]) if "k_act" in blob else 8

    lift = np.load(f"{V2}/lift.npy")
    c0 = np.load(f"{V2}/c0.npy")
    train = np.load(f"{V2}/train_chart.npy")
    tokens = np.load(f"{V2}/train_tokens.npy")
    seqs = np.load(f"{V2}/seqs.npy")
    seq_pos = np.load(f"{V2}/train_seq_pos.npy")
    print(f"SAE decoder {W.shape}; train chart {train.shape}", flush=True)

    # Route the training rows once to find which atoms carry traffic and what
    # amplitudes they actually take, so the move size is the atom's own scale.
    sub = np.random.default_rng(0).choice(len(train), size=40000, replace=False)
    X = train[sub]
    norms = (W * W).sum(1)
    R = X - bp
    taken = np.zeros((len(X), len(W)), dtype=bool)
    picks = np.zeros((len(X), k_act), dtype=np.int64)
    amps = np.zeros((len(X), k_act))
    for step in range(k_act):
        gain = 2.0 * (R @ W.T) - norms
        gain[taken] = -np.inf
        p = gain.argmax(1)
        picks[:, step] = p
        taken[np.arange(len(X)), p] = True
        coef = np.maximum((R * W[p]).sum(1) / np.maximum(norms[p], 1e-30), 0.0)
        amps[:, step] = coef
        R = R - coef[:, None] * W[p]

    usage = np.bincount(picks.reshape(-1), minlength=len(W))
    chosen = np.argsort(-usage)[:N_ATOMS]
    print(f"steering the {N_ATOMS} most-used SAE atoms (usage {usage[chosen[0]]}..{usage[chosen[-1]]})", flush=True)

    name = "Qwen/Qwen3.5-4B-Base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
    model.eval()
    layers = None
    for _n, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (layers is None or len(mod) > len(layers)):
            layers = mod
    block = layers[LAYER]

    inject = {"delta": None, "pos": None}

    def hook(_m, _i, output):
        if inject["delta"] is None:
            return output
        tup = isinstance(output, tuple)
        h = output[0] if tup else output
        h = h.clone()
        h[:, inject["pos"], :] += torch.tensor(
            inject["delta"], dtype=h.dtype, device=h.device)
        return (h,) + output[1:] if tup else h

    handle = block.register_forward_hook(hook)
    rng = np.random.default_rng(0)
    results = []

    for rank, atom in enumerate(chosen):
        hit = np.argwhere(picks == atom)
        if len(hit) < N_ROWS:
            continue
        take = hit[rng.choice(len(hit), size=N_ROWS, replace=False)]
        a = amps[take[:, 0], take[:, 1]]
        lo, hi = np.percentile(a, 10), np.percentile(a, 90)
        foreign = chosen[(rank + 1) % len(chosen)]

        per_arm = {"atom": [], "random": [], "foreign": []}
        gained = []
        for (row_i, _slot) in take:
            row = int(sub[row_i])
            d_atom = (hi - lo) * W[atom] @ lift
            n = np.linalg.norm(d_atom)
            if n < 1e-9:
                continue
            d_rand = rng.normal(size=d_atom.shape)
            d_rand *= n / np.linalg.norm(d_rand)
            d_for = (hi - lo) * W[foreign] @ lift
            nf = np.linalg.norm(d_for)
            if nf < 1e-9:
                continue
            d_for *= n / nf

            s_i, s_p = seq_pos[row]
            start = max(int(s_p) - 63, 0)
            ids = torch.tensor(
                seqs[int(s_i), start:int(s_p) + 1][None, :],
                dtype=torch.long, device="cuda:0")
            pos = ids.shape[1] - 1
            with torch.inference_mode():
                inject["delta"] = None
                base = torch.log_softmax(
                    model(input_ids=ids, use_cache=False).logits[0, -1].float(), -1)
                out = {}
                for arm, delta in (("atom", d_atom), ("random", d_rand), ("foreign", d_for)):
                    inject["delta"], inject["pos"] = delta, pos
                    lg = torch.log_softmax(
                        model(input_ids=ids, use_cache=False).logits[0, -1].float(), -1)
                    per_arm[arm].append(float(torch.sum(lg.exp() * (lg - base))))
                    out[arm] = lg
                inject["delta"] = None
            dlp = (out["atom"] - base).cpu().numpy()
            for j in np.argsort(-dlp)[:TOP]:
                gained.append((float(dlp[j]), tok.decode([int(j)])))

        if not per_arm["atom"]:
            continue
        m = {k: float(np.mean(v)) for k, v in per_arm.items()}
        gained.sort(reverse=True)
        seen, top = set(), []
        for _s, w in gained:
            if w not in seen:
                seen.add(w)
                top.append(w)
            if len(top) == TOP:
                break
        rec = {
            "atom": int(atom), "usage": int(usage[atom]), "n": len(per_arm["atom"]),
            "kl_atom": m["atom"], "kl_random": m["random"], "kl_foreign": m["foreign"],
            "ratio_vs_random": m["atom"] / max(m["random"], 1e-12),
            "ratio_vs_foreign": m["atom"] / max(m["foreign"], 1e-12),
            "top_promoted": top,
        }
        results.append(rec)
        print(json.dumps(rec), flush=True)

    handle.remove()
    json.dump(results, open(f"{V2}/steering_baseline.json", "w"), indent=1)
    if results:
        rv = float(np.median([r["ratio_vs_random"] for r in results]))
        rf = float(np.median([r["ratio_vs_foreign"] for r in results]))
        both = sum(1 for r in results
                   if r["ratio_vs_random"] > 1.5 and r["ratio_vs_foreign"] > 1.5)
        print(f"BASELINE STEER DONE atoms={len(results)} median vs random={rv:.2f} "
              f"vs foreign={rf:.2f}; cleared BOTH: {both}/{len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
