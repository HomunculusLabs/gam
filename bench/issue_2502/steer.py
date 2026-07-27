"""Causal steering along a fitted manifold atom (#2502 deliverable 3).

The claim under test is causal, so it needs a control that can refute it. Moving
a residual stream at all changes the output distribution; that is not evidence
the ATOM means anything. Three arms are run on identical rows and identical
token positions:

  atom      move along the fitted atom's own curve, t -> t'
  random    an isotropic direction rescaled to the SAME residual-space norm
  foreign   another atom's curve delta, rescaled to the same norm

If `atom` is not separated from `random` and `foreign`, the steering claim is
refused. The norm match is what makes the arms comparable -- an unmatched
control only shows that bigger perturbations do more.

Reads the artifacts the Rust fit dumped (curve_*.bin are Rust-decoded, not
recomputed here) plus the chart basis saved by the harvest.
"""

import json
import os
import sys

import numpy as np

I2502 = os.path.expanduser("~/i2502")
LAYER = 16
P = 128
N_ROWS = 24          # token positions steered per atom
TOP = 8


def load_f64(path, cols):
    a = np.frombuffer(open(path, "rb").read(), dtype=np.float64)
    return a.reshape(-1, cols)


def main() -> int:
    fit_dir = sys.argv[1]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = json.load(open(f"{fit_dir}/manifest.json"))
    lift = np.load(f"{I2502}/lift.npy")           # (P, hidden)
    c0 = np.load(f"{I2502}/c0.npy")
    test_chart = np.load(f"{I2502}/test_chart.npy")
    train_chart = np.load(f"{I2502}/train_chart.npy")
    tokens = np.load(f"{I2502}/train_tokens.npy")
    seqs = np.load(f"{I2502}/seqs.npy")
    seq_pos = np.load(f"{I2502}/train_seq_pos.npy")
    print(f"lift {lift.shape} train_chart {train_chart.shape} seqs {seqs.shape}", flush=True)

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

    # A steering vector is injected at one (batch, position) cell only.
    inject = {"delta": None, "pos": None}

    def hook(_m, _i, output):
        if inject["delta"] is None:
            return output
        tup = isinstance(output, tuple)
        h = output[0] if tup else output
        h = h.clone()
        d = torch.tensor(inject["delta"], dtype=h.dtype, device=h.device)
        h[:, inject["pos"], :] += d
        return (h,) + output[1:] if tup else h

    handle = block.register_forward_hook(hook)
    rng = np.random.default_rng(0)
    results = []

    for entry in manifest:
        idx, atom_id, kind = entry["idx"], entry["atom"], entry["kind"]
        curve = load_f64(f"{fit_dir}/curve_{idx}.bin", P)          # (161, P)
        toks = load_f64(f"{fit_dir}/tokens_{idx}.bin", 3)          # (row, t, value)
        if len(toks) < N_ROWS:
            continue
        lo, hi = entry["grid_lo"], entry["grid_hi"]
        # foreign arm: the next atom's curve, same shape of move
        other = manifest[(manifest.index(entry) + 1) % len(manifest)]
        curve_o = load_f64(f"{fit_dir}/curve_{other['idx']}.bin", P)

        sel = toks[rng.choice(len(toks), size=N_ROWS, replace=False)]
        per_arm = {"atom": [], "random": [], "foreign": []}
        gained = []

        for row, t, _value in sel:
            row = int(row)
            # move to the far end of the atom's own used range
            g = int(round((t - lo) / max(hi - lo, 1e-12) * 160))
            g = min(max(g, 0), 160)
            g2 = 160 - g
            dz = curve[g2] - curve[g]                    # chart-space move
            d_atom = dz @ lift                            # -> residual space
            n = np.linalg.norm(d_atom)
            if n < 1e-9:
                continue
            d_rand = rng.normal(size=d_atom.shape)
            d_rand *= n / np.linalg.norm(d_rand)
            d_for = (curve_o[g2] - curve_o[g]) @ lift
            nf = np.linalg.norm(d_for)
            if nf < 1e-9:
                continue
            d_for *= n / nf

            # Real left-context: the harvested rows are a SUBSAMPLE, so slicing
            # them would splice unrelated tokens together and the steered logits
            # would describe a context the model never saw. Index back into the
            # original sequence grid instead and steer that row's true position.
            s_i, s_p = seq_pos[row]
            start = max(int(s_p) - 63, 0)
            ids = torch.tensor(
                seqs[int(s_i), start:int(s_p) + 1][None, :],
                dtype=torch.long, device="cuda:0")
            pos = ids.shape[1] - 1

            with torch.inference_mode():
                inject["delta"] = None
                base = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                base = torch.log_softmax(base, -1)
                out = {}
                for arm, d in (("atom", d_atom), ("random", d_rand), ("foreign", d_for)):
                    inject["delta"], inject["pos"] = d, pos
                    lg = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                    lg = torch.log_softmax(lg, -1)
                    kl = float(torch.sum(lg.exp() * (lg - base)))
                    per_arm[arm].append(kl)
                    out[arm] = lg
                inject["delta"] = None
            d_lp = (out["atom"] - base).cpu().numpy()
            for j in np.argsort(-d_lp)[:TOP]:
                gained.append((float(d_lp[j]), tok.decode([int(j)])))

        if not per_arm["atom"]:
            continue
        m = {a: float(np.mean(v)) for a, v in per_arm.items()}
        se = {a: float(np.std(v) / np.sqrt(len(v))) for a, v in per_arm.items()}
        gained.sort(reverse=True)
        seen, top = set(), []
        for _s, w in gained:
            if w not in seen:
                seen.add(w)
                top.append(w)
            if len(top) == TOP:
                break
        rec = {
            "atom": atom_id, "kind": kind, "usage": entry["usage"], "n": len(per_arm["atom"]),
            "kl_atom": m["atom"], "kl_atom_se": se["atom"],
            "kl_random": m["random"], "kl_random_se": se["random"],
            "kl_foreign": m["foreign"], "kl_foreign_se": se["foreign"],
            "ratio_vs_random": m["atom"] / max(m["random"], 1e-12),
            "ratio_vs_foreign": m["atom"] / max(m["foreign"], 1e-12),
            "top_promoted": top,
        }
        results.append(rec)
        print(json.dumps(rec), flush=True)

    handle.remove()
    json.dump(results, open(f"{fit_dir}/steering.json", "w"), indent=1)
    if results:
        rv = float(np.mean([r["ratio_vs_random"] for r in results]))
        rf = float(np.mean([r["ratio_vs_foreign"] for r in results]))
        print(f"STEER DONE atoms={len(results)} mean KL ratio vs random={rv:.2f} "
              f"vs foreign={rf:.2f}", flush=True)
        print("VERDICT:", "atom-specific" if rv > 1.5 and rf > 1.5
              else "NOT separated from controls -- steering claim refused", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
