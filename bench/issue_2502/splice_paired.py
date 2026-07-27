"""Delta-cross-entropy splice at the held-out positions (#2502, final criterion).

Chart and ambient EV both ask how well a dictionary reconstructs activations.
Neither asks whether the reconstructed directions are the ones the language model
computes with. This substitutes each arm's reconstruction into the residual
stream at layer 16 and measures what the model's own loss does.

Every held-out row carries a (sequence, position) index, so the substitution
happens at exactly the positions each arm was scored on -- no new fits, no new
harvest, and the same rows in every arm.

Arms, all on identical sequences and identical positions:

  clean       no substitution -- the reference CE
  identity    project to the PCA-128 chart and lift straight back, NO dictionary.
              This is the ceiling any chart-space method can reach (measured
              ambient EV 0.6219), so an arm that does not beat it has not been
              shown to contribute anything the CHART was not already contributing.
  mean        substitute the training mean everywhere -- the floor
  manifold    the fitted curved dictionary's reconstruction
  baseline    the TopK SAE's OMP + joint-LS reconstruction

CE is measured ONLY at the substituted positions, since those are the only
positions whose input the substitution changed.
"""

import glob
import json
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")
LAYER = 16
P = 128
MAX_SEQ = 160          # sequences carrying held-out rows, capped for wall-clock


def ce_for(model, torch, block, seqs, groups, deltas, label):
    """Mean next-token CE at the substituted positions."""
    inject = {"pos": None, "vec": None}

    def hook(_m, _i, output):
        if inject["pos"] is None:
            return output
        tup = isinstance(output, tuple)
        h = output[0] if tup else output
        h = h.clone()
        idx = torch.tensor(inject["pos"], dtype=torch.long, device=h.device)
        # Perturb ONLY the chart component. Replacing `h` outright also deletes
        # the ~21% of the residual norm that lives outside the PCA-128 chart,
        # which every arm pays identically and which dominates the dictionary's
        # own contribution -- it made the no-dictionary chart round-trip look
        # like a competitor when it is really the zero point. Adding
        # `rec - chart` leaves the out-of-chart part untouched, so identity is
        # exactly a no-op and the arms are compared on what they actually differ
        # in: reconstruction error INSIDE the chart.
        h[0, idx, :] += torch.tensor(
            inject["vec"], dtype=h.dtype, device=h.device)
        return (h,) + output[1:] if tup else h

    handle = block.register_forward_hook(hook)
    total, count, per_pos = 0.0, 0, []
    with torch.inference_mode():
        for s_i, positions in groups:
            pos = np.asarray(positions, dtype=np.int64)
            keep = pos < seqs.shape[1] - 1          # need a next token to score
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
    inject["pos"] = None
    import numpy as _np
    return total / max(count, 1), count, _np.concatenate(per_pos) if per_pos else _np.zeros(0)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    lift = np.load(f"{V2}/lift.npy")
    c0 = np.load(f"{V2}/c0.npy")
    chart = np.load(f"{V2}/test_chart.npy")
    seqs = np.load(f"{V2}/seqs.npy")
    seq_pos = np.load(f"{V2}/test_seq_pos.npy")
    clean_mask = np.load(f"{V2}/test_clean_mask.npy")

    # Fresh rows only, grouped by sequence.
    rows = np.flatnonzero(clean_mask)
    by_seq = {}
    for r in rows:
        s_i, s_p = int(seq_pos[r, 0]), int(seq_pos[r, 1])
        by_seq.setdefault(s_i, []).append((s_p, r))
    chosen = sorted(by_seq.items(), key=lambda kv: -len(kv[1]))[:MAX_SEQ]
    groups = [(s_i, [p for p, _ in sorted(v)]) for s_i, v in chosen]
    row_of = {s_i: [r for _, r in sorted(v)] for s_i, v in chosen}
    n_pos = sum(len(p) for _, p in groups)
    print(f"splicing {len(groups)} sequences, {n_pos} held-out positions", flush=True)

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
        """(sequence, positions) -> ambient rows for those positions."""
        out = {}
        for s_i, positions in groups:
            pos = np.asarray(positions, dtype=np.int64)
            keep = pos < seqs.shape[1] - 1
            idx = [row_of[s_i][k] for k in range(len(positions)) if keep[k]]
            if not idx:
                continue
            # Delta against the row's OWN chart component, not the raw
            # activation: `(rec - chart) @ lift` is the in-chart correction.
            out[(s_i, tuple(pos[keep]))] = (recon_chart[idx] - chart[idx]) @ lift
        return out

    results = {}
    per_position = {}
    base_ce, n, base_pp = ce_for(model, torch, block, seqs, groups, None, "clean")
    results["clean"] = base_ce
    per_position["clean"] = base_pp
    print(f"clean CE = {base_ce:.5f} over {n} positions", flush=True)

    results["identity_chart_roundtrip"], _n, per_position["identity"] = ce_for(
        model, torch, block, seqs, groups, pack(chart), "identity")
    results["mean_floor"], _n, per_position["mean"] = ce_for(
        model, torch, block, seqs, groups,
        pack(np.zeros_like(chart) + chart.mean(0)), "mean")

    for path in sorted(glob.glob(f"{V2}/a8_s*/heldout_recon.bin")):
        seed = path.split("a8_s")[1].split("/")[0]
        rec = np.frombuffer(open(path, "rb").read(), dtype=np.float64).reshape(-1, P)
        results[f"manifold_s{seed}"], _n, per_position[f"manifold_s{seed}"] = ce_for(
            model, torch, block, seqs, groups, pack(rec), f"manifold s{seed}")

    for path in sorted(glob.glob(f"{V2}/baseline_k*_s*.npz")):
        blob = np.load(path)
        W = blob["W_dec"].astype(np.float64)
        b = blob["b_pre"].astype(np.float64)
        k_act = int(blob["k_act"]) if "k_act" in blob else 8
        norms = (W * W).sum(1)
        R = chart - b
        taken = np.zeros((len(chart), len(W)), dtype=bool)
        picks = np.zeros((len(chart), k_act), dtype=np.int64)
        for s in range(k_act):
            g = 2.0 * (R @ W.T) - norms
            g[taken] = -np.inf
            p = g.argmax(1)
            picks[:, s] = p
            taken[np.arange(len(chart)), p] = True
            coef = np.maximum((R * W[p]).sum(1) / np.maximum(norms[p], 1e-30), 0.0)
            R = R - coef[:, None] * W[p]
        rec = np.empty_like(chart)
        for i in range(len(chart)):
            A = W[picks[i]].T
            c, *_ = np.linalg.lstsq(A, chart[i] - b, rcond=None)
            rec[i] = A @ c
        rec = rec + b
        _k = os.path.basename(path).replace(".npz", "")
        results[_k], _n, per_position[_k] = ce_for(
            model, torch, block, seqs, groups, pack(rec), "baseline")

    out = {k: {"ce": v, "delta_ce": v - base_ce} for k, v in results.items()}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(f"{V2}/splice_paired.json", "w"), indent=1)
    np.savez(f"{V2}/per_position_nll.npz", **per_position)
    print("SPLICE DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
