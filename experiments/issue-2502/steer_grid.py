"""Steering selectivity for fitted manifold atoms vs the steelman's directions.

Steering vector for a manifold atom = difference of two points on its DUMPED
decoded curve (parity-exact: the Rust decoder produced those points), lifted
chart -> ambient. Injected at the positions ROUTED to that atom.

Controls, all at identical injection norm and identical positions:
  random     an equal-norm random ambient direction (per atom, fixed seed)
  steelman   the K=21,050 steelman's best-aligned decoder direction

Metric per atom: mean |delta CE| at routed positions. Selectivity =
atom_effect / random_effect. A dictionary whose directions are causally
special has selectivity >> 1; random has selectivity 1 by construction.
"""
import json
import os
import sys

import numpy as np

V2 = os.path.expanduser("~/i2502v2")
LAYER = 16
MAX_ATOMS = 8
MAX_POS_PER_ATOM = 160
ALPHA = 6.0  # injection norm in chart units, ~1 sd of chart rows


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    fit_dir = sys.argv[1] if len(sys.argv) > 1 else f"{V2}/s_auto8000"
    lift = np.load(f"{V2}/lift.npy")
    seqs = np.load(f"{V2}/seqs.npy")
    seq_pos = np.load(f"{V2}/train_seq_pos.npy")
    man = json.load(open(os.path.join(fit_dir, "manifest.json")))
    census = np.fromfile(f"{fit_dir}/census.bin").reshape(-1, 4)
    usage = census[:, 0]
    picked = sorted(man, key=lambda a: -usage[a["atom"]])[:MAX_ATOMS]

    steel = np.load(f"{V2}/steel_k21050_s0.npz")
    w_dec = steel["W_dec"].astype(np.float64)
    w_unit = w_dec / np.maximum(np.linalg.norm(w_dec, axis=1, keepdims=True), 1e-30)

    name = "Qwen/Qwen3.5-4B-Base"
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
    model.eval()
    layers = None
    for _n, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (layers is None or len(mod) > len(layers)):
            layers = mod
    block = layers[LAYER]

    inject = {"pos": None, "vec": None}

    def hook(_m, _i, output):
        if inject["pos"] is None:
            return output
        tup = isinstance(output, tuple)
        h = output[0] if tup else output
        h = h.clone()
        idx = torch.tensor(inject["pos"], dtype=torch.long, device=h.device)
        h[0, idx, :] += torch.tensor(inject["vec"], dtype=h.dtype, device=h.device)
        return (h,) + output[1:] if tup else h

    handle = block.register_forward_hook(hook)

    def ce_at(groups, vec):
        total, count = 0.0, 0
        with torch.inference_mode():
            for s_i, positions in groups:
                pos = np.asarray(positions, dtype=np.int64)
                pos = pos[pos < seqs.shape[1] - 1]
                if len(pos) == 0:
                    continue
                ids = torch.tensor(seqs[s_i][None, :], dtype=torch.long, device="cuda:0")
                inject["pos"] = None if vec is None else pos
                inject["vec"] = None if vec is None else np.repeat(
                    vec[None, :], len(pos), 0)
                logits = model(input_ids=ids, use_cache=False).logits.float()
                lp = torch.log_softmax(logits[0, :-1], -1)
                tgt = ids[0, 1:]
                nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                total += float(nll[pos].sum())
                count += len(pos)
        return total / max(count, 1), count

    rng = np.random.default_rng(0)
    report = []
    for atom in picked:
        toks = np.fromfile(f"{fit_dir}/tokens_{atom['idx']}.bin").reshape(-1, 3)
        rows = toks[:, 0].astype(int)[:MAX_POS_PER_ATOM]
        by_seq = {}
        for r in rows:
            s_i, s_p = int(seq_pos[r, 0]), int(seq_pos[r, 1])
            by_seq.setdefault(s_i, []).append(s_p)
        groups = sorted(by_seq.items(), key=lambda kv: -len(kv[1]))[:24]

        curve = np.fromfile(f"{fit_dir}/curve_{atom['idx']}.bin").reshape(-1, 128)
        # steering chord: between the coordinate quartiles of the curve
        n = curve.shape[0]
        chord = curve[(3 * n) // 4] - curve[n // 4]
        chord_a = chord @ lift
        norm = np.linalg.norm(chord_a)
        if norm < 1e-9:
            continue
        vec_atom = chord_a / norm * ALPHA
        rand = rng.standard_normal(vec_atom.shape[0])
        vec_rand = rand / np.linalg.norm(rand) * ALPHA
        # steelman: its unit decoder direction best aligned with the chord.
        # Alignment in CHART space (both live there); the chosen direction is
        # lifted to ambient and normalized to the SAME injection norm.
        chord_unit = chord / max(np.linalg.norm(chord), 1e-30)
        sims = w_unit @ chord_unit
        steel_dir = w_unit[int(np.argmax(np.abs(sims)))] @ lift
        vec_steel = steel_dir / max(np.linalg.norm(steel_dir), 1e-30) * ALPHA

        base, n_pos = ce_at(groups, None)
        ce_atom, _ = ce_at(groups, vec_atom)
        ce_rand, _ = ce_at(groups, vec_rand)
        ce_steel, _ = ce_at(groups, vec_steel)
        entry = {
            "atom": atom["atom"], "kind": atom["kind"], "positions": n_pos,
            "d_atom": ce_atom - base, "d_rand": ce_rand - base,
            "d_steel": ce_steel - base,
        }
        report.append(entry)
        print(f"atom {atom['atom']} ({atom['kind']}, n={n_pos}): "
              f"dCE atom={entry['d_atom']:+.4f} rand={entry['d_rand']:+.4f} "
              f"steel={entry['d_steel']:+.4f}", flush=True)

    handle.remove()
    a = np.array([abs(e["d_atom"]) for e in report])
    r = np.array([abs(e["d_rand"]) for e in report])
    st = np.array([abs(e["d_steel"]) for e in report])
    print(f"\nselectivity |dCE| ratios over {len(report)} atoms:")
    print(f"  manifold/random   = {a.mean()/max(r.mean(),1e-12):.2f}")
    print(f"  steelman/random   = {st.mean()/max(r.mean(),1e-12):.2f}")
    print(f"  manifold/steelman = {a.mean()/max(st.mean(),1e-12):.2f}")
    json.dump(report, open(f"{V2}/steer_grid_report.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
