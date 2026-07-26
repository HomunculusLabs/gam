"""#2502 GPU stage B: patch + measure.

  1. SPLICE: replace the L16 residual stream of the captured validation batch
     with each arm's reconstruction; report ΔCE + loss-recovered.
  2. STEER: patch the last-token residual of each calendar base context with
     the pipeline's on-manifold deltas (and the matched-norm torch-SAE control);
     measure full-softmax target-token probability + target-excluded collateral
     KL. Records share the E1 schema (e1_plots.py renders them).
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B-Base"
LAYER = 16
HOME = os.path.expanduser("~")
FLAG = f"{HOME}/i2502/flagship"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
model.eval()
best = None
for _n, mod in model.named_modules():
    if isinstance(mod, torch.nn.ModuleList) and (best is None or len(mod) > len(best)):
        best = mod
layer = best[LAYER]

A = np.load(f"{HOME}/i2502/stage_a.npz")
meta = json.load(open(f"{HOME}/i2502/stage_a_meta.json"))

# ---------- 1. splice ----------
S = np.load(f"{FLAG}/splice_recons.npz")
ids = torch.tensor(A["val_ids"].astype(np.int64), device="cuda:0")
B, T = ids.shape
results = {"base_ce": meta["val_base_ce"]}
patch_buf = {}


def patch_hook(_m, _i, output):
    h = output[0] if isinstance(output, tuple) else output
    e = patch_buf["r"].to(device=h.device, dtype=h.dtype)
    return (e,) + output[1:] if isinstance(output, tuple) else e


for arm in ("chart", "manifold", "torch_topk", "pca8", "mean_ablate"):
    if arm not in S:
        continue
    R = torch.from_numpy(S[arm].astype(np.float32)).reshape(B, T, -1)
    tot, cnt = 0.0, 0
    handle = layer.register_forward_hook(patch_hook)
    with torch.inference_mode():
        for i in range(0, B, 2):
            patch_buf["r"] = R[i:i + 2]
            out = model(input_ids=ids[i:i + 2], use_cache=False)
            tot += F.cross_entropy(
                out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]),
                ids[i:i + 2, 1:].reshape(-1), reduction="sum").item()
            cnt += (ids.shape[1] - 1) * min(2, B - i)
            del out
    handle.remove()
    results[arm + "_ce"] = tot / cnt
    print(f"[spliceB] {arm}: CE={results[arm + '_ce']:.4f}", flush=True)
floor, base = results.get("mean_ablate_ce"), results["base_ce"]
for arm in ("chart", "manifold", "torch_topk", "pca8"):
    if arm + "_ce" in results and floor and floor > base:
        results["loss_recovered_" + arm] = (floor - results[arm + "_ce"]) / (floor - base)
json.dump(results, open(f"{FLAG}/splice_results.json", "w"), indent=2)
print("[spliceB] DONE", json.dumps(results), flush=True)

# ---------- 2. steering ----------
D = np.load(f"{FLAG}/steer_deltas.npz")
SM = json.load(open(f"{FLAG}/steer_meta.json"))
last_pos = {}


def last_hook_factory(deltas_t):
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        e = h.clone()
        e[:, last_pos["p"], :] += deltas_t.to(device=h.device, dtype=h.dtype)
        return (e,) + output[1:] if isinstance(output, tuple) else e
    return hook


def log_softmax_np(v):
    v = v - v.max()
    return v - np.log(np.exp(v).sum())


records = []
for cyc, n in (("week", 7), ("month", 12)):
    sm = SM.get(cyc)
    if not sm or sm.get("atom") is None or f"{cyc}_deltas" not in D:
        continue
    deltas = D[f"{cyc}_deltas"]          # (nb, ns, nd, 2560)
    flat_dir = D[f"{cyc}_flat_dir"]
    prompts = meta[f"{cyc}_base_prompts"]
    labs = A[f"{cyc}_base_lab"]
    base_logits = A[f"{cyc}_base_logits"].astype(np.float32)
    cand = meta[f"{cyc}_cand_ids"]
    shifts, doses = sm["shifts"], sm["doses"]
    nb, ns, nd = deltas.shape[:3]
    for i in range(nb):
        enc = tok(prompts[i], return_tensors="pt", add_special_tokens=False).to("cuda:0")
        L = enc["input_ids"].shape[1]
        last_pos["p"] = L - 1
        base_lp = log_softmax_np(base_logits[i].astype(np.float64))
        variants, keys = [], []
        for js in range(ns):
            for jd in range(nd):
                d = deltas[i, js, jd]
                variants.append(d)
                keys.append(("manifold", js, jd))
                variants.append(np.linalg.norm(d) * flat_dir)
                keys.append(("flat", js, jd))
        V = torch.from_numpy(np.stack(variants).astype(np.float32))
        outs = []
        with torch.inference_mode():
            for c0_ in range(0, len(V), 20):
                chunk = V[c0_:c0_ + 20]
                handle = layer.register_forward_hook(last_hook_factory(chunk))
                out = model(input_ids=enc["input_ids"].expand(len(chunk), -1),
                            use_cache=False)
                outs.append(out.logits[:, L - 1, :].float().cpu().numpy())
                handle.remove()
                del out
        logits_all = np.concatenate(outs)
        for (arm, js, jd), lg in zip(keys, logits_all):
            lp = log_softmax_np(lg.astype(np.float64))
            b = int(labs[i])
            tgt = (b + shifts[js] + 1) % n
            tid = cand[tgt]
            keep = np.ones(len(lp), dtype=bool)
            keep[tid] = False
            mlp = log_softmax_np(lg.astype(np.float64)[keep])
            blp = log_softmax_np(base_logits[i].astype(np.float64)[keep])
            coll = float(np.sum(np.exp(mlp) * (mlp - blp)))
            probs = np.exp(lp[np.asarray(cand)])
            records.append(dict(
                cycle=cyc, arm=arm, base_day_index=b,
                target_shift_days=int(shifts[js]), target_day_index=tgt,
                dose_fraction=float(doses[jd]),
                delta_norm=float(np.linalg.norm(deltas[i, js, jd])),
                realized_top_weekday_index=int(np.argmax(probs)),
                target_token_probability=float(np.exp(lp[tid])),
                base_target_token_probability=float(np.exp(base_lp[tid])),
                target_probability_mass_moved=float(np.exp(lp[tid]) - np.exp(base_lp[tid])),
                collateral_kl_model_to_base_non_target=max(coll, 0.0),
                weekday_token_probabilities=[float(x) for x in probs]))
        print(f"[steerB] {cyc} context {i + 1}/{nb}", flush=True)

with open(f"{FLAG}/steer_records.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
summary = {}
for cyc in ("week", "month"):
    for arm in ("manifold", "flat"):
        rs = [r for r in records if r["arm"] == arm and r["cycle"] == cyc
              and r["dose_fraction"] == 1.0]
        if rs:
            summary[f"{cyc}_{arm}"] = dict(
                endpoint_target_accuracy=float(np.mean(
                    [r["realized_top_weekday_index"] == r["target_day_index"] for r in rs])),
                mean_endpoint_target_prob=float(np.mean(
                    [r["target_token_probability"] for r in rs])),
                mean_endpoint_mass_moved=float(np.mean(
                    [r["target_probability_mass_moved"] for r in rs])),
                mean_endpoint_collateral=float(np.mean(
                    [r["collateral_kl_model_to_base_non_target"] for r in rs])))
json.dump(summary, open(f"{FLAG}/steer_summary.json", "w"), indent=2)
print("STAGE_B DONE", json.dumps(summary), flush=True)
