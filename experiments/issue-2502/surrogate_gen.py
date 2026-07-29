"""Surrogate vs actual: the model continues text after READING the prefix
through each dictionary. Prefix positions get the in-chart splice (same
mechanics as splice_paired); the KV cache bakes it in; sampled continuations
then show what the model understood from each surrogate reading.
"""
import glob, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

V2 = os.path.expanduser("~/i2502v2")
LAYER = 16
P = 128
rng = np.random.default_rng(1)

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
dense = sorted(by_seq.items(), key=lambda kv: -len(kv[1]))[:3]

name = "Qwen/Qwen3.5-4B-Base"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                             trust_remote_code=True, device_map="cuda:0")
model.eval()
layers = None
for _n, mod in model.named_modules():
    if isinstance(mod, torch.nn.ModuleList) and (layers is None or len(mod) > len(layers)):
        layers = mod
block = layers[LAYER]

ours = np.frombuffer(open(f"{V2}/f_eu8096x/heldout_recon.bin", "rb").read(),
                     dtype=np.float64).reshape(-1, P)

def steel_recon(npz_path):
    blob = np.load(npz_path)
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
    return rec + b

steel = steel_recon(f"{V2}/baseline_k5333_s2.npz")
floor = np.zeros_like(chart) + chart.mean(0)

inject = {"pos": None, "vec": None}

def hook(_m, _i, output):
    tup = isinstance(output, tuple)
    h = output[0] if tup else output
    if inject["pos"] is None or h.shape[1] == 1:
        return output          # decode steps run clean; prefill gets spliced
    h = h.clone()
    idx = torch.tensor(inject["pos"], dtype=torch.long, device=h.device)
    h[0, idx, :] += torch.tensor(inject["vec"], dtype=h.dtype, device=h.device)
    return (h,) + output[1:] if tup else h

handle = block.register_forward_hook(hook)

def continue_from(ids, deltas_pos, deltas_vec, seed):
    torch.manual_seed(seed)
    inject["pos"], inject["vec"] = deltas_pos, deltas_vec
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=55, do_sample=True,
                             temperature=0.8, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    inject["pos"], inject["vec"] = None, None
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " / ")

for s_i, pv in dense:
    pv = sorted(pv)
    cut = pv[int(len(pv) * 0.7)][0] + 1       # prefix ends inside covered zone
    pos = np.array([p for p, _ in pv if p < cut], dtype=np.int64)
    rws = [r for p, r in pv if p < cut]
    ids = torch.tensor(seqs[s_i][None, :cut], dtype=torch.long, device="cuda:0")
    prefix_tail = tok.decode(seqs[s_i][max(0, cut - 40):cut], skip_special_tokens=True).replace("\n", " / ")
    print("=" * 90)
    print("PREFIX (...%s)" % prefix_tail)
    print("[%d of %d prefix positions spliced]" % (len(pos), cut))
    for seed in (0, 1):
        print("- sample seed", seed)
        print("  actual   :", continue_from(ids, None, None, seed))
        print("  ours     :", continue_from(ids, pos, (ours[rws] - chart[rws]) @ lift, seed))
        print("  steelman :", continue_from(ids, pos, (steel[rws] - chart[rws]) @ lift, seed))
        print("  lobotomy :", continue_from(ids, pos, (floor[rws] - chart[rws]) @ lift, seed))
handle.remove()
print("DONE")
