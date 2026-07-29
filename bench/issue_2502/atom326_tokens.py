import json, os, unicodedata, collections
import numpy as np
from transformers import AutoTokenizer

V2 = os.path.expanduser("~/i2502v2"); d = os.path.join(V2, "b_mix1818")
man = json.load(open(d + "/manifest.json"))
a = sorted([x for x in man if x["kind"] == "periodic"], key=lambda x: -x["usage"])[0]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", trust_remote_code=True)
seqs = np.load(V2 + "/seqs.npy"); sp = np.load(V2 + "/train_seq_pos.npy")

t = np.fromfile(d + "/tokens_%d.bin" % a["idx"]).reshape(-1, 3)
rows = t[:, 0].astype(int)
lo, hi = float(a.get("grid_lo", 0)), float(a.get("grid_hi", 1))
frac = np.clip((t[:, 1] - lo) / max(hi - lo, 1e-12), 0, 1)

def script_of(s):
    best = collections.Counter()
    for ch in s:
        if not ch.strip() or ch.isdigit() or unicodedata.category(ch).startswith("P"):
            continue
        try:
            nm = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        best[nm] += 1
    return best.most_common(1)[0][0] if best else "OTHER"

def profile(rs, label):
    c = collections.Counter(); words = collections.Counter()
    for r in rs:
        s_i, s_p = int(sp[r, 0]), int(sp[r, 1])
        w = tok.decode(seqs[s_i][s_p:s_p + 1])
        c[script_of(w)] += 1; words[w] += 1
    tot = sum(c.values())
    print("\n%s  (n=%d)" % (label, tot))
    for k, v in c.most_common(8):
        print("   %-12s %5.1f%%" % (k, 100 * v / tot))
    print("   top tokens: %s" % " ".join("%r x%d" % (w, n) for w, n in words.most_common(12)))

rng = np.random.default_rng(0)
N = 4000
profile(rng.choice(rows, min(N, len(rows)), replace=False), "ROUTED TO ATOM %d" % a["atom"])
# control: the same number of random corpus positions
allr = rng.choice(len(sp), N, replace=False)
profile(allr, "CORPUS BASELINE (random positions)")
# split the supported arc into its two visual lobes
lobeA = rows[frac > 0.85]; lobeB = rows[frac < 0.15]
profile(rng.choice(lobeA, min(1500, len(lobeA)), replace=False), "arc lobe A (phase > 0.85)")
profile(rng.choice(lobeB, min(1500, len(lobeB)), replace=False), "arc lobe B (phase < 0.15)")
