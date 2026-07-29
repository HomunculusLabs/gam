import os, unicodedata, collections
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
V2=os.path.expanduser("~/i2502v2")
name="Qwen/Qwen3.5-4B-Base"
tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True)

def is_cjk(s):
    return any(unicodedata.name(c,"").startswith(("CJK","HIRAGANA","KATAKANA","HANGUL")) for c in s)

V=len(tok)
samp=np.random.default_rng(0).choice(V,20000,replace=False)
cjk=sum(is_cjk(tok.decode([int(i)])) for i in samp)
print("Qwen vocab size %d ; CJK share of vocabulary: %.1f%%"%(V,100*cjk/len(samp)))

model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.bfloat16,
        trust_remote_code=True,device_map="cuda:0").eval()
lift=np.load(V2+"/lift.npy"); c0=np.load(V2+"/c0.npy")
D=lift.shape[1]
rng=np.random.default_rng(1)

def top_tokens(vecs,k=4):
    x=torch.tensor(vecs,dtype=torch.bfloat16,device="cuda:0")
    with torch.no_grad():
        p=model.lm_head(model.model.norm(x)).float().softmax(-1)
        t=p.topk(k,dim=-1)
    return [[tok.decode([int(i)]) for i in row] for row in t.indices]

for scale,label in [(1.0,"scale 1 (in-distribution magnitude)"),(8.0,"scale 8 (what the figure used)")]:
    # random chart directions, lifted the same way the figure lifts curve points
    chart=rng.normal(size=(12,128)); chart/=np.linalg.norm(chart,axis=1,keepdims=True)
    amb=(chart@lift+c0)*scale
    rows=top_tokens(amb)
    n_cjk=sum(any(is_cjk(w) for w in r) for r in rows)
    print("\nRANDOM directions, %s"%label)
    print("  rows whose top-4 contain CJK: %d of %d"%(n_cjk,len(rows)))
    for r in rows[:6]: print("   ", " ".join(repr(w) for w in r))
