import json, os, unicodedata, collections
import numpy as np
from transformers import AutoTokenizer
V2=os.path.expanduser("~/i2502v2"); d=os.path.join(V2,"b_mix1818")
man=json.load(open(d+"/manifest.json"))
a=sorted([x for x in man if x["kind"]=="periodic"],key=lambda x:-x["usage"])[0]
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base",trust_remote_code=True)
seqs=np.load(V2+"/seqs.npy"); sp=np.load(V2+"/train_seq_pos.npy")
rows=np.fromfile(d+"/tokens_%d.bin"%a["idx"]).reshape(-1,3)[:,0].astype(int)

def script(ch):
    try: n=unicodedata.name(ch)
    except ValueError: return None
    for k in ("CJK","HIRAGANA","KATAKANA","HANGUL","CYRILLIC","ARABIC","HEBREW",
              "DEVANAGARI","THAI","GREEK","LATIN"):
        if k in n: return "CJK" if k in ("CJK","HIRAGANA","KATAKANA") else k
    return None

def ctx_profile(rs,label,W=24):
    c=collections.Counter(); nonlatin_rows=0
    for r in rs:
        s_i,s_p=int(sp[r,0]),int(sp[r,1])
        seq=seqs[s_i]; lo=max(0,s_p-W); hi=min(len(seq),s_p+W)
        txt=tok.decode(seq[lo:hi])
        sc=collections.Counter(x for x in map(script,txt) if x)
        if not sc: continue
        dom=sc.most_common(1)[0][0]
        c[dom]+=1
        if sum(v for k,v in sc.items() if k!="LATIN")>0.10*sum(sc.values()):
            nonlatin_rows+=1
    tot=sum(c.values())
    print("\n%s  (n=%d contexts, +/-%d tokens)"%(label,tot,W))
    for k,v in c.most_common(6): print("   dominant %-11s %5.1f%%"%(k,100*v/tot))
    print("   >=10%% non-Latin characters in context: %.1f%%"%(100*nonlatin_rows/tot))

rng=np.random.default_rng(0); N=3000
ctx_profile(rng.choice(rows,N,replace=False),"CONTEXTS AROUND ATOM %d's TOKENS"%a["atom"])
ctx_profile(rng.choice(len(sp),N,replace=False),"CORPUS BASELINE CONTEXTS")
