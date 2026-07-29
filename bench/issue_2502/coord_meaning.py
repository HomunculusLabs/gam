import json, os, collections
import numpy as np
from transformers import AutoTokenizer
V2=os.path.expanduser("~/i2502v2"); d=os.path.join(V2,"b_mix1818")
man=json.load(open(d+"/manifest.json"))
a=sorted([x for x in man if x["kind"]=="periodic"],key=lambda x:-x["usage"])[0]
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base",trust_remote_code=True)
seqs=np.load(V2+"/seqs.npy"); sp=np.load(V2+"/train_seq_pos.npy")
t=np.fromfile(d+"/tokens_%d.bin"%a["idx"]).reshape(-1,3)
rows=t[:,0].astype(int)
lo,hi=float(a.get("grid_lo",0)),float(a.get("grid_hi",1))
ph=np.clip((t[:,1]-lo)/max(hi-lo,1e-12),0,1)
# rotate so the occupied arc is contiguous: phases >0.5 wrap to negative
x=np.where(ph>0.5, ph-1.0, ph)
occ=(x>-0.16)&(x<0.16)
print("atom %d: %d of %d tokens inside the occupied arc"%(a["atom"],occ.sum(),len(x)))
xs=x[occ]; rs=rows[occ]
qs=np.quantile(xs,np.linspace(0,1,9))
print("\nthe coordinate, sliced into 8 equal-count bins along the arc:\n")
for i in range(8):
    m=(xs>=qs[i])&(xs<=qs[i+1] if i==7 else xs<qs[i+1])
    sel=rs[m]
    if len(sel)==0: continue
    c=collections.Counter()
    for r in sel[:2500]:
        s_i,s_p=int(sp[r,0]),int(sp[r,1])
        c[tok.decode(seqs[s_i][s_p:s_p+1])]+=1
    tot=sum(c.values())
    top=" ".join("%r %.0f%%"%(w,100*n/tot) for w,n in c.most_common(6))
    print("  t=%+.3f..%+.3f  n=%-5d  %s"%(qs[i],qs[i+1],len(sel),top))
