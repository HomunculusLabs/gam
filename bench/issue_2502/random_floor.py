import numpy as np
K=11010;k=8;P=128
B=np.fromfile('lin_pm/decoder_blocks.bin',dtype=np.float64).reshape(K,2,P)
A=B[:,0,:].copy();U=B[:,1,:].copy()
Xtr=np.fromfile('doc_chart.bin',dtype=np.float64).reshape(-1,P);mu=Xtr.mean(0)
Xte=np.fromfile('test_chart.bin',dtype=np.float64).reshape(-1,P)-mu
Xs=Xtr[:50000]-mu
def ev(X,A,U,k=8,bs=4000):
    nrm=np.linalg.norm(U,axis=1);good=nrm>0
    Un=np.zeros_like(U);Un[good]=U[good]/nrm[good,None]
    Asq=(A*A).sum(1);AdU=(A*Un).sum(1);sse=0.0
    for lo in range(0,X.shape[0],bs):
        R=X[lo:lo+bs].copy()
        for _ in range(k):
            rA=R@A.T;rU=R@Un.T
            g=2*rA-Asq[None,:]+(rU-AdU[None,:])**2;g[:,~good]=-np.inf
            b=g.argmax(1);t=rU[np.arange(len(b)),b]-AdU[b]
            R-=A[b]+t[:,None]*Un[b]
        sse+=(R*R).sum()
    return 1.0-sse/(X*X).sum()
rng=np.random.default_rng(2502)
nA=np.linalg.norm(A,axis=1);nU=np.linalg.norm(U,axis=1)
# (a) isotropic gaussian, norms matched to ours
Ar=rng.standard_normal((K,P)); Ar/=np.linalg.norm(Ar,axis=1,keepdims=True); Ar*=nA[:,None]
Ur=rng.standard_normal((K,P)); Ur/=np.linalg.norm(Ur,axis=1,keepdims=True); Ur*=nU[:,None]
# (b) random DATA rows as atoms -- the cheapest non-learned dictionary there is
idx=rng.choice(Xs.shape[0],size=2*K,replace=False)
Ad=Xs[idx[:K]].copy(); Ud=Xs[idx[K:]].copy()
Ad*= (nA/np.linalg.norm(Ad,axis=1))[:,None]; Ud*=(nU/np.linalg.norm(Ud,axis=1))[:,None]
print('held-out EV, K=11010, k=8, affine form, 2.819M params')
print('  ours (fitted)              : %.4f'%ev(Xte,A,U),flush=True)
print('  random gaussian, norms matched : %.4f'%ev(Xte,Ar,Ur),flush=True)
print('  random DATA rows as atoms      : %.4f'%ev(Xte,Ad,Ud),flush=True)
print('  steel_k10525 (1.347M)          : 0.8846  [reference]')
