import numpy as np
K=11010; k=8; P=128
B=np.fromfile('lin_pm/decoder_blocks.bin',dtype=np.float64).reshape(K,2,P)
A=B[:,0,:].copy(); U=B[:,1,:].copy()

Xtr=np.fromfile('doc_chart.bin',dtype=np.float64).reshape(-1,P)
mu=Xtr.mean(0)
Xte=np.fromfile('test_chart.bin',dtype=np.float64).reshape(-1,P)-mu
Xtr=Xtr[:50000]-mu

def omp(X,A,U,k=8,bs=4000):
    n=X.shape[0]; nrm=np.linalg.norm(U,axis=1); good=nrm>0
    Un=np.zeros_like(U); Un[good]=U[good]/nrm[good,None]
    Asq=(A*A).sum(1); AdU=(A*Un).sum(1)
    cnt=np.zeros(A.shape[0],dtype=np.int64); sse=0.0
    for lo in range(0,n,bs):
        R=X[lo:lo+bs].copy()
        for _ in range(k):
            rA=R@A.T; rU=R@Un.T
            g=2*rA-Asq[None,:]+(rU-AdU[None,:])**2
            g[:,~good]=-np.inf
            b=g.argmax(1); cnt+=np.bincount(b,minlength=A.shape[0])
            t=rU[np.arange(len(b)),b]-AdU[b]
            R-=A[b]+t[:,None]*Un[b]
        sse+=(R*R).sum()
    return cnt, 1.0-sse/(X*X).sum()

cnt_tr,ev_tr=omp(Xtr,A,U)
_,ev_te=omp(Xte,A,U)
print('BASELINE  train EV %.4f | HELD-OUT EV %.4f   (harness reported 0.8540)'%(ev_tr,ev_te),flush=True)

dead=np.where(cnt_tr<=1)[0]
print('dead atoms (<=1 of %d picks): %d = %.2f%%'%(cnt_tr.sum(),dead.size,100*dead.size/K),flush=True)

# reclaim: point each dead atom at an unexplained direction, no refit (lower bound)
R=Xtr.copy(); nrm=np.linalg.norm(U,axis=1); good=nrm>0
Un=np.zeros_like(U); Un[good]=U[good]/nrm[good,None]
Asq=(A*A).sum(1); AdU=(A*Un).sum(1)
for _ in range(k):
    rA=R@A.T; rU=R@Un.T
    g=2*rA-Asq[None,:]+(rU-AdU[None,:])**2; g[:,~good]=-np.inf
    b=g.argmax(1); t=rU[np.arange(len(b)),b]-AdU[b]
    R-=A[b]+t[:,None]*Un[b]
res=np.linalg.norm(R,axis=1)
top=np.argsort(res)[::-1][:dead.size]
A2=A.copy(); U2=U.copy()
d=R[top]/np.linalg.norm(R[top],axis=1,keepdims=True)
A2[dead]=0.0
U2[dead]=d*np.median(np.linalg.norm(U[good],axis=1))
_,ev_te2=omp(Xte,A2,U2)
print('RECLAIMED (no refit)  HELD-OUT EV %.4f   delta %+.4f'%(ev_te2,ev_te2-ev_te),flush=True)
