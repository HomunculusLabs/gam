import numpy as np
P=128;k=8
X=np.fromfile('doc_chart.bin',dtype=np.float64).reshape(-1,P)
mu=X.mean(0)
Xtr=X[:50000]-mu
Xte=np.fromfile('test_chart.bin',dtype=np.float64).reshape(-1,P)-mu
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
K=11010
B=np.fromfile('lin_pm/decoder_blocks.bin',dtype=np.float64).reshape(K,2,P)
A,U=B[:,0,:].copy(),B[:,1,:].copy()
otr,ote=ev(Xtr,A,U),ev(Xte,A,U)
print('ours     train %.4f  held-out %.4f  gap %+.4f'%(otr,ote,otr-ote),flush=True)
z=np.load('steel_k10525_s0.npz')
D=z['W_dec'].astype(np.float64); b=z['b_pre'].astype(np.float64)
Z=np.zeros_like(D)
str_,ste=ev(X[:50000]-b,Z,D),ev(np.fromfile('test_chart.bin',dtype=np.float64).reshape(-1,P)-b,Z,D)
print('steelman train %.4f  held-out %.4f  gap %+.4f'%(str_,ste,str_-ste),flush=True)
