import numpy as np
K=11010;k=8;P=128
B=np.fromfile('lin_pm/decoder_blocks.bin',dtype=np.float64).reshape(K,2,P)
A=B[:,0,:].copy();U=B[:,1,:].copy()
Xtr=np.fromfile('doc_chart.bin',dtype=np.float64).reshape(-1,P);mu=Xtr.mean(0)
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

print('lin_pm K=11010, held-out, k=8, ALL AT 2.819M DECODER PARAMS')
print('  affine  b0 + t*b1, K=11010      : %.4f'%ev(Xte,A,U))
# same numbers, reparameterised: every vector becomes its own pure direction
D=np.concatenate([A,U],0)                     # 22020 directions, 2.819M params
print('  split   t*d, K=22020 directions : %.4f'%ev(Xte,np.zeros_like(D),D))
# control: only the b1 halves, half the budget
print('  b1 only t*b1, K=11010           : %.4f  (1.409M, half budget)'%ev(Xte,np.zeros_like(U),U))
print('  b0 only t*b0, K=11010           : %.4f  (1.409M, half budget)'%ev(Xte,np.zeros_like(A),A))
