"""gam#2484 — an independent replication of the Murphy–Topel channel for a
global-empirical latent measure.

This is a SECOND implementation of the derivation, in another language and off
another CDF library, checked against finite differences of its own objective.
The Rust gate in `crates/gam-models/src/bms/empirical_measure_2484_tests.rs`
checks the shipped code against the shipped builder; this one checks the
DERIVATION, so a formula that was transcribed consistently-but-wrongly into
both the code and its test would still be caught here.

What it measures, and what it printed when it was written:

    D max abs err vs FD:                    1.46e-10
    total mixed derivative max rel err:     1.28e-07
    |direct| = 3.855   |cross| = 0.418

The last line is the one worth keeping: the cross-row channel is ~11% of the
direct channel in norm on an ordinary fixture, so it is neither negligible
(which would make the issue a rounding error) nor dominant.

    D           = d(node)/d(zeta), the measure's sensitivity to the sample it
                  was compressed from;
    total       = d2(log L)/d(beta) d(zeta_j), the number the covariance seam
                  consumes, against a DOUBLE central difference of the
                  log-likelihood with the grid rebuilt at every perturbed zeta.

Run: python3 scripts/probe_2484_empirical_measure_sensitivity.py
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

GRID=5
def build_grid(zeta, w):
    idx=[i for i in range(len(zeta)) if w[i]>0]
    idx.sort(key=lambda i: zeta[i])
    tot=sum(w[i] for i in idx)
    m=min(GRID,len(idx))
    target=tot/m
    nodes=[];wts=[];alpha=[];binmass=[]
    cur=0; rem=w[idx[0]]
    for _ in range(m):
        need=target; bw=0.0; bs=0.0; ba=[]
        while need>1e-14*target and cur<len(idx):
            take=min(rem,need)
            bs+=take*zeta[idx[cur]]; bw+=take; ba.append((idx[cur],take))
            need-=take; rem-=take
            if rem<=1e-14*w[idx[cur]]:
                cur+=1
                if cur<len(idx): rem=w[idx[cur]]
        if bw>0:
            b=len(nodes); nodes.append(bs/bw); wts.append(bw/tot); binmass.append(bw)
            for (r,t) in ba: alpha.append((b,r,t))
    nodes=np.array(nodes); wts=np.array(wts)
    total=wts.sum()
    mu=(wts*nodes).sum()/total
    var=(wts*(nodes-mu)**2).sum()/total
    sd=np.sqrt(var)
    if sd>1e-12: nodes=(nodes-mu)/sd
    else: sd=None
    wts=wts/total
    return nodes,wts,alpha,np.array(binmass),sd

def D_matrix(zeta,w):
    x,pi,alpha,binmass,sd=build_grid(zeta,w)
    m=len(x); n=len(zeta)
    A=np.zeros((m,n))
    for (b,r,t) in alpha: A[b,r]+=t/binmass[b]
    if sd is None: M=np.eye(m); inv=1.0
    else:
        M=np.zeros((m,m))
        for b in range(m):
            for c in range(m):
                M[b,c]=( (1.0 if b==c else 0.0) - pi[c]) - x[b]*pi[c]*x[c]
        inv=1.0/sd
    return inv*M@A, x, pi

np.random.seed(0)
n=12
zeta=np.array([-1.83,-1.10,-0.74,-0.31,-0.05,0.17,0.42,0.68,0.95,1.31,1.77,2.40])
w=np.array([0.7,1.2,0.9,3.5,0.0,1.1,0.8,1.4,0.6,0.9,0.7,0.6])
D,x,pi=D_matrix(zeta,w)
h=1e-6
err=0
for i in range(n):
    zp=zeta.copy(); zp[i]+=h
    zm=zeta.copy(); zm[i]-=h
    xp,_,_,_,_=build_grid(zp,w); xm,_,_,_,_=build_grid(zm,w)
    fd=(xp-xm)/(2*h)
    err=max(err,np.max(np.abs(fd-D[:,i])))
print("D max abs err vs FD:", err)

# ---- row channels ----
from scipy.optimize import brentq
S=0.87
def intercept(mu_t, g, x, pi):
    f=lambda a: (pi*norm.cdf(a+S*g*x)).sum()-mu_t
    lo,hi=-50.0,50.0
    return brentq(f,lo,hi,xtol=1e-14,rtol=1e-15,maxiter=200)

nrow=10
Xm=np.zeros((nrow,2)); Xg=np.zeros((nrow,2)); y=np.zeros(nrow); zr=np.zeros(nrow); wr=np.zeros(nrow)
for i in range(nrow):
    t=i/(nrow-1)
    Xm[i]=[1.0, 2*t-1]
    Xg[i]=[1.0, np.sin(3*t-1.4)]
    y[i]=1.0 if i%3==0 else 0.0
    zr[i]=-1.7+0.41*i+0.07*np.sqrt(i*i)
    wr[i]=0.6+0.13*(i%4)
beta=np.array([0.15,-0.42,0.55,0.23])

def loglik(beta, zeta):
    x,pi,_,_,_=build_grid(zeta,wr)
    me=Xm@beta[:2]; ge=Xg@beta[2:]
    tot=0.0
    for i in range(len(zeta)):
        mu_t=norm.cdf(me[i])
        a=intercept(mu_t, ge[i], x, pi)
        e=a+S*ge[i]*zeta[i]
        tot+=wr[i]*norm.logcdf((2*y[i]-1)*e)
    return tot

def analytic(beta, zeta):
    x,pi,alpha,binmass,sd=build_grid(zeta,wr)
    Dm,_,_=D_matrix(zeta,wr)
    m=len(x); n=len(zeta); p=4
    me=Xm@beta[:2]; ge=Xg@beta[2:]
    direct=np.zeros((n,p)); Cm=np.zeros((n,m)); Cg=np.zeros((n,m))
    for i in range(n):
        mu_t=norm.cdf(me[i]); mu1=norm.pdf(me[i])
        g=ge[i]; a=intercept(mu_t,g,x,pi)
        eta=a+S*g*x
        p1=pi*norm.pdf(eta); p2=pi*(-eta*norm.pdf(eta))
        Psi1=p1.sum(); Psi2=p2.sum()
        Xi1=(p1*(S*x)).sum(); Xi2=(p2*(S*x)).sum()
        a_m=mu1/Psi1; a_g=-Xi1/Psi1
        sig=2*y[i]-1
        e=a+S*g*zeta[i]; r=sig*e
        # L = w*logPhi(r); L' = w*mills, L'' = w*(-mills*(r+mills))
        mills=np.exp(norm.logpdf(r)-norm.logcdf(r))
        L1=wr[i]*mills; L2=wr[i]*(-mills*(r+mills))
        e_m=a_m; e_g=a_g+S*zeta[i]
        # d2 logL/dtheta dzeta = L2*e_theta*e_zeta + sig*L1*e_{theta,zeta}
        direct[i,:2]=(L2*e_m*(S*g))*Xm[i]
        direct[i,2:]=(L2*e_g*(S*g)+sig*L1*S)*Xg[i]
        for b in range(m):
            a_xb=-S*g*p1[b]/Psi1
            dPsi=Psi2*a_xb+S*g*p2[b]
            dXi=Xi2*a_xb+S*g*p2[b]*(S*x[b])+S*p1[b]
            a_mx=-a_m*dPsi/Psi1
            a_gx=-(dXi+a_g*dPsi)/Psi1
            Cm[i,b]=L2*e_m*a_xb+sig*L1*a_mx
            Cg[i,b]=L2*e_g*a_xb+sig*L1*a_gx
    U=np.zeros((m,p)); U[:,:2]=Cm.T@Xm; U[:,2:]=Cg.T@Xg
    return direct + Dm.T@U, direct, Dm.T@U

A,direct,cross=analytic(beta,zeta[:nrow]*0+zr)
hb=1e-4; hz=1e-4
worst=0
for j in range(nrow):
    for k in range(4):
        v=0.0
        for sb,sz in [(1,1),(-1,-1),(1,-1),(-1,1)]:
            b2=beta.copy(); z2=zr.copy()
            b2[k]+=sb*hb; z2[j]+=sz*hz
            v+=sb*sz*loglik(b2,z2)
        fd=v/(4*hb*hz)
        worst=max(worst, abs(fd-A[j,k])/(1+abs(fd)))
print("total mixed derivative max rel err:", worst)
print("|direct| =", np.linalg.norm(direct), " |cross| =", np.linalg.norm(cross))
