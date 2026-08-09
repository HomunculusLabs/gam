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
w=np.array([0.7,1.2,0.9,6.0,0.0,1.1,0.8,1.4,0.6,0.9,0.7,0.6])
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


# ---------------------------------------------------------------------------
# The differentiability certificate, brute-forced.
#
# The shipped rule is "refuse iff an equal-mass bin boundary lies strictly
# inside a tied group's cumulative-mass span", and it is narrower than "refuse
# on ties". This measures the thing the rule is a proxy for: the ONE-SIDED
# derivatives of the grid nodes at a tied row, from each side of the tie.
#
# Measured when written, six unit-weight rows with two of them tied at 0.5:
#
#     bins=3 (the tie lies inside one bin):   6.66e-09   -- roundoff, differentiable
#     bins=4 (a boundary cuts the tie):       5.39e-01   -- genuinely two-sided
#
# So the contained tie must NOT be refused and the cut tie must be, which is
# exactly what the certificate does.
# ---------------------------------------------------------------------------


def one_sided_node_derivative(zeta, w, row, h, sign):
    """Right (sign=+1) or left (sign=-1) difference quotient of every node."""
    base, _, _, _, _ = build_grid(np.array(zeta, dtype=float), w)
    moved = np.array(zeta, dtype=float)
    moved[row] += sign * h
    shifted, _, _, _, _ = build_grid(moved, w)
    return (shifted - base) / (sign * h)


def tie_certificate_report():
    global GRID
    tied = [-2.0, -1.0, 0.5, 0.5, 1.0, 2.0]
    weights = np.ones(6)
    for bins in (3, 4):
        GRID = bins
        right = one_sided_node_derivative(tied, weights, 2, 1e-7, +1)
        left = one_sided_node_derivative(tied, weights, 2, 1e-7, -1)
        gap = float(np.max(np.abs(right - left)))
        verdict = "differentiable" if gap < 1e-6 else "TWO-SIDED"
        print(f"bins={bins}: max |right - left| at the tied row = {gap:.6e}  ({verdict})")
    GRID = 5


tie_certificate_report()


# ---------------------------------------------------------------------------
# What the cross-row channel is worth ON THE PUBLISHED INTERVAL, and why it is
# smaller there than its norm suggests.
#
# `|cross| / |direct|` is a property of the sensitivity MATRIX. What a user
# sees is the standard error, after `G = S_eff^T J`, `Vb G`, and the `V1`
# congruence — three contractions that can and do damp it. Reporting only the
# matrix ratio would overstate the change.
#
# There is a structural reason for the damping, and it is worth stating because
# it bounds the whole channel: the grid is STANDARDIZED. Its nodes carry
# weighted mean 0 and weighted sd 1 by construction, so no perturbation of zeta
# can move the measure's location or scale — only its SHAPE. `M` is exactly the
# projection that enforces that (`(M^T v)_c = v_c − w_c*sum_b v_b −
# w_c*x_c*sum_b x_b v_b` annihilates a node channel that is constant, and one
# that is proportional to the nodes). A node channel with no shape content
# contributes exactly nothing.
#
# Measured over grid size and logslope magnitude (V1 a scaled KMS correlation,
# Vb = 0.25*I + 0.02 off-diagonal, so the ABSOLUTE ratios are fixture-specific;
# the pattern across the sweep is not):
#
#     grid  slope  |cross|/|direct|  max SE corr/naive  max SE corr/direct-only
#        3    0.3            0.0058             1.0577                 1.000012
#        3    1.0            0.1053             1.3189                 1.000634
#        3    3.0            0.6774             2.5301                 1.008743
#        5    0.3            0.0066             1.0577                 1.000018
#        5    1.0            0.1084             1.3190                 1.000831
#        5    3.0            0.8048             2.5372                 1.009017
#        8    0.3            0.0070             1.0577                 1.000020
#        8    1.0            0.1116             1.3190                 1.000893
#        8    3.0            0.7951             2.5343                 1.008965
#
# Three things fall out, and the second is the one that has to be said plainly.
#
# 1. The CORRECTION matters: 1.06x to 2.53x on the SE against the naive
#    covariance. Publishing the naive matrix — the thing this seam refuses to
#    do — understates the interval by that much.
# 2. The CROSS-ROW HALF of it is second-order on these fixtures: 1.2e-5 to
#    9.0e-3 relative on the SE. An implementation that used only the direct
#    channel would be wrong by well under a percent here. That is exactly why
#    it would have been a dangerous shortcut rather than an obvious one — it is
#    the size of error nobody notices.
# 3. It is NOT uniformly small, and what it scales with is the logslope, not
#    the grid size: 3x the slope moves it by ~700x on the matrix and ~14x on
#    the SE, while 3x the grid size barely moves either. The channel exists
#    because the slope is what lets a row see the latent axis at all
#    (`da/dx_b = 0` at `g = 0`), so a strongly-sloped fit is where it bites.
# ---------------------------------------------------------------------------


def standard_error_impact_report():
    global GRID
    conditioning = np.hstack([np.ones((nrow, 1)),
                              np.array([[(i - 4.5) / 4.5] for i in range(nrow)])])
    mean_c = np.array([0.05, 0.31])
    var_c = np.array([0.9, 0.12])
    floor = 1.0e-6
    raw_z = 0.4 * zr + 0.05
    jac = np.zeros((nrow, 4))
    for i in range(nrow):
        m_ = conditioning[i] @ mean_c
        v_ = max(conditioning[i] @ var_c, floor)
        jac[i, :2] = (-1.0 / np.sqrt(v_)) * conditioning[i]
        centered = (raw_z[i] - m_) / np.sqrt(v_)
        jac[i, 2:] = (-centered / (2.0 * v_)) * conditioning[i]
    v1 = np.array([[0.04 * 0.6 ** abs(i - j) for j in range(4)] for i in range(4)])
    vb = np.full((4, 4), 0.02)
    np.fill_diagonal(vb, 0.25)

    def congruence(sensitivity):
        g = sensitivity.T @ jac
        vg = vb @ g
        return vg @ v1 @ vg.T

    naive = np.sqrt(np.diag(vb))
    print(f"{'grid':>5} {'slope':>6} {'|cross|/|direct|':>17} {'SE corr/naive':>14}"
          f" {'SE corr/direct-only':>21}")
    for grid_size in (3, 5, 8):
        for slope_scale in (0.3, 1.0, 3.0):
            GRID = grid_size
            scaled = beta.copy()
            scaled[2:] = beta[2:] * slope_scale
            total, direct_only, cross_only = analytic(scaled, zr)
            se_total = np.sqrt(np.diag(vb) + np.diag(congruence(total)))
            se_direct = np.sqrt(np.diag(vb) + np.diag(congruence(direct_only)))
            print(f"{grid_size:>5} {slope_scale:>6} "
                  f"{np.linalg.norm(cross_only) / np.linalg.norm(direct_only):>17.4f} "
                  f"{np.max(se_total / naive):>14.4f} "
                  f"{np.max(se_total / se_direct):>21.6f}")
    GRID = 5


standard_error_impact_report()
