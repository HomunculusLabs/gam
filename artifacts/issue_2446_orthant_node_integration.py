"""(e) integrate f against the orthant cubature's own nodes -- correct moments
AND feasible support at once. Replicates the shipped scheme faithfully:
Genz separation-of-variables + tent-periodized Kronecker lattice with generator
frac(sqrt(p_i)) over the primes (constrained_posterior.rs:915-975, :1255-1266).
Compared against the exact pushforward by rejection sampling.
"""
import numpy as np
from scipy.stats import norm
from sympy import prime
rng = np.random.default_rng(3)

def generator(q):  return np.array([np.sqrt(float(prime(i+1)))%1.0 for i in range(q)])

def orthant_nodes(mu, S, n):
    """Feasible points of {beta >= 0} with their weights, exactly as shipped."""
    q=len(mu); L=np.linalg.cholesky(S); g=generator(q)
    pts=np.zeros((n,q)); logw=np.zeros(n)
    for k in range(n):
        off=k+0.5; z=np.zeros(q); lw=0.0
        for i in range(q):
            bound=-mu[i]-L[i,:i]@z[:i]
            lower=bound/L[i,i]
            lt=norm.logsf(lower)
            if not np.isfinite(lt): lw=-np.inf; break
            lw+=lt
            raw=off*g[i]; frac=raw-np.floor(raw)
            lattice=1.0-abs(2.0*frac-1.0)
            lu=np.log(max(1.0-lattice,np.finfo(float).tiny))+lt
            lu=min(lu,-np.finfo(float).tiny)
            z[i]=-norm.ppf(np.exp(lu))
        logw[k]=lw
        pts[k]=mu+L@z
    ok=np.isfinite(logw)
    w=np.exp(logw[ok]-logw[ok].max())
    return pts[ok], w

def f(w, shift): return 0.5*(1.0-np.tanh((shift+w)/1.7))

def compare(mu,S,b,shift,label,N=40_000_000):
    p=len(mu); L=np.linalg.cholesky(S)
    d=mu+rng.standard_normal((N,p))@L.T
    acc=d[np.all(d>=0.0,axis=1)]; wex=acc@b
    exact=f(wex,shift).mean()
    m_c,v_c=wex.mean(),wex.var()
    zz=rng.standard_normal(4_000_000)
    ec=abs(f(m_c+np.sqrt(v_c)*zz,shift).mean()-exact)      # (c) untrunc+constrained
    print(f"\n{label}  exact={exact:.9f}")
    print(f"   (c) untrunc + exact constrained moments : err {ec:.2e}   [infeasible mass {norm.cdf(-m_c/np.sqrt(v_c)):.3f}]")
    for n in (2048, 8192, 32768):
        pts,w=orthant_nodes(mu,S,n)
        feas=np.all(pts>=-1e-12,axis=1).mean()
        est=(w*f(pts@b,shift)).sum()/w.sum()
        print(f"   (e) node-sum n={n:<6}                      : err {abs(est-exact):.2e}   [feasible nodes {feas:.4f}]")

b2=np.array([0.7,0.3])
compare(np.array([0.05,0.02]), np.array([[0.025,0.006],[0.006,0.018]]), b2, 0.3, "q=2 near-wall")
b4=np.array([0.5,0.3,0.15,0.05])
S4=np.array([[0.030,0.008,0.004,0.002],[0.008,0.025,0.006,0.003],
             [0.004,0.006,0.020,0.005],[0.002,0.003,0.005,0.015]])
compare(np.array([0.03,0.02,0.02,0.01]), S4, b4, 0.3, "q=4 near-wall")
