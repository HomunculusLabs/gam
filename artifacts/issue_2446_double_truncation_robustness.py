"""Does deleting the second truncation still win at the moments that SHIP?

The exact-moment table said (c) beats (d) by 2-3 orders. But the shipped
constrained moments come from `orthant_truncated_moments`, converged only to
ORTHANT_MOMENT_RELATIVE_TOLERANCE = 1e-3 for q>=2 (closed-form/exact at q=1).
So: perturb the constrained moments by relative error eps and find where the
ordering flips.
"""
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(5)
N = 40_000_000
def f(w, shift): return 0.5*(1.0 - np.tanh((shift + w)/1.7))

def study(mu, S, b, shift, label):
    p=len(mu); L=np.linalg.cholesky(S)
    d=mu + rng.standard_normal((N,p)) @ L.T
    acc=d[np.all(d>=0.0,axis=1)]
    w=acc@b
    exact=f(w,shift).mean()
    m_c, v_c = w.mean(), w.var()          # EXACT cone-truncated scalar moments
    m_u, v_u = b@mu, b@S@b                # unconstrained
    zz=rng.standard_normal(6_000_000)
    # (b) truncated + unconstrained  (pre-cutover, = released 0.1.259 behaviour)
    wb=m_u+np.sqrt(v_u)*zz; eb=f(wb[wb>=0],shift).mean()
    print(f"\n{label}  exact={exact:.8f}   [(b) released = {abs(eb-exact):.2e}]")
    print(f"   {'rel err in moments':>20} | {'(d) trunc+constr':>17} | {'(c) untrunc+constr':>19} | winner")
    for eps in (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        # worst-case coherent perturbation: bias mean and sd the same direction
        mm = m_c*(1.0+eps); ss = np.sqrt(v_c)*(1.0+eps)
        wc = mm+ss*zz
        ec = abs(f(wc,shift).mean()-exact)
        wd = mm+ss*zz; wd = wd[wd>=0]
        ed = abs(f(wd,shift).mean()-exact)
        win = "(c)" if ec<ed else "(d)"
        print(f"   {eps:20.0e} | {ed:17.2e} | {ec:19.2e} | {win}")

b2=np.array([0.7,0.3])
study(np.array([0.05,0.02]), np.array([[0.025,0.006],[0.006,0.018]]), b2, 0.3, "q=2 near-wall")
b4=np.array([0.5,0.3,0.15,0.05])
S4=np.array([[0.030,0.008,0.004,0.002],[0.008,0.025,0.006,0.003],
             [0.004,0.006,0.020,0.005],[0.002,0.003,0.005,0.015]])
study(np.array([0.03,0.02,0.02,0.01]), S4, b4, 0.3, "q=4 near-wall")
