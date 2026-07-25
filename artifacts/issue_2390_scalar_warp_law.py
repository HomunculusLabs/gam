"""Which scalar law should the locscale warp integral use?

Exact object: E_pi[f(b.beta_w)] where pi = N(mu, S) truncated to the CONE
{beta_w >= 0}. Compared (no gam code):
  (b) [0,inf)-truncated Gaussian, UNCONSTRAINED moments   <- today, post-#2390
  (c) untruncated Gaussian, CONSTRAINED moments           <- iss-2417 proposal
  (d) [0,inf)-truncated Gaussian, CONSTRAINED moments     <- double-counts
  (e) EXACT pushforward by rejection sampling             <- the reference
f is survival-like and nonlinear.
"""
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(11)
N = 60_000_000

def f(w, shift):  return 0.5*(1.0 - np.tanh((shift + w)/1.7))

def trunc_moments(m, s):           # E,Var of N(m,s^2) | >=0
    a = m/s; lam = np.exp(norm.logpdf(a) - norm.logcdf(a))
    e = m + s*lam; v = s*s*(1 - lam*(lam + a))
    return e, max(v, 0.0)

def run(mu, S, b, shift, label):
    p = len(mu)
    L = np.linalg.cholesky(S)
    draws = mu + rng.standard_normal((N, p)) @ L.T
    feas = np.all(draws >= 0.0, axis=1)
    acc = draws[feas]
    w_exact = acc @ b
    exact = f(w_exact, shift).mean()

    m_u, v_u = b @ mu, b @ S @ b            # unconstrained scalar moments
    s_u = np.sqrt(v_u)
    # (b) truncated Gaussian, unconstrained moments
    zz = rng.standard_normal(4_000_000)
    wb = m_u + s_u*zz; wb = wb[wb >= 0.0]
    approx_b = f(wb, shift).mean()
    # constrained (cone-truncated) scalar moments, from the SAME accepted draws
    m_c, v_c = w_exact.mean(), w_exact.var()
    s_c = np.sqrt(v_c)
    # (c) untruncated Gaussian with constrained moments
    wc = m_c + s_c*zz
    approx_c = f(wc, shift).mean()
    # (d) truncated Gaussian with constrained moments (double-count)
    wd = m_c + s_c*zz; wd = wd[wd >= 0.0]
    approx_d = f(wd, shift).mean()
    print(f"{label}: accept={feas.mean():.3f}  exact E[f]={exact:.8f}")
    print(f"   (b) trunc + unconstrained moments : {approx_b:.8f}  err {abs(approx_b-exact):.2e}")
    print(f"   (c) untrunc + constrained moments : {approx_c:.8f}  err {abs(approx_c-exact):.2e}")
    print(f"   (d) trunc  + constrained moments  : {approx_d:.8f}  err {abs(approx_d-exact):.2e}")
    print(f"   frac of (c) mass at w<0: {norm.cdf(-m_c/s_c):.4f}")

b2 = np.array([0.7, 0.3])
run(np.array([0.05, 0.02]), np.array([[0.025,0.006],[0.006,0.018]]), b2, 0.3, "q=2 near-wall")
run(np.array([0.40, 0.35]), np.array([[0.025,0.006],[0.006,0.018]]), b2, 0.3, "q=2 interior  ")
b4 = np.array([0.5,0.3,0.15,0.05])
S4 = np.array([[0.030,0.008,0.004,0.002],[0.008,0.025,0.006,0.003],
               [0.004,0.006,0.020,0.005],[0.002,0.003,0.005,0.015]])
run(np.array([0.03,0.02,0.02,0.01]), S4, b4, 0.3, "q=4 near-wall")
