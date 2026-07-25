"""Independent numerical check of the #2417 estimand claim.

The claim: for a constrained fit, the posterior is the unconstrained Gaussian
N(beta_unc, H^-1) TRUNCATED to the feasible cone C = {A beta >= b}, and the two
shipped answers are the endpoints of one decomposition:

  * PIRLS path reports the FULL unconstrained covariance H^-1   (too large)
  * blockwise path reports Z (Z' H Z)^-1 Z' , exactly ZERO variance
    in the constraint-normal directions                          (too small)

and the truth is strictly between. Also checks the two identities I verified by
hand, and the point-estimate claim (posterior MEAN != constrained MODE).

Everything here is brute force: sample the truncated Gaussian by rejection, and
compare against the closed forms. No gam code involved -- the point is to check
the MATH independently of the implementation.
"""
import numpy as np

rng = np.random.default_rng(20260725)

# --- a small constrained problem -----------------------------------------
p = 4
M = rng.normal(size=(p, p))
H = M @ M.T + 2.0 * np.eye(p)          # SPD "Hessian" of the penalized objective
Hinv = np.linalg.inv(H)

# one active inequality row: beta_0 >= 0  (the monotone/nonneg cone face)
A = np.zeros((1, p)); A[0, 0] = 1.0
b = np.zeros(1)

# Put the unconstrained optimum OUTSIDE the feasible set so the constraint binds.
beta_unc = np.array([-0.60, 0.30, -0.20, 0.10])

# constrained mode: minimize 1/2 (x-mu)' H (x-mu) s.t. A x >= b, active => A x = b
# KKT on the active face:  H(x-mu) = A' lam ,  A x = b
W = A @ Hinv @ A.T
lam = np.linalg.solve(W, b - A @ beta_unc)
beta_mode = beta_unc + Hinv @ A.T @ lam
assert abs(A @ beta_mode - b) < 1e-12, "mode must sit ON the face"
assert lam[0] > 0, f"multiplier must be strictly positive (binding); got {lam[0]}"

# --- the three candidate covariances -------------------------------------
# 1. PIRLS path: full unconstrained
cov_pirls = Hinv

# 2. blockwise path: reduce to the face.  Z = null(A)
Z = np.linalg.svd(A)[2][A.shape[0]:].T          # p x (p-q)
cov_face = Z @ np.linalg.inv(Z.T @ H @ Z) @ Z.T

# --- the decomposition's two identities (the ones I checked by hand) -----
G = Hinv @ A.T @ np.linalg.inv(W)
P = np.eye(p) - G @ A
m = beta_unc - beta_mode                        # = -H^-1 g, the offset to the unc mean
print("identity  Cov(t,u) = P H^-1 A'  = 0 :", np.abs(P @ Hinv @ A.T).max())
print("identity  E[t]     = P m        = 0 :", np.abs(P @ m).max())
print("identity  P H^-1 P' = Z(Z'HZ)^-1 Z' :", np.abs(P @ Hinv @ P.T - cov_face).max())

# --- brute force: sample the truncated Gaussian ---------------------------
L = np.linalg.cholesky(Hinv)
keep = []
target = 4_000_000
while len(keep) < 400_000:
    z = rng.normal(size=(target, p))
    x = beta_unc + z @ L.T
    ok = (x @ A.T >= b).all(axis=1)
    keep.append(x[ok])
    if sum(len(k) for k in keep) > 400_000:
        break
S = np.vstack(keep)
print(f"\ntruncated sample: {len(S)} draws (acceptance {len(S)/target:.3f})")

cov_true = np.cov(S.T)
mean_true = S.mean(axis=0)

print("\n--- POINT ESTIMATE ---")
print(f"constrained MODE  beta_0 = {beta_mode[0]:+.6f}   (on the bound, by construction)")
print(f"posterior MEAN    beta_0 = {mean_true[0]:+.6f}   (strictly interior)")
print(f"  => reporting the mode as the estimate is MAP by the back door (SPEC line 3)")

print("\n--- VARIANCE of the constrained coordinate beta_0 ---")
print(f"  PIRLS (unconstrained) : {cov_pirls[0,0]:.6f}   <- too LARGE")
print(f"  truncated (truth)     : {cov_true[0,0]:.6f}")
print(f"  blockwise (face)      : {cov_face[0,0]:.6f}   <- too SMALL (exactly zero)")
between = cov_face[0, 0] < cov_true[0, 0] < cov_pirls[0, 0]
print(f"  strictly between?     : {between}")

print("\n--- full-matrix distance to the truth (Frobenius) ---")
print(f"  ||PIRLS     - truth|| = {np.linalg.norm(cov_pirls - cov_true):.6f}")
print(f"  ||blockwise - truth|| = {np.linalg.norm(cov_face  - cov_true):.6f}")

print("\n--- what a 95% interval on beta_0 would cover ---")
lo, hi = np.quantile(S[:, 0], [0.025, 0.975])
print(f"  truth (empirical)     : [{lo:+.4f}, {hi:+.4f}]  width {hi-lo:.4f}")
sd_p = np.sqrt(cov_pirls[0, 0])
print(f"  PIRLS  mode +- 1.96sd : [{beta_mode[0]-1.96*sd_p:+.4f}, {beta_mode[0]+1.96*sd_p:+.4f}]"
      f"  width {2*1.96*sd_p:.4f}   (and straddles the infeasible region)")
print(f"  blockwise             : [{beta_mode[0]:+.4f}, {beta_mode[0]:+.4f}]  width 0.0000"
      f"   (degenerate at the bound)")

# --- figure ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10.5, 5.2))
grid = np.linspace(-1.0, 1.0, 900)

ax.hist(S[:, 0], bins=160, range=(-1.0, 1.0), density=True, color="#2B8A3E",
        alpha=0.30, label="truncated posterior (truth, by rejection sampling)")

sd_p = np.sqrt(cov_pirls[0, 0])
ax.plot(grid, np.exp(-0.5*((grid-beta_mode[0])/sd_p)**2)/(sd_p*np.sqrt(2*np.pi)),
        color="#C92A2A", lw=2.4,
        label=f"PIRLS path: N(mode, {cov_pirls[0,0]:.3f})  — var {cov_pirls[0,0]/cov_true[0,0]:.1f}x too large")

ax.axvline(beta_mode[0], color="#495057", lw=2.6,
           label="blockwise path: exactly zero variance (a point mass at the bound)")
ax.axvspan(-1.0, 0.0, color="#868e96", alpha=0.16, zorder=0)
ax.text(-0.5, ax.get_ylim()[1]*0.86, "INFEASIBLE\n(beta_0 < 0)", ha="center",
        fontsize=10, color="#495057")

ax.axvline(mean_true[0], color="#2B8A3E", lw=2.2, ls="--",
           label=f"posterior MEAN = {mean_true[0]:+.3f} (strictly interior; the MODE is 0)")

ax.set_xlabel(r"constrained coefficient $\beta_0$   (feasible set: $\beta_0 \geq 0$)")
ax.set_ylabel("posterior density")
ax.set_title("#2417  Both shipped answers are wrong, in opposite directions", fontsize=13)
ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
ax.grid(alpha=0.22)
fig.tight_layout()
out = "/private/tmp/claude-501/-Users-user-gam/05d0692f-b15d-4ac0-87e3-8351b744c410/scratchpad/issue2417_truncated.png"
fig.savefig(out, dpi=155)
print("\nwrote", out)
