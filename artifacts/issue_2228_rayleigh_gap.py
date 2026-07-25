"""Is 'lambda_i + v_i' E v_i >= -floor for every i' enough to prove M+E is PSD?

That is what construction_exact_hessian.rs:3265-3277 checks: for each eigenpair
(lambda_i, v_i) of M it forms basin = lambda_i + v_i'E v_i and refuses only if
basin < -floor. But v_i'(M+E)v_i is the RAYLEIGH QUOTIENT of M+E at v_i, and the
v_i are eigenvectors of M, not of M+E. Checking those quantities is checking the
DIAGONAL of M+E in M's eigenbasis -- and a matrix with a positive diagonal can
still be indefinite.
"""
import numpy as np
rng = np.random.default_rng(7)

worst = None
for _ in range(200000):
    # M diagonal in its own eigenbasis (wlog), E symmetric PSD-ish "clamp curvature"
    lam = np.array([-0.02, 0.5])                 # M has a small negative eigenvalue
    V = np.eye(2)                                # M's eigenbasis
    E = rng.normal(size=(2, 2)); E = (E + E.T) / 2
    M = np.diag(lam)
    basin = np.array([lam[i] + V[:, i] @ E @ V[:, i] for i in range(2)])
    if basin.min() < 0:            # the code would refuse; not the case we want
        continue
    true_min = np.linalg.eigvalsh(M + E).min()
    if true_min < 0 and (worst is None or true_min < worst[0]):
        worst = (true_min, basin.copy(), E.copy())

tm, basin, E = worst
print("COUNTEREXAMPLE — the per-eigendirection check passes, M+E is indefinite:\n")
print("M           = diag(-0.02, 0.5)")
print("E           =", np.round(E, 4).tolist())
print("basin_i = lambda_i + v_i'E v_i =", np.round(basin, 6), " -> all >= 0, check PASSES")
print("true eigenvalues of M+E        =", np.round(np.linalg.eigvalsh(M + E), 6))
print(f"\n=> certifies PSD while the true minimum eigenvalue is {tm:.6f}")
print("\nWhy: v_i'(M+E)v_i is the Rayleigh quotient of M+E at M's eigenvector.")
print("It equals the DIAGONAL of M+E in M's eigenbasis. A matrix can have an")
print("entirely positive diagonal and still be indefinite -- [[1,2],[2,1]] has")
print("diagonal (1,1) and eigenvalues (3,-1).")
