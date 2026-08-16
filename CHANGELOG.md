## Unreleased

- **The post-fit certification was 60.5 % of the fit, and after the half of it
  that had been fixed it was still 99.96 % of `fit_diagnostics_report` — for a
  completely different reason (#2757).** The issue was filed on a dense
  symmetric eigendecomposition of a `param_dim x param_dim` curvature Gram
  (3160.5 s / 45.97 GiB at `p = 4096`). Holding that curvature in the block
  structure the decoder-frame parameterization gives it removed the
  eigendecomposition on the branch a Euclidean metric takes. Nothing measured
  what was left. Phase by phase at `n = 256, p = 64, charts = 32`:

  ```text
  [curvature build]                                  0.071s
  [residual gauge: reduce + generators + verdicts]   0.195s
  [coordinate fidelity]                              0.149s
  [decoder embeddedness]                             0.014s
  [topology persistence]                           547.387s   <====
  ```

  The whole of what the issue names is 0.195 s. The wall is
  `atom_topology_persistence`, and it had never been timed.

  **Root cause 1 — the filtration reduced the wrong matrix.** The persistent
  homology of a Vietoris-Rips complex takes the `(d+1)`-simplices as its
  reduction COLUMNS. At the `PERSISTENCE_H1_MAX_POINTS = 256` cover that is
  `C(256,3) = 2 763 520` triangles, while its pivots are EDGES, of which there
  are `C(256,2) = 32 640`. So at most ~32 000 columns can ever pair and
  ~2 730 000 exist only to be ground down to zero, each costing a chain of
  GF(2) column additions — 99 % of the filtration by measurement, with the
  simplex construction 0.071 s of a 5.8 s call. Persistent COHOMOLOGY has the
  same barcode and the opposite cost profile: its columns are the
  `d`-simplices, the 32 640 edges, and the triangles appear only as entries
  enumerated on demand as cofacets. At `m = 256` the engine now never
  materializes a triangle at all. Clearing (a pivot of degree `d−1` is a death
  partner whose own column is known to reduce to zero) becomes available in the
  order the degrees are already computed, which is exactly what it is not in
  the boundary direction; and the zero-length pairs that dominate a
  Vietoris-Rips filtration resolve on a column's first iteration with no
  addition, because the pivot is the EARLIEST entry.

  **Root cause 2 — the per-atom certificates ran one after another.** The three
  surviving per-atom reads (coordinate fidelity, decoder embeddedness, topology
  persistence) each take `&self` and an atom index and write only their own
  slot, so the serial `(0..k).map()` bought nothing but an evaluation order,
  which an indexed collect keeps. This is the "unparallelised, ~1.0 of 16 cores"
  the issue's own body records.

  ```text
  filtration, circle(160):    reference 9.001s -> 0.374s     24.1x
  filtration, circle(256):    ~17s      -> 3.13s
  topology phase, 32 charts:  547.387s  -> 95.46s
  fit_diagnostics_report:     547.8s    -> 52.0s             10.5x
  ```

  The rewrite is judged against the PRE-#2757 engine, kept verbatim as a
  control that shares none of its reasoning, differenced bar by bar on every
  endpoint's bits across circle, Clifford torus (the `H2` tetrahedron branch),
  line, separated clusters, exactly-tied filtration values, coincident and
  identical points, a far outlier, the four-point floor, DTM-weighted and
  flat-weighted arms, and a 12-member random family. Bars are compared as
  multisets — a barcode is one, and cohomology finds the same pairs in a
  different sequence — and the four functions that read a barcode are shown by
  measurement to agree between the engines and to be unchanged when their own
  bar list is reversed.

  **And the original defect, on the branch the block fix never reached.** With a
  metric that couples output coordinates (an output-Fisher or structured-residual
  harvest), the builder still assembled the dense `param_dim`-square Gram
  whenever `root_rows = n * metric_rank` exceeded `param_dim` — which at the
  #2283 production row count means `m = 480 000` against `param_dim = 65 536`,
  i.e. 1.0e15 flops and a 69 GiB peak. Those rows are now folded into a
  `param_dim`-square upper-triangular factor `T` with `TᵀT = RᵀR`, so every
  production representation carries a ROOT and every rank decision is taken on
  `σ` rather than on `λ = σ²` floored at the eigensolver's own resolution
  (`1.5e-11·λ_max` against `1e-16·λ_max` at that width). Peak memory halves:
  `eigh` allocated a second `param_dim`-square array for eigenvectors the
  reduction discarded.

  This does not make a coupling metric affordable at production width, and the
  module says so rather than implying otherwise: `H` there is a sum of
  `n * metric_rank` rank-one terms with no exploitable structure, and
  `min(rows, param_dim)²` is what an exact full spectrum costs from either side.
  The route out is for the certificate to stop asking for one — of the three
  things it reads off `H`, only `ξᵀHξ` and `λ_max` enter a verdict, and both are
  streamable — which is a change to what the certificate reads, not to how the
  curvature is stored.

- **A two-class multinomial with a smooth term published `β ≡ 0` — every
  predicted probability was the uniform simplex, at `edf_per_class = 4.09`
  (#2612).** The fit was not refused and did not look degenerate from outside:
  it selected interior smoothing parameters, published a full-rank posterior
  covariance and a ρ-uncertainty correction, and reported four effective degrees
  of freedom — next to eight coefficients that were all exactly `0.0`. Nothing
  in the payload contradicted itself loudly enough for any gate to notice,
  because the EDF comes from `H⁻¹S_λ` and the covariance from `H`, and neither
  reads `β`.

  Measured on the #1891 coverage fixture, 40 replications, truth ranging over
  `p ∈ [0.21, 0.82]`:

  ```text
  #2612-INT rep=  0 x*=0.2803 p_true=0.716264 mean=0.500000 sd=0.062987
  #2612-INT rep= 39 x*=0.5941 p_true=0.618759 mean=0.500000 sd=0.046169
  ```

  `mean = 0.500000` on every replication. The standing coverage gate read that
  as an under-covering interval; the centre was a constant.

  One axis varied at a time isolates it exactly — `K = 2` AND a smooth term
  (`K = 2` parametric and `K = 3` smooth are both fine):

  ```text
  K=2 tp k=8      max|beta|=0.000000e0  plugin=[0.500000, 0.500000]  edf=[4.091]
  K=2 parametric  max|beta|=1.020912e0  plugin=[0.411708, 0.660159]  edf=[2.0]
  K=3 tp k=8      max|beta|=6.181604e0  plugin=[0.208740, 0.642326]  edf=[3.38, 2.92]
  ```

  **Root cause.** `MultinomialFamily::specs_match_workspace_shape` required
  `spec.penalties.len() == self.penalties.len()`. Penalties are not part of a
  workspace's GEOMETRY — no penalty ever enters `X_aᵀ diag(w_ab) X_b`; the
  solver adds `s_lambdas` and the joint bundle itself, on the other side of that
  call. The clause predates #1587, which moved this family's entire smoothing
  onto the JOINT penalty and made `build_block_specs` attach
  `penalties: Vec::new()` deliberately ("The per-class blocks attach NO smooth
  penalty"). So from #1587 onward the predicate was FALSE for every penalized
  multinomial: the family declared "I cannot serve a joint workspace" about the
  workspace it does serve. It gates all three `*_available` capabilities and,
  through `has_workspace_source`, the solver's routing:

  ```rust
  use_joint_newton = has_joint_exacthessian && (specs.len() >= 2 || has_workspace_source)
  ```

  `K ≥ 3` has two blocks and reaches the joint path anyway, so there it cost
  only the workspace gradient/log-likelihood/HVP fast paths. `K = 2` has ONE
  block, so the stale clause WAS the routing decision, and the fit fell onto the
  block-coordinate path:

  ```text
  [PIRLS/blockwise step]  block=0 |delta|inf=1.037938e1 block_s_lambda_frob=0.000000e0
  [PIRLS/blockwise trial] bt=0 alpha=1.000e0    -trial_ll=2.210072311e2 prev=1.663553233e2
  [PIRLS/blockwise trial] bt=7 alpha=7.8125e-3  -trial_ll=1.666445362e2 prev=1.663553233e2
  [PIRLS/blockwise convergence] cycle 0 | max_proposed_step=1.038e1 (tol=1.000e-11)
                                | max_accepted_step=0.000e0 | obj_change=0.000e0
  ```

  Eight backtracks, none accepted, converged at cycle 0.

  **And the guard that made it silent.** `exact_joint_stationarity_ok` was
  ASSUMED `true` for single-block fits, and the surrounding test is
  `max_accepted_step <= tol && objective_change <= tol` — both exactly zero when
  the line search accepts nothing. "Nothing moved" and "nothing needed to move"
  are the same two numbers; only the residual separates them, and it was the one
  quantity not consulted. The comment's premise (for one block the blockwise
  iteration IS the joint iteration) is true and licenses TRUSTING the
  block-conditional verdict, not skipping it; the cost it cites is a multi-block
  phenomenon that cannot arise with one block. Now measured for every block
  count.

  After: `max|beta| = 1.023557e1`, plug-in range `[0.313129, 0.822891]` against a
  truth range of `[0.2142, 0.8176]`, deviance `296.59` against the uniform
  model's `332.71`. `K = 3` unchanged to `1e-13`.

- **The multinomial was the one family in the library whose published
  uncertainty never got the covariance-mode axis (#2612).**
  `fit_penalized_multinomial_formula` read `fit.covariance_conditional` and
  stopped; the same `fit_custom_family_with_rho_prior` call had already computed
  the first-order ρ-uncertainty correction `C = J·Var(ρ̂)·Jᵀ` (#2346) and
  published it on the inference block. So every multinomial band answered "how
  wide is the posterior once λ̂ is the truth" while every other family defaults
  to `SmoothingCorrected`.

  It could not even be expressed: `InferenceCovarianceMode` was declared in
  `gam-predict`, which sits ABOVE `gam-models`, so the one family that owns its
  own predict surface was the one family that structurally could not name the
  distinction. The enum now lives in `gam-solve::model_types` beside
  `SmoothingCorrectionMethod`, and `gam-predict` re-exports it — every existing
  path is unchanged.

  The correction reaches the response scale by the law of total variance,
  `Var(p_c) = Var(p_c | ρ̂) + Var_ρ(E[p_c|ρ])`, whose second term is `gᵀCg` with
  `g = ∂p_c/∂θ` the softmax Jacobian at the mode: the response-scale statement
  of `V_c = V_cond + C`, with no new object, no new constant and no new
  approximation order. `SmoothingCorrected` on a fit that retained no correction
  is an error, never a silent downgrade.

  The band is also built on the log-odds scale and transformed
  (`MeanIntervalMethod::TransformEta`, which this library already prefers for
  every nonlinear link). A symmetric `m ± z·sd` band clamped into `[0, 1]` is
  wrong twice where a class probability lives: symmetric about a bounded, skewed
  posterior, and the clamp DELETES the mass that fell outside, so a nominal 95%
  band could carry less than 95% while still reporting `level = 0.95`. `expit`
  is a bijection onto `(0, 1)`, so nothing is ever clipped.
- **A corrected log-determinant and the kernel that differentiates it were two
  fields, so a lane that dropped one kept the other (#2765).**
  Two producers — the custom-family joint assembly and the dense GLM assembly —
  compute the REML pseudo-log-determinant and its trace kernel from ONE
  eigendecomposition of ONE matrix, and both handed them back as two unrelated
  things: the scalar `projected_logdet - hessian_op.logdet()` into
  `InnerSolution::hessian_logdet_correction`, whose documented meaning is a
  UNIFORM CURVATURE RESCALE `-p*log(s)` and nothing else, and the kernel into
  `penalty_subspace_trace`.

  Two fields that travel separately can be separated, and the tangent-projection
  entry separated them. When the inner solve returns on an active inequality
  face the criterion becomes `1/2 log|Z^T H Z|` over that face;
  `try_tangent_projected_evaluate` drops the kernel — correctly, a `p`-space
  subspace kernel does not act on an `m`-dimensional face — while KEEPING the
  scalar, rank-rescaled by `m/p` as though it were the other kind of correction.
  The criterion's VALUE then carried a theta-varying term that no kernel
  anywhere differentiates, and the outer gradient was short by exactly that
  term's derivative, on every theta coordinate.

  `PenaltySubspaceTrace` now carries its own `logdet_correction` and the
  evaluator reads the value correction from the kernel, so the pairing is
  structural: a lane that drops the kernel drops the correction with it. This
  also removes, by construction, the collapse `joint_penalty_subspace_trace_parts`
  documents in its own signature — when the route yields NO kernel the old code
  still applied `0 - hop.logdet()` and silently deleted `1/2 log|H_pen|` from the
  cost while the gradient kept its `1/2 tr(H^-1 dH)` derivative.

- **The log-determinant operator carried a term whose drift is unobtainable
  (#2765).**
  `completion_in_operator` folded the Jeffreys second-order completion into
  `hessian_op` whenever the projected-logdet route was going to own the value and
  the traces — sound on its own terms (the operator is then used only for solves,
  and its `logdet()` cancels exactly), and a PRECONDITION a downstream lane can
  invalidate. On an active face the tangent evaluator takes its determinant from
  that operator's dense assembly directly, so the completion lands in the value
  while the drift that would differentiate it is `D_beta[completion][v]` — a
  third directional derivative no family exposes. The term was not merely
  missing, it was unobtainable, which is exactly why the completion is kept out
  of the scalar everywhere else.

  The completion now goes to the IFT operator unconditionally: the #2612
  separation stated as an invariant instead of as a route-dependent convenience.
  The tangent entry projects that operator onto the same face (`Z^T M_true Z`),
  because on a face `dbeta/dtheta` lies in `range(Z)`; and the cost-side IFT
  displacement `w = H^-1 r` reads `mode_response_operator()` rather than `hop`,
  making true its own claim to be "bit-identical to the gradient side".

  Measured on the #2765 survival marginal-slope fixture (`n=160`, 7.6 s), the
  analytic outer gradient against its own Ridders finite difference:

  ```text
             BEFORE (rel)              AFTER (rel)
    rho_0    8.256e-1  <- sign flip    1.767e-8
    rho_1    1.434e-1                  1.775e-8
    psi_0    9.435e-3                  5.677e-10
    psi_1    8.848e-3                  4.957e-10
  ```

  and `tests/survival/.../survival_marginal_slope_outer_gradient_fd_1040.rs`,
  whose own comment records "this is the analytic marginal-slope psi gradient,
  and it is wrong", now passes its matern arm at `rel = 5.5e-5` against the
  `1.377e-1` it recorded.

- **The composed monotone warp was built one derivative short of the objective
  that differentiates it, so the inner objective had an O(1) JUMP (#2695).**
  `linkwiggle(...)` puts a monotone I-spline on the model's own index —
  `q = q0 + sum_j betaw_j * I_j(q0)` with `q0 = -eta_t * exp(-eta_ls)` — so `q0`
  moves with `beta` while the knots stay where the seed put them, and the
  objective DIFFERENTIATES the basis rather than evaluating it. The row program
  composes it twice, and the second is the one that sets the requirement:

  ```text
    q1w = q1 + sum betaw_j * I_j(q1)      stack [I, I(1), I(2), I(3), I(4)]
    m1  = 1  + sum betaw_j * I(1)_j(q1)   stack [I(1), I(2), I(3), I(4), ...]  <- SHIFTED
    g   = eta_t(1) + m1 * q0dot,  and  the row NLL contains  -d * log g
  ```

  `H = d2(-l)/dbeta2` is the order-2 coefficient of that jet, and `m1`'s order-2
  coefficient reads its stack's slot 2 — which, because `m1` is built from the
  basis's FIRST derivative, is the basis's THIRD. `Phi = 1/2 sum g(lambda(Z_J^T H
  Z_J))` is a TERM OF THE OBJECTIVE (the inner NLL is `-l + 1/2 beta^T S beta -
  Phi`), so the objective's own value reads `I(3)` — while a degree-`d` I-spline
  is only `C^(d-1)` at a simple knot. At the shipped `degree=2` the accept test
  was comparing two points on two different functions:

  ```text
  |delta|inf = 7.094e-13    d_obj = -2.976461e-1
     trial_ll   -1.896965289627e1   IDENTICAL to 12 digits
     trial_pen   6.367949854901e-3  IDENTICAL to 12 digits
     trial_phi  -1.185962102549e1   vs  -1.156197496286e1   <- the whole jump
  ```

  the same `-2.976461e-1` at every step norm from `1.436e-10` to `7.094e-13`.

  A composed warp is now BUILT at `composed_warp_minimum_degree()`, derived as
  `COMPOSED_WARP_OBJECTIVE_BASIS_DERIVATIVE_ORDER + 1 = 4`, and the raise is
  logged with its derivation rather than refused: an earlier attempt refused
  below the floor and was reverted because a refusal breaks every working
  degree-2 fit and buys none of them a fit. The realised degree is what the
  knots, the design, the penalties and the saved metadata all carry. It is
  scoped to simple-ended warps: at a boundary knot of multiplicity `degree + 1`
  the ramp is `C^-1` at EVERY degree, so raising a clamped warp's degree would
  move those fits while fixing nothing.

  **The fourth derivative had to land with it.** The row program's five-slot
  tower ended in the literal `0.0`, which is the fourth derivative of a
  degree-`d` I-spline only for `d <= 3` — exactly the degrees the floor now
  excludes. `survival_wiggle_fourth_basis` supplies it, so no order-3 or order-4
  lowering differentiates a different function than the value it is paired with.
  That coupling is why the earlier attempt could not work: degree 3 was not
  enough (the `betaw`-weighted third-derivative channel) and degree 4 could not
  work while the tower it needed was a literal.

- **A fraction-to-the-boundary backoff was multiplicative, so an active-set
  method behind it could never identify a face (#2695).**
  `feasible_step_fraction` applied `alpha <- 0.995 * alpha` when a row clipped
  the step. The surviving slack after a clipped step is then
  `s + alpha*d = (1 - 0.995)*s`, so every clipped cycle keeps `1/200` of the
  slack and NO finite number of cycles reaches the face. Measured on the #2695
  witness: exactly `200x` per cycle for 400 cycles, with the QP's proposal
  constant at `1.554e-2`, the joint trust radius held, and the objective change
  exactly zero — the solve spending its whole budget walking one warp
  coefficient from `1e-3` to `1e-163` while the row it approached never became
  active.

  A backoff answers ROUND-OFF in an exact ratio test, which is a statement about
  resolution in the scaled-slack metric the contract is denominated in, not
  about how far the step happened to travel. It is now
  `alpha = fraction - PRIMAL_FEASIBILITY_TOL / |scaled drift|`: a step with room
  stops one feasibility tolerance short of the face, and a step whose remaining
  slack is already inside that tolerance yields `alpha <= 0`, which the contract
  reports as `BlockedByActiveFace` and the caller answers with a projection onto
  that face. The row becomes active in ONE cycle. `ContractFeasibleStep` already
  publishes the blocking row's scaled drift, so there is no new geometry and no
  new constant. Landing on the face is not a hazard for these constraints and
  that is checked rather than assumed: the row programs that take a logarithm of
  a bounded quantity carry their own guard (`log g` below the event-Jacobian
  floor is a CONTINUED logarithm), so the interior-point rationale that would
  justify stopping strictly short does not apply.

  Four existing exact-value pins asserted the old constant (`0.995*0.05`,
  `0.995*0.2`, `0.995*0.4`, `0.995*0.5`); they now assert the derived value
  computed from each fixture's stated row geometry, so they still pin an exact
  number and now pin the right one.

  Together these move the witness fit
  (`survival_location_scale_saved_fit_preserves_linkwiggle_metadata`) from a
  terminal stationarity residual of `4.79e-1` against a `7.9e-11` tolerance to
  `1.40e-8` against `1.08e-10` — seven orders — with the inner solve now exiting
  on a measured geometric convergence RATE (`0.9882x/cycle`) rather than on a
  discontinuity, and one seed reaching `KKT/certificate-converged`. The residue
  is a trust-region controller question and is recorded on the issue.

- **A penalty map within `1.5e-8` of a linear dependency was certified EXACTLY
  dependent, because its rank was decided on the SQUARE of the defect (#2676).**
  `PenaltyMapInvariance` licenses the curvature certificate's deflation by
  certifying `sum_i w_i A_i = 0`, and it decided that by eigendecomposing the
  Gram `G_ij = <A_i, A_j>_F` in `f64` and admitting eigenvalues at or under the
  eigensolver's backward error. But `lambda_min(G) = min ||sum_i w_i A_i||_F^2 =
  delta^2`, so the Gram carries the defect squared and a rank test at `G`'s own
  `eps` is a defect test at `sqrt(eps)`.

  Measured on this issue's own headline cell — `geo_disease_matern`,
  `centers=24, n=4000`, via `examples/repro2676_geo_disease_matern`:

  ```
  [INDEF-HESS] pair=(0,2) relative_defect=1.238259e-8 best_scale=1.000000e0
  [INDEF-HESS] active_rank=2/3 structural_zero=1 curvature_resolution=1.170e-8
  [INDEF-HESS] classifications=["Z", "A", "A"]
  [INDEF-HESS] reparam_split ... intrinsic=[-1.1702972950948233e-8, ...]
  ```

  `Z` is "certified null of the penalty map, excused by STRUCTURE". The pair is
  `1.238e-8` apart. **And the error compounds:** with the direction deflated,
  #2748's `invariance_residual_2norm` measures the residual of
  `T' H_rho T = T' diag(g_rho) T` on it and hands the result to the certificate
  as a MEASURED `||dH||_2`. On an exact invariance that residual is error and
  only error — the whole licence for the instrument. On a `1.238e-8` near one it
  is the criterion's genuine curvature, and the dump says so to four digits:
  `curvature_resolution = 1.170e-8` IS `intrinsic = -1.1703e-8`. The certificate
  was told its Hessian was uncertain to `1.17e-8` by a direction whose curvature
  it had just declined to look at — an inflated resolution masking genuine
  negative curvature up to that size at every site that spends it.

  The repair is the classical one — never form the normal equations to get a
  rank — taken in the currency the site can actually afford. Factoring the
  operator stack directly costs `k * block^2` doubles, hundreds of megabytes on a
  wide shared block; doing the same arithmetic in DOUBLE-DOUBLE costs a constant
  factor of time and no memory. The Gram is accumulated with exact products
  (`two_product`, one `mul_add`) and exactly renormalized sums, so an `O(1)`
  entry carries `O(m*eps^2)` instead of `O(eps)`; its pivoted Cholesky runs in
  the same precision, where the pivot `d_j` is the squared norm of `A_j`'s
  residual against the span already accepted, so `sqrt(d_j)` IS that column's
  defect to full relative accuracy at any magnitude down to `eps`; and the null
  space comes out of the FACTOR (`L[:, 0..rank]' w = 0`, back substitution)
  rather than out of an eigenvector of `G`. The boundary is denominated in the
  defect: `sqrt(entries) * EPSILON * ||A||_F * sqrt(accepted + 1)`, one
  operator-construction error per operator entering the residual, in quadrature.
  Nothing is chosen — the model is the arithmetic, calibrated by the two
  populations it separates (a pair known equal at `2.079e-15` against a floor of
  `2.0e-15`; the nearest pair known distinct at `8.75e-9`, six orders away).

  The same defect, in the same coordinate, was in the two human-facing
  instruments and is fixed with it. `report_penalty_pair_redundancy`
  thresholded `cos > 1 - 1e-8` and printed `cos` to six decimals, and
  `1 - cos = delta^2 / 2` — so a pair `1.9e-5` apart printed as `cos = 1.000000`
  and read as an exact identity, while the bar itself admitted anything closer
  than `1.4e-4`. The `[INDEF-HESS]` dump printed
  `structural_redundancy_detected pair=(0,2) cos=1.000000 one_minus_cos=2.42e-9`
  — gated on `cos > 0.999`, a defect of `4.5e-2` — two lines below its own
  `structural_zero=0`. Both are now denominated in `delta`, formed directly from
  the residual at the least-squares scale, and both distinguish the exact case
  (at the residual norm's own arithmetic floor) from the near one, which gets a
  new `near_degenerate_not_an_invariance` line saying what it is: the criterion
  carries genuine curvature of order `delta^2` there, the penalty map certifies
  nothing, and a negative curvature is a resolution question, not a structure
  one.

  Measured before/after through the real fit on the cell the mis-certification
  fired on (`examples/repro2676_geo_disease_matern 24 4000 16 info base`,
  first certification, everything else byte-identical):

  ```
  before  [PENALTY-REDUNDANCY] penalties i=0 j=2 are structurally identical (cos=1.000000)
          [INDEF-HESS] active_rank=2/3 structural_zero=1 curvature_resolution=1.170e-8
          [INDEF-HESS] classifications=["Z", "A", "A"]

  after   [PENALTY-SIMILARITY] penalties i=0 j=2 are close but MEASURABLY distinct
            (relative defect 1.238259e-8 at the best scale c=1.000000e0) ... NOT an invariance
          [INDEF-HESS] active_rank=2/3 structural_zero=0 curvature_resolution=3.780e-16
          [INDEF-HESS] classifications=["G", "A", "A"]
  ```

  Seven and a half orders of fictitious Hessian uncertainty removed, and the fit
  still admits -- the direction is excused by the chain rule (`G`), which is what
  it was always entitled to, rather than by a structure that was not there. The
  cell where nothing was ever mis-certified (`10 1500 16`) is byte-identical
  before and after, which is the control.

  Regression, on this host: `gam-solve --lib` 1930 passed / 0 failed (1726 s),
  `gam-terms --lib` 947 passed / 0 failed, `penalty_invariance` 17 passed / 0
  failed, and the issue's own acceptance 2 passed / 0 failed. `gam-models --lib`
  is 1712 passed / 23 failed BOTH before and after -- the two failure sets are
  identical name for name, measured by reverting exactly the four changed files
  in the worktree and rerunning the same suite, so none of those 23 is this
  lane's.

  One thing the sweep turned up underneath, recorded with the repair it
  FALSIFIED rather than with a guess. The operator penalties' raw Frobenius
  norms on the same cell, as the length scale shrinks:

  ```
  length_scale   mass      tension     stiffness    tension max|entry|
    1.64e-1      3.00e0    2.71e-8     1.72e4       6.72e-1
    2.05e-2      3.00e0    3.30e-97    7.03e7       6.00e-1
    1.03e-2      3.00e0    1.00e0*     1.12e9       3.26e-203
  ```

  `*` at `1.03e-2` the normalizer declines to divide (its `all(|v| <= 1e-12)`
  branch), so the scale reads `1.0` and the matrix ships un-normalized with
  entries at `3.26e-203`. Either way the tension operator is numerically
  annihilated and then carried as an ACTIVE penalty with its own smoothing
  parameter -- which is where the certified nullity of 2 at that scale comes
  from -- while mass and stiffness saturate to the same projector, which is the
  "exact redundancy" this issue was built on. The obvious repair (drop a
  candidate whose raw energy is under `EPSILON x` the strongest sibling on its
  block) reds
  `scale_contract::tests::every_wrapper_preserves_its_declared_inner_abscissa_pullback_2315`
  (5 active penalties -> 3), and correctly: operators of derivative order `q`
  carry dimensions `[f/x^q]^2`, so their raw energies move by `factor^(-2q)`
  under a rescaling of the abscissa and a cross-order ratio is not a
  scale-invariant quantity at all. Rule withdrawn. What is left is a magnitude
  question inside the operator construction (`1/ls = 97.4` and
  `97.4^-48 ~ 1e-95` is the shape of a `kappa`-power prefactor underflowing in
  the closed-form branch), and it belongs to that subsystem rather than to the
  curvature certificate.

  That premise is what this issue ran on for its whole life, and the sweep that
  killed it is `examples/probe2676_penalty_map_defect`: the
  `geo_disease_*_matern` redundancy is a small-length-scale LIMIT of two
  genuinely different operators — `delta = 2.079e-15` below `4e-2`, `1.874e-5` at
  the cold `Auto` geometry, `3.396e-1` at the geometry the fit settles on. The
  end-to-end acceptance is re-derived accordingly: one arm finds a geometry where
  the premise is true (by measurement, not by a pinned constant) and gates the
  deflation there; the other pins the honest fact about the `Auto` geometry — the
  fit certifies and NOTHING is deflated — so the false premise cannot return by
  inheritance.

- **The SAE inner solve had no mover for the block its own convergence measure
  removes, so it declared a fixed point while holding 559 stall-resolutions of
  objective decrease (#2762).** The chart-gauge orbit is an exact first-order
  symmetry of the RECONSTRUCTION and not of the penalized objective — the ARD
  prior on `t` and the smoothness prior on `β` are written on the chart
  coordinates — so the data-fit Hessian contributes nothing along it and the only
  curvature there is the priors'. A live gradient on near-zero curvature needs a
  LONG step, and every globalization in this solver is a step-SHORTENING device:
  Armijo backtracks, the LM gain ratio grows the ridge, the terminal polish's
  damping ladder suppresses exactly the near-null modes. The orbit component of
  the residual was therefore the one part of `g` no mover reduced — and
  `quotient_residual_norm_sq` removes that same span from the convergence
  measure, so it was not reported either.

  Measured at the `zz2015_tiny_inner_crawl_terminates` refusal: `‖g‖ = 2.075e-1`
  of which `‖Π∥gauge g‖ = 2.016e-1` — 94% of the residual ENERGY inside a
  4-dimensional span, with `maxᵢ |gᵀvᵢ| = 1.535e-1` against a `1.782e-3`
  tolerance (86x over the precondition the accept path assumes and never
  checked). The discriminating control:

  | direction | best objective drop | at α |
  |---|---|---|
  | steepest descent `−g/‖g‖` | `1.879e-4` | `1e-3` |
  | the removed span | `1.090e-1` | `1.0` |

  against a material floor of `1.949e-4`, with `fd/analytic = 0.9998913` — so the
  assembled gradient IS the gradient of the scalar the line search descends, and
  this was never a desync. Steepest descent lands BELOW the floor (the stall
  detector was right); the removed span buys 580x more at a 1000x longer step,
  because the 6% transverse component of `−g` is stiff enough to cap the ambient
  line search three decades early.

  The fix is plain block-coordinate descent on the objective's own parameter
  space, not a new model: `descend_gauge_orbit` minimizes
  `penalized_objective_total` over exactly the span `gauge_quotient_basis`
  removes, and the Newton/MM movers keep the transverse block where they are
  well-conditioned. Both blocks descend the same scalar the inner Armijo descends
  and the KKT gradient differentiates, so the composition is monotone and a joint
  fixed point is stationary in both. No estimand moves; the gauge coordinate
  stops being arbitrary and starts being chosen by the objective, and at a state
  this converges on `Π∥gauge g ≈ 0`, so the precondition the quotient measure's
  removal assumes holds BY CONSTRUCTION rather than by assertion.

  Every bound is derived: the sweep's far end is `inner_iterate_scale` (the same
  trust radius the Newton step already clips against), its near end is
  `material_floor / ‖Π_V g‖` (below which the first-order model itself predicts
  less than the objective's resolution — a proof the sweep is complete, not a cap
  on it), and the bracket is golden-sectioned to `√ε` relative width, the
  information bound for locating a smooth minimum from f64 values alone. A round
  commits only a decrease clearing the same material floor the Armijo and
  proximal gates use, so a commit is never the ε-harvest that makes an inner map
  non-idempotent.

  The block is consulted at all THREE fixed-point claims — the joint fit's
  no-strict-decrease exit, its objective-stall shortcut (whose own comment
  already said it fires "on the gauge-orbit crawl … immediately", naming the
  mechanism and treating it as a reason to stop), and the refine loop's stall
  over whole rounds — each armed once per plateau on the doctrine this codebase
  already uses for the terminal Newton polish. End to end on the repro:
  `‖g‖ 2.075e-1 → 8.389e-2`, gate `27.67x → 16.42x` of tolerance,
  `‖Π∥gauge g‖ 2.016e-1 → 7.859e-2`, unspent decrease in the removed span
  `1.090e-1 → 6.43e-3`, objective `1.949279e4 → 1.948573e4`.

- **The smooth-term LR reference replayed `λ̂`'s selection on a grid `60×`
  coarser than the selection it was replaying, and the law it produced was 23%
  short exactly where `α = 0.05` is read (#2672).** The replay draws the tested
  block, minimises the same REML criterion the outer search minimises, and reads
  `W` at the `t` it picked. `SMOOTH_LR_SELECTION_GRID_BUDGET` is a TOTAL — 441
  points however many scales the term has — so a default double-penalty `s(z)`
  (`m = 2`, the shape every fixture on this issue fits) got 21 points per axis
  over a window the `ρ` box opens to 60 wide: a spacing of `3.0` in `ln λ`,
  against the `0.05` the one-dimensional lane next door commits to, and against
  a CONTINUUM for the fit whose choice this is the reference for.

  A grid that cannot find the criterion's minimum returns a law that is selected
  LESS than the statistic it is the reference for, and the error is one-signed:
  it under-disperses, the upper tail is too thin, the test over-rejects.
  Measured on a whitened bending+ridge pair at the `ρ̂` separations a null-true
  smooth actually reaches, 2048 draws:

  ```text
  arm             grid  per_axis  spacing   E[W(t̂)]      sd      q95    wall
  grid only        441     21      3.000     2.1334   2.9094   7.1898   0.06s
  grid only       1681     41      1.500     2.4212   3.3266   9.4427   0.21s
  grid only       6561     81      0.750     2.4928   3.3656   9.2994   0.83s
  grid only      25921    161      0.375     2.5192   3.3783   9.3892   3.30s
  grid + descent   441     21      3.000     2.5258   3.3779   9.3278   0.20s
  grid + descent   121     11      6.000     2.5258   3.3779   9.3278   0.18s
  ```

  `15%` short in the mean, `23%` short at `q95`. That is the residual this issue
  was left holding after its four reference defects closed — pooled
  `size@.05 = 0.0564` on the light grid and `0.0669` on the small-n one, both
  anti-conservative, both inside their bands only because the bands are wide.
  It is also `n`-INDEPENDENT, which is what the `..._versus_n` sweep's
  flattening at `0.065` for `n ≥ 200` is once the small-n quadratic-expansion
  error has decayed out of it.

  **A bigger grid is not the fix.** `25921` points costs `3.3 s` per term and is
  still at spacing `0.375`. The grid is the wrong instrument: it is a BRACKET,
  and a selection is a DESCENT. Each draw now descends the criterion from its
  own bracket node by a compass search that halves its step whenever a sweep
  fails, to the same `0.05` floor the diagonal lane uses. That reaches the `161²`
  law to `0.3%` from a bracket of 121 points — by making the grid SMALLER. The
  bracket stays at 441 because its only remaining job is not to miss a basin.

  **And the descent needed an evaluator the eigen route cannot be.**
  `SelectionGeometry::at` returns the full eigensystem, which is right when 2048
  draws share a point and wrong when one draw chose it. `SelectionFactor` prices
  a point from two triangular factorizations of `r × r` objects — exactly, not
  approximately, the same criterion and statistic:

  ```text
  C(t) = UᵀT(t)U = RᵀR,  R = qr(M(t)),  D = (I + C)⁻¹C,  v = Uᵀu
  criterion = vᵀDv + log|I + C| − log|C|
  statistic = ‖u‖² − ‖Dv‖²
  ```

  because `D`'s eigenvalues are the shares `f = e/(1 + e)` and
  `w = 2f̄ − f̄² = 1 − f²`, and a direction outside `range(T)` has `f = 0`, so it
  drops out of the first and carries its whole square into the second — which is
  what the eigen route's `log(1 + 0) = 0` and `w = 1` say. The #2644 conditioning
  split is kept rather than lost: `log|C|` comes from the triangular factor of
  the SCALED ROOTS (`κ(C)` reaches `e^60` on a null-true double penalty, where an
  assembled Cholesky has no small pivots left), while `log|I + C|` and `D` come
  from the assembled `I + C`, which is benign — an absolute `ε‖C‖` in a mode near
  zero moves `log(1 + e)` and `e/(1 + e)` by that much and no more. The whole
  replay goes `0.06 s → 0.20 s` per term for it, against `3.3 s` for the grid
  that would otherwise be needed.

  Three contracts carry it, none of them the probe that found it:
  `the_descent_reaches_a_grid_it_cannot_afford_2672` scores the shipped replay
  against the `161²` grid on mean AND `q95` at 3%, stated as a CONTRAST so it
  cannot pass by both arms drifting — the bracket alone must still miss `q95` by
  more than 10%, and must miss DOWNWARD, because a coarser selection cannot
  select more; `the_two_evaluators_price_a_point_identically_2672` pins the two
  routes against each other on a DENSE information with two dense components at
  separations to 40 (the descent compares its trials against a baseline the
  bracket produced, so a gap between the routes is a search descending one
  function while reporting another's value); and
  `the_multiscale_replay_is_bit_identical_across_generations` is #1017 for the
  lane that now searches rather than enumerates.

  Verified on one 4-core box, `--test-threads=1`:

  ```text
                                                        at main        after
  exhaustive_null_simulation_size_grid              pooled .0564   ok   pooled .0542
  null_simulation_size_is_calibrated_small_n        pooled .0669   ok   pooled .0638
  poisson_smooth_lr_is_bartlett_corrected_...            ok        ok
  the_two_routes_to_the_null_spectrum_agree_on_real_fits ok        ok
  the_two_moment_summary_is_exact_when_shrunk_...        ok        ok
  per_term_edf_plus_unpenalized_columns_equals_edf_total ok        ok
  the_null_spectrum_reaches_the_reference_with_a_param.. ok        ok
  cargo test -p gam-models --lib selection_replay lr_null      23 passed
  ```

  (these fixtures are deterministic, so the before/after comparison is exact
  rather than distributional.)

  **What is left, and it is one cell family.** On the small-n grid the other six
  cells average `0.046` against nominal `0.05`; `bernoulli/logit, k = 12` sits at
  `0.119` at both `n = 30` and `n = 50` and carries the pooled figure by itself.
  A Gaussian arm — added here as
  `gaussian_null_size_is_calibrated_where_the_expansion_is_exact_2672`, because
  the residual's two readings (a wrong reference versus the QUADRATIC EXPANSION
  the reference and the Lawley factor both are) are separated by a family whose
  likelihood IS that quadratic — reads `0.0750` pooled at `n ∈ {30, 50}` and
  `0.0588` at `n ∈ {100, 200}` (pooled s.e. `0.0077`), with the Lawley factor
  inert in every cell.

  That decay is the PROFILED SCALE, and it is a defect of its own rather than
  more of this one. `σ` is estimated from the same data, so
  `W = 2(ℓ_full − ℓ_null) ≈ Q/(V/ν)` with `V ~ χ²_ν`, `ν = n − edf_total`, while
  the reference scores `Q` alone. Scored on one set of fits, an `F` reference
  removes `0.135 → 0.100` of it at `n = 30` and `0.120 → 0.115` at `n = 200` —
  the right size and the right decay — and `mean(W)/E[W(λ̂)]` runs `1.34` at
  `n = 30` down to `1.005` at `n = 200` against the `n/(ν−2) = 1.18` that
  mechanism predicts. It applies to every scale-ESTIMATED family and to none of
  the fixed-scale ones this issue's grid is built from, so it is separable, and
  `zz_measure_gaussian_reference_against_the_profiled_scale_2672` is the
  measurement it starts from.

- **The conditional-transformation-normal likelihood renormalized every row by
  the standard-normal mass between two FITTED endpoints, and that is what left
  the fit with no mode to find (#2600).** The row density was
  `φ(h(y)) · h'(y) / [Φ(h(y_hi)) − Φ(h(y_lo))]`: the model CONDITIONED on the
  response lying inside the fitted knot range, with both endpoints functions of
  the coefficients being estimated. That divisor removes both properties a
  most-likely-transformation model needs.

  *Concavity.* `log Z = log[Φ(u) − Φ(l)]` is concave in `(l, u)` by Prékopa (the
  Gaussian measure of a convex set is log-concave) and `(l, u)` are linear in β,
  so subtracting it turned a convex negative log-likelihood — `½Σh² − Σlog h'`,
  a quadratic plus a `−log(linear)` barrier — into a convex-plus-concave sum. At
  one feasible β on the wine fixture, Hessian by central second differences:
  truncated `λ_min = −6.365756e-1` against `λ_max = 7.418500e1`; untruncated
  `λ_min = +2.346524e-1`. That single negative eigenvalue is the whole of
  `resolvable_negative_curvature=true`, which the solver reported on every
  terminal cycle of every refusal on this issue.

  *Coercivity.* Raise the unpenalized location column to `c` and contract the
  shape to `t/c`: `h`, `h_lo` and `h_hi` move together, the conditional law
  converges to a truncated exponential in the normalized shape coordinate, and
  the `−½Σh²` that would punish `c` is divided out by `Z`. The profile
  likelihood over the location column, maximized over the shape at each `c`,
  runs `141.0858 → 141.0604164` over `c ∈ [1, ∞)` with `c·Σα → 1.2235` —
  monotone, never stationary, supremum attained only at `c = ∞`. The MLE did not
  exist, at any λ: every penalty term is `O(‖shape‖²)` on that ray and vanishes,
  so the penalty only sharpened the escape rather than causing it.

  This is what five refuted hypotheses on that issue were all symptoms of — the
  strict-interior dead band, the missing box-KKT repair, the face that would not
  release, the Moré–Sorensen hard-case fill, and trust-region growth. The solver
  was correctly refusing a problem with no solution.

  The fitted density is now `φ(h) · h'` with no renormalization
  (Hothorn–Möst–Bühlmann 2018), and with it: the model's CDF is `F(y|x) = Φ(h)`,
  so the PIT is `Φ(h)` and the calibrated score is `h`; the
  `OutsideCertifiedDomain` refusal is gone, because it existed only to stop the
  conditional PIT fabricating a clamped `0`/`1` off the fitted range, and a
  held-out response beyond the training range is now predicted rather than
  refused; and `score_influence_jacobian` loses its endpoint-mass denominator,
  its three `φ/D` coefficients and its `1/φ(z)` inversion, because `z = h`
  identically on the interior.

  Both transformation-normal quality arms produced no fit at all before
  (`generated=2, screened=2, exact_validated=2, solver_started=0`) and now pass:
  held-out PIT `KS=0.1597` against a `0.2517` bar, and wine-price normality
  `W_gam=0.9533` against a `0.95` floor and `W_boxcox−0.02 = 0.9460`
  match-or-beat. Two pins carry the properties rather than the fixtures —
  `ctn_penalized_objective_is_coercive_in_the_location_column_2600` walks the
  escape ray to the family's own `|h|` domain bound and requires divergence, not
  merely monotone rise (a monotone sequence can be bounded, and bounded IS the
  defect), and `ctn_observed_information_is_positive_semidefinite_2600`
  eigendecomposes the exact SCOP information at nine feasible points.

- **The constrained posterior's retention ladder searched a real number where
  the object being chosen is a set of rows, and floating point stopped it
  descending (#2714).** `assemble_retained_face` keeps a constraint row iff
  `pivot > (k+1)·ε·diagonal/d`, and the ladder named the next face by the floor
  at which its worst-conditioned accepted row drops, `d_r =
  (k+1)·ε·diagonal_r/pivot_r`, on the argument that the retention test then
  reads `pivot > pivot`. It does not. Both sides are ROUNDED quotients: the step
  divides by `pivot`, the rebuilt floor divides that quotient back into the same
  numerator, and the round trip lands strictly below `pivot` for a measurable
  fraction of `(k, diagonal, pivot)` triples. The aimed-at row is then retained,
  the rebuilt face is **bit-identical**, and the next step is recomputed from
  unchanged inputs to the value the floor already holds — so
  `assert!(next < demanded_accuracy)` fires and a library panics.

  `the_floor_round_trip_retains_the_row_it_was_aimed_at_2714` measures the round
  trip on its own, over the magnitudes a penalized posterior produces, and
  asserts the sharper fact: **every retention is a stall**, because a
  bit-identical face recomputes a bit-identical step. There is no rounding of
  the quotient the other way that repairs this — the retention test *is* the
  definition of the face, so only the test can decide the face.

  The walk now carries the face. A rejected face names its
  `least_independent_direction` — the accepted row with the smallest
  `pivot/diagonal`, i.e. the smallest squared sine to the span of the rows
  before it in the `Σ` metric — and that row is excluded BY INDEX before the
  face is rebuilt at an unchanged floor. Termination becomes structural: the
  excluded set only grows, never re-adds, and is a subset of the candidates; the
  first unexcluded row always clears the floor; and the last face is a single
  row whose `1×1` lift is exact. No floating-point comparison is inverted
  anywhere on that argument.

  What leaves is a **direction**, not a row, and that distinction is a
  correctness requirement rather than a nicety. A row anti-parallel to an
  accepted one is refused as a direction and keeps its wall as that row's upper
  limit (#2523), so a two-sided bound reaches the moments as ONE retained row
  carrying a finite `upper`. Dropping that row while leaving its partner in the
  pool would let the next pass accept the partner in its place — with a full
  half-line, because the row that carried the fold is gone — turning a two-sided
  bound into a one-sided one, and on the wrong side: the walk is ordered by
  ascending slack, so the row that leaves is the tighter wall.
  `record_opposed_face_limit` therefore reports which accepted position it
  folded into, the face carries those partners with its least independent row,
  and `dropping_a_direction_takes_its_opposite_face_with_it_2714` asserts the
  invariant on a system where every direction is two-sided: no retained row may
  report an infinite upper limit, and no retained row may be a far wall.

  Excluding at the unchanged floor is also strictly less lossy than the old
  step, which tightened the floor for every surviving row as a side effect of
  dropping one: every face the floor ladder could reach is still reachable
  (exclude precisely the rows that floor rejected, and each survivor clears the
  looser floor a fortiori), while faces only a tighter floor would have
  destroyed are kept. `the_walk_returns_the_largest_admissible_face_2714`
  accordingly grades against brute force over EVERY exclusion set — the full
  family the walk searches — rather than the floor-indexed subfamily the
  previous oracle swept.

  The module doc also stops conflating the two cuts. The PER-ROW retention
  floor is nearly free — a row it refuses is within `θ < 5e-7` radians of the
  span of the accepted ones, and the accepted row it is parallel to has the
  smaller slack, so it imposes nothing new. The WHOLE-FACE check is not, and no
  `O(θ)` argument covers it: it fires exactly when every row cleared the floor
  and the face is still worse conditioned than any of its rows, and the
  direction it then drops can sit at `pivot/diagonal = 1e-3`, i.e. `θ ≈ 0.03`
  radians. That is a real constraint being dropped. The trade is still the
  honest one — the moments are computed for a BOX, a face keeping mutually
  dependent rows cuts the same region along diagonals, and the lift cannot be
  formed at all at a numerically singular `W` — so the alternatives are a
  subset-truncated posterior or none, not a subset-truncated posterior or an
  exact one.

  The walk also reports itself now, on both outcomes. A correction that had to
  drop rows says so in one `log::info!` line — which rows survived is a fact
  about the ANSWER, not about the solve, since the reported truncation is then
  carried by a subset of the constraints the user wrote — and a terminal refusal
  prints the last refused face's `W` at full precision, because the walk's
  decisions are a function of `W` alone and reproducing one otherwise costs a
  three-quarter-hour fit. The face dump fires only on the terminal paths:
  dropping rows is the walk working, and a `q × q` matrix per rung would be a
  megabyte of warnings for a correction that then succeeds.

  Reached by the #2714 witness because the fix for its titled defect let the fit
  get as far as final posterior assembly, where a monotonicity guard imposed at
  every observed exit time puts far more constraint rows than the time block has
  coefficients: `W = A Σ Aᵀ` is then structurally rank-deficient and the walk is
  the only thing standing between the fit and a face it can lift. That geometry
  now has a unit fixture of its own —
  `a_rank_deficient_constraint_system_still_yields_a_liftable_face_2714`, 40
  Vandermonde rows at closely spaced nodes on 5 columns, which is the shape with
  the data removed. It panics on the old ladder at pass 26 and returns a
  5-row face whose lift misses its identity by `1e-6` on the new walk.

- **The Jeffreys/Firth span is MEASURED, not derived from a penalty's kernel
  (#2612).** Two derived spans have shipped here and both answer structurally
  what has to be measured. The FULL identifiable span says *the model bounds
  nothing*, justified by "the Jeffreys score is `O(1)` against the data's `O(n)`
  Fisher information" — a premise that fails on a quasi-separated softmax, where
  `W = diag(p) − ppᵀ ≈ 0.005` per row, so the term acts at full strength on
  directions the penalty bounds up to `2298` (measured cost: mean argmax
  probability `0.828` against held-out accuracy `0.965`). `ker(S_λ)` says *the
  model bounds `range(S)`*, justified by `(H + S_λ)v = Hv + λSv` — true for any
  `λ > 0` and false in MAGNITUDE when `λ` rails at its floor. Measured at the
  penguins stride-4 unbiased mode:

  ```text
    ker(S_λ):                  2 of 74 directions, λ_min(H + S_λ) = 1.9e-3
    whole identifiable span:                       λ_min(H + S_λ) = 5.1e-5
  ```

  The worst-bounded direction — five orders below one observation-equivalent —
  is **not** in the kernel. It is a `range(S)` direction whose selected `λ`
  railed at `MULTINOMIAL_FORMULA_PRIOR_PSEUDO_OBS = 8e-4` pseudo-observations,
  and `8e-4` pseudo-observations is not a prior. Left unarmed the coefficient
  runs to `|η|∞ ≈ 45`, and the posterior-mean predictive refuses to publish
  because the posterior at that width is not describable by either Laplace
  expansion.

  The span is now `{v : vᵀ(H + S_λ)v < CONDITIONING_GATE_ABSOLUTE}` — the same
  one-observation-equivalent criterion that already decides the term's WEIGHT,
  now also deciding its SUPPORT. It contains the separating members of `ker(S_λ)`
  and excludes its well-determined ones, so it is strictly better than either
  endpoint on both sides. Reading `H + S_λ`'s deficient subspace was previously
  rejected because that matrix moves with `β` and `ρ` while every `Φ` derivative
  formula holds `Z_J` fixed; this does not read it live — it is measured once, at
  the unbiased probe's certified mode and its selected `λ`, and frozen for the
  armed refit.

  **The arming VERDICT is a different question and keeps its own answer.**
  Handing the measured set to the verdict as well broke the three-arm oracle from
  both sides — a genuinely separated design with `S_λ = 0` DISARMED, and widening
  the metric's scale until it armed again took the calibration fixture to
  `−0.0525` against a `0.05` bar. Threading that with a scale constant would be
  choosing an estimand on a curve. So *"does this model need a proper prior?"*
  stays a statement about STRUCTURE (`ker(S_λ)` and the gate's predicate on it,
  byte-unchanged), and *"where does that prior belong?"* is the statement about
  the fit's ARITHMETIC above.

  **The threshold is taken in the CLR metric, because a threshold is a statement
  about coordinates and the multinomial's are a gauge choice.** Relabelling
  classes acts on `θ` by a non-orthogonal contrast change, so `H + S_λ`
  transforms by congruence and its eigenvalues move; a kernel is
  congruence-invariant and never had this problem. Measured cost of ignoring it:
  `multinomial_fit_is_invariant_to_reference_class_1587` saw predicted-probability
  drift `4.093e-3` across three labelings of one dataset against a `1e-3` bar,
  with refit noise exactly `0`. Generalized eigenvalues against `G = (M/K) ⊗ I_P`
  — the same `M` the reference-symmetric penalty is built from — are gauge
  invariant, and `G`'s scale is derived rather than chosen: one observation's
  Fisher block in the ALR active frame is `W_ab = p_a(δ_ab − p_b)`, which at the
  most-informative point `p_c = 1/K` is exactly `(1/K)·M_ab`, so `M/K` IS one
  maximally-informative observation's curvature per unit design.

- **A cached inner mode was identified by the penalty's state, not the
  objective's, so the coefficient-mode continuation's corrector was disabled by
  its own refinement (#2612).** `InnerPenaltyState` carried the per-block and
  joint `log λ` and called itself "the complete smoothing state an inner Laplace
  mode is a function of", with the reuse contract written on it: *a cached mode
  is reusable only when the penalized objective it minimises is the identical
  function*. The inner coefficient objective is
  `−ℓ(β) + ½βᵀS_λβ − τ·Φ(β)`, and `τ` — the family's Jeffreys/Firth augmentation
  strength — was not in the state, so two different objectives at the same ρ
  compared equal.

  That is not a corner case for the one path that varies `τ`: #2366's
  `coefficient_mode_homotopy_member` DEFINES the armed coefficient mode as the
  endpoint of a continuation in `τ` **at fixed ρ**, so on that path the ρ half is
  constant by construction and the key matched at every waypoint. The only thing
  left between the corrector and a no-op was the fresh curvature certificate
  refusing a non-PSD incoming point — and the finer the discretization, the less
  the mode moves per waypoint and the more often that certificate passes.
  Measured on the penguins stride-4 armed refit, 8-step sweep, `λ_min` at each
  incoming mode: `−7.9e-2, −8.7e-6, −7.3e-6, −4.8e-6, −2.5e-6, −7.4e-7, −5.2e-8,
  +2.8e-8` — the last one reused, so the sweep's ENDPOINT was the `τ = 0.875`
  mode relabelled as the `τ = 1` mode, printing a bit-identical log-likelihood,
  penalty and cycle count. Refining the discretization, which is the
  continuation's entire convergence mechanism, is what disabled its corrector.
  This is #2615 one level up: the same key comparing equal at every `τ` because
  it is missing a coordinate, rather than at every ρ because it was empty. The
  state is now `InnerObjectiveState`, built from the family so no production site
  can supply the wrong strength, and the persistent-warm-start key carries it
  too.

- **The continuation ladder's endpoint sequence is MODE-VALUED, so the dyadic
  contraction premise could not read it (#2612).** Every sweep's last waypoint
  corrects at the target objective to the inner solver's own KKT tolerance, so
  each endpoint is an *exact* mode of the *same* function — refining does not
  shrink an error, it changes which mode the path arrives at. The measured trail,
  same fixture:

  ```text
    steps  1 → 2    endpoint discrepancy 1.521120e0
    steps  2 → 4                         3.328619e-5
    steps  4 → 8                         1.695145e0
    steps  8 → 16                        5.413481e-5
  ```

  Two values four orders apart, alternating — `O(1)` between different modes and
  `O(5e-5)` (the corrector's own reproducibility) between two arrivals at the
  same one. There is no rate to observe. `d_k ≤ ½·d_{k−1}` therefore fired at the
  first refinement that actually tracked the branch, using as its baseline the
  accidental agreement of two coarse sweeps that had both jumped to the same
  wrong mode. Three things follow and all three are now fixed:

  1. the contraction ratio is reported evidence, not a verdict;
  2. one agreement is not a limit — the `2 → 4` agreement above is a full
     agreement on a mode the 8-step sweep leaves — so certification needs
     consecutive agreements;
  3. the yardstick could not be `options.inner_tol`. That is a KKT-residual
     tolerance and the discrepancy is a relative sup-norm over linear
     predictors; on this fixture two sweeps reaching the SAME mode differ by
     `3.3e-5` and `5.4e-5` against a `1e-5` bar, so "the same mode, twice" was
     not certifiable at any depth. The reason is physical: the armed mode is
     nearly flat in one direction (`λ_min ≈ 4.7e-7`), so `β̂` is poorly determined
     by a residual while the criterion built from it is well determined — the
     same two endpoints agree to `5.0e-7` in the criterion. The bar is now the
     outer solver's own relative-cost resolution, in the criterion's units, which
     is also the only quantity the seed exists to make well defined.

  #2661's requirement — accepting arbitrarily slow progress makes the loop
  operationally unbounded, since each refinement doubles the corrector count — is
  preserved and is now bounded as the resource it is: **the seed may not cost
  more correctors than the outer search it seeds is budgeted for**, i.e.
  `2^{D+1} ≤ outer_max_iter`. A ladder that exhausts it refuses with the full
  trail rather than with two numbers out of a sequence.

- **A coefficient-objective continuation that cannot certify now DECLINES
  instead of killing the fit (#2612).** Its sibling `anchored_continuation_seed`
  has carried that contract since #2366 — "the production caller logs a refusal
  and keeps its existing seed, so declining a continuation still never turns a
  fit that works today into a failure" — and the homotopy call site was the one
  place that read the same kind of refusal as fatal. Refusing the whole fit does
  not make `V(ρ)` well defined; it only denies the caller the answer the
  pre-#2366 seed would have produced. What a decline costs is logged rather than
  hidden: the mode is then selected by the caller's coefficients, so `θ̂` is a
  functional of the seed for that fit.

- **What the follow-up-varying slope still cannot do, measured rather than
  guessed (#2765 / #2767).** With the one-channel pullback repaired, the
  acceptance fixture's outer solve goes from *zero* iterations (its first line
  search died at `StepSizeTooSmall after 50 attempt(s)`) to **1500+ outer
  evaluations across five BFGS multi-starts**, steps accepted via Strong Wolfe,
  descending `2148.09 → 2134.79`. It still does not certify, and the reason is
  now bounded on three sides:

  1. **Every criterion atom except `½ log|H|` is exact.** At the fixture's own
     shape the θ-wide audit gives `fixed_beta` to six digits and `logdet_s` to
     seven on all five coordinates; `logdet_h` disagrees on all five.
  2. **It is the mode-response half**, and on the ρ block that is proved by a
     bound with no oracle in it (`½ tr(K·λ_kS_k) ∈ [0, rank/2]`), not inferred.
  3. **It is not the follow-up axis.** The `logslope_time_k`-unset control
     reproduces the same `logdet_h` disagreement bit for bit, so it predates
     this issue — it is the `#979`/`#1040` lane, where
     `survival_marginal_slope_outer_gradient_fd_1040.rs` has recorded a wrong
     analytic ψ gradient since `#2461`.

  Every object that atom is built from is now differenced against its own
  Ridders-certified oracle and passes: `D_β H[δ]` (five gates, both slope
  frames, block-confined directions, plus an oracle-free constant-margin
  reduction), `D²_β H[u,v]` (three gates), `D_β H_Φ[δ]` (one gate in
  `gam-custom-family`, on a family whose Jeffreys information genuinely depends
  on β), and the ψ coefficient mode response `dβ̂/dψ` itself (`5.0e-8` relative
  against its finite difference, from `3.3e-2` before the repair). A binomial
  `matern(x1,x2)` control through the shipped GLM assembly — `c`-nontrivial, so
  the same mode-response term is live — agrees to `1e-6…1e-11` on every
  coordinate, which puts the residual inside the custom-family joint lane rather
  than in machinery every penalized non-Gaussian fit uses.

  The fixture also shows what the outer search now runs into instead: inner
  solves that exit at `residual ≈ 1.9e3` with the trust radius collapsed to
  `8e-12`, and an outer cost-stall guard that measures the criterion's own
  evaluation noise at `σ ≈ 1.0` nat. That is the `#979` inner-solve stall, not a
  gradient defect, and it is what the acceptance fixture is waiting on.

- **`D_β H` pulled the row Hessian back through ONE slope channel, so the outer
  criterion's whole mode-response term was the derivative of a different model
  (#2765 / #2767).** `add_pullback_primary_hessian` — the pullback that
  `RowKernel::add_pullback_hessian` routes through — was still written for the
  four-primary static frame: `h[[3,3]]` against a single `coefficient_design()`
  for the g–g block, `h[[0,3]]+h[[1,3]]` for m–g, `h[[a,3]]` for t–g. On a
  follow-up-varying slope the row Hessian carries `g₀, g₁, ġ₁` at primaries 3/4/5
  against three *different* designs (`X_cov ⊗ B_entry`, `X_cov ⊗ B_exit`,
  `X_cov ⊗ B′_exit`), so all three blocks were wrong.

  **Why every existing gate passed.** The joint Hessian itself is assembled by
  `hessian_dense_override`, which does loop the channels — so `H`, the score, and
  the ψ triple were all right and all gated. This pullback has exactly one
  consumer, `row_kernel_directional_derivative` (`D_β H[δ]`), and `D_β H` has
  exactly one consumer, the outer criterion's `½ tr(K · D_β H[dβ̂/dθ])`. The
  defect was invisible to everything except the outer gradient.

  The attribution chain, because each step needed an instrument that did not
  exist:

  1. The outer-gradient FD audit graded **ψ only** — its own doc said
     "smoothing-parameter ρ coordinates are deliberately excluded". It now grades
     the whole θ vector on request (`enable_outer_gradient_fd_capture_over_theta`),
     through one extracted ladder (`difference_theta_coordinate`) so the two
     blocks cannot drift apart. That showed `fixed_beta` right to six digits and
     `logdet_s` right to seven on **every** coordinate, with `logdet_h` wrong on
     all five — twice with the wrong sign.
  2. `logdet_h` is the only atom that reads `dβ̂/dθ`. Splitting it into its
     frozen and mode-response halves for the ρ block gave a bound with no oracle
     in it: `½ tr(K·λ_k S_k)` has both factors PSD, so it lies in
     `[0, rank(S_k)/2]`. At ρ₂ the frozen half was `+0.4976` and the finite
     difference of the total was `+0.7671`; a correct mode-response half would
     have forced the frozen half to `+1.528`, outside that interval. So the
     mode-response half was the wrong one — proved, not inferred.
  3. A binomial `matern(x1,x2)` control through the shipped GLM assembly — `c`
     nontrivial, so the same mode-response term is live — agreed to `1e-6…1e-11`
     on every coordinate, which put the defect in the survival lane rather than
     in machinery every penalized non-Gaussian fit uses. (The Gaussian sibling
     gate cannot say this: under the identity link `D_β H ≡ 0`.)
  4. Four new gates difference `D_β H[δ]` against the family's own joint Hessian
     in both slope frames and along block-confined directions. The follow-up
     frame failed at the marginal↔log-slope cross entry
     (`analytic −3.740e-1` vs `fd +5.438e-1`, oracle `2.6e-9`) and the static
     frame passed, which named the frame rather than the algebra.

  Also routed rather than patched: the sparse/mixed `evaluate_blockwise_exact_newton_*`
  paths (`p ≥ 512`) scatter the log-slope block as one `f_pipi[[3,3]]` rank-1 over
  one CSR — a shape that cannot express three channel designs at all. They are a
  storage optimisation for sparse designs, not a different model, so a
  follow-up-varying slope now takes the exact dense blockwise route at every `p`.

- **The repair, measured end to end: `rel_l2` 0.3395 → 0.1042, and the optimizer
  finds the signal axis by itself (#2735).** Same data, same seed, the fixture's
  own generator at `n=3000, K=60, pc_dim=6` — a shape 17× smaller in `n` and 8×
  smaller in `K` than the one the `0.10` bar is written for:

  ```text
      route                                  η spread   REML      rel_l2   wall
      fit_term_collection_forspec (before)      0.140   1736.54   0.3395    3.3 s
      production entry + per-axis ψ (after)     2.854    819.81   0.1042   1289.5 s

      learned_eta = [+1.8891, -0.0467, +0.1071, -0.0457, -0.9387, -0.9651]
      learned_length_scale = 2.0225  (seeded at 1.0)
  ```

  The criterion falls 917 nats and the held-out error falls 3.26×, essentially
  onto the bar. The largest contrast by a wide margin lands on **axis 0** — the
  axis carrying the entire `0.4·sin(π x₀)` — while axes 4 and 5, which carry
  linear coefficients `−0.15` and `+0.10` and no non-linear content at all, are
  pushed to the far end. Nothing told it which axis mattered.

  Cost, stated rather than buried: 1289.5 s against 3.3 s. The outer problem
  went from 9 coordinates to 15 and every ψ trial rebuilds the basis.

- **The Duchon operator-penalty ψ derivative differentiated a penalty the design
  never ships — two pre-existing desyncs, both on the ISOTROPIC route (#2735).**

  Found by running the production entry, which refused with *"spatial kappa
  optimization is unavailable for one or more eligible spatial terms"* and named
  no term and no reason. Every `Ok(None)` on that path now logs which term
  declined and why; the first run with those lines said it outright — the
  producer emitted 4 active penalty blocks against the realized design's 9.

  1. **The metric.** `duchon_operator_penalty_candidates` builds its collocation
     operators with `aniso = None`, deliberately, and its own doc says why: *"the
     anisotropy lives entirely in the curvature (Primary) RKHS Gram … Keeping
     these low-order stabilizers isotropic makes their η-gradient identically
     zero."* The derivative read `spec.aniso_log_scales` instead.
  2. **The split.** When `scale_dims` is on, the value REPLACES the single
     `Σ‖∇f‖²` with `dim` per-axis `Σ(∂f/∂x_a)²` blocks — one
     `PenaltySource::OperatorRelevance { axis }` each. The derivative emitted one
     `OperatorTension` regardless, and the consumer zips positionally, so the
     tension ψ-derivative was attributed to `OperatorRelevance { 0 }` and the
     other `dim − 1` relevance blocks had no ψ-derivative at all.

  Fixing (1) simplifies the per-axis work it invalidates: with the operators
  isotropic their η-gradient is identically zero, so `∂S/∂ψ_a = (1/d)·∂S/∂log κ`
  and `∂²S/∂ψ_a² = (1/d²)·∂²S/∂log κ²`. Normalization passes that scaling through
  exactly, so the per-axis bundles are the isotropic one scaled.

  Worth recording, because it bears on a claim in the tree: penalty ARD
  (`OperatorRelevance`, documented as *"the replacement for brittle kernel-η
  optimization"*) and metric anisotropy are not substitutes. `λ_a ∫(∂f/∂x_a)²`
  controls how much the fit VARIES along an axis; it cannot add resolution the
  kernel does not have. A radial kernel with an isotropic metric cannot wiggle
  fast along `x₀` and slowly elsewhere at any `λ`. That is the same statement as
  this fixture's own reference table, where an isotropic 500-centre smoother tops
  out at `0.2290`.

  **The gate that would have caught it did not exist.** A sum-identity or
  second-vs-first check cannot see either desync — both compare the derivative
  against itself. Only differencing the SHIPPED VALUE can, which is why the
  native half (which had such a test from the start) was right from the start.

- **A metric estimated from the knot cloud cannot see which axis the response
  varies along, so freezing it there is not "standardize the geometry, then
  learn the smoothness" — it is not learning the geometry at all (#2735).**
  `spatial_term_uses_per_axis_psi` excluded every `SmoothBasisSpec::Duchon` from
  per-axis ψ enrollment, so a hybrid Duchon term's `aniso_log_scales` were set
  once by `initial_aniso_contrasts` — the per-axis spread of the knot cloud —
  and never moved again. That seed is a statement about where the inputs ARE.
  On `large_scale_reml_stress`, whose inputs are iid `N(0, I)` and whose entire
  non-linear content is `0.4·sin(π x₀)`, it is sampling noise.

  Measured on the fixture's own generator at `n=6000, K=150, pc_dim=6`, one fit
  per explicit η along the single ray `η = (c, −c/5, −c/5, −c/5, −c/5, −c/5)`:

  ```text
      η ray        REML criterion    held-out rel_l2
      sentinel           3359.16             0.3261
      c = 0.25           3221.59             0.3038
      c = 0.50           2783.65             0.2430
      c = 0.75           2164.18             0.1650
      c = 1.00           1734.85             0.1112
      c = 1.50           1563.84             0.0914
  ```

  The criterion falls **1795 nats** and the held-out error falls **3.6×**,
  crossing the `0.10` bar at this shape, along a direction the outer loop was
  structurally forbidden from taking. The criterion and the held-out error agree
  about that direction at every step, which is what makes it a defect rather
  than an objective/estimand disagreement.

  The contrasts are identifiable — they change the kernel's SHAPE, not merely
  its scale — so REML can and should estimate them, exactly as it already does
  for the anisotropic Matérn. The repair enrolls them.

- **The isotropic ψ derivative is now a CONTRACTION of the per-axis one, not a
  parallel derivation (#2735).** For a radial scalar `F(r; κ) = κ^E G(κ r)`,
  with `A = r F_r`, `B = r² F_rr − r F_r`, `σ_a = s_a/r²` and `c = E/d`:

  ```text
      ∂F/∂ψ_a       = c F + A σ_a
      ∂²F/∂ψ_a∂ψ_b  = B σ_a σ_b + c A (σ_a + σ_b) + 2 A σ_a δ_ab + c² F
  ```

  `Σ_a` of the first is `E F + r F_r`; `Σ_{a,b}` of the second is
  `E² F + (2E+1) r F_r + r² F_rr`. Those are `scaled_log_kappa_derivatives`
  verbatim. `A` and `B` are the same two combinations the isotropic helper
  already forms, and both vanish with `r`, so the per-axis jet is finite at
  collision with no `1/r` anywhere. `duchon_radial_core_psi_triplet` — the old
  single-direction bundle — is retired: keeping a second way to spell the
  isotropic contraction is the drift the split exists to prevent.

  For a block carrying `m` explicit metric weights the same algebra collapses to
  `∂B/∂ψ_a = M_a(B) + (δ/d)·B`, where `M_a` is the scale-free per-axis
  derivative. That is not an analogy to the anisotropic Matérn: the Matérn
  helpers `hessian_operator_eta_entry` / `_eta2_entry` ARE this construction at
  `δ = 0` (its kernel carries no `κ^δ` prefactor), and they are reused rather
  than re-derived. `E_F − 2m = δ` for every block the operator penalty
  assembles — `D0` (`F = φ`, `m = 0`), the `D1` gradient (`F = q`, `m = 1`), the
  `D2` diagonal (`F = q`, `m = 1`) and its mixed term (`F = t`, `m = 2`).

  Collision is handled exactly and separately, NOT through the lift: every `s_a`
  vanishes at `r = 0`, so the block's only ψ dependence is `w_axis · φ_rr(0; κ)`
  — and `φ_rr` at the origin is not a pure power of `κ`, because the
  even-dimensional log-Riesz representative carries κ-dependent finite parts.
  That is precisely why the existing code refuses the scaling shortcut there.

- **A capability predicate, so the enrollment cannot outrun the derivative
  (#2735).** `duchon_spec_supports_axis_psi` answers from the spec alone.
  It declines — leaving the term on its single isotropic ψ axis,
  bit-identically to before — the scale-free spectrum, the periodic path,
  fractional spectral powers, terms with no contrasts to learn, terms carrying a
  joint null rotation (which the per-axis consumer does not apply and the
  isotropic one does), and any spec whose ACTIVE operator penalty routes through
  the closed-form Lebesgue block, whose ψ-derivative exists only for the
  isotropic direction. Shipping one of those would mean a block whose value and
  gradient came from two different constructions. The closed-form sweep covers
  every null-space order the realized build could degrade to, because
  `duchon_effective_nullspace_order` only ever reduces and the predicate is
  asked before centers exist.

- **A fixture's entry point is part of what it claims to test (#2735).**
  `large_scale_reml_stress_main` called `fit_term_collection_forspec` — the
  fixed-geometry entry, which builds the design once and optimizes λ — while its
  header promised "the full Duchon-on-PC GAM pipeline end-to-end". Neither the
  global length scale nor the per-axis anisotropy ever moved, so the held-out
  reconstruction it scored was the best a smoother can do at one arbitrary
  length scale under a response-blind metric. It now calls
  `fit_term_collectionwith_spatial_length_scale_optimization`, the entry
  `StandardFitRequest` uses, with the pilot geometry initializer disabled so the
  measurement is of what the full-data outer solve learns rather than partly of
  a subsample, and scores `fitted.resolvedspec` — the trained spec — because
  refreezing the caller's spec would have scored the seed.

- **An escape that RETIRES a coordinate onto a rail is not the pathology the
  small cap exists for (#2612).** `OUTER_SADDLE_ESCAPE_BUDGET = 3` carried the
  premise *"a genuine saddle is cleared in one escape"*, which this lane measured
  false twice: the banded multinomial fixture descends monotonically for six
  e-folds to the wall, and penguins takes four successive escapes that each run
  to a face (`α_box = 9.39, 4.77, 9.14, 6.11`) while the criterion falls
  `2.158034 → 2.156725`.

  The distinction the count could not make: the escape direction is exactly zero
  on every railed coordinate (`judged_subspace_basis`), so the ray's box
  intersection can only be set by a FREE one, and a reseed that lands ON the face
  has therefore retired a previously-free coordinate onto a rail. There are only
  `n` coordinates to retire and the criterion strictly decreased on the way, so
  such an escape cannot be the repeating pathology; it is bounded by
  `OUTER_CERTIFY_RESUME_BUDGET` like every other reseed kind and by
  `certify_resume_made_progress`, which stops the loop the moment a resume fails
  to strictly improve. An INTERIOR escape retires nothing and can in principle
  repeat forever — and the pathology #2155/#2363 names, a bimodal inner solve
  whose warm re-descent reports a phantom improvement the cold certificate cannot
  reproduce, is exactly the case that fools the descent gate — so the small cap
  stays and now applies only to the escapes it was written for.

  The fixture is the premise itself:
  `f(ρ) = ½ρ₀² − ½Σ_{j=1..5} ε_j ρ_j²`, `ε = (5,4,3,2,1)·10⁻³`, stationary at the
  origin and indefinite in every `ρ_j`, whose `argmin` over the box is the CORNER.
  A concave quadratic on a box attains its minimum at a vertex; refusing that
  point is refusing the answer. It needs five escapes because the
  minimum-curvature eigenvector is a single axis and each expanded step retires
  one coordinate. The `#2357`/`#2155` double wells sit exactly one unit from
  their minima and the ridge fixture has one indefinite coordinate, so neither can
  see a cap of any size — this is the first fixture in that file outside the
  premise. Measured on one binary, changing only this: `12 passed; 1 failed` →
  `13 passed; 0 failed`; `gam-solve --lib -- rho_optimizer::` 363/363.

- **A negative-curvature direction has no interior minimiser, so its escape step
  could not come from the falsifiability ladder (#2612).**
  `adjudicate_negative_curvature` built ONE step ladder — `α = 1, ½, ¼, …` down to
  `α_min = sqrt(2·objective_resolution/|λ_min|)` — and used it for two different
  jobs. As a *falsifier* it is exactly right and is untouched: the smallest step
  at which the claim `½|λ_min|α²` still predicts something the criterion can
  represent is the end of the range in which the claim could be refuted. As a
  *step rule* it is wrong, because along a direction of negative curvature

  ```text
      V(ρ + αv) − V(ρ) ≈ α(g·v) + ½λ_min α²,    λ_min < 0
  ```

  decreases without bound once the sign is chosen so the linear term is
  non-positive. A model with no interior minimiser cannot supply a step length;
  it has to come from the objective and the feasible box — the standard treatment
  of a negative-curvature direction, and exactly what a trust region does when
  its solution lands on the boundary. Capping the reseed at the falsifier's
  largest rung silently asserted the opposite: that one e-fold in log-λ is as far
  as any such descent ever runs.

  Measured on the `#2612` banded quasi-separated fixture, where the escape
  direction is `−e₁` to six digits:

  ```text
    baseline        1.786314898942e1
    ladder  α=1     1.786314894043e1
    ladder  α=½     1.786314883184e1   <- the ladder's pick, decrease 1.6e-7
    α=1             1.786314862766e1
    α=2             1.786314814710e1
    α=4             1.786314708132e1
    α=8             1.786314488769e1   <- box intersection, decrease 4.1e-6
  ```

  Monotone to the wall, and the wall step is worth **26×** the ladder's. The BFGS
  resume seeded at the ladder's point made *no* progress (reseed and next refused
  point bit-identical), so the escape was the only thing moving ρ — one e-fold per
  escape, against `OUTER_SADDLE_ESCAPE_BUDGET = 3`, on a ridge six e-folds long.
  The fit refused with `hessian_psd=NO curvature_source=terminal-analytic
  railed=[2,3,4,5]`.

  The rule: double the confirmed step while the criterion strictly improves,
  clamped to the exact box intersection along the ray. No constant — termination
  is structural, since the intersection is finite whenever the ray moves any
  bounded coordinate, doubling reaches it in `⌈log₂(α_box/α)⌉` steps, and any
  non-improving trial stops the sweep. The accepted point is the lowest measured,
  so it is never worse than the ladder's. `MAX_EXPANSIONS = 64` bounds a
  pathologically small confirmed step and is LOGGED when it binds.

  One extra evaluation re-measures the incumbent in the expansion's own instrument
  state. That is not tidiness: the same point (`sign = −1, α = 1`) evaluated in
  the falsifiability ladder and again afterwards differs by `3.1e-7` on this
  fixture — larger than the descent being adjudicated — because the profiled
  criterion carries warm-start hysteresis well above the `ε_f = 7.45e-10` the
  symmetric ladder measures on itself.

  ```text
    before   FIT FAILED after 6.5 s
    after    FIT OK in 2.9 s, ONE escape (4 doublings, α_box = 5.470155,
             "the accepted step IS it")
             acc=0.9750 logloss=0.07682 mean_argmax_p=0.9599 calib_gap=-0.01513
  ```

  bit-identical to a control with the escape budget raised to 40 (not landed).
  The escaped coordinate lands on the zero-smoothing rail, `λ = 2.0000e-4 =
  exp(−8.517193)`, where it is railed and leaves the certificate a PSD reduced
  block. `multinomial_separation_arming_2612` 3/3 in 4.6 s (was 1 FAIL);
  `gam-solve --lib -- rho_optimizer::` 362/362, and with the expansion disabled
  the two new travel assertions are the only reds.

  The `#2357`/`#2155` escape fixtures are double wells whose saddle sits *exactly
  one unit* from its minima, so nothing in that file could ever see the cap. The
  new fixtures are the shape that can: a stationary ridge with **no interior
  minimiser**, plus the guard that the expansion must not overshoot a genuine
  well, plus an end-to-end run pinned to `prefer_gradient_only` — because an ARC
  search reads the analytic Hessian and can follow negative curvature by itself,
  so letting the planner choose ARC would make a pipeline test green either way.

  Rejected: raising the escape budget (moves a constant to clear a bar and leaves
  the one-e-fold cap in place everywhere else); relaxing the curvature gate (the
  criterion's own symmetric ladder CONFIRMS the sign and resolves it against its
  Law 1 floor, so the verdict is correct and it was the response that was wrong);
  switching the outer search to ARC (`prefer_gradient_only` exists because the
  generic REML/LAML Hessian consumes the order-four family tower, and the escape
  has to work for the gradient-only plan regardless).

- **`76a520c45` withheld a deletion the geometry did not license and dropped the
  ORTHOGONALIZATION with it: the smooth-ownership hierarchy was inert for every
  dependent smooth (#2747).** `apply_global_smooth_identifiability` exists to
  enforce one invariant — the realized smooth block is orthogonal to
  `[intercept | owned linear axes | owner smooths]` — and it enforced it by
  DELETING one coefficient direction per constraint direction. `76a520c45`
  established that the deletion is free only under CONTAINMENT (the parametric
  direction inside the design's span, so the deleted function IS the parametric
  column) and withheld it otherwise. It left nothing in its place.

  The premise the deletion had always rested on —
  `smooth_requires_parametric_orthogonality`'s *"their realized column span
  contains the constant … a structural rank-1 collision"* — is measured false for
  half the class it names (`examples/probe_2747_containment_registry`,
  `‖1 − P_X 1‖/‖1‖` on the realized design against the `√ε` bar):

  ```text
  thinplate                                      9.90e-15   contained
  duchon                                         1.33e-14   contained
  matern (both policies, ν = 3/2 and 5/2)   7.8e-4 .. 8.4e-1   NOT
  curv (κ ∈ {−1,0,+1}, ℓ = 0.2 … 100)       5.1e-2 .. 9.5e-1   NOT
  ```

  and the Matérn column carries the point that decides how the gate has to be
  written: the residual falls monotonically toward the bar as the range grows
  (`8.4e-1 → 7.8e-4` over `ℓ = 0.2 → 10`), and the range is an ESTIMATED
  coordinate. Containment is a function of a fitted parameter, not a property of
  a family, so no per-family list can encode it and a delete/don't gate makes the
  model DIMENSION step by one when a fit walks its own range across a threshold.

  What that cost, measured through the shipped pipeline
  (`examples/probe_2747_parametric_orthogonality`, `‖XᵀC‖/(‖X‖‖C‖)` against the
  `1e-8` bar the same function asserts whenever a transform IS applied):

  ```text
                             before      after     deleted directions
  curv(x1,x2)               4.72e-1   1.17e-14      0 (was 1)
  x1 + curv(x1,x2)          4.89e-1   1.13e-14      0 (was 2)
  s(x1) + curv(x1,x2)       2.70e-1   7.15e-15      0
  s(x1) + tps(x1,x2)        1.64e-1   1.30e-14      0
  tps / duchon, ± x1        3.0e-14   unchanged     bit-identical
  ```

  The `s(x1) + tps` row is the one that shows the reach: thin-plate is the
  CONTAINED class and it was still `1.64e-1` against its owner, because the
  constraint block for a dependent smooth is `[1 | owner's realized columns]` and
  an owner's basis columns are contained in no other basis's span. So the
  containment gate withheld the whole block rather than one direction of it, and
  `analyze_smooth_ownership`'s hierarchy — the machinery that stops a broader
  smooth refitting structure its owner already carries (#978, #1470) — stopped
  binding for every dependent smooth in the library.

  **The fix is that the fork was never delete-or-nothing.** A deletion is
  licensed by containment; an ORTHOGONALIZATION is licensed always:

  ```text
  X̃ = X − C(CᵀWC)⁻CᵀWX        span([C | X̃]) = span([C | X])   for every X, C
  ```

  — a column operation on a block whose partner is in the model. So
  `apply_global_smooth_identifiability` now deletes where the block is wholly
  contained (bit-identical to what shipped; thin-plate and Duchon do not move)
  and PROJECTS everywhere else, keeping every coefficient direction while making
  `X̃ᵀWC = 0` exactly. The rank of `X̃` falls by `dim(col X ∩ col C)` and by
  nothing else, so the whitener drops precisely the directions the deletion is
  entitled to drop. It is also continuous in the containment residual — the
  direction the classical constraint removes has residualized norm exactly
  `sin θ = ‖1 − P_X 1‖/‖1‖` — so the two constructions AGREE at containment
  instead of meeting at a threshold.

  The fit is preserved where the theory says it must be: `y ~ curv` residual ss
  `9.926900 → 9.926900`, edf `23.455911 → 23.455912`, because with an unpenalized
  parametric block residualization is a reparametrization of the same fitted
  model. Where the partner is penalized — the `s(x1) + …` rows — the fit moves,
  which is the hierarchy binding again.

  Numerics: `G̃ = X̃ᵀWX̃` is streamed from the EXPLICIT residual rather than formed
  as `G − M N⁻ Mᵀ`, which is a difference of near-equal `O(‖G‖)` quantities
  precisely in the contained case it has to resolve. Storage: `X` and `C` are
  stacked into one `BlockDesignOperator` and the minus sign lives inside a single
  `CoefficientTransformOperator`, so a lazy design stays lazy and `C·R` is never
  materialized.

  Replay: `ParametricResidualizationChart` on `SmoothTerm`, frozen onto
  `SmoothTermSpec` beside the joint-null rotation. It carries `R`, the owner
  terms it was built against and whether the parametric block led — because `R`
  is TRAINING-ROW data and must be replayed, not re-derived (#978), while `C`
  itself is rebuilt at the new rows, which is what `C` is.

  Gated by `parametric_orthogonality_costs_no_dimension_2747` (five gates on the
  shipped pipeline, each asserting its own premise — the realized span must NOT
  contain the constant, or the deletion would be free and the test would measure
  nothing) and by three unit gates on the numerical core, whose orthogonality bar
  is DERIVED (`n·ε·‖X‖/‖X̃‖`, the floor of an `n`-term accumulation carrying the
  `1/sin θ` amplification) and asserted to sit far below the shipped `1e-8` so it
  cannot pass vacuously. The replay gate's negative control drops the chart from
  the same frozen spec and requires the design to MOVE, because a rederivation's
  output looks exactly like a design.

  **Named and deliberately NOT fixed here: a second producer.**
  `freeze_geometry_from_metadata` (`spatial_optimization.rs:4849`) freezes the κ
  optimizer's cold-build chart as `MaternIdentifiability::FrozenTransform`; that
  chart comes from `realize_single_smooth_term`, whose own comment says it "never
  runs the global ownership pass"; and
  `smooth_requires_parametric_orthogonality`'s doc excludes `FrozenTransform`
  bases on the premise that such a transform "already has the parametric
  orthogonalization composed in". For that producer the premise is false, so
  every spatial smooth whose geometry the κ optimizer froze skips the global step
  entirely, and has done since long before #2747 — which is why a Matérn fit
  still measures `4.15e-1` through `fit_from_formula` with the projection arm
  landed. `an_unfrozen_matern_smooth_is_orthogonalized_at_no_cost_2747` pins that
  the arm is not at fault: from an unfrozen spec the identical basis comes out
  orthogonal at the shipped bar with all `centers − 1` columns.

- **The κ criterion's acceptance was met and unmeasured (#2747).** Both fixtures
  are green at `ee7b9a2fa`: `profile_ci_covers_planted_curvature_across_replicates`
  covers 9/9 with 0 unresolved, κ̂ interior on 8 of 9 (the state the issue opened
  on was `railed_at_upper_bound` 9/9), mean κ̂ `+1.070` against a planted `+1.0`,
  sign right 9/9, over replicates that now span `0.5×`/`1×`/`2×` the auto range;
  `flatness_test_holds_size_across_flat_replicates` rejects 1/9 at α = 0.05 with
  0 unresolved. The estimator half of that issue was finished by
  `1f76fb35f` (one kernel, one range) → `337e6aa86` (ψ = (κ, η)) → `4b618f0ba`
  (the contrast gauge) → `76a520c45`, and the last of those had never been run
  against the fixture it was written for.

- **The λ̂-selection replay was refused on every real fit, and where it was not
  refused it minimised a criterion whose Occam term was noise (#2672).** The
  smooth-term LR reference prices `λ̂` as CHOSEN rather than as given by
  replaying the outer selection: draw the tested block, minimise the replayed
  REML criterion over `ln t`, read `W`. Four defects, each found by measuring the
  previous repair rather than reasoning about it.

  **1. The Occam term was priced by the route `#2644` had already rejected.**
  The criterion is

  ```text
  V(t) = ½ Σ_j c_j² e_j/(1 + e_j)  +  ½ [ log|I + T(t)| − log|T(t)|₊ ],
  T(t) = Σ_i t_i λ̂_i · Wᵀ S_i W
  ```

  and the bracket is its whole Occam half — the only term that stops the
  selection running to `t → 0`. Both replay lanes computed it as
  `Σ_{e > 0} log(1 + 1/e)` over the eigenvalues of the ASSEMBLED whitened sum,
  with no structural rank and no noise floor. `penalty_logdet.rs`'s own
  `SpectrumScale` says why that cannot work, and names this configuration: the
  assembled route prices `log|S_λ|₊` to `O(ε·κ)` against `O(ε·√κ)` from the
  stacked scaled roots, and `κ` goes past `1e14` when "one λ [is] at its ceiling
  beside a null-space shrinkage λ near zero" — which is what a null-true default
  `s(z)` IS, since `double_penalty` is on by default. Measured on a whitened
  `q = 9` bending+ridge pair:

  ```text
  ρ̂ = (0, 0)      offset  20.564 vs  20.564     error   0.000
  ρ̂ = (12, −12)   offset  29.813 vs  29.813     error   0.000
  ρ̂ = (18, −24)   offset  19.189 vs  53.811     error −34.623
  ρ̂ = (29, −29)   offset   8.103 vs  63.811     error −55.709
  ```

  The error does not perturb the selection, it replaces it: the modes lost are
  the ones carrying `−ln t_i`, i.e. the coercivity that makes the criterion blow
  up as `λ_i → 0`, so what is left is monotone in `ln t_i` and the replay picks a
  wall. Same mechanism `from_components` documents under #1237, from the replay's
  side. An independent numpy lab minimising a Gaussian-σ-known REML both ways
  puts the two argmins `5.7`–`13.0` nats apart under the exact criterion, always
  by railing the smaller λ down.

  **2. The geometry was refused on every real fit, silently.** `Wᵀ S W` is
  symmetric as an object and asymmetric by summation order, and
  `strict_symmetric_eigh` VALIDATES its input rather than symmetrizing it — the
  right contract, since a caller with a genuinely non-symmetric matrix has a
  defect. Every unit test in the module handed the whitening an identity
  information and a diagonal penalty, where the congruence comes out EXACTLY
  symmetric and the validator never fires. The integration sweep's first cell is
  a dense fit, and it declined.

  **3. The whitener formed `Ĩ_jj = ([H⁻¹]_jj)⁻¹ − S_jj`, which is a
  cancellation.** Two matrices whose ratio is `1/(1 − p)`, after an explicit
  inverse of an ill-conditioned block: at `p = 1 − 1e-12` — an ordinary
  heavily-shrunk direction — the difference is roundoff amplified twelve orders.
  The relative eigenvalue floor was then set by a SPURIOUS largest eigenvalue and
  discarded every direction the data could see. Four of the first twenty
  replicates of the `n = 60` fixture:

  ```text
  rep   reference mean   replayed E[W|λ̂]   q(replay)   q(reference)
    7          0.8557           0.0003        10           11
    8          0.9600           0.0000         9           11
   13          0.7060           0.0004        10           11
   15          0.7894           0.0000        10           11
  ```

  The published p-value is `p_conditional + [P̂(W_sel ≥ w) − P̂(W_cond ≥ w)]`, a
  control variate — and a control variate is only one if the subtracted term has
  the SAME law as the exactly-integrated one. On those replicates the shift was
  measured against a law with zero mass and added to the tail of a law with all
  of it. Invisible on any single fit; visible only in the size.

  The same object is available with no cancellation. With
  `A = B^{1/2} S B^{1/2} = QΛQᵀ`, `Λ = diag(p)`, `B = [H⁻¹]_jj`,

  ```text
  B^{1/2} Ĩ B^{1/2} = B^{1/2}(B⁻¹ − S)B^{1/2} = I − A = Q(I − Λ)Qᵀ,
  ```

  so `W = B^{1/2}Q(I − Λ)^{-1/2}` satisfies `WWᵀ = Ĩ⁻¹`, the only subtraction
  left is the SCALAR `1 − p` (which loses digits exactly when the direction is
  genuinely unidentified — a statement about the fit), and the retained set
  becomes `1 − p > 100·q·ε`, a meaningful criterion instead of a relative floor
  on a cancelled matrix. As a check on the construction, the whitened total
  penalty at `λ̂` comes out `(I − Λ)^{-1/2}Λ(I − Λ)^{-1/2} = diag(p/(1 − p))` —
  exactly the generalized spectrum, diagonal, for free. `lr_schur_information` is
  deleted rather than repaired: nothing needs `Ĩ` itself.

  **4. The conditional tail was resolved six orders below its own answer's
  noise.** The Imhof tolerance was derived from `FitOptions::tol`, and at the
  shipped `1e-10` that request is `~1e-10` — `gam-math`'s strict default, priced
  at 0.13–3.3 s PER P-VALUE, three or four per term.
  `null_simulation_size_is_calibrated_small_n` runs 960 of them and DID NOT
  FINISH IN 4000 s, against nextest's 600 s kill: the test this issue exists to
  un-hide had become a timeout again, by construction rather than by contention.
  The published accuracy is `quadrature + 2·se`, so resolving the conditional
  half below the selection shift's own standard error cannot improve the sum,
  while Imhof's truncation point grows like `ε^{-2/3}`. The request is now
  floored at `se`, capping the published bound at `3·se` against an irreducible
  `2·se`.

  **What replaces all four is one object.** `SelectionGeometry` carries the
  term's λ-FREE components, whitened by the cancellation-free `W`, factored into
  their own roots, plus their `ρ̂` and the structural rank of their sum; one thin
  SVD of the stacked scaled roots per grid point supplies the eigenbasis, the
  criterion's data operator, the statistic's null weights and both
  log-determinants, with `log|T|₊` over the `t`-free structural rank instead of a
  sign test on a number `1e18` below the largest. Three things fall out rather
  than being fixed separately: the one-dimensional lane stops reconstructing
  `ν_k = p_k/(1 − p_k)` from the shares (a share lives in `[0, 1]`, so a
  structural zero and `1e-17` of roundoff are one epsilon apart there — and the
  log-determinant is the one place that difference is worth `log(1 + 1e17)`, as a
  term LINEAR in `ln t`); its grid gains `ln t = 0` explicitly; and `generalized`
  is published on every lane, where the multi-scale one had returned
  `Vec::new()`.

  **And the `ln t` window is now per scale.** The outer search moved each `ρ_i`
  independently inside its box, so scale `i` reaches `[−B − ρ̂_i, B − ρ̂_i]`. The
  single common-shift window the `m`-dimensional grid used to receive is the
  INTERSECTION of those, which truncates every axis to the narrowest and is EMPTY
  as soon as one λ̂ rails — the normal state of a null-true double-penalty smooth.
  `generate_common_scale` still derives the intersection, because that lane
  genuinely moves every scale together.

  A missing replay is no longer an `Option::None` a reader has to attribute by
  elimination: `SmoothLrSelection::{Replayed, Declined}` names the step that
  refused, and that is how defect 2 was found.

  **Two tests were stating claims true only of the reference this issue
  replaced.** The Bartlett file compared `mean(W)` against `ref_df` — the
  CONDITIONAL mean `E[W | λ̂]`, which the empirical mean does not converge to
  because `λ̂` is chosen from the same data. Measured: `2.034` against `0.870`, a
  ratio of `2.34` and `4.18` standard errors, matching the `2.4–2.5` already on
  the issue for an independent harness. Against the mean of the law the p-value
  is actually read from — `E[W(λ̂)] = 1.452`, now published — it is `2.09` se. The
  bar stays at `3·se`; only the quantity moves, and it still fails by ~20 se on
  the state this issue opened at.

  The grid's per-cell band carried a fixed `+0.015` for "the second-order
  residual the correction itself leaves (`O(n⁻²)`)". Measured on the grid's
  hardest cell across `n` at 200 replicates, that residual is neither `O(n⁻²)`
  nor constant:

  ```text
  n         30      50     100     200     400
  first  0.141   0.111   0.080   0.060   0.065
  est    0.106   0.096   0.070   0.055   0.065      (MC s.e. 0.0154)
  ```

  — monotone toward nominal, inside the MC band by `n = 200`, quasi-separation
  rate `0.0` throughout. So it is the quadratic expansion's own finite-sample
  error and not a defect in the reference: a wrong reference gives an
  `n`-INDEPENDENT offset. The band now carries half of the cell's OWN first-order
  distortion instead of a constant, which states a claim about the correction
  rather than a tolerance and TIGHTENS the band from `0.075` to `0.060` wherever
  the test is in its regime.

  Two hypotheses died on data collected for something else: the estimated-λ
  lane's ρ̂-variation term is NOT a double count against the replay (dropping it
  makes every anti-conservative cell worse), and the Lawley factor's own
  magnitude does not track the residual it is meant to remove (`c ≈ 1.008`
  against a distortion of `0.056` at `n = 30`).

  Verified on one 4-core box, `--test-threads=1`:

  ```text
                                                    at main        after
  the_two_routes_..._agree_on_real_fits_2672            RED     ok    29s
  exhaustive_null_simulation_size_grid              pooled .0962  ok   191s  pooled .0564
  null_simulation_size_is_calibrated_small_n        >4000s, unfinished
                                                                  ok   358s  pooled .0669
  poisson_smooth_lr_is_bartlett_corrected_...           RED     ok    58s
  cargo test -p gam-models --lib selection_replay lr_null        20 passed
  ```

- **The `geo_disease_*_matern` / `papuan_oce*_matern` cluster refused a fit on a
  curvature the criterion itself measures with the OPPOSITE SIGN, because the
  only measurement in the room was thrown away after a boolean (#2748).**

  Ten of the eleven scenarios the last benchmark verdict lists as `errored` fail
  with one signature. Reproduced locally in 22 s through `bench/run_suite.py`:

  ```
  rho Hessian has negative curvature -6.404e-6 below the outer certificate's own
  bar 6.379e-6 ... measured here as 2.396439e-16 [analytic (Weyl, ||dH||_2); set
  by eigensolver backward error] ... the penalty map certified 0 null direction(s)
  ```

  The whole verdict rides on `intrinsic = sigma - sum_k g_k v_k^2 = -2.473e-8`,
  the only part of that eigenvalue that is a statement about the criterion, and
  it is judged against `2.4e-16` — the EIGENSOLVER's backward error.
  `curvature_resolution`'s own module doc says in bold that this number answers
  *"given this matrix, how wrong is sigma?"* and that a site asking *"how wrong
  is this matrix?"* must not be handed it. That warning was firing in production.

  **Why #2676's deflation does not fire here, measured rather than assumed.**
  Inside ONE fit:

  ```
  one_minus_cos(S_0, S_2) = 6.164524e-11   at an earlier point
  one_minus_cos(S_0, S_2) = 6.017409e-14   at the REFUSING point
  ```

  Three orders apart. The penalties do not depend on rho; they depend on psi, the
  jointly-optimised length scale. Round-off does not move three orders with psi.
  So the Matern mass and stiffness operators are genuinely DISTINCT operators
  that become proportional as the length scale collapses the kernel matrix — a
  real near-invariance, not an exact one. `PenaltyMapInvariance` certifies only
  exact ones, so it certifies nothing, and with no certified subspace every
  measured component of `||dH||_2` at that site is vacuous or absent:

  | component | value |
  |---|---|
  | eigensolver backward error | `2.396439e-16` |
  | rho-Hessian symmetrization defect | `0.000000e0` (symmetrized in place) |
  | outer-gradient re-evaluation defect | `0.000000e0` |
  | penalty-map invariance residual | unavailable (certified nullity 0) |

  **And the outer certificate had already ruled the other way on the same
  number.** Two lines above the refusal, same run:

  ```
  [CERTIFICATE] standard REML: the criterion CONTRADICTS the reported negative
  curvature. lambda_min=-6.404092e-6 on the judged sub-block, and 2 feasible
  trial(s) along its eigenvector -- both signs -- lowered the objective nowhere.
  ```

  Same matrix to six digits, same point, opposite verdicts — #2428 exactly, with
  the subsystem that actually evaluated the criterion losing.

  **The repair is a measurement, not a bar.** No floor moved, no tolerance was
  chosen. `adjudicate_negative_curvature` already evaluates the criterion on both
  sides of the point along the disputed eigenvector; that is a symmetric probe
  ladder, and `curvature_resolution`'s header already states that `eps_f` and
  `M4` "come free from any symmetric probe ladder that has already been run". It
  was being spent on one boolean.

  * `measure_symmetric_ladder` fits `N(alpha) = c*alpha^2 + (M4/12)*alpha^4` to
    the raw second-difference NUMERATOR, whose noise is step-independent — so
    plain least squares there IS the inverse-variance-weighted fit of the
    quotient, whose noise is `4 eps_f/alpha^2`. It returns the criterion's own
    curvature with a standard error, `M4` as twelve times the slope, and `eps_f`
    as the residual scatter over four.
  * The ladder is EXTENDED until it can determine that fit. The falsifiability
    ladder stops at `sqrt(2*objective_resolution/|lambda_min|)`, which for a
    small claim is `>= 1`, i.e. ONE rung — and one rung cannot fit two
    parameters. The extension halves to `alpha_end =
    sqrt(roundoff_floor/|lambda_min|)`, where the claim's own predicted numerator
    reaches the objective's arithmetic floor, plus two halvings so the plateau
    `eps_f` is read from is more than a point. Both ends are derived from the
    claim in dispute.
  * `v'Hv` and `d2/dalpha2 V(theta+alpha v)|_0` are the same number computed two
    ways, so their difference is exactly zero in exact arithmetic and, by Weyl,
    a certified LOWER BOUND on `||dH||_2`. It is carried on `OuterResult` into
    `invert_identified_rho_hessian` as a `MeasuredHessianError`, and only when
    the disputed direction lies entirely inside the rho block.

  **What it measured on the failing fixture.**

  ```
  c_criterion = +8.153228e-5 +/- 6.616477e-6   vs analytic lambda_min = -6.404082e-6
  12 rungs; measured eps_f = 3.254032e-7, M4 = 1.094027e-2
  measured ||dH||_2 from the disagreement = 8.131990e-5
  ```

  The criterion's curvature along that eigenvector is POSITIVE and thirteen times
  the magnitude the analytic Hessian claimed. `zero_bound` goes
  `2.396439e-16 -> 8.131990e-5`, the classification goes
  `["G","A","A"] -> ["Z","Z","A"]`, and the scenario mints
  (`status = ok`, 87 s).

  **The negative control fired in production, unprompted.** The same fit's
  iso-kappa joint arm adjudicated a different point and measured
  `c_criterion = -1.060226e-6` against `lambda_min = -1.060914e-6` — agreement to
  `6.9e-10`, so `||dH||_2 = 6.3e-10` and nothing widened. Where the analytic
  Hessian is right, this measures nothing.

  **Two names stopped being true and were fixed with it.** `zero_bound` used to be
  only an eigensolver backward error, so `|sigma| <= zero_bound` and "the penalty
  map's certified null" were the same population and sharing the name
  `StructuralZero` cost nothing. They are eleven orders apart now.
  `UnresolvableCurvature` is a third variant, and the three finally say three
  things: excused by STRUCTURE (exactly flat, no measurement can change it), by
  RESOLUTION (may be real, but the matrix is not known well enough for its sign
  to be a measurement), by the CHAIN RULE (`sum_k g_k v_k^2` carries no
  second-order content) — the split the #2676 thread argued for and did not
  build. And `InvertedRhoHessian::eigenvalue_backward_error_bound`, which has
  carried the MAXIMUM over several measured components since #2748's architecture
  landed, is renamed `curvature_resolution`.

  **`haberman_5yr` is NOT this and is not fixed here.** It fails
  `NOT STATIONARY (|Pg|=1.101e0 > bound=3.636e-6)` with `railed=[5]` and
  `line_search=StepSizeTooSmall after 50 attempt(s)` — an outer BFGS
  non-convergence, a separate population, exactly as #2748's body predicted.

- **"The box does not bind at its bound" is the wrong reading of #2705 group C's
  residual: the box binds exactly, and the reported coefficient is the truncated
  posterior MEAN — matching its closed form to 8 significant figures.** On the
  noise-free line `y = 2 + 5x`, `y ~ linear(x, min=0, max=1)` reports a slope of
  `0.902139` where `bounded(x, min=0, max=1)` reports `1.000000`, and three tests
  assert the reported coefficient must sit at the bound.

  The mode IS at the bound. The fit's own `deviance` is `229.6`, and
  `(5 − 1)²·XᵀX = 16·14.35 = 229.6` exactly — the residual sum of squares at
  `β = 1`. What is reported is a different estimand: for
  `X ~ N(β̂_unc, φ̂/XᵀX)` truncated to `[min, max]`, evaluated at the fit's own
  published `φ̂`,

  ```text
  bound   sd          closed form     reported        difference
  1       0.640513    0.902138628     0.902138522     1.1e-7
  2       0.480384    1.926593682     1.926593656     2.6e-8
  3       0.320256    2.951062455     2.951062437     1.7e-8
  4       0.160128    3.975531227     3.975531219     8.7e-9
  ```

  and the reported VARIANCE agrees too — `covariance_conditional[1,1] =
  9.163014e-3` against the truncated-normal variance `9.163460e-3`, inside the
  orthant cubature's own `1e-3` relative certificate. The apparent deficit is not
  a solver shortfall that happens to be the right size; it is a closed form
  evaluated correctly, and it is SPEC rule 3 — *"posterior mean must always be
  the default (never MAP)"* — working as written, as `constrained_posterior`'s
  module documentation states outright.

  What remains is a question about the ESTIMAND rather than about the active-set
  solver. `bounded()` publishes `1.000000` on the same data because its latent
  interval transform `β = min + width·σ(θ)` stretches the boundary to `θ = ±∞`,
  so ITS posterior concentrates at the bound: the two documented ways to box a
  coefficient impose different priors and therefore publish different numbers.
  Deciding which one a user asking for a box should receive is a scope call, and
  moving either number to clear the bar is the failure mode SPEC warns about — so
  nothing was moved. What landed is the part that is provable: a regression that
  pins the reported coefficient to its closed form across four bounds AND on the
  one-sided half-line (the `nonnegative()` family's `0.007857`), asserts via the
  deviance identity that the mode really is at the bound, and refuses to pass
  vacuously if the reported value ever becomes the bound itself.

  One thing the exercise corrected in the test rather than in the engine: the
  half-line reference first missed by `1.06e-5` relative, because
  `Φ̄(6.245) = 2.1e-10` formed as `1 − Φ(6.245)` in binary64 keeps only six
  significant figures. Recomputed in log space the reference and the engine agree
  to eleven. The engine was on the accurate side throughout.

- **A shape-constrained fit could not certify its own inner mode, for two
  reasons, and both were units errors rather than convergence failures (#2705
  group B).** `smooths::shape_constrained_fit_survives_its_own_inference_2601`
  refused three of four shapes with `inner status StalledAtValidMinimum`. The
  refusal named the inner status and then quoted the OUTER stationarity residual,
  because that was the only certificate it held — so the first change was to make
  it carry the inner one: the effective KKT tolerance, both certificate bounds,
  the natural gradient scale, the inner iteration count and last realized
  deviance change, and, when constraints are present, the four constraint-KKT
  channels with the one that DECIDED the max named explicitly. That measurement
  is what the rest of this entry is built on; no gate moved to produce it.

  **The certificate compared a distance against a gradient bound.**
  `constrained_stationarity_norm` returned
  `max(primal_feasibility, dual_feasibility, complementarity, stationarity)` and
  handed that scalar to `WorkingState::certifies_kkt`, whose two bounds —
  `τ·√n·√p` and `τ·(1 + ‖score‖ + ‖Sβ‖)` — are both derived FOR A GRADIENT. Only
  two of the four channels are gradient-space quantities: `primal_feasibility` is
  a Euclidean DISTANCE in coefficient space (the constraint rows are
  unit-normalized before it is measured) and `complementarity` is a gradient
  TIMES a distance. Measured at the refused iterate on `y ~ s(x, shape=convex)`,
  300 rows of clean linear data: `stationarity = 3.148471e-10` against a
  dimension bound of `6.244998e-9` — twenty times inside the certificate — while
  `primal_feasibility = 6.301146e-9` pushed the max past it by a factor of
  `1.009`. That feasibility number is itself inside
  `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL = 1e-8`, documented as the tolerance the
  active-set solver **guarantees** on the iterate it returns, in exactly that
  metric. The solver delivered its contract and the certificate refused it.

  The gradient certificate now reads `max(stationarity, dual_feasibility)`, and
  the geometric channels are certified against the contracts that define them by
  `constraint_geometry_is_certified`, which every acceptance path requires —
  including the strict one, which was the odd one out, since the soft paths
  already applied the primal-feasibility conjunct. Complementarity's bound is
  scaled by the gradient magnitude its multipliers live at; without that factor
  the same fit passes or fails under a response rescale `y → c·y`. This is not
  uniformly looser: at `τ = 1e-6` the old test admitted primal feasibility up to
  `6.2e-5`, four orders past the solver's guarantee, and the new one does not.

  **The one machinery for an exhausted objective was switched off by INACTIVE
  rows.** The remaining shape reported `last_deviance_change = 2.220446e-16` —
  exactly `f64::EPSILON`, i.e. the penalized objective had stopped moving at its
  own arithmetic resolution, leaving no line search and no gain ratio with
  anything to choose a step by — and `iterations = 300`, the full budget ground
  out in that state. The exact bare-Hessian Newton decrement and the undamped
  polish that pursues it exist for exactly that state, and were gated on
  `linear_constraints.is_none() && coefficient_lower_bounds.is_none() && …`
  while the comment above that gate stated the actual requirement: *"active
  constraints carry multipliers"*. Those are different questions, and
  `active = 0/11` is the difference. With an empty active set every multiplier is
  zero, `∇L − Aᵀλ = ∇L`, and the constrained KKT system IS the unconstrained
  stationarity system — the coefficient-space certificate is valid verbatim.
  Gating on the EXISTENCE of the constraint system denied it to every constrained
  fit sitting strictly inside its cone, which is the whole population of
  `shape=monotone_increasing` fitted to data that is already monotone.

  The predicate is now split into its structural half (`arrow_schur.is_none()`,
  which cannot change during a solve) and its geometric half
  (`inequalities_are_all_inactive`, asked per use at all three sites, because the
  active set is a property of the iterate). Because the polish takes
  UNCONSTRAINED Newton steps — exact while the active set is empty, silent about
  where they land — each candidate is checked for primal feasibility and refused
  if it would leave the cone. Refused, not projected: a projection is not a
  Newton step, and the strict-improvement guard would then be certifying a
  different point than the one it measured.

  Neither change touches an iteration budget or widens a tolerance, the two
  levers #2705 records as SPEC-forbidden. Verified: all four shapes of
  `every_shape_constraint_fits_clean_linear_data_2601` fit and honour their
  constraint, on the same runner that measured the failure.

- **A shape-constrained fit published two covariance matrices that were not
  covariances, for two different reasons, and both were refused as
  non-convergence (#2705 group A).** `misc::shape_constrained_alo_seed_validation_aborts_1191`
  died at `posterior covariance diagonal 4 is not positive and representable:
  -3.08607306376274e-15`, and the corrected covariance of the same fixture had
  earlier been measured at `-9.954853058256977e-9`. Neither number is a small
  variance; both are what is left when a subtraction has spent all of its digits.

  **The composition.** `beta_covariance_corrected` was assembled as
  `beta_covariance + smoothing_correction`, i.e.
  `(Σ − GΔGᵀ) + (Vp − Σ) = Vp − GΔGᵀ` — with the lift `G` and the removed
  variance `Δ` derived from `Σ = Vb`, the ρ̂-CONDITIONAL covariance, and then
  subtracted from `Vp = Vb + J·V_ρ·Jᵀ`, the ρ-MARGINAL one. That matrix is the
  truncation of neither covariance. Along a coordinate the constraint pins,
  `(GΔGᵀ)_ii` cancels `Σ_ii` to eleven digits, so whatever `(Vp − Σ)_ii` happens
  to be becomes the WHOLE published variance — and that increment is legitimately
  sign-indefinite: the cubature branch computes
  `φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂] − φ̂·H_opt⁻¹`, a difference of two averages which is
  positive semidefinite only as a SUM with `Vb`. The measured decomposition on
  `y ~ s(x, shape=convex)` reads `Σ_ii = 2.302618e-2` removed to
  `6.229531e-13`, with a `−3.025454e-9` smoothing increment on top.

  The right composition follows from the estimand rather than from the sign. The
  feasible set constrains `β` and says nothing about `ρ`, so the indicator
  `1_C(β)` factors straight out of the ρ-integral —
  `∫ π(β,ρ|y)·1_C(β) dρ = 1_C(β)·∫ π(β,ρ|y) dρ` — i.e. the β-marginal of the
  TRUNCATED joint posterior is exactly the truncation of the β-marginal of the
  untruncated one. So the truncation belongs on `Vp`, applied last, with its own
  lift `G_p = Vp·Aᵀ·W_p⁻¹` and its own orthant moments at `W_p = A·Vp·Aᵀ`. The
  ρ̂-conditional covariance keeps its truncation at `Σ`, which is right, because
  that estimand really is conditional on `ρ̂`.

  **The assembly.** `Σ − GΔGᵀ` has no digits left on a pinned coordinate, and `Δ`
  is a cubature result certified to `1e-3` RELATIVE, so `Δ_ii` overshooting
  `Σ_ii` by an ulp is admissible arithmetic that publishes a negative variance.
  Splitting the correction at `Δ = W − C`, with `C = Cov[u] ⪰ 0` the truncated
  constraint-normal covariance, writes the identical quantity as two Grams:

  ```text
  Σ − GΔGᵀ = (Σ − G W Gᵀ) + G C Gᵀ = P Σ Pᵀ + G C Gᵀ
           = (P L)(P L)ᵀ + (G L_C)(G L_C)ᵀ,     P = I − G A
  ```

  so every diagonal entry is a sum of squares. The cancellation does not
  disappear — it moves INSIDE `P L`, where each entry carries an absolute error
  `O(ε‖L‖)` and is then SQUARED, so a pinned coordinate's variance picks up
  `O(p ε² Σ_ii)` instead of `O(ε Σ_ii)`: sixteen orders smaller, and non-negative
  by construction rather than by luck. `P L = L − G(A L)` costs `O(p²q)`, so the
  only new `O(p³)` work is one Cholesky of a matrix the dense branch has already
  inverted.

  Three consequences landed at the sites they belong to. The dense standard-error
  gate accepts an exactly-zero diagonal **when a truncation was applied** — zero
  is the `λ → ∞` limit of the removal and is now reported cleanly instead of as a
  `±ε·Σ_ii` residue, and a strict `> 0` test would refuse the fit for producing
  the right answer; unconstrained fits keep `> 0`, which is the singular-Hessian
  catch that gate exists for. The FACTORIZED branch has no dense `Σ` to factor,
  so it keeps the subtraction and now carries the subtraction's own MEASURED
  resolution `16·ε·max(base, removed)`, reading a residue inside that band as the
  zero it approximates and refusing anything outside it with the decomposition
  attached. And a materially indefinite `C` — past the cubature's own `1e-3`
  relative certificate — is refused rather than clamped, because that is a broken
  moment computation and not a rounding question.

  Verified: `misc::shape_constrained_alo_seed_validation_aborts_1191` passes
  (all four shapes, 400 rows of `sqrt(x)`); five unit tests in `gam-solve`
  including one that reproduces the negative variance under a two-ulp cubature
  overshoot and one that asserts the Gram assembly and the subtraction agree
  entry by entry; and a new property-side regression that reads BOTH published
  matrices on all four shapes and asserts each has a non-negative spectrum to its
  own assembly resolution, refusing to pass vacuously if no shape publishes a
  corrected covariance at all.

- **The certified REML score's VALUE enclosure was a natural interval extension,
  so its overestimation was FIRST ORDER in the cell width with constant `rank`,
  and the certified search refused designs it could certify (residual of
  #2758).** `AffineRemlProfile::enclose` evaluated each mode kernel on the
  interval `λ` and accumulated. The score is
  `−0.5·(D·normalized_logdet + residual_dof·Σ_d log(R_d/dof))`, and near a REML
  optimum those two brackets CANCEL — each block's `d/dρ` is `O(rank)` while
  their sum is not. Interval addition cannot see that the two movements are the
  same quantity with opposite signs, so the extension carried `rank·w` of slack
  the exact function does not have.

  Measured on a 33-mode cascade profile, the value range came out at `33.0·w`
  **exactly, over six decades of cell width**, while the same cell's derivative
  enclosure bounded the score's movement across it by up to `7.4e5` times less —
  and the ratio DIVERGED as the cell shrank, one side being `O(w)` and the other
  `O(w²)`. Both enclosures were sound; this was overestimation.

  It was not a loose number. `maximize_score_1d` retires a cell as
  resolution-flat when its score range fits inside `2·evaluation_error`; against
  an `O(w)` range that needs a cell `rank/|f′|` times narrower than the function
  does. On a 36-row / 1725-column cascade — what a geometric box-filling net
  produces on a small sample — the flat test needed `w ≤ 6.7e-8`, 29 levels down
  a 40.6-wide domain, so no cell could be retired, none could be
  derivative-excluded, and the search refused at 8193/8192 subdivisions with
  `RemlScoreSearchUndecomposable`. That refusal names the design's rank and the
  sample's identifiability and reads as a statement about the data; it was a
  statement about the enclosure.

  The enclosure is now the **centred (mean-value) form**, intersected with the
  natural one: for `m` the cell midpoint,
  `f(x) ∈ F({m}) + F′([a,b])·[a−m, b−m]` and
  `f′(x) ∈ F′({m}) + F″([a,b])·[a−m, b−m]`, with `F({m})` obtained by calling the
  same natural extension on the degenerate interval `[m, m]`. Both forms are
  outer enclosures of one exact range, so intersecting is rigorous and can only
  tighten. The derivative is centred first and the value is centred on the
  RESULT, because a mean value remainder is only as tight as the range fed into
  it — and the curvature is centred before both, on an exact third-derivative
  enclosure the profile now builds.

  Centring the curvature is not an optional third helping. The curvature had the
  identical defect one derivative up (halfwidth `≈ 49.5·w` against an analytic
  `f″` of `1.249e-5`, a factor of 8000 at `w = 2e-3`), and the curvature is not
  merely a width: `maximize_score_1d` reads its SIGN to isolate a stationary
  point, so a first-order-loose curvature is what stops a root being isolated at
  all. The mode kernels are analytic, so the third derivative is closed form —
  `t(1−t)/(1+t)³` for the determinant, which is already the `k` kernel, and
  `t(1−4t+t²)/(1+t)⁴` for the residual, whose critical points are the roots of
  `(t−1)(t²−10t+1)`, enclosed exactly like the `k` kernel's `2 ± √3` — with
  `(log R)‴ = R‴/R − 3(R″/R)(R′/R) + 2(R′/R)³` closing the deviance block.
  `evaluate` is untouched: only the INTERVAL third derivative is needed, and no
  proof reads the scalar one.

  Overestimation becomes second order on the value and higher still with the
  curvature centred: the value range converges as `w⁴` and reaches the
  point-enclosure floor a full decade of cell width earlier. Against the
  original natural extension at `w = 2e-3` that is a factor of `7.6e7`. The same
  36-row / 1725-column design now
  certifies: `fit_reml` returns `DenseExact` at `log λ = −1.679` in 1.2 s, where
  it previously refused in 5.6 s, and the certified search's terminal value
  range is the mean-value bound to the last digit at every width tested.

  Two claims in the tree were falsified on the way and are corrected in place.
  `dense_cascade_spectrum` said this design "still spins in
  `AffineRemlProfile::enclose` under `maximize_score_1d` past 900 s" — it never
  spun; it returned the typed budget refusal in 5.6 s, #2546 having closed that
  axis. And `subdivision_budget`'s own recommendation, "the request, not the
  budget, is what actually binds", is not the repair here: the search refused at
  every requested resolution from `1.49e-8` to `1e-3`, the terminal cell merely
  walking down the domain as the request coarsened.

  **Cost.** Centring doubles the per-cell work (one extra degenerate-cell
  evaluation), so the net was measured rather than assumed, on three domains of
  the same profile: the 40.6-wide declared domain goes from a 9.94 s refusal to a
  0.58 s certification (**17×**), and a three-wide window around the optimum —
  where the natural extension already finished in a handful of cells and there
  was nothing left to remove — is still **1.26× faster**, because the tighter
  derivative range excludes cells by sign a level or two earlier. The 2× per-cell
  cost does not show up anywhere. The `residual_cascade` integration suite went
  643 s → 541 s alongside.

  One consequence points the other way and is named in the code: a tighter value
  range makes `resolution_flat_region` easier to satisfy, and an optimum landing
  in a flat region is a refusal rather than a fit. It does not happen, because
  the flat test is the last thing a cell is offered and centring strengthens
  dominance, derivative exclusion and stationary isolation by more — the gate
  asserts the located optimum is a decided one.

  Gated from four angles:
  `the_value_enclosure_never_exceeds_the_bound_its_own_derivative_certifies`
  (the invariant the natural extension broke, on a fixture built to cancel, plus
  convergence better than 50× per decade to the point-enclosure floor),
  `the_centred_enclosure_holds_on_degenerate_adjacent_and_extreme_cells` (point
  cells return the natural extension untouched; adjacent-float cells centre
  inside themselves; the centred range is always inside the natural one), and
  `auto_reml_certifies_a_design_the_data_cannot_identify` (end to end, with its
  rank-deficiency and inside-the-budget premises asserted), and
  `the_natural_extension_cannot_decompose_a_domain_the_centred_form_certifies`,
  which runs ONE search twice with the two enclosure forms — the natural
  extension is kept callable by the fix, so the before/after is a controlled
  comparison inside one test rather than a claim about a previous commit, and it
  asserts its own premise so a fixture that stops exercising the defect says so.

  Verified at `0b3b0fbd8`, release profile, 4-core runner:
  `gam-math` 284/284; `gam-terms` 936/936; `gam-solve` 1899 of 1902, the three
  reds being the pre-existing `jeffreys_subspace` and two `run_plan` failures
  already attributed to the #2612 lane at `250a04729`; `gam --test misc
  residual_cascade` 26/26.

- **The constant-curvature range coordinate was confounded with `ρ` and
  fabricated past `ℓ ≈ 10⁶`, and both were the KERNEL'S GAUGE (#2747).** The
  kernel is only ever consumed through the coefficient sum-to-zero frame `z` —
  the realized design is `K z`, the penalty `zᵀK z` — and `z` annihilates
  constants while `λ` absorbs a positive scale, so `exp(−d_κ/ℓ)` and
  `ℓ·(e^{−d_κ/ℓ} − 1)` are the SAME model in two gauges. The gauge is not free.

  `exp(−d/ℓ)z = −(1/ℓ)Dz + O(1/ℓ²)`, so design and penalty both COLLAPSE like
  `1/ℓ` and `λ̂` has to chase the range one-for-one: measured on the κ=1 sphere
  fixture, `ρ̂` falls `−5.49 → −18.91` as `ℓ` goes `1 → 10⁶` while the criterion
  value is unchanged to eight significant figures. `constant_curvature_profile.rs`
  already had this from the other side ("each ×100 in ℓ costs 4.6 in ρ̂") and
  worked around it by refusing every point whose `ρ̂` railed against the absolute
  `ρ` box.

  Worse, all of the model's range information lives in `K − 1`, formed by
  subtracting from an implicit 1 numbers that agree to `log₁₀(ℓ/d)` digits — and
  the Gram then squares what is left. The shipped criterion was **78.8 nats below
  the truth AT the derived box top** `ℓ_hi = d_min/√ε = 2.53e6`, 476 nats at
  `10⁸`, descending ~100 nats per decade into its own rounding with `edf` railed
  at `p`. That is what `20bde053f` read as "the criterion is monotone in ell all
  the way to its asymptote … ell-hat ran to 1.5e6, a readout of the box rather
  than of the data": not a flat likelihood, a false one.

  The kernel is now `k = ℓ·(e^{−d_κ/ℓ} − 1)`, evaluated as `ℓ·expm1(−d_κ/ℓ)`.
  No subtraction of near-equal numbers; `X` and `S` no longer collapse; `ρ̂` is
  flat in the range (`−5.0978 ± 1e-4` over eleven decades); and `k → −d_κ`
  exactly as `ℓ → ∞`, so the far face of the range is the geodesic-DISTANCE
  kernel — `−d` is conditionally negative definite on all three space forms, so
  it is an ordinary non-degenerate smooth rather than nothing. Three
  consequences, each handled rather than worked around: the raw `m × m` matrix is
  no longer PSD (it is conditionally negative definite), so the penalty is built
  from the RESTRICTED Gram `zᵀkz = ℓ·zᵀe^{−d/ℓ}z ≻ 0`, which is also where the
  cancellation would otherwise reappear; the ψ jets change shape, with the two
  `η` blocks becoming `ℓ·φ(q)` / `ℓ·χ(q)` for `φ = e^{−u}(1+u) − 1`,
  `χ = e^{−u}(1+u+u²) − 1`, both evaluated by series below `u = ½` because both
  have a second-order zero at the origin; and the declared scale contract goes
  from invariance to equivariance of weight one, because the kernel is a LENGTH.

  With the cancellation gone nothing numerical bounds the range from above, so
  the chart is truncated where the MODEL stops moving: `ℓ_hi = d_max/(2√ε)`, past
  which every design entry is within `√ε` of the geodesic-distance design.
  Arriving there is `RangeSolveOutcome::DistanceKernelLimit`, published as
  `RangeEstimateSupport` on the curvature report and the Python row — an arrival,
  not a rail, and the range's version of the `KappaEstimateSupport` `146f9232d`
  added for the same reason. A criterion that converges to a member of its own
  family does not need a stopping rule; it needs its limit to be a point of the
  chart. On that basis the pinned-κ/free-range enrollment `20bde053f` reverted is
  restored: a pinned `kappa=` fixes the geometry, not the resolution.

  Verified by `the_contrast_gauge_is_the_same_model_and_the_exp_gauge_loses_it_2747`
  (the two gauges agree to <1e-12 across the mid box; at the box top the `exp`
  gauge's error must land inside the DERIVED bracket `ε·ℓ/d` taken over the
  geometry's own evaluated span — measured 1.78e-10 against a predicted
  [1.1e-10, 8.8e-10]) and `the_range_limit_is_the_geodesic_distance_kernel_2747`
  (on both branches and flat, the design converges to `−D z` at first order in
  `1/ℓ`, reaching <1e-8 at `ℓ=10⁹`, with the restricted Gram strictly PD
  throughout). `gam-terms --lib constant_curvature`: 18 passed.
  `gam-models --lib constant_curvature`: 7 passed, including the 3×3
  curvature×range identification gate and the reverse-mode adjoint FD check.

- **A family had two log-likelihoods, and the joint-Newton trust ratio divided
  one by the derivative of the other (#2714).** The accept test compares
  `old_objective − trial_objective`, and the two ends came from different family
  hooks: `old_objective` is built from `current_log_likelihood`, which
  `load_joint_gradient_evaluation` reads off
  `exact_newton_joint_gradient_evaluation`, while `trial_objective` is built from
  `log_likelihood_only`, which the line search calls at `β + δ`. For the latent
  survival family those were two independent implementations — the gradient hook
  sums the row program's `∂_a^j K₀`-basis value channel, the line search summed
  `LatentSurvivalRowJet`'s rung-basis assembly.

  Writing `b(β)` for the gap, and noting that the base point does not move across
  a backtracking ladder,

  ```text
  actual_reduction = −[ℓ(β+δ) − ℓ(β)] − b(β) + (penalty terms),
  ```

  so `b` is a **constant of the ladder**: shrinking the radius shrinks the
  bracket and leaves `b` alone, and `actual_reduction → −b` instead of `→ 0`.
  Below the radius where the true reduction falls under `|b|`, the sign of `b`
  decides every attempt outright — which is the
  `rejects[model,likelihood,objective,feasibility] = [0,0,2,0]` partition at trust
  radius `1e-12` this issue was filed on. The two bases are an exact integer
  change of basis in real arithmetic (`m^k K_k = (−1)^k (∂_a)_k K₀`) and two
  different quadratures in f64, so `b ≠ 0` by construction on any row whose term
  list reaches `k ≥ 1` — every exact-event row. `k = 0` lists agreed anyway, which
  is why right-censored rows were silent about it.

  `log_likelihood_only` now evaluates the same row expression through a value-only
  lift of the same row program and sums it through the same deterministic
  reduction, so the two scalars are **bit-identical** and `b ≡ 0`. Measured over a
  35-state sweep: `worst |accept − gradient| = 0.0` exactly, and the value lift
  matches the order-two lift bit-for-bit at 100 states. It is cheaper than what it
  replaces, not dearer — the value backend skips the `K + K(K+1)/2` normalised
  moments while building the same kernel bundle.

  Two sub-faults of the same shape were fixed under it, because the value lift is
  only bit-identical if neither of them holds:

  * **The log-survival panel was placed from the requested derivative order.**
    `log_survival_panel` chose its window and node count from `order`, so two
    consumers at one `(μ, σ)` read the same integral off two Clenshaw–Curtis
    rules. On one latent-survival row the value, gradient/Hessian, contracted
    third and fourth ask for `max_k = 4, 5, 6, 7` — so the Hessian was assembled
    from a different `∂_a K₀` than the gradient it was paired with. The placement
    is now one of exactly two surfaces, each a pure function of `(branch, μ, σ)`,
    and the hot value route (every `ln S` in the tree, `max_k + 1` per bundle) is
    byte-identical and pays nothing; only the single tower request per bundle
    moves, by ~1.15× in nodes.
  * **Tower certification was all-or-nothing, so the BASIS depended on the
    request too.** Refusing the whole tower when its last rung cancelled denied a
    consumer needing rungs `0..=1` a basis well conditioned at every rung it
    reads, and let two consumers of one term list be routed differently because
    one also wanted a Hessian. It is now the longest certified prefix; a term list
    needing a rung past the truncation still falls back whole, so nothing is ever
    a partial mix.

- **The constraint-face retention ladder skipped faces, and could exit in one
  pass while reporting that it had exhausted double precision (#2714).**
  `constrained_posterior_correction` retains constraint rows by a per-row floor
  and then checks the assembled face against the identity that defines its lift
  (`max|A G − I| ≤ 1e-3`), lowering the floor on a miss by `departure/tolerance`.
  That factor is read off the per-row model `departure ≈ ε·diagonal/pivot`, which
  the filter's own documentation says bounds the face's conditioning only when the
  elimination is ordered by pivot magnitude — and this walk is ordered by slack,
  which is exactly the case the ladder runs in. So it skipped larger admissible
  faces, and on a badly conditioned face one pass carried the floor from `1e-3`
  past `f64::EPSILON`, out of the loop and into a terminal message describing a
  ladder that had taken one rung.

  Retention is a step function of the floor `d`: a row is kept iff
  `d > d_r = (k+1)·ε·diagonal_r/pivot_r`, so the retained set changes only at the
  accepted rows' own breakpoints and the face is bit-identical between them. The
  ladder now steps to `max_r d_r`, which drops exactly the worst-conditioned
  accepted row. It is exhaustive (no admissible face can be skipped), minimal (it
  stops at the largest face that delivers the accuracy), and terminates in at most
  `q` passes rather than ~40 — ending at a single retained row, where `W` is `1×1`
  and the departure gate cannot fail. No constant changed; the step size is read
  off the retention rule instead of off an error model. Gated by a brute-force
  oracle that sweeps the floor on a 64-per-octave grid and asserts the ladder
  returns the largest face satisfying its own identity.

- **The joint trust region measured the step in one norm and the radius in
  another, so on any multi-block fit it could only ever shrink (#2612).** The
  coupled joint-Newton solve carries two trust constraints: one `D`-metric ball
  on the whole step, which `WhitenedHessianSpectrum::trust_region_step` scales
  `‖δ‖_D` to, and one box per coefficient block. The controller was handed
  `max_b ‖δ_b‖` — the largest *per-block* norm — alongside the *joint* radius.
  Because `‖δ‖² = Σ_b ‖δ_b‖²`, a step sitting exactly on the joint sphere has
  `max_b ‖δ_b‖ = ‖δ‖/√K` when `K` blocks carry comparable mass, so
  `hit_boundary = step_norm ≥ 0.99·r` was **false on a boundary step for every
  `K ≥ 2`** and the region became a one-way ratchet.

  Measured on the #2612 penguins witness, over all 6784 accepted trust attempts
  of one fit:

  | `‖δ‖/r` | attempts |
  |---|---|
  | ≥ 0.99 (what the controller looks for) | 1 |
  | 0.70 – 0.99 (the `1/√2` band, median 0.781) | 2018 |
  | < 0.70 | 4765 |

  1454 of those 2018 carried a Newton proposal at least `1.5×` the step actually
  taken, 563 of them `≥ 10×`. The fit died with two inner solves at the ratchet's
  floor — `|prop|∞ = 7.686e-5` against an accepted `|δ|∞ = 5.270e-7`, the residual
  crawling `0.9932×/cycle` — and 50 of the run's inner solves ended
  non-converged. With the norms paired correctly that is **3**, and the fit takes
  116.6 s instead of 333.4 s.

  For `K = 1` the joint norm *is* the block norm and the joint radius *is* the
  block radius, so every single-block family is byte-identical; the change
  reaches exactly the multi-block joint solves (multinomial, location-scale,
  marginal-slope) where the test was in the wrong units. It also explains why
  the #2612 `objective_unreadable_at_this_step` growth clause never fired on
  multinomial: it is gated on the same structurally-false boundary test.

  Two consequences of the same reading are fixed with it. **"Held, not grown" is
  a fixed point when the region is what limited the step** — the accept-below-
  model-noise-floor branch (#2637) is right to hold an *interior* step, but on
  the boundary the step is short because the radius is small, the prediction is
  unreadable *because* the step is short, and the radius then freezes forever
  (measured: `r = 1.463e-6` held for 167 cycles against a `|prop|∞ = 8.961e-5`).
  It now grows on the three facts that are measurements rather than predictions:
  a realized decrease above the noise floor, geometric boundary contact, and a
  stationarity residual still above tolerance.

  **Measured and reverted (recorded because the negative result is the useful
  part).** A third repair looked equally well-founded and is wrong: a ladder
  verdict is one realization of a random band, so the ladder was made to publish
  the envelope it proved (`remainder ≤ c·‖δ‖²` below a step length it had shown
  to be noise-dominated) and every later attempt inside that envelope widened the
  measurement. The motivating observation is real — the penguins refit measures
  `4.396e-11` at cycle 149 and then rejects a `1.168e-10` realized change at
  cycle 162, at a step length the ladder had already certified — but the repair
  makes the fit *worse*, not better:

  | build | wall | inner solves ending non-converged |
  |---|---|---|
  | before the norm repair | 333.4 s | 50 |
  | norm repair | 116.6 s | 3 |
  | + boundary growth | 119.0 s | **2** |
  | + certified noise envelope | 291.9 s | **47** |

  Over-measuring the objective's resolution is worse than under-measuring it,
  because the accept test then reads genuine objective changes as rounding and
  the solve stops being able to tell a good step from a neutral one — the
  measured resolutions rose from `~4e-11` to `~3e-10` and the worst terminal
  residual went from `3.8e-7` to `4.0e-5`. The under-measurement is therefore
  real and is NOT the binding constraint; the envelope is not the way to fix it.

- **The inner solve conceded on a step model that is not the objective's
  Hessian, and with that repaired the multinomial outer search converges for the
  first time (#2612).** `H_Φ` is the Daleckii–Krein divided-difference part of
  `−∇²Φ`; the exact second-order completion `−½ tr(K D_ab)` is the rest of it,
  and until it is formed the Newton step is built on a matrix that is not the
  Hessian of the objective the certificate is taken against.
  `JEFFREYS_COMPLETION_RESIDUAL_BAND` arms it on a proximity proxy — the residual
  reaching `300 × residual_tol` — which is circular wherever the distance from
  tolerance is *caused* by the inexact model.

  Measured on the penguins witness once the trust region had been repaired
  enough to stop being the binding constraint: cycle 155 takes the **full**
  Newton step (`|δ|∞ = |prop|∞ = 1.069e-4`, interior at `r = 9.290e-3`) and the
  residual still does not contract — it drifts at `1.0031×/cycle` at `2.398e-6`,
  five hundred times outside a band of `4.3e-9`, with
  `jeffreys_completion_calls = 0`. The step is exactly as long as the model
  wants; the model is the wrong matrix. Arming the completion where the solve
  would otherwise concede:

  ```text
  cycle 39  about to concede at residual 4.593e-6  → arm the completion
  cycle 40  ρ=+0.9922   residual 4.593e-6 → 7.410e-8
  cycle 41  ρ=+1.000    residual 7.410e-8 → 4.298e-12   (tol 1.441e-11)
  ```

  The repair is an invariant rather than a threshold — *the inner solve may not
  concede while its step model is still the surrogate* — asked at the
  residual-stall guard, the slow-geometric-rate projection, and per cycle
  against the budget **this solve actually has** (the stall guards are allowed
  to defer to a historic floor of 100 cycles; a screening evaluation with a
  64-cycle budget is not). End to end on `zz_probe_2612_penguins_stride3_inner_trail`:

  | build | wall | inner solves non-converged | outer verdict |
  |---|---|---|---|
  | before | 333.4 s | 50 | `line_search_failed` at `\|g\| = 1.556e-1` |
  | trust-region norms | 116.6 s | 3 | inner infeasible |
  | + boundary growth | 119.0 s | 2 | inner infeasible |
  | + completion invariant | 493.3 s | 4 | **Converged**, `\|g\| = 1.357e-3 < 2.290e-3` |

  The cost is stated rather than hidden: an armed solve pays an extra
  `O(n·M²·P²)` contraction per cycle, and a solve that never needed it never
  arms. Regression surface: `gam-custom-family --lib` 275/275,
  `gam-models --lib -- location_scale` 224/224, `-- marginal_slope` 229/229,
  `multinomial_separation_arming_2612` 3/3 (accuracy 0.9750, calibration gap
  −0.0151) — the other multi-block joint families exercise the same two repairs
  and none of them moved.

  **Where #2612 now stands.** The fit still does not mint, and the blocker is a
  different subsystem: the outer search converges to an *interior strict saddle*
  (`λ_min = −1.074e1` on the un-railed sub-block, 19 of 24 `θ` railed), the
  `#2357` negative-curvature escape reseed fires and lowers the objective
  (`7.545783 → 7.545502`), and the re-run climbs back to the identical ρ and the
  identical saddle, at which point the one-shot escape is spent and the
  certificate refuses. That is the `#2357`/`#2665` family — a gradient-only BFGS
  search (`search_hessian_source=BfgsApprox`) cannot see the negative curvature
  it is sitting on.

- **A Jeffreys drift GEMM panicked on a column-major product (#2612).**
  `as_slice` is C-order-only and neither `dot` nor `+` promises that order, so
  `dw_rows.dot(&a_rows.t()) + …` could return column-major and the `expect`
  fired — in production code. `binomial_location_scale_expected_hphi_drift_matches_finite_difference`
  died on it. `as_standard_layout` borrows in the C-order case, so the GEMM path
  is unchanged.

- **One sentinel, one resolver — the marginal-slope branch never reached the
  measure-jet range screen (#2754, #2761).** `length_scale == 0.0` is an
  UNRESOLVED representer range, and the tree carries two resolvers for it: the
  pure-geometry median-nearest-node rule inside the basis builder, and the
  #2750 response screen. `fit_standard_model` runs the screen so that every
  standard-fit branch gets the same one. The Bernoulli marginal-slope family has
  its own entry point and never passed through it, so the identical declaration
  on byte-identical rows realized two different spans:

  ```text
  [2754 geometry gaussian-seed] ell=2.5197  m=(10,2) extent=[2.671, 2.726] band0=1.0807
  [2754 geometry bms-marginal ] ell=1.0807  m=(10,2) extent=[2.671, 2.726] band0=1.0807
  [2754 geometry bms-logslope ] ell=1.0807  m=(10,2) extent=[2.671, 2.726] band0=1.0807
  ```

  Same 10 centers, same extent, same band floor, **2.33× apart in ℓ** — and the
  BMS value is exactly `eps_band[0]`, which is the geometry heuristic's own
  output by construction, i.e. the fingerprint of a term that reached no
  resolver at all. `ℓ` decides WHICH span the representers occupy and `λ` cannot
  move a span, so this is not a tuning difference between entry points; it is a
  different model reached by typing a different family name. It is also exactly
  the mechanism #2761 named: #2750 measured the geometry heuristic sitting 21.7
  nats away from the criterion's global optimum, and #2761 measured its span
  floor four orders above the chosen range's.

  **What the range was worth on this fixture, before the fix.** The parity test
  cites a length-scale sweep (`zz_mjs_lengthscale_sweep_1041`) for the claim
  that "the auto ℓ is already the BEST — every explicit ℓ is worse". That test
  is not in the tree; `grep` finds only the citation. Rebuilt as
  `examples/probe_2754_bms_length_scale_sweep.rs` on the parity fixture's own
  data law and its own held-out score, the claim inverts — the auto range is the
  WORST of the eleven measured:

  | ℓ (standardized) | held-out marginal RMSE |
  |---|---|
  | 1.08 (auto / geometry) | 0.04441 |
  | 2.14 | 0.04157 |
  | 8.56 | 0.04170 |
  | 17.12 | 0.04011 |
  | 25.68 | 0.03985 |
  | 68.48 | 0.03788 |
  | matérn(k=10) | 0.05234 |
  | duchon(k=10) | 0.03705 |

  **Not in tension with the ℓ-learning freeze** two screens above it in the same
  function. The freeze is about the SEARCH: a design-moving dial on covariates
  shared by the coupled marginal/log-slope pair lets the outer optimizer trade
  one surface against the other into a separation-scale runaway. The screen is
  about the SEED, runs once before the fit, and hands the frozen dial a
  data-chosen basin. Freezing a dial is a reason to seed it better, not worse.

  **Each surface is screened against its own target.** The marginal block takes
  `y`. The log-slope block cannot: `β` never appears in `E[y | x]`, so ranking
  its spans against `y` ranks them by their fit to the MARGINAL surface. It
  takes the first-order score surrogate `s = (y − ȳ)(z − z̄)`, whose conditional
  mean is the planted log-slope surface times a strictly positive smooth
  modulation —

  ```text
  Cov(y, z | x) = E[ z·F(α(x) + β(x)·z) ] = F'(α(x))·β(x) + O(β³)
  ```

  by expanding the link about `α(x)` (the odd moments of `z` kill the even
  terms). The profiled Gaussian REML the screen ranks with is invariant to a
  global rescale of its response, so the unknown `E[z²]` and `F'` scales are
  both free. `logslope_screen_surrogate_tracks_the_slope_surface_not_the_marginal_2754`
  checks that derivation against a 200k-row probit sample rather than asserting
  it, and scores the binned surrogate against BOTH candidate truths so the
  separation from `E[y | x]` is the thing being pinned.

  Gated by `measure_jet_auto_range_is_the_same_through_every_family_entry_point_2754`,
  which asserts EXACT `f64` equality of the realized range across the two entry
  points — the screen is a deterministic function of (feature columns, response,
  weights, spec), so handed the same four it must return the same number, and a
  tolerance would hide a second resolver that happens to land nearby on one
  fixture. It asserts the realized geometry matches first, so a range difference
  cannot be explained away as the two entries having realized different center
  layouts.

  The same bypass was in `fit_transformation_normal`, fixed in the same lane:
  its covariate surface enters the linear predictor of the transformed response,
  so `response` is its own screening target. The reached/unreached inventory now
  lives in the doc comment on `seed_measure_jet_auto_ranges` itself — three
  entry points screen (standard, BMS, CTN), five still take the geometry
  heuristic (survival marginal-slope, the two latent families, the
  location-scale families, survival-transformation) and are marked **not
  derived** rather than fixed: for those the raw response is not a readout of
  the surface being screened (a survival marginal-slope block is modulated by
  the risk set in `age_entry`/`age_exit`; a location-scale SCALE block enters
  through a variance, so ranking its spans against `y` ranks them by their fit
  to the LOCATION surface). Inventing one target per family without a fixture
  that can grade it would be landing an unmeasured modelling choice in five
  places at once; the honest state is that the table says so out loud.

- **The #1041 parity bar is now policed by a statistic that can resolve it
  (#2754).** The gate fitted ONE draw and compared one ratio to `1.10`; the
  ratio's sampling spread under redraws of the identical generator had never
  been measured. The argument on #2754 used the BETWEEN-method spread
  (matérn/duchon = 1.42×) as if it were a noise estimate, and it is not — two
  estimators differing by 1.42× says nothing about how much ONE estimator moves
  when only the draw changes. Measured, the within-method sd of the log ratio is
  **0.119** at a mean ratio of 0.97, so the single-draw gate sat ~1.1 sd below
  its own bar and failed about **one run in eight** for no reason but the draw.

  The bar is unchanged at `1.10×` and the comparator stays Matérn: it is the
  only statement in the tree that measure-jet must remain competitive with its
  own estimator class as both change. What changed is the instrument. The gate
  now reports the mean log-ratio over `REPLICATES` independent draws and asserts
  both that it clears the bar and that it clears it **by at least three standard
  errors** — #2754's finding made permanent, so a fixture whose noise grows
  relative to the margin it polices says "under-powered" in as many words
  instead of flipping a coin, and says it about the FIXTURE rather than the
  estimator. `REPLICATES` is derived from `3·sd/√k ≤ margin`, not chosen.

- **The range screen's walk stopped at a proxy for the wall, and on a frozen
  dial a stopping rule IS the wall (#2761).** The #2750 response screen walks
  geometrically past the top band node while its criterion improves, and stopped
  at the node bounding-box diameter, on the argument that at a range that long
  every representer pair overlaps at `≥ exp(−1/2)` so "there is no distinct
  model past it". Two places in the tree already recorded the opposite —
  `measure_jet_ln_range_window`'s docs (*"measured on three fixtures, the
  profiled criterion genuinely prefers a range AT or ABOVE the node
  diameter"*) and a test that pinned the search window as strictly wider,
  calling the diameter *"a stopping rule for the screen's walk over NODES, not a
  wall in the model"*. Those reconcile only while something else keeps searching
  past the stopping rule; on a term whose `ℓ` dial is frozen (the marginal-slope
  pair, or any `learn_length_scale=false`) nothing does.

  Derived on the #1041 parity fixture from the shipped numbers and the walk's
  own control flow: band `[1.08074, 1.43607, 1.90823]`, `log_step = 0.284265`,
  diameter `3.81645`, walk nodes `2.53562`, `3.36930`, `4.47708`; the range the
  screen chose was `3.36930` — the last node below the diameter, to every
  printed digit. The walk pushes a node and only then breaks if it failed to
  improve, so an argmin that IS the last pushed node improved, and the loop
  therefore left through the ceiling test with the criterion still descending.

  `measure_jet_range_feasibility_ceiling(spacing)` is now the single definition
  of `spacing/√(2√ε)`, read by both the outer search's window and the screen's
  new `MeasureJetRangeBracket::feasibility_ceiling`; the diameter survives as
  `node_diameter`, reported as the geometric fact it is and no longer
  load-bearing. **Measured after the change rather than assumed from the shape
  of the defect:** the node the walk can now score, `4.47708`, does NOT improve,
  so the criterion has an interior optimum here and the old ceiling cut just
  past it. What the change buys is the parabolic refinement, which cannot fire
  on an argmin that is the last element — a stop that cannot be stepped past
  also cannot be refined at. It lands at `ℓ = 3.10543` with a better criterion
  value and held-out RMSE `0.04185 → 0.04179`. The much larger held-out number
  further out on the same sweep (`0.03788` at `ℓ = 68.5`, `edf = 7.47`, not
  degenerate) is explicitly NOT what this recovers: the criterion does not want
  to go there, and that gap is a question about the screening criterion rather
  than about where the walk stops.

- **A chart records the `θ` it was ASKED to realize, because `ln(exp(θ))` is not
  `θ` (#2765, #2767).** `SurvivalMarginalSlopeFrozenOffsetChart::evaluate(θ)`
  decoded `θ → cfg` and then called the CONFIG-authored geometry builder, which
  closes the loop by re-encoding `cfg → θ`. For a Weibull that loop is
  `ln(exp(θ))`, and in `f64` it is not the identity: over a grid on `[-3, 3]`,
  **17.3%** of coordinates come back a ulp or more away, and `θ = 1e-5` comes
  back **57 269 ulps** away.

  `SurvivalMarginalSlopeFamilyHyperState` stores that `theta` as the family's
  realized coordinates, and `validate_layout` compares it to the outer manifest
  with `to_bits()` equality — deliberately, so a workspace cannot reuse row
  geometry from a neighbouring outer probe. So a lost ulp in a transcendental
  round trip made the inner solve REFUSE a point the outer optimizer was only
  trying to evaluate:

  ```
  inner solve refused this trial point: SurvivalMarginalSlopeFamily row
  geometry does not bitwise match the family-coordinate manifest
  ```

  **The measurement, probe by probe.** From the #2765 replay fixture's terminal
  certificate at `θ = [7.0218, 5.5967, 6.0955, 0.75749637812226, 0.0]`, step
  `1e-5`, against the round-trip error of each displaced coordinate:

  | probe | `ln(exp(θ)) − θ` | certificate |
  |---|---|---|
  | coord 3, side `−` | `+1` ulp | REFUSED |
  | coord 3, side `+` | `0` | (not reported) |
  | coord 4, side `+` | `−57 269` ulp | REFUSED |
  | coord 4, side `−` | `+3 383` ulp | REFUSED |
  | the seed itself | `0` | evaluates |

  Four out of four, and the one exact round trip is the one that evaluated. A
  refusal reads to Armijo as "no improvement" at *every* step size, so a
  backtracking search halving 50 times produces `StepSizeTooSmall after 50
  attempt(s)` and `after 0 outer iteration(s)` with no gradient defect required
  — and it explains why the earlier #2765 probe saw trial points that were
  genuinely better than the base rejected anyway.

  **The repair is the seam, not the check.** The bitwise invariant is right and
  stays exact; what was wrong is that the chart threw away the coordinate it was
  handed. `build_survival_marginal_slope_baseline_geometry_at_theta` records the
  caller's `θ` verbatim while `cfg` still drives every row's arithmetic. The
  config-authored entry is unchanged: where a config IS the authority, deriving
  `θ` from it is correct. Loosening the manifest comparison to a tolerance was
  rejected — that check exists so "the same coordinates" means the same thing to
  the family and to the outer manifest, and a tolerance trades a loud refusal
  for a workspace that can silently serve a neighbouring probe's row geometry.

  End to end on the replay fixture (n=900, Weibull baseline, `logslope_time_k`),
  same seed, same cost `8.193691e2`, same gradient `|g| = 8.164517e0`: the outer
  solve goes from refusing in 788 s `after 0 outer iteration(s)` to completing
  seed 0 and moving on to seeds 1 and 2. Nothing about the value or the gradient
  changed; the displaced points became evaluable.

- **The ψ calculus and the joint-Hessian OPERATOR never got the slope's
  follow-up axis (#2765, #2767).** `c9ad097f1` generalized the survival
  marginal-slope blockwise assemblers from the four-primary frame
  `(q₀, q₁, q̇₁, g)` to the six-primary one `(q₀, q₁, q̇₁, g₀, g₁, ġ₁)` by routing
  every log-slope pullback through `logslope_layout.primary_channels()` — one
  `(primary, design)` pair for a time-constant slope, three for a varying one —
  and `fit_entry` then refused, by name, every surface whose chain rule was
  still lowered through the old frame. Its own comment states the standard:

  > refusing by name is honest, whereas running them would silently
  > differentiate a model that is not the one being fitted.

  Five sites did not get the treatment and are not on that refusal list. Each
  reads the slope through the literal index `3` and a single design:

  | site | what it assembled |
  |---|---|
  | `hessian::add_pullback_with_q_geometry` | `ph[[3,3]]` into `H_gg`; `coefficient_design()` for `H_mg` / `H_tg` |
  | `psi_terms::accumulate_score_blockwise` | `coefficient_design().axpy_row_into(row, primary[3], …)` |
  | `psi_terms::accumulate_score_with_q_geometry` | the same, one channel |
  | `primary_geometry::spatial_block_primary_loading` / `primary_direction_from_psi_row` / `primary_psi_action_from_psi_row` | length-**4** vectors contracted against a length-**6** primary gradient |
  | `timepoint_exact::row_primary_fourth_contracted` | instantiated at `STATIC_SLOPE_PRIMARIES` — the *time-constant row program* — whichever frame the family is in |

  The first one is the widest: `exact_newton_joint_hessian_operator` — the
  INNER Newton's matrix-free joint Hessian on the dynamic-q path — is one of its
  callers. Its gradient half (`accumulate_dynamic_q_core_gradient`) and its
  dense sibling (`accumulate_dynamic_q_core_hessian`) were both converted; the
  operator's Hessian half was not, so on a follow-up-varying slope the inner
  solve ran with a curvature missing the `g₁` and `ġ₁` rows and columns
  entirely. A mode certified against the wrong curvature is not the argmin of
  the criterion the outer search is profiling, which is exactly the
  "`converged=true` with `last_residual_below_tol=false`" signature the #2765
  probe recorded 55 times in one run.

  **The two holes in the refusal list.** The refusal names a spatial length
  scale on the *log-slope* surface, Gaussian-shift frailty, the score-warp /
  link-deviation flex blocks, the CTN Stage-1 absorber, and a time-wiggle
  baseline. It does not name:

  * **a parametric baseline chart** — `baseline_exact_joint_psi_terms_with_options`
    reaches `accumulate_score_with_q_geometry` and `add_pullback_with_q_geometry`
    unconditionally, so a Weibull / Gompertz / Gompertz-Makeham ψ dropped two of
    the slope's three channels. This is the configuration #2765's own acceptance
    fixture uses (`baseline_target: "weibull"` plus `logslope_time_k`), and its
    outer search direction was measured to be essentially pure baseline-ψ;
  * **a spatial length scale on the MARGINAL surface** — `psi_terms_inner`
    contracted a length-4 primary direction against a length-6 primary
    gradient, which is a shape error rather than an approximation.

  **The repair is the generalization, not a guard.** Every site above is now a
  loop over `primary_channels()` or is sized from `core_primary_dimension()`. A
  time-constant slope still performs exactly one rank-1 update per row and its
  arithmetic is unchanged, so nothing about the static path changes shape or
  cost. The log-slope-surface spatial refusal STAYS and is now enforced one
  layer down as well: with a time margin that block's three channel designs are
  `X_cov ⊗ B_entry`, `X_cov ⊗ B_exit` and `X_cov ⊗ B′_exit`, while the ψ
  design-derivative contract carries a single `X_ψ`, so the other two channels
  are not recoverable from what the caller holds — a fact the primary-space
  helpers now state rather than assume. The batched ψ fast path, whose tower is
  instantiated at `STATIC_SLOPE_PRIMARIES`, declines a follow-up-varying slope
  and defers to the per-axis route instead of truncating it.

  **The gate that was missing.** `psi_terms_inner` publishes
  `∂_ψ ℓ̄`, `∂_ψ ∇_β ℓ̄` and `∂_ψ ∇²_β ℓ̄`, and nothing had ever differenced them
  against the functions they name. The shipped ψ coverage checks finiteness,
  subsample-vs-unsampled equality, and batched-vs-per-axis agreement — a
  *consistently* wrong derivative passes all three.
  `marginal_slope/psi_terms_fd_tests.rs` now differences every ψ lane (marginal
  design, log-slope design, and each baseline-chart coordinate) against the
  family's own `(objective, score, Hessian)` triple, in BOTH slope frames, with
  a Richardson pair certifying the oracle so a gap cannot be charged to the
  finite difference and an unresolved component declines to grade rather than
  fails.

- **A monotone warp's corner is its boundary knot's MULTIPLICITY, and no
  extrapolation rule can remove it (#2695).** `0167ed853` gave the warp basis a
  linear tail so `I′_j` would stop stepping at the knot hull's edge, and the
  witness fit. It did not close the issue: a warp with real amplitude still
  refused at degree 2, 3 and 4, all on OBJECTIVE rejections, while the same data
  fitted cleanly with the `linkwiggle(...)` term removed.

  **What no degree helps means, measured.** `f19e2bee4` reads the one-sided gap
  in `I^(k)` across a point at `h = 1e-3` and `h = 1e-6`; a ratio of `1e3` says
  the gap falls with `h` (continuous), `1e0` says it does not (a step):

  | degree | interior knot | hull edge (linear tail) |
  |---|---|---|
  | 2 | steps at order **2** (2.0) | steps at order **2** (2.0) |
  | 3 | steps at order **3** (6.0) | steps at order **2** (6.0) |
  | 4 | steps at order **4** (24) | steps at order **2** (12) |
  | 5 | no step at `k ≤ 4` | steps at order **2** (20) |

  The hull edge steps at order 2 at EVERY degree — the tail zeroes `I″` while
  the interior one-sided `I″(right⁻)` is `2, 6, 12, 20`. That is why the
  four-arm degree sweep found no degree that helps: raising the degree moves the
  INTERIOR step up the tower and leaves the edge exactly where it was.

  **Why order 2 is the order that bites.** `m1 = 1 + Σ_j βw_j·I′_j(q1)` enters
  the event Jacobian `g = η_t′ + m1·q̇₀`, so `∂²m1/∂βw_j∂β_thr = I″_j·∂q1/∂β_thr`
  is an entry of the observed information `H` carrying `I″_j` with **no `βw`
  factor** — it survives at `βw = 0`. `Φ = ½ Σ g(λ(Z_JᵀHZ_J))` is inside the
  accept test, so a step in `I″` is a step in the OBJECTIVE, and
  `actual/predicted` cannot approach `1` at any step size. (The channel is live
  only on EVENT rows: `log g` is added when `w·d ≠ 0`, so a censored crossing row
  reports "continuous" for the wrong reason.)

  **Why a better tail is not available.** At a clamped edge most columns have
  `I′ = 0` and `I″ ≠ 0` — at degree 2 the right edge carries `I′ = [0, 0, 0, 2]`
  and `I″ = [0, 0, −1, 2]`. A monotone `C²` extension of a column with
  `I′(e) = 0` and `I″(e) < 0` **does not exist**: `C²` forces
  `I′(e+ε) ≈ I″(e)·ε < 0`. The corner is not the extrapolation rule; it is the
  boundary knot's multiplicity (`degree + 1` on a clamped vector), and the cure
  is a knot vector whose ends are SIMPLE.

  **The repair.** Two pieces, one contract — *the warp is one `C^{degree−1}`
  function on all of `ℝ`*:

  * `gam_terms::basis::ispline_ramp_basis_dense` evaluates the I-spline as what
    it is — `I_c(x) = ∫_{t_{c+1}}^{x} M_{c+1}`, exactly `0` before that support
    and `1` after — for ANY knot vector. `create_ispline_dense` computes the
    same right-cumulative sum but reads it only where the degree-`bs` B-splines
    are a partition of unity and imposes `0`/`1` outside by convention; that
    convention is exactly right on a clamped vector (pinned: the two agree to
    `1e-12` at every degree, inside the hull and outside it) and wrong on any
    other, because a ramp whose support runs past that interval gets truncated
    mid-rise. Evaluating on a padded knot vector removes the truncation without
    changing a single clamped value.
  * `gam_terms::basis::monotone_warp_knots` builds the warp's knots as
    `num_internal_knots + 1` uniform spans across the seed range, continued by
    `degree` further spans at the same width on each side, all knots simple. The
    column count is unchanged (`num_internal_knots + degree` either way), so
    nothing downstream changes shape.

  `monotone_wiggle_basis_with_derivative_order` is now one call into the ramp
  evaluator: the hull, the clamp, the linear tail and the `orders ≥ 2 → 0` rule
  are all deleted, not patched. Warp blocks (survival link and time wiggle,
  GAMLSS, BMS) build their knots with the warp generator;
  `initializewiggle_knots_from_seed` stays clamped for bases evaluated on FIXED
  data — a response transform — where a boundary knot's multiplicity is
  invisible because the evaluation point never moves across it.


- **The multinomial's Firth/Jeffreys separation certificate is now taken on
  `ker(S_λ)` — the directions no smoothing parameter reaches — instead of the
  whole identifiable span (#2612).** `27301d428` fixed which *matrix* the arming
  verdict is taken on (`H + S_λ`, the curvature the fit has). This is the rest of
  the same sentence #715 derives: `(H + S_λ)v = Hv + λSv`, so a direction is
  beyond every `λ`'s reach exactly when `S_λ v = 0`. Where `S_λ v ≠ 0` the model
  already carries a proper prior on `v`, and since the ratio-of-normalising-
  constants predictive landed, that width is integrated exactly rather than
  approximated — so it is already in the published probability.

  Measured, the certificate now names the subspace it decided on: `2/16`
  unreached directions on a one-smooth quasi-separated fixture (`H + S_λ` there
  in `[9.84e-1, 6.56e0]`) and `2/74` on the penguins witness (`[2.86e-3,
  9.64e-1]`) — in both cases the class intercepts, holding under one
  observation-equivalent apiece. The refusal also reports what it did *not*
  decide on (the whole penalized span, and the likelihood alone), and both
  branches log the unreached dimension, so a disarmed fit says why it disarmed.

  `jeffreys_subspace_from_penalty` now computes the kernel its own return type
  has always advertised instead of discarding its argument and returning `I_p`;
  the zero operator short-circuits to an exact identity, so every existing caller
  is byte-identical. `multinomial_reml::measured_penalty_nullspace` delegates to
  it, so "which directions does this penalty reach" has one answer in one place.

  **Recorded and NOT landed, with the controlled measurement.** Restricting the
  *term* to that same subspace — the natural completion, since
  `jeffreys_antiderivative` acts wherever a reduced eigenvalue is under
  `CONDITIONING_GATE_ABSOLUTE_CLEAR = 16` and a quasi-separated softmax has
  `λ_max(H) = 1.44` over a 74-dimensional span at `n = 228`, so the term acts on
  the entire basis — fixes the calibration outright: an armed quasi-separated
  smooth fit goes from a held-out calibration gap of `−0.0802` to `−0.0151` and
  log-loss `0.13224 → 0.07682`. It also costs the penguins witness its fit. The
  full-span term is incidentally a *regulariser of the inner problem*: with it
  the joint Newton reaches the LAML derivative lane's `1e-11`; without it the
  residual plateaus at `9.84e-7`, and loosening the target to the objective's own
  measured floor (`MULTINOMIAL_FORMULA_INNER_TOL = 1e-5`) then desyncs the
  analytic outer gradient by exactly the amount #1820 documents, so the outer
  line search terminates with `StepSizeTooSmall` at `|g| = 1.76e-1`. The span
  cannot be narrowed until the inner joint-Newton can certify a near-separable
  multinomial mode to the accuracy the derivative lane needs — iterative
  refinement on the inner KKT solve, not a looser target.
  `a_quasi_separated_smooth_fit_is_calibrated_2612` is left RED against a
  four-standard-error bar as the standing measurement of that gap.

- **A negative-curvature saddle escape now judges its own trial against the
  CRITERION's resolution, and the adjudication is no longer gated by the
  reseed's one-shot budget (#2612).** `adjudicate_negative_curvature` derives
  where to stop probing from the criterion's own resolution — the ladder ends at
  `α_min = sqrt(2·objective_resolution/|λ_min|)`, on the stated ground that below
  it "the claim predicts nothing the criterion can represent" — and then decided
  whether a probe had DESCENDED against `16·ε·|V|`, the arithmetic's resolution.
  One function, two notions of "a decrease the criterion can represent", ten
  orders of magnitude apart.

  Measured on the two fixtures #2612 is decided on: the penguins witness's
  unbiased probe minted four escape reseeds at `λ_min` of `−6.4e−7 … −2.0e−6`
  (machine zero against `‖H‖ ≈ 1`) on objective decreases of `2e−6 … 4e−6`
  against a resolution of `1.228e−3`; a one-smooth quasi-separated fixture minted
  three on decreases of `3.4e−4`, `1.4e−4` and `5.0e−5` against a *measured*
  cost-stall noise floor of `1.91e−4`. Each reseed spent the one-shot budget, and
  the retry pass — which `allow_tail_snap` forbade from adjudicating at all —
  then refused on the matrix's word. Both fits died; the fit that shipped was the
  Firth/Jeffreys-armed one.

  At `λ_min = −6.4e−7` the claim's predicted decrease at the LARGEST feasible
  step is `3.2e−7`: it was unfalsifiable over its whole derived range, which is
  exactly the state `CurvatureEvidence::CriterionContradicted` exists to record.
  The strict-decrease floor is now `objective_resolution` — the same
  `rel_cost_tolerance`-anchored anchor the ladder's own limit is derived from —
  floored at the arithmetic roundoff that remains the hard lower limit, so no
  constant is introduced. And the adjudication runs on every refusal: that flag
  is the one-shot budget for the RESEED, and gating a MEASUREMENT on it made the
  retry pass refuse for want of a measurement it could have made for free.
  `Descended` still spends the budget, and on the retry pass now records that the
  criterion CONFIRMS the curvature, so a measured saddle and an unmeasured one
  stay distinguishable.

  Effect: a one-smooth quasi-separated multinomial (`cls ~ s(x, k=8)`) that
  produced no fit at all now fits in 7.5 s, and the penguins witness's unbiased
  criterion reaches a certified stationary optimum instead of exhausting its
  strategy fallbacks.

- **A follow-up-varying marginal slope can now be SAVED, predicted from, and
  leave-one-out replayed (#2765, #2767).** `logslope_time_k` fitted a real model
  since the kernel work landed, but persistence refused it outright: the on-disk
  contract rebuilt the log-slope block from its covariate term spec alone, which
  names `p_cov` columns against a `p_cov · p_time` coefficient vector. A fitted
  surface that cannot leave the process is half a feature.

  **What was missing was one fact, not one code path.** The block's authority is
  the covariate spec *plus* the resolved time margin, and only the first half was
  persisted. `logslope_time_basis` (degree + knots) now rides on the saved model
  beside the threshold and log-σ margins it was built by the same primitive as,
  and every consumer rebuilds `X_cov ⊗ᵣ B(log t)` from it. The knots are fit-time
  values, so a prediction sample can never move the basis by re-estimating
  quantiles — the same contract the location-scale margins already hold.

  **Two places where "replay it" is not the obvious thing.**

  A predicted survival curve evaluates `b` **at each time on the curve**, not at
  the row's observed exit time. The family is `S(t) = Φ(−η(t))` with
  `η(t) = q(t)·c(t) + b(t)·z`; freezing `b` at `t_exit` would return a curve
  assembled from a different model at every point but one. The per-`(row, t)`
  evaluator therefore re-tensors the row's covariate factor against the margin at
  the time being predicted, exactly as it already rebuilds the time basis there.

  The leave-one-out (`--alo`) replay re-evaluates the row program, which reads
  the slope at entry, at exit, and as an exit-time rate — three channels, because
  the likelihood is `log S(t₁) − log S(t₀)` and an event row also carries
  `log η′(t₁)`. Handing it the exit design alone would not have been an
  approximation; it would have reported the influence of a time-CONSTANT slope,
  and the widths would have agreed while it did so. All three channels are
  rebuilt, and the ALO input refuses a follow-up triple whose shapes disagree
  rather than indexing through them.

  **What guards the replay.** One function evaluates the log-slope time axis, so
  the batch replay and the per-cell replay cannot ask for different bases; a test
  asserts the replayed exit margin is the fit-time margin *bit for bit* rather
  than merely close, because a margin that is nearly right is a model that is
  quietly wrong. Load-time validation refuses a saved block whose width is not a
  multiple of its own margin — a payload in that state cannot be replayed under
  any covariate design at all.

- **The multinomial's Firth/Jeffreys separation certificate now judges the
  curvature the fit HAS (`H + S_λ`) instead of the likelihood's alone (`H`),
  because reading `H` alone made "arm only on separation evidence" fire on every
  multinomial GAM carrying a smooth (#2612).** The conditional engagement
  (#715 arm (b) / #753) exists because the proper prior is not free: it pulls
  fitted class probabilities toward the uniform simplex `1/K`, and it routes the
  fit through a lane whose outer Hessian omits `D²_β H_Φ` by construction, so its
  curvature certificate is deliberately weaker. A false positive costs a whole
  re-solve on a biased objective and a certificate that cannot be as strong.

  The decision came from the reduced conditioning gate at the certified mode,
  whose absolute arm is derived from **one observation-equivalent** of curvature
  and whose doc block states the premise that makes it conservative: *"it never
  fires on a genuinely well-conditioned large-`n` fit, whose `λ_min = O(n) ≫ 1`"*.
  That premise fails for every penalized smooth basis. Measured through the
  shipped Python surface on labels DRAWN from a smooth softmax truth — every
  class keeping appreciable probability everywhere, nothing separating anywhere:

  ```text
    y ~ x1 + x2 (parametric)       n=  300 unbiased
    y ~ x1 + x2 (parametric)       n=  900 unbiased
    y ~ x1 + x2 (parametric)       n= 3000 unbiased
    y ~ s(x1,k=5)+s(x2,k=5)        n=  300 ARMED    lmin=2.115e-1  lmax=7.647e+1
    y ~ s(x1,k=8)+s(x2,k=8)        n=  900 ARMED    lmin=4.579e-1  lmax=2.494e+2
    y ~ s(x1,k=12)+s(x2,k=12)      n=  900 ARMED    lmin=2.531e-1  lmax=2.488e+2
    y ~ s(x1,k=5)+s(x2,k=5)        n= 3000 ARMED    lmin=1.989e+0  lmax=8.113e+2
  ```

  Every parametric fit unbiased, every fit with a smooth armed, on identical
  data. `λ_min` IS `O(n)` — `0.211 → 0.591 → 1.989` across `n = 300 → 900 →
  3000` — but with a per-observation constant of `≈ 7e-4`, because the
  least-resolved direction of a `k`-dimensional spline is barely resolved *by
  construction*; that is why it is penalized. The premise holds only for
  `n ≫ 1/c_k`, and `1/c_k` grows with `k` (`0.591 → 0.458 → 0.253` at fixed
  `n = 900` as `k` goes `5 → 8 → 12`).

  The distinction #715 derives is the one that was missing: a direction `v` is
  beyond `λ`'s reach only when `S v = 0`, because `(H + S_λ)v = Hv + λSv`. Where
  `Sv ≠ 0` the smoothing parameter supplies the missing curvature and the
  direction is identified — by the prior the model already has. The certificate
  now forms `H + S_λ` at the λ the mode was certified at, through
  `multinomial_joint_penalty_operator`, the fit's ONE assembly of `S_λ`, so the
  arming decision and the published penalty cannot describe different priors. No
  threshold moves: the Frobenius-normalized `S` makes `λS` directly comparable to
  data Fisher information, which `MULTINOMIAL_FORMULA_FISHER_INFO_PER_OBS`
  already asserts. The refusal reports both spectra, because "the data do not
  determine this direction" and "and no λ repairs it" are different statements
  and the verdict rests on the second.

  **The deciding threshold was `16`, not `1`, and that is the second half of the
  defect — the measurement is what forced it.** With `H + S_λ` in place the
  fixture's deciding eigenvalue moved `0.405 → 5.508`, a factor of `13.6`, and
  the fit **still armed**, at gate weight `0.783`:

  ```text
    identifiable-span PENALIZED curvature H+S_lambda is under-identified at the
    certified mode: lambda_min=5.5076e0, lambda_max=8.8358e2, ratio=6.233e-3,
    Jeffreys gate weight=7.8336e-1
    (likelihood alone: lambda_min=4.0514e-1, lambda_max=1.5754e2)
  ```

  Five and a half observations' worth of curvature in the worst-determined
  direction, still called separated — because the verdict was taken on
  `JointJeffreysPlan::is_active`, which is `gate_weight != 0` and therefore
  boundaries where the C¹ ramp reaches exactly zero,
  `CONDITIONING_GATE_ABSOLUTE_CLEAR = 16`. That band exists so `Φ(ρ)` stays C¹ as
  `β̂(ρ)` carries the spectrum across the boundary — a binary gate makes the
  outer objective jump, which is the #787 "outer smoothing did not converge"
  regression — and it is generous FOR THAT REASON, not because a direction
  holding fifteen observations' worth of curvature is unidentified. `weight != 0`
  is the support of a smoothing device, and it was choosing an estimand.

  `JointJeffreysPlan::is_under_identified()` is the gate's derived predicate
  instead — below one observation-equivalent, or below the relative knot —
  expressed as `conditioning_gate_weight == 1` so one arithmetic authority
  decides both the weight and the verdict. `is_active` is untouched and still
  governs the term's own contribution, where the weight IS the answer; only the
  multinomial's conditional engagement, which re-solves the whole fit against a
  different objective, takes the new one. The two halves are independent and the
  fixture needs both: with `H` alone `λ_min = 0.405 < 1`, so the derived
  predicate would arm anyway; with `is_active` the penalized `5.508` arms anyway.

  **Rejected**: widening the absolute knot (derived, and not the thing that is
  wrong); restricting the Jeffreys BASIS back to `ker(S)` (deliberately widened
  to the full span for the BMS-probit near-separation on a penalized direction,
  where the term is `O(1/n)` by design — it is the arming DECISION, not the
  basis, that cannot afford a false positive); touching the universal always-on
  term's internal gate (same predicate, but a skip optimisation rather than a
  verdict).

  **`MULTINOMIAL_UNBIASED_PROBE_OUTER_MAX_ITER` is deleted in the same change,
  and the measurement says it was not the cause.** The unbiased probe ran under
  `outer_max_iter.min(20)` — a ceiling introduced as a bare one-line
  `perf(#1082)` commit with no test and no measurement — while every `Err` from
  it was routed on as separation evidence. SPEC forbids minting a fit from an
  exhausted budget, so a search stopped by that ceiling returns
  `RemlDidNotConverge`, and the ceiling could convert "this probe was still
  descending when I stopped it" into "the data separate". Run at both budgets on
  the same non-separating fixture it in fact decided nothing — the probe
  certifies either way, and the reported spectra agree to four digits — which is
  what falsified the first hypothesis on this issue. It goes anyway, because the
  only runs it can shorten are the ones where it stops a still-descending search,
  i.e. exactly the runs whose verdict it changes: its entire benefit is
  co-extensive with the misdecision. SPEC: *"Wall-clock time budgets and
  deadlines are never allowed, except in tests. In general, do not paper over
  solver issues."* The dead helper written for that branch and never wired
  (`multinomial_formula_unresolved_probe_separation_evidence`, kept in a test
  module under the note that "the production routing that would consume this is
  not currently wired") goes with it.

  **The fit now records which estimand it published.** `separation_evidence` is
  `None` for the unbiased penalized-REML mode and carries the certificate itself
  when the proper prior was armed. The two branches are different estimands, and
  the decision previously existed only in a `log::info!` line the caller never
  sees — while the CLI rejects `--firth` on this family with "the stabilizer is
  armed automatically". A user told the decision is automatic is owed the
  decision, and the CLI summary now reports it.

- **The refinement tolerance is now DERIVED from the candidate set instead of
  being a fixed fraction of the residual, because a fixed fraction charges
  nothing for the set's width (#2759).** #2759's first half closed a two-sided
  bracket on the level-`(L+1)` gain and established that the cascade fixtures
  refusing at the rank-maximal design have the exact remaining
  penalized-objective decrease bounded away from `REFINE_TOL·rss_pen` from
  BELOW — so those refusals were the cascade's remaining gain, not the
  certificate's conservatism. What it left open is whether that decrease is the
  thing the criterion should be reading. It is not.

  At the `smoothness_ceiling_...` refusal the candidate level is **32790
  columns against 5997 identifiable directions** on `n = 6000`. Past the data's
  identifiable rank those candidates are redundant against the sample's own row
  space; what they buy is penalty dilution and noise capacity, and
  `1e-3·rss_pen` cannot tell that from discretization bias because it never
  looks at the set.

  The missing charge is the set's own Occam factor — the log-determinant of the
  SAME Schur complement the gain is a quadratic form in:

  ```text
      gain  = gᵀS⁻¹g,   occam = log det(S/(λd)),   S = X₂ᵀW(I − H)X₂ + λd·I
      2·evidence = dof·log(rss_pen/rss_pen_refined) − occam
  ```

  The second line is an IDENTITY at the profiled σ̂², where the `rss_pen/σ̂² =
  dof` quadratic cancels on both sides — so one more level is warranted exactly
  when `gain > rss_pen·(1 − e^{−occam/dof})`. That break-even gain is the
  tolerance, `REFINE_TOL` is deleted, and the cascade has no tolerance constant
  left.

  **Both numbers come from ONE fixed-λ evaluation of the design with the
  complete candidate level appended**, and that evaluation is available past
  every capacity budget the automatic route enforces.
  `CERTIFIED_SPECTRUM_MAX` bounds the λ-independent Schur eigendecomposition
  the score SEARCH is certified in; `n − nullity` bounds the rank that search
  needs a stationary point in. A single evaluation at a fixed λ needs neither —
  only a factorization, which the sparse route supplies far wider. So the
  question "does one more level explain the data better?" has an exact answer
  exactly where the cascade used to have only a bound.

  The bracket is kept as the SCREEN, and it is not a different instrument: read
  the gain from its LOWER end and the Occam factor from Hadamard on
  `S ⪯ diag(X₂ᵀWX₂) + λd` (a reduction over the Jacobi preconditioner the
  bracket already forms), and both readings understate the evidence, so a
  positive evidence there PROVES the level warranted without building anything.
  It settles the positive side only — a fit is never minted on it — which is
  what lets it carry the memory-boundary refusal at `n = 525_000` without
  materializing a design wider than the budget that refusal exists to protect.

  Measured, `n = 6000`, the fixture's own ladder:

  ```text
   rung centers  cand   rss_pen  1e-3·rss  gain_hi   occam   break-even  Δ logL   rmse -> refined
    0        57     -   2.266e2  2.266e-1  1.985e3  5.17e2   1.872e1     +6293    0.1896 -> 0.0562
    1       189     -   2.204e1  2.204e-2  8.452e1  1.55e3   5.011e0     +3240    0.0553 -> 0.0220
    2       655     -   5.514e0  5.514e-3  3.517e0  2.21e3   1.697e0     + 654    0.0220 -> 0.0156
    3      2166     -   2.997e0  2.997e-3  6.205e-1 1.33e3   5.962e-1    +  16.7  0.0157 -> 0.0155
    4      5997 32790   2.184e0  2.184e-3  1.055e-1 3.23e2   1.146e-1    -  13.2  0.015583 -> 0.015589
  ```

  Rung 4 is the refusal in the issue body. The fixed bar is 48x below the gain
  and demands another level; the restricted likelihood says the finer prior is
  13.2 nats WORSE; and the held-out RMSE against the planted truth agrees with
  the likelihood, not with the bar — refining makes it worse. `n = 2000`
  reproduces the same turnover at its own rank-maximal rung (−18.7 nats,
  0.03087 → 0.03129).

  **The obvious objection, run and killed.** The comparison is at the
  incumbent's λ, so the refined design might win it back at a λ of its own.
  Swept `log λ ± 1, 2, 3` on the refined design at every rung: at both turnover
  rungs the best λ IS the incumbent's, to the printed digit. Structurally that
  is what has to happen — at the turnover the extra columns are redundant, so
  the score surface barely moves and its optimum does not — and the sweep is
  now a gate rather than an observation.

  **The identifiability frontier is where the cascade STOPS, not where it
  refuses.** Two fixtures asserted a refusal there and now mint the
  rank-maximal fit, for the same reason: at `n = 240` the level proposes 504
  penalized modes against 237 identifiable directions and is worth −0.30 nats;
  at `n = 2000`, 6968 against 1997. Neither exists to test the certificate —
  both exist to keep the automatic route out of a rank-deficient score search
  whose cost is exponential in the subdivision depth — and stopping at the
  frontier keeps it out just as a refusal did, while returning a usable fit.
  The measured boundaries are asserted exactly as before; only the verdict
  taken on them moved. The typed `Underresolved` refusal is not orphaned: it is
  the MEMORY boundary, where the design is capped BELOW the identifiable rank
  and the levels it cannot reach do pay for themselves
  (`past_cliff_...`, `n = 525_000`, unchanged).
  `cascade_matches_or_beats_dense_duchon_on_truth_recovery` reports RMSE
  0.02781 against the dense comparator's 0.03018, so the shallower stopping
  point costs no accuracy.

  Verified: the Occam term read off the two restricted log-likelihoods is
  checked against the candidate Schur log-determinant formed DENSELY, one
  column at a time through the same matrix-free operator, over a λ sweep — one
  side is two profiled REML evaluations, the other is `m₂` cascade solves, and
  they share nothing but the arithmetic they must agree on. The comparison also
  differences a `fit_reml` restricted likelihood (normalized through the
  certified Schur eigenbasis) against a `fit_at` one (a factorization at that
  λ), at O(1) nats while each side is O(10³), so the two routes agreeing is a
  premise and is now charged on both width regimes.

  **All four fixtures are back on the route a caller takes.** This issue's
  acceptance was "either it certifies, or its refusal is shown to be a true
  statement about the data at that `n`". Its first half took the second branch
  and moved three of them off `fit_residual_cascade` — a serialization gate
  going red because the cascade has remaining gain is not measuring
  serialization — which was the honest reading while the criterion was what it
  was. With the tolerance derived from the candidate set they take the FIRST
  branch: `cascade_state_rejects_corruption`,
  `cascade_state_roundtrip_reproduces_mean_and_variance` and the benign arm of
  `quasi_uniformity_guard_rejects_degenerate_metric_keeps_benign` all fit
  end-to-end again. `cargo test -p gam --test misc residual_cascade` is 26 of
  26, from 25 of 26 with three fixtures held off the route; `cargo test
  -p gam-solve --lib residual_cascade` is 29 of 29.

  A candidate column whose bump covers no observation is exactly zero, and its
  `λd` diagonal cancels between `log|A|` and `log|λD|₊`, so it contributes
  nothing to the gain, nothing to the Occam factor, and nothing to the
  restricted likelihood. Dropping such columns from the design the comparison is
  built on is therefore an identity, and it is worth taking: 4976 of 7176
  candidates are structurally empty at level 7 on 240 rows.

  One latent defect fell out of it: `NextLevelPlan::exhausted` hard-coded
  `extends_last: false`. That was invisible while the flag was read only on the
  refine path — an exhausted plan is never refined into — and fatal the moment
  the comparison materializes the candidate set FROM the plan, which is exactly
  what the capacity refusals need. It is now decided from the radius, before
  any candidate set exists, and carried by every outcome.

- **The Murphy–Topel correction now exists for a `GlobalEmpirical` second-stage
  latent measure, and the refusal it replaces was resting on a false
  obstruction (#2484).** A BMS fit whose conditional location-scale calibration
  fires and whose calibrated residual `ζ` then fails the standard-normal
  adequacy gate selects an empirical latent measure built from `ζ` itself. That
  pair used to withhold the coefficient covariance, on the argument that the
  generated-regressor correction needs a per-row mixed derivative and `ℓ_i`
  depends on `ζ_i` twice — directly, and through a grid every other `ζ_j`
  helped build — so "the honest object is a full `n × n` sensitivity" and "the
  measure is itself estimated from the same data".

  The first half is a factorization, not a dense object; the second is the
  chain rule, not a violated assumption.

  `build_empirical_z_grid` cuts bins by cumulative **weight**, so for a fixed
  sort order the bin allocation `α` is *exactly* constant in `ζ` and the grid
  weights carry no `ζ`-sensitivity at all — only the `m ≈ 32` node VALUES move.
  And Murphy–Topel conditions on the data: given `z`, the measure is a
  deterministic function of `θ₁`, which is precisely what the correction
  propagates. So the total derivative splits into a direct channel and a
  rank-`m` cross-row channel,

  ```text
      d score_β/d ζ_j = s_j + Σ_b u_b·D_{bj}        ⇒        S_eff = S + Dᵀ·U_Qᵀ
  ```

  and the seam substitutes `S_eff` for `S`. `generated_regressor_correction`,
  `build_zeta_theta1_jacobian`, `beta_theta1_sensitivity` and the
  `(V_β G) V₁ (V_β G)ᵀ` congruence are untouched, and PSD-ness is preserved for
  free.

  Neither channel is the closed-form kernel's. The empirical row is
  `−w·logΦ(σ·(a(m,g) + s·g·ζ_i))` around an implicitly solved intercept `a`
  rather than `q·√(1+(s·g)²) + s·g·ζ_i`, so reusing
  `rigid_standard_normal_score_zeta_sensitivity` for the DIRECT half would have
  been a subtler wrong answer than refusing. Both come from one pass over the
  rows, sharing the per-row intercept solve, with the node derivatives from the
  same calibration root the row jet lifts:

  ```text
      a_x_b       = −s·g·π_b·φ(η_b)/Ψ₁
      a_{m,x_b}   = −a_m·(dΨ₁/dx_b)/Ψ₁
      a_{g,x_b}   = −[dΞ₁/dx_b + a_g·(dΨ₁/dx_b)]/Ψ₁
  ```

  `Σ_b a_x_b = −s·g` (a uniform node shift must be absorbed exactly by the
  intercept) and `a_x_b ≡ 0` at `g = 0` (a fit with no slope cannot see the
  latent axis) are the identities that pin the sign and scale.

  **Rejected: seeding each node as a third jet axis** through the existing
  `filtered_implicit_solve_scalar` lift, which is the more mechanical route.
  It costs `O(n·m²)` lifts against the closed form's `O(n·m)`, on a path that
  runs at biobank `n`.

  `EmpiricalZGrid` and its `PartialEq` are untouched — it is the measure's
  identity and it is on the persistence wire. The allocation record rides on a
  fit-time-only `EmpiricalZGridBuild` returned by the one builder the fit
  itself uses, so the recorded `α` is the fill loop's own rather than a
  reconstruction.

  **What still withholds, and it now names the missing CHANNEL rather than the
  measure:** a score-warp / link-deviation block (the latent score enters
  through a basis as well as through the intercept, so the rigid node channel
  does not describe the row); a `local-empirical` measure (per-row grids, only
  produced by deserializing a saved model, so there is no fit-time allocation);
  and data on which the compression is genuinely non-differentiable — a tied
  `ζ` group that a bin boundary cuts, where the left and right derivatives of
  the nodes differ. That certificate is narrower than "no ties": a tied group
  entirely inside one bin is order-invariant and is not refused.
  `CovarianceDeclined::BmsGeneratedRegressorLatentMeasureNotStandardNormal`
  gains a `#[serde(default)] unavailable_channel`, so older payloads still
  deserialize.

  Verified against difference quotients of PRODUCTION code, never a
  reimplementation, and separately against a second implementation of the
  DERIVATION in another language
  (`scripts/probe_2484_empirical_measure_sensitivity.py`) — the two catch
  different things, since a formula transcribed consistently-but-wrongly into
  both the code and its own test survives the first check and not the second.

  The acceptance gate is the total `∂²(log L)/∂β∂ζ_j` against a double central
  difference of the production log-likelihood with the grid REBUILT at every
  perturbed `ζ` — blind to how the channels are split, so it fails alike on an
  IFT sign error, a missing cross-row term, or a wrong `1/sd`. Below it:
  allocation mass conservation on both margins; `D` against a central FD of the
  production builder, including a row heavier than the per-bin target (it lands
  in THREE bins — there is no two-entries-per-row bound) and a zero-weight row
  (exactly zero sensitivity); the two projection identities `D·1 = 0` and
  `D·ζ = 0` as exact assertions; the tie certificate firing on a cut tie and not
  on a contained one; and the assembled correction being symmetric, PSD, and
  strictly widening.

  ```
  cargo test -p gam-models --lib empirical_measure_2484
    test result: ok. 10 passed; 0 failed

  scripts/probe_2484_empirical_measure_sensitivity.py
    D max abs err vs FD:                 1.63e-10
    total mixed derivative max rel err:  1.28e-07
    bins=3 (tie inside one bin):  |right − left| = 6.66e-09   differentiable
    bins=4 (a boundary cuts it):  |right − left| = 5.39e-01   TWO-SIDED
  ```

  **The witnesses.** The three `..._starts_outer_solver` fixtures gam#2484 was
  filed against:

  ```
  binary_outcome_shape_bms_shared_matern_prs_pc_confound_starts_outer_solver ..... ok
  production_like_binary_outcome_shared_matern_centers10_confound_starts_outer_solver ... ok
  production_like_binary_outcome_shared_matern_learned_kappa_starts_outer_solver ....... FAILED
  ```

  The third fails on `INDEFINITE CURVATURE AT INTERIOR OPTIMUM` (`|g| = 1.241e-2`
  against `bound = 3.850e-2`, `hessian_psd = NO`) — the outer-stationarity
  cluster, which is not this seam and never reaches it. That is the state
  gam#2484's own 2026-08-01 measurement recorded: *"masked, not resolved … it is
  now blocked one stage earlier."* Its calibrated residual still selects
  `global-empirical`, so it would be corrected if the outer solve certified.
  Both witnesses that reach the seam pass.

  `tests/bms_covariance_declined_2718.rs` runs in **2.3 s** (4 arms) against
  22 min before, with strictly more coverage: the end-to-end withholding witness
  moved to the classifier, where all four arms are decided with no fit in the
  loop, and the wire contract is asserted on the payload — including a payload
  written before the channel field existed, which must still load with the
  channel empty rather than fail.

  **Subsystem sweep.** `cargo test -p gam-models --lib bms::` — **297 passed,
  2 failed**, and neither failure is this change:
  `bms::gradient_paths::jet_tower_oracle_tests::rigid_third_and_fourth_full_shares_one_tower_bit_identical`
  (a last-ulp tower difference) and
  `bms::tests::bernoulli_batched_outer_gradient_matches_hypercoord_path_for_rho_and_psi`
  (`psi[1]` at `rel = 4.132e-3`). Both are already recorded in
  `bench/gha_results/rust-test-suite/MASTER_FAILURES.md`, written by CI run
  31291341087 in `e146df43a`, which `git merge-base --is-ancestor` confirms
  predates the first commit of this work. Neither path is touched here: the only
  production edit outside the BMS covariance seam is the `clamp` fix below, and
  it is bit-identical for `n >= 1024` (`rows.min(1024) == 1024`) and converts a
  panic into a value below it — it cannot change a number that previously
  computed.

  **What the channel is worth, stated honestly.** `|cross| / |direct|` is a
  property of the sensitivity MATRIX; a user sees a standard error, three
  contractions downstream. The correction as a whole moves the SE by
  **1.06x–2.53x** against the naive covariance — that is what publishing the
  naive matrix would have cost. The CROSS-ROW half of it moves the SE by
  **1.2e-5 to 9.0e-3** relative, and what it scales with is the LOGSLOPE rather
  than the grid size. Small enough that a direct-only implementation would pass
  casual inspection, which is an argument for being exact and not an argument
  that the channel is optional.

- **A composed monotone warp was a function with a CORNER, and the Firth term
  put that corner into the objective (#2695).** `create_ispline_dense` is
  constant outside its knot hull `[left, right]`, and says so; `a3304985f` made
  the reported derivative agree with that value by zeroing it strictly outside.
  Both halves are right, and together they name what is wrong: a
  constant-extended I-spline is continuous with a **corner** at each hull edge —
  `I_j` joins, `I'_j` steps from its interior one-sided slope straight to `0`.

  A corner in a shape basis on fixed data is harmless; the evaluation point
  never moves. A corner in a *warp* is not. The warp is composed onto the
  model's own index, `q = q₀ + Σ_j βw_j·I_j(q₀)` with `q₀ = −η_t·e^{−η_ls}`, so
  `q₀` moves with β while the hull is frozen at the seed `q₀`, and the basis is
  evaluated on both sides of the edge inside a single inner solve. Two of the
  chain-rule channels carry `I'_j` with **no `βw` factor at all**,

  ```text
      ∂²q/∂β_thr ∂βw_j = I'_j(q₀)·∂q₀/∂β_thr        ∂q̇/∂βw_j = I'_j(q₀)·r
  ```

  so the observed information jumps by `O(1)` across the edge **even with the
  warp switched off** — and `Φ = ½ Σ g(λ(Z_JᵀHZ_J))` is part of the inner
  objective the trust region accepts on. The objective is therefore
  discontinuous, and `actual/predicted` cannot approach `1` at any step size.

  Measured on `survival_location_scale_saved_fit_preserves_linkwiggle_metadata`,
  cycle 13, five attempts from one base point along a bit-identical direction:

  | ‖δ‖ | pred | actual | `d(−ℓ+½βᵀSβ)` | `dΦ` | max `dH` | at |
  |---|---|---|---|---|---|---|
  | 4.885e-5 | 1.760e-4 | −5.5209e-1 | 1.1404e-4 | −5.5220e-1 | 1.00001 | (5,5) |
  | 1.221e-5 | 4.400e-5 | −5.5213e-1 | 2.8512e-5 | −5.5216e-1 | 0.99999 | (5,5) |
  | 3.053e-6 | 1.100e-5 | −5.5214e-1 | 7.1280e-6 | −5.5215e-1 | 0.99999 | (5,5) |
  | 7.633e-7 | 2.750e-6 | **+2.7502e-6** | 1.7820e-6 | +9.6818e-7 | 9.16e-6 | (0,1) |

  The `−ℓ + ½βᵀSβ` half tracks its own linear model to six digits at every
  attempt including the three that cross, so the likelihood is not the defect;
  the whole error is `Φ`, as a jump. `dH` is ONE entry and its size is `1.0000`.
  Against the frozen hull edge `right = +7.261500860e-1`, one row's exit `q₀`
  sits `1.3e-7` outside it at the third attempt and `3.5e-6` inside it at the
  fourth, and `I'_3` steps `9.9999823e-1 → 0` between them. The value is
  continuous there (`[1,1,1,0.99999646] → [1,1,1,1]`).

  **Rejected: raise the spline degree.** `w''` is indeed piecewise constant at
  degree 2, but a four-arm A/B on the witness has `degree = 2/3/4/5` all still
  refusing while `degree = 0` (no `linkwiggle`) fits. The corner belongs to the
  extrapolation convention, not the polynomial degree, which is exactly why no
  degree touches it.

  `monotone_wiggle_basis_with_derivative_order` is now the single definition of
  the warp on all of `ℝ`, with `x̄ = clamp(x, left, right)`:

  ```text
      I_j(x)    = I_j(x̄) + I'_j(x̄)·(x − x̄)
      I'_j(x)   = I'_j(x̄)
      I⁽ᵏ⁾_j(x) = interior value inside the hull, 0 outside        (k ≥ 2)
  ```

  and `monotone_wiggle_basis_from_knots` routes through it, so the fit design,
  the derivative stack, prediction and inference all read one function. The
  interior is bitwise unchanged, so no fit whose rows stay inside the hull
  moves. The tail is the basis's own first-order expansion about the join, so
  the two halves are one differentiable function rather than two that meet. An
  I-spline is non-decreasing, so both tails have non-negative constant slope and
  `βw ≥ 0` still gives a monotone warp on all of `ℝ`.

  **Behaviour change worth stating:** the `[0, 1]` RANGE of the basis is given
  up outside the hull, and with it the old "the warp does nothing beyond the
  observed range" convention — a `linkwiggle` term now continues at its boundary
  slope instead of flattening. That range is precisely why
  `create_ispline_dense` saturates, and its own doc already directs callers who
  need otherwise to *"clamp inputs and add their own extrapolation
  correction"*; this is that correction, at the caller, and it is the standard
  convention for a spline *transformation* as opposed to a spline *shape*
  (restricted / linear-tail splines, as in flexible parametric survival models).
  Ordinary I-spline *smooths* are untouched — only the monotone-warp entry
  points route through the tail.

  Orders `k ≥ 2` are zero on the tail, so `I''_j` is still discontinuous at the
  join; it reaches the objective only as `m₂ = Σ_j βw_j·I''_j`, i.e. weighted by
  the warp amplitude, exactly as it already is at every interior knot of a
  degree-2 basis. The hull edge is therefore no rougher than a knot, which is
  the most a finite-degree spline can offer.

- **One sentinel, one resolver: the measure-jet auto range is now screened
  against the response on every standard-fit branch (#2750).**
  `length_scale == 0.0` is an unresolved request, and it had TWO resolvers — a
  pure-geometry rule inside the basis builder (the median nearest-node spacing)
  and the #2750 response screen — with which one a model got decided by which
  branch of the standard-fit dispatch it happened to take. The screen ran inside
  `fit_term_collectionwith_spatial_length_scale_optimization`, so a collection
  carrying a latent coordinate or coefficient groups was resolved by geometry
  alone.

  That is not a tuning difference between branches. `ℓ` decides WHICH span the
  representers occupy and a smoothing parameter cannot move a span, so the two
  resolvers produce different models, and the measured gap between them on the
  fixtures that do reach the screen is a factor of `1.6`–`13` in held-out error.

  The screen now runs once at the top of `fit_standard_model`, before the
  three-way dispatch and before the Tweedie-`p` profile, so every branch passes
  it. It is idempotent by construction — it only fires on the `0.0` sentinel — so
  the call still inside the spatial driver (reached directly by other drivers and
  by tests) is a no-op afterwards, and the #1762 Firth retry re-enters with the
  range already resolved instead of screening a second time.

- **The measure-jet range screen's downward walk was bounded by a guard that
  could not fire (#2750).** The screen walks geometrically off either end of the
  realized scale band while that end is still the incumbent, and its own comment
  said the ends were the bracket's — "so the walk introduces no length of its
  own". The upward cap was enforced. The downward one was

  ```rust
  if !upward && next_ln < floor_ln - bracket.log_step * (scored.len() as f64) { break; }
  ```

  and it recedes by one log step for every node the walk pushes, exactly as fast
  as `next_ln` descends — so the comparison is false at every iteration for any
  bracket with two or more nodes. The only stops left were "the criterion stopped
  improving" and "the basis refused to build".

  That matters now rather than before, because the outer search's own `ln ℓ`
  window is floored at the same node spacing: a screen that seeded below it would
  be widened INTO by the #2454 incumbent-containment rule, reintroducing exactly
  the region the floor excludes.

  The walk is upward-only, which is what the coordinate says. The band's bottom
  node IS the floor — the median nearest-node spacing — and it is already scored,
  so there is nowhere below it to walk to.

- **Farthest-point knot selection compared a squared LENGTH against the number
  one, so it stopped being scale-equivariant below unit radius (#2750).**
  `select_thin_plate_knots` is the shared center selector for every radial
  spatial smooth — `thinplate`, `duchon`, `matern` and `mjs` all reach it — and
  its maximin/centroid tie tolerance was

  ```rust
  let knot_scale2 = dist2_to_centroid.iter().copied().fold(0.0_f64, f64::max).max(1.0);
  let tie_tol = KNOT_MAXIMIN_TIE_REL_TOL * knot_scale2;
  ```

  The constant's own doc states the requirement: it must sit "several orders of
  magnitude above [the `ε·‖x‖²` round-off floor] yet **far below any genuine gap
  between geometrically-distinct candidates**". `.max(1.0)` substitutes `1` for
  `‖x‖²` for every cloud smaller than unit radius, which breaks that second half
  outright. Measured on a 240-row 1-D chart of half-width `5.2e-4`: squared
  radius `2.7e-7`, genuine maximin gap between neighbouring candidates `~6e-10`,
  floored tolerance `1e-9` — **the tolerance is larger than the gap it had to sit
  far below**. Every candidate ties, the invariant support-distance profile
  decides a selection it was only meant to referee, and the knots come out
  different from the ones the same configuration gets in different units.

  Downstream that is not cosmetic. The knots ARE the measure-jet quadrature
  seeds, so the median nearest-node spacing moves, and with it the auto
  representer range, the scale band, and the `ln ℓ` search window below.

  The floor is removed rather than replaced. With `tie_tol = 1e-9·‖x‖²` every
  ingredient of the comparison scales as `c²` under an isotropic rescale, so the
  selection commutes with it exactly. The degenerate end is unchanged: a
  coincident cloud has `‖x‖² = 0` and now `tie_tol = 0`, but every squared
  distance there is exactly zero, so the same candidates tie as before.

- **A measure-jet term's `ln ℓ` search box was a chosen absolute interval, and
  `ℓ` is a length in the data's own chart (#2750).** Every measure-jet ψ
  coordinate got the same kind of box:

  ```rust
  pub const MEASURE_JET_PSI_LN_LENGTH_SCALE_BOUNDS: (f64, f64) =
      (-6.907755278982137, 4.605170185988092);   // ln[1e-3, 1e2]
  ```

  and the doc said why: *"Absolute (not seed-relative) so the bound producer
  needs no data view, matching the other dial boxes."* For the two PENALTY dials
  that is right — `α` and `ln τ` are dimensionless and no geometry bounds them.
  For `ln ℓ` it is not: `ℓ` decides which span the representers occupy, it is a
  LENGTH in the frame the basis is realized in, and both of its walls are the
  same measured length — the median nearest-node spacing `s`, which is also the
  auto range and the scale band's floor — read at the two ranges where the kernel
  stops saying anything about the pair it separates:

  * **floor `ℓ = s`**: neighbouring representers overlap at exactly `exp(−1/2)`;
    below it they stop overlapping, the design degenerates from a partition of
    unity into a bump-per-node indicator, and rows between nodes fall outside
    every representer's support;
  * **ceiling `ℓ = s/√(2√ε)`**: that same pair's kernel value has come within
    `√ε` of 1, so it is no longer distinguishable from a coincident pair in the
    arithmetic the chart is built in. `√ε` is the chart's own bar — the same
    half-mantissa `condition_representer_section` spends.

  So the window is `[ln s, ln s − ½ln(2√ε)]`: it TRANSLATES with the chart and
  its width is `8.664`, a pure function of `f64::EPSILON` rather than a number
  anybody picked.

  The measured harm of the absolute box was the first trial step. On
  `measure_jet_perf_parity` the first `ln ℓ` step is `−0.693`, landing at
  `ℓ = 0.488` against a floor of `0.5145` — **outside the term's own geometry** —
  and it is rejected, each rejection a full design realization; the search then
  excursions to `ℓ = 0.34`, a range `1.5×` below the node spacing where the
  representers no longer overlap. Clamping to the derived window:

  ```text
                          outer evals   design realizations   wall (min of 3)
    before                    105                57                0.99 s
    after                      58                32                0.62 s
    matern(k=16) control       18                 1                0.52 s
  ```

  **The ceiling is deliberately NOT the node bounding-box diagonal.** That is
  where the response screen stops WALKING — a stopping rule for a search over
  nodes — and a first attempt used it as the box. It railed the outer search on
  three fixtures and refused their fits: the profiled criterion genuinely prefers
  a range at or above the node diameter, because as `ℓ` grows the
  gauge-quotiented representer span tends to a polynomial one, which is the right
  basis for a smooth target. A long range is a legitimate model, so the upper end
  has to be a feasibility statement and nothing weaker.

  The regression test is an INVARIANCE rather than a level: the window is made of
  lengths, so rescaling the chart by `c` must shift both ends by exactly `ln c`
  and leave the width fixed. An absolute window fails both halves by
  construction — the same node configuration in metres and in millimetres would
  be handed two different search problems, and at `c = 10³` the seed moves `6.9`
  log units inside a window only `11.5` wide.

  `[KAPPA-PHASE]` records now carry the SIGNED ψ coordinates beside `‖ψ‖`.
  A norm is the right summary for a multi-axis anisotropy block and the wrong
  one for a single signed coordinate: `‖ψ‖ = 0.718` is consistent with a trial
  at `ℓ = 2.05` and with one at `ℓ = 0.49`, and only the second is outside the
  window — which is exactly the distinction the diagnosis above turns on.

- **A curvature refusal is now adjudicated BY THE CRITERION instead of asserted
  by the matrix (#2612).** `negative_curvature_escape_point` already stepped the
  criterion along the reported minimum eigenvector, and the code threw half of
  its verdict away: it returned `Option<Array1<f64>>`, so "a strictly-descending
  feasible point exists" arrived as `Some` while "no descending trial exists
  anywhere I looked" arrived as `None` — bit-identical to "the escape was never
  runnable" — after which the refusal proceeded on the matrix's word alone.

  That second case is not an absence of evidence. Evaluating the objective at
  trial points is not a finite difference (SPEC 2 is untouched); it is the
  criterion answering the exact question the Hessian claimed to answer, and a
  direction along which the criterion does not fall is not a descent direction of
  that criterion. #2665 is the same defect from the other side: an analytic
  `λ_min = −1721.5` whose objective curvature along its OWN eigenvector is
  `+121.6`. No resolution bound catches that — the matrix there is not
  imprecise, it is wrong.

  The step ladder also stopped in the wrong place:

  ```rust
  const ESCAPE_STEP_SCALES: [f64; 5] = [1.0, 0.5, 0.25, 0.125, 0.0625];
  ```

  `0.0625` because five entries had been written down, so a claim whose descent
  only appears below that step read identically to a claim with no descent at
  all. At a stationary point the claim's own quadratic model predicts
  `½|λ_min|α²`; once that reaches the criterion's resolution the claim predicts
  nothing the criterion can represent and no smaller step can falsify it. The
  ladder now runs `1 → α_min = sqrt(2·objective_resolution/|λ_min|)` by halving
  in both signs, with the resolution being the same
  `rel_cost_tolerance`-anchored quantity the rail and cost-stall machinery
  already spend. No constant is chosen; the only other stop is `f64::EPSILON`,
  where halving stops changing `ρ + αv` at all.

  `SaddleAdjudication::{Descended, Contradicted, Declined}` replaces the
  `Option` (`probed == 0` is `Declined` — nothing evaluated, nothing falsified),
  and `CurvatureEvidence::CriterionContradicted` records the withdrawn verdict.
  It is deliberately **not** `Measured { psd: true }`: nothing established that
  the point is a minimum, only that this matrix's negative direction has no
  operational content. `psd()` stays `None`, so the published `hessian_psd`
  contract (`null | true | false`) is unchanged.

  Measured at the penguins terminal ρ with the Jeffreys term armed, which is
  what put this on the table and also killed the first hypothesis about it:

  ```text
    v'Hv (analytic)    = -6.709810e-5
    v'J(g)v (measured) = -9.248844e-5    stable to 5 digits over h = 1e-4 .. 1e-2
    gap/|analytic|     =  3.784e-1
    max_k |g_k| over the judged coordinates = 9.6243e-4
  ```

  The sign agrees, so that negative curvature is real; and `|λ_min|` sits 14×
  INSIDE its own gradient floor, so the curvature conjunct does not refuse there
  at all. The `3.784e-1` gap is the omitted `D²_β H_Φ[−v_l, −v_k]` term measured
  at production scale — it exceeds the `0.25` bar the armed gate applies to
  `λ_min` on its own seconds-scale fixture, which is now on the record rather
  than papered over.

- **Every wholly parametric multinomial fit was being refused for not having a
  penalty, and "no penalty" was the answer (#2612).** The posterior-mean
  predictive needs `S_λ` as an operator, because it evaluates the penalized
  log-posterior away from the mode. It read that operator off the
  influence-matrix reconstruction, which returns `None` on two conditions the
  penalty does not depend on:

  ```rust
  let joint_recon = fit.artifacts.joint_log_lambdas.as_ref().and_then(|jll| {
      let n_components = penalties_arc.len();
      if n_components == 0 { return None; }                    // unpenalized
      let hinv = fit.covariance_conditional.as_ref()
          .filter(|c| c.nrows() == expected_joint && ...)?;     // a DIFFERENT measurement
  ```

  `n_components == 0` means the model is *unpenalized*, so `S_λ` is the **zero
  operator** — a value, not an absence; and `H⁻¹` is a measurement of a different
  object, so conditioning the penalty's availability on it makes a covariance
  failure surface as a missing penalty. A hard refusal on `None` then converted
  both into no fit at all: `y ~ x1 + x2` — no smooths, hence no penalty
  components — stopped fitting, and both
  `quality_vs_statsmodels_multinomial` arms that use it went `GAM_ERROR`.

  `S_λ` is now measured in its own right from the family's equivariant specs and
  the selected `λ`, with the unpenalized case returning the zero operator it is.
  The specs are assembled ONCE (they materialize `n_specs` dense `(P·M)²`
  matrices) and the influence matrix and the published payload both read that one
  list, so they cannot describe different penalties — the property the previous
  two-site assembly asserted in a comment and only approximated. It refuses only
  on genuine inconsistencies: a `λ` vector that does not match the spec list, a
  spec of the wrong dimension, or `λ` reported for a model with nothing to
  multiply.

  Both regression bars are statements about the same operator from opposite
  sides, so a payload that was merely publishable could not pass both: the
  unpenalized arm must publish `S_λ = 0` **and** an influence matrix that is
  exactly `I` (which `F = I − H⁻¹S_λ` forces), and the penalized arm's published
  `S_λ` must reproduce its own published influence matrix through
  `H⁻¹S_λ = I − F`.

- **A curvature gate was refusing on numbers smaller than its own instrument's
  measured error, and nine `matern` benchmark scenarios died of it (#2748).**
  `invert_identified_rho_hessian`'s entire `‖δH‖₂` was
  `eigenpair_backward_error_bound` — the eigensolver's residual, which answers
  *"given this matrix, how wrong is σ?"*. The question at a criterion-resolution
  site is *"how wrong is this matrix?"*, and
  `gam-linalg/src/curvature_resolution.rs` already says in its own module doc
  that "a site that needs the second must not be handed the first".

  The second is measurable in situ, with no new tolerance. On the penalty map's
  certified invariance `T`, lifted to ρ, the criterion is exactly constant in
  `λ`, so `ρ''(0)_k = −t_k²` gives

  ```text
      T' H_rho T  -  T' diag(g_rho) T  =  0        EXACTLY, at every rho
  ```

  and whatever its residual returns is error and only error — in exactly the
  currency this gate spends, since it compares a Hessian eigenvalue against a
  gradient-built floor. Measured at the refusing ρ of
  `geo_disease_eas_matern_k6`:

  ```text
    eigensolver backward error                    8.342e-19
    |T'(H_rho - diag(g_rho))T|_2                  9.872e-8      <- eleven orders larger
    refused curvature                            -2.010e-8      <- INSIDE it, by 4.9x
  ```

  `CurvatureResolution::analytic_weyl_from_components` now takes several NAMED
  measured components and resolves to their maximum — each is a certified lower
  bound on `‖δH‖₂`, so the largest is the strongest available fact, and a sum
  would not be derived from anything. Three components are supplied: the
  eigensolver's backward error, the penalty-map invariance residual above, and
  the rho-Hessian's **symmetrization defect** `‖(H − Hᵀ)/2‖₂`, which is exactly
  zero for any twice-differentiable criterion and was being computed and thrown
  away by the `symmetrize_in_place` call that precedes the gate.

  Nothing was widened by hand. With one component the resolution is
  `analytic_weyl` bit for bit, so a fit whose penalty map has no invariance does
  not move by an ulp; #2665's `λ_min = −1.6e3` saddle is ten orders outside every
  measured component and still refuses.

  Two further defects in the same cluster, both "one channel, two verdicts":

  * the penalty-map Gram was accumulated naively over `m = block²` products, so
    its error was `m·ε·Σ|S_i S_j|` — three orders above the bar its own rank
    decision is taken at, and enough to make an exactly proportional penalty pair
    read as independent (measured: the same pair gave `1 − cos = 1.11e-16` in one
    cell and `4.80e-14` in another). Neumaier compensation removes the length
    dependence; the bar is untouched.
  * `try_exact_joint_spatial_length_scale_optimization` returned `None` both when
    the joint κ route could not be BUILT and when it ran, graded its own candidate
    against the shipped scalar-route score and correctly DECLINED it. The caller
    mapped both to `"spatial kappa optimization is unavailable"` and failed the
    whole fit, so a route that had just decided the incumbent was better had that
    decision converted into a fatal error. `JointSpatialKappaOutcome` now says
    which, and a decline ships the incumbent — which is what its own log line
    promises.

  Measured end to end, one scenario per run, on a wheel built from the checkout:
  `geo_disease_eas_matern_k6`, `geo_disease_eas_matern_k12` and
  `papuan_oce_matern_k12` go from red to green.

- **The multinomial posterior mean was being computed by a method that is not
  an approximation of it (#2612).** `predict_multinomial_formula` publishes
  `E[softmax(x'β) | data]`. It computed that by approximating the coefficient
  posterior with the Laplace Gaussian `N(β̂, H⁻¹)` and integrating `softmax`
  against it. The `O(n⁻¹)` correction to a posterior mean has a curvature half
  and a skewness half; integrating a nonlinear functional over the Gaussian
  keeps the first and drops the second. On a well-conditioned fit both are
  small and nobody notices. On a (quasi-)separated softmax they are neither
  small nor same-signed — the likelihood is flat toward more separation and
  steep away from it, so the true posterior is skewed toward larger `|η|` while
  the symmetric Gaussian puts half its mass where the likelihood has already
  excluded the coefficient — and `softmax`'s concavity turns that misplaced
  mass into under-confidence at unchanged argmax. That is the penguin signature
  exactly: right class, flattened probabilities.

  The estimand is fine and stays. What replaces the method is the ratio form,
  which for a POSITIVE functional has the two Laplace errors cancel
  (`O(n⁻²)` rather than `O(n⁻¹)`), and which for a class probability is not a
  device but an identity — the extra row's likelihood factor IS the quantity
  being averaged:

  ```text
  E[p_c(x)]         = Z(D ∪ {(x, c)}) / Z(D)
  E[p_c(x)·p_d(x)]  = Z(D ∪ {(x, c), (x, d)}) / Z(D)
  ```

  Measured against an MCMC posterior on a `K = 3`, `p = 10` asymmetric
  quasi-separated fixture (an importance sampler was tried first and rejected:
  ESS ≈ 800 of 200000 in that skewed 20-dimensional posterior is a Monte Carlo
  error larger than the accuracy being certified):

  ```text
  max |Gaussian − exact| = 2.121e-1        max |ratio − exact| = 4.39e-3
  max |ratio E[p_c p_d] − exact|           = 3.89e-3
  ```

  and across basis widths, with the exact posterior tracking the plug-in at
  every width while only the Gaussian diverges — monotonically in the number of
  nearly-unconstrained directions, which is the amplifier that separates a
  four-smooth fixture from a two-parameter reduction:

  ```text
    p   max sd(eta)   plug-in   Gaussian     exact   Gaussian/exact
    2        8.08     0.05503   0.05753    0.05603       1.027
    8       15.88     0.05498   0.07039    0.05608       1.255
   16       26.49     0.05259   0.09452    0.05349       1.767
  ```

  Consequences worth stating:

  * **The saved model now carries its rows.** A Laplace summary cannot produce
    a posterior mean — that summary IS the quadratic model whose inadequacy is
    the defect — so `MultinomialSavedModel` stores the raw training design, the
    class index, the weights and the coupled joint penalty `S_λ`. This is the
    same choice `mgcv` makes in keeping the model frame with the fitted object,
    and it is required rather than optional: a payload without it cannot answer
    the question `predict` is asked.
  * **The Smolyak accuracy/level control is gone rather than ignored.** It
    existed because the old mechanism was a quadrature whose answer could be
    bought with more nodes. The new one's accuracy is a property of the
    expansion. What replaces it needs no configuring: `Σ_c E[p_c] = 1` is an
    identity of the estimand, so the computed sum's deviation from one is the
    error at that row, and a row past that tolerance is refused rather than
    published.
  * **The Gaussian-integrated quantity keeps its place and loses its name.**
    `MultinomialFitOutputs::predict_probabilities_with_se` is now
    `logistic_normal_softmax_moments`: the exact moments of `softmax` under a
    STATED Gaussian are a well-defined object with their own uses, and naming
    them for the Gaussian rather than for the posterior is what keeps the two
    from being confused again.

- **The multinomial outer-curvature gate was handed a count `#1587` stopped
  producing (#2612).** The exact-outer-curvature route was selected from
  `(K − 1) · n_penalties`. Since #1587, `equivariant_class_penalty_specs` emits
  one spec PER CLASS per penalty component whenever `K > 2`, so the four-smooth
  penguin fixture carries `8 × 3 = 24` outer coordinates and the gate was handed
  `2 × 8 = 16` — confirmed against the refusal's own `last_evaluated_rho`, which
  has 24 entries. The gate now reads
  `MultinomialFamily::joint_smoothing_dimension()`, and
  `MULTINOMIAL_EXACT_OUTER_HESSIAN_MAX_DIM` moves `16 → 24` without moving its
  calibration point: the same fixture, re-read with a corrected ruler. At
  `K = 3` the classification is unchanged (`3n ≤ 24` and `2n ≤ 16` are both
  `n ≤ 8` components).

- **`λ̂` is CHOSEN, and the smooth-term LR reference was pricing it as given
  (#2672).** The pooled size of this issue's own null-simulation grid, measured
  at main for the first time since `7dbd1dc43` landed: `size@.05 = 0.0962`
  against nominal `0.05` and a band of `±0.0449`. The figure on the record is
  `0.0272` — conservative by 1.8x. It is now anti-conservative by 1.9x, and
  `7dbd1dc43` is what moved it: it removed the Wood–Pya–Säfken reference-df
  inflation on the correct ground that the term is not in the fixed-`λ` null
  law, and nothing replaced what that inflation had been standing in for. The
  grid was not re-run.

  A Gaussian null with `σ` KNOWN — so Lawley's `Δε` is exactly zero and the
  reference is the only thing under test — reproduces it with none of gam's
  machinery, and names the mechanism: `corr(W, Σw) = 0.94–0.96`. REML picks `λ̂`
  on the same data that produced `W`, so the reference moves *with* the
  statistic, but not by enough.

  ```text
  conditional at λ̂       α = .20   .10    .05    .01
    n = 30,  k = 12          .2060 .1320  .0840  .0180
    n = 100, k = 12          .2160 .1100  .0580  .0140
    n = 200, k = 12          .1850 .1025  .0650  .0150
  ```

  **It is not a mean problem and must not be fixed as one.** On those runs
  `E[W]/E[Σw] = 2.4–2.5`, and dividing `W` by that ratio takes `size@.05` from
  `.087` to `.0000`. Two further candidates were measured and refused:
  restoring the WPS term is the `0.027` arm, and substituting the `λ̂`-corrected
  covariance `Vp` for `Vb` makes it *worse* (`.040 → .065`) on a `Var(ρ̂)` that
  measures `9e6` because the criterion is flat under the null — which is exactly
  the objection `7dbd1dc43` raised against that term, now confirmed.

  **What the statistic needs is the law of `W(λ̂)` with the selection replayed.**
  Diagonalize the term's fitted penalty against the Schur-complemented
  information: the pair is symmetric-definite, so one basis diagonalizes both,
  with generalized eigenvalues `ν_k = p_k/(1 − p_k)` read straight off the
  penalty shares already computed. In that basis the tested block is `q`
  independent standard normals, and BOTH the statistic and the criterion that
  selects `λ` are closed forms in them and in `t = λ/λ̂`:

  ```text
  W(t) = Σ_k (2f_k − f_k²) u_k²,          f_k = 1/(1 + t·ν_k)
  V(t) = ½ Σ_k u_k²·t·ν_k/(1 + t·ν_k) + ½ Σ_{ν_k>0} log((1 + t·ν_k)/(t·ν_k))
  ```

  So the whole selection — draw, choose `λ̂`, read `W` — is a function of `q`
  numbers, and the null law is generated with no design, no response and no
  refit, over the same `ρ` box the solver used. `t = 1` reproduces the
  conditional law exactly, so this is a strict generalization rather than a
  different reference.

  ```text
  selection-aware        α = .20   .10    .05    .01
    n = 30,  k = 12          .1940 .1160  .0560  .0120
    n = 100, k = 12          .2020 .0840  .0440  .0080
    n = 200, k = 12          .1775 .0925  .0425  .0075
  ```

  Closer to nominal at every level in every cell — twelve of twelve — with power
  untouched (`.9967` at `α = .05` either way on a planted alternative).

  Two readings that measurement killed on the way: selection does not *inflate*
  the statistic, it *disperses* it (`E[W(λ̂)] = 1.13` against `E[W(1)] = 2.17`,
  because a fresh null draw usually wants more shrinkage than the fit chose,
  while a draw that looks wiggly buys a smaller `λ` and a much larger `W`); and
  the control variate does not always tighten the Monte-Carlo error, so that
  error is measured per query and published. `p_value_bound` carries the
  quadrature bound plus twice the replay's standard error, and
  `p_value_conditional` publishes what the fixed-`λ` law alone said, so the
  correction is visible rather than folded in.

- **The refinement certificate had ONE side, so "the cascade has remaining gain"
  and "the bound is too loose to tell" were the same sentence (#2759).** Four
  cascade fixtures refuse at the rank-maximal design, and the issue's own framing
  put the closest one "inside the gain bound's own measured 1.30x slack". There
  was no way to decide that from the certificate, because the certificate was
  `‖g‖²/(λd)` and nothing else.

  Appending the candidate columns `X₂` with penalty `λd` decreases the penalized
  objective by exactly `gᵀS⁻¹g`, with `g = X₂ᵀW r̂` and
  `S = X₂ᵀW(I − H)X₂ + λd·I`. The shipped bound discarded the ENTIRE data term —
  `S ⪰ λd·I` — which is the `x = 0` member of a family no later member of which
  had ever been evaluated. For ANY `x`, writing `r = g − Sx`:

  ```text
      2xᵀg − xᵀSx   ⩽   gᵀS⁻¹g   ⩽   2xᵀg − xᵀSx + ‖r‖²/(λd)
  ```

  Left is `(x − S⁻¹g)ᵀS(x − S⁻¹g) ⩾ 0`; right adds `rᵀS⁻¹r ⩽ ‖r‖²/λ_min(S)` with
  `λ_min(S) ⩾ λd`, the same structural fact the shipped bound rests on and the
  only inequality used. Both ends are computed from an explicit `Sx`, never from
  a conjugate-gradient recurrence, and the upper end is floored by the shipped
  number, so the certificate can never be looser than the one it replaces. `S` is
  matrix-free: one apply of the candidate design through the hash grid the gain
  vector already builds, ONE cascade solve for `(I − H)`, one apply back.

  **The stopping rule is the decision, not a tolerance.** Iteration stops as soon
  as the bracket lands entirely on one side of `REFINE_TOL·rss_pen`. There is no
  accuracy constant to pick because accuracy is not what is being asked for — a
  comparison is. The ceiling is the Krylov dimension, past which the answer is
  exact by construction.

  **The hypothesis this was built on is FALSIFIED, and it was ours.** The claim
  was that discarding the data term "is not a small conservatism" in the
  rank-maximal regime. Measured:

  ```text
    fixture                        lower        upper        tolerance    lower/tol  slack
    cascade_state_rejects_corrupt  7.944884e-3  7.946538e-3  6.547001e-3   1.214x    1.024x
    ..._roundtrip_reproduces_...   6.326893e-2  6.327000e-2  1.690929e-2   3.742x    1.018x
    quasi_uniformity_guard_...     3.560937e-2  3.562072e-2  2.485628e-3  14.326x    1.063x
    smoothness_ceiling_...         1.050801e-1  1.055257e-1  2.184455e-3  48.104x    1.223x
  ```

  The slack is 1.8% to 22%, the bracket has closed to four to six digits, and the
  exact gain exceeds the tolerance by 1.21x to 48x. No tightening can pass these
  fixtures because there is nothing left to tighten — and that is the answer the
  issue asked for, now impossible to confuse with conservatism. `Underresolved`
  and `RefinementCertificate` both carry the bracket, and `Display` says out loud
  when the lower end is above tolerance.

  Three of the four fixtures were then found to be gated on a certificate they do
  not test — two persistence gates and a metric guard, all reaching a fit through
  `fit_residual_cascade`. They take a fixed-depth design and `fit_reml` now, or
  assert what they name. `smoothness_ceiling_...` is left RED with its assertion
  standing: it IS about the certificate, and at `num_centers = 5997` on
  `n = 6000` with `rss_pen = 2.1845` against a noise floor of `n·σ² = 2.4`, what a
  finer level buys is interpolation inside a row space the design already spans,
  not discretization bias. `REFINE_TOL·rss_pen` is a bias criterion being read
  where more columns do not reduce bias. That is a criterion question, it is not
  answered by moving `REFINE_TOL`, and a gate that says so is worth more than a
  green one that does not.

  Verified: the bracket is gated against the objective decrease it bounds by a
  route sharing no code with it — build the design with the candidate level
  appended, solve at the same fixed λ, difference the two penalized objectives —
  reproducing the truth to every printed digit in 3-10 CG steps across a
  six-decade λ sweep. `cargo test -p gam --test misc residual_cascade`: 25 of 26,
  from 12 of 17 at the start of the run.

- **The SAE terminal Newton polish is a Levenberg–Marquardt trust region on the
  stationarity residual, judged in the currency its own gate reads (#2762).**
  The phase accepted 100% of its steps while the raw KKT gradient rose 15x–107x
  per accepted step. Two defects, stacked.

  **The merit.** The acceptance test compared the TRIAL state's Newton decrement
  in the MAJORIZER metric, `gᵀB(θ₊)⁻¹g(θ₊)`, against the PRE state's decrement in
  the EXACT-Hessian metric, `gᵀA⁺g`. Same bilinear form, two different
  operators, measured 67x apart on the witness — so the test was satisfiable by
  any step at all. `gᵀB(θ)⁻¹g(θ)` is not a function of the state alone: it falls
  whenever `B` stiffens, however far `g` rises, so it cannot referee a
  comparison between two states. The merit is now `½‖g‖²`, which is the quantity
  the KKT gate is a bound on and a function of the state alone.

  **The step, and this is the one that decides convergence.** Making the merit
  self-consistent does not converge this phase — measured at 482 accepted steps
  / 0 rejected — and the reason is that the step is outside its own model. At
  the `#2015` witness, `‖g‖ = 1.226522e-4` with the WHOLE residual inside the
  operator's retained range, the undamped step is `‖Δ‖ = 4.416833e-1`; applying
  it drives the merit `7.52e-9 → 6.07e0`, and an Armijo test on `½‖g‖²` first
  passes at `α = 4.9e-4`, buying 0.03%. **The step's LENGTH is set entirely by
  the near-null eigendirections of `A` while the residual is carried by the
  well-conditioned ones**, and no scalar step length separates those: shrinking
  the step to keep the flat direction inside the model shrinks the useful
  directions by the same factor. Every earlier attempt on this issue traded one
  fixture for another against that wall.

  Damping separates them, and `A` is already materialized and diagonalized here,
  so the entire Levenberg–Marquardt path — and the model residual it predicts —
  is closed form at one diagonal pass per point:

  ```text
  Δ(ν) = Σ_i u_i λ_i (u_iᵀ rhs)/(λ_i² + ν)      g + AΔ(ν) = Σ_i u_i c_i ν/(λ_i² + ν)
  ```

  On the same state, sweeping ν and MEASURING each point:

  ```text
  ν=0        ‖Δ‖=4.42e-1  merit 7.52e-9 -> 6.07e0      ratio -8.1e8
  ν=5.73e-8  ‖Δ‖=4.37e-3                -> 5.38e-8     ratio -6.2e0
  ν=5.73e-7  ‖Δ‖=4.58e-4                -> 6.18e-11    ratio  0.9992
  ```

  `‖g‖ 1.23e-4 → 1.11e-5` against a `7.13e-5` tolerance, in one step, with the
  objective falling too.

  **Every number in the ladder is derived.** The first trial is `ν = 0` — the
  step this phase has always taken — so the quadratic tail near a
  well-conditioned root is unchanged and a state that never needed damping never
  pays for one. The ladder then spans `λ_min²` to `λ_max²` over the RETAINED
  spectrum by `RIDGE_GROWTH`: below `λ_min²` a damping cannot move the flattest
  resolved direction, above `λ_max²` every direction is already flattened. A
  trial is accepted when its MEASURED reduction is at least `ARMIJO_C1` of the
  reduction its own model predicted, with the shared round-off cushion. The
  accepted `ν` is carried to the next step divided by the same growth and
  snapped to `0` under `λ_min²`, so a converging tail returns to pure Newton by
  itself. Predicted reduction is monotone decreasing in `ν`, so a ladder that
  falls under `DIRECTIONAL_DECREASE_REL_FLOOR × merit` is exhausted — a proof of
  termination, not a cap.

  **The merit is the gate's currency, and the ambient residual is an
  invariant.** The gate is `raw ≤ tol OR quotient ≤ tol` with the quotient norm
  clamped at or below the raw one, so the gate IS the quotient bound and the
  quotient merit `½‖Π⊥gauge g‖²` is what this phase descends. That distinction
  is not cosmetic: on the `zz2015` witness the terminal state carries
  `gauge_share = 0.76`, so the ambient merit is 94% orbit and sits at its own
  floor while the gate is still 28x out — a 784x improvement in the quantity the
  gate reads registers as a 6% move in the ambient one. A projected norm can
  also fall without the residual falling, so acceptance additionally requires the
  ambient merit not to GROW: one acceptance currency, plus the invariant that
  quotient progress may never be bought by pumping residual into the orbit.

  **The budget is spent only while the phase is on track to finish.** A step
  costs one dense materialization + eigendecomposition of `A` — measured 13.4 s
  at `dim = 519`, against 0.14 s of assembly and 0.30 s for the entire damping
  ladder — so the step COUNT is the whole cost of this phase and a fixed cap
  prices every entry at the worst case. At the contraction the accepted step
  actually delivered, the band is `ln(tol/gate)/ln(contraction)` steps away; if
  that exceeds the steps left, the phase stops on its trajectory rather than on
  its budget. The test reads the system the accepted trial already assembled, so
  it costs nothing and fires one whole eigendecomposition earlier than the next
  loop top could. Stopping is neither a refusal nor final: the merit is
  monotone, so everything gained is kept, and the refine loop may re-arm the
  phase and re-measure. Measured on `zz2015`, whose inner solve fails in both
  arms: the refusal moves from `‖Π⊥gauge g‖ = 8.24` (`4660x` over the band) to
  `4.93e-2` (`27.7x`) — a 168x better terminal state, and `intensive_over_bound`
  falls 21 orders, `2.1e24 → 5.8e3`.

  Properties, not hopes: every accepted step strictly decreases the quantity the
  refusal is denominated in, so this phase **cannot leave the state worse than
  it found it**, which is measurably what it did before. The merit is monotone
  across steps by construction, so the dual-currency cross-iteration contraction
  bail is deleted rather than kept dead. A trial costs ONE assembly, where it
  used to cost an assembly plus a full arrow factorization because the merit it
  evaluated needed one. Indefiniteness needs no special case: `Δ(ν)` solves
  `(A² + ν)Δ = −Ag`, positive semidefinite for every symmetric `A`, so a
  resolved negative mode is descended rather than reflected.

- **The smooth-term likelihood-ratio test is scored against its own null law,
  not against a distribution fitted to two of that law's moments (#2672).** At
  fixed `λ` the whole-term LR is exactly `W = Σ_j w_j χ²_1` with
  `w = eig(2F_jj − F_jj²)`, so `Σ_j w_j` is Wood's `edf1` *and* the statistic's
  null mean. The reference was `g·χ²_ν` with `(ν, g)` matched to the first two
  moments of that spectrum. It is now the spectrum itself, inverted by
  `gam_math::probability::weighted_chi_square_sf` (Imhof) — which had landed
  under this same issue and had no consumer.

  The two-moment surrogate is exact when the weights are equal and one-signed
  wrong otherwise, with the error growing as the tail deepens. Measured against
  the exact law on `f_j = 1/(1 + λγ_j)` for a second-difference penalty, six
  decades of `λ` × `k ∈ {6, 12, 20}`, the size delivered at a nominal `α`:

  ```text
  α = 0.05   0.99 – 1.02 x       α = 1e-3   1.01 – 1.31 x
  α = 0.01   1.00 – 1.11 x       α = 1e-4   1.14 – 1.61 x
  ```

  **Where the gap lives is not where the intuition puts it, and the measurement
  is what says so.** The surrogate is exact at BOTH ends of the shrinkage range:
  `w ≡ 1` on an unpenalized term, and a single distinct weight once REML has
  shrunk a term to its null space — measured on a null-true `k = 12` fit,
  `w = (0.322, 5.9e-7, 7.1e-8, …)`, where the two references agree to eight
  figures. It opens in the middle, on a term carrying real signal. So a
  null-simulation size grid — which spends all of its time in the collapsed
  regime, at `α = 0.05`, where the surrogate is exact — could not have detected
  this, and its passing was never evidence the reference was right.

  **The whole spectrum, without a general eigensolver.** The moments were traces
  of powers of `F_jj` precisely because reading the weights off `F_jj` would need
  one — it is not symmetric. It need not be read off `F` at all: the penalty is
  block-diagonal by term, so `(I − F)_jj = [H⁻¹S]_jj = [H⁻¹]_jj S_jj =: P`
  exactly, hence `2F_jj − F_jj² = I − P²` and `w = 1 − eig(P)²`. `P = B·S` with
  both factors symmetric PSD is similar to `B^{1/2} S B^{1/2}`, reachable with
  the self-adjoint entry point already in `gam-linalg`, through `B = UΛUᵀ` rather
  than a Cholesky so a singular block is a zero eigenvalue and not a
  factorization failure. `[H⁻¹]_jj` is `beta_covariance()` divided by the
  family's own `coefficient_covariance_scale()`; the two factors are in
  reciprocal units, so that multiplier has to come off exactly or every weight is
  wrong by it.

  **The identity is measured, not assumed.** It holds only if the penalty is
  block-diagonal by term AND `Vb`, `F` and `S(λ)` are published in one
  coefficient basis — and both halves have been wrong in this exact path before
  (the similarity-map drop, the internal-basis first-order correction, the
  block-local `coeff_range`). None of the three is visible by reading. So the
  driver assembles the spectrum both ways whenever the fit supports it and
  publishes `moment_residual`, the relative disagreement; a test pins it under
  `1e-8` across two families, three model shapes and both shrinkage regimes.

  **The tail is resolved as finely as the statistic is known, and no finer.**
  Imhof's truncation point grows like `ε^{-2/(2+m)}` in the number `m` of weights
  active at it, and a shrunk smooth has one weight of order one over a tail of
  dust, so `m = 1` and the cost is `ε^{-2/3}`: at `gam-math`'s default `1e-11` a
  single p-value measures 0.13 s to 3.3 s. `W = 2(ℓ_full − ℓ_null)` is a
  difference of two separately converged optimizations, so it is known to
  `ΔW = tol·(W + E[W])`, and a p-value is known to `|S(W) − S(W + ΔW)|` however
  well the integral is done. That is what the quadrature is asked for — evaluated
  through the two-moment summary, which costs nothing and is being used as a
  SCALE rather than as a value. `SmoothTermLrInference::p_value_bound` publishes
  what was reached; the integration test compares the published p-value against
  the strict default THROUGH that bound, so the bound has to be honest.

  Three named lanes replace one switch: `NullSpectrum` (exact),
  `SpectralMomentMatch` (the old `g·χ²_ν`, when `H⁻¹` or the penalty is
  unavailable but `F` is), `UnitWeightFallback` (scalar EDF). Their errors have
  different signs and sizes, and the Python surface carries the lane, the
  spectrum, the residual and the bound alongside the p-values.

- **The certified cascade held seven `m²` blocks to deliver two objects that
  need one, and the admissible design width is DERIVED from that residency
  (#2758).** `smoothness_ceiling_forces_refinement_and_certifies_residual_bias`
  refused with `CertifiedSpectrumCapacity`: a 6000-row cascade identifies 5997
  penalized directions and the certified route stopped at 2893, because
  `CERTIFIED_SPECTRUM_MAX = isqrt(BYTES / (BLOCKS·8))` over a measured
  `BLOCKS = 8`. Raising `CERTIFIED_SPECTRUM_BYTES` was ruled out on the issue and
  is not what happened; it is untouched at 512 MiB.

  The measurement was honest and the constant was not the defect. What the route
  SPENT it on was. The criterion consumes `Θ` and `Vᵀβ` — every eigenvalue of the
  penalty-whitened Schur complement, and the whitened response in its eigenbasis.
  `eigenvectors` was read at exactly ONE site, to form that projection, and
  `eigh` cannot hand it over without building all of `V` plus faer's
  tridiagonalization workspace. On top of it a full `m × m` upper `X'WX` was
  assembled, of which only the `rank × rank` penalized block and a `q × rank`
  cross block (`q ≤ 4`) are ever read.

  ```text
    before                                    after
    m x m upper Gram          8 B / m^2       (not assembled at all)
    rank x rank Schur         8 B / m^2       packed upper triangle   4 B / m^2
    rank x rank eigenvectors  8 B / m^2       (never formed)
    faer EVD workspace       ~40 B / m^2      (no EVD)
    ----------------------------------        ----------------------------
    measured  6.41-6.84 blocks = 51-55 B/m^2  measured 4.03 B/m^2
    cap       2896 columns                    cap      10362 columns
  ```

  `V = QW` for the Householder `Q` and the QL `W`, so `Vᵀβ = Wᵀ(Qᵀβ)`. The new
  `gam_linalg::packed_symmetric_spectrum` reduces the packed triangle IN PLACE,
  applying each reflector to `β` as it is formed, then runs the implicit-shift QL
  accumulating every Givens rotation into that same single vector instead of into
  an `n × n` accumulator — the Golub–Welsch "keep one row of the eigenvector
  matrix" device, with a general start vector rather than `e₁`. Neither factor is
  ever formed. The mathematics is unchanged: all eigenvalues, the exact
  projection, the same roundoff floor, the same dropped null modes.

  `CERTIFIED_SPECTRUM_BLOCKS` is replaced by
  `CERTIFIED_SPECTRUM_BYTES_PER_COLUMN_SQUARED = 5`, an INVENTORY again rather
  than a black box: one packed `f64` triangle is `8/2 = 4` bytes per `m²`, plus
  the next integer of headroom. `5997 < 10362`, so the binding constraint on that
  fixture returns to identifiability.

  **Two defects in the new reduction, both found by measurement rather than
  review.** The classical relative deflation test `|e_i| ⩽ ε(|d_i|+|d_{i+1}|)`
  DOES NOT TERMINATE on a rank-deficient Gram: on `F Fᵀ` with `F` 296×148
  standard normal, `‖T‖ ≈ 9e2` while the 148 null directions arrive as
  `d ≈ e ≈ 1e-13`, so the test asks for `4e-29` against rotations that re-inject
  `ε‖T‖ ≈ 2e-13` every sweep. The absolute floor `ε‖T‖_∞` is taken alongside it.
  And the reflector produced NaN — `τ` and `v` are invariant to a rescaling of
  `x` but `1/(α − β)` is not, so a row decayed into the denormal range (what a
  rank-deficient trailing block becomes after a thousand steps) OVERFLOWED it and
  `0·inf` on that row's exact zeros wrote NaN; the whole trailing block followed,
  and the QL reported it as a 30-sweep non-convergence at an index that meant
  nothing. The reflector is now built on `x / max|x|`, where `|α_s − β_s| ⩾ 1`
  holds at every input scale, and an `O(n)` finiteness check between the
  reduction and the sweep names a reduction defect as one.

  **Two gates moved with the design, stated rather than quietly retuned.** The
  peak-memory arms are now BOTH past `DENSE_GRAM_MAX` and assert it: the narrow
  arm sat at `m = 891`, where the design carries a persistent `dense_gram` cache
  — an `m²·8` term present in one reading and absent in the other, which does not
  cancel in the difference and biases the differenced marginal DOWNWARD, i.e. in
  the direction that makes an under-declared residency look fine. And
  `spectral_and_solved_residual_forms_agree` charged its comparator bound to ONE
  of two INDEPENDENT computations of the same quantity, treating the
  decomposition as exact: at `m = 20` and `cond(A) = 1.005` that is `4.46e-15`
  against a measured `4.60e-15`, so the gate failed on the last bit the moment
  the rounding differed. Both comparands are charged now — two equal terms, not a
  factor chosen to admit a number.

  Timing at the new widths, measured so it is not rediscovered: a rank-6795
  profile builds in 46.2 s (9.1 GFLOP/s through the packed symv/spr2 on 4 cores),
  rank 1922 in 1.16 s.

- **`ln S` was seven approximations wearing one name, and its error was a step
  function of `(mu, sigma)` (#2714).** The latent-survival / frailty inner solve
  stalled at `stationarity_residual = 1.741e-2` against a `3.6e-10` tolerance
  with the trust radius railed at `1e-12` and both terminal rejections coming
  from the OBJECTIVE — the model and the likelihood accepted every step. The
  diagnosis on that issue localized it to `cloglog_log_survival_term_controlled`
  taking `value.ln()` of a value-space evaluator, and reasoned about the size of
  that error with the representation-floor model `EPSILON/S`, which at the
  failing point `(mu, sigma) = (3.2, 0.15)` predicts `6.9e-14` and therefore
  cannot explain the measured `6.5e-2` disagreement between the analytic score
  and a finite difference of the value.

  Graded on the shipped path against a 60-digit reference (peak-shifted, so the
  reference is accurate ABSOLUTELY in `ln S` even at `ln S = -1.3e5`; the naive
  un-shifted high-precision integral is itself wrong by `8.7e-3` at
  `(8, 0.005)`), the error at that point is **`1.242e-1`** — nine orders above
  the model the thread was using. And it is not one bad branch:

  ```text
      mu     sigma    ln S (ref)        shipped err   route
      12.0   0.002    -1.28982e5        4.371e+06     QuadratureFallback
       8.0   0.002    -2.96340e3        4.975e+05     QuadratureFallback
      12.0   1.000    -5.81967e1        8.201e+01     ExactSpecialFunction
       8.0   0.500    -7.09799e1        3.693e+01     ExactSpecialFunction
       3.2   0.150    -2.01463e1        1.242e-01     ControlledAsymptotic
  ```

  Three worst rows, three different routes — including the fixed-window
  Gumbel-mixing escape hatch that #798 added FOR the underflow corner, whose
  `Phi((eta-mu)/sigma)` transition is unresolved at small sigma by a node ladder
  #2469 had pinned inert at a constant 513. A fourth defect sits in the
  rare-event asymptotic `ln1p(-e^{mu+sigma^2/2})`: at `(-50, 8)`, exactly on its
  own `rare_log = -18` gate, it is **20.8x** wrong, because its first-order
  model needs the higher cumulants to be small and at `sigma = 8` they are not.

  So no threshold repairs this — there is no pair of these routes accurate on
  either side of a common cut, and an analytic derivative cannot be the
  derivative of a surface whose error jumps.

  **One surface.** `S` and `1 - S` are integrated in the standardized variable
  `z = (eta - mu)/sigma` on a Clenshaw-Curtis panel placed by the integrand:
  `L(z) = -z^2/2 - e^{mu+sigma z}` is strictly concave
  (`L'' = -(1 + sigma^2 e^{mu+sigma z}) <= -1`), so it is unimodal, its peak is
  the unique root of a monotone equation — solved in LOG form
  (`ln sigma + mu + sigma z - ln(-z) = 0`) so no `e^{mu+sigma z}` is ever
  materialized, which the value form cannot do when the root sits at
  `z ~ -mu/sigma` — and the points 60 e-folds below the peak bracket everything
  representable. Node count is the panel's local-scale arclength
  `T = int sqrt(1 + sigma^2 e^{mu+sigma z}) dz` (closed form) times a measured
  density, plus a `sqrt(sigma)` term for the Bernstein-ellipse shrinkage that
  `T` does not price: `T` is nearly constant at ~22 across the plane while the
  measured requirement walks `65, 97, 193, 385, 769` at
  `sigma = 0.5, 2, 8, 60, 200`, because `exp(-e^{sigma z})` is entire but its
  modulus grows off the real axis with period `2 pi / sigma`. Everything
  accumulates by (signed) log-sum-exp, so there is no value-space underflow to
  escape and no `.ln()` of a cancelled quantity. Past `S ~ 0.6` the complement
  panel supplies `1 - S` and `ln1p` finishes it, which is what retires the
  rare-event asymptotic instead of re-gating it.

  Worst absolute error of the panel on the same grid: **`9.3e-10`**, at
  `ln S = -5.7e6`, i.e. `1.6e-16` relative — the representation floor of the
  answer.

  **The `sigma >= 8` derivative gate is gone.** Gaussian integration by parts in
  `z` gives `sigma^j d^j S/d mu^j = int He_j(z) phi(z) f(z) dz`, so order 0 IS
  the value: the tower is the same sum with a different Hermite weight on the
  same nodes, and value/derivative cannot be on two surfaces even in principle.
  What remained was only which derivative BASIS is better conditioned — the
  direct tower (degrades at small sigma, where the answer is genuinely
  `O(sigma^j)` while the summands are `O(1)`) or the rung/Touchard combination
  (degrades at large sigma, #2610). A signed log-sum-exp knows its own
  cancellation `cond = ln(sum|terms| / |sum terms|)` and its relative error is
  `eps * e^cond`, so the tower is admitted on the solve of `eps e^cond = 1e-13`,
  `cond <= 6.1` — exactly while it is at the working floor of what it displaces.
  All 132 grid rows with `sigma >= 8` clear it, so the measured gate is a strict
  superset of the constant.

  Gates, replacing `cloglog_gumbel_quad_node_ladder_is_inert_at_a_constant_513_2469`
  (which pinned the inertness that caused the escape hatch's failure): a 28-row
  high-precision accuracy table (`6.7e-16` worst, relative); the
  value/derivative consistency assertion whose ABSENCE let this live, since
  every earlier gate scored an analytic derivative against its own value
  function (`9.8e-10` worst, against the `6.5e-2` that named the defect); a
  Richardson two-stencil smoothness sweep that converts a jump of size `d` into
  `0.75 d/h^2` and so bounds any residual step at ~1 ulp; and a two-sided
  statement that the node ladder now moves with sigma.

- **The outer gradient contracted an inverse that belonged to neither operator
  (#2515).** The Laplace criterion ranks `½log|A|` for the exact observed
  information `A = ∇²_θθ L`; `B` is the Gauss--Newton majorizer, the
  positive-definite scale the Newton and IFT solves factor. #2509 Phase-2b moved
  every production VALUE route onto `A`. The bundle/matrix-free route's
  DERIVATIVE did not follow, and the thread recorded that as "the gradient still
  prices `B`". Reading the lane settled that it was worse than that:

  ```rust
  let a_sys = chunk_term.exact_a_evidence_system(target, rho, &sys)?;   // A
  let (log_det_tt, log_det_schur) = matrix_free_arrow_evidence_log_det_surrogate(
      &a_sys, ..., lane.as_deref_mut())?;                               // value AND bundle off A
  return Ok((log_det_tt + log_det_schur, Some(sys)));                   // returns B
  ```

  The from-probes channels reconstruct `(H⁻¹)_tt = A_i⁻¹ + G_i S⁻¹ G_iᵀ` — row
  factors from a factor CACHE, `S⁻¹` from the probe bundle. Production paired
  `A`'s reduced Schur with `B`'s row factors, so the reconstructed inverse
  factored neither operator.

  Four changes make the routes one criterion. `BundleEvidenceGeometry` carries
  the operator and its OWN factor cache (`cache` stays `B` everywhere: promoting
  it would double-count `ΔC`, which `solve_exact_stationarity_matrix_free` adds
  back). The streaming lane takes the gradient-bearing evidence entry point, so
  value, bundle and row factorization come from one factorization of one system.
  `ArdAxisPrior::log_precision_curvature` and
  `softmax_sparse_curvature_rho_derivative_block` emit `∂B/∂ρ` or `∂A/∂ρ` from
  one function each — the same functions whose difference is the `ΔC` map. And
  the coordinate-block θ-adjoint and the #2330 Patch-D residual
  third-derivative leg, the last two channels still on `B` operands, were
  ported.

  Measured at a fixed state where BOTH routes are admitted (`α = 10`, `A` PD,
  the periodic ARD clamp active on 12 of 24 rows so `ΔC ≠ 0`), bundle route
  against dense exact-`A`:

  ```text
                            bundle on exact-A geometry     dense exact-A         |Δ|
  value ½(log|A|−log|A_tt|)  5.15457065258939906e0         5.15457065258939906e0  0
  logdet_trace smooth        1.41527087036750832e0         1.41527087036749344e0  1.49e-14
  logdet_trace ard          -1.51312923828332035e0        -1.51312923828329660e0  2.38e-14
  COMPLETE GRADIENT                                                               1.57e-14
  ```

  against `8.46e-1` for the majorizer-rooted carrier the witness now asserts as
  a control. The ARD coordinate does not merely shrink — it changes sign, which
  is what the unmajorized `α·cos κt` on twelve clamped rows is supposed to do.

  Two boundaries are recorded rather than papered over. The historical `α = 250`
  witness is OUTSIDE the region where route parity exists: there `A` has a
  per-row eigenvalue of `−1.93e2`, the streaming route refuses outright, and only
  the globally-priced dense route survives — so the witness was re-premised onto
  the admitted regime, with the admission itself asserted. And on a cache that
  actually DEFLATES the two routes still disagree (`9.13` against
  `‖g‖∞ = 5.00`), because the dense route floors the spectrum of `A` globally
  while the arrow route conditions per row and pseudo-inverts the reduced Schur.
  The streaming lane's spectral-deflation refusal therefore stays, now reasoned
  from that number and reading BOTH caches; the number is under test, so the
  refusal cannot decay into folklore the way its previous justification did.

- **The "logslope" block is the SLOPE, and the name was the only thing wrong
  (#2764).** `rigid_observed_logslope(g, s) = s·g` — the identity, no `exp`
  anywhere on that path — so the penalty is on `b`, not on `log b`. The issue
  proposed making the map a genuine log on two grounds: scale-invariance of the
  penalty, and positivity. Both were measured, and the proposed remedy is the
  wrong half of the repair.

  **The slope is signed and the sign is an estimand.**
  `survival_multi_z_fit_hard::survival_multi_z_fit_truth_neglog_minimised_at_true_slopes_30_seeds`
  plants `(0.32, −0.21)` and pins that the population negative log-likelihood is
  minimised there over 30 seeds. `b = exp(g)` cannot represent that fit at all,
  so it is a strictly smaller model rather than a naming repair. The zero
  crossing the identity map permits is the covariate value at which the score
  stops predicting.

  **The scale-invariance argument is about `λ`'s units, not about the fit.**
  Rescaling `z → z/κ` sends `g → κg` and `Σ → Σ/κ²`, and the row negative
  log-likelihood, the preserving scale `c` and the index `η` are all POINTWISE
  invariant under that — now measured at `κ ∈ {2, ¼, 10, 0.3}` in
  `survival_multi_z_slope_scale_equivariance_2764`, together with the
  admissibility of a negative slope and the evenness of `c` in the slope. What
  is left over is `λβᵀSβ`, which needs `λ → λ/κ²`; REML supplies it, because
  under `β̃ = κβ` its Laplace criterion satisfies
  `Ṽ(λ/κ²) = V(λ) − ½·nullity·log κ²`, a shift constant in `λ`, so `λ̂ → λ̂/κ²`
  and the fitted surface does not move. Two honest caveats travel with that, and
  are recorded on the function: the `ρ = log λ` box is absolute, so a large
  enough rescaling breaks the correspondence at the wall; and so would any
  absolute — as opposed to relative — solver tolerance.

  So the fix is the name, at the point where the mathematics is stated:
  `rigid_observed_logslope` → `rigid_observed_slope` in both marginal-slope
  families, carrying the decision record above. The block, the
  `logslope_formula=` keyword, the CLI flag and the on-disk fields keep the
  historical spelling — renaming a public keyword and a saved-model contract is
  a breaking change and is not this commit's to take — and
  `docs/marginal-slope.md` now says plainly that the surface is the signed
  slope and why.

- **The measure-jet representer chart spent the same half-mantissa twice, and
  paid for it in span (#2761, #2754, #2751).** `condition_representer_section`
  whitened the chart against the SQUARED operator `G = EᵀE` and cut at
  `√ε·λ_max(G)`. Since `λ = σ²` that is a bar of `ε^{1/4}` on `E`'s own singular
  values, i.e. it admitted only `cond(E) ≤ 8·10³` and deleted everything else
  from the design. At the ranges REML actively selects that is most of the
  basis. Measured on the #2761 fixture (1-D curve in 3-D, 16 centers), *span
  floor* = least-squares residual RMSE of the NOISELESS truth on the realized
  design's own column span — the bound no `λ` can beat, because `λ` shrinks
  inside a span and never moves one:

  ```text
    ℓ/ℓ_seed  cond(E)   ε^{1/4} cut: p  span floor    repaired: p  span floor
       1      3.0e+01        12          6.11e-2          16       6.11e-2
       2      2.8e+04        11          2.43e-2          16       1.94e-2
       4      4.2e+07         8          1.50e-2          16       2.10e-3
       8      9.1e+09         6          1.67e-2          16       1.81e-4
      16      2.7e+11         4          8.92e-2          16       3.54e-5
  ```

  Note the old column going back UP past `4×`: the truncated chart was worse
  than the seed range it was introduced to improve on. The repaired column at
  `8×` reproduces an 80-digit projection of the same span to every printed
  digit, so nothing being kept is below what binary64 can see.

  **The justification did not survive its own remedy.** The cut's stated reason
  was "the energy pullback squares `E`". That is true of an *unwhitened* section
  only: once `EᵀE = I`, `S = UᵀQU` with `U` orthonormal, and `Q` is `ℓ`-INVARIANT
  (a form on center VALUES), so `cond(S)` stops being a function of the range at
  all — measured `cond(E) = 1.0`, `cond(S₊) = 21.1`, `log|S|₊ = 0.800` at every
  `ℓ` from `1×` to `16×` the seed.

  The chart now reads `E`'s own SVD, and carries three bars in place of one,
  all anchored on `‖K_cc‖₂` — the norm of the operator whose product formed `E`,
  never `E`'s own `σ_max`. That distinction is load-bearing: `σ_max(E)`
  COLLAPSES with `ℓ` (the constraint that makes `Z` head-orthogonal is exactly
  what annihilates the flat limit `K_cc → 𝟙𝟙ᵀ`) while `σ_min(E)` sits flat at the
  roundoff floor, so a *relative* bar sinks below that floor and starts admitting
  it as signal — at `ℓ = 11` on the 1-D sweep fixture the bar is `1.5e-17`
  against entries of `1.7e-17`, and the chart handed the fit **48 columns of pure
  rounding noise**. Columns of noise fit anything, so the criterion had a
  spurious minimum out there: profiled Gaussian REML on the shipped design gives
  `V = −447.8` at `ℓ = 6.16` against `≈ −44` in the honest region, and the
  failing fit's own checkpoint was `ψ = 1.81`, i.e. `ℓ = 6.1`.

  * **existence** `σ > ε·‖K_cc‖·max(dim)` — the backward-error bar of forming
    `E = K_cc·Z`. Below it there is no direction, only roundoff.
  * **amplification** — the scaling, not the membership, is floored at
    `√ε·‖K_cc‖`. `1/σ` is the factor by which a direction lifts the design's own
    roundoff and the criterion squares the design into `XᵀWX`, so the lift keeps
    significant digits exactly when `ε·(‖K_cc‖/σ)² < 1`. A direction below the
    floor is DAMPED, not deleted: a span is unchanged by any invertible
    rescaling, so it contributes the same column space and enters at a small
    norm, carrying its roundoff in at that same small norm.
  * **visibility** `(σ/floor)² > dim · SPECTRAL_RANK_RELATIVE_TOLERANCE` — a
    damped direction whose squared weight falls under the canonical
    penalty-spectrum rank cutoff is classified UNPENALIZED, i.e. an accidentally
    free design direction, and it makes `log|S|₊` a step function of `ℓ`.
    Measured: the primary's nullity flapping by one moved the profiled criterion
    by **8.5** at fixed `λ`, which is a barrier no `ln ℓ` line search can cross
    (`line_search=StepSizeTooSmall` on both 1-D `s(x, bs="mjs")` fixtures). The
    chart and the penalty classifier now reach the same constant and can no
    longer disagree about which directions are penalized.

  As a consequence the arithmetic reproduces `MeasureJetRangeBracket::ceiling`'s
  own physics instead of merely asserting it beside them: past the node diameter
  `σ_max(E)` falls under the amplification floor and the whole representer block
  damps smoothly toward zero, leaving the affine head — "the block is numerically
  one function plus the affine head and there is no distinct model past it".

- **A Gram that cannot be indefinite was coming out indefinite, because it was
  formed by squaring first (#2761).** `rebuild_metric_consistent_ridge` computed
  the restricted null-function metric as `M = Nᵀ(G_c N)` — materialize the
  metric, then contract. With `G_c = A_Gᵀ A_G` that is identically
  `(A_G N)ᵀ(A_G N)`, and the two differ sharply in floating point: the first
  squares `A_G`'s condition number and leaves a SIGNED `O(ε‖G_c‖)` error on `M`'s
  entries, unbounded *relative to `M`* wherever `N` sits where `G_c` is small. So
  the caller — correctly reasoning that a Gram restricted to a subspace cannot be
  indefinite — refused, at `eigenvalue −3.31e-5 below tolerance −1.30e-11`, on
  four of four measure-jet sweep cases. The eigenpairs now come from an SVD of
  `B = A_G N` and `M` is never formed, so the indefiniteness branch is gone
  because indefiniteness is impossible rather than unlikely. This is the #2318
  rule — *rank revelation acts on `A`, not on `AᵀA`* — which the sibling
  `null(S_c)` computation twenty lines above already followed.

- **The measure-jet interval band omitted the third variance term its own file
  declares (#2761, #2752).** `measure_jet_web_quality_contracts` was failing at
  coverage `0.3648` against `[0.85, 1.0]`, and it is not the basis, not `λ`, and
  not the covariance. The fitted design's span floor on the noiseless truth is
  `0.0024` against a held-out bias of `0.0301` — `12.5×`, so the basis can
  represent this target. Refitting the same design at scaled `λ` over six
  decades leaves the bias between `0.0221` (essentially unpenalized, `edf` 23.9)
  and `0.0301`, so no `λ` makes the band honest and over-smoothing is refuted.
  And the control settles it: refitting on training rows whose coordinates are
  CLEAN — same latents, same `y` draw, everything else identical — the SHIPPED
  two-term band covers at `0.9843`.

  What remains is errors-in-variables, and it is the confound the fixture's own
  header identifies — but the header fixed only the QUERY half. It moved the
  query rows to clean on-web locations (correct) and left the TRAINING rows at
  `embed(z) + ε`, so the fit is a consistent estimator of `E[y | x_observed]`,
  displaced from `f` at an exactly-known location by
  `σ_coord·‖∇f‖ = 0.02 × 1.5 = 0.030` — the measured `0.0301` to two digits.
  That term is contract 6 of the same file, with `σ_coord` estimated at fit and
  frozen and its `∇f̂` FD-gated against the fit's own `η`; contract 5 simply never
  consumed it. Both contracts now read one producer and contract 6 asserts the
  band's number IS its own reconstruction, so they cannot drift into two
  definitions of `Var_input`. Coverage `0.3648 → 0.8871` with the window
  unchanged.

- **`Σ` in the marginal-slope identity is `Var(z | a)`, and the fit was
  supplying one global matrix (#2766).** `bms/gradient_paths.rs` writes the
  identity the whole marginal-slope family is defined by as

  ```text
    z | a ~ N(0, Σ(a)),   η = c(a)·q(t,a) + r(a)ᵀz
    E_z[Φ(−η) | a] = Φ(−q(t,a))    ⟺    c(a) = √(1 + r(a)ᵀ Σ(a) r(a))
  ```

  — `Σ(a)`, conditional on the marginal-index span — and then supplied it from
  `marginal_slope_covariance_from_scores`, ONE weighted empirical covariance
  pooled over every row. Substituting a constant `c̄` into the exact integral
  leaves `E_z[Φ(−η)|a] = Φ(−q·c̄/c(a))`: the realized marginal index is
  `q·c̄/c(a)`, a multiplicative, covariate-dependent distortion of the one
  estimand this family exists to deliver. Measured on `K = 2` scores whose
  conditional correlation moves over `±0.8` while both conditional marginals
  stay exactly `N(0,1)`, the distortion reaches **1.46×** at a shared slope of
  1.0, and a Monte-Carlo of the actual integral over 20000 rows reads a worst
  relative marginal-index error of **0.145**. It is the same failure the K=1
  `homoskedastic_var` field doc records (`Φ(q√(1+b²)/√(1+b²v))`), one dimension
  up: #2768's per-coordinate location-scale gate forces `E[z_j|a] = 0` and
  `Var(z_j|a) = 1`, and no per-coordinate map can reach the OFF-DIAGONAL.

  `Σ(a)` is now a fitted object, parameterised by Pourahmadi's **modified
  Cholesky decomposition** — `T(a)Σ(a)T(a)ᵀ = D(a)` with `T` unit lower
  triangular carrying `φ_jk(a)` and `log d_j(a) = γ_jᵀ[1|a]`. Every parameter is
  unconstrained, so every `a` — including rows off the training hull — yields a
  positive-definite `Σ(a)`; a regression on the ENTRIES of `Σ` does not, and one
  row at `|ρ| > 1` makes `c(a)` the square root of a negative number. Read
  forwards it is a triangular system of the same two regressions #2768 already
  ships, and `L(a) = T(a)⁻¹D(a)^{1/2}` is the exact Cholesky factor — the
  `Σ = LLᵀ` low-rank shape `MarginalSlopeCovariance` already admits — so the row
  program's quadratic forms stay exact sums of squares with no runtime
  eigendecomposition and no PSD tolerance on this path.

  The couplings are a weighted ridge and the innovation variances a
  line-searched Fisher scoring of the Gaussian log-linear variance model —
  damped rather than raw, because `Σ w A Aᵀ` is the EXPECTED information and the
  undamped step overshoots. An earlier undamped loop that stopped when the step
  norm failed to shrink was measured returning a `log d` 4.5 nats short of the
  optimum, and it was an independent nonparametric oracle (bin the rows, compare
  the fitted surface to each bin's own empirical second moments) that caught it,
  at 3.42× the bin's sampling band before the fix and 0.63× after.

  **The escalation trigger is one robust Rao score test per score PAIR** on
  `ζ_j·ζ_k`, on the same centred conditioning span and at the same α the #2768
  gate uses: the statistic for the sentence the issue is titled with and nothing
  wider. No pair fires ⇒ the pooled object stays in place byte for byte, and
  `K = 1` never escalates at all (there is no off-diagonal there, and
  `Var(z|a)` is #2768's branch — a second, differently parameterised variance
  model on top of it would double-correct).

  Every covariance consumer in the survival row program moved onto a row-indexed
  `ScoreCovarianceField`: the shared lane's cached `1ᵀΣ1`, both vector
  workspaces, the `c_i` in `SurvivalMarginalSlopeFamilyScalars`, and both
  `LogslopeBlockJacobian` branches. Because `Σ(a_i)` is a per-row constant and
  not a function of `β`, every existing derivative formula holds verbatim.

  Saving a fit that consumed a conditional `Σ(a)` is refused at the point of
  loss (`persistable_score_covariance`). That state needs `K ≥ 2`, which the
  on-disk contract already refuses at load — it carries one `z_column` and
  validates a 1×1 score covariance — so nothing new becomes unsaveable; the
  refusal exists so the reason travels with it instead of arriving later as a
  shape mismatch. Murphy–Topel is unaffected: `rigid_score_zeta_sensitivity`
  already refuses at `K > 1`.

- **The iso-κ joint outer search was walking a certified SURROGATE and nobody
  was putting it away (#2760).** The joint `[ρ, ψ]` spatial search refused at
  `n ≥ 4000` with `NOT STATIONARY` after a Strong-Wolfe line search that
  backtracked to `StepSizeTooSmall` — 50 attempts, 48 of them at a step below
  the fifth printed digit of `θ`. Three independent defects were stacked
  underneath it, and all three are fixed.

  **1. The joint ρ box's wall passed through the point the result is graded
  against.** `#2454` widened the `±JOINT_RHO_BOUND = ±12` search box "only as
  far as the incumbent" — `(-12).min(seed)` — so every coordinate whose
  scalar-route `ln λ̂` fell below `−12` began the joint search exactly ON its
  lower bound: an active constraint from iteration zero, its outward gradient
  KKT-projected to zero, unable to descend even where the joint criterion at
  the ψ the search was moving to wanted it lower. Containment in a closed set
  is not the property this route needs; the graded point has to be INTERIOR.
  Measured on a noiseless 1-D Duchon `y = sin(t)`: REML drives `λ̂` down as `n`
  grows, so the incumbents cross `−12` one at a time — 4 of 5 coordinates
  pasted onto the wall at `n = 1000…8000`, all 5 at `n = 16000`, where
  `∂V/∂ρ₀ = +1.484` at the wall against a whole stationarity bound of `1.030`.
  A coordinate whose incumbent is not strictly inside the joint prior now falls
  back to the engine's own `±RHO_BOUND` — the box the incumbent was found in.
  Everything strictly inside keeps the historical box byte-for-byte.

  **2. The mint had no curvature — which is how the real defect was found.**
  The #1033 n-free ψ-lane declares `DeclaredHessianForm::Unavailable` "so the
  planner selects BFGS instead of ARC". It does not need to:
  `with_prefer_gradient_only` is unconditional on this problem and
  `capability::plan` reads `(Analytic, Analytic) if prefer_gradient_only → Bfgs`
  *before* the ARC arm. What `Unavailable` actually erases is the one terminal
  evaluation #2359 reserves for the mint, and with it the
  `curvature-resolvability` rung, the #2348 asymptote-rail certificate, the
  curvature-scaled flat-valley widening and the #2299 large-step flatness
  certificate. Restoring it here made `run.rs`'s value-agreement guard fire —
  that guard only compares lanes when the mint asks for the analytic one — and
  that is what surfaced defect 3 below. It is NOT part of the shipped repair:
  restoring it also makes `exact_spatial_joint_engine_aniso_iso_parity_1d`
  refuse (`|Pg| = 5.143e-3` against a `8.100e-3` bound, so stationary, but
  `interior lambda_min = -1.585e-3` against a `3.061e-3` gradient floor, with ψ
  railed at its own box edge), and that indefinite-curvature verdict is a real
  finding about this lane's terminal geometry that deserves its own issue rather
  than arriving as a side effect. The ladder is green at all five rungs without
  it. An instrument is not a fix.

  **3. The criterion the search ranks is not the criterion. THIS is the line
  search.** With the mint asking for the analytic lane, `run.rs`'s existing
  value-agreement guard named it at once, at the point the `n = 2000` search
  stopped, both inner solves converged:

  ```text
  value-only      = -1.2781058170149880e4
  analytic-sample = -1.2781006804748626e4
  disagreement = 5.137e-2   roundoff bound = 1.905e-4
  ```

  `270×` `outer_value_agreement_bound`, i.e. `4e-6` relative where `√ε` is the
  contract. The two lanes are the #1033b certified n-free ψ-Gram tensor and the
  exact realized design. The tensor is certified on the **Gram**
  (`PSI_GRAM_CERT_RTOL = 1e-9`) and on the reduced-basis **subspace**
  (`PSI_GRAM_SKIP_PROJ_ATOL = 1e-7`); nothing in that certification bounds the
  **criterion** the optimizer ranks, and `β̂ = (G + λS)⁻¹r` amplifies a Gram
  residual by the radial-kernel conditioning — which is the regime this search
  lives in, at `λ = e⁻³⁰`. A value probe that crosses a skip-eligibility
  boundary therefore sees the criterion JUMP by more than the decrease the line
  search is hunting. So the surrogate is a SEARCH object, the same kind of
  thing as the staged-pilot row subsample the sibling N-block driver already
  retires, and it gets the same exit: `begin_exact_polish` retires it at the
  search checkpoint and the optimizer continues, and certifies, on the exact
  streamed criterion. Every in-window trial of the search stays n-free.

  **What the gate at `theta0` could and could not say.** The joint and scalar
  routes' criteria at `theta0` disagree by `−1.4e-13`, `−1.7e-13`, `+5.5e-13`,
  `+6.0e-8`, `+6.0e-8` relative at `n = 1000, 2000, 4000, 8000, 16000`. Five
  orders in one step, and the step is not in `n`: it is the rung at which a
  SECOND penalty block reaches `λ = e⁻³⁰ ≈ 9.4e-14` and stops contributing to
  `H = XᵀWX + S_λ` at working precision, after which `log|H|` is a sum of logs
  across the raw Duchon Gram's `~1e15` spectrum and two independent assemblies
  part company at exactly the scale `ε·κ` predicts. No fixed relative constant
  can be both tight enough to catch the `5.047e-5` formula difference #2671
  found and loose enough to admit that. So the cross-route number keeps its
  full decomposition as a warning, and the REFUSAL moves to a comparison both
  sides of which are `fit_score` of a **scalar-route** fit: the incumbent at
  `theta0` against the accept-fit at `θ*`. Like for like, one arithmetic, on
  the quantity that ships — which is what the gate's own sentence ("the joint
  search is minimizing a different function than the one its result is graded
  against") asks for.

- **The CTN fit and every replay of it read the coefficients through two
  different charts (#2680).** `#2306` moved the conditional-transformation-normal
  likelihood onto the direct-α chart

  ```text
  h(y, x) = α₀(x) + Σ_{k≥1} I_k(y)·α_k(x) + offset + ε·(y − median),
  α_k(x) = ψ(x)ᵀ A[k, :],
  ```

  with the shape coordinates held non-negative by the factored Khatri-Rao
  monotonicity cone rather than by squaring a latent coordinate. The likelihood,
  the exact-Newton Hessian, the function-space penalties and the ALO row replay
  all moved. **Three consumers did not**, and kept reading the same
  `blocks[0].beta` as `Σ_{k≥1} I_k(y)·γ_k(x)²`: the observed-score path behind
  `model.transformation_score(df)`, the `E[Y|x]` inversion grid behind `predict`
  and `generate` on a CTM, and `score_influence_jacobian` — the out-of-fold
  generated regressor the calibrated marginal-slope chain consumes, together with
  its Murphy–Topel Jacobian.

  **What it did to the numbers.** The lower endpoint basis is `[1, 0, …, 0]`, so
  `L(x) = α₀(x)` is the same on both charts, while `U(x) = Σ_k α_k` becomes
  `Σ_k α_k²`. With the shape coordinates near a common `c` that makes the
  reported score

  ```text
  z_reported ≈ c·z + (1 − c)·L,     sd(z_reported) = c,  mean(z_reported) = |L|·(c − 1),
  ```

  i.e. exactly right at `c = 1` and wrong in both location and scale otherwise —
  and `c ≈ range(h)/p_shape` grows with the sample range of the response, so the
  error is invisible on small fixtures and severe at production `n`. On #2680's
  own fixture the fit's latent score is `N(+0.001, 1.011)` while the reported one
  is `N(+0.957, 1.469)`; the reported means reproduce the issue's published
  numbers to every printed digit. It also explains the issue's separate
  saturated-row population: `U` pushed into the far tail makes `Φ(h)` and `Φ(U)`
  both return exactly `1.0` in binary64, so those rows clip to `Φ⁻¹(1 − 1e-12) =
  7.034` rather than being a tail of any normal.

  Everything downstream of a CTN stage 1 consumed the wrong score: the
  `bernoulli-marginal-slope` / `survival-marginal-slope` `z` moment gate (which
  refuses to fit when the score is not `N(0,1)`), the Murphy–Topel
  generated-regressor covariance, and the documented `transformation_normal_stage1`
  chain.

  **The repair is one evaluator, not five edits.** A new
  `transformation_normal::chart` module is now the single definition of what `β`
  means: `ctn_row_geometry` computes `(h, h', L, U)` from the covariate-side
  coordinates, `ctn_component_sensitivity` states the derivative (for an affine
  chart, the response-basis entry itself — no chart factor), `ctn_endpoint_bases`
  states the structural endpoint bases, and `ctn_response_bases_at` assembles
  `[1, I_k(y)·T]` / `[0, M_k(y)·T]` so the fit and every replay prepend the
  location column identically. The family's `row_quantities` accumulates in the
  same order it always did, so routing it through the shared kernel is
  bit-identical rather than merely equivalent.

  `ctn_row_geometry` **takes** the `TransformationNormalParameterization` marker
  and matches on it. That marker has been persisted since `#2306`, and its own
  doc says it exists "so a reader can reject coefficients written under any other
  chart as a typed mismatch instead of silently reinterpreting them" — every
  reader validated it and then reinterpreted them anyway. It is now load-bearing:
  a replay path must name the chart it believes it is evaluating, and a second
  variant becomes a compile error in one function instead of a silent divergence
  in five. In `gam-predict` the two independent transcriptions of the CTN payload
  collapse into one `SavedCtnChart` reader that carries the saved marker into the
  evaluator, and the support endpoints now come from the structural bases instead
  of a second I-spline evaluation at the boundary knots.

  Pinned by `ctn_predict_score_reproduces_the_fitted_score_2680` (the predict
  path's score equals `block_states[0].eta` to round-off — a chart-agnostic
  invariant that catches this defect *and* its mirror image),
  `ctn_observed_score_clears_the_generated_regressor_moment_gate_2680` (the
  `bms::gradient_paths` moment bars at `n = 500`, where the squared chart reports
  `sd ≈ 1.4` against its own `0.13` bound), and
  `ctn_score_influence_jacobian_matches_its_own_finite_difference_2680` (the
  Jacobian is the derivative of the score the same call emits, which is what
  catches the `2·γ_k` shape factor independently of the value fix).

- **An I-spline and its own derivative described two different functions
  outside the knot domain (#2695).** `create_ispline_dense` SATURATES there —
  the value is the all-zero row below `knots[degree+1]` and a constant row above
  `knots[n_bspline]` — and says so, with the reason: a linear extension would
  produce negative I-spline entries below the left boundary and entries above
  one past the right, breaking the non-negativity and the `[0, 1]` range the
  basis exists to guarantee. `create_ispline_derivative_dense` differentiated
  through a *clamped* B-spline, whose exterior convention is linear extension,
  and so returned the boundary SLOPE where the I-spline value is flat. Orders 1,
  3 and 4 were affected; order 2 was already zeroed there.

  **How it surfaced.** The survival link warp is
  `q = q0 + Σ_j βw_j·I_j(q0)`. The link-wiggle block reaches `q` through the
  VALUE (`∂q/∂βw_j = I_j(q0)`) and its gradient was always right; the threshold
  and log-sigma blocks reach `q` only through `q0`, so every one of their
  chain-rule channels carries `m1 = 1 + Σ_j βw_j·I'_j(q0)`. Outside the knot
  domain the warp is flat and `m1` was not, so the joint-Newton RHS asserted a
  first-order change the objective does not make — at any step size. That is
  gam#2695's headline: on `survival_location_scale_saved_fit_preserves_linkwiggle_metadata`,
  zero of the linear-dominated trust attempts have `actual/(rhs·δ)` within 50%
  of 1, and all six outer seeds refuse with
  `rejects [model, likelihood, objective, feasibility] = [0, 0, 2, 0]` at trust
  radius `1e-12`.

  **Why it stayed hidden.** The error is proportional to the warp amplitude, and
  the wiggle knots are frozen at fit setup from the SEED `q0` with no margin
  (`initializewiggle_knots_from_seed` spans exactly `[min, max]` of the seed),
  while `q0 = −η_t·e^{−η_ls}` moves by orders of magnitude during the outer
  search. So the seed iterate is inside the domain and essentially every later
  one is not — and every gradient oracle in the tree ran at the seed, at
  `βw ≈ 0`, or both.

  The same file already applies exactly this argument to an OPEN knot vector
  (gam#1348: "A constant function has zero derivative, so BOTH the first and
  second derivative must be zero in the exterior spans"). The case never covered
  is that an I-spline is constant-extended on a CLAMPED vector too. Endpoints
  keep the interior one-sided slope, because `right` is routinely the largest
  observed value and the transformation-normal shape derivative `h'(y)` must
  stay positive there.

- **The survival location-scale event Jacobian was floored in the value and not
  in the derivative tower (#2695).** `exact_row_kernel_from_parts` clamped
  `g = dη/dt` to `derivative_guard` on three branches and then read
  `(log g, d log g, …)` at the FLOORED value, so inside the band the row
  log-likelihood is bitwise constant in `qdot` while the tower reports a slope
  of `1/guard = 1e6` and a curvature of `−1/guard² = −1e12`. The three branches
  are replaced by one derived object: `ln` exactly on the modelled feasible set
  `g ≥ guard` (bit-identical, so no fit that never reaches the floor changes),
  and below it the degree-4 Taylor continuation of `ln` about `guard`, with the
  returned tower being that polynomial's own derivatives. It is C⁴ at the knot,
  strictly increasing and strictly concave on the continued branch, and unlike a
  flat clamp it charges for leaving the feasible region instead of paying
  `ln(guard)` at `g = 0`. The monotonicity refusal predicate is unchanged by
  construction — the floors ran before the `g ≤ 0` test and lifted every `g`
  above `−(guard + roundoff_slack)`, which is now written directly — so no state
  that was accepted becomes a refusal and none that was refused becomes
  accepted. Cf. `survival/base.rs`'s `stabilized_structural_derivative`, which
  states the same contract for the Royston–Parmar arm and resolves it with a
  zero-slope clamp.

  **What the two repairs above do and do not close on #2695.** Measured on that
  issue's own witness (`gam-cli`
  `survival_location_scale_saved_fit_preserves_linkwiggle_metadata`), the
  first-order gradient/objective disagreement the issue is titled for is gone:
  `d ℓ / (∇ℓ·δ)` lands within 10% of 1 on **96 of 96** resolvable small-step
  trust attempts, against **40 of 77** before, with a median relative residual of
  `1.5e-5`; the quadratic penalty gradient was already exact to `2.9e-10`.

  The fit still does not mint, and what remains is a different mechanism —
  a **discontinuity in the Jeffreys value Φ**, not a derivative error. Along ONE
  ray from ONE base point, with the trial direction bit-identical across all five
  attempts and the cone projection inactive:

  ```
   t = 2.003e-4  Φ = -11.48618      λ_min = -8.870e-1   λ_max = 1.9435e1   gate 1.000000
   t = 5.008e-5  Φ = -11.48601      λ_min = -8.869e-1   λ_max = 1.9435e1   gate 1.000000
   t = 1.252e-5  Φ = -11.48597      λ_min = -8.868e-1   λ_max = 1.9435e1   gate 1.000000
   t = 3.130e-6  Φ = -11.48596      λ_min = -8.868e-1   λ_max = 1.9435e1   gate 1.000000
   t = 7.826e-7  Φ = -10.93381      λ_min = -6.922e-1   λ_max = 1.9645e1   gate 1.000000
  ```

  Φ steps by `-0.5522` between `t = 7.8e-7` and `t = 3.1e-6`, and the extreme
  eigenvalues of `Z_Jᵀ H Z_J` step with it, so the discontinuity is in the
  observed information rather than in the Jeffreys machinery reading it — the
  conditioning gate is saturated at `1.000000` on both sides, so it is not the
  gate's smooth band, and the floor regime does not move. `actual` is therefore
  constant across the backtracking ladder while `pred` quarters, `ρ` runs
  `-784, -3137, -12548, -50192`, and every attempt is refused: the
  `rejects [model, likelihood, objective, feasibility] = [0, 0, 2, 0]` signature.
  The direction is dominated by the threshold coordinate
  (`u ≈ (-0.9753, +0.2207, ~0, ~0, 0, 0)`).

- **The Jeffreys/Firth-armed outer REML gradient was not the gradient of the
  criterion it reports (#2612).** On the penguins real-data multinomial arm the
  fit did not produce a probability at all: the unbiased probe converged to a
  separated mode (identifiable-span Fisher information `lambda_min = 2.06e-18`
  against `lambda_max = 1.443`), the Jeffreys gate fired at full weight, and the
  armed refit then died in the outer smoothing search at
  `line_search=StepSizeTooSmall` — the solver's own gloss on which is *"the
  direction descended but no step improved the objective"* — with an indefinite
  terminal analytic Hessian and no fit assembled.

  **What was wrong.** The IFT mode response `v_k = d beta_hat / d rho_k` is a
  property of the INNER stationarity system, so it must be solved against
  `M_true = H + S_lambda + H_Phi + completion`, the exact Hessian of the
  Phi-augmented objective the inner Newton converged on. It was instead
  borrowing the LAML logdet's operator `M_DD = H + S_lambda + H_Phi`. The
  envelope theorem kills `v_k`'s other route into the outer gradient
  (`grad_beta f = 0` at the mode, whatever `v_k` is), so `v_k` reaches it only
  through the drift trace `0.5 tr(M_DD^-1 D_beta M_DD[v_k])` — and a `v_k` from
  the wrong operator makes the analytic gradient the derivative of a different
  function than the value. Central differences of the production outer criterion
  against its own analytic gradient, at the refit's own stalling rho: `1.5e-9 ..
  7.7e-8` with the term disarmed, `5.3e-2 .. 1.5e0` with it armed, the sign
  wrong on three of eight coordinates, and h-independent between `1e-3` and
  `1e-2`.

  The half-fix that hid it: `completion_in_operator` folded the completion into
  the operator on the projected/`Smooth` route, where the projected kernel
  already owns the value and the traces so the operator is free to carry it.
  Every family that overrides `pseudo_logdet_mode` away from `Smooth` — the
  multinomial (`PositiveDefinite`), BMS, the binomial location-scale and wiggle
  families (`HardPseudo`) — takes the route where the operator IS the value and
  trace object, so the completion could not go in, and the mode response
  silently inherited that constraint. Folding it in there instead is not
  available either: the scalar would then need the completion's own beta-drift
  (third directional derivatives no family exposes) for the trace to stay
  consistent with it, which is a measured ~38% gradient bias.

  **The fix.** Stop making one object serve both roles. `InnerSolution` gains an
  optional `mode_response_op`, read through `mode_response_operator()` by all
  three `ThetaModeResponseKernel::select` sites (gradient, dense Hessian,
  Hessian operator), so no two can disagree about which system `beta_hat(theta)`
  is differentiated through. The custom-family assembly builds it from the same
  operator assembly with `H_Phi + completion` in place of `H_Phi`, exactly when
  a completion exists and is not already in `hessian_op`. `None` everywhere
  else: with no completion, or no Jeffreys term, `M_true == M_DD` and there is
  nothing to separate, so every other family and every clean fit is
  byte-identical.

  Rejected on measurement: the seed. The formula path warm-starts the armed
  refit at the saturated unbiased mode, which the fixed-lambda sibling documents
  as catastrophic. Warm `|Pg| = 1.759e-2`, cold (`beta = 0`) `1.607e-2`, both
  against `bound = 2.290e-3`, both `hessian_psd=NO`. Same failure; not the seed.

- **A curvature certificate no longer decides on a direction along which the
  criterion is exactly constant (#2676).** Three `geo_disease_*_matern`
  scenarios refused with `INDEFINITE CURVATURE AT INTERIOR OPTIMUM`
  (`interior lambda_min = -5.048e-6`) or with a smoothing-correction
  contradiction decided by a **0.55% margin**. Neither refusal was a
  measurement of the fit.

  **What was wrong.** `rho = log lambda` is a nonlinear reparameterisation, so
  for any smooth criterion `H_rho = diag(l) H_lambda diag(l) + diag(g_rho)`
  holds exactly — the second term is pure chain rule and carries no curvature.
  Every criterion here sees `lambda` only through the assembled penalty
  `sum_i lambda_i (b - mu_i)' S_i (b - mu_i)`, so a `w` with `sum_i w_i S_i = 0`
  makes the criterion EXACTLY constant along `lambda + s w`. Lift it to rho by
  `t = diag(lambda)^-1 w` and

      t' H_rho t = sum_k (g_rho)_k t_k^2      exactly, at every point,

  which is bounded by `sum_k |(g_rho)_k| t_k^2` — *verbatim* the per-direction
  floor both gates compare against, with equality when the gradient shares a
  sign on the support. The direction did not sit near the decision boundary of
  those gates; it sat **on it, by identity**, and which side it landed on was
  the sign of the disagreement between the gradient code and the Hessian code.
  Measured on `geo_disease_matern`: `sigma = 2.0930992e-5`,
  `sum_k g_k v_k^2 = 2.0946774e-5`, intrinsic `-1.578e-8` — the identity holding
  to `7.5e-4`, on a minimum eigenvector equal to the antisymmetric direction of
  a penalty pair with `cos = 1.000000`.

  **The fix, and what it is not.** Not a wider floor: the comparison was
  degenerate, not under-resolved. `gam_solve::penalty_invariance` computes the
  invariance from the penalty map alone — the null space of the Gram of the
  augmented operators, so a nonzero prior mean that BREAKS a proportionality is
  seen rather than assumed away — lifts it to rho, and returns the orthogonal
  complement. The outer certificate (`run.rs`) and the smoothing correction
  (`invert_identified_rho_hessian`) both deflate that subspace and apply the
  existing rule, unchanged, to what is left. No tolerance is chosen: the rank
  boundary is the eigensolver's own Weyl backward error, the instrument already
  used for this Gram.

  **What this does not change, as a theorem rather than an anecdote.** An
  objective declaring no invariance — every objective except the two REML arms
  and the spatial joint arm, and those only on a redundant penalty map —
  reaches a bit-identical verdict; the deflated path is not taken. And what
  deflation can hide is bounded by Cauchy interlacing: `Z' H Z` is a compression
  onto a subspace of codimension `d`, so `lambda_1(Z'HZ) <= lambda_{d+1}(H)`.
  Deflating `d` directions can lose at most the `d` SMALLEST eigenvalues and
  never one beyond them — with the one-dimensional invariance here, a matrix
  carrying two negative directions still refuses, and #2665's
  `lambda_min = -1.6e3` saddle is not in the deflated subspace at all.

  Only the part of the invariance that lies INSIDE the judged face is deflated.
  The identity is a statement about the FULL direction, so an invariance
  direction with a material component on a railed coordinate is deliberately
  left in the judged block rather than restricted and deflated — restricting it
  would break the identity and, in the extreme, hide real curvature.

  **Related corrections.** The saddle-escape search now looks in the same
  subspace the certificate judged, so it can no longer step along the invariance
  — where the only "negative curvature" available is the residual gradient
  wearing a curvature's clothes — instead of along the genuine saddle that
  refused. `interior_min_eigenvalue` is now reported from the block the verdict
  was reached on. And the `[PENALTY-REDUNDANCY]` warning no longer says a
  redundant penalty map produces "a Z2-symmetric saddle": the criterion is flat
  there, not descending, and the fit is unaffected — what is lost is only the
  separate identifiability of the individual smoothing parameters.

- **The survival marginal-slope's effect can vary along the follow-up axis
  (#2765, #2767).** `logslope_time_k` / `--logslope-time-k` make `b` a fitted
  surface in `(x, t)` instead of a per-row constant, so a latent score whose
  effect attenuates with age is now a model the family can express.

  **What was wrong.** The family carried three follow-up channels for the
  location index `q` — its value at entry, at exit, and its exit-time derivative,
  because the likelihood is `log S(t₁) − log S(t₀)` and an event row picks up
  `log η′(t₁)` — and exactly **one** channel for the slope. Time reached `η` only
  through `q`. `logslopespec` was a static term collection, and the row program's
  primary vector was `(q₀, q₁, q̇₁, g)`.

  **Why episode splitting was not a workaround.** This is a *transformation*
  model, `S(t|x,z) = Φ(−η(t))`, not a hazard model. Splitting a subject into
  intervals with a piecewise-constant `b` gives per-row contributions
  `log S(t₁;b₁) − log S(t₀;b₁)` that do not telescope into any survival function.
  The slope had to move inside the row program.

  **Why the generalization is the right one.** The factor
  `c = √(1 + bᵀΣb)` in `η = q·c + bᵀz` is not decoration: it is exactly the
  rescaling that makes the *marginal* law invariant to the slope, since
  `E_z Φ(−(q·c + bᵀz)) = Φ(−q)`. That identity holds **pointwise in `t`**, so
  `b → b(t)` preserves the family's defining property — and it forces `c` to
  inherit the time dependence, giving
  `η′(t) = q′(t)·c(t) + q(t)·c′(t) + b′(t)ᵀz`. The last two terms are what the
  rigid kernel was missing.

  **The shape of the fix.** A `SlopeRowGeometry` the row program is generic over:
  `StaticSlopeGeometry` is the four-primary frame every existing model uses and
  is the `db/dt = 0` face of the six-primary `DynamicSlopeGeometry`. Both feed
  the *same* `row_program!` declaration — only the feature map differs — so the
  likelihood is still written down once. The frames are compile-time distinct
  because the row towers are dense in the primary count (the fourth-order tower
  is `P⁴`): a model that does not ask for a varying slope must not pay `5×` for a
  channel that is structurally a copy and a zero.

  The log-slope design is tensored against a `log t` B-spline margin by the same
  `build_time_varying_survival_covariate_template` the threshold and sigma
  margins use, with the standard anisotropic penalty pair `S_cov ⊗ I_t` and
  `I_c ⊗ S_t` so smoothness in `x` and in `t` keep independent smoothing
  parameters.

  Two consequences the generalization forced. `q′ ≥ derivative_guard` is the
  *marginal* monotonicity constraint and it implied the likelihood-domain
  condition `η′₁ > 0` only because `η′₁ = q′·c` with `c ≥ 1`; a varying slope
  breaks that implication, so `η′₁ > 0` is now an explicit domain check (on the
  static frame it is unreachable, so no existing fit moves). And
  `LogslopeBlockJacobian` gained the exact `(η₀, η₁, η′₁)` rows for the varying
  case — the identifiability audit would otherwise have been reading the Jacobian
  of a model that is not being fitted.

  **Refused rather than reinterpreted:** a per-score log-slope topology; a
  non-zero smooth anchor, coefficient bounds, or linear constraints on the
  log-slope surface; and *saving* a fit that used the margin, because the on-disk
  contract rebuilds the block from the covariate term spec alone and would
  evaluate a different model at predict. The resolved knots ride on the fit
  result for the predictor that will replay them.

- **The survival marginal-slope runs the automatic latent-measure gate, and the
  conditional calibration it escalates to now actually delivers a unit-variance
  score (#2768).** Three things, in the order they were found.

  **The gap.** The Bernoulli marginal-slope has run an automatic gate on its
  latent score since #905: a Rao score test on `E[z|C]` and `Var(z|C)` over the
  marginal-index span, escalating to `ζ = (z − m(C))/√v(C)` when it fires. The
  survival marginal-slope ran none of it. It called
  `standardize_latent_z_with_policy` and nothing else, and under the default
  policy — `Frozen { mean: 0, sd: 1 }` — that transform is the identity: it
  checked, it warned, and it passed `z` through unchanged.

  That is not cosmetic. The survival row index is `η = q·c(g) + s(g)·z`, so a
  conditional shift `E[z|C] = m(C) ≠ 0` puts `s(g(C))·m(C)` into the *influence*
  channel `q` — in a model whose entire point is that `q` is the marginal index.
  The pooled marginal gate cannot see it (the marginal law of `z` can be exactly
  N(0,1) while every conditional law is shifted) and rank-INT provably cannot fix
  it. On a fixture that is exactly N(0,1) marginally with `Corr(z, x) = 0.5`,
  slope `b = 0.6`, and a true marginal coefficient `β_x = 0.5`, the uncalibrated
  axis returns

  ```
  fitted marginal x-coefficient    0.195   against a truth of   0.500
  ```

  a 61% attenuation, derived in closed form in the fixture and reproduced by the
  fit. The two arms of that fixture — the shifted score, and the conditionally
  standardised score the outcome was generated on — now agree.

  **The defect underneath it.** `ζ = (z − m(C))/√v(C)` was dividing by the
  **marginal** variance of `z` whenever the Breusch-Pagan stage did not fire. The
  right constant is the **residual** variance of the conditional-mean regression,
  and the two are never equal on a fired gate: with `z` standardised,
  `1 = Var(m(C)) + E[Var(z|C)]`, so the residual variance is `1 − R²` and sits
  strictly below the marginal variance *whenever there is any conditional
  structure at all*. The error was therefore present on every firing and grew
  with exactly the structure the correction exists to remove.

  ```
  sd(ζ) at R² = 0.25       0.8586  ->  1.0000
  ```

  against the `post_sd ≈ 1` the struct's own field doc claims. The marginal-index
  identity `E_ζ[Φ(q√(1+b²) + bζ)] = Φ(q)` holds only at `Var(ζ|C) = 1`; at `v` it
  becomes `Φ(q√(1+b²)/√(1+b²v))`, so every marginal coefficient carried a ~4%
  multiplicative distortion. Worse, the calibrated residual then failed the
  standard-normal adequacy re-check **on the SD clause alone** (`|sd−1| = 0.134`
  against a `0.045` tolerance at n = 4000), which sent BMS to the empirical
  measure and, per #2718, withheld the covariance. The field is renamed
  `homoskedastic_var` and keeps its on-disk name `global_var`, so a model saved
  before the fix keeps applying the map it was *fitted* with.

  **The seams.** The gate is one object with the family's kernel capability as an
  argument (`EmpiricalLatentMeasureSupport::{Available, StandardNormalOnly}`),
  not two copies: the survival row program is the closed-form standard-normal
  probit lowering and owns no empirical-grid branch, so a `StandardNormalOnly`
  caller keeps the best available pre-transform and gets the failing adequacy
  ledger back rather than a measure it cannot evaluate. On the predict side the
  conditioning span is now named explicitly — the survival predictor's primary
  design is the q-design `[time | timewiggle | marginal]`, so reusing it as
  `a(C)` (which is what the shared code did) would have conditioned on time
  columns and applied a different map than the fit. And the naive covariance,
  which treats the generated regressor `ζ` as known, is corrected: the per-row
  channel `∂(score_β)/∂ζ` is derived mechanically from the sole
  `rigid_feature_program` declaration and gated against a central difference of
  that program's own gradient over 360 cells. Shapes the rigid channel does not
  cover (score-warp / link-deviation, `K > 1`) withhold the covariance with a
  typed reason instead of publishing one that is too narrow.

- **The measure-jet head spans the energy's WHOLE affine null space, so the
  term collection's centering can no longer delete a linear direction
  (#2751).** The `mjs` design's extrapolation head carried only the LINEAR part
  `{x_1..x_d}` of the jet energy's affine null space. The term-collection
  chokepoint then applies its parametric orthogonalization `Z = null(1ᵀX)`,
  which removes exactly one coefficient direction, and the constrained null
  space is `{γ : Zγ ∈ null(S)}` — so with no constant in `null(S)` for that
  removal to be charged to, **it came out of the null space itself**, leaving
  `d − 1` free ambient-linear directions where the theorem says `d`.

  The consequence only appears once REML selects a large energy `λ`, which it
  does for any near-affine truth: the fit is then confined to what the energy
  leaves free, and what it left free was one accidental direction — the single
  linear combination whose data-mean happens to vanish. Measured with the ridge
  limit against the shipped design's own Primary (no fit, no family, no
  smoothing search: least squares of the noiseless plane `0.2 + 0.9·x₁` with
  `λ·S_primary` added):

  ```
  lambda    d/dx1   rms[x1]  rms[x2]   pearson
  1e2       0.8994  0.29980  -0.00640  0.9993
  1e6       0.4838  0.16126  -0.13235  0.7729
  1e10      0.4342  0.14472  -0.14644  0.7029    <- one linear direction left
  ```

  At `λ → ∞` the surviving direction is `(0.695, −0.716)`; projecting the
  planted `(0.9, 0)` onto it gives `|cos 45°| = 0.707`, which is exactly the
  `0.7051` Pearson the `mjs`-backed BMS fixture reported end to end. Duchon,
  whose null space `{1, x₁, x₂}` *does* contain the constant, survives the
  identical chokepoint with the plane intact (`0.9000` at `λ = 1e10`).

  Two upstream hypotheses were killed before the collection was implicated, and
  both are now gates rather than beliefs: the energy form annihilates the affine
  span to `1e-17` relative on a regular grid, on scattered centers, on a
  10×-anisotropic layout and at a single scale; and the emitted basis-level
  Primary had nullity 2 with both directions exactly affine. Nothing upstream of
  the collection was wrong — a 2-dimensional null space is simply one dimension
  too small to survive a 1-dimensional constraint.

  `measure_jet_affine_head_lift` now returns the `(d+1) × (1 + head_rank)` lift
  acting on `[1 | x]`, `measure_jet_affine_head_block` realizes it, and
  `measure_jet_affine_value_basis` — which is both the gauge's `A` and the
  null-component penalty's projector — is literally the same object evaluated at
  the centers, so "the head spans exactly the energy's null space" is a property
  of the code instead of a comment two call sites have to keep agreeing on.

  ```
                          before -> after
  raw design width          15 -> 16
  Primary nullity            2 ->  3     (all three exactly affine)
  declared null frame        2 ->  3
  null-component rank        2 ->  3
  FIT chart width           14 -> 15     = m - 1, matching Duchon's k - 1
  ```

  End to end on the fixture that reported the defect, all four surface bases on
  byte-identical rows (the comparators are unchanged to every printed digit, so
  the change is confined to `mjs`):

  ```
  basis                            pearson   d/dx1   rms[x1]  rms[x2]  rms[nl]
  mjs(x1,x2,centers=16,scales=3)    0.9936   0.993   0.3311   0.0233   0.0298
  matern(x1,x2,k=16)                0.9416   0.701   0.2337   0.0336   0.0765
  duchon(x1,x2,k=16)                0.9975   0.961   0.3204   0.0202   0.0102
  s(x1,k=8) + s(x2,k=8)             0.9837   1.052   0.3506   0.0095   0.0634
  truth  0.2 + 0.9*x1               1.0000   0.900   0.3000   0.0000   0.0000
  ```

  The predict-side ambient gradient takes the affine lift (row 0 is the constant
  and contributes nothing to `∇f̂`; its FD gate now carries a nonzero constant
  column, so a mis-indexed row fails it), and the errors-in-variables
  reconstruction in `model.rs` rebuilds the same lift. A model frozen before
  this change carries an `m + head_rank` row transform and is refused by the
  frozen-width check with that exact message rather than silently replaying a
  different basis.

  Also corrected: `measure_jet_bms_backend`'s penalty-count assertion demanded
  ONE penalty per surface, describing a "nullspace ridge folded into the
  Primary" the builder deliberately does not do — it emits the Primary
  independently of `double_penalty`, so the realized count is two. The wrong
  number survived because that assertion had never executed: the truth-recovery
  assertion above it failed first.

  Verification: the `measure_jet` integration target is 11 passed / 5 failed and
  `gam-terms` is 919 passed / 1 failed. Both failure sets are pre-existing and
  both belong to #2761's `ln ℓ` dial, established by reverting
  `crates/gam-terms/src` + `crates/gam-models/src` in place and re-running:
  identical five, and `psi_producer_matches_fd_length_scale` red at both
  sources (`analytic −6.988657e-5 vs FD 0` before, `analytic 1.574896e-4 vs
  FD 0` after). A printing replica of that comparison attaches the magnitudes —
  `|analytic|max` `3.78e-3` (pre) / `3.50e-3` (post) against `|FD|max` `8.3e-13`
  / `4.2e-13`, i.e. the shipped null-component candidate does not move with `ℓ`
  at all while the producer reports a jet three orders above its central
  difference, in both arms, while the Primary agrees with its own FD to `3.7e-9`
  in both. The builder ships the rebuilt metric-consistent ridge `R = N M Nᵀ`
  (whose frame `N` has zero representer coefficients, so `E·N` is `ℓ`-invariant
  by construction) while the ψ producer differentiates the raw pullback
  `E(ℓ)ᵀH₀E(ℓ)`, which is not. That is an objective↔gradient desync on the `ℓ`
  coordinate and a far better candidate for the five "the direction descended
  but no step improved the objective" line-search refusals than anything in
  this issue; it is #2761's, not this one's.

  One more measured consequence, recorded because it is a real cost of this
  change and not a wash: in `tests/regressions/misc/`, the same revert-in-place
  comparison shows the two `mjs` `ln ℓ` fixtures **swapping**, not improving —
  `measure_jet_formula_fit_succeeds_like_the_cli` was red before and is green
  after; `measure_jet_5d_converges_when_aniso_loses_to_isotropic` was green
  before and is red after, refusing with `hessian_psd=NO` at a point the solver
  itself calls stationary (`|Pg| = 2.156e-1` under `bound = 2.778e-1`). Adding
  one column to the design moves the ψ landscape, and both fixtures sit on the
  same knife edge — which is the state #2761's `ln ℓ` search is actually in.
  That instability is the next thing to fix, not a reason to leave the null
  space one dimension short.

- **The matrix-free from-probes selected-inverse channels price per-row
  deflation instead of refusing it — and there was never anything to derive
  (#2712).** Three channels of the #2080 wide-`p` analytic-gradient cluster —
  `logdet_theta_adjoint_from_probes`, `ard_log_precision_hessian_trace_from_probes`
  and `assignment_log_strength_hessian_trace_from_probes` — hard-refused any
  cache carrying `deflated_row_directions`, on the stated grounds that "the
  plain-`S⁻¹` bundle carries the UNdeflated block" and so could not rebuild the
  Daleckii–Krein correction `tr(inv_vv·(D − DΦ[D]))`. They convert as one
  all-or-nothing cluster, so one deflated row routed the whole fit to a dense
  channel the lane cannot afford at massive `K`.

  **The premise was a misreading of `undamped_factor`.** That accessor returns
  the Cholesky of the spectrally CONDITIONED `Φ(H_tt^(i))` — the block that
  pinned `λ̃ = 1` on each deflated direction — not of the raw `H_tt^(i)`, and the
  reduced Schur behind the bundle is that same conditioned arrow's. So
  `A_i⁻¹ + G_i S⁻¹ G_iᵀ`, which is literally what BOTH routes build, already IS
  the deflated `(H⁻¹)_tt`. Measured on the deflating fixture, rebuilding
  `A_i = L Lᵀ` from the cached factor:

  ```
  ||A v - v||               = 1.97e-16      (the unit-stiffness pin itself)
  ||A - U diag(cond) U^T||  = 4.97e-16
  ||A - U diag(raw)  U^T||  = 9.999999e-1   <- 10^15 larger
  ```

  and the from-probes reconstruction matches `selected_inverse_row_blocks` to
  `~2e-16` RELATIVE on every deflated row. What the probe routes actually lacked
  was the correction TERM, whose remaining operands — `deflated_row_directions`,
  `deflation_row_spectra`, and the raw per-slot `D` each channel already
  assembles from its own row jets — never involved `S⁻¹` at all. Each channel now
  applies the same `deflation_block_correction` its dense sibling applies, on the
  t-slot channels, the border channels, and the ordered Beta–Bernoulli
  shared-mass diagonal (that last one was a second, latent gap: the from-probes
  site tuple carried no `diag_deflation_weight` field at all).

  The three private copies of the row-block reconstruction collapse into one
  `arrow_solver::row_selected_inverse_from_probes`, the matrix-free sibling of
  `DeflatedArrowSolver::selected_inverse_row_blocks`, which documents the
  conditioned-block fact once at the place it is used.

  **A test-methodology finding came out of the acceptance requirement, and it is
  worth stating on its own.** The deflated and deflation-blind operators agree
  wherever the deflation is inactive, so machine-precision parity is also what a
  port that ignored deflation would produce. The instrument for that is
  `deflation_blind_cache`: a clone of the cache with ONLY the deflation metadata
  emptied, against which the PRODUCTION dense adjoint yields exactly the
  deflation-blind operator — no test-only flag, no second code path. Measured on
  the ordered Beta–Bernoulli anchor, the correction moves `Γ` by `8.47e-8`
  against `‖Γ‖∞ = 98.9`, because that fixture's unit-deflated direction is a
  near-null the raw derivative barely touches. That is **below** the historical
  per-entry parity tolerance `1e-8·(1+|Γ|) = 1e-6`, so on a deflated cache those
  element-wise assertions alone would have passed a port that dropped the
  correction entirely. The gates now tighten to `1e-10·(1+|Γ|)` on a deflated
  cache, state non-vacuity as a ratio (`parity·1e3 ≤ separation`), and assert the
  thing that actually decides whether a gate can see the defect it exists to
  catch: the per-entry tolerance must itself be finer than the separation.

  **Which SUBSPACE deflates decides which channel can be gated at all**, and that
  also had to be measured. The ARD correction contracts `D = hess·eₛeₛᵀ` at ONE
  coordinate slot, so `M = UᵀDU` carries a factor `U[s,d]` and the whole
  correction vanishes when the deflated direction misses that slot — separation
  exactly `0.0` on both real deflating fixtures, while the θ-adjoint separates by
  `8.47e-8` on the same cache because its `D` is a full `q×q` block. The ARD gate
  therefore sweeps every local slot with the deflation RECORD redirected onto
  whichever eigendirection that slot loads (factors, Schur and eigenbasis
  untouched; both routes read the same four operands, which is the whole claim).
  Slot 0 is a logit slot there and still gives `0.0`; slot 1 gives separation
  `2.606` against parity `0.0`.

  **What this does NOT do is flip the wide-`p` routing, and the reason is
  measured.** The complete outer ρ-gradient still disagrees between the dense and
  bundle routes on a deflated fit — `8.45` against `‖g‖∞ = 5.00` — but that gap is
  BIT-IDENTICAL on the cache and on its deflation-blind clone, so deflation
  cannot be its cause. It is #2499/#2515's β-Schur smoothness-EDF desync, landing
  on the two smoothness coordinates and leaking into the ARD ones through the
  shared single-adjoint IFT contraction. The end-to-end gate asserts exactly
  that decomposition — the two routes price the *deflation contribution*
  identically, and the surviving gap is deflation-independent — so it doubles as
  a tripwire: if the residual desync ever acquires a deflation-dependent part it
  comes back here rather than staying with #2515. The fourth refusal on the same
  false premise (the streaming outer evaluation) is therefore corrected in place
  rather than lifted, with the measurement that would lift it written on it.

  Measured, same filter and host, `76770446e` with this work's files reverted vs
  after: **31 passed / 6 failed → 38 passed / 5 failed.** The six baseline
  failures are the identical set; the one that left is
  `sae_logdet_theta_adjoint_from_probes_matches_dense_softmax_2080`, which was red
  on its own premise (it declares `NoRowDeflates`, and every member of its ladder
  now deflates) — a premise that only existed because the route used to refuse
  deflated rows.

  All seven #2712 gates pass at `ff1ee8a24`. The five remaining failures under
  that filter are the identical baseline set — `#2330` Patch-D coordinate gap
  `1.774e-3`, `#1625`'s unresolved invariant-subspace block, two `#2500` gates,
  and the dense ARD FD deflation trace — every one of them a dense-route or
  fixture-stratum failure, and no dense computation path is touched here: the
  production diff is confined to the three from-probes functions, the new
  `arrow_solver` helper, and comments.

- **SAE post-fit certification no longer costs `dim³`: the residual-gauge
  curvature is `p` blocks of `D × D`, not one `(p·D)²` Gram (#2757).**
  `fit_diagnostics_report` was materializing the curvature `H = RᵀR` as a dense
  `param_dim × param_dim` matrix and taking its dense symmetric
  eigendecomposition — 45.97 GiB and 60.5% of the whole fit at `p = 4096`, on
  a quantity that is *certification*, not the fit.

  It is block diagonal. The certificate's parameter vector is the atoms'
  flattened frames, so column `c = offset_k + i·d_k + a` names (atom, **output
  coordinate**, axis), and a frame perturbation of output coordinate `i` moves
  the reconstruction only on `i`. The per-row pinning Jacobian is therefore
  output-coordinate diagonal, and

  ```
  H[(k,i,a), (k',i',a')] = Σ_n M_n[i,i'] · g_n[i,(k,a)] · g_n[i',(k',a')]
  ```

  inherits exactly the row metric's output-coordinate coupling and nothing
  else. Under the metric `diagnostic_metric()` installs whenever no
  output-Fisher harvest ran, `M_n = I`, and every off-block entry is never
  written — measured at bit-zero, on a fixture whose decoder touches every
  output coordinate. So the object is `p·D²` numbers and `p·D³` flops
  (`D = Σ_k d_k`), against `(p·D)²` and `(p·D)³` before: a factor of `p` in
  memory and `p²` in time, i.e. 45.1 GiB → 11 MB at the `p = 4096, D = 19`
  shape. Measured end to end on the issue's own fixture:

  | `p` | `param_dim` | before | after |
  |---|---|---|---|
  | 256 | 1024 | 0.316 s | 0.131 s |
  | 512 | 2048 | 1.318 s | 0.152 s |
  | 1024 | 4096 | 7.960 s | 0.206 s |

  Growth per doubling of `p` falls from 6.0× (cubic) to 1.36×.

  **The reported `pinning_rank` was also wrong, for a related reason, and is
  now right.** The rank decision is `σ_i(R) > α·ε·max(m, param_dim)·σ_max` with
  `α = 100` — deliberately 100× *above* an SVD's backward error, which is what
  makes it meaningful. Testing the algebraically equivalent `λ > τ²` on the
  Gram instead puts the threshold a factor `α²·ε·N` *below* a symmetric
  eigensolver's own resolution, so every roundoff eigenvalue clears it: a
  curvature of true rank 12 in 80 parameters was reported as rank **45**. The
  blocks are now accumulated as triangular roots by streaming Givens rotations
  (same memory, same cost, no squaring) and the rank is read off their singular
  values; the dense fallback floors its threshold at the standard
  `|λ̃ − λ| ≤ dim·ε·‖H‖` resolution bound. All representations now agree and all
  respect `rank(RᵀR) ≤ rows(R)`.

  **The other branch of the same function was worse.** With an isometry pin
  installed — reachable from the shipped `IsometryPenalty` API —
  `to_residual_gauge_model` materialized each per-row pinning Jacobian as a
  dense `p × param_dim` block and retained all `n` of them: `8·n·p²·D` bytes,
  which is **2.55 GiB per observation** at `p = 4096, D = 19`. The pin's rows
  genuinely cannot be folded into the blocks — eliminating one against block
  `i` scatters that block's row into every other, so the QR of `[⊕R_i ; L]`
  fills in completely — but `H = ⊕_i B_i + VVᵀ` is block diagonal plus a
  symmetric update of rank `Σ_k d_k`, and Sylvester's law of inertia gives

  ```
  n₊(H − sI) = n₊(B − sI) + n₊(−I_k − Vᵀ(B − sI)⁻¹V)
  ```

  — its eigenvalue count above *any* shift, exactly, in `O(p·D·k²)`. That is
  all both consumers need: the rank is the count above `τ²`, and `λ_max` is the
  shift at which the count reaches zero. Both branches now stream the same
  structured curvature through one entry point, and no production path
  materializes a per-row Jacobian at all.

  Certificate output is otherwise unchanged: verdicts, group signature,
  residual gauge dimension and per-generator energy fractions are identical
  whichever representation the reduction ran on, which is gated from fifteen
  independent angles in `tests_frame_curvature_2757` — including #2267's own
  eigendecomposition census showing that nothing at the parameter dimension is
  decomposed at all.

- **The measure-jet representer range is a basis coordinate again, so REML
  selects it (#2761).** `lambda` shrinks a coefficient vector INSIDE a span; it
  never moves the span. The measure-jet design is
  `X = K(data, centers; ell) * z`, so `ell` decides WHICH m-dimensional subspace
  the representers occupy — the same standing the Matern `kappa` has, and the
  module header already called it "matern's log_kappa analog". Freezing it at a
  geometric heuristic therefore makes an error no smoothing parameter can
  repair.

  Measured on `measure_jet_perf_parity`'s 1-D-curve-in-3-D Gaussian fixture
  (n=1500, sigma=0.10, 16 centers, p=15), where `span floor` is the
  least-squares projection residual of the NOISELESS truth onto the realized
  design's column span — the bound no `lambda` can beat:

  | arm | ell | edf | span floor | unpen. LS | held-out |
  |---|---|---|---|---|---|
  | frozen (auto ell) | 0.5144 | 14.684 | 0.152488 | 0.155484 | 0.155584 |
  | REML-selected ell | 3.8813 | 14.006 | 0.000014 | 0.008155 | 0.009642 |
  | `matern(k=16)` | - | 14.619 | 0.006077 | 0.011989 | 0.011639 |
  | `duchon(k=16)` | - | 15.016 | 0.002443 | 0.011308 | 0.010521 |

  At the frozen range the fitted `0.1556` IS the span floor: unpenalized least
  squares on the same design gives `0.1555`, dropping the null-component penalty
  moves the fourth decimal, and `edf/p = 0.98` says the fit was already spending
  everything it had. Freeing `ell` puts measure-jet past both comparators at
  LOWER edf, so nothing is traded for the accuracy.

  The dial itself was not new: `299c83ffc` introduced it default-ON precisely to
  remove this fixture's 13x, `a3afd17a2` found its one hazard — a BMS fit shares
  one measure-jet basis between the marginal mean and the log-slope surface,
  where a design-moving kernel scale on shared covariates reached a
  separation-scale runaway — and contained it AT THE BMS ENTRY POINT, and
  `b1d94d1a5` then flipped the GLOBAL default off anyway. The 13x came back. The
  scoped freeze is untouched and still runs where the hazard is.

  Rejected: raising `MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR`. That is what #1041
  did (x2 -> x1) as the dial's replacement and it is what main measured 13.4x
  with. No fixed multiple of the center spacing can be the answer, because the
  range that aligns the span depends on the target's smoothness relative to the
  center layout — data, not geometry. The constant survives as the SEED of the
  outer coordinate and its doc now says so.

  Behaviour change for callers: an explicit `mjs(..., length_scale=X)` now PINS
  the range instead of seeding a search, mirroring the short-circuit an
  explicitly-scaled Matern gets. `learn_length_scale=` overrides either way.

- **The constant-curvature smooth stops pinning its kernel range, so `kappa_hat`
  measures curvature instead of the range error (#2747).** `exp(-d_kappa/ell)`
  carries a curvature and a range in one exponent and they are strongly
  confounded: to first order `d_kappa = d_0*(1 + kappa*a(x,y))`, so the MEAN of
  `a` over the evaluated pairs acts exactly like a rescaling of `ell` and only
  its VARIATION is genuine curvature. The smooth fitted `kappa` while pinning
  `ell` to a heuristic (median chart center spacing, doubled), so `kappa`
  absorbed whatever range error the heuristic carried — and range
  mis-specification is monotone in one direction, which is the railed
  `V_p(kappa)` the issue reported.

  Measured on truths planted INSIDE the fitted span, three planted curvatures x
  three planted ranges: with `ell` pinned the criterion recovers `kappa*` only in
  the one cell where the truth's own radial length scale IS the auto `ell_ref`.
  At half or twice that range it rails at a box endpoint, reports the WRONG SIGN
  (`kappa_hat = -0.35` against a planted `+1.0`), or reads a confident interior
  `kappa_hat = -0.94` / `+0.94` on genuinely FLAT data. That one working cell is
  the configuration the acceptance fixture happened to use, which is why the
  fixture was green while the estimator was not.

  The construction is now one kernel at one range —
  `X = K_{kappa,ell}(data,C)z` and `S = z' K_{kappa,ell}(C,C) z` at the same
  `ell`, so `S` is again the RKHS roughness of the function `X` realizes and the
  model is the ordinary subset-of-regressors GP. `#944`'s fill-invariant
  `L(kappa)` and `#1464`'s separate penalty length `L_S(kappa)` are deleted with
  their implicit-function jets and the two 100-iteration Newton solves they cost
  on every basis build; both were attempts to remove the confounding by
  CONSTRAINT, and pinning a scalar summary of the design selects a
  one-dimensional curve through the `(kappa, ell)` plane a priori, on which
  `dV/dkappa = V_kappa + V_ell*L'(kappa)` keeps a range term that vanishes only
  if `ell_ref` was already optimal. On the profile curve it vanishes identically
  by the envelope theorem.

  Pinning `kappa=` no longer pins the range with it. It used to take the whole
  term out of the curvature profile — the only owner of either coordinate — and
  leave the range at the auto heuristic, which is a worse fit for no stated
  reason: fixing the geometry is not a statement about the kernel's resolution.
  A pinned-`kappa=` term now gets its range profiled at that curvature, and
  because the profile is Gaussian-identity-only it drops out with a log line
  rather than turning a working non-Gaussian fit into a refusal (a FREE `kappa`
  still refuses, since `kappa_hat` is the estimand the caller asked for and
  shipping it unfitted would be worse).

  `psi = (kappa, eta = ln ell)` now carries a full second-order tower, the outer
  solve is one-dimensional over the range-PROFILED criterion
  `V_p(kappa) = min_eta V(kappa, eta)` so the point estimate, the profile CI and
  the flatness LR are extrema of the same object, and `length_scale=` follows the
  same mgcv-`sp=` convention `kappa=` does: explicit pins, omitted estimates.
  `Model.curvature(...)` rows gain `length_scale_hat` and
  `length_scale_estimated`, because every statistic in the row is a profile over
  the range and a reader who cannot see it cannot tell an estimate anchored at a
  sensible resolution from one anchored at a degenerate corner.

  With the range profiled, `kappa_hat` lands within 0.19 of the planted
  curvature in all nine cells (median 0.07), `ell_hat` recovers the planted range
  to 3%, and there are no rails and no sign inversions. The acceptance fixture
  now cycles the planted range `{0.5, 1, 2}x ell_ref` across its replicates, and
  its flat arm is a real signal again rather than the constant mean the
  confounding had forced on it.

  NOT fixed by this, and stated so the next reader does not have to rediscover
  it: on truths that are in NO fitted span the criterion still prefers `+kappa`.
  An origin-radial plant is curvature-blind as a function class
  (`d_kappa(x,0) = 2*arctan(sqrt(kappa) r)/sqrt(kappa)` is a strictly monotone
  reparametrization of the chart radius at every `kappa`), and a multi-reference
  plant that does carry curvature still rails with a sign that flips as the
  center count sweeps 6 -> 12 -> 24 -> 40. That is span-approximation ordering
  under misspecification — the residue of `#1464` — and a different root cause
  from the range confounding.

- **A monotone link warp no longer ships an unpenalized rescale of the index it
  is composed onto (#2647).** `binomial_location_scalewiggle_termswith_matern_spatial_blocks_fit_finitely`
  refused all four startup seeds with `did not converge after 48 cycle(s)`, which
  reads like a budget problem and is not one: at 600 inner cycles the arms are
  bit-identical to 200 and one seed is *worse* than at 48. The per-cycle trace
  says what is happening — `|beta|inf` climbs 230x while `0.5 b'Sb` falls 41x
  (fitting `pen ~ |beta|^-2` on two seeds) and `-loglik` is flat to `8e-4`. The
  solve was descending toward an infimum at infinity that is never attained.

  The free direction is the warp's LINEAR element. The model is
  `q = q0 + w(q0)` with `q0 = -eta_t*exp(-eta_ls)`, so a linear warp element is a
  rescale of the index; the index block is penalized, so the penalty falls along
  that orbit while the likelihood does not move. Measured on the failing
  fixture's own knots, the anchored I-spline span contains `u -> (u - left)` to
  `2.7e-15`, its coefficient vector is componentwise non-negative (the whole ray
  stays inside the monotone cone `beta_w >= 0`), and the order-2 roughness
  charges `3.0e-14` for it. `ispline_function_penalties` sets
  `roughness_nullspace_dim = derivative_order - 1`, so this is structural: every
  configuration whose smallest requested order exceeds one shipped an unpenalized
  warp direction unless `double_penalty` happened to close it.

  `canonical_wiggle_function_penalties` now closes the assembled set's own joint
  null space unconditionally, reading `null(sum_j S_j)` in the function metric off
  a per-block-normalized sum and appending one shrinkage coordinate spanning it.
  This is the same treatment, and the same argument, that
  `build_binomial_threshold_and_scale_blocks` already applies unconditionally to
  the log-sigma block, where `(beta_t, beta_ls) -> (c*beta_t, beta_ls + ln c)` is
  the exactly analogous index-scale gauge. It is a no-op on every configuration
  that was already well posed, including the shipped default (`orders = [1,2,3]`,
  whose order-one roughness is full rank on the anchored basis). The smallest
  eigenvalue of the exact joint penalized Hessian at the fixture's seed went from
  the `~1e-10` the family's own source comment records to `7.254550e-1`, the fit
  completes in 0.1 s at its original 48-cycle budget, and the same objective comes
  back at 48, 200 and 600 cycles.

  A model saved before this change whose warp set gains a coordinate will refuse
  to load with a `SchemaMismatch` naming the reason: its coefficients and
  log-lambdas index a penalty system this build no longer assembles, and they were
  obtained from a criterion with no minimiser. Refit.

- **The CLI and the engine no longer disagree about where a survival fit is
  anchored (#2631).** The survival time-basis centering anchor was decided in two
  places. `materialize_survival` — the engine path behind `fit_from_formula` and
  the Python FFI — promoted the robust median-exit anchor whenever ANY entry age
  exceeded the origin threshold, and hardcoded the caller override to `None`.
  `gam-cli`'s `run_survival` promoted it only for marginal-slope, and owned the
  `--survival-time-anchor` override. Each was internally consistent, so nothing
  was mis-persisted; the same formula, data and config simply produced a
  different fit depending on which front end ran it. Measured on a 500-row
  delayed-entry cohort (`Surv(entry, exit, event) ~ s(x)`, location-scale) from
  byte-identical inputs: the CLI persisted `survival_time_anchor = 4.0579` (the
  earliest entry) where `gamfit` persisted `12.0317` (the median exit).
  Re-centering is an exact affine reparameterization of the design, so this is
  not cosmetic metadata — it is the conditioning the smoothing selection sees,
  which is the whole point of the `#751`/`#1790` robust anchor.

  Three further consequences fell out of the same duplication. Because the
  override lived only in the CLI's copy, and the CLI's own default
  (transformation / Weibull) route delegates to the engine copy,
  `--survival-time-anchor` was **silently ignored on the default route** — parsed,
  validated, then dropped, while the code comment claimed it was "honored by all
  paths". `FitRequestConfigDocument` had no field for the anchor at all, even
  though `--survival-time-anchor` declares a conflict with `--request` on the
  premise that the document carries the complete scientific model configuration.
  And the CLI's third branch was unreachable dead code carrying a *second,
  different* definition of left truncation — `min(entry) > threshold` against the
  materializer's `any(entry > threshold)` — which under-triggers on staggered
  entry, the ordinary shape of a real registry cohort.

  The rule is now one function, `resolve_survival_time_anchor_for_mode`, composed
  of three orthogonal primitives (validate-override, earliest-entry,
  robust-interior) and one left-truncation predicate. The three per-mode
  resolvers collapse into it and the transformation-specific one is deleted.
  `SURVIVAL_DELAYED_ENTRY_THRESHOLD` goes too: it was a second constant kept in
  lockstep with `ENTRY_AT_ORIGIN_THRESHOLD` by comment, the same failure mode one
  level down. The override became model configuration —
  `FitConfig::survival_time_anchor`, a `survival_time_anchor` key in the fit-request
  document, and a `gamfit.fit(survival_time_anchor=...)` kwarg — validated once in
  `FitConfig::resolve()` and refused on a non-survival response exactly as
  `survival_likelihood` already was. Engine behaviour is bit-identical when no
  explicit anchor is set; left-truncated CLI location-scale and latent fits now
  agree with the engine, which is the intended change.

  The mechanism itself is now measured rather than asserted. On a staggered-entry
  cohort, one I-spline basis centered at each candidate anchor: the earliest-entry
  anchor leaves the trend coordinate at `max|trend| = 5.000` with **every** row
  one-signed; the median-exit anchor leaves `1.140` with an exact 6/6 sign split.

- **A separated binomial fit is no longer refused for being right (#2273).** On
  exactly-separated data — a genuine gap between the classes — `y ~ smooth(x)`
  could not be fitted at any `n`. The in-loop separation guard turned a
  converged, finite, well-penalized logit fit into `Unstable (possible
  separation)` whenever its fitted linear predictor separated the classes by more
  than an η-gap of `1e-3`, or saturated, or collapsed its working weights, or
  drove the deviance below `1e-6` per sample. On separable data every one of
  those is a property of the CORRECT fit: a good fit's η *does* order the
  classes, its μ *are* near {0,1}, its weights *do* collapse. The guard exists
  for the case where the penalized objective has no finite minimizer, which
  happens only when a direction of recession of the log-likelihood lies in
  `null(S(λ))` — and under the double penalty it never does, so `β̂(λ)` is finite
  and unique even under exact separation. The criterion was therefore `+∞` over
  the whole region containing its own optimum, and the reported symptom was a
  line search unable to move at the seed. The saturation heuristics are gone from
  the penalized branch; the genuinely unbounded λ are still refused, by the
  conditioning and convergence machinery that measures them rather than by a
  guess. The same fixture that hard-failed now mints at every `n`, with the
  monotone, essentially linear fit the data supports (edf ≈ 1.95), and the suite
  runs in 3.1 s instead of 17.7 s because nothing burns 200 iterations at a
  refused trial point any more.

- **Firth bias reduction is now a Newton solve on every binomial link (#2273).**
  `WorkingState`'s Hessian is `XᵀWX + S` and deliberately omits the Jeffreys
  coefficient Hessian `HΦ`, because the outer Laplace layer consumes the two
  separately. Four consumers have to fold it back in, and only one did. The
  augmented-square-root direction solve folded it in by congruence — but that
  route is reached only when the realized curvature is Fisher, which is true just
  for the canonical logit link. Every non-canonical binomial link (probit,
  cloglog, …) fell through to a dense solve with the Jeffreys score in the
  gradient and no Jeffreys curvature in the matrix, and so did the constrained
  and bounded active-set solves, the post-loop undamped Newton polish, and the
  exact-decrement certificate. The result was an iteration that is not Newton for
  any objective: on the issue's 6-row separated probit fixture it contracted
  linearly at 0.4937 per step, stopped 23 iterations later at `‖g‖ = 4.3e-7`,
  failed its own convergence certificate and was refused — at a β̂ an independent
  reference confirms was the right one. The omitted term is now named once, as
  the matrix behind the quadratic correction that already existed, and folded in
  at every site through one helper that owns the sign convention. The same solve
  now reaches `‖g‖ = 1.2e-15` and clears its certificate by eleven orders, and
  `link(type=probit)`/`link(type=cloglog)` fits on separated data mint through
  the automatic Firth rescue the README promises.

- **A saturating binomial row is evaluated instead of refused (#2273).** Two
  numerical defects in the non-canonical observed-information path aborted fits
  over quantities that are perfectly representable. First, the Bernoulli variance
  was rebuilt from `μ` alone: a bounded inverse link reaches `μ == 1.0` exactly
  far inside its tail — cloglog at `η ≈ 3.62`, probit at `η ≈ 8.29` — so
  `1.0 − μ` is a hard zero while the true complement is still `1e-17`, `1e-45`,
  `1e-120`, and `V = μ(1−μ)` collapsed to zero with the whole
  observed-information jet dividing by it. The cancellation-free complement
  already existed and the sibling Fisher path already used it; it now reaches the
  variance and the working residual too. Second, the closed forms for the
  observed weight and its first two `η`-derivatives divided by `φV²`, `φV³` and
  `φV⁴`, and `V⁴` underflows to zero at `V = 2.9e-84` — so a `d²W/dη²` of
  `3.87e-75` came back NaN. Those expansions are replaced by the Leibniz
  recurrence the third derivative already used one order higher, which divides by
  `φV` once per order and never forms a power of `V`; the two orders of one object
  are now one recurrence instead of two independently-maintained expansions.
  Checked against two oracles that share no code with the engine: the exact
  identity that a cloglog row with `y = 0` has `−ℓ = e^η`, so its observed
  information and every `η`-derivative are exactly `e^η`; and mpmath at 220
  decimal digits for `y = 1`.

- **A flat-valley verdict now requires a flat valley (#2613).** The outer
  cost-stall guard — the mgcv-style stop that halts a smoothing-parameter search
  once the criterion stops improving over six consecutive **accepted** steps —
  was being fed every gradient evaluation, on the premise that the optimizer
  only asks for a gradient at points it has accepted. It does not: a
  strong-Wolfe line search evaluates the gradient at every trial that clears
  Armijo, because the curvature condition needs it. A search bisecting toward a
  point therefore reported six "steps" whose criterion values differed
  negligibly — of course they did, they were converging to a point — and the
  guard halted the fit *inside* one iteration, shipping a non-stationary
  checkpoint labelled "weakly-identified valley floor" and a refusal that
  counted line-search probes as outer iterations. The guard now consumes the
  optimizer's own accepted-step signal.
- **Outer stationarity no longer depends on where the search started (#2613).**
  The threshold a fit is judged converged against carried a component resolved
  once, at the seed, against the criterion's value *there*. Across the seeds of
  a single fit that spread the threshold over eighteen orders of magnitude, and
  a seed that happened to land somewhere absurd produced a threshold no gradient
  can fail — so the search claimed convergence against the wrong smoothing rail.
  The solver's threshold is now a function of the declared problem alone, the
  certificate keeps the per-point form it always meant, and the certificate can
  never be *stricter* than the threshold the solver was told to reach — which
  closes the "solver claimed convergence, certificate refused" family by
  construction rather than by retry.

- **A fit that has no criterion says so, everywhere (#2595).** `Summary.reml_score`
  and `raw_reml_score` were `0.0` on every exactly-interpolating Gaussian fit,
  because `UnifiedFitResult` had no way to express "no criterion exists here" and
  the exact-fit route had to write a placeholder to satisfy a finiteness contract.
  When the fitted mean reproduces the response to floating-point resolution the
  profiled scale is exactly zero and the restricted likelihood is unbounded, so
  the criterion is not small — it does not exist. It is now typed-absent, the
  boundary is recognized where the dispersion is estimated (so every entry point
  reaches the same verdict on the same data), the one constructor rejects a
  criterion at that boundary, and `Model.evidence`, `Model.bayes_factor_vs` and
  `gamfit.compare_models` refuse such a model by name instead of ranking a
  stand-in. `Summary.reml_score_unavailable` carries the explanation. Saved
  models load without a migration pass.

---

Entries for already-released versions continue in
[`CHANGELOG-ARCHIVE.md`](CHANGELOG-ARCHIVE.md).
