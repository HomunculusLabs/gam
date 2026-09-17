//! Manifold parameter decomposition (#2951).
//!
//! The object is an executable decomposition of a network's parameterized
//! computation, not a reconstruction of its activations: `crate::manifold` fits
//! `Z_i ~= sum_k a_ik g_k(t_ik)` and has no source-weight action. Here every
//! operation stays tied to the original tensors, every approximation has an exact
//! native reference, and fidelity is checked under declared finite interventions,
//! not only at the all-on point.
//!
//! # Four objects
//!
//! * **Native lift** (`lift`, `occurrence`, `apply`). A tensor registry (stable
//!   ids, shapes, alias and transpose ties, use sites, a teacher hash). A global
//!   edit acts on every tied use of a tensor and a use-specific edit acts on one
//!   occurrence; they are different experiments. Components enter through the
//!   exact residual anchor
//!
//!   ```text
//!   Theta(m) = m_Delta Theta_* + B sum_c (m_c - m_Delta) v_c
//!   ```
//!
//!   which equals `sum_c m_c P_c + m_Delta (Theta_* - sum_c P_c)` with
//!   `P_c = B v_c`, so the residual is carried exactly and never refitted. The
//!   anchor is applied matrix-free. Algebraic equality is not bitwise equality, so
//!   the all-on setting executes the original tensors on their original path.
//! * **Parameter field** (`field`). `Gamma(z) = sum_j phi_j(z) B_j` over GAM's
//!   existing bases, with fixed instances `P_c = w_c Gamma(z_c)` and
//!   `v_c = w_c phi(z_c)`. The labels `z_c` and scales `w_c` do not depend on the
//!   input; input dependence only selects or masks instances.
//! * **Ablation geometry** (`moments`, `adversary`, `supports`, `bounds`). Mask
//!   moments, the zonotope of admissible moments, witness masks, supports as
//!   hitting sets, and bounds.
//! * **Mechanism program** (`program`, `fit`, `codec`, `precision`). A typed graph
//!   of Sum, Compose, native primitives, reads and writes, and calls to shared
//!   bodies, with its interface, native reference, code and validity domain.
//!
//! Exact execution under masks is `rewrite` (MLP component coordinates),
//! `gated_rewrite` (gated activations, norms, biases, residuals) and `attention`
//! (the component query-key kernel under the source's joint softmax). Gauge and
//! operator structure is `operators`, `spectral` and `state`.
//!
//! # Types that are never coerced into one another
//!
//! * Four manifolds: a parameter-family label (fixed per instance), a
//!   computational-state coordinate (per input), an implementation-gauge
//!   coordinate, and a permitted structured-edit coordinate (e.g. a rotation
//!   angle).
//! * A global edit and a use-specific edit.
//! * A mask group (tied controls) and a macro (a packaged subgraph with
//!   independent internal controls).
//! * A quotient contract `E' T = g E` and a realization contract `T D = D' g`.
//!
//! # Evidence and inputs
//!
//! Every reported quantity carries its evidence status: exact (algebraic, or
//! exhaustive over a stated finite family), a uniform bound over a stated region
//! including numerical error, a statistical estimate with its law and standard
//! error, a counterexample, or unresolved (lower witness, upper bound, gap). A
//! result never returns a stronger status than it proved; a stochastic-mask mean,
//! an observed worst case and a certified bound are three different numbers.
//!
//! The mask domain and the fidelity tolerance are experiment declarations with no
//! default. Every other tolerance is derived (a roundoff bound, an eigengap, a
//! Lipschitz covering). Derivatives are analytic; finite differences appear only
//! in tests.
