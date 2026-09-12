//! Outer-Hessian representation routing.
//!
//! Cost — never capability — selects between the dense `K × K` assembly and
//! the matrix-free Hv-operator, and both deliver the same math. The operator
//! build pays the same per-coordinate and pair work as the dense assembly:
//! every coordinate's drift correction and spectral rotation, the leverage and
//! adjoint caches, the fourth-derivative trace matrix and the `K²` pair cross
//! traces. The operator then pays its products on top. On the Gaussian kernel
//! a product contracts the precomputed tables and the operator is densified
//! for ARC; on the scalar-GLM and callback kernels every product re-traces, and
//! a matrix-free trust-region solve takes up to `K` products at CG's worst
//! case. So the dense assembly never costs more, and the operator is taken only
//! when the dense workspace exceeds the memory governor's single-materialization
//! cap.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OuterHessianRoutePlan {
    pub use_operator: bool,
    pub reason: &'static str,
    pub scale_prefers_operator: bool,
    pub dense_workspace_bytes: usize,
}

impl OuterHessianRoutePlan {
    pub(crate) fn choice(self) -> &'static str {
        if self.use_operator {
            "operator"
        } else {
            "dense"
        }
    }
}

pub(crate) fn saturating_f64_matrix_bytes(rows: usize, cols: usize) -> usize {
    rows.saturating_mul(cols)
        .saturating_mul(std::mem::size_of::<f64>())
}

pub(crate) fn outer_hessian_dense_workspace_bytes(p: usize, k: usize) -> usize {
    // Dense assembly keeps first-order drifts for each coordinate and uses at
    // least one transient second-order drift while filling the K x K Hessian.
    // Charge a small safety multiple so the route never depends on fitting a
    // single p x p matrix while the actual dense path holds several.
    let drift_count = k.saturating_mul(2).saturating_add(3).max(1);
    saturating_f64_matrix_bytes(p, p).saturating_mul(drift_count)
}

pub(crate) fn outer_hessian_dense_workspace_budget_bytes() -> usize {
    gam_runtime::resource::ResourcePolicy::default_library().max_single_materialization_bytes
}

pub(crate) fn dense_outer_hessian_workspace_fits(p: usize, k: usize) -> bool {
    outer_hessian_dense_workspace_bytes(p, k) <= outer_hessian_dense_workspace_budget_bytes()
}

/// Outer-Hessian route for `p` coefficients and `k` outer coordinates: the
/// dense assembly whenever its workspace fits the materialization cap, the
/// matrix-free operator otherwise. `kernel_available` says whether the operator
/// can be built at all, and `subspace_trace` only relabels an operator route.
pub fn outer_hessian_route_plan(
    p: usize,
    k: usize,
    kernel_available: bool,
    subspace_trace: bool,
) -> OuterHessianRoutePlan {
    let dense_workspace_bytes = outer_hessian_dense_workspace_bytes(p, k);
    if !kernel_available {
        return OuterHessianRoutePlan {
            use_operator: false,
            reason: "kernel_absent",
            scale_prefers_operator: false,
            dense_workspace_bytes,
        };
    }
    if dense_outer_hessian_workspace_fits(p, k) {
        return OuterHessianRoutePlan {
            use_operator: false,
            reason: "dense_workspace_fits",
            scale_prefers_operator: false,
            dense_workspace_bytes,
        };
    }
    if subspace_trace {
        return OuterHessianRoutePlan {
            use_operator: true,
            reason: "subspace_projected_operator",
            scale_prefers_operator: true,
            dense_workspace_bytes,
        };
    }
    OuterHessianRoutePlan {
        use_operator: true,
        reason: "dense_memory_budget",
        scale_prefers_operator: true,
        dense_workspace_bytes,
    }
}
