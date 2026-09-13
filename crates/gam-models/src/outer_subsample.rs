//! Outer-score row subsampling and the model-layer row-measure bridge.
//!
//! The canonical definitions of [`OuterScoreSubsample`], [`RowSet`], and
//! [`WeightedOuterRow`] are neutral low-layer primitives that live in the
//! `gam-problem` crate so the `CustomFamily` trait layer (and the model
//! families above it) can depend on them downward without duplication. This
//! module is the single model-layer authority that translates those primitives
//! into custom-family fit options.

pub use gam_problem::outer_subsample::*;

/// Derive the exact family-facing outer options for the spatial optimizer,
/// which evaluates every row.
///
/// This is the sole bridge from the spatial driver into custom-family options.
/// It clears both an inherited pilot mask and automatic sampling; otherwise a
/// pilot mask could reach the family and make the inner mode, objective,
/// gradient, and Hessian describe different measures.
pub(crate) fn exact_outer_options(
    options: &crate::custom_family::BlockwiseFitOptions,
) -> crate::custom_family::BlockwiseFitOptions {
    let mut effective = options.clone();
    effective.auto_outer_subsample = false;
    effective.outer_score_subsample = None;
    effective
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_outer_options_bind_analytic_and_efs_lanes_to_one_row_measure() {
        let mut options = crate::custom_family::BlockwiseFitOptions::default();
        options.auto_outer_subsample = true;
        options.outer_score_subsample = Some(std::sync::Arc::new(
            OuterScoreSubsample::from_uniform_inclusion_mask(vec![9], 10, 17),
        ));

        let full = exact_outer_options(&options);
        assert!(!full.auto_outer_subsample);
        assert!(
            full.outer_score_subsample.is_none(),
            "full-data replay must clear every stale pilot mask"
        );
        assert!(
            options.outer_score_subsample.is_some(),
            "deriving the effective measure must not mutate caller options"
        );
    }
}
