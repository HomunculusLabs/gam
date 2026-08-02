//! #2757 — the residual-gauge curvature is held in the structure the
//! decoder-frame parameterization gives it, and the certificate it produces is
//! unchanged by that.
//!
//! The defect: `fit_diagnostics_report` materialized `H = RᵀR` as a dense
//! `param_dim × param_dim = (p·D)²` matrix and took its dense symmetric
//! eigendecomposition, at `(p·D)³` flops and `(p·D)²` memory — 45.97 GiB and
//! 60.5% of the whole fit at `p = 4096`. The per-row pinning Jacobian is
//! output-coordinate diagonal, so with a metric that does not couple output
//! coordinates `H` is exactly block diagonal: `p` blocks of `D × D`.
//!
//! These are the gates on that claim, from four independent angles:
//!
//! 1. the structured curvature equals the dense Gram **entry by entry**, so
//!    the cheap object is the same object;
//! 2. the off-block entries are structurally zero, on a fixture whose decoder
//!    is dense on every output coordinate (so this cannot be read as sparsity);
//! 3. the certificate — verdicts, pinning rank, per-generator energy fractions
//!    — is identical whether reduced from the blocks or from the dense Gram;
//! 4. the representation actually holds `p·D²` scalars rather than `(p·D)²`,
//!    which is the cost claim itself and is load-immune.
//!
//! Plus the gauge-driving arm, where the metric *does* couple output
//! coordinates: there the curvature falls back to its root, whose dual Gram
//! carries the same spectrum, and the certificate must again be unchanged.

use crate::identifiability::{FrameColumnLayout, ResidualGaugeCurvature};
use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::time::Instant;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

/// `k_atoms` planted circles in `p`-dimensional output space, assembled (not
/// fitted) so these gates measure the certificate rather than the optimizer.
///
/// `dense_tail` puts a nonzero decoder weight on **every** output coordinate of
/// every atom. Without it the frames would be axis-sparse and the
/// block-diagonality claim would be indistinguishable from "the fixture's
/// decoder happens to be sparse".
fn planted_term(n: usize, p: usize, k_atoms: usize, dense_tail: bool) -> SaeManifoldTerm {
    let mut s = 0x2757_0000_0000_0001u64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).expect("harmonic order 3"));
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let theta: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| theta[r]);
        let (phi, jet) = evaluator
            .evaluate(coords.view())
            .expect("periodic evaluate");
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, (2 * k) % p]] = 1.0;
        decoder[[2, (2 * k + 1) % p]] = 1.0;
        if dense_tail {
            for c in 0..p {
                decoder[[0, c]] = 0.05 * (lcg(&mut s) - 0.5);
                decoder[[1, c]] += 0.05 * (lcg(&mut s) - 0.5);
                decoder[[2, c]] += 0.05 * (lcg(&mut s) - 0.5);
            }
        }
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("circle{k}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .expect("atom blocks agree")
            .with_basis_second_jet(evaluator.clone()),
        );
        coord_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let logits = Array2::<f64>::from_elem((n, k_atoms), 3.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .expect("assignment blocks agree");
    let mut term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    term.set_guards_enabled(false);
    term
}

fn unit_rho(k_atoms: usize) -> SaeManifoldRho {
    SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k_atoms])
}

/// `H = Σ_n J_nᵀ M_n J_n` written out from the definition, with the per-row
/// pinning Jacobian materialized as the dense `p × param_dim` block the
/// certificate's doc comment describes. Deliberately naive: this is the
/// reference the structured builder is judged against, so it must not share
/// any of its reasoning.
fn reference_dense_gram(
    term: &SaeManifoldTerm,
    metric: &gam_problem::RowMetric,
    layout: &FrameColumnLayout,
) -> Array2<f64> {
    let n = term.n_obs();
    let p = term.output_dim();
    let param_dim = layout.param_dim();
    let assignments = term.assignment.assignments();
    let mut gram = Array2::<f64>::zeros((param_dim, param_dim));
    let mut tangent = vec![0.0_f64; p];
    let rank = metric.metric_rank();
    for row in 0..n {
        let mut j = Array2::<f64>::zeros((p, param_dim));
        let mut base = 0usize;
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            let d = atom.latent_dim();
            let a_nk = assignments[[row, atom_idx]];
            if a_nk > 0.0 {
                for axis in 0..d {
                    atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                    for i in 0..p {
                        j[[i, base + i * d + axis]] += a_nk * tangent[i];
                    }
                }
            }
            base += p * d;
        }
        // `M_n = U_n U_nᵀ`, so `JᵀM J = (U_nᵀJ)ᵀ(U_nᵀJ)`.
        let mut whitened = Array2::<f64>::zeros((rank, param_dim));
        for r in 0..rank {
            for c in 0..param_dim {
                let mut acc = 0.0_f64;
                for i in 0..p {
                    acc += metric.factor_entry(row, i, r) * j[[i, c]];
                }
                whitened[[r, c]] = acc;
            }
        }
        gram = gram + whitened.t().dot(&whitened);
    }
    gram
}

/// Gate 1 + 2 — the structured curvature IS the dense Gram, and its off-block
/// entries are structurally zero on a decoder that is dense everywhere.
#[test]
fn output_block_curvature_equals_the_dense_gram_and_has_no_off_block_mass() {
    let (n, p, k_atoms) = (48usize, 24usize, 3usize);
    let term = planted_term(n, p, k_atoms, true);
    let metric = term.diagnostic_metric().expect("metric");
    assert!(
        !metric.drives_gauge(),
        "the diagnostic fallback must be the Euclidean (non-gauge-driving) metric"
    );
    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let curvature = term
        .residual_gauge_streamed_data_curvature(&metric, &layout)
        .expect("streamed curvature");
    assert_eq!(curvature.structure_tag(), "output_block_roots");
    assert_eq!(curvature.root_rows(), n * metric.metric_rank());

    let reference = reference_dense_gram(&term, &metric, &layout);
    let structured = curvature.to_dense_gram();
    assert_eq!(structured.dim(), reference.dim());

    let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "the fixture must produce a nonzero curvature");
    let mut worst = 0.0_f64;
    let mut off_block = 0.0_f64;
    let mut off_block_reference = 0.0_f64;
    for a in 0..reference.nrows() {
        let ia = layout.output_of(a).expect("column in range");
        for b in 0..reference.ncols() {
            let ib = layout.output_of(b).expect("column in range");
            worst = worst.max((structured[[a, b]] - reference[[a, b]]).abs());
            if ia != ib {
                off_block = off_block.max(structured[[a, b]].abs());
                off_block_reference = off_block_reference.max(reference[[a, b]].abs());
            }
        }
    }
    assert!(
        worst <= 1.0e-12 * scale,
        "structured curvature must reproduce the dense Gram: worst |Δ| {worst:.3e} \
         against scale {scale:.3e}"
    );
    assert_eq!(
        off_block, 0.0,
        "the structured curvature must carry no mass between two output coordinates"
    );
    assert_eq!(
        off_block_reference, 0.0,
        "and neither does the dense reference — the block structure is the operator's, \
         not the representation's"
    );

    // Diagonal mass is spread over every output coordinate, so the zero
    // off-block is not a statement about a decoder that touches only a few.
    let touched = (0..p)
        .filter(|&i| {
            (0..k_atoms).any(|l| {
                let c = layout.column(i, l);
                reference[[c, c]].abs() > 0.0
            })
        })
        .count();
    assert_eq!(
        touched, p,
        "every output coordinate must carry curvature for this gate to bite"
    );
}

/// Gate 4 — the cost claim, as an exact count rather than a wall-clock
/// threshold: the curvature holds `p·D²` scalars where the dense Gram holds
/// `(p·D)²`.
#[test]
fn output_block_curvature_stores_p_times_fewer_scalars_than_the_dense_gram() {
    let (n, p, k_atoms) = (24usize, 64usize, 4usize);
    let term = planted_term(n, p, k_atoms, true);
    let metric = term.diagnostic_metric().expect("metric");
    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let curvature = term
        .residual_gauge_streamed_data_curvature(&metric, &layout)
        .expect("streamed curvature");
    let param_dim = layout.param_dim();
    let d = layout.block_dim();
    assert_eq!(curvature.stored_scalars(), p * d * d);
    assert_eq!(
        curvature.stored_scalars() * p,
        param_dim * param_dim,
        "the saving is exactly the factor p the dense layout was padding by"
    );
}

/// Gate 3 — the certificate is identical whichever representation it is
/// reduced from. This is the one that makes the change a cost change and not a
/// behaviour change.
#[test]
fn certificate_is_identical_under_the_structured_and_dense_reductions() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms) = (40usize, 20usize, 3usize);
    let term = planted_term(n, p, k_atoms, true);
    let metric = term.diagnostic_metric().expect("metric");
    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    let structured = streamed.expect("unpinned path streams its curvature");
    assert_eq!(structured.structure_tag(), "output_block_roots");
    let dense = ResidualGaugeCurvature::DenseGram {
        gram: structured.to_dense_gram(),
        root_rows: structured.root_rows(),
    };

    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        vec![None; model.atoms.len()];
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();

    let from_blocks = residual_gauge_exact_from_curvature(&model, &views, &ops, structured)
        .expect("structured certificate");
    let from_dense = residual_gauge_exact_from_curvature(&model, &views, &ops, dense)
        .expect("dense certificate");

    assert_eq!(
        from_blocks.pinning_rank, from_dense.pinning_rank,
        "pinning rank must not depend on the representation"
    );
    assert_eq!(from_blocks.generators.len(), from_dense.generators.len());
    assert!(
        !from_blocks.generators.is_empty(),
        "the fixture must enumerate generators for this gate to bite"
    );
    for (b, dsn) in from_blocks
        .generators
        .iter()
        .zip(from_dense.generators.iter())
    {
        assert_eq!(b.description, dsn.description);
        assert_eq!(b.family, dsn.family);
        assert_eq!(
            b.unpinned, dsn.unpinned,
            "generator '{}' verdict must not depend on the representation",
            b.description
        );
        let gap = (b.pinned_energy_fraction - dsn.pinned_energy_fraction).abs();
        assert!(
            gap <= 1.0e-12,
            "generator '{}' energy fraction differs by {gap:.3e} between representations",
            b.description
        );
    }
    assert_eq!(from_blocks.group_signature(), from_dense.group_signature());
    assert_eq!(
        from_blocks.residual_gauge_dim,
        from_dense.residual_gauge_dim
    );
}

/// A curvature with rank-deficient blocks: half the output coordinates carry no
/// decoder mass at all, so their blocks are exactly zero and must contribute
/// nothing to the pinning rank — the same answer the dense spectrum gives.
#[test]
fn rank_deficient_blocks_agree_with_the_dense_spectrum() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms) = (32usize, 16usize, 2usize);
    // `dense_tail = false` leaves each atom's decoder supported on two output
    // coordinates, so most blocks are identically zero.
    let term = planted_term(n, p, k_atoms, false);
    let metric = term.diagnostic_metric().expect("metric");
    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    let structured = streamed.expect("unpinned path streams its curvature");
    let dense = ResidualGaugeCurvature::DenseGram {
        gram: structured.to_dense_gram(),
        root_rows: structured.root_rows(),
    };
    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        vec![None; model.atoms.len()];
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let from_blocks = residual_gauge_exact_from_curvature(&model, &views, &ops, structured)
        .expect("structured certificate");
    let from_dense = residual_gauge_exact_from_curvature(&model, &views, &ops, dense)
        .expect("dense certificate");
    assert!(
        from_blocks.pinning_rank < p * k_atoms,
        "the fixture must be rank deficient for this gate to bite (rank {} of {})",
        from_blocks.pinning_rank,
        p * k_atoms
    );
    assert_eq!(from_blocks.pinning_rank, from_dense.pinning_rank);
}

/// The gauge-driving arm: an output-Fisher metric couples output coordinates,
/// so the curvature is NOT block diagonal and the builder must say so — and the
/// root it returns must still reproduce the dense Gram exactly.
#[test]
fn gauge_driving_metric_falls_back_to_a_root_that_reproduces_the_dense_gram() {
    let (n, p, k_atoms, rank) = (12usize, 10usize, 2usize, 3usize);
    let mut term = planted_term(n, p, k_atoms, true);
    let mut s = 0x2757_FEED_0000_0001u64;
    let factors = Array2::<f64>::from_shape_fn((n, p * rank), |_| lcg(&mut s) - 0.5);
    let metric = gam_problem::RowMetric::output_fisher(Arc::new(factors), p, rank)
        .expect("output-Fisher metric");
    term.set_row_metric(metric.clone())
        .expect("metric is conformable with the term");
    assert!(metric.drives_gauge());

    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let curvature = term
        .residual_gauge_streamed_data_curvature(&metric, &layout)
        .expect("streamed curvature");
    // n·rank = 36 root rows against param_dim = 20 columns, so the root is the
    // larger object and the dense Gram is the right store.
    assert_eq!(curvature.structure_tag(), "dense_gram");

    let reference = reference_dense_gram(&term, &metric, &layout);
    let built = curvature.to_dense_gram();
    let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0);
    let worst = built
        .iter()
        .zip(reference.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 1.0e-12 * scale,
        "gauge-driving curvature must reproduce the dense Gram: worst |Δ| {worst:.3e}"
    );
    // A gauge-driving metric genuinely couples output coordinates, so the block
    // structure must NOT be claimed here.
    let mut off_block = 0.0_f64;
    for a in 0..reference.nrows() {
        let ia = layout.output_of(a).expect("in range");
        for b in 0..reference.ncols() {
            if ia != layout.output_of(b).expect("in range") {
                off_block = off_block.max(reference[[a, b]].abs());
            }
        }
    }
    assert!(
        off_block > 0.0,
        "an output-Fisher metric must couple output coordinates, else this arm is vacuous"
    );
}

/// The same arm at a shape where the root has fewer rows than columns.
///
/// This is also the gate on the *rank* half of #2757. `H = RᵀR` with 12 root
/// rows in 80 parameters has rank at most 12 — a mathematical bound, not an
/// estimate. Before the fix the dense-Gram path reported **45**, because the
/// tolerance `τ = 100·ε·N·σ_max` (deliberately 100x above an SVD's backward
/// error) was being squared into `τ²`, which lands ~10⁹x BELOW a symmetric
/// eigensolver's own resolution, so 33 roundoff eigenvalues cleared it. Both
/// representations must now agree, and both must respect the bound.
#[test]
fn dual_root_and_dense_gram_agree_on_a_rank_neither_may_exceed() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms, rank) = (6usize, 40usize, 2usize, 2usize);
    let mut term = planted_term(n, p, k_atoms, true);
    let mut s = 0x2757_FEED_0000_0002u64;
    let factors = Array2::<f64>::from_shape_fn((n, p * rank), |_| lcg(&mut s) - 0.5);
    let metric = gam_problem::RowMetric::output_fisher(Arc::new(factors), p, rank)
        .expect("output-Fisher metric");
    term.set_row_metric(metric.clone())
        .expect("metric is conformable");

    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let curvature = term
        .residual_gauge_streamed_data_curvature(&metric, &layout)
        .expect("streamed curvature");
    // n*rank = 12 root rows against param_dim = 80 columns.
    assert_eq!(curvature.structure_tag(), "dual_root");
    assert_eq!(curvature.stored_scalars(), n * rank * layout.param_dim());

    let reference = reference_dense_gram(&term, &metric, &layout);
    let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = curvature
        .to_dense_gram()
        .iter()
        .zip(reference.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 1.0e-12 * scale,
        "dual-root curvature must reproduce the dense Gram: worst |Δ| {worst:.3e}"
    );

    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    let structured = streamed.expect("unpinned path streams its curvature");
    let root_rows = structured.root_rows();
    let dense = ResidualGaugeCurvature::DenseGram {
        gram: structured.to_dense_gram(),
        root_rows,
    };
    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let from_root = residual_gauge_exact_from_curvature(&model, &views, &ops, structured)
        .expect("dual-root certificate");
    let from_dense = residual_gauge_exact_from_curvature(&model, &views, &ops, dense)
        .expect("dense certificate");

    let bound = root_rows.min(layout.param_dim());
    assert!(
        from_root.pinning_rank <= bound,
        "rank(RᵀR) <= rows(R): root path reported {} against the bound {bound}",
        from_root.pinning_rank
    );
    assert!(
        from_dense.pinning_rank <= bound,
        "rank(RᵀR) <= rows(R): dense-Gram path reported {} against the bound {bound} —          the Gram is counting eigenvalues below its own resolution",
        from_dense.pinning_rank
    );
    assert_eq!(
        from_root.pinning_rank, from_dense.pinning_rank,
        "the two representations must decide one rank"
    );
    for (r, d) in from_root
        .generators
        .iter()
        .zip(from_dense.generators.iter())
    {
        assert_eq!(r.unpinned, d.unpinned, "generator '{}'", r.description);
        assert!(
            (r.pinned_energy_fraction - d.pinned_energy_fraction).abs() <= 1.0e-10,
            "generator '{}' energy fraction",
            r.description
        );
    }
}

/// The wall #2757 was filed on, measured on the fixed path. The dense path's
/// own numbers on this fixture (committed in `854ed7caa`) were
/// `eigh_s` 0.157 / 0.970 / 7.013 at `p` = 256 / 512 / 1024 — a clean cubic and
/// 88.1% of the whole report at the widest cell.
#[test]
fn diagnostics_report_no_longer_grows_cubically_in_the_output_dimension() {
    let (n, k_atoms) = (48usize, 4usize);
    let mut timings: Vec<(usize, f64)> = Vec::new();
    println!("\n#2757: fit_diagnostics_report on the structured curvature (n={n}, K={k_atoms})");
    for &p in &[256usize, 512, 1024] {
        let term = planted_term(n, p, k_atoms, true);
        let rho = unit_rho(k_atoms);
        let fitted = term
            .try_fitted_target_aware(Array2::<f64>::zeros((n, p)).view(), Some(&rho))
            .expect("fitted");
        let metric = term.diagnostic_metric().expect("metric");
        let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
        let curvature = term
            .residual_gauge_streamed_data_curvature(&metric, &layout)
            .expect("streamed curvature");
        assert_eq!(curvature.structure_tag(), "output_block_roots");
        assert_eq!(curvature.stored_scalars(), p * k_atoms * k_atoms);

        let started = Instant::now();
        let report = term
            .fit_diagnostics_report(None, false, None, fitted.view(), None)
            .expect("diagnostics report");
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "  p={p:>5} param_dim={:>6} report {seconds:>8.3}s  pinning_rank={}",
            p * k_atoms,
            report.residual_gauge.pinning_rank
        );
        timings.push((p, seconds));
    }
    // Cubic in `param_dim ∝ p` is 8x per doubling; the dense path measured 6.2x
    // and 7.2x here. The structured path is linear in `p` at fixed `D`, so a
    // generous 4x ceiling separates the two regimes without turning host noise
    // into a red test.
    for pair in timings.windows(2) {
        let (p_lo, t_lo) = pair[0];
        let (p_hi, t_hi) = pair[1];
        let growth = t_hi / t_lo.max(1.0e-6);
        assert!(
            growth <= 4.0,
            "doubling p from {p_lo} to {p_hi} multiplied the report by {growth:.2}x; \
             the certification cost must not be cubic in the output dimension"
        );
    }
}
