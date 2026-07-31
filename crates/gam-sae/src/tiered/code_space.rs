//! The code-space curvature pass: Tier-2's *first* substrate is the Tier-1
//! CODE GROUPS, not the Tier-1 residual (#2023 / #2502).
//!
//! # Why the residual substrate is blind by construction
//!
//! A centered closed curve's reconstruction cone IS its plane: sweeping radius
//! and phase over a centered circle fills the 2-plane the circle spans, so two
//! linear atoms reconstruct the ring **exactly** and the block's linear residual
//! is identically zero. A Tier-2 that charts the post-Tier-1 residual therefore
//! cannot see precisely the curvature the tiered model exists to find — not
//! because the linear fit is poor but because it is perfect
//! ([`crate::manifold::curve_promotion`] states the impossibility result; the
//! campaign measured its corollary as the EV null at matched budget in #2502).
//! Where curvature IS visible is the **measure**: the joint law of the block's
//! code amplitudes. A ring in code space is a constraint AMONG the directions
//! Tier-1 already spans, and the only statistics with power against the flat
//! null are description-length bits and the radial law — never residual EV.
//!
//! # What this pass does
//!
//! Each Tier-1 block is already a co-firing linear community: its `b` decoder
//! rows are the atoms, and the rows that routed to it carry a `b`-dimensional
//! code cloud. This pass walks every fitted block, hands its community to the
//! atomic bits-denominated adjudicator
//! ([`crate::manifold::curve_promotion::propose_curve_promotion`] — previously a
//! pure producer with no consumer), and reports, per block, whether ONE curved
//! chart describes the same firings in fewer bits than the block's linear atoms.
//! The decoder stays linear either way — an accepted promotion bends the CODE
//! (one phase + one amplitude replace `b` amplitudes), not the dictionary.
//!
//! Every quantity the ledger prices is measured, not configured: the dictionary
//! size and the mean active coordinates per token are read off the fit, and the
//! distortion floor is the reconstruction tolerance the linear tier actually
//! achieved ([`linear_distortion_floor`]). The pass introduces no knob.
//!
//! The report's `fraction_curved` is the honesty measurement the field actually
//! wants from a manifold SAE: how much of the dictionary curves, feature by
//! feature, with the refusals recorded next to the acceptances.

use ndarray::{Array2, ArrayView2};

use crate::atom_codes::SparseAtomCodes;
use crate::manifold::curve_promotion::{
    CurvePromotionProposal, LinearCommunity, PromotionContext, propose_curve_promotion,
};
use crate::sparse_dict::BlockSparseFit;

/// The code-space curvature census over one fitted Tier-1 dictionary: every
/// per-block promotion proposal (accepted AND refused — the refusals are part of
/// the measurement), plus the measured context the ledger priced against.
#[derive(Clone, Debug)]
pub struct CodeSpacePromotionReport {
    /// One proposal per block whose community reached the adjudicator (blocks
    /// with ≥ 2 firings and a ≥ 2-dimensional code plane). `accept` on each entry
    /// is the atomic DL verdict; nothing here mutates the fit.
    pub proposals: Vec<CurvePromotionProposal>,
    /// Cross-block shell census: one proposal per CO-FIRING block pair whose
    /// joint code cloud reached the adjudicator, keyed by the pair. A ring the
    /// dictionary shattered across two blocks (one straight atom per half — the
    /// shape the Gemma Scope census found as "pairs of straight atoms whose
    /// joint amplitude law is a shell") is invisible to every single-block
    /// community; the union community is where it lives.
    pub pair_proposals: Vec<(usize, usize, CurvePromotionProposal)>,
    /// Total blocks in the Tier-1 dictionary (`G`).
    pub n_blocks_scanned: usize,
    /// Blocks whose community yielded a proposal (the eligible denominator).
    pub n_communities: usize,
    /// Proposals the atomic bits ledger accepted.
    pub n_accepted: usize,
    /// Total bits the accepted promotions save (`Σ dl_old − dl_new` over
    /// accepted proposals).
    pub dl_saved_bits: f64,
    /// `n_accepted / n_communities` (`0.0` when no community was eligible) —
    /// the fraction of the co-firing dictionary that genuinely curves.
    pub fraction_curved: f64,
    /// The measured per-coordinate distortion floor `δ` the ledger priced at.
    pub tolerance: f64,
    /// The measured mean active scalar coordinates per token (`L0`).
    pub l0: f64,
}

/// The measured per-coordinate distortion floor `δ` for the code-space ledger:
/// the RMS per-element reconstruction error the linear tier actually achieved,
/// floored at the corpus's own f64 measurement resolution
/// (`RMS(R0) · √ε`) so an exactly-reconstructed corpus still carries a positive
/// quantisation cell instead of a zero that would poison the rate terms.
///
/// `residual` is the post-Tier-1 centered residual; `baseline_energy` is
/// `‖R0‖²` over the same `N×P` grid (the Tier-0 baseline the tiered EV is
/// measured against). Errors when the corpus itself carries no energy — with
/// nothing to reconstruct there is no distortion scale to measure.
pub fn linear_distortion_floor(
    residual: ArrayView2<'_, f64>,
    baseline_energy: f64,
) -> Result<f64, String> {
    let n_elems = residual.len();
    if n_elems == 0 {
        return Err("linear_distortion_floor: empty residual".to_string());
    }
    if !(baseline_energy > 0.0 && baseline_energy.is_finite()) {
        return Err(format!(
            "linear_distortion_floor: baseline energy must be finite and > 0, got {baseline_energy}"
        ));
    }
    let residual_ms = residual.iter().map(|&r| r * r).sum::<f64>() / n_elems as f64;
    if !residual_ms.is_finite() {
        return Err(format!(
            "linear_distortion_floor: residual energy is not finite ({residual_ms})"
        ));
    }
    let corpus_rms = (baseline_energy / n_elems as f64).sqrt();
    let resolution_floor = corpus_rms * f64::EPSILON.sqrt();
    Ok(residual_ms.sqrt().max(resolution_floor))
}

/// Walk every Tier-1 block as a linear community and adjudicate its code cloud
/// for curved replacement, in bits. Pure: reads the fit, mutates nothing.
///
/// `n_tokens` is the corpus row count `N` the firing rate is measured against;
/// `tolerance` is the per-coordinate distortion floor `δ` (use
/// [`linear_distortion_floor`] for the measured one).
pub fn harvest_code_space_promotions(
    tier1: &BlockSparseFit,
    n_tokens: usize,
    tolerance: f64,
) -> Result<CodeSpacePromotionReport, String> {
    let b = tier1.block_size;
    let p = tier1.decoder.ncols();
    let k_atoms = tier1.decoder.nrows();
    if b == 0 || p == 0 || k_atoms == 0 || k_atoms % b != 0 {
        return Err(format!(
            "harvest_code_space_promotions: malformed block geometry (K={k_atoms}, b={b}, P={p})"
        ));
    }
    let n_blocks = k_atoms / b;
    let n_rows = tier1.blocks.nrows();
    if n_tokens < n_rows {
        return Err(format!(
            "harvest_code_space_promotions: n_tokens {n_tokens} < routed rows {n_rows}"
        ));
    }

    // One pass over the sparse routing: per-block firing code lists (flattened
    // f×b), the co-firing block-pair joint code lists (flattened f×2b), and the
    // measured mean active scalar coordinates per token (L0).
    let mut firings: Vec<Vec<f64>> = vec![Vec::new(); n_blocks];
    let mut pair_firings: std::collections::BTreeMap<(usize, usize), Vec<f64>> =
        std::collections::BTreeMap::new();
    let mut active_scalars = 0usize;
    let mut row_live: Vec<(usize, usize)> = Vec::with_capacity(tier1.block_topk);
    for i in 0..n_rows {
        row_live.clear();
        for j in 0..tier1.block_topk {
            if tier1.gates[[i, j]] == 0.0 {
                continue; // padded slot: no live block routed here
            }
            let g = tier1.blocks[[i, j]] as usize;
            if g >= n_blocks {
                return Err(format!(
                    "harvest_code_space_promotions: routed block {g} out of range G={n_blocks}"
                ));
            }
            for r in 0..b {
                let code = tier1.codes[[i, j, r]] as f64;
                if code != 0.0 {
                    active_scalars += 1;
                }
                firings[g].push(code);
            }
            row_live.push((g, j));
        }
        // Joint clouds for every co-firing pair on this row (ordered by block id
        // so the pair key and its code-column order are canonical).
        for a in 0..row_live.len() {
            for c in (a + 1)..row_live.len() {
                let (mut ga, mut ja) = row_live[a];
                let (mut gb, mut jb) = row_live[c];
                if ga == gb {
                    continue;
                }
                if ga > gb {
                    std::mem::swap(&mut ga, &mut gb);
                    std::mem::swap(&mut ja, &mut jb);
                }
                let joint = pair_firings.entry((ga, gb)).or_default();
                for r in 0..b {
                    joint.push(tier1.codes[[i, ja, r]] as f64);
                }
                for r in 0..b {
                    joint.push(tier1.codes[[i, jb, r]] as f64);
                }
            }
        }
    }
    let l0 = active_scalars as f64 / n_tokens as f64;

    let ctx = PromotionContext {
        n_tokens: n_tokens as f64,
        g_dict: k_atoms,
        l0,
        tolerance,
    };

    let mut proposals = Vec::new();
    let mut n_communities = 0usize;
    let mut n_accepted = 0usize;
    let mut dl_saved_bits = 0.0f64;
    for g in 0..n_blocks {
        let f = firings[g].len() / b;
        if f < 2 {
            continue;
        }
        // The block's atoms (b×P, f64) and its code cloud (f×b, f64).
        let mut atoms = Array2::<f64>::zeros((b, p));
        for r in 0..b {
            let src = tier1.decoder.row(g * b + r);
            for c in 0..p {
                atoms[[r, c]] = src[c] as f64;
            }
        }
        let codes = Array2::from_shape_vec((f, b), std::mem::take(&mut firings[g]))
            .map_err(|err| format!("harvest_code_space_promotions: code reshape failed: {err}"))?;
        let community = LinearCommunity {
            block_id: g,
            atoms: atoms.view(),
            codes: codes.view(),
        };
        let Some(proposal) = propose_curve_promotion(community, &ctx)? else {
            continue; // rank-1 / point community: no plane to host a ring
        };
        n_communities += 1;
        if proposal.accept {
            n_accepted += 1;
            dl_saved_bits += proposal.dl_old - proposal.dl_new;
        }
        proposals.push(proposal);
    }
    // Cross-block shells: adjudicate every co-firing pair's JOINT cloud. A ring
    // shattered across two blocks presents per block as a line (refused above)
    // and only closes in the union community.
    let mut pair_proposals = Vec::new();
    for ((ga, gb), flat) in pair_firings {
        let s = 2 * b;
        let f = flat.len() / s;
        if f < 2 {
            continue;
        }
        let mut atoms = Array2::<f64>::zeros((s, p));
        for (slot, block) in [ga, gb].into_iter().enumerate() {
            for r in 0..b {
                let src = tier1.decoder.row(block * b + r);
                for col in 0..p {
                    atoms[[slot * b + r, col]] = src[col] as f64;
                }
            }
        }
        let codes = Array2::from_shape_vec((f, s), flat)
            .map_err(|err| format!("harvest_code_space_promotions: pair reshape failed: {err}"))?;
        let community = LinearCommunity {
            block_id: ga,
            atoms: atoms.view(),
            codes: codes.view(),
        };
        let Some(proposal) = propose_curve_promotion(community, &ctx)? else {
            continue;
        };
        n_communities += 1;
        if proposal.accept {
            n_accepted += 1;
            dl_saved_bits += proposal.dl_old - proposal.dl_new;
        }
        pair_proposals.push((ga, gb, proposal));
    }

    let fraction_curved = if n_communities > 0 {
        n_accepted as f64 / n_communities as f64
    } else {
        0.0
    };

    Ok(CodeSpacePromotionReport {
        proposals,
        pair_proposals,
        n_blocks_scanned: n_blocks,
        n_communities,
        n_accepted,
        dl_saved_bits,
        fraction_curved,
        tolerance,
        l0,
    })
}

/// The atom-level census over ANY dictionary: every co-firing ATOM pair of an
/// arbitrary atom bank (`decoder`, `K×P`) with solved codes on it is adjudicated
/// for curved replacement, in bits. This is the universal-improver entry — the
/// input dictionary can be a TopK / JumpReLU / block-sparse decoder imported
/// from anywhere; the census consumes only its atoms and its code measure, which
/// is exactly where the Gemma Scope shells live (pairs of straight atoms whose
/// joint amplitude law is a shell). Communities are the observed co-firing
/// pairs; `G` and `L0` are measured off the codes; nothing here mutates the
/// dictionary.
///
/// The report reuses [`CodeSpacePromotionReport`]: pair verdicts land in
/// `pair_proposals` keyed by the atom pair (single-atom `proposals` stays empty
/// — a lone atom has no plane).
pub fn harvest_code_space_pair_promotions(
    decoder: ArrayView2<'_, f64>,
    codes: &SparseAtomCodes,
    n_tokens: usize,
    tolerance: f64,
) -> Result<CodeSpacePromotionReport, String> {
    let k_atoms = decoder.nrows();
    let p = decoder.ncols();
    if k_atoms == 0 || p == 0 {
        return Err(format!(
            "harvest_code_space_pair_promotions: empty decoder ({k_atoms}×{p})"
        ));
    }
    if codes.k_atoms() != k_atoms {
        return Err(format!(
            "harvest_code_space_pair_promotions: codes carry K={} but the decoder has K={k_atoms}",
            codes.k_atoms()
        ));
    }
    let n_rows = codes.n_obs();
    if n_tokens < n_rows {
        return Err(format!(
            "harvest_code_space_pair_promotions: n_tokens {n_tokens} < coded rows {n_rows}"
        ));
    }

    // Pass 0 — count. Per-pair joint-firing counts and the measured mean active
    // atoms per token (L0). Counting first is what makes the census tractable at
    // SAE scale: with L0 ≈ 60 a row contributes ~1.8k pairs, and materialising a
    // weight cloud for every observed pair would be O(10⁸) allocations.
    let mut pair_counts: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    let mut active_total = 0usize;
    let mut support: Vec<(usize, f64)> = Vec::new();
    for row in codes.iter() {
        let n_active = row.active_mask.count_ones();
        active_total += n_active;
        support.clear();
        for atom in row.active_mask.iter_ones() {
            support.push((atom, row.weights[atom]));
        }
        for a in 0..support.len() {
            for b in (a + 1)..support.len() {
                let key = ((support[a].0 as u64) << 32) | support[b].0 as u64;
                *pair_counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    let l0 = active_total as f64 / n_tokens as f64;
    let ctx = PromotionContext {
        n_tokens: n_tokens as f64,
        g_dict: k_atoms,
        l0,
        tolerance,
    };

    // The ledger's own admission floor, not a knob: a circle's code term in the
    // prescreen is exactly zero, so acceptance REQUIRES the support dividend to
    // beat the dictionary surcharge — `f·(ŝ−1)·log₂(G/L0) > (m−ŝ)·P·½log₂N`,
    // and the circle (ŝ=2, m=3) minimises the right/left ratio over the raced
    // topologies. Pairs co-firing fewer than f_min times can therefore never be
    // accepted, and skipping them prunes the candidate set from O(L0²·N) to the
    // handful that could pay.
    let unit_sel = if l0 > 0.0 {
        (k_atoms as f64 / l0).log2().max(0.0)
    } else {
        0.0
    };
    let log2_n = if n_tokens >= 2 { (n_tokens as f64).log2() } else { 0.0 };
    let f_min = if unit_sel > 0.0 {
        ((p as f64 * 0.5 * log2_n) / unit_sel).ceil().max(2.0) as u32
    } else {
        u32::MAX
    };

    // Pass 1 — accumulate weight clouds only for admissible pairs.
    let mut pair_firings: std::collections::BTreeMap<(usize, usize), Vec<f64>> =
        std::collections::BTreeMap::new();
    for row in codes.iter() {
        support.clear();
        for atom in row.active_mask.iter_ones() {
            support.push((atom, row.weights[atom]));
        }
        for a in 0..support.len() {
            for b in (a + 1)..support.len() {
                let (atom_a, w_a) = support[a];
                let (atom_b, w_b) = support[b];
                let key = ((atom_a as u64) << 32) | atom_b as u64;
                if pair_counts.get(&key).copied().unwrap_or(0) < f_min {
                    continue;
                }
                let joint = pair_firings.entry((atom_a, atom_b)).or_default();
                joint.push(w_a);
                joint.push(w_b);
            }
        }
    }

    let mut pair_proposals = Vec::new();
    let mut n_communities = 0usize;
    let mut n_accepted = 0usize;
    let mut dl_saved_bits = 0.0f64;
    for ((atom_a, atom_b), flat) in pair_firings {
        let f = flat.len() / 2;
        if f < 2 {
            continue;
        }
        let mut atoms = Array2::<f64>::zeros((2, p));
        atoms.row_mut(0).assign(&decoder.row(atom_a));
        atoms.row_mut(1).assign(&decoder.row(atom_b));
        let pair_codes = Array2::from_shape_vec((f, 2), flat).map_err(|err| {
            format!("harvest_code_space_pair_promotions: pair reshape failed: {err}")
        })?;
        let community = LinearCommunity {
            block_id: atom_a,
            atoms: atoms.view(),
            codes: pair_codes.view(),
        };
        let Some(proposal) = propose_curve_promotion(community, &ctx)? else {
            continue;
        };
        n_communities += 1;
        if proposal.accept {
            n_accepted += 1;
            dl_saved_bits += proposal.dl_old - proposal.dl_new;
        }
        pair_proposals.push((atom_a, atom_b, proposal));
    }
    let fraction_curved = if n_communities > 0 {
        n_accepted as f64 / n_communities as f64
    } else {
        0.0
    };

    Ok(CodeSpacePromotionReport {
        proposals: Vec::new(),
        pair_proposals,
        n_blocks_scanned: k_atoms,
        n_communities,
        n_accepted,
        dl_saved_bits,
        fraction_curved,
        tolerance,
        l0,
    })
}

#[cfg(test)]
mod code_space_tests {
    use super::*;
    use crate::sparse_dict::BlockSparseConvergence;
    use ndarray::{Array2 as A2, Array3};
    use std::f64::consts::TAU;

    /// Mint an OVERCOMPLETE fit (`G` blocks of `b=2`, only block 0 ever fired)
    /// whose routing is hand-authored: every row fires block 0 with the supplied
    /// 2-D code. The fired atoms are `e0, e1` in `P` ambient dims; the remaining
    /// blocks are engineered capacity that never routes. The width matters: a
    /// circle's promotion is funded ONLY by the support dividend
    /// `(ŝ−1)·log₂(G/L0)` (its code term is exactly zero), so at `G = L0` the
    /// prescreen provably defers every ring — overcompleteness is what the
    /// compression win is made of.
    fn one_fired_block_fit(codes2d: &A2<f64>, p: usize, n_blocks: usize) -> BlockSparseFit {
        let n = codes2d.nrows();
        let mut decoder = A2::<f32>::zeros((2 * n_blocks, p));
        decoder[[0, 0]] = 1.0;
        decoder[[1, 1]] = 1.0;
        let blocks = A2::<u32>::zeros((n, 1));
        let mut gates = A2::<f32>::zeros((n, 1));
        let mut codes = Array3::<f32>::zeros((n, 1, 2));
        for i in 0..n {
            let (a, b) = (codes2d[[i, 0]], codes2d[[i, 1]]);
            gates[[i, 0]] = ((a * a + b * b) as f32).sqrt();
            codes[[i, 0, 0]] = a as f32;
            codes[[i, 0, 1]] = b as f32;
        }
        let mut block_utilization = vec![0.0f32; n_blocks];
        block_utilization[0] = 1.0;
        BlockSparseFit {
            decoder,
            blocks,
            gates,
            codes,
            gamma: 1.0,
            block_utilization,
            block_stable_rank: vec![1.0; n_blocks],
            matryoshka_prefix_losses: Vec::new(),
            explained_variance: 1.0,
            epochs: 0,
            convergence: BlockSparseConvergence::trivially_converged(),
            block_topk: 1,
            block_size: 2,
        }
    }

    /// The move class the residual substrate cannot make, now made from the
    /// tiered lane: a ring carried ENTIRELY by one block's code cloud — the
    /// block's linear residual is identically zero — is proposed and ACCEPTED
    /// by the code-space pass, with the bits saving recorded.
    #[test]
    fn zero_residual_ring_block_is_promoted_from_code_space() {
        let n = 512;
        let mut ring = A2::<f64>::zeros((n, 2));
        for i in 0..n {
            let theta = TAU * (i as f64) / (n as f64);
            ring[[i, 0]] = theta.cos();
            ring[[i, 1]] = theta.sin();
        }
        let fit = one_fired_block_fit(&ring, 16, 128);
        let report = harvest_code_space_promotions(&fit, n, 0.05).expect("harvest runs");
        assert_eq!(report.n_blocks_scanned, 128);
        assert_eq!(report.n_communities, 1);
        assert_eq!(
            report.n_accepted, 1,
            "a densely-fired zero-residual ring must be promoted (proposal: {:?})",
            report.proposals.first()
        );
        assert!(
            report.dl_saved_bits > 0.0,
            "the accepted promotion must save bits, got {}",
            report.dl_saved_bits
        );
        assert!((report.fraction_curved - 1.0).abs() < 1.0e-15);
        // L0 is measured off the routing: every row fires 2 scalar coordinates
        // (θ=0's sin is exactly 0.0, so one scalar of one row drops out).
        assert!(
            (report.l0 - 2.0).abs() < 0.01,
            "measured L0 must be ~2.0, got {}",
            report.l0
        );
    }

    /// The flat null: a deterministic 2-D Gaussian code cloud (Box–Muller over
    /// an LCG) has a Rayleigh radial law (κ ≈ 2), so the ring screen refuses and
    /// the pass records a refusal, not a curved atom. `fraction_curved` is the
    /// honesty number: 0 here, 1 above.
    #[test]
    fn gaussian_code_cloud_is_refused_in_code_space() {
        let n = 1024;
        let mut cloud = A2::<f64>::zeros((n, 2));
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next_uniform = move || {
            // xorshift64*: deterministic, no RNG dependency.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let raw = state.wrapping_mul(0x2545F4914F6CDD1D);
            ((raw >> 11) as f64 / (1u64 << 53) as f64).clamp(f64::MIN_POSITIVE, 1.0)
        };
        for i in 0..n {
            let (u1, u2) = (next_uniform(), next_uniform());
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = TAU * u2;
            cloud[[i, 0]] = radius * angle.cos();
            cloud[[i, 1]] = radius * angle.sin();
        }
        let fit = one_fired_block_fit(&cloud, 16, 128);
        let report = harvest_code_space_promotions(&fit, n, 0.05).expect("harvest runs");
        assert_eq!(report.n_communities, 1, "the cloud spans a plane: eligible");
        assert_eq!(
            report.n_accepted, 0,
            "a Gaussian code cloud must be refused (proposal: {:?})",
            report.proposals.first()
        );
        assert_eq!(report.fraction_curved, 0.0);
        let proposal = &report.proposals[0];
        assert!(
            !proposal.verdict.recommend_curl,
            "the ring geometry screen must refuse a Rayleigh radial law (κ={})",
            proposal.verdict.kappa
        );
    }

    /// The cross-block move: a ring the dictionary SHATTERED across two b=1
    /// blocks (atom e0 carries cosθ, atom e1 carries sinθ) presents to each
    /// single-block community as a 1-D line — no plane, no proposal — and only
    /// the co-firing pair census can close it. This is the Gemma Scope shell
    /// shape (#2280/#2502), discovered from the tiered lane's own routing.
    #[test]
    fn shattered_ring_across_two_blocks_is_promoted_from_the_pair_census() {
        let n = 512;
        let p = 16;
        // 256 b=1 blocks (only two ever fire): the same overcompleteness that
        // funds the support dividend in production; see one_fired_block_fit.
        let n_blocks = 256;
        let mut decoder = A2::<f32>::zeros((n_blocks, p));
        decoder[[0, 0]] = 1.0;
        decoder[[1, 1]] = 1.0;
        let mut blocks = A2::<u32>::zeros((n, 2));
        let mut gates = A2::<f32>::zeros((n, 2));
        let mut codes = Array3::<f32>::zeros((n, 2, 1));
        for i in 0..n {
            let theta = TAU * (i as f64) / (n as f64);
            blocks[[i, 0]] = 0;
            blocks[[i, 1]] = 1;
            codes[[i, 0, 0]] = theta.cos() as f32;
            codes[[i, 1, 0]] = theta.sin() as f32;
            gates[[i, 0]] = codes[[i, 0, 0]].abs();
            gates[[i, 1]] = codes[[i, 1, 0]].abs();
        }
        // A gate of exactly zero marks a padded slot, so nudge the four
        // axis-crossing rows off the axis rather than losing their firing.
        for i in 0..n {
            for j in 0..2 {
                if gates[[i, j]] == 0.0 {
                    codes[[i, j, 0]] = 1.0e-6;
                    gates[[i, j]] = 1.0e-6;
                }
            }
        }
        let fit = BlockSparseFit {
            decoder,
            blocks,
            gates,
            codes,
            gamma: 1.0,
            block_utilization: {
                let mut util = vec![0.0f32; n_blocks];
                util[0] = 1.0;
                util[1] = 1.0;
                util
            },
            block_stable_rank: vec![1.0; n_blocks],
            matryoshka_prefix_losses: Vec::new(),
            explained_variance: 1.0,
            epochs: 0,
            convergence: BlockSparseConvergence::trivially_converged(),
            block_topk: 2,
            block_size: 1,
        };
        let report = harvest_code_space_promotions(&fit, n, 0.05).expect("harvest runs");
        // Each b=1 block alone is a line: no single-block proposal possible.
        assert!(
            report.proposals.is_empty(),
            "single-atom communities cannot host a ring"
        );
        assert_eq!(report.pair_proposals.len(), 1, "one co-firing pair");
        let (ga, gb, proposal) = &report.pair_proposals[0];
        assert_eq!((*ga, *gb), (0, 1));
        assert!(
            proposal.accept,
            "the shattered ring must be promoted from the joint cloud: {proposal:?}"
        );
        assert_eq!(report.n_accepted, 1);
        assert!(report.dl_saved_bits > 0.0);
    }

    /// The universal-improver entry: a FOREIGN dictionary (any K×P atom bank)
    /// with solved codes on it is censused pairwise straight from
    /// `SparseAtomCodes` — no tiered fit, no BlockSparseFit. The same shattered
    /// ring, imported as someone else's dictionary, is discovered and promoted.
    #[test]
    fn foreign_dictionary_pair_census_promotes_a_shattered_ring() {
        let n = 512;
        let p = 16;
        let k = 256;
        let mut decoder = A2::<f64>::zeros((k, p));
        decoder[[0, 0]] = 1.0;
        decoder[[1, 1]] = 1.0;
        let mut codes = SparseAtomCodes::empty(n, k);
        for i in 0..n {
            let theta = TAU * (i as f64) / (n as f64);
            let row = codes.row_mut(i);
            row.assign(0, theta.cos());
            row.assign(1, theta.sin());
        }
        let report =
            harvest_code_space_pair_promotions(decoder.view(), &codes, n, 0.05).expect("runs");
        assert!(report.proposals.is_empty());
        assert_eq!(report.pair_proposals.len(), 1);
        let (a, b, proposal) = &report.pair_proposals[0];
        assert_eq!((*a, *b), (0, 1));
        assert!(
            proposal.accept,
            "an imported shattered ring must be promoted: {proposal:?}"
        );
        assert!(report.dl_saved_bits > 0.0);
        assert!((report.l0 - 2.0).abs() < 1.0e-12);
        // Contract errors are loud, not silent.
        let narrow = SparseAtomCodes::empty(n, k - 1);
        assert!(
            harvest_code_space_pair_promotions(decoder.view(), &narrow, n, 0.05).is_err(),
            "a K mismatch must refuse"
        );
    }

    /// The measured distortion floor: nonzero residual reads back as its RMS;
    /// an exactly-reconstructed corpus falls to the f64 resolution floor instead
    /// of a zero that would poison the rate terms.
    #[test]
    fn distortion_floor_is_measured_with_a_resolution_backstop() {
        let residual = ndarray::array![[3.0, 0.0], [0.0, 4.0]];
        // mean square = (9 + 16) / 4 = 6.25 ⇒ RMS 2.5.
        let delta = linear_distortion_floor(residual.view(), 100.0).expect("floor");
        assert!((delta - 2.5).abs() < 1.0e-12, "measured RMS, got {delta}");
        let zero = A2::<f64>::zeros((2, 2));
        let backstop = linear_distortion_floor(zero.view(), 100.0).expect("floor");
        let corpus_rms = (100.0f64 / 4.0).sqrt();
        assert!(
            (backstop - corpus_rms * f64::EPSILON.sqrt()).abs() < 1.0e-18,
            "zero residual must fall to the resolution floor, got {backstop}"
        );
        assert!(linear_distortion_floor(zero.view(), 0.0).is_err());
    }
}
