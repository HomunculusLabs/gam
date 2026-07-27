//! #1026's dominance measurement, re-run on the #2502 corpus.
//!
//! #1026 closed on this comparison and not on the one #2502 has been running.
//! Its result: the hybrid (flat tier + curved tier on the flat tier's residual)
//! beat the external TopK bar at K=32,768 on both seeds, *while the hybrid's own
//! flat tier alone sat BELOW that bar* — so the margin was attributable to the
//! curved tier, not to a stronger flat baseline.
//!
//! That is the measurement this example makes, self-contained: the same
//! `fit_tiered` entry with Tier-2 off and on. Tier-2 off is mu + L (the linear
//! bulk); Tier-2 on is mu + L + C. The delta is what the curved tier adds over
//! spending the corpus on the linear tier alone, at the same Tier-1 geometry.
//!
//! Why this differs from what #2502 measured all session: a pure curved
//! dictionary on raw activations spends its whole active budget reconstructing
//! the dense low-rank bulk (PCA-8 alone carries 0.32 EV on this chart), which
//! the flat baseline reconstructs equally well. Peeling that bulk first is what
//! lets the curved atoms work on the structured residual where curvature is
//! supposed to pay.
//!
//! ```text
//! tiered_dominance <chart.bin> <rows> <cols> <blocks> <block_size> <curved_K> <curved_s>
//! ```

use gam_sae::tiered::{TieredFitConfig, fit_tiered};
use ndarray::Array2;
use std::time::Instant;

fn main() -> Result<(), String> {
    env_logger::try_init().ok();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 8 {
        return Err(
            "usage: tiered_dominance <chart.bin> <rows> <cols> <blocks> <block_size> <curved_K> <curved_s>"
                .into(),
        );
    }
    let rows: usize = args[2].parse().map_err(|e| format!("rows: {e}"))?;
    let cols: usize = args[3].parse().map_err(|e| format!("cols: {e}"))?;
    let blocks: usize = args[4].parse().map_err(|e| format!("blocks: {e}"))?;
    let block_size: usize = args[5].parse().map_err(|e| format!("block_size: {e}"))?;
    let curved_k: usize = args[6].parse().map_err(|e| format!("curved_K: {e}"))?;
    let curved_s: usize = args[7].parse().map_err(|e| format!("curved_s: {e}"))?;

    let bytes = std::fs::read(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!("chart holds {} bytes != rows*cols*8", bytes.len()));
    }
    let data: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect();
    let z = Array2::from_shape_vec((rows, cols), data).map_err(|e| e.to_string())?;
    println!("corpus {rows}x{cols}  tier1 {blocks} blocks of {block_size}  tier2 K={curved_k} s={curved_s}");

    // Arm 1: mu + L. Tier-2 disabled — the linear bulk alone, which is the
    // baseline the curved tier has to beat from its own residual.
    let t0 = Instant::now();
    let flat_config = TieredFitConfig::linear_bulk(blocks, block_size);
    let flat = fit_tiered(z.view(), &flat_config)?;
    println!(
        "FLAT   (mu+L)   EV={:.6}  {:.0}s",
        flat.explained_variance,
        t0.elapsed().as_secs_f64()
    );

    // Arm 2: mu + L + C. Identical Tier-1 geometry, curved refinement on its
    // residual, so the delta isolates the curved tier's contribution.
    let t1 = Instant::now();
    let mut hybrid_config = TieredFitConfig::tiered(blocks, block_size);
    hybrid_config.tier2.n_atoms = curved_k;
    hybrid_config.tier2.support_k = curved_s;
    let hybrid = fit_tiered(z.view(), &hybrid_config)?;
    println!(
        "HYBRID (mu+L+C) EV={:.6}  {:.0}s",
        hybrid.explained_variance,
        t1.elapsed().as_secs_f64()
    );

    if let Some(tier2) = hybrid.tier2.as_ref() {
        println!(
            "  tier2: requested {} atoms, retained {}, outer iters {}, criterion {:.6e}",
            tier2.requested_atoms, tier2.retained_atoms, tier2.outer_iterations, tier2.criterion
        );
    }
    println!(
        "TIER2_ADDS_EV = {:.6}",
        hybrid.explained_variance - flat.explained_variance
    );
    Ok(())
}
