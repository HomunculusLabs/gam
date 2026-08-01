//! The gam manifold-SAE entry for the FVU-at-fixed-bits benchmark
//! (block-crosscoder-experiment protocol shape): fit the tiered engine's
//! block-sparse dictionary per site on RAW GPT-2 activations, run the
//! code-space census over the fitted blocks, and evaluate held-out FVU under
//! fixed-width bit accounting — with census-accepted curved blocks decoded
//! through their chart (mean train radius × (cos θ, sin θ) in the block frame,
//! one 7-bit θ per firing) instead of two 7-bit amplitudes.
//!
//! Usage: manifold_fvu_bench <dump_dir> <layer>
//! Reads  raw_l<L>_train.f32 / raw_l<L>_eval.f32 (rows×768 LE f32) + raw_meta.json.
//! Prints JSON lines: per-topk (bits/token/site, FVU) for the linear tier and
//! for the chart-decoded tier, plus the census summary.

use std::fs;
use std::path::Path;

use gam_sae::sparse_dict::{
    BlockSeedPolicy, BlockSparseConfig, block_sparse_dictionary_transform,
    fit_block_sparse_dictionary_with_seed,
};
use gam_sae::tiered::{Tier0Mean, harvest_code_space_promotions, linear_distortion_floor};
use ndarray::{Array2, Axis};

const P: usize = 768;
const N_BLOCKS: usize = 4096;
const BLOCK_SIZE: usize = 2;
const AMP_BITS: u32 = 7;

fn read_f32_matrix(path: &Path, p: usize) -> Result<Array2<f64>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{path:?}: {e}"))?;
    if bytes.len() % (4 * p) != 0 {
        return Err(format!("{path:?}: byte length {} not divisible by 4·{p}", bytes.len()));
    }
    let n = bytes.len() / (4 * p);
    let mut out = Array2::<f64>::zeros((n, p));
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[[i / p, i % p]] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
    }
    Ok(out)
}

fn quantize(v: f64, vmax: f64, bits: u32) -> f64 {
    let levels = ((1u32 << bits) - 1) as f64;
    let clamped = (v / vmax).clamp(-1.0, 1.0);
    (clamped * levels / 2.0).round() / (levels / 2.0) * vmax
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let layer: usize = args
        .get(2)
        .ok_or("usage: manifold_fvu_bench <dir> <layer>")?
        .parse()
        .map_err(|e| format!("layer: {e}"))?;

    let train = read_f32_matrix(&dir.join(format!("raw_l{layer}_train.f32")), P)?;
    let eval = read_f32_matrix(&dir.join(format!("raw_l{layer}_eval.f32")), P)?;
    eprintln!("layer {layer}: train {:?} eval {:?}", train.dim(), eval.dim());

    // Tier-0: train-split mean, held fixed for eval (the honest baseline).
    let tier0 = Tier0Mean::fit(train.view())?;
    let train_c = tier0.apply(train.view())?;
    let eval_c = tier0.apply(eval.view())?;
    let train_f32 = train_c.mapv(|v| v as f32);

    // Tier-1: the engine's block-sparse dictionary (G=4096 blocks of b=2).
    let mut config = BlockSparseConfig::new(N_BLOCKS, BLOCK_SIZE);
    config.block_topk = 4;
    config.aux_k = 4;
    let fit = fit_block_sparse_dictionary_with_seed(
        train_f32.view(),
        &config,
        BlockSeedPolicy::CoordinatePartition,
    )?;
    eprintln!(
        "fit done: ev={:.4} epochs={} certified={}",
        fit.explained_variance, fit.epochs, fit.convergence.certified
    );

    // Code-space census over the fitted blocks (train codes).
    let train_recon = fit.reconstruct();
    let residual = &train_c - &train_recon.mapv(|v| v as f64);
    let baseline_energy: f64 = train_c.iter().map(|v| v * v).sum();
    let delta = linear_distortion_floor(residual.view(), baseline_energy)?;
    let census = harvest_code_space_promotions(&fit, train_c.nrows(), delta)?;
    // Per accepted single-block promotion: the chart's stored radius = mean
    // train radius of that block's code cloud, and its own frame IS the chart
    // frame (b=2). Collect radii for chart decoding.
    let mut chart_radius = vec![f64::NAN; N_BLOCKS];
    let mut n_chart = 0usize;
    for proposal in &census.proposals {
        if proposal.accept {
            chart_radius[proposal.block] = proposal.verdict.radius;
            n_chart += 1;
        }
    }
    println!(
        "{{\"layer\":{layer},\"census\":{{\"communities\":{},\"accepted\":{},\"pair_accepted\":{},\"dl_saved_bits\":{:.1},\"delta\":{:.5},\"fit_ev\":{:.4}}}}}",
        census.n_communities,
        census.n_accepted,
        census.pair_proposals.iter().filter(|v| v.proposal.accept).count(),
        census.dl_saved_bits,
        delta,
        fit.explained_variance
    );

    // Amplitude scale for the fixed-width quantizer, from TRAIN codes.
    let mut cmax = 0.0f64;
    for &c in fit.codes.iter() {
        cmax = cmax.max((c as f64).abs());
    }

    let sel_bits = (N_BLOCKS as f64).log2().ceil() as u32;
    let eval_f32 = eval_c.mapv(|v| v as f32);
    let n_eval = eval_c.nrows();
    let tss: f64 = {
        let em = eval_c.mean_axis(Axis(0)).ok_or("eval mean")?;
        eval_c
            .rows()
            .into_iter()
            .map(|r| {
                r.iter()
                    .zip(em.iter())
                    .map(|(v, m)| (v - m) * (v - m))
                    .sum::<f64>()
            })
            .sum()
    };

    for topk in [1usize, 2, 3, 4] {
        let (blocks, _gates, codes) = block_sparse_dictionary_transform(
            eval_f32.view(),
            fit.decoder.view(),
            fit.gamma,
            BLOCK_SIZE,
            topk,
            config.block_tile,
        )?;
        // Arm L: linear fixed-width — every slot pays sel + b·AMP bits.
        // Arm M: manifold — census-accepted blocks decode r̄·(cosθ, sinθ) with
        // ONE 7-bit θ; everything else decodes as arm L.
        let mut rss_lin = 0.0f64;
        let mut rss_man = 0.0f64;
        let mut bits_lin = 0u64;
        let mut bits_man = 0u64;
        let mut recon_l = Array2::<f64>::zeros((n_eval, P));
        let mut recon_m = Array2::<f64>::zeros((n_eval, P));
        for i in 0..n_eval {
            for j in 0..topk {
                let g = blocks[[i, j]] as usize;
                let c0 = codes[[i, j, 0]] as f64;
                let c1 = codes[[i, j, 1]] as f64;
                if c0 == 0.0 && c1 == 0.0 {
                    continue;
                }
                let q0 = quantize(c0, cmax, AMP_BITS);
                let q1 = quantize(c1, cmax, AMP_BITS);
                let row0 = fit.decoder.row(g * BLOCK_SIZE);
                let row1 = fit.decoder.row(g * BLOCK_SIZE + 1);
                for cix in 0..P {
                    recon_l[[i, cix]] += q0 * row0[cix] as f64 + q1 * row1[cix] as f64;
                }
                bits_lin += (sel_bits + 2 * AMP_BITS) as u64;
                if chart_radius[g].is_finite() {
                    // chart decode: θ quantized, radius stored in the model
                    let theta = c1.atan2(c0);
                    let levels = ((1u32 << AMP_BITS) - 1) as f64;
                    let tq = (theta / std::f64::consts::TAU * levels).round() / levels
                        * std::f64::consts::TAU;
                    let r = chart_radius[g];
                    let (m0, m1) = (r * tq.cos(), r * tq.sin());
                    for cix in 0..P {
                        recon_m[[i, cix]] += m0 * row0[cix] as f64 + m1 * row1[cix] as f64;
                    }
                    bits_man += (sel_bits + AMP_BITS) as u64;
                } else {
                    for cix in 0..P {
                        recon_m[[i, cix]] += q0 * row0[cix] as f64 + q1 * row1[cix] as f64;
                    }
                    bits_man += (sel_bits + 2 * AMP_BITS) as u64;
                }
            }
        }
        for i in 0..n_eval {
            for cix in 0..P {
                let t = eval_c[[i, cix]];
                let dl = t - recon_l[[i, cix]];
                let dm = t - recon_m[[i, cix]];
                rss_lin += dl * dl;
                rss_man += dm * dm;
            }
        }
        println!(
            "{{\"layer\":{layer},\"topk\":{topk},\"bits_lin\":{:.2},\"fvu_lin\":{:.6},\"bits_man\":{:.2},\"fvu_man\":{:.6},\"chart_blocks\":{n_chart}}}",
            bits_lin as f64 / n_eval as f64,
            rss_lin / tss,
            bits_man as f64 / n_eval as f64,
            rss_man / tss
        );
    }
    Ok(())
}
