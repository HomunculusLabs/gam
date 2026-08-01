//! Run the atom-level code-space census over an EXTERNAL dictionary dump:
//! `meta.json` + `decoder.f32` (K·P LE f32) + `codes.bin` (12-byte records:
//! u32 row, u32 atom, f32 weight, LE) — the format `dump_codes.py` writes from
//! a public SAE on real activations. Prints the census report and the accepted
//! pair promotions as JSON lines.
//!
//! Usage: `code_space_census <dump_dir> [max_rows]`

use std::fs;
use std::path::Path;

use gam_sae::atom_codes::SparseAtomCodes;
use gam_sae::tiered::{fit_pair_chart, harvest_code_space_pair_promotions};
use ndarray::Array2;

fn json_field(meta: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let start = meta.find(&pat)? + pat.len();
    let rest = meta[start..].trim_start();
    let end = rest
        .find(|c: char| c == ',' || c == '}')
        .unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("out"));
    let max_rows: usize = args
        .get(2)
        .map(|s| s.parse().map_err(|e| format!("bad max_rows: {e}")))
        .transpose()?
        .unwrap_or(usize::MAX);

    let meta = fs::read_to_string(dir.join("meta.json"))
        .map_err(|e| format!("meta.json: {e}"))?;
    let n_rows_all = json_field(&meta, "n_rows").ok_or("meta: n_rows")? as usize;
    let k_atoms = json_field(&meta, "k_atoms").ok_or("meta: k_atoms")? as usize;
    let p = json_field(&meta, "p").ok_or("meta: p")? as usize;
    let delta_rms = json_field(&meta, "delta_rms").ok_or("meta: delta_rms")?;
    let n_rows = n_rows_all.min(max_rows);
    eprintln!("census: N={n_rows} (of {n_rows_all}) K={k_atoms} P={p} delta={delta_rms}");

    let dec_bytes = fs::read(dir.join("decoder.f32")).map_err(|e| format!("decoder.f32: {e}"))?;
    if dec_bytes.len() != k_atoms * p * 4 {
        return Err(format!(
            "decoder.f32 has {} bytes, expected {}",
            dec_bytes.len(),
            k_atoms * p * 4
        ));
    }
    let mut decoder = Array2::<f64>::zeros((k_atoms, p));
    for (i, chunk) in dec_bytes.chunks_exact(4).enumerate() {
        decoder[[i / p, i % p]] =
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
    }

    let code_bytes = fs::read(dir.join("codes.bin")).map_err(|e| format!("codes.bin: {e}"))?;
    if code_bytes.len() % 12 != 0 {
        return Err(format!("codes.bin length {} not /12", code_bytes.len()));
    }
    let mut codes = SparseAtomCodes::empty(n_rows, k_atoms);
    let mut kept = 0u64;
    for rec in code_bytes.chunks_exact(12) {
        let row = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as usize;
        let atom = u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]) as usize;
        let w = f32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]) as f64;
        if row < n_rows && atom < k_atoms && w != 0.0 {
            codes.row_mut(row).assign(atom, w);
            kept += 1;
        }
    }
    eprintln!("census: kept {kept} nonzero code entries");

    let report = harvest_code_space_pair_promotions(decoder.view(), &codes, n_rows, delta_rms)?;
    println!(
        "{{\"n_atoms\":{},\"n_communities\":{},\"n_accepted\":{},\"fraction_curved\":{:.6},\"dl_saved_bits\":{:.1},\"l0\":{:.3},\"tolerance\":{:.6}}}",
        report.n_blocks_scanned,
        report.n_communities,
        report.n_accepted,
        report.fraction_curved,
        report.dl_saved_bits,
        report.l0,
        report.tolerance
    );
    let mut accepted: Vec<_> = report
        .pair_proposals
        .iter()
        .filter(|v| v.proposal.accept)
        .collect();
    accepted.sort_by(|x, y| {
        (y.proposal.dl_old - y.proposal.dl_new).total_cmp(&(x.proposal.dl_old - x.proposal.dl_new))
    });
    for v in accepted.iter().take(200) {
        let pr = &v.proposal;
        println!(
            "{{\"pair\":[{},{}],\"bits_saved\":{:.1},\"radius\":{:.4},\"kappa\":{:.3},\"span\":{:.3},\"firings\":{:.0},\"prescreen\":{:.1},\"null_p_hat\":{:.4},\"null_exceedances\":{},\"topology\":{},\"topology_dim\":{},\"topology_err\":{}}}",
            v.atom_a,
            v.atom_b,
            pr.dl_old - pr.dl_new,
            pr.verdict.radius,
            pr.verdict.kappa,
            pr.span,
            pr.firing_rate * n_rows as f64,
            pr.crossover_prescreen_bits,
            v.null_p_hat,
            v.null_exceedances,
            v.topology_kind
                .as_deref()
                .map(|k| format!("\"{k}\""))
                .unwrap_or_else(|| "null".to_string()),
            v.topology_dim
                .map(|d| d.to_string())
                .unwrap_or_else(|| "null".to_string()),
            v.topology_error
                .as_deref()
                .map(|e| format!("{:?}", e))
                .unwrap_or_else(|| "null".to_string())
        );
    }
    // Full REML chart fits on the top accepted pairs: rebuild each pair's
    // joint cloud from the codes and run the grouped-LAML outer engine.
    for v in accepted.iter().take(12) {
        let mut cloud_rows: Vec<[f64; 2]> = Vec::new();
        for row in codes.iter() {
            if row.active_mask.get(v.atom_a) && row.active_mask.get(v.atom_b) {
                cloud_rows.push([row.weights[v.atom_a], row.weights[v.atom_b]]);
            }
        }
        let f = cloud_rows.len();
        let mut cloud = Array2::<f64>::zeros((f, 2));
        for (i, r) in cloud_rows.iter().enumerate() {
            cloud[[i, 0]] = r[0];
            cloud[[i, 1]] = r[1];
        }
        match fit_pair_chart(cloud.view(), 0xC0FF_EE00_D15E_A5E5) {
            Ok(chart) => println!(
                "{{\"chart_pair\":[{},{}],\"lambda\":{:?},\"chart_ev\":{:.4},\"certified\":{},\"recurred\":{},\"outer_iters\":{}}}",
                v.atom_a,
                v.atom_b,
                chart.lambda_smooth,
                chart.explained_variance,
                chart.certified,
                chart.recurred,
                chart.outer_iterations
            ),
            Err(error) => println!(
                "{{\"chart_pair\":[{},{}],\"chart_error\":{error:?}}}",
                v.atom_a, v.atom_b
            ),
        }
    }
    Ok(())
}
