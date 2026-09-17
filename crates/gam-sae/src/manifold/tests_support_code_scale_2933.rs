//! #2933 F43: the native KT/Chow–Liu support code at the advertised dictionary
//! scale. With `G = 32768` atoms and one to three active atoms per token the
//! code's sufficient statistics are a handful of firing counts and co-firing
//! pairs, yet `SparseAtomCodes::support_entropy` used to allocate a dense
//! `G × G` f64 co-occurrence array (8 GiB at this `G`) and run an `O(G²)` Prim,
//! so the description-length report was unavailable at exactly the scale a
//! support-sparse fit reaches. Declared as a sibling `#[cfg(test)] mod` in
//! `mod.rs` so it can read the lib-test binary's allocation ledger, which lives
//! in `tests_row_jet_and_outer_objective_780`.

use super::tests_row_jet_and_outer_objective_780::{
    begin_row_jet_allocation_measurement, end_row_jet_allocation_measurement,
};
use crate::atom_codes::SparseAtomCodes;
use crate::description_length::manifold_fit_description_length;
use std::collections::BTreeSet;

/// Sequential Krichevsky–Trofimov code length of a binary sequence, replaying
/// the decoder's predictive probabilities in row order.
fn sequential_kt_bits(values: impl IntoIterator<Item = bool>) -> f64 {
    let mut counts = [0.5_f64, 0.5_f64];
    let mut bits = 0.0;
    for value in values {
        let index = usize::from(value);
        bits -= (counts[index] / (counts[0] + counts[1])).log2();
        counts[index] += 1.0;
    }
    bits
}

#[test]
fn native_support_code_report_allocates_linearly_at_32768_atoms_2933_f43() {
    let k_atoms = 1_usize << 15;
    let n_rows = 48_usize;
    let mut codes = SparseAtomCodes::empty(n_rows, k_atoms);
    for row in 0..n_rows {
        // Hub atoms shared across rows give genuine co-firing edges; a
        // row-specific tail atom spreads the supports across the dictionary.
        codes.row_mut(row).assign(row % 4, 1.0);
        if row % 3 != 0 {
            codes
                .row_mut(row)
                .assign(k_atoms - 1 - (row * 977) % (k_atoms / 2), 0.5);
        }
        if row % 2 == 0 {
            codes.row_mut(row).assign(4 + row % 5, 0.25);
        }
    }
    let mut active_entries = 0_usize;
    let mut co_firing_pairs = BTreeSet::new();
    for code in codes.iter() {
        let support: Vec<usize> = code.active_mask.iter_ones().collect();
        active_entries += support.len();
        for (index, &u) in support.iter().enumerate() {
            for &v in &support[index + 1..] {
                co_firing_pairs.insert((u, v));
            }
        }
    }
    let atom_coord_dims = vec![1.0_f64; k_atoms];
    let coord_variances = [1.0_f64, 0.5];

    // The support coder is serial, so the calling thread's ledger sees every
    // allocation it makes. The ledger sums allocated bytes without subtracting
    // frees, which bounds the peak from above.
    begin_row_jet_allocation_measurement();
    let report = manifold_fit_description_length(
        &codes,
        &coord_variances,
        0.3,
        &atom_coord_dims,
        0.9,
        0,
        None,
    );
    let (allocation_calls, allocated_bytes) = end_row_jet_allocation_measurement();

    // Every array the report keeps is indexed by an atom, a row, an active
    // entry or a co-firing pair, and together, Vec growth included, they hold
    // fewer than 64 machine words per such unit: 512 bytes per unit.
    let linear_units = k_atoms + n_rows + active_entries + co_firing_pairs.len();
    let linear_budget = 512 * linear_units as u64;
    let dense_square_bytes = 8 * k_atoms as u64 * k_atoms as u64;
    assert!(
        allocated_bytes <= linear_budget,
        "the native support-code report allocated {allocated_bytes} bytes in {allocation_calls} \
         calls at G = {k_atoms}, N = {n_rows}, {active_entries} active entries, {} co-firing \
         pairs; the linear budget is {linear_budget} bytes, and a dense G x G f64 buffer alone \
         is {dense_square_bytes} bytes",
        co_firing_pairs.len()
    );

    // The report at scale is the same code, not a placeholder: its selection
    // price is the support code's tree length, the independent reference agrees
    // with replaying the KT predictor over every atom's column, and the tree
    // length carries the Cayley charge for naming a labelled tree on G atoms.
    let support = codes.support_entropy();
    assert_eq!(
        report.selection_bits_per_token.to_bits(),
        support.tree_bits.to_bits()
    );
    let independent_reference = (0..k_atoms)
        .map(|atom| sequential_kt_bits(codes.iter().map(|code| code.active_mask.get(atom))))
        .sum::<f64>()
        / n_rows as f64;
    assert!(
        (support.independent_bits - independent_reference).abs() <= 1e-9 * independent_reference,
        "independent KT bits {} != replayed reference {independent_reference}",
        support.independent_bits
    );
    let cayley_bits = (k_atoms as f64 - 2.0) * (k_atoms as f64).log2() / n_rows as f64;
    assert!(
        support.tree_bits.is_finite() && support.tree_bits > cayley_bits,
        "tree bits {} must be finite and exceed the Cayley charge {cayley_bits}",
        support.tree_bits
    );
}
