//! Support-code, rare-atom and spectrum-sampling contracts of the Eq. 4 scorer
//! (#2933 F09, F16, F19). The support oracles are exact integer combinatorics, the
//! Kraft sum and explicit Krichevsky–Trofimov products, never the scorer's own
//! helpers; the spectrum oracles are row
//! permutations and closed-form raw second moments of the contributions.

use super::*;

/// `C(g, k)` in exact integer arithmetic (each partial product is itself a
/// binomial coefficient, so every division is exact).
fn binomial(g: u64, k: u64) -> u64 {
    (0..k).fold(1_u64, |acc, index| acc * (g - index) / (index + 1))
}

/// Score a gate whose atoms transmit no coordinates (`code_dims = 0`) over a
/// perfectly reconstructed, positively varying one-column `test_x`, so only the
/// support terms carry information.
fn support_only(gate: &Array2<f64>) -> Eq4DescriptionLength {
    let rows = gate.nrows();
    let test_x = Array2::from_shape_fn((rows, 1), |(row, _)| row as f64);
    eq4_fixed_distortion_description_length(
        test_x.view(),
        test_x.view(),
        gate.view(),
        &vec![0_i64; gate.ncols()],
        0,
        2,
        &[0.5],
        None,
        |_, take| Ok(Array2::zeros((take.len(), 1))),
    )
    .expect("support-only Eq. 4 fixture must score")
}

/// Every support over `G ≤ 4` atoms is scored as a constant-support source. The
/// charged per-token lengths must form a COMPLETE prefix code: `Σ_S 2^{−L(S)} = 1`
/// over all `2^G` supports, each `L(S) = log₂((G+1)·C(G, |S|))`. The rounded-mean
/// price `log₂ C(G, |S|)` gives Kraft sum `G+1` here, so it names no decodable
/// support at all.
#[test]
fn eq4_support_code_is_kraft_complete_over_every_support() {
    const ROWS: usize = 3;
    for atoms in 1_usize..=4 {
        let mut kraft = 0.0_f64;
        for mask in 0_u32..(1_u32 << atoms) {
            let gate = Array2::from_shape_fn((ROWS, atoms), |(_, atom)| {
                if (mask >> atom) & 1 == 1 { 1.0 } else { 0.0 }
            });
            let report = support_only(&gate);
            let cardinality = mask.count_ones() as u64;
            let expected = ((atoms as u64 + 1) * binomial(atoms as u64, cardinality)) as f64;
            let expected = expected.log2();
            assert!(
                (report.support_bits - expected).abs() <= 1.0e-12,
                "G={atoms} support {mask:#b}: charged {} bits, code length {expected}",
                report.support_bits
            );
            kraft += 2.0_f64.powf(-report.support_bits);
        }
        assert!(
            (kraft - 1.0).abs() <= 1.0e-12,
            "G={atoms}: Kraft sum of the charged support code is {kraft}, not 1"
        );
    }
}

/// The audit counterexample: one atom on half the rows is a one-bit-per-token
/// source. The rounded mean `round(0.5) = 0` priced it at `log₂ C(1, 0) = 0`. The
/// complete code charges one bit per token for the on/off pattern, and the same
/// bit for an always-on or always-off atom, whose cardinality it still transmits.
#[test]
fn eq4_support_charges_one_bit_per_token_for_a_one_atom_source() {
    let alternating = Array2::from_shape_fn((8, 1), |(row, _)| (row % 2) as f64);
    assert_eq!(support_only(&alternating).support_bits, 1.0);
    for state in [0.0, 1.0] {
        let constant = Array2::from_elem((8, 1), state);
        assert_eq!(support_only(&constant).support_bits, 1.0);
    }
}

/// Mixed cardinalities are priced row by row, never at the rounded mean, and a
/// genuine fixed-TopK source pays its subset price plus the constant-cardinality
/// header `log₂(G+1)`.
#[test]
fn eq4_support_prices_each_row_cardinality_not_the_rounded_mean() {
    // G = 6, cardinalities [0, 1, 1, 2, 5, 6, 3, 3]: mean 21/8 = 2.625 rounds to 3.
    let supports: [&[usize]; 8] = [
        &[],
        &[4],
        &[0],
        &[1, 5],
        &[0, 1, 2, 3, 5],
        &[0, 1, 2, 3, 4, 5],
        &[2, 3, 4],
        &[0, 3, 5],
    ];
    let gate = Array2::from_shape_fn((supports.len(), 6), |(row, atom)| {
        if supports[row].contains(&atom) { 1.0 } else { 0.0 }
    });
    let report = support_only(&gate);
    let subset_bits: f64 = supports
        .iter()
        .map(|support| (binomial(6, support.len() as u64) as f64).log2())
        .sum();
    let expected = 7.0_f64.log2() + subset_bits / supports.len() as f64;
    assert_eq!(report.achieved_block_l0, 2.625);
    assert!(
        (report.support_bits - expected).abs() <= 1.0e-12,
        "mixed cardinalities: charged {}, row-by-row code {expected}",
        report.support_bits
    );
    let rounded_mean_price = (binomial(6, 3) as f64).log2();
    assert!((report.support_bits - rounded_mean_price).abs() > 0.5);

    // Fixed TopK: G = 8, k = 3 on every row, five different subsets.
    let topk: [[usize; 3]; 5] = [[0, 1, 2], [5, 6, 7], [1, 3, 5], [0, 4, 7], [2, 3, 6]];
    let gate = Array2::from_shape_fn((topk.len(), 8), |(row, atom)| {
        if topk[row].contains(&atom) { 1.0 } else { 0.0 }
    });
    let expected = 9.0_f64.log2() + (binomial(8, 3) as f64).log2();
    assert!((support_only(&gate).support_bits - expected).abs() <= 1.0e-12);
}

/// The reported independent support code is the native Krichevsky–Trofimov code,
/// not plug-in Bernoulli entropy. Over `n` rows, an atom firing `m` times has KT
/// probability `Π_{j<m}(j+½)·Π_{j<n−m}(j+½) / n!`, evaluated here as a product
/// rather than through `lnΓ`. A dead atom is therefore NOT free (the plug-in says
/// zero), and a rare atom costs more than its plug-in entropy.
#[test]
fn eq4_independent_support_is_the_kt_code_not_plug_in_entropy() {
    const ROWS: usize = 8;
    // Atom 0 fires once, atom 1 never fires.
    let gate = Array2::from_shape_fn((ROWS, 2), |(row, atom)| {
        if atom == 0 && row == 5 { 1.0 } else { 0.0 }
    });
    let kt_bits = |fired: usize| -> f64 {
        let half_counts = |count: usize| (0..count).map(|j| (j as f64 + 0.5).log2()).sum::<f64>();
        let log2_factorial: f64 = (1..=ROWS).map(|t| (t as f64).log2()).sum();
        log2_factorial - half_counts(fired) - half_counts(ROWS - fired)
    };
    let expected = (kt_bits(1) + kt_bits(0)) / ROWS as f64;
    let report = support_only(&gate);
    assert!(
        (report.independent_support_bits - expected).abs() <= 1.0e-10 * expected,
        "KT support {} != closed form {expected}",
        report.independent_support_bits
    );
    let p = 1.0 / ROWS as f64;
    let plug_in = -(p * p.log2() + (1.0 - p) * (1.0 - p).log2());
    // The dead atom alone costs 2.35 bits over eight rows.
    assert!(kt_bits(0) > 2.3);
    assert!(report.independent_support_bits > plug_in + 0.25);
}

/// Rare atoms are priced from the firings they have (#2933 F16). Atom 0 fires on
/// the first `m ∈ {1,…,6}` of 16 rows with contributions `a·(1, 2, …, m)`. That
/// includes the audit's three firings `(1, 2, 3)`, which have sample variance one
/// and scored free. Atom 1 fires on every row with an alternating `±1`
/// background. Reconstruction is exact. The retired branch gave every atom
/// firing fewer than `max(code_dim+1, 4) = 4` times an all-zero spectrum.
///
/// Oracle, independent of the scorer: no mean is transmitted, so each atom is
/// priced at its raw per-firing second moment, `λ_0 = Σc²/m` and `λ_1 = 1`. With
/// both above the water level, the level is `θ = D / (1 + p_0)` with
/// `D = (1−R²)·v_x`, and the code bits are `p_0·½log₂(λ_0/θ) + ½log₂(λ_1/θ)`.
/// That is one continuous expression in the firings on both sides of the old
/// threshold, at two amplitudes, with no free zero and no jump at `m = 4`.
#[test]
fn eq4_rare_atoms_are_priced_from_their_few_firings() {
    const ROWS: usize = 16;
    const TARGET: f64 = 0.99;
    for firings in 1_usize..=6 {
        for amplitude in [1.0_f64, 4.0] {
            let rare: Vec<f64> = (0..firings).map(|j| amplitude * (j as f64 + 1.0)).collect();
            let background: Vec<f64> = (0..ROWS).map(|row| if row % 2 == 0 { 1.0 } else { -1.0 }).collect();
            let contribution = |atom: usize, row: usize| -> f64 {
                match atom {
                    0 if row < firings => rare[row],
                    0 => 0.0,
                    _ => background[row],
                }
            };
            let test_x = Array2::from_shape_fn((ROWS, 1), |(row, _)| {
                contribution(0, row) + contribution(1, row)
            });
            let gate = Array2::from_shape_fn((ROWS, 2), |(row, atom)| {
                if atom == 1 || row < firings { 1.0 } else { 0.0 }
            });
            let report = eq4_fixed_distortion_description_length(
                test_x.view(),
                test_x.view(),
                gate.view(),
                &[1, 1],
                0,
                2,
                &[TARGET],
                None,
                |atom, take| {
                    Ok(Array2::from_shape_fn((take.len(), 1), |(out_row, _)| {
                        contribution(atom, take[out_row])
                    }))
                },
            )
            .expect("rare-atom Eq. 4 fixture must score");

            let mean_x = test_x.iter().sum::<f64>() / ROWS as f64;
            let variance_x =
                test_x.iter().map(|&value| (value - mean_x).powi(2)).sum::<f64>() / ROWS as f64;
            let p_rare = firings as f64 / ROWS as f64;
            let theta = (1.0 - TARGET) * variance_x / (1.0 + p_rare);
            let lambda_rare = rare.iter().map(|value| value * value).sum::<f64>() / firings as f64;
            let lambda_background = 1.0;
            assert!(lambda_rare > theta && lambda_background > theta);
            let expected = p_rare * 0.5 * (lambda_rare / theta).log2()
                + 0.5 * (lambda_background / theta).log2();
            let code_bits = report.per_target[0].code_bits;
            assert!(
                (code_bits - expected).abs() <= 1.0e-10 * expected,
                "{firings} firings at amplitude {amplitude}: code bits {code_bits}, the raw \
                 per-firing moment costs {expected}"
            );
        }
    }
}

/// Every contribution varies around mean zero: rows `≡ 0 (mod stride)`
/// contribute zero and the remaining rows alternate `−1, +1`. The row count
/// `4096·stride` made the retired sampler keep exactly the zero rows, erasing the
/// whole spectrum.
fn aliased_contribution(stride: usize) -> Vec<f64> {
    let rows = 4096 * stride;
    let mut sign = -1.0;
    (0..rows)
        .map(|row| {
            if row % stride == 0 {
                0.0
            } else {
                let value = sign;
                sign = -sign;
                value
            }
        })
        .collect()
}

/// Score a one-atom, always-firing featurizer that reconstructs `test_x` exactly,
/// with contribution columns `values` and `2·values` (a rank-one `(N, 2)` matrix).
fn score_single_atom(values: &[f64], code_dim: i64) -> Eq4DescriptionLength {
    let rows = values.len();
    let contribution = Array2::from_shape_fn((rows, 2), |(row, column)| {
        values[row] * (column + 1) as f64
    });
    let gate = Array2::ones((rows, 1));
    eq4_fixed_distortion_description_length(
        contribution.view(),
        contribution.view(),
        gate.view(),
        &[code_dim],
        0,
        2,
        &[0.9, 0.5],
        None,
        |_, take| {
            let mut selected = Array2::zeros((take.len(), 2));
            for (out_row, &source_row) in take.iter().enumerate() {
                selected.row_mut(out_row).assign(&contribution.row(source_row));
            }
            Ok(selected)
        },
    )
    .expect("single-atom Eq. 4 fixture must score")
}

/// Row-order invariance at every stride the retired `ceil(n/4096)` sampler could
/// take (2, 3, 4), for the flat fast path (`code_dim = 1`) and the SVD path
/// (`code_dim = 2`). Four orderings of the same rows: the stride-periodic one,
/// its reverse, the affine permutation `row ↦ (5·row + 7) mod N`, and all zero
/// rows interleaved first.
///
/// Magnitude oracle: the contribution has mean zero, perfect reconstruction, and
/// output variance `v = mean(c²)·5`. The fixed distortion is `D = (1−R²)·v`, and
/// the atom's raw per-firing moment over ALL firings is `λ = mean(c²)·5`, so its
/// code bits are exactly `½·log₂(λ/D) = ½·log₂(1/(1−R²))`. The aliased sampler
/// scored zero bits.
#[test]
fn eq4_atom_spectrum_is_row_order_invariant_at_every_stride() {
    for stride in [2_usize, 3, 4] {
        let values = aliased_contribution(stride);
        let rows = values.len();
        assert!(values.iter().sum::<f64>() == 0.0);
        let orderings: [Vec<f64>; 4] = [
            values.clone(),
            values.iter().rev().copied().collect(),
            (0..rows).map(|row| values[(5 * row + 7) % rows]).collect(),
            {
                let mut interleaved: Vec<f64> =
                    values.iter().copied().filter(|&value| value == 0.0).collect();
                interleaved.extend(values.iter().copied().filter(|&value| value != 0.0));
                interleaved
            },
        ];
        for code_dim in [1_i64, 2] {
            let reports: Vec<Eq4DescriptionLength> = orderings
                .iter()
                .map(|ordering| score_single_atom(ordering, code_dim))
                .collect();
            for (target_index, &target) in [0.9_f64, 0.5].iter().enumerate() {
                let expected = 0.5 * (1.0 / (1.0 - target)).log2();
                let reference = reports[0].per_target[target_index];
                for (ordering_index, report) in reports.iter().enumerate() {
                    let row = report.per_target[target_index];
                    assert!(
                        (row.code_bits - expected).abs() <= 1.0e-9 * expected,
                        "stride {stride}, code_dim {code_dim}, ordering {ordering_index}, \
                         R²={target}: code bits {}, the full firing sample costs {expected}",
                        row.code_bits
                    );
                    assert!(
                        (row.bits - reference.bits).abs() <= 1.0e-9 * (1.0 + reference.bits.abs()),
                        "stride {stride}, code_dim {code_dim}, R²={target}: ordering \
                         {ordering_index} scores {} bits, ordering 0 scores {}",
                        row.bits,
                        reference.bits
                    );
                }
            }
        }
    }
}
