//! The invariant sorted-profile tie-break shared by both farthest-point center
//! selectors (#2420).
//!
//! Both [`select_thin_plate_knots`](super::duchon_thinplate::select_thin_plate_knots)
//! and [`select_spherical_farthest_point_centers`](super::workspace_cache::select_spherical_farthest_point_centers)
//! are greedy maximin recursions whose per-step key is a lexicographic composite:
//! two `O(1)` scalars (a fill-distance extremum and a centrality key) refined, on
//! an exact tie, by the *sorted pairwise profile* of a row — the multiset
//! `{κ(i, l) : l = 1..n}` of that row's pairwise kernel against the whole
//! support, in ascending total order.
//!
//! The profile is the only key that is simultaneously
//!
//! * **isometry invariant** — it is a function of the unordered pairwise
//!   geometry alone, so a rigid motion of the cloud leaves it fixed (a
//!   rotation-invariant kernel `k(r)` demands a knot SET that rotates rigidly
//!   with the data), and
//! * **row-permutation invariant** — it is a sorted multiset, so it cannot see
//!   the order the rows arrived in.
//!
//! It is also the only key that costs `O(n log n)` to evaluate, and that is what
//! this module exists to charge correctly.
//!
//! # Charge the key only where it can decide something
//!
//! Written as a two-argument comparator `cmp(i, j)` driving a running-incumbent
//! scan, the profile key builds **two** profiles per comparison and is quadratic
//! in the tie class:
//!
//! ```text
//! pivot = candidates.reduce(|a, b| if cmp(a, b).is_lt() { a } else { b });   // 2·(|C| − 1)
//! candidates.retain(|i| cmp(i, pivot).is_eq());                              // 2·|C|
//! ```
//!
//! On a candidate set of size one — the overwhelmingly common case, since a
//! cloud with no exact symmetry has a unique maximin winner at every step — the
//! `reduce` compares nothing and the `retain` still spends two `O(n log n)`
//! sorts establishing that a row equals itself.
//!
//! [`resolve_sorted_profile_tie`] states the same total preorder in
//! extremum-then-refine form instead. Lexicographic minimization is associative,
//! so minimizing `(cheap…, profile)` in one pass is the same as minimizing the
//! cheap prefix and then minimizing `profile` over the rows that attain it — and
//! in that form a lone candidate is *already* the extremum of any refinement, so
//! it needs no profile at all. A tie class of size `|C|` builds exactly `|C|`
//! profiles, each used for both the choice and the class filter, where the
//! comparator form built `4·|C| − 2`.
//!
//! Every row the resolver returns has a profile bit-identical to every other, so
//! which one a caller treats as the representative cannot change any subsequent
//! key comparison. A tie that survives this key is a genuine symmetry orbit:
//! no row-permutation-equivariant rule can choose one member of it, so callers
//! take the class atomically.

use rayon::prelude::*;

/// One row's sorted pairwise profile, with the thread pool spent on the row axis.
///
/// `total_cmp` is a total order and the elements are plain values, so an
/// unstable parallel sort produces the identical sequence a sequential `sort_by`
/// would; only the order of equal-comparing bit patterns could differ, and equal
/// bit patterns are indistinguishable to every consumer.
fn sorted_profile_row_parallel<K>(n: usize, anchor: usize, pair_key: &K) -> Vec<f64>
where
    K: Fn(usize, usize) -> f64 + Sync,
{
    let mut profile: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|row| pair_key(anchor, row))
        .collect();
    profile.par_sort_by(f64::total_cmp);
    profile
}

/// [`sorted_profile_row_parallel`] with the pool spent on the CANDIDATES
/// instead of on one profile's rows. Bit-identical to the parallel form: the
/// mapped sequence is index-ordered either way, and sorting `f64` by `total_cmp`
/// can only permute identical bit patterns.
fn sorted_profile_serial<K>(n: usize, anchor: usize, pair_key: &K) -> Vec<f64>
where
    K: Fn(usize, usize) -> f64 + Sync,
{
    let mut profile: Vec<f64> = (0..n).map(|row| pair_key(anchor, row)).collect();
    profile.sort_by(f64::total_cmp);
    profile
}

/// Lexicographic order on two sorted profiles. `Equal` means the unordered
/// geometry cannot distinguish the two rows: choosing one by row index would
/// violate permutation invariance, so the selector must treat their whole class
/// atomically.
pub(crate) fn sorted_profile_cmp(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let ordering = x.total_cmp(y);
        if !ordering.is_eq() {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

/// Reduce a candidate list that is already tied on every `O(1)` invariant key to
/// the sub-list attaining the lexicographically LEAST sorted pairwise profile,
/// in ascending candidate order.
///
/// `n` is the support size (the profile's length), `tied` the candidate rows in
/// canonical order, and `pair_key(anchor, row)` the isometry-invariant pairwise
/// scalar whose multiset over `row` forms the profile — a dot product on the
/// sphere, a squared distance in the plane.
///
/// `on_profile_builds` observes the number of `O(n log n)` profiles this
/// actually constructs. Production callers pass a zero-sized no-op closure that
/// optimizes away; tests use the same path to state the asymptotic contract in
/// operation counts rather than in wall-clock noise.
pub(crate) fn resolve_sorted_profile_tie<K, F>(
    n: usize,
    tied: &[usize],
    pair_key: K,
    on_profile_builds: &mut F,
) -> Vec<usize>
where
    K: Fn(usize, usize) -> f64 + Sync,
    F: FnMut(usize),
{
    // A lone candidate is already the extremum of any refinement of the keys it
    // has already won, and is trivially profile-equal to itself. This is the
    // whole of the saving on data with no exact symmetry.
    if tied.len() <= 1 {
        return tied.to_vec();
    }
    on_profile_builds(tied.len());
    // Spend the pool on whichever axis actually has work: across candidates once
    // there are enough of them to fill the pool, otherwise across each profile's
    // rows. Both forms produce bit-identical profiles, so this is a scheduling
    // choice only.
    let profiles: Vec<Vec<f64>> = if tied.len() >= rayon::current_num_threads() {
        tied.par_iter()
            .map(|&i| sorted_profile_serial(n, i, &pair_key))
            .collect()
    } else {
        tied.iter()
            .map(|&i| sorted_profile_row_parallel(n, i, &pair_key))
            .collect()
    };
    let mut least = 0usize;
    for candidate in 1..profiles.len() {
        if sorted_profile_cmp(&profiles[candidate], &profiles[least]).is_lt() {
            least = candidate;
        }
    }
    (0..tied.len())
        .filter(|&k| sorted_profile_cmp(&profiles[k], &profiles[least]).is_eq())
        .map(|k| tied[k])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The resolver must agree, row for row, with the running-incumbent
    /// comparator scan it replaces — that equivalence is the entire licence for
    /// the cheaper form. Checked on a fixture with a genuine four-fold orbit
    /// (the corners of a square) plus asymmetric filler.
    #[test]
    fn extremum_then_refine_matches_the_incumbent_comparator_scan() {
        let points: Vec<[f64; 2]> = vec![
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.25, -0.5],
            [-0.75, 0.125],
        ];
        let n = points.len();
        let pair_key = |i: usize, j: usize| {
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            dx * dx + dy * dy
        };
        let profile = |i: usize| sorted_profile_serial(n, i, &pair_key);

        for tied in [
            vec![0_usize, 1, 2, 3],
            vec![0, 1, 2, 3, 4],
            vec![4, 5, 6],
            vec![5],
            vec![1, 3],
            (0..n).collect::<Vec<usize>>(),
        ] {
            // The shape this replaces: a two-profile comparator behind a running
            // incumbent, then a filter against that incumbent.
            let pivot = tied
                .iter()
                .copied()
                .reduce(|a, b| {
                    if sorted_profile_cmp(&profile(a), &profile(b)).is_lt() {
                        a
                    } else {
                        b
                    }
                })
                .expect("non-empty tie class");
            let expected: Vec<usize> = tied
                .iter()
                .copied()
                .filter(|&i| sorted_profile_cmp(&profile(i), &profile(pivot)).is_eq())
                .collect();

            let got = resolve_sorted_profile_tie(n, &tied, &pair_key, &mut |_| {});
            assert_eq!(got, expected, "tie class {tied:?} resolved differently");
        }
    }

    /// A singleton class is the case that dominates real data, and it must cost
    /// nothing: the replaced comparator form still built two `O(n log n)`
    /// profiles there to establish that a row equals itself.
    #[test]
    fn a_singleton_tie_class_builds_no_profile() {
        let mut builds = 0usize;
        let got =
            resolve_sorted_profile_tie(1_000, &[7], |i, j| (i as f64) - (j as f64), &mut |b| {
                builds += b
            });
        assert_eq!(got, vec![7]);
        assert_eq!(
            builds, 0,
            "a lone candidate needs no profile to win a refinement"
        );
    }

    /// A real class charges exactly one profile per candidate — used for both the
    /// choice and the class filter — not the `4·|C| − 2` of the comparator form.
    #[test]
    fn a_real_tie_class_builds_exactly_one_profile_per_candidate() {
        let mut builds = 0usize;
        let tied = [0_usize, 1, 2, 3, 4];
        let resolved =
            resolve_sorted_profile_tie(64, &tied, |i, j| ((i * j) % 7) as f64, &mut |b| {
                builds += b
            });
        assert_eq!(builds, tied.len());
        // The resolution is a subset of the class it was handed, and never empty:
        // a tie must resolve to at least one of its own members, whatever the key.
        assert!(!resolved.is_empty());
        assert!(resolved.iter().all(|index| tied.contains(index)));
    }
}
