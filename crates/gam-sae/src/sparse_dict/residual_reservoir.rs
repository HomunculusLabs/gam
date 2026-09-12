//! Bounded reservoir of the worst-reconstructed rows seen in a streaming epoch.
//!
//! Shared by the atom lane (`stream.rs`, dead-atom revival) and the block lane
//! (`block_stream.rs`, dead-block birth proposals): both keep the top-`cap`
//! residual rows by energy with one-shot's deterministic tie-break, and both
//! used to carry a private copy of this type (#2470). The capacity is the
//! caller's: `K` for revival (at most one atom per row, at most `K` dead
//! atoms) and `k_aux · b` for block births. Peak memory is `cap × P` f32 —
//! never `N × K`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Squared norm at or below which a residual row `x − Σ_t w_t·d_t` is rounding
/// rather than structure. Each entry rounds by at most `γ_k` (at `unit_roundoff`,
/// over the `k = rounded_operations` operations that formed it) of
/// `|x_c| + Σ_t |w_t·d_{tc}|`, and by the triangle inequality those bounds have
/// Euclidean norm at most `γ_k·(‖x‖ + Σ_t |w_t|·‖d_t‖)`, which the caller passes
/// as `row_norm + reconstruction_mass`.
pub(super) fn residual_rounding_energy(
    unit_roundoff: f64,
    rounded_operations: usize,
    row_norm: f64,
    reconstruction_mass: f64,
) -> f64 {
    let scaled = rounded_operations as f64 * unit_roundoff;
    if !(scaled < 1.0) {
        return f64::INFINITY;
    }
    let band = scaled / (1.0 - scaled) * (row_norm + reconstruction_mass);
    band * band
}

/// One candidate row: its residual vector (under the pre-refresh decoder) and
/// the energy used to rank it. Ordered so the [`BinaryHeap`]'s max is the
/// MOST-evictable entry (smallest energy, ties broken toward the larger global
/// index) — that keeps the reservoir holding the worst-reconstructed rows with
/// one-shot's deterministic tie-break (descending energy, ascending row index).
pub(super) struct ResidRow {
    pub(super) norm2: f64,
    pub(super) global_index: u64,
    pub(super) residual: Vec<f32>,
}

impl PartialEq for ResidRow {
    fn eq(&self, other: &Self) -> bool {
        self.norm2 == other.norm2 && self.global_index == other.global_index
    }
}
impl Eq for ResidRow {}
impl Ord for ResidRow {
    fn cmp(&self, other: &Self) -> Ordering {
        // "Greater" == more evictable == smaller residual energy, then larger
        // global index. `total_cmp` keeps this total and NaN-free (norms are
        // finite sums of squares).
        match other.norm2.total_cmp(&self.norm2) {
            Ordering::Equal => self.global_index.cmp(&other.global_index),
            ord => ord,
        }
    }
}
impl PartialOrd for ResidRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bounded reservoir of the worst-reconstructed rows seen this epoch.
pub(super) struct ResidualReservoir {
    cap: usize,
    heap: BinaryHeap<ResidRow>,
}

impl ResidualReservoir {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            heap: BinaryHeap::new(),
        }
    }

    /// Offer a row's residual to the reservoir. A row whose residual energy is at
    /// or below its `rounding_energy` ([`residual_rounding_energy`]) is an exact
    /// reconstruction to within its own arithmetic, seeds nothing, and is dropped.
    pub(super) fn offer(
        &mut self,
        norm2: f64,
        rounding_energy: f64,
        global_index: u64,
        residual: Vec<f32>,
    ) {
        if norm2 <= rounding_energy {
            return;
        }
        let row = ResidRow {
            norm2,
            global_index,
            residual,
        };
        if self.heap.len() < self.cap {
            self.heap.push(row);
            return;
        }
        // The heap's max is the most-evictable held row; replace it only when the
        // newcomer is strictly LESS evictable (a worse-reconstructed row, or an
        // equal-energy row with a smaller index).
        if let Some(worst_kept) = self.heap.peek() {
            if row.cmp(worst_kept) == Ordering::Less {
                self.heap.pop();
                self.heap.push(row);
            }
        }
    }

    pub(super) fn clear(&mut self) {
        self.heap.clear();
    }

    /// Rows ranked worst-first: descending residual energy, ties by ascending
    /// global index — the one-shot `revive_dead_atoms` /
    /// `dead_block_birth_proposals` order.
    pub(super) fn ranked(&self) -> Vec<&ResidRow> {
        let mut rows: Vec<&ResidRow> = self.heap.iter().collect();
        rows.sort_by(|a, b| {
            b.norm2
                .total_cmp(&a.norm2)
                .then_with(|| a.global_index.cmp(&b.global_index))
        });
        rows
    }
}
