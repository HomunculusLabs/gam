//! The standard-normal FLEX row's fifth information contraction.
//!
//! Under the standard-normal latent measure a FLEX row calibrates its intercept
//! through the cell integral `M(a, θ) = Σ_cells ∫ φ(z)·Φ(η(z; a, θ)) dz − μ(q)`,
//! and its loss is `ℓ = −w·logΦ(s·G)` with the observed index
//! `G(a, θ) = η(z_obs; a, θ)` at the root `a(θ)`. The Jeffreys third information
//! derivative reads `T_c[k][l] = Σ_{d,e} ℓ_{klcde}·u_d·v_e` for every primary `c`.
//!
//! The contraction is hand-derived in three passes.
//! 1. Explicit partials of `M` and of `G` in `(a, θ)`. A slot is the raw
//!    direction `u`, the raw direction `v`, the intercept, or one of at most
//!    three primary coordinates. Cell integrals come from the set-partition
//!    moment kernel, and every interior link-knot crossing adds its
//!    moving-boundary Leibniz flux at orders four and five.
//! 2. The implicit intercept derivatives `a_B` for every subset `B` of the slot
//!    labels `{u, v, c, k, l}`, solved from `D^B M = 0`.
//! 3. `D^B G` through the same chain rule, then Faà di Bruno with the loss.

use super::family::*;
use super::gradient_paths::signed_probit_neglog_unary_stack_fifth;
use super::hessian_paths::{BernoulliMarginalSlopeRowExactContext, PrimarySlices};
use super::*;

/// Label masks of the total derivative `D⁵ℓ[u, v, c, k, l]`: bit 0 is `u`, bit 1
/// is `v`, and bits 2 to 4 are the free primary coordinates `c`, `k`, `l`.
const LABEL_U: usize = 1;
const LABEL_V: usize = 2;
const FREE_LABELS: usize = 0b11100;
const ALL_LABELS: usize = 0b11111;

/// Every ordering of up to three coordinate slots, for filling a symmetric tensor.
const ORDERINGS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// One derivative slot of an explicit partial in `(a, θ)`.
#[derive(Clone, Copy)]
enum ExplicitSlot {
    U,
    V,
    Intercept,
    Coordinate(usize),
}

/// Every explicit partial the chain rule reads: `m_u` and `m_v` raw direction
/// slots, `j` intercept slots and `n ≤ 3` coordinate slots, of total order one
/// through five.
fn explicit_signatures() -> impl Iterator<Item = (usize, usize, usize, usize)> {
    (0..2_usize)
        .flat_map(|m_u| {
            (0..2_usize).flat_map(move |m_v| {
                (0..6_usize).flat_map(move |j| (0..4_usize).map(move |n| (m_u, m_v, j, n)))
            })
        })
        .filter(|&(m_u, m_v, j, n)| (1..=5).contains(&(m_u + m_v + j + n)))
}

/// The slot list `[u]^{m_u} [v]^{m_v} [a]^j` followed by `coordinates`.
fn explicit_slots(
    m_u: usize,
    m_v: usize,
    j: usize,
    coordinates: &[usize],
) -> ([ExplicitSlot; 5], usize) {
    let mut slots = [ExplicitSlot::Intercept; 5];
    let mut len = 0;
    if m_u == 1 {
        slots[len] = ExplicitSlot::U;
        len += 1;
    }
    if m_v == 1 {
        slots[len] = ExplicitSlot::V;
        len += 1;
    }
    len += j;
    for &coordinate in coordinates {
        slots[len] = ExplicitSlot::Coordinate(coordinate);
        len += 1;
    }
    (slots, len)
}

/// Visit every multiset of `n ≤ 3` entries of `active`, in sorted order.
fn for_each_sorted_tuple(
    active: &[usize],
    n: usize,
    mut visit: impl FnMut(&[usize]) -> Result<(), String>,
) -> Result<(), String> {
    let m = active.len();
    match n {
        0 => visit(&[]),
        1 => {
            for i in 0..m {
                visit(&[active[i]])?;
            }
            Ok(())
        }
        2 => {
            for i in 0..m {
                for k in i..m {
                    visit(&[active[i], active[k]])?;
                }
            }
            Ok(())
        }
        _ => {
            for i in 0..m {
                for k in i..m {
                    for l in k..m {
                        visit(&[active[i], active[k], active[l]])?;
                    }
                }
            }
            Ok(())
        }
    }
}

/// Visit every set partition of the labels in `mask` exactly once, as the label
/// masks of its blocks. The empty set has one partition, with no blocks.
fn for_each_partition(mask: usize, mut visit: impl FnMut(&[usize])) {
    let mut elements = [0_usize; 5];
    let mut count = 0;
    for label in 0..5 {
        if mask & (1 << label) != 0 {
            elements[count] = label;
            count += 1;
        }
    }
    if count == 0 {
        visit(&[]);
        return;
    }
    let mut block = [0_usize; 5];
    loop {
        let mut masks = [0_usize; 5];
        let mut blocks = 0;
        for (position, &index) in block[..count].iter().enumerate() {
            masks[index] |= 1 << elements[position];
            blocks = blocks.max(index + 1);
        }
        visit(&masks[..blocks]);
        let mut position = count;
        loop {
            if position <= 1 {
                return;
            }
            position -= 1;
            let ceiling = 1 + block[..position].iter().copied().max().unwrap_or(0);
            if block[position] < ceiling {
                block[position] += 1;
                for later in block[position + 1..count].iter_mut() {
                    *later = 0;
                }
                break;
            }
        }
    }
}

#[inline]
fn free_count(mask: usize) -> u32 {
    (mask & FREE_LABELS).count_ones()
}

/// Flat index of a tensor over the free labels of `mask`, first free label fastest.
#[inline]
fn tensor_index(mask: usize, labels: &[usize; 5], r: usize) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for (label, &index) in labels.iter().enumerate() {
        if mask & FREE_LABELS & (1 << label) != 0 {
            flat += index * stride;
            stride *= r;
        }
    }
    flat
}

/// Per-label primary indices of entry `flat` of the tensor over `mask`'s free labels.
#[inline]
fn decode_labels(mask: usize, flat: usize, r: usize) -> [usize; 5] {
    let mut labels = [0_usize; 5];
    let mut remainder = flat;
    for (label, index) in labels.iter_mut().enumerate() {
        if mask & FREE_LABELS & (1 << label) != 0 {
            *index = remainder % r;
            remainder /= r;
        }
    }
    labels
}

/// Explicit partials of one map in `(a, θ)`, keyed by their signature
/// `(m_u, m_v, j, n)` and stored as dense symmetric tensors over the `n`
/// coordinate slots.
struct ExplicitTensors {
    r: usize,
    values: Vec<Vec<f64>>,
    vanishing: Vec<bool>,
}

impl ExplicitTensors {
    fn new(r: usize) -> Self {
        let mut values = vec![Vec::new(); 2 * 2 * 6 * 4];
        for (m_u, m_v, j, n) in explicit_signatures() {
            values[Self::slot(m_u, m_v, j, n)] = vec![0.0; r.pow(n as u32)];
        }
        let vanishing = vec![true; values.len()];
        Self {
            r,
            values,
            vanishing,
        }
    }

    #[inline]
    fn slot(m_u: usize, m_v: usize, j: usize, n: usize) -> usize {
        ((m_u * 2 + m_v) * 6 + j) * 4 + n
    }

    #[inline]
    fn get(&self, m_u: usize, m_v: usize, j: usize, n: usize, flat: usize) -> f64 {
        self.values[Self::slot(m_u, m_v, j, n)][flat]
    }

    /// Add `value` at every distinct ordering of `coordinates`.
    fn add_symmetric(&mut self, m_u: usize, m_v: usize, j: usize, coordinates: &[usize], value: f64) {
        let r = self.r;
        let n = coordinates.len();
        let target = &mut self.values[Self::slot(m_u, m_v, j, n)];
        let mut written = [usize::MAX; 6];
        let mut count = 0;
        for ordering in &ORDERINGS {
            if ordering[..n].iter().any(|&position| position >= n) {
                continue;
            }
            let flat = ordering[..n]
                .iter()
                .rev()
                .fold(0, |acc, &position| acc * r + coordinates[position]);
            if written[..count].contains(&flat) {
                continue;
            }
            written[count] = flat;
            count += 1;
            target[flat] += value;
        }
    }

    fn seal(&mut self) {
        for (flag, values) in self.vanishing.iter_mut().zip(&self.values) {
            *flag = values.iter().all(|&value| value == 0.0);
        }
    }
}

/// The z-polynomials every mixed partial of one de-nested index is built from.
///
/// The index is `scale·(a + b·z + b·H(z) + L(a + b·z))` and is linear in each
/// score-warp and link-deviation coefficient, so a partial over two deviation
/// coefficients vanishes and every other partial is a cubic:
/// `base[n_a][n_b] = ∂_a^{n_a} ∂_b^{n_b} η`, and for a deviation coordinate `p`
/// `deviation[p][n_a][n_b] = ∂_a^{n_a} ∂_b^{n_b} ∂_p η`. Both vanish beyond third
/// order in `(a, b)`. `directional[e]` contracts the deviation columns with the
/// deviation components of direction `e`.
struct IndexAtoms {
    q: usize,
    slope: usize,
    base: [[[f64; 4]; 4]; 4],
    deviation: Vec<[[[f64; 4]; 4]; 4]>,
    directional: [[[[f64; 4]; 4]; 4]; 2],
    slope_components: [f64; 2],
    /// Raw leading coefficient of each link-deviation column's local cubic.
    link_leading: Vec<f64>,
    /// The slope and every deviation coordinate whose column does not vanish here.
    active: Vec<usize>,
}

impl IndexAtoms {
    fn new(
        family: &BernoulliMarginalSlopeFamily,
        primary: &PrimarySlices,
        a: f64,
        b: f64,
        base: [[[f64; 4]; 4]; 4],
        score_point: f64,
        link_point: f64,
        directions: [&Array1<f64>; 2],
    ) -> Result<Self, String> {
        let r = primary.total;
        let scale = family.probit_frailty_scale();
        let mut deviation = vec![[[[0.0; 4]; 4]; 4]; r];
        let mut link_leading = vec![0.0; r];
        if let (Some(range), Some(runtime)) = (primary.h.as_ref(), family.score_warp.as_ref()) {
            BernoulliMarginalSlopeFamily::for_each_deviation_basis_cubic_at(
                runtime,
                range,
                score_point,
                "score-warp fifth-order atoms",
                |_, index, span| {
                    deviation[index][0][0] =
                        scale_coeff4(exact_kernel::score_basis_cell_coefficients(span, b), scale);
                    deviation[index][0][1] =
                        scale_coeff4(exact_kernel::score_basis_cell_coefficients(span, 1.0), scale);
                    Ok(())
                },
            )?;
        }
        if let (Some(range), Some(runtime)) = (primary.w.as_ref(), family.link_dev.as_ref()) {
            BernoulliMarginalSlopeFamily::for_each_deviation_basis_cubic_at(
                runtime,
                range,
                link_point,
                "link-deviation fifth-order atoms",
                |_, index, span| {
                    let column = &mut deviation[index];
                    column[0][0] =
                        scale_coeff4(exact_kernel::link_basis_cell_coefficients(span, a, b), scale);
                    let (da, db) = exact_kernel::link_basis_cell_coefficient_partials(span, a, b);
                    column[1][0] = scale_coeff4(da, scale);
                    column[0][1] = scale_coeff4(db, scale);
                    let (daa, dab, dbb) = exact_kernel::link_basis_cell_second_partials(span, a, b);
                    column[2][0] = scale_coeff4(daa, scale);
                    column[1][1] = scale_coeff4(dab, scale);
                    column[0][2] = scale_coeff4(dbb, scale);
                    let (daaa, daab, dabb, dbbb) = exact_kernel::link_basis_cell_third_partials(span);
                    column[3][0] = scale_coeff4(daaa, scale);
                    column[2][1] = scale_coeff4(daab, scale);
                    column[1][2] = scale_coeff4(dabb, scale);
                    column[0][3] = scale_coeff4(dbbb, scale);
                    link_leading[index] = span.c3;
                    Ok(())
                },
            )?;
        }
        let mut directional = [[[[0.0; 4]; 4]; 4]; 2];
        for (target, direction) in directional.iter_mut().zip(directions) {
            for (p, column) in deviation.iter().enumerate() {
                let weight = direction[p];
                if weight == 0.0 || p == primary.q || p == primary.slope {
                    continue;
                }
                for (target_row, column_row) in target.iter_mut().zip(column) {
                    for (target_cell, column_cell) in target_row.iter_mut().zip(column_row) {
                        for (slot, &value) in target_cell.iter_mut().zip(column_cell) {
                            *slot += weight * value;
                        }
                    }
                }
            }
        }
        let active = (0..r)
            .filter(|&p| {
                p == primary.slope
                    || (p != primary.q
                        && deviation[p].iter().flatten().flatten().any(|&value| value != 0.0))
            })
            .collect();
        Ok(Self {
            q: primary.q,
            slope: primary.slope,
            base,
            deviation,
            directional,
            slope_components: [directions[0][primary.slope], directions[1][primary.slope]],
            link_leading,
            active,
        })
    }

    /// `∂^S η` over the slots of `slots` selected by `mask`. Each direction
    /// differentiates through its slope component or through its deviation
    /// components, and at most one deviation derivative survives.
    fn subset_polynomial(&self, slots: &[ExplicitSlot], mask: usize) -> [f64; 4] {
        let mut n_a = 0;
        let mut n_b = 0;
        let mut coordinate_deviation = None;
        let mut deviations = 0;
        let mut directions = [0_usize; 2];
        let mut direction_count = 0;
        for (position, &slot) in slots.iter().enumerate() {
            if mask & (1 << position) == 0 {
                continue;
            }
            match slot {
                ExplicitSlot::U => {
                    directions[direction_count] = 0;
                    direction_count += 1;
                }
                ExplicitSlot::V => {
                    directions[direction_count] = 1;
                    direction_count += 1;
                }
                ExplicitSlot::Intercept => n_a += 1,
                ExplicitSlot::Coordinate(p) => {
                    if p == self.q {
                        return [0.0; 4];
                    }
                    if p == self.slope {
                        n_b += 1;
                    } else {
                        deviations += 1;
                        coordinate_deviation = Some(p);
                    }
                }
            }
        }
        let mut out = [0.0; 4];
        for on_slope in 0..(1_usize << direction_count) {
            let mut weight = 1.0;
            let mut b_order = n_b;
            let mut direction_deviation = None;
            let mut total_deviations = deviations;
            for (position, &e) in directions[..direction_count].iter().enumerate() {
                if on_slope & (1 << position) != 0 {
                    weight *= self.slope_components[e];
                    b_order += 1;
                } else {
                    total_deviations += 1;
                    direction_deviation = Some(e);
                }
            }
            if weight == 0.0 || total_deviations > 1 || n_a + b_order > 3 {
                continue;
            }
            let polynomial = if let Some(p) = coordinate_deviation {
                &self.deviation[p][n_a][b_order]
            } else if let Some(e) = direction_deviation {
                &self.directional[e][n_a][b_order]
            } else {
                &self.base[n_a][b_order]
            };
            for (slot, &value) in out.iter_mut().zip(polynomial) {
                *slot += weight * value;
            }
        }
        out
    }
}

/// The `(a, b)` partials of a partition cell's index, scaled.
fn cell_base_partials(
    partition_cell: &exact_kernel::DenestedPartitionCell,
    a: f64,
    b: f64,
    scale: f64,
) -> [[[f64; 4]; 4]; 4] {
    let mut base = [[[0.0; 4]; 4]; 4];
    let (da, db) = exact_kernel::denested_cell_coefficient_partials(
        partition_cell.score_span,
        partition_cell.link_span,
        a,
        b,
    );
    let (daa, dab, dbb) =
        exact_kernel::link_basis_cell_second_partials(partition_cell.link_span, a, b);
    let (daaa, daab, dabb, dbbb) = exact_kernel::denested_cell_third_partials(partition_cell.link_span);
    base[1][0] = scale_coeff4(da, scale);
    base[0][1] = scale_coeff4(db, scale);
    base[2][0] = scale_coeff4(daa, scale);
    base[1][1] = scale_coeff4(dab, scale);
    base[0][2] = scale_coeff4(dbb, scale);
    base[3][0] = scale_coeff4(daaa, scale);
    base[2][1] = scale_coeff4(daab, scale);
    base[1][2] = scale_coeff4(dabb, scale);
    base[0][3] = scale_coeff4(dbbb, scale);
    base
}

/// The observed index and its `(a, b)` partials, already scaled.
fn observed_base_partials(observed: &ObservedDenestedCellPartials) -> [[[f64; 4]; 4]; 4] {
    let mut base = [[[0.0; 4]; 4]; 4];
    base[0][0] = observed.coeff;
    base[1][0] = observed.dc_da;
    base[0][1] = observed.dc_db;
    base[2][0] = observed.dc_daa;
    base[1][1] = observed.dc_dab;
    base[0][2] = observed.dc_dbb;
    base[3][0] = observed.dc_daaa;
    base[2][1] = observed.dc_daab;
    base[1][2] = observed.dc_dabb;
    base[0][3] = observed.dc_dbbb;
    base
}

#[derive(Clone, Copy)]
struct FluxSlot {
    /// `∂u/∂s` at the crossing, `u = a + b·z`.
    crossing: f64,
    /// `∂b/∂s`.
    slope: f64,
    /// `∂η/∂s` at the crossing.
    eta: f64,
    /// `∂Δc₃/∂s`.
    jump: f64,
}

impl FluxSlot {
    const ZERO: Self = Self {
        crossing: 0.0,
        slope: 0.0,
        eta: 0.0,
        jump: 0.0,
    };
}

/// Moving-boundary Leibniz flux of one interior link-knot crossing
/// `z* = (τ − a)/b`.
///
/// Across the crossing the index changes by `scale·Δc₃·(u − τ)³`, so the
/// calibration integrand is `C²` and orders through three need no boundary
/// term. Differentiating `Σ_cells ∫` further moves `z*` (`∂z*/∂s = −u_s/b`):
///   order 4: `K·(−Δc₃/b)·Π u_s`;
///   order 5: `K·[−(1/b)·Σ_t (∂_tΔc₃ − η*·∂_tη·Δc₃)·Π_{r≠t} u_r
///            − (Δc₃/b²)·(z* + η*·η_z)·Π u_s + (Δc₃/b²)·Σ_t ∂_t b·Π_{r≠t} u_r]`,
/// with `K = 6·scale·e^{−q(z*)}/2π` and per slot `u_s = ∂u/∂s`, `∂_t b`, `∂_tη`
/// and `∂_tΔc₃` at the crossing. The order-five form is the jump of the fourth
/// integrand carried by the moving crossing plus the order-four flux
/// differentiated along it, and it is symmetric in its slots.
struct KnotFlux {
    prefactor: f64,
    jump: f64,
    slope: f64,
    eta_star: f64,
    q_z: f64,
    intercept: FluxSlot,
    coordinates: Vec<FluxSlot>,
    directions: [FluxSlot; 2],
}

impl KnotFlux {
    fn new(
        cell: exact_kernel::DenestedCubicCell,
        atoms: &IndexAtoms,
        column_jumps: &[f64],
        jump: f64,
        b: f64,
        scale: f64,
        primary: &PrimarySlices,
        directions: [&Array1<f64>; 2],
    ) -> Self {
        let z_star = cell.right;
        let eta_star = cell.eta(z_star);
        let eta_z = cell.c1 + z_star * (2.0 * cell.c2 + 3.0 * cell.c3 * z_star);
        let mut coordinates = vec![FluxSlot::ZERO; primary.total];
        for (p, slot) in coordinates.iter_mut().enumerate() {
            if p == primary.slope {
                *slot = FluxSlot {
                    crossing: z_star,
                    slope: 1.0,
                    eta: eval_coeff4_at(&atoms.base[0][1], z_star),
                    jump: 0.0,
                };
            } else if p != primary.q {
                *slot = FluxSlot {
                    crossing: 0.0,
                    slope: 0.0,
                    eta: eval_coeff4_at(&atoms.deviation[p][0][0], z_star),
                    jump: column_jumps[p],
                };
            }
        }
        let direction_slot = |direction: &Array1<f64>| {
            let mut slot = FluxSlot {
                crossing: direction[primary.slope] * z_star,
                slope: direction[primary.slope],
                eta: 0.0,
                jump: 0.0,
            };
            for (p, coordinate) in coordinates.iter().enumerate() {
                slot.eta += direction[p] * coordinate.eta;
                slot.jump += direction[p] * coordinate.jump;
            }
            slot
        };
        let direction_slots = [direction_slot(directions[0]), direction_slot(directions[1])];
        Self {
            prefactor: 6.0 * scale * (-cell.q(z_star)).exp() / std::f64::consts::TAU,
            jump,
            slope: b,
            eta_star,
            q_z: z_star + eta_star * eta_z,
            intercept: FluxSlot {
                crossing: 1.0,
                slope: 0.0,
                eta: eval_coeff4_at(&atoms.base[1][0], z_star),
                jump: 0.0,
            },
            coordinates,
            directions: direction_slots,
        }
    }

    fn slot(&self, slot: ExplicitSlot) -> FluxSlot {
        match slot {
            ExplicitSlot::U => self.directions[0],
            ExplicitSlot::V => self.directions[1],
            ExplicitSlot::Intercept => self.intercept,
            ExplicitSlot::Coordinate(p) => self.coordinates[p],
        }
    }

    fn flux(&self, slots: &[ExplicitSlot]) -> f64 {
        let mut values = [FluxSlot::ZERO; 5];
        for (value, &slot) in values.iter_mut().zip(slots) {
            *value = self.slot(slot);
        }
        let values = &values[..slots.len()];
        let crossing_product_without = |skip: usize| {
            let mut product = 1.0;
            for (position, value) in values.iter().enumerate() {
                if position != skip {
                    product *= value.crossing;
                }
            }
            product
        };
        let b = self.slope;
        let all = crossing_product_without(values.len());
        match values.len() {
            4 => self.prefactor * (-self.jump / b) * all,
            5 => {
                let mut total = -(self.jump / (b * b)) * self.q_z * all;
                for (position, value) in values.iter().enumerate() {
                    let others = crossing_product_without(position);
                    total -= (value.jump - self.eta_star * value.eta * self.jump) / b * others;
                    total += self.jump / (b * b) * value.slope * others;
                }
                self.prefactor * total
            }
            _ => 0.0,
        }
    }
}

/// Add `Σ_{R ⊆ B} Σ_{π ∈ Π(B∖R)} X[R; |π|]·Π_{P ∈ π} a_P` to `out`: the chain rule
/// of an explicit map `X(a(θ), θ)` over the labels of `mask`, where `R` holds the
/// labels the map differentiates directly and each block `P` enters through the
/// intercept derivative `a_P`. With `solving`, the single-block term `X_a·a_B`
/// is left out so the caller can solve for `a_B`.
fn accumulate_chain_rule(
    out: &mut [f64],
    mask: usize,
    explicit: &ExplicitTensors,
    intercept: &[Vec<f64>],
    solving: bool,
) {
    let r = explicit.r;
    let mut raw = mask;
    loop {
        let rest = mask & !raw;
        let m_u = usize::from(raw & LABEL_U != 0);
        let m_v = usize::from(raw & LABEL_V != 0);
        let n = free_count(raw) as usize;
        for_each_partition(rest, |blocks| {
            if solving && raw == 0 && blocks.len() == 1 {
                return;
            }
            let signature = ExplicitTensors::slot(m_u, m_v, blocks.len(), n);
            if explicit.vanishing[signature] {
                return;
            }
            let coefficients = &explicit.values[signature];
            for (flat, target) in out.iter_mut().enumerate() {
                let labels = decode_labels(mask, flat, r);
                let mut term = coefficients[tensor_index(raw, &labels, r)];
                for &block in blocks {
                    term *= intercept[block][tensor_index(block, &labels, r)];
                }
                *target += term;
            }
        });
        if raw == 0 {
            break;
        }
        raw = (raw - 1) & mask;
    }
}

/// `a_B` for every label subset `B`, in order of size, from `D^B M = 0`.
fn implicit_intercept_totals(calibration: &ExplicitTensors, f_a: f64) -> Vec<Vec<f64>> {
    let r = calibration.r;
    let mut intercept: Vec<Vec<f64>> = (0..32)
        .map(|mask| vec![0.0; r.pow(free_count(mask))])
        .collect();
    let mut masks: Vec<usize> = (1..32).collect();
    masks.sort_by_key(|mask| mask.count_ones());
    for mask in masks {
        let mut accumulated = vec![0.0; intercept[mask].len()];
        accumulate_chain_rule(&mut accumulated, mask, calibration, &intercept, true);
        for value in &mut accumulated {
            *value /= -f_a;
        }
        intercept[mask] = accumulated;
    }
    intercept
}

/// `D^B G` for every label subset `B`.
fn observed_index_totals(index: &ExplicitTensors, intercept: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r = index.r;
    (0..32)
        .map(|mask| {
            let mut out = vec![0.0; r.pow(free_count(mask))];
            if mask != 0 {
                accumulate_chain_rule(&mut out, mask, index, intercept, false);
            }
            out
        })
        .collect()
}

/// `D⁵ℓ[u, v, c, k, l] = Σ_π ℓ⁽|π|⁾·Π_{P ∈ π} D^P G` over the partitions of all five labels.
fn compose_loss(index_totals: &[Vec<f64>], loss: &[f64; 6], r: usize) -> Vec<f64> {
    let mut fifth = vec![0.0; r * r * r];
    for_each_partition(ALL_LABELS, |blocks| {
        for (flat, target) in fifth.iter_mut().enumerate() {
            let labels = decode_labels(ALL_LABELS, flat, r);
            let mut term = loss[blocks.len()];
            for &block in blocks {
                term *= index_totals[block][tensor_index(block, &labels, r)];
            }
            *target += term;
        }
    });
    fifth
}

impl BernoulliMarginalSlopeFamily {
    /// `{T_c[k][l] = Σ_{d,e} ℓ_{klcde}·u_d·v_e}` over every primary `c` of one
    /// standard-normal FLEX row, from the hand cell-moment kernel. The directions
    /// are primary-space.
    pub(super) fn standard_normal_flex_row_fifth_axis_slabs(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let r = primary.total;
        if dir_u.len() != r || dir_v.len() != r {
            return Err(format!(
                "bernoulli standard-normal flex fifth contraction direction lengths ({},{}) != primary dimension {r}",
                dir_u.len(),
                dir_v.len()
            ));
        }
        if dir_u.iter().all(|value| *value == 0.0) || dir_v.iter().all(|value| *value == 0.0) {
            return Ok((0..r).map(|_| Array2::<f64>::zeros((r, r))).collect());
        }
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err("non-finite standard-normal flexible row context in fifth contraction".into());
        }
        let a = row_ctx.intercept;
        let scale = self.probit_frailty_scale();
        let marginal = self.marginal_link_map(q)?;
        let directions = [dir_u, dir_v];

        let cells = self.denested_partition_cells(a, b, beta_h, beta_w)?;
        let mut calibration = ExplicitTensors::new(r);
        let mut cell_atoms = Vec::with_capacity(cells.len());
        for partition_cell in &cells {
            let cell = partition_cell.cell;
            let state = exact_kernel::evaluate_cell_derivative_moments_uncached(cell, 27)?;
            let partition = exact_kernel::CellPartitionMoments::new(cell, &state.moments, 5)?;
            let z_mid = exact_kernel::interval_probe_point(cell.left, cell.right)?;
            let atoms = IndexAtoms::new(
                self,
                primary,
                a,
                b,
                cell_base_partials(partition_cell, a, b, scale),
                z_mid,
                a + b * z_mid,
                directions,
            )?;
            for (m_u, m_v, j, n) in explicit_signatures() {
                for_each_sorted_tuple(&atoms.active, n, |coordinates| {
                    let (slots, len) = explicit_slots(m_u, m_v, j, coordinates);
                    let slots = &slots[..len];
                    let mut subsets = [[0.0_f64; 4]; 32];
                    for (mask, polynomial) in subsets.iter_mut().enumerate().take(1 << len).skip(1) {
                        *polynomial = atoms.subset_polynomial(slots, mask);
                    }
                    let value = partition.derivative(len, &subsets)?;
                    calibration.add_symmetric(m_u, m_v, j, coordinates, value);
                    Ok(())
                })?;
            }
            cell_atoms.push(atoms);
        }

        let mu = [
            marginal.mu,
            marginal.mu1,
            marginal.mu2,
            marginal.mu3,
            marginal.mu4,
            marginal.mu5,
        ];
        let marginal_coordinates = [primary.q; 3];
        for (m_u, m_v, j, n) in explicit_signatures() {
            if j != 0 {
                continue;
            }
            let weight = dir_u[primary.q].powi(m_u as i32) * dir_v[primary.q].powi(m_v as i32);
            calibration.add_symmetric(
                m_u,
                m_v,
                j,
                &marginal_coordinates[..n],
                -mu[m_u + m_v + n] * weight,
            );
        }

        for (left_index, window) in cells.windows(2).enumerate() {
            let (left, right) = (&window[0], &window[1]);
            let z_star = left.cell.right;
            if !z_star.is_finite() || (right.cell.left - z_star).abs() > 1.0e-12 {
                continue;
            }
            let jump = left.link_span.c3 - right.link_span.c3;
            let column_jumps: Vec<f64> = cell_atoms[left_index]
                .link_leading
                .iter()
                .zip(&cell_atoms[left_index + 1].link_leading)
                .map(|(left_c3, right_c3)| left_c3 - right_c3)
                .collect();
            if jump == 0.0 && column_jumps.iter().all(|value| *value == 0.0) {
                continue;
            }
            let knot = KnotFlux::new(
                left.cell,
                &cell_atoms[left_index],
                &column_jumps,
                jump,
                b,
                scale,
                primary,
                directions,
            );
            for (m_u, m_v, j, n) in explicit_signatures() {
                let order = m_u + m_v + j + n;
                if order < 4 {
                    continue;
                }
                // Only the slope moves the crossing, so an order-four flux lands on
                // slope coordinates alone; at order five one other coordinate may
                // carry the jump of the fourth integrand.
                let mut coordinates = [primary.slope; 3];
                let (slots, len) = explicit_slots(m_u, m_v, j, &coordinates[..n]);
                calibration.add_symmetric(m_u, m_v, j, &coordinates[..n], knot.flux(&slots[..len]));
                if order == 5 && n > 0 {
                    for deviation in 0..r {
                        if deviation == primary.q || deviation == primary.slope {
                            continue;
                        }
                        coordinates[n - 1] = deviation;
                        let (slots, len) = explicit_slots(m_u, m_v, j, &coordinates[..n]);
                        calibration.add_symmetric(
                            m_u,
                            m_v,
                            j,
                            &coordinates[..n],
                            knot.flux(&slots[..len]),
                        );
                    }
                }
            }
        }
        calibration.seal();

        let z_obs = self.z[row];
        let observed = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let observed_atoms = IndexAtoms::new(
            self,
            primary,
            a,
            b,
            observed_base_partials(&observed),
            z_obs,
            a + b * z_obs,
            directions,
        )?;
        let mut index = ExplicitTensors::new(r);
        for (m_u, m_v, j, n) in explicit_signatures() {
            for_each_sorted_tuple(&observed_atoms.active, n, |coordinates| {
                let (slots, len) = explicit_slots(m_u, m_v, j, coordinates);
                let polynomial = observed_atoms.subset_polynomial(&slots[..len], (1 << len) - 1);
                index.add_symmetric(m_u, m_v, j, coordinates, eval_coeff4_at(&polynomial, z_obs));
                Ok(())
            })?;
        }
        index.seal();

        let response_sign = 2.0 * self.y[row] - 1.0;
        let margin = response_sign * eval_coeff4_at(&observed.coeff, z_obs);
        if !margin.is_finite() {
            return Err(format!(
                "non-finite signed margin in standard-normal flex fifth contraction: {margin}"
            ));
        }
        let stack = signed_probit_neglog_unary_stack_fifth(margin, self.weights[row]);
        let mut loss = [0.0_f64; 6];
        for (order, (value, derivative)) in loss.iter_mut().zip(stack).enumerate() {
            *value = response_sign.powi(order as i32) * derivative;
        }

        let f_a = calibration.get(0, 0, 1, 0, 0);
        if !(f_a.is_finite() && f_a > 0.0) {
            return Err(format!(
                "standard-normal flex fifth contraction has calibration slope {f_a}"
            ));
        }
        let intercept = implicit_intercept_totals(&calibration, f_a);
        let index_totals = observed_index_totals(&index, &intercept);
        let fifth = compose_loss(&index_totals, &loss, r);
        Ok((0..r)
            .map(|c| Array2::from_shape_fn((r, r), |(k, l)| fifth[c + k * r + l * r * r]))
            .collect())
    }
}
