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
//!    moment kernel, and every link-knot crossing adds its moving-boundary
//!    terms from order two on. The link deviation is `C²` at an interior knot
//!    but only `C⁰` at a support edge, where its tails turn constant, so an edge
//!    crossing carries jumps in `L′` and `L″` as well as `L‴`.
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
pub(super) enum ExplicitSlot {
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

/// The label type of a mask: whether it holds `u` and `v`, and how many free
/// labels. Every label subset of one type holds the same symmetric tensor, so one
/// tensor per type is stored.
#[inline]
fn label_type(mask: usize) -> usize {
    (usize::from(mask & LABEL_U != 0) * 2 + usize::from(mask & LABEL_V != 0)) * 4
        + free_count(mask) as usize
}

/// The mask of a label type whose free labels are the lowest ones.
#[inline]
fn type_representative(label_type: usize) -> usize {
    let mut mask = ((1 << (label_type % 4)) - 1) << 2;
    if label_type / 4 & 2 != 0 {
        mask |= LABEL_U;
    }
    if label_type / 4 & 1 != 0 {
        mask |= LABEL_V;
    }
    mask
}

/// Every non-decreasing assignment of primary indices to the free labels of
/// `mask`, with its flat index: one entry per distinct value of a symmetric tensor.
fn sorted_entries(mask: usize, r: usize) -> Vec<([usize; 5], usize)> {
    let free: Vec<usize> = (0..5)
        .filter(|&label| mask & FREE_LABELS & (1 << label) != 0)
        .collect();
    let n = free.len();
    let mut indices = [0_usize; 3];
    let mut entries = Vec::new();
    loop {
        let mut labels = [0_usize; 5];
        for (position, &label) in free.iter().enumerate() {
            labels[label] = indices[position];
        }
        entries.push((labels, tensor_index(mask, &labels, r)));
        let mut position = n;
        loop {
            if position == 0 {
                return entries;
            }
            position -= 1;
            if indices[position] + 1 < r {
                indices[position] += 1;
                for later in position + 1..n {
                    indices[later] = indices[position];
                }
                break;
            }
        }
    }
}

/// Copy each sorted entry of a symmetric tensor over the free labels of `mask`
/// to every ordering of its indices.
fn symmetrize(values: &mut [f64], mask: usize, entries: &[([usize; 5], usize)], r: usize) {
    let free: Vec<usize> = (0..5)
        .filter(|&label| mask & FREE_LABELS & (1 << label) != 0)
        .collect();
    let n = free.len();
    for (labels, flat) in entries {
        let value = values[*flat];
        for ordering in &ORDERINGS {
            if ordering[..n].iter().any(|&position| position >= n) {
                continue;
            }
            let target = ordering[..n]
                .iter()
                .rev()
                .fold(0, |acc, &position| acc * r + labels[free[position]]);
            values[target] = value;
        }
    }
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
#[derive(Clone)]
struct IndexAtoms {
    q: usize,
    slope: usize,
    base: [[[f64; 4]; 4]; 4],
    deviation: Vec<[[[f64; 4]; 4]; 4]>,
    directional: [[[[f64; 4]; 4]; 4]; 2],
    slope_components: [f64; 2],
    /// The slope and every deviation coordinate whose column does not vanish here.
    active: Vec<usize>,
    /// `∂^S η` for every block content: a direction code (`u` is 2, `v` is 1),
    /// the intercept and slope orders, and the deviation coordinate plus one
    /// (zero for none). Filled once per cell, so a subset is a lookup.
    content: Vec<[f64; 4]>,
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
        let mut atoms = Self {
            q: primary.q,
            slope: primary.slope,
            base,
            deviation,
            directional,
            slope_components: [directions[0][primary.slope], directions[1][primary.slope]],
            active,
            content: Vec::new(),
        };
        let mut content = vec![[0.0; 4]; 4 * 4 * 4 * (r + 1)];
        for direction_code in 0..4 {
            for n_a in 0..4 {
                for n_b in 0..4 - n_a {
                    for deviation in 0..=r {
                        content[Self::content_index(direction_code, n_a, n_b, deviation, r)] = atoms
                            .content_polynomial(direction_code, n_a, n_b, deviation.checked_sub(1));
                    }
                }
            }
        }
        atoms.content = content;
        Ok(atoms)
    }

    /// The same atoms with the content table re-expanded about `z`, so a subset
    /// lookup returns `∂^S η(z + t)` as a cubic in `t`.
    fn about(&self, z: f64) -> Self {
        let mut shifted = self.clone();
        for polynomial in &mut shifted.content {
            *polynomial = cubic_about(polynomial, z);
        }
        shifted
    }

    #[inline]
    fn content_index(
        direction_code: usize,
        n_a: usize,
        n_b: usize,
        deviation: usize,
        r: usize,
    ) -> usize {
        ((direction_code * 4 + n_a) * 4 + n_b) * (r + 1) + deviation
    }

    /// `∂^S η` over the slots of `slots` selected by `mask`, read from the
    /// content table.
    fn subset_polynomial(&self, slots: &[ExplicitSlot], mask: usize) -> [f64; 4] {
        let mut direction_code = 0;
        let mut n_a = 0;
        let mut n_b = 0;
        let mut deviation = 0;
        let mut deviations = 0;
        for (position, &slot) in slots.iter().enumerate() {
            if mask & (1 << position) == 0 {
                continue;
            }
            match slot {
                ExplicitSlot::U => direction_code |= 2,
                ExplicitSlot::V => direction_code |= 1,
                ExplicitSlot::Intercept => n_a += 1,
                ExplicitSlot::Coordinate(p) => {
                    if p == self.q {
                        return [0.0; 4];
                    }
                    if p == self.slope {
                        n_b += 1;
                    } else {
                        deviations += 1;
                        deviation = p + 1;
                    }
                }
            }
        }
        if deviations > 1 || n_a + n_b > 3 {
            return [0.0; 4];
        }
        self.content[Self::content_index(direction_code, n_a, n_b, deviation, self.deviation.len())]
    }

    /// `∂^S η` for one block content. Each direction differentiates through its
    /// slope component or through its deviation components, and at most one
    /// deviation derivative survives.
    fn content_polynomial(
        &self,
        direction_code: usize,
        n_a: usize,
        n_b: usize,
        coordinate_deviation: Option<usize>,
    ) -> [f64; 4] {
        let mut directions = [0_usize; 2];
        let mut direction_count = 0;
        if direction_code & 2 != 0 {
            directions[direction_count] = 0;
            direction_count += 1;
        }
        if direction_code & 1 != 0 {
            directions[direction_count] = 1;
            direction_count += 1;
        }
        let deviations = usize::from(coordinate_deviation.is_some());
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

/// Taylor coefficients in `t = z − z*`. A moving-boundary term of order five
/// reads at most the fourth `z`-derivative of the integrand jump.
type Taylor = [f64; 5];

const FACTORIALS: [f64; 5] = [1.0, 1.0, 2.0, 6.0, 24.0];

/// The product of two truncated series, through `len` coefficients.
fn taylor_product(left: &Taylor, right: &Taylor, len: usize) -> Taylor {
    let mut out = [0.0; 5];
    for (i, &value) in left.iter().enumerate().take(len) {
        if value == 0.0 {
            continue;
        }
        for (j, &other) in right.iter().enumerate().take(len - i) {
            out[i + j] += value * other;
        }
    }
    out
}

/// `p(z + t)` for a cubic `p`, as a cubic in `t`.
fn cubic_about(polynomial: &[f64; 4], z: f64) -> [f64; 4] {
    [
        polynomial[0] + z * (polynomial[1] + z * (polynomial[2] + z * polynomial[3])),
        polynomial[1] + z * (2.0 * polynomial[2] + 3.0 * z * polynomial[3]),
        polynomial[2] + 3.0 * z * polynomial[3],
        polynomial[3],
    ]
}

/// `φ⁽ʲ⁾(x) = (−1)ʲ·He_j(x)·φ(x)` for `j ≤ 7`, from the probabilists' Hermite
/// recurrence `He_{j+1} = x·He_j − j·He_{j−1}`.
fn density_derivatives(x: f64) -> [f64; 8] {
    let mut hermite = [0.0; 8];
    hermite[0] = 1.0;
    hermite[1] = x;
    for j in 1..7 {
        hermite[j + 1] = x * hermite[j] - j as f64 * hermite[j - 1];
    }
    let density = (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt();
    std::array::from_fn(|j| {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        sign * hermite[j] * density
    })
}

/// One cell at a link-knot crossing, re-expanded about `z*`: its atoms,
/// `φ⁽ᵏ⁾(η(z* + t))` for `k ≤ 3`, and `Φ(η(z* + t)) − Φ(η*)`.
struct CrossingSide {
    atoms: IndexAtoms,
    density: [Taylor; 4],
    probability: Taylor,
}

impl CrossingSide {
    fn new(
        cell: exact_kernel::DenestedCubicCell,
        atoms: &IndexAtoms,
        z_star: f64,
        eta_density: &[f64; 8],
    ) -> Self {
        let index = cubic_about(&[cell.c0, cell.c1, cell.c2, cell.c3], z_star);
        // Both cells take the same index value at the crossing, so only the
        // motion away from it enters the expansion.
        let motion: Taylor = [0.0, index[1], index[2], index[3], 0.0];
        let mut powers = [[0.0; 5]; 5];
        powers[0][0] = 1.0;
        for order in 1..5 {
            powers[order] = taylor_product(&powers[order - 1], &motion, 5);
        }
        let mut density = [[0.0; 5]; 4];
        for (k, series) in density.iter_mut().enumerate() {
            for (order, power) in powers.iter().enumerate() {
                let weight = eta_density[k + order] / FACTORIALS[order];
                for (slot, &value) in series.iter_mut().zip(power) {
                    *slot += weight * value;
                }
            }
        }
        let mut probability = [0.0; 5];
        for (order, power) in powers.iter().enumerate().skip(1) {
            let weight = eta_density[order - 1] / FACTORIALS[order];
            for (slot, &value) in probability.iter_mut().zip(power) {
                *slot += weight * value;
            }
        }
        Self {
            atoms: atoms.about(z_star),
            density,
            probability,
        }
    }
}

/// The moving-boundary terms of one link-knot crossing `z*(θ) = (τ − a)/b`.
///
/// Across the crossing the calibration integrand `φ(z)·Φ(η)` switches from the
/// left cell's index to the right cell's. With the crossing's base location
/// `z₀`, `Σ_cells ∫` carries the extra `E(θ) = ∫_{z₀}^{z*(θ)} J dz` of the jump
/// `J = φ(z)·(Φ(η_L) − Φ(η_R))`, and Faà di Bruno gives
///   `D^S E = Σ_{R ⊊ S} Σ_{π ∈ Π(S∖R)} (∂^R ∂_z^{|π|−1} J)(z*)·Π_{B ∈ π} D^B z*`.
/// The index is continuous at every crossing, so `J(z*) = 0` and order one has
/// no boundary term. At an interior knot the index is `C²` and the terms start
/// at order four. At a support edge it is only `C⁰`, and they start at order two.
/// Because `b·z* = τ − a` with `a` and `b` linear in the slots,
/// `D_s z* = −(∂_s a + ∂_s b·z*)/b` and `D^B z* = −(1/b)·Σ_{t ∈ B} ∂_t b·D^{B∖t} z*`
/// for `|B| ≥ 2`. Only the intercept and slope-bearing slots move the crossing,
/// so every slot outside `R` is one of them.
struct LinkCrossing {
    z_star: f64,
    slope: f64,
    slope_index: usize,
    slope_components: [f64; 2],
    /// `φ(z* + t)`.
    density: Taylor,
    left: CrossingSide,
    right: CrossingSide,
}

impl LinkCrossing {
    fn new(
        left_cell: exact_kernel::DenestedCubicCell,
        left_atoms: &IndexAtoms,
        right_cell: exact_kernel::DenestedCubicCell,
        right_atoms: &IndexAtoms,
        b: f64,
    ) -> Self {
        let z_star = left_cell.right;
        let z_density = density_derivatives(z_star);
        let eta_density = density_derivatives(left_cell.eta(z_star));
        Self {
            z_star,
            slope: b,
            slope_index: left_atoms.slope,
            slope_components: left_atoms.slope_components,
            density: std::array::from_fn(|order| z_density[order] / FACTORIALS[order]),
            left: CrossingSide::new(left_cell, left_atoms, z_star, &eta_density),
            right: CrossingSide::new(right_cell, right_atoms, z_star, &eta_density),
        }
    }

    /// `(∂^R ∂_z^j J)(z*)/j!` for `j < len`, over the slots selected by `mask`.
    fn jump(&self, polynomials: &[[[f64; 4]; 32]; 2], mask: usize, len: usize) -> Taylor {
        let mut difference = [0.0; 5];
        if mask == 0 {
            for (slot, (left, right)) in difference
                .iter_mut()
                .zip(self.left.probability.iter().zip(&self.right.probability))
                .take(len)
            {
                *slot = left - right;
            }
        } else {
            for_each_partition(mask, |blocks| {
                for (side, cubics, sign) in [
                    (&self.left, &polynomials[0], 1.0),
                    (&self.right, &polynomials[1], -1.0),
                ] {
                    let mut term = side.density[blocks.len() - 1];
                    for &block in blocks {
                        let cubic = cubics[block];
                        term = taylor_product(
                            &term,
                            &[cubic[0], cubic[1], cubic[2], cubic[3], 0.0],
                            len,
                        );
                    }
                    for (slot, &value) in difference.iter_mut().zip(&term).take(len) {
                        *slot += sign * value;
                    }
                }
            });
        }
        taylor_product(&self.density, &difference, len)
    }

    /// `D^S E` over `slots`.
    fn partial(&self, slots: &[ExplicitSlot]) -> f64 {
        let n = slots.len();
        let mut velocity = [0.0; 5];
        let mut rate = [0.0; 5];
        let mut moving = 0_usize;
        for (position, &slot) in slots.iter().enumerate() {
            let (intercept, slope) = match slot {
                ExplicitSlot::U => (0.0, self.slope_components[0]),
                ExplicitSlot::V => (0.0, self.slope_components[1]),
                ExplicitSlot::Intercept => (1.0, 0.0),
                ExplicitSlot::Coordinate(p) => (0.0, if p == self.slope_index { 1.0 } else { 0.0 }),
            };
            velocity[position] = intercept + slope * self.z_star;
            rate[position] = slope;
            if intercept != 0.0 || slope != 0.0 {
                moving |= 1 << position;
            }
        }
        if n < 2 || moving == 0 {
            return 0.0;
        }
        let full = (1_usize << n) - 1;
        let mut polynomials = [[[0.0; 4]; 32]; 2];
        for mask in 1..full {
            polynomials[0][mask] = self.left.atoms.subset_polynomial(slots, mask);
            polynomials[1][mask] = self.right.atoms.subset_polynomial(slots, mask);
        }
        let mut motion = [0.0; 32];
        for mask in 1..=full {
            if mask & !moving != 0 {
                continue;
            }
            motion[mask] = if mask.count_ones() == 1 {
                -velocity[mask.trailing_zeros() as usize] / self.slope
            } else {
                let mut sum = 0.0;
                for (position, &value) in rate.iter().enumerate().take(n) {
                    if mask & (1 << position) != 0 {
                        sum += value * motion[mask & !(1 << position)];
                    }
                }
                -sum / self.slope
            };
        }
        let mut total = 0.0;
        let mut rest = moving;
        while rest != 0 {
            let jump = self.jump(&polynomials, full & !rest, rest.count_ones() as usize);
            for_each_partition(rest, |blocks| {
                let product: f64 = blocks.iter().map(|&block| motion[block]).product();
                total += FACTORIALS[blocks.len() - 1] * jump[blocks.len() - 1] * product;
            });
            rest = (rest - 1) & moving;
        }
        total
    }
}

/// The link-knot crossings of one row's partition, for adding their
/// moving-boundary terms to explicit calibration partials a lowering
/// accumulates cell by cell.
pub(super) struct CalibrationCrossings {
    crossings: Vec<LinkCrossing>,
}

impl CalibrationCrossings {
    /// `Σ_crossings D^S E` over `slots`.
    pub(super) fn partial(&self, slots: &[ExplicitSlot]) -> f64 {
        self.crossings
            .iter()
            .map(|crossing| crossing.partial(slots))
            .sum()
    }
}

impl BernoulliMarginalSlopeFamily {
    /// Every link-knot crossing of the partition `cells` at intercept `a` and
    /// slope `b`. The direction slots read `directions`.
    pub(super) fn standard_normal_flex_calibration_crossings(
        &self,
        primary: &PrimarySlices,
        a: f64,
        b: f64,
        cells: &[exact_kernel::DenestedPartitionCell],
        directions: [&Array1<f64>; 2],
    ) -> Result<CalibrationCrossings, String> {
        let scale = self.probit_frailty_scale();
        let atoms = |partition_cell: &exact_kernel::DenestedPartitionCell| {
            let cell = partition_cell.cell;
            let z_mid = exact_kernel::interval_probe_point(cell.left, cell.right)?;
            IndexAtoms::new(
                self,
                primary,
                a,
                b,
                cell_base_partials(partition_cell, a, b, scale),
                z_mid,
                a + b * z_mid,
                directions,
            )
        };
        let mut crossings = Vec::new();
        for window in cells.windows(2) {
            let (left, right) = (&window[0], &window[1]);
            if !matches!(left.right_edge, exact_kernel::PartitionEdge::Crossing { .. })
                || right.cell.left != left.cell.right
            {
                continue;
            }
            crossings.push(LinkCrossing::new(
                left.cell,
                &atoms(left)?,
                right.cell,
                &atoms(right)?,
                b,
            ));
        }
        Ok(CalibrationCrossings { crossings })
    }
}

/// Add `Σ_{R ⊆ B} Σ_{π ∈ Π(B∖R)} X[R; |π|]·Π_{P ∈ π} a_P` to the sorted `entries`
/// of `out`: the chain rule of an explicit map `X(a(θ), θ)` over the labels of
/// `mask`, where `R` holds the labels the map differentiates directly and each
/// block `P` enters through the intercept derivative `a_P`, read by its label
/// type. With `solving`, the single-block term `X_a·a_B` is left out so the caller
/// can solve for `a_B`.
fn accumulate_chain_rule(
    out: &mut [f64],
    mask: usize,
    entries: &[([usize; 5], usize)],
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
            for (labels, flat) in entries {
                let mut term = coefficients[tensor_index(raw, labels, r)];
                for &block in blocks {
                    term *= intercept[label_type(block)][tensor_index(block, labels, r)];
                }
                out[*flat] += term;
            }
        });
        if raw == 0 {
            break;
        }
        raw = (raw - 1) & mask;
    }
}

/// `a_B` for every label type, in order of size, from `D^B M = 0`.
fn implicit_intercept_totals(calibration: &ExplicitTensors, f_a: f64) -> Vec<Vec<f64>> {
    let r = calibration.r;
    let mut intercept: Vec<Vec<f64>> = (0..16)
        .map(|label_type| vec![0.0; r.pow(free_count(type_representative(label_type)))])
        .collect();
    let mut types: Vec<usize> = (1..16).collect();
    types.sort_by_key(|&label_type| type_representative(label_type).count_ones());
    for label_type in types {
        let mask = type_representative(label_type);
        let entries = sorted_entries(mask, r);
        let mut accumulated = vec![0.0; intercept[label_type].len()];
        accumulate_chain_rule(&mut accumulated, mask, &entries, calibration, &intercept, true);
        for entry in &entries {
            accumulated[entry.1] /= -f_a;
        }
        symmetrize(&mut accumulated, mask, &entries, r);
        intercept[label_type] = accumulated;
    }
    intercept
}

/// `D^B G` for every label type.
fn observed_index_totals(index: &ExplicitTensors, intercept: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r = index.r;
    (0..16)
        .map(|label_type| {
            let mask = type_representative(label_type);
            let mut out = vec![0.0; r.pow(free_count(mask))];
            if mask != 0 {
                let entries = sorted_entries(mask, r);
                accumulate_chain_rule(&mut out, mask, &entries, index, intercept, false);
                symmetrize(&mut out, mask, &entries, r);
            }
            out
        })
        .collect()
}

/// `D^{|B|}ℓ = Σ_{π ∈ Π(B)} ℓ⁽|π|⁾·Π_{P ∈ π} D^P G` over the labels of `mask`, as a
/// dense symmetric tensor over its free labels.
fn compose_loss(index_totals: &[Vec<f64>], loss: &[f64; 6], mask: usize, r: usize) -> Vec<f64> {
    let mut contraction = vec![0.0; r.pow(free_count(mask))];
    let entries = sorted_entries(mask, r);
    for_each_partition(mask, |blocks| {
        let derivative = loss[blocks.len()];
        for (labels, flat) in &entries {
            let mut term = derivative;
            for &block in blocks {
                term *= index_totals[label_type(block)][tensor_index(block, labels, r)];
            }
            contraction[*flat] += term;
        }
    });
    symmetrize(&mut contraction, mask, &entries, r);
    contraction
}

/// Every total derivative `D^B G` of one standard-normal FLEX row's observed index
/// over the slot labels `{u, v, c, k, l}`, with the loss derivative stack. Any
/// contraction of the row loss over those labels composes from them.
pub(super) struct StandardNormalFlexRowTotals {
    index_totals: Vec<Vec<f64>>,
    loss: [f64; 6],
    r: usize,
}

impl StandardNormalFlexRowTotals {
    /// `D^{|B|}ℓ` over the labels of `mask` (bit 0 is `u`, bit 1 is `v`, bits 2 to
    /// 4 are free primary coordinates), as a dense tensor over its free labels with
    /// the first free label fastest.
    pub(super) fn contraction(&self, mask: usize) -> Vec<f64> {
        compose_loss(&self.index_totals, &self.loss, mask, self.r)
    }
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
        if dir_u.len() == r
            && dir_v.len() == r
            && (dir_u.iter().all(|value| *value == 0.0) || dir_v.iter().all(|value| *value == 0.0))
        {
            return Ok((0..r).map(|_| Array2::<f64>::zeros((r, r))).collect());
        }
        let fifth = self
            .standard_normal_flex_row_totals(row, primary, q, b, beta_h, beta_w, row_ctx, dir_u, dir_v)?
            .contraction(ALL_LABELS);
        Ok((0..r)
            .map(|c| Array2::from_shape_fn((r, r), |(k, l)| fifth[c + k * r + l * r * r]))
            .collect())
    }

    /// The observed-index totals and loss stack of one standard-normal FLEX row over
    /// the slot labels `{u, v, c, k, l}`, for primary-space directions `u` and `v`.
    pub(super) fn standard_normal_flex_row_totals(
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
    ) -> Result<StandardNormalFlexRowTotals, String> {
        let r = primary.total;
        if dir_u.len() != r || dir_v.len() != r {
            return Err(format!(
                "bernoulli standard-normal flex fifth contraction direction lengths ({},{}) != primary dimension {r}",
                dir_u.len(),
                dir_v.len()
            ));
        }
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err("non-finite standard-normal flexible row context in fifth contraction".into());
        }
        let a = row_ctx.intercept;
        let directions = [dir_u, dir_v];
        let calibration = self
            .standard_normal_flex_calibration_partials(primary, q, a, b, beta_h, beta_w, directions)?;
        self.standard_normal_flex_row_totals_from_calibration(
            row,
            primary,
            a,
            b,
            beta_h,
            beta_w,
            directions,
            calibration,
        )
    }

    /// Explicit partials of the calibration map `M(a, θ) = Σ_cells ∫φΦ(η) − μ(q)`
    /// at an arbitrary intercept `a`, the link-knot fluxes included.
    fn standard_normal_flex_calibration_partials(
        &self,
        primary: &PrimarySlices,
        q: f64,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        directions: [&Array1<f64>; 2],
    ) -> Result<ExplicitTensors, String> {
        let r = primary.total;
        let [dir_u, dir_v] = directions;
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
            // Score breaks sit at fixed z. Only a link-knot crossing moves with the
            // row scalars, and adjacent cells share its edge bit for bit.
            if !matches!(left.right_edge, exact_kernel::PartitionEdge::Crossing { .. })
                || right.cell.left != left.cell.right
            {
                continue;
            }
            let crossing = LinkCrossing::new(
                left.cell,
                &cell_atoms[left_index],
                right.cell,
                &cell_atoms[left_index + 1],
                b,
            );
            let mut active: Vec<usize> = cell_atoms[left_index]
                .active
                .iter()
                .chain(&cell_atoms[left_index + 1].active)
                .copied()
                .collect();
            active.sort_unstable();
            active.dedup();
            for (m_u, m_v, j, n) in explicit_signatures() {
                if m_u + m_v + j + n < 2 {
                    continue;
                }
                for_each_sorted_tuple(&active, n, |coordinates| {
                    let (slots, len) = explicit_slots(m_u, m_v, j, coordinates);
                    calibration.add_symmetric(
                        m_u,
                        m_v,
                        j,
                        coordinates,
                        crossing.partial(&slots[..len]),
                    );
                    Ok(())
                })?;
            }
        }
        calibration.seal();
        Ok(calibration)
    }

    /// The observed-index totals and loss stack of one row, from the explicit
    /// calibration partials at the row's intercept root `a`.
    fn standard_normal_flex_row_totals_from_calibration(
        &self,
        row: usize,
        primary: &PrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        directions: [&Array1<f64>; 2],
        calibration: ExplicitTensors,
    ) -> Result<StandardNormalFlexRowTotals, String> {
        let r = primary.total;
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
        Ok(StandardNormalFlexRowTotals {
            index_totals: observed_index_totals(&index, &intercept),
            loss,
            r,
        })
    }
}

#[cfg(test)]
mod explicit_calibration_tests {
    use super::*;

    /// At a fixed row state, with no root solve, each explicit calibration partial
    /// is the intercept or slope derivative of the one below it. Order five against
    /// a Richardson difference of order four exercises the degree-27 cell fold, and
    /// every order from two up exercises the link-knot crossings' moving-boundary
    /// terms, deviation jumps included. The fixture's link support edges are
    /// crossings where the index is only `C⁰`, so the low orders read their jumps
    /// in `L′` and `L″`. The implicit chain rule and the loss composition take no
    /// part.
    #[test]
    fn standard_normal_flex_calibration_partials_differentiate_along_intercept_and_slope() {
        let row = 0usize;
        let (family, states) = super::super::flex_verify_932_tests::standard_normal_flex_fixture();
        let cache = family
            .build_exact_eval_cache(&states)
            .expect("StandardNormal FLEX exact cache");
        let primary = cache.primary.clone();
        let point = family
            .primary_point_from_block_states(row, &states, &primary)
            .expect("StandardNormal FLEX primary point");
        let (q, b, beta_h, beta_w) = family.primary_point_components(&point, &primary);
        let a = BernoulliMarginalSlopeFamily::row_ctx(&cache, row).intercept;
        let r = primary.total;
        let h_range = primary.h.clone().expect("active score-warp range");
        let w_range = primary.w.clone().expect("active link-deviation range");
        let mut dir_u = Array1::<f64>::zeros(r);
        dir_u[primary.q] = 0.55;
        dir_u[primary.slope] = -0.35;
        dir_u[h_range.start] = 0.45;
        dir_u[w_range.start] = -0.40;
        let mut dir_v = Array1::<f64>::zeros(r);
        dir_v[primary.q] = -0.30;
        dir_v[primary.slope] = 0.50;
        dir_v[h_range.end - 1] = 0.25;
        dir_v[w_range.end - 1] = 0.60;
        let partials = |intercept: f64, slope: f64| {
            family
                .standard_normal_flex_calibration_partials(
                    &primary,
                    q,
                    intercept,
                    slope,
                    beta_h.as_ref(),
                    beta_w.as_ref(),
                    [&dir_u, &dir_v],
                )
                .expect("explicit calibration partials")
        };
        let flat = |coordinates: &[usize]| coordinates.iter().rev().fold(0, |acc, &p| acc * r + p);
        let slope = primary.slope;
        let (w, h, w_last) = (w_range.start, h_range.start, w_range.end - 1);
        // (label, m_u, m_v, j, lower coordinates, along the slope instead of the intercept)
        let cases: Vec<(&str, usize, usize, usize, Vec<usize>, bool)> = vec![
            ("a->b (order two)", 0, 0, 1, vec![], true),
            ("aa->a (order three)", 0, 0, 2, vec![], false),
            ("ab->b (order three)", 0, 0, 1, vec![slope], true),
            ("bw->b (order three)", 0, 0, 0, vec![slope, w_last], true),
            ("aaaa->a", 0, 0, 4, vec![], false),
            ("aaaa->b", 0, 0, 4, vec![], true),
            ("aaab->a", 0, 0, 3, vec![slope], false),
            ("aaa->a (order four)", 0, 0, 3, vec![], false),
            ("aab->b (order four)", 0, 0, 2, vec![slope], true),
            ("uvaa->a", 1, 1, 2, vec![], false),
            ("uaab->b", 1, 0, 2, vec![slope], true),
            ("aaaw->a", 0, 0, 3, vec![w], false),
            ("aabw->b", 0, 0, 2, vec![slope, w], true),
            ("aaah->a", 0, 0, 3, vec![h], false),
            ("vaaw->a", 0, 1, 2, vec![w_last], false),
        ];
        let base = partials(a, b);
        let mut worst = 0.0_f64;
        for (label, m_u, m_v, j, lower, along_slope) in &cases {
            let (j_higher, higher) = if *along_slope {
                let mut higher = lower.clone();
                higher.push(slope);
                (*j, higher)
            } else {
                (*j + 1, lower.clone())
            };
            let analytic = base.get(*m_u, *m_v, j_higher, higher.len(), flat(&higher[..]));
            let lower_at = |step: f64| {
                let shifted = if *along_slope {
                    partials(a, b + step)
                } else {
                    partials(a + step, b)
                };
                shifted.get(*m_u, *m_v, *j, lower.len(), flat(&lower[..]))
            };
            let central = |step: f64| (lower_at(step) - lower_at(-step)) / (2.0 * step);
            let witness = (4.0 * central(1.0e-4) - central(2.0e-4)) / 3.0;
            let error = (analytic - witness).abs() / analytic.abs().max(witness.abs()).max(1.0);
            worst = worst.max(error);
            eprintln!(
                "#2901 calibration {label}: analytic={analytic:+.12e} richardson={witness:+.12e} rel={error:.3e}"
            );
        }
        assert!(
            worst <= 1e-7,
            "explicit calibration partials do not differentiate each other: worst relative gap {worst:.3e}"
        );
    }
}
