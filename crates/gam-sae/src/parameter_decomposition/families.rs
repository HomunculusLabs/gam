//! Parameter-family labels for fixed components (#2951).
//!
//! A decomposition's fixed components `P_c` of one `rows x cols` tensor class are
//! grouped into families. A family holds an affine matrix-valued field over labels
//! `z_c` in `R^d`,
//!
//! ```text
//! Gamma(z) = B_0 + sum_{a=1..d} z_a U_a,      P_c = Gamma(z_c) for every member c,
//! ```
//!
//! and the labels belong to the component, never to the input: they are
//! parameter-family labels, not computational-state coordinates. A family of one
//! component at `d = 0` is the literal component. Sharing merges families into one
//! field, splitting takes a component out into its own literal family, and refining
//! or reducing moves a family's label dimension ([`FamilyPartition`]). This file builds
//! those proposals and their labels. Nothing here compares two proposals: a proposal
//! is decided on decoded code at a declared fidelity by `fit`'s rule.
//!
//! # Labels are principal coordinates
//!
//! For a family with `n` members take the center `B_0 = (1/n) sum_c P_c`, the
//! centered members `X_c = P_c - B_0`, their Frobenius Gram `G_ck = <X_c, X_k>_F`
//! and its eigenpairs `(lambda_a, V_a)` in descending order. With
//!
//! ```text
//! U_a = sum_c V_ca X_c / sqrt(lambda_a),      z_ca = sqrt(lambda_a) V_ca,
//! ```
//!
//! `<U_a, U_b>_F = V_a^T G V_b / sqrt(lambda_a lambda_b) = delta_ab` and
//! `z_ca = (G V_a)_c / sqrt(lambda_a) = <X_c, U_a>_F`. So the labels are isometric: a
//! label error `dz` moves the instance by `||dz||_2` in Frobenius norm, the unit of a
//! coefficient error. `B_0 + sum_{a<=d} z_ca U_a` is the best rank-`d` affine
//! approximation of the members in `sum_c ||.||_F^2` (Eckart-Young on the centered
//! stack), and at `d` equal to the rank of `G` it reproduces every member. No
//! coefficient is regressed here. Fitting coefficients against network responses is
//! `fit`'s conditionally Gaussian block.
//!
//! The label gauge `z -> Q z`, `U -> U Q^T` with `Q` in `O(d)` leaves every instance
//! unchanged, and the eigensolver's representative is returned.
//!
//! # Resolved label dimension
//!
//! Each Gram entry is an inner product over `m = rows cols` products of two rounded
//! differences, at most `m + 2` rounded operations on a path, so its formation error
//! is at most `gamma_{m+2} sum_i |X_ci X_ki| <= gamma_{m+2} ||X_c||_F ||X_k||_F`, and
//! the Frobenius norm of the Gram's formation error is at most `gamma_{m+2} tr(G)` to
//! first order. A label dimension `d` is admitted only when
//! `d <= resolved_eigenvalue_count(lambda, gamma_{m+2} tr(G))`. A direction built from
//! an eigenvalue inside that band is rounding divided by `sqrt(lambda)`, and it never
//! enters a proposal. The Gram route costs `O(n^2 m)` time and `O(n^2 + (d + 2) m)`
//! memory and never stacks the members. It squares their conditioning, so a mode whose
//! singular value lies below the band's square root is not proposed.
//!
//! # Relation to `field`
//!
//! A family is `field`'s instance form `P_c = w_c Gamma'(z_c)` over the degree-one
//! polynomial basis `(1, z_1, ..., z_d)`, with `w_c = 1/n` on the scale gauge and
//! `Gamma' = n Gamma`.

use std::fmt;

use faer::Side;
use gam_linalg::faer_ndarray::{FaerLinalgError, strict_symmetric_eigh};
use gam_linalg::roundoff::{accumulation_growth, resolved_eigenvalue_count};
use ndarray::{Array2, ArrayView2};

/// Why a family proposal was not built.
#[derive(Debug)]
pub enum FamilyError {
    /// The families are not a canonical partition of the components.
    InvalidPartition(String),
    /// The components are not one finite, nonempty tensor class.
    InvalidComponents(String),
    /// A family asks for more label dimensions than its centered Gram resolves.
    UnresolvedDimension {
        first_member: usize,
        dimension: usize,
        resolved: usize,
    },
    /// The dense state exceeds the host in-core budget. `required_bytes` is `None`
    /// when the count overflows `usize`.
    AdmissionRefused {
        required_bytes: Option<usize>,
        budget_bytes: usize,
    },
    /// The eigensolver refused the Gram.
    Eigen(FaerLinalgError),
}

impl fmt::Display for FamilyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPartition(reason) => write!(f, "family partition refused: {reason}"),
            Self::InvalidComponents(reason) => write!(f, "family components refused: {reason}"),
            Self::UnresolvedDimension {
                first_member,
                dimension,
                resolved,
            } => write!(
                f,
                "the family led by component {first_member} asks for {dimension} label \
                 dimensions, while its centered Gram resolves {resolved} above its rounding band"
            ),
            Self::AdmissionRefused {
                required_bytes: Some(required),
                budget_bytes,
            } => write!(
                f,
                "family refused: its dense state needs at least {required} bytes, above the host \
                 in-core budget of {budget_bytes} bytes"
            ),
            Self::AdmissionRefused {
                required_bytes: None,
                budget_bytes,
            } => write!(
                f,
                "family refused: its byte count overflows usize (host in-core budget \
                 {budget_bytes} bytes)"
            ),
            Self::Eigen(error) => write!(f, "family spectrum refused: {error}"),
        }
    }
}

impl std::error::Error for FamilyError {}

/// Refuses a dense state whose byte count overflows or exceeds the host in-core
/// budget.
fn admit_bytes(required_bytes: Option<usize>) -> Result<(), FamilyError> {
    let budget_bytes = crate::manifold::sae_host_in_core_budget_bytes().0;
    match required_bytes {
        Some(required) if required <= budget_bytes => Ok(()),
        required_bytes => Err(FamilyError::AdmissionRefused {
            required_bytes,
            budget_bytes,
        }),
    }
}

/// One family: its members, ascending, and its label dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilySpec {
    pub members: Vec<usize>,
    pub dimension: usize,
}

/// A partition of `components` fixed components into families, held in canonical
/// order (by smallest member).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyPartition {
    components: usize,
    families: Vec<FamilySpec>,
}

impl FamilyPartition {
    /// Refuses a partition with no components, an empty family, members that are not
    /// strictly ascending or not below `components`, a component in two families or
    /// in none, and a label dimension of at least the family's member count.
    pub fn new(components: usize, mut families: Vec<FamilySpec>) -> Result<Self, FamilyError> {
        if components == 0 {
            return Err(FamilyError::InvalidPartition(
                "a partition needs at least one component".to_string(),
            ));
        }
        let mut assigned = vec![false; components];
        for (index, family) in families.iter().enumerate() {
            check_members(&family.members, components)
                .map_err(|reason| FamilyError::InvalidPartition(format!("family {index}: {reason}")))?;
            if family.dimension >= family.members.len() {
                return Err(FamilyError::InvalidPartition(format!(
                    "family {index} has {} members, which span at most {} label dimensions, not {}",
                    family.members.len(),
                    family.members.len() - 1,
                    family.dimension
                )));
            }
            for &member in &family.members {
                if assigned[member] {
                    return Err(FamilyError::InvalidPartition(format!(
                        "component {member} is in two families"
                    )));
                }
                assigned[member] = true;
            }
        }
        if let Some(member) = assigned.iter().position(|&taken| !taken) {
            return Err(FamilyError::InvalidPartition(format!(
                "component {member} is in no family"
            )));
        }
        families.sort_by_key(|family| family.members[0]);
        Ok(Self {
            components,
            families,
        })
    }

    /// Every component in its own family at label dimension zero.
    pub fn literal(components: usize) -> Result<Self, FamilyError> {
        Self::new(
            components,
            (0..components)
                .map(|component| FamilySpec {
                    members: vec![component],
                    dimension: 0,
                })
                .collect(),
        )
    }

    pub fn components(&self) -> usize {
        self.components
    }

    /// The families in canonical order.
    pub fn families(&self) -> &[FamilySpec] {
        &self.families
    }

    fn family(&self, index: usize) -> Result<&FamilySpec, FamilyError> {
        self.families.get(index).ok_or_else(|| {
            FamilyError::InvalidPartition(format!(
                "family {index} does not exist; the partition has {}",
                self.families.len()
            ))
        })
    }

    /// Share: families `first` and `second` become one family at label dimension
    /// `dimension`.
    pub fn share(&self, first: usize, second: usize, dimension: usize) -> Result<Self, FamilyError> {
        if first == second {
            return Err(FamilyError::InvalidPartition(format!(
                "family {first} cannot be shared with itself"
            )));
        }
        let mut members = self.family(first)?.members.clone();
        members.extend_from_slice(&self.family(second)?.members);
        members.sort_unstable();
        let mut families = Vec::with_capacity(self.families.len() - 1);
        for (index, family) in self.families.iter().enumerate() {
            if index != first && index != second {
                families.push(family.clone());
            }
        }
        families.push(FamilySpec { members, dimension });
        Self::new(self.components, families)
    }

    /// Split: `member` leaves family `family` as a literal, and the rest of the family
    /// keeps label dimension `remaining_dimension`.
    pub fn split(
        &self,
        family: usize,
        member: usize,
        remaining_dimension: usize,
    ) -> Result<Self, FamilyError> {
        let source = self.family(family)?;
        if source.members.len() < 2 {
            return Err(FamilyError::InvalidPartition(format!(
                "family {family} has one member and nothing to split off"
            )));
        }
        let position = source.members.binary_search(&member).map_err(|insertion| {
            FamilyError::InvalidPartition(format!(
                "component {member} is not in family {family} (it would sit at position {insertion})"
            ))
        })?;
        let mut families = self.families.clone();
        families[family].members.remove(position);
        families[family].dimension = remaining_dimension;
        families.push(FamilySpec {
            members: vec![member],
            dimension: 0,
        });
        Self::new(self.components, families)
    }

    /// Refine or reduce: family `family` moves to label dimension `dimension`.
    pub fn with_dimension(&self, family: usize, dimension: usize) -> Result<Self, FamilyError> {
        self.family(family)?;
        let mut families = self.families.clone();
        families[family].dimension = dimension;
        Self::new(self.components, families)
    }
}

/// Refuses an empty member list, members that are not strictly ascending, and a
/// member at or above `components`.
fn check_members(members: &[usize], components: usize) -> Result<(), String> {
    let Some(&largest) = members.last() else {
        return Err("a family needs at least one member".to_string());
    };
    if let Some(pair) = members.windows(2).find(|pair| pair[0] >= pair[1]) {
        return Err(format!(
            "members must be strictly ascending, got {} then {}",
            pair[0], pair[1]
        ));
    }
    if largest >= components {
        return Err(format!(
            "member {largest} is outside the {components} components"
        ));
    }
    Ok(())
}

/// The shape of a finite, nonempty tensor class shared by every component.
fn component_class(components: &[ArrayView2<'_, f64>]) -> Result<(usize, usize), FamilyError> {
    let first = components
        .first()
        .ok_or_else(|| FamilyError::InvalidComponents("no components".to_string()))?;
    let (rows, cols) = first.dim();
    if rows == 0 || cols == 0 {
        return Err(FamilyError::InvalidComponents(format!(
            "components must be nonempty tensors, got {rows} x {cols}"
        )));
    }
    for (index, component) in components.iter().enumerate() {
        if component.dim() != (rows, cols) {
            return Err(FamilyError::InvalidComponents(format!(
                "component {index} is {:?}, component 0 is {rows} x {cols}",
                component.dim()
            )));
        }
        if component.iter().any(|value| !value.is_finite()) {
            return Err(FamilyError::InvalidComponents(format!(
                "component {index} has a non-finite entry"
            )));
        }
    }
    Ok((rows, cols))
}

/// The principal affine field of one family's members.
#[derive(Clone, Debug)]
pub struct PrincipalField {
    /// `B_0`, the members' mean.
    pub center: Array2<f64>,
    /// `U_1, ..., U_d`, orthonormal in the Frobenius product.
    pub directions: Vec<Array2<f64>>,
    /// `n x d`: row `i` is the label of the `i`-th member in ascending order.
    pub labels: Array2<f64>,
    /// Every eigenvalue of the centered Gram, descending.
    pub eigenvalues: Vec<f64>,
    /// `gamma_{m+2} tr(G)`, the Gram's formation band.
    pub assembly_band: f64,
    /// The number of label dimensions the spectrum resolves.
    pub resolved_dimension: usize,
}

/// The principal affine field of `members` at label dimension `dimension` (see the
/// module docs).
///
/// Refuses components that are not one finite tensor class, malformed members, a
/// dimension of at least the member count or above the resolved dimension, and a
/// dense state beyond the host in-core budget.
pub fn principal_field(
    components: &[ArrayView2<'_, f64>],
    members: &[usize],
    dimension: usize,
) -> Result<PrincipalField, FamilyError> {
    let (rows, cols) = component_class(components)?;
    check_members(members, components.len()).map_err(FamilyError::InvalidPartition)?;
    let n = members.len();
    if dimension >= n {
        return Err(FamilyError::InvalidPartition(format!(
            "{n} members span at most {} label dimensions, not {dimension}",
            n - 1
        )));
    }
    let entries = rows * cols;
    // The Gram and its eigenvectors, the center and the directions.
    admit_bytes(
        n.checked_mul(n)
            .and_then(|square| square.checked_mul(2))
            .and_then(|squares| {
                dimension
                    .checked_add(1)
                    .and_then(|tensors| tensors.checked_mul(entries))
                    .and_then(|doubles| doubles.checked_add(squares))
            })
            .and_then(|doubles| doubles.checked_mul(std::mem::size_of::<f64>())),
    )?;
    let mut center = Array2::<f64>::zeros((rows, cols));
    for &member in members {
        center += &components[member];
    }
    let count = n as f64;
    center.mapv_inplace(|value| value / count);
    let mut gram = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for k in i..n {
            let value: f64 = components[members[i]]
                .iter()
                .zip(components[members[k]].iter())
                .zip(center.iter())
                .map(|((&left, &right), &mean)| (left - mean) * (right - mean))
                .sum();
            gram[[i, k]] = value;
            gram[[k, i]] = value;
        }
    }
    let trace: f64 = (0..n).map(|i| gram[[i, i]]).sum();
    let assembly_band = accumulation_growth(entries + 2) * trace;
    let (values, vectors) = strict_symmetric_eigh(&gram, Side::Lower).map_err(FamilyError::Eigen)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| values[right].total_cmp(&values[left]));
    let eigenvalues: Vec<f64> = order.iter().map(|&index| values[index]).collect();
    let resolved_dimension = resolved_eigenvalue_count(&eigenvalues, assembly_band);
    if dimension > resolved_dimension {
        return Err(FamilyError::UnresolvedDimension {
            first_member: members[0],
            dimension,
            resolved: resolved_dimension,
        });
    }
    let roots: Vec<f64> = eigenvalues[..dimension]
        .iter()
        .map(|value| value.sqrt())
        .collect();
    let directions = (0..dimension)
        .map(|a| {
            let mut direction = Array2::<f64>::zeros((rows, cols));
            for (i, &member) in members.iter().enumerate() {
                let weight = vectors[[i, order[a]]] / roots[a];
                for ((slot, &value), &mean) in direction
                    .iter_mut()
                    .zip(components[member].iter())
                    .zip(center.iter())
                {
                    *slot += weight * (value - mean);
                }
            }
            direction
        })
        .collect();
    let labels = Array2::from_shape_fn((n, dimension), |(i, a)| roots[a] * vectors[[i, order[a]]]);
    Ok(PrincipalField {
        center,
        directions,
        labels,
        eigenvalues,
        assembly_band,
        resolved_dimension,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::symmetric_spectrum_rounding_band;

    const ROWS: usize = 4;
    const COLS: usize = 5;
    const MEMBERS: usize = 6;

    /// Deterministic fixture entries in `[-1, 1)`.
    fn entries(seed: usize, count: usize) -> Vec<f64> {
        (0..count)
            .map(|k| {
                let x = ((seed * 7919 + k * 104_729 + 1) as f64).sin() * 43_758.545_3;
                (x - x.floor()) * 2.0 - 1.0
            })
            .collect()
    }

    fn matrix(seed: usize, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), entries(seed, rows * cols)).expect("fixture shape")
    }

    fn views(components: &[Array2<f64>]) -> Vec<ArrayView2<'_, f64>> {
        components.iter().map(|component| component.view()).collect()
    }

    /// Six components on a planted two-dimensional affine field `B_0 + z_1 U_1 + z_2 U_2`.
    fn planted_components() -> Vec<Array2<f64>> {
        let center = matrix(1, ROWS, COLS);
        let first = matrix(2, ROWS, COLS);
        let second = matrix(3, ROWS, COLS);
        let labels = entries(4, 2 * MEMBERS);
        (0..MEMBERS)
            .map(|c| &center + &(&first * labels[2 * c]) + &(&second * labels[2 * c + 1]))
            .collect()
    }

    /// Six components drawn as a random initialization: no shared field.
    fn random_components() -> Vec<Array2<f64>> {
        (0..MEMBERS).map(|c| matrix(10 + c, ROWS, COLS)).collect()
    }

    /// `P_c = (2/C) v(t_c) v(t_c)^T` at `t_c = c pi / C`, which sums to `I_2` (P11).
    fn projector_components(instances: usize) -> Vec<Array2<f64>> {
        (0..instances)
            .map(|c| {
                let angle = std::f64::consts::PI * c as f64 / instances as f64;
                let direction = [angle.cos(), angle.sin()];
                Array2::from_shape_fn((2, 2), |(r, k)| 2.0 / instances as f64 * direction[r] * direction[k])
            })
            .collect()
    }

    /// The quantities the tests' bands are built from.
    ///
    /// The premise is the owner band's (`symmetric_spectrum_rounding_band`): the
    /// eigensolver returns an orthonormal `V` and a `Lambda` that exactly diagonalize
    /// `G_x + E`, where `G_x` is the exact Gram of the computed centered members and
    /// `||E||_2 <= beta = p eps ||G||_2 + gamma_{m+2} tr(G)`. Every bound below is first
    /// order in `eps`.
    struct Bands {
        /// The computed centered members `X_c = P_c - B_0`, in member order.
        centered: Vec<Array2<f64>>,
        /// `sqrt(sum_k X_ki^2)` per entry. For a unit `V_a`,
        /// `sum_k |V_ka X_ki| <= reach_i`, so forming `U_a` rounds each entry by at most
        /// `gamma reach_i / sqrt(lambda_a)`.
        reach: Array2<f64>,
        beta: f64,
    }

    fn bands(components: &[Array2<f64>], members: &[usize], field: &PrincipalField) -> Bands {
        let centered: Vec<Array2<f64>> = members
            .iter()
            .map(|&member| &components[member] - &field.center)
            .collect();
        let mut reach = Array2::<f64>::zeros(field.center.dim());
        for x in &centered {
            reach += &x.mapv(|value| value * value);
        }
        reach.mapv_inplace(f64::sqrt);
        Bands {
            centered,
            reach,
            beta: symmetric_spectrum_rounding_band(&field.eigenvalues) + field.assembly_band,
        }
    }

    /// `|U_a| + reach / sqrt(lambda_a)`: the magnitude of the terms that formed `U_a`.
    fn direction_magnitude(field: &PrincipalField, bands: &Bands, a: usize) -> Array2<f64> {
        field.directions[a].mapv(f64::abs) + &(&bands.reach / field.eigenvalues[a].sqrt())
    }

    fn frobenius(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        left.iter().zip(right.iter()).map(|(x, y)| x * y).sum()
    }

    /// `sqrt(sum_c ||X_c - sum_{a<=d} z_ca U_a||_F^2)` and the bound it must meet.
    ///
    /// `sum_a z_ca U_a = sum_k (V_d V_d^T)_ck X_k`, so the squared residual is
    /// `tr((I - P_d) G_x)`. Under the premise that is
    /// `sum_{a>d} lambda_a - tr((I - P_d) E) <= sum_{a>d} lambda_a + (n - d) beta`,
    /// and every unresolved eigenvalue is at most `beta`, so the residual is at most
    /// `sqrt(2 (n - d) beta)`. The reconstruction adds its own formation rounding: a
    /// path forms `U_a` (at most `n + 2` operations), the label (2), the product and the
    /// `d + 1` accumulations, at most `n + 2d + 8` operations against the magnitude
    /// `|X_c| + sum_a |z_ca| (|U_a| + reach / sqrt(lambda_a))`.
    fn reproduction(field: &PrincipalField, bands: &Bands) -> (f64, f64) {
        let n = bands.centered.len();
        let dimension = field.directions.len();
        let growth = accumulation_growth(n + 2 * dimension + 8);
        let mut squared = 0.0;
        let mut formation = 0.0;
        for (i, x) in bands.centered.iter().enumerate() {
            let mut residual = x.clone();
            let mut magnitude = x.mapv(f64::abs);
            for a in 0..dimension {
                residual.scaled_add(-field.labels[[i, a]], &field.directions[a]);
                magnitude.scaled_add(field.labels[[i, a]].abs(), &direction_magnitude(field, bands, a));
            }
            squared += residual.iter().map(|value| value * value).sum::<f64>();
            formation += magnitude.iter().map(|value| (growth * value).powi(2)).sum::<f64>();
        }
        let sum_growth = 1.0 + accumulation_growth(n * field.center.len());
        let bound = sum_growth * ((2.0 * (n - dimension) as f64 * bands.beta).sqrt() + formation.sqrt());
        (squared.sqrt(), bound)
    }

    /// Asserts `<U_a, U_b>_F = delta_ab` and `z_ca = <X_c, U_a>_F` within their bands,
    /// and returns whether the uncentered `<P_c, U_a>_F` is refuted for some label.
    /// That negative control applies only where the center has a component along the
    /// directions: the projector family's center `I/C` is orthogonal to its traceless
    /// directions.
    ///
    /// Under the premise `V_a^T G_x V_b = lambda_a delta_ab - V_a^T E V_b`, so
    /// `|<U_a, U_b> - delta_ab| <= beta / sqrt(lambda_a lambda_b)` and
    /// `|<X_c, U_a> - z_ca| <= beta / sqrt(lambda_a)`. Each inner product adds the
    /// rounding of forming its factors and summing its `m` products, at most
    /// `m + n + 6` operations against the magnitudes of the terms.
    fn assert_isometric_labels(components: &[Array2<f64>], members: &[usize], field: &PrincipalField) -> bool {
        let bands = bands(components, members, field);
        let dimension = field.directions.len();
        let growth = accumulation_growth(field.center.len() + members.len() + 6);
        let magnitudes: Vec<Array2<f64>> = (0..dimension)
            .map(|a| direction_magnitude(field, &bands, a))
            .collect();
        for a in 0..dimension {
            for b in 0..dimension {
                let inner = frobenius(&field.directions[a], &field.directions[b]);
                let target = if a == b { 1.0 } else { 0.0 };
                let bound = bands.beta / (field.eigenvalues[a] * field.eigenvalues[b]).sqrt()
                    + growth * frobenius(&magnitudes[a], &magnitudes[b]);
                assert!(
                    (inner - target).abs() <= bound,
                    "<U_{a}, U_{b}> = {inner}, not {target} within {bound:e}"
                );
            }
        }
        let mut uncentered_refuted = false;
        for (i, (&member, x)) in members.iter().zip(&bands.centered).enumerate() {
            for a in 0..dimension {
                let label = field.labels[[i, a]];
                let centered_inner = frobenius(x, &field.directions[a]);
                let bound = bands.beta / field.eigenvalues[a].sqrt()
                    + growth * frobenius(&x.mapv(f64::abs), &magnitudes[a])
                    + accumulation_growth(2) * label.abs();
                assert!(
                    (label - centered_inner).abs() <= bound,
                    "member {member} label {a}: z = {label} against <X_c, U_a> = {centered_inner} \
                     (bound {bound:e})"
                );
                let uncentered_inner = frobenius(&components[member], &field.directions[a]);
                let uncentered_band = bound + growth * frobenius(&components[member].mapv(f64::abs), &magnitudes[a]);
                uncentered_refuted |= (label - uncentered_inner).abs() > uncentered_band;
            }
        }
        uncentered_refuted
    }

    #[test]
    fn planted_field_labels_are_isometric_and_reproduce_the_members_at_the_resolved_dimension() {
        let components = planted_components();
        let view = views(&components);
        let everyone: Vec<usize> = (0..MEMBERS).collect();
        let field = principal_field(&view, &everyone, 2).expect("the planted field is resolved");
        assert_eq!(
            field.resolved_dimension, 2,
            "eigenvalues {:?} against the formation band {:e}",
            field.eigenvalues, field.assembly_band
        );
        // Guard: a label dimension above the resolved spectrum is refused; d = 2 above is
        // the positive control.
        let refused = principal_field(&view, &everyone, 3);
        assert!(
            matches!(refused, Err(FamilyError::UnresolvedDimension { first_member: 0, dimension: 3, resolved: 2 })),
            "got {refused:?}"
        );
        assert!(
            assert_isometric_labels(&components, &everyone, &field),
            "negative control: the uncentered inner product must disagree with some planted label"
        );
        let (residual, bound) = reproduction(&field, &bands(&components, &everyone, &field));
        assert!(
            residual <= bound,
            "the d = 2 field misses the planted members by {residual:e}, above {bound:e}"
        );
        // Negative control: one label dimension fewer discards a direction the members use.
        let reduced = principal_field(&view, &everyone, 1).expect("d = 1 is resolved");
        let (reduced_residual, reduced_bound) = reproduction(&reduced, &bands(&components, &everyone, &reduced));
        assert!(
            reduced_residual > reduced_bound,
            "the d = 1 field must miss the members: residual {reduced_residual:e}, bound {reduced_bound:e}"
        );
    }

    #[test]
    fn random_init_components_need_every_centered_direction() {
        let components = random_components();
        let view = views(&components);
        let everyone: Vec<usize> = (0..MEMBERS).collect();
        let full = principal_field(&view, &everyone, MEMBERS - 1).expect("every centered direction is resolved");
        assert_eq!(
            full.resolved_dimension,
            MEMBERS - 1,
            "random components resolve every centered direction: eigenvalues {:?}, band {:e}",
            full.eigenvalues,
            full.assembly_band
        );
        assert!(
            assert_isometric_labels(&components, &everyone, &full),
            "negative control: the uncentered inner product must disagree with some random label"
        );
        let (residual, bound) = reproduction(&full, &bands(&components, &everyone, &full));
        assert!(
            residual <= bound,
            "the full field misses the members by {residual:e}, above {bound:e}"
        );
        // A shared field through a random initialization needs n - 1 directions: one
        // fewer misses the members.
        let truncated = principal_field(&view, &everyone, MEMBERS - 2).expect("n - 2 directions are resolved");
        let (truncated_residual, truncated_bound) =
            reproduction(&truncated, &bands(&components, &everyone, &truncated));
        assert!(
            truncated_residual > truncated_bound,
            "n - 2 directions must miss random members: residual {truncated_residual:e}, bound \
             {truncated_bound:e}"
        );
        // n members span n - 1 directions: d = n is refused.
        assert!(matches!(
            principal_field(&view, &everyone, MEMBERS),
            Err(FamilyError::InvalidPartition(..))
        ));
    }

    #[test]
    fn the_projector_family_is_a_two_dimensional_affine_field_p11() {
        let instances = 16;
        let components = projector_components(instances);
        let view = views(&components);
        let everyone: Vec<usize> = (0..instances).collect();
        let field = principal_field(&view, &everyone, 2).expect("the projector family is an affine field");
        assert_eq!(
            field.resolved_dimension, 2,
            "the centered projectors span the two traceless symmetric directions: eigenvalues \
             {:?}, band {:e}",
            field.eigenvalues, field.assembly_band
        );
        assert!(matches!(
            principal_field(&view, &everyone, 3),
            Err(FamilyError::UnresolvedDimension { resolved: 2, .. })
        ));
        assert_isometric_labels(&components, &everyone, &field);
        let (residual, bound) = reproduction(&field, &bands(&components, &everyone, &field));
        assert!(
            residual <= bound,
            "the d = 2 field misses the projectors by {residual:e}, above {bound:e}"
        );
        let line = principal_field(&view, &everyone, 1).expect("d = 1 is resolved");
        let (line_residual, line_bound) = reproduction(&line, &bands(&components, &everyone, &line));
        assert!(
            line_residual > line_bound,
            "one label dimension must miss the projectors: residual {line_residual:e}, bound {line_bound:e}"
        );
    }

    #[test]
    fn principal_field_refuses_malformed_components_and_members() {
        let components = planted_components();
        let view = views(&components);
        // Positive control: a valid family of three members.
        assert!(principal_field(&view, &[0, 2, 4], 1).is_ok());
        for members in [&[][..], &[2, 0][..], &[0, 6][..], &[1, 1][..]] {
            assert!(
                matches!(principal_field(&view, members, 0), Err(FamilyError::InvalidPartition(..))),
                "members {members:?} were admitted"
            );
        }
        let mut misshapen = components.clone();
        misshapen[3] = matrix(40, ROWS, COLS + 1);
        let mut non_finite = components.clone();
        non_finite[1][[0, 0]] = f64::NAN;
        for refused in [views(&misshapen), views(&non_finite), Vec::new()] {
            assert!(
                matches!(principal_field(&refused, &[0, 1], 0), Err(FamilyError::InvalidComponents(..))),
                "malformed components were admitted"
            );
        }
    }

    #[test]
    fn partition_moves_keep_a_canonical_cover_and_refuse_malformed_families() {
        let family = |members: &[usize], dimension: usize| FamilySpec {
            members: members.to_vec(),
            dimension,
        };
        let literal = FamilyPartition::literal(4).expect("literal partition");
        assert_eq!(literal.components(), 4);
        let shared = literal.share(0, 2, 1).expect("share components 0 and 2");
        assert_eq!(
            shared.families(),
            &[family(&[0, 2], 1), family(&[1], 0), family(&[3], 0)]
        );
        assert_eq!(shared.split(0, 2, 0).expect("split component 2 back out"), literal);
        assert!(shared.with_dimension(0, 0).is_ok());
        assert!(shared.with_dimension(0, 2).is_err(), "two members span one label dimension");
        assert!(literal.share(1, 1, 0).is_err(), "a family is not shared with itself");
        assert!(literal.share(1, 4, 0).is_err(), "there is no fifth family");
        assert!(shared.split(1, 1, 0).is_err(), "a literal family has nothing to split off");
        assert!(shared.split(0, 3, 0).is_err(), "component 3 is not in family 0");

        // Positive control: families given out of order are accepted in canonical order.
        let reordered = FamilyPartition::new(3, vec![family(&[2], 0), family(&[0, 1], 1)]).expect("a cover");
        assert_eq!(reordered.families()[0].members, vec![0, 1]);
        for (components, families) in [
            (3, vec![family(&[0, 1], 0), family(&[1, 2], 0)]),
            (3, vec![family(&[0], 0), family(&[1], 0)]),
            (3, vec![family(&[1, 0], 0), family(&[2], 0)]),
            (3, vec![family(&[], 0), family(&[0, 1, 2], 0)]),
            (3, vec![family(&[0, 1, 3], 0)]),
            (3, vec![family(&[0, 1, 2], 3)]),
            (0, vec![]),
        ] {
            assert!(
                FamilyPartition::new(components, families.clone()).is_err(),
                "{components} components {families:?} were admitted"
            );
        }
    }
}
