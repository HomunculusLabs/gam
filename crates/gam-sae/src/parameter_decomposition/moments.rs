//! Mask-moment geometry of an executable parameter decomposition (#2951 P8–P10).
//!
//! # Moments (P8)
//!
//! A decomposition edits its tensors through deletion amounts `t_c = 1 − m_c`, and the edited
//! tensors depend on the mask only through the moment `q = Σ_c t_c·v_c`. Under the anchored lift
//! `Θ(m) = m_Δ·Θ* + B·Σ_c (m_c − m_Δ)·v_c`, this needs `m_Δ = 1`. When the residual mask is a
//! control, `Θ(m) = Θ* − t_Δ·(Θ* − B·Σ_c v_c) − B·q`, so it enters as one more generator, the scalar
//! `t_Δ` in a one-dimensional block.
//!
//! A control's generator lives in one or more [`MomentBlock`]s. Occurrence-level masks give
//! separate blocks, and a control that edits every tied use of a tensor carries a part in each
//! use's block.
//!
//! The mask domain is a declared experiment input with no default (#2951 SPEC tension 2): a
//! product of intervals `m_c ∈ [lower_c, upper_c]` that contain the all-on value 1. With the kept
//! set `S` pinned at all-on, the admissible moments form the zonotope
//! `Z_S = Σ_{c∉S} [(1 − upper_c)·v_c, (1 − lower_c)·v_c]`, which is the image of the box under
//! `m ↦ q`. So the supremum of any discrepancy over masks EQUALS its supremum over `Z_S`. The
//! network stays nonlinear; nothing here linearizes it.
//!
//! # Support function and witness mask (P8)
//!
//! `h(u) = Σ_{c∉S} max((1 − upper_c)·⟨u, v_c⟩, (1 − lower_c)·⟨u, v_c⟩)`. It is attained by the
//! endpoint mask that deletes most where the computed `⟨u, v_c⟩ > 0` and least elsewhere. A
//! control whose pairing lies inside its roundoff band ([`accumulation_band`]) is reported
//! unresolved: both of its endpoints attain the support to within the band, so a derivative taken
//! at the fixed witness (Danskin, P16) is one-sided there.
//!
//! # Planar boundary (P9)
//!
//! A planar zonotope has at most `2G` vertices, where `G` is the number of distinct generator
//! directions, and sorting the generators by angle finds them in `O(C log C)`. Directions are
//! compared through an exact determinant sign. The boundary's combinatorics, including which
//! generators are collinear and share one edge, are therefore exact for the stored `f64`
//! generators; vertex positions carry a derived roundoff band.
//!
//! # Fragmentation (P10)
//!
//! A positive collinear refinement preserves the set, `Σ_i [0, α_i·v] = [0, v]` for `α_i > 0`, so
//! [`MaskMomentSystem::merge_positive_collinear`] folds each such group into one control. Splitting a
//! cancelling pair `±a` into `n` pieces with independent uniform masks drives the mean square of
//! the error to `a²/(6n)`, while `sup|E_n| = |a|`. The expectation under a mask law, an observed
//! worst case and a certified supremum are therefore three different numbers, and they live in
//! different types: [`UniformMaskLawMoments`], a caller's own witness, and
//! [`MaskMomentSystem::absolute_supremum`] with [`ClaimedBoundCheck`].
//!
//! # What the reported numbers are
//!
//! Per the #2946 fr-census overclaim audit (comment 5716123817), each value states what it is.
//! - A support value is the computed sum with a derived band that contains the exact support of
//!   the stored generators.
//! - A claimed bound is refuted only by a witness mask exceeding it beyond that band, and it is
//!   certified only when the supremum plus the band is at or below it.
//! - Mask-law moments are expectations under a declared law. They are never bounds.

use std::cmp::Ordering;

use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_band, accumulation_growth};
use ndarray::Array1;

/// One moment coordinate space: the coefficient space of one edited tensor use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MomentBlock {
    pub dimension: usize,
}

/// One coordinate of the moment space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MomentCoordinate {
    pub block: usize,
    pub index: usize,
}

/// A control's generator in one block.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorPart {
    pub block: usize,
    pub vector: Array1<f64>,
}

/// A moment, or a direction paired with moments, laid out block by block.
#[derive(Clone, Debug, PartialEq)]
pub struct MomentVector {
    pub blocks: Vec<Array1<f64>>,
}

impl MomentVector {
    /// The value at one coordinate, if the vector has it.
    pub fn component(&self, coordinate: MomentCoordinate) -> Option<f64> {
        self.blocks
            .get(coordinate.block)
            .and_then(|block| block.get(coordinate.index))
            .copied()
    }
}

/// Where a witness puts one control's mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WitnessEndpoint {
    /// Pinned at all-on, `m_c = 1`, as a member of the kept set.
    Kept,
    /// `m_c = lower_c`, the most deletion the declared domain admits.
    Lower,
    /// `m_c = upper_c`, the least deletion the declared domain admits.
    Upper,
}

impl WitnessEndpoint {
    fn switched(self) -> Self {
        match self {
            WitnessEndpoint::Kept => WitnessEndpoint::Kept,
            WitnessEndpoint::Lower => WitnessEndpoint::Upper,
            WitnessEndpoint::Upper => WitnessEndpoint::Lower,
        }
    }
}

/// A malformed moment system, domain, mask, direction or claim.
#[derive(Clone, Debug, PartialEq)]
pub enum MomentGeometryError {
    GeneratorBlockOutOfRange { control: usize, block: usize, blocks: usize },
    RepeatedGeneratorBlock { control: usize, block: usize },
    GeneratorDimension { control: usize, block: usize, expected: usize, found: usize },
    NonFiniteGenerator { control: usize, block: usize },
    MaskDomainExcludesAllOn { control: usize, lower: f64, upper: f64 },
    ControlCount { expected: usize, found: usize },
    MaskOutsideDomain { control: usize, value: f64, lower: f64, upper: f64 },
    DirectionBlockCount { expected: usize, found: usize },
    DirectionDimension { block: usize, expected: usize, found: usize },
    NonFiniteDirection { block: usize },
    CoordinateOutOfRange { coordinate: MomentCoordinate },
    RepeatedPlaneCoordinate { coordinate: MomentCoordinate },
    NonFiniteClaim { claimed: f64 },
}

/// The declared product mask domain `m_c ∈ [lower_c, upper_c]`, one interval per control.
///
/// It is an experiment declaration, not a search box, and has no default (#2951 SPEC tension 2).
#[derive(Clone, Debug, PartialEq)]
pub struct MaskDomain {
    intervals: Vec<(f64, f64)>,
}

impl MaskDomain {
    /// Every interval must be finite and contain the all-on value 1, so that the kept set is
    /// admissible.
    pub fn new(intervals: Vec<(f64, f64)>) -> Result<Self, MomentGeometryError> {
        for (control, &(lower, upper)) in intervals.iter().enumerate() {
            if !(lower.is_finite() && upper.is_finite() && lower <= 1.0 && upper >= 1.0) {
                return Err(MomentGeometryError::MaskDomainExcludesAllOn {
                    control,
                    lower,
                    upper,
                });
            }
        }
        Ok(Self { intervals })
    }

    pub fn control_count(&self) -> usize {
        self.intervals.len()
    }

    pub fn interval(&self, control: usize) -> Option<(f64, f64)> {
        self.intervals.get(control).copied()
    }

    /// The mask values a witness names.
    pub fn mask_at(&self, witness: &[WitnessEndpoint]) -> Result<Vec<f64>, MomentGeometryError> {
        if witness.len() != self.intervals.len() {
            return Err(MomentGeometryError::ControlCount {
                expected: self.intervals.len(),
                found: witness.len(),
            });
        }
        Ok(witness
            .iter()
            .zip(&self.intervals)
            .map(|(endpoint, &(lower, upper))| match endpoint {
                WitnessEndpoint::Kept => 1.0,
                WitnessEndpoint::Lower => lower,
                WitnessEndpoint::Upper => upper,
            })
            .collect())
    }

    /// The least and the most deletion, `(1 − upper, 1 − lower)`.
    fn deletion_range(&self, control: usize) -> (f64, f64) {
        let (lower, upper) = self.intervals[control];
        (1.0 - upper, 1.0 - lower)
    }
}

/// The support `h(u)` of the admissible zonotope, with a mask attaining it.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportEvaluation {
    /// The computed support.
    pub value: f64,
    /// Radius around `value` containing the exact support of the stored generators. It covers the
    /// pairings' roundoff bands, the rounding of each term and of their sum, and, for every
    /// unresolved control, the most its endpoint choice can lose.
    pub band: f64,
    /// A mask attaining `value`: most deletion where the computed pairing is positive, least
    /// elsewhere.
    pub witness: Vec<WitnessEndpoint>,
    /// Free controls whose pairing lies inside its roundoff band, so either endpoint attains the
    /// support to within `band`.
    pub unresolved: Vec<usize>,
}

/// One edge of a planar zonotope boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarEdge {
    /// Free controls whose planar generators share this edge's direction exactly, in increasing
    /// order. A positive collinear refinement lands in one edge and leaves the set unchanged
    /// (#2951 P10).
    pub controls: Vec<usize>,
    /// The edge vector: the sum of the members' oriented generator segments.
    pub step: [f64; 2],
}

/// The counterclockwise boundary of a planar zonotope (#2951 P9).
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarBoundary {
    /// Counterclockwise vertices: one for a point, two for a segment, otherwise `2·edges.len()`.
    pub vertices: Vec<[f64; 2]>,
    /// Edges in increasing angle from the first vertex; the boundary walks them forward and then
    /// back.
    pub edges: Vec<PlanarEdge>,
    /// Free controls with a zero planar generator or a degenerate interval: both endpoints give
    /// the same point.
    pub inert: Vec<usize>,
    /// Bound on each vertex coordinate's roundoff.
    pub band: f64,
    start: Vec<WitnessEndpoint>,
}

impl PlanarBoundary {
    /// The endpoint mask whose moment is `vertices[vertex]`.
    pub fn witness(&self, vertex: usize) -> Option<Vec<WitnessEndpoint>> {
        (vertex < self.vertices.len()).then(|| self.switched_witness(vertex))
    }

    fn switched_witness(&self, vertex: usize) -> Vec<WitnessEndpoint> {
        let groups = self.edges.len();
        let switched = if vertex <= groups {
            0..vertex
        } else {
            vertex - groups..groups
        };
        let mut witness = self.start.clone();
        for edge in &self.edges[switched] {
            for &control in &edge.controls {
                witness[control] = witness[control].switched();
            }
        }
        witness
    }
}

/// Moments of `⟨u, q⟩` when each free control's mask is drawn independently and uniformly on its
/// declared interval.
///
/// It is an expectation under a declared mask law, not a bound. Splitting a cancelling pair `±a`
/// into `n` pieces drives the mean square to `a²/(6n)`, while the supremum of `|⟨u, q⟩|` stays `|a|`
/// (#2951 P10). Nothing converts it into a supremum or a [`ClaimedBoundCheck`].
#[derive(Clone, Debug, PartialEq)]
pub struct UniformMaskLawMoments {
    pub mean: f64,
    pub mean_band: f64,
    pub variance: f64,
    pub variance_band: f64,
    pub mean_square: f64,
    pub mean_square_band: f64,
}

/// A claimed bound on `sup |⟨u, q⟩|` over the admissible masks, checked against the support
/// function (#2951 A6).
#[derive(Clone, Debug, PartialEq)]
pub enum ClaimedBoundCheck {
    /// The witness mask attains more than the claim, beyond the roundoff band: a counterexample.
    Refuted {
        attained: f64,
        band: f64,
        witness: Vec<WitnessEndpoint>,
    },
    /// The supremum plus its band is at most the claim.
    Certified { supremum: f64, band: f64 },
    /// The claim lies inside the supremum's band.
    Unresolved { supremum: f64, band: f64 },
}

/// Controls merged by positive collinear refinement (#2951 P10).
#[derive(Clone, Debug, PartialEq)]
pub struct CollinearMerge {
    /// One control per group; its generator is the sum of the group's generators.
    pub system: MaskMomentSystem,
    /// Each group's shared declared interval.
    pub domain: MaskDomain,
    /// `groups[g]` lists the original controls merged into control `g`, in increasing order. Groups
    /// are ordered by their first member.
    pub groups: Vec<Vec<usize>>,
    /// Bound on each merged generator coordinate's roundoff.
    pub band: f64,
}

/// Controls, their generators and the moment blocks those generators live in (#2951 P8).
#[derive(Clone, Debug, PartialEq)]
pub struct MaskMomentSystem {
    blocks: Vec<MomentBlock>,
    generators: Vec<Vec<GeneratorPart>>,
}

impl MaskMomentSystem {
    /// `generators[c]` lists control `c`'s parts, at most one per block.
    pub fn new(
        blocks: Vec<MomentBlock>,
        generators: Vec<Vec<GeneratorPart>>,
    ) -> Result<Self, MomentGeometryError> {
        for (control, parts) in generators.iter().enumerate() {
            let mut seen = vec![false; blocks.len()];
            for part in parts {
                let block = blocks.get(part.block).ok_or(MomentGeometryError::GeneratorBlockOutOfRange {
                    control,
                    block: part.block,
                    blocks: blocks.len(),
                })?;
                if seen[part.block] {
                    return Err(MomentGeometryError::RepeatedGeneratorBlock {
                        control,
                        block: part.block,
                    });
                }
                seen[part.block] = true;
                if part.vector.len() != block.dimension {
                    return Err(MomentGeometryError::GeneratorDimension {
                        control,
                        block: part.block,
                        expected: block.dimension,
                        found: part.vector.len(),
                    });
                }
                if part.vector.iter().any(|value| !value.is_finite()) {
                    return Err(MomentGeometryError::NonFiniteGenerator {
                        control,
                        block: part.block,
                    });
                }
            }
        }
        Ok(Self { blocks, generators })
    }

    pub fn blocks(&self) -> &[MomentBlock] {
        &self.blocks
    }

    pub fn control_count(&self) -> usize {
        self.generators.len()
    }

    pub fn generator(&self, control: usize) -> Option<&[GeneratorPart]> {
        self.generators.get(control).map(Vec::as_slice)
    }

    /// The moment `q(m) = Σ_c (1 − m_c)·v_c` of a mask inside the declared domain.
    pub fn moment(&self, domain: &MaskDomain, mask: &[f64]) -> Result<MomentVector, MomentGeometryError> {
        self.check_controls(domain, mask.len())?;
        let mut blocks: Vec<Array1<f64>> = self
            .blocks
            .iter()
            .map(|block| Array1::zeros(block.dimension))
            .collect();
        for (control, (&value, parts)) in mask.iter().zip(&self.generators).enumerate() {
            let (lower, upper) = domain.intervals[control];
            if !(lower <= value && value <= upper) {
                return Err(MomentGeometryError::MaskOutsideDomain {
                    control,
                    value,
                    lower,
                    upper,
                });
            }
            let deletion = 1.0 - value;
            for part in parts {
                blocks[part.block].scaled_add(deletion, &part.vector);
            }
        }
        Ok(MomentVector { blocks })
    }

    /// The support function of `Z_S` in `direction`, with its witness mask (#2951 P8).
    ///
    /// `kept[c]` pins control `c` at all-on.
    pub fn support(
        &self,
        domain: &MaskDomain,
        kept: &[bool],
        direction: &MomentVector,
    ) -> Result<SupportEvaluation, MomentGeometryError> {
        self.check_controls(domain, kept.len())?;
        self.check_direction(direction)?;
        let mut value = 0.0;
        let mut absolute_terms = 0.0;
        let mut band = 0.0;
        let mut free = 0usize;
        let mut witness = Vec::with_capacity(self.generators.len());
        let mut unresolved = Vec::new();
        for control in 0..self.generators.len() {
            if kept[control] {
                witness.push(WitnessEndpoint::Kept);
                continue;
            }
            let (pairing, pairing_band) = self.pairing(control, direction);
            let (least, most) = domain.deletion_range(control);
            let (endpoint, deletion) = if pairing > 0.0 {
                (WitnessEndpoint::Lower, most)
            } else {
                (WitnessEndpoint::Upper, least)
            };
            let term = deletion * pairing;
            value += term;
            absolute_terms += term.abs();
            free += 1;
            // The deletion amount and the product each round once; the pairing's own error scales by
            // the deletion.
            band += 2.0 * UNIT_ROUNDOFF * term.abs() + deletion.abs() * pairing_band;
            if pairing.abs() <= pairing_band {
                unresolved.push(control);
                // The exact pairing is at most twice the band in magnitude, so the other endpoint
                // gains at most that much over the interval's length.
                band += 2.0 * (most - least) * pairing_band;
            }
            witness.push(endpoint);
        }
        // The terms were formed before summing, so the sum commits `free − 1` rounded additions.
        band += accumulation_growth(free.saturating_sub(1)) * absolute_terms;
        Ok(SupportEvaluation {
            value,
            band,
            witness,
            unresolved,
        })
    }

    /// `sup |⟨u, q⟩|` over the admissible zonotope, `max(h(u), h(−u))`, with the larger side's
    /// witness.
    ///
    /// The band is the larger of the two sides' bands. The exact maximum lies within it: the larger
    /// computed side's lower end bounds it from below, and neither side's upper end exceeds the
    /// larger value plus the larger band.
    pub fn absolute_supremum(
        &self,
        domain: &MaskDomain,
        kept: &[bool],
        direction: &MomentVector,
    ) -> Result<SupportEvaluation, MomentGeometryError> {
        let upward = self.support(domain, kept, direction)?;
        let negated = MomentVector {
            blocks: direction.blocks.iter().map(|block| -block).collect(),
        };
        let downward = self.support(domain, kept, &negated)?;
        let band = upward.band.max(downward.band);
        let mut larger = if downward.value > upward.value {
            downward
        } else {
            upward
        };
        larger.band = band;
        Ok(larger)
    }

    /// Checks a claimed bound on `sup |⟨u, q⟩|` against the certified supremum (#2951 A6).
    ///
    /// When the supremum exceeds the claim beyond the band, the witness mask refutes it. That
    /// includes an expectation under a mask law presented as a bound.
    pub fn check_claimed_absolute_bound(
        &self,
        domain: &MaskDomain,
        kept: &[bool],
        direction: &MomentVector,
        claimed: f64,
    ) -> Result<ClaimedBoundCheck, MomentGeometryError> {
        if !claimed.is_finite() {
            return Err(MomentGeometryError::NonFiniteClaim { claimed });
        }
        let supremum = self.absolute_supremum(domain, kept, direction)?;
        Ok(if supremum.value - supremum.band > claimed {
            ClaimedBoundCheck::Refuted {
                attained: supremum.value,
                band: supremum.band,
                witness: supremum.witness,
            }
        } else if supremum.value + supremum.band <= claimed {
            ClaimedBoundCheck::Certified {
                supremum: supremum.value,
                band: supremum.band,
            }
        } else {
            ClaimedBoundCheck::Unresolved {
                supremum: supremum.value,
                band: supremum.band,
            }
        })
    }

    /// Mean, variance and mean square of `⟨u, q⟩` under independent uniform masks on each free
    /// control's declared interval.
    ///
    /// A deletion uniform on `[least, most]` has mean `(least + most)/2` and variance
    /// `(most − least)²/12`, so `E⟨u, q⟩ = Σ_c (least_c + most_c)/2·⟨u, v_c⟩` and
    /// `Var⟨u, q⟩ = Σ_c (most_c − least_c)²/12·⟨u, v_c⟩²`.
    pub fn uniform_mask_law_moments(
        &self,
        domain: &MaskDomain,
        kept: &[bool],
        direction: &MomentVector,
    ) -> Result<UniformMaskLawMoments, MomentGeometryError> {
        self.check_controls(domain, kept.len())?;
        self.check_direction(direction)?;
        let mut mean = 0.0;
        let mut variance = 0.0;
        let mut mean_absolute = 0.0;
        let mut variance_absolute = 0.0;
        let mut mean_band = 0.0;
        let mut variance_band = 0.0;
        let mut free = 0usize;
        for control in (0..self.generators.len()).filter(|&control| !kept[control]) {
            let (pairing, pairing_band) = self.pairing(control, direction);
            let (least, most) = domain.deletion_range(control);
            let centre = 0.5 * (least + most);
            let width = most - least;
            let mean_term = centre * pairing;
            let variance_term = width * width * pairing * pairing / 12.0;
            mean += mean_term;
            variance += variance_term;
            mean_absolute += mean_term.abs();
            variance_absolute += variance_term.abs();
            mean_band += centre.abs() * pairing_band;
            variance_band += width * width * (2.0 * pairing.abs() + pairing_band) * pairing_band / 12.0;
            free += 1;
        }
        // A mean term costs three roundings (the two endpoints, their sum, the product) and a
        // variance term six (the endpoints, the width, its square, two products, the division),
        // before the `free − 1` additions.
        mean_band += accumulation_growth(free + 2) * mean_absolute;
        variance_band += accumulation_growth(free + 5) * variance_absolute;
        let mean_square = mean * mean + variance;
        let mean_square_band = (2.0 * mean.abs() + mean_band) * mean_band
            + variance_band
            + accumulation_growth(2) * (mean * mean + variance.abs());
        Ok(UniformMaskLawMoments {
            mean,
            mean_band,
            variance,
            variance_band,
            mean_square,
            mean_square_band,
        })
    }

    /// The boundary of the admissible zonotope's projection onto two coordinates (#2951 P9).
    ///
    /// When every free generator lies in the plane, this is the zonotope itself.
    pub fn planar_boundary(
        &self,
        domain: &MaskDomain,
        kept: &[bool],
        plane: [MomentCoordinate; 2],
    ) -> Result<PlanarBoundary, MomentGeometryError> {
        self.check_controls(domain, kept.len())?;
        for coordinate in plane {
            self.check_coordinate(coordinate)?;
        }
        if plane[0] == plane[1] {
            return Err(MomentGeometryError::RepeatedPlaneCoordinate {
                coordinate: plane[0],
            });
        }
        let projections: Vec<[f64; 2]> = (0..self.generators.len())
            .map(|control| {
                [
                    self.coordinate_value(control, plane[0]),
                    self.coordinate_value(control, plane[1]),
                ]
            })
            .collect();
        Ok(planar_walk(domain, kept, &projections))
    }

    /// Merges every group of positively collinear generators that share one declared interval into
    /// a single control whose generator is their sum (#2951 P10).
    ///
    /// For `α_i > 0`, `Σ_i [(1 − upper)·α_i·v, (1 − lower)·α_i·v] = [(1 − upper)·Σ_i α_i·v,
    /// (1 − lower)·Σ_i α_i·v]`, so the admissible zonotope is unchanged under every kept set that
    /// keeps whole groups. Collinearity is decided exactly on the stored generators.
    ///
    /// Candidates are sorted by a key they share whenever they are exactly collinear: the interval,
    /// the nonzero pattern, the first nonzero entry's sign, and the entries divided by that first
    /// entry. The division is correctly rounded, so equal exact quotients round to equal keys. Each
    /// run of equal keys is then confirmed with exact determinant signs, since unequal quotients can
    /// also round to one key.
    pub fn merge_positive_collinear(&self, domain: &MaskDomain) -> Result<CollinearMerge, MomentGeometryError> {
        self.check_controls(domain, domain.control_count())?;
        let entries: Vec<Vec<(usize, usize, f64)>> = self
            .generators
            .iter()
            .map(|parts| {
                let mut flat: Vec<(usize, usize, f64)> = parts
                    .iter()
                    .flat_map(|part| {
                        part.vector
                            .iter()
                            .enumerate()
                            .filter(|entry| *entry.1 != 0.0)
                            .map(move |(index, &value)| (part.block, index, value))
                    })
                    .collect();
                flat.sort_by_key(|entry| (entry.0, entry.1));
                flat
            })
            .collect();
        let key_order = |left: usize, right: usize| {
            let (left_lower, left_upper) = domain.intervals[left];
            let (right_lower, right_upper) = domain.intervals[right];
            left_lower
                .total_cmp(&right_lower)
                .then(left_upper.total_cmp(&right_upper))
                .then_with(|| collinearity_key_order(&entries[left], &entries[right]))
        };
        let mut order: Vec<usize> = (0..self.generators.len()).collect();
        order.sort_by(|&left, &right| key_order(left, right));
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut run_start = 0;
        for (position, &control) in order.iter().enumerate() {
            if position == 0 || key_order(order[position - 1], control) != Ordering::Equal {
                run_start = groups.len();
            }
            let joined = groups[run_start..]
                .iter_mut()
                .find(|group| positively_collinear(&entries[group[0]], &entries[control]));
            match joined {
                Some(group) => group.push(control),
                None => groups.push(vec![control]),
            }
        }
        for group in &mut groups {
            group.sort_unstable();
        }
        groups.sort_by_key(|group| group[0]);
        let mut band = 0.0_f64;
        let mut generators = Vec::with_capacity(groups.len());
        for group in &groups {
            let mut parts = Vec::new();
            for (block, spec) in self.blocks.iter().enumerate() {
                let members: Vec<&GeneratorPart> = group
                    .iter()
                    .filter_map(|&control| self.generators[control].iter().find(|part| part.block == block))
                    .collect();
                if members.is_empty() {
                    continue;
                }
                let mut vector = Array1::<f64>::zeros(spec.dimension);
                let mut absolute = Array1::<f64>::zeros(spec.dimension);
                for member in &members {
                    vector += &member.vector;
                    absolute += &member.vector.mapv(f64::abs);
                }
                // Adding the first member to zero is exact; each further member commits one rounding.
                let growth = accumulation_growth(members.len() - 1);
                band = band.max(growth * absolute.iter().fold(0.0_f64, |acc, &value| acc.max(value)));
                parts.push(GeneratorPart { block, vector });
            }
            generators.push(parts);
        }
        let merged_domain = MaskDomain {
            intervals: groups.iter().map(|group| domain.intervals[group[0]]).collect(),
        };
        Ok(CollinearMerge {
            system: MaskMomentSystem::new(self.blocks.clone(), generators)?,
            domain: merged_domain,
            groups,
            band,
        })
    }

    fn check_controls(&self, domain: &MaskDomain, found: usize) -> Result<(), MomentGeometryError> {
        let expected = self.generators.len();
        for found in [domain.control_count(), found] {
            if found != expected {
                return Err(MomentGeometryError::ControlCount { expected, found });
            }
        }
        Ok(())
    }

    fn check_direction(&self, direction: &MomentVector) -> Result<(), MomentGeometryError> {
        if direction.blocks.len() != self.blocks.len() {
            return Err(MomentGeometryError::DirectionBlockCount {
                expected: self.blocks.len(),
                found: direction.blocks.len(),
            });
        }
        for (block, (spec, covector)) in self.blocks.iter().zip(&direction.blocks).enumerate() {
            if covector.len() != spec.dimension {
                return Err(MomentGeometryError::DirectionDimension {
                    block,
                    expected: spec.dimension,
                    found: covector.len(),
                });
            }
            if covector.iter().any(|value| !value.is_finite()) {
                return Err(MomentGeometryError::NonFiniteDirection { block });
            }
        }
        Ok(())
    }

    fn check_coordinate(&self, coordinate: MomentCoordinate) -> Result<(), MomentGeometryError> {
        match self.blocks.get(coordinate.block) {
            Some(block) if coordinate.index < block.dimension => Ok(()),
            _ => Err(MomentGeometryError::CoordinateOutOfRange { coordinate }),
        }
    }

    fn coordinate_value(&self, control: usize, coordinate: MomentCoordinate) -> f64 {
        self.generators[control]
            .iter()
            .find(|part| part.block == coordinate.block)
            .map_or(0.0, |part| part.vector[coordinate.index])
    }

    /// `⟨u, v_c⟩` and its roundoff band: an inner product over every part's entries.
    fn pairing(&self, control: usize, direction: &MomentVector) -> (f64, f64) {
        let mut product = 0.0;
        let mut absolute = 0.0;
        let mut terms = 0;
        for part in &self.generators[control] {
            for (&generator, &covector) in part.vector.iter().zip(direction.blocks[part.block].iter()) {
                let term = generator * covector;
                product += term;
                absolute += term.abs();
                terms += 1;
            }
        }
        (product, accumulation_band(terms, absolute))
    }
}

/// The planar zonotope boundary from each control's projected generator.
///
/// Every nonzero generator is oriented into the half-open upper half-plane (`y > 0`, or `y = 0` and
/// `x > 0`), and a flipped generator starts at its far endpoint. The oriented generators are sorted
/// by angle with the exact determinant sign, and consecutive equal directions merge into one edge.
/// The walk starts at the lowest-leftmost vertex, adds the edges in angular order, then subtracts
/// them.
fn planar_walk(domain: &MaskDomain, kept: &[bool], projections: &[[f64; 2]]) -> PlanarBoundary {
    let mut start = Vec::with_capacity(projections.len());
    let mut oriented: Vec<(usize, [f64; 2])> = Vec::new();
    let mut inert = Vec::new();
    let mut origin = [0.0, 0.0];
    let mut absolute = 0.0;
    let mut free = 0usize;
    for (control, &[x, y]) in projections.iter().enumerate() {
        if kept[control] {
            start.push(WitnessEndpoint::Kept);
            continue;
        }
        free += 1;
        let (least, most) = domain.deletion_range(control);
        let width = most - least;
        let upward = y > 0.0 || (y == 0.0 && x > 0.0);
        let downward = y < 0.0 || (y == 0.0 && x < 0.0);
        let moves = width > 0.0 && (upward || downward);
        let (endpoint, deletion) = if moves && downward {
            (WitnessEndpoint::Lower, most)
        } else {
            (WitnessEndpoint::Upper, least)
        };
        origin[0] += deletion * x;
        origin[1] += deletion * y;
        absolute += (deletion.abs() + 2.0 * width) * (x.abs() + y.abs());
        if moves {
            let sign = if upward { 1.0 } else { -1.0 };
            oriented.push((control, [sign * x, sign * y]));
        } else {
            inert.push(control);
        }
        start.push(endpoint);
    }
    // Upper-half-plane vectors have angles in [0, π), so `a` precedes `b` exactly when
    // `det(a, b) > 0`. The sort is stable, so each edge lists its controls in increasing order.
    oriented.sort_by(|left, right| determinant_sign(left.1[0], left.1[1], right.1[0], right.1[1]).reverse());
    let mut edges: Vec<PlanarEdge> = Vec::new();
    let mut representative: Option<[f64; 2]> = None;
    for (control, direction) in oriented {
        let (least, most) = domain.deletion_range(control);
        let width = most - least;
        let step = [width * direction[0], width * direction[1]];
        let same_direction = representative.is_some_and(|previous| {
            determinant_sign(previous[0], previous[1], direction[0], direction[1]) == Ordering::Equal
        });
        if same_direction && let Some(edge) = edges.last_mut() {
            edge.controls.push(control);
            edge.step[0] += step[0];
            edge.step[1] += step[1];
            continue;
        }
        edges.push(PlanarEdge {
            controls: vec![control],
            step,
        });
        representative = Some(direction);
    }
    let mut vertices = Vec::with_capacity(2 * edges.len().max(1));
    vertices.push(origin);
    let mut point = origin;
    for edge in &edges {
        point = [point[0] + edge.step[0], point[1] + edge.step[1]];
        vertices.push(point);
    }
    if edges.len() >= 2 {
        for edge in &edges[..edges.len() - 1] {
            point = [point[0] - edge.step[0], point[1] - edge.step[1]];
            vertices.push(point);
        }
    }
    // A vertex coordinate accumulates the origin's products and sums, the edges' sums and the walk:
    // at most `4·free + 3` rounded operations over terms bounded by `absolute`.
    let band = accumulation_growth(4 * free + 3) * absolute;
    PlanarBoundary {
        vertices,
        edges,
        inert,
        band,
        start,
    }
}

/// The order behind [`MaskMomentSystem::merge_positive_collinear`]'s candidate sort: the nonzero
/// pattern, the first entry's sign, then the entries divided by the first entry.
fn collinearity_key_order(left: &[(usize, usize, f64)], right: &[(usize, usize, f64)]) -> Ordering {
    let pivot = |entries: &[(usize, usize, f64)]| entries.first().map_or(1.0, |entry| entry.2);
    let (left_pivot, right_pivot) = (pivot(left), pivot(right));
    left.len()
        .cmp(&right.len())
        .then_with(|| (left_pivot > 0.0).cmp(&(right_pivot > 0.0)))
        .then_with(|| {
            left.iter()
                .zip(right)
                .map(|(a, b)| {
                    (a.0, a.1)
                        .cmp(&(b.0, b.1))
                        .then((a.2 / left_pivot).total_cmp(&(b.2 / right_pivot)))
                })
                .find(|ordering| *ordering != Ordering::Equal)
                .unwrap_or(Ordering::Equal)
        })
}

/// Whether two flattened generators are positive multiples of each other, decided exactly: the
/// same nonzero pattern, first entries of one sign, and every entry's 2×2 minor against the first
/// entries zero.
fn positively_collinear(left: &[(usize, usize, f64)], right: &[(usize, usize, f64)]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let (Some(&(.., left_pivot)), Some(&(.., right_pivot))) = (left.first(), right.first()) else {
        return true;
    };
    (left_pivot > 0.0) == (right_pivot > 0.0)
        && left.iter().zip(right).all(|(a, b)| {
            a.0 == b.0 && a.1 == b.1 && determinant_sign(left_pivot, a.2, right_pivot, b.2) == Ordering::Equal
        })
}

/// Exact sign of `a·d − b·c` for finite `f64` entries.
///
/// Each product is split into a mantissa product in `[1, 4)` and a binary exponent. Products whose
/// exponents differ by at least two are ordered by exponent alone. Otherwise the difference is
/// formed as an exact four-term floating-point expansion and read off its largest component
/// (Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates",
/// 1997, §2).
fn determinant_sign(a: f64, b: f64, c: f64, d: f64) -> Ordering {
    match (split_product(a, d), split_product(b, c)) {
        (None, None) => Ordering::Equal,
        (Some((mantissa, _, _)), None) => sign_of(mantissa),
        (None, Some((mantissa, _, _))) => sign_of(mantissa).reverse(),
        (Some((left, left_error, left_exponent)), Some((right, right_error, right_exponent))) => {
            if left_exponent >= right_exponent + 2 {
                return sign_of(left);
            }
            if right_exponent >= left_exponent + 2 {
                return sign_of(right).reverse();
            }
            let common = left_exponent.min(right_exponent);
            let left_scale = 2.0_f64.powi(left_exponent - common);
            let right_scale = 2.0_f64.powi(right_exponent - common);
            expansion_sign([
                left * left_scale,
                left_error * left_scale,
                -right * right_scale,
                -right_error * right_scale,
            ])
        }
    }
}

/// `x·y = (product + error)·2^exponent` exactly, with `|product| ∈ [1, 4)`; `None` when either
/// factor is zero.
fn split_product(x: f64, y: f64) -> Option<(f64, f64, i32)> {
    if x == 0.0 || y == 0.0 {
        return None;
    }
    let (x_mantissa, x_exponent) = split_exponent(x);
    let (y_mantissa, y_exponent) = split_exponent(y);
    let product = x_mantissa * y_mantissa;
    let error = x_mantissa.mul_add(y_mantissa, -product);
    Some((product, error, x_exponent + y_exponent))
}

/// `x = mantissa·2^exponent` with `|mantissa| ∈ [1, 2)`, for finite nonzero `x`.
fn split_exponent(x: f64) -> (f64, i32) {
    const EXPONENT_MASK: u64 = 0x7ff << 52;
    let bits = x.to_bits();
    let biased = ((bits & EXPONENT_MASK) >> 52) as i32;
    if biased == 0 {
        // Scaling a subnormal by 2^54 is exact and makes it normal.
        let (mantissa, exponent) = split_exponent(x * 2.0_f64.powi(54));
        return (mantissa, exponent - 54);
    }
    (f64::from_bits((bits & !EXPONENT_MASK) | (1023 << 52)), biased - 1023)
}

/// Sign of the exact sum of four `f64` terms, via Shewchuk's grow-expansion.
///
/// Each growth step keeps the expansion nonoverlapping and ordered by increasing magnitude, so its
/// largest nonzero component carries the sign of the sum.
fn expansion_sign(terms: [f64; 4]) -> Ordering {
    let mut expansion = [0.0_f64; 4];
    for (length, &term) in terms.iter().enumerate() {
        let mut carry = term;
        for component in expansion[..length].iter_mut() {
            let (sum, error) = two_sum(carry, *component);
            *component = error;
            carry = sum;
        }
        expansion[length] = carry;
    }
    expansion
        .iter()
        .rev()
        .find(|component| **component != 0.0)
        .map_or(Ordering::Equal, |component| sign_of(*component))
}

/// Knuth's branch-free two-sum: `a + b = sum + error` exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    (sum, (a - a_virtual) + (b - b_virtual))
}

fn sign_of(value: f64) -> Ordering {
    if value > 0.0 {
        Ordering::Greater
    } else if value < 0.0 {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn part(block: usize, vector: &[f64]) -> GeneratorPart {
        GeneratorPart {
            block,
            vector: Array1::from(vector.to_vec()),
        }
    }

    fn coordinate(block: usize, index: usize) -> MomentCoordinate {
        MomentCoordinate { block, index }
    }

    fn block(dimension: usize) -> MomentBlock {
        MomentBlock { dimension }
    }

    fn unit_domain(controls: usize) -> MaskDomain {
        MaskDomain::new(vec![(0.0, 1.0); controls]).expect("the unit interval contains all-on")
    }

    /// Integer generators in a 2-D and a 1-D block, so every moment, pairing and vertex below is
    /// computed without rounding and the comparisons are exact. Control 1 is tied across both
    /// blocks, 4 is a positive multiple of 0, 5 is antiparallel to 0, 6 is zero, and control 2
    /// takes a signed mask.
    fn integer_fixture() -> (MaskMomentSystem, MaskDomain) {
        let system = MaskMomentSystem::new(
            vec![block(2), block(1)],
            vec![
                vec![part(0, &[3.0, 1.0])],
                vec![part(0, &[-1.0, 2.0]), part(1, &[1.0])],
                vec![part(0, &[2.0, -2.0])],
                vec![part(1, &[-3.0])],
                vec![part(0, &[6.0, 2.0])],
                vec![part(0, &[-3.0, -1.0])],
                vec![part(0, &[0.0, 0.0]), part(1, &[0.0])],
            ],
        )
        .expect("well-formed generators");
        let mut intervals = vec![(0.0, 1.0); 7];
        intervals[2] = (-1.0, 1.0);
        (system, MaskDomain::new(intervals).expect("every interval contains all-on"))
    }

    /// A single 2-D block with integer generators and one signed control; (1, 2) and (2, 4) share a
    /// direction.
    fn planar_fixture() -> (MaskMomentSystem, MaskDomain) {
        let generators = [[1.0, 2.0], [-2.0, 1.0], [3.0, -1.0], [0.0, 2.0], [-1.0, -1.0], [2.0, 4.0], [1.0, 0.0], [-3.0, 2.0]];
        let system = MaskMomentSystem::new(
            vec![block(2)],
            generators.iter().map(|vector| vec![part(0, vector)]).collect(),
        )
        .expect("well-formed generators");
        let mut intervals = vec![(0.0, 1.0); 8];
        intervals[6] = (-1.0, 1.0);
        (system, MaskDomain::new(intervals).expect("every interval contains all-on"))
    }

    /// Every endpoint mask of the declared domain, with the kept controls at all-on.
    fn endpoint_masks(domain: &MaskDomain, kept: &[bool]) -> Vec<Vec<f64>> {
        let free: Vec<usize> = (0..kept.len()).filter(|&control| !kept[control]).collect();
        (0..1usize << free.len())
            .map(|bits| {
                let mut mask = vec![1.0; kept.len()];
                for (position, &control) in free.iter().enumerate() {
                    let (lower, upper) = domain.interval(control).expect("control in range");
                    mask[control] = if (bits >> position) & 1 == 1 { lower } else { upper };
                }
                mask
            })
            .collect()
    }

    fn pair(moment: &MomentVector, direction: &MomentVector) -> f64 {
        moment
            .blocks
            .iter()
            .zip(&direction.blocks)
            .map(|(q, u)| q.dot(u))
            .sum()
    }

    fn covector(values: [f64; 3]) -> MomentVector {
        MomentVector {
            blocks: vec![array![values[0], values[1]], array![values[2]]],
        }
    }

    fn planar_covector(values: [f64; 2]) -> MomentVector {
        MomentVector {
            blocks: vec![array![values[0], values[1]]],
        }
    }

    fn cross(origin: [f64; 2], to: [f64; 2], point: [f64; 2]) -> f64 {
        (to[0] - origin[0]) * (point[1] - origin[1]) - (to[1] - origin[1]) * (point[0] - origin[0])
    }

    fn planar(moment: &MomentVector, plane: [MomentCoordinate; 2]) -> [f64; 2] {
        [
            moment.component(plane[0]).expect("plane coordinate"),
            moment.component(plane[1]).expect("plane coordinate"),
        ]
    }

    #[test]
    fn support_equals_the_exhaustive_endpoint_maximum_and_bounds_interior_masks_2951() {
        let (system, domain) = integer_fixture();
        let directions = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, -1.0, 3.0],
            [-1.0, -4.0, 1.0],
            [1.0, -3.0, -2.0],
            [0.0, 0.0, 0.0],
        ];
        for kept in [vec![false; 7], vec![false, false, false, true, false, true, false]] {
            let masks = endpoint_masks(&domain, &kept);
            for values in directions {
                let direction = covector(values);
                let support = system.support(&domain, &kept, &direction).expect("valid inputs");
                let exhaustive = masks
                    .iter()
                    .map(|mask| pair(&system.moment(&domain, mask).expect("endpoint mask admissible"), &direction))
                    .fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(support.value, exhaustive, "direction {values:?}, kept {kept:?}");
                let witness_mask = domain.mask_at(&support.witness).expect("one endpoint per control");
                let attained = pair(&system.moment(&domain, &witness_mask).expect("witness admissible"), &direction);
                assert_eq!(attained, support.value, "the witness attains the support");
                for (control, &is_kept) in kept.iter().enumerate() {
                    assert_eq!(support.witness[control] == WitnessEndpoint::Kept, is_kept);
                }
                // Interior dyadic masks are exact too, and no admissible mask exceeds the support.
                for shade in [0.25, 0.5, 0.75] {
                    let mask: Vec<f64> = (0..7)
                        .map(|control| {
                            if kept[control] {
                                1.0
                            } else {
                                let (lower, upper) = domain.interval(control).expect("control in range");
                                lower + shade * (upper - lower)
                            }
                        })
                        .collect();
                    let interior = pair(&system.moment(&domain, &mask).expect("interior admissible"), &direction);
                    assert!(interior <= support.value, "interior {interior} exceeds support {}", support.value);
                }
            }
        }
        // Positive control: a support that treats the signed control's interval as [0, 1] undershoots
        // the exhaustive maximum where that control's pairing is positive.
        let direction = covector([1.0, -1.0, 0.0]);
        let unit_intervals: f64 = (0..7)
            .map(|control| {
                system
                    .generator(control)
                    .expect("control in range")
                    .iter()
                    .map(|part| part.vector.dot(&direction.blocks[part.block]))
                    .sum::<f64>()
                    .max(0.0)
            })
            .sum();
        let support = system.support(&domain, &[false; 7], &direction).expect("valid inputs");
        assert_eq!((unit_intervals, support.value), (10.0, 14.0));
        // A mask outside the declared domain is refused rather than silently reduced.
        let mut outside = vec![1.0; 7];
        outside[0] = -1.0;
        assert_eq!(
            system.moment(&domain, &outside),
            Err(MomentGeometryError::MaskOutsideDomain {
                control: 0,
                value: -1.0,
                lower: 0.0,
                upper: 1.0
            })
        );
    }

    #[test]
    fn support_reports_orthogonal_controls_as_unresolved_2951() {
        let (system, domain) = integer_fixture();
        // ⟨(1, −3), (3, 1)⟩ = 0 for controls 0, 4 and 5; control 3 lives only in the block the direction
        // leaves at 0; control 6 is zero.
        let direction = covector([1.0, -3.0, 0.0]);
        let support = system.support(&domain, &[false; 7], &direction).expect("valid inputs");
        assert_eq!(support.unresolved, vec![0, 3, 4, 5, 6]);
        let masks = endpoint_masks(&domain, &[false; 7]);
        let exhaustive = masks
            .iter()
            .map(|mask| pair(&system.moment(&domain, mask).expect("admissible"), &direction))
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(support.value, exhaustive);
        // Negative control: a direction with no orthogonal generator resolves every nonzero control.
        let generic = system
            .support(&domain, &[false; 7], &covector([2.0, -1.0, 3.0]))
            .expect("valid inputs");
        assert_eq!(generic.unresolved, vec![6]);
    }

    #[test]
    fn planar_boundary_is_the_hull_of_the_exhaustive_endpoint_moments_2951() {
        let (system, domain) = integer_fixture();
        let plane = [coordinate(0, 0), coordinate(0, 1)];
        let kept = [false; 7];
        let boundary = system.planar_boundary(&domain, &kept, plane).expect("valid plane");
        // Directions (3, 1) for controls 0, 4 and 5; (−1, 2) for 1; (−2, 2) for 2. Controls 3 and 6 have
        // no planar generator.
        assert_eq!(boundary.edges.len(), 3);
        assert_eq!(boundary.vertices.len(), 6);
        assert_eq!(boundary.inert, vec![3, 6]);
        assert!(boundary.edges.iter().any(|edge| edge.controls == vec![0, 4, 5]));
        let count = boundary.vertices.len();
        for (index, &vertex) in boundary.vertices.iter().enumerate() {
            let witness = boundary.witness(index).expect("vertex in range");
            let moment = system
                .moment(&domain, &domain.mask_at(&witness).expect("one endpoint per control"))
                .expect("witness admissible");
            assert_eq!(planar(&moment, plane), vertex, "vertex {index} is its witness's moment");
            let next = boundary.vertices[(index + 1) % count];
            let after = boundary.vertices[(index + 2) % count];
            assert!(cross(vertex, next, after) > 0.0, "strict left turn at vertex {}", (index + 1) % count);
        }
        assert_eq!(boundary.witness(count), None);
        let points: Vec<[f64; 2]> = endpoint_masks(&domain, &kept)
            .iter()
            .map(|mask| planar(&system.moment(&domain, mask).expect("admissible"), plane))
            .collect();
        for &point in &points {
            for index in 0..count {
                let edge_cross = cross(boundary.vertices[index], boundary.vertices[(index + 1) % count], point);
                assert!(edge_cross >= 0.0, "{point:?} outside edge {index}");
            }
        }
        // Positive control: dropping a vertex leaves an exhaustive moment strictly outside the polygon.
        let mut clipped = boundary.vertices.clone();
        clipped.remove(1);
        let outside = points.iter().any(|&point| {
            (0..clipped.len()).any(|index| cross(clipped[index], clipped[(index + 1) % clipped.len()], point) < 0.0)
        });
        assert!(outside, "the containment check must see a clipped polygon");
        // Seven distinct directions give 14 vertices; keeping controls 1 and 4 leaves five and 10.
        let (planar_system, planar_domain) = planar_fixture();
        let plane = [coordinate(0, 0), coordinate(0, 1)];
        for (kept, vertex_count) in [
            (vec![false; 8], 14),
            (vec![false, true, false, false, true, false, false, false], 10),
        ] {
            let boundary = planar_system
                .planar_boundary(&planar_domain, &kept, plane)
                .expect("valid plane");
            assert_eq!(boundary.vertices.len(), vertex_count);
        }
    }

    #[test]
    fn positive_collinear_refinement_leaves_the_zonotope_unchanged_2951() {
        let coarse = MaskMomentSystem::new(
            vec![block(2)],
            vec![vec![part(0, &[4.0, 2.0])], vec![part(0, &[-1.0, 3.0])], vec![part(0, &[2.0, -6.0])]],
        )
        .expect("well-formed generators");
        // (4, 2) split into the dyadic pieces 1/2, 1/4 and 1/4, which are exact.
        let refined = MaskMomentSystem::new(
            vec![block(2)],
            vec![
                vec![part(0, &[2.0, 1.0])],
                vec![part(0, &[-1.0, 3.0])],
                vec![part(0, &[2.0, -6.0])],
                vec![part(0, &[1.0, 0.5])],
                vec![part(0, &[1.0, 0.5])],
            ],
        )
        .expect("well-formed generators");
        // Negative control: pieces that sum to (4, 2) but are not collinear change the set.
        let bent = MaskMomentSystem::new(
            vec![block(2)],
            vec![
                vec![part(0, &[2.0, 2.0])],
                vec![part(0, &[-1.0, 3.0])],
                vec![part(0, &[2.0, -6.0])],
                vec![part(0, &[2.0, 0.0])],
            ],
        )
        .expect("well-formed generators");
        let plane = [coordinate(0, 0), coordinate(0, 1)];
        let coarse_boundary = coarse.planar_boundary(&unit_domain(3), &[false; 3], plane).expect("valid plane");
        let refined_boundary = refined.planar_boundary(&unit_domain(5), &[false; 5], plane).expect("valid plane");
        assert_eq!(refined_boundary.vertices, coarse_boundary.vertices);
        assert!(refined_boundary.edges.iter().any(|edge| edge.controls == vec![0, 3, 4]));
        let bent_boundary = bent.planar_boundary(&unit_domain(4), &[false; 4], plane).expect("valid plane");
        assert_ne!(bent_boundary.vertices.len(), coarse_boundary.vertices.len());
        // Only a direction splitting the bent pieces' signs, like (−1, 2), sees the bend: 9 against 7.
        for values in [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [-2.0, 1.0], [3.0, 1.0], [-1.0, 2.0]] {
            let direction = planar_covector(values);
            let coarse_value = coarse.support(&unit_domain(3), &[false; 3], &direction).expect("valid").value;
            let refined_value = refined.support(&unit_domain(5), &[false; 5], &direction).expect("valid").value;
            assert_eq!(refined_value, coarse_value, "direction {values:?}");
            let bent_value = bent.support(&unit_domain(4), &[false; 4], &direction).expect("valid").value;
            if values == [-1.0, 2.0] {
                assert_eq!((bent_value, coarse_value), (9.0, 7.0));
            }
        }
    }

    #[test]
    fn collinear_merge_folds_positive_multiples_and_keeps_the_support_2951() {
        let (system, domain) = integer_fixture();
        let merge = system.merge_positive_collinear(&domain).expect("valid domain");
        // (3, 1) and (6, 2) share a direction and an interval. (−3, −1) is antiparallel. The zero
        // generator, the tied control and the signed control each stay alone.
        assert_eq!(merge.groups, vec![vec![0, 4], vec![1], vec![2], vec![3], vec![5], vec![6]]);
        assert_eq!(merge.system.generator(0).expect("merged control")[0].vector, array![9.0, 3.0]);
        assert_eq!(merge.band, accumulation_growth(1) * 9.0);
        for values in [[1.0, 0.0, 0.0], [2.0, -1.0, 3.0], [-1.0, -4.0, 1.0], [1.0, -3.0, 0.0]] {
            let direction = covector(values);
            let original = system.support(&domain, &[false; 7], &direction).expect("valid");
            let merged = merge.system.support(&merge.domain, &[false; 6], &direction).expect("valid");
            assert_eq!(merged.value, original.value, "direction {values:?}");
        }
        // Negative controls: the same direction under a different interval, and a direction off by two
        // ulps, are not merged.
        let near = MaskMomentSystem::new(
            vec![block(2)],
            vec![
                vec![part(0, &[3.0, 1.0])],
                vec![part(0, &[6.0, 2.0])],
                vec![part(0, &[6.0, 2.0 + 4.0 * f64::EPSILON])],
                vec![part(0, &[1.5, 0.5])],
            ],
        )
        .expect("well-formed generators");
        let near_domain =
            MaskDomain::new(vec![(0.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (0.0, 1.0)]).expect("contains all-on");
        let near_merge = near.merge_positive_collinear(&near_domain).expect("valid domain");
        assert_eq!(near_merge.groups, vec![vec![0, 3], vec![1], vec![2]]);
    }

    #[test]
    fn a_cancelling_pair_average_is_never_reported_as_a_bound_2951() {
        // E_n = Σ_i (t_i − s_i)·a/n with a = 1: n pieces of +1/n and n of −1/n, all exact dyadics.
        let split = |pieces: usize| {
            let piece = 1.0 / pieces as f64;
            let generators = (0..2 * pieces)
                .map(|index| vec![part(0, &[if index < pieces { piece } else { -piece }])])
                .collect();
            MaskMomentSystem::new(vec![block(1)], generators).expect("well-formed generators")
        };
        let direction = MomentVector {
            blocks: vec![array![1.0]],
        };
        let mut previous_mean_square = f64::INFINITY;
        for pieces in [4usize, 16] {
            let system = split(pieces);
            let domain = unit_domain(2 * pieces);
            let kept = vec![false; 2 * pieces];
            let law = system.uniform_mask_law_moments(&domain, &kept, &direction).expect("valid");
            // Mean square a²/(6n), within the law's band and the literal's own rounding.
            let expected = 1.0 / (6.0 * pieces as f64);
            assert_eq!(law.mean, 0.0);
            assert!((law.mean_square - expected).abs() <= law.mean_square_band + 2.0 * UNIT_ROUNDOFF * expected);
            assert!(law.mean_square < previous_mean_square, "the average shrinks with n");
            previous_mean_square = law.mean_square;
            // The certified supremum stays |a| = 1, attained by fully deleting every positive piece.
            let supremum = system.absolute_supremum(&domain, &kept, &direction).expect("valid");
            assert_eq!(supremum.value, 1.0);
            assert!(supremum.witness[..pieces].iter().all(|&endpoint| endpoint == WitnessEndpoint::Lower));
            assert!(supremum.witness[pieces..].iter().all(|&endpoint| endpoint == WitnessEndpoint::Upper));
            // An observed worst case over interior dyadic masks is a third, different number.
            let observed = [0.25, 0.5, 0.75]
                .iter()
                .flat_map(|&shade| {
                    [
                        (0..2 * pieces)
                            .map(|index| if index < pieces { shade } else { 1.0 - shade })
                            .collect::<Vec<f64>>(),
                        (0..2 * pieces)
                            .map(|index| if index % 2 == 0 { shade } else { 1.0 })
                            .collect::<Vec<f64>>(),
                    ]
                })
                .map(|mask| system.moment(&domain, &mask).expect("admissible").blocks[0][0].abs())
                .fold(0.0_f64, f64::max);
            assert_eq!(observed, 0.5);
            assert!(observed < supremum.value && observed > law.mean_square.sqrt());
            // The guard: the root mean square claimed as a bound is refuted by the witness mask.
            match system
                .check_claimed_absolute_bound(&domain, &kept, &direction, law.mean_square.sqrt())
                .expect("finite claim")
            {
                ClaimedBoundCheck::Refuted { attained, witness, .. } => {
                    assert_eq!(attained, 1.0);
                    let moment = system
                        .moment(&domain, &domain.mask_at(&witness).expect("same controls"))
                        .expect("admissible");
                    assert_eq!(moment.blocks[0][0], 1.0);
                }
                other => panic!("an average claimed as a bound must be refuted, got {other:?}"),
            }
            // Negative control: the supremum plus its band is certified, and the bare supremum sits
            // inside the band.
            assert_eq!(
                system
                    .check_claimed_absolute_bound(&domain, &kept, &direction, supremum.value + supremum.band)
                    .expect("finite claim"),
                ClaimedBoundCheck::Certified {
                    supremum: 1.0,
                    band: supremum.band
                }
            );
            assert!(matches!(
                system
                    .check_claimed_absolute_bound(&domain, &kept, &direction, 1.0)
                    .expect("finite claim"),
                ClaimedBoundCheck::Unresolved { .. }
            ));
        }
        // A single non-cancelling component keeps its bias: E q² = 1/3 whole and 1/4 + 1/(12n) split.
        let single = MaskMomentSystem::new(vec![block(1)], vec![vec![part(0, &[1.0])]]).expect("valid");
        let whole = single
            .uniform_mask_law_moments(&unit_domain(1), &[false], &direction)
            .expect("valid");
        assert!((whole.mean_square - 1.0 / 3.0).abs() <= whole.mean_square_band + UNIT_ROUNDOFF);
        let pieces = MaskMomentSystem::new(vec![block(1)], std::iter::repeat_n(vec![part(0, &[0.25])], 4).collect())
            .expect("valid");
        let split_law = pieces
            .uniform_mask_law_moments(&unit_domain(4), &[false; 4], &direction)
            .expect("valid");
        let expected = 0.25 + 1.0 / 48.0;
        assert!((split_law.mean_square - expected).abs() <= split_law.mean_square_band + 2.0 * UNIT_ROUNDOFF * expected);
    }

    #[test]
    fn determinant_sign_is_exact_where_the_rounded_determinant_is_not_2951() {
        // (1 + ε)(1 − ε) − 1 = −ε², which rounds away in the naive formula.
        let epsilon = f64::EPSILON;
        let naive = (1.0 + epsilon) * (1.0 - epsilon) - 1.0 * 1.0;
        assert_eq!(naive, 0.0, "positive control: the naive determinant loses the sign");
        assert_eq!(determinant_sign(1.0 + epsilon, 1.0, 1.0, 1.0 - epsilon), Ordering::Less);
        assert_eq!(determinant_sign(1.0, 1.0 - epsilon, 1.0 + epsilon, 1.0), Ordering::Greater);
        // Exactly collinear vectors at very different scales.
        let small = 2.0_f64.powi(-600);
        let large = 2.0_f64.powi(500);
        assert_eq!(determinant_sign(3.0 * small, small, 3.0 * large, large), Ordering::Equal);
        assert_eq!(
            determinant_sign(3.0 * small, small, 3.0 * large, large * (1.0 + epsilon)),
            Ordering::Greater
        );
        // Subnormal entries.
        let tiny = f64::from_bits(1);
        assert_eq!(determinant_sign(tiny, 1.0, 1.0, tiny), Ordering::Less);
        assert_eq!(determinant_sign(3.0 * tiny, tiny, 6.0 * tiny, 2.0 * tiny), Ordering::Equal);
        assert_eq!(determinant_sign(0.0, 2.0, 0.0, 5.0), Ordering::Equal);
    }

    #[test]
    fn support_witness_and_boundary_are_invariant_under_a_moment_basis_change_2951() {
        let (system, domain) = planar_fixture();
        let kept = [false; 8];
        let apply = |matrix: [[f64; 2]; 2], vector: [f64; 2]| {
            [
                matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
                matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
            ]
        };
        let transpose = |matrix: [[f64; 2]; 2]| [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]];
        // q → G·q with a unimodular G, so G⁻¹ is integer and every transformed quantity stays exact.
        for (forward, inverse) in [
            ([[2.0, 1.0], [1.0, 1.0]], [[1.0, -1.0], [-1.0, 2.0]]),
            ([[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]),
        ] {
            let transformed = MaskMomentSystem::new(
                vec![block(2)],
                (0..8)
                    .map(|control| {
                        let vector = &system.generator(control).expect("control in range")[0].vector;
                        vec![part(0, &apply(forward, [vector[0], vector[1]]))]
                    })
                    .collect(),
            )
            .expect("well-formed generators");
            // u → G⁻ᵀ·u keeps every pairing ⟨u, v_c⟩.
            for values in [[1.0, 0.0], [2.0, -3.0], [-1.0, 4.0], [5.0, 1.0]] {
                let original = system.support(&domain, &kept, &planar_covector(values)).expect("valid");
                let changed = transformed
                    .support(&domain, &kept, &planar_covector(apply(transpose(inverse), values)))
                    .expect("valid");
                assert_eq!(changed.value, original.value);
                assert_eq!(changed.witness, original.witness);
                // Positive control: for a non-orthogonal G, transforming the direction like a moment
                // breaks the invariance. At u = (2, −3), GᵀG·u = (1, 0) gives support 8 against 14.
                if forward != transpose(inverse) && values == [2.0, -3.0] {
                    let wrong = transformed
                        .support(&domain, &kept, &planar_covector(apply(forward, values)))
                        .expect("valid");
                    assert_eq!((wrong.value, original.value), (8.0, 14.0));
                }
            }
            // The boundary maps vertex by vertex: G carries each witness's moment onto the transformed
            // boundary, in the same cyclic order when det G > 0 and reversed when det G < 0.
            let plane = [coordinate(0, 0), coordinate(0, 1)];
            let before = system.planar_boundary(&domain, &kept, plane).expect("valid plane");
            let after = transformed.planar_boundary(&domain, &kept, plane).expect("valid plane");
            let count = before.vertices.len();
            assert_eq!(after.vertices.len(), count);
            let offset = (0..count)
                .find(|&index| after.vertices[index] == apply(forward, before.vertices[0]))
                .expect("G maps the first vertex onto the transformed boundary");
            let determinant = forward[0][0] * forward[1][1] - forward[0][1] * forward[1][0];
            for index in 0..count {
                let image = if determinant > 0.0 {
                    (offset + index) % count
                } else {
                    (offset + count - index) % count
                };
                assert_eq!(after.vertices[image], apply(forward, before.vertices[index]));
                assert_eq!(after.witness(image), before.witness(index));
            }
        }
    }
}
