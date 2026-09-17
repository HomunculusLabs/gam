//! Gauge and operator structure of parameter edits (#2951 P1, P2, P4).
//!
//! Three facts decide what an intervention on a factored or rotational edit
//! means, and each has its own type here.
//!
//! * **Mask gauge (P1).** A rank-`r` operator `P = U Vᵀ` with full-column-rank
//!   factors is the same operator after `U → U S`, `V → V S⁻ᵀ` for every
//!   `S ∈ GL(r)`. An internal mask `M` held fixed while the basis changes gives
//!   `U S M S⁻¹ Vᵀ`, which equals `U M Vᵀ` for every `S ∈ GL(r)` iff `M = cI`:
//!   `S M = M S` for all `S`, diagonal `S` forces `M` diagonal and permutations
//!   force equal entries. Over a declared block gauge `GL(r₁) ⊕ … ⊕ GL(r_K)` the
//!   same argument leaves exactly the block scalars `⊕ c_k I_{r_k}`, one tied
//!   control per block. Any other mask names an intervention only together with
//!   the basis it was written in, and is carried as an operator, `M → S⁻¹ M S`.
//!   Independent diagonal masks on an arbitrary basis of one block are not
//!   intrinsic interventions.
//! * **Curved versus straight paths (P2).** For a rotation
//!   `W = I + U (R(α) − I) Uᵀ` in orthonormal planes, the angle path
//!   `W(t) = I + U (R(tα) − I) Uᵀ` is orthogonal for every `t` and moves the
//!   in-plane angle, `R_{tα} D(φ) = D(φ + tα)` with `D(φ) = (cos φ, sin φ)`. The
//!   chord path `L(t) = (1 − t) I + t W`, which is what the residual anchor's mask
//!   does to `Δ = W − I`, satisfies `L(t)ᵀ L(t) = [1 − 2t(1 − t)(1 − cos α)] I` on
//!   each plane, so its midpoint radius is `|cos(α/2)|`. The two paths answer
//!   different questions: removing a contribution versus varying an angle.
//! * **Composition (P4).** `(I + Δ_A)(I + Δ_B) = I + Δ_A + Δ_B + Δ_A Δ_B`. The
//!   cross term is induced by the composition and is not a third mechanism;
//!   dropping it turns Compose into Sum. The order-swap gap of two edits is
//!   exactly the commutator `[Δ_A, Δ_B]`, and for rotation generators the edit
//!   loop is `e^{sA} e^{tB} e^{−sA} e^{−tB} = I + st [A, B] + O(s²t + st²)`.
//!
//! Every operator is applied to a vector: nothing here forms a `d × d` matrix.
//! The rotation blocks come from the `SO(2)` owner in `gam-geometry`.

use std::ops::Range;

use gam_geometry::manifolds::lie_so::rho_so2;
use gam_linalg::faer_ndarray::{FaerSvd, fast_ab, fast_abt, fast_atb, fast_atv, fast_av};
use gam_linalg::roundoff::factor_singular_band;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Why an operator construction or application was declined.
#[derive(Clone, Debug, PartialEq)]
pub enum OperatorRefusal {
    /// A dimension disagrees with the object it is applied to.
    DimensionMismatch {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    /// An input carries a non-finite entry.
    NonFinite { what: &'static str },
    /// No gauge block was declared, or a declared block has no coordinates.
    EmptyGaugeBlock,
    /// A factor of `P = U Vᵀ` has fewer singular values above its SVD's rounding
    /// band than the declared rank. The operator then has a gauge larger than the
    /// declared one, and a basis-dependent verdict would not be a counterexample.
    RankDeficientFactor {
        factor: &'static str,
        resolved: usize,
        declared: usize,
    },
    /// A gauge change couples coordinates of two declared blocks, so it is not an
    /// element of the declared group.
    GaugeOutsideDeclaredGroup { row: usize, col: usize },
    /// A gauge change has fewer singular values above its SVD's rounding band
    /// than its order.
    SingularGaugeChange { resolved: usize, order: usize },
    /// A plane basis has an odd column count, so its columns do not pair into
    /// planes.
    OddPlaneBasis { columns: usize },
    /// The linear-algebra backend failed to decompose a matrix.
    DecompositionFailed { what: &'static str, detail: String },
}

/// A declared implementation gauge of an `r`-coordinate internal space: the group
/// `GL(r₁) ⊕ … ⊕ GL(r_K)` acting on consecutive coordinate blocks.
///
/// One block of size `r` is the full `GL(r)` of an unstructured rank-`r` factor.
/// `r` blocks of size one is the per-component scale gauge of `r` rank-one
/// components `u_c v_cᵀ`, under which independent diagonal masks are intrinsic.
#[derive(Clone, Debug, PartialEq)]
pub struct GaugeBlocks {
    /// Block `k` holds coordinates `offsets[k]..offsets[k + 1]`.
    offsets: Vec<usize>,
}

impl GaugeBlocks {
    pub fn new(sizes: &[usize]) -> Result<Self, OperatorRefusal> {
        if sizes.is_empty() || sizes.contains(&0) {
            return Err(OperatorRefusal::EmptyGaugeBlock);
        }
        let mut offsets = Vec::with_capacity(sizes.len() + 1);
        offsets.push(0);
        for &size in sizes {
            offsets.push(offsets[offsets.len() - 1] + size);
        }
        Ok(Self { offsets })
    }

    /// The internal dimension `r = Σ r_k`.
    pub fn rank(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }

    pub fn block_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The coordinates of block `k`.
    pub fn block(&self, k: usize) -> Range<usize> {
        self.offsets[k]..self.offsets[k + 1]
    }

    /// The block holding coordinate `i`: the offsets start at 0 and increase
    /// strictly, so it is the last block whose first coordinate is at most `i`.
    fn block_of(&self, i: usize) -> usize {
        self.offsets.partition_point(|&offset| offset <= i) - 1
    }
}

/// An internal mask on a factored operator, typed by how it transforms under the
/// operator's declared gauge.
#[derive(Clone, Debug, PartialEq)]
pub enum InternalMask {
    /// `⊕ c_k I_{r_k}`, one tied control per declared gauge block. It commutes with
    /// every element of the declared group, so `U M Vᵀ` does not depend on the
    /// basis the factors were written in.
    BlockScalar(Vec<f64>),
    /// An `r × r` operator mask written in the internal basis of the factors it is
    /// carried with. A gauge change `U → U S`, `V → V S⁻ᵀ` carries it to
    /// `S⁻¹ M S`, so the intervention is covariant; the matrix alone names no
    /// intervention.
    Operator(Array2<f64>),
}

/// Whether a fixed internal mask is an intrinsic intervention under a declared
/// gauge.
#[derive(Clone, Debug, PartialEq)]
pub enum MaskGaugeVerdict {
    /// The mask is exactly `⊕ c_k I_{r_k}`: it commutes with the whole declared
    /// group.
    Intrinsic { controls: Vec<f64> },
    /// A gauge change `S` in the declared group with `S M S⁻¹ ≠ M`, the internal
    /// entry it moves, and by how much. With full-column-rank factors the fixed
    /// mask gives a different intervention after `U → U S`, `V → V S⁻ᵀ`: a
    /// counterexample, not an estimate.
    BasisDependent {
        gauge_change: Array2<f64>,
        moved: (usize, usize),
        shift: f64,
    },
}

/// Classify a fixed `r × r` internal mask under a declared gauge (P1).
///
/// `M` commutes with every element of `GL(r₁) ⊕ … ⊕ GL(r_K)` iff it is a block
/// scalar. The check is algebraic on the declared entries and carries no
/// tolerance: an entry that differs by any amount moves the intervention by that
/// amount along the witness. The witness is the reflection `S = I − 2 e_i e_iᵀ`,
/// which negates an off-diagonal entry `M_ij`, or the transposition of two
/// coordinates of one block, which swaps two unequal diagonal entries. Both lie
/// in the declared group.
pub fn classify_internal_mask(
    mask: ArrayView2<'_, f64>,
    gauge: &GaugeBlocks,
) -> Result<MaskGaugeVerdict, OperatorRefusal> {
    let r = gauge.rank();
    check_square("internal mask order", r, mask)?;
    if !mask.iter().all(|entry| entry.is_finite()) {
        return Err(OperatorRefusal::NonFinite {
            what: "internal mask",
        });
    }
    for ((i, j), &entry) in mask.indexed_iter() {
        if i != j && entry != 0.0 {
            let mut gauge_change = Array2::<f64>::eye(r);
            gauge_change[[i, i]] = -1.0;
            return Ok(MaskGaugeVerdict::BasisDependent {
                gauge_change,
                moved: (i, j),
                shift: -2.0 * entry,
            });
        }
    }
    let mut controls = Vec::with_capacity(gauge.block_count());
    for k in 0..gauge.block_count() {
        let block = gauge.block(k);
        let lead = block.start;
        let control = mask[[lead, lead]];
        for i in block.start + 1..block.end {
            if mask[[i, i]] != control {
                let mut gauge_change = Array2::<f64>::eye(r);
                gauge_change[[lead, lead]] = 0.0;
                gauge_change[[i, i]] = 0.0;
                gauge_change[[lead, i]] = 1.0;
                gauge_change[[i, lead]] = 1.0;
                return Ok(MaskGaugeVerdict::BasisDependent {
                    gauge_change,
                    moved: (lead, lead),
                    shift: mask[[i, i]] - control,
                });
            }
        }
        controls.push(control);
    }
    Ok(MaskGaugeVerdict::Intrinsic { controls })
}

/// A rank-`r` operator `P = U Vᵀ` with full-column-rank factors and a declared
/// gauge, stored as its `d_out × r` and `d_in × r` factors and applied to vectors
/// only.
#[derive(Clone, Debug)]
pub struct FactoredOperator {
    u: Array2<f64>,
    v: Array2<f64>,
    gauge: GaugeBlocks,
}

impl FactoredOperator {
    /// Refuses factors whose column count is not the gauge's rank, non-finite
    /// entries, and a factor with a singular value inside its SVD's rounding band
    /// [`factor_singular_band`]: P1's characterization needs full column rank.
    pub fn new(u: Array2<f64>, v: Array2<f64>, gauge: GaugeBlocks) -> Result<Self, OperatorRefusal> {
        let r = gauge.rank();
        for (what, factor) in [("left factor", &u), ("right factor", &v)] {
            check_len(what, r, factor.ncols())?;
            if !factor.iter().all(|entry| entry.is_finite()) {
                return Err(OperatorRefusal::NonFinite { what });
            }
            let sigma = factor
                .svd(false, false)
                .map_err(|failure| decomposition_failed(what, &failure))?
                .1;
            let resolved = resolved_singular_count(&sigma, factor.nrows(), factor.ncols());
            if resolved < r {
                return Err(OperatorRefusal::RankDeficientFactor {
                    factor: what,
                    resolved,
                    declared: r,
                });
            }
        }
        Ok(Self { u, v, gauge })
    }

    pub fn gauge(&self) -> &GaugeBlocks {
        &self.gauge
    }

    /// `y = U M Vᵀ x`, through the `r` internal coordinates only.
    pub fn apply(&self, mask: &InternalMask, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        check_len("operator input", self.v.nrows(), x.len())?;
        let internal = fast_atv(&self.v, &x);
        let masked = match mask {
            InternalMask::BlockScalar(controls) => {
                check_len("block-scalar controls", self.gauge.block_count(), controls.len())?;
                let mut scaled = internal;
                for (k, &control) in controls.iter().enumerate() {
                    for i in self.gauge.block(k) {
                        scaled[i] *= control;
                    }
                }
                scaled
            }
            InternalMask::Operator(matrix) => {
                check_square("operator mask order", self.gauge.rank(), matrix.view())?;
                fast_av(matrix, &internal)
            }
        };
        Ok(fast_av(&self.u, &masked))
    }

    /// Reparametrize by a gauge change `S` in the declared group, `U → U S` and
    /// `V → V S⁻ᵀ`, carrying the mask covariantly. A block-scalar mask is unchanged
    /// because it commutes with `S`; an operator mask becomes `S⁻¹ M S`. The
    /// operator and every masked intervention are unchanged up to roundoff.
    pub fn regauge(
        &self,
        gauge_change: ArrayView2<'_, f64>,
        mask: &InternalMask,
    ) -> Result<(Self, InternalMask), OperatorRefusal> {
        let r = self.gauge.rank();
        check_square("gauge change order", r, gauge_change)?;
        if !gauge_change.iter().all(|entry| entry.is_finite()) {
            return Err(OperatorRefusal::NonFinite {
                what: "gauge change",
            });
        }
        for ((i, j), &entry) in gauge_change.indexed_iter() {
            if entry != 0.0 && self.gauge.block_of(i) != self.gauge.block_of(j) {
                return Err(OperatorRefusal::GaugeOutsideDeclaredGroup { row: i, col: j });
            }
        }
        let inverse = invert_gauge_change(gauge_change)?;
        let carried = match mask {
            InternalMask::BlockScalar(controls) => {
                check_len("block-scalar controls", self.gauge.block_count(), controls.len())?;
                InternalMask::BlockScalar(controls.clone())
            }
            InternalMask::Operator(matrix) => {
                check_square("operator mask order", r, matrix.view())?;
                InternalMask::Operator(fast_ab(&fast_ab(&inverse, matrix), &gauge_change))
            }
        };
        let regauged = Self {
            u: fast_ab(&self.u, &gauge_change),
            v: fast_abt(&self.v, &inverse),
            gauge: self.gauge.clone(),
        };
        Ok((regauged, carried))
    }

    /// This operator under `mask` read as the residual edit `Δ = U M Vᵀ` of a
    /// square map; refuses a rectangular operator.
    pub fn residual_edit<'a>(&'a self, mask: &'a InternalMask) -> Result<MaskedOperator<'a>, OperatorRefusal> {
        check_len("residual edit output dimension", self.v.nrows(), self.u.nrows())?;
        Ok(MaskedOperator { operator: self, mask })
    }
}

fn decomposition_failed(what: &'static str, failure: &gam_linalg::faer_ndarray::FaerLinalgError) -> OperatorRefusal {
    OperatorRefusal::DecompositionFailed {
        what,
        detail: format!("{failure:?}"),
    }
}

/// The number of singular values of a `rows × cols` matrix above the SVD's
/// rounding band `max(rows, cols)·ε·σ_max` ([`factor_singular_band`]).
fn resolved_singular_count(sigma: &Array1<f64>, rows: usize, cols: usize) -> usize {
    let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
    let band = factor_singular_band(rows, cols, sigma_max);
    sigma.iter().filter(|&&value| value > band).count()
}

/// `S⁻¹ = V Σ⁻¹ Uᵀ` from the SVD `S = U Σ Vᵀ`, refusing a singular value inside
/// the SVD's rounding band.
fn invert_gauge_change(gauge_change: ArrayView2<'_, f64>) -> Result<Array2<f64>, OperatorRefusal> {
    let what = "gauge change";
    let order = gauge_change.nrows();
    let (left, sigma, right_t) = gauge_change
        .svd(true, true)
        .map_err(|failure| decomposition_failed(what, &failure))?;
    let resolved = resolved_singular_count(&sigma, order, order);
    if resolved < order {
        return Err(OperatorRefusal::SingularGaugeChange { resolved, order });
    }
    let (Some(left), Some(right_t)) = (left, right_t) else {
        return Err(OperatorRefusal::DecompositionFailed {
            what,
            detail: "singular vectors were requested but not returned".to_string(),
        });
    };
    let mut right_scaled = right_t.t().to_owned();
    for (j, &value) in sigma.iter().enumerate() {
        right_scaled.column_mut(j).mapv_inplace(|entry| entry / value);
    }
    Ok(fast_abt(&right_scaled, &left))
}

/// A parameter edit written as a residual `W = I + Δ` on a `dim`-dimensional
/// space and applied matrix-free.
pub trait ResidualEdit {
    /// The dimension of the space `W` acts on.
    fn dim(&self) -> usize;

    /// `Δ x`.
    fn apply_delta(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal>;
}

/// A square factored operator under an internal mask, as the residual edit
/// `Δ = U M Vᵀ`. Built by [`FactoredOperator::residual_edit`].
#[derive(Clone, Copy, Debug)]
pub struct MaskedOperator<'a> {
    operator: &'a FactoredOperator,
    mask: &'a InternalMask,
}

impl ResidualEdit for MaskedOperator<'_> {
    fn dim(&self) -> usize {
        self.operator.v.nrows()
    }

    fn apply_delta(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        self.operator.apply(self.mask, x)
    }
}

/// Which path a rotation edit is read along (P2). The two are different
/// experiments, so a path is never inferred from a bare scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RotationPath {
    /// `W(t) = I + U (R(tα) − I) Uᵀ`: a structured-edit coordinate. Orthogonal for
    /// every `t`; it varies the angles.
    Angle(f64),
    /// `L(t) = I + t U (R(α) − I) Uᵀ`: the residual anchor's mask on `Δ = W − I`.
    /// It removes the contribution and contracts plane `j` by
    /// [`PlaneRotation::chord_radii`].
    Chord(f64),
}

/// A rotation in `k` mutually orthogonal planes, `W = I + U (R(α) − I) Uᵀ`, where
/// columns `2j` and `2j + 1` of the `d × 2k` basis `U` span plane `j` and
/// `R(α) = ⊕_j R_{α_j}` with `R_θ = [[cos θ, −sin θ], [sin θ, cos θ]]`.
#[derive(Clone, Debug)]
pub struct PlaneRotation {
    basis: Array2<f64>,
    angles: Array1<f64>,
    orthonormality_defect: f64,
}

impl PlaneRotation {
    /// The basis is not refused for a loss of orthonormality; the defect
    /// `‖UᵀU − I‖_F` is measured once and reported, and every identity of the
    /// module holds for this edit only up to it.
    pub fn new(basis: Array2<f64>, angles: Array1<f64>) -> Result<Self, OperatorRefusal> {
        if basis.ncols() % 2 != 0 {
            return Err(OperatorRefusal::OddPlaneBasis {
                columns: basis.ncols(),
            });
        }
        check_len("plane angles", basis.ncols() / 2, angles.len())?;
        if !basis.iter().all(|entry| entry.is_finite()) {
            return Err(OperatorRefusal::NonFinite { what: "plane basis" });
        }
        if !angles.iter().all(|angle| angle.is_finite()) {
            return Err(OperatorRefusal::NonFinite { what: "plane angles" });
        }
        let gram = fast_atb(&basis, &basis);
        let defect_sq: f64 = gram
            .indexed_iter()
            .map(|((i, j), &entry)| {
                let deviation = if i == j { entry - 1.0 } else { entry };
                deviation * deviation
            })
            .sum();
        Ok(Self {
            basis,
            angles,
            orthonormality_defect: defect_sq.sqrt(),
        })
    }

    pub fn dim(&self) -> usize {
        self.basis.nrows()
    }

    pub fn plane_count(&self) -> usize {
        self.angles.len()
    }

    pub fn angles(&self) -> ArrayView1<'_, f64> {
        self.angles.view()
    }

    /// `‖UᵀU − I‖_F`, measured at construction.
    pub fn orthonormality_defect(&self) -> f64 {
        self.orthonormality_defect
    }

    /// `Δ x` along `path`.
    pub fn apply_delta(&self, path: RotationPath, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        match path {
            RotationPath::Angle(t) => self.plane_delta(&self.angles.mapv(|alpha| t * alpha), x),
            RotationPath::Chord(t) => Ok(self.plane_delta(&self.angles, x)? * t),
        }
    }

    /// `x + Δ x` along `path`.
    pub fn apply(&self, path: RotationPath, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        Ok(self.apply_delta(path, x)? + &x)
    }

    /// This rotation read along `path` as a [`ResidualEdit`].
    pub fn edit(&self, path: RotationPath) -> RotationEdit<'_> {
        RotationEdit { rotation: self, path }
    }

    /// The singular value of the chord path `L(t)` on each plane,
    /// `√(1 − 2t(1 − t)(1 − cos α_j))`, which is `|cos(α_j/2)|` at `t = ½`.
    ///
    /// The radicand is evaluated as a sum of non-negative terms, so nothing
    /// cancels: `(1 − 2t)² + 4t(1 − t) cos²(α/2)` where `t(1 − t) ≥ 0`, and
    /// `1 + 4t(t − 1) sin²(α/2)` elsewhere.
    pub fn chord_radii(&self, t: f64) -> Array1<f64> {
        self.angles.mapv(|alpha| {
            let half = 0.5 * alpha;
            let spread = t * (1.0 - t);
            let squared = if spread >= 0.0 {
                let lever = 1.0 - 2.0 * t;
                lever * lever + 4.0 * spread * half.cos().powi(2)
            } else {
                1.0 - 4.0 * spread * half.sin().powi(2)
            };
            squared.sqrt()
        })
    }

    /// `A x = U Ω Uᵀ x` with `Ω = ⊕_j α_j J`, `J = [[0, −1], [1, 0]]`: the tangent of
    /// the angle path at `t = 0`, and `W(t) = exp(tA)` for an orthonormal basis.
    ///
    /// `A` is skew for every basis, since `(U Ω Uᵀ)ᵀ = U Ωᵀ Uᵀ = −A`. The edit loop's
    /// third-order bound (P4) holds only for skew generators, whose exponentials are
    /// orthogonal, and this is the only generator the module exposes.
    pub fn apply_generator(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        check_len("rotation input", self.dim(), x.len())?;
        let internal = fast_atv(&self.basis, &x);
        let mut turned = Array1::<f64>::zeros(internal.len());
        for (j, &alpha) in self.angles.iter().enumerate() {
            turned[2 * j] = -alpha * internal[2 * j + 1];
            turned[2 * j + 1] = alpha * internal[2 * j];
        }
        Ok(fast_av(&self.basis, &turned))
    }

    /// `U (R(θ) − I) Uᵀ x` for per-plane angles `θ`.
    fn plane_delta(&self, theta: &Array1<f64>, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        check_len("rotation input", self.dim(), x.len())?;
        let internal = fast_atv(&self.basis, &x);
        let rotations = rho_so2(theta.view());
        let mut moved = Array1::<f64>::zeros(internal.len());
        for j in 0..theta.len() {
            let (first, second) = (internal[2 * j], internal[2 * j + 1]);
            moved[2 * j] = (rotations[[j, 0, 0]] - 1.0) * first + rotations[[j, 0, 1]] * second;
            moved[2 * j + 1] = rotations[[j, 1, 0]] * first + (rotations[[j, 1, 1]] - 1.0) * second;
        }
        Ok(fast_av(&self.basis, &moved))
    }
}

/// A plane rotation read along one path, as a [`ResidualEdit`]. Built by
/// [`PlaneRotation::edit`].
#[derive(Clone, Copy, Debug)]
pub struct RotationEdit<'a> {
    rotation: &'a PlaneRotation,
    path: RotationPath,
}

impl ResidualEdit for RotationEdit<'_> {
    fn dim(&self) -> usize {
        self.rotation.dim()
    }

    fn apply_delta(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, OperatorRefusal> {
        self.rotation.apply_delta(self.path, x)
    }
}

/// The terms of the composition `(I + Δ_A)(I + Δ_B) x`, with `B` applied first
/// (P4).
#[derive(Clone, Debug, PartialEq)]
pub struct ComposeTerms {
    /// `Δ_B x`.
    pub first: Array1<f64>,
    /// `Δ_A x`.
    pub second: Array1<f64>,
    /// `Δ_A Δ_B x`: induced by the composition, with no control and no code of its
    /// own.
    pub cross: Array1<f64>,
}

impl ComposeTerms {
    /// `x + Δ_A x + Δ_B x`: the Sum of the two edits, which omits the cross term.
    pub fn sum(&self, x: ArrayView1<'_, f64>) -> Array1<f64> {
        &self.first + &self.second + &x
    }

    /// `x + Δ_A x + Δ_B x + Δ_A Δ_B x`: the Compose.
    pub fn compose(&self, x: ArrayView1<'_, f64>) -> Array1<f64> {
        self.sum(x) + &self.cross
    }
}

/// Split `(I + Δ_second)(I + Δ_first) x` into its two edit terms and the induced
/// cross term.
pub fn compose_terms(
    second: &dyn ResidualEdit,
    first: &dyn ResidualEdit,
    x: ArrayView1<'_, f64>,
) -> Result<ComposeTerms, OperatorRefusal> {
    check_len("composed edit dimensions", first.dim(), second.dim())?;
    check_len("composition input", first.dim(), x.len())?;
    let first_delta = first.apply_delta(x)?;
    let second_delta = second.apply_delta(x)?;
    let cross = second.apply_delta(first_delta.view())?;
    Ok(ComposeTerms {
        first: first_delta,
        second: second_delta,
        cross,
    })
}

/// `[Δ_A, Δ_B] x = Δ_A Δ_B x − Δ_B Δ_A x`: exactly the order-swap gap
/// `(I + Δ_A)(I + Δ_B) x − (I + Δ_B)(I + Δ_A) x`, since `I` commutes with both.
pub fn commutator_apply(
    a: &dyn ResidualEdit,
    b: &dyn ResidualEdit,
    x: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, OperatorRefusal> {
    check_len("commuted edit dimensions", a.dim(), b.dim())?;
    check_len("commutator input", a.dim(), x.len())?;
    let ab = a.apply_delta(b.apply_delta(x)?.view())?;
    let ba = b.apply_delta(a.apply_delta(x)?.view())?;
    Ok(ab - &ba)
}

fn check_len(what: &'static str, expected: usize, found: usize) -> Result<(), OperatorRefusal> {
    if expected == found {
        Ok(())
    } else {
        Err(OperatorRefusal::DimensionMismatch { what, expected, found })
    }
}

fn check_square(what: &'static str, order: usize, matrix: ArrayView2<'_, f64>) -> Result<(), OperatorRefusal> {
    check_len(what, order, matrix.nrows())?;
    check_len(what, order, matrix.ncols())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::accumulation_growth;
    use ndarray::s;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn uniform_matrix(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((rows, cols));
        for entry in out.iter_mut() {
            *entry = rng.random_range(-1.0..1.0);
        }
        out
    }

    fn uniform_vector(rng: &mut StdRng, len: usize) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(len);
        for entry in out.iter_mut() {
            *entry = rng.random_range(-1.0..1.0);
        }
        out
    }

    fn norm(v: &Array1<f64>) -> f64 {
        v.dot(v).sqrt()
    }

    fn frobenius(m: &Array2<f64>) -> f64 {
        m.iter().map(|entry| entry * entry).sum::<f64>().sqrt()
    }

    /// One modified Gram–Schmidt sweep over the columns.
    fn orthogonalize_columns(q: &mut Array2<f64>) {
        for j in 0..q.ncols() {
            for i in 0..j {
                let earlier = q.column(i).to_owned();
                let projection = earlier.dot(&q.column(j));
                q.column_mut(j).scaled_add(-projection, &earlier);
            }
            let length = q.column(j).dot(&q.column(j)).sqrt();
            q.column_mut(j).mapv_inplace(|entry| entry / length);
        }
    }

    /// `columns` orthonormal columns in `ℝ^dim`: two Gram–Schmidt sweeps, so the
    /// defect is at roundoff, and `PlaneRotation::new` still measures it.
    fn orthonormal_columns(rng: &mut StdRng, dim: usize, columns: usize) -> Array2<f64> {
        let mut q = uniform_matrix(rng, dim, columns);
        orthogonalize_columns(&mut q);
        orthogonalize_columns(&mut q);
        q
    }

    /// `U_j D(φ)`: the unit vector at angle `φ` in plane `j` of `basis`.
    fn plane_vector(basis: &Array2<f64>, plane: usize, angle: f64) -> Array1<f64> {
        &basis.column(2 * plane) * angle.cos() + &basis.column(2 * plane + 1) * angle.sin()
    }

    /// Rounding band of `plane_vector` at `angle`: the angle's own sum, `cos`, `sin`,
    /// two products and an add each round once, on magnitudes at most
    /// `2 + |angle|`.
    fn plane_vector_band(angle: f64) -> f64 {
        accumulation_growth(6) * (2.0 + angle.abs())
    }

    fn mask_norm(mask: &InternalMask, gauge: &GaugeBlocks) -> f64 {
        match mask {
            InternalMask::BlockScalar(controls) => controls
                .iter()
                .enumerate()
                .map(|(k, control)| control * control * gauge.block(k).len() as f64)
                .sum::<f64>()
                .sqrt(),
            InternalMask::Operator(matrix) => frobenius(matrix),
        }
    }

    /// First-order rounding band of comparing `U M Vᵀ x` with the regauged
    /// `(U S) M' (V Ŝ⁻ᵀ)ᵀ x`.
    ///
    /// The computed inverse is the exact inverse of `S + E` with `‖E‖₂ ≤ r·ε·‖S‖₂`
    /// (the SVD backward error, [`factor_singular_band`]'s convention), formed from
    /// products of at most `2r` rounded terms whose absolute magnitudes
    /// `√r·‖Σ⁻¹‖_F·√r` majorizes. So `‖S Ŝ⁻¹ − I‖` is at most
    /// `η = κ_F·r·(ε + γ_{2r})` with `κ_F = ‖S‖_F ‖Ŝ⁻¹‖_F ≥ κ₂(S)`. A block-scalar
    /// mask meets `S Ŝ⁻¹` once and a carried operator mask twice, so `2η` covers
    /// both. The six products of the two evaluations (`Vᵀx`, `U S`, `V Ŝ⁻ᵀ`,
    /// `Ŝ⁻¹ M S`, `M z`, `U z`) each round at most `max(d_in, d_out, r)` terms per
    /// entry, on absolute magnitudes majorized by `κ_F² ‖U‖_F ‖M‖_F ‖V‖_F ‖x‖`.
    fn regauge_band(before: &FactoredOperator, gauge_change: &Array2<f64>, mask: &InternalMask, x: &Array1<f64>) -> f64 {
        let r = before.gauge.rank();
        let inverse = invert_gauge_change(gauge_change.view()).expect("the fixture's gauge change is invertible");
        let kappa = frobenius(gauge_change) * frobenius(&inverse);
        let eta = kappa * r as f64 * (f64::EPSILON + accumulation_growth(2 * r));
        let widest = before.u.nrows().max(before.v.nrows()).max(r);
        let stages = accumulation_growth(6 * widest) * kappa * kappa;
        (2.0 * eta + stages) * frobenius(&before.u) * mask_norm(mask, &before.gauge) * frobenius(&before.v) * norm(x)
    }

    /// First-order rounding band of one executed rotation edit
    /// `x + t U (R − I) Uᵀ x`, with `lever = max(1, |t|)`.
    ///
    /// `Uᵀx` rounds `d` terms per coordinate, the plane map at most three (the
    /// chord's scale included), `U z` `2k`, and the final add one, so each stage
    /// sits inside `γ_{d + 2k + 4}` times its absolute magnitude. Frobenius norms
    /// majorize those magnitudes: `‖Uᵀx‖ ≤ ‖U‖_F ‖x‖`, `‖|R − I|‖_F ≤ 4` because its
    /// entries are at most 2, and `‖U z‖ ≤ ‖U‖_F ‖z‖`. Propagating the four stage
    /// errors and the rounding of `cos` and `sin` gives `γ·(1 + 11 lever ‖U‖_F²)‖x‖`.
    /// This is rounding only. The basis defect enters through [`plane_vector_defect`]
    /// or [`loop_bar`], wherever an orthonormal-basis map is the reference.
    fn rotation_apply_band(rotation: &PlaneRotation, lever: f64, x_norm: f64) -> f64 {
        let basis_sq = frobenius(&rotation.basis).powi(2);
        let stage = accumulation_growth(rotation.dim() + 2 * rotation.plane_count() + 4);
        stage * (1.0 + 11.0 * lever * basis_sq) * x_norm
    }

    /// How far the basis defect `δ = ‖UᵀU − I‖_F` moves an executed edit of a plane
    /// vector `x = U d`, `‖d‖ = 1`, from the edit on an orthonormal basis.
    ///
    /// With `G = UᵀU`, the executed `x + t U (R − I) Uᵀ x` is
    /// `U M d + t U (R − I)(G − I) d`. Here `M` is the map on an orthonormal basis:
    /// `I + t(R − I)` on the chord, `R(tα)` on the angle path. The second term is at
    /// most `‖U‖₂ · 2 lever · δ`, with `‖R − I‖₂ ≤ 2` and `‖U‖₂² = ‖G‖₂ ≤ 1 + δ`.
    /// Norms read through `U` move as well: `‖U w‖² − ‖w‖² = wᵀ(G − I)w`, so
    /// `|‖U w‖ − ‖w‖| ≤ δ‖w‖`, and a norm comparison at radius `r` pays that twice,
    /// `2rδ`, at the call site.
    fn plane_vector_defect(rotation: &PlaneRotation, lever: f64) -> f64 {
        let delta = rotation.orthonormality_defect();
        2.0 * lever * (1.0 + delta).sqrt() * delta
    }

    /// A2 (P1) over one `GL(4)` block: after `U → U S`, `V → V S⁻ᵀ` a block-scalar
    /// mask and a carried operator mask give the same intervention, while a
    /// diagonal mask held fixed moves it. The fixed diagonal mask is the positive
    /// control the invariance bar must fail on.
    #[test]
    fn scalar_and_carried_masks_survive_a_gauge_change_and_a_fixed_diagonal_mask_moves() {
        let mut rng = StdRng::seed_from_u64(295_101);
        let (d_out, d_in, r) = (7, 6, 4);
        let gauge = GaugeBlocks::new(&[r]).expect("one block");
        let operator = FactoredOperator::new(
            uniform_matrix(&mut rng, d_out, r),
            uniform_matrix(&mut rng, d_in, r),
            gauge,
        )
        .expect("random factors have full column rank");
        let gauge_change = uniform_matrix(&mut rng, r, r);
        let x = uniform_vector(&mut rng, d_in);

        let scalar = InternalMask::BlockScalar(vec![0.37]);
        let carried_operator = InternalMask::Operator(uniform_matrix(&mut rng, r, r));
        for mask in [&scalar, &carried_operator] {
            let before = operator.apply(mask, x.view()).expect("apply");
            let (regauged, carried) = operator.regauge(gauge_change.view(), mask).expect("regauge");
            let after = regauged.apply(&carried, x.view()).expect("apply regauged");
            let moved = norm(&(&after - &before));
            let band = regauge_band(&operator, &gauge_change, mask, &x);
            assert!(
                moved <= band,
                "a covariant mask moved the intervention by {moved:e} > band {band:e}: {mask:?}"
            );
        }

        let diagonal = InternalMask::Operator(Array2::from_diag(&Array1::from(vec![1.0, 0.0, 1.0, 0.5])));
        let before = operator.apply(&diagonal, x.view()).expect("apply");
        let (regauged, carried) = operator.regauge(gauge_change.view(), &diagonal).expect("regauge");
        let band = regauge_band(&operator, &gauge_change, &diagonal, &x);
        let carried_moved = norm(&(&regauged.apply(&carried, x.view()).expect("apply") - &before));
        let fixed_moved = norm(&(&regauged.apply(&diagonal, x.view()).expect("apply") - &before));
        assert!(
            carried_moved <= band,
            "the carried diagonal mask moved the intervention by {carried_moved:e} > band {band:e}"
        );
        assert!(
            fixed_moved > band,
            "positive control: a diagonal mask held fixed under S must move the intervention, moved {fixed_moved:e} <= band {band:e}"
        );
    }

    /// P1's counterexamples are exact: the witness of a non-scalar mask is a
    /// reflection or a transposition, so `S M Sᵀ` only negates or permutes entries,
    /// and the fixed mask moves a full-rank intervention past its rounding band.
    #[test]
    fn a_non_scalar_mask_has_an_exact_gauge_counterexample() {
        let mut rng = StdRng::seed_from_u64(295_102);
        let (dim, r) = (6, 4);
        let gauge = GaugeBlocks::new(&[r]).expect("one block");
        assert_eq!(
            classify_internal_mask((Array2::<f64>::eye(r) * 0.6).view(), &gauge).expect("classify"),
            MaskGaugeVerdict::Intrinsic { controls: vec![0.6] }
        );

        let operator = FactoredOperator::new(
            uniform_matrix(&mut rng, dim, r),
            uniform_matrix(&mut rng, dim, r),
            gauge.clone(),
        )
        .expect("random factors have full column rank");
        let x = uniform_vector(&mut rng, dim);
        let diagonal = Array2::from_diag(&Array1::from(vec![1.0, 0.0, 1.0, 0.5]));
        let mut off_diagonal = Array2::<f64>::eye(r);
        off_diagonal[[0, 2]] = 0.3;
        for (mask, expected_moved, expected_shift) in [(&diagonal, (0, 0), -1.0), (&off_diagonal, (0, 2), -0.6)] {
            let verdict = classify_internal_mask(mask.view(), &gauge).expect("classify");
            assert!(
                matches!(verdict, MaskGaugeVerdict::BasisDependent { .. }),
                "a non-scalar mask on one GL(4) block was classified intrinsic: {verdict:?}"
            );
            if let MaskGaugeVerdict::BasisDependent {
                gauge_change,
                moved,
                shift,
            } = verdict
            {
                assert_eq!(moved, expected_moved);
                assert_eq!(shift, expected_shift);
                let conjugated = gauge_change.dot(mask).dot(&gauge_change.t());
                assert_eq!(conjugated[moved] - mask[moved], shift);

                let fixed = InternalMask::Operator(mask.clone());
                let before = operator.apply(&fixed, x.view()).expect("apply");
                let (regauged, carried) = operator.regauge(gauge_change.view(), &fixed).expect("regauge");
                let band = regauge_band(&operator, &gauge_change, &fixed, &x);
                let carried_moved = norm(&(&regauged.apply(&carried, x.view()).expect("apply") - &before));
                let fixed_moved = norm(&(&regauged.apply(&fixed, x.view()).expect("apply") - &before));
                assert!(carried_moved <= band, "carried mask moved {carried_moved:e} > band {band:e}");
                assert!(
                    fixed_moved > band,
                    "the witness gauge change must move the fixed mask's intervention, moved {fixed_moved:e} <= band {band:e}"
                );
            }
        }
    }

    /// Declaring the four components as separate scale gauges makes the same
    /// diagonal mask intrinsic, a diagonal gauge change leaves its intervention in
    /// place, and a gauge change that mixes the components is refused as outside
    /// the declared group.
    #[test]
    fn diagonal_masks_are_intrinsic_under_the_per_component_scale_gauge() {
        let mut rng = StdRng::seed_from_u64(295_103);
        let (dim, r) = (6, 4);
        let per_component = GaugeBlocks::new(&[1, 1, 1, 1]).expect("four scale blocks");
        let diagonal = Array2::from_diag(&Array1::from(vec![1.0, 0.0, 1.0, 0.5]));
        let controls = vec![1.0, 0.0, 1.0, 0.5];
        assert_eq!(
            classify_internal_mask(diagonal.view(), &per_component).expect("classify"),
            MaskGaugeVerdict::Intrinsic {
                controls: controls.clone()
            }
        );
        assert!(matches!(
            classify_internal_mask(diagonal.view(), &GaugeBlocks::new(&[r]).expect("one block")),
            Ok(MaskGaugeVerdict::BasisDependent { .. })
        ));

        let operator = FactoredOperator::new(
            uniform_matrix(&mut rng, dim, r),
            uniform_matrix(&mut rng, dim, r),
            per_component,
        )
        .expect("random factors have full column rank");
        let x = uniform_vector(&mut rng, dim);
        let mask = InternalMask::BlockScalar(controls);
        let scale_change = Array2::from_diag(&uniform_vector(&mut rng, r));
        let before = operator.apply(&mask, x.view()).expect("apply");
        let (regauged, carried) = operator.regauge(scale_change.view(), &mask).expect("regauge");
        let moved = norm(&(&regauged.apply(&carried, x.view()).expect("apply") - &before));
        let band = regauge_band(&operator, &scale_change, &mask, &x);
        assert!(
            moved <= band,
            "a per-component scale change moved the intervention by {moved:e} > band {band:e}"
        );

        let mixing_change = uniform_matrix(&mut rng, r, r);
        assert!(matches!(
            operator.regauge(mixing_change.view(), &mask),
            Err(OperatorRefusal::GaugeOutsideDeclaredGroup { .. })
        ));
    }

    /// A factor with an exactly dependent column is refused, and the same factor
    /// with the column restored is accepted.
    #[test]
    fn a_rank_deficient_factor_is_refused() {
        let mut rng = StdRng::seed_from_u64(295_104);
        let (dim, r) = (7, 4);
        let gauge = GaugeBlocks::new(&[r]).expect("one block");
        let independent = uniform_matrix(&mut rng, dim, r);
        let v = uniform_matrix(&mut rng, dim, r);
        let mut dependent = independent.clone();
        let doubled = &independent.column(0) * 2.0;
        dependent.column_mut(3).assign(&doubled);
        assert!(matches!(
            FactoredOperator::new(dependent, v.clone(), gauge.clone()),
            Err(OperatorRefusal::RankDeficientFactor {
                factor: "left factor",
                declared: 4,
                ..
            })
        ));
        assert!(FactoredOperator::new(independent, v, gauge).is_ok());
    }

    /// P2: along the angle path a plane vector turns by `tα` at unit norm; along the
    /// chord its midpoint is `cos(α/2)` times the bisector, `‖L(t) x‖` matches
    /// `chord_radii(t)` for `t` inside and outside `[0, 1]`, and at `t = ½` the
    /// radius is `|cos(α/2)|`. The chord's norm loss is the positive control the
    /// angle path's norm bar must fail on.
    #[test]
    fn the_chord_midpoint_contracts_by_cos_half_angle_while_the_angle_path_turns() {
        let mut rng = StdRng::seed_from_u64(295_105);
        let dim = 8;
        let basis = orthonormal_columns(&mut rng, dim, 4);
        let angles = Array1::from(vec![1.1, 2.6]);
        let rotation = PlaneRotation::new(basis.clone(), angles.clone()).expect("rotation");
        let phi = 0.4;
        let midpoint_radii = rotation.chord_radii(0.5);
        for plane in 0..rotation.plane_count() {
            let alpha = angles[plane];
            let x = plane_vector(&basis, plane, phi);
            let x_norm = norm(&x);
            let executed_band =
                rotation_apply_band(&rotation, 1.0, x_norm) + plane_vector_band(phi) + plane_vector_defect(&rotation, 1.0);

            for t in [0.25, 0.5, 1.0] {
                let turned = rotation.apply(RotationPath::Angle(t), x.view()).expect("angle path");
                let target = plane_vector(&basis, plane, phi + t * alpha);
                let band = executed_band + plane_vector_band(phi + t * alpha);
                let gap = norm(&(&turned - &target));
                assert!(gap <= band, "angle path at t={t} missed D(φ + tα) by {gap:e} > {band:e}");
            }

            let chord = rotation.apply(RotationPath::Chord(0.5), x.view()).expect("chord path");
            let bisector = plane_vector(&basis, plane, phi + 0.5 * alpha) * (0.5 * alpha).cos();
            let band = executed_band + plane_vector_band(phi + 0.5 * alpha) + accumulation_growth(2);
            let gap = norm(&(&chord - &bisector));
            assert!(gap <= band, "chord midpoint missed cos(α/2)·D(φ + α/2) by {gap:e} > {band:e}");

            let half_cos = (0.5 * alpha).cos().abs();
            assert!(
                (midpoint_radii[plane] - half_cos).abs() <= accumulation_growth(3) * half_cos,
                "chord radius at t=½ is {} but |cos(α/2)| = {half_cos}",
                midpoint_radii[plane]
            );

            let turned = rotation.apply(RotationPath::Angle(0.5), x.view()).expect("angle path");
            // The angle path's radius is 1, so reading the two norms through `U` costs `2δ`.
            let norm_band = executed_band + accumulation_growth(dim + 1) * x_norm + 2.0 * rotation.orthonormality_defect();
            let angle_norm_gap = (norm(&turned) - x_norm).abs();
            let chord_norm_gap = (norm(&chord) - x_norm).abs();
            assert!(
                angle_norm_gap <= norm_band,
                "the angle path changed the norm by {angle_norm_gap:e} > {norm_band:e}"
            );
            assert!(
                chord_norm_gap > norm_band,
                "positive control: the chord midpoint must contract the norm, gap {chord_norm_gap:e} <= {norm_band:e}"
            );

            for t in [0.3_f64, 1.7, -0.4] {
                let executed = norm(&rotation.apply(RotationPath::Chord(t), x.view()).expect("chord path"));
                let predicted = rotation.chord_radii(t)[plane] * x_norm;
                let lever = 1.0_f64.max(t.abs());
                let radius = rotation.chord_radii(t)[plane];
                let band = rotation_apply_band(&rotation, lever, x_norm)
                    + plane_vector_band(phi)
                    + plane_vector_defect(&rotation, lever)
                    + 2.0 * radius * rotation.orthonormality_defect()
                    + accumulation_growth(2 * dim + 8) * (predicted + executed);
                assert!(
                    (executed - predicted).abs() <= band,
                    "chord radius at t={t}: executed {executed} vs predicted {predicted}"
                );
            }
        }
    }

    /// A4 (P4), the cross term: Compose equals the executed composition and Sum
    /// misses it by the induced cross term (the positive control), for two rotation
    /// edits read along different paths and for a factored edit composed onto a
    /// rotation. The order-swap gap is the commutator, which is resolved for
    /// overlapping planes and inside the rounding band for disjoint ones.
    #[test]
    fn compose_carries_the_induced_cross_term_and_the_order_swap_gap_is_the_commutator() {
        let mut rng = StdRng::seed_from_u64(295_106);
        let dim = 6;
        let q = orthonormal_columns(&mut rng, dim, 4);
        let a = PlaneRotation::new(q.slice(s![.., 0..2]).to_owned(), Array1::from(vec![0.9])).expect("a");
        let b = PlaneRotation::new(q.slice(s![.., 1..3]).to_owned(), Array1::from(vec![1.3])).expect("b");
        let disjoint = PlaneRotation::new(q.slice(s![.., 2..4]).to_owned(), Array1::from(vec![-0.7])).expect("c");
        let x = uniform_vector(&mut rng, dim);
        let x_norm = norm(&x);

        let edit_a = a.edit(RotationPath::Angle(1.0));
        let edit_b = b.edit(RotationPath::Chord(0.6));
        let rotation_band = rotation_apply_band(&a, 1.0, x_norm).max(rotation_apply_band(&b, 1.0, x_norm));
        // Two executed edits against three term evaluations and two sums: every
        // stage stays inside a per-edit band on inputs of norm at most 2‖x‖, so six
        // per-edit bands majorize the comparison.
        let band = 6.0 * rotation_band;

        let executed_b = b.apply(RotationPath::Chord(0.6), x.view()).expect("b");
        let executed = a.apply(RotationPath::Angle(1.0), executed_b.view()).expect("a");
        let terms = compose_terms(&edit_a, &edit_b, x.view()).expect("compose terms");
        let compose_gap = norm(&(&terms.compose(x.view()) - &executed));
        let sum_gap = norm(&(&terms.sum(x.view()) - &executed));
        assert!(
            compose_gap <= band,
            "Compose missed the executed composition by {compose_gap:e} > {band:e}"
        );
        assert!(
            sum_gap > band,
            "positive control: Sum must miss the executed composition by the cross term, gap {sum_gap:e} <= {band:e}"
        );
        assert!((sum_gap - norm(&terms.cross)).abs() <= band);

        let executed_a = a.apply(RotationPath::Angle(1.0), x.view()).expect("a");
        let swapped = b.apply(RotationPath::Chord(0.6), executed_a.view()).expect("b");
        let commutator = commutator_apply(&edit_a, &edit_b, x.view()).expect("commutator");
        let swap_gap = norm(&(&(&executed - &swapped) - &commutator));
        assert!(
            swap_gap <= 2.0 * band,
            "the order-swap gap missed the commutator by {swap_gap:e} > {:e}",
            2.0 * band
        );
        assert!(
            norm(&commutator) > 2.0 * band,
            "positive control: edits in overlapping planes must not commute"
        );

        let edit_disjoint = disjoint.edit(RotationPath::Angle(1.0));
        let commuting = commutator_apply(&edit_a, &edit_disjoint, x.view()).expect("commutator");
        // `Δ_A Δ_C = U_A (R_A − I) U_Aᵀ U_C (R_C − I) U_Cᵀ`, so the commutator is at most
        // `2 ‖U_A‖₂ · 2 · ‖U_Aᵀ U_C‖₂ · 2 · ‖U_C‖₂ ‖x‖`. The measured cross block of the
        // two bases decides how far disjoint planes commute; neither basis's own
        // defect does.
        let cross_block = frobenius(&fast_atb(&a.basis, &disjoint.basis));
        let disjoint_band =
            2.0 * band + 8.0 * frobenius(&a.basis) * frobenius(&disjoint.basis) * cross_block * x_norm;
        assert!(
            norm(&commuting) <= disjoint_band,
            "edits in disjoint planes must commute, commutator {:e} > {disjoint_band:e}",
            norm(&commuting)
        );

        let gauge = GaugeBlocks::new(&[2]).expect("one block");
        let factored = FactoredOperator::new(
            uniform_matrix(&mut rng, dim, 2),
            uniform_matrix(&mut rng, dim, 2),
            gauge,
        )
        .expect("random factors have full column rank");
        let factored_mask = InternalMask::Operator(uniform_matrix(&mut rng, 2, 2));
        let factored_edit = factored.residual_edit(&factored_mask).expect("square operator");
        let factored_executed = &executed_b + &factored.apply(&factored_mask, executed_b.view()).expect("factored");
        let factored_terms = compose_terms(&factored_edit, &edit_b, x.view()).expect("terms");
        let factored_norm = frobenius(&factored.u) * mask_norm(&factored_mask, factored.gauge()) * frobenius(&factored.v);
        // The factored edit's stages round at most `dim` terms on magnitudes the
        // product of its Frobenius norms majorizes, on inputs of norm at most 2‖x‖.
        let factored_band = band * (1.0 + factored_norm);
        let factored_gap = norm(&(&factored_terms.compose(x.view()) - &factored_executed));
        assert!(
            factored_gap <= factored_band,
            "Compose with a factored edit missed by {factored_gap:e} > {factored_band:e}"
        );
        assert!(
            norm(&factored_terms.cross) > factored_band,
            "positive control: the factored edit's cross term must be resolved"
        );
    }

    /// `e^{sA} e^{tB} e^{−sA} e^{−tB} x`, executed along angle paths.
    fn edit_loop(a: &PlaneRotation, b: &PlaneRotation, s: f64, t: f64, x: &Array1<f64>) -> Array1<f64> {
        let step1 = b.apply(RotationPath::Angle(-t), x.view()).expect("e^{-tB}");
        let step2 = a.apply(RotationPath::Angle(-s), step1.view()).expect("e^{-sA}");
        let step3 = b.apply(RotationPath::Angle(t), step2.view()).expect("e^{tB}");
        a.apply(RotationPath::Angle(s), step3.view()).expect("e^{sA}")
    }

    /// The loop bar `3(s²|t| ‖A‖²‖B‖ + |s|t² ‖A‖‖B‖²)‖x‖`, valid ONLY for skew `A` and
    /// `B`, whose exponentials are orthogonal.
    ///
    /// With `P = e^{sA}` and `Q = e^{tB}`, `F − I = [P, Q] P⁻¹ Q⁻¹`, so
    /// `F − I − st[A, B] = [P, Q](P⁻¹Q⁻¹ − I) + [P − I − sA, Q − I] + [sA, Q − I − tB]`.
    /// For a skew generator `‖e^{sA} − I‖ ≤ |s|‖A‖` and `‖e^{sA} − I − sA‖ ≤ s²‖A‖²/2`.
    /// So `‖[P, Q]‖ = ‖[P − I, Q − I]‖ ≤ 2|st|‖A‖‖B‖` and `‖P⁻¹Q⁻¹ − I‖ ≤ |s|‖A‖ + |t|‖B‖`,
    /// and the last two commutators are at most `s²|t|‖A‖²‖B‖` and `|s|t²‖A‖‖B‖²`: three
    /// of each monomial in total (mpd-verify's route, #2951). A generator that is not
    /// skew breaks the first two inequalities; see
    /// `the_loop_bar_fails_for_non_skew_generators`.
    fn skew_loop_bar(s: f64, t: f64, norm_a: f64, norm_b: f64, x_norm: f64) -> f64 {
        3.0 * (s * s * t.abs() * norm_a * norm_a * norm_b + s.abs() * t * t * norm_a * norm_b * norm_b) * x_norm
    }

    /// The derived bar of the executed loop against `x + st[A, B] x`, where
    /// `A = U_A Ω_A U_Aᵀ` and each factor executes as `I + U (R(sα) − I) Uᵀ`.
    ///
    /// Let `δ = ‖UᵀU − I‖_F`, `U = Ũ H` with `H = (UᵀU)^{1/2}`, so `‖H − I‖₂ ≤ δ`. The
    /// bar has three terms.
    /// - **Skew bar.** [`skew_loop_bar`] holds for the skew `A`, using
    ///   `‖A‖₂ ≤ ‖U‖₂² ‖Ω‖ ≤ (1 + δ) max_j |α_j|`.
    /// - **Executed factors versus exponentials.** An executed factor is within
    ///   `‖H (R − I) H − (R − I)‖ ≤ ‖R − I‖ δ(2 + δ) ≤ |s| ‖Ω‖ δ(2 + δ)` of the orthogonal
    ///   `I + Ũ (R(sα) − I) Ũᵀ = e^{s Ũ Ω Ũᵀ}`.
    ///   - That is within `|s| ‖Ω − HΩH‖ ≤ |s| ‖Ω‖ δ(2 + δ)` of `e^{sA}`, because both
    ///     generators are skew.
    ///   - So each factor is within `ε = 2|s| ‖Ω‖ δ(2 + δ)` of its exponential.
    ///   - Telescoping the four factors against orthogonal exponentials gives
    ///     `2(ε_A + ε_B)(1 + max ε)³ ‖x‖`. The metric defect enters through `ε`, not with
    ///     coefficient 1 (mpd-verify's question, #2951).
    /// - **Rounding.** Six per-edit bands majorize the four executed edits, the
    ///   commutator's four generator applications scaled by `st ≤ 1`, and the differences.
    fn loop_bar(a: &PlaneRotation, b: &PlaneRotation, s: f64, t: f64, x_norm: f64) -> f64 {
        let omega_a = a.angles.iter().fold(0.0_f64, |largest, angle| largest.max(angle.abs()));
        let omega_b = b.angles.iter().fold(0.0_f64, |largest, angle| largest.max(angle.abs()));
        let (delta_a, delta_b) = (a.orthonormality_defect(), b.orthonormality_defect());
        let eps_a = 2.0 * s.abs() * omega_a * delta_a * (2.0 + delta_a);
        let eps_b = 2.0 * t.abs() * omega_b * delta_b * (2.0 + delta_b);
        let executed = 2.0 * (eps_a + eps_b) * (1.0 + eps_a.max(eps_b)).powi(3) * x_norm;
        let rounding = 6.0 * rotation_apply_band(a, 1.0, x_norm).max(rotation_apply_band(b, 1.0, x_norm));
        skew_loop_bar(s, t, omega_a * (1.0 + delta_a), omega_b * (1.0 + delta_b), x_norm) + executed + rounding
    }

    /// A4 (P4), the loop: for the generators of two plane rotations,
    /// `e^{sA} e^{tB} e^{−sA} e^{−tB} x − x − st[A, B] x` is inside [`loop_bar`].
    /// `apply_generator` builds `A = U Ω Uᵀ` with `Ω` skew, so `A` is skew for every basis
    /// and [`skew_loop_bar`] applies; the basis defect enters through `loop_bar`'s
    /// executed-factor term. Dropping the commutator term is the positive control: at the smallest
    /// step the second-order term of a vector the commutator moves is outside the
    /// third-order bar.
    #[test]
    fn the_edit_loop_remainder_is_inside_its_third_order_bar() {
        let mut rng = StdRng::seed_from_u64(295_107);
        let dim = 5;
        let q = orthonormal_columns(&mut rng, dim, 3);
        let a = PlaneRotation::new(q.slice(s![.., 0..2]).to_owned(), Array1::from(vec![1.0])).expect("a");
        let b = PlaneRotation::new(q.slice(s![.., 1..3]).to_owned(), Array1::from(vec![-0.8])).expect("b");
        let steps = [0.2, 0.1, 0.05, 0.025, 0.0125];
        let random_x = uniform_vector(&mut rng, dim);
        // [A, B] = αβ (u₀u₂ᵀ − u₂u₀ᵀ), so u₂ is moved by the full |αβ|.
        let aligned_x = q.column(2).to_owned();

        for x in [&random_x, &aligned_x] {
            let x_norm = norm(x);
            let bracket = &a.apply_generator(b.apply_generator(x.view()).expect("Bx").view()).expect("ABx")
                - &b.apply_generator(a.apply_generator(x.view()).expect("Ax").view()).expect("BAx");
            for h in steps {
                let (s, t) = (h, 0.6 * h);
                let looped = edit_loop(&a, &b, s, t, x);
                let remainder = norm(&(&(&looped - x) - &(&bracket * (s * t))));
                let bar = loop_bar(&a, &b, s, t, x_norm);
                assert!(
                    remainder <= bar,
                    "loop remainder at h={h}: {remainder:e} > third-order bar {bar:e}"
                );
            }
        }

        let x_norm = norm(&aligned_x);
        let (s, t) = (steps[steps.len() - 1], 0.6 * steps[steps.len() - 1]);
        let bar = loop_bar(&a, &b, s, t, x_norm);
        let without_commutator = norm(&(&edit_loop(&a, &b, s, t, &aligned_x) - &aligned_x));
        assert!(
            without_commutator > bar,
            "positive control: without the commutator term the loop must leave the third-order bar, {without_commutator:e} <= {bar:e}"
        );
    }

    /// The skew premise of [`skew_loop_bar`] is load-bearing (mpd-verify's
    /// counterexample, #2951). Take `A = diag(λ, 0)` and `B = e₁e₂ᵀ`.
    /// - `e^{sA} = diag(e^{sλ}, 1)`, and `e^{tB} = I + tB` because `B² = 0`.
    /// - So `e^{sA} B e^{−sA} = e^{sλ} B`, and the loop is `I + t(e^{sλ} − 1) B`.
    /// - `[A, B] = λB`, so the remainder is `t(e^{sλ} − 1 − sλ) B`.
    ///
    /// At `sλ = 5` the remainder is about `142 t`. That is outside both the constant-4
    /// bar that gate 1175070 tested and [`skew_loop_bar`]: a bound derived for
    /// orthogonal factors does not hold for these generators. The exponentials are the
    /// closed forms of a diagonal and a nilpotent matrix, written out in the fixture.
    #[test]
    fn the_loop_bar_fails_for_non_skew_generators() {
        let (lambda, s, t) = (1.0_f64, 5.0_f64, 1e-3_f64);
        let (norm_a, norm_b) = (lambda.abs(), 1.0_f64);
        let mut a = Array2::<f64>::zeros((2, 2));
        a[[0, 0]] = lambda;
        let mut b = Array2::<f64>::zeros((2, 2));
        b[[0, 1]] = 1.0;
        let exp_a = |scale: f64| Array2::from_diag(&Array1::from(vec![(scale * lambda).exp(), 1.0]));
        let exp_b = |scale: f64| Array2::<f64>::eye(2) + &b * scale;
        let looped = exp_a(s).dot(&exp_b(t)).dot(&exp_a(-s)).dot(&exp_b(-t));
        let bracket = a.dot(&b) - b.dot(&a);
        let x = Array1::from(vec![0.0, 1.0]);
        let x_norm = norm(&x);
        let remainder = norm(&(&(&looped.dot(&x) - &x) - &(bracket.dot(&x) * (s * t))));
        let predicted = t * ((s * lambda).exp() - 1.0 - s * lambda);
        // Rounding: seven 2×2 products and three differences, each at most two
        // rounded terms per entry. The factors' absolute magnitudes are majorized by
        // `e^{|sλ|}(1 + t)²`.
        let rounding = accumulation_growth(32) * (s * lambda).abs().exp() * (1.0 + t).powi(2) * x_norm;
        assert!(
            (remainder - predicted).abs() <= rounding + accumulation_growth(4) * predicted,
            "executed non-skew remainder {remainder:e} vs closed form t(e^(sλ) − 1 − sλ) = {predicted:e}"
        );
        let constant_four_bar = 4.0 * (s * s * t * norm_a * norm_a * norm_b + s * t * t * norm_a * norm_b * norm_b) * x_norm;
        let skew_bar = skew_loop_bar(s, t, norm_a, norm_b, x_norm);
        assert!(
            remainder > constant_four_bar + rounding,
            "positive control: non-skew generators must leave the constant-4 bar, {remainder:e} <= {constant_four_bar:e}"
        );
        assert!(
            remainder > skew_bar + rounding,
            "positive control: non-skew generators must leave the skew bar, {remainder:e} <= {skew_bar:e}"
        );
    }
}
