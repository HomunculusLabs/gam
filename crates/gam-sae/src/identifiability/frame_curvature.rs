//! The residual-gauge curvature operator, held in the structure the
//! decoder-frame parameterization actually gives it (#2757).
//!
//! # The theorem this module is built on
//!
//! The certificate's free-parameter vector is the concatenation of the fitted
//! atoms' flattened frames, `vec(frame_0) ⊕ vec(frame_1) ⊕ …`, each
//! `frame_k ∈ ℝ^{p × d_k}` stored row-major. So a parameter index decomposes
//! exactly as
//!
//! ```text
//! c = offset_k + i·d_k + a        atom k, OUTPUT coordinate i, frame axis a
//! ```
//!
//! and the per-row pinning Jacobian is, by construction,
//!
//! ```text
//! J_n[i, (k, i', a)] = δ_{i,i'} · a_{nk} · ∂g_k/∂t_a(n)[i]
//! ```
//!
//! — a frame perturbation of output coordinate `i'` moves the reconstruction
//! only on output coordinate `i'`. `J_n` is therefore **output-coordinate
//! diagonal**: it has `p·D` nonzeros (`D = Σ_k d_k`), not `p · param_dim`.
//!
//! The data curvature `H = Σ_n J_nᵀ M_n J_n` inherits from that exactly the
//! output-coordinate coupling the *metric* has and nothing else:
//!
//! ```text
//! H[(k,i,a), (k',i',a')] = Σ_n M_n[i,i'] · g_n[i,(k,a)] · g_n[i',(k',a')]
//! ```
//!
//! So when every per-row metric is diagonal in the output coordinates — in
//! particular when it is the identity, which is what
//! `SaeManifoldTerm::diagnostic_metric` installs whenever no output-Fisher
//! harvest ran — `H` is **exactly block diagonal**, `p` blocks of size
//! `D × D`, and the off-block entries are not small but structurally never
//! written.
//!
//! Holding that object as a dense `param_dim × param_dim = (p·D)²` matrix and
//! taking its dense symmetric eigendecomposition costs `(p·D)²` memory and
//! `(p·D)³` flops for a quantity that is `p·D²` numbers and `p·D³` flops — a
//! factor of `p` in memory and `p²` in time. At the `p = 4096` shape #2757 was
//! filed on that is 45.1 GiB and ~1.7·10⁷× the necessary work.
//!
//! # What this module provides
//!
//! [`FrameColumnLayout`] — the `c ↔ (i, l)` bijection above, the single place
//! the frame-column index arithmetic is written down — and
//! [`ResidualGaugeCurvature`], the curvature as the streaming builder is able
//! to produce it: output-coordinate blocks when the metric does not couple
//! output coordinates, the stacked root `R` when `H = RᵀR` has fewer rows than
//! columns (its nonzero spectrum is the dual Gram `RRᵀ`'s, so the same rank
//! decision costs `m³` instead of `param_dim³`), and the dense Gram only when
//! neither applies.

use ndarray::{Array2, Array3, ArrayView1, ArrayViewMut2, s};

/// One atom's slot in the joint frame-column layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameAtomSlot {
    /// `offset_k` — the atom's first column in the joint parameter vector.
    offset: usize,
    /// `d_k` — the atom's frame axis count (its latent dimension).
    axes: usize,
    /// `dstart_k = Σ_{k' < k} d_{k'}` — the atom's first local axis index.
    axis_start: usize,
}

/// A local axis index `l ∈ [0, D)` resolved to the atom slot it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LocalAxis {
    /// `offset_k` of the owning atom.
    offset: usize,
    /// `d_k` of the owning atom.
    axes: usize,
    /// `a = l − dstart_k`, the axis within that atom's frame.
    axis: usize,
}

/// The joint parameter vector's frame-column layout: the bijection between a
/// parameter index `c` and the pair `(i, l)` of OUTPUT coordinate
/// `i ∈ [0, p)` and LOCAL frame axis `l ∈ [0, D)`, `D = Σ_k d_k`.
///
/// This is the whole content of "the pinning Jacobian is output-coordinate
/// diagonal": every consumer that wants to group parameters by the output
/// coordinate they drive reads the grouping from here rather than
/// re-deriving `offset_k + i·d_k + a` for itself.
///
/// A layout exists only when every atom's frame has the same height `p` —
/// which the fitted model always satisfies, since each frame is built as
/// `(p, d_k)` against the term's own output dimension. A model whose frames
/// disagree in height has no common output coordinate to group by, and
/// [`FrameColumnLayout::for_frames`] reports `None` rather than inventing one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameColumnLayout {
    p: usize,
    d_total: usize,
    param_dim: usize,
    atoms: Vec<FrameAtomSlot>,
    locals: Vec<LocalAxis>,
}

impl FrameColumnLayout {
    /// The layout for atoms whose frames are `(p, axes_per_atom[k])`.
    ///
    /// `p == 0` or an empty atom list still yields a valid (degenerate) layout:
    /// `param_dim == 0`, which every consumer already treats as the empty
    /// certificate.
    pub fn new(p: usize, axes_per_atom: &[usize]) -> Self {
        let mut atoms = Vec::with_capacity(axes_per_atom.len());
        let mut locals = Vec::new();
        let mut offset = 0usize;
        let mut axis_start = 0usize;
        for &axes in axes_per_atom {
            atoms.push(FrameAtomSlot {
                offset,
                axes,
                axis_start,
            });
            for axis in 0..axes {
                locals.push(LocalAxis { offset, axes, axis });
            }
            offset += p * axes;
            axis_start += axes;
        }
        Self {
            p,
            d_total: axis_start,
            param_dim: offset,
            atoms,
            locals,
        }
    }

    /// The layout of a fitted model's frames, or `None` when the frames do not
    /// share one output dimension (no common coordinate to group by).
    pub fn for_frames<'a>(frames: impl IntoIterator<Item = &'a Array2<f64>>) -> Option<Self> {
        let dims: Vec<(usize, usize)> = frames.into_iter().map(|f| f.dim()).collect();
        // An atom with no axes contributes no columns, so its (vacuous) frame
        // height must not veto the layout the remaining atoms agree on.
        let mut p: Option<usize> = None;
        for &(rows, cols) in &dims {
            if cols == 0 {
                continue;
            }
            match p {
                None => p = Some(rows),
                Some(known) if known == rows => {}
                Some(_) => return None,
            }
        }
        let p = p.unwrap_or(0);
        let axes: Vec<usize> = dims.iter().map(|&(_, cols)| cols).collect();
        Some(Self::new(p, &axes))
    }

    /// Output dimension `p`.
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.p
    }

    /// `D = Σ_k d_k` — the block size, i.e. the number of frame axes that
    /// share each output coordinate.
    #[inline]
    pub fn block_dim(&self) -> usize {
        self.d_total
    }

    /// `param_dim = p · D`.
    #[inline]
    pub fn param_dim(&self) -> usize {
        self.param_dim
    }

    /// Number of atoms.
    #[inline]
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Atom `k`'s first local axis index `dstart_k`, so its axis `a` is local
    /// axis `local_axis_base(k) + a`.
    #[inline]
    pub fn local_axis_base(&self, atom: usize) -> usize {
        self.atoms[atom].axis_start
    }

    /// The parameter index `c` of `(output coordinate i, local axis l)`.
    #[inline]
    pub fn column(&self, output: usize, local: usize) -> usize {
        let slot = self.locals[local];
        slot.offset + output * slot.axes + slot.axis
    }

    /// The output coordinate a parameter index drives, or `None` when the
    /// index is out of range.
    pub fn output_of(&self, column: usize) -> Option<usize> {
        let slot = self
            .atoms
            .iter()
            .rev()
            .find(|slot| slot.axes > 0 && column >= slot.offset)?;
        if column >= slot.offset + self.p * slot.axes {
            return None;
        }
        Some((column - slot.offset) / slot.axes)
    }

    /// Gather the `D` entries of `v` that live on output coordinate `output`
    /// into `out` (length `D`), in local-axis order.
    #[inline]
    pub fn gather_output(&self, v: ArrayView1<'_, f64>, output: usize, out: &mut [f64]) {
        for (local, slot) in self.locals.iter().enumerate() {
            out[local] = v[slot.offset + output * slot.axes + slot.axis];
        }
    }
}

/// Fold one row `v` into the upper-triangular factor `r` (`D × D`) by Givens
/// rotations, so that afterwards `rᵀr` equals the old `rᵀr + vvᵀ`.
///
/// This is the whole reason the block accumulator keeps a *root* rather than a
/// Gram. The certificate's rank decision is
/// `σ_i(R) > α·ε·max(m, param_dim)·max(σ_max, 1)` — a statement about singular
/// values of `R`. Accumulating `RᵀR` and recovering `σ = √λ` squares the
/// condition number: eigenvalues below `ε·λ_max` are indistinguishable from the
/// decomposition's own roundoff, so every singular value below `√ε·σ_max ≈
/// 1.5e-8·σ_max` is unresolvable while the tolerance asks about ones near
/// `1e-12·σ_max`. Rotating rows in instead keeps `σ` to full precision at the
/// same `O(D²)` per row and the same `p·D²` storage — the cheap representation
/// is also the accurate one.
///
/// `v` is consumed (left as the annihilated residual) so the caller can reuse
/// one buffer.
fn fold_row_into_triangular_factor(r: &mut ArrayViewMut2<'_, f64>, v: &mut [f64]) {
    let d = v.len();
    for j in 0..d {
        let vj = v[j];
        if vj == 0.0 {
            continue;
        }
        let rjj = r[[j, j]];
        let norm = rjj.hypot(vj);
        if norm == 0.0 {
            continue;
        }
        let (c, s) = (rjj / norm, vj / norm);
        r[[j, j]] = norm;
        v[j] = 0.0;
        for k in (j + 1)..d {
            let rjk = r[[j, k]];
            let vk = v[k];
            r[[j, k]] = c * rjk + s * vk;
            v[k] = c * vk - s * rjk;
        }
    }
}

/// Streaming accumulator for the output-coordinate block roots.
///
/// One `D × D` upper-triangular factor per output coordinate, grown by
/// [`fold_row_into_triangular_factor`] as observations arrive. Memory `p·D²`
/// and cost `O(rows · p · D²)` — the same as accumulating the blocks' Grams,
/// without the squaring.
pub struct OutputBlockRootAccumulator {
    roots: Array3<f64>,
    layout: FrameColumnLayout,
    scratch: Vec<f64>,
}

impl OutputBlockRootAccumulator {
    pub fn new(layout: FrameColumnLayout) -> Self {
        let d = layout.block_dim();
        Self {
            roots: Array3::<f64>::zeros((layout.output_dim(), d, d)),
            layout,
            scratch: vec![0.0_f64; d],
        }
    }

    /// Fold one observation's frame Jacobian `g` (`p × D`, `g[i, l]`) in: it
    /// contributes the rank-one term `g(i)g(i)ᵀ` to output coordinate `i`'s
    /// block and nothing anywhere else.
    pub fn push_row_jacobian(&mut self, g: &Array2<f64>) {
        let d = self.layout.block_dim();
        for i in 0..self.layout.output_dim() {
            let mut any = false;
            for l in 0..d {
                let v = g[[i, l]];
                self.scratch[l] = v;
                any |= v != 0.0;
            }
            if !any {
                continue;
            }
            let mut block = self.roots.slice_mut(s![i, .., ..]);
            fold_row_into_triangular_factor(&mut block, &mut self.scratch);
        }
    }

    pub fn finish(self, root_rows: usize) -> ResidualGaugeCurvature {
        ResidualGaugeCurvature::OutputBlockRoots {
            roots: self.roots,
            layout: self.layout,
            root_rows,
        }
    }
}

/// The residual-gauge curvature `H = RᵀR` as the streaming builder produced it.
///
/// Every variant carries `root_rows` — the row count `R` would have had — which
/// is the scale the rank tolerance is calibrated against, and is therefore part
/// of the object rather than a parameter threaded beside it.
///
/// Two of the three variants hold a *root* rather than a Gram. That is not an
/// implementation preference: the certificate's rank decision is about singular
/// values of `R`, and forming `RᵀR` squares the condition number before that
/// decision is taken. [`Self::DenseGram`] is the only representation that
/// cannot avoid it, and it is also the only one that costs `param_dim³`.
pub enum ResidualGaugeCurvature {
    /// `p` independent `D × D` upper-triangular roots, one per output
    /// coordinate, in `layout`'s `(i, l)` coordinates:
    /// `roots[i]ᵀ roots[i] = H[(i,·), (i,·)]`, and `H` has no entries between
    /// two output coordinates at all.
    ///
    /// Produced exactly when the per-row metric does not couple output
    /// coordinates.
    OutputBlockRoots {
        roots: Array3<f64>,
        layout: FrameColumnLayout,
        root_rows: usize,
    },
    /// The stacked root `R ∈ ℝ^{m × param_dim}` with `m ≤ param_dim`.
    ///
    /// `H = RᵀR` has `param_dim − m` structural zero eigenvalues and `R`'s `m`
    /// singular values, so the rank decision and `σ_max` come from an `m`-sized
    /// decomposition instead of a `param_dim`-sized one.
    DualRoot { root: Array2<f64>, root_rows: usize },
    /// The dense Gram, for a curvature with neither structure.
    ///
    /// The only variant whose rank decision must be taken on `λ = σ²`, and
    /// therefore the only one that cannot resolve a singular value below
    /// `√ε·σ_max`. It is reached only when the root is the *larger* object,
    /// which is also when there is nothing cheaper to decompose.
    DenseGram { gram: Array2<f64>, root_rows: usize },
}

impl ResidualGaugeCurvature {
    /// The row count of the root `R` this curvature came from.
    pub fn root_rows(&self) -> usize {
        match self {
            Self::OutputBlockRoots { root_rows, .. }
            | Self::DualRoot { root_rows, .. }
            | Self::DenseGram { root_rows, .. } => *root_rows,
        }
    }

    /// How many `f64` this representation holds.
    ///
    /// The certificate's cost claim is a claim about this number: the
    /// output-coordinate roots are `p·D²` scalars where the dense Gram is
    /// `(p·D)²`, a factor of `p`. It is exact and load-immune, which is what
    /// makes it the regression gate a wall-clock threshold cannot be.
    pub fn stored_scalars(&self) -> usize {
        match self {
            Self::OutputBlockRoots { roots, .. } => roots.len(),
            Self::DualRoot { root, .. } => root.len(),
            Self::DenseGram { gram, .. } => gram.len(),
        }
    }

    /// Whether every stored entry is finite.
    ///
    /// The certificate's refusal contract: a non-finite curvature must produce
    /// a typed error, not a spectrum. Checked on the stored representation
    /// rather than on the fit that produced it, so it holds for every builder.
    pub fn is_finite(&self) -> bool {
        match self {
            Self::OutputBlockRoots { roots, .. } => roots.iter().all(|v| v.is_finite()),
            Self::DualRoot { root, .. } => root.iter().all(|v| v.is_finite()),
            Self::DenseGram { gram, .. } => gram.iter().all(|v| v.is_finite()),
        }
    }

    /// A stable tag for the representation, for diagnostics and gates.
    pub fn structure_tag(&self) -> &'static str {
        match self {
            Self::OutputBlockRoots { .. } => "output_block_roots",
            Self::DualRoot { .. } => "dual_root",
            Self::DenseGram { .. } => "dense_gram",
        }
    }

    /// The parameter dimension this curvature is defined over.
    pub fn param_dim(&self) -> usize {
        match self {
            Self::OutputBlockRoots { layout, .. } => layout.param_dim(),
            Self::DualRoot { root, .. } => root.ncols(),
            Self::DenseGram { gram, .. } => gram.ncols(),
        }
    }

    /// Materialize the dense Gram this curvature represents.
    ///
    /// **Not for production use** — it is exactly the `param_dim²` object the
    /// structured forms exist to avoid. It is the equivalence witness: a test
    /// can assert that a structured curvature and the dense path agree entry by
    /// entry, which is what makes "the off-block entries are structurally zero"
    /// a checked claim rather than an argued one.
    pub fn to_dense_gram(&self) -> Array2<f64> {
        match self {
            Self::OutputBlockRoots { roots, layout, .. } => {
                let n = layout.param_dim();
                let d = layout.block_dim();
                let mut gram = Array2::<f64>::zeros((n, n));
                for i in 0..layout.output_dim() {
                    let block = roots.slice(s![i, .., ..]);
                    let dense = block.t().dot(&block);
                    for a in 0..d {
                        let ca = layout.column(i, a);
                        for b in 0..d {
                            gram[[ca, layout.column(i, b)]] = dense[[a, b]];
                        }
                    }
                }
                gram
            }
            Self::DualRoot { root, .. } => root.t().dot(root),
            Self::DenseGram { gram, .. } => gram.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_column_and_output_are_inverse() {
        let layout = FrameColumnLayout::new(5, &[2, 1, 3]);
        assert_eq!(layout.param_dim(), 5 * 6);
        assert_eq!(layout.block_dim(), 6);
        for i in 0..5 {
            for l in 0..6 {
                let c = layout.column(i, l);
                assert_eq!(layout.output_of(c), Some(i), "column {c} = (i={i}, l={l})");
            }
        }
        // The map is a bijection onto [0, param_dim).
        let mut seen = vec![false; layout.param_dim()];
        for i in 0..5 {
            for l in 0..6 {
                let c = layout.column(i, l);
                assert!(!seen[c], "column {c} produced twice");
                seen[c] = true;
            }
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn layout_matches_the_certificate_index_arithmetic() {
        // `offset_k + i·d_k + a`, written out independently.
        let axes = [2usize, 1, 3];
        let p = 4;
        let layout = FrameColumnLayout::new(p, &axes);
        let mut offset = 0usize;
        for (k, &d) in axes.iter().enumerate() {
            for i in 0..p {
                for a in 0..d {
                    let expected = offset + i * d + a;
                    let local = layout.local_axis_base(k) + a;
                    assert_eq!(layout.column(i, local), expected);
                }
            }
            offset += p * d;
        }
    }

    #[test]
    fn layout_for_frames_rejects_disagreeing_frame_heights() {
        let a = Array2::<f64>::zeros((4, 2));
        let b = Array2::<f64>::zeros((3, 1));
        assert!(FrameColumnLayout::for_frames([&a, &b]).is_none());
        let c = Array2::<f64>::zeros((4, 1));
        let layout = FrameColumnLayout::for_frames([&a, &c]).expect("agreeing heights");
        assert_eq!(layout.output_dim(), 4);
        assert_eq!(layout.block_dim(), 3);
        assert_eq!(layout.param_dim(), 12);
    }

    #[test]
    fn layout_for_frames_ignores_an_axisless_atom_height() {
        let empty = Array2::<f64>::zeros((0, 0));
        let real = Array2::<f64>::zeros((6, 2));
        let layout = FrameColumnLayout::for_frames([&empty, &real]).expect("layout");
        assert_eq!(layout.output_dim(), 6);
        assert_eq!(layout.block_dim(), 2);
        assert_eq!(layout.param_dim(), 12);
        assert_eq!(layout.local_axis_base(1), 0);
    }

    #[test]
    fn folding_rows_into_a_triangular_factor_reproduces_their_gram() {
        // The Givens accumulator's whole contract: after folding rows
        // `v_1 … v_m`, `rᵀr = Σ_j v_j v_jᵀ` — exactly the Gram it replaces,
        // without ever forming it.
        let d = 5usize;
        let mut seed = 0x2757_ACC0_0000_0001u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let rows: Vec<Vec<f64>> = (0..17).map(|_| (0..d).map(|_| next()).collect()).collect();
        let mut expected = Array2::<f64>::zeros((d, d));
        for row in &rows {
            for a in 0..d {
                for b in 0..d {
                    expected[[a, b]] += row[a] * row[b];
                }
            }
        }
        let mut factor = Array2::<f64>::zeros((d, d));
        for row in &rows {
            let mut v = row.clone();
            fold_row_into_triangular_factor(&mut factor.view_mut(), &mut v);
        }
        // Upper triangular by construction.
        for a in 0..d {
            for b in 0..a {
                assert_eq!(factor[[a, b]], 0.0, "({a},{b}) below the diagonal");
            }
        }
        let recovered = factor.t().dot(&factor);
        let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let worst = recovered
            .iter()
            .zip(expected.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst <= 1.0e-13 * scale,
            "rᵀr must equal the accumulated Gram: worst |Δ| {worst:.3e} against {scale:.3e}"
        );
    }

    #[test]
    fn folding_survives_magnitudes_that_overflow_a_gram() {
        // `hypot` is not a convenience here: a Gram accumulation squares its
        // entries, so a decoder tangent near `1e200` overflows to infinity
        // before any decomposition sees it. The root accumulation never
        // squares, so the same rows stay finite and the singular value is
        // recovered exactly.
        let mut factor = Array2::<f64>::zeros((2, 2));
        for scale in [1.0e200_f64, 2.0e200] {
            let mut v = vec![scale, 0.0];
            fold_row_into_triangular_factor(&mut factor.view_mut(), &mut v);
        }
        assert!(
            factor.iter().all(|v| v.is_finite()),
            "the triangular factor must stay finite where the Gram would overflow"
        );
        let expected = (1.0e200_f64).hypot(2.0e200);
        assert!(
            ((factor[[0, 0]] - expected) / expected).abs() <= 1.0e-14,
            "σ = {} against {expected}",
            factor[[0, 0]]
        );
        assert!(
            !(1.0e200_f64 * 1.0e200).is_finite(),
            "the Gram entry does overflow"
        );
    }

    #[test]
    fn is_finite_refuses_a_nan_in_any_representation() {
        let layout = FrameColumnLayout::new(2, &[1]);
        let mut roots = Array3::<f64>::zeros((2, 1, 1));
        roots[[0, 0, 0]] = 1.0;
        let clean = ResidualGaugeCurvature::OutputBlockRoots {
            roots: roots.clone(),
            layout: layout.clone(),
            root_rows: 3,
        };
        assert!(clean.is_finite());
        roots[[1, 0, 0]] = f64::NAN;
        let dirty = ResidualGaugeCurvature::OutputBlockRoots {
            roots,
            layout,
            root_rows: 3,
        };
        assert!(!dirty.is_finite());
        assert!(
            !ResidualGaugeCurvature::DualRoot {
                root: Array2::<f64>::from_elem((1, 2), f64::INFINITY),
                root_rows: 1,
            }
            .is_finite()
        );
        assert!(
            !ResidualGaugeCurvature::DenseGram {
                gram: Array2::<f64>::from_elem((2, 2), f64::NAN),
                root_rows: 1,
            }
            .is_finite()
        );
    }

    #[test]
    fn output_blocks_densify_to_a_block_diagonal_gram() {
        let layout = FrameColumnLayout::new(3, &[1, 2]);
        let mut roots = Array3::<f64>::zeros((3, 3, 3));
        for i in 0..3 {
            for a in 0..3 {
                for b in a..3 {
                    roots[[i, a, b]] = (i + 1) as f64 * ((a + 1) + (b + 1)) as f64;
                }
            }
        }
        let curvature = ResidualGaugeCurvature::OutputBlockRoots {
            roots,
            layout: layout.clone(),
            root_rows: 7,
        };
        let dense = curvature.to_dense_gram();
        assert_eq!(dense.dim(), (9, 9));
        for a in 0..9 {
            for b in 0..9 {
                let (ia, ib) = (
                    layout.output_of(a).expect("in range"),
                    layout.output_of(b).expect("in range"),
                );
                if ia != ib {
                    assert_eq!(dense[[a, b]], 0.0, "off-block ({a},{b}) must be zero");
                } else {
                    assert!(dense[[a, b]] != 0.0, "in-block ({a},{b}) must be populated");
                }
            }
        }
    }
}
