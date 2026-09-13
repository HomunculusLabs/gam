//! Exact third coefficient-directional derivative `D³H[u, v, e_a]` of the joint Hessian for
//! a time wiggle with a score warp or link deviation (gam#2893).
//!
//! A row's negative log-likelihood is `ℓ(G(ζ))`. The row coordinate ζ is affine in β. Its
//! z block holds the entry index `h₀`, the exit index `h₁`, the raw derivative index `d`
//! and the wiggle coefficients γ; after it come the linear primaries, which are the slope
//! and the identity-mapped flex coordinates. `G` maps ζ to the primaries. Only three of them
//! are nonlinear: `q₀ = h₀ + B(h₀)·γ`, `q₁ = h₁ + B(h₁)·γ` and `q̇₁ = (1 + B'(h₁)·γ)·d`,
//! and each is at most linear in γ and in `d`. The row Hessian is `Ãᵀ ∇²_ζ(ℓ∘G) Ã`, so
//! `D³H[u, v, w] = Ãᵀ ∇⁵_ζ(ℓ∘G)[Ãu, Ãv, Ãw] Ã`.
//!
//! The fifth derivative of the composition is Faà di Bruno over the 52 set partitions of
//! the two free axes and the three directions. When a partition puts both free axes in one
//! block, the ℓ contraction of the other blocks weights a curvature of `G` over both axes.
//! When it separates them, an ℓ derivative sits between two one-axis derivatives of `G`.
//! Every derivative of `G` has a closed form in the wiggle basis derivatives. Every ℓ
//! contraction is linear in the third direction. So each row contracts once per primary
//! axis, assembles once per ζ axis, and pulls back once per coefficient axis.

use super::*;

/// z-block positions of the entry index, the exit index and the raw derivative index; the
/// wiggle coefficients follow.
const ZETA_H0: usize = 0;
const ZETA_H1: usize = 1;
const ZETA_DR: usize = 2;
const ZETA_GAMMA: usize = 3;
/// The wiggle basis derivatives `B⁽⁰⁾..B⁽⁶⁾`: the fifth derivative of `q̇₁` reaches `m₆`.
const WIGGLE_ORDERS: usize = 7;
/// Bits of the directions `u`, `v` and the ζ-axis direction `z` in a subset mask.
const U: u8 = 0b001;
const V: u8 = 0b010;
const Z: u8 = 0b100;

/// One timepoint's basis derivative rows `B⁽ᵏ⁾(h)` and `m_k = B⁽ᵏ⁾(h)·γ`.
type SideBasis = (Vec<Array1<f64>>, [f64; WIGGLE_ORDERS]);

/// One timepoint's image of a ζ direction: its index component `h` and `B⁽ᵏ⁾(h)·γ` of its
/// wiggle component at every order `k`.
#[derive(Clone, Copy)]
struct SideImage {
    h: f64,
    b: [f64; WIGGLE_ORDERS],
}

/// The z-block image of a ζ direction on both timepoints.
#[derive(Clone, Copy)]
struct ZetaDirection {
    entry: SideImage,
    exit: SideImage,
    dr: f64,
}

impl ZetaDirection {
    const ZERO: Self = Self {
        entry: SideImage {
            h: 0.0,
            b: [0.0; WIGGLE_ORDERS],
        },
        exit: SideImage {
            h: 0.0,
            b: [0.0; WIGGLE_ORDERS],
        },
        dr: 0.0,
    };
}

/// The members of a subset mask over the three directions.
fn members(set: u8) -> impl Iterator<Item = usize> {
    (0..3usize).filter(move |&j| set & (1 << j) != 0)
}

/// `Π_{j∈S} h_j` on one timepoint.
fn side_product(dirs: [&SideImage; 3], set: u8) -> f64 {
    members(set).map(|j| dirs[j].h).product()
}

/// `D^{|S|} m_k [S]` on one timepoint,
/// `m_{k+|S|}·Π_{j∈S} h_j + Σ_{j∈S} (B^{(k+|S|−1)}·γ_j)·Π_{i∈S∖j} h_i`.
/// `m_k` is linear in γ and `h` is affine in β, so no other term survives.
fn side_moved_m(m: &[f64; WIGGLE_ORDERS], dirs: [&SideImage; 3], k: usize, set: u8) -> f64 {
    let size = set.count_ones() as usize;
    let mut value = m[k + size] * side_product(dirs, set);
    for j in members(set) {
        value += dirs[j].b[k + size - 1] * side_product(dirs, set & !(1 << j));
    }
    value
}

/// A row's time-wiggle geometry on both timepoints: the basis derivative rows `B⁽ᵏ⁾(h)` and
/// `m_k = B⁽ᵏ⁾(h)·γ`, with `m₁` carrying the identity of `q = h + B(h)·γ`.
struct WiggleRowGeometry {
    dr: f64,
    entry_basis: Vec<Array1<f64>>,
    exit_basis: Vec<Array1<f64>>,
    entry_m: [f64; WIGGLE_ORDERS],
    exit_m: [f64; WIGGLE_ORDERS],
}

impl WiggleRowGeometry {
    fn gamma_width(&self) -> usize {
        self.entry_basis[0].len()
    }

    /// The image of a z-block direction.
    fn direction(&self, z: ArrayView1<'_, f64>) -> ZetaDirection {
        let gamma = z.slice(s![ZETA_GAMMA..]);
        ZetaDirection {
            entry: SideImage {
                h: z[ZETA_H0],
                b: std::array::from_fn(|k| self.entry_basis[k].dot(&gamma)),
            },
            exit: SideImage {
                h: z[ZETA_H1],
                b: std::array::from_fn(|k| self.exit_basis[k].dot(&gamma)),
            },
            dr: z[ZETA_DR],
        }
    }

    /// `Dⁿ(q₀, q₁, q̇₁)[S]` for a nonempty subset `S`.
    fn q_derivative(&self, dirs: [&ZetaDirection; 3], set: u8) -> [f64; 3] {
        let entry = dirs.map(|direction| &direction.entry);
        let exit = dirs.map(|direction| &direction.exit);
        let mut qd1 = side_moved_m(&self.exit_m, exit, 1, set) * self.dr;
        for j in members(set) {
            qd1 += dirs[j].dr * side_moved_m(&self.exit_m, exit, 1, set & !(1 << j));
        }
        [
            side_moved_m(&self.entry_m, entry, 0, set),
            side_moved_m(&self.exit_m, exit, 0, set),
            qd1,
        ]
    }

    /// `D^{1+|S|}(q₀, q₁, q̇₁)[·, S]` over the z-block axes, one row per nonlinear primary.
    fn q_rows(&self, dirs: [&ZetaDirection; 3], set: u8) -> Array2<f64> {
        let size = set.count_ones() as usize;
        let entry = dirs.map(|direction| &direction.entry);
        let exit = dirs.map(|direction| &direction.exit);
        let entry_product = side_product(entry, set);
        let exit_product = side_product(exit, set);
        let mut rows = Array2::<f64>::zeros((3, ZETA_GAMMA + self.gamma_width()));
        rows[[0, ZETA_H0]] = side_moved_m(&self.entry_m, entry, 1, set);
        rows[[1, ZETA_H1]] = side_moved_m(&self.exit_m, exit, 1, set);
        let mut qd1_h1 = side_moved_m(&self.exit_m, exit, 2, set) * self.dr;
        for j in members(set) {
            qd1_h1 += dirs[j].dr * side_moved_m(&self.exit_m, exit, 2, set & !(1 << j));
        }
        rows[[2, ZETA_H1]] = qd1_h1;
        rows[[2, ZETA_DR]] = side_moved_m(&self.exit_m, exit, 1, set);
        for l in 0..self.gamma_width() {
            rows[[0, ZETA_GAMMA + l]] = self.entry_basis[size][l] * entry_product;
            rows[[1, ZETA_GAMMA + l]] = self.exit_basis[size][l] * exit_product;
            let mut qd1_gamma = self.exit_basis[size + 1][l] * exit_product * self.dr;
            for j in members(set) {
                qd1_gamma +=
                    dirs[j].dr * self.exit_basis[size][l] * side_product(exit, set & !(1 << j));
            }
            rows[[2, ZETA_GAMMA + l]] = qd1_gamma;
        }
        rows
    }

    /// `D^{2+|S|}(q₀, q₁, q̇₁)[·, ·, S]` over the z-block axes, one matrix per nonlinear
    /// primary.
    fn q_matrices(&self, dirs: [&ZetaDirection; 3], set: u8) -> [Array2<f64>; 3] {
        let size = set.count_ones() as usize;
        let entry = dirs.map(|direction| &direction.entry);
        let exit = dirs.map(|direction| &direction.exit);
        let entry_product = side_product(entry, set);
        let exit_product = side_product(exit, set);
        let width = ZETA_GAMMA + self.gamma_width();
        let mut q0 = Array2::<f64>::zeros((width, width));
        let mut q1 = Array2::<f64>::zeros((width, width));
        let mut qd1 = Array2::<f64>::zeros((width, width));
        q0[[ZETA_H0, ZETA_H0]] = side_moved_m(&self.entry_m, entry, 2, set);
        q1[[ZETA_H1, ZETA_H1]] = side_moved_m(&self.exit_m, exit, 2, set);
        let mut qd1_hh = side_moved_m(&self.exit_m, exit, 3, set) * self.dr;
        for j in members(set) {
            qd1_hh += dirs[j].dr * side_moved_m(&self.exit_m, exit, 3, set & !(1 << j));
        }
        qd1[[ZETA_H1, ZETA_H1]] = qd1_hh;
        let qd1_hd = side_moved_m(&self.exit_m, exit, 2, set);
        qd1[[ZETA_H1, ZETA_DR]] = qd1_hd;
        qd1[[ZETA_DR, ZETA_H1]] = qd1_hd;
        for l in 0..self.gamma_width() {
            let g = ZETA_GAMMA + l;
            let entry_cross = self.entry_basis[size + 1][l] * entry_product;
            q0[[ZETA_H0, g]] = entry_cross;
            q0[[g, ZETA_H0]] = entry_cross;
            let exit_cross = self.exit_basis[size + 1][l] * exit_product;
            q1[[ZETA_H1, g]] = exit_cross;
            q1[[g, ZETA_H1]] = exit_cross;
            qd1[[ZETA_DR, g]] = exit_cross;
            qd1[[g, ZETA_DR]] = exit_cross;
            let mut qd1_hg = self.exit_basis[size + 2][l] * exit_product * self.dr;
            for j in members(set) {
                qd1_hg += dirs[j].dr
                    * self.exit_basis[size + 1][l]
                    * side_product(exit, set & !(1 << j));
            }
            qd1[[ZETA_H1, g]] = qd1_hg;
            qd1[[g, ZETA_H1]] = qd1_hg;
        }
        [q0, q1, qd1]
    }
}

/// The sparse ζ image of one coefficient axis. A base time column reaches the entry, exit
/// and derivative indices, which is the most entries any axis has.
#[derive(Clone, Copy, Default)]
struct ZetaImage {
    len: usize,
    entries: [(usize, f64); 3],
}

impl ZetaImage {
    fn push(&mut self, zeta: usize, weight: f64) {
        if weight != 0.0 {
            self.entries[self.len] = (zeta, weight);
            self.len += 1;
        }
    }

    fn entries(&self) -> &[(usize, f64)] {
        &self.entries[..self.len]
    }
}

/// Where the primaries sit in ζ: the nonlinear primaries are the rows of `Ĵ` over the z
/// block, and each linear primary is one ζ coordinate after it.
struct ZetaLayout {
    /// Primary indices of `q₀`, `q₁` and `q̇₁`.
    q: [usize; 3],
    /// `(primary index, ζ index)` of every linear primary.
    linear: Vec<(usize, usize)>,
    z_width: usize,
    width: usize,
}

impl ZetaLayout {
    fn nonlinear_block(&self, m: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn((3, 3), |(i, j)| m[[self.q[i], self.q[j]]])
    }

    /// `Ĵ·ζ`, where `Ĵ = ∂G/∂ζ` has the nonlinear rows `jq` and the identity on the linear
    /// primaries.
    fn primary_image(
        &self,
        jq: &Array2<f64>,
        zeta: &Array1<f64>,
        primary_total: usize,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(primary_total);
        let z = zeta.slice(s![..self.z_width]);
        for (row, &k) in self.q.iter().enumerate() {
            out[k] = jq.row(row).dot(&z);
        }
        for &(k, index) in &self.linear {
            out[k] = zeta[index];
        }
        out
    }

    /// A nonlinear-primary vector on its primary indices.
    fn on_primaries(&self, values: [f64; 3], primary_total: usize) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(primary_total);
        for (row, &k) in self.q.iter().enumerate() {
            out[k] = values[row];
        }
        out
    }

    /// `Φ += Σ_q c_q·K_q` over the z block.
    fn add_curvature(
        &self,
        phi: &mut Array2<f64>,
        coefficients: &Array1<f64>,
        matrices: &[Array2<f64>; 3],
    ) {
        let mut block = phi.slice_mut(s![..self.z_width, ..self.z_width]);
        for (row, &k) in self.q.iter().enumerate() {
            if coefficients[k] != 0.0 {
                block.scaled_add(coefficients[k], &matrices[row]);
            }
        }
    }

    /// `Φ += Ĵᵀ M Ĵ`.
    fn add_full_sandwich(&self, phi: &mut Array2<f64>, jq: &Array2<f64>, m: &Array2<f64>) {
        let zw = self.z_width;
        let zz = jq.t().dot(&self.nonlinear_block(m).dot(jq));
        {
            let mut block = phi.slice_mut(s![..zw, ..zw]);
            block += &zz;
        }
        for &(k, zeta) in &self.linear {
            let mut column = Array1::<f64>::zeros(zw);
            for (row, &qk) in self.q.iter().enumerate() {
                column.scaled_add(m[[qk, k]], &jq.row(row));
            }
            {
                let mut lower = phi.slice_mut(s![..zw, zeta]);
                lower += &column;
            }
            {
                let mut upper = phi.slice_mut(s![zeta, ..zw]);
                upper += &column;
            }
            for &(l, zeta_l) in &self.linear {
                phi[[zeta, zeta_l]] += m[[k, l]];
            }
        }
    }

    /// `Φ += Xᵀ M Ĵ + (Xᵀ M Ĵ)ᵀ` for nonlinear rows `X` over the z block.
    fn add_symmetric_half_sandwich(
        &self,
        phi: &mut Array2<f64>,
        x: &Array2<f64>,
        m: &Array2<f64>,
        jq: &Array2<f64>,
    ) {
        let zw = self.z_width;
        let zz = x.t().dot(&self.nonlinear_block(m).dot(jq));
        {
            let mut block = phi.slice_mut(s![..zw, ..zw]);
            block += &zz;
            block += &zz.t();
        }
        for &(k, zeta) in &self.linear {
            let mut column = Array1::<f64>::zeros(zw);
            for (row, &qk) in self.q.iter().enumerate() {
                column.scaled_add(m[[qk, k]], &x.row(row));
            }
            {
                let mut lower = phi.slice_mut(s![..zw, zeta]);
                lower += &column;
            }
            let mut upper = phi.slice_mut(s![zeta, ..zw]);
            upper += &column;
        }
    }

    /// `Φ += Xᵀ M Y + (Xᵀ M Y)ᵀ` for nonlinear rows `X` and `Y` over the z block.
    fn add_symmetric_row_sandwich(
        &self,
        phi: &mut Array2<f64>,
        x: &Array2<f64>,
        m: &Array2<f64>,
        y: &Array2<f64>,
    ) {
        let zz = x.t().dot(&self.nonlinear_block(m).dot(y));
        let mut block = phi.slice_mut(s![..self.z_width, ..self.z_width]);
        block += &zz;
        block += &zz.t();
    }
}

/// `Σ_k w_k·axes[k]` over the primary axes.
fn combine_axes(axes: &[Array2<f64>], weights: &Array1<f64>, primary_total: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((primary_total, primary_total));
    for (axis, &weight) in axes.iter().zip(weights.iter()) {
        if weight != 0.0 {
            out.scaled_add(weight, axis);
        }
    }
    out
}

/// `acc += Ãᵀ Φ Ã` for one row, where column `c` of `Ã` is `images[c]`; `right` is scratch
/// for `Φ Ã`.
fn add_zeta_pullback(
    phi: &Array2<f64>,
    images: &[ZetaImage],
    right: &mut Array2<f64>,
    acc: &mut Array2<f64>,
) {
    right.fill(0.0);
    for (c, image) in images.iter().enumerate() {
        let mut column = right.column_mut(c);
        for &(zeta, weight) in image.entries() {
            column.scaled_add(weight, &phi.column(zeta));
        }
    }
    for (c, image) in images.iter().enumerate() {
        let mut row = acc.row_mut(c);
        for &(zeta, weight) in image.entries() {
            row.scaled_add(weight, &right.row(zeta));
        }
    }
}

impl SurvivalMarginalSlopeFamily {
    /// Third directional derivative `D³H[u, v, e_a]` of the joint Hessian along every
    /// coefficient axis, for a time wiggle with a score warp or link deviation (gam#2893).
    /// The module documentation derives the ζ composition this evaluates.
    pub(crate) fn exact_newton_joint_hessian_third_directional_derivative_timewiggle_flex_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        if self.influence_absorber.is_some() {
            return Err(
                "time-wiggle third information derivative: the influence absorber's primary has \
                 no ζ coordinate"
                    .to_string(),
            );
        }
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Err(
                "time-wiggle third information derivative: the family has no time-wiggle basis"
                    .to_string(),
            );
        };
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let p_primary = primary.total;
        let p_marginal = slices.marginal.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let gamma_width = time_tail.len();
        let z_width = ZETA_GAMMA + gamma_width;
        let q = [primary.q0, primary.q1, primary.qd1];
        let linear: Vec<(usize, usize)> = (0..p_primary)
            .filter(|k| !q.contains(k))
            .enumerate()
            .map(|(position, k)| (k, z_width + position))
            .collect();
        let mut zeta_of_primary = vec![None; p_primary];
        for &(k, zeta) in &linear {
            zeta_of_primary[k] = Some(zeta);
        }
        let layout = ZetaLayout {
            q,
            width: z_width + linear.len(),
            z_width,
            linear,
        };
        let slope_zeta = zeta_of_primary[primary.g].ok_or_else(|| {
            "time-wiggle third information derivative: the slope primary has no ζ coordinate"
                .to_string()
        })?;
        let mut identity_images = Vec::new();
        for (primary_range, joint_range) in flex_identity_block_pairs(&primary, &slices) {
            if primary_range.len() != joint_range.len() {
                return Err(format!(
                    "time-wiggle third information derivative: flex primaries {primary_range:?} \
                     and coefficients {joint_range:?} differ in width"
                ));
            }
            for local in 0..primary_range.len() {
                let zeta = zeta_of_primary[primary_range.start + local].ok_or_else(|| {
                    "time-wiggle third information derivative: a flex primary has no ζ coordinate"
                        .to_string()
                })?;
                identity_images.push((joint_range.start + local, zeta));
            }
        }
        let beta_time = &block_states[0].beta;
        let beta_base = beta_time.slice(s![..p_base]);
        let gamma = beta_time.slice(s![time_tail.clone()]);
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let mut right = Array2::<f64>::zeros((layout.width, p_total));
                let mut phi_axis = Array2::<f64>::zeros((layout.width, layout.width));
                for row in range {
                    // ── Row geometry: design rows, indices, wiggle basis derivatives ──
                    let entry_chunk = self
                        .design_entry
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_entry try_row_chunk: {e}"))?;
                    let exit_chunk = self
                        .design_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_exit try_row_chunk: {e}"))?;
                    let derivative_chunk = self
                        .design_derivative_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_derivative_exit try_row_chunk: {e}"))?;
                    let marginal_chunk = self
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("marginal_design try_row_chunk: {e}"))?;
                    let slope_chunk = self
                        .slope_layout
                        .coefficient_design()
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("slope_design try_row_chunk: {e}"))?;
                    let xe = entry_chunk.row(0).slice(s![..p_base]).to_owned();
                    let xx = exit_chunk.row(0).slice(s![..p_base]).to_owned();
                    let xd = derivative_chunk.row(0).slice(s![..p_base]).to_owned();
                    let mr = marginal_chunk.row(0);
                    let gr = slope_chunk.row(0);
                    let bm = block_states[1].eta[row];
                    let h0 = xe.dot(&beta_base) + self.offset_entry[row] + bm;
                    let h1 = xx.dot(&beta_base) + self.offset_exit[row] + bm;
                    let dr = xd.dot(&beta_base) + self.derivative_offset_exit[row];
                    let side = |h: f64| -> Result<SideBasis, String> {
                        let seed = Array1::from_vec(vec![h]);
                        let mut basis = Vec::with_capacity(WIGGLE_ORDERS);
                        for order in 0..WIGGLE_ORDERS {
                            let rows = monotone_wiggle_basis_with_derivative_order(
                                seed.view(),
                                knots,
                                degree,
                                order,
                            )?;
                            if rows.ncols() != gamma_width {
                                return Err(format!(
                                    "time-wiggle third information derivative: basis derivative \
                                     {order} has {} columns for {gamma_width} wiggle coefficients",
                                    rows.ncols()
                                ));
                            }
                            basis.push(rows.row(0).to_owned());
                        }
                        let mut m: [f64; WIGGLE_ORDERS] =
                            std::array::from_fn(|k| basis[k].dot(&gamma));
                        m[1] += 1.0;
                        Ok((basis, m))
                    };
                    let (entry_basis, entry_m) = side(h0)?;
                    let (exit_basis, exit_m) = side(h1)?;
                    let geometry = WiggleRowGeometry {
                        dr,
                        entry_basis,
                        exit_basis,
                        entry_m,
                        exit_m,
                    };

                    // ── Ã: the ζ image of every coefficient axis ──
                    let mut images = vec![ZetaImage::default(); p_total];
                    for a in 0..p_base {
                        let image = &mut images[slices.time.start + a];
                        image.push(ZETA_H0, xe[a]);
                        image.push(ZETA_H1, xx[a]);
                        image.push(ZETA_DR, xd[a]);
                    }
                    for l in 0..gamma_width {
                        images[slices.time.start + time_tail.start + l].push(ZETA_GAMMA + l, 1.0);
                    }
                    for j in 0..p_marginal {
                        let image = &mut images[slices.marginal.start + j];
                        image.push(ZETA_H0, mr[j]);
                        image.push(ZETA_H1, mr[j]);
                    }
                    for b in 0..slices.slope.len() {
                        images[slices.slope.start + b].push(slope_zeta, gr[b]);
                    }
                    for &(joint, zeta) in &identity_images {
                        images[joint].push(zeta, 1.0);
                    }
                    let zeta_image = |direction: &Array1<f64>| -> Array1<f64> {
                        let mut zeta = Array1::<f64>::zeros(layout.width);
                        for (c, image) in images.iter().enumerate() {
                            if direction[c] != 0.0 {
                                for &(index, weight) in image.entries() {
                                    zeta[index] += weight * direction[c];
                                }
                            }
                        }
                        zeta
                    };
                    let u_zeta = zeta_image(d_u);
                    let v_zeta = zeta_image(d_v);

                    // ── Derivatives of G that do not read the ζ axis ──
                    let dir_u = geometry.direction(u_zeta.slice(s![..z_width]));
                    let dir_v = geometry.direction(v_zeta.slice(s![..z_width]));
                    let fixed = [&dir_u, &dir_v, &ZetaDirection::ZERO];
                    let jq = geometry.q_rows(fixed, 0);
                    let ju = layout.primary_image(&jq, &u_zeta, p_primary);
                    let jv = layout.primary_image(&jq, &v_zeta, p_primary);
                    let g2uv = layout.on_primaries(geometry.q_derivative(fixed, U | V), p_primary);
                    let ju1 = geometry.q_rows(fixed, U);
                    let jv1 = geometry.q_rows(fixed, V);
                    let juv2 = geometry.q_rows(fixed, U | V);
                    let k0 = geometry.q_matrices(fixed, 0);
                    let ku = geometry.q_matrices(fixed, U);
                    let kv = geometry.q_matrices(fixed, V);
                    let kuv = geometry.q_matrices(fixed, U | V);

                    // ── ℓ contractions, once per primary axis ──
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, gradient, hessian) = self
                        .compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
                            &primary,
                        )?;
                    let base =
                        self.build_row_flex_fifth_base_with_states(row, block_states, &primary)?;
                    let mut third = Vec::with_capacity(p_primary);
                    let mut fourth_u = Vec::with_capacity(p_primary);
                    let mut fourth_v = Vec::with_capacity(p_primary);
                    let mut fourth_uv = Vec::with_capacity(p_primary);
                    for k in 0..p_primary {
                        let mut axis = Array1::<f64>::zeros(p_primary);
                        axis[k] = 1.0;
                        third.push(self.row_flex_third_contract_from_base(&base, &axis)?);
                        fourth_u.push(self.row_flex_fourth_contract_from_base(&base, &ju, &axis)?);
                        fourth_v.push(self.row_flex_fourth_contract_from_base(&base, &jv, &axis)?);
                        fourth_uv
                            .push(self.row_flex_fourth_contract_from_base(&base, &g2uv, &axis)?);
                    }
                    let fifth =
                        self.row_flex_fifth_contract_all_primary_axes_from_base(&base, &ju, &jv)?;
                    let t_u = combine_axes(&third, &ju, p_primary);
                    let t_v = combine_axes(&third, &jv, p_primary);
                    let q_uv = combine_axes(&fourth_u, &jv, p_primary);
                    let m_z = &q_uv + &combine_axes(&third, &g2uv, p_primary);
                    let c_z = t_u.dot(&jv) + hessian.dot(&g2uv);
                    let c_uz = hessian.dot(&jv);
                    let c_vz = hessian.dot(&ju);

                    // ── ∇⁵_ζ(ℓ∘G)[Ãu, Ãv, e_ζ], once per ζ axis ──
                    let mut phi_axes = Vec::with_capacity(layout.width);
                    for zeta_axis in 0..layout.width {
                        let mut unit = Array1::<f64>::zeros(layout.width);
                        unit[zeta_axis] = 1.0;
                        let dir_z = geometry.direction(unit.slice(s![..z_width]));
                        let dirs = [&dir_u, &dir_v, &dir_z];
                        let jz = layout.primary_image(&jq, &unit, p_primary);
                        let g2uz =
                            layout.on_primaries(geometry.q_derivative(dirs, U | Z), p_primary);
                        let g2vz =
                            layout.on_primaries(geometry.q_derivative(dirs, V | Z), p_primary);
                        let g3uvz =
                            layout.on_primaries(geometry.q_derivative(dirs, U | V | Z), p_primary);
                        let jz1 = geometry.q_rows(dirs, Z);
                        let juz2 = geometry.q_rows(dirs, U | Z);
                        let jvz2 = geometry.q_rows(dirs, V | Z);
                        let juvz3 = geometry.q_rows(dirs, U | V | Z);
                        let t_z = combine_axes(&third, &jz, p_primary);
                        let mut phi = Array2::<f64>::zeros((layout.width, layout.width));

                        // Both free axes in one block: the ℓ contraction of the remaining
                        // blocks weights the curvature of G over both axes.
                        let c_empty = q_uv.dot(&jz)
                            + t_z.dot(&g2uv)
                            + t_v.dot(&g2uz)
                            + t_u.dot(&g2vz)
                            + hessian.dot(&g3uvz);
                        let c_u = t_v.dot(&jz) + hessian.dot(&g2vz);
                        let c_v = t_u.dot(&jz) + hessian.dot(&g2uz);
                        let c_uv = hessian.dot(&jz);
                        layout.add_curvature(&mut phi, &c_empty, &k0);
                        layout.add_curvature(&mut phi, &c_u, &ku);
                        layout.add_curvature(&mut phi, &c_v, &kv);
                        layout.add_curvature(&mut phi, &c_z, &geometry.q_matrices(dirs, Z));
                        layout.add_curvature(&mut phi, &c_uv, &kuv);
                        layout.add_curvature(&mut phi, &c_uz, &geometry.q_matrices(dirs, U | Z));
                        layout.add_curvature(&mut phi, &c_vz, &geometry.q_matrices(dirs, V | Z));
                        layout.add_curvature(
                            &mut phi,
                            &gradient,
                            &geometry.q_matrices(dirs, U | V | Z),
                        );

                        // Free axes in separate blocks: an ℓ derivative between one-axis
                        // derivatives of G.
                        let m_empty = combine_axes(&fifth, &jz, p_primary)
                            + combine_axes(&fourth_uv, &jz, p_primary)
                            + combine_axes(&fourth_v, &g2uz, p_primary)
                            + combine_axes(&fourth_u, &g2vz, p_primary)
                            + combine_axes(&third, &g3uvz, p_primary);
                        let m_u = combine_axes(&fourth_v, &jz, p_primary)
                            + combine_axes(&third, &g2vz, p_primary);
                        let m_v = combine_axes(&fourth_u, &jz, p_primary)
                            + combine_axes(&third, &g2uz, p_primary);
                        layout.add_full_sandwich(&mut phi, &jq, &m_empty);
                        layout.add_symmetric_half_sandwich(&mut phi, &ju1, &m_u, &jq);
                        layout.add_symmetric_half_sandwich(&mut phi, &jv1, &m_v, &jq);
                        layout.add_symmetric_half_sandwich(&mut phi, &jz1, &m_z, &jq);
                        layout.add_symmetric_row_sandwich(&mut phi, &ju1, &t_z, &jv1);
                        layout.add_symmetric_row_sandwich(&mut phi, &ju1, &t_v, &jz1);
                        layout.add_symmetric_row_sandwich(&mut phi, &jv1, &t_u, &jz1);
                        layout.add_symmetric_half_sandwich(&mut phi, &juv2, &t_z, &jq);
                        layout.add_symmetric_half_sandwich(&mut phi, &juz2, &t_v, &jq);
                        layout.add_symmetric_half_sandwich(&mut phi, &jvz2, &t_u, &jq);
                        layout.add_symmetric_row_sandwich(&mut phi, &juv2, &hessian, &jz1);
                        layout.add_symmetric_row_sandwich(&mut phi, &juz2, &hessian, &jv1);
                        layout.add_symmetric_row_sandwich(&mut phi, &jvz2, &hessian, &ju1);
                        layout.add_symmetric_half_sandwich(&mut phi, &juvz3, &hessian, &jq);
                        phi_axes.push(phi);
                    }

                    // ── One pullback per coefficient axis ──
                    for (c, image) in images.iter().enumerate() {
                        if image.entries().is_empty() {
                            continue;
                        }
                        phi_axis.fill(0.0);
                        for &(zeta, weight) in image.entries() {
                            phi_axis.scaled_add(weight, &phi_axes[zeta]);
                        }
                        add_zeta_pullback(&phi_axis, &images, &mut right, &mut acc[c]);
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }
}
