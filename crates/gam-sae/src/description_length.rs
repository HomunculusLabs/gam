//! Description-length (MDL) reporting surface (#2085).
//!
//! A manifold-SAE / dictionary fit is priced as a description length in bits,
//! decomposed as
//!
//! * **code** bits — the rate to transmit each firing's coefficients at the
//!   achieved distortion;
//! * **selection** bits — naming which atoms fired per token;
//! * **dictionary** bits — the amortised cost of storing the decoder.
//!
//! Every quantity is read off an existing fit; nothing here recomputes it. The
//! surface is ported from the hand-verified `Manifold-SAE
//! experiments/mdl_ladder/mdl.py` reference: the rate-distortion primitives, the
//! closed-form curved-birth pre-screen ([`predicted_birth_dl_bits`]), matched
//! curved-vs-flat description lengths ([`matched_dl`]), and the fit-level
//! [`manifold_fit_description_length`].

use crate::atom_codes::SparseAtomCodes;
use crate::manifold::SaeAtomGeometryPlan;
use ndarray::ArrayView2;

/// Bits to code one Gaussian scalar of variance `signal_var` to per-sample MSE
/// `delta2`: the Gaussian rate-distortion law
/// `½ max(log₂(signal_var / delta2), 0)`.
pub(crate) fn scalar_rate_bits(signal_var: f64, delta2: f64) -> f64 {
    if signal_var <= 0.0 {
        return 0.0;
    }
    if delta2 <= 0.0 {
        return f64::INFINITY;
    }
    (0.5 * (signal_var / delta2).log2()).max(0.0)
}

/// `log₂ C(G, k)`: bits to name which `k` of `G` dictionary atoms fired. Computed
/// as `Σ_{i=1..k} log₂((G−k+i)/i)` so it never overflows a binomial (exact, and
/// `k` is small in practice). Zero when `G ≤ 0` or `k ≤ 0`; `k` is capped at `G`.
pub fn selection_bits(g_dict: i64, k_active: i64) -> f64 {
    if g_dict <= 0 || k_active <= 0 {
        return 0.0;
    }
    let k = k_active.min(g_dict);
    let mut bits = 0.0;
    for i in 1..=k {
        bits += ((g_dict - k + i) as f64 / i as f64).log2();
    }
    bits
}

fn exact_weighted_water_level(breakpoints: &mut Vec<(f64, f64)>, total_distortion: f64) -> f64 {
    breakpoints.sort_by(|(left, _), (right, _)| left.total_cmp(right));
    let mut saturated_distortion = 0.0_f64;
    let mut active_weight: f64 = breakpoints.iter().map(|(_, weight)| weight).sum();
    let mut index = 0usize;
    loop {
        let next_breakpoint = breakpoints[index].0;
        let candidate = (total_distortion - saturated_distortion) / active_weight;
        if candidate <= next_breakpoint {
            return candidate;
        }
        while index < breakpoints.len() && breakpoints[index].0 == next_breakpoint {
            let (variance, weight) = breakpoints[index];
            saturated_distortion += weight * variance;
            active_weight -= weight;
            index += 1;
        }
        if index == breakpoints.len() {
            // A budget below total variance guarantees an earlier segment;
            // this protects against a last-bit rounding inversion only.
            return next_breakpoint;
        }
    }
}

/// Validate one covariance eigen-spectrum, clipping only rounding-sized negatives.
///
/// A covariance is positive semidefinite, so a negative eigenvalue is admissible
/// only as eigensolver rounding. A backward-stable symmetric eigensolver returns
/// the exact spectrum of a matrix within `len·ε·max|λ|` of its input, which bounds
/// how far below zero a rounded zero eigenvalue can fall. Such values are clipped
/// to zero. A nonfinite value, or one more negative than that bound, is malformed
/// input and is reported rather than silently deleted from the variance.
fn validated_variance_spectrum(spectrum: &[f64], component: usize) -> Result<Vec<f64>, String> {
    let mut scale = 0.0_f64;
    for (index, &value) in spectrum.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "component {component} spectrum: eigenvalue [{index}] must be finite, got {value}"
            ));
        }
        scale = scale.max(value.abs());
    }
    let rounding_bound = spectrum.len() as f64 * f64::EPSILON * scale;
    spectrum
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if value < -rounding_bound {
                Err(format!(
                    "component {component} spectrum: eigenvalue [{index}] = {value} lies below \
                     the eigensolver rounding bound -{rounding_bound}; the covariance is not \
                     positive semidefinite"
                ))
            } else {
                Ok(value.max(0.0))
            }
        })
        .collect()
}

/// A solved weighted reverse-water-filling allocation.
struct WeightedAllocation {
    /// One rate in bits per component, already multiplied by its weight.
    rates: Vec<f64>,
    /// The shared water level `θ`: zero at a zero budget, `+∞` when the budget
    /// covers every weighted variance.
    water_level: f64,
    /// The validated spectra with their weights, in component order.
    spectra: Vec<(f64, Vec<f64>)>,
}

fn solve_weighted_allocation(
    components: &[(f64, Vec<f64>)],
    total_distortion: f64,
) -> Result<WeightedAllocation, String> {
    if !total_distortion.is_finite() || total_distortion < 0.0 {
        return Err(format!(
            "total distortion must be finite and nonnegative, got {total_distortion}"
        ));
    }

    let mut breakpoints: Vec<(f64, f64)> = Vec::new();
    let mut spectra: Vec<(f64, Vec<f64>)> = Vec::with_capacity(components.len());
    let mut total_variance = 0.0_f64;
    for (index, (weight, spectrum)) in components.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(format!(
                "component weight must be finite and nonnegative, got {weight}"
            ));
        }
        let variances = validated_variance_spectrum(spectrum, index)?;
        for &variance in &variances {
            total_variance += *weight * variance;
            if *weight > 0.0 {
                breakpoints.push((variance, *weight));
            }
        }
        spectra.push((*weight, variances));
    }
    if !total_variance.is_finite() {
        return Err(format!(
            "weighted total variance must be finite, got {total_variance}"
        ));
    }

    let water_level = if total_distortion >= total_variance || breakpoints.is_empty() {
        f64::INFINITY
    } else if total_distortion == 0.0 {
        0.0
    } else {
        exact_weighted_water_level(&mut breakpoints, total_distortion)
    };

    let rates = spectra
        .iter()
        .map(|(weight, variances)| {
            if *weight == 0.0 {
                return 0.0;
            }
            *weight
                * variances
                    .iter()
                    .map(|&variance| scalar_rate_bits(variance, water_level))
                    .sum::<f64>()
        })
        .collect();
    Ok(WeightedAllocation {
        rates,
        water_level,
        spectra,
    })
}

/// Joint reverse-water-filling of weighted Gaussian spectra to a nonnegative
/// total-distortion budget.  A component weight scales both its distortion and
/// its rate (for Eq. 4 this is an atom's firing probability; the residual has
/// weight one).  Returns one rate in bits per component.
///
/// The water level is solved exactly, without iterative tolerances.  After
/// sorting the variance breakpoints, distortion is affine between adjacent
/// breakpoints:
/// `D(theta) = sum_{v<=theta} w*v + theta*sum_{v>theta} w`.
/// The unique segment containing the requested budget therefore gives `theta`
/// in closed form.
///
/// A zero budget is a valid boundary, not an error: every positive-weight
/// component with a positive variance costs `+∞` bits, because an exact
/// continuous Gaussian value needs unbounded rate. Nonfinite inputs and
/// materially negative eigenvalues are rejected (see
/// [`validated_variance_spectrum`]).
pub fn weighted_reverse_water_filling(
    components: &[(f64, Vec<f64>)],
    total_distortion: f64,
) -> Result<Vec<f64>, String> {
    solve_weighted_allocation(components, total_distortion).map(|allocation| allocation.rates)
}

/// Rate (bits/sample) of the optimal linear (reverse-water-filling) code of a
/// Gaussian source with covariance eigenvalues `eigs`, coded to total MSE
/// `delta2`. Returns `(total_rate_bits, per_coordinate_bits)`. This is the
/// best a LINEAR featurizer can do at that distortion — the block/direction lower
/// bound a chart must beat.
///
/// `delta2 = 0` gives the legitimate `+∞` rate of every positive variance.
/// A nonfinite or negative `delta2`, a nonfinite eigenvalue, or an eigenvalue
/// below the eigensolver rounding bound is an `Err`.
pub fn reverse_water_filling(eigs: &[f64], delta2: f64) -> Result<(f64, Vec<f64>), String> {
    let allocation = solve_weighted_allocation(&[(1.0, eigs.to_vec())], delta2)?;
    let per: Vec<f64> = allocation.spectra[0]
        .1
        .iter()
        .map(|&variance| scalar_rate_bits(variance, allocation.water_level))
        .collect();
    Ok((allocation.rates[0], per))
}

/// The spectra-only inputs to the #2233 closed-form curved-birth MDL pre-screen.
///
/// Every field is estimated at PROPOSAL time from quantities the structured
/// residual-factor fit already produced — no candidate refit is run. See
/// [`predicted_birth_dl_bits`] for the crossover formula they feed.
#[derive(Clone, Copy, Debug)]
pub struct BirthMdlPrescreen {
    /// Activation rate `ρ̂ ∈ [0, 1]`: the fraction of tokens whose residual
    /// projects onto the birth decoder direction above that direction's
    /// idiosyncratic-noise floor.
    pub rho: f64,
    /// Ambient span `ŝ`: the participation ratio `(Σλ)²/Σλ²` of the residual
    /// factor-energy spectrum — the effective number of significant residual
    /// directions the manifold image occupies (circle ≈ 2, sphere ≈ 3, torus ≈ 4).
    pub span: f64,
    /// Intrinsic dimension `d` of the candidate topology matched to `span`.
    pub intrinsic_dim: usize,
    /// Basis size `m` of the candidate topology matched to `span` (the curved
    /// atom's dictionary width per output channel).
    pub basis_size: usize,
    /// Factor signal variance `λ̂` along the birth direction (its explained
    /// residual energy `‖Λ_:,j‖²`).
    pub signal_var: f64,
    /// Per-direction idiosyncratic-noise floor `δ` (`u_jᵀ D u_j`, the residual
    /// diagonal projected onto the unit birth direction).
    pub noise_floor: f64,
    /// Token count `N` (residual rows).
    pub n_tokens: f64,
    /// Output dimension `P` (residual channels) — the per-parameter multiplier of
    /// the dictionary surcharge.
    pub p_out: usize,
    /// Dictionary size `G` (current atom count) for the `log₂(G/L0)` support term.
    pub g_dict: usize,
    /// Mean active atoms per token `L0` (the support-budget denominator).
    pub l0: f64,
}

/// The #2233 closed-form curved-birth MDL pre-screen: the predicted NET
/// description-length saving (bits) of a curved birth over the flat `s`-latent
/// alternative, from spectra alone.
///
/// From the Eq-4 crossover theorem (positive ⇒ the birth strictly lowers Eq-4
/// bits and should reach the e-process gate):
///
/// ```text
///   ΔMDL = ρ̂·N·[ (ŝ−d−1)·½log₂(λ̂/δ) + (ŝ−1)·log₂(G/L0) ]
///          − (m−ŝ)·P·½log₂(N)
/// ```
///
/// The code coefficient is the Gaussian rate-distortion rate `scalar_rate_bits`
/// (`½max(log₂(λ̂/δ),0)`) — the SAME per-scalar rate the Eq-4 scorer water-fills,
/// so the pre-screen is priced in the scorer's own currency (not the `½log₂(1+SNR)`
/// channel-capacity form, which the scorer never uses).
///
/// * the **code** term `(ŝ−d−1)·½log₂(λ̂/δ)` credits the scalars the curved
///   atom transmits fewer of than the flat span (zero for circle/sphere, positive
///   for torus/helix — signed, so a topology that needs MORE code dims than the
///   span is honestly charged);
/// * the **support** term `(ŝ−1)·log₂(G/L0)` credits the extra active slots the
///   flat span spends that the single curved atom does not — the term that scales
///   with dictionary overcompleteness;
/// * the **signed dictionary** term `−(m−ŝ)·P·½log₂(N)` is the BIC decoder-parameter
///   delta: a SURCHARGE when the curved basis is wider than the flat span it
///   replaces (m>ŝ, e.g. an H-harmonic circle m=2H+1>2), and a genuine SAVING when
///   it is narrower (m<ŝ) — the principal win for a high-codimension manifold
///   spanning many ambient directions on a compact basis (`ŝ ≥ m`). Never clamped:
///   the delta is an exact decoder-column count, so its sign never over-admits.
///
/// The second-order residual (Eckart–Young) term `Δresid ≥ 0` is OMITTED: the
/// pre-screen therefore under-credits the birth, so it can only DEFER a proposal
/// (which returns next round when the residual changes), never accept one — the
/// e-process gate stays the sole arbiter. Returns a finite value (all logs are
/// floored on degenerate inputs).
#[must_use]
pub fn predicted_birth_dl_bits(p: &BirthMdlPrescreen) -> f64 {
    let span = p.span;
    let code_bits =
        (span - p.intrinsic_dim as f64 - 1.0) * scalar_rate_bits(p.signal_var, p.noise_floor);
    let support_bits = if p.g_dict > 0 && p.l0 > 0.0 {
        (span - 1.0) * (p.g_dict as f64 / p.l0).log2()
    } else {
        0.0
    };
    let n = p.n_tokens.max(0.0);
    let saving = p.rho.clamp(0.0, 1.0) * n * (code_bits + support_bits);
    // `½log₂(N)` needs N ≥ 2 to be non-negative; a degenerate token count charges
    // no dictionary term. The dictionary delta `−(m−ŝ)·P·½log₂N` is SIGNED: a charge
    // when the curved basis is wider than the flat span it replaces (m>ŝ), and a
    // genuine SAVING when it is narrower (m<ŝ) — the principal Eq-4 win for a
    // high-codimension manifold carried by a compact basis (`s ≥ m`). The BIC
    // decoder-parameter delta is exact (a difference of decoder column counts), so
    // crediting its sign never over-admits a birth; the pre-screen's conservatism
    // comes solely from omitting the Eckart–Young residual term, never from clamping
    // this one (a clamp would indefinitely defer exactly the births the theorem targets).
    let log2_n = if n >= 2.0 { n.log2() } else { 0.0 };
    let dictionary_delta = (p.basis_size as f64 - span) * p.p_out as f64 * 0.5 * log2_n;
    saving - dictionary_delta
}

// ===========================================================================
// Rate–distortion currency: the curved-coding gain (Theorem 3 of the
// "Superposed Geometry" memo).
// ===========================================================================
//
// Coding a firing against a curved chart beats the flat Gaussian code by a
// closed-form gain: every pinned-down ambient direction saves `½ log₂(1/δ²)` bits.
// This gain is ACTIVATION-space compression, measured in bits of reconstruction
// code. It is orthogonal to the behavioral nats of the Rung-1/Rung-2 fits:
// curvature can pay here and be behaviorally inert.

/// The EXACT circle coding gain (Theorem 3, circle case), in bits:
/// `Δ_circle = ½ · log₂( 3 a² / (π² δ²) )`.
///
/// `a` is the circle radius, `delta = δ` the tolerance: the Theorem-3 gain at
/// codimension one, with the circle's shape constant folded in.
pub(crate) fn circle_coding_gain_bits(a: f64, delta: f64) -> f64 {
    if !(a > 0.0) || !(delta > 0.0) {
        return 0.0;
    }
    use std::f64::consts::PI;
    0.5 * (3.0 * a * a / (PI * PI * delta * delta)).log2()
}

// ===========================================================================
// Matched description length (curved-vs-flat in bits): the honest headline
// currency for a birth. EV alone is not comparable across topologies — a circle
// chart and a line atom that reach the same EV pay DIFFERENT description lengths,
// so the fair comparison is total bits, parameter charge PLUS per-firing coding.
// ===========================================================================
//
// # The uniform-quantization coding argument (per-firing coordinate bits)
//
// A firing's coordinate — a circle chart's PHASE `t ∈ [0, 1)`, or a flat atom's
// AMPLITUDE on its unit range — is recovered with a delta-method standard error
// `SE = σ / (2π·‖z‖)` (the already-computed coordinate SE; `σ` the per-component
// residual scale, `‖z‖` the firing radius). To TRANSMIT that coordinate we quantize
// it with a uniform quantizer of cell width `Δ`. A uniform quantizer of width `Δ`
// has quantization-noise variance `Δ²/12` (the variance of `U(−Δ/2, Δ/2)`). There
// is no point resolving the coordinate finer than the estimator's own uncertainty,
// so we MATCH the quantizer to the estimate — set the quantization noise equal to
// the estimation variance, `Δ²/12 = SE²`, i.e. cell width `Δ = SE·√12` (a `±SE·√12/2`
// uniform resolution). Coding a coordinate that ranges over a unit interval at that
// resolution costs
//
// ```text
//   bits(SE) = log₂(range / Δ) = log₂(1 / (SE·√12)) = −½·log₂(12·SE²)
//            = ½·log₂( 1 / (12·SE²) ).
// ```
//
// The cost is floored at 0: once `SE ≥ 1/√12` (the SD of `U(0,1)` — the maximum-
// entropy prior on a unit-range coordinate, exactly the phase-SE ceiling the
// coordinate readout clamps to), the coordinate is not localized beyond its prior
// and carries no code bits.
//
// # The matched description length of a chart / atom
//
// A featurizer that stores `C` dictionary columns in ambient dim `p`, each scalar
// quantized to `l_param` bits, and fires `f` times, has description length
//
// ```text
//   total_dl_bits = C·p·l_param            (parameter-column charge)
//                 + Σ_{i=1..f} bits(SE_i)  (per-firing coordinate coding)
// ```
//
// A **circle chart** of harmonic order `H` charges `C = 2H + 1` columns (a cos and
// a sin row per harmonic, plus the constant/DC row) and per-firing PHASE bits. A
// **line / flat atom** charges `C = 1` column and per-firing AMPLITUDE bits under
// the same `bits(SE)` rule. The curved-vs-flat comparison then reads directly in
// bits via [`matched_dl_delta`] (flat − chart; positive ⇒ the curved chart is the
// shorter code) and per-chart [`MatchedDl::dl_per_ev`].

/// Uniform-quantization coding cost, in bits, of one unit-range coordinate known to
/// standard error `se`: `½·log₂(1/(12·se²))`, floored at 0.
///
/// Derived in the module note: matching a uniform quantizer's noise variance
/// `Δ²/12` to the estimation variance `se²` gives cell width `Δ = se·√12` and cost
/// `log₂(1/Δ) = ½·log₂(1/(12·se²))`. Returns `0` for `se ≥ 1/√12` (the coordinate
/// is not localized beyond its `U(0,1)` prior) and `+∞` for a perfectly-known
/// `se = 0` (an exact continuous value needs unbounded bits). A non-finite or
/// negative `se` is treated as unidentified (`0` bits).
pub fn se_resolution_bits(se: f64) -> f64 {
    if !se.is_finite() || se < 0.0 {
        return 0.0;
    }
    if se == 0.0 {
        return f64::INFINITY;
    }
    let bits = -0.5 * (12.0 * se * se).log2();
    bits.max(0.0)
}

/// The matched description length of one chart / atom, in bits: the parameter-column
/// charge plus the summed per-firing coordinate coding bits (see the module note).
#[derive(Clone, Copy, Debug)]
pub struct MatchedDl {
    /// Dictionary columns charged (`2H+1` for a circle chart, `1` for a flat atom).
    pub coded_columns: i64,
    /// Ambient dimension `p` each stored column spans.
    pub ambient_p: i64,
    /// Bits per stored dictionary scalar.
    pub l_param_bits: f64,
    /// Parameter-column charge `C·p·l_param` (bits).
    pub param_bits: f64,
    /// Coordinates transmitted PER FIRING: `d_atom` for a chart (1 for a circle),
    /// `block_size` for a flat block that codes every coefficient. This is the
    /// code-economy axis — at matched per-scalar distortion, a chart spanning the
    /// same subspace as a b-dim block saves `(b − d)` coded scalars per firing.
    pub coords_per_firing: i64,
    /// Summed per-firing coordinate coding bits `coords_per_firing · Σ_i bits(SE_i)`.
    pub coding_bits: f64,
    /// Number of firings coded.
    pub n_firings: i64,
    /// Total description length `param_bits + coding_bits` (bits).
    pub total_dl_bits: f64,
    /// Explained variance the chart / atom achieves (the reported dose).
    pub ev: f64,
    /// Matched-DL cost per unit EV, `total_dl_bits / ev` (`+∞` when `ev ≤ 0`).
    pub dl_per_ev: f64,
}

/// Assemble the matched description length of a chart / atom from its column count,
/// ambient dim, per-scalar precision, per-firing coordinate SEs, and achieved EV.
///
/// `coded_columns` is `2H+1` for a circle chart or the
/// column count of a flat block. `coords_per_firing` is how many coordinates each
/// FIRING transmits — `d_atom` for a chart (1 for a circle's phase), `block_size`
/// for a flat block coding every coefficient: at matched per-scalar distortion the
/// per-firing bits are `coords_per_firing · se_resolution_bits(SE_i)`, so the
/// chart's code economy (fewer transmitted scalars per firing) is priced, not
/// erased. `per_firing_se` are the delta-method coordinate SEs (`σ/(2π‖z‖)`), one
/// per firing. The total is `coded_columns·ambient_p·l_param_bits +
/// coords_per_firing·Σ_i se_resolution_bits(SE_i)`.
pub fn matched_dl(
    coded_columns: i64,
    coords_per_firing: i64,
    ambient_p: i64,
    l_param_bits: f64,
    per_firing_se: &[f64],
    ev: f64,
) -> MatchedDl {
    let coded_columns = coded_columns.max(0);
    let coords_per_firing = coords_per_firing.max(0);
    let ambient_p = ambient_p.max(0);
    let param_bits = coded_columns as f64 * ambient_p as f64 * l_param_bits.max(0.0);
    let coding_bits: f64 = coords_per_firing as f64
        * per_firing_se
            .iter()
            .map(|&se| se_resolution_bits(se))
            .sum::<f64>();
    let total = param_bits + coding_bits;
    let dl_per_ev = if ev > 0.0 { total / ev } else { f64::INFINITY };
    MatchedDl {
        coded_columns,
        ambient_p,
        l_param_bits,
        param_bits,
        coords_per_firing,
        coding_bits,
        n_firings: per_firing_se.len() as i64,
        total_dl_bits: total,
        ev,
        dl_per_ev,
    }
}

/// Matched-DL delta `flat − chart`, in bits: the description length the curved chart
/// SAVES over the flat/line atom at the SAME firings. Positive ⇒ the curved chart is
/// the shorter code (curvature pays in bits); negative ⇒ the flat atom is cheaper
/// (the honest "curvature does not pay here" verdict).
pub(crate) fn matched_dl_delta(flat: &MatchedDl, chart: &MatchedDl) -> f64 {
    flat.total_dl_bits - chart.total_dl_bits
}

// ===========================================================================
// Fit-level bits/token: the headline currency for a WHOLE manifold-SAE fit.
// This prices the entire reconstruction at its achieved explained variance, so
// the user-facing report can LEAD with bits/token instead of the
// manifold-insensitive matched-EV number (see
// `experiments/real_manifold_sae/results.md`).
// ===========================================================================

/// Occupancy `p_k = firings(k)/N` of each atom in a support matrix: the per-token
/// weight atom `k`'s code rate and code distortion carry. All zero when `N = 0`.
pub fn atom_occupancy(codes: &SparseAtomCodes) -> Vec<f64> {
    let mut firings = vec![0.0_f64; codes.k_atoms()];
    for code in codes.iter() {
        for atom in code.active_mask.iter_ones() {
            firings[atom] += 1.0;
        }
    }
    let n = codes.n_obs() as f64;
    if n > 0.0 {
        for count in &mut firings {
            *count /= n;
        }
    }
    firings
}

/// One stored decoder block `B_k ∈ R^{M_k×P}`, priced as a uniform
/// scalar-quantizer message in the output distortion budget.
///
/// Coefficient errors `E` perturb the output as `δf_i = a_ik φ_k(t_ik)ᵀE`. With
/// independent zero-mean errors of variance `d_m` on basis row `m` (a dithered
/// uniform quantizer), the expected per-token squared output error is
/// `Σ_m g_m·P·d_m`, where `g_m = (1/N) Σ_i a_ik² φ_km(t_ik)²`. A uniform quantizer
/// with cell width `Δ` on the block's coefficient support `R` spends `log₂(R/Δ)`
/// bits and leaves error variance `Δ²/12`, so one coefficient costs
/// `½log₂(v_m/d_out)` bits for output distortion `d_out = g_m Δ²/12`, with
/// `v_m = g_m R²/12`. Sending no bits (the support midpoint) leaves `v_m`. Each
/// coefficient is therefore a scalar source of variance `v_m` in output units.
#[derive(Clone, Debug)]
pub struct DecoderBlockCode {
    /// Uniform quantizer support `max − min` over the block's coefficients.
    pub coefficient_range: f64,
    /// Output channels `P` each basis row spans.
    pub p_out: usize,
    /// Output sensitivity `g_m = (1/N) Σ_i a_ik² φ_km(t_ik)²` of each basis row.
    pub row_sensitivity: Vec<f64>,
}

/// How the decoder message is coded. The decoder's precision is a property of
/// the decoder and its effect on the output, never of the latent coordinates'
/// variance.
#[derive(Clone, Debug)]
pub enum DictionaryCode {
    /// A declared per-scalar storage precision (for example 16 bits for fp16):
    /// `n_params · bits_per_scalar`. This is a declared approximation. Its
    /// precision is not derived from the decoder's effect on the output, and it
    /// spends none of the distortion budget.
    DeclaredPrecision {
        n_params: usize,
        bits_per_scalar: f64,
    },
    /// Decoder-aware uniform quantization of every stored block (see
    /// [`DecoderBlockCode`]), allocated in the SAME weighted water-filling budget
    /// as the codes, plus `header_bits` of declared side information the receiver
    /// needs to rebuild the decoder (basis plans, quantizer supports, output
    /// mean/scale).
    Quantized {
        blocks: Vec<DecoderBlockCode>,
        header_bits: f64,
    },
}

/// Which [`DictionaryCode`] priced a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DictionaryCodeKind {
    DeclaredPrecision,
    Quantized,
}

impl DictionaryCodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DeclaredPrecision => "declared_precision",
            Self::Quantized => "decoder_quantized",
        }
    }
}

/// Build the decoder-aware [`DictionaryCode::Quantized`] of a persisted atom set.
///
/// Row sensitivities evaluate each atom's analytic basis at its stored
/// coordinates, weighted by the squared assignment masses: the same product
/// [`crate::manifold::reconstruct_persisted_atom_set`] decodes. The header is
/// the declared side information a receiver needs to rebuild the decoder: the
/// geometry plans at their persisted JSON encoding, each block's quantizer
/// support as two `f64`, and `output_side_scalars` persisted output mean/scale
/// values as `f64`.
pub fn persisted_decoder_dictionary_code(
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ArrayView2<'_, f64>],
    coords: &[ArrayView2<'_, f64>],
    assignments: ArrayView2<'_, f64>,
    output_side_scalars: usize,
) -> Result<DictionaryCode, String> {
    let k_atoms = geometry_plans.len();
    if decoder_blocks.len() != k_atoms || coords.len() != k_atoms || assignments.ncols() != k_atoms
    {
        return Err(format!(
            "persisted decoder dictionary code: {k_atoms} geometry plans need as many decoder \
             blocks ({}), coordinate blocks ({}) and assignment columns ({})",
            decoder_blocks.len(),
            coords.len(),
            assignments.ncols()
        ));
    }
    let n_rows = assignments.nrows();
    if n_rows == 0 {
        return Err("persisted decoder dictionary code requires at least one token".to_string());
    }
    let mut blocks = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let decoder = decoder_blocks[atom];
        let (basis_width, p_out) = decoder.dim();
        if !decoder.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "persisted decoder dictionary code: decoder block {atom} must be finite"
            ));
        }
        if coords[atom].nrows() != n_rows {
            return Err(format!(
                "persisted decoder dictionary code: coords[{atom}] has {} rows, expected {n_rows}",
                coords[atom].nrows()
            ));
        }
        let (phi, _) = geometry_plans[atom].build_evaluator()?.evaluate(coords[atom])?;
        if phi.dim() != (n_rows, basis_width) {
            return Err(format!(
                "persisted decoder dictionary code: atom {atom} basis {:?} != ({n_rows}, {basis_width})",
                phi.dim()
            ));
        }
        let mut row_sensitivity = vec![0.0_f64; basis_width];
        for row in 0..n_rows {
            let gate = assignments[[row, atom]];
            if !gate.is_finite() {
                return Err(format!(
                    "persisted decoder dictionary code: assignments[{row}, {atom}] must be finite, got {gate}"
                ));
            }
            let gate_sq = gate * gate;
            if gate_sq == 0.0 {
                continue;
            }
            for (basis_row, sensitivity) in row_sensitivity.iter_mut().enumerate() {
                let value = phi[[row, basis_row]];
                *sensitivity += gate_sq * value * value;
            }
        }
        for sensitivity in &mut row_sensitivity {
            *sensitivity /= n_rows as f64;
        }
        let coefficient_range = if decoder.is_empty() {
            0.0
        } else {
            let (low, high) = decoder
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &value| {
                    (low.min(value), high.max(value))
                });
            high - low
        };
        blocks.push(DecoderBlockCode {
            coefficient_range,
            p_out,
            row_sensitivity,
        });
    }
    let plan_bytes = serde_json::to_vec(geometry_plans)
        .map_err(|error| format!("persisted decoder dictionary code: plan encoding failed: {error}"))?;
    let header_bits = f64::from(u8::BITS) * plan_bytes.len() as f64
        + f64::from(u64::BITS) * (2 * k_atoms + output_side_scalars) as f64;
    Ok(DictionaryCode::Quantized {
        blocks,
        header_bits,
    })
}

/// The fit-level description length of a manifold-SAE reconstruction, in bits,
/// decomposed into three ledgers: CODE (the coordinates transmitted per firing),
/// SELECTION (naming which atoms fired), and DICTIONARY (the amortised decoder).
///
/// # Currency (a Gaussian rate–distortion surrogate)
///
/// A token is coded by (1) naming which atoms fired — priced at the empirical
/// support-distribution universal code `H(S)` (cardinality entropy plus
/// conditional co-firing prices), a decodable code that does NOT overpay a
/// predictable tiling dictionary the way the combinatorial worst case
/// `log₂ C(G, k)` does — and (2) transmitting each firing atom's coordinates.
/// Atom `k` fires on a fraction `p_k` of tokens, so its code distortion and its
/// rate both enter the per-token budget with weight `p_k`. The allocation
/// solves `Σ_k p_k Σ_j min(λ_kj, θ) (+ dictionary distortion) = D` with
/// [`weighted_reverse_water_filling`], and atom `k` charges
/// `p_k Σ_j ½log₂(λ_kj/θ)⁺` bits per token. Each rate stays attached to the atom
/// that incurs it and is never averaged across atoms.
///
/// `bits_per_token = total_bits / n_tokens` is the headline. It is the code
/// length per token of the WHOLE representation (codes + amortised dictionary),
/// so two fits at matched EV but different topologies are comparable in
/// the currency the manifold thesis is stated in.
#[derive(Clone, Debug)]
pub struct ManifoldFitDl {
    /// Explained variance the reconstruction achieves (the demoted EV line).
    pub ev: f64,
    /// Number of coded tokens `N`.
    pub n_tokens: i64,
    /// Mean active atoms per token `k̄` (the firing count charged per token).
    pub k_active: f64,
    /// Mean coded coordinates per active atom `d̄`.
    pub coord_dim: f64,
    /// Dictionary size `G` (atom count) the selection cost names into.
    pub g_dict: i64,
    /// Decoder scalar count `n_params = Σ_k M_k·p` the dictionary codes.
    pub n_params: i64,
    /// Occupancy `p_k` of each atom (its weight in the shared allocation).
    pub atom_occupancy: Vec<f64>,
    /// Code bits per token contributed by each atom, `p_k Σ_j ½log₂(λ_kj/θ)⁺`.
    pub atom_code_bits_per_token: Vec<f64>,
    /// Firing-weighted mean bits per transmitted coordinate,
    /// `code_bits / Σ_k firings(k)·d_k` (zero when nothing is transmitted).
    pub coordinate_rate_bits: f64,
    /// Which dictionary code priced `dict_bits`.
    pub dictionary_code: DictionaryCodeKind,
    /// Mean bits per stored decoder scalar: the declared precision, or the
    /// quantized coefficient bits over `n_params`.
    pub l_param_bits: f64,
    /// Declared decoder side-information bits inside `dict_bits`.
    pub dictionary_header_bits: f64,
    /// Expected per-token output distortion the quantized decoder spends from the
    /// shared budget (zero for a declared precision).
    pub dictionary_distortion: f64,
    /// Selection bits per token: the empirical support-entropy universal code
    /// `H(S)` ([`SparseAtomCodes::support_entropy`]`.tree_bits`).
    pub selection_bits_per_token: f64,
    /// Code bits per token, `Σ_k atom_code_bits_per_token[k]`.
    pub code_bits_per_token: f64,
    /// Amortised dictionary bits per token, `dict_bits / N`.
    pub dict_bits_per_token: f64,
    /// Total code bits over the corpus, `N · code_bits_per_token`.
    pub code_bits: f64,
    /// Total selection bits over the corpus, `N · selection_bits_per_token`.
    pub selection_bits: f64,
    /// Total dictionary bits (not per token).
    pub dict_bits: f64,
    /// Total description length in bits, `code + selection + dict`.
    pub total_bits: f64,
    /// The headline currency: `total_bits / n_tokens`.
    pub bits_per_token: f64,
}

/// Assemble the fit-level [`ManifoldFitDl`] from a fit's own empirical byproducts.
///
/// * `codes` — the empirical binary support matrix `S_n ⊆ {0,…,G−1}` (which
///   atoms fired per token). The SELECTION price is charged as the empirical
///   support-distribution code [`SparseAtomCodes::support_entropy`] (a decodable
///   universal code: variable per-token cardinality plus conditional co-firing
///   prices), NOT the invalid rounded-mean combinatorial `log₂ C(G, round k̄)`
///   (which is not even an upper bound — a uniform support over all `2^G`
///   subsets carries `G` bits, yet `log₂ C(G, G/2) < G`). Occupancies `p_k` are
///   read off the same matrix ([`atom_occupancy`]).
/// * `atom_code_spectra` — one variance spectrum per atom (length `G`), in the
///   distortion metric of `distortion_budget`. Its length is the atom's coded
///   dimension `d_k`.
/// * `distortion_budget` — the expected per-token distortion `D` the codes and a
///   quantized dictionary share. `D = 0` gives the legitimate `+∞` rate.
/// * `ev` — the achieved output explained variance, reported alongside. It may
///   be negative (held-out data) but never exceeds one.
/// * `dictionary` — the decoder message ([`DictionaryCode`]).
///
/// Every quantity is READ OFF an existing fit; nothing is re-fit. Malformed
/// input (no tokens, a spectrum count that disagrees with `G`, nonfinite values,
/// EV above one, materially negative eigenvalues, negative precision) is an `Err`.
pub fn manifold_fit_description_length(
    codes: &SparseAtomCodes,
    atom_code_spectra: &[Vec<f64>],
    distortion_budget: f64,
    ev: f64,
    dictionary: &DictionaryCode,
) -> Result<ManifoldFitDl, String> {
    let n_obs = codes.n_obs();
    let k_atoms = codes.k_atoms();
    if n_obs == 0 {
        return Err("manifold fit description length requires at least one token".to_string());
    }
    if atom_code_spectra.len() != k_atoms {
        return Err(format!(
            "manifold fit description length expected {k_atoms} atom code spectra, got {}",
            atom_code_spectra.len()
        ));
    }
    if !ev.is_finite() || ev > 1.0 {
        return Err(format!(
            "manifold fit description length ev must be finite and at most one, got {ev}"
        ));
    }
    let n = n_obs as f64;
    let n_tokens = i64::try_from(n_obs)
        .map_err(|_| "manifold fit description length token count exceeds i64".to_string())?;
    let g_dict = i64::try_from(k_atoms)
        .map_err(|_| "manifold fit description length atom count exceeds i64".to_string())?;

    // SELECTION: the empirical support-distribution universal code per token
    // (cardinality entropy + conditional co-firing prices), a decodable code
    // that prices a predictable tiling dictionary honestly.
    let support = codes.support_entropy();
    let selection_bits_per_token = support.tree_bits;
    let k_active = support.mean_support;

    let atom_occupancy = atom_occupancy(codes);
    let occupied_scalars: f64 = atom_occupancy
        .iter()
        .zip(atom_code_spectra)
        .map(|(&occupancy, spectrum)| occupancy * spectrum.len() as f64)
        .sum();
    let occupancy_total: f64 = atom_occupancy.iter().sum();
    let coord_dim = if occupancy_total > 0.0 {
        occupied_scalars / occupancy_total
    } else {
        0.0
    };

    // CODE components: atom k enters with weight p_k on its own spectrum.
    let mut components: Vec<(f64, Vec<f64>)> = atom_occupancy
        .iter()
        .zip(atom_code_spectra)
        .map(|(&occupancy, spectrum)| (occupancy, spectrum.clone()))
        .collect();

    // DICTIONARY: a declared precision is priced directly; a quantized decoder
    // joins the same allocation. The decoder is sent once for N tokens, so a
    // block of P-channel coefficients of output variance v_m enters as weight
    // P/N on spectrum N·v_m: it spends P·Σ_m min(v_m, θ/N) distortion per token
    // and P·Σ_m ½log₂(N v_m/θ)⁺ bits over the corpus.
    match dictionary {
        DictionaryCode::DeclaredPrecision {
            bits_per_scalar, ..
        } => {
            if !bits_per_scalar.is_finite() || *bits_per_scalar < 0.0 {
                return Err(format!(
                    "declared dictionary precision must be finite and nonnegative, got {bits_per_scalar}"
                ));
            }
        }
        DictionaryCode::Quantized {
            blocks,
            header_bits,
        } => {
            if !header_bits.is_finite() || *header_bits < 0.0 {
                return Err(format!(
                    "dictionary header bits must be finite and nonnegative, got {header_bits}"
                ));
            }
            for (index, block) in blocks.iter().enumerate() {
                let range = block.coefficient_range;
                if !range.is_finite() || range < 0.0 {
                    return Err(format!(
                        "decoder block {index} coefficient range must be finite and nonnegative, got {range}"
                    ));
                }
                let spectrum = block
                    .row_sensitivity
                    .iter()
                    .enumerate()
                    .map(|(row, &sensitivity)| {
                        if !sensitivity.is_finite() || sensitivity < 0.0 {
                            Err(format!(
                                "decoder block {index} row {row} sensitivity must be finite and nonnegative, got {sensitivity}"
                            ))
                        } else {
                            Ok(n * sensitivity * range * range / 12.0)
                        }
                    })
                    .collect::<Result<Vec<f64>, String>>()?;
                components.push((block.p_out as f64 / n, spectrum));
            }
        }
    }

    let allocation = solve_weighted_allocation(&components, distortion_budget)?;
    let atom_code_bits_per_token = allocation.rates[..k_atoms].to_vec();
    let code_bits_per_token: f64 = atom_code_bits_per_token.iter().sum();

    let (n_params, dictionary_code, l_param_bits, dictionary_header_bits, dictionary_distortion, dict_bits) =
        match dictionary {
            DictionaryCode::DeclaredPrecision {
                n_params,
                bits_per_scalar,
            } => (
                *n_params,
                DictionaryCodeKind::DeclaredPrecision,
                *bits_per_scalar,
                0.0,
                0.0,
                *n_params as f64 * bits_per_scalar,
            ),
            DictionaryCode::Quantized {
                blocks,
                header_bits,
            } => {
                let n_params = blocks.iter().try_fold(0_usize, |total, block| {
                    block
                        .row_sensitivity
                        .len()
                        .checked_mul(block.p_out)
                        .and_then(|count| total.checked_add(count))
                });
                let n_params = n_params
                    .ok_or_else(|| "decoder coefficient count overflowed".to_string())?;
                let coefficient_bits = n * allocation.rates[k_atoms..].iter().sum::<f64>();
                let distortion: f64 = allocation.spectra[k_atoms..]
                    .iter()
                    .map(|(weight, variances)| {
                        weight
                            * variances
                                .iter()
                                .map(|&variance| variance.min(allocation.water_level))
                                .sum::<f64>()
                    })
                    .sum();
                let l_param_bits = if n_params > 0 {
                    coefficient_bits / n_params as f64
                } else {
                    0.0
                };
                (
                    n_params,
                    DictionaryCodeKind::Quantized,
                    l_param_bits,
                    *header_bits,
                    distortion,
                    coefficient_bits + header_bits,
                )
            }
        };
    let n_params = i64::try_from(n_params)
        .map_err(|_| "manifold fit description length parameter count exceeds i64".to_string())?;

    let code_bits = n * code_bits_per_token;
    let selection_bits = n * selection_bits_per_token;
    let total_bits = code_bits + selection_bits + dict_bits;
    let transmitted_scalars = n * occupied_scalars;
    let coordinate_rate_bits = if transmitted_scalars > 0.0 {
        code_bits / transmitted_scalars
    } else {
        0.0
    };

    Ok(ManifoldFitDl {
        ev,
        n_tokens,
        k_active,
        coord_dim,
        g_dict,
        n_params,
        atom_occupancy,
        atom_code_bits_per_token,
        coordinate_rate_bits,
        dictionary_code,
        l_param_bits,
        dictionary_header_bits,
        dictionary_distortion,
        selection_bits_per_token,
        code_bits_per_token,
        dict_bits_per_token: dict_bits / n,
        code_bits,
        selection_bits,
        dict_bits,
        total_bits,
        bits_per_token: total_bits / n,
    })
}

#[cfg(test)]
#[path = "description_length_tests.rs"]
mod description_length_tests;
