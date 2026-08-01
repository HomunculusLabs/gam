//! #2691 — the chart's OWN dispersion, and the refusal that makes a collapsed
//! chart impossible to return silently.
//!
//! `sae_manifold_fit` could converge, certify, report a healthy REML trajectory
//! and hand back a `d_atom = 1` chart coordinate that was a single value to
//! fourteen decimal places. Every consumer downstream (steering, #2234's E1/E2)
//! then measured exact zeros, because a chart with one point has no
//! displacements in it. Nothing in the returned object said so.
//!
//! Two properties of that defect fix the shape of this module:
//!
//! 1. **It must not be denominated in reconstruction quality.** The #2691
//!    ledger measured the fully collapsed arm at `fit_ev = 0.0581`, BELOW a
//!    partially collapsed arm at `0.0883`, while the only arm that recovered
//!    anything had the HIGHEST EV (`0.3179`). A chart compressed into a small
//!    arc still lets the harmonic decoder trace the ring — the decoder simply
//!    rescales — so EV keeps reporting a fine reconstruction while the
//!    coordinate has stopped being a coordinate. Every quantity here is a
//!    property of the coordinate alone; the target is never consulted.
//!
//! 2. **It must be measured in the chart's own manifold.** On a period-`P`
//!    circle `t` and `t + P` are the SAME point, so the raw `f64` standard
//!    deviation of the coordinate is not its dispersion: the #2691 chart-
//!    dimension scan reported `coord_std ≈ 4.98e-1` with two distinct values at
//!    `pca-dim` 3, 8 and 16 and read them as healthy, when `{0.0, 1.0}` is one
//!    point of the circle and every one of those dimensions was collapsed. A
//!    periodic axis is therefore measured by its circular variance
//!    `1 − |mean exp(i κ t)|`, a Euclidean axis by its standard deviation.

use ndarray::Array1;

use super::SaeManifoldTerm;

/// The dispersion of one atom's chart axis, measured in that axis's own
/// manifold, together with the floor below which the axis carries no
/// coordinate at all.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartAxisDispersion {
    pub atom: usize,
    pub atom_name: String,
    pub axis: usize,
    /// The axis's period when it is periodic; `None` for a Euclidean axis.
    pub period: Option<f64>,
    /// Periodic axis: the circular variance `1 − |mean exp(i κ t)| ∈ [0, 1]`.
    /// Euclidean axis: the coordinate's standard deviation.
    pub dispersion: f64,
    /// The dispersion `n` rows would show if they were the SAME chart point up
    /// to floating-point representation at this axis's own magnitude. Below it,
    /// the axis is a constant and the chart has one point along it.
    pub floor: f64,
    /// Number of chart points the axis resolves, after wrapping a periodic axis
    /// into one period. `1` is a collapsed axis by construction.
    pub resolved_points: usize,
}

impl ChartAxisDispersion {
    /// Whether this axis has stopped being a coordinate: its rows are one
    /// point of its manifold, up to floating-point representation.
    pub fn degenerate(&self) -> bool {
        self.resolved_points <= 1 || !(self.dispersion > self.floor)
    }
}

/// Every chart axis of a fitted dictionary, in `(atom, axis)` order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartDegeneracyReport {
    pub axes: Vec<ChartAxisDispersion>,
    /// Number of atoms in the dictionary the report was taken from.
    pub atom_count: usize,
}

impl ChartDegeneracyReport {
    pub fn degenerate_axes(&self) -> impl Iterator<Item = &ChartAxisDispersion> {
        self.axes.iter().filter(|axis| axis.degenerate())
    }

    /// Whether ANY chart axis has collapsed to a point. Telemetry: a `d_atom ≥
    /// 2` atom that loses one axis is a genuine (if unannounced) dimension
    /// selection, not necessarily a defect.
    pub fn any_axis_degenerate(&self) -> bool {
        self.axes.iter().any(ChartAxisDispersion::degenerate)
    }

    /// Whether some atom has lost its ENTIRE chart — every one of its axes is a
    /// single point, so the "manifold atom" decodes to one point of the ambient
    /// space and carries no displacements. This is the #2691 condition.
    pub fn atoms_without_a_chart(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for atom in 0..self.atom_count {
            let mut axes = self.axes.iter().filter(|entry| entry.atom == atom).peekable();
            if axes.peek().is_none() {
                continue;
            }
            if axes.all(ChartAxisDispersion::degenerate) {
                out.push(atom);
            }
        }
        out
    }

    /// One line per collapsed atom, for a refusal message.
    pub fn collapsed_atom_evidence(&self) -> String {
        self.atoms_without_a_chart()
            .into_iter()
            .map(|atom| {
                let detail = self
                    .axes
                    .iter()
                    .filter(|entry| entry.atom == atom)
                    .map(|entry| {
                        let kind = match entry.period {
                            Some(period) => format!("periodic(P={period:.6e}) circular variance"),
                            None => "euclidean standard deviation".to_string(),
                        };
                        format!(
                            "axis {} {kind} {:.6e} <= floor {:.6e} ({} resolved chart point(s) \
                             over the rows)",
                            entry.axis, entry.dispersion, entry.floor, entry.resolved_points
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                let name = self
                    .axes
                    .iter()
                    .find(|entry| entry.atom == atom)
                    .map(|entry| entry.atom_name.clone())
                    .unwrap_or_default();
                format!("atom {atom} ('{name}'): {detail}")
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

impl SaeManifoldTerm {
    /// Measure every chart axis's dispersion in its own manifold. Pure read of
    /// the fitted coordinates — no target, no reconstruction, no EV.
    pub fn chart_degeneracy_report(&self) -> ChartDegeneracyReport {
        let mut axes = Vec::new();
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let periods = coord.effective_axis_periods();
            let matrix = coord.as_matrix();
            let n = matrix.nrows();
            if n == 0 {
                continue;
            }
            let atom_name = self
                .atoms
                .get(atom_idx)
                .map(|atom| atom.name.clone())
                .unwrap_or_default();
            for axis in 0..coord.latent_dim() {
                let column: Array1<f64> = matrix.column(axis).to_owned();
                let magnitude = column.iter().fold(0.0_f64, |m, &t| m.max(t.abs()));
                // The resolution at which two coordinate values on this axis are
                // the SAME f64 point, at the axis's own magnitude. This is a
                // representation limit, not a tuned tolerance.
                let resolution = f64::EPSILON * magnitude;
                let (dispersion, floor, resolved_points) = match periods[axis] {
                    Some(period) if period > 0.0 => {
                        let kappa = std::f64::consts::TAU / period;
                        let (mut re, mut im) = (0.0_f64, 0.0_f64);
                        for &t in column.iter() {
                            let phase = kappa * t;
                            re += phase.cos();
                            im += phase.sin();
                        }
                        re /= n as f64;
                        im /= n as f64;
                        let resultant = (re * re + im * im).sqrt().min(1.0);
                        // Rows separated by `resolution` in `t` are separated by
                        // `kappa * resolution` in phase; the circular variance of
                        // a spread that small is `½ (κ·resolution)²` to leading
                        // order. That is the dispersion an axis shows when every
                        // row is the same chart point.
                        let phase_resolution = kappa * resolution;
                        let floor = 0.5 * phase_resolution * phase_resolution;
                        let mut wrapped: Vec<i64> = column
                            .iter()
                            .map(|&t| {
                                let unit = t.rem_euclid(period) / period;
                                (unit / f64::EPSILON).round() as i64
                            })
                            .collect();
                        wrapped.sort_unstable();
                        wrapped.dedup();
                        (1.0 - resultant, floor, wrapped.len())
                    }
                    _ => {
                        let mean = column.iter().sum::<f64>() / n as f64;
                        let variance = column
                            .iter()
                            .map(|&t| (t - mean) * (t - mean))
                            .sum::<f64>()
                            / n as f64;
                        let mut distinct: Vec<u64> = column.iter().map(|t| t.to_bits()).collect();
                        distinct.sort_unstable();
                        distinct.dedup();
                        (variance.sqrt(), resolution, distinct.len())
                    }
                };
                axes.push(ChartAxisDispersion {
                    atom: atom_idx,
                    atom_name: atom_name.clone(),
                    axis,
                    period: periods[axis],
                    dispersion,
                    floor,
                    resolved_points,
                });
            }
        }
        ChartDegeneracyReport {
            axes,
            atom_count: self.k_atoms(),
        }
    }
}

/// Certificate wrapper for [`ChartDegeneracyReport`], so a fit's ledger carries
/// "the chart is still a coordinate" as an explicit claim rather than leaving
/// it to a caller to notice its absence.
#[derive(Clone, Debug)]
pub struct ChartNondegeneracyCertificate {
    axes: usize,
    degenerate_axes: usize,
    collapsed_atoms: usize,
    atom_count: usize,
    /// Smallest per-axis ratio `dispersion / floor` over all axes. `< 1` on any
    /// axis means that axis is a constant up to floating-point representation.
    min_dispersion_over_floor: f64,
    min_resolved_points: usize,
}

impl ChartNondegeneracyCertificate {
    pub fn new(report: &ChartDegeneracyReport) -> Self {
        let min_dispersion_over_floor = report
            .axes
            .iter()
            .map(|axis| {
                if axis.floor > 0.0 {
                    axis.dispersion / axis.floor
                } else if axis.dispersion > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                }
            })
            .fold(f64::INFINITY, f64::min);
        Self {
            axes: report.axes.len(),
            degenerate_axes: report.degenerate_axes().count(),
            collapsed_atoms: report.atoms_without_a_chart().len(),
            atom_count: report.atom_count,
            min_dispersion_over_floor,
            min_resolved_points: report
                .axes
                .iter()
                .map(|axis| axis.resolved_points)
                .min()
                .unwrap_or(0),
        }
    }
}

impl gam_problem::topology_certificates::Certificate for ChartNondegeneracyCertificate {
    fn claim(&self) -> gam_problem::topology_certificates::Claim {
        gam_problem::topology_certificates::Claim::new(
            "chart-nondegeneracy",
            "every fitted chart axis still separates rows in its OWN manifold \
             (circular variance on a periodic axis, standard deviation on a \
             Euclidean one) by more than floating-point representation at that \
             axis's magnitude; a chart that has collapsed to one point is \
             reported here rather than certified as a fit. This claim is \
             deliberately independent of reconstruction quality: #2691 measured \
             a fully collapsed chart with a HIGHER explained variance than a \
             partially collapsed one",
        )
    }

    fn evidence(&self) -> gam_problem::topology_certificates::Evidence {
        let mut evidence = gam_problem::topology_certificates::Evidence::new();
        evidence.insert("chart_axes", self.axes.into());
        evidence.insert("degenerate_axes", self.degenerate_axes.into());
        evidence.insert("atoms_without_a_chart", self.collapsed_atoms.into());
        evidence.insert("atoms", self.atom_count.into());
        evidence.insert("min_resolved_chart_points", self.min_resolved_points.into());
        if self.min_dispersion_over_floor.is_finite() {
            evidence.insert(
                "min_dispersion_over_floor",
                self.min_dispersion_over_floor.into(),
            );
        } else {
            evidence.insert("min_dispersion_over_floor", "n/a".into());
        }
        evidence
    }

    fn verdict(&self) -> gam_problem::topology_certificates::Verdict {
        use gam_problem::topology_certificates::Verdict;
        if self.axes == 0 {
            Verdict::Unavailable
        } else if self.degenerate_axes == 0 {
            Verdict::Certified
        } else {
            Verdict::Insufficient
        }
    }
}
